/*
 * Adapted from SGLang's sgl-kernel implementation, which was adapted from
 * https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/kernels/dsv3MinLatencyKernels/dsv3RouterGemm.cu
 * https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/thop/dsv3RouterGemmOp.cpp
 *
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// =============================================================================
// 中文注释：DeepSeek V3 Router GEMM kernel（float32 输出版本）
//
// 本文件实现了 DSV3 Router GEMM 的 float32 输出版本。
// 计算 output = mat_a @ mat_b.T，其中 output 为 float32。
//
// 线程组织：
// - 每个 block 处理一个专家列（blockIdx.x = expert index）
// - 每个 warp 处理 hidden_dim 的一部分
// - kBlockSize=128 线程，kNumWarps=4 个 warp
//
// 优化要点：
// - 使用 PTX fma.rn.f32x2 指令进行 float2 向量化 FMA
// - 使用 bf16_uint4_to_float8 批量 bf16->float 转换
// - 使用 griddepcontrol 实现 PDL 优化
// =============================================================================

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "dsv3_router_gemm_utils.h"

// 中文注释：fma —— 使用 PTX 汇编实现的 float2 融合乘加（FMA）操作。
// 利用 fma.rn.f32x2 指令一次性处理 2 个 float 的乘加，比两次标量 FMA 更高效。
// 这是 DSV3 Router GEMM kernel 的性能关键路径。
__device__ __forceinline__ void fma(float2& d, float2 const& a, float2 const& b,
                                    float2 const& c) {
  asm volatile("fma.rn.f32x2 %0, %1, %2, %3;\n"
               : "=l"(reinterpret_cast<uint64_t&>(d))
               : "l"(reinterpret_cast<uint64_t const&>(a)),
                 "l"(reinterpret_cast<uint64_t const&>(b)),
                 "l"(reinterpret_cast<uint64_t const&>(c)));
}

// 中文注释：bf16_uint4_to_float8 —— 将 uint4（128-bit）中的 bf16 值转换为 float 数组。
// uint4 包含 8 个 bf16 值，逐个转换为 float 后存入 dst。
// 用于从全局内存加载 bf16 权重后转换为 float 进行累加，避免精度损失。
template <int VPT>
__device__ __forceinline__ void bf16_uint4_to_float8(uint4 const& vec,
                                                     float* dst) {
  __nv_bfloat16* bf16_ptr =
      reinterpret_cast<__nv_bfloat16*>(const_cast<uint4*>(&vec));

#pragma unroll
  for (int i = 0; i < VPT; i++) {
    dst[i] = __bfloat162float(bf16_ptr[i]);
  }
}

template <typename T, int kBlockSize, int VPT, int kNumTokens, int kNumExperts,
          int kHiddenDim>
__global__ __launch_bounds__(128, 1) void router_gemm_kernel_float_output(
    float* out, T const* mat_a, T const* mat_b) {
  // Each block handles one expert column
  int const n_idx = blockIdx.x;
  int const tid = threadIdx.x;
  constexpr int kWarpSize = 32;
  constexpr int kNumWarps = kBlockSize / kWarpSize;
  // Constants for this kernel
  constexpr int k_elems_per_k_iteration = VPT * kBlockSize;
  constexpr int k_iterations =
      kHiddenDim / k_elems_per_k_iteration;  // Total K iterations

  // Initialize accumulators for all M rows
  float acc[kNumTokens] = {};

  // Shared memory for warp-level reduction
  __shared__ float sm_reduction[kNumTokens][kNumWarps];  // kNumWarps

  // B matrix is in column-major order, so we can directly load a column for the
  // n_idx expert
  T const* b_col = mat_b + n_idx * kHiddenDim;

  // Pre-compute k_base values for each iteration to help compiler optimize
  int k_bases[k_iterations];
#pragma unroll
  for (int ki = 0; ki < k_iterations; ki++) {
    k_bases[ki] = ki * k_elems_per_k_iteration + tid * VPT;
  }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
#endif

  // Process the GEMM in chunks
  for (int ki = 0; ki < k_iterations; ki++) {
    int const k_base = k_bases[ki];

    // Load B matrix values using vector load (8 bf16 values)
    uint4 b_vec = *reinterpret_cast<uint4 const*>(b_col + k_base);

    // Convert B values to float
    float b_float[VPT];
    bf16_uint4_to_float8<VPT>(b_vec, b_float);

// Process each token
#pragma unroll
    for (int m_idx = 0; m_idx < kNumTokens; m_idx++) {
      // Load both rows of A matrix using vector loads
      uint4 a_vec = *reinterpret_cast<uint4 const*>(
          mat_a + (m_idx * kHiddenDim) + k_base);

      // Convert A values to float
      float a_float[VPT];
      bf16_uint4_to_float8<VPT>(a_vec, a_float);

// Process elements in this chunk
#pragma unroll
      for (int k = 0; k < VPT; k++) {
        float a = a_float[k];
        float b = b_float[k];
        acc[m_idx] += a * b;
      }
    }
  }

  // Perform warp-level reduction
  int const warpSize = 32;
  int const warpId = tid / warpSize;
  int const laneId = tid % warpSize;

  // Register for warp-level reduction results
  float warp_result[kNumTokens];

#pragma unroll
  for (int m_idx = 0; m_idx < kNumTokens; m_idx++) {
    warp_result[m_idx] = acc[m_idx];
  }

// Perform warp-level reduction using optimized butterfly pattern
#pragma unroll
  for (int m = 0; m < kNumTokens; m++) {
    float sum = warp_result[m];

    // Butterfly reduction pattern
    sum += __shfl_xor_sync(0xffffffff, sum, 16);
    sum += __shfl_xor_sync(0xffffffff, sum, 8);
    sum += __shfl_xor_sync(0xffffffff, sum, 4);
    sum += __shfl_xor_sync(0xffffffff, sum, 2);
    sum += __shfl_xor_sync(0xffffffff, sum, 1);

    // Only the first thread in each warp stores to shared memory
    if (laneId == 0) {
      sm_reduction[m][warpId] = sum;
    }
  }

  __syncthreads();

  // Final reduction across warps (only first thread)
  if (tid == 0) {
#pragma unroll
    for (int m = 0; m < kNumTokens; m++) {
      float final_sum = 0.0f;

// Sum across the kNumWarps
#pragma unroll
      for (int w = 0; w < kNumWarps; w++) {
        final_sum += sm_reduction[m][w];
      }

      // Write final result
      out[m * kNumExperts + n_idx] = final_sum;
    }
  }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

template <typename T, int kNumTokens, int kNumExperts, int kHiddenDim>
void invokeRouterGemmFloatOutput(float* output, T const* mat_a, T const* mat_b,
                                 cudaStream_t stream) {
  constexpr int VPT = 16 / sizeof(T);
  constexpr int kBlockSize = 128;
  cudaLaunchConfig_t config;
  config.gridDim = kNumExperts;
  config.blockDim = kBlockSize;
  config.dynamicSmemBytes = 0;
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = 1;
  config.numAttrs = 1;
  config.attrs = attrs;
  cudaLaunchKernelEx(
      &config,
      router_gemm_kernel_float_output<T, kBlockSize, VPT, kNumTokens,
                                      kNumExperts, kHiddenDim>,
      output, mat_a, mat_b);
}

// Template instantiations for DEFAULT_NUM_EXPERTS experts
template void invokeRouterGemmFloatOutput<__nv_bfloat16, 1, 256, 7168>(
    float*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);

template void invokeRouterGemmFloatOutput<__nv_bfloat16, 2, 256, 7168>(
    float*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);

template void invokeRouterGemmFloatOutput<__nv_bfloat16, 3, 256, 7168>(
    float*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);

template void invokeRouterGemmFloatOutput<__nv_bfloat16, 4, 256, 7168>(
    float*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);

template void invokeRouterGemmFloatOutput<__nv_bfloat16, 5, 256, 7168>(
    float*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);

template void invokeRouterGemmFloatOutput<__nv_bfloat16, 6, 256, 7168>(
    float*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);

template void invokeRouterGemmFloatOutput<__nv_bfloat16, 7, 256, 7168>(
    float*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);

template void invokeRouterGemmFloatOutput<__nv_bfloat16, 8, 256, 7168>(
    float*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);

template void invokeRouterGemmFloatOutput<__nv_bfloat16, 9, 256, 7168>(
    float*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);

template void invokeRouterGemmFloatOutput<__nv_bfloat16, 10, 256, 7168>(
    float*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);

template void invokeRouterGemmFloatOutput<__nv_bfloat16, 11, 256, 7168>(
    float*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);

template void invokeRouterGemmFloatOutput<__nv_bfloat16, 12, 256, 7168>(
    float*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);

template void invokeRouterGemmFloatOutput<__nv_bfloat16, 13, 256, 7168>(
    float*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);

template void invokeRouterGemmFloatOutput<__nv_bfloat16, 14, 256, 7168>(
    float*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);

template void invokeRouterGemmFloatOutput<__nv_bfloat16, 15, 256, 7168>(
    float*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);

template void invokeRouterGemmFloatOutput<__nv_bfloat16, 16, 256, 7168>(
    float*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);

// Template instantiations for KIMI_K2_NUM_EXPERTS experts
template void invokeRouterGemmFloatOutput<__nv_bfloat16, 1, 384, 7168>(
    float*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);

template void invokeRouterGemmFloatOutput<__nv_bfloat16, 2, 384, 7168>(
    float*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);

template void invokeRouterGemmFloatOutput<__nv_bfloat16, 3, 384, 7168>(
    float*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);

template void invokeRouterGemmFloatOutput<__nv_bfloat16, 4, 384, 7168>(
    float*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);

template void invokeRouterGemmFloatOutput<__nv_bfloat16, 5, 384, 7168>(
    float*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);

template void invokeRouterGemmFloatOutput<__nv_bfloat16, 6, 384, 7168>(
    float*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);

template void invokeRouterGemmFloatOutput<__nv_bfloat16, 7, 384, 7168>(
    float*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);

template void invokeRouterGemmFloatOutput<__nv_bfloat16, 8, 384, 7168>(
    float*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);

template void invokeRouterGemmFloatOutput<__nv_bfloat16, 9, 384, 7168>(
    float*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);

template void invokeRouterGemmFloatOutput<__nv_bfloat16, 10, 384, 7168>(
    float*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);

template void invokeRouterGemmFloatOutput<__nv_bfloat16, 11, 384, 7168>(
    float*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);

template void invokeRouterGemmFloatOutput<__nv_bfloat16, 12, 384, 7168>(
    float*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);

template void invokeRouterGemmFloatOutput<__nv_bfloat16, 13, 384, 7168>(
    float*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);

template void invokeRouterGemmFloatOutput<__nv_bfloat16, 14, 384, 7168>(
    float*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);

template void invokeRouterGemmFloatOutput<__nv_bfloat16, 15, 384, 7168>(
    float*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);

template void invokeRouterGemmFloatOutput<__nv_bfloat16, 16, 384, 7168>(
    float*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);
