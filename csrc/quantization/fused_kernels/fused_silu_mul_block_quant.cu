// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

// =============================================================================
// 模块功能概述: 融合 SiLU 门控 + 分组量化的 CUDA Kernel
// =============================================================================
// 本文件实现了 SiLU(gate) * up 的融合计算，并直接输出 per-block 量化结果。
// 与 activation_kernels.cu 中的 kernel 不同，本文件使用分组量化 (group quantization)，
// 每个 group 独立计算缩放因子，适用于更细粒度的量化场景。
//
// 核心设计:
//   1. 每个 thread block 处理一个 (token, group) 对
//   2. grid 维度: (num_tokens, num_groups)，其中 num_groups = hidden_size / group_size
//   3. block 维度: group_size 个线程，每个线程处理 group 内的一个元素
//
// 计算流程 (4 步):
//   Step 1: 每个线程加载 gate 和 up 对应元素，计算 SiLU(gate) * up
//   Step 2: 通过 shared memory 归约求 group 内绝对值最大值
//   Step 3: thread 0 计算量化缩放因子 (scale = max / quant_range)，广播给所有线程
//   Step 4: 每个线程用缩放因子量化结果，写入全局内存
//
// 支持的量化类型: FP8 (E4M3) 和 INT8
// 支持的 group_size: 64 或 128
// 支持的缩放因子布局: 行优先或列优先 (通过 is_scale_transposed 模板参数控制)
// =============================================================================

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "../../dispatch_utils.h"
#include "libtorch_stable/quantization/fused_kernels/quant_conversions.cuh"

namespace vllm {

// 中文注释: 每个 thread block 处理一个 (token, group) 对
// Logic: one thread block per (token, group) pair

template <typename scalar_t, typename scalar_out_t, bool is_scale_transposed,
          int32_t group_size>
__global__ void silu_and_mul_per_block_quant_kernel(
    scalar_out_t* __restrict__ out,  // Output: [num_tokens, hidden_size] in
                                     // FP8/INT8
    float* __restrict__ scales,      // Output: [num_tokens, hidden_size /
                                 // group_size] or [hidden_size / group_size,
                                 // num_tokens]
    scalar_t const* __restrict__ input,  // Input: [num_tokens, hidden_size * 2]
    float const* scale_ub,               // Optional scale upper bound
    int32_t const hidden_size  // Output hidden size (input is 2x this)
) {
  static_assert((group_size & (group_size - 1)) == 0,
                "group_size must be a power of 2 for correct reduction");

  // Grid: (num_tokens, num_groups)
  int const token_idx = blockIdx.x;
  int const group_idx = blockIdx.y;
  int const tid = threadIdx.x;  // tid in [0, group_size)
  int const num_tokens = gridDim.x;

  // Input layout: [gate || up] concatenated along last dimension
  int const input_stride = hidden_size * 2;
  int const group_start = group_idx * group_size;

  // Pointers to this token's data
  scalar_t const* token_input_gate =
      input + token_idx * input_stride + group_start;
  scalar_t const* token_input_up = token_input_gate + hidden_size;
  scalar_out_t* token_output = out + token_idx * hidden_size + group_start;

  // Scale pointer for this group
  int const num_groups = gridDim.y;
  float* group_scale_ptr = is_scale_transposed
                               ? scales + group_idx * num_tokens + token_idx
                               : scales + token_idx * num_groups + group_idx;

  // Shared memory for reduction (compile-time sized)
  __shared__ float shared_max[group_size];

  // Step 1: Each thread loads one element, computes SiLU, stores in register
  float gate = static_cast<float>(token_input_gate[tid]);
  float up = static_cast<float>(token_input_up[tid]);

  // Compute SiLU(gate) * up
  float sigmoid_gate = 1.0f / (1.0f + expf(-gate));
  float silu_gate = gate * sigmoid_gate;
  float result = silu_gate * up;  // Keep in register

  // Step 2: Reduce to find group max
  shared_max[tid] = fabsf(result);
  __syncthreads();

// Power-of-2 reduction (group_size guaranteed to be power of 2)
#pragma unroll
  for (int stride = group_size / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + stride]);
    }
    __syncthreads();
  }

  // Step 3: Compute scale (thread 0), broadcast via shared memory
  if (tid == 0) {
    float group_max = shared_max[0];

    float const quant_range = quant_type_max_v<scalar_out_t>;
    float group_scale = group_max / quant_range;

    // Apply scale upper bound if provided
    if (scale_ub != nullptr) {
      group_scale = fminf(group_scale, *scale_ub);
    }

    // Use minimum safe scaling factor
    group_scale = fmaxf(group_scale, min_scaling_factor<scalar_out_t>::val());

    // Store scale to global memory
    *group_scale_ptr = group_scale;

    // Reuse shared_max[0] to broadcast scale
    shared_max[0] = group_scale;
  }
  __syncthreads();

  float group_scale = shared_max[0];

  // Step 4: Quantize and write output
  token_output[tid] =
      vllm::ScaledQuant<scalar_out_t, false>::quant_fn(result, group_scale);
}

}  // namespace vllm

void silu_and_mul_per_block_quant(torch::Tensor& out,
                                  torch::Tensor const& input,
                                  torch::Tensor& scales, int64_t group_size,
                                  std::optional<torch::Tensor> scale_ub,
                                  bool is_scale_transposed) {
  static c10::ScalarType kFp8Type = is_fp8_ocp()
                                        ? c10::ScalarType::Float8_e4m3fn
                                        : c10::ScalarType::Float8_e4m3fnuz;

  TORCH_CHECK(out.dtype() == kFp8Type || out.dtype() == torch::kInt8);
  TORCH_CHECK(out.is_contiguous() && input.is_contiguous());
  TORCH_CHECK(
      input.dtype() == torch::kFloat16 || input.dtype() == torch::kBFloat16,
      "Input must be FP16 or BF16");
  TORCH_CHECK(scales.dtype() == torch::kFloat32, "Scales must be FP32");
  TORCH_CHECK(group_size == 128 || group_size == 64,
              "Unsupported group size: ", group_size);

  if (scale_ub.has_value()) {
    TORCH_CHECK(out.dtype() == kFp8Type);
  }

  int32_t hidden_size = out.size(-1);
  auto num_tokens = input.size(0);
  int32_t num_groups = hidden_size / group_size;

  TORCH_CHECK(input.size(-1) == hidden_size * 2,
              "input last dim must be 2x output hidden_size");
  TORCH_CHECK(hidden_size % group_size == 0,
              "hidden_size must be divisible by group_size");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  dim3 grid(num_tokens, num_groups);
  dim3 block(group_size);

  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "silu_and_mul_per_block_quant", [&] {
        using scalar_in_t = scalar_t;

        VLLM_DISPATCH_QUANT_TYPES(
            out.scalar_type(), "silu_and_mul_per_block_quant", [&] {
              using scalar_out_t = scalar_t;

              VLLM_DISPATCH_GROUP_SIZE(group_size, gs, [&] {
                VLLM_DISPATCH_BOOL(is_scale_transposed, transpose_scale, [&] {
                  vllm::silu_and_mul_per_block_quant_kernel<
                      scalar_in_t, scalar_out_t, transpose_scale, gs>
                      <<<grid, block, 0, stream>>>(
                          out.data_ptr<scalar_out_t>(),
                          scales.data_ptr<float>(),
                          input.data_ptr<scalar_in_t>(),
                          scale_ub.has_value() ? scale_ub->data_ptr<float>()
                                               : nullptr,
                          hidden_size);
                });
              });
            });
      });
}