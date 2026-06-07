/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

/*
 * =============================================================================
 * 文件功能概述（中文）
 * =============================================================================
 * 本文件定义了 MiniMax AllReduce + RMS Norm 融合 kernel 的参数结构体和接口。
 *
 * 【模块功能】
 *   在 Tensor Parallel 推理场景中，注意力层的 Q/K/V 计算后需要执行：
 *     1. AllReduce —— 合并各 GPU 的部分结果
 *     2. RMS Norm  —— 对归一化后的 Q 和 K 分别做 RMS Norm
 *
 *   本模块将这两步融合为单个 kernel，减少了一次显存读写和 kernel launch 开销。
 *
 * 【参数结构体 MiniMaxReduceRMSParams】
 *   包含融合 kernel 所需的所有参数，由 host 端填充后传入 kernel。
 *   支持同时处理 Q 和 K 两个张量的 AllReduce + RMS Norm。
 *
 * 【使用场景】
 *   vLLM 在 Tensor Parallel 模式下的注意力前处理中调用此融合 kernel，
 *   将 all-reduce 和 Q/K 的 RMS Norm 合并执行，降低推理延迟。
 * =============================================================================
 */

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <torch/types.h>

namespace vllm {
namespace tensorrt_llm {

// 【ElemsPerAccess】向量化访问的元素数量和向量类型映射
// 不同数据类型在单次内存访问中读取的元素数不同：
//   - half/bfloat16：8 个元素（8 * 2 字节 = 16 字节 = 128 位）
//   - float：4 个元素（4 * 4 字节 = 16 字节 = 128 位）
// 目标是每次访问加载 128 位数据，最大化显存带宽利用率。
template <typename DType>
struct ElemsPerAccess;

template <>
struct ElemsPerAccess<half> {
  static constexpr int value = 8;
  using vec_type = float4;
};

template <>
struct ElemsPerAccess<nv_bfloat16> {
  static constexpr int value = 8;
  using vec_type = float4;
};

template <>
struct ElemsPerAccess<float> {
  static constexpr int value = 4;
  using vec_type = float4;
};

template <typename DType>
static constexpr int kElemsPerAccess = ElemsPerAccess<DType>::value;

// 【MiniMaxReduceRMSParams】AllReduce + RMS Norm 融合 kernel 的参数结构体
// 由 host 端填充，传入 CUDA kernel 使用。
//
// 参数说明：
//   nranks        - 参与 AllReduce 的 GPU 总数（Tensor Parallel world size）
//   rank          - 当前 GPU 的 rank 编号
//   dtype         - 输入/输出张量的数据类型（half / bfloat16 / float）
//   size_q        - Q 张量的 token 数量（行数）
//   hidden_dim    - Q 张量的隐藏维度（列数）
//   size_k        - K 张量的 token 数量（行数）
//   hidden_dim_k  - K 张量的隐藏维度（列数）
//   stride_q      - Q 输入的行步长（元素数）。当 stride_q > hidden_dim 时，
//                   Q 是融合 QKV 张量的一部分（QKV 联合存储布局）
//   stride_k      - K 输入的行步长（元素数）。同上，支持 QKV 联合存储
//   stride_q_out  - Q 输出的行步长（元素数），0 表示连续存储
//   stride_k_out  - K 输出的行步长（元素数），0 表示连续存储
//   workspace     - 各 rank 的 workspace 指针数组（用于 AllReduce 中间结果交换）
//   allreduce_in  - AllReduce 的输入张量指针（Q 部分）
//   rms_norm_out  - RMS Norm 的输出张量指针（Q 部分）
//   rms_gamma     - RMS Norm 的缩放参数（Q 部分，可学习权重 gamma）
//   allreduce_in_k  - AllReduce 的输入张量指针（K 部分）
//   rms_norm_out_k  - RMS Norm 的输出张量指针（K 部分）
//   rms_gamma_k     - RMS Norm 的缩放参数（K 部分）
//   rms_eps       - RMS Norm 的 epsilon 值，防止除零
//   stream        - CUDA 流，kernel 在此流上异步执行
struct MiniMaxReduceRMSParams {
  int nranks{};
  int rank{};
  at::ScalarType dtype{at::ScalarType::Undefined};
  int size_q{};
  int hidden_dim{};
  int size_k{};
  int hidden_dim_k{};
  int stride_q{};  // row stride for q input (elements); when > hidden_dim,
                   // q is part of a wider qkv tensor
  int stride_k{};  // row stride for k input (elements); when > hidden_dim_k,
                   // k is part of a wider qkv tensor
  int stride_q_out{};  // row stride for q output (elements); 0 = contiguous
  int stride_k_out{};  // row stride for k output (elements); 0 = contiguous
  void** workspace{};
  void* allreduce_in{};
  void* rms_norm_out{};
  void* rms_gamma{};
  void* allreduce_in_k{};
  void* rms_norm_out_k{};
  void* rms_gamma_k{};
  float rms_eps{};
  cudaStream_t stream{};
};

void minimax_reduce_rms_op(MiniMaxReduceRMSParams const& params);

}  // namespace tensorrt_llm
}  // namespace vllm
