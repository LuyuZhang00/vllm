// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Adapted from SGLang:
// https://github.com/sgl-project/sglang/blob/ded068a76e00878881d52d5bfb791e0f60d7311b/sgl-kernel/csrc/expert_specialization/es_sm100_mxfp8_blockscaled.cu

// =============================================================================
// 中文注释：MXFP8 分组矩阵乘法（MoE 专家计算）入口文件
//
// 本文件实现了基于 CUTLASS 的 MXFP8（Microscaling FP8）分组矩阵乘法，
// 用于 MoE 模型中多个专家的批量矩阵乘法计算。
//
// MXFP8 量化方案：
// - 激活和权重使用 FP8 格式（E4M3 或 E5M2）
// - 每个 128 元素块共享一个缩放因子（blockscale）
// - 通过 blockscale 实现更细粒度的量化，比 per-tensor 量化精度更高
//
// 算法：使用 CUTLASS 的 grouped GEMM 实现，一次调用处理所有专家的矩阵乘法。
// 输入：
//   a: [num_tokens, k] — 激活（FP8）
//   b: [num_experts, k, n] — 专家权重（FP8）
//   sfa/sfb: 激活和权重的 blockscale 因子
//   problem_sizes: [num_experts, 3] — 每个专家的 (m, n, k) 尺寸
//   expert_offsets: [num_experts] — 每个专家在 token 维度上的偏移
// 输出：
//   d: [num_tokens, n] — 矩阵乘法结果（bf16/fp16）
//
// 要求：SM >= 100（Blackwell 架构），k 和 n 必须对齐到 128。
// =============================================================================

#include <torch/all.h>

#include "cutlass_mxfp8_grouped_mm_launcher.cuh"

void cutlass_mxfp8_grouped_mm(const torch::Tensor& a, const torch::Tensor& b,
                              const torch::Tensor& sfa,
                              const torch::Tensor& sfb, torch::Tensor& d,
                              const torch::Tensor& problem_sizes,
                              const torch::Tensor& expert_offsets,
                              const torch::Tensor& blockscale_offsets) {
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
  TORCH_CHECK(problem_sizes.dim() == 2, "problem_sizes must be 2D tensor");
  TORCH_CHECK(problem_sizes.size(1) == 3,
              "problem_sizes must have shape (num_experts, 3)");
  TORCH_CHECK(problem_sizes.size(0) == expert_offsets.size(0),
              "Number of experts in problem_sizes must match expert_offsets");
  TORCH_CHECK(problem_sizes.dtype() == torch::kInt32,
              "problem_sizes must be int32");
  TORCH_CHECK(expert_offsets.dtype() == torch::kInt32,
              "expert_offsets must be int32");
  TORCH_CHECK(blockscale_offsets.dtype() == torch::kInt32,
              "blockscale_offsets must be int32");
  TORCH_CHECK(a.dim() == 2, "a must be a 2D tensor of shape (num_tokens, k)");
  TORCH_CHECK(b.dim() == 3,
              "b must be a 3D tensor of shape (num_experts, k, n)");
  TORCH_CHECK(a.size(1) == b.size(1) && a.size(1) % 128 == 0,
              "k should align 128");
  TORCH_CHECK(b.size(2) % 128 == 0, "n should align 128");
  TORCH_CHECK(a.strides()[1] == 1, "a must be row major");
  TORCH_CHECK(b.strides()[1] == 1, "b must be column major");

  auto stream = at::cuda::getCurrentCUDAStream();
  if (d.dtype() == torch::kBFloat16) {
    expert_specialization::cutlass_mxfp8_grouped_mm_dispatch_out_dtype<
        cutlass::bfloat16_t>(a, b, sfa, sfb, d, problem_sizes, expert_offsets,
                             blockscale_offsets, stream);
  } else if (d.dtype() == torch::kFloat16) {
    expert_specialization::cutlass_mxfp8_grouped_mm_dispatch_out_dtype<
        cutlass::half_t>(a, b, sfa, sfb, d, problem_sizes, expert_offsets,
                         blockscale_offsets, stream);
  } else {
    TORCH_CHECK(false, "dtype must be kFloat16 or kBFloat16");
  }
#else
  TORCH_CHECK(false,
              "No implemented cutlass_mxfp8_grouped_mm for "
              "current device");
#endif
}

#include "core/registration.h"

TORCH_LIBRARY_IMPL_EXPAND(TORCH_EXTENSION_NAME, CUDA, m) {
  m.impl("cutlass_mxfp8_grouped_mm", cutlass_mxfp8_grouped_mm);
}