// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Adapted from SGLang:
// https://github.com/sgl-project/sglang/blob/ded068a76e00878881d52d5bfb791e0f60d7311b/sgl-kernel/csrc/expert_specialization/es_sm100_mxfp8_blockscaled_group_quant.cu

// =============================================================================
// 中文注释：MXFP8 专家量化 kernel 入口文件
//
// 本文件实现了 MoE 模型中将激活量化为 MXFP8 格式的 kernel。
// 在专家计算之前，需要将 bf16/fp16 的激活量化为 FP8，并计算对应的 blockscale 因子。
//
// 输入：
//   input: [num_tokens, k] — 激活（bf16/fp16）
//   problem_sizes: [num_experts, 3] — 每个专家的 (m, n, k) 尺寸
//   expert_offsets: [num_experts] — 每个专家在 token 维度上的偏移
//   blockscale_offsets: [num_experts] — 每个专家在 blockscale 维度上的偏移
// 输出：
//   quant_output: 量化后的 FP8 激活
//   scale_factor: 每个 128 元素块的缩放因子
//
// 要求：SM >= 100（Blackwell 架构），k 必须对齐到 128。
// =============================================================================

#include <torch/all.h>

#include "mxfp8_experts_quant.cuh"

void mxfp8_experts_quant(const torch::Tensor& input,
                         const torch::Tensor& problem_sizes,
                         const torch::Tensor& expert_offsets,
                         const torch::Tensor& blockscale_offsets,
                         torch::Tensor& quant_output,
                         torch::Tensor& scale_factor) {
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
  TORCH_CHECK(input.dim() == 2, "input must be 2D tensor");
  TORCH_CHECK(input.size(1) % 128 == 0, "k must align to 128");
  TORCH_CHECK(input.strides()[1] == 1, "input must be row major");
  TORCH_CHECK(problem_sizes.dim() == 2, "problem_sizes must be 2D tensor");
  TORCH_CHECK(problem_sizes.dtype() == torch::kInt32,
              "problem_sizes must be int32");
  TORCH_CHECK(expert_offsets.dtype() == torch::kInt32,
              "expert_offsets must be int32");
  TORCH_CHECK(blockscale_offsets.dtype() == torch::kInt32,
              "blockscale_offsets must be int32");

  auto groups = problem_sizes.size(0);
  TORCH_CHECK(
      expert_offsets.dim() == 1 && expert_offsets.size(0) == groups,
      "expert_offsets must be 1D and have size equal to the number of groups");
  TORCH_CHECK(
      blockscale_offsets.dim() == 1 && blockscale_offsets.size(0) == groups,
      "blockscale_offsets must be 1D and have size equal to the number of "
      "groups");

  auto stream = at::cuda::getCurrentCUDAStream();
  if (input.dtype() == torch::kBFloat16) {
    expert_specialization::launch_mxfp8_experts_quant<__nv_bfloat16>(
        input, problem_sizes, expert_offsets, blockscale_offsets, quant_output,
        scale_factor);
  } else if (input.dtype() == torch::kFloat16) {
    expert_specialization::launch_mxfp8_experts_quant<__half>(
        input, problem_sizes, expert_offsets, blockscale_offsets, quant_output,
        scale_factor);
  } else {
    TORCH_CHECK(false, "dtype must be kFloat16 or kBFloat16");
  }
#else
  TORCH_CHECK(false,
              "No implemented mxfp8_experts_quant for "
              "current device");
#endif
}

#include "core/registration.h"

TORCH_LIBRARY_IMPL_EXPAND(TORCH_EXTENSION_NAME, CUDA, m) {
  m.impl("mxfp8_experts_quant", mxfp8_experts_quant);
}