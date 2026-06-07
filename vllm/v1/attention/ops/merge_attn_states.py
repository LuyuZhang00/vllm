# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# 中文注释：本文件实现了注意力状态合并操作，用于将 prefix（KV cache 中已有的历史 KV）
# 和 suffix（当前 step 新计算的 KV）的注意力输出合并为最终结果。
#
# 【使用场景】
#   在 vLLM 的 prefix cache 机制中，当一个请求的前缀已经被其他请求计算过并缓存了 KV，
#   那么当前请求可以复用这些已缓存的 KV（prefix），只需计算新 token 的 KV（suffix）。
#   但注意力输出需要同时考虑 prefix 和 suffix 两部分的贡献，因此需要合并。
#
# 【合并原理】（基于 LSE 校正方法，参考 https://arxiv.org/pdf/2501.01005 第 2.2 节）
#   对于每个 query token，其正确的注意力输出为：
#     out = softmax(QK^T/sqrt(d)) * V
#   由于 prefix 和 suffix 是分别计算的，各自有独立的 softmax 归一化：
#     out_prefix = softmax_prefix(QK_prefix^T/sqrt(d)) * V_prefix  （已缓存）
#     out_suffix = softmax_suffix(QK_suffix^T/sqrt(d)) * V_suffix  （新计算）
#   合并时需要使用 LSE (Log-Sum-Exp) 进行校正：
#     out_merged = (out_prefix * exp(LSE_prefix) + out_suffix * exp(LSE_suffix))
#                  / exp(LSE_merged)
#   其中 LSE_merged = log(exp(LSE_prefix) + exp(LSE_suffix))
#
# 【实现选择】
#   - CUDA 平台且 head_size 满足对齐要求时，使用自定义 C++/CUDA kernel（更高效）
#   - 其他情况回退到 Triton kernel 实现

import torch

from vllm.platforms import current_platform


def merge_attn_states(
    output: torch.Tensor,
    prefix_output: torch.Tensor,
    prefix_lse: torch.Tensor,
    suffix_output: torch.Tensor,
    suffix_lse: torch.Tensor,
    output_lse: torch.Tensor | None = None,
    prefill_tokens_with_context: int | None = None,
    output_scale: torch.Tensor | None = None,
) -> None:
    """Merge partial attention outputs from prefix (KV cache) and suffix
    (new tokens) into a single output tensor using the log-sum-exp (LSE)
    rescaling method described in section 2.2 of
    https://www.arxiv.org/pdf/2501.01005.

    For tokens that have prefix context (token index < prefill_tokens_with_context),
    the prefix and suffix partial outputs are combined as a weighted sum.
    For tokens without prefix context, the suffix output is copied directly.

    Args:
        output: Output tensor of shape [NUM_TOKENS, NUM_HEADS, HEAD_SIZE].
        prefix_output: Partial attention output over the prefix (KV cache),
            shape [NUM_TOKENS, NUM_HEADS, HEAD_SIZE].
        prefix_lse: Log-sum-exp values for the prefix attention,
            shape [NUM_HEADS, NUM_TOKENS].
        suffix_output: Partial attention output over the suffix (new KV),
            shape [NUM_TOKENS, NUM_HEADS, HEAD_SIZE].
        suffix_lse: Log-sum-exp values for the suffix attention,
            shape [NUM_HEADS, NUM_TOKENS].
        output_lse: Optional tensor to store the merged LSE values,
            shape [NUM_HEADS, NUM_TOKENS]. If None, LSE is not written out.
        prefill_tokens_with_context: Number of prefill tokens that have
            prefix context and therefore require merging. Tokens at indices
            >= this value are decode or context-free prefill tokens whose
            output is taken directly from suffix_output. If None, all tokens
            are treated as having context.
        output_scale: Optional scalar tensor for FP8 static quantization.
            When provided, output must be FP8 dtype.
    """

    # NOTE(DefTruth): Currently, custom merge_attn_states CUDA kernel
    # does not support FP8 dtype for inputs, fallback to use Triton kernel.
    # However, when output_scale is provided, the inputs are still BF16/FP16
    # and the output is FP8 — both CUDA and Triton support this.
    # FP8 output requires output_scale to be set.

    # 中文注释：FP8 输出必须提供 output_scale（静态量化缩放因子）
    if output.dtype not in (torch.float32, torch.half, torch.bfloat16):
        assert output_scale is not None, (
            f"output_scale is required when output is {output.dtype}"
        )

    # 中文注释：检查输入 dtype 是否被 CUDA kernel 支持
    # CUDA kernel 目前只支持 float32, float16, bfloat16 三种 dtype
    def supported_dtypes(prefix: torch.Tensor) -> bool:
        return prefix.dtype in [torch.float32, torch.half, torch.bfloat16]

    # NOTE(DefTruth): Currently, custom merge_attn_states CUDA
    # kernel load/store 128b(16 bytes) per memory issue within
    # thread. Namely, the headsize(headdim) must be multiple of
    # pack_size based on input dtype (float32 -> 4, half/bfloat16 -> 8).

    # 中文注释：检查 head_size 是否满足 CUDA kernel 的对齐要求。
    # CUDA kernel 每个线程加载/存储 128 bits (16 bytes)，因此 head_size 必须是：
    #   - float32 (4 bytes): head_size % 4 == 0
    #   - float16/bfloat16 (2 bytes): head_size % 8 == 0
    # 不满足时回退到 Triton kernel（无此对齐要求）。
    def supported_headdim(prefix: torch.Tensor) -> bool:
        headdim = prefix.shape[2]  # [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
        if prefix.dtype == torch.float32:
            return headdim % 4 == 0
        return headdim % 8 == 0

    # 中文注释：根据平台和 dtype/对齐条件选择 kernel 实现。
    # 优先使用 CUDA 自定义 kernel（性能更好），否则回退到 Triton kernel。
    if (
        current_platform.is_cuda()
        and supported_dtypes(prefix_output)
        and supported_headdim(prefix_output)
    ):
        from vllm._custom_ops import merge_attn_states

        return merge_attn_states(
            output,
            prefix_output,
            prefix_lse,
            suffix_output,
            suffix_lse,
            output_lse,
            prefill_tokens_with_context,
            output_scale,
        )
    else:
        from vllm.v1.attention.ops.triton_merge_attn_states import (
            merge_attn_states,
        )

        return merge_attn_states(
            output,
            prefix_output,
            prefix_lse,
            suffix_output,
            suffix_lse,
            output_lse,
            prefill_tokens_with_context,
            output_scale,
        )
