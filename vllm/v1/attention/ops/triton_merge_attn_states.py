# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# 本模块实现了合并分裂 KV（split-KV）注意力结果的操作。
# 参考论文：https://www.arxiv.org/pdf/2501.01005 第 2.2 节
#
# 核心功能：
# merge_attn_states - 合并 prefix（前缀）和 suffix（后缀）的局部注意力结果
#
# 使用场景：
# 当 KV 序列被分成 prefix 和 suffix 两部分分别计算注意力后，
# 需要将它们合并为一个完整的注意力输出。
# 例如在分块预填充（chunked prefill）中，长序列被分成多个块处理。
#
# 合并算法（LSE 加权）：
# 1. 加载 prefix 和 suffix 的输出及其 LSE（log-sum-exp）值
# 2. 计算全局最大 LSE：max_lse = max(p_lse, s_lse)
# 3. 计算归一化权重：p_scale = exp(p_lse - max_lse) / (exp(p_lse - max_lse) + exp(s_lse - max_lse))
# 4. 加权合并：output = p_out * p_scale + s_out * s_scale
#
# 数值稳定性：
# - 使用 max_lse 进行数值稳定化，防止 exp 溢出
# - 处理 FA2 和 FA3 在空序列时 LSE 值的差异（inf vs -inf）
# - 支持 FP8 输出（通过 output_scale 参数）

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

# FP8 数据类型的数值范围信息
float8_info = torch.finfo(current_platform.fp8_dtype())


# Implements section 2.2 of https://www.arxiv.org/pdf/2501.01005
# can be used to combine partial attention results (in the split-KV case)
def merge_attn_states(
    # 合并分裂 KV 的局部注意力结果的 Python 封装函数
    output: torch.Tensor,
    prefix_output: torch.Tensor,
    prefix_lse: torch.Tensor,
    suffix_output: torch.Tensor,
    suffix_lse: torch.Tensor,
    output_lse: torch.Tensor | None = None,
    prefill_tokens_with_context: int | None = None,
    output_scale: torch.Tensor | None = None,
) -> None:
    # 获取维度信息
    num_tokens = output.shape[0]
    num_query_heads = output.shape[1]
    head_size = output.shape[2]
    # 填充到 2 的幂次以优化 Triton 向量化访问
    padded_head_size = triton.next_power_of_2(head_size)
    # We assume the output stride on num_head is not always as same as the
    # `suffix_output` and `prefix_output`, as them might be padded by the
    # attention backend.
    prefix_head_stride = prefix_output.stride(1)
    output_head_stride = output.stride(1)

    # 如果 prefill_tokens_with_context 为 None，所有 token 都使用 prefix 上下文
    if prefill_tokens_with_context is None:
        prefill_tokens_with_context = num_tokens

    # TODO(woosuk): Use CUDA kernel instead of Triton to minimize CPU overhead.
    merge_attn_states_kernel[(num_tokens, num_query_heads)](
        output,
        output_lse,
        prefix_output,
        prefix_lse,
        suffix_output,
        suffix_lse,
        prefix_head_stride,
        output_head_stride,
        output_scale,
        head_size,
        padded_head_size,
        output_lse is not None,
        prefill_tokens_with_context,
        output_scale is not None,
    )


@triton.jit
def merge_attn_states_kernel(
    # 合并注意力状态的 Triton kernel
    # 网格维度：(num_tokens, num_query_heads)
    # 每个程序处理一个 (token, head) 对
    output,  # [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
    output_lse,  # [NUM_HEADS, NUM_TOKENS]
    prefix_output,  # [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
    prefix_lse,  # [NUM_HEADS, NUM_TOKENS]
    suffix_output,  # [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
    suffix_lse,  # [NUM_HEADS, NUM_TOKENS]
    prefix_head_stride,
    output_head_stride,
    output_scale,  # scale tensor or None
    HEAD_SIZE: tl.constexpr,
    PADDED_HEAD_SIZE: tl.constexpr,
    OUTPUT_LSE: tl.constexpr,
    prefill_tokens_with_context: tl.constexpr,
    USE_FP8: tl.constexpr,
    FP8_MIN: tl.constexpr = float8_info.min,
    FP8_MAX: tl.constexpr = float8_info.max,
):
    token_idx = tl.program_id(0)
    num_tokens = tl.num_programs(0)
    head_idx = tl.program_id(1)
    num_heads = tl.num_programs(1)

    # 判断当前 token 是否有 prefix 上下文
    prefix_mask = token_idx < prefill_tokens_with_context

    head_arange = tl.arange(0, PADDED_HEAD_SIZE)
    head_mask = head_arange < HEAD_SIZE

    # 对于没有 prefix 上下文的 token（token_idx >= prefill_tokens_with_context），
    # 直接复制 suffix 输出
    if not prefix_mask:
        s_lse = tl.load(suffix_lse + head_idx * num_tokens + token_idx)
        if OUTPUT_LSE:
            tl.store(output_lse + head_idx * num_tokens + token_idx, s_lse)

        s_out = tl.load(
            suffix_output
            + token_idx * num_heads * prefix_head_stride
            + head_idx * prefix_head_stride
            + head_arange,
            mask=head_mask,
        )

        if USE_FP8:
            s_out = s_out * (1.0 / tl.load(output_scale))
            s_out = tl.clamp(s_out, FP8_MIN, FP8_MAX)
            s_out = s_out.to(output.dtype.element_ty)

        tl.store(
            output
            + token_idx * num_heads * output_head_stride
            + head_idx * output_head_stride
            + head_arange,
            s_out,
            mask=head_mask,
        )
        return

    # 对于有 prefix 上下文的 token，执行正常的合并操作
    # 加载 prefix 和 suffix 的 LSE 值
    p_lse = tl.load(prefix_lse + head_idx * num_tokens + token_idx)
    s_lse = tl.load(suffix_lse + head_idx * num_tokens + token_idx)

    # FA2 和 FA3 在 sum-exp 为 0 时行为不同（空序列）。
    # FA3 返回 -inf，FA2 返回 inf。
    # 如果看到 inf，假设是 FA2 并转换为 -inf 以保持一致性和正确性。
    p_lse = float("-inf") if p_lse == float("inf") else p_lse
    s_lse = float("-inf") if s_lse == float("inf") else s_lse

    # 计算全局最大 LSE 以确保数值稳定性
    max_lse = tl.maximum(p_lse, s_lse)
    # 将 LSE 减去最大值（数值稳定化）
    p_lse = p_lse - max_lse
    s_lse = s_lse - max_lse
    # 计算指数值（用于权重计算）
    p_se = tl.exp(p_lse)
    s_se = tl.exp(s_lse)
    out_se = p_se + s_se

    # 如果需要输出 LSE，计算并存储全局 LSE
    if OUTPUT_LSE:
        out_lse = tl.log(out_se) + max_lse
        tl.store(output_lse + head_idx * num_tokens + token_idx, out_lse)

    # 加载 prefix 和 suffix 的输出
    p_out = tl.load(
        prefix_output
        + token_idx * num_heads * prefix_head_stride
        + head_idx * prefix_head_stride
        + head_arange,
        mask=head_mask,
    )
    s_out = tl.load(
        suffix_output
        + token_idx * num_heads * prefix_head_stride
        + head_idx * prefix_head_stride
        + head_arange,
        mask=head_mask,
    )

    # 注意数值稳定性：先计算缩放因子，再与输出相乘。
    # 不要直接用 tl.exp(p_lse) 或 tl.exp(s_lse) 乘以输出。
    # 归一化权重：每个部分的贡献比例
    p_scale = p_se / out_se
    s_scale = s_se / out_se
    # 加权合并
    out = p_out * p_scale + s_out * s_scale

    if USE_FP8:
        out = out * (1.0 / tl.load(output_scale))
        out = tl.clamp(out, FP8_MIN, FP8_MAX)
        out = out.to(output.dtype.element_ty)

    tl.store(
        output
        + token_idx * num_heads * output_head_stride
        + head_idx * output_head_stride
        + head_arange,
        out,
        mask=head_mask,
    )
