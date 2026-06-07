# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
This file contains ops for ViT attention to be compatible with torch.compile
as there are operations here not supported by torch.compile (for instance,
`.item()` in flash attention)

Using these ops and wrapping vision blocks with `torch.compile` can speed up
throughput in vision models by ~5% relative on H100, and improve token
latencies by ~7% (see qwen2_5_vl for example usage)

To use these ops, you must have a recent version of PyTorch installed (>= 2.4.0)
"""

# 本模块为 Vision Transformer（ViT）提供多种注意力后端的包装器，
# 使 ViT 能够与 torch.compile 兼容并获得性能优化。
#
# 提供的注意力后端：
# 1. flash_attn_maxseqlen_wrapper - FlashAttention 包装器
#    支持 FlashAttention v2/v3 和 ROCm AITER
#    使用 varlen 接口处理变长序列
#
# 2. triton_attn_wrapper - Triton 注意力包装器
#    使用 context_attention_fwd 进行预填充注意力
#
# 3. torch_sdpa_wrapper - PyTorch SDPA 包装器
#    使用 F.scaled_dot_product_attention
#    支持 GQA（Grouped Query Attention）
#
# 4. flashinfer_wrapper - FlashInfer 包装器
#    使用 cuDNN batch prefill with KV cache
#    支持 FP8 量化和自定义数据类型
#
# 性能优化：
# - 使用 torch.compile 兼容的自定义算子（direct_register_custom_op）
# - 在 H100 上可提升约 5% 的吞吐量和约 7% 的延迟
# - 需要 PyTorch >= 2.4.0

import einops
import torch
import torch.nn.functional as F

from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op


def flash_attn_maxseqlen_wrapper(
    # FlashAttention 包装器：使用 varlen 接口处理变长序列
    # 支持 ROCm AITER 和 FlashAttention v2/v3
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    batch_size: int,
    is_rocm_aiter: bool,
    fa_version: int | None,
    scale: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
    max_seqlen: torch.Tensor | None = None,
) -> torch.Tensor:
    # 根据平台选择 FlashAttention 实现
    kwargs = {}
    if is_rocm_aiter:
        from aiter import flash_attn_varlen_func
    else:
        from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func

        # 非 ROCm 平台且指定了 fa_version 时传入版本参数
        if not current_platform.is_rocm() and fa_version is not None:
            kwargs["fa_version"] = fa_version

    q_len = q.size(1)
    # 如果未提供 cu_seqlens，构建均匀分布的累积序列长度
    if cu_seqlens is None:
        cu_seqlens = torch.arange(
            0, (batch_size + 1) * q_len, step=q_len, dtype=torch.int32, device=q.device
        )
    max_seqlen = q_len if max_seqlen is None else max_seqlen.item()

    # 将 [batch, seq_len, ...] 重塑为 [(batch * seq_len), ...] 以适配 varlen 接口
    q, k, v = (einops.rearrange(x, "b s ... -> (b s) ...") for x in [q, k, v])
    # 调用 FlashAttention varlen 接口（非因果，ViT 使用双向注意力）
    output = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        dropout_p=0.0,
        causal=False,
        softmax_scale=scale,
        **kwargs,
    )
    # 将输出重塑回 [batch, seq_len, num_heads, head_dim]
    context_layer = einops.rearrange(output, "(b s) h d -> b s h d", b=batch_size)
    return context_layer


def flash_attn_maxseqlen_wrapper_fake(
    # FlashAttention 包装器的 fake 实现（用于 torch.compile 的形状推断）
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    batch_size: int,
    is_rocm_aiter: bool,
    fa_version: int | None,
    scale: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
    max_seqlen: torch.Tensor | None = None,
) -> torch.Tensor:
    return torch.empty_like(q)


# 注册为 torch.compile 兼容的自定义算子
direct_register_custom_op(
    op_name="flash_attn_maxseqlen_wrapper",
    op_func=flash_attn_maxseqlen_wrapper,
    fake_impl=flash_attn_maxseqlen_wrapper_fake,
)


def vit_flash_attn_wrapper(
    # ViT FlashAttention 包装器的公开接口
    # 通过 torch.ops.vllm.flash_attn_maxseqlen_wrapper 调用注册的自定义算子
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    batch_size: int,
    is_rocm_aiter: bool,
    fa_version: int | None,
    scale: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
    max_seqlen: torch.Tensor | None = None,
) -> torch.Tensor:
    return torch.ops.vllm.flash_attn_maxseqlen_wrapper(
        q,
        k,
        v,
        batch_size,
        is_rocm_aiter,
        fa_version,
        scale,
        cu_seqlens,
        max_seqlen,
    )


def triton_attn_wrapper(
    # Triton 注意力包装器：使用 context_attention_fwd 进行预填充注意力
    # 适用于无法使用 FlashAttention 的场景
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    batch_size: int,
    scale: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
    max_seqlen: torch.Tensor | None = None,
) -> torch.Tensor:
    from vllm.v1.attention.ops.triton_prefill_attention import context_attention_fwd

    q_len = q.size(1)
    # 构建累积序列长度（如果未提供）
    if cu_seqlens is None:
        cu_seqlens = torch.arange(
            0, (batch_size + 1) * q_len, step=q_len, dtype=torch.int32, device=q.device
        )
    max_seqlen = q_len if max_seqlen is None else max_seqlen.item()

    # 重塑为 varlen 格式
    q, k, v = (einops.rearrange(x, "b s ... -> (b s) ...") for x in [q, k, v])
    output = torch.empty_like(q)
    context_attention_fwd(
        q,
        k,
        v,
        output,
        b_start_loc=cu_seqlens[:-1],
        b_seq_len=cu_seqlens[1:] - cu_seqlens[:-1],
        max_input_len=max_seqlen,
        is_causal=False,
        sliding_window_q=None,
        sliding_window_k=None,
        softmax_scale=scale,
    )

    context_layer = einops.rearrange(output, "(b s) h d -> b s h d", b=batch_size)
    return context_layer


def triton_attn_wrapper_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    batch_size: int,
    scale: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
    max_seqlen: torch.Tensor | None = None,
) -> torch.Tensor:
    return torch.empty_like(q)


direct_register_custom_op(
    op_name="triton_attn_wrapper",
    op_func=triton_attn_wrapper,
    fake_impl=triton_attn_wrapper_fake,
)


def vit_triton_attn_wrapper(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    batch_size: int,
    scale: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
    max_seqlen: torch.Tensor | None = None,
) -> torch.Tensor:
    return torch.ops.vllm.triton_attn_wrapper(
        q,
        k,
        v,
        batch_size,
        scale,
        cu_seqlens,
        max_seqlen,
    )


def apply_sdpa(
    # 应用 PyTorch 的 scaled_dot_product_attention
    # 输入形状：(batch_size, seq_len, num_heads, head_size)
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
    enable_gqa: bool = False,
) -> torch.Tensor:
    """
    Input shape:
    (batch_size x seq_len x num_heads x head_size)

    应用 PyTorch 的 scaled_dot_product_attention。
    将输入从 [batch, seq, heads, dim] 转换为 [batch, heads, seq, dim] 以适配 SDPA。
    """
    # 转换为 SDPA 期望的 [batch, heads, seq, dim] 格式
    q, k, v = (einops.rearrange(x, "b s h d -> b h s d") for x in [q, k, v])
    # 调用 PyTorch SDPA（非因果，无 dropout）
    output = F.scaled_dot_product_attention(
        q, k, v, dropout_p=0.0, scale=scale, enable_gqa=enable_gqa
    )
    # 转换回 [batch, seq, heads, dim] 格式
    output = einops.rearrange(output, "b h s d -> b s h d ")
    return output


# TODO: Once we have a torch 2.10, we can use tensor slices
# so we won't need to wrap this in custom ops
def torch_sdpa_wrapper(
    # PyTorch SDPA 包装器：处理变长序列的注意力计算
    # 当提供 cu_seqlens 时，按序列长度分割后逐个计算
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
    enable_gqa: bool = False,
) -> torch.Tensor:
    # ROCm 平台必须保持张量连续性，否则会出现幻觉问题
    if current_platform.is_rocm():
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

    # 如果没有提供 cu_seqlens，直接调用 apply_sdpa
    if cu_seqlens is None:
        return apply_sdpa(q, k, v, scale=scale, enable_gqa=enable_gqa)

    # 变长序列处理：按序列长度分割后逐个计算
    outputs = []

    lens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
    q_chunks = torch.split(q, lens, dim=1)
    k_chunks = torch.split(k, lens, dim=1)
    v_chunks = torch.split(v, lens, dim=1)
    for q_i, k_i, v_i in zip(q_chunks, k_chunks, v_chunks):
        output_i = apply_sdpa(q_i, k_i, v_i, scale=scale, enable_gqa=enable_gqa)
        outputs.append(output_i)
    context_layer = torch.cat(outputs, dim=1)
    return context_layer


def torch_sdpa_wrapper_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None,
    cu_seqlens: torch.Tensor | None,
    enable_gqa: bool = False,
) -> torch.Tensor:
    return torch.empty_like(q)


direct_register_custom_op(
    op_name="torch_sdpa_wrapper",
    op_func=torch_sdpa_wrapper,
    fake_impl=torch_sdpa_wrapper_fake,
)


def vit_torch_sdpa_wrapper(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
    enable_gqa: bool = False,
) -> torch.Tensor:
    return torch.ops.vllm.torch_sdpa_wrapper(
        q, k, v, scale, cu_seqlens, enable_gqa=enable_gqa
    )


def flashinfer_wrapper(
    # FlashInfer 包装器：使用 cuDNN batch prefill with KV cache
    # 支持 FP8 量化和自定义输出数据类型
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    workspace_buffer: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
    max_seqlen: torch.Tensor | None = None,
    sequence_lengths: torch.Tensor | None = None,
    q_scale: torch.Tensor | None = None,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
    o_data_type: torch.dtype | None = None,
) -> torch.Tensor:
    from flashinfer.prefill import cudnn_batch_prefill_with_kv_cache

    # 检查输入是否已经是 4D 格式
    is_reshaped = q.dim() == 4

    if is_reshaped:
        reshape_batch_size = q.shape[0]
        # 重塑为 varlen 格式
        q, k, v = (einops.rearrange(x, "b s ... -> (b s) ...") for x in [q, k, v])
    # cuDNN <= 9.10.2.21 要求 q, k 是连续的
    # 对于使用 RoPE 的 ViT 没有额外开销，因为 RoPE 已经使 q 和 k 连续
    q, k = q.contiguous(), k.contiguous()

    assert cu_seqlens is not None
    assert max_seqlen is not None
    assert sequence_lengths is not None
    # cu_seqlens 包含两部分：前半部分是 Q/K 的累积长度，后半部分是 V 的累积长度
    assert len(cu_seqlens) % 2 == 0, "cu_seqlens must be divisible by 2"
    cu_seqlength = len(cu_seqlens) // 2
    batch_offsets_qko = cu_seqlens[:cu_seqlength].view(-1, 1, 1, 1)
    batch_offsets_v = cu_seqlens[cu_seqlength:].view(-1, 1, 1, 1)
    sequence_lengths = sequence_lengths.view(-1, 1, 1, 1)
    max_seqlen = max_seqlen.item()

    # 调用 cuDNN batch prefill with KV cache（非因果）
    output, _ = cudnn_batch_prefill_with_kv_cache(
        q,
        k,
        v,
        scale,
        workspace_buffer,
        max_token_per_sequence=max_seqlen,
        max_sequence_kv=max_seqlen,
        actual_seq_lens_q=sequence_lengths,
        actual_seq_lens_kv=sequence_lengths,
        causal=False,
        return_lse=False,
        batch_offsets_q=batch_offsets_qko,
        batch_offsets_k=batch_offsets_qko,
        batch_offsets_v=batch_offsets_v,
        batch_offsets_o=batch_offsets_qko,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        o_data_type=o_data_type,
    )

    # 如果输入是 4D 格式，将输出重塑回去
    if is_reshaped:
        output = einops.rearrange(output, "(b s) h d -> b s h d", b=reshape_batch_size)

    return output


def vit_flashinfer_wrapper_fake(
    # FlashInfer 包装器的 fake 实现（用于 torch.compile 的形状推断）
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    workspace_buffer: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
    max_seqlen: torch.Tensor | None = None,
    sequence_lengths: torch.Tensor | None = None,
    q_scale: torch.Tensor | None = None,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
    o_data_type: torch.dtype | None = None,
) -> torch.Tensor:
    return torch.empty_like(q, dtype=o_data_type or q.dtype)


# 注册为 torch.compile 兼容的自定义算子
direct_register_custom_op(
    op_name="flashinfer_wrapper",
    op_func=flashinfer_wrapper,
    fake_impl=vit_flashinfer_wrapper_fake,
)


def vit_flashinfer_wrapper(
    # ViT FlashInfer 包装器的公开接口
    # 通过 torch.ops.vllm.flashinfer_wrapper 调用注册的自定义算子
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    workspace_buffer: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
    max_seqlen: torch.Tensor | None = None,
    sequence_lengths: torch.Tensor | None = None,
    q_scale: torch.Tensor | None = None,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
    o_data_type: torch.dtype | None = None,
) -> torch.Tensor:
    return torch.ops.vllm.flashinfer_wrapper(
        q,
        k,
        v,
        scale,
        workspace_buffer,
        cu_seqlens,
        max_seqlen,
        sequence_lengths,
        q_scale,
        k_scale,
        v_scale,
        o_data_type,
    )
