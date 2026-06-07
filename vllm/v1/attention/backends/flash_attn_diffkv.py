# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention layer with FlashAttention."""
"""
FlashAttention DiffKV 注意力后端。

本模块实现了支持 K/V 头维度不同的 FlashAttention 变体（DiffKV）。

标准 FlashAttention 中，Key 和 Value 的头维度（head_size）相同。但在某些
新型模型架构（如 DeepSeek-V2 等）中，Key 和 Value 可以有不同的头维度：
- head_size_k: Key 的头维度（如 192）
- head_size_v: Value 的头维度（如 128）

这种设计可以：
1. 减少 KV 缓存的内存占用（Value 使用较小的头维度）
2. 保持 Key 的高维表示以提高注意力精度
3. 总体上在性能和精度之间取得更好的平衡

与标准 FlashAttention 的关键差异：
1. KV 缓存形状：[num_blocks, block_size, num_kv_heads, head_size_k + head_size_v]
   K 和 V 合并存储在最后一个维度上
2. KV 缓存更新：使用 triton_reshape_and_cache_flash_diffkv 处理合并布局
3. 前向计算：从合并的 KV 缓存中分别提取 K 和 V 的缓存
4. FA 版本检测：考虑 DiffKV 的头维度差异选择正确的 FA 版本
5. 步长修复：对 size-1 维度的退化步长进行规范化（FA3/4 TMA 要求 >=16 字节对齐）

继承自 FlashAttentionBackend，大部分能力声明复用父类，仅重写：
- get_kv_cache_shape：考虑 head_size_v 差异
- get_kv_cache_stride_order：处理合并维度的步长顺序
- get_impl_cls：返回 FlashAttentionDiffKVImpl

输出形状：[num_tokens, num_heads * head_size_v]
（注意：输出维度基于 Value 的头维度，而非 Key 的头维度）
"""

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.torch_utils import (
    canonicalize_singleton_dim_strides,
    is_quantized_kv_cache,
)
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.fa_utils import (
    get_flash_attn_version,
    is_flash_attn_varlen_func_available,
)
from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
    triton_reshape_and_cache_flash_diffkv,
)

if is_flash_attn_varlen_func_available():
    from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func
from vllm.v1.attention.backends.utils import get_kv_cache_layout

from .flash_attn import (
    FlashAttentionBackend,
    FlashAttentionImpl,
    FlashAttentionMetadata,
    cascade_attention,
)

logger = init_logger(__name__)


class FlashAttentionDiffKVBackend(FlashAttentionBackend):
    """
    FlashAttention DiffKV 后端类。

    继承自标准 FlashAttention 后端，主要修改：
    1. 支持 Value 头维度与 Key 头维度不同
    2. KV 缓存中 K 和 V 合并存储（最后一个维度为 head_size_k + head_size_v）
    3. 步长顺序（stride order）需要考虑合并维度

    默认 head_size_v = 128，可通过 set_head_size_v 类方法修改。
    """

    # Default to 128 for this backend
    # 默认 Value 头维度为 128
    head_size_v: int = 128

    @classmethod
    def set_head_size_v(cls, head_size_v: int) -> None:
        """
        设置 Value 的头维度大小。

        需要在模型加载后、推理开始前调用。
        """
        cls.head_size_v = head_size_v

    @staticmethod
    def get_name() -> str:
        """返回后端名称标识。"""
        return "FLASH_ATTN_DIFFKV"

    @staticmethod
    def get_impl_cls() -> type["FlashAttentionImpl"]:
        """返回 DiffKV 注意力实现类。"""
        return FlashAttentionDiffKVImpl

    # Do not modify the interface of get_kv_cache_shape,
    # but consider head_size_v when returning result.
    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        """
        返回 DiffKV 的 KV 缓存形状。

        与标准 FlashAttn 不同，K 和 V 合并存储在最后一个维度：
        [num_blocks, block_size, num_kv_heads, head_size_k + head_size_v]

        其中 head_size_k = head_size（Key 头维度），
        head_size_v = FlashAttentionDiffKVBackend.head_size_v（Value 头维度）。
        """
        if block_size % 16 != 0:
            raise ValueError("Block size must be a multiple of 16.")
        return (
            num_blocks,
            block_size,
            num_kv_heads,
            head_size + FlashAttentionDiffKVBackend.head_size_v,
        )

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        """
        返回 KV 缓存的步长顺序（stride order）。

        步长顺序是一个排列（permutation），指示如何从 get_kv_cache_shape
        的逻辑形状转换到实际的内存布局。

        根据 KV 缓存布局格式（NHD 或 HND）返回不同的步长顺序：
        - NHD（Num_blocks-Head-Dim）：标准布局
        - HND（Head-Num_blocks-Dim）：头优先布局

        当包含 num_layers 维度时（用于 KV 共享场景），步长顺序会相应调整。
        """
        # `stride_order` indicates the permutation that gets
        # us from `get_kv_cache_shape` to the actual memory layout we want.
        cache_layout = get_kv_cache_layout()
        if cache_layout == "NHD" and include_num_layers_dimension:
            # (num_blocks, num_layers, block_size,
            # num_kv_heads, head_size + head_size_v)
            return (1, 0, 2, 3, 4)
        elif cache_layout == "NHD":
            stride_order = (0, 1, 2, 3)
        elif cache_layout == "HND" and include_num_layers_dimension:
            # (num_blocks, num_kv_heads, num_layers,
            # block_size, head_size + head_size_v)
            return (1, 3, 0, 2, 4)
        elif cache_layout == "HND":
            stride_order = (0, 2, 1, 3)
        else:
            raise ValueError(f"Unknown cache layout format {cache_layout}.")
        return stride_order


class FlashAttentionDiffKVImpl(FlashAttentionImpl):
    """
    FlashAttention DiffKV 前向计算实现类。

    继承自标准 FlashAttentionImpl，主要修改：
    1. KV 缓存更新：使用 DiffKV 专用的 triton kernel
    2. 前向计算：从合并的 KV 缓存中分离 K 和 V
    3. FA 版本检测：考虑 DiffKV 的头维度差异
    4. 步长修复：规范化 size-1 维度的退化步长

    关键实现细节：
    - KV 缓存是合并的，最后一个维度为 head_size_k + head_size_v
    - 通过切片 [:head_size] 和 [head_size:] 分别获取 K 和 V 缓存
    - canonicalize_singleton_dim_strides 修复 TMA 对齐要求
    """

    vllm_flash_attn_version: int | None

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Re-derive the FA version with diff-kv context so that
        # get_flash_attn_version can apply the FA3 -> FA4 upgrade rule
        # for sinks + hdim != hdim_v.
        # 重新检测 FA 版本，考虑 DiffKV 的 head_size_v 差异
        # 当存在 sinks 且 hdim != hdim_v 时，可能需要 FA3 -> FA4 升级
        self.vllm_flash_attn_version = get_flash_attn_version(
            requires_alibi=self.alibi_slopes is not None,
            head_size=self.head_size,
            head_size_v=FlashAttentionDiffKVBackend.head_size_v,
            has_sinks=self.sinks is not None,
        )

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        """
        更新 DiffKV 的 KV 缓存。

        与标准 FlashAttn 不同，DiffKV 将 K 和 V 合并存储在同一个张量中：
        kv_cache shape: [num_blocks, block_size, num_kv_heads, head_size_k + head_size_v]

        使用 triton_reshape_and_cache_flash_diffkv kernel 处理这种合并布局。

        注意：key 和 value 可能有 padding，但 slot_mapping 没有 padding。
        reshape_and_cache_flash op 使用 slot_mapping 的形状来确定实际 token 数。
        """
        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            # For encoder attention,
            # we use direct Q, K, V tensors without caching
            return

        # Unlike standard FlashAttn which splits kv_cache via unbind(0),
        # DiffKV packs K and V into a single tensor along the last dim:
        #   kv_cache shape: [num_blocks, block_size, num_kv_heads,
        #                    head_size_k + head_size_v]
        # The triton kernel handles this combined layout directly.
        #
        # NOTE(woosuk): key and value are padded while slot_mapping is
        # not padded. However, we don't need to do key[:num_actual_tokens]
        # and value[:num_actual_tokens] because the reshape_and_cache_flash
        # op uses the slot_mapping's shape to determine the number of
        # actual tokens.
        triton_reshape_and_cache_flash_diffkv(
            key,
            value,
            kv_cache,
            slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
        )

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with FlashAttention.

        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size_v]
            kv_cache: shape =
                [num_blocks, block_size, num_kv_heads, head_size + head_size_v]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size_v]
        NOTE: FP8 quantization, flash-attn expect the size of
              {q,k,v}_descale to be (num_sequences, num_kv_heads).
              We use torch's .expand() to avoid duplicating values
        """
        assert self.vllm_flash_attn_version is not None, (
            "FlashAttention version not detected."
        )

        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not yet supported for FlashAttentionImpl"
            )

        if attn_metadata is None:
            # Profiling run.
            return output.fill_(0)

        attn_type = self.attn_type

        # IMPORTANT!
        # NOTE(woosuk): With piece-wise CUDA graphs, this method is executed in
        # eager-mode PyTorch. Thus, we need to be careful about any CPU overhead
        # in this method. For example, `view` and `slice` (or `[:n]`) operations
        # are surprisingly slow even in the case they do not invoke any GPU ops.
        # Minimize the PyTorch ops in this method as much as possible.
        # Whenever making a change in this method, please benchmark the
        # performance to make sure it does not introduce any overhead.

        num_actual_tokens = attn_metadata.num_actual_tokens

        # Handle encoder attention differently - no KV cache needed
        if attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            # For encoder attention,
            # we use direct Q, K, V tensors without caching
            return self._forward_encoder_attention(
                query[:num_actual_tokens],
                key[:num_actual_tokens],
                value[:num_actual_tokens],
                output[:num_actual_tokens],
                attn_metadata,
                layer,
            )

        # For decoder and cross-attention, use KV cache as before
        # Different head_size for K and V
        # 从合并的 KV 缓存中分离 K 和 V 缓存
        # K 缓存在前 head_size 维度，V 缓存在后 head_size_v 维度
        key_cache = kv_cache[..., : self.head_size]
        value_cache = kv_cache[..., self.head_size :]
        # Fix degenerate strides on size-1 dims (e.g. num_kv_heads=1 with TP).
        # FA3/4 on H100+ uses TMA, which requires >=16-byte stride alignment.
        # See vllm.utils.torch_utils.canonicalize_singleton_dim_strides.
        # 修复 size-1 维度的退化步长。
        # FA3/4 在 H100+ 上使用 TMA（Tensor Memory Access），
        # 要求步长 >=16 字节对齐。
        fixed_k = canonicalize_singleton_dim_strides(key_cache)
        fixed_v = canonicalize_singleton_dim_strides(value_cache)
        if fixed_k is not key_cache or fixed_v is not value_cache:
            logger.debug(
                "Canonicalized degenerate KV cache strides (FlashAttentionDiffKV): "
                "shape=%s, key strides before=%s after=%s, "
                "value strides before=%s after=%s",
                key_cache.shape,
                key_cache.stride(),
                fixed_k.stride(),
                value_cache.shape,
                value_cache.stride(),
                fixed_v.stride(),
            )
        key_cache, value_cache = fixed_k, fixed_v

        if is_quantized_kv_cache(self.kv_cache_dtype):
            # queries are quantized in the attention layer
            # 将缓存视图转换为 FP8 数据类型
            key_cache = key_cache.view(current_platform.fp8_dtype())
            value_cache = value_cache.view(current_platform.fp8_dtype())

        if not attn_metadata.use_cascade:
            cu_seqlens_q = attn_metadata.query_start_loc
            seqused_k = attn_metadata.seq_lens
            max_seqlen_q = attn_metadata.max_query_len
            max_seqlen_k = attn_metadata.max_seq_len
            block_table = attn_metadata.block_table
            scheduler_metadata = attn_metadata.scheduler_metadata

            # FP8 反量化缩放因子的形状
            descale_shape = (cu_seqlens_q.shape[0] - 1, self.num_kv_heads)

            if self.dcp_world_size > 1:
                # 分布式上下文并行（DCP）路径
                self._forward_with_dcp(
                    query[:num_actual_tokens],
                    key[:num_actual_tokens],
                    value[:num_actual_tokens],
                    key_cache,
                    value_cache,
                    output[:num_actual_tokens],
                    attn_metadata,
                    q_descale=layer._q_scale.expand(descale_shape),
                    k_descale=layer._k_scale.expand(descale_shape),
                    v_descale=layer._v_scale.expand(descale_shape),
                )
                return output
            else:
                # 标准单卡路径
                sliding_window_size = (
                    list(self.sliding_window)
                    if self.sliding_window is not None
                    else None
                )
                flash_attn_varlen_func(
                    q=query[:num_actual_tokens],
                    k=key_cache,
                    v=value_cache,
                    out=output[:num_actual_tokens],
                    cu_seqlens_q=cu_seqlens_q,
                    max_seqlen_q=max_seqlen_q,
                    seqused_k=seqused_k,
                    max_seqlen_k=max_seqlen_k,
                    softmax_scale=self.scale,
                    causal=attn_metadata.causal,
                    alibi_slopes=self.alibi_slopes,
                    window_size=sliding_window_size,
                    block_table=block_table,
                    softcap=self.logits_soft_cap,
                    scheduler_metadata=scheduler_metadata,
                    fa_version=self.vllm_flash_attn_version,
                    q_descale=layer._q_scale.expand(descale_shape),
                    k_descale=layer._k_scale.expand(descale_shape),
                    v_descale=layer._v_scale.expand(descale_shape),
                    num_splits=attn_metadata.max_num_splits,
                    s_aux=self.sinks,
                )
                return output

        # Cascade attention (rare case).
        # 级联注意力路径（较少使用）：多个请求共享前缀时的优化
        cascade_attention(
            output[:num_actual_tokens],
            query[:num_actual_tokens],
            key_cache,
            value_cache,
            cu_query_lens=attn_metadata.query_start_loc,
            max_query_len=attn_metadata.max_query_len,
            cu_prefix_query_lens=attn_metadata.cu_prefix_query_lens,
            prefix_kv_lens=attn_metadata.prefix_kv_lens,
            suffix_kv_lens=attn_metadata.suffix_kv_lens,
            max_kv_len=attn_metadata.max_seq_len,
            softmax_scale=self.scale,
            alibi_slopes=self.alibi_slopes,
            sliding_window=self.sliding_window,
            logits_soft_cap=self.logits_soft_cap,
            block_table=attn_metadata.block_table,
            common_prefix_len=attn_metadata.common_prefix_len,
            max_num_splits=attn_metadata.max_num_splits,
            fa_version=self.vllm_flash_attn_version,
            prefix_scheduler_metadata=attn_metadata.prefix_scheduler_metadata,
            suffix_scheduler_metadata=attn_metadata.scheduler_metadata,
            q_descale=layer._q_scale,
            k_descale=layer._k_scale,
            v_descale=layer._v_scale,
            s_aux=self.sinks,
        )
        return output
