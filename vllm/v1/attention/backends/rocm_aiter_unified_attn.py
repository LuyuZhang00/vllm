# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention layer with PagedAttention and Triton prefix prefill."""
# ROCm AITER 统一注意力后端模块。
#
# 本模块实现了基于 AITER unified_attention 的 ROCm 注意力后端。
# "统一"意味着 decode 和 prefill 使用同一个内核入口点，
# 通过不同的参数配置来处理不同的注意力模式。
#
# 关键特性：
# 1. 基于 RocmAttentionBackend/Impl 扩展，复用其元数据构建逻辑
# 2. 使用 aiter.ops.triton.unified_attention 作为核心内核
# 3. 支持所有注意力类型：DECODER、ENCODER、ENCODER_ONLY、ENCODER_DECODER
# 4. 支持 FP8 量化查询输入
# 5. 支持 RoPE 与 KV 缓存更新的融合操作
# 6. 支持滑动窗口注意力和 sink token
# 7. 不支持级联注意力（cascade attention）
#
# 与 rocm_aiter_fa.py 的区别：
# - rocm_aiter_fa.py: 使用 flash_attn_varlen_func + paged_attention 的组合
# - rocm_aiter_unified_attn.py: 使用 unified_attention 统一内核

import torch

from vllm import _custom_ops as ops
from vllm._aiter_ops import rocm_aiter_ops
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8StaticTensorSym,
)
from vllm.utils.torch_utils import is_quantized_kv_cache
from vllm.v1.attention.backend import AttentionLayer, AttentionType, MultipleOf
from vllm.v1.attention.backends.rocm_attn import (
    RocmAttentionBackend,
    RocmAttentionImpl,
    RocmAttentionMetadata,
    RocmAttentionMetadataBuilder,
)

logger = init_logger(__name__)


class RocmAiterUnifiedAttentionBackend(RocmAttentionBackend):
    """ROCm AITER 统一注意力后端。

    继承自 RocmAttentionBackend，使用 unified_attention 内核
    替代默认的 paged_attention 内核。
    """

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        """支持的块大小：16 的倍数。"""
        return [MultipleOf(16)]

    @classmethod
    def get_preferred_block_size(cls, default_block_size: int) -> int:
        """首选块大小为 64。"""
        return 64

    @classmethod
    def supports_block_size(cls, block_size: int | None) -> bool:
        """块大小必须是 16 的倍数。"""
        if block_size is None:
            return True
        return block_size % 16 == 0

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        """支持的头维度 >= 32。"""
        return head_size >= 32

    @classmethod
    def supports_mm_prefix(cls) -> bool:
        """支持多模态前缀。"""
        return True

    @classmethod
    def supports_sink(cls) -> bool:
        """支持 sink token（注意力汇聚 token）。"""
        return True

    @classmethod
    def supports_non_causal(cls) -> bool:
        """不支持非因果注意力。"""
        return False

    # KV 缓存更新不包含在 forward 中（独立操作）
    forward_includes_kv_cache_update: bool = False

    @staticmethod
    def get_name() -> str:
        return "ROCM_AITER_UNIFIED_ATTN"

    @staticmethod
    def get_impl_cls() -> type["RocmAiterUnifiedAttentionImpl"]:
        return RocmAiterUnifiedAttentionImpl

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        if block_size % 16 != 0:
            raise ValueError("Block size must be a multiple of 16.")
        return (num_blocks, 2, block_size, num_kv_heads, head_size)

    @staticmethod
    def use_cascade_attention(*args, **kwargs) -> bool:
        return False

    @staticmethod
    def get_builder_cls() -> type["RocmAttentionMetadataBuilder"]:
        return RocmAttentionMetadataBuilder

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        """RocmAiterUnifiedAttention supports all attention types.
        支持所有注意力类型：DECODER、ENCODER、ENCODER_ONLY、ENCODER_DECODER。
        """
        return attn_type in (
            AttentionType.DECODER,
            AttentionType.ENCODER,
            AttentionType.ENCODER_ONLY,
            AttentionType.ENCODER_DECODER,
        )


class RocmAiterUnifiedAttentionImpl(RocmAttentionImpl):
    """ROCm AITER 统一注意力实现。

    使用 aiter.ops.triton.unified_attention 内核执行注意力计算。
    继承自 RocmAttentionImpl，复用其大部分逻辑，替换核心注意力内核。
    """

    def fused_output_quant_supported(self, quant_key: QuantKey):
        """仅支持 FP8 静态张量对称量化作为融合输出量化。"""
        return quant_key == kFp8StaticTensorSym

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: int | None = None,
        sinks: torch.Tensor | None = None,
    ) -> None:
        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            alibi_slopes,
            sliding_window,
            kv_cache_dtype,
            logits_soft_cap,
            attn_type,
            kv_sharing_target_layer_name,
            sinks,
        )
        logger.info_once(
            "Using aiter unified attention for RocmAiterUnifiedAttentionImpl"
        )
        from aiter.ops.triton.unified_attention import unified_attention

        self.unified_attention = unified_attention
        self.supports_quant_query_input = True

    def _split_kv_cache(
        self, kv_cache: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """将组合 KV 缓存拆分为独立的 K 缓存和 V 缓存。

        对于非 ENCODER_DECODER 类型，直接沿维度 1 拆分。
        对于 ENCODER_DECODER 类型，使用 as_strided 重新解释内存布局，
        因为编码器-解码器层可能与 ROCM_ATTN 解码器层共享同一块物理内存。
        """
        if self.attn_type != AttentionType.ENCODER_DECODER:
            return kv_cache.unbind(1)

        # NOTE: Encoder-decoder layers can share the same raw KV allocation with
        # ROCM_ATTN decoder layers, whose physical layout is K/V first. Keep
        # this cross-attention path on that physical layout so block IDs do not
        # alias different bytes across the shared allocation.
        # 注意：编码器-解码器层可能与 ROCM_ATTN 解码器层共享同一块物理内存，
        # 其物理布局是 K/V 优先。保持交叉注意力路径在这种物理布局上，
        # 以避免块 ID 在共享内存中产生字节别名。
        num_blocks, _, block_size, num_kv_heads, head_size = kv_cache.shape
        block_stride = block_size * num_kv_heads * head_size
        kv_cache = kv_cache.as_strided(
            (2, num_blocks, block_size, num_kv_heads, head_size),
            (
                num_blocks * block_stride,
                block_stride,
                num_kv_heads * head_size,
                head_size,
                1,
            ),
        )
        return kv_cache.unbind(0)

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: RocmAttentionMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with FlashAttention.
        使用 AITER unified_attention 的前向传播。

        处理流程：
        1. 如果是编码器注意力，直接使用 Q/K/V 计算（无缓存）
        2. 否则，从 KV 缓存中获取 K/V，调用 unified_attention 内核

        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache: shape =
                [num_blocks, 2, block_size, num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        if output_block_scale is not None:
            raise NotImplementedError(
                "fused block_scale output quantization is not yet supported"
                " for RocmAttentionImpl"
            )

        if attn_metadata is None:
            # Profiling run.
            return output.fill_(0)

        assert attn_metadata.use_cascade is False

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
        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
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

        key_cache, value_cache = self._split_kv_cache(kv_cache)

        softmax_scale = self.scale
        if is_quantized_kv_cache(self.kv_cache_dtype):
            key_cache = key_cache.view(self.fp8_dtype)
            value_cache = value_cache.view(self.fp8_dtype)

        cu_seqlens_q = attn_metadata.query_start_loc
        seqused_k = attn_metadata.seq_lens
        max_seqlen_q = attn_metadata.max_query_len
        max_seqlen_k = attn_metadata.max_seq_len
        block_table = attn_metadata.block_table

        self.unified_attention(
            q=query[:num_actual_tokens],
            k=key_cache,
            v=value_cache,
            out=output[:num_actual_tokens],
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            seqused_k=seqused_k,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=softmax_scale,
            causal=True,
            alibi_slopes=self.alibi_slopes,
            window_size=self.sliding_window,
            block_table=block_table,
            softcap=self.logits_soft_cap,
            q_descale=layer._q_scale if query.dtype == self.fp8_dtype else None,
            k_descale=layer._k_scale,
            v_descale=layer._v_scale,
            sinks=self.sinks,
            output_scale=output_scale,
        )

        return output

    def do_kv_cache_update(
        self,
        layer: AttentionLayer,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ):
        """更新 KV 缓存。将新的 K/V 存入分页缓存。

        对于编码器注意力（ENCODER_ONLY、ENCODER），不需要更新缓存。
        对于解码器注意力，使用 reshape_and_cache_flash 将 K/V
        重塑并存入缓存的对应槽位。
        """
        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            # For encoder attention,
            # we use direct Q, K, V tensors without caching
            return
        key_cache, value_cache = self._split_kv_cache(kv_cache)

        # Reshape the input keys and values and store them in the cache.
        ops.reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
        )

    def fused_rope_kvcache_supported(self):
        """检查是否支持 RoPE 与 KV 缓存更新的融合操作。

        当 AITER 操作启用时支持融合，可以将 RoPE 位置编码计算
        与 KV 缓存更新合并为一个内核，减少内核启动开销。
        """
        return rocm_aiter_ops.is_enabled()

    def do_rope_and_kv_cache_update(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        is_neox: bool,
        kv_cache: torch.Tensor,
        layer_slot_mapping: torch.Tensor,
    ):
        """融合执行 RoPE 位置编码和 KV 缓存更新。

        将以下操作合并为单个内核调用：
        1. 对 Q 和 K 应用 RoPE（旋转位置编码）
        2. 将更新后的 K/V 存入分页 KV 缓存

        对于编码器注意力，跳过此操作。
        """
        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            # For encoder attention,
            # we use direct Q, K, V tensors without caching
            return
        key_cache, value_cache = self._split_kv_cache(kv_cache)
        flash_layout = True

        is_fp8_kv_cache = is_quantized_kv_cache(self.kv_cache_dtype)
        if is_fp8_kv_cache:
            key_cache = key_cache.view(self.fp8_dtype)
            value_cache = value_cache.view(self.fp8_dtype)

        rocm_aiter_ops.triton_rope_and_cache(
            query,
            key,
            value,
            positions,
            cos_sin_cache,
            is_neox,
            key_cache,
            value_cache,
            layer_slot_mapping,
            layer._k_scale,
            layer._v_scale,
            flash_layout,
            is_fp8_kv_cache,
        )
