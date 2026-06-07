# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Triton MLA 注意力后端模块。

本模块实现了基于 Triton 的 MLA（Multi-Latent Attention）注意力后端，
作为通用的 fallback 后端，适用于所有支持 Triton 的 GPU。

核心特性：
1. 使用 Triton 实现的 decode 注意力内核（decode_attention_fwd）
2. 支持 FP8 KV 缓存（内核内部反量化为 BF16）
3. 支持 batch invariance（通过 num_kv_splits=1）
4. 支持动态 KV split 数量调整
5. 可返回 LSE（LogSumExp）用于分布式归约

与其他 MLA 后端的区别：
- 通用性最强，支持所有 GPU
- 性能可能不如专用内核（如 FlashMLA、AITER）
- 使用 Triton 的 split-KV 算法进行注意力计算
"""

from typing import ClassVar

import torch

import vllm.envs as envs
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.mla_attention import (
    MLACommonBackend,
    MLACommonImpl,
    MLACommonMetadata,
    MLACommonMetadataBuilder,
)
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.triton_utils import triton
from vllm.utils.torch_utils import is_quantized_kv_cache
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    AttentionLayer,
    AttentionType,
    MultipleOf,
)
from vllm.v1.attention.ops.triton_decode_attention import decode_attention_fwd

logger = init_logger(__name__)


class TritonMLAMetadataBuilder(MLACommonMetadataBuilder[MLACommonMetadata]):
    """Triton MLA 的元数据构建器。使用默认的 MLACommonMetadataBuilder。"""
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH


class TritonMLABackend(MLACommonBackend):
    """
    Triton MLA 注意力后端。

    定义了后端的基本属性和支持的配置。
    通用性最强，支持所有 GPU。
    """
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
    ]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        """不限制 head size。"""
        return []

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        """支持的块大小：16 的倍数。"""
        return [MultipleOf(16)]

    @classmethod
    def supports_block_size(cls, block_size: int | None) -> bool:
        """检查块大小是否支持（必须是 16 的倍数）。"""
        if block_size is None:
            return True
        return block_size % 16 == 0

    @staticmethod
    def get_name() -> str:
        return "TRITON_MLA"

    @classmethod
    def supports_batch_invariance(cls) -> bool:
        """支持批量不变性。"""
        return True

    @staticmethod
    def get_impl_cls() -> type["TritonMLAImpl"]:
        return TritonMLAImpl

    @staticmethod
    def get_builder_cls() -> type["TritonMLAMetadataBuilder"]:
        return TritonMLAMetadataBuilder

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        """支持所有 GPU（通用 Triton 后端）。"""
        return True


class TritonMLAImpl(MLACommonImpl[MLACommonMetadata]):
    """
    Triton MLA 注意力的具体实现。

    使用 Triton 的 split-KV decode 注意力内核执行注意力计算。
    支持 FP8 KV 缓存（内核内部反量化为 BF16）。
    """
    can_return_lse_for_decode: bool = True

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        # MLA 特有参数
        **mla_args,
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
            **mla_args,
        )

        unsupported_features = [alibi_slopes, sliding_window, logits_soft_cap]
        if any(unsupported_features):
            raise NotImplementedError(
                "TritonMLAImpl does not support one of the following: "
                "alibi_slopes, sliding_window, logits_soft_cap"
            )

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "Encoder self-attention and "
                "encoder/decoder cross-attention "
                "are not implemented for "
                "TritonMLAImpl"
            )

        # 对于 FP8 KV 缓存，我们在 Triton 内核加载时反量化为 BF16。
        # 告诉公共层不要将 query 量化为 FP8 — 我们使用 BF16 query
        # 处理 FP8 KV 缓存（模式 1）。
        if is_quantized_kv_cache(self.kv_cache_dtype):
            self.supports_quant_query_input = False

        self._sm_count = current_platform.num_compute_units()

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: MLACommonMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Decode 注意力的前向传播。

        流程：
        1. 如果 q 是元组，拼接为完整 q
        2. 计算 KV split 数量（基于 SM 数量和序列长度）
        3. 分配 logits 和输出缓冲区
        4. 调用 decode_attention_fwd 计算注意力
        5. 返回输出和 LSE
        """
        assert kv_c_and_k_pe_cache.numel() > 0
        assert attn_metadata.decode is not None

        if type(q) is tuple:
            q = torch.cat(q, dim=-1)

        assert isinstance(q, torch.Tensor)
        B = q.shape[0]
        q_num_heads = q.shape[1]
        o = torch.zeros(
            B, q_num_heads, self.kv_lora_rank, dtype=q.dtype, device=q.device
        )
        lse = torch.zeros(B, q_num_heads, dtype=q.dtype, device=q.device)

        # 对于批量不变性，使用 1 个 split 以确保确定性归约
        if envs.VLLM_BATCH_INVARIANT:
            num_kv_splits = 1
        else:
            # 每个 split 的最小工作量（硬件相关）
            min_work_per_split = 512

            ideal_splits = max(1, attn_metadata.max_seq_len // min_work_per_split)

            # 使用 2 的幂避免过多的内核实例化
            ideal_splits = triton.next_power_of_2(ideal_splits)

            # 基于 SM 的最大分割数，乘以占用率乘数
            # 2-4x 允许每个 SM 多个 block 以隐藏延迟（硬件相关）
            occupancy_multiplier = 2
            max_splits = self._sm_count * occupancy_multiplier
            num_kv_splits = min(ideal_splits, max_splits)

        # TODO(lucas) Allocate ahead of time
        # TODO(lucas) 提前分配
        attn_logits = torch.empty(
            (
                B,
                q_num_heads,
                num_kv_splits,
                # NOTE: the +1 stores the LogSumExp (LSE) that the stage2
                # kernel uses to merge partial attention outputs across splits.
                # +1 存储 LogSumExp（LSE），stage2 内核使用它来合并跨 split 的部分注意力输出。
                self.kv_lora_rank + 1,
            ),
            dtype=torch.float32,
            device=q.device,
        )

        # 添加 head 维度 1
        kv_c_and_k_pe_cache = kv_c_and_k_pe_cache.unsqueeze(2)
        kv_c_cache = kv_c_and_k_pe_cache[..., : self.kv_lora_rank]
        PAGE_SIZE = kv_c_and_k_pe_cache.size(1)

        # 运行 MQA — 始终传递层缩放因子。当 KV 缓存为 BF16 时，
        # 内核的 `if dtype.is_fp8()` 检查是无操作。
        decode_attention_fwd(
            q,
            kv_c_and_k_pe_cache,
            kv_c_cache,
            o,
            lse,
            attn_metadata.decode.block_table,
            attn_metadata.decode.seq_lens,
            attn_logits,
            num_kv_splits,
            self.scale,
            PAGE_SIZE,
            k_scale=layer._k_scale,
            v_scale=layer._k_scale,
            is_mla=True,
        )

        return o, lse
