# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# =============================================================================
# KV Cache 接口定义模块
# =============================================================================
# 本模块定义了 vLLM v1 引擎中 KV 缓存的核心抽象接口。
# 主要职责包括：
#   1. 定义 KV 缓存的量化模式（KVQuantMode）及对应的辅助函数
#   2. 定义各类注意力机制的缓存规格说明（KVCacheSpec 及其子类），
#      用于描述不同注意力层如何存储和管理 KV 缓存
#   3. 定义缓存的分组和配置结构（KVCacheGroupSpec、KVCacheConfig），
#      用于组织和管理多层模型的缓存资源
#   4. 提供工具函数来判断缓存类型、滑动窗口大小等
#
# 类继承层次概览：
#   KVCacheSpec (基类)
#     +-- AttentionSpec (注意力缓存基类)
#     |     +-- FullAttentionSpec (全注意力)
#     |     |     +-- TQFullAttentionSpec (TQ 量化全注意力)
#     |     |     +-- MLAAttentionSpec (多头潜在注意力, DeepSeek 系列)
#     |     |     |     +-- HiddenStateCacheSpec (隐藏状态缓存标记)
#     |     |     +-- SinkFullAttentionSpec (带 sink 的全注意力)
#     |     +-- ChunkedLocalAttentionSpec (分块局部注意力)
#     |     +-- SlidingWindowSpec (滑动窗口注意力)
#     |     |     +-- SlidingWindowMLASpec (滑动窗口 + MLA)
#     |     +-- EncoderOnlyAttentionSpec (仅编码器注意力, 无需缓存)
#     |     +-- CrossAttentionSpec (交叉注意力, 编码器-解码器模型)
#     +-- MambaSpec (Mamba 状态空间模型)
#     +-- UniformTypeKVCacheSpecs (统一类型多层缓存规格)
# =============================================================================

from __future__ import annotations

import copy
from collections import Counter
from dataclasses import dataclass, fields, replace
from enum import Enum, IntEnum
from math import prod
from typing import TYPE_CHECKING

import torch
from typing_extensions import Self

from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv, round_up
from vllm.utils.torch_utils import get_dtype_size, nvfp4_kv_cache_full_dim
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# KV cache quantization mode
# KV 缓存量化模式
# ---------------------------------------------------------------------------
# 量化模式用于在不依赖字符串匹配的情况下，
# 通过枚举值快速分发量化逻辑到注意力后端和内核。


class KVQuantMode(IntEnum):
    """KV cache quantization mode.

    Used by attention backends and kernels to dispatch quantization logic
    without string matching on ``kv_cache_dtype``.
    """
    # KV 缓存量化模式枚举。
    # 注意力后端和内核使用此枚举来分发量化逻辑，
    # 避免对 kv_cache_dtype 字符串进行匹配。

    NONE = 0  # 不使用量化
    FP8_PER_TENSOR = 1  # per-tensor scales (current fp8 path)
    # FP8 按张量缩放：整个张量使用同一个缩放因子（当前 FP8 路径）
    INT8_PER_TOKEN_HEAD = 2  # per-token-head dynamic scales for int8
    # INT8 按 token-head 动态缩放：每个 token 的每个 head 有独立的缩放因子
    FP8_PER_TOKEN_HEAD = 3  # per-token-head dynamic scales for fp8
    # FP8 按 token-head 动态缩放：同上，但数据类型为 FP8
    NVFP4 = 4  # packed fp4 data + fp8 block scales
    # NVFP4 打包格式：FP4 数据 + FP8 块级缩放因子

    @property
    def is_per_token_head(self) -> bool:
        """True for any per-token-head quantization mode."""
        # 是否为按 token-head 动态缩放的量化模式
        return self in (
            KVQuantMode.INT8_PER_TOKEN_HEAD,
            KVQuantMode.FP8_PER_TOKEN_HEAD,
        )

    @property
    def is_nvfp4(self) -> bool:
        """True for NVFP4 packed quantization mode."""
        # 是否为 NVFP4 打包量化模式
        return self == KVQuantMode.NVFP4


def get_kv_quant_mode(kv_cache_dtype: str) -> KVQuantMode:
    """Map a ``kv_cache_dtype`` string to a :class:`KVQuantMode`."""
    # 将 kv_cache_dtype 字符串映射到 KVQuantMode 枚举值。
    # 这是量化模式的统一入口，避免各处重复的字符串判断。
    if kv_cache_dtype == "int8_per_token_head":
        return KVQuantMode.INT8_PER_TOKEN_HEAD
    if kv_cache_dtype == "fp8_per_token_head":
        return KVQuantMode.FP8_PER_TOKEN_HEAD
    if kv_cache_dtype == "nvfp4":
        return KVQuantMode.NVFP4
    if isinstance(kv_cache_dtype, str) and kv_cache_dtype.startswith("fp8"):
        return KVQuantMode.FP8_PER_TENSOR
    return KVQuantMode.NONE


def is_quantized_kv_cache(kv_cache_dtype: str) -> bool:
    # 判断 KV 缓存是否使用了量化。
    return get_kv_quant_mode(kv_cache_dtype) != KVQuantMode.NONE


def kv_cache_uses_per_token_head_scales(kv_cache_dtype: str) -> bool:
    """Return True if *kv_cache_dtype* needs per-token-head scales."""
    # 判断指定的 kv_cache_dtype 是否需要按 token-head 的缩放因子。
    return get_kv_quant_mode(kv_cache_dtype).is_per_token_head


class KVCacheSpecKind(str, Enum):
    # KV 缓存规格类型枚举，用于标识不同种类的注意力缓存规格。
    FULL_ATTENTION = "full_attention"  # 全注意力
    MLA_ATTENTION = "mla_attention"  # 多头潜在注意力 (Multi-head Latent Attention)
    SLIDING_WINDOW = "sliding_window"  # 滑动窗口注意力
    SLIDING_WINDOW_MLA = "sliding_window_mla"  # 滑动窗口 + MLA
    MAMBA = "mamba"  # Mamba 状态空间模型
    CHUNKED_LOCAL_ATTENTION = "chunked_local_attention"  # 分块局部注意力
    SINK_FULL_ATTENTION = "sink_full_attention"  # 带 sink token 的全注意力
    ENCODER_ONLY_ATTENTION = "encoder_only_attention"  # 仅编码器注意力
    CROSS_ATTENTION = "cross_attention"  # 交叉注意力（编码器-解码器）
    UNKNOWN = "unknown"  # 未知类型


@dataclass(frozen=True)
class KVCacheSpec:
    """
    A base class for specifying the KV cache format of one layer.
    """
    # KV 缓存规格基类。
    # 用于描述单个注意力层的 KV 缓存格式，包括块大小、页面大小等。
    # 所有具体的注意力缓存规格都继承自此类。
    # 使用 frozen=True 使其不可变，确保配置一旦创建就不会被意外修改。

    # number of tokens in a block
    # 每个缓存块中包含的 token 数量（块大小）
    block_size: int

    @property
    def page_size_bytes(self) -> int:
        """
        The size of a page with `block_size` tokens in bytes.

        Returns:
            The page size
        """
        # 计算一个包含 block_size 个 token 的页面所占的字节数。
        # 子类必须实现此方法来返回实际的页面大小。
        raise NotImplementedError

    @property
    def storage_block_size(self) -> int:
        # 存储块大小。默认等于 block_size，
        # 对于有压缩的规格（如 MLA）可能不同。
        return self.block_size

    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
        """
        The maximum possible memory usage of this KV cache in bytes.

        Returns:
            The KV cache size in bytes
        """
        # 计算此 KV 缓存规格可能使用的最大内存（字节）。
        # 子类根据各自的注意力类型来实现不同的计算逻辑。
        raise NotImplementedError

    def copy_with_new_block_size(self, block_size: int) -> Self:
        """
        Create a new KVCacheSpec from self but replacing the block size.
        """
        # 创建一个副本，但使用新的 block_size。
        # 用于在运行时调整块大小配置。
        return replace(self, block_size=block_size)

    @classmethod
    def merge(cls, specs: list[Self]) -> Self:
        """
        Merge a list of KVCacheSpec objects into a single KVCacheSpec object.
        """
        # 将同一 KV 缓存组中多个层的规格合并为一个。
        # 要求所有层的规格完全相同（同一组内的层共享相同的缓存块表）。
        assert all(spec == specs[0] for spec in specs[1:]), (
            "All layers in the same KV cache group must be the same."
        )
        return copy.deepcopy(specs[0])


@dataclass(frozen=True, kw_only=True)
class AttentionSpec(KVCacheSpec):
    # 注意力缓存规格基类。
    # 继承自 KVCacheSpec，添加了注意力层特有的属性：
    # KV head 数量、head 维度、数据类型、量化模式等。
    # 所有基于注意力机制的缓存规格都继承自此类。

    num_kv_heads: int  # KV 注意力头的数量
    head_size: int  # 每个注意力头的维度大小
    dtype: torch.dtype  # 缓存数据的 PyTorch 数据类型
    kv_quant_mode: KVQuantMode = KVQuantMode.NONE  # KV 缓存量化模式
    page_size_padded: int | None = None  # 填充后的页面大小（可选，用于对齐）

    @property
    def page_size_bytes(self) -> int:
        # 计算页面大小（字节）。
        # 流程：
        #   1. 先获取真实页面大小 real_page_size_bytes
        #   2. 如果使用 per-token-head 量化，需要额外预算缩放因子的内存
        #      （缩放因子存在单独的张量中，但内存从 KV 缓存分配中划分）
        #   3. 如果设置了 page_size_padded，确保不小于真实大小后返回
        real_page_size = self.real_page_size_bytes
        # Per-token-head scales are stored in separate tensors managed
        # by the attention backend, but the memory is carved from the
        # raw KV cache allocation so it must be budgeted here.
        if self.kv_quant_mode.is_per_token_head:
            real_page_size += (
                2 * self.block_size * self.num_kv_heads * get_dtype_size(torch.float32)
            )
        if self.page_size_padded is not None:
            assert self.page_size_padded >= real_page_size
            return self.page_size_padded
        return real_page_size

    @property
    def real_page_size_bytes(self) -> int:
        # 计算真实的页面大小（不含填充）。
        # NVFP4 格式使用特殊的打包布局：fp4 数据 + fp8 块级缩放因子。
        # 标准格式为：2 (K和V) * block_size * num_kv_heads * head_size * dtype_size
        if self.kv_quant_mode.is_nvfp4:
            # Packed layout: fp4 data + fp8 block scales per head.
            full_dim = nvfp4_kv_cache_full_dim(self.head_size)
            return (
                2
                * self.block_size
                * self.num_kv_heads
                * full_dim
                * get_dtype_size(self.dtype)
            )
        return (
            2
            * self.block_size
            * self.num_kv_heads
            * self.head_size
            * get_dtype_size(self.dtype)
        )


@dataclass(frozen=True, kw_only=True)
class FullAttentionSpec(AttentionSpec):
    """
    When hybrid allocator is disabled and the model contains both full
    attention layers and sliding window attention layers, sliding
    window attention are regarded as full attention in KV cache manager
    (blocks are allocated for all tokens), while computed as sliding window
    attention in model runner.
    In this case, we use FullAttentionSpec and record the sliding window size.
    """
    # 全注意力缓存规格。
    # 当混合分配器被禁用时，滑动窗口注意力层也会被视为全注意力层：
    #   - KV 缓存管理器：按全注意力分配所有 token 的缓存块
    #   - 模型运行时：仍按滑动窗口注意力进行计算
    # 此时使用 FullAttentionSpec 并记录滑动窗口大小。
    # 另外，head_size_v 允许 K 和 V 的 head 维度不同（如某些模型的 V 维度更大）。

    head_size_v: int = None  # type: ignore[assignment]
    # V 的 head 维度。默认等于 head_size（K 的 head 维度）。
    # 某些模型（如 Gemma2）的 K 和 V 维度不同。

    sliding_window: int | None = None
    """
    Default to None for not using sliding window attention.
    """
    # 滑动窗口大小。None 表示不使用滑动窗口注意力。
    attention_chunk_size: int | None = None
    # 注意力分块大小。用于分块局部注意力。

    def __post_init__(self):
        # 初始化后处理：如果 head_size_v 未设置，默认等于 head_size。
        if self.head_size_v is None:
            object.__setattr__(self, "head_size_v", self.head_size)

    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
        # 计算全注意力的最大内存使用量。
        # 公式：ceil(max_model_len / block_size) * page_size_bytes
        # 注意：对于上下文并行（DCP/PCP），每个 rank 只需存储部分 token。
        max_model_len = vllm_config.model_config.max_model_len
        dcp_world_size = vllm_config.parallel_config.decode_context_parallel_size
        pcp_world_size = vllm_config.parallel_config.prefill_context_parallel_size
        # Note(hc): each dcp rank only need save
        # (max_model_len//dcp_world_size) tokens locally.
        if dcp_world_size * pcp_world_size > 1:
            max_model_len = cdiv(max_model_len, dcp_world_size * pcp_world_size)
        return cdiv(max_model_len, self.block_size) * self.page_size_bytes

    @classmethod
    def merge_window_sizes(cls, window_sizes: set[int]) -> int | None:
        # 合并窗口大小集合。要求同一组内所有层的窗口大小相同。
        if len(window_sizes) == 0:
            return None
        elif len(window_sizes) == 1:
            return window_sizes.pop()
        else:
            raise ValueError(
                "All attention layers in the same KV cache group must have the "
                "same window size."
            )

    @classmethod
    def merge(cls, specs: list[Self]) -> Self:
        """
        Merge a list of FullAttentionSpec objects into a single
        FullAttentionSpec object.
        """
        # 合并多个 FullAttentionSpec 对象为一个。
        # 流程：
        #   1. 验证所有层都是 FullAttentionSpec
        #   2. 收集所有滑动窗口大小和分块大小，确保一致
        #   3. 合并为一个新的 FullAttentionSpec
        #   4. 验证合并后的规格与每个输入层的注意力规格完全一致
        assert all(isinstance(spec, FullAttentionSpec) for spec in specs), (
            "All attention layers in the same KV cache group must be FullAttentionSpec."
        )

        sliding_window = set(
            spec.sliding_window for spec in specs if spec.sliding_window is not None
        )
        attention_chunk_size = set(
            spec.attention_chunk_size
            for spec in specs
            if spec.attention_chunk_size is not None
        )
        assert not any(isinstance(spec, MLAAttentionSpec) for spec in specs), (
            "MLAAttentionSpec should be merged in MLAAttentionSpec.merge"
        )
        merged_spec = cls(
            block_size=specs[0].block_size,
            num_kv_heads=specs[0].num_kv_heads,
            head_size=specs[0].head_size,
            head_size_v=specs[0].head_size_v,
            dtype=specs[0].dtype,
            kv_quant_mode=specs[0].kv_quant_mode,
            page_size_padded=specs[0].page_size_padded,
            sliding_window=cls.merge_window_sizes(sliding_window),
            attention_chunk_size=cls.merge_window_sizes(attention_chunk_size),
        )
        for spec in specs:
            for f in fields(AttentionSpec):
                assert getattr(spec, f.name) == getattr(merged_spec, f.name), (
                    "All attention layers in the same KV cache group must have "
                    "the same attention spec."
                )
        assert (merged_spec.sliding_window is not None) + (
            merged_spec.attention_chunk_size is not None
        ) <= 1, (
            "Model with both sliding window layers and chunked local attention "
            "layers is not supported."
        )
        return merged_spec

    @property
    def real_page_size_bytes(self) -> int:
        # 计算真实页面大小（不含填充）。
        # NVFP4 格式：K 和 V 的维度可能不同，需要分别计算。
        # 标准格式：block_size * num_kv_heads * (head_size + head_size_v) * dtype_size
        if self.kv_quant_mode.is_nvfp4:
            # Packed layout per head: fp4 data + fp8 block scales.
            # fp4 data: head_size//2 bytes (2 fp4 values per byte)
            # fp8 block scale: head_size//16 bytes (1 scale per 16 elements)
            last_dim = nvfp4_kv_cache_full_dim(
                self.head_size
            ) + nvfp4_kv_cache_full_dim(self.head_size_v)
            return (
                self.block_size
                * self.num_kv_heads
                * last_dim
                * get_dtype_size(self.dtype)
            )
        return (
            self.block_size
            * self.num_kv_heads
            * (self.head_size + self.head_size_v)
            * get_dtype_size(self.dtype)
        )


def _apply_alignment_padding(spec: MLAAttentionSpec | SlidingWindowMLASpec):
    # 对 MLA 或 SlidingWindowMLA 规格应用对齐填充。
    # 如果设置了 alignment（对齐字节数），则将页面大小向上取整到对齐边界。
    # 这对于某些硬件或内存分配器的对齐要求是必要的。
    if spec.alignment is None:
        return
    actual_page_size = spec.real_page_size_bytes
    padded_page_size = round_up(actual_page_size, spec.alignment)
    if padded_page_size != actual_page_size:
        object.__setattr__(spec, "page_size_padded", padded_page_size)


@dataclass(frozen=True, kw_only=True)
class TQFullAttentionSpec(FullAttentionSpec):
    """FullAttentionSpec with TQ-aware page size.

    Python equivalent of the C++ TQ4FullAttentionSpec. Overrides
    real_page_size_bytes to use TQ slot bytes instead of the raw
    head_size * dtype formula.
    """
    # TQ（TensorQuantization）感知的全注意力缓存规格。
    # 是 C++ TQ4FullAttentionSpec 的 Python 等价物。
    # 重写了 real_page_size_bytes，使用 TQ slot 字节数来代替原始的
    # head_size * dtype 公式。tq_slot_size 由 TQ 量化后端提供。

    tq_slot_size: int = 0  # TQ 量化后每个 slot 的字节数

    @property
    def real_page_size_bytes(self) -> int:
        # 如果设置了 tq_slot_size，使用 TQ 公式：block_size * num_kv_heads * tq_slot_size
        # 否则回退到父类的标准计算方式。
        if self.tq_slot_size > 0:
            return self.block_size * self.num_kv_heads * self.tq_slot_size
        return super().real_page_size_bytes

    @classmethod
    def merge(cls, specs: list[Self]) -> Self:
        # 合并 TQ 规格，要求所有层的 tq_slot_size 相同。
        merged = super().merge(specs)
        assert all(s.tq_slot_size == specs[0].tq_slot_size for s in specs), (
            "All TQ layers in the same KV cache group must use the same tq_slot_size."
        )
        return replace(merged, tq_slot_size=specs[0].tq_slot_size)


@dataclass(frozen=True, kw_only=True)
class MLAAttentionSpec(FullAttentionSpec):
    # MLA (Multi-head Latent Attention) 多头潜在注意力缓存规格。
    # 继承自 FullAttentionSpec，主要针对 DeepSeek 系列模型。
    # MLA 通过将 KV 缓存压缩到低秩潜在空间来减少内存占用。
    # 关键特性：
    #   - cache_dtype_str: 指定缓存数据类型的字符串标识
    #   - alignment: 内存对齐字节数
    #   - compress_ratio: 压缩比率，用于减少存储量
    #   - model_version: 模型版本标识（如 "deepseek_v4"）

    # TODO(Lucas/Chen): less hacky way to do this
    cache_dtype_str: str | None = None  # 缓存数据类型字符串标识
    # DeepseekV4 only fields. Non-DeepseekV4 MLA models leave these at defaults.
    # DeepSeekV4 专用字段。非 DeepSeekV4 的 MLA 模型使用默认值。
    alignment: int | None = None  # Default to None for no padding.
    # 内存对齐字节数。None 表示不进行填充对齐。
    compress_ratio: int = 1  # Default to 1 for no compression.
    # 压缩比率。1 表示不压缩，>1 表示按比例压缩存储。
    model_version: str | None = None  # 模型版本标识

    def __post_init__(self):
        super().__post_init__()
        _apply_alignment_padding(self)

    @property
    def storage_block_size(self) -> int:
        # 存储块大小 = block_size / compress_ratio。
        # 压缩后的实际存储块大小可能小于逻辑块大小。
        return self.block_size // self.compress_ratio

    @property
    def real_page_size_bytes(self) -> int:
        # 计算 MLA 格式的真实页面大小。
        # 不同模型版本和缓存类型有不同的字节布局：
        #   - fp8_ds_mla + deepseek_v4: 448B NoPE + 128B RoPE + 8B fp8 scale = 584B/token
        #   - fp8_ds_mla 其他版本: 656 字节自定义布局
        #   - 标准格式: storage_block_size * num_kv_heads * head_size * dtype_size
        if self.cache_dtype_str == "fp8_ds_mla":
            if self.model_version == "deepseek_v4":
                # DeepseekV4: 448B NoPE + 128B RoPE + 8B fp8 scale = 584B per token.
                # head_size stays semantic (512); bytes are determined here.
                return self.storage_block_size * 584
            # V3.2 main MLA: 656-byte custom layout (kv_lora_rank=512 +
            # qk_rope_head_dim=64, head_size=576). See flashmla_sparse.py.
            return self.block_size * 656
        return (
            self.storage_block_size
            * self.num_kv_heads
            * self.head_size
            * get_dtype_size(self.dtype)
        )

    @classmethod
    def merge(cls, specs: list[Self]) -> Self:
        # 合并 MLA 规格。
        # 要求所有层的 cache_dtype_str、compress_ratio、model_version 完全一致。
        assert all(isinstance(spec, MLAAttentionSpec) for spec in specs), (
            "All attention layers in the same KV cache group must be MLAAttentionSpec."
        )
        cache_dtype_str_set = set(spec.cache_dtype_str for spec in specs)
        compress_ratio_set = set(spec.compress_ratio for spec in specs)
        model_version_set = set(spec.model_version for spec in specs)
        assert (
            len(cache_dtype_str_set) == 1
            and len(compress_ratio_set) == 1
            and len(model_version_set) == 1
        ), (
            "All attention layers in the same KV cache group must use the same "
            "quantization method, compress ratio, and model version."
        )
        return cls(
            block_size=specs[0].block_size,
            num_kv_heads=specs[0].num_kv_heads,
            head_size=specs[0].head_size,
            dtype=specs[0].dtype,
            kv_quant_mode=specs[0].kv_quant_mode,
            page_size_padded=specs[0].page_size_padded,
            cache_dtype_str=cache_dtype_str_set.pop(),
            compress_ratio=compress_ratio_set.pop(),
            model_version=model_version_set.pop(),
        )


@dataclass(frozen=True, kw_only=True)
class HiddenStateCacheSpec(MLAAttentionSpec):
    """Marker for hidden-state cache layers used by extract_hidden_states."""
    # 隐藏状态缓存规格标记。
    # 用于标记那些通过 extract_hidden_states 使用的隐藏状态缓存层。
    # 目前仅作为一个标记类，不添加额外逻辑。

    pass


@dataclass(frozen=True, kw_only=True)
class ChunkedLocalAttentionSpec(AttentionSpec):
    # 分块局部注意力缓存规格。
    # 注意力只在一个固定大小的 chunk 内计算，而非整个序列。
    # 适用于使用分块注意力机制的模型。

    attention_chunk_size: int  # 注意力分块大小（每个 chunk 的 token 数）

    def max_admission_blocks_per_request(
        self, max_num_batched_tokens: int, max_model_len: int
    ) -> int:
        """Per-request admission cap, in blocks.

        Single source of truth for both startup pool sizing
        (`max_memory_usage_bytes`) and the runtime admission gate, so requests
        admitted by startup can also be admitted at runtime.
        """
        # 每个请求的最大允许块数。
        # 此方法是启动池大小和运行时准入门控的唯一真实来源，
        # 确保启动时准入的请求在运行时也能被准入。
        # 计算逻辑：在分块预填充期间，最多同时缓存一个 chunk 窗口的 KV。
        # During chunked prefill, we hold KV for at most one chunk window.
        num_tokens = min(
            self.attention_chunk_size + max_num_batched_tokens, max_model_len
        )
        return cdiv(num_tokens, self.block_size)

    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
        # 计算分块局部注意力的最大内存使用量。
        # 公式：max_blocks * page_size_bytes
        max_model_len = vllm_config.model_config.max_model_len
        max_num_batched_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        max_blocks = self.max_admission_blocks_per_request(
            max_num_batched_tokens=max_num_batched_tokens, max_model_len=max_model_len
        )
        return max_blocks * self.page_size_bytes


@dataclass(frozen=True, kw_only=True)
class SlidingWindowSpec(AttentionSpec):
    # 滑动窗口注意力缓存规格。
    # 只保留最近 sliding_window 个 token 的 KV 缓存，
    # 旧的 token 会被逐出。适用于长序列但只需要关注近期上下文的场景。
    # 与 FullAttentionSpec 的区别在于内存使用量计算和运行时块管理。

    sliding_window: int  # 滑动窗口大小（保留的 token 数量）
    head_size_v: int = None  # type: ignore[assignment]
    # V 的 head 维度，默认等于 head_size。

    def __post_init__(self):
        # 初始化后处理：如果 head_size_v 未设置，默认等于 head_size。
        if self.head_size_v is None:
            object.__setattr__(self, "head_size_v", self.head_size)

    @property
    def real_page_size_bytes(self) -> int:
        # 计算真实页面大小。
        # NVFP4 格式使用特殊布局，标准格式与 FullAttentionSpec 相同。
        # Mirror ``FullAttentionSpec.real_page_size_bytes`` for NVFP4 KV cache.
        if self.kv_quant_mode.is_nvfp4:
            last_dim = nvfp4_kv_cache_full_dim(
                self.head_size
            ) + nvfp4_kv_cache_full_dim(self.head_size_v)
            return (
                self.block_size
                * self.num_kv_heads
                * last_dim
                * get_dtype_size(self.dtype)
            )
        return (
            self.block_size
            * self.num_kv_heads
            * (self.head_size + self.head_size_v)
            * get_dtype_size(self.dtype)
        )

    def max_admission_blocks_per_request(
        self, max_num_batched_tokens: int, max_model_len: int
    ) -> int:
        """Per-request admission cap, in blocks.

        Single source of truth for both startup pool sizing
        (`max_memory_usage_bytes`) and the runtime admission gate. Per-request
        real-held blocks plateau at this bound because
        `SlidingWindowManager.remove_skipped_blocks` runs from `allocate_slots`
        before each chunk's `get_num_blocks_to_allocate`.
        """
        # 每个请求的最大允许块数。
        # 在分块预填充期间，同时缓存：
        #   - 最近 sliding_window-1 个已计算 token 的 KV
        #   - 当前新调度的 token
        # 且总数不超过 max_model_len。
        # +1 是因为滑动窗口的起始位置可能不在块边界。
        # 例如：block_size=4, num_token=4 需要两个块 [XXCD][EF]
        # 来存储 6-token 窗口 [CDEF]。
        # During chunked prefill, we hold KV for the last `sliding_window-1`
        # computed tokens plus the newly scheduled tokens, and never more
        # than `max_model_len`.
        num_tokens = min(
            self.sliding_window - 1 + max_num_batched_tokens, max_model_len
        )
        # +1 because the sliding window may not start from the beginning of
        # the block. E.g. block size 4 and num_token 4 needs two blocks
        # [XXCD][EF] to store the 6-token window [CDEF].
        return cdiv(num_tokens, self.block_size) + 1

    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
        # 计算滑动窗口注意力的最大内存使用量。
        # 注意：不支持上下文并行（DCP）。
        assert vllm_config.parallel_config.decode_context_parallel_size == 1, (
            "DCP not support sliding window."
        )
        max_model_len = vllm_config.model_config.max_model_len
        max_num_batched_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        max_blocks = self.max_admission_blocks_per_request(
            max_num_batched_tokens=max_num_batched_tokens, max_model_len=max_model_len
        )
        return max_blocks * self.page_size_bytes


@dataclass(frozen=True, kw_only=True)
class SlidingWindowMLASpec(SlidingWindowSpec):
    """Sliding window attention with MLA cache format."""
    # 滑动窗口 + MLA 缓存规格。
    # 结合了滑动窗口注意力和 MLA 压缩特性。
    # 主要用于 DeepSeek 系列模型中同时使用滑动窗口和 MLA 的层。

    cache_dtype_str: str | None = None  # 缓存数据类型字符串标识
    # DeepseekV4-only: see MLAAttentionSpec.model_version.
    alignment: int | None = None  # Default to None for no padding.
    # 内存对齐字节数。
    compress_ratio: int = 1  # 压缩比率
    model_version: str | None = None  # 模型版本标识

    def __post_init__(self):
        _apply_alignment_padding(self)

    @property
    def storage_block_size(self) -> int:
        # 存储块大小 = block_size / compress_ratio
        return self.block_size // self.compress_ratio

    @property
    def real_page_size_bytes(self) -> int:
        # 计算真实页面大小。
        # DeepSeekV4: 584 字节/token（与 MLAAttentionSpec 相同的布局）。
        # 其他模型版本：标准 MLA 布局。
        if self.model_version == "deepseek_v4":
            # DeepseekV4: 448B NoPE + 128B RoPE + 8B fp8 scale = 584B per token.
            return self.storage_block_size * 584
        assert self.model_version is None, (
            f"Unsupported model version: {self.model_version}"
        )
        return (
            self.storage_block_size
            * self.num_kv_heads
            * self.head_size
            * get_dtype_size(self.dtype)
        )

    @classmethod
    def merge(cls, specs: list[Self]) -> Self:
        # 合并 SlidingWindowMLA 规格。
        # 要求所有层的 cache_dtype_str、compress_ratio、model_version、
        # sliding_window 完全一致。
        assert all(isinstance(spec, SlidingWindowMLASpec) for spec in specs), (
            "All attention layers in the same KV cache group must be "
            "SlidingWindowMLASpec."
        )
        cache_dtype_str_set = set(spec.cache_dtype_str for spec in specs)
        compress_ratio_set = set(spec.compress_ratio for spec in specs)
        model_version_set = set(spec.model_version for spec in specs)
        sliding_window_set = set(spec.sliding_window for spec in specs)
        assert (
            len(cache_dtype_str_set) == 1
            and len(compress_ratio_set) == 1
            and len(model_version_set) == 1
            and len(sliding_window_set) == 1
        ), (
            "All attention layers in the same KV cache group must use the same "
            "quantization method, compress ratio, model version and sliding "
            "window size."
        )
        return cls(
            block_size=specs[0].block_size,
            num_kv_heads=specs[0].num_kv_heads,
            head_size=specs[0].head_size,
            dtype=specs[0].dtype,
            page_size_padded=specs[0].page_size_padded,
            sliding_window=sliding_window_set.pop(),
            cache_dtype_str=cache_dtype_str_set.pop(),
            compress_ratio=compress_ratio_set.pop(),
            model_version=model_version_set.pop(),
        )


@dataclass(frozen=True)
class MambaSpec(KVCacheSpec):
    # Mamba 状态空间模型的缓存规格。
    # Mamba 不使用传统的注意力机制，而是使用状态空间模型 (SSM)。
    # 其"缓存"是 SSM 的隐状态，形状和数据类型由模型架构决定。
    # 与注意力缓存不同，Mamba 的缓存形状是任意的元组。

    shapes: tuple[tuple[int, ...], ...]  # 各缓存张量的形状
    dtypes: tuple[torch.dtype]  # 各缓存张量的数据类型
    page_size_padded: int | None = None  # 填充后的页面大小
    mamba_type: MambaAttentionBackendEnum = MambaAttentionBackendEnum.MAMBA2
    # Mamba 类型（如 Mamba1、Mamba2）
    mamba_cache_mode: str = "none"  # Mamba 缓存模式（"none"/"align"/"all"）
    num_speculative_blocks: int = 0  # 投机解码所需的额外块数

    @property
    def page_size_bytes(self) -> int:
        # 计算页面大小。
        # 页面大小 = 所有缓存张量的元素总数 * 对应数据类型的字节数之和。
        page_size = sum(
            prod(shape) * get_dtype_size(dtype)
            for (shape, dtype) in zip(self.shapes, self.dtypes)
        )
        if self.page_size_padded is not None:
            assert self.page_size_padded >= page_size
            return self.page_size_padded
        return page_size

    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
        # 计算 Mamba 缓存的最大内存使用量。
        # 三种模式：
        #   - "all": 为所有可能的 token 位置分配缓存
        #   - "align": 只分配 2 个块（当前块 + 下一个块）
        #   - 其他: 只分配 1 个块
        # 每种模式都额外加上投机解码所需的块数。
        if vllm_config.cache_config.mamba_cache_mode == "all":
            max_model_len = vllm_config.model_config.max_model_len
            return (
                cdiv(max_model_len, self.block_size) + self.num_speculative_blocks
            ) * self.page_size_bytes
        elif vllm_config.cache_config.mamba_cache_mode == "align":
            return self.page_size_bytes * (2 + self.num_speculative_blocks)
        else:
            return self.page_size_bytes * (1 + self.num_speculative_blocks)


@dataclass(frozen=True)
class EncoderOnlyAttentionSpec(AttentionSpec):
    # 仅编码器注意力缓存规格。
    # 用于仅包含编码器的模型（如 BERT），这些层不需要 KV 缓存，
    # 因为编码器的注意力是双向的，不需要自回归缓存。
    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
        # Encoder-only layers do not need KV cache
        return 0


@dataclass(frozen=True)
class CrossAttentionSpec(AttentionSpec):
    """
    KV cache spec for cross-attention layers in encoder-decoder models.
    """
    # 交叉注意力缓存规格。
    # 用于编码器-解码器模型（如 Whisper）中的交叉注意力层。
    # 交叉注意力需要缓存编码器的输出状态，
    # 其大小由编码器输入的最大长度决定。

    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
        # 计算交叉注意力的最大内存使用量。
        # 缓存大小取决于编码器输入的最大 token 数量。
        # For cross-attention, we need to cache encoder states
        # Get encoder length (e.g., 1500 for Whisper).
        max_encoder_len = vllm_config.scheduler_config.max_num_encoder_input_tokens
        return cdiv(max_encoder_len, self.block_size) * self.page_size_bytes


@dataclass(frozen=True)
class SinkFullAttentionSpec(FullAttentionSpec):
    # 带 Sink Token 的全注意力缓存规格。
    # Sink Attention 是一种优化技术：保留序列开头的少量 "sink" token 的 KV 缓存，
    # 即使在滑动窗口模式下也不会被逐出，以保持模型对全局上下文的感知能力。
    # sink_len 指定了需要保留的开头 token 数量。

    sink_len: int | None = None  # 需要保留的 sink token 数量

    @classmethod
    def merge(cls, specs: list[Self]) -> Self:
        """
        Merge a list of FullAttentionSpec objects into a single
        FullAttentionSpec object.
        """
        # 合并 SinkFullAttentionSpec。
        # 逻辑与 FullAttentionSpec.merge 基本相同，
        # 额外保留 sink_len 字段。
        assert all(isinstance(spec, FullAttentionSpec) for spec in specs), (
            "All attention layers in the same KV cache group must be FullAttentionSpec."
        )

        sliding_window = set(
            spec.sliding_window for spec in specs if spec.sliding_window is not None
        )
        attention_chunk_size = set(
            spec.attention_chunk_size
            for spec in specs
            if spec.attention_chunk_size is not None
        )
        assert not any(isinstance(spec, MLAAttentionSpec) for spec in specs), (
            "MLAAttentionSpec should be merged in MLAAttentionSpec.merge"
        )
        merged_spec = cls(
            block_size=specs[0].block_size,
            num_kv_heads=specs[0].num_kv_heads,
            head_size=specs[0].head_size,
            head_size_v=specs[0].head_size_v,
            sink_len=specs[0].sink_len,
            dtype=specs[0].dtype,
            kv_quant_mode=specs[0].kv_quant_mode,
            page_size_padded=specs[0].page_size_padded,
            sliding_window=cls.merge_window_sizes(sliding_window),
            attention_chunk_size=cls.merge_window_sizes(attention_chunk_size),
        )
        for spec in specs:
            for f in fields(AttentionSpec):
                assert getattr(spec, f.name) == getattr(merged_spec, f.name), (
                    "All attention layers in the same KV cache group must have "
                    "the same attention spec."
                )
        assert (merged_spec.sliding_window is not None) + (
            merged_spec.attention_chunk_size is not None
        ) <= 1, (
            "Model with both sliding window layers and chunked local attention "
            "layers is not supported."
        )
        return merged_spec


@dataclass(frozen=True)
class UniformTypeKVCacheSpecs(KVCacheSpec):
    """
    A KV cache spec for multiple layers with the same type of attention. Here,
    same types means always need the same number of token slots. For example,
    sliding window attentions with different window sizes are not the same type
    and should not be merged into one UniformTypeKVCacheSpecs.
    """
    # 统一类型多层 KV 缓存规格。
    # 用于表示多个注意力层共享相同类型的缓存格式。
    # "相同类型"意味着始终需要相同数量的 token 槽位。
    # 例如，不同窗口大小的滑动窗口注意力不是相同类型，不应合并。
    #
    # 核心设计思想：
    #   - 多个层可以共享同一组缓存块（block table）
    #   - 页面大小 = 各层页面大小之和（因为每层需要自己的存储空间）
    #   - 最大内存使用量由需求最大的层决定（按页数对齐）

    kv_cache_specs: dict[str, KVCacheSpec]
    # 层名到 KVCacheSpec 的映射，包含此组内所有层的缓存规格。

    @property
    def page_size_bytes(self) -> int:
        # 页面大小 = 所有层页面大小之和。
        # 因为同一组内的层共享缓存块，每个块需要容纳所有层的数据。
        return sum(spec.page_size_bytes for spec in self.kv_cache_specs.values())

    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
        # 最大内存使用量 = max(各层所需页数) * 统一页面大小。
        # 取各层所需页数的最大值，确保块数能满足所有层的需求。
        max_num_pages = max(
            cdiv(spec.max_memory_usage_bytes(vllm_config), spec.page_size_bytes)
            for spec in self.kv_cache_specs.values()
        )
        return max_num_pages * self.page_size_bytes

    @classmethod
    def is_uniform_type(cls, kv_cache_specs: dict[str, KVCacheSpec]) -> bool:
        """
        Whether all layers have the same type of KV cache spec.
        """
        # 判断所有层是否具有相同类型的 KV 缓存规格。
        # 判断逻辑：
        #   1. 首先检查所有层的 block_size 是否相同
        #   2. 然后检查所有层的规格类型是否兼容：
        #      - SlidingWindowMLASpec: 需要相同的 sliding_window
        #      - FullAttentionSpec: 只要类型相同即可
        #      - CrossAttentionSpec: 只要类型相同即可
        #      - SlidingWindowSpec: 需要相同的 sliding_window
        #      - ChunkedLocalAttentionSpec: 需要相同的 attention_chunk_size
        #      - MambaSpec: 需要相同的 num_speculative_blocks
        block_sizes = set(spec.block_size for spec in kv_cache_specs.values())
        if len(block_sizes) > 1:
            # Different block sizes, not uniform.
            return False
        one_spec = next(iter(kv_cache_specs.values()))
        # NOTE: Check subclasses before parent classes since isinstance()
        # returns True for subclasses.
        if isinstance(one_spec, SlidingWindowMLASpec):
            # SlidingWindowMLASpec is uniform if all specs are SlidingWindowMLASpec
            # with the same sliding_window size.
            return all(
                isinstance(spec, SlidingWindowMLASpec)
                and spec.sliding_window == one_spec.sliding_window
                for spec in kv_cache_specs.values()
            )
        elif isinstance(one_spec, FullAttentionSpec):
            return all(
                isinstance(spec, FullAttentionSpec) for spec in kv_cache_specs.values()
            )
        elif isinstance(one_spec, CrossAttentionSpec):
            return all(
                isinstance(spec, CrossAttentionSpec) for spec in kv_cache_specs.values()
            )
        elif isinstance(one_spec, SlidingWindowSpec):
            return all(
                isinstance(spec, SlidingWindowSpec)
                and spec.sliding_window == one_spec.sliding_window
                for spec in kv_cache_specs.values()
            )
        elif isinstance(one_spec, ChunkedLocalAttentionSpec):
            return all(
                isinstance(spec, ChunkedLocalAttentionSpec)
                and spec.attention_chunk_size == one_spec.attention_chunk_size
                for spec in kv_cache_specs.values()
            )
        elif isinstance(one_spec, MambaSpec):
            return all(
                isinstance(spec, MambaSpec)
                and spec.num_speculative_blocks == one_spec.num_speculative_blocks
                for spec in kv_cache_specs.values()
            )
        else:
            # NOTE(Chen): Please add new branches for new KV cache spec types.
            raise NotImplementedError(
                f"Unsupported KV cache spec type: {type(one_spec)}"
            )

    @classmethod
    def from_specs(cls, kv_cache_specs: dict[str, KVCacheSpec]) -> Self | None:
        """
        Return a SameTypeKVCacheSpecs object if all layers have the same type
        of KV cache spec. Return None if not.
        """
        # 工厂方法：如果所有层类型一致则创建 UniformTypeKVCacheSpecs，否则返回 None。
        if cls.is_uniform_type(kv_cache_specs):
            block_size = next(iter(kv_cache_specs.values())).block_size
            return cls(block_size=block_size, kv_cache_specs=kv_cache_specs)
        else:
            return None

    # NOTE: below util functions are only used by DeepseekV4 for now.
    # 以下工具函数目前仅用于 DeepSeekV4 模型。
    def get_page_sizes(self) -> list[int]:
        # 获取所有不同页面大小的集合（去重）。
        return list(set(spec.page_size_bytes for spec in self.kv_cache_specs.values()))

    def get_num_layer_tuples(self) -> int:
        # 获取出现次数最多的页面大小对应的层数。
        # 用于 DeepSeekV4 的层分组优化。
        return Counter(
            spec.page_size_bytes for spec in self.kv_cache_specs.values()
        ).most_common(1)[0][1]

    def max_memory_usage_pages(self, vllm_config: VllmConfig) -> int:
        # 获取各层所需页数的最大值。
        return max(
            cdiv(spec.max_memory_usage_bytes(vllm_config), spec.page_size_bytes)
            for spec in self.kv_cache_specs.values()
        )


def get_kv_cache_spec_kind(kv_cache_spec: KVCacheSpec) -> KVCacheSpecKind:
    # 获取 KV 缓存规格的类型枚举。
    # 使用 isinstance 进行类型判断，子类检查在父类之前，
    # 以确保特殊化的规格返回更精确的类型。
    # 对于 UniformTypeKVCacheSpecs，递归获取内部规格的类型。
    if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
        inner_kinds = {
            get_kv_cache_spec_kind(spec)
            for spec in kv_cache_spec.kv_cache_specs.values()
        }
        if len(inner_kinds) == 1:
            return next(iter(inner_kinds))
        return KVCacheSpecKind.UNKNOWN
    # Keep subclass checks before base classes so specialized specs keep their
    # more precise kind.
    if isinstance(kv_cache_spec, SlidingWindowMLASpec):
        return KVCacheSpecKind.SLIDING_WINDOW_MLA
    if isinstance(kv_cache_spec, MLAAttentionSpec):
        return KVCacheSpecKind.MLA_ATTENTION
    if isinstance(kv_cache_spec, SinkFullAttentionSpec):
        return KVCacheSpecKind.SINK_FULL_ATTENTION
    if isinstance(kv_cache_spec, FullAttentionSpec):
        return KVCacheSpecKind.FULL_ATTENTION
    if isinstance(kv_cache_spec, ChunkedLocalAttentionSpec):
        return KVCacheSpecKind.CHUNKED_LOCAL_ATTENTION
    if isinstance(kv_cache_spec, SlidingWindowSpec):
        return KVCacheSpecKind.SLIDING_WINDOW
    if isinstance(kv_cache_spec, MambaSpec):
        return KVCacheSpecKind.MAMBA
    if isinstance(kv_cache_spec, EncoderOnlyAttentionSpec):
        return KVCacheSpecKind.ENCODER_ONLY_ATTENTION
    if isinstance(kv_cache_spec, CrossAttentionSpec):
        return KVCacheSpecKind.CROSS_ATTENTION
    return KVCacheSpecKind.UNKNOWN


def get_kv_cache_spec_sliding_window(kv_cache_spec: KVCacheSpec) -> int | None:
    # 从缓存规格中提取滑动窗口大小。
    # 如果是 UniformTypeKVCacheSpecs，递归获取内部规格的窗口大小。
    # 仅当所有内部规格的窗口大小一致时才返回该值。
    if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
        inner_windows = {
            get_kv_cache_spec_sliding_window(spec)
            for spec in kv_cache_spec.kv_cache_specs.values()
        }
        return next(iter(inner_windows)) if len(inner_windows) == 1 else None
    if isinstance(kv_cache_spec, SlidingWindowSpec):
        return kv_cache_spec.sliding_window
    return None


@dataclass
class KVCacheTensor:
    """
    A class for specifying how the workers should initialize the KV cache.
    """
    # KV 缓存张量规格。
    # 描述 worker 进程应如何初始化 KV 缓存张量。
    # 包含张量大小和共享此张量的层名列表。
    # 多个层可以共享同一个缓存张量（如 MLA 中 K 和 V 共享存储）。

    size: int  # size of the KV cache tensor in bytes
    # KV 缓存张量的字节大小
    shared_by: list[str]  # layer names that share the same KV cache tensor
    # 共享此缓存张量的层名列表


@dataclass
class KVCacheGroupSpec:
    """
    Represents a group of model layers that share the same KV cache block table.
    These layers are regarded as one layer in the KV cache manager.
    """
    # KV 缓存组规格。
    # 表示共享同一缓存块表（block table）的一组模型层。
    # 这些层在 KV 缓存管理器中被视为一个逻辑层。
    #
    # 设计意义：
    #   - 同组内的层共享相同的缓存块分配
    #   - 减少块管理的复杂度和内存碎片
    #   - 典型场景：MLA 中多个注意力层共享潜在空间缓存

    # The names of model layers in this group
    # 此组中的模型层名称列表
    layer_names: list[str]
    # The KV cache spec of this manager layer
    # 此管理器层的 KV 缓存规格
    kv_cache_spec: KVCacheSpec
    # Whether this group contains EAGLE/MTP draft attention layers.
    # 此组是否包含 EAGLE/MTP 草稿注意力层（用于投机解码）
    is_eagle_group: bool = False


@dataclass
class KVCacheConfig:
    """
    The KV cache configuration of a model.
    """
    # 模型的 KV 缓存配置。
    # 这是整个 KV 缓存系统的顶层配置结构，
    # 包含总块数、缓存张量列表和缓存组列表。
    #
    # 整体架构：
    #   - num_blocks: 全局可用的缓存块总数（由内存预算决定）
    #   - kv_cache_tensors: 描述每个层如何初始化自己的缓存张量
    #   - kv_cache_groups: 将模型层分组，同组层共享块表

    num_blocks: int
    """The number of KV cache blocks"""
    # KV 缓存块的总数。由可用 GPU 内存和每块大小计算得出。
    kv_cache_tensors: list[KVCacheTensor]
    """How should model runner initialize the KV cache tensors for each layer"""
    # 模型运行时为每层初始化 KV 缓存张量的方式。
    kv_cache_groups: list[KVCacheGroupSpec]
    """
    The kv cache groups of the model.
    For models with only one type of attention, there is only one group that
    contains all layers.
    For models with multiple types of attention, there will be multiple groups,
    see `_get_kv_cache_config_uniform_page_size` for more details.
    """
    # 模型的 KV 缓存组列表。
    # 对于只有一种注意力类型的模型，只有一个组包含所有层。
    # 对于有多种注意力类型的模型，会有多个组。
    # 详见 `_get_kv_cache_config_uniform_page_size`。

    @property
    def has_mamba_layers(self) -> bool:
        # 判断模型是否包含 Mamba 层。
        return any(isinstance(g.kv_cache_spec, MambaSpec) for g in self.kv_cache_groups)

    @property
    def needs_kv_cache_zeroing(self) -> bool:
        # 判断是否需要将 KV 缓存初始化为零。
        # Mamba 层需要零初始化，因为其状态空间模型对初始值敏感。
        return self.has_mamba_layers
