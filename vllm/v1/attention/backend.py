# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# =============================================================================
# 中文说明：vLLM V1 Attention Backend 抽象接口模块
# =============================================================================
# 本文件定义了 vLLM V1 推理引擎中注意力（Attention）子系统的核心抽象接口。
# 整体架构采用"后端可插拔"设计：通过抽象基类定义统一接口，各具体后端
# （如 FlashAttention、FlashInfer、Triton 等）只需实现这些接口即可无缝接入。
#
# 模块中的核心类层次结构如下：
#
# 1. AttentionBackend（注意力后端抽象基类）
#    - 定义后端的元信息：名称、支持的数据类型、KV cache shape 等
#    - 提供能力查询接口（supports_head_size, supports_dtype 等）
#    - 工厂方法：get_impl_cls() 返回注意力计算实现类，
#      get_builder_cls() 返回元数据构建器类
#
# 2. AttentionMetadataBuilder（注意力元数据构建器抽象基类）
#    - 每个后端对应一个 Builder，负责将 SchedulerOutput 转换为
#      模型前向传播所需的注意力元数据（如 slot_mapping、block_table 等）
#    - build() 方法是核心，接收 CommonAttentionMetadata 并产出
#      后端特定的元数据对象
#    - 支持 CUDA Graph 捕获、投机解码等高级场景
#
# 3. CommonAttentionMetadata（通用注意力元数据数据类）
#    - 跨层、跨后端共享的 per-batch 元数据
#    - 包含 query_start_loc、seq_lens、block_table_tensor、slot_mapping 等
#    - 是 GPUModelRunner 构建后、传递给各层 Builder 的中间数据结构
#
# 4. AttentionImplBase / AttentionImpl / MLAAttentionImpl
#    - 注意力计算的实现基类，包含 forward() 方法
#    - AttentionImpl：标准注意力（如 FlashAttention）
#    - MLAAttentionImpl：Multi-head Latent Attention（如 DeepSeek 系列模型）
#    - SparseMLAAttentionImpl：稀疏 MLA，仅支持 decode 阶段
#
# 5. AttentionLayer（Protocol）
#    - 定义注意力层的 forward 接口协议，供类型检查使用
#
# 数据流：Scheduler -> SchedulerOutput -> GPUModelRunner
#   -> CommonAttentionMetadata -> AttentionMetadataBuilder.build()
#   -> AttentionMetadata -> AttentionImpl.forward()
# =============================================================================

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from enum import Enum
from typing import TYPE_CHECKING, Any, ClassVar, Generic, Protocol, TypeVar

import numpy as np
import torch
from typing_extensions import deprecated

from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8Dynamic64Sym,
    kFp8Dynamic128Sym,
    kFp8StaticTensorSym,
    kNvfp4Dynamic,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.config.cache import CacheDType
    from vllm.model_executor.layers.linear import ColumnParallelLinear
    from vllm.model_executor.layers.quantization.utils.quant_utils import QuantKey
    from vllm.platforms.interface import DeviceCapability
    from vllm.v1.attention.backends.utils import KVCacheLayoutType
    from vllm.v1.kv_cache_interface import AttentionSpec, KVQuantMode

from vllm.v1.kv_cache_interface import get_kv_quant_mode


# 中文注释：注意力类型枚举。
# 使用字符串枚举是为了与 torch.compile 兼容（torch.compile 不支持普通 Enum）。
# 不同的注意力类型对应不同的模型架构：
# - DECODER: 标准 decoder-only 模型（如 LLaMA、GPT）的自注意力
# - ENCODER: encoder-decoder 模型中 encoder 侧的自注意力
# - ENCODER_ONLY: 纯 encoder 模型（如 BERT）的自注意力
# - ENCODER_DECODER: encoder-decoder 模型中 decoder 对 encoder 的交叉注意力
class AttentionType(str, Enum):
    """
    Attention type.
    Use string to be compatible with `torch.compile`.
    """

    DECODER = "decoder"
    """Decoder attention between previous layer Q/K/V."""
    ENCODER = "encoder"
    """Encoder attention between previous layer Q/K/V for encoder-decoder."""
    ENCODER_ONLY = "encoder_only"
    """Encoder attention between previous layer Q/K/V."""
    ENCODER_DECODER = "encoder_decoder"
    """Attention between dec. Q and enc. K/V for encoder-decoder."""


class MultipleOf:
    """中文注释：块大小倍数约束。
    用于表示 attention kernel 要求的 block size 必须是某个基数的整数倍。
    例如 MultipleOf(16) 表示 block size 必须是 16 的倍数。
    在 supports_block_size() 中用于检查用户配置的 block size 是否满足 kernel 要求。
    """
    base: int

    def __init__(self, base: int):
        self.base = base


class AttentionBackend(ABC):
    """Abstract class for attention backends."""

    # 中文注释：注意力后端抽象基类。
    # 这是 vLLM V1 中"后端可插拔"注意力架构的核心抽象。
    # 每种具体的注意力后端（如 FlashAttention、FlashInfer、Triton 等）
    # 都需要继承此类并实现其抽象方法。
    #
    # 主要职责：
    # 1. 声明后端的元信息（名称、支持的数据类型、KV cache shape 等）
    # 2. 提供一系列能力查询接口（supports_head_size, supports_dtype 等），
    #    供 Scheduler 和 Config 在启动时校验后端兼容性
    # 3. 提供工厂方法 get_impl_cls() 和 get_builder_cls()，
    #    分别返回"注意力计算实现类"和"元数据构建器类"
    # 4. 定义 KV cache 的内存布局（shape、block dim、stride order 等）

    # 中文注释：后端支持的计算数据类型列表。
    # 大多数后端支持 float16 和 bfloat16，某些后端可能额外支持其他类型。
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    # 中文注释：后端支持的 KV cache 数据类型列表。
    # "auto" 表示跟随模型权重的数据类型，也可显式指定为 float16/bfloat16。
    supported_kv_cache_dtypes: ClassVar[list["CacheDType"]] = [
        "auto",
        "float16",
        "bfloat16",
    ]

    # 中文注释：标记该后端的 forward() 方法是否包含 KV cache 写入操作。
    # 如果为 True，表示 forward 内部会自动将新的 K/V 写入 KV cache；
    # 如果为 False，则需要外部（如 GPUModelRunner）单独调用 KV cache 更新。
    # Does attention's forward() include kv cache update?
    forward_includes_kv_cache_update: bool = True

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        """中文注释：返回该后端 attention kernel 支持的 block size 列表。
        列表中可以是整数（固定大小）或 MultipleOf 对象（倍数约束）。
        默认值 MultipleOf(1) 表示对 block size 无特殊要求。
        """
        return [MultipleOf(1)]

    @staticmethod
    @abstractmethod
    def get_name() -> str:
        """中文注释：返回后端的唯一名称字符串，如 "FLASH_ATTN"、"FLASHINFER" 等。
        用于日志、配置识别和调试信息中标识当前使用的后端。
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_impl_cls() -> type["AttentionImplBase"]:
        """中文注释：返回该后端对应的注意力计算实现类（AttentionImpl 的子类）。
        GPUModelRunner 在初始化各注意力层时会调用此方法获取实现类并实例化。
        实现类包含真正的 forward() 方法，执行 Q*K^T*V 的注意力计算。
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_builder_cls():  # -> Type["AttentionMetadataBuilder"]:
        """中文注释：返回该后端对应的元数据构建器类（AttentionMetadataBuilder 的子类）。
        Builder 负责将 CommonAttentionMetadata 转换为后端特定的 AttentionMetadata，
        供 forward() 方法使用。每个 attention layer group 对应一个 Builder 实例。
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        """中文注释：返回 KV cache 张量的逻辑形状。
        不同后端对 KV cache 的内存布局要求不同（如 [2, num_blocks, block_size, num_kv_heads, head_size]），
        此方法返回对应的 shape 元组，用于在初始化时预分配 KV cache 显存。
        参数:
        - num_blocks: 预分配的物理 block 数量
        - block_size: 每个 block 中包含的 token 数
        - num_kv_heads: KV head 数量（GQA 场景下通常小于 Q head 数）
        - head_size: 每个 head 的维度
        - cache_dtype_str: KV cache 的数据类型字符串
        """
        raise NotImplementedError

    @classmethod
    def get_kv_cache_block_dim(
        cls,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> int:
        """Discover which tensor dim is the block index, since different
        backends lay out dims differently."""
        # 中文注释：自动发现 KV cache 张量中"block 索引"所在的维度位置。
        # 不同后端的 KV cache 内存布局不同，block 维度可能在不同位置。
        # 此方法通过在 get_kv_cache_shape 中注入一个特殊的哨兵值 _S，
        # 然后在返回的 shape 中查找该值的位置，从而确定 block 维度的索引。
        # 这一信息用于构建 slot_mapping 时确定写入位置。
        _S = 1234567
        shape = cls.get_kv_cache_shape(
            _S,
            block_size,
            num_kv_heads,
            head_size,
            cache_dtype_str=cache_dtype_str,
        )
        return shape.index(_S)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        """中文注释：获取 KV cache 各维度的物理内存排列顺序。
        逻辑 shape 和物理内存布局可能不同——某些后端为了 kernel 访问效率，
        会将某些维度调整到更优的位置。此方法返回一个维度排列元组，
        描述物理内存中各维度的实际顺序。

        例如：逻辑 shape 为 [2, num_blocks, block_size, num_heads, head_size]，
        如果返回 (1, 3, 0, 2, 4)，则物理布局为
        [num_blocks, num_heads, 2, block_size, head_size]。

        如果此方法未实现（抛出 NotImplementedError），则物理布局与逻辑 shape 一致。

        Args:
            include_num_layers_dimension: 如果为 True，表示在逻辑 shape 前面
                额外包含一个 num_layers 维度。

        Returns:
            一个整数元组，是 range(len(shape)) 的一个排列。
        """
        raise NotImplementedError

    @classmethod
    def full_cls_name(cls) -> tuple[str, str]:
        """中文注释：返回后端类的完整限定名（模块路径 + 类名），用于序列化和日志。"""
        return (cls.__module__, cls.__qualname__)

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        """中文注释：返回该后端支持的 head_size 列表。
        空列表表示不设限制（即支持任意 head_size）。
        """
        return []

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        """中文注释：检查该后端是否支持指定的 head_size。
        如果 get_supported_head_sizes() 返回空列表，则默认支持所有 head_size。
        """
        supported_head_sizes = cls.get_supported_head_sizes()
        return (not supported_head_sizes) or head_size in supported_head_sizes

    @classmethod
    def supports_dtype(cls, dtype: torch.dtype) -> bool:
        """中文注释：检查该后端是否支持指定的计算数据类型（如 float16、bfloat16）。"""
        return dtype in cls.supported_dtypes

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: "CacheDType | None") -> bool:
        """中文注释：检查该后端是否支持指定的 KV cache 数据类型。
        None 表示未配置，此时默认支持。
        """
        if kv_cache_dtype is None:
            return True
        return (not cls.supported_kv_cache_dtypes) or (
            kv_cache_dtype in cls.supported_kv_cache_dtypes
        )

    @classmethod
    def supports_block_size(cls, block_size: int | None) -> bool:
        """中文注释：检查该后端是否支持指定的 block_size。
        支持两种约束方式：
        1. 固定大小：block_size 必须等于某个值
        2. MultipleOf：block_size 必须是某个基数的整数倍
        例如 block_size=32 对于 MultipleOf(16) 是合法的（32 % 16 == 0）。
        """
        if block_size is None:
            return True

        supported_kernel_block_sizes = cls.get_supported_kernel_block_sizes()
        if not supported_kernel_block_sizes:
            return True

        for supported_size in supported_kernel_block_sizes:
            if isinstance(supported_size, MultipleOf):
                supported_size = supported_size.base
            # With hybrid_blocks feature, the framework-level block size
            # only needs to be a multiple of the kernel's requirement,
            # even if the kernel requires a fixed block_size.
            if block_size % supported_size == 0:
                return True
        return False

    @classmethod
    def get_preferred_block_size(cls, default_block_size: int) -> int:
        """中文注释：获取该后端推荐的 block_size。
        如果默认 block_size 满足后端要求则直接使用；
        否则返回后端支持的最小 block_size。
        """
        supported_sizes = cls.get_supported_kernel_block_sizes()
        if not supported_sizes:
            return default_block_size

        if cls.supports_block_size(default_block_size):
            return default_block_size

        return min(s.base if isinstance(s, MultipleOf) else s for s in supported_sizes)

    @classmethod
    def is_mla(cls) -> bool:
        """中文注释：是否为 Multi-head Latent Attention（MLA）后端。
        MLA 是 DeepSeek 系列模型采用的注意力变体，通过低秩压缩 KV 来减少显存占用。
        """
        return False

    @classmethod
    def supports_sink(cls) -> bool:
        """中文注释：是否支持 Attention Sink（注意力汇聚）机制。
        Attention Sink 保留初始 token 的 KV cache，用于处理超长序列。
        """
        return False

    @classmethod
    def supports_alibi_sqrt(cls) -> bool:
        """中文注释：是否支持 ALiBi sqrt 位置编码变体。"""
        return False

    @classmethod
    def supports_mm_prefix(cls) -> bool:
        """中文注释：是否支持多模态（Multimodal）前缀的 partial full attention。
        某些视觉-语言模型需要对图像 token 使用全注意力而非因果注意力。
        """
        return False

    @classmethod
    def is_sparse(cls) -> bool:
        """中文注释：是否为稀疏注意力后端。
        稀疏注意力只计算部分 token 对之间的注意力分数，用于降低长序列的计算量。
        """
        return False

    @classmethod
    def supports_per_head_quant_scales(cls) -> bool:
        """中文注释：是否支持逐 head 的量化 scale。
        某些量化方案（如 FP8）可以为每个 attention head 使用独立的缩放因子。
        """
        return False

    @classmethod
    def supports_non_causal(cls) -> bool:
        """Check if backend supports non-causal (bidirectional) attention
        for decoder models.

        Unlike ENCODER_ONLY attention type which implies a different
        execution model, this refers to non-causal attention within the
        standard paged-KV-cache decoder path.
        """
        return False

    @classmethod
    def supports_batch_invariance(cls) -> bool:
        """中文注释：是否支持批不变性（batch invariance）。
        某些场景要求相同的请求在不同 batch 组合下产生完全相同的输出。
        """
        return False

    @classmethod
    def supports_kv_connector(cls) -> bool:
        """中文注释：是否支持 KV Connector。
        KV Connector 用于在分离式 prefill/decode 架构中传输 KV cache。
        """
        return True

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        """Check if backend supports a given attention type.

        By default, only supports decoder attention.
        Backends should override this to support other attention types.
        """
        return attn_type == AttentionType.DECODER

    @classmethod
    def supports_compute_capability(cls, capability: "DeviceCapability") -> bool:
        """中文注释：是否支持指定的 GPU 计算能力（如 SM 80、SM 90 等）。
        某些 kernel 只在特定架构的 GPU 上可用。
        """
        return True

    @classmethod
    def supports_combination(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: "CacheDType | None",
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        device_capability: "DeviceCapability",
    ) -> str | None:
        """中文注释：检查多个配置参数的组合是否被后端支持。
        返回 None 表示支持；返回非空字符串表示不支持的原因。
        子类可以覆写此方法来实现更复杂的组合校验逻辑。
        """
        return None

    @classmethod
    def validate_configuration(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: "CacheDType | None",
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        use_mm_prefix: bool,
        use_per_head_quant_scales: bool,
        device_capability: "DeviceCapability",
        attn_type: str,
        use_non_causal: bool = False,
        use_batch_invariant: bool = False,
        use_kv_connector: bool = False,
    ) -> list[str]:
        """中文注释：综合校验后端是否支持给定的完整配置。
        依次检查 head_size、dtype、kv_cache_dtype、block_size、MLA、sink、sparse、
        计算能力、注意力类型、batch invariance、KV connector 等各项约束。
        返回所有不满足条件的原因列表，空列表表示全部通过。
        此方法在 vLLM 启动阶段被调用，用于快速检测配置错误。
        """
        invalid_reasons = []
        if not cls.supports_head_size(head_size):
            invalid_reasons.append("head_size not supported")
        if not cls.supports_dtype(dtype):
            invalid_reasons.append("dtype not supported")
        if not cls.supports_kv_cache_dtype(kv_cache_dtype):
            invalid_reasons.append("kv_cache_dtype not supported")
        if not cls.supports_block_size(block_size):
            invalid_reasons.append("block_size not supported")
        if use_mm_prefix and not cls.supports_mm_prefix():
            invalid_reasons.append(
                "partial multimodal token full attention not supported"
            )
        if use_mla != cls.is_mla():
            if use_mla:
                invalid_reasons.append("MLA not supported")
            else:
                invalid_reasons.append("non-MLA not supported")
        if has_sink and not cls.supports_sink():
            invalid_reasons.append("attention sinks not supported")
        if use_sparse != cls.is_sparse():
            if use_sparse:
                invalid_reasons.append("sparse not supported")
            else:
                invalid_reasons.append("non-sparse not supported")
        if use_per_head_quant_scales and not cls.supports_per_head_quant_scales():
            invalid_reasons.append("per-head quant scales not supported")
        if not cls.supports_compute_capability(device_capability):
            invalid_reasons.append("compute capability not supported")
        if not cls.supports_attn_type(attn_type):
            invalid_reasons.append(f"attention type {attn_type} not supported")
        if use_non_causal and not cls.supports_non_causal():
            invalid_reasons.append("non-causal attention not supported")
        if use_batch_invariant and not cls.supports_batch_invariance():
            invalid_reasons.append("batch invariance not supported")
        if use_kv_connector and not cls.supports_kv_connector():
            invalid_reasons.append("KV connector not supported")
        combination_reason = cls.supports_combination(
            head_size,
            dtype,
            kv_cache_dtype,
            block_size,
            use_mla,
            has_sink,
            use_sparse,
            device_capability,
        )
        if combination_reason is not None:
            invalid_reasons.append(combination_reason)
        return invalid_reasons

    @classmethod
    def get_required_kv_cache_layout(cls) -> "KVCacheLayoutType | None":
        """中文注释：返回该后端要求的 KV cache 布局类型。
        某些后端（如 FlashInfer）要求 KV cache 使用特定的内存布局。
        返回 None 表示对布局没有特殊要求。
        """
        return None

    @classmethod
    def is_ssm(cls) -> bool:
        """中文注释：是否为状态空间模型（State Space Model）后端。
        SSM（如 Mamba）使用与标准 Transformer 注意力不同的计算范式。
        """
        return False


class AttentionMetadata:
    # 中文注释：注意力元数据的基类（空壳）。
    # 各具体后端的元数据（如 FlashAttentionMetadata、FlashInferMetadata 等）
    # 会继承此基类。AttentionImpl.forward() 的 attn_metadata 参数就是这个类型。
    # 此基类本身不包含字段，仅作为类型层次结构的根节点。
    pass


# 中文注释：泛型类型变量，绑定到 AttentionMetadata 的子类。
# 用于 AttentionMetadataBuilder 和 AttentionImplBase 的泛型参数，
# 使得 Builder 能产出特定后端的元数据类型，Impl 能接收对应类型的元数据。
T = TypeVar("T", bound=AttentionMetadata)


# 中文注释：通用注意力元数据数据类（CommonAttentionMetadata）。
# 这是 vLLM V1 注意力子系统的核心中间数据结构，由 GPUModelRunner 构建后
# 传递给各层的 AttentionMetadataBuilder，再由 Builder 转换为后端特定的元数据。
#
# 数据流：Scheduler -> SchedulerOutput -> GPUModelRunner
#   -> CommonAttentionMetadata -> AttentionMetadataBuilder.build()
#   -> 后端特定 AttentionMetadata -> AttentionImpl.forward()
#
# 此数据类包含"跨层共享"的 per-batch 元数据：
# - query_start_loc: 每个请求在 batch 中的 query 起始位置（用于区分不同请求的 token）
# - seq_lens: 每个请求的已计算上下文长度（含本轮新增 token）
# - block_table_tensor: 逻辑 block -> 物理 block 的映射表
# - slot_mapping: 每个 token 在 KV cache 中的写入位置索引
#
# 为减少 CPU-GPU 同步开销，很多张量同时维护 GPU 版和 CPU 版本。
@dataclass
class CommonAttentionMetadata:
    """
    Per-batch attention metadata, shared across layers and backends.
    AttentionMetadataBuilder instances use it to construct per-layer metadata.

    For many of the tensors we keep both GPU and CPU versions.
    """

    # 中文注释：每个请求在 query 张量中的起始位置（前缀和格式）。
    # 例如 batch 中有 3 个请求，query 长度分别为 [5, 3, 7]，
    # 则 query_start_loc = [0, 5, 8, 15]。
    # 这样可以通过 query_start_loc[i]:query_start_loc[i+1] 定位第 i 个请求的 query。
    query_start_loc: torch.Tensor
    query_start_loc_cpu: torch.Tensor
    """(batch_size + 1,), the start location of each request in query Tensor"""

    # 中文注释：每个请求的"上下文总长度"（= num_computed_tokens + num_new_tokens）。
    # 即该请求到本轮为止已有的 KV cache 长度加上本轮新增的 query 长度。
    # 注意：这不是 prompt 总长度，而是当前已处理到的上下文长度。
    seq_lens: torch.Tensor
    """(batch_size,), the number of computed tokens for each request"""

    # 中文注释：本轮 batch 中实际的请求数量。
    num_reqs: int
    """Number of requests"""
    # TODO(lucas): rename to num_tokens since it may be padded and this is misleading
    # 中文注释：本轮 batch 中实际的 token 总数（可能因 padding 而小于张量实际大小）。
    num_actual_tokens: int
    """Total number of tokens in batch"""
    # 中文注释：本轮 batch 中最长的 query 长度，用于 Builder 决定内部循环策略。
    max_query_len: int
    """Longest query in batch"""
    # 中文注释：本轮 batch 中最长的上下文长度（可能为上界值）。
    # 用于 Builder 分配临时 buffer 大小或选择 attention 策略。
    max_seq_len: int
    """Longest context length (may be an upper bound)"""

    # 中文注释：逻辑 block table 张量，形状 [batch_size, max_num_blocks_per_req]。
    # block_table_tensor[req_idx][logical_block_idx] = physical_block_idx，
    # 表示第 req_idx 个请求的第 logical_block_idx 个逻辑 block
    # 对应物理 KV cache 中的第 physical_block_idx 个 block。
    # 这是 PagedAttention 的核心数据结构——允许不同请求的 KV cache 在物理显存上不连续。
    block_table_tensor: torch.Tensor
    # 中文注释：slot mapping 张量，形状 [num_actual_tokens]。
    # slot_mapping[token_idx] 给出第 token_idx 个 token 在 KV cache 中的
    # 写入/读取的线性地址（physical_slot）。
    # 计算公式：physical_block_idx * block_size + offset_in_block。
    # GPUModelRunner 在构建此张量时会用 block_table_tensor 和 token 在 block 内的偏移量计算。
    slot_mapping: torch.Tensor

    # 中文注释：是否使用因果注意力掩码（causal attention mask）。
    # True 表示标准的 decoder-only 自注意力（每个 token 只能看到自身及之前的 token）。
    # False 用于某些特殊场景（如 encoder-only 模型的双向注意力、部分多模态前缀等）。
    causal: bool = True

    # 中文注释：FastPrefillAttentionBuilder 所需的字段。
    # logits_indices_padded: 经过 padding 后需要计算 logits 的 token 位置索引。
    # num_logits_indices: 实际需要计算 logits 的 token 数量（去除 padding 后）。
    # 这些字段用于 prefill 阶段优化——只对需要输出 logits 的位置（通常是最后一个 token）计算。
    # Needed by FastPrefillAttentionBuilder
    logits_indices_padded: torch.Tensor | None = None
    num_logits_indices: int | None = None

    # 中文注释：CrossAttentionBuilder 所需的字段，用于 encoder-decoder 架构的交叉注意力。
    # encoder_seq_lens: encoder 侧各序列的长度（GPU 张量）。
    # encoder_seq_lens_cpu: 同上的 CPU 版本（numpy 数组），避免 CPU-GPU 同步。
    # Needed by CrossAttentionBuilder
    encoder_seq_lens: torch.Tensor | None = None
    encoder_seq_lens_cpu: np.ndarray | None = None

    # 中文注释：Decode Context Parallelism（DCP）场景下本地 rank 的序列长度。
    # DCP 将长序列的 KV cache 分散到多个 GPU 上，每个 GPU 只持有部分序列。
    # dcp_local_seq_lens 记录的是当前 rank 负责的那段序列的长度。
    dcp_local_seq_lens: torch.Tensor | None = None
    dcp_local_seq_lens_cpu: torch.Tensor | None = None
    """Sequence lengths of the local rank in decode context parallelism world"""

    # 中文注释：每个 token 在序列中的位置索引（可选字段）。
    # 当调用方已持有 positions 时设置此字段，使 Builder 可以预计算位置相关的元数据。
    # 例如 DeepSeek V4 的 C128A 需要根据 token 位置计算 topk 索引。
    positions: torch.Tensor | None = None
    """(num_actual_tokens,) token positions.  Optional; set when the caller
    has positions available so that builders can pre-compute position-dependent
    metadata (e.g. C128A topk indices for DeepSeek V4)."""

    # 中文注释：标记每个请求是否仍在 prefill 阶段（布尔张量）。
    # True 表示该请求的已计算 token 数 < prompt 总 token 数（即还在做 prefill）。
    # 某些后端需要区分"真正的 decode"和"短 extend"（即 prefill 中的最后几个 token），
    # 以便使用不同的 kernel 策略。
    is_prefilling: torch.Tensor | None = None
    """(batch_size,) bool tensor: True if request is still in prefill phase
    (num_computed_tokens < num_prompt_tokens). Used by some backends to
    distinguish actual decodes from short extends."""

    # 中文注释：seq_lens 的 CPU 端上界值。
    # 对于 prefill 行和非异步投机解码的行，此值是精确的；
    # 对于异步投机解码的行，此值是乐观上界（假设所有 draft token 都被接受）。
    # 需要注意：对于需要精确逐行上下文长度的 kernel，此值不安全。
    seq_lens_cpu_upper_bound: torch.Tensor | None = None
    """(batch_size,) CPU upper bound on seq_lens. Precise for prefill rows
    and for all rows outside async spec decode; optimistic for async-spec
    decode rows (assumes every draft was accepted). Not safe for kernels
    that need exact per-row context lengths on decode rows."""

    # WARNING: Deprecated fields. Will be removed in a future release (v0.15.0)
    # 中文注释：以下为已废弃的字段，将在未来版本移除。
    # 建议直接使用设备端的 seq_lens 或通过 query_start_loc_cpu 推导。
    _seq_lens_cpu: torch.Tensor | None = None
    _num_computed_tokens_cpu: torch.Tensor | None = None

    # 中文注释：num_computed_tokens 的计算缓存（设备端）。
    # 避免重复计算 seq_lens - query_lens。
    _num_computed_tokens_cache: torch.Tensor | None = None

    def batch_size(self) -> int:
        """中文注释：返回当前 batch 中的请求数量。"""
        return self.seq_lens.shape[0]

    def naive_query_lens(self) -> torch.Tensor:
        """Naive because it assumes that query ends where the next query starts."""
        # 中文注释：朴素地计算每个请求的 query 长度。
        # 通过相邻 query_start_loc 的差值得到。
        # "朴素"是因为它假设每个请求的 query 紧密排列（中间无空隙）。
        return self.query_start_loc[1:] - self.query_start_loc[:-1]

    def replace(self, **kwargs) -> "CommonAttentionMetadata":
        """中文注释：创建一个新的 CommonAttentionMetadata 副本，替换指定字段。
        使用 dataclasses.replace 实现浅拷贝+字段替换。
        """
        return replace(self, **kwargs)

    @property
    @deprecated(
        """
    Prefer using device seq_lens directly to avoid implicit H<>D sync.
    If a CPU copy is needed, use `seq_lens.cpu()` instead.
    Will be removed in a future release, please migrate as soon as possible.
    """
    )
    def seq_lens_cpu(self) -> torch.Tensor:
        if self._seq_lens_cpu is None:
            self._seq_lens_cpu = self.seq_lens.to("cpu")
        return self._seq_lens_cpu

    @property
    @deprecated(
        """
    Prefer using device seq_lens directly to avoid implicit H<>D sync which breaks full
    async scheduling. If a CPU copy is needed, it can be derived from 
    query_start_loc_cpu and seq_lens.
    Will be removed in a future release, please migrate as soon as possible.
    """
    )
    def num_computed_tokens_cpu(self) -> torch.Tensor:
        if self._num_computed_tokens_cpu is None:
            query_seq_lens = (
                self.query_start_loc_cpu[1:] - self.query_start_loc_cpu[:-1]
            )
            self._num_computed_tokens_cpu = self.seq_lens_cpu - query_seq_lens
        return self._num_computed_tokens_cpu

    def compute_num_computed_tokens(self) -> torch.Tensor:
        """Compute num_computed_tokens on device (seq_lens - query_lens)."""
        # 中文注释：在设备端计算每个请求"已计算的 token 数"。
        # num_computed_tokens = seq_lens - query_lens，即当前上下文长度减去本轮新增的 query 长度，
        # 得到上一轮结束时该请求已经有多少 token 的 KV cache 被计算过了。
        # 使用缓存避免重复计算。
        if self._num_computed_tokens_cache is None:
            query_lens = self.query_start_loc[1:] - self.query_start_loc[:-1]
            self._num_computed_tokens_cache = self.seq_lens - query_lens
        return self._num_computed_tokens_cache

    # TODO(lucas): remove once we have FULL-CG spec-decode support
    def unpadded(
        self, num_actual_tokens: int, num_actual_reqs: int
    ) -> "CommonAttentionMetadata":
        """中文注释：去除 padding，返回只包含实际请求和 token 的元数据子集。
        在 CUDA Graph 捕获时，batch 可能被 pad 到固定大小；
        此方法用于在 spec-decode 等场景下提取实际有效的部分。
        参数:
        - num_actual_tokens: 实际 token 数（去除 padding）
        - num_actual_reqs: 实际请求数（去除 padding）
        """
        maybe_slice_reqs = lambda x: x[:num_actual_reqs] if x is not None else None
        return CommonAttentionMetadata(
            query_start_loc=self.query_start_loc[: num_actual_reqs + 1],
            query_start_loc_cpu=self.query_start_loc_cpu[: num_actual_reqs + 1],
            seq_lens=self.seq_lens[:num_actual_reqs],
            _seq_lens_cpu=self._seq_lens_cpu[:num_actual_reqs]
            if self._seq_lens_cpu is not None
            else None,
            _num_computed_tokens_cpu=self._num_computed_tokens_cpu[:num_actual_reqs]
            if self._num_computed_tokens_cpu is not None
            else None,
            num_reqs=num_actual_reqs,
            num_actual_tokens=num_actual_tokens,
            max_query_len=self.max_query_len,
            max_seq_len=self.max_seq_len,
            block_table_tensor=self.block_table_tensor[:num_actual_reqs],
            slot_mapping=self.slot_mapping[:num_actual_tokens],
            causal=self.causal,
            logits_indices_padded=self.logits_indices_padded,
            num_logits_indices=self.num_logits_indices,
            encoder_seq_lens=maybe_slice_reqs(self.encoder_seq_lens),
            encoder_seq_lens_cpu=maybe_slice_reqs(self.encoder_seq_lens_cpu),
            dcp_local_seq_lens=maybe_slice_reqs(self.dcp_local_seq_lens),
            dcp_local_seq_lens_cpu=maybe_slice_reqs(self.dcp_local_seq_lens_cpu),
            is_prefilling=maybe_slice_reqs(self.is_prefilling),
        )


M = TypeVar("M")


# 中文注释：CUDA Graph 支持级别枚举。
# CUDA Graph 可以将一系列 GPU 操作预录制为图，避免 CPU 端的 kernel launch 开销，
# 从而显著提升 decode 阶段的吞吐量。
# 但 CUDA Graph 要求 batch 的结构（token 数、请求数等）在多次调用间保持一致，
# 因此不同后端对 CUDA Graph 的支持程度不同。
#
# 四个级别从高到低：
# - ALWAYS: 始终支持 CUDA Graph，包括 prefill+decode 混合 batch
# - UNIFORM_BATCH: 仅当 batch 中所有请求的 query 长度相同时支持（如投机解码场景）
# - UNIFORM_SINGLE_TOKEN_DECODE: 仅当 batch 中所有请求都是单 token decode 时支持
# - NEVER: 完全不支持 CUDA Graph
class AttentionCGSupport(Enum):
    """Constants for the cudagraph support of the attention backend
    Here we do not consider the cascade attention, as currently
    it is never cudagraph supported."""

    ALWAYS = 3
    """Cudagraph always supported; supports mixed-prefill-decode"""
    UNIFORM_BATCH = 2
    """Cudagraph supported for batches the only contain query lengths that are
    the same, this can be used for spec-decode
        i.e. "decodes" are 1 + num_speculative_tokens"""
    UNIFORM_SINGLE_TOKEN_DECODE = 1
    """Cudagraph supported for batches the only contain query_len==1 decodes"""
    NEVER = 0
    """NO cudagraph support"""


# 中文注释：注意力元数据构建器抽象基类（AttentionMetadataBuilder）。
# 每个注意力后端对应一个 Builder 子类，负责将通用的 CommonAttentionMetadata
# 转换为后端特定的 AttentionMetadata 对象，供 forward() 方法使用。
#
# 核心职责：
# 1. 将 SchedulerOutput -> GPUModelRunner 构建的 CommonAttentionMetadata
#    转换为后端所需的特定格式（如 FlashAttention 的 block_table、cu_seqlens 等）
# 2. 管理 CUDA Graph 捕获的元数据构建
# 3. 支持 batch 重排序（reorder_batch）以优化 kernel 效率
# 4. 支持投机解码（speculative decoding）的元数据构建
#
# 生命周期：每个 attention layer group 对应一个 Builder 实例，
# 在 GPUModelRunner 初始化时创建，在每轮 forward 前调用 build()。
class AttentionMetadataBuilder(ABC, Generic[M]):
    # Does this backend/builder support CUDA Graphs for attention (default: no).
    # Do not access directly. Call get_cudagraph_support() instead.
    # 中文注释：该 Builder 的 CUDA Graph 支持级别（类变量）。
    # 子类应覆写此变量以声明其 CUDA Graph 能力。
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.NEVER
    # Does this backend/builder reorder the batch?
    # If not, set this to None. Otherwise set it to the query
    # length that will be pulled into the front of the batch.
    # 中文注释：batch 重排序的 query 长度阈值。
    # 如果不为 None，则 query 长度 <= 此值的请求会被移到 batch 前部。
    # 这样可以将 decode 请求集中在一起，减少 padding，提升 kernel 效率。
    reorder_batch_threshold: int | None = None
    # Does this backend/builder support updating the block table in existing
    # metadata
    # 中文注释：是否支持在已构建的元数据中增量更新 block table。
    # 如果为 True，当多个 KV cache group 共享相同的元数据但 block table 不同时，
    # 可以复用元数据只更新 block table，避免重复构建。
    supports_update_block_table: bool = False

    @abstractmethod
    def __init__(
        self,
        kv_cache_spec: "AttentionSpec",
        layer_names: list[str],
        vllm_config: "VllmConfig",
        device: torch.device,
    ):
        # 中文注释：Builder 初始化时保存必要的配置引用。
        # kv_cache_spec: 该 Builder 管理的 KV cache 规格（如 block_size、num_heads 等）
        # layer_names: 该 Builder 对应的注意力层名称列表（一个 Builder 可以服务多个层）
        # vllm_config: 全局配置对象
        # device: 运行设备（如 cuda:0）
        self.kv_cache_spec = kv_cache_spec
        self.layer_names = layer_names
        self.vllm_config = vllm_config
        self.device = device

    @classmethod
    def get_cudagraph_support(
        cls: type["AttentionMetadataBuilder"],
        vllm_config: "VllmConfig",
        kv_cache_spec: "AttentionSpec",
    ) -> AttentionCGSupport:
        """Get the cudagraph support level of this builder class."""
        # 中文注释：获取该 Builder 的 CUDA Graph 支持级别。
        # 默认直接返回类变量 _cudagraph_support。
        # 子类可覆写此方法，根据 vllm_config 或 kv_cache_spec 动态决定支持级别。
        return cls._cudagraph_support

    def _init_reorder_batch_threshold(
        self,
        reorder_batch_threshold: int | None = 1,
        supports_spec_as_decode: bool = False,
        supports_dcp_with_varlen: bool = False,
    ) -> None:
        self.reorder_batch_threshold = reorder_batch_threshold
        if self.reorder_batch_threshold is not None and supports_spec_as_decode:
            # If the backend supports spec-as-decode kernels, then we can set
            # the reorder_batch_threshold based on the number of speculative
            # tokens from the config.
            speculative_config = self.vllm_config.speculative_config
            if (
                speculative_config is not None
                and speculative_config.num_speculative_tokens is not None
            ):
                max_num_queries_for_spec = (
                    1
                    + (2 if speculative_config.parallel_drafting else 1)
                    * speculative_config.num_speculative_tokens
                )
                self.reorder_batch_threshold = max(
                    self.reorder_batch_threshold,
                    max_num_queries_for_spec,
                )

        if (
            self.vllm_config.parallel_config.decode_context_parallel_size > 1
            and not supports_dcp_with_varlen
        ):
            self.reorder_batch_threshold = 1

    @abstractmethod
    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> M:
        """
        Central method that builds attention metadata.
        Some builders (MLA) require reorder_batch to be called prior to build.

        Args:
            common_prefix_len: The length of the common prefix of the batch.
            common_attn_metadata: The common attention metadata.
            fast_build: The meta-data will prioritize speed of building over
                then speed at execution. Can be used for spec-decode where the
                result of a build call may only be used for few layers/iters.
        """
        raise NotImplementedError

    def update_block_table(
        self,
        metadata: M,
        blk_table: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> M:
        # 中文注释：更新注意力元数据中的 block table。
        # 当存在多个 KV cache group 时（如 MLA 中 k_pe 和 kv_c 分属不同 group），
        # 它们共用相同的注意力元数据结构，只是 block table 不同。
        # 此方法允许复用已构建的元数据，仅替换 block table 和 slot mapping，
        # 避免重复构建几乎相同的元数据，从而提升性能。
        # 仅当 supports_update_block_table 为 True 时需要实现。
        """
        Update the block table for the attention metadata.
        Faster when theres multiple kv-cache groups that create virtually the
        same metadata but just with different block tables.

        Only needs to be implemented if supports_update_block_table is True.
        """
        raise NotImplementedError

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> M:
        # 中文注释：为 CUDA Graph 捕获构建注意力元数据。
        # CUDA Graph 捕获时，batch 结构和序列长度是固定的（dummy batch），
        # 因此 common_prefix_len 设为 0。
        # 子类若重写此方法，必须调用 self.build() 或 super() 以保证一致性。
        """
        Build attention metadata for CUDA graph capture. Uses build by default.
        Subclasses that override this method should call self.build or
        super().build_for_cudagraph_capture.
        """
        return self.build(
            common_prefix_len=0, common_attn_metadata=common_attn_metadata
        )

    def build_for_drafting(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        draft_index: int,
    ) -> M:
        # 中文注释：为投机解码（speculative decoding）的 draft 模型构建注意力元数据。
        # 投机解码中，draft 模型会生成多个候选 token（链式或树式），
        # draft_index 标识当前是第几个候选 token 的尝试。
        # fast_build=True 表示优先构建速度而非执行速度，
        # 因为投机解码的元数据通常只用于少量层或迭代，很快会被丢弃。
        """
        Build attention metadata for draft model. Uses build by default.

        Args:
            common_attn_metadata: The common attention metadata.
            draft_index: The index of the current draft operation.
                When speculating a chain of tokens, this index refers to the
                draft attempt for the i-th token.
                For tree-based attention, this index instead refers to the
                draft attempt for the i-th level in the tree of tokens.
        """
        return self.build(
            common_prefix_len=0,
            common_attn_metadata=common_attn_metadata,
            fast_build=True,
        )

    def use_cascade_attention(
        self,
        common_prefix_len: int,
        query_lens: np.ndarray,
        num_query_heads: int,
        num_kv_heads: int,
        use_alibi: bool,
        use_sliding_window: bool,
        use_local_attention: bool,
        num_sms: int,
        dcp_world_size: int,
    ) -> bool:
        # 中文注释：判断是否启用级联注意力（Cascade Attention）。
        # 级联注意力是一种优化技术：当 batch 中所有请求共享较长的公共前缀时，
        # 可以将公共前缀的 K/V 只计算一次并缓存，后续请求直接复用，
        # 从而大幅减少重复计算。默认返回 False，子类可根据硬件能力
        # 和 batch 特征来决定是否启用。
        return False


# 中文注释：注意力层协议（Protocol）。
# 使用 Python Protocol 机制定义注意力层的接口契约，供类型检查使用。
# 具体的注意力层类（如 model_executor/layers/attention/ 中定义的层）
# 只要实现了 forward() 方法和相应的 scale 属性，即可满足此协议。
# _q_scale / _k_scale / _v_scale：用于 FP8 量化的缩放因子（张量形式）。
# _q_scale_float / _k_scale_float / _v_scale_float：对应缩放因子的 float 标量形式。
# _prob_scale：softmax 概率的缩放因子，用于量化注意力输出。
class AttentionLayer(Protocol):
    _q_scale: torch.Tensor
    _k_scale: torch.Tensor
    _v_scale: torch.Tensor
    _q_scale_float: float
    _k_scale_float: float
    _v_scale_float: float
    _prob_scale: torch.Tensor

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor: ...


# 中文注释：注意力实现基类。
# 这是所有注意力计算实现的抽象基类，包含标准 AttentionImpl 和 MLAAttentionImpl。
# 本基类定义了所有实现共享的公共属性和初始化逻辑，但不定义 forward() 方法
# ——各子类根据自身计算模式定义各自的 forward 接口。
#
# 关键设计说明：
# - 使用 __new__ 方法进行初始化（而非 __init__），确保所有子类在实例化时
#   都会自动调用此初始化逻辑，无需显式调用 super().__init__()。
# - 支持两种上下文并行模式：
#   DCP (Decode Context Parallelism): decode 阶段的上下文并行
#   PCP (Prefill Context Parallelism): prefill 阶段的上下文并行
#   总并行度 = PCP 并行度 * DCP 并行度。
class AttentionImplBase(ABC, Generic[T]):
    """Base class for attention implementations.

    Contains common attributes and initialization logic shared by both
    standard AttentionImpl and MLAAttentionImpl. Does not define a forward
    method - subclasses define their own forward interfaces.
    """

    # Required attributes that all impls should have
    num_heads: int
    head_size: int
    scale: float

    # Whether the attention impl can return the softmax lse for decode.
    # Some features like decode context parallelism require the softmax lse.
    can_return_lse_for_decode: bool = False

    # Whether the attention impl supports Prefill Context Parallelism.
    supports_pcp: bool = False
    # Whether the attention impl(or ops) supports MTP
    # when cp_kv_cache_interleave_size > 1
    supports_mtp_with_cp_non_trivial_interleave_size: bool = False

    # some attention backends might not always want to return lse
    # even if they can return lse (for efficiency reasons)
    need_to_return_lse_for_decode: bool = False

    # Whether this attention implementation supports pre-quantized query input.
    # When True, the attention layer will quantize queries before passing them
    # to this backend, allowing torch.compile to fuse the quantization with
    # previous operations. This is typically supported when using FP8 KV cache
    # with compatible attention kernels (e.g., TRT-LLM).
    # Subclasses should set this in __init__.
    # TODO add support to more backends:
    # https://github.com/vllm-project/vllm/issues/25584
    supports_quant_query_input: bool = False

    # 中文注释：上下文并行（Context Parallelism）相关属性。
    # dcp_world_size / dcp_rank: Decode Context Parallelism 的总节点数和当前节点编号。
    # pcp_world_size / pcp_rank: Prefill Context Parallelism 的总节点数和当前节点编号。
    # total_cp_world_size / total_cp_rank: 全局上下文并行的总节点数和当前节点编号。
    dcp_world_size: int
    dcp_rank: int

    pcp_world_size: int
    pcp_rank: int

    total_cp_world_size: int
    total_cp_rank: int

    def __new__(cls, *args, **kwargs):
        # 中文注释：使用 __new__ 而非 __init__ 进行初始化。
        # 这样可以确保所有子类在实例化时都会自动执行此逻辑，
        # 即使子类没有显式调用 super().__init__()。
        # 这里初始化上下文并行（DCP/PCP）相关的 rank 信息。
        # use __new__ so that all subclasses will call this
        self = super().__new__(cls)
        try:
            from vllm.distributed.parallel_state import get_dcp_group

            self.dcp_world_size = get_dcp_group().world_size
            self.dcp_rank = get_dcp_group().rank_in_group
        except AssertionError:
            # DCP might not be initialized in testing
            self.dcp_world_size = 1
            self.dcp_rank = 0
        try:
            from vllm.distributed.parallel_state import get_pcp_group

            self.pcp_world_size = get_pcp_group().world_size
            self.pcp_rank = get_pcp_group().rank_in_group
        except AssertionError:
            self.pcp_world_size = 1
            self.pcp_rank = 0
        # 中文注释：计算全局上下文并行参数。
        # total_cp_world_size = PCP 并行度 * DCP 并行度。
        # total_cp_rank 使用二维编号方案：PCP rank 为高维，DCP rank 为低维。
        self.total_cp_world_size = self.pcp_world_size * self.dcp_world_size
        self.total_cp_rank = self.pcp_rank * self.dcp_world_size + self.dcp_rank

        # 中文注释：当 DCP 多卡并行时，decode 阶段需要跨节点合并注意力结果，
        # 此时需要各节点返回 softmax lse（log-sum-exp）用于数值稳定的合并。
        self.need_to_return_lse_for_decode = (
            self.dcp_world_size > 1 and self.can_return_lse_for_decode
        )
        return self

    def process_weights_after_loading(self, act_dtype: torch.dtype):
        pass


# 中文注释：标准注意力实现类。
# 继承自 AttentionImplBase，定义了标准的 forward() 方法接口。
# 适用于传统的 Multi-Head Attention (MHA) / Multi-Query Attention (MQA) /
# Grouped-Query Attention (GQA) 架构，如 FlashAttention、FlashInfer 等后端。
#
# 关键接口：
# - forward(): 执行注意力前向计算，接收 Q/K/V、KV cache、注意力元数据，
#   输出注意力结果。
# - fused_output_quant_supported(): 是否支持融合输出量化。
# - fused_rope_kvcache_supported(): 是否支持 RoPE + KV cache 更新融合。
# - do_rope_and_kv_cache_update(): 执行融合的 RoPE 和 KV cache 更新操作。
class AttentionImpl(AttentionImplBase[T], Generic[T]):
    """Standard attention implementation with forward method."""

    # 中文注释：KV cache 的数据类型字符串表示。
    # "auto" 表示使用模型的默认精度，也可以是量化类型如 "fp8_e5m2" 等。
    kv_cache_dtype: str

    @property
    def kv_quant_mode(self) -> "KVQuantMode":
        # 中文注释：根据 kv_cache_dtype 字符串返回对应的 KV 量化模式枚举值。
        # 用于 KV cache manager 和 attention backend 判断量化行为。
        """Return the KV cache quantization mode for this layer."""
        return get_kv_quant_mode(self.kv_cache_dtype)

    @abstractmethod
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: T,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # 中文注释：标准注意力前向计算的核心方法。
        # 参数说明：
        # - layer: 当前注意力层，用于获取量化 scale 等层参数
        # - query/key/value: 当前 token 的 Q/K/V 张量
        # - kv_cache: 物理 KV cache 缓冲区（分页管理）
        # - attn_metadata: 注意力元数据，包含 slot_mapping、block_table 等
        # - output: 输出张量（预先分配）
        # - output_scale / output_block_scale: 输出量化的缩放因子
        # 此方法由 Attention 层的 forward() 调用，是注意力计算的实际执行入口。
        raise NotImplementedError

    def fused_output_quant_supported(self, quant_key: "QuantKey"):
        # 中文注释：判断此注意力实现是否支持融合输出量化。
        # 当支持时，torch.compile 的 AttnFusionPass 会将量化操作
        # 融合到注意力计算中，避免额外的 kernel launch 开销。
        # 默认返回 False，需要显式量化的后端（如 FP8 attention）应重写此方法。
        """
        Does this attention implementation support fused output quantization.
        This is used by the AttnFusionPass to only fuse output quantization
        onto implementations that support it.

        :param quant_key: QuantKey object that describes the quantization op
        :return: is fusion supported for this type of quantization
        """
        return False

    def fused_rope_kvcache_supported(self):
        # 中文注释：判断此注意力实现是否支持 RoPE + KV cache 更新的融合操作。
        # 当支持时，RopeKVCacheFusionPass 会将旋转位置编码（RoPE）计算
        # 与 KV cache 的写入操作融合为一个 kernel，减少显存读写和 kernel 开销。
        """
        Does this attention implementation support RoPE+KVCache fusion.
        This is used by the RopeKVCacheFusionPass to only fuse the RoPE ops
        with the KV cache update for implementations that support it.
        """
        return False

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
        # 中文注释：执行融合的 RoPE + KV cache 更新操作。
        # 当 fused_rope_kvcache_supported() 返回 True 时，
        # 此方法会被 torch.ops.vllm.fused_rope_and_unified_kv_cache_update
        # 自动调用，就地（inplace）完成：
        #   1. 对 Q/K 应用旋转位置编码（RoPE）
        #   2. 将 K/V 写入 KV cache 的对应 slot
        # 这样避免了 RoPE 输出和 KV cache 写入之间的中间显存搬运。
        """
        If `fused_rope_kvcache_supported` returns True, this method will be called
        by torch.ops.vllm.fused_rope_and_unified_kv_cache_update
        to perform the inplace RoPE and KV cache update.
        """
        raise NotImplementedError


# 中文注释：Multi-head Latent Attention (MLA) 实现类。
# MLA 是 DeepSeek 系列模型采用的注意力架构，其核心思想是：
# 使用低秩联合压缩来减少 KV cache 的显存占用。
#
# MLA 的 KV cache 存储格式与标准 MHA 不同：
# - KV cache 中存储的是压缩后的 latent 向量 kv_c (维度 kv_lora_rank)
#   以及位置编码后的 k_pe (维度 qk_rope_head_dim)
# - 这两者拼接后存储在 kv_c_and_k_pe_cache 中，总维度 = kv_lora_rank + qk_rope_head_dim
# - 相比标准 MHA 的 K/V 存储 (num_kv_heads * head_size * 2)，MLA 大幅减少了 KV cache 大小
#
# MLA 区分两种计算模式：
# 1. forward_mha() —— prefill 阶段：标准 MHA 式计算，Q 和 K 都是完整的多头
# 2. forward_mqa() —— decode 阶段：MQA 式计算，Q 为当前 token，K/V 从 cache 中读取
#
# 关键 MLA 参数：
# - q_lora_rank: Q 的低秩压缩维度（如果为 None 则不做 Q 压缩）
# - kv_lora_rank: KV 的低秩压缩维度（核心参数，决定 KV cache 的 latent 维度）
# - qk_nope_head_dim: Q/K 中不含位置编码部分的头维度
# - qk_rope_head_dim: Q/K 中含旋转位置编码部分的头维度
# - qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
# - v_head_dim: V 的头维度（可以与 qk_head_dim 不同）
# - kv_b_proj: 将 latent 向量投影回完整 K/V 空间的线性层
class MLAAttentionImpl(AttentionImplBase[T], Generic[T]):
    """MLA attention implementation with forward_mqa and forward_mha methods."""

    @abstractmethod
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
        # MLA Specific Arguments
        q_lora_rank: int | None,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        qk_head_dim: int,
        v_head_dim: int,
        kv_b_proj: "ColumnParallelLinear",
        indexer: object | None = None,
        q_pad_num_heads: int | None = None,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def forward_mha(
        self,
        q: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: T,
        k_scale: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        # 中文注释：MLA 的 MHA 式 prefill 前向计算。
        # 在 prefill 阶段，所有 token 的 Q/K/V 都是完整计算的：
        # - q: 当前 batch 中所有 token 的 query 张量（已通过 Q 投影和低秩压缩）
        # - kv_c_normed: 当前 batch 的 KV latent 向量（归一化后）
        # - k_pe: 当前 batch 的位置编码 key 向量
        # - kv_c_and_k_pe_cache: KV cache 中存储的 [kv_c | k_pe] 拼接向量
        # 在 prefill 中，cache 主要用于写入；若存在 prefix cache 命中，
        # 则 cache 中已有的部分也需要参与注意力计算。
        """MHA-style prefill forward pass."""
        raise NotImplementedError

    @abstractmethod
    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: T,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # 中文注释：MLA 的 MQA 式 decode 前向计算。
        # 在 decode 阶段，每个请求只计算 1 个新 token 的 query：
        # - q: 当前 token 的 query，可能是单个张量或 (q_nope, q_rope) 的元组
        # - kv_c_and_k_pe_cache: 完整的 KV cache 缓冲区
        # - attn_metadata: 包含 slot_mapping 等，用于从 cache 中读取正确的 K/V
        # - layer: 注意力层引用，用于获取量化 scale 等
        # 返回值: (attention_output, softmax_lse)
        # softmax_lse 在 DCP 多卡并行时用于跨节点合并注意力结果。
        """MQA-style decode forward pass."""
        raise NotImplementedError

    def fused_output_quant_supported(self, quant_key: "QuantKey"):
        # 中文注释：判断 MLA 实现是否支持融合输出量化。
        # MLA 的量化在 forward_impl（公共代码）中手动完成，
        # 因此所有 MLA 后端默认支持以下量化类型：
        # - FP8 静态张量对称量化 (kFp8StaticTensorSym)
        # - NVFP4 动态量化 (kNvfp4Dynamic)
        # - FP8 动态 128 元素对称量化 (kFp8Dynamic128Sym)
        # - FP8 动态 64 元素对称量化 (kFp8Dynamic64Sym)
        """
        Does this attention implementation support fused output quantization.
        Since MLA quantization is done manually in forward_impl (common code),
        all MLA backends support it by default.
        """
        return quant_key in (
            kFp8StaticTensorSym,
            kNvfp4Dynamic,
            kFp8Dynamic128Sym,
            kFp8Dynamic64Sym,
        )

    def do_kv_cache_update(
        self,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        k_scale: torch.Tensor,
    ) -> None:
        # 中文注释：将 MLA 的 KV latent 和位置编码 key 写入 KV cache。
        # 执行流程：
        # 1. 若 KV cache 为空（size=0），直接返回
        # 2. 调用自定义 CUDA kernel ops.concat_and_cache_mla()：
        #    - 将 kv_c_normed（KV latent）和 k_pe（位置编码 key）在最后一维拼接
        #    - 按照 slot_mapping 指定的位置写入 kv_cache 的对应 slot
        #    - 支持可选的量化（通过 kv_cache_dtype 和 k_scale 控制）
        # 这样 KV cache 中每个 slot 存储的是 [kv_c | k_pe] 拼接后的向量，
        # decode 时可以直接从 cache 读取并解压回完整的 K/V 空间。
        if kv_cache.numel() == 0:
            return
        from vllm import _custom_ops as ops

        ops.concat_and_cache_mla(
            kv_c_normed,
            k_pe.squeeze(1),
            kv_cache,
            slot_mapping.flatten(),
            kv_cache_dtype=kv_cache_dtype,
            scale=k_scale,
        )


# 中文注释：稀疏 MLA（Sparse MLA）注意力实现类。
# 与完整 MLAAttentionImpl 不同，Sparse MLA 仅支持 decode 阶段（MQA 式计算），
# 不支持 prefill 阶段（MHA 式计算）。
#
# 使用场景：某些模型架构（如 DeepSeek 的部分变体）在 decode 阶段
# 使用稀疏注意力模式来进一步减少计算量，但 prefill 阶段仍使用标准注意力。
# 因此需要将 Sparse MLA 与标准 MLA 分开定义。
#
# 接口与 MLAAttentionImpl 基本相同，但去除了 forward_mha() 方法。
class SparseMLAAttentionImpl(AttentionImplBase[T], Generic[T]):
    """Sparse MLA attention implementation with only forward_mqa method.

    Sparse MLA implementations only support decode (MQA-style) attention.
    They do not support prefill (MHA-style) attention.
    """

    def fused_output_quant_supported(self, quant_key: "QuantKey"):
        # 中文注释：与 MLAAttentionImpl 相同，所有 MLA 后端默认支持融合输出量化。
        """
        Does this attention implementation support fused output quantization.
        Since MLA quantization is done manually in forward_impl (common code),
        all MLA backends support it by default.
        """
        return quant_key in (
            kFp8StaticTensorSym,
            kNvfp4Dynamic,
            kFp8Dynamic128Sym,
            kFp8Dynamic64Sym,
        )

    @abstractmethod
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
        # MLA Specific Arguments
        q_lora_rank: int | None,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        qk_head_dim: int,
        v_head_dim: int,
        kv_b_proj: "ColumnParallelLinear",
        indexer: object | None = None,
        q_pad_num_heads: int | None = None,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: T,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # 中文注释：稀疏 MLA 的 MQA 式 decode 前向计算。
        # 与 MLAAttentionImpl.forward_mqa() 接口一致。
        # 在 decode 阶段，稀疏注意力只会计算与当前 token 最相关的
        # 部分 KV cache 条目，而非全部，从而降低 decode 的计算开销。
        """MQA-style decode forward pass."""
        raise NotImplementedError

    def do_kv_cache_update(
        self,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        k_scale: torch.Tensor,
    ) -> None:
        # 中文注释：将 MLA 的 KV latent 和位置编码 key 写入 KV cache。
        # 逻辑与 MLAAttentionImpl.do_kv_cache_update() 完全一致：
        # 调用 ops.concat_and_cache_mla() 将 kv_c_normed 和 k_pe 拼接后
        # 按 slot_mapping 写入 kv_cache。
        if kv_cache.numel() == 0:
            return
        from vllm import _custom_ops as ops

        ops.concat_and_cache_mla(
            kv_c_normed,
            k_pe.squeeze(1),
            kv_cache,
            slot_mapping.flatten(),
            kv_cache_dtype=kv_cache_dtype,
            scale=k_scale,
        )


# 中文注释：动态创建注意力后端子类的工具函数。
# 该函数通过 Python 的 type() 动态创建一个新的类，继承自 attention_backend_cls，
# 并将 get_builder_cls() 方法重写为返回指定的 builder_cls。
#
# 典型使用场景：当同一注意力后端（如 FlashAttention）需要在不同场景下
# 使用不同的元数据构建器时（如正常的 Transformer 构建器 vs. CUDA Graph 构建器），
# 可以用此函数动态创建特化的后端子类，而无需修改原始后端类。
#
# 例如：subclass_attention_backend("CudaGraphFlashAttention", FlashAttentionBackend, CudaGraphBuilder)
# 会创建一个新的 CudaGraphFlashAttentionBackend 类，其 get_builder_cls() 返回 CudaGraphBuilder。
def subclass_attention_backend(
    name_prefix: str,
    attention_backend_cls: type[AttentionBackend],
    builder_cls: type[AttentionMetadataBuilder[M]],
) -> type[AttentionBackend]:
    """
    Return a new subclass where `get_builder_cls` returns `builder_cls`.
    """
    name: str = name_prefix + attention_backend_cls.__name__  # type: ignore

    return type(
        name, (attention_backend_cls,), {"get_builder_cls": lambda: builder_cls}
    )


# 中文注释：带属性覆盖的注意力后端子类创建工具函数。
# 与 subclass_attention_backend() 类似，但更通用：允许覆盖任意属性和方法，
# 而不仅仅是 get_builder_cls()。
#
# 参数 overrides 是一个字典，键为属性/方法名，值为新的实现。
# 通过 type() 动态创建继承自 attention_backend_cls 的新类，
# 并将 overrides 中的键值对作为类属性注入。
#
# 典型使用场景：需要同时覆盖多个方法（如 builder、impl、capability 查询等）
# 时使用此函数，比多次调用 subclass_attention_backend 更简洁。
def subclass_attention_backend_with_overrides(
    name_prefix: str,
    attention_backend_cls: type[AttentionBackend],
    overrides: dict[str, Any],
) -> type[AttentionBackend]:
    name: str = name_prefix + attention_backend_cls.__name__  # type: ignore
    return type(name, (attention_backend_cls,), overrides)
