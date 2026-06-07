# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
CPU 注意力后端（CPU Attention Backend）。

本模块实现了在 CPU 平台上运行的注意力计算后端。适用于没有 GPU 或需要在 CPU
上进行推理的场景，支持多种 CPU 指令集架构：
1. x86 架构：AMX（Advanced Matrix Extensions）和 AVX-512
2. ARM 架构：NEON
3. RISC-V 架构：RVV（RISC-V Vector Extension）
4. S390X 架构：VXE
5. PowerPC 架构：VSX

核心组件：
- CPUAttentionBackend：后端能力声明，包括支持的数据类型、头大小、KV 缓存布局等
- CPUAttentionMetadata：注意力元数据数据类，存储推理步骤所需的张量信息
- CPUAttentionMetadataBuilder：元数据构建器，从调度器输出构建 CPU 注意力元数据
- CPUAttentionBackendImpl：核心前向计算实现，调用 C++ 自定义 kernel 或 PyTorch SDPA

执行流程：
1. 每步推理前，调度器输出 CommonAttentionMetadata
2. CPUAttentionMetadataBuilder.build() 构建 CPU 特定的元数据，包括 ISA 选择、
   调度元数据等
3. CPUAttentionBackendImpl.forward() 执行前向计算：
   a. 编码器注意力：直接使用 SDPA（Scaled Dot-Product Attention）
   b. 解码器注意力：先更新 KV 缓存，再调用 cpu_attention_with_kv_cache kernel
   c. 混合批次：对 prefill 部分用 SDPA，对 decode 部分用自定义 kernel

ISA（指令集架构）选择逻辑：
  根据 head_size、block_size、数据类型和 CPU 能力自动选择最优的计算路径，
  以充分利用硬件向量/矩阵计算单元。
"""
import functools
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from vllm.config.cache import CacheDType

import torch

from vllm import _custom_ops as ops
from vllm import envs
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.platforms import CpuArchEnum, current_platform
from vllm.utils.torch_utils import is_quantized_kv_cache
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.utils import (
    KVCacheLayoutType,
    split_decodes_and_prefills,
)
from vllm.v1.kv_cache_interface import AttentionSpec, CrossAttentionSpec

logger = init_logger(__name__)

# 支持混合批次（mixed batch）的 CPU 架构列表。
# 在这些架构上，decode 和 prefill 请求可以在同一批次中混合处理，
# 不需要将 decode 重新排序到 prefill 前面。
_CPU_ARCH_PREFER_MIXED_BATCH = (
    CpuArchEnum.X86,
    CpuArchEnum.ARM,
    CpuArchEnum.S390X,
    CpuArchEnum.RISCV,
    CpuArchEnum.POWERPC,
)


class CPUAttentionBackend(AttentionBackend):
    """
    CPU 注意力后端类。

    声明 CPU 平台注意力的能力和支持范围，包括：
    - 支持的数据类型（fp16, bf16, fp32）
    - 支持的 KV 缓存数据类型（包括 FP8 量化）
    - 支持的头大小列表
    - 支持的注意力类型（解码器、编码器、编码器-解码器）
    - KV 缓存形状和布局（HND：[Head, blockN, blockD]）
    """

    # 支持的查询/键/值数据类型
    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ]
    # 支持的 KV 缓存数据类型，auto 表示与模型数据类型一致，
    # fp8/fp8_e4m3/fp8_e5m2 表示使用 FP8 量化以节省内存
    supported_kv_cache_dtypes: ClassVar[list["CacheDType"]] = [
        "auto",
        "fp8",
        "fp8_e4m3",
        "fp8_e5m2",
    ]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        """
        返回支持的 KV 缓存块大小。

        CPU 注意力 kernel 要求块大小为 16 的倍数。
        MultipleOf(16) 表示任何 16 的倍数都是合法的块大小。
        """
        return [MultipleOf(16)]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        """
        返回支持的注意力头大小列表。

        这些大小对应 C++ kernel 中已实现的特化版本。
        较大的头大小（如 512）用于某些特殊模型架构。
        """
        return [32, 64, 80, 96, 112, 128, 160, 192, 224, 256, 512]

    @staticmethod
    def get_name() -> str:
        """返回后端的唯一名称标识符，用于配置选择和日志。"""
        return "CPU_ATTN"

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        """CPU attention supports decoder,
        encoder-only and encoder-decoder attention."""
        return attn_type in (
            AttentionType.DECODER,
            AttentionType.ENCODER,
            AttentionType.ENCODER_ONLY,
            AttentionType.ENCODER_DECODER,
        )

    @staticmethod
    def get_impl_cls() -> type["CPUAttentionBackendImpl"]:
        """返回 CPU 注意力前向计算实现类。"""
        return CPUAttentionBackendImpl

    @staticmethod
    def get_builder_cls() -> type["CPUAttentionMetadataBuilder"]:
        """返回 CPU 注意力元数据构建器类。"""
        return CPUAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        """
        返回 KV 缓存的张量形状。

        CPU 后端使用 HND 布局：
        - 维度 0 (2): key 和 value 各一个
        - 维度 1 (num_blocks): 缓存块数量
        - 维度 2 (num_kv_heads): KV 头数量
        - 维度 3 (block_size): 每个块中的 token 数
        - 维度 4 (head_size): 每个头的维度
        """
        return 2, num_blocks, num_kv_heads, block_size, head_size

    @classmethod
    def get_required_kv_cache_layout(cls) -> "KVCacheLayoutType | None":
        """
        返回所需的 KV 缓存布局类型。

        CPU 后端强制使用 HND（Head-BlockN-BlockD）布局，
        这种布局更适合 CPU 的内存访问模式和 SIMD 计算。
        """
        return "HND"

    @staticmethod
    def use_cascade_attention(*args, **kwargs) -> bool:
        """
        CPU 后端不支持级联注意力（cascade attention）。

        级联注意力用于多个请求共享相同前缀时的优化，
        CPU 后端暂未实现此优化。
        """
        return False


@dataclass
class CPUAttentionMetadata:
    """
    CPU 注意力元数据数据类。

    存储单步推理所需的全部注意力相关元数据信息。

    属性说明：
    - isa: 使用的指令集架构标识（如 "amx", "neon", "rvv" 等）
    - num_actual_tokens: 实际有效 token 数（不含 padding）
    - max_query_len: 当前批次中最大的查询长度
    - query_start_loc: 每个请求的查询起始位置（累积和），shape=[num_reqs+1]
    - max_seq_len: 当前批次中最大的序列长度
    - seq_lens: 每个请求的总序列长度（含已计算的 token），shape=[num_reqs]
    - block_table: 块表，映射逻辑块到物理块，shape=[num_reqs, max_blocks]
    - slot_mapping: 槽映射，将 token 位置映射到 KV 缓存中的物理槽位
    - scheduler_metadata: 调度器预计算的元数据，用于优化 CPU kernel 执行
    - causal: 是否使用因果注意力（解码器为 True，编码器为 False）
    - use_sdpa_prefill: 是否对 prefill 部分使用 SDPA（某些 CPU 架构需要）
    - num_decode_tokens: decode 请求的 token 总数
    - sdpa_attn_masks: SDPA 使用的注意力掩码列表（ALiBi 或滑动窗口）
    - sdpa_start_loc: SDPA 处理的起始位置
    """
    isa: str
    num_actual_tokens: int  # Number of tokens excluding padding.
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    scheduler_metadata: torch.Tensor | None
    causal: bool = True

    # can be removed after deprecate sdpa
    # SDPA 相关字段，用于在不支持混合批次的架构上分离 prefill 和 decode
    use_sdpa_prefill: bool = False
    num_decode_tokens: int = 0
    sdpa_attn_masks: list[torch.Tensor | None] | None = None
    sdpa_start_loc: torch.Tensor | None = None


class CPUAttentionMetadataBuilder(AttentionMetadataBuilder[CPUAttentionMetadata]):
    """
    CPU 注意力元数据构建器。

    负责在每步推理前构建 CPU 特定的注意力元数据。主要工作包括：
    1. 根据 CPU 架构决定是否使用 SDPA 处理 prefill（混合批次 vs 分离批次）
    2. 选择最优的 ISA（指令集架构）计算路径
    3. 生成调度器元数据（scheduler_metadata），用于优化 CPU kernel 的并行策略
    4. 构建 SDPA 注意力掩码（如 ALiBi 偏置、滑动窗口掩码）

    混合批次策略说明：
    - x86/ARM/RISC-V/S390X/PowerPC 架构：支持混合批次，decode 和 prefill
      在同一批次中由同一个 kernel 处理
    - 其他架构：需要将 decode 重新排序到前面，prefill 用 SDPA 单独处理
    """

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

        # 是否使用 SDPA 处理 prefill（默认不使用）
        self.use_sdpa_prefill = False
        reorder_batch_threshold = None
        # 检查当前 CPU 架构是否支持混合批次
        if current_platform.get_cpu_architecture() not in _CPU_ARCH_PREFER_MIXED_BATCH:
            # 不支持混合批次的架构：将 decode 序列重新排序到 prefill 前面，
            # 然后对 prefill 使用 SDPA，对 decode 使用 cpu_attention_with_kv_cache
            # reorder_batch_threshold=1 表示 query 长度为 1 的视为 decode
            reorder_batch_threshold = 1
            self.use_sdpa_prefill = True

        # 初始化批次重排序阈值，False 表示不要求均匀批次（uniform batch）
        self._init_reorder_batch_threshold(reorder_batch_threshold, False)

        self.kv_cache_spec = kv_cache_spec
        self.vllm_config = vllm_config

        # 从配置中获取模型参数
        parallel_config = vllm_config.parallel_config
        self.num_kv_heads = vllm_config.model_config.get_num_kv_heads(parallel_config)
        self.num_heads = vllm_config.model_config.get_num_attention_heads(
            parallel_config
        )
        self.head_dim = kv_cache_spec.head_size
        self.dtype = vllm_config.model_config.dtype
        # 滑动窗口大小，-1 表示不使用滑动窗口
        self.window_size = getattr(kv_cache_spec, "sliding_window", -1)
        if self.window_size is None:
            self.window_size = -1
        self.block_size = vllm_config.cache_config.block_size
        kv_cache_dtype_str = vllm_config.cache_config.cache_dtype
        # 根据硬件能力选择最优的 ISA 计算路径
        self.isa = _get_attn_isa(
            self.dtype,
            self.block_size,
            self.head_dim,
            kv_cache_dtype_str,
        )
        # 判断是否为交叉注意力（encoder-decoder 架构中的 cross-attention）
        self.is_cross_attention = isinstance(kv_cache_spec, CrossAttentionSpec)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> CPUAttentionMetadata:
        """
        构建 CPU 注意力元数据。

        处理流程：
        1. 从通用元数据中提取基本信息
        2. 如果需要 SDPA prefill，将批次拆分为 decode 和 prefill 两部分
        3. 调用 C++ 操作获取调度器元数据
        4. 组装并返回 CPUAttentionMetadata

        参数：
            common_prefix_len: 共享前缀长度（用于级联注意力优化）
            common_attn_metadata: 调度器输出的通用注意力元数据
            fast_build: 是否使用快速构建模式
        """
        num_reqs = common_attn_metadata.num_reqs
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        max_query_len = common_attn_metadata.max_query_len
        max_seq_len = common_attn_metadata.max_seq_len
        query_start_loc = common_attn_metadata.query_start_loc
        seq_lens = common_attn_metadata.seq_lens
        block_table_tensor = common_attn_metadata.block_table_tensor
        slot_mapping = common_attn_metadata.slot_mapping
        # 交叉注意力不使用因果掩码
        causal = False if self.is_cross_attention else common_attn_metadata.causal

        sdpa_start_loc = query_start_loc
        num_decode_tokens = 0
        if self.use_sdpa_prefill and causal:
            # 需要将 decode 和 prefill 分离处理
            # 将 decode 请求重排序到批次前面，prefill 请求排在后面
            assert self.reorder_batch_threshold
            (num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens) = (
                split_decodes_and_prefills(
                    common_attn_metadata,
                    decode_threshold=self.reorder_batch_threshold,
                    require_uniform=True,
                )
            )
            # 更新为仅包含 decode 请求的元数据
            num_reqs = num_decodes
            # SDPA 处理 prefill 部分，调整起始位置
            sdpa_start_loc = sdpa_start_loc[num_decodes:] - num_decode_tokens
            seq_lens = seq_lens[:num_decodes]
            query_start_loc = query_start_loc[: num_decodes + 1]
            block_table_tensor = block_table_tensor[:num_decodes]

        # 调用 C++ 操作获取调度器元数据
        # 该元数据包含 kernel 执行时需要的并行策略和分块信息
        scheduler_metadata = ops.cpu_attn_get_scheduler_metadata(
            num_reqs=num_reqs,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            seq_lens=seq_lens,
            dtype=self.dtype,
            query_start_loc=query_start_loc,
            causal=causal,
            sliding_window_size=self.window_size,
            isa=self.isa,
            enable_kv_split=envs.VLLM_CPU_ATTN_SPLIT_KV,
        )

        # 组装最终的注意力元数据
        attn_metadata = CPUAttentionMetadata(
            isa=self.isa,
            num_actual_tokens=num_actual_tokens,
            max_query_len=max_query_len,
            query_start_loc=query_start_loc,
            max_seq_len=max_seq_len,
            seq_lens=seq_lens,
            block_table=block_table_tensor,
            slot_mapping=slot_mapping,
            scheduler_metadata=scheduler_metadata,
            causal=causal,
            use_sdpa_prefill=self.use_sdpa_prefill,
            num_decode_tokens=num_decode_tokens,
            sdpa_start_loc=sdpa_start_loc,
        )

        return attn_metadata


class CPUAttentionBackendImpl(AttentionImpl):
    """
    CPU 注意力后端的前向计算实现类。

    核心职责：
    1. 管理 KV 缓存的读写（reshape_and_cache）
    2. 根据注意力类型选择不同的计算路径：
       - 编码器注意力：直接使用 SDPA 计算，无需 KV 缓存
       - 解码器注意力：使用自定义 C++ kernel，配合 KV 缓存和块表
       - 交叉注意力：使用 KV 缓存，key/value 来自编码器输出
    3. 支持各种注意力变体：ALiBi 位置编码、滑动窗口、logits soft cap 等
    4. 支持 KV 缓存 FP8 量化以节省内存

    性能注意事项：
    - 此方法在 eager 模式 PyTorch 下执行（非 CUDA Graph 模式）
    - 需要尽量减少 Python 层面的 CPU 开销
    - view、slice 等操作虽然不涉及 GPU 计算，但 Python 层面的开销不可忽视
    """

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
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        sinks: torch.Tensor | None = None,
    ) -> None:
        """
        初始化 CPU 注意力实现。

        参数说明：
        - num_heads: 查询头数量
        - head_size: 每个头的维度
        - scale: 注意力缩放因子，通常为 1/sqrt(head_size)
        - num_kv_heads: KV 头数量（GQA 时可能少于 num_heads）
        - alibi_slopes: ALiBi 位置编码的斜率，None 表示不使用
        - sliding_window: 滑动窗口大小，None 表示不使用
        - kv_cache_dtype: KV 缓存数据类型字符串
        - logits_soft_cap: logits 软封顶值（用于 Gemma 等模型）
        - attn_type: 注意力类型（解码器/编码器/交叉注意力等）
        - kv_sharing_target_layer_name: KV 共享目标层名（用于 KV 共享优化）
        - sinks: 注意力汇聚（attention sink）张量
        """
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        # 编码器注意力不支持 logits soft cap，发出警告
        if logits_soft_cap is not None and attn_type in (
            AttentionType.ENCODER,
            AttentionType.ENCODER_ONLY,
        ):
            logger.warning_once(
                "CPU_ATTN does not support logits softcap for"
                " ENCODER and ENCODER_ONLY, outputs may be slightly off"
            )
        if logits_soft_cap is None:
            logits_soft_cap = 0
        self.logits_soft_cap = logits_soft_cap

        self.num_kv_heads = num_kv_heads
        # 将 ALiBi 斜率转换为 float32 张量
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        # 滑动窗口转换为 (left, right) 元组格式
        # -1 表示该方向无窗口限制
        if sliding_window is None:
            self.sliding_window = (-1, -1)
        elif attn_type == AttentionType.ENCODER_ONLY:
            # 编码器仅模式：双向滑动窗口
            self.sliding_window = (sliding_window - 1, sliding_window - 1)
        else:
            # 解码器模式：仅左侧（历史）窗口
            self.sliding_window = (sliding_window - 1, 0)
        self.kv_cache_dtype = kv_cache_dtype
        # 每个 KV 头服务的查询头数量（用于 GQA/MQA）
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        # 是否使用 FP8 量化 KV 缓存
        self.is_fp8_kv_cache = is_quantized_kv_cache(kv_cache_dtype)
        self.attn_type = attn_type

        # 注意力汇聚（Attention Sink）：保留初始几个 token 的 KV 缓存不被驱逐，
        # 以维持模型在长序列上的稳定性
        self.sinks = sinks
        if self.sinks is not None:
            assert self.sinks.shape[0] == num_heads, (
                "Sinks must have the same number of heads as the number of "
                "heads in the layer"
            )

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: CPUAttentionMetadata | None,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass for CPU attention backend.

        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache: shape =
                [2, num_blocks, num_kv_heads, block_size, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        # CPU 后端暂不支持融合输出量化
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not yet supported"
                " for CPUAttentionBackendImpl"
            )

        # For warming-up
        # 预热阶段（profiling run），元数据为 None，直接返回零输出
        if attn_metadata is None:
            return output

        num_actual_tokens = attn_metadata.num_actual_tokens

        # Handle encoder attention differently - no KV cache needed
        # 编码器注意力：直接对 Q、K、V 计算注意力，无需 KV 缓存
        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            # For encoder attention,
            return self._run_sdpa_forward(
                query[:num_actual_tokens],
                key[:num_actual_tokens],
                value[:num_actual_tokens],
                output[:num_actual_tokens],
                attn_metadata,
                self.attn_type,
            )

        # For decoder and cross-attention, use KV cache, size are
        # [num_blocks, num_kv_heads, block_size, head_size]
        # 解码器和交叉注意力：使用 KV 缓存
        # 将 KV 缓存沿第 0 维拆分为 key_cache 和 value_cache
        key_cache, value_cache = kv_cache.unbind(0)

        # key and value may be None in the case of cross attention. They are
        # calculated once based on the output from the encoder and then cached
        # in KV cache.
        # 交叉注意力中，key/value 可能为 None（已缓存在 KV cache 中）
        # 只有在非 KV 共享模式且有新的 key/value 时才更新缓存
        if (
            self.kv_sharing_target_layer_name is None
            and key is not None
            and value is not None
        ):
            # 将新的 key/value 写入 KV 缓存的对应槽位
            ops.cpu_attn_reshape_and_cache(
                key,
                value,
                key_cache,
                value_cache,
                attn_metadata.slot_mapping,
                attn_metadata.isa,
                k_scale=layer._k_scale_float,
                v_scale=layer._v_scale_float,
                kv_cache_dtype=self.kv_cache_dtype,
            )

        if attn_metadata.use_sdpa_prefill:
            # 不支持混合批次的架构：prefill 部分使用 SDPA 处理
            assert self.sinks is None, "Attention sink is unsupported in SDPA prefill"
            num_decode_tokens = attn_metadata.num_decode_tokens
            # prefill 请求排在 decode 后面，取 [num_decode_tokens:] 范围
            self._run_sdpa_forward(
                query[num_decode_tokens:num_actual_tokens],
                key[num_decode_tokens:num_actual_tokens],
                value[num_decode_tokens:num_actual_tokens],
                output[num_decode_tokens:num_actual_tokens],
                attn_metadata,
                self.attn_type,
            )
            # 更新实际 token 数为仅 decode 部分
            num_actual_tokens = num_decode_tokens

        if num_actual_tokens > 0:
            # decode 部分使用 C++ 自定义 kernel 计算注意力
            # 该 kernel 直接从 KV 缓存（分页存储）中读取，效率更高
            ops.cpu_attention_with_kv_cache(
                query=query[:num_actual_tokens],
                key_cache=key_cache,
                value_cache=value_cache,
                output=output[:num_actual_tokens],  # type: ignore
                query_start_loc=attn_metadata.query_start_loc,
                seq_lens=attn_metadata.seq_lens,
                scale=self.scale,
                causal=attn_metadata.causal,
                alibi_slopes=self.alibi_slopes,  # type: ignore
                sliding_window=self.sliding_window,
                block_table=attn_metadata.block_table,
                softcap=self.logits_soft_cap,
                scheduler_metadata=attn_metadata.scheduler_metadata,
                s_aux=self.sinks,
                k_scale=layer._k_scale_float,
                v_scale=layer._v_scale_float,
                kv_cache_dtype=self.kv_cache_dtype,
            )

        return output

    def _run_sdpa_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: CPUAttentionMetadata,
        attn_type: str,
    ) -> torch.Tensor:
        """
        使用 PyTorch 原生 SDPA（Scaled Dot-Product Attention）计算注意力。

        用于以下场景：
        1. 编码器注意力（无需 KV 缓存）
        2. 不支持混合批次架构上的 prefill 计算

        处理流程：
        1. 构建或复用注意力掩码（ALiBi 偏置或滑动窗口掩码）
        2. 调整张量维度布局以适配 SDPA 接口
        3. 逐请求调用 SDPA 计算（因为每个请求长度不同，无法直接批处理）
        """
        # 获取或创建注意力掩码
        attn_masks = attn_metadata.sdpa_attn_masks
        if attn_masks is None:
            if self.alibi_slopes is not None:
                # ALiBi 位置编码：创建距离衰减偏置矩阵
                attn_masks = _make_alibi_bias(
                    self.alibi_slopes,
                    query.dtype,
                    attn_metadata.sdpa_start_loc,
                )
            elif self.sliding_window[0] != -1 or self.sliding_window[1] != -1:
                # 滑动窗口注意力：创建窗口范围内的掩码
                assert attn_metadata.seq_lens is not None
                attn_masks = _make_sliding_window_bias(
                    attn_metadata.sdpa_start_loc,
                    self.sliding_window[0],
                    self.sliding_window[1],
                    query.dtype,
                )
            else:
                # 标准注意力：无需特殊掩码
                attn_masks = [None] * (attn_metadata.sdpa_start_loc.size(0) - 1)  # type: ignore
            attn_metadata.sdpa_attn_masks = attn_masks

        # 调整维度顺序：将序列维度移到倒数第二维，适配 SDPA 的输入格式
        # SDPA 期望 [batch, num_heads, seq_len, head_dim]
        query = query.movedim(0, query.dim() - 2)
        key = key.movedim(0, key.dim() - 2)
        value = value.movedim(0, value.dim() - 2)

        causal_attn = attn_type == AttentionType.DECODER

        # 逐请求调用 SDPA（因为每个请求序列长度不同）
        sdpa_start_loc = attn_metadata.sdpa_start_loc.numpy()  # type: ignore
        for i in range(len(attn_masks)):
            mask = attn_masks[i]
            start_q = sdpa_start_loc[i]
            end_q = sdpa_start_loc[i + 1]
            sub_out = (
                torch.nn.functional.scaled_dot_product_attention(
                    query[None, :, start_q:end_q, :],
                    key[None, :, start_q:end_q, :],
                    value[None, :, start_q:end_q, :],
                    attn_mask=mask,
                    dropout_p=0.0,
                    is_causal=causal_attn and mask is None,
                    scale=self.scale,
                    enable_gqa=self.num_heads > self.num_kv_heads,
                )
                .squeeze(0)
                .movedim(query.dim() - 2, 0)
            )
            output[start_q:end_q, :, :] = sub_out
        return output


def _make_alibi_bias(
    alibi_slopes: torch.Tensor,
    dtype: torch.dtype,
    sdpa_start_loc: torch.Tensor,
) -> list[torch.Tensor]:
    """
    为 ALiBi（Attention with Linear Biases）位置编码创建注意力偏置矩阵。

    ALiBi 不使用显式的位置编码嵌入，而是在注意力分数上添加与距离成比例的
    线性偏置。偏置值 = -slope * |i - j|，其中 slope 是每个头不同的斜率。

    参数：
        alibi_slopes: 每个注意力头的 ALiBi 斜率，shape=[num_heads]
        dtype: 输出偏置矩阵的数据类型
        sdpa_start_loc: 每个请求的查询起始位置

    返回：
        每个请求的 ALiBi 偏置矩阵列表，shape=[num_heads, seq_len, seq_len]
    """
    attn_biases: list[torch.Tensor] = []
    seq_num = sdpa_start_loc.size(0) - 1
    sdpa_start_loc = sdpa_start_loc.numpy()  # type: ignore
    for i in range(seq_num):
        seq_len = sdpa_start_loc[i + 1] - sdpa_start_loc[i]
        bias = torch.arange(seq_len, dtype=dtype)  # type: ignore
        # NOTE(zhuohan): HF uses
        #     `bias = bias[None, :].repeat(seq_len, 1)`
        # here. We find that both biases give the same results, but
        # the bias below more accurately follows the original ALiBi
        # paper.
        # 计算距离矩阵：bias[i][j] = j - i（正数表示未来位置）
        bias = bias[None, :] - bias[:, None]

        num_heads = alibi_slopes.shape[0]
        # 对每个头复制距离矩阵并乘以对应的斜率
        bias = bias[None, :].repeat((num_heads, 1, 1))
        bias.mul_(alibi_slopes[:, None, None]).unsqueeze_(0)
        # 创建上三角无穷掩码，屏蔽未来位置（因果注意力）
        inf_mask = (
            torch.empty((1, seq_len, seq_len), dtype=bias.dtype)  # type: ignore
            .fill_(-torch.inf)
            .triu_(diagonal=1)
        )
        attn_biases.append((bias + inf_mask).to(dtype))

    return attn_biases


def _make_sliding_window_bias(
    sdpa_start_loc: torch.Tensor,
    left_window_size: int,
    right_window_size: int,
    dtype: torch.dtype,
) -> list[torch.Tensor]:
    """
    为滑动窗口注意力创建注意力掩码。

    滑动窗口注意力只允许每个 token 关注其周围固定窗口内的 token，
    从而将注意力计算的复杂度从 O(n^2) 降低到 O(n * window_size)。

    参数：
        sdpa_start_loc: 每个请求的查询起始位置
        left_window_size: 左侧窗口大小（关注历史 token 的范围），-1 表示无限制
        right_window_size: 右侧窗口大小（关注未来 token 的范围），-1 表示无限制
        dtype: 输出掩码的数据类型

    返回：
        每个请求的滑动窗口掩码列表，shape=[1, seq_len, seq_len]
        掩码中 0 的位置表示允许关注，-inf 表示屏蔽
    """
    attn_biases: list[torch.Tensor] = []
    seq_num = sdpa_start_loc.size(0) - 1
    sdpa_start_loc = sdpa_start_loc.numpy()  # type: ignore
    for i in range(seq_num):
        seq_len = sdpa_start_loc[i + 1] - sdpa_start_loc[i]
        mask = torch.full(  # type: ignore
            (1, seq_len, seq_len),  # type: ignore
            fill_value=1,
            dtype=dtype,
        )

        # 右侧窗口限制：屏蔽超过右窗口距离的未来位置
        if right_window_size != -1:
            mask = torch.tril(mask, diagonal=right_window_size)
        # 左侧窗口限制：屏蔽超过左窗口距离的历史位置
        if left_window_size != -1:
            mask = torch.triu(mask, diagonal=-left_window_size)
        # 将 1 转为 0（log(1)=0，不影响注意力分数），
        # 将 0 转为 -inf（log(0)=-inf，屏蔽该位置）
        mask = torch.log(mask)
        attn_biases.append(mask)

    return attn_biases


@functools.lru_cache(maxsize=1)
def _riscv_supports_rvv() -> bool:
    """Whether the C++ RVV attention path is usable.

    The kernel in csrc/cpu/cpu_attn_rvv.hpp uses VLEN-agnostic RVVI()
    macros and supports VLEN=128 and VLEN=256.  CMake auto-detects the
    largest zvl<N>b from /proc/cpuinfo and passes it via -mrvv-vector-bits.
    The RVV path is compiled whenever __riscv_v_min_vlen is defined, so
    we check that at least one supported zvl<N>b is advertised.
    """
    # 检测 RISC-V 向量扩展（RVV）支持。
    # 读取 /proc/cpuinfo 检查是否支持 zvl128b 或 zvl256b，
    # 排除 zvl512b 和 zvl1024b（当前 kernel 不支持这些 VLEN）
    try:
        with open("/proc/cpuinfo") as f:
            cpuinfo = f.read()
    except OSError:
        return False
    return any(f"zvl{n}b" in cpuinfo for n in (128, 256)) and all(
        f"zvl{n}b" not in cpuinfo for n in (512, 1024)
    )


def _get_attn_isa(
    dtype: torch.dtype,
    block_size: int,
    head_size: int | None = None,
    kv_cache_dtype: str | None = None,
) -> str:
    """
    根据硬件能力和模型参数选择最优的 ISA（指令集架构）计算路径。

    选择优先级：
    1. head_size 不是 32 的倍数但是 16 的倍数 -> "vec16"（向量路径，16 对齐）
    2. 支持 AMX 且为 bf16 且 block_size 为 32 的倍数 -> "amx"（Intel AMX 矩阵加速）
    3. block_size 为 32 的倍数时，按架构选择：
       - ARM -> "neon"（ARM NEON 指令）
       - RISC-V RVV -> "rvv"（RISC-V 向量扩展）
       - S390X -> "vxe"（IBM z 系列向量扩展）
       - PowerPC -> "vsx"（PowerPC 向量标量扩展）
       - 其他 -> "vec"（通用向量路径）
    4. 其他情况 -> "vec16"

    参数：
        dtype: 模型数据类型
        block_size: KV 缓存块大小
        head_size: 注意力头维度
        kv_cache_dtype: KV 缓存数据类型字符串

    返回：
        ISA 标识字符串
    """
    fp8_kv = is_quantized_kv_cache(kv_cache_dtype) if kv_cache_dtype else False
    # head_size 不是 32 的倍数时，只能使用 16 对齐的向量路径
    if head_size is not None and head_size % 32 != 0 and head_size % 16 == 0:
        if fp8_kv:
            raise NotImplementedError(
                "FP8 KV cache requires head_size divisible by 32 on CPU."
            )
        return "vec16"
    # 检测各种硬件加速支持
    supports_amx = torch.cpu._is_amx_tile_supported()
    arch = current_platform.get_cpu_architecture()
    supports_arm = arch == CpuArchEnum.ARM
    supports_vxe = arch == CpuArchEnum.S390X
    supports_riscv = arch == CpuArchEnum.RISCV
    supports_vsx = arch == CpuArchEnum.POWERPC
    supports_avx512 = torch.cpu._is_avx512_supported()
    # FP8 KV 缓存需要 AVX-512 或 AMX 支持
    if fp8_kv and not supports_amx and not supports_avx512:
        raise NotImplementedError(
            "FP8 KV cache on CPU requires x86 with AVX-512 or AMX."
        )
    # AMX 路径：Intel AMX 对 bf16 有专门的矩阵指令加速
    if supports_amx and dtype in (torch.bfloat16,) and block_size % 32 == 0:
        return "amx"
    elif block_size % 32 == 0:
        if supports_arm:
            # support ARM NEON FMLA and BFMMLA (bf16) for block size 32
            return "neon"
        elif supports_riscv and _riscv_supports_rvv():
            return "rvv"
        elif supports_vxe:
            return "vxe"
        elif supports_vsx:
            return "vsx"
        else:
            return "vec"
    else:
        return "vec16"
