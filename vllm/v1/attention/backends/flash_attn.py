# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention layer with FlashAttention."""

# =============================================================================
# FlashAttention 后端总览
# =============================================================================
# 本文件实现了 vLLM V1 的 FlashAttention 注意力后端，是 vLLM 中最核心的
# 注意力计算后端之一。
#
# 整体架构分为三层：
#   1. FlashAttentionBackend（后端入口类）：
#      - 向 vLLM 调度层声明本后端支持的数据类型、KV Cache 格式、硬件能力等约束。
#      - 提供工厂方法，返回元数据构建器（FlashAttentionMetadataBuilder）
#        和注意力实现（FlashAttentionImpl）。
#
#   2. FlashAttentionMetadataBuilder（元数据构建器）：
#      - 每个调度步调用 build() 方法，根据当前 batch 的请求信息生成
#        FlashAttentionMetadata，供前向传播使用。
#      - 支持三种模式的元数据构建：
#        a) 普通模式：标准的分页注意力元数据。
#        b) 级联模式（cascade）：当多个请求共享长公共前缀时，
#           将 KV 拆为前缀+后缀分别构建。
#        c) DCP 模式（Decode Context Parallelism）：多 rank 分布式解码，
#           将 KV 上下文按 rank 切分。
#      - 管理 AOT（Ahead-Of-Time）调度元数据和 CUDA Graph 缓冲区的预分配。
#
#   3. FlashAttentionImpl（注意力前向实现）：
#      - 每个 Attention 层实例化一个 FlashAttentionImpl。
#      - forward() 方法是核心，负责：
#        a) 编码器注意力：直接在 Q/K/V 上做双向注意力，不经过 KV Cache。
#        b) 普通解码器注意力：从分页 KV Cache 中取出 K/V，
#           调用 flash_attn_varlen_func 完成注意力计算。
#        c) DCP 注意力：多 rank 分布式上下文注意力。
#        d) 级联注意力：公共前缀+后缀分别计算后合并。
#      - do_kv_cache_update() 将新产生的 K/V 写入分页 KV Cache。
#
# 辅助函数：
#   - use_cascade_attention(): 启发式决策是否使用级联注意力。
#   - cascade_attention(): 级联注意力的具体实现。
#
# FlashAttention 的核心优势：
#   - 分块计算（tiling）：将 Q/K/V 分块加载到 SRAM，减少 HBM 访问次数。
#   - 重计算（recomputation）：反向传播时重新计算中间结果而非存储，节省显存。
#   - 分页注意力（paged attention）：通过 block_table 将逻辑 KV 映射到
#     物理显存中的 KV block，无需连续存储，支持高效的内存管理。
# =============================================================================

# 中文注释：本文件是 vLLM V1 的 FlashAttention 注意力后端实现。
# FlashAttention 是当前最主流的高效注意力计算方案，核心优势包括：
#   1. 分块计算（tiling）：将 Q/K/V 分块加载到 GPU SRAM，减少 HBM 带宽需求。
#   2. 分页注意力（paged attention）：通过 block_table 将逻辑 KV 块映射到物理显存中的
#      KV block，无需连续存储，支持高效的内存管理和 prefix cache 复用。
#   3. 变长序列支持：通过 cu_seqlens_q/k 和 flash_attn_varlen_func 支持
#      多个不同长度序列打包在一个 batch 中计算，避免 padding 浪费。
#   4. 支持级联注意力（cascade）、DCP 分布式解码等高级优化。
#
# 本文件的核心类层次结构：
#   - FlashAttentionBackend（后端注册类）：声明硬件约束，提供工厂方法。
#   - FlashAttentionMetadataBuilder（元数据构建器）：每步调度构建注意力元数据。
#   - FlashAttentionMetadata（元数据数据类）：存储前向传播所需的全部元数据。
#   - FlashAttentionImpl（注意力实现类）：执行实际的注意力前向计算。
# 辅助函数：
#   - use_cascade_attention()：启发式决策是否使用级联注意力。
#   - cascade_attention()：级联注意力的具体实现。

import copy
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import torch

from vllm.model_executor.layers.attention import Attention
from vllm.platforms import current_platform
from vllm.utils.torch_utils import (
    canonicalize_singleton_dim_strides,
    is_quantized_kv_cache,
)
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionType,
    MultipleOf,
)
from vllm.v1.attention.backends.fa_utils import (
    flash_attn_supports_quant_query_input,
    get_flash_attn_version,
    is_fa_version_supported,
    is_flash_attn_varlen_func_available,
)
from vllm.v1.attention.backends.utils import get_dcp_local_seq_lens
from vllm.v1.attention.ops.common import cp_lse_ag_out_rs
from vllm.v1.attention.ops.dcp_alltoall import dcp_a2a_lse_reduce
from vllm.v1.attention.ops.merge_attn_states import merge_attn_states
from vllm.v1.worker.workspace import current_workspace_manager

if is_flash_attn_varlen_func_available():
    from vllm.v1.attention.backends.fa_utils import (
        flash_attn_supports_sinks,
        flash_attn_varlen_func,
        get_scheduler_metadata,
        reshape_and_cache_flash,
    )
import vllm.envs as envs
from vllm.config import (
    VllmConfig,
    get_current_vllm_config,
    get_current_vllm_config_or_none,
    get_layers_from_vllm_config,
)
from vllm.config.cache import CacheDType
from vllm.distributed.parallel_state import get_dcp_group
from vllm.logger import init_logger
from vllm.platforms.interface import DeviceCapability
from vllm.utils.math_utils import cdiv, round_up
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.utils import (
    get_kv_cache_layout,
)
from vllm.v1.kv_cache_interface import AttentionSpec

logger = init_logger(__name__)


# 中文注释：FlashAttentionBackend 是 FlashAttention 后端的入口注册类。
# 它继承自 AttentionBackend，作用是：
#   1. 声明该后端支持的数据类型、KV Cache 类型、计算能力等硬件约束。
#   2. 提供工厂方法，返回元数据构建器（FlashAttentionMetadataBuilder）
#      和注意力实现（FlashAttentionImpl）。
#   3. vLLM 的注意力调度层通过此类的各个 supports_* 方法判断是否可以使用 FlashAttention。
#   4. 定义 KV Cache 的逻辑形状和内存布局（NHD/HND）。
class FlashAttentionBackend(AttentionBackend):
    # 中文注释：FlashAttention 支持的计算数据类型，只支持半精度浮点。
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    # 中文注释：FlashAttention 支持的 KV Cache 存储数据类型。
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
    ]

    # 中文注释：返回该后端支持的 KV block 大小列表。
    # FlashAttention 要求 block_size 必须是 16 的倍数。
    # 对于混合架构（如 Jamba = Attention + Mamba），如果 Mamba 使用 float32 缓存，
    # 则需要限制为 16/32/64 以避免 FA 的 NaN 传播问题。
    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        vllm_config = get_current_vllm_config()
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        if (
            model_config
            and model_config.is_hybrid
            and (
                cache_config.mamba_ssm_cache_dtype == "float32"
                or cache_config.mamba_cache_dtype == "float32"
            )
        ):
            # NOTE(tdoublep): while in principle, FA supports
            # MultipleOf(16), these are the block sizes that do not
            # suffer from the NaN propagation problem described here:
            # https://github.com/Dao-AILab/flash-attention/issues/1974
            return [16, 32, 64]
        return [MultipleOf(16)]

    # 中文注释：标识该后端在 forward() 中不包含 KV Cache 更新操作。
    # KV Cache 的写入由单独的 do_kv_cache_update() 方法完成，
    # 这样可以将 KV Cache 更新和注意力计算解耦，便于调度优化。
    forward_includes_kv_cache_update: bool = False

    # 中文注释：返回该后端偏好的 block 大小。
    # XPU 平台要求 block 大小至少为 64，以获得更好的性能。
    @classmethod
    def get_preferred_block_size(cls, default_block_size: int) -> int:
        if current_platform.is_xpu():
            return max(default_block_size, 64)
        return super().get_preferred_block_size(default_block_size)

    # 中文注释：返回后端名称，用于日志和调试。
    @staticmethod
    def get_name() -> str:
        return "FLASH_ATTN"

    # 中文注释：声明该后端支持 batch invariance（批不变性）。
    # batch invariance 意味着无论 batch 大小如何，计算结果都完全相同，
    # 这对于确定性推理和调试非常重要。
    @classmethod
    def supports_batch_invariance(cls) -> bool:
        return True

    # 中文注释：声明该后端支持非因果注意力（bidirectional attention），
    # 适用于编码器模式。
    @classmethod
    def supports_non_causal(cls) -> bool:
        return True

    # 中文注释：声明该后端支持的注意力类型。
    # FlashAttention 支持所有四种注意力类型：解码器、编码器、纯编码器、编码器-解码器。
    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        """FlashAttention supports all attention types."""
        return attn_type in (
            AttentionType.DECODER,
            AttentionType.ENCODER,
            AttentionType.ENCODER_ONLY,
            AttentionType.ENCODER_DECODER,
        )

    # 中文注释：声明该后端是否支持 per-head 量化缩放因子。
    # FA3 及以上版本支持，允许每个注意力头使用独立的量化缩放值。
    @classmethod
    def supports_per_head_quant_scales(cls) -> bool:
        fa_version = get_flash_attn_version()
        return fa_version is not None and fa_version >= 3

    # 中文注释：工厂方法，返回注意力前向计算的实现类。
    @staticmethod
    def get_impl_cls() -> type["FlashAttentionImpl"]:
        return FlashAttentionImpl

    # 中文注释：工厂方法，返回注意力元数据的构建器类。
    @staticmethod
    def get_builder_cls() -> type["FlashAttentionMetadataBuilder"]:
        return FlashAttentionMetadataBuilder

    # 中文注释：定义 KV Cache 的逻辑形状。
    # 返回 (num_blocks, 2, block_size, num_kv_heads, head_size)，
    # 其中维度 2 表示 K 和 V 两部分合并存储。
    # FlashAttention 要求 block_size 必须是 16 的倍数（对齐硬件要求）。
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

    # 中文注释：返回 KV Cache 内存布局的 stride 排列顺序。
    # 根据配置的 cache_layout（NHD 或 HND），返回对应的维度排列，
    # 使得 KV Cache 的物理内存布局与 FlashAttention 内核的期望一致。
    # NHD: [num_blocks, 2, block_size, num_kv_heads, head_size]（head 维度在后）
    # HND: [num_blocks, 2, num_kv_heads, block_size, head_size]（head 维度在前）
    # 当 include_num_layers_dimension=True 时，额外包含层维度，用于跨层 KV 共享。
    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        # `stride_order` indicates the permutation that gets
        # us from `get_kv_cache_shape` to the actual memory layout we want.
        cache_layout = get_kv_cache_layout()
        if cache_layout == "NHD" and include_num_layers_dimension:
            # (num_blocks, num_layers, 2, block_size, num_kv_heads, head_size)
            return (1, 0, 2, 3, 4, 5)
        elif cache_layout == "NHD":
            stride_order = (0, 1, 2, 3, 4)
        elif cache_layout == "HND" and include_num_layers_dimension:
            # (num_blocks, num_kv_heads, num_layers, 2, block_size, head_size)
            return (1, 4, 0, 2, 3, 5)
        elif cache_layout == "HND":
            stride_order = (0, 1, 3, 2, 4)
        else:
            raise ValueError(f"Unknown cache layout format {cache_layout}.")
        return stride_order

    # 中文注释：声明该后端支持的 head_size 范围。
    # head_size 必须是 8 的倍数，且不超过 256（FA2）或 512（FA4）。
    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        if head_size % 8 != 0:
            return False
        if head_size <= 256:
            return True
        if is_fa_version_supported(4):
            return head_size <= 512
        return False

    # 中文注释：声明该后端支持的 KV Cache 数据类型。
    # 标准支持 auto/float16/bfloat16。
    # FP8 量化支持需要 FA3 + Hopper 架构（计算能力 9.0）。
    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: CacheDType | None) -> bool:
        if kv_cache_dtype is None:
            return True
        if kv_cache_dtype in ("fp8", "fp8_e4m3"):
            if current_platform.is_xpu():
                return True
            return (
                get_flash_attn_version() == 3
                and current_platform.is_device_capability_family(90)
            )
        return kv_cache_dtype in ["auto", "float16", "bfloat16"]

    # 中文注释：声明该后端是否支持 sink token 机制。
    # Sink token 是 FA3 引入的机制，在注意力计算中加入特殊的 sink token，
    # 用于稳定注意力分数分布，防止注意力权重过度集中在少数 token 上。
    @classmethod
    def supports_sink(cls) -> bool:
        if not is_flash_attn_varlen_func_available():
            return False
        return flash_attn_supports_sinks()

    # 要求 GPU 计算能力 >= 8.0（Ampere 及以上），因为 FlashAttention 依赖这些架构的硬件特性。
    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability >= DeviceCapability(8, 0)

    # 组合兼容性检查：在给定 head_size、dtype、KV Cache 类型、block_size 等参数下，
    # 判断该后端是否支持该配置组合。返回 None 表示支持，返回字符串表示不支持的原因。
    @classmethod
    def supports_combination(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: CacheDType | None,
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        device_capability: DeviceCapability,
    ) -> str | None:
        if has_sink and device_capability < DeviceCapability(9, 0):
            return "sink not supported on compute capability < 9.0"
        return None


# FlashAttentionMetadata: FlashAttention 前向传播所需的全部元数据。
# 由 FlashAttentionMetadataBuilder.build() 在每一步调度时构建，
# 传递给 FlashAttentionImpl.forward() 指导注意力计算。
# 包含三类信息：(1) 基本的序列/块表信息；(2) 级联注意力（cascade）的前缀/后缀切分信息；
# (3) DCP（Decode Context Parallelism）的上下文分片信息。
@dataclass
class FlashAttentionMetadata:
    # NOTE(sang): Definition of context_len, query_len, and seq_len.
    # |---------- N-1 iteration --------|
    # |---------------- N iteration ---------------------|
    # |- tokenA -|......................|-- newTokens ---|
    # |---------- context_len ----------|
    # |-------------------- seq_len ---------------------|
    #                                   |-- query_len ---|

    num_actual_tokens: int  # Number of tokens excluding padding.
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor

    # 级联注意力（cascade attention）相关字段。
    # 当多个请求共享较长公共前缀时，将 KV 分为公共前缀和各请求后缀两部分分别计算，
    # 再合并结果，以减少重复计算量。use_cascade 标记是否启用此优化。
    # For cascade attention.
    use_cascade: bool
    common_prefix_len: int
    cu_prefix_query_lens: torch.Tensor | None
    prefix_kv_lens: torch.Tensor | None
    suffix_kv_lens: torch.Tensor | None

    # DCP（Decode Context Parallelism）相关字段。
    # DCP 将长序列的 KV 上下文按 rank 切分，每个 rank 只处理本地切片，
    # 再通过通信合并注意力结果，用于支持超长上下文的分布式解码。
    # max_dcp_context_kv_len 用于避免 GPU->CPU 同步，预先计算最大值。
    # For GQA DCP
    max_dcp_context_kv_len: int | None = None
    dcp_context_kv_lens: torch.Tensor | None = None

    # AOT（Ahead-Of-Time）调度元数据，由 FA3 的 get_scheduler_metadata() 生成。
    # 用于在 CUDA Graph 捕获时预计算调度信息，避免运行时开销。
    # max_num_splits 控制 split-KV 的分片数，0 表示使用 FA3 的启发式。
    # Optional aot scheduling
    scheduler_metadata: torch.Tensor | None = None
    prefix_scheduler_metadata: torch.Tensor | None = None
    max_num_splits: int = 0

    causal: bool = True


# 中文注释：获取模型中所有使用 FlashAttention 的注意力层的滑动窗口配置集合。
# 该函数在首次 build() 调用时执行，用于确定 AOT（Ahead-Of-Time）调度器
# 是否可以使用滑动窗口优化。
# 注意：只检查使用 FlashAttentionImpl 的层，跳过其他后端（如 TurboQuant、MLA）。
# 如果所有层的滑动窗口配置不一致（例如某些层有滑动窗口，某些没有），
# 则禁用 AOT 调度以避免配置冲突。
def _get_sliding_window_configs(
    vllm_config: VllmConfig,
) -> set[tuple[int, int] | None]:
    """Get the set of all sliding window configs used in the model.

    Only inspects FlashAttentionImpl layers. Other backends (e.g.
    TurboQuant, MLA) use their own metadata builders and are skipped.
    """
    sliding_window_configs: set[tuple[int, int] | None] = set()
    layers = get_layers_from_vllm_config(vllm_config, Attention)
    for layer in layers.values():
        if not isinstance(layer.impl, FlashAttentionImpl):
            continue
        sliding_window_configs.add(layer.impl.sliding_window)
    return sliding_window_configs


# 中文注释：FlashAttentionMetadataBuilder 是 FlashAttention 的元数据构建器。
# 它继承自 AttentionMetadataBuilder，负责在每个调度步（iteration）构建
# FlashAttentionMetadata，供前向传播使用。
#
# 核心职责：
#   1. 预分配持久化缓冲区（如 scheduler_metadata），在后续 build() 中原地复用。
#   2. 管理 CUDA Graph 配置，决定是否使用 AOT 调度和 CUDA Graph 捕获。
#   3. 根据当前 batch 的请求信息，决定使用哪种注意力模式（普通/级联/DCP）。
#   4. 构建并返回 FlashAttentionMetadata 实例。
#
# CUDA Graph 支持说明：
#   - FA3：支持所有情况的完整 CUDA Graph（ALWAYS 模式）。
#   - FA2：仅支持 UNIFORM_BATCH 模式，即所有请求的 query 长度相同。
#     这是因为 FA2 在 max_query_len=1 时有特殊的 packed-GQA 优化路径，
#     该路径与混合 prefill-decode 场景不兼容。
class FlashAttentionMetadataBuilder(AttentionMetadataBuilder[FlashAttentionMetadata]):
    # FA3:
    # Supports full cudagraphs for all cases.
    #
    # FA2:
    # For FA2, a graph is captured with max_query_len=1, (which is what we
    # capture by default for num_tokens <= max_num_seqs when there is no
    # spec-decode) then these graphs will not work for mixed prefill-decode
    # (unlike FA3). This is due to special max_query_len=1 packed-GQA handling
    # in FA2.
    # In summary if we are running with spec decodes the graphs would
    # work for mixed prefill-decode and uniform-decode. But for non-spec decodes
    # the graphs would not work for mixed prefill-decode; sorta the inverse
    # of UNIFORM_SINGLE_TOKEN_DECODE.
    # There's probably a better way to describe this using `AttentionCGSupport`
    # but for now just set it to `UNIFORM_BATCH` to get use to drop down
    # to FULL_AND_PIECEWISE.
    # TODO(luka, lucas): audit FA2 as part of:
    #  https://github.com/vllm-project/vllm/issues/22945
    # 中文注释：CUDA Graph 支持级别。
    # FA3/XPU：ALWAYS，所有情况都可以使用 CUDA Graph。
    # FA2：UNIFORM_BATCH，仅支持 batch 内所有请求 query 长度相同的情况。
    _cudagraph_support = (
        AttentionCGSupport.ALWAYS
        if get_flash_attn_version() == 3 or current_platform.is_xpu()
        else AttentionCGSupport.UNIFORM_BATCH
    )
    # 中文注释：标识该构建器支持运行时更新 block_table，
    # 用于增量分配 KV block 的场景（如 prefill 分步执行）。
    supports_update_block_table: bool = True

    # 中文注释：返回该构建器的 CUDA Graph 支持级别。
    # 被调度层调用以决定是否可以使用 CUDA Graph 优化。
    @classmethod
    def get_cudagraph_support(
        cls,
        vllm_config: "VllmConfig",
        kv_cache_spec: "AttentionSpec",
    ) -> AttentionCGSupport:
        return cls._cudagraph_support

    # 中文注释：FlashAttentionMetadataBuilder 的初始化方法。
    # 在引擎启动时调用一次，完成以下工作：
    #   1. 读取模型配置（注意力头数、head 大小、block 大小等）。
    #   2. 读取 DCP（Decode Context Parallelism）并行配置（world_size、rank）。
    #   3. 如果启用 CUDA Graph + FA3，预分配 scheduler_metadata 缓冲区，
    #      该缓冲区在后续每次 build() 中被原地复用，避免反复分配内存。
    #   4. 如果启用 DCP，预分配 dcp_context_kv_lens 缓冲区。
    #   5. 初始化 AOT 滑动窗口配置（延迟到首次 build() 时填充）。
    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.model_config = vllm_config.model_config
        self.parallel_config = vllm_config.parallel_config
        self.cache_config = vllm_config.cache_config
        self.compilation_config = vllm_config.compilation_config
        self.attention_config = vllm_config.attention_config

        self.num_heads_q = self.model_config.get_num_attention_heads(
            self.parallel_config
        )
        self.num_heads_kv = self.model_config.get_num_kv_heads(self.parallel_config)
        self.kv_cache_dtype = kv_cache_spec.dtype
        self.headdim = self.model_config.get_head_size()
        self.block_size = kv_cache_spec.block_size

        self.max_num_splits = 0  # No upper bound on the number of splits.
        self.aot_schedule = get_flash_attn_version() == 3

        try:
            from vllm.distributed.parallel_state import get_dcp_group

            self.dcp_world_size = get_dcp_group().world_size
            self.dcp_rank = get_dcp_group().rank_in_group
        except AssertionError:
            # DCP might not be initialized in testing
            self.dcp_world_size = 1
            self.dcp_rank = 0

        self.cp_kv_cache_interleave_size = (
            self.parallel_config.cp_kv_cache_interleave_size
        )

        self.use_full_cuda_graph = (
            self.compilation_config.cudagraph_mode.has_full_cudagraphs()
        )
        self.max_cudagraph_size = self.compilation_config.max_cudagraph_capture_size

        if self.use_full_cuda_graph and self.aot_schedule:
            # FA3 scheduler_metadata size: 1 + round_up(batch_size, 4) * 4
            # The +1 is for the tile_count_semaphore (synchronization).
            # The 4 slots per batch element (num_prepare_batch_vectors) are:
            #   prepare_varlen + dynamic_split + sort_batches + head_swizzle
            # See: https://github.com/vllm-project/flash-attention/blob/5824e6e/hopper/flash_api.cpp#L664-L671  # noqa: E501
            max_batch_size = max(
                vllm_config.scheduler_config.max_num_seqs,
                self.max_cudagraph_size or 0,
            )
            self.scheduler_metadata = torch.zeros(
                1 + round_up(max_batch_size, 4) * 4,
                dtype=torch.int32,
                device=self.device,
            )
            # When using cuda graph, we need to set the upper bound of the
            # number of splits so that large enough intermediate buffers are
            # pre-allocated during capture.
            self.max_num_splits = (
                self.attention_config.flash_attn_max_num_splits_for_cuda_graph
            )

        if self.dcp_world_size > 1:
            max_num_reqs = vllm_config.scheduler_config.max_num_seqs
            self._dcp_context_kv_lens = torch.zeros(
                max_num_reqs,
                dtype=torch.int32,
                device=self.device,
            )

        # Sliding window size to be used with the AOT scheduler will be
        # populated on first build() call.
        self.aot_sliding_window: tuple[int, int] | None = None

    # 中文注释：build() 是核心元数据构建方法，每个调度步（iteration）调用一次。
    # 根据当前 batch 的序列信息和配置，决定使用哪种注意力模式并构建元数据。
    #
    # 三种注意力模式：
    # (1) 普通模式：直接构建 scheduler_metadata，调用标准 flash_attn_varlen_func。
    #     这是最常见的路径，适用于无共享前缀、无 DCP 的场景。
    # (2) 级联模式（cascade）：当 common_prefix_len > 0 时启用。
    #     将 KV 拆为公共前缀和各请求后缀两部分，分别构建 scheduler_metadata。
    #     前缀部分以 causal=False 计算（所有请求共享），后缀部分以 causal=True 计算。
    #     构建 cu_prefix_query_lens、prefix_kv_lens、suffix_kv_lens 等字段。
    # (3) DCP 模式：当 dcp_world_size > 1 时启用。
    #     计算每个 rank 的本地 KV 长度切片，构建 dcp_context_kv_lens 和
    #     max_dcp_context_kv_len。注意 DCP 模式下 causal=False（因为上下文被切分，
    #     每个 rank 只看到部分 KV，需要全注意力）。
    #
    # 参数说明：
    #   - common_prefix_len: 公共前缀长度（token 数），>0 时触发级联注意力。
    #   - common_attn_metadata: 通用注意力元数据，包含所有请求的序列信息。
    #   - fast_build: 是否跳过 AOT 调度。投机解码等迭代次数少的场景使用，
    #     因为 AOT 调度的开销在少量迭代中不值得。
    #
    # AOT 调度（Ahead-Of-Time）说明：
    #   FA3 的 get_scheduler_metadata() 会预先计算调度信息（如 split-KV 的分片策略），
    #   存储在 scheduler_metadata 缓冲区中。在 CUDA Graph 捕获时，
    #   这些信息被固化到图中，避免运行时重新计算，从而减少 CPU 开销。
    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> FlashAttentionMetadata:
        """
        fast_build disables AOT scheduling, used when there will be few
        iterations i.e. spec-decode
        """
        # 中文注释：从通用注意力元数据中提取各项字段。
        # num_reqs: 本步调度的请求数量。
        # num_actual_tokens: 本步实际需要计算的 token 数（不含 padding）。
        # max_query_len: 所有请求中 query 的最大长度。
        # max_seq_len: 所有请求中完整序列的最大长度（含已计算的 context）。
        # query_start_loc: 每个请求的 query 起始位置的前缀和，用于变长序列打包。
        # seq_lens: 每个请求的完整序列长度。
        # block_table_tensor: 逻辑 block 到物理 block 的映射表。
        # slot_mapping: 逻辑 slot 到物理 slot 的映射，用于 KV Cache 的 scatter write。
        # causal: 是否使用因果注意力（解码器为 True，编码器为 False）。
        num_reqs = common_attn_metadata.num_reqs
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        max_query_len = common_attn_metadata.max_query_len
        max_seq_len = common_attn_metadata.max_seq_len
        query_start_loc = common_attn_metadata.query_start_loc
        seq_lens = common_attn_metadata.seq_lens
        block_table_tensor = common_attn_metadata.block_table_tensor
        slot_mapping = common_attn_metadata.slot_mapping
        causal = common_attn_metadata.causal

        # 中文注释：决定是否启用 AOT 调度。
        # 以下情况禁用 AOT：
        #   1. fast_build=True（投机解码场景，迭代次数少，不值得 AOT 开销）。
        #   2. 批不变模式（VLLM_BATCH_INVARIANT），因为调度会随 max_seqlen_q/k 变化。
        # Disable AOT schedule for spec-decode proposer (not worth the overhead)
        # and for batch invariance (schedule varies with max_seqlen_q/k).
        aot_schedule = (
            self.aot_schedule and not fast_build and not envs.VLLM_BATCH_INVARIANT
        )

        if self.aot_sliding_window is None:
            self.aot_sliding_window = (-1, -1)
            # For the AOT scheduler we need the sliding window value to be
            # constant for all layers to. We have to populate this on the first
            # build() call so the layers are constructed (cannot populate)
            # in __init__.
            if aot_schedule:
                sliding_window_configs = _get_sliding_window_configs(self.vllm_config)
                if len(sliding_window_configs) == 1:
                    sliding_window_config = sliding_window_configs.pop()
                    if sliding_window_config is not None:
                        self.aot_sliding_window = sliding_window_config
                elif len(sliding_window_configs) > 1:
                    self.aot_schedule = False
                    aot_schedule = False

        # 中文注释：确定 split-KV 的分片数（num_splits）。
        # split-KV 是 FlashAttention 的一种优化：当序列很长时，
        # 将 KV 分成多个切片并行计算，最后合并结果，以提高 GPU SM 利用率。
        # max_num_splits=0 表示使用 FA3 的启发式自动决定分片数（不兼容 CUDA Graph）。
        # 使用 CUDA Graph 时，需要预先确定分片数以分配固定大小的中间缓冲区。
        max_num_splits = 0  # 0 means use FA3's heuristics, not CG compatible
        if (
            self.use_full_cuda_graph
            and self.max_cudagraph_size is not None
            and num_actual_tokens <= self.max_cudagraph_size
        ):
            # NOTE(woosuk): Setting num_splits > 1 may increase the memory
            # usage, because the intermediate buffers of size [num_splits,
            # num_heads, num_tokens, head_size] are allocated. Therefore,
            # we only set num_splits when using cuda graphs.
            max_num_splits = self.max_num_splits

        # 中文注释：批不变模式下强制 max_num_splits=1，确保计算结果与 batch 大小无关。
        if envs.VLLM_BATCH_INVARIANT:
            max_num_splits = 1

        def schedule(
            batch_size, cu_query_lens, max_query_len, seqlens, max_seq_len, causal
        ):
            cache_dtype = self.cache_config.cache_dtype
            if is_quantized_kv_cache(cache_dtype):
                qkv_dtype = current_platform.fp8_dtype()
            else:
                qkv_dtype = self.kv_cache_dtype
            if aot_schedule:
                return get_scheduler_metadata(
                    batch_size=batch_size,
                    max_seqlen_q=max_query_len,
                    max_seqlen_k=max_seq_len,
                    num_heads_q=self.num_heads_q * self.dcp_world_size,
                    num_heads_kv=self.num_heads_kv,
                    headdim=self.headdim,
                    cache_seqlens=seqlens,
                    qkv_dtype=qkv_dtype,
                    cu_seqlens_q=cu_query_lens,
                    page_size=self.block_size,
                    causal=causal,
                    window_size=self.aot_sliding_window,
                    num_splits=max_num_splits,
                )
            return None

        # 中文注释：初始化级联注意力和 DCP 的字段。
        # use_cascade: 当公共前缀长度 > 0 时启用级联注意力。
        # 级联注意力将注意力计算拆为前缀（共享）+ 后缀（独立）两部分，
        # 公共前缀的 KV 只需加载和计算一次，显著减少带宽和算力消耗。
        use_cascade = common_prefix_len > 0
        max_dcp_context_kv_len = 0
        dcp_context_kv_lens = None

        cu_prefix_query_lens = None
        prefix_kv_lens = None
        suffix_kv_lens = None
        prefix_scheduler_metadata = None

        # 中文注释：根据配置选择不同的元数据构建路径。
        # 中文注释：DCP（Decode Context Parallelism）模式的元数据构建。
        # DCP 将长上下文的 KV Cache 按 rank 切分，每个 rank 只存储和计算一部分 KV。
        # 流程：
        #   1. 计算每个请求的 context_kv_lens（已计算的 KV 长度）。
        #   2. 将 context_kv_lens 按 DCP world_size 和 interleave_size 切分，
        #      得到每个 rank 的本地 KV 长度 local_context_kv_lens。
        #   3. 计算 max_dcp_context_kv_len 用于预分配缓冲区，
        #      避免运行时 GPU->CPU 同步。
        if self.dcp_world_size > 1:
            query_lens = query_start_loc[1:] - query_start_loc[:-1]
            context_kv_lens = seq_lens - query_lens
            local_context_kv_lens = get_dcp_local_seq_lens(
                context_kv_lens,
                self.dcp_world_size,
                self.dcp_rank,
                self.cp_kv_cache_interleave_size,
            )
            self._dcp_context_kv_lens[:num_reqs] = local_context_kv_lens
            self._dcp_context_kv_lens[num_reqs:] = 0
            dcp_context_kv_lens = self._dcp_context_kv_lens[:num_reqs]

            # After DCP distribution, the maximum number of tokens for any rank is
            # ceil(L / (N * I)) * I, where L is max_seq_len, N is dcp_world_size,
            # and I is cp_kv_cache_interleave_size.
            # This eliminates GPU->CPU sync while minimizing workspace over-allocation.
            num_partitions = self.dcp_world_size * self.cp_kv_cache_interleave_size
            max_dcp_context_kv_len = (
                (max_seq_len + num_partitions - 1) // num_partitions
            ) * self.cp_kv_cache_interleave_size

            scheduler_metadata = schedule(
                batch_size=num_reqs,
                cu_query_lens=query_start_loc,
                max_query_len=max_query_len,
                seqlens=dcp_context_kv_lens,
                max_seq_len=max_dcp_context_kv_len,
                causal=False,
            )
        elif use_cascade:
            cu_prefix_query_lens = torch.tensor(
                [0, num_actual_tokens], dtype=torch.int32, device=self.device
            )
            prefix_kv_lens = torch.tensor(
                [common_prefix_len], dtype=torch.int32, device=self.device
            )
            # Use GPU tensor directly - no CPU sync needed
            suffix_kv_lens = seq_lens[:num_reqs] - common_prefix_len
            prefix_scheduler_metadata = schedule(
                batch_size=1,
                cu_query_lens=cu_prefix_query_lens,
                max_query_len=num_actual_tokens,
                seqlens=prefix_kv_lens,
                max_seq_len=common_prefix_len,
                causal=False,
            )
            scheduler_metadata = schedule(
                batch_size=num_reqs,
                cu_query_lens=query_start_loc,
                max_query_len=max_query_len,
                seqlens=suffix_kv_lens,
                max_seq_len=max_seq_len - common_prefix_len,
                causal=True,
            )
        else:
            scheduler_metadata = schedule(
                batch_size=num_reqs,
                cu_query_lens=query_start_loc,
                max_query_len=max_query_len,
                seqlens=seq_lens,
                max_seq_len=max_seq_len,
                causal=causal,
            )
        # For FA3 + full cudagraph
        if self.use_full_cuda_graph and scheduler_metadata is not None:
            n = scheduler_metadata.shape[0]
            self.scheduler_metadata[:n] = scheduler_metadata
            # NOTE(woosuk): We should zero out the rest of the scheduler
            # metadata to guarantee the correctness. Otherwise, some thread
            # blocks may use the invalid scheduler metadata and overwrite the
            # output buffer.
            self.scheduler_metadata[n:] = 0
            scheduler_metadata = self.scheduler_metadata[:n]

        attn_metadata = FlashAttentionMetadata(
            num_actual_tokens=num_actual_tokens,
            max_query_len=max_query_len,
            query_start_loc=query_start_loc,
            max_seq_len=max_seq_len,
            seq_lens=seq_lens,
            block_table=block_table_tensor,
            slot_mapping=slot_mapping,
            max_dcp_context_kv_len=max_dcp_context_kv_len,
            dcp_context_kv_lens=dcp_context_kv_lens,
            use_cascade=use_cascade,
            common_prefix_len=common_prefix_len,
            scheduler_metadata=scheduler_metadata,
            cu_prefix_query_lens=cu_prefix_query_lens,
            prefix_kv_lens=prefix_kv_lens,
            suffix_kv_lens=suffix_kv_lens,
            prefix_scheduler_metadata=prefix_scheduler_metadata,
            max_num_splits=max_num_splits,
            causal=causal,
        )
        return attn_metadata

    def update_block_table(
        self,
        metadata: FlashAttentionMetadata,
        blk_table: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> FlashAttentionMetadata:
        new_metadata = copy.copy(metadata)
        new_metadata.block_table = blk_table
        new_metadata.slot_mapping = slot_mapping
        return new_metadata

    def use_cascade_attention(self, *args, **kwargs) -> bool:
        return use_cascade_attention(*args, **kwargs)


# FlashAttentionImpl: FlashAttention 的前向计算实现。
# 每个注意力层（Attention）实例化一个 FlashAttentionImpl，负责：
# (1) 将 Q/K/V 与 KV Cache 传入 flash_attn_varlen_func 完成注意力计算；
# (2) 管理 KV Cache 的写入（do_kv_cache_update）；
# (3) 支持编码器/解码器/交叉注意力/DCP/级联等多种注意力路径。
class FlashAttentionImpl(AttentionImpl):
    can_return_lse_for_decode: bool = True

    # __init__(): 初始化注意力实现层的配置参数。
    # 关键参数说明：
    # - sliding_window: 滑动窗口注意力的窗口大小。(-1,-1) 表示无滑动窗口。
    #   编码器模式下左右窗口对称；解码器模式下右侧窗口为 0（只看左侧历史）。
    # - logits_soft_cap: Gemma 风格的 soft cap 值，限制注意力 logits 的范围。
    #   FlashAttention 中设为 0 表示不启用 soft cap。
    # - sinks: FlashAttention 3 特有的 sink token 机制，用于稳定注意力分布。
    # - dcp_combine: DCP 模式下的通信原语选择。
    #   a2a 模式使用 all-to-all，否则使用 all-gather + reduce-scatter。
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
        kv_sharing_target_layer_name: str | None = None,
        sinks: torch.Tensor | None = None,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        if sliding_window is None:
            self.sliding_window = (-1, -1)
        elif attn_type == AttentionType.ENCODER_ONLY:
            self.sliding_window = (sliding_window - 1, sliding_window - 1)
        else:
            self.sliding_window = (sliding_window - 1, 0)
        self.kv_cache_dtype = kv_cache_dtype
        if logits_soft_cap is None:
            # In flash-attn, setting logits_soft_cap as 0 means no soft cap.
            logits_soft_cap = 0
        self.logits_soft_cap = logits_soft_cap
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name

        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        self.attn_type = attn_type
        self.vllm_flash_attn_version = get_flash_attn_version(
            requires_alibi=alibi_slopes is not None,
            head_size=head_size,
        )
        logger.info_once(
            "Using FlashAttention version %s",
            self.vllm_flash_attn_version,
        )
        # Cache the batch invariant result for use in forward passes
        self.batch_invariant_enabled = envs.VLLM_BATCH_INVARIANT

        self.sinks = sinks
        if self.sinks is not None:
            assert flash_attn_supports_sinks(), (
                "Sinks are only supported in FlashAttention 3"
            )
            assert self.sinks.shape[0] == num_heads, (
                "Sinks must have the same number of heads as the number of "
                "heads in the layer"
            )

        self.supports_quant_query_input = flash_attn_supports_quant_query_input()

        vllm_config = get_current_vllm_config_or_none()
        dcp_a2a = (
            vllm_config is not None
            and vllm_config.parallel_config.decode_context_parallel_size > 1
            and vllm_config.parallel_config.dcp_comm_backend == "a2a"
        )
        self.dcp_combine = dcp_a2a_lse_reduce if dcp_a2a else cp_lse_ag_out_rs

        self._dcp_dtype: torch.dtype | None = None
        if vllm_config is not None and self.dcp_world_size > 1:
            self._dcp_dtype = vllm_config.model_config.dtype

    # forward(): 注意力前向传播的核心调度方法。
    # 根据注意力类型和元数据中的模式标记，分发到不同计算路径：
    # (1) 编码器路径：encoder_only/encoder 类型直接调用 _forward_encoder_attention，
    #     不经过 KV Cache，直接在 Q/K/V 上做双向注意力。
    # (2) 普通解码器路径：从 KV Cache 中取出 key_cache/value_cache，
    #     调用 flash_attn_varlen_func 执行标准的分页注意力。
    # (3) DCP 路径：当 dcp_world_size > 1 时，调用 _forward_with_dcp，
    #     将上下文 KV 按 rank 切分计算后通过通信合并。
    # (4) 级联路径：当 use_cascade=True 时，调用 cascade_attention()，
    #     将公共前缀和各请求后缀分别计算后合并。
    # 注意：此方法在 piece-wise CUDA Graph 模式下以 eager PyTorch 执行，
    # 因此需要尽量减少 CPU 开销（避免不必要的 view/slice 操作）。
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
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache: shape =
                [num_blocks, 2, block_size, num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
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

        # 中文注释：num_actual_tokens 是本轮 batch 中实际有效的 token 数量。
        # 由于 CUDA Graph 等优化，输入张量可能包含 padding，只有前 num_actual_tokens
        # 个 token 是真实需要计算的。后续所有切片操作都基于此值。
        num_actual_tokens = attn_metadata.num_actual_tokens

        # 中文注释：编码器注意力路径。编码器（如 BERT）不需要 KV Cache，
        # 直接在当前 step 的 Q/K/V 上做双向注意力，因此走独立的处理函数。
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

        # 中文注释：以下进入解码器/交叉注意力路径，需要使用 KV Cache。
        # kv_cache 的 shape 为 [num_blocks, 2, block_size, num_kv_heads, head_size]，
        # 其中 dim=1 的 2 分别是 key 和 value。unbind(1) 将其拆分为两个独立张量。
        # key_cache/value_cache 的 shape 为 [num_blocks, block_size, num_kv_heads, head_size]，
        # 对应物理显存中的分页 KV 存储。
        # For decoder and cross-attention, use KV cache as before
        key_cache, value_cache = kv_cache.unbind(1)
        # Fix degenerate strides on size-1 dims (e.g. num_kv_heads=1 with TP).
        # FA3/4 on H100+ uses TMA, which requires ≥16-byte stride alignment.
        # See vllm.utils.torch_utils.canonicalize_singleton_dim_strides.
        fixed_k = canonicalize_singleton_dim_strides(key_cache)
        fixed_v = canonicalize_singleton_dim_strides(value_cache)
        if fixed_k is not key_cache or fixed_v is not value_cache:
            logger.debug(
                "Canonicalized degenerate KV cache strides (FlashAttention): "
                "shape=%s, key strides before=%s after=%s, "
                "value strides before=%s after=%s",
                key_cache.shape,
                key_cache.stride(),
                fixed_k.stride(),
                value_cache.stride(),
                fixed_v.stride(),
            )
        key_cache, value_cache = fixed_k, fixed_v

        # 中文注释：如果 KV Cache 使用 FP8 量化存储，需要将 key_cache/value_cache
        # 的 dtype 视图切换为 FP8 类型，以便 FlashAttention 内核能正确读取量化后的数据。
        # 注意：这里只改变 dtype 视图，不改变底层数据。
        if is_quantized_kv_cache(self.kv_cache_dtype):
            # queries are quantized in the attention layer
            key_cache = key_cache.view(current_platform.fp8_dtype())
            value_cache = value_cache.view(current_platform.fp8_dtype())

        # 中文注释：根据是否使用级联注意力（cascade）分发到不同计算路径。
        # 大多数情况下 use_cascade=False，走标准的 flash_attn_varlen_func 路径。
        if not attn_metadata.use_cascade:
            cu_seqlens_q = attn_metadata.query_start_loc
            seqused_k = attn_metadata.seq_lens
            max_seqlen_q = attn_metadata.max_query_len
            max_seqlen_k = attn_metadata.max_seq_len
            block_table = attn_metadata.block_table
            scheduler_metadata = attn_metadata.scheduler_metadata

            descale_shape = (cu_seqlens_q.shape[0] - 1, self.num_kv_heads)

            q_descale = (
                layer._q_scale.expand(descale_shape)
                if self.supports_quant_query_input
                else None
            )
            k_descale = layer._k_scale.expand(descale_shape)
            v_descale = layer._v_scale.expand(descale_shape)

            # 中文注释：DCP（Decode Context Parallelism）路径。
            # 当 dcp_world_size > 1 时，KV Cache 被切分到多个 rank 上，
            # 需要通过跨 rank 通信来完成完整的注意力计算。
            if self.dcp_world_size > 1:
                self._forward_with_dcp(
                    query[:num_actual_tokens],
                    key[:num_actual_tokens],
                    value[:num_actual_tokens],
                    key_cache,
                    value_cache,
                    output[:num_actual_tokens],
                    attn_metadata,
                    q_descale=q_descale,
                    k_descale=k_descale,
                    v_descale=v_descale,
                )
                return output
            # 中文注释：标准解码器注意力路径（非 DCP、非级联）。
            # 这是最常见的路径，调用 flash_attn_varlen_func 执行分页注意力计算。
            # 关键参数说明：
            # - q: 本轮需要计算注意力的 query（仅包含实际 token）
            # - k/v: 从 KV Cache 中读取的 key/value（分页存储，不连续）
            # - cu_seqlens_q: 每个请求的 query 在拼接序列中的起止位置（cumulative sum）
            # - seqused_k: 每个请求实际使用的 KV 长度（受 prefix cache 命中影响）
            # - block_table: 逻辑 block 到物理 block 的映射表，用于分页寻址
            # - scheduler_metadata: FA3/4 的 TMA 调度元数据，优化内存访问模式
            else:
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
                    q_descale=q_descale,
                    k_descale=k_descale,
                    v_descale=v_descale,
                    num_splits=attn_metadata.max_num_splits,
                    s_aux=self.sinks,
                )
                return output

        # 中文注释：级联注意力路径（较少使用）。
        # 当多个请求共享较长的公共前缀时，级联注意力将计算拆分为
        # "公共前缀"和"各请求后缀"两部分，前缀只计算一次，所有请求共享结果。
        # 这可以显著减少重复计算，特别是在 system prompt 较长的场景。
        # Cascade attention (rare case).
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

    # do_kv_cache_update(): 将当前 step 产生的新 K/V 写入分页 KV Cache。
    # 使用 reshape_and_cache_flash 算子，通过 slot_mapping 索引进行 scatter write，
    # 将连续的 key/value 张量写入 paged cache 的对应物理槽位。
    # 编码器注意力不需要 KV Cache，直接跳过。
    # kv_sharing_target_layer_name 非空时也跳过（共享其他层的 KV Cache）。
    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        # 中文注释：编码器注意力不使用 KV Cache，Q/K/V 直接来自当前 step 的输入，
        # 因此无需更新缓存，直接返回。
        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            # For encoder attention,
            # we use direct Q, K, V tensors without caching
            return

        # 中文注释：将 KV Cache 拆分为 key_cache 和 value_cache。
        # kv_cache shape: [num_blocks, 2, block_size, num_kv_heads, head_size]
        # key_cache/value_cache shape: [num_blocks, block_size, num_kv_heads, head_size]
        # 这里不需要做 stride 修正，因为不涉及 TMA 内核调用。
        # Scatter write into the KV cache using slot_mapping indices.
        # No TMA kernel is invoked here, so stride canonicalization is not needed.
        key_cache, value_cache = kv_cache.unbind(1)

        # 中文注释：调用 reshape_and_cache_flash 将当前 step 产生的新 K/V 写入分页 KV Cache。
        # 这是一个 scatter write 操作：通过 slot_mapping 索引，将连续排列的 key/value
        # 张量写入 KV Cache 中对应的物理槽位（slot）。
        # slot_mapping 是一个一维张量，长度等于本轮实际 token 数，每个元素是一个
        # 全局 slot 索引，指向 KV Cache 中的 (block_idx, block_offset) 位置。
        # 例如：slot_mapping[i] = 515 表示第 i 个 token 的 KV 应写入
        #   block_idx = 515 // block_size, offset = 515 % block_size。
        # 这种设计使得不同请求的 KV Cache 在物理显存中不需要连续存放，
        # 实现了 PagedAttention 的分页管理。
        # Reshape the input keys and values and store them in the cache.
        # Skip this if sharing KV cache with an earlier attention layer.
        # NOTE(woosuk): Here, key and value are padded while slot_mapping is
        # not padded. However, we don't need to do key[:num_actual_tokens]
        # and value[:num_actual_tokens] because the reshape_and_cache_flash
        # op uses the slot_mapping's shape to determine the number of
        # actual tokens.
        reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
        )

    # _forward_with_dcp(): DCP（Decode Context Parallelism）模式的前向计算。
    # DCP 将长上下文的 KV Cache 按 rank 切分，每个 rank 只存储和计算一部分 KV。
    # 计算流程分三步：
    # 1. all_gather 查询：将 query 在所有 DCP rank 间做 all_gather，
    #    使每个 rank 拿到完整的 query（用于计算注意力时 Q 侧需要完整信息）。
    # 2. 上下文注意力：用 gather 后的 query 与本地 KV 切片计算注意力，
    #    得到 context_attn_out 和 context_lse。
    # 3. DCP combine：通过 dcp_combine（a2a 或 ag+rs）在 rank 间通信合并
    #    上下文注意力的输出和 LSE。
    # 4. 本地查询注意力：用原始 query 与本地 K/V（非缓存）计算因果注意力，
    #    捕获本地 token 间的依赖关系。
    # 5. 最终合并：使用 merge_attn_states() 合并上下文注意力和本地查询注意力的结果。
    def _forward_with_dcp(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        q_descale: torch.Tensor | None = None,
        k_descale: torch.Tensor | None = None,
        v_descale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert self.vllm_flash_attn_version is not None, (
            "FlashAttention version not detected."
        )

        # 中文注释：提取注意力元数据中的 query 序列信息。
        # cu_seqlens_q: 各请求 query 的 cumulative sequence lengths，
        #   用于 flash_attn_varlen_func 处理变长序列。
        # max_seqlen_q: 本轮 batch 中最长的 query 序列长度。
        # block_table: 逻辑 block 到物理 block 的映射表。
        cu_seqlens_q = attn_metadata.query_start_loc
        max_seqlen_q = attn_metadata.max_query_len
        block_table = attn_metadata.block_table

        # 中文注释：DCP 步骤 1 - all_gather 查询。
        # 将当前 rank 的 query 在所有 DCP rank 间做 all_gather，
        # 使每个 rank 拿到完整的 query。因为注意力计算需要每个 query token
        # 与所有 KV token 做点积，而 KV 被切分到不同 rank，所以 query 侧需要完整信息。
        # all_gather 后 query_across_dcp 的 num_tokens 维度扩大 dcp_world_size 倍。
        query = query.contiguous()
        query_across_dcp = get_dcp_group().all_gather(query, dim=1)
        sliding_window_size = (
            list(self.sliding_window) if self.sliding_window is not None else None
        )
        # 中文注释：从 workspace manager 预分配 DCP 上下文注意力的输出缓冲区。
        # 形状为 [num_gathered_tokens, num_heads * dcp_world_size, head_size]，
        # 因为 all_gather 后每个 rank 的 query 数量扩大了 dcp_world_size 倍，
        # 同时 attention heads 也对应扩大（每个 rank 处理一部分 KV 的注意力输出）。
        n = query_across_dcp.shape[0]
        (dcp_context_out,) = current_workspace_manager().get_simultaneous(
            (
                (n, self.num_heads * self.dcp_world_size, self.head_size),
                self._dcp_dtype,
            ),
        )
        # 中文注释：DCP 步骤 2 - 上下文注意力计算。
        # 使用 all_gather 后的完整 query 与本地 KV 切片计算注意力。
        # causal=False：因为这是对"已缓存的历史 KV"的注意力，不需要因果掩码。
        # return_softmax_lse=True：返回 log-sum-exp 值，用于后续跨 rank 合并。
        # dcp_context_kv_lens：DCP 模式下每个请求在当前 rank 上的 KV 长度。
        context_attn_out, context_lse = flash_attn_varlen_func(
            q=query_across_dcp,
            k=key_cache,
            v=value_cache,
            out=dcp_context_out,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            seqused_k=attn_metadata.dcp_context_kv_lens,
            max_seqlen_k=attn_metadata.max_dcp_context_kv_len,
            softmax_scale=self.scale,
            causal=False,
            alibi_slopes=self.alibi_slopes,
            window_size=sliding_window_size,
            block_table=block_table,
            softcap=self.logits_soft_cap,
            return_softmax_lse=True,
            scheduler_metadata=attn_metadata.scheduler_metadata,
            fa_version=self.vllm_flash_attn_version,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
            num_splits=attn_metadata.max_num_splits,
        )
        # 中文注释：DCP 步骤 3 - 跨 rank 合并上下文注意力结果。
        # dcp_combine 根据配置选择不同的通信原语：
        # - a2a (all-to-all)：直接交换注意力输出和 LSE
        # - ag+rs (all-gather + reduce-scatter)：先收集再归约
        # 合并后 context_attn_out_cor 包含所有 rank 的 KV 切片的加权注意力结果。
        # LSE (log-sum-exp) 用于数值稳定的 softmax 合并——
        # 不同 rank 的注意力分数需要基于 LSE 做加权平均，而非简单相加。
        # FA returns LSE in shape [ H, B ] but DCP combine wants [ B, H ]
        context_attn_out_cor, context_lse_cor = self.dcp_combine(
            context_attn_out,
            context_lse.transpose(0, 1),
            get_dcp_group(),
            return_lse=True,
        )
        context_lse_cor = context_lse_cor.transpose(0, 1).contiguous()

        # 中文注释：分配本地查询注意力的输出缓冲区。
        # 这里使用原始 query（未经 all_gather），形状与原始 query 一致。
        (dcp_query_out,) = current_workspace_manager().get_simultaneous(
            ((query.shape[0], self.num_heads, self.head_size), self._dcp_dtype),
        )
        # 中文注释：DCP 步骤 4 - 本地查询注意力计算。
        # 用原始 query 与本地 K/V（非缓存的、当前 step 产生的 K/V）计算因果注意力。
        # causal=True：因为 query 是当前 step 的 token，需要遵循因果掩码。
        # 这一步捕获的是"本地 token 之间的依赖关系"——即当前 step 内新产生的
        # token 之间的注意力交互，这部分 KV 不在远程 rank 的缓存中。
        # cu_seqlens_k=cu_seqlens_q：Q 和 K 来自同一序列（都是当前 step 的 token）。
        query_attn_out, query_lse = flash_attn_varlen_func(
            q=query,
            k=key,
            v=value,
            out=dcp_query_out,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            cu_seqlens_k=cu_seqlens_q,
            max_seqlen_k=max_seqlen_q,
            softmax_scale=self.scale,
            causal=attn_metadata.causal,
            alibi_slopes=self.alibi_slopes,
            window_size=sliding_window_size,
            softcap=self.logits_soft_cap,
            return_softmax_lse=True,
            fa_version=self.vllm_flash_attn_version,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
            num_splits=attn_metadata.max_num_splits,
        )
        # 中文注释：DCP 步骤 5 - 合并上下文注意力和本地查询注意力。
        # merge_attn_states 基于 LSE 做数值稳定的加权合并：
        # 最终输出 = (context_out * exp(context_lse) + query_out * exp(query_lse))
        #            / (exp(context_lse) + exp(query_lse))
        # 这确保了两部分注意力分数在 softmax 空间中的正确合并。
        assert context_attn_out_cor.shape == query_attn_out.shape
        assert context_lse_cor.shape == query_lse.shape
        merge_attn_states(
            output,
            context_attn_out_cor,
            context_lse_cor,
            query_attn_out,
            query_lse,
        )

    # _forward_encoder_attention(): 编码器注意力的前向计算。
    # 与解码器不同，编码器注意力不使用 KV Cache，直接在 Q/K/V 张量上做双向注意力
    # （causal=False），适用于 BERT-style 的编码器模型或 encoder-decoder 模型的编码器侧。
    # cu_seqlens_q 和 cu_seqlens_k 相同（因为 Q 和 K 来自同一序列）。
    def _forward_encoder_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        layer: torch.nn.Module,
    ) -> torch.Tensor:
        """Forward pass for encoder attention without KV cache.

        Args:
            query: shape = [num_encoder_tokens, num_heads, head_size]
            key: shape = [num_encoder_tokens, num_kv_heads, head_size]
            value: shape = [num_encoder_tokens, num_kv_heads, head_size]
            output: shape = [num_encoder_tokens, num_heads, head_size]
            attn_metadata: Encoder attention metadata
            layer: The attention layer
        """
        assert self.vllm_flash_attn_version is not None, (
            "FlashAttention version not detected."
        )

        # 中文注释：编码器注意力不支持 FP8 量化，因为编码器通常不需要 KV Cache，
        # 且 FP8 量化主要针对 KV Cache 的存储优化。
        # For encoder attention, process FP8 quantization if needed
        if is_quantized_kv_cache(self.kv_cache_dtype):
            raise NotImplementedError(
                "quantization is not supported for encoder attention"
            )

        # 中文注释：编码器注意力的序列信息。
        # 对于编码器，Q 和 K 来自同一序列（都是输入 token），
        # 因此 cu_seqlens_q 和 cu_seqlens_k 相同。
        # max_seqlen_q 和 max_seqlen_k 也相同，表示最长的输入序列长度。
        # Use encoder-specific metadata for sequence information
        cu_seqlens_q = attn_metadata.query_start_loc
        cu_seqlens_k = attn_metadata.query_start_loc
        max_seqlen_q = attn_metadata.max_query_len
        max_seqlen_k = attn_metadata.max_query_len

        descale_shape = (
            cu_seqlens_q.shape[0] - 1,  # type: ignore[union-attr]
            self.num_kv_heads,
        )

        # 中文注释：调用 flash_attn_varlen_func 执行编码器的双向注意力计算。
        # 关键区别：
        # - causal=False：编码器注意力是双向的，每个 token 可以看到所有其他 token。
        # - 不使用 block_table：编码器没有分页 KV Cache，直接使用输入的 Q/K/V。
        # - 不使用 scheduler_metadata：编码器不需要 TMA 调度优化。
        # - num_splits=1 if batch_invariant：确保批量一致性模式下结果可复现。
        # Call flash attention directly on Q, K, V tensors
        sliding_window_size = (
            list(self.sliding_window) if self.sliding_window is not None else None
        )
        flash_attn_varlen_func(
            q=query,
            k=key,
            v=value,
            out=output,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=self.scale,
            causal=False,  # Encoder attention is bidirectional
            alibi_slopes=self.alibi_slopes,
            window_size=sliding_window_size,
            softcap=self.logits_soft_cap,
            fa_version=self.vllm_flash_attn_version,
            q_descale=layer._q_scale.expand(descale_shape)
            if self.supports_quant_query_input
            else None,
            k_descale=layer._k_scale.expand(descale_shape),
            v_descale=layer._v_scale.expand(descale_shape),
            num_splits=1 if self.batch_invariant_enabled else 0,
        )

        return output


# use_cascade_attention(): 启发式决策函数，判断是否值得使用级联注意力。
# 级联注意力将注意力计算拆为"公共前缀"和"各请求后缀"两部分：
# - 公共前缀只计算一次，所有请求共享结果，避免重复计算；
# - 后缀部分按各请求独立计算。
# 适用场景：多个请求共享较长的公共前缀（如 system prompt）。
# 决策逻辑分两步：
# (1) 快速排除：公共前缀太短（<256 tokens）、ALiBi/滑动窗口/局部注意力、
#     请求数太少（<8）、DCP 模式下均不使用级联。
# (2) 性能模型比较：计算级联注意力和 FlashDecoding 各自需要的 CTA 数和 wave 数，
#     选择 GPU SM 利用率更高的方案。
def use_cascade_attention(
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
    """Decide whether to use cascade attention.

    This function 1) checks whether cascade attention is supported with the
    given configuration, and 2) heuristically decides whether using cascade
    attention can improve performance.
    """
    # Too short common prefix. Probably not worth using cascade attention.
    # We use an arbitrary threshold of 256 tokens. TODO: Tune this threshold.
    # NOTE(woosuk): This is the common case. We should return False as soon as
    # possible to avoid any unnecessary computation.
    if common_prefix_len < 256:
        return False
    # Cascade attention is currently not supported with these variants.
    if use_alibi or use_sliding_window or use_local_attention:
        return False
    # Too few queries. Probably not worth using cascade attention.
    # We use an arbitrary threshold of 8 queries. TODO: Tune this threshold.
    num_reqs = len(query_lens)
    if num_reqs < 8:
        return False
    # disable cascade attention for DCP
    if dcp_world_size > 1:
        return False

    # 中文注释：启发式性能模型——比较级联注意力和 FlashDecoding 的开销。
    # 核心思路：估算两种方案各自需要多少个 CTA（CUDA Thread Block）和
    # 需要多少个 wave 才能覆盖所有工作，选择 SM 利用率更高的方案。
    # Heuristics to decide whether using cascade attention is beneficial.
    # 1. When FlashDecoding is not used for normal attention, cascade attention
    #    is likely to be faster since it saves memory bandwidth.
    num_queries_per_kv = num_query_heads // num_kv_heads
    # The criteria for using FlashDecoding can be found in the following link:
    # https://github.com/vllm-project/flash-attention/blob/96266b1111111f3d11aabefaf3bacbab6a89d03c/csrc/flash_attn/flash_api.cpp#L535
    use_flash_decoding = (
        num_queries_per_kv > 1
        and not use_sliding_window
        and not use_alibi
        and np.all(query_lens == 1)
    )
    if not use_flash_decoding:
        # Use cascade attention.
        return True

    # 中文注释：当 FlashDecoding 可用时（GQA 场景，query_lens == 1），
    # 需要更精细的性能比较。FlashDecoding 可以为每个 KV head 启动独立的 CTA，
    # 并行度更高；而级联注意力需要先计算前缀再计算后缀。
    # 这里通过估算 CTA 数量和 wave 数来比较两种方案的 GPU 利用率。
    # 2. When FlashDecoding is used for normal attention, it is not clear
    #    whether cascade attention is beneficial, because FlashDecoding can
    #    launch more CTAs than cascade attention.
    #    We use a simple performance model to compare the two methods.
    #    NOTE(woosuk): The performance model is very rough and may not be
    #    accurate.
    num_tokens = num_reqs
    # NOTE(woosuk): These are default tile sizes. flash-attn might use
    # different tile sizes (e.g., 64 or 256) depending on the configuration.
    q_tile_size = 128
    kv_tile_size = 128
    # 中文注释：级联注意力的 CTA 数估算。
    # cascade_ctas = num_query_heads * ceil(num_tokens / q_tile_size)
    #   即每个 query head 需要 ceil(num_tokens / 128) 个 CTA。
    # cascade_waves = ceil(cascade_ctas / num_sms)
    #   需要多少个 wave 才能覆盖所有 CTA（num_sms 是 GPU 的 SM 数量）。
    # cascade_time = cascade_waves * num_prefix_tiles
    #   总时间正比于 wave 数乘以前缀 tile 数（因为需要扫描整个前缀）。
    num_prefix_tiles = cdiv(common_prefix_len, kv_tile_size)

    cascade_ctas = num_query_heads * cdiv(num_tokens, q_tile_size)
    cascade_waves = cdiv(cascade_ctas, num_sms)
    cascade_time = cascade_waves * num_prefix_tiles

    # 中文注释：FlashDecoding 的 CTA 数估算。
    # flash_decoding_ctas = num_reqs * num_kv_heads * ceil(num_queries_per_kv / q_tile_size)
    #   每个请求的每个 KV head 需要 ceil(GQA_ratio / 128) 个 CTA。
    # flash_decoding_ctas *= num_prefix_tiles
    #   再乘以前缀 tile 数（FlashDecoding 也需要扫描整个 KV 前缀）。
    # flash_decoding_time = ceil(flash_decoding_ctas / num_sms)
    #   总 wave 数。
    flash_decoding_ctas = (
        num_reqs * num_kv_heads * cdiv(num_queries_per_kv, q_tile_size)
    )
    flash_decoding_ctas *= num_prefix_tiles
    flash_decoding_time = cdiv(flash_decoding_ctas, num_sms)

    # 中文注释：选择 wave 数更少（即 GPU 利用率更高）的方案。
    # 级联注意力在请求多、前缀长时更有优势；FlashDecoding 在 GQA 比率高时更有优势。
    # Use cascade attention if it is faster than FlashDecoding.
    return cascade_time < flash_decoding_time


# cascade_attention(): 级联注意力的具体实现。
# 核心思路——将 KV 分为前缀（prefix）和后缀（suffix）两部分分别计算，再合并：
# 1. 前缀部分（prefix）：所有请求共享的公共 KV 前缀，使用 block_table[:1]（单一共享块表），
#    以 causal=False 计算，返回 prefix_output 和 prefix_lse（log-sum-exp）。
#    s_aux（sink tokens）在内核中被融入 prefix_lse，确保在最终合并时生效。
# 2. 后缀部分（suffix）：各请求独立的 KV 后缀，使用 block_table[:, num_common_kv_blocks:]
#    （跳过公共前缀的块），以 causal=True 计算，返回 suffix_output 和 suffix_lse。
# 3. 合并：使用 merge_attn_states() 基于 LSE 做数值稳定的加权合并。
# 这种设计的好处是公共前缀的 KV 只需加载和计算一次，显著减少带宽和算力消耗。
def cascade_attention(
    output: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cu_query_lens: torch.Tensor,
    max_query_len: int,
    cu_prefix_query_lens: torch.Tensor,
    prefix_kv_lens: torch.Tensor,
    suffix_kv_lens: torch.Tensor,
    max_kv_len: int,
    softmax_scale: float,
    alibi_slopes: torch.Tensor | None,
    sliding_window: tuple[int, int],
    logits_soft_cap: float,
    block_table: torch.Tensor,
    common_prefix_len: int,
    max_num_splits: int,
    fa_version: int,
    prefix_scheduler_metadata: torch.Tensor | None = None,
    suffix_scheduler_metadata: torch.Tensor | None = None,
    q_descale: torch.Tensor | None = None,
    k_descale: torch.Tensor | None = None,
    v_descale: torch.Tensor | None = None,
    s_aux: torch.Tensor | None = None,
) -> torch.Tensor:
    # 中文注释：级联注意力的前置条件检查。
    # 当前不支持 ALiBi 和滑动窗口，因为这两种注意力机制的掩码模式
    # 会破坏"前缀共享"的假设（不同请求的注意力掩码可能不同）。
    assert alibi_slopes is None, "Cascade attention does not support ALiBi."
    # TODO: Support sliding window.
    assert sliding_window == (-1, -1), (
        "Cascade attention does not support sliding window."
    )

    # 中文注释：计算公共前缀对应的 block 数量。
    # num_common_kv_blocks = common_prefix_len // block_size
    # 例如：common_prefix_len=1024, block_size=16 -> num_common_kv_blocks=64
    # 这些 block 是所有请求共享的，只计算一次。
    num_tokens = query.shape[0]
    block_size = key_cache.shape[-3]
    assert common_prefix_len % block_size == 0
    num_common_kv_blocks = common_prefix_len // block_size
    assert num_common_kv_blocks > 0
    descale_shape = (cu_prefix_query_lens.shape[0] - 1, key_cache.shape[-2])

    # 中文注释：级联注意力第 1 步——处理共享前缀。
    # 所有请求共享相同的公共前缀 KV，因此：
    # - block_table[:1]：只使用第一个请求的 block_table（因为前缀 block 对所有请求相同）
    # - causal=False：前缀部分不需要因果掩码（所有 token 都可以看到前缀中所有 token）
    # - prefix_kv_lens：每个请求在前缀部分的 KV 长度
    # - cu_prefix_query_lens：前缀注意力中每个请求的 query 起止位置
    # - s_aux：sink token 在内核中被融入 prefix_lse，确保最终合并时生效
    # 返回 prefix_output（注意力输出）和 prefix_lse（log-sum-exp 值），
    # 用于后续与后缀结果合并。
    # Process shared prefix.
    prefix_output, prefix_lse = flash_attn_varlen_func(
        q=query,
        k=key_cache,
        v=value_cache,
        cu_seqlens_q=cu_prefix_query_lens,
        seqused_k=prefix_kv_lens,
        max_seqlen_q=num_tokens,
        max_seqlen_k=common_prefix_len,
        softmax_scale=softmax_scale,
        causal=False,
        window_size=list(sliding_window),
        block_table=block_table[:1],
        softcap=logits_soft_cap,
        return_softmax_lse=True,
        scheduler_metadata=prefix_scheduler_metadata,
        fa_version=fa_version,
        q_descale=q_descale.expand(descale_shape) if q_descale is not None else None,
        k_descale=k_descale.expand(descale_shape) if k_descale is not None else None,
        v_descale=v_descale.expand(descale_shape) if v_descale is not None else None,
        # s_aux is incorporated into prefix_lse inside the GPU kernel,
        # enabling its effect during the final attention merge.
        s_aux=s_aux,
        num_splits=1 if envs.VLLM_BATCH_INVARIANT else max_num_splits,
    )

    descale_shape = (cu_query_lens.shape[0] - 1, key_cache.shape[-2])

    # 中文注释：级联注意力第 2 步——处理各请求独立的后缀。
    # 每个请求有自己独立的后缀 KV（不共享），因此：
    # - block_table[:, num_common_kv_blocks:]：跳过公共前缀的 block，
    #   只使用后缀部分的 block_table
    # - causal=True：后缀部分需要因果掩码（新 token 只能看到之前的 token）
    # - suffix_kv_lens：每个请求在后缀部分的 KV 长度
    # - max_seqlen_k=max_kv_len - common_prefix_len：后缀的最大长度
    # 返回 suffix_output 和 suffix_lse，与前缀结果合并。
    # Process suffix per query.
    suffix_output, suffix_lse = flash_attn_varlen_func(
        q=query,
        k=key_cache,
        v=value_cache,
        cu_seqlens_q=cu_query_lens,
        seqused_k=suffix_kv_lens,
        max_seqlen_q=max_query_len,
        max_seqlen_k=max_kv_len - common_prefix_len,
        softmax_scale=softmax_scale,
        causal=True,
        window_size=list(sliding_window),
        block_table=block_table[:, num_common_kv_blocks:],
        softcap=logits_soft_cap,
        return_softmax_lse=True,
        scheduler_metadata=suffix_scheduler_metadata,
        fa_version=fa_version,
        q_descale=q_descale.expand(descale_shape) if q_descale is not None else None,
        k_descale=k_descale.expand(descale_shape) if k_descale is not None else None,
        v_descale=v_descale.expand(descale_shape) if v_descale is not None else None,
        num_splits=1 if envs.VLLM_BATCH_INVARIANT else max_num_splits,
    )

    # 中文注释：级联注意力第 3 步——合并前缀和后缀的注意力输出。
    # merge_attn_states 基于 LSE（log-sum-exp）做数值稳定的加权合并：
    # 最终 attention = softmax([prefix_scores; suffix_scores]) @ [prefix_V; suffix_V]
    # 通过 LSE 合并避免了直接拼接 scores 再 softmax 的数值溢出问题。
    # 合并公式：output = (prefix_out * exp(prefix_lse) + suffix_out * exp(suffix_lse))
    #                    / (exp(prefix_lse) + exp(suffix_lse))
    # Merge prefix and suffix outputs, and store the result in output.
    merge_attn_states(output, prefix_output, prefix_lse, suffix_output, suffix_lse)
