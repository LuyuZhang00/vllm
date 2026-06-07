# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
FlashMLA 稀疏注意力后端模块。

本模块实现了基于 FlashMLA 的稀疏 MLA（Multi-Latent Attention）注意力后端，
主要用于 NVIDIA Hopper（SM90）和 Blackwell（SM100）GPU 上的 DeepSeek V3.2/V4 模型。

核心特性：
1. 稀疏注意力：只关注索引器（Indexer）选出的 topk 个最重要的 KV token
2. FP8 KV 缓存支持：支持 fp8_ds_mla 格式的量化 KV 缓存，大幅节省内存
3. 混合批处理：支持 decode 和 prefill 请求在同一 batch 中混合处理
4. CUDA Graph 兼容：元数据构建支持 CUDA Graph 以获得最佳性能

架构概述：
- FlashMLASparseBackend: 注意力后端定义（支持的 dtype、block size、设备能力等）
- FlashMLASparseMetadataBuilder: 元数据构建器（将调度器输出转换为内核所需格式）
- FlashMLASparseImpl: 注意力实现（执行实际的注意力计算）

FP8 KV 缓存格式（模块文档字符串中有详细说明）：
- DeepSeek V3.2: 每 token 656 字节（512B NoPE FP8 + 16B Scale + 128B RoPE BF16）
- DeepSeek V4: 每 token 584 字节（448B NoPE FP8 + 128B RoPE BF16 + 8B Scale）

两种 FP8 实现策略：
1. 混合批处理模式（Mixed Batch）：将所有 token 视为单个 batch，使用 FP8 decode 内核
   - 适用于 TP 场景（每 rank head 数较少，BF16 prefill 需要 padding 到 128 heads）
2. 分离 prefill/decode 模式（Separate）：prefill 用 BF16 内核，decode 用 FP8 内核
   - 适用于每 rank head 数 >= MIN_HEADS_FOR_BF16_PREFILL(32) 的场景
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import torch

from vllm import _custom_ops as ops
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.mla_attention import (
    get_mla_dims,
)
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import cdiv
from vllm.utils.platform_utils import num_compute_units
from vllm.utils.torch_utils import is_quantized_kv_cache
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionLayer,
    AttentionMetadata,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    MultipleOf,
    SparseMLAAttentionImpl,
)
from vllm.v1.attention.backends.mla.compressor_utils import get_compressed_slot_mapping
from vllm.v1.attention.backends.mla.sparse_utils import (
    triton_convert_req_index_to_global_index,
)
from vllm.v1.attention.backends.utils import (
    reshape_attn_output_for_spec_decode,
    reshape_query_for_spec_decode,
    split_decodes_and_prefills,
    split_prefill_chunks,
)
from vllm.v1.attention.ops.flashmla import (
    FlashMLASchedMeta,
    flash_mla_sparse_fwd,
    flash_mla_with_kvcache,
    get_mla_metadata,
)
from vllm.v1.kv_cache_interface import AttentionSpec
from vllm.v1.worker.workspace import current_workspace_manager

if TYPE_CHECKING:
    from vllm.model_executor.models.deepseek_v2 import Indexer

logger = init_logger(__name__)

# For FP8 sparse attention we have two implementations:
# 1. Mixed batch mode: use the FP8 decode kernel for both prefill and decode this is
#    done by treating all tokens as single batch.
# 2. Separate prefill and decode mode: use the BF16 prefill kernel for prefill
#    (upconverting the FP8 cache to BF16 then calling the prefill kernel) and using
#    the FP8 decode kernel for decode.
# Currently we use #1 when the number of heads per rank is low (i.e. TP) since the BF16
# prefill kernel requires padding the number of heads to 128 while the decode does not
# so when the per ranke head count is below MIN_HEADS_FOR_BF16_PREFILL we use the mixed
# batch mode (#2).
# 对于 FP8 稀疏注意力，有两种实现方式：
# 1. 混合批处理模式：将所有 token 视为单个 batch，使用 FP8 decode 内核
# 2. 分离 prefill/decode 模式：prefill 用 BF16 内核，decode 用 FP8 内核
# 当每 rank head 数较低时使用 #1，因为 BF16 prefill 内核需要 padding 到 128 heads
MIN_HEADS_FOR_BF16_PREFILL = 32

"""
NOTE: FlashMLA Sparse uses an fp8 cache with the following format

For DeepSeek V3.2, in the "FP8 with scale" format, each token's KV cache is 656
Bytes, structured as:
-   **First 512 bytes:** The "quantized NoPE" part, containing 512
    `float8_e4m3` values.
-   **Next 16 bytes:** Scale factors, containing 4 `float32` values.
    The first `float32` is the scale for the first 128 `float8_e4m3` values,
    the second for the next 128, and so on.
-   **Last 128 bytes:** The "RoPE" part, containing 64 `bfloat16` values. This
    part is not quantized for accuracy.

For DeepSeek V4, in the "FP8 with scale" format, each token's KV cache is 584
Bytes, structured as:
-   **First 448 bytes:** The "quantized NoPE" part, containing 448
    `float8_e4m3` values.
-   **Next 128 bytes:** The "RoPE" part, containing 64 `bfloat16` values. This
    part is not quantized for accuracy.
-   **Last 8 bytes:** Scale factors, containing 7 `ue8m0` values + 1B pad.
    The first `ue8m0` is the scale for the first 64 `float8_e4m3` values,
    the second for the next 64, and so on.

FlashMLA 稀疏注意力使用 FP8 缓存格式：
- DeepSeek V3.2: 每 token 656 字节（512B NoPE FP8 + 16B Scale + 128B RoPE BF16）
- DeepSeek V4: 每 token 584 字节（448B NoPE FP8 + 128B RoPE BF16 + 8B Scale）
"""


class FlashMLASparseBackend(AttentionBackend):
    """
    FlashMLA 稀疏注意力后端。

    定义了后端的基本属性和支持的配置，包括：
    - 支持的数据类型：BF16
    - 支持的 KV 缓存数据类型：auto, BF16, FP8_DS_MLA, FP8
    - 支持的块大小：64
    - 支持的 head size：576（V3.2）或 512（V4）
    - 支持的设备能力：SM90（Hopper）、SM100（Blackwell）
    """
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "bfloat16",
        "fp8_ds_mla",
        "fp8",  # fp8_ds_mla 的别名
    ]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        """返回支持的内核块大小列表。FlashMLA 稀疏内核固定使用块大小 64。"""
        return [64]

    @staticmethod
    def get_name() -> str:
        """返回后端名称，用于日志和配置识别。"""
        return "FLASHMLA_SPARSE"

    @staticmethod
    def get_builder_cls() -> type["FlashMLASparseMetadataBuilder"]:
        """返回元数据构建器类。"""
        return FlashMLASparseMetadataBuilder

    @staticmethod
    def get_impl_cls() -> type[SparseMLAAttentionImpl[Any]]:
        """返回注意力实现类。"""
        return FlashMLASparseImpl

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        """
        返回支持的 head size 列表。

        DeepSeek V3.2 布局：512 NoPE + 64 RoPE = 576。
        DeepSeek V4 使用 448 NoPE + 64 RoPE = 512，并在
        vllm/models/deepseek_v4/nvidia/flashmla.py 中重写此方法。
        """
        return [576]

    @classmethod
    def is_mla(cls) -> bool:
        """标识这是一个 MLA 注意力后端。"""
        return True

    @classmethod
    def is_sparse(cls) -> bool:
        """标识这是一个稀疏注意力后端。"""
        return True

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        """检查是否支持给定的 GPU 计算能力。SM90=Hopper, SM100=Blackwell。"""
        return capability.major in [9, 10]

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,  # 对于 MLA 假设为 1
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        """
        计算 KV 缓存的形状。

        对于 fp8_ds_mla 格式，使用自定义的 656 字节存储格式（见模块文档字符串）。
        对于其他格式，使用标准的 head_size 维度。
        """
        if cache_dtype_str == "fp8_ds_mla":
            # V3.2 主 MLA：656 字节自定义存储格式。参见模块文档字符串。
            return (num_blocks, block_size, 656)
        else:
            return (num_blocks, block_size, head_size)


@dataclass
class FlashMLASparseMetadata(AttentionMetadata):
    """
    FlashMLA 稀疏注意力的元数据。

    包含注意力计算所需的所有元数据，包括：
    - 基本信息：请求数量、最大 query 长度、最大序列长度
    - 位置映射：query_start_loc、slot_mapping、req_id_per_token
    - 块表：block_table
    - FP8 特有元数据：FP8KernelMetadata 或 FP8SeparatePrefillDecode
    - C128A 特有元数据（DeepSeekV4）：压缩后的 topk 索引

    属性说明：
    - num_reqs: 请求数量
    - max_query_len: 最大 query 长度
    - max_seq_len: 最大序列长度
    - num_actual_tokens: 实际 token 数量（不含 padding）
    - query_start_loc: 每个请求的 query 起始位置累积和
    - slot_mapping: KV 缓存 slot 映射
    - block_table: 物理块映射表
    - req_id_per_token: 每个 token 所属的请求 ID
    - block_size: 块大小（默认 64）
    - topk_tokens: 每个 token 选择的 topk 数量（默认 2048）
    """
    num_reqs: int
    max_query_len: int
    max_seq_len: int

    num_actual_tokens: int  # 不含 padding 的实际 token 数量
    query_start_loc: torch.Tensor
    slot_mapping: torch.Tensor

    block_table: torch.Tensor
    req_id_per_token: torch.Tensor
    block_size: int = 64
    topk_tokens: int = 2048

    @dataclass
    class FP8KernelMetadata:
        """FP8 内核元数据，用于混合批处理模式。"""
        scheduler_metadata: FlashMLASchedMeta  # FlashMLA 调度器元数据
        dummy_block_table: torch.Tensor  # 虚拟块表（FP8 内核使用 indices 参数时忽略）
        cache_lens: torch.Tensor  # 缓存长度

    @dataclass
    class FP8SeparatePrefillDecode:
        """FP8 分离 prefill/decode 模式的元数据。"""

        @dataclass
        class Decode:
            """Decode 部分的元数据。"""
            seq_lens: torch.Tensor  # 序列长度
            kernel_metadata: "FlashMLASparseMetadata.FP8KernelMetadata"  # FP8 内核元数据
            decode_query_len: int  # decode query 长度（spec decode 时需要用于 reshape）

        @dataclass
        class Prefill:
            """Prefill 部分的元数据。"""
            # 序列长度（context + query），形状：[num_prefill_reqs]
            seq_lens: torch.Tensor

            # 每个 token 的请求 ID：decode token 为 -1，prefill token 为请求索引 (0, 1, 2, ...)
            # 形状：[num_actual_tokens]
            request_ids: torch.Tensor

            # 所有 prefill 请求的 workspace 起始偏移
            # 形状：[num_prefill_reqs]，每个 chunk 中原地调整为 0 索引
            # 用于在 convert_logical_index_to_physical_index 中将 prefill token 映射到 workspace 偏移
            workspace_starts: torch.Tensor

            @dataclass
            class Chunk:
                """prefill 请求 chunk 的元数据。

                prefill 请求可能会被分块以适应固定的 workspace 大小。
                """
                seq_lens: torch.Tensor  # chunk 内的序列长度
                tokens_slice: slice  # chunk 对应的 token 范围
                block_table: torch.Tensor  # chunk 的块表
                req_start_idx: int  # chunk 在全局请求列表中的起始索引
                workspace_starts: torch.Tensor  # chunk 的 workspace 起始偏移
                chunk_tot_seqlen: int  # chunk 的总序列长度

            chunks: list[Chunk]  # chunk 列表

        num_prefills: int = 0  # prefill 请求数量
        num_decodes: int = 0  # decode 请求数量
        num_prefill_tokens: int = 0  # prefill token 数量
        num_decode_tokens: int = 0  # decode token 数量

        decode: Decode | None = None  # decode 元数据
        prefill: Prefill | None = None  # prefill 元数据

    fp8_extra_metadata: FP8SeparatePrefillDecode | FP8KernelMetadata | None = None
    fp8_use_mixed_batch: bool = False  # 是否使用混合批处理模式

    # 预计算的 C128A 元数据（仅 DeepseekV4，compress_ratio == 128）
    # Decode：全局 slot ID + 有效条目计数（从位置融合计算）
    c128a_global_decode_topk_indices: torch.Tensor | None = None
    c128a_decode_topk_lens: torch.Tensor | None = None
    # Prefill：本地 topk 索引（由 combine_topk_swa_indices 使用）
    c128a_prefill_topk_indices: torch.Tensor | None = None


def get_prefill_workspace_size(max_model_len: int):
    # NOTE(Lucas): 5 is a magic number for controlling the prefill buffer size.
    # May be tuned later.
    # Memory usage: 5 * max_model_len * 576 * 2 bytes
    #   Example: DeepSeek-V3.2 with max_model_len=163840 ->
    #            5 * 163840 * 576 * 2 = ~900 MB
    # This fits nicely below the typical MoE workspace size of >2GB so this is "free"
    # 计算 prefill workspace 大小。5 是控制 buffer 大小的魔法数字。
    return max_model_len * 5


class FlashMLASparseMetadataBuilder(AttentionMetadataBuilder[FlashMLASparseMetadata]):
    """
    FlashMLA 稀疏注意力的元数据构建器。

    负责将调度器输出的 CommonAttentionMetadata 转换为 FlashMLA 内核所需的
    FlashMLASparseMetadata。

    主要职责：
    1. 构建 req_id_per_token 映射（每个 token -> 所属请求 ID）
    2. 处理压缩 slot mapping（DeepSeekV4 compress_ratio > 1）
    3. 构建 FP8 元数据（混合批处理或分离 prefill/decode）
    4. 构建 C128A 元数据（DeepSeekV4 compress_ratio == 128）
    """
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        """
        初始化元数据构建器。

        预分配所有需要的缓冲区以支持 CUDA Graph 的地址稳定性。
        """
        self.vllm_config = vllm_config
        self.layer_names = layer_names
        cache_config = vllm_config.cache_config
        self.kv_cache_spec = kv_cache_spec
        self.model_config = vllm_config.model_config
        parallel_config = vllm_config.parallel_config
        self.device = device

        # 初始化批处理重排序阈值
        # 单 token query（加上 num_speculative_tokens 通过 supports_spec_as_decode=True）
        # 被归类为 decode；更长的 query 归类为 prefill。
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=True)

        sm_count = num_compute_units(device.index)

        self.num_heads = self.model_config.get_num_attention_heads(parallel_config)
        self.mla_dims = get_mla_dims(self.model_config)
        # FP8 decode 内核只支持 h_q = 64 或 128，所以需要 padding
        self.fp8_decode_padded_heads = (
            FlashMLASparseImpl._compute_fp8_decode_padded_heads(self.num_heads)
        )

        self.topk_tokens = vllm_config.model_config.hf_config.index_topk
        self.use_fp8_kv_cache = cache_config.cache_dtype == "fp8_ds_mla"
        max_num_seqs = vllm_config.scheduler_config.max_num_seqs
        # 形状：[max_num_seqs]，所有元素 = topk_tokens（用于完整 CUDA Graph 的常量）
        self.topk_tokens_tensor = torch.full(
            (max_num_seqs,), self.topk_tokens, device=device, dtype=torch.int32
        )
        # 形状：[max_num_seqs]，所有元素 = max_model_len
        self.max_model_len_tensor = torch.full(
            (max_num_seqs,),
            self.model_config.max_model_len,
            device=device,
            dtype=torch.int32,
        )
        # 当 indices 不为 None 时，flash_mla_with_kvcache 会忽略此值
        self.dummy_block_table = torch.empty(
            (max_num_seqs, 1), dtype=torch.int32, device=self.device
        )

        # 方程来自 FlashMLA/csrc/api/sparse_decode.h
        # 对于稀疏 FP8 decode，公式取决于架构：
        # - SM90（Hopper）：num_sm_parts = num_sms / s_q / (h_q/64)
        # - SM100（Blackwell head64/head64x2）：num_sm_parts = num_sms / s_q
        # - SM100（Blackwell head128）：num_sm_parts = num_sms / s_q / 2
        # 对于最大 buffer 大小，使用 s_q = 1（产生最大输出的情况）
        # 使用 padded head 计数，因为那是传递给内核的值
        h_q = self.fp8_decode_padded_heads
        if current_platform.is_device_capability_family(100):
            # SM100 head64 或 head64x2 使用完整 SM 数量
            max_num_sm_parts = sm_count
        else:
            # SM90 使用 h_q/64 除数
            max_num_sm_parts = sm_count // max(1, h_q // 64)
        self.tile_scheduler_metadata_buffer = torch.empty(
            # TileSchedulerMetaDataSize = 8
            # 参见：FlashMLA/csrc/params.h
            (max_num_sm_parts, 8),
            dtype=torch.int32,
            device=device,
        )
        # 为每请求批处理（num_decodes + 1）调整大小
        self.num_splits_buffer = torch.empty(
            (max_num_seqs + 1,),
            dtype=torch.int32,
            device=device,
        )
        self.req_id_per_token_buffer = torch.empty(
            (vllm_config.scheduler_config.max_num_batched_tokens,),
            dtype=torch.int32,
            device=device,
        )

        # DeepseekV4：hf_config 中有 compress_ratios。
        hf_config = vllm_config.model_config.hf_config
        self.is_deepseek_v4 = (
            hasattr(hf_config, "compress_ratios") and len(hf_config.compress_ratios) > 0
        )
        self.compress_ratio = 1
        if self.is_deepseek_v4:
            assert hasattr(self.kv_cache_spec, "compress_ratio")
            self.compress_ratio = self.kv_cache_spec.compress_ratio
            # 当 compress_ratio > 1 时，预分配压缩 slot mapping 缓冲区
            # 以支持 CUDA Graph 的地址稳定性。
            if self.compress_ratio > 1:
                max_num_batched_tokens = (
                    vllm_config.scheduler_config.max_num_batched_tokens
                )
                self.compressed_slot_mapping_buffer = torch.empty(
                    max_num_batched_tokens,
                    dtype=torch.int64,
                    device=self.device,
                )

            # 为 CUDA Graph 地址稳定性预分配 C128A topk 缓冲区。
            if self.compress_ratio == 128:
                max_num_batched_tokens = (
                    vllm_config.scheduler_config.max_num_batched_tokens
                )
                # 对齐到 B_TOPK（128 覆盖 h_q=64 B_TOPK=64 和 h_q=128 B_TOPK=128）
                # FlashMLA decode 断言 extra_topk % B_TOPK == 0；未对齐的宽度
                #（例如 17 = ceil(2136/128)）会导致 sm100 head64 内核崩溃。
                # 填充的 slot 保持 -1，decode_lens 通过 topk_length 限制它们，
                # 所以填充在内核级别是无操作的。
                # 与 cache_utils.py 中的 _SPARSE_PREFILL_TOPK_ALIGNMENT 对齐。
                _C128A_TOPK_ALIGNMENT = 128
                c128a_max_compressed = cdiv(
                    self.model_config.max_model_len, self.compress_ratio
                )
                c128a_max_compressed = (
                    cdiv(c128a_max_compressed, _C128A_TOPK_ALIGNMENT)
                    * _C128A_TOPK_ALIGNMENT
                )
                # 存储以便 _build_c128a_metadata 将其作为内核的 max_compressed_tokens 传递，
                # 匹配缓冲区步幅。否则内核默认的 8192 会迭代超出行宽度，
                # 导致写入溢出到相邻行（在 _build_c128a_topk_metadata_kernel 的
                # decode 和 prefill 分支中都存在）。
                self.c128a_max_compressed = c128a_max_compressed
                self.c128a_global_decode_buffer = torch.empty(
                    (max_num_batched_tokens, c128a_max_compressed),
                    dtype=torch.int32,
                    device=self.device,
                )
                self.c128a_decode_lens_buffer = torch.empty(
                    max_num_batched_tokens,
                    dtype=torch.int32,
                    device=self.device,
                )
                self.c128a_prefill_buffer = torch.empty(
                    (max_num_batched_tokens, c128a_max_compressed),
                    dtype=torch.int32,
                    device=self.device,
                )

    def _build_fp8_mixed_decode_prefill(
        self,
        common_attn_metadata: CommonAttentionMetadata,
    ) -> "FlashMLASparseMetadata.FP8KernelMetadata":
        """
        构建 FP8 混合 decode/prefill 元数据。

        将所有 token 视为单个混合 batch，匹配主分支的方法，避免使用
        BF16 prefill 内核（当 num_heads 较小时有 head padding 开销）。

        流程：
        1. 获取 padded head 数量（FP8 decode 内核只支持 64 或 128）
        2. 调用 get_mla_metadata 为所有 token 构建单一 batch 的调度器元数据
        3. 返回 FP8KernelMetadata（包含调度器元数据、虚拟块表、缓存长度）
        """
        num_tokens = common_attn_metadata.num_actual_tokens

        # 使用 padded head 计数，因为那是内核将看到的值
        padded_heads = self.fp8_decode_padded_heads

        # 将所有 token 构建为单一 batch 的元数据
        scheduler_metadata, _ = get_mla_metadata(
            cache_seqlens=self.topk_tokens_tensor[:1],  # 单一 batch
            num_q_tokens_per_head_k=num_tokens * padded_heads,
            topk=self.topk_tokens,
            num_heads_q=padded_heads,
            num_heads_k=1,
            is_fp8_kvcache=True,
        )

        fp8_metadata = FlashMLASparseMetadata.FP8KernelMetadata(
            scheduler_metadata=scheduler_metadata,
            cache_lens=self.max_model_len_tensor[:1],
            dummy_block_table=self.dummy_block_table[:1],
        )

        return fp8_metadata

    def _build_fp8_separate_prefill_decode(
        self,
        common_attn_metadata: CommonAttentionMetadata,
    ) -> "FlashMLASparseMetadata.FP8SeparatePrefillDecode":
        """
        构建 FP8 分离 prefill/decode 元数据。

        将 batch 拆分为 decode 和 prefill 两部分：
        - Decode：使用 FP8 decode 内核（不需要 head padding）
        - Prefill：使用 BF16 prefill 内核（需要将 FP8 缓存上转为 BF16）

        流程：
        1. 调用 split_decodes_and_prefills 拆分 batch
        2. 为 prefill 构建 request_id 映射和 workspace 偏移
        3. 将 prefill 请求分块以适应 workspace 大小
        4. 为 decode 构建 FP8 内核元数据
        """
        num_tokens = common_attn_metadata.num_actual_tokens

        (num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens) = (
            split_decodes_and_prefills(
                common_attn_metadata,
                decode_threshold=self.reorder_batch_threshold or 1,
                require_uniform=True,
            )
        )

        FP8Meta = FlashMLASparseMetadata.FP8SeparatePrefillDecode
        fp8_metadata = FP8Meta(
            num_decodes=num_decodes,
            num_prefills=num_prefills,
            num_decode_tokens=num_decode_tokens,
            num_prefill_tokens=num_prefill_tokens,
        )

        # 提取 prefill 序列长度（context + query，不仅仅是 query）
        # Decode 请求在 batch 中排在前面，prefill 请求在后面
        prefill_seq_lens = None
        prefill_request_id = None
        prefill_workspace_starts = None
        prefill_chunks = None

        # 对于纯 decode batch，prefill_request_id 将为 None
        # 对于混合 batch，decode 为 -1，prefill 为 request_id
        if num_prefills > 0:
            # 上界对于 prefill 行是精确的（下面的 `[num_decodes:]` 切片），
            # 所以不需要 D2H 同步。
            seq_lens_cpu = common_attn_metadata.seq_lens_cpu_upper_bound
            assert seq_lens_cpu is not None
            seq_lens = common_attn_metadata.seq_lens
            query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu

            prefill_seq_lens_cpu = seq_lens_cpu[num_decodes:]
            prefill_seq_lens = seq_lens[num_decodes:]

            # 构建 prefill_request_id：decode 为 -1，prefill 为请求索引
            # 这使得可以对所有 token 进行单次
            # convert_logical_index_to_physical_index 调用
            prefill_request_id = torch.full(
                (num_tokens,), -1, dtype=torch.int32, device=self.device
            )
            # 将 prefill token 映射到它们的请求 ID (0, 1, 2, ...)
            for req_idx in range(num_prefills):
                # 获取该 prefill 请求的 query token 范围
                global_req_idx = num_decodes + req_idx
                req_query_start = query_start_loc_cpu[global_req_idx]
                req_query_end = query_start_loc_cpu[global_req_idx + 1]
                prefill_request_id[req_query_start:req_query_end] = req_idx

            # 将由 chunk 循环调整
            prefill_workspace_starts_cpu = torch.zeros(
                num_prefills, dtype=torch.int32, pin_memory=True
            )
            prefill_workspace_starts_cpu[1:] = torch.cumsum(
                prefill_seq_lens_cpu[:-1], dim=0
            )
            # 在 prefill_workspace_starts_cpu 被每个 chunk 更新后，
            # 通过非阻塞拷贝填充
            prefill_workspace_starts = torch.empty(
                num_prefills, dtype=torch.int32, device=self.device
            )

            # 将 prefill 请求分块以适应 workspace 大小
            max_prefill_buffer_size = get_prefill_workspace_size(
                self.vllm_config.model_config.max_model_len
            )
            chunk_bounds = split_prefill_chunks(
                prefill_seq_lens_cpu, max_prefill_buffer_size
            )

            prefill_chunks = []
            for chunk_start, chunk_end in chunk_bounds:
                # 每个 chunk 中原地调整 workspace_starts 为 0 索引
                # 示例：seq_lens=[10,15,20,5], chunks=[[0,2],[2,4]]
                #   初始：workspace_starts=[0,10,25,45]
                #   调整后：workspace_starts=[0,10,0,20]
                #          （chunk 0 从 0 开始，chunk 1 从 0 开始）
                offset = prefill_workspace_starts_cpu[chunk_start].item()
                prefill_workspace_starts_cpu[chunk_start:chunk_end] -= offset

                chunk_seq_lens = prefill_seq_lens[chunk_start:chunk_end]
                chunk_tot_seqlen = prefill_seq_lens_cpu[chunk_start:chunk_end].sum()
                token_start = query_start_loc_cpu[num_decodes + chunk_start].item()
                token_end = query_start_loc_cpu[num_decodes + chunk_end].item()
                tokens_slice = slice(token_start, token_end)

                # 创建 chunk 的 GPU tensor 视图
                chunk_workspace_starts = prefill_workspace_starts[chunk_start:chunk_end]
                chunk_block_table = common_attn_metadata.block_table_tensor[
                    num_decodes + chunk_start : num_decodes + chunk_end
                ]

                prefill_chunks.append(
                    FP8Meta.Prefill.Chunk(
                        seq_lens=chunk_seq_lens,
                        tokens_slice=tokens_slice,
                        block_table=chunk_block_table,
                        req_start_idx=chunk_start,
                        workspace_starts=chunk_workspace_starts,
                        chunk_tot_seqlen=chunk_tot_seqlen,
                    )
                )

            prefill_workspace_starts.copy_(
                prefill_workspace_starts_cpu, non_blocking=True
            )

            fp8_metadata.prefill = FP8Meta.Prefill(
                seq_lens=prefill_seq_lens,
                request_ids=prefill_request_id,
                workspace_starts=prefill_workspace_starts,
                chunks=prefill_chunks,
            )

        if num_decodes > 0:
            # 为 spec decode 计算 decode_query_len（由于 require_uniform=True 是均匀的）
            query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
            decode_query_len = (query_start_loc_cpu[1] - query_start_loc_cpu[0]).item()

            # 使用 padded head 计数，因为那是内核将看到的值
            scheduler_metadata, _ = get_mla_metadata()

            kernel_meta = FlashMLASparseMetadata.FP8KernelMetadata(
                scheduler_metadata=scheduler_metadata,
                dummy_block_table=self.dummy_block_table[:num_decodes],
                cache_lens=self.max_model_len_tensor[:num_decodes],
            )
            fp8_metadata.decode = FP8Meta.Decode(
                seq_lens=common_attn_metadata.seq_lens[:num_decodes],
                kernel_metadata=kernel_meta,
                decode_query_len=decode_query_len,
            )

        return fp8_metadata

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> FlashMLASparseMetadata:
        """
        构建 FlashMLA 稀疏注意力的元数据。

        主要流程：
        1. 构建 req_id_per_token 映射
        2. 处理压缩 slot mapping（如果 compress_ratio > 1）
        3. 构建 FP8 元数据（如果使用 FP8 KV 缓存且非 DeepSeekV4）
        4. 构建 C128A 元数据（如果 DeepSeekV4 且 compress_ratio == 128）
        5. 组装并返回 FlashMLASparseMetadata
        """
        cm = common_attn_metadata
        num_tokens = cm.num_actual_tokens
        starts = np.asarray(cm.query_start_loc_cpu, dtype=np.int32)
        seg_lengths = np.diff(starts)
        req_id_per_token = np.repeat(
            np.arange(seg_lengths.shape[0], dtype=np.int32), seg_lengths
        )
        # 为 CUDA Graph 零填充
        self.req_id_per_token_buffer.fill_(0)
        self.req_id_per_token_buffer[: req_id_per_token.shape[0]].copy_(
            torch.from_numpy(req_id_per_token), non_blocking=True
        )
        req_id_per_token = self.req_id_per_token_buffer[:num_tokens]

        slot_mapping = cm.slot_mapping
        if self.compress_ratio > 1:
            slot_mapping = get_compressed_slot_mapping(
                common_attn_metadata.num_actual_tokens,
                common_attn_metadata.query_start_loc,
                common_attn_metadata.seq_lens,
                common_attn_metadata.block_table_tensor.clamp(min=0),
                int(self.kv_cache_spec.storage_block_size),
                self.compress_ratio,
                out=self.compressed_slot_mapping_buffer,
            )

        fp8_extra_metadata: (
            FlashMLASparseMetadata.FP8SeparatePrefillDecode
            | FlashMLASparseMetadata.FP8KernelMetadata
            | None
        ) = None
        fp8_use_mixed_batch = (
            self.num_heads < MIN_HEADS_FOR_BF16_PREFILL and not self.is_deepseek_v4
        )
        # DeepseekV4 有自己的注意力实现（DeepseekV4MLAAttention），不使用
        # fp8_extra_metadata。在此跳过构建可以避免在每个 prefill 步骤上触发
        # 强制的 D2H seq_lens 同步，将长 prefill 工作负载（如 LongBench）的
        # GPU 利用率从 ~83% 提升到 ~100%。
        if self.use_fp8_kv_cache and not self.is_deepseek_v4:
            if fp8_use_mixed_batch:
                fp8_extra_metadata = self._build_fp8_mixed_decode_prefill(cm)
            else:
                fp8_extra_metadata = self._build_fp8_separate_prefill_decode(cm)

        # 为 DeepseekV4 预计算 C128A topk 索引。
        c128a_fields = {}
        if self.is_deepseek_v4 and self.compress_ratio == 128:
            c128a_fields = self._build_c128a_metadata(cm, req_id_per_token)

        metadata = FlashMLASparseMetadata(
            num_reqs=cm.num_reqs,
            max_query_len=cm.max_query_len,
            max_seq_len=cm.max_seq_len,
            num_actual_tokens=cm.num_actual_tokens,
            query_start_loc=cm.query_start_loc,
            slot_mapping=slot_mapping,
            block_table=cm.block_table_tensor,
            req_id_per_token=req_id_per_token,
            block_size=self.kv_cache_spec.block_size,
            topk_tokens=self.topk_tokens,
            fp8_extra_metadata=fp8_extra_metadata,
            fp8_use_mixed_batch=fp8_use_mixed_batch,
            **c128a_fields,
        )

        return metadata

    def _build_c128a_metadata(
        self,
        cm: CommonAttentionMetadata,
        req_id_per_token: torch.Tensor,
    ) -> dict[str, torch.Tensor | None]:
        """为 DeepseekV4（compress_ratio >= 128）预计算 C128A topk 索引。"""
        # 必须匹配 SWA 的 decode 拆分（无 `require_uniform=True`），
        # 以便 `c128a_global_decode_topk_indices.shape[0]` 与
        # `_forward_decode` 中的 q 对齐。每 token 的 C128A 内核处理非均匀 query 长度。
        (num_decodes, _, num_decode_tokens, num_prefill_tokens) = (
            split_decodes_and_prefills(
                cm,
                decode_threshold=self.reorder_batch_threshold or 1,
            )
        )

        num_total = num_decode_tokens + num_prefill_tokens
        if num_total == 0:
            return {}

        assert cm.positions is not None, (
            "positions is required for C128A metadata build"
        )
        block_size = self.kv_cache_spec.block_size // self.compress_ratio
        global_decode, decode_lens, prefill_local = build_c128a_topk_metadata(
            cm.positions[:num_total],
            self.compress_ratio,
            num_decode_tokens,
            req_id_per_token,
            cm.block_table_tensor[:num_decodes],
            block_size,
            cm.slot_mapping,
            self.c128a_global_decode_buffer,
            self.c128a_decode_lens_buffer,
            self.c128a_prefill_buffer,
            max_compressed_tokens=self.c128a_max_compressed,
        )

        result: dict[str, torch.Tensor | None] = {}
        if num_decode_tokens > 0:
            result["c128a_global_decode_topk_indices"] = global_decode.view(
                num_decode_tokens, 1, -1
            )
            result["c128a_decode_topk_lens"] = decode_lens
        if num_prefill_tokens > 0:
            result["c128a_prefill_topk_indices"] = prefill_local
        return result


class FlashMLASparseImpl(SparseMLAAttentionImpl[FlashMLASparseMetadata]):
    """
    FlashMLA 稀疏注意力的具体实现。

    执行稀疏 MLA 注意力计算，支持：
    1. BF16 KV 缓存：使用 _forward_bf16_kv
    2. FP8 KV 缓存（混合批处理）：使用 _forward_fp8_kv_mixed_batch
    3. FP8 KV 缓存（分离 prefill/decode）：使用 _forward_fp8_kv_separate_prefill_decode
    """

    @staticmethod
    def _compute_fp8_decode_padded_heads(num_heads: int) -> int:
        """计算 FP8 decode 内核的 padded head 数量。FP8 decode 内核只支持 h_q = 64 或 128。"""
        return 64 if num_heads <= 64 else 128

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
        topk_indices_buffer: torch.Tensor | None = None,
        indexer: "Indexer | None" = None,
        **mla_args,
    ) -> None:
        """
        初始化 FlashMLA 稀疏注意力实现。

        参数：
        - num_heads: 注意力头数量
        - head_size: head 维度
        - scale: softmax 缩放因子
        - num_kv_heads: KV 头数量（MLA 为 1）
        - kv_cache_dtype: KV 缓存数据类型
        - indexer: 索引器实例，用于获取 topk_indices_buffer
        - mla_args: MLA 特有参数（kv_lora_rank 等）
        """
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.kv_lora_rank: int = mla_args["kv_lora_rank"]
        self.softmax_scale = scale
        assert indexer is not None
        self.topk_indices_buffer: torch.Tensor | None = indexer.topk_indices_buffer
        # Prefill BF16 内核要求 Hopper 上为 64，Blackwell 上为 128
        self.prefill_padding = (
            128 if current_platform.is_device_capability_family(100) else 64
        )
        self.fp8_decode_padded_heads = self._compute_fp8_decode_padded_heads(num_heads)

        vllm_config = get_current_vllm_config()
        max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        q_concat_shape = (max_tokens, num_heads, head_size)
        if is_quantized_kv_cache(kv_cache_dtype):
            assert kv_cache_dtype == "fp8_ds_mla", (
                "FlashMLA Sparse Attention backend fp8 only supports "
                "fp8_ds_mla kv-cache dtype"
            )

        if kv_cache_dtype == "fp8_ds_mla":
            # 在初始化期间预留 workspace
            assert vllm_config is not None and vllm_config.model_config is not None
            prefill_workspace_size = get_prefill_workspace_size(
                vllm_config.model_config.max_model_len
            )
            self.prefill_workspace_shape = (prefill_workspace_size, head_size)
            self.q_concat_buffer, self.prefill_bf16_workspace = (
                current_workspace_manager().get_simultaneous(
                    (q_concat_shape, torch.bfloat16),
                    (self.prefill_workspace_shape, torch.bfloat16),
                )
            )
        else:
            (self.q_concat_buffer,) = current_workspace_manager().get_simultaneous(
                (q_concat_shape, torch.bfloat16),
            )

    def _forward_bf16_kv(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        topk_indices: torch.Tensor,
        attn_metadata: FlashMLASparseMetadata,
    ) -> torch.Tensor:
        """
        BF16 KV 缓存的前向传播。

        流程：
        1. 将 per-request 索引转换为全局 slot（decode）或 workspace 偏移（prefill）
        2. 调用 BF16 FlashMLA 内核计算注意力
        """
        # 将 per-request 索引转换为全局 slot（decode）或 workspace 偏移（prefill）
        topk_indices = triton_convert_req_index_to_global_index(
            attn_metadata.req_id_per_token,
            attn_metadata.block_table,
            topk_indices,
            BLOCK_SIZE=attn_metadata.block_size,
            NUM_TOPK_TOKENS=topk_indices.shape[1],
        )

        return self._bf16_flash_mla_kernel(
            q,
            kv_c_and_k_pe_cache,
            topk_indices,
        )

    def _forward_fp8_kv_separate_prefill_decode(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        topk_indices: torch.Tensor,
        attn_metadata: FlashMLASparseMetadata,
    ) -> torch.Tensor:
        """
        FP8 KV 缓存的分离 prefill/decode 前向传播。

        流程：
        1. 将 per-request 索引转换为全局 slot 或 workspace 偏移
        2. Decode 部分：使用 FP8 decode 内核
        3. Prefill 部分：逐 chunk 将 FP8 缓存上转为 BF16，然后使用 BF16 prefill 内核
        4. 合并 decode 和 prefill 的输出
        """
        fp8_metadata = attn_metadata.fp8_extra_metadata
        assert isinstance(fp8_metadata, FlashMLASparseMetadata.FP8SeparatePrefillDecode)
        num_decodes = fp8_metadata.num_decodes

        prefill_request_ids = None
        prefill_workspace_starts = None
        has_prefill_workspace = False
        if fp8_metadata.prefill is not None:
            prefill_request_ids = fp8_metadata.prefill.request_ids
            prefill_workspace_starts = fp8_metadata.prefill.workspace_starts
            has_prefill_workspace = True

        # 将 per-request 索引转换为全局 slot（decode）或 workspace 偏移（prefill）
        # 对于 FP8 缓存：prefill 使用 workspace 映射（上转为 BF16）
        # 对于 BF16 缓存：始终使用全局缓存 slot（无 workspace）
        # prefill_workspace_starts 已在每个 chunk 中原地调整，所以
        # prefill 索引自动输出为 chunk 局部的
        topk_indices = triton_convert_req_index_to_global_index(
            attn_metadata.req_id_per_token,
            attn_metadata.block_table,
            topk_indices,
            BLOCK_SIZE=attn_metadata.block_size,
            NUM_TOPK_TOKENS=topk_indices.shape[1],
            HAS_PREFILL_WORKSPACE=has_prefill_workspace,
            prefill_workspace_request_ids=prefill_request_ids,
            prefill_workspace_starts=prefill_workspace_starts,
        )

        fp8_metadata = attn_metadata.fp8_extra_metadata
        assert isinstance(fp8_metadata, FlashMLASparseMetadata.FP8SeparatePrefillDecode)

        def _fp8_decode(
            q: torch.Tensor,
            topk_indices: torch.Tensor,
        ) -> torch.Tensor:
            """FP8 decode 内核调用。"""
            # 重塑 q: (num_decode_tokens, num_heads, head_dim)
            #        -> (num_decodes, seq_len, num_heads, head_dim)
            q = reshape_query_for_spec_decode(q, num_decodes)
            seq_len = q.shape[1]
            # 重塑 topk_indices: (num_decode_tokens, topk)
            #                  -> (num_decodes, seq_len, topk)
            topk_indices = topk_indices.view(num_decodes, seq_len, -1)
            assert fp8_metadata.decode is not None
            attn_out, _ = self._fp8_flash_mla_kernel(
                q=q,
                kv_c_and_k_pe_cache=kv_c_and_k_pe_cache,
                topk_indices=topk_indices,
                kernel_metadata=fp8_metadata.decode.kernel_metadata,
            )
            # 重塑输出: (num_decodes, seq_len, num_heads, head_dim_v)
            #        -> (num_decode_tokens, num_heads, head_dim_v)
            return reshape_attn_output_for_spec_decode(attn_out)

        num_decode_tokens = fp8_metadata.num_decode_tokens
        num_prefill_tokens = fp8_metadata.num_prefill_tokens

        # 纯 decode：直接调用，无需分配
        if num_decode_tokens > 0 and num_prefill_tokens == 0:
            assert fp8_metadata.decode is not None
            attn_out = _fp8_decode(q, topk_indices)
        else:
            # 混合或纯 prefill：分配输出 tensor
            attn_out = q.new_empty(
                (attn_metadata.num_actual_tokens, self.num_heads, self.kv_lora_rank),
                dtype=q.dtype,
                device=q.device,
            )

            if num_decode_tokens > 0:
                attn_out[:num_decode_tokens] = _fp8_decode(
                    q[:num_decode_tokens],
                    topk_indices[:num_decode_tokens],
                )

            assert fp8_metadata.prefill is not None
            for chunk in fp8_metadata.prefill.chunks:
                # 将 FP8 缓存上转为 BF16 到 workspace
                chunk_workspace = self.prefill_bf16_workspace[: chunk.chunk_tot_seqlen]
                ops.cp_gather_and_upconvert_fp8_kv_cache(
                    kv_c_and_k_pe_cache,
                    chunk_workspace,
                    chunk.block_table,
                    chunk.seq_lens,
                    chunk.workspace_starts,
                    len(chunk.block_table),
                )

                chunk_q = q[chunk.tokens_slice]
                chunk_topk_indices_workspace = topk_indices[chunk.tokens_slice]

                # 使用 BF16 prefill 内核计算注意力
                attn_out[chunk.tokens_slice] = self._bf16_flash_mla_kernel(
                    chunk_q,
                    chunk_workspace,
                    chunk_topk_indices_workspace,
                )

        return attn_out

    def _forward_fp8_kv_mixed_batch(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        topk_indices: torch.Tensor,
        attn_metadata: FlashMLASparseMetadata,
    ) -> torch.Tensor:
        """
        FP8 混合批处理前向路径，将所有 token 视为单一 batch。

        等同于主分支的方法，避免使用 BF16 prefill 内核
        （当 num_heads 较小时有 head padding 开销）。
        当 use_mixed_batch 为 True 时使用。

        流程：
        1. 将 per-request 索引转换为全局 slot
        2. 调用 FP8 FlashMLA 内核（所有 token 作为单一 batch）
        3. 返回注意力输出
        """
        # 将 per-request 索引转换为全局 slot（decode）或 workspace 偏移（prefill）
        topk_indices = triton_convert_req_index_to_global_index(
            attn_metadata.req_id_per_token,
            attn_metadata.block_table,
            topk_indices,
            BLOCK_SIZE=attn_metadata.block_size,
            NUM_TOPK_TOKENS=topk_indices.shape[1],
        )

        assert attn_metadata.fp8_extra_metadata is not None
        assert isinstance(
            attn_metadata.fp8_extra_metadata, FlashMLASparseMetadata.FP8KernelMetadata
        )
        fp8_metadata = attn_metadata.fp8_extra_metadata

        _attn_out, _ = self._fp8_flash_mla_kernel(
            q=q.unsqueeze(0),  # 添加 batch 维度: (T, H, D) -> (1, T, H, D)
            kv_c_and_k_pe_cache=kv_c_and_k_pe_cache,
            topk_indices=topk_indices.unsqueeze(0),  # (T, topk) -> (1, T, topk)
            kernel_metadata=fp8_metadata,
        )

        # 输出为 (1, T, H, D_v)，压缩回 (T, H, D_v)
        return _attn_out.squeeze(0)

    def _fp8_flash_mla_kernel(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        topk_indices: torch.Tensor,
        kernel_metadata: FlashMLASparseMetadata.FP8KernelMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        FP8 FlashMLA 内核调用。

        流程：
        1. 如果实际 head 数 < padded head 数，padding query
        2. 调用 flash_mla_with_kvcache 计算注意力
        3. 如果进行了 padding，裁切输出回实际 head 数
        """
        # q 形状: (batch, seq_len, num_heads, head_dim)
        actual_num_heads = q.size(2)
        padded_num_heads = self.fp8_decode_padded_heads

        # 如果需要，padding query（内核只支持 h_q = 64 或 128）
        if actual_num_heads < padded_num_heads:
            logger.warning_once(
                f"Padding num_heads from {actual_num_heads} to "
                f"{padded_num_heads} for FP8 sparse decode kernel"
            )
            q_padded = q.new_zeros((q.size(0), q.size(1), padded_num_heads, q.size(3)))
            q_padded[:, :, :actual_num_heads, :] = q
            q = q_padded

        out, lse = flash_mla_with_kvcache(
            q=q,
            k_cache=kv_c_and_k_pe_cache.view(torch.uint8).unsqueeze(-2),
            block_table=kernel_metadata.dummy_block_table,
            head_dim_v=512,
            cache_seqlens=kernel_metadata.cache_lens,
            tile_scheduler_metadata=kernel_metadata.scheduler_metadata,
            is_fp8_kvcache=True,
            indices=topk_indices,
            softmax_scale=self.softmax_scale,
        )

        # 如果进行了 padding，裁切输出回实际 head 数
        if actual_num_heads < padded_num_heads:
            out = out[:, :, :actual_num_heads, :]

        return out, lse

    def _bf16_flash_mla_kernel(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        topk_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        BF16 FlashMLA 稀疏前向内核调用。

        流程：
        1. 如果需要，padding query head 数到对齐要求
        2. 调用 flash_mla_sparse_fwd 计算注意力
        3. 裁切输出回实际 head 数
        """
        num_tokens = q.shape[0]
        kv_c_and_k_pe_cache = kv_c_and_k_pe_cache.view(
            -1, 1, kv_c_and_k_pe_cache.shape[-1]
        )

        # NOTE(Chen): kernel requires num_local_head to be a multiple of
        # 64 on hopper and 128 on blackwell
        # 内核要求 num_local_head 在 Hopper 上是 64 的倍数，在 Blackwell 上是 128 的倍数
        if self.num_heads % self.prefill_padding != 0:
            assert self.prefill_padding % self.num_heads == 0
            logger.warning_once(
                f"Padding num_heads from {self.num_heads} to "
                f"{self.prefill_padding} for BF16 sparse prefill kernel"
            )
            q_padded = q.new_empty((q.shape[0], self.prefill_padding, q.shape[2]))
            q_padded[:, : self.num_heads, :] = q
            q = q_padded

        topk_indices = topk_indices.view(num_tokens, 1, -1)
        output = flash_mla_sparse_fwd(q, kv_c_and_k_pe_cache, topk_indices, self.softmax_scale)[0]

        output = output[:, : self.num_heads, :]
        return output

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: FlashMLASparseMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        MLA 稀疏注意力的主前向传播入口。

        对于稀疏 FlashMLA 内核，prefill 和 decode 都使用 MQA 576/512 方法。

        流程：
        1. 如果 q 是元组（ql_nope, q_pe），拼接为完整 q
        2. 获取 topk 索引
        3. 根据 KV 缓存类型选择前向路径：
           - BF16: _forward_bf16_kv
           - FP8 混合批处理: _forward_fp8_kv_mixed_batch
           - FP8 分离: _forward_fp8_kv_separate_prefill_decode
        """
        # NOTE(lucas): for the sparse FlashMLA kernels the kernels want to use
        # MQA 576/512 approach for both prefill and decode
        # 对于稀疏 FlashMLA 内核，prefill 和 decode 都使用 MQA 576/512 方法

        # 如果 q 是元组（ql_nope, q_pe），拼接
        if isinstance(q, tuple):
            ql_nope, q_pe = q
            q = self.q_concat_buffer[: ql_nope.shape[0]]
            ops.concat_mla_q(ql_nope, q_pe, q)

        num_actual_toks = q.shape[0]

        # 获取 topk 索引
        assert self.topk_indices_buffer is not None
        topk_indices = self.topk_indices_buffer[:num_actual_toks]

        use_fp8_cache = self.kv_cache_dtype == "fp8_ds_mla"

        if not use_fp8_cache:
            attn_out = self._forward_bf16_kv(
                q, kv_c_and_k_pe_cache, topk_indices, attn_metadata
            )
        elif attn_metadata.fp8_use_mixed_batch:
            attn_out = self._forward_fp8_kv_mixed_batch(
                q, kv_c_and_k_pe_cache, topk_indices, attn_metadata
            )
        else:
            attn_out = self._forward_fp8_kv_separate_prefill_decode(
                q, kv_c_and_k_pe_cache, topk_indices, attn_metadata
            )

        return attn_out, None


def build_c128a_topk_metadata(
    positions: torch.Tensor,
    compress_ratio: int,
    num_decode_tokens: int,
    token_to_req_indices: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    slot_mapping: torch.Tensor,
    global_decode_buffer: torch.Tensor,
    decode_lens_buffer: torch.Tensor,
    prefill_buffer: torch.Tensor,
    max_compressed_tokens: int = 8192,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    为所有 C128A token（decode + prefill）构建 topk 元数据的单一内核。

    Decode token：position -> block_table 查找 -> 全局 slot ID + topk_lens。
    Prefill token：position -> 本地索引 [0, ..., n-1, -1, ...]。

    写入预分配的缓冲区以支持 CUDA Graph 地址稳定性。
    返回缓冲区的切片。

    参数：
    - positions: token 位置 [num_tokens]
    - compress_ratio: 压缩比率（128）
    - num_decode_tokens: decode token 数量
    - token_to_req_indices: 每个 token 的请求索引
    - block_table: 块表 [num_decode_reqs, max_num_blocks]
    - block_size: 压缩后的块大小
    - slot_mapping: slot 映射
    - global_decode_buffer: decode 输出缓冲区
    - decode_lens_buffer: decode 长度输出缓冲区
    - prefill_buffer: prefill 输出缓冲区
    - max_compressed_tokens: 最大压缩 token 数
    """
    num_tokens = positions.shape[0]
    num_prefill_tokens = num_tokens - num_decode_tokens

    global_decode = global_decode_buffer[:num_decode_tokens]
    decode_lens = decode_lens_buffer[:num_decode_tokens]
    prefill_local = prefill_buffer[:num_prefill_tokens]

    if num_tokens == 0:
        return global_decode, decode_lens, prefill_local

    _build_c128a_topk_metadata_kernel[(num_tokens,)](
        global_decode_buffer,
        global_decode_buffer.stride(0),
        decode_lens_buffer,
        prefill_buffer,
        prefill_buffer.stride(0),
        positions,
        compress_ratio,
        max_compressed_tokens,
        num_decode_tokens,
        token_to_req_indices,
        block_table,
        block_table.stride(0),
        block_size,
        slot_mapping,
        BLOCK_SIZE=1024,
    )
    return global_decode, decode_lens, prefill_local


@triton.jit
def _build_c128a_topk_metadata_kernel(
    # Decode 输出
    global_decode_ptr,
    global_decode_stride,
    decode_lens_ptr,
    # Prefill 输出
    prefill_local_ptr,
    prefill_local_stride,
    # 输入
    positions_ptr,
    compress_ratio,
    max_compressed_tokens,
    num_decode_tokens,
    token_to_req_indices_ptr,
    block_table_ptr,
    block_table_stride,
    block_size,
    slot_mapping_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton JIT 内核：为 C128A token 构建 topk 元数据。

    每个 Triton 程序处理一个 token。根据 token 类型（decode 或 prefill）：
    - Decode：通过 position -> block_table 查找 -> 生成全局 slot ID 列表
    - Prefill：生成本地索引 [0, 1, ..., n-1, -1, -1, ...]

    C128A 压缩机制：
    - 压缩比率 = 128，即每 128 个原始 token 压缩为 1 个
    - 对于位置 pos 的 token，压缩后有 (pos + 1) // 128 个条目
    - 每个条目对应一个压缩后的 KV 缓存位置
    """
    token_idx = tl.program_id(0)
    position = tl.load(positions_ptr + token_idx)
    num_compressed = (position + 1) // compress_ratio
    num_compressed = tl.minimum(num_compressed, max_compressed_tokens)
    is_decode = token_idx < num_decode_tokens

    if is_decode:
        # --- Decode：block-table 查找 -> 全局 slot ID + 计数 ---
        is_valid_token = tl.load(slot_mapping_ptr + token_idx) >= 0
        req_idx = tl.load(token_to_req_indices_ptr + token_idx)
        count = tl.zeros((), dtype=tl.int32)
        for i in range(0, max_compressed_tokens, BLOCK_SIZE):
            offset = i + tl.arange(0, BLOCK_SIZE)
            mask = offset < max_compressed_tokens
            is_valid = offset < num_compressed

            block_indices = offset // block_size
            block_numbers = tl.load(
                block_table_ptr + req_idx * block_table_stride + block_indices,
                mask=mask & is_valid,
            )
            block_offsets = offset % block_size
            slot_ids = block_numbers * block_size + block_offsets
            slot_ids = tl.where(is_valid, slot_ids, -1)
            tl.store(
                global_decode_ptr + token_idx * global_decode_stride + offset,
                slot_ids,
                mask=mask,
            )
            count += tl.sum(is_valid.to(tl.int32), axis=0)

        tl.store(
            decode_lens_ptr + token_idx,
            tl.where(is_valid_token, count, 0),
        )
    else:
        # --- Prefill：写入本地索引 ---
        pfx_idx = token_idx - num_decode_tokens
        for i in range(0, max_compressed_tokens, BLOCK_SIZE):
            offset = i + tl.arange(0, BLOCK_SIZE)
            mask = offset < max_compressed_tokens
            tl.store(
                prefill_local_ptr + pfx_idx * prefill_local_stride + offset,
                tl.where(offset < num_compressed, offset, -1),
                mask=mask,
            )
