# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
ROCm AITER MLA 注意力后端模块。

本模块实现了基于 AMD AITER（AI Tensor Engine for ROCm）的 MLA 注意力后端，
用于 AMD GPU（gfx950 等）上的 DeepSeek 模型推理。

核心特性：
1. 使用 AITER 库的 MLA decode 内核（mla_decode_fwd）
2. 支持 FP8 KV 缓存（fp8, fp8_e4m3, fp8_e5m2）
3. 支持 FP8 MLA prefill（gfx950 上的汇编内核 mla_prefill_ps_asm_fwd）
4. 支持持久化 MLA 元数据以优化 CUDA Graph 性能
5. 支持 MTP（Multi-Token Prediction）多 token decode

架构概述：
- AiterMLABackend: 后端定义（支持的 dtype、block size 等）
- AiterMLAMetadataBuilder: 元数据构建器
- AiterMLAImpl: 注意力实现
- AiterMLAHelper: 辅助类（处理 head padding）

关键实现细节：
- AITER 内核内部始终使用 page_size=1（通过 .view(-1,1,1,H) 展平 KV buffer）
- 当 num_heads < 16 时，需要 padding 到 16（AITER 最低要求 16 heads）
- FP8 prefill 使用两阶段流水线：mla_prefill_ps_asm_fwd + mla_reduce_v1
"""

import functools
from dataclasses import dataclass
from typing import ClassVar, Final

import torch

from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.mla_attention import (
    MLACommonBackend,
    MLACommonDecodeMetadata,
    MLACommonImpl,
    MLACommonMetadata,
    MLACommonMetadataBuilder,
    QueryLenSupport,
)
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    AttentionLayer,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.kv_cache_interface import AttentionSpec

logger = init_logger(__name__)


@functools.lru_cache(maxsize=1)
def _fp8_mla_prefill_supported() -> bool:
    """
    自动检测是否支持 FP8 MLA prefill（通过 mla_prefill_ps_asm_fwd + mla_reduce_v1）。

    需要 gfx950 GPU 以及导出两个内核的 AITER 构建。当缺少任何一个时，
    静默回退到 flash_attn_varlen_func。

    返回：
    - True: 支持 FP8 MLA prefill
    - False: 不支持，回退到标准 prefill
    """
    try:
        from vllm.platforms.rocm import on_gfx950
    except Exception:  # noqa: BLE001
        return False
    if not on_gfx950():
        return False
    try:
        from aiter import mla_prefill_ps_asm_fwd, mla_reduce_v1  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


class AiterMLABackend(MLACommonBackend):
    """
    ROCm AITER MLA 注意力后端。

    定义了后端的基本属性和支持的配置。
    """
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
        "fp8_e5m2",
    ]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        """AITER MLA 不通过 head size 限制，而是通过内核内部处理。"""
        return []

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        """
        AITER MLA decode 内核内部始终使用 page_size=1
        （包装器通过 .view(-1, 1, 1, H) 展平 kv_buffer）。
        我们支持任何 kernel_block_size，方法是在元数据构建器中将块级索引
        展开为每 token 的平坦索引。
        """
        return [MultipleOf(1)]

    @staticmethod
    def get_name() -> str:
        return "ROCM_AITER_MLA"

    @staticmethod
    def get_impl_cls() -> type["AiterMLAImpl"]:
        return AiterMLAImpl

    @staticmethod
    def get_builder_cls() -> type["AiterMLAMetadataBuilder"]:
        return AiterMLAMetadataBuilder


@dataclass
class AiterMLADecodeMetadata(MLACommonDecodeMetadata):
    """
    AITER MLA decode 元数据。

    包含 paged KV 缓存的索引结构，用于 AITER 内核。

    属性：
    - paged_kv_indptr: paged KV 缓存的 indptr，形状 [batch_size + 1]
    - paged_kv_indices: paged KV 缓存的页面索引
    - paged_kv_last_page_len: 每个请求在 paged KV 缓存中最后一页的条目数，形状 [batch_size]
    - qo_indptr: query indptr，形状 [num_decode + 1]
    - attn_out_dtype: MLA 输出 tensor 的数据类型
    - max_qo_len: 最大 query 输出长度
    - has_persistent_metadata: 是否计算了持久化 MLA 元数据（仅 qseqlen=1 时）
    """
    paged_kv_indptr: torch.Tensor | None = None
    paged_kv_indices: torch.Tensor | None = None
    paged_kv_last_page_len: torch.Tensor | None = None
    qo_indptr: torch.Tensor | None = None
    attn_out_dtype: torch.dtype = torch.bfloat16
    max_qo_len: int | None = None
    has_persistent_metadata: bool = False


@dataclass
class AiterMLAMetadata(MLACommonMetadata[AiterMLADecodeMetadata]):
    """
    AITER MLA 的完整元数据。

    包含 decode 和 prefill 两部分的元数据，以及 AITER 特有的工作缓冲区。
    """
    work_meta_data: torch.Tensor | None = None
    work_indptr: torch.Tensor | None = None
    work_info_set: torch.Tensor | None = None
    reduce_indptr: torch.Tensor | None = None
    reduce_final_map: torch.Tensor | None = None
    reduce_partial_map: torch.Tensor | None = None

    # FP8 ASM prefill 持久化调度（PS）元数据。当 prefill token 存在且设备支持
    # FP8 MLA prefill 时，由 AiterMLAMetadataBuilder._build_fp8_prefill_ps_metadata 填充。
    # 在不支持的主机/配置上设为 None，回退到 flash_attn_varlen_func。
    fp8_prefill_qo_indptr: torch.Tensor | None = None
    fp8_prefill_kv_indptr: torch.Tensor | None = None
    fp8_prefill_kv_indices: torch.Tensor | None = None
    fp8_prefill_work_indptr: torch.Tensor | None = None
    fp8_prefill_work_info_set: torch.Tensor | None = None
    fp8_prefill_reduce_indptr: torch.Tensor | None = None
    fp8_prefill_reduce_final_map: torch.Tensor | None = None
    fp8_prefill_reduce_partial_map: torch.Tensor | None = None
    fp8_prefill_max_q_len: int | None = None
    fp8_prefill_num_partial_tiles: int | None = None


# mla_prefill_ps_asm_fwd 汇编内核使用的 tile 大小
_FP8_PREFILL_TILE_Q = 256


class AiterMLAMetadataBuilder(MLACommonMetadataBuilder[AiterMLAMetadata]):
    """
    AITER MLA 的元数据构建器。

    负责构建 AITER 内核所需的元数据，包括：
    1. paged KV 缓存的索引结构（indptr, indices, last_page_len）
    2. 持久化 MLA 元数据（用于 work-stealing 路径）
    3. FP8 prefill PS 元数据（gfx950 上）
    """
    # TODO(luka, lucas): audit this as part of:
    #  https://github.com/vllm-project/vllm/issues/22945
    # TODO(luka, lucas): 审计此设置
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH
    query_len_support: ClassVar[QueryLenSupport] = QueryLenSupport.UNIFORM

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(
            kv_cache_spec, layer_names, vllm_config, device, AiterMLAMetadata
        )

        self.compilation_config = vllm_config.compilation_config
        self.decode_attn_out_dtype = vllm_config.model_config.dtype

        # 存储 spec 的内核块大小。当 kernel_block_size=1（无 spec-dec）时，
        # 行为与原始相同。当 > 1（例如 Eagle3 的 16）时，将块级索引展开为
        # 每 token 的平坦索引，因为 aiter 内核内部始终使用 page_size=1。
        self.kernel_block_size = kv_cache_spec.block_size

        # 在平坦视图（.view(-1,1,1,H)）中，每个 token 就是自己的页面，
        # 所以 max_num_pages_per_req = max_model_len，无论 kernel_block_size 是多少。
        max_num_pages_per_req = vllm_config.model_config.max_model_len
        max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        max_num_pages = max_num_reqs * max_num_pages_per_req

        # 准备持久化缓冲区
        # TODO: we can disambiguate between decode and mixed-prefill decode here
        # so we can only use the persistent buffer if a cudagraph is actually
        # being used.
        # 我们可以在此区分 decode 和混合 prefill/decode，以便只在实际使用
        # cudagraph 时才使用持久化缓冲区。

        # paged_kv_last_page_len 始终为 1（aiter 内核在 .view(-1,1,1,H) 展平后
        # 始终看到 page_size=1），所以我们创建一次并在 eager 和 cudagraph 模式下
        # 重用切片。
        self.paged_kv_last_page_len = torch.ones(
            max_num_reqs, dtype=torch.int32, device=device
        )

        # paged_kv_indices 的持久化缓冲区，避免阻塞布尔 mask 索引
        # （block_table_tensor[mask]），其输出大小依赖于数据。
        self.paged_kv_indices = torch.zeros(
            max_num_pages, dtype=torch.int32, device=device
        )

        from aiter import dtypes, get_mla_metadata_info_v1

        # 对于 num_attention_heads < 16（例如 kimi-k2.5 head=8 with TP8），
        # 确保 get_mla_metadata_info_v1 / get_mla_metadata_v1 与传递给
        # mla_decode_fwd 的实际 tensor 形状一致。
        self._num_attention_heads = max(16, self.num_heads)
        q_dtype = self.decode_attn_out_dtype
        kv_cache_dtype_str = getattr(vllm_config.cache_config, "cache_dtype", "auto")
        if kv_cache_dtype_str in ("fp8", "fp8_e4m3", "fp8_e5m2"):
            kv_cache_dtype_str = "fp8"
        else:
            kv_cache_dtype_str = "bf16"
        kv_dtype = dtypes.d_dtypes.get(kv_cache_dtype_str, dtypes.bf16)
        (
            (work_meta_data_size, work_meta_data_type),
            (work_indptr_size, work_indptr_type),
            (work_info_set_size, work_info_set_type),
            (reduce_indptr_size, reduce_indptr_type),
            (reduce_final_map_size, reduce_final_map_type),
            (reduce_partial_map_size, reduce_partial_map_type),
        ) = get_mla_metadata_info_v1(
            max_num_reqs,
            1,
            self._num_attention_heads,
            q_dtype,
            kv_dtype,
            is_sparse=False,
            fast_mode=True,
        )
        self._mla_work_meta_data = torch.empty(
            work_meta_data_size, dtype=work_meta_data_type, device=device
        )
        self._mla_work_indptr = torch.empty(
            work_indptr_size, dtype=work_indptr_type, device=device
        )
        self._mla_work_info_set = torch.empty(
            work_info_set_size, dtype=work_info_set_type, device=device
        )
        self._mla_reduce_indptr = torch.empty(
            reduce_indptr_size, dtype=reduce_indptr_type, device=device
        )
        self._mla_reduce_final_map = torch.empty(
            reduce_final_map_size, dtype=reduce_final_map_type, device=device
        )
        self._mla_reduce_partial_map = torch.empty(
            reduce_partial_map_size,
            dtype=reduce_partial_map_type,
            device=device,
        )

        self._fp8_prefill_enabled = _fp8_mla_prefill_supported()
        if self._fp8_prefill_enabled:
            max_prefill_qlen = min(
                vllm_config.model_config.max_model_len,
                vllm_config.scheduler_config.max_num_batched_tokens,
            )
            self._init_fp8_prefill_ps_buffers(max_num_reqs, max_prefill_qlen, device)

        if self.compilation_config.cudagraph_mode.has_full_cudagraphs():
            self.paged_kv_indptr = torch.zeros(
                max_num_reqs + 1, dtype=torch.int32, device=device
            )

            self.qo_indptr = torch.zeros(
                max_num_reqs + 1, dtype=torch.int32, device=device
            )

    def _init_fp8_prefill_ps_buffers(
        self,
        max_num_reqs: int,
        max_prefill_qlen: int,
        device: torch.device,
    ) -> None:
        """
        为 FP8 MLA prefill PS 元数据预分配持久化缓冲区。

        使用 get_ps_metadata_info_v1 和最大值，使缓冲区足够大以容纳任何 batch。
        get_ps_metadata_v1 在 build() 中逐 batch 填充它们。

        参数：
        - max_num_reqs: 最大并发请求数
        - max_prefill_qlen: 单个请求在一次 prefill batch 中的最大 Q 长度。
          应为 min(max_model_len, max_num_batched_tokens) — 分块 prefill 调度器
          每 batch 最多发出 max_num_batched_tokens 个新 token。
        - device: 缓冲区的目标设备
        """
        from aiter import get_ps_metadata_info_v1

        # kv_b_proj 解压缩后，K 有 num_heads 个头（与 Q 相同）。
        # 所以 gqa_ratio=1，num_head_k=num_heads（PS 内核使用）。
        num_head_k = self.num_heads
        # gqa_ratio = 1
        # qlen_granularity = _FP8_PREFILL_TILE_Q // max(gqa_ratio, 1)
        qlen_granularity = _FP8_PREFILL_TILE_Q

        (
            (work_metadata_size, work_metadata_dtype),
            (work_indptr_size, work_indptr_dtype),
            (work_info_size, work_info_dtype),
            (reduce_indptr_size, reduce_indptr_dtype),
            (reduce_final_map_size, reduce_final_map_dtype),
            (reduce_partial_map_size, reduce_partial_map_dtype),
        ) = get_ps_metadata_info_v1(
            batch_size=max_num_reqs,
            num_head_k=num_head_k,
            max_qlen=max_prefill_qlen,
            qlen_granularity=qlen_granularity,
        )

        self.fp8_ps_work_metadata = torch.empty(
            work_metadata_size, dtype=work_metadata_dtype, device=device
        )
        self.fp8_ps_work_indptr = torch.empty(
            work_indptr_size, dtype=work_indptr_dtype, device=device
        )
        self.fp8_ps_work_info = torch.empty(
            *work_info_size, dtype=work_info_dtype, device=device
        )
        self.fp8_ps_reduce_indptr = torch.empty(
            reduce_indptr_size, dtype=reduce_indptr_dtype, device=device
        )
        self.fp8_ps_reduce_final_map = torch.empty(
            *reduce_final_map_size, dtype=reduce_final_map_dtype, device=device
        )
        self.fp8_ps_reduce_partial_map = torch.empty(
            reduce_partial_map_size,
            dtype=reduce_partial_map_dtype,
            device=device,
        )

        logger.info(
            "FP8 MLA prefill PS buffers allocated "
            "(max_batch=%d, max_qlen=%d, num_head_k=%d)",
            max_num_reqs,
            max_prefill_qlen,
            num_head_k,
        )

    def _build_fp8_prefill_ps_metadata(
        self,
        metadata: AiterMLAMetadata,
        common_attn_metadata: CommonAttentionMetadata,
    ) -> None:
        """
        构建每 batch 的 FP8 MLA prefill PS 元数据并附加到 metadata。

        当 prefill token 存在且 FP8 MLA prefill 启用时从 build() 调用。
        """
        from aiter import get_ps_metadata_v1

        prefill = metadata.prefill
        # 调用者（build()）仅在 prefill token 存在时调用此方法，
        # 所以 metadata.prefill 保证非 None。断言以缩小 mypy 类型。
        assert prefill is not None
        qo_indptr = prefill.query_start_loc
        kv_indptr = qo_indptr  # 新 token：KV 长度 == Q 长度

        # 重用 query_start_loc 的现有 CPU 视图，而不是强制 device->host 复制。
        # Prefill batch 位于请求列表的尾部，所以我们从 num_decodes 开始切片
        # 并重新基准化为零，与父级 build 在设备 tensor 上应用的变换相同。
        num_decodes = metadata.num_decodes
        qsl_cpu = common_attn_metadata.query_start_loc_cpu
        qo_indptr_cpu = (qsl_cpu[num_decodes:] - qsl_cpu[num_decodes]).to(torch.int32)
        kv_indptr_cpu = qo_indptr_cpu.clone()
        seq_lens_cpu = (qo_indptr_cpu[1:] - qo_indptr_cpu[:-1]).to(torch.int32)

        num_head_k = self.num_heads
        # gqa_ratio = 1
        # qhead_granularity = max(gqa_ratio, 1)
        # qlen_granularity = _FP8_PREFILL_TILE_Q // qhead_granularity
        gqa_ratio = 1
        qhead_granularity = 1
        qlen_granularity = _FP8_PREFILL_TILE_Q
        kvlen_granularity = 128
        block_size = 1  # 非分页：每个 "page" 是一个 token

        get_ps_metadata_v1(
            qo_indptr_cpu,
            kv_indptr_cpu,
            seq_lens_cpu,
            gqa_ratio,
            num_head_k,
            self.fp8_ps_work_metadata,
            self.fp8_ps_work_indptr,
            self.fp8_ps_work_info,
            self.fp8_ps_reduce_indptr,
            self.fp8_ps_reduce_final_map,
            self.fp8_ps_reduce_partial_map,
            qhead_granularity=qhead_granularity,
            qlen_granularity=qlen_granularity,
            kvlen_granularity=kvlen_granularity,
            block_size=block_size,
            is_causal=True,
        )

        total_prefill_tokens = int(qo_indptr_cpu[-1].item())
        kv_indices = torch.arange(
            total_prefill_tokens, device=qo_indptr.device, dtype=torch.int32
        )

        # 此 batch 的实际活跃 partial tile 数量是 reduce_indptr 的最终值。
        # 在此（元数据构建期间）解析它，使其远离每层前向路径，
        # 在那里同步会破坏 CUDA Graph 捕获。使用设备端 reduce_indptr 是
        # 可接受的，因为 build 允许偶尔的同步。
        num_partial_tiles = int(self.fp8_ps_reduce_indptr[-1].item())

        # 将 PS 元数据附加到 metadata 对象，以便 forward_mha 可以读取。
        metadata.fp8_prefill_qo_indptr = qo_indptr
        metadata.fp8_prefill_kv_indptr = kv_indptr
        metadata.fp8_prefill_kv_indices = kv_indices
        metadata.fp8_prefill_work_indptr = self.fp8_ps_work_indptr
        metadata.fp8_prefill_work_info_set = self.fp8_ps_work_info
        metadata.fp8_prefill_reduce_indptr = self.fp8_ps_reduce_indptr
        metadata.fp8_prefill_reduce_final_map = self.fp8_ps_reduce_final_map
        metadata.fp8_prefill_reduce_partial_map = self.fp8_ps_reduce_partial_map
        metadata.fp8_prefill_max_q_len = prefill.max_query_len
        metadata.fp8_prefill_num_partial_tiles = num_partial_tiles

    def _build_decode(
        self,
        block_table_tensor: torch.Tensor,
        seq_lens_device: torch.Tensor,
        max_seq_len: int,
        query_start_loc_cpu: torch.Tensor,
        query_start_loc_device: torch.Tensor,
        num_decode_tokens: int,
        dcp_tot_seq_lens_device: torch.Tensor | None,
    ) -> AiterMLADecodeMetadata:
        """
        构建 decode 部分的元数据。

        流程：
        1. 构建 paged_kv_indptr（seq_lens 的累积和）
        2. 展开 block_table 为每 token 的平坦索引
        3. 如果 max_qo_len == 1，计算持久化 MLA 元数据
        4. 返回 AiterMLADecodeMetadata
        """
        device = self.device
        num_reqs = seq_lens_device.size(0)

        # aiter 内核内部始终使用 page_size=1（包装器展平 kv_buffer）。
        # last_page_len 始终为 1。
        paged_kv_last_page_len = self.paged_kv_last_page_len[:num_reqs]

        # indptr：seq_lens 的累积和（平坦视图中每个 token 一个页面）
        paged_kv_indptr = torch.cat(
            [
                torch.zeros(1, dtype=seq_lens_device.dtype, device=device),
                seq_lens_device.cumsum(dim=0, dtype=torch.int32),
            ]
        )
        qo_len = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
        max_qo_len = qo_len.max().item()

        if self.compilation_config.cudagraph_mode.has_full_cudagraphs():
            self.paged_kv_indices.fill_(-1)

        # 将 block_table 条目展开为每 token 的平坦索引。
        # 当 kernel_block_size=1 时，退化为直接复制（与原始 _copy_page_indices_kernel 相同）。
        # 当 kernel_block_size=K>1 时，覆盖 K 个 token 的 block_table 条目 b
        # 被展开为平坦索引 b*K, b*K+1, ..., b*K+(K-1)。
        _expand_page_indices_kernel[(num_reqs,)](
            self.paged_kv_indices,
            block_table_tensor,
            block_table_tensor.stride(0),
            paged_kv_indptr,
            seq_lens_device,
            KERNEL_BLOCK_SIZE=self.kernel_block_size,
            BLOCK_SIZE=1024,
        )
        paged_kv_indices = self.paged_kv_indices

        if self.compilation_config.cudagraph_mode.has_full_cudagraphs():
            self.paged_kv_indptr[: 1 + num_reqs].copy_(
                paged_kv_indptr, non_blocking=True
            )
            self.paged_kv_indptr[1 + num_reqs :].fill_(paged_kv_indptr[-1])
            paged_kv_indptr = self.paged_kv_indptr[: 1 + num_reqs]

            # paged_kv_last_page_len 已使用预初始化的缓冲区切片（上面设置），
            # 所以不需要复制 - 缓冲区始终为 1。

            self.qo_indptr[: 1 + num_reqs].copy_(
                query_start_loc_device, non_blocking=True
            )
            self.qo_indptr[1 + num_reqs :] = query_start_loc_device[-1]
            qo_indptr = self.qo_indptr[: 1 + num_reqs]

        else:
            qo_indptr = torch.arange(
                0, num_reqs + 1, step=1, dtype=torch.int32, device=device
            )

        # AITER MLA ASM 内核只支持 qseqlen=1（单 token decode）。
        # 使用投机解码时，验证步骤有 qseqlen > 1（例如 spec7 的 8）。
        # get_mla_metadata_v1 调用 get_heuristic_kernel_mla，
        # 对 qseqlen > 1 会失败。
        # 我们跟踪是否成功计算了持久化元数据，以便 forward_mqa 可以
        # 跳过传递它（回退到内核自己内部计算元数据，如 v0.18.0）。
        has_persistent_metadata = False
        if max_qo_len == 1:
            from aiter import get_mla_metadata_v1

            get_mla_metadata_v1(
                qo_indptr,
                paged_kv_indptr,
                paged_kv_last_page_len,
                self._num_attention_heads,
                1,
                True,
                self._mla_work_meta_data,
                self._mla_work_info_set,
                self._mla_work_indptr,
                self._mla_reduce_indptr,
                self._mla_reduce_final_map,
                self._mla_reduce_partial_map,
                page_size=1,
                kv_granularity=16,
                max_seqlen_qo=max_qo_len,
                uni_seqlen_qo=max_qo_len,
                fast_mode=True,
            )
            has_persistent_metadata = True

        attn_metadata = AiterMLADecodeMetadata(
            block_table=block_table_tensor,
            seq_lens=seq_lens_device,
            paged_kv_indptr=paged_kv_indptr,
            paged_kv_indices=paged_kv_indices,
            paged_kv_last_page_len=paged_kv_last_page_len,
            qo_indptr=qo_indptr,
            dcp_tot_seq_lens=dcp_tot_seq_lens_device,
            max_qo_len=max_qo_len,
            attn_out_dtype=self.decode_attn_out_dtype,
            has_persistent_metadata=has_persistent_metadata,
        )

        return attn_metadata

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> AiterMLAMetadata:
        """
        构建 AITER MLA 的完整元数据。

        流程：
        1. 调用父类 build 构建基本元数据
        2. 如果 decode 有持久化元数据，附加工作缓冲区
        3. 如果启用了 FP8 prefill 且有 prefill token，构建 PS 元数据
        """
        attn_metadata = super().build(
            common_prefix_len, common_attn_metadata, fast_build
        )
        if (
            attn_metadata.decode is not None
            and attn_metadata.decode.has_persistent_metadata
        ):
            attn_metadata.work_meta_data = self._mla_work_meta_data
            attn_metadata.work_indptr = self._mla_work_indptr
            attn_metadata.work_info_set = self._mla_work_info_set
            attn_metadata.reduce_indptr = self._mla_reduce_indptr
            attn_metadata.reduce_final_map = self._mla_reduce_final_map
            attn_metadata.reduce_partial_map = self._mla_reduce_partial_map
        if self._fp8_prefill_enabled and attn_metadata.prefill is not None:
            self._build_fp8_prefill_ps_metadata(attn_metadata, common_attn_metadata)
        return attn_metadata


@triton.jit
def _expand_page_indices_kernel(
    page_indices,
    block_table,
    block_table_stride,
    cu_num_tokens,
    seq_lens,
    KERNEL_BLOCK_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    将 block table 条目展开为每 token 的平坦页面索引。

    AITER MLA 内核内部始终使用 page_size=1（kv_buffer 通过 .view(-1, 1, 1, H) 展平）。
    此内核将块表中的块级索引转换为展平 KV buffer 中的各个 token 位置。

    当 KERNEL_BLOCK_SIZE=1 时：block_idx=t, offset=0, flat=block_id
    （等同于直接复制 -- 与原始内核相比无回归）。

    当 KERNEL_BLOCK_SIZE=K 时：覆盖 K 个 token 的 block table 条目 b
    被展开为平坦索引 b*K, b*K+1, ..., b*K+(K-1)。

    参数：
    - page_indices: 输出的平坦页面索引
    - block_table: 输入的块表
    - block_table_stride: 块表的行步幅
    - cu_num_tokens: 每个请求的累积 token 数
    - seq_lens: 每个请求的序列长度
    - KERNEL_BLOCK_SIZE: 内核块大小（编译期常量）
    - BLOCK_SIZE: Triton 的 tile 大小
    """
    req_idx = tl.program_id(0)
    row_ptr = block_table + req_idx * block_table_stride
    start_idx = tl.load(cu_num_tokens + req_idx)
    num_tokens = tl.load(seq_lens + req_idx)

    offset = tl.arange(0, BLOCK_SIZE)
    for i in tl.range(0, num_tokens, BLOCK_SIZE):
        token_offsets = i + offset
        mask = token_offsets < num_tokens

        # 此 token 属于块表中的哪个块？
        block_idx = token_offsets // KERNEL_BLOCK_SIZE
        # 该块内的偏移
        offset_in_block = token_offsets % KERNEL_BLOCK_SIZE

        # 从块表加载块 ID
        block_ids = tl.load(row_ptr + block_idx, mask=mask)

        # 计算展平 kv_buffer 中的平坦索引
        flat_indices = block_ids * KERNEL_BLOCK_SIZE + offset_in_block

        tl.store(
            page_indices + start_idx + token_offsets,
            flat_indices,
            mask=mask,
        )


class AiterMLAHelper:
    """
    AITER MLA 辅助类。

    AITER MLA 实现要求 num_heads >= 16。如果 num_heads < 16 且 16 % num_heads == 0，
    我们可以将 q padding 到 16 个头；否则 AITER 必须失败。

    提供以下静态方法：
    - check_num_heads_validity: 检查 head 数量是否有效
    - is_valid_num_heads: 判断 head 数量是否有效
    - get_actual_mla_num_heads: 获取实际使用的 MLA head 数量（至少 16）
    - get_mla_padded_q: 获取 padding 后的 query
    - get_mla_unpadded_o: 获取去 padding 后的输出
    """

    _AITER_MIN_MLA_HEADS: Final = 16
    _AITER_UNSUPPORTED_HEADS: ClassVar[tuple[int, ...]] = ()

    @staticmethod
    def check_num_heads_validity(num_heads: int):
        """检查 head 数量是否有效，无效时抛出断言错误。"""
        assert AiterMLAHelper.is_valid_num_heads(num_heads), (
            f"Aiter MLA requires that num_heads be multiples or divisors of 16, "
            f"but provided {num_heads} number of heads.\n"
            f"Try adjusting tensor_parallel_size value."
        )

    @staticmethod
    def is_valid_num_heads(num_heads: int) -> bool:
        """判断 head 数量是否有效（是 16 的倍数或 16 是它的倍数）。"""
        return (
            num_heads % AiterMLAHelper._AITER_MIN_MLA_HEADS == 0
            if num_heads >= AiterMLAHelper._AITER_MIN_MLA_HEADS
            else AiterMLAHelper._AITER_MIN_MLA_HEADS % num_heads == 0
        )

    @staticmethod
    def get_actual_mla_num_heads(num_heads: int) -> int:
        """获取实际使用的 MLA head 数量（至少 16）。"""
        return max(num_heads, AiterMLAHelper._AITER_MIN_MLA_HEADS)

    @staticmethod
    def get_mla_padded_q(num_heads: int, q: torch.Tensor) -> torch.Tensor:
        """获取 padding 后的 query。如果 num_heads >= 16，直接返回原 q。"""
        return (
            q
            if num_heads >= AiterMLAHelper._AITER_MIN_MLA_HEADS
            else q.repeat_interleave(
                AiterMLAHelper._AITER_MIN_MLA_HEADS // num_heads, dim=1
            )
        )

    @staticmethod
    def get_mla_unpadded_o(num_heads: int, o: torch.Tensor) -> torch.Tensor:
        """获取去 padding 后的输出。如果 num_heads >= 16，直接返回原 o。"""
        return (
            o
            if num_heads >= AiterMLAHelper._AITER_MIN_MLA_HEADS
            else o[:, :: AiterMLAHelper._AITER_MIN_MLA_HEADS // num_heads, :]
        )


class AiterMLAImpl(MLACommonImpl[AiterMLAMetadata]):
    """
    AITER MLA 注意力的具体实现。

    使用 AITER 库的 mla_decode_fwd 内核执行 decode 注意力计算。
    对于 prefill，支持两种路径：
    1. FP8 ASM prefill（gfx950 上的 mla_prefill_ps_asm_fwd + mla_reduce_v1）
    2. 标准 flash_attn_varlen_func（回退路径）
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
        AiterMLAHelper.check_num_heads_validity(num_heads)

        unsupported_features = [alibi_slopes, sliding_window, logits_soft_cap]
        if any(unsupported_features):
            raise NotImplementedError(
                "Aiter MLA does not support one of the following: "
                "alibi_slopes, sliding_window, logits_soft_cap"
            )

        from aiter import flash_attn_varlen_func

        self.flash_attn_varlen_func = flash_attn_varlen_func

        # FP8 MLA prefill 内核导入（延迟加载，仅在启用时）。
        # 在 gfx950 上当 AITER 提供内核时自动启用。
        self._fp8_prefill_enabled = _fp8_mla_prefill_supported()
        if self._fp8_prefill_enabled:
            from aiter import mla_prefill_ps_asm_fwd, mla_reduce_v1

            self._mla_prefill_ps_asm_fwd = mla_prefill_ps_asm_fwd
            self._mla_reduce_v1 = mla_reduce_v1

    def _flash_attn_varlen_diff_headdims(self, q, k, v, return_softmax_lse=False, softmax_scale=None, **kwargs):
        """调用 AITER 的 flash_attn_varlen_func，支持不同 head 维度。"""
        output = self.flash_attn_varlen_func(  # type: ignore[call-arg]
            q=q,
            k=k,
            v=v,
            softmax_scale=softmax_scale,
            return_lse=return_softmax_lse,
            **kwargs,
        )

        return output

    def _mla_fp8_prefill_attn(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_metadata: AiterMLAMetadata,
        out: torch.Tensor,
    ) -> None:
        """
        通过 mla_prefill_ps_asm_fwd + mla_reduce_v1 运行 FP8 MLA prefill。

        Q、K、V 已经解压缩（后 kv_b_proj），所以 K 和 V 有 num_heads 个头
        （与 Q 相同），gqa_ratio=1。将结果原地写入 out，out 是 forward_mha
        提供的 [total_q, nhead * v_head_dim] 输出缓冲区；不需要额外的分配或复制。

        两阶段流水线：
        1. mla_prefill_ps_asm_fwd：持久化调度汇编 prefill 内核
        2. mla_reduce_v1：跨 KV split 的归约
        """
        from vllm.platforms import current_platform
        from vllm.v1.worker.workspace import current_workspace_manager

        fp8_dtype = current_platform.fp8_dtype()
        total_q = q.shape[0]
        nhead = self.num_heads
        v_head_dim = self.v_head_dim
        tile_q = _FP8_PREFILL_TILE_Q

        # FP8 ASM 内核期望 FP8 输入；q_scale/k_scale/v_scale 参数选择
        # 每 tensor 的反量化缩放因子。Q/K/V 从 kv_b_proj 以 bf16 到达，所以在此转换
        #（one_scale=1.0 禁用缩放）。
        if q.dtype != fp8_dtype:
            q = q.to(fp8_dtype)
        if k.dtype != fp8_dtype:
            k = k.to(fp8_dtype)
        if v.dtype != fp8_dtype:
            v = v.to(fp8_dtype)

        one_scale = torch.ones((), dtype=torch.float32, device=q.device)

        # num_partial_tiles 在元数据构建期间解析，以避免在 forward 中的 .item() 同步
        # 会阻止 CUDA Graph 捕获。forward_mha 通过 fp8_prefill_qo_indptr 是否设置
        # 来门控 FP8 路径，而构建器总是同时设置所有 fp8_prefill_* 字段，
        # 所以 num_partial_tiles 在此非 None。
        num_partial_tiles = attn_metadata.fp8_prefill_num_partial_tiles
        assert num_partial_tiles is not None

        # 重用调用者的输出缓冲区以跳过每次调用的分配 + 复制。
        # ASM 和 reduce 内核都写入 [total_q, nhead, v_head_dim] 视图，
        # 它与 out 的 [total_q, nhead * v_head_dim] 存储别名。
        out_3d = out.view(total_q, nhead, v_head_dim)

        # 每次调用的 scratch（logits, attn_lse, final_lse）由 workspace manager 提供，
        # 以便 prefill 热路径中的分配器开销在预热后有界，匹配 PR #41002 中的模式。
        logits, attn_lse, final_lse = current_workspace_manager().get_simultaneous(
            ((num_partial_tiles * tile_q, nhead, v_head_dim), torch.float32),
            ((num_partial_tiles * tile_q, nhead), torch.float32),
            ((total_q, nhead), torch.float32),
        )

        # 阶段 1：持久化调度汇编 prefill 内核
        self._mla_prefill_ps_asm_fwd(
            q,
            k,
            v,
            attn_metadata.fp8_prefill_qo_indptr,
            attn_metadata.fp8_prefill_kv_indptr,
            attn_metadata.fp8_prefill_kv_indices,
            attn_metadata.fp8_prefill_work_indptr,
            attn_metadata.fp8_prefill_work_info_set,
            attn_metadata.fp8_prefill_max_q_len,
            self.scale,
            True,  # is_causal
            logits,
            attn_lse,
            out_3d,
            one_scale,
            one_scale,
            one_scale,
        )

        # 阶段 2：跨 KV split 的归约
        self._mla_reduce_v1(
            logits,
            attn_lse,
            attn_metadata.fp8_prefill_reduce_indptr,
            attn_metadata.fp8_prefill_reduce_final_map,
            attn_metadata.fp8_prefill_reduce_partial_map,
            tile_q,
            out_3d,
            final_lse,
        )

    def forward_mha(
        self,
        q: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: MLACommonMetadata,
        k_scale: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        """
        当 FP8 ASM 可用时，将 prefill 分派到 FP8 ASM 内核。

        当 FP8 MLA prefill 禁用、PS 元数据缺失或分块上下文需要两阶段合并时，
        回退到父类（flash_attn_varlen_func）。

        注解使用基类 MLACommonMetadata 以遵守 LSP 与 MLACommonImpl.forward_mha 的关系；
        AITER 构建器在运行时总是生成 AiterMLAMetadata 实例，所以我们用 isinstance
        缩小类型后再读取 AITER 特有的 FP8 字段。
        """
        if (
            not self._fp8_prefill_enabled
            or not isinstance(attn_metadata, AiterMLAMetadata)
            or attn_metadata.fp8_prefill_qo_indptr is None
        ):
            return super().forward_mha(
                q,
                kv_c_normed,
                k_pe,
                kv_c_and_k_pe_cache,
                attn_metadata,
                k_scale,
                output,
            )

        assert attn_metadata.prefill is not None
        prefill_metadata = attn_metadata.prefill
        has_context = prefill_metadata.chunked_context is not None

        if has_context:
            return super().forward_mha(
                q,
                kv_c_normed,
                k_pe,
                kv_c_and_k_pe_cache,
                attn_metadata,
                k_scale,
                output,
            )

        kv_nope = self.kv_b_proj(kv_c_normed)[0].view(
            -1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
        )
        k_nope, v = kv_nope.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k = self._concat_k_nope_k_pe(k_nope, k_pe)

        self._mla_fp8_prefill_attn(q, k, v, attn_metadata, output)

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: AiterMLAMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Decode 注意力的前向传播。

        流程：
        1. 如果 q 是元组，拼接为完整 q
        2. 将 q padding 到至少 16 个头
        3. 调用 mla_decode_fwd 计算注意力
        4. 去 padding 后返回输出
        """
        assert kv_c_and_k_pe_cache.numel() > 0
        assert attn_metadata.decode is not None
        assert attn_metadata.decode.max_qo_len is not None

        if type(q) is tuple:
            q = torch.cat(q, dim=-1)

        assert isinstance(q, torch.Tensor)
        B = q.shape[0]

        mla_padded_q = AiterMLAHelper.get_mla_padded_q(self.num_heads, q)
        mla_num_heads = AiterMLAHelper.get_actual_mla_num_heads(self.num_heads)
        o = torch.empty(
            B,
            mla_num_heads,
            self.kv_lora_rank,
            dtype=attn_metadata.decode.attn_out_dtype,
            device=q.device,
        )

        kv_buffer = kv_c_and_k_pe_cache.unsqueeze(2)

        # 构建 mla_decode_fwd 的 kwargs。仅在成功计算了持久化元数据时
        # 传递持久化元数据（qseqlen=1 decode 步骤）。
        # 对于多 token 验证步骤（spec-dec），内核回退到内部计算元数据。
        mla_kwargs = dict(
            q_scale=layer._q_scale,
            kv_scale=layer._k_scale,
        )
        if attn_metadata.work_meta_data is not None:
            mla_kwargs.update(
                work_meta_data=attn_metadata.work_meta_data,
                work_indptr=attn_metadata.work_indptr,
                work_info_set=attn_metadata.work_info_set,
                reduce_indptr=attn_metadata.reduce_indptr,
                reduce_final_map=attn_metadata.reduce_final_map,
                reduce_partial_map=attn_metadata.reduce_partial_map,
            )

        rocm_aiter_ops.mla_decode_fwd(
            mla_padded_q,
            kv_buffer,
            o,
            self.scale,
            attn_metadata.decode.qo_indptr,
            attn_metadata.decode.max_qo_len,
            attn_metadata.decode.paged_kv_indptr,
            attn_metadata.decode.paged_kv_indices,
            attn_metadata.decode.paged_kv_last_page_len,
            **mla_kwargs,
        )

        return AiterMLAHelper.get_mla_unpadded_o(self.num_heads, o), None
