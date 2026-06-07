# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
稀疏滑动窗口注意力（SWA）后端模块。

本模块实现了 DeepSeekV4 模型的稀疏滑动窗口注意力后端，用于 SWA 层的高效推理。

核心特性：
1. 滑动窗口注意力：每个 token 只关注窗口内的 KV token
2. 支持 DeepSeekV4 的多种压缩比率（compress_ratio: 1, 4, 128）
3. 使用 FlashMLA 内核进行 decode 注意力计算
4. 支持混合 decode/prefill batch
5. 支持 MTP（Multi-Token Prediction）多 token decode

DeepSeekV4 层类型：
- swaonly: 纯 SWA 层（compress_ratio <= 1）
- c4a: C4A 层（compress_ratio = 4，每 4 个 token 压缩为 1 个）
- c128a: C128A 层（compress_ratio = 128，每 128 个 token 压缩为 1 个）

每种层类型有独立的 FlashMLA tile-scheduler 计划，因为它们的
（topk, extra_topk, extra_page_block_size）配置不同。

架构概述：
- DeepseekV4SWACache: SWA 缓存层定义
- DeepseekSparseSWABackend: 注意力后端定义
- DeepseekSparseSWAMetadataBuilder: 元数据构建器
- DeepseekSparseSWAMetadata: 元数据
"""

from dataclasses import dataclass
from typing import ClassVar, cast

import torch

from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.attention.ops.flashmla import FlashMLASchedMeta, get_mla_metadata
from vllm.v1.kv_cache_interface import (
    KVCacheSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
)

# DeepSeekV4 decode 层类型，按 compress_ratio 键控。每种类型有独特的
#（topk, extra_topk, extra_page_block_size）配置，所以不能共享
# FlashMLA tile-scheduler 计划。在同一种类型内，所有 ~60 个 DeepSeekV4 层
# 每步共享一个计划，因为 b / s_q / h_q / page_block_sizes / topks 是相同的。
_LAYER_TYPE_SWAONLY = "swaonly"  # 纯 SWA 层
_LAYER_TYPE_C4A = "c4a"  # C4A 层（compress_ratio=4）
_LAYER_TYPE_C128A = "c128a"  # C128A 层（compress_ratio=128）


def _layer_type_for(compress_ratio: int) -> str:
    """根据压缩比率返回层类型字符串。"""
    if compress_ratio <= 1:
        return _LAYER_TYPE_SWAONLY
    if compress_ratio == 4:
        return _LAYER_TYPE_C4A
    if compress_ratio == 128:
        return _LAYER_TYPE_C128A
    raise ValueError(
        f"Unsupported DeepseekV4 compress_ratio={compress_ratio}; "
        "expected 1, 4, or 128."
    )


class DeepseekV4SWACache(torch.nn.Module, AttentionLayerBase):
    """
    DeepSeekV4 SWA 缓存层。

    定义 SWA 缓存的基本属性，包括：
    - head_dim: head 维度
    - window_size: 滑动窗口大小
    - block_size: 块大小（固定为 64，因为 C4A 和 SWA 共享物理 tensor）
    """

    def __init__(
        self,
        head_dim: int,
        window_size: int,
        dtype: torch.dtype,
        prefix: str,
        cache_config: CacheConfig,
    ):
        super().__init__()
        self.kv_cache = torch.tensor([])
        self.head_dim = head_dim
        self.window_size = window_size
        self.prefix = prefix
        self.cache_config = cache_config
        self.dtype = dtype
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

        # 块大小受 SWA 和 C4A KV 块之间的 tensor 共享约束。
        # 由于两种块类型共享相同的物理 tensor，它们必须使用相同的页面大小。
        # C4A KV 块形状 [256//4, head_dim] = [64, head_dim] 决定了
        # SWA 块大小为每块 64 个 token。
        # TODO(yifan): make SWA block size automatically determined and configurable.
        # TODO(yifan): 使 SWA 块大小自动确定和可配置。
        self.block_size = 64
        assert self.dtype == torch.uint8

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        """返回 SWA 的 KV 缓存规格。"""
        return SlidingWindowMLASpec(
            block_size=self.block_size,
            num_kv_heads=1,
            head_size=self.head_dim,
            dtype=self.dtype,
            sliding_window=self.window_size,
            cache_dtype_str=self.cache_config.cache_dtype,
            alignment=576,  # NOTE: FlashMLA requires 576B alignment
            # FlashMLA 要求 576B 对齐
            model_version="deepseek_v4",
        )

    def forward(self): ...

    def get_attn_backend(self) -> type[AttentionBackend]:
        return DeepseekSparseSWABackend


class DeepseekSparseSWABackend(AttentionBackend):
    """
    DeepSeek 稀疏 SWA 注意力后端。

    定义了后端的基本属性和支持的配置。
    """

    @staticmethod
    def get_name() -> str:
        return "DEEPSEEK_SPARSE_SWA"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(64)]

    @classmethod
    def get_preferred_block_size(cls, default_block_size: int) -> int:
        """首选块大小为 256。"""
        return 256

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        """支持的 head size：512。"""
        return [512]

    @staticmethod
    def get_builder_cls() -> type["DeepseekSparseSWAMetadataBuilder"]:
        """根据平台返回不同的元数据构建器。"""
        if current_platform.is_rocm():
            from vllm.models.deepseek_v4.amd.rocm import (
                DeepseekV4ROCMAiterSparseSWAMetadataBuilder,
            )

            return DeepseekV4ROCMAiterSparseSWAMetadataBuilder
        return DeepseekSparseSWAMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        """
        计算 KV 缓存形状。

        对于 fp8_ds_mla 格式：每个 token 584 字节
        （448 NoPE + 128 RoPE + 8 fp8 scale）
        """
        assert num_kv_heads == 1
        if cache_dtype_str == "fp8_ds_mla":
            # DeepSeekV4 SWA：每 token 584B（448 NoPE + 128 RoPE + 8 fp8 scale）。
            # 传入的 head_size 是语义 head_dim（512）。
            return (num_blocks, block_size, 584)
        else:
            return (num_blocks, block_size, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        if include_num_layers_dimension:
            return (0, 1, 2, 3)
        return (0, 1, 2)


@dataclass
class DeepseekSparseSWAMetadata:
    """
    DeepSeek 稀疏 SWA 注意力的元数据。

    包含 SWA 注意力计算所需的所有元数据，包括：
    - 基本映射：block_table, slot_mapping, seq_lens 等
    - Decode 部分：SWA 索引和长度
    - Prefill 部分：预计算的 prefill 元数据
    - FlashMLA tile-scheduler 计划（每种层类型一个）
    """
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    block_size: int
    seq_lens: torch.Tensor | None = None  # [num_seqs]
    query_start_loc: torch.Tensor | None = None  # [num_seqs + 1]
    query_start_loc_cpu: torch.Tensor | None = None  # [num_seqs + 1]

    is_valid_token: torch.Tensor | None = None  # [num_tokens]
    token_to_req_indices: torch.Tensor | None = None  # [num_tokens]
    decode_swa_indices: torch.Tensor | None = None  # [num_decode_tokens, window_size]
    decode_swa_lens: torch.Tensor | None = None  # [num_decode_tokens]

    # Decode/prefill 请求数/token 数（batch 重排序：decode 在前）
    num_decodes: int = 0
    num_prefills: int = 0
    num_decode_tokens: int = 0
    num_prefill_tokens: int = 0

    # 预计算的 prefill 元数据，所有 DeepSeekV4 注意力层共享
    prefill_seq_lens: torch.Tensor | None = None
    prefill_gather_lens: torch.Tensor | None = None

    # 每层类型的 FlashMLA tile-scheduler 元数据。每种存在的 DeepSeekV4 层类型
    # 一个 FlashMLASchedMeta，在同一步中所有 ~60 个同类型层共享。
    # 同类型的第一个 forward 调用触发内核中的 planner（也通过 PyTorch 的
    # graph-aware 分配器分配 tile_scheduler_metadata 和 num_splits）；
    # 后续同类型调用跳过 planning 并重用计划。每 build() 新实例，
    # 所以 have_initialized 在步骤开始时总是 False，计划从当前
    # seq_lens / topk_length 重新派生。
    # 对于模型不使用的层类型（或当 num_decode_tokens 为零时）为 None。
    tile_sched_swaonly: "FlashMLASchedMeta | None" = None
    tile_sched_c4a: "FlashMLASchedMeta | None" = None
    tile_sched_c128a: "FlashMLASchedMeta | None" = None


class DeepseekSparseSWAMetadataBuilder(AttentionMetadataBuilder):
    """
    DeepSeekV4 SWA 缓存的元数据构建器。

    与索引器类似，通过以下方式处理混合 batch：
    1. 使用 split_decodes_and_prefills() 确定边界
    2. 为 decode 和 prefill 部分分别构建元数据

    支持：
    - 混合 decode/prefill batch
    - MTP（Multi-Token Prediction）decode 有 query_len > 1
    - 分块 prefill（与索引器的分块对齐）
    """

    # 基础阈值：query_len <= 1 是 decode
    reorder_batch_threshold: int = 1
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(self.kv_cache_spec, SlidingWindowMLASpec | MLAAttentionSpec)
        mla_spec = cast(SlidingWindowMLASpec | MLAAttentionSpec, self.kv_cache_spec)
        self.head_size = mla_spec.head_size  # 已考虑量化
        self.compress_ratio = mla_spec.compress_ratio
        self.block_size = mla_spec.block_size

        # 处理 MTP：像索引器一样调整 decode_threshold
        self.num_speculative_tokens = (
            self.vllm_config.speculative_config.num_speculative_tokens
            if self.vllm_config.speculative_config
            else 0
        )
        # 使用 MTP 时，decode 可以有 query_len 高达 1 + num_speculative_tokens。
        # 必须与索引器和 flashmla_sparse 使用的阈值匹配，
        # 以便所有后端在 decode/prefill 拆分上达成一致。
        self.decode_threshold = (
            self.reorder_batch_threshold + self.num_speculative_tokens
        )

        hf_config = self.vllm_config.model_config.hf_config
        assert hasattr(hf_config, "sliding_window")
        self.window_size = hf_config.sliding_window

        # 检测此模型使用的 DeepSeekV4 层类型，以便只为实际会调用的类型
        # 构建 FlashMLA tile-scheduler 计划。
        # 没有 compress_ratios 的模型（纯 SWA）回退到 swaonly。
        compress_ratios = getattr(hf_config, "compress_ratios", None) or [1]
        self._layer_types: set[str] = set()
        for ratio in compress_ratios:
            self._layer_types.add(_layer_type_for(int(ratio)))

        max_tokens = self.vllm_config.scheduler_config.max_num_batched_tokens
        self.token_to_req_indices = torch.zeros(
            max_tokens, dtype=torch.int32, device=self.device,
        )
        self.decode_swa_indices = torch.zeros(
            max_tokens, 1, self.window_size, dtype=torch.int32, device=self.device,
        )
        self.decode_swa_lens = torch.zeros(
            max_tokens, dtype=torch.int32, device=self.device,
        )
        self.is_valid_token = torch.zeros(
            max_tokens, dtype=torch.bool, device=self.device,
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> DeepseekSparseSWAMetadata:
        """
        为混合 decode/prefill batch 构建 SWA 元数据。

        假设 batch 已被 vLLM 调度器重排序为 decode 在前。
        使用 split_decodes_and_prefills() 找到边界，然后为每个部分
        分别构建 window_topk_idxs。

        流程：
        1. 拆分 decode 和 prefill
        2. 构建 token_to_req_indices 映射
        3. 计算 decode 的 SWA 索引和长度
        4. 构建 DeepSeekV4 prefill 元数据
        5. 构建每层类型的 tile-scheduler 计划
        """
        num_reqs = common_attn_metadata.num_reqs
        seq_lens = common_attn_metadata.seq_lens
        query_start_loc = common_attn_metadata.query_start_loc
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        block_table = common_attn_metadata.block_table_tensor
        slot_mapping = common_attn_metadata.slot_mapping

        # 使用可配置阈值拆分为 decode 和 prefill 部分
        (num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens) = (
            split_decodes_and_prefills(
                common_attn_metadata, decode_threshold=self.decode_threshold
            )
        )

        # NOTE: Ensure all metadata tensors maintain fixed memory addresses
        # for CUDA graph compatibility.
        # 确保所有元数据 tensor 保持固定内存地址以支持 CUDA Graph。
        query_lens = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
        x = torch.repeat_interleave(torch.arange(num_reqs), query_lens).pin_memory()
        token_to_req_indices = self.token_to_req_indices[: x.shape[0]]
        token_to_req_indices.copy_(x, non_blocking=True)

        is_valid_token = self.is_valid_token[: slot_mapping.shape[0]]
        is_valid_token.copy_(slot_mapping >= 0)

        if num_decode_tokens > 0:
            self.decode_swa_lens[num_decode_tokens:] = 0
            _compute_swa_indices_and_lens_kernel[(num_decode_tokens,)](
                self.decode_swa_indices,
                self.decode_swa_indices.stride(0),
                self.decode_swa_lens,
                self.window_size,
                query_start_loc,
                seq_lens,
                token_to_req_indices,
                is_valid_token,
                block_table,
                block_table.stride(0),
                self.block_size,
                TRITON_BLOCK_SIZE=1024,
            )

        # 预计算所有注意力层共享的 DeepSeekV4 prefill 元数据
        deepseek_v4_fields = self._build_deepseek_v4_metadata(
            num_decodes, num_prefills, seq_lens, query_start_loc,
        )

        # 每层类型的 tile-scheduler 计划持有者。
        # 每种存在的 DeepSeekV4 层类型一个空的 FlashMLASchedMeta；
        # 同类型的第一个 flash_mla_with_kvcache 调用触发 planner，
        # 所有同类型层在步骤的其余部分重用结果计划。
        tile_sched = self.build_tile_scheduler(num_decode_tokens)

        return DeepseekSparseSWAMetadata(
            seq_lens=seq_lens,
            query_start_loc=query_start_loc,
            query_start_loc_cpu=query_start_loc_cpu,
            block_table=block_table,
            slot_mapping=slot_mapping,
            is_valid_token=is_valid_token,
            token_to_req_indices=token_to_req_indices,
            decode_swa_indices=self.decode_swa_indices[:num_decode_tokens],
            decode_swa_lens=self.decode_swa_lens[:num_decode_tokens],
            block_size=self.block_size,
            num_decodes=num_decodes,
            num_prefills=num_prefills,
            num_decode_tokens=num_decode_tokens,
            num_prefill_tokens=num_prefill_tokens,
            tile_sched_swaonly=tile_sched[_LAYER_TYPE_SWAONLY],
            tile_sched_c4a=tile_sched[_LAYER_TYPE_C4A],
            tile_sched_c128a=tile_sched[_LAYER_TYPE_C128A],
            **deepseek_v4_fields,
        )

    def build_tile_scheduler(
        self, num_decode_tokens: int
    ) -> dict[str, FlashMLASchedMeta | None]:
        """
        为每种存在的 DeepSeekV4 层类型分配一个空的 FlashMLASchedMeta。

        返回的实例的 tile_scheduler_metadata / num_splits 设为 None；
        FlashMLA C++ decode 路径将在每种类型的第一个 flash_mla_with_kvcache 调用时
        分配它们并运行 tile-scheduler planner。后续同类型调用重用计划，
        因为 tensor（和 have_initialized）已在结构上填充。

        当此步没有 decode token 时返回全 None，以便 _forward_decode 看到干净的哨兵。
        """
        out: dict[str, FlashMLASchedMeta | None] = {
            _LAYER_TYPE_SWAONLY: None,
            _LAYER_TYPE_C4A: None,
            _LAYER_TYPE_C128A: None,
        }
        if (
            num_decode_tokens == 0
            or current_platform.is_rocm()
            or current_platform.is_xpu()
        ):
            return out
        for layer_type in self._layer_types:
            # get_mla_metadata() 是官方 FlashMLA 入口点，返回新的空 FlashMLASchedMeta；
            # 使用它保持此调用点与已通过相同 stub 的其他 vLLM FlashMLA 后端对齐。
            out[layer_type] = get_mla_metadata()[0]
        return out

    def _build_deepseek_v4_metadata(
        self,
        num_decodes: int,
        num_prefills: int,
        seq_lens: torch.Tensor,
        query_start_loc: torch.Tensor,
    ) -> dict[str, torch.Tensor | None]:
        """
        在元数据构建阶段预计算 DeepSeekV4 prefill 元数据。

        返回关键字参数字典，传递给 DeepseekSparseSWAMetadata 构造函数。

        注意：C128A topk 索引由 FlashMLASparse 构建器（拥有 C128A block_table）
        计算，不在此处。
        """
        result: dict[str, torch.Tensor | None] = {}

        # --- Prefill query 元数据（单个 Triton 内核 + CPU 切片）---
        if num_prefills > 0:
            pfx_gather_lens = torch.empty(
                num_prefills, dtype=torch.int32, device=seq_lens.device
            )
            _compute_prefill_metadata_kernel[(1,)](
                pfx_gather_lens,
                seq_lens,
                query_start_loc,
                num_prefills,
                num_decodes,
                self.window_size,
                BLOCK_SIZE=triton.next_power_of_2(num_prefills),
            )

            result["prefill_seq_lens"] = seq_lens[num_decodes:]
            result["prefill_gather_lens"] = pfx_gather_lens

        return result


@triton.jit
def _compute_prefill_metadata_kernel(
    # 输出
    prefill_gather_lens_ptr,
    # 输入
    seq_lens_ptr,
    query_start_loc_ptr,
    num_prefills,
    num_decodes,
    window_size,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton JIT 内核：在单次遍历中计算 prefill gather_lens。

    对于每个 prefill 请求，计算需要 gather 的 KV token 数量：
    gather_len = query_len + min(prefix_len, window_size - 1)
    """
    offset = tl.arange(0, BLOCK_SIZE)
    mask = offset < num_prefills

    seq_len = tl.load(seq_lens_ptr + num_decodes + offset, mask=mask)
    qsl_start = tl.load(query_start_loc_ptr + num_decodes + offset, mask=mask)
    qsl_end = tl.load(query_start_loc_ptr + num_decodes + offset + 1, mask=mask)

    query_len = qsl_end - qsl_start
    prefix_len = seq_len - query_len
    gather_len = query_len + tl.minimum(prefix_len, window_size - 1)

    tl.store(prefill_gather_lens_ptr + offset, gather_len, mask=mask)


@triton.jit
def _compute_swa_indices_and_lens_kernel(
    swa_indices_ptr,
    swa_indices_stride,
    swa_lens_ptr,
    window_size,
    query_start_loc_ptr,
    seq_lens_ptr,
    token_to_req_indices_ptr,
    is_valid_token_ptr,
    block_table_ptr,
    block_table_stride,
    block_size,
    TRITON_BLOCK_SIZE: tl.constexpr,
):
    """
    Triton JIT 内核：为 decode token 计算 SWA 索引和长度。

    每个 Triton 程序处理一个 decode token。对于该 token：
    1. 计算其在序列中的位置 pos
    2. 确定滑动窗口范围 [start_pos, end_pos)
    3. 通过 block_table 查找每个窗口位置的物理 slot ID
    4. 存储 SWA 索引和长度

    这使得后续的注意力内核可以只关注窗口内的 KV token。
    """
    token_idx = tl.program_id(0)
    is_valid = tl.load(is_valid_token_ptr + token_idx)
    if not is_valid:
        tl.store(swa_lens_ptr + token_idx, 0)
        return

    req_idx = tl.load(token_to_req_indices_ptr + token_idx)

    query_start = tl.load(query_start_loc_ptr + req_idx)
    query_end = tl.load(query_start_loc_ptr + req_idx + 1)
    query_len = query_end - query_start

    seq_len = tl.load(seq_lens_ptr + req_idx)
    prefix_len = seq_len - query_len

    pos = prefix_len + token_idx - query_start
    start_pos = tl.maximum(pos - window_size + 1, 0)
    end_pos = pos + 1

    swa_len = end_pos - start_pos
    tl.store(swa_lens_ptr + token_idx, swa_len)

    for i in range(0, window_size, TRITON_BLOCK_SIZE):
        offset = i + tl.arange(0, TRITON_BLOCK_SIZE)

        pos_offset = start_pos + offset
        block_indices = pos_offset // block_size
        block_numbers = tl.load(
            block_table_ptr + req_idx * block_table_stride + block_indices,
            mask=pos_offset < end_pos,
        )
        block_offsets = pos_offset % block_size
        slot_ids = block_numbers * block_size + block_offsets

        slot_ids = tl.where(offset < swa_len, slot_ids, -1)
        tl.store(
            swa_indices_ptr + token_idx * swa_indices_stride + offset,
            slot_ids,
            mask=offset < window_size,
        )
