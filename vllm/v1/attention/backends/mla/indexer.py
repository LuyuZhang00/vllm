# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
MLA 索引器后端模块。

本模块实现了 DeepSeek V3.2/V4 模型的 MLA 索引器后端，负责为稀疏注意力计算
topk 索引。索引器是稀疏注意力的核心组件，它决定每个 query token 应该关注哪些
KV token。

核心功能：
1. 对于 decode token：使用 paged MQA logits 计算 topk 索引
2. 对于 prefill token：使用 chunked prefill 计算 topk 索引
3. 支持 MTP（Multi-Token Prediction）的多 token decode
4. 支持 FP4/FP8 索引器缓存（Blackwell GPU）

架构概述：
- DeepseekV32IndexerBackend: V3.2 索引器后端定义
- DeepseekV4IndexerBackend: V4 索引器后端定义（继承 V3.2）
- DeepseekV32IndexerMetadataBuilder: 元数据构建器
- build_prefill_chunk_metadata: 构建 prefill chunk 元数据的辅助函数

处理流程：
1. 调度器将请求分为 decode 和 prefill 两部分
2. 对于 decode：展开 seq_lens/block_table 以支持 MTP 多 token decode
3. 对于 prefill：将请求分块以适应 workspace 和 logits 内存限制
4. 构建 Triton 内核所需的元数据（cu_seqlens, token_to_seq 等）
"""

from dataclasses import dataclass

import torch

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.deep_gemm import (
    get_paged_mqa_logits_metadata,
    has_deep_gemm,
)
from vllm.utils.math_utils import cdiv
from vllm.utils.platform_utils import num_compute_units
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.mla.compressor_utils import get_compressed_slot_mapping
from vllm.v1.attention.backends.utils import (
    split_decodes_and_prefills,
)
from vllm.v1.kv_cache_interface import AttentionSpec, MLAAttentionSpec
from vllm.v1.worker.cp_utils import get_total_cp_world_size

logger = init_logger(__name__)


@triton.jit
def _prepare_uniform_decode_kernel(
    seq_lens_ptr,
    decode_seq_lens_ptr,
    block_table_ptr,
    block_table_stride,
    expanded_block_table_ptr,
    expanded_bt_stride,
    decode_lens_ptr,
    max_decode_len,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton JIT 内核：为均匀 decode 长度准备展开的元数据。

    当所有 decode 请求具有相同的 decode_len 时使用此内核。
    将每个请求展开为 max_decode_len 个独立的条目，每个条目有：
    - 自己的 seq_len（token j 关注 L - max_decode_len + j + 1 个 KV token）
    - 自己的 block_table 行（从原请求复制）
    - decode_len = 1（每个条目视为单 token decode）

    这使得后续的稀疏注意力内核可以统一处理所有 token。

    参数说明：
    - seq_lens_ptr: 原始序列长度 [num_reqs]
    - decode_seq_lens_ptr: 输出的展开后序列长度 [num_reqs * max_decode_len]
    - block_table_ptr: 原始块表
    - expanded_block_table_ptr: 输出的展开后块表
    - decode_lens_ptr: 输出的 decode 长度（全部为 1）
    - max_decode_len: 最大 decode 长度（MTP 时 > 1）
    """
    idx = tl.program_id(0)
    req_id = idx // max_decode_len
    local_idx = idx % max_decode_len

    # 计算该 token 关注的 KV 数量
    seq_len = tl.load(seq_lens_ptr + req_id)
    per_token_seq_len = seq_len - max_decode_len + local_idx + 1
    tl.store(decode_seq_lens_ptr + idx, per_token_seq_len)

    # 复制块表行
    src = block_table_ptr + req_id * block_table_stride
    dst = expanded_block_table_ptr + idx * expanded_bt_stride
    for i in tl.range(0, expanded_bt_stride, BLOCK_SIZE):
        off = i + tl.arange(0, BLOCK_SIZE)
        mask = off < expanded_bt_stride
        src_block = tl.load(src + off, mask=mask)
        tl.store(dst + off, src_block, mask=mask)

    # 所有 req 现在都有 decode_len = 1
    tl.store(decode_lens_ptr + idx, 1)


def split_indexer_prefill_chunks(
    seq_lens_cpu: torch.Tensor,
    query_lens_cpu: torch.Tensor,
    workspace_size: int,
    max_logits_bytes: int,
    request_offset: int = 0,
) -> list[tuple[slice, slice]]:
    """
    将 prefill 请求拆分为 chunk，用于稀疏索引器。

    遵循两个约束：
    1. N 约束：total_seq_lens <= workspace_size（现有的 O(N) workspace）
    2. Logits 约束：M * N * 4 <= max_logits_bytes

    当单个请求级 chunk 仍然超过 logits 预算时，
    在 query 维度（M）上进行子分块以限制峰值内存。

    参数：
    - seq_lens_cpu: 每个请求的序列长度（CPU tensor）
    - query_lens_cpu: 每个请求的 query 长度（CPU tensor）
    - workspace_size: workspace 大小限制
    - max_logits_bytes: logits 缓冲区最大字节数
    - request_offset: 请求在全局列表中的偏移

    返回：
    - chunks: (req_slice, query_slice) 元组列表
    """
    chunks: list[tuple[slice, slice]] = []
    n = len(seq_lens_cpu)
    max_logits_elems = max_logits_bytes // 4  # float32 每元素 4 字节
    end = 0

    while end < n:
        start, chunk_m, chunk_n = end, 0, 0

        while end < n:
            q, s = query_lens_cpu[end].item(), seq_lens_cpu[end].item()
            new_m, new_n = chunk_m + q, chunk_n + s
            if new_n <= workspace_size and new_m * new_n <= max_logits_elems:
                chunk_m, chunk_n = new_m, new_n
                end += 1
            else:
                break

        # 单个请求可能超过预算，需要在 query 维度上进行子分块
        if end == start:
            chunk_m, chunk_n = query_lens_cpu[end].item(), seq_lens_cpu[end].item()
            end += 1

        req_slice = slice(start + request_offset, end + request_offset)
        max_q = max(1, max_logits_elems // chunk_n) if chunk_n > 0 else chunk_m
        for q_off in range(0, chunk_m, max_q):
            sub_m = min(max_q, chunk_m - q_off)
            chunks.append((req_slice, slice(q_off, q_off + sub_m)))

    return chunks


class DeepseekV32IndexerBackend(AttentionBackend):
    """
    DeepSeek V3.2 索引器后端。

    索引器不是传统的注意力后端，而是为稀疏注意力提供 topk 索引的组件。
    它使用 paged MQA logits 计算每个 token 应该关注的 topk 个最重要的 KV token。
    """

    @staticmethod
    def get_name() -> str:
        return "DEEPSEEK_V32_INDEXER"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        """ROCm 支持块大小 1 和 64，CUDA 只支持 64。"""
        return [1, 64] if current_platform.is_rocm() else [64]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        """支持的 head size：32, 64, 128。"""
        return [32, 64, 128]

    @staticmethod
    def get_builder_cls() -> type["DeepseekV32IndexerMetadataBuilder"]:
        return DeepseekV32IndexerMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        assert num_kv_heads == 1
        return (num_blocks, block_size, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        if include_num_layers_dimension:
            # DeepseekV32Indexer 内核不支持跨层 KV 缓存布局。
            # 恒等排列保持 num_layers 在首位，表示不兼容。
            return (0, 1, 2, 3)
        return (0, 1, 2)


class DeepseekV4IndexerBackend(DeepseekV32IndexerBackend):
    """DeepSeek V4 索引器后端，继承自 V3.2，块大小改为 256。"""

    @staticmethod
    def get_name() -> str:
        return "DEEPSEEK_V4_INDEXER"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [256]


@dataclass
class DeepseekV32IndexerPrefillChunkMetadata:
    """
    单个 prefill chunk 的元数据。

    属性：
    - block_table: 该 chunk 的块表
    - cu_seqlen_ks: 每个 query token 的 KV 序列起始位置（累积和格式）
    - cu_seqlen_ke: 每个 query token 的 KV 序列结束位置（累积和格式）
    - cu_seq_lens: 请求级别的序列长度累积和
    - token_to_seq: 每个 KV token 到其所属请求的映射
    - total_seq_lens: 该 chunk 的总序列长度
    - token_start/token_end: 该 chunk 对应的 token 范围
    - num_reqs: 该 chunk 中的请求数量
    - skip_kv_gather: 是否跳过 KV gather（当 query_slice.start > 0 时）
    """
    block_table: torch.Tensor
    cu_seqlen_ks: torch.Tensor
    cu_seqlen_ke: torch.Tensor
    cu_seq_lens: torch.Tensor
    token_to_seq: torch.Tensor
    total_seq_lens: int
    token_start: int
    token_end: int
    num_reqs: int
    skip_kv_gather: bool = False


@dataclass
class DeepseekV32IndexerPrefillMetadata:
    """prefill 部分的元数据，包含所有 chunk。"""
    chunks: list[DeepseekV32IndexerPrefillChunkMetadata]


@dataclass
class DeepSeekV32IndexerDecodeMetadata:
    """
    decode 部分的元数据。

    属性：
    - block_table: 块表
    - seq_lens: 每个 token 的有效上下文长度
      - 展开路径/普通 decode：1D (batch_size,)
      - 原生 MTP 路径：2D (B, next_n)，其中 [b,j] = L_b - next_n + j + 1
      fp8_fp4_paged_mqa_logits 和 topk 内核都接受两种形状。
    - decode_lens: 每个请求的 decode 长度
    - requires_padding: 是否需要 padding（非均匀 decode 长度时）
    - schedule_metadata: 调度器元数据（DeepGEMM 使用）
    """
    block_table: torch.Tensor
    seq_lens: torch.Tensor
    decode_lens: torch.Tensor
    requires_padding: bool
    schedule_metadata: torch.Tensor


@dataclass
class DeepseekV32IndexerMetadata:
    """
    索引器的整体元数据。

    包含 decode 和 prefill 两部分的元数据，以及公共信息。
    """
    # FIXME (zyongye)
    # 目前的 hacky 方式访问数据，需要移到 chunked meta 中
    seq_lens: torch.Tensor
    max_seq_len: int
    slot_mapping: torch.Tensor

    # MLA 新增（与 FlashAttention 相比）
    # 用于处理 prefill/decode 拆分
    num_decodes: int
    num_decode_tokens: int
    num_prefills: int
    num_prefill_tokens: int

    decode: DeepSeekV32IndexerDecodeMetadata | None = None
    prefill: DeepseekV32IndexerPrefillMetadata | None = None


def get_max_prefill_buffer_size(vllm_config: VllmConfig):
    max_model_len = vllm_config.model_config.max_model_len
    # NOTE(Chen): 40 is a magic number for controlling the prefill buffer size.
    # Each entry is 128 fp8 bytes and 4 scale bytes for a total of 132 bytes.
    # The flashmla_sparse backend uses a workspace size of 5 * max_model_len.
    # The memory usage of the workspace there is 576 * 2 bytes; so we size this as
    # (576 * 2 // 132) * 5 = 40 to maximize this workspace size while still fitting
    # within the flashmla_sparse workspace.
    # For DeepSeek-V3.2, the max_model_len is 163840.
    #   40 * 163840 * 132 = 865075200 bytes = 825 MB
    # 计算 prefill buffer 的最大大小。40 是控制 buffer 大小的魔法数字。
    return max_model_len * 40


class DeepseekV32IndexerMetadataBuilder(AttentionMetadataBuilder):
    """
    DeepSeek V3.2 索引器的元数据构建器。

    负责构建索引器内核所需的元数据，包括：
    1. Decode 部分：展开 seq_lens/block_table 以支持 MTP
    2. Prefill 部分：将请求分块并构建 chunk 元数据
    3. 压缩 KV 缓存支持（DeepSeekV4）

    支持的特性：
    - 混合 decode/prefill batch
    - MTP（Multi-Token Prediction）多 token decode
    - 分块 prefill
    - FP4/FP8 索引器缓存（Blackwell GPU）
    """
    reorder_batch_threshold: int = 1
    natively_supported_next_n_fp4: list[int] = [1, 2]
    # TODO (matt): integrate kernel with next_n = 4 support
    # TODO (matt): 集成 next_n = 4 支持的内核

    @classmethod
    def get_cudagraph_support(
        cls,
        vllm_config: VllmConfig,
        kv_cache_spec: AttentionSpec,
    ) -> AttentionCGSupport:
        """返回 CUDA Graph 支持级别：均匀 batch。"""
        return AttentionCGSupport.UNIFORM_BATCH

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        scheduler_config = self.vllm_config.scheduler_config
        # NOTE(Chen):an estimated max size of flattened_kv. Need to double check.
        # 注意：flattened_kv 的估计最大大小，需要再次检查。
        self.max_prefill_buffer_size = get_max_prefill_buffer_size(self.vllm_config)
        self.num_speculative_tokens = (
            self.vllm_config.speculative_config.num_speculative_tokens
            if self.vllm_config.speculative_config
            else 0
        )
        self.use_fp4_indexer_cache = (
            self.vllm_config.attention_config.use_fp4_indexer_cache
        )

        assert (
            current_platform.is_device_capability_family(100)
            or not self.use_fp4_indexer_cache
        ), (
            "use_fp4_indexer_cache requires Blackwell datacenter GPUs "
            "(sm_10x, e.g. B200/GB200); sm_120 (consumer Blackwell) and "
            "earlier architectures are not supported."
        )

        next_n = self.num_speculative_tokens + 1
        self.reorder_batch_threshold += self.num_speculative_tokens
        # NOTE(zyongye) fp4 indexer cache only natively supports next_n in
        # natively_supported_next_n_fp4; for other next_n values we fall back
        # to the flattening path. Outside the SM100 datacenter family the FP8
        # paged MQA logits kernel has the same [1, 2] constraint (deepgemm
        # smxx_fp8_fp4_paged_mqa_logits.hpp:233), so flatten there too.
        # FP4 索引器缓存原生支持的 next_n 值有限，其他值回退到展开路径。
        self.use_flattening = (
            self.use_fp4_indexer_cache
            or not current_platform.is_device_capability_family(100)
        ) and next_n not in self.natively_supported_next_n_fp4

        sm_count = num_compute_units(self.device.index)
        self.num_sms = sm_count

        self.offsets_buffer = torch.arange(
            next_n, device=self.device, dtype=torch.int32
        )
        self.decode_lens_buffer = torch.zeros(
            (scheduler_config.max_num_batched_tokens,),
            dtype=torch.int32,
            device=self.device,
        )
        # decode seq_lens 的共享 workspace。原生 MTP 在运行时将其视为
        # (B, max_decode_len)，即使 max_decode_len 小于 next_n 也能保持
        # context_lens 连续。
        self.decode_seq_lens_buffer = torch.zeros(
            (scheduler_config.max_num_batched_tokens,),
            dtype=torch.int32,
            device=self.device,
        )
        self.arange_buffer = torch.arange(
            max(
                scheduler_config.max_num_seqs * next_n,
                scheduler_config.max_num_batched_tokens,
            ),
            dtype=torch.int32,
            device=self.device,
        )
        max_num_blocks_per_req = cdiv(
            self.vllm_config.model_config.max_model_len,
            self.kv_cache_spec.block_size * get_total_cp_world_size(),
        )
        self.expanded_block_table_buffer = torch.zeros(
            (
                scheduler_config.max_num_batched_tokens,
                max_num_blocks_per_req,
            ),
            dtype=torch.int32,
            device=self.device,
        )

        # 参见：DeepGMM/csrc/apis/attention.hpp
        self.scheduler_metadata_buffer = torch.empty(
            (self.num_sms + 1, 2), dtype=torch.int32, device=self.device
        )

        # KV 压缩。默认为 1 表示不压缩。
        self.compress_ratio = 1
        # 获取 DeepseekV4 的 compress_ratio
        if isinstance(self.kv_cache_spec, MLAAttentionSpec):
            self.compress_ratio = self.kv_cache_spec.compress_ratio

        # 为 CUDA Graph 兼容性预分配缓冲区
        if self.compress_ratio > 1:
            # compress_ratio > 1（DeepseekV4）
            # 压缩 slot mapping 输出缓冲区
            self.compressed_slot_mapping_buffer = torch.zeros(
                (scheduler_config.max_num_batched_tokens,),
                dtype=torch.int64,
                device=self.device,
            )
            # decode 路径中压缩 seq_lens 的缓冲区
            self.expanded_seq_lens_buffer = torch.zeros(
                (scheduler_config.max_num_batched_tokens,),
                dtype=torch.int32,
                device=self.device,
            )

    def _prepare_decode_tensors(
        self,
        seq_lens: torch.Tensor,
        block_table: torch.Tensor,
        decode_lens: torch.Tensor,
        decode_lens_cpu: torch.Tensor,
        query_start_loc: torch.Tensor,
        num_decodes: int,
        num_decode_tokens: int,
        use_native: bool,
        next_n: int,
        max_decode_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, bool]:
        """
        为 decode 内核展开 seq_lens/block_table/decode_lens。

        展开路径（非 use_native, max_decode_len > 1）：
          每个多 token decode 请求被展开为独立的单 token 条目，
          使内核始终看到 next_n=1。

        原生路径（use_native 或 max_decode_len == 1）：
          普通 decode 或带有 2D 每 token 上下文长度的 spec-decode。

        返回：(seq_lens, block_table, decode_lens, batch_size, requires_padding)
        - seq_lens：展开/普通为 1D (batch_size,)，原生 MTP 为 2D (B, max_decode_len)
        """
        min_decode_len = int(decode_lens_cpu.min().item())
        if not use_native and max_decode_len > 1:
            assert self.decode_seq_lens_buffer.dim() == 1
            if min_decode_len == max_decode_len:
                # 均匀 decode 长度
                num_decode_tokens = num_decodes * max_decode_len
                _prepare_uniform_decode_kernel[(num_decode_tokens,)](
                    seq_lens,
                    self.decode_seq_lens_buffer,
                    block_table,
                    block_table.stride(0),
                    self.expanded_block_table_buffer,
                    self.expanded_block_table_buffer.stride(0),
                    self.decode_lens_buffer,
                    max_decode_len,
                    BLOCK_SIZE=1024,
                )
                self.decode_seq_lens_buffer[num_decode_tokens:] = 0
                seq_lens = self.decode_seq_lens_buffer[:num_decode_tokens]
                block_table = self.expanded_block_table_buffer[:num_decode_tokens]
                decode_lens = self.decode_lens_buffer[:num_decode_tokens]
                return seq_lens, block_table, decode_lens, num_decode_tokens, False
            else:
                # 可变 decode 长度
                # 假设有 4 个请求，seq_lens [10, 7, 12, 0]（最后一个是 padding），
                # decode_lens [3, 1, 4, 0]。上下文长度为
                # [10-3, 7-1, 12-4, 0-0] = [7, 6, 8, 0]。

                # 3 + 1 + 4 + 0 = 8
                actual_expanded = int(decode_lens_cpu.sum().item())

                # 将 expanded_base 和 expanded_offsets 融合为单个 repeat_interleave：
                # seq_len_i = (context_start[b] - query_start_loc[b]) + arange[i] + 1
                # 其中 context_start[b] = seq_lens[b] - decode_lens[b]。
                # 示例：offsets = [7-0, 6-3, 8-4, 0-8] = [7, 3, 4, -8]
                # expanded_offsets  = [7, 7, 7, 3, 4, 4, 4, 4]
                # result            = [8, 9, 10, 7, 9, 10, 11, 12]
                expanded_offsets = torch.repeat_interleave(
                    seq_lens - decode_lens - query_start_loc,
                    decode_lens,
                    output_size=actual_expanded,
                )

                # [8, 9, 10, 7, 9, 10, 11, 12, ...] 其中 ... 是未使用的 buffer 空间
                self.decode_seq_lens_buffer[:actual_expanded] = (
                    expanded_offsets + self.arange_buffer[:actual_expanded] + 1
                )
                self.decode_seq_lens_buffer[actual_expanded:] = 0
                seq_lens = self.decode_seq_lens_buffer[:num_decode_tokens]

                # 为每个展开的条目提供与原请求相同的块表行
                self.expanded_block_table_buffer[:actual_expanded] = (
                    torch.repeat_interleave(
                        block_table, decode_lens, dim=0, output_size=actual_expanded
                    )
                )
                if actual_expanded < num_decode_tokens:
                    self.expanded_block_table_buffer[
                        actual_expanded:num_decode_tokens, 0
                    ] = 0
                block_table = self.expanded_block_table_buffer[:num_decode_tokens]

                # 所有 req 现在都有 decode_len=1
                self.decode_lens_buffer[:num_decode_tokens] = 1
                decode_lens = self.decode_lens_buffer[:num_decode_tokens]
                return seq_lens, block_table, decode_lens, num_decode_tokens, False
        else:
            # 原生路径：普通 decode（next_n==1）或带有 2D 每 token 上下文长度
            # 的 spec decode（next_n > 1）。
            #
            # 当 decode_lens 不是真正均匀时（例如某些请求由于 padding 或短 prefill
            # 而 decode_len < next_n），sparse_attn_indexer 中的简单 reshape 不会工作。
            # 使用 pack_seq_triton（requires_padding）代替。
            requires_padding = min_decode_len != max_decode_len
            if use_native and next_n > 1:
                assert self.decode_seq_lens_buffer.dim() == 1
                # (B, max_decode_len)：token j 关注
                # L - max_decode_len + j + 1 个 KV token。
                seq_lens_buffer = self.decode_seq_lens_buffer[
                    : num_decodes * max_decode_len
                ].view(num_decodes, max_decode_len)
                seq_lens_buffer[:] = (
                    seq_lens.unsqueeze(1)
                    - max_decode_len
                    + 1
                    + self.offsets_buffer[:max_decode_len]
                )
                seq_lens = seq_lens_buffer
            return seq_lens, block_table, decode_lens, num_decodes, requires_padding

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> DeepseekV32IndexerMetadata:
        """
        构建索引器的元数据。

        主要流程：
        1. 将 batch 拆分为 decode 和 prefill 两部分
        2. 处理压缩 slot mapping（DeepSeekV4）
        3. 构建 prefill chunk 元数据（如果有 prefill 请求）
        4. 构建 decode 元数据（如果有 decode 请求）
        5. 组装并返回 DeepseekV32IndexerMetadata
        """
        num_reqs = common_attn_metadata.num_reqs
        num_tokens = common_attn_metadata.num_actual_tokens
        query_start_loc = common_attn_metadata.query_start_loc
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        seq_lens = common_attn_metadata.seq_lens
        slot_mapping = common_attn_metadata.slot_mapping
        block_table = common_attn_metadata.block_table_tensor

        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(
                common_attn_metadata,
                decode_threshold=self.reorder_batch_threshold,
                require_uniform=not self.use_flattening,
            )
        )

        assert num_decodes + num_prefills == num_reqs
        assert num_decode_tokens + num_prefill_tokens == num_tokens

        compressed_slot_mapping = slot_mapping
        compressed_seq_lens = seq_lens
        if self.compress_ratio > 1:
            compressed_slot_mapping = get_compressed_slot_mapping(
                num_tokens,
                query_start_loc,
                seq_lens,
                block_table,
                self.kv_cache_spec.storage_block_size,
                self.compress_ratio,
                out=self.compressed_slot_mapping_buffer,
            )
            compressed_seq_lens = seq_lens // self.compress_ratio

        prefill_metadata = None
        if num_prefills > 0:
            # 此 CPU 值是 async-spec extend 行的上界。对于 chunking/allocation 是安全的，
            # 因为下面的 CUDA 元数据是从精确的设备 seq_lens 构建的，gather 忽略尾部。
            assert common_attn_metadata.seq_lens_cpu_upper_bound is not None
            seq_lens_cpu = common_attn_metadata.seq_lens_cpu_upper_bound
            compressed_seq_lens_cpu = (
                seq_lens_cpu // self.compress_ratio
                if self.compress_ratio > 1
                else seq_lens_cpu
            )
            prefill_query_lens_cpu = torch.diff(
                query_start_loc_cpu[num_decodes : num_decodes + num_prefills + 1]
            )
            max_logits_bytes = envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024
            # 上界对于 prefill 行是精确的（下面的 `[num_decodes:]` 切片）。
            assert common_attn_metadata.seq_lens_cpu_upper_bound is not None
            seq_lens_cpu = common_attn_metadata.seq_lens_cpu_upper_bound
            chunk_specs = split_indexer_prefill_chunks(
                compressed_seq_lens_cpu[num_decodes:],
                prefill_query_lens_cpu,
                self.max_prefill_buffer_size,
                max_logits_bytes,
                request_offset=num_decodes,
            )

            chunks = []
            for req_slice, query_slice in chunk_specs:
                metadata = build_prefill_chunk_metadata(
                    req_slice.start,
                    req_slice.stop,
                    query_start_loc,
                    query_start_loc_cpu,
                    seq_lens,
                    compressed_seq_lens,
                    compressed_seq_lens_cpu,
                    common_attn_metadata.block_table_tensor,
                    self.compress_ratio,
                    query_slice=query_slice,
                    skip_kv_gather=query_slice.start > 0,
                )
                # 当 total_seq_lens 为 0 时跳过（即没有压缩 token）。
                if metadata is not None:
                    chunks.append(metadata)
            prefill_metadata = DeepseekV32IndexerPrefillMetadata(chunks)

        decode_metadata = None
        if num_decodes > 0:
            torch.diff(
                common_attn_metadata.query_start_loc[: num_decodes + 1],
                out=self.decode_lens_buffer[:num_decodes],
            )
            decode_lens = self.decode_lens_buffer[:num_decodes]
            decode_lens_cpu = torch.diff(
                common_attn_metadata.query_start_loc_cpu[: num_decodes + 1]
            )

            seq_lens = common_attn_metadata.seq_lens[:num_decodes]
            block_table = common_attn_metadata.block_table_tensor[:num_decodes, ...]

            max_decode_len = int(decode_lens_cpu.max().item())
            next_n = 1 + self.num_speculative_tokens
            use_native = not self.use_flattening and max_decode_len <= next_n

            seq_lens, block_table, decode_lens, batch_size, requires_padding = (
                self._prepare_decode_tensors(
                    seq_lens=seq_lens,
                    block_table=block_table,
                    decode_lens=decode_lens,
                    decode_lens_cpu=decode_lens_cpu,
                    query_start_loc=common_attn_metadata.query_start_loc[:num_decodes],
                    num_decodes=num_decodes,
                    num_decode_tokens=num_decode_tokens,
                    use_native=use_native,
                    next_n=next_n,
                    max_decode_len=max_decode_len,
                )
            )

            # 对于 DeepseekV4（compress_ratio > 1），索引器 KV 缓存存储压缩 token。
            # 将未压缩的 seq_lens 转换为压缩的。
            if self.compress_ratio > 1:
                # 当且仅当 seq_lens 别名为 decode_seq_lens_buffer（展开或原生写入时）；
                # 否则别名为 common_attn_metadata。
                seq_lens_is_local_view = (use_native and next_n > 1) or (
                    not use_native and max_decode_len > 1
                )
                if seq_lens_is_local_view:
                    seq_lens //= self.compress_ratio
                else:
                    # 复制以避免修改共享状态；保持 CG 地址稳定。
                    self.expanded_seq_lens_buffer[:num_decodes] = (
                        seq_lens // self.compress_ratio
                    )
                    self.expanded_seq_lens_buffer[num_decodes:num_decode_tokens] = 0
                    seq_lens = self.expanded_seq_lens_buffer[:num_decode_tokens]

            # 非 MTP：deep_gemm paged MQA logits 需要 2D context_lens
            #（csrc/apis/attention.hpp）。Unsqueeze 到 (B, 1) 使下游内核
            # 看到与 MTP 路径相同的 (B, next_n) 布局。
            if seq_lens.dim() == 1:
                seq_lens = seq_lens.unsqueeze(-1)

            # CUDA 设备上需要 DeepGEMM 来计算 paged MQA logits
            if current_platform.is_cuda() and has_deep_gemm():
                self.scheduler_metadata_buffer[:] = get_paged_mqa_logits_metadata(
                    seq_lens,
                    self.kv_cache_spec.storage_block_size,
                    self.num_sms,
                )

            decode_metadata = DeepSeekV32IndexerDecodeMetadata(
                block_table=block_table,
                seq_lens=seq_lens,
                decode_lens=decode_lens,
                requires_padding=requires_padding,
                schedule_metadata=self.scheduler_metadata_buffer,
            )

        attn_metadata = DeepseekV32IndexerMetadata(
            seq_lens=common_attn_metadata.seq_lens,
            max_seq_len=common_attn_metadata.max_seq_len,
            slot_mapping=compressed_slot_mapping,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            prefill=prefill_metadata,
            decode=decode_metadata,
        )

        return attn_metadata


def build_prefill_chunk_metadata(
    start_idx: int,
    end_idx: int,
    query_start_loc: torch.Tensor,
    query_start_loc_cpu: torch.Tensor,
    uncompressed_seq_lens: torch.Tensor,
    compressed_seq_lens: torch.Tensor,
    compressed_seq_lens_cpu: torch.Tensor,
    block_table: torch.Tensor,
    compress_ratio: int,
    query_slice: slice | None = None,
    skip_kv_gather: bool = False,
) -> DeepseekV32IndexerPrefillChunkMetadata | None:
    """
    构建单个 prefill chunk 的元数据。

    为索引器的 prefill 内核准备所需的元数据，包括：
    1. cu_seq_lens: 请求级别的序列长度累积和
    2. token_to_seq: 每个 KV token 到其所属请求的映射
    3. cu_seq_len_ks/ke: 每个 query token 的 KV 序列起始/结束位置

    参数：
    - start_idx/end_idx: 该 chunk 在全局请求列表中的范围
    - query_start_loc: query 起始位置累积和
    - uncompressed_seq_lens: 未压缩的序列长度
    - compressed_seq_lens: 压缩后的序列长度
    - compressed_seq_lens_cpu: 压缩后的序列长度（CPU）
    - block_table: 块表
    - compress_ratio: 压缩比率
    - query_slice: query 维度的切片（用于子分块）
    - skip_kv_gather: 是否跳过 KV gather

    返回：
    - DeepseekV32IndexerPrefillChunkMetadata 或 None（当 total_seq_lens 为 0 时）
    """
    total_seq_lens = compressed_seq_lens_cpu[start_idx:end_idx].sum().item()
    if total_seq_lens == 0:
        return None

    num_reqs = end_idx - start_idx
    device = block_table.device
    token_to_seq = torch.empty(total_seq_lens, dtype=torch.int32, device=device)

    cu_seq_lens = torch.empty(num_reqs + 1, dtype=torch.int32, device=device)
    # 分配给切片避免 CPU 同步。
    cu_seq_lens[:1] = 0
    torch.cumsum(compressed_seq_lens[start_idx:end_idx], dim=0, out=cu_seq_lens[1:])

    query_start_loc = (
        query_start_loc[start_idx : end_idx + 1] - query_start_loc[start_idx]
    )

    total_query_len = int(
        (query_start_loc_cpu[end_idx] - query_start_loc_cpu[start_idx]).item()
    )
    if query_slice is not None:
        qs_start = query_slice.start
        qs_stop = query_slice.stop
    else:
        qs_start = 0
        qs_stop = total_query_len
    output_query_len = qs_stop - qs_start

    cu_seq_len_ks = torch.empty(output_query_len, dtype=torch.int32, device=device)
    cu_seq_len_ke = torch.empty(output_query_len, dtype=torch.int32, device=device)

    _build_prefill_chunk_metadata_kernel[(num_reqs,)](
        query_start_loc,
        uncompressed_seq_lens[start_idx:end_idx],
        cu_seq_lens,
        token_to_seq,
        cu_seq_len_ks,
        cu_seq_len_ke,
        qs_start,
        qs_stop,
        BLOCK_SIZE=1024,
        COMPRESS_RATIO=compress_ratio,
    )

    token_start = query_start_loc_cpu[start_idx].item()
    if query_slice is not None:
        token_end = token_start + qs_stop
        token_start = token_start + qs_start
        skip_kv_gather = skip_kv_gather or qs_start > 0
    else:
        token_end = query_start_loc_cpu[end_idx].item()

    return DeepseekV32IndexerPrefillChunkMetadata(
        cu_seqlen_ks=cu_seq_len_ks,
        cu_seqlen_ke=cu_seq_len_ke,
        cu_seq_lens=cu_seq_lens,
        token_to_seq=token_to_seq,
        total_seq_lens=total_seq_lens,
        block_table=block_table[start_idx:end_idx],
        token_start=token_start,
        token_end=token_end,
        num_reqs=num_reqs,
        skip_kv_gather=skip_kv_gather,
    )


@triton.jit
def _build_prefill_chunk_metadata_kernel(
    # 输入
    query_start_loc_ptr,
    uncompressed_seq_lens_ptr,
    cu_compressed_seq_lens_ptr,
    # 输出
    token_to_seq_ptr,
    cu_compressed_seq_len_ks_ptr,
    cu_compressed_seq_len_ke_ptr,
    query_slice_start,
    query_slice_stop,
    BLOCK_SIZE: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
):
    """
    Triton JIT 内核：构建 prefill chunk 的元数据。

    每个 Triton 程序处理一个请求。为该请求中的每个 query token 计算：
    1. cu_seq_len_ks: 该 token 的 KV 序列起始位置（在累积和中）
    2. cu_seq_len_ke: 该 token 的 KV 序列结束位置
    3. token_to_seq: KV token 到请求的映射

    压缩逻辑：
    - 对于位置 pos 的 query token，其关注的压缩 KV 序列长度为
      (start_pos + 1 + offset) // COMPRESS_RATIO
    - start_pos = uncompressed_seq_len - query_len
    """
    batch_idx = tl.program_id(0)

    query_start = tl.load(query_start_loc_ptr + batch_idx)
    query_end = tl.load(query_start_loc_ptr + batch_idx + 1)
    query_len = query_end - query_start

    seq_start = tl.load(cu_compressed_seq_lens_ptr + batch_idx)
    seq_end = tl.load(cu_compressed_seq_lens_ptr + batch_idx + 1)
    compressed_seq_len = seq_end - seq_start

    uncompressed_seq_len = tl.load(uncompressed_seq_lens_ptr + batch_idx)
    start_pos = uncompressed_seq_len - query_len

    for i in range(0, query_len, BLOCK_SIZE):
        offset = i + tl.arange(0, BLOCK_SIZE)
        abs_pos = query_start + offset
        mask = (
            (offset < query_len)
            & (abs_pos >= query_slice_start)
            & (abs_pos < query_slice_stop)
        )
        out_pos = abs_pos - query_slice_start

        # 计算 cu_seq_len_ks（KV 序列起始位置）
        tl.store(cu_compressed_seq_len_ks_ptr + out_pos, seq_start, mask=mask)

        # 计算 cu_seq_len_ke（KV 序列结束位置）
        seq_len_per_token = (start_pos + 1 + offset) // COMPRESS_RATIO
        tl.store(
            cu_compressed_seq_len_ke_ptr + out_pos,
            seq_start + seq_len_per_token,
            mask=mask,
        )

    # 计算 token_to_seq（KV token 到请求的映射）
    for i in range(0, compressed_seq_len, BLOCK_SIZE):
        offset = i + tl.arange(0, BLOCK_SIZE)
        mask = offset < compressed_seq_len
        tl.store(token_to_seq_ptr + seq_start + offset, batch_idx, mask=mask)
