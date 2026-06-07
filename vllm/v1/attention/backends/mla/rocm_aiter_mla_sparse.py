# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
ROCm AITER MLA 稀疏注意力后端模块。

本模块实现了基于 AMD AITER 的稀疏 MLA 注意力后端，用于 AMD GPU 上的 DeepSeek
稀疏模型推理。

核心特性：
1. 稀疏注意力：只关注索引器选出的 topk 个最重要的 KV token
2. 使用 AITER 的 mla_decode_fwd 内核执行稀疏 decode 注意力
3. 支持 FP8 KV 缓存（fp8, fp8_e4m3）
4. 支持持久化 MLA 元数据以优化性能
5. 使用 Triton 内核将 per-request 索引转换为全局物理索引

与非稀疏版本（rocm_aiter_mla.py）的区别：
- 使用 SparseMLAAttentionImpl 而非 MLACommonImpl
- 需要将 topk 逻辑索引转换为物理索引
- 每个 query token 独立处理（qo_indptr = [0, 1, 2, ..., num_tokens]）
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import torch

from vllm import _custom_ops as ops
from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.mla_attention import (
    get_mla_dims,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
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
from vllm.v1.attention.backends.mla.rocm_aiter_mla import (
    AiterMLAHelper,
)
from vllm.v1.kv_cache_interface import AttentionSpec

if TYPE_CHECKING:
    from vllm.model_executor.models.deepseek_v2 import Indexer
logger = init_logger(__name__)


@triton.jit
def _convert_req_index_to_global_index_kernel(
    req_id_ptr,  # int32 [num_tokens] - 每个 token 的请求 ID
    block_table_ptr,  # int32 [num_requests, max_num_blocks_per_req] - 块表
    token_indices_ptr,  # int32 [num_tokens, NUM_TOPK_TOKENS] - 稀疏 topk 逻辑索引
    cu_seqlens_ptr,  # int32 [num_tokens + 1] - 累积序列长度
    out_ptr,  # int32 [num_tokens, NUM_TOPK_TOKENS] - 输出全局物理索引
    # 形状（尽可能使用编译期常量）
    max_num_blocks_per_req: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,  # 列方向上的 tile 宽度
    # 步幅（以元素为单位）
    bt_stride0,
    bt_stride1,
    ti_stride0,
    ti_stride1,
):
    """
    Triton JIT 内核：将 per-request 逻辑索引转换为全局物理索引。

    与 sparse_utils.py 中的内核类似，但针对 ROCm AITER 稀疏后端进行了适配：
    - 使用 cu_seqlens 而非独立的 out_stride 来确定输出位置
    - 只处理 decode 路径（无 prefill workspace 支持）

    转换公式：
    out[token_id, indice_id] =
        block_table[req_id[token_id], token_indices[token_id, indice_id] // BLOCK_SIZE]
        * BLOCK_SIZE + token_indices[token_id, indice_id] % BLOCK_SIZE
    """
    # program_id(0) -> token_id（行）
    # program_id(1) -> 列 tile 索引
    token_id = tl.program_id(0)
    tile_id = tl.program_id(1)

    # 每个程序覆盖 BLOCK_N 个连续列
    indice_id = tile_id * BLOCK_N + tl.arange(0, BLOCK_N)

    # 加载该 token 的请求 ID（无 mask：grid 精确匹配）
    req = tl.load(req_id_ptr + token_id)

    # 加载累积序列长度以获取该请求的起始索引
    seq_start = tl.load(cu_seqlens_ptr + token_id)
    seq_end = tl.load(cu_seqlens_ptr + token_id + 1)

    if tile_id * BLOCK_N + seq_start >= seq_end:
        return

    # 加载该 tile 的 token 索引
    ti_ptr = token_indices_ptr + token_id * ti_stride0 + indice_id * ti_stride1
    tok = tl.load(ti_ptr)  # int32

    # 只有 token == -1 时应传播为 -1
    is_invalid_tok = tok < 0

    # 计算块 ID 和块内偏移
    block_id = tok // BLOCK_SIZE
    inblock_off = tok % BLOCK_SIZE

    # 保护 block_table 访问
    valid_block = (block_id < max_num_blocks_per_req) & (block_id >= 0)
    bt_ptr = block_table_ptr + req * bt_stride0 + block_id * bt_stride1
    base = tl.load(bt_ptr, mask=valid_block, other=0)

    # 如果 token == -1 或 block_id 越界，输出 0；否则 base * BLOCK_SIZE + offset
    out_val = tl.where(
        is_invalid_tok | (~valid_block), 0, base * BLOCK_SIZE + inblock_off
    )
    out_ptr_ij = out_ptr + seq_start + indice_id
    out_ptr_ij_mask = (seq_start + indice_id) < seq_end

    # 带 mask 存储结果
    tl.store(out_ptr_ij, out_val, mask=out_ptr_ij_mask)


def triton_convert_req_index_to_global_index(
    req_id: torch.Tensor,  # int32 [num_tokens]
    block_table: torch.Tensor,  # int32 [num_requests, max_num_blocks_per_req]
    token_indices: torch.Tensor,  # int32 [num_tokens, NUM_TOPK_TOKENS]
    cu_seqlens: torch.Tensor,  # int32 [num_tokens + 1]
    paged_kv_indices: torch.Tensor,  # int32 [num_tokens * topk] 输出缓冲区
    BLOCK_SIZE: int = 64,
    NUM_TOPK_TOKENS: int = 2048,
    BLOCK_N: int = 128,  # 列方向上的 tile 宽度
):
    """
    ROCm AITER 稀疏后端的索引转换包装函数。

    将稀疏 topk 的 per-request 逻辑索引转换为全局物理索引。
    与 sparse_utils.py 的版本不同，此版本使用 cu_seqlens 来确定输出位置，
    并直接写入 paged_kv_indices 缓冲区。

    参数：
    - req_id: 每个 token 的请求 ID
    - block_table: 块表
    - token_indices: 稀疏 topk 逻辑索引
    - cu_seqlens: 累积序列长度
    - paged_kv_indices: 输出缓冲区
    - BLOCK_SIZE: KV 缓存块大小
    - NUM_TOPK_TOKENS: 每个 token 的 topk 数量
    - BLOCK_N: 列 tile 宽度
    """
    assert req_id.dtype == torch.int32
    assert block_table.dtype == torch.int32
    assert token_indices.dtype == torch.int32
    assert token_indices.shape[1] == NUM_TOPK_TOKENS
    assert NUM_TOPK_TOKENS % BLOCK_N == 0, (
        f"NUM_TOPK_TOKENS ({NUM_TOPK_TOKENS}) must be divisible by BLOCK_N ({BLOCK_N})"
    )
    num_tokens = req_id.shape[0]
    _, max_num_blocks_per_req = block_table.shape
    tiles_per_row = NUM_TOPK_TOKENS // BLOCK_N

    # 确保张量在同一设备上且连续
    req_id_c = req_id.contiguous()
    block_table_c = block_table.contiguous()
    token_indices_c = token_indices.contiguous()

    # 步幅（以元素为单位）
    bt_stride0, bt_stride1 = block_table_c.stride()
    ti_stride0, ti_stride1 = token_indices_c.stride()

    # 精确的 2D grid: tokens x 列 tiles
    grid = (num_tokens, tiles_per_row)

    _convert_req_index_to_global_index_kernel[grid](
        req_id_c,
        block_table_c,
        token_indices_c,
        cu_seqlens,
        paged_kv_indices,
        # 形状 / 编译期常量
        max_num_blocks_per_req,
        BLOCK_SIZE,
        BLOCK_N,
        # 步幅
        bt_stride0,
        bt_stride1,
        ti_stride0,
        ti_stride1,
    )
    return


@triton.jit
def generate_sparse_seqlen_kernel(
    seq_len_ptr,  # [num_seq] - 序列长度
    cu_query_lens_ptr,  # [num_seq + 1] - 累积 query 长度
    out_ptr,  # [num_query_tokens] - 输出稀疏序列长度
    topk_token: tl.constexpr,  # topk token 数量
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton JIT 内核：生成稀疏序列长度。

    对于每个 query token，计算其实际需要关注的 KV token 数量（受 topk 限制）。
    公式：sparse_seqlen = min(context_start + query_offset + 1, topk_token)

    这确保了每个 token 最多只关注 topk 个最重要的 KV token。
    """
    seq_id = tl.program_id(0)
    query_offset = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    query_start = tl.load(cu_query_lens_ptr + seq_id)
    query_end = tl.load(cu_query_lens_ptr + seq_id + 1)
    if query_start + tl.program_id(1) * BLOCK_SIZE > query_end:
        return
    query_len = query_end - query_start
    query_mask = query_offset + query_start < query_end
    seq_len = tl.load(seq_len_ptr + seq_id)
    # 直接返回，因为 out_ptr 是零初始化的。
    if seq_len == 0:
        return
    context_start_point = seq_len - query_len
    sparse_seqlen = context_start_point + query_offset
    sparse_seqlen_masked = tl.where(
        sparse_seqlen + 1 < topk_token, sparse_seqlen + 1, topk_token
    )
    tl.store(
        out_ptr + query_start + query_offset, sparse_seqlen_masked, mask=query_mask
    )


def generate_sparse_seqlen_triton(
    query_lens: torch.Tensor,
    seq_lens: torch.Tensor,
    cu_query_lens: torch.Tensor,
    topk_token: int,
    num_tokens: int,
    max_query_len: int,
):
    """
    生成稀疏序列长度的包装函数。

    为每个 query token 计算其实际需要关注的 KV token 数量，
    受 topk_token 限制。
    """
    num_seqs = query_lens.size(0)
    # 零初始化 tensor 以确保无效位置为零
    out = torch.zeros([num_tokens], dtype=torch.int32, device=query_lens.device)
    block_size = 64
    num_block_per_row = triton.cdiv(max_query_len, block_size)
    grid = (
        num_seqs,
        num_block_per_row,
    )
    generate_sparse_seqlen_kernel[grid](
        seq_lens,
        cu_query_lens,
        out,
        topk_token,
        block_size,
    )
    return out


@triton.jit
def fetch_id_to_ragged_kernel(
    in_tensor_ptr,  # [num_seq, topk]
    cumsum_ptr,  # [num_seq + 1]
    out_tensor_ptr,  # [max_num_seq * topk]
    in_tensor_ptr_stride,
    TOPK: tl.constexpr,
    TOKEN_NUM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton JIT 内核：将按请求组织的索引转换为 ragged 格式。

    将 in_tensor[seq_id, :] 的数据复制到 out_tensor 的连续位置，
    使用 cumsum 来确定每个序列在输出中的起始位置。
    """
    seq_id = tl.program_id(0)
    block_id = tl.program_id(1)
    offset = tl.arange(0, BLOCK_SIZE)
    token_start = tl.load(cumsum_ptr + seq_id)
    token_end = tl.load(cumsum_ptr + seq_id + 1)
    token_num = token_end - token_start
    row_offset = block_id * BLOCK_SIZE
    if row_offset >= token_num:
        return
    in_tensor_offset = seq_id * in_tensor_ptr_stride + row_offset + offset
    in_tensor_mask = (row_offset + offset) < TOPK
    in_tensor_val = tl.load(in_tensor_ptr + in_tensor_offset, mask=in_tensor_mask)
    out_tensor_offset = token_start + row_offset + offset
    out_tensor_mask = (out_tensor_offset < token_end) & in_tensor_mask
    tl.store(out_tensor_ptr + out_tensor_offset, in_tensor_val, mask=out_tensor_mask)


def fetch_id_to_ragged_triton(
    in_tensor: torch.Tensor, cumsum: torch.Tensor, out_tensor: torch.Tensor, topk
):
    """将按请求组织的索引转换为 ragged 格式的包装函数。"""
    num_tokens = in_tensor.size(0)
    block_size = 64
    num_block_per_row = triton.cdiv(topk, block_size)
    grid = (
        num_tokens,
        num_block_per_row,
    )
    fetch_id_to_ragged_kernel[grid](
        in_tensor, cumsum, out_tensor, in_tensor.stride(0), topk, num_tokens, block_size
    )


class ROCMAiterMLASparseBackend(AttentionBackend):
    """
    ROCm AITER MLA 稀疏注意力后端。

    定义了后端的基本属性和支持的配置。
    """
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
    ]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [1, 64]

    @staticmethod
    def get_name() -> str:
        return "ROCM_AITER_MLA_SPARSE"

    @staticmethod
    def get_metadata_cls() -> type["ROCMAiterMLASparseMetadata"]:
        return ROCMAiterMLASparseMetadata

    @staticmethod
    def get_builder_cls() -> type["ROCMAiterMLASparseMetadataBuilder"]:
        return ROCMAiterMLASparseMetadataBuilder

    @staticmethod
    def get_impl_cls() -> type["ROCMAiterMLASparseImpl"]:
        return ROCMAiterMLASparseImpl

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,  # 对于 MLA 假设为 1
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        return (num_blocks, block_size, head_size)

    @classmethod
    def is_mla(cls) -> bool:
        return True

    @classmethod
    def is_sparse(cls) -> bool:
        return True


@dataclass
class ROCMAiterMLASparseMetadata(AttentionMetadata):
    """
    ROCm AITER MLA 稀疏注意力的元数据。

    包含稀疏注意力所需的所有元数据，包括基本映射、
    paged KV 缓存索引结构、以及持久化 MLA 元数据。
    """
    num_reqs: int
    max_query_len: int
    max_seq_len: int

    num_actual_tokens: int  # 不含 padding 的实际 token 数量
    query_start_loc: torch.Tensor
    slot_mapping: torch.Tensor

    block_table: torch.Tensor
    req_id_per_token: torch.Tensor

    qo_indptr: torch.Tensor  # query/output indptr
    paged_kv_last_page_len: torch.Tensor  # 每个请求最后一页的长度
    paged_kv_indices: torch.Tensor  # paged KV 缓存的页面索引
    paged_kv_indptr: torch.Tensor  # paged KV 缓存的 indptr
    attn_out_dtype: torch.dtype

    block_size: int = 1
    topk_tokens: int = 2048

    # 持久化 MLA 元数据（仅在启用持久化模式时填充，
    # 即当 aiter 稀疏 decode 内核支持 work-stealing 分割时）。
    work_meta_data: torch.Tensor | None = None
    work_indptr: torch.Tensor | None = None
    work_info_set: torch.Tensor | None = None
    reduce_indptr: torch.Tensor | None = None
    reduce_final_map: torch.Tensor | None = None
    reduce_partial_map: torch.Tensor | None = None


@dataclass
class ROCMAiterMLASparseMetadataBuilder(
    AttentionMetadataBuilder[ROCMAiterMLASparseMetadata]
):
    """
    ROCm AITER MLA 稀疏注意力的元数据构建器。

    负责构建稀疏注意力内核所需的元数据，包括：
    1. req_id_per_token 映射
    2. 稀疏序列长度（受 topk 限制）
    3. paged KV 缓存的索引结构
    4. 持久化 MLA 元数据
    """
    _cudagraph_support: ClassVar[AttentionCGSupport] = (
        AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
    )

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        self.kv_cache_spec = kv_cache_spec
        self.model_config = vllm_config.model_config
        self.model_dtype = vllm_config.model_config.dtype
        parallel_config = vllm_config.parallel_config
        self.device = device
        max_num_batched_tokens = vllm_config.scheduler_config.max_num_batched_tokens

        self.num_heads = self.model_config.get_num_attention_heads(parallel_config)
        self.mla_dims = get_mla_dims(self.model_config)
        self.topk_tokens = vllm_config.model_config.hf_config.index_topk
        self.max_model_len_tensor = torch.tensor(
            [self.model_config.max_model_len], device=device, dtype=torch.int32
        )
        # 当 indices 不为 None 时，flash_mla_with_kvcache 会忽略此值
        self.dummy_block_table = torch.empty(
            (1, 1), dtype=torch.int32, device=self.device
        )

        self.req_id_per_token_buffer = torch.zeros(
            (vllm_config.scheduler_config.max_num_batched_tokens,),
            dtype=torch.int32,
            device=device,
        )
        self.qo_indptr = torch.arange(
            0, max_num_batched_tokens + 1, dtype=torch.int32, device=device
        )
        self.paged_kv_last_page_len = torch.ones(
            max_num_batched_tokens, dtype=torch.int32, device=device
        )

        # 这两个需要在运行时计算，但仍需准备缓冲区
        self.paged_kv_indices = torch.zeros(
            [max_num_batched_tokens * self.topk_tokens],
            dtype=torch.int32,
            device=device,
        )
        self.paged_kv_indptr = torch.zeros(
            [max_num_batched_tokens + 1], dtype=torch.int32, device=device
        )

        # ----- 持久化 MLA 元数据缓冲区 -----
        # AITER 稀疏 decode 内核支持"持久化"路径，使用预计算的 work-splitting
        # 元数据以获得更好的 CU 间负载均衡。与 rocm_aiter_mla.py 中的方法类似。
        #
        # 在稀疏情况下，每个 query token 是 qo_indptr 中自己的"batch"条目
        #（qo_indptr = [0, 1, 2, ..., num_tokens]），max_qo_len=1。
        # 我们将 get_mla_metadata_info_v1 的 batch_size 填充到 max_num_batched_tokens，
        # 使缓冲区足够大以容纳任何 decode 形状。
        from aiter import dtypes, get_mla_metadata_info_v1

        # AITER 稀疏 MLA 也要求 num_heads >= 16（将在 forward 中由
        # AiterMLAHelper.get_mla_padded_q padding）。
        self._num_attention_heads = max(16, self.num_heads)

        q_dtype = self.model_dtype
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
            max_num_batched_tokens,
            1,
            self._num_attention_heads,
            q_dtype,
            kv_dtype,
            is_sparse=True,
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

        self._prev_req_extent: int = 0
        self._prev_indices_extent: int = 0
        self._prev_metadata_key: tuple | None = None

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> ROCMAiterMLASparseMetadata:
        """
        构建 ROCm AITER MLA 稀疏注意力的元数据。

        主要流程：
        1. 构建 req_id_per_token 映射
        2. 生成稀疏序列长度（受 topk 限制）
        3. 计算 paged KV 缓存的 indptr
        4. 计算持久化 MLA 元数据（如果未变化则跳过）
        5. 组装并返回元数据
        """
        num_tokens = common_attn_metadata.num_actual_tokens
        starts = np.asarray(common_attn_metadata.query_start_loc_cpu, dtype=np.int32)
        seg_lengths = np.diff(starts)
        req_id_per_token = np.repeat(
            np.arange(seg_lengths.shape[0], dtype=np.int32), seg_lengths
        )
        # 只清零缩小的尾部。paged_kv_indptr 由下面的 cumsum 完全重写。
        # new_indices_extent 之后的 paged_kv_indices 条目永远不会被读取
        #（注意力内核只触及 paged_kv_indptr 定义的范围）。
        new_req_extent = int(req_id_per_token.shape[0])
        new_indices_extent = num_tokens * self.topk_tokens
        if self._prev_req_extent > new_req_extent:
            self.req_id_per_token_buffer[new_req_extent : self._prev_req_extent].fill_(
                0
            )
        if self._prev_indices_extent > new_indices_extent:
            self.paged_kv_indices[new_indices_extent : self._prev_indices_extent].fill_(
                0
            )
        self._prev_req_extent = new_req_extent
        self._prev_indices_extent = new_indices_extent
        self.req_id_per_token_buffer[:new_req_extent].copy_(
            torch.from_numpy(req_id_per_token), non_blocking=True
        )
        query_lens = (
            common_attn_metadata.query_start_loc[1:]
            - common_attn_metadata.query_start_loc[:-1]
        )
        seq_lens = common_attn_metadata.seq_lens
        sparse_seqlen = generate_sparse_seqlen_triton(
            query_lens,
            seq_lens,
            common_attn_metadata.query_start_loc,
            self.topk_tokens,
            num_tokens,
            common_attn_metadata.max_query_len,
        )

        torch.cumsum(sparse_seqlen, dim=0, out=self.paged_kv_indptr[1 : num_tokens + 1])
        self.paged_kv_indptr[num_tokens + 1 :].fill_(self.paged_kv_indptr[num_tokens])

        req_id_per_token = self.req_id_per_token_buffer[:num_tokens]
        qo_indptr = self.qo_indptr[: num_tokens + 1]
        paged_kv_last_page_len = self.paged_kv_last_page_len[:num_tokens]
        paged_kv_indptr = self.paged_kv_indptr[: num_tokens + 1]
        paged_kv_indices = self.paged_kv_indices[: num_tokens * self.topk_tokens]

        # ----- 计算持久化 MLA 元数据 -----
        # AITER 稀疏 decode 内核使用 qseqlen=1（每个 query token 视为自己的 batch 条目），
        # 所以持久化元数据可以始终在此预计算。当 work_meta_data 非 None 时，
        # 内核自动切换到持久化 work-stealing 路径。
        # 输出是 (num_tokens, max_query_len, num_heads, min(seq_lens, topk_tokens))
        # 的确定性函数；在 CPU 端指纹化这些值，当无变化时跳过启动。
        num_reqs = common_attn_metadata.num_reqs
        clamped_seq_lens = np.minimum(
            common_attn_metadata.seq_lens_cpu[:num_reqs].numpy(),
            self.topk_tokens,
        )
        metadata_key = (
            num_tokens,
            int(common_attn_metadata.max_query_len),
            self._num_attention_heads,
            clamped_seq_lens.tobytes(),
        )
        if metadata_key != self._prev_metadata_key:
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
                max_seqlen_qo=1,
                uni_seqlen_qo=1,
                fast_mode=True,
            )
            self._prev_metadata_key = metadata_key

        metadata = ROCMAiterMLASparseMetadata(
            num_reqs=common_attn_metadata.num_reqs,
            max_query_len=common_attn_metadata.max_query_len,
            max_seq_len=common_attn_metadata.max_seq_len,
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            query_start_loc=common_attn_metadata.query_start_loc,
            slot_mapping=common_attn_metadata.slot_mapping,
            block_table=common_attn_metadata.block_table_tensor,
            req_id_per_token=req_id_per_token,
            block_size=self.kv_cache_spec.block_size,
            attn_out_dtype=self.model_dtype,
            topk_tokens=self.topk_tokens,
            qo_indptr=qo_indptr,
            paged_kv_last_page_len=paged_kv_last_page_len,
            paged_kv_indices=paged_kv_indices,
            paged_kv_indptr=paged_kv_indptr,
            work_meta_data=self._mla_work_meta_data,
            work_indptr=self._mla_work_indptr,
            work_info_set=self._mla_work_info_set,
            reduce_indptr=self._mla_reduce_indptr,
            reduce_final_map=self._mla_reduce_final_map,
            reduce_partial_map=self._mla_reduce_partial_map,
        )
        return metadata


# 参考实现，来自
# https://github.com/deepseek-ai/FlashMLA/blob/main/tests/test_flash_mla_prefill.py#L72
def reference_mla_sparse_prefill(
    q: torch.Tensor, kv: torch.Tensor, indices: torch.Tensor, sm_scale: float, d_v: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    MLA 稀疏 prefill 的参考实现（用于测试和验证）。

    使用 PyTorch 原生操作实现稀疏 MLA 注意力计算，
    用于验证内核实现的正确性。
    """
    import math

    def log2sumexp2(a: torch.Tensor, dim: int) -> torch.Tensor:
        return torch.logsumexp(a * math.log(2), dim=dim) * math.log2(math.e)

    skv = kv.shape[0]
    sq = q.shape[0]
    topk = indices.shape[-1]
    dqk = q.shape[-1]
    indices = indices[:, 0, :]  # [s_q, topk]
    invalid_indices_mask = (indices < 0) | (indices >= skv)
    indices[invalid_indices_mask] = 0
    qs = q  # [s_q, h_q, d_qk]
    kvs = kv[:, 0, :][indices].view(sq, topk, dqk)  # [s_q, topk, d_qk]

    attn_score = (qs @ kvs.transpose(1, 2)).float()  # [s_q, h_q, topk]
    attn_score.masked_fill_(invalid_indices_mask.unsqueeze(1), float("-inf"))
    attn_score *= sm_scale * math.log2(math.e)
    lse = log2sumexp2(attn_score, dim=-1)  # [s_q, h_q]
    attn_score = torch.exp2(attn_score - lse.unsqueeze(-1))  # [s_q, h_q, topk]
    result = attn_score.to(q.dtype) @ kvs[:, :, :d_v]
    return (result, lse)


class ROCMAiterMLASparseImpl(SparseMLAAttentionImpl[ROCMAiterMLASparseMetadata]):
    """
    ROCm AITER MLA 稀疏注意力的具体实现。

    使用 AITER 的 mla_decode_fwd 内核执行稀疏 decode 注意力计算。
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
        topk_indice_buffer: torch.Tensor | None = None,
        indexer: "Indexer | None" = None,
        **mla_args,
    ) -> None:
        AiterMLAHelper.check_num_heads_validity(num_heads)

        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.kv_lora_rank: int = mla_args["kv_lora_rank"]
        self.softmax_scale = scale
        assert indexer is not None
        self.topk_indices_buffer: torch.Tensor | None = indexer.topk_indices_buffer

    def _forward_mla(
        self,
        layer: AttentionLayer,
        q: torch.Tensor,  # [sq, heads, d_qk]
        kv_c_and_k_pe_cache: torch.Tensor,  # [blocks, heads, d_qk]
        attn_metadata: ROCMAiterMLASparseMetadata,
    ) -> torch.Tensor:
        """
        AITER MLA 稀疏 decode 的核心前向传播。

        流程：
        1. 分配输出 tensor
        2. 构建 mla_decode_fwd 的 kwargs（包括持久化元数据）
        3. 调用 mla_decode_fwd 计算注意力
        4. 去 padding 后返回输出
        """
        num_tokens = q.shape[0]
        mla_num_heads = AiterMLAHelper.get_actual_mla_num_heads(self.num_heads)
        output = torch.empty(
            [num_tokens, mla_num_heads, self.kv_lora_rank],
            dtype=attn_metadata.attn_out_dtype,
            device=q.device,
        )

        # 构建 kwargs 并在计算了持久化 MLA 元数据时传递它。
        # AITER mla_decode_fwd 在给出 work_meta_data 时切换到
        # work-stealing 持久化内核路径。
        mla_kwargs: dict = dict(
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
            q,
            kv_c_and_k_pe_cache,
            output,
            self.scale,
            attn_metadata.qo_indptr,
            1,
            attn_metadata.paged_kv_indptr,
            attn_metadata.paged_kv_indices,
            attn_metadata.paged_kv_last_page_len,
            **mla_kwargs,
        )

        return AiterMLAHelper.get_mla_unpadded_o(self.num_heads, output)

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: ROCMAiterMLASparseMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        稀疏 MLA 注意力的主前向传播入口。

        对于稀疏 FlashMLA 内核，prefill 和 decode 都使用 MQA 576/512 方法。

        流程：
        1. 如果 q 是元组，拼接为完整 q
        2. 将 topk 逻辑索引转换为全局物理索引
        3. 如果使用 FP8 缓存，量化 query
        4. 将 q padding 到至少 16 个头
        5. 调用 _forward_mla 计算注意力
        """
        # NOTE(lucas): for the sparse FlashMLA kernels the kernels want to use
        # MQA 576/512 approach for both prefill and decode
        # 对于稀疏 FlashMLA 内核，prefill 和 decode 都使用 MQA 576/512 方法

        # 如果 q 是元组（ql_nope, q_pe），拼接
        if isinstance(q, tuple):
            q = torch.cat(q, dim=-1)

        num_actual_toks = attn_metadata.num_actual_tokens

        # 获取 topk 索引
        assert self.topk_indices_buffer is not None
        topk_indices = self.topk_indices_buffer[:num_actual_toks]

        # 将 per-request 逻辑索引转换为全局物理索引
        triton_convert_req_index_to_global_index(
            attn_metadata.req_id_per_token,
            attn_metadata.block_table,
            topk_indices,
            attn_metadata.paged_kv_indptr,
            attn_metadata.paged_kv_indices,
            BLOCK_SIZE=attn_metadata.block_size,
            NUM_TOPK_TOKENS=attn_metadata.topk_tokens,
        )

        # 将 latent 和 rope 写入 kv 缓存
        fp8_attention = self.kv_cache_dtype.startswith("fp8")
        if fp8_attention:
            original_q_shape = q.shape
            kv_c_and_k_pe_cache = kv_c_and_k_pe_cache.view(current_platform.fp8_dtype())
            q, _ = ops.scaled_fp8_quant(q.view(q.shape[0], -1), layer._q_scale)
            q = q.view(original_q_shape)
        mla_padded_q = AiterMLAHelper.get_mla_padded_q(self.num_heads, q)
        attn_out = self._forward_mla(
            layer, mla_padded_q, kv_c_and_k_pe_cache, attn_metadata
        )

        return attn_out, None
