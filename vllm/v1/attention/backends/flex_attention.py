# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention layer with FlexAttention."""

# ===========================================================================
# FlexAttention 后端整体说明
# ===========================================================================
# 本文件实现了 vLLM V1 的 FlexAttention 注意力后端，基于 PyTorch 原生的
# flex_attention 算子。FlexAttention 是一种可编程的注意力内核，允许用户通过
# mask_mod（掩码修改函数）和 score_mod（分数修改函数）自定义注意力行为，
# 支持因果遮罩、滑动窗口、前缀 LM、双向注意力（encoder-only）等多种模式。
#
# 核心设计思路：
# 1. FlexAttention 使用 BlockMask（块级稀疏掩码）来跳过全被遮罩的 KV 块，
#    避免不必要的计算，从而提高效率。
# 2. vLLM 使用 PagedAttention 的分页 KV cache，逻辑 block 和物理 block 不同，
#    因此需要将物理索引转换为逻辑索引，再传给 mask_mod / score_mod 函数。
# 3. 本后端通过 physical_to_logical_mapping 建立物理到逻辑的反向映射，
#    并在 mask_mod 中进行索引转换，使得用户自定义的 mask_mod 可以直接
#    使用逻辑索引（不感知底层物理分页布局）。
#
# 主要类：
# - FlexAttentionBackend:      后端注册类，声明支持的数据类型、注意力类型等
# - FlexAttentionMetadata:     注意力元数据，包含 block mask、mask_mod、索引映射等
# - FlexAttentionMetadataBuilder: 元数据构建器，每步调度后构建 FlexAttentionMetadata
# - FlexAttentionImpl:         注意力实现类，执行实际的 forward 计算
#
# 关键函数：
# - physical_to_logical_mapping(): 构建物理 block 到逻辑 block 的反向映射
# - unique_static_unsorted():      静态去重，用于构建 BlockMask 的 KV 索引
# - causal_mask_mod():              因果遮罩（decoder 默认）
# - bidirectional_mask_mod():       双向遮罩（encoder-only）
# - get_kernel_options():           根据硬件和配置选择内核参数
# ===========================================================================

import math
from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property
from typing import ClassVar, NamedTuple

import torch
import torch._dynamo.decorators
import torch.nn.functional as F
from torch.nn.attention.flex_attention import (
    BlockMask,
    _mask_mod_signature,
    _score_mod_signature,
    and_masks,
    create_block_mask,
    flex_attention,
    or_masks,
)

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import is_quantized_kv_cache, is_torch_equal_or_newer
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.kv_cache_interface import AttentionSpec, EncoderOnlyAttentionSpec

logger = init_logger(__name__)

# 中文注释：提高 torch.compile 的重编译限制，避免因动态形状变化频繁触发重编译。
# FlexAttention 的 mask_mod 函数签名在不同批次间可能不同，导致需要更多编译缓存。
torch._dynamo.config.recompile_limit = 16

# 中文注释：预编译 create_block_mask，使用 fullgraph=True 确保整个函数被编译为一张计算图，
# mode="reduce-overhead" 通过 CUDA graph 等技术减少 Python 端开销。
create_block_mask_compiled = torch.compile(
    create_block_mask, fullgraph=True, mode="reduce-overhead"
)

# 中文注释：预编译 flex_attention 核心算子，fullgraph=True 确保端到端编译优化。
# 这是 PyTorch 提供的灵活注意力前向内核，支持自定义 mask_mod 和 score_mod。
flex_attention_compiled = torch.compile(flex_attention, fullgraph=True)


def _offsets_to_doc_ids_tensor(
    offsets_cpu: torch.Tensor, device: torch.device
) -> torch.Tensor:
    """将 query_start_loc 偏移量转换为每个 token 对应的文档 ID（请求 ID）。

    例如，如果 query_start_loc = [0, 5, 12]，表示请求 0 有 token 0-4，
    请求 1 有 token 5-11，则返回 [0,0,0,0,0, 1,1,1,1,1,1,1]。

    这个映射在 FlexAttention 的 mask_mod 中用于确定每个 query token
    属于哪个请求，从而正确地应用请求级遮罩。

    为什么在 CPU 上构建：
    repeat_interleave 的输出长度依赖于输入数据（data-dependent），
    如果在 GPU 上执行会导致 GPU->CPU 同步以获取输出长度，性能很差。
    因此在 CPU 上构建后异步上传到 GPU。
    """
    # 中文注释：在链路中的作用——该函数是 FlexAttention 请求级遮罩的基础。
    # vLLM 将多个请求打包（packing）到一个大序列中，FlexAttention 内核
    # 需要知道每个 token 属于哪个请求，才能正确地阻止跨请求的注意力计算。
    # 该映射存储在 FlexAttentionMetadata.doc_ids 中，被 mask_mod 函数引用。
    # Build on CPU (so `repeat_interleave` doesn't force a GPU->CPU sync to
    # learn the data-dependent output length) and upload non-blocking.
    counts = offsets_cpu[1:] - offsets_cpu[:-1]
    doc_ids = torch.repeat_interleave(
        torch.arange(len(counts), dtype=torch.int32), counts
    )
    return doc_ids.to(device, non_blocking=True)


def pad_to_multiple(x: torch.Tensor, multiple: int, dim: int):
    """将张量在指定维度上填充到 multiple 的整数倍。

    FlexAttention 的 BlockMask 要求 query 和 KV 维度必须能被 block_size 整除，
    因此需要对输入进行填充。填充使用常数 0（对于索引张量，0 通常对应无效位置）。
    """
    # 中文注释：在 _build_block_mask_direct 中被调用，用于将 used_pages 张量
    # 沿 token 维度填充到 q_block_size 的整数倍，以便后续 reshape 为
    # (num_query_groups, max_num_kv_indices) 形状进行块级去重。
    difference = (multiple - (x.shape[dim] % multiple)) % multiple
    if difference == 0:
        return x

    dim = dim if dim >= 0 else x.ndim + dim
    pad_list = []

    for i in range(x.ndim - 1, dim - 1, -1):
        if i == dim:
            pad_list.extend([0, difference])
        else:
            pad_list.extend([0, 0])

    return F.pad(x, pad_list, mode="constant", value=0)


class FlexAttentionBackend(AttentionBackend):
    """FlexAttention 后端注册类。

    该类声明 FlexAttention 后端的能力和限制，包括支持的数据类型、
    注意力类型（decoder / encoder-only）、是否支持级联注意力等。
    同时提供元数据构建器（FlexAttentionMetadataBuilder）和
    实现类（FlexAttentionImpl）的工厂方法。

    注意：
    - forward_includes_kv_cache_update = False，表示 FlexAttention 的 forward
      不自动包含 KV cache 更新，需要单独调用 do_kv_cache_update。
    - 不支持级联注意力（cascade attention），即不支持公共前缀 KV 共享。
    - 支持 batch invariance（批次不变性），即相同输入在不同批次大小下产生相同结果。
    """
    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
    ]

    # 中文注释：False 表示 KV cache 的写入不在 forward 中自动完成，
    # 而是由 Model Runner 在 forward 之前单独调用 do_kv_cache_update。
    forward_includes_kv_cache_update: bool = False

    @staticmethod
    def get_name() -> str:
        return "FLEX_ATTENTION"

    @classmethod
    def supports_non_causal(cls) -> bool:
        return True

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        """FlexAttention supports both decoder and encoder-only attention."""
        return attn_type in (AttentionType.DECODER, AttentionType.ENCODER_ONLY)

    @classmethod
    def supports_batch_invariance(cls) -> bool:
        return True

    @classmethod
    def supports_mm_prefix(cls) -> bool:
        """FlexAttention supports full attention for image tokens."""
        return True

    @staticmethod
    def get_impl_cls() -> type["FlexAttentionImpl"]:
        return FlexAttentionImpl

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        """返回 FlexAttention 的 KV cache 张量形状。

        形状为 (num_blocks, 2, block_size, num_kv_heads, head_size)，
        其中维度 1 的大小为 2，分别存储 K 和 V。
        这与 FlashAttention 后端的 (num_blocks, 2, block_size, num_kv_heads, head_size) 布局一致，
        但与 FlashMLA 等后端的 (num_layers, num_blocks, 2, ...) 不同。
        """
        return (num_blocks, 2, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        """返回 KV cache 维度的步幅重排序，用于 reshape_and_cache 等操作。

        FlexAttention 的 KV cache 布局是 (num_blocks, 2, block_size, num_kv_heads, head_size)，
        步幅顺序为 (0, 2, 1, 3, 4)，即 block_size 维度排在 K/V 维度之前。
        """
        if include_num_layers_dimension:
            return (1, 0, 3, 2, 4, 5)
        return (0, 2, 1, 3, 4)

    @staticmethod
    def get_builder_cls() -> type["FlexAttentionMetadataBuilder"]:
        return FlexAttentionMetadataBuilder

    @staticmethod
    def use_cascade_attention(*args, **kwargs) -> bool:
        return False

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return []

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(16)]


# @torch.compile(fullgraph=True, mode="reduce-overhead")
# 中文注释：构建物理 block 到逻辑 block 的反向映射。
# 在 vLLM 的 PagedAttention 中，block_table 存储的是 逻辑->物理 的映射，
# 但 FlexAttention 的 mask_mod 需要知道每个物理 KV 位置属于哪个逻辑位置，
# 因此需要构建反向映射（物理->逻辑），使得内核可以通过物理索引查到逻辑索引。
def physical_to_logical_mapping(
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int,
    total_blocks: int,
) -> torch.Tensor:
    """
    Creates an inverse mapping from physical block locations to logical indices.

    The original block_table maps from logical blocks to physical locations:

    Logical to Physical (Original block_table):
    ┌───────────────────────────────────────────┐
    │ Request 0:                                │
    │                                           │
    │ Logical Blocks:  0  1  2  3  4  5  6  7   │
    │                  │  │  │  │  │  │  │  │   │
    │                  v  v  v  v  v  v  v  v   │
    │ Physical Blocks: 3  5  1  7  4  2  0  6   │
    └───────────────────────────────────────────┘

    This function creates the inverse mapping:

    Physical to Logical (Inverse mapping):
    ┌───────────────────────────────────────────┐
    │ Request 0:                                │
    │                                           │
    │ Physical Blocks: 0  1  2  3  4  5  6  7   │
    │                  │  │  │  │  │  │  │  │   │
    │                  v  v  v  v  v  v  v  v   │
    │ Logical Blocks:  6  2  5  0  4  1  7  3   │
    └───────────────────────────────────────────┘

    If multiple logical blocks map to the same physical block,
    this function returns the latest (maximum) logical block index.

    If a physical block is not mapped to by any logical block,
    its value in the result will be -1.

    IMPORTANT: Garbage Value Protection
    ────────────────────────────────────
    The block_table tensor may contain garbage values in unused positions
    (beyond the actual sequence length). For example, if a sequence only
    needs 3 blocks but the table has space for 8:

        block_table[0] = [10, 25, 7, 999, 1234, 888, ...]
                                    ^^^^^^^^^^^^^^^^^^^^
                                    garbage values

    These garbage values can cause issues because:
    1. They may map to valid physical blocks by coincidence
    2. The scatter_ operation will assign them logical indices
    3. Later attention computations may incorrectly access these blocks

    To prevent this, we use seq_lens and block_size to mask out unused
    entries, ensuring only valid block references are processed.

    IMPORTANT: Reused physical blocks (sliding-window / hybrid attention)
    ────────────────────────────────────────────────────────────────────
    For some attention types, physical cache blocks can be reused over time.
    This can cause the same physical block id to appear multiple times in a row
    of `block_table` at different logical block indices. In that case, only the
    latest logical block index corresponds to the current contents of that
    physical block. Therefore, the inverse mapping must pick the maximum logical
    block index for each physical block id.

    Args:
        block_table: Tensor of shape [max_reqs, max_num_blocks]
            mapping logical blocks to physical locations. May contain
            garbage values in unused positions.
        seq_lens: Tensor of sequence lengths for each request. Used to
            determine how many blocks are actually needed per sequence.
        block_size: Size of each block in tokens. Used with seq_lens to
            compute the number of valid blocks per sequence.
        total_blocks: Total number of physical blocks available

    Returns:
        A tensor of shape [max_reqs, total_blocks] where each entry
        physical_to_logical[req_id, physical_block] contains the logical
        block index for that physical block, or -1 if unused.
    """
    max_reqs, max_num_blocks = block_table.shape
    device = block_table.device

    # 中文注释：初始化反向映射表，默认值为 -1（表示该物理 block 未被任何逻辑 block 使用）。
    # 形状为 [max_reqs, total_blocks]，即每个请求对所有物理 block 都有一个映射条目。
    physical_to_logical = torch.full(
        (max_reqs, total_blocks), -1, dtype=torch.long, device=device
    )

    # Only process valid blocks to avoid garbage values
    # 中文注释：计算每个序列实际需要的 block 数量，并构建有效 block 掩码。
    # block_table 中超出序列实际长度的位置包含垃圾值，需要通过掩码排除。
    num_blocks_per_seq: torch.Tensor = cdiv(seq_lens, block_size)
    mask = (
        torch.arange(max_num_blocks, device=device)[None, :]
        < num_blocks_per_seq[:, None]
    )

    # 中文注释：将无效位置的 block_table 值置为 0（避免垃圾值影响 scatter），
    # 同时将无效位置的逻辑索引也置为 0（后续会被掩码过滤掉）。
    valid_block_table = torch.where(mask, block_table, 0)
    valid_logical_indices = torch.where(
        mask, torch.arange(max_num_blocks, device=device)[None, :], 0
    )

    # 中文注释：使用 scatter_reduce_ 的 amax（取最大值）模式构建反向映射。
    # 当同一个物理 block 被多个逻辑 block 映射时（如滑动窗口场景中的复用），
    # 取最大的逻辑 block 索引，因为最新的逻辑 block 对应当前物理 block 的内容。
    # 这避免了多次写入冲突，一次 scatter 即可完成所有映射。
    physical_to_logical.scatter_reduce_(
        -1, valid_block_table.to(torch.int64), valid_logical_indices, reduce="amax"
    )
    # NB - Seems like block 0 is always empty so we reset it manually
    physical_to_logical[:, 0] = -1
    return physical_to_logical


def unique_static_unsorted(
    x: torch.Tensor,
    *,
    M: int,  # maximum positive value (0 is “skip me”)
    dim: int = -1,  # axis along which to deduplicate
    ignored_val: int = 0,  # value to ignore
    pad_val: int = -1,  # sentinel for unused slots
) -> torch.Tensor:
    """
    - Keeps the first occurrence of each non-zero value while preserving order,
      then left-packs those uniques and fills the rest with `pad_val`.
    - Returns (packed, keep_mask) with the *same shape* as `x`.
    - Requires that all values be in the range [0, M]
    - Skips ignored_val

    Works on CPU or GPU, no Python loops, O(B·N) time / O(B·M) memory.

    Example:
    x =[3, 1, 0, 1, 2], M=3, ignored_val=0 => [3, 1, 2, -1, -1]
    """
    if not (-1 <= pad_val <= M):
        raise ValueError("`pad_val` must lie in [-1, M]")

    # ── move `dim` to the end so we can treat tensor as [B, N] ──────────
    dim = dim % x.ndim
    x_perm = x.movedim(dim, -1)  # shape [..., N]
    B, N = x_perm.numel() // x_perm.shape[-1], x_perm.shape[-1]
    x_flat = x_perm.reshape(B, N)  # [B, N]

    device = x.device
    idx = torch.arange(N, device=device).expand(B, N)  # per-row indices

    # ── build first-occurrence table for every v ∈ [0, M] ───────────────
    first_idx = torch.full((B, M + 1), N, device=device)  # “∞”
    # scatter_reduce_: first_idx[b, v] = min(first_idx[b, v], i) for each i
    first_idx.scatter_reduce_(1, x_flat, idx, reduce="amin")

    # ── keep mask: first occurrence *and* value ≠ 0 ─────────────────────
    keep = (x_flat != ignored_val) & (idx == first_idx.gather(1, x_flat))  # [B, N]

    # ── left-pack uniques into a fresh tensor ───────────────────────────
    # Route non-kept entries to a garbage slot at column N so we can do a
    # single scatter rather than using torch.nonzero (which would force a
    # GPU->CPU sync to enumerate kept positions).
    dest_pos = torch.cumsum(keep.to(torch.long), dim=1) - 1  # where to go
    dest_pos = torch.where(keep, dest_pos, N)
    packed_extended = torch.full((B, N + 1), pad_val, device=device, dtype=x_flat.dtype)
    packed_flat = packed_extended.scatter_(1, dest_pos, x_flat)[:, :N]

    # ── restore original layout ─────────────────────────────────────────
    packed = packed_flat.reshape(x_perm.shape).movedim(-1, dim)
    return packed


def causal_mask_mod(
    b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
):
    return q_idx >= kv_idx


def bidirectional_mask_mod(
    b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
):
    return q_idx >= 0


# 中文注释：块稀疏性提示的函数类型别名。
# 该函数接收 (q_block_idx, kv_block_idx, block_size)，返回布尔张量，
# 指示哪些 (query block, KV block) 对可能包含非遮罩元素。
# 用于在 FlexAttention 内核调用前剪枝完全被遮罩的 KV 块。
_block_sparsity_hint_signature = Callable[
    [torch.Tensor, torch.Tensor, int], torch.Tensor
]


# 中文注释：块稀疏性提示，用于自定义 mask_mod 中稀疏注意力的 KV 块剪枝。
# 当用户自定义的 mask_mod 只关注特定范围的 KV 时，通过 hint_fn 告诉
# FlexAttention 内核哪些 KV 块可以跳过，避免不必要的计算。
class BlockSparsityHint(NamedTuple):
    """This prunes KV blocks from the BlockMask before the flex_attention kernel
    is invoked, so that blocks that are fully masked never get loaded.
    Use this with custom mask_mods that are sparse to avoid
    the kernel iterating over all KV blocks unnecessarily.

    Attributes:
        hint_fn: (q_block_idx [num_tokens, 1], kv_block_idx [1, num_kv_blocks],
            block_size int) -> bool Tensor [num_tokens, num_kv_blocks].
            Returns True for block pairs that may contain non-masked elements.
    """

    hint_fn: _block_sparsity_hint_signature


def copy_to_persistent(dst, src):
    """中文注释：将 src 数据拷贝到预分配的持久化缓冲区 dst 中。

    FlexAttention 使用 torch.compile 进行编译优化，为了避免每次前向传播时
    分配新的张量导致编译器重新编译（recompilation），预先分配最大尺寸的
    持久化缓冲区（persistent buffer），然后在每次前向传播时只写入有效数据。

    这样做的好处：
    1. 避免动态形状导致 torch.compile 频繁触发重编译。
    2. 减少每次前向传播时的内存分配开销。
    3. 持久化缓冲区在 CUDA graph 捕获时也不会改变地址。

    注意：dst 的形状始终 >= src 的形状，只拷贝 src 对应的切片。
    """
    sliced = dst[tuple(slice(0, s) for s in src.shape)]
    sliced.copy_(src)
    return sliced


# 中文注释：FlexAttention 的注意力元数据类，封装了每次前向传播所需的全部信息。
# 包括：
# 1. 基本信息：token 数量、序列长度、query 起始位置、block table、slot mapping
# 2. 分页信息：物理->逻辑映射（physical_to_logical）、block 数量、block 大小
# 3. Flex 专用：BlockMask（块级稀疏掩码）、mask_mod（掩码函数）、score_mod（分数修改函数）
# 4. 多模态支持：mm_prefix_range（图文前缀 LM 的文档范围）
#
# 生命周期：由 FlexAttentionMetadataBuilder.build() 创建，每步调度后更新。
# 传给 FlexAttentionImpl.forward() 后用于构建 BlockMask 并执行注意力计算。
@dataclass
class FlexAttentionMetadata:
    causal: bool
    num_actual_tokens: int  # Number of tokens excluding padding.
    max_query_len: int
    query_start_loc: torch.Tensor
    # CPU-resident copy of query_start_loc used to derive doc_ids without a
    # GPU->CPU sync from repeat_interleave's data-dependent output size.
    query_start_loc_cpu: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor

    use_cascade: bool
    common_prefix_len: int
    cu_prefix_query_lens: torch.Tensor | None
    prefix_kv_lens: torch.Tensor | None
    suffix_kv_lens: torch.Tensor | None

    # 中文注释：分页 KV cache 相关的块信息。
    # total_cache_tokens: GPU 上所有物理 KV block 的总 token 容量
    # physical_to_logical: 物理 block -> 逻辑 block 的反向映射，形状 [num_reqs, total_blocks]
    # decode_offset: 每个请求已计算的 token 数（来自 prefix cache 命中），用于计算逻辑 query 索引
    # persistent_*: 预分配的持久化缓冲区，避免 torch.compile 重编译
    # Block info
    total_cache_tokens: int
    block_size: int
    max_possible_sequence_length: int
    num_reqs: int
    physical_to_logical: torch.Tensor
    decode_offset: torch.Tensor
    num_blocks_per_seq: torch.Tensor
    persistent_kv_indices: torch.Tensor
    persistent_kv_num_blocks: torch.Tensor
    persistent_doc_ids: torch.Tensor

    # For logging.
    num_input_tokens: int = 0  # Number of tokens including padding.

    # 中文注释：FlexAttention 专用的元数据字段。
    # num_blocks: 物理 KV 块的总数（total_cache_tokens // block_size）
    # block_mask: FlexAttention 的块级稀疏掩码，控制哪些 KV 块需要参与计算
    # score_mod: 用户自定义的注意力分数修改函数（如温度缩放等）
    # logical_mask_mod: 逻辑索引层面的掩码函数（因果遮罩或双向遮罩）
    # uses_paged_kv: 是否使用分页 KV cache（decoder 使用，encoder-only 不使用）
    # doc_ids: 每个 query token 对应的请求 ID，用于请求级遮罩
    # direct_build: 是否使用高效的直接构建路径（BlockMask.from_kv_blocks）
    # transformed_score_mod: 经过物理->逻辑索引转换后的 score_mod 包装函数
    # sliding_window: 滑动窗口大小，None 表示不使用滑动窗口
    # mm_prefix_range: 多模态前缀 LM 的文档范围，用于图文混合注意力
    # block_sparsity_hint: 自定义的块稀疏性提示，用于剪枝完全被遮罩的 KV 块
    # Flex Metadata
    num_blocks = 0
    block_mask: BlockMask | None = None
    score_mod: _score_mod_signature | None = None
    logical_mask_mod: _mask_mod_signature = causal_mask_mod
    uses_paged_kv: bool = True
    doc_ids: torch.Tensor | None = None
    direct_build: bool = True
    q_block_size: int = 16
    kv_block_size: int = 16
    transformed_score_mod: _score_mod_signature | None = None
    sliding_window: int | None = None
    mm_prefix_range: dict[int, list[tuple[int, int]]] | None = None
    block_sparsity_hint: BlockSparsityHint | None = None

    # 中文注释：缓存的逻辑 block 索引序列，范围 [0, max_num_blocks)。
    # 用于构建 BlockMask 时的块级索引计算，避免每次重新创建。
    @cached_property
    def logical_block_ids(self):
        return torch.arange(
            cdiv(self.max_seq_len, self.block_size),
            device=self.block_table.device,
            dtype=torch.long,
        )

    def _convert_physical_to_logical(
        self,
        request_lookup: torch.Tensor,
        q_idx: torch.Tensor,
        physical_kv_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert physical indices to logical indices for both query and kv.

        NB is_within_lower_bound: do sequences start on block_boundaries?

        Returns:
            tuple of (is_valid, logical_q_idx, logical_kv_idx)
        """
        # 中文注释：核心的物理->逻辑索引转换函数。
        #
        # 背景：vLLM 的 PagedAttention 使用分页 KV cache，KV 按物理 block 存储，
        # 但 FlexAttention 的 mask_mod/score_mod 需要使用逻辑索引来判断遮罩关系。
        # 因此需要将物理索引（内核实际访问的位置）转换为逻辑索引（序列中的位置）。
        #
        # 转换步骤：
        # 1. 通过 request_lookup (doc_ids) 确定每个 query token 属于哪个请求
        # 2. 将物理 KV 索引分解为 (physical_kv_block, physical_kv_offset)
        # 3. 通过 physical_to_logical 映射表查到逻辑 block 索引
        # 4. 逻辑 KV 索引 = logical_block_idx * block_size + physical_kv_offset
        # 5. 通过 validity 检查排除无效位置（未分配的 block、超出序列长度的位置）

        # Map query indices to corresponding request indices
        q_req = request_lookup[q_idx]

        # Convert physical KV indices to logical indices
        physical_kv_block = physical_kv_idx // self.block_size
        physical_kv_offset = physical_kv_idx % self.block_size
        logical_block_idx = self.physical_to_logical[q_req, physical_kv_block]
        logical_kv_idx = logical_block_idx * self.block_size + physical_kv_offset

        # Determine valid kv indices
        # 中文注释：有效性检查——block 必须已分配（>= 0），且逻辑索引必须在序列范围内。
        live_block = logical_block_idx >= 0
        within_upper_bound = logical_kv_idx < self.seq_lens[q_req]
        within_lower_bound = logical_kv_idx >= 0
        is_valid = live_block & within_upper_bound & within_lower_bound

        # Convert physical query indices to logical indices
        # 中文注释：物理 query 索引 -> 逻辑 query 索引。
        # local_q_idx 是请求内的相对位置，加上 decode_offset（prefix cache 命中的 token 数）
        # 得到绝对逻辑位置。这对 prefill 中跳过已计算 token 很关键。
        local_q_idx = q_idx - self.query_start_loc[q_req]
        logical_q_idx = local_q_idx + self.decode_offset[q_req]

        return is_valid, logical_q_idx, logical_kv_idx

    def get_paged_mask_mod(self) -> _mask_mod_signature:
        """Creates the mask_mod function for FlexAttention.

        This function creates the combined mask mod function that handles:
            1. The paged attention block mapping
            2. The mapping from packed query sequences to logical query entries

        It also by defaults adds the decoding offset to the query indices.
        With this info we create the "logical" indices that are passed to
        mask_mod functions. This allows mask mod functions to be agnostic to
        layout of the query and key/value tensors.
        """
        assert self.doc_ids is not None

        def final_mask_mod(
            b: torch.Tensor,
            h: torch.Tensor,
            q_idx: torch.Tensor,
            physical_kv_idx: torch.Tensor,
        ) -> torch.Tensor:
            (is_valid, logical_q_idx, logical_kv_idx) = (
                self._convert_physical_to_logical(self.doc_ids, q_idx, physical_kv_idx)
            )
            return is_valid & self.logical_mask_mod(b, h, logical_q_idx, logical_kv_idx)

        return final_mask_mod

    def get_bidirectional_mask_mod(self) -> _mask_mod_signature:
        """Creates the encoder mask_mod function for FlexAttention.

        Since the encoder bidirectional attention doesn't run with
        KV cache, this function creates a mask based on the
        packed query sequences.
        """
        # Create a lookup mapping from query indices -> request number
        request_lookup = _offsets_to_doc_ids_tensor(
            self.query_start_loc_cpu, self.query_start_loc.device
        )

        def final_mask_mod(
            b: torch.Tensor,
            h: torch.Tensor,
            q_idx: torch.Tensor,
            kv_idx: torch.Tensor,
        ) -> torch.Tensor:
            return request_lookup[q_idx] == request_lookup[kv_idx]

        return final_mask_mod

    def get_sliding_window_mask_mod(self) -> _mask_mod_signature:
        """Creates the sliding window mask_mod function for FlexAttention.

        Note that the sliding window mask here is bidirectional, we need
        to mask it with the bidirectional/causal mask for encoder/decoder.
        """

        if self.sliding_window is None:
            raise ValueError("sliding_window must be set for sliding window attention")

        def sliding_window_mask_mod(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ):
            return torch.abs(q_idx - kv_idx) < self.sliding_window

        def final_mask_mod(
            b: torch.Tensor,
            h: torch.Tensor,
            q_idx: torch.Tensor,
            physical_kv_idx: torch.Tensor,
        ) -> torch.Tensor:
            (is_valid, logical_q_idx, logical_kv_idx) = (
                self._convert_physical_to_logical(self.doc_ids, q_idx, physical_kv_idx)
            )
            return torch.where(
                is_valid,
                sliding_window_mask_mod(b, h, logical_q_idx, logical_kv_idx),
                False,
            )

        return final_mask_mod if self.uses_paged_kv else sliding_window_mask_mod

    def get_prefix_lm_mask_mod(self) -> _mask_mod_signature:
        """Creates the prefix LM mask_mod function for FlexAttention."""

        assert self.doc_ids is not None
        request_lookup = self.doc_ids

        def prefix_lm_mask_mod(
            b: torch.Tensor,
            h: torch.Tensor,
            cu_q_idx: torch.Tensor,
            q_idx: torch.Tensor,
            kv_idx: torch.Tensor,
        ):
            mask = torch.zeros_like(q_idx, dtype=torch.bool)
            for req, doc_range_lst in (self.mm_prefix_range or {}).items():
                req_mask = request_lookup[cu_q_idx] == req
                for start, end in doc_range_lst:
                    doc_mask_q = (q_idx >= start) & (q_idx <= end)
                    doc_mask_kv = (kv_idx >= start) & (kv_idx <= end)
                    mask = mask | (req_mask & doc_mask_q & doc_mask_kv)
            return mask

        def final_mask_mod(
            b: torch.Tensor,
            h: torch.Tensor,
            q_idx: torch.Tensor,
            physical_kv_idx: torch.Tensor,
        ) -> torch.Tensor:
            (is_valid, logical_q_idx, logical_kv_idx) = (
                self._convert_physical_to_logical(self.doc_ids, q_idx, physical_kv_idx)
            )
            return torch.where(
                is_valid,
                prefix_lm_mask_mod(b, h, q_idx, logical_q_idx, logical_kv_idx),
                False,
            )

        return final_mask_mod

    def get_mask_mod(self):
        """中文注释：组合多层掩码，生成最终的 mask_mod 函数。

        构建过程分两个阶段：
        1. 基础掩码：根据是否使用分页 KV，选择因果遮罩（decoder）或双向遮罩（encoder-only）。
           分页 KV 路径还会添加物理->逻辑索引转换。
        2. 组合掩码：在基础掩码之上叠加额外的掩码约束：
           - 滑动窗口（sliding_window）：通过 and_masks 与基础掩码取交集
           - 前缀 LM（mm_prefix_range）：通过 or_masks 与基础掩码取并集，
             使得图文混合输入中的图像 token 可以看到彼此（全注意力区域）

        返回的 mask_mod 接受 (b, h, q_idx, physical_kv_idx) 参数，
        内部自动完成物理->逻辑索引转换，使上层用户只需关注逻辑索引。
        """
        # Stage-1: initialize the base mask_mod
        # (causal mask for decoder or bidirectional mask for encoder)
        if self.uses_paged_kv:
            mask_mod = self.get_paged_mask_mod()
        else:
            mask_mod = self.get_bidirectional_mask_mod()
        # stage-2: add external mask_mod for special attention during
        # forwarding runtime to create the combined mask_mod.
        if self.sliding_window is not None:
            # Add sliding window mask for sliding window attention
            sliding_window_mask_mod = self.get_sliding_window_mask_mod()
            mask_mod = and_masks(mask_mod, sliding_window_mask_mod)
        if self.mm_prefix_range:
            # Add prefix LM mask for vision-language prefix LM attention
            prefix_lm_mask_mod = self.get_prefix_lm_mask_mod()
            mask_mod = or_masks(mask_mod, prefix_lm_mask_mod)
        return mask_mod

    def get_transformed_score_mod(self) -> _score_mod_signature | None:
        """Creates the transformed score_mod function for FlexAttention.

        This function wraps the user's score_mod to handle physical-to-logical
        index conversion, similar to how get_mask_mod works for mask functions.
        """
        if self.score_mod is None:
            return None

        # Create a lookup mapping from query indices -> request number
        request_lookup = _offsets_to_doc_ids_tensor(
            self.query_start_loc_cpu, self.query_start_loc.device
        )
        user_score_mod = self.score_mod

        def transformed_score_mod(
            score: torch.Tensor,
            b: torch.Tensor,
            h: torch.Tensor,
            q_idx: torch.Tensor,
            physical_kv_idx: torch.Tensor,
        ) -> torch.Tensor:
            (is_valid, logical_q_idx, logical_kv_idx) = (
                self._convert_physical_to_logical(
                    request_lookup, q_idx, physical_kv_idx
                )
            )

            return torch.where(
                is_valid,
                user_score_mod(
                    score, b, h, logical_q_idx, logical_kv_idx, physical_q=q_idx
                ),
                -float("inf"),
            )

        return transformed_score_mod

    def _build_block_mask_direct(self) -> BlockMask:
        """Direct block mask construction for paged KV cache attention.

        This method constructs the block mask directly using
        BlockMask.from_kv_blocks which is much more efficient than the
        generic create_block_mask approach.

        The direct path works as follows:
        1. For each query token, fetch blocks from block_table using max_seq_len
           and exclude out of sliding window blocks if needed.
           (this fetches more blocks than needed for shorter sequences)
        2. Group query tokens into chunks of q_block_size
        3. For each group, deduplicate the blocks using unique_static_unsorted
        4. Create BlockMask using the deduplicated block indices

        Over-estimation occurs when a group of q_block_size tokens contains
        multiple sequence IDs (doc_ids). In this case, we fetch ALL blocks for
        each sequence represented in the group, even though individual query
        tokens may only need a subset of those blocks based on causal masking
        and their position.

        """
        # 中文注释：高效的 BlockMask 直接构建路径。
        #
        # 背景：FlexAttention 使用 BlockMask 来表示块级稀疏性——哪些 (Q块, KV块) 对
        # 可能包含非遮罩元素。通用路径（create_block_mask）需要调用 mask_mod 逐块检查，
        # 而直接路径通过分析 block_table 和 sliding_window 直接构造，效率更高。
        #
        # 算法步骤：
        # 1. page_to_block_ratio: 检查 KV cache block 与 FlexAttention block 的大小关系
        # 2. used_pages: 从 block_table 中获取每个 token 对应的所有 KV 块
        # 3. sliding_window 过滤：排除滑动窗口之外的 KV 块
        # 4. custom_hint 过滤：应用用户自定义的稀疏性提示
        # 5. 按 q_block_size 分组，对每组的 KV 块去重（unique_static_unsorted）
        # 6. 使用 BlockMask.from_kv_blocks 构建最终的 BlockMask
        page_to_block_ratio = self.kv_block_size // self.block_size
        if page_to_block_ratio != 1:
            raise ValueError(
                f"FlexAttention currently requires the cache block size "
                f"({self.block_size}) to be equal to the kv_block_size "
                f"({self.kv_block_size}). Please check your model's "
                f"configuration."
            )

        used_pages = self.block_table[
            self.doc_ids, : cdiv(self.max_seq_len, self.block_size)
        ]

        custom_hint = self.block_sparsity_hint is not None

        if self.sliding_window or custom_hint:
            device = used_pages.device
            assert self.doc_ids is not None
            token_indices = torch.arange(
                self.doc_ids.shape[0], device=device, dtype=torch.long
            )
            logical_q_idx = (
                token_indices
                - self.query_start_loc[self.doc_ids]
                + self.decode_offset[self.doc_ids]
            )

            if self.sliding_window:
                assert self.sliding_window is not None
                min_kv_idx = torch.clamp(
                    logical_q_idx - (self.sliding_window - 1), min=0
                )
                min_block_idx = min_kv_idx // self.block_size
                sliding_mask = self.logical_block_ids >= min_block_idx[:, None]
                used_pages.masked_fill_(~sliding_mask, 0)
            if custom_hint:
                assert self.block_sparsity_hint is not None
                q_block_idx = logical_q_idx // self.block_size
                hint_mask = self.block_sparsity_hint.hint_fn(
                    q_block_idx[:, None],
                    self.logical_block_ids[None, :],
                    self.block_size,
                )
                used_pages.masked_fill_(~hint_mask, 0)

        used_pages_padded = pad_to_multiple(
            used_pages, multiple=self.q_block_size, dim=0
        )
        used_pages_padded = used_pages_padded.reshape(
            used_pages_padded.shape[0] // self.q_block_size, -1
        )
        used_pages_padded = used_pages_padded // page_to_block_ratio
        kv_indices = unique_static_unsorted(
            (used_pages_padded.long()), M=self.num_blocks
        ).to(torch.int32)
        kv_indices = copy_to_persistent(self.persistent_kv_indices, kv_indices)

        kv_num_blocks = (kv_indices >= 0).sum(dim=-1).to(torch.int32)
        kv_num_blocks = copy_to_persistent(self.persistent_kv_num_blocks, kv_num_blocks)

        block_mask_kwargs = {
            "seq_lengths": (self.num_actual_tokens, self.total_cache_tokens),
            "kv_num_blocks": kv_num_blocks[None, None],
            "kv_indices": kv_indices[None, None],
            "full_kv_num_blocks": None,
            "full_kv_indices": None,
            "BLOCK_SIZE": (self.q_block_size, self.kv_block_size),
            "mask_mod": self.mask_mod,
        }

        # compute_q_blocks parameter is available in PyTorch 2.9+
        if is_torch_equal_or_newer("2.9.0.dev0"):
            block_mask_kwargs["compute_q_blocks"] = False
        return BlockMask.from_kv_blocks(**block_mask_kwargs)

    def build_block_mask(self) -> BlockMask:
        """中文注释：通用的 BlockMask 构建路径（非 direct_build 时使用）。

        与 _build_block_mask_direct 不同，该方法使用 PyTorch 提供的
        create_block_mask 函数，通过调用 mask_mod 逐块检查来确定稀疏性。

        适用场景：
        - encoder-only 模型（双向遮罩，无法直接从 block_table 推导）
        - KV block 与 FlexAttention block 大小不一致时
        - 自定义复杂 mask_mod 时

        代价：比 direct_build 路径慢，因为需要实际调用 mask_mod 函数。
        """
        mask_mod = self.get_mask_mod()
        kv_len = (
            self.total_cache_tokens if self.uses_paged_kv else self.num_actual_tokens
        )
        return create_block_mask_compiled(
            mask_mod,
            None,
            None,
            self.num_actual_tokens,
            kv_len,
            device=self.block_table.device,
            BLOCK_SIZE=(self.q_block_size, self.kv_block_size),
        )

    def __post_init__(self):
        """中文注释：数据类初始化后自动调用的后处理方法。

        主要完成以下工作：
        1. 断言检查：级联注意力（cascade attention）尚未实现，需要 common_prefix_len == 0
        2. 构建 doc_ids：将 query_start_loc 转换为每个 token 的请求 ID 映射
        3. 计算 num_blocks：物理 KV 块的总数
        4. 预构建 mask_mod 和 transformed_score_mod 函数

        注意：BlockMask 的构建延迟到首次调用时执行（由 FlexAttentionImpl 触发），
        因为构建依赖于 CUDA graph 安全检查。
        """
        assert self.use_cascade is False, "Not implemented yet."
        assert self.common_prefix_len == 0, "Not implemented yet."
        assert self.cu_prefix_query_lens is None, "Not implemented yet."
        assert self.prefix_kv_lens is None, "Not implemented yet."
        assert self.suffix_kv_lens is None, "Not implemented yet."
        # Create a lookup mapping from query indices -> request number
        self.doc_ids = _offsets_to_doc_ids_tensor(
            self.query_start_loc_cpu, self.query_start_loc.device
        )
        self.doc_ids = copy_to_persistent(self.persistent_doc_ids, self.doc_ids)
        self.num_blocks = self.total_cache_tokens // self.block_size

        self.mask_mod = self.get_mask_mod()
        self.transformed_score_mod = self.get_transformed_score_mod()


# 中文注释：FlexAttention 元数据构建器，负责每步调度后构建 FlexAttentionMetadata。
#
# 工作流程（build 方法）：
# 1. 从 CommonAttentionMetadata 获取基础信息（请求数、token 数、block table 等）
# 2. 调用 physical_to_logical_mapping() 构建物理->逻辑反向映射
# 3. 计算每个请求的 decode_offset（prefix cache 命中的 token 数）
# 4. 确定掩码类型（因果/双向）并组装 FlexAttentionMetadata
# 5. 预构建 BlockMask，确保 CUDA graph 捕获前已准备就绪
#
# 关键设计：
# - 使用持久化缓冲区（persistent buffers）避免 torch.compile 重编译
# - _cudagraph_support = ALWAYS，表示该后端始终支持 CUDA graph
class FlexAttentionMetadataBuilder(AttentionMetadataBuilder[FlexAttentionMetadata]):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.ALWAYS

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

        # 中文注释：初始化元数据构建器，预分配持久化缓冲区。
        # 持久化缓冲区的作用：避免每步调度时分配新张量，防止 torch.compile 重编译。
        # 缓冲区大小按最大可能值预分配，实际使用时只写入有效部分（通过 copy_to_persistent）。
        self.num_heads_q = self.model_config.get_num_attention_heads(
            self.parallel_config
        )
        self.num_heads_kv = self.model_config.get_num_kv_heads(self.parallel_config)
        self.headdim = self.model_config.get_head_size()
        self.block_size = kv_cache_spec.block_size
        self.kv_cache_spec = kv_cache_spec
        # 中文注释：PyTorch 2.9+ 支持小块（16）的 BlockMask，启用 direct_build 路径。
        supports_small_blocks = is_torch_equal_or_newer("2.9.0.dev0")
        self.direct_build: bool = supports_small_blocks

        self.q_block_size, self.kv_block_size = self._get_block_sizes(
            vllm_config.attention_config,
            supports_small_blocks,
            self.block_size,
        )

        # 中文注释：如果 KV block 大小与 cache block 大小不一致，回退到通用构建路径。
        # direct_build 路径要求两者相等，否则无法直接从 block_table 推导 BlockMask。
        if self.direct_build and self.kv_block_size != self.block_size:
            self.direct_build = False

        self.max_model_len = self.model_config.max_model_len
        max_num_seqs = vllm_config.scheduler_config.max_num_seqs
        max_num_batched_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.max_num_query_groups = cdiv(max_num_batched_tokens, self.q_block_size)
        max_num_pages_per_seq = cdiv(self.max_model_len, self.block_size)
        self.max_num_kv_indices = self.q_block_size * max_num_pages_per_seq
        self.persistent_kv_num_blocks = torch.empty(
            self.max_num_query_groups, dtype=torch.int32, device=device
        )
        self.persistent_offset_tensor = torch.empty(
            max_num_seqs, dtype=torch.int32, device=device
        )
        self.persistent_doc_ids = torch.empty(
            max_num_batched_tokens, dtype=torch.int32, device=device
        )

        # initialize later when we can access block_table
        self.persistent_physical_to_logical = None
        self.persistent_kv_indices = None

    @staticmethod
    def _get_block_sizes(
        attn_cfg,
        supports_small_blocks: bool,
        cache_block_size: int,
    ) -> tuple[int, int]:
        q_block_size = 16 if supports_small_blocks else 128
        kv_block_size = cache_block_size if supports_small_blocks else 128

        q_block_size = attn_cfg.flex_attn_q_block_size or q_block_size
        if (q_block_size & (q_block_size - 1)) != 0 or (
            attn_cfg.flex_attn_block_m is not None
            and q_block_size % attn_cfg.flex_attn_block_m != 0
        ):
            raise ValueError(
                f"flex_attn_q_block_size must be a power of 2 "
                f"and divisible by flex_attn_block_m, got "
                f"{q_block_size}, {attn_cfg.flex_attn_block_m}"
            )

        kv_block_size = attn_cfg.flex_attn_kv_block_size or kv_block_size
        if (kv_block_size & (kv_block_size - 1)) != 0 or (
            attn_cfg.flex_attn_block_n is not None
            and kv_block_size % attn_cfg.flex_attn_block_n != 0
        ):
            raise ValueError(
                f"flex_attn_kv_block_size must be a power of 2 "
                f"and divisible by flex_attn_block_n, got "
                f"{kv_block_size}, {attn_cfg.flex_attn_block_n}"
            )

        return q_block_size, kv_block_size

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> FlexAttentionMetadata:
        """中文注释：为 CUDA graph 捕获构建元数据。

        CUDA graph 捕获时要求所有张量形状固定，因此使用实际的 max_seq_len
        （而非 max_model_len），避免 torch.compile 因形状变化触发重编译。

        该方法在 CUDA graph 初始化阶段调用，构建的元数据用于后续 replay 时的模板。
        """
        # Use actual max_seq_len (not max_model_len) to avoid torch.compile
        # recompilation during CUDA graph capture.
        assert common_attn_metadata.seq_lens_cpu_upper_bound is not None
        common_attn_metadata.max_seq_len = int(
            common_attn_metadata.seq_lens_cpu_upper_bound.max().item()
        )
        return self.build(
            common_prefix_len=0, common_attn_metadata=common_attn_metadata
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> FlexAttentionMetadata:
        """中文注释：构建 FlexAttentionMetadata 的核心方法。

        该方法在每步调度后被调用，将 CommonAttentionMetadata（通用注意力元数据）
        转换为 FlexAttention 专用的元数据。

        主要步骤：
        1. 提取基本参数：请求数、token 数、序列长度、block table、slot mapping
        2. 构建物理->逻辑反向映射（physical_to_logical_mapping）
           这是 FlexAttention 特有的需求，因为 mask_mod 需要逻辑索引
        3. 计算 decode_offset（每个请求已计算的 token 数，来自 prefix cache）
        4. 确定逻辑掩码类型：非因果用双向遮罩，否则用因果遮罩
        5. 组装 FlexAttentionMetadata 对象
        6. 预构建 BlockMask（在 CUDA graph 捕获前完成，避免非图安全操作）

        Args:
            common_prefix_len: 公共前缀长度（级联注意力用，当前未实现）
            common_attn_metadata: 通用注意力元数据，包含所有请求的信息
            fast_build: 是否快速构建（预留参数，当前未使用）
        """
        num_reqs = common_attn_metadata.num_reqs
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        max_query_len = common_attn_metadata.max_query_len

        max_seq_len = common_attn_metadata.max_seq_len
        query_start_loc = common_attn_metadata.query_start_loc
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        seq_lens = common_attn_metadata.seq_lens
        block_table_tensor = common_attn_metadata.block_table_tensor
        slot_mapping = common_attn_metadata.slot_mapping
        num_blocks_per_seq = cdiv(seq_lens, self.block_size)

        use_cascade = common_prefix_len > 0
        cu_prefix_query_lens = None
        prefix_kv_lens = None
        suffix_kv_lens = None
        if use_cascade:
            raise NotImplementedError(
                "Cascade prefix attention is not yet implemented "
                "for FlexAttention backend"
            )

        block_size = self.kv_cache_spec.block_size
        max_possible_seq_len = self.model_config.max_model_len
        num_gpu_blocks = self.cache_config.num_gpu_blocks

        assert num_gpu_blocks is not None, (
            "FlexAttention requires num_gpu_blocks to be set"
        )
        total_cache_tokens = num_gpu_blocks * block_size

        inverse_block_table = physical_to_logical_mapping(
            block_table_tensor, seq_lens, block_size, num_gpu_blocks
        )
        if self.persistent_physical_to_logical is None:
            max_num_seqs = self.vllm_config.scheduler_config.max_num_seqs
            self.persistent_physical_to_logical = torch.empty(
                max_num_seqs,
                num_gpu_blocks,
                dtype=torch.long,
                device=self.device,
            )

        if self.persistent_kv_indices is None:
            self.persistent_kv_indices = torch.empty(
                self.max_num_query_groups,
                self.max_num_kv_indices,
                dtype=torch.int32,
                device=self.device,
            )

        inverse_block_table = copy_to_persistent(
            self.persistent_physical_to_logical, inverse_block_table
        )

        offset_tensor = common_attn_metadata.compute_num_computed_tokens()
        offset_tensor = copy_to_persistent(self.persistent_offset_tensor, offset_tensor)

        uses_paged_kv = not isinstance(self.kv_cache_spec, EncoderOnlyAttentionSpec)
        logical_mask_mod = (
            bidirectional_mask_mod
            if uses_paged_kv and not common_attn_metadata.causal
            else causal_mask_mod
        )

        out = FlexAttentionMetadata(
            causal=common_attn_metadata.causal,
            logical_mask_mod=logical_mask_mod,
            num_actual_tokens=num_actual_tokens,
            max_query_len=max_query_len,
            query_start_loc=query_start_loc,
            query_start_loc_cpu=query_start_loc_cpu,
            max_seq_len=max_seq_len,
            seq_lens=seq_lens,
            block_table=block_table_tensor,
            slot_mapping=slot_mapping,
            use_cascade=use_cascade,
            common_prefix_len=common_prefix_len,
            cu_prefix_query_lens=cu_prefix_query_lens,
            prefix_kv_lens=prefix_kv_lens,
            suffix_kv_lens=suffix_kv_lens,
            block_size=block_size,
            max_possible_sequence_length=max_possible_seq_len,
            num_reqs=num_reqs,
            physical_to_logical=inverse_block_table,
            total_cache_tokens=total_cache_tokens,
            decode_offset=offset_tensor,
            num_blocks_per_seq=num_blocks_per_seq,
            uses_paged_kv=uses_paged_kv,
            # FIXME(Isotr0py): direct build has issue to build bidirectional
            # attention block mask for encoder-only models, disable it temporarily.
            # see: https://github.com/vllm-project/vllm/pull/27329#issuecomment-3431484053
            direct_build=self.direct_build and uses_paged_kv,
            q_block_size=self.q_block_size,
            kv_block_size=self.kv_block_size,
            persistent_kv_indices=self.persistent_kv_indices,
            persistent_kv_num_blocks=self.persistent_kv_num_blocks,
            persistent_doc_ids=self.persistent_doc_ids,
        )

        # Pre-build block_mask so it is ready before CUDA graph capture.
        # Without this, the lazy build in forward() would run non-graph-safe
        # ops (e.g. torch.nonzero) inside capture.
        if out.block_mask is None:
            if out.direct_build:
                out.block_mask = out._build_block_mask_direct()
            else:
                out.block_mask = out.build_block_mask()

        return out

    def use_cascade_attention(self, *args, **kwargs) -> bool:
        return False


# 中文注释：FlexAttention 的注意力实现类，执行实际的前向计算。
#
# 与 Model Runner 的交互：
# 1. Model Runner 每步调度后调用 MetadataBuilder.build() 构建元数据
# 2. 在 forward 中，FlexAttentionImpl 接收 query/key/value/kv_cache 和元数据
# 3. 如果 KV cache 更新不在 forward 中（forward_includes_kv_cache_update=False），
#    Model Runner 会先调用 do_kv_cache_update 写入 KV，再调用 forward 计算注意力
# 4. forward 中调用 PyTorch 的 flex_attention 算子执行计算
#
# 关键特性：
# - 支持 decoder（因果遮罩 + 分页 KV）和 encoder-only（双向遮罩 + 无 KV cache）
# - 支持 GQA（Grouped Query Attention），通过 num_queries_per_kv 控制
# - 支持 CUDA graph 捕获（与 MetadataBuilder 的 ALWAYS 支持配合）
# - 通过 block_m/block_n 参数控制内核块大小，支持批次不变性（VLLM_BATCH_INVARIANT）
class FlexAttentionImpl(AttentionImpl):
    sliding_window: int | None
    alibi_slopes: torch.Tensor | None
    logits_soft_cap: float | None
    mm_prefix_range: dict[int, list[tuple[int, int]]] | None = None
    logical_mask_mod: _mask_mod_signature | None = None
    block_sparsity_hint: BlockSparsityHint | None = None

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
        block_m: int | None = None,
        block_n: int | None = None,
        **kwargs,
    ) -> None:
        """中文注释：初始化 FlexAttention 实现类。

        该类在模型初始化时由 AttentionBackend.get_impl_cls() 创建，每个注意力层
        持有一个实例。配置参数来自模型配置和 VllmConfig。

        关键参数：
        - num_heads / num_kv_heads: 用于 GQA 支持，num_queries_per_kv = num_heads / num_kv_heads
        - sliding_window: 滑动窗口大小，非 None 时启用滑动窗口注意力
        - block_m / block_n: 内核块大小，用于控制批次不变性（VLLM_BATCH_INVARIANT）
        - attn_type: DECODER（使用分页 KV cache）或 ENCODER_ONLY（无 KV cache）
        """
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.attn_type = attn_type

        if attn_type not in (AttentionType.ENCODER_ONLY, AttentionType.DECODER):
            raise NotImplementedError(
                f"FlexAttention does not support {attn_type} attention"
            )

        if alibi_slopes is not None:
            raise NotImplementedError(
                "FlexAttention does not support alibi slopes yet."
            )
        else:
            self.alibi_slopes = None

        self.sliding_window = sliding_window

        self.kv_cache_dtype = kv_cache_dtype
        self.logits_soft_cap = logits_soft_cap
        if self.logits_soft_cap is not None:
            raise NotImplementedError(
                "FlexAttention does not support logits soft cap yet."
            )

        assert self.num_heads % self.num_kv_heads == 0
        # 中文注释：GQA (Grouped Query Attention) 的分组比。
        # 例如 num_heads=32, num_kv_heads=8 时，num_queries_per_kv=4，
        # 表示每 4 个 query head 共享一组 KV head。
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        if kv_sharing_target_layer_name is not None:
            raise NotImplementedError("FlexAttention does not support kv sharing yet.")

        if is_quantized_kv_cache(self.kv_cache_dtype):
            raise NotImplementedError(
                "FlexAttention does not support quantized kv-cache. Yet"
            )

        # 中文注释：批次不变性模式下，固定内核块大小为 16，
        # 确保不同批次大小下产生相同的计算结果（数值一致性）。
        self.block_m = 16 if envs.VLLM_BATCH_INVARIANT else None
        self.block_n = 16 if envs.VLLM_BATCH_INVARIANT else None

        if block_m is not None:
            self.block_m = block_m
        if block_n is not None:
            self.block_n = block_n

    @staticmethod
    def view_as_4d(tensor: torch.Tensor) -> torch.Tensor:
        """View a 3d tensor as 4D."""
        if tensor.ndim == 4:
            return tensor
        assert tensor.ndim == 3
        return tensor[None, :, :, :]

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        """中文注释：将当前 step 的 key/value 写入分页 KV cache。

        由于 FlexAttentionBackend.forward_includes_kv_cache_update = False，
        该方法由 Model Runner 在 forward 之前单独调用，与 forward 解耦。

        执行流程：
        1. encoder-only 模式不使用 KV cache，直接返回
        2. 将 kv_cache 按 K/V 维度拆分（unbind(1)）
        3. 调用 C++ 扩展 reshape_and_cache_flash，将 key/value 按 slot_mapping
           写入分页 KV cache 的对应物理位置

        Args:
            layer: 注意力层模块（包含 _k_scale, _v_scale 用于量化缩放）
            key: 当前 step 的 key 张量，形状 [num_tokens, num_kv_heads, head_size]
            value: 当前 step 的 value 张量，形状 [num_tokens, num_kv_heads, head_size]
            kv_cache: 分页 KV cache 张量，形状 [num_blocks, 2, block_size, num_kv_heads, head_size]
            slot_mapping: 逻辑 token 到物理 slot 的映射，指示每个 token 写入 cache 的位置
        """
        if self.attn_type == AttentionType.ENCODER_ONLY:
            return

        key_cache, value_cache = kv_cache.unbind(1)
        torch.ops._C_cache_ops.reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
        )

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlexAttentionMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with FLexAttention.

        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache: shape =
                [num_blocks, 2, block_size, num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not yet supported for FlexAttentionImpl"
            )

        enable_gqa = self.num_kv_heads != self.num_heads

        if attn_metadata is None:
            # Profiling run.
            return output.fill_(0)
            # query = self.view_as_4d(query).permute(0, 2, 1, 3)
            # return torch.empty_like(query)

        num_actual_tokens = attn_metadata.num_actual_tokens

        # 中文注释：检查是否需要重建 BlockMask。
        # BlockMask 是 FlexAttention 的核心数据结构，当以下条件变化时需要重建：
        # 1. 滑动窗口大小变化（如首次使用滑动窗口或窗口大小改变）
        # 2. 多模态前缀范围变化（如新图像 token 到达）
        # 3. 层级的逻辑掩码函数变化（如某些层使用不同遮罩策略）
        # 4. 层级的块稀疏性提示变化（如自定义稀疏模式）
        needs_rebuild_block_mask = False
        if attn_metadata.sliding_window != self.sliding_window:
            attn_metadata.sliding_window = self.sliding_window
            if attn_metadata.direct_build:
                # update mask mod in attention metadata
                attn_metadata.mask_mod = attn_metadata.get_mask_mod()
            needs_rebuild_block_mask = True

        if self.mm_prefix_range != getattr(attn_metadata, "mm_prefix_range", None):
            self.mm_prefix_range = attn_metadata.mm_prefix_range
            attn_metadata.mask_mod = attn_metadata.get_mask_mod()
            needs_rebuild_block_mask = True

        layer_mask_mod = getattr(layer, "logical_mask_mod", None)
        if (
            layer_mask_mod is not None
            and attn_metadata.logical_mask_mod is not layer_mask_mod
        ):
            attn_metadata.logical_mask_mod = layer_mask_mod
            attn_metadata.mask_mod = attn_metadata.get_mask_mod()
            needs_rebuild_block_mask = True

        layer_hint = getattr(layer, "block_sparsity_hint", None)
        if (
            layer_hint is not None
            and attn_metadata.block_sparsity_hint is not layer_hint
        ):
            attn_metadata.block_sparsity_hint = layer_hint
            needs_rebuild_block_mask = True

        # 中文注释：如果需要重建或 BlockMask 尚未创建，则重新构建。
        # direct_build 路径使用 BlockMask.from_kv_blocks（高效），
        # 非 direct 路径使用 create_block_mask（通用但较慢）。
        if needs_rebuild_block_mask or attn_metadata.block_mask is None:
            if attn_metadata.direct_build:
                attn_metadata.block_mask = attn_metadata._build_block_mask_direct()
            else:
                attn_metadata.block_mask = attn_metadata.build_block_mask()

        # 中文注释：根据注意力类型准备 QKV 张量，统一 reshape 为 4D 格式。
        # FlexAttention 要求的输入格式：[batch=1, num_heads, seq_len, head_size]
        # （batch=1 是因为 vLLM 将所有请求打包成一个大序列）
        if self.attn_type == AttentionType.ENCODER_ONLY:
            # 中文注释：encoder-only 模式：Q/K/V 都来自当前输入，不使用 KV cache。
            query, key_tensor, value_tensor = map(
                lambda x: self.view_as_4d(x).permute(0, 2, 1, 3),
                (query, key, value),
            )

            query = query[:, :, :num_actual_tokens, :]
            if (key_tensor.size(-2) > num_actual_tokens) or (
                value_tensor.size(-2) > num_actual_tokens
            ):
                # In the encoder-only model with torch.compile,
                # qkv might be padded, which might cause exception.
                # see: https://github.com/vllm-project/vllm/pull/24872#discussion_r2353252290
                key_tensor = key_tensor[:, :, :num_actual_tokens, :]
                value_tensor = value_tensor[:, :, :num_actual_tokens, :]

        else:
            # 中文注释：decoder 模式：Q 来自当前输入，K/V 来自分页 KV cache。
            # 将 kv_cache 按 K/V 拆分后展平为 (total_tokens, num_kv_heads, head_size)，
            # 使得 FlexAttention 内核可以直接按物理索引访问 KV cache 中的任意位置。
            assert self.attn_type == AttentionType.DECODER
            key_cache, value_cache = kv_cache.unbind(1)

            # Flatten (num_blocks, block_size) into a single token dim
            key_cache = key_cache.view(-1, self.num_kv_heads, self.head_size)
            value_cache = value_cache.view(-1, self.num_kv_heads, self.head_size)
            query, key_tensor, value_tensor = map(
                lambda x: self.view_as_4d(x).permute(0, 2, 1, 3),
                (query, key_cache, value_cache),
            )

            query = query[:, :, :num_actual_tokens, :]

        # Doesn't work for now -> constraint violation
        # torch._dynamo.try_mark_dynamic(query, 2)

        assert attn_metadata.block_mask is not None
        block_m, block_n = attn_metadata.block_mask.BLOCK_SIZE

        # 中文注释：获取内核选项，包括 BLOCK_M/BLOCK_N（内核块大小）、
        # FORCE_USE_FLEX_ATTENTION 等参数。根据硬件特性（共享内存大小等）自动调整。
        kernel_options = get_kernel_options(
            query, block_m, block_n, attn_metadata.direct_build
        )

        if self.block_m is not None:
            kernel_options["BLOCK_M"] = self.block_m
        if self.block_n is not None:
            kernel_options["BLOCK_N"] = self.block_n
        if envs.VLLM_BATCH_INVARIANT:
            kernel_options["IS_DIVISIBLE"] = False

        # 中文注释：调用 PyTorch 的 flex_attention 编译内核执行注意力计算。
        # flex_attention 是一个可编程的注意力算子：
        # - query/key_tensor/value_tensor: 4D QKV 张量
        # - transformed_score_mod: 经过物理->逻辑转换的分数修改函数（可选）
        # - block_mask: 块级稀疏掩码，跳过完全被遮罩的 KV 块
        # - scale: 注意力缩放因子（通常为 1/sqrt(head_size)）
        # - enable_gqa: 是否启用 Grouped Query Attention
        # - kernel_options: 内核配置参数
        out = flex_attention_compiled(
            query,
            key_tensor,
            value_tensor,
            attn_metadata.transformed_score_mod,
            attn_metadata.block_mask,
            self.scale,
            enable_gqa=enable_gqa,
            kernel_options=kernel_options,
        )

        # Flex doesn't have an out variant today, rely on epilogue fusion
        # 中文注释：将输出从 [1, num_heads, seq_len, head_size] 转换回
        # [num_tokens, num_heads, head_size] 格式，并拷贝到预分配的输出缓冲区。
        out = out.permute(0, 2, 1, 3).squeeze(0)
        output[:num_actual_tokens, :, :].copy_(out)
        return output


def get_kernel_options(
    query, block_m, block_n, use_direct_build: bool
) -> dict[str, int | bool]:
    """中文注释：获取 FlexAttention 内核的配置参数。

    该函数根据硬件特性和输入配置，选择合适的内核块大小（BLOCK_M, BLOCK_N）。

    选择策略：
    1. direct_build 路径：直接使用 BlockMask 指定的块大小，无需调整
    2. 非 direct_build 路径：
       a. 首选块大小：float32 用 32，其他用 64
       b. 确保块大小能被逻辑 block_size 整除（通过 ensure_divisible）
       c. 根据 GPU 共享内存大小调整：共享内存 < 144KB 时减半
       d. 最小块大小为 16

    Args:
        query: 查询张量，用于推断数据类型和设备信息
        block_m: 逻辑 query 块大小（来自 BlockMask）
        block_n: 逻辑 KV 块大小（来自 BlockMask）
        use_direct_build: 是否使用直接构建路径

    Returns:
        内核选项字典，包含 BLOCK_M、BLOCK_N 等参数
    """
    kernel_options: dict[str, int | bool] = {
        "FORCE_USE_FLEX_ATTENTION": True,
    }

    def ensure_divisible(candidate: int, block_size: int) -> int:
        """Pick a kernel block size that divides the logical block."""
        if block_size <= 0:
            return candidate
        candidate = min(candidate, block_size)
        if candidate <= 0:
            return block_size
        if block_size % candidate == 0:
            return candidate

        candidate = math.gcd(candidate, block_size)
        if candidate <= 1:
            return block_size
        return candidate

    if use_direct_build:
        kernel_options["BLOCK_M"] = block_m
        kernel_options["BLOCK_N"] = block_n
        return kernel_options
    else:
        preferred_block = 32 if query.dtype == torch.float32 else 64
        block_lower_bound = 16

        block_m_candidate = ensure_divisible(preferred_block, block_m)
        block_n_candidate = ensure_divisible(preferred_block, block_n)

        if torch.cuda.is_available():
            device_props = torch.cuda.get_device_properties()
            # ROCm doesn't expose shared_memory_per_block_optin attribute
            # AMD GPUs typically have 64KB LDS (Local Data Share) per workgroup
            if hasattr(device_props, "shared_memory_per_block_optin"):
                max_shared_memory = device_props.shared_memory_per_block_optin
            elif current_platform.is_rocm():
                # ROCm fallback: use 64KB
                max_shared_memory = 65536
            else:
                raise RuntimeError(
                    "Unable to determine shared memory size on this hardware."
                )

            if max_shared_memory < 144 * 1024:
                block_m_candidate = ensure_divisible(
                    max(1, block_m_candidate // 2), block_m
                )
                block_n_candidate = ensure_divisible(
                    max(1, block_n_candidate // 2), block_n
                )

        block_m_candidate = max(block_m_candidate, block_lower_bound)
        block_n_candidate = max(block_n_candidate, block_lower_bound)

        kernel_options["BLOCK_M"] = block_m_candidate
        kernel_options["BLOCK_N"] = block_n_candidate

    return kernel_options
