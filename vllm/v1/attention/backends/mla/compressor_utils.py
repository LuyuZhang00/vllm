# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
KV 缓存压缩工具模块。

本模块提供用于 DeepSeekV4 模型的 KV 缓存压缩功能。DeepSeekV4 使用多种压缩比率
（compress_ratio），将多个连续 token 的 KV 缓存压缩存储为一个条目。

压缩机制说明：
- compress_ratio=1: 不压缩（标准 MLA）
- compress_ratio=4: 每 4 个 token 压缩为 1 个（C4A 模式）
- compress_ratio=128: 每 128 个 token 压缩为 1 个（C128A 模式）

压缩 slot mapping 的计算规则：
- 对于位置 pos 的 token，只有当 (pos + 1) % compress_ratio == 0 时才产生有效 slot
- 压缩后的位置 = pos // compress_ratio
- 无效位置的 slot mapping 设为 -1（PAD_ID）

这样设计的好处是：只有满足压缩对齐条件的 token 才会写入 KV 缓存，
从而节省内存空间。注意力计算时，通过 slot mapping 将原始位置映射到压缩后的缓存位置。
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _compressed_slot_mapping_kernel(
    # [num_tokens] - 输出的压缩 slot mapping
    slot_mapping_ptr,
    # [num_reqs + 1] - 每个请求的 query 起始位置（累积和）
    query_start_loc_ptr,
    # [num_reqs] - 每个请求的序列长度
    seq_lens_ptr,
    # [num_reqs, max_num_blocks] - 块表
    block_table_ptr,
    block_table_stride,
    block_size,  # 压缩后的块大小
    COMPRESS_RATIO: tl.constexpr,  # 压缩比率
    PAD_ID: tl.constexpr,  # 填充 ID（-1 表示无效）
    TRITON_BLOCK_SIZE: tl.constexpr,
):
    """
    Triton JIT 内核：计算压缩后的 slot mapping。

    每个 Triton 程序处理一个请求（batch 维度）。对于每个请求中的每个 query token：
    1. 计算该 token 在完整序列中的绝对位置 pos
    2. 判断 pos 是否满足压缩对齐条件：(pos + 1) % COMPRESS_RATIO == 0
    3. 如果满足条件：计算压缩后位置 -> 查找块表 -> 计算 slot ID
    4. 如果不满足条件：slot ID 设为 PAD_ID（-1）

    这样只有满足对齐条件的 token 才会写入 KV 缓存，其余 token 的 slot mapping 为 -1。
    """
    batch_idx = tl.program_id(0)

    query_start = tl.load(query_start_loc_ptr + batch_idx)
    query_end = tl.load(query_start_loc_ptr + batch_idx + 1)
    query_len = query_end - query_start

    seq_len = tl.load(seq_lens_ptr + batch_idx)
    start_pos = seq_len - query_len  # query token 在序列中的起始位置

    for i in range(0, query_len, TRITON_BLOCK_SIZE):
        offset = i + tl.arange(0, TRITON_BLOCK_SIZE)
        mask = offset < query_len

        pos = start_pos + i + tl.arange(0, TRITON_BLOCK_SIZE)
        # 判断是否满足压缩对齐条件
        is_valid = (pos + 1) % COMPRESS_RATIO == 0
        # 计算压缩后的位置
        pos_after_compress = pos // COMPRESS_RATIO

        # 通过块表查找物理块 ID
        block_ids = pos_after_compress // block_size
        block_numbers = tl.load(
            block_table_ptr + batch_idx * block_table_stride + block_ids,
            mask=mask & is_valid,
        )
        # 计算最终的 slot ID = 物理块号 * 块大小 + 块内偏移
        slot_ids = block_numbers * block_size + pos_after_compress % block_size

        # NOTE
        # 对于不满足压缩条件的位置，设为 PAD_ID（-1）
        slot_ids = tl.where(is_valid, slot_ids, PAD_ID)
        tl.store(slot_mapping_ptr + query_start + offset, slot_ids, mask=mask)


def get_compressed_slot_mapping(
    num_tokens: int,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    compress_ratio: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    计算压缩 KV 缓存的 slot mapping。

    对于给定的 query token 序列，根据压缩比率计算每个 token 在压缩 KV 缓存中的
    物理 slot 位置。只有满足对齐条件的 token 才有有效的 slot mapping。

    参数：
    - num_tokens: 总 token 数量
    - query_start_loc: 每个请求的 query 起始位置累积和 [num_reqs + 1]
    - seq_lens: 每个请求的序列长度 [num_reqs]
    - block_table: 块表 [num_reqs, max_num_blocks]
    - block_size: 压缩后的块大小
    - compress_ratio: 压缩比率（1, 4, 128 等）
    - out: 可选的输出缓冲区（用于 CUDA graph 地址稳定性）

    返回：
    - slot_mapping: 压缩后的 slot mapping [num_tokens]，无效位置为 -1
    """
    if out is not None:
        # Guard: for padded / invalid sequences.
        # Negative positions produce bogus block indices that lead to illegal memory
        # accesses inside the block_table load.
        # NOTE: Fill -1 to the whole tensor, not just the first `num_tokens`.
        # 保护：对于填充/无效序列。负位置会产生错误的块索引。
        # 填充整个张量，而不仅仅是前 num_tokens 个元素。
        out.fill_(-1)
        slot_mapping = out[:num_tokens]
    else:
        slot_mapping = torch.full(
            (num_tokens,), -1, dtype=torch.int64, device=query_start_loc.device
        )

    num_reqs = block_table.shape[0]
    _compressed_slot_mapping_kernel[(num_reqs,)](
        slot_mapping,
        query_start_loc,
        seq_lens,
        block_table,
        block_table.stride(0),
        block_size,
        compress_ratio,
        PAD_ID=-1,
        TRITON_BLOCK_SIZE=1024,
    )
    return slot_mapping
