# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
稀疏 MLA 后端的通用工具函数模块。

本模块提供将索引器（Indexer）输出的 per-request 逻辑索引转换为全局物理索引的核心
Triton 内核和包装函数。这些物理索引用于在 KV 缓存中定位实际的数据位置。

核心功能：
- triton_convert_req_index_to_global_index(): 将稀疏 topk 逻辑索引转换为全局
  物理 slot 索引，支持 decode 和 prefill 两种模式
"""

import torch

from vllm.triton_utils import tl, triton


# 内核支持 prefill workspace 映射和有效索引计数追踪
@triton.jit
def _convert_req_index_to_global_index_kernel(
    req_id_ptr,  # int32 [num_tokens] - 每个 token 所属的请求 ID
    block_table_ptr,  # int32 [num_requests, max_num_blocks_per_req] - 块表
    token_indices_ptr,  # int32 [num_tokens, NUM_TOPK_TOKENS] - 稀疏 topk 逻辑索引
    out_ptr,  # int32 [num_tokens, NUM_TOPK_TOKENS] - 输出全局物理索引
    valid_count_ptr,  # int32 [num_tokens] - 输出每行有效索引数量
    prefill_request_id_ptr,  # int32 [num_tokens], -1 表示 decode, >=0 表示 prefill
    workspace_starts_ptr,  # int32 [num_prefill_reqs+1] 或 nullptr - prefill workspace 起始偏移
    # 形状（尽可能使用编译期常量）
    max_num_blocks_per_req: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,  # 列方向上的 tile 宽度
    HAS_PREFILL: tl.constexpr,  # 是否启用 prefill workspace 映射
    COUNT_VALID: tl.constexpr,  # 是否统计有效索引数量
    # 步幅（以元素为单位）
    bt_stride0,
    bt_stride1,
    ti_stride0,
    ti_stride1,
    out_stride0,
    out_stride1,
):
    """
    Triton JIT 内核：将稀疏索引从 per-request 逻辑索引转换为全局物理索引。

    每个 Triton 程序处理一个 token 行的一个列 tile。对于 decode token，
    通过 block_table 查找物理位置；对于 prefill token，映射到 workspace 偏移。

    转换公式：
      out[token_id, indice_id] =
        block_table[req_id[token_id], token_indices[token_id, indice_id] // BLOCK_SIZE]
        * BLOCK_SIZE + token_indices[token_id, indice_id] % BLOCK_SIZE

    特殊处理：
    - token_indices == -1 时输出 -1（无效标记）
    - block_id 越界时输出 -1
    - prefill token 映射到 workspace 偏移而非全局缓存 slot
    """
    # program_id(0) -> token_id（行索引）
    # program_id(1) -> tile 索引（列方向）
    token_id = tl.program_id(0)
    tile_id = tl.program_id(1)

    # 每个程序覆盖 BLOCK_N 个连续列
    indice_id = tile_id * BLOCK_N + tl.arange(0, BLOCK_N)

    # 加载该 token 的请求 ID（无 mask：grid 精确匹配）
    req = tl.load(req_id_ptr + token_id)

    # 加载该 tile 的 token 索引
    ti_ptr = token_indices_ptr + token_id * ti_stride0 + indice_id * ti_stride1
    tok = tl.load(ti_ptr)  # int32

    # 只有 token == -1 时应传播为 -1
    is_invalid_tok = tok < 0
    is_prefill = False
    if HAS_PREFILL:
        prefill_req_id = tl.load(prefill_request_id_ptr + token_id)
        is_prefill = prefill_req_id >= 0
    # 计算块 ID 和块内偏移
    block_id = tok // BLOCK_SIZE
    inblock_off = tok % BLOCK_SIZE

    # 保护 block_table 访问
    valid_block = (block_id < max_num_blocks_per_req) & (block_id >= 0)
    bt_ptr = block_table_ptr + req * bt_stride0 + block_id * bt_stride1
    is_invalid_tok |= ~valid_block
    base = tl.load(bt_ptr, mask=valid_block & ~is_prefill, other=0)
    out_val = base * BLOCK_SIZE + inblock_off

    # 如果启用 prefill，则覆盖输出值为 workspace 偏移
    if HAS_PREFILL:
        workspace_start = tl.load(
            workspace_starts_ptr + prefill_req_id, mask=is_prefill, other=0
        )
        prefill_out = workspace_start + tok
        out_val = tl.where(is_prefill, prefill_out, out_val)
    out_val = tl.where(is_invalid_tok, -1, out_val)

    # 存储结果
    out_ptr_ij = out_ptr + token_id * out_stride0 + indice_id * out_stride1
    tl.store(out_ptr_ij, out_val)

    # 统计该 tile 中的有效索引数量，并原子加到行总计
    if COUNT_VALID:
        tile_valid_count = tl.sum((~is_invalid_tok).to(tl.int32))
        tl.atomic_add(valid_count_ptr + token_id, tile_valid_count)


def triton_convert_req_index_to_global_index(
    req_id: torch.Tensor,  # int32 [num_tokens] - 每个 token 所属的请求 ID
    block_table: torch.Tensor,  # int32 [num_requests, max_num_blocks_per_req] - 块表
    token_indices: torch.Tensor,  # int32 [num_tokens, NUM_TOPK_TOKENS] - 稀疏 topk 逻辑索引
    BLOCK_SIZE: int = 64,  # KV 缓存块大小
    NUM_TOPK_TOKENS: int = 2048,  # 每个 token 的 topk 数量
    BLOCK_N: int = 128,  # 列方向上的 tile 宽度
    HAS_PREFILL_WORKSPACE: bool = False,  # 是否启用 prefill workspace 映射
    prefill_workspace_request_ids: torch.Tensor | None = None,  # prefill 请求 ID 映射
    prefill_workspace_starts: torch.Tensor | None = None,  # prefill workspace 起始偏移
    return_valid_counts: bool = False,  # 是否返回每行有效索引数量
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    将稀疏 topk 的 per-request 逻辑索引转换为全局物理索引。

    转换公式：
    out[token_id, indice_id] =
        block_table[req_id[token_id],
            token_indices[token_id, indice_id] // BLOCK_SIZE] * BLOCK_SIZE
        + token_indices[token_id, indice_id] % BLOCK_SIZE

    只有当 token_indices[token_id, indice_id] == -1 时才输出 -1。
    为了安全起见，如果派生的 block_id 越界，也会输出 -1。

    参数说明：
    - req_id: 每个 token 所属的请求 ID
    - block_table: 每个请求的物理块映射表
    - token_indices: 索引器输出的稀疏 topk 逻辑索引
    - BLOCK_SIZE: KV 缓存的块大小
    - NUM_TOPK_TOKENS: 每个 token 选择的 topk token 数量
    - BLOCK_N: Triton 内核的列 tile 宽度（性能调优参数）
    - HAS_PREFILL_WORKSPACE: 是否为 prefill token 使用 workspace 映射
    - prefill_workspace_request_ids: prefill 请求 ID 映射（-1 表示 decode）
    - prefill_workspace_starts: 每个 prefill 请求的 workspace 起始偏移
    - return_valid_counts: 是否同时返回每行的有效索引计数

    返回：
    - 如果 return_valid_counts=False: 返回转换后的物理索引张量
    - 如果 return_valid_counts=True: 返回 (物理索引, 有效计数) 元组
    """
    assert req_id.dtype == torch.int32
    assert block_table.dtype == torch.int32
    assert token_indices.dtype == torch.int32
    assert token_indices.shape[1] == NUM_TOPK_TOKENS
    assert NUM_TOPK_TOKENS % BLOCK_N == 0, (
        f"NUM_TOPK_TOKENS ({NUM_TOPK_TOKENS}) must be divisible by BLOCK_N ({BLOCK_N})"
    )

    if HAS_PREFILL_WORKSPACE:
        assert prefill_workspace_request_ids is not None
        assert prefill_workspace_starts is not None
        assert prefill_workspace_request_ids.dtype == torch.int32
        assert prefill_workspace_starts.dtype == torch.int32

    num_tokens = req_id.shape[0]
    max_num_blocks_per_req = block_table.shape[1]
    tiles_per_row = NUM_TOPK_TOKENS // BLOCK_N

    # 确保张量在同一个设备上且连续
    req_id_c = req_id.contiguous()
    block_table_c = block_table.contiguous()
    token_indices_c = token_indices.contiguous()
    out = torch.empty_like(token_indices_c)

    # 如果需要统计有效数量，则分配零初始化的缓冲区（用于原子操作）
    valid_counts: torch.Tensor | None = None
    if return_valid_counts:
        valid_counts = torch.zeros(
            num_tokens, dtype=torch.int32, device=token_indices.device
        )

    # 步幅（以元素为单位）
    bt_stride0, bt_stride1 = block_table_c.stride()
    ti_stride0, ti_stride1 = token_indices_c.stride()
    out_stride0, out_stride1 = out.stride()

    # 准备 prefill 指针
    if HAS_PREFILL_WORKSPACE:
        assert prefill_workspace_request_ids is not None  # 用于 mypy 类型检查
        assert prefill_workspace_starts is not None  # 用于 mypy 类型检查
        assert prefill_workspace_request_ids.is_contiguous()
        assert prefill_workspace_starts.is_contiguous()

    # 精确的 2D grid: tokens x 列 tiles
    grid = (num_tokens, tiles_per_row)

    _convert_req_index_to_global_index_kernel[grid](
        req_id_c,
        block_table_c,
        token_indices_c,
        out,
        valid_counts,
        prefill_workspace_request_ids,
        prefill_workspace_starts,
        # 形状 / 编译期常量
        max_num_blocks_per_req,
        BLOCK_SIZE,
        BLOCK_N,
        HAS_PREFILL_WORKSPACE,
        return_valid_counts,
        # 步幅
        bt_stride0,
        bt_stride1,
        ti_stride0,
        ti_stride1,
        out_stride0,
        out_stride1,
    )

    if return_valid_counts:
        assert valid_counts is not None
        return out, valid_counts
    return out
