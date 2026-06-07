# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
块表管理模块。

本模块实现了 vLLM v1 引擎中的块表（Block Table）管理功能。
块表是 KV 缓存管理系统的核心数据结构，用于将请求的逻辑 token 位置
映射到 KV 缓存中的物理存储位置。

核心概念：
1. 逻辑块 vs 物理块：
   - 逻辑块：请求视角的连续 token 分组
   - 物理块：KV 缓存中实际分配的存储单元
   - 块表维护逻辑块到物理块的映射关系

2. 块表结构：
   - 每个 KV 缓存组有独立的块表
   - 块表形状为 [max_num_reqs, max_num_blocks_per_req]
   - 使用 StagedWriteTensor 实现高效的批量更新

3. Slot 映射：
   - 将每个 token 的位置映射到 KV 缓存中的具体存储槽位
   - 考虑了块大小、上下文并行（CP）等因素

4. Triton 内核：
   - _gather_block_tables_kernel: 从分散的请求块表中收集到连续的批次块表
   - _compute_slot_mappings_kernel: 高效计算 slot 映射
"""
from collections.abc import Iterable

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.worker.gpu.buffer_utils import StagedWriteTensor, UvaBackedTensor


class BlockTables:
    """块表管理器，负责管理所有 KV 缓存组的块表。

    该类维护以下数据结构：
    - block_tables: 每个 KV 缓存组的块表（StagedWriteTensor），支持延迟写入
    - num_blocks: 每个请求在每个缓存组中已分配的块数（UVA 张量）
    - input_block_tables: 用于模型前向传播的块表副本
    - slot_mappings: 全局 slot 映射缓冲区

    属性:
        block_sizes: 每个 KV 缓存组的逻辑块大小
        kernel_block_sizes: 每个 KV 缓存组的内核块大小
        max_num_reqs: 最大请求数
        max_num_batched_tokens: 最大批处理 token 数
        cp_size: 上下文并行的进程数
        cp_rank: 当前进程在上下文并行中的排名
        cp_interleave: 上下文并行的交错因子
    """

    def __init__(
        self,
        block_sizes: list[int],
        max_num_reqs: int,
        max_num_batched_tokens: int,
        max_num_blocks_per_group: list[int],
        device: torch.device,
        kernel_block_sizes: list[int],
        cp_size: int = 1,
        cp_rank: int = 0,
        cp_interleave: int = 1,
    ):
        self.block_sizes = block_sizes
        self.kernel_block_sizes = kernel_block_sizes
        self.max_num_reqs = max_num_reqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.device = device

        self.cp_size = cp_size
        self.cp_rank = cp_rank
        self.cp_interleave = cp_interleave

        self.num_kv_cache_groups = len(self.block_sizes)
        assert len(max_num_blocks_per_group) == self.num_kv_cache_groups

        # 计算每个逻辑块对应的内核块数量
        self.blocks_per_kv_block = [
            bs // kbs for bs, kbs in zip(block_sizes, kernel_block_sizes)
        ]

        # 为每个 KV 缓存组分配块表缓冲区
        # 形状: num_kv_cache_groups x [max_num_reqs, max_num_blocks]
        self.block_tables: list[StagedWriteTensor] = []
        for i in range(self.num_kv_cache_groups):
            max_num_blocks = max_num_blocks_per_group[i] * self.blocks_per_kv_block[i]
            block_table = StagedWriteTensor(
                (self.max_num_reqs, max_num_blocks), dtype=torch.int32, device=device
            )
            self.block_tables.append(block_table)

        # 每个请求在每个缓存组中已分配的块数（UVA 张量，CPU/GPU 共享访问）
        self.num_blocks = UvaBackedTensor(
            (self.num_kv_cache_groups, self.max_num_reqs),
            dtype=torch.int32,
        )

        # 用于模型前向传播的块表（与 block_tables 形状相同）
        # num_kv_cache_groups x [max_num_reqs, max_num_blocks]
        self.input_block_tables: list[torch.Tensor] = [
            torch.zeros_like(b.gpu) for b in self.block_tables
        ]

        # 全局 slot 映射缓冲区
        self.slot_mappings = torch.zeros(
            self.num_kv_cache_groups,
            self.max_num_batched_tokens,
            dtype=torch.int64,
            device=self.device,
        )

        # 初始化布局相关的张量（指针、步进等）
        self.init_block_table_layout_tensors()

    def _make_ptr_tensor(self, x: Iterable[torch.Tensor]) -> torch.Tensor:
        """创建包含张量指针（数据地址）的张量。

        使用 uint64 类型以覆盖所有可能的内存地址。

        参数:
            x: 张量的可迭代对象

        返回:
            torch.Tensor: 包含各张量数据指针的 uint64 张量
        """
        # NOTE(woosuk): Use uint64 instead of int64 to cover all possible addresses.
        return torch.tensor(
            [t.data_ptr() for t in x], dtype=torch.uint64, device=self.device
        )

    def init_block_table_layout_tensors(self) -> None:
        """初始化块表布局相关的张量。

        在初始化时以及 CuMem KV 缓存唤醒后调用。指针张量缓存的是原始
        data_ptr() 值，当底层张量在唤醒后被重新分配时这些值会失效；
        block_sizes_tensor 需要重新填充，因为它的存储位于 KV 缓存池标签下，
        恢复后内容未定义。
        """
        self.block_table_ptrs = self._make_ptr_tensor(
            [b.gpu for b in self.block_tables]
        )
        self.block_table_strides = torch.tensor(
            [b.gpu.stride(0) for b in self.block_tables],
            dtype=torch.int64,
            device=self.device,
        )
        self.block_sizes_tensor = torch.tensor(
            self.kernel_block_sizes, dtype=torch.int32, device=self.device
        )
        self.input_block_table_ptrs = self._make_ptr_tensor(self.input_block_tables)

    def append_block_ids(
        self,
        req_index: int,
        new_block_ids: tuple[list[int], ...],
        overwrite: bool,
    ) -> None:
        """向指定请求的块表追加新的块 ID。

        当新分配了物理块时调用此方法。支持追加模式和覆写模式。

        参数:
            req_index: 请求在批次中的索引
            new_block_ids: 每个 KV 缓存组的新块 ID 列表
            overwrite: 是否覆写（而非追加）现有块 ID
        """
        for i in range(self.num_kv_cache_groups):
            start = self.num_blocks.np[i, req_index] if not overwrite else 0
            block_ids = new_block_ids[i]
            bpk = self.blocks_per_kv_block[i]
            if bpk > 1:
                # 将逻辑块 ID 展开为内核块 ID
                block_ids = [b * bpk + k for b in block_ids for k in range(bpk)]
            self.block_tables[i].stage_write(req_index, start, block_ids)
            self.num_blocks.np[i, req_index] = start + len(block_ids)

    def apply_staged_writes(self) -> None:
        """将所有暂存的写入操作应用到 GPU。

        将 CPU 端暂存的块 ID 写入操作批量提交到 GPU 上的块表中，
        同时将块数更新同步到 UVA 缓冲区。
        """
        # TODO(woosuk): This can be inefficient since it launches one kernel per
        # block table. Implement a kernel to handle all block tables at once.
        for block_table in self.block_tables:
            block_table.apply_write()
        self.num_blocks.copy_to_uva()

    def gather_block_tables(
        self,
        idx_mapping: torch.Tensor,
        num_reqs_padded: int,
    ) -> tuple[torch.Tensor, ...]:
        """根据批次索引映射收集块表到输入缓冲区。

        使用 Triton 内核将分散的请求块表收集到连续的批次块表中，
        以便模型前向传播使用。同时处理 padding 行的清零。

        参数:
            idx_mapping: 批次索引到请求状态索引的映射 [num_reqs]
            num_reqs_padded: 填充后的请求数（用于 CUDA Graph 兼容）

        返回:
            tuple[torch.Tensor, ...]: 每个 KV 缓存组的输入块表
        """
        num_reqs = idx_mapping.shape[0]
        # 使用填充后的请求数启动内核，融合 padding 行的清零操作
        _gather_block_tables_kernel[(self.num_kv_cache_groups, num_reqs_padded)](
            idx_mapping,
            self.block_table_ptrs,
            self.input_block_table_ptrs,
            self.block_table_strides,
            self.num_blocks.gpu,
            self.num_blocks.gpu.stride(0),
            num_reqs,
            BLOCK_SIZE=1024,  # type: ignore
        )
        return tuple(bt[:num_reqs_padded] for bt in self.input_block_tables)

    def get_dummy_block_tables(self, num_reqs: int) -> tuple[torch.Tensor, ...]:
        """获取用于 CUDA Graph 捕获的虚拟块表。

        返回持久化张量的视图，与模型前向传播使用相同的内存地址，
        而非分配新的张量。这对 CUDA Graph 捕获至关重要，
        因为 CUDA Graph 要求内存地址在捕获和重放时一致。

        参数:
            num_reqs: 请求数量

        返回:
            tuple[torch.Tensor, ...]: 每个 KV 缓存组的虚拟块表
        """
        # NOTE(woosuk): The output may be used for CUDA graph capture.
        # Therefore, this method must return the persistent tensor
        # with the same memory address as that used during the model's forward pass,
        # rather than allocating a new tensor.
        return tuple(block_table[:num_reqs] for block_table in self.input_block_tables)

    def compute_slot_mappings(
        self,
        idx_mapping: torch.Tensor,
        query_start_loc: torch.Tensor,
        positions: torch.Tensor,
        num_tokens_padded: int,
    ) -> torch.Tensor:
        """计算 slot 映射。

        使用 Triton 内核高效计算每个 token 在 KV 缓存中的物理存储位置。
        Slot 映射 = block_number * block_size + block_offset。

        对于上下文并行（CP）模式，还需要根据 CP 的分片策略
        计算本地 slot 映射，非本地的 slot 被标记为 PAD_SLOT_ID。

        参数:
            idx_mapping: 批次索引到请求状态索引的映射
            query_start_loc: 每个请求的查询起始位置
            positions: 每个 token 在其请求中的位置
            num_tokens_padded: 填充后的 token 数量

        返回:
            torch.Tensor: slot 映射 [num_groups, num_tokens_padded]
        """
        num_reqs = idx_mapping.shape[0]
        num_groups = self.num_kv_cache_groups
        _compute_slot_mappings_kernel[(num_groups, num_reqs + 1)](
            self.max_num_batched_tokens,
            idx_mapping,
            query_start_loc,
            positions,
            self.block_table_ptrs,
            self.block_table_strides,
            self.block_sizes_tensor,
            self.slot_mappings,
            self.slot_mappings.stride(0),
            self.cp_rank,
            CP_SIZE=self.cp_size,
            CP_INTERLEAVE=self.cp_interleave,
            PAD_ID=PAD_SLOT_ID,
            TRITON_BLOCK_SIZE=1024,  # type: ignore
        )
        return self.slot_mappings[:, :num_tokens_padded]

    def get_dummy_slot_mappings(self, num_tokens: int) -> torch.Tensor:
        """获取用于 CUDA Graph 捕获的虚拟 slot 映射。

        填充整个 slot_mappings 缓冲区为 PAD_SLOT_ID，而非仅前 num_tokens 个。
        这是因为 padding 逻辑复杂，内核可能访问超出请求范围的区域。

        返回持久化张量的视图，确保与模型前向传播使用相同的内存地址。

        参数:
            num_tokens: token 数量

        返回:
            torch.Tensor: 虚拟 slot 映射 [num_groups, num_tokens]
        """
        # Fill the entire slot_mappings tensor, not just the first `num_tokens` entries.
        # This is because the padding logic is complex and kernels may access beyond
        # the requested range.
        self.slot_mappings.fill_(PAD_SLOT_ID)
        # NOTE(woosuk): The output may be used for CUDA graph capture.
        # Therefore, this method must return the persistent tensor
        # with the same memory address as that used during the model's forward pass,
        # rather than allocating a new tensor.
        return self.slot_mappings[:, :num_tokens]


@triton.jit(do_not_specialize=["num_reqs"])
def _gather_block_tables_kernel(
    batch_idx_to_req_idx,  # [batch_size]
    src_block_table_ptrs,  # [num_kv_cache_groups]
    dst_block_table_ptrs,  # [num_kv_cache_groups]
    block_table_strides,  # [num_kv_cache_groups]
    num_blocks_ptr,  # [num_kv_cache_groups, max_num_reqs]
    num_blocks_stride,
    num_reqs,  # actual number of requests (for padding)
    BLOCK_SIZE: tl.constexpr,
):
    """收集块表的 Triton 内核。

    将分散在不同请求中的块表数据收集到连续的批次块表中。
    每个 Triton 程序处理一个 (KV 缓存组, 批次索引) 对。

    工作流程：
    1. 确定当前处理的 KV 缓存组和批次索引
    2. 如果批次索引超出实际请求数，清零该 padding 行
    3. 否则，通过 idx_mapping 找到对应的请求状态索引
    4. 读取该请求的已分配块数和源块表数据
    5. 将源块表数据拷贝到目标块表的对应行

    参数:
        batch_idx_to_req_idx: 批次索引到请求状态索引的映射
        src_block_table_ptrs: 源块表的 GPU 指针数组
        dst_block_table_ptrs: 目标块表的 GPU 指针数组
        block_table_strides: 每个块表的行步进
        num_blocks_ptr: 每个请求的已分配块数
        num_blocks_stride: num_blocks 的步进
        num_reqs: 实际请求数（用于区分 padding）
        BLOCK_SIZE: Triton 块大小
    """
    # kv cache group id
    group_id = tl.program_id(0)
    batch_idx = tl.program_id(1)

    stride = tl.load(block_table_strides + group_id)
    max_num_blocks = stride  # stride equals max_num_blocks for this group.
    dst_block_table_ptr = _load_ptr(dst_block_table_ptrs + group_id, tl.int32)
    dst_row_ptr = dst_block_table_ptr + batch_idx * stride

    if batch_idx >= num_reqs:
        # 超出实际请求数的行清零（处理 CUDA Graph 的 padding）
        for i in tl.range(0, max_num_blocks, BLOCK_SIZE):
            offset = i + tl.arange(0, BLOCK_SIZE)
            tl.store(dst_row_ptr + offset, 0, mask=offset < max_num_blocks)
        return

    # 获取请求状态索引和已分配块数
    req_idx = tl.load(batch_idx_to_req_idx + batch_idx)
    group_num_blocks_ptr = num_blocks_ptr + group_id * num_blocks_stride
    num_blocks = tl.load(group_num_blocks_ptr + req_idx)

    src_block_table_ptr = _load_ptr(src_block_table_ptrs + group_id, tl.int32)
    src_row_ptr = src_block_table_ptr + req_idx * stride

    # 拷贝块表数据
    for i in tl.range(0, num_blocks, BLOCK_SIZE):
        offset = i + tl.arange(0, BLOCK_SIZE)
        block_ids = tl.load(src_row_ptr + offset, mask=offset < num_blocks)
        tl.store(dst_row_ptr + offset, block_ids, mask=offset < num_blocks)


@triton.jit
def _compute_slot_mappings_kernel(
    max_num_tokens,
    idx_mapping,  # [num_reqs]
    query_start_loc,  # [num_reqs + 1]
    pos,  # [num_tokens]
    block_table_ptrs,  # [num_kv_cache_groups]
    block_table_strides,  # [num_kv_cache_groups]
    block_sizes,  # [num_kv_cache_groups]
    slot_mappings_ptr,  # [num_kv_cache_groups, max_num_tokens]
    slot_mappings_stride,
    cp_rank,
    CP_SIZE: tl.constexpr,
    CP_INTERLEAVE: tl.constexpr,
    PAD_ID: tl.constexpr,
    TRITON_BLOCK_SIZE: tl.constexpr,
):
    """计算 slot 映射的 Triton 内核。

    将每个 token 的位置映射到 KV 缓存中的物理存储槽位。

    计算公式（无上下文并行时）：
        block_index = position // block_size
        block_offset = position % block_size
        slot_id = block_table[req_idx, block_index] * block_size + block_offset

    计算公式（有上下文并行时）：
        在 CP 模式下，KV 缓存被交替分配给不同的 CP 排名。
        使用 round-robin 策略决定每个 token 属于哪个 CP 排名。

    最后一个程序块负责将未使用的 slot 位置填充为 PAD_ID。

    参数:
        max_num_tokens: 最大 token 数量
        idx_mapping: 批次索引到请求状态索引的映射
        query_start_loc: 每个请求的查询起始位置
        pos: 每个 token 的位置
        block_table_ptrs: 块表的 GPU 指针数组
        block_table_strides: 块表的行步进
        block_sizes: 每个 KV 缓存组的块大小
        slot_mappings_ptr: 输出 slot 映射的指针
        slot_mappings_stride: slot 映射的行步进
        cp_rank: 当前上下文并行排名
        CP_SIZE: 上下文并行的进程数
        CP_INTERLEAVE: 上下文并行的交错因子
        PAD_ID: padding 标识符
        TRITON_BLOCK_SIZE: Triton 块大小
    """
    # kv cache group id
    group_id = tl.program_id(0)
    batch_idx = tl.program_id(1)
    slot_mapping_ptr = slot_mappings_ptr + group_id * slot_mappings_stride

    if batch_idx == tl.num_programs(1) - 1:
        # 将剩余的 slot 填充为 PAD_ID。这对 CUDA Graph 是必需的。
        # 从实际 token 数（非填充后）开始填充，以覆盖分块预填充期间
        # 实际 token 和填充 token 之间可能残留的有效 slot ID。
        actual_num_tokens = tl.load(query_start_loc + batch_idx)
        for i in range(actual_num_tokens, max_num_tokens, TRITON_BLOCK_SIZE):
            offset = i + tl.arange(0, TRITON_BLOCK_SIZE)
            tl.store(slot_mapping_ptr + offset, PAD_ID, mask=offset < max_num_tokens)
        return

    block_table_ptr = _load_ptr(block_table_ptrs + group_id, tl.int32)
    block_table_stride = tl.load(block_table_strides + group_id)
    block_size = tl.load(block_sizes + group_id)

    req_state_idx = tl.load(idx_mapping + batch_idx)
    start_idx = tl.load(query_start_loc + batch_idx)
    end_idx = tl.load(query_start_loc + batch_idx + 1)
    for i in range(start_idx, end_idx, TRITON_BLOCK_SIZE):
        offset = i + tl.arange(0, TRITON_BLOCK_SIZE)
        positions = tl.load(pos + offset, mask=offset < end_idx, other=0)

        block_indices = positions // (block_size * CP_SIZE)
        block_offsets = positions % (block_size * CP_SIZE)
        block_numbers = tl.load(
            block_table_ptr + req_state_idx * block_table_stride + block_indices
        )

        if CP_SIZE == 1:
            # 常见情况：不使用上下文并行
            slot_ids = block_numbers * block_size + block_offsets
        else:
            # 使用上下文并行：根据交错策略计算本地 slot
            is_local = block_offsets // CP_INTERLEAVE % CP_SIZE == cp_rank
            rounds = block_offsets // (CP_INTERLEAVE * CP_SIZE)
            remainder = block_offsets % CP_INTERLEAVE
            local_offsets = rounds * CP_INTERLEAVE + remainder
            slot_ids = block_numbers * block_size + local_offsets
            slot_ids = tl.where(is_local, slot_ids, PAD_ID)

        tl.store(slot_mapping_ptr + offset, slot_ids, mask=offset < end_idx)


@triton.jit
def _load_ptr(ptr_to_ptr, elem_dtype):
    """从指针到指针加载实际指针值，并转换为目标元素类型。

    参数:
        ptr_to_ptr: 指向指针的指针
        elem_dtype: 目标元素数据类型

    返回:
        转换后的类型化指针，保证 16 字节对齐
    """
    ptr = tl.load(ptr_to_ptr)
    ptr = tl.cast(ptr, tl.pointer_type(elem_dtype))
    return tl.multiple_of(ptr, 16)
