# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import torch

from vllm.distributed import get_dcp_group, get_pcp_group
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.utils import CpuGpuBuffer
from vllm.v1.worker.cp_utils import get_total_cp_world_size

logger = init_logger(__name__)


class BlockTable:
    """
    块表类，用于管理 KV 缓存的块映射关系。
    
    该类维护了一个二维表格，记录每个请求使用的块 ID 列表，
    并提供槽位映射计算功能，用于注意力机制中快速定位 KV 缓存。
    """
    
    def __init__(
        self,
        block_size: int,
        max_num_reqs: int,
        max_num_blocks_per_req: int,
        max_num_batched_tokens: int,
        pin_memory: bool,
        device: torch.device,
        kernel_block_size: int,
        cp_kv_cache_interleave_size: int,
    ):
        """
        初始化块表。
        
        Args:
            block_size: Block size used for KV cache memory allocation
                KV 缓存内存分配的块大小
            max_num_reqs: Maximum number of concurrent requests supported.
                支持的最大并发请求数
            max_num_blocks_per_req: Maximum number of blocks per request.
                每个请求的最大块数
            max_num_batched_tokens: Maximum number of tokens in a batch.
                批次中的最大令牌数
            pin_memory: Whether to pin memory for faster GPU transfers.
                是否锁定内存以加速 GPU 传输
            device: Target device for the block table.
                块表的目标设备
            kernel_block_size: The block_size of underlying attention kernel.
                Will be the same as `block_size` if `block_size` is supported
                by the attention kernel.
                底层注意力内核的块大小。如果 `block_size` 被注意力内核支持，
                则与 `block_size` 相同
            cp_kv_cache_interleave_size: Context parallel KV cache interleave size.
                上下文并行 KV 缓存交错大小
        """
        self.max_num_reqs = max_num_reqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.pin_memory = pin_memory
        self.device = device

        if kernel_block_size == block_size:
            # Standard case: allocation and computation use same block size
            # No block splitting needed, direct mapping
            # 标准情况：分配和计算使用相同的块大小，不需要块拆分，直接映射
            self.block_size = block_size
            self.blocks_per_kv_block = 1
            self.use_hybrid_blocks = False
        else:
            # Hybrid case: allocation block size differs from kernel block size
            # Memory blocks are subdivided to match kernel requirements
            # Example: 32-token memory blocks with 16-token kernel blocks
            # → Each memory block corresponds to 2 kernel blocks
            # 混合情况：分配块大小与内核块大小不同
            # 内存块被细分以匹配内核需求
            # 例如：32 令牌的内存块配合 16 令牌的内核块
            # → 每个内存块对应 2 个内核块
            if block_size % kernel_block_size != 0:
                raise ValueError(
                    f"kernel_block_size {kernel_block_size} must divide "
                    f"kv_manager_block_size size {block_size} evenly"
                )

            self.block_size = kernel_block_size
            self.blocks_per_kv_block = block_size // kernel_block_size
            self.use_hybrid_blocks = True

        self.max_num_blocks_per_req = max_num_blocks_per_req * self.blocks_per_kv_block

        self.block_table = self._make_buffer(
            self.max_num_reqs, self.max_num_blocks_per_req, dtype=torch.int32
        )
        self.num_blocks_per_row = np.zeros(max_num_reqs, dtype=np.int32)

        self.slot_mapping = self._make_buffer(
            self.max_num_batched_tokens, dtype=torch.int64
        )

        if self.use_hybrid_blocks:
            self._kernel_block_arange = np.arange(0, self.blocks_per_kv_block).reshape(
                1, -1
            )
        else:
            self._kernel_block_arange = None

        try:
            self.pcp_world_size = get_pcp_group().world_size
            self.pcp_rank = get_pcp_group().rank_in_group
        except AssertionError:
            # PCP might not be initialized in testing
            self.pcp_world_size = 1
            self.pcp_rank = 0
        try:
            self.dcp_world_size = get_dcp_group().world_size
            self.dcp_rank = get_dcp_group().rank_in_group
        except AssertionError:
            # DCP might not be initialized in testing
            self.dcp_world_size = 1
            self.dcp_rank = 0
        self.cp_kv_cache_interleave_size = cp_kv_cache_interleave_size

    def append_row(
        self,
        block_ids: list[int],
        row_idx: int,
    ) -> None:
        """
        向指定行追加块 ID 列表。
        
        Args:
            block_ids: 要追加的块 ID 列表
            row_idx: 目标行索引
        """
        if not block_ids:
            return

        if self.use_hybrid_blocks:
            block_ids = self.map_to_kernel_blocks(
                np.array(block_ids), self.blocks_per_kv_block, self._kernel_block_arange
            )

        num_blocks = len(block_ids)
        start = self.num_blocks_per_row[row_idx]
        self.num_blocks_per_row[row_idx] += num_blocks
        self.block_table.np[row_idx, start : start + num_blocks] = block_ids

    def add_row(self, block_ids: list[int], row_idx: int) -> None:
        """
        添加新行到块表（会先清空该行）。
        
        Args:
            block_ids: 要添加的块 ID 列表
            row_idx: 目标行索引
        """
        self.num_blocks_per_row[row_idx] = 0
        self.append_row(block_ids, row_idx)

    def clear_row(self, row_idx: int) -> None:
        """
        清空指定行的块表数据。
        
        Args:
            row_idx: 要清空的行索引
        """
        num_blocks = self.num_blocks_per_row[row_idx]
        if num_blocks > 0:
            self.block_table.np[row_idx, :num_blocks] = 0
        self.num_blocks_per_row[row_idx] = 0

    def move_row(self, src: int, tgt: int) -> None:
        """
        将源行的数据移动到目标行。
        
        Args:
            src: 源行索引
            tgt: 目标行索引
        """
        num_blocks = self.num_blocks_per_row[src]
        block_table_np = self.block_table.np
        block_table_np[tgt, :num_blocks] = block_table_np[src, :num_blocks]
        self.num_blocks_per_row[tgt] = num_blocks

    def swap_row(self, src: int, tgt: int) -> None:
        """
        交换两行的数据。
        
        Args:
            src: 第一行索引
            tgt: 第二行索引
        """
        src_tgt, tgt_src = [src, tgt], [tgt, src]
        self.num_blocks_per_row[src_tgt] = self.num_blocks_per_row[tgt_src]
        self.block_table.np[src_tgt] = self.block_table.np[tgt_src]

    def compute_slot_mapping(
        self,
        num_reqs: int,
        query_start_loc: torch.Tensor,
        positions: torch.Tensor,
    ) -> None:
        """
        计算槽位映射，将令牌位置映射到 KV 缓存槽位。
        
        使用 Triton 内核并行计算每个令牌在 KV 缓存中的槽位 ID。
        
        Args:
            num_reqs: 请求数量
            query_start_loc: 查询起始位置张量 [num_reqs + 1]
            positions: 令牌位置张量 [num_tokens]
        """
        num_tokens = positions.shape[0]
        total_cp_world_size = self.pcp_world_size * self.dcp_world_size
        total_cp_rank = self.pcp_rank * self.dcp_world_size + self.dcp_rank
        _compute_slot_mapping_kernel[(num_reqs + 1,)](
            num_tokens,
            self.max_num_batched_tokens,
            query_start_loc,
            positions,
            self.block_table.gpu,
            self.block_table.gpu.stride(0),
            self.block_size,
            self.slot_mapping.gpu,
            TOTAL_CP_WORLD_SIZE=total_cp_world_size,
            TOTAL_CP_RANK=total_cp_rank,
            CP_KV_CACHE_INTERLEAVE_SIZE=self.cp_kv_cache_interleave_size,
            PAD_ID=PAD_SLOT_ID,
            BLOCK_SIZE=1024,
        )

    def commit_block_table(self, num_reqs: int) -> None:
        """
        将块表数据从 CPU 复制到 GPU。
        
        Args:
            num_reqs: 需要复制的请求数量
        """
        self.block_table.copy_to_gpu(num_reqs)

    def clear(self) -> None:
        """
        清空整个块表（CPU 和 GPU）。
        """
        self.block_table.gpu.fill_(0)
        self.block_table.cpu.fill_(0)

    @staticmethod
    def map_to_kernel_blocks(
        kv_manager_block_ids: np.ndarray,
        blocks_per_kv_block: int,
        kernel_block_arange: np.ndarray,
    ) -> np.ndarray:
        """Convert kv_manager_block_id IDs to kernel block IDs.
        
        将 KV 管理器块 ID 转换为内核块 ID。

        Example:
            # kv_manager_block_ids: 32 tokens,
            # Kernel block size: 16 tokens
            # blocks_per_kv_block = 2
            >>> kv_manager_block_ids = np.array([0, 1, 2])
            >>> Result: [0, 1, 2, 3, 4, 5]

            # Each kv_manager_block_id maps to 2 kernel block id:
            # kv_manager_block_id 0 → kernel block id [0, 1]
            # kv_manager_block_id 1 → kernel block id [2, 3]
            # kv_manager_block_id 2 → kernel block id [4, 5]
            
            示例：
            # kv_manager_block_ids: 32 令牌
            # 内核块大小: 16 令牌
            # blocks_per_kv_block = 2
            >>> kv_manager_block_ids = np.array([0, 1, 2])
            >>> 结果: [0, 1, 2, 3, 4, 5]
            
            # 每个 kv_manager_block_id 映射到 2 个内核块 ID：
            # kv_manager_block_id 0 → 内核块 ID [0, 1]
            # kv_manager_block_id 1 → 内核块 ID [2, 3]
            # kv_manager_block_id 2 → 内核块 ID [4, 5]
        
        Args:
            kv_manager_block_ids: KV 管理器块 ID 数组
            blocks_per_kv_block: 每个 KV 块包含的内核块数量
            kernel_block_arange: 内核块范围数组
            
        Returns:
            转换后的内核块 ID 数组
        """
        if blocks_per_kv_block == 1:
            return kv_manager_block_ids

        kernel_block_ids = (
            kv_manager_block_ids.reshape(-1, 1) * blocks_per_kv_block
            + kernel_block_arange
        )

        return kernel_block_ids.reshape(-1)

    def get_device_tensor(self, num_reqs: int) -> torch.Tensor:
        """Returns the device tensor of the block table.
        
        返回块表的设备（GPU）张量。
        
        Args:
            num_reqs: 需要获取的请求数量
            
        Returns:
            GPU 上的块表张量
        """
        return self.block_table.gpu[:num_reqs]

    def get_cpu_tensor(self) -> torch.Tensor:
        """Returns the CPU tensor of the block table.
        
        返回块表的 CPU 张量。
        
        Returns:
            CPU 上的块表张量
        """
        return self.block_table.cpu

    def get_numpy_array(self) -> np.ndarray:
        """Returns the numpy array of the block table.
        
        返回块表的 NumPy 数组。
        
        Returns:
            块表的 NumPy 数组
        """
        return self.block_table.np

    def _make_buffer(
        self, *size: int | torch.SymInt, dtype: torch.dtype
    ) -> CpuGpuBuffer:
        """
        创建 CPU-GPU 双缓冲区。
        
        Args:
            *size: 缓冲区大小
            dtype: 数据类型
            
        Returns:
            CpuGpuBuffer 实例
        """
        return CpuGpuBuffer(
            *size, dtype=dtype, device=self.device, pin_memory=self.pin_memory
        )


class MultiGroupBlockTable:
    """The BlockTables for each KV cache group.
    
    多组块表类，管理每个 KV 缓存组的块表。
    
    当模型有多种类型的 KV 缓存（如混合注意力机制）时，
    每种类型需要一个独立的块表，该类负责统一管理这些块表。
    """

    def __init__(
        self,
        max_num_reqs: int,
        max_model_len: int,
        max_num_batched_tokens: int,
        pin_memory: bool,
        device: torch.device,
        block_sizes: list[int],
        kernel_block_sizes: list[int],
        max_num_blocks: list[int] | None = None,
        cp_kv_cache_interleave_size: int = 1,
    ) -> None:
        """
        初始化多组块表。
        
        Args:
            max_num_reqs: 最大并发请求数
            max_model_len: 最大模型长度
            max_num_batched_tokens: 批次中最大令牌数
            pin_memory: 是否锁定内存
            device: 目标设备
            block_sizes: 各组的块大小列表
            kernel_block_sizes: 各组的内核块大小列表
            max_num_blocks: 各组的最大块数列表（可选）
            cp_kv_cache_interleave_size: 上下文并行 KV 缓存交错大小
        """
        if len(kernel_block_sizes) != len(block_sizes):
            raise ValueError(
                f"kernel_block_sizes length ({len(kernel_block_sizes)}) "
                f"must match block_sizes length ({len(block_sizes)})"
            )
        if max_num_blocks is None:
            # Note(hc): each dcp rank only store
            # (max_model_len//dcp_world_size) tokens in kvcache,
            # so the block_size which used for calc max_num_blocks_per_req
            # must be multiplied by dcp_world_size.
            # 注意(hc): 每个 dcp rank 只存储 (max_model_len//dcp_world_size) 个令牌到 kvcache，
            # 所以用于计算 max_num_blocks_per_req 的 block_size 必须乘以 dcp_world_size。
            total_cp_world_size = get_total_cp_world_size()
            max_num_blocks = [
                cdiv(max_model_len, block_size * total_cp_world_size)
                for block_size in block_sizes
            ]

        if len(max_num_blocks) != len(block_sizes):
            raise ValueError(
                f"max_num_blocks length ({len(max_num_blocks)}) "
                f"must match block_sizes length ({len(block_sizes)})"
            )

        # Align to a multiple of (128 / block_size) as required
        # by some attention backends such as TRTLLM (#39324)
        # 对齐到 (128 / block_size) 的倍数，这是某些注意力后端（如 TRTLLM）的要求
        max_num_blocks = [
            cdiv(n, 128 // bs) * (128 // bs) if bs <= 128 else n
            for n, bs in zip(max_num_blocks, block_sizes)
        ]

        self.block_tables = [
            BlockTable(
                block_size,
                max_num_reqs,
                max_num_blocks_per_req,
                max_num_batched_tokens,
                pin_memory,
                device,
                kernel_block_size,
                cp_kv_cache_interleave_size,
            )
            for block_size, kernel_block_size, max_num_blocks_per_req in zip(
                block_sizes, kernel_block_sizes, max_num_blocks
            )
        ]

    def append_row(self, block_ids: tuple[list[int], ...], row_idx: int) -> None:
        """
        向所有块表的指定行追加块 ID。
        
        Args:
            block_ids: 各组的块 ID 元组
            row_idx: 目标行索引
        """
        for i, block_table in enumerate(self.block_tables):
            block_table.append_row(block_ids[i], row_idx)

    def add_row(self, block_ids: tuple[list[int], ...], row_idx: int) -> None:
        """
        向所有块表添加新行。
        
        Args:
            block_ids: 各组的块 ID 元组
            row_idx: 目标行索引
        """
        for i, block_table in enumerate(self.block_tables):
            block_table.add_row(block_ids[i], row_idx)

    def clear_row(self, row_idx: int) -> None:
        """
        清空所有块表的指定行。
        
        Args:
            row_idx: 要清空的行索引
        """
        for block_table in self.block_tables:
            block_table.clear_row(row_idx)

    def move_row(self, src: int, tgt: int) -> None:
        """
        在所有块表中移动行数据。
        
        Args:
            src: 源行索引
            tgt: 目标行索引
        """
        for block_table in self.block_tables:
            block_table.move_row(src, tgt)

    def swap_row(self, src: int, tgt: int) -> None:
        """
        在所有块表中交换两行数据。
        
        Args:
            src: 第一行索引
            tgt: 第二行索引
        """
        for block_table in self.block_tables:
            block_table.swap_row(src, tgt)

    def compute_slot_mapping(
        self,
        num_reqs: int,
        query_start_loc: torch.Tensor,
        positions: torch.Tensor,
    ) -> None:
        """
        计算所有块表的槽位映射。
        
        Args:
            num_reqs: 请求数量
            query_start_loc: 查询起始位置张量
            positions: 令牌位置张量
        """
        for block_table in self.block_tables:
            block_table.compute_slot_mapping(num_reqs, query_start_loc, positions)

    def commit_block_table(self, num_reqs: int) -> None:
        """
        将所有块表数据提交到 GPU。
        
        Args:
            num_reqs: 请求数量
        """
        for block_table in self.block_tables:
            block_table.commit_block_table(num_reqs)

    def clear(self) -> None:
        """
        清空所有块表。
        """
        for block_table in self.block_tables:
            block_table.clear()

    def __getitem__(self, idx: int) -> "BlockTable":
        """Returns the BlockTable for the i-th KV cache group.
        
        返回第 i 个 KV 缓存组的块表。
        
        Args:
            idx: 组索引
            
        Returns:
            对应的 BlockTable 实例
        """
        return self.block_tables[idx]


@triton.jit
def _compute_slot_mapping_kernel(
    num_tokens,
    max_num_tokens,
    query_start_loc_ptr,  # [num_reqs + 1], int32
    positions_ptr,  # [num_tokens], int64
    block_table_ptr,  # [max_num_reqs, max_num_blocks_per_req], int32 (flat)
    block_table_stride,  # max_num_blocks_per_req
    block_size,
    slot_mapping_ptr,  # [max_num_tokens], int64
    TOTAL_CP_WORLD_SIZE: tl.constexpr,
    TOTAL_CP_RANK: tl.constexpr,
    CP_KV_CACHE_INTERLEAVE_SIZE: tl.constexpr,
    PAD_ID: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton 内核：计算槽位映射。
    
    该内核并行处理每个请求，将令牌位置映射到 KV 缓存槽位 ID。
    支持上下文并行（CP）场景下的槽位计算。
    
    Args:
        num_tokens: 令牌总数
        max_num_tokens: 最大令牌数（用于填充）
        query_start_loc_ptr: 查询起始位置指针 [num_reqs + 1]
        positions_ptr: 令牌位置指针 [num_tokens]
        block_table_ptr: 块表指针 [max_num_reqs, max_num_blocks_per_req]
        block_table_stride: 块表步长
        block_size: 块大小
        slot_mapping_ptr: 槽位映射输出指针 [max_num_tokens]
        TOTAL_CP_WORLD_SIZE: 总上下文并行世界大小（编译时常量）
        TOTAL_CP_RANK: 总上下文并行秩（编译时常量）
        CP_KV_CACHE_INTERLEAVE_SIZE: CP KV 缓存交错大小（编译时常量）
        PAD_ID: 填充 ID（编译时常量）
        BLOCK_SIZE: Triton 块大小（编译时常量）
    """
    req_idx = tl.program_id(0)
    # 获取请求索引

    if req_idx == tl.num_programs(0) - 1:
        # Pad remaining slots for CUDA graph compatibility.
        # 为 CUDA 图兼容性填充剩余槽位
        for i in range(num_tokens, max_num_tokens, BLOCK_SIZE):
            offsets = i + tl.arange(0, BLOCK_SIZE)
            tl.store(
                slot_mapping_ptr + offsets,
                PAD_ID,
                mask=offsets < max_num_tokens,
            )
        return

    # 加载当前请求的令牌范围
    start_idx = tl.load(query_start_loc_ptr + req_idx).to(tl.int64)
    end_idx = tl.load(query_start_loc_ptr + req_idx + 1).to(tl.int64)

    # 计算虚拟块大小（考虑上下文并行）
    virtual_block_size = block_size * TOTAL_CP_WORLD_SIZE
    row_offset = req_idx * block_table_stride
    
    # 遍历当前请求的所有令牌
    for i in range(start_idx, end_idx, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < end_idx
        
        # 加载令牌位置
        pos = tl.load(positions_ptr + offsets, mask=mask, other=0)
        
        # 计算块索引和块内偏移
        block_indices = pos // virtual_block_size
        block_numbers = tl.load(block_table_ptr + row_offset + block_indices).to(
            tl.int64
        )

        # 计算虚拟块内偏移
        virtual_block_offsets = pos - block_indices * virtual_block_size
        
        # 判断当前令牌是否属于本 rank（上下文并行场景）
        is_local = (
            virtual_block_offsets // CP_KV_CACHE_INTERLEAVE_SIZE
        ) % TOTAL_CP_WORLD_SIZE == TOTAL_CP_RANK
        
        # 计算本地块内偏移
        local_block_offsets = (
            virtual_block_offsets // (TOTAL_CP_WORLD_SIZE * CP_KV_CACHE_INTERLEAVE_SIZE)
        ) * CP_KV_CACHE_INTERLEAVE_SIZE + (
            virtual_block_offsets % CP_KV_CACHE_INTERLEAVE_SIZE
        )

        # 计算最终槽位 ID
        slot_ids = block_numbers * block_size + local_block_offsets
        slot_ids = tl.where(is_local, slot_ids, PAD_ID)
        
        # 存储槽位映射结果
        tl.store(slot_mapping_ptr + offsets, slot_ids, mask=mask)
