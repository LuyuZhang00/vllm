# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# ==============================================================================
# KV Offload 缓存替换策略基类定义
# ==============================================================================
#
# 本文件定义了 CPU 端 KV 缓存卸载（offload）的缓存替换策略的抽象基类。
#
# 背景说明：
#   vLLM 中，GPU 显存有限，当 KV 缓存过多时，需要将部分 KV 缓存卸载到 CPU 内存。
#   卸载到 CPU 的 KV 缓存块需要一个缓存替换策略来管理：当 CPU 内存也满时，
#   决定哪些块应该被驱逐（evict），为新的块腾出空间。
#
# 本文件包含两个核心类：
#   1. BlockStatus - 用 C 结构体表示的单个 KV 块的卸载状态
#      - ref_cnt: 引用计数，表示有多少传输正在使用此块作为数据源
#      - block_id: 物理 CPU 缓冲区槽位的索引
#
#   2. CachePolicy - 缓存替换策略的抽象基类（ABC）
#      - 定义了所有缓存策略必须实现的接口
#      - 已实现的具体策略：LRU（最近最少使用）和 ARC（自适应替换缓存）
#
# 设计思路：
#   CachePolicy 封装了数据结构组织和驱逐决策两个维度。
#   LRU 和 ARC 在这两个维度上都有差异——ARC 的 ghost list 和 target_t1_size
#   处于存储和驱逐的交叉点，因此无法干净地分离。
# ==============================================================================

import ctypes
from abc import ABC, abstractmethod
from collections.abc import Iterable

from vllm.v1.kv_offload.base import OffloadKey


# 中文注释：BlockStatus 使用 ctypes.Structure 定义为 C 结构体，
# 这样可以在 Python 和 C/CUDA 代码之间高效传递，避免 Python 对象的开销。
# 每个 BlockStatus 对应一个 CPU 端的 KV 缓存块的状态。
class BlockStatus(ctypes.Structure):
    """
    Offloading status for a single block of KV data.
    Holds the following information:

    ref_cnt - the current number of transfers using this block as a source.
        A value of -1 indicates the block is not yet ready to be read.
    block_id - index of the physical CPU buffer slot.
    """

    # 中文注释：定义 C 结构体字段布局
    # ref_cnt (int32): 引用计数，-1 表示块尚未就绪（正在写入中），
    #   0 表示空闲可用，>0 表示有正在使用此块的传输操作
    # block_id (int64): 物理 CPU 缓冲区的槽位索引
    _fields_ = [("ref_cnt", ctypes.c_int32), ("block_id", ctypes.c_int64)]

    def __init__(self, block_id: int):
        super().__init__()
        # initialize block as "not ready" (ref_cnt = -1)
        # 中文注释：新创建的块初始化为"未就绪"状态（ref_cnt = -1），
        # 表示该块正在被写入数据，还不能被读取。
        # 当数据写入完成后，ref_cnt 会被设置为 0，此时块才变为可读状态。
        self.ref_cnt = -1
        self.block_id = block_id

    @property
    def is_ready(self) -> bool:
        """
        Returns whether the block is ready to be read.
        """
        # 中文注释：判断块是否就绪可读。
        # ref_cnt >= 0 表示数据已写入完成，可以安全读取。
        # ref_cnt == -1 表示正在写入中，此时不应读取该块的数据。
        return self.ref_cnt >= 0


# 中文注释：CachePolicy 是缓存替换策略的抽象基类。
# 它定义了缓存策略必须实现的核心操作接口：
#   - 查找 (get)：根据 key 查找缓存块
#   - 插入 (insert)：将新分配的块加入缓存
#   - 删除 (remove)：移除缓存块（用于清理失败的写入）
#   - 触摸 (touch)：标记块为最近使用（影响替换优先级）
#   - 驱逐 (evict)：选择并移除 n 个可驱逐的块
#   - 清空 (clear)：清空所有缓存块和状态
class CachePolicy(ABC):
    """
    Encapsulates both block organization (data structures) and replacement
    decisions (which block to evict). LRU and ARC differ in both dimensions —
    ARC's ghost lists and target_t1_size live at the intersection of storage
    and eviction, so they cannot be separated cleanly.
    """

    @abstractmethod
    def __init__(self, cache_capacity: int) -> None: ...

    @abstractmethod
    def get(self, key: OffloadKey) -> BlockStatus | None:
        """Find block in data structures. Returns None if not present."""
        # 中文注释：在缓存数据结构中查找指定 key 的块。
        # 如果找到返回 BlockStatus（包含块的状态信息），否则返回 None。

    @abstractmethod
    def insert(self, key: OffloadKey, block: BlockStatus) -> None:
        """Add a newly allocated block. For ARC: also removes from ghost lists."""
        # 中文注释：将新分配的块插入缓存。
        # 对于 ARC 策略，还会从 ghost list（B1/B2）中移除该 key，
        # 因为该块现在已经在实际缓存中了。

    @abstractmethod
    def remove(self, key: OffloadKey) -> None:
        """Remove a block (used to clean up after a failed store)."""
        # 中文注释：从缓存中移除指定块。
        # 主要用于清理失败的写入操作（例如写入过程中出错时回滚）。

    @abstractmethod
    def touch(self, keys: Iterable[OffloadKey]) -> None:
        """Mark blocks as recently used."""
        # 中文注释：标记指定的块为"最近使用"。
        # 这会影响缓存替换的优先级——最近使用的块更不容易被驱逐。
        # 对于 LRU：将块移到 OrderedDict 的末尾（MRU 位置）。
        # 对于 ARC：可能触发 T1 -> T2 的晋升，或调整 target_t1_size。

    @abstractmethod
    def evict(
        self, n: int, protected: set[OffloadKey]
    ) -> list[tuple[OffloadKey, BlockStatus]] | None:
        """
        Evict exactly n blocks, skipping any in protected.

        Returns a list of (key, block) for the evicted blocks,
        or None if n evictions cannot be satisfied. The operation is atomic:
        if None is returned, no state changes are made.

        For ARC: ghost list cleanup (trimming to cache_capacity) is performed
        at the end of a successful eviction.
        """
        # 中文注释：驱逐恰好 n 个可驱逐的块，跳过 protected 集合中的块。
        #
        # 参数说明：
        #   n: 需要驱逐的块数量
        #   protected: 受保护的块集合，这些块不会被驱逐
        #
        # 返回值：
        #   - 成功时返回 [(key, block), ...] 列表，包含被驱逐块的 key 和状态
        #   - 如果无法驱逐 n 个块（可用块不足），返回 None
        #
        # 重要：操作是原子性的——如果返回 None，不会有任何状态变更。
        # 只有在找到所有 n 个候选块后，才会真正执行驱逐操作。

    @abstractmethod
    def clear(self) -> None:
        """
        Remove ALL blocks regardless of ref_cnt.

        Ghost lists and adaptive state are also reset.
        """
        # 中文注释：清空所有缓存块，无论其引用计数如何。
        # 同时重置 ghost list 和自适应状态（如 ARC 的 target_t1_size）。
        # 通常在系统重置或上下文切换时调用。
