# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# ==============================================================================
# LRU (Least Recently Used) 缓存替换策略实现
# ==============================================================================
#
# LRU 是一种简单且广泛使用的缓存替换策略。其核心思想是：
#   当缓存满时，优先驱逐最久未被访问的块。
#
# 实现方式：
#   使用 Python 的 OrderedDict 维护一个有序字典。
#   OrderedDict 内部是一个双向链表 + 哈希表的组合：
#     - 哈希表提供 O(1) 的查找
#     - 双向链表维护插入/访问顺序
#
# LRU 的工作原理：
#   1. 插入（insert）：新块被插入到 OrderedDict 末尾（最"新"的位置）
#   2. 访问（touch）：被访问的块通过 move_to_end 移到末尾（标记为最近使用）
#   3. 驱逐（evict）：从 OrderedDict 开头（最"旧"的位置）开始驱逐
#
# 与其他策略的对比：
#   - LRU 实现简单，开销低，但对扫描型访问模式（scan resistance）表现不佳
#   - ARC（自适应替换缓存）通过维护 recent 和 frequent 两个列表，以及 ghost list
#     来自适应地平衡 recency 和 frequency，但实现更复杂
# ==============================================================================

from collections import OrderedDict
from collections.abc import Iterable

from vllm.v1.kv_offload.base import OffloadKey
from vllm.v1.kv_offload.cpu.policies.base import BlockStatus, CachePolicy


# 中文注释：LRUCachePolicy 实现了 LRU 缓存替换策略。
# 使用 OrderedDict 作为底层数据结构，利用其有序性来追踪访问顺序。
# OrderedDict 的头部是最近最少使用的块，尾部是最近使用的块。
class LRUCachePolicy(CachePolicy):
    """LRU cache policy backed by a single OrderedDict."""

    def __init__(self, cache_capacity: int):
        # cache_capacity unused by LRU but accepted for a uniform constructor
        # 中文注释：cache_capacity 参数在 LRU 中未被使用（LRU 不需要预设容量限制），
        # 但为了保持所有 CachePolicy 子类构造函数签名一致而保留。
        self.blocks: OrderedDict[OffloadKey, BlockStatus] = OrderedDict()

    def get(self, key: OffloadKey) -> BlockStatus | None:
        # 中文注释：根据 key 查找缓存块。
        # 只是查找，不改变块在 LRU 顺序中的位置。
        # 如果需要标记为最近使用，应该调用 touch() 方法。
        return self.blocks.get(key)

    def insert(self, key: OffloadKey, block: BlockStatus) -> None:
        # 中文注释：将新块插入缓存。
        # OrderedDict 会将新 key 放在末尾（最近使用的位置），
        # 所以新插入的块是最不容易被驱逐的。
        self.blocks[key] = block

    def remove(self, key: OffloadKey) -> None:
        # 中文注释：从缓存中移除指定块。
        # 主要用于清理失败的写入操作（例如数据传输失败时回滚）。
        del self.blocks[key]

    def touch(self, keys: Iterable[OffloadKey]) -> None:
        # 中文注释：标记指定的块为"最近使用"。
        # 按照 keys 的逆序处理，这样如果 keys 中有重复的 key，
        # 最后一次 touch 会生效（因为从后往前处理）。
        # move_to_end(key) 将块移到 OrderedDict 的末尾，标记为最近使用。
        for key in reversed(list(keys)):
            if key in self.blocks:
                self.blocks.move_to_end(key)

    def clear(self) -> None:
        # 中文注释：清空所有缓存块。
        # OrderedDict.clear() 会清空所有键值对。
        self.blocks.clear()

    def evict(
        self, n: int, protected: set[OffloadKey]
    ) -> list[tuple[OffloadKey, BlockStatus]] | None:
        # 中文注释：驱逐恰好 n 个可驱逐的块。
        #
        # 算法流程：
        #   1. 遍历 OrderedDict（从头到尾，即从最旧到最新）
        #   2. 对于每个块，检查是否可驱逐：
        #      - ref_cnt == 0：没有正在使用此块的传输操作
        #      - key not in protected：不在受保护集合中
        #   3. 收集 n 个候选块
        #   4. 如果找到的候选块不足 n 个，返回 None（原子性保证）
        #   5. 否则，删除这些块并返回结果
        #
        # 注意：由于 OrderedDict 按照访问顺序排列，遍历时从头部开始
        # 就是最近最少使用的块，这正是 LRU 策略的核心。
        if n == 0:
            return []
        candidates: list[tuple[OffloadKey, BlockStatus]] = []
        for key, block in self.blocks.items():
            if block.ref_cnt == 0 and key not in protected:
                candidates.append((key, block))
                if len(candidates) == n:
                    break
        if len(candidates) < n:
            return None
        for key, _ in candidates:
            del self.blocks[key]
        return candidates
