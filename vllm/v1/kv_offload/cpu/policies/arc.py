# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# ==============================================================================
# ARC (Adaptive Replacement Cache) 缓存替换策略实现
# ==============================================================================
#
# ARC 是一种自适应缓存替换算法，由 Megiddo 和 Modha 在 2003 年提出。
# 它的核心优势是能够自动平衡 recency（最近访问）和 frequency（频繁访问）
# 两种访问模式，而不需要手动调参。
#
# ARC 的数据结构：
#   T1 (Recent): 存放只被访问过一次的块（最近访问）
#   T2 (Frequent): 存放被访问过多次的块（频繁访问）
#   B1 (Ghost List 1): 追踪最近从 T1 中被驱逐的块的 key（不存实际数据）
#   B2 (Ghost List 2): 追踪最近从 T2 中被驱逐的块的 key（不存实际数据）
#   target_t1_size: T1 的自适应目标大小
#
# ARC 的自适应机制：
#   当在 B1（T1 的 ghost list）中找到一个 key 时，说明最近访问模式更重要，
#   于是增加 target_t1_size，让更多缓存空间用于存储"最近访问"的块。
#   当在 B2（T2 的 ghost list）中找到一个 key 时，说明频繁访问模式更重要，
#   于是减少 target_t1_size，让更多缓存空间用于存储"频繁访问"的块。
#
# ARC 相比 LRU 的优势：
#   1. 对扫描型访问（scan）有更好的抵抗力——扫描的块只进入 T1，
#      不会污染 T2 中的频繁访问块
#   2. 自适应调整 recency vs frequency 的平衡，无需手动调参
#   3. 利用 ghost list 记住被驱逐块的历史，做出更智能的替换决策
#
# ARC 的驱逐策略：
#   当 T1 大小 >= target_t1_size 时，从 T1 驱逐（加入 B1）
#   否则从 T2 驱逐（加入 B2）
# ==============================================================================

from collections import OrderedDict
from collections.abc import Iterable

from vllm.v1.kv_offload.base import OffloadKey
from vllm.v1.kv_offload.cpu.policies.base import BlockStatus, CachePolicy


# 中文注释：ARCCachePolicy 实现了 ARC（自适应替换缓存）算法。
# ARC 维护四个有序字典：T1、T2、B1、B2，以及一个自适应的目标大小 target_t1_size。
# T1 和 B1 共享一个空间配额（由 target_t1_size 决定），
# T2 和 B2 共享另一个空间配额（cache_capacity - target_t1_size）。
class ARCCachePolicy(CachePolicy):
    """
    ARC (Adaptive Replacement Cache) cache policy.

    Data Structures:
        T1: Recent cache containing blocks accessed once.
        T2: Frequent cache containing blocks accessed multiple times.
        B1/B2: Ghost lists tracking recently evicted blocks from T1/T2.
        target_t1_size: Adaptive target size for the T1 partition.

    Algorithm Flow:
        1. Cache lookup (lookup):
           Searches T1 and T2 for block hashes and counts consecutive hits
           until a miss or non-ready block is encountered.

        2. Cache touch (touch) - Adaptive Learning:
           For each key (in reverse order):
           - If in T1: Move to T2 (promotion from recent to frequent).
           - If in T2: Move to MRU position (end of queue).
           - If in B1 ghost list: Increase target_t1_size.
           - If in B2 ghost list: Decrease target_t1_size.

        3. Block eviction (evict) - Adaptive Replacement:
           Determines eviction source based on adaptive target:
           - If T1 size >= target_t1_size: Evict from T1, add to B1.
           - Otherwise: Evict from T2, add to B2.
           Finally, bound each ghost list size.

        4. Block insertion (insert):
           New blocks are always inserted into T1 and removed from B1/B2 if
           present. Blocks may later be promoted to T2 during touch operations.

    Adaptive Behavior:
        The algorithm self-tunes the recency vs. frequency trade-off:
        - B1 hit: Recent access patterns matter more → increase T1.
        - B2 hit: Frequent access patterns matter more → decrease T1.
    """

    def __init__(self, cache_capacity: int):
        # 中文注释：cache_capacity 是整个缓存的总容量。
        # T1 + T2 的实际块数之和不应超过此值。
        # B1 + B2 是 ghost list，只存储 key，不存储实际数据，
        # 但它们的大小也会被限制在 cache_capacity 以内。
        self.cache_capacity: int = cache_capacity

        # 中文注释：target_t1_size 是 T1 的自适应目标大小。
        # 它会根据 B1/B2 的命中情况动态调整：
        #   - B1 命中 -> 增加 target_t1_size（更偏向 recency）
        #   - B2 命中 -> 减少 target_t1_size（更偏向 frequency）
        # 初始值为 0.0，表示刚开始时完全偏向 frequency。
        self.target_t1_size: float = 0.0

        # 中文注释：T1 (Recent) - 存放只被访问过一次的块。
        # 新插入的块先进入 T1。当再次被访问时，会晋升到 T2。
        self.t1: OrderedDict[OffloadKey, BlockStatus] = OrderedDict()

        # 中文注释：T2 (Frequent) - 存放被访问过多次的块。
        # 从 T1 晋升过来的块进入 T2，后续访问时更新其 LRU 位置。
        self.t2: OrderedDict[OffloadKey, BlockStatus] = OrderedDict()

        # 中文注释：B1 (Ghost List 1) - 追踪最近从 T1 中被驱逐的块的 key。
        # 只存储 key（值为 None），不存储实际的 BlockStatus 数据。
        # 当在 B1 中找到一个 key 时，说明该块之前在 T1 中被驱逐过，
        # 意味着"最近访问"模式更重要，应增加 target_t1_size。
        # key -> None (only care about presence)
        self.b1: OrderedDict[OffloadKey, None] = OrderedDict()

        # 中文注释：B2 (Ghost List 2) - 追踪最近从 T2 中被驱逐的块的 key。
        # 只存储 key（值为 None），不存储实际的 BlockStatus 数据。
        # 当在 B2 中找到一个 key 时，说明该块之前在 T2 中被驱逐过，
        # 意味着"频繁访问"模式更重要，应减少 target_t1_size。
        self.b2: OrderedDict[OffloadKey, None] = OrderedDict()

    def get(self, key: OffloadKey) -> BlockStatus | None:
        # 中文注释：在 T1 和 T2 中查找指定 key 的块。
        # 注意：这里不检查 ghost list（B1/B2），因为 ghost list 不存储实际数据。
        # 只是查找，不改变块在 LRU 顺序中的位置。
        return self.t1.get(key) or self.t2.get(key)

    def insert(self, key: OffloadKey, block: BlockStatus) -> None:
        # 中文注释：将新块插入到 T1（Recent 缓存）中。
        # 新块总是先进入 T1，表示它只被访问过一次。
        # 如果后续再次被访问，会通过 touch() 操作晋升到 T2。
        #
        # 同时从 B1/B2 中移除该 key：
        #   - 如果该 key 在 B1 中，说明它之前从 T1 被驱逐过，现在又回来了
        #   - 如果该 key 在 B2 中，说明它之前从 T2 被驱逐过，现在又回来了
        #   - 无论哪种情况，既然块现在已经在实际缓存中了，
        #     ghost list 中就不需要再保留这个 key 了
        self.t1[key] = block
        self.b1.pop(key, None)
        self.b2.pop(key, None)

    def remove(self, key: OffloadKey) -> None:
        # 中文注释：从 T1 或 T2 中移除指定块。
        # 先尝试从 T1 移除，如果 T1 中没有，再从 T2 移除。
        # 注意：这里不更新 ghost list，因为 remove 主要用于清理失败的写入。
        if self.t1.pop(key, None) is None:
            self.t2.pop(key, None)

    def touch(self, keys: Iterable[OffloadKey]) -> None:
        # 中文注释：标记指定的块为"最近使用"，并触发 ARC 的自适应学习。
        #
        # 这是 ARC 算法的核心操作，根据 key 当前所在的位置执行不同的操作：
        #
        # 情况 1：key 在 T1 中
        #   - 如果块已就绪（is_ready == True）：从 T1 晋升到 T2
        #     这表示该块被访问了两次，应该从"最近"变为"频繁"
        #   - 如果块未就绪（is_ready == False）：保留在 T1 中
        #     这表示块刚被准备好存储，还没有真正被访问两次，
        #     只是移动到 T1 的 MRU 位置
        #
        # 情况 2：key 在 T2 中
        #   - 将块移到 T2 的 MRU 位置（末尾），更新其 LRU 顺序
        #
        # 情况 3：key 在 B1（ghost list）中
        #   - 增加 target_t1_size，让更多空间用于 T1（最近访问）
        #   - 计算 delta = max(1, |B2|/|B1|)，这是一个自适应调整量
        #   - 将 key 移到 B1 的 MRU 位置，保持 ghost list 的新鲜度
        #
        # 情况 4：key 在 B2（ghost list）中
        #   - 减少 target_t1_size，让更多空间用于 T2（频繁访问）
        #   - 计算 delta = max(1, |B1|/|B2|)，这是一个自适应调整量
        #   - 将 key 移到 B2 的 MRU 位置，保持 ghost list 的新鲜度
        #
        # 注意：按逆序处理 keys，这样如果 keys 中有重复的 key，
        # 最后一次 touch 会生效（因为从后往前处理）。
        for key in reversed(list(keys)):
            if key in self.t1:
                block = self.t1.pop(key)
                if not block.is_ready:
                    # block was just prepared to be stored, not really touched
                    # twice — keep it in T1 and mark as most recently used
                    self.t1[key] = block
                else:
                    self.t2[key] = block

            elif key in self.t2:
                self.t2.move_to_end(key)

            elif key in self.b1:
                # 中文注释：B1 命中——说明"最近访问"模式更重要。
                # 增加 target_t1_size，让 T1 分配更多空间。
                # delta 的计算使得调整量与 B1、B2 的相对大小成正比：
                #   - 如果 B2 比 B1 大很多，delta 就大，调整幅度大
                #   - 如果 B1 和 B2 大小相近，delta 最小为 1
                delta = max(1, len(self.b2) / len(self.b1))
                self.target_t1_size = min(
                    self.target_t1_size + delta, self.cache_capacity
                )
                # move to MRU position (end) to keep it fresh in the ghost list
                self.b1.move_to_end(key)

            elif key in self.b2:
                # 中文注释：B2 命中——说明"频繁访问"模式更重要。
                # 减少 target_t1_size，让 T2 分配更多空间。
                # delta 的计算使得调整量与 B1、B2 的相对大小成正比：
                #   - 如果 B1 比 B2 大很多，delta 就大，调整幅度大
                #   - 如果 B1 和 B2 大小相近，delta 最小为 1
                delta = max(1, len(self.b1) / len(self.b2))
                self.target_t1_size = max(self.target_t1_size - delta, 0)
                # move to MRU position (end) to keep it fresh in the ghost list
                self.b2.move_to_end(key)

    def clear(self) -> None:
        # 中文注释：清空所有缓存数据结构和自适应状态。
        # T1、T2、B1、B2 全部清空，target_t1_size 重置为 0。
        self.t1.clear()
        self.t2.clear()
        self.b1.clear()
        self.b2.clear()
        self.target_t1_size = 0.0

    def evict(
        self, n: int, protected: set[OffloadKey]
    ) -> list[tuple[OffloadKey, BlockStatus]] | None:
        # 中文注释：驱逐恰好 n 个可驱逐的块。
        #
        # ARC 的驱逐算法流程：
        #
        # 步骤 1：原子性收集候选块
        #   使用 virtual_t1_size 虚拟变量模拟 T1 大小的变化，
        #   但不实际修改数据结构。这样可以保证操作的原子性：
        #   如果最终无法找到 n 个候选块，返回 None，不会有任何状态变更。
        #
        # 步骤 2：对于每个需要驱逐的块，根据 target_t1_size 决定驱逐来源：
        #   - 如果 virtual_t1_size >= target_t1_size：
        #     从 T1（最近访问）中驱逐，选择最旧的可驱逐块
        #     被驱逐的块的 key 加入 B1（ghost list）
        #   - 否则：
        #     从 T2（频繁访问）中驱逐，选择最旧的可驱逐块
        #     被驱逐的块的 key 加入 B2（ghost list）
        #
        # 步骤 3：执行实际驱逐
        #   从 T1 或 T2 中删除被驱逐的块
        #   将被驱逐块的 key 加入对应的 ghost list
        #
        # 步骤 4：裁剪 ghost list
        #   将 B1 和 B2 的大小限制在 cache_capacity 以内
        #   从最旧（LRU 位置）的 ghost 条目开始删除
        #
        # 参数说明：
        #   n: 需要驱逐的块数量
        #   protected: 受保护的块集合，这些块不会被驱逐
        #
        # 返回值：
        #   - 成功时返回 [(key, block), ...] 列表
        #   - 如果无法驱逐 n 个块，返回 None
        if n == 0:
            return []

        # Collect candidates atomically: simulate T1 size changes as we select,
        # but do not modify actual data structures until all n are found.
        candidates: list[
            tuple[OffloadKey, BlockStatus, bool]
        ] = []  # (key, block, from_t1)
        already_selected: set[OffloadKey] = set()
        virtual_t1_size = len(self.t1)

        for _ in range(n):
            candidate: tuple[OffloadKey, BlockStatus, bool] | None = None

            # 中文注释：如果 T1 的虚拟大小 >= target_t1_size，优先从 T1 驱逐。
            # 这是 ARC 自适应替换的核心：根据 target_t1_size 动态调整
            # 从 T1 还是 T2 驱逐的比例。
            if virtual_t1_size >= int(self.target_t1_size):
                for key, block in self.t1.items():
                    if (
                        block.ref_cnt == 0
                        and key not in protected
                        and key not in already_selected
                    ):
                        candidate = (key, block, True)
                        virtual_t1_size -= 1
                        break

            # 中文注释：如果 T1 中没有找到合适的候选块（或 T1 大小不足），
            # 则从 T2 中驱逐。这是 ARC 的回退策略。
            if candidate is None:
                for key, block in self.t2.items():
                    if (
                        block.ref_cnt == 0
                        and key not in protected
                        and key not in already_selected
                    ):
                        candidate = (key, block, False)
                        break
                # 中文注释：如果 T1 和 T2 中都无法找到合适的候选块，
                # 说明没有足够的可驱逐块（ref_cnt > 0 或都在 protected 中），
                # 返回 None，表示无法完成驱逐。
                if candidate is None:
                    return None

            candidates.append(candidate)
            already_selected.add(candidate[0])

        # Apply all evictions now that we know n candidates exist.
        # 中文注释：确定所有 n 个候选块后，执行实际的驱逐操作。
        # 这保证了操作的原子性：要么全部成功，要么全部不执行。
        result: list[tuple[OffloadKey, BlockStatus]] = []
        for key, block, from_t1 in candidates:
            if from_t1:
                # 中文注释：从 T1 驱逐的块，其 key 加入 B1（ghost list）。
                # B1 记住了"这个块之前作为 recent 块存在过"这一历史信息。
                del self.t1[key]
                self.b1[key] = None
            else:
                # 中文注释：从 T2 驱逐的块，其 key 加入 B2（ghost list）。
                # B2 记住了"这个块之前作为 frequent 块存在过"这一历史信息。
                del self.t2[key]
                self.b2[key] = None
            result.append((key, block))

        # Trim ghost lists to cache_capacity.
        # 中文注释：裁剪 ghost list，确保 B1 和 B2 的大小不超过 cache_capacity。
        # 从最旧的 ghost 条目开始删除（popitem(last=False) 删除头部，即最旧的）。
        # 这样可以保证 ghost list 中保留的是最近被驱逐的块的信息，
        # 而不是太久远的历史信息。
        for ghost in (self.b1, self.b2):
            for _ in range(len(ghost) - self.cache_capacity):
                ghost.popitem(last=False)

        return result
