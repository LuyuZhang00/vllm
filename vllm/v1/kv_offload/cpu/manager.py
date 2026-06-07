# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
CPU 卸载管理器模块 (vllm/v1/kv_offload/cpu/manager.py)

本模块实现了 CPU 端的 KV 缓存卸载管理器，负责管理存储在 CPU 内存中的 KV 缓存块
的生命周期。

核心组件：
1. CPUOffloadingManager: 卸载管理器，实现 OffloadingManager 接口
2. 缓存策略 (CachePolicy): 可插拔的淘汰策略（LRU 或 ARC）

管理器职责：
- 块池管理：分配、回收 CPU 缓存块
- 引用计数：跟踪每个缓存块的引用数，防止正在使用的块被淘汰
- 淘汰决策：当 CPU 内存不足时，按策略淘汰最不常用的块
- 事件通知：记录缓存事件（新增、淘汰），供监控和调试使用

缓存块状态机：
1. 未分配 -> 准备存储 (prepare_store) -> 存储中 (is_ready=False)
2. 存储中 -> 存储完成 (complete_store) -> 就绪 (is_ready=True)
3. 就绪 -> 准备加载 (prepare_load) -> 使用中 (ref_cnt > 0)
4. 使用中 -> 加载完成 (complete_load) -> 就绪 (ref_cnt -> 0)
5. 就绪/使用中 -> 淘汰 (evict) -> 未分配

工作流程（store 为例）：
1. 调度器调用 prepare_store(keys)：
   a. 过滤掉已存储的块
   b. 如果 CPU 内存不足，淘汰旧块
   c. 分配新块并插入缓存策略
   d. 返回需要存储的块 ID 和被淘汰的块
2. 工作器执行实际的数据传输
3. 调度器调用 complete_store(keys)：
   a. 将块标记为就绪 (is_ready=True)
   b. 记录存储事件
"""

from collections import OrderedDict
from collections.abc import Collection, Iterable
from typing import Literal

from vllm.v1.kv_offload.base import (
    LoadStoreSpec,
    OffloadingEvent,
    OffloadingManager,
    OffloadKey,
    PrepareStoreOutput,
    ReqContext,
    RequestOffloadingContext,
)
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from vllm.v1.kv_offload.cpu.policies.arc import ARCCachePolicy
from vllm.v1.kv_offload.cpu.policies.base import BlockStatus, CachePolicy
from vllm.v1.kv_offload.cpu.policies.lru import LRUCachePolicy

# 支持的缓存策略注册表
_CACHE_POLICIES: dict[str, type[CachePolicy]] = {
    "lru": LRUCachePolicy,    # 最近最少使用策略
    "arc": ARCCachePolicy,     # 自适应替换缓存策略
}


class CPUOffloadingManager(OffloadingManager):
    """
    CPU KV 缓存卸载管理器。

    实现了 OffloadingManager 接口，管理存储在 CPU 内存中的 KV 缓存块。
    使用可插拔的 CachePolicy（LRU 或 ARC）来决定淘汰顺序。

    管理器负责的共享逻辑：
    - 引用计数管理
    - 事件发射（缓存新增/淘汰通知）
    - 块池管理（分配/回收）
    - prepare_store/complete_store 骨架逻辑

    策略特定的逻辑委托给 CachePolicy 实现：
    - 块的组织方式
    - 淘汰候选的选择

    参数：
        num_blocks: CPU 缓存块总数
        cache_policy: 缓存策略名称（"lru" 或 "arc"）
        enable_events: 是否启用事件记录（用于 KV 缓存事件通知）
        store_threshold: 块在 lookup 中出现多少次后才允许卸载到 CPU
            （阈值 < 2 时禁用过滤）
        max_tracker_size: 引用计数跟踪器的最大容量
    """

    def __init__(
        self,
        num_blocks: int,
        cache_policy: Literal["lru", "arc"] = "lru",
        enable_events: bool = False,
        store_threshold: int = 1,
        max_tracker_size: int = 64_000,
    ):
        # 存储介质标识
        self.medium: str = CPULoadStoreSpec.medium()
        # CPU 缓存块总数
        self._num_blocks: int = num_blocks
        # 已分配的块数（单调递增，用于分配新块）
        self._num_allocated_blocks: int = 0
        # 空闲块列表（被淘汰后回收的块 ID）
        self._free_list: list[int] = []
        # 事件记录列表（None 表示禁用事件）
        self.events: list[OffloadingEvent] | None = [] if enable_events else None
        # 初始化缓存策略
        policy_cls = _CACHE_POLICIES.get(cache_policy)
        if policy_cls is None:
            raise ValueError(
                f"Unknown cache policy: {cache_policy!r}. "
                f"Supported: {list(_CACHE_POLICIES)}"
            )
        self._policy: CachePolicy = policy_cls(cache_capacity=num_blocks)
        # 卸载阈值：块需要在 lookup 中出现此次数后才允许卸载
        self.store_threshold: int = store_threshold
        # 跟踪器最大容量
        self.max_tracker_size: int = max_tracker_size

        # 引用计数有序字典。使用 OrderedDict 以便在 O(1) 时间内淘汰 LRU 条目。
        # 仅当 store_threshold >= 2 时启用（阈值 < 2 时无需过滤）。
        self.counts: OrderedDict[OffloadKey, int] | None = (
            OrderedDict() if store_threshold >= 2 else None
        )

    # --- 块池管理 ---

    def _get_num_free_blocks(self) -> int:
        """
        获取当前可用的空闲块数。

        包括：空闲列表中的块 + 尚未分配的新块。

        Returns:
            当前可用的空闲块数
        """
        return len(self._free_list) + self._num_blocks - self._num_allocated_blocks

    def _allocate_blocks(self, keys: list[OffloadKey]) -> list[BlockStatus]:
        """
        为给定的键分配缓存块。

        优先分配新块（尚未使用过的块 ID），不足时从空闲列表中回收。

        Args:
            keys: 需要分配块的键列表

        Returns:
            分配的 BlockStatus 列表
        """
        # 计算可以分配的新块数
        num_fresh = min(len(keys), self._num_blocks - self._num_allocated_blocks)
        # 剩余需要从空闲列表中回收的块数
        num_reused = len(keys) - num_fresh
        assert len(self._free_list) >= num_reused

        # 分配新块（从未使用过的块 ID）
        blocks: list[BlockStatus] = []
        for _ in range(num_fresh):
            blocks.append(BlockStatus(self._num_allocated_blocks))
            self._num_allocated_blocks += 1

        # 从空闲列表中回收块
        for _ in range(num_reused):
            blocks.append(BlockStatus(self._free_list.pop()))
        return blocks

    def _free_block(self, block: BlockStatus) -> None:
        """
        将块归还到空闲列表。

        Args:
            block: 要释放的块状态对象
        """
        self._free_list.append(block.block_id)

    def _get_load_store_spec(
        self,
        keys: Iterable[OffloadKey],
        blocks: Iterable[BlockStatus],
    ) -> CPULoadStoreSpec:
        """
        创建 CPU 加载/存储规格。

        Args:
            keys: 键列表
            blocks: 对应的块状态列表

        Returns:
            包含块 ID 列表的 CPULoadStoreSpec
        """
        return CPULoadStoreSpec([block.block_id for block in blocks])

    # --- OffloadingManager 接口实现 ---

    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext:
        """
        新请求到达时的回调。

        为新请求创建卸载上下文（目前为空实现）。

        Args:
            req_context: 请求上下文

        Returns:
            请求卸载上下文
        """
        return RequestOffloadingContext()

    def lookup(self, key: OffloadKey, req_context: ReqContext) -> bool | None:
        """
        查找键对应的缓存块是否存在于 CPU 缓存中。

        查找逻辑：
        1. 如果启用了引用计数跟踪器，更新计数和 LRU 顺序
        2. 查询缓存策略中是否存在该键
        3. 返回状态：
           - True: 块存在且已就绪，可以加载
           - False: 块不存在
           - None: 块存在但正在写入中（is_ready=False），调用者应重试

        Args:
            key: 要查找的缓存键
            req_context: 请求上下文

        Returns:
            True/False/None 表示缓存块状态
        """
        if self.counts is not None:
            if key in self.counts:
                # 更新 LRU 顺序并增加引用计数
                self.counts.move_to_end(key)
                self.counts[key] += 1
            else:
                # 跟踪器已满时，淘汰最旧的条目
                if len(self.counts) >= self.max_tracker_size:
                    self.counts.popitem(last=False)
                self.counts[key] = 1
        block = self._policy.get(key)
        if block is None:
            return False
        if not block.is_ready:
            return None  # 写入中，调用者应重试
        return True

    def prepare_load(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
    ) -> LoadStoreSpec:
        """
        准备从 CPU 缓存加载 KV 缓存块。

        为每个键查找对应的缓存块，增加引用计数，并返回加载规格。

        Args:
            keys: 需要加载的缓存键集合
            req_context: 请求上下文

        Returns:
            包含块 ID 的 CPULoadStoreSpec，用于指导数据传输
        """
        blocks = []
        for key in keys:
            block = self._policy.get(key)
            assert block is not None, f"Block {key!r} not found in cache"
            assert block.is_ready, f"Block {key!r} is not ready for reading"
            block.ref_cnt += 1
            blocks.append(block)
        return self._get_load_store_spec(keys, blocks)

    def touch(self, keys: Collection[OffloadKey], req_context: ReqContext) -> None:
        """
        触摸缓存块，更新其在缓存策略中的访问记录。

        用于通知缓存策略这些块最近被访问过，影响淘汰优先级。

        Args:
            keys: 需要触摸的缓存键集合
            req_context: 请求上下文
        """
        self._policy.touch(keys)

    def complete_load(
        self, keys: Collection[OffloadKey], req_context: ReqContext
    ) -> None:
        """
        完成从 CPU 缓存的加载操作。

        减少每个块的引用计数，表示该块不再被当前请求使用。

        Args:
            keys: 已完成加载的缓存键集合
            req_context: 请求上下文
        """
        for key in keys:
            block = self._policy.get(key)
            assert block is not None, f"Block {key!r} not found"
            assert block.ref_cnt > 0, f"Block {key!r} ref_cnt is already 0"
            block.ref_cnt -= 1

    def prepare_store(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
    ) -> PrepareStoreOutput | None:
        """
        准备将 KV 缓存块存储到 CPU 缓存。

        执行流程：
        1. 如果启用了引用计数过滤，过滤掉未达到卸载阈值的键
        2. 过滤掉已存储的块
        3. 如果需要淘汰旧块来腾出空间，执行淘汰
        4. 分配新块并插入缓存策略
        5. 返回需要存储的块 ID 和被淘汰的块

        Args:
            keys: 需要存储的缓存键集合
            req_context: 请求上下文

        Returns:
            PrepareStoreOutput 包含需要存储的键、存储规格和被淘汰的键。
            如果无法腾出足够空间则返回 None。
        """
        if self.counts is not None:
            # 仅保留达到卸载阈值的键
            keys = [k for k in keys if self.counts.get(k, 0) >= self.store_threshold]
        # 过滤掉已存储的块
        keys_to_store = [k for k in keys if self._policy.get(k) is None]

        if not keys_to_store:
            return PrepareStoreOutput(
                keys_to_store=[],
                store_spec=self._get_load_store_spec([], []),
                evicted_keys=[],
            )

        # 计算需要淘汰的块数
        num_blocks_to_evict = len(keys_to_store) - self._get_num_free_blocks()

        to_evict: list[OffloadKey] = []
        if num_blocks_to_evict > 0:
            # 保护本次输入中的键不被淘汰：
            # 已经存储的块在本次调用后必须保留在缓存中。
            protected = set(keys)
            evicted = self._policy.evict(num_blocks_to_evict, protected)
            if evicted is None:
                return None
            for key, block in evicted:
                self._free_block(block)
                to_evict.append(key)

        # 记录淘汰事件
        if to_evict and self.events is not None:
            self.events.append(
                OffloadingEvent(
                    keys=to_evict,
                    medium=self.medium,
                    removed=True,
                )
            )

        # 分配新块
        blocks = self._allocate_blocks(keys_to_store)
        assert len(blocks) == len(keys_to_store), (
            "Block pool did not allocate the expected number of blocks"
        )

        # 将新块插入缓存策略
        for key, block in zip(keys_to_store, blocks):
            self._policy.insert(key, block)

        # 构建存储规格
        store_spec = self._get_load_store_spec(keys_to_store, blocks)

        return PrepareStoreOutput(
            keys_to_store=keys_to_store,
            store_spec=store_spec,
            evicted_keys=to_evict,
        )

    def complete_store(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
        success: bool = True,
    ) -> None:
        """
        完成存储操作的回调。

        根据存储结果更新块状态：
        - 成功：将块标记为就绪 (is_ready=True, ref_cnt=0)
        - 失败：移除块并归还到空闲列表

        Args:
            keys: 已完成存储的缓存键集合
            req_context: 请求上下文
            success: 存储是否成功
        """
        stored_keys: list[OffloadKey] = []

        if success:
            for key in keys:
                block = self._policy.get(key)
                if block is not None and not block.is_ready:
                    # 标记为就绪，引用计数清零
                    block.ref_cnt = 0
                    stored_keys.append(key)
        else:
            for key in keys:
                block = self._policy.get(key)
                if block is not None and not block.is_ready:
                    # 存储失败，移除块并归还
                    self._policy.remove(key)
                    self._free_block(block)

        # 记录存储事件
        if stored_keys and self.events is not None:
            self.events.append(
                OffloadingEvent(
                    keys=stored_keys,
                    medium=self.medium,
                    removed=False,
                )
            )

    def reset_cache(self) -> None:
        """
        重置缓存，清除所有块。

        无条件清除所有块。调度器的 _stale_job_threshold 保证了
        complete_load / complete_store 不会在重置前的旧任务上被调用，
        因此不需要延迟清理。调度器还会在新的存储开始前将进行中的
        load 任务 ID 刷新到工作器，防止重用的卸载块 ID 上的跨方向数据竞争。
        """
        self._policy.clear()

        self._free_list.clear()
        self._num_allocated_blocks = 0

    def take_events(self) -> Iterable[OffloadingEvent]:
        """
        获取并清空所有缓存事件。

        用于事件消费者（如 KV 缓存事件处理器）获取自上次调用以来的所有事件。

        Yields:
            缓存事件（新增或淘汰）
        """
        if self.events is not None:
            yield from self.events
            self.events.clear()
