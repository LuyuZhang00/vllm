# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
TieringOffloadingManager: Multi-tier KV cache offloading orchestrator.

This manager coordinates between a CPU primary tier (with direct GPU access)
and zero or more secondary tiers (Storage, Network, etc.) to provide
hierarchical KV cache offloading.

Key Design Principles:
1. Always offload to all tiers — When a block is stored to the primary tier,
   it is cascaded to ALL secondary tiers
2. Primary tier is the gateway — Secondary tiers cannot access GPU memory
   directly; all data flows through the CPU primary tier
3. Staged promotion — Blocks in secondary tiers must be promoted to the
   primary tier before GPU can access them
4. Transparent retry mechanism — Return None from lookup() to signal
   "data is being promoted, try later"
5. ref_cnt as eviction protection — primary.prepare_read() increments ref_cnt,
   protecting blocks from eviction until complete_read() is called
"""
# =============================================================================
# TieringOffloadingManager：多级 KV cache 卸载编排器
#
# 本模块实现了 vLLM v1 的多级 KV cache 卸载管理器，是多级卸载架构的核心。
# 它在 CPU 一级层（直接可被 GPU 访问）和零个或多个二级层（存储、网络等）
# 之间进行协调，提供层级化的 KV cache 卸载能力。
#
# ==================== 核心设计原则 ====================
#
# 1. 全量级联（Always offload to all tiers）：
#    当一个 block 存入一级层后，会被级联到所有二级层。
#    这确保了数据在多个存储层都有副本，提高容错性。
#
# 2. 一级层是网关（Primary tier is the gateway）：
#    二级层不能直接访问 GPU 显存，所有数据流必须经过 CPU 一级层中转。
#    GPU <-> 一级层的传输由继承的 CPUOffloadingManager 处理。
#
# 3. 分步提升（Staged promotion）：
#    二级层中的 block 必须先提升到一级层，然后 GPU 才能访问。
#    提升路径：Secondary -> CPU Primary -> GPU
#
# 4. 透明重试（Transparent retry）：
#    lookup() 返回 None 表示"数据正在提升中，稍后重试"。
#    Scheduler 看到 None 后会将请求延迟处理，避免阻塞。
#
# 5. 引用计数保护（ref_cnt as eviction protection）：
#    primary.prepare_read() 增加 ref_cnt，在异步传输期间保护 block 不被淘汰。
#    传输完成后 complete_read() 减少 ref_cnt，恢复淘汰能力。
#
# ==================== 整体数据流 ====================
#
# 存储路径（Store/Cascade）：
#   GPU 计算完 KV -> prepare_store() 分配一级层 slot
#   -> GPU->CPU DMA 传输 -> complete_store()
#   -> 为每个二级层 prepare_read() 增加一级层 ref_cnt
#   -> submit_store() 提交异步级联到二级层
#   -> get_finished_jobs() 轮询完成 -> complete_read() 释放 ref_cnt
#
# 加载路径（Load/Promotion）：
#   Scheduler 查询 lookup() -> 一级层未命中
#   -> 查询二级层命中 -> _initiate_promotion() 在一级层分配 slot
#   -> _flush_pending_promotions() 批量提交 submit_load() 提升任务
#   -> get_finished_jobs() 轮询完成 -> complete_write() 标记 slot 可用
#   -> prepare_load() 一级层 -> GPU DMA 传输
# =============================================================================

from collections import defaultdict
from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import (
    LoadStoreSpec,
    OffloadingEvent,
    OffloadingManager,
    OffloadKey,
    OffloadPolicy,
    PrepareStoreOutput,
    ReqContext,
    RequestOffloadingContext,
)
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion
from vllm.v1.kv_offload.tiering.base import (
    JobId,
    JobMetadata,
    SecondaryTierManager,
)

logger = init_logger(__name__)


@dataclass
class PendingPromotion:
    """Accumulator for blocks awaiting submit_load() for one (tier, request)."""
    # 中文注释：待提交的提升（promotion）任务累加器。
    #
    # 在一个 engine step 的 lookup 阶段，可能会有多个 block 被判定需要从
    # 同一个二级层提升到一级层。为了避免每个 block 都单独提交一个异步任务，
    # 系统将同一个 (tier, request) 组合下的所有待提升 block 累积在这里，
    # 在 engine step 结束时通过 _flush_pending_promotions() 批量提交。
    #
    # 这种批处理设计减少了异步任务提交的开销，并允许二级层进行更高效的批量 I/O。
    #
    # 【数据结构设计】
    #   - keys 和 block_ids 是一一对应的平行数组。
    #   - keys[i] 标识要提升的逻辑 block（OffloadKey），
    #     block_ids[i] 标识一级层中分配的物理 slot。
    #   - req_context 用于在提交时关联到正确的请求。

    req_context: ReqContext
    # 中文注释：待提升 block 的 OffloadKey 列表，与 block_ids 一一对应。
    keys: list[OffloadKey] = field(default_factory=list)
    # 中文注释：一级层中已分配的物理 block ID 列表，
    # 提升数据将被写入这些 slot。由 primary.prepare_write() 分配。
    block_ids: list[int] = field(default_factory=list)


class CPUPrimaryTierOffloadingManager(CPUOffloadingManager):
    """CPUOffloadingManager with a primary/secondary transfer interface.

    The inherited prepare_store/complete_store/prepare_load/complete_load are the
    GPU-facing OffloadingManager interface. These aliases expose the same operations
    from the secondary tier perspective, where read/write refers to secondary
    accessing primary. This avoids confusion when reading TieringOffloadingManager
    code (e.g. calling prepare_load inside a cascade/store path would be misleading).
    """
    # 中文注释：CPU 一级层卸载管理器，扩展了 CPUOffloadingManager 以支持
    # 与二级层之间的数据传输接口。
    #
    # 该类在基础的 CPU<->GPU 传输接口之外，增加了面向二级层的别名：
    #   - load/store：GPU <-> CPU 一级层 的传输（继承自父类）。
    #   - read/write：二级层 <-> CPU 一级层 的传输（新增别名）。
    #
    # 【为什么需要区分命名？】
    # 在多级卸载架构中，CPU 一级层同时参与两种方向的数据传输：
    #   方向 A：GPU <-> CPU 一级层（通过 load/store 方法，继承自父类）
    #   方向 B：二级层 <-> CPU 一级层（通过 read/write 方法别名）
    #
    # 在级联（store）路径中，会调用"二级层读取一级层数据"的操作，
    # 如果直接叫 prepare_load() 会与"从一级层加载到 GPU"混淆。
    # 因此使用 prepare_read/complete_read 语义更清晰：
    #   - prepare_read：二级层从一级层读取数据（增加 ref_cnt 保护）
    #   - complete_read：二级层读取完毕（释放 ref_cnt）
    #   - prepare_write：二级层向一级层写入数据（分配 slot）
    #   - complete_write：二级层写入完毕（标记 slot 可用）
    #
    # 【核心方法映射关系】
    #   prepare_read  = prepare_load  （二级层从一级层读 -> 增加 ref_cnt 保护）
    #   complete_read = complete_load （读取完成 -> 释放 ref_cnt）
    #   prepare_write = prepare_store （二级层向一级层写 -> 分配 slot）
    #   complete_write= complete_store（写入完成 -> 标记 slot 可用）

    def __init__(
        self,
        num_blocks: int,
        mmap_region: SharedOffloadRegion,
        cache_policy: str = "lru",
        enable_events: bool = False,
    ):
        super().__init__(
            num_blocks=num_blocks,
            cache_policy=cache_policy,  # type: ignore[arg-type]
            enable_events=enable_events,
        )
        # 中文注释：保存 mmap 区域引用，用于创建 memoryview 和清理资源。
        self._mmap_region = mmap_region
        # read/write is for CPU<->secondary transfers,
        # load/store is for CPU<->GPU transfers.
        # These aliases avoid calling prepare_load inside a store path.
        # 中文注释：建立语义别名——将面向二级层的 read/write 操作映射到
        # 父类的 load/store 实现，避免在级联路径中出现语义混淆。
        self.prepare_read = self.prepare_load
        self.complete_read = self.complete_load
        self.prepare_write = self.prepare_store
        self.complete_write = self.complete_store

        # 中文注释：创建一级层 KV cache 的 memoryview。
        # 此 memoryview 将传递给二级层，作为二级层读写一级层数据的唯一通道。
        # 形状为 (num_blocks, row_stride_bytes)，底层是 mmap 共享内存。
        # 二级层通过 primary_kv_view[block_id] 来读写一级层中指定 block 的数据。
        self._kv_memoryview = mmap_region.create_kv_memoryview()

    def get_kv_memoryview(self) -> memoryview:
        """Return the memoryview over the primary tier's KV cache buffer.

        The view has shape (num_blocks, row_stride_bytes) and is backed by the
        SharedOffloadRegion mmap.  Secondary tiers address block *b* as
        ``view[b]``.
        """
        # 中文注释：返回一级层 KV cache 缓冲区的 memoryview 视图。
        # 二级层通过 view[block_id] 来读写一级层中指定 block 的数据。
        return self._kv_memoryview

    def shutdown(self) -> None:
        super().shutdown()
        # 中文注释：释放 memoryview 引用并清理 mmap 共享内存区域。
        self._kv_memoryview.release()
        self._mmap_region.cleanup()


class TieringOffloadingManager(OffloadingManager):
    """
    Orchestrates multi-tier KV cache offloading.

    This manager coordinates between a CPU primary tier (with direct GPU access)
    and zero or more secondary tiers (Storage, Network, etc.) to provide
    hierarchical KV cache offloading.

    Key internal state:
      - Minimal state tracking; relies on secondary tiers to report completion
        via get_finished_jobs()
      - Secondary tiers return JobResult objects containing all necessary
        information
      - job_id_counter: monotonically increasing counter for job IDs
    """
    # 中文注释：多级 KV cache 卸载编排器，是多级卸载架构的核心管理类。
    #
    # 职责：
    #   1. 协调 CPU 一级层和多个二级层之间的数据流转。
    #   2. 管理异步传输任务的生命周期（提交、追踪、完成处理）。
    #   3. 实现透明的"查找-提升-重试"机制。
    #   4. 批量处理待提交的提升请求，优化 I/O 效率。
    #
    # 该类实现了 OffloadingManager 接口，运行在 Scheduler 进程中。
    # Scheduler 通过 lookup/prepare_load/prepare_store 等方法与之交互。

    def __init__(
        self,
        primary_tier: CPUPrimaryTierOffloadingManager,
        secondary_tiers: list[SecondaryTierManager] | None = None,
        enable_events: bool = False,
    ):
        """
        Initialize the TieringOffloadingManager.

        Args:
            primary_tier: The primary tier manager (CPU-based).
            secondary_tiers: List of secondary tier managers (e.g., Storage,
                            Network). Can be None or empty list.
            enable_events: Whether to track offloading events
        """
        # 中文注释：CPU 一级层管理器，处理 GPU <-> CPU 之间的数据传输。
        self.primary_tier: CPUPrimaryTierOffloadingManager = primary_tier
        # 中文注释：二级层管理器列表，可为空。每个二级层独立管理自己的存储。
        self.secondary_tiers = secondary_tiers or []

        # 中文注释：单调递增的任务 ID 计数器，为每个异步传输任务分配唯一标识。
        self._job_id_counter: int = 0
        # 中文注释：卸载事件列表（如 block 存储/移除事件），用于上层监控。
        # 当 enable_events=False 时为 None，表示不收集事件。
        self.events: list[OffloadingEvent] | None = [] if enable_events else None

        # Job tracking: maps job_id to metadata for all in-flight transfers.
        # JobMetadata.is_promotion distinguishes direction:
        #   True:  secondary → primary (promotion)
        #   False: primary → secondary (cascade)
        # 中文注释：所有 in-flight 异步传输任务的追踪字典。
        # key 为 job_id，value 为任务元数据（JobMetadata）。
        # 通过 is_promotion 字段区分传输方向：
        #   True  = 提升（secondary -> primary）
        #   False = 级联（primary -> secondary）
        #
        # 【生命周期管理】
        #   1. 任务提交时：创建 JobMetadata，存入 _transfer_jobs。
        #   2. 任务完成时：从 _transfer_jobs 中取出并移除，执行清理操作。
        #   3. 异常时：任务可能永远不完成，需要考虑超时或清理机制（当前未实现）。
        self._transfer_jobs: dict[JobId, JobMetadata] = {}

        # Pending promotion requests accumulated during lookup() calls; flushed
        # as one batched submit_load() per (tier, request) in take_events().
        # Outer key: tier. Inner key: req_context.req_id — the same ReqContext
        # object is reused for all block lookups of a given request per engine step.
        # 中文注释：待提交的提升请求累加器。
        # 结构：{ 二级层 -> { 请求ID -> PendingPromotion } }
        #
        # 在一个 engine step 的 lookup 阶段，多个 block 可能需要从同一个二级层提升。
        # 为了避免逐个提交，先将它们累积到 _pending_load_submissions 中，
        # 在 engine step 结束时（take_events）批量提交给二级层。
        #
        # 【为什么使用两级嵌套字典？】
        #   - 外层 key 是 SecondaryTierManager 实例，区分不同的二级层。
        #   - 内层 key 是请求 ID，确保同一个请求的多个 block 合并为一个批量任务。
        #   - 这种结构使得 _flush_pending_promotions() 可以按 (tier, request) 为单位
        #     提交批量任务，减少异步 I/O 的固定开销。
        self._pending_load_submissions: dict[
            SecondaryTierManager, dict[str, PendingPromotion]
        ] = {}

        # Gate for once-per-step execution of _maybe_process_finished_jobs().
        # Reset at the end of each step in take_events().
        # 中文注释：每步执行一次的门控标志，确保 _maybe_process_finished_jobs()
        # 在同一个 engine step 内只真正执行一次。在 take_events() 结束时重置。
        #
        # 【为什么要门控？】
        #   在一个 engine step 中，lookup/prepare_load/prepare_store 等多个方法
        #   都可能需要轮询已完成的任务。如果每次都真正轮询，会产生大量重复开销。
        #   通过门控标志，确保同一步内只有第一次调用真正执行轮询，后续调用直接返回。
        self._processed_jobs_this_step: bool = False

        # Per-request set of secondary tiers that requested REQUEST_LEVEL
        # policy. Populated in on_new_request(),
        # cleaned up in on_request_finished().
        # 中文注释：记录每个请求中哪些二级层请求了 REQUEST_LEVEL 策略。
        # REQUEST_LEVEL 策略要求卸载该请求的所有 block（包括 prefix 命中的）。
        # 在 on_new_request() 中填充，在 on_request_finished() 中清理。
        #
        # 【使用场景】
        #   prepare_store() 中会检查此字典，对于 REQUEST_LEVEL 策略的二级层，
        #   即使某些 block 已经在一级层中（prefix cache 命中），也会发起级联操作。
        #   这确保了 REQUEST_LEVEL 策略的二级层能获得请求的完整 KV 上下文。
        self._request_level_tiers: defaultdict[str, set[SecondaryTierManager]] = (
            defaultdict(set)
        )

    def _next_job_id(self) -> JobId:
        """Generate a unique job ID for async transfer tracking."""
        # 中文注释：生成唯一的异步传输任务 ID。单调递增，用于关联提交和完成事件。
        #
        # 【设计考量】
        #   - 使用简单的单调递增计数器，而非 UUID 等复杂方案，因为 ID 只需在
        #     单个 TieringOffloadingManager 实例内唯一即可。
        #   - Job ID 用于在 _transfer_jobs 字典中关联任务元数据和完成结果，
        #     是异步任务生命周期管理的核心标识。
        job_id = self._job_id_counter
        self._job_id_counter += 1
        return job_id

    def _maybe_process_finished_jobs(self):
        """
        Poll secondary tiers for completed jobs (at most once per step).

        Guarded by _processed_jobs_this_step: the first call in an engine step
        does the actual polling; subsequent calls are no-ops. The flag is reset
        in take_events() at the end of each step.
        """
        # 中文注释：尝试轮询已完成的异步传输任务（每步最多执行一次）。
        #
        # 设计动机：在一个 engine step 中，lookup/prepare_load/prepare_store
        # 等多个方法都可能调用此方法。为了避免重复轮询，使用门控标志确保
        # 同一步内只真正执行一次 _process_finished_jobs()。
        #
        # 为什么在多个入口都调用？因为需要在以下时机确保已完成的任务被处理：
        #   - lookup() 调用前：确保刚完成的提升任务被标记为可用。
        #   - prepare_load() 调用前：确保 block 已准备好。
        #   - prepare_store() 调用前：确保一级层的 ref_cnt 是最新的。
        if self._processed_jobs_this_step:
            return
        self._processed_jobs_this_step = True
        self._process_finished_jobs()

    def _process_finished_jobs(self):
        """
        Unconditionally poll all secondary tiers for completed jobs.

        This method:
        1. Calls get_finished_jobs() on each secondary tier
        2. For completed stores (primary→secondary): calls primary.complete_read()
           to decrement ref_cnt
        3. For completed loads (secondary→primary): calls primary.complete_write()
           to make blocks available
        """
        # 中文注释：无条件轮询所有二级层的已完成任务，并执行相应的资源清理。
        #
        # 处理流程：
        #   1. 遍历每个二级层，调用 get_finished_jobs() 获取已完成的任务。
        #   2. 对每个已完成任务，从 _transfer_jobs 中取出元数据。
        #   3. 根据传输方向（is_promotion）执行不同的清理操作：
        #      - 提升完成（secondary -> primary）：调用 complete_write()
        #        使一级层 block 可用，后续可被 GPU 加载。
        #      - 级联完成（primary -> secondary）：调用 complete_read()
        #        减少一级层 block 的 ref_cnt，恢复淘汰能力。
        for i, tier in enumerate(self.secondary_tiers):
            for completed_job in tier.get_finished_jobs():
                job_id = completed_job.job_id
                # 中文注释：从追踪字典中取出并移除该任务的元数据。
                job_metadata = self._transfer_jobs.pop(job_id, None)
                assert job_metadata is not None, (
                    f"Finished job_id {job_id} from tier #{i}"
                    f" ({tier.tier_type}) not in _transfer_jobs"
                )

                if job_metadata.is_promotion:
                    # secondary→primary transfer (promotion) completed.
                    # Make blocks available in primary tier.
                    # 中文注释：提升任务完成——二级层数据已写入一级层 slot。
                    # 调用 complete_write() 使这些 slot 标记为可用，
                    # 后续 prepare_load() 可以将数据从一级层加载到 GPU。
                    self.primary_tier.complete_write(
                        job_metadata.keys,
                        job_metadata.req_context,
                        completed_job.success,
                    )
                else:
                    # primary→secondary transfer completed.
                    # Decrement ref_cnt on primary blocks.
                    # 中文注释：级联任务完成——一级层数据已复制到二级层。
                    # 调用 complete_read() 减少一级层 block 的引用计数，
                    # 使这些 block 重新可以被淘汰（当一级层空间不足时）。
                    self.primary_tier.complete_read(
                        job_metadata.keys, job_metadata.req_context
                    )

    def lookup(self, key: OffloadKey, req_context: ReqContext) -> bool | None:
        """
        Check whether a single block is offloaded and ready.

        Algorithm:
            1. Process any completed async jobs first.
            2. Query primary tier — short-circuit on hit or in-flight.
            3. On primary miss, query secondary tiers — stop on first
               hit and initiate promotion.

        Args:
            key: Block hash to look up.
            req_context: Per-request context.

        Returns:
            True  — block is ready in the primary tier.
            None  — block found but not yet ready (primary in-flight,
                    promotion started, or a secondary tier is busy).
            False — block not found in any tier, or primary is full
                    and cannot accept a promotion.
        """
        # 中文注释：查询单个 block 是否已卸载且可用。
        #
        # 这是 Scheduler 查询 KV cache 命中的核心入口之一。
        # 算法按优先级依次查询，采用短路策略：
        #
        # 步骤 1：先轮询已完成的异步任务，确保状态最新。
        # 步骤 2：查询一级层（CPU）——
        #   - 返回 True：block 在一级层且可用，直接返回 True。
        #   - 返回 None：block 正在一级层传输中（in-flight），返回 None 让 Scheduler 重试。
        #   - 返回 False：一级层未命中，继续查二级层。
        # 步骤 3：依次查询每个二级层——
        #   - 某二级层返回 True：命中，立即发起提升（promote），返回 None。
        #   - 某二级层返回 None：正在传输中，标记 any_none，继续查其他二级层。
        #   - 返回 False：未找到，继续查下一个二级层。
        #
        # 返回值对 Scheduler 的影响：
        #   True  -> block 可以立即用于 GPU 计算。
        #   None  -> block 正在传输中，Scheduler 会将该请求延迟到下一步处理。
        #   False -> block 未在任何层找到，需要从 GPU 重新计算。
        #
        # 【为什么先调用 _maybe_process_finished_jobs()？】
        #   在查询之前，需要确保上一步发起的异步传输任务的完成事件已被处理。
        #   否则，一个已经完成的提升任务可能还没被标记为"可用"，导致误判为未命中。
        self._maybe_process_finished_jobs()

        primary_hit = self.primary_tier.lookup(key, req_context)
        if primary_hit is True:
            return True
        if primary_hit is None:
            return None

        # 中文注释：一级层未命中，依次查询二级层。
        any_none = False
        for tier in self.secondary_tiers:
            result = tier.lookup(key, req_context)
            if result is True:
                # 中文注释：在某个二级层找到该 block，发起提升操作。
                # 如果一级层已满无法分配 slot，返回 False。
                if not self._initiate_promotion(tier, key, req_context):
                    return False  # primary full, block unavailable
                return None  # promotion started, retry later
            if result is None:
                any_none = True

        # 中文注释：如果有二级层返回 None（正在传输中），返回 None 让 Scheduler 重试。
        # 否则所有层都未找到，返回 False。
        if any_none:
            return None
        return False

    def _initiate_promotion(
        self,
        tier: SecondaryTierManager,
        key: OffloadKey,
        req_context: ReqContext,
    ) -> bool:
        """
        Queue a block for promotion from a secondary tier to the primary tier.

        Allocates space in the primary tier immediately (sets ref_cnt=-1 so
        subsequent lookups within the same step see the slot as in-flight),
        then defers the actual submit_load() call to _flush_pending_promotions()
        so all blocks queued during one engine step are submitted as a single
        batched job.

        Args:
            tier: The secondary tier to promote from
            key: Block to promote
            req_context: Per-request context forwarded to primary.prepare_write().

        Returns:
            True if promotion was initiated, False if primary tier is full.
        """
        # 中文注释：发起从二级层到一级层的提升（promotion）操作。
        #
        # 关键设计——立即分配 + 延迟提交：
        #   1. 立即在一级层分配 slot（调用 prepare_write）。
        #      分配后该 slot 的 ref_cnt 设为 -1（in-flight 状态），
        #      后续同一 step 内对该 key 的 lookup() 会返回 None，
        #      从而防止重复发起提升。
        #   2. 将实际的 submit_load() 推迟到 engine step 结束时批量提交。
        #      这样同一个 (tier, request) 组合下的所有 block 会被合并为
        #      一个批量异步任务，减少 I/O 开销。
        #
        # Returns:
        #   True = 提升已发起（已排队等待提交）。
        #   False = 一级层已满，无法分配 slot，该 block 不可用。

        # Allocate space in primary tier for promoted block.
        # Must happen immediately so primary.lookup() returns None (in-flight)
        # for this key on any subsequent lookup() call within the same step,
        # preventing duplicate promotion attempts.
        # 中文注释：立即在一级层为待提升的 block 分配 slot。
        primary_write_result = self.primary_tier.prepare_write([key], req_context)

        if primary_write_result is None:
            # Primary tier is full; caller should treat the block as unavailable
            # rather than retrying indefinitely.
            # 中文注释：一级层已满，无法分配 slot。调用方应将该 block 视为不可用。
            return False

        store_spec = primary_write_result.store_spec
        assert isinstance(store_spec, CPULoadStoreSpec)
        # Defer submit_load to take_events(). Group by (tier, request) so each
        # request's blocks are submitted as one batched job per tier.
        # 中文注释：将待提升的 block 累积到 _pending_load_submissions 中。
        # 按 (二级层, 请求) 分组，同一组的 block 将在 step 结束时批量提交。
        tier_pending = self._pending_load_submissions.setdefault(tier, {})
        ctx_id = req_context.req_id
        if ctx_id not in tier_pending:
            tier_pending[ctx_id] = PendingPromotion(
                keys=[], block_ids=[], req_context=req_context
            )
        entry = tier_pending[ctx_id]
        entry.keys.extend(primary_write_result.keys_to_store)
        entry.block_ids.extend(store_spec.block_ids)
        return True

    def _flush_pending_promotions(self) -> None:
        """Submit one batched submit_load() per (tier, request).

        Called from take_events() at the end of each engine step, flushing
        all promotion requests deferred during lookup().
        """
        # 中文注释：将所有累积的待提升请求批量提交给对应的二级层。
        #
        # 此方法在每个 engine step 结束时被 take_events() 调用。
        # 它遍历 _pending_load_submissions，为每个 (二级层, 请求) 组合
        # 创建一个 JobMetadata 并调用二级层的 submit_load()。
        #
        # 为什么需要批量提交？
        #   1. 减少异步任务提交的固定开销。
        #   2. 允许二级层进行批量 I/O 优化（如合并写入）。
        #   3. 保持同一个请求的 block 在同一个任务中，便于追踪。
        if not self._pending_load_submissions:
            return

        for tier, pending_by_ctx in self._pending_load_submissions.items():
            for entry in pending_by_ctx.values():
                job_id = self._next_job_id()
                job_metadata = JobMetadata(
                    job_id=job_id,
                    keys=entry.keys,
                    block_ids=np.array(entry.block_ids, dtype=np.int64),
                    is_promotion=True,
                    req_context=entry.req_context,
                )
                # 中文注释：将任务记录到追踪字典中，用于后续完成处理。
                self._transfer_jobs[job_id] = job_metadata
                # 中文注释：提交异步提升任务给二级层。
                tier.submit_load(job_metadata)

        # 中文注释：清空待提交队列。
        self._pending_load_submissions.clear()

    def prepare_load(
        self, keys: Collection[OffloadKey], req_context: ReqContext
    ) -> LoadStoreSpec:
        """
        Prepare blocks to be loaded from primary tier to GPU.

        CRITICAL: This method calls _maybe_process_finished_jobs() FIRST to ensure
        that any completed promotions have been finalized and blocks are ready.

        This increments ref_cnt on the blocks in the primary tier, protecting
        them from eviction during the transfer.

        Args:
            keys: Blocks to prepare for loading.
            req_context: Per-request context.

        Returns:
            LoadStoreSpec for reading from primary tier.
        """
        # 中文注释：准备从一级层加载 block 到 GPU。
        #
        # 此方法是 GPU 加载路径的入口。流程：
        #   1. 先轮询已完成的异步任务，确保刚完成的提升 block 已标记为可用。
        #   2. 调用一级层的 prepare_load()，增加 ref_cnt 保护 block 不被淘汰。
        #   3. 返回 LoadStoreSpec，Worker 据此执行 GPU DMA 传输。
        #
        # 为什么必须先调用 _maybe_process_finished_jobs()？
        # 如果上一步发起了一个提升任务且该任务已完成，但还没来得及处理完成事件，
        # 那么一级层的 block 状态可能还不是"可用"。先轮询确保状态一致。

        # Process completed promotions to ensure blocks are ready
        self._maybe_process_finished_jobs()

        return self.primary_tier.prepare_load(keys, req_context)

    def touch(self, keys: Collection[OffloadKey], req_context: ReqContext):
        """
        Mark blocks as recently used in all tiers.

        Args:
            keys: Blocks to mark as recently used.
            req_context: Per-request context.
        """
        # 中文注释：在所有存储层中标记指定 block 为"最近使用"。
        # 当 GPU prefix cache 命中某个 block 时，虽然不需要从 offload 层加载，
        # 但仍需更新该 block 在各层的访问时间，防止被淘汰。
        self.primary_tier.touch(keys, req_context)
        for tier in self.secondary_tiers:
            tier.touch(keys, req_context)

    def complete_load(self, keys: Collection[OffloadKey], req_context: ReqContext):
        """
        Mark blocks as done loading from primary tier to GPU.

        This decrements ref_cnt on the blocks in the primary tier, allowing
        them to be evicted again.

        Args:
            keys: Blocks that finished loading.
            req_context: Per-request context.
        """
        # 中文注释：标记 block 从一级层到 GPU 的加载完成。
        # 减少一级层 block 的 ref_cnt，使其重新可以被淘汰。
        # 调用时机：Worker 完成 GPU DMA 传输后。
        self.primary_tier.complete_load(keys, req_context)

    def prepare_store(
        self, keys: Collection[OffloadKey], req_context: ReqContext
    ) -> PrepareStoreOutput | None:
        """
        Prepare blocks to be stored from GPU to primary tier.

        CRITICAL: This method calls _maybe_process_finished_jobs() FIRST to ensure
        that any completed async transfers have their ref_cnt decremented
        before the primary tier makes eviction decisions.

        For request-level tiers, blocks already present in the primary tier
        are immediately cascaded via submit_store().

        Args:
            keys: Blocks to prepare for storing.
            req_context: Per-request context.

        Returns:
            PrepareStoreOutput describing where to store blocks and what was
            evicted, or None if store cannot proceed.
        """
        # 中文注释：准备将 block 从 GPU 存储到一级层。
        #
        # 存储流程分为三步：
        #   步骤 1：轮询已完成的异步任务，确保一级层的 ref_cnt 是最新的。
        #           这样一级层在做淘汰决策时才能看到准确的引用状态。
        #   步骤 2：在一级层中为新 block 分配 slot（仅处理不在一级层中的 block）。
        #           此时还不进行级联——级联在 complete_store() 中进行，
        #           即 GPU->CPU 传输完成后才触发到二级层的级联。
        #   步骤 3：对于 REQUEST_LEVEL 策略的二级层，对已在一级层中的 block
        #           立即发起级联（不需要等 GPU->CPU 传输完成，因为数据已在一级层）。

        # Step 1: Poll for completed async jobs FIRST
        # This decrements ref_cnt on primary blocks that have been
        # successfully transferred to secondary tiers.
        self._maybe_process_finished_jobs()

        # Step 2: Store to primary tier (new blocks only).
        # Cascading of these newly-stored blocks to ALL secondary tiers
        # happens later in complete_store(), after the GPU→Primary transfer
        # completes.
        # 中文注释：在一级层为新 block 分配 slot，返回需要存储的 key 列表
        # 和存储规格（LoadStoreSpec）。已被一级层缓存的 block 不会出现在结果中。
        primary_result = self.primary_tier.prepare_store(keys, req_context)

        if primary_result is None:
            return None

        # Step 3: For request-level tiers, cascade blocks already in primary
        # 中文注释：对于 REQUEST_LEVEL 策略的二级层，对已在一级层中的 block
        # 立即发起级联。这些 block 可能来自之前的请求（prefix cache 命中），
        # REQUEST_LEVEL 策略要求即使不是新计算的 block 也要卸载到二级层。
        request_level_tiers = self._request_level_tiers.get(req_context.req_id)
        if request_level_tiers is not None:
            keys_to_store_set = set(primary_result.keys_to_store)
            # 中文注释：找出已在一级层中（不需要从 GPU 重新存储）的 block。
            keys_already_in_primary = tuple(
                k for k in keys if k not in keys_to_store_set
            )
            if keys_already_in_primary:
                self._cascade_existing_blocks_to_request_level_tiers(
                    keys_already_in_primary, req_context, request_level_tiers
                )

        return primary_result

    def _cascade_existing_blocks_to_request_level_tiers(
        self,
        keys: Sequence[OffloadKey],
        req_context: ReqContext,
        request_level_tiers: set[SecondaryTierManager],
    ) -> None:
        """
        For tiers that requested request-level policy, submit_store() for
        blocks that are already present in the primary tier.
        """
        # 中文注释：对 REQUEST_LEVEL 策略的二级层，级联已在一级层中的 block。
        #
        # 场景：某个 block 是之前其他请求计算的，已存在于一级层（prefix cache 命中）。
        # 但当前请求的某个二级层要求 REQUEST_LEVEL（卸载所有 block），
        # 因此需要将这些"旧 block"也级联到该二级层。
        #
        # 流程：
        #   1. 过滤出一级层中已就绪的 block（排除正在传输中的）。
        #   2. 对每个请求级二级层，调用 prepare_read() 增加 ref_cnt。
        #   3. 提交异步级联任务。

        # Filter out keys that are not ready in primary (e.g. in-flight)
        ready_keys = tuple(
            k for k in keys if self.primary_tier.lookup(k, req_context) is True
        )
        if not ready_keys:
            return

        for tier in request_level_tiers:
            # 中文注释：调用 prepare_read() 为这些 block 增加 ref_cnt，
            # 保护它们在异步传输期间不被淘汰。
            primary_blocks_spec = self.primary_tier.prepare_read(
                ready_keys, req_context
            )

            job_id = self._next_job_id()
            assert isinstance(primary_blocks_spec, CPULoadStoreSpec)
            job_metadata = JobMetadata(
                job_id=job_id,
                keys=ready_keys,
                block_ids=primary_blocks_spec.block_ids,
                is_promotion=False,
                req_context=req_context,
            )
            self._transfer_jobs[job_id] = job_metadata
            tier.submit_store(job_metadata)

    def complete_store(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
        success: bool = True,
    ):
        """
        Mark blocks as done storing from GPU to primary tier.

        This is where secondary tier cascading happens — after blocks are
        confirmed to be in the primary tier, they are cascaded to ALL
        secondary tiers.

        For each secondary tier:
        1. Call primary.prepare_read() to get LoadStoreSpec AND increment
           ref_cnt (protecting blocks during async transfer)
        2. Call tier.submit_store() to start async transfer: primary→secondary
        3. Track the job in _store_jobs dictionary

        Args:
            keys: Blocks that finished storing.
            success: Whether the GPU→primary transfer succeeded.
            req_context: Per-request context forwarded to primary.prepare_read().
        """
        # 中文注释：标记 block 从 GPU 到一级层的存储完成，并触发级联到所有二级层。
        #
        # 【完整流程】
        #   步骤 1：完成一级层的存储操作，使 block 在一级层中变为"可加载"状态。
        #   步骤 2：如果 GPU->一级层传输失败，直接返回，不进行级联。
        #   步骤 3：对每个二级层：
        #     a. 调用 primary.prepare_read() 为 block 增加 ref_cnt，防止在
        #        异步传输期间被淘汰。同时获取一级层的 LoadStoreSpec。
        #     b. 创建 JobMetadata，记录传输方向为 is_promotion=False（级联）。
        #     c. 调用 tier.submit_store() 提交异步级联任务。
        #     d. 将任务记录到 _transfer_jobs 中用于后续追踪。
        #
        # 【为什么级联到所有二级层？】
        #   这是"全量级联"设计原则的体现：数据在多个存储层都有副本，
        #   提高容错性。即使某个二级层故障，其他层仍有备份。

        # Step 1: Complete store in primary tier (makes blocks loadable)
        self.primary_tier.complete_store(keys, req_context, success)

        if not success:
            # If GPU→Primary transfer failed, don't cascade to secondary tiers
            return

        # Step 2: Cascade to ALL secondary tiers
        # For each secondary tier, call primary.prepare_read() to get the
        # LoadStoreSpec AND to increment ref_cnt (protecting blocks from
        # eviction during the async transfer). One prepare_read() call per
        # secondary tier.
        for tier in self.secondary_tiers:
            primary_blocks_spec = self.primary_tier.prepare_read(keys, req_context)

            # Submit async store job: primary→secondary
            job_id = self._next_job_id()

            # Track this store job
            assert isinstance(primary_blocks_spec, CPULoadStoreSpec)
            job_metadata = JobMetadata(
                job_id=job_id,
                keys=keys,
                block_ids=primary_blocks_spec.block_ids,
                is_promotion=False,
                req_context=req_context,
            )
            self._transfer_jobs[job_id] = job_metadata

            tier.submit_store(job_metadata)

        # Note: The async transfers are now in flight. Their completion is
        # tracked via get_finished_jobs() / _maybe_process_finished_jobs().

    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext:
        """
        Query each secondary tier for its offload policy preference.

        Returns REQUEST_LEVEL if ANY secondary tier wants request-level.
        Only stores REQUEST_LEVEL tier decisions for use in prepare_store.
        """
        # 中文注释：新请求到达时，查询每个二级层的卸载策略偏好。
        #
        # 【策略决策逻辑】
        #   1. 遍历所有二级层，调用 on_new_request() 获取各自的策略偏好。
        #   2. 如果任意二级层请求了 REQUEST_LEVEL，则该请求的整体策略为 REQUEST_LEVEL。
        #   3. 只有 REQUEST_LEVEL 的二级层才会被记录到 _request_level_tiers 中，
        #      因为 BLOCK_LEVEL 策略的二级层只需要处理新计算的 block（标准级联即可），
        #      而 REQUEST_LEVEL 策略的二级层需要额外处理 prefix cache 命中的 block。
        #
        # 【为什么采用"任意一个 REQUEST_LEVEL 则全部 REQUEST_LEVEL"的策略？】
        #   因为 REQUEST_LEVEL 策略要求卸载该请求的所有 block（包括 prefix 命中的），
        #   这个需求一旦存在，就必须在 prepare_store() 中为这些 block 额外发起级联。
        #   即使其他二级层是 BLOCK_LEVEL，也不会受影响（它们在 complete_store() 中
        #   的标准级联流程会自动处理新计算的 block）。
        for tier in self.secondary_tiers:
            tier_ctx = tier.on_new_request(req_context)
            if tier_ctx.policy == OffloadPolicy.REQUEST_LEVEL:
                self._request_level_tiers[req_context.req_id].add(tier)

        policy = (
            OffloadPolicy.REQUEST_LEVEL
            if req_context.req_id in self._request_level_tiers
            else OffloadPolicy.BLOCK_LEVEL
        )
        return RequestOffloadingContext(policy=policy)

    def on_request_finished(self, req_context: ReqContext) -> None:
        # 中文注释：请求完成时的清理回调。
        #
        # 【清理内容】
        #   1. 通知一级层清理该请求相关的临时状态（如临时引用计数等）。
        #   2. 通知所有二级层清理该请求相关的临时状态。
        #   3. 从 _request_level_tiers 中移除该请求的记录，避免内存泄漏。
        #
        # 【调用时机】
        #   Scheduler 在请求完成（生成结束 token 或被中止）后调用此方法。
        self.primary_tier.on_request_finished(req_context)
        for tier in self.secondary_tiers:
            tier.on_request_finished(req_context)
        self._request_level_tiers.pop(req_context.req_id, None)

    def take_events(self) -> Iterable[OffloadingEvent]:
        """
        End-of-step hook: flush deferred work, yield events, reset per-step state.

        Called once per engine step from Scheduler.update_from_output() →
        connector.take_events(). Ensures _maybe_process_finished_jobs() has run
        at least once this step, flushes pending promotions, yields collected
        events, and resets the per-step flag.

        Yields:
            New OffloadingEvents collected since the last call.
        """
        # TODO: Move _flush_pending_promotions() to a dedicated end_of_batch()
        # hook once one exists. For now, take_events() serves as the flush
        # point under the assumption that it is called at the end of each
        # engine step (Scheduler.update_from_output() → connector.take_events()).
        # When the dedicated hook is added, update tests that rely on
        # take_events() to signal end of step.

        # 中文注释：engine step 结束时的清理钩子，完成以下三项任务：
        #
        # 【任务 1：轮询已完成的异步任务】
        #   确保本步中所有已完成的传输任务都被处理，资源被正确释放。
        self._maybe_process_finished_jobs()

        # 【任务 2：批量提交累积的提升请求】
        #   将 lookup() 阶段累积的 PendingPromotion 批量提交给二级层。
        #   这是整个 engine step 中提升请求的实际提交时机。
        self._flush_pending_promotions()

        # 【任务 3：重置门控标志】
        #   允许下一步的 _maybe_process_finished_jobs() 真正执行。
        # Reset the per-step gate so next step's first call does real work.
        self._processed_jobs_this_step = False

        # 【任务 4：向上层返回卸载事件】
        #   包括二级层的事件和一级层的事件。
        if self.events is not None:
            yield from self.events
            self.events.clear()

        yield from self.primary_tier.take_events()

    def shutdown(self) -> None:
        """Shutdown all tiers and release resources."""
        # 中文注释：关闭所有存储层并释放资源。
        #
        # 【关闭顺序】
        #   1. 先关闭所有二级层（释放后台线程、文件句柄、网络连接等）。
        #   2. 再关闭一级层（释放 mmap 共享内存等）。
        #   先关闭二级层是因为二级层可能依赖一级层的 memoryview，
        #   确保二级层先停止使用该 memoryview 后再释放它。
        for tier in self.secondary_tiers:
            tier.shutdown()
        self.primary_tier.shutdown()
