# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler-side manager for SimpleCPUOffloadConnector.

调度器端管理器，负责管理KV缓存的CPU卸载操作。
主要职责：
1. 管理CPU块池，跟踪哪些块已缓存到CPU
2. 决策哪些块需要从CPU加载到GPU
3. 决策哪些块需要从GPU卸载到CPU
4. 处理异步传输的完成事件
5. 维护请求级别的卸载状态
"""

import contextlib
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from vllm.config import VllmConfig
from vllm.distributed.kv_events import KVCacheEvent
from vllm.distributed.kv_transfer.kv_connector.utils import yield_req_data
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_coordinator import (
    KVCacheCoordinator,
    get_kv_cache_coordinator,
)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    MambaSpec,
    SlidingWindowSpec,
)
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.simple_kv_offload.metadata import (
    SimpleCPUOffloadMetadata,
    SimpleCPUOffloadWorkerMetadata,
)

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.kv_cache_utils import KVCacheBlock
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)


@dataclass
class TransferMeta:
    """传输元数据，记录GPU块ID和CPU块ID的映射关系。

    用于跟踪一次批量传输操作中的所有块对。
    每个位置i对应一个GPU块到CPU块的映射：gpu_block_ids[i] -> cpu_block_ids[i]
    """
    gpu_block_ids: list[int]  # GPU端的块ID列表
    cpu_block_ids: list[int]  # CPU端的块ID列表


@dataclass
class LoadRequestState:
    """加载请求状态，跟踪单个请求的CPU->GPU加载进度。

    每个请求在加载过程中都会创建一个LoadRequestState实例，
    用于记录该请求的加载事件和完成状态。
    """
    request: "Request"  # 关联的请求对象
    transfer_meta: TransferMeta  # 传输元数据，包含GPU/CPU块ID映射
    load_event: int | None = None  # 加载事件索引，None表示尚未分配事件
    finished: bool = False  # 是否已完成加载


# NOTE: This per-request state is only used in eager mode.
# 注意：这个请求级别的状态只在急切模式下使用。
@dataclass
class StoreRequestState:
    """存储请求状态，跟踪单个请求的GPU->CPU存储进度（仅急切模式）。

    在急切模式下，每个请求需要跟踪其所有块的存储状态，
    包括已累积的块ID、已存储的块数，以及相关的存储事件。
    """
    request: "Request"  # 关联的请求对象
    # Accumulated block IDs from scheduler_output via yield_req_data.
    # 通过yield_req_data从scheduler_output累积的块ID，按组组织
    block_ids: tuple[list[int], ...]
    # Per-group cursors tracking how many blocks have been stored/skipped.
    # 每组的游标，跟踪已存储/跳过的块数量
    num_stored_blocks: list[int]
    store_events: set[int] = field(default_factory=set)  # 与此请求关联的存储事件集合
    finished: bool = False  # 请求是否已完成


class SimpleCPUOffloadScheduler:
    """Scheduler-side manager for CPU offloading.

    CPU卸载的调度器端管理器。
    负责在调度器层面管理KV缓存的CPU卸载，包括：
    1. 维护CPU块池和缓存映射
    2. 处理缓存命中检测（从CPU缓存中查找已有的块）
    3. 构建传输元数据，指导工作节点执行实际的数据传输
    4. 处理传输完成事件，更新缓存状态
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: "KVCacheConfig | None",
        cpu_capacity_bytes: int,
        scheduler_block_size: int,
        hash_block_size: int,
        lazy_offload: bool = False,
    ):
        """
        初始化CPU卸载调度器。

        参数说明：
        - vllm_config: vLLM全局配置
        - kv_cache_config: GPU KV缓存配置
        - cpu_capacity_bytes: CPU内存容量（字节）
        - scheduler_block_size: 调度器使用的块大小（可能是多个hash_block的LCM）
        - hash_block_size: 哈希块大小，用于前缀缓存匹配
        - lazy_offload: 是否使用懒惰卸载模式（True=懒惰，False=急切）
        """
        self.vllm_config = vllm_config
        self.kv_cache_config = kv_cache_config
        # 是否启用KV缓存事件（用于监控和调试）
        self.enable_kv_cache_events = (
            vllm_config.kv_events_config is not None
            and vllm_config.kv_events_config.enable_kv_cache_events
        )
        self.block_size = scheduler_block_size
        self.hash_block_size = hash_block_size
        # 确保调度器块大小是哈希块大小的整数倍
        assert self.block_size % self.hash_block_size == 0
        # Derive a CPU KVCacheConfig from the GPU config and build a coordinator
        # 从GPU配置派生CPU KV缓存配置，并构建协调器
        assert kv_cache_config is not None
        self.cpu_kv_cache_config = self._derive_cpu_config(
            kv_cache_config, cpu_capacity_bytes
        )
        self.num_cpu_blocks = self.cpu_kv_cache_config.num_blocks
        # Find the full attention kv group for prefix cache matching.
        # 查找全注意力KV组，用于前缀缓存匹配
        self.fa_gidx = -1
        for g_idx, g in enumerate(self.cpu_kv_cache_config.kv_cache_groups):
            if isinstance(g.kv_cache_spec, FullAttentionSpec):
                self.fa_gidx = g_idx
                break
        assert 0 <= self.fa_gidx < len(self.cpu_kv_cache_config.kv_cache_groups)
        # FA group's own block_size; divides scheduler_block_size (the LCM)
        # but is NOT assumed to equal it.
        # FA组自己的block_size；能整除scheduler_block_size（LCM），
        # 但不一定等于它
        self.fa_block_size: int = self.cpu_kv_cache_config.kv_cache_groups[
            self.fa_gidx
        ].kv_cache_spec.block_size
        assert self.block_size % self.fa_block_size == 0

        logger.info(
            "SimpleCPUOffloadScheduler: Allocating %d CPU blocks (%.2f GB, mode=%s)",
            self.num_cpu_blocks,
            cpu_capacity_bytes / (1024**3),
            "lazy" if lazy_offload else "eager",
        )

        # TODO (yifan): maybe need to enable kv_cache_events and metrics_collector here.
        dcp_world_size = vllm_config.parallel_config.decode_context_parallel_size
        pcp_world_size = vllm_config.parallel_config.prefill_context_parallel_size
        assert dcp_world_size == 1 and pcp_world_size == 1
        # 创建CPU端的KV缓存协调器，用于管理CPU块池和缓存查找
        self.cpu_coordinator: KVCacheCoordinator = get_kv_cache_coordinator(
            kv_cache_config=self.cpu_kv_cache_config,
            max_model_len=vllm_config.model_config.max_model_len,
            max_num_batched_tokens=(
                vllm_config.scheduler_config.max_num_batched_tokens
            ),
            use_eagle=False,
            enable_caching=True,
            enable_kv_cache_events=self.enable_kv_cache_events,
            dcp_world_size=dcp_world_size,
            pcp_world_size=pcp_world_size,
            hash_block_size=self.hash_block_size,
        )
        self.cpu_block_pool: BlockPool = self.cpu_coordinator.block_pool

        # GPU block pool reference - bound after scheduler builds kv_cache_manager
        # GPU块池引用 - 在调度器构建kv_cache_manager后绑定
        self._gpu_block_pool: BlockPool | None = None

        # Load metadata
        # 加载元数据：记录所有正在加载的请求
        self._reqs_to_load: dict[str, LoadRequestState] = {}
        # Inverse map: load_event_idx -> req_ids. Keyed by load_event_idx because
        # the worker reports completions by event index, not request id.
        # 反向映射：load_event_idx -> req_ids。以load_event_idx为键，因为
        # 工作节点通过事件索引（而非请求ID）报告完成状态
        self._load_event_to_reqs: dict[int, list[str]] = {}

        # Pending (cpu_hit_blocks, hit_length) tuples from find_longest_cache_hit,
        # kept pinned via touch() while awaiting update_state_after_alloc().
        # 待处理的(cpu_hit_blocks, hit_length)元组，来自find_longest_cache_hit，
        # 通过touch()保持固定状态，等待update_state_after_alloc()消费
        self._pending_cpu_hits: dict[
            str, tuple[tuple[list[KVCacheBlock], ...], int]
        ] = {}

        # Store metadata
        # 存储元数据
        self._lazy_mode = lazy_offload  # 是否使用懒惰模式
        # Lazy mode: use a cursor to track the last scanned block in the GPU free queue.
        # 懒惰模式：使用游标跟踪GPU空闲队列中最后扫描的块
        self._cursor: KVCacheBlock | None = None
        if self._lazy_mode:
            # 懒惰模式下计算目标空闲块数，用于决定何时触发卸载
            self._target_free = self._estimate_lazy_target_blocks(
                kv_cache_config,
                vllm_config.scheduler_config.max_num_batched_tokens,
            )
        else:
            self._target_free = 0
        # 存储事件到传输元数据的映射
        self._store_event_to_blocks: dict[int, TransferMeta] = {}
        # Eager mode only
        # 仅急切模式：请求到存储状态的映射
        self._reqs_to_store: dict[str, StoreRequestState] = {}
        # 存储事件到请求ID的映射
        self._store_event_to_reqs: dict[int, list[str]] = {}
        # 正在传输中的GPU块集合，防止重复调度
        self._in_flight_store_gpu_blocks: set[int] = set()

        # Event counters
        # 事件计数器，用于生成唯一的事件索引
        self._load_event_counter: int = 0
        self._store_event_counter: int = 0

        # For TP/PP: track partial store completions across steps.
        # Events must be reported by all world_size workers before considered complete.
        # 用于TP/PP：跨步骤跟踪部分存储完成。
        # 事件必须被所有world_size个工作节点报告后才算完成。
        self._expected_worker_count = vllm_config.parallel_config.world_size
        self._store_event_pending_counts: dict[int, int] = {}

    @staticmethod
    def _derive_cpu_config(
        gpu_config: "KVCacheConfig", cpu_capacity_bytes: int
    ) -> "KVCacheConfig":
        """Derive a CPU KVCacheConfig from the GPU config.
        Same kv_cache_groups, num_blocks scaled by CPU/GPU memory ratio.

        从GPU配置派生CPU KV缓存配置。
        保持相同的kv_cache_groups结构，但根据CPU/GPU内存比例缩放块数量。

        原理：
        1. 计算GPU总内存和块数
        2. 根据CPU容量按比例计算CPU块数
        3. 创建对应的CPU KV缓存张量配置
        """
        # Import here to avoid potential circular imports
        from vllm.v1.kv_cache_interface import KVCacheConfig as KVCacheConfigCls
        from vllm.v1.kv_cache_interface import KVCacheTensor

        assert len(gpu_config.kv_cache_tensors) > 0

        gpu_total_bytes = sum(t.size for t in gpu_config.kv_cache_tensors)
        num_gpu_blocks = gpu_config.num_blocks
        # 按比例计算CPU块数：CPU块数 = GPU块数 * CPU容量 / GPU总容量
        num_cpu_blocks = max(1, num_gpu_blocks * cpu_capacity_bytes // gpu_total_bytes)
        # Create CPU kv_cache_tensors mirroring GPU by scaling size proportionally.
        # 创建CPU KV缓存张量，镜像GPU配置但按比例缩放大小
        cpu_tensors = [
            KVCacheTensor(
                size=t.size // num_gpu_blocks * num_cpu_blocks,
                shared_by=list(t.shared_by),
            )
            for t in gpu_config.kv_cache_tensors
        ]

        return KVCacheConfigCls(
            num_blocks=num_cpu_blocks,
            kv_cache_tensors=cpu_tensors,
            kv_cache_groups=gpu_config.kv_cache_groups,
        )

    @staticmethod
    def _estimate_lazy_target_blocks(
        kv_cache_config: "KVCacheConfig", max_num_batched_tokens: int
    ) -> int:
        """GPU blocks to keep available (free/offloaded) per step in lazy mode.

        估算懒惰模式下每步需要保持可用的GPU块数（空闲或已卸载的块）。

        计算逻辑：
        1. 对于Mamba层：固定需要2个块
        2. 对于滑动窗口层：需要 ceiling(滑动窗口大小/块大小) + 1 个块
        3. 对于全注意力层：需要 ceiling(最大批处理token数/块大小) 个块
        4. 乘以水位系数（默认2.0）以留出余量，避免GPU块耗尽
        """
        WATERMARK_RATIO = 1.0  # Reserve larger space to avoid running out of GPU blocks
        target = 0
        for g in kv_cache_config.kv_cache_groups:
            spec = g.kv_cache_spec
            if isinstance(spec, MambaSpec):
                target += 2
            elif isinstance(spec, SlidingWindowSpec):
                target += cdiv(spec.sliding_window, spec.block_size) + 1
            else:
                target += cdiv(max_num_batched_tokens, spec.block_size)
        return int(target * (1 + WATERMARK_RATIO))

    def bind_gpu_block_pool(self, gpu_block_pool: BlockPool) -> None:
        """Bind GPU block pool so that we can touch blocks during stores.
        Called by Scheduler after kv_cache_manager is ready.

        绑定GPU块池，以便在存储操作期间可以touch块（防止被驱逐）。
        由调度器在kv_cache_manager准备就绪后调用。
        """
        self._gpu_block_pool = gpu_block_pool

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        """Return (num_new_tokens, is_async) from consecutive CPU cache hits.

        返回从连续CPU缓存命中中获得的新token数量和是否异步。

        工作流程：
        1. 释放之前可能存在的待处理CPU命中pin（如果有重试）
        2. 计算剩余需要匹配的哈希序列
        3. 在CPU缓存中查找最长的连续命中
        4. 如果找到命中，通过touch()固定这些块，防止在后续操作前被淘汰
        5. 返回命中长度和异步标志

        参数：
        - request: 请求对象
        - num_computed_tokens: 已计算的token数量

        返回：
        - (命中长度, True) 如果找到CPU缓存命中
        - (0, False) 如果没有命中
        """

        # Pins found CPU blocks so they survive LRU eviction until
        # update_state_after_alloc() consumes them. Any pin from an earlier
        # call on the same request (e.g. retry after a failed allocate_slots)
        # is dropped first.
        # 固定找到的CPU块，使其在update_state_after_alloc()消费之前不会被LRU淘汰。
        # 先释放同一请求的早期pin（例如allocate_slots失败后的重试）。
        if stale := self._pending_cpu_hits.pop(request.request_id, None):
            self._free_pending_cpu_hit(stale)

        num_skipped_hashes = num_computed_tokens // self.hash_block_size
        remaining_hashes = request.block_hashes[num_skipped_hashes:]

        if not remaining_hashes:
            return 0, False
        # Must recompute at least the last token, matching the logic in
        # kv_cache_manager.get_computed_blocks().
        # 必须至少重新计算最后一个token，与kv_cache_manager.get_computed_blocks()的逻辑一致
        max_hit_len = request.num_tokens - 1 - num_computed_tokens
        if max_hit_len <= 0:
            return 0, False
        # 在CPU协调器中查找最长的缓存命中
        cpu_hit_blocks, hit_length = self.cpu_coordinator.find_longest_cache_hit(
            remaining_hashes, max_hit_len
        )

        if hit_length > 0:
            # 收集所有非空块并固定它们
            pin_blocks = [
                blk for grp in cpu_hit_blocks for blk in grp if not blk.is_null
            ]
            self.cpu_block_pool.touch(pin_blocks)
            self._pending_cpu_hits[request.request_id] = (
                cpu_hit_blocks,
                hit_length,
            )
            return hit_length, True
        return 0, False

    # TODO(yifan): this API now only matches the suffix part of the prefix cache. A more
    # general API should scan blocks in both GPU and CPU block pool in a single pass.
    # TODO：此API目前只匹配前缀缓存的后缀部分。更通用的API应该在一次扫描中
    # 同时检查GPU和CPU块池中的块。
    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        """在分配GPU块后更新状态，准备CPU->GPU的加载操作。

        这个方法在调度器为请求分配GPU块后调用，负责：
        1. 在急切模式下注册请求的存储跟踪状态
        2. 从待处理的CPU命中中获取要加载的块
        3. 构建GPU块ID和CPU块ID的映射对
        4. 固定GPU和CPU块，防止在异步加载期间被淘汰
        5. 创建LoadRequestState记录加载任务

        参数：
        - request: 请求对象
        - blocks: 分配的KV缓存块
        - num_external_tokens: 需要从外部（CPU缓存）加载的token数量
        """
        req_id = request.request_id
        block_ids_by_group = blocks.get_block_ids()
        num_groups = len(block_ids_by_group)

        # Store tracking (eager mode only). Register the request;
        # block IDs are accumulated from scheduler_output in
        # _prepare_eager_store_specs via yield_req_data.
        # 存储跟踪（仅急切模式）。注册请求；
        # 块ID通过yield_req_data在_prepare_eager_store_specs中从scheduler_output累积。
        if not self._lazy_mode and req_id not in self._reqs_to_store:
            self._reqs_to_store[req_id] = StoreRequestState(
                request=request,
                block_ids=tuple([] for _ in range(num_groups)),
                num_stored_blocks=[0] * num_groups,
            )

        # Pop the CPU hit cached by get_num_new_matched_tokens(). The
        # found blocks were pinned there to survive LRU eviction in the window
        # between get_num_new_matched_tokens() and this matching call.
        # 弹出get_num_new_matched_tokens()缓存的CPU命中。
        # 找到的块在那里被固定，以在get_num_new_matched_tokens()和此匹配调用之间
        # 的窗口期存活LRU淘汰。
        pending = self._pending_cpu_hits.pop(req_id, None)

        if num_external_tokens == 0:
            if pending is not None:
                logger.warning(
                    "SimpleCPUOffloadScheduler: update_state_after_alloc "
                    "called for req_id=%s with no external tokens but "
                    "get_num_new_matched_tokens() unexpectedly recorded "
                    "a pending CPU hit; releasing the stale pin.",
                    req_id,
                )
                self._free_pending_cpu_hit(pending)
            return

        if pending is None:
            logger.warning(
                "SimpleCPUOffloadScheduler: update_state_after_alloc called "
                "for req_id=%s with num_external_tokens=%d but no pending "
                "CPU hit from get_num_new_matched_tokens(); skipping load.",
                req_id,
                num_external_tokens,
            )
            return

        cpu_hit_blocks_full, _ = pending

        # ``num_external_tokens`` is LCM-aligned (checked per-group below),
        # so this counts whole scheduler-aligned chunks of incoming tokens.
        # ``num_external_tokens``是LCM对齐的（下面按组检查），
        # 所以这里计算的是调度器对齐的整块incoming token数量
        num_blocks_to_load = num_external_tokens // self.block_size
        assert num_blocks_to_load > 0
        # 计算已缓存的FA块数量
        num_cached_fa_blocks = sum(
            blk.block_hash is not None for blk in blocks.blocks[self.fa_gidx]
        )
        num_computed_tokens = num_cached_fa_blocks * self.fa_block_size

        # Build transfer pairs across all groups.
        # 跨所有组构建传输对
        total_computed_tokens = num_computed_tokens + num_external_tokens
        kv_cache_groups = self.cpu_kv_cache_config.kv_cache_groups

        # The scheduler may have accepted fewer blocks than
        # get_num_new_matched_tokens() reported.
        # (e.g. due to token budget in test_partial_gpu_prefix_plus_cpu_load).
        # Take only the leading N blocks per group matching num_external_tokens;
        # the rest will be released along with the temp pin below.
        # 调度器可能接受的块数少于get_num_new_matched_tokens()报告的块数。
        # （例如由于test_partial_gpu_prefix_plus_cpu_load中的token预算限制）
        # 只取每组前N个块匹配num_external_tokens；
        # 其余的将随下面的临时pin一起释放。
        cpu_hit_blocks: list[list[KVCacheBlock]] = []
        for g in range(num_groups):
            g_block_size = kv_cache_groups[g].kv_cache_spec.block_size
            assert num_external_tokens % g_block_size == 0, (
                f"num_external_tokens={num_external_tokens} not aligned to "
                f"group {g} block_size={g_block_size}"
            )
            n_take_g = num_external_tokens // g_block_size
            cpu_hit_blocks.append(cpu_hit_blocks_full[g][:n_take_g])

        gpu_block_ids: list[int] = []
        cpu_block_ids: list[int] = []
        cpu_blocks_to_touch: list[KVCacheBlock] = []

        for g in range(num_groups):
            cpu_blocks_g = cpu_hit_blocks[g]
            n_ext_g = len(cpu_blocks_g)
            if n_ext_g == 0:
                continue

            # Number of blocks in the computed range for this group.
            # 此组在已计算范围内的块数量
            g_block_size = kv_cache_groups[g].kv_cache_spec.block_size
            n_computed_g = cdiv(total_computed_tokens, g_block_size)

            # Back-trace: ext blocks sit at the tail of the computed range.
            # 回溯：外部块位于已计算范围的尾部
            gpu_ext_start = n_computed_g - n_ext_g
            group_gpu_ids = block_ids_by_group[g]

            for i, cpu_blk in enumerate(cpu_blocks_g):
                # Skip null blocks (e.g. sliding window or mamba padding).
                # 跳过空块（例如滑动窗口或mamba填充）
                if cpu_blk.is_null:
                    continue
                gpu_block_ids.append(group_gpu_ids[gpu_ext_start + i])
                cpu_block_ids.append(cpu_blk.block_id)
                cpu_blocks_to_touch.append(cpu_blk)

        # Touch CPU blocks to prevent eviction during async load.
        # 固定CPU块，防止在异步加载期间被淘汰
        self.cpu_block_pool.touch(cpu_blocks_to_touch)
        # Release the temporary pin held since get_num_new_matched_tokens().
        # 释放自get_num_new_matched_tokens()以来持有的临时pin
        self._free_pending_cpu_hit(pending)

        # Touch GPU blocks to prevent freeing during async load
        # 固定GPU块，防止在异步加载期间被释放
        assert self._gpu_block_pool is not None
        self._gpu_block_pool.touch(
            [self._gpu_block_pool.blocks[bid] for bid in gpu_block_ids]
        )

        assert self._reqs_to_load.get(req_id) is None
        # 创建加载请求状态记录
        self._reqs_to_load[req_id] = LoadRequestState(
            request=request, transfer_meta=TransferMeta(gpu_block_ids, cpu_block_ids)
        )

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> SimpleCPUOffloadMetadata:
        """构建连接器元数据，包含本步骤的加载和存储操作。

        这是调度器端的核心方法，每步调用一次，负责：
        1. 准备存储操作规格（哪些GPU块需要卸载到CPU）
        2. 收集所有待加载的请求，准备加载操作规格
        3. 为操作分配事件索引，用于异步完成通知
        4. 构建并返回元数据对象，传递给工作节点

        返回：
        - SimpleCPUOffloadMetadata: 包含加载/存储操作详情的元数据
        """
        # --- Stores ---
        # 准备存储操作
        store_event = -1
        store_gpu, store_cpu, store_req_ids = self.prepare_store_specs(scheduler_output)
        if store_gpu:
            store_event = self._store_event_counter
            self._store_event_counter += 1
            self._store_event_to_blocks[store_event] = TransferMeta(
                store_gpu, store_cpu
            )
            if store_req_ids:  # For eager mode only, track req->blocks mapping
                # 仅急切模式：跟踪req->blocks映射
                self._store_event_to_reqs[store_event] = store_req_ids
                for req_id in store_req_ids:
                    store_state = self._reqs_to_store.get(req_id)
                    if store_state is not None:
                        store_state.store_events.add(store_event)

        # --- Loads ---
        # 准备加载操作：收集所有尚未分配事件的加载请求
        load_event = -1
        load_gpu: list[int] = []
        load_cpu: list[int] = []
        load_req_ids: list[str] = []
        for req_id, load_state in self._reqs_to_load.items():
            if load_state.load_event is not None:
                continue
            assert load_state.transfer_meta is not None
            load_gpu.extend(load_state.transfer_meta.gpu_block_ids)
            load_cpu.extend(load_state.transfer_meta.cpu_block_ids)
            load_req_ids.append(req_id)
        if load_req_ids:
            load_event = self._load_event_counter
            self._load_event_counter += 1
            for req_id in load_req_ids:
                self._reqs_to_load[req_id].load_event = load_event
            self._load_event_to_reqs[load_event] = load_req_ids

        # 构建最终的元数据对象
        result = SimpleCPUOffloadMetadata(
            load_event=load_event,
            load_gpu_blocks=load_gpu,
            load_cpu_blocks=load_cpu,
            load_event_to_reqs=self._load_event_to_reqs,
            store_event=store_event,
            store_gpu_blocks=store_gpu,
            store_cpu_blocks=store_cpu,
            need_flush=bool(scheduler_output.preempted_req_ids),
        )
        return result

    def prepare_store_specs(
        self, scheduler_output: SchedulerOutput
    ) -> tuple[list[int], list[int], list[str]]:
        """Prepare store specs for the store event.

        准备存储事件的规格。
        根据当前模式（懒惰或急切）选择不同的准备策略。

        返回：
        - (gpu_block_ids, cpu_block_ids, req_ids) 三元组
        """
        if self._lazy_mode:
            return self._prepare_lazy_store_specs()
        else:
            return self._prepare_eager_store_specs(scheduler_output)

    def _prepare_lazy_store_specs(
        self,
    ) -> tuple[list[int], list[int], list[str]]:
        """Single-pass cursor walk: offload cached GPU blocks near eviction.

        单遍游标遍历：卸载接近被淘汰的已缓存GPU块。

        懒惰模式的存储策略：
        1. 从GPU空闲队列的游标位置开始遍历
        2. 统计空闲或已卸载的块（对分配器来说是可安全淘汰的）
        3. 当覆盖了target_free个块或CPU容量用尽时停止
        4. 对于每个有哈希且CPU中未缓存的GPU块，创建CPU副本
        5. 批量分配CPU块并记录哈希值

        返回：
        - (gpu_block_ids, cpu_block_ids, []) 三元组，req_ids始终为空
        """
        gpu_pool = self._gpu_block_pool
        if gpu_pool is None or self._target_free <= 0:
            return [], [], []

        free_queue = gpu_pool.free_block_queue
        cpu_pool = self.cpu_block_pool
        num_cpu_free = cpu_pool.get_num_free_blocks()

        # Validate cursor: stale if block was removed from free queue.
        # 验证游标：如果块已从空闲队列中移除，则游标失效
        if self._cursor is not None and self._cursor.ref_cnt > 0:
            self._cursor = None

        # Determine start node.
        # 确定起始节点
        if self._cursor is None:
            node = free_queue.fake_free_list_head.next_free_block
        else:
            node = self._cursor.next_free_block

        tail = free_queue.fake_free_list_tail
        gpu_ids: list[int] = []
        block_hashes: list[bytes] = []
        covered = 0
        last_visited = self._cursor

        # 遍历空闲队列，收集需要卸载的块
        while (
            node is not None
            and node is not tail
            and covered < self._target_free
            and len(gpu_ids) < num_cpu_free
        ):
            last_visited = node
            bhash = node.block_hash

            if (
                bhash is not None
                and not node.is_null
                and cpu_pool.cached_block_hash_to_block.get_one_block(bhash) is None
            ):
                gpu_ids.append(node.block_id)
                block_hashes.append(bhash)

            covered += 1
            node = node.next_free_block

        self._cursor = last_visited

        # Batch-allocate CPU blocks and stamp hashes.
        # 批量分配CPU块并记录哈希值
        if gpu_ids:
            cpu_blocks = cpu_pool.get_new_blocks(len(gpu_ids))
            cpu_ids = [blk.block_id for blk in cpu_blocks]
            for cpu_blk, bhash in zip(cpu_blocks, block_hashes):  # type: ignore[assignment]
                cpu_blk._block_hash = bhash  # type: ignore[assignment]
            # Touch GPU blocks to prevent eviction during async copy.
            # 固定GPU块，防止在异步复制期间被淘汰
            gpu_pool.touch([gpu_pool.blocks[bid] for bid in gpu_ids])
        else:
            cpu_ids = []

        return gpu_ids, cpu_ids, []

    def _prepare_eager_store_specs(
        self, scheduler_output: SchedulerOutput
    ) -> tuple[list[int], list[int], list[str]]:
        """Identify newly computed blocks to offload from scheduler requests.

        从调度器请求中识别新计算的块进行卸载（急切模式）。

        急切模式的存储策略：
        1. 遍历调度器输出中的所有请求
        2. 对于每个请求，检查已确认计算完成的块
        3. 跳过已缓存到CPU的块和正在传输中的块
        4. 为新块分配CPU空间并记录哈希值
        5. 固定GPU块，防止在异步复制期间被释放

        注意：只有KV数据已被GPU确认写入的块才会被存储。
        当前步骤的块要到下一步骤才会被存储。
        如果请求在同一步骤完成，其最后一个完整块可能会被遗漏。

        返回：
        - (merged_gpu_block_ids, merged_cpu_block_ids, req_ids) 三元组
        """

        merged_gpu_block_ids: list[int] = []
        merged_cpu_block_ids: list[int] = []
        req_ids: list[str] = []

        gpu_block_pool = self._gpu_block_pool
        if gpu_block_pool is None:
            return [], [], []
        cpu_block_pool = self.cpu_block_pool
        num_free = cpu_block_pool.get_num_free_blocks()
        kv_cache_groups = self.cpu_kv_cache_config.kv_cache_groups
        num_groups = len(kv_cache_groups)
        # Dedup against blocks already scheduled.
        # 去重：排除已调度的块
        in_flight = self._in_flight_store_gpu_blocks

        for req_id, new_block_id_groups, preempted in yield_req_data(scheduler_output):
            state = self._reqs_to_store.get(req_id)
            if state is None or state.finished:
                continue

            # Accumulate new block IDs.
            # 累积新的块ID
            if preempted:
                state.block_ids = tuple([] for _ in range(num_groups))
                state.num_stored_blocks = [0] * num_groups
            if new_block_id_groups:
                for g in range(min(num_groups, len(new_block_id_groups))):
                    if new_block_id_groups[g] is not None:
                        state.block_ids[g].extend(new_block_id_groups[g])

            num_new_tokens = scheduler_output.num_scheduled_tokens.get(req_id, 0)
            if num_new_tokens == 0:
                continue

            block_ids_by_group = state.block_ids
            if not block_ids_by_group:
                continue

            # --- Phase 1: Scan blocks, classify as cached vs to-store ---
            # 阶段1：扫描块，分类为已缓存或待存储
            gpu_block_ids: list[int] = []
            block_hashes_to_store: list[bytes] = []
            advanced_per_group: list[int] = [0] * num_groups
            out_of_space = False
            # Confirmed tokens: KV data written and visible to all streams.
            # 已确认的token：KV数据已写入且对所有流可见
            req = state.request
            confirmed_tokens = req.num_computed_tokens - req.num_output_placeholders
            # Cap to blocks with confirmed KV data.
            # 限制到已确认KV数据的块
            aligned_tokens = confirmed_tokens // self.block_size * self.block_size

            for g in range(num_groups):
                # FIXME (yifan): handle CPU cache eviction, where
                # num_stored_blocks can be stale and omit evicted blocks in
                # the middle of the request.
                # FIXME (yifan)：处理CPU缓存淘汰，其中
                # num_stored_blocks可能过时，遗漏请求中间被淘汰的块
                already_stored_g = state.num_stored_blocks[g]
                group_gpu_ids = block_ids_by_group[g]

                g_block_size = kv_cache_groups[g].kv_cache_spec.block_size
                ready_blocks_g = aligned_tokens // g_block_size
                scannable = group_gpu_ids[already_stored_g:ready_blocks_g]

                for gpu_block_id in scannable:
                    gpu_block = gpu_block_pool.blocks[gpu_block_id]
                    if gpu_block.is_null:
                        advanced_per_group[g] += 1
                        continue

                    bhash_with_group = gpu_block.block_hash
                    if bhash_with_group is None:
                        # Masked-out SWA position the coordinator chose not to
                        # hash; it can never serve a prefix-cache hit, so skip.
                        # 被屏蔽的SWA位置，协调器选择不哈希；
                        # 它永远无法提供前缀缓存命中，所以跳过
                        advanced_per_group[g] += 1
                        continue

                    # Skip if already scheduled for store or already cached in CPU.
                    # 如果已调度存储或已缓存到CPU，则跳过
                    if (
                        gpu_block_id in in_flight
                        or cpu_block_pool.cached_block_hash_to_block.get_one_block(
                            bhash_with_group
                        )
                        is not None
                    ):
                        advanced_per_group[g] += 1
                        continue

                    if num_free <= 0:
                        out_of_space = True
                        break
                    num_free -= 1

                    gpu_block_ids.append(gpu_block_id)
                    block_hashes_to_store.append(bhash_with_group)
                    advanced_per_group[g] += 1

                if out_of_space:
                    break

            # --- Phase 2: Batch allocate CPU blocks and stamp hashes ---
            # 阶段2：批量分配CPU块并记录哈希值
            n_to_alloc = len(gpu_block_ids)
            if n_to_alloc > 0:
                cpu_blocks_alloc = cpu_block_pool.get_new_blocks(n_to_alloc)
                cpu_block_ids = [blk.block_id for blk in cpu_blocks_alloc]
                for cpu_blk, bhash in zip(cpu_blocks_alloc, block_hashes_to_store):
                    cpu_blk._block_hash = bhash  # type: ignore[assignment]
            else:
                cpu_block_ids = []

            if cpu_block_ids:
                req_ids.append(req_id)
                merged_gpu_block_ids.extend(gpu_block_ids)
                merged_cpu_block_ids.extend(cpu_block_ids)
                in_flight.update(gpu_block_ids)

                # Touch GPU blocks to prevent freeing during async copy
                # 固定GPU块，防止在异步复制期间被释放
                gpu_block_pool.touch(
                    [gpu_block_pool.blocks[bid] for bid in gpu_block_ids]
                )

                logger.debug(
                    "Request %s: Scheduling store of %d blocks to CPU (%d groups)",
                    req_id,
                    len(cpu_block_ids),
                    num_groups,
                )

            # Advance per-group cursors (includes cached hits + newly stored)
            # 推进每组游标（包括缓存命中 + 新存储的块）
            for g in range(num_groups):
                state.num_stored_blocks[g] += advanced_per_group[g]

        return merged_gpu_block_ids, merged_cpu_block_ids, req_ids

    def update_connector_output(self, connector_output: KVConnectorOutput) -> None:
        """Handle async transfer completions from worker.

        处理来自工作节点的异步传输完成通知。

        完成通知的类型：
        1. 加载完成：通过finished_recving报告，包含真实的请求ID
        2. 存储完成：通过kv_connector_worker_meta报告，包含每个事件的工作节点计数

        对于存储完成：
        - 跨步骤累积计数
        - 只有当所有工作节点都报告完成时，才处理该存储事件
        - 处理包括：将CPU块注册到缓存映射，释放GPU块的引用
        """
        # --- Load completions ---
        # 处理加载完成
        for req_id in list(connector_output.finished_recving or []):
            self._cleanup_load_request(req_id)

        # --- Store completions ---
        # 处理存储完成
        meta = connector_output.kv_connector_worker_meta
        if not isinstance(meta, SimpleCPUOffloadWorkerMetadata):
            return
        for event_idx, count in meta.completed_store_events.items():
            total = self._store_event_pending_counts.get(event_idx, 0) + count
            if total >= self._expected_worker_count:
                self._store_event_pending_counts.pop(event_idx, None)
                self._process_store_event(event_idx)
            else:
                self._store_event_pending_counts[event_idx] = total

    def _process_store_event(self, event_idx: int) -> None:
        """Process a fully-completed store event.

        处理完全完成的存储事件。

        当所有工作节点都报告某个存储事件完成后调用此方法。
        负责：
        1. 获取传输元数据
        2. 从飞行中集合移除GPU块ID（急切模式）
        3. 处理存储完成，将块注册到CPU缓存
        4. 更新请求级别的存储状态
        """
        transfer = self._store_event_to_blocks.pop(event_idx)
        if not self._lazy_mode:
            self._in_flight_store_gpu_blocks.difference_update(transfer.gpu_block_ids)
        self._process_store_completion(transfer.gpu_block_ids, transfer.cpu_block_ids)
        logger.debug(
            "Store event %d completed: cached %d blocks to CPU",
            event_idx,
            len(transfer.cpu_block_ids),
        )

        # Eager only: update per-req state
        # 仅急切模式：更新请求级别的状态
        if not self._lazy_mode:
            for req_id in self._store_event_to_reqs.pop(event_idx, []):
                state = self._reqs_to_store.get(req_id)
                if state is None:
                    continue
                state.store_events.discard(event_idx)
                if state.finished and not state.store_events:
                    self._cleanup_store_request(req_id)

    def _process_store_completion(
        self, gpu_block_ids: list[int], cpu_block_ids: list[int]
    ) -> None:
        """Cache CPU blocks per-group and release GPU refs.

        将CPU块注册到缓存映射并释放GPU引用。

        块哈希在分配时已经记录到CPU块上（在_prepare_*_store_specs中）。
        这里只需将它们注册到缓存映射中，使它们可以被加载路径发现。

        完成后：
        1. CPU块进入前缀缓存，可以被后续请求命中
        2. GPU块的引用计数减少，可能被释放回空闲池
        """
        assert len(cpu_block_ids) == len(gpu_block_ids)

        cpu_blocks = [self.cpu_block_pool.blocks[bid] for bid in cpu_block_ids]

        # 将CPU块注册到缓存映射
        for cpu_block in cpu_blocks:
            bhash = cpu_block.block_hash
            assert bhash is not None
            self.cpu_block_pool.cached_block_hash_to_block.insert(bhash, cpu_block)

        # Free CPU and GPU blocks' ref counts to turn them into prefix cache
        # 释放CPU和GPU块的引用计数，使它们成为前缀缓存
        self.cpu_block_pool.free_blocks(cpu_blocks)
        assert self._gpu_block_pool is not None
        self._gpu_block_pool.free_blocks(
            self._gpu_block_pool.blocks[bid] for bid in gpu_block_ids
        )

    def has_pending_stores(self) -> bool:
        """Return True if there are in-flight store transfers.

        返回是否有正在传输中的存储操作。
        """
        return bool(self._store_event_to_blocks)

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        """Always returns (False, None). GPU blocks are protected by ref_cnt,
        so the scheduler can free blocks immediately.

        请求完成处理。始终返回(False, None)。
        GPU块通过引用计数保护，因此调度器可以立即释放块。

        处理逻辑：
        1. 释放可能存在的待处理CPU命中pin
        2. 处理加载状态：如果有飞行中的加载，标记为延迟清理
        3. 处理存储状态（急切模式）：如果有飞行中的存储，标记为延迟清理

        返回：
        - (False, None): 表示不需要额外的块释放，GPU块由引用计数管理
        """
        req_id = request.request_id

        # Release any temp CPU hit pin from get_num_new_matched_tokens()
        # if request is canceled or preempted before update_state_after_alloc()
        # 释放来自get_num_new_matched_tokens()的临时CPU命中pin，
        # 如果请求在update_state_after_alloc()之前被取消或抢占
        pending = self._pending_cpu_hits.pop(req_id, None)
        if pending is not None:
            self._free_pending_cpu_hit(pending)

        # Handle load: defer cleanup if load is in-flight
        # 处理加载：如果有飞行中的加载，则延迟清理
        load_state = self._reqs_to_load.get(req_id)
        if load_state is not None:
            if load_state.load_event is not None:
                load_state.finished = True  # Defer: load in-flight
            else:
                self._cleanup_load_request(req_id)

        # Handle store (eager mode only): defer cleanup if stores in-flight
        # 处理存储（仅急切模式）：如果有飞行中的存储，则延迟清理
        if not self._lazy_mode:
            store_state = self._reqs_to_store.get(req_id)
            if store_state is not None:
                if store_state.store_events:
                    store_state.finished = True  # Defer: stores in-flight
                else:
                    self._cleanup_store_request(req_id)

        return False, None

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        """处理所有组的请求完成。委托给request_finished方法。"""
        return self.request_finished(request, block_ids=[])

    def _free_pending_cpu_hit(self, pending: tuple) -> None:
        """Release the temporary CPU block pin taken in get_num_new_matched_tokens().

        释放在get_num_new_matched_tokens()中获取的临时CPU块pin。
        这些块在查找缓存命中时被固定，现在需要释放它们。
        """
        cpu_hit_blocks, _ = pending
        blocks_to_free = [
            blk for grp in cpu_hit_blocks for blk in grp if not blk.is_null
        ]
        if blocks_to_free:
            self.cpu_block_pool.free_blocks(blocks_to_free)

    def _cleanup_load_request(self, req_id: str) -> None:
        """Release all load resources for a request.

        释放请求的所有加载资源。

        清理流程：
        1. 从_reqs_to_load中移除请求状态
        2. 从_load_event_to_reqs中移除事件映射
        3. 释放CPU块的touch引用
        4. 释放GPU块的touch引用

        此方法在以下情况被调用：
        - request_finished(): 请求完成时
        - update_connector_output(): 加载事件完成时
        """
        state = self._reqs_to_load.pop(req_id, None)
        if state is None:
            return
        # Remove from load event mapping (only this req, not whole event)
        # 从加载事件映射中移除（仅此请求，不是整个事件）
        if state.load_event is not None:
            reqs = self._load_event_to_reqs.get(state.load_event)
            if reqs is not None:
                with contextlib.suppress(ValueError):
                    reqs.remove(req_id)
                if not reqs:
                    self._load_event_to_reqs.pop(state.load_event, None)

        if state.transfer_meta is not None:
            # Free CPU touch refs
            # 释放CPU touch引用
            self.cpu_block_pool.free_blocks(
                self.cpu_block_pool.blocks[bid]
                for bid in state.transfer_meta.cpu_block_ids
            )
            # Free GPU touch refs
            # 释放GPU touch引用
            assert self._gpu_block_pool is not None
            self._gpu_block_pool.free_blocks(
                self._gpu_block_pool.blocks[bid]
                for bid in state.transfer_meta.gpu_block_ids
            )

    def _cleanup_store_request(self, req_id: str) -> None:
        """Release store metadata for a request.

        释放请求的存储元数据。

        注意：这只是元数据清理，不释放块。
        块的缓存和GPU引用释放通过_process_store_completion()在作业完成时处理。

        清理流程：
        1. 从_reqs_to_store中移除请求状态
        2. 从_store_event_to_reqs中移除事件映射
        3. 清空请求的存储事件集合
        """
        state = self._reqs_to_store.pop(req_id, None)
        if state is None:
            return
        for event_idx in list(state.store_events):
            if (reqs := self._store_event_to_reqs.get(event_idx)) is not None:
                with contextlib.suppress(ValueError):
                    reqs.remove(req_id)
                if not reqs:
                    self._store_event_to_reqs.pop(event_idx, None)
        state.store_events.clear()

    def take_events(self) -> Iterable[KVCacheEvent]:
        """获取CPU块池的KV缓存事件。

        返回CPU块池中累积的事件，用于监控和调试。
        """
        return self.cpu_block_pool.take_events()
