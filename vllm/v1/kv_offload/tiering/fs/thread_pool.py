# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Dual-queue thread pool for KV cache offloading I/O operations.

本模块实现了文件系统二级层的 I/O 线程池，负责执行实际的磁盘读写操作。

整体架构：
  ┌─────────────┐    ┌─────────────┐
  │  Load Queue  │    │ Store Queue │
  └──────┬───┬──┘    └──┬────┬─────┘
         │   │          │    │
    ┌────┘   └────┐  ┌──┘    └──┐
    ▼             ▼  ▼          ▼
  ┌────────────┐  ┌─────────────┐
  │ Load-Prio  │  │ Store-Prio  │
  │  Threads   │  │  Threads    │
  │ (读优先线程)│  │ (写优先线程) │
  └────────────┘  └─────────────┘

设计要点：
  1. 双队列：Load 和 Store 任务分别入队，避免大写入任务阻塞读取请求。
  2. 优先级调度：读优先线程优先处理 load 队列，写优先线程优先处理 store 队列，
     但两者都可以在主队列为空时"借用"对方队列，避免线程饥饿。
  3. 非阻塞提交：FileSystemTierManager 的 submit_load/submit_store 在
     Scheduler 线程中运行，只做入队操作，不阻塞调度流程。
  4. 任务粒度：每个 Job（如"加载请求 X 的 8 个 block"）被拆分为多个
     独立的 per-block 子任务，可被不同线程并行执行。
  5. 完成追踪：JobState 追踪每个 Job 的所有子任务完成情况，
     所有子任务完成后将 Job 结果放入 finished 队列。
"""

# ===== 文件级别概述 =====
#
# 本文件实现了 vLLM v1 的 KV cache 二级存储（文件系统）I/O 线程池。
# 在 vLLM 的分层存储架构中，KV cache 可能存在于：
#   - 一级层（GPU 显存）：速度快但容量有限
#   - 二级层（文件系统/磁盘）：容量大但速度慢
#
# 当需要将 KV cache 在一级层和二级层之间转移时（提升/promotion 或
# 级联/cascade），实际的磁盘 I/O 操作由本文件中的 DualQueueThreadPool 执行。
#
# 整体数据流：
#   Scheduler 线程                I/O 线程池                  Scheduler 线程
#   ─────────────                ────────────                  ─────────────
#   submit_load/store()
#       │
#       ▼
#   enqueue_load/store()  ──→  load_q / store_q  ──→  _worker 线程执行
#       │                                                    │
#       │                                              task_done()
#       │                                                    │
#       │                                                    ▼
#       └──────────── get_finished() ◄────────────  finished_q
#
# 关键类：
#   - JobState: 追踪单个 Job 的所有 per-block 子任务完成状态（线程安全）
#   - DualQueueThreadPool: 双队列线程池，管理读/写线程组和任务调度
#
# 关键设计决策：
#   1. 为什么用双队列而非单队列？
#      因为存储任务（store）通常是大批量的（如请求结束后批量写入所有 block），
#      如果和加载任务混在同一个队列中，会严重阻塞延迟敏感的读取请求。
#      双队列 + 优先级线程确保读取请求始终能获得优先服务。
#
#   2. 为什么每个 Job 拆分为 per-block 子任务？
#      为了最大化 I/O 并行度。一个 Job 可能涉及多个 block 的读写，
#      拆分后这些 block 可以被不同的 I/O 线程同时处理。
#
#   3. 为什么使用 daemon 线程？
#      daemon 线程在主线程退出时会自动终止，不会阻止进程退出。
#      这对于推理引擎的优雅关闭很重要。

import threading
from collections import deque
from collections.abc import Callable, Iterable

from vllm.logger import init_logger
from vllm.v1.kv_offload.tiering.base import JobId

logger = init_logger(__name__)


class JobState:
    """
    Thread-safe completion tracker for a set of per-block I/O tasks.

    Each task calls task_done(success) when it finishes.
    """
    # 中文注释：JobState 是一个线程安全的"任务完成追踪器"。
    #
    # 在二级层 I/O 架构中，一个 Job（如"加载请求 X 的 8 个 KV block"）
    # 会被拆分为多个独立的 per-block 子任务，分散到不同 I/O 线程执行。
    # JobState 负责追踪所有子任务的完成状态。
    #
    # 工作流程：
    #   1. 提交 Job 时，创建 JobState(job_id, n_tasks=8)。
    #   2. 每个 per-block 子任务完成后调用 task_done(success)。
    #   3. task_done() 原子地更新完成计数和成功标志。
    #   4. 当 completed == n_tasks 时，返回 (True, success)，
    #      表示整个 Job 已完成，调用方将结果放入 finished 队列。
    #   5. 任何一个子任务失败（success=False），整个 Job 标记为失败。
    #      框架据此进行资源清理（如释放一级层 block 引用计数）。

    # 中文注释：使用 __slots__ 限制实例属性，减少内存开销。
    # JobState 可能大量创建（每个 Job 一个），__slots__ 能显著降低内存占用。
    __slots__ = ("_job_id", "_n_tasks", "_completed", "_success", "_lock")

    def __init__(self, job_id: JobId, n_tasks: int) -> None:
        # 中文注释：job_id — 唯一标识一个 Job，由上层（FileSystemTierManager）生成。
        self._job_id: JobId = job_id
        # 中文注释：n_tasks — 该 Job 被拆分的 per-block 子任务总数。
        # 例如加载 8 个 KV block 时，n_tasks=8。
        self._n_tasks = n_tasks
        # 中文注释：_completed — 已完成的子任务计数，初始为 0。
        self._completed = 0
        # 中文注释：_success — 累积的成功标志，初始为 True。
        # 采用"短路失败"设计：一旦任何子任务失败，此标志变为 False 且不可恢复。
        self._success = True
        # 中文注释：_lock — 保护 _completed 和 _success 的线程锁。
        # 多个 I/O 线程可能同时完成同一 Job 的不同子任务，必须加锁。
        self._lock = threading.Lock()

    @property
    def job_id(self) -> JobId:
        # 中文注释：返回该 Job 的唯一标识符。
        return self._job_id

    def task_done(self, success: bool) -> tuple[bool, bool]:
        """Returns if job completed and success flag"""
        # 中文注释：标记一个 per-block 子任务完成。
        #
        # 使用锁保证线程安全，因为多个 I/O 线程可能同时完成同一 Job 的不同子任务。
        #
        # 参数：
        #   success: 该子任务是否成功完成。
        #     True  — 子任务正常完成（如磁盘读取/写入成功）。
        #     False — 子任务失败（如 I/O 错误、文件不存在等）。
        #
        # 返回值 (tuple[bool, bool])：
        #   (True,  success) — Job 的所有子任务已完成，success 为整体结果。
        #   (False, success) — Job 仍有未完成的子任务，success 为当前累积结果。
        #
        # "短路失败"设计：一旦某个子任务失败（success=False），
        # 后续所有子任务的 success 都不会改变最终结果（保持 False）。
        # 这避免了复杂的错误传播逻辑。
        with self._lock:
            self._completed += 1
            if not success:
                self._success = False
            return self._completed == self._n_tasks, self._success


class DualQueueThreadPool:
    """
    Thread pool with two task queues (load and store) and two thread groups.

    Load-priority threads drain the load queue first, then fall back to the
    store queue.  Store-priority threads do the reverse.  Both queues share
    a single condition variable.
    """
    # 中文注释：双队列线程池——文件系统二级层的核心 I/O 执行引擎。
    #
    # 核心设计：
    #   1. 两个任务队列：
    #      - load_q:  存放"从磁盘读取 block 到内存"的任务（提升/promotion）。
    #      - store_q: 存放"从内存写入 block 到磁盘"的任务（级联/cascade）。
    #      分离队列避免大量写入任务（如请求结束后批量存储）淹没读取请求。
    #
    #   2. 两类线程组：
    #      - 读优先线程（load_priority=True）：优先处理 load_q，空时处理 store_q。
    #      - 写优先线程（load_priority=False）：优先处理 store_q，空时处理 load_q。
    #      这种"优先级 + 回退"设计确保：
    #        a) 读取请求（影响用户延迟）获得优先服务。
    #        b) 写入任务不会被无限拖延（线程空闲时会处理写入）。
    #        c) 两类线程都不会饥饿。
    #
    #   3. 共享条件变量：所有线程通过同一个 Condition 等待任务，
    #      避免每类线程需要独立的信号机制。
    #
    #   4. 完成队列：finished_q 收集已完成的 Job 结果，
    #      由 FileSystemTierManager.get_finished_jobs() 轮询读取。

    def __init__(
        self,
        n_read_threads: int,
        n_write_threads: int,
        thread_name_prefix: str = "fs_secondary_tier",
    ) -> None:
        # 中文注释：构造函数，初始化双队列线程池。
        #
        # 参数：
        #   n_read_threads:  读优先线程数量，负责优先处理 load（读取）任务。
        #   n_write_threads: 写优先线程数量，负责优先处理 store（写入）任务。
        #   thread_name_prefix: 线程名称前缀，用于调试和日志追踪。
        #
        # 初始化的内部状态：
        #   _load_q:  加载任务队列（deque），存放 (fn, JobState) 元组。
        #   _store_q: 存储任务队列（deque），存放 (fn, JobState) 元组。
        #   _condition: 共享的条件变量，用于线程等待任务和唤醒通知。
        #   _stop: 停止标志，为 True 时所有工作线程退出循环。
        #   _threads: 所有工作线程的引用列表，用于 shutdown 时 join。
        #   _finished_q: 已完成 Job 的结果队列，存放 (JobId, success) 元组。
        self._load_q: deque = deque()
        self._store_q: deque = deque()
        self._condition = threading.Condition(threading.Lock())
        self._stop = False
        self._threads: list[threading.Thread] = []
        self._finished_q: deque[tuple[JobId, bool]] = deque()

        # 中文注释：创建读优先线程（load_priority=True）。
        # 每个线程运行 _worker(True) 循环，优先从 load_q 取任务。
        for i in range(n_read_threads):
            t = threading.Thread(
                target=self._worker,
                args=(True,),
                name=f"{thread_name_prefix}_l{i}",
                daemon=True,
            )
            t.start()
            self._threads.append(t)

        # 中文注释：创建写优先线程（load_priority=False）。
        # 每个线程运行 _worker(False) 循环，优先从 store_q 取任务。
        for i in range(n_write_threads):
            t = threading.Thread(
                target=self._worker,
                args=(False,),
                name=f"{thread_name_prefix}_s{i}",
                daemon=True,
            )
            t.start()
            self._threads.append(t)

    def enqueue_load(
        self,
        job_id: JobId,
        n_tasks: int,
        tasks: Iterable[Callable],
    ) -> None:
        """Enqueue load tasks for a job (high-priority for load-priority threads)."""
        # 中文注释：将一个"加载"Job 的所有 per-block 子任务入队到 load_q。
        #
        # 执行流程：
        #   1. 创建 JobState 来追踪该 Job 的 n_tasks 个子任务的完成情况。
        #   2. 在锁保护下，将每个子任务作为 (fn, state) 元组追加到 load_q。
        #      fn 是一个 functools.partial 回调（如 load_block(path, view, offset, size)）。
        #   3. notify(n_tasks) 唤醒 n_tasks 个等待中的线程来并行处理这些任务。
        #
        # 此方法由 FileSystemTierManager.submit_load() 调用，
        # 在 Scheduler 线程中运行，必须非阻塞（只做入队和通知）。
        #
        # 参数：
        #   job_id:  Job 的唯一标识符，用于追踪和报告完成状态。
        #   n_tasks: 该 Job 被拆分的子任务数量（通常等于需要加载的 block 数）。
        #   tasks:   可迭代的回调函数，每个函数对应一个 per-block 的 I/O 操作。
        state = JobState(job_id, n_tasks)
        with self._condition:
            for fn in tasks:
                self._load_q.append((fn, state))
            # 中文注释：notify(n_tasks) 唤醒 n_tasks 个正在等待的线程。
            # 这样可以最大化并行度，让多个线程同时处理同一个 Job 的不同 block。
            # 注意：如果等待中的线程少于 n_tasks，多余的 notify 会被忽略（无害）。
            self._condition.notify(n_tasks)

    def enqueue_store(
        self,
        job_id: JobId,
        n_tasks: int,
        tasks: Iterable[Callable],
    ) -> None:
        """Enqueue store tasks for a job (high-priority for store-priority threads)."""
        # 中文注释：将一个"存储"Job 的所有 per-block 子任务入队到 store_q。
        #
        # 与 enqueue_load 对称，区别在于任务进入 store_q，会被写优先线程优先消费。
        # 读优先线程也可以在 load_q 为空时消费 store_q 中的任务。
        #
        # 参数：
        #   job_id:  Job 的唯一标识符。
        #   n_tasks: 该 Job 被拆分的子任务数量。
        #   tasks:   可迭代的回调函数，每个函数对应一个 per-block 的 I/O 操作。
        state = JobState(job_id, n_tasks)
        with self._condition:
            for fn in tasks:
                self._store_q.append((fn, state))
            self._condition.notify(n_tasks)

    def get_finished(self) -> list[tuple[JobId, bool]]:
        # 中文注释：获取所有已完成的 Job 结果，并从完成队列中移除。
        #
        # 此方法由 FileSystemTierManager.get_finished_jobs() 调用，
        # 在 Scheduler 线程的主循环中定期轮询。
        #
        # 返回值 (list[tuple[JobId, bool]])：
        #   每个元组为 (job_id, success)：
        #     - job_id: 完成的 Job 标识符。
        #     - success: True 表示所有子任务成功，False 表示有子任务失败。
        #   返回空列表表示当前没有已完成的 Job。
        #
        # 注意：此方法不是线程安全的，因为它只在 Scheduler 线程中调用，
        # 而 _finished_q 的写入（在 _worker 中）和读取（在此方法中）
        # 实际上发生在不同线程。但由于 deque 的 append/popleft 在 CPython
        # 中受 GIL 保护，且结果最多延迟一个轮询周期，这是可接受的。
        jobs = []
        while self._finished_q:
            jobs.append(self._finished_q.popleft())
        return jobs

    def shutdown(self, wait: bool = True) -> None:
        # 中文注释：关闭线程池，停止所有工作线程。
        #
        # 执行流程：
        #   1. 在锁保护下设置 _stop=True，清空两个任务队列。
        #   2. notify_all() 唤醒所有正在等待任务的线程。
        #   3. 被唤醒的线程检测到 _stop=True 后退出 _worker 循环。
        #   4. 如果 wait=True，等待所有线程真正结束（join）。
        #
        # 参数：
        #   wait: 是否等待所有线程结束。
        #     True  — 同步等待，确保所有线程退出后才返回。
        #     False — 仅设置停止标志，不等待线程退出（daemon 线程会自动终止）。
        with self._condition:
            self._stop = True
            self._load_q.clear()
            self._store_q.clear()
            self._condition.notify_all()
        if wait:
            for t in self._threads:
                t.join()

    def _worker(self, load_priority: bool) -> None:
        # Wait for tasks, process from primary queue first, fall back to secondary.
        # 中文注释：工作线程的主循环——从队列中取任务并执行。
        #
        # 此方法是每个工作线程的入口函数，持续循环直到 _stop 被设置。
        #
        # 参数：
        #   load_priority: 线程的优先级方向。
        #     True  — 读优先线程：primary=load_q, secondary=store_q。
        #     False — 写优先线程：primary=store_q, secondary=load_q。
        #
        # 单次循环的执行流程：
        #   1. 【等待阶段】在条件变量上等待，直到有任务可处理或收到停止信号。
        #   2. 【停止检查】如果 _stop=True，退出循环，线程结束。
        #   3. 【取任务阶段】根据线程优先级，从 primary 队列取任务；
        #      如果 primary 队列为空，则从 secondary 队列取（"借用"对方队列）。
        #   4. 【执行阶段】释放锁后执行任务（task()），这是实际的磁盘 I/O 操作。
        #   5. 【完成追踪】调用 state.task_done() 更新子任务完成状态。
        #      如果 Job 的所有子任务都已完成，将结果放入 finished_q。
        #
        # 关键设计细节：
        #   - 锁只在"等待"和"取任务"阶段持有，"执行任务"阶段释放锁，
        #     这样多个线程可以并行执行 I/O 操作，不会互相阻塞。
        #   - "借用"机制（primary 为空时取 secondary）确保线程不会空闲等待，
        #     即使某一类任务暂时没有，线程也能处理另一类任务。
        while True:
            with self._condition:
                # 中文注释：条件等待——线程在此阻塞，直到以下任一条件满足：
                #   - _stop=True（关闭信号）
                #   - _load_q 非空（有加载任务）
                #   - _store_q 非空（有存储任务）
                self._condition.wait_for(
                    lambda: self._stop or self._load_q or self._store_q
                )
                # 中文注释：收到停止信号，退出循环，线程结束。
                if self._stop:
                    return
                # 中文注释：根据线程优先级确定主队列和备用队列。
                # 读优先线程：主队列=load_q，备用队列=store_q。
                # 写优先线程：主队列=store_q，备用队列=load_q。
                primary = self._load_q if load_priority else self._store_q
                secondary = self._store_q if load_priority else self._load_q
                # 中文注释：优先从主队列取任务；主队列为空时从备用队列取。
                # 这是"优先级 + 回退"机制的核心实现。
                task, state = primary.popleft() if primary else secondary.popleft()
            # 中文注释：释放锁后执行实际的 I/O 操作。
            # 此时其他线程可以继续从队列中取任务，实现并行 I/O。
            try:
                task()
                # 中文注释：子任务成功完成，标记 success=True。
                job_finished, success = state.task_done(True)
            except Exception as exc:
                # 中文注释：子任务执行失败（如磁盘 I/O 错误），记录错误日志，
                # 标记 success=False。JobState 会将整个 Job 标记为失败。
                logger.error(
                    "Job %s block I/O failed: %s",
                    state.job_id,
                    exc,
                )
                job_finished, success = state.task_done(False)

            if job_finished:
                # 中文注释：Job 的所有子任务已完成，将最终结果放入完成队列。
                # FileSystemTierManager 会通过 get_finished() 读取这些结果，
                # 并进行后续处理（如更新 block 引用计数、通知上层）。
                self._finished_q.append((state.job_id, success))
