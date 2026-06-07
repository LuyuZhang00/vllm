# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
FileSystemTierManager: Pure-Python file system secondary tier for KV cache offloading.

Store path:
    Data is written to a temp file (<dest_path.tmp>) via os.write,
    then os.replace'd to the final path (without .tmp).

Load path:
    Data is read from the block file directly via os.readv into the
    provided memoryview slice.

File naming:  <base_path>_r<rank>/<hhh>/<hh>_g<group_idx>/<hash_hex>.bin
              (hash-based subdirectories to limit directory fan-out)
"""

# =============================================================================
# 文件概述：FileSystemTierManager —— 基于文件系统的 KV cache 二级存储层管理器
# =============================================================================
#
# 本模块是 vLLM v1 多级 KV cache 卸载架构中"二级存储层"的文件系统实现。
#
# 【架构定位】
#   在 vLLM 的多级缓存架构中，数据流如下：
#     存储（级联）：GPU -> CPU（一级层） -> 磁盘（二级层，本模块）
#     加载（提升）：磁盘（二级层，本模块） -> CPU（一级层） -> GPU
#
#   本模块只负责 CPU 一级层 <-> 磁盘 之间的数据搬运，不直接接触 GPU 显存。
#
# 【核心组件协作关系】
#   FileSystemTierManager（调度器侧，本文件）
#     ├── FileMapper（文件路径映射）—— 将 OffloadKey 映射为磁盘文件路径
#     ├── DualQueueThreadPool（I/O 线程池）—— 执行异步磁盘读写
#     │     ├── JobState（任务完成追踪）—— 追踪每个 Job 的子任务完成状态
#     │     └── I/O 线程组（读优先 + 写优先）
#     └── store_block / load_block（底层 I/O 回调）—— 实际的磁盘读写操作
#
# 【非阻塞设计】
#   所有公共方法（submit_store、submit_load、get_finished_jobs）都在
#   Scheduler 线程中调用，必须非阻塞：
#   - submit_* 方法只做任务入队，不等待 I/O 完成。
#   - get_finished_jobs() 只从完成队列中取结果，不阻塞。
#   实际的磁盘 I/O 由 DualQueueThreadPool 中的后台线程执行。
#
# 【线程池的优先级策略】
#   - 读优先线程（n_read_threads 个）：优先处理 load（提升）任务，
#     空闲时也能处理 store（级联）任务。
#   - 写优先线程（n_write_threads 个）：优先处理 store 任务，
#     空闲时也能处理 load 任务。
#   - 这种设计保证：读取请求（影响用户推理延迟）获得优先服务，
#     同时写入任务不会被无限拖延（线程空闲时会回退处理）。
#
# 【存储路径设计】
#   文件命名遵循三级分层 + rank 隔离：
#     <base_path>_r<rank>/<hash前3位>/<hash第4-5位>_g<group_idx>/<完整hash>.bin
#   这样做的目的是：
#     1. 避免单目录文件数过多导致文件系统性能退化。
#     2. 不同 worker rank 的缓存互不干扰。
#     3. group_idx 支持多 KV cache group（如混合模型中 Mamba+Attention）。
#
# 【读写路径的原子性保证】
#   - 写入：先写临时文件（.tmp），再 os.replace 原子替换到目标路径。
#     保证读取方永远不会看到半写入的文件。
#   - 读取：使用 os.readv（scatter-gather I/O）直接读入 memoryview，
#     避免额外的内存拷贝。读取失败时删除可能损坏的源文件。
# =============================================================================

import functools
import json
import os
from collections.abc import Iterable
from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import OffloadKey, ReqContext
from vllm.v1.kv_offload.file_mapper import FileMapper
from vllm.v1.kv_offload.tiering.base import (
    JobMetadata,
    JobResult,
    RequestOffloadingContext,
    SecondaryTierManager,
)
from vllm.v1.kv_offload.tiering.fs.io import load_block, store_block
from vllm.v1.kv_offload.tiering.fs.thread_pool import DualQueueThreadPool

if TYPE_CHECKING:
    from vllm.v1.kv_offload.base import OffloadingSpec

logger = init_logger(__name__)


class FileSystemTierManager(SecondaryTierManager):
    """
    Pure-Python disk-backed secondary tier.

    Read-priority threads service load jobs preferentially; write-priority
    threads service store jobs preferentially.  Both groups can drain either
    queue, so neither starves.

    submit_store / submit_load are non-blocking: they enqueue tasks and return.
    get_finished_jobs() polls job completion and returns completed JobResults.

    """
    # 中文注释：FileSystemTierManager 是二级存储层管理器的文件系统实现。
    #
    # 继承自 SecondaryTierManager（抽象基类），实现其定义的生命周期方法：
    #   - on_new_request(): 新请求到达时创建卸载上下文。
    #   - lookup():       查询 block 是否已存在于磁盘上。
    #   - submit_store(): 提交异步级联存储（CPU -> 磁盘）。
    #   - submit_load():  提交异步提升加载（磁盘 -> CPU）。
    #   - get_finished_jobs(): 轮询已完成的异步任务。
    #   - shutdown():     关闭线程池，释放资源。
    #
    # 核心设计原则：
    #   1. 所有 submit 方法必须非阻塞（在 Scheduler 线程中运行）。
    #   2. 实际 I/O 由 DualQueueThreadPool 后台线程异步执行。
    #   3. 通过 primary_kv_view 与一级层交换数据，不直接访问 GPU。
    #   4. 使用 FileMapper 将 OffloadKey 映射为磁盘文件路径。

    def __init__(
        self,
        offloading_spec: "OffloadingSpec",
        primary_kv_view: memoryview,
        tier_type: str,
        root_dir: str,
        n_read_threads: int = 16,
        n_write_threads: int = 16,
    ):
        """
        Args:
            offloading_spec: contains the vllm_config, kv_cache_config
                and block_size_factor.
            primary_kv_view: Memoryview of the primary tier's CPU KV cache.
            tier_type: Tier type identifier, set by SecondaryTierFactory.
            root_dir: Root directory for block files.
            n_read_threads: Number of read-priority I/O threads.
            n_write_threads: Number of write-priority I/O threads.
        """
        # 中文注释：初始化参数详解
        #   offloading_spec: 卸载配置规范，包含 vllm_config、kv_cache_config
        #     以及 block_size_factor（卸载 block 大小 / GPU block 大小的比值）。
        #   primary_kv_view: 一级层 CPU KV cache 的 memoryview 视图。
        #     二级层通过此视图直接读写一级层的 block 数据。
        #     这是二级层与一级层交换数据的唯一通道（不经过 GPU）。
        #   tier_type: 二级层类型标识（如 "fs"），由 SecondaryTierFactory 设定。
        #   root_dir: 磁盘上 KV block 文件的根目录。
        #   n_read_threads: 读优先 I/O 线程数，默认 16。
        #     这些线程优先处理 load（提升）任务，空闲时也能处理 store 任务。
        #   n_write_threads: 写优先 I/O 线程数，默认 16。
        #     这些线程优先处理 store（级联）任务，空闲时也能处理 load 任务。
        super().__init__(offloading_spec, primary_kv_view, tier_type)

        # Extract block size from primary view
        # 中文注释：从 primary_kv_view 的 strides 中提取单个 block 的字节大小。
        # strides[0] 表示第一维（即 block 维度）的步长，等于一个 block 的字节数。
        # 这个值在 store/load 时用于计算 buffer 中的偏移：offset = block_id * _block_size。
        assert primary_kv_view.strides is not None, (
            "primary_kv_view.strides cannot be None"
        )
        self._block_size: int = primary_kv_view.strides[0]

        # Create file mapper
        # 中文注释：创建文件路径映射器。
        # FileMapper 负责将 OffloadKey（block 的唯一标识）映射为磁盘上的文件路径。
        # 路径格式：{base_path}_r{rank}/{hash前3位}/{hash第4-5位}_g{group}/{完整hash}.bin
        # gpu_blocks_per_file 参数在此场景下等于 block_size_factor，
        # 表示一个卸载文件中包含多少个 GPU block 的数据。
        self.file_mapper = FileMapper.from_offloading_spec(
            root_dir=root_dir,
            offloading_spec=offloading_spec,
            gpu_blocks_per_file=offloading_spec.block_size_factor,
        )

        # Write config file
        # 中文注释：将运行配置写入 config.json 文件。
        # config.json 位于 base_path 根目录下（不含 rank 后缀），所有 rank 共享。
        # 记录了模型名、block 大小、并行度等参数，用于后续加载时验证配置一致性。
        # 只在文件不存在时写入（幂等性），避免重启后覆盖已有配置。
        config_path = self.file_mapper.get_config_file_path()
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        if not os.path.exists(config_path):
            with open(config_path, "w") as f:
                json.dump(
                    self.file_mapper.get_run_config(), f, indent=2, sort_keys=True
                )

        # 中文注释：创建双队列 I/O 线程池。
        # DualQueueThreadPool 内部维护两个任务队列（load_q 和 store_q）和
        # 两类优先级线程（读优先和写优先）。
        # 读优先线程优先处理 load（磁盘->CPU）任务，写优先线程优先处理 store 任务。
        # 两者都可以在主队列为空时"借用"对方队列，避免线程饥饿。
        self._pool = DualQueueThreadPool(
            n_read_threads,
            n_write_threads,
            thread_name_prefix="vllm_kv_py_fs",
        )

    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext:
        # 中文注释：新请求到达时的回调。
        #
        # 返回 RequestOffloadingContext，其中 policy 默认为 BLOCK_LEVEL。
        # BLOCK_LEVEL 策略的含义：只卸载新计算的 block（跳过 prefix cache 已命中的 block）。
        # 这对于文件系统二级层是合理的——如果某个 block 已经存在于磁盘上
        # （由之前的请求存储过），就没有必要再次存储。
        #
        # 对比 REQUEST_LEVEL 策略（如分布式 KV 传输场景需要），
        # 会卸载请求的所有 block（包括 prefix-hit 的），适用于需要完整 KV 上下文的场景。
        return RequestOffloadingContext()

    def lookup(
        self, key: OffloadKey, req_context: ReqContext | None = None
    ) -> bool | None:
        # 中文注释：查询指定 block 是否已存在于磁盘上。
        #
        # 执行逻辑：
        #   1. 通过 FileMapper 将 OffloadKey 映射为磁盘文件路径。
        #   2. 检查该文件是否存在（os.path.exists）。
        #
        # 返回值语义：
        #   True  — block 文件存在且可用，可以立即发起提升（promote）。
        #   False — block 文件不存在，需要从 GPU 重新计算后级联存储。
        #   None  — block 正在传输中（本实现不使用此返回值，因为文件系统
        #           的查找操作是瞬时的，不存在"正在传输"的中间状态）。
        #
        # 调用时机：Scheduler 在每轮调度前查询 block 可用性。
        # 如果一级层（CPU）未命中，会继续查询二级层（磁盘/本实现）。
        # 命中后 Scheduler 会调用 submit_load() 发起异步提升。
        return os.path.exists(self.file_mapper.get_file_name(key))

    def submit_store(self, job_metadata: JobMetadata) -> None:
        # 中文注释：提交异步级联存储任务 —— 将 KV block 从 CPU 一级层写入磁盘。
        #
        # 执行流程：
        #   1. 遍历 job_metadata 中的每个 (key, block_id) 对。
        #   2. 为每个 block 构造一个 functools.partial 回调（store_block），
        #      封装了：目标文件路径、源 memoryview、偏移量、block 大小。
        #   3. 将所有 per-block 子任务批量入队到写优先队列（store_q）。
        #   4. 线程池中的 I/O 线程会异步执行这些任务。
        #
        # 数据流：
        #   CPU 一级层 memoryview[bid * block_size : (bid+1) * block_size]
        #     -> store_block() -> 临时文件 -> os.replace -> 目标文件
        #
        # 关键参数解析：
        #   - job_metadata.keys: 需要存储的 block 的 OffloadKey 集合。
        #   - job_metadata.block_ids: 一级层中对应的物理 block ID 数组。
        #     通过 block_id * block_size 计算出在 primary_kv_view 中的字节偏移。
        #   - self._primary_kv_view: 一级层 CPU KV cache 的 memoryview，
        #     store_block 从此处读取数据。
        #
        # 此方法在 Scheduler 线程中调用，只做任务入队，不阻塞调度流程。
        # 实际的磁盘写入由 DualQueueThreadPool 的后台线程异步执行。
        tasks = (
            functools.partial(
                store_block,
                self.file_mapper.get_file_name(key),
                self._primary_kv_view,
                int(bid) * self._block_size,
                self._block_size,
            )
            for key, bid in zip(job_metadata.keys, job_metadata.block_ids)
        )
        self._pool.enqueue_store(job_metadata.job_id, len(job_metadata.keys), tasks)

    def submit_load(self, job_metadata: JobMetadata) -> None:
        # 中文注释：提交异步提升加载任务 —— 将 KV block 从磁盘读回 CPU 一级层。
        #
        # 执行流程：
        #   1. 遍历 job_metadata 中的每个 (key, block_id) 对。
        #   2. 为每个 block 构造一个 functools.partial 回调（load_block），
        #      封装了：源文件路径、目标 memoryview、偏移量、block 大小。
        #   3. 将所有 per-block 子任务批量入队到读优先队列（load_q）。
        #   4. 读优先线程会优先处理这些任务，以最小化推理延迟。
        #
        # 数据流：
        #   磁盘文件 -> load_block() -> os.readv
        #     -> CPU 一级层 memoryview[bid * block_size : (bid+1) * block_size]
        #
        # 关键参数解析：
        #   - job_metadata.keys: 需要加载的 block 的 OffloadKey 集合。
        #   - job_metadata.block_ids: 一级层中已分配的物理 block ID 数组。
        #     数据将被写入这些 block 对应的 memoryview 位置。
        #   - self._primary_kv_view: 一级层 CPU KV cache 的 memoryview，
        #     load_block 将数据写入此视图的对应偏移位置。
        #
        # 此方法在 Scheduler 线程中调用，只做任务入队，不阻塞调度流程。
        # 加载完成后，框架通过 get_finished_jobs() 感知完成状态，
        # 并将一级层 block 标记为可用，供后续 GPU 加载。
        tasks = (
            functools.partial(
                load_block,
                self.file_mapper.get_file_name(key),
                self._primary_kv_view,
                int(bid) * self._block_size,
                self._block_size,
            )
            for key, bid in zip(job_metadata.keys, job_metadata.block_ids)
        )
        self._pool.enqueue_load(job_metadata.job_id, len(job_metadata.keys), tasks)

    def get_finished_jobs(self) -> Iterable[JobResult]:
        """
        Collect completed jobs from the finished-jobs queue.
        """
        # 中文注释：获取自上次调用以来完成的所有异步传输任务。
        #
        # 此方法在每个 engine step 中被框架调用（至少一次），用于轮询
        # I/O 线程池中已完成的任务。
        #
        # 工作流程：
        #   1. 从 DualQueueThreadPool 的 finished_q 中取出所有已完成的
        #      (job_id, success) 元组。
        #   2. 将每个元组包装为 JobResult 对象返回给框架。
        #
        # 框架收到 JobResult 后的处理：
        #   - 对于级联存储（store）完成：释放一级层 block 的引用计数，
        #     表示这些 block 已安全持久化到磁盘，一级层可以回收。
        #   - 对于提升加载（load）完成：标记一级层 block 数据可用，
        #     后续可从一级层加载到 GPU 供模型推理使用。
        #   - 对于失败（success=False）：框架进行资源清理（如释放引用计数、
        #     标记 block 为无效），并可能触发重试或跳过该 block。
        return (
            JobResult(job_id=job_id, success=success)
            for job_id, success in self._pool.get_finished()
        )

    def shutdown(self) -> None:
        """
        Release resources held by this tier.

        Shuts down the thread pool, clearing pending tasks and waiting for
        active threads to complete.
        """
        # 中文注释：关闭二级层管理器，释放所有资源。
        #
        # 执行流程：
        #   1. 设置线程池的停止标志（_stop = True）。
        #   2. 清空 load_q 和 store_q 中尚未执行的任务。
        #   3. 唤醒所有等待中的 I/O 线程，使其检测到停止标志并退出。
        #   4. 等待所有活跃线程完成当前正在执行的任务后退出（wait=True）。
        #
        # 调用时机：vLLM 引擎关闭时调用，确保所有后台 I/O 线程正确终止，
        # 不会有线程泄漏或资源悬挂。
        self._pool.shutdown(wait=True)
