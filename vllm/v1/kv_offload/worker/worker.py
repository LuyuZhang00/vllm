# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
KV cache offloading worker 模块。

中文注释：本模块是 vLLM v1 KV cache offloading 架构中的"执行层"，
负责在 Worker 进程中实际执行 KV 数据的异步搬运（如 GPU <-> CPU 内存）。

整体架构回顾（三层设计）：
  1. OffloadingSpec（规范层，在 base.py 中）—— 定义 offloading 配置参数。
  2. OffloadingManager（管理层，在 base.py 中）—— 运行在 Scheduler 进程，
     负责追踪哪些 block 已被卸载、管理 LRU 淘汰策略、协调 load/store 生命周期。
  3. OffloadingHandler / OffloadingWorker（执行层，本模块）—— 运行在 Worker 进程，
     负责实际的数据搬运。

本模块的关键设计：
  - OffloadingHandler 是抽象基类，定义了单个传输通道（如 GPU->CPU）的接口，
    包括发起异步传输（transfer_async）、查询已完成传输（get_finished）、
    阻塞等待特定传输完成（wait）。
  - OffloadingWorker 是聚合器，管理多个 OffloadingHandler 实例，
    根据 transfer type（src_medium, dst_medium）将传输请求路由到对应的 handler。
    例如：("GPU", "CPU") 类型的传输走 GPU->CPU handler，
    ("CPU", "GPU") 类型的传输走 CPU->GPU handler。
  - TransferResult 是传输完成后的结果数据类，包含 job_id、是否成功、
    传输大小和耗时等信息。

数据流：
  Scheduler 通过 OffloadingManager 准备好 LoadStoreSpec（包含源/目标地址信息），
  然后将 TransferSpec = (src_spec, dst_spec) 传递给 Worker 端的 OffloadingWorker，
  OffloadingWorker 根据 (src.medium(), dst.medium()) 选择对应的 OffloadingHandler
  来执行实际的异步数据传输。
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import LoadStoreSpec

# 中文注释：单次传输规范，由源端和目标端的 LoadStoreSpec 组成。
# 例如 src 是 GPU 的 KV block 地址信息，dst 是 CPU 内存的地址信息。
# a single transfer spec (src_blocks_spec, dst_blocks_spec)
TransferSpec = tuple[LoadStoreSpec, LoadStoreSpec]
# 中文注释：传输类型标识，由 (源存储介质名, 目标存储介质名) 构成。
# 用于将传输请求路由到对应的 OffloadingHandler。
# 例如 ("GPU", "CPU") 表示从 GPU 到 CPU 的卸载，("CPU", "GPU") 表示从 CPU 到 GPU 的加载。
# transfers are forwarded to workers by (src_medium, dst_medium)
TransferType = tuple[str, str]

logger = init_logger(__name__)


@dataclass
class TransferResult:
    """
    中文注释：传输结果数据类，记录一次异步 KV 数据传输的完成状态。

    每次传输由唯一的 job_id 标识，传输完成后会生成一个 TransferResult，
    上层（如 OffloadingManager）可以通过 job_id 将结果与原始传输请求关联起来，
    从而更新内部状态（如标记 block 为已卸载/已加载）。

    字段说明：
      - job_id: 传输任务的唯一标识符，与发起传输时传入的 job_id 对应。
      - success: 传输是否成功完成。
      - transfer_size: 传输的数据量（字节数），可用于统计和监控。
      - transfer_time: 传输耗时（秒），可用于性能分析。
      - transfer_type: 传输类型 (src_medium, dst_medium)，用于区分不同方向的传输。
    """
    job_id: int
    success: bool
    transfer_size: int | None = None  # Size in bytes
    transfer_time: float | None = None
    transfer_type: TransferType | None = None


class OffloadingHandler(ABC):
    """
    OffloadingHandler class for managing asynchronous KV data transfers

    This class runs in the worker.
    It kicks off async KV data transfer requests, and allows
    collecting back completion statuses.

    The class provides the following primitives:
        transfer_async() - kicks off a new transfer job
        get_finished() - returns a list of newly finished job IDs.

    中文注释：OffloadingHandler 是单个传输通道的抽象基类，运行在 Worker 进程中。

    设计要点：
      1. 每个 OffloadingHandler 负责一种特定方向的异步数据传输，
         例如 GPU->CPU 的卸载或 CPU->GPU 的加载。
      2. 传输是异步的：transfer_async() 提交传输后立即返回，
         调用方通过 get_finished() 轮询或 wait() 阻塞等待来获取完成状态。
      3. 具体的传输实现（如使用 cudaMemcpyAsync、NCCL、自定义 RDMA 等）
         由子类负责，本基类只定义接口契约。

    提供的原语操作：
      - transfer_async(): 发起一个新的异步传输任务
      - get_finished(): 返回自上次调用以来新完成的传输结果列表
      - wait(): 阻塞等待指定的传输任务完成
      - shutdown(): 关闭 handler，释放底层资源
    """

    @abstractmethod
    def transfer_async(self, job_id: int, spec: TransferSpec) -> bool:
        """
        Initiates an asynchronous transfer of KV data.

        Args:
            job_id: a unique ID that will be used when notifying back on
                transfer completion.
            spec: the (src, dst) spec of the KV data transfer.

        Returns:
            True if transfer was submitted successfully.

        中文注释：发起一个异步 KV 数据传输任务。
          - job_id: 传输任务的唯一标识，完成时会通过 get_finished() 返回此 ID。
          - spec: 包含源端和目标端地址信息的传输规范 (src_spec, dst_spec)。
          - 返回 True 表示传输已成功提交到异步队列。
        """
        pass

    @abstractmethod
    def get_finished(self) -> list[TransferResult]:
        """
        Get transfers finished since last call.

        Returns:
            A list of (job_id, success) of transfers.

        中文注释：获取自上次调用以来新完成的传输任务结果。
          返回 TransferResult 列表，每个元素包含 job_id 和成功/失败状态。
          调用方（如 OffloadingManager）可据此更新内部状态。
        """
        pass

    @abstractmethod
    def wait(self, job_ids: set[int]) -> None:
        """
        Wait for jobs to finish (blocking).
        Args:
            job_ids: The set of job IDs to wait for.

        中文注释：阻塞等待指定的一组传输任务全部完成。
          在某些需要确保数据已到位的场景下使用（如 shutdown 前清理）。
        """

    def shutdown(self) -> None:
        """Shutdown the handler and release any resources."""
        return


class OffloadingWorker:
    """
    OffloadingWorker class for managing asynchronous KV data transfers
    using multiple OffloadingHandlers

    This class runs in the worker.
    It kicks off async KV data transfer requests, by delegating
    to one of its registered OffloadingHandlers, based on the transfer type.

    The class provides the following primitives:
        register_handler() - registers a new handler to handle
            a specific transfer type
        transfer_async() - kicks off a new transfer job
            using one of the registered handlers.
        get_finished() - returns a list of newly finished job IDs
            from all handlers.

    中文注释：OffloadingWorker 是 Worker 进程中的传输聚合器，
    管理多个 OffloadingHandler，根据传输类型（src_medium, dst_medium）
    将传输请求路由到对应的 handler。

    设计要点：
      1. 一个 Worker 进程通常只有一个 OffloadingWorker 实例。
      2. OffloadingWorker 内部维护一个 handler 注册表，
         key 是 TransferType = (src_medium, dst_medium)，value 是对应的 handler。
         例如：("GPU", "CPU") -> GPU->CPU handler，
               ("CPU", "GPU") -> CPU->GPU handler。
      3. 发起传输时，根据 spec 中 src 和 dst 的 medium() 类型
         查找对应的 handler 并委托执行。
      4. get_finished() 会聚合所有 handler 的完成结果，避免调用方逐个查询。
      5. 生命周期管理：register_handler() 在初始化阶段注册，
         shutdown() 按序关闭所有 handler。

    与 OffloadingManager 的关系：
      - OffloadingManager（Scheduler 端）负责元数据管理和决策
        （哪些 block 需要卸载/加载、LRU 淘汰等）。
      - OffloadingWorker（Worker 端）负责实际执行数据搬运。
      - 两者通过 SchedulerOutput / Worker 通信链路协作。
    """

    def __init__(self):
        # 中文注释：所有已注册的 handler 集合，用于遍历（如 get_finished、shutdown）。
        self.handlers: set[OffloadingHandler] = set()
        # 中文注释：传输类型到 handler 的映射表。
        # key 是 (src_medium, dst_medium)，例如 ("GPU", "CPU")。
        # transfer_async() 通过此表查找应该委托哪个 handler 执行传输。
        self.transfer_type_to_handler: dict[TransferType, OffloadingHandler] = {}

    def register_handler(
        self,
        src_cls: type[LoadStoreSpec],
        dst_cls: type[LoadStoreSpec],
        handler: OffloadingHandler,
    ) -> None:
        """
        Registers a new handler.

        Args:
            src_cls: the source type of transfers handled by this handler.
            dst_cls: the destination type of transfers handled by this handler.
            handler: the handler that will handle transfers.

        中文注释：注册一个新的传输 handler。
          通过 src_cls.medium() 和 dst_cls.medium() 确定传输类型，
          例如注册一个 GPU->CPU 的 handler。
          断言确保同一传输类型不会重复注册。
          此方法通常在 Worker 初始化阶段（get_handlers() 返回后）调用。
        """
        transfer_type = (src_cls.medium(), dst_cls.medium())
        assert transfer_type not in self.transfer_type_to_handler
        self.handlers.add(handler)
        self.transfer_type_to_handler[transfer_type] = handler

    def transfer_async(self, job_id: int, spec: TransferSpec) -> bool:
        """
        Initiates an asynchronous transfer of KV data.

        Args:
            job_id: a unique ID that will be used when notifying back on
                transfer completion.
            spec: the (src, dst) spec of the KV data transfer.

        Returns:
            True if transfer was submitted successfully.

        中文注释：发起一个异步 KV 数据传输。
          执行流程：
            1. 从 spec 中提取源端和目标端，确定 transfer_type。
            2. 根据 transfer_type 从注册表中查找对应的 handler。
            3. 委托 handler 发起实际的异步传输。
            4. 异常处理：如果 handler 抛出异常，捕获并记录日志，
               返回 False 表示提交失败（不影响其他传输）。
            5. 日志记录：成功时记录 debug 日志，失败时记录 warning 日志。
        """
        src, dst = spec
        transfer_type = (src.medium(), dst.medium())
        handler = self.transfer_type_to_handler.get(transfer_type)
        assert handler is not None
        try:
            success = handler.transfer_async(job_id, spec)
        except Exception as e:
            logger.warning(
                "Exception in %r transfer %d: %r",
                transfer_type,
                job_id,
                e,
                exc_info=True,
            )
            return False

        if not success:
            logger.warning("Failed to submit %r transfer %d", transfer_type, job_id)
        else:
            logger.debug("Submitted %r transfer %d: %r", transfer_type, job_id, spec)
        return success

    def get_finished(self) -> list[TransferResult]:
        """
        Get transfers finished since last call.

        Returns:
            A list of TransferResults

        中文注释：聚合查询所有 handler 的已完成传输结果。
          遍历所有已注册的 handler，收集自上次调用以来新完成的传输。
          返回的 TransferResult 列表中可能包含不同传输类型的结果
          （如 GPU->CPU 和 CPU->GPU 混合），调用方通过 job_id
          和 transfer_type 字段区分。
        """
        finished = []
        for handler in self.handlers:
            finished.extend(handler.get_finished())
        return finished

    def wait(self, job_ids: set[int]) -> None:
        """
        Wait for jobs to finish (blocking).

        Args:
            job_ids: The set of job IDs to wait for.

        中文注释：阻塞等待指定的一组传输任务全部完成。
          遍历所有 handler，逐个调用 wait()。
          注意：每个 handler 会过滤掉不属于自己的 job_id，
          因此可以安全地将所有 job_id 传给每个 handler。
        """
        for handler in self.handlers:
            handler.wait(job_ids)

    def shutdown(self) -> None:
        """
        中文注释：关闭 OffloadingWorker，释放所有 handler 的资源。
          通常在 Worker 进程退出时调用，确保异步传输队列被清理、
          底层 CUDA stream 或线程被正确销毁。
        """
        for handler in self.handlers:
            handler.shutdown()
