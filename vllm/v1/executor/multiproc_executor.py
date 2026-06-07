# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# =============================================================================
# 模块概述: 多进程执行器 (MultiprocExecutor)
# =============================================================================
#
# 本模块实现了 vLLM 的多进程执行器，是 vLLM 的主要生产环境执行器。
#
# 核心架构:
# ┌─────────────────────────────────────────────────────────────────┐
# │  Executor 进程 (EngineCore 所在进程)                              │
# │  ┌───────────────────────────────────────────────────────────┐  │
# │  │  MultiprocExecutor                                        │  │
# │  │  ├── rpc_broadcast_mq (共享内存, 单写多读广播)              │  │
# │  │  │   → 广播 (method, args, kwargs) 给所有 Worker           │  │
# │  │  │                                                        │  │
# │  │  ├── response_mqs[0..N] (共享内存, 单写单读)                │  │
# │  │  │   ← 每个 Worker 返回结果                                │  │
# │  │  │                                                        │  │
# │  │  └── futures_queue (异步结果队列)                           │  │
# │  └───────────────────────────────────────────────────────────┘  │
# └─────────────────────────────────────────────────────────────────┘
#        │ rpc_broadcast_mq (共享内存)           │ response_mqs
#        ▼                                       ▲
# ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
# │  Worker 0    │  │  Worker 1    │  │  Worker N    │
# │  (GPU 0)     │  │  (GPU 1)     │  │  (GPU N)     │
# │              │  │              │  │              │
# │ worker_busy_ │  │ worker_busy_ │  │ worker_busy_ │
# │ loop()       │  │ loop()       │  │ loop()       │
# └──────────────┘  └──────────────┘  └──────────────┘
#
# 主要类:
# 1. FutureWrapper: 异步 Future 包装器，支持顺序结果收集
# 2. MultiprocExecutor: 多进程执行器，管理 Worker 进程池
# 3. UnreadyWorkerProcHandle: Worker 进程句柄 (就绪前)
# 4. WorkerProcHandle: Worker 进程句柄 (就绪后)
# 5. WorkerProc: Worker 进程包装器，在独立进程中运行 GPU Worker
#
# 通信机制:
# 1. Executor → Workers: rpc_broadcast_mq (单写多读广播)
#    - Executor 写一次，所有 Worker 都能读
#    - 零拷贝: 大张量通过共享内存直接传递
#
# 2. Workers → Executor: response_mqs (每 Worker 一个)
#    - 每个 Worker 有独立的 response_mq
#    - 只有 output_rank 指定的 Worker 需要返回结果
#
# 生命周期:
# 1. Executor 进程启动，创建 MultiprocExecutor 实例
# 2. MultiprocExecutor 创建 Worker 子进程池
# 3. 每个 Worker 子进程初始化 GPU、加载模型
# 4. Worker 发送 READY 信号，进入 worker_busy_loop 等待命令
# 5. Executor 通过 collective_rpc 广播命令给所有 Worker
# 6. Worker 执行命令，将结果放入 response_mq
# 7. Executor 从 response_mq 收集结果并返回
# 8. 关闭时，Executor 通知所有 Worker 退出，释放资源
#
# =============================================================================

# =============================================================================
# 标准库导入
# =============================================================================
# multiprocessing: Python 多进程库，用于创建和管理 Worker 子进程
import multiprocessing
# os: 操作系统接口，用于环境变量和文件描述符管理
import os
# pickle: 序列化库，用于 RPC 方法的序列化
import pickle
# queue: 线程安全队列，用于异步输出处理
import queue
# signal: 信号处理，用于优雅终止 Worker 进程
import signal
# threading: 线程库，用于 Worker 监控和异步输出线程
import threading
# time: 时间库，用于超时控制和性能计时
import time
# traceback: 追踪库，用于异常堆栈记录
import traceback
# weakref: 弱引用库，用于 Executor 析构函数，避免循环引用
import weakref
# collections: 集合库，deque 用于 FIFO 未来队列
from collections import deque
# collections.abc: 抽象基类，用于类型标注
from collections.abc import Callable, Sequence
# concurrent.futures: 并发库，Future 用于异步结果
from concurrent.futures import Future, InvalidStateError
# contextlib: 上下文管理工具，suppress 用于忽略特定异常
from contextlib import suppress
# dataclass: 数据类装饰器，用于 Worker 进程句柄
from dataclasses import dataclass
# enum: 枚举库，ResponseStatus 用于标记 Worker 响应状态
from enum import Enum, auto
# functools: 函数工具，cached_property 和 partial 用于优化
from functools import cached_property, partial
# multiprocessing.connection: 进程间通信管道
from multiprocessing.connection import Connection
# multiprocessing.process: 进程基类
from multiprocessing.process import BaseProcess
# multiprocessing.synchronize: 进程同步原语 (锁)
from multiprocessing.synchronize import Lock as LockType
# threading.Thread: 守护线程，用于死亡监控和异步输出
from threading import Thread
# typing: 类型标注
from typing import Any, cast

# =============================================================================
# 第三方库导入
# =============================================================================
# cloudpickle: 增强版 pickle，支持序列化 lambda 和闭包
# 用于序列化无法用标准 pickle 序列化的 RPC 方法
import cloudpickle
# torch: PyTorch 库，用于线程并行度设置
import torch

# =============================================================================
# vLLM 内部模块导入
# =============================================================================
# envs: vLLM 环境变量管理
import vllm.envs as envs
# VllmConfig: vLLM 统一配置对象
from vllm.config import VllmConfig
# 分布式环境销毁函数
from vllm.distributed import destroy_distributed_environment, destroy_model_parallel
# Handle 和 MessageQueue: 共享内存消息队列的核心类
# Handle: 消息队列的句柄，用于跨进程共享
# MessageQueue: 基于共享内存的高效消息队列实现
from vllm.distributed.device_communicators.shm_broadcast import Handle, MessageQueue
# KVOutputAggregator: KV 缓存输出聚合器
from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
# 并行组管理: 获取各种并行维度的组信息
from vllm.distributed.parallel_state import (
    get_dcp_group,
    get_dp_group,
    get_ep_group,
    get_inner_dp_world_group,
    get_pcp_group,
    get_pp_group,
    get_tp_group,
    model_parallel_is_initialized,
)
# enable_envs_cache: 启用环境变量缓存 (优化性能)
from vllm.envs import enable_envs_cache
# init_logger: vLLM 日志初始化
from vllm.logger import init_logger
# current_platform: 当前平台抽象 (CUDA/ROCm/CPU/TPU)
from vllm.platforms import current_platform
# instrument: 分布式追踪装饰器
from vllm.tracing import instrument, maybe_init_worker_tracer
# NUMA 绑定工具
from vllm.utils import numa_utils
# 网络工具: 分布式初始化方法、IP 获取等
from vllm.utils.network_utils import (
    get_distributed_init_method,
    get_ip,
    get_loopback_ip,
    get_open_port,
)
# OMPProcessManager: OpenMP 线程管理器
from vllm.utils.ompmultiprocessing import OMPProcessManager
# 系统工具: 进程管理、日志装饰等
from vllm.utils.system_utils import (
    _maybe_force_spawn,
    decorate_logs,
    get_mp_context,
    set_process_title,
)
# SchedulerOutput: 调度器输出，包含模型执行所需的批次信息
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
# Executor: 执行器基类，定义了 execute_model 等核心接口
from vllm.v1.executor.abstract import Executor, FailureCallback
# set_worker_net_device: 设置 Worker 网络设备
from vllm.v1.executor.vllm_net_devices import set_worker_net_device
# 模型输出类型
from vllm.v1.outputs import AsyncModelRunnerOutput, DraftTokenIds, ModelRunnerOutput
# WorkerWrapperBase: Worker 包装器基类，延迟加载实际 Worker 类
from vllm.v1.worker.worker_base import WorkerWrapperBase

# 模块级日志器
logger = init_logger(__name__)


# =============================================================================
# FutureWrapper: 异步 Future 包装器
# =============================================================================
# 核心设计思想:
#   vLLM 的 execute_model 调用是非阻塞的: Executor 发送命令后立即返回 Future。
#   当 Scheduler 需要获取结果时，才调用 future.result() 阻塞等待。
#
# 顺序保证:
#   由于 Executor 和 Worker 之间是单通道通信 (rpc_broadcast_mq)，
#   Worker 按照接收顺序执行命令。因此 Future 的结果也必须按顺序收集。
#   FutureWrapper 使用 FIFO 队列 (futures_queue) 来保证这一点:
#   - 每次创建 FutureWrapper 时，将自身加入队列头部 (appendleft)
#   - 调用 result() 时，从队列尾部依次取出并等待结果 (pop)
#   - 这样确保了先发出的命令先被收集结果
#
# 示例:
#   futures_queue: [F3, F2, F1]  (F3 最新，F1 最旧)
#   F1.result() → pop F1, wait → F2, wait → F3, wait
#   F2.result() → (F1 已完成) → pop F2, wait → F3, wait
#   F3.result() → (F1, F2 已完成) → pop F3, wait
class FutureWrapper(Future):
    """
    异步 Future 包装器 —— 支持顺序结果收集的 Future 子类。

    继承自 concurrent.futures.Future，增加了:
    1. futures_queue: FIFO 队列，保证结果按顺序收集
    2. get_response: 实际获取响应的回调函数
    3. aggregate: 结果聚合函数 (用于 KV 输出聚合)

    为什么需要这个类:
    - 标准 Future 不保证多个 Future 的结果收集顺序
    - vLLM 的 Worker 按命令接收顺序执行，结果也必须按顺序收集
    - FutureWrapper 通过 FIFO 队列强制保证顺序
    """

    def __init__(
        self,
        futures_queue: deque["FutureWrapper"],
        get_response: Callable[[], Any],
        aggregate: Callable = lambda x: x,
    ):
        # futures_queue: 共享的 FIFO 队列，所有 FutureWrapper 实例共享同一个队列
        self.futures_queue = futures_queue
        # get_response: 回调函数，实际从 MessageQueue 中 dequeue 结果
        self.get_response = get_response
        # aggregate: 结果聚合函数，用于 KV 输出聚合场景
        self.aggregate = aggregate
        super().__init__()
        # 将自身加入队列头部 (最新创建的 Future 在队列头部)
        self.futures_queue.appendleft(self)

    def result(self, timeout=None):
        """
        获取异步结果 —— 阻塞等待直到结果可用。

        顺序保证机制:
        1. 从队列尾部 (最旧) 开始取出 Future
        2. 依次等待每个 Future 的结果
        3. 确保先发出的命令先被处理

        参数:
            timeout: 超时时间 (秒)，当前未实现

        返回:
            异步命令的执行结果
        """
        if timeout is not None:
            raise RuntimeError("timeout not implemented")

        # 排空队列中在我们之前的所有 future (确保顺序执行)
        # 从队列尾部开始 (最旧的 Future)
        while not self.done():
            future = self.futures_queue.pop()
            future._wait_for_response()
        return super().result()

    def _wait_for_response(self):
        """
        内部方法: 实际等待并获取响应。

        流程:
        1. 调用 get_response() 从 MessageQueue 中 dequeue
        2. 如果配置了 aggregate，应用聚合函数
        3. 将结果设置到 Future 中 (set_result)
        4. 如果发生异常，设置异常 (set_exception)
        """
        try:
            # 从 MessageQueue 获取响应并应用聚合
            response = self.aggregate(self.get_response())
            # suppress(InvalidStateError): 忽略重复设置结果的异常
            # 这可能在某些竞态条件下发生
            with suppress(InvalidStateError):
                self.set_result(response)
        except Exception as e:
            with suppress(InvalidStateError):
                self.set_exception(e)


class MultiprocExecutor(Executor):
    """
    多进程执行器 —— vLLM 的主要生产环境执行器

    架构:
    ┌─────────────────────────────────────────────────────────┐
    │  Executor 进程 (EngineCore 所在进程)                      │
    │  ┌───────────────────────────────────────────────────┐  │
    │  │  MultiprocExecutor                                │  │
    │  │  ├── rpc_broadcast_mq (共享内存, 单写多读广播)      │  │
    │  │  │   → 广播 (method, args, kwargs) 给所有 Worker   │  │
    │  │  │                                                │  │
    │  │  ├── response_mqs[0..N] (共享内存, 单写单读)        │  │
    │  │  │   ← 每个 Worker 返回结果                        │  │
    │  │  │                                                │  │
    │  │  └── futures_queue (异步结果队列)                   │  │
    │  └───────────────────────────────────────────────────┘  │
    └─────────────────────────────────────────────────────────┘
           │ rpc_broadcast_mq (共享内存)           │ response_mqs
           ▼                                       ▲
    ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
    │  Worker 0    │  │  Worker 1    │  │  Worker N    │
    │  (GPU 0)     │  │  (GPU 1)     │  │  (GPU N)     │
    │              │  │              │  │              │
    │ worker_busy_ │  │ worker_busy_ │  │ worker_busy_ │
    │ loop()       │  │ loop()       │  │ loop()       │
    └──────────────┘  └──────────────┘  └──────────────┘

    通信机制:
    1. Executor → Workers: rpc_broadcast_mq (单写多读广播)
       - Executor enqueue 一次，所有 Worker 都能 dequeue
       - 零拷贝: 大张量通过共享内存直接传递

    2. Workers → Executor: response_mqs (每 Worker 一个)
       - 每个 Worker 有独立的 response_mq
       - 只有 output_rank 指定的 Worker 需要返回结果
    """

    # 标记此类支持流水线并行 (Pipeline Parallelism)
    supports_pp: bool = True

    def __init__(self, vllm_config: VllmConfig, monitor_workers: bool = True):
        """
        MultiprocExecutor 构造函数。

        参数:
            vllm_config: vLLM 统一配置对象，包含模型、并行、调度等所有配置
            monitor_workers: 是否启动 Worker 健康监控线程
                - True (默认): 启动守护线程监控 Worker 存活状态
                - False: 不启动监控，用于测试或特殊场景
        """
        # 是否监控 Worker 进程存活状态
        self.monitor_workers = monitor_workers
        # 调用父类 Executor.__init__()，它会调用 _init_executor()
        super().__init__(vllm_config)

    def _init_executor(self) -> None:
        """
        初始化执行器的核心方法。

        流程:
        ① 验证并行配置 (TP × PP × PCP = world_size)
        ② 创建共享内存广播队列 (rpc_broadcast_mq)
        ③ 为每个 GPU 创建 Worker 子进程
        ④ 等待所有 Worker 就绪
        ⑤ 启动 Worker 健康监控线程
        ⑥ 收集 Worker 的响应队列 (response_mqs)
        """
        # 注册析构函数，确保进程退出时清理 Worker
        self._finalizer = weakref.finalize(self, self.shutdown)
        self.is_failed = False
        self.failure_callback: FailureCallback | None = None

        # ① 验证并行配置
        tp_size, pp_size, pcp_size = self._get_parallel_sizes()
        assert self.world_size == tp_size * pp_size * pcp_size, (
            f"world_size ({self.world_size}) must be equal to the "
            f"tensor_parallel_size ({tp_size}) x pipeline"
            f"_parallel_size ({pp_size}) x prefill_context"
            f"_parallel_size ({pcp_size}). "
        )
        assert self.world_size == tp_size * pp_size * pcp_size, (
            f"world_size ({self.world_size}) must be equal to the "
            f"tensor_parallel_size ({tp_size}) x pipeline"
            f"_parallel_size ({pp_size}) x prefill_context"
            f"_parallel_size ({pcp_size}). "
        )

        set_multiprocessing_worker_envs()

        # ② 创建分布式初始化方法 (使用 loopback IP)
        distributed_init_method = get_distributed_init_method(
            get_loopback_ip(), get_open_port()
        )
        self.rpc_broadcast_mq: MessageQueue | None = None
        scheduler_output_handle: Handle | None = None

        # ② 创建共享内存广播队列 (仅 DP 主节点)
        # rpc_broadcast_mq: Executor → Workers 的命令广播通道
        # - 单写多读: Executor 写一次，所有 Worker 都能读
        # - 用于广播 SchedulerOutput 和 RPC 调用
        if self.parallel_config.node_rank_within_dp == 0:
            # 每个 DP rank 的主节点，每个 DP 都有自己的主多进程执行器
            max_chunk_bytes = envs.VLLM_MQ_MAX_CHUNK_BYTES_MB * 1024 * 1024
            mq_connect_ip = get_ip()
            logger.info(
                "DP group leader: node_rank=%d, node_rank_within_dp=%d, "
                "master_addr=%s, mq_connect_ip=%s (local), "
                "world_size=%d, local_world_size=%d",
                self.parallel_config.node_rank,
                self.parallel_config.node_rank_within_dp,
                self.parallel_config.master_addr,
                mq_connect_ip,
                self.world_size,
                self.local_world_size,
            )
            self.rpc_broadcast_mq = MessageQueue(
                self.world_size,
                self.local_world_size,
                max_chunk_bytes=max_chunk_bytes,
                connect_ip=mq_connect_ip,
            )
            scheduler_output_handle = self.rpc_broadcast_mq.export_handle()

        # ③ 创建 Worker 子进程
        # 每个 Worker 对应一个 GPU，运行独立的 Python 进程
        # Worker 之间通过 NCCL 通信，Worker 与 Executor 通过共享内存通信
        context = get_mp_context()
        shared_worker_lock = context.Lock()
        unready_workers: list[UnreadyWorkerProcHandle] = []
        success = False
        try:
            global_start_rank = (
                self.local_world_size * self.parallel_config.node_rank_within_dp
            )
            # 使用 fork 时，跟踪 Worker 继承的 socket 文件描述符，
            # 以便在后续 Worker 中关闭它们
            inherited_fds: list[int] | None = (
                [] if context.get_start_method() == "fork" else None
            )

            # 为每个 local_rank 创建一个 Worker 子进程
            # 每个 Worker 进程:
            #   1. 初始化分布式环境 (NCCL)
            #   2. 初始化 GPU 设备
            #   3. 加载模型权重
            #   4. 创建消息队列
            #   5. 进入 worker_busy_loop 等待命令
            cpu_omp_manager = OMPProcessManager(self.vllm_config)
            for local_rank in range(self.local_world_size):
                global_rank = global_start_rank + local_rank
                is_driver_worker = self._is_driver_worker(global_rank)
                with cpu_omp_manager.configure_omp_envs(
                    rank=global_rank, local_rank=local_rank
                ):
                    unready_worker_handle = WorkerProc.make_worker_process(
                        vllm_config=self.vllm_config,
                        local_rank=local_rank,
                        rank=global_rank,
                        distributed_init_method=distributed_init_method,
                        input_shm_handle=scheduler_output_handle,
                        shared_worker_lock=shared_worker_lock,
                        is_driver_worker=is_driver_worker,
                        inherited_fds=inherited_fds,
                    )
                unready_workers.append(unready_worker_handle)
                if inherited_fds is not None:
                    inherited_fds.append(unready_worker_handle.death_writer.fileno())
                    inherited_fds.append(unready_worker_handle.ready_pipe.fileno())

            # ④ 等待所有 Worker 就绪
            # Worker 在子进程中完成初始化后，通过 ready_pipe 发送 READY 信号
            self.workers = WorkerProc.wait_for_ready(unready_workers)

            # ⑤ 启动 Worker 健康监控线程
            if self.monitor_workers:
                self.start_worker_monitor()

            # ⑥ 收集 Worker 的响应队列
            # response_mqs: Workers → Executor 的结果返回通道
            # - 本地 Worker: 直接使用 worker_response_mq (共享内存)
            # - 远程 Worker: 通过 peer_worker_response_mqs 访问 (跨节点)
            self.response_mqs = []
            if self.parallel_config.node_rank_within_dp == 0:
                for rank in range(self.world_size):
                    if rank < self.local_world_size:
                        # 本地 Worker: 直接使用共享内存队列
                        local_message_queue = self.workers[rank].worker_response_mq
                        assert local_message_queue is not None
                        self.response_mqs.append(local_message_queue)
                    else:
                        # 远程 Worker: 通过 peer 队列访问
                        remote_message_queue = self.workers[0].peer_worker_response_mqs[
                            rank
                        ]
                        assert remote_message_queue is not None
                        self.response_mqs.append(remote_message_queue)

            # 确保消息队列就绪。顺序错误会导致死锁。
            # 必须与 WorkerProc 保持一致。

            # 等待所有输入消息队列就绪
            if self.rpc_broadcast_mq is not None:
                self.rpc_broadcast_mq.wait_until_ready()
            # 等待所有远程响应消息队列就绪
            for response_mq in self.response_mqs:
                response_mq.wait_until_ready()

            self.futures_queue = deque[FutureWrapper]()

            self._post_init_executor()

            success = True
        finally:
            if not success:
                # 清理失败的 Worker 进程
                # 先关闭 death_writers 通知 Worker 退出
                for uw in unready_workers:
                    if uw.death_writer is not None:
                        uw.death_writer.close()
                        uw.death_writer = None
                self._ensure_worker_termination([uw.proc for uw in unready_workers])

        self.output_rank = self._get_output_rank()

    def _get_parallel_sizes(self) -> tuple[int, int, int]:
        """
        获取并行配置的各个维度大小。

        返回:
            (tp_size, pp_size, pcp_size): 三元组
            - tp_size: 张量并行大小 (Tensor Parallelism)
            - pp_size: 流水线并行大小 (Pipeline Parallelism)
            - pcp_size: 预填充上下文并行大小 (Prefill Context Parallelism)

        验证:
            world_size == tp_size * pp_size * pcp_size
            这是分布式训练/推理的基本约束
        """
        # 获取全局 world_size (所有参与计算的 GPU 总数)
        self.world_size = self.parallel_config.world_size
        # 验证 world_size 必须能被节点数整除
        assert self.world_size % self.parallel_config.nnodes_within_dp == 0, (
            f"global world_size ({self.parallel_config.world_size}) must be "
            f"divisible by nnodes_within_dp "
            f"({self.parallel_config.nnodes_within_dp}). "
        )
        # 获取本地 world_size (当前节点上的 GPU 数量)
        self.local_world_size = self.parallel_config.local_world_size
        # 获取各个并行维度的大小
        tp_size = self.parallel_config.tensor_parallel_size
        pp_size = self.parallel_config.pipeline_parallel_size
        pcp_size = self.parallel_config.prefill_context_parallel_size
        return tp_size, pp_size, pcp_size

    def _post_init_executor(self) -> None:
        """
        执行器初始化后的钩子方法。

        默认实现为空。子类 (如 RayExecutor) 可以覆盖此方法
        添加额外的初始化逻辑，例如:
        - Ray 集群连接
        - 远程节点注册
        - 特定硬件初始化
        """
        pass

    def _is_driver_worker(self, rank: int) -> bool:
        """
        判断指定 rank 是否为驱动 Worker (Driver Worker)。

        驱动 Worker 是每个张量并行组中 rank=0 的 Worker。
        它负责:
        1. 收集 TP 组内所有 Worker 的结果
        2. 将最终结果返回给 Executor
        3. 执行一些只在主 Worker 上运行的操作

        参数:
            rank: 全局 rank 编号

        返回:
            True 如果是驱动 Worker (rank % tp_size == 0)
        """
        return rank % self.parallel_config.tensor_parallel_size == 0

    def start_worker_monitor(self, inline=False) -> None:
        """
        启动 Worker 进程健康监控。

        参数:
            inline: 是否在当前线程内执行监控 (用于调试)
                - False (默认): 启动守护线程异步监控
                - True: 在当前线程同步执行 (阻塞)

        监控机制:
        1. 使用 multiprocessing.connection.wait() 监听所有 Worker 的 sentinel
        2. sentinel 是进程级别的，当进程退出时自动触发
        3. 如果任何 Worker 意外死亡:
           a. 设置 is_failed = True
           b. 记录死亡 Worker 的名称
           c. 调用 shutdown() 关闭所有 Worker
           d. 调用 failure_callback 通知引擎层

        弱引用设计:
        使用 weakref.ref(self) 避免监控线程阻止 Executor 被垃圾回收。
        如果 Executor 已被回收，监控线程会自动退出。
        """
        workers = self.workers
        # 使用弱引用，避免监控线程阻止 Executor 被垃圾回收
        self_ref = weakref.ref(self)

        # 监控 Worker 进程存活状态。如果任何 Worker 意外死亡，
        # 记录错误，关闭执行器并调用失败回调通知引擎。
        def monitor_workers():
            # 获取所有 Worker 进程的 sentinel (进程级别的通知机制)
            sentinels = [h.proc.sentinel for h in workers]
            # 阻塞等待，直到任意一个 Worker 进程退出
            died = multiprocessing.connection.wait(sentinels)
            # 通过弱引用获取 Executor 实例
            _self = self_ref()
            if not _self or getattr(_self, "shutting_down", False):
                logger.debug("MultiprocWorkerMonitor: shutdown already initiated")
                return
            # 标记执行器已失败
            _self.is_failed = True
            # 找到死亡的 Worker 进程名称
            proc_name = next(h.proc.name for h in workers if h.proc.sentinel == died[0])
            logger.error(
                "Worker proc %s died unexpectedly, shutting down executor.", proc_name
            )
            # 关闭所有 Worker
            _self.shutdown()
            # 调用失败回调通知引擎层
            callback = _self.failure_callback
            if callback is not None:
                _self.failure_callback = None
                callback()

        if not inline:
            # 启动守护线程进行异步监控
            Thread(
                target=monitor_workers, daemon=True, name="MultiprocWorkerMonitor"
            ).start()
            return

        # 同步执行监控 (用于调试)
        monitor_workers()

    def register_failure_callback(self, callback: FailureCallback):
        """
        注册失败回调函数。

        当 Executor 检测到 Worker 失败时，会调用此回调通知引擎层。
        引擎层通常会触发引擎重启或优雅关闭。

        参数:
            callback: 失败回调函数，无参数，无返回值

        注意:
            如果注册时 Executor 已经失败 (is_failed=True)，
            会立即调用回调，而不是等待未来的失败事件。
        """
        if self.is_failed:
            # 如果已经失败，立即调用回调
            callback()
        else:
            # 否则注册回调，等待未来的失败事件
            self.failure_callback = callback

    def execute_model(  # type: ignore[override]
        self, scheduler_output: SchedulerOutput, non_block: bool = False
    ) -> ModelRunnerOutput | None | Future[ModelRunnerOutput | None]:
        """
        执行模型前向传播 —— vLLM 的核心推理方法。

        这是 Scheduler 调用的主入口，将 SchedulerOutput 广播给所有 Worker，
        Worker 执行模型前向传播并返回结果。

        参数:
            scheduler_output: 调度器输出，包含:
                - 需要执行的请求批次
                - KV 缓存操作指令
                - 采样参数等
            non_block: 是否非阻塞
                - True: 立即返回 Future，不等待结果
                - False (默认): 阻塞等待结果

        返回:
            non_block=True: Future[ModelRunnerOutput | None]
            non_block=False: ModelRunnerOutput | None

        调用链:
            Scheduler → execute_model() → collective_rpc() → Worker.execute_model()
        """
        return self.collective_rpc(
            "execute_model",
            args=(scheduler_output,),
            unique_reply_rank=self.output_rank,
            non_block=non_block,
            timeout=envs.VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS,
            kv_output_aggregator=self.kv_output_aggregator,
        )

    def sample_tokens(  # type: ignore[override]
        self, grammar_output: GrammarOutput | None, non_block: bool = False
    ) -> ModelRunnerOutput | Future[ModelRunnerOutput]:
        """
        采样 token —— 从模型输出 logits 中采样下一个 token。

        通常在 execute_model 之后调用，用于:
        1. 从 logits 中采样 token
        2. 应用语法约束 (GrammarOutput)
        3. 返回采样结果

        参数:
            grammar_output: 语法约束输出 (可选)
            non_block: 是否非阻塞

        返回:
            ModelRunnerOutput 或 Future
        """
        return self.collective_rpc(
            "sample_tokens",
            args=(grammar_output,),
            unique_reply_rank=self.output_rank,
            non_block=non_block,
            timeout=envs.VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS,
            kv_output_aggregator=self.kv_output_aggregator,
        )

    def execute_dummy_batch(self) -> None:
        """
        执行虚拟批次 —— 用于预热或 CUDA 图捕获。

        在模型初始化阶段，执行一个虚拟批次来:
        1. 预热 GPU (warmup)
        2. 捕获 CUDA 图 (如果启用)
        3. 初始化内部状态

        此方法不需要返回结果，所以不从 Worker 收集输出。
        """
        self.collective_rpc("execute_dummy_batch", unique_reply_rank=self.output_rank)

    def take_draft_token_ids(self) -> DraftTokenIds | None:
        """
        获取草稿 token ID —— 用于推测解码 (Speculative Decoding)。

        推测解码中，草稿模型会生成一组候选 token ID，
        主模型验证这些候选 token 是否正确。

        返回:
            DraftTokenIds | None: 草稿 token ID 列表

        优化: 只从单个 Worker (output_rank) 获取输出，减少 IPC 开销
        """
        return self.collective_rpc(
            "take_draft_token_ids", unique_reply_rank=self.output_rank
        )

    def collective_rpc(  # type: ignore[override]
        self,
        method: str | Callable,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
        non_block: bool = False,
        unique_reply_rank: int | None = None,
        kv_output_aggregator: KVOutputAggregator | None = None,
    ) -> Any:
        """
        集体远程过程调用 —— Executor 与 Workers 的核心通信方法。

        流程:
        ① 将 (method, args, kwargs, output_rank) 序列化
        ② 通过 rpc_broadcast_mq 广播给所有 Worker
        ③ Worker 执行方法，将结果放入 worker_response_mq
        ④ Executor 从 response_mqs 收集结果

        参数:
          method: 方法名 (str) 或可调用对象 (cloudpickle 序列化)
          timeout: 超时时间 (秒)
          args, kwargs: 方法参数
          non_block: 是否非阻塞 (返回 Future)
          unique_reply_rank: 只从此 rank 收集结果 (其他 rank 不返回)
          kv_output_aggregator: KV 输出聚合器

        返回:
          non_block=True: 返回 Future
          non_block=False: 返回结果

        通信示意:
        ┌──────────────┐     rpc_broadcast_mq      ┌──────────────┐
        │   Executor   │ ─────────────────────────→ │  Worker 0    │
        │              │ ─────────────────────────→ │  Worker 1    │
        │              │ ─────────────────────────→ │  Worker N    │
        │              │                            │              │
        │              │ ←───────────────────────── │  (仅 output  │
        │              │     response_mqs[rank]     │   rank 返回) │
        └──────────────┘                            └──────────────┘
        """
        assert self.rpc_broadcast_mq is not None, (
            "collective_rpc should not be called on follower node"
        )
        if self.is_failed:
            raise RuntimeError("Executor failed.")

        deadline = None if timeout is None else time.monotonic() + timeout
        kwargs = kwargs or {}

        # 确定 output_rank: 哪个 Worker 需要返回结果
        if kv_output_aggregator is not None:
            output_rank = None  # 所有 Worker 都返回，由聚合器处理
            aggregate: Callable[[Any], Any] = partial(
                kv_output_aggregator.aggregate, output_rank=unique_reply_rank or 0
            )
        else:
            output_rank = unique_reply_rank  # 只有指定 rank 返回
            aggregate = lambda x: x

        # 序列化方法: 字符串直接发送，可调用对象用 cloudpickle
        if isinstance(method, str):
            send_method = method
        else:
            send_method = cloudpickle.dumps(method, protocol=pickle.HIGHEST_PROTOCOL)

        # ① 广播命令给所有 Worker
        self.rpc_broadcast_mq.enqueue((send_method, args, kwargs, output_rank))

        # ② 确定需要收集结果的队列
        response_mqs: Sequence[MessageQueue] = self.response_mqs
        if output_rank is not None:
            # 只从指定 rank 收集结果
            response_mqs = (response_mqs[output_rank],)

        # ③ 定义结果收集函数
        def get_response():
            responses = []
            for mq in response_mqs:
                dequeue_timeout = (
                    None if deadline is None else (deadline - time.monotonic())
                )
                try:
                    status, result = mq.dequeue(timeout=dequeue_timeout)
                except TimeoutError as e:
                    raise TimeoutError(f"RPC call to {method} timed out.") from e
                if status != WorkerProc.ResponseStatus.SUCCESS:
                    raise RuntimeError(
                        f"Worker failed with error '{result}', please check the"
                        " stack trace above for the root cause"
                    )
                responses.append(result)
            return responses[0] if output_rank is not None else responses

        # ④ 创建 Future 并返回
        future = FutureWrapper(
            self.futures_queue,
            get_response=get_response,
            aggregate=aggregate,
        )

        # non_block=True: 立即返回 Future (不等待结果)
        # non_block=False: 阻塞等待结果
        return future if non_block else future.result()

    @staticmethod
    def _ensure_worker_termination(worker_procs: list[BaseProcess]):
        """
        确保所有 Worker 进程已终止 —— 多级终止策略。

        假设 Worker 已收到终止请求 (例如关闭 rpc_broadcast_mq)。
        此方法执行多级终止策略，确保进程最终被杀死。

        终止策略 (三阶段):
        1. 等待 (4 秒): 给 Worker 时间自行清理资源
        2. SIGTERM (4 秒): 发送软终止信号，允许进程捕获并优雅退出
        3. SIGKILL: 发送硬终止信号，强制杀死进程 (无法捕获)

        参数:
            worker_procs: Worker 进程列表

        注意:
            晚期关闭阶段，Python 解释器可能将 `time` 模块替换为 `None`，
            所以需要检查 `if not time` 的情况。
        """

        def wait_for_termination(procs, shutdown_timeout):
            """
            等待进程终止。

            参数:
                procs: 需要等待的进程列表
                shutdown_timeout: 超时时间 (秒)

            返回:
                True: 所有进程已终止
                False: 超时后仍有进程存活
            """
            if not time:
                # 晚期关闭阶段，解释器可能将 `time` 替换为 `None`
                return all(not proc.is_alive() for proc in procs)
            start_time = time.time()
            while time.time() - start_time < shutdown_timeout:
                if all(not proc.is_alive() for proc in procs):
                    return True
                time.sleep(0.1)
            return False

        # lambda 获取当前仍然存活的进程
        active_procs = lambda: [proc for proc in worker_procs if proc.is_alive()]
        # 第一阶段: 等待进程自行清理 (4 秒)
        logger.debug("Worker Termination: allow workers to gracefully shutdown")
        if wait_for_termination(active_procs(), 4):
            return

        # 第二阶段: 发送 SIGTERM (软终止)
        logger.debug("Worker Termination: workers still running sending SIGTERM")
        for p in active_procs():
            p.terminate()
        if not wait_for_termination(active_procs(), 4):
            # 第三阶段: 发送 SIGKILL (硬终止)
            logger.debug(
                "Worker Termination: resorting to SIGKILL to take down workers"
            )
            for p in active_procs():
                p.kill()

    def shutdown(self):
        """
        关闭执行器和所有 Worker 进程。

        流程:
        ① 设置 shutting_down 标志
        ② 关闭 rpc_broadcast_mq (Worker 会检测到并退出)
        ③ 等待 Worker 自行退出 (graceful shutdown)
        ④ 如果 Worker 未退出，发送 SIGTERM
        ⑤ 如果仍未退出，发送 SIGKILL
        """
        if not getattr(self, "shutting_down", False):
            logger.debug("Triggering shutdown of workers")
            self.shutting_down = True

            # 确保所有 Worker 进程先被终止
            if workers := getattr(self, "workers", None):
                for w in workers:
                    # 关闭 death_writer 通知子进程退出
                    if w.death_writer is not None:
                        w.death_writer.close()
                        w.death_writer = None
                self._ensure_worker_termination([w.proc for w in workers])

                for w in workers:
                    # 关闭响应队列
                    if w.worker_response_mq is not None:
                        w.worker_response_mq.shutdown()
                        w.worker_response_mq = None

        if rpc_broadcast_mq := getattr(self, "rpc_broadcast_mq", None):
            rpc_broadcast_mq.shutdown()
            self.rpc_broadcast_mq = None
        if response_mqs := getattr(self, "response_mqs", None):
            for mq in response_mqs:
                mq.shutdown()
            self.response_mqs = []

    def check_health(self) -> None:
        """
        检查执行器健康状态。

        通过向所有 Worker 发送 check_health RPC 调用来验证:
        1. Worker 进程仍然存活
        2. 消息队列通信正常
        3. Worker 内部状态正常

        超时: 10 秒

        异常:
            如果任何 Worker 响应失败，抛出 RuntimeError
        """
        self.collective_rpc("check_health", timeout=10)
        return

    @cached_property
    def max_concurrent_batches(self) -> int:
        """
        获取最大并发批次数。

        逻辑:
        - PP > 1 时: 返回 PP 大小 (需要填满流水线)
        - PP <= 1 且启用异步调度: 返回 2 (允许双缓冲)
        - 其他: 返回 PP 大小

        为什么 PP 需要 PP 大小的并发批次:
        流水线并行将模型分成多个阶段，每个阶段在不同的 GPU 上运行。
        为了填满流水线，需要同时有 PP 个批次在不同阶段执行。
        """
        pp_size = self.parallel_config.pipeline_parallel_size
        return 2 if pp_size <= 1 and self.scheduler_config.async_scheduling else pp_size

    def _get_output_rank(self) -> int:
        """
        获取输出 rank —— 决定哪个 Worker 返回 ModelRunnerOutput。

        逻辑:
        只从最后一个 PP 阶段的 TP rank=0 返回结果。
        这是因为:
        1. 流水线并行中，只有最后一个阶段有最终输出
        2. 张量并行中，只有 rank=0 需要返回结果 (其他 rank 的结果相同)

        计算公式:
        output_rank = world_size - tp_size * pcp_size

        示例 (TP=8, PP=4):
        world_size = 32
        阶段 0: rank 0-7
        阶段 1: rank 8-15
        阶段 2: rank 16-23
        阶段 3: rank 24-31  (最后一个阶段)
        output_rank = 32 - 8 = 24 (阶段 3 的 TP rank=0)
        """
        return (
            self.world_size
            - self.parallel_config.tensor_parallel_size
            * self.parallel_config.prefill_context_parallel_size
        )

    @classmethod
    def supports_async_scheduling(cls) -> bool:
        """
        检查是否支持异步调度。

        异步调度允许:
        1. 模型执行和结果收集并行进行
        2. 提高 GPU 利用率
        3. 减少空闲时间

        MultiprocExecutor 支持异步调度，返回 True。
        """
        return True


@dataclass
class UnreadyWorkerProcHandle:
    """
    Worker 进程句柄 (就绪前) —— Worker 子进程启动后、初始化完成前的状态。

    字段说明:
        proc: Worker 子进程对象 (multiprocessing.Process)
        rank: Worker 的全局 rank 编号
        ready_pipe: 子→父 管道，用于发送 READY 信号
        death_writer: 父→子 管道的写端，父进程关闭时子进程检测到 EOF

    生命周期:
        创建于 make_worker_process()，销毁于 wait_for_ready()
    """

    proc: BaseProcess
    rank: int
    ready_pipe: Connection
    death_writer: Connection | None = None


@dataclass
class WorkerProcHandle:
    """
    Worker 进程句柄 (就绪后) —— Worker 初始化完成、可以接收命令的状态。

    字段说明:
        proc: Worker 子进程对象
        rank: Worker 的全局 rank 编号
        worker_response_mq: Worker 的响应消息队列
            - 单节点模式: 共享内存队列，Worker 直接写入
            - 多节点模式: 跨节点队列，通过 NCCL/TCP 传输
        peer_worker_response_mqs: 对等 Worker 的响应消息队列列表
            - 只在主节点 (node_rank_within_dp=0) 上非空
            - 用于收集远程 Worker 的结果
        death_writer: 父→子 管道的写端

    与 UnreadyWorkerProcHandle 的区别:
        UnreadyWorkerProcHandle: Worker 进程已启动，但还在初始化
        WorkerProcHandle: Worker 初始化完成，消息队列已建立
    """

    proc: BaseProcess
    rank: int
    # 单节点模式下，Worker 进程写入此消息队列
    worker_response_mq: MessageQueue | None
    # 只在主节点上非空，对等 Worker 进程 i 写入
    # `peer_worker_response_mqs[i]`
    peer_worker_response_mqs: list[MessageQueue | None]
    death_writer: Connection | None = None

    @classmethod
    def from_unready_handle(
        cls,
        unready_handle: UnreadyWorkerProcHandle,
        worker_response_mq: MessageQueue | None,
        peer_worker_response_mqs: list[MessageQueue | None],
    ) -> "WorkerProcHandle":
        """
        从 UnreadyWorkerProcHandle 创建 WorkerProcHandle。

        这是一个工厂方法，在 Worker 初始化完成后调用，
        将"未就绪"句柄转换为"已就绪"句柄，并添加消息队列信息。

        参数:
            unready_handle: 未就绪的 Worker 句柄
            worker_response_mq: Worker 的响应消息队列
            peer_worker_response_mqs: 对等 Worker 的响应队列列表

        返回:
            WorkerProcHandle: 已就绪的 Worker 句柄
        """
        return cls(
            proc=unready_handle.proc,
            rank=unready_handle.rank,
            worker_response_mq=worker_response_mq,
            peer_worker_response_mqs=peer_worker_response_mqs,
            death_writer=unready_handle.death_writer,
        )


class WorkerProc:
    """
    Worker 进程包装器 —— 在独立进程中运行一个 GPU Worker。

    生命周期:
    ┌─────────────────────────────────────────────────────────┐
    │  make_worker_process()                                  │
    │  ├── 创建 ready_pipe (子→父通信)                        │
    │  ├── 创建 death_pipe (父→子通信)                        │
    │  ├── 启动子进程 (target=worker_main)                    │
    │  └── 返回 UnreadyWorkerProcHandle                       │
    │                                                         │
    │  worker_main() [子进程]                                 │
    │  ├── 创建 WorkerProc 实例 (__init__)                    │
    │  │   ├── init_worker() → 加载模型                       │
    │  │   ├── init_device() → 初始化 GPU                     │
    │  │   ├── load_model() → 加载权重                        │
    │  │   └── _init_message_queues() → 创建消息队列          │
    │  ├── 发送 READY 信号                                    │
    │  └── 进入 worker_busy_loop()                            │
    │                                                         │
    │  worker_busy_loop() [子进程主循环]                       │
    │  while True:                                            │
    │      (method, args, kwargs, output_rank) = dequeue()    │
    │      result = getattr(worker, method)(*args, **kwargs)  │
    │      if rank == output_rank: enqueue(result)            │
    │                                                         │
    │  death_pipe_monitor() [守护线程]                         │
    │  监控父进程是否退出，如果退出则终止子进程                 │
    └─────────────────────────────────────────────────────────┘
    """

    # READY 信号字符串，Worker 初始化完成后发送给 Executor
    READY_STR = "READY"
    # 消息队列引用 (在 __init__ 中初始化)
    rpc_broadcast_mq: MessageQueue | None
    worker_response_mq: MessageQueue | None

    def _init_message_queues(
        self, input_shm_handle: Handle, vllm_config: VllmConfig
    ) -> None:
        """
        初始化消息队列 —— Worker 与 Executor 之间的通信通道。

        两种模式:
        ① 单节点 (nnodes_within_dp == 1):
           - rpc_broadcast_mq: 共享内存广播队列 (接收 Executor 命令)
           - worker_response_mq: 共享内存响应队列 (返回结果给 Executor)

        ② 多节点 (nnodes_within_dp > 1):
           - rpc_broadcast_mq: 跨节点广播队列 (通过 NCCL/TCP)
           - worker_response_mq: 跨节点响应队列 (通过 NCCL/TCP)
        """
        if vllm_config.parallel_config.nnodes_within_dp == 1:
            # 单节点模式: 使用共享内存
            # rpc_broadcast_mq: 从 Executor 导出的 handle 创建
            self.rpc_broadcast_mq = MessageQueue.create_from_handle(
                input_shm_handle, self.worker.rank
            )

            # worker_response_mq: 每个 Worker 创建自己的响应队列
            self.worker_response_mq = MessageQueue(1, 1)
            self.peer_response_handles = []
        else:
            # 多节点模式: 使用跨节点通信
            # rpc_broadcast_mq: 通过 DP 组的广播器创建
            self.rpc_broadcast_mq = get_inner_dp_world_group().create_mq_broadcaster(
                external_writer_handle=input_shm_handle,
                blocking=False,  # 非阻塞，等待 wait_until_ready() 触发握手
            )
            # worker_response_mq: 通过 DP 组创建单读者广播器
            # reader_rank_in_group=0: 只有 rank 0 读取所有 Worker 的结果
            self.worker_response_mq, self.peer_response_handles = (
                get_inner_dp_world_group().create_single_reader_mq_broadcasters(
                    reader_rank_in_group=0
                )
            )

    @instrument(span_name="Worker init")
    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        input_shm_handle: Handle,
        shared_worker_lock: LockType,
        is_driver_worker: bool,
    ):
        """
        Worker 进程初始化 —— 在子进程中执行。

        流程:
        ① 创建 WorkerWrapperBase (延迟加载实际 Worker 类)
        ② init_worker() → 解析 Worker 类并实例化
        ③ init_device() → 初始化 GPU、NCCL 分布式环境
        ④ load_model() → 加载模型权重到 GPU
        ⑤ 初始化消息队列 (共享内存或跨节点)
        """
        self.rank = rank

        # ① 创建 WorkerWrapperBase
        # WorkerWrapperBase 是一个延迟包装器:
        #   - 不直接导入 Worker 类 (避免 CUDA 初始化)
        #   - 在 init_worker() 时动态加载
        wrapper = WorkerWrapperBase(rpc_rank=local_rank, global_rank=rank)

        # ② 初始化 Worker
        # all_kwargs: 每个 rank 的参数 (只有当前 rank 填充)
        all_kwargs: list[dict] = [
            {} for _ in range(vllm_config.parallel_config.world_size)
        ]
        all_kwargs[local_rank] = {
            "vllm_config": vllm_config,
            "local_rank": local_rank,
            "rank": rank,
            "distributed_init_method": distributed_init_method,
            "is_driver_worker": is_driver_worker,
            "shared_worker_lock": shared_worker_lock,
        }
        wrapper.init_worker(all_kwargs)
        self.worker = wrapper

        self.setup_proc_title_and_log_prefix(
            enable_ep=vllm_config.parallel_config.enable_expert_parallel
        )

        # ③ 初始化 GPU 设备
        # - 设置 CUDA 设备
        # - 初始化 NCCL 分布式环境
        # - 创建 GPUModelRunner
        self.worker.init_device()

        # 更新进程标题 (现在并行组已初始化)
        self.setup_proc_title_and_log_prefix(
            enable_ep=vllm_config.parallel_config.enable_expert_parallel
        )

        # ④ 加载模型权重
        if envs.VLLM_ELASTIC_EP_SCALE_UP_LAUNCH:
            self.worker.elastic_ep_execute("load_model")
        else:
            self.worker.load_model()

        # ⑤ 设置异步调度 (如果启用)
        # 异步调度时，创建独立的输出拷贝线程
        scheduler_config = vllm_config.scheduler_config
        self.use_async_scheduling = scheduler_config.async_scheduling
        if self.use_async_scheduling:
            self.async_output_queue: queue.Queue = queue.Queue()
            self.async_output_copy_thread = Thread(
                target=self.async_output_busy_loop,
                daemon=True,
                name="WorkerAsyncOutputCopy",
            )
            self.async_output_copy_thread.start()

        # 根据注意力后端设置块大小
        current_platform.update_block_size_for_backend(vllm_config)

        # ⑥ 初始化消息队列
        # 必须在 init_device() 之后，因为多节点需要分布式组已初始化
        self._init_message_queues(input_shm_handle, vllm_config)

        # 启用环境变量缓存 (假设此后不再有环境变量覆盖)
        enable_envs_cache()

    @staticmethod
    def make_worker_process(
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        input_shm_handle,  # Receive SchedulerOutput
        shared_worker_lock: LockType,
        is_driver_worker: bool,
        inherited_fds: list[int] | None = None,
    ) -> UnreadyWorkerProcHandle:
        """
        创建 Worker 子进程 —— 启动一个新的 Python 进程运行 Worker。

        流程:
        ① 创建 ready_pipe (子→父: 通知就绪)
        ② 创建 death_pipe (父→子: 检测父进程退出)
        ③ 启动子进程 (target=worker_main)
        ④ 关闭子进程端的管道
        ⑤ 返回 UnreadyWorkerProcHandle

        管道机制:
        ┌──────────────┐     ready_pipe      ┌──────────────┐
        │   父进程      │ ←────────────────── │   子进程      │
        │  (Executor)  │                      │  (Worker)    │
        │              │     death_pipe       │              │
        │              │ ──────────────────→  │              │
        │              │  (父退出时 EOF)       │              │
        └──────────────┘                      └──────────────┘
        """
        context = get_mp_context()
        # ready_pipe: 子进程初始化完成后，通过此管道通知父进程
        ready_reader, ready_writer = context.Pipe(duplex=False)
        # death_pipe: 父进程退出时，子进程通过 EOF 检测到并自行终止
        death_reader, death_writer = context.Pipe(duplex=False)

        if inherited_fds is not None:
            inherited_fds = inherited_fds.copy()
            inherited_fds.extend((ready_reader.fileno(), death_writer.fileno()))

        process_kwargs = {
            "vllm_config": vllm_config,
            "local_rank": local_rank,
            "rank": rank,
            "distributed_init_method": distributed_init_method,
            "input_shm_handle": input_shm_handle,
            "ready_pipe": ready_writer,       # 子进程写入就绪信号
            "death_pipe": death_reader,        # 子进程读取父进程状态
            "shared_worker_lock": shared_worker_lock,
            "is_driver_worker": is_driver_worker,
            "inherited_fds": inherited_fds if inherited_fds is not None else [],
        }

        # 启动子进程
        proc = context.Process(
            target=WorkerProc.worker_main,
            kwargs=process_kwargs,
            name=f"VllmWorker-{rank}",
            daemon=True,
        )

        # 应用 NUMA 绑定 (如果配置)
        with numa_utils.configure_subprocess(
            vllm_config, local_rank, process_kind="worker"
        ):
            proc.start()

        # 关闭子进程端的管道
        ready_writer.close()   # 父进程不需要写 ready_pipe
        death_reader.close()   # 父进程不需要读 death_pipe
        # 保持 death_writer 打开: 父进程退出时，子进程的 death_reader 会收到 EOF
        return UnreadyWorkerProcHandle(proc, rank, ready_reader, death_writer)

    @staticmethod
    def wait_for_response_handle_ready(
        handles: dict[str, Any], proc_handle: UnreadyWorkerProcHandle
    ) -> WorkerProcHandle:
        response_handle = handles["handle"]
        worker_response_mq: MessageQueue | None = None
        if len(response_handle.local_reader_ranks) > 0:
            worker_response_mq = MessageQueue.create_from_handle(response_handle, 0)
        peer_response_handles = handles["peer_response_handles"]
        peer_worker_response_mqs = [
            MessageQueue.create_from_handle(handle, -1)
            if handle.remote_subscribe_addr is not None
            else None
            for handle in peer_response_handles
        ]
        return WorkerProcHandle.from_unready_handle(
            proc_handle,
            worker_response_mq,
            peer_worker_response_mqs=peer_worker_response_mqs,
        )

    @staticmethod
    def wait_for_ready(
        unready_proc_handles: list[UnreadyWorkerProcHandle],
    ) -> list[WorkerProcHandle]:
        """
        等待所有 Worker 子进程就绪。

        流程:
        ① 使用 multiprocessing.connection.wait() 监听所有 ready_pipe
        ② 当 Worker 发送 READY 信号时，接收响应句柄
        ③ 创建 WorkerProcHandle (包含消息队列句柄)
        ④ 返回所有就绪的 WorkerProcHandle

        如果任何 Worker 初始化失败，抛出异常。
        """
        e = Exception(
            "WorkerProc initialization failed due to an exception in a "
            "background process. See stack trace for root cause."
        )

        pipes = {handle.ready_pipe: handle for handle in unready_proc_handles}
        ready_proc_handles: list[WorkerProcHandle | None] = [None] * len(
            unready_proc_handles
        )
        while pipes:
            # 阻塞等待任意一个 ready_pipe 有数据
            ready = multiprocessing.connection.wait(pipes.keys())
            for pipe in ready:
                assert isinstance(pipe, Connection)
                try:
                    # 接收 Worker 的就绪响应
                    unready_proc_handle = pipes.pop(pipe)
                    response: dict[str, Any] = pipe.recv()
                    if response["status"] != "READY":
                        raise e

                    idx = unready_proc_handle.rank % len(ready_proc_handles)
                    ready_proc_handles[idx] = WorkerProc.wait_for_response_handle_ready(
                        response, unready_proc_handle
                    )
                except EOFError:
                    e.__suppress_context__ = True
                    raise e from None

                finally:
                    # Close connection.
                    pipe.close()

        return cast(list[WorkerProcHandle], ready_proc_handles)

    def shutdown(self):
        """
        关闭 Worker 进程 —— 释放所有资源。

        流程:
        1. 关闭 rpc_broadcast_mq (接收 Executor 命令的队列)
        2. 关闭 worker_response_mq (返回结果给 Executor 的队列)
        3. 关闭 Worker 内部资源 (模型、CUDA 上下文等)
        4. 销毁模型并行组
        5. 销毁分布式环境 (NCCL)

        注意:
        此方法在 Worker 子进程中调用，由 worker_main 的 finally 块触发。
        """
        if self.rpc_broadcast_mq is not None:
            self.rpc_broadcast_mq.shutdown()
        if self.worker_response_mq is not None:
            self.worker_response_mq.shutdown()
        self.worker.shutdown()
        self.rpc_broadcast_mq = None
        self.worker_response_mq = None
        destroy_model_parallel()
        destroy_distributed_environment()

    def monitor_death_pipe(self, death_pipe, shutdown_requested: threading.Event):
        """
        监控父进程退出的管道 —— 孤儿进程检测机制。

        工作原理:
        1. 子进程通过 death_pipe 读取数据
        2. 父进程退出时，OS 自动关闭管道的所有写端
        3. 子进程的 recv() 会收到 EOFError
        4. 子进程检测到父进程已退出，自行终止

        这是一种轻量级的孤儿进程检测机制:
        - 不需要心跳或定时器
        - 依赖 OS 的管道关闭语义
        - 父进程崩溃时也能正确触发

        参数:
            death_pipe: 父→子 管道的读端
            shutdown_requested: 线程事件，用于通知主循环退出
        """
        if death_pipe is None:
            return

        def death_pipe_monitor(queues_to_shutdown: list[MessageQueue]):
            """
            死亡管道监控线程的入口函数。

            阻塞等待父进程退出 (通过 EOF 检测)。
            检测到后，设置 shutdown_requested 事件并关闭消息队列。
            """
            try:
                # 阻塞直到父进程退出 (管道关闭，收到 EOF)
                death_pipe.recv()
            except EOFError:
                # 父进程已退出，开始清理
                logger.info_once("Parent process exited, terminating worker queues")
                shutdown_requested.set()
                # 关闭所有消息队列，这会使 worker_busy_loop 中的
                # dequeue() 抛出异常，从而退出主循环
                for mq in queues_to_shutdown:
                    if mq is not None:
                        mq.shutdown()
            except Exception as e:
                logger.warning("Death monitoring error: %s", e)

        # 直接传递队列引用，避免传递 self 导致 GC 问题
        Thread(
            target=death_pipe_monitor,
            args=([self.rpc_broadcast_mq, self.worker_response_mq],),
            daemon=True,
            name="DeathPipeMonitor",
        ).start()

    @staticmethod
    def worker_main(*args, **kwargs):
        """Worker 初始化和执行循环。在后台进程中运行。"""

        # 用于优雅终止的信号处理器。
        # SystemExit 异常只触发一次，允许 Worker 进程无错误退出
        shutdown_requested = threading.Event()

        def signal_handler(signum, frame):
            nonlocal shutdown_requested
            if not shutdown_requested.is_set():
                shutdown_requested.set()
                logger.debug(
                    "WorkerProc handling signal %d, raising SystemExit", signum
                )
                raise SystemExit()

        # SIGTERM 或 SIGINT 都会终止 Worker
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        # 如果设置了 VLLM_GPU_NIC_PCIE_MAPPING，为 Worker 设置网络设备环境变量
        set_worker_net_device(kwargs.get("local_rank", 0), kwargs["vllm_config"])

        worker = None
        ready_writer = kwargs.pop("ready_pipe")
        death_pipe = kwargs.pop("death_pipe", None)

        # 关闭从父进程继承的管道 (包括其他 Worker 的管道)
        # 显式传递并关闭现有管道，使 fork 模式下管道正常工作。
        # 否则，子进程中存在对管道的隐藏引用，阻止 EOF 关闭。
        for fd in kwargs.pop("inherited_fds", []):
            try:
                os.close(fd)
            except Exception as e:
                logger.warning("Error closing inherited connection: %s: %s", type(e), e)

        try:
            # 初始化追踪器
            rank = kwargs.get("rank", 0)
            maybe_init_worker_tracer(
                instrumenting_module_name="vllm.worker",
                process_kind="worker",
                process_name=f"Worker_{rank}",
            )

            worker = WorkerProc(*args, **kwargs)
            assert worker.worker_response_mq is not None
            if kwargs["vllm_config"].parallel_config.numa_bind:
                numa_utils.log_current_affinity_state(f"Worker_{worker.rank}")

            worker.monitor_death_pipe(death_pipe, shutdown_requested)

            # 所有内容加载完成后，发送 READY 信号
            ready_writer.send(
                {
                    "status": WorkerProc.READY_STR,
                    "handle": worker.worker_response_mq.export_handle(),
                    "peer_response_handles": worker.peer_response_handles,
                }
            )

            # 确保消息队列就绪。顺序错误会导致死锁。
            # 必须与 Executor 保持一致。
            if worker.rpc_broadcast_mq is not None:
                worker.rpc_broadcast_mq.wait_until_ready()
            worker.worker_response_mq.wait_until_ready()
            ready_writer.close()
            ready_writer = None

            # 进入主循环，等待并执行命令
            worker.worker_busy_loop()

        except Exception:
            # 注意: 如果 busy_loop 中出现异常，通过 MQ RPC 发送
            # FAILURE 消息通知 Executor，触发系统关闭。
            # TODO(rob): 处理 MQ 本身损坏的情况。

            if ready_writer is not None:
                logger.exception("WorkerProc failed to start.")
            elif shutdown_requested.is_set():
                logger.info("WorkerProc shutting down.")
            else:
                logger.exception("WorkerProc failed.")

            # The parent sends a SIGTERM to all worker processes if
            # any worker dies. Set this value so we don't re-throw
            # 设置 SystemExit() 以避免 __del__ 中的 zmq 异常
            shutdown_requested.set()

        except SystemExit as e:
            # SystemExit 在 SIGTERM 或 SIGKILL 时触发，通常表示优雅关闭未成功
            logger.warning("WorkerProc was terminated")
            # SystemExit 绝不能被忽略
            raise e

        finally:
            if ready_writer is not None:
                ready_writer.close()
            if death_pipe is not None:
                death_pipe.close()
            # Worker 退出 busy loop 后清理资源
            if worker is not None:
                worker.shutdown()

    class ResponseStatus(Enum):
        """响应状态枚举"""
        SUCCESS = auto()
        FAILURE = auto()

    def enqueue_output(self, output: Any):
        """准备 Worker 的输出并入队到 worker_response_mq。
        如果输出是异常，转换为 FAILURE 响应。
        """
        if isinstance(output, AsyncModelRunnerOutput):
            output = output.get_output()

        if isinstance(output, Exception):
            result = (WorkerProc.ResponseStatus.FAILURE, str(output))
        else:
            result = (WorkerProc.ResponseStatus.SUCCESS, output)
        if (response_mq := self.worker_response_mq) is not None:
            response_mq.enqueue(result)

    def handle_output(self, output: Any):
        """处理 Worker 的输出。如果启用异步调度，传递给 async_output_busy_loop 线程。
        否则直接入队到 worker_response_mq。
        """
        if self.use_async_scheduling:
            self.async_output_queue.put(output)
        else:
            self.enqueue_output(output)

    def async_output_busy_loop(self):
        """异步输出处理线程的入口点。"""

        # 为线程设置 Worker 设备。
        # 线程不会继承主线程的上下文。
        # 调用任何 CUDA 运行时函数时，会隐式地在 device 0 上
        # 创建新的 CUDA 上下文，消耗额外内存。
        # 这里我们为线程设置 Worker 设备，确保上下文与主线程一致。
        from vllm.platforms import current_platform

        if hasattr(self.worker, "device"):
            current_platform.set_device(self.worker.device)

        while True:
            output = self.async_output_queue.get()
            self.enqueue_output(output)

    def worker_busy_loop(self):
        """
        Worker 主循环 —— 持续接收并执行 Executor 的 RPC 命令。

        流程:
        ┌─────────────────────────────────────────────────────────┐
        │  while True:                                            │
        │    ① rpc_broadcast_mq.dequeue() → 等待命令              │
        │    ② 解析方法: str → getattr, bytes → cloudpickle       │
        │    ③ 执行方法: func(*args, **kwargs)                     │
        │    ④ 如果是 output_rank: handle_output(result)           │
        │    ⑤ 异常处理: 记录日志，发送错误响应                     │
        └─────────────────────────────────────────────────────────┘

        output_rank 优化:
        - 大多数方法只需要一个 Worker 返回结果 (如 execute_model)
        - output_rank 指定哪个 Worker 返回结果
        - 其他 Worker 执行方法但不返回结果，减少 IPC 开销
        """
        assert self.rpc_broadcast_mq is not None
        while True:
            # ① 从广播队列接收命令 (阻塞等待)
            method, args, kwargs, output_rank = self.rpc_broadcast_mq.dequeue(
                indefinite=True
            )
            try:
                # ② 解析方法
                if isinstance(method, str):
                    # 字符串方法名: 直接获取属性
                    func = getattr(self.worker, method)
                elif isinstance(method, bytes):
                    # 字节序列: cloudpickle 反序列化
                    func = partial(cloudpickle.loads(method), self.worker)

                # ③ 执行方法
                output = func(*args, **kwargs)
            except Exception as e:
                # 异常处理: 记录日志并发送错误响应
                if hasattr(e, "add_note"):
                    e.add_note(traceback.format_exc())
                logger.exception("WorkerProc hit an exception.")
                # 异常可能不可序列化，转换为字符串
                if output_rank is None or self.rank == output_rank:
                    self.handle_output(e)
                continue

            # ④ 只有 output_rank 指定的 Worker 返回结果
            if output_rank is None or self.rank == output_rank:
                self.handle_output(output)

    @staticmethod
    def setup_proc_title_and_log_prefix(enable_ep: bool) -> None:
        """
        设置 Worker 进程的标题和日志前缀。

        进程标题用于:
        1. ps/top 命令中显示，便于识别 Worker 进程
        2. 日志中添加前缀，便于区分不同 Worker 的输出

        命名规则:
        Worker[_DP{dp_rank}][_PP{pp_rank}][_PCP{pcp_rank}][_TP{tp_rank}][_DCP{dcp_rank}][_EP{ep_rank}]

        示例:
        - 单 GPU: Worker
        - TP=4: Worker_TP0, Worker_TP1, Worker_TP2, Worker_TP3
        - TP=4, PP=2: Worker_PP0_TP0, Worker_PP0_TP1, Worker_PP1_TP0, Worker_PP1_TP1

        参数:
            enable_ep: 是否启用专家并行 (Expert Parallelism)

        注意:
            此方法在两个地方调用:
            1. init_device() 之前: 使用默认名称 "Worker"
            2. init_device() 之后: 使用完整的并行信息
        """
        # Check if parallel groups are initialized first
        if not model_parallel_is_initialized():
            # Parallel groups not yet initialized, use default process name
            set_process_title(name="Worker")
            decorate_logs("Worker")
            return

        # 获取各个并行组的信息
        dp_size = get_dp_group().world_size
        dp_rank = get_dp_group().rank_in_group
        pp_size = get_pp_group().world_size
        pp_rank = get_pp_group().rank_in_group
        pcp_size = get_pcp_group().world_size
        pcp_rank = get_pcp_group().rank_in_group
        tp_size = get_tp_group().world_size
        tp_rank = get_tp_group().rank_in_group
        dcp_size = get_dcp_group().world_size
        dcp_rank = get_dcp_group().rank_in_group
        # 构建进程名称
        process_name = "Worker"
        if dp_size > 1:
            process_name += f"_DP{dp_rank}"
        if pp_size > 1:
            process_name += f"_PP{pp_rank}"
        if pcp_size > 1:
            process_name += f"_PCP{pcp_rank}"
        if tp_size > 1:
            process_name += f"_TP{tp_rank}"
        if dcp_size > 1:
            process_name += f"_DCP{dcp_rank}"
        if enable_ep:
            ep_rank = get_ep_group().rank_in_group
            process_name += f"_EP{ep_rank}"
        # 设置进程标题 (ps/top 中显示)
        set_process_title(name=process_name)
        # 设置日志前缀 (日志输出中显示)
        decorate_logs(process_name)


def set_multiprocessing_worker_envs():
    """
    设置多进程 Worker 环境变量。

    此函数在父进程创建 Worker 子进程之前调用，用于:
    1. 强制使用 spawn 启动方式 (避免 fork 导致的 CUDA 问题)
    2. 配置 OpenMP 线程数，避免 CPU 竞争

    为什么需要限制 OpenMP 线程数:
    - 默认情况下，每个 CPU 核心会创建一个线程
    - 多进程环境中，每个 GPU Worker 都会创建这么多线程
    - 这导致严重的 CPU 竞争，降低性能
    - 特别是在容器环境中，CPU 限制会导致节流 (throttling)

    示例:
    - 8 GPU 系统，每个核心 16 线程 → 128 个线程竞争
    - 设置 OMP_NUM_THREADS=1 → 只有 8 个线程，避免竞争
    """
    # 强制使用 spawn 启动方式
    # fork 会导致 CUDA 上下文被继承，引起各种问题
    _maybe_force_spawn()

    if not current_platform.is_cpu():
        # Configure thread parallelism if OMP_NUM_THREADS isn't set
        #
        # Helps to avoid CPU contention. The default of spawning a thread per
        # core combined with multiprocessing for each GPU can have a negative
        # impact on performance. The contention is amplified when running in a
        # container where CPU limits can cause throttling.
        default_omp_num_threads = 1
        if (
            "OMP_NUM_THREADS" not in os.environ
            and (current_parallelism := torch.get_num_threads())
            > default_omp_num_threads
        ):
            logger.warning_once(
                "Reducing Torch parallelism from %d threads to %d to avoid "
                "unnecessary CPU contention. Set OMP_NUM_THREADS in the "
                "external environment to tune this value as needed.",
                current_parallelism,
                default_omp_num_threads,
            )
            # 设置环境变量，子进程会继承
            os.environ["OMP_NUM_THREADS"] = str(default_omp_num_threads)
            # 设置当前进程的 PyTorch 线程数
            torch.set_num_threads(default_omp_num_threads)
