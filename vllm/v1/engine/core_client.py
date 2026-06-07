# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# =============================================================================
# 模块概述 (Module Overview)
# =============================================================================
# 本模块实现了与 EngineCore 进程进行 IPC（进程间通信）的客户端。
# EngineCore 是 vLLM v1 引擎的核心组件，负责调度和执行模型推理请求。
# 客户端通过 ZMQ（ZeroMQ）消息传递库与运行在后台进程中的 EngineCore 通信。
#
# 通信架构：
#   客户端 (EngineCoreClient)              引擎 (EngineCoreProc)
#   ┌─────────────────┐                   ┌─────────────────┐
#   │ input_socket     │ ── ROUTER/DEALER → │ Input 守护线程  │
#   │  (ROUTER, bind)  │                   │  → input_queue  │
#   ├─────────────────┤                   ├─────────────────┤
#   │ output_socket    │ ← PULL/PUSH ───── │ Output 守护线程 │
#   │  (PULL)          │                   │  ← output_queue │
#   └─────────────────┘                   └─────────────────┘
#
# 请求类型 (EngineCoreRequestType):
#   - ADD (0x00): 添加新请求
#   - ABORT (0x01): 中止请求
#   - UTILITY (0x03): 工具方法调用（profile, sleep, wake_up 等）
#   - START_DP_WAVE (0x02): 启动数据并行波次（DP 模式）
#
# 客户端层次结构：
#   - EngineCoreClient (抽象基类)
#     ├── InprocClient        : 进程内客户端，直接调用 EngineCore（用于调试）
#     └── MPClient            : 多进程客户端基类（ZMQ 通信）
#         ├── SyncMPClient    : 同步多进程客户端（用于 LLM 离线推理）
#         └── AsyncMPClient   : 异步多进程客户端（用于 AsyncLLM 生产环境）
#             └── DPAsyncMPClient : 数据并行异步客户端（外部负载均衡）
#                 └── DPLBAsyncMPClient : 数据并行异步客户端（内部负载均衡）
#
# 关键通信模式：
#   - 同步模式 (SyncMPClient):
#     后台线程从 output_socket 接收输出 → queue.Queue → 主线程 get_output()
#   - 异步模式 (AsyncMPClient):
#     asyncio 任务从 output_socket 接收输出 → asyncio.Queue → 协程 get_output_async()
#   - 工具方法调用:
#     客户端发送 UTILITY 请求 → 引擎执行方法 → 返回 UtilityOutput → 设置 Future 结果
#
# 序列化:
#   - MsgpackEncoder/Decoder: 高效的二进制序列化（比 JSON 快 5-10 倍）
#   - 支持带外张量传输 (TensorIpcSender): 多模态数据通过共享内存传输
# =============================================================================

import asyncio
import contextlib
import queue
import sys
import uuid
import weakref
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable, Sequence
from concurrent.futures import Future
from dataclasses import dataclass
from multiprocessing.connection import Connection
from multiprocessing.queues import Queue
from threading import Thread
from typing import Any, TypeAlias, TypeVar

import msgspec.msgpack
import zmq
import zmq.asyncio

from vllm.config import VllmConfig
from vllm.envs import VLLM_ENGINE_READY_TIMEOUT_S
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.tasks import SupportedTask
from vllm.tracing import instrument
from vllm.utils.async_utils import in_loop
from vllm.utils.network_utils import (
    close_sockets,
    get_open_zmq_inproc_path,
    make_zmq_socket,
)
from vllm.v1.engine import (
    EEP_NOTIFICATION_CALL_ID,
    EEPNotificationType,
    EngineCoreOutputs,
    EngineCoreReadyResponse,
    EngineCoreRequest,
    EngineCoreRequestType,
    PauseMode,
    ReconfigureDistributedRequest,
    ReconfigureRankType,
    UtilityOutput,
)
from vllm.v1.engine.coordinator import DPCoordinator
from vllm.v1.engine.core import EngineCore, EngineCoreProc
from vllm.v1.engine.exceptions import EngineDeadError
from vllm.v1.engine.tensor_ipc import TensorIpcSender
from vllm.v1.engine.utils import (
    CoreEngineActorManager,
    CoreEngineProcManager,
    get_engine_zmq_addresses,
    launch_core_engines,
)
from vllm.v1.executor import Executor
from vllm.v1.pool.late_interaction import get_late_interaction_engine_index
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder, bytestr

logger = init_logger(__name__)

# AnyFuture: 同步 Future 或 asyncio Future 的类型别名，用于统一处理同步/异步场景
AnyFuture: TypeAlias = asyncio.Future[Any] | Future[Any]

_R = TypeVar("_R")  # collective_rpc 的返回类型

# EngineIdentity: 引擎的唯一标识，由 dp_rank 编码为 2 字节小端序
EngineIdentity = bytes


class EngineCoreClient(ABC):
    """
    EngineCoreClient: subclasses handle different methods for pushing
        and pulling from the EngineCore for asyncio / multiprocessing.

    Subclasses:
    * InprocClient: In process EngineCore (for V0-style LLMEngine use)
    * SyncMPClient: ZMQ + background proc EngineCore (for LLM)
    * AsyncMPClient: ZMQ + background proc EngineCore w/ asyncio (for AsyncLLM)
    """

    # EngineCoreClient: EngineCore 客户端的抽象基类。
    # 定义了与 EngineCore 通信的统一接口，子类根据不同的使用场景
    # （进程内、同步多进程、异步多进程）实现具体通信逻辑。
    #
    # 主要职责：
    #   1. 管理 EngineCore 的生命周期（启动、关闭）
    #   2. 发送推理请求（add_request）
    #   3. 获取推理输出（get_output）
    #   4. 调用工具方法（profile, sleep, wake_up 等）
    #   5. 管理 LoRA 适配器
    #
    # 每个操作都提供同步和异步两种版本，子类根据需要实现对应版本。

    @staticmethod
    def make_client(
        multiprocess_mode: bool,
        asyncio_mode: bool,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
    ) -> "EngineCoreClient":
        # 工厂方法：根据运行模式创建对应的客户端实例
        #
        # 参数说明：
        #   - multiprocess_mode: 是否使用多进程模式
        #   - asyncio_mode: 是否使用异步模式
        #   - vllm_config: vLLM 全局配置
        #   - executor_class: 执行器类型
        #   - log_stats: 是否记录统计信息
        #
        # 选择逻辑：
        #   1. 异步 + 多进程 -> AsyncMPClient（用于 AsyncLLM，生产环境默认）
        #   2. 同步 + 多进程 -> SyncMPClient（用于 LLM 离线推理）
        #   3. 同步 + 单进程 -> InprocClient（进程内直接调用，用于调试）
        #   4. 异步 + 单进程 -> 不支持（抛出 NotImplementedError）

        # TODO: support this for debugging purposes.
        if asyncio_mode and not multiprocess_mode:
            raise NotImplementedError(
                "Running EngineCore in asyncio without multiprocessing "
                "is not currently supported."
            )

        if multiprocess_mode and asyncio_mode:
            return EngineCoreClient.make_async_mp_client(
                vllm_config, executor_class, log_stats
            )

        if multiprocess_mode and not asyncio_mode:
            return SyncMPClient(vllm_config, executor_class, log_stats)

        return InprocClient(vllm_config, executor_class, log_stats)

    @staticmethod
    @instrument(span_name="Overall Loading")
    def make_async_mp_client(
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        client_addresses: dict[str, Any] | None = None,
        client_count: int = 1,
        client_index: int = 0,
    ) -> "AsyncMPClient":
        # 创建异步多进程客户端的工厂方法。
        #
        # 数据并行 (Data Parallel) 场景下的客户端选择：
        #   1. data_parallel_size == 1: 普通单引擎 -> AsyncMPClient
        #   2. data_parallel_size > 1 且外部负载均衡: -> DPAsyncMPClient
        #      每个客户端负责一个 DP rank，由外部 LB 分配请求
        #   3. data_parallel_size > 1 且内部负载均衡: -> DPLBAsyncMPClient
        #      客户端内部负责在多个 DP rank 之间分配请求
        #
        # 参数说明：
        #   - client_addresses: 外部提供的 ZMQ 地址（多 API 服务器场景）
        #   - client_count: 客户端总数（多 API 服务器场景）
        #   - client_index: 当前客户端索引

        parallel_config = vllm_config.parallel_config
        client_args = (
            vllm_config,
            executor_class,
            log_stats,
            client_addresses,
            client_count,
            client_index,
        )
        if parallel_config.data_parallel_size > 1:
            if parallel_config.data_parallel_external_lb:
                # External load balancer - client per DP rank.
                return DPAsyncMPClient(*client_args)
            # Internal load balancer - client balances to all DP ranks.
            return DPLBAsyncMPClient(*client_args)
        return AsyncMPClient(*client_args)

    @abstractmethod
    def shutdown(self, timeout: float | None = None) -> None: ...

    def get_output(self) -> EngineCoreOutputs:
        raise NotImplementedError

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        raise NotImplementedError

    def add_request(self, request: EngineCoreRequest) -> None:
        raise NotImplementedError

    def profile(self, is_start: bool = True, profile_prefix: str | None = None) -> None:
        raise NotImplementedError

    def reset_mm_cache(self) -> None:
        raise NotImplementedError

    def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        raise NotImplementedError

    def reset_encoder_cache(self) -> None:
        raise NotImplementedError

    def sleep(self, level: int = 1, mode: PauseMode = "abort") -> None:
        raise NotImplementedError

    def wake_up(self, tags: list[str] | None = None) -> None:
        raise NotImplementedError

    def is_sleeping(self) -> bool:
        raise NotImplementedError

    def execute_dummy_batch(self) -> None:
        raise NotImplementedError

    async def execute_dummy_batch_async(self) -> None:
        raise NotImplementedError

    def abort_requests(self, request_ids: list[str]) -> None:
        raise NotImplementedError

    def add_lora(self, lora_request: LoRARequest) -> bool:
        raise NotImplementedError

    def remove_lora(self, lora_id: int) -> bool:
        raise NotImplementedError

    def list_loras(self) -> set[int]:
        raise NotImplementedError

    def pin_lora(self, lora_id: int) -> bool:
        raise NotImplementedError

    def save_sharded_state(
        self, path: str, pattern: str | None = None, max_size: int | None = None
    ) -> None:
        raise NotImplementedError

    def collective_rpc(
        self,
        method: str | Callable[..., _R],
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
    ) -> list[_R]:
        raise NotImplementedError

    def dp_engines_running(self) -> bool:
        """Returns True if data parallel engines are collectively in a
        running state."""
        raise NotImplementedError

    async def scale_elastic_ep(self, new_data_parallel_size: int) -> None:
        raise NotImplementedError

    async def get_output_async(self) -> EngineCoreOutputs:
        raise NotImplementedError

    async def get_supported_tasks_async(self) -> tuple[SupportedTask, ...]:
        raise NotImplementedError

    async def add_request_async(self, request: EngineCoreRequest) -> None:
        raise NotImplementedError

    async def profile_async(
        self, is_start: bool = True, profile_prefix: str | None = None
    ) -> None:
        raise NotImplementedError

    async def reset_mm_cache_async(self) -> None:
        raise NotImplementedError

    async def reset_prefix_cache_async(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        raise NotImplementedError

    async def reset_encoder_cache_async(self) -> None:
        raise NotImplementedError

    async def sleep_async(self, level: int = 1, mode: PauseMode = "abort") -> None:
        raise NotImplementedError

    async def wake_up_async(self, tags: list[str] | None = None) -> None:
        raise NotImplementedError

    async def is_sleeping_async(self) -> bool:
        raise NotImplementedError

    async def abort_requests_async(self, request_ids: list[str]) -> None:
        raise NotImplementedError

    async def add_lora_async(self, lora_request: LoRARequest) -> bool:
        raise NotImplementedError

    async def remove_lora_async(self, lora_id: int) -> bool:
        raise NotImplementedError

    async def list_loras_async(self) -> set[int]:
        raise NotImplementedError

    async def pin_lora_async(self, lora_id: int) -> bool:
        raise NotImplementedError

    async def save_sharded_state_async(
        self, path: str, pattern: str | None = None, max_size: int | None = None
    ) -> None:
        raise NotImplementedError

    async def collective_rpc_async(
        self,
        method: str | Callable[..., _R],
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
    ) -> list[_R]:
        raise NotImplementedError


class InprocClient(EngineCoreClient):
    """
    InprocClient: client for in-process EngineCore. Intended
    for use in LLMEngine for V0-style add_request() and step()
        EngineCore setup in this process (no busy loop).

        * pushes EngineCoreRequest directly into the EngineCore
        * pulls EngineCoreOutputs by stepping the EngineCore
    """

    # InprocClient: 进程内客户端。
    # 最简单的客户端实现，直接在当前进程内创建并调用 EngineCore。
    # 不涉及任何 IPC 通信，适用于单进程调试或 V0 风格的 LLMEngine。
    #
    # 工作方式：
    #   - add_request: 直接调用 engine_core.add_request()
    #   - get_output: 调用 engine_core.step_fn() 执行一步推理并返回结果
    #   - 其他方法直接委托给 engine_core 对应方法
    #
    # 注意：不支持异步模式和 sleep 的 "wait" 暂停模式。

    def __init__(self, *args, **kwargs):
        # 直接在当前进程创建 EngineCore 实例
        self.engine_core = EngineCore(*args, **kwargs)

    def get_output(self) -> EngineCoreOutputs:
        # 执行一步推理并返回输出。
        # step_fn() 执行一轮调度和模型推理，post_step() 处理推理后的清理工作。
        # 返回 dp_rank=0 的输出，如果没有输出则返回空的 EngineCoreOutputs。
        outputs, model_executed = self.engine_core.step_fn()
        self.engine_core.post_step(model_executed=model_executed)
        return outputs and outputs.get(0) or EngineCoreOutputs()

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        return self.engine_core.get_supported_tasks()

    def add_request(self, request: EngineCoreRequest) -> None:
        # 添加推理请求。
        # preprocess_add_request 对请求进行预处理（如 tokenization），
        # 然后将处理后的请求添加到 EngineCore 的等待队列中。
        req, request_wave = self.engine_core.preprocess_add_request(request)
        self.engine_core.add_request(req, request_wave)

    def abort_requests(self, request_ids: list[str]) -> None:
        if len(request_ids) > 0:
            self.engine_core.abort_requests(request_ids)

    def shutdown(self, timeout: float | None = None) -> None:
        self.engine_core.shutdown()

    def profile(self, is_start: bool = True, profile_prefix: str | None = None) -> None:
        self.engine_core.profile(is_start, profile_prefix)

    def reset_mm_cache(self) -> None:
        self.engine_core.reset_mm_cache()

    def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        return self.engine_core.reset_prefix_cache(
            reset_running_requests, reset_connector
        )

    def reset_encoder_cache(self) -> None:
        self.engine_core.reset_encoder_cache()

    def sleep(self, level: int = 1, mode: PauseMode = "abort") -> None:
        if mode == "wait":
            raise ValueError("'wait' pause mode is not supported in inproc-engine mode")
        result = self.engine_core.sleep(level, mode)
        assert result is None

    def wake_up(self, tags: list[str] | None = None) -> None:
        self.engine_core.wake_up(tags)

    def is_sleeping(self) -> bool:
        return self.engine_core.is_sleeping()

    def execute_dummy_batch(self) -> None:
        self.engine_core.execute_dummy_batch()

    def add_lora(self, lora_request: LoRARequest) -> bool:
        return self.engine_core.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        return self.engine_core.remove_lora(lora_id)

    def list_loras(self) -> set[int]:
        return self.engine_core.list_loras()

    def pin_lora(self, lora_id: int) -> bool:
        return self.engine_core.pin_lora(lora_id)

    def save_sharded_state(
        self, path: str, pattern: str | None = None, max_size: int | None = None
    ) -> None:
        self.engine_core.save_sharded_state(path, pattern, max_size)

    def collective_rpc(
        self,
        method: str | Callable[..., _R],
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
    ) -> list[_R]:
        return self.engine_core.collective_rpc(method, timeout, args, kwargs)

    def dp_engines_running(self) -> bool:
        return False


@dataclass
class BackgroundResources:
    """Used as a finalizer for clean shutdown, avoiding
    circular reference back to the client object."""

    # BackgroundResources: 后台资源管理器。
    # 作为 weakref.finalizer 的回调对象，确保在客户端被垃圾回收时
    # 能够正确清理所有后台资源（ZMQ 套接字、后台任务、引擎进程等）。
    #
    # 为什么不直接持有客户端引用？
    #   - 避免循环引用：客户端 -> 资源 -> 客户端
    #   - 通过 weakref.finalizer 机制，在客户端被 GC 时自动触发清理
    #
    # 资源清单：
    #   1. ctx: ZMQ 上下文（管理所有 ZMQ 套接字的生命周期）
    #   2. engine_manager: 引擎进程管理器（管理 EngineCore 后台进程）
    #   3. coordinator: 数据并行协调器
    #   4. input_socket: 请求发送套接字（ROUTER 类型，绑定模式）
    #   5. output_socket: 输出接收套接字（PULL 类型）
    #   6. first_req_send/rcv_socket: 用于唤醒暂停引擎的信号套接字
    #   7. stats_update_socket: 统计信息订阅套接字（XSUB 类型）
    #   8. output_queue_task: 异步输出处理任务
    #   9. stats_update_task: 统计信息更新任务
    #   10. shutdown_path: 同步模式下的关闭信号路径

    ctx: zmq.Context
    # If CoreEngineProcManager, it manages local engines;
    # if CoreEngineActorManager, it manages all engines.
    engine_manager: CoreEngineProcManager | CoreEngineActorManager | None = None
    coordinator: DPCoordinator | None = None
    output_socket: zmq.Socket | zmq.asyncio.Socket | None = None
    input_socket: zmq.Socket | zmq.asyncio.Socket | None = None
    first_req_send_socket: zmq.asyncio.Socket | None = None
    first_req_rcv_socket: zmq.asyncio.Socket | None = None
    stats_update_socket: zmq.asyncio.Socket | None = None
    output_queue_task: asyncio.Task | None = None
    stats_update_task: asyncio.Task | None = None
    shutdown_path: str | None = None

    # Set if any of the engines are dead. Here so that the output
    # processing threads can access it without holding a ref to the client.
    # engine_dead: 引擎死亡标志。
    # 当任何 EngineCore 进程意外终止时设置为 True，
    # 后续所有操作都会抛出 EngineDeadError。
    engine_dead: bool = False

    def __call__(self):
        """Clean up background resources."""
        # 清理所有后台资源。此方法在客户端被 GC 时由 weakref.finalizer 调用。
        #
        # 清理顺序：
        #   1. 设置 engine_dead 标志（防止后续操作）
        #   2. 关闭引擎进程管理器
        #   3. 关闭数据并行协调器
        #   4. 关闭 ZMQ 套接字和异步任务
        #
        # 异步模式特殊处理：
        #   - 需要在正确的事件循环中关闭套接字和取消任务
        #   - 如果在事件循环线程中，直接关闭
        #   - 否则通过 call_soon_threadsafe 调度关闭
        #
        # 同步模式特殊处理：
        #   - 需要通过发送关闭信号来终止输出处理线程

        self.engine_dead = True
        if self.engine_manager is not None:
            self.engine_manager.shutdown()
        if self.coordinator is not None:
            self.coordinator.shutdown()

        if isinstance(self.output_socket, zmq.asyncio.Socket):
            # Async case.
            loop = self.output_queue_task._loop if self.output_queue_task else None

            sockets = (
                self.output_socket,
                self.input_socket,
                self.first_req_send_socket,
                self.first_req_rcv_socket,
                self.stats_update_socket,
            )

            tasks = (self.output_queue_task, self.stats_update_task)

            def close_sockets_and_tasks():
                close_sockets(sockets)
                for task in tasks:
                    if task is not None and not task.done():
                        with contextlib.suppress(Exception):
                            task.cancel()

            if loop is not None:
                if in_loop(loop):
                    close_sockets_and_tasks()
                elif not loop.is_closed():
                    loop.call_soon_threadsafe(close_sockets_and_tasks)
            else:
                # Loop has been closed, try to clean up directly.
                del tasks
                del close_sockets_and_tasks
                close_sockets(sockets)
                del self.output_queue_task
                del self.stats_update_task
        else:
            # Sync case.

            # ZMQ context termination can hang if the sockets
            # aren't explicitly closed first.
            close_sockets((self.output_socket, self.input_socket))

            if self.shutdown_path is not None:
                # We must ensure that the sync output socket is
                # closed cleanly in its own thread.
                with self.ctx.socket(zmq.PAIR) as shutdown_sender:
                    shutdown_sender.connect(self.shutdown_path)
                    # Send shutdown signal.
                    shutdown_sender.send(b"")

    def validate_alive(self, frames: Sequence[zmq.Frame]):
        # 验证引擎是否存活。
        # 检查接收到的 ZMQ 帧是否为引擎死亡信号（单帧且内容为 ENGINE_CORE_DEAD）。
        # 如果是，设置 engine_dead 标志并抛出异常。
        if len(frames) == 1 and (frames[0].buffer == EngineCoreProc.ENGINE_CORE_DEAD):
            self.engine_dead = True
            raise EngineDeadError()


@dataclass
class ElasticScalingCache:
    # ElasticScalingCache: 弹性伸缩缓存。
    # 用于弹性专家并行 (Elastic EP) 的扩缩容操作。
    # 缓存扩缩容过程中的状态信息，包括：
    #   - existing_core_engines: 扩缩容前的引擎列表
    #   - num_new_core_engines: 新增/减少的引擎数量（正数为扩容，负数为缩容）
    #   - pending_notifications: 等待处理的通知（按通知类型分组）
    existing_core_engines: list[EngineIdentity]
    num_new_core_engines: int
    pending_notifications: dict[EEPNotificationType, set[int]]


class MPClient(EngineCoreClient):
    """
    MPClient: base client for multi-proc EngineCore.
        EngineCore runs in a background process busy loop, getting
        new EngineCoreRequests and returning EngineCoreOutputs

        * pushes EngineCoreRequests via input_socket
        * pulls EngineCoreOutputs via output_socket

        * AsyncMPClient subclass for AsyncLLM usage
        * SyncMPClient subclass for LLM usage
    """

    # MPClient: 多进程客户端基类。
    # EngineCore 运行在后台进程中，通过 ZMQ 进行 IPC 通信。
    #
    # 通信架构：
    #   客户端 (MPClient)                    引擎 (EngineCoreProc)
    #   ┌─────────────┐                     ┌─────────────────┐
    #   │ input_socket │ ──── ROUTER ──────> │  DEALER socket  │
    #   │  (ROUTER)    │                     │                 │
    #   ├─────────────┤                     ├─────────────────┤
    #   │output_socket │ <─── PULL ──────── │  PUSH socket    │
    #   │  (PULL)      │                     │                 │
    #   └─────────────┘                     └─────────────────┘
    #
    # ZMQ 套接字类型选择理由：
    #   - ROUTER (input): 支持按身份路由，允许多个引擎连接
    #   - PULL (output): 简单的接收端，支持多个引擎向同一地址推送
    #
    # 请求格式: (EngineIdentity, RequestType, SerializedPayload, ...)
    # 响应格式: (SerializedEngineCoreOutputs) 或 (ENGINE_CORE_DEAD,)
    #
    # 关键设计：
    #   - 使用 MsgpackEncoder/Decoder 进行高效的二进制序列化
    #   - 支持带外张量传输 (TensorIpcSender) 用于多模态数据
    #   - 引擎就绪握手：启动时等待所有引擎发送 READY 消息
    #   - 引擎存活监控：后台线程检测引擎进程意外终止

    def __init__(
        self,
        asyncio_mode: bool,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        client_addresses: dict[str, Any] | None = None,
    ):
        self.vllm_config = vllm_config

        # ZMQ setup.
        # 创建 ZMQ 上下文。io_threads=2 允许 ZMQ 使用 2 个 I/O 线程处理消息。
        # 异步模式下包装为 zmq.asyncio.Context 以支持 await。
        sync_ctx = zmq.Context(io_threads=2)
        self.ctx = zmq.asyncio.Context(sync_ctx) if asyncio_mode else sync_ctx

        # This will ensure resources created so far are closed
        # when the client is garbage collected, even if an
        # exception is raised mid-construction.
        # 创建资源管理器并注册 weakref.finalizer，确保资源被正确清理。
        # weakref.finalizer 在客户端对象被 GC 时自动调用 resources() 清理资源。
        self.resources = BackgroundResources(ctx=sync_ctx)
        self._finalizer = weakref.finalize(self, self.resources)
        success = False
        try:
            # State used for data parallel.
            self.engines_running = False
            parallel_config = vllm_config.parallel_config
            # Elastic EP can remove a rank and later add it back with the same
            # identity. The client input ROUTER needs handover to allow the new
            # engine to replace the dead connection.
            # 弹性 EP 模式下启用 ROUTER 套接字的 handover 功能，
            # 允许新引擎替换已死亡引擎的连接。
            enable_input_socket_handover = parallel_config.enable_elastic_ep

            self.stats_update_address: str | None = None
            tensor_queue: Queue | None = None
            if client_addresses:
                # Engines are managed externally to this client.
                # 外部管理模式：引擎由外部进程管理（如多 API 服务器场景）
                # 客户端只负责创建 ZMQ 套接字并绑定到指定地址
                input_address = client_addresses["input_address"]
                output_address = client_addresses["output_address"]
                self.stats_update_address = client_addresses.get("stats_update_address")
                # Tensor queues passed via client_addresses for multi-API-server case
                # 多模态张量 IPC 队列，用于多 API 服务器场景
                tensor_queue = client_addresses.get("tensor_queue")
                # 创建输入套接字（ROUTER 类型，绑定模式）
                # ROUTER 套接字按身份路由消息，支持多个引擎连接
                self.input_socket = self.resources.input_socket = make_zmq_socket(
                    self.ctx,
                    input_address,
                    zmq.ROUTER,
                    bind=True,
                    router_handover=enable_input_socket_handover,
                )
                # 创建输出套接字（PULL 类型）
                # PULL 套接字从多个 PUSH 套接字接收消息
                self.resources.output_socket = make_zmq_socket(
                    self.ctx, output_address, zmq.PULL
                )

                # Report bound endpoints back so the parent can forward
                # them to engines (mirrors the DPCoordinator pattern).
                # 将实际绑定的端点地址回传给父进程，
                # 以便父进程将地址转发给引擎进程。
                actual_address_pipe: Connection | None = client_addresses.get(
                    "actual_address_pipe"
                )
                if actual_address_pipe is not None:
                    try:
                        actual_input = self.input_socket.getsockopt(
                            zmq.LAST_ENDPOINT
                        ).decode()
                        actual_output = self.resources.output_socket.getsockopt(
                            zmq.LAST_ENDPOINT
                        ).decode()
                        actual_address_pipe.send(
                            {
                                "input_address": actual_input,
                                "output_address": actual_output,
                            }
                        )
                    finally:
                        actual_address_pipe.close()
            else:
                # Engines are managed by this client.
                # 自管理模式：客户端负责启动和管理引擎进程
                addresses = get_engine_zmq_addresses(vllm_config)
                self.input_socket = self.resources.input_socket = make_zmq_socket(
                    self.ctx,
                    addresses.inputs[0],
                    zmq.ROUTER,
                    bind=True,
                    router_handover=enable_input_socket_handover,
                )
                self.resources.output_socket = make_zmq_socket(
                    self.ctx, addresses.outputs[0], zmq.PULL
                )

                # Resolve ``tcp://host:0`` placeholders to bound endpoints
                # before engines DEALER-connect. No-op for IPC.
                # 解析 tcp://host:0 占位符为实际绑定的端点地址。
                # 这对于 TCP 传输是必需的（端口 0 表示自动分配端口），
                # 对于 IPC 传输则是空操作。
                addresses.inputs[0] = self.input_socket.getsockopt(
                    zmq.LAST_ENDPOINT
                ).decode()
                addresses.outputs[0] = self.resources.output_socket.getsockopt(
                    zmq.LAST_ENDPOINT
                ).decode()

                # 启动引擎进程。
                # launch_core_engines 返回一个上下文管理器，
                # 退出时会关闭引擎进程管理器和协调器。
                with launch_core_engines(
                    vllm_config, executor_class, log_stats, addresses
                ) as (engine_manager, coordinator, addresses, tensor_queue):
                    self.resources.coordinator = coordinator
                    self.resources.engine_manager = engine_manager

                self.stats_update_address = addresses.frontend_stats_publish_address
                if coordinator is not None:
                    assert self.stats_update_address == (
                        coordinator.get_stats_publish_address()
                    )

            # Serialization setup with tensor queues for multimodal tensor IPC.
            # 序列化设置，支持多模态张量的带外传输。
            # 对于多模态模型（如视觉语言模型），输入可能包含大型张量（如图像）。
            # 使用 torch_shm 模式时，张量通过共享内存队列传输，
            # 避免 ZMQ 消息的序列化/反序列化开销。
            tensor_ipc_sender: TensorIpcSender | None = None
            model_config = getattr(vllm_config, "model_config", None)
            if model_config is not None and model_config.multimodal_config is not None:
                mm_tensor_ipc = model_config.multimodal_config.mm_tensor_ipc
                if mm_tensor_ipc == "torch_shm" and tensor_queue is not None:
                    tensor_ipc_sender = TensorIpcSender(tensor_queue)

            # MsgpackEncoder: 将请求对象序列化为 msgpack 二进制格式
            # MsgpackDecoder: 将 msgpack 二进制数据反序列化为 EngineCoreOutputs
            self.encoder = MsgpackEncoder(oob_tensor_consumer=tensor_ipc_sender)
            self.decoder = MsgpackDecoder(EngineCoreOutputs)

            # 数据并行配置。
            # 确定此客户端负责管理哪些 DP rank 的引擎。
            #   - dp_size: 总的 DP 大小
            #   - dp_rank: 当前 DP rank
            #   - dp_local_size: 本地 DP 大小
            #   - offline_mode: 离线模式（程序化使用 LLM 类）
            #
            # 管理范围：
            #   - 离线模式: 只管理一个 rank（dp_rank）
            #   - 内部 LB: 管理所有 rank
            #   - 混合/外部 LB: 只管理本地 rank
            dp_size = parallel_config.data_parallel_size
            dp_rank = parallel_config.data_parallel_index
            dp_local_size = parallel_config.data_parallel_size_local
            offline_mode = parallel_config.data_parallel_rank_local is not None
            # Client manages local+remote EngineCores in pure internal LB case.
            # Client manages local EngineCores in hybrid and external LB case.
            num_ranks = dp_local_size if parallel_config.local_engines_only else dp_size
            self.engine_ranks_managed = (
                [dp_rank] if offline_mode else list(range(dp_rank, dp_rank + num_ranks))
            )
            assert parallel_config.data_parallel_size_local <= len(
                self.engine_ranks_managed
            )

            # ZMQ identity of each engine that this client will talk to.
            # 为每个引擎生成 ZMQ 身份标识（2 字节小端序的 rank 编号）。
            # ROUTER 套接字使用此身份将消息路由到正确的引擎。
            self.core_engines: list[EngineIdentity] = [
                rank.to_bytes(2, "little") for rank in self.engine_ranks_managed
            ]

            # Wait for ready messages from each engine on the input socket.
            # 等待所有引擎发送就绪消息。
            # 这是一个同步阻塞操作，确保所有引擎都已完成初始化（包括模型加载）
            # 才继续处理请求。
            #
            # 就绪消息包含引擎的初始配置信息（如 max_model_len、num_gpu_blocks），
            # 通过 _apply_ready_response() 同步到客户端的配置中。
            identities = set(self.core_engines)
            sync_input_socket = zmq.Socket.shadow(self.input_socket)
            while identities:
                if not sync_input_socket.poll(
                    timeout=VLLM_ENGINE_READY_TIMEOUT_S * 1000  # convert to ms
                ):
                    raise TimeoutError(
                        f"Timed out waiting for engine core processes to "
                        f"start. This is often caused by slow weight loading "
                        f"for large models. Waited "
                        f"{VLLM_ENGINE_READY_TIMEOUT_S}s (configured by "
                        f"VLLM_ENGINE_READY_TIMEOUT_S). To increase the "
                        f"timeout, set the environment variable: "
                        f"VLLM_ENGINE_READY_TIMEOUT_S=<seconds>"
                    )
                identity, payload = sync_input_socket.recv_multipart()
                identities.remove(identity)
                self._apply_ready_response(payload)

            # 默认引擎身份（单引擎场景使用）
            self.core_engine: EngineIdentity = self.core_engines[0]
            # utility_results: 工具方法调用的结果缓存。
            # 键为 call_id（UUID），值为 Future 对象。
            # 当引擎返回工具方法结果时，通过 call_id 查找并设置结果。
            self.utility_results: dict[int, AnyFuture] = {}

            # Request objects which may contain pytorch-allocated tensors
            # that we need to keep references to until zmq is done with the
            # underlying data.
            # 待发送消息队列。
            # 当请求包含 PyTorch 张量时，ZMQ 需要引用底层数据直到发送完成。
            # 通过 MessageTracker 追踪发送状态，完成后释放引用。
            self.pending_messages = deque[tuple[zmq.MessageTracker, Any]]()

            # Start monitoring engine core processes for unexpected failures
            # 启动引擎存活监控线程。
            # 如果任何引擎进程意外终止，设置 engine_dead 标志并关闭客户端。
            self.start_engine_core_monitor()

            success = True
        finally:
            if not success:
                self._finalizer()

    def shutdown(self, timeout: float | None = None) -> None:
        """Shutdown engine manager under timeout and clean up resources."""
        if self._finalizer.detach() is not None:
            if self.resources.engine_manager is not None:
                self.resources.engine_manager.shutdown(timeout=timeout)
            self.resources()

    def _format_exception(self, e: Exception) -> Exception:
        """If errored, use EngineDeadError so root cause is clear."""
        # 格式化异常。
        # 如果引擎已死亡，返回 EngineDeadError 以明确根本原因，
        # 否则返回原始异常。
        return (
            EngineDeadError(suppress_context=True) if self.resources.engine_dead else e
        )

    def ensure_alive(self):
        # 确保引擎存活，如果已死亡则抛出 EngineDeadError。
        if self.resources.engine_dead:
            raise EngineDeadError()

    def add_pending_message(self, tracker: zmq.MessageTracker, msg: Any):
        # 添加待发送消息到队列。
        # 当 ZMQ 发送包含张量的消息时，需要追踪发送状态。
        if not tracker.done:
            self.pending_messages.appendleft((tracker, msg))

    def free_pending_messages(self):
        # 释放已完成发送的消息引用。
        # 从队列尾部开始释放（FIFO 顺序），直到遇到未完成的消息。
        while self.pending_messages and self.pending_messages[-1][0].done:
            self.pending_messages.pop()

    def dp_engines_running(self) -> bool:
        return self.engines_running

    def start_engine_core_monitor(self):
        """Start a monitor thread for engine core processes."""
        # 启动引擎存活监控线程。
        # 此线程在后台持续监控引擎进程的存活状态。
        # 如果任何引擎进程意外终止（如 OOM、段错误），
        # 线程会设置 engine_dead 标志并关闭客户端。
        engine_manager = self.resources.engine_manager
        if engine_manager is None:
            # No engine processes to monitor
            return

        self_ref = weakref.ref(self)

        # Monitor engine core process liveness. If any die unexpectedly,
        # marks the engine as dead, and shuts down the client.
        def monitor_engine_cores():
            engine_manager.monitor_engine_liveness()
            _self = self_ref()
            if not _self or not _self._finalizer.alive or _self.resources.engine_dead:
                return
            _self.resources.engine_dead = True
            _self.shutdown()
            # Note: For MPClient, we don't have a failure callback mechanism
            # like MultiprocExecutor, but we set engine_dead flag which will
            # cause subsequent operations to raise EngineDeadError

        Thread(
            target=monitor_engine_cores, daemon=True, name="MPClientEngineMonitor"
        ).start()

    def _apply_ready_response(self, payload: bytes) -> None:
        """Decode an EngineCoreReadyResponse and sync any post-initialization
        config changes (e.g. auto-fitted max_model_len) back to the frontend."""
        # 处理引擎就绪响应。
        # 解码引擎发送的就绪消息，并将引擎初始化后的配置同步到客户端。
        #
        # 同步的配置包括：
        #   1. max_model_len: 取客户端和引擎中的较小值（引擎可能根据显存自动调整）
        #   2. num_gpu_blocks: 累加所有引擎的 GPU KV 缓存块数量
        #   3. dp_stats_address: 数据并行统计信息地址（外部 LB 模式）
        if not payload:
            return
        vllm_config = self.vllm_config
        response = msgspec.msgpack.decode(payload, type=EngineCoreReadyResponse)
        vllm_config.model_config.max_model_len = min(
            vllm_config.model_config.max_model_len, response.max_model_len
        )

        # Setup KV cache config with initialization state from
        # engine core process. Sum values from all engines in DP case.
        # 设置 KV 缓存配置。
        # 在 DP 场景下，累加所有引擎的 GPU 块数量。
        num_gpu_blocks = vllm_config.cache_config.num_gpu_blocks or 0
        num_gpu_blocks += response.num_gpu_blocks
        vllm_config.cache_config.num_gpu_blocks = num_gpu_blocks

        # In external DP LB mode, the coordinator address that the
        # front-end procs connect to is obtained by each engine via it's
        # initial handshake with the rank 0 front-end.
        # 外部 DP LB 模式下，协调器地址通过引擎的就绪消息传递。
        if response.dp_stats_address is not None:
            if self.stats_update_address is None:
                self.stats_update_address = response.dp_stats_address
            else:
                assert response.dp_stats_address == self.stats_update_address


def _process_utility_output(
    output: UtilityOutput, utility_results: dict[int, AnyFuture]
):
    """Set the result from a utility method in the waiting future."""
    # 处理工具方法的输出结果。
    #
    # 工具方法调用流程：
    #   1. 客户端发送 UTILITY 请求，生成 call_id，创建 Future
    #   2. 引擎执行工具方法，返回 UtilityOutput（包含 call_id 和结果）
    #   3. 此函数通过 call_id 查找对应的 Future 并设置结果
    #   4. 客户端的 call_utility / call_utility_async 等待 Future 完成并返回结果
    #
    # 错误处理：
    #   - 如果引擎返回失败消息，设置异常而非结果
    #   - 如果 Future 已被取消（如任务超时），记录错误日志
    future = utility_results.pop(output.call_id)
    failure_message = output.failure_message
    try:
        if failure_message is not None:
            future.set_exception(Exception(failure_message))
        else:
            assert output.result is not None
            future.set_result(output.result.result)
    except asyncio.InvalidStateError:
        # This can happen if the future is cancelled due to the
        # original calling task being cancelled.
        if failure_message is not None:
            logger.error(
                "Cancelled call to utility method failed with error: %s",
                failure_message,
            )


class SyncMPClient(MPClient):
    """Synchronous client for multi-proc EngineCore."""

    # SyncMPClient: 同步多进程客户端。
    # 用于 LLM 类的离线推理场景，提供同步阻塞的 API。
    #
    # 工作原理：
    #   1. 启动后台线程 (process_outputs_socket) 从 output_socket 接收输出
    #   2. 后台线程将输出放入 queue.Queue
    #   3. 主线程通过 get_output() 从队列获取输出
    #   4. 工具方法调用通过 call_utility() 发送请求并阻塞等待结果
    #
    # 线程模型：
    #   - 主线程: 调用 add_request(), get_output(), call_utility() 等
    #   - 输出处理线程: 从 output_socket 接收并解码输出
    #   - 引擎监控线程: 监控引擎进程存活状态
    #
    # 关闭机制：
    #   - 使用 ZMQ PAIR 套接字发送关闭信号给输出处理线程
    #   - 输出处理线程收到信号后退出循环并关闭套接字

    @instrument(span_name="SyncMPClient init")
    def __init__(
        self, vllm_config: VllmConfig, executor_class: type[Executor], log_stats: bool
    ):
        super().__init__(
            asyncio_mode=False,
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=log_stats,
        )

        self.is_dp = self.vllm_config.parallel_config.data_parallel_size > 1
        # 输出队列：后台线程将解码后的输出放入此队列，主线程从中获取
        self.outputs_queue = queue.Queue[EngineCoreOutputs | Exception]()

        # Ensure that the outputs socket processing thread does not have
        # a ref to the client which prevents gc.
        # 提取局部变量引用，避免后台线程持有客户端对象的引用，
        # 这样客户端可以被正常垃圾回收。
        ctx = self.ctx
        out_socket = self.resources.output_socket
        decoder = self.decoder
        utility_results = self.utility_results
        outputs_queue = self.outputs_queue

        # 创建用于关闭信号的 ZMQ PAIR 套接字路径
        shutdown_path = get_open_zmq_inproc_path()
        resources = self.resources
        resources.shutdown_path = shutdown_path

        def process_outputs_socket():
            # 输出处理线程主函数。
            # 持续监听 output_socket 和 shutdown_socket，
            # 接收引擎输出并放入队列，直到收到关闭信号。
            #
            # 处理逻辑：
            #   1. 使用 Poller 同时监听两个套接字
            #   2. 收到关闭信号 -> 退出循环
            #   3. 收到输出 -> 验证引擎存活 -> 解码 -> 分发
            #   4. 工具方法输出 -> 设置 Future 结果
            #   5. 普通推理输出 -> 放入输出队列
            assert isinstance(out_socket, zmq.Socket)
            shutdown_socket = ctx.socket(zmq.PAIR)
            try:
                shutdown_socket.bind(shutdown_path)
                poller = zmq.Poller()
                poller.register(shutdown_socket, zmq.POLLIN)
                poller.register(out_socket, zmq.POLLIN)
                while True:
                    socks = poller.poll()
                    if not socks:
                        continue
                    if len(socks) == 2 or socks[0][0] == shutdown_socket:
                        # shutdown signal, exit thread.
                        break

                    frames = out_socket.recv_multipart(copy=False)
                    resources.validate_alive(frames)
                    outputs: EngineCoreOutputs = decoder.decode(frames)
                    if outputs.utility_output:
                        _process_utility_output(outputs.utility_output, utility_results)
                    else:
                        outputs_queue.put_nowait(outputs)
            except Exception as e:
                outputs_queue.put_nowait(e)
            finally:
                # Close sockets.
                shutdown_socket.close(linger=0)
                out_socket.close(linger=0)

        # Process outputs from engine in separate thread.
        self.output_queue_thread = Thread(
            target=process_outputs_socket,
            name="EngineCoreOutputQueueThread",
            daemon=True,
        )
        self.output_queue_thread.start()

        # The thread takes on responsibility for closing the socket.
        self.resources.output_socket = None

    def get_output(self) -> EngineCoreOutputs:
        # 从输出队列获取推理输出（同步阻塞）。
        # 如果后台线程遇到异常，会将异常放入队列，
        # 此处捕获并重新抛出以关闭服务器。
        outputs = self.outputs_queue.get()

        if isinstance(outputs, Exception):
            raise self._format_exception(outputs) from None
        # wave_complete 表示一轮推理批次完成，引擎不再运行
        if outputs.wave_complete is not None:
            self.engines_running = False
        return outputs

    def _send_input(self, request_type: EngineCoreRequestType, request: Any):
        # 发送输入到引擎。
        #
        # 消息格式: (EngineIdentity, RequestType, SerializedPayload, ...)
        #   - EngineIdentity: 引擎的 ZMQ 身份标识
        #   - RequestType: 请求类型枚举值
        #   - SerializedPayload: msgpack 序列化的请求数据
        #
        # 零拷贝优化：
        #   - 如果没有辅助缓冲区（张量），直接发送（copy=False）
        #   - 如果有张量，使用 track=True 追踪发送状态，
        #     防止在 ZMQ 发送完成前释放张量内存
        self.ensure_alive()
        self.free_pending_messages()
        # (Identity, RequestType, SerializedRequest)
        msg = (self.core_engine, request_type.value, *self.encoder.encode(request))

        if len(msg) <= 3:
            # No auxiliary buffers => no tensor backing buffers in request.
            self.input_socket.send_multipart(msg, copy=False)
            return

        tracker = self.input_socket.send_multipart(msg, copy=False, track=True)
        self.add_pending_message(tracker, request)

    def call_utility(self, method: str, *args) -> Any:
        # 调用工具方法（同步阻塞）。
        #
        # 流程：
        #   1. 生成唯一的 call_id
        #   2. 创建 Future 对象并注册到 utility_results
        #   3. 发送 UTILITY 请求到引擎
        #   4. 阻塞等待 Future 完成（由 _process_utility_output 设置结果）
        #
        # 调用的工具方法包括：profile, sleep, wake_up, reset_prefix_cache,
        # add_lora, remove_lora, collective_rpc 等。
        call_id = uuid.uuid1().int >> 64
        future: Future[Any] = Future()
        self.utility_results[call_id] = future
        self._send_input(EngineCoreRequestType.UTILITY, (0, call_id, method, args))

        return future.result()

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        return self.call_utility("get_supported_tasks")

    def add_request(self, request: EngineCoreRequest) -> None:
        if self.is_dp:
            self.engines_running = True
        self._send_input(EngineCoreRequestType.ADD, request)

    def abort_requests(self, request_ids: list[str]) -> None:
        if request_ids and not self.resources.engine_dead:
            self._send_input(EngineCoreRequestType.ABORT, request_ids)

    def profile(self, is_start: bool = True, profile_prefix: str | None = None) -> None:
        self.call_utility("profile", is_start, profile_prefix)

    def reset_mm_cache(self) -> None:
        self.call_utility("reset_mm_cache")

    def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        return self.call_utility(
            "reset_prefix_cache", reset_running_requests, reset_connector
        )

    def reset_encoder_cache(self) -> None:
        self.call_utility("reset_encoder_cache")

    def add_lora(self, lora_request: LoRARequest) -> bool:
        return self.call_utility("add_lora", lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        return self.call_utility("remove_lora", lora_id)

    def list_loras(self) -> set[int]:
        return self.call_utility("list_loras")

    def pin_lora(self, lora_id: int) -> bool:
        return self.call_utility("pin_lora", lora_id)

    def sleep(self, level: int = 1, mode: PauseMode = "abort") -> None:
        self.call_utility("sleep", level, mode)

    def wake_up(self, tags: list[str] | None = None) -> None:
        self.call_utility("wake_up", tags)

    def is_sleeping(self) -> bool:
        return self.call_utility("is_sleeping")

    def execute_dummy_batch(self) -> None:
        self.call_utility("execute_dummy_batch")

    def collective_rpc(
        self,
        method: str | Callable[..., _R],
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
    ) -> list[_R]:
        return self.call_utility("collective_rpc", method, timeout, args, kwargs)

    def save_sharded_state(
        self, path: str, pattern: str | None = None, max_size: int | None = None
    ) -> None:
        self.call_utility("save_sharded_state", path, pattern, max_size)


class AsyncMPClient(MPClient):
    """Asyncio-compatible client for multi-proc EngineCore."""

    # AsyncMPClient: 异步多进程客户端。
    # 用于 AsyncLLM 的生产环境场景，提供异步非阻塞的 API。
    #
    # 与 SyncMPClient 的区别：
    #   - 使用 asyncio.Queue 而非 queue.Queue
    #   - 使用 asyncio.Task 而非 Thread 处理输出
    #   - 所有 I/O 操作都是 awaitable 的
    #   - 支持高并发的异步请求处理
    #
    # 工作原理：
    #   1. _ensure_output_queue_task() 启动异步输出处理任务
    #   2. 输出处理任务从 output_socket 接收并解码输出
    #   3. 主协程通过 get_output_async() 从 asyncio.Queue 获取输出
    #   4. 工具方法通过 call_utility_async() 异步等待结果
    #
    # 惰性初始化：
    #   - 输出处理任务在第一次请求时才启动（如果不在事件循环中）
    #   - 这样可以避免在事件循环启动前创建任务

    @instrument(span_name="AsyncMPClient init")
    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        client_addresses: dict[str, Any] | None = None,
        client_count: int = 1,
        client_index: int = 0,
    ):
        super().__init__(
            asyncio_mode=True,
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=log_stats,
            client_addresses=client_addresses,
        )

        self.client_count = client_count
        self.client_index = client_index
        # 异步输出队列：输出处理任务放入，主协程取出
        self.outputs_queue = asyncio.Queue[EngineCoreOutputs | Exception]()
        try:
            # If we are running in an asyncio event loop, start the queue task.
            # Otherwise, it will be started lazily. If it is not started here,
            # we could miss EXECUTOR_FAILED messages from engine core if they
            # occur prior to any requests being sent.
            asyncio.get_running_loop()
            self._ensure_output_queue_task()
        except RuntimeError:
            pass

    def _ensure_output_queue_task(self):
        # 确保输出处理任务已启动。
        # 如果任务已存在则直接返回，否则创建并启动。
        resources = self.resources
        if resources.output_queue_task is not None:
            return

        # Perform IO in separate task to parallelize as much as possible.
        # Avoid task having direct reference back to the client.
        # 在独立任务中执行 I/O 以最大化并行性。
        # 避免任务直接引用客户端对象（使用 weakref）。
        decoder = self.decoder
        utility_results = self.utility_results
        outputs_queue = self.outputs_queue
        output_handler: (
            Callable[[AsyncMPClient, EngineCoreOutputs], Awaitable[None]] | None
        ) = getattr(self.__class__, "process_engine_outputs", None)
        _self_ref = weakref.ref(self) if output_handler else None
        output_socket = resources.output_socket
        assert output_socket is not None

        # 弹性 EP 通知回调处理器
        notification_callback_handler: (
            Callable[[AsyncMPClient, Sequence[Any]], Any] | None
        ) = getattr(self.__class__, "eep_process_engine_core_notification", None)

        async def process_outputs_socket():
            # 异步输出处理协程。
            # 持续从 output_socket 接收输出，处理逻辑：
            #   1. 弹性 EP 通知 -> 异步调用通知回调
            #   2. 工具方法结果 -> 设置 Future 结果
            #   3. 普通推理输出 -> 放入 asyncio.Queue
            #   4. 异常 -> 放入 asyncio.Queue 以通知主协程
            try:
                while True:
                    frames = await output_socket.recv_multipart(copy=False)
                    resources.validate_alive(frames)
                    outputs: EngineCoreOutputs = decoder.decode(frames)
                    if outputs.utility_output:
                        if (
                            outputs.utility_output.call_id == EEP_NOTIFICATION_CALL_ID
                            and notification_callback_handler is not None
                        ):
                            # 弹性 EP 通知处理
                            assert _self_ref is not None
                            _self = _self_ref()
                            if not _self:
                                return
                            if outputs.utility_output.result is None:
                                continue
                            notification_data = outputs.utility_output.result.result
                            assert isinstance(notification_data, Sequence)
                            assert len(notification_data) == 2
                            asyncio.create_task(
                                notification_callback_handler(_self, notification_data)
                            )
                        else:
                            _process_utility_output(
                                outputs.utility_output, utility_results
                            )
                        continue

                    if output_handler is not None:
                        # 子类自定义的输出处理（如 DPLBAsyncMPClient 的请求追踪）
                        assert _self_ref is not None
                        _self = _self_ref()
                        if not _self:
                            # Client has been garbage collected, abort.
                            return
                        await output_handler(_self, outputs)

                    if outputs.outputs or outputs.scheduler_stats:
                        outputs_queue.put_nowait(outputs)
            except Exception as e:
                outputs_queue.put_nowait(e)
            except asyncio.CancelledError:
                outputs_queue.put_nowait(EngineDeadError())

        # 创建异步任务并保存引用（防止被 GC）
        resources.output_queue_task = asyncio.create_task(
            process_outputs_socket(), name="EngineCoreOutputQueueTask"
        )

    async def get_output_async(self) -> EngineCoreOutputs:
        # 异步获取推理输出。
        # 从 asyncio.Queue 中等待并获取输出，确保输出处理任务已启动。
        self._ensure_output_queue_task()
        # If an exception arises in process_outputs_socket task,
        # it is forwarded to the outputs_queue so we can raise it
        # from this (run_output_handler) task to shut down the server.
        assert self.outputs_queue is not None
        outputs = await self.outputs_queue.get()
        if isinstance(outputs, Exception):
            raise self._format_exception(outputs) from None
        return outputs

    def _send_input(
        self,
        request_type: EngineCoreRequestType,
        request: Any,
        engine: EngineIdentity | None = None,
    ) -> Awaitable[Any]:
        # 发送输入到引擎（异步版本）。
        # 返回 Awaitable，可以被 await 以等待发送完成。
        if engine is None:
            engine = self.core_engine

        message = (request_type.value, *self.encoder.encode(request))
        return self._send_input_message(message, engine, request)

    def _send_input_message(
        self, message: tuple[bytestr, ...], engine: EngineIdentity, objects: Any
    ) -> Awaitable[Any]:
        """
        objects is a reference to retain until zmq is finished with the
        buffers, in case they were extracted from tensors in the request.
        """
        # 发送消息到引擎（底层方法）。
        #
        # 参数：
        #   - message: 序列化后的消息元组
        #   - engine: 目标引擎的 ZMQ 身份
        #   - objects: 请求对象引用（保持张量内存有效直到发送完成）
        #
        # 零拷贝发送优化：
        #   - 无张量时直接发送（copy=False）
        #   - 有张量时使用 track=True + done_callback 确保内存安全
        self.ensure_alive()
        self.free_pending_messages()

        msg = (engine,) + message
        if not objects or len(msg) <= 3:
            # No auxiliary buffers => no tensor backing buffers in request.
            return self.input_socket.send_multipart(msg, copy=False)

        future: asyncio.Future[zmq.MessageTracker]
        future = self.input_socket.send_multipart(msg, copy=False, track=True)

        def add_pending(f: asyncio.Future[zmq.MessageTracker]):
            with contextlib.suppress(BaseException):
                self.add_pending_message(f.result(), objects)

        future.add_done_callback(add_pending)
        return future

    async def call_utility_async(self, method: str, *args) -> Any:
        # 异步调用工具方法（默认引擎）
        return await self._call_utility_async(method, *args, engine=self.core_engine)

    async def _call_utility_async(
        self, method: str, *args, engine: EngineIdentity
    ) -> Any:
        # 异步调用工具方法（底层实现）。
        #
        # 流程：
        #   1. 生成 call_id，创建 asyncio.Future
        #   2. 发送 UTILITY 请求到指定引擎
        #   3. 确保输出处理任务已启动
        #   4. await Future 等待结果
        #
        # 与同步版本的区别：
        #   - 使用 asyncio.Future 而非 concurrent.futures.Future
        #   - 使用 await 而非阻塞等待
        call_id = uuid.uuid1().int >> 64
        future = asyncio.get_running_loop().create_future()
        self.utility_results[call_id] = future
        message = (
            EngineCoreRequestType.UTILITY.value,
            *self.encoder.encode((self.client_index, call_id, method, args)),
        )
        await self._send_input_message(message, engine, args)
        self._ensure_output_queue_task()
        return await future

    async def get_supported_tasks_async(self) -> tuple[SupportedTask, ...]:
        return await self.call_utility_async("get_supported_tasks")

    async def add_request_async(self, request: EngineCoreRequest) -> None:
        request.client_index = self.client_index
        await self._send_input(EngineCoreRequestType.ADD, request)
        self._ensure_output_queue_task()

    async def abort_requests_async(self, request_ids: list[str]) -> None:
        if request_ids and not self.resources.engine_dead:
            await self._send_input(EngineCoreRequestType.ABORT, request_ids)

    async def pause_scheduler_async(
        self, mode: PauseMode = "abort", clear_cache: bool = True
    ) -> None:
        await self.call_utility_async("pause_scheduler", mode, clear_cache)

    async def resume_scheduler_async(self) -> None:
        await self.call_utility_async("resume_scheduler")

    async def is_scheduler_paused_async(self) -> bool:
        return await self.call_utility_async("is_scheduler_paused")

    async def profile_async(
        self, is_start: bool = True, profile_prefix: str | None = None
    ) -> None:
        await self.call_utility_async("profile", is_start, profile_prefix)

    async def reset_mm_cache_async(self) -> None:
        await self.call_utility_async("reset_mm_cache")

    async def reset_prefix_cache_async(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        return await self.call_utility_async(
            "reset_prefix_cache", reset_running_requests, reset_connector
        )

    async def reset_encoder_cache_async(self) -> None:
        await self.call_utility_async("reset_encoder_cache")

    async def sleep_async(self, level: int = 1, mode: PauseMode = "abort") -> None:
        await self.call_utility_async("sleep", level, mode)

    async def wake_up_async(self, tags: list[str] | None = None) -> None:
        await self.call_utility_async("wake_up", tags)

    async def is_sleeping_async(self) -> bool:
        return await self.call_utility_async("is_sleeping")

    async def execute_dummy_batch_async(self) -> None:
        await self.call_utility_async("execute_dummy_batch")

    async def add_lora_async(self, lora_request: LoRARequest) -> bool:
        return await self.call_utility_async("add_lora", lora_request)

    async def remove_lora_async(self, lora_id: int) -> bool:
        return await self.call_utility_async("remove_lora", lora_id)

    async def list_loras_async(self) -> set[int]:
        return await self.call_utility_async("list_loras")

    async def pin_lora_async(self, lora_id: int) -> bool:
        return await self.call_utility_async("pin_lora", lora_id)

    async def save_sharded_state_async(
        self, path: str, pattern: str | None = None, max_size: int | None = None
    ) -> None:
        await self.call_utility_async("save_sharded_state", path, pattern, max_size)

    async def collective_rpc_async(
        self,
        method: str | Callable[..., _R],
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
    ) -> list[_R]:
        return await self.call_utility_async(
            "collective_rpc", method, timeout, args, kwargs
        )


class DPAsyncMPClient(AsyncMPClient):
    """Asyncio-compatible client for multi-proc, multi-engine (data parallel)
    EngineCore. Assumes external load-balancing by default."""

    # DPAsyncMPClient: 数据并行异步客户端（外部负载均衡模式）。
    # 用于数据并行 (Data Parallel) 场景，支持多个 EngineCore 并行处理请求。
    #
    # 与普通 AsyncMPClient 的区别：
    #   1. 支持多个引擎（core_engines 列表）
    #   2. 支持引擎暂停/唤醒协调
    #   3. 支持弹性 EP 扩缩容
    #   4. 统计信息更新任务（从协调器接收负载信息）
    #
    # 引擎唤醒机制：
    #   - 引擎可能处于暂停状态（sleep）
    #   - 发送第一个请求时需要通知协调器唤醒引擎
    #   - 通过 first_req_send_socket 发送唤醒信号
    #   - 协调器通知所有引擎执行 dummy EP 循环

    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        client_addresses: dict[str, Any] | None = None,
        client_count: int = 1,
        client_index: int = 0,
    ):
        # current_wave: 当前推理波次，用于协调多引擎的调度
        self.current_wave = 0

        super().__init__(
            vllm_config,
            executor_class,
            log_stats,
            client_addresses,
            client_count,
            client_index,
        )

        # List of [waiting, running] pair per engine.
        # Used only by DPLBAsyncMPClient subclass.
        # lb_engines: 每个引擎的负载信息 [waiting, running]。
        # 用于 DPLBAsyncMPClient 的负载均衡决策。
        self.lb_engines: list[list[int]] = [[0, 0] for _ in self.core_engines]

        # eep_scaling_cache: 弹性 EP 扩缩容缓存
        self.eep_scaling_cache: ElasticScalingCache | None = None

        # first_req_sock_addr: 用于发送第一个请求通知的 ZMQ 地址
        self.first_req_sock_addr = get_open_zmq_inproc_path()
        self.first_req_send_socket = self.resources.first_req_send_socket = (
            make_zmq_socket(self.ctx, self.first_req_sock_addr, zmq.PAIR, bind=True)
        )
        try:
            # If we are running in an asyncio event loop, start the stats task.
            # Otherwise, it will be started lazily.
            asyncio.get_running_loop()
            self._ensure_stats_update_task()
        except RuntimeError:
            pass

    def _ensure_stats_update_task(self):
        # 确保统计信息更新任务已启动。
        # 此任务负责：
        #   1. 订阅协调器的统计信息（负载、波次、运行状态）
        #   2. 处理第一个请求的通知（唤醒暂停的引擎）
        #   3. 处理弹性 EP 扩缩容通知
        resources = self.resources
        if resources.stats_update_task is not None:
            return

        assert self.stats_update_address is not None
        stats_addr: str = self.stats_update_address
        assert len(self.engine_ranks_managed) > 0

        async def run_engine_stats_update_task():
            # 统计信息更新任务主协程。
            #
            # 使用两个 ZMQ 套接字：
            #   1. XSUB 套接字：订阅协调器的统计信息
            #   2. PAIR 套接字：接收第一个请求通知和扩缩容通知
            #
            # 统计信息格式：(counts, wave, running)
            #   - counts: [[waiting, running], ...] 每个引擎的负载
            #   - wave: 当前推理波次
            #   - running: 引擎是否正在运行
            with (
                make_zmq_socket(self.ctx, stats_addr, zmq.XSUB, linger=0) as socket,
                make_zmq_socket(
                    self.ctx, self.first_req_sock_addr, zmq.PAIR, bind=False, linger=0
                ) as first_req_rcv_socket,
            ):
                assert isinstance(socket, zmq.asyncio.Socket)
                assert isinstance(first_req_rcv_socket, zmq.asyncio.Socket)
                self.resources.stats_update_socket = socket
                self.resources.first_req_rcv_socket = first_req_rcv_socket
                # Send subscription message.
                # XSUB 套接字需要发送订阅消息（\x01 表示订阅所有）
                await socket.send(b"\x01")

                poller = zmq.asyncio.Poller()
                poller.register(socket, zmq.POLLIN)
                poller.register(first_req_rcv_socket, zmq.POLLIN)

                while True:
                    events = await poller.poll()
                    if (
                        not self.engines_running
                        and len(events) == 2
                        or (events[0][0] == first_req_rcv_socket)
                    ):
                        # Check if this is a regular request notification or
                        # scale up notification
                        # 检查是普通请求通知还是扩缩容通知
                        buf = first_req_rcv_socket.recv(flags=zmq.NOBLOCK).result()

                        decoded = msgspec.msgpack.decode(buf)
                        if (
                            isinstance(decoded, (list, tuple))
                            and len(decoded) == 2
                            and decoded[0] == "SCALE_ELASTIC_EP"
                        ):
                            # 弹性 EP 扩缩容通知
                            # Extract new engine count from the decoded message
                            new_engine_count = decoded[1]
                            # Update engine_ranks_managed and count_slice
                            parallel_config = self.vllm_config.parallel_config
                            dp_size = parallel_config.data_parallel_size
                            dp_rank = parallel_config.data_parallel_rank
                            assert dp_rank == 0
                            assert dp_size == new_engine_count
                            assert not (
                                parallel_config.data_parallel_hybrid_lb
                                or parallel_config.data_parallel_external_lb
                            )
                            num_ranks = dp_size
                            self.engine_ranks_managed = list(
                                range(dp_rank, dp_rank + num_ranks)
                            )
                            if len(self.lb_engines) < new_engine_count:
                                self.lb_engines = self.lb_engines + [
                                    [0, 0]
                                    for _ in range(
                                        new_engine_count - len(self.lb_engines)
                                    )
                                ]
                            else:
                                self.lb_engines = self.lb_engines[:new_engine_count]
                            # Send scale up notification to coordinator
                            scale_msg = msgspec.msgpack.encode(
                                ("SCALE_ELASTIC_EP", new_engine_count)
                            )
                            await socket.send(scale_msg)
                            continue

                        # we're sending a request while the engines are
                        # paused, so that it can wake the others up
                        # (to run dummy EP loop).
                        # 第一个请求通知：唤醒暂停的引擎
                        assert decoded[0] == "FIRST_REQ"
                        target_eng_index = decoded[1]
                        self.engines_running = True
                        msg = msgspec.msgpack.encode(
                            (target_eng_index, self.current_wave)
                        )
                        await socket.send(msg)

                    buf = None
                    while True:
                        # Drain all stats events (we only care about latest).
                        # 排空所有统计事件，只保留最新的
                        future: asyncio.Future[bytes] = socket.recv(flags=zmq.NOBLOCK)
                        if isinstance(future.exception(), zmq.Again):
                            break
                        buf = future.result()
                    if buf is None:
                        continue

                    # Update local load-balancing state.
                    # 更新本地负载均衡状态
                    counts, wave, running = msgspec.msgpack.decode(buf)
                    self.current_wave = wave
                    self.engines_running = running
                    if counts is not None:
                        # Running and waiting counts are global from the
                        # Coordinator including all EngineCores. Slice to get
                        # just the cores managed by this client.
                        # 从全局负载信息中切片获取此客户端管理的引擎负载
                        ranks = self.engine_ranks_managed
                        count_slice = slice(ranks[0], ranks[-1] + 1)
                        sliced_counts = counts[count_slice]
                        self.lb_engines = sliced_counts
                        logger.debug(
                            "Received counts: %s (%s)", sliced_counts, count_slice
                        )

        resources.stats_update_task = asyncio.create_task(
            run_engine_stats_update_task()
        )

    async def add_request_async(self, request: EngineCoreRequest) -> None:
        # 异步添加推理请求（数据并行版本）。
        #
        # 与普通 AsyncMPClient 的区别：
        #   1. 设置当前波次信息（用于协调多引擎）
        #   2. 选择目标引擎（子类可覆盖选择策略）
        #   3. 如果引擎暂停，通知协调器唤醒
        self._ensure_stats_update_task()

        request.current_wave = self.current_wave
        request.client_index = self.client_index

        # 选择目标引擎（默认为第一个，DPLBAsyncMPClient 覆盖为负载均衡选择）
        chosen_engine = self.get_core_engine_for_request(request)
        to_await = self._send_input(EngineCoreRequestType.ADD, request, chosen_engine)
        if not self.engines_running:
            # Notify coordinator that we're sending a request
            # 引擎暂停时，通知协调器唤醒所有引擎
            req_msg = msgspec.msgpack.encode(("FIRST_REQ", chosen_engine))
            await self.first_req_send_socket.send(req_msg)

        await to_await

        self._ensure_output_queue_task()

    def get_core_engine_for_request(self, request: EngineCoreRequest):
        # 默认返回第一个引擎（外部 LB 模式下由外部决定路由）
        return self.core_engine


class DPLBAsyncMPClient(DPAsyncMPClient):
    """Asyncio-compatible client for multi-proc, multi-engine (data parallel)
    EngineCore. Load-balances between multiple engine processes."""

    # DPLBAsyncMPClient: 数据并行异步客户端（内部负载均衡模式）。
    # 在多个引擎进程之间进行负载均衡，选择最优引擎处理请求。
    #
    # 负载均衡算法：
    #   - 简单的加权评分：score = waiting * 4 + running
    #   - 选择分数最低（最空闲）的引擎
    #   - 从 client_index 开始遍历，帮助多客户端场景下的均衡
    #
    # 特殊路由：
    #   - 显式指定 dp_rank 的请求直接路由到对应引擎
    #   - Late interaction 求 pooling 请求路由到特定引擎
    #
    # 请求追踪：
    #   - reqs_in_flight 追踪每个请求被路由到哪个引擎
    #   - 用于中止请求时能找到正确的引擎

    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        client_addresses: dict[str, Any] | None = None,
        client_count: int = 1,
        client_index: int = 0,
    ):
        self.client_count = client_count

        # To route aborts to the correct engine.
        # reqs_in_flight: 请求到引擎的映射，用于中止请求时路由到正确的引擎
        self.reqs_in_flight: dict[str, EngineIdentity] = {}

        super().__init__(
            vllm_config,
            executor_class,
            log_stats,
            client_addresses,
            client_count,
            client_index,
        )

        assert len(self.core_engines) > 1

        # eng_start_index: 此客户端的起始引擎索引，
        # 用于多客户端场景下的负载分散
        self.eng_start_index = (
            len(self.core_engines) * self.client_index
        ) // client_count

    def get_core_engine_for_request(self, request: EngineCoreRequest) -> EngineIdentity:
        # 为请求选择目标引擎（负载均衡策略）。
        #
        # 选择优先级：
        #   1. 显式指定 data_parallel_rank -> 直接使用
        #   2. Late interaction pooling -> 根据 pool 参数选择
        #   3. 默认 -> 加权评分选择最空闲的引擎
        # Engines are in rank order.
        if (eng_index := request.data_parallel_rank) is None and (
            eng_index := get_late_interaction_engine_index(
                request.pooling_params, len(self.core_engines)
            )
        ) is None:
            current_counts = self.lb_engines
            # TODO use P2C alg for larger DP sizes
            # 加权评分选择：waiting * 4 + running
            # waiting 权重更高，因为等待队列对延迟影响更大
            num_engines = len(current_counts)
            min_score = sys.maxsize
            eng_index = 0
            for i in range(num_engines):
                # Start from client_index to help with balancing when engines
                # are empty.
                # 从 client_index 开始遍历，帮助多客户端场景下的均衡
                idx = (self.eng_start_index + i) % num_engines
                waiting, running = current_counts[idx]
                score = waiting * 4 + running
                if score < min_score:
                    min_score = score
                    eng_index = idx
            # Increment local waiting count for better balancing between stats
            # updates from the coordinator (which happen every 100ms).
            # 本地递增等待计数，改善协调器统计更新间隔（100ms）内的均衡
            current_counts[eng_index][0] += self.client_count

        chosen_engine = self.core_engines[eng_index]
        # Record which engine is chosen for this request, to handle aborts.
        # 记录请求被路由到哪个引擎，用于中止请求时能找到正确的引擎
        self.reqs_in_flight[request.request_id] = chosen_engine
        return chosen_engine

    async def call_utility_async(self, method: str, *args) -> Any:
        # 向所有引擎发送工具方法调用，只返回第一个引擎的结果。
        # 使用 asyncio.gather 并行发送到所有引擎。
        return (
            await asyncio.gather(
                *[
                    self._call_utility_async(method, *args, engine=engine)
                    for engine in self.core_engines
                ]
            )
        )[0]

    @staticmethod
    async def process_engine_outputs(
        self: "DPLBAsyncMPClient", outputs: EngineCoreOutputs
    ):
        # 处理引擎输出，清理已完成请求的追踪记录。
        # 当请求完成时，从 reqs_in_flight 中移除。
        if outputs.finished_requests and self.reqs_in_flight:
            for req_id in outputs.finished_requests:
                self.reqs_in_flight.pop(req_id, None)

    @staticmethod
    async def eep_process_engine_core_notification(
        self: "DPLBAsyncMPClient", notification_data: tuple[str, int]
    ):
        # 处理弹性 EP 引擎核心通知。
        #
        # 通知类型：
        #   1. RECONFIGURE_FINISHED: 重配置完成，解决等待的 Future
        #   2. SHUTDOWN_COMPLETE: 缩容完成，移除引擎
        #   3. NEW_CORE_ENGINES_WEIGHTS_INIT_READY: 新引擎权重初始化完成
        #   4. 其他通知: 转发给所有现有引擎处理
        #
        # 等待机制：
        #   - 收集所有引擎的通知
        #   - 当所有引擎都发送了同类型通知后，执行相应操作
        cache = self.eep_scaling_cache
        notification_type_str, dp_rank = notification_data
        try:
            notification_type = EEPNotificationType(notification_type_str)
        except ValueError as e:
            raise ValueError(
                f"Unknown EEP notification type: {notification_type_str}"
            ) from e

        if notification_type == EEPNotificationType.RECONFIGURE_FINISHED:
            from vllm.v1.engine import UtilityResult

            # NOTE(yongji): process a dummy UtilityOutput to resolve the future
            # awaited in _eep_wait_for_setup_switch_complete(), signaling that
            # all engine cores have completed reconfiguration.
            # 处理虚拟输出以解决 _eep_wait_for_setup_switch_complete() 中等待的 Future
            dummy_output = UtilityOutput(
                call_id=EEP_NOTIFICATION_CALL_ID, result=UtilityResult(None)
            )
            _process_utility_output(dummy_output, self.utility_results)
            return
        assert cache is not None
        if notification_type not in cache.pending_notifications:
            cache.pending_notifications[notification_type] = set()
        if dp_rank in cache.pending_notifications[notification_type]:
            raise ValueError(
                f"Duplicate notification {notification_type} from dp_rank {dp_rank}"
            )
        cache.pending_notifications[notification_type].add(dp_rank)
        if len(cache.pending_notifications[notification_type]) >= abs(
            cache.num_new_core_engines
        ):
            # 所有引擎都已发送通知，执行相应操作
            if notification_type == EEPNotificationType.SHUTDOWN_COMPLETE:
                # 缩容完成：移除引擎
                assert isinstance(self.resources.engine_manager, CoreEngineActorManager)
                assert cache.num_new_core_engines < 0
                old_dp_size = len(cache.existing_core_engines)
                new_dp_size = old_dp_size + cache.num_new_core_engines
                self.resources.engine_manager.scale_down_elastic_ep(
                    old_dp_size, new_dp_size
                )
            else:
                # 其他通知：转发给所有现有引擎处理
                await asyncio.gather(
                    *[
                        self._call_utility_async(
                            "eep_handle_engine_core_notification",
                            notification_type,
                            engine=engine,
                        )
                        for engine in cache.existing_core_engines
                    ]
                )
            cache.pending_notifications[notification_type] = set()
            if notification_type in [
                EEPNotificationType.SHUTDOWN_COMPLETE,
                EEPNotificationType.NEW_CORE_ENGINES_WEIGHTS_INIT_READY,
            ]:
                # 最终通知，清除缓存
                self.eep_scaling_cache = None

    async def abort_requests_async(self, request_ids: list[str]) -> None:
        # 异步中止请求（数据并行版本）。
        # 根据 reqs_in_flight 将中止请求路由到正确的引擎。
        if not request_ids or self.resources.engine_dead:
            return

        if len(request_ids) == 1:
            # Fast-path common case.
            # 快速路径：单个请求直接查找
            if engine := self.reqs_in_flight.get(request_ids[0]):
                await self._abort_requests(request_ids, engine)
            return

        # 多个请求：按引擎分组后批量中止
        by_engine = defaultdict[EngineIdentity, list[str]](list)
        for req_id in request_ids:
            if engine := self.reqs_in_flight.get(req_id):
                by_engine[engine].append(req_id)
        for engine, req_ids in by_engine.items():
            await self._abort_requests(req_ids, engine)

    async def _abort_requests(
        self, request_ids: list[str], engine: EngineIdentity
    ) -> None:
        # 发送中止请求到指定引擎
        await self._send_input(EngineCoreRequestType.ABORT, request_ids, engine)

    async def scale_elastic_ep(self, new_data_parallel_size: int) -> None:
        """Scale elastic EP data parallel size"""
        # 弹性 EP 扩缩容入口。
        #
        # 参数：
        #   - new_data_parallel_size: 新的 DP 大小
        #
        # 前置条件：
        #   - 新大小必须与当前不同
        #   - 必须使用 Ray DP 后端
        #
        # 根据新大小是增加还是减少，调用对应的扩/缩容方法。
        cur_data_parallel_size = len(self.core_engines)

        assert new_data_parallel_size != cur_data_parallel_size, (
            f"new_data_parallel_size {new_data_parallel_size} must be "
            f"different from cur_data_parallel_size {cur_data_parallel_size}"
        )

        assert self.vllm_config.parallel_config.data_parallel_backend == "ray", (
            "Only ray DP backend supports scaling elastic EP"
        )

        scale_up = new_data_parallel_size > cur_data_parallel_size

        if scale_up:
            await self._scale_up_elastic_ep(
                cur_data_parallel_size, new_data_parallel_size
            )
        else:
            await self._scale_down_elastic_ep(
                cur_data_parallel_size, new_data_parallel_size
            )

    async def _eep_wait_for_setup_switch_complete(self) -> None:
        """
        Wait for core engines to switch to the new setup.

        In eep_process_engine_core_notification(), a dummy UtilityOutput with
        EEP_NOTIFICATION_CALL_ID will be set when RECONFIGURE_FINISHED
        notification is received from engine 0. We create a future with
        that call_id and wait for it to be resolved.
        """
        # 等待所有引擎完成重配置切换。
        #
        # 机制：
        #   1. 创建一个 Future 并注册到 utility_results[EEP_NOTIFICATION_CALL_ID]
        #   2. 当引擎发送 RECONFIGURE_FINISHED 通知时，
        #      eep_process_engine_core_notification() 会解决此 Future
        #   3. await 此 Future 阻塞直到所有引擎完成切换
        future = asyncio.get_running_loop().create_future()
        self.utility_results[EEP_NOTIFICATION_CALL_ID] = future
        self._ensure_output_queue_task()
        await future

    def _setup_elastic_ep_reconfig_bootstrap(self) -> tuple[str, int]:
        # 设置弹性 EP 重配置的引导信息。
        #
        # 创建 TCP Store 用于引擎间的协调通信。
        # 返回 (ip, port) 供引擎连接。
        from vllm.distributed.utils import create_tcp_store
        from vllm.utils.network_utils import get_open_ports_list

        parallel_config = self.vllm_config.parallel_config
        parallel_config._data_parallel_master_port_list = get_open_ports_list(5)
        parallel_config.data_parallel_master_port = (
            parallel_config._data_parallel_master_port_list.pop()
        )

        ip = parallel_config.data_parallel_master_ip
        store = create_tcp_store(
            ip,
            0,
            is_master=True,
            world_size=-1,
            wait_for_workers=False,
        )
        parallel_config._coord_store_port = store.port
        self._coord_store = store
        return ip, store.port

    async def _scale_up_elastic_ep(
        self, cur_data_parallel_size: int, new_data_parallel_size: int
    ) -> None:
        """Scale up the data parallel size by creating new engine cores
        and reconfiguring existing ones."""
        # 弹性 EP 扩容。
        #
        # 扩容流程（三阶段）：
        #   阶段 1: 向现有引擎发送重配置消息
        #     - 通知引擎新的 DP 大小和协调信息
        #     - 引擎准备接受新引擎加入
        #
        #   阶段 2: 创建新引擎
        #     - 通过 CoreEngineActorManager 启动新引擎进程
        #     - 新引擎初始化模型和权重
        #
        #   阶段 3: 等待完成
        #     - 等待新引擎发送就绪消息
        #     - 等待所有引擎完成重配置切换
        #     - 更新配置并通知协调器
        cur_data_parallel_size = len(self.core_engines)

        self.eep_scaling_cache = ElasticScalingCache(
            existing_core_engines=self.core_engines.copy(),
            num_new_core_engines=new_data_parallel_size - cur_data_parallel_size,
            pending_notifications=dict(),
        )

        parallel_config = self.vllm_config.parallel_config
        ip, coord_store_port = self._setup_elastic_ep_reconfig_bootstrap()

        # Phase 1: Send reconfig messages to existing engines
        # 阶段 1: 向现有引擎发送重配置消息
        reconfig_futures = []
        for engine in self.core_engines:
            reconfig_request = ReconfigureDistributedRequest(
                new_data_parallel_size=new_data_parallel_size,
                new_data_parallel_rank=ReconfigureRankType.KEEP_CURRENT_RANK,
                new_data_parallel_rank_local=ReconfigureRankType.KEEP_CURRENT_RANK,
                new_data_parallel_master_ip=ip,
                new_data_parallel_master_port=parallel_config.data_parallel_master_port,
                new_data_parallel_master_port_list=parallel_config._data_parallel_master_port_list,
                coord_store_port=coord_store_port,
            )
            coro = self._call_utility_async(
                "reinitialize_distributed", reconfig_request, engine=engine
            )
            reconfig_futures.append(asyncio.create_task(coro))

        # Phase 2: Create new engines
        # 阶段 2: 创建新引擎
        assert isinstance(self.resources.engine_manager, CoreEngineActorManager)
        parallel_config.eplb_config.num_redundant_experts = 0
        start_new_worker_future = asyncio.to_thread(
            self.resources.engine_manager.scale_up_elastic_ep,
            self.vllm_config,
            new_data_parallel_size,
        )
        wait_future = self._eep_wait_for_setup_switch_complete()

        # Phase 3: Wait for new engines to be created
        # and reconfig messages to be received
        # 阶段 3: 等待新引擎创建和重配置完成
        await asyncio.gather(start_new_worker_future, *reconfig_futures)
        logger.info("[Elastic EP] Successfully started new engines")

        # Create new CoreEngine objects for the new engines
        # 为新引擎创建 CoreEngine 对象
        new_engine_identities = set()
        for i in range(cur_data_parallel_size, new_data_parallel_size):
            new_engine = i.to_bytes(2, "little")
            self.core_engines.append(new_engine)
            # NOTE(yongji): we don't update lb_engines here,
            # we let run_engine_stats_update_task to update it.
            new_engine_identities.add(new_engine)

        # Wait for ready messages from new engines on the input socket
        # 等待新引擎发送就绪消息
        sync_input_socket = zmq.Socket.shadow(self.input_socket)
        while new_engine_identities:
            if not sync_input_socket.poll(
                timeout=VLLM_ENGINE_READY_TIMEOUT_S * 1000  # convert to ms
            ):
                raise TimeoutError(
                    f"Timed out waiting for new engine core processes to "
                    f"start. Waited "
                    f"{VLLM_ENGINE_READY_TIMEOUT_S}s (configured by "
                    f"VLLM_ENGINE_READY_TIMEOUT_S). To increase the "
                    f"timeout, set the environment variable: "
                    f"VLLM_ENGINE_READY_TIMEOUT_S=<seconds>"
                )
            identity, payload = sync_input_socket.recv_multipart()
            new_engine_identities.discard(identity)
            self._apply_ready_response(payload)

        # NOTE(yongji): Before we schedule any requests on the new workers,
        # we should wait for them to switch to the new setup.
        # 等待所有引擎完成重配置切换，然后再调度请求到新引擎
        await wait_future
        # Update the parallel config
        self.vllm_config.parallel_config.data_parallel_size = new_data_parallel_size
        # Notify coordinator about scale up through existing
        # stats_update_task connection
        # 通知协调器扩容完成
        self._ensure_stats_update_task()
        scale_up_marker = msgspec.msgpack.encode(
            ("SCALE_ELASTIC_EP", new_data_parallel_size)
        )
        await self.first_req_send_socket.send(scale_up_marker)

        logger.info(
            "[Elastic EP] Scale up completed, new data parallel size: %s",
            new_data_parallel_size,
        )

    async def _scale_down_elastic_ep(
        self, cur_data_parallel_size: int, new_data_parallel_size: int
    ) -> None:
        """Scale down the data parallel size by shutting down and
        reconfiguring existing engine cores."""
        # 弹性 EP 缩容。
        #
        # 缩容流程：
        #   1. 创建扩缩容缓存
        #   2. 设置引导信息
        #   3. 移除待删除引擎的运行引用
        #   4. 向所有引擎发送重配置消息
        #      - 保留的引擎：KEEP_CURRENT_RANK
        #      - 待删除的引擎：SHUTDOWN_CURRENT_RANK
        #   5. 立即停止向待删除引擎发送请求
        #   6. 等待重配置完成
        #   7. 更新配置并通知协调器
        cur_data_parallel_size = len(self.core_engines)

        self.eep_scaling_cache = ElasticScalingCache(
            existing_core_engines=self.core_engines.copy(),
            num_new_core_engines=new_data_parallel_size - cur_data_parallel_size,
            pending_notifications=dict(),
        )

        parallel_config = self.vllm_config.parallel_config
        ip, coord_store_port = self._setup_elastic_ep_reconfig_bootstrap()

        removed_dp_size = cur_data_parallel_size - new_data_parallel_size
        assert isinstance(self.resources.engine_manager, CoreEngineActorManager)
        self.resources.engine_manager.remove_run_refs_for_scale_down(removed_dp_size)
        reconfig_futures = []
        for cur_dp_rank, engine in enumerate(self.core_engines):
            reconfig_request = ReconfigureDistributedRequest(
                new_data_parallel_size=new_data_parallel_size,
                new_data_parallel_rank=ReconfigureRankType.KEEP_CURRENT_RANK,
                new_data_parallel_rank_local=ReconfigureRankType.KEEP_CURRENT_RANK,
                new_data_parallel_master_ip=ip,
                new_data_parallel_master_port=parallel_config.data_parallel_master_port,
                new_data_parallel_master_port_list=parallel_config._data_parallel_master_port_list,
                coord_store_port=coord_store_port,
            )
            if cur_dp_rank >= new_data_parallel_size:
                # 待删除引擎：标记为 SHUTDOWN_CURRENT_RANK
                reconfig_request.new_data_parallel_rank = (
                    ReconfigureRankType.SHUTDOWN_CURRENT_RANK
                )
            coro = self._call_utility_async(
                "reinitialize_distributed", reconfig_request, engine=engine
            )
            reconfig_futures.append(asyncio.create_task(coro))

        # NOTE(yongji): Immediately stop sending requests to the removing engines.
        # 立即停止向待删除引擎发送请求
        self.core_engines = self.core_engines[:new_data_parallel_size]
        self.lb_engines = self.lb_engines[:new_data_parallel_size]
        wait_future = self._eep_wait_for_setup_switch_complete()

        await asyncio.gather(*reconfig_futures)

        self.vllm_config.parallel_config.data_parallel_size = new_data_parallel_size
        self._ensure_stats_update_task()
        scale_down_marker = msgspec.msgpack.encode(
            ("SCALE_ELASTIC_EP", new_data_parallel_size)
        )
        await self.first_req_send_socket.send(scale_down_marker)

        # NOTE(yongji): Unlike scaling up,
        # here we don't actually need to wait for the setup switch to complete.
        # We may want to remove it in the future.
        # 与扩容不同，缩容时不需要等待切换完成
        await wait_future
        logger.info(
            "[Elastic EP] Scale down completed, new data parallel size: %s",
            new_data_parallel_size,
        )
