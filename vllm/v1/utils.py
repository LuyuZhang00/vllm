# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
v1 引擎工具模块 (vllm/v1/utils.py)

本模块提供 vLLM v1 引擎使用的核心工具类和函数。

模块包含以下组件：

1. ConstantList: 不可变列表
   - 包装一个普通列表，禁止所有修改操作（append、extend、pop 等）
   - 用于暴露不应被外部修改的数据，如模型配置列表
   - 支持只读操作（索引、迭代、包含检查等）

2. CpuGpuBuffer: CPU-GPU 缓冲区
   - 在 CPU 和 GPU 上各分配一个相同大小的张量
   - 提供 copy_to_gpu() 和 copy_to_cpu() 方法进行数据传输
   - 支持 numpy 数组视图（用于高效的 CPU 端操作）

3. APIServerProcessManager: API 服务器进程管理器
   - 管理一组 API 服务器工作进程的创建、监控和终止
   - 支持 gather_actual_addresses() 收集子进程绑定的实际地址
   - 使用 weakref.finalize 确保垃圾回收时自动关闭进程

4. RustFrontendProcessManager: Rust 前端进程管理器
   - 管理 Rust vllm-rs 二进制文件的前端进程
   - 提供与 APIServerProcessManager 相同的监控接口

5. wait_for_completion_or_failure: 进程监控函数
   - 等待所有进程完成或检测进程失败
   - 使用 multiprocessing.connection.wait() 高效监控

6. 工具函数:
   - get_engine_client_zmq_addr: 获取 ZMQ IPC 地址
   - copy_slice: 非阻塞的张量切片拷贝
   - report_usage_stats: 上报使用统计
   - record_function_or_nullcontext: 性能分析上下文管理器
   - tensor_data: 获取张量的原始数据（用于序列化和哈希）
   - compute_iteration_details: 计算迭代的上下文/生成请求数和 token 数
"""

import argparse
import contextlib
import json
import multiprocessing
import threading
import time
import weakref
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from multiprocessing import connection
from multiprocessing.process import BaseProcess
from multiprocessing.queues import Queue
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    TypeVar,
    Union,
    overload,
)

import torch
import uvloop
from torch.autograd.profiler import record_function

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.usage.usage_lib import UsageContext, is_usage_stats_enabled, usage_message
from vllm.utils.network_utils import get_open_zmq_ipc_path, get_tcp_uri
from vllm.utils.system_utils import decorate_logs, kill_process_tree, set_process_title
from vllm.v1.core.sched.output import SchedulerOutput

if TYPE_CHECKING:
    import numpy as np

    from vllm.v1.engine.coordinator import DPCoordinator
    from vllm.v1.engine.utils import CoreEngineActorManager, CoreEngineProcManager

logger = init_logger(__name__)

T = TypeVar("T")


class ConstantList(Generic[T], Sequence):
    """
    不可变列表，包装一个普通列表并禁止所有修改操作。

    设计目的：某些配置数据在初始化后不应被修改，使用 ConstantList 可以在运行时
    检测意外的修改尝试并抛出 TypeError。

    支持的操作：
    - 只读：索引、切片、迭代、包含检查、长度查询、copy
    - 禁止：append、extend、insert、pop、remove、clear、__setitem__、__delitem__

    使用方式：
        config_list = ConstantList([1, 2, 3])
        print(config_list[0])  # OK
        config_list.append(4)  # TypeError
    """

    def __init__(self, x: list[T]) -> None:
        self._x = x

    def append(self, item):
        raise TypeError("Cannot append to a constant list")

    def extend(self, item):
        raise TypeError("Cannot extend a constant list")

    def insert(self, item):
        raise TypeError("Cannot insert into a constant list")

    def pop(self, item):
        raise TypeError("Cannot pop from a constant list")

    def remove(self, item):
        raise TypeError("Cannot remove from a constant list")

    def clear(self):
        raise TypeError("Cannot clear a constant list")

    def index(self, item: T, start: int = 0, stop: int | None = None) -> int:
        return self._x.index(item, start, stop if stop is not None else len(self._x))

    @overload
    def __getitem__(self, item: int) -> T: ...

    @overload
    def __getitem__(self, s: slice, /) -> list[T]: ...

    def __getitem__(self, item: int | slice) -> T | list[T]:
        return self._x[item]

    @overload
    def __setitem__(self, item: int, value: T): ...

    @overload
    def __setitem__(self, s: slice, value: T, /): ...

    def __setitem__(self, item: int | slice, value: T | list[T]):
        raise TypeError("Cannot set item in a constant list")

    def __delitem__(self, item):
        raise TypeError("Cannot delete item from a constant list")

    def __iter__(self):
        return iter(self._x)

    def __contains__(self, item):
        return item in self._x

    def __len__(self):
        return len(self._x)

    def __repr__(self):
        return f"ConstantList({self._x})"

    def copy(self) -> list[T]:
        return self._x.copy()


class CpuGpuBuffer:
    """在 CPU 和 GPU 上各分配一个相同大小的张量，便于数据传输。

    设计目的：
    - 在模型推理中，很多数据（如 token IDs、位置信息等）需要从 CPU 传到 GPU
    - CpuGpuBuffer 预分配好两个张量，避免每次传输都重新分配内存
    - copy_to_gpu/copy_to_cpu 使用 non_blocking 实现异步传输

    属性：
        cpu: CPU 上的张量（可使用锁页内存加速传输）
        gpu: GPU 上的张量
        np: CPU 张量的 numpy 视图（可选，bfloat16 不支持）

    使用方式：
        buf = CpuGpuBuffer(1024, dtype=torch.int32, device="cuda", pin_memory=True)
        buf.cpu[:10] = some_data  # 在 CPU 上准备数据
        buf.copy_to_gpu(10)       # 异步传输到 GPU
    """

    def __init__(
        self,
        *size: int | torch.SymInt,
        dtype: torch.dtype,
        device: torch.device,
        pin_memory: bool,
        with_numpy: bool = True,
    ) -> None:
        # 这些缓冲区是可变的运行时状态，所以在非推理模式下分配
        with torch.inference_mode(False):
            self.cpu = torch.zeros(
                *size, dtype=dtype, device="cpu", pin_memory=pin_memory
            )
            self.gpu = torch.zeros_like(self.cpu, device=device)
        self.np: np.ndarray
        # 为了保持类型提示简单（避免泛型和子类），仅在需要时创建 numpy 数组属性。
        # 当 with_numpy=False 时访问 self.np 会触发 AttributeError。
        if with_numpy:
            if dtype == torch.bfloat16:
                raise ValueError(
                    "Bfloat16 torch tensors cannot be directly cast to a "
                    "numpy array, so call CpuGpuBuffer with with_numpy=False"
                )
            self.np = self.cpu.numpy()

    def copy_to_gpu(self, n: int | None = None) -> torch.Tensor:
        """
        将 CPU 数据异步拷贝到 GPU。

        Args:
            n: 拷贝的元素数量。None 表示拷贝全部。

        Returns:
            GPU 张量
        """
        if n is None:
            return self.gpu.copy_(self.cpu, non_blocking=True)
        return self.gpu[:n].copy_(self.cpu[:n], non_blocking=True)

    def copy_to_cpu(self, n: int | None = None) -> torch.Tensor:
        """将 GPU 数据异步拷贝到 CPU。

        注意：由于此方法是非阻塞的，需要显式同步以确保数据已拷贝到 CPU。

        Args:
            n: 拷贝的元素数量。None 表示拷贝全部。

        Returns:
            CPU 张量
        """
        if n is None:
            return self.cpu.copy_(self.gpu, non_blocking=True)
        return self.cpu[:n].copy_(self.gpu[:n], non_blocking=True)


def get_engine_client_zmq_addr(
    local_only: bool,
    host: str,
    port: int = 0,
) -> str:
    """返回 ZMQ IPC 路径（local_only=True 时）或 tcp://host:port。

    port=0 时内核会在 bind() 时分配端口；
    调用者需要通过 getsockopt(zmq.LAST_ENDPOINT) 获取实际端口。

    Args:
        local_only: 是否仅本地通信
        host: 主机地址
        port: 端口号（0 表示自动分配）

    Returns:
        ZMQ 地址字符串
    """
    if local_only:
        return get_open_zmq_ipc_path()
    return get_tcp_uri(host, port)


class APIServerProcessManager:
    """管理一组 API 服务器进程。

    负责 API 服务器工作进程的创建、监控和终止。
    同时监控额外的进程以检查它们是否健康。

    工作流程：
    1. 初始化时为每个 API 服务器创建子进程
    2. 每个子进程通过 Pipe 报告绑定的实际 ZMQ 地址
    3. gather_actual_addresses() 收集所有子进程的地址
    4. shutdown() 负责优雅关闭所有子进程
    """

    def __init__(
        self,
        listen_address: str,
        sock: Any,
        args: argparse.Namespace,
        num_servers: int,
        input_addresses: list[str],
        output_addresses: list[str],
        target_server_fn: Callable | None = None,
        stats_update_address: str | None = None,
        tensor_queue: Queue | None = None,
    ):
        """初始化并启动 API 服务器工作进程。

        ``input_addresses``/``output_addresses`` 可能包含 ``tcp://host:0`` 占位符；
        每个子进程通过 ``actual_address_pipe`` 报告实际绑定的端点，
        父进程通过 gather_actual_addresses() 收集。

        Args:
            target_server_fn: 覆盖函数，用于每个 API 服务器进程
            listen_address: 监听客户端连接的地址
            sock: 客户端连接的套接字
            args: 命令行参数
            num_servers: 要启动的 API 服务器进程数
            input_addresses: 每个 API 服务器的输入地址
            output_addresses: 每个 API 服务器的输出地址
            stats_update_address: 可选的统计更新地址
            tensor_queue: 可选的张量 IPC 队列（用于共享多模态张量）
        """
        self.listen_address = listen_address
        self.sock = sock
        self.args = args

        # 使用 spawn 上下文创建子进程（避免 fork 相关问题）
        spawn_context = multiprocessing.get_context("spawn")
        self.processes: list[BaseProcess] = []
        # 地址报告管道列表（每个子进程一个）
        self._address_pipes: list[connection.Connection] = []

        for i, in_addr, out_addr in zip(
            range(num_servers), input_addresses, output_addresses
        ):
            client_config: dict[str, Any] = {
                "input_address": in_addr,
                "output_address": out_addr,
                "client_count": num_servers,
                "client_index": i,
            }
            if stats_update_address is not None:
                client_config["stats_update_address"] = stats_update_address
            if tensor_queue is not None:
                client_config["tensor_queue"] = tensor_queue

            # 创建父子进程之间的单向管道（子进程 -> 父进程）
            parent_recv, child_send = spawn_context.Pipe(duplex=False)
            self._address_pipes.append(parent_recv)
            client_config["actual_address_pipe"] = child_send

            proc = spawn_context.Process(
                target=target_server_fn or run_api_server_worker_proc,
                name=f"ApiServer_{i}",
                args=(listen_address, sock, args, client_config),
            )
            self.processes.append(proc)
            proc.start()

            # 关闭父进程的写端，这样子进程关闭时父进程会看到 EOF
            child_send.close()

        logger.info("Started %d API server processes", len(self.processes))

        # 垃圾回收时仅关闭 API 服务器进程
        # 额外进程由其所有者管理
        self._finalizer = weakref.finalize(self, shutdown, self.processes)

    def gather_actual_addresses(
        self,
        timeout: float = envs.VLLM_ENGINE_READY_TIMEOUT_S,
    ) -> tuple[list[str], list[str]]:
        """收集每个子进程报告的 (inputs, outputs) 地址。

        使用 connection.wait() 高效等待所有子进程报告地址或检测子进程退出。

        Args:
            timeout: 等待超时时间（秒）

        Returns:
            (input_addresses, output_addresses) 列表，按 client_index 索引

        Raises:
            RuntimeError: 超时或子进程提前退出
        """
        n = len(self._address_pipes)
        inputs: list[str | None] = [None] * n
        outputs: list[str | None] = [None] * n
        # 待处理的管道 -> 索引映射
        pending: dict[connection.Connection, int] = {
            pipe: i for i, pipe in enumerate(self._address_pipes)
        }
        # 进程 sentinel -> 索引映射（用于检测进程退出）
        sentinel_to_idx: dict[Any, int] = {
            proc.sentinel: i for i, proc in enumerate(self.processes)
        }

        deadline = time.monotonic() + timeout
        try:
            while pending:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    missing = [self.processes[i].name for i in pending.values()]
                    raise RuntimeError(
                        f"Timed out after {timeout:.1f}s waiting for "
                        f"API server(s) to report bound ZMQ addresses: "
                        f"{missing}"
                    )
                waitables: list[Any] = list(pending.keys()) + list(
                    sentinel_to_idx.keys()
                )
                ready = connection.wait(waitables, timeout=remaining)
                # 先处理管道消息再检查 sentinel：
                # 一个子进程可能同时发送消息并退出，在同一轮 poll 中产生两个事件，
                # 必须先记录成功消息。
                for item in ready:
                    if isinstance(item, connection.Connection) and item in pending:
                        idx = pending.pop(item)
                        try:
                            msg: dict[str, str] = item.recv()
                        except EOFError as e:
                            raise RuntimeError(
                                f"API server {self.processes[idx].name} "
                                f"closed its address pipe without "
                                f"reporting its bound ZMQ addresses"
                            ) from e
                        inputs[idx] = msg["input_address"]
                        outputs[idx] = msg["output_address"]
                        item.close()
                for item in ready:
                    if item in sentinel_to_idx:
                        idx = sentinel_to_idx.pop(item)
                        pipe = self._address_pipes[idx]
                        if pipe in pending:
                            proc = self.processes[idx]
                            raise RuntimeError(
                                f"API server process {proc.name} exited "
                                f"(code={proc.exitcode}) before reporting "
                                f"its bound ZMQ addresses"
                            )
        finally:
            for pipe in pending:
                with contextlib.suppress(Exception):
                    pipe.close()

        return inputs, outputs  # type: ignore[return-value]

    def shutdown(self, timeout: float | None = None) -> None:
        """关闭 API 服务器进程，支持可配置的超时。"""
        for pipe in self._address_pipes:
            with contextlib.suppress(Exception):
                pipe.close()
        self._address_pipes = []

        if self._finalizer.detach() is not None:
            shutdown(self.processes, timeout=timeout)


class RustFrontendProcessManager:
    """管理单个 Rust 前端子进程。

    以 'frontend' 模式启动 Rust vllm-rs 二进制文件，传递监听套接字 fd
    和 ZMQ 传输地址。提供与 APIServerProcessManager 相同的进程监控接口。

    Rust 前端的优势：
    - HTTP 请求解析和序列化在 Rust 中完成，性能更高
    - 减少 Python GIL 对请求处理的影响
    """

    def __init__(
        self,
        binary_path: str,
        sock: Any,
        args: argparse.Namespace,
        input_address: str,
        output_address: str,
        engine_count: int,
        stats_update_address: str | None = None,
    ):
        import os
        import subprocess

        fd = sock.fileno()
        os.set_inheritable(fd, True)

        cmd = [
            binary_path,
            "frontend",
            "--listen-fd",
            str(fd),
            "--input-address",
            input_address,
            "--output-address",
            output_address,
            "--engine-count",
            str(engine_count),
        ]
        if stats_update_address is not None:
            cmd.extend(["--coordinator-address", stats_update_address])
        from vllm.entrypoints.utils import jsonify_non_default_args

        args_json = json.dumps(
            jsonify_non_default_args(args, exclude={"api_server_count"}),
            sort_keys=True,
        )
        cmd.extend(["--args-json", args_json])

        logger.info("Launching Rust frontend: %s", " ".join(cmd))
        self._proc = subprocess.Popen(cmd, pass_fds=(fd,))

        # 创建进程包装器，提供统一的 sentinel fd 用于监控
        self.processes: list[_SubprocessWrapper] = [
            _SubprocessWrapper(self._proc, "RustFrontend")
        ]

        self._finalizer = weakref.finalize(self, _shutdown_subprocesses, self.processes)

    def shutdown(self, timeout: float | None = None) -> None:
        if self._finalizer.detach() is not None:
            _shutdown_subprocesses(self.processes, timeout=timeout)


class _SubprocessWrapper:
    """包装 subprocess.Popen，提供 wait_for_completion_or_failure 所需的
    BaseProcess-like 接口。

    使用基于 Pipe 的 sentinel，使得 subprocess 监控可以与
    multiprocessing.connection.wait() 统一使用。
    """

    def __init__(self, proc, name: str):
        self._proc = proc
        self.name = name
        self.pid = proc.pid
        self._sentinel_conn: connection.Connection | None = None
        self._sentinel_send: connection.Connection | None = None

        # 使用 Pipe 作为 sentinel，使 subprocess 监控在所有平台上
        # 都能与 multiprocessing.connection.wait() 一起工作。
        recv, send = connection.Pipe(duplex=False)
        self._sentinel_conn = recv
        self._sentinel_send = send

        def monitor_subprocess() -> None:
            try:
                proc.wait()
            finally:
                with contextlib.suppress(Exception):
                    send.close()

        threading.Thread(
            target=monitor_subprocess, daemon=True, name=f"{name}Monitor"
        ).start()

    @property
    def sentinel(self):
        return self._sentinel_conn

    @property
    def exitcode(self) -> int | None:
        return self._proc.returncode if self._proc.poll() is not None else None

    def is_alive(self) -> bool:
        return self._proc.poll() is None

    def terminate(self):
        self._proc.terminate()

    def join(self, timeout=None):
        with contextlib.suppress(Exception):
            self._proc.wait(timeout=timeout)

    def __del__(self):
        with contextlib.suppress(Exception):
            if self._sentinel_conn is not None:
                self._sentinel_conn.close()
            if self._sentinel_send is not None:
                self._sentinel_send.close()


def _shutdown_subprocesses(
    procs: list[_SubprocessWrapper], timeout: float | None = None
) -> None:
    """关闭子进程包装器（与 shutdown() 函数功能相同）。"""
    if timeout is None:
        timeout = 0.0
    timeout = max(timeout, 5.0)

    for proc in procs:
        if proc.is_alive():
            proc.terminate()

    deadline = time.monotonic() + timeout
    for proc in procs:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        if proc.is_alive():
            proc.join(remaining)

    for proc in procs:
        if proc.is_alive() and (pid := proc.pid) is not None:
            kill_process_tree(pid)


def run_api_server_worker_proc(
    listen_address, sock, args, client_config=None, **uvicorn_kwargs
) -> None:
    """单个 API 服务器工作进程的入口点。

    设置进程标题、装饰日志输出，然后运行 uvicorn 服务器。
    """

    from vllm.entrypoints.openai.api_server import run_server_worker

    client_config = client_config or {}
    server_index = client_config.get("client_index", 0)

    # 设置进程标题并在 stdout/stderr 添加进程特定前缀
    set_process_title("APIServer", str(server_index))
    decorate_logs()

    uvloop.run(
        run_server_worker(listen_address, sock, args, client_config, **uvicorn_kwargs)
    )


def wait_for_completion_or_failure(
    api_server_manager: "APIServerProcessManager | RustFrontendProcessManager",
    engine_manager: Union["CoreEngineProcManager", "CoreEngineActorManager"]
    | None = None,
    coordinator: "DPCoordinator | None" = None,
) -> None:
    """等待所有进程完成或检测是否有进程失败。

    如果任何进程以非零状态退出，抛出异常。

    监控机制：
    1. 将所有进程的 sentinel（或管道）加入 connection.wait() 等待集合
    2. 当有 sentinel 变为 ready 时，检查对应进程的退出码
    3. 如果引擎管理器报告了失败的进程，也抛出异常

    Args:
        api_server_manager: API 服务器管理器
        engine_manager: 引擎进程管理器（CoreEngineProcManager 或 CoreEngineActorManager）
        coordinator: 数据并行协调器
    """

    try:
        logger.info("Waiting for API servers to complete ...")
        # 创建 sentinel 到进程的映射，用于高效查找
        sentinel_to_proc: dict[Any, BaseProcess | _SubprocessWrapper | None] = {
            proc.sentinel: proc for proc in api_server_manager.processes
        }

        if coordinator:
            sentinel_to_proc[coordinator.proc.sentinel] = coordinator.proc

        if engine_manager:
            core_shutdown_recv, core_shutdown_send = connection.Pipe(duplex=False)

            def monitor_engines():
                try:
                    engine_manager.monitor_engine_liveness()
                finally:
                    core_shutdown_send.close()
                    core_shutdown_recv.close()

            # 启动引擎活跃度监控线程
            threading.Thread(target=monitor_engines, daemon=True).start()
            sentinel_to_proc[core_shutdown_recv] = None  # type: ignore[assignment]

        # 检查是否有进程终止
        while sentinel_to_proc:
            # 等待任何进程终止（或引擎关闭信号）
            ready_sentinels: list[Any] = connection.wait(sentinel_to_proc)

            # 处理已终止的进程
            for sentinel in ready_sentinels:
                proc = sentinel_to_proc.pop(sentinel)

                # 检查进程是否以错误退出
                if proc is not None and proc.exitcode != 0:
                    raise RuntimeError(
                        f"Process {proc.name} (PID: {proc.pid}) "
                        f"died with exit code {proc.exitcode}"
                    )
                if engine_manager and engine_manager.failed_proc_name is not None:
                    raise RuntimeError(
                        f"Engine core process {engine_manager.failed_proc_name} "
                        "died unexpectedly."
                    )

    except KeyboardInterrupt:
        logger.info("Received KeyboardInterrupt, shutting down API servers...")
    except Exception as e:
        logger.exception("Exception occurred while running API servers: %s", str(e))
        raise


# 注意：shutdown 不能是绑定方法，否则 gc 无法回收对象。
def shutdown(procs: list[BaseProcess], timeout: float | None = None) -> None:
    """关闭进程，支持超时。

    三步关闭流程：
    1. 向所有存活进程发送 SIGTERM（优雅关闭）
    2. 等待超时，让进程自行退出
    3. 对仍在运行的进程发送 SIGKILL（强制终止）

    Args:
        procs: 要关闭的进程列表
        timeout: 等待优雅关闭的最大时间（秒）
    """
    if timeout is None:
        # 为没有用户配置超时的最佳努力清理路径保留一个小的宽限期
        timeout = 5.0

    # 向所有存活进程发送终止信号
    for proc in procs:
        if proc.is_alive():
            proc.terminate()

    # 等待进程终止
    deadline = time.monotonic() + timeout
    for proc in procs:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        if proc.is_alive():
            proc.join(remaining)

    # 强制杀死仍在运行的进程
    for proc in procs:
        if proc.is_alive() and (pid := proc.pid) is not None:
            kill_process_tree(pid)


def copy_slice(
    from_tensor: torch.Tensor, to_tensor: torch.Tensor, length: int
) -> torch.Tensor:
    """
    将张量的前 length 个元素非阻塞拷贝到另一个张量。

    用于将锁页 CPU 张量数据拷贝到预分配的 GPU 张量。
    非阻塞传输允许与 GPU 计算重叠。

    Args:
        from_tensor: 源张量
        to_tensor: 目标张量
        length: 要拷贝的元素数

    Returns:
        切片后的目标张量
    """
    return to_tensor[:length].copy_(from_tensor[:length], non_blocking=True)


def report_usage_stats(
    vllm_config, usage_context: UsageContext = UsageContext.ENGINE_CONTEXT
) -> None:
    """上报使用统计信息（如果启用）。

    收集模型架构、配置参数、并行设置等信息并上报。
    """

    if not is_usage_stats_enabled():
        return

    from vllm.model_executor.model_loader import get_architecture_class_name

    parallel_config = vllm_config.parallel_config

    # 准备 KV 连接器字符串（如果适用）
    kv_connector = None
    if vllm_config.kv_transfer_config is not None:
        kv_connector = vllm_config.kv_transfer_config.kv_connector

    usage_message.report_usage(
        get_architecture_class_name(vllm_config.model_config),
        usage_context,
        extra_kvs={
            # 通用配置
            "dtype": str(vllm_config.model_config.dtype),
            "block_size": vllm_config.cache_config.block_size,
            "gpu_memory_utilization": vllm_config.cache_config.gpu_memory_utilization,
            "kv_cache_memory_bytes": vllm_config.cache_config.kv_cache_memory_bytes,
            # 量化配置
            "quantization": vllm_config.model_config.quantization,
            "kv_cache_dtype": str(vllm_config.cache_config.cache_dtype),
            # 功能标志
            "enable_lora": bool(vllm_config.lora_config),
            "enable_prefix_caching": vllm_config.cache_config.enable_prefix_caching,
            "enforce_eager": vllm_config.model_config.enforce_eager,
            "disable_custom_all_reduce": parallel_config.disable_custom_all_reduce,
            # 分布式并行设置
            "tensor_parallel_size": parallel_config.tensor_parallel_size,
            "data_parallel_size": parallel_config.data_parallel_size,
            "pipeline_parallel_size": parallel_config.pipeline_parallel_size,
            "enable_expert_parallel": parallel_config.enable_expert_parallel,
            # MoE 专家并行的 All2All 后端
            "all2all_backend": parallel_config.all2all_backend,
            # 使用的 KV 连接器
            "kv_connector": kv_connector,
        },
    )


_PROFILER_FUNC = None


def record_function_or_nullcontext(name: str) -> AbstractContextManager:
    """
    根据配置返回性能分析上下文管理器或空上下文。

    支持三种模式：
    1. 默认：返回 nullcontext（无开销）
    2. VLLM_CUSTOM_SCOPES_FOR_PROFILING=True：返回 torch.profiler.record_function
    3. VLLM_NVTX_SCOPES_FOR_PROFILING=True：返回 nvtx.annotate（NVTX 标记）

    使用全局缓存避免重复检查环境变量。

    Args:
        name: 性能分析范围的名称

    Returns:
        上下文管理器
    """
    global _PROFILER_FUNC

    # 快速路径：假设已设置
    if _PROFILER_FUNC is not None:
        return _PROFILER_FUNC(name)

    func = contextlib.nullcontext
    if envs.VLLM_CUSTOM_SCOPES_FOR_PROFILING:
        func = record_function
    elif envs.VLLM_NVTX_SCOPES_FOR_PROFILING:
        import nvtx

        func = nvtx.annotate

    _PROFILER_FUNC = func
    return func(name)


def tensor_data(tensor: torch.Tensor) -> memoryview:
    """获取张量的原始数据作为 uint8 memoryview，用于序列化和哈希。

    处理流程：flatten -> CPU -> contiguous -> view(uint8) -> numpy -> memoryview

    Args:
        tensor: 输入张量

    Returns:
        uint8 类型的 memoryview
    """
    return tensor.flatten().cpu().contiguous().view(torch.uint8).numpy().data


@dataclass
class IterationDetails:
    """
    迭代详情数据类，记录一次调度迭代中的请求和 token 统计。

    属性：
        num_ctx_requests: 上下文（prefill）请求数
        num_ctx_tokens: 上下文 token 数
        num_generation_requests: 生成（decode）请求数
        num_generation_tokens: 生成 token 数
    """
    num_ctx_requests: int
    num_ctx_tokens: int
    num_generation_requests: int
    num_generation_tokens: int

    def __repr__(self) -> str:
        return f"IterationDetails(num_ctx_requests={self.num_ctx_requests},\
                 num_ctx_tokens={self.num_ctx_tokens}, \
                 num_generation_requests={self.num_generation_requests}, \
                 num_generation_tokens={self.num_generation_tokens})"


def compute_iteration_details(scheduler_output: SchedulerOutput) -> IterationDetails:
    """
    计算当前迭代调度输出的上下文/生成请求数和 token 数。

    分类逻辑：
    - 上下文请求 (context request): 输出 token 数为 0 的请求，
      包括新请求和 chunked prefill 的扩展 chunk
    - 生成请求 (generation request): 已开始自回归生成的请求

    Args:
        scheduler_output: 当前迭代的调度输出

    Returns:
        IterationDetails 对象，包含上下文/生成请求数和 token 数
    """
    num_context_requests = 0
    num_context_tokens = 0
    num_generation_requests = 0
    num_generation_tokens = 0
    new_req_ids = {new_req.req_id for new_req in scheduler_output.scheduled_new_reqs}
    for req_id, num_tokens in scheduler_output.num_scheduled_tokens.items():
        if scheduler_output.scheduled_cached_reqs.is_context_phase(req_id) or (
            req_id in new_req_ids
        ):
            num_context_requests += 1
            num_context_tokens += num_tokens
        else:
            num_generation_requests += 1
            num_generation_tokens += num_tokens
    return IterationDetails(
        num_context_requests,
        num_context_tokens,
        num_generation_requests,
        num_generation_tokens,
    )
