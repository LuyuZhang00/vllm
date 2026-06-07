# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
引擎工具模块 (Engine Utilities Module)

本模块提供了 vLLM v1 引擎的核心辅助功能，主要包括以下内容：
1. 引擎进程管理：创建、监控和关闭后台引擎进程
2. ZMQ 通信地址管理：分配和管理引擎与客户端之间的通信地址
3. 数据并行 (DP) 支持：通过 Ray 或本地进程管理多个 DP 实例
4. 引擎启动握手协议：协调引擎核心进程的启动和就绪状态
5. 设备控制：管理 CUDA/GPU 设备的分配和隔离
"""

import contextlib
import os
import threading
import weakref
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from enum import Enum, auto
from multiprocessing import Process, connection
from multiprocessing.process import BaseProcess
from multiprocessing.queues import Queue
from typing import TYPE_CHECKING, cast
from unittest.mock import patch

import msgspec
import zmq

from vllm import envs
from vllm.config import CacheConfig, ParallelConfig, VllmConfig
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.ray.ray_env import get_env_vars_to_copy
from vllm.utils import numa_utils
from vllm.utils.network_utils import (
    get_open_port,
    get_open_zmq_ipc_path,
    get_tcp_uri,
    zmq_socket_ctx,
)
from vllm.utils.system_utils import get_mp_context
from vllm.v1.engine.coordinator import DPCoordinator
from vllm.v1.executor import Executor
from vllm.v1.executor.ray_utils import WORKER_SPECIFIC_ENV_VARS
from vllm.v1.utils import get_engine_client_zmq_addr, shutdown

if TYPE_CHECKING:
    from ray.util.placement_group import PlacementGroup

logger = init_logger(__name__)

# 启动轮询周期（毫秒），用于等待引擎核心进程启动完成
STARTUP_POLL_PERIOD_MS = 10000


class CoreEngineState(Enum):
    """
    引擎核心状态枚举

    表示引擎核心进程在握手过程中的状态：
    - NEW: 刚创建，尚未连接
    - CONNECTED: 已与前端建立连接，等待就绪
    - READY: 引擎已完成初始化，可以处理请求
    """
    NEW = auto()
    CONNECTED = auto()
    READY = auto()


class CoreEngine:
    """
    引擎核心跟踪类

    每个数据并行 rank 对应一个实例，用于在握手过程中跟踪状态。
    属性：
    - local: 是否为本地引擎（与前端在同一节点）
    - identity: 引擎的唯一标识（2 字节小端序编码）
    - state: 引擎当前状态（NEW -> CONNECTED -> READY）
    """
    """One per data parallel rank, used to track state during handshaking."""

    def __init__(self, index: int = 0, local: bool = True):
        self.local = local
        self.identity = index.to_bytes(2, "little")

        self.state = CoreEngineState.NEW


@dataclass
class EngineZmqAddresses:
    """
    引擎 ZMQ 地址数据类

    存储引擎与前端客户端之间通信所需的 ZMQ 地址信息：
    - inputs: 每个前端客户端的输入 socket 地址（用于发送请求）
    - outputs: 每个前端客户端的输出 socket 地址（用于接收响应）
    - coordinator_input: DP 协调器的输入 socket 地址（可选）
    - coordinator_output: DP 协调器的输出 socket 地址（可选）
    - frontend_stats_publish_address: 前端统计信息发布地址（可选，用于外部 DP 负载均衡）
    """
    # ZMQ input socket addresses for each front-end client (requests)
    inputs: list[str]
    # ZMQ output socket addresses for each front-end client (responses)
    outputs: list[str]
    # ZMQ input socket address of DP coordinator if applicable
    coordinator_input: str | None = None
    # ZMQ output socket address of DP coordinator if applicable
    coordinator_output: str | None = None
    # ZMQ socket for front-end to connect to DP coordinator.
    # Not used by engine, just relayed to front-end in handshake response.
    # Only required for external DP LB case.
    frontend_stats_publish_address: str | None = None


@dataclass
class EngineHandshakeMetadata:
    """
    引擎握手元数据

    在启动握手期间发送给每个引擎进程的元数据，包含前端 ZMQ 队列的地址信息。
    引擎进程通过这些地址与前端建立通信连接。

    属性：
    - addresses: ZMQ 地址集合
    - parallel_config: 并行配置参数（字典形式，用于 DP 协调）
    """
    """Metadata sent to each engine process during startup handshake,
    including addresses of the front-end ZMQ queues that they should
    connect to.
    """

    addresses: EngineZmqAddresses
    parallel_config: dict[str, int | str | list[int]]


def _make_control_bundle(node_ip: str) -> dict[str, float]:
    """
    创建控制 bundle（用于 Ray placement group）

    引擎 Actor 被调度到最后一个 CPU-only bundle 中。
    该 bundle 与组中的第一个 GPU bundle 共置，以防止 Actor
    漂移到不相关的节点，从而避免 worker rank 顺序与
    DP 引导主机不一致。

    参数：
        node_ip: 节点 IP 地址

    返回：
        包含 CPU 资源和节点亲和性的 bundle 字典
    """
    # The engine actor is scheduled on the final CPU-only bundle. Keep that
    # bundle colocated with the group's first GPU bundle so the actor does not
    # float to an unrelated node and reorder worker ranks away from the
    # advertised DP bootstrap host.
    return {"CPU": 1.0, "node:" + node_ip: 0.001}


def _get_bundle_node_ip(bundle: dict[str, float]) -> str:
    """
    从 bundle 中提取节点 IP 地址

    参数：
        bundle: Ray placement group 的 bundle 字典

    返回：
        节点 IP 地址字符串

    异常：
        ValueError: 如果 bundle 中没有 node: 前缀的键
    """
    for key in bundle:
        if key.startswith("node:"):
            return key.split(":", 1)[1]
    raise ValueError(f"Missing node affinity in placement bundle: {bundle}")


class CoreEngineProcManager:
    """
    引擎核心进程管理器

    用于管理由 AsyncLLM 和 LLMEngine 使用的后台引擎核心进程。
    主要功能包括：
    1. 创建：为每个本地 DP rank 启动一个 EngineCoreProc 进程
    2. 监控：检测进程存活状态，处理异常退出
    3. 关闭：安全地终止所有引擎进程

    属性：
    - processes: 引擎进程列表
    - _finalizer: 弱引用终结器，确保进程在对象销毁时被清理
    - manager_stopped: 线程事件，标记管理器是否已停止
    - failed_proc_name: 失败进程的名称（用于错误报告）
    """

    def __init__(
        self,
        local_engine_count: int,
        start_index: int,
        local_start_index: int,
        vllm_config: VllmConfig,
        local_client: bool,
        handshake_address: str,
        executor_class: type[Executor],
        log_stats: bool,
        client_handshake_address: str | None = None,
        tensor_queue: Queue | None = None,
    ):
        # 获取多进程上下文（支持 fork/spawn）
        context = get_mp_context()

        # 公共参数：所有引擎进程共享的配置
        common_kwargs = {
            "vllm_config": vllm_config,
            "local_client": local_client,
            "handshake_address": handshake_address,
            "executor_class": executor_class,
            "log_stats": log_stats,
            "tensor_queue": tensor_queue,
        }

        # 如果有客户端握手地址，添加到公共参数中
        if client_handshake_address:
            common_kwargs["client_handshake_address"] = client_handshake_address

        # 判断是否为数据并行模式
        is_dp = vllm_config.parallel_config.data_parallel_size > 1

        # 延迟导入，避免循环依赖
        from vllm.v1.engine.core import EngineCoreProc

        self.processes: list[BaseProcess] = []
        local_dp_ranks = []

        # 为每个本地引擎创建进程
        for index in range(local_engine_count):
            local_index = local_start_index + index
            global_index = start_index + index

            # 记录本地 DP rank
            local_dp_ranks.append(local_index)
            # 创建并启动 EngineCore 后台进程
            self.processes.append(
                context.Process(
                    target=EngineCoreProc.run_engine_core,
                    name=f"EngineCore_DP{global_index}" if is_dp else "EngineCore",
                    kwargs=common_kwargs
                    | {"dp_rank": global_index, "local_dp_rank": local_index},
                )
            )

        # 使用弱引用终结器确保进程在对象销毁时被清理
        self._finalizer = weakref.finalize(self, shutdown, self.processes)
        # 管理器停止事件，用于通知监控线程
        self.manager_stopped = threading.Event()
        # 记录失败进程的名称
        self.failed_proc_name: str | None = None

        try:
            # 逐个启动引擎进程
            for proc, local_dp_rank in zip(self.processes, local_dp_ranks):
                # 设备控制上下文：用于在 DP 模式下设置正确的 GPU 设备
                # 适用于无法依赖 torch.accelerator.set_device_index() 的平台
                # 以及 Ray 启动器
                device_control_context: contextlib.AbstractContextManager[None] = (
                    contextlib.nullcontext()
                )
                needs_device_env_isolation = not (
                    current_platform.is_cuda_alike() or current_platform.is_xpu()
                )
                if is_dp and (
                    needs_device_env_isolation or vllm_config.parallel_config.use_ray
                ):
                    device_control_context = set_device_control_env_var(
                        vllm_config, local_dp_rank
                    )

                with (
                    device_control_context,
                    numa_utils.configure_subprocess(
                        # EngineCore 本身没有 TP/PP 本地 rank。
                        # 当 DP 启用时，set_device_control_env_var()
                        # 会先将可见设备缩小到此 DP 分片，
                        # 因此 local_rank=0 表示"此分片中的第一个本地 GPU"。
                        # 实际的 TP/PP worker 进程由执行器单独绑定，
                        # 使用各自的 local_rank 值。
                        vllm_config,
                        local_rank=0,
                        dp_local_rank=local_dp_rank,
                        process_kind="EngineCore",
                    ),
                ):
                    proc.start()
        finally:
            # 如果有任何进程未运行，关闭所有进程
            if self.finished_procs():
                self.shutdown()

    def shutdown(self, timeout: float | None = None) -> None:
        """
        关闭引擎核心进程

        参数：
            timeout: 关闭超时时间（秒），None 表示无限等待
        """
        """Shutdown engine core processes with configurable timeout."""
        self.manager_stopped.set()
        if self._finalizer.detach() is not None:
            shutdown(self.processes, timeout=timeout)

    def monitor_engine_liveness(self) -> None:
        """
        监控引擎核心进程存活状态

        使用 connection.wait() 监听进程 sentinel，当进程退出时：
        - 如果退出码非 0 且管理器未停止，记录失败进程名称
        - 任何引擎退出都会触发整个系统的关闭
        - 未来的工作（如弹性 EP 和容错）将添加更细粒度的处理
        """
        """Monitor engine core process liveness."""

        # 建立 sentinel 到进程的映射
        sentinel_to_proc = {proc.sentinel: proc for proc in self.processes}
        sentinels = set(sentinel_to_proc.keys())

        # 持续监控，直到管理器停止或所有 sentinel 都已处理
        while sentinels and not self.manager_stopped.is_set():
            # 等待最多 1 秒，检查是否有进程退出
            died_sentinels = connection.wait(sentinels, timeout=1)

            for sentinel in died_sentinels:
                proc = sentinel_to_proc.pop(cast(int, sentinel))
                exitcode = proc.exitcode
                # 如果进程异常退出且管理器未停止，记录失败信息
                if exitcode != 0 and not self.manager_stopped.is_set():
                    self.failed_proc_name = proc.name
            if died_sentinels:
                # Any engine exit currently triggers a shutdown. Future
                # work (e.g., Elastic and fault-tolerant EP) will add finer-grained
                # handling for different exit scenarios.
                break

        # 触发系统关闭
        self.shutdown()

    def sentinels(self) -> list:
        """返回所有进程的 sentinel 列表，用于监控进程状态"""
        return [proc.sentinel for proc in self.processes]

    def finished_procs(self) -> dict[str, int]:
        """返回已完成进程的字典，键为进程名，值为退出码"""
        """Returns dict of proc name -> exit code for any finished procs."""
        return {
            proc.name: proc.exitcode
            for proc in self.processes
            if proc.exitcode is not None
        }


class SignalCallback:
    """
    信号回调类

    用于从信号处理上下文中安全地触发回调。
    由于信号处理函数中只能执行有限的操作，
    本类通过专用线程来执行实际的回调逻辑。

    工作原理：
    1. 创建时启动一个守护线程，该线程等待事件触发
    2. 当调用 trigger() 时，设置事件，唤醒等待的线程
    3. 线程被唤醒后执行回调函数
    4. 调用 stop() 可以取消回调执行

    属性：
    - _callback: 要执行的回调函数
    - _event: 线程事件，用于同步触发
    - _stopped: 标记是否已停止
    - _thread: 守护线程
    """

    def __init__(self, callback: Callable[[], None]):
        self._callback = callback
        self._event = threading.Event()
        self._stopped = False
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="signal-callback",
        )
        self._thread.start()

    def _run(self):
        """线程运行函数：等待事件触发，然后执行回调"""
        self._event.wait()
        if not self._stopped:
            self._callback()

    def trigger(self):
        """触发回调执行"""
        self._event.set()

    def stop(self):
        """停止回调执行"""
        self._stopped = True
        self._event.set()


@contextlib.contextmanager
def set_device_control_env_var(
    vllm_config: VllmConfig, local_dp_rank: int
) -> Iterator[None]:
    """
    临时设置设备控制环境变量的上下文管理器

    用于为引擎子进程设置 CUDA_VISIBLE_DEVICES 或等效的环境变量。
    在数据并行模式下，每个 DP rank 需要看到不同的 GPU 设备子集。

    参数：
        vllm_config: vLLM 配置对象
        local_dp_rank: 本地数据并行 rank

    使用方式：
        with set_device_control_env_var(config, rank):
            # 在此上下文中，环境变量已设置
            pass
        # 退出上下文后，环境变量恢复原值
    """
    """
    Temporarily set CUDA_VISIBLE_DEVICES or equivalent
    for engine subprocess.
    """
    world_size = vllm_config.parallel_config.world_size
    local_world_size = vllm_config.parallel_config.local_world_size
    evar = current_platform.device_control_env_var

    value = get_device_indices(evar, local_dp_rank, world_size, local_world_size)
    with patch.dict(os.environ, values=((evar, value),)):
        yield


def get_device_indices(
    device_control_env_var: str,
    local_dp_rank: int,
    world_size: int,
    local_world_size: int | None = None,
) -> str:
    """
    获取指定数据并行 rank 的设备索引字符串

    根据 DP rank 和 world_size 计算该 rank 应使用的 GPU 设备索引。
    返回逗号分隔的设备索引字符串。

    示例：
        - world_size=2, local_dp_rank=1, 4 个设备
          -> 返回 "2,3"（选择设备 2 和 3）

    参数：
        device_control_env_var: 设备控制环境变量名（如 CUDA_VISIBLE_DEVICES）
        local_dp_rank: 本地数据并行 rank
        world_size: 模型并行 world_size（TP * PP）
        local_world_size: 本地 world_size（可选，默认等于 world_size）

    返回：
        逗号分隔的设备索引字符串

    异常：
        Exception: 如果设备索引超出范围
    """
    """
    Returns a comma-separated string of device indices for the specified
    data parallel rank.

    For example, if world_size=2 and local_dp_rank=1, and there are 4 devices,
    this will select devices 2 and 3 for local_dp_rank=1.
    """
    if local_world_size is None:
        local_world_size = world_size
    try:
        value = ",".join(
            str(current_platform.device_id_to_physical_device_id(i))
            for i in range(
                local_dp_rank * world_size,
                local_dp_rank * world_size + local_world_size,
            )
        )
    except IndexError as e:
        raise Exception(
            f"Error setting {device_control_env_var}: "
            f"local range: [{local_dp_rank * world_size}, "
            f"{(local_dp_rank + 1) * world_size}) "
            "base value: "
            f'"{os.getenv(device_control_env_var)}"'
        ) from e
    return value


def _apply_dp_identity_suffix(dp_vllm_config, dp_rank: int) -> None:
    """
    为数据并行引擎添加身份后缀

    Ray Actor 名称和 KV-connector engine_id 在兄弟 DP 引擎之间必须唯一，
    否则会导致注册冲突。使用全局 DP rank（而非节点本地 rank），
    因为兄弟 DP 引擎可能跨越多个节点。

    参数：
        dp_vllm_config: vLLM 配置对象（会被修改）
        dp_rank: 全局数据并行 rank
    """
    # Ray actor names (RayExecutorV2) and KV-connector engine_ids must
    # be unique across sibling DP engines or registration collides.
    # Use the global DP rank, not a node-local rank, since sibling DP
    # engines can span multiple nodes.
    dp_vllm_config.instance_id = f"{dp_vllm_config.instance_id}_dp{dp_rank}"
    if dp_vllm_config.kv_transfer_config is not None:
        dp_vllm_config.kv_transfer_config.engine_id = (
            f"{dp_vllm_config.kv_transfer_config.engine_id}_dp{dp_rank}"
        )


class CoreEngineActorManager:
    """
    引擎核心 Ray Actor 管理器

    用于管理由 AsyncLLM 和 LLMEngine 使用的 Ray Actor 形式的核心引擎。
    与 CoreEngineProcManager 不同，本类管理本地和远程节点上的核心引擎。

    主要功能：
    1. 创建：为每个 DP rank 创建 Ray Actor（本地或远程）
    2. Placement Group 管理：创建和管理 Ray placement group
    3. 弹性扩缩容：支持动态增加/减少 DP 实例数
    4. 监控：检测 Actor 存活状态，处理异常退出
    5. 关闭：安全地终止所有 Actor 和 placement group

    属性：
    - local_engine_actors: 本地引擎 Actor 列表
    - remote_engine_actors: 远程引擎 Actor 列表
    - created_placement_groups: 创建的 placement group 列表
    - placement_group_is_local: 每个 placement group 是否为本地的标志列表
    - run_refs: Actor 运行引用列表
    - actor_run_ref_dict: Actor 到运行引用的映射字典
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        addresses: EngineZmqAddresses,
        executor_class: type[Executor],
        log_stats: bool,
        placement_groups: list["PlacementGroup"] | None = None,
        local_dp_ranks: list[int] | None = None,
    ):
        import copy

        import ray
        from ray.runtime_env import RuntimeEnv
        from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

        from vllm.v1.engine.core import DPMoEEngineCoreActor, EngineCoreActor

        dp_size = vllm_config.parallel_config.data_parallel_size

        # 根据是否为 MoE 模型选择 Actor 类
        actor_class = (
            DPMoEEngineCoreActor
            if dp_size > 1 and vllm_config.model_config.is_moe
            else EngineCoreActor
        )

        self.local_engine_actors: list[ray.ActorHandle] = []
        self.remote_engine_actors: list[ray.ActorHandle] = []

        # 获取需要传递给 Ray Actor 的环境变量
        env_vars_list = get_env_vars_to_copy(
            destination=actor_class.__name__,
            exclude_vars=WORKER_SPECIFIC_ENV_VARS,
        )
        self.env_vars_dict = {
            name: os.environ[name] for name in env_vars_list if name in os.environ
        }
        runtime_env = RuntimeEnv(env_vars=self.env_vars_dict)

        self.addresses = addresses
        self.executor_class = executor_class
        self.log_stats = log_stats
        local_engine_count = vllm_config.parallel_config.data_parallel_size_local
        world_size = vllm_config.parallel_config.world_size
        self.manager_stopped = threading.Event()
        self.failed_proc_name: str | None = None

        # 初始化 Ray（如果尚未初始化）
        if ray.is_initialized():
            logger.info("Ray is already initialized. Skipping Ray initialization.")
        else:
            ray.init()

        parallel_config = vllm_config.parallel_config
        # 如果启用了弹性 EP，创建 TCP 存储用于协调
        if parallel_config.enable_elastic_ep:
            from vllm.distributed.utils import create_tcp_store

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

        # 处理 placement group：使用传入的或创建新的
        if placement_groups is not None:
            assert local_dp_ranks is not None, (
                "local_dp_ranks must be provided if placement_groups is provided"
            )
            assert len(placement_groups) == len(local_dp_ranks), (
                "placement_groups and local_dp_ranks must have the same length"
            )
            logger.info("Using provided placement groups")
            # TODO(rui): validate passed-in placement groups
            self.created_placement_groups = []
        else:
            # 创建新的 placement group
            placement_groups, local_dp_ranks = (
                CoreEngineActorManager.create_dp_placement_groups(vllm_config)
            )
            self.created_placement_groups = placement_groups
        assert len(placement_groups) == dp_size, (
            "Number of placement groups must match data parallel size"
        )

        # 为每个 DP rank 创建 Ray Actor
        self.placement_group_is_local = []
        refs = []
        for index, local_index, pg in zip(
            range(dp_size), local_dp_ranks, placement_groups
        ):
            # 深拷贝配置，避免修改原始配置
            dp_vllm_config = copy.deepcopy(vllm_config)
            if dp_size > 1:
                _apply_dp_identity_suffix(dp_vllm_config, index)
            dp_vllm_config.parallel_config.placement_group = pg
            # 判断是否为本地客户端（在 DP 主节点上）
            local_client = index < local_engine_count

            # Ray XPU 已知问题：dpctl 会提前初始化 GPU 运行时，
            # 因此在 Ray Actor 初始化方法中设置设备环境变量不会影响设备选择。
            if current_platform.is_xpu():
                device_evar = current_platform.device_control_env_var
                device_indices = get_device_indices(
                    device_evar, local_index, world_size
                )
                actor_env_vars = self.env_vars_dict.copy()
                actor_env_vars[device_evar] = device_indices
                runtime_env = RuntimeEnv(env_vars=actor_env_vars)

            # 创建 Ray Actor
            actor = (
                ray.remote(actor_class)
                .options(
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=pg,
                        placement_group_bundle_index=world_size,
                    ),
                    runtime_env=runtime_env,
                )
                .remote(
                    vllm_config=dp_vllm_config,
                    executor_class=executor_class,
                    log_stats=log_stats,
                    local_client=local_client,
                    addresses=addresses,
                    dp_rank=index,
                    local_dp_rank=local_index,
                )
            )
            # 分类存储本地和远程 Actor
            if local_client:
                self.local_engine_actors.append(actor)
            else:
                self.remote_engine_actors.append(actor)
            self.placement_group_is_local.append(local_client)
            # 等待 Actor 初始化完成
            refs.append(actor.wait_for_init.remote())

        # 阻塞等待所有 Actor 初始化完成
        ray.get(refs)

        # 启动所有 Actor 的运行循环
        self.run_refs = []
        self.actor_run_ref_dict = dict()
        for actor in self.local_engine_actors + self.remote_engine_actors:
            ref = actor.run.remote()
            self.run_refs.append(ref)
            self.actor_run_ref_dict[actor] = ref

    @staticmethod
    def create_dp_placement_groups(
        vllm_config: VllmConfig,
    ) -> tuple[list["PlacementGroup"], list[int]]:
        """
        为数据并行创建 placement group

        根据 VLLM_RAY_DP_PACK_STRATEGY 环境变量选择不同的打包策略：
        - strict: STRICT_PACK 策略，每个 DP rank 的所有 GPU 必须在同一节点
        - fill: STRICT_PACK 策略，但会尽可能填满每个节点
        - span: PACK 策略，允许跨节点（适用于大模型多节点场景）

        参数：
            vllm_config: vLLM 配置对象

        返回：
            (placement_groups, local_dp_ranks) 元组
            - placement_groups: placement group 列表
            - local_dp_ranks: 每个 placement group 的本地 DP rank 列表

        异常：
            ValueError: 如果资源不足或配置不兼容
        """
        """
        Create placement groups for data parallel.
        """

        import ray
        from ray._private.state import available_resources_per_node

        logger.info("Creating placement groups for data parallel")
        dp_master_ip = vllm_config.parallel_config.data_parallel_master_ip
        dp_size = vllm_config.parallel_config.data_parallel_size
        dp_size_local = vllm_config.parallel_config.data_parallel_size_local

        # 获取每个节点的可用资源
        available_resources = available_resources_per_node()
        world_size = vllm_config.parallel_config.world_size
        placement_groups: list[PlacementGroup] = []
        local_dp_ranks: list[int] = []

        # 按节点排序，确保 DP 主节点排在第一位
        dp_master_ip_key = f"node:{dp_master_ip}"
        nodes = sorted(
            available_resources.values(), key=lambda x: dp_master_ip_key not in x
        )
        assert len(nodes) > 0, "No nodes with resources found in Ray cluster."
        assert dp_master_ip_key in nodes[0], (
            f"The DP master node (ip: {dp_master_ip}) is missing or dead"
        )

        # 获取设备类型标识符
        device_str = current_platform.ray_device_key
        n_node_devices: list[int] = [
            int(node_resources[device_str])
            for node_resources in nodes
            if device_str in node_resources
        ]
        assert n_node_devices, f"No {device_str} found in Ray cluster."
        max_device_per_node = max(n_node_devices)

        # 验证打包策略
        pack_strategy = envs.VLLM_RAY_DP_PACK_STRATEGY
        _supported_pack_strategies = ("strict", "fill", "span")
        if pack_strategy not in _supported_pack_strategies:
            raise ValueError(
                f"{envs.VLLM_RAY_DP_PACK_STRATEGY} is not supported. "
                "Make sure to set `VLLM_RAY_DP_PACK_STRATEGY` "
                f"to one of {_supported_pack_strategies}"
            )

        # DeepEP 内核要求 EP ranks [0,7]（同理 [8,15]...）在同一节点
        # 但 fill 策略无法保证这一点
        all2all_backend = vllm_config.parallel_config.all2all_backend
        if pack_strategy == "fill" and (
            all2all_backend == "deepep_high_throughput"
            or all2all_backend == "deepep_low_latency"
        ):
            raise ValueError(
                "DeepEP kernels require EP ranks [0,7] (same for [8,15], ...) "
                "to be on the same node, but VLLM_RAY_DP_PACK_STRATEGY=fill "
                "does not guarantee that. "
                "Please use VLLM_RAY_DP_PACK_STRATEGY=strict instead."
            )

        # 根据策略选择 Ray placement 策略
        if pack_strategy in ("strict", "fill"):
            placement_strategy = "STRICT_PACK"
        else:
            # span 策略使用 PACK，允许跨节点
            placement_strategy = "PACK"
            assert world_size > max_device_per_node, (
                f"World size {world_size} is smaller than the "
                "maximum number of devices per node "
                f"{max_device_per_node}. Make sure to set "
                "`VLLM_RAY_DP_PACK_STRATEGY` to `strict` or `fill`"
            )

            # 多节点 DP 组要求节点资源同构
            assert set(n_node_devices) == {max_device_per_node}, (
                f"Nodes are not homogeneous, {nodes}"
            )
            assert world_size % max_device_per_node == 0, (
                f"For multi-node data parallel groups, world_size ({world_size}) must "
                f"be a multiple of number of devices per node ({max_device_per_node})."
            )
            assert len(n_node_devices) * max_device_per_node >= world_size * dp_size, (
                f"Not enough total available nodes ({len(n_node_devices)}) "
                f"and devices per node ({max_device_per_node}) "
                f"to satisfy required world size {world_size} and data parallel size "
                f"{dp_size}"
            )
            assert dp_size_local == 1, (
                f"data-parallel-size-local {dp_size_local} should be set as the "
                "default (1) for VLLM_RAY_DP_PACK_STRATEGY=span. "
                "The actual data-parallel-size-local will be auto determined."
            )

        # 用于 span 策略的 bundle 收集器
        # bundles collected for a single DP rank from multiple nodes,
        # for "span" pack strategy
        collected_bundles = []
        for node_resources in nodes:
            # 提取节点 IP 键
            node_ip_keys = [
                key
                for key in node_resources
                if key != "node:__internal_head__"
                and key.startswith("node:")
                and "_group_" not in key
            ]
            assert len(node_ip_keys) == 1, (
                f"Zero or multiple node IP keys found in node resources: {node_ip_keys}"
            )
            node_ip_key = node_ip_keys[0]
            node_ip = node_ip_key.split(":")[1]

            # 计算此节点上可用的设备数和 DP rank 数
            n_device_on_node = int(node_resources.get(device_str, 0))
            if pack_strategy == "span" and n_device_on_node != 0:
                # 严格来说，dp_size_available = n_device_on_node / world_size
                # 是一个分数，但这里使用 1 以便处理
                dp_size_available = 1
            else:
                dp_size_available = n_device_on_node // world_size

            # 确定此节点上要分配的 DP rank 数量
            if node_ip == dp_master_ip:
                if dp_size_available < dp_size_local:
                    raise ValueError(
                        f"Not enough resources to allocate {dp_size_local} DP ranks "
                        f"on DP master node {dp_master_ip}, possible to fit "
                        f"{dp_size_available} DP ranks."
                    )
                dp_size_to_allocate = dp_size_local
            elif pack_strategy == "strict":
                if dp_size_available < dp_size_local:
                    logger.info(
                        "Skipping node %s as %s DP ranks could not fit, "
                        "possible to fit %s DP ranks",
                        node_ip,
                        dp_size_local,
                        dp_size_available,
                    )
                    continue
                dp_size_to_allocate = dp_size_local
            else:
                # fill 和 span 策略：使用所有可用资源
                dp_size_to_allocate = dp_size_available

            # 为此节点上的 DP rank 创建 placement group
            for i in range(dp_size_to_allocate):
                device_bundle = [{device_str: 1.0, "node:" + node_ip: 0.001}]
                if pack_strategy == "span":
                    collected_bundles += device_bundle * n_device_on_node
                    assert len(collected_bundles) <= world_size, (
                        "collected_bundles should be <= world_size, "
                        f"but got {len(collected_bundles)=} and {world_size=}"
                    )

                    # 只有收集到足够的设备时才创建 placement group
                    if len(collected_bundles) < world_size:
                        continue

                    # 添加控制 bundle（用于 Ray Actor 调度）
                    control_node_ip = _get_bundle_node_ip(collected_bundles[0])
                    bundles = collected_bundles + [
                        _make_control_bundle(control_node_ip)
                    ]
                    collected_bundles = []
                else:
                    # STRICT_PACK 已经确保所有 bundle 在同一节点，
                    # 所以控制 bundle 的节点亲和性是冗余的。
                    # 但为了与 span 路径保持一致，仍然添加。
                    bundles = device_bundle * world_size + [
                        _make_control_bundle(node_ip)
                    ]

                # 创建 placement group
                pg = ray.util.placement_group(
                    name=f"dp_rank_{len(placement_groups)}",
                    strategy=placement_strategy,
                    bundles=bundles,
                )
                placement_groups.append(pg)
                local_dp_ranks.append(i)
                if len(placement_groups) == dp_size:
                    break

            if len(placement_groups) == dp_size:
                break

        # 验证是否创建了足够的 placement group
        if len(placement_groups) < dp_size:
            raise ValueError(
                f"Not enough resources to allocate {dp_size} "
                "placement groups, only created "
                f"{len(placement_groups)} placement groups. "
                "Available resources: "
                f"{available_resources}"
            )
        assert len(placement_groups) == dp_size, (
            f"Created {len(placement_groups)} DP placement groups, expected {dp_size}"
        )
        assert len(local_dp_ranks) == dp_size, (
            f"local_dp_ranks length {len(local_dp_ranks)} does not match "
            f"expected {dp_size}"
        )
        return placement_groups, local_dp_ranks

    @staticmethod
    def add_dp_placement_groups(
        old_vllm_config: VllmConfig, new_data_parallel_size: int
    ) -> tuple[list["PlacementGroup"], list[int]]:
        """
        为新的数据并行大小添加 placement group

        用于弹性扩缩容场景，计算需要新增的 placement group 数量，
        并在有足够资源的节点上创建它们。

        参数：
            old_vllm_config: 旧的 vLLM 配置对象
            new_data_parallel_size: 新的数据并行大小

        返回：
            (placement_groups, local_dp_ranks) 元组
        """
        """
        Add placement groups for new data parallel size.
        """
        import ray
        from ray._private.state import (
            available_resources_per_node,
            total_resources_per_node,
        )
        from ray.util.state import list_nodes

        old_dp_size = old_vllm_config.parallel_config.data_parallel_size
        num_pg_to_create = new_data_parallel_size - old_dp_size

        # 如果不需要创建新的 placement group，直接返回
        if num_pg_to_create <= 0:
            return [], []

        dp_master_ip = old_vllm_config.parallel_config.data_parallel_master_ip
        world_size = old_vllm_config.parallel_config.world_size

        # 获取节点列表并按 DP 主节点排序
        nodes = list_nodes()
        nodes = sorted(nodes, key=lambda node: node.node_ip != dp_master_ip)
        assert nodes[0].node_ip == dp_master_ip, "The first node must be the head node"
        assert len(nodes) == 1 or nodes[1].node_ip != dp_master_ip, (
            "There can only be one head node"
        )

        available_resources = available_resources_per_node()
        total_resources = total_resources_per_node()

        placement_groups = []
        local_dp_ranks = []
        num_pg_created = 0

        device_str = current_platform.ray_device_key
        for node in nodes:
            if num_pg_created >= num_pg_to_create:
                break

            node_ip = node.node_ip
            node_id = node.node_id
            if device_str not in available_resources[node_id]:
                continue
            available_gpus = int(available_resources[node_id][device_str])

            # 计算此节点上已使用的 GPU 和引擎数
            total_gpus = int(total_resources[node_id][device_str])
            used_gpus = max(0, total_gpus - available_gpus)
            used_engines_on_node = used_gpus // world_size

            # 计算此节点可以容纳的新引擎数
            available_engine_count = available_gpus // world_size

            # 为此节点上的新引擎创建 placement group
            for i in range(available_engine_count):
                if num_pg_created >= num_pg_to_create:
                    break

                rank = old_dp_size + num_pg_created

                # 为主节点创建带节点约束的 bundle
                if node_ip == dp_master_ip:
                    bundles = [
                        {device_str: 1.0, "node:" + dp_master_ip: 0.001}
                    ] * world_size + [{"CPU": 1.0}]
                else:
                    bundles = [{device_str: 1.0}] * world_size + [{"CPU": 1.0}]

                pg = ray.util.placement_group(
                    name=f"dp_rank_{rank}",
                    strategy="STRICT_PACK",
                    bundles=bundles,
                )
                placement_groups.append(pg)

                # 本地 rank 从该节点上已使用的引擎数开始
                local_rank = used_engines_on_node + i
                local_dp_ranks.append(local_rank)
                num_pg_created += 1

        return placement_groups, local_dp_ranks

    def scale_up_elastic_ep(
        self, cur_vllm_config: VllmConfig, new_data_parallel_size: int
    ) -> None:
        """
        弹性 EP 扩容

        动态增加数据并行实例数量。流程：
        1. 计算需要新增的 placement group 数量
        2. 创建新的 placement group 和 Actor
        3. 等待新 Actor 初始化完成
        4. 启动新 Actor 的运行循环
        5. 更新配置中的数据并行大小

        参数：
            cur_vllm_config: 当前的 vLLM 配置对象（会被修改）
            new_data_parallel_size: 新的数据并行大小
        """
        import copy

        import ray
        from ray.runtime_env import RuntimeEnv
        from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

        from vllm.v1.engine.core import DPMoEEngineCoreActor, EngineCoreActor

        # 根据是否为 MoE 模型选择 Actor 类
        actor_class = (
            DPMoEEngineCoreActor
            if cur_vllm_config.model_config.is_moe
            else EngineCoreActor
        )

        cur_data_parallel_size = len(self.local_engine_actors) + len(
            self.remote_engine_actors
        )

        assert new_data_parallel_size > cur_data_parallel_size, (
            f"New data parallel size {new_data_parallel_size} must be greater "
            f"than current data parallel size {cur_data_parallel_size} "
            "for scale up"
        )

        # 创建新的 placement group
        placement_groups, local_dp_ranks = self.add_dp_placement_groups(
            cur_vllm_config, new_data_parallel_size
        )

        world_size = cur_vllm_config.parallel_config.world_size
        dp_master_ip = cur_vllm_config.parallel_config.data_parallel_master_ip
        new_local_engines = 0

        # 设置运行时环境变量，标记为扩容启动
        runtime_env = RuntimeEnv(
            env_vars=self.env_vars_dict | {"VLLM_ELASTIC_EP_SCALE_UP_LAUNCH": "1"}
        )

        # 为每个新的 placement group 创建 Actor
        for i, (pg, local_rank) in enumerate(zip(placement_groups, local_dp_ranks)):
            rank = cur_data_parallel_size + i
            dp_vllm_config = copy.deepcopy(cur_vllm_config)
            if new_data_parallel_size > 1:
                _apply_dp_identity_suffix(dp_vllm_config, rank)
            dp_vllm_config.parallel_config.data_parallel_size = new_data_parallel_size
            dp_vllm_config.parallel_config.placement_group = pg

            # 检查此 placement group 是否在主节点上
            local_client = any(
                bundle.get("node:" + dp_master_ip, 0) > 0 for bundle in pg.bundle_specs
            )

            if local_client:
                new_local_engines += 1
                # 更新本地数据并行大小
                dp_vllm_config.parallel_config.data_parallel_size_local = (
                    cur_vllm_config.parallel_config.data_parallel_size_local
                    + new_local_engines
                )

            # 创建 Ray Actor
            actor = (
                ray.remote(actor_class)
                .options(
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=pg,
                        placement_group_bundle_index=world_size,
                    ),
                    runtime_env=runtime_env,
                )
                .remote(
                    vllm_config=dp_vllm_config,
                    executor_class=self.executor_class,
                    log_stats=self.log_stats,
                    local_client=local_client,
                    addresses=self.addresses,
                    dp_rank=rank,
                    local_dp_rank=local_rank,
                )
            )

            # 分类存储本地和远程 Actor
            if local_client:
                self.local_engine_actors.append(actor)
            else:
                self.remote_engine_actors.append(actor)
            self.created_placement_groups.append(pg)
            self.placement_group_is_local.append(local_client)

        # 等待所有新 Actor 初始化完成
        ray.get(
            [
                actor.wait_for_init.remote()
                for actor in (
                    self.local_engine_actors[-new_local_engines:]
                    if new_local_engines > 0
                    else []
                )
                + self.remote_engine_actors[
                    -(len(placement_groups) - new_local_engines) :
                ]
            ]
        )

        # 启动所有新 Actor 的运行循环
        actors = (
            self.local_engine_actors[-new_local_engines:]
            if new_local_engines > 0
            else []
        ) + self.remote_engine_actors[-(len(placement_groups) - new_local_engines) :]

        for actor in actors:
            ref = actor.run.remote()
            self.run_refs.append(ref)
            self.actor_run_ref_dict[actor] = ref

        # 更新配置中的数据并行大小
        cur_vllm_config.parallel_config.data_parallel_size = new_data_parallel_size
        # 如果添加了新的本地引擎，更新本地数据并行大小
        if new_local_engines > 0:
            cur_vllm_config.parallel_config.data_parallel_size_local += (
                new_local_engines
            )

    def scale_down_elastic_ep(
        self, cur_data_parallel_size: int, new_data_parallel_size: int
    ) -> None:
        """
        弹性 EP 缩容

        动态减少数据并行实例数量。流程：
        1. 计算需要移除的 placement group 数量
        2. 从列表末尾移除 Actor 和 placement group
        3. 删除 Ray placement group 释放资源

        参数：
            cur_data_parallel_size: 当前的数据并行大小
            new_data_parallel_size: 新的数据并行大小
        """
        import ray

        assert cur_data_parallel_size > new_data_parallel_size, (
            f"cur_data_parallel_size {cur_data_parallel_size} must be greater "
            f"than new_data_parallel_size {new_data_parallel_size} "
            "for scale down"
        )
        for _ in range(cur_data_parallel_size - new_data_parallel_size):
            pg = self.created_placement_groups.pop()
            is_local = self.placement_group_is_local.pop()
            if is_local:
                self.local_engine_actors.pop()
            else:
                self.remote_engine_actors.pop()
            ray.util.remove_placement_group(pg)

    def remove_run_refs_for_scale_down(self, removed_dp_size: int) -> None:
        """
        移除缩容相关的运行引用

        从运行引用列表中移除被缩容的 Actor 引用。

        参数：
            removed_dp_size: 要移除的 DP 实例数量
        """
        if removed_dp_size <= 0:
            return
        # 获取被移除的 placement group 是否为本地的标志
        flags = self.placement_group_is_local[-removed_dp_size:]
        li = len(self.local_engine_actors) - 1
        ri = len(self.remote_engine_actors) - 1
        for is_local in reversed(flags):
            if is_local:
                actor = self.local_engine_actors[li]
                li -= 1
            else:
                actor = self.remote_engine_actors[ri]
                ri -= 1
            ref = self.actor_run_ref_dict.pop(actor)
            self.run_refs.remove(ref)

    def get_run_refs(self):
        """返回所有 Actor 的运行引用列表"""
        return self.run_refs

    def monitor_engine_liveness(self) -> None:
        """
        监控引擎 Actor 存活状态

        使用 ray.wait() 检测 Actor 是否完成运行。
        如果发现意外失败（非正常退出），记录失败信息并触发关闭。
        """
        import ray

        while not self.manager_stopped.is_set():
            actor_run_refs = list(self.get_run_refs())
            if not actor_run_refs:
                logger.info(
                    "There are no actors to monitor currently. "
                    "The monitoring function is about to terminate."
                )
                break
            # 等待最多 5 秒，检查是否有 Actor 完成
            actor_done_refs, _ = ray.wait(actor_run_refs, timeout=5)
            unexpected_failure = False
            for actor_ref in actor_done_refs:
                if self.manager_stopped.is_set():
                    break
                # 检查是否为弹性缩容导致的引用更新
                if actor_ref not in self.get_run_refs():
                    # The run refs may have been updated by elastic scale-down.
                    continue
                try:
                    ray.get(actor_ref)
                except ray.exceptions.RayActorError:
                    self.failed_proc_name = f"Actor {actor_ref}"
                    unexpected_failure = True

            if unexpected_failure:
                break

        # 触发系统关闭
        self.shutdown()

    def shutdown(self, timeout: float | None = None) -> None:
        """
        关闭所有引擎 Actor 和 placement group

        参数：
            timeout: 关闭超时时间（秒），当前未使用
        """
        import ray

        self.manager_stopped.set()
        # 终止所有 Actor
        for actor in self.local_engine_actors + self.remote_engine_actors:
            ray.kill(actor)
        # 删除所有创建的 placement group
        for pg in self.created_placement_groups:
            ray.util.remove_placement_group(pg)


def get_engine_zmq_addresses(
    vllm_config: VllmConfig,
    num_api_servers: int = 1,
    *,
    defer_api_server_ports: bool = True,
) -> EngineZmqAddresses:
    """
    分配引擎-客户端通信的 ZMQ 地址

    根据通信模式选择不同的地址类型：
    - 本地模式：使用 IPC 路径（Unix domain socket），性能更好
    - 远程模式：使用 TCP 地址，支持跨节点通信

    参数：
        vllm_config: vLLM 配置对象
        num_api_servers: API 服务器数量（默认 1）
        defer_api_server_ports: 是否延迟端口分配（默认 True）
            - True: 使用 tcp://host:0 占位符，绑定时由内核分配端口
            - False: 立即分配端口（用于 Rust 前端等无法报告绑定端口的场景）

    返回：
        EngineZmqAddresses 对象，包含输入和输出地址列表
    """
    """Allocate ZMQ addresses for engine-client communication.

    By default each TCP address is a ``tcp://host:0`` placeholder; the
    consumer (API-server child or single-process ``MPClient``) binds, then
    recovers the kernel-assigned port via ``getsockopt(zmq.LAST_ENDPOINT)``
    and writes it back into ``addresses`` before the engine handshake.

    Set ``defer_api_server_ports=False`` only when the consumer cannot
    report a bound port back (e.g. the Rust front-end). IPC paths are
    unaffected."""
    parallel_config = vllm_config.parallel_config
    local_engine_count = parallel_config.data_parallel_size_local
    local_start_index = parallel_config.data_parallel_rank_local
    dp_size = parallel_config.data_parallel_size
    host = parallel_config.data_parallel_master_ip
    local_engines_only = parallel_config.local_engines_only

    # 在离线模式下，每个 DP rank 有一个 LLM 实例，
    # 每个 LLM 有一个核心引擎。
    # 参见 examples/features/data_parallel/data_parallel_offline.py
    offline_mode = local_start_index is not None

    # client_local_only = True 表示此前端只向本地引擎发送请求
    client_local_only = (
        offline_mode or local_engines_only or (local_engine_count == dp_size)
    )
    # 处理从节点内扩展到节点间的场景
    # NOTE(yongji): handling scaling from intra-node to inter-node
    if parallel_config.enable_elastic_ep:
        client_local_only = False

    def _addr() -> str:
        """生成单个地址：IPC 或 TCP"""
        if client_local_only:
            return get_open_zmq_ipc_path()
        return get_tcp_uri(host, 0 if defer_api_server_ports else get_open_port())

    return EngineZmqAddresses(
        inputs=[_addr() for _ in range(num_api_servers)],
        outputs=[_addr() for _ in range(num_api_servers)],
    )


@contextlib.contextmanager
def launch_core_engines(
    vllm_config: VllmConfig,
    executor_class: type[Executor],
    log_stats: bool,
    addresses: EngineZmqAddresses,
    num_api_servers: int = 1,
) -> Iterator[
    tuple[
        CoreEngineProcManager | CoreEngineActorManager | None,
        DPCoordinator | None,
        EngineZmqAddresses,
        Queue | None,
    ]
]:
    """
    启动引擎和 DP 协调器进程的上下文管理器

    根据配置启动必要的组件：
    1. DP 协调器（如果需要）：负责收集队列统计信息和协调 MoE 波次
    2. 引擎核心进程/Actor：
       - Ray 模式：使用 CoreEngineActorManager
       - 本地模式：使用 CoreEngineProcManager
    3. 多模态张量队列（如果需要）：用于在 API 服务器和引擎核心之间共享张量

    参数：
        vllm_config: vLLM 配置对象
        executor_class: 执行器类
        log_stats: 是否记录统计信息
        addresses: ZMQ 地址集合
        num_api_servers: API 服务器数量

    产出：
        (engine_manager, coordinator, addresses, tensor_queue) 元组
    """
    """Launch engine and DP coordinator processes as needed."""

    parallel_config = vllm_config.parallel_config
    dp_size = parallel_config.data_parallel_size
    local_engine_count = parallel_config.data_parallel_size_local
    local_start_index = parallel_config.data_parallel_rank_local
    dp_rank = parallel_config.data_parallel_rank
    host = parallel_config.data_parallel_master_ip
    local_engines_only = parallel_config.local_engines_only

    offline_mode = local_start_index is not None

    # 创建多模态张量 IPC 队列
    # 用于在 API 服务器和引擎核心之间共享多模态张量。
    # 目前仅支持 DP=1 的数据流。
    tensor_queue: Queue | None = None
    multimodal_config = vllm_config.model_config.multimodal_config
    if multimodal_config is not None and multimodal_config.mm_tensor_ipc == "torch_shm":
        tensor_queue = get_mp_context().Queue()

    # 在在线 DP 模式下，rank 0 运行 DP 协调器。
    # 协调器用于：
    # 1. 内部/混合负载均衡：收集和发布队列统计信息
    # 2. MoE 模型：波次协调（除了统计信息外）
    run_coordinator = (
        vllm_config.needs_dp_coordinator and not offline_mode and dp_rank == 0
    )

    if run_coordinator:
        coordinator = DPCoordinator(
            parallel_config,
            enable_wave_coordination=vllm_config.model_config.is_moe,
        )

        # 获取协调器的 socket 地址
        addresses.coordinator_input, addresses.coordinator_output = (
            coordinator.get_engine_socket_addresses()
        )
        addresses.frontend_stats_publish_address = (
            coordinator.get_stats_publish_address()
        )

        logger.info("Started DP Coordinator process (PID: %d)", coordinator.proc.pid)
    else:
        coordinator = None

    # Ray 后端：使用 CoreEngineActorManager
    if parallel_config.data_parallel_backend == "ray":
        logger.info("Starting ray-based data parallel backend")

        engine_actor_manager = CoreEngineActorManager(
            vllm_config=vllm_config,
            addresses=addresses,
            executor_class=executor_class,
            log_stats=log_stats,
        )

        yield engine_actor_manager, coordinator, addresses, tensor_queue
        return

    # 本地后端：使用 CoreEngineProcManager
    # 确定需要握手的引擎列表
    if offline_mode:
        assert local_engine_count == 1
        engines_to_handshake = [CoreEngine(index=dp_rank, local=True)]
    elif dp_rank == 0:
        # Rank 0 持有协调器，因此与所有核心引擎握手
        # （包括外部负载均衡和内部负载均衡模式）
        engines_to_handshake = [
            CoreEngine(index=i, local=(i < local_engine_count)) for i in range(dp_size)
        ]
    else:
        # Rank > 0 只与它管理的本地核心引擎握手
        assert local_engines_only, (
            "Attempting to launch core_engines from dp_rank > 0, but "
            "found internal DPLB, which is incompatible."
        )
        engines_to_handshake = [
            CoreEngine(index=i, local=True)
            for i in range(dp_rank, dp_rank + local_engine_count)
        ]

    # 判断启动的引擎是否只与本地前端握手
    # 在 external_dp_lb 模式下，rank > 0 会同时与本地前端和 rank 0 前端握手
    handshake_local_only = offline_mode or local_engine_count == dp_size

    # 处理从节点内扩展到节点间的场景
    # NOTE(yongji): handling scaling from intra-node to inter-node
    if parallel_config.enable_elastic_ep:
        handshake_local_only = False

    # 保留 "port=0 表示自动选择" 的握手地址语义。
    # 握手地址在此进程中生成的引擎使用，因此不能延迟端口分配。
    rpc_port = parallel_config.data_parallel_rpc_port or get_open_port()
    handshake_address = get_engine_client_zmq_addr(handshake_local_only, host, rpc_port)

    # 处理本地引擎的特殊情况
    if local_engines_only and dp_rank > 0:
        assert not handshake_local_only
        local_handshake_address = get_open_zmq_ipc_path()
        client_handshake_address = local_handshake_address
    else:
        local_handshake_address = handshake_address
        client_handshake_address = None

    # 创建握手 socket 并启动引擎
    with zmq_socket_ctx(
        local_handshake_address, zmq.ROUTER, bind=True
    ) as handshake_socket:
        # 启动本地引擎
        if local_engine_count:
            local_engine_manager = CoreEngineProcManager(
                vllm_config=vllm_config,
                executor_class=executor_class,
                log_stats=log_stats,
                handshake_address=handshake_address,
                client_handshake_address=client_handshake_address,
                local_client=True,
                local_engine_count=local_engine_count,
                start_index=dp_rank,
                local_start_index=local_start_index or 0,
                tensor_queue=tensor_queue,
            )
        else:
            local_engine_manager = None

        # 产出管理器实例，供调用者使用
        yield local_engine_manager, coordinator, addresses, tensor_queue

        # 等待引擎启动完成
        wait_for_engine_startup(
            handshake_socket,
            addresses,
            engines_to_handshake,
            parallel_config,
            dp_size > 1 and vllm_config.model_config.is_moe,
            vllm_config.cache_config,
            local_engine_manager,
            coordinator.proc if coordinator else None,
        )


def wait_for_engine_startup(
    handshake_socket: zmq.Socket,
    addresses: EngineZmqAddresses,
    core_engines: list[CoreEngine],
    parallel_config: ParallelConfig,
    coordinated_dp: bool,
    cache_config: CacheConfig,
    proc_manager: CoreEngineProcManager | None,
    coord_process: Process | None,
):
    """
    等待引擎核心进程启动完成

    实现引擎启动握手协议：
    1. 引擎进程发送 HELLO 消息，表示已启动
    2. 前端回复初始化消息（包含 ZMQ 地址和配置信息）
    3. 引擎进程完成初始化后发送 READY 消息
    4. 前端验证所有引擎都已就绪

    状态转换：NEW -> CONNECTED -> READY

    参数：
        handshake_socket: ZMQ ROUTER socket，用于接收引擎消息
        addresses: ZMQ 地址集合，发送给引擎用于建立连接
        core_engines: 引擎核心列表，跟踪每个引擎的状态
        parallel_config: 并行配置
        coordinated_dp: 是否为协调 DP 模式（MoE 模型需要）
        cache_config: 缓存配置
        proc_manager: 进程管理器（本地模式）
        coord_process: 协调器进程（如果存在）

    异常：
        RuntimeError: 如果引擎启动失败、配置不匹配或收到意外消息
    """
    # Wait for engine core process(es) to send ready messages.
    local_count = parallel_config.data_parallel_size_local
    remote_count = len(core_engines) - local_count
    # [local, remote] 计数器：跟踪待连接和待启动的引擎数
    conn_pending, start_pending = [local_count, remote_count], [0, 0]
    poller = zmq.Poller()
    poller.register(handshake_socket, zmq.POLLIN)

    # 远程引擎是否应该为 headless 模式
    remote_should_be_headless = (
        not parallel_config.data_parallel_hybrid_lb
        and not parallel_config.data_parallel_external_lb
    )

    # 注册进程 sentinel 到 poller，用于检测进程退出
    if proc_manager is not None:
        for sentinel in proc_manager.sentinels():
            poller.register(sentinel, zmq.POLLIN)
    if coord_process is not None:
        poller.register(coord_process.sentinel, zmq.POLLIN)

    # 主循环：等待所有引擎完成握手
    while any(conn_pending) or any(start_pending):
        events = poller.poll(STARTUP_POLL_PERIOD_MS)
        if not events:
            # 超时，打印等待状态日志
            if any(conn_pending):
                logger.debug(
                    "Waiting for %d local, %d remote core engine proc(s) to connect.",
                    *conn_pending,
                )
            if any(start_pending):
                logger.debug(
                    "Waiting for %d local, %d remote core engine proc(s) to start.",
                    *start_pending,
                )
            continue
        if len(events) > 1 or events[0][0] != handshake_socket:
            # 本地核心进程退出（非正常）
            finished = proc_manager.finished_procs() if proc_manager else {}
            if coord_process is not None and coord_process.exitcode is not None:
                finished[coord_process.name] = coord_process.exitcode
            raise RuntimeError(
                "Engine core initialization failed. "
                "See root cause above. "
                f"Failed core proc(s): {finished}"
            )

        # 从握手 socket 接收 HELLO 和 READY 消息
        eng_identity, ready_msg_bytes = handshake_socket.recv_multipart()
        eng_index = int.from_bytes(eng_identity, "little")
        # 查找对应的引擎核心对象
        engine = next((e for e in core_engines if e.identity == eng_identity), None)
        if engine is None:
            raise RuntimeError(
                f"Message from engine with unexpected data parallel rank: {eng_index}"
            )
        # 解码消息
        msg = msgspec.msgpack.decode(ready_msg_bytes)
        status, local, headless = msg["status"], msg["local"], msg["headless"]

        # 验证引擎的本地/远程属性
        if local != engine.local:
            raise RuntimeError(
                f"{status} message from "
                f"{'local' if local else 'remote'} "
                f"engine {eng_index}, expected it to be "
                f"{'local' if engine.local else 'remote'}"
            )

        # 验证远程引擎的 headless 模式
        if not local and headless != remote_should_be_headless:
            if headless:
                raise RuntimeError(
                    f"Remote engine {eng_index} must not use "
                    f"--headless in external or hybrid dp lb "
                    f"mode"
                )
            else:
                raise RuntimeError(
                    f"Remote engine {eng_index} must use "
                    f"--headless unless in external or hybrid "
                    f"dp lb mode"
                )

        # 处理 HELLO 消息（状态：NEW -> CONNECTED）
        if status == "HELLO" and engine.state == CoreEngineState.NEW:
            # 构建初始化消息，包含 ZMQ 地址和并行配置
            init_message = msgspec.msgpack.encode(
                EngineHandshakeMetadata(
                    addresses=addresses,
                    parallel_config={
                        k: getattr(parallel_config, k)
                        for k in (
                            "data_parallel_master_ip",
                            "data_parallel_master_port",
                            "_data_parallel_master_port_list",
                            "data_parallel_size",
                        )
                    }
                    if coordinated_dp
                    else {},
                )
            )
            # 发送初始化消息给引擎
            handshake_socket.send_multipart((eng_identity, init_message), copy=False)
            # 更新计数器
            conn_pending[0 if local else 1] -= 1
            start_pending[0 if local else 1] += 1
            engine.state = CoreEngineState.CONNECTED
        # 处理 READY 消息（状态：CONNECTED -> READY）
        elif status == "READY" and engine.state == CoreEngineState.CONNECTED:
            # 验证 DP worker 之间的配置哈希一致性（MoE 模型需要）
            if coordinated_dp:
                worker_config_hash = msg.get("parallel_config_hash")
                expected_hash = parallel_config.compute_hash()
                if worker_config_hash != expected_hash:
                    raise RuntimeError(
                        f"Configuration mismatch detected for engine "
                        f"{eng_index}. All DP workers must have identical "
                        f"configurations for parameters that affect collective "
                        f"communication (e.g., enable_eplb, "
                        f"eplb_config.log_balancedness). "
                        f"Worker hash: {worker_config_hash}, "
                        f"Expected hash: {expected_hash}. "
                        f"Please ensure all workers are started with the same "
                        f"command-line arguments."
                    )

            # 更新计数器
            start_pending[0 if local else 1] -= 1
            engine.state = CoreEngineState.READY
        else:
            # 收到意外消息
            raise RuntimeError(
                f"Unexpected {status} message for "
                f"{'local' if local else 'remote'} engine "
                f"{eng_index} in {engine.state} state."
            )

        # 记录握手进度日志
        logger.debug(
            "%s from %s core engine process %s.",
            status,
            "local" if local else "remote",
            eng_index,
        )
