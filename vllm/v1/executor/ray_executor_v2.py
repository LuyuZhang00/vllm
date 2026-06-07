# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
RayExecutorV2 -- 基于 Ray 的 v2 分布式执行器模块。

模块概述:
    本模块实现了 vLLM v1 引擎的第二代 Ray 分布式执行器。
    与 v1 版本的 RayExecutor 相比，v2 有以下核心区别:
      1. 继承自 MultiprocExecutor（多进程执行器），复用其基于 MessageQueue 的
         控制面和 NCCL 数据面通信机制，而非 v1 中通过 Ray ObjectRef 传递输入输出。
      2. Worker 以 Ray Actor 的方式运行，但内部使用共享内存（同节点）或 TCP
         （跨节点）的 MessageQueue 进行高效通信，避免了 Ray ObjectRef 的序列化开销。
      3. 采用两阶段初始化策略: __init__ 仅做轻量级配置存储，initialize_worker
         在 GPU ID 确定后才执行完整的 WorkerProc 初始化。
      4. 支持异步调度（继承自 MultiprocExecutor），这对 RayExecutorV2 的性能至关重要。

通信架构:
    - 输入广播: 驱动端通过 rpc_broadcast_mq（MessageQueue）向所有 worker 广播
      SchedulerOutput。同节点 worker 使用共享内存，跨节点 worker 使用 TCP。
    - 响应收集: 每个 worker 通过各自的 worker_response_mq 将结果返回给驱动端。
    - 数据并行: 模型权重和梯度同步通过 NCCL 完成（继承自 MultiprocExecutor）。

与 v1 RayExecutor 的主要区别:
    | 特性               | v1 RayExecutor            | v2 RayExecutorV2          |
    |--------------------|---------------------------|---------------------------|
    | 基类               | Executor（抽象基类）       | MultiprocExecutor          |
    | 通信方式           | Ray ObjectRef             | MessageQueue (SHM/TCP)     |
    | 控制面             | Ray remote 调用           | MessageQueue 广播          |
    | 序列化开销         | cloudpickle               | 共享内存零拷贝             |
    | 异步调度           | 需自行实现                | 继承自 MultiprocExecutor   |
    | Worker 生命周期    | Actor 生命周期             | 两阶段初始化 + 监控线程    |
"""
import copy
import os
import threading
import weakref
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.distributed.device_communicators.shm_broadcast import (
    Handle,
    MessageQueue,
)
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.network_utils import (
    get_distributed_init_method,
    get_open_port,
)
from vllm.v1.executor.multiproc_executor import (
    FutureWrapper,
    MultiprocExecutor,
    WorkerProc,
)
from vllm.v1.executor.ray_env_utils import get_driver_env_vars
from vllm.v1.executor.ray_utils import (
    WORKER_SPECIFIC_ENV_VARS,
    build_actor_name,
    get_bundles_for_indices,
    get_bundles_sorted_by_node,
    initialize_ray_cluster,
    ray,
)

if ray is not None:
    from ray.actor import ActorHandle
    from ray.types import ObjectRef
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
else:
    ActorHandle = None

logger = init_logger(__name__)


@dataclass
class RayWorkerHandle:
    """Ray worker Actor 的句柄，兼容 MultiprocExecutor 的接口。

    作用:
        封装一个 Ray Actor 的引用及其元信息（rank、节点 ID 等），
        使 RayExecutorV2 能像管理本地进程一样管理远程 Ray Actor。
        其中 run_ref 字段用于健康监控 -- 当 Actor 意外退出时，
        ray.wait() 会检测到该 ObjectRef 完成，从而触发故障处理。
    """

    actor: ActorHandle
    """Ray worker Actor 句柄"""

    rank: int
    """Worker 的全局 rank（在整个分布式组中的编号）"""

    local_rank: int
    """Worker 在其所在节点上的本地 rank"""

    node_id: str
    """Worker 所在的 Ray 节点 ID"""

    bundle_id_idx: int = -1
    """Placement group 中的 bundle 索引，用于将 worker 绑定到特定 GPU"""

    run_ref: ObjectRef | None = None
    """run() 方法的 ObjectRef，同时用作健康监控的哨兵值。
    当 Actor 正常运行时此 ref 持续 pending；Actor 退出时 ref 完成。"""

    def run(self):
        """启动 worker 的主循环（busy loop）。

        通过 Ray remote 调用触发 Actor 上的 run() 方法，
        返回的 ObjectRef 存储在 run_ref 中用于后续健康检查。
        """
        self.run_ref = self.actor.run.remote()


class RayWorkerProc(WorkerProc):
    """运行在 Ray Actor 内部的 Worker 进程。

    与标准 WorkerProc 的区别:
        标准 WorkerProc 在 __init__ 中完成所有初始化（包括 CUDA 设备设置）。
        而 RayWorkerProc 将初始化拆分为两个阶段，以解决 Ray 环境下
        CUDA_VISIBLE_DEVICES 需要在 Actor 调度后才能确定的问题。

    两阶段初始化流程:
        1. __init__: 仅存储初始化参数，不进行任何设备或模型初始化。
           此阶段在 Actor 创建时立即执行，开销极小。
        2. initialize_worker(): 在 GPU ID 通过 Ray 运行时上下文发现后调用，
           完成完整的 WorkerProc 初始化，包括设置正确的 local_rank 和
           CUDA_VISIBLE_DEVICES。

    CUDA_VISIBLE_DEVICES 设置流程（关键设计）:
        1. RayExecutorV2 启用 RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES 环境变量，
           阻止 Ray 在 Actor 创建时自动设置 CUDA_VISIBLE_DEVICES。
        2. 每个 Actor 被调度到 placement group 的特定 bundle 上；
           Ray 在调度时解析该 bundle 对应的物理 GPU ID。
        3. Actor 调度完成后，worker 通过 Ray 运行时上下文发现 GPU ID，
           然后在完成 WorkerProc 初始化之前设置 CUDA_VISIBLE_DEVICES。

    为什么需要这个 unset-and-reset 序列:
        当 placement group 由外部管理时，必须等待调度完成才能确定
        CUDA_VISIBLE_DEVICES 应该绑定到哪个 GPU。这个序列允许多个
        vLLM 实例共存于同一节点: 每个实例无需知道其他实例占用了哪些
        物理设备，外部管理的 placement group 通过将 worker 绑定到特定
        bundle 来避免 CUDA_VISIBLE_DEVICES 冲突。
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        rank: int,
        distributed_init_method: str,
        input_shm_handle: Handle,
        is_driver_worker: bool,
        is_driver_node: bool = False,
    ):
        # 延迟调用 WorkerProc.__init__，直到 GPU ID 确定后再执行。
        # 这里只保存参数，不做任何设备初始化。
        self._is_driver_node = is_driver_node
        self._init_kwargs = dict(
            vllm_config=vllm_config,
            rank=rank,
            distributed_init_method=distributed_init_method,
            input_shm_handle=input_shm_handle,
            shared_worker_lock=None,
            is_driver_worker=is_driver_worker,
        )

    def get_node_and_gpu_ids(self) -> tuple[str, list[int]]:
        """返回 Ray 分配给此 Actor 的 (node_id, gpu_ids)。

        通过 Ray 运行时上下文获取:
        - node_id: Actor 所在节点的唯一标识
        - gpu_ids: 该 Actor 被分配的 GPU ID 列表

        Returns:
            (node_id, gpu_ids): 节点 ID 和 GPU ID 列表的元组
        """
        node_id = ray.get_runtime_context().get_node_id()
        device_key = current_platform.ray_device_key
        if not device_key:
            raise RuntimeError(
                f"current platform {current_platform.device_name} does not support ray."
            )
        gpu_ids = ray.get_runtime_context().get_accelerator_ids()[device_key]
        return node_id, [int(x) for x in gpu_ids]

    def initialize_worker(
        self,
        local_rank: int,
        env_vars: dict[str, str],
        driver_env_vars: dict[str, str] | None = None,
    ) -> None:
        """第二阶段初始化: 在 GPU 分配确定后完成 WorkerProc 的完整初始化。

        环境变量设置策略:
        - driver_env_vars（驱动端环境变量）: 使用 setdefault 语义，即只填充
          缺失的变量，不覆盖节点本地已有的值。这保证了节点级别的配置优先。
        - env_vars（worker 特定环境变量，如 CUDA_VISIBLE_DEVICES）: 始终覆盖，
          因为这些值必须精确匹配 Ray 调度结果。

        Args:
            local_rank: 该 worker 在节点内的本地 rank
            env_vars: 需要强制设置的环境变量（会覆盖已有值）
            driver_env_vars: 驱动端环境变量（仅填充缺失项）
        """
        if driver_env_vars:
            for key, value in driver_env_vars.items():
                os.environ.setdefault(key, value)
        for key, value in env_vars.items():
            os.environ[key] = value

        self.local_rank = local_rank
        super().__init__(
            local_rank=local_rank,
            **self._init_kwargs,
        )

    def _init_message_queues(
        self, input_shm_handle: Handle, vllm_config: VllmConfig
    ) -> None:
        """初始化 MessageQueue 通信通道。

        通信策略:
        - 与驱动端在同一节点的 worker: 使用共享内存（SHM）进行广播输入和
          响应通信，延迟极低。
        - 与驱动端不在同一节点的 worker: 使用 TCP 进行通信（n_local_reader=0）。

        注意: 使用 ray.util.get_node_ip_address() 获取 Ray 内部 IP，
        而非 get_ip()。因为 get_ip() 返回主机的外部 IP，在集群内部节点之间
        通常不可路由。
        """
        self.rpc_broadcast_mq = MessageQueue.create_from_handle(
            input_shm_handle, self.worker.rank
        )

        n_local = 1 if self._is_driver_node else 0
        # 使用 Ray 内部 IP 地址，确保跨节点可路由性
        self.worker_response_mq = MessageQueue(
            n_reader=1,
            n_local_reader=n_local,
            connect_ip=ray.util.get_node_ip_address(),
        )
        self.peer_response_handles: list[dict] = []

    def wait_for_init(self) -> dict:
        """响应驱动端的 wait_until_ready() 屏障同步。

        返回包含就绪状态和响应 MessageQueue 句柄的字典，
        驱动端据此创建对应的 MessageQueue 实例以接收该 worker 的响应。

        Returns:
            包含 status 和 handle 的字典
        """
        assert self.worker_response_mq is not None
        return {
            "status": self.READY_STR,
            "handle": self.worker_response_mq.export_handle(),
        }

    def run(self) -> None:
        """Actor 主入口，通过 actor.run.remote() 调用。

        执行流程:
        1. 等待广播 MessageQueue 和响应 MessageQueue 就绪
        2. 进入 worker_busy_loop()（从 MultiprocExecutor 继承的主循环）
        3. 退出时执行 shutdown() 清理资源

        异常处理:
            捕获所有异常并记录日志，然后重新抛出，确保 Actor 以失败状态退出。
            RayWorkerMonitor 会检测到 Actor 退出并触发故障处理。
        """
        try:
            assert self.rpc_broadcast_mq is not None
            self.rpc_broadcast_mq.wait_until_ready()
            assert self.worker_response_mq is not None
            self.worker_response_mq.wait_until_ready()

            self.worker_busy_loop()
        except Exception as e:
            logger.exception("RayWorkerProc failed: %s", e)
            raise
        finally:
            self.shutdown()


class RayExecutorV2(MultiprocExecutor):
    """基于 Ray 的 v2 分布式执行器，使用 MessageQueue 进行通信。

    设计理念:
        继承自 MultiprocExecutor，复用其基于 MessageQueue 的控制面
        和 NCCL 数据面。Worker 以 Ray Actor 的形式运行，但内部通信
        使用高效的 MessageQueue（共享内存/ TCP），而非 Ray ObjectRef。

    关键特性:
        1. 异步调度: 继承自 MultiprocExecutor 的异步调度能力，
           这对 RayExecutorV2 的性能至关重要。
        2. Pipeline Parallelism (PP) 支持: supports_pp = True。
        3. 健康监控: 通过监控 run() ObjectRef 检测 worker 意外退出。
        4. 两阶段初始化: Actor 创建时轻量级，GPU 分配后完整初始化。

    适用场景:
        - 多节点分布式推理（跨节点通信通过 TCP MessageQueue）
        - 需要与 Ray 集群集成的部署环境
        - 多个 vLLM 实例共存于同一节点的场景
    """

    uses_ray: bool = True
    """标识此执行器使用 Ray 作为分布式运行时"""

    supports_pp: bool = True
    """支持 Pipeline Parallelism（流水线并行）"""

    def __init__(self, vllm_config: VllmConfig):
        super().__init__(vllm_config)

    def _build_runtime_env(self) -> dict:
        """构建 Ray Actor 的 runtime_env 配置字典。

        包含以下配置:
        1. 用户自定义的 ray_runtime_env（来自 parallel_config）
        2. 平台特定的设备环境变量（阻止 Ray 自动设置 CUDA_VISIBLE_DEVICES）
        3. Nsight 性能分析配置（如果启用了 ray_workers_use_nsight）

        注意: 驱动端环境变量（driver env vars）不在这里设置，
        而是通过 initialize_worker 以 setdefault 语义单独应用，
        以避免覆盖节点本地的配置值。

        Returns:
            Ray runtime_env 字典
        """
        base = self.parallel_config.ray_runtime_env
        runtime_env: dict = copy.deepcopy(dict(base)) if base else {}

        env_vars = runtime_env.setdefault("env_vars", {})
        env_vars.update({v: "1" for v in current_platform.ray_noset_device_env_vars})
        if self.parallel_config.ray_workers_use_nsight:
            runtime_env["nsight"] = {
                "t": "cuda,cudnn,cublas",
                "o": "'worker_process_%p'",
                "cuda-graph-trace": "node",
            }
        return runtime_env

    @staticmethod
    def _get_actor_resource_kwargs() -> dict[str, Any]:
        """返回当前平台的 Ray Actor 资源请求参数。

        不同平台使用不同的资源标识:
        - GPU 平台: 使用 num_gpus 参数
        - 其他平台（如 TPU、XPU）: 使用 resources 字典指定自定义资源

        Returns:
            包含资源请求的关键字参数字典
        """
        num_devices = envs.VLLM_RAY_PER_WORKER_GPUS
        device_key = current_platform.ray_device_key
        if device_key == "GPU":
            return {"num_gpus": num_devices}
        return {"num_gpus": 0, "resources": {device_key: num_devices}}

    def _init_executor(self) -> None:
        """初始化 RayExecutorV2 执行器。

        这是整个执行器的核心初始化方法，包含 10 个步骤:

        Step 1: 初始化 Ray 集群并获取 placement group
        Step 2: 构建 bundle 分配方案，将 rank 映射到 placement group bundle
        Step 3: 解析 torch.distributed TCPStore 的 IP 地址（rank 0 所在节点）
        Step 4: 创建广播 MessageQueue（同节点用 SHM，跨节点用 TCP）
        Step 5: 创建 RayWorkerProc Actor（延迟初始化，仅存储参数）
        Step 6: 通过 Ray 运行时上下文发现每个 worker 的 GPU ID
        Step 7: 使用正确的 local_rank 和 CUDA_VISIBLE_DEVICES 初始化 worker
        Step 8: 收集各 worker 的响应 MessageQueue 句柄
        Step 9: 启动 worker 的 run() 主循环
        Step 10: 执行 wait_until_ready() 屏障同步，确保所有通信通道就绪
        """
        # 注册析构器，确保执行器被垃圾回收时能正确清理资源
        self._finalizer = weakref.finalize(self, self.shutdown)
        self.is_failed = False
        self.failure_callback = None
        self.shutting_down = False
        self.shutdown_lock = threading.Lock()

        # Step 1: 初始化 Ray 集群并获取 placement group。
        # require_gpu_on_driver=False 表示驱动节点不需要 GPU（纯调度角色）。
        if ray is None:
            raise ImportError("Using Ray backend requires installation of ray.")
        initialize_ray_cluster(self.parallel_config, require_gpu_on_driver=False)
        placement_group = self.parallel_config.placement_group

        # 验证 world_size 与各并行维度的乘积一致
        tp_size, pp_size, pcp_size = self._get_parallel_sizes()
        assert self.world_size == tp_size * pp_size * pcp_size, (
            f"world_size ({self.world_size}) must be equal to the "
            f"tensor_parallel_size ({tp_size}) x pipeline"
            f"_parallel_size ({pp_size}) x prefill_context"
            f"_parallel_size ({pcp_size}). "
        )

        # Step 2: 构建 bundle 分配方案。
        # 如果用户指定了 VLLM_RAY_BUNDLE_INDICES，则按指定索引分配；
        # 否则按节点排序分配，确保同一节点的 worker 连续编号。
        if envs.VLLM_RAY_BUNDLE_INDICES:
            bundle_to_node_id = get_bundles_for_indices(
                placement_group,
                list(map(int, envs.VLLM_RAY_BUNDLE_INDICES.split(","))),
                self.world_size,
            )
        else:
            bundle_to_node_id = get_bundles_sorted_by_node(placement_group)
        driver_node = ray.get_runtime_context().get_node_id()

        bundle_assignments: list[dict[str, Any]] = []
        for rank, (bundle_id_idx, node_id, node_ip) in enumerate(bundle_to_node_id):
            bundle_assignments.append(
                {
                    "rank": rank,
                    "bundle_id_idx": bundle_id_idx,
                    "node_id": node_id,
                    "node_ip": node_ip,
                }
            )

        # Step 3: 解析 torch.distributed TCPStore 的地址。
        # TCPStore 服务端运行在 rank 0 所在的节点上，
        # 所有 worker（包括跨节点的）必须能够访问此地址。
        dist_ip = bundle_assignments[0]["node_ip"]
        distributed_init_method = get_distributed_init_method(dist_ip, get_open_port())

        # Step 4: 创建广播 MessageQueue。
        # n_local 表示与驱动端在同一节点的 worker 数量，
        # 这些 worker 使用共享内存通信；其余 worker 使用 TCP。
        max_chunk_bytes = envs.VLLM_MQ_MAX_CHUNK_BYTES_MB * 1024 * 1024
        n_local = sum(1 for a in bundle_assignments if a["node_id"] == driver_node)
        self.rpc_broadcast_mq = MessageQueue(
            self.world_size,
            n_local,
            max_chunk_bytes=max_chunk_bytes,
            connect_ip=ray.util.get_node_ip_address(),
        )
        scheduler_output_handle = self.rpc_broadcast_mq.export_handle()

        # Step 5: 创建 RayWorkerProc Actor（延迟初始化）。
        # 此阶段 Actor 仅存储初始化参数，不进行设备/模型初始化。
        # 完整初始化在 Step 7 中 GPU ID 发现后执行。
        self.ray_worker_handles: list[RayWorkerHandle] = []
        instance_id = self.vllm_config.instance_id

        # 收集驱动端环境变量，后续通过 setdefault 语义应用到 worker
        self.driver_env_vars = get_driver_env_vars(
            worker_specific_vars=WORKER_SPECIFIC_ENV_VARS,
        )

        runtime_env = self._build_runtime_env()
        resource_kwargs = self._get_actor_resource_kwargs()

        for bundle_idx in range(self.world_size):
            bundle = bundle_assignments[bundle_idx]
            is_driver_worker = self._is_driver_worker(bundle["rank"])
            is_driver_node = bundle["node_id"] == driver_node

            # 使用 PlacementGroupSchedulingStrategy 将 Actor 绑定到特定 bundle，
            # 确保 Actor 被调度到正确的节点和 GPU 上
            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=placement_group,
                placement_group_bundle_index=bundle["bundle_id_idx"],
            )

            # 构建 Actor 名称，包含实例 ID、rank 和并行配置信息
            actor_name = build_actor_name(
                instance_id, bundle["rank"], tp_size, pp_size, pcp_size
            )

            # 创建 Ray Actor: num_cpus=0 表示不占用 CPU 资源（GPU 密集型任务）
            actor = (
                ray.remote(RayWorkerProc)
                .options(
                    name=actor_name,
                    num_cpus=0,
                    **resource_kwargs,
                    scheduling_strategy=scheduling_strategy,
                    runtime_env=runtime_env,
                )
                .remote(
                    vllm_config=self.vllm_config,
                    rank=bundle["rank"],
                    distributed_init_method=distributed_init_method,
                    input_shm_handle=scheduler_output_handle,
                    is_driver_worker=is_driver_worker,
                    is_driver_node=is_driver_node,
                )
            )

            handle = RayWorkerHandle(
                actor=actor,
                rank=bundle["rank"],
                local_rank=-1,  # 在 Step 7 GPU ID 发现后设置
                node_id=bundle["node_id"],
                bundle_id_idx=bundle["bundle_id_idx"],
            )
            self.ray_worker_handles.append(handle)

        # Step 6: 通过 Ray 运行时上下文发现每个 worker 的 GPU ID。
        # 这一步必须在 Actor 调度完成后执行，因为 GPU ID 由 Ray 在调度时确定。
        worker_node_and_gpu_ids = ray.get(
            [h.actor.get_node_and_gpu_ids.remote() for h in self.ray_worker_handles]
        )

        # 按节点分组，记录每个节点上的 worker 索引和 GPU ID 列表
        node_workers: dict[str, list[int]] = defaultdict(list)
        node_gpus: dict[str, list[int]] = defaultdict(list)
        for i, (node_id, gpu_ids) in enumerate(worker_node_and_gpu_ids):
            node_workers[node_id].append(i)
            node_gpus[node_id].extend(gpu_ids)
        for node_id, gpu_ids in node_gpus.items():
            node_gpus[node_id] = sorted(gpu_ids)

        # Step 7: 使用正确的 local_rank 和 CUDA_VISIBLE_DEVICES 初始化每个 worker。
        # local_rank 是 worker 在其节点内的索引；
        # CUDA_VISIBLE_DEVICES 包含该节点上所有分配给此执行器的 GPU ID。
        init_worker_refs = []
        for i, (node_id, _) in enumerate(worker_node_and_gpu_ids):
            local_rank = node_workers[node_id].index(i)
            worker_env_vars = {
                current_platform.device_control_env_var: ",".join(
                    map(str, node_gpus[node_id])
                ),
            }
            self.ray_worker_handles[i].local_rank = local_rank
            init_worker_refs.append(
                self.ray_worker_handles[i].actor.initialize_worker.remote(
                    local_rank, worker_env_vars, self.driver_env_vars
                )
            )
        # 等待所有 worker 完成初始化
        ray.get(init_worker_refs)

        # Step 8: 收集各 worker 的响应 MessageQueue 句柄。
        # 每个 worker 在 wait_for_init() 中导出其 worker_response_mq 的句柄，
        # 驱动端据此创建 MessageQueue 实例以接收该 worker 的响应。
        init_results = ray.get(
            [h.actor.wait_for_init.remote() for h in self.ray_worker_handles]
        )

        self.response_mqs: list[MessageQueue] = []
        for i, result in enumerate(init_results):
            if result["status"] != RayWorkerProc.READY_STR:
                raise RuntimeError(f"Worker {i} failed to initialize: {result}")
            self.response_mqs.append(
                MessageQueue.create_from_handle(result["handle"], 0)
            )

        # Step 9: 启动 worker 的 run() 主循环。
        # 注意: 必须在 wait_until_ready() 之前启动，因为 worker 在 run() 内部
        # 发送订阅消息；如果顺序颠倒会导致死锁。
        for handle in self.ray_worker_handles:
            handle.run()

        # Step 10: 执行 wait_until_ready() 屏障同步。
        # 确保所有 MessageQueue 通信通道（广播和响应）都已就绪，
        # 之后驱动端才能安全地向 worker 发送调度命令。
        self.rpc_broadcast_mq.wait_until_ready()
        for response_mq in self.response_mqs:
            response_mq.wait_until_ready()

        self.futures_queue = deque[FutureWrapper]()
        self._post_init_executor()

        # 启动 worker 健康监控线程
        self.start_worker_monitor()
        self.output_rank = self._get_output_rank()

    def start_worker_monitor(self, inline=False) -> None:
        """启动 worker 存活监控线程。

        监控机制:
            使用 ray.wait() 轮询所有 worker 的 run() ObjectRef。
            当某个 worker 的 Actor 意外退出时，其对应的 ObjectRef 会完成
            （无论正常退出还是异常退出），ray.wait() 检测到后触发故障处理。

        故障处理流程:
            1. 检测到 worker 的 run_ref 完成
            2. 标记执行器为失败状态 (is_failed = True)
            3. 记录错误日志
            4. 调用 shutdown() 清理所有资源
            5. 调用 failure_callback（如果有）通知上层组件

        轮询策略:
            使用 5 秒超时的 ray.wait() 而非阻塞调用，原因是在 Ray 被销毁时，
            阻塞在 ray.wait() 内部会导致段错误（segfault）。超时轮询允许
            监控线程在 _should_stop() 返回 True 时安全退出。
        """
        run_refs = [h.run_ref for h in self.ray_worker_handles if h.run_ref is not None]
        if not run_refs:
            raise RuntimeError("Ray workers have not started successfully.")

        self_ref = weakref.ref(self)
        ref_to_rank = {
            h.run_ref: h.rank for h in self.ray_worker_handles if h.run_ref is not None
        }

        def _should_stop() -> bool:
            """检查监控线程是否应该停止。"""
            executor = self_ref()
            return not executor or executor.shutting_down

        def monitor_workers():
            """监控线程主函数，持续轮询 worker 存活状态。"""
            # 使用超时轮询而非阻塞调用，避免 Ray 销毁时的段错误
            while not _should_stop() and ray.is_initialized():
                try:
                    done, _ = ray.wait(run_refs, num_returns=1, timeout=5.0)
                except Exception:
                    logger.exception(
                        "RayWorkerMonitor: unexpected error, exiting monitor thread"
                    )
                    return
                if not done or _should_stop():
                    continue

                # 检测到有 worker 退出，触发故障处理
                dead_ranks = [ref_to_rank[r] for r in done]
                executor = self_ref()
                if not executor:
                    return
                executor.is_failed = True
                logger.error(
                    "RayWorkerProc rank=%s died unexpectedly, shutting down executor.",
                    dead_ranks,
                )
                executor.shutdown()
                if executor.failure_callback is not None:
                    callback = executor.failure_callback
                    executor.failure_callback = None
                    callback()
                return

        t = threading.Thread(
            target=monitor_workers, daemon=True, name="RayWorkerMonitor"
        )
        t.start()
        self._monitor_thread = t

    def _join_monitor_thread(self) -> None:
        """等待监控线程退出。

        必须在销毁 Ray 资源之前调用此方法。原因:
        监控线程可能正阻塞在 ray.wait() 内部，如果此时销毁 Ray 会导致段错误。
        通过 join() 等待监控线程安全退出后再清理 Ray 资源。

        特殊情况:
        当监控线程自身调用 shutdown() 时（即 worker 故障触发的关闭），
        跳过 join，因为该线程即将自行返回。
        """
        monitor = getattr(self, "_monitor_thread", None)
        if (
            monitor is not None
            and monitor.is_alive()
            and threading.current_thread() is not monitor
        ):
            monitor.join(timeout=10)

    def shutdown(self) -> None:
        """正确关闭执行器及其所有 worker。

        关闭流程:
        1. 获取 shutdown_lock，防止并发关闭
        2. 标记 shutting_down = True，阻止重复关闭
        3. 等待监控线程退出（_join_monitor_thread）
        4. 终止所有 Ray Actor（ray.kill）
        5. 关闭广播 MessageQueue
        6. 关闭所有响应 MessageQueue

        线程安全:
            使用 shutdown_lock 保证 shutdown() 可被多个线程安全调用
            （例如监控线程和主线程同时调用的情况）。
        """
        lock = getattr(self, "shutdown_lock", None)
        if lock is None:
            return

        with lock:
            if getattr(self, "shutting_down", False):
                return
            self.shutting_down = True

        self._join_monitor_thread()

        # 终止所有 Ray Actor
        for handle in getattr(self, "ray_worker_handles", []):
            try:
                ray.kill(handle.actor)
                logger.debug("Killed actor rank=%d", handle.rank)
            except Exception:
                logger.exception("Failed to kill actor rank=%d", handle.rank)

        # 关闭广播 MessageQueue
        if rpc_broadcast_mq := getattr(self, "rpc_broadcast_mq", None):
            rpc_broadcast_mq.shutdown()
            self.rpc_broadcast_mq = None

        # 关闭所有响应 MessageQueue
        for mq in getattr(self, "response_mqs", []):
            mq.shutdown()
        self.response_mqs = []
