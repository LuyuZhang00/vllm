# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# ===========================================================================
# 中文注释：本模块实现了基于 Ray 框架的分布式执行器（RayDistributedExecutor）。
#
# 【模块概述】
#   Ray 是一个用于构建分布式应用的开源框架。本模块利用 Ray 的 Actor 模型
#   和 Compiled DAG（编译有向无环图）能力，将模型推理任务分发到多个 GPU
#   Worker 上并行执行。
#
# 【与 MultiprocExecutor 的主要区别】
#   1. 进程管理方式不同：
#      - MultiprocExecutor 使用 Python 原生 multiprocessing 模块创建子进程，
#        子进程的生命周期由 Executor 直接管理。
#      - RayDistributedExecutor 使用 Ray Actor 创建远程 Worker，Worker 的
#        生命周期由 Ray 集群管理，天然支持跨节点分布式部署。
#
#   2. 通信机制不同：
#      - MultiprocExecutor 使用 Python multiprocessing.Queue 进行进程间通信。
#      - RayDistributedExecutor 使用 Ray 的远程方法调用（remote call）和
#        Compiled DAG 进行通信，支持 NCCL 或共享内存通道。
#
#   3. 调度策略不同：
#      - MultiprocExecutor 通常在同一台机器的多个 GPU 上运行。
#      - RayDistributedExecutor 通过 Placement Group 支持跨多台机器的 GPU
#        调度，适合大规模分布式推理场景。
#
#   4. Pipeline Parallelism 支持：
#      - RayDistributedExecutor 原生支持流水线并行（Pipeline Parallelism），
#        通过 Compiled DAG 将不同 PP 阶段串联，实现高效的跨阶段数据传输。
#      - MultiprocExecutor 也支持 PP，但通信效率可能不如 Ray Compiled DAG。
#
# 【核心执行流程】
#   1. _init_executor()           —— 初始化 Ray 集群连接、创建 Worker Actor
#   2. _init_workers_ray()        —— 创建所有 Ray Worker，设置 GPU 绑定和网络
#   3. execute_model()            —— 接收 SchedulerOutput，触发模型执行
#   4. _execute_dag()             —— 通过 Compiled DAG 执行模型前向推理
#   5. _compiled_ray_dag()        —— 构建并编译 Ray DAG，定义 PP/TP 执行拓扑
#   6. collective_rpc()           —— 向所有 Worker 广播执行指定方法
# ===========================================================================

import os
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import cloudpickle

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.ray.ray_env import get_env_vars_to_copy
from vllm.utils.network_utils import (
    get_distributed_init_method,
    get_ip,
    get_open_port,
)
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.engine import ReconfigureDistributedRequest, ReconfigureRankType
from vllm.v1.executor.abstract import Executor
from vllm.v1.executor.ray_utils import (
    WORKER_SPECIFIC_ENV_VARS,
    FutureWrapper,
    RayWorkerWrapper,
    detach_zero_copy_from_model_runner_output,
    initialize_ray_cluster,
    ray,
)
from vllm.v1.outputs import ModelRunnerOutput

# 中文注释：条件导入 Ray 的类型，仅在 Ray 可用时导入 ActorHandle 和调度策略
# ActorHandle：Ray Actor 的句柄，用于远程调用 Actor 方法
# PlacementGroupSchedulingStrategy：基于 Placement Group 的调度策略，
#   确保 Worker 被调度到指定的资源组中
if ray is not None:
    from ray.actor import ActorHandle
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
else:
    ActorHandle = None

if TYPE_CHECKING:
    from ray.util.placement_group import PlacementGroup

logger = init_logger(__name__)

# 中文注释：预创建的已完成 Future 对象，结果为 None。
# 当模型不需要实际执行时（例如没有需要调度的 token），直接返回此 Future，
# 避免每次都创建新的 Future 对象，是一种性能优化手段。
COMPLETED_NONE_FUTURE: Future[ModelRunnerOutput | None] = Future()
COMPLETED_NONE_FUTURE.set_result(None)


@dataclass
class RayWorkerMetaData:
    """
    Metadata for a Ray worker.
    The order of ray worker creation can be random,
    and we need to reset the rank after creating all workers.
    """

    # 中文注释：Ray Worker 的元数据类，用于记录每个 Worker 的关键信息。
    #
    # 【字段说明】
    #   worker:        Ray Actor 句柄，用于远程调用该 Worker 的方法
    #   created_rank:  Worker 创建时的初始 rank（顺序可能与最终 rank 不同）
    #   adjusted_rank: 经过重新排序后的最终 rank（初始化为 -1 表示尚未分配）
    #   ip:            Worker 所在节点的 IP 地址
    #
    # 【为什么要重新排序？】
    #   Ray Worker 的创建顺序是不确定的（可能跨多个节点），但分布式训练
    #   需要一个确定的 rank 顺序（例如 driver 节点的 Worker 排在前面）。
    #   因此需要在创建所有 Worker 后，根据 IP 地址等信息重新分配 rank。

    worker: ActorHandle
    created_rank: int
    adjusted_rank: int = -1
    ip: str = ""


# ===========================================================================
# 中文注释：RayDistributedExecutor 类
#
# 【类职责】
#   RayDistributedExecutor 是 Executor 抽象基类的具体实现之一，
#   基于 Ray 框架实现分布式模型推理。它管理一组 Ray Actor Worker，
#   通过 Ray Compiled DAG 实现高效的多 GPU/多节点模型执行。
#
# 【关键类属性】
#   uses_ray = True        —— 标识此执行器使用 Ray（与 MultiprocExecutor 区分）
#   supports_pp = True     —— 标识此执行器支持 Pipeline Parallelism
#
# 【核心数据结构】
#   workers:       所有 Ray Worker Actor 的列表（按调整后的 rank 排序）
#   pp_tp_workers: 二维列表，按 [PP rank][TP rank] 索引 Worker，
#                  用于构建 Pipeline Parallelism 的执行 DAG
#   forward_dag:   编译后的 Ray Compiled DAG，用于高效执行模型前向推理
# ===========================================================================
class RayDistributedExecutor(Executor):
    """Ray-based distributed executor"""

    uses_ray: bool = True
    supports_pp: bool = True

    # ===================================================================
    # 中文注释：_init_executor() —— 执行器初始化入口
    #
    # 【执行流程】
    #   1. 初始化 Ray 集群连接
    #   2. 创建所有 Ray Worker Actor
    #   3. 初始化 Worker（设置分布式环境、加载模型等）
    #   4. 构建 PP/TP Worker 拓扑结构
    #
    # 【注意】
    #   对于 TPU/XPU 平台，需要将 DAG 通道类型设为共享内存（shm），
    #   因为这些平台不支持 NVIDIA NCCL。
    # ===================================================================
    def _init_executor(self) -> None:
        self.forward_dag: ray.dag.CompiledDAG | None = None

        # For TPU or XPU, avoid compiling NVIDIA's NCCL
        # 中文注释：对于 TPU 或 XPU 平台，设置 DAG 通道类型为共享内存，
        # 因为这些平台没有 NVIDIA GPU，无法使用 NCCL 进行通信。
        if current_platform.is_tpu() or current_platform.is_xpu():
            os.environ["VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE"] = "shm"

        assert self.uses_ray
        # 中文注释：初始化 Ray 集群，确保当前进程已连接到 Ray 集群，
        # 并获取 Placement Group（GPU 资源组）信息。
        initialize_ray_cluster(self.parallel_config)
        placement_group = self.parallel_config.placement_group

        # Disable Ray usage stats collection.
        # 中文注释：禁用 Ray 的使用统计收集，避免不必要的网络请求。
        ray_usage = os.environ.get("RAY_USAGE_STATS_ENABLED", "0")
        if ray_usage != "1":
            os.environ["RAY_USAGE_STATS_ENABLED"] = "0"

        # Create the parallel GPU workers.
        # 中文注释：创建所有并行 GPU Worker（核心初始化步骤）。
        self._init_workers_ray(placement_group)

        # KV connector setup
        # 中文注释：检查是否配置了 KV 缓存传输连接器（用于分布式 KV 缓存共享）。
        self.has_connector = self.vllm_config.kv_transfer_config is not None

        # 中文注释：判断是否使用采样器（sampler）。
        # 以下两种情况不使用采样器：
        #   1. 模型类型为 pooling（如嵌入模型），不产生 token
        #   2. 配置了 EC 传输且当前节点不是 EC 消费者
        self.uses_sampler = self.vllm_config.model_config.runner_type != "pooling" and (
            self.vllm_config.ec_transfer_config is None
            or self.vllm_config.ec_transfer_config.is_ec_consumer
        )

        # 中文注释：缓存最新的 SchedulerOutput，用于延迟执行模型。
        # 在 execute_model() 中暂存，在 sample_tokens() 中使用。
        self.scheduler_output: SchedulerOutput | None = None

    # ===================================================================
    # 中文注释：max_concurrent_batches 属性
    #
    # 【功能】返回此执行器支持的最大并发批次数。
    #
    # 【逻辑】
    #   - 当 PP（流水线并行）> 1 时，返回 PP 的大小。这是因为流水线并行
    #     可以同时处理多个 micro-batch，每个 PP 阶段处理一个 batch。
    #   - 当 PP <= 1 且使用异步调度时，返回 2。这意味着可以预取下一个
    #     batch，在当前 batch 执行的同时准备下一个 batch。
    #   - 否则返回 1（同步执行，一次只处理一个 batch）。
    # ===================================================================
    @property
    def max_concurrent_batches(self) -> int:
        """Ray distributed executor supports pipeline parallelism,
        meaning that it allows PP size batches to be executed concurrently.
        """
        pp_size = self.parallel_config.pipeline_parallel_size
        return 2 if pp_size <= 1 and self.scheduler_config.async_scheduling else pp_size

    # ===================================================================
    # 中文注释：shutdown() —— 关闭执行器，清理 Ray 资源
    #
    # 【执行流程】
    #   1. 如果存在编译好的 DAG，先调用 teardown() 释放 DAG 资源
    #   2. 使用 ray.kill() 终止所有 Worker Actor
    #   3. 将 forward_dag 置为 None
    #
    # 【注意】
    #   关闭过程中可能会看到来自 Ray logging.cc 的 SIGTERM 错误日志，
    #   这是正常的终止流程，可以忽略。
    # ===================================================================
    def shutdown(self) -> None:
        if logger:
            # Somehow logger can be None here.
            logger.info(
                "Shutting down Ray distributed executor. If you see error log "
                "from logging.cc regarding SIGTERM received, please ignore "
                "because this is the expected termination process in Ray."
            )
        if hasattr(self, "forward_dag") and self.forward_dag is not None:
            self.forward_dag.teardown()
            import ray

            for worker in self.workers:
                ray.kill(worker)
            self.forward_dag = None

    # ===================================================================
    # 中文注释：配置 Nsight 性能分析
    #
    # 【功能】当启用 Ray Nsight 分析时，为 Worker 配置 CUDA profiling 参数。
    # Nsight 是 NVIDIA 的性能分析工具，用于分析 CUDA 内核的执行情况。
    #
    # 【配置参数说明】
    #   t: "cuda,cudnn,cublas"   —— 分析的目标库
    #   o: "'worker_process_%p'" —— 输出文件名格式（%p 为进程 ID）
    #   cuda-graph-trace: "node" —— CUDA Graph 追踪级别
    # ===================================================================
    def _configure_ray_workers_use_nsight(self, ray_remote_kwargs) -> dict[str, Any]:
        # If nsight profiling is enabled, we need to set the profiling
        # configuration for the ray workers as runtime env.
        runtime_env = ray_remote_kwargs.setdefault("runtime_env", {})
        runtime_env.update(
            {
                "nsight": {
                    "t": "cuda,cudnn,cublas",
                    "o": "'worker_process_%p'",
                    "cuda-graph-trace": "node",
                }
            }
        )

        return ray_remote_kwargs

    # ===================================================================
    # 中文注释：设置平台特定的 "不设置设备" 环境变量
    #
    # 【功能】某些平台（如 XPU）需要设置特定的环境变量来告诉 Ray
    # 不要自动设置设备（CUDA_VISIBLE_DEVICES 等），而是由 vLLM 内部
    # 通过 local_rank 来索引正确的 GPU。
    # ===================================================================
    def _update_noset_device_env_vars(self, ray_remote_kwargs):
        runtime_env = ray_remote_kwargs.setdefault("runtime_env", {})
        env_vars = runtime_env.setdefault("env_vars", {})
        env_vars.update(
            {env_var: "1" for env_var in current_platform.ray_noset_device_env_vars}
        )
        return ray_remote_kwargs

    # child class could overwrite this to return actual env vars.
    # 中文注释：获取需要传递给所有 Worker 的环境变量。
    # 子类可以覆盖此方法以返回不同的环境变量集合。
    def _get_env_vars_to_be_updated(self):
        return self._env_vars_for_all_workers

    # ===================================================================
    # 中文注释：_init_workers_ray() —— 创建并初始化所有 Ray Worker
    #
    # 这是 Ray 执行器初始化的核心方法，执行以下步骤：
    #
    # 【步骤 1：确定 bundle 索引】
    #   bundle 是 Placement Group 中的资源单元（通常对应一个 GPU）。
    #   可以通过环境变量 VLLM_RAY_BUNDLE_INDICES 手动指定使用哪些 bundle，
    #   也可以自动选择前 N 个包含 GPU 的 bundle。
    #
    # 【步骤 2：创建 Ray Worker Actor】
    #   对于每个 bundle，创建一个 RayWorkerWrapper Actor。
    #   Actor 是 Ray 的核心概念，每个 Actor 是一个有状态的远程进程。
    #   - 对于 GPU 平台，使用 num_gpus 参数分配 GPU 资源
    #   - 对于其他平台，使用 resources 参数分配自定义资源
    #
    # 【步骤 3：收集 Worker IP 并重新排序】
    #   按以下优先级对 Worker 进行排序：
    #   1. 与 driver（引擎进程）在同一节点的 Worker 排在最前面
    #   2. 同一节点上 Worker 数量较少的排在前面（负载均衡）
    #   3. IP 地址较小的排在前面（确定性排序）
    #
    # 【步骤 4：收集节点和 GPU 信息】
    #   获取每个 Worker 所在的节点 ID 和 GPU ID，用于后续设置
    #   CUDA_VISIBLE_DEVICES 等环境变量。
    #
    # 【步骤 5：设置环境变量】
    #   为每个 Worker 设置 CUDA_VISIBLE_DEVICES（该节点上所有 GPU 的 ID），
    #   以便 Worker 能通过 local_rank 索引到正确的 GPU。
    #   同时复制 driver 进程的环境变量到 Worker。
    #
    # 【步骤 6：初始化 Worker】
    #   调用 init_worker、init_device、load_model 等方法完成 Worker 初始化。
    #
    # 【步骤 7：构建 PP/TP 拓扑】
    #   将 Worker 按 [PP rank][TP rank] 组织成二维列表，用于后续
    #   构建 Pipeline Parallelism 的执行 DAG。
    # ===================================================================
    def _init_workers_ray(self, placement_group: "PlacementGroup", **ray_remote_kwargs):
        # 中文注释：每个 Worker 分配的 GPU 数量，可通过环境变量配置。
        num_gpus = envs.VLLM_RAY_PER_WORKER_GPUS

        # The driver dummy worker does not actually use any resources.
        # It holds the resource for the driver worker.
        # 中文注释：driver 的虚拟 Worker，不实际占用资源，
        # 仅用于在分布式初始化中表示 driver 节点。
        self.driver_dummy_worker: RayWorkerWrapper | None = None
        # The remaining workers are the actual ray actors.
        # 中文注释：实际执行推理任务的 Ray Worker Actor 列表。
        self.workers: list[RayWorkerWrapper] = []

        # Used in ray compiled DAG: indexed first by PP rank,
        # and then TP rank. In other words, the inner list is
        # the TP group of workers for a PP rank.
        # 中文注释：按 PP rank 和 TP rank 组织的 Worker 二维列表。
        # 外层索引为 PP rank，内层列表为该 PP 阶段的 TP 组 Worker。
        # 例如 PP=2, TP=4 时：[[w0,w1,w2,w3], [w4,w5,w6,w7]]
        self.pp_tp_workers: list[list[RayWorkerWrapper]] = []

        # 中文注释：如果启用了 Nsight 性能分析，配置相应的 Ray 运行时参数。
        if self.parallel_config.ray_workers_use_nsight:
            ray_remote_kwargs = self._configure_ray_workers_use_nsight(
                ray_remote_kwargs
            )

        # The way ray actors are setup in vllm is that the visible devices are
        # not set by actors, they are left unset by ray. Internally we index
        # the right gpu with local_rank. This is similar to how mp mode works.
        # 中文注释：设置平台特定的环境变量，告诉 Ray 不要自动设置设备。
        # vLLM 内部通过 local_rank 来索引正确的 GPU。
        self._update_noset_device_env_vars(ray_remote_kwargs)

        # Create the workers.
        # 中文注释：步骤 1 —— 确定要使用的 bundle 索引。
        bundle_indices: list[int]
        if envs.VLLM_RAY_BUNDLE_INDICES:
            # Use the bundle indices specified by the user.
            # 中文注释：使用用户通过环境变量指定的 bundle 索引。
            bundle_indices = list(map(int, envs.VLLM_RAY_BUNDLE_INDICES.split(",")))
            assert len(bundle_indices) == self.parallel_config.world_size, (
                "VLLM_RAY_BUNDLE_INDICES must have the same size"
                f" as the world size, but got {bundle_indices=} "
                f"and {self.parallel_config.world_size=}"
            )
            assert len(set(bundle_indices)) == len(bundle_indices), (
                "VLLM_RAY_BUNDLE_INDICES cannot have duplicate values,"
                f" but got {bundle_indices=}"
            )
        else:
            # use the first N bundles that have GPU resources.
            # 中文注释：自动选择 Placement Group 中前 N 个包含 GPU 资源的 bundle。
            bundle_indices = []
            for bundle_id, bundle in enumerate(placement_group.bundle_specs):
                if bundle.get(current_platform.ray_device_key, 0):
                    bundle_indices.append(bundle_id)
            bundle_indices = bundle_indices[: self.parallel_config.world_size]

        # 中文注释：步骤 2 —— 为每个 bundle 创建 Ray Worker Actor。
        worker_metadata: list[RayWorkerMetaData] = []
        driver_ip = get_ip()
        for rank, bundle_id in enumerate(bundle_indices):
            # 中文注释：设置调度策略，确保 Actor 被调度到指定的 Placement Group bundle 上。
            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=placement_group,
                placement_group_capture_child_tasks=True,
                placement_group_bundle_index=bundle_id,
            )

            if current_platform.ray_device_key == "GPU":
                # NV+AMD GPUs, and Intel XPUs
                # 中文注释：对于 GPU 平台（NVIDIA/AMD/Intel），使用 num_gpus 参数。
                worker = ray.remote(
                    num_cpus=0,
                    num_gpus=num_gpus,
                    scheduling_strategy=scheduling_strategy,
                    **ray_remote_kwargs,
                )(RayWorkerWrapper).remote(rpc_rank=rank)
            else:
                # 中文注释：对于非 GPU 平台（如 TPU），使用自定义 resources 参数。
                worker = ray.remote(
                    num_cpus=0,
                    num_gpus=0,
                    resources={current_platform.ray_device_key: num_gpus},
                    scheduling_strategy=scheduling_strategy,
                    **ray_remote_kwargs,
                )(RayWorkerWrapper).remote(rpc_rank=rank)

            worker_metadata.append(RayWorkerMetaData(worker=worker, created_rank=rank))

        # 中文注释：步骤 3 —— 收集每个 Worker 的 IP 地址，用于后续排序。
        worker_ips = ray.get(
            [
                each.worker.get_node_ip.remote()  # type: ignore[attr-defined]
                for each in worker_metadata
            ]
        )

        for each, ip in zip(worker_metadata, worker_ips):
            each.ip = ip

        logger.debug("workers: %s", worker_metadata)
        logger.debug("driver_dummy_worker: %s", self.driver_dummy_worker)

        # 中文注释：统计每个 IP 地址上的 Worker 数量，用于排序时的负载均衡。
        ip_counts: dict[str, int] = {}
        for ip in worker_ips:
            ip_counts[ip] = ip_counts.get(ip, 0) + 1

        def sort_by_driver_then_worker_ip(item: RayWorkerMetaData):
            """
            Sort the workers based on 3 properties:
            1. If the worker is on the same node as the driver (vllm engine),
                it should be placed first.
            2. Then, if the worker is on a node with fewer workers, it should
                be placed first.
            3. Finally, if the work is on a node with smaller IP address, it
                should be placed first.
            """
            # 中文注释：排序规则：
            # 1. 与 driver 同节点的 Worker 排在最前面（返回 0 < 1）
            # 2. 同节点 Worker 数量少的排在前面（有利于负载均衡）
            # 3. IP 地址小的排在前面（确定性排序）
            ip = item.ip
            return 0 if ip == driver_ip else 1, ip_counts[ip], ip

        # After sorting, the workers on the same node will be
        # close to each other, and the workers on the driver
        # node will be placed first.
        # 中文注释：对 Worker 按上述规则排序，并分配最终的 adjusted_rank。
        sorted_worker_metadata = sorted(
            worker_metadata, key=sort_by_driver_then_worker_ip
        )
        for i, item in enumerate(sorted_worker_metadata):
            item.adjusted_rank = i
        self.workers = [item.worker for item in sorted_worker_metadata]
        # 中文注释：将重新排序的映射关系（created_rank -> adjusted_rank）通知所有 Worker，
        # 让它们更新自己的 rank。
        rerank_mapping = {
            item.created_rank: item.adjusted_rank for item in sorted_worker_metadata
        }
        self.collective_rpc("adjust_rank", args=(rerank_mapping,))

        # Get the set of GPU IDs used on each node.
        # 中文注释：步骤 4 —— 收集每个 Worker 所在节点的 ID 和 GPU ID。
        worker_node_and_gpu_ids = []
        for worker in [self.driver_dummy_worker] + self.workers:
            if worker is None:
                # driver_dummy_worker can be None when using ray spmd worker.
                continue
            worker_node_and_gpu_ids.append(
                ray.get(worker.get_node_and_gpu_ids.remote())  # type: ignore[attr-defined]
            )

        # 中文注释：按节点分组，记录每个节点上的 Worker rank 和 GPU ID。
        node_workers = defaultdict(list)  # node id -> list of worker ranks
        node_gpus = defaultdict(list)  # node id -> list of gpu ids

        for i, (node_id, gpu_ids) in enumerate(worker_node_and_gpu_ids):
            node_workers[node_id].append(i)
            # `gpu_ids` can be a list of strings or integers.
            # convert them to integers for consistency.
            # NOTE: gpu_ids can be larger than 9 (e.g. 16 GPUs),
            # string sorting is not sufficient.
            # see https://github.com/vllm-project/vllm/issues/5590
            gpu_ids = [int(x) for x in gpu_ids]
            node_gpus[node_id].extend(gpu_ids)
        for node_id, gpu_ids in node_gpus.items():
            node_gpus[node_id] = sorted(gpu_ids)

        # 中文注释：验证每个节点都有唯一的 IP 地址，避免网络配置错误。
        all_ips = set(worker_ips + [driver_ip])
        n_ips = len(all_ips)
        n_nodes = len(node_workers)

        if n_nodes != n_ips:
            raise RuntimeError(
                f"Every node should have a unique IP address. Got {n_nodes}"
                f" nodes with node ids {list(node_workers.keys())} and "
                f"{n_ips} unique IP addresses {all_ips}. Please check your"
                " network configuration. If you set `VLLM_HOST_IP`"
                " environment variable, make sure it is unique for"
                " each node."
            )

        # Set environment variables for the driver and workers.
        # We set CUDA_VISIBLE_DEVICES to ALL GPUs on the node for each worker.
        # This is needed because:
        # 1. Ray's compiled DAG needs to find the allocated GPU in
        #    CUDA_VISIBLE_DEVICES.
        # 2. vLLM's communication layer (NCCL, CustomAllreduce) needs to see
        #    all GPUs for P2P checks and communication setup. Though if it was
        #    just this reason, we could have also just kept the visible devices
        #    unset.
        # Each worker will use local_rank to index into the visible devices.
        # 中文注释：步骤 5 —— 为每个 Worker 设置环境变量。
        # 将 CUDA_VISIBLE_DEVICES 设置为该节点上所有 GPU 的 ID。
        # 原因有两个：
        #   1. Ray Compiled DAG 需要在 CUDA_VISIBLE_DEVICES 中找到分配的 GPU
        #   2. vLLM 的通信层（NCCL、CustomAllreduce）需要看到所有 GPU
        #      以进行 P2P 检查和通信设置
        # 每个 Worker 内部通过 local_rank 来索引正确的 GPU。
        all_args_to_update_environment_variables = [
            {
                current_platform.device_control_env_var: ",".join(
                    map(str, node_gpus[node_id])
                ),
            }
            for (node_id, _) in worker_node_and_gpu_ids
        ]

        # Environment variables to copy from driver to workers
        # 中文注释：从 driver 进程复制环境变量到 Worker，
        # 排除 Worker 特有的环境变量（如 CUDA_VISIBLE_DEVICES，已单独设置）。
        env_vars_to_copy = get_env_vars_to_copy(
            exclude_vars=WORKER_SPECIFIC_ENV_VARS,
            additional_vars=set(current_platform.additional_env_vars),
            destination="workers",
        )

        # Copy existing env vars to each worker's args
        # 中文注释：将需要复制的环境变量添加到每个 Worker 的参数中。
        for args in all_args_to_update_environment_variables:
            # TODO: refactor platform-specific env vars
            for name in env_vars_to_copy:
                if name in os.environ:
                    args[name] = os.environ[name]

        self._env_vars_for_all_workers = all_args_to_update_environment_variables

        # 中文注释：通过 collective_rpc 将环境变量广播到所有 Worker。
        self.collective_rpc(
            "update_environment_variables", args=(self._get_env_vars_to_be_updated(),)
        )

        # 中文注释：确定分布式初始化方法（初始化地址）。
        # 单节点情况下使用回环地址 127.0.0.1，避免网络接口问题。
        # 多节点情况下使用 driver 的 IP 地址。
        if len(node_gpus) == 1:
            # in single node case, we don't need to get the IP address.
            # the loopback address is sufficient
            # NOTE: a node may have several IP addresses, one for each
            # network interface. `get_ip()` might return any of them,
            # while they might not work for communication inside the node
            # if the network setup is complicated. Using the loopback address
            # solves this issue, as it always works for communication inside
            # the node.
            driver_ip = "127.0.0.1"
        distributed_init_method = get_distributed_init_method(
            driver_ip, get_open_port()
        )

        # Initialize the actual workers inside worker wrapper.
        # 中文注释：步骤 6 —— 初始化每个 Worker。
        # 为每个 Worker 准备初始化参数，包括：
        #   - vllm_config: 全局配置
        #   - local_rank: 本节点内的 GPU 序号
        #   - rank: 全局 rank
        #   - distributed_init_method: 分布式初始化地址（如 tcp://ip:port）
        #   - is_driver_worker: 是否为 driver Worker（每 TP 组的第一个）
        all_kwargs = []
        for rank, (node_id, _) in enumerate(worker_node_and_gpu_ids):
            local_rank = node_workers[node_id].index(rank)
            kwargs = dict(
                vllm_config=self.vllm_config,
                local_rank=local_rank,
                rank=rank,
                distributed_init_method=distributed_init_method,
                is_driver_worker=(not self.parallel_config)
                or (rank % self.parallel_config.tensor_parallel_size == 0),
            )
            all_kwargs.append(kwargs)
        # 中文注释：依次调用 init_worker（初始化 Worker 对象）、
        # init_device（初始化 GPU 设备）、load_model（加载模型权重）。
        self.collective_rpc("init_worker", args=(all_kwargs,))

        self.collective_rpc("init_device")
        if envs.VLLM_ELASTIC_EP_SCALE_UP_LAUNCH:
            self.collective_rpc("elastic_ep_execute", args=("load_model",))
        else:
            self.collective_rpc("load_model")

        # 中文注释：更新每个 Worker 的 block size 配置，以适配特定的后端。
        def _update_block_size(worker):
            current_platform.update_block_size_for_backend(worker.vllm_config)

        self.collective_rpc(_update_block_size)

        # 中文注释：步骤 7 —— 构建 PP/TP Worker 拓扑结构。
        # 将 Worker 按 [PP rank][TP rank] 组织成二维列表。
        # 例如 PP=2, TP=4 时：
        #   pp_tp_workers = [[w0, w1, w2, w3], [w4, w5, w6, w7]]
        # 其中 [w0,w1,w2,w3] 是 PP 阶段 0 的 TP 组，
        #      [w4,w5,w6,w7] 是 PP 阶段 1 的 TP 组。
        for pp_rank in range(self.parallel_config.pipeline_parallel_size):
            self.pp_tp_workers.append([])
            for tp_rank in range(self.parallel_config.tensor_parallel_size):
                # PP=2, TP=4
                # pp_tp_workers = [[0, 1, 2, 3], [4, 5, 6, 7]]
                rank = (pp_rank * self.parallel_config.tensor_parallel_size) + tp_rank
                assert len(self.pp_tp_workers[pp_rank]) == tp_rank
                assert pp_rank < len(self.pp_tp_workers)
                self.pp_tp_workers[pp_rank].append(self.workers[rank])

    # ===================================================================
    # 中文注释：reinitialize_distributed() —— 重新初始化分布式配置
    #
    # 【功能】在运行时重新配置分布式设置（例如弹性 EP 扩缩容场景）。
    # 将重配置请求广播到所有 Worker。如果当前 rank 需要关闭（缩容），
    # 则调用 shutdown() 关闭执行器。
    # ===================================================================
    def reinitialize_distributed(
        self, reconfig_request: ReconfigureDistributedRequest
    ) -> None:
        self.collective_rpc("reinitialize_distributed", args=(reconfig_request,))
        if (
            reconfig_request.new_data_parallel_rank
            == ReconfigureRankType.SHUTDOWN_CURRENT_RANK
        ):
            self.shutdown()

    # ===================================================================
    # 中文注释：execute_model() —— 执行模型前向推理
    #
    # 【功能】接收 Scheduler 的输出（SchedulerOutput），触发模型执行。
    #
    # 【执行逻辑】
    #   1. 如果上次 execute_model() 返回 None（表示延迟执行），
    #      但调用方未调用 sample_tokens() 就再次调用 execute_model()，
    #      则抛出状态错误。
    #
    #   2. 如果不使用采样器或没有需要调度的 token，直接执行 DAG 并返回结果。
    #      这种情况下不需要等待 grammar_output（语法输出）。
    #
    #   3. 否则，将 SchedulerOutput 暂存，返回 None（或 COMPLETED_NONE_FUTURE），
    #      等待后续调用 sample_tokens() 时再实际执行。
    #
    # 【为什么需要延迟执行？】
    #   在 vLLM V1 中，execute_model() 和 sample_tokens() 是分离的。
    #   execute_model() 先确定要执行的批次，然后异步处理 grammar_output
    #   （用于结构化输出），最后在 sample_tokens() 中实际执行模型并采样。
    #   这种分离设计允许在两者之间进行其他异步操作。
    # ===================================================================
    def execute_model(  # type: ignore[override]
        self,
        scheduler_output: SchedulerOutput,
        non_block: bool = False,
    ) -> ModelRunnerOutput | None | Future[ModelRunnerOutput | None]:
        if self.scheduler_output is not None:
            raise RuntimeError(
                "State error: sample_tokens() must be called "
                "after execute_model() returns None."
            )

        if not self.uses_sampler or not scheduler_output.total_num_scheduled_tokens:
            # Model will not execute, call model runner immediately.
            # 中文注释：模型不会执行（无 token 需要处理或不使用采样器），
            # 直接执行 DAG。
            return self._execute_dag(scheduler_output, None, non_block)

        # Model will execute, defer to sample_tokens() call.
        # 中文注释：模型将执行，暂存 SchedulerOutput，延迟到 sample_tokens() 再执行。
        self.scheduler_output = scheduler_output
        return COMPLETED_NONE_FUTURE if non_block else None

    # ===================================================================
    # 中文注释：sample_tokens() —— 执行模型并采样生成 token
    #
    # 【功能】使用之前在 execute_model() 中暂存的 SchedulerOutput
    # 实际执行模型前向推理，并结合 grammar_output 进行采样。
    #
    # 【参数】
    #   grammar_output: 结构化输出的语法 bitmask（用于 constrained decoding）
    #   non_block: 是否非阻塞，如果为 True 则返回 Future 而非阻塞等待
    #
    # 【执行逻辑】
    #   1. 获取暂存的 SchedulerOutput
    #   2. 清空暂存（设为 None）
    #   3. 调用 _execute_dag() 执行模型推理
    # ===================================================================
    def sample_tokens(  # type: ignore[override]
        self,
        grammar_output: "GrammarOutput | None",
        non_block: bool = False,
    ) -> ModelRunnerOutput | None | Future[ModelRunnerOutput | None]:
        """Execute the model on the Ray workers.

        The scheduler output to use should have been provided in
        a prior call to execute_model().

        Args:
            grammar_output: The structured outputs grammar bitmask, if applicable.
            non_block: If True, the method will return a Future.

        Returns:
            The model runner output.
        """
        scheduler_output = self.scheduler_output
        if scheduler_output is None:
            return COMPLETED_NONE_FUTURE if non_block else None

        self.scheduler_output = None

        return self._execute_dag(scheduler_output, grammar_output, non_block)

    # ===================================================================
    # 中文注释：_execute_dag() —— 通过 Ray Compiled DAG 执行模型推理
    #
    # 【功能】这是实际执行模型前向推理的核心方法。
    #
    # 【执行流程】
    #   1. 如果 Compiled DAG 尚未构建，先调用 _compiled_ray_dag() 构建。
    #   2. 将 SchedulerOutput 和 GrammarOutput 作为输入，执行 DAG。
    #   3. 根据是否有 KV connector 和是否阻塞，决定返回方式：
    #      a. 无 connector + 阻塞：直接从 output_rank Worker 获取结果
    #      b. 无 connector + 非阻塞：返回 FutureWrapper
    #      c. 有 connector + 阻塞：从所有 Worker 获取结果并聚合
    #      d. 有 connector + 非阻塞：返回聚合后的 FutureWrapper
    #
    # 【关于 Compiled DAG】
    #   Ray Compiled DAG 是 Ray 的一种高效执行机制，它将 DAG 中的操作
    #   编译为优化的执行计划，避免了普通 Ray remote call 的调度开销。
    #   对于 PP 场景，中间张量通过 NCCL 或共享内存直接传输，无需序列化。
    #
    # 【关于 detach_zero_copy】
    #   由于 Ray Compiled DAG 使用零拷贝（zero-copy）传输数据，输出对象
    #   的内存可能被后续操作覆盖。因此需要调用 detach_zero_copy 将数据
    #   复制到独立的内存空间，确保数据的生命周期独立于 DAG 执行。
    # ===================================================================
    def _execute_dag(
        self,
        scheduler_output: SchedulerOutput,
        grammar_output: "GrammarOutput | None",
        non_block: bool = False,
    ) -> ModelRunnerOutput | None | Future[ModelRunnerOutput | None]:
        # Build the compiled DAG for the first time.
        # 中文注释：首次执行时构建 Compiled DAG（懒初始化）。
        if self.forward_dag is None:  # type: ignore
            self.forward_dag = self._compiled_ray_dag(enable_asyncio=False)

        # 中文注释：执行 DAG，传入 SchedulerOutput 和 GrammarOutput 作为输入。
        refs = self.forward_dag.execute((scheduler_output, grammar_output))  # type: ignore

        if not self.has_connector:
            # Get output only from a single worker (output_rank)
            # When PP is not used, we block here until the result is available.
            # 中文注释：无 KV connector 时，只需从一个 Worker（output_rank）获取结果。
            if not non_block:
                # 中文注释：阻塞模式 —— 等待结果返回，然后执行零拷贝分离。
                output = refs[0].get()
                detach_zero_copy_from_model_runner_output(output)
                return output

            # When PP is used, we return a FutureWrapper immediately so that
            # the scheduler can yield to the next batch.
            # 中文注释：非阻塞模式 —— 返回 FutureWrapper，调度器可以继续处理其他批次。
            return FutureWrapper(refs[0])

        # Get output from all workers when connector is present
        # 中文注释：有 KV connector 时，需要从所有 Worker 获取结果并聚合。
        assert self.kv_output_aggregator is not None
        if not non_block:
            # Block and get results from all workers
            # 中文注释：阻塞模式 —— 从所有 Worker 获取结果，执行零拷贝分离，然后聚合。
            outputs = ray.get(refs)
            for output in outputs:
                detach_zero_copy_from_model_runner_output(output)
            return self.kv_output_aggregator.aggregate(outputs)

        # Return a future that will aggregate outputs from all workers
        # 中文注释：非阻塞模式 —— 返回一个 Future，会在所有结果就绪后自动聚合。
        return FutureWrapper(refs, self.kv_output_aggregator)

    # ===================================================================
    # 中文注释：collective_rpc() —— 集体远程过程调用
    #
    # 【功能】向所有 Worker 广播执行指定的方法。这是 Executor 的核心通信机制。
    #
    # 【参数】
    #   method: 要调用的方法名（字符串）或可调用对象（会通过 cloudpickle 序列化）
    #   timeout: 等待结果的超时时间（秒），None 表示无限等待
    #   args: 方法的位置参数元组
    #   kwargs: 方法的关键字参数字典
    #   non_block: 是否非阻塞，如果为 True 则返回 FutureWrapper
    #
    # 【执行流程】
    #   1. 如果 method 是可调用对象，使用 cloudpickle 序列化为字节
    #   2. 向每个 Worker 发送 execute_method 远程调用
    #   3. 根据 non_block 决定是阻塞等待结果还是返回 Future
    #
    # 【与 MultiprocExecutor 的区别】
    #   MultiprocExecutor 使用 multiprocessing.Queue 发送消息，
    #   而 RayDistributedExecutor 使用 Ray 的远程方法调用（.remote()），
    #   后者天然支持跨节点通信和自动重试。
    # ===================================================================
    def collective_rpc(  # type: ignore[override]
        self,
        method: str | Callable,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
        non_block: bool = False,
    ) -> list[Any] | Future[list[Any]]:
        """Runs the given method on all workers."""
        # 中文注释：如果 method 是可调用对象，使用 cloudpickle 序列化。
        # cloudpickle 可以序列化 lambda、闭包等普通 pickle 无法处理的对象。
        sent_method = method if isinstance(method, str) else cloudpickle.dumps(method)
        del method

        if kwargs is None:
            kwargs = {}
        # 中文注释：向每个 Worker 发送远程方法调用。
        # execute_method 是 RayWorkerWrapper 上的方法，负责在 Worker 内部
        # 解析方法名或反序列化可调用对象，然后执行。
        ray_worker_outputs = [
            worker.execute_method.remote(  # type: ignore[attr-defined]
                sent_method, *args, **kwargs
            )
            for worker in self.workers
        ]

        # Get the results of the ray workers.
        # 中文注释：根据 non_block 决定返回方式。
        if non_block:
            return FutureWrapper(ray_worker_outputs)

        return ray.get(ray_worker_outputs, timeout=timeout)

    # ===================================================================
    # 中文注释：_check_ray_cgraph_installation() —— 检查 Ray Compiled Graph 安装
    #
    # 【功能】验证 Ray 版本和 Compiled Graph 扩展是否正确安装。
    #
    # 【检查项】
    #   1. Ray 版本 >= 2.43.0（Compiled Graph 的最低版本要求）
    #   2. ray.experimental.compiled_dag_ref 模块是否存在
    #   3. 如果使用 NCCL 通道，检查 cupy 是否安装（NCCL 通信需要 cupy）
    # ===================================================================
    def _check_ray_cgraph_installation(self):
        import importlib.metadata

        from packaging import version

        required_version = version.parse("2.43.0")
        current_version = version.parse(importlib.metadata.version("ray"))
        if current_version < required_version:
            raise ValueError(
                f"Ray version {required_version} is "
                f"required, but found {current_version}"
            )

        import importlib.util

        cgraph_spec = importlib.util.find_spec("ray.experimental.compiled_dag_ref")
        if cgraph_spec is None:
            raise ValueError(
                "Ray Compiled Graph is not installed. "
                "Run `pip install ray[cgraph]` to install it."
            )

        cupy_spec = importlib.util.find_spec("cupy")
        if cupy_spec is None and envs.VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE == "nccl":
            raise ValueError(
                "cupy is not installed but required since "
                "VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE is set to 'nccl'. "
                "Run `pip install ray[cgraph]` and check cupy installation."
            )

    # ===================================================================
    # 中文注释：_compiled_ray_dag() —— 构建并编译 Ray Compiled DAG
    #
    # 【功能】构建模型前向推理的执行 DAG，并编译为高效的执行计划。
    #
    # 【DAG 结构说明】
    #   以 PP=2, TP=4 为例：
    #
    #   SchedulerOutput -> Worker0 (PP0,TP0) -> 中间张量 -> Worker4 (PP1,TP0) -> ModelRunnerOutput
    #   SchedulerOutput -> Worker1 (PP0,TP1) -> 中间张量 -> Worker5 (PP1,TP1) -> ModelRunnerOutput
    #   SchedulerOutput -> Worker2 (PP0,TP2) -> 中间张量 -> Worker6 (PP1,TP2) -> ModelRunnerOutput
    #   SchedulerOutput -> Worker3 (PP0,TP3) -> 中间张量 -> Worker7 (PP1,TP3) -> ModelRunnerOutput
    #
    #   - 第一个 PP 阶段的所有 Worker 接收相同的 SchedulerOutput 输入
    #   - 每个 PP 阶段的 Worker 以 SPMD（单程序多数据）方式执行
    #   - PP 阶段之间通过中间张量（IntermediateTensors）传递数据
    #   - 最后一个 PP 阶段的 Worker 输出 ModelRunnerOutput
    #
    # 【通道类型】
    #   - "shm"（共享内存）：同一节点内的 Worker 通过共享内存传输数据，默认选项
    #   - "nccl"（NVIDIA NCCL）：使用 NCCL 进行 GPU 间通信，适合跨节点场景
    #   - "auto"：自动选择
    #
    # 【编译参数】
    #   - enable_asyncio: 是否启用异步 IO
    #   - _overlap_gpu_communication: 是否将 GPU 通信与计算重叠（流水线优化）
    #   - RAY_CGRAPH_get_timeout: DAG 执行的超时时间，默认 300 秒
    # ===================================================================
    def _compiled_ray_dag(self, enable_asyncio: bool):
        assert self.parallel_config.use_ray
        self._check_ray_cgraph_installation()
        # Enlarge the default value of "RAY_CGRAPH_get_timeout" to 300 seconds
        # (it is 10 seconds by default). This is a Ray environment variable to
        # control the timeout of getting result from a compiled graph execution,
        # i.e., the distributed execution that includes model forward runs and
        # intermediate tensor communications, in the case of vllm.
        # Note: we should set this env var before importing
        # ray.dag, otherwise it will not take effect.
        # 中文注释：设置 DAG 执行超时时间为 300 秒（默认 10 秒太短，
        # 因为模型推理和张量通信可能耗时较长）。
        # 必须在导入 ray.dag 之前设置，否则不生效。
        os.environ.setdefault("RAY_CGRAPH_get_timeout", "300")  # noqa: SIM112
        from ray.dag import InputNode, MultiOutputNode

        logger.info(
            "RAY_CGRAPH_get_timeout is set to %s",
            os.environ["RAY_CGRAPH_get_timeout"],  # noqa: SIM112
        )
        logger.info(
            "VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE = %s",
            envs.VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE,
        )
        logger.info(
            "VLLM_USE_RAY_COMPILED_DAG_OVERLAP_COMM = %s",
            envs.VLLM_USE_RAY_COMPILED_DAG_OVERLAP_COMM,
        )

        # 中文注释：验证通道类型配置是否合法。
        channel_type = envs.VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE
        if channel_type not in ("auto", "nccl", "shm"):
            raise ValueError(
                "Invalid value for VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE: "
                f"{channel_type}. Valid values are: 'auto', 'nccl', or 'shm'."
            )

        # 中文注释：构建 DAG。
        # InputNode 是 DAG 的输入节点，代表传入的 (SchedulerOutput, GrammarOutput)。
        with InputNode() as input_data:
            # Example DAG: PP=2, TP=4
            #
            # SchedulerOutput -> 0 -> (SchedulerOutput, IntermediateTensors) -> 4 -> ModelRunnerOutput   # noqa: E501
            # SchedulerOutput -> 1 -> (SchedulerOutput, IntermediateTensors) -> 5 -> ModelRunnerOutput   # noqa: E501
            # SchedulerOutput -> 2 -> (SchedulerOutput, IntermediateTensors) -> 6 -> ModelRunnerOutput   # noqa: E501
            # SchedulerOutput -> 3 -> (SchedulerOutput, IntermediateTensors) -> 7 -> ModelRunnerOutput   # noqa: E501

            # All workers in the first TP group will take in the
            # ExecuteModelRequest as input.
            # 中文注释：第一个 PP 阶段的所有 Worker 接收相同的输入。
            outputs = [input_data for _ in self.pp_tp_workers[0]]
            for pp_rank, tp_group in enumerate(self.pp_tp_workers):
                # Each PP worker takes in the output of the previous PP worker,
                # and the TP group executes in SPMD fashion.
                # 中文注释：每个 PP 阶段的 Worker 接收上一阶段的输出。
                # 同一 PP 阶段内的 TP Worker 以 SPMD 方式执行（相同代码，不同数据切片）。
                outputs = [
                    worker.execute_model_ray.bind(outputs[i])  # type: ignore[attr-defined]
                    for i, worker in enumerate(tp_group)
                ]

                last_pp_rank = len(self.pp_tp_workers) - 1
                if (
                    pp_rank < last_pp_rank
                    and envs.VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE != "shm"
                ):
                    # Specify how intermediate tensors should be passed
                    # between pp stages, no need to specify for the last
                    # pp stage or when using shared memory (the default).
                    # 中文注释：为中间 PP 阶段的输出指定张量传输方式。
                    # 最后一个 PP 阶段不需要指定（输出是 ModelRunnerOutput），
                    # 使用共享内存时也不需要指定（默认行为）。
                    transport = envs.VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE
                    outputs = [
                        output.with_tensor_transport(transport=transport)
                        for output in outputs
                    ]

            # 中文注释：MultiOutputNode 将多个输出合并为 DAG 的最终输出。
            forward_dag = MultiOutputNode(outputs)

        # 中文注释：如果启用了 Ray 包装的 PP 通信器，注册加速器上下文。
        # RayPPCommunicator 包装了 vLLM 的 _PP GroupCoordinator，
        # 使 Ray Compiled DAG 能够使用 vLLM 的通信层。
        if envs.VLLM_USE_RAY_WRAPPED_PP_COMM:
            from ray.experimental.channel.accelerator_context import (
                register_accelerator_context,
            )

            from vllm.distributed.device_communicators.ray_communicator import (
                RayPPCommunicator,
            )

            register_accelerator_context(
                torch_module_name="cuda", communicator_cls=RayPPCommunicator
            )
            logger.info(
                "Using RayPPCommunicator "
                "(which wraps vLLM _PP GroupCoordinator) "
                "for Ray Compiled Graph communication."
            )
        else:
            logger.info(
                "Using Ray's NCCL communicator for Ray Compiled Graph communication."
            )

        # 中文注释：编译 DAG 为高效的执行计划。
        # enable_asyncio: 是否支持异步执行
        # _overlap_gpu_communication: 是否将通信与计算重叠（提升吞吐量）
        return forward_dag.experimental_compile(
            enable_asyncio=enable_asyncio,
            _overlap_gpu_communication=envs.VLLM_USE_RAY_COMPILED_DAG_OVERLAP_COMM,
        )

    # ===================================================================
    # 中文注释：析构函数 —— 确保执行器被垃圾回收时清理 Ray 资源。
    # ===================================================================
    def __del__(self):
        self.shutdown()

    # ===================================================================
    # 中文注释：check_health() —— 健康检查
    #
    # 【功能】检查 Ray Worker 是否健康。
    # 当前实现假设 Ray Worker 始终健康（直接返回）。
    # TODO: 实现真正的 Worker 健康检查。
    # ===================================================================
    def check_health(self) -> None:
        # Assume that the Ray workers are healthy.
        # TODO: check the health of the Ray workers
        return
