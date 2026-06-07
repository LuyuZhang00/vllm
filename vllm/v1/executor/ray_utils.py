# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Ray 分布式执行工具模块 (Ray Distributed Execution Utilities)
==============================================================

本模块为 vLLM v1 引擎提供 Ray 分布式框架的核心支撑功能，主要包括以下内容：

1. **RayWorkerWrapper** -- Ray Worker 包装类
   - 继承自 WorkerWrapperBase，将 vLLM Worker 封装为 Ray Actor
   - 支持延迟初始化（Ray 设置 CUDA_VISIBLE_DEVICES 后才真正初始化 Worker）
   - 提供模型执行、设备管理、环境变量覆盖等方法
   - 支持 Ray Compiled Graph 的执行模式

2. **FutureWrapper** -- 异步结果包装器
   - 将 Ray ObjectRef 封装为标准 Python Future 接口
   - 支持单 Worker 和多 Worker 聚合两种模式
   - 处理 Ray 共享内存零拷贝缓冲区的分离

3. **Ray 集群初始化与 Placement Group 管理**
   - `initialize_ray_cluster()` -- 初始化 Ray 集群并创建 Placement Group
   - `_wait_until_pg_ready()` / `_wait_until_pg_removed()` -- PG 生命周期等待
   - `_verify_bundles()` -- 验证 PG 的资源分配是否满足需求

4. **Worker 资源查询工具**
   - `get_bundles_for_indices()` -- 根据显式索引获取 GPU Bundle 信息
   - `get_bundles_sorted_by_node()` -- 按节点排序获取 Bundle（Driver 节点优先）
   - `get_num_tpu_nodes()` / `get_num_nodes_in_placement_group()` -- 节点数量查询

5. **零拷贝缓冲区处理**
   - `detach_zero_copy_from_model_runner_output()` -- 分离 Ray SHM 零拷贝缓冲区
   - 防止跨调度迭代时保留 Ray 共享内存引用导致通道阻塞

关键设计要点：
- 通过 Placement Group 确保 GPU 资源在集群中的合理分配
- 支持多节点张量并行（TP）和流水线并行（PP）
- 使用指数退避策略处理资源等待超时
- 处理不同 Ray 版本的 API 兼容性问题
"""

import os
import time
from collections import defaultdict
from concurrent.futures import Future
from typing import TYPE_CHECKING, Union

import numpy as np

import vllm.platforms
from vllm.config import ParallelConfig
from vllm.distributed import get_pp_group
from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.utils.network_utils import get_ip
from vllm.v1.outputs import AsyncModelRunnerOutput
from vllm.v1.serial_utils import run_method
from vllm.v1.worker.worker_base import WorkerWrapperBase

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
    from vllm.v1.outputs import ModelRunnerOutput

logger = init_logger(__name__)

# Placement Group 等待超时时间（秒），默认 30 分钟
# 当 Ray 集群资源不足时，会在此时间内持续等待并输出日志
PG_WAIT_TIMEOUT = 1800

# Env vars that are worker-specific and must NOT be copied from the
# driver to Ray workers — they are set per-worker after GPU discovery.
# 工作进程专属的环境变量集合，这些变量不能从 Driver 进程复制到 Ray Worker。
# 原因：每个 Worker 在 GPU 发现阶段会独立设置这些变量，以确保每个 Worker
# 绑定到正确的 GPU 设备和网络端口。如果从 Driver 复制，会导致设备冲突。
WORKER_SPECIFIC_ENV_VARS: set[str] = {
    "VLLM_HOST_IP",               # Worker 的主机 IP 地址
    "VLLM_HOST_PORT",             # Worker 的主机端口
    "VLLM_NIXL_SIDE_CHANNEL_HOST", # NIXL 侧通道主机地址（用于 KV 传输）
    "LOCAL_RANK",                 # 本地 GPU 排序编号
    "CUDA_VISIBLE_DEVICES",       # CUDA 可见设备列表
    "HIP_VISIBLE_DEVICES",        # AMD HIP 可见设备列表
    "ROCR_VISIBLE_DEVICES",       # AMD ROCR 可见设备列表
}

try:
    import ray
    from ray.util import placement_group_table
    from ray.util.placement_group import PlacementGroup

    # Ray 2.9.x 版本不直接暴露 available_resources_per_node 函数，
    # 需要通过内部状态对象间接获取。这是 Ray 版本兼容性的处理。
    try:
        from ray._private.state import available_resources_per_node
    except ImportError:
        # Ray 2.9.x doesn't expose `available_resources_per_node`
        from ray._private.state import state as _state

        available_resources_per_node = _state._available_resources_per_node

    class RayWorkerWrapper(WorkerWrapperBase):
        """
        Ray Worker 包装类，将 vLLM 的 Worker 封装为 Ray Actor。

        设计目的：
        1. 支持延迟初始化 -- Ray 在分配 GPU 后会设置 CUDA_VISIBLE_DEVICES，
           Worker 需要在此之后才进行真正的初始化（加载模型等）
        2. 提供与 Ray Compiled Graph 的兼容性
        3. 处理跨进程方法调用和异常传播

        关键属性：
        - rpc_rank: Worker 的 RPC 通信编号，用于分布式通信中的标识
        - compiled_dag_cuda_device_set: 标记 CUDA 设备是否已在 Compiled DAG
          的后台线程中设置
        """

        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            # Since the compiled DAG runs a main execution
            # in a different thread that calls cuda.set_device.
            # The flag indicates is set_device is called on
            # that thread.
            # Ray Compiled Graph 在后台线程执行，该线程会调用 cuda.set_device。
            # 此标志记录是否已在该线程中设置了 CUDA 设备，避免重复设置。
            self.compiled_dag_cuda_device_set = False

        rpc_rank: int

        def adjust_rank(self, rank_mapping: dict[int, int]) -> None:
            """
            Adjust the rpc_rank based on the given mapping.
            It is only used during the initialization of the executor,
            to adjust the rpc_rank of workers after we create all workers.

            根据给定的映射关系调整 Worker 的 RPC 编号。

            使用场景：
            - 仅在 Executor 初始化阶段使用
            - 当所有 Worker 创建完成后，可能需要重新编号以匹配实际的
              分布式拓扑结构（如 TP/PP 分组）

            参数：
                rank_mapping: {旧编号 -> 新编号} 的映射字典
            """
            if self.rpc_rank in rank_mapping:
                self.rpc_rank = rank_mapping[self.rpc_rank]

        def execute_method(self, method: str | bytes, *args, **kwargs):
            """
            执行远程方法调用，是 Ray Actor 方法的核心入口。

            流程：
            1. 调用 run_method() 反序列化并执行指定的方法
            2. 如果发生异常，记录日志并重新抛出

            注意：如果 Driver Worker 也执行方法，其他 Worker 的异常可能导致
            RPC 死锁（参见 GitHub issue #3455）。因此需要捕获异常并记录详细信息。
            """
            try:
                return run_method(self, method, args, kwargs)
            except Exception as e:
                # if the driver worker also execute methods,
                # exceptions in the rest worker may cause deadlock in rpc
                # see https://github.com/vllm-project/vllm/issues/3455
                msg = (
                    f"Error executing method {method!r}. "
                    "This might cause deadlock in distributed execution."
                )
                logger.exception(msg)
                raise e

        def get_node_ip(self) -> str:
            """
            获取当前 Worker 所在节点的 IP 地址。

            用于分布式通信中建立节点间的网络连接。
            """
            return get_ip()

        def get_node_and_gpu_ids(self) -> tuple[str, list[int]]:
            """
            获取当前 Worker 的节点 ID 和 GPU 设备 ID 列表。

            返回：
                (node_id, gpu_ids) -- Ray 节点 ID 和 GPU ID 列表

            用途：
            - 用于 Executor 建立 Worker 到物理设备的映射
            - 用于确定哪些 Worker 在同一节点（用于优化通信）

            设备键（device_key）根据平台不同可能是 "GPU"、"NPU" 等。
            """
            node_id = ray.get_runtime_context().get_node_id()
            device_key = vllm.platforms.current_platform.ray_device_key
            if not device_key:
                raise RuntimeError(
                    "current platform %s does not support ray.",
                    vllm.platforms.current_platform.device_name,
                )
            gpu_ids = ray.get_runtime_context().get_accelerator_ids()[device_key]
            return node_id, gpu_ids

        def setup_device_if_necessary(self):
            """
            在执行模型前确保 CUDA 设备已正确设置。

            背景：
            Ray Compiled Graph (CG) 在后台线程中执行，而 PyTorch 的当前设备
            是线程局部的。因此需要在 CG 线程中重新设置 CUDA 设备，
            否则模型会在错误的 GPU 上执行。

            流程：
            1. 检查 Worker 是否已初始化
            2. 如果是 TPU 平台则跳过（不需要设置设备）
            3. 否则调用平台特定的 set_device 方法
            4. 标记设备已设置，避免重复操作
            """
            # TODO(swang): This is needed right now because Ray CG executes
            # on a background thread, so we need to reset torch's current
            # device.
            # We can remove this API after it is fixed in compiled graph.
            assert self.worker is not None, "Worker is not initialized"
            if not self.compiled_dag_cuda_device_set:
                if current_platform.is_tpu():
                    # Not needed
                    # TPU 平台不需要显式设置设备
                    pass
                else:
                    assert self.worker.device is not None
                    current_platform.set_device(self.worker.device)

                self.compiled_dag_cuda_device_set = True

        def execute_model_ray(
            self,
            execute_model_input: tuple["SchedulerOutput", "GrammarOutput"]
            | tuple["SchedulerOutput", "GrammarOutput", "IntermediateTensors"],
        ) -> Union[
            "ModelRunnerOutput",
            tuple["SchedulerOutput", "GrammarOutput", "IntermediateTensors"],
        ]:
            """
            Ray Compiled Graph 专用的模型执行入口。

            与普通 execute_method 不同，此方法专门用于 Ray CG 场景，
            包含以下特殊处理：
            1. 调用 setup_device_if_necessary() 确保 CUDA 设备正确
            2. 支持流水线并行（PP）的中间张量传递
            3. 处理异步模型输出的同步化

            参数：
                execute_model_input: 包含调度输出和语法输出的元组，
                    可选包含中间张量（PP 场景）

            返回：
                - 如果是中间 PP 阶段：返回 (scheduler_output, grammar_output,
                  intermediate_tensors)
                - 如果是最后 PP 阶段：返回 ModelRunnerOutput

            流程：
            1. 设置 CUDA 设备
            2. 解析输入参数（区分有/无中间张量）
            3. 执行模型前向传播
            4. 判断输出类型：
               a. 中间张量 -> 剥离多模态特征后传递给下一 PP 阶段
               b. 异步输出 -> 同步获取结果
               c. 非最后阶段且无请求 -> 返回空中间张量
               d. 最后阶段 -> 执行 token 采样
            """
            # This method is used by Ray Compiled Graph to execute the model,
            # and it needs a special logic of self.setup_device_if_necessary()
            self.setup_device_if_necessary()
            assert self.worker is not None, "Worker is not initialized"
            if len(execute_model_input) == 3:
                # PP 场景：包含中间张量
                scheduler_output, grammar_output, intermediate_tensors = (
                    execute_model_input
                )
            else:
                # 非 PP 场景或第一 PP 阶段：无中间张量
                scheduler_output, grammar_output = execute_model_input
                intermediate_tensors = None
            assert self.worker.model_runner is not None
            # 执行模型前向传播
            output = self.worker.model_runner.execute_model(
                scheduler_output, intermediate_tensors
            )
            if self._is_intermediate_tensors(output):
                # 输出是中间张量，说明当前不是最后的 PP 阶段
                if (
                    self.worker.model_runner.supports_mm_inputs
                    and get_pp_group().is_first_rank
                ):
                    # Strip mm_features before Ray forwards it to the next PP Stage.
                    # PP Stage>0 only needs the intermediate tensors,
                    # not preprocessed multimodal data.
                    # 第一 PP 阶段需要在传递中间张量前剥离多模态特征。
                    # 后续 PP 阶段只需要中间张量，不需要预处理后的多模态数据。
                    # 这样可以减少跨节点传输的数据量。

                    # scheduled_new_reqs is a required field of SchedulerOutput,
                    # so accessing it directly will raise AttributeError if missing.
                    for req in scheduler_output.scheduled_new_reqs:
                        req.mm_features = []
                return scheduler_output, grammar_output, output

            # 处理异步模型输出（包含 CUDA 事件，需要同步等待）
            if isinstance(output, AsyncModelRunnerOutput):
                output = output.get_output()
            if not self._is_last_rank():
                # Case where there are no scheduled requests
                # but may still be finished requests.
                # 非最后 PP 阶段：没有调度的请求，但仍可能有完成的请求。
                # 返回包含 None 中间张量的元组，以保持管道一致性。
                assert not output or not output.req_ids
                output = scheduler_output, grammar_output, None
            elif output is None:
                # 最后 PP 阶段且无输出：执行 token 采样
                output = self.worker.model_runner.sample_tokens(grammar_output)
                # Ensure outputs crossing Ray compiled DAG are serializable.
                # AsyncModelRunnerOutput holds CUDA events and cannot be
                # pickled.
                # 确保跨 Ray Compiled DAG 传递的输出是可序列化的。
                # AsyncModelRunnerOutput 持有 CUDA 事件对象，无法被 pickle 序列化，
                # 因此需要提前获取同步结果。
                if isinstance(output, AsyncModelRunnerOutput):
                    output = output.get_output()
            return output

        def override_env_vars(self, vars: dict[str, str]):
            """
            覆盖当前进程的环境变量。

            用途：Driver 可以向 Worker 发送环境变量更新，
            例如动态调整日志级别或配置参数。
            """
            os.environ.update(vars)

        def _is_intermediate_tensors(self, output) -> bool:
            """判断输出是否为中间张量（PP 中间阶段的输出）。"""
            return isinstance(output, IntermediateTensors)

        def _is_last_rank(self) -> bool:
            """判断当前 Worker 是否为流水线并行的最后一个阶段。"""
            return get_pp_group().is_last_rank

    # Ray 成功导入
    ray_import_err = None

except ImportError as e:
    # Ray 未安装，设置为 None 并记录错误信息。
    # 仅捕获错误字符串（而非异常对象），避免 traceback 中的变量引用
    # 阻止垃圾回收。
    ray = None  # type: ignore
    # only capture string to avoid variable references in the traceback that can
    # prevent garbage collection in some cases
    ray_import_err = str(e)
    RayWorkerWrapper = None  # type: ignore


def detach_zero_copy_from_model_runner_output(output: "ModelRunnerOutput") -> None:
    """
    将 Ray SHM 通道的零拷贝缓冲区从 ModelRunnerOutput 中就地分离。

    问题背景：
    Ray Compiled DAG 的共享内存（SHM）通道可能返回零拷贝对象（如 np.ndarray），
    这些对象底层使用 Ray 的共享内存对象存储。Ray 文档明确警告：
    如果此类对象仍在作用域内，后续读取可能会阻塞。

    vLLM 的 ModelRunnerOutput.logprobs 中可能包含基于 numpy 的 logprobs 数组。
    如果这些数组底层是 Ray SHM（通常是只读的），在调度迭代之间保留这些引用
    会导致通道阻塞，最终触发 RAY_CGRAPH_get_timeout 超时错误。

    解决方案：
    复制只读的 numpy 数组，使返回的输出不再持有对 Ray 共享内存缓冲区的引用。

    注意事项：
    - 不处理 prompt_logprobs_dict：那些条目是 LogprobsTensors，底层是
      PyTorch 管理的 CPU 张量（to_cpu_nonblocking 或 empty_cpu），
      不是从 Ray 通道解码的 NumPy 视图。
    - cu_num_generated_tokens 已经是普通 Python 列表（或 None），
      不会关联 Ray SHM 缓冲区，可直接复用。

    参数：
        output: 模型运行器的输出对象，会被就地修改
    """
    if output.logprobs is None:
        return

    token_ids, logprobs, ranks, cu_num_generated_tokens = output.logprobs

    def _copy_if_readonly(arr):
        """如果 numpy 数组是只读的，返回其副本；否则返回原数组。"""
        if isinstance(arr, np.ndarray) and not arr.flags.writeable:
            return arr.copy()
        return arr

    # `cu_num_generated_tokens` is already a plain Python list (or None), so it
    # never aliases Ray SHM buffers and can be reused as-is.
    # cu_num_generated_tokens 已经是普通 Python 列表，不会关联 Ray SHM
    token_ids_c = _copy_if_readonly(token_ids)
    logprobs_c = _copy_if_readonly(logprobs)
    ranks_c = _copy_if_readonly(ranks)
    # 如果没有任何数组被复制，直接返回（优化：避免不必要的元组重建）
    if token_ids_c is token_ids and logprobs_c is logprobs and ranks_c is ranks:
        return

    # 用复制后的数组重建 logprobs 元组
    output.logprobs = type(output.logprobs)(
        token_ids_c, logprobs_c, ranks_c, cu_num_generated_tokens
    )


class FutureWrapper(Future):
    """
    Ray 输出引用的 Future 包装器。

    设计目的：
    vLLM 的核心调度循环（core busy loop）期望 execute_model() 返回一个
    标准的 Future 对象，通过 .result() 方法阻塞等待并获取单个输出。
    但 Ray 返回的是 ObjectRef（或 ObjectRef 列表），不直接兼容。
    本类桥接了这个接口差异。

    两种使用模式：
    1. 单 Worker 模式（aggregator=None）：
       - ref_or_refs 是单个 Ray ObjectRef
       - result() 直接返回该 Worker 的输出

    2. 多 Worker 聚合模式（aggregator 提供）：
       - ref_or_refs 是 ObjectRef 列表（每个 Worker 一个）
       - result() 等待所有 Worker 完成后，使用 aggregator 聚合结果
       - 用于 KV 缓存传输场景（分布式 KV cache 聚合）

    关键处理：
    - 调用 detach_zero_copy_from_model_runner_output() 确保输出不持有
      Ray SHM 缓冲区引用，避免后续调度阻塞
    """

    def __init__(self, ref_or_refs, aggregator: KVOutputAggregator | None = None):
        super().__init__()
        self.ref_or_refs = ref_or_refs  # Ray ObjectRef 或 ObjectRef 列表
        self.aggregator = aggregator    # 输出聚合器（可选）

    def result(self, timeout=None):
        """
        阻塞等待并返回模型执行结果。

        参数：
            timeout: 超时时间（秒），None 表示无限等待

        返回：
            单 Worker 模式：直接返回 ModelRunnerOutput
            多 Worker 模式：返回聚合后的输出

        流程：
        1. 调用 ray.get() 阻塞等待 Ray ObjectRef 解析
        2. 分离零拷贝缓冲区（防止 SHM 通道阻塞）
        3. 如果有聚合器，聚合所有 Worker 的输出
        """
        outputs = ray.get(self.ref_or_refs, timeout=timeout)
        if self.aggregator is None:
            # 单 Worker 模式：直接分离并返回
            detach_zero_copy_from_model_runner_output(outputs)
            return outputs

        # 多 Worker 模式：分离所有输出的缓冲区后聚合
        for output in outputs:
            detach_zero_copy_from_model_runner_output(output)
        return self.aggregator.aggregate(outputs, output_rank=0)


def ray_is_available() -> bool:
    """检查 Ray 是否可用（是否成功导入）。"""
    return ray is not None


def assert_ray_available():
    """断言 Ray 可用，否则抛出异常并提示安装命令。"""
    if ray is None:
        raise ValueError(
            f"Failed to import Ray: {ray_import_err}."
            "Please install Ray with `pip install ray`."
        )


def _verify_bundles(
    placement_group: "PlacementGroup",
    parallel_config: ParallelConfig,
    device_str: str,
    require_gpu_on_driver: bool = True,
):
    """
    验证 Placement Group 的 Bundle 分配是否满足要求。

    验证规则：
    1. 警告规则：如果单节点的 GPU 数量不足以容纳所有张量并行（TP）Worker，
       则发出警告。TP Worker 跨节点会导致性能下降（除非有高速互联如 InfiniBand）。
    2. 强制规则：如果 require_gpu_on_driver=True，Driver 节点必须包含在
       Placement Group 中。Driver 本身也是一个 Worker，需要 GPU。

    参数：
        placement_group: Ray Placement Group 对象
        parallel_config: 并行配置（包含 TP/PP 大小等）
        device_str: 设备类型字符串（如 "GPU"、"NPU"）
        require_gpu_on_driver: 是否要求 Driver 节点有 GPU
    """
    assert ray.is_initialized(), (
        "Ray is not initialized although distributed-executor-backend is ray."
    )
    pg_data = placement_group_table(placement_group)
    # bundle_idx -> node_id: 每个 Bundle 分配到了哪个节点
    bundle_to_node_ids = pg_data["bundles_to_node_id"]
    # bundle_idx -> bundle (e.g., {"GPU": 1}): 每个 Bundle 的资源规格
    bundles = pg_data["bundles"]
    # node_id -> List of bundle: 按节点聚合 Bundle 列表
    node_id_to_bundle: dict[str, list[dict[str, float]]] = defaultdict(list)

    for bundle_idx, node_id in bundle_to_node_ids.items():
        node_id_to_bundle[node_id].append(bundles[bundle_idx])
    driver_node_id = ray.get_runtime_context().get_node_id()

    # 规则 2：验证 Driver 节点是否包含在 PG 中
    if require_gpu_on_driver and driver_node_id not in node_id_to_bundle:
        raise RuntimeError(
            f"driver node id {driver_node_id} is not included in a placement "
            f"group {placement_group.id}. Node id -> bundles "
            f"{node_id_to_bundle}. "
            "You don't have enough GPUs available in a current node. Check "
            "`ray status` and `ray list nodes` to see if you have available "
            "GPUs in a node `{driver_node_id}` before starting an vLLM engine."
        )

    # 规则 1：检查每个节点的 GPU 是否足以容纳 TP Worker
    for node_id, bundles in node_id_to_bundle.items():
        if len(bundles) < parallel_config.tensor_parallel_size:
            logger.warning(
                "tensor_parallel_size=%d "
                "is bigger than a reserved number of %ss (%d "
                "%ss) in a node %s. Tensor parallel workers can be "
                "spread out to 2+ nodes which can degrade the performance "
                "unless you have fast interconnect across nodes, like "
                "Infiniband. To resolve this issue, make sure you have more "
                "than %d GPUs available at each node.",
                parallel_config.tensor_parallel_size,
                device_str,
                len(bundles),
                device_str,
                node_id,
                parallel_config.tensor_parallel_size,
            )


def build_actor_name(
    instance_id: str,
    rank: int,
    tp_size: int,
    pp_size: int,
    pcp_size: int,
) -> str:
    """
    构建具有描述性的 Ray Actor 名称，便于在 Ray Dashboard 中识别和调试。

    名称格式：vllm_Worker_{instance_id}[_TP{x}][_PP{y}][_PCP{z}]

    参数：
        instance_id: vLLM 引擎实例的唯一标识符
        rank: Worker 的全局编号
        tp_size: 张量并行大小
        pp_size: 流水线并行大小
        pcp_size: 专家缓存并行大小（用于 MoE 模型）

    示例：
        - 单 GPU: "vllm_Worker_abc123"
        - TP=4: "vllm_Worker_abc123_TP0", "vllm_Worker_abc123_TP1", ...
        - TP=4, PP=2: "vllm_Worker_abc123_TP0_PP0", "vllm_Worker_abc123_TP0_PP1", ...

    编号计算逻辑：
    - TP 编号 = rank % tp_size（在 TP 组内的位置）
    - PP 编号 = (rank // tp_size) % pp_size（在 PP 组内的位置）
    - PCP 编号 = rank // (tp_size * pp_size)（在 PCP 组内的位置）
    """
    name = f"vllm_Worker_{instance_id}"
    if tp_size > 1:
        name += f"_TP{rank % tp_size}"
    if pp_size > 1:
        name += f"_PP{(rank // tp_size) % pp_size}"
    if pcp_size > 1:
        name += f"_PCP{rank // (tp_size * pp_size)}"
    return name


def get_bundles_for_indices(
    placement_group: "PlacementGroup",
    bundle_indices: list[int],
    world_size: int,
) -> list[tuple[int, str, str]]:
    """
    根据显式指定的 Bundle 索引列表，获取对应的 GPU Bundle 信息。

    用途：
    通过环境变量 VLLM_RAY_BUNDLE_INDICES 可以显式指定使用哪些 Bundle，
    用于精确控制 Worker 到 GPU 的映射关系。

    参数：
        placement_group: Ray Placement Group 对象
        bundle_indices: 要使用的 Bundle 索引列表
        world_size: 世界大小（总 Worker 数），必须与 bundle_indices 长度一致

    返回：
        列表，每个元素为 (bundle_idx, node_id, node_ip) 三元组

    约束：
    - bundle_indices 长度必须等于 world_size
    - bundle_indices 中不能有重复值
    """
    assert len(bundle_indices) == world_size, (
        "VLLM_RAY_BUNDLE_INDICES must have the same size"
        f" as the world size, but got {bundle_indices=} "
        f"and {world_size=}"
    )
    assert len(set(bundle_indices)) == len(bundle_indices), (
        "VLLM_RAY_BUNDLE_INDICES cannot have duplicate values,"
        f" but got {bundle_indices=}"
    )

    pg_data = placement_group_table(placement_group)
    pg_bundle_to_node = pg_data["bundles_to_node_id"]
    # 构建 node_id -> node_ip 的映射（仅包含存活节点）
    node_id_to_ip = {
        n["NodeID"]: n["NodeManagerAddress"] for n in ray.nodes() if n["Alive"]
    }
    return [
        (bid, pg_bundle_to_node[bid], node_id_to_ip[pg_bundle_to_node[bid]])
        for bid in bundle_indices
    ]


def get_bundles_sorted_by_node(
    placement_group: "PlacementGroup",
) -> list[tuple[int, str, str]]:
    """
    获取 Placement Group 中所有 GPU Bundle 的信息，并按节点排序（Driver 节点优先）。

    排序策略：
    1. Driver 节点的 Bundle 排在最前面
    2. 其他节点按 node_id 字典序排序

    这种排序确保 Driver 节点的 Worker 优先启动，因为 Driver 通常也是
    调度器和协调器所在节点。

    注意：此函数必须从 Driver 节点调用（因为需要获取 runtime_context）。

    参数：
        placement_group: Ray Placement Group 对象

    返回：
        列表，每个元素为 (bundle_idx, node_id, node_ip) 三元组，
        按 Driver 节点优先排序

    示例（3 节点集群，Driver 在 node-A）：
      输入 Bundle 分布: [(0,node-C), (1,node-A), (2,node-B),
                          (3,node-C), (4,node-A), (5,node-B)]
      排序后输出:        [(1,node-A), (4,node-A), (2,node-B),
                          (5,node-B), (0,node-C), (3,node-C)]
    """
    pg_data = placement_group_table(placement_group)
    bundle_to_node = pg_data["bundles_to_node_id"]

    # 获取当前平台的 Ray 设备键（如 "GPU"、"NPU"）
    ray_device_key = current_platform.ray_device_key
    if not ray_device_key:
        raise ValueError(
            f"current platform {current_platform.device_name} does not support ray."
        )

    # 构建 node_id -> node_ip 的映射（仅包含存活节点）
    node_id_to_ip = {
        n["NodeID"]: n["NodeManagerAddress"] for n in ray.nodes() if n["Alive"]
    }

    bundle_specs = placement_group.bundle_specs
    assert bundle_specs is not None
    # 筛选出包含目标设备的 Bundle，并关联节点信息
    bundle_to_node_id: list[tuple[int, str, str]] = []
    for bundle_idx, bundle in enumerate(bundle_specs):
        if bundle.get(ray_device_key):
            node_id = bundle_to_node.get(bundle_idx)
            bundle_to_node_id.append((bundle_idx, node_id, node_id_to_ip[node_id]))

    # 获取 Driver 节点 ID，用于排序时优先排在前面
    driver_node = ray.get_runtime_context().get_node_id()

    def _sort_key(item):
        """
        排序键：Driver 节点排第一（键值 0），其他节点排第二（键值 1），
        同优先级内按 node_id 字典序排序。
        """
        _, node_id, _ = item
        return (0 if node_id == driver_node else 1, node_id)

    bundle_to_node_id.sort(key=_sort_key)

    return bundle_to_node_id


def _wait_until_pg_ready(current_placement_group: "PlacementGroup"):
    """
    等待 Placement Group 准备就绪。

    此函数会阻塞等待 Ray 分配所需的全部资源（GPU 等），直到：
    1. 所有 Bundle 的资源都可用 -> 正常返回
    2. 超过 PG_WAIT_TIMEOUT 时间 -> 抛出超时异常

    等待策略：
    - 使用指数退避（exponential backoff）打印日志信息
    - 初始等待间隔 10 秒，每次翻倍（10, 20, 40, 80, ...）
    - 这样可以在资源长时间不可用时减少日志刷屏

    错误处理：
    - 多 GPU 请求超时：提示可能是 TP 大小超过集群 GPU 数量
    - 单 GPU 请求超时：提示检查集群资源状态

    参数：
        current_placement_group: 要等待的 Ray Placement Group
    """
    # Wait until PG is ready - this will block until all
    # requested resources are available, and will time out
    # if they cannot be provisioned.
    placement_group_specs = current_placement_group.bundle_specs

    s = time.time()
    pg_ready_ref = current_placement_group.ready()
    wait_interval = 10  # 初始等待间隔（秒）
    while time.time() - s < PG_WAIT_TIMEOUT:
        ready, _ = ray.wait([pg_ready_ref], timeout=wait_interval)
        if len(ready) > 0:
            break

        # Exponential backoff for warning print.
        # 指数退避：每次等待间隔翻倍，减少日志频率
        wait_interval *= 2
        logger.info(
            "Waiting for creating a placement group of specs for "
            "%d seconds. specs=%s. Check `ray status` and "
            "`ray list nodes` to see if you have enough resources,"
            " and make sure the IP addresses used by ray cluster"
            " are the same as VLLM_HOST_IP environment variable"
            " specified in each node if you are running on a multi-node.",
            int(time.time() - s),
            placement_group_specs,
        )

    try:
        ray.get(pg_ready_ref, timeout=0)
    except ray.exceptions.GetTimeoutError:
        # Provide more helpful error message when GPU count is exceeded
        # 计算所需的 GPU 总数
        total_gpu_required = sum(spec.get("GPU", 0) for spec in placement_group_specs)
        # If more than one GPU is required for the placement group, provide a
        # more specific error message.
        # We use >1 here because multi-GPU (tensor parallel) jobs are more
        # likely to fail due to insufficient cluster resources, and users may
        # need to adjust tensor_parallel_size to fit available GPUs.
        if total_gpu_required > 1:
            # 多 GPU 场景：提示 TP 大小可能超过集群可用 GPU 数量
            raise ValueError(
                f"Cannot provide a placement group requiring "
                f"{total_gpu_required} GPUs "
                f"(placement_group_specs={placement_group_specs}) within "
                f"{PG_WAIT_TIMEOUT} seconds.\n"
                f"Tensor parallel size may exceed available GPUs in your "
                f"cluster. Check resources with `ray status` and "
                f"`ray list nodes`.\n"
                f"If running on K8s with limited GPUs, consider reducing "
                f"--tensor-parallel-size to match available GPU resources."
            ) from None
        else:
            # 单 GPU 场景：通用的资源不足提示
            raise ValueError(
                "Cannot provide a placement group of "
                f"{placement_group_specs=} within "
                f"{PG_WAIT_TIMEOUT} seconds. See "
                "`ray status` and `ray list nodes` to make sure the cluster "
                "has enough resources."
            ) from None


def _wait_until_pg_removed(current_placement_group: "PlacementGroup"):
    """
    等待 Placement Group 被完全移除。

    流程：
    1. 请求 Ray 移除 Placement Group
    2. 轮询检查 PG 是否已被移除
    3. 使用指数退避策略（10, 20, 40, ... 秒）打印等待日志
    4. 超过 PG_WAIT_TIMEOUT 后返回（不抛异常，因为这是清理操作）

    用途：在引擎关闭或重建集群时清理资源。

    参数：
        current_placement_group: 要移除的 Ray Placement Group
    """
    ray.util.remove_placement_group(current_placement_group)
    s = time.time()
    wait_interval = 10  # 初始等待间隔（秒）
    while time.time() - s < PG_WAIT_TIMEOUT:
        pg = ray.util.get_current_placement_group()
        if pg is None:
            break

        # Exponential backoff for warning print.
        # 指数退避：每次等待间隔翻倍
        wait_interval *= 2
        logger.info(
            "Waiting for removing a placement group of specs for %d seconds.",
            int(time.time() - s),
        )
        time.sleep(wait_interval)


def initialize_ray_cluster(
    parallel_config: ParallelConfig,
    ray_address: str | None = None,
    require_gpu_on_driver: bool = True,
):
    """
    使用 Ray 初始化分布式集群。

    此函数是 vLLM Ray 分布式执行的入口点，完成以下工作：
    1. 初始化或连接 Ray 集群
    2. 创建或复用 Placement Group（PG）以分配 GPU 资源
    3. 验证资源分配是否满足并行需求
    4. 将 PG 绑定到 parallel_config 中供后续使用

    整体流程：
    1. 前置检查：验证 Ray 可用性
    2. 环境准备：禁用 Ray 使用统计、预验证 GPU 需求
    3. Ray 初始化：根据平台选择不同的初始化策略
    4. Placement Group 管理：
       a. 如果已有 PG（用户预创建或 config 指定）-> 验证并复用
       b. 如果没有 PG -> 创建新的 PG 并等待资源就绪
    5. 后验证：确认 Bundle 分配满足 TP/PP 需求

    参数：
        parallel_config: 并行配置，包含 world_size、TP/PP 大小、
            ray_runtime_env、placement_group 等
        ray_address: Ray 集群地址。None 表示使用默认地址。
            也可以是 "auto"（自动发现已有集群）或具体地址。
        require_gpu_on_driver: 是否要求 Driver 节点有 GPU。
            - True（默认）：Driver 也是一个 Worker，需要 GPU
            - False：用于 RayExecutorV2 等场景，所有 GPU 工作委托给远程 Actor
    """
    assert_ray_available()
    from vllm.platforms import current_platform

    # Disable Ray usage stats collection
    # 禁用 Ray 使用统计收集，避免隐私问题和网络请求
    if os.environ.get("RAY_USAGE_STATS_ENABLED", "0") != "1":
        os.environ["RAY_USAGE_STATS_ENABLED"] = "0"

    # Prevalidate GPU requirements before Ray processing
    # 在 Ray 处理前预验证 GPU 需求，提前给出友好的警告信息
    if current_platform.is_cuda() and parallel_config.world_size > 1:
        available_gpus = current_platform.device_count()
        if parallel_config.world_size > available_gpus:
            logger.warning(
                "Tensor parallel size (%d) exceeds available GPUs (%d). "
                "This may result in Ray placement group allocation failures. "
                "Consider reducing tensor_parallel_size to %d or less, "
                "or ensure your Ray cluster has %d GPUs available.",
                parallel_config.world_size,
                available_gpus,
                available_gpus,
                parallel_config.world_size,
            )

    # Ray 初始化策略（根据平台不同）：
    # 1. 已初始化 -> 跳过
    # 2. ROCm/XPU -> 先尝试连接已有集群，失败则新建
    # 3. 其他平台 -> 直接初始化
    if ray.is_initialized():
        logger.info("Ray is already initialized. Skipping Ray initialization.")
    elif current_platform.is_rocm() or current_platform.is_xpu():
        # Try to connect existing ray instance and create a new one if not found
        # ROCm 和 XPU 平台：优先连接已有的 Ray 实例
        try:
            ray.init("auto")
        except ConnectionError:
            logger.warning(
                "No existing RAY instance detected. "
                "A new instance will be launched with current node resources."
            )
            ray.init(
                address=ray_address,
                num_gpus=parallel_config.world_size,
                runtime_env=parallel_config.ray_runtime_env,
            )
    else:
        ray.init(address=ray_address, runtime_env=parallel_config.ray_runtime_env)

    # 获取当前平台的设备类型字符串（如 "GPU"、"NPU"、"TPU"）
    device_str = current_platform.ray_device_key
    if not device_str:
        raise ValueError(
            f"current platform {current_platform.device_name} does not support ray."
        )

    # Create or get the placement group for worker processes
    # 优先使用用户在 config 中指定的 PG，否则获取当前 PG
    if parallel_config.placement_group:
        current_placement_group = parallel_config.placement_group
    else:
        current_placement_group = ray.util.get_current_placement_group()

    if current_placement_group:
        # 已有 Placement Group：验证其资源是否满足需求
        logger.info("Using the existing placement group")

        # We are in a placement group
        bundles = current_placement_group.bundle_specs
        # Verify that we can use the placement group.
        # 验证规则：
        # 1. 每个 Bundle 最多只能有 1 个设备（不允许一个 Bundle 绑定多个 GPU）
        # 2. 设备 Bundle 总数必须 >= world_size
        device_bundles = 0
        for bundle in bundles:
            bundle_devices = bundle.get(device_str, 0)
            if bundle_devices > 1:
                raise ValueError(
                    f"Placement group bundle cannot have more than 1 {device_str}."
                )
            if bundle_devices:
                device_bundles += 1
        if parallel_config.world_size > device_bundles:
            raise ValueError(
                f"The number of required {device_str}s exceeds the total "
                f"number of available {device_str}s in the placement group. "
                f"Required number of devices: {parallel_config.world_size}. "
                f"Total number of devices: {device_bundles}."
            )
    else:
        # 没有现有 PG：创建新的 Placement Group
        logger.info("No current placement group found. Creating a new placement group.")
        num_devices_in_cluster = ray.cluster_resources().get(device_str, 0)
        # Log a warning message and delay resource allocation failure response.
        # Avoid immediate rejection to allow user-initiated placement group
        # created and wait cluster to be ready
        # 只记录警告而不立即失败，给用户留出时间手动创建 PG 或等待集群就绪
        if parallel_config.world_size > num_devices_in_cluster:
            logger.warning(
                "The number of required %ss exceeds the total "
                "number of available %ss in the placement group.",
                device_str,
                device_str,
            )
        # Create a new placement group
        # 为每个 Worker 创建一个 Bundle（每个 Bundle 包含 1 个设备）
        placement_group_specs: list[dict[str, float]] = [
            {device_str: 1.0} for _ in range(parallel_config.world_size)
        ]

        # vLLM engine is also a worker to execute model with an accelerator,
        # so it requires to have the device in a current node. Check if
        # the current node has at least one device.
        # vLLM 引擎本身也是一个 Worker（需要 GPU 来执行模型），
        # 因此需要确保当前节点至少有一个设备。
        current_ip = get_ip()
        current_node_id = ray.get_runtime_context().get_node_id()
        current_node_resource = available_resources_per_node()[current_node_id]
        # TODO (jeffreywang): require_gpu_on_driver should be always False
        # after deprecating RayDistributedExecutor.
        if require_gpu_on_driver:
            if current_node_resource.get(device_str, 0) < 1:
                raise ValueError(
                    f"Current node has no {device_str} available. "
                    f"{current_node_resource=}. vLLM engine cannot start "
                    f"without {device_str}. Make sure you have at least 1 "
                    f"{device_str} available in a node "
                    f"{current_node_id=} {current_ip=}."
                )
            # This way, at least bundle is required to be created in a
            # current node.
            # 通过在第一个 Bundle 中添加 "node:{current_ip}" 软约束，
            # 确保至少有一个 Bundle 被分配到当前节点（Driver 节点）。
            # 0.001 是软约束的权重（Ray 会尽量满足但不强制）。
            placement_group_specs[0][f"node:{current_ip}"] = 0.001

        # By default, Ray packs resources as much as possible.
        # 使用 PACK 策略：尽量将所有 Bundle 放在同一节点（减少跨节点通信）
        current_placement_group = ray.util.placement_group(
            placement_group_specs, strategy="PACK"
        )
        # 等待 PG 准备就绪（阻塞直到资源分配完成或超时）
        _wait_until_pg_ready(current_placement_group)

    # 后验证：确认 PG 满足 TP/PP 需求
    assert current_placement_group is not None
    _verify_bundles(
        current_placement_group, parallel_config, device_str, require_gpu_on_driver
    )
    # Set the placement group in the parallel config
    # 将 PG 绑定到 parallel_config，供 Executor 和 Worker 后续使用
    parallel_config.placement_group = current_placement_group


def get_num_tpu_nodes() -> int:
    """
    获取 Ray 集群中 TPU 节点的数量。

    计算方法：集群 TPU 总数 / 每节点 TPU 数量

    用于 TPU 分布式执行时确定流水线并行的节点数。

    返回：
        TPU 节点数量（整数）
    """
    from ray._private.accelerators import TPUAcceleratorManager

    cluster_resources = ray.cluster_resources()
    total_tpus = int(cluster_resources["TPU"])
    tpus_per_node = TPUAcceleratorManager.get_current_node_num_accelerators()
    assert total_tpus % tpus_per_node == 0
    return total_tpus // tpus_per_node


def get_num_nodes_in_placement_group() -> int:
    """
    获取当前 Placement Group 涉及的节点数量。

    遍历 PG 的所有 Bundle，收集不同的 node_id，返回去重后的数量。

    用途：
    - 判断分布式执行是否跨节点
    - 用于优化通信策略（同节点 vs 跨节点）

    返回：
        PG 中的节点数量。如果没有当前 PG，返回 0。
    """
    pg_table = ray.util.placement_group_table()
    current_pg = ray.util.get_current_placement_group()
    num_nodes = 0

    if current_pg:
        # 使用集合去重，统计 PG 中涉及的不同节点
        nodes_in_pg = set()
        for pg_key, pg in pg_table.items():
            if pg_key == current_pg.id.hex():
                for _, node in pg["bundles_to_node_id"].items():
                    nodes_in_pg.add(node)
        num_nodes = len(nodes_in_pg)

    return num_nodes
