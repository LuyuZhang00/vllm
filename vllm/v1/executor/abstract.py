# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# ===========================================================================
# 中文注释：本模块定义了 vLLM V1 引擎中 Executor（执行器）的抽象基类。
#
# 【模块职责】
#   Executor 是 vLLM V1 引擎的核心组件之一，位于 Scheduler（调度器）和
#   Worker（工作进程）之间，负责将调度器的输出（SchedulerOutput）分发到
#   各个 Worker 进程中执行模型推理。
#
# 【在 vLLM V1 请求处理链路中的位置】
#   EngineCore（引擎核心）
#     -> Scheduler 产出 SchedulerOutput（调度输出，包含要执行的批次信息）
#       -> Executor.collective_rpc() 将指令广播到所有 Worker 进程
#         -> Worker（如 GPUWorker）执行模型前向推理
#           -> 返回 ModelRunnerOutput（模型运行结果，包含生成的 token）
#
# 【类层次结构】
#   Executor（抽象基类，本文件）
#     ├── UniProcExecutor     —— 单进程执行器，适用于单 GPU 场景
#     ├── MultiprocExecutor   —— 多进程执行器，使用 Python multiprocessing
#     ├── RayDistributedExecutor —— 基于 Ray 的分布式执行器
#     ├── RayExecutorV2       —— Ray 执行器 V2 版本
#     └── ExecutorWithExternalLauncher —— 外部启动器模式的执行器
#
# 【核心设计理念】
#   1. 抽象接口统一：所有 Executor 子类都必须实现相同的接口，上层（EngineCore）
#      无需关心底层使用哪种执行后端。
#   2. collective_rpc 模式：Executor 通过 collective_rpc() 方法将操作广播到
#      所有 Worker，这是一种"集体远程过程调用"模式，确保所有 GPU 上的 Worker
#      执行相同的操作。
#   3. 控制面与数据面分离：collective_rpc 用于传递控制消息（如"执行模型"、
#      "初始化 KV 缓存"），而实际的模型数据通过 CUDA IPC 等机制直接传输。
# ===========================================================================

import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from concurrent.futures import Future
from functools import cached_property
from typing import TYPE_CHECKING, Literal, TypeVar, overload

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorHandshakeMetadata,
)
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.tasks import SupportedTask
from vllm.tracing import instrument
from vllm.utils.import_utils import resolve_obj_by_qualname
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.engine import ReconfigureDistributedRequest
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec
from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput
from vllm.v1.worker.worker_base import CompilationTimes, WorkerBase

if TYPE_CHECKING:
    from vllm.distributed.kv_transfer.kv_connector.base import KVConnectorBase

logger = init_logger(__name__)

# 泛型类型变量，用于 collective_rpc 的返回值类型标注
_R = TypeVar("_R")

# 失败回调函数类型：当 Executor 进入永久失败状态时调用的无参函数
FailureCallback = Callable[[], None]


class Executor(ABC):
    """Abstract base class for vLLM executors.

    An executor is responsible for executing the model on one device,
    or it can be a distributed executor that can execute the model on multiple devices.
    """

    # 中文注释：Executor 抽象基类
    #
    # 【职责说明】
    #   Executor 是 vLLM V1 引擎中连接 Scheduler 和 Worker 的中间层。
    #   它接收 Scheduler 产出的 SchedulerOutput，并将其分发到一个或多个
    #   Worker 进程中执行模型的前向推理。
    #
    # 【关键属性】
    #   - uses_ray: 是否使用 Ray 框架进行分布式编排（如多节点推理）
    #   - supports_pp: 是否支持流水线并行（Pipeline Parallelism）
    #
    # 【核心方法】
    #   - __init__(): 初始化执行器，提取各类配置
    #   - _init_executor(): 抽象方法，子类必须实现具体的初始化逻辑
    #   - collective_rpc(): 核心方法，向所有 Worker 广播 RPC 调用
    #   - execute_model(): 执行模型推理，返回推理结果
    #   - sample_tokens(): 对模型输出进行 token 采样
    #   - check_health(): 健康检查，确保执行器正常运行
    #   - sleep()/wake_up(): 休眠/唤醒执行器，用于资源管理
    #
    # 【子类实现要求】
    #   子类必须实现以下抽象方法：
    #   1. _init_executor(): 完成子类特定的初始化逻辑
    #   2. collective_rpc(): 实现向 Worker 广播 RPC 的具体机制
    #   3. check_health(): 实现健康检查逻辑

    uses_ray: bool = False  # whether the executor uses Ray for orchestration.
    # 中文注释：标记此执行器是否使用 Ray 进行分布式编排。
    # Ray 是一个分布式计算框架，用于跨节点的 GPU 协调。

    supports_pp: bool = False  # whether the executor supports PP
    # 中文注释：标记此执行器是否支持流水线并行（Pipeline Parallelism）。
    # 流水线并行将模型的不同层分配到不同的设备上。

    @staticmethod
    def get_class(vllm_config: VllmConfig) -> type["Executor"]:
        # 中文注释：静态工厂方法，根据配置返回对应的 Executor 子类。
        #
        # 【工作流程】
        #   1. 从 vllm_config 中读取 parallel_config.distributed_executor_backend
        #   2. 根据 backend 的类型选择对应的 Executor 子类：
        #      - 如果是 type 对象：直接使用（必须是 Executor 的子类）
        #      - "ray": 使用 Ray 分布式执行器（支持 V1 和 V2 两个版本）
        #      - "mp":  使用 MultiprocExecutor（基于 Python multiprocessing）
        #      - "uni": 使用 UniProcExecutor（单进程，适用于单 GPU）
        #      - "external_launcher": 使用外部启动器模式
        #      - 如果是字符串：通过 resolve_obj_by_qualname 动态解析类名
        #   3. 返回解析后的 Executor 子类
        #
        # 【参数说明】
        #   - vllm_config: vLLM 全局配置对象，包含所有配置信息
        #
        # 【返回值】
        #   - 返回 Executor 的某个子类（type[Executor]）

        executor_class: type[Executor]
        parallel_config = vllm_config.parallel_config
        distributed_executor_backend = parallel_config.distributed_executor_backend
        # distributed_executor_backend must be set in VllmConfig.__post_init__
        if isinstance(distributed_executor_backend, type):
            if not issubclass(distributed_executor_backend, Executor):
                raise TypeError(
                    "distributed_executor_backend must be a subclass of "
                    f"Executor. Got {distributed_executor_backend}."
                )
            executor_class = distributed_executor_backend
        elif distributed_executor_backend == "ray":
            if envs.VLLM_USE_RAY_V2_EXECUTOR_BACKEND:
                from vllm.v1.executor.ray_executor_v2 import RayExecutorV2

                executor_class = RayExecutorV2
            else:
                from vllm.v1.executor.ray_executor import RayDistributedExecutor

                executor_class = RayDistributedExecutor
        elif distributed_executor_backend == "mp":
            from vllm.v1.executor.multiproc_executor import MultiprocExecutor

            executor_class = MultiprocExecutor
        elif distributed_executor_backend == "uni":
            from vllm.v1.executor.uniproc_executor import UniProcExecutor

            executor_class = UniProcExecutor
        elif distributed_executor_backend == "external_launcher":
            # TODO: make v1 scheduling deterministic
            # to support external launcher
            executor_class = ExecutorWithExternalLauncher
        elif isinstance(distributed_executor_backend, str):
            executor_class = resolve_obj_by_qualname(distributed_executor_backend)
            if not issubclass(executor_class, Executor):
                raise TypeError(
                    "distributed_executor_backend must be a subclass of "
                    f"Executor. Got {executor_class}."
                )
        else:
            raise ValueError(
                f"Unknown distributed executor backend: {distributed_executor_backend}"
            )
        return executor_class

    @instrument(span_name="Executor init")
    def __init__(
        self,
        vllm_config: VllmConfig,
    ) -> None:
        # 中文注释：Executor 构造函数
        #
        # 【功能说明】
        #   初始化 Executor 实例，从 vllm_config 中提取并缓存各类子配置，
        #   然后调用子类实现的 _init_executor() 完成具体初始化。
        #
        # 【初始化步骤】
        #   1. 缓存各种配置引用（model、cache、lora、load、parallel 等）
        #   2. 调用 _init_executor() 执行子类特定的初始化逻辑
        #   3. 初始化睡眠状态和 KV 输出聚合器
        #
        # 【参数说明】
        #   - vllm_config: vLLM 全局配置对象，包含所有模块的配置
        #
        # 【注意】
        #   @instrument 装饰器用于 OpenTelemetry 分布式追踪，记录初始化耗时
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.load_config = vllm_config.load_config
        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self.device_config = vllm_config.device_config
        self.speculative_config = vllm_config.speculative_config
        self.observability_config = vllm_config.observability_config
        self._init_executor()
        self.is_sleeping = False
        # 中文注释：记录当前处于休眠状态的资源标签集合
        # 可能的标签值包括 "weights"（模型权重）和 "kv_cache"（KV 缓存）
        self.sleeping_tags: set[str] = set()
        # 中文注释：KV 输出聚合器，用于在 KV 缓存传输场景下聚合多个 Worker 的输出
        self.kv_output_aggregator: KVOutputAggregator | None = None

    @abstractmethod
    def _init_executor(self) -> None:
        # 中文注释：抽象方法，子类必须实现的初始化逻辑。
        #
        # 【功能说明】
        #   每个 Executor 子类在此方法中完成其特定的初始化工作，例如：
        #   - UniProcExecutor: 初始化单个 Worker 实例
        #   - MultiprocExecutor: 启动多个子进程，每个子进程运行一个 Worker
        #   - RayDistributedExecutor: 初始化 Ray 集群和远程 Worker
        #
        # 【调用时机】
        #   在 __init__() 中被调用，在配置缓存完成后执行
        raise NotImplementedError

    def initialize_from_config(self, kv_cache_configs: list[KVCacheConfig]) -> None:
        """
        Initialize the KV caches and begin the model execution loop of the
        underlying workers.
        """
        # 中文注释：从配置初始化 KV 缓存并启动模型执行循环
        #
        # 【功能说明】
        #   1. 通过 collective_rpc 将 KV 缓存配置广播到所有 Worker，
        #      调用每个 Worker 的 initialize_from_config() 方法
        #   2. 调用 compile_or_warm_up_model() 让所有 Worker 编译或预热模型
        #   3. 将 Worker 的编译时间回传到主进程的配置中
        #
        # 【为什么需要回传编译时间】
        #   当使用张量并行（TP > 1）时，模型编译发生在 Worker 子进程中，
        #   主进程的 compilation_config 不会自动更新编译时间。这里使用所有
        #   Worker 中的最大编译时间，因为它们是并行编译的。
        #
        # 【参数说明】
        #   - kv_cache_configs: 每个 Worker 的 KV 缓存配置列表
        self.collective_rpc("initialize_from_config", args=(kv_cache_configs,))
        compilation_times: list[CompilationTimes] = self.collective_rpc(
            "compile_or_warm_up_model"
        )
        # Propagate compilation time from workers back to the main process.
        # With TP>1, compilation happens in worker processes, so the main
        # process config is never updated. Use max across workers since they
        # compile in parallel.
        if compilation_times:
            self.vllm_config.compilation_config.compilation_time = max(
                t.language_model for t in compilation_times
            )
            self.vllm_config.compilation_config.encoder_compilation_time = max(
                t.encoder for t in compilation_times
            )

    def register_failure_callback(self, callback: FailureCallback):  # noqa: B027
        """
        Register a function to be called if the executor enters a permanent
        failed state.
        """
        # 中文注释：注册失败回调函数
        #
        # 【功能说明】
        #   当 Executor 进入永久失败状态时，调用注册的回调函数。
        #   基类默认为空操作（pass），子类可以覆盖此方法来实现
        #   具体的失败处理逻辑（如通知上层组件、清理资源等）。
        #
        # 【参数说明】
        #   - callback: 失败时要调用的回调函数，无参数无返回值
        pass

    def determine_available_memory(self) -> list[int]:  # in bytes
        # 中文注释：探测每个 Worker 的可用显存大小
        #
        # 【功能说明】
        #   通过 collective_rpc 调用所有 Worker 的 determine_available_memory()，
        #   获取每个 GPU 上的可用显存（单位：字节）。
        #   返回值列表中的每个元素对应一个 Worker 的可用显存。
        #   这些信息用于 KV 缓存管理器计算可以分配多少个 KV 缓存块。
        return self.collective_rpc("determine_available_memory")

    def get_kv_cache_specs(self) -> list[dict[str, KVCacheSpec]]:
        # 中文注释：获取每个 Worker 的 KV 缓存规格说明
        #
        # 【功能说明】
        #   通过 collective_rpc 调用所有 Worker 的 get_kv_cache_spec()，
        #   获取每个 Worker 关于 KV 缓存的规格信息。
        #   KVCacheSpec 描述了模型需要的 KV 缓存结构（如层数、注意力头数、
        #   每个头的维度等），用于 KV 缓存管理器的初始化和块分配。
        #
        # 【返回值】
        #   - list[dict[str, KVCacheSpec]]: 每个 Worker 返回一个字典，
        #     键为缓存名称，值为对应的 KVCacheSpec
        return self.collective_rpc("get_kv_cache_spec")

    @overload
    def collective_rpc(
        self,
        method: str | Callable[[WorkerBase], _R],
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
        non_block: Literal[False] = False,
    ) -> list[_R]:
        """
        Execute an RPC call on all workers.

        Args:
            method: Name of the worker method to execute, or a callable that
                is serialized and sent to all workers to execute.

                If the method is a callable, it should accept an additional
                `self` argument, in addition to the arguments passed in `args`
                and `kwargs`. The `self` argument will be the worker object.
            timeout: Maximum time in seconds to wait for execution. Raises a
                [`TimeoutError`][] on timeout. `None` means wait indefinitely.
            args: Positional arguments to pass to the worker method.
            kwargs: Keyword arguments to pass to the worker method.
            non_block: If `True`, returns a list of Futures instead of waiting
                for the results.

        Returns:
            A list containing the results from each worker.

        Note:
            It is recommended to use this API to only pass control messages,
            and set up data-plane communication to pass data.
        """
        # 中文注释：collective_rpc 非阻塞模式的类型重载签名（non_block=False）
        #
        # 【功能说明】
        #   collective_rpc 是 Executor 的核心方法，用于向所有 Worker 广播
        #   远程过程调用（RPC）。这是 Executor 与 Worker 通信的主要机制。
        #
        # 【设计理念】
        #   这里使用 @overload 装饰器为不同的 non_block 值提供不同的返回类型：
        #   - non_block=False 时：返回 list[_R]（同步等待结果）
        #   - non_block=True 时：返回 Future[list[_R]]（异步返回 Future）
        #
        # 【method 参数的两种形式】
        #   1. 字符串：Worker 方法名，如 "execute_model"、"shutdown" 等
        #   2. 可调用对象：会被序列化后发送到各 Worker 执行，需要接受一个
        #      额外的 self 参数（即 Worker 实例）
        #
        # 【使用建议】
        #   建议仅通过此 API 传递控制消息（如"执行推理"、"初始化缓存"），
        #   实际的数据（如张量）应通过 CUDA IPC 等高效的数据面通道传输。
        pass

    @overload
    def collective_rpc(
        self,
        method: str | Callable[[WorkerBase], _R],
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
        non_block: Literal[True] = True,
    ) -> Future[list[_R]]:
        # 中文注释：collective_rpc 阻塞模式的类型重载签名（non_block=True）
        #
        # 【功能说明】
        #   当 non_block=True 时，方法立即返回一个 Future 对象，
        #   调用者可以在需要结果时通过 future.result() 获取。
        #   这允许调用者在等待结果的同时执行其他操作（异步调度）。
        pass

    @abstractmethod
    def collective_rpc(
        self, method, timeout=None, args=(), kwargs=None, non_block: bool = False
    ):
        # 中文注释：collective_rpc 的实际抽象方法实现
        #
        # 【功能说明】
        #   子类必须实现此方法，具体实现向所有 Worker 广播 RPC 调用的机制。
        #
        # 【不同子类的实现方式】
        #   - UniProcExecutor: 直接在当前进程调用 Worker 方法（无 IPC 开销）
        #   - MultiprocExecutor: 通过 Python multiprocessing 的 IPC 机制
        #     将调用分发到各个子进程中的 Worker
        #   - RayDistributedExecutor: 通过 Ray 的远程调用机制分发到远程 Worker
        #
        # 【参数说明】
        #   - method: 要调用的 Worker 方法名或可调用对象
        #   - timeout: 超时时间（秒），None 表示无限等待
        #   - args: 位置参数元组
        #   - kwargs: 关键字参数字典
        #   - non_block: 是否非阻塞，True 时返回 Future 对象
        #
        # 【返回值】
        #   - non_block=False: list[_R]，包含每个 Worker 的返回值
        #   - non_block=True: Future[list[_R]]，异步返回的结果
        raise NotImplementedError

    def get_kv_connector_handshake_metadata(
        self,
    ) -> list[dict[int, KVConnectorHandshakeMetadata]]:
        # 中文注释：获取 KV 缓存连接器的握手元数据
        #
        # 【功能说明】
        #   在分布式 KV 缓存传输场景中（如分离式 prefill/decode 架构），
        #   各节点之间需要先进行"握手"以交换元数据（如节点地址、缓存布局等）。
        #   此方法获取每个 Worker 返回的握手元数据。
        #
        # 【返回值】
        #   - list[dict[int, KVConnectorHandshakeMetadata]]:
        #     每个 Worker 返回一个字典，键为连接器 ID，值为握手元数据
        return self.collective_rpc("get_kv_connector_handshake_metadata")

    @overload
    def execute_model(
        self, scheduler_output: SchedulerOutput, non_block: Literal[False] = False
    ) -> ModelRunnerOutput | None:
        pass

    @overload
    def execute_model(
        self, scheduler_output: SchedulerOutput, non_block: Literal[True] = True
    ) -> Future[ModelRunnerOutput | None]:
        pass

    def execute_model(
        self, scheduler_output: SchedulerOutput, non_block: bool = False
    ) -> ModelRunnerOutput | None | Future[ModelRunnerOutput | None]:
        # 中文注释：执行模型推理的核心方法
        #
        # 【功能说明】
        #   1. 将 SchedulerOutput 通过 collective_rpc 广播到所有 Worker
        #   2. 每个 Worker 根据 SchedulerOutput 中的批次信息执行模型前向推理
        #   3. 从第一个 Worker（driver worker）获取推理结果
        #
        # 【执行流程】
        #   SchedulerOutput 包含：
        #   - 要处理的请求列表及其 token 信息
        #   - KV 缓存块的分配/释放信息
        #   - 运行的批次元数据
        #   Worker 收到后：
        #   - 准备输入张量
        #   - 执行模型前向推理
        #   - 返回 ModelRunnerOutput（包含生成的 token ID 等）
        #
        # 【参数说明】
        #   - scheduler_output: 调度器输出，包含本轮要执行的批次信息
        #   - non_block: 是否非阻塞执行
        #
        # 【返回值】
        #   - non_block=False: ModelRunnerOutput 或 None（第一个 Worker 的结果）
        #   - non_block=True: Future[ModelRunnerOutput | None]
        #
        # 【为什么只返回 output[0]】
        #   在张量并行（TP）模式下，只有 driver worker（rank 0）返回完整结果，
        #   其他 worker 返回 None。因此只需取第一个结果即可。
        output = self.collective_rpc(  # type: ignore[call-overload]
            "execute_model", args=(scheduler_output,), non_block=non_block
        )
        return output[0]

    @overload
    def sample_tokens(
        self, grammar_output: GrammarOutput | None, non_block: Literal[False] = False
    ) -> ModelRunnerOutput:
        pass

    @overload
    def sample_tokens(
        self, grammar_output: GrammarOutput | None, non_block: Literal[True] = True
    ) -> Future[ModelRunnerOutput]:
        pass

    def sample_tokens(
        self, grammar_output: GrammarOutput | None, non_block: bool = False
    ) -> ModelRunnerOutput | Future[ModelRunnerOutput]:
        # 中文注释：对模型输出进行 token 采样
        #
        # 【功能说明】
        #   在模型前向推理完成后，调用此方法对输出的 logits 进行采样，
        #   选择最终生成的 token。
        #
        # 【与 execute_model 的关系】
        #   在 V1 引擎中，推理和采样是分离的两个步骤：
        #   1. execute_model(): 执行模型前向推理，计算 logits
        #   2. sample_tokens(): 对 logits 进行采样，得到最终 token
        #   这种分离允许更灵活的调度策略（如异步调度）。
        #
        # 【参数说明】
        #   - grammar_output: 语法引导输出（用于结构化生成），
        #     如果为 None 则不使用语法约束
        #
        # 【返回值】
        #   - ModelRunnerOutput: 包含采样后的 token ID 等信息
        output = self.collective_rpc(  # type: ignore[call-overload]
            "sample_tokens", args=(grammar_output,), non_block=non_block
        )
        return output[0]

    def execute_dummy_batch(self) -> None:
        # 中文注释：执行一个空批次（dummy batch）
        #
        # 【功能说明】
        #   在所有 Worker 上执行一个虚拟的空批次。这通常用于：
        #   1. 预热 CUDA 内核和 JIT 编译
        #   2. 触发延迟初始化逻辑
        #   3. 在首次真实推理前确保所有组件就绪
        self.collective_rpc("execute_dummy_batch")

    def take_draft_token_ids(self) -> DraftTokenIds | None:
        # 中文注释：获取推测解码（speculative decoding）的草稿 token ID
        #
        # 【功能说明】
        #   在推测解码场景中，草稿模型（draft model）会先快速生成多个
        #   候选 token（草稿 token），然后由目标模型（target model）验证。
        #   此方法从 driver worker 获取这些草稿 token ID。
        #
        # 【返回值】
        #   - DraftTokenIds: 草稿 token ID 信息
        #   - None: 如果没有草稿 token
        output: list[DraftTokenIds] = self.collective_rpc("take_draft_token_ids")
        return output[0]

    @property
    def max_concurrent_batches(self) -> int:
        # 中文注释：最大并发批次数
        #
        # 【功能说明】
        #   返回此执行器支持的最大并发批次数。默认为 1，表示一次只处理一个批次。
        #   某些高级执行器（如支持异步调度的执行器）可能覆盖此属性以支持
        #   多个批次的流水线执行。
        return 1

    def profile(self, is_start: bool = True, profile_prefix: str | None = None):
        # 中文注释：性能分析（profiling）控制
        #
        # 【功能说明】
        #   启动或停止所有 Worker 上的性能分析（如 PyTorch Profiler、
        #   Nsight Systems 等）。
        #
        # 【参数说明】
        #   - is_start: True 表示开始分析，False 表示停止分析
        #   - profile_prefix: 分析结果文件的前缀路径
        self.collective_rpc("profile", args=(is_start, profile_prefix))

    def save_sharded_state(
        self,
        path: str,
        pattern: str | None = None,
        max_size: int | None = None,
    ) -> None:
        # 中文注释：保存分片的模型状态
        #
        # 【功能说明】
        #   将模型权重以分片（sharded）格式保存到磁盘。每个 Worker 保存
        #   自己负责的那部分权重。
        #
        # 【参数说明】
        #   - path: 保存路径
        #   - pattern: 文件名匹配模式（可选，用于过滤保存哪些层）
        #   - max_size: 单个分片文件的最大大小（字节），用于控制文件数量
        self.collective_rpc(
            "save_sharded_state",
            kwargs=dict(path=path, pattern=pattern, max_size=max_size),
        )

    @abstractmethod
    def check_health(self) -> None:
        """Checks if the executor is healthy. If not, it should raise an
        exception."""
        # 中文注释：健康检查抽象方法
        #
        # 【功能说明】
        #   检查执行器及其所有 Worker 是否健康运行。
        #   如果检测到异常，应抛出相应的异常。
        #
        # 【使用场景】
        #   API 服务的健康检查端点（/health）会调用此方法来判断
        #   服务是否正常。
        #
        # 【子类实现要求】
        #   子类应检查：
        #   1. Worker 进程是否仍然存活
        #   2. GPU 是否正常工作
        #   3. 进程间通信是否正常
        raise NotImplementedError

    def shutdown(self) -> None:
        """Shutdown the executor."""
        # 中文注释：关闭执行器
        #
        # 【功能说明】
        #   通过 collective_rpc 通知所有 Worker 执行 shutdown 操作。
        #   Worker 收到后会清理资源（释放 GPU 显存、关闭通信通道等）。
        #   子类可以在调用此方法前/后执行额外的清理逻辑。
        self.collective_rpc("shutdown")

    def init_kv_output_aggregator(self, connector: "KVConnectorBase") -> None:
        """Init KVOutputAggregator"""
        # 中文注释：初始化 KV 输出聚合器
        #
        # 【功能说明】
        #   在分布式 KV 缓存传输场景中，多个 Worker 可能分别产生 KV 缓存
        #   的输出数据。KVOutputAggregator 负责聚合这些分散的输出，
        #   以便统一传输到目标节点。
        #
        # 【参数说明】
        #   - connector: KV 缓存连接器实例，用于确定聚合策略
        self.kv_output_aggregator = KVOutputAggregator.from_connector(
            connector, self.parallel_config.world_size
        )

    @cached_property  # Avoid unnecessary RPC calls
    def supported_tasks(self) -> tuple[SupportedTask, ...]:
        # 中文注释：获取当前模型支持的任务类型（带缓存）
        #
        # 【功能说明】
        #   通过 collective_rpc 查询 driver worker 支持的任务类型，
        #   并使用 @cached_property 缓存结果，避免重复的 RPC 调用。
        #
        # 【支持的任务类型示例】
        #   - "generate": 文本生成
        #   - "embedding": 文本嵌入
        #   - "classify": 文本分类
        #   - "score": 文本评分
        #
        # 【返回值】
        #   - tuple[SupportedTask, ...]: 支持的任务类型元组
        output: list[tuple[SupportedTask, ...]]
        output = self.collective_rpc("get_supported_tasks")
        return output[0]

    def add_lora(self, lora_request: LoRARequest) -> bool:
        # 中文注释：添加 LoRA 适配器
        #
        # 【功能说明】
        #   在所有 Worker 上加载并注册一个 LoRA（Low-Rank Adaptation）适配器。
        #   LoRA 是一种高效的模型微调方法，可以在不修改基础模型权重的情况下
        #   适配不同的下游任务。
        #
        # 【参数说明】
        #   - lora_request: LoRA 请求对象，包含 LoRA 模型路径、ID 等信息
        #
        # 【返回值】
        #   - bool: 如果所有 Worker 都成功加载则返回 True
        assert lora_request.lora_int_id > 0, "lora_id must be greater than 0."
        return all(self.collective_rpc("add_lora", args=(lora_request,)))

    def remove_lora(self, lora_id: int) -> bool:
        # 中文注释：移除 LoRA 适配器
        #
        # 【功能说明】
        #   从所有 Worker 上卸载指定 ID 的 LoRA 适配器，释放相关资源。
        #
        # 【参数说明】
        #   - lora_id: 要移除的 LoRA 适配器的整数 ID
        #
        # 【返回值】
        #   - bool: 如果所有 Worker 都成功移除则返回 True
        assert lora_id > 0, "lora_id must be greater than 0."
        return all(self.collective_rpc("remove_lora", args=(lora_id,)))

    def pin_lora(self, lora_id: int) -> bool:
        # 中文注释：固定 LoRA 适配器
        #
        # 【功能说明】
        #   将指定 ID 的 LoRA 适配器"固定"（pin），防止其被自动卸载。
        #   被固定的 LoRA 会常驻显存，适用于高频使用的 LoRA 适配器。
        #
        # 【参数说明】
        #   - lora_id: 要固定的 LoRA 适配器的整数 ID
        #
        # 【返回值】
        #   - bool: 如果所有 Worker 都成功固定则返回 True
        assert lora_id > 0, "lora_id must be greater than 0."
        return all(self.collective_rpc("pin_lora", args=(lora_id,)))

    def list_loras(self) -> set[int]:
        # 中文注释：列出所有已加载的 LoRA 适配器 ID
        #
        # 【功能说明】
        #   获取所有 Worker 上已加载的 LoRA 适配器 ID 集合，并断言
        #   所有 Worker 上的 LoRA 集合保持一致。
        #
        # 【返回值】
        #   - set[int]: 已加载的 LoRA 适配器 ID 集合
        #
        # 【一致性断言】
        #   所有 Worker 必须持有相同的 LoRA 集合，否则说明存在同步问题
        sets: list[set[int]] = self.collective_rpc("list_loras")
        for s in sets:
            assert s == sets[0], "All workers should have the same LORAs."
        return sets[0]

    def reset_mm_cache(self) -> None:
        """Reset the multi-modal cache in each worker."""
        # 中文注释：重置每个 Worker 中的多模态缓存
        #
        # 【功能说明】
        #   清除所有 Worker 中缓存的多模态输入数据（如图像特征、音频特征等）。
        #   这在处理完一批多模态请求后，或需要释放缓存显存时使用。
        self.collective_rpc("reset_mm_cache")

    def reset_encoder_cache(self) -> None:
        """Reset the encoder cache in each worker to clear cached encoder outputs."""
        # 中文注释：重置每个 Worker 中的编码器缓存
        #
        # 【功能说明】
        #   清除所有 Worker 中缓存的编码器输出（如视觉编码器、音频编码器
        #   的输出）。与 reset_mm_cache 类似，用于释放缓存的显存。
        self.collective_rpc("reset_encoder_cache")

    def sleep(self, level: int = 1):
        # 中文注释：让执行器进入休眠状态
        #
        # 【功能说明】
        #   将执行器切换到休眠模式，释放 GPU 资源（如模型权重、KV 缓存等）
        #   以节省显存。休眠期间无法执行推理。
        #
        # 【休眠级别】
        #   - level=1: 释放模型权重和 KV 缓存（默认）
        #   - 更高级别可能释放更多资源（取决于 Worker 实现）
        #
        # 【执行流程】
        #   1. 检查是否已在休眠状态
        #   2. 通过 collective_rpc 通知所有 Worker 进入休眠
        #   3. 记录休眠的资源标签（weights、kv_cache）
        #   4. 记录休眠耗时日志
        #
        # 【参数说明】
        #   - level: 休眠级别，控制释放多少资源
        if self.is_sleeping:
            logger.warning("Executor is already sleeping.")
            return
        time_before_sleep = time.perf_counter()
        self.collective_rpc("sleep", kwargs=dict(level=level))
        time_after_sleep = time.perf_counter()
        self.sleeping_tags = {"weights", "kv_cache"}
        self.is_sleeping = True
        logger.info(
            "It took %.6f seconds to fall asleep.", time_after_sleep - time_before_sleep
        )

    def wake_up(self, tags: list[str] | None = None):
        # 中文注释：唤醒休眠的执行器
        #
        # 【功能说明】
        #   将休眠的执行器唤醒，恢复指定资源到 GPU 显存中。
        #   支持部分唤醒（只恢复指定标签的资源）。
        #
        # 【唤醒流程】
        #   1. 检查是否确实处于休眠状态
        #   2. 验证请求的标签是否都在已休眠的标签集合中
        #   3. 通过 collective_rpc 通知所有 Worker 唤醒指定资源
        #   4. 从 sleeping_tags 中移除已唤醒的标签
        #   5. 如果所有标签都已唤醒，标记 is_sleeping = False
        #
        # 【参数说明】
        #   - tags: 要唤醒的资源标签列表（如 ["weights", "kv_cache"]）。
        #     如果为 None，则唤醒所有休眠的资源。
        #
        # 【使用示例】
        #   executor.sleep(level=1)  # 休眠，释放 weights 和 kv_cache
        #   executor.wake_up(["weights"])  # 只恢复模型权重
        #   executor.wake_up(["kv_cache"])  # 再恢复 KV 缓存
        if not self.is_sleeping:
            logger.warning("Executor is not sleeping.")
            return
        if tags:
            for tag in tags:
                if tag not in self.sleeping_tags:
                    logger.warning(
                        "Tag %s is not in sleeping tags %s", tag, self.sleeping_tags
                    )
                    return
        time_before_wakeup = time.perf_counter()
        self.collective_rpc("wake_up", kwargs=dict(tags=tags))
        time_after_wakeup = time.perf_counter()
        logger.info(
            "It took %.6f seconds to wake up tags %s.",
            time_after_wakeup - time_before_wakeup,
            tags if tags is not None else self.sleeping_tags,
        )
        if tags:
            for tag in tags:
                self.sleeping_tags.remove(tag)
        else:
            self.sleeping_tags.clear()
        if not self.sleeping_tags:
            self.is_sleeping = False

    def reinitialize_distributed(
        self, reconfig_request: ReconfigureDistributedRequest
    ) -> None:
        # 中文注释：重新初始化分布式通信
        #
        # 【功能说明】
        #   在运行时重新配置分布式通信（如添加/移除节点、更改并行度等）。
        #   这是一个高级操作，基类默认抛出 NotImplementedError，
        #   仅在支持动态重配置的执行器子类中实现。
        #
        # 【参数说明】
        #   - reconfig_request: 分布式重配置请求，包含新的并行配置等信息
        raise NotImplementedError

    @classmethod
    def supports_async_scheduling(cls) -> bool:
        """
        Whether the executor supports async scheduling.
        """
        # 中文注释：查询是否支持异步调度
        #
        # 【功能说明】
        #   异步调度允许在等待当前批次执行结果的同时，提前准备下一个批次。
        #   这可以提高 GPU 利用率和整体吞吐量。默认返回 False，
        #   支持异步调度的子类可以覆盖此方法返回 True。
        #
        # 【返回值】
        #   - bool: True 表示支持异步调度，False 表示不支持
        return False


from vllm.v1.executor.uniproc_executor import (  # noqa: E402
    ExecutorWithExternalLauncher as _ExecutorWithExternalLauncher,
)
from vllm.v1.executor.uniproc_executor import (  # noqa: E402
    UniProcExecutor as _UniProcExecutor,
)

# For backwards compatibility.
# 中文注释：向后兼容性导入
#   将 UniProcExecutor 和 ExecutorWithExternalLauncher 重新导出，
#   确保旧代码中直接从 abstract 模块导入这些类的路径仍然可用。
UniProcExecutor = _UniProcExecutor
ExecutorWithExternalLauncher = _ExecutorWithExternalLauncher
