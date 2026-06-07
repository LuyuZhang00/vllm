# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# =====================================================================
# 中文注释：本模块定义了 vLLM V1 引擎中 Worker 的基础抽象层。
#
# 【模块职责】
# 本文件包含两个核心基类：
#   1. WorkerBase  —— Worker 的抽象接口，定义了 Worker 需要实现的所有方法
#      （如初始化设备、加载模型、执行前向推理等），让不同硬件后端可以各自实现。
#   2. WorkerWrapperBase —— Worker 的进程级包装器，负责 Worker 的懒加载生命周期管理。
#      在多进程执行架构中，每个子进程对应一个 WorkerWrapperBase 实例。
#
# 【在 vLLM V1 链路中的位置】
#   EngineCore (调度器)
#     -> Scheduler 产出 SchedulerOutput
#       -> Executor 将 SchedulerOutput 分发给各 Worker 进程
#         -> WorkerWrapperBase.execute_model() 处理多模态缓存后委托给 WorkerBase
#           -> WorkerBase (具体子类如 GPUWorker) 调用 ModelRunner 执行前向推理
#             -> 返回 ModelRunnerOutput（包含采样结果）
#
# 【为什么需要两层抽象】
#   - WorkerBase：定义接口契约，让 GPU/CPU/TPU 等不同平台各自实现具体逻辑。
#   - WorkerWrapperBase：处理进程级生命周期（环境变量、插件加载、类动态扩展等），
#     与 Worker 的具体实现解耦。这样做使得 Executor 只需要和 Wrapper 交互，
#     而无需关心底层 Worker 的初始化细节。
# =====================================================================

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, NamedTuple, TypeVar

import torch
import torch.nn as nn

import vllm.ir
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.tracing import instrument
from vllm.utils.import_utils import resolve_obj_by_qualname
from vllm.utils.system_utils import update_environment_variables
from vllm.v1.kv_cache_interface import KVCacheSpec

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
    from vllm.v1.outputs import AsyncModelRunnerOutput, ModelRunnerOutput
else:
    SchedulerOutput = object
    GrammarOutput = object
    AsyncModelRunnerOutput = object
    ModelRunnerOutput = object

logger = init_logger(__name__)

_R = TypeVar("_R")


# 中文注释：记录模型编译/预热所花费的时间（秒）。
# language_model: 语言模型部分的编译时间；encoder: 编码器部分的编译时间。
# 这些信息用于性能监控和诊断，帮助开发者了解启动阶段的时间开销。
class CompilationTimes(NamedTuple):
    language_model: float
    encoder: float


# 中文注释：WorkerBase 是所有硬件 Worker 的抽象基类。
#
# 【设计目的】
# vLLM 需要支持多种硬件后端（GPU、CPU、TPU、XPU 等），每种硬件的设备初始化、
# 模型加载、前向推理、KV cache 操作等实现各不相同。WorkerBase 通过定义统一的接口
# 契约，让各平台可以各自实现具体逻辑，同时保持上层调度/执行逻辑的一致性。
#
# 【核心职责】
#   1. 管理配置信息（模型配置、缓存配置、并行配置等）
#   2. 定义设备初始化和模型加载接口
#   3. 定义前向推理执行接口（execute_model）
#   4. 定义 LoRA 管理接口
#   5. 定义 KV cache 规格查询接口
#
# 【典型子类】
#   - GPUWorker: GPU 后端的 Worker 实现
#   - 其他平台对应的 Worker 实现
class WorkerBase:
    """Worker interface that allows vLLM to cleanly separate implementations for
    different hardware. Also abstracts control plane communication, e.g., to
    communicate request metadata to other workers.
    """

    # 中文注释：Worker 的初始化流程。
    #
    # 【初始化步骤】
    #   1. 保存完整的 vLLM 配置（VllmConfig），并从中提取各子配置的快捷引用
    #     （model_config、cache_config、lora_config 等），方便后续方法中快速访问。
    #   2. 记录当前硬件平台信息（current_platform）。
    #   3. 设置分布式训练相关参数：rank（全局排名）、local_rank（本机设备索引）、
    #      distributed_init_method（分布式初始化方式，如 NCCL）。
    #   4. 标记是否为 driver worker（负责收集输出和协调其他 worker）。
    #   5. 初始化设备和模型运行器的占位符（device 和 model_runner 在 init_device 中设置）。
    #   6. 设置 IR（中间表示）操作优先级和 torch 包装状态，这些在 Worker 生命周期内保持不变。
    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ) -> None:
        """
        Initialize common worker components.

        Args:
            vllm_config: Complete vLLM configuration
            local_rank: Local device index
            rank: Global rank in distributed setup
            distributed_init_method: Distributed initialization method
            is_driver_worker: Whether this worker handles driver
                responsibilities
        """
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
        self.kv_transfer_config = vllm_config.kv_transfer_config
        self.compilation_config = vllm_config.compilation_config

        from vllm.platforms import current_platform

        self.current_platform = current_platform

        self.parallel_config.rank = rank
        self.local_rank = local_rank
        self.rank = rank
        self.distributed_init_method = distributed_init_method
        self.is_driver_worker = is_driver_worker

        # 中文注释：设备和模型运行器的占位符，后续在 init_device() 中由具体子类初始化。
        # device: 当前 worker 使用的计算设备（如 cuda:0）
        # model_runner: 模型运行器，负责管理模型的前向推理、CUDA graph、KV cache 操作等
        # Device and model state
        self.device: torch.device | None = None
        self.model_runner: nn.Module | None = None

        # IR op priority and torch-wrap state are constant for the worker's
        # lifetime.
        vllm_config.kernel_config.ir_op_priority.set_default()
        vllm.ir.set_default_torch_wrap(
            vllm_config.compilation_config.ir_enable_torch_wrap
        )

    # 中文注释：获取 KV cache 的规格说明。
    # 返回值是一个字典，key 为 KV cache 层的名称，value 为该层的 KVCacheSpec
    # （包含 block 大小、数据类型等信息）。
    # 这些信息被 KV cache manager 用于分配和管理物理 KV block。
    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        """Get specifications for KV cache implementation."""
        raise NotImplementedError

    # 中文注释：编译或预热模型，使其准备好接受推理请求。
    # 【为什么需要预热】
    #   - CUDA Graph 捕获需要至少运行一次才能录制。
    #   - torch.compile 首次编译开销很大，提前编译可以避免推理延迟抖动。
    #   - 内核预分配可以减少首次推理的延迟。
    # 【返回值】语言模型和编码器各自的编译耗时（秒），用于性能监控。
    def compile_or_warm_up_model(self) -> CompilationTimes:
        """Prepare model for execution through compilation/warmup.

        Returns:
            Compilation times (language_model, encoder) in seconds.
        """
        raise NotImplementedError

    def check_health(self) -> None:
        """Basic health check (override for device-specific checks)."""
        return

    # 中文注释：初始化计算设备状态。
    # 典型操作包括：
    #   1. 设置当前 CUDA 设备（torch.cuda.set_device）
    #   2. 初始化分布式进程组（NCCL 等）
    #   3. 加载模型权重到 GPU 显存
    #   4. 初始化 ModelRunner
    # 这是 Worker 生命周期中的关键步骤，在 init_device 完成后才能执行推理。
    def init_device(self) -> None:
        """Initialize device state, such as loading the model or other on-device
        memory allocations.
        """
        raise NotImplementedError

    def reset_mm_cache(self) -> None:
        reset_fn = getattr(self.model_runner, "reset_mm_cache", None)
        if callable(reset_fn):
            reset_fn()

    def get_model(self) -> nn.Module:
        raise NotImplementedError

    def apply_model(self, fn: Callable[[nn.Module], _R]) -> _R:
        """Apply a function on the model inside this worker."""
        return fn(self.get_model())

    def get_model_inspection(self) -> str:
        """Return a transformers-style hierarchical view of the model."""
        from vllm.model_inspection import format_model_inspection

        return format_model_inspection(self.get_model())

    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        """Load model onto target device."""
        raise NotImplementedError

    # 中文注释：执行模型的前向推理，这是 Worker 最核心的方法。
    #
    # 【在 vLLM V1 主链路中的位置】
    #   Scheduler 产出 SchedulerOutput（本轮要运行的请求及其 token 信息）
    #     -> Executor 将 SchedulerOutput 通过 IPC 发送给各 Worker
    #       -> Worker.execute_model() 被调用
    #         -> 内部委托给 ModelRunner 执行实际的模型前向传播和采样
    #
    # 【返回值含义】
    #   - 正常情况下返回 ModelRunnerOutput（包含每个请求的采样结果 token）
    #   - 特殊情况（如结构化输出/grammar-based decoding）下返回 None，
    #     此时调用方必须立即调用 sample_tokens() 来获取最终输出。
    #     这种两阶段设计是为了支持结构化输出的并行处理：
    #     execute_model 完成模型前向，sample_tokens 等待 grammar 引导后完成采样。
    def execute_model(
        self, scheduler_output: SchedulerOutput
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | None:
        """If this method returns None, sample_tokens should be called immediately after
        to obtain the ModelRunnerOutput.

        Note that this design may be changed in future if/when structured outputs
        parallelism is re-architected.
        """
        raise NotImplementedError

    # 中文注释：当 execute_model 返回 None 时，调用此方法完成最终采样。
    # grammar_output 包含结构化输出约束（如 JSON schema）的处理结果，
    # 此方法将其与模型 logits 结合，产出满足约束的采样 token。
    def sample_tokens(
        self, grammar_output: GrammarOutput
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput:
        """Should be called immediately after execute_model iff it returned None."""
        raise NotImplementedError

    def get_cache_block_size_bytes(self) -> int:
        """Return the size of a single cache block, in bytes. Used in
        speculative decoding.
        """
        raise NotImplementedError

    def add_lora(self, lora_request: LoRARequest) -> bool:
        raise NotImplementedError

    def remove_lora(self, lora_id: int) -> bool:
        raise NotImplementedError

    def pin_lora(self, lora_id: int) -> bool:
        raise NotImplementedError

    def list_loras(self) -> set[int]:
        raise NotImplementedError

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size from model configuration."""
        return self.model_config.get_vocab_size()

    def shutdown(self) -> None:
        """Clean up resources held by the worker."""
        return


# 中文注释：WorkerWrapperBase 是 Worker 在多进程执行器中的进程级包装器。
#
# 【设计背景】
# vLLM V1 采用多进程架构，每个 GPU 对应一个独立的 Worker 进程。
# WorkerWrapperBase 就是这个子进程中的"宿主"对象，负责：
#   1. 懒加载 Worker 实例（先记住类名，后续再实际初始化）
#   2. 管理 Worker 的完整生命周期（环境变量 -> 初始化 -> 设备初始化 -> 推理 -> 关闭）
#   3. 处理多模态缓存等跨 Worker 的共享逻辑
#
# 【生命周期流程】
#   Step 1: __init__() 记录 rpc_rank 和 global_rank
#   Step 2: update_environment_variables() 设置进程环境变量（如 CUDA_VISIBLE_DEVICES）
#   Step 3: init_worker() 加载插件、解析 Worker 类、动态注入扩展类、实例化 Worker
#   Step 4: initialize_from_config() 使用 KV cache 配置初始化 Worker
#   Step 5: init_device() 初始化设备并加载模型
#   Step 6: execute_model() 循环执行推理
#   Step 7: shutdown() 清理资源
#
# 【为什么需要 Wrapper 而不是直接用 Worker】
#   - 进程隔离：Wrapper 确保 Worker 的初始化在正确的进程上下文中进行。
#   - 懒加载：Worker 类名可以在主进程指定，但实际实例化延迟到子进程中。
#   - 动态扩展：Wrapper 支持通过 worker_extension_cls 动态注入额外功能，
#     而不修改 Worker 本身的代码（例如分布式推理的自定义通信逻辑）。
#   - 环境隔离：不同 Worker 进程需要不同的环境变量（如不同的 CUDA 设备）。
class WorkerWrapperBase:
    """
    This class represents one process in an executor/engine. It is responsible
    for lazily initializing the worker and handling the worker's lifecycle.
    We first instantiate the WorkerWrapper, which remembers the worker module
    and class name. Then, when we call `update_environment_variables`, and the
    real initialization happens in `init_worker`.
    """

    def __init__(
        self,
        rpc_rank: int = 0,
        global_rank: int | None = None,
    ) -> None:
        """
        Initialize the worker wrapper with the given vllm_config and rpc_rank.
        Note: rpc_rank is the rank of the worker in the executor. In most cases,
        it is also the rank of the worker in the distributed group. However,
        when multiple executors work together, they can be different.
        e.g. in the case of SPMD-style offline inference with TP=2,
        users can launch 2 engines/executors, each with only 1 worker.
        All workers have rpc_rank=0, but they have different ranks in the TP
        group.
        """
        self.rpc_rank: int = rpc_rank
        self.global_rank: int = self.rpc_rank if global_rank is None else global_rank

        # Initialized after init_worker is called
        self.worker: WorkerBase
        self.vllm_config: VllmConfig

    def shutdown(self) -> None:
        if self.worker is not None:
            self.worker.shutdown()

    def update_environment_variables(
        self,
        envs_list: list[dict[str, str]],
    ) -> None:
        envs = envs_list[self.rpc_rank]
        update_environment_variables(envs)

    @instrument(span_name="Worker init")
    def init_worker(self, all_kwargs: list[dict[str, Any]]) -> None:
        """
        Here we inject some common logic before initializing the worker.
        Arguments are passed to the worker class constructor.
        """
        kwargs = all_kwargs[self.rpc_rank]

        vllm_config: VllmConfig | None = kwargs.get("vllm_config")
        assert vllm_config is not None, (
            "vllm_config is required to initialize the worker"
        )
        self.vllm_config = vllm_config

        vllm_config.enable_trace_function_call_for_thread()

        from vllm.plugins import load_general_plugins

        load_general_plugins()

        parallel_config = vllm_config.parallel_config
        if isinstance(parallel_config.worker_cls, str):
            worker_class: type[WorkerBase] = resolve_obj_by_qualname(
                parallel_config.worker_cls
            )
        else:
            raise ValueError(
                "passing worker_cls is no longer supported. "
                "Please pass keep the class in a separate module "
                "and pass the qualified name of the class as a string."
            )

        if parallel_config.worker_extension_cls:
            worker_extension_cls = resolve_obj_by_qualname(
                parallel_config.worker_extension_cls
            )
            extended_calls = []
            if worker_extension_cls not in worker_class.__bases__:
                # check any conflicts between worker and worker_extension_cls
                for attr in dir(worker_extension_cls):
                    if attr.startswith("__"):
                        continue
                    assert not hasattr(worker_class, attr), (
                        f"Worker class {worker_class} already has an attribute"
                        f" {attr}, which conflicts with the worker"
                        f" extension class {worker_extension_cls}."
                    )
                    if callable(getattr(worker_extension_cls, attr)):
                        extended_calls.append(attr)
                # dynamically inherit the worker extension class
                worker_class.__bases__ = worker_class.__bases__ + (
                    worker_extension_cls,
                )
                logger.info(
                    "Injected %s into %s for extended collective_rpc calls %s",
                    worker_extension_cls,
                    worker_class,
                    extended_calls,
                )

        shared_worker_lock = kwargs.pop("shared_worker_lock", None)
        if shared_worker_lock is None:
            msg = (
                "Missing `shared_worker_lock` argument from executor. "
                "This argument is needed for mm_processor_cache_type='shm'."
            )

            mm_config = vllm_config.model_config.multimodal_config
            if mm_config and mm_config.mm_processor_cache_type == "shm":
                raise ValueError(msg)
            else:
                logger.warning_once(msg)

            self.mm_receiver_cache = None
        else:
            self.mm_receiver_cache = (
                MULTIMODAL_REGISTRY.worker_receiver_cache_from_config(
                    vllm_config,
                    shared_worker_lock,
                )
            )

        with set_current_vllm_config(self.vllm_config):
            # To make vLLM config available during worker initialization
            self.worker = worker_class(**kwargs)

    def initialize_from_config(self, kv_cache_configs: list[Any]) -> None:
        kv_cache_config = kv_cache_configs[self.global_rank]
        assert self.vllm_config is not None
        with set_current_vllm_config(self.vllm_config):
            self.worker.initialize_from_config(kv_cache_config)  # type: ignore

    def init_device(self):
        assert self.vllm_config is not None
        with set_current_vllm_config(self.vllm_config):
            # To make vLLM config available during device initialization
            self.worker.init_device()  # type: ignore

    def __getattr__(self, attr: str):
        return getattr(self.worker, attr)

    def _apply_mm_cache(self, scheduler_output: SchedulerOutput) -> None:
        mm_cache = self.mm_receiver_cache
        if mm_cache is None:
            return

        for req_data in scheduler_output.scheduled_new_reqs:
            req_data.mm_features = mm_cache.get_and_update_features(
                req_data.mm_features
            )

    def execute_model(
        self, scheduler_output: SchedulerOutput
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | None:
        self._apply_mm_cache(scheduler_output)

        return self.worker.execute_model(scheduler_output)

    def reset_mm_cache(self) -> None:
        mm_receiver_cache = self.mm_receiver_cache
        if mm_receiver_cache is not None:
            mm_receiver_cache.clear_cache()

        self.worker.reset_mm_cache()
