# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# =============================================================================
# 模块概述 (Module Overview)
# =============================================================================
# 本模块实现了 LLMEngine 类，它是 vLLM v1 引擎的同步（非异步）封装器。
# 该类作为对外的主要接口，将用户请求转换为 EngineCore 可以处理的格式，
# 并将 EngineCore 的输出转换为用户可读的 RequestOutput。
#
# 核心架构：
#   用户请求 --> InputProcessor --> EngineCoreClient --> EngineCore
#                                                              |
#   用户输出 <-- OutputProcessor <-- EngineCoreClient <-- EngineCore
#
# 关键组件：
#   1. InputProcessor - 将用户输入（PromptType / EngineInput）转换为
#      EngineCoreRequest 格式
#   2. OutputProcessor - 将 EngineCore 的原始输出转换为
#      RequestOutput / PoolingRequestOutput 格式
#   3. EngineCoreClient - 与 EngineCore 通信的客户端（支持同步/异步/多进程模式）
#
# 请求生命周期 (Request Lifecycle):
#   1. 用户调用 add_request() 提交请求
#   2. InputProcessor 处理输入，执行分词等预处理
#   3. 请求通过 EngineCoreClient 发送到 EngineCore
#   4. EngineCore 中的调度器决定何时执行该请求
#   5. 用户调用 step() 获取输出
#   6. EngineCoreClient 从 EngineCore 获取原始输出
#   7. OutputProcessor 将原始输出转换为最终结果并返回给用户
# =============================================================================

import time
from collections.abc import Callable, Mapping
from copy import copy
from typing import Any

import torch.nn as nn
from typing_extensions import TypeVar

import vllm.envs as envs
from vllm.config import ParallelConfig, VllmConfig
from vllm.distributed import stateless_destroy_torch_distributed_process_group
from vllm.distributed.parallel_state import get_dp_group
from vllm.engine.arg_utils import EngineArgs
from vllm.inputs import EngineInput, PromptType
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.outputs import PoolingRequestOutput, RequestOutput
from vllm.pooling_params import PoolingParams
from vllm.renderers import renderer_from_config
from vllm.renderers.inputs.preprocess import extract_prompt_components
from vllm.sampling_params import SamplingParams
from vllm.tasks import SupportedTask
from vllm.tokenizers import TokenizerLike
from vllm.tracing import init_tracer
from vllm.usage.usage_lib import UsageContext
from vllm.v1.engine import EngineCoreRequest, PauseMode
from vllm.v1.engine.core_client import EngineCoreClient
from vllm.v1.engine.input_processor import InputProcessor
from vllm.v1.engine.output_processor import OutputProcessor
from vllm.v1.engine.parallel_sampling import ParentRequest
from vllm.v1.executor import Executor
from vllm.v1.metrics.loggers import StatLoggerFactory, StatLoggerManager
from vllm.v1.metrics.reader import Metric, get_metrics_snapshot
from vllm.v1.metrics.stats import IterationStats
from vllm.v1.utils import record_function_or_nullcontext
from vllm.v1.worker.worker_base import WorkerBase

# 初始化模块日志记录器
logger = init_logger(__name__)

# 类型变量，用于 collective_rpc 和 apply_model 方法的泛型返回值
_R = TypeVar("_R", default=Any)


class LLMEngine:
    """Legacy LLMEngine for backwards compatibility."""

    # =====================================================================
    # 构造函数 __init__
    # =====================================================================
    # 初始化 LLMEngine 实例，完成以下工作：
    #   1. 保存配置并初始化分布式进程组（如果需要）
    #   2. 创建 InputProcessor（输入处理器）用于将用户输入转为 EngineCoreRequest
    #   3. 创建 OutputProcessor（输出处理器）用于将 EngineCore 输出转为 RequestOutput
    #   4. 创建 EngineCoreClient 用于与底层 EngineCore 通信
    #   5. 初始化统计日志管理器（可选）
    #   6. 清除多模态缓存
    #
    # 参数说明：
    #   - vllm_config: 包含所有配置的 VllmConfig 对象
    #   - executor_class: 执行器类（如单进程/多进程/分布式执行器）
    #   - log_stats: 是否记录统计信息
    #   - aggregate_engine_logging: 是否聚合引擎日志
    #   - usage_context: 使用场景上下文
    #   - stat_loggers: 自定义统计日志工厂列表
    #   - mm_registry: 多模态注册表
    #   - multiprocess_mode: 是否使用多进程模式
    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        aggregate_engine_logging: bool = False,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: list[StatLoggerFactory] | None = None,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        multiprocess_mode: bool = False,
    ) -> None:
        # 保存配置引用，方便后续使用
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.observability_config = vllm_config.observability_config

        # 初始化分布式追踪（OpenTelemetry）
        # 如果配置了 OTLP 追踪端点，则初始化追踪器
        tracing_endpoint = self.observability_config.otlp_traces_endpoint
        if tracing_endpoint is not None:
            init_tracer("vllm.llm_engine", tracing_endpoint)

        self.log_stats = log_stats

        # 处理数据并行（Data Parallelism）相关逻辑
        # 数据并行允许模型在多个 GPU 上并行处理不同的请求批次
        parallel_config = vllm_config.parallel_config
        executor_backend = parallel_config.distributed_executor_backend

        # 判断是否使用外部启动器的数据并行模式
        # 在这种模式下，数据并行组由外部进程管理（如 torchrun）
        self.external_launcher_dp = (
            parallel_config.data_parallel_size > 1
            and executor_backend == "external_launcher"
        )
        # important: init dp group before init the engine_core
        # In the decoupled engine case this is handled in EngineCoreProc.
        # 在非多进程模式下，如果启用了数据并行且不是外部启动器模式，
        # 则需要在此处初始化数据并行进程组
        if (
            not multiprocess_mode
            and parallel_config.data_parallel_size > 1
            and not self.external_launcher_dp
        ):
            self.dp_group = parallel_config.stateless_init_dp_group()
        else:
            self.dp_group = None
        # 标记是否需要执行虚拟批次（用于数据并行中的空闲 worker 同步）
        self.should_execute_dummy_batch = False

        # 创建渲染器，用于处理多模态输入（图像、音频等）
        self.renderer = renderer = renderer_from_config(self.vllm_config)

        # Convert EngineInput --> EngineCoreRequest.
        # 创建输入处理器：将用户输入（文本、图像等）转换为 EngineCore 可理解的请求格式
        # 包括分词、多模态输入预处理等
        self.input_processor = InputProcessor(self.vllm_config, renderer)

        # Converts EngineCoreOutputs --> RequestOutput.
        # 创建输出处理器：将 EngineCore 的原始输出（token IDs）转换为用户可读的输出
        # 包括反分词、流式输出控制等
        self.output_processor = OutputProcessor(
            renderer.tokenizer,
            log_stats=self.log_stats,
            stream_interval=self.vllm_config.scheduler_config.stream_interval,
            tracing_enabled=tracing_endpoint is not None,
        )

        # EngineCore (gets EngineCoreRequests and gives EngineCoreOutputs)
        # 创建 EngineCore 客户端：这是与底层推理引擎通信的桥梁
        # 支持多种模式：
        #   - multiprocess_mode=False, asyncio_mode=False: 同步单进程模式
        #   - multiprocess_mode=True: 多进程模式（每个 GPU 一个 worker 进程）
        #   - asyncio_mode=True: 异步模式（用于 AsyncLLMEngine）
        self.engine_core = EngineCoreClient.make_client(
            multiprocess_mode=multiprocess_mode,
            asyncio_mode=False,
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=self.log_stats,
        )

        # 初始化统计日志管理器（可选）
        # 用于记录和输出性能指标（如吞吐量、延迟等）
        self.logger_manager: StatLoggerManager | None = None
        if self.log_stats:
            self.logger_manager = StatLoggerManager(
                vllm_config=vllm_config,
                custom_stat_loggers=stat_loggers,
                enable_default_loggers=log_stats,
                aggregate_engine_logging=aggregate_engine_logging,
            )
            # 记录引擎初始化完成事件
            self.logger_manager.log_engine_initialized()

        if not multiprocess_mode:
            # for v0 compatibility
            # 在非多进程模式下，暴露 model_executor 属性以保持向后兼容性
            self.model_executor = self.engine_core.engine_core.model_executor  # type: ignore

        if self.external_launcher_dp:
            # If we use DP in external launcher mode, we reuse the
            # existing DP group used for data communication.
            # 在外部启动器模式下，复用已有的数据并行进程组
            self.dp_group = get_dp_group().cpu_group

        # Don't keep the dummy data in memory
        # 清除多模态缓存中的临时数据
        self.reset_mm_cache()

    # =====================================================================
    # 工厂方法 from_vllm_config
    # =====================================================================
    # 从 VllmConfig 创建 LLMEngine 实例的便捷方法
    # 自动检测是否启用多进程模式，并根据配置选择合适的执行器
    @classmethod
    def from_vllm_config(
        cls,
        vllm_config: VllmConfig,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: list[StatLoggerFactory] | None = None,
        disable_log_stats: bool = False,
    ) -> "LLMEngine":
        return cls(
            vllm_config=vllm_config,
            executor_class=Executor.get_class(vllm_config),
            log_stats=(not disable_log_stats),
            usage_context=usage_context,
            stat_loggers=stat_loggers,
            multiprocess_mode=envs.VLLM_ENABLE_V1_MULTIPROCESSING,
        )

    # =====================================================================
    # 工厂方法 from_engine_args
    # =====================================================================
    # 从命令行参数创建 LLMEngine 实例的便捷方法
    # 这是最常用的创建方式，通常由 CLI 或 API 服务器调用
    @classmethod
    def from_engine_args(
        cls,
        engine_args: EngineArgs,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: list[StatLoggerFactory] | None = None,
        enable_multiprocessing: bool = False,
    ) -> "LLMEngine":
        """Creates an LLM engine from the engine arguments."""

        # 从引擎参数创建配置对象
        vllm_config = engine_args.create_engine_config(usage_context)
        # 根据配置选择合适的执行器类
        executor_class = Executor.get_class(vllm_config)

        # 检查环境变量是否强制启用多进程模式
        if envs.VLLM_ENABLE_V1_MULTIPROCESSING:
            logger.debug("Enabling multiprocessing for LLMEngine.")
            enable_multiprocessing = True

        # Create the LLMEngine.
        return cls(
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=not engine_args.disable_log_stats,
            usage_context=usage_context,
            stat_loggers=stat_loggers,
            multiprocess_mode=enable_multiprocessing,
        )

    # =====================================================================
    # get_num_unfinished_requests
    # =====================================================================
    # 获取未完成的请求数量
    # 这个数字包括正在等待调度、正在执行、正在生成输出的请求
    def get_num_unfinished_requests(self) -> int:
        return self.output_processor.get_num_unfinished_requests()

    # =====================================================================
    # has_unfinished_requests
    # =====================================================================
    # 检查是否还有未完成的请求
    # 用于主循环判断是否需要继续调用 step()
    #
    # 逻辑：
    #   1. 首先检查 OutputProcessor 中是否有未完成的请求
    #   2. 如果没有数据并行组，还需要检查 EngineCore 中是否有运行中的数据并行引擎
    #   3. 如果有数据并行组，需要在所有并行 worker 间同步状态
    def has_unfinished_requests(self) -> bool:
        has_unfinished = self.output_processor.has_unfinished_requests()
        if self.dp_group is None:
            return has_unfinished or self.engine_core.dp_engines_running()
        return self.has_unfinished_requests_dp(has_unfinished)

    # =====================================================================
    # has_unfinished_requests_dp
    # =====================================================================
    # 在数据并行模式下检查是否有未完成的请求
    # 需要在所有数据并行 worker 间同步状态（使用集合通信）
    #
    # 重要：如果某个 worker 自己没有未完成请求，但其他 worker 有，
    # 则该 worker 需要执行虚拟批次以保持同步
    def has_unfinished_requests_dp(self, has_unfinished: bool) -> bool:
        aggregated_has_unfinished = ParallelConfig.has_unfinished_dp(
            self.dp_group, has_unfinished
        )
        if not has_unfinished and aggregated_has_unfinished:
            self.should_execute_dummy_batch = True
        return aggregated_has_unfinished

    # =====================================================================
    # get_supported_tasks
    # =====================================================================
    # 获取当前模型支持的任务类型（如文本生成、嵌入、分类等）
    # 结果会被缓存以避免重复查询
    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        if not hasattr(self, "_supported_tasks"):
            # Cache the result
            self._supported_tasks = self.engine_core.get_supported_tasks()

        return self._supported_tasks

    # =====================================================================
    # abort_request
    # =====================================================================
    # 中止指定的请求
    # 同时从 OutputProcessor（输出处理器）和 EngineCore（引擎核心）中移除请求
    #
    # 参数：
    #   - request_ids: 要中止的请求 ID 列表
    #   - internal: 是否为内部中止（例如由停止字符串触发）
    def abort_request(self, request_ids: list[str], internal: bool = False) -> None:
        """Remove request_ids from EngineCore and Detokenizer."""

        # 先从输出处理器中移除，返回实际需要中止的请求 ID
        request_ids = self.output_processor.abort_requests(request_ids, internal)
        # 再通知 EngineCore 中止这些请求
        self.engine_core.abort_requests(request_ids)

    # =====================================================================
    # add_request - 添加请求的核心方法
    # =====================================================================
    # 这是用户提交推理请求的主要入口点
    #
    # 请求处理流程：
    #   1. 验证 request_id 类型
    #   2. 如果 prompt 是 EngineCoreRequest（已处理的请求），直接使用
    #   3. 否则，通过 InputProcessor 处理原始输入（分词、预处理等）
    #   4. 处理并行采样（n > 1 的情况，即从同一 prompt 生成多个候选）
    #   5. 为每个子请求创建输出处理器状态
    #   6. 将请求发送到 EngineCore
    #
    # 参数说明：
    #   - request_id: 请求的唯一标识符
    #   - prompt: 输入提示，可以是已处理的 EngineCoreRequest 或原始 PromptType
    #   - params: 采样参数（SamplingParams）或池化参数（PoolingParams）
    #   - arrival_time: 请求到达时间
    #   - lora_request: LoRA 适配器请求
    #   - tokenization_kwargs: 分词相关的额外参数
    #   - trace_headers: 分布式追踪头部
    #   - priority: 请求优先级
    #   - prompt_text: 原始提示文本（用于日志和追踪）
    def add_request(
        self,
        request_id: str,
        prompt: EngineCoreRequest | PromptType | EngineInput,
        params: SamplingParams | PoolingParams,
        arrival_time: float | None = None,
        lora_request: LoRARequest | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
        trace_headers: Mapping[str, str] | None = None,
        priority: int = 0,
        prompt_text: str | None = None,
    ) -> str:
        # Validate the request_id type.
        if not isinstance(request_id, str):
            raise TypeError(f"request_id must be a string, got {type(request_id)}")

        # Process raw inputs into the request.
        # 检查输入是否已经是处理过的 EngineCoreRequest
        if isinstance(prompt, EngineCoreRequest):
            logger.warning_once(
                "Passing EngineCoreRequest to LLMEngine.generate() and .add_requests() "
                "is deprecated and will be removed in v0.18. You should instead pass "
                "the outputs of Renderer.render_cmpl() or Renderer.render_chat()."
            )

            request = prompt
            if request_id != request.request_id:
                logger.warning_once(
                    "LLMEngine.add_request() was passed a request_id parameter that "
                    "does not match the EngineCoreRequest.request_id attribute. The "
                    "latter will be used, and the former will be ignored."
                )
        else:
            # 如果是原始输入，需要通过 InputProcessor 进行处理
            # 处理过程包括：分词、多模态输入处理、参数验证等
            request = self.input_processor.process_inputs(
                request_id,
                prompt,
                params,
                supported_tasks=self.get_supported_tasks(),
                arrival_time=arrival_time,
                lora_request=lora_request,
                tokenization_kwargs=tokenization_kwargs,
                trace_headers=trace_headers,
                priority=priority,
            )
            # 提取提示文本组件（用于日志和追踪）
            prompt_text, _, _ = extract_prompt_components(self.model_config, prompt)

        # 为请求分配唯一 ID（如果尚未分配）
        self.input_processor.assign_request_id(request)

        req_id = request.request_id

        # Use cloned params that may have been updated in process_inputs()
        params = request.params

        # 获取并行采样数量 n
        # n > 1 表示从同一 prompt 生成多个候选回复
        n = params.n if isinstance(params, SamplingParams) else 1

        if n == 1:
            # 普通情况：n=1，只生成一个回复
            # Make a new RequestState and queue.
            # 在输出处理器中创建请求状态
            self.output_processor.add_request(request, prompt_text, None, 0)
            # Add the request to EngineCore.
            # 将请求发送到 EngineCore 进行处理
            self.engine_core.add_request(request)
            return req_id

        # Fan out child requests (for n>1).
        # 并行采样：从同一 prompt 生成 n 个候选回复
        # 创建一个 ParentRequest 来管理所有子请求
        parent_req = ParentRequest(request)
        for idx in range(n):
            request_id, child_params = parent_req.get_child_info(idx)
            # 复制请求对象（最后一个子请求复用原始对象以节省内存）
            child_request = request if idx == n - 1 else copy(request)
            child_request.request_id = request_id
            child_request.sampling_params = child_params

            # Make a new RequestState and queue.
            # 为每个子请求创建独立的输出处理状态
            self.output_processor.add_request(
                child_request, prompt_text, parent_req, idx
            )
            # Add the request to EngineCore.
            # 将每个子请求发送到 EngineCore
            self.engine_core.add_request(child_request)

        return req_id

    # =====================================================================
    # step - 推理引擎的核心步进方法
    # =====================================================================
    # 这是同步推理循环的核心，每次调用执行一个推理步骤
    #
    # 执行流程：
    #   1. 如果需要执行虚拟批次（数据并行同步），则执行并返回空结果
    #   2. 从 EngineCore 获取原始输出（EngineCoreOutput）
    #   3. 通过 OutputProcessor 处理输出，转换为 RequestOutput
    #   4. 中止因停止字符串而完成的请求
    #   5. 记录统计信息（如果启用）
    #   6. 返回处理后的输出列表
    #
    # 返回值：RequestOutput 或 PoolingRequestOutput 的列表
    def step(self) -> list[RequestOutput | PoolingRequestOutput]:
        # 处理数据并行模式下的虚拟批次
        # 当某些 worker 没有请求但其他 worker 有时，需要执行虚拟批次保持同步
        if self.should_execute_dummy_batch:
            self.should_execute_dummy_batch = False
            self.engine_core.execute_dummy_batch()
            return []

        # 1) Get EngineCoreOutput from the EngineCore.
        # 从 EngineCore 获取本轮迭代的原始输出
        # 输出包括：生成的 token、完成状态、调度器统计信息等
        with record_function_or_nullcontext("llm_engine step: get_output"):
            outputs = self.engine_core.get_output()

        # 2) Process EngineCoreOutputs.
        # 处理原始输出：反分词、检测停止条件、组装最终输出
        with record_function_or_nullcontext("llm_engine step: process_outputs"):
            # 创建迭代统计对象（如果启用日志记录）
            iteration_stats = IterationStats() if self.log_stats else None
            processed_outputs = self.output_processor.process_outputs(
                outputs.outputs,
                engine_core_timestamp=outputs.timestamp,
                iteration_stats=iteration_stats,
            )
            # 更新调度器统计信息
            self.output_processor.update_scheduler_stats(outputs.scheduler_stats)

        # 3) Abort any reqs that finished due to stop strings.
        # 中止因停止字符串匹配而完成的请求
        # （这些请求的输出已经生成完毕，需要通知 EngineCore 释放资源）
        with record_function_or_nullcontext("llm_engine step: abort_requests"):
            self.engine_core.abort_requests(processed_outputs.reqs_to_abort)

        # 4) Record stats
        # 记录统计信息（吞吐量、延迟、缓存命中率等）
        with record_function_or_nullcontext("llm_engine step: record_stats"):
            if (
                self.logger_manager is not None
                and outputs.scheduler_stats is not None
                and len(outputs.outputs) > 0
            ):
                self.logger_manager.record(
                    scheduler_stats=outputs.scheduler_stats,
                    iteration_stats=iteration_stats,
                    mm_cache_stats=self.renderer.stat_mm_cache(),
                )
                # 按时间间隔输出日志
                self.do_log_stats_with_interval()

        return processed_outputs.request_outputs

    # =====================================================================
    # 性能分析（Profiling）方法
    # =====================================================================
    # start_profile: 开始性能分析，profile_prefix 用于标识分析文件
    # stop_profile: 停止性能分析
    def start_profile(self, profile_prefix: str | None = None):
        self.engine_core.profile(True, profile_prefix)

    def stop_profile(self):
        self.engine_core.profile(False)

    # =====================================================================
    # 缓存管理方法
    # =====================================================================

    # reset_mm_cache: 清除多模态缓存（图像、音频的编码结果缓存）
    def reset_mm_cache(self):
        self.renderer.clear_mm_cache()
        self.engine_core.reset_mm_cache()

    # reset_prefix_cache: 重置前缀缓存
    # 前缀缓存用于加速相同前缀的请求（如系统提示）
    # 参数：
    #   - reset_running_requests: 是否同时重置正在运行请求的缓存
    #   - reset_connector: 是否重置连接器
    def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        return self.engine_core.reset_prefix_cache(
            reset_running_requests, reset_connector
        )

    # reset_encoder_cache: 重置编码器缓存
    # 当模型权重更新后，需要调用此方法以确保不使用旧权重计算的视觉嵌入
    def reset_encoder_cache(self) -> None:
        """Reset the encoder cache to invalidate all cached encoder outputs.

        This should be called when model weights are updated to ensure
        stale vision embeddings computed with old weights are not reused.
        """
        self.engine_core.reset_encoder_cache()

    # =====================================================================
    # 睡眠/唤醒方法（用于动态资源管理）
    # =====================================================================

    # sleep: 让引擎进入睡眠状态以释放 GPU 内存
    # 参数：
    #   - level: 睡眠级别（1 = 释放 KV 缓存，2 = 释放模型权重）
    #   - mode: 暂停模式（"abort" = 中止所有请求，"pause" = 暂停请求）
    def sleep(self, level: int = 1, mode: PauseMode = "abort"):
        if level >= 1:
            self.renderer.clear_mm_cache()
        self.engine_core.sleep(level, mode)

        if self.logger_manager is not None:
            self.logger_manager.record_sleep_state(1, level)

    # wake_up: 唤醒引擎
    # 参数：
    #   - tags: 要唤醒的资源标签（如 "weights", "kv_cache"）
    def wake_up(self, tags: list[str] | None = None):
        self.engine_core.wake_up(tags)

        if self.logger_manager is not None:
            self.logger_manager.record_sleep_state(0, 0)

    # is_sleeping: 检查引擎是否处于睡眠状态
    def is_sleeping(self) -> bool:
        return self.engine_core.is_sleeping()

    # =====================================================================
    # 指标和统计方法
    # =====================================================================

    # get_metrics: 获取当前的性能指标快照
    # 返回 Metric 对象列表，包含各种计数器和直方图
    def get_metrics(self) -> list[Metric]:
        assert self.log_stats, "Stat logging disabled"
        return get_metrics_snapshot()

    # =====================================================================
    # 分词器访问方法
    # =====================================================================

    # tokenizer 属性：直接访问分词器（可能返回 None）
    @property
    def tokenizer(self) -> TokenizerLike | None:
        return self.renderer.tokenizer

    # get_tokenizer: 获取分词器（保证不返回 None）
    def get_tokenizer(self) -> TokenizerLike:
        return self.renderer.get_tokenizer()

    # =====================================================================
    # 统计日志方法
    # =====================================================================

    # do_log_stats: 立即输出统计日志
    def do_log_stats(self) -> None:
        """Log stats if logging is enabled."""
        if self.logger_manager:
            self.logger_manager.log()

    # do_log_stats_with_interval: 按时间间隔输出统计日志
    # 避免过于频繁地输出日志，使用 VLLM_LOG_STATS_INTERVAL 环境变量控制间隔
    def do_log_stats_with_interval(self) -> None:
        """Log stats when the time interval has passed."""
        now = time.time()
        if not hasattr(self, "_last_log_time"):
            self._last_log_time = now
        if now - self._last_log_time >= envs.VLLM_LOG_STATS_INTERVAL:
            self.do_log_stats()
            self._last_log_time = now

    # =====================================================================
    # LoRA 适配器管理方法
    # =====================================================================
    # LoRA (Low-Rank Adaptation) 是一种高效的模型微调技术
    # 这些方法允许在运行时动态加载、卸载和管理 LoRA 适配器

    # add_lora: 加载新的 LoRA 适配器
    def add_lora(self, lora_request: LoRARequest) -> bool:
        """Load a new LoRA adapter into the engine for future requests."""
        return self.engine_core.add_lora(lora_request)

    # remove_lora: 卸载已加载的 LoRA 适配器
    def remove_lora(self, lora_id: int) -> bool:
        """Remove an already loaded LoRA adapter."""
        return self.engine_core.remove_lora(lora_id)

    # list_loras: 列出所有已注册的 LoRA 适配器 ID
    def list_loras(self) -> set[int]:
        """List all registered adapters."""
        return self.engine_core.list_loras()

    # pin_lora: 钉住 LoRA 适配器，防止其被驱逐
    # 当内存不足时，未钉住的适配器可能被卸载以腾出空间
    def pin_lora(self, lora_id: int) -> bool:
        """Prevent an adapter from being evicted."""
        return self.engine_core.pin_lora(lora_id)

    # =====================================================================
    # 集体 RPC 和模型操作方法
    # =====================================================================

    # collective_rpc: 在所有 worker 上执行 RPC 调用
    # 这是一种分布式操作机制，可以在所有 GPU worker 上执行相同的函数
    # 参数：
    #   - method: 方法名字符串或可调用对象
    #   - timeout: 超时时间
    #   - args: 位置参数
    #   - kwargs: 关键字参数
    # 返回值：每个 worker 的返回值列表
    def collective_rpc(
        self,
        method: str | Callable[[WorkerBase], _R],
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
    ) -> list[_R]:
        return self.engine_core.collective_rpc(method, timeout, args, kwargs)

    # apply_model: 在所有 worker 的模型上应用函数
    # 这是 collective_rpc 的便捷封装，专门用于操作模型
    # 例如：加载权重、修改模型参数等
    def apply_model(self, func: Callable[[nn.Module], _R]) -> list[_R]:
        return self.collective_rpc("apply_model", args=(func,))

    # =====================================================================
    # 析构函数 __del__
    # =====================================================================
    # 清理数据并行进程组
    # 注意：外部启动器模式下的进程组不由本类管理
    def __del__(self):
        dp_group = getattr(self, "dp_group", None)
        if dp_group is not None and not self.external_launcher_dp:
            stateless_destroy_torch_distributed_process_group(dp_group)
