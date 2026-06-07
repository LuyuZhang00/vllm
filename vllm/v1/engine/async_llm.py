# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# ==============================================================================
# 模块概述: vLLM v1 引擎的异步前端 (Async Frontend)
# ==============================================================================
# 本模块实现了 AsyncLLM 类，它是 vLLM v1 引擎的核心异步接口。
#
# 架构设计:
#   AsyncLLM 是运行在 API 服务器进程中的异步前端，负责:
#   1) 接收来自 API 层 (如 OpenAI 兼容 API) 的请求
#   2) 将用户输入 (prompt) 转换为 EngineCore 能理解的请求格式
#   3) 通过 ZMQ IPC 将请求发送到独立的 EngineCore 进程
#   4) 从 EngineCore 接收输出并转换为用户可读的格式
#   5) 通过异步生成器 (AsyncGenerator) 将结果流式返回给调用方
#
# 请求生命周期:
#   用户请求 -> add_request()/generate() -> InputProcessor (tokenize)
#   -> EngineCoreClient (IPC) -> EngineCore 进程 -> Scheduler -> Executor
#   -> 输出通过 IPC 返回 -> OutputProcessor -> RequestOutputCollector
#   -> 调用方通过 AsyncGenerator 迭代获取结果
#
# 核心组件:
#   - InputProcessor: 将 EngineInput/PromptType 转换为 EngineCoreRequest
#   - OutputProcessor: 将 EngineCoreOutput 转换为 RequestOutput
#   - EngineCoreClient: 与 EngineCore 进程进行 IPC 通信
#   - RequestOutputCollector: 每个请求的输出队列，用于流式返回结果
#   - output_handler: 后台异步任务，持续从 EngineCore 拉取输出并分发
# ==============================================================================

import asyncio
import os
import socket
import time
import warnings
from collections.abc import AsyncGenerator, Iterable, Mapping
from copy import copy
from typing import Any

import torch

import vllm.envs as envs
from vllm import TokensPrompt
from vllm.config import VllmConfig
from vllm.distributed.weight_transfer.base import (
    WeightTransferInitRequest,
    WeightTransferUpdateRequest,
)
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.protocol import EngineClient, StreamingInput
from vllm.entrypoints.serve.elastic_ep.middleware import set_scaling_elastic_ep
from vllm.inputs import EngineInput, PromptType
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.outputs import STREAM_FINISHED, PoolingRequestOutput, RequestOutput
from vllm.pooling_params import PoolingParams
from vllm.renderers import renderer_from_config
from vllm.renderers.inputs.preprocess import extract_prompt_components
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.tasks import SupportedTask
from vllm.tokenizers import TokenizerLike
from vllm.tracing import init_tracer
from vllm.transformers_utils.config import maybe_register_config_serialize_by_value
from vllm.usage.usage_lib import UsageContext
from vllm.utils.async_utils import cancel_task_threadsafe
from vllm.utils.collection_utils import as_list
from vllm.v1.engine import EngineCoreRequest, PauseMode
from vllm.v1.engine.core_client import EngineCoreClient
from vllm.v1.engine.exceptions import EngineDeadError, EngineGenerateError
from vllm.v1.engine.input_processor import InputProcessor
from vllm.v1.engine.output_processor import OutputProcessor, RequestOutputCollector
from vllm.v1.engine.parallel_sampling import ParentRequest
from vllm.v1.executor import Executor
from vllm.v1.metrics.loggers import (
    StatLoggerFactory,
    StatLoggerManager,
    load_stat_logger_plugin_factories,
)
from vllm.v1.metrics.prometheus import shutdown_prometheus
from vllm.v1.metrics.stats import IterationStats

logger = init_logger(__name__)


class InputStreamError(Exception):
    """Wrapper for errors from the input stream generator.

    This is used to propagate errors from the user's input generator
    without wrapping them in EngineGenerateError.
    """
    # 输入流错误包装类
    # 当用户提供的异步输入流生成器 (StreamingInput) 发生异常时，
    # 使用此类包装原始异常。这样在 generate() 方法中可以区分
    # "输入流本身的错误" 和 "引擎内部错误"，
    # 避免输入流的错误被错误地包装为 EngineGenerateError。
    # 原始异常存储在 self.cause 中，调用方可直接 re-raise 原始异常。

    def __init__(self, cause: Exception):
        self.cause = cause
        super().__init__(str(cause))


class AsyncLLM(EngineClient):
    """An asynchronous wrapper for the vLLM engine."""
    # AsyncLLM 类: vLLM v1 引擎的异步前端封装
    #
    # 核心职责:
    #   1) 作为 API 服务器与 EngineCore 之间的桥梁
    #   2) 管理请求的完整生命周期 (添加、生成、中止)
    #   3) 协调输入处理 (tokenization) 和输出处理 (detokenization)
    #   4) 通过后台 output_handler 任务实现高效的流式输出
    #
    # 线程/进程模型:
    #   - AsyncLLM 运行在主进程的 asyncio 事件循环中
    #   - EngineCore 运行在独立的后台进程中
    #   - 两者通过 ZMQ IPC 进行通信 (EngineCoreClient 负责)
    #
    # 关键组件关系:
    #   AsyncLLM
    #     ├── InputProcessor    (输入处理: prompt -> token -> EngineCoreRequest)
    #     ├── OutputProcessor   (输出处理: EngineCoreOutput -> RequestOutput)
    #     ├── EngineCoreClient  (IPC 通信: 与 EngineCore 进程交互)
    #     └── output_handler    (后台任务: 持续拉取并分发输出)

    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        log_requests: bool = True,
        start_engine_loop: bool = True,
        stat_loggers: list[StatLoggerFactory] | None = None,
        aggregate_engine_logging: bool = False,
        client_addresses: dict[str, Any] | None = None,
        client_count: int = 1,
        client_index: int = 0,
    ) -> None:
        """
        Create an AsyncLLM.

        Args:
            vllm_config: global configuration.
            executor_class: an Executor impl, e.g. MultiprocExecutor.
            log_stats: Whether to log stats.
            usage_context: Usage context of the LLM.
            mm_registry: Multi-modal registry.
            log_requests: Whether to log requests.
            start_engine_loop: Whether to start the engine loop.
            stat_loggers: customized stat loggers for the engine.
                If not provided, default stat loggers will be used.
                PLEASE BE AWARE THAT STAT LOGGER IS NOT STABLE
                IN V1, AND ITS BASE CLASS INTERFACE MIGHT CHANGE.

        Returns:
            None
        """
        # Ensure we can serialize custom transformer configs
        maybe_register_config_serialize_by_value()

        # 保存全局配置和子配置的引用
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.observability_config = vllm_config.observability_config

        # 初始化分布式追踪 (OpenTelemetry)
        tracing_endpoint = self.observability_config.otlp_traces_endpoint
        if tracing_endpoint is not None:
            init_tracer("vllm.llm_engine", tracing_endpoint)

        self.log_requests = log_requests

        # 加载自定义统计日志器 (包括插件形式的日志器)
        custom_stat_loggers = list(stat_loggers or [])
        custom_stat_loggers.extend(load_stat_logger_plugin_factories())

        # 如果有自定义日志器但 log_stats 为 False，仍启用日志记录
        has_custom_loggers = bool(custom_stat_loggers)
        self.log_stats = log_stats or has_custom_loggers
        if not log_stats and has_custom_loggers:
            logger.info(
                "AsyncLLM created with log_stats=False, "
                "but custom stat loggers were found; "
                "enabling logging without default stat loggers."
            )

        # 初始化渲染器 (Renderer): 负责将原始输入渲染为引擎可处理的格式
        # 包括 tokenizer 管理、多模态输入预处理等
        self.renderer = renderer = renderer_from_config(self.vllm_config)

        # 输入处理器: 将 EngineInput/PromptType 转换为 EngineCoreRequest
        # 主要完成 tokenization 和输入验证
        # Convert EngineInput --> EngineCoreRequest.
        self.input_processor = InputProcessor(self.vllm_config, renderer)

        # 输出处理器: 将 EngineCore 的原始输出转换为 RequestOutput
        # 主要完成 detokenization、统计信息收集等
        # Converts EngineCoreOutputs --> RequestOutput.
        self.output_processor = OutputProcessor(
            renderer.tokenizer,
            log_stats=self.log_stats,
            stream_interval=self.vllm_config.scheduler_config.stream_interval,
            tracing_enabled=tracing_endpoint is not None,
        )

        # 创建 EngineCore 的异步多进程客户端
        # 这会在后台启动 EngineCore 进程 (包含 Scheduler + Executor)
        # EngineCore (starts the engine in background process).
        self.engine_core = EngineCoreClient.make_async_mp_client(
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=self.log_stats,
            client_addresses=client_addresses,
            client_count=client_count,
            client_index=client_index,
        )

        # 初始化统计日志管理器 (Prometheus 指标等)
        # Loggers.
        self.logger_manager: StatLoggerManager | None = None
        if self.log_stats:
            self.logger_manager = StatLoggerManager(
                vllm_config=vllm_config,
                engine_idxs=self.engine_core.engine_ranks_managed,
                custom_stat_loggers=custom_stat_loggers,
                enable_default_loggers=log_stats,
                client_count=client_count,
                aggregate_engine_logging=aggregate_engine_logging,
            )
            self.logger_manager.log_engine_initialized()

        self._client_count = client_count

        # 输出处理后台任务 (output_handler)
        # 该任务持续从 EngineCore 拉取输出，处理后分发到各个请求的队列中
        # 如果当前已在 asyncio 事件循环中，立即启动该任务
        self.output_handler: asyncio.Task | None = None
        try:
            # Start output handler eagerly if we are in the asyncio eventloop.
            asyncio.get_running_loop()
            self._run_output_handler()
        except RuntimeError:
            # 不在事件循环中 (例如在同步 __init__ 调用中)
            # output_handler 会在第一次调用 add_request() 时启动
            pass

        # 初始化 PyTorch CPU 性能分析器 (可选)
        # 用于分析 AsyncLLM 前端的 CPU 瓶颈，输出 TensorBoard 格式
        if (
            vllm_config.profiler_config.profiler == "torch"
            and not vllm_config.profiler_config.ignore_frontend
        ):
            profiler_dir = vllm_config.profiler_config.torch_profiler_dir
            logger.info(
                "Torch profiler enabled. AsyncLLM CPU traces will be collected under %s",  # noqa: E501
                profiler_dir,
            )
            worker_name = f"{socket.gethostname()}_{os.getpid()}.async_llm"
            self.profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                ],
                with_stack=vllm_config.profiler_config.torch_profiler_with_stack,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    profiler_dir,
                    worker_name=worker_name,
                    use_gzip=vllm_config.profiler_config.torch_profiler_use_gzip,
                ),
            )
        else:
            self.profiler = None

    @classmethod
    def from_vllm_config(
        cls,
        vllm_config: VllmConfig,
        start_engine_loop: bool = True,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: list[StatLoggerFactory] | None = None,
        enable_log_requests: bool = False,
        aggregate_engine_logging: bool = False,
        disable_log_stats: bool = False,
        client_addresses: dict[str, Any] | None = None,
        client_count: int = 1,
        client_index: int = 0,
    ) -> "AsyncLLM":
        # 工厂方法: 从已创建的 VllmConfig 实例化 AsyncLLM
        # 这是从 VllmConfig 构造 AsyncLLM 的标准方式
        # Create the LLMEngine.
        return cls(
            vllm_config=vllm_config,
            executor_class=Executor.get_class(vllm_config),
            start_engine_loop=start_engine_loop,
            stat_loggers=stat_loggers,
            log_requests=enable_log_requests,
            log_stats=not disable_log_stats,
            aggregate_engine_logging=aggregate_engine_logging,
            usage_context=usage_context,
            client_addresses=client_addresses,
            client_count=client_count,
            client_index=client_index,
        )

    @classmethod
    def from_engine_args(
        cls,
        engine_args: AsyncEngineArgs,
        start_engine_loop: bool = True,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: list[StatLoggerFactory] | None = None,
    ) -> "AsyncLLM":
        """Create an AsyncLLM from the EngineArgs."""
        # 工厂方法: 从命令行参数 (AsyncEngineArgs) 创建 AsyncLLM
        # 这是 vllm serve CLI 入口使用的方式
        # 流程: EngineArgs -> VllmConfig -> AsyncLLM

        # Create the engine configs.
        vllm_config = engine_args.create_engine_config(usage_context)
        executor_class = Executor.get_class(vllm_config)

        # Create the AsyncLLM.
        return cls(
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_requests=engine_args.enable_log_requests,
            log_stats=not engine_args.disable_log_stats,
            start_engine_loop=start_engine_loop,
            usage_context=usage_context,
            stat_loggers=stat_loggers,
        )

    def __del__(self):
        # 析构函数: 确保资源被正确释放
        self.shutdown()

    def shutdown(self, timeout: float | None = None) -> None:
        """Shutdown, cleaning up the background proc and IPC."""
        # 关闭 AsyncLLM，释放所有资源
        # 关闭顺序:
        #   1) 关闭 Prometheus 指标导出
        #   2) 关闭渲染器 (释放 tokenizer 等资源)
        #   3) 关闭 EngineCore (终止后台进程、释放 GPU 资源)
        #   4) 取消 output_handler 后台任务
        shutdown_prometheus()

        if renderer := getattr(self, "renderer", None):
            renderer.shutdown()

        if engine_core := getattr(self, "engine_core", None):
            engine_core.shutdown(timeout=timeout)

        handler = getattr(self, "output_handler", None)
        if handler is not None:
            cancel_task_threadsafe(handler)

    async def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        # 获取引擎支持的任务类型 (如 text_generation, embedding 等)
        # 结果会被缓存，避免重复查询 EngineCore
        if not hasattr(self, "_supported_tasks"):
            # Cache the result
            self._supported_tasks = await self.engine_core.get_supported_tasks_async()

        return self._supported_tasks

    async def add_request(
        self,
        request_id: str,
        prompt: EngineCoreRequest
        | PromptType
        | EngineInput
        | AsyncGenerator[StreamingInput, None],
        params: SamplingParams | PoolingParams,
        arrival_time: float | None = None,
        lora_request: LoRARequest | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
        trace_headers: Mapping[str, str] | None = None,
        priority: int = 0,
        data_parallel_rank: int | None = None,
        prompt_text: str | None = None,
        reasoning_ended: bool | None = None,
        reasoning_parser_kwargs: dict[str, Any] | None = None,
    ) -> RequestOutputCollector:
        """Add new request to the AsyncLLM."""
        # 添加新请求到 AsyncLLM
        #
        # 请求处理流程:
        #   1) 检查引擎状态 (是否已出错)
        #   2) 根据输入类型分派:
        #      - AsyncGenerator (流式输入): 调用 _add_streaming_input_request
        #      - EngineCoreRequest (已处理的请求): 直接使用 (已废弃)
        #      - 其他 (PromptType/EngineInput): 通过 InputProcessor 处理
        #   3) 创建 RequestOutputCollector (每请求的输出队列)
        #   4) 对于 n>1 的请求，扇出 (fan out) 为多个子请求
        #   5) 将请求添加到 OutputProcessor 和 EngineCore
        #
        # 返回: RequestOutputCollector，调用方可通过其 get() 方法获取输出

        if self.errored:
            raise EngineDeadError()

        is_pooling = isinstance(params, PoolingParams)

        # 校验: kv_sharing_fast_prefill 模式下不支持 prompt_logprobs
        if (
            self.vllm_config.cache_config.kv_sharing_fast_prefill
            and not is_pooling
            and params.prompt_logprobs
        ):
            raise ValueError(
                "--kv-sharing-fast-prefill produces incorrect logprobs for "
                "prompt tokens, please disable it when the requests need "
                "prompt logprobs"
            )

        # 分支 1: 流式输入 (AsyncGenerator)
        # 用于支持逐块输入的场景 (如实时语音流)
        if isinstance(prompt, AsyncGenerator):
            if reasoning_ended is not None or reasoning_parser_kwargs is not None:
                raise NotImplementedError

            # Streaming input case.
            return await self._add_streaming_input_request(
                request_id,
                prompt,
                params,
                arrival_time,
                lora_request,
                tokenization_kwargs,
                trace_headers,
                priority,
                data_parallel_rank,
            )

        # 分支 2: EngineCoreRequest (已废弃，直接传入已处理的请求)
        # Convert Input --> Request.
        if isinstance(prompt, EngineCoreRequest):
            logger.warning_once(
                "Passing EngineCoreRequest to AsyncLLM.generate() and .add_requests() "
                "is deprecated and will be removed in v0.18. You should instead pass "
                "the outputs of Renderer.render_cmpl() or Renderer.render_chat()."
            )

            request = prompt
            if request_id != request.request_id:
                logger.warning_once(
                    "AsyncLLM.add_request() was passed a request_id parameter that "
                    "does not match the EngineCoreRequest.request_id attribute. The "
                    "latter will be used, and the former will be ignored."
                )
        else:
            # 分支 3: 标准输入 (PromptType/EngineInput)
            # 通过 InputProcessor 处理输入: tokenization、验证、构建 EngineCoreRequest
            request = self.input_processor.process_inputs(
                request_id,
                prompt,
                params,
                supported_tasks=await self.get_supported_tasks(),
                arrival_time=arrival_time,
                lora_request=lora_request,
                tokenization_kwargs=tokenization_kwargs,
                trace_headers=trace_headers,
                priority=priority,
                data_parallel_rank=data_parallel_rank,
            )
            prompt_text, _, _ = extract_prompt_components(self.model_config, prompt)

        # 设置推理相关参数 (用于 reasoning model，如 DeepSeek-R1)
        if reasoning_ended is not None:
            request.reasoning_ended = reasoning_ended
        if reasoning_parser_kwargs is not None:
            request.reasoning_parser_kwargs = reasoning_parser_kwargs

        self.input_processor.assign_request_id(request)

        # 惰性启动 output_handler: 第一次调用 add_request() 时启动
        # 这样可以在事件循环启动前调用 __init__，方便 OpenAI 服务器优雅处理启动错误
        # We start the output_handler on the first call to add_request() so
        # we can call __init__ before the event loop, which enables us
        # to handle startup failure gracefully in the OpenAI server.
        self._run_output_handler()

        # 为该请求创建输出收集器 (队列)
        # 所有该请求的输出 token 都会被放入此队列
        # Create a new output collector for the request.
        queue = RequestOutputCollector(params.output_kind, request.request_id)

        # 使用可能在 process_inputs() 中被更新的克隆参数
        # Use cloned params that may have been updated in process_inputs()
        params = request.params

        # 简单情况: pooling 请求或 n=1 (单个采样)
        if is_pooling or params.n == 1:
            await self._add_request(request, prompt_text, None, 0, queue)
            return queue

        # 复杂情况: n>1 时需要扇出 (fan out) 为多个子请求
        # 例如 n=3 时，会创建 3 个独立的子请求，共享同一个输出队列
        # 最终结果会被合并后返回给调用方
        parent_params = params
        assert isinstance(parent_params, SamplingParams)

        # Fan out child requests (for n>1).
        parent_request = ParentRequest(request)
        for idx in range(parent_params.n):
            request_id, child_params = parent_request.get_child_info(idx)
            # 最后一个子请求复用原始请求对象 (避免不必要的拷贝)
            child_request = request if idx == parent_params.n - 1 else copy(request)
            child_request.request_id = request_id
            child_request.sampling_params = child_params
            await self._add_request(
                child_request, prompt_text, parent_request, idx, queue
            )
        return queue

    async def _add_request(
        self,
        request: EngineCoreRequest,
        prompt: str | None,
        parent_req: ParentRequest | None,
        index: int,
        queue: RequestOutputCollector,
    ):
        # 内部方法: 将请求同时添加到 OutputProcessor 和 EngineCore
        #
        # 双重注册:
        #   1) OutputProcessor (本进程): 跟踪请求状态，处理输出
        #   2) EngineCore (后台进程): 调度和执行请求
        #
        # Add the request to OutputProcessor (this process).
        self.output_processor.add_request(request, prompt, parent_req, index, queue)

        # Add the EngineCoreRequest to EngineCore (separate process).
        await self.engine_core.add_request_async(request)

        if self.log_requests:
            logger.info("Added request %s.", request.request_id)

    async def _add_streaming_input_request(
        self,
        request_id: str,
        input_stream: AsyncGenerator[StreamingInput, None],
        sampling_params: SamplingParams | PoolingParams,
        arrival_time: float | None = None,
        lora_request: LoRARequest | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
        trace_headers: Mapping[str, str] | None = None,
        priority: int = 0,
        data_parallel_rank: int | None = None,
    ) -> RequestOutputCollector:
        # 流式输入请求处理
        #
        # 用于处理逐块到达的输入 (如实时语音流、渐进式文本输入等)
        # 流程:
        #   1) 验证采样参数 (不支持 pooling、n>1、stop 字符串等)
        #   2) 创建最终请求 (用于标记输入流结束)
        #   3) 启动异步任务 handle_inputs() 消费输入流
        #      - 每收到一个 input_chunk，就创建一个 EngineCoreRequest 并发送
        #      - 输入流结束后发送空的 final_req 作为结束信号
        #   4) 返回输出队列，调用方通过它获取结果
        self._validate_streaming_input_sampling_params(sampling_params)

        inputs = dict(
            supported_tasks=await self.get_supported_tasks(),
            arrival_time=arrival_time,
            lora_request=lora_request,
            tokenization_kwargs=tokenization_kwargs,
            trace_headers=trace_headers,
            priority=priority,
            data_parallel_rank=data_parallel_rank,
        )

        # 克隆 sampling_params 以避免修改原始对象
        if not sampling_params.skip_clone:
            sampling_params = sampling_params.clone()
            sampling_params.skip_clone = True

        # 创建最终请求: 用于验证参数，并在输入流结束后作为结束信号
        # Create request for validation, also used as the finished signal
        # once the input stream is closed.
        final_req = self.input_processor.process_inputs(
            request_id=request_id,
            prompt=TokensPrompt(prompt_token_ids=[0]),
            params=sampling_params,
            **inputs,  # type: ignore[arg-type]
        )
        self.input_processor.assign_request_id(final_req)
        internal_req_id = final_req.request_id

        queue = RequestOutputCollector(sampling_params.output_kind, internal_req_id)

        # 定义异步任务: 持续消费输入流，将每个 chunk 转换为请求并发送
        async def handle_inputs():
            cancelled = False
            try:
                async for input_chunk in input_stream:
                    sp = input_chunk.sampling_params
                    if sp:
                        self._validate_streaming_input_sampling_params(sp)
                    else:
                        sp = sampling_params
                    # TODO(nick): Avoid re-validating reused sampling parameters
                    req = self.input_processor.process_inputs(
                        request_id=internal_req_id,
                        prompt=input_chunk.prompt,
                        params=sp,
                        resumable=True,
                        **inputs,  # type: ignore[arg-type]
                    )
                    req.external_req_id = request_id
                    if req.prompt_embeds is not None:
                        raise ValueError(
                            "prompt_embeds not supported for streaming inputs"
                        )
                    prompt_text, _, _ = extract_prompt_components(
                        self.model_config, input_chunk.prompt
                    )
                    await self._add_request(req, prompt_text, None, 0, queue)
            except (asyncio.CancelledError, GeneratorExit):
                cancelled = True
            except Exception as error:
                # 包装为 InputStreamError，避免被 generate() 中的
                # 通用异常处理器包装为 EngineGenerateError
                # Wrap in InputStreamError so generate() can propagate it
                # without wrapping in EngineGenerateError.
                queue.put(InputStreamError(error))
            finally:
                queue._input_stream_task = None
                if not cancelled:
                    # 发送空的 final_req 表示输入流已结束
                    # 如果是取消 (cancel)，则不发送 (会话已被中止)
                    # Send empty final request to indicate that inputs have
                    # finished. Don't send if cancelled (session was aborted).
                    await self._add_request(final_req, None, None, 0, queue)

        # Ensure output handler is running.
        self._run_output_handler()

        # 创建并保存输入处理任务的引用
        queue._input_stream_task = asyncio.create_task(handle_inputs())
        return queue

    @staticmethod
    def _validate_streaming_input_sampling_params(
        params: SamplingParams | PoolingParams,
    ):
        # 验证流式输入的采样参数
        # 流式输入不支持以下场景:
        #   1) Pooling 模型 (如 embedding 模型)
        #   2) n > 1 (多个采样结果)
        #   3) FINAL_ONLY 输出模式 (只返回最终结果)
        #   4) stop 字符串 (需要完整文本才能匹配)
        if (
            not isinstance(params, SamplingParams)
            or params.n > 1
            or params.output_kind == RequestOutputKind.FINAL_ONLY
            or params.stop
        ):
            raise ValueError(
                "Input streaming not currently supported "
                "for pooling models, n > 1, request_kind = FINAL_ONLY "
                "or with stop strings."
            )

    # TODO: we should support multiple prompts in one call, as you
    # can do with LLM.generate. So that for multi-prompt completion
    # requests we don't need to send multiple messages to core proc,
    # and so we don't need multiple streams which then get
    # re-multiplexed in the API server anyhow.
    async def generate(
        self,
        prompt: EngineCoreRequest
        | PromptType
        | EngineInput
        | AsyncGenerator[StreamingInput, None],
        sampling_params: SamplingParams,
        request_id: str,
        *,
        prompt_text: str | None = None,
        lora_request: LoRARequest | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
        trace_headers: Mapping[str, str] | None = None,
        priority: int = 0,
        data_parallel_rank: int | None = None,
        reasoning_ended: bool | None = None,
        reasoning_parser_kwargs: dict[str, Any] | None = None,
    ) -> AsyncGenerator[RequestOutput, None]:
        """
        Main function called by the API server to kick off a request
            * 1) Making an AsyncStream corresponding to the Request.
            * 2) Processing the Input.
            * 3) Adding the Request to the Detokenizer.
            * 4) Adding the Request to the EngineCore (separate process).

        A separate output_handler loop runs in a background AsyncIO task,
        pulling outputs from EngineCore and putting them into the
        per-request AsyncStream.

        The caller of generate() iterates the returned AsyncGenerator,
        returning the RequestOutput back to the caller.
        """
        # generate(): 核心生成方法，由 API 服务器调用以发起文本生成请求
        #
        # 完整流程:
        #   1) 调用 add_request() 将请求添加到引擎
        #   2) 返回一个 AsyncGenerator，调用方迭代获取输出
        #   3) 内部通过 RequestOutputCollector 队列接收输出:
        #      - output_handler 后台任务从 EngineCore 拉取输出
        #      - OutputProcessor 处理后将结果放入队列
        #      - generate() 的 while 循环从队列取出并 yield 给调用方
        #
        # 错误处理策略:
        #   - CancelledError/GeneratorExit: 客户端断开连接，中止请求
        #   - EngineDeadError: 引擎已死，不中止 (正在关闭)
        #   - ValueError: 请求参数错误，直接抛出
        #   - InputStreamError: 输入流错误，中止请求并传播原始异常
        #   - 其他异常: 包装为 EngineGenerateError 并中止请求
        #
        # 资源清理:
        #   finally 块中调用 queue.close() 确保队列被正确关闭

        q: RequestOutputCollector | None = None
        try:
            # 步骤 1: 添加请求到引擎
            q = await self.add_request(
                request_id,
                prompt,
                sampling_params,
                lora_request=lora_request,
                tokenization_kwargs=tokenization_kwargs,
                trace_headers=trace_headers,
                priority=priority,
                data_parallel_rank=data_parallel_rank,
                prompt_text=prompt_text,
                reasoning_ended=reasoning_ended,
                reasoning_parser_kwargs=reasoning_parser_kwargs,
            )

            # 步骤 2: 从队列中拉取输出并 yield 给调用方
            # output_handler 后台任务将输出推入队列，
            # 此处的 while 循环从队列拉取并 yield 给调用方
            # The output_handler task pushes items into the queue.
            # This task pulls from the queue and yields to caller.
            finished = False
            while not finished:
                # 优先使用 get_nowait() 避免不必要的任务切换，
                # 在高负载下可以提升性能
                # Note: drain queue without await if possible (avoids
                # task switching under load which helps performance).
                out = q.get_nowait() or await q.get()

                # OutputProcessor 和 EngineCore 各自根据 finished 标志
                # 处理请求清理工作
                # Note: both OutputProcessor and EngineCore handle their
                # own request cleanup based on finished.
                assert isinstance(out, RequestOutput)
                finished = out.finished
                if out is not STREAM_FINISHED:
                    yield out

        # 错误处理: 客户端断开连接 (请求被取消或生成器被垃圾回收)
        # If the request is disconnected by the client, generate()
        # is cancelled or the generator is garbage collected. So,
        # we abort the request if we end up here.
        except (asyncio.CancelledError, GeneratorExit):
            if q is not None:
                await self.abort(q.request_id, internal=True)
            if self.log_requests:
                logger.info("Request %s aborted.", request_id)
            raise

        # 错误处理: 引擎已死 (正在关闭，无需中止)
        # Engine is dead. Do not abort since we shut down.
        except EngineDeadError:
            if self.log_requests:
                logger.info("Request %s failed (engine dead).", request_id)
            raise

        # 错误处理: 请求参数验证失败
        # Request validation error.
        except ValueError as e:
            if self.log_requests:
                logger.info("Request %s failed (bad request): %s.", request_id, e)
            raise

        # 错误处理: 输入流生成器的错误，直接传播原始异常
        # Error from input stream generator - propagate directly.
        except InputStreamError as e:
            if q is not None:
                await self.abort(q.request_id, internal=True)
            if self.log_requests:
                logger.info("Request %s failed (input error): %s.", request_id, e)
            raise e.cause from e

        # 错误处理: 未预期的异常 (可能是可恢复的)
        # 包装为 EngineGenerateError 以便上层统一处理
        # Unexpected error in the generate() task (possibly recoverable).
        except Exception as e:
            if q is not None:
                await self.abort(q.request_id, internal=True)
            if self.log_requests:
                try:
                    s = f"{e.__class__.__name__}: {e}"
                except Exception as e2:
                    s = (
                        f"{e.__class__.__name__}: "
                        "error during printing an exception of class"
                        + e2.__class__.__name__
                    )
                logger.info("Request %s failed due to %s.", request_id, s)
            raise EngineGenerateError() from e
        finally:
            # 确保输出队列被关闭，释放相关资源
            if q is not None:
                q.close()

    def _run_output_handler(self):
        """Background loop: pulls from EngineCore and pushes to AsyncStreams."""
        # 启动输出处理后台任务
        #
        # output_handler 是 AsyncLLM 的核心后台任务，负责:
        #   1) 从 EngineCore 拉取原始输出 (EngineCoreOutputs)
        #   2) 通过 OutputProcessor 处理输出 (detokenization、统计等)
        #   3) 处理结果自动分发到各请求的 RequestOutputCollector 队列
        #   4) 中止因 stop 字符串结束的请求
        #   5) 记录统计信息 (Prometheus 指标等)
        #
        # 分块处理: 输出按 VLLM_V1_OUTPUT_PROC_CHUNK_SIZE 分块，
        # 避免长时间阻塞事件循环，每个 chunk 处理后让出控制权

        if self.output_handler is not None:
            return

        # 将 self 的属性提取为局部变量，避免 output_handler 闭包
        # 持有对 AsyncLLM 的循环引用，导致无法被垃圾回收
        # Ensure that the task doesn't have a circular ref back to the AsyncLLM
        # object, or else it won't be garbage collected and cleaned up properly.
        engine_core = self.engine_core
        output_processor = self.output_processor
        log_stats = self.log_stats
        # 使用可变列表存储 logger_manager 引用，这样在弹性 EP 扩缩容时
        # 可以更新 logger 而不需要通过 self 创建循环引用
        # We use a mutable list for logger_manager so that it can be updated
        # during elastic EP scaling (see scale_elastic_ep) without creating
        # a circular reference via self.
        self._logger_ref = [self.logger_manager]
        logger_ref = self._logger_ref
        renderer = self.renderer
        chunk_size = envs.VLLM_V1_OUTPUT_PROC_CHUNK_SIZE

        async def output_handler():
            try:
                while True:
                    # 步骤 1: 从 EngineCore 拉取输出
                    # 1) Pull EngineCoreOutputs from the EngineCore.
                    outputs = await engine_core.get_output_async()
                    num_outputs = len(outputs.outputs)

                    # 仅在有输出且启用日志时创建统计对象
                    iteration_stats = (
                        IterationStats() if (log_stats and num_outputs) else None
                    )

                    # 步骤 2: 分块处理输出，避免阻塞事件循环
                    # Split outputs into chunks of at most
                    # VLLM_V1_OUTPUT_PROC_CHUNK_SIZE, so that we don't block the
                    # event loop for too long.
                    engine_core_outputs = outputs.outputs
                    for start in range(0, num_outputs, chunk_size):
                        end = start + chunk_size
                        outputs_slice = engine_core_outputs[start:end]
                        # 2) Process EngineCoreOutputs.
                        processed_outputs = output_processor.process_outputs(
                            outputs_slice, outputs.timestamp, iteration_stats
                        )
                        # NOTE: RequestOutputs are pushed to their queues.
                        # 处理后的输出已被推送到各请求的队列中
                        assert not processed_outputs.request_outputs

                        # 在 chunk 之间让出控制权，允许其他 asyncio 任务运行
                        # Allow other asyncio tasks to run between chunks
                        if end < num_outputs:
                            await asyncio.sleep(0)

                        # 步骤 3: 中止因 stop 字符串而结束的请求
                        # 3) Abort any reqs that finished due to stop strings.
                        if processed_outputs.reqs_to_abort:
                            await engine_core.abort_requests_async(
                                processed_outputs.reqs_to_abort
                            )

                    # 更新调度器统计信息
                    output_processor.update_scheduler_stats(outputs.scheduler_stats)

                    # 步骤 4: 记录统计信息 (Prometheus 指标、多模态缓存统计等)
                    # 4) Logging.
                    # TODO(rob): make into a coroutine and launch it in
                    # background thread once Prometheus overhead is non-trivial.
                    if logger_ref[0]:
                        logger_ref[0].record(
                            engine_idx=outputs.engine_index,
                            scheduler_stats=outputs.scheduler_stats,
                            iteration_stats=iteration_stats,
                            mm_cache_stats=renderer.stat_mm_cache(),
                        )
            except Exception as e:
                # 输出处理任务发生未预期异常
                # 通过 propagate_error 通知所有等待中的请求
                logger.exception("AsyncLLM output_handler failed.")
                output_processor.propagate_error(e)

        self.output_handler = asyncio.create_task(output_handler())

    async def abort(
        self, request_id: str | Iterable[str], internal: bool = False
    ) -> None:
        """Abort RequestId in OutputProcessor and EngineCore."""
        # 中止请求: 同时在 OutputProcessor 和 EngineCore 中清理
        #
        # 流程:
        #   1) 在 OutputProcessor 中标记请求为已中止 (本进程)
        #   2) 向 EngineCore 发送中止请求 (后台进程)
        #
        # 参数:
        #   request_id: 要中止的请求 ID (单个或多个)
        #   internal: 是否为内部中止 (如客户端断开连接)

        request_ids = (
            (request_id,) if isinstance(request_id, str) else as_list(request_id)
        )
        all_request_ids = self.output_processor.abort_requests(request_ids, internal)
        await self.engine_core.abort_requests_async(all_request_ids)

        if self.log_requests:
            logger.info("Aborted request(s) %s.", ",".join(request_ids))

    async def notify_kv_transfer_request_rejected(
        self,
        request_id: str,
        kv_transfer_params: dict[str, Any],
        *,
        data_parallel_rank: int | None = None,
    ) -> None:
        """Submit a pre-aborted request so the connector's request_finished
        hook runs to free any pre-admission KV-transfer resources (e.g. NIXL
        prefill blocks pinned on the P node)."""
        # 通知 KV 传输请求被拒绝
        #
        # 用于分布式 KV 缓存传输场景 (如 disaggregated prefill/decode):
        # 当请求被拒绝时，需要触发 connector 的 request_finished 钩子
        # 以释放预先分配的 KV 传输资源 (如 NIXL prefill 节点上固定的缓存块)
        #
        # 实现方式: 提交一个 "立即中止" 的请求，EngineCore 处理时
        # 会触发 request_finished 回调，从而清理资源
        request = EngineCoreRequest(
            request_id=request_id,
            prompt_token_ids=[0],
            mm_features=None,
            sampling_params=SamplingParams(
                max_tokens=1,
                extra_args={"kv_transfer_params": dict(kv_transfer_params)},
            ),
            pooling_params=None,
            arrival_time=time.time(),
            lora_request=None,
            cache_salt=None,
            data_parallel_rank=data_parallel_rank,
            abort_immediately=True,
        )
        await self.engine_core.add_request_async(request)

    async def pause_generation(
        self,
        *,
        mode: PauseMode = "abort",
        wait_for_inflight_requests: bool | None = None,
        clear_cache: bool = True,
    ) -> None:
        """
        Pause generation to allow model weight updates.

        All mode handling (abort / wait / keep) and cache clearing is done
        in the engine. New generation/encoding requests will not be scheduled
        until resume is called.

        Args:
            mode: How to handle in-flight requests:
                - ``"abort"``: Abort all in-flight requests immediately
                  (default).
                - ``"wait"``: Wait for in-flight requests to complete.
                - ``"keep"``: Freeze requests in queue; they resume on
                  :meth:`resume_generation`.
            wait_for_inflight_requests: DEPRECATED: use mode argument.
            clear_cache: Whether to clear KV cache and prefix cache after
                draining. Set to ``False`` to preserve cache for faster resume.
        """
        # 暂停生成: 用于模型权重更新 (如 RL 训练)
        #
        # 暂停模式 (mode):
        #   - "abort": 立即中止所有正在处理的请求 (默认)
        #   - "wait": 等待正在处理的请求完成
        #   - "keep": 冻结队列中的请求，恢复时继续执行
        #
        # 典型使用场景:
        #   在 RL (强化学习) 训练中，需要暂停推理引擎来更新模型权重，
        #   更新完成后调用 resume_generation() 恢复推理
        if wait_for_inflight_requests:
            warnings.warn(
                "The `wait_for_inflight_requests` parameter in "
                "`AsyncLLM.pause_generation()` is deprecated. "
                "Please use `mode` argument instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            mode = "wait"
        if clear_cache:
            await self.renderer.clear_mm_cache_async()
        await self.engine_core.pause_scheduler_async(mode=mode, clear_cache=clear_cache)
        # 短暂休眠，确保正在处理的请求的最终输出在方法返回前被处理
        # 这不是正确性要求，只是为了让调用方看到更合理的事件顺序
        # Small sleep to help ensure that final outputs from any in-flight requests are
        # returned prior to this method returning. These outputs come out of the engine
        # prior to the wait-for-idle completion event, but involve additional async
        # tasks in output processing.
        # Note that this is not required for correctness, just more intuitive ordering
        # of events from caller's pov.
        await asyncio.sleep(0.02)

    async def resume_generation(self) -> None:
        """Resume generation after :meth:`pause_generation`."""
        # 恢复生成: 在 pause_generation() 之后恢复引擎运行
        await self.engine_core.resume_scheduler_async()

    async def is_paused(self) -> bool:
        """Return whether the engine is currently paused."""
        # 查询引擎是否处于暂停状态
        return await self.engine_core.is_scheduler_paused_async()

    async def encode(
        self,
        prompt: PromptType | EngineInput,
        pooling_params: PoolingParams,
        request_id: str,
        lora_request: LoRARequest | None = None,
        trace_headers: Mapping[str, str] | None = None,
        priority: int = 0,
        tokenization_kwargs: dict[str, Any] | None = None,
        reasoning_ended: bool | None = None,
    ) -> AsyncGenerator[PoolingRequestOutput, None]:
        """
        Main function called by the API server to kick off a request
            * 1) Making an AsyncStream corresponding to the Request.
            * 2) Processing the Input.
            * 3) Adding the Request to the EngineCore (separate process).

        A separate output_handler loop runs in a background AsyncIO task,
        pulling outputs from EngineCore and putting them into the
        per-request AsyncStream.

        The caller of generate() iterates the returned AsyncGenerator,
        returning the RequestOutput back to the caller.
        """
        # encode(): 用于 embedding/pooling 模型的编码方法
        #
        # 与 generate() 类似，但适用于 Pooling 模型 (如 sentence-transformers)
        # 主要区别:
        #   - 使用 PoolingParams 而非 SamplingParams
        #   - 返回 PoolingRequestOutput (包含 embedding 向量)
        #   - 不支持 n>1 等采样参数

        q: RequestOutputCollector | None = None
        try:
            # 添加请求并从队列获取输出 (与 generate() 逻辑类似)
            q = await self.add_request(
                request_id,
                prompt,
                pooling_params,
                lora_request=lora_request,
                tokenization_kwargs=tokenization_kwargs,
                trace_headers=trace_headers,
                priority=priority,
                reasoning_ended=reasoning_ended,
            )

            # The output_handler task pushes items into the queue.
            # This task pulls from the queue and yields to caller.
            finished = False
            while not finished:
                # Note: drain queue without await if possible (avoids
                # task switching under load which helps performance).
                out = q.get_nowait() or await q.get()
                assert isinstance(out, PoolingRequestOutput)
                # Note: both OutputProcessor and EngineCore handle their
                # own request cleanup based on finished.
                finished = out.finished
                yield out

        # 错误处理: 客户端断开连接
        # If the request is disconnected by the client, generate()
        # is cancelled. So, we abort the request if we end up here.
        except asyncio.CancelledError:
            if q is not None:
                await self.abort(q.request_id, internal=True)
            if self.log_requests:
                logger.info("Request %s aborted.", request_id)
            raise

        # 错误处理: 引擎已死
        # Engine is dead. Do not abort since we shut down.
        except EngineDeadError:
            if self.log_requests:
                logger.info("Request %s failed (engine dead).", request_id)
            raise

        # 错误处理: 请求参数错误
        # Request validation error.
        except ValueError:
            if self.log_requests:
                logger.info("Request %s failed (bad request).", request_id)
            raise

        # 错误处理: 未预期的异常
        # Unexpected error in the generate() task (possibly recoverable).
        except Exception as e:
            if q is not None:
                await self.abort(q.request_id, internal=True)
            if self.log_requests:
                logger.info("Request %s failed.", request_id)
            raise EngineGenerateError() from e
        finally:
            if q is not None:
                q.close()

    @property
    def tokenizer(self) -> TokenizerLike | None:
        # 获取 tokenizer 实例 (可能为 None)
        return self.renderer.tokenizer

    def get_tokenizer(self) -> TokenizerLike:
        # 获取 tokenizer 实例 (确保不为 None)
        return self.renderer.get_tokenizer()

    async def is_tracing_enabled(self) -> bool:
        # 查询是否启用了分布式追踪 (OpenTelemetry)
        return self.observability_config.otlp_traces_endpoint is not None

    async def do_log_stats(self) -> None:
        # 手动触发统计日志记录
        if self.logger_manager:
            self.logger_manager.log()

    async def check_health(self) -> None:
        # 健康检查: 如果引擎出错则抛出异常
        logger.debug("Called check_health.")
        if self.errored:
            raise self.dead_error

    async def start_profile(self, profile_prefix: str | None = None) -> None:
        # 启动性能分析 (同时启动 EngineCore 和前端的 profiler)
        coros = [self.engine_core.profile_async(True, profile_prefix)]
        if self.profiler is not None:
            coros.append(asyncio.to_thread(self.profiler.start))
        await asyncio.gather(*coros)

    async def stop_profile(self) -> None:
        # 停止性能分析
        coros = [self.engine_core.profile_async(False)]
        if self.profiler is not None:
            coros.append(asyncio.to_thread(self.profiler.stop))
        await asyncio.gather(*coros)

    async def reset_mm_cache(self) -> None:
        # 重置多模态缓存 (图像/音频预处理结果缓存)
        await self.renderer.clear_mm_cache_async()
        await self.engine_core.reset_mm_cache_async()

    async def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        # 重置前缀缓存 (用于 prompt 共享的 KV 缓存)
        return await self.engine_core.reset_prefix_cache_async(
            reset_running_requests, reset_connector
        )

    async def reset_encoder_cache(self) -> None:
        # 重置编码器缓存
        await self.engine_core.reset_encoder_cache_async()

    async def sleep(self, level: int = 1, mode: PauseMode = "abort") -> None:
        # 让引擎进入休眠状态 (释放 GPU 资源)
        # 用于动态资源管理场景
        if level >= 1:
            await self.renderer.clear_mm_cache_async()
        await self.engine_core.sleep_async(level, mode)

        if self.logger_manager is not None:
            self.logger_manager.record_sleep_state(1, level)

    async def wake_up(self, tags: list[str] | None = None) -> None:
        # 唤醒休眠的引擎
        await self.engine_core.wake_up_async(tags)

        if self.logger_manager is not None:
            self.logger_manager.record_sleep_state(0, 0)

    async def is_sleeping(self) -> bool:
        # 查询引擎是否处于休眠状态
        return await self.engine_core.is_sleeping_async()

    async def add_lora(self, lora_request: LoRARequest) -> bool:
        """Load a new LoRA adapter into the engine for future requests."""
        # 动态加载 LoRA 适配器
        return await self.engine_core.add_lora_async(lora_request)

    async def remove_lora(self, lora_id: int) -> bool:
        """Remove an already loaded LoRA adapter."""
        # 移除已加载的 LoRA 适配器
        return await self.engine_core.remove_lora_async(lora_id)

    async def list_loras(self) -> set[int]:
        """List all registered adapters."""
        # 列出所有已注册的 LoRA 适配器 ID
        return await self.engine_core.list_loras_async()

    async def pin_lora(self, lora_id: int) -> bool:
        """Prevent an adapter from being evicted."""
        # 固定 LoRA 适配器，防止被驱逐
        return await self.engine_core.pin_lora_async(lora_id)

    async def collective_rpc(
        self,
        method: str,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
    ):
        """
        Perform a collective RPC call to the given path.
        """
        # 集合 RPC 调用: 向所有 worker 广播执行指定方法
        # 用于需要所有 worker 协同操作的场景 (如权重更新)
        return await self.engine_core.collective_rpc_async(
            method, timeout, args, kwargs
        )

    async def wait_for_requests_to_drain(self, drain_timeout: int = 300):
        """Wait for all requests to be drained."""
        # 等待所有请求被排空 (用于扩缩容前的安全等待)
        start_time = time.time()
        while time.time() - start_time < drain_timeout:
            if not self.engine_core.dp_engines_running():
                logger.info("Engines are idle, requests have been drained")
                return

            logger.info("Engines are still running, waiting for requests to drain...")
            await asyncio.sleep(1)  # Wait 1 second before checking again

        raise TimeoutError(
            f"Timeout reached after {drain_timeout} seconds "
            "waiting for requests to drain."
        )

    async def scale_elastic_ep(
        self, new_data_parallel_size: int, drain_timeout: int = 300
    ):
        """
        Scale up or down the data parallel size by adding or removing
        engine cores.
        Args:
            new_data_parallel_size: The new number of data parallel workers
            drain_timeout:
                Maximum time to wait for requests to drain (seconds)
        """
        # 弹性 EP (Expert Parallelism) 扩缩容
        #
        # 功能: 动态增加或减少数据并行 worker 数量
        # 流程:
        #   1) 检查是否需要扩缩容
        #   2) 可选: 等待当前请求排空
        #   3) 重建统计日志器 (扩容时)
        #   4) 调用 EngineCore 执行实际的扩缩容
        #
        # 使用场景: 在线服务中根据负载动态调整并行度
        old_data_parallel_size = self.vllm_config.parallel_config.data_parallel_size
        if old_data_parallel_size == new_data_parallel_size:
            logger.info(
                "Data parallel size is already %s, skipping scale",
                new_data_parallel_size,
            )
            return

        if envs.VLLM_ELASTIC_EP_DRAIN_REQUESTS:
            logger.info(
                "VLLM_ELASTIC_EP_DRAIN_REQUESTS is set, "
                "waiting for requests to drain before scaling"
            )
            await self.wait_for_requests_to_drain(drain_timeout)

        # 扩容时重建统计日志器 (因为 engine 数量变化了)
        # recreate stat loggers
        if new_data_parallel_size > old_data_parallel_size and self.log_stats:
            # TODO(rob): fix this after talking with Ray team.
            # This resets all the prometheus metrics since we
            # unregister during initialization. Need to understand
            # the intended behavior here better.
            self.logger_manager = StatLoggerManager(
                vllm_config=self.vllm_config,
                engine_idxs=list(range(new_data_parallel_size)),
                custom_stat_loggers=None,
            )
            # 更新可变引用，让 output_handler 使用新的 logger
            # 而不需要通过 self 创建循环引用
            # Update the mutable ref so output_handler picks up the
            # new logger without creating a circular reference via self.
            if hasattr(self, "_logger_ref"):
                self._logger_ref[0] = self.logger_manager
            self.logger_manager.log_engine_initialized()

        set_scaling_elastic_ep(True)
        try:
            await self.engine_core.scale_elastic_ep(new_data_parallel_size)
            self.vllm_config.parallel_config.data_parallel_size = new_data_parallel_size
        finally:
            set_scaling_elastic_ep(False)

    @property
    def is_running(self) -> bool:
        # 引擎是否正在运行 (output_handler 启动前返回 True)
        # Is None before the loop is started.
        return self.output_handler is None or not self.output_handler.done()

    @property
    def is_stopped(self) -> bool:
        # 引擎是否已停止 (出错即停止)
        return self.errored

    @property
    def errored(self) -> bool:
        # 引擎是否出错: EngineCore 死亡 或 output_handler 任务已完成
        return self.engine_core.resources.engine_dead or not self.is_running

    @property
    def dead_error(self) -> BaseException:
        # 返回引擎死亡的异常对象
        return EngineDeadError()

    async def init_weight_transfer_engine(
        self, request: WeightTransferInitRequest
    ) -> None:
        """
        Initialize weight transfer for RL training.

        Args:
            request: Weight transfer initialization request with backend-specific info
        """
        # 初始化权重传输引擎 (用于 RL 训练中的在线权重更新)
        # 支持分布式权重传输后端 (如 NCCL、NIXL 等)
        from vllm.distributed.weight_transfer.base import (
            WeightTransferInitRequest,
        )

        if isinstance(request, WeightTransferInitRequest):
            init_info_dict = request.init_info
        else:
            raise TypeError(f"Expected WeightTransferInitRequest, got {type(request)}")

        await self.collective_rpc(
            "init_weight_transfer_engine", kwargs={"init_info": init_info_dict}
        )

    async def start_weight_update(self, is_checkpoint_format: bool = True) -> None:
        """Start a new weight update."""
        # 开始新的权重更新 (通知所有 worker 准备接收新权重)
        await self.collective_rpc(
            "start_weight_update",
            kwargs={"is_checkpoint_format": is_checkpoint_format},
        )

    async def update_weights(self, request: WeightTransferUpdateRequest) -> None:
        """
        Batched weight update for RL training.

        Args:
            request: Weight update request with backend-specific update info
        """
        # 执行批量权重更新 (将新权重传输到所有 worker)

        if isinstance(request, WeightTransferUpdateRequest):
            update_info_dict = request.update_info
        else:
            raise TypeError(
                f"Expected WeightTransferUpdateRequest, got {type(request)}"
            )

        await self.collective_rpc(
            "update_weights", kwargs={"update_info": update_info_dict}
        )

    async def finish_weight_update(self) -> None:
        """Finish the current weight update."""
        # 完成当前权重更新 (所有 worker 应用新权重)
        await self.collective_rpc("finish_weight_update")
