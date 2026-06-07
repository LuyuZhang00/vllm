# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# =============================================================================
# 模块概述: vLLM v1 引擎核心主循环 (Engine Core Main Loop)
# =============================================================================
# 本模块是 vLLM v1 引擎的核心，负责调度和执行模型推理请求。
#
# 核心类层次:
#   EngineCore            — 引擎核心基类，包含调度→执行→采样→更新的主循环
#   ├── EngineCoreProc    — 在后台进程中运行 EngineCore，通过 ZMQ IPC 通信
#   │   └── DPEngineCoreProc — 数据并行版本，支持 MoE 模型的 DP 部署
#   └── EngineCoreActorMixin — Ray Actor 混入类，用于 Ray 集群部署
#
# 请求生命周期:
#   API 请求 → AsyncLLM → InputProcessor (tokenize) → EngineCoreClient (IPC)
#   → EngineCoreProc (Input线程 → input_queue → 主线程)
#   → Scheduler.schedule() → Executor.execute_model() → 采样
#   → output_queue → Output线程 → ZMQ → AsyncLLM (OutputProcessor)
#
# 主循环 (run_busy_loop) 每轮执行:
#   ① _process_input_queue(): 从 input_queue 取出请求，交给调度器
#   ② _process_engine_step(): 调度 → 模型前向 → 采样 → 更新状态 → 放入 output_queue
#
# 线程模型 (EngineCoreProc):
#   - 主线程: 运行 run_busy_loop，执行调度和模型推理
#   - Input 守护线程: 从 ZMQ socket 接收请求，反序列化后放入 input_queue
#   - Output 守护线程: 从 output_queue 取出结果，序列化后通过 ZMQ 发送
#   这些线程释放 GIL，使得 ZMQ IO 与 GPU 计算可以真正并行执行。
# =============================================================================

import gc
import os
import queue
import signal
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable, Generator
from concurrent.futures import Future
from contextlib import ExitStack, contextmanager
from enum import IntEnum
from functools import partial
from inspect import isclass, signature
from logging import DEBUG
from multiprocessing.queues import Queue
from typing import Any, TypeVar, cast

import msgspec
import zmq

import vllm.envs as envs
from vllm.config import ParallelConfig, VllmConfig
from vllm.distributed import (
    cleanup_dist_env_and_memory,
    stateless_destroy_torch_distributed_process_group,
)
from vllm.envs import enable_envs_cache
from vllm.logger import init_logger
from vllm.logging_utils.dump_input import dump_engine_exception
from vllm.lora.request import LoRARequest
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.tasks import POOLING_TASKS, SupportedTask
from vllm.tracing import instrument, maybe_init_worker_tracer
from vllm.transformers_utils.config import maybe_register_config_serialize_by_value
from vllm.utils import numa_utils
from vllm.utils.gc_utils import (
    freeze_gc_heap,
    maybe_attach_gc_debug_callback,
)
from vllm.utils.hashing import get_hash_fn_by_name
from vllm.utils.network_utils import make_zmq_socket
from vllm.utils.system_utils import decorate_logs, set_process_title
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    generate_scheduler_kv_cache_config,
    get_kv_cache_configs,
    get_request_block_hasher,
    init_none_hash,
    resolve_kv_cache_block_sizes,
)
from vllm.v1.core.sched.interface import PauseState, SchedulerInterface
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.engine import (
    EEP_NOTIFICATION_CALL_ID,
    EEPNotificationType,
    EngineCoreOutput,
    EngineCoreOutputs,
    EngineCoreReadyResponse,
    EngineCoreRequest,
    EngineCoreRequestType,
    FinishReason,
    PauseMode,
    ReconfigureDistributedRequest,
    ReconfigureRankType,
    UtilityOutput,
    UtilityResult,
)
from vllm.v1.engine.tensor_ipc import TensorIpcReceiver
from vllm.v1.engine.utils import (
    EngineHandshakeMetadata,
    EngineZmqAddresses,
    SignalCallback,
    get_device_indices,
)
from vllm.v1.executor import Executor
from vllm.v1.kv_cache_interface import KVCacheConfig, get_kv_cache_spec_kind
from vllm.v1.metrics.stats import SchedulerStats
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder
from vllm.v1.structured_output import StructuredOutputManager
from vllm.v1.utils import IterationDetails, compute_iteration_details
from vllm.version import __version__ as VLLM_VERSION

logger = init_logger(__name__)

HANDSHAKE_TIMEOUT_MINS = 5

_R = TypeVar("_R")  # Return type for collective_rpc


class EngineCore:
    """Inner loop of vLLM's Engine."""

    # EngineCore: vLLM v1 引擎的核心基类。
    #
    # 核心职责:
    #   1. 管理 Scheduler（调度器）和 Executor（执行器）的生命周期
    #   2. 提供 step() 方法执行一轮推理（调度→执行→采样→更新）
    #   3. 管理 KV 缓存的初始化和配置
    #   4. 处理请求的添加、中止和状态管理
    #
    # 初始化顺序（严格依赖关系）:
    #   1. 插件加载 (load_general_plugins) — 必须最先执行
    #   2. 模型执行器 (model_executor) — 需要先实例化以进行内存分析
    #   3. KV 缓存 (_initialize_kv_caches) — 依赖执行器的内存分析结果
    #   4. 调度器 (scheduler) — 依赖 KV 缓存配置
    #   5. 批处理队列 (batch_queue) — 用于流水线并行
    #
    # step() 执行流程:
    #   1. scheduler.schedule() — 从等待队列选取请求，分配 KV 缓存块
    #   2. executor.execute_model() — 异步提交模型前向计算到 GPU
    #   3. scheduler.get_grammar_bitmask() — CPU 端并行计算结构化输出约束
    #   4. future.result() — 等待 GPU 执行完成
    #   5. executor.sample_tokens() — 使用 grammar 约束采样
    #   6. scheduler.update_from_output() — 更新调度器状态（完成请求、释放 KV 块等）

    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        executor_fail_callback: Callable | None = None,
        include_finished_set: bool = False,
    ):
        # 初始化序列遵循严格的依赖顺序:
        # 1. 插件加载 (load_general_plugins) — 必须最先执行，后续组件可能依赖插件
        # 2. 模型执行器 (model_executor) — 需要先实例化以进行内存分析
        # 3. KV 缓存 (_initialize_kv_caches) — 依赖执行器的内存分析结果来分配缓存
        # 4. 调度器 (scheduler) — 依赖 KV 缓存配置来管理请求调度
        # 5. 批处理队列 (batch_queue) — 用于流水线并行，消除 pipeline bubble

        # plugins need to be loaded at the engine/scheduler level too
        from vllm.plugins import load_general_plugins

        load_general_plugins()

        self.vllm_config = vllm_config
        if not vllm_config.parallel_config.data_parallel_rank_local:
            logger.info(
                "Initializing a V1 LLM engine (v%s) with config: %s",
                VLLM_VERSION,
                vllm_config,
            )

        self.log_stats = log_stats

        # Setup Model.
        self.model_executor = executor_class(vllm_config)
        if executor_fail_callback is not None:
            self.model_executor.register_failure_callback(executor_fail_callback)

        self.available_gpu_memory_for_kv_cache = -1

        if envs.VLLM_ELASTIC_EP_SCALE_UP_LAUNCH:
            self._eep_scale_up_before_kv_init()

        # Setup KV Caches and update CacheConfig after profiling.
        kv_cache_config = self._initialize_kv_caches(vllm_config)
        self.structured_output_manager = StructuredOutputManager(vllm_config)

        # Setup scheduler.
        Scheduler = vllm_config.scheduler_config.get_scheduler_cls()

        if len(kv_cache_config.kv_cache_groups) == 0:  # noqa: SIM102
            # Encoder models without KV cache don't support
            # chunked prefill. But do SSM models?
            if vllm_config.scheduler_config.enable_chunked_prefill:
                logger.warning("Disabling chunked prefill for model without KVCache")
                vllm_config.scheduler_config.enable_chunked_prefill = False

        scheduler_block_size, hash_block_size = resolve_kv_cache_block_sizes(
            kv_cache_config, vllm_config
        )

        self.scheduler: SchedulerInterface = Scheduler(
            vllm_config=vllm_config,
            kv_cache_config=kv_cache_config,
            structured_output_manager=self.structured_output_manager,
            include_finished_set=include_finished_set,
            log_stats=self.log_stats,
            block_size=scheduler_block_size,
            hash_block_size=hash_block_size,
        )
        self.use_spec_decode = vllm_config.speculative_config is not None
        if self.scheduler.connector is not None:  # type: ignore
            self.model_executor.init_kv_output_aggregator(self.scheduler.connector)  # type: ignore

        mm_registry = MULTIMODAL_REGISTRY
        self.mm_receiver_cache = mm_registry.engine_receiver_cache_from_config(
            vllm_config
        )

        # If a KV connector is initialized for scheduler, we want to collect
        # handshake metadata from all workers so the connector in the scheduler
        # will have the full context
        kv_connector = self.scheduler.get_kv_connector()
        if kv_connector is not None:
            # Collect and store KV connector xfer metadata from workers
            # (after KV cache registration)
            xfer_handshake_metadata = (
                self.model_executor.get_kv_connector_handshake_metadata()
            )

            if xfer_handshake_metadata:
                # xfer_handshake_metadata is list of dicts from workers
                # Each dict already has structure {tp_rank: metadata}
                # Merge all worker dicts into a single dict
                content: dict[int, Any] = {}
                for worker_dict in xfer_handshake_metadata:
                    if worker_dict is not None:
                        content.update(worker_dict)
                kv_connector.set_xfer_handshake_metadata(content)

        # Setup batch queue for pipeline parallelism.
        # Batch queue for scheduled batches. This enables us to asynchronously
        # schedule and execute batches, and is required by pipeline parallelism
        # to eliminate pipeline bubbles.
        #
        # batch_queue 的作用: 流水线并行的批处理队列。
        # 当 batch_queue_size > 1 时（流水线并行场景），调度和执行可以重叠：
        # 在等待当前批次 GPU 执行的同时，可以提前调度下一个批次，
        # 从而消除流水线气泡（pipeline bubble），提高 GPU 利用率。
        # 队列中每个元素是一个三元组: (采样结果Future, 调度输出, 模型执行Future)
        self.batch_queue_size = self.model_executor.max_concurrent_batches
        self.batch_queue: (
            deque[tuple[Future[ModelRunnerOutput], SchedulerOutput, Future[Any]]] | None
        ) = None
        if self.batch_queue_size > 1:
            logger.debug("Batch queue is enabled with size %d", self.batch_queue_size)
            self.batch_queue = deque(maxlen=self.batch_queue_size)

        self.is_ec_consumer = (
            vllm_config.ec_transfer_config is None
            or vllm_config.ec_transfer_config.is_ec_consumer
        )
        self.is_pooling_model = vllm_config.model_config.runner_type == "pooling"

        self.request_block_hasher: Callable[[Request], list[BlockHash]] | None = None
        if vllm_config.cache_config.enable_prefix_caching or kv_connector is not None:
            caching_hash_fn = get_hash_fn_by_name(
                vllm_config.cache_config.prefix_caching_hash_algo
            )
            init_none_hash(caching_hash_fn)

            self.request_block_hasher = get_request_block_hasher(
                hash_block_size, caching_hash_fn
            )

        # step_fn 的选择: 根据是否启用批处理队列选择不同的步进函数。
        # - step(): 普通模式，调度→执行→采样→更新 串行执行
        # - step_with_batch_queue(): 流水线并行模式，调度和执行可以重叠，
        #   通过批处理队列实现异步调度，消除流水线气泡
        self.step_fn = (
            self.step if self.batch_queue is None else self.step_with_batch_queue
        )
        self.async_scheduling = vllm_config.scheduler_config.async_scheduling

        self.aborts_queue = queue.Queue[list[str]]()

        self._idle_state_callbacks: list[Callable] = []

        # Mark the startup heap as static so that it's ignored by GC.
        # Reduces pause times of oldest generation collections.
        freeze_gc_heap()
        # If enable, attach GC debugger after static variable freeze.
        maybe_attach_gc_debug_callback()
        # Enable environment variable cache (e.g. assume no more
        # environment variable overrides after this point)
        enable_envs_cache()

    @instrument(span_name="Prepare model")
    def _initialize_kv_caches(self, vllm_config: VllmConfig) -> KVCacheConfig:
        """
        初始化 KV 缓存配置，包括内存分析、配置生成和缓存初始化。

        完整流程:
          1. get_kv_cache_specs(): 获取模型所需的 KV 缓存规格
             （不同注意力层可能有不同的 block_size、sliding_window 等）
          2. determine_available_memory(): 分析模型峰值内存，计算可用于 KV 缓存的显存
          3. get_kv_cache_configs(): 根据可用显存和规格，生成 KV 缓存配置
             （可能会自动调整 max_model_len 以适应可用内存）
          4. generate_scheduler_kv_cache_config(): 生成调度器使用的 KV 缓存配置
          5. initialize_from_config(): 在 worker 上初始化 KV 缓存并预热模型

        Args:
            vllm_config: vLLM 整体配置对象

        Returns:
            KVCacheConfig: 调度器使用的 KV 缓存配置
        """
        # 记录初始化开始时间
        start = time.time()

        # 获取模型所需的所有 KV 缓存规格
        kv_cache_specs = self.model_executor.get_kv_cache_specs()

        # 检查是否需要 KV 缓存
        has_kv_cache = any(kv_cache_spec for kv_cache_spec in kv_cache_specs)
        if has_kv_cache:
            # 如果启用了弹性 EP 扩容模式
            if envs.VLLM_ELASTIC_EP_SCALE_UP_LAUNCH:
                # 注意：可用内存应该已经在 _eep_scale_up_before_kv_init 中设置
                assert self.available_gpu_memory_for_kv_cache > 0
                available_gpu_memory = [self.available_gpu_memory_for_kv_cache] * len(
                    kv_cache_specs
                )
            else:
                # 分析模型的峰值内存使用情况，确定可以为 KV 缓存分配多少内存
                available_gpu_memory = self.model_executor.determine_available_memory()
                self.available_gpu_memory_for_kv_cache = available_gpu_memory[0]
        else:
            # 无注意力机制的模型不需要 KV 缓存内存
            available_gpu_memory = [0] * len(kv_cache_specs)

        # 验证 KV 缓存规格和可用内存数组长度匹配
        assert len(kv_cache_specs) == len(available_gpu_memory)

        # 记录 KV 缓存配置之前的 max_model_len，用于检测自动适配时的变化
        max_model_len_before = vllm_config.model_config.max_model_len

        # 获取 KV 缓存配置（可能会自动调整 max_model_len 以适应可用内存）
        kv_cache_configs = get_kv_cache_configs(
            vllm_config, kv_cache_specs, available_gpu_memory
        )

        # 如果自动适配调整了 max_model_len，需要将新值同步到所有 worker
        # 这是必要的，因为 worker 在内存分析之前就已经启动，缓存了原始的（更大的）max_model_len
        max_model_len_after = vllm_config.model_config.max_model_len
        if max_model_len_after != max_model_len_before:
            self.collective_rpc("update_max_model_len", args=(max_model_len_after,))

        # 生成调度器使用的 KV 缓存配置
        scheduler_kv_cache_config = generate_scheduler_kv_cache_config(kv_cache_configs)
        # 更新配置中的 GPU 块数量
        vllm_config.cache_config.num_gpu_blocks = scheduler_kv_cache_config.num_blocks
        # 如果有多个 KV 缓存组，选择最小的块大小作为配置
        kv_cache_groups = scheduler_kv_cache_config.kv_cache_groups
        if kv_cache_groups:
            vllm_config.cache_config.block_size = min(
                g.kv_cache_spec.block_size for g in kv_cache_groups
            )

        # 验证块大小是否有效
        vllm_config.validate_block_size()

        # 初始化 KV 缓存并预热模型执行
        self.model_executor.initialize_from_config(kv_cache_configs)

        # 计算初始化耗时
        elapsed = time.time() - start
        compile_time = vllm_config.compilation_config.compilation_time
        encoder_compile_time = vllm_config.compilation_config.encoder_compilation_time
        # 根据是否有编码器编译时间记录不同详细程度的日志
        if encoder_compile_time > 0:
            logger.info_once(
                "init engine (profile, create kv cache, warmup model) took "
                "%.2f s (compilation: %.2f s — language_model: %.2f s, "
                "encoder: %.2f s)",
                elapsed,
                compile_time + encoder_compile_time,
                compile_time,
                encoder_compile_time,
            )
        elif compile_time > 0:
            logger.info_once(
                "init engine (profile, create kv cache, warmup model) took "
                "%.2f s (compilation: %.2f s)",
                elapsed,
                compile_time,
            )
        else:
            logger.info_once(
                "init engine (profile, create kv cache, warmup model) took %.2f s",
                elapsed,
            )
        return scheduler_kv_cache_config

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        return self.model_executor.supported_tasks

    def get_kv_cache_group_metadata(self) -> list[dict[str, int | str | None]]:
        """Return msgspec-serializable metadata for scheduler KV cache groups."""
        kv_cache_config = getattr(self.scheduler, "kv_cache_config", None)
        if kv_cache_config is None:
            return []

        metadata: list[dict[str, int | str | None]] = []
        for group_idx, group in enumerate(kv_cache_config.kv_cache_groups):
            spec = group.kv_cache_spec
            metadata.append(
                {
                    "group_idx": group_idx,
                    "kind": get_kv_cache_spec_kind(spec).value,
                    "block_size": spec.block_size,
                    "sliding_window": getattr(spec, "sliding_window", None),
                }
            )
        return metadata

    def add_request(self, request: Request, request_wave: int = 0):
        """Add request to the scheduler.

        `request_wave`: indicate which wave of requests this is expected to
        belong to in DP case
        """
        # 请求验证流程:
        # 1. 验证 request_id 类型必须为字符串
        # 2. 验证 pooling 任务是否被模型支持（仅对 pooling 模型）
        # 3. 验证 KV 传输参数与 KV connector 的一致性
        # 4. 通过验证后将请求提交给调度器

        # Validate the request_id type.
        if not isinstance(request.request_id, str):
            raise TypeError(
                f"request_id must be a string, got {type(request.request_id)}"
            )

        if pooling_params := request.pooling_params:
            supported_pooling_tasks = [
                task for task in self.get_supported_tasks() if task in POOLING_TASKS
            ]

            if pooling_params.task not in supported_pooling_tasks:
                raise ValueError(
                    f"Unsupported task: {pooling_params.task!r} "
                    f"Supported tasks: {supported_pooling_tasks}"
                )

        if request.kv_transfer_params is not None and (
            not self.scheduler.get_kv_connector()
        ):
            logger.warning(
                "Got kv_transfer_params, but no KVConnector found. "
                "Disabling KVTransfer for this request."
            )

        self.scheduler.add_request(request)
        # abort_immediately 边界情况:
        # 某些请求在 KV 传输场景下需要立即中止（例如传输失败或资源不足）。
        # 虽然请求已提交给调度器，但立即中止可以触发 connector 的
        # request_finished 钩子，释放预分配的 KV 传输资源，避免资源泄漏。
        if request.abort_immediately:
            # Immediately abort so the connector's request_finished hook runs
            # to free any pre-admission KV-transfer resources.
            self.abort_requests([request.request_id])

    def abort_requests(self, request_ids: list[str]):
        """Abort requests from the scheduler."""

        # TODO: The scheduler doesn't really need to know the
        # specific finish reason, TBD whether we propagate that
        # (i.e. client-aborted vs stop criteria met).
        self.scheduler.finish_requests(request_ids, RequestStatus.FINISHED_ABORTED)

    @contextmanager
    def log_error_detail(self, scheduler_output: SchedulerOutput):
        """Execute the model and log detailed info on failure."""
        try:
            yield
        except Exception as err:
            # We do not want to catch BaseException here since we're only
            # interested in dumping info when the exception is due to an
            # error from execute_model itself.

            # NOTE: This method is exception-free
            dump_engine_exception(
                self.vllm_config, scheduler_output, self.scheduler.make_stats()
            )
            raise err

    @contextmanager
    def log_iteration_details(self, scheduler_output: SchedulerOutput | None):
        if not self.vllm_config.observability_config.enable_logging_iteration_details:
            yield
            return
        # 0-token step: let the dummy_batch wrapper log it (avoids double-log).
        if scheduler_output and scheduler_output.total_num_scheduled_tokens == 0:
            yield
            return
        self._iteration_index = getattr(self, "_iteration_index", 0)
        # scheduler_output=None marks a DP dummy iteration.
        if scheduler_output is None:
            iteration_details = IterationDetails(0, 0, 0, 0)
            is_dummy = True
        else:
            iteration_details = compute_iteration_details(scheduler_output)
            is_dummy = False
        before = time.monotonic()
        yield
        logger.info(
            "".join(
                [
                    "Iteration(",
                    str(self._iteration_index),
                    "): ",
                    str(iteration_details.num_ctx_requests),
                    " context requests, ",
                    str(iteration_details.num_ctx_tokens),
                    " context tokens, ",
                    str(iteration_details.num_generation_requests),
                    " generation requests, ",
                    str(iteration_details.num_generation_tokens),
                    " generation tokens, iteration elapsed time: ",
                    format((time.monotonic() - before) * 1000, ".2f"),
                    " ms",
                    " (dummy)" if is_dummy else "",
                ]
            )
        )
        self._iteration_index += 1

    def step(self) -> tuple[dict[int, EngineCoreOutputs], bool]:
        """Schedule, execute, and make output.

        Returns tuple of outputs and a flag indicating whether the model
        was executed.

        这是引擎的核心步进方法，每个推理步执行以下流程:

        流水线阶段:
          阶段 1 (CPU): scheduler.schedule()
            - 从 waiting 队列选取请求
            - 根据 token budget 决定本轮计算多少 token
            - 分配/复用 KV 缓存块
            - 生成 SchedulerOutput

          阶段 2 (GPU 异步): executor.execute_model(non_block=True)
            - 将 SchedulerOutput 提交到 GPU
            - non_block=True: 不等待 GPU 完成，立即返回 Future
            - GPU 执行模型前向计算（prefill 或 decode）

          阶段 3 (CPU 并行): scheduler.get_grammar_bitmask()
            - 在 GPU 计算期间，CPU 端并行计算结构化输出的 grammar 位图
            - 实现 CPU-GPU 重叠（overlap），最大化硬件利用率

          阶段 4 (GPU→CPU): future.result()
            - 等待 GPU 执行完成，获取模型输出（logits）

          阶段 5 (CPU): executor.sample_tokens(grammar_output)
            - 使用 grammar 位图约束采样，生成输出 token
            - 支持 top-k、top-p、temperature 等采样策略

          阶段 6 (CPU): scheduler.update_from_output()
            - 更新请求状态（num_computed_tokens 等）
            - 释放已完成请求的 KV 缓存块
            - 处理抢占（preemption）逻辑
            - 返回 EngineCoreOutputs 供前端使用
        """
        # 核心调度循环，每个推理步执行以下流程:
        # 1. schedule(): 调度器从等待队列中选取请求，分配 KV 缓存块，生成调度输出
        # 2. execute_model(non_block=True): 异步提交模型前向计算到 GPU，
        #    non_block=True 表示不等待 GPU 计算完成，立即返回 Future 对象
        # 3. get_grammar_bitmask(): 在 GPU 执行期间，CPU 端并行计算结构化输出的
        #    grammar 位图，用于约束 token 采样。这实现了 CPU-GPU 重叠（overlap）
        # 4. future.result(): 等待 GPU 执行完成，获取模型输出
        # 5. sample_tokens(): 使用 grammar 位图约束采样，生成输出 token
        # 6. update_from_output(): 更新调度器状态（完成请求、释放 KV 缓存等）

        # Check for any requests remaining in the scheduler - unfinished,
        # or finished and not yet removed from the batch.
        if not self.scheduler.has_requests():
            return {}, False
        scheduler_output = self.scheduler.schedule()
        # non_block=True: 异步执行，不阻塞当前线程，允许 CPU 在 GPU 计算期间
        # 执行其他工作（如计算 grammar 位图），实现 CPU-GPU 并行
        future = self.model_executor.execute_model(scheduler_output, non_block=True)
        # grammar 位图计算与 GPU 执行重叠:
        # 在 GPU 执行模型前向计算的同时，CPU 端计算结构化输出约束的位图，
        # 两者并行执行，最大化硬件利用率
        grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)
        with (
            self.log_error_detail(scheduler_output),
            self.log_iteration_details(scheduler_output),
        ):
            model_output = future.result()
            if model_output is None:
                model_output = self.model_executor.sample_tokens(grammar_output)

        # Before processing the model output, process any aborts that happened
        # during the model execution.
        self._process_aborts_queue()
        engine_core_outputs = self.scheduler.update_from_output(
            scheduler_output, model_output
        )

        return engine_core_outputs, scheduler_output.total_num_scheduled_tokens > 0

    def post_step(self, model_executed: bool) -> None:
        # When using async scheduling we can't get draft token ids in advance,
        # so we update draft token ids in the worker process and don't
        # need to update draft token ids here.
        if not self.async_scheduling and self.use_spec_decode and model_executed:
            # Take the draft token ids.
            draft_token_ids = self.model_executor.take_draft_token_ids()
            if draft_token_ids is not None:
                self.scheduler.update_draft_token_ids(draft_token_ids)

    def step_with_batch_queue(
        self,
    ) -> tuple[dict[int, EngineCoreOutputs] | None, bool]:
        """Schedule and execute batches with the batch queue.
        Note that if nothing to output in this step, None is returned.

        The execution flow is as follows:
        1. Try to schedule a new batch if the batch queue is not full.
        If a new batch is scheduled, directly return an empty engine core
        output. In other words, fulfilling the batch queue has a higher priority
        than getting model outputs.
        2. If there is no new scheduled batch, meaning that the batch queue
        is full or no other requests can be scheduled, we block until the first
        batch in the job queue is finished.
        3. Update the scheduler from the output.
        """
        # 流水线并行的批处理队列逻辑:
        # 与普通 step() 的串行模式不同，这里通过批处理队列实现调度和执行的重叠。
        # 核心思想是: 当队列未满时，优先填充队列（调度新批次）而非等待结果，
        # 从而让 GPU 始终有工作可做，消除流水线气泡。
        #
        # 工作流程:
        # 1. 如果队列未满 → 调度新批次，提交 GPU 执行，入队，立即返回 None（不等待结果）
        # 2. 如果队列已满 → 从队列尾部取出最早的结果，等待完成后更新调度器
        # 3. 这样调度和执行在时间上重叠，GPU 利用率更高

        batch_queue = self.batch_queue
        assert batch_queue is not None

        # Try to schedule a new batch if the batch queue is not full, but
        # the scheduler may return an empty batch if all requests are scheduled.
        # Note that this is not blocking.
        assert len(batch_queue) < self.batch_queue_size

        model_executed = False
        deferred_scheduler_output = None
        if self.scheduler.has_requests():
            scheduler_output = self.scheduler.schedule()
            with self.log_error_detail(scheduler_output):
                exec_future = self.model_executor.execute_model(
                    scheduler_output, non_block=True
                )
            if self.is_ec_consumer:
                model_executed = scheduler_output.total_num_scheduled_tokens > 0

            if self.is_pooling_model or not model_executed:
                # No sampling required (no requests scheduled).
                future = cast(Future[ModelRunnerOutput], exec_future)
            else:
                if not scheduler_output.pending_structured_output_tokens:
                    # We aren't waiting for any tokens, get any grammar output
                    # and sample immediately.
                    grammar_output = self.scheduler.get_grammar_bitmask(
                        scheduler_output
                    )
                    future = self.model_executor.sample_tokens(
                        grammar_output, non_block=True
                    )
                else:
                    # We need to defer sampling until we have processed the model output
                    # from the prior step.
                    deferred_scheduler_output = scheduler_output

            if not deferred_scheduler_output:
                # Add this step's future to the queue.
                batch_queue.appendleft((future, scheduler_output, exec_future))
                # 队列未满时不等待结果: 如果队列还有空位且最早的批次尚未完成，
                # 直接返回 None 让调用者尽快再次调用本方法来调度更多批次，
                # 实现"调度优先于等待"的策略，最大化流水线并行度
                if (
                    model_executed
                    and len(batch_queue) < self.batch_queue_size
                    and not batch_queue[-1][0].done()
                ):
                    # Don't block on next worker response unless the queue is full
                    # or there are no more requests to schedule.
                    return None, True

        elif not batch_queue:
            # Queue is empty. We should not reach here since this method should
            # only be called when the scheduler contains requests or the queue
            # is non-empty.
            return None, False

        # Block until the next result is available.
        future, scheduler_output, exec_model_fut = batch_queue.pop()
        with (
            self.log_error_detail(scheduler_output),
            self.log_iteration_details(scheduler_output),
        ):
            model_output = future.result()
            if model_output is None:
                # None from sample_tokens() implies that the original execute_model()
                # call failed - raise that exception.
                exec_model_fut.result()
                raise RuntimeError("unexpected error")

        # Before processing the model output, process any aborts that happened
        # during the model execution.
        self._process_aborts_queue()
        engine_core_outputs = self.scheduler.update_from_output(
            scheduler_output, model_output
        )

        # NOTE(nick): We can either handle the deferred tasks here or save
        # in a field and do it immediately once step_with_batch_queue is
        # re-called. The latter slightly favors TTFT over TPOT/throughput.
        if deferred_scheduler_output:
            # If we are doing speculative decoding with structured output,
            # we need to get the draft token ids from the prior step before
            # we can compute the grammar bitmask for the deferred request.
            if self.use_spec_decode:
                draft_token_ids = self.model_executor.take_draft_token_ids()
                assert draft_token_ids is not None
                # Update the draft token ids in the scheduler output to
                # filter out the invalid spec tokens, which will be padded
                # with -1 and skipped by the grammar bitmask computation.
                self.scheduler.update_draft_token_ids_in_output(
                    draft_token_ids, deferred_scheduler_output
                )
            # We now have the tokens needed to compute the bitmask for the
            # deferred request. Get the bitmask and call sample tokens.
            grammar_output = self.scheduler.get_grammar_bitmask(
                deferred_scheduler_output
            )
            future = self.model_executor.sample_tokens(grammar_output, non_block=True)
            batch_queue.appendleft((future, deferred_scheduler_output, exec_future))

        return engine_core_outputs, model_executed

    def _process_aborts_queue(self):
        """
        处理中止队列 —— 批量中止请求。

        流程:
        ① 检查 aborts_queue 是否有中止请求
        ② 非阻塞地取出所有中止请求 ID
        ③ 批量调用 abort_requests() (更高效)

        为什么需要单独的 aborts_queue:
        - ABORT 请求同时放入 input_queue 和 aborts_queue
        - input_queue 保证顺序: ABORT 不会在 ADD 之前处理
        - aborts_queue 允许批量处理: 在模型执行期间积累的 ABORT
          可以在执行完成后一次性处理
        - 调度器的 abort 是幂等的，重复中止不会出错
        """
        if not self.aborts_queue.empty():
            request_ids = []
            while not self.aborts_queue.empty():
                ids = self.aborts_queue.get_nowait()
                request_ids.extend((ids,) if isinstance(ids, str) else ids)
            # 批量中止更高效
            self.abort_requests(request_ids)

    def shutdown(self):
        self.structured_output_manager.clear_backend()
        if self.model_executor:
            self.model_executor.shutdown()
        if self.scheduler:
            self.scheduler.shutdown()

        # Undo the gc.freeze() from __init__ so that the objects allocated
        # during engine startup (model weights, KV caches, etc.) become
        # visible to the garbage collector again. Without this, deleting
        # the engine in-process (e.g. unit tests) leaks GPU memory.
        gc.unfreeze()
        # Tear down distributed state initialized in this EngineCore process
        # before it exits and release cached memory.
        cleanup_dist_env_and_memory()

    def profile(self, is_start: bool = True, profile_prefix: str | None = None):
        self.model_executor.profile(is_start, profile_prefix)

    def reset_mm_cache(self):
        # NOTE: Since this is mainly for debugging, we don't attempt to
        # re-sync the internal caches (P0 sender, P1 receiver)
        if self.scheduler.has_unfinished_requests():
            logger.warning(
                "Resetting the multi-modal cache when requests are "
                "in progress may lead to desynced internal caches."
            )

        # The cache either exists in EngineCore or WorkerWrapperBase
        if self.mm_receiver_cache is not None:
            self.mm_receiver_cache.clear_cache()

        self.model_executor.reset_mm_cache()

    def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        return self.scheduler.reset_prefix_cache(
            reset_running_requests, reset_connector
        )

    def reset_encoder_cache(self) -> None:
        """Reset the encoder cache to invalidate all cached encoder outputs.

        This should be called when model weights are updated to ensure
        stale vision embeddings computed with old weights are not reused.
        Clears both the scheduler's cache manager and the GPU model runner's cache.
        """
        # NOTE: Since this is mainly for debugging, we don't attempt to
        # re-sync the internal caches (P0 sender, P1 receiver)
        if self.scheduler.has_unfinished_requests():
            logger.warning(
                "Resetting the encoder cache when requests are "
                "in progress may lead to desynced internal caches."
            )

        # Reset the scheduler's encoder cache manager (logical state)
        self.scheduler.reset_encoder_cache()
        # Reset the GPU model runner's encoder cache (physical storage)
        self.model_executor.reset_encoder_cache()

    def _reset_caches(
        self,
        reset_running_requests: bool = True,
        reset_connector: bool = True,
    ) -> None:
        # reset_connector=True so external connectors clear alongside
        # local caches, matching the pause_generation(clear_cache=True)
        # contract. No-op when no connector is configured.
        self.reset_prefix_cache(
            reset_running_requests=reset_running_requests,
            reset_connector=reset_connector,
        )
        self.reset_mm_cache()
        self.reset_encoder_cache()

    def pause_scheduler(
        self, mode: PauseMode = "abort", clear_cache: bool = True
    ) -> Future | None:
        """Pause generation; behavior depends on mode.

        All pause modes queue new adds -- "abort" and "keep" skip step();
        "wait" allows step() so in-flight requests can drain.

        - ``abort``: Set PAUSED_NEW, abort all requests, wait for abort
          outputs to be sent (when running with output_queue), optionally
          clear caches, then complete the returned Future.
        - ``wait``: Set PAUSED_NEW (queue adds, keep stepping); when drained,
          optionally clear caches, then complete the returned Future.
        - ``keep``: Set PAUSED_ALL; return a Future that completes when the
          output queue is empty.
        """
        if mode not in ("keep", "abort", "wait"):
            raise ValueError(f"Invalid pause mode: {mode}")
        if mode == "wait":
            raise ValueError("'wait' mode can't be used in inproc-engine mode")

        if mode == "abort":
            self.scheduler.finish_requests(None, RequestStatus.FINISHED_ABORTED)

        pause_state = PauseState.PAUSED_ALL if mode == "keep" else PauseState.PAUSED_NEW
        self.scheduler.set_pause_state(pause_state)
        if clear_cache:
            self._reset_caches()

        return None

    def resume_scheduler(self) -> None:
        """Resume the scheduler and flush any requests queued while paused."""
        self.scheduler.set_pause_state(PauseState.UNPAUSED)

    def is_scheduler_paused(self) -> bool:
        """Return whether the scheduler is in any pause state."""
        return self.scheduler.pause_state != PauseState.UNPAUSED

    def sleep(self, level: int = 1, mode: PauseMode = "abort") -> None | Future:
        """Put the engine to sleep at the specified level.

        Args:
            level: Sleep level.
                - Level 0: Pause scheduling only. Requests are still accepted
                           but not processed. No GPU memory changes.
                - Level 1: Offload model weights to CPU, discard KV cache.
                - Level 2: Discard all GPU memory.
            mode: Pause mode - how to deal with any existing requests, see
                documentation of pause_scheduler method.
        """

        # Pause scheduler before sleeping.
        clear_prefix_cache = level >= 1
        pause_future = self.pause_scheduler(mode=mode, clear_cache=clear_prefix_cache)
        if level < 1:
            return pause_future

        # Level 1+: Delegate to executor for GPU memory management
        model_executor = self.model_executor
        if pause_future is None:
            model_executor.sleep(level)
            return None

        future = Future[Any]()

        def pause_complete(f: Future):
            try:
                f.result()  # propagate any exception
                future.set_result(model_executor.sleep(level))
            except Exception as e:
                future.set_exception(e)

        logger.info("Waiting for in-flight requests to complete before sleeping...")
        pause_future.add_done_callback(pause_complete)
        return future

    def wake_up(self, tags: list[str] | None = None):
        """Wake up the engine from sleep.

        Args:
            tags: Tags to wake up. Use ["scheduling"] for level 0 wake up.
        """
        if tags is not None and "scheduling" in tags:
            # Remove "scheduling" from tags if there are other tags to process.
            tags = [t for t in tags if t != "scheduling"]

        if tags is None or tags:
            self.model_executor.wake_up(tags)

        # Resume scheduling (applies to all levels)
        self.resume_scheduler()

    def is_sleeping(self) -> bool:
        """Check if engine is sleeping at any level."""
        return self.is_scheduler_paused() or self.model_executor.is_sleeping

    def execute_dummy_batch(self):
        self.model_executor.execute_dummy_batch()

    def add_lora(self, lora_request: LoRARequest) -> bool:
        return self.model_executor.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        return self.model_executor.remove_lora(lora_id)

    def list_loras(self) -> set[int]:
        return self.model_executor.list_loras()

    def pin_lora(self, lora_id: int) -> bool:
        return self.model_executor.pin_lora(lora_id)

    def save_sharded_state(
        self,
        path: str,
        pattern: str | None = None,
        max_size: int | None = None,
    ) -> None:
        self.model_executor.save_sharded_state(
            path=path, pattern=pattern, max_size=max_size
        )

    def collective_rpc(
        self,
        method: str | Callable[..., _R],
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
    ) -> list[_R]:
        return self.model_executor.collective_rpc(method, timeout, args, kwargs)

    def preprocess_add_request(self, request: EngineCoreRequest) -> tuple[Request, int]:
        """Preprocess the request.

        This function could be directly used in input processing thread to allow
        request initialization running in parallel with Model forward
        """
        # 请求预处理: 将 EngineCoreRequest 转换为调度器可使用的 Request 对象。
        #
        # 此方法在 Input 守护线程中调用，与 GPU 模型前向计算并行执行，
        # 从而实现 CPU 预处理与 GPU 计算的重叠（overlap）。
        #
        # 处理步骤:
        #   1. 多模态特征缓存: 如果启用了多模态缓存，检查并更新特征缓存
        #      （相同图片的编码结果可以复用，避免重复计算）
        #   2. 创建 Request 对象: 从 EngineCoreRequest 构建调度器内部的 Request
        #      （包括 block hash 计算，用于 prefix cache 匹配）
        #   3. 结构化输出初始化: 如果请求需要结构化输出（如 JSON schema），
        #      初始化 grammar 编译器（异步编译，调度器会在调度前检查编译状态）
        # Note on thread safety: no race condition.
        # `mm_receiver_cache` is reset at the end of LLMEngine init,
        # and will only be accessed in the input processing thread afterwards.
        if self.mm_receiver_cache is not None and request.mm_features:
            request.mm_features = self.mm_receiver_cache.get_and_update_features(
                request.mm_features
            )

        req = Request.from_engine_core_request(request, self.request_block_hasher)
        if req.use_structured_output:
            # Note on thread safety: no race condition.
            # `grammar_init` is only invoked in input processing thread. For
            # `structured_output_manager`, each request is independent and
            # grammar compilation is async. Scheduler always checks grammar
            # compilation status before scheduling request.
            self.structured_output_manager.grammar_init(req)
        return req, request.current_wave

    def _eep_scale_up_before_kv_init(self):
        raise NotImplementedError

    def _eep_send_engine_core_notification(
        self,
        notification_type: EEPNotificationType,
        vllm_config: VllmConfig | None = None,
    ):
        raise NotImplementedError


class EngineShutdownState(IntEnum):
    RUNNING = 0
    REQUESTED = 1
    SHUTTING_DOWN = 2


class EngineCoreProc(EngineCore):
    """
    ZMQ 包装器 —— 在后台进程中运行 EngineCore。

    架构:
    ┌─────────────────────────────────────────────────────────────────┐
    │  API Server 进程 (AsyncLLM)                                     │
    │  ┌─────────────────────────────────────────────────────────┐   │
    │  │  AsyncMPClient                                          │   │
    │  │  input_socket (ROUTER) ──→ ZMQ ──→ EngineCore 进程      │   │
    │  │  output_socket (PULL) ←── ZMQ ←── EngineCore 进程       │   │
    │  └─────────────────────────────────────────────────────────┘   │
    └─────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────┐
    │  EngineCore 进程 (EngineCoreProc)                               │
    │                                                                 │
    │  ┌─────────────────────────────────────────────────────────┐   │
    │  │  主线程: run_busy_loop()                                │   │
    │  │  while True:                                            │   │
    │  │    _process_input_queue()  ← 从 input_queue 取请求       │   │
    │  │    _process_engine_step()  → 调度+执行+放入 output_queue │   │
    │  └─────────────────────────────────────────────────────────┘   │
    │                                                                 │
    │  ┌─────────────────────────────────────────────────────────┐   │
    │  │  Input 守护线程: process_input_sockets()                │   │
    │  │  ZMQ DEALER socket → 反序列化 → input_queue             │   │
    │  └─────────────────────────────────────────────────────────┘   │
    │                                                                 │
    │  ┌─────────────────────────────────────────────────────────┐   │
    │  │  Output 守护线程: process_output_sockets()              │   │
    │  │  output_queue → 序列化 → ZMQ PUSH socket               │   │
    │  └─────────────────────────────────────────────────────────┘   │
    │                                                                 │
    │  队列:                                                          │
    │  input_queue:  Input线程 → 主线程 (请求)                        │
    │  output_queue: 主线程 → Output线程 (结果)                       │
    │  aborts_queue: Input线程 → 主线程 (中止请求)                    │
    └─────────────────────────────────────────────────────────────────┘

    设计优势:
    1. GIL 释放: ZMQ IO 线程释放 GIL，GPU 计算与网络 IO 可以并行
    2. 进程隔离: 引擎崩溃不影响 API 服务进程
    3. 数据并行: 通过 ZMQ 实现多个 EngineCore 进程间的协调
    """

    ENGINE_CORE_DEAD = b"ENGINE_CORE_DEAD"
    addresses: EngineZmqAddresses

    @instrument(span_name="EngineCoreProc init")
    def __init__(
        self,
        vllm_config: VllmConfig,
        local_client: bool,
        handshake_address: str,
        executor_class: type[Executor],
        log_stats: bool,
        client_handshake_address: str | None = None,
        tensor_queue: Queue | None = None,
        *,
        engine_index: int = 0,
    ):
        """
        初始化 EngineCoreProc。

        流程:
        ① 创建 input_queue 和 output_queue (线程安全队列)
        ② ZMQ 握手 (与前端交换地址)
        ③ 启动 Input/Output 守护线程
        ④ 初始化 EngineCore (模型、KV Cache、调度器)
        """
        # ① 创建队列
        # input_queue: Input 线程 → 主线程
        #   - Input 线程从 ZMQ socket 接收请求，反序列化后放入
        #   - 主线程从队列取出请求，交给调度器处理
        # output_queue: 主线程 → Output 线程
        #   - 主线程将执行结果放入
        #   - Output 线程取出结果，序列化后通过 ZMQ socket 发送
        # aborts_queue: Input 线程 → 主线程
        #   - 中止请求同时放入 input_queue 和 aborts_queue
        #   - aborts_queue 允许在主循环中批量处理中止
        self.input_queue = queue.Queue[tuple[EngineCoreRequestType, Any]]()
        self.output_queue = queue.Queue[tuple[int, EngineCoreOutputs] | bytes]()
        executor_fail_callback = lambda: self.input_queue.put_nowait(
            (EngineCoreRequestType.EXECUTOR_FAILED, b"")
        )

        self.engine_index = engine_index
        identity = self.engine_index.to_bytes(length=2, byteorder="little")
        self.engines_running = False
        self.shutdown_state = EngineShutdownState.RUNNING

        # Receiver for tensor IPC
        self.tensor_ipc_receiver: TensorIpcReceiver | None = None
        if tensor_queue is not None:
            self.tensor_ipc_receiver = TensorIpcReceiver(tensor_queue)
            logger.info("Using tensor IPC queue for multimodal tensor sharing")

        with self._perform_handshakes(
            handshake_address,
            identity,
            local_client,
            vllm_config,
            client_handshake_address,
        ) as addresses:
            # Set up data parallel environment.
            self.has_coordinator = addresses.coordinator_output is not None
            self.frontend_stats_publish_address = (
                addresses.frontend_stats_publish_address
            )
            logger.debug(
                "Has DP Coordinator: %s, stats publish address: %s",
                self.has_coordinator,
                self.frontend_stats_publish_address,
            )
            internal_dp_balancing = (
                self.has_coordinator
                and not vllm_config.parallel_config.data_parallel_external_lb
            )
            # Only publish request queue stats to coordinator for "internal"
            # and "hybrid" LB modes.
            self.publish_dp_lb_stats = internal_dp_balancing

            self.addresses = addresses
            self.process_input_queue_block = True
            if envs.VLLM_ELASTIC_EP_SCALE_UP_LAUNCH:
                self._eep_send_engine_core_notification(
                    EEPNotificationType.NEW_CORE_ENGINES_INIT_READY,
                    vllm_config=vllm_config,
                )
            self._init_data_parallel(vllm_config)

            super().__init__(
                vllm_config,
                executor_class,
                log_stats,
                executor_fail_callback,
                internal_dp_balancing,
            )

            # Background Threads and Queues for IO. These enable us to
            # overlap ZMQ socket IO with GPU since they release the GIL,
            # and to overlap some serialization/deserialization with the
            # model forward pass.
            # Threads handle Socket <-> Queues and core_busy_loop uses Queue.
            # ZMQ 通信的后台线程架构:
            # - input_thread: 从 ZMQ socket 读取请求，反序列化后放入 input_queue
            # - output_thread: 从 output_queue 取出结果，序列化后通过 ZMQ socket 发送
            # 这些线程释放 GIL，使得 ZMQ IO 与 GPU 计算可以真正并行执行
            ready_event = threading.Event()
            input_thread = threading.Thread(
                target=self.process_input_sockets,
                args=(
                    addresses.inputs,
                    addresses.coordinator_input,
                    identity,
                    ready_event,
                ),
                daemon=True,
            )
            input_thread.start()

            self.output_thread = threading.Thread(
                target=self.process_output_sockets,
                args=(
                    addresses.outputs,
                    addresses.coordinator_output,
                    self.engine_index,
                ),
                daemon=True,
            )
            self.output_thread.start()

            # Don't complete handshake until DP coordinator ready message is
            # received.
            while not ready_event.wait(timeout=10):
                if not input_thread.is_alive():
                    raise RuntimeError("Input socket thread died during startup")
                assert addresses.coordinator_input is not None
                logger.info("Waiting for READY message from DP Coordinator...")

    @contextmanager
    def _perform_handshakes(
        self,
        handshake_address: str,
        identity: bytes,
        local_client: bool,
        vllm_config: VllmConfig,
        client_handshake_address: str | None,
    ) -> Generator[EngineZmqAddresses, None, None]:
        """
        Perform startup handshakes.

        For DP=1 or offline mode, this is with the colocated front-end process.

        For DP>1 with internal load-balancing this is with the shared front-end
        process which may reside on a different node.

        For DP>1 with external or hybrid load-balancing, two handshakes are
        performed:
            - With the rank 0 front-end process which retrieves the
              DP Coordinator ZMQ addresses and DP process group address.
            - With the colocated front-end process which retrieves the
              client input/output socket addresses.
        with the exception of the rank 0 and colocated engines themselves which
        don't require the second handshake.

        Here, "front-end" process can mean the process containing the engine
        core client (which is the API server process in the case the API
        server is not scaled out), OR the launcher process running the
        run_multi_api_server() function in serve.py.
        """
        input_ctx = zmq.Context()
        is_local = local_client and client_handshake_address is None
        headless = not local_client
        handshake = self._perform_handshake(
            input_ctx,
            handshake_address,
            identity,
            is_local,
            headless,
            vllm_config,
            vllm_config.parallel_config,
        )
        if client_handshake_address is None:
            # We only need to handshake with one party.
            with handshake as addresses:
                yield addresses
        else:
            # We need to handshake with rank 0 front-end and our colocated frontend.
            assert local_client
            local_handshake = self._perform_handshake(
                input_ctx, client_handshake_address, identity, True, False, vllm_config
            )
            with handshake as addresses, local_handshake as client_addresses:
                # 1. Obtain DP Coordinator zmq address and DP process group address
                #    (addresses).
                # 2. Add front-end input/output addresses from colocated front-end
                #    (client_addresses).
                addresses.inputs = client_addresses.inputs
                addresses.outputs = client_addresses.outputs
                yield addresses

        # Update config which may have changed from the handshake
        vllm_config.__post_init__()

    @contextmanager
    def _perform_handshake(
        self,
        ctx: zmq.Context,
        handshake_address: str,
        identity: bytes,
        local_client: bool,
        headless: bool,
        vllm_config: VllmConfig,
        parallel_config_to_update: ParallelConfig | None = None,
    ) -> Generator[EngineZmqAddresses, None, None]:
        with make_zmq_socket(
            ctx,
            handshake_address,
            zmq.DEALER,
            identity=identity,
            linger=5000,
            bind=False,
        ) as handshake_socket:
            # Register engine with front-end.
            addresses = self.startup_handshake(
                handshake_socket, local_client, headless, parallel_config_to_update
            )
            yield addresses

            # Send ready message.
            ready_msg = {
                "status": "READY",
                "local": local_client,
                "headless": headless,
            }
            # Include config hash for DP configuration validation
            if vllm_config.parallel_config.data_parallel_size > 1:
                ready_msg["parallel_config_hash"] = (
                    vllm_config.parallel_config.compute_hash()
                )

            handshake_socket.send(msgspec.msgpack.encode(ready_msg))

    @staticmethod
    def startup_handshake(
        handshake_socket: zmq.Socket,
        local_client: bool,
        headless: bool,
        parallel_config: ParallelConfig | None = None,
    ) -> EngineZmqAddresses:
        # Send registration message.
        handshake_socket.send(
            msgspec.msgpack.encode(
                {
                    "status": "HELLO",
                    "local": local_client,
                    "headless": headless,
                }
            )
        )

        # Receive initialization message.
        logger.debug("Waiting for init message from front-end.")
        if not handshake_socket.poll(timeout=HANDSHAKE_TIMEOUT_MINS * 60_000):
            raise RuntimeError(
                "Did not receive response from front-end "
                f"process within {HANDSHAKE_TIMEOUT_MINS} "
                f"minutes"
            )
        init_bytes = handshake_socket.recv()
        init_message: EngineHandshakeMetadata = msgspec.msgpack.decode(
            init_bytes, type=EngineHandshakeMetadata
        )
        logger.debug("Received init message: %s", init_message)

        if parallel_config is not None:
            for key, value in init_message.parallel_config.items():
                setattr(parallel_config, key, value)

        return init_message.addresses

    @staticmethod
    def run_engine_core(*args, dp_rank: int = 0, local_dp_rank: int = 0, **kwargs):
        """Launch EngineCore busy loop in background process."""

        # Ensure we can serialize transformer config after spawning
        maybe_register_config_serialize_by_value()

        engine_core: EngineCoreProc | None = None
        signal_callback: SignalCallback | None = None
        try:
            vllm_config: VllmConfig = kwargs["vllm_config"]
            parallel_config: ParallelConfig = vllm_config.parallel_config
            data_parallel = parallel_config.data_parallel_size > 1 or dp_rank > 0
            if data_parallel:
                parallel_config.data_parallel_rank_local = local_dp_rank
                process_title = f"EngineCore_DP{dp_rank}"
            else:
                process_title = "EngineCore"
            set_process_title(process_title)
            maybe_init_worker_tracer("vllm.engine_core", "engine_core", process_title)
            decorate_logs()
            if parallel_config.numa_bind:
                numa_utils.log_current_affinity_state(process_title)

            if data_parallel and vllm_config.kv_transfer_config is not None:
                # modify the engine_id and append the local_dp_rank to it to ensure
                # that the kv_transfer_config is unique for each DP rank.
                vllm_config.kv_transfer_config.engine_id = (
                    f"{vllm_config.kv_transfer_config.engine_id}_dp{local_dp_rank}"
                )
                logger.debug(
                    "Setting kv_transfer_config.engine_id to %s",
                    vllm_config.kv_transfer_config.engine_id,
                )

            parallel_config.data_parallel_index = dp_rank
            if data_parallel and vllm_config.model_config.is_moe:
                # Set data parallel rank for this engine process.
                parallel_config.data_parallel_rank = dp_rank
                engine_core = DPEngineCoreProc(*args, **kwargs)
            else:
                # Non-MoE DP ranks are completely independent, so treat like DP=1.
                # Note that parallel_config.data_parallel_index will still reflect
                # the original DP rank.
                parallel_config.data_parallel_size = 1
                parallel_config.data_parallel_size_local = 1
                parallel_config.data_parallel_rank = 0
                engine_core = EngineCoreProc(*args, engine_index=dp_rank, **kwargs)

            assert engine_core is not None

            def wakeup_engine():
                # Wakes up idle engine via input_queue when shutdown is requested
                # Not safe in a signal handler - we may interrupt the main thread
                # while it is holding the non-reentrant input_queue.mutex
                engine_core.input_queue.put_nowait((EngineCoreRequestType.WAKEUP, None))

            signal_callback = SignalCallback(wakeup_engine)

            def signal_handler(signum, frame):
                engine_core.shutdown_state = EngineShutdownState.REQUESTED
                signal_callback.trigger()

            signal.signal(signal.SIGTERM, signal_handler)
            signal.signal(signal.SIGINT, signal_handler)

            engine_core.run_busy_loop()

        except SystemExit:
            logger.debug("EngineCore exiting.")
            raise
        except Exception as e:
            if engine_core is None:
                logger.exception("EngineCore failed to start.")
            else:
                logger.exception("EngineCore encountered a fatal error.")
                engine_core._send_engine_dead()
            raise e
        finally:
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            if signal_callback is not None:
                signal_callback.stop()
            if engine_core is not None:
                engine_core.shutdown()

    def _init_data_parallel(self, vllm_config: VllmConfig):
        pass

    def has_work(self) -> bool:
        """Returns true if the engine should be stepped."""
        return (
            self.engines_running
            or self.scheduler.has_requests()
            or bool(self.batch_queue)
        )

    def is_running(self) -> bool:
        """Returns true if shutdown has not been requested."""
        return self.shutdown_state == EngineShutdownState.RUNNING

    def run_busy_loop(self):
        """
        EngineCore 主循环 —— 持续处理请求和执行推理。

        流程:
        ┌─────────────────────────────────────────────────────────┐
        │  while True:                                            │
        │    ① _process_input_queue()                             │
        │       从 input_queue 取出请求，交给调度器                 │
        │       如果没有请求，阻塞等待                              │
        │                                                         │
        │    ② _process_engine_step()                             │
        │       调度 → 执行模型 → 采样 → 更新状态                  │
        │       结果放入 output_queue                              │
        └─────────────────────────────────────────────────────────┘
        """
        while self._handle_shutdown():
            # ① 处理输入队列 (阻塞直到有工作)
            self._process_input_queue()
            # ② 执行引擎步进 (调度+执行)
            self._process_engine_step()

        raise SystemExit

    def _process_input_queue(self):
        """
        处理输入队列 —— 从 input_queue 取出请求并交给调度器。

        流程:
        ┌─────────────────────────────────────────────────────────┐
        │  while 没有工作 且 引擎运行中:                            │
        │    ① 通知空闲回调                                        │
        │    ② 如果 input_queue 为空:                              │
        │       - 清空 aborts_queue                               │
        │       - 阻塞等待新请求                                   │
        │    ③ 从 input_queue 取出请求                             │
        │    ④ 调用 _handle_client_request() 处理请求              │
        │                                                         │
        │  处理完阻塞等待后，非阻塞地处理剩余请求                    │
        └─────────────────────────────────────────────────────────┘

        队列交互:
        ┌──────────────┐     input_queue      ┌──────────────┐
        │ Input 线程    │ ──────────────────→  │ 主线程       │
        │ (ZMQ 接收)   │                      │ (调度处理)   │
        │              │     aborts_queue     │              │
        │              │ ──────────────────→  │              │
        └──────────────┘                      └──────────────┘
        """
        waited = False
        while not self.has_work() and self.is_running():
            # 通知等待引擎空闲的回调
            self._notify_idle_state_callbacks()
            if self.input_queue.empty():
                # 清空 aborts_queue (所有 abort 也通过 input_queue 处理)
                with self.aborts_queue.mutex:
                    self.aborts_queue.queue.clear()
                if logger.isEnabledFor(DEBUG):
                    logger.debug("EngineCore waiting for work.")
                    waited = True
            block = self.process_input_queue_block
            try:
                # 从 input_queue 取出请求 (阻塞或非阻塞)
                req = self.input_queue.get(block=block)
                self._handle_client_request(*req)
            except queue.Empty:
                break
            if not block:
                break

        if waited:
            logger.debug("EngineCore loop active.")

        # 非阻塞地处理剩余请求
        while not self.input_queue.empty():
            req = self.input_queue.get_nowait()
            self._handle_client_request(*req)

    def _process_engine_step(self) -> bool:
        """
        执行引擎步进 —— 调度 + 执行 + 输出。

        流程:
        ① step_fn(): 调度 → 执行模型 → 采样 → 更新状态
        ② 将结果放入 output_queue (Output 线程会取出发送)
        ③ post_step(): 更新推测解码的草稿 token

        队列交互:
        ┌──────────────┐     output_queue     ┌──────────────┐
        │ 主线程       │ ──────────────────→  │ Output 线程  │
        │ (调度+执行)  │                      │ (ZMQ 发送)   │
        └──────────────┘                      └──────────────┘
        """
        # ① 执行引擎步进
        outputs, model_executed = self.step_fn()
        # ② 将结果放入 output_queue
        for output in outputs.items() if outputs else ():
            self.output_queue.put_nowait(output)
        # ③ 后处理 (更新推测解码草稿)
        self.post_step(model_executed)

        # 如果没有模型执行但仍有调度工作 (如等待远程 KV)，
        # 短暂释放 GIL 让后台传输线程有机会推进
        if not model_executed and self.scheduler.has_requests():
            time.sleep(0.001)

        return model_executed

    def _notify_idle_state_callbacks(self) -> None:
        while self._idle_state_callbacks:
            callback = self._idle_state_callbacks.pop()
            callback(self)

    def _handle_shutdown(self) -> bool:
        # Check if shutdown was requested and handle it
        if self.shutdown_state == EngineShutdownState.RUNNING:
            return True

        if self.shutdown_state == EngineShutdownState.REQUESTED:
            shutdown_timeout = self.vllm_config.shutdown_timeout

            logger.info("Shutdown initiated (timeout=%d)", shutdown_timeout)

            if shutdown_timeout == 0:
                num_requests = self.scheduler.get_num_unfinished_requests()
                if num_requests > 0:
                    logger.info("Aborting %d requests", num_requests)
                aborted_reqs = self.scheduler.finish_requests(
                    None, RequestStatus.FINISHED_ABORTED
                )
                self._send_abort_outputs(aborted_reqs)
            else:
                num_requests = self.scheduler.get_num_unfinished_requests()
                if num_requests > 0:
                    logger.info(
                        "Draining %d in-flight requests (timeout=%ds)",
                        num_requests,
                        shutdown_timeout,
                    )

            self.shutdown_state = EngineShutdownState.SHUTTING_DOWN

        # Exit when no work remaining
        if not self.has_work():
            logger.info("Shutdown complete")
            return False

        return True

    def _handle_client_request(
        self, request_type: EngineCoreRequestType, request: Any
    ) -> None:
        """
        分发客户端请求 —— 根据请求类型执行不同操作。

        请求类型:
        ┌─────────────────────────────────────────────────────────┐
        │  WAKEUP: 唤醒引擎 (无操作)                              │
        │  ADD: 添加新请求 → scheduler.add_request()              │
        │  ABORT: 中止请求 → scheduler.finish_requests()          │
        │  UTILITY: 工具调用 → 执行方法并返回结果                  │
        │  EXECUTOR_FAILED: 执行器失败 → 抛出异常                 │
        └─────────────────────────────────────────────────────────┘
        """
        if request_type == EngineCoreRequestType.WAKEUP:
            return
        elif request_type == EngineCoreRequestType.ADD:
            # 添加新请求
            req, request_wave = request
            if self._reject_add_in_shutdown(req):
                return
            self.add_request(req, request_wave)
        elif request_type == EngineCoreRequestType.ABORT:
            # 中止请求
            self.abort_requests(request)
        elif request_type == EngineCoreRequestType.UTILITY:
            # 工具调用 (如 sleep, wake_up, reset_prefix_cache 等)
            client_idx, call_id, method_name, args = request
            if self._reject_utility_in_shutdown(client_idx, call_id, method_name):
                return
            output = UtilityOutput(call_id)
            # 延迟查找工具方法，失败时会返回错误
            get_result = lambda: (
                (method := getattr(self, method_name))
                and method(*self._convert_msgspec_args(method, args))
            )
            enqueue_output = lambda out: self.output_queue.put_nowait(
                (client_idx, EngineCoreOutputs(utility_output=out))
            )
            self._invoke_utility_method(method_name, get_result, output, enqueue_output)
        elif request_type == EngineCoreRequestType.EXECUTOR_FAILED:
            # 执行器失败
            raise RuntimeError("Executor failed.")
        else:
            logger.error(
                "Unrecognized input request type encountered: %s", request_type
            )

    def _reject_add_in_shutdown(self, request: Request) -> bool:
        if self.shutdown_state == EngineShutdownState.RUNNING:
            return False

        logger.info("Rejecting request %s (server shutting down)", request.request_id)
        self._send_abort_outputs_to_client([request.request_id], request.client_index)
        return True

    def _reject_utility_in_shutdown(
        self, client_idx: int, call_id: int, method_name: str
    ) -> bool:
        if self.shutdown_state == EngineShutdownState.RUNNING:
            return False

        logger.warning("Rejecting utility call %s (server shutting down)", method_name)
        output = UtilityOutput(call_id, failure_message="Server shutting down")
        self.output_queue.put_nowait(
            (client_idx, EngineCoreOutputs(utility_output=output))
        )
        return True

    @staticmethod
    def _invoke_utility_method(
        name: str, get_result: Callable, output: UtilityOutput, enqueue_output: Callable
    ):
        try:
            result = get_result()
            if isinstance(result, Future):
                # Defer utility output handling until future completion.
                callback = lambda future: EngineCoreProc._invoke_utility_method(
                    name, future.result, output, enqueue_output
                )
                result.add_done_callback(callback)
                return
            output.result = UtilityResult(result)
        except Exception as e:
            logger.exception("Invocation of %s method failed", name)
            output.failure_message = f"Call to {name} method failed: {str(e)}"
        enqueue_output(output)

    @staticmethod
    def _convert_msgspec_args(method, args):
        """If a provided arg type doesn't match corresponding target method
        arg type, try converting to msgspec object."""
        if not args:
            return args
        arg_types = signature(method).parameters.values()
        assert len(args) <= len(arg_types)
        return tuple(
            msgspec.convert(v, type=p.annotation)
            if isclass(p.annotation)
            and issubclass(p.annotation, msgspec.Struct)
            and not isinstance(v, p.annotation)
            else v
            for v, p in zip(args, arg_types)
        )

    def _send_engine_dead(self):
        """Send EngineDead status to the EngineCoreClient."""

        # Put ENGINE_CORE_DEAD in the queue.
        self.output_queue.put_nowait(EngineCoreProc.ENGINE_CORE_DEAD)

        # Wait until msg sent by the daemon before shutdown.
        self.output_thread.join(timeout=5.0)
        if self.output_thread.is_alive():
            logger.fatal(
                "vLLM shutdown signal from EngineCore failed "
                "to send. Please report this issue."
            )

    def process_input_sockets(
        self,
        input_addresses: list[str],
        coord_input_address: str | None,
        identity: bytes,
        ready_event: threading.Event,
    ):
        """
        Input Socket IO 线程 —— 从 ZMQ socket 接收请求并放入 input_queue。

        流程:
        ┌─────────────────────────────────────────────────────────────────┐
        │  ① 创建 ZMQ DEALER socket 连接到前端 ROUTER socket            │
        │  ② 创建 ZMQ XSUB socket 连接到 DPCoordinator (如果有的话)      │
        │  ③ 注册 Poller，等待消息                                       │
        │  ④ 循环:                                                       │
        │     - 从 socket 接收消息                                       │
        │     - 反序列化请求 (MsgpackDecoder)                            │
        │     - 如果是 ADD 请求: 预处理 (preprocess_add_request)         │
        │     - 如果是 ABORT 请求: 同时放入 aborts_queue                 │
        │     - 放入 input_queue (主线程会取出处理)                       │
        └─────────────────────────────────────────────────────────────────┘

        Socket 类型:
        ┌──────────────┐                    ┌──────────────┐
        │ 前端进程      │  ROUTER/DEALER     │ EngineCore   │
        │ (AsyncMPClient)│ ←───────────────→ │ Input 线程   │
        └──────────────┘                    └──────────────┘

        ┌──────────────┐  XSUB/XPUB         ┌──────────────┐
        │ DPCoordinator│ ─────────────────→  │ EngineCore   │
        │              │  (订阅消息)          │ Input 线程   │
        └──────────────┘                    └──────────────┘
        """
        # 创建解码器 (支持带外张量 IPC)
        add_request_decoder = MsgpackDecoder(
            EngineCoreRequest, oob_tensor_provider=self.tensor_ipc_receiver
        )
        generic_decoder = MsgpackDecoder(oob_tensor_provider=self.tensor_ipc_receiver)

        with ExitStack() as stack, zmq.Context() as ctx:
            # ① 创建 ZMQ DEALER socket 连接到前端
            input_sockets = [
                stack.enter_context(
                    make_zmq_socket(
                        ctx, input_address, zmq.DEALER, identity=identity, bind=False
                    )
                )
                for input_address in input_addresses
            ]

            # ② 创建 ZMQ XSUB socket 连接到 DPCoordinator
            if coord_input_address is None:
                coord_socket = None
            else:
                coord_socket = stack.enter_context(
                    make_zmq_socket(
                        ctx,
                        coord_input_address,
                        zmq.XSUB,
                        identity=identity,
                        bind=False,
                    )
                )
                # 发送订阅消息给协调器
                coord_socket.send(b"\x01")

            # ③ 注册 Poller，发送 READY 消息
            poller = zmq.Poller()
            ready_response = EngineCoreReadyResponse(
                max_model_len=self.vllm_config.model_config.max_model_len,
                num_gpu_blocks=self.vllm_config.cache_config.num_gpu_blocks or 0,
                dp_stats_address=self.frontend_stats_publish_address,
                dtype=str(self.vllm_config.model_config.dtype).removeprefix("torch."),
                vllm_version=VLLM_VERSION,
            )
            ready_payload = msgspec.msgpack.encode(ready_response)
            for input_socket in input_sockets:
                # 发送 READY 消息给前端 (握手)
                input_socket.send(ready_payload)
                poller.register(input_socket, zmq.POLLIN)

            if coord_socket is not None:
                # 等待协调器的 READY 消息
                assert coord_socket.recv() == b"READY"
                poller.register(coord_socket, zmq.POLLIN)

            ready_event.set()
            del ready_event

            # ④ 主循环: 接收并处理消息
            while True:
                for input_socket, _ in poller.poll():
                    # 接收消息: (RequestType, RequestData)
                    type_frame, *data_frames = input_socket.recv_multipart(copy=False)

                    # 忽略 DP 协调器的 READY 消息
                    if type_frame.buffer == b"READY":
                        assert input_socket == coord_socket
                        continue

                    request_type = EngineCoreRequestType(bytes(type_frame.buffer))

                    # 反序列化请求
                    request: Any
                    if request_type == EngineCoreRequestType.ADD:
                        # ADD 请求: 解码并预处理
                        req: EngineCoreRequest = add_request_decoder.decode(data_frames)
                        try:
                            request = self.preprocess_add_request(req)
                        except Exception:
                            self._handle_request_preproc_error(req)
                            continue
                    else:
                        # 其他请求: 通用解码
                        request = generic_decoder.decode(data_frames)

                        if request_type == EngineCoreRequestType.ABORT:
                            # ABORT 请求同时放入 aborts_queue
                            # 这样主循环可以批量处理中止请求
                            self.aborts_queue.put_nowait(request)

                    # 放入 input_queue (主线程会取出处理)
                    self.input_queue.put_nowait((request_type, request))

    def process_output_sockets(
        self, output_paths: list[str], coord_output_path: str | None, engine_index: int
    ):
        """
        Output Socket IO 线程 —— 从 output_queue 取出结果并通过 ZMQ 发送。

        流程:
        ┌─────────────────────────────────────────────────────────────────┐
        │  ① 创建 ZMQ PUSH socket 连接到前端 PULL socket                 │
        │  ② 创建 ZMQ PUSH socket 连接到 DPCoordinator (如果有的话)       │
        │  ③ 循环:                                                       │
        │     - 从 output_queue 取出结果                                  │
        │     - 如果是 ENGINE_CORE_DEAD: 发送给所有 socket 并退出         │
        │     - 序列化结果 (MsgpackEncoder)                              │
        │     - 通过 socket 发送给对应的前端                              │
        │     - 复用发送缓冲区以减少内存分配                               │
        └─────────────────────────────────────────────────────────────────┘

        Socket 类型:
        ┌──────────────┐                    ┌──────────────┐
        │ 前端进程      │  PULL/PUSH         │ EngineCore   │
        │ (AsyncMPClient)│ ←─────────────── │ Output 线程  │
        └──────────────┘                    └──────────────┘

        ┌──────────────┐  PUSH/PULL          ┌──────────────┐
        │ DPCoordinator│ ←────────────────── │ EngineCore   │
        │              │  (统计+wave通知)     │ Output 线程  │
        └──────────────┘                    └──────────────┘

        缓冲区复用:
        - 输出结果可能很大 (包含 logprobs 等)
        - 每次分配新缓冲区开销大
        - 使用 reuse_buffers 列表复用已完成发送的缓冲区
        """
        # 创建编码器
        encoder = MsgpackEncoder()
        # 发送缓冲区复用列表
        reuse_buffers: list[bytearray] = []
        # 待完成的发送操作 (用于回收缓冲区)
        pending = deque[tuple[zmq.MessageTracker, Any, bytearray]]()

        # linger=4000: 确保 ENGINE_CORE_DEAD 消息在关闭 socket 前发送
        with ExitStack() as stack, zmq.Context() as ctx:
            # ① 创建 ZMQ PUSH socket 连接到前端
            sockets = [
                stack.enter_context(
                    make_zmq_socket(ctx, output_path, zmq.PUSH, linger=4000)
                )
                for output_path in output_paths
            ]

            # ② 创建 ZMQ PUSH socket 连接到 DPCoordinator
            coord_socket = (
                stack.enter_context(
                    make_zmq_socket(
                        ctx, coord_output_path, zmq.PUSH, bind=False, linger=4000
                    )
                )
                if coord_output_path is not None
                else None
            )
            max_reuse_bufs = len(sockets) + 1

            # ③ 主循环
            while True:
                # 从 output_queue 取出结果
                output = self.output_queue.get()

                # 处理引擎死亡信号
                if output == EngineCoreProc.ENGINE_CORE_DEAD:
                    for socket in sockets:
                        socket.send(output)
                    break

                assert not isinstance(output, bytes)
                client_index, outputs = output
                outputs.engine_index = engine_index

                # 协调器消息: 直接发送，不复用缓冲区
                if client_index == -1:
                    assert coord_socket is not None
                    coord_socket.send_multipart(encoder.encode(outputs))
                    continue

                # 回收 ZMQ 已完成发送的缓冲区
                while pending and pending[-1][0].done:
                    reuse_buffers.append(pending.pop()[2])

                # 序列化并发送
                buffer = reuse_buffers.pop() if reuse_buffers else bytearray()
                buffers = encoder.encode_into(outputs, buffer)
                tracker = sockets[client_index].send_multipart(
                    buffers, copy=False, track=True
                )
                # 跟踪发送状态，完成后回收缓冲区
                if not tracker.done:
                    ref = outputs if len(buffers) > 1 else None
                    pending.appendleft((tracker, ref, buffer))
                elif len(reuse_buffers) < max_reuse_bufs:
                    reuse_buffers.append(buffer)

    def _handle_request_preproc_error(self, request: EngineCoreRequest) -> None:
        """Log and return a request-scoped error response for exceptions raised
        from the add request preprocessing in the input socket processing thread.
        """
        logger.exception(
            "Unexpected error pre-processing request %s", request.request_id
        )
        self._send_error_outputs_to_client([request.request_id], request.client_index)

    def pause_scheduler(
        self, mode: PauseMode = "abort", clear_cache: bool = True
    ) -> Future | None:
        """Pause generation; behavior depends on mode.

        All pause modes queue new adds -- "abort" and "keep" skip step();
        "wait" allows step() so in-flight requests can drain.

        - ``abort``: Set PAUSED_NEW, abort all requests, wait for abort
          outputs to be sent (when running with output_queue), optionally
          clear caches, then complete the returned Future.
        - ``wait``: Set PAUSED_NEW (queue adds, keep stepping); when drained,
          optionally clear caches, then complete the returned Future.
        - ``keep``: Set PAUSED_ALL; return a Future that completes when the
          output queue is empty.
        """
        if mode not in ("keep", "abort", "wait"):
            raise ValueError(f"Invalid pause mode: {mode}")

        def engine_idle_callback(engine: "EngineCoreProc", future: Future[Any]) -> None:
            if clear_cache:
                engine._reset_caches()
            future.set_result(None)

        if mode == "abort":
            aborted_reqs = self.scheduler.finish_requests(
                None, RequestStatus.FINISHED_ABORTED
            )
            self._send_abort_outputs(aborted_reqs)

        pause_state = PauseState.PAUSED_ALL if mode == "keep" else PauseState.PAUSED_NEW
        self.scheduler.set_pause_state(pause_state)

        if self._pause_complete():
            if clear_cache:
                self._reset_caches()
            return None

        future = Future[Any]()
        self._idle_state_callbacks.append(partial(engine_idle_callback, future=future))
        return future

    def _pause_complete(self) -> bool:
        """Returns True if the pause has fully completed and the caller can
        return ``None`` synchronously; False if the pause is still pending
        and the caller should register an idle-state callback to finish it.
        """
        return not self.has_work()

    def _send_finish_outputs_to_client(
        self, req_ids: list[str], client_index: int, finish_reason: FinishReason
    ) -> None:
        outputs = [
            EngineCoreOutput(req_id, [], finish_reason=finish_reason)
            for req_id in req_ids
        ]
        eco = EngineCoreOutputs(finished_requests=req_ids, outputs=outputs)
        self.output_queue.put_nowait((client_index, eco))

    def _send_abort_outputs_to_client(
        self, req_ids: list[str], client_index: int
    ) -> None:
        self._send_finish_outputs_to_client(req_ids, client_index, FinishReason.ABORT)

    def _send_error_outputs_to_client(
        self, req_ids: list[str], client_index: int
    ) -> None:
        self._send_finish_outputs_to_client(req_ids, client_index, FinishReason.ERROR)

    def _send_abort_outputs(self, aborted_reqs: list[tuple[str, int]]) -> None:
        # TODO(nick) this will be moved inside the scheduler
        if aborted_reqs:
            # Map client_index to list of request_ids that belong to that client.
            by_client = defaultdict[int, set[str]](set)
            for req_id, client_index in aborted_reqs:
                by_client[client_index].add(req_id)
            for client_index, req_ids in by_client.items():
                self._send_abort_outputs_to_client(list(req_ids), client_index)


class DPEngineCoreProc(EngineCoreProc):
    """
    数据并行引擎核心进程 —— 用于 MoE 模型的 DP 部署。

    与 EngineCoreProc 的区别:
    1. 支持 DP 协调: 通过 all-reduce 同步全局状态
    2. 支持 dummy batch: 无本地请求时执行空 batch 保持 MoE all-to-all 同步
    3. 支持 wave 机制: 引擎在 running/paused 状态之间交替
    4. 支持弹性 EP: 运行时动态增减 DP rank

    Wave 机制:
    ┌─────────────────────────────────────────────────────────────┐
    │  1. 所有 DP rank 空闲 → PAUSED 状态                         │
    │  2. 新请求到达 → DPCoordinator 广播 START_DP_WAVE           │
    │  3. 所有 DP rank 唤醒 → RUNNING 状态                        │
    │  4. 处理请求                                                │
    │  5. 所有 DP rank 空闲 → all-reduce 确认 → PAUSED (wave++)   │
    └─────────────────────────────────────────────────────────────┘

    Dummy batch:
    ┌─────────────────────────────────────────────────────────────┐
    │  MoE 的 all-to-all 要求所有 DP rank 同步参与。               │
    │  如果某个 rank 没有本地请求，仍需执行 dummy batch             │
    │  以保持与其他 rank 的同步。                                   │
    └─────────────────────────────────────────────────────────────┘
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_client: bool,
        handshake_address: str,
        executor_class: type[Executor],
        log_stats: bool,
        client_handshake_address: str | None = None,
        tensor_queue: Queue | None = None,
    ):
        assert vllm_config.model_config.is_moe, (
            "DPEngineCoreProc should only be used for MoE models"
        )

        # 步计数器: 用于每 N 步与 DP peer 同步状态
        self.step_counter = 0
        # 当前 wave 编号
        self.current_wave = 0
        # 上一次发布的请求计数
        self.last_counts = (0, 0)

        # 两阶段暂停协议状态:
        # pending_pause=True 时，引擎继续执行 (dummy batch)
        # 等待所有 DP rank 也设置 pending_pause
        # all-reduce 确认后，设置 ignore_start_dp_wave 防止过期消息唤醒引擎
        self.pending_pause = False
        self.ignore_start_dp_wave = False

        from vllm.distributed.elastic_ep.elastic_state import ElasticEPScalingState

        self.eep_scaling_state: ElasticEPScalingState | None = None

        # 初始化引擎
        dp_rank = vllm_config.parallel_config.data_parallel_rank
        super().__init__(
            vllm_config,
            local_client,
            handshake_address,
            executor_class,
            log_stats,
            client_handshake_address,
            engine_index=dp_rank,
            tensor_queue=tensor_queue,
        )

    def _init_data_parallel(self, vllm_config: VllmConfig):
        # Configure GPUs and stateless process group for data parallel.
        parallel_config = vllm_config.parallel_config
        dp_rank = parallel_config.data_parallel_rank
        dp_size = parallel_config.data_parallel_size
        local_dp_rank = parallel_config.data_parallel_rank_local

        assert dp_size > 1
        assert local_dp_rank is not None
        assert 0 <= local_dp_rank <= dp_rank < dp_size

        self.dp_rank = dp_rank
        self.dp_size = dp_size
        dp_group, dp_store = parallel_config.stateless_init_dp_group(return_store=True)
        self.dp_group, self.dp_store = dp_group, dp_store

    def shutdown(self):
        super().shutdown()
        if dp_group := getattr(self, "dp_group", None):
            stateless_destroy_torch_distributed_process_group(dp_group)

    def _pause_complete(self) -> bool:
        """Two-phase DP-aware pause.

        Phase 1: Set local pause state and ``pending_pause`` flag. If the
        engines are idle, kick-start them by setting ``engines_running`` to
        True so ranks enter the stepping loop and reach the all-reduce
        consensus checkpoint in ``_has_global_unfinished_reqs``.

        Phase 2 (in ``_has_global_unfinished_reqs``): Once the all-reduce
        confirms that **all** ranks have ``pending_pause`` set, collectively
        stop stepping and set ``ignore_start_dp_wave`` so that stale
        ``START_DP_WAVE`` messages cannot re-wake any engine.
        """
        self.pending_pause = True
        self.engines_running = True

        return False

    def add_request(self, request: Request, request_wave: int = 0):
        super().add_request(request, request_wave)
        if self.has_coordinator and request_wave != self.current_wave:
            if request_wave > self.current_wave:
                self.current_wave = request_wave
            elif (
                not self.engines_running
                and self.scheduler.pause_state == PauseState.UNPAUSED
            ):
                # Request received for an already-completed wave, notify
                # front-end that we need to start the next one.
                self.engines_running = True
                self.output_queue.put_nowait(
                    (-1, EngineCoreOutputs(start_wave=self.current_wave))
                )

    def resume_scheduler(self):
        if self.pending_pause or (self.engines_running and self.ignore_start_dp_wave):
            raise RuntimeError(
                "resume_scheduler called while pause is still in "
                "flight. Wait for the pause future to resolve before "
                "resuming."
            )
        if self.engines_running:
            logger.debug("Resume called while engines are not paused, ignoring.")
            return

        super().resume_scheduler()
        self.ignore_start_dp_wave = False

        # Barrier: wait for all DP ranks to have resumed (and cleared
        # ignore_start_dp_wave) before any rank starts stepping. Uses
        # the existing all-reduce which is safe because engines are
        # stopped.
        has_global_unfinished = ParallelConfig.has_unfinished_dp(
            self.dp_group, self.scheduler.has_unfinished_requests()
        )

        if has_global_unfinished:
            self.engines_running = True

    def barrier(self):
        """Blocking barrier on the DP process group (test-only utility)."""
        import torch.distributed as dist

        dist.barrier(group=self.dp_group)

    def _handle_client_request(
        self, request_type: EngineCoreRequestType, request: Any
    ) -> None:
        """
        处理客户端请求 —— DPEngineCoreProc 版本。

        与 EngineCoreProc 的区别:
        - 新增 START_DP_WAVE 处理: DPCoordinator 广播的 wave 唤醒消息
        - 其他请求类型委托给父类处理
        """
        if request_type == EngineCoreRequestType.START_DP_WAVE:
            # 处理 wave 唤醒消息
            if self.ignore_start_dp_wave:
                # 忽略过期的 wave 消息 (暂停协议已确认)
                return
            new_wave, exclude_eng_index = request
            if exclude_eng_index != self.engine_index and (
                new_wave >= self.current_wave
            ):
                self.current_wave = new_wave
                if not self.engines_running:
                    logger.debug(
                        "EngineCore starting idle loop for wave %d.",
                        new_wave,
                    )
                    # 唤醒引擎
                    self.engines_running = True
        else:
            # 其他请求委托给父类
            super()._handle_client_request(request_type, request)

    def _maybe_publish_request_counts(self):
        if not self.publish_dp_lb_stats:
            return

        # Publish our request counts (if they've changed).
        counts = self.scheduler.get_request_counts()
        if counts != self.last_counts:
            self.last_counts = counts
            stats = SchedulerStats(
                *counts, step_counter=self.step_counter, current_wave=self.current_wave
            )
            self.output_queue.put_nowait((-1, EngineCoreOutputs(scheduler_stats=stats)))

    def run_busy_loop(self):
        """
        数据并行的主循环 —— 与 EngineCoreProc.run_busy_loop 的区别:
        ① 支持 dummy batch: 无本地请求时执行空 batch 保持 MoE 同步
        ② 支持 wave 机制: 通过 all-reduce 同步全局状态
        ③ 支持弹性 EP: 运行时动态增减 DP rank

        流程:
        ┌─────────────────────────────────────────────────────────┐
        │  while True:                                            │
        │    ① _process_input_queue()  ← 处理输入请求             │
        │    ② _maybe_publish_request_counts() → 发布负载统计     │
        │    ③ _process_engine_step()  → 调度+执行                │
        │    ④ 如果没有执行:                                       │
        │       - 没有本地请求且引擎暂停 → 跳过                   │
        │       - 否则 → execute_dummy_batch() 保持 MoE 同步      │
        │    ⑤ _has_global_unfinished_reqs() → all-reduce 同步    │
        │       - 所有 rank 空闲 → PAUSED                         │
        │       - 有 rank 有请求 → RUNNING                        │
        └─────────────────────────────────────────────────────────┘
        """
        # 循环直到收到 SIGINT 或 SIGTERM
        while self._handle_shutdown():
            # ① 处理输入队列
            self._process_input_queue()
            # 发布请求计数 (调度前后各一次，确保数据新鲜)
            self._maybe_publish_request_counts()

            # 处理弹性 EP 状态
            if self.eep_scaling_state is not None:
                _ = self.eep_scaling_state.progress()
                if self.eep_scaling_state.is_complete():
                    if self.eep_scaling_state.worker_type == "removing":
                        raise SystemExit
                    self.process_input_queue_block = True
                    self.eep_scaling_state = None

            # ② 执行引擎步进
            executed = self._process_engine_step()
            self._maybe_publish_request_counts()

            # ③ 处理 dummy batch
            local_unfinished_reqs = self.scheduler.has_unfinished_requests()
            if not executed:
                if not local_unfinished_reqs and not self.engines_running:
                    # 所有引擎都空闲，跳过
                    continue

                # 引擎处于运行状态但没有本地请求
                # 执行 dummy batch 保持 MoE all-to-all 同步
                with self.log_iteration_details(None):
                    self.execute_dummy_batch()

            # ④ all-reduce 同步全局状态
            self.engines_running = self._has_global_unfinished_reqs(
                local_unfinished_reqs
            )

            if not self.engines_running:
                if self.dp_rank == 0 or not self.has_coordinator:
                    # Notify client that we are pausing the loop.
                    logger.debug(
                        "Wave %d finished, pausing engine loop.", self.current_wave
                    )
                    # In the coordinator case, dp rank 0 sends updates to the
                    # coordinator. Otherwise (offline spmd case), each rank
                    # sends the update to its colocated front-end process.
                    client_index = -1 if self.has_coordinator else 0
                    self.output_queue.put_nowait(
                        (
                            client_index,
                            EngineCoreOutputs(wave_complete=self.current_wave),
                        )
                    )
                # Increment wave count and reset step counter.
                self.current_wave += 1
                self.step_counter = 0

        raise SystemExit

    def _has_global_unfinished_reqs(self, local_unfinished: bool) -> bool:
        """
        全局状态同步 —— 通过 all-reduce 检查所有 DP rank 是否有未完成请求。

        流程:
        ① 每 32 步执行一次 all-reduce (优化: 减少通信频率)
        ② sync_dp_state(): 2 元素张量的 all-reduce
           - 元素 0: has_unfinished (是否有未完成请求)
           - 元素 1: pending_pause (是否请求暂停)
        ③ 如果所有 rank 都同意暂停 → 设置 ignore_start_dp_wave

        返回:
          True: 至少一个 DP rank 有未完成请求 → 引擎继续运行
          False: 所有 DP rank 都空闲 → 引擎暂停
        """
        # 优化: 每 32 步才执行一次 all-reduce
        self.step_counter += 1
        if self.step_counter % 32 != 0:
            return True

        # all-reduce 同步全局状态
        has_unfinished, pause_consensus = ParallelConfig.sync_dp_state(
            self.dp_group,
            has_unfinished=local_unfinished,
            pending_pause=self.pending_pause,
        )

        # 如果所有 rank 都同意暂停
        if pause_consensus:
            self.ignore_start_dp_wave = True
            self.pending_pause = False
            logger.debug("DP pause consensus reached, ignoring START_DP_WAVE.")

        return has_unfinished

    def reinitialize_distributed(
        self, reconfig_request: ReconfigureDistributedRequest
    ) -> None:
        from copy import deepcopy

        from vllm.distributed.elastic_ep.elastic_state import ElasticEPScalingState

        new_parallel_config = deepcopy(self.vllm_config.parallel_config)
        old_dp_size = new_parallel_config.data_parallel_size
        new_parallel_config.data_parallel_size = reconfig_request.new_data_parallel_size
        if (
            reconfig_request.new_data_parallel_rank
            != ReconfigureRankType.KEEP_CURRENT_RANK
        ):
            new_parallel_config.data_parallel_rank = (
                reconfig_request.new_data_parallel_rank
            )
        new_parallel_config.data_parallel_master_ip = (
            reconfig_request.new_data_parallel_master_ip
        )
        new_parallel_config.data_parallel_master_port = (
            reconfig_request.new_data_parallel_master_port
        )
        new_parallel_config._data_parallel_master_port_list = (
            reconfig_request.new_data_parallel_master_port_list
        )
        new_parallel_config._coord_store_port = reconfig_request.coord_store_port

        is_scale_down = reconfig_request.new_data_parallel_size < old_dp_size
        is_shutdown = (
            reconfig_request.new_data_parallel_rank
            == ReconfigureRankType.SHUTDOWN_CURRENT_RANK
        )

        self.eep_scaling_state = ElasticEPScalingState(
            model_executor=self.model_executor,
            engine_core=self,
            vllm_config=self.vllm_config,
            new_parallel_config=new_parallel_config,
            worker_type="removing" if is_shutdown else "existing",
            scale_type="scale_down" if is_scale_down else "scale_up",
            reconfig_request=reconfig_request,
        )
        self.process_input_queue_block = False
        logger.info(
            "[Elastic EP] Received reconfiguration request and starting scaling up/down"
        )

    def _eep_send_engine_core_notification(
        self,
        notification_type: EEPNotificationType,
        vllm_config: VllmConfig | None = None,
    ):
        """
        Send notifications to EngineCoreClient, which can then forward
        the notifications to other engine core processes. It is used for:
        1) In scale up: new core engines to notify existing core engines
           that they are ready;
        2) In scale down: removing core engines to notify EngineCoreClient
           so EngineCoreClient can release their ray placement groups;
        3) Both scale up/down: to notify EngineCoreClient that existing
           core engines have already switched to the new parallel setup.
        """
        if vllm_config is None:
            dp_rank = self.vllm_config.parallel_config.data_parallel_rank
        else:
            dp_rank = vllm_config.parallel_config.data_parallel_rank
        notification_data = (notification_type.value, dp_rank)
        outputs = EngineCoreOutputs(
            utility_output=UtilityOutput(
                call_id=EEP_NOTIFICATION_CALL_ID,
                result=UtilityResult(notification_data),
            )
        )
        outputs.engine_index = self.engine_index

        if hasattr(self, "output_thread") and self.output_thread.is_alive():
            self.output_queue.put_nowait((0, outputs))
        else:
            encoder = MsgpackEncoder()
            with (
                zmq.Context() as ctx,
                make_zmq_socket(
                    ctx, self.addresses.outputs[0], zmq.PUSH, linger=4000
                ) as socket,
            ):
                socket.send_multipart(encoder.encode(outputs))

    def eep_handle_engine_core_notification(
        self, notification_type: str | EEPNotificationType
    ):
        """
        Handle notification received from EngineCoreClient
        (forwarded from new core engines).
        """
        assert self.eep_scaling_state is not None
        if isinstance(notification_type, str):
            notification_type = EEPNotificationType(notification_type)
        self.eep_scaling_state.handle_notification(notification_type)

    def _eep_scale_up_before_kv_init(self):
        from vllm.distributed.elastic_ep.elastic_state import ElasticEPScalingState

        self.eep_scaling_state = ElasticEPScalingState(
            model_executor=self.model_executor,
            engine_core=self,
            vllm_config=self.vllm_config,
            new_parallel_config=self.vllm_config.parallel_config,
            worker_type="new",
            scale_type="scale_up",
            reconfig_request=None,
        )
        self.eep_scaling_state.run_pre_kv_init_states()
        self.process_input_queue_block = False


class EngineCoreActorMixin:
    """
    Ray Actor 混入类 —— 用于在 Ray 集群中运行 EngineCore。

    与 EngineCoreProc 的区别:
    - EngineCoreProc: 使用 multiprocessing.Process，通过 ZMQ 通信
    - EngineCoreActorMixin: 使用 Ray Actor，通过 Ray 对象存储通信

    优势:
    - 自动故障恢复
    - 跨节点调度
    - 与 Ray 生态集成
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        addresses: EngineZmqAddresses,
        dp_rank: int = 0,
        local_dp_rank: int = 0,
    ):
        # Initialize tracer for distributed tracing if configured.
        maybe_init_worker_tracer(
            instrumenting_module_name="vllm.engine_core",
            process_kind="engine_core",
            process_name=f"DPEngineCoreActor_DP{dp_rank}",
        )

        self.addresses = addresses
        vllm_config.parallel_config.data_parallel_index = dp_rank
        vllm_config.parallel_config.data_parallel_rank_local = local_dp_rank

        self._set_nixl_side_channel_host()

        # Set CUDA_VISIBLE_DEVICES as early as possible in actor life cycle
        # NOTE: in MP we set CUDA_VISIBLE_DEVICES at process creation time,
        # and this cannot be done in the same way for Ray because:
        # 1) Ray manages life cycle of all ray workers (including
        # DPEngineCoreActor)
        # 2) Ray sets CUDA_VISIBLE_DEVICES based on num_gpus configuration
        # To bypass 2, we need to also set
        # RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES, but vLLM workers created
        # thereafter would have CUDA_VISIBLE_DEVICES set, which is sticky:
        # https://github.com/ray-project/ray/blob/e752fc319ddedd9779a0989b6d3613909bad75c9/python/ray/_private/worker.py#L456 # noqa: E501
        # This is problematic because when the vLLM worker (a Ray actor)
        # executes a task, it indexes into the sticky CUDA_VISIBLE_DEVICES
        # rather than directly using the GPU ID, potentially resulting in
        # index out of bounds error. See:
        # https://github.com/ray-project/ray/pull/40461/files#diff-31e8159767361e4bc259b6d9883d9c0d5e5db780fcea4a52ead4ee3ee4a59a78R1860 # noqa: E501
        # and get_accelerator_ids_for_accelerator_resource() in worker.py
        # of ray.
        self._set_visible_devices(vllm_config, local_dp_rank)

    @staticmethod
    def _set_nixl_side_channel_host():
        import ray

        # The driver-side value is excluded from Ray actor env propagation.
        # Fill in an actor-local default while preserving explicit overrides.
        os.environ.setdefault(
            "VLLM_NIXL_SIDE_CHANNEL_HOST", ray.util.get_node_ip_address()
        )

    def _set_visible_devices(self, vllm_config: VllmConfig, local_dp_rank: int):
        from vllm.platforms import current_platform

        if current_platform.is_xpu():
            pass
        else:
            device_control_env_var = current_platform.device_control_env_var
            self._set_cuda_visible_devices(
                vllm_config, local_dp_rank, device_control_env_var
            )

    def _set_cuda_visible_devices(
        self, vllm_config: VllmConfig, local_dp_rank: int, device_control_env_var: str
    ):
        world_size = vllm_config.parallel_config.world_size
        # Set CUDA_VISIBLE_DEVICES or equivalent.
        try:
            value = get_device_indices(
                device_control_env_var, local_dp_rank, world_size
            )
            os.environ[device_control_env_var] = value
        except IndexError as e:
            raise Exception(
                f"Error setting {device_control_env_var}: "
                f"local range: [{local_dp_rank * world_size}, "
                f"{(local_dp_rank + 1) * world_size}) "
                f'base value: "{os.getenv(device_control_env_var)}"'
            ) from e

    @contextmanager
    def _perform_handshakes(
        self,
        handshake_address: str,
        identity: bytes,
        local_client: bool,
        vllm_config: VllmConfig,
        client_handshake_address: str | None,
    ):
        """
        For Ray, we don't need to actually perform handshake.
        All addresses information is known before the actor creation.
        Therefore, we simply yield these addresses.
        """
        yield self.addresses

    def wait_for_init(self):
        """
        Wait until the engine core is initialized.

        This is just an empty method. When ray.get() on this method
        (or any other method of the actor) returns, it is guaranteed
        that actor creation (i.e., __init__) is complete.
        """
        pass

    def run(self):
        """
        Run the engine core busy loop.
        """
        try:
            self.run_busy_loop()  # type: ignore[attr-defined]
        except SystemExit:
            logger.debug("EngineCore exiting.")
            raise
        except Exception:
            logger.exception("EngineCore encountered a fatal error.")
            raise
        finally:
            self.shutdown()  # type: ignore[attr-defined]


class DPMoEEngineCoreActor(EngineCoreActorMixin, DPEngineCoreProc):
    """
    MoE 模型的数据并行 Ray Actor。

    继承:
    - EngineCoreActorMixin: Ray Actor 生命周期管理
    - DPEngineCoreProc: 数据并行引擎核心 (dummy batch, wave 同步)

    使用场景: DeepSeek V4 等 MoE 模型的 DP 部署
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_client: bool,
        addresses: EngineZmqAddresses,
        executor_class: type[Executor],
        log_stats: bool,
        dp_rank: int = 0,
        local_dp_rank: int = 0,
    ):
        vllm_config.parallel_config.data_parallel_rank = dp_rank

        EngineCoreActorMixin.__init__(
            self, vllm_config, addresses, dp_rank, local_dp_rank
        )
        DPEngineCoreProc.__init__(
            self, vllm_config, local_client, "", executor_class, log_stats
        )


class EngineCoreActor(EngineCoreActorMixin, EngineCoreProc):
    """
    非 MoE / 非 DP 场景的 Ray Actor。

    与 DPMoEEngineCoreActor 的区别:
    - 不需要 dummy batch (没有 MoE all-to-all)
    - 不需要 wave 同步 (DP=1)
    - 每个 Actor 独立运行
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_client: bool,
        addresses: EngineZmqAddresses,
        executor_class: type[Executor],
        log_stats: bool,
        dp_rank: int = 0,
        local_dp_rank: int = 0,
    ):
        vllm_config.parallel_config.data_parallel_size = 1
        vllm_config.parallel_config.data_parallel_size_local = 1
        vllm_config.parallel_config.data_parallel_rank = 0

        EngineCoreActorMixin.__init__(
            self, vllm_config, addresses, dp_rank, local_dp_rank
        )
        EngineCoreProc.__init__(
            self,
            vllm_config,
            local_client,
            "",
            executor_class,
            log_stats,
            engine_index=dp_rank,
        )
