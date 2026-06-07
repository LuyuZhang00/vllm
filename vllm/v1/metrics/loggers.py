# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
# ============================================================================
# 模块概述 (Module Overview)
# ============================================================================
# 本模块实现了 vLLM v1 引擎的指标日志记录系统。
# 核心职责：
#   1. 收集引擎运行时的各类指标（吞吐量、延迟、缓存命中率等）
#   2. 将指标输出到多种后端（标准输出日志、Prometheus 监控系统）
#   3. 支持多引擎（数据并行 DP）场景下的指标聚合
#   4. 提供插件机制，允许用户自定义日志记录器
#
# 主要类及其作用：
#   - StatLoggerBase: 所有日志记录器的抽象基类，定义统一接口
#   - LoggingStatLogger: 将指标输出到标准输出（控制台日志）的记录器
#   - AggregatedLoggingStatLogger: 聚合多个 DP 引擎的指标后输出到标准输出
#   - PerEngineStatLoggerAdapter: 为每个引擎维护独立的 PerEngine 记录器
#   - PrometheusStatLogger: 将指标输出到 Prometheus 监控系统的记录器
#   - StatLoggerManager: 管理所有日志记录器的统一入口，供 AsyncLLM 调用
#
# 数据流：
#   AsyncLLM -> StatLoggerManager.record() -> 各个 StatLogger.record()
#   定时触发 -> StatLoggerManager.log() -> 各个 StatLogger.log()
# ============================================================================
"""

import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Callable

from prometheus_client import Counter, Gauge, Histogram

import vllm.envs as envs
from vllm.compilation.cuda_graph import CUDAGraphLogging
from vllm.config import SupportsMetricsInfo, VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    KVConnectorLogging,
    KVConnectorProm,
)
from vllm.logger import init_logger
from vllm.plugins import STAT_LOGGER_PLUGINS_GROUP, load_plugins_by_group
from vllm.v1.engine import FinishReason
from vllm.v1.metrics.perf import PerfMetricsLogging, PerfMetricsProm
from vllm.v1.metrics.prometheus import unregister_vllm_metrics
from vllm.v1.metrics.stats import (
    CachingMetrics,
    IterationStats,
    MultiModalCacheStats,
    PromptTokenStats,
    SchedulerStats,
)
from vllm.v1.metrics.utils import create_metric_per_engine
from vllm.v1.spec_decode.metrics import SpecDecodingLogging, SpecDecodingProm

logger = init_logger(__name__)

# User-facing reason labels for waiting request breakdown
# 等待请求的分类标签：capacity 表示因容量不足而等待，
# deferred 表示因临时约束（如 LoRA 预算、KV 传输等）而被延迟
WAITING_REASON_CAPACITY = "capacity"
WAITING_REASON_DEFERRED = "deferred"

# 类型别名：PerEngineStatLoggerFactory 接受 VllmConfig 和引擎索引，
# 返回一个 StatLoggerBase 实例（用于为每个引擎创建独立的记录器）
PerEngineStatLoggerFactory = Callable[[VllmConfig, int], "StatLoggerBase"]

# 类型别名：AggregateStatLoggerFactory 是 AggregateStatLoggerBase 的类型
# （用于创建跨多引擎聚合的记录器）
AggregateStatLoggerFactory = type["AggregateStatLoggerBase"]

# StatLoggerFactory 可以是聚合记录器工厂或单引擎记录器工厂
StatLoggerFactory = AggregateStatLoggerFactory | PerEngineStatLoggerFactory


class StatLoggerBase(ABC):
    """Interface for logging metrics.

    API users may define custom loggers that implement this interface.
    However, note that the `SchedulerStats` and `IterationStats` classes
    are not considered stable interfaces and may change in future versions.
    """

    # ========================================================================
    # StatLoggerBase - 所有指标日志记录器的抽象基类
    # ========================================================================
    # 作用：定义统一的指标记录接口。
    # 用户可以继承此类实现自定义日志记录器。
    #
    # 三个核心抽象方法：
    #   1. __init__: 初始化记录器，接收引擎配置和引擎索引
    #   2. record: 记录一次迭代的统计数据（调度器统计 + 迭代统计）
    #   3. log_engine_initialized: 引擎初始化完成时的日志
    #
    # 注意：SchedulerStats 和 IterationStats 的接口可能会在未来版本中变更。
    # ========================================================================

    @abstractmethod
    def __init__(self, vllm_config: VllmConfig, engine_index: int = 0): ...

    @abstractmethod
    def record(
        self,
        scheduler_stats: SchedulerStats | None,
        iteration_stats: IterationStats | None,
        mm_cache_stats: MultiModalCacheStats | None = None,
        engine_idx: int = 0,
    ): ...

    @abstractmethod
    def log_engine_initialized(self): ...

    def log(self):  # noqa
        pass

    def record_sleep_state(self, is_awake: int, level: int):  # noqa
        pass


def load_stat_logger_plugin_factories() -> list[StatLoggerFactory]:
    # ========================================================================
    # 从插件系统加载自定义的 StatLogger 工厂类
    # ========================================================================
    # 流程：
    #   1. 通过 load_plugins_by_group 加载属于 STAT_LOGGER_PLUGINS_GROUP
    #      分组的所有插件
    #   2. 校验每个插件必须是 StatLoggerBase 的子类
    #   3. 将合法的插件类作为工厂加入列表并返回
    # ========================================================================
    factories: list[StatLoggerFactory] = []

    for name, plugin_class in load_plugins_by_group(STAT_LOGGER_PLUGINS_GROUP).items():
        if not isinstance(plugin_class, type) or not issubclass(
            plugin_class, StatLoggerBase
        ):
            raise TypeError(
                f"Stat logger plugin {name!r} must be a subclass of "
                f"StatLoggerBase (got {plugin_class!r})."
            )

        factories.append(plugin_class)

    return factories


class AggregateStatLoggerBase(StatLoggerBase):
    """Abstract base class for loggers that
    aggregate across multiple DP engines."""

    # ========================================================================
    # AggregateStatLoggerBase - 聚合型日志记录器的基类
    # ========================================================================
    # 作用：为支持数据并行（DP）场景下多引擎指标聚合提供基类。
    # 与 StatLoggerBase 的区别：构造函数接收 engine_indexes 列表
    # 而非单个 engine_index，用于同时管理多个引擎的指标。
    # ========================================================================

    @abstractmethod
    def __init__(self, vllm_config: VllmConfig, engine_indexes: list[int]): ...


class LoggingStatLogger(StatLoggerBase):
    # ========================================================================
    # LoggingStatLogger - 基于标准输出的日志记录器
    # ========================================================================
    # 作用：将引擎运行指标以人类可读的格式输出到控制台日志。
    #
    # 记录的核心指标包括：
    #   1. 平均 prompt 吞吐量（tokens/s）
    #   2. 平均 generation 吞吐量（tokens/s）
    #   3. 正在运行的请求数（Running）
    #   4. 等待中的请求数（Waiting）
    #   5. 被延迟的请求数（Deferred，可选）
    #   6. 抢占次数（Preemptions，可选）
    #   7. GPU KV 缓存使用率（%）
    #   8. 前缀缓存命中率（%）
    #   9. 外部前缀缓存命中率（可选）
    #  10. 多模态缓存命中率（可选）
    #
    # 日志级别策略：
    #   - 引擎空闲时使用 DEBUG 级别，避免生产环境日志噪音
    #   - 引擎活跃时使用 INFO 级别
    # ========================================================================

    def __init__(self, vllm_config: VllmConfig, engine_index: int = 0):
        self.engine_index = engine_index
        self.vllm_config = vllm_config
        # 初始化计时器和统计计数器
        self._reset(time.monotonic())

        self.last_scheduler_stats = SchedulerStats()

        # 缓存相关指标（跨日志间隔累积，不可重置）
        # TODO: 使日志间隔可配置
        self.prefix_caching_metrics = CachingMetrics()       # 本地前缀缓存
        self.connector_prefix_caching_metrics = CachingMetrics()  # 外部连接器前缀缓存
        self.mm_caching_metrics = CachingMetrics()            # 多模态缓存

        # 投机解码日志记录器
        self.spec_decoding_logging = SpecDecodingLogging()
        # KV 传输连接器日志记录器
        kv_transfer_config = self.vllm_config.kv_transfer_config
        self.kv_connector_logging = KVConnectorLogging(kv_transfer_config)
        # CUDA Graph 日志记录器（仅在配置启用时创建）
        self.cudagraph_logging = None
        if self.vllm_config.observability_config.cudagraph_metrics:
            self.cudagraph_logging = CUDAGraphLogging(
                self.vllm_config.compilation_config.cudagraph_mode,
                self.vllm_config.compilation_config.cudagraph_capture_sizes,
            )
        # 用于判断引擎是否空闲的上下文变量
        self.last_prompt_throughput: float = 0.0
        self.last_generation_throughput: float = 0.0
        self.engine_is_idle = False
        # 标记当前记录器是否为聚合模式（AggregatedLoggingStatLogger 会设为 True）
        self.aggregated = False

        # 性能指标（如 MFU 等），仅在配置启用时创建
        if self._enable_perf_stats():
            self.perf_metrics_logging = PerfMetricsLogging(vllm_config)

    def _reset(self, now):
        # ====================================================================
        # 重置当前日志间隔内的计数器
        # 每次调用 log() 输出后会重置，开始统计下一个间隔的数据
        # ====================================================================
        self.last_log_time = now

        # Tracked stats over current local logging interval.
        # 当前日志间隔内累计的各类 token / 事件计数
        self.num_prompt_tokens: int = 0          # prompt token 总数
        self.num_generation_tokens: int = 0      # generation token 总数
        self.num_corrupted_reqs: int = 0         # 含 NaN 的损坏请求数
        self.num_preemptions: int = 0            # 抢占次数

    def _enable_perf_stats(self) -> bool:
        # 是否启用 MFU（Model FLOPS Utilization）等性能指标
        return self.vllm_config.observability_config.enable_mfu_metrics

    def _track_iteration_stats(self, iteration_stats: IterationStats):
        # ====================================================================
        # 累加一次迭代的统计数据到当前间隔计数器中
        # 注意：prompt 使用 computed tokens（排除了缓存/传输命中的 token），
        # 这样吞吐量才能反映实际的计算量。
        # ====================================================================
        # Save tracked stats for token counters.
        # Use computed tokens for prompt throughput (excludes cached/transferred)
        self.num_prompt_tokens += iteration_stats.prompt_token_stats.computed
        self.num_generation_tokens += iteration_stats.num_generation_tokens
        self.num_corrupted_reqs += iteration_stats.num_corrupted_reqs
        self.num_preemptions += iteration_stats.num_preempted_reqs

    def _get_throughput(self, tracked_stats: int, now: float) -> float:
        # ====================================================================
        # 计算吞吐量：累计 token 数 / 经过的时间
        # 返回 tokens/s，若时间间隔 <= 0 则返回 0
        # ====================================================================
        # Compute summary metrics for tracked stats
        delta_time = now - self.last_log_time
        if delta_time <= 0.0:
            return 0.0
        return float(tracked_stats / delta_time)

    @property
    def log_prefix(self):
        # 日志前缀，格式如 "Engine 000: "
        return "Engine {:03d}: ".format(self.engine_index)

    def record(
        self,
        scheduler_stats: SchedulerStats | None,
        iteration_stats: IterationStats | None,
        mm_cache_stats: MultiModalCacheStats | None = None,
        engine_idx: int = 0,
    ):
        """Log Stats to standard output."""
        # ====================================================================
        # record() - 核心方法，记录一次迭代的统计数据
        # ====================================================================
        # 流程：
        #   1. 如果有迭代统计，累加 token 计数等信息
        #   2. 如果有调度器统计，更新各类缓存指标
        #   3. 更新投机解码、KV 连接器、CUDA Graph 等子系统的指标
        #   4. 如果非聚合模式，保存最新的调度器统计快照
        #   5. 如果有多模态缓存统计，更新多模态缓存指标
        # ====================================================================
        if iteration_stats:
            self._track_iteration_stats(iteration_stats)

        if scheduler_stats is not None:
            # 观察本地前缀缓存的命中情况
            self.prefix_caching_metrics.observe(scheduler_stats.prefix_cache_stats)

            # 观察外部连接器的前缀缓存命中情况
            if scheduler_stats.connector_prefix_cache_stats is not None:
                self.connector_prefix_caching_metrics.observe(
                    scheduler_stats.connector_prefix_cache_stats
                )

            # 观察投机解码统计
            if scheduler_stats.spec_decoding_stats is not None:
                self.spec_decoding_logging.observe(scheduler_stats.spec_decoding_stats)
            # 观察 KV 连接器统计
            if kv_connector_stats := scheduler_stats.kv_connector_stats:
                self.kv_connector_logging.observe(kv_connector_stats)
            # 观察 CUDA Graph 统计
            if (
                self.cudagraph_logging is not None
                and scheduler_stats.cudagraph_stats is not None
            ):
                self.cudagraph_logging.observe(scheduler_stats.cudagraph_stats)
            # 非聚合模式下保存最新调度器统计（用于 log() 输出）
            if not self.aggregated:
                self.last_scheduler_stats = scheduler_stats
            # 观察性能统计（MFU 等）
            if (perf_stats := scheduler_stats.perf_stats) and self._enable_perf_stats():
                self.perf_metrics_logging.observe(perf_stats)
        # 观察多模态缓存统计
        if mm_cache_stats:
            self.mm_caching_metrics.observe(mm_cache_stats)

    def _update_stats(self):
        # ====================================================================
        # 更新统计：计算当前间隔的吞吐量，重置计数器，判断引擎是否空闲
        # 空闲判断逻辑：当前间隔和上一间隔的 prompt/generation 吞吐量全部为 0
        # ====================================================================
        now = time.monotonic()
        prompt_throughput = self._get_throughput(self.num_prompt_tokens, now)
        generation_throughput = self._get_throughput(self.num_generation_tokens, now)

        self._reset(now)
        self.engine_is_idle = not any(
            (
                prompt_throughput,
                generation_throughput,
                self.last_prompt_throughput,
                self.last_generation_throughput,
            )
        )
        self.last_generation_throughput = generation_throughput
        self.last_prompt_throughput = prompt_throughput

    def aggregate_scheduler_stats(self):
        # 对于单引擎记录器，此方法为空操作（noop）
        # 聚合逻辑由子类 AggregatedLoggingStatLogger 实现
        return

    def log(self):
        # ====================================================================
        # log() - 将累积的指标输出到控制台日志
        # ====================================================================
        # 流程：
        #   1. 更新统计（计算吞吐量、重置计数器）
        #   2. 执行调度器统计聚合（对于聚合记录器）
        #   3. 根据引擎是否空闲选择日志级别（空闲时用 DEBUG 减少噪音）
        #   4. 格式化并输出主要指标
        #   5. 输出各子系统的额外指标（投机解码、KV 连接器等）
        # ====================================================================
        self._update_stats()
        self.aggregate_scheduler_stats()
        # Avoid log noise on an idle production system
        # 引擎空闲时使用 debug 级别，避免干扰生产环境日志
        log_fn = logger.debug if self.engine_is_idle else logger.info
        # Format and print output.
        # 构建日志格式字符串和参数
        log_parts = [
            "Avg prompt throughput: %.1f tokens/s",
            "Avg generation throughput: %.1f tokens/s",
            "Running: %d reqs",
            "Waiting: %d reqs",
        ]
        total_waiting = (
            self.last_scheduler_stats.num_waiting_reqs
            + self.last_scheduler_stats.num_skipped_waiting_reqs
        )
        log_args: list[int | float | str] = [
            self.last_prompt_throughput,
            self.last_generation_throughput,
            self.last_scheduler_stats.num_running_reqs,
            total_waiting,
        ]

        # 如果存在被延迟的请求，追加 Deferred 指标
        if self.last_scheduler_stats.num_skipped_waiting_reqs > 0:
            log_parts.append("Deferred: %d reqs")
            log_args.append(self.last_scheduler_stats.num_skipped_waiting_reqs)

        # 如果发生了抢占，追加 Preemptions 指标
        if self.num_preemptions > 0:
            log_parts.append("Preemptions: %d")
            log_args.append(self.num_preemptions)

        log_parts.extend(
            [
                "GPU KV cache usage: %.1f%%",
                "Prefix cache hit rate: %.1f%%",
            ]
        )
        log_args.extend(
            [
                self.last_scheduler_stats.kv_cache_usage * 100,
                self.prefix_caching_metrics.hit_rate * 100,
            ]
        )

        # 以下指标仅在对应条件满足时输出
        if envs.VLLM_COMPUTE_NANS_IN_LOGITS:
            log_parts.append("Corrupted: %d reqs")
            log_args.append(self.num_corrupted_reqs)
        if not self.connector_prefix_caching_metrics.empty:
            log_parts.append("External prefix cache hit rate: %.1f%%")
            log_args.append(self.connector_prefix_caching_metrics.hit_rate * 100)
        if not self.mm_caching_metrics.empty:
            log_parts.append("MM cache hit rate: %.1f%%")
            log_args.append(self.mm_caching_metrics.hit_rate * 100)

        # 输出主指标日志
        log_fn(
            self.log_prefix + ", ".join(log_parts),
            *log_args,
        )

        # 输出各子系统的额外日志
        self.spec_decoding_logging.log(log_fn=log_fn)
        self.kv_connector_logging.log(log_fn=log_fn)
        if self.cudagraph_logging is not None:
            self.cudagraph_logging.log(log_fn=log_fn)
        if self._enable_perf_stats():
            self.perf_metrics_logging.log(log_fn=log_fn, log_prefix=self.log_prefix)

    def log_engine_initialized(self):
        # 引擎初始化完成时输出缓存配置信息
        if self.vllm_config.cache_config.num_gpu_blocks:
            logger.debug(
                "Engine %03d: vllm cache_config_info with initialization "
                "after num_gpu_blocks is: %d",
                self.engine_index,
                self.vllm_config.cache_config.num_gpu_blocks,
            )


class AggregatedLoggingStatLogger(LoggingStatLogger, AggregateStatLoggerBase):
    # ========================================================================
    # AggregatedLoggingStatLogger - 聚合型日志记录器
    # ========================================================================
    # 作用：将多个 DP 引擎的指标聚合后输出到标准输出。
    # 继承自 LoggingStatLogger 和 AggregateStatLoggerBase。
    #
    # 与 LoggingStatLogger 的关键区别：
    #   1. 维护 last_scheduler_stats_dict 字典，分别存储每个引擎的调度器统计
    #   2. aggregate_scheduler_stats() 将多个引擎的统计求和/平均
    #   3. 性能指标（per_gpu perf stats）被禁用，因为跨引擎相加会产生误导
    #   4. 记录器本身不保存调度器统计（aggregated=True 标记）
    # ========================================================================

    def __init__(
        self,
        vllm_config: VllmConfig,
        engine_indexes: list[int],
    ):
        self.engine_indexes = engine_indexes
        # 为每个引擎初始化独立的调度器统计快照
        self.last_scheduler_stats_dict: dict[int, SchedulerStats] = {
            idx: SchedulerStats() for idx in self.engine_indexes
        }
        # 使用 engine_index=-1 表示这是一个聚合记录器
        LoggingStatLogger.__init__(self, vllm_config, engine_index=-1)
        self.aggregated = True

    @property
    def log_prefix(self):
        # 日志前缀，格式如 "4 Engines Aggregated: "
        return "{} Engines Aggregated: ".format(len(self.engine_indexes))

    def _enable_perf_stats(self) -> bool:
        # 禁用性能指标：将多个引擎的 per_gpu 性能数据相加会产生误导性的数字
        # Adding per_gpu perf stats across engines can lead to misleading numbers.
        return False

    def record(
        self,
        scheduler_stats: SchedulerStats | None,
        iteration_stats: IterationStats | None,
        mm_cache_stats: MultiModalCacheStats | None = None,
        engine_idx: int = 0,
    ):
        # 校验 engine_idx 是否在预期范围内
        if engine_idx not in self.engine_indexes:
            logger.warning("Unexpected engine_idx: %d", engine_idx)
            return
        # 调用父类的 record 方法处理公共指标
        LoggingStatLogger.record(
            self,
            scheduler_stats,
            iteration_stats,
            mm_cache_stats=mm_cache_stats,
            engine_idx=engine_idx,
        )
        # 将该引擎的调度器统计保存到字典中，供聚合使用
        if scheduler_stats is not None:
            self.last_scheduler_stats_dict[engine_idx] = scheduler_stats

    def aggregate_scheduler_stats(self):
        # ====================================================================
        # 聚合所有引擎的调度器统计
        # ====================================================================
        # 聚合策略：
        #   - num_waiting_reqs: 求和（总等待请求数）
        #   - num_running_reqs: 求和（总运行请求数）
        #   - num_skipped_waiting_reqs: 求和（总延迟请求数）
        #   - kv_cache_usage: 求平均（平均 KV 缓存使用率）
        # ====================================================================
        self.last_scheduler_stats = SchedulerStats()
        for last_scheduler_stats in self.last_scheduler_stats_dict.values():
            self.last_scheduler_stats.num_waiting_reqs += (
                last_scheduler_stats.num_waiting_reqs
            )
            self.last_scheduler_stats.num_running_reqs += (
                last_scheduler_stats.num_running_reqs
            )
            self.last_scheduler_stats.num_skipped_waiting_reqs += (
                last_scheduler_stats.num_skipped_waiting_reqs
            )
            self.last_scheduler_stats.kv_cache_usage += (
                last_scheduler_stats.kv_cache_usage
            )
        # KV 缓存使用率取平均值
        self.last_scheduler_stats.kv_cache_usage /= len(self.last_scheduler_stats_dict)

    def log(self):
        # 调用父类的 log 方法输出聚合后的日志
        LoggingStatLogger.log(self)

    def log_engine_initialized(self):
        # 引擎初始化完成时输出缓存配置信息
        if self.vllm_config.cache_config.num_gpu_blocks:
            logger.info(
                "%d Engines: vllm cache_config_info with initialization "
                "after num_gpu_blocks is: %d",
                len(self.engine_indexes),
                self.vllm_config.cache_config.num_gpu_blocks,
            )


class PerEngineStatLoggerAdapter(AggregateStatLoggerBase):
    # ========================================================================
    # PerEngineStatLoggerAdapter - 单引擎记录器适配器
    # ========================================================================
    # 作用：将 PerEngine 类型的记录器（如自定义插件记录器）适配为聚合记录器接口。
    #       内部为每个引擎创建独立的记录器实例，record/log 时分发到对应引擎。
    #
    # 使用场景：当用户通过插件系统注册了自定义的 PerEngine 记录器时，
    #           StatLoggerManager 会用此适配器将其包装为聚合记录器。
    # ========================================================================

    def __init__(
        self,
        vllm_config: VllmConfig,
        engine_indexes: list[int],
        per_engine_stat_logger_factory: PerEngineStatLoggerFactory,
    ) -> None:
        # 为每个引擎创建独立的记录器实例
        self.per_engine_stat_loggers = {}
        self.engine_indexes = engine_indexes
        for engine_index in engine_indexes:
            self.per_engine_stat_loggers[engine_index] = per_engine_stat_logger_factory(
                vllm_config, engine_index
            )

    def record(
        self,
        scheduler_stats: SchedulerStats | None,
        iteration_stats: IterationStats | None,
        mm_cache_stats: MultiModalCacheStats | None = None,
        engine_idx: int = 0,
    ):
        # 将记录请求分发到对应引擎的记录器
        if engine_idx not in self.per_engine_stat_loggers:
            logger.warning("Unexpected engine_idx: %d", engine_idx)
            return
        self.per_engine_stat_loggers[engine_idx].record(
            scheduler_stats,
            iteration_stats,
            mm_cache_stats=mm_cache_stats,
            engine_idx=engine_idx,
        )

    def log(self):
        # 依次调用每个引擎记录器的 log 方法
        for per_engine_stat_logger in self.per_engine_stat_loggers.values():
            per_engine_stat_logger.log()

    def log_engine_initialized(self):
        # 依次调用每个引擎记录器的初始化日志
        for per_engine_stat_logger in self.per_engine_stat_loggers.values():
            per_engine_stat_logger.log_engine_initialized()


class PrometheusStatLogger(AggregateStatLoggerBase):
    # ========================================================================
    # PrometheusStatLogger - Prometheus 监控系统日志记录器
    # ========================================================================
    # 作用：将引擎指标以 Prometheus 格式暴露，供 Grafana 等工具采集和可视化。
    #
    # Prometheus 指标类型说明：
    #   - Gauge（仪表盘）：可增可减的瞬时值，如请求数、缓存使用率
    #   - Counter（计数器）：只增不减的累计值，如 token 总数、请求成功数
    #   - Histogram（直方图）：记录值的分布，如延迟分布、token 数分布
    #
    # 标签系统：
    #   所有指标都带有 model_name 和 engine 两个标签维度，
    #   用于区分不同模型和不同引擎实例。
    #
    # 指标分类：
    #   1. 调度器状态指标（Gauge）：运行中/等待中的请求数、KV 缓存使用率
    #   2. 缓存指标（Counter）：前缀缓存/多模态缓存的查询和命中数
    #   3. Token 计数器（Counter）：prompt tokens、generation tokens
    #   4. 请求计数器（Counter）：按完成原因分类的成功请求数
    #   5. 直方图（Histogram）：各类延迟和 token 数分布
    #   6. KV 缓存驻留指标（Histogram）：块生命周期、空闲时间等
    #   7. LoRA 指标（Gauge）：LoRA 适配器的运行状态
    #   8. 子系统指标：投机解码、KV 连接器、性能指标
    # ========================================================================

    _gauge_cls = Gauge
    _counter_cls = Counter
    _histogram_cls = Histogram
    _spec_decoding_cls = SpecDecodingProm
    _kv_connector_cls = KVConnectorProm
    _perf_metrics_cls = PerfMetricsProm

    def __init__(
        self, vllm_config: VllmConfig, engine_indexes: list[int] | None = None
    ):
        if engine_indexes is None:
            engine_indexes = [0]

        self.engine_indexes = engine_indexes

        # 取消注册之前可能残留的 vLLM 指标，避免重复注册冲突
        unregister_vllm_metrics()
        self.vllm_config = vllm_config
        # Use this flag to hide metrics that were deprecated in
        # a previous release and which will be removed future
        self.show_hidden_metrics = vllm_config.observability_config.show_hidden_metrics
        # 是否启用 KV 缓存驻留指标
        self.kv_cache_metrics_enabled = (
            vllm_config.observability_config.kv_cache_metrics
        )

        # 所有指标共用的标签维度：模型名称 + 引擎索引
        labelnames = ["model_name", "engine"]
        model_name = vllm_config.model_config.served_model_name
        max_model_len = vllm_config.model_config.max_model_len

        # 为每个引擎创建标签值列表，如 [model_name, "0"], [model_name, "1"]
        self.per_engine_labelvalues: dict[int, list[object]] = {
            idx: [model_name, str(idx)] for idx in engine_indexes
        }
        per_engine_labelvalues = self.per_engine_labelvalues

        # 初始化子系统的 Prometheus 指标记录器
        self.spec_decoding_prom = self._spec_decoding_cls(
            vllm_config.speculative_config, labelnames, per_engine_labelvalues
        )
        self.kv_connector_prom = self._kv_connector_cls(
            vllm_config, labelnames, per_engine_labelvalues
        )
        self.perf_metrics_prom = self._perf_metrics_cls(
            vllm_config, labelnames, per_engine_labelvalues
        )

        #
        # Scheduler state
        # 调度器状态指标（Gauge 类型，表示当前瞬时值）
        #

        # 正在执行模型推理的请求数
        gauge_scheduler_running = self._gauge_cls(
            name="vllm:num_requests_running",
            documentation="Number of requests in model execution batches.",
            multiprocess_mode="mostrecent",
            labelnames=labelnames,
        )
        self.gauge_scheduler_running = create_metric_per_engine(
            gauge_scheduler_running, per_engine_labelvalues
        )

        # 等待被处理的请求数
        gauge_scheduler_waiting = self._gauge_cls(
            name="vllm:num_requests_waiting",
            documentation="Number of requests waiting to be processed.",
            multiprocess_mode="mostrecent",
            labelnames=labelnames,
        )
        self.gauge_scheduler_waiting = create_metric_per_engine(
            gauge_scheduler_waiting, per_engine_labelvalues
        )

        # 按原因分类的等待请求数（capacity / deferred）
        gauge_waiting_by_reason = self._gauge_cls(
            name="vllm:num_requests_waiting_by_reason",
            documentation=(
                "Number of waiting requests by reason. "
                "Reason labels: 'capacity' = waiting for scheduling capacity; "
                "'deferred' = deferred by transient constraints "
                "(LoRA budget, KV transfer, blocked status). "
                "Sum of all reasons equals vllm:num_requests_waiting."
            ),
            multiprocess_mode="mostrecent",
            labelnames=labelnames + ["reason"],
        )
        self.gauge_waiting_by_reason: dict[str, dict[int, Gauge]] = {}
        for waiting_reason in [WAITING_REASON_CAPACITY, WAITING_REASON_DEFERRED]:
            per_engine_labelvalues_with_reason = {
                idx: labelvalues + [waiting_reason]
                for idx, labelvalues in per_engine_labelvalues.items()
            }
            self.gauge_waiting_by_reason[waiting_reason] = create_metric_per_engine(
                gauge_waiting_by_reason, per_engine_labelvalues_with_reason
            )

        # 引擎休眠状态指标
        # awake=0 表示引擎休眠，awake=1 表示引擎唤醒
        # weights_offloaded=1 表示休眠级别 1（卸载权重）
        # discard_all=1 表示休眠级别 2（丢弃所有缓存）
        gauge_engine_sleep_state = self._gauge_cls(
            name="vllm:engine_sleep_state",
            documentation=(
                "Engine sleep state; awake = 0 means engine is sleeping; "
                "awake = 1 means engine is awake; "
                "weights_offloaded = 1 means sleep level 1; "
                "discard_all = 1 means sleep level 2."
            ),
            labelnames=labelnames + ["sleep_state"],
            multiprocess_mode="mostrecent",
        )

        self.gauge_engine_sleep_state = {}
        sleep_state = ["awake", "weights_offloaded", "discard_all"]

        for s in sleep_state:
            self.gauge_engine_sleep_state[s] = {
                idx: gauge_engine_sleep_state.labels(
                    engine=idx, model_name=model_name, sleep_state=s
                )
                for idx in engine_indexes
            }

        # Setting default values
        self.record_sleep_state()

        # GPU KV 缓存使用率（0~1 之间的比例）
        gauge_kv_cache_usage = self._gauge_cls(
            name="vllm:kv_cache_usage_perc",
            documentation="KV-cache usage. 1 means 100 percent usage.",
            multiprocess_mode="mostrecent",
            labelnames=labelnames,
        )
        self.gauge_kv_cache_usage = create_metric_per_engine(
            gauge_kv_cache_usage, per_engine_labelvalues
        )

        # 损坏请求数（logits 中含 NaN），仅在启用 NaN 检测时记录
        if envs.VLLM_COMPUTE_NANS_IN_LOGITS:
            counter_corrupted_requests = self._counter_cls(
                name="vllm:corrupted_requests",
                documentation=(
                    "Corrupted requests, in terms of total number of requests "
                    "with NaNs in logits."
                ),
                labelnames=labelnames,
            )
            self.counter_corrupted_requests = create_metric_per_engine(
                counter_corrupted_requests, per_engine_labelvalues
            )

        # 前缀缓存查询 token 数（Counter，只增不减）
        counter_prefix_cache_queries = self._counter_cls(
            name="vllm:prefix_cache_queries",
            documentation=(
                "Prefix cache queries, in terms of number of queried tokens."
            ),
            labelnames=labelnames,
        )
        self.counter_prefix_cache_queries = create_metric_per_engine(
            counter_prefix_cache_queries, per_engine_labelvalues
        )

        # 前缀缓存命中 token 数（Counter，只增不减）
        counter_prefix_cache_hits = self._counter_cls(
            name="vllm:prefix_cache_hits",
            documentation=("Prefix cache hits, in terms of number of cached tokens."),
            labelnames=labelnames,
        )
        self.counter_prefix_cache_hits = create_metric_per_engine(
            counter_prefix_cache_hits, per_engine_labelvalues
        )

        #
        # External - KV connector prefix cache
        # 外部 KV 连接器前缀缓存指标（跨实例缓存共享）
        #

        counter_connector_prefix_cache_queries = self._counter_cls(
            name="vllm:external_prefix_cache_queries",
            documentation=(
                "External prefix cache queries from KV connector "
                "cross-instance cache sharing, in terms of number of queried tokens."
            ),
            labelnames=labelnames,
        )
        self.counter_connector_prefix_cache_queries = create_metric_per_engine(
            counter_connector_prefix_cache_queries, per_engine_labelvalues
        )

        counter_connector_prefix_cache_hits = self._counter_cls(
            name="vllm:external_prefix_cache_hits",
            documentation=(
                "External prefix cache hits from KV connector "
                "cross-instance cache sharing, in terms of number of cached tokens."
            ),
            labelnames=labelnames,
        )
        self.counter_connector_prefix_cache_hits = create_metric_per_engine(
            counter_connector_prefix_cache_hits, per_engine_labelvalues
        )

        #
        # Multi-modal cache
        # 多模态缓存指标（如图片、音频等多模态输入的缓存）
        #

        counter_mm_cache_queries = self._counter_cls(
            name="vllm:mm_cache_queries",
            documentation=(
                "Multi-modal cache queries, in terms of number of queried items."
            ),
            labelnames=labelnames,
        )
        self.counter_mm_cache_queries = create_metric_per_engine(
            counter_mm_cache_queries, per_engine_labelvalues
        )

        counter_mm_cache_hits = self._counter_cls(
            name="vllm:mm_cache_hits",
            documentation=(
                "Multi-modal cache hits, in terms of number of cached items."
            ),
            labelnames=labelnames,
        )
        self.counter_mm_cache_hits = create_metric_per_engine(
            counter_mm_cache_hits, per_engine_labelvalues
        )

        #
        # Counters
        # 通用计数器指标
        #

        # 累计抢占次数
        counter_num_preempted_reqs = self._counter_cls(
            name="vllm:num_preemptions",
            documentation="Cumulative number of preemption from the engine.",
            labelnames=labelnames,
        )
        self.counter_num_preempted_reqs = create_metric_per_engine(
            counter_num_preempted_reqs, per_engine_labelvalues
        )

        # 累计处理的 prompt token 数
        counter_prompt_tokens = self._counter_cls(
            name="vllm:prompt_tokens",
            documentation="Number of prefill tokens processed.",
            labelnames=labelnames,
        )
        self.counter_prompt_tokens = create_metric_per_engine(
            counter_prompt_tokens, per_engine_labelvalues
        )

        # 按来源分类的 prompt token 计数（source 标签区分 computed/cached/transferred）
        counter_prompt_tokens_by_source = self._counter_cls(
            name="vllm:prompt_tokens_by_source",
            documentation="Number of prompt tokens by source.",
            labelnames=labelnames + ["source"],
        )
        self.counter_prompt_tokens_by_source: dict[str, dict[int, Counter]] = {}
        for source in PromptTokenStats.ALL_SOURCES:
            self.counter_prompt_tokens_by_source[source] = {
                idx: counter_prompt_tokens_by_source.labels(
                    model_name, str(idx), source
                )
                for idx in engine_indexes
            }

        # 缓存命中的 prompt token 数（本地 + 外部）
        counter_prompt_tokens_cached = self._counter_cls(
            name="vllm:prompt_tokens_cached",
            documentation="Number of cached prompt tokens (local + external).",
            labelnames=labelnames,
        )
        self.counter_prompt_tokens_cached = create_metric_per_engine(
            counter_prompt_tokens_cached, per_engine_labelvalues
        )

        # 累计生成的 token 数
        counter_generation_tokens = self._counter_cls(
            name="vllm:generation_tokens",
            documentation="Number of generation tokens processed.",
            labelnames=labelnames,
        )
        self.counter_generation_tokens = create_metric_per_engine(
            counter_generation_tokens, per_engine_labelvalues
        )

        # 按完成原因（stop/length/abort 等）分类的成功请求数
        self.counter_request_success: dict[FinishReason, dict[int, Counter]] = {}
        counter_request_success_base = self._counter_cls(
            name="vllm:request_success",
            documentation="Count of successfully processed requests.",
            labelnames=labelnames + ["finished_reason"],
        )
        for reason in FinishReason:
            self.counter_request_success[reason] = {
                idx: counter_request_success_base.labels(
                    model_name, str(idx), str(reason)
                )
                for idx in engine_indexes
            }

        #
        # Histograms of counts
        # 数量分布直方图
        #

        # 每个请求的 prompt token 数分布
        histogram_num_prompt_tokens_request = self._histogram_cls(
            name="vllm:request_prompt_tokens",
            documentation="Number of prefill tokens processed.",
            buckets=build_1_2_5_buckets(max_model_len),
            labelnames=labelnames,
        )
        self.histogram_num_prompt_tokens_request = create_metric_per_engine(
            histogram_num_prompt_tokens_request, per_engine_labelvalues
        )

        # 每个请求的 generation token 数分布
        histogram_num_generation_tokens_request = self._histogram_cls(
            name="vllm:request_generation_tokens",
            documentation="Number of generation tokens processed.",
            buckets=build_1_2_5_buckets(max_model_len),
            labelnames=labelnames,
        )
        self.histogram_num_generation_tokens_request = create_metric_per_engine(
            histogram_num_generation_tokens_request, per_engine_labelvalues
        )

        # TODO: This metric might be incorrect in case of using multiple
        # api_server counts which uses prometheus mp.
        # See: https://github.com/vllm-project/vllm/pull/18053
        # 每次引擎迭代步处理的 token 数分布
        histogram_iteration_tokens = self._histogram_cls(
            name="vllm:iteration_tokens_total",
            documentation="Histogram of number of tokens per engine_step.",
            buckets=[1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384],
            labelnames=labelnames,
        )
        self.histogram_iteration_tokens = create_metric_per_engine(
            histogram_iteration_tokens, per_engine_labelvalues
        )

        # 请求的最大 generation token 数分布
        histogram_max_num_generation_tokens_request = self._histogram_cls(
            name="vllm:request_max_num_generation_tokens",
            documentation="Histogram of maximum number of requested generation tokens.",
            buckets=build_1_2_5_buckets(max_model_len),
            labelnames=labelnames,
        )
        self.histogram_max_num_generation_tokens_request = create_metric_per_engine(
            histogram_max_num_generation_tokens_request, per_engine_labelvalues
        )

        # 请求参数 n 的分布（n 表示生成多少个候选输出）
        histogram_n_request = self._histogram_cls(
            name="vllm:request_params_n",
            documentation="Histogram of the n request parameter.",
            buckets=[1, 2, 5, 10, 20],
            labelnames=labelnames,
        )
        self.histogram_n_request = create_metric_per_engine(
            histogram_n_request, per_engine_labelvalues
        )

        # 请求参数 max_tokens 的分布
        histogram_max_tokens_request = self._histogram_cls(
            name="vllm:request_params_max_tokens",
            documentation="Histogram of the max_tokens request parameter.",
            buckets=build_1_2_5_buckets(max_model_len),
            labelnames=labelnames,
        )
        self.histogram_max_tokens_request = create_metric_per_engine(
            histogram_max_tokens_request, per_engine_labelvalues
        )

        #
        # Histogram of timing intervals
        # 延迟分布直方图
        #

        # 首 token 延迟（TTFT）分布：从请求到达到首个输出 token 的时间
        histogram_time_to_first_token = self._histogram_cls(
            name="vllm:time_to_first_token_seconds",
            documentation="Histogram of time to first token in seconds.",
            buckets=[
                0.001,
                0.005,
                0.01,
                0.02,
                0.04,
                0.06,
                0.08,
                0.1,
                0.25,
                0.5,
                0.75,
                1.0,
                2.5,
                5.0,
                7.5,
                10.0,
                20.0,
                40.0,
                80.0,
                160.0,
                640.0,
                2560.0,
            ],
            labelnames=labelnames,
        )
        self.histogram_time_to_first_token = create_metric_per_engine(
            histogram_time_to_first_token, per_engine_labelvalues
        )

        # token 间延迟（ITL）分布：相邻两个输出 token 之间的时间间隔
        histogram_inter_token_latency = self._histogram_cls(
            name="vllm:inter_token_latency_seconds",
            documentation="Histogram of inter-token latency in seconds.",
            buckets=[
                0.01,
                0.025,
                0.05,
                0.075,
                0.1,
                0.15,
                0.2,
                0.3,
                0.4,
                0.5,
                0.75,
                1.0,
                2.5,
                5.0,
                7.5,
                10.0,
                20.0,
                40.0,
                80.0,
            ],
            labelnames=labelnames,
        )
        self.histogram_inter_token_latency = create_metric_per_engine(
            histogram_inter_token_latency, per_engine_labelvalues
        )

        # 每个请求的平均每 token 生成时间分布
        histogram_request_time_per_output_token = self._histogram_cls(
            name="vllm:request_time_per_output_token_seconds",
            documentation="Histogram of time_per_output_token_seconds per request.",
            buckets=[
                0.01,
                0.025,
                0.05,
                0.075,
                0.1,
                0.15,
                0.2,
                0.3,
                0.4,
                0.5,
                0.75,
                1.0,
                2.5,
                5.0,
                7.5,
                10.0,
                20.0,
                40.0,
                80.0,
            ],
            labelnames=labelnames,
        )
        self.histogram_request_time_per_output_token = create_metric_per_engine(
            histogram_request_time_per_output_token, per_engine_labelvalues
        )

        # 请求延迟桶（用于端到端延迟、队列时间等）
        request_latency_buckets = [
            0.3,
            0.5,
            0.8,
            1.0,
            1.5,
            2.0,
            2.5,
            5.0,
            10.0,
            15.0,
            20.0,
            30.0,
            40.0,
            50.0,
            60.0,
            120.0,
            240.0,
            480.0,
            960.0,
            1920.0,
            7680.0,
        ]
        # 端到端请求延迟分布
        histogram_e2e_time_request = self._histogram_cls(
            name="vllm:e2e_request_latency_seconds",
            documentation="Histogram of e2e request latency in seconds.",
            buckets=request_latency_buckets,
            labelnames=labelnames,
        )
        self.histogram_e2e_time_request = create_metric_per_engine(
            histogram_e2e_time_request, per_engine_labelvalues
        )

        # 请求在 WAITING 阶段（队列中等待）的时间分布
        histogram_queue_time_request = self._histogram_cls(
            name="vllm:request_queue_time_seconds",
            documentation="Histogram of time spent in WAITING phase for request.",
            buckets=request_latency_buckets,
            labelnames=labelnames,
        )
        self.histogram_queue_time_request = create_metric_per_engine(
            histogram_queue_time_request, per_engine_labelvalues
        )

        # 请求在 RUNNING 阶段（模型推理）的时间分布
        histogram_inference_time_request = self._histogram_cls(
            name="vllm:request_inference_time_seconds",
            documentation="Histogram of time spent in RUNNING phase for request.",
            buckets=request_latency_buckets,
            labelnames=labelnames,
        )
        self.histogram_inference_time_request = create_metric_per_engine(
            histogram_inference_time_request, per_engine_labelvalues
        )

        # 请求在 PREFILL 阶段的时间分布
        histogram_prefill_time_request = self._histogram_cls(
            name="vllm:request_prefill_time_seconds",
            documentation="Histogram of time spent in PREFILL phase for request.",
            buckets=request_latency_buckets,
            labelnames=labelnames,
        )
        self.histogram_prefill_time_request = create_metric_per_engine(
            histogram_prefill_time_request, per_engine_labelvalues
        )

        # 请求在 DECODE 阶段的时间分布
        histogram_decode_time_request = self._histogram_cls(
            name="vllm:request_decode_time_seconds",
            documentation="Histogram of time spent in DECODE phase for request.",
            buckets=request_latency_buckets,
            labelnames=labelnames,
        )
        self.histogram_decode_time_request = create_metric_per_engine(
            histogram_decode_time_request, per_engine_labelvalues
        )

        # Prefill 阶段新计算的 KV token 数分布（排除缓存命中的 token）
        histogram_prefill_kv_computed_request = self._histogram_cls(
            name="vllm:request_prefill_kv_computed_tokens",
            documentation=(
                "Histogram of new KV tokens computed during prefill "
                "(excluding cached tokens)."
            ),
            buckets=build_1_2_5_buckets(max_model_len),
            labelnames=labelnames,
        )
        self.histogram_prefill_kv_computed_request = create_metric_per_engine(
            histogram_prefill_kv_computed_request, per_engine_labelvalues
        )

        #
        # KV Cache residency metrics
        # KV 缓存驻留指标（记录 KV 缓存块的生命周期和复用模式）
        # 仅在 --kv-cache-metrics 启用时创建，通过采样方式收集
        #
        if self.kv_cache_metrics_enabled:
            kv_cache_residency_buckets = [
                0.001,
                0.002,
                0.005,
                0.01,
                0.02,
                0.05,
                0.1,
                0.2,
                0.5,
                1,
                2,
                5,
                10,
                20,
                30,
                60,
                120,
                300,
                600,
                1200,
                1800,
            ]

            # KV 缓存块从分配到被驱逐的生命周期分布
            histogram_kv_block_lifetime = self._histogram_cls(
                name="vllm:kv_block_lifetime_seconds",
                documentation=(
                    "Histogram of KV cache block lifetime from allocation to eviction. "
                    "Sampled metrics (controlled by --kv-cache-metrics-sample)."
                ),
                buckets=kv_cache_residency_buckets,
                labelnames=labelnames,
            )
            self.histogram_kv_block_lifetime = create_metric_per_engine(
                histogram_kv_block_lifetime, per_engine_labelvalues
            )

            # KV 缓存块被驱逐前的空闲时间分布
            histogram_kv_block_idle_before_evict = self._histogram_cls(
                name="vllm:kv_block_idle_before_evict_seconds",
                documentation=(
                    "Histogram of idle time before KV cache block eviction. "
                    "Sampled metrics (controlled by --kv-cache-metrics-sample)."
                ),
                buckets=kv_cache_residency_buckets,
                labelnames=labelnames,
            )
            self.histogram_kv_block_idle_before_evict = create_metric_per_engine(
                histogram_kv_block_idle_before_evict, per_engine_labelvalues
            )

            # KV 缓存块连续访问之间的时间间隔分布（环形缓冲区记录最近的访问）
            histogram_kv_block_reuse_gap = self._histogram_cls(
                name="vllm:kv_block_reuse_gap_seconds",
                documentation=(
                    "Histogram of time gaps between consecutive KV cache block "
                    "accesses. Only the most recent accesses are recorded "
                    "(ring buffer). Sampled metrics (controlled by "
                    "--kv-cache-metrics-sample)."
                ),
                buckets=kv_cache_residency_buckets,
                labelnames=labelnames,
            )
            self.histogram_kv_block_reuse_gap = create_metric_per_engine(
                histogram_kv_block_reuse_gap, per_engine_labelvalues
            )
        else:
            self.histogram_kv_block_lifetime = {}
            self.histogram_kv_block_idle_before_evict = {}
            self.histogram_kv_block_reuse_gap = {}

        #
        # LoRA metrics
        # LoRA 适配器指标
        #

        # TODO: This metric might be incorrect in case of using multiple
        # api_server counts which uses prometheus mp.
        self.gauge_lora_info: Gauge | None = None
        if vllm_config.lora_config is not None:
            if len(self.engine_indexes) > 1:
                logger.warning(
                    "vllm:lora_requests_info prometheus metrics may be "
                    "incorrect/misleading with data parallel deployments."
                )
            self.labelname_max_lora = "max_lora"
            self.labelname_waiting_lora_adapters = "waiting_lora_adapters"
            self.labelname_running_lora_adapters = "running_lora_adapters"
            self.max_lora = vllm_config.lora_config.max_loras
            # LoRA 请求信息指标（当前正在运行和等待的 LoRA 适配器名称）
            self.gauge_lora_info = self._gauge_cls(
                name="vllm:lora_requests_info",
                documentation="Running stats on lora requests.",
                multiprocess_mode="sum",
                labelnames=[
                    self.labelname_max_lora,
                    self.labelname_waiting_lora_adapters,
                    self.labelname_running_lora_adapters,
                ],
            )

    def log_metrics_info(self, type: str, config_obj: SupportsMetricsInfo):
        # ====================================================================
        # 记录配置信息类型的指标
        # ====================================================================
        # 将配置对象的属性作为标签，创建一个始终为 1 的 Gauge。
        # 因为 Prometheus multiprocessing 模式不支持 Info 类型指标，
        # 所以用 Gauge 设为 1 的方式模拟 Info 指标。
        # 目前仅用于暴露 cache_config 信息。
        # ====================================================================
        metrics_info = config_obj.metrics_info()
        metrics_info["engine"] = ""

        name, documentation = None, None
        if type == "cache_config":
            name = "vllm:cache_config_info"
            documentation = "Information of the LLMEngine CacheConfig"
        assert name is not None, f"Unknown metrics info type {type}"

        # Info type metrics are syntactic sugar for a gauge permanently set to 1
        # Since prometheus multiprocessing mode does not support Info, emulate
        # info here with a gauge.
        info_gauge = self._gauge_cls(
            name=name,
            documentation=documentation,
            multiprocess_mode="mostrecent",
            labelnames=metrics_info.keys(),
        )
        for engine_index in self.engine_indexes:
            metrics_info = config_obj.metrics_info()
            metrics_info["engine"] = str(engine_index)
            info_gauge.labels(**metrics_info).set(1)

    def record(
        self,
        scheduler_stats: SchedulerStats | None,
        iteration_stats: IterationStats | None,
        mm_cache_stats: MultiModalCacheStats | None = None,
        engine_idx: int = 0,
    ):
        """Log to prometheus."""
        # ====================================================================
        # record() - 核心方法，将统计数据记录到 Prometheus 指标中
        # ====================================================================
        # 分为三大部分：
        #   1. 调度器统计 -> 更新 Gauge（请求数、缓存使用率）和 Counter（缓存命中）
        #   2. 迭代统计 -> 更新 Counter（token 数、请求数）和 Histogram（延迟分布）
        #   3. 多模态缓存统计 -> 更新 Counter（缓存查询和命中）
        # ====================================================================
        if scheduler_stats is not None:
            # 更新调度器状态 Gauge
            self.gauge_scheduler_running[engine_idx].set(
                scheduler_stats.num_running_reqs
            )
            total_waiting = (
                scheduler_stats.num_waiting_reqs
                + scheduler_stats.num_skipped_waiting_reqs
            )
            self.gauge_scheduler_waiting[engine_idx].set(total_waiting)
            # 按原因分类的等待请求数
            self.gauge_waiting_by_reason[WAITING_REASON_CAPACITY][engine_idx].set(
                scheduler_stats.num_waiting_reqs
            )
            self.gauge_waiting_by_reason[WAITING_REASON_DEFERRED][engine_idx].set(
                scheduler_stats.num_skipped_waiting_reqs
            )
            # KV 缓存使用率
            self.gauge_kv_cache_usage[engine_idx].set(scheduler_stats.kv_cache_usage)

            # 前缀缓存查询/命中数（Counter 累加）
            self.counter_prefix_cache_queries[engine_idx].inc(
                scheduler_stats.prefix_cache_stats.queries
            )
            self.counter_prefix_cache_hits[engine_idx].inc(
                scheduler_stats.prefix_cache_stats.hits
            )

            # 外部连接器前缀缓存查询/命中数
            if scheduler_stats.connector_prefix_cache_stats is not None:
                self.counter_connector_prefix_cache_queries[engine_idx].inc(
                    scheduler_stats.connector_prefix_cache_stats.queries
                )
                self.counter_connector_prefix_cache_hits[engine_idx].inc(
                    scheduler_stats.connector_prefix_cache_stats.hits
                )

            # 投机解码指标
            if scheduler_stats.spec_decoding_stats is not None:
                self.spec_decoding_prom.observe(
                    scheduler_stats.spec_decoding_stats, engine_idx
                )

            # KV 连接器指标
            if scheduler_stats.kv_connector_stats is not None:
                self.kv_connector_prom.observe(
                    scheduler_stats.kv_connector_stats, engine_idx
                )

            # 性能指标（MFU 等）
            if scheduler_stats.perf_stats is not None:
                self.perf_metrics_prom.observe(scheduler_stats.perf_stats, engine_idx)

            # KV 缓存驻留指标（如果启用且有驱逐事件）
            if (
                self.kv_cache_metrics_enabled
                and scheduler_stats.kv_cache_eviction_events
            ):
                lifetime_hist = self.histogram_kv_block_lifetime[engine_idx]
                idle_hist = self.histogram_kv_block_idle_before_evict[engine_idx]
                reuse_hist = self.histogram_kv_block_reuse_gap[engine_idx]

                for event in scheduler_stats.kv_cache_eviction_events:
                    lifetime_hist.observe(event.lifetime_seconds)
                    idle_hist.observe(event.idle_seconds)
                    for gap in event.reuse_gaps_seconds:
                        reuse_hist.observe(gap)

            # LoRA 适配器状态指标
            if self.gauge_lora_info is not None:
                running_lora_adapters = ",".join(
                    scheduler_stats.running_lora_adapters.keys()
                )
                waiting_lora_adapters = ",".join(
                    scheduler_stats.waiting_lora_adapters.keys()
                )
                lora_info_labels = {
                    self.labelname_running_lora_adapters: running_lora_adapters,
                    self.labelname_waiting_lora_adapters: waiting_lora_adapters,
                    self.labelname_max_lora: self.max_lora,
                }
                self.gauge_lora_info.labels(**lora_info_labels).set_to_current_time()

        # 多模态缓存统计
        if mm_cache_stats is not None:
            self.counter_mm_cache_queries[engine_idx].inc(mm_cache_stats.queries)
            self.counter_mm_cache_hits[engine_idx].inc(mm_cache_stats.hits)

        # 以下是迭代统计相关的指标更新
        if iteration_stats is None:
            return
        # 损坏请求计数
        if envs.VLLM_COMPUTE_NANS_IN_LOGITS:
            self.counter_corrupted_requests[engine_idx].inc(
                iteration_stats.num_corrupted_reqs
            )
        # 抢占计数
        self.counter_num_preempted_reqs[engine_idx].inc(
            iteration_stats.num_preempted_reqs
        )
        # prompt token 总数
        self.counter_prompt_tokens[engine_idx].inc(iteration_stats.num_prompt_tokens)
        # 按来源分类的 prompt token 计数
        pts = iteration_stats.prompt_token_stats
        for source in PromptTokenStats.ALL_SOURCES:
            self.counter_prompt_tokens_by_source[source][engine_idx].inc(
                pts.get_by_source(source)
            )
        # 缓存命中的 prompt token 数
        self.counter_prompt_tokens_cached[engine_idx].inc(pts.cached_tokens)
        # generation token 总数
        self.counter_generation_tokens[engine_idx].inc(
            iteration_stats.num_generation_tokens
        )
        # 每次迭代的 token 总数（prompt computed + generation）
        self.histogram_iteration_tokens[engine_idx].observe(
            iteration_stats.prompt_token_stats.computed
            + iteration_stats.num_generation_tokens
        )

        # 记录各请求级直方图指标
        for max_gen_tokens in iteration_stats.max_num_generation_tokens_iter:
            self.histogram_max_num_generation_tokens_request[engine_idx].observe(
                max_gen_tokens
            )
        for n_param in iteration_stats.n_params_iter:
            self.histogram_n_request[engine_idx].observe(n_param)
        for ttft in iteration_stats.time_to_first_tokens_iter:
            self.histogram_time_to_first_token[engine_idx].observe(ttft)
        for itl in iteration_stats.inter_token_latencies_iter:
            self.histogram_inter_token_latency[engine_idx].observe(itl)

        # 记录已完成请求的各项指标
        for finished_request in iteration_stats.finished_requests:
            # 按完成原因分类的请求成功计数
            self.counter_request_success[finished_request.finish_reason][
                engine_idx
            ].inc()
            # 端到端延迟
            self.histogram_e2e_time_request[engine_idx].observe(
                finished_request.e2e_latency
            )
            # 队列等待时间
            self.histogram_queue_time_request[engine_idx].observe(
                finished_request.queued_time
            )
            # Prefill 时间
            self.histogram_prefill_time_request[engine_idx].observe(
                finished_request.prefill_time
            )
            # 推理时间（RUNNING 阶段）
            self.histogram_inference_time_request[engine_idx].observe(
                finished_request.inference_time
            )
            # Decode 时间
            self.histogram_decode_time_request[engine_idx].observe(
                finished_request.decode_time
            )
            # Calculate prefill KV compute (excludes cached tokens)
            # Prefill 阶段实际新计算的 KV token 数
            prefill_kv_computed = finished_request.num_prompt_tokens - max(
                finished_request.num_cached_tokens, 0
            )
            self.histogram_prefill_kv_computed_request[engine_idx].observe(
                prefill_kv_computed
            )
            # 该请求的 prompt token 数
            self.histogram_num_prompt_tokens_request[engine_idx].observe(
                finished_request.num_prompt_tokens
            )
            # 该请求的 generation token 数
            self.histogram_num_generation_tokens_request[engine_idx].observe(
                finished_request.num_generation_tokens
            )
            # 该请求的平均每 output token 生成时间
            self.histogram_request_time_per_output_token[engine_idx].observe(
                finished_request.mean_time_per_output_token
            )
            if finished_request.max_tokens_param:
                self.histogram_max_tokens_request[engine_idx].observe(
                    finished_request.max_tokens_param
                )

    def record_sleep_state(self, sleep: int = 0, level: int = 0):
        # ====================================================================
        # 记录引擎休眠/唤醒状态
        # ====================================================================
        # 参数：
        #   sleep: 0=唤醒，1=休眠
        #   level: 休眠级别，0=无特殊操作，1=卸载权重，2=丢弃所有缓存
        # 映射关系：
        #   sleep=0 -> awake=1, weights_offloaded=0, discard_all=0
        #   sleep=1, level=1 -> awake=0, weights_offloaded=1, discard_all=0
        #   sleep=1, level=2 -> awake=0, weights_offloaded=0, discard_all=1
        # ========================================================================
        awake = 1
        discard_all = 0
        weights_offloaded = 0

        if sleep == 1:
            awake = 0
            if level == 1:
                weights_offloaded = 1
            elif level == 2:
                discard_all = 1

        for engine_idx in self.engine_indexes:
            self.gauge_engine_sleep_state["discard_all"][engine_idx].set(discard_all)
            self.gauge_engine_sleep_state["weights_offloaded"][engine_idx].set(
                weights_offloaded
            )
            self.gauge_engine_sleep_state["awake"][engine_idx].set(awake)

    def log_engine_initialized(self):
        # 引擎初始化完成时记录缓存配置信息
        self.log_metrics_info("cache_config", self.vllm_config.cache_config)


def build_buckets(mantissa_lst: list[int], max_value: int) -> list[int]:
    """
    Builds a list of buckets with increasing powers of 10 multiplied by
    mantissa values until the value exceeds the specified maximum.

    """
    # ========================================================================
    # 构建直方图桶（buckets）
    # ========================================================================
    # 算法：使用 1-2-5 序列（或自定义尾数序列）生成指数递增的桶边界。
    # 例如 mantissa_lst=[1,2,5], max_value=100 生成：
    #   [1, 2, 5, 10, 20, 50, 100]
    # 这种分布可以很好地覆盖不同数量级的值。
    # ========================================================================
    exponent = 0
    buckets: list[int] = []
    while True:
        for m in mantissa_lst:
            value = m * 10**exponent
            if value <= max_value:
                buckets.append(value)
            else:
                return buckets
        exponent += 1


def build_1_2_5_buckets(max_value: int) -> list[int]:
    """
    Example:
    >>> build_1_2_5_buckets(100)
    [1, 2, 5, 10, 20, 50, 100]
    """
    # 使用标准的 1-2-5 序列构建桶
    return build_buckets([1, 2, 5], max_value)


class StatLoggerManager:
    """
    StatLoggerManager:
        Logging happens at the level of the EngineCore (per scheduler).
         * DP: >1 EngineCore per AsyncLLM - loggers for each EngineCore.
         * With Local Logger, just make N copies for N EngineCores.
         * With Prometheus, we need a single logger with N "labels"

        This class abstracts away this implementation detail from
        the AsyncLLM, allowing the AsyncLLM to just call .record()
        and .log() to a simple interface.
    """

    # ========================================================================
    # StatLoggerManager - 指标日志管理器（统一入口）
    # ========================================================================
    # 作用：管理所有日志记录器，为 AsyncLLM 提供统一的 record/log 接口。
    #
    # 设计背景：
    #   在数据并行（DP）部署中，一个 AsyncLLM 可能包含多个 EngineCore。
    #   本地日志记录器需要为每个 EngineCore 创建独立副本，
    #   而 Prometheus 记录器则使用标签（label）区分不同引擎。
    #   StatLoggerManager 屏蔽了这些实现细节。
    #
    # 初始化流程：
    #   1. 加载用户自定义的日志记录器工厂
    #   2. 添加默认的本地日志记录器（LoggingStatLogger 或聚合版本）
    #   3. 对于 PerEngine 工厂，使用 PerEngineStatLoggerAdapter 适配
    #   4. 始终添加 PrometheusStatLogger（除非用户已提供自定义的）
    #
    # 对外接口：
    #   - record(): 记录一次迭代的统计数据到所有记录器
    #   - log(): 触发所有记录器输出日志
    #   - record_sleep_state(): 记录引擎休眠状态变化
    #   - log_engine_initialized(): 引擎初始化完成时的日志
    # ========================================================================

    def __init__(
        self,
        vllm_config: VllmConfig,
        engine_idxs: list[int] | None = None,
        custom_stat_loggers: list[StatLoggerFactory] | None = None,
        enable_default_loggers: bool = True,
        aggregate_engine_logging: bool = False,
        client_count: int = 1,
    ):
        self.engine_indexes = engine_idxs if engine_idxs else [0]
        self.stat_loggers: list[AggregateStatLoggerBase] = []
        stat_logger_factories: list[StatLoggerFactory] = []
        # 1. 添加用户自定义的记录器工厂
        if custom_stat_loggers is not None:
            stat_logger_factories.extend(custom_stat_loggers)
        # 2. 添加默认的本地日志记录器（如果启用且日志级别 >= INFO）
        if enable_default_loggers and logger.isEnabledFor(logging.INFO):
            if client_count > 1:
                logger.warning(
                    "AsyncLLM created with api_server_count more than 1; "
                    "disabling stats logging to avoid incomplete stats."
                )
            else:
                # 根据是否需要聚合，选择不同的默认记录器
                default_logger_factory = (
                    AggregatedLoggingStatLogger
                    if aggregate_engine_logging
                    else LoggingStatLogger
                )
                stat_logger_factories.append(default_logger_factory)
        # 3. 遍历所有工厂，创建实际的记录器实例
        custom_prometheus_logger: bool = False
        for stat_logger_factory in stat_logger_factories:
            if isinstance(stat_logger_factory, type) and issubclass(
                stat_logger_factory, AggregateStatLoggerBase
            ):
                # 聚合型工厂：直接创建一个管理所有引擎的记录器
                global_stat_logger = stat_logger_factory(
                    vllm_config=vllm_config,
                    engine_indexes=self.engine_indexes,
                )
                if isinstance(global_stat_logger, PrometheusStatLogger):
                    custom_prometheus_logger = True
            else:
                # PerEngine 型工厂：使用适配器为每个引擎创建独立记录器
                global_stat_logger = PerEngineStatLoggerAdapter(
                    vllm_config=vllm_config,
                    engine_indexes=self.engine_indexes,
                    per_engine_stat_logger_factory=stat_logger_factory,  # type: ignore[arg-type]
                )
            self.stat_loggers.append(global_stat_logger)
        # 4. 始终添加默认的 Prometheus 记录器（除非用户已提供自定义的）
        if not custom_prometheus_logger:
            self.stat_loggers.append(
                PrometheusStatLogger(vllm_config, self.engine_indexes)
            )

    def record(
        self,
        scheduler_stats: SchedulerStats | None,
        iteration_stats: IterationStats | None,
        mm_cache_stats: MultiModalCacheStats | None = None,
        engine_idx: int | None = None,
    ):
        # 将一次迭代的统计数据分发到所有已注册的记录器
        if engine_idx is None:
            engine_idx = 0
        for stat_logger in self.stat_loggers:
            stat_logger.record(
                scheduler_stats,
                iteration_stats,
                mm_cache_stats=mm_cache_stats,
                engine_idx=engine_idx,
            )

    def record_sleep_state(self, sleep: int = 0, level: int = 0):
        # 将引擎休眠状态变化分发到所有记录器
        for logger in self.stat_loggers:
            logger.record_sleep_state(sleep, level)

    def log(self):
        # 触发所有记录器输出日志（通常由定时器定期调用）
        for logger in self.stat_loggers:
            logger.log()

    def log_engine_initialized(self):
        # 引擎初始化完成时，通知所有记录器
        for agg_logger in self.stat_loggers:
            agg_logger.log_engine_initialized()
