# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Speculative Decoding 指标收集模块

本模块负责收集和报告投机解码（Speculative Decoding）的性能指标，包括：
1. draft token 的生成数量和接受数量
2. 每个位置的 token 接受率
3. 平均接受长度（mean acceptance length）
4. 吞吐量指标（draft throughput / accepted throughput）

指标有两种输出方式：
- SpecDecodingLogging: 通过日志定期输出聚合指标
- SpecDecodingProm: 通过 Prometheus 暴露计数器指标，支持 Grafana 可视化

数据流：Scheduler 每步汇总 SpecDecodingStats -> 前端 observe() 采集 -> log()/Prometheus 输出
"""

import time
from dataclasses import dataclass, field

import numpy as np
import prometheus_client

from vllm.config import SpeculativeConfig
from vllm.logger import init_logger
from vllm.v1.metrics.utils import create_metric_per_engine

logger = init_logger(__name__)


@dataclass
class SpecDecodingStats:
    """Per-step iteration decoding stats from scheduler.

    Each scheduler step, statistics on spec decoding performance are
    aggregated across requests by the scheduler and returned to the
    frontend in EngineCoreOutputs->SchedulerStats.
    """

    # 中文注释：每轮投机解码最多猜测的 token 数量（即 speculative tokens 上限）
    num_spec_tokens: int
    # 中文注释：本轮调度中参与投机解码的 draft 轮次总数
    num_drafts: int = 0
    # 中文注释：本轮所有 draft 轮次生成的 draft token 总数
    num_draft_tokens: int = 0
    # 中文注释：本轮所有 draft 轮次中被 target model 接受的 token 总数
    num_accepted_tokens: int = 0
    # 中文注释：按位置统计的接受 token 数列表，索引 i 表示第 i 个 draft 位置被接受的次数。
    # 用于计算每个位置的接受率，帮助分析 draft model 在不同位置的预测质量。
    num_accepted_tokens_per_pos: list[int] = field(default_factory=list)

    @classmethod
    def new(cls, num_spec_tokens: int) -> "SpecDecodingStats":
        """中文注释：工厂方法，创建一个新的 SpecDecodingStats 实例。
        初始化每位置接受计数为全零列表，长度等于 num_spec_tokens。
        """
        return cls(
            num_spec_tokens=num_spec_tokens,
            num_accepted_tokens_per_pos=[0] * num_spec_tokens,
        )

    def observe_draft(self, num_draft_tokens: int, num_accepted_tokens: int):
        """中文注释：记录一次 draft 轮次的结果。
        每次 target model 验证完一组 draft token 后调用此方法。

        参数：
            num_draft_tokens: 本轮 draft 生成的 token 数量
            num_accepted_tokens: 被 target model 接受的 token 数量

        流程：
        1. 累加 draft 轮次计数
        2. 累加 draft token 总数和 accepted token 总数
        3. 更新每位置的接受计数（位置 0 到 num_accepted_tokens-1 各加 1）
        """
        self.num_drafts += 1
        self.num_draft_tokens += num_draft_tokens
        self.num_accepted_tokens += num_accepted_tokens
        assert num_accepted_tokens <= self.num_spec_tokens
        for i in range(num_accepted_tokens):
            self.num_accepted_tokens_per_pos[i] += 1


class SpecDecodingLogging:
    """Aggregate and log spec decoding metrics.

    LoggingStatLogger aggregates per-iteration metrics over a set
    time interval using observe() and then logs them using log()
    before resetting to zero.
    """

    # 中文注释：投机解码日志记录器。
    # 工作模式：多次 observe() 采集 -> 一次 log() 聚合输出 -> reset() 清零。
    # 通过时间间隔控制日志输出频率，避免每步都打印。

    def __init__(self):
        self.reset()

    def reset(self):
        """中文注释：重置所有聚合指标列表和上次日志时间戳。
        每次 log() 输出后自动调用，开始新一轮聚合周期。
        """
        # 中文注释：收集每个调度步的 draft 轮次数，后续聚合时求和
        self.num_drafts: list[int] = []
        # 中文注释：收集每个调度步的 draft token 数量
        self.num_draft_tokens: list[int] = []
        # 中文注释：收集每个调度步的 accepted token 数量
        self.num_accepted_tokens: list[int] = []
        # 中文注释：收集每个调度步的按位置接受计数列表，用于计算各位置接受率
        self.accepted_tokens_per_pos_lists: list[list[int]] = []
        # 中文注释：记录上次 log() 的时间，用于计算聚合时间窗口
        self.last_log_time = time.monotonic()

    def observe(self, spec_decoding_stats: SpecDecodingStats):
        """中文注释：采集一个调度步的投机解码统计信息。
        将当前步的指标追加到聚合列表中，等待 log() 统一输出。

        参数：
            spec_decoding_stats: 由 Scheduler 在每个调度步汇总的统计信息
        """
        self.num_drafts.append(spec_decoding_stats.num_drafts)
        self.num_draft_tokens.append(spec_decoding_stats.num_draft_tokens)
        self.num_accepted_tokens.append(spec_decoding_stats.num_accepted_tokens)
        self.accepted_tokens_per_pos_lists.append(
            spec_decoding_stats.num_accepted_tokens_per_pos
        )

    def log(self, log_fn=logger.info):
        """中文注释：聚合并输出投机解码指标日志。

        流程：
        1. 将多个调度步的指标求和得到总量
        2. 计算时间窗口内的吞吐量（tokens/s）
        3. 计算 draft 接受率 = accepted / drafted * 100%
        4. 计算平均接受长度 = 1 + (accepted / drafts)，其中 1 是 bonus token（即 target model 验证后额外生成的一个 token）
        5. 计算每个 draft 位置的接受率向量
        6. 输出日志后调用 reset() 清零，开始新的聚合周期

        参数：
            log_fn: 日志输出函数，默认为 logger.info
        """
        if not self.num_drafts:
            return
        # 中文注释：步骤 1 - 汇总所有调度步的指标
        num_drafts = np.sum(self.num_drafts)
        num_draft_tokens = np.sum(self.num_draft_tokens)
        num_accepted_tokens = np.sum(self.num_accepted_tokens)
        draft_throughput = 0
        accepted_throughput = 0

        # 中文注释：步骤 2 - 计算聚合时间窗口内的吞吐量
        elapsed_time = time.monotonic() - self.last_log_time
        if elapsed_time > 0:
            draft_throughput = num_draft_tokens / elapsed_time
            accepted_throughput = num_accepted_tokens / elapsed_time

        # 中文注释：步骤 3 - 计算整体 draft 接受率（百分比）
        draft_acceptance_rate = (
            num_accepted_tokens / num_draft_tokens * 100
            if num_draft_tokens > 0
            else float("nan")
        )

        # Conventionally, mean acceptance length includes the bonus token
        # 中文注释：步骤 4 - 计算平均接受长度，包含 bonus token。
        # 值越大说明 draft model 预测越准确，投机解码加速效果越好。
        mean_acceptance_length = 1 + (num_accepted_tokens / num_drafts)

        # 中文注释：步骤 5 - 计算每个 draft 位置的接受率。
        # 例如 [0.8, 0.6, 0.4] 表示第 0 位接受率 80%，第 1 位 60%，第 2 位 40%。
        # 通常越靠后的位置接受率越低，因为错误会累积传播。
        pos_matrix = np.array(self.accepted_tokens_per_pos_lists)
        acceptance_rates = np.sum(pos_matrix, axis=0) / num_drafts
        rates_str = ", ".join(f"{p:.3f}" for p in acceptance_rates)

        log_fn(
            "SpecDecoding metrics: "
            "Mean acceptance length: %.2f, "
            "Accepted throughput: %.2f tokens/s, "
            "Drafted throughput: %.2f tokens/s, "
            "Accepted: %d tokens, "
            "Drafted: %d tokens, "
            "Per-position acceptance rate: %s, "
            "Avg Draft acceptance rate: %.1f%%",
            mean_acceptance_length,
            accepted_throughput,
            draft_throughput,
            num_accepted_tokens,
            num_draft_tokens,
            rates_str,
            draft_acceptance_rate,
        )
        self.reset()


class SpecDecodingProm:
    """Record spec decoding metrics in Prometheus.

    The acceptance rate can be calculated using a PromQL query:

      rate(vllm:spec_decode_num_accepted_tokens_total[$interval]) /
      rate(vllm:spec_decode_num_draft_tokens_total[$interval])

    The mean acceptance length (conventionally including bonus tokens)
    can be calculated using:

      1 + (
      rate(vllm:spec_decode_num_accepted_tokens_total[$interval]) /
      rate(vllm:spec_decode_num_drafts[$interval]))

    A per-position acceptance rate vector can be computed using

      vllm:spec_decode_num_accepted_tokens_per_pos[$interval] /
      vllm:spec_decode_num_drafts[$interval]
    """

    # 中文注释：Prometheus 指标记录器。
    # 通过 Prometheus Counter 暴露投机解码的核心指标，配合 Grafana 可实时监控。
    # 支持多引擎（per-engine）标签，每个引擎有独立的计数器。

    _counter_cls = prometheus_client.Counter

    def __init__(
        self,
        speculative_config: SpeculativeConfig | None,
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ):
        """中文注释：初始化 Prometheus 计数器。

        参数：
            speculative_config: 投机解码配置，为 None 表示未启用投机解码
            labelnames: Prometheus 标签名称列表（如 engine_id 等）
            per_engine_labelvalues: 每个引擎对应的标签值字典

        流程：
        1. 检查投机解码是否启用，未启用则直接返回
        2. 创建 4 类计数器：draft 轮次数、draft token 数、accepted token 数、按位置 accepted 数
        3. 为每个引擎实例化独立的计数器（通过 create_metric_per_engine）
        """
        self.spec_decoding_enabled = speculative_config is not None
        if not self.spec_decoding_enabled:
            return

        # 中文注释：创建 draft 轮次计数器 - 统计总 draft 调用次数
        counter_drafts = self._counter_cls(
            name="vllm:spec_decode_num_drafts",
            documentation="Number of spec decoding drafts.",
            labelnames=labelnames,
        )
        self.counter_spec_decode_num_drafts = create_metric_per_engine(
            counter_drafts, per_engine_labelvalues
        )

        # 中文注释：创建 draft token 计数器 - 统计生成的 draft token 总量
        counter_draft_tokens = self._counter_cls(
            name="vllm:spec_decode_num_draft_tokens",
            documentation="Number of draft tokens.",
            labelnames=labelnames,
        )
        self.counter_spec_decode_num_draft_tokens = create_metric_per_engine(
            counter_draft_tokens, per_engine_labelvalues
        )

        # 中文注释：创建 accepted token 计数器 - 统计被接受的 token 总量
        counter_accepted_tokens = self._counter_cls(
            name="vllm:spec_decode_num_accepted_tokens",
            documentation="Number of accepted tokens.",
            labelnames=labelnames,
        )
        self.counter_spec_decode_num_accepted_tokens = create_metric_per_engine(
            counter_accepted_tokens, per_engine_labelvalues
        )

        assert speculative_config is not None
        # 中文注释：创建按位置的 accepted token 计数器。
        # 使用 "position" 标签区分不同 draft 位置，便于在 Grafana 中绘制每个位置的接受率曲线。
        num_spec_tokens = (
            speculative_config.num_speculative_tokens
            if self.spec_decoding_enabled
            else 0
        )
        pos_labelnames = labelnames + ["position"]
        base_counter = self._counter_cls(
            name="vllm:spec_decode_num_accepted_tokens_per_pos",
            documentation="Accepted tokens per draft position.",
            labelnames=pos_labelnames,
        )
        self.counter_spec_decode_num_accepted_tokens_per_pos: dict[
            int, list[prometheus_client.Counter]
        ] = {
            idx: [base_counter.labels(*lv, str(pos)) for pos in range(num_spec_tokens)]
            for idx, lv in per_engine_labelvalues.items()
        }

    def observe(self, spec_decoding_stats: SpecDecodingStats, engine_idx: int = 0):
        """中文注释：将一个调度步的统计信息记录到 Prometheus 计数器。

        参数：
            spec_decoding_stats: 调度器汇总的本步投机解码统计
            engine_idx: 引擎索引，用于选择对应引擎的计数器实例

        流程：
        1. 如果投机解码未启用，直接返回
        2. 递增各计数器：drafts、draft_tokens、accepted_tokens
        3. 逐位置递增 accepted_tokens_per_pos 计数器
        """
        if not self.spec_decoding_enabled:
            return
        self.counter_spec_decode_num_drafts[engine_idx].inc(
            spec_decoding_stats.num_drafts
        )
        self.counter_spec_decode_num_draft_tokens[engine_idx].inc(
            spec_decoding_stats.num_draft_tokens
        )
        self.counter_spec_decode_num_accepted_tokens[engine_idx].inc(
            spec_decoding_stats.num_accepted_tokens
        )
        for pos, counter in enumerate(
            self.counter_spec_decode_num_accepted_tokens_per_pos[engine_idx]
        ):
            counter.inc(spec_decoding_stats.num_accepted_tokens_per_pos[pos])
