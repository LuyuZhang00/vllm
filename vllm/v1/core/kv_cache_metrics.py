# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""KV 缓存指标跟踪模块 (vllm/v1/core/kv_cache_metrics.py)

本模块实现了 KV 缓存的生命周期指标收集功能。

指标收集的目的：
- 了解缓存块的使用模式（访问频率、重用间隔、生命周期等）
- 为缓存策略优化提供数据支持
- 监控缓存效率和淘汰质量

收集的指标：
1. 生命周期 (lifetime): 缓存块从分配到淘汰的总时间
2. 空闲时间 (idle_time): 缓存块最后一次访问到淘汰的时间
3. 重用间隔 (reuse_gaps): 连续访问之间的时间间隔序列

采样策略：
- 使用随机采样（默认 1% 的缓存块）避免全量监控的开销
- 采样率可配置，在精度和开销之间平衡
- 被采样的块会跟踪其完整的生命周期事件

数据流：
1. on_block_allocated: 块分配时，按采样率决定是否跟踪
2. on_block_accessed: 块被访问时，记录访问时间
3. on_block_evicted: 块被淘汰时，收集指标并生成事件
4. drain_events: 消费者获取并清空事件队列
"""

import random
import time
from collections import deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_utils import KVCacheBlock

from vllm.v1.metrics.stats import KVCacheEvictionEvent


class BlockMetricsState:
    """跟踪单个 KV 缓存块的生命周期指标。

    使用单调时钟 (time.monotonic_ns) 确保时间测量不受系统时间调整影响。

    属性：
        birth_time_ns: 块分配时的时间戳（纳秒）
        last_access_ns: 最后一次访问的时间戳（纳秒）
        access_history: 最近的访问时间戳队列（最大长度 4，防止无限增长）
    """

    def __init__(self):
        now_ns = time.monotonic_ns()
        self.birth_time_ns = now_ns
        self.last_access_ns = now_ns
        # 有界队列，防止块被多次访问时无限增长
        self.access_history: deque[int] = deque(maxlen=4)

    def record_access(self) -> None:
        """记录一次访问事件。"""
        now_ns = time.monotonic_ns()
        self.last_access_ns = now_ns
        self.access_history.append(now_ns)

    def get_lifetime_seconds(self) -> float:
        """获取块的总生命周期（秒），从分配到当前时间。"""
        now_ns = time.monotonic_ns()
        return (now_ns - self.birth_time_ns) / 1e9

    def get_idle_time_seconds(self) -> float:
        """获取块的空闲时间（秒），从最后一次访问到当前时间。"""
        now_ns = time.monotonic_ns()
        return (now_ns - self.last_access_ns) / 1e9

    def get_reuse_gaps_seconds(self) -> list[float]:
        """获取连续访问之间的时间间隔列表（秒）。

        Returns:
            时间间隔列表。如果访问次数不足 2 次，返回空列表。
        """
        if len(self.access_history) < 2:
            return []
        history = list(self.access_history)
        return [(history[i] - history[i - 1]) / 1e9 for i in range(1, len(history))]


class KVCacheMetricsCollector:
    """收集 KV 缓存驻留指标，支持随机采样。

    设计为与调度器集成，在缓存块的分配、访问和淘汰事件上回调。

    Args:
        sample_rate: 采样率，范围 (0, 1.0]。1.0 表示全量采样。
    """

    def __init__(self, sample_rate: float = 0.01):
        assert 0 < sample_rate <= 1.0, (
            f"sample_rate must be in (0, 1.0], got {sample_rate}"
        )
        self.sample_rate = sample_rate

        # 被采样块的指标状态映射
        self.block_metrics: dict[int, BlockMetricsState] = {}

        # 淘汰事件队列
        self._eviction_events: list[KVCacheEvictionEvent] = []

    def should_sample_block(self) -> bool:
        """按采样率决定是否采样一个块。"""
        return random.random() < self.sample_rate

    def on_block_allocated(self, block: "KVCacheBlock") -> None:
        """块分配时的回调。按采样率决定是否开始跟踪此块。"""
        if self.should_sample_block():
            self.block_metrics[block.block_id] = BlockMetricsState()

    def on_block_accessed(self, block: "KVCacheBlock") -> None:
        """块被访问时的回调。如果该块正在被跟踪，记录访问。"""
        metrics = self.block_metrics.get(block.block_id)
        if metrics:
            metrics.record_access()

    def on_block_evicted(self, block: "KVCacheBlock") -> None:
        """块被淘汰时的回调。收集该块的指标并生成淘汰事件。"""
        metrics = self.block_metrics.pop(block.block_id, None)
        if not metrics:
            return

        lifetime = metrics.get_lifetime_seconds()
        idle_time = metrics.get_idle_time_seconds()
        reuse_gaps = tuple(metrics.get_reuse_gaps_seconds())

        self._eviction_events.append(
            KVCacheEvictionEvent(
                lifetime_seconds=lifetime,
                idle_seconds=idle_time,
                reuse_gaps_seconds=reuse_gaps,
            )
        )

    def reset(self) -> None:
        """缓存重置时清除所有状态。"""
        self.block_metrics.clear()
        self._eviction_events.clear()

    def drain_events(self) -> list[KVCacheEvictionEvent]:
        """获取并清空所有淘汰事件。

        Returns:
            淘汰事件列表
        """
        events = self._eviction_events
        self._eviction_events = []
        return events
