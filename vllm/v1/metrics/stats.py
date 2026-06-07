# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# 统计数据收集模块
#
# 本模块定义了 vLLM v1 引擎运行过程中的各类统计数据结构。
# 这些数据结构在引擎的不同阶段被填充，最终由指标日志记录器
# （如 PrometheusStatLogger）读取并上报。
#
# 主要数据结构分为以下几类：
#
# 1. 缓存统计（Cache Statistics）：
#    - BaseCacheStats：缓存命中统计的基类
#    - CachingMetrics：滑动窗口缓存命中率计算
#    - PrefixCacheStats：前缀缓存（Prefix Cache）命中统计
#    - MultiModalCacheStats：多模态缓存命中统计
#
# 2. 调度器统计（Scheduler Statistics）：
#    - SchedulerStats：调度器的全局状态（队列长度、KV Cache 使用率等）
#    - KVCacheEvictionEvent：KV Cache 驱逐事件
#
# 3. 请求状态统计（Request State Statistics）：
#    - RequestStateStats：单个请求在其生命周期中需要跟踪的状态
#    - FinishedRequestStats：请求完成时的最终统计
#
# 4. 迭代统计（Iteration Statistics）：
#    - IterationStats：单次调度迭代的统计，聚合了多个请求的输出
#    - PrefillStats：预填充（Prefill）阶段的详细分解
#    - PromptTokenStats：Prompt token 按来源的分类统计
#
# 5. LoRA 统计：
#    - LoRAStats：单个 LoRA 适配器的请求跟踪
#    - LoRARequestStates：所有 LoRA 适配器的请求状态管理

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import vllm.envs as envs
from vllm.compilation.cuda_graph import CUDAGraphStat
from vllm.v1.metrics.perf import PerfStats
from vllm.v1.spec_decode.metrics import SpecDecodingStats

if TYPE_CHECKING:
    from vllm.v1.engine import EngineCoreEvent, EngineCoreOutput, FinishReason


@dataclass
class BaseCacheStats:
    """缓存命中统计的基类。

    用于记录一组请求的缓存查询和命中情况。
    前缀缓存（Prefix Cache）和多模态缓存都继承此类。

    Attributes:
        reset: 是否在本次更新前执行了缓存重置操作。
            如果为 True，需要先清空之前的统计数据。
        requests: 本次更新中的请求数量。
        queries: 本次更新中的缓存查询次数（如 token 数或多模态数据项数）。
        hits: 本次更新中的缓存命中次数。
    """

    reset: bool = False
    """Whether the cache was reset."""

    requests: int = 0
    """The number of requests in this update."""

    queries: int = 0
    """The number of queries in these requests."""

    hits: int = 0
    """The number of hits in these requests."""


class CachingMetrics:
    """缓存命中率指标，基于最近 N 个请求的滑动窗口计算。

    使用滑动窗口机制维护最近 N 个请求的缓存命中率。
    当新请求到来时，将其统计信息加入窗口；当窗口超过大小限制时，
    移除最旧的统计信息。这样可以反映近期的缓存性能。

    Args:
        interval: 滑动窗口大小，即最近多少个请求用于计算命中率。
            默认为 1000。

    工作原理：
    1. 使用 deque 维护一个 FIFO 队列，每个元素为 (requests, queries, hits) 元组
    2. 维护聚合值：aggregated_requests, aggregated_query_total, aggregated_query_hit
    3. 每次 observe 时，将新数据追加到队列并更新聚合值
    4. 当聚合请求数超过窗口大小时，从队列头部移除旧数据并减少聚合值
    5. hit_rate = aggregated_query_hit / aggregated_query_total
    """

    def __init__(self, max_recent_requests: int = 1000) -> None:
        super().__init__()

        # 滑动窗口的最大请求数
        self.max_recent_requests = max_recent_requests
        # 当前窗口内的聚合值
        self.aggregated_requests = 0
        self.aggregated_query_total = 0
        self.aggregated_query_hit = 0

        # 滑动窗口队列：存储最近的 (请求数, 查询数, 命中数) 元组
        self.query_queue = deque[tuple[int, int, int]]()

    def observe(self, stats: BaseCacheStats):
        """观察一组请求的前缀缓存命中情况。

        当新请求被调度并查找已计算的缓存块时调用此函数。

        处理逻辑：
        1. 如果 stats.reset 为 True，说明之前执行了缓存重置，
           需要先清空所有统计数据。
        2. 跳过空统计（requests == 0），避免空数据占用窗口空间
           导致有用数据被移除。
        3. 将新统计数据追加到队列并更新聚合值。
        4. 当聚合请求数超过窗口大小时，从队列头部移除旧数据。
           注意：至少保留最后一条数据，避免窗口为空。

        Args:
            stats: 前缀缓存统计数据。
        """
        # reset_prefix_cache was invoked before the current update.
        # Reset the metrics before aggregating the current stats.
        if stats.reset:
            self.reset()

        # DO NOT appending empty stats to avoid helpful info get kicked out
        # due to sliding window.
        if stats.requests == 0:
            return

        # Update the metrics.
        self.query_queue.append((stats.requests, stats.queries, stats.hits))
        self.aggregated_requests += stats.requests
        self.aggregated_query_total += stats.queries
        self.aggregated_query_hit += stats.hits

        # Remove the oldest stats until number of requests does not exceed
        # the limit.
        # NOTE: We preserve the latest added stats regardless.
        while (
            len(self.query_queue) > 1
            and self.aggregated_requests > self.max_recent_requests
        ):
            old_requests, old_queries, old_hits = self.query_queue.popleft()
            self.aggregated_requests -= old_requests
            self.aggregated_query_total -= old_queries
            self.aggregated_query_hit -= old_hits

    def reset(self):
        """重置所有缓存统计数据。

        在以下场景调用：
        1. 缓存被重置（reset_prefix_cache）后
        2. 滑动窗口观察到 stats.reset 为 True 的统计时
        """
        self.aggregated_requests = 0
        self.aggregated_query_total = 0
        self.aggregated_query_hit = 0
        self.query_queue.clear()

    @property
    def empty(self) -> bool:
        """返回是否尚未观察到任何请求。"""
        return self.aggregated_requests == 0

    @property
    def hit_rate(self) -> float:
        """计算最近 N 个请求的缓存命中率。

        Returns:
            命中率，范围 [0.0, 1.0]。如果没有查询数据则返回 0.0。
        """
        if self.aggregated_query_total == 0:
            return 0.0
        return self.aggregated_query_hit / self.aggregated_query_total


@dataclass
class PrefixCacheStats(BaseCacheStats):
    """前缀缓存（Prefix Cache）命中统计。

    继承自 BaseCacheStats，增加了对被抢占（preempted）请求的统计。
    前缀缓存通过 token 前缀匹配来复用已有的 KV Cache，
    减少重复计算。

    Attributes:
        preempted_requests: 本次更新中被抢占的请求数量。
            被抢占的请求之前已经部分执行过，恢复时需要重新查找缓存。
        preempted_queries: 被抢占请求的查询 token 数。
        preempted_hits: 被抢占请求的缓存命中 token 数。
    """

    preempted_requests: int = 0
    """The number of previously preempted requests in this update."""

    preempted_queries: int = 0
    """The `queries` number for preempted requests."""

    preempted_hits: int = 0
    """The `hits` number for preempted requests."""

    def record(self, num_tokens: int, num_hits: int, preempted: bool) -> None:
        """记录一个请求的缓存查询结果。

        根据请求是否被抢占过，分别累加到不同的统计字段。
        被抢占的请求单独统计，以便分析抢占对缓存命中率的影响。

        Args:
            num_tokens: 该请求查询的 token 数量。
            num_hits: 该请求命中的 token 数量。
            preempted: 该请求是否是被抢占后恢复的。
        """
        if preempted:
            # Previously preempted request
            self.preempted_requests += 1
            self.preempted_queries += num_tokens
            self.preempted_hits += num_hits
        else:
            # New request
            self.requests += 1
            self.queries += num_tokens
            self.hits += num_hits


@dataclass
class MultiModalCacheStats(BaseCacheStats):
    """多模态缓存命中统计。

    用于跟踪多模态数据（如图像、视频）的缓存命中情况。
    多模态数据的预处理（如图像编码）开销较大，缓存可以显著减少延迟。

    Attributes:
        reset: 是否在本次更新前执行了多模态缓存重置。
        queries: 查询的多模态数据项数量。
        hits: 命中的多模态数据项数量。
    """

    def record(self, num_queries: int, num_hits: int) -> None:
        """记录一个多模态缓存查询结果。

        Args:
            num_queries: 查询的多模态数据项数量。
            num_hits: 命中的多模态数据项数量。
        """
        self.requests += 1
        self.queries += num_queries
        self.hits += num_hits


@dataclass
class KVCacheEvictionEvent:
    """KV Cache 块驱逐事件的单次采样记录。

    当 KV Cache 空间不足时，调度器会驱逐一些缓存块来腾出空间。
    本数据结构记录被驱逐块的生命周期信息，用于分析缓存替换策略的效果。

    Attributes:
        lifetime_seconds: 被驱逐块的总生命周期（从分配到驱逐的时长）。
        idle_seconds: 被驱逐块的空闲时间（从最后一次使用到驱逐的时长）。
        reuse_gaps_seconds: 被驱逐块的各次复用间隔时间元组。
            记录了该块在生命周期中每次被复用之间的时间间隔。
    """

    lifetime_seconds: float
    idle_seconds: float
    reuse_gaps_seconds: tuple[float, ...]


@dataclass
class SchedulerStats:
    """调度器的全局统计数据。

    记录调度器在每次迭代中的状态信息，这些数据会被传递给指标日志记录器。
    调度器是 vLLM 的核心组件，负责决定哪些请求在当前迭代中执行。

    Attributes:
        num_running_reqs: 当前正在运行（执行推理）的请求数。
        num_waiting_reqs: 等待队列中的请求数。
        num_skipped_waiting_reqs: 被跳过的等待请求数（因资源不足等原因）。
        step_counter: 步骤计数器，用于内部数据并行负载均衡。
        current_wave: 当前波次，用于内部数据并行负载均衡。
        kv_cache_usage: KV Cache 的使用率，范围 [0.0, 1.0]。
        prefix_cache_stats: 前缀缓存命中统计。
        connector_prefix_cache_stats: KV 连接器的前缀缓存统计（可选）。
        kv_cache_eviction_events: KV Cache 驱逐事件列表。
        spec_decoding_stats: 推测解码统计信息（可选）。
        kv_connector_stats: KV 连接器统计信息（可选）。
        waiting_lora_adapters: 各 LoRA 适配器在等待队列中的请求数。
        running_lora_adapters: 各 LoRA 适配器在运行队列中的请求数。
        cudagraph_stats: CUDA Graph 编译统计信息（可选）。
        perf_stats: 性能统计信息（FLOPs、内存带宽等，可选）。
    """

    num_running_reqs: int = 0

    num_waiting_reqs: int = 0  # length of the "waiting" request queue
    num_skipped_waiting_reqs: int = 0  # length of the "skipped waiting" queue

    # These are used for internal DP load-balancing.
    step_counter: int = 0
    current_wave: int = 0

    kv_cache_usage: float = 0.0

    prefix_cache_stats: PrefixCacheStats = field(default_factory=PrefixCacheStats)
    connector_prefix_cache_stats: PrefixCacheStats | None = None

    kv_cache_eviction_events: list[KVCacheEvictionEvent] = field(default_factory=list)

    spec_decoding_stats: SpecDecodingStats | None = None
    kv_connector_stats: dict[str, Any] | None = None

    waiting_lora_adapters: dict[str, int] = field(default_factory=dict)
    running_lora_adapters: dict[str, int] = field(default_factory=dict)

    cudagraph_stats: CUDAGraphStat | None = None

    perf_stats: PerfStats | None = None


@dataclass
class RequestStateStats:
    """单个请求在其生命周期中需要跟踪的状态。

    这些统计数据在请求的多次迭代更新之间持续存在，
    用于计算端到端延迟、首 token 延迟、token 间延迟等指标。

    时间戳说明：
    - arrival_time: 请求到达引擎前端的墙钟时间（wall-clock time）。
      用于计算端到端延迟。
    - queued_ts: 请求进入等待队列的单调时间（monotonic time）。
    - scheduled_ts: 请求首次被调度执行的单调时间。
    - first_token_ts: 请求生成第一个 token 的单调时间。
    - last_token_ts: 请求生成最后一个 token 的单调时间。

    Attributes:
        num_generation_tokens: 该请求已生成的 token 总数。
        arrival_time: 请求到达的墙钟时间戳。
        queued_ts: 请求入队的单调时间戳。
        scheduled_ts: 请求被调度的单调时间戳（忽略抢占后的重新调度）。
        first_token_ts: 生成第一个 token 的单调时间戳。
        last_token_ts: 生成最后一个 token 的单调时间戳。
        first_token_latency: 首 token 延迟（从到达到第一个 token）。
        is_corrupted: 请求是否包含损坏的 logits（NaN 值）。
    """

    num_generation_tokens: int = 0

    # This is an engine frontend timestamp (wall-clock)
    arrival_time: float = 0.0

    # These are engine core timestamps (monotonic)
    queued_ts: float = 0.0
    scheduled_ts: float = 0.0
    first_token_ts: float = 0.0
    last_token_ts: float = 0.0

    # first token latency
    first_token_latency: float = 0.0

    # Track if this request is corrupted (NaNs in logits)
    is_corrupted: bool = False


@dataclass
class FinishedRequestStats:
    """请求完成时的最终统计数据。

    当一个请求完成（正常结束、超长、被中止等）时，
    记录其完整的生命周期指标。这些数据用于计算各种延迟统计。

    时间区间说明：
    - queued_time: 排队时间 = scheduled_ts - queued_ts
    - prefill_time: 预填充时间 = first_token_ts - scheduled_ts
      （包括因抢占导致的重试）
    - decode_time: 解码时间 = last_token_ts - first_token_ts
      （包括因抢占导致的重试）
    - inference_time: 推理总时间 = last_token_ts - scheduled_ts
      （= prefill_time + decode_time）
    - e2e_latency: 端到端延迟 = 当前时间 - arrival_time
    - mean_time_per_output_token: 平均每输出 token 耗时
      = decode_time / (num_generation_tokens - 1)

    Attributes:
        finish_reason: 请求结束原因（完成、超长、中止等）。
        request_id: 请求 ID。
        e2e_latency: 端到端延迟（秒）。
        num_prompt_tokens: 输入 prompt 的 token 数。
        num_generation_tokens: 生成的 token 数。
        max_tokens_param: 用户设置的最大生成 token 数参数。
        queued_time: 排队等待时间（秒）。
        prefill_time: 预填充阶段耗时（秒）。
        inference_time: 推理总耗时（秒）。
        decode_time: 解码阶段耗时（秒）。
        mean_time_per_output_token: 平均每输出 token 耗时（秒）。
        is_corrupted: 请求是否包含损坏数据。
        num_cached_tokens: 从缓存中命中的 token 数。
    """

    finish_reason: "FinishReason"
    request_id: str | None = None
    e2e_latency: float = 0.0
    num_prompt_tokens: int = 0
    num_generation_tokens: int = 0
    max_tokens_param: int | None = None
    queued_time: float = 0.0
    prefill_time: float = 0.0
    inference_time: float = 0.0
    decode_time: float = 0.0
    mean_time_per_output_token: float = 0.0
    is_corrupted: bool = False
    num_cached_tokens: int = 0


@dataclass
class PrefillStats:
    """预填充（Prefill）阶段计算的详细分解。

    Prefill 阶段处理输入 prompt 的所有 token，生成 KV Cache。
    本数据结构将 prefill 的 token 分为以下几类：

    1. num_prompt_tokens: prompt 的总 token 数。
    2. num_cached_tokens: 从缓存中获取的 token 数（无需实际计算）。
       等于 num_local_cached_tokens + num_external_cached_tokens。
    3. num_computed_tokens: 需要实际计算的 token 数。
       = num_prompt_tokens - num_cached_tokens
    4. num_local_cached_tokens: 从本地前缀缓存命中的 token 数。
    5. num_external_cached_tokens: 通过外部 KV 传输获取的 token 数。
       用于分布式 KV Cache 场景（如 disaggregated prefill/decode）。

    不变量：num_computed_tokens + num_cached_tokens = num_prompt_tokens

    Attributes:
        num_prompt_tokens: prompt 总 token 数。
        num_computed_tokens: 需要本地计算的 token 数。
        num_cached_tokens: 缓存命中的 token 总数。
        num_local_cached_tokens: 本地前缀缓存命中的 token 数。
        num_external_cached_tokens: 外部 KV 传输获取的 token 数。
    """

    num_prompt_tokens: int = 0
    num_computed_tokens: int = 0
    num_cached_tokens: int = 0
    num_local_cached_tokens: int = 0
    num_external_cached_tokens: int = 0

    def set(
        self,
        num_prompt_tokens: int,
        num_local_cached_tokens: int,
        num_external_cached_tokens: int,
    ):
        """设置预填充统计数据。

        Args:
            num_prompt_tokens: prompt 总 token 数。
            num_local_cached_tokens: 本地缓存命中的 token 数。
            num_external_cached_tokens: 外部 KV 传输获取的 token 数。
        """
        num_cached_tokens = num_local_cached_tokens + num_external_cached_tokens
        assert num_cached_tokens <= num_prompt_tokens

        self.num_prompt_tokens = num_prompt_tokens
        self.num_computed_tokens = num_prompt_tokens - num_cached_tokens
        self.num_cached_tokens = num_cached_tokens
        self.num_local_cached_tokens = num_local_cached_tokens
        self.num_external_cached_tokens = num_external_cached_tokens


@dataclass
class PromptTokenStats:
    """Prompt token 按来源的分类统计。

    将 prompt token 按照获取来源分为三类：
    1. local_compute: 本地计算生成的 token（实际的 Prefill 计算）
    2. local_cache_hit: 本地前缀缓存命中的 token
    3. external_kv_transfer: 通过外部 KV 传输获取的 token

    不变量：
    - computed + local_cache_hit + external_kv_transfer = total
    - local_cache_hit + external_kv_transfer = cached_tokens

    Attributes:
        computed: 本地计算的 token 数。
        local_cache_hit: 本地缓存命中的 token 数。
        external_kv_transfer: 外部 KV 传输的 token 数。
        cached_tokens: 跳过计算的 token 总数（= local_cache_hit + external_kv_transfer）。
        total: prompt token 总数。
    """

    ALL_SOURCES: tuple[str, ...] = (
        "local_compute",
        "local_cache_hit",
        "external_kv_transfer",
    )

    computed: int = 0
    local_cache_hit: int = 0
    external_kv_transfer: int = 0
    cached_tokens: int = 0
    total: int = 0

    def update_from_output(self, prefill_stats: PrefillStats) -> None:
        """从预填充输出更新统计数据。

        Args:
            prefill_stats: 预填充阶段的统计数据。
        """
        self.computed += prefill_stats.num_computed_tokens
        self.cached_tokens += prefill_stats.num_cached_tokens
        self.total += prefill_stats.num_prompt_tokens

        self.local_cache_hit += prefill_stats.num_local_cached_tokens
        self.external_kv_transfer += prefill_stats.num_external_cached_tokens

    def get_by_source(self, source: str) -> int:
        """按来源标签获取 token 计数。

        Args:
            source: 来源标签，必须是 "local_compute"、
                "local_cache_hit" 或 "external_kv_transfer" 之一。

        Returns:
            对应来源的 token 计数。

        Raises:
            ValueError: 如果 source 不是有效的来源标签。
        """
        source_map = {
            "local_compute": self.computed,
            "local_cache_hit": self.local_cache_hit,
            "external_kv_transfer": self.external_kv_transfer,
        }
        if source not in source_map:
            raise ValueError(f"Unknown source: {source}")
        return source_map[source]


class IterationStats:
    """单次调度迭代的统计数据。

    每次调度迭代（step）会处理一批请求的输出（EngineCoreOutput），
    本类聚合该迭代中所有请求的统计数据。

    聚合的指标包括：
    1. 生成 token 总数
    2. Prompt token 来源分布（本地计算 vs 缓存 vs 外部传输）
    3. 被抢占的请求数
    4. 完成的请求列表及其延迟统计
    5. 首 token 延迟（TTFT）和 token 间延迟（ITL）的采样
    6. 损坏请求计数

    Attributes:
        iteration_timestamp: 本次迭代的墙钟时间戳。
        num_generation_tokens: 本次迭代生成的 token 总数。
        prompt_token_stats: Prompt token 来源统计。
        num_preempted_reqs: 本次迭代中被抢占的请求数。
        finished_requests: 本次迭代完成的请求列表。
        max_num_generation_tokens_iter: 本次迭代各请求的最大生成 token 数列表。
        n_params_iter: 本次迭代各请求的参数数量列表。
        time_to_first_tokens_iter: 本次迭代的首 token 延迟列表。
        inter_token_latencies_iter: 本次迭代的 token 间延迟列表。
        num_corrupted_reqs: 本次迭代中发现的损坏请求数。
    """

    def __init__(self):
        self.iteration_timestamp = time.time()
        self.num_generation_tokens = 0
        self.prompt_token_stats = PromptTokenStats()
        self.num_preempted_reqs = 0
        self.finished_requests: list[FinishedRequestStats] = []
        self.max_num_generation_tokens_iter: list[int] = []
        self.n_params_iter: list[int] = []
        self.time_to_first_tokens_iter: list[float] = []
        self.inter_token_latencies_iter: list[float] = []
        self.num_corrupted_reqs: int = 0

    def __repr__(self) -> str:
        field_to_value_str = ", ".join(f"{k}={v}" for k, v in vars(self).items())
        return f"{self.__class__.__name__}({field_to_value_str})"

    @property
    def num_prompt_tokens(self) -> int:
        """Prompt token 总数（向后兼容属性）。"""
        return self.prompt_token_stats.total

    def _time_since(self, start: float) -> float:
        """计算相对于本次迭代时间戳的时间间隔。

        Args:
            start: 起始时间戳。

        Returns:
            从 start 到本次迭代时间戳的间隔（秒）。
        """
        return self.iteration_timestamp - start

    def update_from_output(
        self,
        output: "EngineCoreOutput",
        engine_core_timestamp: float,
        is_prefilling: bool,
        req_stats: RequestStateStats,
        lora_states: "LoRARequestStates",
        lora_name: str | None,
    ):
        """根据引擎核心输出更新迭代统计。

        每次引擎核心返回一个请求的输出时调用此方法。
        处理逻辑：

        1. 更新生成 token 计数
        2. 如果是预填充阶段：
           - 更新 prompt token 来源统计
           - 计算首 token 延迟（TTFT）
        3. 更新请求级别的生成 token 计数
        4. 检查 logits 中的 NaN（如果启用了相关环境变量）
        5. 处理请求级别的引擎核心事件（入队、调度、抢占等）
        6. 更新 token 间延迟（ITL）：
           - 预填充阶段：记录 first_token_ts
           - 解码阶段：计算与上一个 token 的时间间隔

        Args:
            output: 引擎核心输出，包含新生成的 token 和事件。
            engine_core_timestamp: 引擎核心的单调时间戳。
            is_prefilling: 该请求当前是否在预填充阶段。
            req_stats: 该请求的状态统计数据。
            lora_states: LoRA 适配器的请求状态管理器。
            lora_name: 该请求使用的 LoRA 适配器名称（可选）。
        """
        num_new_generation_tokens = len(output.new_token_ids)

        self.num_generation_tokens += num_new_generation_tokens
        if is_prefilling:
            if output.prefill_stats is not None:
                self.prompt_token_stats.update_from_output(output.prefill_stats)

            first_token_latency = self._time_since(req_stats.arrival_time)
            self.time_to_first_tokens_iter.append(first_token_latency)
            req_stats.first_token_latency = first_token_latency

        req_stats.num_generation_tokens += num_new_generation_tokens

        # Track if this request is corrupted (only check once per request)
        # Early exit if already marked as corrupted to avoid redundant checks
        if (
            envs.VLLM_COMPUTE_NANS_IN_LOGITS
            and not req_stats.is_corrupted
            and output.num_nans_in_logits > 0
        ):
            req_stats.is_corrupted = True

        # Process request-level engine core events
        if output.events is not None:
            self.update_from_events(
                output.request_id,
                output.events,
                is_prefilling,
                req_stats,
                lora_states,
                lora_name,
            )

        # Process the batch-level "new tokens" engine core event
        if is_prefilling:
            req_stats.first_token_ts = engine_core_timestamp
        else:
            itl = engine_core_timestamp - req_stats.last_token_ts
            self.inter_token_latencies_iter.append(itl)

        req_stats.last_token_ts = engine_core_timestamp

    def update_from_events(
        self,
        req_id: str,
        events: list["EngineCoreEvent"],
        is_prefilling: bool,
        req_stats: RequestStateStats,
        lora_states: "LoRARequestStates",
        lora_name: str | None,
    ):
        """处理请求级别的引擎核心事件。

        引擎核心事件标记了请求生命周期中的关键状态转换：
        1. QUEUED：请求进入等待队列，记录入队时间戳
        2. SCHEDULED：请求被调度执行，记录首次调度时间戳
           （忽略抢占后的重新调度）
        3. PREEMPTED：请求被抢占（因资源不足被暂停），
           更新抢占计数并将 LoRA 状态改回等待

        Args:
            req_id: 请求 ID。
            events: 引擎核心事件列表。
            is_prefilling: 该请求是否在预填充阶段。
            req_stats: 该请求的状态统计数据。
            lora_states: LoRA 适配器状态管理器。
            lora_name: LoRA 适配器名称。
        """
        # Avoid circular dependency
        from vllm.v1.engine import EngineCoreEventType

        for event in events:
            if event.type == EngineCoreEventType.QUEUED:
                req_stats.queued_ts = event.timestamp
                lora_states.request_waiting(req_id, lora_name)
            elif event.type == EngineCoreEventType.SCHEDULED:
                if req_stats.scheduled_ts == 0.0:  # ignore preemptions
                    req_stats.scheduled_ts = event.timestamp
                lora_states.request_running(req_id, lora_name)
            elif event.type == EngineCoreEventType.PREEMPTED:
                self.num_preempted_reqs += 1
                lora_states.request_waiting(req_id, lora_name)

    def update_from_finished_request(
        self,
        finish_reason: "FinishReason",
        request_id: str,
        num_prompt_tokens: int,
        max_tokens_param: int | None,
        req_stats: RequestStateStats,
        num_cached_tokens: int = 0,
    ):
        """处理请求完成时的统计更新。

        当一个请求完成时，计算其各种延迟指标并创建 FinishedRequestStats。

        延迟计算说明：
        - queued_time = scheduled_ts - queued_ts
          排队时间：从入队到首次被调度的时间
        - prefill_time = first_token_ts - scheduled_ts
          预填充时间：从首次调度到生成第一个 token 的时间
          （如果发生抢占，抢占时间也计入其中）
        - decode_time = last_token_ts - first_token_ts
          解码时间：从第一个 token 到最后一个 token 的时间
          （如果发生抢占，抢占时间也计入其中）
        - inference_time = last_token_ts - scheduled_ts
          推理总时间：从首次调度到最后一个 token 的时间
        - mean_time_per_output_token = decode_time / (num_generation_tokens - 1)
          平均每 token 解码时间：不计算预填充阶段生成的第一个 token

        Args:
            finish_reason: 请求结束原因。
            request_id: 请求 ID。
            num_prompt_tokens: 输入 prompt 的 token 数。
            max_tokens_param: 用户设置的最大生成 token 数。
            req_stats: 该请求的状态统计数据。
            num_cached_tokens: 从缓存命中的 token 数。
        """
        e2e_latency = self._time_since(req_stats.arrival_time)

        # Queued interval is from first QUEUED event to first SCHEDULED
        queued_time = req_stats.scheduled_ts - req_stats.queued_ts

        # Prefill interval is from first SCHEDULED to first NEW_TOKEN
        # Any preemptions during prefill is included in the interval
        prefill_time = req_stats.first_token_ts - req_stats.scheduled_ts

        # Decode interval is from first NEW_TOKEN to last NEW_TOKEN
        # Any preemptions during decode are included
        decode_time = req_stats.last_token_ts - req_stats.first_token_ts

        # Inference interval is from first SCHEDULED to last NEW_TOKEN
        # Any preemptions during prefill or decode are included
        inference_time = req_stats.last_token_ts - req_stats.scheduled_ts

        # Do not count the token generated by the prefill phase
        mean_time_per_output_token = (
            decode_time / (req_stats.num_generation_tokens - 1)
            if req_stats.num_generation_tokens - 1 > 0
            else 0
        )

        finished_req = FinishedRequestStats(
            finish_reason=finish_reason,
            request_id=request_id,
            e2e_latency=e2e_latency,
            num_prompt_tokens=num_prompt_tokens,
            num_generation_tokens=req_stats.num_generation_tokens,
            max_tokens_param=max_tokens_param,
            queued_time=queued_time,
            prefill_time=prefill_time,
            inference_time=inference_time,
            decode_time=decode_time,
            mean_time_per_output_token=mean_time_per_output_token,
            is_corrupted=req_stats.is_corrupted,
            num_cached_tokens=num_cached_tokens,
        )
        self.finished_requests.append(finished_req)

        # Count corrupted requests when they finish (only once per request)
        if req_stats.is_corrupted:
            self.num_corrupted_reqs += 1


class LoRAStats:
    """跟踪单个 LoRA 适配器的等待和运行请求 ID 集合。

    LoRA（Low-Rank Adaptation）允许在同一个基础模型上运行多个微调版本。
    每个 LoRA 适配器有自己的等待请求集和运行请求集，
    用于计算各适配器的负载情况。

    Attributes:
        waiting: 等待执行的请求 ID 集合。
        running: 正在运行的请求 ID 集合。
    """

    def __init__(self):
        self.waiting: set[str] = set()
        self.running: set[str] = set()

    def update(self, req_id: str, waiting: bool, running: bool):
        """更新请求在 LoRA 适配器中的状态。

        Args:
            req_id: 请求 ID。
            waiting: 是否在等待状态。
            running: 是否在运行状态。
        """
        assert not (waiting and running)
        if waiting:
            self.waiting.add(req_id)
        else:
            self.waiting.discard(req_id)

        if running:
            self.running.add(req_id)
        else:
            self.running.discard(req_id)

    @property
    def empty(self) -> bool:
        """返回该 LoRA 适配器是否没有任何活跃请求。"""
        return not (self.waiting or self.running)


class LoRARequestStates:
    """所有 LoRA 适配器的请求状态管理器。

    维护一个按 LoRA 名称索引的请求状态字典。
    每个 LoRA 适配器的等待和运行请求数会被同步到 SchedulerStats 中，
    用于指标上报。

    使用 defaultdict 自动为新的 LoRA 名称创建 LoRAStats 实例。
    当某个 LoRA 的所有请求都完成后，自动清理其条目以节省内存。

    Attributes:
        log_stats: 是否启用统计日志记录。
        requests: LoRA 名称到 LoRAStats 的映射字典。
    """

    def __init__(self, log_stats: bool = False):
        self.log_stats = log_stats
        self.requests: defaultdict[str, LoRAStats] = defaultdict(LoRAStats)

    def _request_update(
        self, req_id: str, lora_name: str | None, waiting: bool, running: bool
    ):
        """内部方法：更新请求在 LoRA 适配器中的状态。

        如果未启用统计日志或 lora_name 为 None，则跳过更新。

        Args:
            req_id: 请求 ID。
            lora_name: LoRA 适配器名称。
            waiting: 是否在等待状态。
            running: 是否在运行状态。
        """
        if not self.log_stats or lora_name is None:
            return

        lora_stats = self.requests[lora_name]
        lora_stats.update(req_id, waiting, running)
        if lora_stats.empty:
            del self.requests[lora_name]

    def request_waiting(self, req_id: str, lora_name: str | None):
        """标记请求进入等待状态。"""
        self._request_update(req_id, lora_name, waiting=True, running=False)

    def request_running(self, req_id: str, lora_name: str | None):
        """标记请求进入运行状态。"""
        self._request_update(req_id, lora_name, waiting=False, running=True)

    def request_finished(self, req_id: str, lora_name: str | None):
        """标记请求完成（从等待和运行状态中移除）。"""
        self._request_update(req_id, lora_name, waiting=False, running=False)

    def update_scheduler_stats(self, scheduler_stats: SchedulerStats | None):
        """将各 LoRA 适配器的请求计数同步到调度器统计中。

        遍历所有活跃的 LoRA 适配器，将其等待和运行的请求数
        写入 SchedulerStats 的对应字典中。

        Args:
            scheduler_stats: 调度器统计数据对象。
        """
        if not self.log_stats or scheduler_stats is None:
            return
        for lora_name, stats in self.requests.items():
            scheduler_stats.waiting_lora_adapters[lora_name] = len(stats.waiting)
            scheduler_stats.running_lora_adapters[lora_name] = len(stats.running)
