# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# 指标读取模块
#
# 本模块提供从 Prometheus 内存注册表中读取指标快照的 API。
# 它将 Prometheus 的内部指标格式转换为简洁的 Python 数据类，
# 方便用户程序化地访问 vLLM 的运行时指标。
#
# 支持的指标类型：
# 1. Counter  - 单调递增计数器（如：总请求数、总生成 token 数）
# 2. Gauge    - 可变数值（如：当前队列长度、KV Cache 使用率）
# 3. Histogram - 直方图（如：请求延迟分布、每 token 生成时间）
# 4. Vector   - 向量型计数器（特殊类型，用于推测解码的每个位置的接受 token 数）
#
# 典型使用方式：
#     for metric in get_metrics_snapshot():
#         if isinstance(metric, Counter):
#             print(f"{metric.name} = {metric.value}")

from dataclasses import dataclass

from prometheus_client import REGISTRY
from prometheus_client import Metric as PromMetric
from prometheus_client.samples import Sample


@dataclass
class Metric:
    """Prometheus 指标的基类。

    每个指标可以关联一组 key=value 标签。
    在某些情况下（如数据并行），同一个 vLLM 实例可能有多个
    同名但标签不同的指标。

    Attributes:
        name: 指标名称，如 "vllm:num_requests_running"。
        labels: 标签字典，如 {"engine_index": "0"}。
    """

    name: str
    labels: dict[str, str]


@dataclass
class Counter(Metric):
    """单调递增的整数计数器。

    用于记录只增不减的累计值，例如：
    - vllm:num_requests_total：总请求数
    - vllm:generation_tokens_total：总生成 token 数

    Attributes:
        value: 计数器的当前整数值。
    """

    value: int


@dataclass
class Vector(Metric):
    """有序的整数计数器数组。

    这是一个 Prometheus 中不存在的特殊类型，专门用于建模
    vllm:spec_decode_num_accepted_tokens_per_pos 指标。

    在推测解码中，每个位置（position）对应一个接受 token 计数，
    将这些计数组成一个有序数组更便于分析。

    Attributes:
        values: 按位置索引排列的整数计数值列表。
            例如：[10, 7, 2] 表示位置 0 接受了 10 个 token，
            位置 1 接受了 7 个，位置 2 接受了 2 个。
    """

    values: list[int]


@dataclass
class Gauge(Metric):
    """可增可减的数值型指标。

    用于记录可以双向变化的瞬时值，例如：
    - vllm:num_requests_running：当前正在运行的请求数
    - vllm:gpu_cache_usage_perc：GPU 缓存使用率

    Attributes:
        value: 指标的当前浮点数值。
    """

    value: float


@dataclass
class Histogram(Metric):
    """直方图型指标，按可配置的桶记录观测值。

    用于记录值的分布情况，例如：
    - vllm:e2e_request_latency_seconds：端到端请求延迟分布
    - vllm:time_per_output_token_seconds：每输出 token 耗时分布

    Prometheus 直方图的工作原理：
    1. 定义一组桶边界（如 0.1, 0.5, 1.0, 5.0, +Inf）
    2. 每个观测值会被累加到所有大于等于它的桶中（累积计数）
    3. '+Inf' 桶始终存在，其值等于总观测次数

    Attributes:
        count: 总观测次数（等于 '+Inf' 桶的值）。
        sum: 所有观测值的总和。
        buckets: 桶分布字典。键为桶的上界字符串（如 "1.0", "+Inf"），
            值为该桶中的累积观测次数。
    """

    count: int
    sum: float
    buckets: dict[str, int]


def get_metrics_snapshot() -> list[Metric]:
    """获取当前 Prometheus 注册表中所有 vLLM 指标的快照。

    本函数遍历 Prometheus 注册表中的所有指标，筛选出以 "vllm:" 为前缀的指标，
    并将其转换为对应的 Python 数据类对象。

    处理流程：
    1. 遍历 REGISTRY 中的所有指标
    2. 过滤出名称以 "vllm:" 开头的指标
    3. 根据指标类型（gauge/counter/histogram）分别处理：
       - Gauge：直接转换为 Gauge 数据类
       - Counter：转换为 Counter 数据类（特殊处理推测解码向量指标）
       - Histogram：解析桶样本，转换为 Histogram 数据类

    Returns:
        包含所有 vLLM 指标快照的列表。每个指标是一个 Metric 子类实例。

    Example:
        >>> for metric in llm.get_metrics():
        ...     if isinstance(metric, Counter):
        ...         print(f"{metric} = {metric.value}")
        ...     elif isinstance(metric, Gauge):
        ...         print(f"{metric} = {metric.value}")
        ...     elif isinstance(metric, Histogram):
        ...         print(f"{metric}")
        ...         print(f"    sum = {metric.sum}")
        ...         print(f"    count = {metric.count}")
        ...         for bucket_le, value in metrics.buckets.items():
        ...             print(f"    {bucket_le} = {value}")
    """
    collected: list[Metric] = []
    for metric in REGISTRY.collect():
        if not metric.name.startswith("vllm:"):
            continue
        if metric.type == "gauge":
            samples = _get_samples(metric)
            for s in samples:
                collected.append(
                    Gauge(name=metric.name, labels=s.labels, value=s.value)
                )
        elif metric.type == "counter":
            samples = _get_samples(metric, "_total")
            if metric.name == "vllm:spec_decode_num_accepted_tokens_per_pos":
                #
                # Ugly vllm:num_accepted_tokens_per_pos special case.
                #
                # This metric is a vector of counters - for each spec
                # decoding token position, we observe the number of
                # accepted tokens using a Counter labeled with 'position'.
                # We convert these into a vector of integer values.
                #
                # 特殊处理推测解码的"每个位置接受 token 数"指标。
                # 该指标使用带 'position' 标签的 Counter 来记录每个
                # 推测解码位置的接受 token 数。这里将它们合并为整数数组。
                for labels, values in _digest_num_accepted_by_pos_samples(samples):
                    collected.append(
                        Vector(name=metric.name, labels=labels, values=values)
                    )
            else:
                for s in samples:
                    collected.append(
                        Counter(name=metric.name, labels=s.labels, value=int(s.value))
                    )

        elif metric.type == "histogram":
            #
            # A histogram has a number of '_bucket' samples where
            # the 'le' label represents the upper limit of the bucket.
            # We convert these bucketized values into a dict of values
            # indexed by the value of the 'le' label. The 'le=+Inf'
            # label is a special case, catching all values observed.
            #
            # Histogram 类型的指标包含以下后缀的样本：
            #   _bucket：各桶的累积计数（'le' 标签表示桶的上界）
            #   _count：总观测次数
            #   _sum：所有观测值的总和
            # 将这些样本解析并合并为 Histogram 数据类。
            bucket_samples = _get_samples(metric, "_bucket")
            count_samples = _get_samples(metric, "_count")
            sum_samples = _get_samples(metric, "_sum")
            for labels, buckets, count_value, sum_value in _digest_histogram(
                bucket_samples, count_samples, sum_samples
            ):
                collected.append(
                    Histogram(
                        name=metric.name,
                        labels=labels,
                        buckets=buckets,
                        count=count_value,
                        sum=sum_value,
                    )
                )
        else:
            raise AssertionError(f"Unknown metric type {metric.type}")

    return collected


def _get_samples(metric: PromMetric, suffix: str | None = None) -> list[Sample]:
    """从指标中筛选匹配指定后缀的样本。

    Args:
        metric: Prometheus 指标对象。
        suffix: 指标名称后缀（如 "_total", "_bucket", "_count", "_sum"）。
            如果为 None，则匹配指标原名。

    Returns:
        匹配的样本列表。
    """
    name = (metric.name + suffix) if suffix is not None else metric.name
    return [s for s in metric.samples if s.name == name]


def _strip_label(labels: dict[str, str], key_to_remove: str) -> dict[str, str]:
    """从标签字典中移除指定键，返回新的字典副本。

    Args:
        labels: 原始标签字典。
        key_to_remove: 要移除的键名。

    Returns:
        移除指定键后的新标签字典。
    """
    labels_copy = labels.copy()
    labels_copy.pop(key_to_remove)
    return labels_copy


def _digest_histogram(
    bucket_samples: list[Sample], count_samples: list[Sample], sum_samples: list[Sample]
) -> list[tuple[dict[str, str], dict[str, int], int, float]]:
    """解析直方图样本，将分散的桶/计数/求和样本合并为结构化数据。

    在数据并行（DP）场景下，每个引擎实例都有自己的直方图样本，
    且样本带有引擎标识标签。本函数将这些样本按标签分组，
    合并为每个引擎实例的完整直方图数据。

    处理流程：
    1. 将桶样本按标签分组，构建 {标签集合 -> {桶上界 -> 计数}} 映射
    2. 将计数样本按标签分组，构建 {标签集合 -> 总计数} 映射
    3. 将求和样本按标签分组，构建 {标签集合 -> 总和} 映射
    4. 验证三者的标签集合一致
    5. 将三者合并为输出列表

    Args:
        bucket_samples: 桶累积计数样本列表，每个样本的 'le' 标签表示桶上界。
        count_samples: 总观测次数样本列表。
        sum_samples: 观测值总和样本列表。

    Returns:
        列表，每个元素为 (标签字典, 桶分布字典, 总计数, 总和) 的元组。

    Example:
        输入样本（DP 场景，2 个引擎）：
        bucket_samples:
          labels={le: 100, idx: 0}, value=2
          labels={le: 200, idx: 0}, value=4
          labels={le: Inf, idx: 0}, value=10
          labels={le: 100, idx: 1}, value=1
          labels={le: 200, idx: 1}, value=5
          labels={le: Inf, idx: 1}, value=7
        count_samples:
          labels={idx: 0}, value=10
          labels={idx: 1}, value=7
        sum_samples:
          labels={idx: 0}, value=2000
          labels={idx: 1}, value=1200

        输出：
        [
          ({idx: 0}, {"100": 2, "200": 4, "Inf": 10}, 10, 2000),
          ({idx: 1}, {"100": 1, "200": 5, "Inf": 7},   7, 1200),
        ]
    #
    # In the case of DP, we have an indigestable
    # per-bucket-per-engine count as a list of labelled
    # samples, along with total and sum samples
    #
    # bucket_samples (in):
    #   labels = {bucket: 100, idx: 0}, value = 2
    #   labels = {bucket: 200, idx: 0}, value = 4
    #   labels = {bucket: Inf, idx: 0}, value = 10
    #   labels = {bucket: 100, idx: 1}, value = 1
    #   labels = {bucket: 200, idx: 2}, value = 5
    #   labels = {bucket: Inf, idx: 3}, value = 7
    # count_samples (in):
    #   labels = {idx: 0}, value = 10
    #   labels = {idx: 1}, value = 7
    # sum_samples (in):
    #   labels = {idx: 0}, value = 2000
    #   labels = {idx: 1}, value = 1200
    #
    # output: [
    #   {idx: 0}, {"100": 2, "200": 4, "Inf": 10}, 10, 2000
    #   {idx: 1}, {"100": 1, "200": 5, "Inf": 7},   7, 1200
    # ]
    """
    buckets_by_labels: dict[frozenset[tuple[str, str]], dict[str, int]] = {}
    for s in bucket_samples:
        bucket = s.labels["le"]
        labels_key = frozenset(_strip_label(s.labels, "le").items())
        if labels_key not in buckets_by_labels:
            buckets_by_labels[labels_key] = {}
        buckets_by_labels[labels_key][bucket] = int(s.value)

    counts_by_labels: dict[frozenset[tuple[str, str]], int] = {}
    for s in count_samples:
        labels_key = frozenset(s.labels.items())
        counts_by_labels[labels_key] = int(s.value)

    sums_by_labels: dict[frozenset[tuple[str, str]], float] = {}
    for s in sum_samples:
        labels_key = frozenset(s.labels.items())
        sums_by_labels[labels_key] = s.value

    assert (
        set(buckets_by_labels.keys())
        == set(counts_by_labels.keys())
        == set(sums_by_labels.keys())
    )

    output = []
    label_keys = list(buckets_by_labels.keys())
    for k in label_keys:
        labels = dict(k)
        output.append(
            (labels, buckets_by_labels[k], counts_by_labels[k], sums_by_labels[k])
        )
    return output


def _digest_num_accepted_by_pos_samples(
    samples: list[Sample],
) -> list[tuple[dict[str, str], list[int]]]:
    """解析推测解码的"每个位置接受 token 数"样本。

    在数据并行（DP）场景下，每个引擎实例有按位置（position）标记的样本。
    本函数将这些样本按引擎标签分组，并将每个位置的计数值合并为有序数组。

    处理流程：
    1. 遍历所有样本，按标签分组并记录每个位置的值
    2. 确定最大位置索引，以便创建固定长度的数组
    3. 为每组标签创建一个长度为 (max_pos + 1) 的数组，
       将每个位置的值填入对应索引

    Args:
        samples: 带有 'position' 标签的 Prometheus 样本列表。

    Returns:
        列表，每个元素为 (标签字典, 位置值数组) 的元组。

    Example:
        输入样本（DP 场景，2 个引擎）：
          labels={pos: 0, idx: 0}, value=10
          labels={pos: 1, idx: 0}, value=7
          labels={pos: 2, idx: 0}, value=2
          labels={pos: 0, idx: 1}, value=5
          labels={pos: 1, idx: 1}, value=3
          labels={pos: 2, idx: 1}, value=1

        输出：
        [
          ({idx: 0}, [10, 7, 2]),
          ({idx: 1}, [5, 3, 1]),
        ]
    #
    # In the case of DP, we have an indigestable
    # per-position-per-engine count as a list of
    # labelled samples
    #
    # samples (in):
    #   labels = {pos: 0, idx: 0}, value = 10
    #   labels = {pos: 1, idx: 0}, value = 7
    #   labels = {pos: 2, idx: 0}, value = 2
    #   labels = {pos: 0, idx: 1}, value = 5
    #   labels = {pos: 1, idx: 1}, value = 3
    #   labels = {pos: 2, idx: 1}, value = 1
    #
    # output: [
    #   {idx: 0}, [10, 7, 2]
    #   {idx: 1}, [5, 3, 1]
    # ]
    """
    max_pos = 0
    values_by_labels: dict[frozenset[tuple[str, str]], dict[int, int]] = {}

    for s in samples:
        position = int(s.labels["position"])
        max_pos = max(max_pos, position)

        labels_key = frozenset(_strip_label(s.labels, "position").items())
        if labels_key not in values_by_labels:
            values_by_labels[labels_key] = {}
        values_by_labels[labels_key][position] = int(s.value)

    output = []
    for labels_key, values_by_position in values_by_labels.items():
        labels = dict(labels_key)
        values = [0] * (max_pos + 1)
        for pos, val in values_by_position.items():
            values[pos] = val
        output.append((labels, values))
    return output
