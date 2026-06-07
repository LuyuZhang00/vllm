# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Ray 指标包装器模块
#
# 本模块将 Prometheus 风格的指标 API 适配为 Ray 的指标库（ray.util.metrics），
# 使 vLLM 在 Ray Serve 环境下能够正确上报指标。
#
# 设计思路：
# Prometheus 和 Ray 的指标 API 在接口上有所不同。本模块通过包装器模式，
# 提供与 prometheus_client 完全兼容的 API，但底层使用 Ray 的指标实现。
# 这样上层代码（如 PrometheusStatLogger）无需修改即可在两种环境下工作。
#
# 包装器层次结构：
#   RayPrometheusMetric（基类）
#     ├── RayGaugeWrapper    （包装 ray.util.metrics.Gauge）
#     ├── RayCounterWrapper  （包装 ray.util.metrics.Counter）
#     └── RayHistogramWrapper（包装 ray.util.metrics.Histogram）
#
# 特殊适配类：
#   - RaySpecDecodingProm   ：推测解码指标的 Ray 版本
#   - RayKVConnectorProm    ：KV 连接器指标的 Ray 版本
#   - RayPerfMetricsProm    ：性能指标（MFU）的 Ray 版本
#   - RayPrometheusStatLogger：完整的 Ray 版统计日志记录器
#
# 关键差异处理：
# 1. 指标名称清理：Ray 使用 OpenTelemetry，不允许 ':' 等字符，
#    需要将非法字符替换为 '_'
# 2. 标签处理：Ray 指标自动按 WorkerId 分键，不支持 multiprocess_mode
# 3. ReplicaId：自动注入 Ray Serve 的副本 ID 作为标签

import copy
import time

from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorProm
from vllm.v1.metrics.loggers import PrometheusStatLogger
from vllm.v1.metrics.perf import PerfMetricsProm
from vllm.v1.spec_decode.metrics import SpecDecodingProm

try:
    from ray import serve as ray_serve
    from ray.util import metrics as ray_metrics
    from ray.util.metrics import Metric
except ImportError:
    ray_metrics = None
    ray_serve = None
import regex as re


def _get_replica_id() -> str | None:
    """获取当前 Ray Serve 副本的唯一 ID。

    在 Ray Serve 环境中，每个 vLLM 实例运行在一个副本（replica）中。
    本函数尝试获取当前副本的唯一标识符，用于指标标签区分不同副本。

    Returns:
        副本的唯一 ID 字符串，如果不在 Ray Serve 环境中则返回 None。
    """
    if ray_serve is None:
        return None
    try:
        return ray_serve.get_replica_context().replica_id.unique_id
    except ray_serve.exceptions.RayServeException:
        return None


class RayPrometheusMetric:
    """Ray 指标包装器的基类。

    提供了将 Prometheus 风格的指标操作（如 labels()、inc()、set()）
    映射到 Ray 指标库的公共基础设施。

    主要功能：
    1. 自动管理标签（tags），包括自动注入 ReplicaId
    2. 提供 labels() 方法来创建带标签的指标克隆
    3. 指标名称清理，确保兼容 OpenTelemetry 命名规范

    Attributes:
        _is_labeled: 标记该实例是否已经通过 labels() 创建（已标签化）。
            已标签化的实例不能再次调用 labels()。
        metric: 底层的 Ray 指标对象（由子类设置）。
        _tags: 标签字典，包含所有标签键值对。
    """

    _is_labeled: bool = False

    def __init__(self):
        if ray_metrics is None:
            raise ImportError("RayPrometheusMetric requires Ray to be installed.")
        self.metric: Metric = None
        self._tags: dict[str, str] = {"ReplicaId": _get_replica_id() or ""}

    @staticmethod
    def _get_tag_keys(labelnames: list[str] | None) -> tuple[str, ...]:
        """构建标签键列表，自动追加 ReplicaId。

        Ray 的指标系统要求在创建指标时指定标签键。
        本方法将用户指定的标签名与 ReplicaId 合并。

        Args:
            labelnames: 用户指定的标签名列表。

        Returns:
            包含所有标签键（含 ReplicaId）的元组。
        """
        labels = list(labelnames) if labelnames else []
        labels.append("ReplicaId")
        return tuple(labels)

    def _build_tags(self, *labels, **labelskwargs) -> dict[str, str]:
        """构建标签字典，用于 Ray 指标的 tags 参数。

        处理两种标签传递方式：
        1. 位置参数（*labels）：按标签键的顺序一一对应
        2. 关键字参数（**labelskwargs）：按名称指定标签值

        两种方式可以混合使用。所有标签值会被转换为字符串类型。
        ReplicaId 始终自动注入。

        Args:
            *labels: 位置参数形式的标签值（数量应等于标签键数 - 1，
                因为 ReplicaId 会自动添加）。
            **labelskwargs: 关键字参数形式的标签值。

        Returns:
            完整的标签字典。

        Raises:
            ValueError: 如果位置参数数量与预期不符。
        """
        if labels:
            # -1 because ReplicaId was added automatically
            expected = len(self.metric._tag_keys) - 1
            if len(labels) != expected:
                raise ValueError(
                    "Number of labels must match the number of tag keys. "
                    f"Expected {expected}, got {len(labels)}"
                )
            labelskwargs.update(zip(self.metric._tag_keys, labels))

        labelskwargs["ReplicaId"] = _get_replica_id() or ""

        return {k: v if isinstance(v, str) else str(v) for k, v in labelskwargs.items()}

    def labels(self, *labels, **labelskwargs) -> "RayPrometheusMetric":
        """创建一个带标签的指标克隆实例。

        模仿 prometheus_client 的 labels() API。返回的克隆实例
        包含指定的标签值，后续的操作（如 inc()、set()）会自动
        附带这些标签。

        注意：已标签化的实例不能再次调用 labels()。

        Args:
            *labels: 位置参数形式的标签值。
            **labelskwargs: 关键字参数形式的标签值。

        Returns:
            带有指定标签的新指标克隆实例。

        Raises:
            ValueError: 如果在已标签化的实例上调用。
        """
        if self._is_labeled:
            raise ValueError("labels() cannot be called on an already-labeled metric.")
        clone = copy.copy(self)
        clone._tags = self._build_tags(*labels, **labelskwargs)
        clone._is_labeled = True
        return clone

    @staticmethod
    def _get_sanitized_opentelemetry_name(name: str) -> str:
        """清理指标名称以兼容 OpenTelemetry 规范。

        Ray 使用 OpenTelemetry 作为底层指标系统，对指标名称有严格要求：
        - 允许的字符：a-z, A-Z, 0-9, _
        - 不允许的字符：如 ':' 等会被替换为 '_'

        例如："vllm:num_requests_running" -> "vllm_num_requests_running"

        Args:
            name: 原始指标名称。

        Returns:
            清理后的指标名称。

        References:
            - OpenTelemetry 规范：
              https://github.com/open-telemetry/opentelemetry-cpp/blob/main/sdk/src/metrics/instrument_metadata_validator.cc#L22-L23
            - Ray 实现：
              https://github.com/ray-project/ray/blob/master/src/ray/stats/metric.cc#L107
        """

        return re.sub(r"[^a-zA-Z0-9_]", "_", name)


class RayGaugeWrapper(RayPrometheusMetric):
    """Gauge 指标的 Ray 包装器。

    将 ray.util.metrics.Gauge 包装为与 prometheus_client.Gauge 兼容的 API。

    Gauge 是可增可减的数值型指标，用于记录瞬时值。
    例如：当前运行请求数、GPU 缓存使用率等。

    注意：Ray 的指标系统按 WorkerId 自动分键，
    multiprocess_mode（如 "mostrecent"、"all"、"sum"）不适用。
    这些聚合逻辑需要在可观测性层（Prometheus/Grafana）手动实现。
    """

    def __init__(
        self,
        name: str,
        documentation: str | None = "",
        labelnames: list[str] | None = None,
        multiprocess_mode: str | None = "",
    ):
        # All Ray metrics are keyed by WorkerId, so multiprocess modes like
        # "mostrecent", "all", "sum" do not apply. This logic can be manually
        # implemented at the observability layer (Prometheus/Grafana).
        del multiprocess_mode

        super().__init__()
        tag_keys = self._get_tag_keys(labelnames)
        name = self._get_sanitized_opentelemetry_name(name)

        self.metric = ray_metrics.Gauge(
            name=name,
            description=documentation,
            tag_keys=tag_keys,
        )

    def set(self, value: int | float):
        """设置 Gauge 指标的值。

        Args:
            value: 要设置的数值。
        """
        return self.metric.set(value, tags=self._tags)

    def set_to_current_time(self):
        """将 Gauge 设置为当前时间戳。

        Ray 的指标库没有原生的 set_to_current_time 方法，
        因此使用 time.time() 获取当前时间并通过 set() 设置。
        """
        # ray metrics doesn't have set_to_current time, https://docs.ray.io/en/latest/_modules/ray/util/metrics.html
        return self.set(time.time())


class RayCounterWrapper(RayPrometheusMetric):
    """Counter 指标的 Ray 包装器。

    将 ray.util.metrics.Counter 包装为与 prometheus_client.Counter 兼容的 API。

    Counter 是单调递增的计数器，用于记录累计值。
    例如：总请求数、总生成 token 数等。
    """

    def __init__(
        self,
        name: str,
        documentation: str | None = "",
        labelnames: list[str] | None = None,
    ):
        super().__init__()
        tag_keys = self._get_tag_keys(labelnames)
        name = self._get_sanitized_opentelemetry_name(name)
        self.metric = ray_metrics.Counter(
            name=name,
            description=documentation,
            tag_keys=tag_keys,
        )

    def inc(self, value: int | float = 1.0):
        """增加 Counter 指标的值。

        Args:
            value: 增量值。如果为 0 则跳过（避免不必要的更新）。
        """
        if value == 0:
            return
        return self.metric.inc(value, tags=self._tags)


class RayHistogramWrapper(RayPrometheusMetric):
    """Histogram 指标的 Ray 包装器。

    将 ray.util.metrics.Histogram 包装为与 prometheus_client.Histogram 兼容的 API。

    Histogram 用于记录值的分布情况，按桶（bucket）统计。
    例如：请求延迟分布、每 token 生成时间分布等。
    """

    def __init__(
        self,
        name: str,
        documentation: str | None = "",
        labelnames: list[str] | None = None,
        buckets: list[float] | None = None,
    ):
        super().__init__()
        tag_keys = self._get_tag_keys(labelnames)
        name = self._get_sanitized_opentelemetry_name(name)

        boundaries = buckets if buckets else []
        self.metric = ray_metrics.Histogram(
            name=name,
            description=documentation,
            tag_keys=tag_keys,
            boundaries=boundaries,
        )

    def observe(self, value: int | float):
        """向 Histogram 记录一个观测值。

        Args:
            value: 要记录的观测值。
        """
        return self.metric.observe(value, tags=self._tags)


class RaySpecDecodingProm(SpecDecodingProm):
    """推测解码指标的 Ray 版本。

    继承自 SpecDecodingProm，将 Counter 类替换为 RayCounterWrapper，
    使推测解码的指标能够在 Ray 环境下正确上报。

    推测解码（Speculative Decoding）使用小型"草稿"模型快速生成候选 token，
    然后由大模型验证。相关指标包括接受率、每位置接受 token 数等。
    """

    _counter_cls = RayCounterWrapper


class RayKVConnectorProm(KVConnectorProm):
    """KV 连接器指标的 Ray 版本。

    继承自 KVConnectorProm，将所有指标类替换为 Ray 包装器。

    KV 连接器用于分布式 KV Cache 场景（如 disaggregated prefill/decode），
    在不同节点之间传输 KV Cache 数据。相关指标包括传输延迟、
    缓存命中率等。
    """

    _gauge_cls = RayGaugeWrapper
    _counter_cls = RayCounterWrapper
    _histogram_cls = RayHistogramWrapper


class RayPerfMetricsProm(PerfMetricsProm):
    """性能指标（MFU）的 Ray 版本。

    继承自 PerfMetricsProm，将 Counter 类替换为 RayCounterWrapper。

    MFU（Model Flops Utilization）指标记录模型的 FLOPs 和内存带宽利用率，
    用于评估硬件利用效率。
    """

    _counter_cls = RayCounterWrapper


class RayPrometheusStatLogger(PrometheusStatLogger):
    """Ray 版本的统计日志记录器。

    继承自 PrometheusStatLogger，将所有指标类替换为 Ray 包装器版本。
    这是 vLLM 在 Ray Serve 环境下使用的完整指标记录器。

    替换的指标类：
    - _gauge_cls          -> RayGaugeWrapper
    - _counter_cls        -> RayCounterWrapper
    - _histogram_cls      -> RayHistogramWrapper
    - _spec_decoding_cls  -> RaySpecDecodingProm
    - _kv_connector_cls   -> RayKVConnectorProm
    - _perf_metrics_cls   -> RayPerfMetricsProm

    特殊处理：
    - _unregister_vllm_metrics()：在 Ray 环境下为空操作，
      因为 Ray 有自己的指标生命周期管理。
    """

    _gauge_cls = RayGaugeWrapper
    _counter_cls = RayCounterWrapper
    _histogram_cls = RayHistogramWrapper
    _spec_decoding_cls = RaySpecDecodingProm
    _kv_connector_cls = RayKVConnectorProm
    _perf_metrics_cls = RayPerfMetricsProm

    @staticmethod
    def _unregister_vllm_metrics():
        # No-op on purpose
        pass
