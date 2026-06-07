# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# 指标工具模块
#
# 本模块提供 Prometheus 指标的辅助工具函数。
# 主要功能是为每个引擎实例（engine）创建带有标签的指标子实例。
# 在多引擎（如数据并行）场景下，每个引擎实例需要独立的指标计数器，
# 通过不同的标签值来区分。

from typing import TypeAlias

from prometheus_client import Counter, Gauge, Histogram

# PromMetric 类型别名：表示 Prometheus 支持的三种指标类型
# - Gauge：可增可减的数值型指标（如当前队列长度）
# - Counter：单调递增的计数器（如总请求数）
# - Histogram：直方图型指标（如请求延迟分布）
PromMetric: TypeAlias = Gauge | Counter | Histogram


def create_metric_per_engine(
    metric: PromMetric,
    per_engine_labelvalues: dict[int, list[object]],
) -> dict[int, PromMetric]:
    """为每个引擎索引创建带标签的指标子实例。

    在多引擎部署场景（如数据并行 DP）中，同一个指标需要为每个引擎
    实例分别记录。本函数将一个基础指标通过不同的标签值（如引擎索引）
    派生出多个子指标实例。

    Args:
        metric: 基础的 Prometheus 指标对象（Counter/Gauge/Histogram）。
        per_engine_labelvalues: 引擎索引到标签值列表的映射。
            键为引擎索引（int），值为该引擎对应的标签值列表。
            例如：{0: ["engine_0"], 1: ["engine_1"]}

    Returns:
        一个字典，键为引擎索引，值为该引擎对应的已标签化指标子实例。
    """
    return {
        idx: metric.labels(*labelvalues)
        for idx, labelvalues in per_engine_labelvalues.items()
    }
