# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# 指标（Metrics）包
#
# 本包负责 vLLM v1 引擎的指标采集、统计与上报。主要包含以下模块：
#
# 1. perf.py        - 性能指标估算模块，基于模型配置解析 FLOPs 和内存带宽，
#                     用于计算 MFU（Model Flops Utilization，模型算力利用率）。
# 2. stats.py       - 统计数据收集模块，跟踪请求生命周期中的各项统计数据，
#                     包括缓存命中率、KV Cache 使用率、延迟等。
# 3. prometheus.py  - Prometheus 集成模块，处理多进程环境下 Prometheus
#                     指标的注册、收集和清理。
# 4. reader.py      - 指标读取模块，提供从 Prometheus 内存注册表中读取
#                     当前指标快照的 API。
# 5. ray_wrappers.py - Ray 指标包装器模块，将 Prometheus 风格的指标 API
#                      适配为 Ray 的指标库，用于 Ray Serve 环境。
# 6. utils.py       - 指标工具模块，提供创建每引擎标签化指标的辅助函数。
