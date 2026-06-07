# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
KV 缓存分层卸载模块 (vllm/v1/kv_offload/tiering/__init__.py)

本模块实现了 KV 缓存的多层 (tiering) 卸载策略框架。

分层卸载 (Tiering) 的核心思想：
- KV 缓存数据根据访问频率和重要性，存储在不同性能层级的存储介质上：
  1. GPU 显存（最高性能，容量最小）
  2. CPU 内存（中等性能，容量较大）
  3. 文件系统/SSD（较低性能，容量最大）
- 高频访问的数据保留在 GPU/内存中，低频访问的数据下沉到慢速存储
- 当需要使用下沉数据时，再从慢速存储加载回快速存储

本模块提供了分层策略的基础设施，具体实现分布在子模块中：
- fs/: 基于文件系统的分层实现（将 KV 缓存持久化到磁盘）
- example/: 分层策略的示例实现

该框架通过 OffloadingSpec / OffloadingManager 接口与调度器和工作器集成。
"""
