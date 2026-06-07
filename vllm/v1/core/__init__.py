# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
引擎核心模块 (vllm/v1/core/__init__.py)

本模块是 vLLM v1 引擎核心组件的顶层包。

核心组件包括：
1. KV 缓存管理 (kv_cache_manager.py): 管理 KV 缓存块的分配和释放
2. 调度器 (sched/): 决定每个迭代运行哪些请求
3. 编码器缓存管理 (encoder_cache_manager.py): 管理多模态编码器输出的缓存
4. KV 缓存指标 (kv_cache_metrics.py): 收集 KV 缓存使用统计

调度器是引擎的"大脑"，负责：
- 维护请求队列（等待、运行、已完成）
- 决定每个迭代处理哪些请求以及每个请求处理多少 token
- 管理 KV 缓存块的分配和回收
- 处理 preemption（抢占）和 chunked prefill（分块预填充）
"""
