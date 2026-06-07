# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
文件系统分层卸载模块 (vllm/v1/kv_offload/tiering/fs/__init__.py)

本模块实现了基于文件系统的 KV 缓存分层卸载策略。

文件系统分层的核心思路：
- 将 KV 缓存块持久化存储到本地文件系统（通常是 SSD 或 NVMe 设备）
- 适用于 GPU/CPU 内存不足以容纳全部 KV 缓存的场景
- 通过文件 I/O 实现数据的卸载和加载

典型使用场景：
1. 长上下文推理：大量 KV 缓存数据超出 GPU/CPU 内存容量
2. 多请求共享前缀：相同前缀的请求可以共享已持久化的 KV 缓存
3. 冷数据卸载：长时间未访问的 KV 缓存下沉到文件系统

本模块目前为空占位，具体实现待后续添加。
"""
