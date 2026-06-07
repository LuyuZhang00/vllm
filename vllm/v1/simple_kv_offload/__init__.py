# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
简单KV缓存CPU卸载模块 (Simple KV Cache CPU Offloading Module)

本模块实现了将GPU上的KV缓存卸载到CPU内存的功能，以支持更大的上下文窗口。
当GPU显存不足时，可以将不活跃的KV缓存块转移到CPU内存中，需要时再加载回来。

模块组成：
1. manager.py - 调度器端管理器，负责决策哪些块需要卸载/加载
2. worker.py - 工作节点端处理器，执行实际的数据传输
3. copy_backend.py - DMA复制后端，使用后台线程执行批量内存拷贝
4. cuda_mem_ops.py - CUDA内存操作，包括内存锁定和批量DMA传输
5. metadata.py - 元数据定义，用于调度器和工作节点之间的通信

工作流程：
1. 调度器分析请求的KV缓存需求，决定需要从CPU加载哪些块
2. 调度器管理器构建元数据，包含加载/存储操作的块ID映射
3. 元数据传递给工作节点
4. 工作节点使用DMA后端执行异步的GPU<->CPU数据传输
5. 传输完成后，通过事件机制通知调度器

支持两种卸载模式：
- 急切模式(Eager)：立即将新计算的块卸载到CPU
- 懒惰模式(Lazy)：只在GPU内存紧张时才卸载接近被淘汰的块
"""

from vllm.v1.simple_kv_offload.manager import SimpleCPUOffloadScheduler
from vllm.v1.simple_kv_offload.worker import SimpleCPUOffloadWorker

__all__ = [
    "SimpleCPUOffloadScheduler",
    "SimpleCPUOffloadWorker",
]
