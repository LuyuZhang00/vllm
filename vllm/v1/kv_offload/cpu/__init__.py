# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
CPU KV 缓存卸载模块 (vllm/v1/kv_offload/cpu/__init__.py)

本模块实现了基于 CPU 内存的 KV 缓存卸载功能，允许将 GPU 上的 KV 缓存数据
迁移到 CPU 内存中，以释放 GPU 显存用于新的计算。

CPU 卸载的工作原理：
1. 在 CPU 内存中预分配一块连续区域作为 KV 缓存的"二级缓存"
2. 当 GPU 显存紧张时，将不常用的 KV 缓存块传输到 CPU 内存
3. 当需要再次使用这些 KV 缓存块时，从 CPU 内存加载回 GPU
4. 传输通过 CUDA 流异步执行，尽量与模型计算重叠

模块组成：
- spec.py: CPUOffloadingSpec，定义 CPU 卸载的配置规格
- manager.py: CPUOffloadingManager，管理 CPU 端的缓存块生命周期
- gpu_worker.py: CpuGpuOffloadingHandlers，处理 GPU <-> CPU 数据传输
- shared_offload_region.py: SharedOffloadRegion，多进程共享的 mmap 内存区域
- common.py: CPULoadStoreSpec，CPU 加载/存储规格
- policies/: 缓存淘汰策略（LRU、ARC 等）
"""
