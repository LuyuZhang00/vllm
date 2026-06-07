# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
KV 缓存卸载模块 (vllm/v1/kv_offload/__init__.py)

本模块实现了 KV 缓存的卸载 (offloading) 功能，允许将 GPU 上的 KV 缓存数据
迁移到其他存储介质（如 CPU 内存、文件系统等），以释放 GPU 显存。

KV 缓存卸载的核心概念：
1. OffloadingManager: 卸载管理器，负责管理缓存块的生命周期（分配、查找、淘汰）
2. OffloadingSpec: 卸载规格，定义卸载目标的配置（如 CPU 内存大小、淘汰策略等）
3. LoadStoreSpec: 加载/存储规格，描述数据传输的具体参数（源/目标块 ID）
4. OffloadingHandler: 卸载处理器，执行实际的数据传输（GPU <-> CPU、GPU <-> 磁盘等）

卸载流程概述：
1. 调度器决定哪些 KV 缓存块需要卸载（通过 OffloadingManager 的 prepare_store）
2. 工作器通过 OffloadingHandler 执行实际的数据传输
3. 传输完成后，通过 complete_store 确认存储成功
4. 后续需要使用时，通过 prepare_load 从卸载目标加载回 GPU

本模块包含以下子模块：
- cpu/: 基于 CPU 内存的卸载实现
- tiering/: 多层卸载策略框架
- worker/: 卸载工作器（执行实际数据传输）
- base.py: 基类和接口定义
"""
