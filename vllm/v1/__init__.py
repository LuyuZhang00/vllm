# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
vLLM v1 引擎模块 (vllm/v1/__init__.py)

本模块是 vLLM v1 引擎的顶层包。

vLLM v1 引擎是当前的默认引擎，采用多进程架构，包含以下核心组件：
1. 调度器 (Scheduler): 决定每个迭代运行哪些请求
2. 模型运行器 (ModelRunner): 执行模型前向传播
3. 执行器 (Executor): 管理工作器进程
4. 引擎核心 (EngineCore): 主引擎循环，协调各组件
5. 输入/输出处理器: 负责 tokenize 和 detokenize

与旧版引擎 (vllm/engine/) 的区别：
- v1 使用 ZMQ IPC 进行进程间通信（而非共享内存）
- v1 支持更灵活的调度策略（如 chunked prefill、speculative decoding）
- v1 的 KV 缓存管理更高效（支持前缀缓存、分层卸载等）

主要子模块：
- core/: 引擎核心组件（调度器、KV 缓存管理等）
- engine/: 引擎进程和通信
- executor/: 工作器执行器
- worker/: GPU/CPU 工作器
- pool/: 池化（pooling）相关功能
- kv_offload/: KV 缓存卸载
- metrics/: 指标收集
- structured_output/: 结构化输出（JSON schema、正则表达式等）
"""
