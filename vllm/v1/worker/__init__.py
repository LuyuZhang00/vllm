# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
vllm/v1/worker -- Worker 模块（工作进程模块）

=============================================================
【模块概述】
=============================================================
本模块实现了 vLLM V1 引擎的工作进程（Worker）层。
Worker 是引擎调度器（Scheduler）与实际模型推理之间的桥梁，
负责设备初始化、模型加载、KV Cache 分配以及执行前向推理。

=============================================================
【主要组件】
=============================================================
1. GPUWorker（gpu_worker.py）
   - GPU 工作进程的主实现，管理 CUDA 设备上的模型推理。
   - 包含模型加载、KV Cache 初始化、执行推理等功能。

2. CPUWorker（cpu_worker.py）
   - CPU 后端的工作进程，继承自 GPUWorker 但移除了所有 CUDA 相关逻辑。
   - 用于在纯 CPU 环境下运行模型推理。

3. GPUModelRunner（gpu_model_runner.py）
   - GPU 模型运行器，管理输入批次（InputBatch）、执行模型前向传播。
   - 支持 CUDA Graph 捕获/回放以优化推理性能。

4. CPUModelRunner（cpu_model_runner.py）
   - CPU 模型运行器，继承自 GPUModelRunner 并替换 GPU 特有的操作。

5. InputBatch（gpu_input_batch.py / tpu_input_batch.py）
   - 输入批次数据结构，管理一个批次中所有请求的状态。
   - 包括 token ID、采样参数、block table 等。

6. EncoderCudaGraphManager（encoder_cudagraph.py）
   - 视觉编码器的 CUDA Graph 管理器，用于优化多模态模型的编码器推理。

=============================================================
【Worker 与 Scheduler 的交互流程】
=============================================================
1. Scheduler 决定当前迭代要执行哪些请求，生成 SchedulerOutput。
2. SchedulerOutput 通过 IPC（ZMQ）发送给 Worker。
3. Worker 将 SchedulerOutput 转化为模型所需的输入张量。
4. Worker 调用 ModelRunner 执行前向传播。
5. 推理结果（输出 token）通过 IPC 返回给 Scheduler。
"""
