# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
KV 缓存卸载工作器模块 (vllm/v1/kv_offload/worker/__init__.py)

本模块实现了 KV 缓存卸载的工作器端逻辑。

工作器端的职责：
- 执行实际的 KV 缓存数据传输（GPU <-> CPU、GPU <-> 磁盘等）
- 管理传输流和事件（异步传输）
- 报告传输结果给调度器

核心组件（定义在 worker.py 中）：
1. OffloadingHandler: 卸载处理器接口
   - transfer_async(): 异步启动数据传输
   - get_finished(): 获取已完成的传输
   - wait(): 等待特定传输完成
   - shutdown(): 关闭处理器

2. TransferSpec: 传输规格，描述源和目标的加载/存储规格
3. TransferResult: 传输结果，包含成功状态、传输大小和耗时

工作流程：
1. 调度器通过 SchedulerOutput 通知工作器需要执行的传输
2. 工作器调用 handler.transfer_async() 启动传输
3. 工作器在模型计算间隙调用 handler.get_finished() 检查完成的传输
4. 完成的传输结果通过 EngineCoreOutputs 返回给调度器
"""
