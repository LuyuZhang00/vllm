# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
调度器模块 (vllm/v1/core/sched/__init__.py)

本模块实现了 vLLM v1 引擎的调度器，是引擎中最核心的组件之一。

调度器的职责：
1. 请求管理：维护等待队列、运行集合和已完成集合
2. 调度决策：每个迭代决定处理哪些请求以及每个请求处理多少 token
3. KV 缓存管理：分配和回收 KV 缓存块
4. 前缀缓存：利用已缓存的公共前缀避免重复计算
5. 抢占 (Preemption)：当显存不足时，暂停低优先级请求
6. 分块预填充 (Chunked Prefill)：将长 prompt 分成多个 chunk 处理

调度输出 (SchedulerOutput)：
- scheduled_new_reqs: 首次调度的新请求（包含完整请求数据）
- scheduled_cached_reqs: 已缓存请求的增量数据（减少通信开销）
- num_scheduled_tokens: 每个请求的调度 token 数
- scheduled_spec_decode_tokens: 投机解码的 token
- finished_req_ids: 已完成的请求 ID

调度流程：
1. schedule() 被引擎主循环反复调用
2. 从等待队列中取出请求，检查 KV 缓存块是否足够
3. 如果足够，将请求移入运行集合并分配缓存块
4. 如果不够，尝试抢占低优先级请求或跳过
5. 生成 SchedulerOutput 发送给模型运行器
6. update_from_output() 处理模型输出，更新请求状态

模块组成：
- scheduler.py: 主调度器实现
- async_scheduler.py: 异步调度器（支持投机解码）
- interface.py: 调度器接口定义和暂停状态枚举
- output.py: 调度输出数据结构
- request_queue.py: 请求队列实现（FCFS 和优先级队列）
- utils.py: 调度辅助函数（重复检测、停止条件等）
"""
