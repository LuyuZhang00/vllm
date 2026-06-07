# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Metadata for SimpleCPUOffloadConnector.

SimpleCPUOffloadConnector的元数据定义。

本模块定义了调度器和工作节点之间通信的元数据结构：
1. SimpleCPUOffloadMetadata: 调度器->工作节点的元数据
2. SimpleCPUOffloadWorkerMetadata: 工作节点->调度器的元数据

这些元数据用于：
- 传递需要执行的加载/存储操作的块ID映射
- 跟踪异步传输的完成状态
- 在多工作节点（TP/PP）场景下聚合完成事件
"""

from dataclasses import dataclass, field

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata,
    KVConnectorWorkerMetadata,
)

# 无效的作业ID，表示没有需要执行的操作
INVALID_JOB_ID = -1


@dataclass
class SimpleCPUOffloadMetadata(KVConnectorMetadata):
    """
    Metadata passed from scheduler to worker for CPU offload operations.

    从调度器传递给工作节点的CPU卸载操作元数据。

    设计原则：
    - 工作节点接收扁平的块列表，通过单调递增的event_idx索引
    - 作业->请求ID的翻译由调度器端管理器处理（通过反向映射）
    - 工作节点永远不需要知道请求的身份

    这种设计简化了工作节点的实现，使其只关注数据传输，
    而请求级别的状态管理完全由调度器负责。
    """

    # Load event per step. INVALID_JOB_ID means no blocks to load this step.
    # 每步的加载事件。INVALID_JOB_ID表示本步没有需要加载的块。
    load_event: int = INVALID_JOB_ID
    load_gpu_blocks: list[int] = field(default_factory=list)  # 需要加载到GPU的块ID列表
    load_cpu_blocks: list[int] = field(default_factory=list)  # 从CPU加载的块ID列表
    # Reverse map: load_event->req_ids, for tracking requests with finished load events
    # 反向映射：load_event->req_ids，用于跟踪加载事件完成的请求
    load_event_to_reqs: dict[int, list[str]] = field(default_factory=dict)

    # Store event per step. INVALID_JOB_ID means no blocks to store this step.
    # 每步的存储事件。INVALID_JOB_ID表示本步没有需要存储的块。
    store_event: int = INVALID_JOB_ID
    store_gpu_blocks: list[int] = field(default_factory=list)  # 从GPU存储的块ID列表
    store_cpu_blocks: list[int] = field(default_factory=list)  # 存储到CPU的块ID列表

    # Whether any requests were preempted this step and need flush pending transfers.
    # 本步是否有请求被抢占，需要刷新待处理的传输
    need_flush: bool = False


@dataclass
class SimpleCPUOffloadWorkerMetadata(KVConnectorWorkerMetadata):
    """Worker -> Scheduler metadata for completed store events.

    工作节点->调度器的元数据，用于报告已完成的存储事件。

    完成事件报告机制：
    1. 每个工作节点报告 {event_idx: 1} 表示新完成的存储
    2. aggregate() 方法在步骤内跨工作节点求和计数
    3. 调度器端管理器跨步骤累积计数
    4. 只有当计数达到 world_size 时才处理存储完成

    这种机制确保在多工作节点（TP/PP）场景下，
    所有工作节点都完成传输后才认为存储完成。
    """

    completed_store_events: dict[int, int]  # 已完成的存储事件字典，键为event_idx，值为完成计数

    def aggregate(
        self, other: "KVConnectorWorkerMetadata"
    ) -> "KVConnectorWorkerMetadata":
        """聚合两个工作节点的元数据。

        将另一个工作节点的完成事件合并到当前元数据中。
        用于在步骤内跨工作节点聚合完成计数。

        参数：
        - other: 另一个工作节点的元数据

        返回：
        - 合并后的元数据
        """
        assert isinstance(other, SimpleCPUOffloadWorkerMetadata)
        merged = dict(self.completed_store_events)
        for k, v in other.completed_store_events.items():
            merged[k] = merged.get(k, 0) + v
        return SimpleCPUOffloadWorkerMetadata(completed_store_events=merged)
