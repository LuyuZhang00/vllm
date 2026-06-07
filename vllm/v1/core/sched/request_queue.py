# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
请求队列模块 (vllm/v1/core/sched/request_queue.py)

本模块实现了调度器使用的请求队列，支持不同的调度策略。

调度策略概述：
1. FCFS (First Come First Served): 先来先服务
   - 请求按到达顺序排队
   - 最简单、最公平的策略
   - 使用双端队列 (deque) 实现

2. PRIORITY: 优先级调度
   - 请求按优先级排序，高优先级请求先处理
   - 相同优先级的请求按到达时间排序
   - 使用堆 (heap) 实现

请求队列接口 (RequestQueue)：
- add_request: 添加请求到队列
- pop_request: 从队列头部弹出请求
- peek_request: 查看队列头部请求（不弹出）
- prepend_request: 将请求插入队列头部（用于抢占恢复）
- prepend_requests: 将另一个队列的所有请求插入头部
- remove_request: 从队列中移除特定请求
- remove_requests: 从队列中移除多个请求

使用场景：
- waiting_queue: 等待调度的请求队列（通常使用 FCFS 或优先级）
- 被抢占的请求通过 prepend_request/prepend_requests 重新加入等待队列
"""

import heapq
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Iterable, Iterator
from enum import Enum

from vllm.v1.request import Request


class SchedulingPolicy(Enum):
    """调度策略枚举。"""

    FCFS = "fcfs"        # 先来先服务
    PRIORITY = "priority"  # 优先级调度


class RequestQueue(ABC):
    """请求队列抽象基类。

    定义了请求队列必须实现的接口。
    不同的调度策略通过不同的具体实现来支持。
    """

    @abstractmethod
    def add_request(self, request: Request) -> None:
        """按策略向队列添加请求。"""
        pass

    @abstractmethod
    def pop_request(self) -> Request:
        """按策略从队列弹出请求。"""
        pass

    @abstractmethod
    def peek_request(self) -> Request:
        """查看队列头部的请求（不移除）。"""
        pass

    @abstractmethod
    def prepend_request(self, request: Request) -> None:
        """将请求插入队列头部。"""
        pass

    @abstractmethod
    def prepend_requests(self, requests: "RequestQueue") -> None:
        """将另一个队列的所有请求插入此队列头部。"""
        pass

    @abstractmethod
    def remove_request(self, request: Request) -> None:
        """从队列中移除特定请求。"""
        pass

    @abstractmethod
    def remove_requests(self, requests: Iterable[Request]) -> None:
        """从队列中移除多个特定请求。"""
        pass

    @abstractmethod
    def __bool__(self) -> bool:
        """检查队列是否有请求。"""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """获取队列中的请求数。"""
        pass

    @abstractmethod
    def __iter__(self) -> Iterator[Request]:
        """按策略顺序迭代队列。"""
        pass


class FCFSRequestQueue(deque[Request], RequestQueue):
    """先来先服务 (FCFS) 请求队列。

    基于双端队列 (deque) 实现，请求按到达顺序排列。
    最简单、最常用的调度策略。

    特点：
    - O(1) 的添加和弹出操作
    - 严格按到达顺序处理
    - 公平性好，但不支持优先级
    """

    def add_request(self, request: Request) -> None:
        """按 FCFS 策略将请求添加到队列尾部。"""
        self.append(request)

    def pop_request(self) -> Request:
        """按 FCFS 策略从队列头部弹出请求。"""
        return self.popleft()

    def peek_request(self) -> Request:
        """查看队列头部的请求（不移除）。"""
        if not self:
            raise IndexError("peek from an empty queue")
        return self[0]

    def prepend_request(self, request: Request) -> None:
        """将请求插入队列头部（用于抢占恢复）。"""
        self.appendleft(request)

    def prepend_requests(self, requests: RequestQueue) -> None:
        """将另一个队列的所有请求插入此队列头部。

        注意：请求将以其在 `requests` 队列中出现的逆序被插入。
        """
        self.extendleft(requests)

    def remove_request(self, request: Request) -> None:
        """从队列中移除特定请求。"""
        self.remove(request)

    def remove_requests(self, requests: Iterable[Request]) -> None:
        """从队列中移除多个特定请求。"""
        requests_to_remove = set(requests)
        filtered_requests = [req for req in self if req not in requests_to_remove]
        # deque 不支持原地过滤，需要清除后重新添加
        self.clear()
        self.extend(filtered_requests)

    def __bool__(self) -> bool:
        """检查队列是否有请求。"""
        return len(self) > 0

    def __len__(self) -> int:
        """获取队列中的请求数。"""
        return super().__len__()

    def __iter__(self) -> Iterator[Request]:
        """按 FCFS 策略迭代队列。"""
        return super().__iter__()


class PriorityRequestQueue(RequestQueue):
    """
    优先级请求队列。

    基于堆 (heap) 实现，请求按优先级排序。
    遵循 Request 类中定义的排序规则：
    - 优先级值较小的请求先处理
    - 相同优先级的请求按到达时间排序

    特点：
    - O(log n) 的添加和弹出操作
    - 支持优先级调度
    - 适用于需要差异化服务的场景
    """

    def __init__(self) -> None:
        # 最小堆
        self._heap: list[Request] = []

    def add_request(self, request: Request) -> None:
        """按优先级策略将请求添加到队列。"""
        heapq.heappush(self._heap, request)

    def pop_request(self) -> Request:
        """按优先级策略从队列弹出最高优先级请求。"""
        if not self._heap:
            raise IndexError("pop from empty heap")
        return heapq.heappop(self._heap)

    def peek_request(self) -> Request:
        """查看最高优先级的请求（不移除）。"""
        if not self._heap:
            raise IndexError("peek from empty heap")
        return self._heap[0]

    def prepend_request(self, request: Request) -> None:
        """将请求添加到队列。

        注意：在优先级队列中，没有"插入到头部"的概念。
        请求按 (priority, arrival_time) 排序。
        """
        self.add_request(request)

    def prepend_requests(self, requests: RequestQueue) -> None:
        """将另一个队列的所有请求添加到此队列。

        注意：在优先级队列中，没有"插入到头部"的概念。
        请求按 (priority, arrival_time) 排序。
        """
        for request in requests:
            self.add_request(request)

    def remove_request(self, request: Request) -> None:
        """从队列中移除特定请求。"""
        self._heap.remove(request)
        heapq.heapify(self._heap)

    def remove_requests(self, requests: Iterable[Request]) -> None:
        """从队列中移除多个特定请求。"""
        requests_to_remove = requests if isinstance(requests, set) else set(requests)
        self._heap = [r for r in self._heap if r not in requests_to_remove]
        heapq.heapify(self._heap)

    def __bool__(self) -> bool:
        """检查队列是否有请求。"""
        return bool(self._heap)

    def __len__(self) -> int:
        """获取队列中的请求数。"""
        return len(self._heap)

    def __iter__(self) -> Iterator[Request]:
        """按优先级顺序迭代队列（不修改原队列）。"""
        heap_copy = self._heap[:]
        while heap_copy:
            yield heapq.heappop(heap_copy)


def create_request_queue(policy: SchedulingPolicy) -> RequestQueue:
    """根据调度策略创建请求队列。

    Args:
        policy: 调度策略枚举值

    Returns:
        对应策略的 RequestQueue 实例

    Raises:
        ValueError: 未知的调度策略
    """
    if policy == SchedulingPolicy.PRIORITY:
        return PriorityRequestQueue()
    elif policy == SchedulingPolicy.FCFS:
        return FCFSRequestQueue()
    else:
        raise ValueError(f"Unknown scheduling policy: {policy}")
