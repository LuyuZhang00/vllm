# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Logits处理器状态管理模块。

本模块提供了两个核心类:

1. BatchUpdateBuilder:
    - 帮助跟踪持久化batch的状态变化
    - 构建用于logits处理器的BatchUpdate数据结构
    - 管理请求的添加、删除和移动操作

2. LogitsProcessors:
    - 封装已初始化的logits处理器对象
    - 将处理器分为两类: argmax不变的和非argmax不变的
    - 提供遍历所有处理器的迭代器

BatchUpdateBuilder的设计假设和保证:

假设:
    1. 所有关于从持久化batch中移除请求的信息在步骤开始时通过
       self.removed_append()调用聚合在self._removed中
    2. 在给定步骤中第一次读取self.removed、self.pop_removed()
       或self.peek_removed()之后，不再注册新的移除操作
    3. self._removed的元素永远不会被直接修改、添加或移除
       （修改仅通过self.removed_append()和self.pop_removed()进行）

保证（在上述假设下）:
    1. self.removed始终按降序排序
    2. self.pop_removed()和self.peek_removed()都返回
       当前步骤中最低的已移除请求索引
"""

from collections.abc import Iterable, Iterator
from itertools import chain
from typing import TYPE_CHECKING

from vllm.v1.sample.logits_processor.interface import (
    AddedRequest,
    BatchUpdate,
    MovedRequest,
    RemovedRequest,
)

if TYPE_CHECKING:
    from vllm.v1.sample.logits_processor.interface import LogitsProcessor


class BatchUpdateBuilder:
    """Batch更新构建器: 帮助跟踪持久化batch状态变化并构建BatchUpdate。

    该类在调度器中使用，用于收集每个解码步骤中batch的变化信息，
    然后传递给logits处理器。

    使用模式:
    1. 在步骤开始时，调用removed_append()注册所有被移除的请求
    2. 通过added和moved列表记录添加和移动操作
    3. 调用get_and_reset()生成BatchUpdate并重置内部状态

    属性:
        _removed: 被移除请求的索引列表（内部存储）
        _is_removed_sorted: _removed是否已排序
        added: 新添加请求的信息列表
        moved: 移动请求的信息列表
        batch_changed: 是否有batch变化（用于池化模式）
    """

    _removed: list[RemovedRequest]
    _is_removed_sorted: bool
    added: list[AddedRequest]
    moved: list[MovedRequest]

    def __init__(
        self,
        removed: list[RemovedRequest] | None = None,
        added: list[AddedRequest] | None = None,
        moved: list[MovedRequest] | None = None,
    ) -> None:
        """
        初始化Batch更新构建器。

        参数:
            removed: 初始的已移除请求列表
            added: 初始的已添加请求列表
            moved: 初始的移动请求列表
        """
        self._removed = removed or []
        self.added = added or []
        self.moved = moved or []
        self._is_removed_sorted = False

        # 用于跟踪池化模式下的变化（此时不填充added列表）
        self.batch_changed = False

    def _ensure_removed_sorted(self) -> None:
        """确保removed列表按降序排序。

        在给定步骤中第一次调用后变为幂等操作，直到reset()。
        """
        if not self._is_removed_sorted:
            self._removed.sort(reverse=True)
            self._is_removed_sorted = True

    @property
    def removed(self) -> list[RemovedRequest]:
        """按降序排列的已移除请求索引列表。

        首次访问时触发排序。

        返回:
            按降序排列的已移除请求索引列表
        """
        self._ensure_removed_sorted()
        return self._removed

    def removed_append(self, index: int) -> None:
        """注册从持久化batch中移除的请求。

        在self.removed、self.pop_removed()或self.peek_removed()被读取后
        不得调用此方法。

        参数:
            index: 请求索引

        异常:
            RuntimeError: 如果在removed被读取后调用
        """
        if self._is_removed_sorted:
            raise RuntimeError(
                "Cannot register new removed request after self.removed has been read."
            )
        self._removed.append(index)
        self.batch_changed = True

    def has_removed(self) -> bool:
        """检查是否有已移除的请求。

        返回:
            是否有已移除的请求
        """
        return bool(self._removed)

    def peek_removed(self) -> int | None:
        """查看最低的已移除请求索引（不移除）。

        返回:
            最低的已移除请求索引，如果没有则返回None
        """
        if self.has_removed():
            self._ensure_removed_sorted()
            return self._removed[-1]
        return None

    def pop_removed(self) -> int | None:
        """弹出最低的已移除请求索引。

        返回:
            最低的已移除请求索引，如果没有则返回None
        """
        if self.has_removed():
            self._ensure_removed_sorted()
            return self._removed.pop()
        return None

    def reset(self) -> bool:
        """重置构建器状态。

        返回:
            在重置之前是否有任何batch变化
        """
        self._is_removed_sorted = False
        self._removed.clear()
        self.added.clear()
        self.moved.clear()
        batch_changed = self.batch_changed
        self.batch_changed = False
        return batch_changed

    def get_and_reset(self, batch_size: int) -> BatchUpdate | None:
        """生成logits处理器的batch更新数据结构并重置内部状态。

        参数:
            batch_size: 当前持久化batch大小

        返回:
            冻结的logits处理器batch更新实例; 如果没有更新则返回None
        """
        # 重置排序逻辑
        self._is_removed_sorted = False
        self.batch_changed = False
        if not any((self._removed, self.moved, self.added)):
            # 无更新; 短路返回
            return None
        # 构建batch状态更新
        batch_update = BatchUpdate(
            batch_size=batch_size,
            removed=self._removed,
            moved=self.moved,
            added=self.added,
        )
        self._removed = []
        self.moved = []
        self.added = []
        return batch_update


class LogitsProcessors:
    """封装已初始化的logits处理器对象。

    将logits处理器分为两类:
    1. argmax_invariant: 不影响贪心采样结果的处理器（如min_p）
       - 仅在随机采样路径中应用（温度缩放后、Top-K/Top-P前）
    2. non_argmax_invariant: 可能影响贪心采样结果的处理器（如min_tokens、logit_bias）
       - 在所有采样路径中应用（温度缩放前）

    属性:
        argmax_invariant: argmax不变的处理器列表
        non_argmax_invariant: 非argmax不变的处理器列表
    """

    def __init__(self, logitsprocs: Iterable["LogitsProcessor"] | None = None) -> None:
        """
        初始化LogitsProcessors容器。

        参数:
            logitsprocs: logits处理器的可迭代对象
        """
        self.argmax_invariant: list[LogitsProcessor] = []
        self.non_argmax_invariant: list[LogitsProcessor] = []
        if logitsprocs:
            for logitproc in logitsprocs:
                (
                    self.argmax_invariant
                    if logitproc.is_argmax_invariant()
                    else self.non_argmax_invariant
                ).append(logitproc)

    @property
    def all(self) -> Iterator["LogitsProcessor"]:
        """遍历所有logits处理器的迭代器。

        先遍历argmax不变的处理器，再遍历非argmax不变的处理器。

        返回:
            所有logits处理器的迭代器
        """
        return chain(self.argmax_invariant, self.non_argmax_invariant)
