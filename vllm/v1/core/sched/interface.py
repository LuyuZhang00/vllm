# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
调度器接口模块 (vllm/v1/core/sched/interface.py)

本模块定义了调度器的抽象接口和相关枚举。

接口设计原则：
1. SchedulerInterface 是所有调度器实现必须实现的抽象基类
2. 调度器的核心方法是 schedule()，每个调度步骤调用一次
3. 调度步骤对应模型的一次前向传播
4. 调度器通过 SchedulerOutput 向模型运行器传递调度决策

暂停状态 (PauseState)：
- UNPAUSED: 正常运行，所有类型的请求都可以调度
- PAUSED_NEW: 暂停新请求的调度，但继续运行已在运行中的请求
  用于：重置前缀缓存时，等待现有请求完成
- PAUSED_ALL: 暂停所有请求的调度
  用于：模型权重更新等需要完全停止推理的场景

接口方法分类：
1. 调度核心：schedule()、update_from_output()、update_draft_token_ids()
2. 请求管理：add_request()、finish_requests()、get_num_unfinished_requests()
3. 缓存管理：reset_prefix_cache()、reset_encoder_cache()
4. 状态查询：has_unfinished_requests()、has_finished_requests()、pause_state
5. 统计和监控：make_stats()、get_request_counts()
"""

import enum
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import TYPE_CHECKING

from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorBase_V1
    from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
    from vllm.v1.engine import EngineCoreOutputs
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.metrics.stats import SchedulerStats
    from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput
    from vllm.v1.request import Request, RequestStatus
    from vllm.v1.structured_output import StructuredOutputManager


class PauseState(enum.IntEnum):
    """调度器暂停状态。

    - UNPAUSED: 正常运行
    - PAUSE_NEW: 不调度新请求，但继续运行已在运行状态的请求
    - PAUSE_ALL: 不调度任何请求
    """

    UNPAUSED = 0
    PAUSED_NEW = 1
    PAUSED_ALL = 2


class SchedulerInterface(ABC):
    """调度器抽象接口。

    定义了调度器必须实现的所有方法。
    具体实现包括 Scheduler（标准调度器）和 AsyncScheduler（异步投机解码调度器）。
    """

    @abstractmethod
    def __init__(
        self,
        vllm_config: "VllmConfig",
        kv_cache_config: "KVCacheConfig",
        structured_output_manager: "StructuredOutputManager",
        block_size: int,
        hash_block_size: int,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        include_finished_set: bool = False,
        log_stats: bool = False,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def schedule(self) -> "SchedulerOutput":
        """调度此步骤要处理的请求。

        调度决策在迭代级别做出。每个调度步骤对应模型的一次前向传播。
        因此，此方法由引擎中的忙循环反复调用。

        本质上，调度器生成一个 {req_id: num_tokens} 字典，
        指定每个请求在此步骤中处理多少 token。例如：
        - num_tokens 可以等于新请求的 prompt token 数
        - 对于自回归生成的请求，num_tokens 为 1
        - 对于 chunked prefill、前缀缓存、投机解码等，num_tokens 介于两者之间

        此外，调度器还返回每个请求或整个批次的有用数据，
        模型运行器将使用这些信息准备模型输入。

        Returns:
            SchedulerOutput 对象，包含已调度请求的信息。
        """
        raise NotImplementedError

    @abstractmethod
    def get_grammar_bitmask(
        self, scheduler_output: "SchedulerOutput"
    ) -> "GrammarOutput | None":
        raise NotImplementedError

    @abstractmethod
    def update_from_output(
        self,
        scheduler_output: "SchedulerOutput",
        model_runner_output: "ModelRunnerOutput",
    ) -> dict[int, "EngineCoreOutputs"]:
        """根据模型运行器输出更新调度器状态。

        在模型运行器处理完已调度的请求后调用。
        输出包括生成的 token ID、下一步的草稿 token ID 等。
        调度器使用这些信息更新状态、检查已完成的请求并返回输出。

        Returns:
            客户端索引到 EngineCoreOutputs 的映射。
        """
        raise NotImplementedError

    @abstractmethod
    def update_draft_token_ids(self, draft_token_ids: "DraftTokenIds") -> None:
        """使用新生成的草稿 token ID 更新请求，应用结构化输出语法验证。

        Args:
            draft_token_ids: 每个请求的输入草稿 token ID。
        """
        raise NotImplementedError

    @abstractmethod
    def update_draft_token_ids_in_output(
        self, draft_token_ids: "DraftTokenIds", scheduler_output: "SchedulerOutput"
    ) -> None:
        """使用新生成的草稿 token ID 更新调度输出。

        Args:
            draft_token_ids: 每个请求的输入草稿 token ID。
            scheduler_output: 要更新的调度输出。
        """
        raise NotImplementedError

    @abstractmethod
    def add_request(self, request: "Request") -> None:
        """向调度器的内部队列添加新请求。

        Args:
            request: 要添加的新请求。
        """
        raise NotImplementedError

    @abstractmethod
    def finish_requests(
        self,
        request_ids: str | Iterable[str] | None,
        finished_status: "RequestStatus",
    ) -> list[tuple[str, int]]:
        """完成调度器内部队列中的请求。

        两种情况下调用：
        1. 客户端中止请求
        2. 前端进程在反 tokenize 生成的 token 后检测到停止字符串

        Args:
            request_ids: 单个或多个请求 ID，或 None 表示全部完成。
            finished_status: 给定请求的完成状态。

        Returns:
            被中止的请求的 (req_id, client_index) 元组列表。
        """
        raise NotImplementedError

    @abstractmethod
    def get_num_unfinished_requests(self) -> int:
        """调度器内部队列中未完成的请求数。"""
        raise NotImplementedError

    def has_unfinished_requests(self) -> bool:
        """如果调度器内部队列中有未完成的请求，返回 True。"""
        return self.get_num_unfinished_requests() > 0

    @abstractmethod
    def has_finished_requests(self) -> bool:
        """如果有需要清除的已完成请求，返回 True。

        注意：这与 `not self.has_unfinished_requests()` 不同。

        调度器维护上一步完成的请求的内部列表。
        此列表从下一次 schedule() 调用返回，发送给模型运行器
        以清除这些请求的缓存状态。

        此方法检查此内部列表是否非空。此信息对 DP attention 有用。
        """
        raise NotImplementedError

    def has_requests(self) -> bool:
        """如果有未完成的请求或尚未在 SchedulerOutput 中返回的已完成请求，
        返回 True。"""
        return self.has_unfinished_requests() or self.has_finished_requests()

    @property
    @abstractmethod
    def pause_state(self) -> PauseState:
        """调度器当前的暂停状态。"""
        raise NotImplementedError

    @abstractmethod
    def set_pause_state(self, pause_state: PauseState) -> None:
        raise NotImplementedError

    @abstractmethod
    def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        """重置 KV 缓存的前缀缓存。

        当模型权重实时更新时特别需要。

        Args:
            reset_running_requests: 如果为 True，所有运行中的请求将被抢占
                并移至等待队列。否则，仅在没有运行请求使用 KV 缓存时重置。
        """
        raise NotImplementedError

    @abstractmethod
    def reset_encoder_cache(self) -> None:
        """重置编码器缓存，使所有缓存的编码器输出失效。

        当模型权重更新时调用，确保不复用过时的视觉嵌入。
        """
        raise NotImplementedError

    @abstractmethod
    def get_request_counts(self) -> tuple[int, int]:
        """返回 (num_running_reqs, num_waiting_reqs)。"""
        raise NotImplementedError

    @abstractmethod
    def make_stats(self) -> "SchedulerStats | None":
        """创建用于日志记录的 SchedulerStats 对象。

        每个调度步骤创建一次。
        """
        raise NotImplementedError

    @abstractmethod
    def shutdown(self) -> None:
        """关闭调度器。"""
        raise NotImplementedError

    def get_kv_connector(self) -> "KVConnectorBase_V1 | None":
        return None
