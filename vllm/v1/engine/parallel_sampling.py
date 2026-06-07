# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
并行采样（Parallel Sampling）模块。

本模块实现了 vLLM 的并行采样功能，允许一个用户请求同时生成 n 个不同的
补全结果（completions）。

核心概念：
1. 父请求（ParentRequest）：用户的原始请求，包含 n>1 的采样参数
2. 子请求（ChildRequest）：由父请求拆分出的 n 个独立请求，每个 n=1
3. 输出聚合：收集所有子请求的输出，合并后返回给用户

工作流程：
1. InputProcessor 收到 n>1 的请求，创建 ParentRequest
2. 为每个子请求生成唯一的 request_id 和独立的 sampling_params
3. 子请求独立调度和执行
4. 收集子请求输出，根据 output_kind 决定流式或一次性返回
5. 所有子请求完成后，记录统计信息

设计要点：
1. 种子管理：如果指定了 seed，每个子请求使用 seed+index 作为种子，
   确保可重现性；如果未指定 seed，所有子请求共享同一个 sampling_params 实例
2. 流式输出：支持 DELTA 模式（逐 token 流式返回）和 FINAL_ONLY 模式
   （所有子请求完成后一次性返回）
3. 统计信息：跟踪最大生成 token 数，用于性能监控
"""

from copy import copy
from typing import cast

from vllm.outputs import CompletionOutput
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.metrics.stats import IterationStats


class ParentRequest:
    """Info, state & processing for parallel sampling request.

    Store parent request ID and sampling params.
    Facilitate generating child request sampling params.
    """
    """
    并行采样请求的父请求类。

    管理并行采样请求的状态和处理逻辑：
    1. 存储父请求 ID 和采样参数
    2. 生成子请求的采样参数（包括种子管理）
    3. 跟踪子请求的完成状态
    4. 聚合子请求的输出结果
    5. 记录性能统计信息
    """

    # 父请求 ID（由 InputProcessor 生成，格式为原始请求 ID 的变体）
    request_id: str
    # 外部请求 ID（用户提供的原始请求 ID）
    external_req_id: str
    # 父请求的采样参数（n > 1）
    sampling_params: SamplingParams

    # To track the completion of child requests
    # 子请求 ID 集合，用于跟踪哪些子请求尚未完成
    # 当集合为空时，表示所有子请求都已完成
    child_requests: set[str]

    # To aggregate child completions when not streaming
    # 输出聚合器：当使用 FINAL_ONLY 模式时，按索引位置存储子请求的输出
    # 使用列表预分配空间，索引对应子请求的 index
    output_aggregator: list[CompletionOutput]

    # To find the max number of generated tokens across all children
    # 所有子请求中最大的生成 token 数量，用于统计报告
    max_num_generation_tokens: int

    # To efficiently obtain child sampling params
    # 缓存的子请求采样参数（仅在未指定 seed 时使用）
    # 所有子请求共享同一个实例以节省内存
    cached_child_sampling_params: SamplingParams | None

    def __init__(self, request: EngineCoreRequest) -> None:
        """
        初始化父请求。

        参数:
            request: 引擎核心请求对象，其 external_req_id 必须不为 None
        """
        assert request.external_req_id is not None
        sampling_params = request.params
        self.request_id = request.request_id
        self.external_req_id = request.external_req_id
        self.sampling_params = sampling_params

        self.child_requests = set()
        # 根据 output_kind 决定初始化方式：
        # - FINAL_ONLY: 预分配 n 个 None 位置，等待填充
        # - 其他（流式）: 使用空列表，直接追加输出
        self.output_aggregator = (
            [cast(CompletionOutput, None)] * sampling_params.n
            if (sampling_params.output_kind == RequestOutputKind.FINAL_ONLY)
            else []
        )
        self.max_num_generation_tokens = 0
        self.cached_child_sampling_params = None

    def _get_child_sampling_params(
        self,
        index: int,
    ) -> SamplingParams:
        """Efficiently obtain child `sampling_params`

        If `sampling_params.seed` is not `None` then
        each child request requires a unique clone of
        parent `sampling_params` with a unique seed.

        Args:
          index: index within `n` child requests

        Returns:
          Child `sampling_params` instance.
        """
        """
        高效获取子请求的采样参数。

        策略：
        1. 如果未指定 seed，所有子请求共享同一个采样参数实例（缓存复用）
        2. 如果指定了 seed，每个子请求需要独立的克隆，seed 为 seed+index

        参数:
            index: 子请求在 n 个请求中的索引（0 到 n-1）

        返回:
            子请求的 SamplingParams 实例
        """
        seed = self.sampling_params.seed
        if self.cached_child_sampling_params:
            # Reuse child sampling_params data structure
            # 复用缓存的采样参数（未指定 seed 时）
            return self.cached_child_sampling_params
        # Build child sampling_params
        # 构建子请求的采样参数（浅拷贝父请求参数）
        child_sampling_params = copy(self.sampling_params)
        # 子请求只生成 1 个结果
        child_sampling_params.n = 1
        if seed is None:
            # Cache child sampling_params for later reuse
            # 未指定 seed，缓存供后续子请求复用
            self.cached_child_sampling_params = child_sampling_params
        else:
            # Each child gets a clone with a unique seed
            # 指定了 seed，每个子请求使用不同的种子以确保多样性
            child_sampling_params.seed = seed + index
        return child_sampling_params

    def get_child_info(self, index: int) -> tuple[str, SamplingParams]:
        """Get child request ID and sampling params.

        Args:
          index: index within `n` child requests.

        Returns:
          (request ID, sampling_params) tuple
        """
        """
        获取子请求的 ID 和采样参数。

        子请求 ID 格式为 "{index}_{parent_request_id}"，
        确保全局唯一性。

        参数:
            index: 子请求在 n 个请求中的索引

        返回:
            (子请求ID, 采样参数) 的元组
        """
        child_req_id = f"{index}_{self.request_id}"
        self.child_requests.add(child_req_id)
        return child_req_id, self._get_child_sampling_params(index)

    @property
    def n(self) -> int:
        """返回并行采样数量。"""
        return self.sampling_params.n

    def get_outputs(
        self,
        child_request_id: str,
        completion_output: CompletionOutput,
    ) -> tuple[list[CompletionOutput], bool]:
        """
        处理子请求的输出并决定返回给客户端的内容。

        处理逻辑：
        1. 如果子请求已完成，从 child_requests 集合中移除
        2. 根据 output_kind 决定输出策略：
           - 流式模式（DELTA/DISABLED）：直接返回当前输出
           - FINAL_ONLY 模式：将输出存入聚合器，所有完成后一次性返回

        参数:
            child_request_id: 子请求 ID
            completion_output: 子请求的补全输出

        返回:
            (输出列表, 是否全部完成) 的元组：
            - 输出列表：要发送给客户端的 CompletionOutput 列表
            - 是否全部完成：True 表示所有子请求都已完成
        """
        already_finished_and_returned: bool = False
        if completion_output.finished():
            if child_request_id in self.child_requests:
                self.child_requests.remove(child_request_id)
            else:
                # child request ID is not available in child_requests
                # which means the request had finished in previous
                # batch step and returned to the client earlier
                # 子请求 ID 不在集合中，说明已在之前的批次步骤中
                # 完成并返回给客户端（流式模式下的已返回子请求）
                already_finished_and_returned = True

        if self.sampling_params.output_kind != RequestOutputKind.FINAL_ONLY:
            # If streaming, just return the current output
            #
            # DO NOT output finished and already returned child request to client again
            # 流式模式：直接返回当前输出
            # 注意：避免重复输出已完成且已返回的子请求
            outputs = [] if already_finished_and_returned else [completion_output]
        else:
            # If not streaming, aggregate the n final outputs.
            # 非流式模式：将输出存入聚合器对应位置
            self.output_aggregator[completion_output.index] = completion_output
            # 所有子请求都完成后才返回完整结果
            outputs = [] if self.child_requests else self.output_aggregator

        # 判断是否所有子请求都已完成
        finished = not self.child_requests
        return outputs, finished

    def observe_num_generation_tokens(self, num_generation_tokens: int):
        """
        观察并更新最大生成 token 数量。

        用于统计所有子请求中最长的生成长度，
        反映用户请求的整体生成规模。

        参数:
            num_generation_tokens: 当前子请求的生成 token 数量

        返回:
            更新后的最大生成 token 数量
        """
        self.max_num_generation_tokens = max(
            num_generation_tokens, self.max_num_generation_tokens
        )
        return self.max_num_generation_tokens

    @staticmethod
    def observe_finished_request(
        parent_req: "ParentRequest | None",
        iteration_stats: IterationStats,
        num_generation_tokens: int,
    ):
        """
        观察已完成的请求并记录统计信息。

        对于并行采样请求，只有当所有子请求都完成后才记录统计信息。
        这避免了单个子请求完成时产生不完整的统计数据。

        参数:
            parent_req: 父请求对象（None 表示普通单采样请求）
            iteration_stats: 迭代统计对象，用于记录统计数据
            num_generation_tokens: 生成的 token 数量
        """
        n_param = parent_req.n if parent_req is not None else 1

        if parent_req is not None:
            num_generation_tokens = parent_req.observe_num_generation_tokens(
                num_generation_tokens
            )

        # Child requests finished, we can now record to iteration stats
        # 只有当所有子请求都完成时（或无父请求时），才记录统计信息
        if parent_req is None or not parent_req.child_requests:
            iteration_stats.max_num_generation_tokens_iter.append(num_generation_tokens)
            iteration_stats.n_params_iter.append(n_param)
