# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
异步调度器模块 (vllm/v1/core/sched/async_scheduler.py)

本模块实现了支持投机解码 (Speculative Decoding) 的异步调度器。

异步调度器与标准调度器的区别：
1. 标准调度器：每一步调度生成 1 个 token，等待结果后再调度下一步
2. 异步调度器：每一步生成 1 个 token + num_spec_tokens 个投机 token，
   无需等待投机 token 的验证结果即可开始下一步调度

投机解码流程：
1. 调度器为每个请求生成 1 个真实 token + N 个占位符投机 token
2. 模型运行器验证投机 token 的正确性
3. 正确的投机 token 被接受，错误的被拒绝
4. 调度器根据验证结果更新请求状态

异步调度器的关键设计：
- num_output_placeholders: 跟踪每个请求尚未验证的输出 token 数
- spec_token_ids: 存储投机 token 的 ID（验证前为占位符 -1）
- 强制抢占处理：reset_prefix_cache 后丢弃过时的异步输出帧
- 缓存块更新：使用 num_output_placeholders 调整缓存位置
"""

from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request, RequestStatus

logger = init_logger(__name__)


class AsyncScheduler(Scheduler):
    """
    异步调度器，支持投机解码。

    继承标准 Scheduler，覆盖以下行为：
    1. _update_after_schedule: 在调度后预增 output_placeholders 和投机 token
    2. _update_request_with_output: 在收到输出后减少 output_placeholders 并更新缓存
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # 可复用的只读占位符列表（投机 token ID 填充为 -1）
        self._spec_token_placeholders: list[int] = [-1] * self.num_spec_tokens

    def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
        """
        调度后的状态更新。

        为每个已调度的请求：
        1. 检查是否有待处理的结构化输出
        2. 增加 output_placeholders（预占位：1 个真实 token + N 个投机 token）
        3. 设置投机 token 占位符（-1 值，实际 ID 在工作器中更新）
        """
        super()._update_after_schedule(scheduler_output)
        spec_decode_tokens = scheduler_output.scheduled_spec_decode_tokens
        for req_id in scheduler_output.num_scheduled_tokens:
            request = self.requests[req_id]
            if request.is_prefill_chunk:
                continue

            # 检查是否有待处理的结构化输出 token
            scheduler_output.pending_structured_output_tokens |= (
                request.use_structured_output and request.num_output_placeholders > 0
            )
            # 请求将在此调度步骤中生成 1 个新 token + num_spec_tokens 个投机 token
            cur_num_spec_tokens = len(spec_decode_tokens.get(req_id, ()))
            request.num_output_placeholders += 1 + cur_num_spec_tokens
            # 添加投机/草稿 token 占位符。
            # 实际的投机 token ID 将在工作器进程中更新。
            request.spec_token_ids = self._spec_token_placeholders

    def _update_request_with_output(
        self, request: Request, new_token_ids: list[int]
    ) -> tuple[list[int], bool]:
        """
        使用模型输出更新请求状态。

        特殊处理：
        1. 强制抢占丢弃：如果 async_tokens_to_discard > 0，丢弃一个过时的输出帧
        2. 标准更新：调用父类的 _update_request_with_output
        3. 更新 output_placeholders：减少已验证的 token 数
        4. 缓存更新：对运行中的请求更新 KV 缓存块

        Args:
            request: 要更新的请求
            new_token_ids: 模型生成的新 token ID 列表

        Returns:
            (new_token_ids, stopped) 元组
        """
        if request.async_tokens_to_discard > 0:
            # 请求在 reset_prefix_cache 中被强制抢占；
            # 每次调用丢弃一个过时的异步输出帧，直到计数器归零。
            request.async_tokens_to_discard -= 1
            return [], False

        status_before_update = request.status
        new_token_ids, stopped = super()._update_request_with_output(
            request, new_token_ids
        )

        # 更新输出占位符计数
        request.num_output_placeholders -= len(new_token_ids)
        assert request.num_output_placeholders >= 0

        # 缓存新 token。被抢占的请求应跳过。
        if status_before_update == RequestStatus.RUNNING:
            self.kv_cache_manager.cache_blocks(
                request, request.num_computed_tokens - request.num_output_placeholders
            )
        return new_token_ids, stopped
