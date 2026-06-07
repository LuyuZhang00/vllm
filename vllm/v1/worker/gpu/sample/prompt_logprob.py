# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Prompt Log 概率计算模块 (Prompt Log Probability Computation Module)

本模块实现了 prompt token 的 log 概率计算功能。与采样 token 的 logprob 不同，
prompt logprob 是计算 prompt 中每个 token 在给定前面 token 条件下的 log 概率。

应用场景：
- 评估 prompt 的"困惑度"（perplexity）
- 分析模型对 prompt 中每个 token 的置信度
- 用于 fine-tuning 数据的质量评估

处理流程：
1. 获取需要计算 logprob 的 prompt token IDs
2. 使用 logits_fn（包含 all-gather 等分布式操作）计算 logits
3. 计算 top-K log 概率
4. 处理分块 prefill 的情况：如果 prompt 被分块处理，
   需要将多个步骤的结果合并

特殊处理：
- 被抢占后恢复的请求：prompt logprob 在抢占前已计算，跳过
- 分块 prefill：prompt 可能分多步处理，需要累积结果
"""
from collections.abc import Callable

import numpy as np
import torch

from vllm.sampling_params import SamplingParams
from vllm.triton_utils import tl, triton
from vllm.v1.outputs import LogprobsTensors
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.sample.logprob import compute_topk_logprobs


class PromptLogprobsWorker:
    """Prompt log 概率计算的工作器。

    管理每个请求的 prompt logprob 状态，处理分块 prefill 的结果累积。
    """

    def __init__(self, max_num_reqs: int):
        """初始化 prompt logprob 工作器。

        Args:
            max_num_reqs: 最大请求数
        """
        self.max_num_reqs = max_num_reqs

        # 标记每个请求是否需要 prompt logprob
        self.uses_prompt_logprobs = np.zeros(self.max_num_reqs, dtype=bool)
        # 每个请求需要的 prompt logprob 数量
        self.num_prompt_logprobs = np.zeros(self.max_num_reqs, dtype=np.int32)
        # 正在处理中的 prompt logprob（用于分块 prefill 的结果累积）
        # req_id -> list of in-progress LogprobsTensors
        self.in_progress_prompt_logprobs: dict[str, list[LogprobsTensors]] = {}

    def add_request(self, req_id: str, req_idx: int, sampling_params: SamplingParams):
        """添加新请求的 prompt logprob 配置。

        Args:
            req_id: 请求 ID
            req_idx: 请求在批次中的索引
            sampling_params: 采样参数
        """
        uses_prompt_logprobs = sampling_params.prompt_logprobs is not None
        self.uses_prompt_logprobs[req_idx] = uses_prompt_logprobs
        self.num_prompt_logprobs[req_idx] = sampling_params.prompt_logprobs or 0
        if uses_prompt_logprobs:
            self.in_progress_prompt_logprobs[req_id] = []

    def remove_request(self, req_id: str) -> None:
        """移除请求的 prompt logprob 状态。

        Args:
            req_id: 请求 ID
        """
        self.in_progress_prompt_logprobs.pop(req_id, None)

    def compute_prompt_logprobs(
        self,
        logits_fn: Callable[[torch.Tensor], torch.Tensor],
        hidden_states: torch.Tensor,
        input_batch: InputBatch,
        # [max_num_reqs, max_model_len]
        all_token_ids: torch.Tensor,
        # [max_num_reqs]
        num_computed_tokens: torch.Tensor,
        # [max_num_reqs]
        prompt_lens: np.ndarray,
        # [max_num_reqs]
        prefill_lens: np.ndarray,
        # [max_num_reqs]
        num_computed_prefill_tokens: np.ndarray,
    ) -> dict[str, LogprobsTensors]:
        """计算 prompt token 的 log 概率。

        处理流程：
        1. 检查是否有请求需要 prompt logprob
        2. 获取需要计算 logprob 的 prompt token IDs
        3. 计算 logits（可能涉及 all-gather 等分布式操作）
        4. 计算 top-K log 概率
        5. 处理分块 prefill 的结果累积和合并

        Args:
            logits_fn: 计算 logits 的函数（可能包含 all-gather 等操作）
            hidden_states: 隐藏状态张量
            input_batch: 输入批次数据
            all_token_ids: 所有 token IDs [max_num_reqs, max_model_len]
            num_computed_tokens: 已计算的 token 数量
            prompt_lens: prompt 长度
            prefill_lens: prefill 长度
            num_computed_prefill_tokens: 已计算的 prefill token 数量

        Returns:
            请求 ID 到 LogprobsTensors 的字典
        """
        idx_mapping_np = input_batch.idx_mapping_np
        needs_prompt_logprobs = self.uses_prompt_logprobs[idx_mapping_np]
        if not np.any(needs_prompt_logprobs):
            # 常见情况：没有请求需要 prompt logprob
            return {}

        num_prompt_logprobs = self.num_prompt_logprobs[idx_mapping_np]
        prompt_lens = prompt_lens[idx_mapping_np]
        computed_prefill = num_computed_prefill_tokens[idx_mapping_np]
        includes_prompt = computed_prefill < prompt_lens
        # 注意：如果请求在被抢占后恢复，其 prompt logprob 在抢占前已计算，跳过
        resumed_after_prompt = prompt_lens < prefill_lens[idx_mapping_np]
        needs_prompt_logprobs &= includes_prompt & ~resumed_after_prompt
        if not np.any(needs_prompt_logprobs):
            return {}

        # 获取本批次中请求的最大 logprob 数量
        requested_num_prompt_logprobs = num_prompt_logprobs[needs_prompt_logprobs]
        max_num_prompt_logprobs = (
            -1
            if np.any(requested_num_prompt_logprobs == -1)
            else int(requested_num_prompt_logprobs.max())
        )

        # 获取需要计算 logprob 的 prompt token IDs
        prompt_logprobs_token_ids = get_prompt_logprobs_token_ids(
            input_batch.num_tokens,
            input_batch.query_start_loc,
            input_batch.idx_mapping,
            num_computed_tokens,
            all_token_ids,
        )
        # 分块计算 prompt logprob（避免内存溢出）
        prompt_token_ids, prompt_logprobs, prompt_ranks = (
            compute_prompt_logprobs_with_chunking(
                prompt_logprobs_token_ids,
                hidden_states[: input_batch.num_tokens],
                logits_fn,
                max_num_prompt_logprobs,
            )
        )

        # 检查 prompt 是否被分块处理
        pos_after_step = computed_prefill + input_batch.num_scheduled_tokens
        is_prompt_chunked = pos_after_step < prompt_lens

        query_start_loc_np = input_batch.query_start_loc_np
        prompt_logprobs_dict: dict[str, LogprobsTensors] = {}
        for i, req_id in enumerate(input_batch.req_ids):
            if not needs_prompt_logprobs[i]:
                continue

            req_is_prompt_chunked = is_prompt_chunked[i]
            req_num_prompt_logprobs = int(num_prompt_logprobs[i])
            start_idx = query_start_loc_np[i]
            end_idx = query_start_loc_np[i + 1]
            assert start_idx < end_idx, (
                f"start_idx ({start_idx}) >= end_idx ({end_idx})"
            )
            if not req_is_prompt_chunked:
                end_idx -= 1

            width = (
                prompt_logprobs.shape[1]
                if req_num_prompt_logprobs == -1
                else req_num_prompt_logprobs + 1
            )
            # 如果 start_idx >= end_idx，没有 logprob
            logprobs = (
                None
                if start_idx >= end_idx
                else LogprobsTensors(
                    logprob_token_ids=prompt_token_ids[start_idx:end_idx, :width],
                    logprobs=prompt_logprobs[start_idx:end_idx, :width],
                    selected_token_ranks=prompt_ranks[start_idx:end_idx],
                )
            )

            prompt_logprobs_list = self.in_progress_prompt_logprobs[req_id]
            if logprobs is not None and (req_is_prompt_chunked or prompt_logprobs_list):
                prompt_logprobs_list.append(logprobs)
            if req_is_prompt_chunked:
                # Prompt 被分块处理，暂不返回 logprob
                continue

            if prompt_logprobs_list:
                # 合并正在处理中的 logprob 结果
                logprobs = LogprobsTensors(
                    logprob_token_ids=torch.cat(
                        [x.logprob_token_ids for x in prompt_logprobs_list]
                    ),
                    logprobs=torch.cat([x.logprobs for x in prompt_logprobs_list]),
                    selected_token_ranks=torch.cat(
                        [x.selected_token_ranks for x in prompt_logprobs_list]
                    ),
                )
                prompt_logprobs_list.clear()

            if logprobs is None:
                continue

            prompt_logprobs_dict[req_id] = logprobs
        return prompt_logprobs_dict


@triton.jit
def _prompt_logprobs_token_ids_kernel(
    prompt_logprobs_token_ids_ptr,
    query_start_loc_ptr,
    idx_mapping_ptr,
    num_computed_tokens_ptr,
    all_token_ids_ptr,
    all_token_ids_stride,
    BLOCK_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + batch_idx)

    query_start = tl.load(query_start_loc_ptr + batch_idx)
    query_end = tl.load(query_start_loc_ptr + batch_idx + 1)
    query_len = query_end - query_start

    num_computed_tokens = tl.load(num_computed_tokens_ptr + req_state_idx)
    for i in range(0, query_len, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < query_len
        # NOTE(woosuk): We should shift the pos by one
        # because the logprob is computed for the next token.
        target_pos = num_computed_tokens + 1 + block
        token_ids = tl.load(
            all_token_ids_ptr + req_state_idx * all_token_ids_stride + target_pos,
            mask=mask,
        )
        tl.store(
            prompt_logprobs_token_ids_ptr + query_start + block, token_ids, mask=mask
        )


def get_prompt_logprobs_token_ids(
    num_tokens: int,
    query_start_loc: torch.Tensor,
    idx_mapping: torch.Tensor,
    num_computed_tokens: torch.Tensor,
    all_token_ids: torch.Tensor,
) -> torch.Tensor:
    token_ids = torch.empty(num_tokens, dtype=torch.int64, device=idx_mapping.device)
    num_reqs = idx_mapping.shape[0]
    _prompt_logprobs_token_ids_kernel[(num_reqs,)](
        token_ids,
        query_start_loc,
        idx_mapping,
        num_computed_tokens,
        all_token_ids,
        all_token_ids.stride(0),
        BLOCK_SIZE=1024,
    )
    return token_ids


def compute_prompt_logprobs_with_chunking(
    prompt_token_ids: torch.Tensor,
    prompt_hidden_states: torch.Tensor,
    logits_fn: Callable[[torch.Tensor], torch.Tensor],
    num_prompt_logprobs: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Since materializing the full prompt logits can take too much memory,
    # we compute it in chunks.
    CHUNK_SIZE = 1024
    token_ids = []
    logprobs = []
    ranks = []
    prompt_token_ids = prompt_token_ids.to(torch.int64)
    for start_idx in range(0, prompt_token_ids.shape[0], CHUNK_SIZE):
        end_idx = start_idx + CHUNK_SIZE
        # NOTE(woosuk): logits_fn can be slow because it involves all-gather.
        prompt_logits = logits_fn(prompt_hidden_states[start_idx:end_idx])
        requested_num_prompt_logprobs = (
            prompt_logits.shape[-1]
            if num_prompt_logprobs == -1
            else num_prompt_logprobs
        )
        prompt_logprobs = compute_topk_logprobs(
            prompt_logits,
            requested_num_prompt_logprobs,
            prompt_token_ids[start_idx:end_idx],
        )
        token_ids.append(prompt_logprobs.logprob_token_ids)
        logprobs.append(prompt_logprobs.logprobs)
        ranks.append(prompt_logprobs.selected_token_ranks)

    token_ids = torch.cat(token_ids, dim=0) if len(token_ids) > 1 else token_ids[0]
    logprobs = torch.cat(logprobs, dim=0) if len(logprobs) > 1 else logprobs[0]
    ranks = torch.cat(ranks, dim=0) if len(ranks) > 1 else ranks[0]
    return token_ids, logprobs, ranks
