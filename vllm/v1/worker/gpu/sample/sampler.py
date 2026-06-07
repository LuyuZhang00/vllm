# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
采样器模块 (Sampler Module)

本模块实现了 vLLM v1 引擎的核心采样器，负责从模型输出的 logits 中采样 token。

采样流程：
1. 获取模型输出的原始 logits
2. 应用各种采样参数和约束：
   a. Logit 偏置（允许的 token IDs、logit 偏置、最小 token 数）
   b. 重复/频率/存在惩罚
   c. 禁用词过滤
   d. 温度缩放
   e. Min-P 过滤
   f. Top-K / Top-P 筛选
3. 使用 Gumbel-Max 算法采样 token
4. 计算 log 概率（如果请求）

支持两种 logprobs 模式：
- raw_logprobs: 使用原始 logits 计算 logprob（在应用采样参数之前）
- processed_logprobs: 使用处理后的 logits 计算 logprob（在应用采样参数之后）
"""
import numpy as np
import torch

import vllm.envs as envs
from vllm.config.model import LogprobsMode
from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.metrics.logits import get_num_nans
from vllm.v1.worker.gpu.sample.bad_words import BadWordsState
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
from vllm.v1.worker.gpu.sample.logit_bias import LogitBiasState
from vllm.v1.worker.gpu.sample.logprob import (
    LogprobTokenIdsState,
    compute_topk_logprobs,
)
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.sample.penalties import PenaltiesState
from vllm.v1.worker.gpu.sample.states import NO_LOGPROBS, SamplingStates
from vllm.v1.worker.gpu.states import RequestState


class Sampler:
    """核心采样器类。

    管理所有采样相关的状态，并执行从 logits 到采样 token 的完整流程。

    组成部分：
    1. SamplingStates: 温度、top-k、top-p、min-p、随机种子等
    2. PenaltiesState: 重复/频率/存在惩罚
    3. LogitBiasState: 允许的 token IDs、logit 偏置、最小 token 数
    4. BadWordsState: 禁用词过滤
    5. LogprobTokenIdsState: 自定义 logprob token IDs
    """

    def __init__(
        self,
        max_num_reqs: int,
        vocab_size: int,
        device: torch.device,
        req_states: RequestState,
        logprobs_mode: LogprobsMode = "raw_logprobs",
        num_speculative_tokens: int = 1,
        use_fp64_gumbel: bool = False,
    ):
        """初始化采样器。

        Args:
            max_num_reqs: 最大请求数
            vocab_size: 词表大小
            device: 计算设备
            req_states: 请求状态管理器
            logprobs_mode: logprobs 计算模式
            num_speculative_tokens: 推测解码的 token 数量
            use_fp64_gumbel: 是否使用 FP64 精度的 Gumbel 噪声
        """
        if logprobs_mode not in ("processed_logprobs", "raw_logprobs"):
            raise NotImplementedError(f"Unsupported logprobs_mode: {logprobs_mode}")
        self.logprobs_mode = logprobs_mode
        self.compute_nans = envs.VLLM_COMPUTE_NANS_IN_LOGITS  # 默认为 False
        self.use_fp64_gumbel = use_fp64_gumbel

        # 初始化各个子状态管理器
        self.sampling_states = SamplingStates(max_num_reqs, vocab_size)
        self.penalties_state = PenaltiesState(req_states)
        self.logit_bias_state = LogitBiasState(max_num_reqs, device)
        self.bad_words_state = BadWordsState(req_states)
        self.logprob_token_ids_state = LogprobTokenIdsState(max_num_reqs, device)
        self.num_speculative_tokens = num_speculative_tokens

    def add_request(
        self, req_idx: int, prompt_len: int, sampling_params: SamplingParams
    ) -> None:
        """添加新请求的采样参数到各个子状态管理器。

        Args:
            req_idx: 请求在批次中的索引
            prompt_len: prompt 的长度
            sampling_params: 采样参数
        """
        self.sampling_states.add_request(req_idx, sampling_params)
        self.penalties_state.add_request(req_idx, sampling_params)
        self.logit_bias_state.add_request(req_idx, prompt_len, sampling_params)
        self.bad_words_state.add_request(req_idx, sampling_params)
        self.logprob_token_ids_state.add_request(req_idx, sampling_params)

    def apply_staged_writes(self) -> None:
        """将所有暂存的采样参数数据批量应用到 GPU。"""
        self.sampling_states.apply_staged_writes()
        self.penalties_state.apply_staged_writes()
        self.logit_bias_state.apply_staged_writes()
        self.bad_words_state.apply_staged_writes()
        self.logprob_token_ids_state.apply_staged_writes()

    def __call__(
        self,
        logits: torch.Tensor,
        input_batch: InputBatch,
    ) -> SamplerOutput:
        """执行采样并返回结果。

        流程：
        1. 可选：计算 NaN 数量（用于数值异常检测）
        2. 应用采样参数并采样 token
        3. 可选：计算 log 概率
        4. 返回采样结果

        Args:
            logits: 模型输出的 logits [num_tokens, vocab_size]
            input_batch: 输入批次数据

        Returns:
            SamplerOutput 对象，包含采样的 token IDs、logprobs 等
        """
        expanded_idx_mapping = input_batch.expanded_idx_mapping
        idx_mapping_np = input_batch.idx_mapping_np
        cu_num_logits_np = input_batch.cu_num_logits_np
        expanded_local_pos = input_batch.expanded_local_pos
        pos = input_batch.positions[input_batch.logits_indices]
        input_ids = input_batch.input_ids[input_batch.logits_indices]

        # 注意：我们有意在采样前计算 num_nans，以明确 num_nans 是在
        # 应用惩罚和温度之前计算的
        num_nans = get_num_nans(logits) if self.compute_nans else None
        sampled, processed_logits = self.sample(
            logits,
            expanded_idx_mapping,
            idx_mapping_np,
            pos,
            input_ids,
            expanded_local_pos,
        )

        # 计算 log 概率（如果请求）
        max_num_logprobs = self.sampling_states.max_num_logprobs(idx_mapping_np)
        max_per_req_token_ids = self.logprob_token_ids_state.max_num_token_ids(
            idx_mapping_np
        )
        if max_num_logprobs != NO_LOGPROBS or max_per_req_token_ids > 0:
            if self.logprobs_mode == "processed_logprobs":
                logits = processed_logits
            expanded_logits = logits.shape[0] != idx_mapping_np.shape[0]
            cu_num_logits = cu_num_logits_np.tolist() if expanded_logits else None
            num_logprobs = max_num_logprobs if max_num_logprobs != NO_LOGPROBS else 0
            logprobs_tensors = compute_topk_logprobs(
                logits,
                num_logprobs,
                sampled,
                cu_num_logits,
                logprob_token_ids_state=self.logprob_token_ids_state,
                expanded_idx_mapping=input_batch.expanded_idx_mapping,
                max_per_req_token_ids=max_per_req_token_ids,
            )
        else:
            logprobs_tensors = None

        # 这些都是 GPU 张量
        sampler_output = SamplerOutput(
            # 采样的 token 被扩展为 2D 张量，形状为 [num_requests, 1]，
            # 每行表示每个请求生成的一个 token
            sampled_token_ids=sampled.view(-1, 1),
            logprobs_tensors=logprobs_tensors,
            num_nans=num_nans,
            num_sampled=input_batch.seq_lens.new_ones(input_batch.num_reqs),
        )
        return sampler_output

    def apply_sampling_params(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
        pos: torch.Tensor,
        input_ids: torch.Tensor,
        expanded_local_pos: torch.Tensor,
    ) -> torch.Tensor:
        """按顺序应用所有采样参数到 logits。

        应用顺序很重要：
        1. Logit 偏置（允许的 token IDs、logit 偏置、最小 token 数）
        2. 重复/频率/存在惩罚
        3. 禁用词过滤
        4. 温度缩放
        5. Min-P 过滤
        6. Top-K / Top-P 筛选

        Args:
            logits: 原始 logits [num_tokens, vocab_size]
            expanded_idx_mapping: 扩展的请求索引映射
            idx_mapping_np: 请求索引映射（numpy）
            pos: 每个 token 的位置
            input_ids: 输入 token IDs
            expanded_local_pos: 扩展的本地位置

        Returns:
            处理后的 logits（可能是一个新的张量）
        """
        # 复制 logits 到新的 FP32 张量（避免修改原始数据）
        logits = torch.empty_like(logits, dtype=torch.float32).copy_(logits)

        # 1. 应用 logit 偏置（允许的 token IDs、logit 偏置、最小 token 数）
        self.logit_bias_state.apply_logit_bias(
            logits, expanded_idx_mapping, idx_mapping_np, pos
        )

        # 2. 应用惩罚（重复/频率/存在惩罚）
        self.penalties_state.apply_penalties(
            logits,
            expanded_idx_mapping,
            idx_mapping_np,
            input_ids,
            expanded_local_pos,
        )

        # 3. 应用禁用词过滤
        self.bad_words_state.apply_bad_words(
            logits,
            expanded_idx_mapping,
            idx_mapping_np,
            input_ids,
            expanded_local_pos,
        )

        # 4. 应用温度缩放
        self.sampling_states.apply_temperature(
            logits, expanded_idx_mapping, idx_mapping_np
        )

        # 5. 应用 min_p 过滤
        self.sampling_states.apply_min_p(logits, expanded_idx_mapping, idx_mapping_np)

        # 6. 应用 top_k 和/或 top_p。这可能会也可能不会返回新的张量
        return self.sampling_states.apply_top_k_top_p(
            logits, expanded_idx_mapping, idx_mapping_np
        )

    def sample(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
        pos: torch.Tensor,
        input_ids: torch.Tensor,
        expanded_local_pos: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """执行采样。

        1. 应用所有采样参数
        2. 使用 Gumbel-Max 算法采样 token

        注意：Gumbel 采样时 apply_temperature=False，因为温度已在 apply_sampling_params 中应用。

        Args:
            logits: 原始 logits
            expanded_idx_mapping: 扩展的请求索引映射
            idx_mapping_np: 请求索引映射（numpy）
            pos: 每个 token 的位置
            input_ids: 输入 token IDs
            expanded_local_pos: 扩展的本地位置

        Returns:
            (sampled, processed_logits): 采样的 token IDs 和处理后的 logits
        """
        processed_logits = self.apply_sampling_params(
            logits,
            expanded_idx_mapping,
            idx_mapping_np,
            pos,
            input_ids,
            expanded_local_pos,
        )

        # 使用 Gumbel-Max 算法采样下一个 token
        sampled = gumbel_sample(
            processed_logits,
            expanded_idx_mapping,
            self.sampling_states.temperature.gpu,
            self.sampling_states.seeds.gpu,
            pos,
            apply_temperature=False,
            use_fp64=self.use_fp64_gumbel,
        )
        return sampled, processed_logits
