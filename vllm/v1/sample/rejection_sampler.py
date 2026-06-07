# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""拒绝采样器模块，用于投机解码(speculative decoding)。

本模块实现了基于拒绝采样的投机解码算法，严格遵循论文
https://arxiv.org/abs/2211.17192 中描述的算法。

投机解码的核心思想:
    1. 使用一个较小的"草稿模型"(draft model)快速生成多个候选token
    2. 使用较大的"目标模型"(target model)并行验证这些候选token
    3. 通过拒绝采样算法，决定接受或拒绝每个候选token
    4. 被拒绝的token从调整后的概率分布中重新采样（恢复token）

该方法可以在不改变目标模型输出分布的前提下，显著加速推理过程。

主要组件:
    - RejectionSampler: 拒绝采样器主类
    - rejection_sample: 执行拒绝采样的核心函数
    - apply_sampling_constraints: 应用采样约束（温度、Top-K、Top-P）
    - sample_recovered_tokens: 采样恢复token
    - Triton kernels: 用于GPU加速的Triton内核
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.v1.outputs import LogprobsLists, LogprobsTensors, SamplerOutput
from vllm.v1.sample.logits_processor.builtin import MinTokensLogitsProcessor
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.ops.bad_words import apply_bad_words_with_drafts
from vllm.v1.sample.ops.penalties import apply_all_penalties
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
from vllm.v1.sample.sampler import Sampler
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.spec_decode.utils import unconditional_to_conditional_rates

if TYPE_CHECKING:
    from vllm.config.speculative import SpeculativeConfig

logger = init_logger(__name__)

# 占位token ID：用于标记被拒绝的token位置
PLACEHOLDER_TOKEN_ID: tl.constexpr = -1
# 贪心采样的温度值
GREEDY_TEMPERATURE: tl.constexpr = 0
# 每个请求在单步中允许的最大投机草稿token数量
# 该值足够大以处理典型用例
MAX_SPEC_LEN = 128


class RejectionSampler(nn.Module):
    """拒绝采样器：用于投机解码的token验证和采样。

    该实现严格遵循论文 https://arxiv.org/abs/2211.17192 中描述的算法。

    术语说明:
    - accepted tokens（接受的token）: 基于草稿概率和目标概率关系被接受的token
    - recovered tokens（恢复的token）: 从调整后的概率分布中采样的token
        （该分布由草稿和目标概率共同决定）
    - bonus tokens（奖励token）: 如果所有提议的token都被接受，在序列末尾添加的额外token
        bonus token仅从目标概率中采样
    - output tokens（输出token）: 最终生成的token
        output tokens = accepted tokens + recovered tokens + bonus tokens

    工作流程:
    1. 从目标模型获取logits
    2. 为bonus token单独采样
    3. 对目标logits应用处理器（惩罚、坏词排除等）
    4. 应用采样约束（温度、Top-K、Top-P）
    5. 执行拒绝采样（通过Triton内核并行处理）
    6. 收集logprobs（如果请求了）
    7. 返回最终的SamplerOutput

    属性:
        sampler: 基础采样器，用于bonus token采样
        is_processed_logprobs_mode: 是否使用处理后的logprobs模式
        is_logits_logprobs_mode: 是否使用logits模式
        synthetic_conditional_rates: 合成条件接受率（用于合成拒绝采样模式）
        synthetic_mode: 是否启用合成拒绝采样模式
    """

    def __init__(
        self,
        sampler: Sampler,
        spec_config: SpeculativeConfig | None = None,
        device: torch.device | None = None,
    ):
        """
        初始化拒绝采样器。

        参数:
            sampler: 基础采样器实例，用于bonus token采样
            spec_config: 投机解码配置，包含拒绝采样方法等参数
            device: 计算设备
        """
        super().__init__()
        self.sampler = sampler
        logprobs_mode = self.sampler.logprobs_mode
        self.is_processed_logprobs_mode = logprobs_mode.startswith("processed")
        self.is_logits_logprobs_mode = logprobs_mode.endswith("logits")

        # 合成条件接受率（用于合成拒绝采样模式）
        self.synthetic_conditional_rates: torch.Tensor | None = None
        if (
            spec_config is not None
            and spec_config.rejection_sample_method == "synthetic"
        ):
            assert spec_config.synthetic_acceptance_rates is not None
            self.synthetic_conditional_rates = torch.tensor(
                unconditional_to_conditional_rates(
                    spec_config.synthetic_acceptance_rates
                ),
                dtype=torch.float32,
                device=device,
            )
        self.synthetic_mode = self.synthetic_conditional_rates is not None

    def forward(
        self,
        metadata: SpecDecodeMetadata,
        # [num_tokens, vocab_size]
        draft_probs: torch.Tensor | None,
        # [num_tokens + batch_size, vocab_size]
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput:
        """执行拒绝采样前向传播。

        参数:
            metadata: 投机解码元数据，包含草稿token信息、索引等
            draft_probs: 草稿token的概率分布 [num_tokens, vocab_size]
                对于ngram投机解码可能为None
            logits: 目标模型的logits [num_tokens + batch_size, vocab_size]
                不同请求的logits被展平为单个张量
                注意: logits可以被原地更新以节省内存
            sampling_metadata: 采样元数据，包含温度、top-k/top-p等参数

        返回:
            SamplerOutput: 包含最终输出token IDs和logprobs（如果请求了）
        """
        assert metadata.max_spec_len <= MAX_SPEC_LEN

        # 获取bonus token和目标token的logits索引
        bonus_logits_indices = metadata.bonus_logits_indices
        target_logits_indices = metadata.target_logits_indices

        # 当使用张量索引（bonus_logits_indices）时，PyTorch会创建一个
        # 与原始logits张量具有独立存储的新张量。这意味着对bonus_logits的
        # 任何原地操作都不会影响原始logits张量。
        assert logits is not None

        # 为bonus token采样
        bonus_logits = logits[bonus_logits_indices]
        bonus_sampler_output = self.sampler(
            logits=bonus_logits,
            sampling_metadata=replace(
                sampling_metadata,
                max_num_logprobs=-1,  # 返回完整logprobs
            ),
            predict_bonus_token=True,
            # 覆盖logprobs模式为返回logits，因为后续需要用于计算接受token的logprobs
            logprobs_mode_override="processed_logits"
            if self.is_processed_logprobs_mode
            else "raw_logits",
        )
        bonus_token_ids = bonus_sampler_output.sampled_token_ids

        # 与bonus_logits类似，target_logits也是一个与原始logits张量
        # 具有独立存储的新张量。因此可以安全地原地更新target_logits。
        raw_target_logits = logits[target_logits_indices]
        # 使用float32精度处理target_logits
        raw_target_logits = raw_target_logits.to(torch.float32)
        target_logits = raw_target_logits
        if not self.is_processed_logprobs_mode:
            # 在应用处理器之前克隆raw_target_logits，以保留原始logits用于logprobs计算，
            # 因为apply_logits_processors会原地修改张量。
            target_logits = target_logits.clone()

        # 应用logits处理器（惩罚、坏词排除等）
        target_logits = self.apply_logits_processors(
            target_logits, sampling_metadata, metadata
        )

        # [num_tokens, vocab_size]
        # NOTE(woosuk): target_logits可能在apply_sampling_constraints函数中被原地更新。
        # 应用采样约束（温度、Top-K、Top-P）
        target_logits = apply_sampling_constraints(
            target_logits,
            metadata.cu_num_draft_tokens,
            sampling_metadata,
        )

        # 执行拒绝采样（核心逻辑）
        output_token_ids = rejection_sample(
            metadata.draft_token_ids,
            metadata.num_draft_tokens,
            metadata.max_spec_len,
            metadata.cu_num_draft_tokens,
            draft_probs,
            target_logits,
            bonus_token_ids,
            sampling_metadata,
            synthetic_mode=self.synthetic_mode,
            synthetic_conditional_rates=self.synthetic_conditional_rates,
        )

        # 收集logprobs（如果请求了）
        logprobs_tensors = None
        if sampling_metadata.max_num_logprobs is not None:
            logprobs_tensors = self._get_logprobs_tensors(
                sampling_metadata.max_num_logprobs,
                metadata,
                logits,
                target_logits if self.is_processed_logprobs_mode else raw_target_logits,
                bonus_sampler_output.logprobs_tensors.logprobs,
                output_token_ids,
            )

        return SamplerOutput(
            sampled_token_ids=output_token_ids,
            logprobs_tensors=logprobs_tensors,
        )

    def _get_logprobs_tensors(
        self,
        max_num_logprobs: int,
        metadata: SpecDecodeMetadata,
        logits: torch.Tensor,
        target_logits: torch.Tensor,
        bonus_logits: torch.Tensor,
        sampled_token_ids: torch.Tensor,
    ) -> LogprobsTensors:
        """获取采样token的logprobs张量。

        收集目标logits和bonus logits，计算接受token的logprobs和排名。

        参数:
            max_num_logprobs: 最大logprobs数量
            metadata: 投机解码元数据
            logits: 原始logits张量
            target_logits: 目标logits张量
            bonus_logits: bonus logits张量
            sampled_token_ids: 采样的token IDs

        返回:
            LogprobsTensors: 包含token IDs、logprobs和排名的张量
        """
        # 计算累积采样token数量（偏移量）
        cu_num_sampled_tokens = torch.zeros_like(metadata.cu_num_sampled_tokens)
        cu_num_sampled_tokens[1:] = metadata.cu_num_sampled_tokens[:-1]

        # 收集目标和bonus logits
        bonus_logits_indices = metadata.bonus_logits_indices
        target_logits_indices = metadata.target_logits_indices
        final_logits = torch.zeros_like(logits, dtype=torch.float32)
        final_logits[target_logits_indices] = target_logits.to(torch.float32)
        final_logits[bonus_logits_indices] = bonus_logits.to(torch.float32)

        # NOTE: 为了避免CPU-GPU同步，我们现在简单地为所有草稿token计算索引，
        # 包括被拒绝的token。被拒绝的token将在parse_output中被过滤掉。
        logit_start_indices = cu_num_sampled_tokens
        offsets = torch.arange(
            sampled_token_ids.shape[-1],
            device=logit_start_indices.device,
            dtype=logit_start_indices.dtype,
        )
        accepted_logit_indices = (
            logit_start_indices.unsqueeze(1) + offsets.unsqueeze(0)
        ).flatten()
        accepted_logit_indices.clamp_(max=final_logits.shape[0] - 1)
        accepted_tokens = sampled_token_ids.clone().flatten()
        # 将被拒绝的token ID（PLACEHOLDER_TOKEN_ID）替换为0，避免gather_logprobs错误
        accepted_tokens[accepted_tokens == PLACEHOLDER_TOKEN_ID] = 0

        # 计算接受token的logprobs
        accepted_logits = final_logits[accepted_logit_indices]
        accepted_logprobs = (
            accepted_logits
            if self.is_logits_logprobs_mode
            else self.sampler.compute_logprobs(accepted_logits)
        )
        return self.sampler.gather_logprobs(
            accepted_logprobs,
            max_num_logprobs,
            accepted_tokens.to(torch.int64),
        )

    @staticmethod
    def parse_output(
        output_token_ids: torch.Tensor,
        vocab_size: int,
        discard_req_indices: Sequence[int] = (),
        logprobs_tensors: LogprobsTensors | None = None,
    ) -> tuple[list[list[int]], LogprobsLists | None]:
        """解析拒绝采样器的输出。

        将拒绝采样器输出的张量转换为token ID列表，并过滤掉被拒绝的token。

        参数:
            output_token_ids: 采样的token IDs，形状为 [batch_size, max_spec_len + 1]
                被拒绝的token被替换为 PLACEHOLDER_TOKEN_ID，将在此函数中被过滤
            vocab_size: 词表大小
            discard_req_indices: 可选的要丢弃token的行索引
            logprobs_tensors: 可选的logprobs张量，需要同步过滤

        返回:
            (token ID列表列表, logprobs列表或None)
        """
        output_token_ids_np = output_token_ids.cpu().numpy()
        # 创建有效token掩码
        valid_mask = (output_token_ids_np != PLACEHOLDER_TOKEN_ID) & (
            output_token_ids_np < vocab_size
        )
        output_logprobs = None
        if logprobs_tensors is not None:
            cu_num_tokens = [0] + valid_mask.sum(axis=1).cumsum().tolist()
            filtered_tensors = logprobs_tensors.filter(valid_mask.flatten())
            output_logprobs = filtered_tensors.tolists(cu_num_tokens)

        if len(discard_req_indices) > 0:
            valid_mask[discard_req_indices] = False
        outputs = [
            row[valid_mask[i]].tolist() for i, row in enumerate(output_token_ids_np)
        ]
        return outputs, output_logprobs

    def apply_logits_processors(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        metadata: SpecDecodeMetadata,
    ) -> torch.Tensor:
        """应用logits处理器到目标logits。

        包括惩罚、坏词排除、允许token白名单、最少token数限制和思考预算处理。

        参数:
            logits: 目标logits张量 [num_tokens, vocab_size]
            sampling_metadata: 采样元数据
            metadata: 投机解码元数据

        返回:
            处理后的logits张量（可能原地修改）
        """
        has_penalties = not sampling_metadata.no_penalties
        any_penalties_or_bad_words = (
            sampling_metadata.bad_words_token_ids or has_penalties
        )
        holder = sampling_metadata.thinking_budget_state_holder
        needs_thinking = holder is not None and holder.has_tracked_requests()

        output_token_ids = sampling_metadata.output_token_ids
        if any_penalties_or_bad_words or needs_thinking:
            # 将基础输出与投机token组合
            output_token_ids = self._combine_outputs_with_spec_tokens(
                output_token_ids,
                sampling_metadata.spec_token_ids,
            )

        # 计算目标logits的重复索引
        repeat_indices: torch.Tensor | None = None
        need_repeat_indices = (
            sampling_metadata.allowed_token_ids_mask is not None
            or has_penalties
            or needs_thinking
        )
        if need_repeat_indices:
            num_requests = len(metadata.num_draft_tokens)
            num_draft_tokens = torch.tensor(metadata.num_draft_tokens, device="cpu")
            original_indices = torch.arange(num_requests, device="cpu")
            # 将batch索引扩展为token级别索引
            # 例如: batch中有3个请求，分别有2、3、1个草稿token
            # 则repeat_indices = [0, 0, 1, 1, 1, 2]
            repeat_indices_cpu = original_indices.repeat_interleave(num_draft_tokens)
            repeat_indices = repeat_indices_cpu.to(
                device=logits.device, non_blocking=True
            )
            # 应用惩罚
            logits = self.apply_penalties(
                logits, sampling_metadata, metadata, repeat_indices, output_token_ids
            )

            # 应用允许的token ID白名单
            if sampling_metadata.allowed_token_ids_mask is not None:
                token_mask = sampling_metadata.allowed_token_ids_mask[repeat_indices]
                logits.masked_fill_(token_mask, float("-inf"))

        # 应用坏词排除
        if bad_words_token_ids := sampling_metadata.bad_words_token_ids:
            apply_bad_words_with_drafts(
                logits, bad_words_token_ids, output_token_ids, metadata.num_draft_tokens
            )

        # 应用非argmax不变的logits处理器（仅最少token数处理器支持投机解码）
        for processor in sampling_metadata.logitsprocs.non_argmax_invariant:
            if isinstance(processor, MinTokensLogitsProcessor):
                logits = processor.apply_with_spec_decode(
                    logits, metadata.num_draft_tokens
                )
        # 应用思考预算处理
        if holder is not None and holder.has_tracked_requests():
            logits = holder.apply_to_logits(
                logits,
                predict_bonus_token=False,
                spec_token_ids=sampling_metadata.spec_token_ids,
            )
        return logits

    @staticmethod
    def apply_penalties(
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        metadata: SpecDecodeMetadata,
        repeat_indices: torch.Tensor,
        output_token_ids: list[list[int]],
    ) -> torch.Tensor:
        """应用采样惩罚到目标logits。

        根据repeat_indices将batch级别的惩罚参数扩展到token级别。

        参数:
            logits: 目标logits张量 [num_tokens, vocab_size]
            sampling_metadata: 采样元数据
            metadata: 投机解码元数据
            repeat_indices: batch到token的索引映射
            output_token_ids: 每个请求的输出token ID列表

        返回:
            应用惩罚后的logits张量
        """
        if sampling_metadata.no_penalties:
            return logits

        assert sampling_metadata.prompt_token_ids is not None

        # 将batch级别的参数扩展到token级别
        prompt_token_ids = sampling_metadata.prompt_token_ids[repeat_indices]
        presence_penalties = sampling_metadata.presence_penalties[repeat_indices]
        frequency_penalties = sampling_metadata.frequency_penalties[repeat_indices]
        repetition_penalties = sampling_metadata.repetition_penalties[repeat_indices]

        logits = apply_all_penalties(
            logits,
            prompt_token_ids,
            presence_penalties,
            frequency_penalties,
            repetition_penalties,
            output_token_ids,
        )
        return logits

    @staticmethod
    def _combine_outputs_with_spec_tokens(
        output_token_ids: list[list[int]],
        spec_token_ids: list[list[int]] | None = None,
    ) -> list[list[int]]:
        """将基础输出token与投机解码token组合（投机解码版本）。

        与Sampler版本不同，此版本为每个投机token位置创建一个独立的输出序列。

        参数:
            output_token_ids: 基础输出token ID列表
            spec_token_ids: 投机token ID列表

        返回:
            组合后的输出token ID列表
        """
        if spec_token_ids is None:
            return output_token_ids

        result = []
        for out, spec in zip(output_token_ids, spec_token_ids):
            if len(spec) == 0:
                continue
            result.append(out)
            for i in range(len(spec) - 1):
                result.append([*result[-1], spec[i]])
        return result


def rejection_sample(
    # [num_tokens]
    draft_token_ids: torch.Tensor,
    # [batch_size]
    num_draft_tokens: list[int],
    max_spec_len: int,
    # [batch_size]
    cu_num_draft_tokens: torch.Tensor,
    # [num_tokens, vocab_size]
    draft_probs: torch.Tensor | None,
    # [num_tokens, vocab_size]
    target_logits: torch.Tensor,
    # [batch_size, 1]
    bonus_token_ids: torch.Tensor,
    sampling_metadata: SamplingMetadata,
    synthetic_mode: bool = False,
    synthetic_conditional_rates: torch.Tensor | None = None,
) -> torch.Tensor:
    """执行拒绝采样。

    这是拒绝采样的核心函数，根据草稿token和目标模型的logits，
    决定接受或拒绝每个草稿token，被拒绝的token从恢复分布中重新采样。

    拒绝采样算法:
    - 对于贪心采样: 如果草稿token的目标argmax与草稿token相同，则接受
    - 对于随机采样: 生成均匀随机数u，如果 target_prob/draft_prob >= u，则接受
    - 被拒绝的token从恢复分布中采样: q(x) = max(target_prob(x) - draft_prob(x), 0)

    参数:
        draft_token_ids: 草稿token IDs [num_tokens]
        num_draft_tokens: 每个请求的草稿token数量列表
        max_spec_len: 最大投机长度
        cu_num_draft_tokens: 草稿token的累积数量 [batch_size]
        draft_probs: 草稿token的概率分布 [num_tokens, vocab_size]，可能为None
        target_logits: 目标模型的logits [num_tokens, vocab_size]
        bonus_token_ids: bonus token IDs [batch_size, 1]
        sampling_metadata: 采样元数据
        synthetic_mode: 是否使用合成拒绝采样模式
        synthetic_conditional_rates: 合成条件接受率

    返回:
        output_token_ids: 输出token IDs [batch_size, max_spec_len + 1]
            被拒绝的token位置填充PLACEHOLDER_TOKEN_ID
    """
    assert draft_token_ids.ndim == 1
    assert draft_probs is None or draft_probs.ndim == 2
    assert cu_num_draft_tokens.ndim == 1
    assert target_logits.ndim == 2

    batch_size = len(num_draft_tokens)
    num_tokens = draft_token_ids.shape[0]
    vocab_size = target_logits.shape[-1]
    device = target_logits.device
    assert draft_token_ids.is_contiguous()
    assert draft_probs is None or draft_probs.is_contiguous()
    assert bonus_token_ids.is_contiguous()
    assert target_logits.shape == (num_tokens, vocab_size)

    # 创建输出缓冲区，用PLACEHOLDER_TOKEN_ID填充
    output_token_ids = torch.full(
        (batch_size, max_spec_len + 1),
        PLACEHOLDER_TOKEN_ID,
        dtype=torch.int32,  # 与SamplerOutput.sampled_token_ids保持一致
        device=device,
    )

    if sampling_metadata.all_greedy:
        is_greedy = None
    else:
        is_greedy = sampling_metadata.temperature == GREEDY_TEMPERATURE

    # 在任何内核之前生成均匀随机概率，因为合成模式在贪心内核中也需要它们。
    # 仅当所有请求都是贪心且合成模式关闭时跳过（标准快速路径）。
    # [num_tokens]
    uniform_probs: torch.Tensor | None = None
    if synthetic_mode or not sampling_metadata.all_greedy:
        uniform_probs = generate_uniform_probs(
            num_tokens,
            num_draft_tokens,
            sampling_metadata.generators,
            device,
        )

    if not sampling_metadata.all_random:
        # 贪心采样请求的拒绝采样
        target_argmax = target_logits.argmax(dim=-1)
        rejection_greedy_sample_kernel[(batch_size,)](
            output_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            target_argmax,
            bonus_token_ids,
            is_greedy,
            max_spec_len,
            uniform_probs,
            synthetic_conditional_rates,
            SYNTHETIC_MODE=synthetic_mode,
        )
        if sampling_metadata.all_greedy:
            return output_token_ids

    # 从目标logits计算概率分布
    target_probs = target_logits.softmax(dim=-1, dtype=torch.float32)
    assert target_probs.is_contiguous()

    # 为每个位置采样恢复token
    # [num_tokens]
    recovered_token_ids = sample_recovered_tokens(
        max_spec_len,
        num_draft_tokens,
        cu_num_draft_tokens,
        draft_token_ids,
        draft_probs,
        target_probs,
        sampling_metadata,
        device,
    )

    # 随机采样请求的拒绝采样
    assert uniform_probs is not None
    rejection_random_sample_kernel[(batch_size,)](
        output_token_ids,
        cu_num_draft_tokens,
        draft_token_ids,
        draft_probs,
        target_probs,
        bonus_token_ids,
        recovered_token_ids,
        uniform_probs,
        is_greedy,
        max_spec_len,
        vocab_size,
        synthetic_conditional_rates,
        NO_DRAFT_PROBS=draft_probs is None,
        SYNTHETIC_MODE=synthetic_mode,
    )
    return output_token_ids


def apply_sampling_constraints(
    logits: torch.Tensor,  # [num_tokens, vocab_size]
    cu_num_draft_tokens: torch.Tensor,  # [batch_size]
    sampling_metadata: SamplingMetadata,
) -> torch.Tensor:
    """应用采样约束到logits。

    该函数对logits应用温度缩放，以及Top-K和Top-P截断。
    对于贪心解码，直接返回原始logits。

    参数:
        logits: 输入logits张量 [num_tokens, vocab_size]
        cu_num_draft_tokens: 草稿token的累积数量 [batch_size]
        sampling_metadata: 包含采样参数（如温度、是否贪心采样等）的元数据

    返回:
        处理后的logits（非贪心采样时），或原始logits（贪心采样时）
    """
    assert logits.ndim == 2
    assert cu_num_draft_tokens.ndim == 1
    if sampling_metadata.all_greedy:
        return logits

    num_tokens = logits.shape[0]
    # 将batch级别的温度扩展到token级别
    temperature = expand_batch_to_tokens(
        sampling_metadata.temperature,
        cu_num_draft_tokens,
        num_tokens,
        replace_from=GREEDY_TEMPERATURE,
        replace_to=1,  # 贪心请求的temperature=0替换为1（不缩放）
    )
    # NOTE(woosuk): 原地更新logits以避免分配新张量
    logits.div_(temperature.unsqueeze(-1))

    # 获取扩展后的top_k和top_p张量
    top_k = None
    if sampling_metadata.top_k is not None:
        top_k = expand_batch_to_tokens(
            sampling_metadata.top_k,
            cu_num_draft_tokens,
            num_tokens,
        )
    top_p = None
    if sampling_metadata.top_p is not None:
        top_p = expand_batch_to_tokens(
            sampling_metadata.top_p,
            cu_num_draft_tokens,
            num_tokens,
        )

    # NOTE(woosuk): apply_top_k_top_p使用排序来计算掩码，
    # 对于大词表大小可能会比较慢。这可能导致性能问题。
    return apply_top_k_top_p(logits, top_k, top_p)


def expand_batch_to_tokens(
    x: torch.Tensor,  # [batch_size]
    cu_num_tokens: torch.Tensor,  # [batch_size]
    num_tokens: int,
    replace_from: int = 0,
    replace_to: int = 0,
) -> torch.Tensor:
    """将 [batch_size] 张量扩展为 [num_tokens] 张量。

    根据cu_num_tokens中每个batch的token数量进行扩展。
    例如，如果 x = [a, b, c] 且 cu_num_tokens = [2, 5, 6]，
    则 num_tokens = 6，expanded_x = [a, a, b, b, b, c]

    参数:
        x: [batch_size] 张量，需要扩展
        cu_num_tokens: [batch_size] 张量，包含每个batch的累积token数量
            每个元素表示到该batch为止的总token数
        num_tokens: 总token数量
        replace_from: 需要替换的值（默认0）
        replace_to: 替换后的值（默认0）

    返回:
        expanded_x: [num_tokens] 张量
    """
    batch_size = x.shape[0]
    assert cu_num_tokens.shape[0] == batch_size
    expanded_x = x.new_empty(num_tokens)
    expand_kernel[(batch_size,)](
        expanded_x,
        x,
        cu_num_tokens,
        replace_from,
        replace_to,
        MAX_NUM_TOKENS=MAX_SPEC_LEN,  # 避免重编译
    )
    return expanded_x


def generate_uniform_probs(
    num_tokens: int,
    num_draft_tokens: list[int],
    generators: dict[int, torch.Generator],
    device: torch.device,
) -> torch.Tensor:
    """生成一批均匀随机样本。

    创建一个形状为 (num_tokens,) 的张量，填充 [0, 1) 范围内的均匀随机值。
    如果提供了generators，具有自己种子的请求将使用提供的 torch.Generator
    以确保可复现性。其他请求的样本将在没有种子的情况下生成。

    参数:
        num_tokens: 总token数量
        num_draft_tokens: 每个请求的草稿token数量列表
        generators: batch索引到 torch.Generator 对象的映射
        device: 分配张量的设备

    返回:
        uniform_rand: 形状为 (num_tokens,) 的张量，包含 [0, 1) 范围内的均匀随机值
    """
    # NOTE(woosuk): 我们故意使用float64而不是float32，因为使用float32时，
    # 有不可忽略的概率uniform_prob被采样为精确的0.0，如
    # https://github.com/pytorch/pytorch/issues/16706 中报告的。
    # 使用float64可以缓解此问题。
    uniform_probs = torch.rand(
        (num_tokens,),
        dtype=torch.float64,
        device=device,
    )
    start_idx = 0
    for req_idx, n in enumerate(num_draft_tokens):
        # 不为没有草稿token的请求生成随机数。
        # 这对可复现性很重要。
        if n == 0:
            continue
        end_idx = start_idx + n
        generator = generators.get(req_idx)
        if generator is not None:
            uniform_probs[start_idx:end_idx].uniform_(generator=generator)
        start_idx = end_idx
    return uniform_probs


def sample_recovered_tokens(
    max_spec_len: int,
    num_draft_tokens: list[int],
    # [batch_size]
    cu_num_draft_tokens: torch.Tensor,
    # [num_tokens]
    draft_token_ids: torch.Tensor,
    # [num_tokens, vocab_size]
    draft_probs: torch.Tensor | None,
    # [num_tokens, vocab_size]
    target_probs: torch.Tensor,
    sampling_metadata: SamplingMetadata,
    device: torch.device,
) -> torch.Tensor:
    """采样恢复token。

    当草稿token被拒绝时，需要从恢复分布中采样一个新的token。
    恢复分布定义为: q(x) = max(target_prob(x) - draft_prob(x), 0)

    使用逆变换采样方法: 对于每个请求生成一个指数分布的随机数，
    然后选择 prob * (1/q) 最大的token作为恢复token。

    参数:
        max_spec_len: 最大投机长度
        num_draft_tokens: 每个请求的草稿token数量
        cu_num_draft_tokens: 草稿token的累积数量 [batch_size]
        draft_token_ids: 草稿token IDs [num_tokens]
        draft_probs: 草稿概率分布 [num_tokens, vocab_size]，可能为None
        target_probs: 目标概率分布 [num_tokens, vocab_size]
        sampling_metadata: 采样元数据
        device: 计算设备

    返回:
        recovered_token_ids: 恢复token IDs [num_tokens]
    """
    # NOTE(woosuk): 为每个请求只创建一个分布
    batch_size = len(num_draft_tokens)
    vocab_size = target_probs.shape[-1]
    q = torch.empty(
        (batch_size, vocab_size),
        dtype=torch.float32,
        device=device,
    )
    q.exponential_()
    for i, generator in sampling_metadata.generators.items():
        # 不为没有草稿token的请求生成随机数
        # 这对可复现性很重要
        if num_draft_tokens[i] > 0:
            q[i].exponential_(generator=generator)

    inv_q = q.reciprocal()

    recovered_token_ids = torch.empty_like(draft_token_ids)
    BLOCK_SIZE = 8192
    sample_recovered_tokens_kernel[(batch_size, max_spec_len)](
        recovered_token_ids,
        cu_num_draft_tokens,
        draft_token_ids,
        draft_probs,
        target_probs,
        inv_q,
        vocab_size,
        BLOCK_SIZE,
        NO_DRAFT_PROBS=draft_probs is None,
    )
    return recovered_token_ids


# NOTE(woosuk): 避免特化以防止不必要的重编译
@triton.jit(do_not_specialize=["max_spec_len"])
def rejection_greedy_sample_kernel(
    output_token_ids_ptr,  # [batch_size, max_spec_len + 1]
    cu_num_draft_tokens_ptr,  # [batch_size]
    draft_token_ids_ptr,  # [num_tokens]
    target_argmax_ptr,  # [num_tokens]
    bonus_token_ids_ptr,  # [batch_size]
    is_greedy_ptr,  # [batch_size] 或 None
    max_spec_len,
    uniform_probs_ptr,  # [num_tokens] 或 None（仅合成模式）
    synthetic_conditional_rates_ptr,  # [num_speculative_tokens] 或 None
    SYNTHETIC_MODE: tl.constexpr,
):
    """贪心采样的拒绝采样Triton内核。

    对于每个请求（由program_id标识），逐个检查草稿token:
    - 标准模式: 如果草稿token == 目标argmax，则接受；否则拒绝
    - 合成模式: 使用均匀随机数和条件接受率决定是否接受

    如果所有token都被接受，在末尾添加bonus token。
    """
    req_idx = tl.program_id(0)
    # FIXME(woosuk): 因为is_greedy_ptr在profiling运行时不为None，
    # 运行时当is_greedy_ptr为None时可能会发生重编译。
    is_greedy = True if is_greedy_ptr is None else tl.load(is_greedy_ptr + req_idx)
    if not is_greedy:
        # 非贪心采样请求提前退出
        return

    start_idx = 0 if req_idx == 0 else tl.load(cu_num_draft_tokens_ptr + req_idx - 1)
    end_idx = tl.load(cu_num_draft_tokens_ptr + req_idx)
    num_draft_tokens = end_idx - start_idx

    rejected = False
    for pos in range(num_draft_tokens):
        if not rejected:
            draft_token_id = tl.load(draft_token_ids_ptr + start_idx + pos)
            target_argmax_id = tl.load(target_argmax_ptr + start_idx + pos).to(tl.int32)
            if SYNTHETIC_MODE:
                # 合成模式: 使用均匀随机数和条件接受率
                uniform_prob = tl.load(uniform_probs_ptr + start_idx + pos)
                rate = tl.load(synthetic_conditional_rates_ptr + pos)
                accepted = uniform_prob < rate
                token_id = draft_token_id if accepted else target_argmax_id
                rejected = not accepted
            else:
                # 标准模式: 比较草稿token和目标argmax
                token_id = target_argmax_id
                rejected = draft_token_id != target_argmax_id
            tl.store(
                output_token_ids_ptr + req_idx * (max_spec_len + 1) + pos,
                token_id,
            )

    if not rejected:
        # 如果所有token都被接受，添加bonus token
        bonus_token_id = tl.load(bonus_token_ids_ptr + req_idx)
        tl.store(
            output_token_ids_ptr + req_idx * (max_spec_len + 1) + num_draft_tokens,
            bonus_token_id,
        )


# NOTE(woosuk): 避免特化以防止不必要的重编译
@triton.jit(do_not_specialize=["max_spec_len"])
def rejection_random_sample_kernel(
    output_token_ids_ptr,  # [batch_size, max_spec_len + 1]
    cu_num_draft_tokens_ptr,  # [batch_size]
    draft_token_ids_ptr,  # [num_tokens]
    draft_probs_ptr,  # [num_tokens, vocab_size] 或 None
    target_probs_ptr,  # [num_tokens, vocab_size]
    bonus_token_ids_ptr,  # [batch_size]
    recovered_token_ids_ptr,  # [num_tokens]
    uniform_probs_ptr,  # [num_tokens]
    is_greedy_ptr,  # [batch_size]
    max_spec_len,
    vocab_size,
    synthetic_conditional_rates_ptr,  # [num_speculative_tokens] 或 None
    NO_DRAFT_PROBS: tl.constexpr,
    SYNTHETIC_MODE: tl.constexpr,
):
    """随机采样的拒绝采样Triton内核。

    对于每个请求，逐个检查草稿token:
    - 标准模式: 生成均匀随机数u，如果 target_prob/draft_prob >= u，则接受
    - 合成模式: 使用均匀随机数和条件接受率决定是否接受

    被拒绝的token使用预计算的恢复token替代。
    如果所有token都被接受，在末尾添加bonus token。
    """
    req_idx = tl.program_id(0)
    is_greedy = tl.load(is_greedy_ptr + req_idx)
    if is_greedy:
        # 贪心采样请求提前退出
        return

    start_idx = 0 if req_idx == 0 else tl.load(cu_num_draft_tokens_ptr + req_idx - 1)
    end_idx = tl.load(cu_num_draft_tokens_ptr + req_idx)
    num_draft_tokens = end_idx - start_idx

    rejected = False
    for pos in range(num_draft_tokens):
        if not rejected:
            draft_token_id = tl.load(draft_token_ids_ptr + start_idx + pos)
            uniform_prob = tl.load(uniform_probs_ptr + start_idx + pos)
            if SYNTHETIC_MODE:
                # 合成模式
                rate = tl.load(synthetic_conditional_rates_ptr + pos)
                accepted = uniform_prob < rate
            else:
                # 标准拒绝采样
                if NO_DRAFT_PROBS:
                    draft_prob = 1
                else:
                    draft_prob = tl.load(
                        draft_probs_ptr
                        + (start_idx + pos) * vocab_size
                        + draft_token_id
                    )
                target_prob = tl.load(
                    target_probs_ptr + (start_idx + pos) * vocab_size + draft_token_id
                )
                # NOTE(woosuk): 虽然草稿概率不应该为0，但我们检查它以避免NaN。
                # 如果恰好为0，则拒绝。
                accepted = draft_prob > 0 and target_prob / draft_prob >= uniform_prob
            if accepted:
                token_id = draft_token_id
            else:
                rejected = True
                token_id = tl.load(recovered_token_ids_ptr + start_idx + pos)
            tl.store(
                output_token_ids_ptr + req_idx * (max_spec_len + 1) + pos, token_id
            )

    if not rejected:
        # 如果所有token都被接受，添加bonus token
        bonus_token_id = tl.load(bonus_token_ids_ptr + req_idx)
        tl.store(
            output_token_ids_ptr + req_idx * (max_spec_len + 1) + num_draft_tokens,
            bonus_token_id,
        )


# NOTE(woosuk): 避免特化以防止不必要的重编译
@triton.jit(do_not_specialize=["replace_from", "replace_to"])
def expand_kernel(
    output_ptr,  # [num_tokens]
    input_ptr,  # [batch_size]
    cu_num_tokens_ptr,  # [batch_size]
    replace_from,
    replace_to,
    MAX_NUM_TOKENS: tl.constexpr,
):
    """扩展内核：将batch级别的值扩展到token级别。

    每个Triton程序处理一个batch元素，将其值复制到对应的token范围。
    如果值等于replace_from，则替换为replace_to。
    """
    req_idx = tl.program_id(0)
    if req_idx == 0:  # noqa: SIM108
        start_idx = 0
    else:
        start_idx = tl.load(cu_num_tokens_ptr + req_idx - 1)
    end_idx = tl.load(cu_num_tokens_ptr + req_idx)
    num_tokens = end_idx - start_idx

    src_val = tl.load(input_ptr + req_idx)
    src_val = tl.where(src_val == replace_from, replace_to, src_val)
    offset = tl.arange(0, MAX_NUM_TOKENS)
    tl.store(output_ptr + start_idx + offset, src_val, mask=offset < num_tokens)


@triton.jit
def sample_recovered_tokens_kernel(
    output_token_ids_ptr,  # [num_tokens]
    cu_num_draft_tokens_ptr,  # [batch_size]
    draft_token_ids_ptr,  # [num_tokens]
    draft_probs_ptr,  # [num_tokens, vocab_size] 或 None
    target_probs_ptr,  # [num_tokens, vocab_size]
    inv_q_ptr,  # [batch_size, vocab_size]
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
    NO_DRAFT_PROBS: tl.constexpr,
):
    """恢复token采样Triton内核。

    使用逆变换采样方法从恢复分布中采样token。
    恢复分布: q(x) = max(target_prob(x) - draft_prob(x), 0)

    选择 prob * inv_q 最大的token作为恢复token（Gumbel-max技巧的变体）。

    每个Triton程序处理一个(batch, position)对。
    通过分块遍历词表来处理大词表。
    """
    req_idx = tl.program_id(0)
    start_idx = 0 if req_idx == 0 else tl.load(cu_num_draft_tokens_ptr + req_idx - 1)
    end_idx = tl.load(cu_num_draft_tokens_ptr + req_idx)
    num_draft_tokens = end_idx - start_idx

    # 超出范围的位置提前退出
    pos = tl.program_id(1)
    if pos >= num_draft_tokens:
        return

    token_idx = start_idx + pos

    if NO_DRAFT_PROBS:
        draft_token_id = tl.load(draft_token_ids_ptr + token_idx)

    max_val = float("-inf")
    recovered_id = 0
    # 分块遍历词表
    for v in range(0, vocab_size, BLOCK_SIZE):
        vocab_offset = v + tl.arange(0, BLOCK_SIZE)
        vocab_mask = vocab_offset < vocab_size

        if NO_DRAFT_PROBS:
            # 无草稿概率模式: 恢复分布 = target_prob，但排除草稿token
            prob = tl.load(
                target_probs_ptr + token_idx * vocab_size + vocab_offset,
                mask=(vocab_mask & (vocab_offset != draft_token_id)),
                other=0.0,
            )
        else:
            # 有草稿概率模式: 恢复分布 = max(target_prob - draft_prob, 0)
            draft_prob = tl.load(
                draft_probs_ptr + token_idx * vocab_size + vocab_offset,
                mask=vocab_mask,
                other=0.0,
            )
            target_prob = tl.load(
                target_probs_ptr + token_idx * vocab_size + vocab_offset,
                mask=vocab_mask,
                other=0.0,
            )
            prob = tl.maximum(target_prob - draft_prob, 0.0)
            # NOTE(woosuk): 我们不需要 `prob = prob / tl.sum(prob)` 因为
            # `tl.argmax` 会选择最大值。

        inv_q = tl.load(
            inv_q_ptr + req_idx * vocab_size + vocab_offset,
            mask=vocab_mask,
            other=0.0,
        )

        # 局部tile归约
        score = prob * inv_q
        local_max, local_id = tl.max(score, axis=0, return_indices=True)

        if local_max > max_val:
            max_val = local_max
            recovered_id = v + local_id

    tl.store(output_token_ids_ptr + token_idx, recovered_id)
