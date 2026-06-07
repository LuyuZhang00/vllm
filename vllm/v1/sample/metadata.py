# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""采样元数据模块。

本模块定义了 `SamplingMetadata` 数据类，用于存储采样过程中所需的所有参数和配置信息。
这些元数据在模型前向传播后、采样之前构建，并传递给采样器使用。

主要包含以下信息:
1. 温度参数 (temperature) - 控制采样的随机程度
2. Top-K/Top-P 参数 - 用于截断低概率token
3. 惩罚参数 - 频率惩罚、存在惩罚、重复惩罚
4. 日志概率配置 - 控制是否返回logprobs及其数量
5. logits处理器 - 内置和自定义的logits处理逻辑
6. 投机解码相关参数
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from vllm.v1.sample.logits_processor import LogitsProcessors
from vllm.v1.sample.thinking_budget_state import ThinkingBudgetStateHolder


@dataclass
class SamplingMetadata:
    """采样元数据：存储一次采样步骤中所有请求的采样参数。

    该数据类在每个解码步骤中由调度器或模型运行器构建，包含批量中所有请求
    共享和独立的采样配置。采样器依据这些元数据来决定如何从logits中采样token。

    字段说明:
    ---------
    温度与采样策略:
        temperature: 温度张量 [batch_size]，值越小越确定性，0表示贪心采样
        all_greedy: 是否所有请求都使用贪心采样（temperature=0）
        all_random: 是否所有请求都使用随机采样（temperature>0）
        top_p: 核采样(nucleus sampling)的累积概率阈值 [batch_size]
        top_k: Top-K采样的K值 [batch_size]

    随机数生成器:
        generators: 请求索引到随机数生成器的映射，用于可复现采样

    日志概率:
        max_num_logprobs: 返回的最大logprobs数量，None表示不返回，0表示仅返回采样token的logprobs

    惩罚参数:
        no_penalties: 是否没有惩罚（优化标志）
        prompt_token_ids: 提示token IDs [batch_size, max_prompt_len]
        frequency_penalties: 频率惩罚系数 [batch_size]
        presence_penalties: 存在惩罚系数 [batch_size]
        repetition_penalties: 重复惩罚系数 [batch_size]

    输出与约束:
        output_token_ids: 每个请求已生成的输出token ID列表
        allowed_token_ids_mask: 允许的token ID掩码 [batch_size, vocab_size]
        bad_words_token_ids: 需要排除的坏词token ID序列 {req_index: [[token_ids]]}

    处理器:
        logitsprocs: 已加载的logits处理器集合

    特殊功能:
        logprob_token_ids: 特定token ID的logprobs请求 {req_index: [token_ids]}
        spec_token_ids: 投机解码的token IDs
        thinking_budget_state_holder: 思考预算状态持有者（用于控制推理模型的思考token数量）
    """

    # ---- 温度与采样策略 ----
    temperature: torch.Tensor | None
    all_greedy: bool
    all_random: bool

    top_p: torch.Tensor | None
    top_k: torch.Tensor | None

    # ---- 随机数生成器 ----
    generators: dict[int, torch.Generator]

    # ---- 日志概率 ----
    # None表示不返回logprobs，0表示仅返回采样token的logprobs
    max_num_logprobs: int | None

    # ---- 惩罚参数 ----
    no_penalties: bool
    prompt_token_ids: torch.Tensor | None
    frequency_penalties: torch.Tensor
    presence_penalties: torch.Tensor
    repetition_penalties: torch.Tensor

    # ---- 输出token IDs ----
    output_token_ids: list[list[int]]

    # ---- token约束 ----
    # `allowed_token_ids_mask` 是一个2D布尔张量，形状为 (最大batch大小, 词表大小)
    # True表示该token被禁止（将被设为-inf）
    allowed_token_ids_mask: torch.Tensor | None

    # ---- 坏词排除 ----
    # 请求索引 -> 需要排除的坏词token ID序列
    bad_words_token_ids: dict[int, list[list[int]]]

    # ---- logits处理器 ----
    # 已加载的logits处理器（包括内置和自定义的）
    logitsprocs: LogitsProcessors

    # ---- 特定token的logprobs ----
    # 当设置时，仅对这些token ID计算logprobs（比全词表更高效）
    # 用于generative_scoring API，获取特定token的logprobs
    # 请求索引 -> 需要计算logprobs的token ID列表
    logprob_token_ids: dict[int, list[int]] | None = None

    # ---- 投机解码 ----
    # 投机解码的token IDs列表（每个请求一个列表）
    spec_token_ids: list[list[int]] | None = None

    # ---- 思考预算 ----
    # 当非None时，使用 ``holder.has_tracked_requests()`` 检查当前batch是否需要
    # 应用思考token预算的logits处理（holder可能存在但跟踪集为空）
    thinking_budget_state_holder: ThinkingBudgetStateHolder | None = None
