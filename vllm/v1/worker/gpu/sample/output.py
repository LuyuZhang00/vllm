# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
采样器输出数据类模块 (Sampler Output Data Class Module)

本模块定义了采样器 (Sampler) 的输出数据结构 SamplerOutput。
"""
from dataclasses import dataclass

import torch

from vllm.v1.outputs import LogprobsTensors


@dataclass
class SamplerOutput:
    """采样器的输出数据结构。

    属性：
        sampled_token_ids: 采样的 token IDs，形状为 [num_requests, 1]
            每行包含一个请求生成的一个 token。
        logprobs_tensors: log 概率相关数据，包含：
            - logprob_token_ids: 需要返回 logprob 的 token IDs
            - logprobs: 对应的 log 概率值
            - selected_token_ranks: 采样 token 的排名
            如果不需要 logprob，则为 None。
        num_nans: 每个请求的 NaN 数量，用于检测数值异常。
            仅在 VLLM_COMPUTE_NANS_IN_LOGITS 环境变量启用时计算。
            如果未启用，则为 None。
        num_sampled: 每个请求采样的 token 数量。
            在标准采样模式下，每个请求总是采样 1 个 token。
    """
    sampled_token_ids: torch.Tensor
    logprobs_tensors: LogprobsTensors | None
    num_nans: torch.Tensor | None
    num_sampled: torch.Tensor | None
