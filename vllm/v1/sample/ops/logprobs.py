# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
对数概率工具模块 (Logprobs Utilities)

本模块提供与对数概率 (log probabilities) 相关的工具函数，
主要用于 API 层面返回 top-k logprobs 等统计信息。

背景知识：
  对数概率 (logprob) 是 token 概率的自然对数：logprob = log(P(token))。
  值范围为 (-inf, 0]，其中 0 表示概率为 1（完全确定），
  负值越大（越接近 -inf）表示概率越低。

  在 API 响应中返回 logprobs 有助于：
  - 用户了解模型对每个 token 的置信度
  - 实现 logprobs 过滤（只返回高于阈值的候选）
  - 用于评估和调试模型的输出质量
"""

import torch

from vllm.platforms import current_platform


@torch.compile(backend=current_platform.simple_compile_backend)
def batched_count_greater_than(x: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """
    统计每行中大于对应阈值的元素数量。

    使用 torch.compile 为该函数生成优化的内核。如果不使用 torch.compile，
    会创建输入张量的额外副本，导致内存问题。

    此函数用于计算每个请求中 logprobs 大于指定阈值的 token 数量，
    以便在 API 响应中只返回 top-k 个 logprobs。

    Args:
        x: [batch_size, n_elements] 的二维张量，通常是 logprobs 张量
        values: [batch_size, 1] 的二维张量，每行的比较阈值

    Returns:
        [batch_size] 的一维张量，每行中 >= 对应阈值的元素数量

    Example:
        x = [[0.1, 0.5, 0.3, 0.8],    # 4 个元素 >= 0.3 → 3
             [0.2, 0.4, 0.6, 0.1]]     # 2 个元素 >= 0.4 → 2
        values = [[0.3], [0.4]]
        → [3, 2]
    """
    torch._check(x.shape[0] >= 1)
    torch._check(x.shape[0] == values.shape[0])
    return (x >= values).sum(-1)
