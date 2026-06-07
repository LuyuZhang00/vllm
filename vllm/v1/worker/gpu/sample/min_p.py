# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Min-P 采样过滤模块 (Min-P Sampling Filter Module)

本模块实现了 Min-P 采样策略，用于在采样前过滤 logits。

Min-P 采样的原理：
1. 找到最大 logit 值 max_val
2. 计算阈值 threshold = max_val + log(min_p)
3. 将所有 logit < threshold 的 token 设为 -inf

等价于：只保留概率 >= min_p * max_probability 的 token。

例如，如果 min_p = 0.1，最大概率 token 的概率为 0.5，
则只保留概率 >= 0.05 的 token。

与 top-p (nucleus) 采样的区别：
- top-p: 累积概率达到 p 的 token 被保留（与 token 数量相关）
- min_p: 概率至少为 max_prob * min_p 的 token 被保留（与绝对概率相关）

Min-p 的优势：
- 更稳定：不受概率分布形状的影响
- 更直观：min_p=0.1 意味着保留概率至少为最大概率 10% 的 token
"""
import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _min_p_kernel(
    logits_ptr,
    logits_stride,
    expanded_idx_mapping_ptr,
    min_p_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    """Min-P 过滤 Triton 内核。

    对每个 token 执行以下操作：
    1. 找到最大 logit 值
    2. 计算阈值 = max_val + log(min_p)
    3. 将 logit < 阈值的 token 设为 -inf

    Args:
        logits_ptr: logits 数据指针
        logits_stride: logits 的行步长
        expanded_idx_mapping_ptr: 扩展索引映射指针
        min_p_ptr: min_p 参数指针
        vocab_size: 词表大小
        BLOCK_SIZE: 每个 block 处理的 vocab 大小（编译时常量）
    """
    token_idx = tl.program_id(0)
    req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx)
    min_p = tl.load(min_p_ptr + req_state_idx).to(tl.float32)
    if min_p == 0.0:
        return

    # 第一遍：找到最大 logit 值
    max_val = float("-inf")
    for i in range(0, vocab_size, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < vocab_size
        logits = tl.load(
            logits_ptr + token_idx * logits_stride + block,
            mask=mask,
            other=float("-inf"),
        )
        max_val = tl.max(tl.maximum(logits, max_val))
    max_val = max_val.to(tl.float32)  # type: ignore

    # 计算阈值：max_val + log(min_p)
    # log(min_p) < 0，所以 threshold < max_val
    threshold = max_val + tl.log(min_p)
    # 第二遍：将低于阈值的 logit 设为 -inf
    for i in range(0, vocab_size, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < vocab_size
        logits = tl.load(
            logits_ptr + token_idx * logits_stride + block,
            mask=mask,
            other=float("-inf"),
        )
        logits = tl.where(logits < threshold, float("-inf"), logits)
        tl.store(logits_ptr + token_idx * logits_stride + block, logits, mask=mask)


def apply_min_p(
    logits: torch.Tensor, expanded_idx_mapping: torch.Tensor, min_p: torch.Tensor
) -> None:
    """应用 Min-P 过滤到 logits。

    Args:
        logits: logits 张量 [num_tokens, vocab_size]
        expanded_idx_mapping: 扩展的请求索引映射 [num_tokens]
        min_p: min_p 参数 [max_num_reqs]
    """
    num_tokens, vocab_size = logits.shape
    BLOCK_SIZE = 1024
    _min_p_kernel[(num_tokens,)](
        logits,
        logits.stride(0),
        expanded_idx_mapping,
        min_p,
        vocab_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
