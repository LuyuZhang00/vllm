# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
采样惩罚模块 (Sampling Penalties)

本模块实现了三种常见的采样惩罚机制，用于控制语言模型生成文本的多样性和
重复度。这些惩罚在采样阶段对 logits 施加调整，使模型更倾向于（或不倾向于）
生成某些 token。

三种惩罚机制详解：

1. 存在惩罚 (Presence Penalty):
   - 公式: logits[i] -= presence_penalty * (1 if token_i appeared else 0)
   - 效果: 对已出现过的所有 token 施加固定的惩罚量，不论其出现次数
   - 参数范围: [-2.0, 2.0]
   - 正值鼓励多样性（减少重复），负值鼓励重复

2. 频率惩罚 (Frequency Penalty):
   - 公式: logits[i] -= frequency_penalty * count(token_i)
   - 效果: 对已出现过的 token 按出现次数线性增加惩罚
   - 参数范围: [-2.0, 2.0]
   - 与存在惩罚类似，但惩罚量随出现次数递增

3. 重复惩罚 (Repetition Penalty):
   - 公式: logits[i] /= repetition_penalty  (if logits[i] > 0 and appeared)
                logits[i] *= repetition_penalty  (if logits[i] < 0 and appeared)
   - 效果: 通过乘性因子缩小已出现 token 的 logits 绝对值
   - 参数范围: [0.0, 2.0]，其中 1.0 表示无惩罚
   - 大于 1 时减少重复，小于 1 时鼓励重复

这三种惩罚可以组合使用，由 vLLM 配置中的对应参数控制。
"""

import torch

from vllm.model_executor.layers.utils import apply_penalties
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.utils.torch_utils import make_tensor_with_pad


def apply_all_penalties(
    logits: torch.Tensor,
    prompt_token_ids: torch.Tensor,
    presence_penalties: torch.Tensor,
    frequency_penalties: torch.Tensor,
    repetition_penalties: torch.Tensor,
    output_token_ids: list[list[int]],
) -> torch.Tensor:
    """
    对 logits 应用所有三种惩罚：存在惩罚、频率惩罚和重复惩罚。

    处理流程：
    1. 将输出 token ID 列表转换为填充后的张量（tensor）
    2. 处理异步调度中的占位符（-1 值替换为有效 token ID）
    3. 调用 apply_penalties 应用惩罚并返回修改后的 logits

    Args:
        logits: [batch_size, vocab_size] 的 logits 张量
        prompt_token_ids: [batch_size, max_prompt_len] 的 prompt token ID 张量
        presence_penalties: [batch_size] 的存在惩罚值张量
        frequency_penalties: [batch_size] 的频率惩罚值张量
        repetition_penalties: [batch_size] 的重复惩罚值张量
        output_token_ids: 列表，每个元素是对应请求已生成的 token ID 列表
            例如 [[10, 20, 30], [5, 6]] 表示 batch[0] 已生成 3 个 token

    Returns:
        经过惩罚处理后的 logits 张量
    """
    _, vocab_size = logits.shape
    # 将不等长的 token ID 列表转换为固定大小的张量（用 vocab_size 作为填充值）
    output_tokens_t = _convert_to_tensors(output_token_ids, vocab_size, logits.device)

    # 在异步调度的情况下，不会应用惩罚的行可能包含 -1 占位符 token ID。
    # 必须将这些替换为有效的 token ID，以确保 apply_penalties 中的 scatter 操作有效。
    # 注意(nick): 当前的惩罚实现效率不高，后续会重新设计。
    output_tokens_t.masked_fill_(output_tokens_t == -1, vocab_size)

    return apply_penalties(
        logits,
        prompt_token_ids,
        output_tokens_t,
        presence_penalties,
        frequency_penalties,
        repetition_penalties,
    )


def _convert_to_tensors(
    output_token_ids: list[list[int]], vocab_size: int, device: torch.device
) -> torch.Tensor:
    """
    将不等长的 token ID 列表结构转换为固定大小的张量。

    使用 make_tensor_with_pad 将变长列表填充为固定长度的张量，
    填充值使用 vocab_size（因为不存在该值的 token ID，不会影响惩罚计算）。

    Args:
        output_token_ids: 变长的 token ID 列表
            例如 [[10, 20, 30], [5]] 表示两个请求分别生成了 3 个和 1 个 token
        vocab_size: 词表大小，用作填充值
        device: 目标设备（GPU/CPU）

    Returns:
        [batch_size, max_len] 的 int64 张量，填充位置的值为 vocab_size
    """
    # 在 CPU 上创建张量并使用 pin_memory 加速后续的 GPU 传输
    output_tokens_tensor = make_tensor_with_pad(
        output_token_ids,
        # 使用 vocab_size 作为填充值，因为我们没有 token ID 等于 vocab_size
        pad=vocab_size,
        device="cpu",
        dtype=torch.int64,
        pin_memory=is_pin_memory_available(),
    )
    # 异步传输到目标设备
    return output_tokens_tensor.to(device, non_blocking=True)
