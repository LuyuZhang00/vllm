# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
调度辅助工具模块 (vllm/v1/core/sched/utils.py)

本模块提供调度器使用的辅助函数。

模块包含以下功能：

1. 重复模式检测 (check_sequence_repetition):
   - 检测生成的 token 序列中是否存在重复模式
   - 用于重复检测停止条件（防止模型陷入重复循环）
   - 支持配置最小/最大模式长度和最小重复次数

2. 列表元素移除 (remove_all):
   - 从列表中移除指定集合中的所有元素
   - 优化单元素移除的常见情况（原地修改）
   - 多元素移除使用列表推导式

3. 停止条件检查 (check_stop):
   - 检查请求是否满足停止条件
   - 支持多种停止条件：EOS token、stop token IDs、最大长度、重复检测
   - 更新请求的完成状态
"""

import contextlib
from collections.abc import Sequence

from vllm.sampling_params import RepetitionDetectionParams
from vllm.v1.request import Request, RequestStatus


def _has_repeating_pattern(
    token_ids: Sequence[int],
    pattern_len: int,
    repetition_min_count: int,
) -> bool:
    """检查 token_ids 的尾部是否包含重复模式。

    将最后 pattern_len 个 token 与前面的 (repetition_min_count - 1)
    个相同长度的重复块进行比较。

    检测逻辑：
    对于每个位置 n (1 到 pattern_len)，检查：
    token_ids[-n] == token_ids[-(pattern_len * 1 + n)] == token_ids[-(pattern_len * 2 + n)] == ...

    Args:
        token_ids: token ID 序列
        pattern_len: 模式长度
        repetition_min_count: 最小重复次数

    Returns:
        True 表示检测到重复模式
    """
    for n in range(1, pattern_len + 1):
        target_token = token_ids[-n]
        for m in range(1, repetition_min_count):
            if token_ids[-(pattern_len * m + n)] != target_token:
                return False
    return True


def check_sequence_repetition(
    token_ids: Sequence[int],
    params: RepetitionDetectionParams,
) -> bool:
    """检查 token ID 序列是否存在重复模式。

    在不同的模式长度上尝试检测重复：
    从 min_pattern_size 到 max_pattern_size，逐步检查。

    Args:
        token_ids: token ID 列表
        params: 重复检测参数

    Returns:
        True 表示检测到重复模式，False 表示未检测到
    """
    max_pattern_size = params.max_pattern_size
    min_pattern_size = params.min_pattern_size
    min_count = params.min_count

    if min_pattern_size <= 0:
        min_pattern_size = 1

    if max_pattern_size <= 0 or min_count < 2 or min_pattern_size > max_pattern_size:
        return False

    for pattern_len in range(
        min_pattern_size,
        max_pattern_size + 1,
    ):
        # 如果需要的 token 数超过序列长度，不可能有重复
        if pattern_len * min_count > len(token_ids):
            return False

        if _has_repeating_pattern(token_ids, pattern_len, min_count):
            return True

    return False


def remove_all(lst: list, items_to_remove: set) -> list:
    """从列表中移除 items_to_remove 集合中的所有元素。

    优化策略：
    - 单元素移除（最常见情况）：原地修改并返回
    - 多元素移除：使用列表推导式创建新列表

    Args:
        lst: 要操作的列表
        items_to_remove: 要移除的元素集合

    Returns:
        修改后的列表（单元素时为原列表，多元素时为新列表）。
        调用者应使用返回值。

    Note:
        单元素移除时原地修改并返回原列表。
        多元素移除时创建并返回新列表。
    """
    if not items_to_remove:
        return lst

    if len(items_to_remove) == 1:
        # 单元素移除的快速路径（最常见情况）
        item = next(iter(items_to_remove))
        with contextlib.suppress(ValueError):
            lst.remove(item)
        return lst
    # 多元素移除使用列表推导式
    return [item for item in lst if item not in items_to_remove]


def check_stop(request: Request, max_model_len: int) -> bool:
    """检查请求是否满足停止条件。

    检查顺序：
    1. 最小 token 数检查：如果输出 token 数不足 min_tokens，不停止
    2. EOS token 检查：最后一个 token 是否为 EOS
    3. Stop token IDs 检查：最后一个 token 是否在 stop_token_ids 中
    4. 长度限制检查：总 token 数是否超过 max_model_len 或输出超过 max_tokens
    5. 重复检测检查：输出序列是否检测到重复模式

    Args:
        request: 要检查的请求
        max_model_len: 模型支持的最大序列长度

    Returns:
        True 表示请求应停止
    """
    assert not request.pooling_params

    sampling_params = request.sampling_params
    assert sampling_params is not None

    # 最小 token 数检查
    if request.num_output_tokens < sampling_params.min_tokens:
        return False

    last_token_id = request.output_token_ids[-1]
    # EOS token 检查
    if last_token_id == sampling_params.eos_token_id:
        request.status = RequestStatus.FINISHED_STOPPED
        return True

    # Stop token IDs 检查
    if last_token_id in (sampling_params.stop_token_ids or ()):
        request.status = RequestStatus.FINISHED_STOPPED
        request.stop_reason = last_token_id
        return True
    # 长度限制检查
    if (
        request.num_tokens >= max_model_len
        or request.num_output_tokens >= request.max_tokens
    ):
        request.status = RequestStatus.FINISHED_LENGTH_CAPPED
        return True

    # 重复检测检查
    repetition_detection = sampling_params.repetition_detection
    if repetition_detection is not None and (
        check_sequence_repetition(
            request.output_token_ids,
            repetition_detection,
        )
    ):
        request.status = RequestStatus.FINISHED_REPETITION
        request.stop_reason = "repetition_detected"
        return True

    return False
