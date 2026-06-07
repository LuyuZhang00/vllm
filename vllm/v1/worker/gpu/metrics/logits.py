# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Logits 指标计算模块 (Logits Metrics Module)

本模块实现了 logits 相关的指标计算功能，用于监控模型输出的数值健康状况。

当前支持的指标：
1. NaN 计数 (num_nans): 统计每个请求的 logits 中包含多少个 NaN 值

NaN 值的出现可能表明：
- 模型权重中存在数值异常
- 输入数据中存在异常值
- 数值计算过程中出现了溢出或下溢

此功能通过环境变量 VLLM_COMPUTE_NANS_IN_LOGITS 启用（默认关闭），
因为 NaN 检查会带来额外的计算开销。

使用 Triton 内核在 GPU 上高效并行统计 NaN 数量。
"""
import torch
from torch._inductor.runtime.triton_helpers import libdevice

from vllm.triton_utils import tl, triton


@triton.jit
def _num_nans_kernel(
    logits_ptr,
    logits_stride,
    num_nans_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    """NaN 计数 Triton 内核。

    统计每个请求的 logits 中包含的 NaN 值数量。
    使用分块处理以支持大词表。

    Args:
        logits_ptr: logits 数据指针 [num_reqs, vocab_size]
        logits_stride: logits 的行步长
        num_nans_ptr: 输出 NaN 计数指针 [num_reqs]
        vocab_size: 词表大小
        BLOCK_SIZE: 每个 block 处理的 vocab 大小（编译时常量）
    """
    req_idx = tl.program_id(0)
    num_nans = 0
    for i in range(0, vocab_size, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < vocab_size
        logits = tl.load(
            logits_ptr + req_idx * logits_stride + block, mask=mask, other=0
        )
        logits = logits.to(tl.float32)
        is_nan = libdevice.isnan(logits).to(tl.int1)
        num_nans += tl.sum(is_nan).to(tl.int32)
    tl.store(num_nans_ptr + req_idx, num_nans)


def get_num_nans(logits: torch.Tensor) -> torch.Tensor:
    """统计每个请求的 logits 中的 NaN 数量。

    Args:
        logits: logits 张量 [num_reqs, vocab_size]

    Returns:
        每个请求的 NaN 数量张量 [num_reqs]
    """
    num_reqs, vocab_size = logits.shape
    BLOCK_SIZE = 8192
    num_nans = torch.empty(num_reqs, dtype=torch.int32, device=logits.device)
    _num_nans_kernel[(num_reqs,)](
        logits,
        logits.stride(0),
        num_nans,
        vocab_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return num_nans
