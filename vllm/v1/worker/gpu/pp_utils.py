# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pipeline Parallelism utils for V2 Model Runner."""

# 流水线并行（Pipeline Parallelism, PP）工具模块。
# ================================================
# 在流水线并行模式下，模型被分割到多个 rank 上依次执行。
# 只有最后一个 rank（pp.last_rank）负责执行采样（sampling）操作，
# 因此需要将采样结果广播给其他 rank，以便所有 rank 保持同步。
#
# 本模块提供两个函数：
#   1. pp_broadcast()  —— 由最后一个 rank 调用，将采样结果广播给其他 rank。
#   2. pp_receive()    —— 由非最后一个 rank 调用，接收最后一个 rank 广播的采样结果。
#
# 广播的数据包含：
#   - sampled_token_ids: 采样得到的 token ID 张量，形状为 (num_reqs, max_sample_len)。
#   - num_sampled:       每个请求实际采样到的 token 数量。
#   - num_rejected:      每个请求被拒绝的 token 数量（用于 speculative decoding 的验证）。

import torch

from vllm.distributed.parallel_state import get_pp_group


def pp_broadcast(
    sampled_token_ids: torch.Tensor,
    num_sampled: torch.Tensor,
    num_rejected: torch.Tensor,
) -> None:
    """
    将采样结果从最后一个 PP rank 广播到所有其他 rank。

    此函数仅在 PP 组的最后一个 rank 上调用（通过 assert 验证）。
    它执行两次 broadcast 操作：
      1. 广播 sampled_token_ids（采样的 token ID）。
      2. 将 num_sampled 和 num_rejected 沿第 0 维堆叠后一起广播，
         以减少通信次数。

    参数：
      sampled_token_ids: 采样得到的 token ID 张量，dtype 必须为 torch.int64。
                         形状为 (num_reqs, max_sample_len)。
      num_sampled:       每个请求实际采样的有效 token 数，dtype 为 torch.int32。
      num_rejected:      每个请求被拒绝（rejection sampling）的 token 数，
                         dtype 为 torch.int32。
    """
    pp = get_pp_group()
    # 确保只有最后一个 rank 调用此函数。
    assert pp.is_last_rank

    # 广播采样得到的 token ID。contiguous() 确保内存连续，broadcast 要求。
    assert sampled_token_ids.dtype == torch.int64
    torch.distributed.broadcast(
        sampled_token_ids.contiguous(), src=pp.last_rank, group=pp.device_group
    )

    # 将 num_sampled 和 num_rejected 堆叠为形状 (2, num_reqs) 的张量，
    # 然后一次广播，减少通信开销。
    combined = torch.stack((num_sampled, num_rejected), dim=0)
    torch.distributed.broadcast(combined, src=pp.last_rank, group=pp.device_group)


def pp_receive(
    num_reqs: int, max_sample_len: int = 1
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    从最后一个 PP rank 接收采样结果。

    此函数在非最后一个 PP rank 上调用（通过 assert 验证）。
    它预分配接收缓冲区，然后通过两次 broadcast 接收数据：
      1. 接收 sampled_token_ids。
      2. 接收 combined（num_sampled + num_rejected），然后拆分。

    参数：
      num_reqs:       当前批处理中的请求数量。
      max_sample_len: 每个请求的最大采样长度，默认为 1（非 speculative 场景）。
                      在 speculative decoding 中可能更大。

    返回值，包含三个元素的元组：
      1. sampled_tokens: 接收到的采样 token ID 张量，
                         形状为 (num_reqs, max_sample_len)，dtype 为 torch.int64。
      2. num_sampled:    每个请求实际采样的有效 token 数，dtype 为 torch.int32。
      3. num_rejected:   每个请求被拒绝的 token 数，dtype 为 torch.int32。
    """
    pp = get_pp_group()
    # 确保非最后一个 rank 调用此函数。
    assert not pp.is_last_rank

    # 预分配接收缓冲区，大小为 (num_reqs, max_sample_len)。
    sampled_tokens = torch.empty(
        num_reqs, max_sample_len, dtype=torch.int64, device=pp.device
    )
    # 从最后一个 rank 接收采样 token ID。
    torch.distributed.broadcast(sampled_tokens, src=pp.last_rank, group=pp.device_group)

    # 预分配接收缓冲区，大小为 (2, num_reqs)，用于接收 num_sampled 和 num_rejected。
    combined = torch.empty(2, num_reqs, dtype=torch.int32, device=pp.device)
    torch.distributed.broadcast(combined, src=pp.last_rank, group=pp.device_group)

    # 将 combined 张量沿第 0 维拆分为 num_sampled 和 num_rejected。
    num_sampled, num_rejected = combined.unbind(dim=0)
    return sampled_tokens, num_sampled, num_rejected
