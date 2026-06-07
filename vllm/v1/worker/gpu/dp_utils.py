# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
数据并行（Data Parallelism, DP）与 CUDA Graph 协调工具模块。

在数据并行模式下，多个 GPU 各自处理批次的不同部分。为了确保
CUDA Graph 的正确性，所有 DP 排名必须使用相同形状的 CUDA Graph。

本模块提供两个关键功能：
1. sync_cudagraph_and_dp_padding: 在所有 DP 排名间同步批次描述符和填充
2. dispatch_cg_and_dp_padding: 调度 CUDA Graph 并同步 DP 排名

协调策略：
- 所有排名必须使用相同的 CUDA Graph 模式（取最小值 = 最严格模式）
- token 数取所有排名中的最大值（不足的排名需要填充）
- 统一 token 计数如果不一致则设为 None
"""
from __future__ import annotations

import torch
import torch.distributed as dist

from vllm.config.compilation import CUDAGraphMode
from vllm.distributed.parallel_state import get_dp_group
from vllm.v1.worker.gpu.cudagraph_utils import (
    BatchExecutionDescriptor,
    CudaGraphManager,
)


def sync_cudagraph_and_dp_padding(
    cudagraph_manager: CudaGraphManager | None,
    desired_batch_desc: BatchExecutionDescriptor,
    num_tokens: int,
    num_reqs: int,
    uniform_token_count: int | None,
    dp_size: int,
    dp_rank: int,
) -> tuple[BatchExecutionDescriptor, torch.Tensor | None]:
    """在所有 DP 排名间协调批次描述符和 DP 填充。

    使用 all_reduce 收集所有排名的 token 数、CG 模式和统一 token 计数，
    然后选择一致的策略。

    协调规则：
    1. CG 模式取所有排名的最小值（最严格模式）
    2. 如果任何排名需要 eager 模式，所有排名都使用 eager
    3. token 数取所有排名的最大值（不足的需要填充）
    4. 统一 token 计数不一致时设为 None

    参数:
        cudagraph_manager: CUDA Graph 管理器（profile 阶段可为 None）
        desired_batch_desc: 本排名期望的批次描述符
        num_tokens: 本排名的 token 数
        num_reqs: 本排名的请求数
        uniform_token_count: 本排名的统一 token 计数
        dp_size: DP 总排名数
        dp_rank: 当前 DP 排名

    返回:
        (synced_batch_desc, num_tokens_across_dp):
        - 同步后的批次描述符
        - 所有 DP 排名的 token 数张量（或 None，如果所有排名的 token 数为 0）
    """
    assert dp_size > 1, "DP size must be greater than 1"
    group = get_dp_group().cpu_group
    tensor = torch.zeros(3, dp_size, dtype=torch.int32, device="cpu")
    tensor[0][dp_rank] = num_tokens
    tensor[1][dp_rank] = desired_batch_desc.cg_mode.value
    tensor[2][dp_rank] = uniform_token_count or 0  # (0 means None)
    dist.all_reduce(tensor, group=group)

    num_tokens_across_dp = tensor[0]
    cg_mode_across_dp = tensor[1]
    uniform_token_counts_across_dp = tensor[2]

    if torch.all(num_tokens_across_dp == 0).item():
        synced_desc = BatchExecutionDescriptor(
            cg_mode=CUDAGraphMode.NONE, num_tokens=0, num_reqs=0
        )
        return synced_desc, None

    # CG 模式取最小值 = 最严格模式
    synced_cg_mode = CUDAGraphMode(int(cg_mode_across_dp.min().item()))

    # 如果任何排名需要 eager，所有排名都使用 eager
    if synced_cg_mode == CUDAGraphMode.NONE:
        return BatchExecutionDescriptor(
            cg_mode=CUDAGraphMode.NONE,
            num_tokens=num_tokens,
            num_reqs=num_reqs,
        ), num_tokens_across_dp

    assert cudagraph_manager is not None, (
        "cudagraph_manager should only be None during profile run, "
        "where synced_cg_mode must be NONE across all DP ranks"
    )
    # token 数取最大值（不足的排名需要填充）
    synced_num_tokens = int(num_tokens_across_dp.max().item())
    synced_uniform_token_count = uniform_token_counts_across_dp[0]
    # 如果排名间不一致或为 0（表示 None），设为 None
    if synced_uniform_token_count == 0 or not torch.all(
        uniform_token_counts_across_dp == synced_uniform_token_count
    ):
        synced_uniform_token_count = None

    # 使用同步后的值调度 CUDA Graph
    synced_desc = cudagraph_manager.dispatch(
        num_reqs, synced_num_tokens, synced_uniform_token_count
    )

    # 更新 num_tokens_across_dp 以反映填充后的大小
    num_tokens_across_dp[:] = synced_desc.num_tokens

    return synced_desc, num_tokens_across_dp


def dispatch_cg_and_sync_dp(
    cudagraph_manager: CudaGraphManager | None,
    num_reqs: int,
    num_tokens: int,
    uniform_token_count: int | None,
    dp_size: int,
    dp_rank: int,
    need_eager: bool = False,
) -> tuple[BatchExecutionDescriptor, torch.Tensor | None]:
    """调度 CUDA Graph 并同步 DP 排名。

    这是对外的高层接口，结合了 CUDA Graph 调度和 DP 同步两个步骤。

    参数:
        cudagraph_manager: CUDA Graph 管理器（profile 阶段可为 None）
        num_reqs: 请求数量
        num_tokens: token 数量
        uniform_token_count: 统一的每请求 token 数
        dp_size: DP 总排名数
        dp_rank: 当前 DP 排名
        need_eager: 是否强制使用 eager 模式

    返回:
        (batch_desc, num_tokens_across_dp):
        - 批次执行描述符
        - 所有 DP 排名的 token 数张量（DP_SIZE=1 时为 None）
    """
    if need_eager:
        batch_desc = BatchExecutionDescriptor(
            cg_mode=CUDAGraphMode.NONE,
            num_tokens=num_tokens,
            num_reqs=num_reqs,
        )
    else:
        assert cudagraph_manager is not None, (
            "cudagraph_manager should only be None during profile run, "
            "where need_eager must be True"
        )
        batch_desc = cudagraph_manager.dispatch(
            num_reqs, num_tokens, uniform_token_count
        )

    if dp_size == 1:
        return batch_desc, None

    return sync_cudagraph_and_dp_padding(
        cudagraph_manager,
        batch_desc,
        num_tokens,
        num_reqs,
        uniform_token_count,
        dp_size,
        dp_rank,
    )
