# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
上下文并行（Context Parallelism, CP）工具模块。

上下文并行是一种分布式推理策略，将长序列的 KV 缓存分片到多个 GPU 上。
每个 GPU 只存储和处理序列的一部分 token 的 KV 缓存。

核心概念：
1. CP 分片策略：使用 round-robin 交错方式分配 token 到不同的 CP 排名
   - 例如 CP_SIZE=2, CP_INTERLEAVE=1 时：
     token 0 -> rank 0, token 1 -> rank 1, token 2 -> rank 0, ...
   - CP_INTERLEAVE 控制连续分配给同一 rank 的 token 数量

2. 本地序列长度：每个 rank 上实际需要处理的 token 数量
   - 由于分片，本地序列长度通常小于全局序列长度

3. 与 CUDA Graph 的兼容性：
   - 使用持久化的缓冲区（dcp_local_seq_lens）填充数据
   - 避免每次调用时分配新的张量
"""
import torch

from vllm.triton_utils import tl, triton


def prepare_dcp_local_seq_lens(
    dcp_local_seq_lens: torch.Tensor,
    seq_lens: torch.Tensor,
    num_reqs: int,
    dcp_size: int,
    dcp_rank: int,
    cp_interleave: int,
) -> None:
    """填充持久化的 DCP 本地序列长度缓冲区（CUDA Graph 安全）。

    计算当前 CP 排名在每个请求中需要处理的本地 token 数量。
    使用 Triton 内核实现高效的并行计算。

    参数:
        dcp_local_seq_lens: 输出缓冲区，存放每个请求的本地序列长度
        seq_lens: 每个请求的全局序列长度
        num_reqs: 当前批次中的请求数量
        dcp_size: 上下文并行的总进程数
        dcp_rank: 当前进程的 CP 排名
        cp_interleave: 交错因子（连续分配给同一 rank 的 token 数量）
    """
    if dcp_size == 1:
        return

    max_num_reqs = dcp_local_seq_lens.shape[0]
    BLOCK_SIZE = 128
    num_blocks = triton.cdiv(max_num_reqs, BLOCK_SIZE)
    _dcp_local_seq_lens_kernel[(num_blocks,)](
        dcp_local_seq_lens,
        seq_lens,
        dcp_size,
        dcp_rank,
        cp_interleave,
        num_reqs,
        max_num_reqs,
        BLOCK_SIZE,
    )


@triton.jit
def _dcp_local_seq_lens_kernel(
    out_ptr,
    seq_lens_ptr,
    dcp_size,
    dcp_rank,
    cp_interleave,
    num_reqs,
    max_num_reqs,
    BLOCK_SIZE: tl.constexpr,
):
    """计算 DCP 本地序列长度的 Triton 内核。

    使用 round-robin 策略将 KV 缓存分配给不同的 CP 排名。

    计算逻辑：
    1. rounds = seq_len // (dcp_size * cp_interleave)
       完整轮数，每轮中每个 rank 获得 cp_interleave 个 token
    2. remainder = seq_len % (dcp_size * cp_interleave)
       剩余部分
    3. local_remainder = clamp(remainder - dcp_rank * cp_interleave, 0, cp_interleave)
       当前 rank 在剩余部分中获得的 token 数
    4. local_seq_len = rounds * cp_interleave + local_remainder

    参数:
        out_ptr: 输出缓冲区指针
        seq_lens_ptr: 全局序列长度指针
        dcp_size: CP 总进程数
        dcp_rank: 当前 CP 排名
        cp_interleave: 交错因子
        num_reqs: 实际请求数
        max_num_reqs: 最大请求数（缓冲区大小）
        BLOCK_SIZE: Triton 块大小
    """
    pid = tl.program_id(0)
    block = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    seq_lens = tl.load(seq_lens_ptr + block, mask=block < num_reqs)

    # Distribute KV cache among different ranks, in a round-robin manner.
    rounds = seq_lens // (dcp_size * cp_interleave)
    remainder = seq_lens % (dcp_size * cp_interleave)

    remainder = tl.maximum(remainder - dcp_rank * cp_interleave, 0)
    remainder = tl.minimum(remainder, cp_interleave)
    local_seq_lens = rounds * cp_interleave + remainder

    # 对于 [num_reqs, max_num_reqs) 范围的请求，填充为 0
    local_seq_lens = tl.where(block < num_reqs, local_seq_lens, 0)
    tl.store(out_ptr + block, local_seq_lens, mask=block < max_num_reqs)
