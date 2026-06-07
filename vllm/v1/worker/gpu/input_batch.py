# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
输入批次管理模块。

本模块负责管理模型推理的输入批次数据，包括：
1. 输入缓冲区（InputBuffers）- 持久化的 GPU 缓冲区
2. 输入批次（InputBatch）- 当前批次的状态和数据
3. 各种 Triton 内核函数 - 用于高效的数据准备和更新

核心概念：
1. 请求状态索引（req_state_idx）：请求在全局状态数组中的位置
   - 批次索引（batch_idx）是当前批次中的位置
   - 通过 idx_mapping 从 batch_idx 映射到 req_state_idx

2. 分块预填充（Chunked Prefill）：长序列的预填充可以分多次进行
   - num_computed_tokens: 已计算的 token 数
   - prefill_len: 需要预填充的 token 总数
   - 当 num_computed_tokens < prefill_len 时，请求处于分块预填充状态

3. 投机解码（Speculative Decoding）：
   - 每个请求可能有多个 draft token
   - num_logits = 1（采样 token）+ num_draft_tokens
   - cu_num_logits 是 num_logits 的前缀和

主要 Triton 内核：
- _prepare_prefill_inputs_kernel: 准备预填充输入
- _prepare_pos_seq_lens_kernel: 准备位置编码和序列长度
- _combine_sampled_and_draft_tokens_kernel: 合并采样和 draft token
- _get_num_sampled_and_rejected_kernel: 计算采样和拒绝的 token 数
- _post_update_kernel: 后处理更新（更新 token ID、计算位置等）
- _expand_idx_mapping_kernel: 展开索引映射（用于投机解码）
"""
from dataclasses import dataclass

import numpy as np
import torch

from vllm.triton_utils import tl, triton
from vllm.utils import random_uuid


class InputBuffers:
    """持久化的 GPU 输入缓冲区。

    这些缓冲区在多次推理迭代间复用，避免频繁的 GPU 内存分配。
    大小基于最大配置（max_num_reqs, max_num_tokens），实际使用时
    通过切片访问有效部分。

    属性:
        input_ids: 输入 token ID [max_num_tokens]
        positions: 位置编码 [max_num_tokens]
        query_start_loc: 每个请求的查询起始位置 [max_num_reqs + 1]
        seq_lens: 每个请求的序列长度 [max_num_reqs]
        dcp_local_seq_lens: DCP 本地序列长度 [max_num_reqs]
    """

    def __init__(
        self,
        max_num_reqs: int,
        max_num_tokens: int,
        device: torch.device,
    ):
        self.max_num_reqs = max_num_reqs
        self.max_num_tokens = max_num_tokens
        self.device = device

        self.input_ids = torch.zeros(max_num_tokens, dtype=torch.int32, device=device)
        self.positions = torch.zeros(max_num_tokens, dtype=torch.int64, device=device)
        self.query_start_loc = torch.zeros(
            max_num_reqs + 1, dtype=torch.int32, device=device
        )
        self.seq_lens = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)
        # DCP: per-request local seq_lens buffer
        self.dcp_local_seq_lens = torch.zeros(
            max_num_reqs, dtype=torch.int32, device=device
        )


@dataclass
class InputBatch:
    """当前推理批次的状态数据。

    该数据类保存当前批次中所有请求的状态信息，包括索引映射、
    token 计数、输入数据和 logits 索引等。

    字段说明：
    - req_ids: 当前批次中的请求 ID 列表
    - num_reqs: 实际请求数（不含 padding）
    - num_reqs_after_padding: 填充后的请求数
    - idx_mapping: 批次索引到请求状态索引的映射
    - num_scheduled_tokens: 每个请求的已调度 token 数
    - num_tokens: 所有请求的 token 总数
    - num_tokens_after_padding: 填充后的 token 总数
    - num_draft_tokens: 投机解码的 draft token 总数
    - query_start_loc: 每个请求的查询起始位置（前缀和）
    - seq_lens: 每个请求的序列长度（已计算 + 当前查询）
    - logits_indices: 需要计算 logits 的 token 索引
    - cu_num_logits: num_logits 的前缀和（用于投机解码）
    """

    # batch_idx -> req_id
    req_ids: list[str]
    num_reqs: int
    num_reqs_after_padding: int

    # batch_idx -> req_state_idx
    idx_mapping: torch.Tensor
    idx_mapping_np: np.ndarray
    # Identical to idx_mapping except for spec decoding.
    expanded_idx_mapping: torch.Tensor
    # [total_num_logits] position within request for each logit
    expanded_local_pos: torch.Tensor

    # [num_reqs]
    # batch_idx -> num_scheduled_tokens
    num_scheduled_tokens: np.ndarray
    # sum(num_scheduled_tokens)
    num_tokens: int
    num_tokens_after_padding: int
    # Sum of draft tokens scheduled across requests.
    num_draft_tokens: int
    # [num_reqs] number of draft tokens scheduled for each request, if any.
    num_draft_tokens_per_req: np.ndarray | None

    # [num_reqs + 1]
    query_start_loc: torch.Tensor
    query_start_loc_np: np.ndarray
    # [num_reqs]
    seq_lens: torch.Tensor
    # [num_reqs] CPU upper bound on seq_lens (see CommonAttentionMetadata).
    seq_lens_cpu_upper_bound: torch.Tensor
    # [num_reqs]
    dcp_local_seq_lens: torch.Tensor | None
    # [num_reqs] CPU bool array.
    is_prefilling_np: np.ndarray

    # [num_tokens_after_padding]
    input_ids: torch.Tensor
    # [num_tokens_after_padding]
    positions: torch.Tensor

    # [total_num_logits]
    logits_indices: torch.Tensor
    # [num_reqs + 1]
    cu_num_logits: torch.Tensor
    cu_num_logits_np: np.ndarray

    # Whether any requests in batch use structured output.
    has_structured_output_reqs: bool

    @classmethod
    def make_dummy(
        cls,
        num_reqs: int,
        num_tokens: int,
        input_buffers: InputBuffers,
    ) -> "InputBatch":
        """创建用于 CUDA Graph 捕获的虚拟 InputBatch。

        生成形状正确但内容无关紧要的 InputBatch，仅用于确定
        CUDA Graph 捕获时的内存布局。

        参数:
            num_reqs: 请求（即线程块）数量
            num_tokens: token 数量
            input_buffers: 持久化的输入缓冲区

        返回:
            InputBatch: 虚拟的输入批次
        """
        assert 0 < num_reqs <= num_tokens
        device = input_buffers.device

        req_ids = [f"req_{i}_{random_uuid()}" for i in range(num_reqs)]
        idx_mapping_np = np.arange(num_reqs, dtype=np.int32)
        idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=device)
        expanded_idx_mapping = idx_mapping
        expanded_local_pos = torch.zeros(num_reqs, dtype=torch.int32, device=device)

        num_scheduled_tokens = np.full(num_reqs, num_tokens // num_reqs, dtype=np.int32)
        num_scheduled_tokens[-1] += num_tokens % num_reqs
        assert int(num_scheduled_tokens.sum()) == num_tokens

        # seq_len equals to query_len
        input_buffers.seq_lens[:num_reqs] = num_tokens // num_reqs
        input_buffers.seq_lens[num_reqs - 1] += num_tokens % num_reqs
        # Pad for full CUDA graph mode.
        input_buffers.seq_lens[num_reqs:] = 0
        seq_lens = input_buffers.seq_lens[:num_reqs]

        query_start_loc_np = np.empty(num_reqs + 1, dtype=np.int32)
        query_start_loc_np[0] = 0
        np.cumsum(num_scheduled_tokens, out=query_start_loc_np[1:])
        input_buffers.query_start_loc[:1] = 0
        torch.cumsum(
            seq_lens, dim=0, out=input_buffers.query_start_loc[1 : num_reqs + 1]
        )
        # Pad for full CUDA graph mode.
        input_buffers.query_start_loc[num_reqs + 1 :] = num_tokens
        query_start_loc = input_buffers.query_start_loc[: num_reqs + 1]

        input_ids = input_buffers.input_ids[:num_tokens].zero_()
        positions = input_buffers.positions[:num_tokens].zero_()

        logits_indices = query_start_loc[1:] - 1
        cu_num_logits = torch.arange(num_reqs + 1, device=device, dtype=torch.int32)
        cu_num_logits_np = np.arange(num_reqs + 1, dtype=np.int32)
        # Dummy: seq_len == query_len (fresh-prefill shape).
        seq_lens_cpu_upper_bound = torch.from_numpy(num_scheduled_tokens.copy())
        return cls(
            req_ids=req_ids,
            num_reqs=num_reqs,
            num_reqs_after_padding=num_reqs,
            idx_mapping=idx_mapping,
            idx_mapping_np=idx_mapping_np,
            expanded_idx_mapping=expanded_idx_mapping,
            expanded_local_pos=expanded_local_pos,
            num_scheduled_tokens=num_scheduled_tokens,
            num_tokens=num_tokens,
            num_tokens_after_padding=num_tokens,
            num_draft_tokens=0,
            num_draft_tokens_per_req=None,
            query_start_loc=query_start_loc,
            query_start_loc_np=query_start_loc_np,
            seq_lens=seq_lens,
            seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
            dcp_local_seq_lens=None,
            is_prefilling_np=np.zeros(num_reqs, dtype=np.bool_),
            input_ids=input_ids,
            positions=positions,
            logits_indices=logits_indices,
            cu_num_logits=cu_num_logits,
            cu_num_logits_np=cu_num_logits_np,
            has_structured_output_reqs=False,
        )


@triton.jit
def _prepare_prefill_inputs_kernel(
    input_ids_ptr,
    next_prefill_tokens_ptr,
    idx_mapping_ptr,
    query_start_loc_ptr,
    all_token_ids_ptr,
    all_token_ids_stride,
    prefill_lens_ptr,
    num_computed_tokens_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    """准备预填充输入的 Triton 内核。

    对于每个请求，将需要预填充的 token 从全局 token ID 数组拷贝到
    输入缓冲区。同时设置下一个预填充 token（用于后续的采样）。

    工作流程：
    1. 获取请求的预填充长度和已计算 token 数
    2. 如果已全部计算完成，跳过（非预填充请求）
    3. 从 all_token_ids 中拷贝 [num_computed, num_computed + query_len) 的 token
    4. 如果还有更多 token 需要预填充，设置 next_prefill_token

    参数:
        input_ids_ptr: 输出输入 token ID 的指针
        next_prefill_tokens_ptr: 输出下一个预填充 token 的指针
        idx_mapping_ptr: 批次索引到请求状态索引的映射
        query_start_loc_ptr: 查询起始位置
        all_token_ids_ptr: 所有请求的完整 token ID
        all_token_ids_stride: all_token_ids 的行步进
        prefill_lens_ptr: 每个请求的预填充长度
        num_computed_tokens_ptr: 每个请求已计算的 token 数
        BLOCK_SIZE: Triton 块大小
    """
    batch_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + batch_idx)
    prefill_len = tl.load(prefill_lens_ptr + req_state_idx)
    num_computed = tl.load(num_computed_tokens_ptr + req_state_idx)
    if num_computed >= prefill_len:
        # Not prefill.
        return

    query_start = tl.load(query_start_loc_ptr + batch_idx)
    query_end = tl.load(query_start_loc_ptr + batch_idx + 1)
    query_len = query_end - query_start

    request_ptr = all_token_ids_ptr + req_state_idx * all_token_ids_stride
    for i in range(0, query_len, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < query_len
        tokens = tl.load(request_ptr + num_computed + block, mask=mask)
        tl.store(input_ids_ptr + query_start + block, tokens, mask=mask)

    next_pos = num_computed + query_len
    if next_pos < prefill_len:
        next_token = tl.load(request_ptr + next_pos)
        tl.store(next_prefill_tokens_ptr + req_state_idx, next_token)


def prepare_prefill_inputs(
    input_ids: torch.Tensor,
    next_prefill_tokens: torch.Tensor,
    idx_mapping: torch.Tensor,
    query_start_loc: torch.Tensor,
    all_token_ids: torch.Tensor,
    prefill_len: torch.Tensor,
    num_computed_tokens: torch.Tensor,
) -> None:
    """准备预填充输入的 Python 接口。

    启动 _prepare_prefill_inputs_kernel 内核来并行处理所有请求。

    参数:
        input_ids: 输出的输入 token ID 张量
        next_prefill_tokens: 输出的下一个预填充 token 张量
        idx_mapping: 批次索引到请求状态索引的映射
        query_start_loc: 查询起始位置
        all_token_ids: 所有请求的完整 token ID
        prefill_len: 每个请求的预填充长度
        num_computed_tokens: 每个请求已计算的 token 数
    """
    num_reqs = idx_mapping.shape[0]
    _prepare_prefill_inputs_kernel[(num_reqs,)](
        input_ids,
        next_prefill_tokens,
        idx_mapping,
        query_start_loc,
        all_token_ids,
        all_token_ids.stride(0),
        prefill_len,
        num_computed_tokens,
        BLOCK_SIZE=1024,
    )


@triton.jit
def _prepare_pos_seq_lens_kernel(
    pos_ptr,
    seq_lens_ptr,
    idx_mapping_ptr,
    query_start_loc_ptr,
    num_computed_tokens_ptr,
    max_num_reqs,
    BLOCK_SIZE: tl.constexpr,
):
    """准备位置编码和序列长度的 Triton 内核。

    为每个请求计算：
    1. 位置编码：从 num_computed_tokens 开始递增
    2. 序列长度：num_computed_tokens + query_len

    最后一个线程块负责将未使用的 seq_lens 填充为 0（用于 FULL CUDA Graph）。

    参数:
        pos_ptr: 输出位置编码的指针
        seq_lens_ptr: 输出序列长度的指针
        idx_mapping_ptr: 批次索引到请求状态索引的映射
        query_start_loc_ptr: 查询起始位置
        num_computed_tokens_ptr: 每个请求已计算的 token 数
        max_num_reqs: 最大请求数（缓冲区大小）
        BLOCK_SIZE: Triton 块大小
    """
    req_id = tl.program_id(0)
    num_reqs = tl.num_programs(0) - 1
    if req_id == num_reqs:
        # Pad unused seq_lens as 0 for full CUDA graphs.
        for i in tl.range(num_reqs, max_num_reqs, BLOCK_SIZE):
            block = i + tl.arange(0, BLOCK_SIZE)
            mask = block < max_num_reqs
            tl.store(seq_lens_ptr + block, 0, mask=mask)
        return

    req_state_idx = tl.load(idx_mapping_ptr + req_id)
    num_computed_tokens = tl.load(num_computed_tokens_ptr + req_state_idx)

    start = tl.load(query_start_loc_ptr + req_id)
    end = tl.load(query_start_loc_ptr + req_id + 1)
    query_len = end - start

    seq_len = num_computed_tokens + query_len
    tl.store(seq_lens_ptr + req_id, seq_len)

    for i in tl.range(0, query_len, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < query_len
        pos = num_computed_tokens + block
        tl.store(pos_ptr + start + block, pos, mask=mask)


def prepare_pos_seq_lens(
    idx_mapping: torch.Tensor,
    query_start_loc: torch.Tensor,
    num_computed_tokens: torch.Tensor,
    pos: torch.Tensor,
    seq_lens: torch.Tensor,
) -> None:
    """准备位置编码和序列长度的 Python 接口。

    参数:
        idx_mapping: 批次索引到请求状态索引的映射
        query_start_loc: 查询起始位置
        num_computed_tokens: 每个请求已计算的 token 数
        pos: 输出位置编码
        seq_lens: 输出序列长度
    """
    num_reqs = idx_mapping.shape[0]
    # NOTE(woosuk): We do +1 because the last thread block is used
    # to pad unused seq_lens as 0 for full CUDA graphs.
    _prepare_pos_seq_lens_kernel[(num_reqs + 1,)](
        pos,
        seq_lens,
        idx_mapping,
        query_start_loc,
        num_computed_tokens,
        seq_lens.shape[0],
        BLOCK_SIZE=1024,
    )


@triton.jit
def _combine_sampled_and_draft_tokens_kernel(
    input_ids_ptr,
    idx_mapping_ptr,
    last_sampled_tokens_ptr,
    query_start_loc_ptr,
    seq_lens_ptr,
    prefill_len_ptr,
    draft_tokens_ptr,
    draft_tokens_stride,
    cu_num_logits_ptr,
    logits_indices_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    """合并采样和 draft token 的 Triton 内核。

    在投机解码场景下，每个请求的输入可能包含：
    - 1 个上一步采样的 token
    - N 个 draft token（来自投机解码器）

    该内核负责：
    1. 计算 logits 索引（哪些位置需要计算 logits）
    2. 将上一步采样的 token 写入 input_ids
    3. 将 draft token 写入 input_ids

    参数:
        input_ids_ptr: 输入 token ID 指针
        idx_mapping_ptr: 批次索引到请求状态索引的映射
        last_sampled_tokens_ptr: 上一步采样的 token
        query_start_loc_ptr: 查询起始位置
        seq_lens_ptr: 序列长度
        prefill_len_ptr: 预填充长度
        draft_tokens_ptr: draft token
        draft_tokens_stride: draft token 的行步进
        cu_num_logits_ptr: num_logits 的前缀和
        logits_indices_ptr: 输出 logits 索引
        BLOCK_SIZE: Triton 块大小
    """
    batch_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + batch_idx)

    # 获取 logits 和 draft token 的数量
    cu_num_logits_start = tl.load(cu_num_logits_ptr + batch_idx)
    cu_num_logits_end = tl.load(cu_num_logits_ptr + batch_idx + 1)
    num_logits = cu_num_logits_end - cu_num_logits_start
    num_draft_tokens = num_logits - 1

    # 计算 logits 索引
    block = tl.arange(0, BLOCK_SIZE)
    query_end = tl.load(query_start_loc_ptr + batch_idx + 1)
    logits_start = query_end - num_logits
    tl.store(
        logits_indices_ptr + cu_num_logits_start + block,
        logits_start + block,
        mask=block < num_logits,
    )

    seq_len = tl.load(seq_lens_ptr + batch_idx)
    prefill_len = tl.load(prefill_len_ptr + req_state_idx)
    if seq_len <= prefill_len:
        # 处理预填充 token，无需采样或 draft token
        return

    # 将上一步采样的 token ID 写入 input_ids
    last_token_id = tl.load(last_sampled_tokens_ptr + req_state_idx)
    tl.store(input_ids_ptr + query_end - num_logits, last_token_id)

    # 将 draft token（如果有）写入 input_ids
    if num_draft_tokens > 0:
        mask = block < num_draft_tokens
        draft_tokens = tl.load(
            draft_tokens_ptr + req_state_idx * draft_tokens_stride + block,
            mask=mask,
        )
        tl.store(
            input_ids_ptr + query_end - num_draft_tokens + block,
            draft_tokens,
            mask=mask,
        )


def combine_sampled_and_draft_tokens(
    input_ids: torch.Tensor,
    idx_mapping: torch.Tensor,
    last_sampled_tokens: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    prefill_len: torch.Tensor,
    draft_tokens: torch.Tensor,
    cu_num_logits: torch.Tensor,
    num_logits: int,
) -> torch.Tensor:
    """合并采样和 draft token 的 Python 接口。

    参数:
        input_ids: 输入 token ID 张量
        idx_mapping: 批次索引到请求状态索引的映射
        last_sampled_tokens: 上一步采样的 token
        query_start_loc: 查询起始位置
        seq_lens: 序列长度
        prefill_len: 预填充长度
        draft_tokens: draft token
        cu_num_logits: num_logits 的前缀和
        num_logits: 总的 logits 数量

    返回:
        torch.Tensor: logits 索引
    """
    # use idx_mapping.shape[0] for actual request count
    num_reqs = idx_mapping.shape[0]
    num_speculative_steps = draft_tokens.shape[-1]

    logits_indices = torch.empty(
        num_logits,
        dtype=torch.int64,
        device=input_ids.device,
    )
    _combine_sampled_and_draft_tokens_kernel[(num_reqs,)](
        input_ids,
        idx_mapping,
        last_sampled_tokens,
        query_start_loc,
        seq_lens,
        prefill_len,
        draft_tokens,
        draft_tokens.stride(0),
        cu_num_logits,
        logits_indices,
        # NOTE(woosuk): Add 1 to ensure the block can cover the last sampled token
        # in addition to all draft tokens.
        BLOCK_SIZE=triton.next_power_of_2(num_speculative_steps + 1),
    )
    return logits_indices


@triton.jit
def _get_num_sampled_and_rejected_kernel(
    num_sampled_ptr,
    num_rejected_ptr,
    seq_lens_ptr,
    cu_num_logits_ptr,
    idx_mapping_ptr,
    prefill_len_ptr,
):
    """计算采样和拒绝的 token 数的 Triton 内核。

    在投机解码中：
    - num_sampled: 实际被接受的 token 数（包括最后的采样 token）
    - num_rejected: 被拒绝的 draft token 数
    - num_logits = num_sampled + num_rejected

    对于分块预填充请求，num_sampled 和 num_rejected 都设为 0。

    参数:
        num_sampled_ptr: 输出采样 token 数的指针
        num_rejected_ptr: 输出拒绝 token 数的指针
        seq_lens_ptr: 序列长度
        cu_num_logits_ptr: num_logits 的前缀和
        idx_mapping_ptr: 批次索引到请求状态索引的映射
        prefill_len_ptr: 预填充长度
    """
    batch_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + batch_idx)

    seq_len = tl.load(seq_lens_ptr + batch_idx)
    prefill_len = tl.load(prefill_len_ptr + req_state_idx)
    is_chunked_prefilling = seq_len < prefill_len

    num_sampled = tl.load(num_sampled_ptr + batch_idx)
    num_sampled = tl.where(is_chunked_prefilling, 0, num_sampled)
    tl.store(num_sampled_ptr + batch_idx, num_sampled)

    logits_start = tl.load(cu_num_logits_ptr + batch_idx)
    logits_end = tl.load(cu_num_logits_ptr + batch_idx + 1)
    num_logits = logits_end - logits_start

    num_rejected = num_logits - num_sampled
    num_rejected = tl.where(is_chunked_prefilling, 0, num_rejected)
    tl.store(num_rejected_ptr + batch_idx, num_rejected)


def get_num_sampled_and_rejected(
    num_sampled: torch.Tensor,
    seq_lens: torch.Tensor,
    cu_num_logits: torch.Tensor,
    idx_mapping: torch.Tensor,
    prefill_len: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """计算采样和拒绝的 token 数的 Python 接口。

    参数:
        num_sampled: 采样 token 数（会被原地修改）
        seq_lens: 序列长度
        cu_num_logits: num_logits 的前缀和
        idx_mapping: 批次索引到请求状态索引的映射
        prefill_len: 预填充长度

    返回:
        (num_sampled, num_rejected): 采样和拒绝的 token 数
    """
    num_reqs = idx_mapping.shape[0]
    num_rejected = torch.empty_like(num_sampled)
    _get_num_sampled_and_rejected_kernel[(num_reqs,)](
        num_sampled,
        num_rejected,
        seq_lens,
        cu_num_logits,
        idx_mapping,
        prefill_len,
    )
    return num_sampled, num_rejected


@triton.jit
def _post_update_kernel(
    idx_mapping_ptr,
    num_computed_tokens_ptr,
    last_sampled_tokens_ptr,
    output_bin_counts_ptr,
    output_bin_counts_stride,
    sampled_tokens_ptr,
    sampled_tokens_stride,
    num_sampled_ptr,
    num_rejected_ptr,
    query_start_loc_ptr,
    all_token_ids_ptr,
    all_token_ids_stride,
    total_len_ptr,
):
    """后处理更新的 Triton 内核。

    在采样完成后执行以下更新：
    1. 更新 last_sampled_tokens（最后一个采样的 token ID）
    2. 更新 total_len（请求的总长度）
    3. 将采样的 token 追加到 all_token_ids
    4. 更新 output_bin_counts（词频统计，用于频率惩罚）
    5. 更新 num_computed_tokens（已计算的 token 数）

    参数:
        idx_mapping_ptr: 批次索引到请求状态索引的映射
        num_computed_tokens_ptr: 已计算的 token 数
        last_sampled_tokens_ptr: 上一步采样的 token
        output_bin_counts_ptr: 输出 token 频率统计
        output_bin_counts_stride: 输出频率统计的步进
        sampled_tokens_ptr: 采样的 token
        sampled_tokens_stride: 采样 token 的步进
        num_sampled_ptr: 采样的 token 数
        num_rejected_ptr: 拒绝的 token 数
        query_start_loc_ptr: 查询起始位置
        all_token_ids_ptr: 所有请求的完整 token ID
        all_token_ids_stride: all_token_ids 的步进
        total_len_ptr: 请求的总长度
    """
    req_id = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + req_id)

    total_len = tl.load(total_len_ptr + req_state_idx)
    num_sampled = tl.load(num_sampled_ptr + req_id)
    if num_sampled > 0:
        token_id = tl.load(
            sampled_tokens_ptr + req_id * sampled_tokens_stride + num_sampled - 1
        )
        tl.store(last_sampled_tokens_ptr + req_state_idx, token_id)
        tl.store(total_len_ptr + req_state_idx, total_len + num_sampled)

    for i in range(num_sampled):
        token_id = tl.load(sampled_tokens_ptr + req_id * sampled_tokens_stride + i)
        tl.store(
            all_token_ids_ptr + req_state_idx * all_token_ids_stride + total_len + i,
            token_id,
        )

        if output_bin_counts_ptr is not None:
            token_ptr = (
                output_bin_counts_ptr
                + req_state_idx * output_bin_counts_stride
                + token_id
            )
            count = tl.load(token_ptr)
            tl.store(token_ptr, count + 1)

    query_start = tl.load(query_start_loc_ptr + req_id)
    query_end = tl.load(query_start_loc_ptr + req_id + 1)
    query_len = query_end - query_start
    num_rejected = tl.load(num_rejected_ptr + req_id)

    num_computed = tl.load(num_computed_tokens_ptr + req_state_idx)
    num_computed += query_len - num_rejected
    tl.store(num_computed_tokens_ptr + req_state_idx, num_computed)


def post_update(
    # [num_reqs]
    idx_mapping: torch.Tensor,
    # [max_num_reqs]
    num_computed_tokens: torch.Tensor,
    # [max_num_reqs]
    last_sampled_tokens: torch.Tensor,
    # [max_num_reqs, vocab_size]
    output_bin_counts: torch.Tensor | None,
    # [num_reqs, num_speculative_steps + 1]
    sampled_tokens: torch.Tensor,
    # [num_reqs]
    num_sampled: torch.Tensor,
    # [num_reqs]
    num_rejected: torch.Tensor,
    # [num_reqs + 1]
    query_start_loc: torch.Tensor,
    # [max_num_reqs, max_model_len]
    all_token_ids: torch.Tensor,
    # [max_num_reqs]
    total_len: torch.Tensor,
) -> None:
    """后处理更新的 Python 接口。

    在采样完成后调用，更新请求状态。

    参数:
        idx_mapping: 批次索引到请求状态索引的映射
        num_computed_tokens: 已计算的 token 数
        last_sampled_tokens: 上一步采样的 token
        output_bin_counts: 输出 token 频率统计
        sampled_tokens: 采样的 token
        num_sampled: 采样的 token 数
        num_rejected: 拒绝的 token 数
        query_start_loc: 查询起始位置
        all_token_ids: 所有请求的完整 token ID
        total_len: 请求的总长度
    """
    num_reqs = idx_mapping.shape[0]
    _post_update_kernel[(num_reqs,)](
        idx_mapping,
        num_computed_tokens,
        last_sampled_tokens,
        output_bin_counts,
        output_bin_counts.stride(0) if output_bin_counts is not None else 0,
        sampled_tokens,
        sampled_tokens.stride(0),
        num_sampled,
        num_rejected,
        query_start_loc,
        all_token_ids,
        all_token_ids.stride(0),
        total_len,
        num_warps=1,
    )


@triton.jit
def _post_update_pool_kernel(
    idx_mapping_ptr,
    num_computed_tokens_ptr,
    query_start_loc_ptr,
):
    """池化模型的后处理更新内核。

    对于池化模型（如嵌入模型），只需更新已计算的 token 数。

    参数:
        idx_mapping_ptr: 批次索引到请求状态索引的映射
        num_computed_tokens_ptr: 已计算的 token 数
        query_start_loc_ptr: 查询起始位置
    """
    batch_id = tl.program_id(0)
    query_start = tl.load(query_start_loc_ptr + batch_id)
    query_end = tl.load(query_start_loc_ptr + batch_id + 1)
    query_len = query_end - query_start

    req_state_idx = tl.load(idx_mapping_ptr + batch_id)
    num_computed = tl.load(num_computed_tokens_ptr + req_state_idx)
    tl.store(num_computed_tokens_ptr + req_state_idx, num_computed + query_len)


def post_update_pool(
    # [num_reqs]
    idx_mapping: torch.Tensor,
    # [max_num_reqs]
    num_computed_tokens: torch.Tensor,
    # [num_reqs + 1]
    query_start_loc: torch.Tensor,
) -> None:
    """池化模型的后处理更新 Python 接口。

    参数:
        idx_mapping: 批次索引到请求状态索引的映射
        num_computed_tokens: 已计算的 token 数
        query_start_loc: 查询起始位置
    """
    num_reqs = idx_mapping.shape[0]
    _post_update_pool_kernel[(num_reqs,)](
        idx_mapping,
        num_computed_tokens,
        query_start_loc,
    )


@triton.jit
def _expand_idx_mapping_kernel(
    idx_mapping_ptr,
    expanded_idx_mapping_ptr,
    expanded_local_pos_ptr,
    cu_num_logits_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    """展开索引映射的 Triton 内核。

    在投机解码中，每个请求可能产生多个 logits。此内核将
    请求级别的索引映射展开为 logits 级别的映射。

    例如，如果请求 0 有 3 个 logits，请求 1 有 2 个 logits：
    - expanded_idx_mapping = [0, 0, 0, 1, 1]
    - expanded_local_pos = [0, 1, 2, 0, 1]

    参数:
        idx_mapping_ptr: 请求级别的索引映射
        expanded_idx_mapping_ptr: 输出展开后的索引映射
        expanded_local_pos_ptr: 输出本地位置
        cu_num_logits_ptr: num_logits 的前缀和
        BLOCK_SIZE: Triton 块大小
    """
    req_idx = tl.program_id(0)
    start_idx = tl.load(cu_num_logits_ptr + req_idx)
    end_idx = tl.load(cu_num_logits_ptr + req_idx + 1)
    num_tokens = end_idx - start_idx

    block = tl.arange(0, BLOCK_SIZE)
    mask = block < num_tokens
    req_state_idx = tl.load(idx_mapping_ptr + req_idx)
    tl.store(expanded_idx_mapping_ptr + start_idx + block, req_state_idx, mask=mask)
    tl.store(expanded_local_pos_ptr + start_idx + block, block, mask=mask)


def expand_idx_mapping(
    idx_mapping: torch.Tensor,
    total_num_logits: int,
    cu_num_logits: torch.Tensor,
    max_expand_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """展开索引映射的 Python 接口。

    参数:
        idx_mapping: 请求级别的索引映射
        total_num_logits: 总的 logits 数量
        cu_num_logits: num_logits 的前缀和
        max_expand_len: 最大的单请求 logits 数（用于确定 BLOCK_SIZE）

    返回:
        (expanded_idx_mapping, expanded_local_pos): 展开后的索引映射和本地位置
    """
    num_reqs = idx_mapping.shape[0]
    expanded_idx_mapping = idx_mapping.new_empty(total_num_logits)
    expanded_local_pos = torch.empty(
        total_num_logits, dtype=torch.int32, device=idx_mapping.device
    )
    _expand_idx_mapping_kernel[(num_reqs,)](
        idx_mapping,
        expanded_idx_mapping,
        expanded_local_pos,
        cu_num_logits,
        BLOCK_SIZE=triton.next_power_of_2(max_expand_len),
    )
    return expanded_idx_mapping, expanded_local_pos
