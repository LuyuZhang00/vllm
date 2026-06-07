# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# =============================================================================
# utils.py - 投机解码工具函数和 Triton kernel 模块
#
# 【模块功能概述】
# 本模块提供了投机解码（Speculative Decoding）中使用的各种工具函数和 GPU kernel。
# 主要功能包括：
#   1. EAGLE 自回归步骤中的 slot mapping 和元数据更新（融合 Triton kernel）
#   2. 投机解码输入准备（计算 token 索引、被拒绝 token 数量等）
#   3. 下一个 token ID 的准备（处理被丢弃请求、备份 token 等）
#   4. 通用注意力元数据的扩展（增加 query 长度、更新 slot mapping 等）
#   5. DFlash 并行草稿生成的输入准备（融合 Triton kernel）
#   6. 异步投机解码的元数据校正
#   7. 无条件接受率到条件接受率的转换
#
# 【关键设计】
# - 使用 Triton JIT 编译的 kernel 实现 GPU 上的融合操作，避免多次 kernel launch
# - 使用预分配的缓冲区避免动态内存分配
# - 支持 CUDA Graph（通过 PADDING_SLOT_ID 处理 padding 位置）
# - 所有 kernel 都支持动态 batch 大小
#
# 【在推测解码链路中的位置】
#   Scheduler -> Proposer(EAGLE/DFlash/...) -> 本模块的 kernel 准备输入 ->
#   Model Runner 执行 forward -> 验证/拒绝 -> 更新元数据
# =============================================================================

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import (
    CommonAttentionMetadata,
)

# 中文注释：padding slot 的特殊 ID，用于标识 KV cache 中不需要写入的位置。
# 在 slot mapping 中，-1 表示该 token 不需要写入 KV cache（padding 用途）。
# 这在 CUDA Graph 捕获时特别重要，因为 CUDA Graph 需要固定的 tensor 形状，
# 但实际 batch 中的 token 数量可能小于最大值，多余的位置用 PADDING_SLOT_ID 填充。
PADDING_SLOT_ID = -1


def next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 >= n."""
    # 中文注释：计算大于等于 n 的最小 2 的幂。
    # 在 Triton kernel 中，BLOCK_SIZE 通常需要是 2 的幂，
    # 此函数用于计算合适的 block 大小。
    # 算法：先减 1（处理 n 已经是 2 的幂的情况），然后通过位或操作
    # 将最高位以下的所有位都置为 1，最后加 1 得到下一个 2 的幂。
    if n <= 0:
        return 1
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n |= n >> 32
    return n + 1


@triton.jit
def eagle_step_slot_mapping_metadata_kernel(
    positions_ptr,  # [batch_size] - current positions (1D view for M-RoPE)
    block_table_ptr,  # [batch_size, n_blocks_per_req]
    block_table_stride,  # stride for block_table dim 1
    seq_lens_ptr,  # [batch_size] - read and write
    out_clamped_positions_ptr,  # [batch_size] (output)
    out_slot_mapping_ptr,  # [input_batch_size] (output)
    block_size: tl.constexpr,
    max_model_len: tl.constexpr,
    n_blocks_per_req: tl.constexpr,
    PAD_ID: tl.constexpr,
    batch_size,
):
    """
    Fused kernel for EAGLE autoregressive step: updates positions, slot mapping,
    and sequence lengths in a single kernel to reduce launch overhead.

    Launched with input_batch_size threads. Threads with req_idx >= batch_size
    are cudagraph padding slots and only write PADDING_SLOT_ID.

    Each real thread handles one request in the batch. Computes:
    - new_position = position + 1, clamped if exceeds max_model_len
    - slot_mapping from block table lookup
    - seq_lens += 1, or 1 if position exceeds max
    """
    # 中文注释：EAGLE 自回归步骤的融合 Triton kernel。
    #
    # 【功能】
    # 在一次 kernel 调用中完成三个操作（减少 kernel launch 开销）：
    #   1. 更新位置编码：position += 1（如果超过 max_model_len 则 clamp 到 0）
    #   2. 计算 slot mapping：通过 block table 查找物理 KV cache 地址
    #   3. 更新序列长度：seq_lens += 1（如果超过 max_model_len 则重置为 1）
    #
    # 【启动方式】
    # 使用 input_batch_size 个线程启动，每个线程处理一个请求。
    # 超出 batch_size 的线程是 CUDA Graph padding，只写入 PADDING_SLOT_ID。
    #
    # 【slot mapping 计算公式】
    #   block_number = position // block_size  （逻辑 block 索引）
    #   block_id = block_table[req_idx, block_number]  （物理 block 索引）
    #   slot_id = block_id * block_size + (position % block_size)  （物理 slot 地址）
    #
    # 【为什么需要这个 kernel】
    # EAGLE 在自回归生成草稿 token 时，每一步都需要更新位置和 slot mapping。
    # 如果用多个独立 kernel，每次 launch 都有开销；融合后只需一次 launch。
    req_idx = tl.program_id(0)

    if req_idx >= batch_size:
        tl.store(out_slot_mapping_ptr + req_idx, PAD_ID)
        return

    # Load current position and increment
    position = tl.load(positions_ptr + req_idx)
    new_position = position + 1

    # Check bounds and compute clamped position
    exceeds_max = new_position >= max_model_len
    clamped_position = tl.where(exceeds_max, 0, new_position)

    # Block table lookup: block_number = position // block_size
    # Clamp block_number to avoid OOB when position is at max
    block_number = clamped_position // block_size
    block_number = tl.minimum(block_number, n_blocks_per_req - 1)

    block_id = tl.load(block_table_ptr + req_idx * block_table_stride + block_number)
    slot_id = block_id * block_size + (clamped_position % block_size)
    slot_id = tl.where(exceeds_max, PAD_ID, slot_id)

    # Update seq_lens: +1 normally, or 1 if exceeded
    seq_len = tl.load(seq_lens_ptr + req_idx)
    new_seq_len = tl.where(exceeds_max, 1, seq_len + 1)
    new_seq_len = tl.minimum(new_seq_len, max_model_len)

    # Store outputs
    tl.store(out_clamped_positions_ptr + req_idx, clamped_position)
    tl.store(out_slot_mapping_ptr + req_idx, slot_id)
    tl.store(seq_lens_ptr + req_idx, new_seq_len)


def eagle_step_update_slot_mapping_and_metadata(
    positions_1d: torch.Tensor,
    block_table_tensor: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int,
    max_model_len: int,
    out_clamped_positions: torch.Tensor,
    out_slot_mapping: torch.Tensor,
    input_batch_size: int | None = None,
) -> None:
    """
    Fused update of slot mapping and metadata for one EAGLE autoregressive step.
    Updates seq_lens in place. Writes to out_clamped_positions and out_slot_mapping.

    When input_batch_size > batch_size, threads beyond batch_size write
    PADDING_SLOT_ID to out_slot_mapping for cudagraph padding.

    Args:
        positions_1d: [batch_size] current positions (use positions[0] for M-RoPE)
        block_table_tensor: [batch_size, n_blocks_per_req]
        seq_lens: [batch_size] updated in place
        block_size: KV cache block size
        max_model_len: max model length for clamping
        out_clamped_positions: [batch_size] output buffer for clamped positions
        out_slot_mapping: [input_batch_size] output buffer for slot mapping
        input_batch_size: total batch size including cudagraph padding;
            defaults to batch_size (no padding)
    """
    # 中文注释：EAGLE 自回归步骤的融合更新函数。
    #
    # 【功能】
    # 调用 eagle_step_slot_mapping_metadata_kernel，在一次 kernel 调用中完成：
    #   1. 位置更新：position += 1（clamp 到 [0, max_model_len)）
    #   2. slot mapping 计算：通过 block table 查找物理 KV cache 地址
    #   3. 序列长度更新：seq_lens += 1
    #
    # 【参数说明】
    # - positions_1d: 当前位置（1D 视图，M-RoPE 时取第一个维度）
    # - block_table_tensor: 逻辑 block -> 物理 block 的映射表
    # - seq_lens: 序列长度（原地更新）
    # - input_batch_size: 包含 CUDA Graph padding 的总 batch 大小
    #
    # 【调用时机】
    # 在 EAGLE 自回归生成草稿 token 的每一步中调用，
    # 由 SpecDecodeBaseProposer._update_positions_dependent_metadata() 触发。
    batch_size = positions_1d.shape[0]
    if input_batch_size is None:
        input_batch_size = batch_size

    n_blocks_per_req = block_table_tensor.shape[1]
    eagle_step_slot_mapping_metadata_kernel[(input_batch_size,)](
        positions_1d,
        block_table_tensor,
        block_table_tensor.stride(0),
        seq_lens,
        out_clamped_positions,
        out_slot_mapping,
        block_size=block_size,
        max_model_len=max_model_len,
        n_blocks_per_req=n_blocks_per_req,
        PAD_ID=PADDING_SLOT_ID,
        batch_size=batch_size,
    )


@triton.jit
def eagle_prepare_inputs_padded_kernel(
    cu_num_draft_tokens_ptr,  # [num_reqs]
    valid_sampled_tokens_count_ptr,  # [num_reqs]
    query_start_loc_gpu_ptr,  # [num_reqs + 1]
    token_indices_to_sample_ptr,  # [num_reqs] (output)
    num_rejected_tokens_gpu_ptr,  # [num_reqs] (output)
    num_reqs,  # tl.int32
):
    """
    Fused kernel for Eagle prepare_input_padded. This kernel computes the
    token index to sample for each request, taking into account the number
    of draft tokens and the number of valid sampled tokens (which is one more than
    the number of accepted tokens).
    """
    # 中文注释：EAGLE 填充批次输入准备的融合 Triton kernel。
    #
    # 【功能】
    # 为每个请求计算两个关键值：
    #   1. token_indices_to_sample: 新采样 token 在扩展序列中的位置索引
    #   2. num_rejected_tokens: 被拒绝的草稿 token 数量
    #
    # 【计算逻辑】
    # 假设某个请求有 N 个草稿 token，验证后接受 A 个（包含新采样的 1 个）：
    #   - num_rejected = N + 1 - A  （N 个草稿 + 1 个 bonus - A 个有效）
    #   - index_to_sample = query_end - num_rejected
    #     （从序列末尾回退 num_rejected 个位置，就是新采样 token 的位置）
    #
    # 【cu_num_draft_tokens 的特殊格式】
    # cu_num_draft_tokens 是"包含式"累积和（第一个元素就是第一个值，不是 0），
    # 所以需要特殊处理 req_idx == 0 的情况。
    #
    # 【调用时机】
    # 由 SpecDecodeBaseProposer.prepare_inputs_padded() 调用，
    # 在投机解码验证阶段之前计算需要采样的位置。
    req_idx = tl.program_id(axis=0)
    if req_idx >= num_reqs:
        return

    # Calculate num_draft_tokens from cu_num_draft_tokens, which is an inclusive
    # cumulative sum (first entry is the first value, not zero).
    cu_draft_curr = tl.load(cu_num_draft_tokens_ptr + req_idx)

    num_draft_tokens = 0
    if req_idx == 0:
        num_draft_tokens = cu_draft_curr
    else:
        cu_draft_prev = tl.load(cu_num_draft_tokens_ptr + req_idx - 1)
        num_draft_tokens = cu_draft_curr - cu_draft_prev

    valid_count = tl.load(valid_sampled_tokens_count_ptr + req_idx)
    num_rejected_tokens = num_draft_tokens + 1 - valid_count
    num_rejected_tokens = tl.where(num_draft_tokens > 0, num_rejected_tokens, 0)

    # query_start_loc[req_idx + 1] is the start position of the next request,
    # which is one past the last token of this request.
    q_last_tok_idx = tl.load(query_start_loc_gpu_ptr + req_idx + 1) - 1

    index_to_sample = q_last_tok_idx - num_rejected_tokens
    tl.store(token_indices_to_sample_ptr + req_idx, index_to_sample)
    tl.store(num_rejected_tokens_gpu_ptr + req_idx, num_rejected_tokens)


@triton.jit
def eagle_prepare_next_token_padded_kernel(
    sampled_token_ids_ptr,  # [num_reqs, num_sampled_tokens_per_req]
    discard_request_mask_ptr,  # [num_reqs]
    backup_next_token_ids_ptr,  # [num_reqs]
    next_token_ids_ptr,  # [num_reqs] (output)
    valid_sampled_tokens_count_ptr,  # [num_reqs] (output)
    vocab_size,  # tl.int32
    num_sampled_tokens_per_req,  # tl.int32 (num_spec_tokens + 1)
    num_reqs,  # tl.int32
    stride_sampled_token_ids,  # tl.int32 (stride for dim 0)
    BLOCK_SIZE_TOKENS: tl.constexpr,  # Power-of-2 >= num_sampled_tokens_per_req
):
    """
    Fused kernel for Eagle prepare_next_token_ids_padded. This kernel computes the
    number of valid (1 + accepted) tokens for each request, and the corresponding
    "next" token id to sample from during speculative decoding. This is the
    "last accepted token" from the sampled tokens, or the backup token if no
    tokens were accepted or if the request is marked as discarded.
    """
    # 中文注释：EAGLE 填充批次下一个 token ID 准备的融合 Triton kernel。
    #
    # 【功能】
    # 为每个请求计算两个值：
    #   1. next_token_ids: 下一个要输入给草稿模型的 token ID
    #   2. valid_sampled_tokens_count: 有效采样 token 的数量（包含被接受的草稿 + 新采样的）
    #
    # 【处理逻辑】
    # 对于每个请求：
    #   - 如果被标记为丢弃（is_discarded）：使用 backup token，有效计数为 0
    #   - 否则：遍历采样 token 序列，统计有效 token 数量（值在 [0, vocab_size) 范围内）
    #     - 如果有有效 token：取最后一个有效 token 作为 next_token_id
    #     - 如果没有有效 token：使用 backup token
    #
    # 【为什么需要 backup token】
    # 当一个请求的所有草稿 token 都被拒绝，且没有新采样到 token 时，
    # 需要一个回退 token 来保证草稿模型有输入。backup token 通常是
    # 该请求在推测解码之前的最后一个已知 token。
    #
    # 【valid_sampled_tokens_count 的含义】
    # 它等于被接受的草稿 token 数 + 1（新采样的 token）。
    # 例如：草稿 token = [A, B, C]，验证后 A 和 B 被接受，C 被拒绝，
    # 新采样 D，则 valid_count = 3（A, B, D）。
    req_idx = tl.program_id(axis=0)
    if req_idx >= num_reqs:
        return

    # Check if this request is discarded.
    is_discarded = tl.load(discard_request_mask_ptr + req_idx)

    if is_discarded:
        backup_token = tl.load(backup_next_token_ids_ptr + req_idx)
        valid_count = tl.full((), 0, dtype=tl.uint32)
        tl.store(next_token_ids_ptr + req_idx, backup_token)
        tl.store(valid_sampled_tokens_count_ptr + req_idx, valid_count)
    else:
        # Count the number of valid tokens among the sampled tokens.
        token_offs = tl.arange(0, BLOCK_SIZE_TOKENS)
        token_mask = token_offs < num_sampled_tokens_per_req

        row_ptr = sampled_token_ids_ptr + req_idx * stride_sampled_token_ids
        token_ids = tl.load(row_ptr + token_offs, mask=token_mask, other=-1)

        # Rejected tokens are -1, valid tokens are in [0, vocab_size)
        is_valid_mask = (token_ids != -1) & (token_ids < vocab_size) & token_mask
        valid_count = tl.sum(is_valid_mask)

        if valid_count > 0:
            # Guaranteed to be well-defined since
            # valid_count > 0 implies is_valid_mask is not empty
            last_valid_index = tl.max(tl.where(is_valid_mask, token_offs, -1))

            # Select the token at that index, using a sum trick since
            # we don't want to load again to access token_ids[last_valid_index].
            last_valid_token = tl.sum(
                tl.where(token_offs == last_valid_index, token_ids, 0)
            )
            tl.store(next_token_ids_ptr + req_idx, last_valid_token)
        else:
            # No valid tokens found, use backup token
            backup_token = tl.load(backup_next_token_ids_ptr + req_idx)
            tl.store(next_token_ids_ptr + req_idx, backup_token)

        tl.store(valid_sampled_tokens_count_ptr + req_idx, valid_count)


def compute_new_slot_mapping(
    cad: CommonAttentionMetadata,
    new_positions: torch.Tensor,
    is_rejected_token_mask: torch.Tensor,
    block_size: int,
    num_new_tokens: int,
    max_model_len: int,
):
    # 中文注释：为扩展后的序列计算新的 slot mapping。
    #
    # 【功能】
    # 当投机解码需要扩展每个请求的 query 长度（增加 bonus token 和 parallel drafting token）时，
    # 需要为这些新 token 计算 slot mapping（逻辑位置 -> 物理 KV cache 地址）。
    #
    # 【计算流程】
    # 1. 为每个 token 确定它属于哪个请求（req_indices）
    # 2. 将位置 clamp 到 [0, max_model_len-1] 范围内
    # 3. 通过 block table 查找物理 block id：
    #    block_number = position // block_size
    #    block_id = block_table[req_idx, block_number]
    # 4. 计算 slot = block_id * block_size + (position % block_size)
    # 5. 将超出 max_model_len 的位置标记为 PADDING_SLOT_ID
    # 6. 将被拒绝的 token 标记为 PADDING_SLOT_ID（避免写入 KV cache）
    #
    # 【调用时机】
    # 由 SpecDecodeBaseProposer.set_inputs_first_pass() 中的"需要额外输入槽位"路径调用，
    # 用于草稿模型和并行草稿生成模式。
    batch_size, n_blocks_per_req = cad.block_table_tensor.shape
    req_indices = torch.arange(batch_size, device=cad.query_start_loc.device)
    req_indices = torch.repeat_interleave(
        req_indices,
        cad.naive_query_lens() + num_new_tokens,
        output_size=len(new_positions),
    )
    # Clamp the positions to prevent an out-of-bounds error when indexing
    # into block_table_tensor.
    clamped_positions = torch.clamp(new_positions, max=max_model_len - 1)
    block_table_indices = (
        req_indices * n_blocks_per_req + clamped_positions // block_size
    )
    block_nums = cad.block_table_tensor.view(-1)[block_table_indices]
    block_offsets = clamped_positions % block_size
    new_slot_mapping = block_nums * block_size + block_offsets
    # Mask out the position ids that exceed the max model length.
    exceeds_max_model_len = new_positions >= max_model_len
    new_slot_mapping.masked_fill_(exceeds_max_model_len, PADDING_SLOT_ID)
    # Mask out rejected tokens to prevent saves to the KV cache.
    new_slot_mapping.masked_fill_(is_rejected_token_mask, PADDING_SLOT_ID)
    return new_slot_mapping


def extend_all_queries_by_N(
    common_attn_metadata: CommonAttentionMetadata,
    N: int,
    arange: torch.Tensor,
    new_slot_mapping: torch.Tensor,
) -> CommonAttentionMetadata:
    """
    Creates a new CommonAttentionMetadata with all query lengths increased by N.
    Also all seq lens are increased by N.
    This is useful e.g. in speculative decoding with parallel drafting, where we
    extend each sequence by N tokens and predict all tokens in one pass.
    The slot mapping is computed externally, as it requires more information.
    """
    # 中文注释：扩展所有请求的 query 长度，增加 N 个 token。
    #
    # 【功能】
    # 创建一个新的 CommonAttentionMetadata，其中：
    #   - 每个请求的 query 长度增加 N
    #   - 每个请求的序列长度增加 N
    #   - 总 token 数增加 batch_size * N
    #   - 使用外部计算的新 slot mapping
    #
    # 【使用场景】
    # 并行草稿生成（parallel drafting）中，每个请求需要额外 N 个 slot
    # 来容纳 bonus token 和 mask token。此函数将这些额外 slot 正确地
    # 注入到注意力元数据中。
    #
    # 【query_start_loc 的更新方式】
    # 原始: [0, q1, q1+q2, q1+q2+q3]
    # 更新后: [0, q1+N, q1+q2+2N, q1+q2+q3+3N]
    # 即第 i 个请求的起始位置偏移 i*N，因为前面 i 个请求各增加了 N 个 token。
    cad = common_attn_metadata
    # query start loc must be increased by [+0, +N, +2N, ..., +batch_size * N]
    new_query_start_loc = cad.query_start_loc + N * arange[: len(cad.query_start_loc)]
    new_query_start_loc_cpu = cad.query_start_loc_cpu + N * torch.arange(
        len(cad.query_start_loc_cpu), dtype=torch.int32
    )
    new_cad = cad.replace(
        query_start_loc=new_query_start_loc,
        query_start_loc_cpu=new_query_start_loc_cpu,
        seq_lens=cad.seq_lens + N,
        # each request is extended by N tokens -> batch_size * N tokens are added
        num_actual_tokens=cad.num_actual_tokens + cad.batch_size() * N,
        # All query lens increase by N, so max query len increases by N
        max_query_len=cad.max_query_len + N,
        max_seq_len=cad.max_seq_len + N,
        slot_mapping=new_slot_mapping,
    )
    return new_cad


# Unified copy/expand kernel
@triton.jit
def copy_and_expand_eagle_inputs_kernel(
    # (Padded) Inputs from the target model
    target_token_ids_ptr,  # [total_tokens_in_batch]
    target_positions_ptr,  # [total_tokens_in_batch]
    next_token_ids_ptr,  # [num_reqs]
    # Outputs to the drafting buffers
    out_input_ids_ptr,  # [total_draft_tokens_in_batch] (output)
    out_positions_ptr,  # [total_draft_tokens_in_batch] (output)
    out_is_rejected_token_mask_ptr,  # [total_draft_tokens_in_batch] (output)
    out_is_masked_token_mask_ptr,  # [total_draft_tokens_in_batch] (output)
    out_new_token_indices_ptr,  # [num_padding_slots_per_request * num_reqs] (output)
    out_hidden_state_mapping_ptr,  # [total_tokens_in_batch]
    # Input metadata
    query_start_loc_ptr,  # [num_reqs + 1], last value is the total num input tokens
    query_end_loc_ptr,  # [num_reqs]
    padding_token_id,  # tl.int32
    parallel_drafting_token_id,  # tl.int32
    # Sizing info
    total_input_tokens,  # tl.int32
    num_padding_slots_per_request,  # tl.int32
    shift_input_ids,  # tl.bool
    BLOCK_SIZE_TOKENS: tl.constexpr,  # Blocks along token dim to handle prefills
):
    """
    Copy and expand inputs from the target model to the drafting buffers for Eagle
    speculative decoding. This kernel handles padding slots and parallel drafting
    tokens, if enabled.
    """
    # 中文注释：EAGLE 投机解码的输入复制和扩展融合 Triton kernel。
    #
    # 【功能】
    # 将 target model 的输入（token_ids、positions）复制到草稿模型的输入缓冲区，
    # 同时插入额外的 token（bonus token、parallel drafting mask token、rejected token padding）。
    #
    # 【输出布局】（每个请求）
    # [0, num_valid_tokens): 从 target model 复制的有效 token
    # [num_valid_tokens]: bonus token（target model 采样的下一个 token）
    # (num_valid_tokens, num_valid_tokens + num_padding_slots): parallel drafting mask token
    # [num_valid_tokens + num_padding_slots, total_output): rejected token padding
    #
    # 【shift_input_ids 的含义】
    # 当 shift_input_ids=True 时（EAGLE 方法），输入 token 需要左移一位：
    #   原始: [a1, b1, b2, c1, c2, c3]
    #   移位后: [b1, b2, c1, c2, c3, c3]（最后一个 token 会被 next_token 替换）
    # 这是因为 EAGLE 的草稿模型需要预测"下一个"token，所以输入要比 target 提前一步。
    #
    # 【2D 网格启动】
    # grid = (batch_size, num_blocks)
    # - 第一维：每个请求一个 program
    # - 第二维：每个请求的 token 分成多个 block 处理（支持 prefill 的长序列）
    request_idx = tl.program_id(axis=0)
    token_batch_idx = tl.program_id(axis=1)

    # Load query locations
    query_start_loc = tl.load(query_start_loc_ptr + request_idx)
    next_query_start_loc = tl.load(query_start_loc_ptr + request_idx + 1)
    query_end_loc = tl.load(query_end_loc_ptr + request_idx)

    # Calculate number of valid tokens to copy and input offset
    # With shift_input_ids=True, we skip the first token
    # Output layout: each request gets (input_len + num_padding_slots_per_request) slots
    # But with shift, we lose one token per request
    if shift_input_ids:
        num_valid_tokens = query_end_loc - query_start_loc
        input_offset = 1
        output_start = query_start_loc + request_idx * (
            num_padding_slots_per_request - 1
        )
    else:
        num_valid_tokens = query_end_loc - query_start_loc + 1
        input_offset = 0
        output_start = query_start_loc + request_idx * num_padding_slots_per_request

    # Number of rejected tokens from previous speculation
    num_rejected = next_query_start_loc - query_end_loc - 1

    # Total output tokens for this request
    total_output_tokens = (
        num_valid_tokens + num_padding_slots_per_request + num_rejected
    )

    # Process tokens in this block
    j = token_batch_idx * BLOCK_SIZE_TOKENS + tl.arange(0, BLOCK_SIZE_TOKENS)

    # Compute masks for different output regions:
    # [0, num_valid_tokens): valid tokens copied from input
    # [num_valid_tokens]: bonus token from next_token_ids
    # (num_valid_tokens, num_valid_tokens + num_padding_slots_per_request):
    #     parallel drafting slots
    # [num_valid_tokens + num_padding_slots_per_request, total_output_tokens):
    #     rejected slots
    in_bounds = j < total_output_tokens
    is_valid_region = j < num_valid_tokens
    is_bonus_region = j == num_valid_tokens
    is_parallel_draft_region = (j > num_valid_tokens) & (
        j < num_valid_tokens + num_padding_slots_per_request
    )
    is_rejected_region = j >= num_valid_tokens + num_padding_slots_per_request

    # Compute output indices
    out_idx = output_start + j

    # For valid tokens, compute input index
    in_idx = query_start_loc + input_offset + j
    # Clamp to avoid out-of-bounds access (masked loads still need valid addresses)
    in_idx_clamped = tl.minimum(in_idx, total_input_tokens - 1)

    # Load input tokens (masked to valid region)
    token_ids = tl.load(
        target_token_ids_ptr + in_idx_clamped, mask=is_valid_region & in_bounds, other=0
    )

    # Load the starting position for this request (first position in the sequence)
    start_pos = tl.load(target_positions_ptr + query_start_loc)

    # Load bonus token for this request
    bonus_token = tl.load(next_token_ids_ptr + request_idx)

    # Build final token_ids based on region
    token_ids = tl.where(is_bonus_region, bonus_token, token_ids)
    token_ids = tl.where(
        is_parallel_draft_region, parallel_drafting_token_id, token_ids
    )
    token_ids = tl.where(is_rejected_region, padding_token_id, token_ids)

    # Build final positions:
    # Positions are NOT shifted - they start from the first input position and increment
    # Output position j gets start_pos + j
    # (e.g., input positions [5,6,7] -> output [5,6,7,8,9,...])
    positions = start_pos + j
    # Rejected positions are don't-care, set to 0
    positions = tl.where(is_rejected_region, 0, positions)

    # Compute output masks
    is_rejected_out = is_rejected_region & in_bounds
    is_masked_out = is_parallel_draft_region & in_bounds

    # Compute indices of new tokens (bonus + parallel drafting) for sampling
    # New tokens are at positions
    #     [num_valid_tokens, num_valid_tokens + num_padding_slots_per_request)
    is_new_token_region = (j >= num_valid_tokens) & (
        j < num_valid_tokens + num_padding_slots_per_request
    )
    new_token_local_idx = (
        j - num_valid_tokens
    )  # 0 for bonus, 1, 2, ... for parallel drafting
    new_token_out_idx = (
        request_idx * num_padding_slots_per_request + new_token_local_idx
    )

    # Compute hidden state mapping (source index -> destination index)
    # This maps each input position to its corresponding output position
    # Hidden states don't get shifted, so we map all input tokens (including rejected)
    if shift_input_ids:
        num_input_tokens_this_request = next_query_start_loc - query_start_loc
        is_input_region = j < num_input_tokens_this_request
        src_idx = query_start_loc + j
        tl.store(out_hidden_state_mapping_ptr + src_idx, out_idx, mask=is_input_region)

    # Store outputs
    tl.store(out_input_ids_ptr + out_idx, token_ids, mask=in_bounds)
    tl.store(out_positions_ptr + out_idx, positions, mask=in_bounds)
    tl.store(out_is_rejected_token_mask_ptr + out_idx, is_rejected_out, mask=in_bounds)
    tl.store(out_is_masked_token_mask_ptr + out_idx, is_masked_out, mask=in_bounds)
    tl.store(
        out_new_token_indices_ptr + new_token_out_idx,
        out_idx,
        mask=is_new_token_region & in_bounds,
    )


@triton.jit
def copy_and_expand_dflash_inputs_kernel(
    # Inputs
    next_token_ids_ptr,  # [num_reqs]
    target_positions_ptr,  # [num_context]
    # Outputs
    out_input_ids_ptr,  # [num_query_total] (output)
    out_context_positions_ptr,  # [num_context] (output)
    out_query_positions_ptr,  # [num_query_total] (output)
    out_context_slot_mapping_ptr,  # [num_context] (output)
    out_query_slot_mapping_ptr,  # [num_query_total] (output)
    out_token_indices_ptr,  # [num_reqs * num_speculative_tokens] (output)
    # Block table
    block_table_ptr,  # [max_reqs, max_blocks]
    block_table_stride,  # stride of block_table dim 0 (in elements)
    # Metadata
    query_start_loc_ptr,  # [num_reqs + 1]
    num_rejected_tokens_ptr,  # [num_reqs] or null (0) when not padded
    # Scalars
    parallel_drafting_token_id,  # tl.int32
    block_size,  # tl.int32
    num_query_per_req,  # tl.int32
    num_speculative_tokens,  # tl.int32
    total_input_tokens,  # tl.int32
    BLOCK_SIZE: tl.constexpr,
    HAS_NUM_REJECTED: tl.constexpr = False,
):
    """
    Fused kernel for DFlash first-pass input setup.

    Per request, this kernel:
      1. Copies context positions from target_positions to
         out_context_positions.
      2. Computes query positions (last_target_pos + 1 + offset) and writes
         them to out_query_positions.
      3. Writes input_ids for query tokens: [next_token, mask, mask, ...].
      4. Computes slot_mapping for context and query positions into separate
         buffers via block_table lookup.
      5. Writes token_indices_to_sample for the mask (speculative) tokens.
    """
    # 中文注释：DFlash 第一轮输入准备的融合 Triton kernel。
    #
    # 【功能】
    # 为每个请求完成以下操作：
    #   1. 复制 context positions（target model 的位置编码）
    #   2. 计算 query positions（last_target_pos + 1 + offset）
    #   3. 生成 query input_ids：[next_token, mask_token, mask_token, ...]
    #   4. 通过 block table 计算 context 和 query 的 slot mapping
    #   5. 记录 token_indices_to_sample（mask token 的位置索引）
    #
    # 【DFlash 的 token 分类】
    # - Context token: target model 的 hidden states，投影为 K/V
    # - Query token: bonus token（next_token）+ mask token（speculative positions）
    #
    # 【2D 网格启动】
    # grid = (batch_size, num_blocks)
    # - 第一维：每个请求一个 program
    # - 第二维：每个请求的 token 分成多个 block 处理
    #
    # 【HAS_NUM_REJECTED 的含义】
    # 在 padded 模式下，ctx_end 包含了被拒绝的 token。
    # 需要用 num_rejected 来找到最后一个被接受的 context 位置。
    req_idx = tl.program_id(axis=0)
    block_idx = tl.program_id(axis=1)

    # Load context token range for this request
    ctx_start = tl.load(query_start_loc_ptr + req_idx)
    ctx_end = tl.load(query_start_loc_ptr + req_idx + 1)
    num_ctx = ctx_end - ctx_start
    total_tokens = num_ctx + num_query_per_req

    j = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    in_bounds = j < total_tokens
    is_ctx = j < num_ctx
    is_query = (~is_ctx) & in_bounds
    query_off = j - num_ctx  # offset within query portion (0-indexed)

    # --- Positions ---
    # Context: load from target_positions
    ctx_pos_idx = tl.minimum(ctx_start + j, total_input_tokens - 1)
    ctx_pos = tl.load(target_positions_ptr + ctx_pos_idx, mask=is_ctx, other=0)

    # Query: last_valid_pos + 1 + query_off
    # In padded mode, ctx_end includes rejected tokens; use valid_ctx_end
    # to find the last accepted context position.
    if HAS_NUM_REJECTED:
        num_rejected = tl.load(num_rejected_tokens_ptr + req_idx)
        valid_ctx_end = ctx_end - num_rejected
    else:
        valid_ctx_end = ctx_end
    last_pos = tl.load(target_positions_ptr + valid_ctx_end - 1)
    query_pos = last_pos + 1 + query_off

    positions = tl.where(is_ctx, ctx_pos, query_pos)

    # Context and query positions go to separate buffers.
    ctx_pos_out = ctx_start + j
    tl.store(out_context_positions_ptr + ctx_pos_out, ctx_pos, mask=is_ctx)
    query_out = req_idx * num_query_per_req + query_off
    tl.store(out_query_positions_ptr + query_out, query_pos, mask=is_query)

    # --- Slot mapping (block_table lookup for all positions) ---
    block_num = positions // block_size
    # # Clamp block_number to avoid OOB when position is at max
    block_num = tl.minimum(block_num, block_table_stride - 1)
    block_id = tl.load(
        block_table_ptr + req_idx * block_table_stride + block_num,
        mask=in_bounds,
        other=0,
    ).to(tl.int64)
    slot = block_id * block_size + (positions % block_size)
    tl.store(out_context_slot_mapping_ptr + ctx_pos_out, slot, mask=is_ctx)
    tl.store(out_query_slot_mapping_ptr + query_out, slot, mask=is_query)

    # --- Input IDs (query tokens only) ---
    bonus_token = tl.load(next_token_ids_ptr + req_idx)
    is_bonus = is_query & (query_off == 0)
    input_id = tl.where(is_bonus, bonus_token, parallel_drafting_token_id)
    tl.store(out_input_ids_ptr + query_out, input_id, mask=is_query)

    # --- Token indices to sample (mask tokens, skip the bonus token) ---
    is_sample = is_query & (query_off > 0)
    sample_out_idx = req_idx * num_speculative_tokens + (query_off - 1)
    tl.store(
        out_token_indices_ptr + sample_out_idx,
        query_out,
        mask=is_sample,
    )


@torch.compile(dynamic=True, backend=current_platform.simple_compile_backend)
def update_num_computed_tokens_for_batch_change(
    num_computed_tokens: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    prev_positions: torch.Tensor,
    valid_sampled_token_count: torch.Tensor,
    prev_num_draft_tokens: torch.Tensor,
    cpu_num_computed_tokens: torch.Tensor,
) -> None:
    """Correct num_computed_tokens for async spec decode drift.

    Requests that had drafts: corrected = prev_gpu + valid_count.
    New requests or non-draft (e.g. prefills): use CPU value directly.
    """
    # 中文注释：校正异步投机解码中 num_computed_tokens 的漂移。
    #
    # 【背景】
    # 在异步投机解码中，Scheduler 和 Model Runner 在不同的进程中运行。
    # 当 batch 中的请求集合发生变化（新增、移除、重排序）时，
    # GPU 上的 num_computed_tokens 可能与 CPU 上的不同步。
    #
    # 【校正逻辑】
    # - 对于参与过投机解码的请求（prev_drafts > 0）：
    #   corrected = prev_computed + valid_count
    #   （prev_computed 是上一步的已计算 token 数，valid_count 是本轮接受的 token 数）
    # - 对于新请求或未参与投机解码的请求：
    #   直接使用 CPU 上的值（cpu_num_computed_tokens）
    #
    # 【为什么需要 clamp(min=0)】
    # 新请求的 prev_positions 为 -1（表示不在上一步的 batch 中），
    # clamp 到 0 避免索引越界。
    # Clamp because prev_positions can be -1 for new requests
    gather_indices = prev_positions.clamp(min=0)

    valid_counts = valid_sampled_token_count[gather_indices]
    prev_computed = num_computed_tokens[gather_indices]
    prev_drafts = prev_num_draft_tokens[gather_indices]

    participating = (prev_positions >= 0) & (prev_drafts > 0)
    corrected = prev_computed + valid_counts.int()

    n = prev_positions.shape[0]
    num_computed_tokens[:n].copy_(
        torch.where(participating, corrected, cpu_num_computed_tokens)
    )
    num_accepted_tokens.copy_(
        torch.where(participating, valid_counts, num_accepted_tokens)
    )


def unconditional_to_conditional_rates(rates: list[float]) -> list[float]:
    """Convert per-position unconditional rates to per-position conditional
    rates for the early-terminating rejection loop (c_i = p_i / p_{i-1})."""
    # 中文注释：将无条件接受率转换为条件接受率。
    #
    # 【背景】
    # 在投机解码的 rejection sampling 中，每个位置的接受率是"无条件"的，
    # 即"位置 i 的 token 被接受的概率"。但在 early-terminating rejection loop 中，
    # 需要的是"条件"接受率，即"在前面所有 token 都被接受的前提下，位置 i 也被接受的概率"。
    #
    # 【转换公式】
    # c_i = p_i / p_{i-1}
    # 其中 p_i 是位置 i 的无条件接受率，c_i 是条件接受率。
    # 例如：p = [0.9, 0.6, 0.3]
    #       c = [0.9, 0.6/0.9, 0.3/0.6] = [0.9, 0.667, 0.5]
    #
    # 【使用场景】
    # 用于 rejection sampling 中的 early stopping：如果某个位置的条件接受率很低，
    # 可以提前终止验证，避免不必要的计算。
    return [p / q if q > 0.0 else 0.0 for p, q in zip(rates, [1.0, *rates[:-1]])]
