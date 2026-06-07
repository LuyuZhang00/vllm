# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
拒绝采样工具模块。

本模块实现了拒绝采样算法的核心 Triton 内核，用于高效地在 GPU 上执行投机解码的验证步骤。

算法概述：
拒绝采样是一种保证输出分布与目标模型一致的采样方法。对于草稿模型生成的每个候选 token x：
1. 计算目标概率 p(x) 和草稿概率 q(x)
2. 以概率 min(1, p(x)/q(x)) 接受该 token
3. 第一个被拒绝的 token 从修正分布 max(p(x)-q(x), 0) / Z 中重新采样

本模块使用分块处理策略来处理大词汇表：
1. 将词汇表分成固定大小的块（block）
2. 并行计算每个块的局部统计信息（最大值、sumexp、argmax）
3. 合并所有块的统计信息得到全局结果
4. 基于全局统计信息执行拒绝采样和重采样

包含的 Triton 内核：
1. _compute_block_stats_kernel: 计算块级别的目标和草稿 logits 统计
2. _rejection_kernel: 执行拒绝采样检验
3. _resample_kernel: 对被拒绝的 token 从修正分布重采样
4. _insert_resampled_kernel: 将重采样的 token 插入输出数组
"""

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.sample.gumbel import gumbel_block_argmax, tl_rand64


@triton.jit
def _compute_block_max_and_sumexp(logits):
    """
    计算单个块内的最大值和 sumexp（指数和）。

    这是数值稳定的 softmax 计算的基础。通过先计算块内最大值，
    然后计算 exp(x - max) 的和，可以避免数值溢出。

    参数:
        logits: 块内的 logits 值。

    返回:
        block_max: 块内最大值。
        block_sumexp: 块内 exp(x - max) 的和。
    """
    block_max = tl.max(logits, axis=0)
    block_sumexp = tl.where(
        block_max > float("-inf"),
        tl.sum(tl.exp(logits - block_max)),
        0.0,
    )
    return block_max, block_sumexp


@triton.jit
def _compute_global_lse(
    local_max_ptr,
    local_max_stride,
    local_sumexp_ptr,
    local_sumexp_stride,
    logit_idx,
    vocab_num_blocks,
    PADDED_VOCAB_NUM_BLOCKS: tl.constexpr,
):
    """
    从所有块的局部统计信息计算全局 log-sum-exp（LSE）。

    使用数值稳定的方式合并各块的结果：
    LSE = global_max + log(sum(sumexp_i * exp(max_i - global_max)))

    参数:
        local_max_ptr: 各块局部最大值的指针。
        local_max_stride: 局部最大值张量的行步长。
        local_sumexp_ptr: 各块局部 sumexp 的指针。
        local_sumexp_stride: 局部 sumexp 张量的行步长。
        logit_idx: 当前处理的 logit 索引。
        vocab_num_blocks: 词汇表分成的块数。
        PADDED_VOCAB_NUM_BLOCKS: 块数的填充大小（2的幂次）。

    返回:
        global_lse: 全局 log-sum-exp 值。
    """
    blocks = tl.arange(0, PADDED_VOCAB_NUM_BLOCKS)
    blocks_mask = blocks < vocab_num_blocks
    maxes = tl.load(
        local_max_ptr + logit_idx * local_max_stride + blocks,
        mask=blocks_mask,
        other=float("-inf"),
    )
    sumexps = tl.load(
        local_sumexp_ptr + logit_idx * local_sumexp_stride + blocks,
        mask=blocks_mask,
        other=0.0,
    )
    global_max = tl.max(maxes, axis=0)
    global_lse = global_max + tl.log(tl.sum(sumexps * tl.exp(maxes - global_max)))
    return global_lse


@triton.jit
def _compute_block_stats_kernel(
    # [num_logits, num_blocks]
    target_local_argmax_ptr,
    target_local_argmax_stride,
    # [num_logits, num_blocks]
    target_local_max_ptr,
    target_local_max_stride,
    # [num_logits, num_blocks]
    target_local_sumexp_ptr,
    target_local_sumexp_stride,
    # [num_logits, num_blocks]
    draft_local_max_ptr,
    draft_local_max_stride,
    # [num_logits, num_blocks]
    draft_local_sumexp_ptr,
    draft_local_sumexp_stride,
    # [num_logits, V]
    target_logits_ptr,
    target_logits_stride,
    # [max_num_reqs, num_speculative_steps, V]
    draft_logits_ptr,
    draft_logits_stride_0,
    draft_logits_stride_1,
    # [num_logits]
    expanded_idx_mapping_ptr,
    # [num_logits]
    expanded_local_pos_ptr,
    # [max_num_reqs]
    temp_ptr,
    vocab_size,
    num_speculative_steps,
    BLOCK_SIZE: tl.constexpr,
    HAS_DRAFT_LOGITS: tl.constexpr,
):
    """
    计算块级别的 logits 统计信息。

    这是拒绝采样的第一步，并行计算每个词汇块的局部统计信息：
    - 目标模型：最大值、sumexp、argmax（用于贪心采样）
    - 草稿模型：最大值、sumexp（用于概率比计算）

    每个程序处理一个 (logit_idx, block_idx) 对，即某个 logit 在某个词汇块上的统计。

    参数:
        target_local_argmax_ptr: 目标模型局部 argmax 输出指针。
        target_local_max_ptr: 目标模型局部最大值输出指针。
        target_local_sumexp_ptr: 目标模型局部 sumexp 输出指针。
        draft_local_max_ptr: 草稿模型局部最大值输出指针。
        draft_local_sumexp_ptr: 草稿模型局部 sumexp 输出指针。
        target_logits_ptr: 目标模型 logits 输入指针。
        draft_logits_ptr: 草稿模型 logits 输入指针。
        expanded_idx_mapping_ptr: 展开的请求索引映射指针。
        expanded_local_pos_ptr: 展开的局部位置指针。
        temp_ptr: 温度参数指针。
        vocab_size: 词汇表大小。
        num_speculative_steps: 投机解码步数。
        BLOCK_SIZE: 每个块处理的词汇数量。
        HAS_DRAFT_LOGITS: 是否有草稿模型的 logits。

    流程:
        1. 获取当前 logit 对应的草稿步骤和请求索引
        2. 跳过 bonus token（不需要统计）
        3. 根据温度参数选择处理路径：
           - 贪心（temp=0）：只计算目标模型的 argmax 和最大值
           - 非贪心：计算目标和草稿的 max + sumexp
    """
    logit_idx = tl.program_id(0)
    draft_step_idx = tl.load(expanded_local_pos_ptr + logit_idx)

    if draft_step_idx >= num_speculative_steps:
        # Bonus token. Max/argmax and summed exponentials are not needed.
        return

    req_state_idx = tl.load(expanded_idx_mapping_ptr + logit_idx)
    temp = tl.load(temp_ptr + req_state_idx).to(tl.float32)

    block_idx = tl.program_id(1)
    block_offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block_offsets < vocab_size

    if temp == 0.0:
        # Greedy sampling. Only the target max/argmax are needed.
        # 贪心采样：只需要目标模型的 argmax，无需计算 sumexp
        target_logits = tl.load(
            target_logits_ptr + logit_idx * target_logits_stride + block_offsets,
            mask=mask,
            other=float("-inf"),
        ).to(tl.float32)
        value, idx = tl.max(target_logits, axis=0, return_indices=True)
        token_id = block_idx * BLOCK_SIZE + idx
        tl.store(
            target_local_argmax_ptr
            + logit_idx * target_local_argmax_stride
            + block_idx,
            token_id,
        )
        tl.store(
            target_local_max_ptr + logit_idx * target_local_max_stride + block_idx,
            value,
        )
    else:
        # Get local target max and summed exponentials.
        # 非贪心采样：计算目标模型的 max 和 sumexp
        target_logits = tl.load(
            target_logits_ptr + logit_idx * target_logits_stride + block_offsets,
            mask=mask,
            other=float("-inf"),
        ).to(tl.float32)
        target_max, target_sumexp = _compute_block_max_and_sumexp(target_logits)
        tl.store(
            target_local_max_ptr + logit_idx * target_local_max_stride + block_idx,
            target_max,
        )
        tl.store(
            target_local_sumexp_ptr
            + logit_idx * target_local_sumexp_stride
            + block_idx,
            target_sumexp,
        )
        if HAS_DRAFT_LOGITS:
            # Get local draft max and summed exponentials.
            # 计算草稿模型的 max 和 sumexp（用于概率比检验）
            draft_logits = tl.load(
                draft_logits_ptr
                + req_state_idx * draft_logits_stride_0
                + draft_step_idx * draft_logits_stride_1
                + block_offsets,
                mask=mask,
                other=float("-inf"),
            ).to(tl.float32)
            draft_max, draft_sumexp = _compute_block_max_and_sumexp(draft_logits)
            tl.store(
                draft_local_max_ptr + logit_idx * draft_local_max_stride + block_idx,
                draft_max,
            )
            tl.store(
                draft_local_sumexp_ptr
                + logit_idx * draft_local_sumexp_stride
                + block_idx,
                draft_sumexp,
            )


@triton.jit
def _rejection_kernel(
    # [num_reqs, num_speculative_steps + 1]
    sampled_ptr,
    sampled_stride,
    # [num_reqs]
    rejected_steps_ptr,
    # [num_reqs]
    target_rejected_logsumexp_ptr,
    # [num_reqs]
    draft_rejected_logsumexp_ptr,
    # [num_logits, V]
    target_logits_ptr,
    target_logits_stride,
    # [num_logits, num_blocks]
    target_local_argmax_ptr,
    target_local_argmax_stride,
    # [num_logits, num_blocks]
    target_local_max_ptr,
    target_local_max_stride,
    # [num_logits, num_blocks]
    target_local_sumexp_ptr,
    target_local_sumexp_stride,
    # [num_logits]
    draft_sampled_ptr,
    # [max_num_reqs, num_speculative_steps, V]
    draft_logits_ptr,
    draft_logits_stride_0,
    draft_logits_stride_1,
    # [num_logits, num_blocks]
    draft_local_max_ptr,
    draft_local_max_stride,
    # [num_logits, num_blocks]
    draft_local_sumexp_ptr,
    draft_local_sumexp_stride,
    # [num_reqs + 1]
    cu_num_logits_ptr,
    # [num_reqs]
    idx_mapping_ptr,
    # [max_num_reqs]
    temp_ptr,
    # [max_num_reqs]
    seed_ptr,
    # [num_logits]
    pos_ptr,
    # [num_speculative_steps]
    synthetic_conditional_rates_ptr,
    vocab_num_blocks,
    PADDED_VOCAB_NUM_BLOCKS: tl.constexpr,
    HAS_DRAFT_LOGITS: tl.constexpr,
    SYNTHETIC_MODE: tl.constexpr,
):
    """
    拒绝采样检验内核。

    对每个请求的草稿 token 序列执行拒绝采样检验。
    从第一个 token 开始，逐个检查每个草稿 token 是否被接受。
    一旦某个 token 被拒绝，后续的 token 全部跳过。

    每个程序处理一个请求。

    参数:
        sampled_ptr: 输出的采样结果指针，形状 [num_reqs, num_speculative_steps + 1]。
        sampled_stride: 采样结果的行步长。
        rejected_steps_ptr: 输出的被拒绝步骤索引指针。
        target_rejected_logsumexp_ptr: 目标模型在拒绝点的 LSE 输出指针。
        draft_rejected_logsumexp_ptr: 草稿模型在拒绝点的 LSE 输出指针。
        target_logits_ptr: 目标模型 logits 指针。
        target_local_argmax_ptr: 目标模型局部 argmax 指针。
        target_local_max_ptr: 目标模型局部最大值指针。
        target_local_sumexp_ptr: 目标模型局部 sumexp 指针。
        draft_sampled_ptr: 草稿模型采样的 token 指针。
        draft_logits_ptr: 草稿模型 logits 指针。
        draft_local_max_ptr: 草稿模型局部最大值指针。
        draft_local_sumexp_ptr: 草稿模型局部 sumexp 指针。
        cu_num_logits_ptr: 累积 logits 数量前缀和指针。
        idx_mapping_ptr: 请求索引映射指针。
        temp_ptr: 温度参数指针。
        seed_ptr: 随机种子指针。
        pos_ptr: 位置索引指针。
        synthetic_conditional_rates_ptr: 合成条件接受率指针。
        vocab_num_blocks: 词汇表块数。
        PADDED_VOCAB_NUM_BLOCKS: 块数填充大小。
        HAS_DRAFT_LOGITS: 是否有草稿 logits。
        SYNTHETIC_MODE: 是否使用合成模式。

    流程:
        1. 对每个草稿步骤（除了 bonus token）：
           a. 贪心模式：比较目标 argmax 和草稿 token
           b. 非贪心模式：计算概率比 p(x)/q(x)，与均匀随机数比较
        2. 记录第一个被拒绝的步骤索引
        3. 保存拒绝点的 LSE（用于后续重采样）
    """
    req_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + req_idx)
    start_idx = tl.load(cu_num_logits_ptr + req_idx)
    end_idx = tl.load(cu_num_logits_ptr + req_idx + 1)
    num_tokens = end_idx - start_idx
    seed = tl.load(seed_ptr + req_state_idx)
    temp = tl.load(temp_ptr + req_state_idx).to(tl.float32)

    rejected_step = 0
    target_lse = 0.0
    draft_lse = 0.0
    accepted = True
    for i in range(num_tokens - 1):
        if accepted:
            logit_idx = start_idx + i
            draft_sampled = tl.load(draft_sampled_ptr + logit_idx + 1).to(tl.int64)
            if temp == 0.0:
                # Greedy sampling. Accept IFF draft matches target argmax.
                # NOTE: Target argmax is stored directly so that resampling
                # can be skipped upon rejection.
                # 贪心采样：只有当草稿 token 与目标 argmax 完全匹配时才接受
                target_blocks = tl.arange(0, PADDED_VOCAB_NUM_BLOCKS)
                target_blocks_mask = target_blocks < vocab_num_blocks
                target_local_max = tl.load(
                    target_local_max_ptr
                    + logit_idx * target_local_max_stride
                    + target_blocks,
                    mask=target_blocks_mask,
                    other=float("-inf"),
                )
                max_target_block_idx = tl.argmax(target_local_max, axis=0)
                target_argmax = tl.load(
                    target_local_argmax_ptr
                    + logit_idx * target_local_argmax_stride
                    + max_target_block_idx
                ).to(tl.int64)

                if SYNTHETIC_MODE:
                    pos = tl.load(pos_ptr + logit_idx)
                    u = tl_rand64(seed, pos, includes_zero=False)
                    rate = tl.load(synthetic_conditional_rates_ptr + i)
                    accepted &= u < rate
                else:
                    accepted &= target_argmax == draft_sampled
                tl.store(
                    sampled_ptr + req_idx * sampled_stride + i,
                    draft_sampled if accepted else target_argmax,
                )
            else:
                # 非贪心采样：使用概率比检验
                target_logit = tl.load(
                    target_logits_ptr + logit_idx * target_logits_stride + draft_sampled
                ).to(tl.float32)
                target_lse = _compute_global_lse(
                    target_local_max_ptr,
                    target_local_max_stride,
                    target_local_sumexp_ptr,
                    target_local_sumexp_stride,
                    logit_idx,
                    vocab_num_blocks,
                    PADDED_VOCAB_NUM_BLOCKS,
                )
                target_log_prob = target_logit - target_lse
                pos = tl.load(pos_ptr + logit_idx)
                u = tl_rand64(seed, pos, includes_zero=False)
                if HAS_DRAFT_LOGITS:
                    draft_logit = tl.load(
                        draft_logits_ptr
                        + req_state_idx * draft_logits_stride_0
                        + i * draft_logits_stride_1
                        + draft_sampled
                    ).to(tl.float32)
                    draft_lse = _compute_global_lse(
                        draft_local_max_ptr,
                        draft_local_max_stride,
                        draft_local_sumexp_ptr,
                        draft_local_sumexp_stride,
                        logit_idx,
                        vocab_num_blocks,
                        PADDED_VOCAB_NUM_BLOCKS,
                    )
                    draft_log_prob = draft_logit - draft_lse
                else:
                    # One-hot draft: q(draft_token) = 1, log_q = 0.
                    # One-hot 草稿：草稿 token 的概率为 1，log_q = 0
                    draft_log_prob = 0

                if SYNTHETIC_MODE:
                    rate = tl.load(synthetic_conditional_rates_ptr + i)
                    accepted &= u < rate
                else:
                    # Probability ratio test: p(x) > u * q(x)
                    # Equivalent log form: log_p(x) > log(u) + log_q(x)
                    # 概率比检验：如果 log_p(x) > log(u) + log_q(x) 则接受
                    accepted &= target_log_prob > tl.log(u) + draft_log_prob
                tl.store(sampled_ptr + req_idx * sampled_stride + i, draft_sampled)
            rejected_step += accepted
    tl.store(rejected_steps_ptr + req_idx, rejected_step)
    tl.store(target_rejected_logsumexp_ptr + req_idx, target_lse)
    tl.store(draft_rejected_logsumexp_ptr + req_idx, draft_lse)


@triton.jit
def _resample_kernel(
    # [num_reqs, num_blocks]
    resampled_local_argmax_ptr,
    resampled_local_argmax_stride,
    # [num_reqs, num_blocks]
    resampled_local_max_ptr,
    resampled_local_max_stride,
    # [num_logits, V]
    target_logits_ptr,
    target_logits_stride,
    # [num_reqs]
    target_rejected_logsumexp_ptr,
    # [max_num_reqs, num_speculative_steps, V]
    draft_logits_ptr,
    draft_logits_stride_0,
    draft_logits_stride_1,
    # [num_reqs]
    draft_rejected_logsumexp_ptr,
    # [num_reqs]
    rejected_step_ptr,
    # [num_reqs + 1]
    cu_num_logits_ptr,
    # [num_logits]
    expanded_idx_mapping_ptr,
    # [num_logits]
    draft_sampled_ptr,
    # [max_num_reqs]
    temp_ptr,
    # [max_num_reqs]
    seed_ptr,
    # [num_logits]
    pos_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
    HAS_DRAFT_LOGITS: tl.constexpr,
    USE_FP64: tl.constexpr,
):
    """
    重采样内核。

    对被拒绝的 token（或 bonus token）从修正分布中重新采样。

    修正分布的计算方式：
    - 如果有草稿 logits：residual_logits = log(max(p(x) - q(x), 0))
    - 如果是 one-hot 草稿：将被拒绝的 token 概率设为 0，其余保持目标分布
    - 如果是 bonus token：直接使用目标分布

    每个程序处理一个 (req_idx, block_idx) 对。

    参数:
        resampled_local_argmax_ptr: 重采样的局部 argmax 输出指针。
        resampled_local_max_ptr: 重采样的局部最大值输出指针。
        target_logits_ptr: 目标模型 logits 指针。
        target_rejected_logsumexp_ptr: 目标模型在拒绝点的 LSE 指针。
        draft_logits_ptr: 草稿模型 logits 指针。
        draft_rejected_logsumexp_ptr: 草稿模型在拒绝点的 LSE 指针。
        rejected_step_ptr: 被拒绝的步骤索引指针。
        cu_num_logits_ptr: 累积 logits 数量前缀和指针。
        expanded_idx_mapping_ptr: 展开的请求索引映射指针。
        draft_sampled_ptr: 草稿模型采样的 token 指针。
        temp_ptr: 温度参数指针。
        seed_ptr: 随机种子指针。
        pos_ptr: 位置索引指针。
        vocab_size: 词汇表大小。
        BLOCK_SIZE: 每个块处理的词汇数量。
        HAS_DRAFT_LOGITS: 是否有草稿 logits。
        USE_FP64: 是否使用 FP64 精度（用于 Gumbel 采样）。

    流程:
        1. 计算残差 logits（修正分布的对数概率）
        2. 使用 Gumbel-max 技巧从残差分布中采样
        3. 保存采样结果的局部 argmax 和最大值
    """
    req_idx = tl.program_id(0)
    resample_idx = tl.load(rejected_step_ptr + req_idx)
    start_idx = tl.load(cu_num_logits_ptr + req_idx)
    end_idx = tl.load(cu_num_logits_ptr + req_idx + 1)
    resample_token_idx = start_idx + resample_idx
    req_state_idx = tl.load(expanded_idx_mapping_ptr + resample_token_idx)

    temp = tl.load(temp_ptr + req_state_idx).to(tl.float32)
    is_bonus = resample_token_idx == end_idx - 1
    if temp == 0.0 and not is_bonus:
        # Greedy + non-bonus token. No resampling needed because
        # the target argmax is already in the sampled tensor.
        # 贪心 + 非 bonus token：不需要重采样，因为目标 argmax 已经在采样结果中
        return

    block_idx = tl.program_id(1)
    block = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block < vocab_size
    target_logits = tl.load(
        target_logits_ptr + resample_token_idx * target_logits_stride + block,
        mask=mask,
        other=float("-inf"),
    ).to(tl.float32)

    # Compute the residual logits to resample the rejected token from.
    if is_bonus:
        # Bonus token (no rejections). Directly use the target logits.
        # Bonus token（没有拒绝）：直接使用目标 logits
        residual_logits = target_logits
    elif HAS_DRAFT_LOGITS:
        draft_logits = tl.load(
            draft_logits_ptr
            + req_state_idx * draft_logits_stride_0
            + resample_idx * draft_logits_stride_1
            + block,
            mask=mask,
            other=float("-inf"),
        ).to(tl.float32)
        target_lse = tl.load(target_rejected_logsumexp_ptr + req_idx)
        draft_lse = tl.load(draft_rejected_logsumexp_ptr + req_idx)
        target_log_probs = target_logits - target_lse
        draft_log_probs = draft_logits - draft_lse
        # Compute the residual: max(p(x) - q(x), 0)
        # Equivalent log form: log(max(exp(log_p(x)) - exp(log_q(x)), 0))
        # The more numerically stable form is:
        # log(max(exp(a) - exp(b), 0)) = a + log(max(1 - exp(b - a), 0))
        # 计算残差分布：max(p(x) - q(x), 0)
        # 使用数值稳定的形式：a + log(max(1 - exp(b - a), 0))
        ratio = tl.exp(draft_log_probs - target_log_probs)
        residual_logits = tl.where(
            ratio < 1.0,
            target_log_probs + tl.log(1 - ratio),
            float("-inf"),
        ).to(tl.float32)
    else:
        # One-hot draft. The residual is just the target distribution with
        # the rejected draft token probability zeroed out.
        # One-hot 草稿：残差分布是目标分布减去被拒绝 token 的概率
        rejected_draft_token = tl.load(draft_sampled_ptr + resample_token_idx + 1)
        residual_logits = tl.where(
            block != rejected_draft_token,
            target_logits,
            float("-inf"),
        ).to(tl.float32)

    # Resample the rejected/bonus token.
    # 使用 Gumbel-max 技巧从残差分布中采样
    value, idx = gumbel_block_argmax(
        residual_logits,
        block,
        mask,
        resample_token_idx,
        expanded_idx_mapping_ptr,
        temp_ptr,
        seed_ptr,
        pos_ptr,
        None,  # processed_logits_ptr
        0,  # processed_logits_stride
        None,  # processed_logits_col_ptr
        vocab_size,
        APPLY_TEMPERATURE=False,
        USE_FP64=USE_FP64,
    )
    token_id = block_idx * BLOCK_SIZE + idx
    tl.store(
        resampled_local_argmax_ptr
        + req_idx * resampled_local_argmax_stride
        + block_idx,
        token_id,
    )
    tl.store(
        resampled_local_max_ptr + req_idx * resampled_local_max_stride + block_idx,
        value,
    )


@triton.jit
def _insert_resampled_kernel(
    # [num_reqs, num_speculative_steps + 1]
    sampled_ptr,
    sampled_stride,
    # [num_reqs]
    num_sampled_ptr,
    # [num_reqs, num_blocks]
    resampled_local_argmax_ptr,
    resampled_local_argmax_stride,
    # [num_reqs, num_blocks]
    resampled_local_max_ptr,
    resampled_local_max_stride,
    resample_num_blocks,
    # [num_reqs + 1]
    cu_num_logits_ptr,
    # [num_logits]
    expanded_idx_mapping_ptr,
    # [max_num_reqs]
    temp_ptr,
    PADDED_RESAMPLE_NUM_BLOCKS: tl.constexpr,
):
    """
    将重采样的 token 插入输出数组。

    从各块的局部结果中选出全局最优的 token，插入到采样结果数组中。

    每个程序处理一个请求。

    参数:
        sampled_ptr: 采样结果数组指针。
        sampled_stride: 采样结果的行步长。
        num_sampled_ptr: 每个请求已采样的 token 数量指针。
        resampled_local_argmax_ptr: 重采样的局部 argmax 指针。
        resampled_local_max_ptr: 重采样的局部最大值指针。
        resample_num_blocks: 重采样的块数。
        cu_num_logits_ptr: 累积 logits 数量前缀和指针。
        expanded_idx_mapping_ptr: 展开的请求索引映射指针。
        temp_ptr: 温度参数指针。
        PADDED_RESAMPLE_NUM_BLOCKS: 重采样块数的填充大小。

    流程:
        1. 递增已采样 token 数量
        2. 如果是贪心 + 非 bonus，直接使用已有的目标 argmax
        3. 否则，从各块中选出全局最优 token
        4. 将选出的 token 插入到采样结果数组
    """
    req_idx = tl.program_id(0)
    num_sampled = tl.load(num_sampled_ptr + req_idx)
    start_idx = tl.load(cu_num_logits_ptr + req_idx)
    end_idx = tl.load(cu_num_logits_ptr + req_idx + 1)
    resample_token_idx = start_idx + num_sampled
    req_state_idx = tl.load(expanded_idx_mapping_ptr + resample_token_idx)

    # Increment the number of sampled tokens.
    tl.store(num_sampled_ptr + req_idx, num_sampled + 1)

    temp = tl.load(temp_ptr + req_state_idx).to(tl.float32)
    is_bonus = resample_token_idx == end_idx - 1
    if temp == 0.0 and not is_bonus:
        # Greedy + non-bonus token. The target argmax is already
        # in the sampled tensor.
        # 贪心 + 非 bonus token：目标 argmax 已经在采样结果中
        return

    # Insert the resampled token.
    # 从各块中选出全局最优的 token
    block = tl.arange(0, PADDED_RESAMPLE_NUM_BLOCKS)
    mask = block < resample_num_blocks
    resampled_local_max = tl.load(
        resampled_local_max_ptr + req_idx * resampled_local_max_stride + block,
        mask=mask,
        other=float("-inf"),
    )
    resampled_max_block_idx = tl.argmax(resampled_local_max, axis=0)
    resampled = tl.load(
        resampled_local_argmax_ptr
        + req_idx * resampled_local_argmax_stride
        + resampled_max_block_idx,
    )
    tl.store(
        sampled_ptr + req_idx * sampled_stride + num_sampled,
        resampled,
    )


def rejection_sample(
    # [num_logits, V]
    target_logits: torch.Tensor,
    # [max_num_reqs, num_speculative_steps, V]
    draft_logits: torch.Tensor | None,
    # [num_logits]
    draft_sampled: torch.Tensor,
    # [num_reqs + 1]
    cu_num_logits: torch.Tensor,
    # [num_logits]
    pos: torch.Tensor,
    # [num_reqs]
    idx_mapping: torch.Tensor,
    # [num_logits]
    expanded_idx_mapping: torch.Tensor,
    # [num_logits]
    expanded_local_pos: torch.Tensor,
    # [max_num_reqs]
    temperature: torch.Tensor,
    # [max_num_reqs]
    seed: torch.Tensor,
    num_speculative_steps: int,
    # [num_speculative_steps]
    synthetic_conditional_rates: torch.Tensor | None = None,
    use_fp64: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    执行拒绝采样算法。

    这是拒绝采样的主函数，协调各个 Triton 内核完成完整的拒绝采样流程。

    参数:
        target_logits (torch.Tensor): 目标模型输出的 logits，形状 [num_logits, vocab_size]。
        draft_logits (torch.Tensor | None): 草稿模型输出的 logits，
            形状 [max_num_reqs, num_speculative_steps, vocab_size]。
            如果为 None，则使用 one-hot 草稿分布。
        draft_sampled (torch.Tensor): 草稿模型采样的 token，形状 [num_logits]。
        cu_num_logits (torch.Tensor): 累积 logits 数量的前缀和，形状 [num_reqs + 1]。
        pos (torch.Tensor): 位置索引，形状 [num_logits]。
        idx_mapping (torch.Tensor): 请求索引映射，形状 [num_reqs]。
        expanded_idx_mapping (torch.Tensor): 展开的请求索引映射，形状 [num_logits]。
        expanded_local_pos (torch.Tensor): 展开的局部位置，形状 [num_logits]。
        temperature (torch.Tensor): 温度参数，形状 [max_num_reqs]。
        seed (torch.Tensor): 随机种子，形状 [max_num_reqs]。
        num_speculative_steps (int): 投机解码步数。
        synthetic_conditional_rates (torch.Tensor | None): 合成条件接受率。
        use_fp64 (bool): 是否使用 FP64 精度。

    返回:
        tuple[torch.Tensor, torch.Tensor]:
            - sampled: 采样结果，形状 [num_reqs, num_speculative_steps + 1]。
            - num_sampled: 每个请求实际采样的 token 数量，形状 [num_reqs]。

    流程:
        1. 计算块级别的统计信息（_compute_block_stats_kernel）
        2. 执行拒绝采样检验（_rejection_kernel）
        3. 对被拒绝的 token 进行重采样（_resample_kernel）
        4. 将重采样的 token 插入输出（_insert_resampled_kernel）
    """
    num_reqs = cu_num_logits.shape[0] - 1
    num_logits, vocab_size = target_logits.shape
    has_draft_logits = draft_logits is not None

    if draft_logits is None:
        # When draft_logits is None, create a dummy tensor so that Triton
        # kernel signatures receive valid pointers/strides. The kernels
        # will never read from it when HAS_DRAFT_LOGITS=False.
        # 当 draft_logits 为 None 时，创建一个虚拟张量以满足 Triton 内核签名要求
        draft_logits = target_logits.new_empty(1, 1, 1)

    # Compute the block-level logits stats, such as target argmax
    # (for greedy requests), and target max + softmax exponential
    # (for non-greedy requests).
    # 计算块级别的 logits 统计信息
    VOCAB_BLOCK_SIZE = 8192
    vocab_num_blocks = triton.cdiv(vocab_size, VOCAB_BLOCK_SIZE)
    padded_vocab_num_blocks = triton.next_power_of_2(vocab_num_blocks)
    target_local_argmax = target_logits.new_empty(
        num_logits, vocab_num_blocks, dtype=torch.int64
    )
    target_local_max = target_logits.new_empty(
        num_logits, vocab_num_blocks, dtype=torch.float32
    )
    target_local_sumexp = target_logits.new_empty(
        num_logits, vocab_num_blocks, dtype=torch.float32
    )
    draft_local_max = target_logits.new_empty(
        num_logits, vocab_num_blocks, dtype=torch.float32
    )
    draft_local_sumexp = target_logits.new_empty(
        num_logits, vocab_num_blocks, dtype=torch.float32
    )
    _compute_block_stats_kernel[(num_logits, vocab_num_blocks)](
        target_local_argmax,
        target_local_argmax.stride(0),
        target_local_max,
        target_local_max.stride(0),
        target_local_sumexp,
        target_local_sumexp.stride(0),
        draft_local_max,
        draft_local_max.stride(0),
        draft_local_sumexp,
        draft_local_sumexp.stride(0),
        target_logits,
        target_logits.stride(0),
        draft_logits,
        draft_logits.stride(0),
        draft_logits.stride(1),
        expanded_idx_mapping,
        expanded_local_pos,
        temperature,
        vocab_size,
        num_speculative_steps,
        BLOCK_SIZE=VOCAB_BLOCK_SIZE,
        HAS_DRAFT_LOGITS=has_draft_logits,
    )

    # Sample up until the first rejected/bonus token, and store
    # the step.
    # 执行拒绝采样检验，直到第一个被拒绝的 token 或 bonus token
    sampled = draft_sampled.new_empty(
        num_reqs, num_speculative_steps + 1, dtype=torch.int64
    )
    num_sampled = sampled.new_empty(num_reqs, dtype=torch.int32)
    target_rejected_logsumexp = target_logits.new_empty(num_reqs, dtype=torch.float32)
    draft_rejected_logsumexp = target_logits.new_empty(num_reqs, dtype=torch.float32)
    _rejection_kernel[(num_reqs,)](
        sampled,
        sampled.stride(0),
        num_sampled,
        target_rejected_logsumexp,
        draft_rejected_logsumexp,
        target_logits,
        target_logits.stride(0),
        target_local_argmax,
        target_local_argmax.stride(0),
        target_local_max,
        target_local_max.stride(0),
        target_local_sumexp,
        target_local_sumexp.stride(0),
        draft_sampled,
        draft_logits,
        draft_logits.stride(0),
        draft_logits.stride(1),
        draft_local_max,
        draft_local_max.stride(0),
        draft_local_sumexp,
        draft_local_sumexp.stride(0),
        cu_num_logits,
        idx_mapping,
        temperature,
        seed,
        pos,
        synthetic_conditional_rates,
        vocab_num_blocks,
        PADDED_VOCAB_NUM_BLOCKS=padded_vocab_num_blocks,
        HAS_DRAFT_LOGITS=has_draft_logits,
        SYNTHETIC_MODE=synthetic_conditional_rates is not None,
        num_warps=1,
    )

    # Resample the rejected/bonus tokens.
    # 对被拒绝的 token 和 bonus token 进行重采样
    RESAMPLE_BLOCK_SIZE = 1024
    resample_num_blocks = triton.cdiv(vocab_size, RESAMPLE_BLOCK_SIZE)
    padded_resample_num_blocks = triton.next_power_of_2(resample_num_blocks)
    resampled_local_argmax = target_logits.new_empty(
        num_reqs, resample_num_blocks, dtype=torch.int64
    )
    resampled_local_max = target_logits.new_empty(
        num_reqs,
        resample_num_blocks,
        dtype=torch.float64 if use_fp64 else torch.float32,
    )
    _resample_kernel[(num_reqs, resample_num_blocks)](
        resampled_local_argmax,
        resampled_local_argmax.stride(0),
        resampled_local_max,
        resampled_local_max.stride(0),
        target_logits,
        target_logits.stride(0),
        target_rejected_logsumexp,
        draft_logits,
        draft_logits.stride(0),
        draft_logits.stride(1),
        draft_rejected_logsumexp,
        num_sampled,
        cu_num_logits,
        expanded_idx_mapping,
        draft_sampled,
        temperature,
        seed,
        pos,
        vocab_size,
        BLOCK_SIZE=RESAMPLE_BLOCK_SIZE,
        HAS_DRAFT_LOGITS=has_draft_logits,
        USE_FP64=use_fp64,
    )

    # Insert the resampled tokens into the output sampled.
    # 将重采样的 token 插入到输出数组
    _insert_resampled_kernel[(num_reqs,)](
        sampled,
        sampled.stride(0),
        num_sampled,
        resampled_local_argmax,
        resampled_local_argmax.stride(0),
        resampled_local_max,
        resampled_local_max.stride(0),
        resample_num_blocks,
        cu_num_logits,
        expanded_idx_mapping,
        temperature,
        PADDED_RESAMPLE_NUM_BLOCKS=padded_resample_num_blocks,
    )
    return sampled, num_sampled
