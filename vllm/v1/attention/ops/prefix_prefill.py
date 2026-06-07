# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# The kernels in this file are adapted from LightLLM's context_attention_fwd:
# https://github.com/ModelTC/lightllm/blob/main/lightllm/models/llama/triton_kernel/context_flashattention_nopad.py

# 中文注释：本文件实现了 prefix prefill（前缀预填充）场景下的 Triton 注意力内核。
#
# 【核心场景】
#   当请求存在 prefix cache 命中时，已缓存的 KV 不需要重新计算，但 attention
#   仍然需要将 query 与这些已缓存的 KV 进行计算。本文件的内核专门优化了这种场景：
#   - query 来自当前 step 新计算的 token（内存中的 Q 张量）
#   - key/value 有两部分来源：
#     (a) 已缓存的 prefix KV（存储在分页 KV cache 中，通过 B_Loc/block table 寻址）
#     (b) 当前 step 新计算的 KV（内存中的 K/V 张量）
#   内核会分两阶段计算注意力：先算 Q 与 prefix KV，再算 Q 与新 KV，最后合并。
#
# 【内核函数】
#   1. _fwd_kernel: 标准注意力内核，支持因果/非因果、滑动窗口、FP8 KV cache、
#      sink tokens、GQA 等特性。
#   2. _fwd_kernel_alibi: 支持 ALiBi (Attention with Linear Biases) 的变体内核。
#   3. context_attention_fwd: Python 入口函数，根据参数选择合适的内核并启动。
#
# 【关键优化】
#   - 分块计算 (tiling)：将 KV 序列分成 BLOCK_SIZE 大小的块逐块处理，
#     避免一次性加载所有 KV 到共享内存。
#   - 在线 softmax：使用 running max/sum 技术，只遍历一次 KV 即可得到正确的
#     softmax 注意力输出，无需两遍扫描。
#   - 分页 KV 寻址：通过 B_Loc (block table) 将逻辑 token 位置映射到
#     物理 KV cache 中的位置，支持任意 block size（包括非 2 的幂次如 544）。

from typing import Any

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

# Static kernels parameters
# 中文注释：内核静态参数。
# BASE_BLOCK: Triton 分块大小，SM 80+ (A100/H100) 使用 128，老架构使用 64。
# NUM_WARPS: 每个 Triton program 使用的 warp 数量，ROCm 上使用 4（避免寄存器溢出），CUDA 上使用 8。
BASE_BLOCK = 128 if current_platform.has_device_capability(80) else 64
NUM_WARPS = 4 if current_platform.is_rocm() else 8

# To check compatibility
# 中文注释：Turing 架构 (SM 7.5, 如 T4) 的 tensor core 不支持 float32 矩阵乘，
# 需要回退到 IEEE 精度模式。同时获取 FP8 的数值范围用于 clamp 操作。
IS_TURING = current_platform.get_device_capability() == (7, 5)
float8_info = torch.finfo(current_platform.fp8_dtype())


# Here's an example autotuner config for this kernel. This config does provide
# a performance improvement, but dramatically increases first call latency in
# triton 3.2. Because of this tradeoff, it's currently commented out.
# @triton.autotune(
#     configs=[
#         triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, \
#                         "num_unroll_cache": 4, \
#                         "num_unroll_request": 1 } | \
#                         ({"kpack": 2, "waves_per_eu": 2} \
#                             if current_platform.is_rocm() else {}), \
#                         num_warps=4, \
#                         num_stages=1)
#     ],
#     key=["BLOCK_SIZE", "MAX_Q_LEN", "MAX_CTX_LEN"]
# )
# 中文注释：标准前向注意力 Triton 内核。
# 【内核结构】
#   每个 Triton program 处理一个 (batch, head, query_block) 三元组。
#   program_id(0) = batch 索引
#   program_id(1) = head 索引（Q 的 head，KV head 通过 GQA 比率换算）
#   program_id(2) = query 分块索引（沿 query_len 维度分块）
#
# 【两阶段计算】
#   阶段 1：Q 与 prefix KV（已缓存的历史 KV，在分页 KV cache 中）计算注意力
#     - 通过 B_Loc (block table) 寻址物理 KV cache
#     - 无 causal mask（prefix 的所有 token 都对当前 query 有贡献）
#     - 支持滑动窗口限制
#   阶段 2：Q 与新计算的 KV（在内存中的 K/V 张量）计算注意力
#     - 应用 causal mask（新 token 之间有因果关系）
#     - 同样支持滑动窗口
#   两个阶段使用在线 softmax 算法合并，只遍历一次即可得到正确的全局 softmax。
#
# 【在线 softmax 原理】
#   维护 running max (m_i) 和 running sum (l_i)，每处理一个新 KV 块：
#     1. 计算新块的 qk 分数，找到新最大值 m_ij = max(m_i, max(qk))
#     2. 用 exp(m_i - m_ij) 缩放之前的累积值 acc 和 l_i
#     3. 用 exp(qk - m_ij) 计算新块的权重，更新 acc 和 l_i
#   最终 acc / l_i 就是正确的 softmax 加权输出。
@triton.jit
def _fwd_kernel(
    Q,
    K,
    V,
    K_cache,
    V_cache,
    sink_ptr,
    B_Loc,
    sm_scale,
    k_scale,
    v_scale,
    out_scale_inv,
    B_Start_Loc,
    B_Seqlen,
    x: tl.constexpr,
    Out,
    stride_b_loc_b,
    stride_b_loc_s,
    stride_qbs,
    stride_qh,
    stride_qd,
    stride_kbs,
    stride_kh,
    stride_kd,
    stride_vbs,
    stride_vh,
    stride_vd,
    stride_obs,
    stride_oh,
    stride_od,
    stride_k_cache_bs,
    stride_k_cache_h,
    stride_k_cache_d,
    stride_k_cache_bl: tl.constexpr,
    stride_k_cache_x,
    stride_v_cache_bs,
    stride_v_cache_h,
    stride_v_cache_d,
    stride_v_cache_bl,
    num_queries_per_kv: tl.constexpr,
    IN_PRECISION: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DMODEL_PADDED: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    PHYSICAL_BLOCK_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr,
    num_unroll_cache: tl.constexpr,
    num_unroll_request: tl.constexpr,
    SKIP_DECODE: tl.constexpr,
    USE_SINKS: tl.constexpr,
    USE_FP8: tl.constexpr,
    CAUSAL: tl.constexpr = True,
    MAX_Q_LEN: tl.constexpr = 0,
    MAX_CTX_LEN: tl.constexpr = 0,
    FP8_MIN: tl.constexpr = float8_info.min,
    FP8_MAX: tl.constexpr = float8_info.max,
):
    # 中文注释：确定当前 program 处理的 batch、head、query 分块
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    start_m = tl.program_id(2)

    # 中文注释：GQA (Grouped Query Attention) 处理 - 多个 Q head 共享同一个 KV head
    cur_kv_head = cur_head // num_queries_per_kv

    # 中文注释：加载当前 batch 的序列信息。
    # cur_batch_seq_len = prefix_len + query_len（总序列长度）
    # cur_batch_query_len = 当前 step 的 query 长度
    # cur_batch_ctx_len = prefix_len（已缓存的上下文长度）
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
    cur_batch_in_all_stop_index = tl.load(B_Start_Loc + cur_batch + 1)
    cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index
    cur_batch_ctx_len = cur_batch_seq_len - cur_batch_query_len

    # 中文注释：SKIP_DECODE 优化 - 当 query_len == 1（decode 阶段）时跳过，
    # 因为 decode 阶段通常由其他更高效的 kernel 处理（如 PagedAttention）。
    if SKIP_DECODE and cur_batch_query_len == 1:
        return

    # start position inside of the query
    # generally, N goes over kv, while M goes over query_len
    block_start_loc = BLOCK_M * start_m

    # initialize offsets
    # [BLOCK_SIZE]; starts at 0
    offs_bs_n = tl.arange(0, BLOCK_SIZE)
    # [N]; starts at 0
    offs_n = tl.arange(0, BLOCK_N)
    # [D]; starts at 0
    offs_d = tl.arange(0, BLOCK_DMODEL_PADDED)
    # [M]; starts at current position in query
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # [M,D]
    off_q = (
        (cur_batch_in_all_start_index + offs_m[:, None]) * stride_qbs
        + cur_head * stride_qh
        + offs_d[None, :] * stride_qd
    )

    dim_mask = tl.where(tl.arange(0, BLOCK_DMODEL_PADDED) < BLOCK_DMODEL, 1, 0).to(
        tl.int1
    )  # [D]

    q = tl.load(
        Q + off_q,
        mask=dim_mask[None, :] & (offs_m[:, None] < cur_batch_query_len),
        other=0.0,
    )  # [M,D]

    # initialize pointer to m and l
    # 中文注释：初始化在线 softmax 的状态变量。
    # m_i: running max，记录已处理 KV 块中的最大注意力分数（用于数值稳定的 softmax）
    # l_i: running sum，记录 exp(qk - m_i) 的累积和（softmax 分母）
    # acc: 输出累积器，记录加权的 V 累积和
    # USE_SINKS: 支持 attention sink 机制（某些特殊 token 的注意力权重被预设保留）
    if not USE_SINKS:
        m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    else:
        m_i = tl.load(
            sink_ptr + tl.full([BLOCK_M], cur_head, dtype=tl.int64),
            mask=(offs_m < cur_batch_query_len),
            other=float("-inf"),
        ).to(dtype=tl.float32)
        l_i = tl.where(m_i > float("-inf"), 1.0, 0.0)

    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL_PADDED], dtype=tl.float32)  # [M,D]

    # ========================================================================
    # 中文注释：阶段 1 - Q 与 prefix KV（已缓存的历史 KV）计算注意力
    # ========================================================================
    # 遍历 prefix 的所有 KV 块（每块 BLOCK_SIZE 个 token），
    # 计算 Q * K^T 和 softmax(Q * K^T) * V。
    # 注意：prefix 部分没有 causal mask，因为所有 prefix token 都在当前 query 之前。
    # compute query against context (no causal mask here)
    for start_n in tl.range(
        0, cur_batch_ctx_len, BLOCK_SIZE, loop_unroll_factor=num_unroll_cache
    ):
        # 中文注释：分页 KV cache 寻址逻辑。
        # 对于非标准 block size（如 Qwen3-next 的 544），PHYSICAL_BLOCK_SIZE 和
        # Triton 内部的 BLOCK_SIZE (32) 不同，需要进行两层映射：
        #   1. token_indices -> logical block index (token_indices // PHYSICAL_BLOCK_SIZE)
        #   2. logical block index -> physical block ID (通过 B_Loc 查表)
        #   3. token 在 block 内的偏移 (token_indices % PHYSICAL_BLOCK_SIZE)
        # Under a block size of 544 (Qwen/Qwen3-Next-80B-A3B-Thinking),
        # replace one physical block every 17 32-Tile blocks
        # Calculate the logical block index of each of the 32 tokens
        # in the current Tile (handling cross-block cases).
        token_indices = start_n + offs_bs_n
        bn_logical_indices = token_indices // PHYSICAL_BLOCK_SIZE

        # 2. Vectorized loading of physical block IDs from B_Loc
        # 中文注释：从 block table (B_Loc) 中加载物理 block ID
        bn = tl.load(
            B_Loc + cur_batch * stride_b_loc_b + bn_logical_indices * stride_b_loc_s
        ).to(tl.int64)

        # 3. Calculate the exact offset of
        # each token within its physical block.
        internal_offsets = token_indices % PHYSICAL_BLOCK_SIZE

        # Addressing of K (5D)
        # 中文注释：计算 K cache 的 5D 内存地址。
        # key cache 布局: [num_blocks, num_kv_heads, head_size/x, block_size, x]
        # 其中 x 分块是为了 GPU 内存合并访问优化。
        off_k = (
            bn[None, :] * stride_k_cache_bs
            + cur_kv_head * stride_k_cache_h
            + (offs_d[:, None] // x) * stride_k_cache_d
            + internal_offsets[None, :] * stride_k_cache_bl
            + (offs_d[:, None] % x) * stride_k_cache_x
        )

        # Addressing of V (4D)
        # 中文注释：计算 V cache 的 4D 内存地址。
        # value cache 布局: [num_blocks, num_kv_heads, head_size, block_size]
        off_v = (
            bn[:, None] * stride_v_cache_bs
            + cur_kv_head * stride_v_cache_h
            + offs_d[None, :] * stride_v_cache_d
            + internal_offsets[:, None] * stride_v_cache_bl
        )

        if (
            start_n + BLOCK_SIZE > cur_batch_ctx_len
            or BLOCK_DMODEL != BLOCK_DMODEL_PADDED
        ):
            k_load = tl.load(
                K_cache + off_k,
                mask=dim_mask[:, None]
                & ((start_n + offs_bs_n[None, :]) < cur_batch_ctx_len),
                other=0.0,
            )  # [D,N]
        else:
            k_load = tl.load(K_cache + off_k)

        if k_load.dtype.is_fp8():
            k = (k_load.to(tl.float32) * tl.load(k_scale)).to(q.dtype)
        else:
            k = k_load

        # qk = tl.zeros([BLOCK_M, BLOCK_SIZE], dtype=tl.float32)  # [M,N]
        # 中文注释：计算 Q * K^T 注意力分数，并应用缩放因子 sm_scale = 1/sqrt(d)
        qk = sm_scale * tl.dot(q, k, input_precision=IN_PRECISION)
        # 中文注释：边界掩码 - 将超出实际 ctx_len 的位置设为 -inf
        qk = tl.where(
            (start_n + offs_bs_n[None, :]) < cur_batch_ctx_len, qk, float("-inf")
        )
        # qk *= sm_scale
        # 中文注释：滑动窗口注意力掩码。
        # 只有当 Q 位置与 KV 位置的距离 < SLIDING_WINDOW 时才保留注意力分数，
        # 否则设为 -inf。这限制了每个 query token 只关注最近的 SLIDING_WINDOW 个 token。
        if SLIDING_WINDOW > 0:
            # (cur_batch_ctx_len + offs_m[:, None]) are the positions of
            # Q entries in sequence
            # (start_n + offs_bs_n[None, :]) are the positions of
            # KV entries in sequence
            # So the condition makes sure each entry in Q only attends
            # to KV entries not more than SLIDING_WINDOW away.
            #
            # We can't use -inf here, because the
            # sliding window may lead to the entire row being masked.
            # This then makes m_ij contain -inf, which causes NaNs in
            # exp().
            qk = tl.where(
                (cur_batch_ctx_len + offs_m[:, None]) - (start_n + offs_bs_n[None, :])
                < SLIDING_WINDOW,
                qk,
                float("-inf"),
            )

        # 中文注释：在线 softmax 更新。
        # 步骤：
        #   1. m_ij = max(m_i, max(qk)) - 新的全局最大值
        #   2. p = exp(qk - m_ij) - 用新最大值重新计算 softmax 权重
        #   3. alpha = exp(m_i - m_ij) - 之前累积值的缩放因子（补偿最大值变化）
        #   4. acc = acc * alpha + p @ V - 缩放旧累积值并加上新贡献
        #   5. l_i = l_i * alpha + sum(p) - 更新 softmax 分母
        # 这样 acc / l_i 最终就是正确的 softmax 注意力输出。
        # compute running maximum
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp(qk - m_ij[:, None])
        p = tl.where(m_ij[:, None] == float("-inf"), 0.0, p)
        l_ij = tl.sum(p, axis=1)
        alpha = tl.exp(m_i - m_ij)
        alpha = tl.where(m_i == float("-inf"), 0.0, alpha)
        acc = acc * alpha[:, None]

        # update acc
        # 中文注释：加载 V 并更新输出累积器。边界块或 padded 维度需要掩码加载。
        if (
            start_n + BLOCK_SIZE > cur_batch_ctx_len
            or BLOCK_DMODEL != BLOCK_DMODEL_PADDED
        ):
            v_load = tl.load(
                V_cache + off_v,
                mask=dim_mask[None, :]
                & ((start_n + offs_bs_n[:, None]) < cur_batch_ctx_len),
                other=0.0,
            )  # [N,D]
        else:
            v_load = tl.load(V_cache + off_v)

        # 中文注释：FP8 KV cache 反量化 - 将 uint8 存储的 FP8 值转换为计算精度
        if v_load.dtype.is_fp8():
            v = (v_load.to(tl.float32) * tl.load(v_scale)).to(q.dtype)
        else:
            v = v_load
        p = p.to(v.dtype)

        acc = tl.dot(p, v, acc=acc, input_precision=IN_PRECISION)
        # # update m_i and l_i
        l_i = l_i * alpha + l_ij
        m_i = m_ij

    # ========================================================================
    # 中文注释：阶段 2 - Q 与新计算的 KV 计算注意力
    # ========================================================================
    # 新 KV 存储在内存中的 K/V 张量中（不是分页 KV cache），
    # 通过 ragged 索引直接寻址，不需要 block table。
    off_k = (
        offs_n[None, :] * stride_kbs
        + cur_kv_head * stride_kh
        + offs_d[:, None] * stride_kd
    )
    off_v = (
        offs_n[:, None] * stride_vbs
        + cur_kv_head * stride_vh
        + offs_d[None, :] * stride_vd
    )
    k_ptrs = K + off_k
    v_ptrs = V + off_v

    # block_mask is 0 when we're already past the current query length
    block_mask = tl.where(block_start_loc < cur_batch_query_len, 1, 0)

    # compute query against itself (causal among queries by default;
    # CAUSAL=False for bidirectional attention over query tokens, e.g. DFlash.)
    # 中文注释：设置 causal mask 的上界。
    # CAUSAL=True 时，每个 query token 只能 attend 到位置 <= 自己的 token（上三角掩码）。
    # CAUSAL=False 时（如 DFlash 双向注意力），所有 query token 可以 attend 到所有新 KV。
    if CAUSAL:
        key_range_upper = block_mask * (start_m + 1) * BLOCK_M
    else:
        q_len_pad = (cur_batch_query_len + BLOCK_N - 1) // BLOCK_N * BLOCK_N
        key_range_upper = block_mask * q_len_pad

    for start_n in tl.range(
        0,
        key_range_upper,
        BLOCK_N,
        loop_unroll_factor=num_unroll_request,
    ):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k = tl.load(
            k_ptrs + (cur_batch_in_all_start_index + start_n) * stride_kbs,
            mask=dim_mask[:, None]
            & ((start_n + offs_n[None, :]) < cur_batch_query_len),
            other=0.0,
        )

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk = tl.dot(q, k, acc=qk, input_precision=IN_PRECISION)
        qk *= sm_scale

        valid_kv = (start_n + offs_n[None, :]) < cur_batch_query_len
        if CAUSAL:
            attn_mask = valid_kv & (offs_m[:, None] >= (start_n + offs_n[None, :]))
        else:
            attn_mask = valid_kv
        if SLIDING_WINDOW > 0:
            attn_mask = attn_mask & (
                offs_m[:, None] - (start_n + offs_n[None, :]) < SLIDING_WINDOW
            )
        qk = tl.where(attn_mask, qk, float("-inf"))

        # compute running maximum
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp(qk - m_ij[:, None])
        p = tl.where(m_ij[:, None] == float("-inf"), 0.0, p)
        l_ij = tl.sum(p, axis=1)
        alpha = tl.exp(m_i - m_ij)
        # To prevent NaN from appearing in the first round
        alpha = tl.where(m_i == float("-inf"), 0.0, alpha)
        acc = acc * alpha[:, None]

        # update acc
        v = tl.load(
            v_ptrs + (cur_batch_in_all_start_index + start_n) * stride_vbs,
            mask=dim_mask[None, :]
            & ((start_n + offs_n[:, None]) < cur_batch_query_len),
            other=0.0,
        )
        p = p.to(v.dtype)

        acc = tl.dot(p, v, acc=acc, input_precision=IN_PRECISION)
        # update m_i and l_i
        l_i = l_i * alpha + l_ij
        m_i = m_ij

    # 中文注释：最终归一化 - acc / l_i 得到正确的 softmax 加权注意力输出。
    # 1e-10 防止除零（当 l_i 为 0 时，即所有 KV 都被掩码的情况）。
    acc = acc / (l_i[:, None] + 1e-10)

    # initialize pointers to output
    off_o = (
        (cur_batch_in_all_start_index + offs_m[:, None]) * stride_obs
        + cur_head * stride_oh
        + offs_d[None, :] * stride_od
    )
    out_ptrs = Out + off_o
    # 中文注释：FP8 输出量化 - 将 float 精度的输出 clamp 到 FP8 范围后存储
    if USE_FP8:
        acc = acc * tl.load(out_scale_inv)
        acc = tl.clamp(acc, FP8_MIN, FP8_MAX)
    tl.store(
        out_ptrs, acc, mask=dim_mask[None, :] & (offs_m[:, None] < cur_batch_query_len)
    )
    return


# 中文注释：支持 ALiBi (Attention with Linear Biases) 的前向注意力内核。
# 【ALiBi 原理】
#   ALiBi 不使用可学习的位置编码，而是在注意力分数上加一个线性偏置：
#     bias = slope * (key_pos - query_pos)
#   其中 slope 是每个 head 独有的斜率（Alibi_slopes），且 bias <= 0（只惩罚远距离 token）。
#   不同 head 的 slope 不同，使得不同 head 关注不同距离范围的 token。
# 【与 _fwd_kernel 的区别】
#   - 额外加载 Alibi_slopes 并计算 alibi bias
#   - 不支持滑动窗口、sink tokens、FP8 输出
#   - 结构与 _fwd_kernel 基本相同，分两阶段计算 prefix KV 和新 KV
@triton.jit
def _fwd_kernel_alibi(
    Q,
    K,
    V,
    K_cache,
    V_cache,
    B_Loc,
    sm_scale,
    k_scale,
    v_scale,
    B_Start_Loc,
    B_Seqlen,
    Alibi_slopes,
    block_size,
    x,
    Out,
    stride_b_loc_b,
    stride_b_loc_s,
    stride_qbs,
    stride_qh,
    stride_qd,
    stride_kbs,
    stride_kh,
    stride_kd,
    stride_vbs,
    stride_vh,
    stride_vd,
    stride_obs,
    stride_oh,
    stride_od,
    stride_k_cache_bs,
    stride_k_cache_h,
    stride_k_cache_d,
    stride_k_cache_bl,
    stride_k_cache_x,
    stride_v_cache_bs,
    stride_v_cache_h,
    stride_v_cache_d,
    stride_v_cache_bl,
    num_queries_per_kv: int,
    IN_PRECISION: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,  # head size
    BLOCK_DMODEL_PADDED: tl.constexpr,  # head size padded to a power of 2
    BLOCK_N: tl.constexpr,
    SKIP_DECODE: tl.constexpr,
):
    # attn_bias[]
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    start_m = tl.program_id(2)

    cur_kv_head = cur_head // num_queries_per_kv

    # cur_batch_seq_len: the length of prompts
    # cur_batch_ctx_len: the length of prefix
    # cur_batch_in_all_start_index: the start id of the dim=0
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
    cur_batch_in_all_stop_index = tl.load(B_Start_Loc + cur_batch + 1)
    cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index
    cur_batch_ctx_len = cur_batch_seq_len - cur_batch_query_len

    if SKIP_DECODE and cur_batch_query_len == 1:
        return

    block_start_loc = BLOCK_M * start_m

    # initialize offsets
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL_PADDED)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_q = (
        (cur_batch_in_all_start_index + offs_m[:, None]) * stride_qbs
        + cur_head * stride_qh
        + offs_d[None, :] * stride_qd
    )

    dim_mask = tl.where(tl.arange(0, BLOCK_DMODEL_PADDED) < BLOCK_DMODEL, 1, 0).to(
        tl.int1
    )

    q = tl.load(
        Q + off_q,
        mask=dim_mask[None, :]
        & (offs_m[:, None] < cur_batch_seq_len - cur_batch_ctx_len),
        other=0.0,
    )

    # # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL_PADDED], dtype=tl.float32)

    alibi_slope = tl.load(Alibi_slopes + cur_head)
    alibi_start_q = tl.arange(0, BLOCK_M) + block_start_loc + cur_batch_ctx_len
    alibi_start_k = 0
    for start_n in range(0, cur_batch_ctx_len, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        bn = tl.load(
            B_Loc
            + cur_batch * stride_b_loc_b
            + ((start_n + offs_n) // block_size) * stride_b_loc_s,
            mask=(start_n + offs_n) < cur_batch_ctx_len,
            other=0,
        ).to(tl.int64)
        off_k = (
            bn[None, :] * stride_k_cache_bs
            + cur_kv_head * stride_k_cache_h
            + (offs_d[:, None] // x) * stride_k_cache_d
            + ((start_n + offs_n[None, :]) % block_size) * stride_k_cache_bl
            + (offs_d[:, None] % x) * stride_k_cache_x
        )
        off_v = (
            bn[:, None] * stride_v_cache_bs
            + cur_kv_head * stride_v_cache_h
            + offs_d[None, :] * stride_v_cache_d
            + (start_n + offs_n[:, None]) % block_size * stride_v_cache_bl
        )
        k_load = tl.load(
            K_cache + off_k,
            mask=dim_mask[:, None] & ((start_n + offs_n[None, :]) < cur_batch_ctx_len),
            other=0.0,
        )  # [D,N]

        if k_load.dtype.is_fp8():
            k = (k_load.to(tl.float32) * tl.load(k_scale)).to(q.dtype)
        else:
            k = k_load

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk = tl.dot(q, k, acc=qk, input_precision=IN_PRECISION)
        qk = tl.where(
            (start_n + offs_n[None, :]) < cur_batch_ctx_len, qk, float("-inf")
        )
        qk *= sm_scale

        # load alibi
        alibi = (
            tl.arange(0, BLOCK_N)[None, :] + alibi_start_k - alibi_start_q[:, None]
        ) * alibi_slope
        alibi = tl.where(
            (alibi <= 0) & (alibi_start_q[:, None] < cur_batch_seq_len),
            alibi,
            float("-inf"),
        )
        qk += alibi
        alibi_start_k += BLOCK_N

        # -- compute m_ij, p, l_ij
        m_ij = tl.max(qk, 1)
        m_i_new = tl.maximum(m_i, m_ij)
        p = tl.math.exp(qk - m_i_new[:, None])
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i

        alpha = tl.math.exp(m_i - m_i_new)
        l_i_new = alpha * l_i + l_ij
        # -- update output accumulator --
        # scale p
        # scale acc
        acc_scale = alpha
        # acc_scale = l_i / l_i_new * alpha
        acc = acc * acc_scale[:, None]
        # update acc
        v_load = tl.load(
            V_cache + off_v,
            mask=dim_mask[None, :] & ((start_n + offs_n[:, None]) < cur_batch_ctx_len),
            other=0.0,
        )
        if v_load.dtype.is_fp8():
            v = (v_load.to(tl.float32) * tl.load(v_scale)).to(q.dtype)
        else:
            v = v_load
        p = p.to(v.dtype)

        acc = tl.dot(p, v, acc=acc, input_precision="ieee")
        # update m_i and l_i
        l_i = l_i_new
        m_i = m_i_new

    off_k = (
        offs_n[None, :] * stride_kbs
        + cur_kv_head * stride_kh
        + offs_d[:, None] * stride_kd
    )
    off_v = (
        offs_n[:, None] * stride_vbs
        + cur_kv_head * stride_vh
        + offs_d[None, :] * stride_vd
    )
    k_ptrs = K + off_k
    v_ptrs = V + off_v

    block_mask = tl.where(block_start_loc < cur_batch_seq_len - cur_batch_ctx_len, 1, 0)

    # init alibi
    alibi_slope = tl.load(Alibi_slopes + cur_head)
    alibi_start_q = tl.arange(0, BLOCK_M) + block_start_loc + cur_batch_ctx_len
    alibi_start_k = cur_batch_ctx_len
    # # init debugger
    # offset_db_q = tl.arange(0, BLOCK_M) + block_start_loc
    # offset_db_k = tl.arange(0, BLOCK_N)
    # calc q[BLOCK_M, BLOCK_MODEL] mul k[prefix_len: , BLOCK_DMODEL]
    for start_n in range(0, block_mask * (start_m + 1) * BLOCK_M, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k = tl.load(
            k_ptrs + (cur_batch_in_all_start_index + start_n) * stride_kbs,
            mask=dim_mask[:, None]
            & ((start_n + offs_n[None, :]) < cur_batch_seq_len - cur_batch_ctx_len),
            other=0.0,
        )

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk = tl.dot(q, k, acc=qk, input_precision="ieee")
        qk *= sm_scale
        qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf"))

        # load alibi
        alibi = (
            tl.arange(0, BLOCK_N)[None, :] + alibi_start_k - alibi_start_q[:, None]
        ) * alibi_slope
        alibi = tl.where(
            (alibi <= 0) & (alibi_start_q[:, None] < cur_batch_seq_len),
            alibi,
            float("-inf"),
        )
        qk += alibi
        alibi_start_k += BLOCK_N

        # -- compute m_ij, p, l_ij
        m_ij = tl.max(qk, 1)
        m_i_new = tl.maximum(m_i, m_ij)
        p = tl.math.exp(qk - m_i_new[:, None])
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i

        alpha = tl.math.exp(m_i - m_i_new)
        l_i_new = alpha * l_i + l_ij
        # -- update output accumulator --
        # scale p
        # scale acc
        acc_scale = alpha
        # acc_scale = l_i / l_i_new * alpha
        acc = acc * acc_scale[:, None]
        # update acc
        v = tl.load(
            v_ptrs + (cur_batch_in_all_start_index + start_n) * stride_vbs,
            mask=dim_mask[None, :]
            & ((start_n + offs_n[:, None]) < cur_batch_seq_len - cur_batch_ctx_len),
            other=0.0,
        )
        p = p.to(v.dtype)

        acc = tl.dot(p, v, acc=acc, input_precision="ieee")
        # update m_i and l_i
        l_i = l_i_new
        m_i = m_i_new

    acc = acc / l_i[:, None]

    # initialize pointers to output
    off_o = (
        (cur_batch_in_all_start_index + offs_m[:, None]) * stride_obs
        + cur_head * stride_oh
        + offs_d[None, :] * stride_od
    )
    out_ptrs = Out + off_o
    tl.store(
        out_ptrs,
        acc,
        mask=dim_mask[None, :]
        & (offs_m[:, None] < cur_batch_seq_len - cur_batch_ctx_len),
    )
    return


@torch.inference_mode()
def context_attention_fwd(
    q,
    k,
    v,
    o,
    kv_cache_dtype: str,
    k_cache,
    v_cache,
    b_loc,
    b_start_loc,
    b_seq_len,
    max_seq_len,
    max_input_len,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    alibi_slopes=None,
    sliding_window=None,
    sm_scale=None,
    skip_decode=False,
    fp8_out_scale=None,
    sinks=None,
    is_block_table_ptr: bool = False,
    causal: bool = True,
):
    # 中文注释：prefix prefill 注意力计算的 Python 入口函数。
    # 【功能】
    #   计算 Q 与 prefix KV cache + 新 KV 的注意力输出。
    #   这是 prefix cache 命中场景下的核心计算函数。
    #
    # 【参数说明】
    #   q, k, v: 当前 step 的 Q/K/V 张量（ragged 格式，所有 batch 拼接）
    #   o: 输出张量（ragged 格式）
    #   k_cache, v_cache: 分页 KV cache，包含已缓存的历史 KV
    #   b_loc: block table [batch, max_blocks]，逻辑 block 到物理 block 的映射
    #   b_start_loc: 每个 batch 在 ragged 序列中的起始位置 [batch+1]
    #   b_seq_len: 每个 batch 的总序列长度 = prefix_len + query_len [batch]
    #   max_input_len: 最大的 query 长度（用于确定 kernel grid 大小）
    #   alibi_slopes: ALiBi 斜率（可选），为 None 时不使用 ALiBi
    #   sliding_window: 滑动窗口大小（可选），为 None 或 0 时不使用滑动窗口
    #   skip_decode: 是否跳过 decode 阶段（query_len==1）的计算
    #   causal: 是否使用 causal mask（默认 True）
    q_dtype_is_f32 = q.dtype is torch.float32

    # Turing does have tensor core for float32 multiplication
    # use ieee as fallback for triton kernels work. There is also
    # warning on vllm/config.py to inform users this fallback
    # implementation
    # 中文注释：Turing 架构 (T4) 的 tensor core 不支持 float32 矩阵乘，
    # 需要使用 IEEE 精度模式作为回退（性能较低但能保证正确性）。
    IN_PRECISION = "ieee" if IS_TURING and q_dtype_is_f32 else None

    # Conversion of FP8 Tensor from uint8 storage to
    # appropriate torch.dtype for interpretation by Triton
    # 中文注释：FP8 KV cache 类型转换。
    # FP8 数据在存储时使用 uint8 类型，使用时需要 view 为正确的 FP8 dtype
    # （如 float8_e4m3fn 或 float8_e5m2），以便 Triton 内核正确解释位模式。
    if "fp8" in kv_cache_dtype:
        assert k_cache.dtype in [torch.uint8, current_platform.fp8_dtype()]
        assert v_cache.dtype in [torch.uint8, current_platform.fp8_dtype()]

        if kv_cache_dtype in ("fp8", "fp8_e4m3"):
            target_dtype = current_platform.fp8_dtype()
        elif kv_cache_dtype == "fp8_e5m2":
            target_dtype = torch.float8_e5m2
        else:
            raise ValueError("Unsupported FP8 dtype:", kv_cache_dtype)

        k_cache = k_cache.view(target_dtype)
        v_cache = v_cache.view(target_dtype)

    if (
        k_cache.dtype == torch.uint8
        or v_cache.dtype == torch.uint8
        and kv_cache_dtype == "auto"
    ):
        raise ValueError(
            "kv_cache_dtype='auto' unsupported for\
            FP8 KV Cache prefill kernel"
        )

    # shape constraints
    # 中文注释：形状约束 - Q/K/V 的 head_size 必须相同
    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    assert Lq == Lk and Lk == Lv
    # round up Lk to a power of 2 - this is required for Triton block size
    # 中文注释：将 head_size 向上取整到 2 的幂次，Triton 的 block size 要求如此。
    # 例如 head_size=128 -> Lk_padded=128, head_size=96 -> Lk_padded=128
    Lk_padded = triton.next_power_of_2(Lk)

    # 中文注释：默认的注意力缩放因子 sm_scale = 1/sqrt(d)
    if sm_scale is None:
        sm_scale = 1.0 / (Lq**0.5)
    batch, head = b_seq_len.shape[0], q.shape[1]
    # 中文注释：GQA 比率 - 每个 KV head 对应多少个 Q head
    num_queries_per_kv = q.shape[1] // k.shape[1]

    assert batch + 1 == len(b_start_loc)

    # 0 means "disable"
    # 中文注释：滑动窗口大小，0 表示禁用（全局注意力）
    if sliding_window is None or sliding_window <= 0:
        sliding_window = 0

    # 中文注释：block table 预处理。
    # is_block_table_ptr=True 时，b_loc 存储的是物理内存地址指针，
    # 需要转换为 block 索引（除以 block 字节步长）。
    # is_block_table_ptr=False 时，b_loc 已经是 block 索引，直接转为 int32。
    if is_block_table_ptr:
        kv_element_size = k_cache.element_size()
        block_byte_stride = k_cache.stride(0) * kv_element_size
        # The physical starting point of the obtained KV Cache Pool
        base_addr = k_cache.data_ptr()

        mask = b_loc > 0
        processed_b_loc = torch.where(
            mask, (b_loc - base_addr) // block_byte_stride, b_loc
        ).to(torch.int32)
    else:
        processed_b_loc = b_loc.to(torch.int32)

    # 中文注释：根据是否有 ALiBi 选择不同的内核。
    # ALiBi 内核不支持非因果、sink tokens、FP8 输出等特性。
    if alibi_slopes is not None:
        assert causal, "Non-causal prefix attention is not supported with alibi"
        assert sinks is None, "Sinks arg is not supported with alibi"
        assert fp8_out_scale is None, "FP8 output not supported with alibi"
        # need to reduce num. blocks when using fp32
        # due to increased use of GPU shared memory
        # if q.dtype is torch.float32:
        # 中文注释：float32 使用更多共享内存，需要减小 block 大小
        BLOCK = BASE_BLOCK // 2 if q_dtype_is_f32 else BASE_BLOCK
        # batch, head,
        grid = (batch, head, triton.cdiv(max_input_len, BLOCK))
        _fwd_kernel_alibi[grid](
            q,
            k,
            v,
            k_cache,
            v_cache,
            b_loc,
            sm_scale,
            k_scale,
            v_scale,
            b_start_loc,
            b_seq_len,
            alibi_slopes,
            v_cache.shape[3],
            k_cache.shape[4],
            o,
            b_loc.stride(0),
            b_loc.stride(1),
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            o.stride(0),
            o.stride(1),
            o.stride(2),
            k_cache.stride(0),
            k_cache.stride(1),
            k_cache.stride(2),
            k_cache.stride(3),
            k_cache.stride(4),  # [num_blocks, num_kv_heads, head_size/x, block_size, x]
            v_cache.stride(0),
            v_cache.stride(1),
            v_cache.stride(2),
            v_cache.stride(3),  # [num_blocks, num_kv_heads, head_size, block_size]
            num_queries_per_kv=num_queries_per_kv,
            IN_PRECISION=IN_PRECISION,
            BLOCK_M=BLOCK,
            BLOCK_DMODEL=Lk,
            BLOCK_DMODEL_PADDED=Lk_padded,
            BLOCK_N=BLOCK,
            SKIP_DECODE=skip_decode,
            num_warps=NUM_WARPS,
            num_stages=1,
        )
        return

    max_seq_len = 0 if max_seq_len is None else max_seq_len
    extra_kargs: dict[str, Any] = {}
    if current_platform.is_rocm():
        extra_kargs = {}

    # 中文注释：根据 KV cache 的物理 block size 选择 Triton 分块参数。
    # 对于标准 2 的幂次 block size（如 128, 64），使用 BLOCK_M=128, BLOCK_N=64。
    # 对于非标准 block size（如 Qwen3-next 的 544），使用较小的 BLOCK_M=BLOCK_N=32，
    # 因为内核需要在 TRITON_BLOCK_SIZE (32) 的粒度上处理跨 block 边界的情况。
    real_block_size = v_cache.shape[3]
    is_pow2 = real_block_size > 0 and (real_block_size & (real_block_size - 1) == 0)
    # For standard models involving powers of 2,
    # follow the original logic (Llama 128/64)
    # For non-standard models (Qwen3-next block_size 544), set to 32.
    if is_pow2:
        BLOCK_M = 128
        BLOCK_N = 64
    else:
        BLOCK_M = 32
        BLOCK_N = 32

    # TRITON_BLOCK_SIZE is kept at 32 to ensure
    # correct alignment logic when the kernel handles
    # non-standard sizes (such as 544).
    # 中文注释：Triton 内部处理的逻辑块大小，固定为 32。
    # 内核在这个粒度上进行 block table 查表和地址计算。
    TRITON_BLOCK_SIZE = 32

    grid_fn = lambda META: (batch, head, triton.cdiv(max_input_len, META["BLOCK_M"]))
    _fwd_kernel[grid_fn](
        q,
        k,
        v,
        k_cache,
        v_cache,
        sinks,
        processed_b_loc,
        sm_scale,
        k_scale,
        v_scale,
        1.0 / fp8_out_scale if fp8_out_scale is not None else 1.0,
        b_start_loc,
        b_seq_len,
        k_cache.shape[4],
        o,
        processed_b_loc.stride(0),
        processed_b_loc.stride(1),
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        stride_k_cache_bs=k_cache.stride(0),
        stride_k_cache_h=k_cache.stride(1),
        stride_k_cache_d=k_cache.stride(2),
        stride_k_cache_bl=k_cache.stride(3),
        stride_k_cache_x=k_cache.stride(4),
        stride_v_cache_bs=v_cache.stride(0),
        stride_v_cache_h=v_cache.stride(1),
        stride_v_cache_d=v_cache.stride(2),
        stride_v_cache_bl=v_cache.stride(3),
        BLOCK_SIZE=TRITON_BLOCK_SIZE,
        PHYSICAL_BLOCK_SIZE=real_block_size,
        num_queries_per_kv=num_queries_per_kv,
        IN_PRECISION=IN_PRECISION,
        BLOCK_DMODEL=Lk,
        BLOCK_DMODEL_PADDED=Lk_padded,
        SLIDING_WINDOW=sliding_window,
        SKIP_DECODE=skip_decode,
        USE_FP8=fp8_out_scale is not None,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_unroll_cache=4,
        num_unroll_request=1,
        num_warps=4,
        num_stages=1,
        USE_SINKS=sinks is not None,
        CAUSAL=causal,
        **extra_kargs,
    )
    return
