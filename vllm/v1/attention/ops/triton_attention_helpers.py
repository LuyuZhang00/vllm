# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared ``@triton.jit`` helpers used by the unified attention kernel
and ``reduce_segments``.

These are plain attention-loop helpers — mask building, ALiBi / QQ-bias
score post-processing, online-softmax bookkeeping, tile-loop bounds,
sequence lookup — extracted so the 2D and 3D paths of the unified
kernel (and any future consumer) share a single implementation.
"""

from __future__ import annotations

from vllm.triton_utils import tl, triton

# ===========================================================================
# 标量辅助函数（被所有 kernel 和 reduce_segments 复用）
# ===========================================================================


@triton.jit
def cdiv_fn(x, y):
    """Ceiling division.  Kept as a helper to keep kernel bodies terse."""
    # 向上取整除法：(x + y - 1) // y
    return (x + y - 1) // y


@triton.jit
def apply_softcap(S, x):
    """Softcap (aka tanh-style clamp) used to bound attention scores.

    ``x * tanh(S / x)`` rewritten to avoid a direct ``tanh`` call.

    软上限函数（softcap），用于限制注意力分数的范围。
    将 tanh(S/x) * x 用指数函数重写，避免直接调用 tanh。
    数学上等价于：x * (exp(S/x) - exp(-S/x)) / (exp(S/x) + exp(-S/x))
    """
    Sdiv = S / x
    p1 = tl.exp(Sdiv)
    p2 = tl.exp(-Sdiv)
    return x * (p1 - p2) / (p1 + p2)


# ===========================================================================
# 注意力循环辅助函数
# ===========================================================================


@triton.jit
def resolve_seq_and_query_len(
    # 根据全局 Q 块索引解析对应的序列和 Q 块信息
    # 所有注意力 kernel 共享此函数，用于从扁平化的 (seq, q_block) 空间中
    # 恢复 (序列索引, 序列内 Q 块索引) 对
    query_start_len_ptr,
    seq_lens_ptr,
    q_block_global_idx,
    num_seqs,
    BLOCK_Q: tl.constexpr,
):
    """Resolve the (sequence, q-block-within-sequence) pair and load the
    per-sequence lengths.

    Shared across every attention kernel — the ``q_block_global_idx``
    program id indexes into the flattened ``(seq, q_block_in_seq)``
    space, and a binary search over ``query_start_len_ptr`` recovers
    the (seq, local-q-block) pair.

    Returns ``(seq_idx, q_block_local_idx, cur_batch_in_all_start_index,
    cur_batch_query_len, seq_len)``.  Callers must still early-return
    when ``q_block_local_idx * BLOCK_Q >= cur_batch_query_len`` (Triton
    helpers cannot return from the caller).

    解析 (序列, 序列内 Q 块) 对，并加载每个序列的长度信息。

    所有注意力 kernel 共享此函数。q_block_global_idx 程序 ID 索引到
    扁平化的 (seq, q_block_in_seq) 空间，通过二分查找 query_start_len_ptr
    恢复 (序列索引, 局部 Q 块索引) 对。

    返回 (seq_idx, q_block_local_idx, cur_batch_in_all_start_index,
           cur_batch_query_len, seq_len)。
    调用者仍需在 q_block_local_idx * BLOCK_Q >= cur_batch_query_len 时
    提前返回（Triton 辅助函数无法从调用者处返回）。
    """
    # 通过二分查找确定当前程序对应的序列索引
    seq_idx = find_seq_idx(
        query_start_len_ptr, q_block_global_idx, num_seqs, BLOCK_Q, True
    )
    # 计算该序列的 Q 块起始索引和局部 Q 块索引
    q_block_start_idx = tl.load(query_start_len_ptr + seq_idx) // BLOCK_Q + seq_idx
    q_block_local_idx = q_block_global_idx - q_block_start_idx
    # 加载该序列的 Q 起始位置、Q 长度和 KV 序列长度
    cur_start = tl.load(query_start_len_ptr + seq_idx)
    cur_stop = tl.load(query_start_len_ptr + seq_idx + 1)
    cur_batch_query_len = cur_stop - cur_start
    seq_len = tl.load(seq_lens_ptr + seq_idx)
    return seq_idx, q_block_local_idx, cur_start, cur_batch_query_len, seq_len


@triton.jit
def find_seq_idx(
    query_start_len_ptr,
    target_idx,
    num_seqs,
    BLOCK_Q: tl.constexpr,
    use_q_block_mode: tl.constexpr,
):
    """Binary search over the cumulative query-length prefix.

    When ``use_q_block_mode`` is True, the prefix values are reshaped
    into units of ``BLOCK_Q`` plus one entry per boundary — matching
    the q-block grid laid out by the attention kernels.  When False
    we search the plain cumulative-length prefix (used by
    ``reduce_segments`` which iterates over raw query tokens).

    在累积查询长度前缀上进行二分查找。

    当 use_q_block_mode 为 True 时，前缀值被重塑为 BLOCK_Q 单位
    加上每个边界一个条目，匹配注意力 kernel 布置的 Q 块网格。
    当为 False 时，搜索普通的累积长度前缀
    （用于 reduce_segments，它遍历原始查询 token）。
    """
    # 标准二分查找算法
    left: tl.int32 = 0
    right = num_seqs
    while left < right:
        mid = (left + right) // 2
        val = tl.load(query_start_len_ptr + mid)
        # 在 Q 块模式下，将累积长度转换为 Q 块索引（加上序列边界偏移）
        mid_val = val // BLOCK_Q + mid if use_q_block_mode else val

        if mid_val <= target_idx:
            left = mid + 1
        else:
            right = mid

    return left - 1


@triton.jit
def init_softmax_M(
    sink_ptr,
    query_offset_1,
    query_mask_1,
    segm_idx_or_0,
    BLOCK_M: tl.constexpr,
    USE_SINKS: tl.constexpr,
    IS_3D: tl.constexpr,
):
    """Initial row-max ``M`` for the online softmax.

    Without sinks: ``-inf``.  With sinks: load the per-head sink bias
    once.  In 3D mode only segment 0 loads — ``reduce_segments`` adds
    the sink contribution exactly once across segments, so other
    segments must start from ``-inf``.

    ``segm_idx_or_0`` is the 3D segment index or 0 for 2D (caller
    passes ``0`` when ``IS_3D`` is False).

    初始化在线 softmax 的行最大值 M。

    无 sink token 时：初始化为 -inf。
    有 sink token 时：加载每个 head 的 sink 偏置。
    在 3D 模式下只有 segment 0 加载 sink 偏置，
    因为 reduce_segments 只在所有 segment 中添加一次 sink 贡献，
    其他 segment 必须从 -inf 开始。

    segm_idx_or_0 是 3D segment 索引，2D 模式下为 0
    （调用者在 IS_3D 为 False 时传入 0）。
    """
    M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    if USE_SINKS:
        load_sinks = (not IS_3D) or (segm_idx_or_0 == 0)
        if load_sinks:
            M = tl.load(
                sink_ptr + query_offset_1,
                mask=query_mask_1,
                other=float("-inf"),
            ).to(tl.float32)
    return M


@triton.jit
def compute_tile_loop_bounds(
    # 计算 KV 块循环的边界 (loop_lo, loop_hi) 和最大序列前缀长度
    # 综合考虑三个因素：
    # 1. 当前 Q 块中任何查询 token 覆盖的最长序列前缀
    # 2. 滑动窗口注意力的裁剪
    # 3. 3D 模式下 segment 的作用域限制
    context_len,
    seq_len,
    cur_batch_query_len,
    q_block_local_idx,
    segm_idx_or_0,
    tiles_per_segment_or_0,
    TILE_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    num_queries_per_kv: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr,
    USE_MM_PREFIX: tl.constexpr,
    IS_3D: tl.constexpr,
    CHUNK_LOOKBACK: tl.constexpr = -1,
    CHUNK_SIZE: tl.constexpr = -1,
):
    """Compute the tile-loop bounds ``(loop_lo, loop_hi)`` and the
    derived ``max_seq_prefix_len`` used for per-tile masking.

    Combines three concerns into one helper:

    1. Longest prefix spanned by any query token in this q-block.
       Clamped to ``seq_len`` (causal) or extended to it when
       mm_prefix is active (bidirectional ranges can reach past the
       causal prefix).
    2. Sliding-window pruning: narrows ``[tile_start, tile_end)`` to
       only tiles that can contain an allowed key under SWA.
    3. 3D scoping: when ``IS_3D`` is True, further narrows to the
       segment's slice via ``(segm_idx * tiles_per_segment,
       (segm_idx + 1) * tiles_per_segment)``.

    计算 KV 块循环边界 (loop_lo, loop_hi) 和用于逐块掩码的
    max_seq_prefix_len。

    将三个关注点合并到一个辅助函数中：

    1. 当前 Q 块中任何查询 token 覆盖的最长前缀。
       在因果模式下截断到 seq_len，在 mm_prefix 活跃时
       扩展到 seq_len（双向范围可以超过因果前缀）。
    2. 滑动窗口裁剪：将 [tile_start, tile_end) 缩小到
       只包含 SWA 下允许的键的块。
    3. 3D 作用域：当 IS_3D 为 True 时，进一步缩小到
       segment 的切片。
    """
    # 计算当前 Q 块中任何查询 token 覆盖的最长序列前缀长度
    # 公式：context_len + q_block_local_idx * BLOCK_Q + (BLOCK_M - 1) // num_queries_per_kv + 1
    max_seq_prefix_len = (
        context_len
        + q_block_local_idx * BLOCK_Q
        + (BLOCK_M - 1) // num_queries_per_kv
        + 1
    )
    if USE_MM_PREFIX:
        # image bidirectional attention ranges require a full range
        # including q_block padding to make sure doc mask is correct
        max_seq_prefix_len = tl.maximum(max_seq_prefix_len, seq_len)
    else:
        max_seq_prefix_len = tl.minimum(max_seq_prefix_len, seq_len)

    num_tiles = cdiv_fn(max_seq_prefix_len, TILE_SIZE)

    # ---- 滑动窗口裁剪 --------------------
    # 默认：保持之前的全局行为
    tile_start = 0
    tile_end = num_tiles
    # TODO(Isotr0py): sliding window pruning with image bidirectional mask
    if SLIDING_WINDOW > 0 and not USE_MM_PREFIX:
        # Query rows covered by this Q-block
        qpos_lo = q_block_local_idx * BLOCK_Q
        qpos_hi = tl.minimum(
            qpos_lo + (BLOCK_M - 1) // num_queries_per_kv,
            cur_batch_query_len - 1,
        )
        # For sliding window, each query position q can only attend to
        # keys in the range [q_abs - SLIDING_WINDOW + 1, q_abs]
        # where q_abs = context_len + q
        # The union of allowed key positions for this Q-block is:
        # [context_len + qpos_lo - SLIDING_WINDOW + 1, context_len + qpos_hi]
        q_abs = context_len + qpos_lo
        if CHUNK_LOOKBACK > -1:
            # Chunked attention: align lower bound to the start of the
            # lookback'th previous chunk.
            first_allowed_key = ((q_abs // CHUNK_SIZE) - CHUNK_LOOKBACK) * CHUNK_SIZE
        else:
            first_allowed_key = q_abs - SLIDING_WINDOW + 1
        last_allowed_key = context_len + qpos_hi
        # Convert to tile indices and clamp
        tile_start = tl.maximum(0, first_allowed_key // TILE_SIZE)
        tile_end = tl.minimum((last_allowed_key // TILE_SIZE) + 1, num_tiles)

    # 3D 模式下进一步缩小到 segment 的切片
    if IS_3D:
        loop_lo = max(segm_idx_or_0 * tiles_per_segment_or_0, tile_start)
        loop_hi = min((segm_idx_or_0 + 1) * tiles_per_segment_or_0, tile_end)
    else:
        loop_lo = tile_start
        loop_hi = tile_end

    return loop_lo, loop_hi, max_seq_prefix_len


@triton.jit
def store_segm_reduce_scalars(
    # 存储每个 segment 的 M（行最大值）和 L（指数和），
    # 供 reduce_segments 合并为最终的 softmax 结果
    # 所有 3D 注意力尾声共享此函数
    segm_max_ptr,
    segm_expsum_ptr,
    query_offset_0,
    query_offset_1,
    segm_idx,
    M,
    L,
    query_mask_0,
    query_mask_1,
    num_query_heads: tl.constexpr,
    NUM_SEGMENTS_PER_SEQ: tl.constexpr,
):
    """Store per-segment ``M`` and ``L`` for ``reduce_segments`` to
    combine into the final softmax.

    Shared across every 3D attention epilogue; the per-token output
    stripes are mode-specific (flat / 2-stream split / 4-stream split)
    and stay inlined.

    存储每个 segment 的 M（行最大值）和 L（指数和），
    供 reduce_segments 合并为最终的 softmax 结果。

    所有 3D 注意力尾声共享此函数；逐 token 的输出条带
    是模式特定的（扁平 / 2 流分割 / 4 流分割）并保持内联。
    """
    segm_offset = (
        query_offset_0.to(tl.int64) * (num_query_heads * NUM_SEGMENTS_PER_SEQ)
        + query_offset_1 * NUM_SEGMENTS_PER_SEQ
        + segm_idx
    )
    tl.store(segm_max_ptr + segm_offset, M, mask=query_mask_0 & query_mask_1)
    tl.store(segm_expsum_ptr + segm_offset, L, mask=query_mask_0 & query_mask_1)


@triton.jit
def compute_kv_seq_mask(
    # 构建一个 KV 块的注意力掩码
    # 默认因果掩码（key <= query）；与分块注意力或滑动窗口进行 AND 运算；
    # 与 mm_prefix 的双向范围进行 OR 运算
    # 顺序匹配 FlexAttention：(因果 AND 窗口) OR mm_prefix
    query_abs_pos,
    seq_offset,
    seq_idx,
    mm_prefix_range_ptr,
    SLIDING_WINDOW: tl.constexpr,
    USE_MM_PREFIX: tl.constexpr,
    MAX_MM_RANGES: tl.constexpr,
    CHUNK_LOOKBACK: tl.constexpr = -1,
    CHUNK_SIZE: tl.constexpr = -1,
):
    """Build the KV mask for one tile.

    Causal (key <= query) by default; AND-ed with either chunked
    attention (``CHUNK_LOOKBACK >= 0``) or sliding window
    (``SLIDING_WINDOW > 0``); OR-ed with the bidirectional ranges from
    ``mm_prefix_range`` when PrefixLM / multimodal attention is active.
    Order matches FlexAttention: ``(causal AND window) OR mm_prefix``.
    Chunked attention takes precedence over sliding window when both
    are non-default — the launcher zeros ``CHUNK_LOOKBACK`` whenever
    sliding window is disabled.

    构建一个 KV 块的注意力掩码。

    默认因果掩码（key <= query）；
    与分块注意力（CHUNK_LOOKBACK >= 0）或滑动窗口（SLIDING_WINDOW > 0）
    进行 AND 运算；
    当 PrefixLM / 多模态注意力活跃时，与 mm_prefix_range 的双向范围
    进行 OR 运算。
    顺序匹配 FlexAttention：(因果 AND 窗口) OR mm_prefix。
    当两者都非默认时，分块注意力优先于滑动窗口
    —— 启动器在滑动窗口禁用时将 CHUNK_LOOKBACK 置零。
    """
    # 计算注意力掩码：默认因果掩码（key <= query）
    seq_mask = seq_offset[None, :] <= query_abs_pos

    # 在 mm_prefix OR 之前应用滑动窗口 / 分块注意力到基础掩码
    # 顺序必须匹配 FlexAttention：(因果 AND 滑动窗口) OR mm_prefix
    if CHUNK_LOOKBACK > -1:
        # 分块注意力：只允许查询 token 回看 CHUNK_LOOKBACK 个块
        seq_mask = seq_mask & (
            (query_abs_pos // CHUNK_SIZE - seq_offset[None, :] // CHUNK_SIZE)
            <= CHUNK_LOOKBACK
        )
    elif SLIDING_WINDOW > 0:
        # 滑动窗口：只允许查询 token 注意窗口内的键
        seq_mask = seq_mask & ((query_abs_pos - seq_offset) < SLIDING_WINDOW)

    # PrefixLM：用多模态 token 的双向范围扩展掩码
    # 在滑动窗口之后应用，使 mm_prefix 范围可以覆盖 SW 限制
    if USE_MM_PREFIX:
        for i in range(MAX_MM_RANGES):
            range_start = tl.load(
                mm_prefix_range_ptr + seq_idx * MAX_MM_RANGES * 2 + i * 2
            )
            range_end = tl.load(
                mm_prefix_range_ptr + seq_idx * MAX_MM_RANGES * 2 + i * 2 + 1
            )
            is_valid = range_start < range_end
            q_in_range = (
                (query_abs_pos >= range_start) & (query_abs_pos <= range_end) & is_valid
            )
            k_in_range = (
                (seq_offset[None, :] >= range_start)
                & (seq_offset[None, :] <= range_end)
                & is_valid
            )
            seq_mask |= q_in_range & k_in_range
    return seq_mask


@triton.jit
def apply_alibi_to_score(
    # 向注意力分数 S 添加 ALiBi 位置偏置（线性或平方根变体）
    # ALiBi 是一种不使用位置编码的位置感知方法，
    # 通过向注意力分数添加与距离成比例的偏置来实现
    S,
    alibi_slope,
    seq_offset,
    context_len,
    query_pos,
    USE_ALIBI_SQRT: tl.constexpr,
):
    """Add the ALiBi positional bias (linear or sqrt variant) to S in-place.

    将 ALiBi 位置偏置（线性或平方根变体）就地添加到 S。
    """
    if USE_ALIBI_SQRT:
        relative_pos = seq_offset - (context_len + query_pos[:, None])
        alibi_offset = tl.where(
            relative_pos <= 0,
            -tl.sqrt((-relative_pos).to(tl.float32)),
            0.0,
        )
    else:
        alibi_offset = seq_offset - context_len
    return S + alibi_slope[:, None] * alibi_offset


@triton.jit
def load_qq_bias_tile(
    # 加载对应查询行的 QQ 偏置切片
    # QQ 偏置用于查询-查询之间的注意力偏置
    qq_bias_row_ptrs,
    seq_offset,
    context_len,
    qq_bias_stride_0,
):
    """Load the qq-bias slice for keys that correspond to query rows.

    加载对应查询行的 QQ 偏置切片。
    """
    key_rel_pos = seq_offset - context_len
    is_query_key = key_rel_pos >= 0 and key_rel_pos < qq_bias_stride_0
    return tl.load(
        qq_bias_row_ptrs + key_rel_pos[None, :],
        mask=is_query_key[None, :],
        other=0.0,
    )


@triton.jit
def softmax_step(S, M, L):
    """Online softmax update for one tile.

    Returns ``(M_new, L_new, P, alpha)``.  Caller is responsible for
    rescaling its accumulator(s) by ``alpha[:, None]`` — done outside so
    kernels with a different number / shape of accumulators can reuse
    the same step.

    一个 KV 块的在线 softmax 更新。

    返回 (M_new, L_new, P, alpha)。调用者负责将其累加器
    乘以 alpha[:, None] —— 在外部完成，以便不同数量/形状的
    累加器的 kernel 可以复用相同的步骤。

    在线 softmax 算法的核心思想：
    1. 维护当前看到的最大值 M 和指数和 L
    2. 每个新块到来时，用新的最大值更新 M，并相应缩放旧的 L
    3. 这样可以在单次遍历中计算 softmax，无需存储所有注意力分数
    """
    # 计算运行最大值
    # m_j : (BLOCK_M,) - 旧 M 和当前块最大值的较大者
    m_j = tl.maximum(M, tl.max(S, axis=1))
    # 滑动窗口可能导致整行被掩码为 -inf，
    # 此时需要将 m_j 设为 0 以避免 NaN
    m_j = tl.where(m_j > float("-inf"), m_j, 0.0)
    # P : (BLOCK_M, TILE_SIZE) - 当前块的 softmax 权重
    P = tl.exp(S - m_j[:, None])
    # l_j : (BLOCK_M,) - 当前块的指数和
    l_j = tl.sum(P, axis=1)
    # alpha : (BLOCK_M,) - 旧累加器需要的缩放因子
    alpha = tl.exp(M - m_j)
    # 更新常量：旧的 L 缩放后加上新块的贡献
    L_new = L * alpha + l_j
    return m_j, L_new, P, alpha
