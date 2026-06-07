# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# =============================================================================
# common.py - 通用注意力辅助操作
#
# 本文件包含三类核心功能：
# 1. 上下文并行 (Context Parallelism, CP) 注意力输出校正：
#    当使用 CP 将序列切分到多个 rank 并行计算注意力时，每个 rank 只计算
#    部分 KV 的注意力输出。本文件提供了将各 rank 的 LSE (Log-Sum-Exp)
#    进行 all-gather 后，校正本地注意力输出并合并为全局结果的 Triton 内核
#    和 Python 封装函数。
#
# 2. 序列打包 (pack_seq)：
#    将变长序列 [N, D] 按 batch 维度打包为固定长度的 [B, Lmax, D] 张量，
#    用于将 ragged 序列转为 padded batch 格式，支持 float 和 uint8 dtype。
#
# 3. 序列解包 (unpack_seq)：
#    打包的逆操作，将 [B, Lmax, D] 恢复为 [N, D] 的变长序列格式。
# =============================================================================

# 中文注释：本文件是 vLLM V1 注意力子系统中的通用辅助操作模块，主要服务于
# 以下场景：
#
# 【上下文并行 (CP) 注意力输出校正】
#   在分布式推理中，长序列的注意力计算可以通过上下文并行 (Context Parallelism)
#   切分到多个 GPU 上并行执行。每个 rank 独立计算其负责的 KV 片段的注意力，
#   并记录对应的 LSE (Log-Sum-Exp) 值。由于 softmax 归一化需要全局分母，
#   各 rank 的输出不能直接相加。本模块提供的 Triton 内核和封装函数会：
#     (1) 通过 all-gather 收集所有 rank 的 LSE
#     (2) 用数值稳定的 log-sum-exp 算法聚合为全局 LSE
#     (3) 用 exp(LSE_local - LSE_global) 作为校正因子缩放本地输出
#   校正后各 rank 的输出可以直接通过 reduce-scatter 或 all-reduce 合并。
#
# 【序列打包/解包】
#   在变长序列批处理中，需要将 ragged 格式（所有 token 拼成一维 [N, D]）
#   转换为 padded batch 格式（[B, Lmax, D]），以便进行批量矩阵运算。
#   pack_seq_triton 和 unpack_seq_triton 使用 Triton 内核高效完成这一转换，
#   支持 float 和 uint8 两种 dtype。
# =============================================================================

import torch

from vllm.distributed.parallel_state import GroupCoordinator
from vllm.triton_utils import tl, triton


# 中文注释：Triton JIT 内核，用于校正上下文并行 (CP) 各 rank 的注意力输出。
# 当 CP 将序列切分到 N 个 rank 时，每个 rank 独立计算各自 KV 片段的注意力输出
# 及对应的 LSE (log-sum-exp)。此内核接收 all-gather 后的全部 N 个 rank 的 LSE，
# 执行以下操作：
#   (1) 计算全局 LSE：将各 rank 的 LSE 做 log-sum-exp 聚合，得到全局 LSE。
#   (2) 校正本地输出：用本地 rank 的 LSE 与全局 LSE 的差值作为缩放因子，
#       乘以本地注意力输出，使其成为加权求和中的正确权重。
# 这样经过校正后，各 rank 的输出可以直接通过 reduce-scatter 或 all-reduce 合并。
@triton.jit
def _correct_attn_cp_out_kernel(
    outputs_ptr,
    new_output_ptr,
    lses_ptr,
    vlse_ptr,
    outputs_stride_B,
    outputs_stride_H,
    outputs_stride_D,
    lses_stride_N,
    lses_stride_B,
    lses_stride_H,
    lse_idx,
    HEAD_DIM: tl.constexpr,
    N_ROUNDED: tl.constexpr,
    IS_BASE_E: tl.constexpr,
):
    """
    Apply the all-gathered lses to correct each local rank's attention
    output. we still need perform a cross-rank reduction to obtain the
    final attention output.

    Args:
        outputs_ptr (triton.PointerType):
            Pointer to input tensor of shape [ B, H, D ]
        lses_ptr (triton.PointerType):
            Pointer to input tensor of shape [ N, B, H ]
        new_output_ptr (triton.PointerType):
            Pointer to output tensor of shape [ B, H, D ]
        vlse_ptr (triton.PointerType):
            Pointer to output tensor of shape [ B, H ]
    """
    batch_idx = tl.program_id(axis=0).to(tl.int64)
    head_idx = tl.program_id(axis=1).to(tl.int64)
    d_offsets = tl.arange(0, HEAD_DIM)
    num_n_offsets = tl.arange(0, N_ROUNDED)

    # shape = [N]
    # 中文注释：计算所有 N 个 rank 的 LSE 在内存中的偏移地址
    lse_offsets = (
        num_n_offsets * lses_stride_N
        + batch_idx * lses_stride_B
        + head_idx * lses_stride_H
    )

    # calc final lse
    # 中文注释：加载所有 rank 的 LSE 值，并计算全局聚合 LSE。
    # 步骤1：加载各 rank 的 LSE，并将 NaN 和 +inf 替换为 -inf（数值安全处理）
    lse = tl.load(lses_ptr + lse_offsets)
    lse = tl.where((lse != lse) | (lse == float("inf")), -float("inf"), lse)
    # 步骤2：数值稳定的 log-sum-exp 聚合
    # 先减去最大值防止 exp 溢出，再求和取对数，最后加回最大值
    lse_max = tl.max(lse, axis=0)
    lse_max = tl.where(lse_max == -float("inf"), 0, lse_max)
    lse -= lse_max
    if IS_BASE_E:
        lse_exp = tl.exp(lse)
        lse_acc = tl.sum(lse_exp, axis=0)
        lse = tl.log(lse_acc)
    else:
        lse_exp = tl.exp2(lse)
        lse_acc = tl.sum(lse_exp, axis=0)
        lse = tl.log2(lse_acc)
    lse += lse_max

    # 中文注释：将全局聚合的 LSE 存储到输出缓冲区
    lse_offsets = batch_idx * lses_stride_B + head_idx * lses_stride_H
    tl.store(vlse_ptr + lse_offsets, lse)

    # shape = [D]
    # 中文注释：计算当前 batch 和 head 对应的输出张量偏移
    output_offsets = (
        batch_idx * outputs_stride_B
        + head_idx * outputs_stride_H
        + d_offsets * outputs_stride_D
    )

    # correct output
    # 中文注释：校正当前 rank (由 lse_idx 指定) 的注意力输出。
    # 校正公式：output_corrected = output_local * exp(LSE_local - LSE_global)
    # 这使得各 rank 校正后的输出可以直接相加得到正确的全局注意力结果
    lse_offset = (
        lse_idx * lses_stride_N + batch_idx * lses_stride_B + head_idx * lses_stride_H
    )
    lse_tmp = tl.load(lses_ptr + lse_offset)
    lse_finally = lse_tmp - lse
    lse_finally = tl.where(
        (lse_finally != lse_finally) | (lse_finally == float("inf")),
        -float("inf"),
        lse_finally,
    )
    factor = tl.exp(lse_finally) if IS_BASE_E else tl.exp2(lse_finally)
    output = tl.load(outputs_ptr + output_offsets)
    output = output * factor

    tl.store(new_output_ptr + output_offsets, output)


class CPTritonContext:
    """The CPTritonContext is used to avoid recompilation of the Triton JIT."""

    # 中文注释：缓存已编译的 Triton 内核对象，避免重复编译。
    # Triton JIT 内核首次调用时需要编译，后续调用可以直接复用编译后的内核。
    # 首次调用传入 const_args 用于特化编译，后续调用只传 regular_args。

    # 中文注释：Triton JIT 内核的编译缓存上下文。
    # 工作原理：
    #   1. Triton JIT 内核首次调用时会触发编译（包括 const_args 的特化），
    #      编译过程可能耗时数百毫秒。
    #   2. 首次调用后，编译好的内核对象被缓存在 self.inner_kernel 中。
    #   3. 后续调用直接使用缓存的内核对象，只传 regular_args，跳过编译。
    # 使用方式：
    #   ctx = CPTritonContext()
    #   ctx.call_kernel(my_kernel, grid, *regular_args, **const_args)  # 首次编译
    #   ctx.call_kernel(my_kernel, grid, *regular_args)               # 后续复用
    def __init__(self):
        self.inner_kernel = None

    def call_kernel(self, kernel, grid, *regular_args, **const_args):
        # 中文注释：调用 Triton 内核，首次调用时编译并缓存，后续直接复用。
        # kernel[grid] 是 Triton 的内核启动语法，grid 指定 GPU 线程块的三维网格大小。
        if self.inner_kernel is None:
            self.inner_kernel = kernel[grid](*regular_args, **const_args)
        else:
            self.inner_kernel[grid](*regular_args)


def correct_attn_out(
    out: torch.Tensor,
    lses: torch.Tensor,
    cp_rank: int,
    ctx: CPTritonContext,
    is_lse_base_on_e: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Correct the attention output using the all-gathered lses.

    Args:
        out: Tensor of shape [ B, H, D ]
        lses: Tensor of shape [ N, B, H ]
        cp_rank: Current rank in the context-parallel group
        ctx: Triton context to avoid recompilation

    Returns:
        Tuple of (out, lse) with corrected attention and final log-sum-exp.
    """
    # 中文注释：校正上下文并行 (CP) 中某个 rank 的注意力输出。
    # 【数学原理】
    #   在 CP 模式下，每个 rank i 计算的注意力输出可以表示为：
    #     out_i = sum_j(exp(QK_j^T/sqrt(d)) * V_j) / exp(LSE_i)
    #   其中 LSE_i = log(sum_j(exp(QK_j^T/sqrt(d)))) 是本地的 log-sum-exp。
    #   全局正确的注意力输出需要所有 rank 的 KV 参与计算，即：
    #     out_global = (sum_i out_i * exp(LSE_i)) / exp(LSE_global)
    #   校正后每个 rank 的输出变为：
    #     out_corrected_i = out_i * exp(LSE_i - LSE_global)
    #   各 rank 校正后直接相加即可得到全局正确的结果。
    #
    # 输入：
    #   out  - 当前 rank 的注意力输出 [B, H, D]
    #   lses - all-gather 后的所有 rank 的 LSE [N, B, H]，N 为 CP world_size
    #   cp_rank - 当前 rank 在 CP 组中的编号
    # 输出：
    #   out - 校正后的注意力输出（可以直接与其他 rank 的输出相加）
    #   lse - 全局聚合后的 LSE [B, H]
    if ctx is None:
        ctx = CPTritonContext()

    # --- Normalize to 3D views ---
    if out.ndim == 4 and out.shape[1] == 1:
        out = out.squeeze(1)
    assert out.ndim == 3, f"expected out [B,H,D] or [B,1,H,D], got {tuple(out.shape)}"

    if lses.ndim == 4 and lses.shape[-1] == 1:
        lses = lses.squeeze(-1)
    if lses.ndim == 4 and lses.shape[1] == 1:
        lses = lses.squeeze(1)
    assert lses.ndim == 3, (
        f"expected lses [N,B,H] (optionally with a 1-sized extra dim), "
        f"got {tuple(lses.shape)}"
    )

    B, H, D = out.shape
    N = lses.shape[0]

    # Strides after we normalized shapes to 3-D views.  The kernel computes
    # offsets for `vlse_ptr` using lses_stride_B/H, so the output buffer must
    # have the same B/H stride layout as a slice of `lses`.
    o_sB, o_sH, o_sD = out.stride()
    l_sN, l_sB, l_sH = lses.stride()

    # Allocate LSE with the same B/H strides as `lses` so writes land correctly
    # even when `lses` is a non-contiguous view (e.g., 4-D to 3-D squeeze).
    lse = torch.empty_strided(
        (B, H), (l_sB, l_sH), device=lses.device, dtype=lses.dtype
    )

    # Kernel launch config
    grid = (B, H, 1)

    regular_args = (
        out,
        out,
        lses,
        lse,
        o_sB,
        o_sH,
        o_sD,
        l_sN,
        l_sB,
        l_sH,
        cp_rank,
    )
    const_args = {"HEAD_DIM": D, "N_ROUNDED": N, "IS_BASE_E": is_lse_base_on_e}
    ctx.call_kernel(_correct_attn_cp_out_kernel, grid, *regular_args, **const_args)
    return out, lse


def _cp_lse_common(
    cp_attn_out: torch.Tensor,
    cp_attn_lse: torch.Tensor,
    cp_group: GroupCoordinator,
    ctx: CPTritonContext | None = None,
    is_lse_base_on_e=True,
):
    """
    cp_attn_out: [ B, H, D ]
    cp_attn_lse: [ B, H ]
    """
    # 中文注释：CP 注意力输出校正的公共逻辑。
    # 步骤：
    #   1. 若 world_size == 1（单 rank），无需校正，直接返回。
    #   2. 通过 all-gather 收集所有 rank 的 LSE，得到 [N, B, H]。
    #   3. 调用 correct_attn_out 用全局 LSE 校正本地输出。
    # 返回校正后的输出和全局 LSE。
    if cp_group.world_size == 1:
        return cp_attn_out

    if ctx is None:
        ctx = CPTritonContext()

    # 中文注释：all-gather 收集所有 rank 的 LSE，shape 从 [B, H] 变为 [N, B, H]
    cp_attn_lse = cp_attn_lse.contiguous()
    lses = cp_group.all_gather(cp_attn_lse, dim=0).reshape(
        (cp_group.world_size,) + cp_attn_lse.shape
    )
    # 中文注释：用全局 LSE 校正本地注意力输出
    out, lse = correct_attn_out(
        cp_attn_out,
        lses,
        cp_group.rank_in_group,
        ctx,
        is_lse_base_on_e=is_lse_base_on_e,
    )
    return out, lse


def cp_lse_ag_out_rs(
    cp_attn_out: torch.Tensor,
    cp_attn_lse: torch.Tensor,
    cp_group: GroupCoordinator,
    ctx: CPTritonContext | None = None,
    return_lse: bool = False,
    is_lse_base_on_e=True,
):
    """
    cp_attn_out: [ B, H, D ]
    cp_attn_lse: [ B, H ]
    """
    # 中文注释：CP 注意力输出校正 + reduce-scatter 合并。
    # 流程：
    #   1. all-gather LSE + 校正本地输出
    #   2. reduce-scatter 在 head 维度上合并各 rank 的输出
    #      适用于 head 并行 (TP) 与 CP 共存的场景，每个 rank 最终只保留自己负责的 head。
    out, lse = _cp_lse_common(
        cp_attn_out, cp_attn_lse, cp_group, ctx=ctx, is_lse_base_on_e=is_lse_base_on_e
    )
    # 中文注释：reduce-scatter 在 dim=1 (head) 维度上合并，每个 rank 得到部分 head 的结果
    out = cp_group.reduce_scatter(out, dim=1)

    if return_lse:
        # 中文注释：若需要返回 LSE，从全局 LSE 中提取当前 rank 负责的 head 片段
        cp_num_heads = lse.shape[1] // cp_group.world_size
        cp_rank = cp_group.rank_in_group
        lse = lse[:, cp_num_heads * cp_rank : cp_num_heads * (cp_rank + 1)]
        return out, lse
    return out


def cp_lse_ag_out_ar(
    cp_attn_out: torch.Tensor,
    cp_attn_lse: torch.Tensor,
    cp_group: GroupCoordinator,
    ctx: CPTritonContext | None = None,
    return_lse: bool = False,
    is_lse_base_on_e=True,
):
    """
    cp_attn_out: [ B, H, D ]
    cp_attn_lse: [ B, H ]
    """
    # 中文注释：CP 注意力输出校正 + all-reduce 合并。
    # 与 cp_lse_ag_out_rs 的区别在于合并方式：
    #   - all-reduce：每个 rank 得到完整的全 head 结果（所有 rank 的输出相加后广播）
    #   - reduce-scatter：每个 rank 只得到部分 head 的结果
    # 适用场景：CP 组内没有 TP 并行，所有 rank 都需要完整的 head 输出。
    out, lse = _cp_lse_common(
        cp_attn_out, cp_attn_lse, cp_group, ctx=ctx, is_lse_base_on_e=is_lse_base_on_e
    )
    # 中文注释：all-reduce 在所有 rank 上合并校正后的输出
    out = cp_group.all_reduce(out)

    if return_lse:
        return out, lse
    return out


@triton.jit
def _pack_seq_kernel(
    x_ptr,  # [N, D]
    out_ptr,  # [B, Lmax, D]
    lengths_ptr,  # *i32, [B]
    N: tl.constexpr,
    D: tl.constexpr,
    Lmax: tl.constexpr,
    PAD_VALUE: tl.constexpr,
    PAD_IS_UINT8: tl.constexpr,
    BLOCK_T: tl.constexpr,  # timesteps per program
    BLOCK_D: tl.constexpr,  # features per program
):
    pid_b = tl.program_id(0)  # batch id
    pid_t = tl.program_id(1)  # block over time dimension
    pid_d = tl.program_id(2)  # block over feature dimension
    off_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)  # [BLOCK_T]
    off_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)  # [BLOCK_D]

    # Compute start index and sequence length from cumulative lengths
    in_start = 0
    for i in range(pid_b):
        in_start += tl.load(lengths_ptr + i)
    seq_len = tl.load(lengths_ptr + pid_b)

    # valid time positions for this block
    t_mask = off_t < Lmax

    # compute input row indices for valid (b, t)
    in_row = in_start + off_t
    valid_row = (off_t < seq_len) & t_mask

    # Pointers
    # x_ptr: row-major [N, D]
    x_row_ptr = x_ptr + in_row[:, None] * D + off_d[None, :]

    # out_ptr: row-major [B, Lmax, D]
    out_row_ptr = out_ptr + (pid_b * Lmax + off_t)[:, None] * D + off_d[None, :]

    # Initialize with PAD. PAD_IS_UINT8 selects the pad tensor's dtype so
    # integer-typed outputs (e.g. MXFP4 packed nibbles, ue8m0 scale bytes)
    # get an exact-byte pad rather than going through an fp32→uint8 cast
    # that's implementation-defined outside of value 0.
    d_mask = off_d[None, :] < D
    if PAD_IS_UINT8:
        pad_vals = tl.full([BLOCK_T, BLOCK_D], PAD_VALUE, tl.uint8)
    else:
        pad_vals = tl.full([BLOCK_T, BLOCK_D], PAD_VALUE, tl.float32)
    tl.store(out_row_ptr, pad_vals, mask=t_mask[:, None] & d_mask)

    # Load & write only where within seq_len
    x_vals = tl.load(x_row_ptr, mask=valid_row[:, None] & d_mask)
    tl.store(out_row_ptr, x_vals, mask=valid_row[:, None] & d_mask)


def pack_seq_triton(
    x: torch.Tensor,
    lengths: torch.Tensor,
    pad_value: float | int = -float("inf"),
    block_t: int = 64,
    block_d: int = 64,
) -> torch.Tensor:
    """Pack sequences of different lengths into a batched tensor.

    Supports float dtypes (any, via fp32 pad) and ``torch.uint8`` (exact-byte
    pad — e.g. MXFP4 packed nibbles or ue8m0 scale bytes). For uint8 inputs
    ``pad_value`` must be an integer in ``[0, 255]``.

    Args:
        x: [N, ...] — input tensor where N is total number of tokens.
        lengths: [B] — sequence lengths for each batch.
        pad_value: value to use for padding. Defaults to ``-inf`` which is
            only sensible for float dtypes; pass ``0`` (or any byte) for
            uint8 inputs.
        block_t: block size for time dimension.
        block_d: block size for feature dimension.

    Returns:
        packed: [B, Lmax, ...] — packed tensor.
    """
    # 中文注释：将变长序列打包为 padded batch 格式。
    # 【功能说明】
    #   输入：x 为所有 token 拼接的一维序列 [N, D]，lengths 为每个 batch 的长度 [B]。
    #   输出：packed 为 [B, Lmax, D] 的 padded 张量，Lmax = max(lengths)。
    #   短于 Lmax 的序列用 pad_value 填充。
    # 【使用场景】
    #   在 vLLM 的注意力计算中，query 序列通常是 ragged 格式（变长拼接），
    #   但某些 attention backend（如 FlashAttention 的 varlen 反向操作）需要
    #   padded batch 格式。此函数高效完成转换。
    # 【实现】
    #   使用 Triton 内核并行处理，grid 为 (B, cdiv(Lmax/block_t), cdiv(D/block_d))，
    #   每个 program 处理一个 batch、一个时间块、一个特征块。
    is_uint8 = x.dtype == torch.uint8
    if is_uint8:
        assert isinstance(pad_value, int) and 0 <= pad_value <= 255, (
            f"uint8 pack requires an integer pad in [0, 255], got {pad_value!r}"
        )
        pad_constexpr: int | float = int(pad_value)
    else:
        pad_constexpr = float(pad_value)

    # Handle multi-dimensional input by reshaping to (N, -1)
    original_shape = x.shape
    if len(original_shape) > 2:
        N = original_shape[0]
        x_reshaped = x.reshape(N, -1)
        D = x_reshaped.shape[1]
    else:
        N, D = x.shape
        x_reshaped = x

    B = lengths.numel()
    Lmax = int(lengths.max().item())

    out = torch.empty((B, Lmax, D), device=x.device, dtype=x.dtype)

    grid = (B, triton.cdiv(Lmax, block_t), triton.cdiv(D, block_d))
    _pack_seq_kernel[grid](
        x_reshaped,
        out,
        lengths.int(),
        N,
        D,
        Lmax,
        PAD_VALUE=pad_constexpr,
        PAD_IS_UINT8=is_uint8,
        BLOCK_T=block_t,
        BLOCK_D=block_d,
        num_warps=4,
        num_stages=2,
    )

    if len(original_shape) > 2:
        out = out.reshape((B, Lmax) + original_shape[1:])

    return out


@triton.jit
def _unpack_seq_triton_kernel(
    packed_ptr,  # [B, Lmax, D]
    out_ptr,  # [N, D]
    lengths_ptr,  # *i32, [B]
    B: tl.constexpr,
    Lmax: tl.constexpr,
    D: tl.constexpr,
    BLOCK_T: tl.constexpr,  # timesteps per program
    BLOCK_D: tl.constexpr,  # features per program
):
    pid_b = tl.program_id(0)  # batch id
    pid_t = tl.program_id(1)  # block over time dimension
    pid_d = tl.program_id(2)  # block over feature dimension
    off_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)  # [BLOCK_T]
    off_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)  # [BLOCK_D]

    # bounds: compute start from cumulative lengths
    in_start = 0
    for i in range(pid_b):
        in_start += tl.load(lengths_ptr + i)
    seq_len = tl.load(lengths_ptr + pid_b)

    # valid time positions for this block
    t_mask = off_t < Lmax
    valid_row = (off_t < seq_len) & t_mask

    # compute output row indices for valid (b, t)
    out_row = in_start + off_t

    # Pointers
    # packed_ptr: row-major [B, Lmax, D]
    packed_row_ptr = packed_ptr + (pid_b * Lmax + off_t)[:, None] * D + off_d[None, :]

    # out_ptr: row-major [N, D]
    out_row_ptr = out_ptr + out_row[:, None] * D + off_d[None, :]

    # Load from packed tensor and store to output
    d_mask = off_d[None, :] < D
    packed_vals = tl.load(packed_row_ptr, mask=valid_row[:, None] & d_mask)
    tl.store(out_row_ptr, packed_vals, mask=valid_row[:, None] & d_mask)


def unpack_seq_triton(
    packed_tensor: torch.Tensor,
    lengths: torch.Tensor,
    block_t: int = 64,
    block_d: int = 64,
) -> torch.Tensor:
    """
    Unpack a packed decode query tensor back to the original format.
    Efficient Triton implementation.

    Args:
        packed_tensor: [B, Lmax, ...] - packed tensor from pack_seq_triton
        lengths: [B] - sequence lengths for each batch
        block_t: block size for time dimension
        block_d: block size for feature dimension

    Returns:
        unpacked_tensor: [N, ...] where N = sum(lengths)
    """
    # 中文注释：将 padded batch 格式解包回 ragged 格式。
    # 【功能说明】
    #   输入：packed_tensor 为 [B, Lmax, D] 的 padded 张量。
    #   输出：unpacked_tensor 为 [N, D] 的 ragged 张量，N = sum(lengths)。
    #   只提取每个序列的有效部分（长度为 lengths[b]），丢弃 padding。
    # 【使用场景】
    #   pack_seq_triton 的逆操作。当 attention 计算完成后，需要将 padded 格式
    #   的输出恢复为 ragged 格式，以便后续的 token 采样和输出处理。
    # 【实现】
    #   同样使用 Triton 内核并行处理，每个 program 只写入有效位置的数据。

    # Handle multi-dimensional input by reshaping to (B, Lmax, -1)
    original_shape = packed_tensor.shape
    if len(original_shape) > 3:
        B, Lmax = original_shape[:2]
        packed_reshaped = packed_tensor.reshape(B, Lmax, -1)
        D = packed_reshaped.shape[2]
    else:
        B, Lmax, D = packed_tensor.shape
        packed_reshaped = packed_tensor

    # Calculate total number of elements
    N = int(lengths.sum().item())

    out = torch.empty((N, D), device=packed_tensor.device, dtype=packed_tensor.dtype)

    grid = (B, triton.cdiv(Lmax, block_t), triton.cdiv(D, block_d))
    _unpack_seq_triton_kernel[grid](
        packed_reshaped,
        out,
        lengths.int(),
        B,
        Lmax,
        D,
        BLOCK_T=block_t,
        BLOCK_D=block_d,
        num_warps=4,
        num_stages=2,
    )

    # Reshape output back to original dimensions (except first dimension)
    if len(original_shape) > 3:
        output_shape = (N,) + original_shape[2:]
        out = out.reshape(output_shape)

    return out
