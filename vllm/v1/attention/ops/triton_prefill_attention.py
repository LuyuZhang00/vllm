# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from
# https://github.com/sgl-project/sglang/blob/97cb762bb65ebf05025eb342de03c184660427a3/python/sglang/srt/layers/attention/triton_ops/prefill_attention.py
# Changes:
# - Add support for sliding window attention

# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Memory-efficient attention for prefill.
It supports page size = 1.
"""

# Adapted from
# https://github.com/ModelTC/lightllm/blob/f2a54f0912293f683bf1d1695fd12c4098a5bf82/lightllm/models/llama/triton_kernel/context_flashattention_nopad.py#L1
#
# 本模块实现了基于 Triton 的预填充阶段（prefill）注意力计算。
# 支持 page_size = 1 的场景。
#
# 核心特点：
# 1. 使用在线 softmax 算法，单次遍历计算完整注意力
# 2. 支持因果掩码（IS_CAUSAL）
# 3. 支持双向滑动窗口注意力（SLIDING_WINDOW_Q 和 SLIDING_WINDOW_K）
# 4. 支持 GQA/MQA（通过 kv_group_num 参数）
# 5. 使用 exp2 代替 exp 进行 softmax 计算（更高效）
#
# 与解码注意力的区别：
# - 预填充阶段 Q 的长度 > 1，需要处理块内的因果掩码
# - 使用 3D 网格：(batch, head, q_blocks)
# - 每个程序处理一个 Q 块（BLOCK_M 个 token）

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import RCP_LN2


@triton.jit
def _fwd_kernel(
    Q,                  # 查询张量 [b*s, num_heads, head_dim]
    K,                  # 键张量 [b*s, num_kv_heads, head_dim]
    V,                  # 值张量 [b*s, num_kv_heads, head_dim]
    sm_scale,           # softmax 缩放因子（已乘以 1/ln(2) 用于 exp2）
    B_Start_Loc,        # 每个序列在扁平化 Q/K/V 中的起始位置 [batch]
    B_Seqlen,           # 每个序列的长度 [batch]
    Out,                # 输出张量 [b*s, num_heads, head_dim]
    stride_qbs,         # Q 的序列维度步长
    stride_qh,          # Q 的头维度步长
    stride_kbs,         # K 的序列维度步长
    stride_kh,          # K 的头维度步长
    stride_vbs,         # V 的序列维度步长
    stride_vh,          # V 的头维度步长
    stride_obs,         # 输出的序列维度步长
    stride_oh,          # 输出的头维度步长
    kv_group_num: tl.constexpr,  # GQA 分组数
    BLOCK_M: tl.constexpr,       # Q 块大小
    BLOCK_DMODEL: tl.constexpr,  # 头维度分块大小
    BLOCK_N: tl.constexpr,       # KV 块大小
    IS_CAUSAL: tl.constexpr,     # 是否使用因果掩码
    SLIDING_WINDOW_Q: tl.constexpr,  # Q 方向滑动窗口大小
    SLIDING_WINDOW_K: tl.constexpr,  # K 方向滑动窗口大小
    Lk: tl.constexpr,            # 实际头维度
):
    """
    预填充注意力 kernel。

    网格维度：(batch, head, q_blocks)
    每个程序处理一个 (序列, 头, Q 块) 三元组。

    工作流程：
    1. 加载当前 Q 块
    2. 遍历所有 KV 块（每块 BLOCK_N 个 token）
    3. 对每个 KV 块：
       a. 计算 QK 点积
       b. 应用因果掩码、滑动窗口掩码
       c. 使用在线 softmax（exp2 版本）更新累加器
    4. 归一化并存储输出
    """
    # 程序 ID 三维网格：(batch, head, q_block)
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    start_m = tl.program_id(2)

    # GQA 分组：确定当前 Q 头对应的 KV 头
    cur_kv_head = cur_head // kv_group_num

    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)

    block_start_loc = BLOCK_M * start_m

    # 初始化偏移量
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_q = (
        (cur_batch_in_all_start_index + offs_m[:, None]) * stride_qbs
        + cur_head * stride_qh
        + offs_d[None, :]
    )
    off_k = offs_n[None, :] * stride_kbs + cur_kv_head * stride_kh + offs_d[:, None]
    off_v = offs_n[:, None] * stride_vbs + cur_kv_head * stride_vh + offs_d[None, :]

    mask_d = offs_d < Lk

    q = tl.load(
        Q + off_q,
        mask=(offs_m[:, None] < cur_batch_seq_len) & (mask_d[None, :]),
        other=0.0,
    )

    k_ptrs = K + off_k
    v_ptrs = V + off_v

    # 初始化在线 softmax 状态
    # m_i: 行最大值，l_i: 指数和，acc: V 的加权累加器
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # 块掩码：过滤超出序列长度的块
    block_mask = tl.where(block_start_loc < cur_batch_seq_len, 1, 0)

    # 计算注意力计算的结束位置
    end_n = cur_batch_seq_len

    # 因果注意力裁剪：只关注当前 Q 块之前的位置
    end_n = tl.minimum(end_n, (start_m + 1) * BLOCK_M) if IS_CAUSAL else end_n

    # 计算滑动窗口的起始位置
    start_n_limit = 0
    end_n_limit = block_mask * end_n

    # 遍历所有 KV 块
    for start_n in range(start_n_limit, end_n_limit, BLOCK_N):
        # -- 准备注意力掩码 ----
        # 序列中的位置索引
        pos_q = offs_m[:, None]  # 查询位置 [BLOCK_M, 1]
        pos_k = start_n + offs_n[None, :]  # 键位置 [1, BLOCK_N]

        # 有效序列掩码：键位置不超过序列长度
        mask = pos_k < cur_batch_seq_len
        # 因果掩码：查询只能关注之前的位置
        if IS_CAUSAL:
            mask &= pos_q >= pos_k

        # 双向滑动窗口掩码
        sliding_mask_q = (
            pos_q - pos_k <= SLIDING_WINDOW_Q if SLIDING_WINDOW_Q > 0 else None
        )
        sliding_mask_k = (
            pos_k - pos_q <= SLIDING_WINDOW_K if SLIDING_WINDOW_K > 0 else None
        )
        if sliding_mask_q is not None:
            mask &= sliding_mask_q
        if sliding_mask_k is not None:
            mask &= sliding_mask_k

        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k = tl.load(
            k_ptrs + (cur_batch_in_all_start_index + start_n) * stride_kbs,
            mask=(pos_k < cur_batch_seq_len) & (mask_d[:, None]),
            other=0.0,
        )

        # 计算 QK 点积并应用掩码
        qk = tl.dot(q, k)
        qk = tl.where(mask, qk * sm_scale, -1.0e8)
        # 在线 softmax 更新（使用 exp2 版本，更高效）
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk -= m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)

        # 更新 m_i 和 l_i
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        # 更新输出累加器
        acc = acc * alpha[:, None]
        # 加载 V 并更新累加器
        v = tl.load(
            v_ptrs + (cur_batch_in_all_start_index + start_n) * stride_vbs,
            mask=((start_n + offs_n[:, None]) < cur_batch_seq_len) & (mask_d[None, :]),
            other=0.0,
        )
        p = p.to(v.dtype)
        acc = tl.dot(p, v, acc)
        # 更新 m_i
        m_i = m_ij

    # 尾声：归一化输出
    acc = acc / l_i[:, None]
    off_o = (
        (cur_batch_in_all_start_index + offs_m[:, None]) * stride_obs
        + cur_head * stride_oh
        + offs_d[None, :]
    )
    out_ptrs = Out + off_o
    tl.store(
        out_ptrs, acc, mask=(offs_m[:, None] < cur_batch_seq_len) & (mask_d[None, :])
    )


def get_block_size(dtype: torch.dtype) -> int:
    """根据数据类型和硬件能力选择 Q/K 块大小。

    fp32：使用 32（精度要求更高）
    CUDA compute capability >= 80（A100/H100）：使用 128
    其他：使用 64
    """
    if dtype == torch.float32:
        return 32
    elif current_platform.is_cuda_alike() and current_platform.has_device_capability(
        80
    ):
        return 128
    else:
        return 64


def context_attention_fwd(
    # 预填充注意力的 Python 封装函数
    # 设置网格、选择块大小、启动 _fwd_kernel
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    b_start_loc: torch.Tensor,
    b_seq_len: torch.Tensor,
    max_input_len: int,
    is_causal: bool = True,
    softmax_scale: float | None = None,
    sliding_window_q: int | None = None,
    sliding_window_k: int | None = None,
):
    """
    q, k, v: [b * s, head, head_dim]
    b_start_loc: [b]
    b_seq_len: [b]
    out: [b * s, head, head_dim]
    """
    BLOCK = get_block_size(q.dtype)

    Lq, Lk, _ = q.shape[-1], k.shape[-1], v.shape[-1]

    sm_scale = 1.0 / (Lq**0.5) if softmax_scale is None else softmax_scale
    # 缩放因子乘以 1/ln(2) 用于 triton exp2（因为 exp2(x) = exp(x * ln(2))）
    sm_scale *= RCP_LN2

    batch, head = b_seq_len.shape[0], q.shape[1]
    kv_group_num = q.shape[1] // k.shape[1]

    grid = (batch, head, triton.cdiv(max_input_len, BLOCK))
    num_warps = 4 if Lk <= 64 else 8

    sliding_window_q = sliding_window_q if sliding_window_q is not None else 0
    sliding_window_k = sliding_window_k if sliding_window_k is not None else 0

    _fwd_kernel[grid](
        q,
        k,
        v,
        sm_scale,
        b_start_loc,
        b_seq_len,
        o,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        o.stride(0),
        o.stride(1),
        kv_group_num=kv_group_num,
        BLOCK_M=BLOCK,
        BLOCK_DMODEL=triton.next_power_of_2(Lk),
        BLOCK_N=BLOCK,
        IS_CAUSAL=is_causal,
        SLIDING_WINDOW_Q=sliding_window_q,
        SLIDING_WINDOW_K=sliding_window_k,
        num_warps=num_warps,
        num_stages=1,
        Lk=Lk,
    )
