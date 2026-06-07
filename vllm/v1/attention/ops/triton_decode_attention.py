# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from
# https://github.com/sgl-project/sglang/blob/9f635ea50de920aa507f486daafba26a5b837574/python/sglang/srt/layers/attention/triton_ops/decode_attention.py
# which was originally adapted from
# https://github.com/ModelTC/lightllm/blob/96353e868a840db4d103138caf15ed9dbea8c186/lightllm/models/deepseek2/triton_kernel/gqa_flash_decoding_stage1.py
# https://github.com/ModelTC/lightllm/blob/96353e868a840db4d103138caf15ed9dbea8c186/lightllm/models/deepseek2/triton_kernel/gqa_flash_decoding_stage2.py

# Changes:
# - Add support for page size >= 1.
#
# 本模块实现了基于 Triton 的解码阶段（decode）注意力计算。
# 核心思想是将 KV 序列沿长度维度分割成多个子段（split-KV），
# 每个子段独立计算部分注意力输出，最后通过第二阶段合并得到最终结果。
#
# 两阶段流程概述：
# 阶段 1（_fwd_kernel_stage1 / _fwd_grouped_kernel_stage1）：
#   - 每个 Triton 程序处理 (batch, head, split_kv_id) 三元组
#   - 加载 Q 向量，遍历该 split 对应的 KV 区间
#   - 计算 QK 注意力分数（支持 FP8 反量化、logit cap）
#   - 使用在线 softmax 算法维护 e_max 和 e_sum，累加加权 V 输出
#   - 将每个 split 的归一化输出和 log-sum-exp（LSE）存入中间缓冲区
# 阶段 2（_fwd_kernel_stage2）：
#   - 每个 Triton 程序处理 (batch, head)
#   - 遍历所有 split 的中间结果，用 LSE 加权合并得到最终输出
#   - 同时输出全局 LSE 值，供后续 merge 操作使用
#
# 支持的注意力模式：
# - MHA（Multi-Head Attention）：kv_group_num == 1，使用 _fwd_kernel_stage1
# - GQA/MQA（Grouped/ Multi-Query Attention）：kv_group_num > 1，
#   使用 _fwd_grouped_kernel_stage1，每个程序处理一组 Q 头
# - MLA（Multi-Head Latent Attention）：is_mla=True，K 和 V 共享同一压缩表示
#
# 分页支持：通过 Req_to_tokens 映射表将逻辑位置转换为物理 KV 缓存位置，
# 支持 page_size >= 1 的分页 KV 缓存。

# Copyright 2025 vLLM Team
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
Memory-efficient attention for decoding.
It supports page size >= 1.
"""

import logging

import torch
from packaging import version

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

# 检测是否在 AMD ROCm（HIP）平台上运行，用于后续针对 AMD GPU 的参数调优
is_hip_ = current_platform.is_rocm()

logger = logging.getLogger(__name__)

# Only print the following warnings when triton version < 3.2.0.
# The issue won't affect performance or accuracy.
if version.parse(triton.__version__) < version.parse("3.2.0"):
    logger.warning(
        "The following error message 'operation scheduled before its operands' "
        "can be ignored."
    )


@triton.jit
def tanh(x):
    # Tanh is just a scaled sigmoid
    # 使用 sigmoid 函数实现 tanh：tanh(x) = 2 * sigmoid(2x) - 1
    # 避免直接调用 tanh，利用 Triton 原生 sigmoid 的更高性能
    return 2 * tl.sigmoid(2 * x) - 1


@triton.jit
def _fwd_kernel_stage1(
    Q,                  # 查询张量 [batch, num_heads, head_dim]
    K_Buffer,           # 键缓存 [num_pages, page_size, num_kv_heads, head_dim]
    V_Buffer,           # 值缓存 [num_pages, page_size, num_kv_heads, head_dim_v]
    sm_scale,           # softmax 缩放因子，通常为 1/sqrt(head_dim)
    Req_to_tokens,      # 请求到 token 位置的映射表 [batch, max_seq_len // page_size]
    B_Seqlen,           # 每个请求的序列长度 [batch]
    Att_Out,            # 中间输出缓冲区 [batch, num_heads, num_kv_splits, head_dim_v + 1]
    stride_req_to_tokens_b,  # Req_to_tokens 的 batch 维度步长
    stride_qbs,        # Q 的 batch 维度步长
    stride_qh,         # Q 的 head 维度步长
    stride_buf_kbs,    # K 缓存的 batch/page 维度步长
    stride_buf_kh,     # K 缓存的 head 维度步长
    stride_buf_vbs,    # V 缓存的 batch/page 维度步长
    stride_buf_vh,     # V 缓存的 head 维度步长
    stride_mid_ob,     # 中间输出的 batch 维度步长
    stride_mid_oh,     # 中间输出的 head 维度步长
    stride_mid_os,     # 中间输出的 split 维度步长
    k_scale,           # K 的 FP8 反量化缩放因子
    v_scale,           # V 的 FP8 反量化缩放因子
    kv_group_num: tl.constexpr,  # 每个 KV 头对应的 Q 头数量（GQA 分组数）
    BLOCK_DMODEL: tl.constexpr,  # Q/K 头维度的分块大小（2 的幂次）
    BLOCK_DV: tl.constexpr,     # V 头维度的分块大小（2 的幂次）
    BLOCK_N: tl.constexpr,      # KV 序列维度的分块大小
    NUM_KV_SPLITS: tl.constexpr, # KV 序列分割数
    PAGE_SIZE: tl.constexpr,     # 分页大小
    logit_cap: tl.constexpr,     # logit 上限值（>0 时启用 tanh cap）
    Lk: tl.constexpr,           # K 的实际头维度
    Lv: tl.constexpr,           # V 的实际头维度
):
    """
    阶段 1 kernel（MHA 版本）：每个 Triton 程序处理一个 (batch, head, split) 三元组。
    计算该 split 区间内的局部注意力输出和 LSE 值。

    工作流程：
    1. 确定当前程序负责的 batch、head 和 split ID
    2. 加载 Q 向量
    3. 计算当前 split 负责的 KV 区间 [split_kv_start, split_kv_end)
    4. 循环遍历该区间的 KV 块（每块 BLOCK_N 个 token）：
       a. 通过 Req_to_tokens 将逻辑位置映射到物理缓存位置
       b. 加载 K 和 V（支持 FP8 反量化）
       c. 计算 QK 点积并缩放
       d. 应用 logit cap（可选）
       e. 使用在线 softmax 算法更新 e_max、e_sum 和累加器 acc
    5. 归一化输出（acc / e_sum）并存入中间缓冲区
    6. 将 LSE = e_max + log(e_sum) 存入中间缓冲区的额外位置
    """
    # 程序 ID 三维网格：(batch, head, split_kv_id)
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    split_kv_id = tl.program_id(2)

    # 通过 GQA 分组数确定当前 Q 头对应的 KV 头索引
    cur_kv_head = cur_head // kv_group_num

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lk
    mask_dv = offs_dv < Lv
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_req_idx = cur_batch

    off_q = cur_batch * stride_qbs + cur_head * stride_qh + offs_d
    q = tl.load(Q + off_q, mask=mask_d, other=0.0)

    # 计算当前 split 负责的 KV 序列区间
    # 将总序列长度均匀分配到 NUM_KV_SPLITS 个 split 中
    kv_len_per_split = tl.cdiv(cur_batch_seq_len, NUM_KV_SPLITS)
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

    # 在线 softmax 的状态变量：
    # e_max: 当前看到的最大 logit 值（用于数值稳定性）
    # e_sum: 指数和（softmax 分母的未归一化版本）
    # acc: V 的加权累加器
    e_max = -float("inf")
    e_sum = 0.0
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)

    if split_kv_end > split_kv_start:
        # 加载 FP8 反量化缩放因子
        ks = tl.load(k_scale)
        vs = tl.load(v_scale)
        # 遍历当前 split 负责的 KV 区间，每次处理 BLOCK_N 个 token
        for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            # 1) 通过 Req_to_tokens 映射表将逻辑 token 位置转换为物理页号
            # offs_n // PAGE_SIZE 得到页内偏移对应的页索引
            kv_page_number = tl.load(
                Req_to_tokens
                + stride_req_to_tokens_b * cur_batch_req_idx
                + offs_n // PAGE_SIZE,
                mask=offs_n < split_kv_end,
                other=0,
            )
            # 2) 计算物理 KV 缓存中的绝对位置：页号 * 页大小 + 页内偏移
            kv_loc = kv_page_number * PAGE_SIZE + offs_n % PAGE_SIZE
            offs_buf_k = (
                kv_loc[:, None] * stride_buf_kbs
                + cur_kv_head * stride_buf_kh
                + offs_d[None, :]
            )
            k = tl.load(
                K_Buffer + offs_buf_k,
                mask=(offs_n[:, None] < split_kv_end) & (mask_d[None, :]),
                other=0.0,
            )
            if k.dtype.is_fp8():
                k = (k.to(tl.float32) * ks).to(q.dtype)
            # 3) 计算 QK 注意力分数：Q 和 K 的点积，乘以缩放因子
            qk = tl.sum(q[None, :] * k, 1)
            qk *= sm_scale

            # 4) 可选的 logit cap：使用 tanh 将注意力分数限制在 [-logit_cap, logit_cap]
            if logit_cap > 0:
                qk = logit_cap * tanh(qk / logit_cap)

            # 5) 掩码：超出 split 边界的位置设为 -inf
            qk = tl.where(offs_n < split_kv_end, qk, float("-inf"))

            offs_buf_v = (
                kv_loc[:, None] * stride_buf_vbs
                + cur_kv_head * stride_buf_vh
                + offs_dv[None, :]
            )
            v = tl.load(
                V_Buffer + offs_buf_v,
                mask=(offs_n[:, None] < split_kv_end) & (mask_dv[None, :]),
                other=0.0,
            )
            if v.dtype.is_fp8():
                v = (v.to(tl.float32) * vs).to(q.dtype)

            # 6) 在线 softmax 更新：
            #   - n_e_max: 新的最大值（取旧 e_max 和当前块最大值的较大者）
            #   - re_scale: 旧累加器需要乘以的缩放因子 exp(e_max - n_e_max)
            #   - p: 当前块的 softmax 权重 exp(qk - n_e_max)
            #   - acc: 更新加权 V 累加器（先缩放旧值，再加新贡献）
            #   - e_sum: 更新指数和
            n_e_max = tl.maximum(tl.max(qk, 0), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max)
            acc *= re_scale
            acc += tl.sum(p[:, None] * v, 0)

            e_sum = e_sum * re_scale + tl.sum(p, 0)
            e_max = n_e_max

        offs_mid_o = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
            + offs_dv
        )

        # 存储当前 split 的归一化输出：acc / e_sum
        tl.store(
            Att_Out + offs_mid_o,
            acc / e_sum,
            mask=(mask_dv),
        )

        # 存储 LSE（log-sum-exp）值到输出的额外位置（紧跟在 head_dim_v 之后）
        # LSE = e_max + log(e_sum)，供阶段 2 合并使用
        offs_mid_o_1 = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
            + Lv
        )

        tl.store(
            Att_Out + offs_mid_o_1,
            e_max + tl.log(e_sum),
        )


def _decode_att_m_fwd(
    # MHA（多头注意力）阶段 1 的 Python 封装函数
    # 负责启动 _fwd_kernel_stage1 Triton kernel
    q,
    k_buffer,
    v_buffer,
    att_out,
    Req_to_tokens,
    B_Seqlen,
    num_kv_splits,
    sm_scale,
    page_size,
    logit_cap,
    k_scale,
    v_scale,
):
    # BLOCK_N: 每次循环处理的 KV token 数量
    # AMD GPU 使用较小的块以适配其硬件特性
    BLOCK = 64 if not is_hip_ else 8

    NUM_KV_SPLITS = num_kv_splits
    Lk = k_buffer.shape[-1]
    Lv = v_buffer.shape[-1]

    batch, head_num = q.shape[0], q.shape[1]

    grid = (batch, head_num, NUM_KV_SPLITS)
    kv_group_num = q.shape[1] // k_buffer.shape[-2]

    num_warps = 4
    if kv_group_num != 1:
        num_warps = 1 if is_hip_ else 2

    BLOCK_DMODEL = triton.next_power_of_2(Lk)
    BLOCK_DV = triton.next_power_of_2(Lv)

    _fwd_kernel_stage1[grid](
        q,
        k_buffer,
        v_buffer,
        sm_scale,
        Req_to_tokens,
        B_Seqlen,
        att_out,
        Req_to_tokens.stride(0),
        q.stride(0),
        q.stride(1),
        k_buffer.stride(-3),  # Assume (..., PAGE_SIZE, NUM_HEADS, HEAD_DIM)
        k_buffer.stride(-2),  # Assume (..., PAGE_SIZE, NUM_HEADS, HEAD_DIM)
        v_buffer.stride(-3),  # Assume (..., PAGE_SIZE, NUM_HEADS, HEAD_DIM)
        v_buffer.stride(-2),  # Assume (..., PAGE_SIZE, NUM_HEADS, HEAD_DIM)
        att_out.stride(0),
        att_out.stride(1),
        att_out.stride(2),
        k_scale,
        v_scale,
        kv_group_num=kv_group_num,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DV=BLOCK_DV,
        BLOCK_N=BLOCK,
        NUM_KV_SPLITS=NUM_KV_SPLITS,
        PAGE_SIZE=page_size,
        logit_cap=logit_cap,
        num_warps=num_warps,
        num_stages=2,
        Lk=Lk,
        Lv=Lv,
    )


@triton.jit
def _fwd_grouped_kernel_stage1(
    # 阶段 1 kernel（GQA/MQA/MLA 版本）：每个 Triton 程序处理一组 Q 头（BLOCK_H 个）
    # 与 _fwd_kernel_stage1 的区别：
    # 1. 使用 tl.dot 进行矩阵乘法而非逐元素点积，支持多头并行
    # 2. 支持 BLOCK_DPE 用于 MLA 模式的额外位置编码维度
    # 3. MLA 模式下 K 和 V 共享同一压缩表示，V = transpose(K)
    Q,
    K_Buffer,
    V_Buffer,
    sm_scale,
    Req_to_tokens,
    B_Seqlen,
    Att_Out,
    stride_req_to_tokens_b,
    stride_qbs,
    stride_qh,
    stride_buf_kbs,
    stride_buf_kh,
    stride_buf_vbs,
    stride_buf_vh,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    k_scale,
    v_scale,
    kv_group_num: tl.constexpr,
    q_head_num: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DPE: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    logit_cap: tl.constexpr,
    Lk: tl.constexpr,
    Lv: tl.constexpr,
    IS_MLA: tl.constexpr = False,
):
    # 程序 ID 三维网格：(batch, head_group_id, split_kv_id)
    cur_batch = tl.program_id(0)
    cur_head_id = tl.program_id(1)
    # 根据 head_group_id 和 GQA 分组数计算对应的 KV 头
    cur_kv_head = cur_head_id // tl.cdiv(kv_group_num, BLOCK_H)
    split_kv_id = tl.program_id(2)

    # 计算当前程序实际处理的 Q 头范围（可能不足 BLOCK_H 个）
    VALID_BLOCK_H: tl.constexpr = BLOCK_H if kv_group_num > BLOCK_H else kv_group_num
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    # 构建头掩码：过滤超出实际头数量的部分
    mask_h = cur_head < (cur_head_id + 1) * VALID_BLOCK_H
    mask_h = mask_h & (cur_head < q_head_num)

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lk
    mask_dv = offs_dv < Lv
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_req_idx = cur_batch

    offs_q = cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_d[None, :]
    q = tl.load(
        Q + offs_q,
        mask=(mask_h[:, None]) & (mask_d[None, :]),
        other=0.0,
        cache_modifier=".ca",
    )

    # MLA 模式：加载额外的位置编码维度（BLOCK_DPE）
    # MLA 的 Q 向量包含两部分：主维度 [0, BLOCK_DMODEL) 和位置编码维度 [BLOCK_DMODEL, Lk)
    if BLOCK_DPE > 0:
        offs_dpe = BLOCK_DMODEL + tl.arange(0, BLOCK_DPE)
        mask_dpe = offs_dpe < Lk
        off_qpe = (
            cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_dpe[None, :]
        )
        qpe = tl.load(
            Q + off_qpe,
            mask=(mask_h[:, None]) & (mask_dpe[None, :]),
            other=0.0,
            cache_modifier=".ca",
        )

    kv_len_per_split = tl.cdiv(cur_batch_seq_len, NUM_KV_SPLITS)
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

    # 在线 softmax 状态变量（多头版本，每个头独立维护）
    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_DV], dtype=tl.float32)

    if split_kv_end > split_kv_start:
        base_offs_k = cur_kv_head * stride_buf_kh + offs_d[:, None]
        base_offs_v = cur_kv_head * stride_buf_vh + offs_dv[None, :]
        if BLOCK_DPE > 0:
            base_offs_kpe = cur_kv_head * stride_buf_kh + offs_dpe[:, None]

        ks = tl.load(k_scale)
        vs = tl.load(v_scale)
        for start_n in tl.range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            # 逻辑位置到物理页号的映射（同 _fwd_kernel_stage1）
            kv_page_number = tl.load(
                Req_to_tokens
                + stride_req_to_tokens_b * cur_batch_req_idx
                + offs_n // PAGE_SIZE,
                mask=offs_n < split_kv_end,
                other=0,
                cache_modifier=".ca",
            )
            kv_loc = kv_page_number * PAGE_SIZE + offs_n % PAGE_SIZE

            # 使用 ".cg" 缓存修饰符显式促进 load/compute 重叠
            # ".cg" = cache global，不污染 L1 缓存
            offs_buf_k = kv_loc[None, :] * stride_buf_kbs + base_offs_k
            k = tl.load(
                K_Buffer + offs_buf_k,
                mask=(offs_n[None, :] < split_kv_end) & (mask_d[:, None]),
                other=0.0,
                cache_modifier=".cg",
            )

            # FP8 反量化：将 K 从 FP8 转为浮点并乘以缩放因子
            if k.dtype.is_fp8():
                k = (k.to(tl.float32) * ks).to(q.dtype)
            # 使用矩阵乘法计算 QK（多头并行）
            qk = tl.dot(q, k.to(q.dtype))
            # MLA 模式：额外计算位置编码维度的贡献
            if BLOCK_DPE > 0:
                offs_buf_kpe = kv_loc[None, :] * stride_buf_kbs + base_offs_kpe
                kpe = tl.load(
                    K_Buffer + offs_buf_kpe,
                    mask=(offs_n[None, :] < split_kv_end) & (mask_dpe[:, None]),
                    other=0.0,
                    cache_modifier=".cg",
                )
                if kpe.dtype.is_fp8():
                    kpe = (kpe.to(tl.float32) * ks).to(qpe.dtype)
                qk += tl.dot(qpe, kpe.to(qpe.dtype))
            qk *= sm_scale

            if logit_cap > 0:
                qk = logit_cap * tanh(qk / logit_cap)

            qk = tl.where(
                mask_h[:, None] & (offs_n[None, :] < split_kv_end), qk, float("-inf")
            )

            # 加载 V 并计算加权输出
            if not IS_MLA:
                # 标准模式：从 V 缓存加载
                offs_buf_v = kv_loc[:, None] * stride_buf_vbs + base_offs_v
                v = tl.load(
                    V_Buffer + offs_buf_v,
                    mask=(offs_n[:, None] < split_kv_end) & (mask_dv[None, :]),
                    other=0.0,
                )
                if v.dtype.is_fp8():
                    v = (v.to(tl.float32) * vs).to(q.dtype)
            else:
                # MLA 模式：K 和 V 共享同一压缩表示 c_kv
                # 无需额外加载 V，直接转置 K 用于后续的 P*V 矩阵乘法
                v = tl.trans(k)

            n_e_max = tl.maximum(tl.max(qk, 1), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])
            acc *= re_scale[:, None]
            acc += tl.dot(p.to(v.dtype), v)

            e_sum = e_sum * re_scale + tl.sum(p, 1)
            e_max = n_e_max

        offs_mid_o = (
            cur_batch * stride_mid_ob
            + cur_head[:, None] * stride_mid_oh
            + split_kv_id * stride_mid_os
            + offs_dv[None, :]
        )

        tl.store(
            Att_Out + offs_mid_o,
            acc / e_sum[:, None],
            mask=(mask_h[:, None]) & (mask_dv[None, :]),
        )

        offs_mid_o_1 = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
            + Lv
        )

        tl.store(
            Att_Out + offs_mid_o_1,
            e_max + tl.log(e_sum),
            mask=mask_h,
        )


def _decode_grouped_att_m_fwd(
    # GQA/MQA/MLA 阶段 1 的 Python 封装函数
    # 负责启动 _fwd_grouped_kernel_stage1 Triton kernel
    # 关键区别于 _decode_att_m_fwd：
    # 1. 支持 MLA 模式，自动计算 BLOCK_DMODEL 和 BLOCK_DPE
    # 2. 使用 BLOCK_H=16 每个程序处理 16 个 Q 头
    # 3. 针对 AMD GPU 有专门的优化参数
    q,
    k_buffer,
    v_buffer,
    att_out,
    Req_to_tokens,
    B_Seqlen,
    num_kv_splits,
    sm_scale,
    page_size,
    logit_cap,
    k_scale,
    v_scale,
    is_mla=False,
):
    # with is_mla there is only a single c_kv in smem.
    # could increase BLOCK or num_stages.
    Lk = k_buffer.shape[-1]
    Lv = v_buffer.shape[-1]

    # MLA 模式下，Q/K 包含主维度和位置编码维度两部分
    # 需要将 BLOCK_DMODEL 对齐到 V 的维度，BLOCK_DPE 对齐到额外的位置编码维度
    # 例如 DeepSeek-V2 的 Lk=576, Lv=512, 则 BLOCK_DMODEL=512, BLOCK_DPE=64
    if is_mla:
        if not is_hip_ and Lk == 576:
            BLOCK_DMODEL = 512
            BLOCK_DPE = 64
        elif not is_hip_ and Lk == 288:
            BLOCK_DMODEL = 256
            BLOCK_DPE = 32
        else:
            BLOCK_DMODEL = triton.next_power_of_2(Lv)
            BLOCK_DPE = triton.next_power_of_2(Lk - Lv) if Lk > Lv else 0
    else:
        BLOCK_DMODEL = triton.next_power_of_2(Lk)
        BLOCK_DPE = 0
    BLOCK_DV = triton.next_power_of_2(Lv)

    BLOCK = 32
    if is_hip_:
        BLOCK = 16

    batch, head_num = q.shape[0], q.shape[1]
    kv_group_num = q.shape[1] // k_buffer.shape[-2]

    BLOCK_H = 16
    NUM_KV_SPLITS = num_kv_splits
    grid = (
        batch,
        triton.cdiv(head_num, min(BLOCK_H, kv_group_num)),
        NUM_KV_SPLITS,
    )

    extra_kargs = {}
    num_stages = 2
    if is_hip_:
        # https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html#mi300x-triton-kernel-performance-optimization
        # https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py
        extra_kargs = {"waves_per_eu": 1, "matrix_instr_nonkdim": 16, "kpack": 2}
        num_stages = 1
    elif not is_hip_ and BLOCK_DMODEL >= 1024:
        # Avoid shared memory overflow on NVIDIA when BLOCK_DMODEL is large
        # like non-MLA D_QK=576, BLOCK_DMODEL=1024, BLOCK_H=16
        # exceeds 101376 bytes limit
        num_stages = 1

    _fwd_grouped_kernel_stage1[grid](
        q,
        k_buffer,
        v_buffer,
        sm_scale,
        Req_to_tokens,
        B_Seqlen,
        att_out,
        Req_to_tokens.stride(0),
        q.stride(0),
        q.stride(1),
        k_buffer.stride(-3),  # Assume (..., PAGE_SIZE, NUM_HEADS, HEAD_DIM)
        k_buffer.stride(-2),  # Assume (..., PAGE_SIZE, NUM_HEADS, HEAD_DIM)
        v_buffer.stride(-3),  # Assume (..., PAGE_SIZE, NUM_HEADS, HEAD_DIM)
        v_buffer.stride(-2),  # Assume (..., PAGE_SIZE, NUM_HEADS, HEAD_DIM)
        att_out.stride(0),
        att_out.stride(1),
        att_out.stride(2),
        k_scale,
        v_scale,
        kv_group_num=kv_group_num,
        q_head_num=head_num,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DPE=BLOCK_DPE,
        BLOCK_DV=BLOCK_DV,
        BLOCK_N=BLOCK,
        BLOCK_H=BLOCK_H,
        NUM_KV_SPLITS=NUM_KV_SPLITS,
        PAGE_SIZE=page_size,
        logit_cap=logit_cap,
        num_warps=4,
        num_stages=num_stages,
        Lk=Lk,
        Lv=Lv,
        IS_MLA=is_mla,
        **extra_kargs,
    )


@triton.jit
def _fwd_kernel_stage2(
    # 阶段 2 kernel：合并所有 split 的中间结果
    # 每个 Triton 程序处理一个 (batch, head) 对
    # 遍历 NUM_KV_SPLITS 个 split 的输出，用 LSE 加权合并
    Mid_O,
    o,
    lse,
    B_Seqlen,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_obs,
    stride_oh,
    stride_lse_bs,
    NUM_KV_SPLITS: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    Lv: tl.constexpr,
    OUTPUT_FP16: tl.constexpr = 0,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)

    offs_d = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lv

    # 合并阶段的在线 softmax 状态
    e_sum = 0.0
    e_max = -float("inf")
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)

    offs_v = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + offs_d
    offs_logic = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + Lv

    # 遍历所有 split，用 LSE 加权合并
    for split_kv_id in range(0, NUM_KV_SPLITS):
        kv_len_per_split = tl.cdiv(cur_batch_seq_len, NUM_KV_SPLITS)
        split_kv_start = kv_len_per_split * split_kv_id
        split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

        if split_kv_end > split_kv_start:
            # 加载该 split 的归一化输出和 LSE 值
            tv = tl.load(
                Mid_O + offs_v + split_kv_id * stride_mid_os, mask=mask_d, other=0.0
            )
            tlogic = tl.load(Mid_O + offs_logic + split_kv_id * stride_mid_os)
            # 在线合并：与阶段 1 相同的数值稳定技巧
            n_e_max = tl.maximum(tlogic, e_max)

            old_scale = tl.exp(e_max - n_e_max)
            acc *= old_scale
            exp_logic = tl.exp(tlogic - n_e_max)
            acc += exp_logic * tv

            e_sum = e_sum * old_scale + exp_logic
            e_max = n_e_max

    # 最终归一化并存储结果
    result = acc / e_sum
    if OUTPUT_FP16:
        result = result.to(tl.float16)
    tl.store(
        o + cur_batch * stride_obs + cur_head * stride_oh + offs_d,
        result,
        mask=mask_d,
    )
    # 存储全局 LSE 值，供后续 merge_attn_states 使用
    lse_val = e_max + tl.log(e_sum)
    tl.store(
        lse + cur_batch * stride_lse_bs + cur_head,
        lse_val,
    )


def _decode_softmax_reducev_fwd(
    # 阶段 2 的 Python 封装函数：启动 _fwd_kernel_stage2
    logits,
    q,
    o,
    lse,
    v_buffer,
    b_seq_len,
    num_kv_splits,
):
    batch, head_num = q.shape[0], q.shape[1]
    Lv = v_buffer.shape[-1]
    BLOCK_DV = triton.next_power_of_2(Lv)

    NUM_KV_SPLITS = num_kv_splits

    extra_kargs = {}
    if is_hip_:
        # https://rocm.docs.amd.com/en/docs-6.2.0/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html
        # https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py
        extra_kargs = {"waves_per_eu": 4, "matrix_instr_nonkdim": 16, "kpack": 2}

    grid = (batch, head_num)
    _fwd_kernel_stage2[grid](
        logits,
        o,
        lse,
        b_seq_len,
        logits.stride(0),
        logits.stride(1),
        logits.stride(2),
        o.stride(0),
        o.stride(1),
        lse.stride(0),
        NUM_KV_SPLITS=NUM_KV_SPLITS,
        BLOCK_DV=BLOCK_DV,
        Lv=Lv,
        num_warps=4,
        num_stages=2,
        **extra_kargs,
    )


def decode_attention_fwd_normal(
    # MHA 模式的解码注意力入口：阶段 1（_decode_att_m_fwd）+ 阶段 2（_decode_softmax_reducev_fwd）
    q,
    k_buffer,
    v_buffer,
    o,
    lse,
    req_to_token,
    b_seq_len,
    attn_logits,
    num_kv_splits,
    sm_scale,
    page_size,
    logit_cap=0.0,
    k_scale=None,
    v_scale=None,
):
    _decode_att_m_fwd(
        q,
        k_buffer,
        v_buffer,
        attn_logits,
        req_to_token,
        b_seq_len,
        num_kv_splits,
        sm_scale,
        page_size,
        logit_cap,
        k_scale,
        v_scale,
    )
    _decode_softmax_reducev_fwd(
        attn_logits, q, o, lse, v_buffer, b_seq_len, num_kv_splits
    )


def decode_attention_fwd_grouped(
    # GQA/MQA/MLA 模式的解码注意力入口：阶段 1（_decode_grouped_att_m_fwd）+ 阶段 2
    q,
    k_buffer,
    v_buffer,
    o,
    lse,
    req_to_token,
    b_seq_len,
    attn_logits,
    num_kv_splits,
    sm_scale,
    page_size,
    logit_cap=0.0,
    k_scale=None,
    v_scale=None,
    is_mla=False,
):
    _decode_grouped_att_m_fwd(
        q,
        k_buffer,
        v_buffer,
        attn_logits,
        req_to_token,
        b_seq_len,
        num_kv_splits,
        sm_scale,
        page_size,
        logit_cap,
        k_scale,
        v_scale,
        is_mla=is_mla,
    )
    _decode_softmax_reducev_fwd(
        attn_logits, q, o, lse, v_buffer, b_seq_len, num_kv_splits
    )


def decode_attention_fwd(
    # 解码注意力的统一入口函数
    # 根据 kv_group_num 自动选择 MHA 或 GQA/MQA/MLA 路径：
    # - kv_group_num == 1 -> MHA（使用 _fwd_kernel_stage1）
    # - kv_group_num > 1  -> GQA/MQA/MLA（使用 _fwd_grouped_kernel_stage1）
    q,
    k_buffer,
    v_buffer,
    o,
    lse,
    req_to_token,
    b_seq_len,
    attn_logits,
    num_kv_splits,
    sm_scale,
    page_size=1,
    logit_cap=0.0,
    k_scale=None,
    v_scale=None,
    is_mla=False,
):
    assert num_kv_splits == attn_logits.shape[2]

    if k_scale is None:
        k_scale = torch.tensor(1.0, dtype=torch.float32, device=q.device)
    if v_scale is None:
        v_scale = torch.tensor(1.0, dtype=torch.float32, device=q.device)

    kv_group_num = q.shape[1] // v_buffer.shape[-2]

    if kv_group_num == 1:
        # MHA：每个 Q 头对应一个 KV 头
        decode_attention_fwd_normal(
            q,
            k_buffer,
            v_buffer,
            o,
            lse,
            req_to_token,
            b_seq_len,
            attn_logits,
            num_kv_splits,
            sm_scale,
            page_size,
            logit_cap,
            k_scale,
            v_scale,
        )
    else:
        # GQA/MQA/MLA：多个 Q 头共享一个 KV 头
        decode_attention_fwd_grouped(
            q,
            k_buffer,
            v_buffer,
            o,
            lse,
            req_to_token,
            b_seq_len,
            attn_logits,
            num_kv_splits,
            sm_scale,
            page_size,
            logit_cap,
            k_scale,
            v_scale,
            is_mla=is_mla,
        )
