# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
基于 Triton 的 Top-K / Top-P 联合采样内核。

本模块实现了高性能的 GPU Top-K 和 Top-P 掩码操作，基于论文：
"Qrita: High-performance Top-k and Top-p Algorithm for GPUs
 using Pivot-based Truncation and Selection"
作者: Park et al. (https://arxiv.org/abs/2602.01518)

核心思想：
传统的 Top-K/Top-P 实现需要对整个词表排序，时间复杂度为 O(V log V)。
本实现使用基于枢轴(pivot)的三分搜索(ternary search)策略，将复杂度
降低到接近 O(V)，显著提升性能。

算法流程概述：
1. 统计预估：从首个数据块采样计算均值和标准差，预估数据分布
2. 离群值收集：基于高斯分布假设，收集超过预估阈值的离群值到缓冲区
3. 枢轴搜索：使用三分搜索在离群值中找到 Top-K 的枢轴值
4. Top-P 搜索：在 Top-K 结果上进一步搜索 Top-P 的概率枢轴
5. 掩码应用：根据枢轴值对 logits 进行掩码处理

三分搜索的优势：
与二分搜索相比，三分搜索每次迭代检查两个候选枢轴点（1/3 和 2/3 位置），
可以更快地收敛到目标值，同时更好地处理重复值。
"""

import torch

from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import next_power_of_2
from vllm.utils.platform_utils import num_compute_units

# ============================================================
# 缓存：避免重复创建设备端张量
# ============================================================
# 表查找缓存：每个设备存储一组查找表（normal_cdf_to_sigma, percentile_to_std）
_TRITON_TABLE_CACHE: dict[tuple[torch.device], tuple[torch.Tensor, torch.Tensor]] = {}
# 缓冲区缓存：每个 (设备, 数据类型, 词表大小) 组合对应一个缓冲区
_TRITON_BUFFER_CACHE: dict[tuple[torch.device, torch.dtype, int], torch.Tensor] = {}

# ============================================================
# 查找表：用于预估分布参数
# ============================================================
# 正态 CDF 到 sigma 映射表：
# 将 Top-P 的百分位数映射到对应的正态分布标准差倍数
# 用于独立 Top-P 场景下估计离群值阈值
# fmt: off
_NORMAL_CDF_TO_SIGMA_TABLE = [
  3.656,  3.650,  3.650,  3.650,  3.626,  3.626,  3.626,  3.514,  3.514,  3.503,
  3.503,  3.434,  3.434,  3.428,  3.428,  3.387,  3.380,  3.380,  3.376,  3.373,
  3.373,  3.356,  3.354,  3.354,  3.291,  3.249,  3.234,  3.214,  3.198,  3.198,
  3.185,  3.177,  3.177,  3.165,  3.164,  3.161,  3.138,  3.120,  3.115,  3.113,
  3.093,  3.066,  3.054,  3.043,  3.037,  3.023,  2.993,  2.991,  2.976,  2.970,
  2.952,  2.946,  2.932,  2.908,  2.902,  2.895,  2.886,  2.874,  2.861,  2.844,
  2.836,  2.810,  2.801,  2.790,  2.784,  2.779,  2.767,  2.757,  2.745,  2.733,
  2.723,  2.716,  2.693,  2.678,  2.671,  2.656,  2.649,  2.629,  2.611,  2.595,
  2.592,  2.585,  2.574,  2.550,  2.543,  2.534,  2.521,  2.518,  2.497,  2.485,
  2.468,  2.450,  2.441,  2.430,  2.412,  2.402,  2.389,  2.383,  2.377,  2.364,
  2.349,  2.338,  2.332,  2.319,  2.310,  2.301,  2.282,  2.274,  2.266,  2.250,
  2.242,  2.236,  2.226,  2.215,  2.207,  2.196,  2.179,  2.171,  2.162,  2.147,
  2.135,  2.121,  2.109,  2.095,  2.085,  2.073,  2.063,  2.045,  2.030,  2.016,
  2.003,  1.992,  1.983,  1.972,  1.960,  1.949,  1.940,  1.928,  1.912,  1.897,
  1.881,  1.869,  1.854,  1.838,  1.824,  1.807,  1.792,  1.779,  1.764,  1.751,
  1.739,  1.726,  1.711,  1.697,  1.685,  1.668,  1.652,  1.636,  1.622,  1.603,
  1.585,  1.568,  1.551,  1.534,  1.513,  1.499,  1.480,  1.464,  1.441,  1.422,
  1.394,  1.373,  1.347,  1.320,  1.296,  1.270,  1.246,  1.219,  1.190,  1.163,
  1.135,  1.104,  1.073,  1.041,  1.006,  0.969,  0.931,  0.894,  0.851,  0.806,
  0.757,  0.702,  0.643,  0.574,  0.498,  0.405,  0.288,  0.134, -0.110, -3.813
]

# 百分位到标准差映射表：
# 将 Top-K 的百分位（k/V * 200）映射到对应的正态分布标准差倍数
# 用于 Top-K 场景下估计离群值阈值
_PERCENTILE_TO_STD_TABLE = [
  2.576,  2.319,  2.178,  2.064,  1.968,  1.892,  1.819,  1.757,  1.708,  1.659,
  1.616,  1.568,  1.526,  1.492,  1.456,  1.420,  1.382,  1.342,  1.309,  1.280,
  1.249,  1.221,  1.193,  1.169,  1.145,  1.121,  1.095,  1.073,  1.050,  1.030,
  1.008,  0.987,  0.966,  0.945,  0.926,  0.910,  0.891,  0.871,  0.854,  0.837,
  0.819,  0.803,  0.784,  0.767,  0.753,  0.734,  0.719,  0.702,  0.690,  0.675,
  0.658,  0.640,  0.625,  0.609,  0.595,  0.578,  0.564,  0.550,  0.537,  0.521,
  0.509,  0.495,  0.481,  0.466,  0.453,  0.439,  0.424,  0.410,  0.397,  0.383,
  0.370,  0.356,  0.343,  0.330,  0.316,  0.302,  0.289,  0.274,  0.261,  0.247,
  0.235,  0.223,  0.209,  0.196,  0.184,  0.172,  0.159,  0.149,  0.137,  0.124,
  0.112,  0.100,  0.086,  0.074,  0.062,  0.050,  0.035,  0.023,  0.009, -0.003,
 -0.015, -0.027, -0.039, -0.052, -0.063, -0.074, -0.085, -0.097, -0.109, -0.122,
 -0.134, -0.147, -0.158, -0.171, -0.184, -0.196, -0.210, -0.223, -0.235, -0.248,
 -0.261, -0.275, -0.289, -0.302, -0.317, -0.328, -0.341, -0.353, -0.368, -0.382,
 -0.396, -0.410, -0.426, -0.439, -0.452, -0.465, -0.480, -0.493, -0.507, -0.521,
 -0.537, -0.551, -0.568, -0.582, -0.597, -0.614, -0.628, -0.643, -0.658, -0.673,
 -0.691, -0.706, -0.721, -0.738, -0.754, -0.769, -0.789, -0.808, -0.824, -0.838,
 -0.857, -0.877, -0.893, -0.912, -0.929, -0.947, -0.965, -0.983, -1.003, -1.027,
 -1.050, -1.070, -1.092, -1.117, -1.139, -1.162, -1.189, -1.216, -1.241, -1.272,
 -1.300, -1.330, -1.367, -1.404, -1.441, -1.485, -1.523, -1.564, -1.607, -1.658,
 -1.710, -1.778, -1.832, -1.901, -1.978, -2.068, -2.174, -2.325, -2.577, -3.813
]
# fmt: on


@triton.jit
def _update_min_larger_stats(data, above_mask, min_larger, num_min_larger, sentinel):
    """更新跨 tile 的 "大于枢轴的最小值" 的运行统计。

    跟踪严格大于枢轴的最小值及其出现次数。
    每个 tile 每个枢轴调用一次；运行状态通过 min_larger / num_min_larger
    在 tile 之间传递。

    合并规则：
      - tile 最小值 < 运行最小值 → 替换两者
      - tile 最小值 == 运行最小值 → 累加计数
      - tile 最小值 > 运行最小值 → 保持运行值不变

    Args:
        data: 当前 tile 的数据
        above_mask: 布尔掩码，标记哪些值大于枢轴
        min_larger: 当前运行的大于枢轴的最小值
        num_min_larger: 当前运行的最小值出现次数
        sentinel: 用于标记非有效值的哨兵值（通常为 inf）

    Returns:
        (min_larger, num_min_larger) 更新后的统计值
    """
    # 找出当前 tile 中大于枢轴的最小值
    tile_min = tl.min(tl.where(above_mask, data, sentinel))
    # 计算当前 tile 中等于该最小值的元素数量
    tile_eq = above_mask & (tl.abs(data - tile_min) < 1e-9)
    tile_cnt = tl.sum(tile_eq)
    # 判断 tile 最小值与运行最小值的关系
    is_new = tile_min < min_larger
    is_same = tl.abs(tile_min - min_larger) < 1e-9
    # 根据关系更新计数：新的则替换，相同的则累加
    num_min_larger = tl.where(is_new, tile_cnt, num_min_larger + tile_cnt * is_same)
    min_larger = tl.minimum(min_larger, tile_min)
    return min_larger, num_min_larger


@triton.jit
def _topk_topp_kernel(
    LOGITS,
    LOGITS_STRIDE_0,
    BUFFER,
    PERCENTILE_TO_STD_TABLE,
    NORMAL_CDF_TO_SIGMA_TABLE,
    K,
    P,
    BATCH_SIZE,
    VOCAB_SIZE: tl.constexpr,
    MASK_VALUE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_SIZE_TRUNC: tl.constexpr,
    TOPK_ENABLED: tl.constexpr,
    TOPP_ENABLED: tl.constexpr,
):
    """
    Top-K / Top-P 联合 Triton 内核。

    本内核是整个模块的核心，为每个 batch 中的请求执行以下操作：
    1. 计算分布统计量（均值、标准差）
    2. 使用高斯假设收集离群值
    3. 使用三分搜索找到 Top-K 枢轴
    4. 可选地在 Top-K 基础上搜索 Top-P 枢轴
    5. 应用最终掩码到 logits

    每个 Triton 程序（program）处理一个或多个请求行（row）。
    多个请求通过 grid 循环分配到不同的 SM 上。

    参数说明：
        LOGITS: logits 张量的指针
        LOGITS_STRIDE_0: logits 张量第一维的步长
        BUFFER: 用于收集离群值的临时缓冲区（每个 program 独立）
        PERCENTILE_TO_STD_TABLE: 百分位到标准差的查找表（Top-K 用）
        NORMAL_CDF_TO_SIGMA_TABLE: 正态 CDF 到 sigma 的查找表（Top-P 用）
        K: Top-K 值的指针
        P: Top-P 值的指针
        BATCH_SIZE: 批次大小
        VOCAB_SIZE: 词表大小（编译时常量）
        MASK_VALUE: 被掩码位置的填充值（编译时常量，默认 -inf）
        BLOCK_SIZE: 主数据块大小（编译时常量）
        BLOCK_SIZE_TRUNC: 离群值缓冲区的块大小（编译时常量）
        TOPK_ENABLED: 是否启用 Top-K（编译时常量）
        TOPP_ENABLED: 是否启用 Top-P（编译时常量）
    """
    # 计算需要处理的 tile 数量
    NUM_TILES: tl.constexpr = (VOCAB_SIZE + BLOCK_SIZE - 1) // BLOCK_SIZE
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    # 循环处理分配给当前 program 的所有行
    for row_id in tl.range(pid, BATCH_SIZE, num_programs):
        LOGITS_ROW = LOGITS + row_id * LOGITS_STRIDE_0
        BUFFER_ROW = BUFFER + pid * VOCAB_SIZE

        # 初始化关键状态变量
        final_pivot = -float("inf")  # 最终用于掩码的枢轴值
        duplicate_logit = float("inf")  # 与枢轴值重复的 logit 值
        num_duplicate_logit = tl.zeros((), dtype=tl.uint32)  # 重复值的总数
        num_keep = tl.zeros((), dtype=tl.uint32)  # 需要保留的重复值数量
        num_kept = tl.zeros((), dtype=tl.uint32)  # 已保留的重复值计数

        max_logit = -float("inf")  # 当前行的最大 logit
        min_logit = float("inf")  # 当前行的最小 logit（不含 -inf）

        if TOPK_ENABLED:
            k = tl.load(K + row_id)
            if k < VOCAB_SIZE:
                # ============================================================
                # 第零遍：从首个数据块计算统计量（均值和标准差）
                # ============================================================
                # 采样第一个 block 的数据来估计整体分布参数
                offs = tl.arange(0, BLOCK_SIZE)
                mask_n = offs < VOCAB_SIZE
                logits_blk0 = tl.load(
                    LOGITS_ROW + offs, mask=mask_n, other=-float("inf")
                )
                # 排除 -inf 值（例如来自语法掩码的值），避免枢轴计算中出现 NaN
                finite_mask = (logits_blk0 > -float("inf")) & mask_n
                num_finite = tl.sum(finite_mask)
                finite_logits = tl.where(finite_mask, logits_blk0, 0.0)
                avg_logit = tl.where(
                    num_finite > 0, tl.sum(finite_logits) / num_finite, 0.0
                )
                sq_avg_logit = tl.where(
                    num_finite > 0,
                    tl.sum(finite_logits * finite_logits) / num_finite,
                    0.0,
                )
                std_logit = tl.sqrt(
                    tl.maximum(sq_avg_logit - avg_logit * avg_logit, 0.0)
                )

                # 根据 Top-K 百分位计算高斯截断的异常值枢轴
                percentile = tl.cast(k / VOCAB_SIZE * 200, tl.uint32)
                percentile = tl.minimum(percentile, 199)
                sigma = tl.load(PERCENTILE_TO_STD_TABLE + percentile)
                # 应用一个负偏移系数 (-0.15) 来扩大收集范围，确保不遗漏
                sigma = sigma + tl.abs(sigma) * -0.15
                outlier_pivot = avg_logit + std_logit * sigma
                num_outliers = tl.zeros((), dtype=tl.uint32)

                # ============================================================
                # 第一遍：计算最大/最小 logits，收集离群值到缓冲区
                # ============================================================
                num_finite_total = tl.zeros((), dtype=tl.uint32)
                for i in range(0, NUM_TILES):
                    offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                    mask_n = offs_n < VOCAB_SIZE
                    logits_blk = tl.load(
                        LOGITS_ROW + offs_n, mask=mask_n, other=-float("inf")
                    )

                    max_logit = tl.maximum(max_logit, tl.max(logits_blk))
                    # 排除 -inf 以保持二分搜索的有限范围（避免 NaN 枢轴）
                    finite_blk_mask = logits_blk > -float("inf")
                    finite_blk = tl.where(finite_blk_mask, logits_blk, float("inf"))
                    min_logit = tl.minimum(min_logit, tl.min(finite_blk))
                    num_finite_total += tl.sum(finite_blk_mask & mask_n)

                    # 收集超过离群值阈值的 logits 到缓冲区
                    outlier_mask = (logits_blk > outlier_pivot) & mask_n
                    cumulative_pos = tl.cast(
                        tl.cumsum(outlier_mask) - 1 + num_outliers, tl.int32
                    )
                    num_outliers += tl.sum(outlier_mask)
                    write_pos = tl.where(outlier_mask, cumulative_pos, -1)
                    tl.store(BUFFER_ROW + write_pos, logits_blk, mask=outlier_mask)

                # 如果没有有限 logits（全是 -inf），将 min 钳位到 max
                # 使搜索收敛到 -inf（不进行掩码）
                min_logit = tl.minimum(min_logit, max_logit)

                # ============================================================
                # 第二遍：在离群值中进行三分搜索找 Top-K 枢轴
                # ============================================================
                num_iters = 0
                k_pivot = float("inf")  # Top-K 枢轴值
                k_pivots_num = tl.zeros((), dtype=tl.uint32)  # 大于枢轴的元素数
                min_larger = float("inf")  # 大于枢轴的最小值
                num_min_larger = tl.zeros((), dtype=tl.uint32)  # 该最小值的出现次数
                if num_outliers > k:
                    # 离群值足够多，仅在离群值缓冲区中搜索
                    max_range = max_logit
                    min_range = outlier_pivot
                    search_range = tl.cast(num_outliers, tl.int32)
                    search_iters = tl.cast(
                        (num_outliers + BLOCK_SIZE_TRUNC - 1) // BLOCK_SIZE_TRUNC,
                        tl.int32,
                    )
                    found_pivot = 0
                    while found_pivot == 0:
                        # 计算两个候选枢轴点（区间的 1/3 和 2/3 位置）
                        k_pivot_0 = (max_range - min_range) * 1.0 / 3.0 + min_range
                        k_pivots_num_0 = tl.zeros((), dtype=tl.uint32)
                        min_larger_0 = float("inf")
                        num_min_larger_0 = tl.zeros((), dtype=tl.uint32)

                        k_pivot_1 = (max_range - min_range) * 2.0 / 3.0 + min_range
                        k_pivots_num_1 = tl.zeros((), dtype=tl.uint32)
                        min_larger_1 = float("inf")
                        num_min_larger_1 = tl.zeros((), dtype=tl.uint32)

                        # 融合的单遍扫描：同时计算 k_pivots_num、min_larger
                        # 和 num_min_larger，避免二次数据扫描
                        for i in range(0, search_iters):
                            offs_n = i * BLOCK_SIZE_TRUNC + tl.arange(
                                0, BLOCK_SIZE_TRUNC
                            )
                            mask_n_2 = offs_n < search_range
                            logits_blk2 = tl.load(
                                BUFFER_ROW + offs_n, mask=mask_n_2, other=-float("inf")
                            )

                            above_0 = logits_blk2 > k_pivot_0
                            above_1 = logits_blk2 > k_pivot_1
                            k_pivots_num_0 += tl.sum(above_0)
                            k_pivots_num_1 += tl.sum(above_1)

                            min_larger_0, num_min_larger_0 = _update_min_larger_stats(
                                logits_blk2,
                                above_0,
                                min_larger_0,
                                num_min_larger_0,
                                float("inf"),
                            )
                            min_larger_1, num_min_larger_1 = _update_min_larger_stats(
                                logits_blk2,
                                above_1,
                                min_larger_1,
                                num_min_larger_1,
                                float("inf"),
                            )

                        # 检查终止条件：大于枢轴的元素数 >= k
                        # 且减去重复值后 < k（说明枢轴恰好落在重复值上）
                        if (
                            k_pivots_num_0 >= k
                            and k_pivots_num_0 - num_min_larger_0 < k
                        ):
                            k_pivot = k_pivot_0
                            k_pivots_num = k_pivots_num_0
                            min_larger = min_larger_0
                            num_min_larger = num_min_larger_0
                            found_pivot = 1
                        if (
                            k_pivots_num_1 >= k
                            and k_pivots_num_1 - num_min_larger_1 < k
                        ):
                            k_pivot = k_pivot_1
                            k_pivots_num = k_pivots_num_1
                            min_larger = min_larger_1
                            num_min_larger = num_min_larger_1
                            found_pivot = 1

                        # 根据搜索结果更新搜索范围
                        if k_pivots_num_1 > k:
                            min_range = k_pivot_1
                        elif k_pivots_num_0 > k:
                            min_range = k_pivot_0

                        if k_pivots_num_0 < k:
                            max_range = k_pivot_0
                        elif k_pivots_num_1 < k:
                            max_range = k_pivot_1

                        num_iters += 1
                        # 最多迭代 18 次或范围足够小时终止
                        if num_iters >= 18 or tl.abs(min_range - max_range) < 1e-9:
                            k_pivot = (max_range + min_range) / 2.0
                            found_pivot = 1
                else:
                    # 离群值收集失败，在整个 logits 空间中搜索
                    max_range = max_logit
                    min_range = min_logit
                    found_pivot = 0
                    while found_pivot == 0:
                        # 使用 1/4 和 2/4 位置作为候选枢轴（四分搜索）
                        k_pivot_0 = (max_range - min_range) * 1.0 / 4.0 + min_range
                        k_pivots_num_0 = tl.zeros((), dtype=tl.uint32)
                        min_larger_0 = float("inf")
                        num_min_larger_0 = tl.zeros((), dtype=tl.uint32)

                        k_pivot_1 = (max_range - min_range) * 2.0 / 4.0 + min_range
                        k_pivots_num_1 = tl.zeros((), dtype=tl.uint32)
                        min_larger_1 = float("inf")
                        num_min_larger_1 = tl.zeros((), dtype=tl.uint32)

                        # 融合的单遍扫描（与上方缓冲区路径相同的方法）
                        for i in range(0, NUM_TILES):
                            offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                            mask_n = offs_n < VOCAB_SIZE
                            logits_blk2 = tl.load(
                                LOGITS_ROW + offs_n, mask=mask_n, other=-float("inf")
                            )

                            above_0 = logits_blk2 > k_pivot_0
                            above_1 = logits_blk2 > k_pivot_1
                            k_pivots_num_0 += tl.sum(above_0)
                            k_pivots_num_1 += tl.sum(above_1)

                            min_larger_0, num_min_larger_0 = _update_min_larger_stats(
                                logits_blk2,
                                above_0,
                                min_larger_0,
                                num_min_larger_0,
                                float("inf"),
                            )
                            min_larger_1, num_min_larger_1 = _update_min_larger_stats(
                                logits_blk2,
                                above_1,
                                min_larger_1,
                                num_min_larger_1,
                                float("inf"),
                            )

                        # 检查终止条件
                        if (
                            k_pivots_num_0 >= k
                            and k_pivots_num_0 - num_min_larger_0 < k
                        ):
                            k_pivot = k_pivot_0
                            k_pivots_num = k_pivots_num_0
                            min_larger = min_larger_0
                            num_min_larger = num_min_larger_0
                            found_pivot = 1
                        if (
                            k_pivots_num_1 >= k
                            and k_pivots_num_1 - num_min_larger_1 < k
                        ):
                            k_pivot = k_pivot_1
                            k_pivots_num = k_pivots_num_1
                            min_larger = min_larger_1
                            num_min_larger = num_min_larger_1
                            found_pivot = 1

                        # 更新搜索范围
                        if k_pivots_num_1 > k:
                            min_range = k_pivot_1
                        elif k_pivots_num_0 > k:
                            min_range = k_pivot_0

                        if k_pivots_num_0 < k:
                            max_range = k_pivot_0
                        elif k_pivots_num_1 < k:
                            max_range = k_pivot_1

                        num_iters += 1
                        if num_iters >= 18 or tl.abs(min_range - max_range) < 1e-9:
                            k_pivot = (max_range + min_range) / 2.0
                            found_pivot = 1

                # 处理与枢轴值相等的重复 logit
                duplicate_logit = min_larger
                num_duplicate_logit = num_min_larger
                # 需要保留的重复值数量 = 总重复数 - 超出 k 的部分
                num_keep = num_duplicate_logit - (k_pivots_num - k)
                num_kept = tl.zeros((), dtype=tl.uint32)

                # 仅 Top-K 路径。如果有限值数量少于 k（例如语法掩码），
                # 保留所有值
                final_pivot = k_pivot if num_finite_total > k else -float("inf")

                if TOPP_ENABLED and num_finite_total > k:
                    #### 在 Top-K 结果上进行 Top-P 采样 ####
                    p = tl.load(P + row_id)
                    if p < 1.0:
                        min_logit = k_pivot
                        sum_exp_logits = 0.0
                        num_outliers_2 = tl.zeros((), dtype=tl.uint32)
                        search_range = tl.cast(num_outliers, tl.int32)
                        search_iters = tl.cast(
                            (num_outliers + BLOCK_SIZE_TRUNC - 1) // BLOCK_SIZE_TRUNC,
                            tl.int32,
                        )

                        # ============================================================
                        # 第三遍：计算 exp(logits) 和总和，收集离群值
                        # ============================================================
                        if num_outliers > k:
                            for i in range(0, search_iters):
                                offs_n = i * BLOCK_SIZE_TRUNC + tl.arange(
                                    0, BLOCK_SIZE_TRUNC
                                )
                                mask_n_2 = offs_n < search_range

                                probs_blk = tl.load(
                                    BUFFER_ROW + offs_n,
                                    mask=mask_n_2,
                                    other=-float("inf"),
                                )

                                outlier_mask = (probs_blk > min_logit) & mask_n_2

                                # 处理与 Top-K 枢轴重复的 logit
                                if num_keep < num_duplicate_logit:
                                    duplicate_mask = (
                                        tl.abs(probs_blk - duplicate_logit) < 1e-9
                                    )
                                    duplicate_count = (
                                        tl.cumsum(duplicate_mask) + num_kept
                                    )
                                    duplicate_keep_mask = (
                                        duplicate_count <= num_keep
                                    ) & duplicate_mask
                                    duplicate_remove_mask = (
                                        duplicate_mask & ~duplicate_keep_mask
                                    )
                                    outlier_mask = outlier_mask & (
                                        ~duplicate_remove_mask
                                    )
                                    num_kept += tl.sum(duplicate_keep_mask)

                                probs_blk = tl.where(
                                    outlier_mask, probs_blk, -float("inf")
                                )
                                probs_blk = probs_blk - max_logit
                                probs_blk = tl.exp(probs_blk)
                                sum_exp_logits += tl.sum(probs_blk)

                            # ============================================================
                            # 第四遍：计算归一化概率并存入缓冲区
                            # ============================================================
                            for i in range(0, search_iters):
                                offs_n = i * BLOCK_SIZE_TRUNC + tl.arange(
                                    0, BLOCK_SIZE_TRUNC
                                )
                                mask_n_2 = offs_n < search_range

                                probs_blk = tl.load(
                                    BUFFER_ROW + offs_n, mask=mask_n_2, other=0.0
                                )
                                probs_blk = probs_blk / sum_exp_logits
                                tl.store(BUFFER_ROW + offs_n, probs_blk, mask=mask_n_2)
                        else:
                            # Top-K 离群值收集失败，使用 Top-K 枢轴重新收集
                            for i in range(0, NUM_TILES):
                                offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                                mask_n = offs_n < VOCAB_SIZE

                                probs_blk = tl.load(
                                    LOGITS_ROW + offs_n,
                                    mask=mask_n,
                                    other=-float("inf"),
                                )

                                outlier_mask = (probs_blk > min_logit) & mask_n

                                # 处理重复 logit
                                duplicate_mask = (
                                    tl.abs(probs_blk - duplicate_logit) < 1e-9
                                )
                                duplicate_count = tl.cumsum(duplicate_mask) + num_kept
                                duplicate_keep_mask = (
                                    duplicate_count <= num_keep
                                ) & duplicate_mask
                                duplicate_remove_mask = (
                                    duplicate_mask & ~duplicate_keep_mask
                                )
                                outlier_mask = outlier_mask & (~duplicate_remove_mask)
                                num_kept += tl.sum(duplicate_keep_mask)

                                probs_blk = tl.where(
                                    outlier_mask, probs_blk, -float("inf")
                                )
                                probs_blk = probs_blk - max_logit
                                probs_blk = tl.exp(probs_blk)
                                sum_exp_logits += tl.sum(probs_blk)

                                cumulative_pos = tl.cast(
                                    tl.cumsum(outlier_mask) - 1 + num_outliers_2,
                                    tl.int32,
                                )
                                num_outliers_2 += tl.sum(outlier_mask)
                                write_pos = tl.where(outlier_mask, cumulative_pos, -1)
                                tl.store(
                                    BUFFER_ROW + write_pos, probs_blk, mask=outlier_mask
                                )

                            search_range = tl.cast(num_outliers_2, tl.int32)
                            search_iters = tl.cast(
                                (num_outliers_2 + BLOCK_SIZE_TRUNC - 1)
                                // BLOCK_SIZE_TRUNC,
                                tl.int32,
                            )

                            # 归一化概率
                            for i in range(0, search_iters):
                                offs_n = i * BLOCK_SIZE_TRUNC + tl.arange(
                                    0, BLOCK_SIZE_TRUNC
                                )
                                mask_n_2 = offs_n < search_range

                                probs_blk = tl.load(
                                    BUFFER_ROW + offs_n, mask=mask_n_2, other=0.0
                                )
                                probs_blk = probs_blk / sum_exp_logits
                                tl.store(BUFFER_ROW + offs_n, probs_blk, mask=mask_n_2)

                        # 计算概率范围用于 Top-P 搜索
                        max_range = tl.exp(max_logit - max_logit) / sum_exp_logits
                        min_range = tl.exp(min_logit - max_logit) / sum_exp_logits

                        p_pivot = 1.0
                        num_iters = 0
                        min_larger_prob = 1.0
                        num_min_larger = tl.zeros((), dtype=tl.uint32)
                        p_pivots_sum = 0.0

                        # ============================================================
                        # 第五遍：搜索 Top-P 枢轴
                        # ============================================================
                        found_pivot = 0
                        while found_pivot == 0:
                            # 计算两个候选概率枢轴（1/3 和 2/3 位置）
                            p_pivot_0 = (max_range - min_range) * 1.0 / 3.0 + min_range
                            p_pivots_sum_0 = 0.0
                            min_larger_0 = 1.0
                            num_min_larger_0 = tl.zeros((), dtype=tl.uint32)

                            p_pivot_1 = (max_range - min_range) * 2.0 / 3.0 + min_range
                            p_pivots_sum_1 = 0.0
                            min_larger_1 = 1.0
                            num_min_larger_1 = tl.zeros((), dtype=tl.uint32)

                            # 第一遍：计算 p_pivots_sum 和 min_larger
                            for i in range(0, search_iters):
                                offs_n = i * BLOCK_SIZE_TRUNC + tl.arange(
                                    0, BLOCK_SIZE_TRUNC
                                )
                                mask_n_2 = offs_n < search_range
                                probs_blk = tl.load(
                                    BUFFER_ROW + offs_n, mask=mask_n_2, other=0.0
                                )

                                p_pivots_sum_0 += tl.sum(
                                    probs_blk * (probs_blk > p_pivot_0)
                                )
                                masked_larger_0 = tl.where(
                                    probs_blk > p_pivot_0, probs_blk, 1.0
                                )
                                min_larger_0 = tl.minimum(
                                    min_larger_0, tl.min(masked_larger_0)
                                )

                                p_pivots_sum_1 += tl.sum(
                                    probs_blk * (probs_blk > p_pivot_1)
                                )
                                masked_larger_1 = tl.where(
                                    probs_blk > p_pivot_1, probs_blk, 1.0
                                )
                                min_larger_1 = tl.minimum(
                                    min_larger_1, tl.min(masked_larger_1)
                                )

                            # 第二遍：计算 num_min_larger
                            for i in range(0, search_iters):
                                offs_n = i * BLOCK_SIZE_TRUNC + tl.arange(
                                    0, BLOCK_SIZE_TRUNC
                                )
                                mask_n_2 = offs_n < search_range
                                probs_blk = tl.load(
                                    BUFFER_ROW + offs_n, mask=mask_n_2, other=0.0
                                )

                                num_min_larger_0 += tl.sum(
                                    tl.abs(probs_blk - min_larger_0) < 1e-9
                                )
                                num_min_larger_1 += tl.sum(
                                    tl.abs(probs_blk - min_larger_1) < 1e-9
                                )

                            # 检查终止条件
                            if p_pivots_sum_1 >= p and (
                                p_pivots_sum_1 - (min_larger_1 * num_min_larger_1) < p
                            ):
                                p_pivot = p_pivot_1
                                min_larger_prob = min_larger_1
                                num_min_larger = num_min_larger_1
                                p_pivots_sum = p_pivots_sum_1
                                found_pivot = 1
                            if p_pivots_sum_0 >= p and (
                                p_pivots_sum_0 - (min_larger_0 * num_min_larger_0) < p
                            ):
                                p_pivot = p_pivot_0
                                min_larger_prob = min_larger_0
                                num_min_larger = num_min_larger_0
                                p_pivots_sum = p_pivots_sum_0
                                found_pivot = 1

                            # 更新搜索范围
                            if p_pivots_sum_1 > p:
                                min_range = p_pivot_1
                            elif p_pivots_sum_0 > p:
                                min_range = p_pivot_0

                            if p_pivots_sum_0 < p:
                                max_range = p_pivot_0
                            elif p_pivots_sum_1 < p:
                                max_range = p_pivot_1

                            num_iters += 1
                            if (max_range - min_range) < 1e-9 or num_iters >= 18:
                                p_pivot = (max_range + min_range) / 2.0
                                found_pivot = 1

                        # 将概率枢轴转换回 logit 空间
                        duplicate_logit = (
                            tl.log(min_larger_prob * sum_exp_logits) + max_logit
                        )
                        num_duplicate_logit = num_min_larger
                        num_keep = num_duplicate_logit - tl.cast(
                            (p_pivots_sum - p) / min_larger_prob, tl.uint32
                        )
                        num_kept = tl.zeros((), dtype=tl.uint32)

                        # Top-K + Top-P 联合路径的最终枢轴
                        final_pivot = tl.log(p_pivot * sum_exp_logits) + max_logit

        if TOPP_ENABLED and final_pivot == -float("inf"):
            #### 独立的 Top-P 采样（无 Top-K） ####
            p = tl.load(P + row_id)
            if p < 1.0:
                # ============================================================
                # 第零遍：从首个数据块计算统计量
                # ============================================================
                offs = tl.arange(0, BLOCK_SIZE)
                mask_n = offs < VOCAB_SIZE
                logits_blk0 = tl.load(
                    LOGITS_ROW + offs, mask=mask_n, other=-float("inf")
                )
                # 排除 -inf 值以避免 NaN
                finite_mask = (logits_blk0 > -float("inf")) & mask_n
                num_finite = tl.sum(finite_mask)
                finite_logits = tl.where(finite_mask, logits_blk0, 0.0)
                avg_logit = tl.where(
                    num_finite > 0, tl.sum(finite_logits) / num_finite, 0.0
                )
                sq_avg_logit = tl.where(
                    num_finite > 0,
                    tl.sum(finite_logits * finite_logits) / num_finite,
                    0.0,
                )
                std_logit = tl.sqrt(
                    tl.maximum(sq_avg_logit - avg_logit * avg_logit, 0.0)
                )
                # 使用均值 + 10 倍标准差作为数值稳定的最大值参考
                max_sample = avg_logit + std_logit * 10.0
                sum_exp_logits = 0.0

                # ============================================================
                # 第一遍：计算最大/最小 logits 和 exp 总和
                # ============================================================
                for i in range(0, NUM_TILES):
                    offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                    mask_n = offs_n < VOCAB_SIZE
                    logits_blk = tl.load(
                        LOGITS_ROW + offs_n, mask=mask_n, other=-float("inf")
                    )
                    max_logit = tl.maximum(max_logit, tl.max(logits_blk))
                    # 排除 -inf 以保持搜索范围有限
                    finite_blk = tl.where(
                        logits_blk > -float("inf"), logits_blk, float("inf")
                    )
                    min_logit = tl.minimum(min_logit, tl.min(finite_blk))

                    probs_blk = tl.exp(logits_blk - max_sample)
                    probs_blk = tl.where(mask_n, probs_blk, 0.0)
                    sum_exp_logits += tl.sum(probs_blk)

                # 如果没有有限 logits，钳位 min 到 max
                min_logit = tl.minimum(min_logit, max_logit)

                # 使用正态 CDF 查找表估计离群值阈值
                idx = tl.cast(p * 200, tl.int32)
                idx = tl.maximum(0, tl.minimum(idx, 199))
                sigma = tl.load(NORMAL_CDF_TO_SIGMA_TABLE + idx)
                # 应用负偏移系数 (-0.25) 扩大收集范围
                sigma = sigma + tl.abs(sigma) * -0.25
                outlier_pivot = avg_logit + std_logit * sigma

                outlier_prob = tl.exp(outlier_pivot - max_sample) / sum_exp_logits
                sum_outlier_probs = 0.0
                num_outliers = tl.zeros((), dtype=tl.uint32)

                # ============================================================
                # 第二遍：计算 softmax 概率并收集离群值
                # ============================================================
                for i in range(0, NUM_TILES):
                    offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                    mask_n = offs_n < VOCAB_SIZE

                    probs_blk = tl.load(
                        LOGITS_ROW + offs_n, mask=mask_n, other=-float("inf")
                    )
                    probs_blk = tl.exp(probs_blk - max_sample)
                    probs_blk = probs_blk / sum_exp_logits

                    outlier_mask = (probs_blk > outlier_prob) & mask_n
                    sum_outlier_probs += tl.sum(outlier_mask * probs_blk)
                    cumulative_pos = tl.cast(
                        tl.cumsum(outlier_mask) - 1 + num_outliers, tl.int32
                    )
                    num_outliers += tl.sum(outlier_mask)
                    write_pos = tl.where(outlier_mask, cumulative_pos, -1)
                    tl.store(BUFFER_ROW + write_pos, probs_blk, mask=outlier_mask)

                max_range = tl.exp(max_logit - max_sample) / sum_exp_logits
                min_range = tl.exp(min_logit - max_sample) / sum_exp_logits

                p_pivot = 1.0
                num_iters = 0
                min_larger_prob = 1.0
                num_min_larger = tl.zeros((), dtype=tl.uint32)
                p_pivots_sum = 0.0

                # ============================================================
                # 第三遍：搜索 Top-P 枢轴
                # ============================================================
                if sum_outlier_probs > p:
                    # 离群值概率总和 > p，仅在离群值中搜索
                    min_range = outlier_prob
                    search_range = tl.cast(num_outliers, tl.int32)
                    search_iters = tl.cast(
                        (num_outliers + BLOCK_SIZE_TRUNC - 1) // BLOCK_SIZE_TRUNC,
                        tl.int32,
                    )

                    found_pivot = 0
                    while found_pivot == 0:
                        p_pivot_0 = (max_range - min_range) * 1.0 / 3.0 + min_range
                        p_pivots_sum_0 = 0.0
                        min_larger_0 = 1.0
                        num_min_larger_0 = tl.zeros((), dtype=tl.uint32)

                        p_pivot_1 = (max_range - min_range) * 2.0 / 3.0 + min_range
                        p_pivots_sum_1 = 0.0
                        min_larger_1 = 1.0
                        num_min_larger_1 = tl.zeros((), dtype=tl.uint32)

                        # 第一遍：计算 p_pivots_sum 和 min_larger
                        for i in range(0, search_iters):
                            offs_n = i * BLOCK_SIZE_TRUNC + tl.arange(
                                0, BLOCK_SIZE_TRUNC
                            )
                            mask_n_2 = offs_n < search_range
                            probs_blk = tl.load(
                                BUFFER_ROW + offs_n, mask=mask_n_2, other=0.0
                            )

                            p_pivots_sum_0 += tl.sum(
                                probs_blk * (probs_blk > p_pivot_0)
                            )
                            masked_larger_0 = tl.where(
                                probs_blk > p_pivot_0, probs_blk, 1.0
                            )
                            min_larger_0 = tl.minimum(
                                min_larger_0, tl.min(masked_larger_0)
                            )

                            p_pivots_sum_1 += tl.sum(
                                probs_blk * (probs_blk > p_pivot_1)
                            )
                            masked_larger_1 = tl.where(
                                probs_blk > p_pivot_1, probs_blk, 1.0
                            )
                            min_larger_1 = tl.minimum(
                                min_larger_1, tl.min(masked_larger_1)
                            )

                        # 第二遍：计算 num_min_larger
                        for i in range(0, search_iters):
                            offs_n = i * BLOCK_SIZE_TRUNC + tl.arange(
                                0, BLOCK_SIZE_TRUNC
                            )
                            mask_n_2 = offs_n < search_range
                            probs_blk = tl.load(
                                BUFFER_ROW + offs_n, mask=mask_n_2, other=0.0
                            )

                            num_min_larger_0 += tl.sum(
                                tl.abs(probs_blk - min_larger_0) < 1e-9
                            )
                            num_min_larger_1 += tl.sum(
                                tl.abs(probs_blk - min_larger_1) < 1e-9
                            )

                        # 检查终止条件
                        if (
                            p_pivots_sum_1 >= p
                            and p_pivots_sum_1 - (min_larger_1 * num_min_larger_1) < p
                        ):
                            p_pivot = p_pivot_1
                            min_larger_prob = min_larger_1
                            num_min_larger = num_min_larger_1
                            p_pivots_sum = p_pivots_sum_1
                            found_pivot = 1
                        if (
                            p_pivots_sum_0 >= p
                            and p_pivots_sum_0 - (min_larger_0 * num_min_larger_0) < p
                        ):
                            p_pivot = p_pivot_0
                            min_larger_prob = min_larger_0
                            num_min_larger = num_min_larger_0
                            p_pivots_sum = p_pivots_sum_0
                            found_pivot = 1

                        # 更新搜索范围
                        if p_pivots_sum_1 > p:
                            min_range = p_pivot_1
                        elif p_pivots_sum_0 > p:
                            min_range = p_pivot_0

                        if p_pivots_sum_0 < p:
                            max_range = p_pivot_0
                        elif p_pivots_sum_1 < p:
                            max_range = p_pivot_1

                        num_iters += 1
                        if (max_range - min_range) < 1e-9 or num_iters >= 18:
                            p_pivot = (max_range + min_range) / 2.0
                            found_pivot = 1
                else:
                    # 离群值概率不足，用完整的 softmax 概率重新填充缓冲区
                    for i in range(0, NUM_TILES):
                        offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                        mask_n = offs_n < VOCAB_SIZE

                        probs_blk = tl.load(
                            LOGITS_ROW + offs_n, mask=mask_n, other=-float("inf")
                        )
                        probs_blk = tl.exp(probs_blk - max_sample)
                        probs_blk = probs_blk / sum_exp_logits
                        tl.store(BUFFER_ROW + offs_n, probs_blk, mask=mask_n)

                    found_pivot = 0
                    while found_pivot == 0:
                        p_pivot_0 = (max_range - min_range) * 1.0 / 3.0 + min_range
                        p_pivots_sum_0 = 0.0
                        min_larger_0 = 1.0
                        num_min_larger_0 = tl.zeros((), dtype=tl.uint32)

                        p_pivot_1 = (max_range - min_range) * 2.0 / 3.0 + min_range
                        p_pivots_sum_1 = 0.0
                        min_larger_1 = 1.0
                        num_min_larger_1 = tl.zeros((), dtype=tl.uint32)

                        # 第一遍：计算 p_pivots_sum 和 min_larger
                        for i in range(0, NUM_TILES):
                            offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                            mask_n = offs_n < VOCAB_SIZE
                            probs_blk = tl.load(
                                BUFFER_ROW + offs_n, mask=mask_n, other=0.0
                            )

                            p_pivots_sum_0 += tl.sum(
                                probs_blk * (probs_blk > p_pivot_0)
                            )
                            masked_larger_0 = tl.where(
                                probs_blk > p_pivot_0, probs_blk, 1.0
                            )
                            min_larger_0 = tl.minimum(
                                min_larger_0, tl.min(masked_larger_0)
                            )

                            p_pivots_sum_1 += tl.sum(
                                probs_blk * (probs_blk > p_pivot_1)
                            )
                            masked_larger_1 = tl.where(
                                probs_blk > p_pivot_1, probs_blk, 1.0
                            )
                            min_larger_1 = tl.minimum(
                                min_larger_1, tl.min(masked_larger_1)
                            )

                        # 第二遍：计算 num_min_larger
                        for i in range(0, NUM_TILES):
                            offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                            mask_n = offs_n < VOCAB_SIZE
                            probs_blk = tl.load(
                                BUFFER_ROW + offs_n, mask=mask_n, other=0.0
                            )

                            num_min_larger_0 += tl.sum(
                                tl.abs(probs_blk - min_larger_0) < 1e-9
                            )
                            num_min_larger_1 += tl.sum(
                                tl.abs(probs_blk - min_larger_1) < 1e-9
                            )

                        # 检查终止条件
                        if (
                            p_pivots_sum_1 >= p
                            and p_pivots_sum_1 - (min_larger_1 * num_min_larger_1) < p
                        ):
                            p_pivot = p_pivot_1
                            min_larger_prob = min_larger_1
                            num_min_larger = num_min_larger_1
                            p_pivots_sum = p_pivots_sum_1
                            found_pivot = 1
                        if (
                            p_pivots_sum_0 >= p
                            and p_pivots_sum_0 - (min_larger_0 * num_min_larger_0) < p
                        ):
                            p_pivot = p_pivot_0
                            min_larger_prob = min_larger_0
                            num_min_larger = num_min_larger_0
                            p_pivots_sum = p_pivots_sum_0
                            found_pivot = 1

                        # 更新搜索范围
                        if p_pivots_sum_1 > p:
                            min_range = p_pivot_1
                        elif p_pivots_sum_0 > p:
                            min_range = p_pivot_0

                        if p_pivots_sum_0 < p:
                            max_range = p_pivot_0
                        elif p_pivots_sum_1 < p:
                            max_range = p_pivot_1

                        num_iters += 1
                        if (max_range - min_range) < 1e-9 or num_iters >= 18:
                            p_pivot = (max_range + min_range) / 2.0
                            found_pivot = 1

                # 将概率枢轴转换回 logit 空间
                duplicate_logit = tl.log(min_larger_prob * sum_exp_logits) + max_logit
                num_duplicate_logit = num_min_larger
                num_keep = num_duplicate_logit - tl.cast(
                    (p_pivots_sum - p) / min_larger_prob, tl.uint32
                )
                num_kept = tl.zeros((), dtype=tl.uint32)

                # 独立 Top-P 路径的最终枢轴
                final_pivot = tl.log(p_pivot * sum_exp_logits) + max_sample

        # ============================================================
        # 第六遍：应用掩码并写入最终输出
        # ============================================================
        # 如果枢轴 >= 最大 logit（或为 NaN），则没有 token 能通过严格的 `>` 掩码
        # 跳过掩码操作。使用 `not <` 而非 `>=` 以捕获 NaN 情况。
        if not (final_pivot < max_logit):
            final_pivot = -float("inf")
        elif final_pivot != -float("inf"):
            for i in range(0, NUM_TILES):
                offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                mask_n = offs_n < VOCAB_SIZE
                logits_blk = tl.load(
                    LOGITS_ROW + offs_n, mask=mask_n, other=-float("inf")
                )
                # 保留大于枢轴的 logits
                keep_mask = (logits_blk > final_pivot) & mask_n

                # 处理与枢轴重复的 logit：只保留 num_keep 个
                if num_keep < num_duplicate_logit:
                    duplicate_mask = (
                        tl.abs(logits_blk - duplicate_logit) < 1e-9
                    ) & mask_n
                    duplicate_count = tl.cumsum(duplicate_mask) + num_kept
                    duplicate_keep_mask = (
                        duplicate_count <= num_duplicate_logit
                    ) & duplicate_mask
                    duplicate_remove_mask = duplicate_mask & ~duplicate_keep_mask
                    num_kept += tl.sum(duplicate_keep_mask)
                    keep_mask = keep_mask & (~duplicate_remove_mask)

                # 应用掩码：保留的值不变，被掩码的设为 MASK_VALUE
                logits_blk = tl.where(keep_mask, logits_blk, MASK_VALUE)
                tl.store(LOGITS_ROW + offs_n, logits_blk, mask=mask_n)


def apply_top_k_top_p_triton(
    logits: torch.Tensor,
    k: torch.Tensor | None,
    p: torch.Tensor | None,
    mask_value: float = float("-inf"),
) -> torch.Tensor:
    """
    使用 Triton 内核对 logits 应用联合 Top-K / Top-P 掩码。

    操作顺序：先应用 Top-K（按 logit 值），再在剩余的 K 个值上
    应用 Top-P（按概率）。

    Args:
        logits: [batch_size, vocab_size] float32 张量。返回的张量可能
            是输入的别名，也可能是新的连续张量（当布局不支持时）。
        k: [batch_size] int32 张量，每行的 Top-K 值，None 表示禁用 Top-K
        p: [batch_size] float32 张量，每行的 Top-P 值（0 到 1），
            None 表示禁用 Top-P
        mask_value: 被掩码位置的填充值（默认: -inf）

    Returns:
        经掩码处理的 logits 张量，可能原地修改也可能不是。
    """
    assert logits.ndim == 2
    assert logits.dtype == torch.float32
    batch_size, vocab_size = logits.shape
    topk_enabled = k is not None
    topp_enabled = p is not None

    if batch_size == 0 or not (topk_enabled or topp_enabled):
        return logits

    # Triton 内核支持任意行步长，但假设词表维度在每行内连续布局
    if logits.stride(1) != 1:
        logits = logits.contiguous()

    if k is not None:
        assert k.ndim == 1 and k.shape[0] == batch_size
        k_ptr = k.to(torch.int32)
    else:
        k_ptr = logits  # 虚拟指针（不会被读取）

    if p is not None:
        assert p.ndim == 1 and p.shape[0] == batch_size
        p_ptr = p.to(torch.float32)
    else:
        p_ptr = logits  # 虚拟指针（不会被读取）

    # 确定启动的 program 数量（不超过 SM 数量和 batch 大小）
    num_sm = num_compute_units(logits.device.index)
    NUM_PROGRAMS = min(num_sm, batch_size)

    # 缓存每个 Triton Program 使用的临时缓冲区
    buf_key = (logits.device, logits.dtype, vocab_size)
    buffer = _TRITON_BUFFER_CACHE.get(buf_key)
    if buffer is None or buffer.shape[0] < NUM_PROGRAMS:
        size = min(next_power_of_2(NUM_PROGRAMS), num_sm)
        buffer = logits.new_empty((size, vocab_size))
        _TRITON_BUFFER_CACHE[buf_key] = buffer
    if buffer.shape[0] > NUM_PROGRAMS:
        buffer = buffer[:NUM_PROGRAMS]

    # 缓存查找表到设备端
    tables = _TRITON_TABLE_CACHE.get(logits.device)
    if tables is None:
        normal_cdf_to_sigma_table = logits.new_tensor(_NORMAL_CDF_TO_SIGMA_TABLE)
        percentile_to_std_table = logits.new_tensor(_PERCENTILE_TO_STD_TABLE)
        _TRITON_TABLE_CACHE[logits.device] = (
            normal_cdf_to_sigma_table,
            percentile_to_std_table,
        )
    else:
        normal_cdf_to_sigma_table, percentile_to_std_table = tables

    # CPU 使用较小的 tile 以加快编译和运行；GPU 使用较大的 tile 获得更好性能
    if logits.device.type == "cpu":
        block_size, block_size_trunc = 256, 128
    else:
        block_size, block_size_trunc = 8192, 4096

    _topk_topp_kernel[(NUM_PROGRAMS,)](
        logits,
        logits.stride(0),
        buffer,
        percentile_to_std_table,
        normal_cdf_to_sigma_table,
        k_ptr,
        p_ptr,
        BATCH_SIZE=batch_size,
        MASK_VALUE=mask_value,
        VOCAB_SIZE=vocab_size,
        BLOCK_SIZE=block_size,
        BLOCK_SIZE_TRUNC=block_size_trunc,
        TOPK_ENABLED=topk_enabled,
        TOPP_ENABLED=topp_enabled,
    )

    return logits


def reset_buffer_cache():
    """
    重置 Triton 内核使用的缓冲区和查找表缓存。

    在模型卸载或显存不足时调用，释放缓存占用的显存。
    """
    _TRITON_BUFFER_CACHE.clear()
    _TRITON_TABLE_CACHE.clear()
    torch.accelerator.empty_cache()
