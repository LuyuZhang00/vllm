# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# =============================================================================
# ubatch_utils.py -- Dual Batch Overlap (DBO) 微批次切分工具模块
# =============================================================================
# 本模块是 vLLM V1 推理引擎中 "Dual Batch Overlap" (DBO，双批次重叠) 优化的
# 核心工具模块。DBO 的基本思想是：将一个大的调度批次 (batch) 切分为多个微批次
# (micro-batch / ubatch)，然后在 GPU 上以流水线方式交错执行，使得通信与计算
# 可以重叠，从而提升吞吐量。
#
# 本模块提供以下功能：
#   1. 判断是否应该启用 ubatch 切分（基于 token 数量阈值）
#   2. 将一个 batch 按 token 数均匀切分为多个 UBatchSlice
#   3. 将 CommonAttentionMetadata（注意力元数据）按 ubatch 进行切分，
#      使得每个 ubatch 拥有独立的注意力元数据，可以独立执行前向计算
#
# 核心概念：
#   - UBatchSlice: 表示一个微批次在"请求维度"和"token 维度"上的切片范围
#   - 请求切片 (request_slice): 标识该 ubatch 包含哪些请求
#   - Token 切片 (token_slice): 标识该 ubatch 包含哪些 token（按全局 token 序号）
#   - 注意：一个请求可能被拆分到两个相邻的 ubatch 中（跨 ubatch 请求），
#     此时需要特殊处理 query_start_loc 和 seq_lens 等元数据
# =============================================================================

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
import torch

from vllm.config import ParallelConfig
from vllm.v1.attention.backend import CommonAttentionMetadata


@dataclass
class UBatchSlice:
    """微批次切片：描述一个 ubatch 在请求维度和 token 维度上的范围。

    每个 UBatchSlice 包含两个 slice 对象：
      - request_slice: 该 ubatch 覆盖的请求索引范围 [start, stop)
      - token_slice:   该 ubatch 覆盖的 token 全局索引范围 [start, stop)

    设计说明：
      一个 batch 中所有请求的 token 被展平为一个一维数组，
      token_slice 直接索引这个展平数组。request_slice 则用于索引
      以请求为粒度的张量（如 query_start_loc、seq_lens、block_table 等）。
      因此同一个 UBatchSlice 需要同时维护两个维度的切片。
    """
    request_slice: slice
    token_slice: slice

    def is_empty(self) -> bool:
        """判断该 ubatch 切片是否为空（不包含任何请求或 token）。"""
        return (
            self.request_slice.start == self.request_slice.stop
            or self.token_slice.start == self.token_slice.stop
        )

    @property
    def num_tokens(self) -> int:
        """该 ubatch 包含的 token 数量。"""
        return self.token_slice.stop - self.token_slice.start


# UBatchSlices: 一个 batch 切分后得到的所有 ubatch 切片列表。
# 列表中的每个元素对应一个微批次，按 token 顺序排列。
UBatchSlices: TypeAlias = list[UBatchSlice]


def is_last_ubatch_empty(
    orig_num_tokens: int, padded_num_tokens: int, num_ubatches: int
) -> bool:
    """判断在按 padded token 数均匀切分后，最后一个 ubatch 是否为空。

    判断逻辑：前 (num_ubatches - 1) 个 ubatch 每个包含
    padded_num_tokens // num_ubatches 个 token，如果这些 token 的总数
    已经 >= 原始实际 token 数 orig_num_tokens，说明最后一个 ubatch
    不包含任何实际 token（全部是 padding），因此为空。

    这个判断在决定是否需要跳过最后一个 ubatch 的前向计算时使用。
    """
    return (padded_num_tokens // num_ubatches) * (num_ubatches - 1) >= orig_num_tokens


def check_ubatch_thresholds(
    config: ParallelConfig, num_tokens: int, uniform_decode: bool
) -> bool:
    """检查当前批次是否满足启用 ubatch（微批次）的条件。

    判断逻辑（按优先级）：
      1. 如果全局配置未启用 ubatching（config.use_ubatching 为 False），直接返回 False
      2. 如果当前批次是纯解码（uniform_decode=True），则 token 数需 >= dbo_decode_token_threshold
      3. 如果当前批次包含 prefill，则 token 数需 >= dbo_prefill_token_threshold

    为什么需要区分 decode 和 prefill 阈值：
      - 纯 decode 批次每个请求通常只有 1 个 token，token 数 = 请求数，
        阈值设置较小（默认 32），因为切分收益主要来自通信-计算重叠。
      - 包含 prefill 的批次单个请求可能有大量 token，切分的粒度和开销不同，
        因此使用更高的阈值（默认 512），确保只有在 batch 足够大时才启用。
    """
    if not config.use_ubatching:
        return False
    if uniform_decode:
        return num_tokens >= config.dbo_decode_token_threshold
    else:
        return num_tokens >= config.dbo_prefill_token_threshold


# This pads the last ubatch slice out to the total number of tokens
# (num_tokens + padding) since we do `create_ubatch_slices` before applying DP padding.
def _pad_out_ubatch_slices(
    ubatch_slices: UBatchSlices, num_total_tokens: int, num_reqs_padded: int
) -> UBatchSlices:
    """将最后一个 ubatch 的切片扩展到 padding 后的总长度。

    背景：DBO 的切分是在 DP padding 之前计算的，但实际执行时 batch 会经过
    padding 以满足数据并行（DP）各 rank 之间的对齐要求。因此最后一个 ubatch
    的 token_slice 和 request_slice 需要扩展到 padding 后的边界，以覆盖
    所有 padding token。

    步骤：
      1. 取出最后一个 ubatch 切片
      2. 将其 request_slice 的 stop 扩展到 num_reqs_padded（padding 后的请求数）
      3. 将其 token_slice 的 stop 扩展到 num_total_tokens（padding 后的 token 总数）
      4. 返回修改后的 ubatch 切片列表
    """
    last_slice = ubatch_slices[-1]
    padded_last_request_slice = slice(last_slice.request_slice.start, num_reqs_padded)
    padded_last_token_slice = slice(last_slice.token_slice.start, num_total_tokens)

    return ubatch_slices[:-1] + [
        UBatchSlice(padded_last_request_slice, padded_last_token_slice)
    ]


def maybe_create_ubatch_slices(
    should_ubatch: bool,
    num_scheduled_tokens: np.ndarray,
    num_tokens_padded: int,
    num_reqs_padded: int,
    num_ubatches: int,
    split_point: list[int] | int | None = None,
) -> tuple[UBatchSlices | None, UBatchSlices | None]:
    """将一个 batch 切分为多个 ubatch（微批次），返回原始切片和 padding 后的切片。

    这是 DBO 模块最核心的切分函数。它将一个 batch 中的所有 token 按数量
    均匀切分为 num_ubatches 个微批次，同时确定每个微批次覆盖的请求范围。

    参数说明：
      - should_ubatch: 是否应该启用 ubatch 切分（由上游 check_ubatch_thresholds 决定）
      - num_scheduled_tokens: 每个请求本次调度的 token 数（numpy 数组，shape=[num_reqs]）
      - num_tokens_padded: DP padding 后的 token 总数
      - num_reqs_padded: DP padding 后的请求总数
      - num_ubatches: 要切分的微批次数
      - split_point: 自定义切分点（可选），默认按 padded token 数均匀切分

    返回值：
      - (ubatch_slices, ubatch_slices_padded):
        ubatch_slices 是按实际 token 数切分的结果，
        ubatch_slices_padded 是扩展到 padding 边界后的结果。
        如果 should_ubatch 为 False，返回 (None, None)。

    切分算法流程：
      1. 计算每个 ubatch 的 token 数：split_point = num_tokens_padded // num_ubatches
      2. 计算 token 切分点列表：[split_point, 2*split_point, ..., (n-1)*split_point]
      3. 构建 cu_num_tokens（每个请求的 token 累积前缀和），用于快速定位请求边界
      4. 遍历每个切分点，确定该 ubatch 的 token 范围和请求范围：
         - token_slice: [start_token, end_token) -- 全局 token 索引
         - request_slice: 通过 searchsorted 在 cu_num_tokens 中二分查找
           确定包含这些 token 的请求索引范围
      5. 将最后一个 ubatch 的切片扩展到 padding 后的边界
      6. 断言验证：所有 ubatch 的 token 数之和 == padded token 总数

    为什么需要 ubatch_slices 和 ubatch_slices_padded 两个版本：
      - ubatch_slices: 反映实际 token 分布，用于 slot_mapping 等需要精确索引的场景
      - ubatch_slices_padded: 扩展到 padding 边界，用于 CUDA Graph 捕获等需要
        固定 shape 的场景
    """
    if not should_ubatch:
        return None, None

    if split_point is None:
        split_point = int(num_tokens_padded) // num_ubatches

    token_split_points = [split_point * i for i in range(1, num_ubatches)]

    # TODO(lucas): Refactor the gpu_model_runner.py so we can pass
    # in cu_num_tokens directly (i.e. query_start_loc)
    # cu_num_tokens: 每个请求 token 数的累积前缀和，cu_num_tokens[i] 表示
    # 前 i 个请求的 token 总数。用于快速通过 token 全局索引定位所属请求。
    cu_num_tokens = np.zeros(len(num_scheduled_tokens) + 1, dtype=np.int32)
    np.cumsum(num_scheduled_tokens, dtype=np.int32, out=cu_num_tokens[1:])

    ubatch_slices = []
    start_token = 0

    # Add the end point to the split points to make iteration easier
    all_points = token_split_points + [cu_num_tokens[-1]]

    for end_token in all_points:
        token_slice = slice(start_token, end_token)

        # Determine request slices using exclusive stop semantics
        # Ubatch includes requests whose tokens overlap [start_token, end_token)

        # Start at the request that contains the start_token
        # or the request starting exactly at start_token (if on boundary)
        # 使用 searchsorted 在 cu_num_tokens 中二分查找，确定 start_token
        # 属于哪个请求（包含该 token 的请求索引）。
        req_start = int(np.searchsorted(cu_num_tokens, start_token, side="right") - 1)

        # Stop at the request that starts at or after end_token
        # 查找第一个 token 起始位置 >= end_token 的请求作为 stop（不含）。
        req_stop = int(np.searchsorted(cu_num_tokens, end_token, side="left"))

        req_slice = slice(req_start, req_stop)
        ubatch_slices.append(UBatchSlice(req_slice, token_slice))

        start_token = end_token

    ubatch_slices_padded = _pad_out_ubatch_slices(
        ubatch_slices, num_tokens_padded, num_reqs_padded
    )

    assert sum(s.num_tokens for s in ubatch_slices_padded) == num_tokens_padded

    return ubatch_slices, ubatch_slices_padded


def slice_query_start_locs(
    query_start_loc: torch.Tensor,
    request_slice: slice,
) -> torch.Tensor:
    """
    Creates a new query_start_loc that corresponds to the requests in
    request_slice.

    Note: This function creates a new tensor to hold the new query_start_locs.
    This will break cudagraph compatibility.
    """
    # 中文注释：从全局 query_start_loc 中截取 request_slice 对应请求的子序列，
    # 并减去起始偏移量使其从 0 开始。
    # query_start_loc 是长度为 num_reqs+1 的张量，其中 query_start_loc[i] 表示
    # 第 i 个请求的 query token 在展平 query 数组中的起始位置。
    # 截取 [start, stop+1] 是因为 stop+1 对应最后一个请求的结束位置，
    # 减去 query_start_loc[start] 使相对偏移从 0 开始，方便后续独立使用。
    return (
        query_start_loc[request_slice.start : request_slice.stop + 1]
        - query_start_loc[request_slice.start]
    )


def _make_metadata_with_slice(
    ubatch_slice: UBatchSlice, attn_metadata: CommonAttentionMetadata
) -> CommonAttentionMetadata:
    """
    This function creates a new CommonAttentionMetadata that corresponds to
    the requests included in ubatch_slice
    """
    # 中文注释：根据 ubatch_slice 从全局注意力元数据中切分出该 ubatch 对应的子集。
    # 这是 DBO 注意力元数据切分的核心函数，处理逻辑较为复杂，
    # 主要难点在于：一个请求可能被拆分到两个相邻 ubatch 中（跨 ubatch 请求），
    # 此时需要调整 query_start_loc 和 seq_lens 等元数据。
    #
    # 整体流程：
    #   步骤 1: 提取请求切片和 token 切片的基本信息
    #   步骤 2: 判断是否存在跨 ubatch 请求（首尾请求是否被拆分）
    #   步骤 3: 截取并调整 query_start_loc（请求级元数据）
    #   步骤 4: 截取 seq_lens 等序列长度元数据，并处理尾部请求拆分
    #   步骤 5: 计算 max_query_len、max_seq_len 等统计值
    #   步骤 6: 截取 block_table_tensor 和 slot_mapping（token 级元数据）
    #   步骤 7: 组装并返回新的 CommonAttentionMetadata

    assert not ubatch_slice.is_empty(), f"Ubatch slice {ubatch_slice} is empty"

    request_slice = ubatch_slice.request_slice
    token_slice = ubatch_slice.token_slice

    start_locs = attn_metadata.query_start_loc_cpu
    first_req = request_slice.start
    first_tok = token_slice.start
    last_req = request_slice.stop - 1
    last_tok = token_slice.stop - 1

    assert start_locs[first_req] <= first_tok < start_locs[first_req + 1], (
        "Token slice start outside of first request"
    )
    # NOTE: last token can be outside of the last request if we have CG padding.

    # If the request is split across ubatches, we have to adjust the metadata.
    # splits_first_request: The first request in this slice is the continuation of
    #                       a request that started in a previous slice.
    # splits_last_request:  The last request in this slice continues into the
    #                       next slice.
    #
    # 中文注释：判断是否存在跨 ubatch 的请求拆分。
    # - splits_first_request: 当前 ubatch 的第一个请求是从上一个 ubatch 延续过来的，
    #   即 first_tok > 该请求在原始 query_start_loc 中的起始位置。
    #   这意味着该请求的前一部分 token 属于前一个 ubatch，当前 ubatch 只包含其后半段。
    # - splits_last_request: 当前 ubatch 的最后一个请求会延续到下一个 ubatch，
    #   即该请求的最后 token 位置 > last_tok。
    #   这意味着当前 ubatch 只包含该请求的前半段 token。
    #
    # 为什么需要检测拆分：
    #   当请求被拆分时，query_start_loc 和 seq_lens 需要重新计算，
    #   因为它们描述的是"每个请求在本 ubatch 中"的 token 数和起始位置，
    #   而不是原始全局的值。
    splits_first_request = first_tok > start_locs[first_req]
    splits_last_request = last_tok < start_locs[last_req + 1] - 1

    query_start_loc_cpu = slice_query_start_locs(start_locs, request_slice)
    query_start_loc = slice_query_start_locs(
        attn_metadata.query_start_loc, request_slice
    )

    assert len(query_start_loc) >= 2, (
        f"query_start_loc must have at least 2 elements, got {len(query_start_loc)}"
    )

    # 中文注释：处理第一个请求被拆分的情况。
    # 当第一个请求的前几个 token 属于上一个 ubatch 时，需要从 query_start_loc
    # 中减去被跳过的 token 数，使得 query_start_loc 正确反映本 ubatch 中的偏移。
    # tokens_skipped = 本 ubatch 的起始 token - 该请求在全局中的起始 token。
    if splits_first_request:
        tokens_skipped = first_tok - start_locs[first_req]
        query_start_loc[1:] -= tokens_skipped
        query_start_loc_cpu[1:] -= tokens_skipped
    seq_lens = attn_metadata.seq_lens[request_slice]
    # Read raw fields to avoid triggering the deprecated D2H-syncing properties.
    seq_lens_cpu = (
        attn_metadata._seq_lens_cpu[request_slice]
        if attn_metadata._seq_lens_cpu is not None
        else None
    )
    seq_lens_cpu_upper_bound = (
        attn_metadata.seq_lens_cpu_upper_bound[request_slice]
        if attn_metadata.seq_lens_cpu_upper_bound is not None
        else None
    )
    num_computed_tokens_cpu = (
        attn_metadata._num_computed_tokens_cpu[request_slice]
        if attn_metadata._num_computed_tokens_cpu is not None
        else None
    )

    # 中文注释：处理最后一个请求被拆分的情况。
    # 当最后一个请求的后几个 token 属于下一个 ubatch 时，需要：
    #   1. 调整 query_start_loc 的最后一个元素，减去被截断的 token 数
    #   2. 缩小 seq_lens 的最后一个元素，使其只反映本 ubatch 中的序列长度
    # 注意：这里使用原始的 start_locs（而非可能已被修改的 query_start_loc_cpu）
    # 来计算 tokens_skipped，确保计算正确。
    if splits_last_request:
        # NOTE: We use start_locs (the original query_start_loc_cpu) to calculate
        # the tokens skipped because query_start_loc_cpu might have been modified
        # if splits_first_request is True.
        tokens_skipped = start_locs[last_req + 1] - token_slice.stop
        query_start_loc[-1] -= tokens_skipped
        query_start_loc_cpu[-1] -= tokens_skipped

        # Make sure we don't modify the seq_lens tensors
        #  (not cudagraph compatible)
        # 中文注释：clone 是必要的，因为 seq_lens 可能是原始张量的视图（view），
        # 直接修改会影响原始全局元数据。clone 后再修改确保不破坏原始数据。
        seq_lens = seq_lens.clone()
        seq_lens[-1] -= tokens_skipped
        if seq_lens_cpu is not None:
            seq_lens_cpu = seq_lens_cpu.clone()
            seq_lens_cpu[-1] -= tokens_skipped
        if seq_lens_cpu_upper_bound is not None:
            seq_lens_cpu_upper_bound = seq_lens_cpu_upper_bound.clone()
            seq_lens_cpu_upper_bound[-1] -= tokens_skipped

    assert seq_lens_cpu_upper_bound is not None
    # Preserve the max_seq_len override set during CUDA-graph capture so
    # the attention backend selects the correct kernel for SWA layers.
    max_seq_len = max(int(seq_lens_cpu_upper_bound.max()), attn_metadata.max_seq_len)

    num_requests = request_slice.stop - request_slice.start
    num_actual_tokens = token_slice.stop - token_slice.start
    max_query_len = int(
        torch.max(torch.abs(query_start_loc_cpu[1:] - query_start_loc_cpu[:-1])).item()
    )

    # This is to account for the case where we are in a dummy
    # run and query_start_loc_cpu is full of 0s
    if max_query_len == 0:
        max_query_len = attn_metadata.max_query_len

    # 中文注释：截取 block_table 和 slot_mapping。
    # block_table_tensor 按请求索引（每个请求一行），所以用 request_slice 切分。
    # slot_mapping 按 token 索引（每个 token 一个值），所以用 token_slice 切分。
    block_table_tensor = attn_metadata.block_table_tensor[request_slice]
    slot_mapping = attn_metadata.slot_mapping[token_slice]

    return CommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc_cpu,
        seq_lens=seq_lens,
        num_reqs=num_requests,
        num_actual_tokens=num_actual_tokens,
        max_query_len=max_query_len,
        max_seq_len=max_seq_len,
        block_table_tensor=block_table_tensor,
        slot_mapping=slot_mapping,
        seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
        _seq_lens_cpu=seq_lens_cpu,
        _num_computed_tokens_cpu=num_computed_tokens_cpu,
    )


def split_attn_metadata(
    ubatch_slices: list[UBatchSlice],
    common_attn_metadata: CommonAttentionMetadata,
) -> list[CommonAttentionMetadata]:
    """
    Creates a new CommonAttentionMetadata instance that corresponds to the
    requests for each UBatchSlice in ubatch_slices.

    Note: This function does not modify common_attn_metadata
    """
    # 中文注释：将全局注意力元数据按 ubatch 切片列表进行拆分。
    # 对每个 ubatch 调用 _make_metadata_with_slice 生成独立的注意力元数据。
    # 每个返回的 CommonAttentionMetadata 只包含该 ubatch 涉及的请求和 token 信息，
    # 可以独立用于该 ubatch 的模型前向计算和注意力 kernel 调用。
    #
    # 这个函数是 DBO 在模型执行阶段的入口：
    # 模型每层的前向计算会遍历这些拆分后的元数据，逐个 ubatch 执行注意力计算。
    results = []
    for ubatch_slice in ubatch_slices:
        results.append(_make_metadata_with_slice(ubatch_slice, common_attn_metadata))

    return results
