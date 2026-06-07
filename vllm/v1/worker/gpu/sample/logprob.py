# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Log 概率计算模块 (Log Probability Computation Module)

本模块实现了采样时的 log 概率 (logprob) 计算功能，用于返回每个采样 token
及其 top-K 候选 token 的 log 概率值。

核心算法：
1. 数值稳定的 log-softmax: log_softmax(x) = x - max(x) - log(sum(exp(x - max(x))))
2. 仅计算指定 token 的 log 概率，避免 materialize 完整的 [batch, vocab] 张量
3. 使用 Triton 内核在 GPU 上高效计算

支持两种模式：
1. 标准 top-K 模式：返回采样 token + top-K 个最高概率的 token
2. 自定义 token 模式：返回采样 token + 用户指定的 token（通过 logprob_token_ids）

还支持计算 token 排名（rank），即有多少个 token 的 logit 高于采样 token。
"""
import numpy as np
import torch

from vllm.sampling_params import MAX_LOGPROB_TOKEN_IDS, SamplingParams
from vllm.triton_utils import tl, triton
from vllm.v1.outputs import LogprobsTensors
from vllm.v1.worker.gpu.buffer_utils import StagedWriteTensor, UvaBackedTensor


@triton.jit
def _topk_log_softmax_kernel(
    output_ptr,
    logits_ptr,
    logits_stride,
    topk_ids_ptr,
    topk,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
    PADDED_TOPK: tl.constexpr,
):
    """Top-K log-softmax Triton 内核。

    计算指定 token IDs 的 log-softmax 值，采用数值稳定的实现：
    1. 第一遍：找到最大值 max_val（用于数值稳定性）
    2. 第二遍：计算 log-sum-exp = log(sum(exp(logits - max_val)))
    3. 对指定的 top-K 个 token 计算 log_softmax = logits - max_val - lse

    这种方法避免了 materialize 完整的 [vocab_size] softmax 向量，
    只计算我们关心的 top-K 个 token 的 log 概率。

    Args:
        output_ptr: 输出指针 [batch_size, topk]
        logits_ptr: 输入 logits 指针 [batch_size, vocab_size]
        logits_stride: logits 的行步长
        topk_ids_ptr: 需要计算 logprob 的 token IDs 指针 [batch_size, topk]
        topk: 需要计算的 token 数量
        vocab_size: 词表大小
        BLOCK_SIZE: 每个 block 处理的 vocab 大小（编译时常量）
        PADDED_TOPK: 填充后的 topk 大小（必须是 2 的幂，用于向量化加载）
    """
    req_idx = tl.program_id(0)
    row_ptr = logits_ptr + req_idx * logits_stride

    # 第一遍：找到最大值（用于数值稳定的 softmax 计算）
    max_val = float("-inf")
    for i in range(0, vocab_size, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        logits = tl.load(row_ptr + block, mask=block < vocab_size, other=float("-inf"))
        max_val = tl.max(tl.maximum(logits, max_val))
    max_val = max_val.to(tl.float32)  # type: ignore

    # 第二遍：计算 log-sum-exp
    se = 0.0
    for i in range(0, vocab_size, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        logits = tl.load(row_ptr + block, mask=block < vocab_size, other=0.0)
        # 注意：确保 logits 和所有后续操作使用 FP32
        logits = logits.to(tl.float32)
        e = tl.exp(logits - max_val)
        e = tl.where(block < vocab_size, e, 0.0)
        se += tl.sum(e)
    lse = tl.log(se)

    # 加载需要计算 logprob 的 top-K token IDs
    k_offset = tl.arange(0, PADDED_TOPK)
    k_mask = k_offset < topk
    topk_ids = tl.load(topk_ids_ptr + req_idx * topk + k_offset, mask=k_mask, other=0)

    # 计算这些 token 的 log-softmax 值
    logits = tl.load(row_ptr + topk_ids, mask=k_mask)
    logits = logits.to(tl.float32)
    o = logits - max_val - lse
    tl.store(output_ptr + req_idx * topk + k_offset, o, mask=k_mask)


@triton.jit
def _ranks_kernel(
    output_ptr,
    logits_ptr,
    logits_stride,
    token_ids_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    """Token 排名计算 Triton 内核。

    计算每个请求中采样 token 的排名（rank）。
    排名定义为：有多少个 token 的 logit 大于或等于采样 token 的 logit。
    排名为 1 表示该 token 的 logit 最高（贪心选择）。

    Args:
        output_ptr: 输出排名指针 [batch_size]
        logits_ptr: 输入 logits 指针 [batch_size, vocab_size]
        logits_stride: logits 的行步长
        token_ids_ptr: 采样 token IDs 指针 [batch_size]
        vocab_size: 词表大小
        BLOCK_SIZE: 每个 block 处理的 vocab 大小（编译时常量）
    """
    req_idx = tl.program_id(0)
    row_ptr = logits_ptr + req_idx * logits_stride

    # 加载采样 token 的 logit 值
    token_id = tl.load(token_ids_ptr + req_idx)
    x = tl.load(row_ptr + token_id)

    # 统计有多少个 token 的 logit >= 采样 token 的 logit
    n = 0
    for i in range(0, vocab_size, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        logits = tl.load(row_ptr + block, mask=block < vocab_size, other=float("-inf"))
        n += tl.sum((logits >= x).to(tl.int32))
    tl.store(output_ptr + req_idx, n)


def compute_token_logprobs(
    logits: torch.Tensor, token_ids: torch.Tensor
) -> torch.Tensor:
    """计算指定 token 的 log 概率。

    使用数值稳定的 log-softmax 算法，仅计算指定 token 的 log 概率，
    避免 materialize 完整的 [batch_size, vocab_size] logprobs 张量以节省 GPU 内存。

    Args:
        logits: 输入 logits 张量 [batch_size, vocab_size]
        token_ids: 需要计算 logprob 的 token IDs [batch_size, num_logprobs]

    Returns:
        log 概率张量 [batch_size, num_logprobs]
    """
    # 注意：为了节省 GPU 内存，我们不 materialize 完整的 [batch_size, vocab_size]
    # logprobs 张量。内核每行计算 max + logsumexp，仅在 token_ids 处输出 logprobs。
    batch_size, vocab_size = logits.shape
    token_ids = token_ids.to(torch.int64)
    num_logprobs = token_ids.shape[1]
    logprobs = logits.new_empty((batch_size, num_logprobs), dtype=torch.float32)
    _topk_log_softmax_kernel[(batch_size,)](
        logprobs,
        logits,
        logits.stride(0),
        token_ids,
        num_logprobs,
        vocab_size,
        BLOCK_SIZE=1024,  # type: ignore
        PADDED_TOPK=triton.next_power_of_2(num_logprobs),
    )
    return logprobs


def compute_topk_logprobs(
    logits: torch.Tensor,
    num_logprobs: int,
    sampled_token_ids: torch.Tensor,
    cu_num_logits: list[int] | None = None,
    logprob_token_ids_state: "LogprobTokenIdsState | None" = None,
    expanded_idx_mapping: torch.Tensor | None = None,
    max_per_req_token_ids: int = 0,
) -> LogprobsTensors:
    """计算 top-K log 概率和 token 排名。

    有两种模式：
    1. 快速路径 (max_per_req_token_ids == 0):
       没有请求指定自定义的 logprob_token_ids。
       返回采样 token + top-K 个最高概率的 token 的 log 概率。

    2. 自定义路径 (max_per_req_token_ids > 0):
       有些请求指定了 logprob_token_ids。
       使用 Triton 内核构建 token_ids 矩阵，将 top-K 列替换为自定义 token。

    Args:
        logits: 输入 logits [batch_size, vocab_size]
        num_logprobs: 需要返回的 logprob 数量（top-K）
        sampled_token_ids: 采样的 token IDs [batch_size]
        cu_num_logits: 累积 logit 计数（用于扩展模式）
        logprob_token_ids_state: 自定义 logprob token IDs 状态
        expanded_idx_mapping: 扩展索引映射
        max_per_req_token_ids: 每个请求自定义 token IDs 的最大数量

    Returns:
        LogprobsTensors 对象，包含 logprob_token_ids、logprobs、selected_token_ranks
    """
    assert num_logprobs >= 0
    batch_size, vocab_size = logits.shape

    if max_per_req_token_ids == 0:
        # 快速路径：没有请求指定自定义的 logprob_token_ids
        logprob_token_ids = sampled_token_ids.unsqueeze(-1)
        if num_logprobs > 0:
            topk_indices = torch.topk(logits, num_logprobs, dim=-1).indices
            logprob_token_ids = torch.cat((logprob_token_ids, topk_indices), dim=1)
        logprobs = compute_token_logprobs(logits, logprob_token_ids)
    else:
        # 有些请求指定了 logprob_token_ids。通过单个 Triton 内核在 GPU 上构建
        # [batch_size, 1 + max_cols] 的 token_ids 矩阵和有效性掩码，
        # 将 top-K 列替换为每个请求的自定义 token。
        assert logprob_token_ids_state is not None
        assert expanded_idx_mapping is not None

        if num_logprobs > 0:
            topk_token_ids = torch.topk(logits, num_logprobs, dim=-1).indices
            topk_token_ids = topk_token_ids.to(torch.int32)
        else:
            # 此张量仅用作 int32 指针，数据不会被访问
            topk_token_ids = logprob_token_ids_state.token_ids.gpu

        num_cols = max(num_logprobs, max_per_req_token_ids)
        logprob_token_ids = sampled_token_ids.new_zeros((batch_size, 1 + num_cols))
        valid_mask = torch.zeros_like(logprob_token_ids, dtype=torch.bool)
        _fill_logprob_token_ids_kernel[(batch_size,)](
            logprob_token_ids,
            logprob_token_ids.stride(0),
            valid_mask,
            valid_mask.stride(0),
            sampled_token_ids,
            topk_token_ids,
            topk_token_ids.stride(0),
            expanded_idx_mapping,
            logprob_token_ids_state.num_token_ids.gpu,
            logprob_token_ids_state.token_ids.gpu,
            logprob_token_ids_state.token_ids.gpu.stride(0),
            NUM_TOPK=num_logprobs,
            PADDED_COLS=triton.next_power_of_2(num_cols),
        )
        logprobs = compute_token_logprobs(logits, logprob_token_ids)
        # 将无效位置的 logprobs 设为 -inf
        logprobs = logprobs.masked_fill(~valid_mask, float("-inf"))

    # 计算采样 token 的排名
    token_ranks = torch.empty(batch_size, dtype=torch.int64, device=logits.device)
    _ranks_kernel[(batch_size,)](
        token_ranks,
        logits,
        logits.stride(0),
        sampled_token_ids,
        vocab_size,
        BLOCK_SIZE=8192,  # type: ignore
    )
    return LogprobsTensors(
        logprob_token_ids=logprob_token_ids,
        logprobs=logprobs,
        selected_token_ranks=token_ranks,
        cu_num_generated_tokens=cu_num_logits,
    )


@triton.jit
def _fill_logprob_token_ids_kernel(
    # [batch_size, 1 + num_cols]
    out_token_ids_ptr,
    out_token_ids_stride,
    # [batch_size, 1 + num_cols]
    out_valid_mask_ptr,
    out_valid_mask_stride,
    sampled_token_ids_ptr,  # [batch_size]
    topk_indices_ptr,  # [batch_size, NUM_TOPK] (unused when NUM_TOPK == 0)
    topk_indices_stride,
    expanded_idx_mapping_ptr,  # [batch_size] -> req_state_idx
    num_per_req_token_ids_ptr,  # [max_num_reqs]
    per_req_token_ids_ptr,  # [max_num_reqs, MAX_LOGPROB_TOKEN_IDS]
    per_req_token_ids_stride,
    NUM_TOPK: tl.constexpr,
    PADDED_COLS: tl.constexpr,
):
    """填充 logprob token IDs 和有效性掩码的 Triton 内核。

    构建 [batch_size, 1 + num_cols] 的 token_ids 矩阵：
    - 列 0: 始终是采样 token，始终有效
    - 列 1+: 如果请求指定了自定义 logprob_token_ids，则使用自定义 token；
             否则使用 top-K 个最高概率的 token

    Args:
        out_token_ids_ptr: 输出 token IDs 指针
        out_valid_mask_ptr: 输出有效性掩码指针
        sampled_token_ids_ptr: 采样 token IDs 指针
        topk_indices_ptr: top-K 索引指针
        expanded_idx_mapping_ptr: 扩展索引映射指针
        num_per_req_token_ids_ptr: 每个请求的自定义 token 数量指针
        per_req_token_ids_ptr: 每个请求的自定义 token IDs 指针
        NUM_TOPK: top-K 数量（编译时常量）
        PADDED_COLS: 填充后的列数（编译时常量，必须是 2 的幂）
    """
    batch_idx = tl.program_id(0)

    # 列 0: 始终是采样 token，始终有效
    sampled = tl.load(sampled_token_ids_ptr + batch_idx)
    tl.store(out_token_ids_ptr + batch_idx * out_token_ids_stride, sampled)
    tl.store(out_valid_mask_ptr + batch_idx * out_valid_mask_stride, 1)

    req_state_idx = tl.load(expanded_idx_mapping_ptr + batch_idx)
    num_custom = tl.load(num_per_req_token_ids_ptr + req_state_idx)

    col = tl.arange(0, PADDED_COLS)
    tid_base = out_token_ids_ptr + batch_idx * out_token_ids_stride + 1
    mask_base = out_valid_mask_ptr + batch_idx * out_valid_mask_stride + 1

    if num_custom > 0:
        # 使用请求指定的自定义 token 覆盖 top-K 列
        src = per_req_token_ids_ptr + req_state_idx * per_req_token_ids_stride
        valid = col < num_custom
    else:
        # 使用 top-K 索引（当 NUM_TOPK == 0 时为空操作）
        src = topk_indices_ptr + batch_idx * topk_indices_stride
        valid = col < NUM_TOPK

    tokens = tl.load(src + col, mask=valid, other=0).to(tl.int64)
    tl.store(tid_base + col, tokens, mask=valid)
    tl.store(mask_base + col, tl.full([PADDED_COLS], 1, tl.int1), mask=valid)


class LogprobTokenIdsState:
    """自定义 logprob token IDs 状态管理器。

    允许每个请求指定需要返回 logprob 的特定 token IDs。
    参见 `SamplingParams.logprob_token_ids`。

    例如，用户可以指定只返回 "yes" 和 "no" 两个 token 的 log 概率，
    而不是默认的 top-K 个 token。
    """

    def __init__(self, max_num_reqs: int, device: torch.device):
        """初始化自定义 logprob token IDs 状态。

        Args:
            max_num_reqs: 最大请求数
            device: 计算设备
        """
        self.max_num_reqs = max_num_reqs
        # 每个请求指定的 token 数量
        self.num_token_ids = UvaBackedTensor(max_num_reqs, dtype=torch.int32)
        # 每个请求指定的 token IDs
        self.token_ids = StagedWriteTensor(
            (max_num_reqs, MAX_LOGPROB_TOKEN_IDS),
            dtype=torch.int32,
            device=device,
        )

    def add_request(self, req_idx: int, sampling_params: SamplingParams) -> None:
        """添加新请求的自定义 logprob token IDs。

        Args:
            req_idx: 请求在批次中的索引
            sampling_params: 采样参数

        Raises:
            ValueError: 如果 token 数量超过限制
        """
        token_ids = sampling_params.logprob_token_ids
        if not token_ids:
            self.num_token_ids.np[req_idx] = 0
            return
        n = len(token_ids)
        if n > MAX_LOGPROB_TOKEN_IDS:
            raise ValueError(
                f"Too many logprob_token_ids: {n}. The max is {MAX_LOGPROB_TOKEN_IDS}."
            )
        self.num_token_ids.np[req_idx] = n
        self.token_ids.stage_write(req_idx, 0, token_ids)

    def apply_staged_writes(self) -> None:
        """将暂存的自定义 logprob token IDs 数据批量应用到 GPU。"""
        self.num_token_ids.copy_to_uva()
        self.token_ids.apply_write()

    def max_num_token_ids(self, idx_mapping_np: np.ndarray) -> int:
        """返回当前批次中所有请求指定的最大自定义 token 数量。

        Args:
            idx_mapping_np: 请求索引映射（numpy）

        Returns:
            最大自定义 token 数量
        """
        return int(self.num_token_ids.np[idx_mapping_np].max(initial=0))
