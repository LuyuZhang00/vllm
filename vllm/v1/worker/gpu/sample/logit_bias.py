# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Logit 偏置模块 (Logit Bias Module)

本模块实现了采样时的 logit 偏置功能，包括三种操作：

1. 允许的 token IDs (Allowed Token IDs):
   - 仅允许采样指定的 token，将其他 token 的 logit 设为 -inf
   - 例如：限制输出为数字 0-9

2. Logit 偏置 (Logit Bias):
   - 为特定 token 添加固定的 logit 偏移量
   - 正值增加该 token 被选中的概率，负值降低

3. 最小 token 数 (Min Tokens):
   - 在生成指定数量的 token 之前，阻止停止 token 被采样
   - 通过将停止 token 的 logit 设为 -inf 实现

所有操作通过一个 Triton 内核在 GPU 上并行执行。
"""
import numpy as np
import torch

from vllm.sampling_params import SamplingParams
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.buffer_utils import StagedWriteTensor, UvaBackedTensor

# 每个请求允许的 token ID 最大数量
MAX_NUM_ALLOWED_TOKEN_IDS = 1024
# 每个请求 logit 偏置的最大 token 数量
MAX_NUM_LOGIT_BIAS_TOKENS = 1024
# 每个请求停止 token 的最大数量
MAX_NUM_STOP_TOKEN_IDS = 128


class LogitBiasState:
    """Logit 偏置状态管理器。

    管理每个请求的 logit 偏置数据，包括允许的 token IDs、logit 偏置值
    和最小 token 数相关的停止 token 信息。
    """

    def __init__(self, max_num_reqs: int, device: torch.device):
        """初始化 logit 偏置状态。

        Args:
            max_num_reqs: 最大请求数
            device: 计算设备
        """
        self.max_num_reqs = max_num_reqs

        # 允许的 token IDs（仅这些 token 可以被采样）
        self.num_allowed_token_ids = UvaBackedTensor(
            self.max_num_reqs, dtype=torch.int32
        )
        self.allowed_token_ids = StagedWriteTensor(
            (self.max_num_reqs, MAX_NUM_ALLOWED_TOKEN_IDS),
            dtype=torch.int32,
            device=device,
        )
        # Logit 偏置（为特定 token 添加的 logit 偏移量）
        self.num_logit_bias = UvaBackedTensor(self.max_num_reqs, dtype=torch.int32)
        self.logit_bias_token_ids = StagedWriteTensor(
            (self.max_num_reqs, MAX_NUM_LOGIT_BIAS_TOKENS),
            dtype=torch.int32,
            device=device,
        )
        self.logit_bias = StagedWriteTensor(
            (self.max_num_reqs, MAX_NUM_LOGIT_BIAS_TOKENS),
            dtype=torch.float32,
            device=device,
        )
        # 最小 token 数（prompt_len + min_tokens）
        self.min_lens = UvaBackedTensor(self.max_num_reqs, dtype=torch.int32)
        self.num_stop_token_ids = UvaBackedTensor(self.max_num_reqs, dtype=torch.int32)
        self.stop_token_ids = StagedWriteTensor(
            (self.max_num_reqs, MAX_NUM_STOP_TOKEN_IDS),
            dtype=torch.int32,
            device=device,
        )

        # 标记每个请求是否使用了任何 logit 偏置功能
        self.use_logit_bias = np.zeros(max_num_reqs, dtype=bool)

    def add_request(
        self, req_idx: int, prompt_len: int, sampling_params: SamplingParams
    ) -> None:
        """添加新请求的 logit 偏置数据。

        处理三种偏置操作：
        1. allowed_token_ids: 允许的 token IDs
        2. logit_bias: token 级别的 logit 偏置
        3. min_tokens: 最小生成 token 数（通过停止 token 实现）

        Args:
            req_idx: 请求在批次中的索引
            prompt_len: prompt 的长度
            sampling_params: 采样参数

        Raises:
            ValueError: 如果任何参数超过最大限制
        """
        # 标记是否使用了任何 logit 偏置功能
        use_logit_bias = False

        # 1. 处理允许的 token IDs
        allowed_token_ids = sampling_params.allowed_token_ids
        if allowed_token_ids:
            num_allowed_token_ids = len(allowed_token_ids)
            if num_allowed_token_ids > MAX_NUM_ALLOWED_TOKEN_IDS:
                raise ValueError(
                    f"Too many allowed token IDs: {num_allowed_token_ids}. "
                    f"The max size is {MAX_NUM_ALLOWED_TOKEN_IDS}."
                )
            self.num_allowed_token_ids.np[req_idx] = num_allowed_token_ids
            self.allowed_token_ids.stage_write(req_idx, 0, allowed_token_ids)
            use_logit_bias = True
        else:
            self.num_allowed_token_ids.np[req_idx] = 0

        # 2. 处理 logit 偏置
        logit_bias = sampling_params.logit_bias
        if logit_bias:
            num_logit_bias = len(logit_bias)
            if num_logit_bias > MAX_NUM_LOGIT_BIAS_TOKENS:
                raise ValueError(
                    f"Too many logit bias tokens: {num_logit_bias}. "
                    f"The max size is {MAX_NUM_LOGIT_BIAS_TOKENS}."
                )
            self.num_logit_bias.np[req_idx] = num_logit_bias
            self.logit_bias_token_ids.stage_write(req_idx, 0, logit_bias.keys())
            self.logit_bias.stage_write(req_idx, 0, logit_bias.values())
            use_logit_bias = True
        else:
            self.num_logit_bias.np[req_idx] = 0

        # 3. 处理最小 token 数和停止 token
        min_tokens = sampling_params.min_tokens
        min_len = prompt_len + min_tokens
        self.min_lens.np[req_idx] = min_len
        stop_token_ids = sampling_params.all_stop_token_ids
        if min_tokens > 0 and stop_token_ids:
            num_stop_token_ids = len(stop_token_ids)
            if num_stop_token_ids > MAX_NUM_STOP_TOKEN_IDS:
                raise ValueError(
                    f"Too many stop tokens: {num_stop_token_ids}. "
                    f"The max size is {MAX_NUM_STOP_TOKEN_IDS}."
                )
            self.num_stop_token_ids.np[req_idx] = num_stop_token_ids
            self.stop_token_ids.stage_write(req_idx, 0, stop_token_ids)
            use_logit_bias = True
        else:
            self.num_stop_token_ids.np[req_idx] = 0

        self.use_logit_bias[req_idx] = use_logit_bias

    def apply_staged_writes(self) -> None:
        """将暂存的 logit 偏置数据批量应用到 GPU。"""
        self.num_allowed_token_ids.copy_to_uva()
        self.allowed_token_ids.apply_write()

        self.num_logit_bias.copy_to_uva()
        self.logit_bias_token_ids.apply_write()
        self.logit_bias.apply_write()

        self.min_lens.copy_to_uva()
        self.num_stop_token_ids.copy_to_uva()
        self.stop_token_ids.apply_write()

    def apply_logit_bias(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
        pos: torch.Tensor,
    ) -> None:
        """应用 logit 偏置到 logits。

        检查当前批次中是否有请求使用 logit 偏置。如果有，则启动 Triton 内核
        执行允许的 token IDs 过滤、logit 偏置调整和最小 token 数限制。

        Args:
            logits: 输出 logits 张量 [num_tokens, vocab_size]
            expanded_idx_mapping: 扩展的请求索引映射
            idx_mapping_np: 请求索引映射（numpy）
            pos: 每个 token 的位置
        """
        if not np.any(self.use_logit_bias[idx_mapping_np]):
            # 没有请求使用 logit 偏置，跳过内核启动
            return

        apply_logit_bias(
            logits,
            expanded_idx_mapping,
            pos,
            self.num_allowed_token_ids.gpu,
            self.allowed_token_ids.gpu,
            self.num_logit_bias.gpu,
            self.logit_bias_token_ids.gpu,
            self.logit_bias.gpu,
            self.min_lens.gpu,
            self.num_stop_token_ids.gpu,
            self.stop_token_ids.gpu,
        )


@triton.jit
def _bias_kernel(
    logits_ptr,
    logits_stride,
    vocab_size,
    expanded_idx_mapping_ptr,
    # Allowed token IDs.
    num_allowed_token_ids_ptr,
    allowed_token_ids_ptr,
    allowed_token_ids_stride,
    # Logit bias.
    num_logit_bias_ptr,
    bias_token_ids_ptr,
    bias_token_ids_stride,
    bias_ptr,
    bias_stride,
    # Min tokens.
    pos_ptr,
    min_lens_ptr,
    num_stop_token_ids_ptr,
    stop_token_ids_ptr,
    stop_token_ids_stride,
    BLOCK_SIZE: tl.constexpr,
    LOGITS_BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx)

    block = tl.arange(0, BLOCK_SIZE)

    # Allowed token IDs.
    num_allowed_token_ids = tl.load(num_allowed_token_ids_ptr + req_state_idx)
    if num_allowed_token_ids > 0:
        block = tl.arange(0, BLOCK_SIZE)
        mask = block < num_allowed_token_ids

        # Save logits for allowed token IDs.
        allowed_token_ids = tl.load(
            allowed_token_ids_ptr + req_state_idx * allowed_token_ids_stride + block,
            mask=mask,
        )
        logits = tl.load(
            logits_ptr + token_idx * logits_stride + allowed_token_ids, mask=mask
        )

        # Set logits to -inf for all tokens.
        for i in range(0, vocab_size, LOGITS_BLOCK_SIZE):
            offset = i + tl.arange(0, LOGITS_BLOCK_SIZE)
            tl.store(
                logits_ptr + token_idx * logits_stride + offset,
                -float("inf"),
                mask=offset < vocab_size,
            )

        # Restore logits for allowed token IDs.
        tl.store(
            logits_ptr + token_idx * logits_stride + allowed_token_ids,
            logits,
            mask=mask,
        )

    # Logit bias.
    num_logit_bias = tl.load(num_logit_bias_ptr + req_state_idx)
    if num_logit_bias > 0:
        mask = block < num_logit_bias
        token_ids = tl.load(
            bias_token_ids_ptr + req_state_idx * bias_token_ids_stride + block,
            mask=mask,
        )
        bias = tl.load(bias_ptr + req_state_idx * bias_stride + block, mask=mask)
        logits = tl.load(logits_ptr + token_idx * logits_stride + token_ids, mask=mask)
        logits += bias
        tl.store(logits_ptr + token_idx * logits_stride + token_ids, logits, mask=mask)

    # Apply min tokens.
    num_stop_token_ids = tl.load(num_stop_token_ids_ptr + req_state_idx)
    pos = tl.load(pos_ptr + token_idx)
    min_len = tl.load(min_lens_ptr + req_state_idx)
    if num_stop_token_ids > 0 and pos < min_len:
        mask = block < num_stop_token_ids
        stop_token_ids = tl.load(
            stop_token_ids_ptr + req_state_idx * stop_token_ids_stride + block,
            mask=mask,
        )
        tl.store(
            logits_ptr + token_idx * logits_stride + stop_token_ids,
            -float("inf"),
            mask=mask,
        )


def apply_logit_bias(
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    pos: torch.Tensor,
    num_allowed_token_ids: torch.Tensor,
    allowed_token_ids: torch.Tensor,
    num_logit_bias: torch.Tensor,
    logit_bias_token_ids: torch.Tensor,
    logit_bias: torch.Tensor,
    min_lens: torch.Tensor,
    num_stop_token_ids: torch.Tensor,
    stop_token_ids: torch.Tensor,
) -> None:
    num_tokens, vocab_size = logits.shape
    BLOCK_SIZE = triton.next_power_of_2(
        max(
            allowed_token_ids.shape[-1],
            logit_bias_token_ids.shape[-1],
            stop_token_ids.shape[-1],
        )
    )
    LOGITS_BLOCK_SIZE = 8192
    _bias_kernel[(num_tokens,)](
        logits,
        logits.stride(0),
        vocab_size,
        expanded_idx_mapping,
        num_allowed_token_ids,
        allowed_token_ids,
        allowed_token_ids.stride(0),
        num_logit_bias,
        logit_bias_token_ids,
        logit_bias_token_ids.stride(0),
        logit_bias,
        logit_bias.stride(0),
        pos,
        min_lens,
        num_stop_token_ids,
        stop_token_ids,
        stop_token_ids.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
        LOGITS_BLOCK_SIZE=LOGITS_BLOCK_SIZE,
    )
