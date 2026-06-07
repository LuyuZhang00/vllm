# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
重复惩罚模块 (Repetition Penalties Module)

本模块实现了三种采样惩罚机制，用于减少模型输出的重复性：

1. 重复惩罚 (Repetition Penalty):
   - 对在 prompt 或已生成输出中出现过的 token 施加惩罚
   - 如果 logit > 0，除以 penalty；如果 logit < 0，乘以 penalty
   - penalty > 1: 降低重复 token 的概率
   - penalty = 1: 不惩罚（默认值）

2. 频率惩罚 (Frequency Penalty):
   - 根据 token 在已生成输出中的出现次数施加惩罚
   - logit -= frequency_penalty * count
   - count 是 token 在输出中出现的次数

3. 存在惩罚 (Presence Penalty):
   - 如果 token 在已生成输出中出现过，施加固定的惩罚
   - logit -= presence_penalty (如果 token 出现过)
   - 不考虑出现次数，只考虑是否出现

实现细节：
- 使用位掩码 (prompt_bin_mask) 高效记录 prompt 中出现的 token
- 使用计数数组 (output_bin_counts) 记录输出中每个 token 的出现次数
- 使用 Triton 内核在 GPU 上并行执行惩罚计算
"""
import numpy as np
import torch

from vllm.sampling_params import SamplingParams
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.worker.gpu.buffer_utils import UvaBackedTensor
from vllm.v1.worker.gpu.states import RequestState


class PenaltiesState:
    """惩罚状态管理器。

    管理每个请求的惩罚参数和统计数据（prompt 中的 token 掩码、输出中的 token 计数）。
    """

    def __init__(self, req_states: RequestState):
        """初始化惩罚状态。

        Args:
            req_states: 请求状态管理器，提供 max_num_reqs、vocab_size 和 device 信息
        """
        self.req_states = req_states

        max_num_reqs = req_states.max_num_reqs
        self.vocab_size = req_states.vocab_size
        self.device = req_states.device

        # 惩罚参数
        self.repetition_penalty = UvaBackedTensor(max_num_reqs, dtype=torch.float32)
        self.frequency_penalty = UvaBackedTensor(max_num_reqs, dtype=torch.float32)
        self.presence_penalty = UvaBackedTensor(max_num_reqs, dtype=torch.float32)
        # 标记每个请求是否使用惩罚
        self.use_penalty = np.zeros(max_num_reqs, dtype=bool)

        # 手动初始化重复惩罚，因为 0 是无效值（会导致除零错误）
        self.repetition_penalty.np.fill(1.0)
        self.repetition_penalty.copy_to_uva()

        # 惩罚统计数据：
        # prompt_bin_mask: 位掩码，记录 prompt 中出现的 token
        #   形状 [max_num_reqs, ceil(vocab_size/32)]
        #   每 32 个 token 压缩为一个 int32
        self.prompt_bin_mask = torch.zeros(
            max_num_reqs,
            cdiv(self.vocab_size, 32),
            dtype=torch.int32,
            device=self.device,
        )
        # output_bin_counts: 计数数组，记录输出中每个 token 的出现次数
        #   形状 [max_num_reqs, vocab_size]
        #   注意：此张量很少使用但可能很大，占用 GB 级 GPU 内存
        self.output_bin_counts = torch.zeros(
            max_num_reqs, self.vocab_size, dtype=torch.int32, device=self.device
        )

        # 新添加的需要初始化统计数据的请求列表
        self._new_penalties_reqs: list[int] = []

    def add_request(self, req_idx: int, sampling_params: SamplingParams) -> None:
        """添加新请求的惩罚参数。

        Args:
            req_idx: 请求在批次中的索引
            sampling_params: 采样参数
        """
        self.repetition_penalty.np[req_idx] = sampling_params.repetition_penalty
        self.frequency_penalty.np[req_idx] = sampling_params.frequency_penalty
        self.presence_penalty.np[req_idx] = sampling_params.presence_penalty

        do_penalty = use_penalty(sampling_params)
        self.use_penalty[req_idx] = do_penalty
        if do_penalty:
            self._new_penalties_reqs.append(req_idx)

    def apply_staged_writes(self) -> None:
        """将暂存的惩罚数据批量应用到 GPU。

        对于新添加的请求，需要初始化统计数据：
        1. 使用 bincount 内核统计 prompt 和已 prefill 输出中的 token
        2. 更新 prompt_bin_mask 和 output_bin_counts
        """
        if self._new_penalties_reqs:
            idx_mapping = async_tensor_h2d(
                self._new_penalties_reqs,
                dtype=torch.int32,
                device=self.device,
            )

            prefill_lens = self.req_states.prefill_len.np[self._new_penalties_reqs]
            max_prefill_len = int(prefill_lens.max())
            # 初始化统计数据：统计 prompt 和已 prefill 输出中的 token
            bincount(
                idx_mapping,
                self.req_states.all_token_ids.gpu,
                self.req_states.prompt_len.gpu,
                self.req_states.prefill_len.gpu,
                self.prompt_bin_mask,
                self.output_bin_counts,
                max_prefill_len,
            )
            self._new_penalties_reqs.clear()

        # 将惩罚参数复制到 UVA 内存
        self.repetition_penalty.copy_to_uva()
        self.frequency_penalty.copy_to_uva()
        self.presence_penalty.copy_to_uva()

    def apply_penalties(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
        input_ids: torch.Tensor,
        expanded_local_pos: torch.Tensor,
    ) -> None:
        if not np.any(self.use_penalty[idx_mapping_np]):
            # No request uses penalties. Skip the kernel launch.
            return

        apply_penalties(
            logits,
            expanded_idx_mapping,
            input_ids,
            expanded_local_pos,
            self.repetition_penalty.gpu,
            self.frequency_penalty.gpu,
            self.presence_penalty.gpu,
            self.prompt_bin_mask,
            self.output_bin_counts,
        )


@triton.jit
def _penalties_kernel(
    logits_ptr,
    logits_stride,
    expanded_idx_mapping_ptr,
    token_ids_ptr,
    expanded_local_pos_ptr,
    repetition_penalty_ptr,
    frequency_penalty_ptr,
    presence_penalty_ptr,
    prompt_bin_mask_ptr,
    prompt_bin_mask_stride,
    output_bin_counts_ptr,
    output_bin_counts_stride,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx)
    rep_penalty = tl.load(repetition_penalty_ptr + req_state_idx)
    freq_penalty = tl.load(frequency_penalty_ptr + req_state_idx)
    pres_penalty = tl.load(presence_penalty_ptr + req_state_idx)

    use_rep_penalty = rep_penalty != 1.0
    use_freq_penalty = freq_penalty != 0.0
    use_pres_penalty = pres_penalty != 0.0
    use_penalty = use_rep_penalty or use_freq_penalty or use_pres_penalty
    if not use_penalty:
        # Early return to avoid loading logits.
        return

    block_idx = tl.program_id(1)
    block = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block < vocab_size
    logits = tl.load(logits_ptr + token_idx * logits_stride + block, mask=mask)
    logits = logits.to(tl.float32)

    base_output_counts = tl.load(
        output_bin_counts_ptr + req_state_idx * output_bin_counts_stride + block,
        mask=mask,
        other=0,
    )

    # Accumulate draft token counts from previous positions directly into
    # output_bin_counts (preserves its native tensor layout, avoiding an
    # expensive shared-memory layout conversion after the loop).
    pos = tl.load(expanded_local_pos_ptr + token_idx)
    start_idx = token_idx - pos
    output_bin_counts = base_output_counts
    for prev_pos in tl.range(pos):
        prev_token = tl.load(token_ids_ptr + start_idx + prev_pos + 1)
        token_match = block == prev_token
        output_bin_counts = output_bin_counts + token_match.to(tl.int32)
    output_bin_mask = output_bin_counts > 0

    # Apply repetition penalties.
    if use_rep_penalty:
        packed_block = block_idx * BLOCK_SIZE // 32 + tl.arange(0, BLOCK_SIZE // 32)
        packed_mask = tl.load(
            prompt_bin_mask_ptr + req_state_idx * prompt_bin_mask_stride + packed_block,
            mask=packed_block < tl.cdiv(vocab_size, 32),
            other=0,
        )
        prompt_bin_mask = (packed_mask[:, None] >> (tl.arange(0, 32)[None, :])) & 1
        prompt_bin_mask = prompt_bin_mask.to(tl.int1)
        prompt_bin_mask = prompt_bin_mask.reshape(BLOCK_SIZE)

        # If token appears in prompt or output, apply, otherwise use 1.0 for no-op.
        scale = tl.where(prompt_bin_mask | output_bin_mask, rep_penalty, 1.0)
        # If logits are positive, divide by penalty, otherwise multiply by penalty.
        logits *= tl.where(logits > 0, 1.0 / scale, scale)

    # Apply frequency penalties.
    logits -= freq_penalty * output_bin_counts
    # Apply presence penalties.
    logits -= pres_penalty * output_bin_mask
    # Store back to logits.
    tl.store(logits_ptr + token_idx * logits_stride + block, logits, mask=mask)


def apply_penalties(
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    token_ids: torch.Tensor,
    expanded_local_pos: torch.Tensor,
    repetition_penalty: torch.Tensor,
    frequency_penalty: torch.Tensor,
    presence_penalty: torch.Tensor,
    prompt_bin_mask: torch.Tensor,
    output_bin_counts: torch.Tensor,
) -> None:
    num_tokens, vocab_size = logits.shape
    BLOCK_SIZE = 8192
    num_blocks = triton.cdiv(vocab_size, BLOCK_SIZE)
    _penalties_kernel[(num_tokens, num_blocks)](
        logits,
        logits.stride(0),
        expanded_idx_mapping,
        token_ids,
        expanded_local_pos,
        repetition_penalty,
        frequency_penalty,
        presence_penalty,
        prompt_bin_mask,
        prompt_bin_mask.stride(0),
        output_bin_counts,
        output_bin_counts.stride(0),
        vocab_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )


@triton.jit
def _bincount_kernel(
    expanded_idx_mapping_ptr,
    all_token_ids_ptr,
    all_token_ids_stride,
    prompt_len_ptr,
    prefill_len_ptr,
    prompt_bin_mask_ptr,
    prompt_bin_mask_stride,
    output_bin_counts_ptr,
    output_bin_counts_stride,
    BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx)

    prefill_len = tl.load(prefill_len_ptr + req_state_idx)
    if block_idx * BLOCK_SIZE >= prefill_len:
        return

    prompt_len = tl.load(prompt_len_ptr + req_state_idx)
    block = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    if block_idx * BLOCK_SIZE < prompt_len:
        mask = block < prompt_len
        prompt_tokens = tl.load(
            all_token_ids_ptr + req_state_idx * all_token_ids_stride + block, mask=mask
        )
        idx = prompt_tokens // 32
        bit_idx = prompt_tokens % 32
        bit = tl.full((BLOCK_SIZE,), 1, tl.int32) << bit_idx
        tl.atomic_or(
            prompt_bin_mask_ptr + req_state_idx * prompt_bin_mask_stride + idx,
            bit,
            mask=mask,
        )

    if (block_idx + 1) * BLOCK_SIZE >= prompt_len:
        mask = block < prefill_len
        mask &= block >= prompt_len
        output_tokens = tl.load(
            all_token_ids_ptr + req_state_idx * all_token_ids_stride + block, mask=mask
        )
        tl.atomic_add(
            output_bin_counts_ptr
            + req_state_idx * output_bin_counts_stride
            + output_tokens,
            1,
            mask=mask,
        )


def bincount(
    expanded_idx_mapping: torch.Tensor,
    all_token_ids: torch.Tensor,
    prompt_len: torch.Tensor,
    prefill_len: torch.Tensor,
    prompt_bin_mask: torch.Tensor,
    output_bin_counts: torch.Tensor,
    max_prefill_len: int,
) -> None:
    # Use index_fill_ instead of `tensor[idx] = 0` to avoid sync.
    idx_long = expanded_idx_mapping.long()
    prompt_bin_mask.index_fill_(0, idx_long, 0)
    output_bin_counts.index_fill_(0, idx_long, 0)
    num_tokens = expanded_idx_mapping.shape[0]
    BLOCK_SIZE = 1024
    num_blocks = triton.cdiv(max_prefill_len, BLOCK_SIZE)
    _bincount_kernel[(num_tokens, num_blocks)](
        expanded_idx_mapping,
        all_token_ids,
        all_token_ids.stride(0),
        prompt_len,
        prefill_len,
        prompt_bin_mask,
        prompt_bin_mask.stride(0),
        output_bin_counts,
        output_bin_counts.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
    )


def use_penalty(sampling_params: SamplingParams) -> bool:
    return (
        sampling_params.repetition_penalty != 1.0
        or sampling_params.frequency_penalty != 0.0
        or sampling_params.presence_penalty != 0.0
    )
