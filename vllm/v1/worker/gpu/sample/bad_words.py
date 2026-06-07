# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
禁用词过滤模块 (Bad Words Filtering Module)

本模块实现了采样时的禁用词 (bad words) 过滤功能。当模型即将生成包含禁用词的
序列时，将对应 token 的 logit 设置为 -inf，从而阻止其被采样。

工作原理：
1. 每个请求可以指定一组禁用词（由 token ID 序列表示）
2. 对于每个待采样的 token，检查其是否会导致已生成的序列形成禁用词
3. 如果匹配，将该 token 的 logit 设为 -inf

数据结构：
- bad_word_token_ids: 将所有禁用词展平存储，形状 [max_num_reqs, MAX_BAD_WORDS_TOTAL_TOKENS]
- bad_word_offsets: 每个禁用词的累积偏移量，用于定位各禁用词在展平数组中的位置
- num_bad_words: 每个请求的禁用词数量

使用 Triton 内核在 GPU 上并行处理所有 token 的禁用词检查。
"""
import numpy as np
import torch

from vllm.sampling_params import SamplingParams
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.buffer_utils import StagedWriteTensor, UvaBackedTensor
from vllm.v1.worker.gpu.states import RequestState

# 每个请求所有禁用词的最大总 token 数
MAX_BAD_WORDS_TOTAL_TOKENS = 1024  # Max total tokens for all bad words per request
# 每个请求的最大禁用词数量
MAX_NUM_BAD_WORDS = 128  # Max number of bad words per request


class BadWordsState:
    """禁用词状态管理器。

    管理每个请求的禁用词数据，包括：
    1. 禁用词的 token ID 序列（展平存储）
    2. 每个禁用词的偏移量（用于在展平数组中定位）
    3. 每个请求的禁用词数量

    使用 StagedWriteTensor 和 UvaBackedTensor 实现高效的 CPU-GPU 数据传输。
    """

    def __init__(self, req_states: RequestState):
        """初始化禁用词状态。

        Args:
            req_states: 请求状态管理器，提供 max_num_reqs 和 device 信息
        """
        self.req_states = req_states
        self.max_num_reqs = req_states.max_num_reqs
        self.device = req_states.device

        # 展平的禁用词 token IDs: [max_num_reqs, MAX_BAD_WORDS_TOTAL_TOKENS]
        self.bad_word_token_ids = StagedWriteTensor(
            (self.max_num_reqs, MAX_BAD_WORDS_TOTAL_TOKENS),
            dtype=torch.int32,
            device=self.device,
        )
        # 禁用词的累积偏移量: [max_num_reqs, MAX_NUM_BAD_WORDS + 1]
        # offsets[i] 和 offsets[i+1] 之间就是第 i 个禁用词的 token 范围
        self.bad_word_offsets = StagedWriteTensor(
            (self.max_num_reqs, MAX_NUM_BAD_WORDS + 1),
            dtype=torch.int32,
            device=self.device,
        )
        # 每个请求的禁用词数量（UVA 内存，CPU 可直接访问）
        self.num_bad_words = UvaBackedTensor(self.max_num_reqs, dtype=torch.int32)

    def add_request(self, req_idx: int, sampling_params: SamplingParams) -> None:
        """添加新请求的禁用词数据。

        将禁用词展平存储，并计算累积偏移量。例如：
        禁用词 = [[1, 2], [3, 4, 5]]
        展平后 = [1, 2, 3, 4, 5]
        偏移量 = [0, 2, 5]

        Args:
            req_idx: 请求在批次中的索引
            sampling_params: 采样参数（包含 bad_words_token_ids）

        Raises:
            ValueError: 如果禁用词数量或总 token 数超过限制
        """
        bad_words_token_ids = sampling_params.bad_words_token_ids
        if not bad_words_token_ids:
            self.num_bad_words.np[req_idx] = 0
            return

        num_bad_words = len(bad_words_token_ids)
        if num_bad_words > MAX_NUM_BAD_WORDS:
            raise ValueError(
                f"Too many bad words: {num_bad_words}. "
                f"The max number is {MAX_NUM_BAD_WORDS}."
            )

        # 展平禁用词并计算偏移量
        flattened_tokens: list[int] = []
        offsets: list[int] = [0]
        for bad_word in bad_words_token_ids:
            flattened_tokens.extend(bad_word)
            offsets.append(len(flattened_tokens))

        if len(flattened_tokens) > MAX_BAD_WORDS_TOTAL_TOKENS:
            raise ValueError(
                f"Too many total bad word tokens: {len(flattened_tokens)}. "
                f"The max is {MAX_BAD_WORDS_TOTAL_TOKENS}."
            )

        # 暂存写入操作（稍后批量应用到 GPU）
        self.bad_word_token_ids.stage_write(req_idx, 0, flattened_tokens)
        self.bad_word_offsets.stage_write(req_idx, 0, offsets)
        self.num_bad_words.np[req_idx] = num_bad_words

    def apply_staged_writes(self) -> None:
        """将暂存的禁用词数据批量应用到 GPU。"""
        self.num_bad_words.copy_to_uva()
        self.bad_word_token_ids.apply_write()
        self.bad_word_offsets.apply_write()

    def apply_bad_words(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
        input_ids: torch.Tensor,
        expanded_local_pos: torch.Tensor,
    ) -> None:
        """应用禁用词过滤到 logits。

        检查当前批次中是否有请求使用禁用词。如果有，则启动 Triton 内核
        将可能导致禁用词的 token 的 logit 设为 -inf。

        Args:
            logits: 输出 logits 张量 [num_tokens, vocab_size]
            expanded_idx_mapping: 扩展的请求索引映射
            idx_mapping_np: 请求索引映射（numpy）
            input_ids: 输入 token IDs
            expanded_local_pos: 扩展的本地位置
        """
        max_num_bad_words = int(self.num_bad_words.np[idx_mapping_np].max())
        if max_num_bad_words == 0:
            # 没有请求使用禁用词，跳过内核启动
            return

        apply_bad_words(
            logits,
            expanded_idx_mapping,
            self.bad_word_token_ids.gpu,
            self.bad_word_offsets.gpu,
            self.num_bad_words.gpu,
            self.req_states.all_token_ids.gpu,
            self.req_states.prompt_len.gpu,
            self.req_states.total_len.gpu,
            input_ids,
            expanded_local_pos,
            max_num_bad_words,
        )


@triton.jit
def _bad_words_kernel(
    logits_ptr,
    logits_stride,
    expanded_idx_mapping_ptr,
    bad_word_token_ids_ptr,
    bad_word_token_ids_stride,
    bad_word_offsets_ptr,
    bad_word_offsets_stride,
    num_bad_words_ptr,
    all_token_ids_ptr,
    all_token_ids_stride,
    prompt_len_ptr,
    total_len_ptr,
    input_ids_ptr,
    expanded_local_pos_ptr,
):
    token_idx = tl.program_id(0)
    bw_idx = tl.program_id(1)

    req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx)
    num_bad_words = tl.load(num_bad_words_ptr + req_state_idx)

    if bw_idx >= num_bad_words:
        return

    pos = tl.load(expanded_local_pos_ptr + token_idx)
    cur_req_first_pos = token_idx - pos

    prompt_len = tl.load(prompt_len_ptr + req_state_idx)
    total_len = tl.load(total_len_ptr + req_state_idx)
    output_len = total_len - prompt_len
    effective_len = output_len + pos

    bd_offsets_base = bad_word_offsets_ptr + req_state_idx * bad_word_offsets_stride
    bd_tokens_base = bad_word_token_ids_ptr + req_state_idx * bad_word_token_ids_stride
    output_base = all_token_ids_ptr + req_state_idx * all_token_ids_stride + prompt_len

    start = tl.load(bd_offsets_base + bw_idx)
    end = tl.load(bd_offsets_base + bw_idx + 1)
    bad_word_len = end - start
    prefix_len = bad_word_len - 1

    if prefix_len > effective_len:
        return

    last_token = tl.load(bd_tokens_base + end - 1)
    match = 1
    for i in range(prefix_len):
        expected = tl.load(bd_tokens_base + start + i)
        actual_pos = effective_len - prefix_len + i

        from_spec_input = actual_pos >= output_len
        if from_spec_input:
            spec_offset = actual_pos - output_len
            actual = tl.load(input_ids_ptr + cur_req_first_pos + spec_offset)
        else:
            actual = tl.load(output_base + actual_pos)

        match = match & (expected == actual)

    if match:
        tl.store(logits_ptr + token_idx * logits_stride + last_token, -float("inf"))


def apply_bad_words(
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    bad_word_token_ids: torch.Tensor,
    bad_word_offsets: torch.Tensor,
    num_bad_words: torch.Tensor,
    all_token_ids: torch.Tensor,
    prompt_len: torch.Tensor,
    total_len: torch.Tensor,
    input_ids: torch.Tensor,
    expanded_local_pos: torch.Tensor,
    max_num_bad_words: int,
) -> None:
    num_tokens = logits.shape[0]
    _bad_words_kernel[(num_tokens, max_num_bad_words)](
        logits,
        logits.stride(0),
        expanded_idx_mapping,
        bad_word_token_ids,
        bad_word_token_ids.stride(0),
        bad_word_offsets,
        bad_word_offsets.stride(0),
        num_bad_words,
        all_token_ids,
        all_token_ids.stride(0),
        prompt_len,
        total_len,
        input_ids,
        expanded_local_pos,
    )
