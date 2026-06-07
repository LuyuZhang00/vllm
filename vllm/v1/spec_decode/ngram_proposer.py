# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# ============================================================
# 模块说明：N-gram 推测解码提议器 (N-gram Proposer for Speculative Decoding)
# ============================================================
# 本模块实现了基于 N-gram 匹配的推测解码（Speculative Decoding）提议器。
#
# 核心思想：
#   在已生成的 token 序列中，查找与当前序列后缀匹配的最长 N-gram，
#   然后提取该 N-gram 之后紧随的 k 个 token 作为"草稿"（draft tokens）。
#   这些草稿 token 会被提交给验证模型进行并行验证，从而加速推理。
#
#   例如：已生成 "A B C D E A B C"，若 N-gram "A B C" 在位置 0-2 处出现，
#   则其后随的 "D E" 可被提议为草稿 token，期望模型接下来也会生成 "D E"。
#
# 实现要点：
#   - 使用 Numba JIT 编译实现高性能的批量 N-gram 匹配。
#   - 使用 KMP 算法（Knuth-Morris-Pratt）的变体在反转序列上高效查找最长前缀匹配。
#   - 支持多线程并行处理批量请求。
# ============================================================

import os

import numpy as np
import torch
from numba import get_num_threads, jit, njit, prange, set_num_threads

from vllm.config import VllmConfig


class NgramProposer:
    """N-gram 推测解码提议器。
    通过在已生成的 token 序列中查找最长匹配的 N-gram，
    提取其后的 token 作为草稿（draft）供推测解码使用。
    这是一种无需额外模型的轻量级推测解码方法，
    仅依赖已有的上下文 token 历史来预测未来 token。
    """
    def __init__(self, vllm_config: VllmConfig):
        # 流程步骤 1：初始化配置参数
        assert vllm_config.speculative_config is not None
        assert vllm_config.speculative_config.prompt_lookup_min is not None
        assert vllm_config.speculative_config.prompt_lookup_max is not None

        # Minimum length of the n-gram to match.
        # N-gram 匹配的最小长度，避免匹配到过短的无意义片段
        self.min_n = vllm_config.speculative_config.prompt_lookup_min
        # Maximum length of the n-gram to match.
        # N-gram 匹配的最大长度，限制匹配长度以节省内存和计算开销
        self.max_n = vllm_config.speculative_config.prompt_lookup_max
        # Number of tokens follow the match. If there are less than k
        # tokens follow the match, we will return the maximum amount of
        # tokens until the end.
        # 每次提议的最大草稿 token 数量（推测步数）
        self.k = vllm_config.speculative_config.num_speculative_tokens
        # Maximum length of the model.
        # 模型支持的最大序列长度，用于防止草稿 token 超出模型限制
        self.max_model_len = vllm_config.model_config.max_model_len

        # Pre-allocate buffers for numba batch propose.
        # 流程步骤 2：预分配 Numba 批量推理所需的缓冲区
        # valid_ngram_draft: 存储每个请求的草稿 token id，形状为 (最大请求数, 推测步数)
        # valid_ngram_num_drafts: 存储每个请求实际生成的草稿 token 数量
        max_num_seqs = vllm_config.scheduler_config.max_num_seqs
        self.valid_ngram_draft = np.zeros((max_num_seqs, self.k), dtype=np.int32)
        self.valid_ngram_num_drafts = np.zeros((max_num_seqs), dtype=np.int32)

        # Threshold of total number of tokens in the batch to enable
        # multi-threading in numba batch propose.
        # 流程步骤 3：配置 Numba 多线程策略
        # 当批次总 token 数超过此阈值时才启用多线程，小批次用单线程避免线程调度开销
        self.num_tokens_threshold = 8192
        tp_size = vllm_config.parallel_config.tensor_parallel_size
        cpu_count = os.cpu_count()
        # Max number of threads for numba parallel processing.
        if cpu_count:
            # Divide by 2 to use physical cores
            # and not logical cores (hyper-threading).
            # Cap the number of threads to 8 to avoid using too many threads
            # since other components like frontend (incl tokenization)
            # and Structured Outputs also use multiple threads.
            # TODO(ekagra-ranjan): bump up the cap from 1 to 8
            # when TP parallelization for ngram is implemented.
            self.num_numba_thread_available = min(1, (cpu_count // 2))
            # Divide by tp_size to ensure each tensor parallel rank
            # has some threads since all ranks will run this.
            self.num_numba_thread_available //= tp_size
        else:
            self.num_numba_thread_available = 1

        # Trigger Numba JIT compilation for N-gram proposer.
        # This usually takes less than 1 second.
        # 流程步骤 4：预触发 Numba JIT 编译
        # 通过一次虚拟调用让 Numba 提前编译核心函数，避免首次推理时的编译延迟
        self.propose(
            [[]] * 1024,
            np.zeros(1024, dtype=np.int32),
            np.zeros((1024, self.max_model_len), dtype=np.int32),
        )

    def batch_propose(
        self,
        num_requests: int,
        valid_ngram_requests: list,
        num_tokens_no_spec: np.ndarray,
        token_ids_cpu: np.ndarray,
    ) -> list[list[int]]:
        """Batch version of ngram proposer using numba for acceleration.

        Args:
            valid_ngram_requests:
                Set of indices of requests that need ngram proposals.
            num_tokens_no_spec:
                Numpy array of shape (batch_size,) representing the number
                of tokens without speculative tokens for each request.
            token_ids_cpu:
                Numpy array of shape (batch_size, max_model_len)
                representing the token IDs for each request.

        Returns:
            list[list[int]]:
                A list where each element is a list of proposed
                token IDs for the corresponding request.
        """
        # 流程步骤 6：执行批量 N-gram 提议
        # 对所有需要 N-gram 推测的请求进行批量处理，利用 Numba 并行加速
        draft_token_ids: list[list[int]] = []

        # Only run batch propose if there are requests needing ngram proposals.
        # avoid calling numba function with empty list which causes error
        # ValueError: cannot compute fingerprint of empty list
        if num_ngram_requests := len(valid_ngram_requests):
            # 保存当前线程数，处理完成后恢复
            original_num_numba_threads = get_num_threads()
            # Ensure we use at least one thread.
            # If total tokens is small, using multiple threads
            # may slow down due to overhead.
            # 根据批次总 token 数动态决定线程数：
            # - token 总量大时启用多线程（受可用线程数和请求数约束）
            # - token 总量小时使用单线程，避免线程调度开销反而降低性能
            total_tokens = np.sum(num_tokens_no_spec)
            if total_tokens >= self.num_tokens_threshold:
                final_num_threads = max(
                    1, min(self.num_numba_thread_available, num_ngram_requests)
                )
                set_num_threads(final_num_threads)
            else:
                set_num_threads(1)

            # 调用 Numba JIT 编译的核心函数，执行批量 N-gram 匹配和提议
            # 结果写入预分配的 self.valid_ngram_draft 和 self.valid_ngram_num_drafts 缓冲区
            batch_propose_numba(
                valid_ngram_requests,
                num_tokens_no_spec,
                token_ids_cpu,
                self.min_n,
                self.max_n,
                self.max_model_len,
                self.k,
                self.valid_ngram_draft,
                self.valid_ngram_num_drafts,
            )

            # Restore original number of threads.
            set_num_threads(original_num_numba_threads)

        # 流程步骤 7：将批量处理结果从缓冲区收集到 Python 列表
        # 遍历所有请求，从预分配的 numpy 缓冲区中提取每个请求的草稿 token
        for i in range(num_requests):
            if i in valid_ngram_requests and self.valid_ngram_num_drafts[i] > 0:
                draft_token_ids.append(
                    self.valid_ngram_draft[i, : self.valid_ngram_num_drafts[i]].tolist()
                )
            else:
                draft_token_ids.append([])

        return draft_token_ids

    def propose(
        self,
        sampled_token_ids: list[list[int]],
        num_tokens_no_spec: np.ndarray,
        token_ids_cpu: np.ndarray,
        slot_mappings: dict[str, torch.Tensor]
        | list[dict[str, torch.Tensor]]
        | None = None,  # unused
    ) -> list[list[int]]:
        """对当前批次中所有请求生成 N-gram 草稿 token 提议。

        流程步骤 5：筛选需要 N-gram 推测的请求
        遍历所有请求，跳过不满足条件的请求（未采样到新 token 或已达到最大长度）。

        Args:
            sampled_token_ids: 上一轮采样得到的 token id 列表（每个请求一个子列表）
            num_tokens_no_spec: 每个请求当前已有的 token 数（不含推测 token）
            token_ids_cpu: 所有请求的完整 token id 数组，形状为 (batch_size, max_model_len)
            slot_mappings: 未使用，仅为接口兼容性保留

        Returns:
            每个请求对应的草稿 token id 列表
        """
        # find which requests need ngram proposals
        valid_ngram_requests = []
        for i, sampled_ids in enumerate(sampled_token_ids):
            num_sampled_ids = len(sampled_ids)
            if not num_sampled_ids:
                # Skip speculative decoding.
                # 没有新采样 token 的请求跳过（可能被抢占或已完成）
                continue

            num_tokens = num_tokens_no_spec[i]
            if num_tokens >= self.max_model_len:
                # Skip requests that have already reached the max model length.
                # 已达到模型最大长度的请求不再生成草稿
                continue

            valid_ngram_requests.append(i)

        # 对筛选出的有效请求执行批量 N-gram 提议
        draft_token_ids = self.batch_propose(
            len(sampled_token_ids),
            valid_ngram_requests,
            num_tokens_no_spec,
            token_ids_cpu,
        )

        return draft_token_ids

    def load_model(self, *args, **kwargs):
        # No model to load.
        # N-gram 提议器不依赖任何模型权重，因此无需加载模型
        pass


@njit(parallel=True)
def batch_propose_numba(
    valid_ngram_requests: list,
    num_tokens_no_spec: np.ndarray,
    token_ids_cpu: np.ndarray,
    min_n: int,
    max_n: int,
    max_model_len: int,
    k: int,
    valid_ngram_draft: np.ndarray,
    valid_ngram_num_drafts: np.ndarray,
):
    """Numba JIT 编译的批量 N-gram 提议核心函数。
    使用 prange 实现请求级并行，每个请求独立执行 N-gram 匹配和草稿生成。

    Args:
        valid_ngram_requests: 需要进行 N-gram 推测的请求索引列表
        num_tokens_no_spec: 每个请求当前已有的 token 数（不含推测 token）
        token_ids_cpu: 所有请求的完整 token id 数组
        min_n: N-gram 匹配的最小长度
        max_n: N-gram 匹配的最大长度
        max_model_len: 模型最大序列长度
        k: 每次提议的最大草稿 token 数
        valid_ngram_draft: 输出缓冲区，存储每个请求的草稿 token id
        valid_ngram_num_drafts: 输出缓冲区，存储每个请求的实际草稿数量
    """
    # 使用 prange 实现并行循环，每个请求的 N-gram 匹配互不依赖
    for i in prange(len(valid_ngram_requests)):
        idx = valid_ngram_requests[i]
        num_tokens = num_tokens_no_spec[idx]
        # 截取当前请求已有的 token 序列作为上下文
        context_token_ids = token_ids_cpu[idx, :num_tokens]
        # 调用核心匹配函数：查找最长匹配 N-gram 并提取后续 token 作为草稿
        drafter_output = _find_longest_matched_ngram_and_propose_tokens(
            origin_tokens=context_token_ids,
            min_ngram=min_n,
            max_ngram=max_n,
            max_model_len=max_model_len,
            k=k,
        )

        # 将结果写入预分配的缓冲区
        valid_ngram_num_drafts[idx] = drafter_output.shape[0]
        if len(drafter_output):
            valid_ngram_draft[idx, : drafter_output.shape[0]] = drafter_output


@jit(nopython=True)
def _find_longest_matched_ngram_and_propose_tokens(
    origin_tokens: np.ndarray,
    min_ngram: int,
    max_ngram: int,
    max_model_len: int,
    k: int,
) -> np.ndarray:
    """
    Find the longest n-gram which matches the suffix of the given tokens
    whose length is within [min_ngram, max_ngram] (inclusive).

    If found, we will extract k right after the matched ngram.
    """
    # 核心算法：在 token 序列中查找与后缀匹配的最长 N-gram，提取其后随 token 作为草稿。
    #
    # 算法流程：
    #   1. 将 token 序列反转，将"查找后缀匹配"转化为"查找前缀匹配"问题
    #   2. 使用 KMP 风格的最长前缀后缀（LPS）数组在线性时间内完成匹配
    #   3. 在反转序列上找到最早的匹配位置（即原始序列中最早的匹配位置）
    #   4. 从匹配位置之后提取 k 个 token 作为草稿提议
    #
    # 示例：origin_tokens = [A, B, C, D, E, A, B, C]
    #   反转后：[C, B, A, E, D, C, B, A]
    #   最长前缀匹配 "C B A"（长度 3），对应原始序列中位置 0-2 的 "A B C"
    #   从匹配位置之后提取 "D E" 作为草稿

    # Do not generate draft tokens is context is shorter than minimum n-gram
    total_token = origin_tokens.shape[0]
    if total_token < min_ngram:
        return np.empty((0,), dtype=origin_tokens.dtype)

    # Do not generate draft tokens beyond the max model length.
    # 限制草稿数量，防止超出模型最大长度
    k = min(k, max_model_len - total_token)
    if k <= 0:
        return np.empty((0,), dtype=origin_tokens.dtype)

    # Flip tokens, and the goal become to find longest ngram
    # on the rightmost position which matches the prefix with
    # length [min_n, max_n] (inclusive).
    # 反转 token 序列，将"查找后缀匹配"转化为"查找前缀匹配"
    tokens = origin_tokens[::-1]

    # Longest prefix (not including itself) which is a suffix of
    # the current position.
    #   lps[i] = max{v, where tokens[0:v] == tokens[i+1-v:i+1]}
    #
    # As ngram is capped by max_ngram to save memory, we only need to
    # store lps for the first max_ngram prefix.
    # LPS 数组（Longest Proper Prefix which is also Suffix）：
    # 这是 KMP 算法的核心数据结构。lps[i] 表示 tokens[0:i+1] 的
    # 最长相等前后缀长度。由于我们只关心长度不超过 max_ngram 的匹配，
    # 只需存储前 max_ngram 个位置的 LPS 值。
    lps = np.zeros(max_ngram, dtype=np.int32)

    # longest_ngram: 目前找到的最长有效匹配长度
    # position: 该匹配在反转序列中的结束位置
    longest_ngram = 0
    position = 0

    # lps[0] always equal to 0, we start with index 1
    # 使用 KMP 风格的线性扫描算法遍历反转后的 token 序列
    # prev_lps: 当前位置的最长前缀后缀长度（类似 KMP 中的 j 指针）
    prev_lps = 0
    i = 1
    while i < total_token:
        # tokens[:prev_lps] is the longest prefix as a suffix of tokens[:i]
        if tokens[prev_lps] == tokens[i]:
            # Token match: tokens[:prev_lps+1] is the longest prefix as
            # a suffix of tokens[:i+1]
            # 当前字符匹配成功，扩展前缀长度
            prev_lps += 1
            # Check if we found a longer valid ngram.
            #
            # Update position when longest_ngram matched prev_lps,
            # as we want to get the target n-gram of the earliest position
            # in the original tokens (i.e.
            # latest position in the reversed tokens)
            # 当匹配长度 >= 当前最长匹配时更新记录
            # 注意：这里用 >= 而非 >，是为了在相同长度时取更晚的位置
            # （反转序列中更晚 = 原始序列中更早），使得提议的 token 在原始序列中位置最早
            if prev_lps >= longest_ngram:
                longest_ngram = prev_lps
                position = i
            if i < max_ngram:
                # Store LPS for the first max_ngram prefix
                lps[i] = prev_lps
            if prev_lps == max_ngram:
                # When prev_lps reached max_ngram, update prev_lps
                # to lps[max_ngram-1] to avoid matching ngram
                # longer than max_ngram
                # 匹配长度已达上限 max_ngram，回退以避免超长匹配
                prev_lps = lps[max_ngram - 1]
            i += 1
        elif prev_lps != 0:
            # Token mismatch: try the second-longest prefix
            # among all suffix of tokens[:i],
            # which is the longest prefix of tokens[:prev_lps]
            # 字符不匹配，利用 LPS 数组回退到次长的前缀后缀位置
            # 这是 KMP 算法避免暴力回溯的关键优化
            prev_lps = lps[prev_lps - 1]
        else:
            # Token mismatch, and no more prefix (except empty string)
            # as a suffix of tokens[:i]
            # 字符不匹配且无更短的前缀后缀可回退，继续前进
            i += 1

    if longest_ngram < min_ngram:
        # No valid ngram is found
        # 未找到满足最小长度要求的匹配，返回空数组
        return np.empty((0,), dtype=origin_tokens.dtype)

    # Flip the position back, so in origin_tokens,
    # origin_tokens[total_token-1-position:total_token-1-position+longest_ngram]
    # is the matched ngram, so we should start drafting tokens from
    # total_token-1-position+longest_ngram
    # 将反转序列中的位置映射回原始序列：
    # - 反转序列中位置 [position - longest_ngram + 1, position] 对应匹配的 N-gram
    # - 在原始序列中，该 N-gram 的结束位置为 total_token - 1 - position
    # - 草稿 token 从匹配 N-gram 之后开始提取
    start_position = total_token - 1 - position + longest_ngram
    # 再次限制草稿数量，防止超出原始序列边界
    k = min(k, total_token - start_position)
    return origin_tokens[start_position : start_position + k]
