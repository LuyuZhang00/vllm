# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
采样状态管理模块 (Sampling States Module)

本模块管理采样过程中使用的核心参数状态，包括：

1. 温度 (Temperature):
   - temperature > 1: 使分布更平坦（更随机）
   - temperature < 1: 使分布更尖锐（更确定）
   - temperature = 0: 退化为贪心采样

2. Top-K 采样:
   - 仅保留概率最高的 K 个 token
   - top_k = vocab_size: 不做 Top-K 过滤（默认值）

3. Top-P (Nucleus) 采样:
   - 保留累积概率达到 P 的最小 token 集合
   - top_p = 1.0: 不做 Top-P 过滤（默认值）

4. Min-P 采样:
   - 保留概率至少为 max_prob * min_p 的 token
   - min_p = 0.0: 不做过滤（默认值）

5. 随机种子 (Seed):
   - 用于可复现的采样
   - seed = None: 使用随机种子

6. Log 概率 (Logprobs):
   - 控制是否返回 log 概率及返回的数量
   - num_logprobs = -1: 不返回 logprob（默认值）

所有参数使用 UvaBackedTensor 存储，支持高效的 CPU-GPU 数据传输。
"""
import numpy as np
import torch

from vllm.sampling_params import SamplingParams
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
from vllm.v1.worker.gpu.buffer_utils import UvaBackedTensor
from vllm.v1.worker.gpu.sample.gumbel import apply_temperature
from vllm.v1.worker.gpu.sample.min_p import apply_min_p

# 不返回 logprob 的标记值
NO_LOGPROBS = -1
# numpy int64 的最小值和最大值（用于生成随机种子）
_NP_INT64_MIN = np.iinfo(np.int64).min
_NP_INT64_MAX = np.iinfo(np.int64).max


class SamplingStates:
    """采样参数状态管理器。

    管理每个请求的温度、top-k、top-p、min-p、随机种子和 logprob 数量。
    使用 UvaBackedTensor 实现高效的 CPU-GPU 数据传输。
    """

    def __init__(self, max_num_reqs: int, vocab_size: int):
        """初始化采样状态。

        Args:
            max_num_reqs: 最大请求数
            vocab_size: 词表大小
        """
        self.max_num_reqs = max_num_reqs
        self.vocab_size = vocab_size

        # 采样参数（UVA 内存，CPU 可直接访问）
        self.temperature = UvaBackedTensor(max_num_reqs, dtype=torch.float32)
        self.top_k = UvaBackedTensor(max_num_reqs, dtype=torch.int32)
        self.top_p = UvaBackedTensor(max_num_reqs, dtype=torch.float32)
        self.min_p = UvaBackedTensor(max_num_reqs, dtype=torch.float32)
        self.seeds = UvaBackedTensor(max_num_reqs, dtype=torch.int64)

        # 手动初始化 top_k 和 top_p，因为 0 是无效值
        # top_k 默认为 vocab_size（不过滤）
        self.top_k.np.fill(self.vocab_size)
        self.top_k.copy_to_uva()
        # top_p 默认为 1.0（不过滤）
        self.top_p.np.fill(1.0)
        self.top_p.copy_to_uva()

        # 每个请求需要的 logprob 数量
        self.num_logprobs = np.empty(self.max_num_reqs, dtype=np.int32)
        # -1 表示不请求 logprob
        self.num_logprobs.fill(NO_LOGPROBS)

    def add_request(self, req_idx: int, sampling_params: SamplingParams) -> None:
        """添加新请求的采样参数。

        Args:
            req_idx: 请求在批次中的索引
            sampling_params: 采样参数
        """
        self.temperature.np[req_idx] = sampling_params.temperature
        self.top_p.np[req_idx] = sampling_params.top_p
        top_k = sampling_params.top_k
        if top_k <= 0 or top_k > self.vocab_size:
            top_k = self.vocab_size
        self.top_k.np[req_idx] = top_k
        self.min_p.np[req_idx] = sampling_params.min_p

        # 如果没有指定种子，使用随机种子
        seed = sampling_params.seed
        if seed is None:
            seed = np.random.randint(_NP_INT64_MIN, _NP_INT64_MAX)
        self.seeds.np[req_idx] = seed

        num_logprobs = sampling_params.logprobs
        if num_logprobs is None:
            num_logprobs = NO_LOGPROBS
        self.num_logprobs[req_idx] = num_logprobs

    def apply_staged_writes(self) -> None:
        """将暂存的采样参数数据批量应用到 GPU。"""
        self.temperature.copy_to_uva()
        self.top_p.copy_to_uva()
        self.top_k.copy_to_uva()
        self.min_p.copy_to_uva()
        self.seeds.copy_to_uva()

    def apply_temperature(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
    ) -> None:
        """应用温度缩放到 logits。

        仅在有请求需要温度缩放时才启动内核。
        temperature = 0 或 1 时不做缩放。

        Args:
            logits: logits 张量 [num_tokens, vocab_size]
            expanded_idx_mapping: 扩展的请求索引映射
            idx_mapping_np: 请求索引映射（numpy）
        """
        temp_np = self.temperature.np[idx_mapping_np]
        if np.all((temp_np == 0.0) | (temp_np == 1.0)):
            # 没有请求需要温度缩放，跳过内核启动
            return

        apply_temperature(logits, expanded_idx_mapping, self.temperature.gpu)

    def apply_min_p(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
    ) -> None:
        """应用 Min-P 过滤到 logits。

        仅在有请求使用 Min-P 时才启动内核。

        Args:
            logits: logits 张量 [num_tokens, vocab_size]
            expanded_idx_mapping: 扩展的请求索引映射
            idx_mapping_np: 请求索引映射（numpy）
        """
        if np.all(self.min_p.np[idx_mapping_np] == 0.0):
            # 没有请求使用 Min-P，跳过内核启动
            return
        apply_min_p(logits, expanded_idx_mapping, self.min_p.gpu)

    def apply_top_k_top_p(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
    ) -> torch.Tensor:
        """应用 Top-K 和/或 Top-P 筛选到 logits。

        仅在有请求需要 Top-K 或 Top-P 时才执行。

        Args:
            logits: logits 张量 [num_tokens, vocab_size]
            expanded_idx_mapping: 扩展的请求索引映射
            idx_mapping_np: 请求索引映射（numpy）

        Returns:
            筛选后的 logits（可能是同一个张量或新的张量）
        """
        do_top_k = np.any(self.top_k.np[idx_mapping_np] != self.vocab_size)
        do_top_p = np.any(self.top_p.np[idx_mapping_np] != 1.0)
        if not (do_top_k or do_top_p):
            return logits

        top_k = self.top_k.gpu[expanded_idx_mapping] if do_top_k else None
        top_p = self.top_p.gpu[expanded_idx_mapping] if do_top_p else None
        return apply_top_k_top_p(logits, top_k, top_p)

    def max_num_logprobs(self, idx_mapping_np: np.ndarray) -> int:
        """返回当前批次中所有请求请求的最大 logprob 数量。

        Args:
            idx_mapping_np: 请求索引映射（numpy）

        Returns:
            最大 logprob 数量（-1 表示没有请求需要 logprob）
        """
        return int(np.max(self.num_logprobs[idx_mapping_np]))
