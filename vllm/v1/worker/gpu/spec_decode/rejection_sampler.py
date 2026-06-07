# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
拒绝采样器模块。

本模块实现了投机解码中的拒绝采样（Rejection Sampling）算法。

拒绝采样的核心原理：
1. 草稿模型生成候选 token，其分布为 q(x)
2. 目标模型计算相同 token 的概率分布 p(x)
3. 对于每个候选 token，以概率 min(1, p(x)/q(x)) 接受它
4. 第一个被拒绝的 token 从修正分布中重新采样
5. 这保证了最终输出的分布与目标模型完全一致

本模块提供了两种采样模式：
- 标准模式：使用草稿模型的 logits 进行概率比检验
- 合成模式（synthetic）：使用预设的条件接受率进行快速拒绝采样

性能优化：
- 使用 Triton 内核进行高效的 GPU 并行计算
- 分块处理大词汇表，避免内存溢出
- 贪心采样（temperature=0）有专门的快速路径
"""

import torch

from vllm.config import SpeculativeConfig
from vllm.triton_utils import tl, triton
from vllm.v1.outputs import LogprobsTensors
from vllm.v1.spec_decode.utils import unconditional_to_conditional_rates
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.metrics.logits import get_num_nans
from vllm.v1.worker.gpu.sample.logprob import compute_topk_logprobs
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.sample.states import NO_LOGPROBS
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
    rejection_sample,
)


@triton.jit
def _flatten_sampled_kernel(
    # [num_logits]
    flat_sampled_ptr,
    # [num_reqs, num_speculative_steps + 1]
    sampled_ptr,
    sampled_stride,
    # [num_reqs]
    num_sampled_ptr,
    # [num_reqs + 1]
    cu_num_logits_ptr,
):
    """
    将二维的采样结果展平为一维数组的 Triton 内核。

    用于将形状为 [num_reqs, num_speculative_steps + 1] 的采样结果
    展平为一维数组，以便后续计算 logprobs。

    每个程序处理一个请求，将该请求的采样 token 从二维张量
    拷贝到一维展平数组的对应位置。

    参数:
        flat_sampled_ptr: 输出的展平采样结果指针。
        sampled_ptr: 输入的二维采样结果指针。
        sampled_stride: 二维采样结果的行步长。
        num_sampled_ptr: 每个请求实际采样的 token 数量指针。
        cu_num_logits_ptr: 累积 logits 数量的前缀和指针。
    """
    req_idx = tl.program_id(0)
    start_idx = tl.load(cu_num_logits_ptr + req_idx)
    num_sampled = tl.load(num_sampled_ptr + req_idx)
    for i in range(num_sampled):
        token_id = tl.load(sampled_ptr + req_idx * sampled_stride + i)
        tl.store(flat_sampled_ptr + start_idx + i, token_id)


class RejectionSampler:
    """
    拒绝采样器类。

    实现了投机解码中的拒绝采样算法，用于验证草稿模型生成的候选 token。

    主要功能：
    1. 对草稿模型的候选 token 进行拒绝采样验证
    2. 对被拒绝的 token 从修正分布中重新采样
    3. 计算采样后的 logprobs（用于输出）

    属性:
        sampler (Sampler): 基础采样器，提供采样参数应用和状态管理。
        num_speculative_steps (int): 投机解码的步数（即草稿 token 的数量）。
        rejection_sample_method (str): 拒绝采样方法，"standard" 或 "synthetic"。
        synthetic_conditional_rates (torch.Tensor | None): 合成模式下的条件接受率。

    采样流程：
        1. 应用采样参数（temperature, top_k, top_p 等）到目标模型的 logits
        2. 计算块级别的统计信息（最大值、sumexp、argmax）
        3. 对每个候选 token 进行拒绝采样检验
        4. 对第一个被拒绝的 token 从修正分布重新采样
        5. 返回最终的采样结果和数量
    """

    def __init__(
        self,
        sampler: Sampler,
        spec_config: SpeculativeConfig,
        device: torch.device,
    ):
        """
        初始化拒绝采样器。

        参数:
            sampler (Sampler): 基础采样器实例。
            spec_config (SpeculativeConfig): 投机解码配置。
            device (torch.device): 计算设备。
        """
        self.sampler = sampler
        self.num_speculative_steps = spec_config.num_speculative_tokens
        self.rejection_sample_method = spec_config.rejection_sample_method
        self.synthetic_conditional_rates: torch.Tensor | None = None
        if self.rejection_sample_method == "synthetic":
            # 合成模式：使用预设的条件接受率进行快速拒绝采样
            assert spec_config.synthetic_acceptance_rates is not None
            self.synthetic_conditional_rates = torch.tensor(
                unconditional_to_conditional_rates(
                    spec_config.synthetic_acceptance_rates
                ),
                dtype=torch.float32,
                device=device,
            )

    def _get_logprobs_tensors(
        self,
        input_batch: InputBatch,
        sampled: torch.Tensor,
        num_sampled: torch.Tensor,
        logits: torch.Tensor,
    ) -> LogprobsTensors | None:
        """
        计算采样结果的 logprobs 张量。

        将采样结果展平后，计算 top-k logprobs 用于输出。

        参数:
            input_batch (InputBatch): 输入批次信息。
            sampled (torch.Tensor): 采样结果，形状为 [num_reqs, num_speculative_steps + 1]。
            num_sampled (torch.Tensor): 每个请求实际采样的 token 数量。
            logits (torch.Tensor): 模型输出的 logits。

        返回:
            LogprobsTensors | None: logprobs 张量，如果不需要 logprobs 则返回 None。
        """
        max_num_logprobs = self.sampler.sampling_states.max_num_logprobs(
            input_batch.idx_mapping_np
        )
        if max_num_logprobs == NO_LOGPROBS:
            return None

        num_reqs = input_batch.cu_num_logits.shape[0] - 1
        num_logits = logits.shape[0]
        flat_sampled = torch.zeros(
            num_logits, dtype=sampled.dtype, device=sampled.device
        )
        # 使用 Triton 内核将二维采样结果展平
        _flatten_sampled_kernel[(num_reqs,)](
            flat_sampled,
            sampled,
            sampled.stride(0),
            num_sampled,
            input_batch.cu_num_logits,
            num_warps=1,
        )
        expanded_logits = num_logits != input_batch.idx_mapping.shape[0]
        return compute_topk_logprobs(
            logits,
            max_num_logprobs,
            flat_sampled,
            input_batch.cu_num_logits_np.tolist() if expanded_logits else None,
        )

    def __call__(
        self,
        logits: torch.Tensor,
        input_batch: InputBatch,
        draft_logits: torch.Tensor | None = None,
    ) -> SamplerOutput:
        """
        执行拒绝采样。

        这是拒绝采样器的主要入口函数，接收目标模型的 logits 和草稿模型的 logits，
        执行拒绝采样算法，返回最终的采样结果。

        参数:
            logits (torch.Tensor): 目标模型输出的 logits，形状为 [num_logits, vocab_size]。
            input_batch (InputBatch): 输入批次信息，包含草稿 token、位置、采样参数等。
            draft_logits (torch.Tensor | None): 草稿模型输出的 logits，
                形状为 [max_num_reqs, num_speculative_steps, vocab_size]。
                如果为 None，则使用 one-hot 草稿分布。

        返回:
            SamplerOutput: 采样结果，包含采样的 token ID、logprobs、NaN 数量和采样数量。

        流程:
            1. 计算 NaN 数量（在应用采样参数之前）
            2. 从输入批次中提取草稿 token 和位置信息
            3. 应用采样参数（temperature, top_k, top_p, penalties）到 logits
            4. 调用 rejection_sample 函数执行拒绝采样
            5. 计算 logprobs 张量
            6. 返回 SamplerOutput 对象
        """
        # NOTE(woosuk): We intentionally compute num_nans before sampling to make clear
        # that num_nans is computed before applying penalties and temperature.
        num_nans = get_num_nans(logits) if self.sampler.compute_nans else None

        # 提取草稿 token（即 input_ids 在 logits 位置对应的 token）
        draft_sampled = input_batch.input_ids[input_batch.logits_indices]
        pos = input_batch.positions[input_batch.logits_indices]
        # 应用采样参数到 logits（包括 temperature scaling, top_k, top_p, penalties）
        processed_logits = self.sampler.apply_sampling_params(
            logits,
            input_batch.expanded_idx_mapping,
            input_batch.idx_mapping_np,
            pos,
            draft_sampled,
            input_batch.expanded_local_pos,
        )
        # 执行拒绝采样算法
        sampled, num_sampled = rejection_sample(
            processed_logits,
            draft_logits,
            draft_sampled,
            input_batch.cu_num_logits,
            pos,
            input_batch.idx_mapping,
            input_batch.expanded_idx_mapping,
            input_batch.expanded_local_pos,
            self.sampler.sampling_states.temperature.gpu,
            self.sampler.sampling_states.seeds.gpu,
            self.num_speculative_steps,
            self.synthetic_conditional_rates,
            use_fp64=self.sampler.use_fp64_gumbel,
        )
        # 计算 logprobs（根据配置使用处理后的 logits 或原始 logits）
        logprobs_tensors = self._get_logprobs_tensors(
            input_batch,
            sampled,
            num_sampled,
            processed_logits
            if self.sampler.logprobs_mode == "processed_logprobs"
            else logits,
        )

        return SamplerOutput(
            sampled_token_ids=sampled,
            logprobs_tensors=logprobs_tensors,
            num_nans=num_nans,
            num_sampled=num_sampled,
        )
