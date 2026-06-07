# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
投机解码（Speculative Decoding）元数据模块。

本模块定义了 SpecDecodeMetadata 数据类，用于封装投机解码过程中
目标模型验证阶段所需的全部元数据。在投机解码中，草稿模型先快速生成
多个候选 token，然后目标模型一次性并行验证这些 token。SpecDecodeMetadata
负责记录：
  1. 草稿 token 的 ID 序列（由草稿模型生成）
  2. 每个请求的草稿 token 数量及其前缀和（用于高效索引）
  3. 目标模型 logits 计算所需的索引信息（哪些位置需要计算 logits）
  4. 采样结果的索引（用于从 logits 中提取最终 token）

典型使用流程：
  - Scheduler 调用草稿模型生成候选 token
  - 将候选 token 信息封装为 SpecDecodeMetadata
  - 传递给 Model Runner，由目标模型进行并行验证
"""
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class SpecDecodeMetadata:
    """
    投机解码元数据数据类。

    封装了目标模型验证草稿 token 所需的全部信息。在 V1 的投机解码流程中，
    Scheduler 会将草稿模型的输出整理成此数据结构，传递给 GPUModelRunner，
    由目标模型一次性前向计算所有草稿 token 的 logits，然后进行验证采样。

    字段说明：
      - draft_token_ids: 所有请求的草稿 token ID 展平后的一维张量。
        例如 batch 中有 3 个请求，分别有 [2, 3, 1] 个草稿 token，
        则此张量长度为 6。
      - num_draft_tokens: 每个请求对应的草稿 token 数量列表。
      - cu_num_draft_tokens: num_draft_tokens 的累积和（cumulative sum），
        用于快速定位某个请求在 draft_token_ids 中的起止范围。
      - cu_num_sampled_tokens: 每个请求最终采样 token 数的累积和，
        包含草稿 token + 1 个 bonus token（即原始位置的 token）。
      - target_logits_indices: 目标模型需要计算 logits 的 token 位置索引，
        对应所有草稿 token 的位置。
      - bonus_logits_indices: 每个请求的 bonus token 在 logits 张量中的索引。
        bonus token 是每条序列中"原始 decode 位置"对应的 token。
      - logits_indices: 合并后的 logits 索引，长度为 num_tokens + batch_size，
        用于从目标模型的完整 logits 输出中提取所需位置的 logits。
    """
    # [num_tokens]
    draft_token_ids: torch.Tensor
    # [batch_size]
    num_draft_tokens: list[int]
    # [batch_size]
    cu_num_draft_tokens: torch.Tensor
    # [batch_size]
    cu_num_sampled_tokens: torch.Tensor
    # [num_tokens]
    target_logits_indices: torch.Tensor
    # [batch_size]
    bonus_logits_indices: torch.Tensor
    # [num_tokens + batch_size]
    logits_indices: torch.Tensor

    def __post_init__(self):
        # 中文注释：计算批次中所有请求的最大草稿长度，用于后续 kernel 的 block size 选择等操作。
        self.max_spec_len = max(self.num_draft_tokens)

    @classmethod
    def make_dummy(
        cls,
        draft_token_ids: list[list[int]],
        device: torch.device,
    ) -> "SpecDecodeMetadata":
        """
        创建一个"占位"用的 SpecDecodeMetadata 实例。

        此方法用于 CUDA graph 捕获等场景，此时需要一个形状正确的元数据对象
        来完成图的录制，但实际 token 值并不重要（可以全为 0）。

        参数：
          - draft_token_ids: 二维列表，每个子列表是对应请求的草稿 token ID。
            例如 [[101, 102], [201, 202, 203]] 表示批次中有 2 个请求，
            分别有 2 和 3 个草稿 token。
          - device: 目标设备（如 cuda:0）。

        返回值：
          填充了正确形状但零值张量的 SpecDecodeMetadata 实例。
        """
        # 中文注释：步骤 1 —— 统计每个请求的草稿 token 数量和总 token 数。
        batch_size = len(draft_token_ids)
        num_draft_tokens = [len(ids) for ids in draft_token_ids]
        # 中文注释：每个请求的最终采样 token 数 = 草稿 token 数 + 1 个 bonus token。
        num_sampled_tokens = [len(ids) + 1 for ids in draft_token_ids]
        # 中文注释：将二维列表展平为一维，便于构建张量。
        flattened_draft_token_ids = sum(draft_token_ids, [])
        num_tokens = len(flattened_draft_token_ids)

        # 中文注释：步骤 2 —— 构建草稿 token ID 张量和累积和张量。
        draft_token_ids_tensor = torch.tensor(
            flattened_draft_token_ids, dtype=torch.int32, device=device
        )
        # 中文注释：cu_num_draft_tokens 是前缀和，用于快速切片某个请求的草稿 token 范围。
        cu_num_draft_tokens = np.cumsum(num_draft_tokens, dtype=np.int32)
        cu_num_draft_tokens_tensor = torch.from_numpy(cu_num_draft_tokens).to(device)
        cu_num_sampled_tokens = np.cumsum(num_sampled_tokens, dtype=np.int32)
        cu_num_sampled_tokens_tensor = torch.from_numpy(cu_num_sampled_tokens).to(
            device
        )

        # 中文注释：步骤 3 —— 创建占位的 logits 索引张量（全零，仅占位）。
        # 实际使用时由 Scheduler 或 Runner 根据真实请求填充。
        target_logits_indices = torch.zeros(
            num_tokens, dtype=torch.int32, device=device
        )
        bonus_logits_indices = torch.zeros(batch_size, dtype=torch.int32, device=device)
        logits_indices = torch.zeros(
            num_tokens + batch_size, dtype=torch.int32, device=device
        )
        return cls(
            draft_token_ids=draft_token_ids_tensor,
            num_draft_tokens=num_draft_tokens,
            cu_num_draft_tokens=cu_num_draft_tokens_tensor,
            cu_num_sampled_tokens=cu_num_sampled_tokens_tensor,
            target_logits_indices=target_logits_indices,
            bonus_logits_indices=bonus_logits_indices,
            logits_indices=logits_indices,
        )
