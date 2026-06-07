# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# =============================================================================
# draft_model.py - 基于独立草稿模型(Draft Model)的推测解码 Proposer
#
# 【模块功能概述】
# 本模块实现了推测解码(Speculative Decoding)中"独立草稿模型"策略。
# 核心思想：使用一个比目标模型更小、更快的独立模型（草稿模型）来快速生成
# 多个候选 token，然后由目标模型并行验证这些候选 token 是否正确。
# 如果草稿模型猜对了，就可以一次跳过多个 token 的解码，从而加速推理。
#
# 【与 EAGLE / MTP 的区别】
# - EAGLE 方法：使用一个轻量的 EAGLE head（不完整的小模型），依赖目标模型
#   的 hidden_states 来预测下一个 token。
# - MTP（Multi-Token Prediction）：模型本身在训练时就学会了同时预测多个 token。
# - Draft Model：完全独立的、更小的模型（如用 7B 模型为 70B 模型起草），
#   不共享 embedding 层和 lm_head 层。
#
# 【在推测解码链路中的位置】
#   Scheduler -> Proposer(DraftModelProposer) -> 目标模型验证 -> 接受/拒绝
#
# 【关键设计决策】
# 1. Draft 模型与目标模型的 TP（张量并行）大小必须一致，避免 torch compile
#    缓存冲突。
# 2. Draft 模型不与目标模型共享 embedding 和 lm_head（区别于 EAGLE/MTP）。
# 3. Draft 模型拥有独立的 VllmConfig，包括独立的模型配置和并行配置。
# =============================================================================

import torch
import torch.nn as nn
from typing_extensions import override

from vllm.config import VllmConfig
from vllm.config.utils import replace
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model
from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer

logger = init_logger(__name__)


class DraftModelProposer(SpecDecodeBaseProposer):
    """基于独立草稿模型的推测解码 Proposer。

    【类功能】
    继承自 SpecDecodeBaseProposer，专门用于管理一个独立于目标模型的草稿模型。
    草稿模型是一个更小的 LLM，能够以极低的延迟生成多个候选 token，
    供目标模型验证。

    【核心职责】
    1. 加载并初始化草稿模型（通过 _get_model）
    2. 校验草稿模型与目标模型的兼容性（vocab_size、TP 大小）
    3. 为草稿模型创建独立的 VllmConfig（模型配置、并行配置）
    4. 管理草稿模型的 embedding 和 lm_head（不与目标模型共享）

    【关键参数】
    - vllm_config: 全局配置，包含推测解码配置（speculative_config）
    - device: 计算设备（如 cuda:0）
    - runner: GPUModelRunner 实例，用于获取注意力后端等信息

    【设计说明】
    pass_hidden_states_to_model=False：独立草稿模型不接收目标模型的 hidden_states，
    这是与 EAGLE（pass_hidden_states_to_model=True）的核心区别。
    EAGLE 依赖目标模型的中间表示来预测，而独立草稿模型完全自主运行。
    """
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ):
        # 初始化基类，关键参数 pass_hidden_states_to_model=False 表示
        # 这是一个独立草稿模型，不需要目标模型的 hidden_states
        super().__init__(
            vllm_config=vllm_config,
            device=device,
            pass_hidden_states_to_model=False,
            runner=runner,
        )
        # 校验 1：确保草稿模型和目标模型的词表大小一致
        # 否则 token ID 无法对齐，验证阶段会出错
        self._raise_if_vocab_size_mismatch()
        # 校验 2：确保草稿模型和目标模型的张量并行(TP)大小一致
        # 否则会导致 torch compile 缓存冲突
        self._raise_if_draft_tp_mismatch()

    def _raise_if_vocab_size_mismatch(self):
        """校验草稿模型与目标模型的词表大小是否一致。

        【为什么需要校验】
        推测解码的核心流程是：草稿模型生成候选 token ID -> 目标模型验证。
        如果两者的词表大小不同，同一个 token ID 在两个模型中可能对应不同的词，
        导致验证结果无效。因此必须确保 vocab_size 完全一致。
        """
        self.speculative_config.verify_equal_vocab_size_if_draft_model()

    def _raise_if_draft_tp_mismatch(self):
        """校验草稿模型与目标模型的张量并行(TP)大小是否一致。

        【为什么需要校验】
        当目标模型使用 TP > 1 而草稿模型使用 TP = 1 时，会出现问题：
        所有 TP rank 都会在 rank 0 上编译草稿模型（因为 TP=1），
        导致 torch compile 缓存被覆盖和损坏。
        相关 issue: https://github.com/vllm-project/vllm/pull/5414

        【解决方法】
        强制要求两者 TP 大小一致，避免缓存冲突。
        """
        # Note(Tomas Ruiz) If we run the target model with TP > 1 and
        # the draft model with TP = 1, then the different TP ranks collide.
        # Specifically when all ranks compile the draft model on rank 0
        # (because TP=1), then the torch compile cache is overwritten and corrupted.
        # We need a mechanism like this: https://github.com/vllm-project/vllm/pull/5414
        # To prevent this error, we assert that both TP sizes must be the same.
        spec_cfg = self.speculative_config
        tgt_tp = spec_cfg.target_parallel_config.tensor_parallel_size
        draft_tp = spec_cfg.draft_parallel_config.tensor_parallel_size
        if draft_tp != tgt_tp:
            raise ValueError(
                f"Currently, 'draft_tensor_parallel_size' and 'tensor_parallel_size' "
                f"must be the same. Got {draft_tp} and {tgt_tp}. "
                "Please pass 'draft_tensor_parallel_size' in the speculative_config."
            )

    @override
    def _create_draft_vllm_config(self) -> VllmConfig:
        """为草稿模型创建独立的 VllmConfig 配置。

        【功能说明】
        草稿模型需要独立于目标模型的配置，包括：
        1. quant_config=None：草稿模型通常不做量化（避免额外开销）
        2. parallel_config：使用草稿模型专属的并行配置，但 rank 保持与目标一致
        3. model_config：使用草稿模型专属的模型配置（不同架构、不同大小）

        【调用链路】
        _create_draft_vllm_config -> _get_model -> get_model (加载草稿模型权重)
        """
        # 基类实现会设置 attention backend 等通用配置
        base = super()._create_draft_vllm_config()
        spec = self.speculative_config

        # 用 replace 创建新配置，覆盖草稿模型专属的字段
        return replace(
            base,
            quant_config=None,
            parallel_config=replace(
                spec.draft_parallel_config,
                rank=self.vllm_config.parallel_config.rank,
            ),
            model_config=spec.draft_model_config,
        )

    @override
    def _get_model(self) -> nn.Module:
        """加载并返回草稿模型实例。

        【功能说明】
        1. 创建草稿模型的独立 VllmConfig
        2. 使用 set_model_tag("draft_model") 标记当前加载的是草稿模型
           （用于 torch compile 缓存区分，避免与目标模型的编译缓存冲突）
        3. 调用 get_model 加载模型权重

        【模型标记机制】
        set_model_tag 会在 torch compile 的缓存 key 中加入 "draft_model" 前缀，
        确保草稿模型和目标模型的编译产物不会互相覆盖。

        【返回值】
        草稿模型的 nn.Module 实例，后续由 propose() 方法调用其 forward 生成候选 token。
        """
        from vllm.compilation.backends import set_model_tag

        draft_vllm_config = self._create_draft_vllm_config()
        with set_model_tag("draft_model"):
            model = get_model(
                vllm_config=draft_vllm_config,
                prefix="draft_model",
            )
        return model

    @override
    def _maybe_share_embeddings(self, target_language_model: nn.Module) -> None:
        """草稿模型不与目标模型共享 embedding 层。

        【设计决策】
        独立草稿模型（如用 Llama-7B 为 Llama-70B 起草）拥有自己的 embedding 层，
        与目标模型的 embedding 层完全独立。这与 EAGLE/MTP 不同——
        EAGLE/MTP 通常共享目标模型的 embedding 权重以节省显存。
        """
        # Draft models don't share embeddings with the target model
        pass

    @override
    def _maybe_share_lm_head(self, target_language_model: nn.Module) -> None:
        """草稿模型不与目标模型共享 lm_head（语言模型头）层。

        【设计决策】
        同 embedding 层，独立草稿模型拥有自己的 lm_head。
        lm_head 负责将 hidden_states 映射到词表大小的 logits，
        独立草稿模型的 lm_head 参数与目标模型不同，不能共享。
        """
        # Draft models don't share lm_head with the target model
        pass
