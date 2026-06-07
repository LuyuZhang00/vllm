# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
EAGLE 投机解码提议器（Proposer）模块。

EAGLE（Extrapolation Algorithm for Greater Language-model Efficiency）是一种
投机解码方法，其核心思想是利用一个轻量级的草稿模型（draft model），基于目标模型
的隐藏状态（hidden states）来预测未来的 token 序列。与传统的独立草稿模型不同，
EAGLE 的草稿模型会接收目标模型前一层的隐藏状态作为输入，从而更好地对齐目标
模型的分布，提高草稿 token 的接受率。

本模块定义了 EagleProposer 类，它继承自 SpecDecodeBaseProposer 基类。
由于 EAGLE 的核心逻辑（模型加载、前向计算、多步迭代采样等）全部在基类
SpecDecodeBaseProposer 中实现，EagleProposer 仅需做少量配置覆盖：
  - 设置 pass_hidden_states_to_model=True，表示草稿模型需要接收目标模型的隐藏状态
  - 转发 runner 参数用于关联 GPUModelRunner

相关基类代码位于: vllm/v1/spec_decode/llm_base_proposer.py
"""

import torch

from vllm.config import VllmConfig
from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer


class EagleProposer(SpecDecodeBaseProposer):
    """
    EAGLE 投机解码提议器。

    EAGLE 草稿模型的关键特性：
      1. 接收目标模型的隐藏状态（hidden states）作为输入特征
      2. 通过自回归方式逐步生成草稿 token，每一步都依赖上一步的输出
      3. 共享目标模型的 embedding 层和 LM head，减少额外参数量
      4. 支持多步推测（num_speculative_tokens > 1），每步生成一个草稿 token

    在 V1 引擎中，EAGLE 的完整推理流程：
      1. 目标模型完成一次前向计算，输出 hidden states
      2. EagleProposer.propose() 被调用，将 hidden states 传入草稿模型
      3. 草稿模型自回归生成 num_speculative_tokens 个候选 token
      4. 目标模型一次性验证所有候选 token（并行计算 logits）
      5. 基于验证结果接受或拒绝草稿 token
    """
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ):
        """
        初始化 EAGLE 提议器。

        参数：
          - vllm_config: vLLM 全局配置对象，包含投机解码配置（speculative_config）、
            模型配置、调度器配置等。
          - device: 计算设备（如 torch.device("cuda:0")）。
          - runner: GPUModelRunner 实例的引用，用于关联模型运行器。
            在初始化 attention 后端等操作时需要用到 runner 的上下文。

        EAGLE 的核心区别在于 pass_hidden_states_to_model=True，
        这使得基类在 propose() 方法中会将目标模型的 hidden_states
        作为额外输入传给草稿模型。基类负责：
          - 草稿模型的加载与权重共享（embedding、lm_head）
          - CUDA graph 捕获与管理
          - 多步自回归采样循环
          - attention 元数据构建
          - 位置编码与 slot mapping 更新
        """
        # 中文注释：调用基类构造函数，关键参数 pass_hidden_states_to_model=True
        # 表示 EAGLE 草稿模型需要接收目标模型的隐藏状态作为输入特征。
        # 这是 EAGLE 区别于普通独立草稿模型的核心设计。
        super().__init__(
            vllm_config,
            device,
            pass_hidden_states_to_model=True,
            runner=runner,
        )
