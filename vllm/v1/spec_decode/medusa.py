# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Medusa 投机解码提议器（Proposer）模块。

Medusa 是一种并行投机解码方法，其核心设计思路是：
  在目标模型的最后一层隐藏状态之上，添加多个独立的"Medusa 头"（head），
  每个头负责预测不同位置的未来 token。与 EAGLE 的自回归逐步生成不同，
  Medusa 的所有头在一次前向计算中同时预测多个位置的 token，实现"并行推测"。

Medusa 的工作流程：
  1. 目标模型完成前向计算，输出最后的隐藏状态（hidden states）
  2. 将隐藏状态输入 Medusa 模型，该模型包含多个并行的预测头
  3. 每个头独立预测一个位置的 token 分布，取 argmax 得到候选 token
  4. 所有头的预测结果组成一棵"推测树"，由目标模型一次性验证

与 EAGLE 的主要区别：
  - Medusa 是并行生成（所有头同时工作），EAGLE 是自回归逐步生成
  - Medusa 不需要目标模型的中间隐藏状态，只需要最终隐藏状态
  - Medusa 的草稿模型结构更简单（多头线性层），参数量更小

注意：MedusaProposer 未继承 SpecDecodeBaseProposer，而是独立实现，
因为其并行推测的模式与 EAGLE 的自回归模式差异较大。
"""

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model
from vllm.model_executor.models.interfaces import is_mixture_of_experts
from vllm.v1.sample.metadata import SamplingMetadata

# Initialize logger
logger = init_logger(__name__)


class MedusaProposer:
    """
    Medusa proposer class for generating token sequences
    using multiple parallel prediction heads.

    # 中文注释：Medusa 投机解码提议器。
    #
    # Medusa 模型由多个并行的预测头组成，每个头负责预测序列中不同偏移位置的
    # token。例如，head_0 预测下一个 token，head_1 预测下下个 token，以此类推。
    # 所有头共享同一个输入（目标模型的隐藏状态），因此只需一次前向计算即可
    # 同时生成多个位置的候选 token。

    属性：
      - vllm_config: 全局配置对象
      - spec_config: 投机解码专用配置
      - model: Medusa 草稿模型实例（由 load_model 加载）
      - hidden_size: 模型隐藏层维度
      - max_num_tokens: 单批次最大 token 数
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        """
        初始化 Medusa 提议器。

        参数：
          - vllm_config: vLLM 全局配置，必须包含 speculative_config
          - device: 计算设备

        注意：MedusaProposer 在初始化时只保存配置，不加载模型权重。
        模型加载需要后续显式调用 load_model()。
        """
        # Save config parameters
        self.vllm_config = vllm_config
        assert vllm_config.speculative_config is not None, (
            "Speculative config must be set"
        )
        self.spec_config = vllm_config.speculative_config
        self.device = device
        # 中文注释：单批次最大 token 数，用于预分配缓冲区
        self.max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        # 中文注释：从草稿模型配置中获取隐藏层维度，用于创建输入张量
        self.hidden_size = self.spec_config.draft_model_config.get_hidden_size()
        self.dtype = vllm_config.model_config.dtype

    def propose(
        self,
        target_hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        slot_mappings: dict[str, torch.Tensor]
        | list[dict[str, torch.Tensor]]
        | None = None,  # unused
    ) -> torch.Tensor:
        """
        生成 Medusa 草稿 token。

        此方法是 Medusa 推理的核心入口。它接收目标模型的隐藏状态，
        通过 Medusa 多头模型一次性预测多个位置的 token。

        参数：
          - target_hidden_states: 目标模型最后一层的隐藏状态，
            形状为 [num_tokens, hidden_size]。
          - sampling_metadata: 采样元数据（当前未使用，预留接口）。
          - slot_mappings: slot 映射（当前未使用，Medusa 不需要 KV cache）。

        返回值：
          草稿 token ID 张量，形状为 [batch_size, num_heads]。
          每行对应一个请求，每列对应一个 Medusa 头预测的 token。
          第 0 列是 head_0 的预测（next token），第 1 列是 head_1 的预测（next next token），依此类推。

        执行流程：
          步骤 1: 将隐藏状态输入 Medusa 模型，得到各头的中间表示（blocks）
          步骤 2: 对中间表示计算 logits（每个头独立的词表分布）
          步骤 3: 对每个头取 argmax，得到最可能的 token ID
          步骤 4: 将各头结果堆叠为 [batch_size, num_heads] 张量返回
        """
        # 中文注释：步骤 1 —— 将目标模型隐藏状态输入 Medusa 模型，
        # 模型内部通过多个并行头分别预测不同位置的 token 表示。
        # blocks 是一个列表，每个元素对应一个头的输出隐藏状态。
        blocks = self.model(target_hidden_states)
        # 中文注释：步骤 2 —— 对每个头的输出计算词表上的 logits 分布。
        # logits 是一个列表，每个元素形状为 [num_tokens, vocab_size]。
        logits = self.model.compute_logits(blocks)

        # 中文注释：步骤 3 —— 对每个头的 logits 取 argmax（贪心解码），
        # 然后在 dim=1 维度堆叠，得到 [batch_size, num_heads] 的草稿 token 张量。
        # 例如，若有 3 个 Medusa 头，每个请求会得到 3 个候选 token。
        # Compute argmax for each Medusa head and stack into a single tensor
        # Shape: [batch_size, num_heads]
        draft_tokens = torch.stack([logit.argmax(dim=-1) for logit in logits], dim=1)

        return draft_tokens

    def load_model(self, target_model: nn.Module) -> None:
        """
        加载 Medusa 草稿模型权重。

        使用 set_model_tag("medusa_head") 标记当前加载的是 Medusa 头模型，
        这会影响编译后端的行为（如 CUDA graph 捕获等）。

        参数：
          - target_model: 目标模型实例（当前未使用，但接口保持一致）。

        注意：
          - Medusa 模型不支持 MoE（混合专家）+ EPLB（专家负载均衡）的组合。
          - Medusa 模型不需要与目标模型共享 embedding，因为它只在隐藏状态上操作。
        """
        from vllm.compilation.backends import set_model_tag

        # 中文注释：使用 medusa_head 标签加载模型，以便编译后端正确识别和处理。
        # get_model 会根据 draft_model_config 自动选择正确的模型架构。
        with set_model_tag("medusa_head"):
            self.model = get_model(
                vllm_config=self.vllm_config,
                model_config=self.spec_config.draft_model_config,
            )
        # 中文注释：Medusa 不支持 MoE + EPLB 组合，因为 Medusa 头是简单的线性层，
        # 不存在专家路由的概念，EPLB 的负载均衡逻辑不适用。
        assert not (
            is_mixture_of_experts(self.model)
            and self.vllm_config.parallel_config.enable_eplb
        ), "EPLB for Medusa is not supported"

    @torch.inference_mode()
    def dummy_run(self, num_tokens: int) -> None:
        """
        执行一次空运行（dummy run），用于初始化 CUDA context 和预热。

        此方法在引擎启动时被调用，目的是：
          1. 触发 CUDA 内核的 JIT 编译（如 Triton kernel）
          2. 预分配 GPU 显存
          3. 验证模型可以正常前向计算

        参数：
          - num_tokens: 本次 dummy run 的 token 数量，用于 set_forward_context
            中的元数据构建。

        注意：使用 set_forward_context 确保 forward 过程中的全局状态
        （如 attention 后端的元数据）被正确设置。
        """
        # 中文注释：创建全零的隐藏状态张量作为 dummy 输入。
        # 形状为 [max_num_tokens, hidden_size]，与真实推理时的输入形状一致。
        hidden_states = torch.zeros(
            (self.max_num_tokens, self.hidden_size),
            dtype=self.dtype,
            device=self.device,
        )
        # 中文注释：在 forward context 下执行模型前向计算。
        # set_forward_context 会设置当前 batch 的全局元数据（如 token 数量），
        # 这些元数据在 attention 计算等操作中会被使用。
        with set_forward_context(None, self.vllm_config, num_tokens=num_tokens):
            self.model(hidden_states)
