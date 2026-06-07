# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
池化运行器模块 (Pooling Runner Module)

本模块实现了池化模型（如嵌入模型）的推理逻辑。
池化模型将输入文本转换为固定维度的向量表示，用于：
1. 文本嵌入 (embedding): 语义搜索、文本相似度计算
2. 文本分类: 情感分析、主题分类
3. 信息检索: 文档检索、问答系统

当前实现：
- 仅支持解码器模型的 "LAST" 池化策略
- 使用最后一个 token 的隐藏状态作为文本表示
- 对输出进行 L2 归一化

TODO: 支持其他池化策略（如 mean pooling、max pooling）和编码器模型。
"""
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.model_executor.models import VllmModelForPooling, is_pooling_model
from vllm.tasks import PoolingTask
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.states import RequestState


# 注意：当前此类仅支持解码器模型的 "LAST" 池化任务。
# 如何支持其他池化任务和模型类型尚待确定。
class PoolingRunner:
    """池化模型的推理运行器。

    负责执行池化操作，将模型的隐藏状态转换为固定维度的向量表示。
    """

    def __init__(self, model: nn.Module):
        """初始化池化运行器。

        Args:
            model: 池化模型实例
        """
        self.model = cast(VllmModelForPooling, model)

    @staticmethod
    def get_supported_tasks(model: nn.Module) -> list[PoolingTask]:
        """返回模型支持的池化任务。

        Args:
            model: 模型实例

        Returns:
            支持的池化任务列表
        """
        if not is_pooling_model(model):
            return []
        assert "embed" in model.pooler.get_supported_tasks()
        return ["embed"]

    def pool(
        self,
        hidden_states: torch.Tensor,
        input_batch: InputBatch,
        req_states: RequestState,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """执行池化操作。

        当前实现使用 "LAST" 策略：取每个序列最后一个 token 的隐藏状态
        作为整个序列的表示，然后进行 L2 归一化。

        Args:
            hidden_states: 模型输出的隐藏状态
            input_batch: 输入批次数据
            req_states: 请求状态管理器

        Returns:
            (last_hidden_states, is_valid):
            - last_hidden_states: L2 归一化后的隐藏状态
            - is_valid: 每个请求是否有效（序列长度等于 prompt 长度时有效）
        """
        # TODO(woosuk): 支持不同类型的池化任务
        # 取每个序列最后一个 token 的隐藏状态
        last_hidden_states = hidden_states[input_batch.logits_indices]
        # TODO(woosuk): 使归一化可选
        # L2 归一化
        last_hidden_states = F.normalize(last_hidden_states, p=2, dim=-1)

        # 检查每个请求是否有效（序列长度等于 prompt 长度时有效）
        prompt_len = req_states.prompt_len.gpu[input_batch.idx_mapping]
        is_valid = input_batch.seq_lens == prompt_len
        return last_hidden_states, is_valid

    def dummy_pooler_run(self, hidden_states: torch.Tensor) -> None:
        """CUDA Graph 捕获时的虚拟池化操作。

        Args:
            hidden_states: 隐藏状态张量
        """
        F.normalize(hidden_states, p=2, dim=-1)
        return
