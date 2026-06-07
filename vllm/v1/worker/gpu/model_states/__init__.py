# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
模型状态初始化模块 (Model States Initialization Module)

本模块负责根据不同的模型架构，初始化对应的模型状态 (ModelState) 对象。
模型状态封装了每种模型在前向推理过程中所需的各种输入准备工作，包括：
1. 位置编码 (RoPE) 的计算与管理
2. 多模态输入（图像、音频等）的编码与嵌入
3. 注意力元数据 (attention metadata) 的构建
4. 模型特定的输入预处理

根据模型类型，分为三种模型状态实现：
- DefaultModelState: 默认的 Transformer 模型状态（如 LLaMA、Qwen 等）
- MambaHybridModelState: Mamba + Transformer 混合架构模型状态
- WhisperModelState: Whisper 语音转录模型状态
"""
import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache


def init_model_state(
    vllm_config: VllmConfig,
    model: nn.Module,
    encoder_cache: EncoderCache | None,
    device: torch.device,
):
    """根据模型架构类型，初始化并返回对应的 ModelState 实例。

    选择逻辑如下：
    1. 如果是 WhisperForConditionalGeneration 架构 -> WhisperModelState
       （Whisper 使用交叉注意力进行语音转录，需要特殊的编码器输出管理）
    2. 如果是混合架构模型 (is_hybrid，如 Jamba 等 Mamba+Transformer 模型)
       -> MambaHybridModelState（需要处理 Mamba 状态和推测解码等特殊逻辑）
    3. 其他情况 -> DefaultModelState（标准 Transformer 模型的默认实现）

    Args:
        vllm_config: vLLM 全局配置对象
        model: PyTorch 模型实例
        encoder_cache: 多模态编码器缓存，None 表示模型不支持多模态输入
        device: 计算设备（如 GPU）

    Returns:
        对应架构的 ModelState 实例
    """
    # 1. 检查是否为 Whisper 语音转录模型
    if "WhisperForConditionalGeneration" in vllm_config.model_config.architectures:
        from vllm.v1.worker.gpu.model_states.whisper import WhisperModelState

        return WhisperModelState(vllm_config, model, encoder_cache, device)

    # 2. 检查是否为 Mamba 混合架构模型（如 Jamba，同时包含 Mamba 层和 Transformer 层）
    if vllm_config.model_config.is_hybrid:
        from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState

        return MambaHybridModelState(vllm_config, model, encoder_cache, device)

    # 3. 默认情况：标准 Transformer 模型
    from vllm.v1.worker.gpu.model_states.default import DefaultModelState

    return DefaultModelState(vllm_config, model, encoder_cache, device)
