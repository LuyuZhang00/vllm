# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
投机解码（Speculative Decoding）模块。

本模块实现了 vLLM v1 引擎中的投机解码功能，用于加速大语言模型的推理过程。
投机解码的核心思想是：
1. 使用一个轻量级的"草稿模型"（draft model）快速生成多个候选 token
2. 使用目标模型（target model）并行验证这些候选 token
3. 通过拒绝采样（rejection sampling）保证最终输出分布与目标模型一致

目前支持的投机解码方法：
- EAGLE：使用额外的线性预测头进行自回归草稿生成
- MTP（Multi-Token Prediction）：多 token 预测
"""

import torch

from vllm.config import VllmConfig


def init_speculator(vllm_config: VllmConfig, device: torch.device):
    """
    初始化投机解码器。

    根据配置中的投机解码方法，创建并返回对应的投机解码器实例。

    参数:
        vllm_config (VllmConfig): vLLM 全局配置对象，包含投机解码相关的配置信息。
        device (torch.device): 计算设备（如 'cuda:0'）。

    返回:
        EagleSpeculator: EAGLE 投机解码器实例。

    异常:
        NotImplementedError: 当指定的投机解码方法尚未实现时抛出。

    流程:
        1. 从配置中提取投机解码配置（speculative_config）
        2. 判断是否使用 EAGLE 方法
        3. 根据方法类型创建对应的投机解码器
    """
    speculative_config = vllm_config.speculative_config
    assert speculative_config is not None
    if speculative_config.use_eagle():
        from vllm.v1.worker.gpu.spec_decode.eagle.speculator import EagleSpeculator

        return EagleSpeculator(vllm_config, device)
    raise NotImplementedError(f"{speculative_config.method} is not supported yet.")
