# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
EAGLE3 辅助工具模块。

本模块提供了 EAGLE3 变体的辅助函数。

EAGLE3 是 EAGLE 的一个变体，其主要区别在于：
1. 使用目标模型的多个中间层的隐状态（auxiliary hidden states）
   而不仅仅是最后一层的隐状态
2. 通过 combine_hidden_states 方法将多层隐状态组合后传递给草稿头
3. 可以从配置或模型默认值中获取辅助层的索引

本模块的主要功能：
1. 配置目标模型以输出指定层的辅助隐状态
2. 从配置中读取辅助层的索引
3. 使用模型的默认辅助层配置
"""

from typing import cast

import torch.nn as nn

from vllm.config import SpeculativeConfig
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import SupportsEagle3, supports_eagle3

logger = init_logger(__name__)


def set_eagle3_aux_hidden_state_layers(
    model: nn.Module,
    spec_config: SpeculativeConfig,
) -> None:
    """
    设置 EAGLE3 的辅助隐状态层。

    配置目标模型以输出指定层的隐状态，用于 EAGLE3 草稿头的输入。

    参数:
        model (nn.Module): 目标模型实例。
        spec_config (SpeculativeConfig): 投机解码配置。

    流程:
        1. 检查模型是否支持 EAGLE3 接口
        2. 尝试从配置中获取辅助层索引
        3. 如果配置中没有，使用模型的默认辅助层
        4. 调用模型的 set_aux_hidden_state_layers 方法设置辅助层

    异常:
        RuntimeError: 当模型不支持 EAGLE3 接口或传入的是类而非实例时抛出。
    """
    if not supports_eagle3(model):
        raise RuntimeError("Model does not support EAGLE3 interface")
    # mypy may infer the class-level overload for supports_eagle3.
    # Narrow explicitly to the runtime protocol instance.
    if isinstance(model, type):
        raise RuntimeError("Expected model instance for EAGLE3 configuration")
    eagle3_model = cast(SupportsEagle3, model)

    # 优先使用配置中的辅助层索引
    aux_layers = get_eagle3_aux_layers_from_config(spec_config)
    if aux_layers:
        logger.info("Using Eagle3 auxiliary layers from config: %s", aux_layers)
    else:
        # 如果配置中没有，使用模型的默认辅助层
        aux_layers = eagle3_model.get_eagle3_default_aux_hidden_state_layers()
        logger.info("Using Eagle3 auxiliary layers from model: %s", aux_layers)
    eagle3_model.set_aux_hidden_state_layers(aux_layers)


def get_eagle3_aux_layers_from_config(
    spec_config: SpeculativeConfig,
) -> tuple[int, ...] | None:
    """
    从配置中获取 EAGLE3 辅助层索引。

    从草稿模型的 HuggingFace 配置中读取 eagle_aux_hidden_state_layer_ids 字段。

    参数:
        spec_config (SpeculativeConfig): 投机解码配置。

    返回:
        tuple[int, ...] | None: 辅助层索引元组，如果配置中没有则返回 None。
    """
    if not (spec_config and spec_config.draft_model_config):
        return None
    hf_config = spec_config.draft_model_config.hf_config
    if not hasattr(hf_config, "eagle_aux_hidden_state_layer_ids"):
        return None
    layer_ids = hf_config.eagle_aux_hidden_state_layer_ids
    if layer_ids and isinstance(layer_ids, (list, tuple)):
        return tuple(layer_ids)
    return None
