# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Selector for MLA prefill backends.

This module provides functions for selecting the appropriate MLA prefill
backend based on device capabilities and configuration.
"""
# MLA prefill 后端选择器模块。
#
# 本模块负责根据设备能力和模型配置自动选择最佳的 MLA prefill 后端。
# 选择流程如下：
#
# 1. 检查用户是否通过 AttentionConfig 显式指定了后端
#    - 如果指定了，验证其有效性并直接使用
#    - 如果无效，抛出 ValueError 异常
#
# 2. 如果用户未指定，进入自动选择流程：
#    a. 获取当前设备的计算能力
#    b. 根据计算能力确定后端优先级列表
#    c. 按优先级逐一尝试，验证每个后端的配置兼容性
#    d. 返回第一个满足所有条件的后端
#
# 3. 如果没有任何后端可用，抛出 ValueError 异常
#
# 后端优先级（Blackwell SM100）：
#   FLASH_ATTN > TRTLLM_RAGGED > FLASHINFER > TOKENSPEED_MLA
#
# 后端优先级（Hopper SM90 及更早）：
#   FLASH_ATTN

from functools import cache
from typing import TYPE_CHECKING, NamedTuple

import torch

from vllm.logger import init_logger
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backends.mla.prefill.registry import MLAPrefillBackendEnum

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.attention.backends.mla.prefill.base import MLAPrefillBackend

logger = init_logger(__name__)


class MLAPrefillSelectorConfig(NamedTuple):
    """Hashable configuration for MLA prefill backend selection.

    This is analogous to AttentionSelectorConfig and contains model-specific
    configuration needed to select an MLA prefill backend, extracted from
    VllmConfig into a hashable form for caching.
    """
    # MLA prefill 后端选择的可哈希配置。
    # 类似于 AttentionSelectorConfig，包含选择后端所需的模型特定配置，
    # 从 VllmConfig 中提取为可哈希形式以支持缓存。

    # 模型使用的数据类型（如 torch.float16, torch.bfloat16）
    dtype: torch.dtype
    # 模型是否具有 DeepSeek R1 兼容的 MLA 维度
    is_r1_compatible: bool


def is_deepseek_r1_mla_compatible(vllm_config: "VllmConfig") -> bool:
    """Check if model has DeepSeek R1 compatible MLA dimensions.
    检查模型是否具有 DeepSeek R1 兼容的 MLA 维度。

    DeepSeek R1 MLA 维度要求：
    - qk_nope_head_dim = 128  （Q/K 中不带 RoPE 的头维度）
    - qk_rope_head_dim = 64   （Q/K 中带 RoPE 的头维度）
    - v_head_dim = 128         （V 的头维度）

    Args:
        vllm_config: vLLM 全局配置

    Returns:
        模型是否具有 R1 兼容的 MLA 维度
    """
    if vllm_config.model_config is None:
        return False
    hf_text_config = vllm_config.model_config.hf_text_config
    qk_nope_head_dim = getattr(hf_text_config, "qk_nope_head_dim", 1)
    qk_rope_head_dim = getattr(hf_text_config, "qk_rope_head_dim", 1)
    v_head_dim = getattr(hf_text_config, "v_head_dim", 1)
    return qk_nope_head_dim == 128 and qk_rope_head_dim == 64 and v_head_dim == 128


def _get_mla_prefill_backend_priorities(
    device_capability: DeviceCapability,
) -> list[MLAPrefillBackendEnum]:
    """Get MLA prefill backend priorities based on device capability.
    根据设备计算能力获取 MLA prefill 后端的优先级列表。

    不同 GPU 架构支持不同的后端：
    - Blackwell (SM100): 支持所有后端，FlashAttn 优先
    - Hopper (SM90) 及更早: 仅支持 FlashAttn

    Args:
        device_capability: 设备的计算能力

    Returns:
        按优先级排序的后端枚举列表（最高优先级在前）
    """
    if device_capability.major == 10:  # Blackwell
        return [
            MLAPrefillBackendEnum.FLASH_ATTN,
            MLAPrefillBackendEnum.TRTLLM_RAGGED,
            MLAPrefillBackendEnum.FLASHINFER,
            MLAPrefillBackendEnum.TOKENSPEED_MLA,
        ]
    else:  # Hopper (SM90) and older
        return [
            MLAPrefillBackendEnum.FLASH_ATTN,
        ]


def get_mla_prefill_backend(
    vllm_config: "VllmConfig",
) -> "type[MLAPrefillBackend]":
    """Select the MLA prefill backend based on configuration and device.
    根据配置和设备选择 MLA prefill 后端。

    选择流程：
    1. 获取当前设备的计算能力
    2. 构建 MLAPrefillSelectorConfig（包含 dtype 和 R1 兼容性信息）
    3. 如果用户显式指定了后端（attention_config.mla_prefill_backend），
       验证其有效性后直接返回
    4. 否则调用 _auto_select_mla_prefill_backend 进行自动选择

    Args:
        vllm_config: vLLM 全局配置

    Returns:
        选定的 prefill 后端类

    Raises:
        ValueError: 如果用户指定的后端无效，或没有任何后端可用
    """
    from vllm.platforms import current_platform

    device_capability = current_platform.get_device_capability()
    if device_capability is None:
        logger.info_once(
            "Device capability not available, using FlashAttention MLA prefill backend."
        )
        return MLAPrefillBackendEnum.FLASH_ATTN.get_class()

    attention_config = vllm_config.attention_config

    selector_config = MLAPrefillSelectorConfig(
        dtype=vllm_config.model_config.dtype,
        is_r1_compatible=is_deepseek_r1_mla_compatible(vllm_config),
    )

    # 如果用户显式指定了后端
    if attention_config.mla_prefill_backend is not None:
        selected_backend = attention_config.mla_prefill_backend
        backend_cls: type[MLAPrefillBackend] | None = None
        try:
            backend_cls = selected_backend.get_class()
            invalid_reasons = backend_cls.validate_configuration(
                device_capability, selector_config
            )
        except ImportError:
            invalid_reasons = ["ImportError"]
        if invalid_reasons:
            raise ValueError(
                f"Selected MLA prefill backend {selected_backend.name} "
                f"is not valid for this configuration. "
                f"Reason: {invalid_reasons}"
            )
        assert backend_cls is not None
        logger.info("Using %s MLA prefill backend.", selected_backend.name)
        return backend_cls

    # 用户未指定，自动选择
    return _auto_select_mla_prefill_backend(
        device_capability,
        selector_config,
    )


@cache
def _auto_select_mla_prefill_backend(
    device_capability: DeviceCapability,
    selector_config: MLAPrefillSelectorConfig,
) -> "type[MLAPrefillBackend]":
    """Auto-select the best available MLA prefill backend.
    自动选择最佳可用的 MLA prefill 后端。

    按优先级遍历所有后端，验证每个后端的配置兼容性（计算能力、
    数据类型、依赖可用性等），返回第一个满足所有条件的后端。

    使用 @cache 装饰器缓存结果，相同参数的重复调用不会重新计算。

    Args:
        device_capability: 设备的计算能力
        selector_config: 可哈希的后端选择配置

    Returns:
        选定的 prefill 后端类

    Raises:
        ValueError: 如果没有任何后端满足条件
    """
    priorities = _get_mla_prefill_backend_priorities(device_capability)
    all_invalid_reasons: dict[str, list[str]] = {}

    for backend_enum in priorities:
        backend_cls: type[MLAPrefillBackend] | None = None
        try:
            backend_cls = backend_enum.get_class()
            invalid_reasons = backend_cls.validate_configuration(
                device_capability, selector_config
            )
        except ImportError:
            invalid_reasons = ["ImportError"]
        if not invalid_reasons:
            assert backend_cls is not None
            logger.info_once("Using %s MLA prefill backend.", backend_enum.name)
            return backend_cls
        all_invalid_reasons[backend_enum.name] = invalid_reasons

    # 所有后端都不可用，记录详细原因并抛出异常
    reasons_str = (
        "{"
        + ", ".join(
            f"{name}: [{', '.join(reasons)}]"
            for name, reasons in all_invalid_reasons.items()
        )
        + "}"
    )
    config_str = repr(selector_config)
    logger.debug_once(
        "Some MLA prefill backends are not valid with %s. Reasons: %s.",
        config_str,
        reasons_str,
    )

    raise ValueError(
        f"No valid MLA prefill backend found with {config_str}. Reasons: {reasons_str}."
    )
