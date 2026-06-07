# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Abstract base class for MLA prefill backends."""
# MLA (Multi-head Latent Attention) prefill 后端的抽象基类模块。
#
# MLA 是 DeepSeek 系列模型提出的高效注意力机制，其核心思想是通过低秩
# 联合压缩来减少 KV 缓存的显存占用。在 prefill 阶段，MLA 需要将输入
# 序列的全部 token 进行注意力计算。
#
# 本模块定义了所有 MLA prefill 后端必须实现的抽象接口，包括：
# 1. 后端名称和可用性检查
# 2. 设备计算能力和数据类型的支持验证
# 3. 配置验证（验证模型维度、依赖等是否满足）
# 4. prefill 核心计算方法（新 token 的 prefill 和上下文分块 prefill）

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

import torch

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.model_executor.layers.attention.mla_attention import (
        MLACommonPrefillMetadata,
    )
    from vllm.platforms.interface import DeviceCapability
    from vllm.v1.attention.backends.mla.prefill.selector import (
        MLAPrefillSelectorConfig,
    )


class MLAPrefillBackend(ABC):
    """Abstract base class for MLA prefill backends.
    MLA prefill 后端的抽象基类。所有具体的 MLA prefill 实现
    （如 FlashAttn、FlashInfer、TRT-LLM 等）都必须继承此类并实现
    其抽象方法。
    """

    # 1. 默认支持的数据类型列表，子类可以覆盖
    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
        torch.bfloat16,
    ]
    # 2. 是否要求模型具有 DeepSeek R1 的 MLA 维度配置
    #    DeepSeek R1 的 MLA 维度：qk_nope_head_dim=128, qk_rope_head_dim=64,
    #    v_head_dim=128。部分后端（如 FlashInfer、TRT-LLM）要求这些精确维度。
    requires_r1_mla_dimensions: ClassVar[bool] = False

    @staticmethod
    @abstractmethod
    def get_name() -> str:
        """获取后端的唯一名称标识符（如 "FLASH_ATTN"、"FLASHINFER"）。
        子类必须实现此方法。
        """
        raise NotImplementedError

    @classmethod
    def supports_compute_capability(cls, device_capability: "DeviceCapability") -> bool:
        """检查当前后端是否支持给定的设备计算能力。
        默认返回 True（支持所有设备）。子类可覆盖以限制特定 GPU 架构。

        Args:
            device_capability: 设备的计算能力（如 SM90=Hopper, SM100=Blackwell）

        Returns:
            是否支持该计算能力
        """
        return True

    @classmethod
    def supports_dtype(cls, dtype: torch.dtype) -> bool:
        """检查当前后端是否支持给定的数据类型。

        Args:
            dtype: 待检查的数据类型（如 torch.float16, torch.bfloat16）

        Returns:
            是否支持该数据类型
        """
        return dtype in cls.supported_dtypes

    @classmethod
    def is_available(cls) -> bool:
        """检查当前后端的依赖库是否已安装可用。
        默认返回 True。子类应覆盖此方法以检查特定依赖
        （如 flash_attn、flashinfer 等库是否已安装）。

        Returns:
            后端是否可用
        """
        return True

    @classmethod
    def validate_configuration(
        cls,
        device_capability: "DeviceCapability",
        selector_config: "MLAPrefillSelectorConfig",
    ) -> list[str]:
        """验证当前后端是否满足所有配置要求。

        按顺序检查以下条件：
        1. 设备计算能力是否支持
        2. 数据类型是否支持
        3. 依赖库是否可用
        4. 模型是否具有 R1 MLA 维度（如果后端要求的话）

        Args:
            device_capability: 设备的计算能力
            selector_config: 后端选择配置（包含 dtype、R1 兼容性等）

        Returns:
            不满足条件的原因列表。空列表表示所有条件均满足。
        """
        invalid_reasons: list[str] = []

        if not cls.supports_compute_capability(device_capability):
            invalid_reasons.append(
                f"compute capability {device_capability.major}."
                f"{device_capability.minor} not supported"
            )

        if not cls.supports_dtype(selector_config.dtype):
            invalid_reasons.append(f"dtype {selector_config.dtype} not supported")

        if not cls.is_available():
            invalid_reasons.append("required dependencies not available")

        if cls.requires_r1_mla_dimensions and not selector_config.is_r1_compatible:
            invalid_reasons.append(
                "model does not have DeepSeek R1 MLA dimensions "
                "(qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128)"
            )

        return invalid_reasons

    def __init__(
        self,
        num_heads: int,
        scale: float,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        vllm_config: "VllmConfig",
    ) -> None:
        """初始化 MLA prefill 后端。

        MLA 的注意力维度由以下几个部分组成：
        - num_heads: Q 的注意力头数
        - scale: 注意力缩放因子（通常为 1/sqrt(head_dim)）
        - kv_lora_rank: KV 的低秩压缩维度（MLA 的核心参数，用于减少 KV 缓存大小）
        - qk_nope_head_dim: Q/K 中不带 RoPE 旋转位置编码的维度
        - qk_rope_head_dim: Q/K 中带 RoPE 旋转位置编码的维度
        - v_head_dim: V 的头维度
        - vllm_config: vLLM 的全局配置对象
        """
        self.num_heads = num_heads
        self.scale = scale
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.vllm_config = vllm_config

    def prepare_metadata(  # noqa: B027
        self,
        prefill_metadata: "MLACommonPrefillMetadata",
    ) -> None:
        """Prepare backend-specific metadata before the forward pass.

        Called by the metadata builder after constructing the prefill metadata.
        在前向传播前准备后端特定的元数据。由元数据构建器在构造 prefill 元数据后调用。
        子类可以覆盖此方法以进行额外的元数据预处理。

        Args:
            prefill_metadata: MLA prefill 的通用元数据，包含 query 的起始位置、
                最大序列长度、分块上下文信息等。
        """
        self._prefill_metadata = prefill_metadata

    @abstractmethod
    def run_prefill_new_tokens(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        return_softmax_lse: bool,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """对新 token 执行 prefill 注意力计算。

        这是 prefill 阶段的核心方法，计算查询（Q）与键值对（K, V）之间的
        因果注意力。适用于序列中尚未缓存过 KV 的新 token。

        Args:
            q: 查询张量，形状为 (total_q_tokens, num_heads, qk_head_dim)
            k: 键张量，形状为 (total_kv_tokens, num_kv_heads, qk_head_dim)
            v: 值张量，形状为 (total_kv_tokens, num_kv_heads, v_head_dim)
            return_softmax_lse: 是否返回 softmax 的 log-sum-exp 值，
                用于后续的分块注意力合并

        Returns:
            如果 return_softmax_lse 为 False，返回注意力输出张量；
            否则返回 (注意力输出, softmax_lse) 的元组。
        """
        raise NotImplementedError

    @abstractmethod
    def run_prefill_context_chunk(
        self,
        chunk_idx: int,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """对上下文的一个分块执行 prefill 注意力计算。

        对于长序列，上下文（已缓存的 KV）会被分成多个块进行分块计算，
        以避免一次性处理过长的序列导致显存不足。每个块的注意力输出和
        softmax_lse 会在后续被合并。

        Args:
            chunk_idx: 当前上下文分块的索引
            q: 查询张量，形状为 (total_q_tokens, num_heads, qk_head_dim)
            k: 键张量（当前分块的 KV），形状由分块大小决定
            v: 值张量（当前分块的 KV），形状由分块大小决定

        Returns:
            (注意力输出, softmax_lse) 的元组。输出形状为
            (total_q_tokens, num_heads, v_head_dim)，lse 形状为
            (num_heads, total_q_tokens)。
        """
        raise NotImplementedError
