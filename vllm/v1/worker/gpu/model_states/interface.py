# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
模型状态接口定义模块 (Model State Interface)

本模块定义了模型状态 (ModelState) 的抽象基类接口。ModelState 是 vLLM v1 引擎中
连接调度器输出与模型前向计算之间的桥梁，负责：

1. 管理请求生命周期：添加新请求、暂存写入、后处理状态
2. 准备模型输入：多模态嵌入、位置编码、注意力元数据等
3. 支持 CUDA Graph 捕获：为 CUDA Graph 准备虚拟输入

每个具体的模型架构（默认 Transformer、Mamba 混合、Whisper 等）都需要实现此接口。
"""
from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.tasks import GenerationTask
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.states import RequestState
from vllm.v1.worker.utils import AttentionGroup


class ModelSpecificAttnMetadata:
    """模型特定注意力元数据的基类。

    某些模型架构（如 Whisper 的交叉注意力、Mamba 的混合注意力）需要在标准注意力
    元数据之外传递额外的参数。此类提供两个钩子方法：

    1. get_extra_common_attn_kwargs: 返回传递给所有注意力层的额外公共参数
       （如 is_prefilling 标记、encoder_seq_lens 等）
    2. get_extra_attn_kwargs: 返回针对特定注意力后端的额外参数
       （如推测解码的 token 数量等）
    """

    def get_extra_common_attn_kwargs(
        self,
        kv_cache_group_id: int,
        num_reqs: int,
    ) -> dict[str, Any]:
        """返回传递给注意力公共构建逻辑的额外参数。

        Args:
            kv_cache_group_id: KV 缓存组的 ID
            num_reqs: 当前批次中的请求数

        Returns:
            额外的关键字参数字典
        """
        return {}

    def get_extra_attn_kwargs(
        self,
        attn_metadata_builder: Any,
        num_reqs: int,
    ) -> dict[str, Any]:
        """返回针对特定注意力元数据构建器的额外参数。

        Args:
            attn_metadata_builder: 注意力元数据构建器实例
            num_reqs: 当前批次中的请求数

        Returns:
            额外的关键字参数字典
        """
        return {}


class ModelState(ABC):
    """模型状态的抽象基类，定义了所有模型架构必须实现的接口。

    生命周期中的关键方法调用顺序：
    1. add_request: 当新请求被调度时调用，初始化请求相关的状态
    2. apply_staged_writes: 将暂存的写入操作批量应用到 GPU
    3. get_mm_embeddings: 获取多模态输入的嵌入向量（如图像嵌入）
    4. prepare_inputs: 准备模型前向计算所需的输入（如位置编码）
    5. prepare_attn: 构建注意力元数据（如 block table、slot mapping 等）
    6. postprocess_state: 模型前向计算完成后的后处理（如更新 Mamba 状态）
    """

    @abstractmethod
    def __init__(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
        encoder_cache: EncoderCache | None,
        device: torch.device,
    ) -> None:
        """初始化模型状态。

        Args:
            vllm_config: vLLM 全局配置
            model: PyTorch 模型实例
            encoder_cache: 多模态编码器缓存（None 表示不支持多模态）
            device: 计算设备
        """
        raise NotImplementedError

    @abstractmethod
    def get_supported_generation_tasks(self) -> tuple[GenerationTask, ...]:
        """返回模型支持的生成任务类型。

        Returns:
            支持的任务元组，如 ("generate",) 或 ("transcription",) 等
        """
        raise NotImplementedError

    def add_request(self, req_index: int, new_req_data: NewRequestData) -> None:
        """当新请求被添加到批次时调用。

        子类可重写此方法以初始化请求特定的状态，如 RoPE 位置编码的初始值。

        Args:
            req_index: 请求在批次中的索引
            new_req_data: 新请求的数据（包含 token IDs、多模态特征等）
        """
        return None

    def apply_staged_writes(self) -> None:
        """将所有暂存的写入操作批量应用到 GPU。

        采用"暂存-批量写入"模式 (staged-write pattern) 来减少 GPU 同步开销：
        先在 CPU 端暂存数据，再通过一次批量操作写入 GPU。
        """
        return None

    def postprocess_state(
        self,
        input_batch: InputBatch,
        num_sampled: torch.Tensor,
    ) -> None:
        """模型前向计算完成后的后处理。

        用于更新模型特定的状态，例如 Mamba 混合模型需要跟踪每个请求的采样 token 数量。

        Args:
            input_batch: 输入批次数据
            num_sampled: 每个请求采样的 token 数量
        """
        return None

    @abstractmethod
    def get_mm_embeddings(
        self,
        scheduled_encoder_inputs: dict[str, list[int]],
        input_batch: InputBatch,
        req_states: RequestState,
    ) -> torch.Tensor | None:
        """获取多模态输入的嵌入向量。

        对于支持多模态的模型（如 LLaVA、Qwen-VL 等），此方法：
        1. 准备多模态输入数据
        2. 执行多模态编码器
        3. 收集并返回嵌入向量

        Args:
            scheduled_encoder_inputs: 调度的编码器输入，格式为 {req_id: [mm_input_ids]}
            input_batch: 输入批次数据
            req_states: 请求状态管理器

        Returns:
            多模态嵌入张量，或 None（如 Whisper 使用交叉注意力，不需要此方法）
        """
        raise NotImplementedError

    @abstractmethod
    def prepare_inputs(
        self, input_batch: InputBatch, req_states: RequestState
    ) -> dict[str, Any]:
        """准备模型前向计算所需的输入。

        不同模型需要不同的输入，例如：
        - 标准 Transformer: 需要 positions（位置编码）
        - 多模态模型: 可能需要 inputs_embeds
        - Whisper: 需要 encoder_outputs

        Args:
            input_batch: 输入批次数据
            req_states: 请求状态管理器

        Returns:
            模型输入字典
        """
        raise NotImplementedError

    @abstractmethod
    def prepare_dummy_inputs(self, num_reqs: int, num_tokens: int) -> dict[str, Any]:
        """为 CUDA Graph 捕获准备虚拟输入。

        CUDA Graph 要求在捕获阶段使用固定形状的输入。此方法生成虚拟输入，
        使得 CUDA Graph 可以在这些虚拟输入上完成捕获。

        Args:
            num_reqs: 虚拟请求的数量
            num_tokens: 虚拟 token 的数量

        Returns:
            虚拟输入字典
        """
        raise NotImplementedError

    @abstractmethod
    def prepare_attn(
        self,
        input_batch: InputBatch,
        cudagraph_mode: CUDAGraphMode,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        for_capture: bool = False,
    ) -> dict[str, Any]:
        """构建注意力元数据。

        此方法将输入批次信息转换为注意力层所需的元数据，包括：
        1. 请求和 token 的数量信息
        2. query 起始位置
        3. 序列长度
        4. block table 和 slot mapping
        5. 模型特定的注意力参数

        Args:
            input_batch: 输入批次数据
            cudagraph_mode: CUDA Graph 模式（FULL / PIECEWISE / NONE）
            block_tables: KV 缓存的 block table
            slot_mappings: token 到 KV 缓存 slot 的映射
            attn_groups: 注意力组信息
            kv_cache_config: KV 缓存配置
            for_capture: 是否用于 CUDA Graph 捕获

        Returns:
            注意力元数据字典
        """
        raise NotImplementedError
