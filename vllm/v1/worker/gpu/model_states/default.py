# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
默认模型状态模块 (Default Model State)

本模块实现了标准 Transformer 模型（如 LLaMA、Qwen、GPT 等）的模型状态。
这是最常用的模型状态实现，支持：
1. 文本生成 (generate)
2. 多模态输入（图像、音频等）的编码与嵌入
3. 多维旋转位置编码 (M-RoPE / XD-RoPE)
4. 语音转录 (transcription)
5. 实时流式输出 (realtime)

工作流程：
1. 新请求到来时，初始化 RoPE 位置编码
2. 如果是多模态请求，执行多模态编码器获取嵌入
3. 准备位置编码（1D 或多维）
4. 构建注意力元数据
"""
from typing import Any

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.tasks import GenerationTask
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.attn_utils import build_attn_metadata
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.mm.encoder_runner import EncoderRunner
from vllm.v1.worker.gpu.mm.rope import get_rope_state
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.gpu.states import RequestState
from vllm.v1.worker.utils import AttentionGroup


class DefaultModelState(ModelState):
    """默认的模型状态实现，适用于标准 Transformer 模型。

    主要职责：
    1. 管理多模态编码器（EncoderRunner）：处理图像/音频等多模态输入
    2. 管理旋转位置编码（RopeState）：计算 1D 或多维位置编码
    3. 构建注意力元数据：将输入批次信息转换为注意力层所需的格式
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
        encoder_cache: EncoderCache | None,
        device: torch.device,
    ):
        """初始化默认模型状态。

        Args:
            vllm_config: vLLM 全局配置
            model: PyTorch 模型实例
            encoder_cache: 多模态编码器缓存（None 表示不支持多模态）
            device: 计算设备
        """
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.scheduler_config = vllm_config.scheduler_config
        self.model = model
        self.device = device

        # 是否支持多模态输入（取决于 encoder_cache 是否存在）
        self.supports_mm_inputs = encoder_cache is not None
        self.max_model_len = self.model_config.max_model_len
        self.max_num_reqs = self.scheduler_config.max_num_seqs
        self.max_num_tokens = self.scheduler_config.max_num_batched_tokens
        self.inputs_embeds_size = self.model_config.get_inputs_embeds_size()
        self.dtype = self.model_config.dtype

        # 初始化多模态编码器运行器（仅在支持多模态时创建）
        if self.supports_mm_inputs:
            assert encoder_cache is not None
            self.encoder_cache = encoder_cache
            self.encoder_runner = EncoderRunner(
                model=self.model,
                max_num_tokens=self.max_num_tokens,
                hidden_size=self.inputs_embeds_size,
                encoder_cache=encoder_cache,
                dtype=self.dtype,
                device=self.device,
            )

        # 初始化 RoPE 状态（如果模型使用多维旋转位置编码）
        self.rope_state = get_rope_state(
            self.model_config,
            model,
            max_num_reqs=self.max_num_reqs,
            max_num_tokens=self.max_num_tokens,
            max_model_len=self.max_model_len,
            device=self.device,
        )

    def get_supported_generation_tasks(self) -> tuple[GenerationTask, ...]:
        """返回模型支持的生成任务类型。

        检查模型是否支持以下任务：
        1. generate: 文本生成（标准 LLM 功能）
        2. transcription: 语音转录（如 Whisper）
        3. realtime: 实时流式输出

        Returns:
            支持的任务元组
        """
        from vllm.model_executor.models.interfaces import (
            supports_realtime,
            supports_transcription,
        )
        from vllm.model_executor.models.interfaces_base import is_text_generation_model

        supported_tasks = list[GenerationTask]()

        # 检查是否为文本生成模型
        if is_text_generation_model(self.model):
            supported_tasks.append("generate")

        # 检查是否支持语音转录
        if supports_transcription(self.model):
            # 如果模型仅支持转录（如 Whisper），则直接返回
            if self.model.supports_transcription_only:
                return ("transcription",)
            supported_tasks.append("transcription")

        # 检查是否支持实时流式输出
        if supports_realtime(self.model):
            supported_tasks.append("realtime")

        return tuple(supported_tasks)

    def add_request(self, req_index: int, new_req_data: NewRequestData) -> None:
        """添加新请求时，初始化 RoPE 位置编码。

        对于使用多维 RoPE 的模型，需要根据 token IDs 和多模态特征
        计算每个 token 在各维度上的位置编码。

        Args:
            req_index: 请求在批次中的索引
            new_req_data: 新请求的数据
        """
        if self.rope_state is not None:
            assert new_req_data.prefill_token_ids is not None
            self.rope_state.init_prefill_positions(
                req_index,
                self.model,
                new_req_data.prefill_token_ids,
                mm_features=new_req_data.mm_features,
            )

    def apply_staged_writes(self) -> None:
        """将暂存的 RoPE 位置编码写入 GPU。"""
        if self.rope_state is not None:
            self.rope_state.apply_staged_writes()

    def get_mm_embeddings(
        self,
        scheduled_encoder_inputs: dict[str, list[int]],
        input_batch: InputBatch,
        req_states: RequestState,
    ) -> torch.Tensor:
        """获取多模态输入的嵌入向量。

        处理流程：
        1. 准备多模态输入数据（提取需要编码的多模态特征）
        2. 执行多模态编码器（如 ViT 视觉编码器）
        3. 将编码结果缓存到 encoder_cache 中（以 mm_hash 为键）
        4. 收集当前步骤需要的嵌入向量
        5. 将文本 token IDs 和多模态嵌入合并为统一的输入嵌入

        Args:
            scheduled_encoder_inputs: 本步骤调度的编码器输入
            input_batch: 输入批次数据
            req_states: 请求状态管理器

        Returns:
            合并后的输入嵌入张量
        """
        # 1. 准备多模态输入：提取需要编码的特征和对应的 hash
        mm_hashes, mm_kwargs = self.encoder_runner.prepare_mm_inputs(
            scheduled_encoder_inputs
        )
        if mm_kwargs:
            # 2. 执行多模态编码器（如视觉 Transformer）
            encoder_outputs = self.encoder_runner.execute_mm_encoder(mm_kwargs)
            # 3. 将编码结果缓存（key: mm_hash, value: encoder_output）
            self.encoder_cache.encoder_outputs.update(zip(mm_hashes, encoder_outputs))

        # 4. 收集当前步骤需要的多模态嵌入
        mm_embeds, is_mm_embed = self.encoder_runner.gather_mm_embeddings(
            input_batch.req_ids,
            input_batch.num_tokens,
            input_batch.num_scheduled_tokens,
            input_batch.query_start_loc_np,
            req_states.prefill_len.np[input_batch.idx_mapping_np],
            req_states.num_computed_prefill_tokens[input_batch.idx_mapping_np],
        )
        # 使用未填充的 input_ids 以匹配 is_mm_embed 的大小（num_tokens）。
        # input_batch.input_ids 可能为了 CUDA Graph 而被填充。
        input_ids_unpadded = input_batch.input_ids[: input_batch.num_tokens]
        # 5. 将文本 token IDs 和多模态嵌入合并
        inputs_embeds = self.encoder_runner.get_inputs_embeds(
            input_ids_unpadded, mm_embeds, is_mm_embed
        )
        return inputs_embeds[: input_batch.num_tokens_after_padding]

    def prepare_inputs(
        self, input_batch: InputBatch, req_states: RequestState
    ) -> dict[str, torch.Tensor | None]:
        """准备模型前向计算所需的位置编码输入。

        对于大多数标准 1D 位置编码模型，此方法不需要做任何事情（位置编码由
        模型内部计算）。只有使用多维 RoPE（如 M-RoPE、XD-RoPE）的模型才需要
        在此处显式计算位置编码。

        Args:
            input_batch: 输入批次数据
            req_states: 请求状态管理器

        Returns:
            包含 "positions" 键的字典（1D 模型返回空字典）
        """
        if self.rope_state is None:
            return {}  # 常见情况：标准 1D 位置编码

        # 准备多维 RoPE 位置编码
        self.rope_state.prepare_positions(
            input_batch.idx_mapping,
            input_batch.query_start_loc,
            req_states.prefill_len.gpu,
            req_states.num_computed_tokens.gpu,
        )
        positions = self.rope_state.get_positions(input_batch.num_tokens_after_padding)
        return {"positions": positions}

    def prepare_dummy_inputs(self, num_reqs: int, num_tokens: int) -> dict[str, Any]:
        """为 CUDA Graph 捕获准备虚拟输入。

        在 CUDA Graph 捕获阶段，需要提供固定形状的虚拟输入。此方法：
        1. 如果支持多模态，使用预分配的 inputs_embeds 缓冲区
        2. 如果使用多维 RoPE，生成虚拟位置编码

        Args:
            num_reqs: 虚拟请求数量
            num_tokens: 虚拟 token 数量

        Returns:
            虚拟输入字典
        """
        model_inputs = {}
        if self.supports_mm_inputs:
            inputs_embeds = self.encoder_runner.inputs_embeds[:num_tokens]
            model_inputs["inputs_embeds"] = inputs_embeds
        if self.rope_state is not None:
            model_inputs["positions"] = self.rope_state.get_positions(num_tokens)
        return model_inputs

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

        根据 CUDA Graph 模式决定使用填充或未填充的尺寸：
        - FULL CUDA Graph 模式：使用填充后的尺寸（padding 由 model_runner 处理）
        - 分段 CUDA Graph 或 eager 模式：使用未填充的实际尺寸

        对于 CUDA Graph 捕获，使用最大序列长度作为 max_seq_len，以确保
        捕获的 Graph 在任何回放场景下都有效。

        Args:
            input_batch: 输入批次数据
            cudagraph_mode: CUDA Graph 模式
            block_tables: KV 缓存的 block table
            slot_mappings: token 到 KV 缓存 slot 的映射
            attn_groups: 注意力组信息
            kv_cache_config: KV 缓存配置
            for_capture: 是否用于 CUDA Graph 捕获

        Returns:
            注意力元数据字典
        """
        if cudagraph_mode == CUDAGraphMode.FULL:
            # 使用填充后的尺寸 - 填充由 model_runner.prepare_attn 处理
            num_reqs = input_batch.num_reqs_after_padding
            num_tokens = input_batch.num_tokens_after_padding
        else:
            # 分段 CUDA Graph 和 eager 模式使用未填充的尺寸
            num_reqs = input_batch.num_reqs
            num_tokens = input_batch.num_tokens
        query_start_loc_cpu = torch.from_numpy(input_batch.query_start_loc_np)
        max_query_len = input_batch.num_scheduled_tokens.max().item()
        seq_lens_cpu_upper_bound = input_batch.seq_lens_cpu_upper_bound
        if for_capture:
            # 捕获时使用最坏情况的 max_seq_len，确保 Graph 在任何回放时都有效
            max_seq_len = self.max_model_len
        else:
            max_seq_len = int(seq_lens_cpu_upper_bound[:num_reqs].max().item())
        attn_metadata = build_attn_metadata(
            attn_groups=attn_groups,
            num_reqs=num_reqs,
            num_tokens=num_tokens,
            query_start_loc_gpu=input_batch.query_start_loc,
            query_start_loc_cpu=query_start_loc_cpu,
            max_query_len=max_query_len,
            seq_lens=input_batch.seq_lens,
            max_seq_len=max_seq_len,
            block_tables=block_tables,
            slot_mappings=slot_mappings,
            kv_cache_config=kv_cache_config,
            seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
            dcp_local_seq_lens=input_batch.dcp_local_seq_lens,
            positions=input_batch.positions,
            for_cudagraph_capture=for_capture,
        )
        return attn_metadata
