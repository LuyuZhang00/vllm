# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
多模态编码器运行器模块 (Multi-Modal Encoder Runner Module)

本模块负责执行多模态编码器（如视觉 Transformer）并管理其输入输出。

主要职责：
1. 准备编码器输入：从调度器输出中提取需要编码的多模态数据
2. 执行编码器：运行多模态编码器（如 ViT、音频编码器等）
3. 收集嵌入：从编码器输出中收集当前步骤需要的嵌入向量
4. 合并嵌入：将文本嵌入和多模态嵌入合并为统一的输入

工作流程：
1. prepare_mm_inputs: 提取需要编码的多模态特征
2. execute_mm_encoder: 执行编码器，获取编码器输出
3. gather_mm_embeddings: 收集当前步骤需要的嵌入
4. get_inputs_embeds: 合并文本和多模态嵌入

优化策略：
- 跳过解码阶段的请求（不需要新的编码器输出）
- 跳过已处理的编码器输出（已存储在 KV 缓存中）
- 使用预分配的缓冲区支持 CUDA Graph
"""
import numpy as np
import torch

from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.multimodal.inputs import MultiModalKwargsItem
from vllm.multimodal.utils import group_and_batch_mm_kwargs
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.utils import sanity_check_mm_encoder_outputs


class EncoderRunner:
    """多模态编码器运行器。

    管理多模态编码器的执行和输入输出。
    """

    def __init__(
        self,
        model: SupportsMultiModal,
        max_num_tokens: int,
        hidden_size: int,
        encoder_cache: EncoderCache,
        dtype: torch.dtype,
        device: torch.device,
    ):
        """初始化编码器运行器。

        Args:
            model: 支持多模态的模型实例
            max_num_tokens: 最大 token 数量
            hidden_size: 隐藏层大小
            encoder_cache: 编码器缓存
            dtype: 数据类型
            device: 计算设备
        """
        self.model = model
        self.max_num_tokens = max_num_tokens
        self.hidden_size = hidden_size
        self.encoder_cache = encoder_cache
        self.dtype = dtype
        self.device = device

        # 预分配的输入嵌入缓冲区（用于 CUDA Graph）
        self.inputs_embeds = torch.zeros(
            max_num_tokens, hidden_size, dtype=dtype, device=device
        )

    def prepare_mm_inputs(
        self, scheduled_encoder_inputs: dict[str, list[int]]
    ) -> tuple[list[str], list[tuple[str, MultiModalKwargsItem]]]:
        """准备编码器输入数据。

        从调度器输出中提取需要编码的多模态特征。

        Args:
            scheduled_encoder_inputs: 调度的编码器输入，格式为 {req_id: [mm_input_ids]}

        Returns:
            (mm_hashes, mm_kwargs): 多模态输入的 hash 列表和对应的 kwargs 列表
        """
        mm_hashes: list[str] = []
        mm_kwargs: list[tuple[str, MultiModalKwargsItem]] = []
        for req_id, encoder_input_ids in scheduled_encoder_inputs.items():
            mm_features = self.encoder_cache.mm_features[req_id]
            for mm_input_id in encoder_input_ids:
                mm_feature = mm_features[mm_input_id]
                if mm_feature.data is None:
                    continue
                mm_hashes.append(mm_feature.identifier)
                mm_kwargs.append((mm_feature.modality, mm_feature.data))

        return mm_hashes, mm_kwargs

    @torch.inference_mode()
    def execute_mm_encoder(
        self,
        mm_kwargs: list[tuple[str, MultiModalKwargsItem]],
    ) -> list[torch.Tensor]:
        """执行多模态编码器。

        按模态分组并批量执行编码器，获取编码器输出。

        Args:
            mm_kwargs: 多模态输入的 kwargs 列表

        Returns:
            编码器输出张量列表
        """
        encoder_outputs: list[torch.Tensor] = []
        for modality, num_items, mm_kwargs_batch in group_and_batch_mm_kwargs(
            mm_kwargs, device=self.device, pin_memory=False
        ):
            batch_outputs = self.model.embed_multimodal(**mm_kwargs_batch)
            sanity_check_mm_encoder_outputs(batch_outputs, expected_num_items=num_items)
            encoder_outputs.extend(batch_outputs)
        return encoder_outputs

    def gather_mm_embeddings(
        self,
        req_ids: list[str],
        total_num_scheduled_tokens: int,
        num_scheduled_tokens: np.ndarray,
        query_start_loc: np.ndarray,
        prefill_lens: np.ndarray,
        computed_prefill_lens: np.ndarray,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        """收集当前步骤需要的多模态嵌入。

        遍历所有请求，从编码器缓存中收集当前步骤需要的嵌入向量。
        跳过解码阶段的请求和已处理的编码器输出。

        Args:
            req_ids: 请求 ID 列表
            total_num_scheduled_tokens: 总调度 token 数
            num_scheduled_tokens: 每个请求的调度 token 数
            query_start_loc: 每个请求的 query 起始位置
            prefill_lens: prefill 长度
            computed_prefill_lens: 已计算的 prefill 长度

        Returns:
            (mm_embeds, is_mm_embed): 多模态嵌入列表和标记张量
        """
        is_prefilling = (computed_prefill_lens < prefill_lens).tolist()
        all_decode = not any(is_prefilling)
        if all_decode:
            # 全部是解码请求，不需要收集任何嵌入
            return [], torch.zeros(
                total_num_scheduled_tokens, dtype=torch.bool, device=self.device
            )

        query_start = computed_prefill_lens.tolist()
        query_end = (computed_prefill_lens + num_scheduled_tokens).tolist()

        mm_embeds: list[torch.Tensor] = []
        is_mm_embed = torch.zeros(
            total_num_scheduled_tokens, dtype=torch.bool, device="cpu"
        )
        for i, req_id in enumerate(req_ids):
            if not is_prefilling[i]:
                # 优化：跳过解码请求
                continue

            mm_features = self.encoder_cache.mm_features[req_id]
            for mm_feature in mm_features:
                pos_info = mm_feature.mm_position
                start_pos = pos_info.offset
                num_encoder_tokens = pos_info.length

                if start_pos >= query_end[i]:
                    # 编码器输出在此步骤中不需要
                    break
                if start_pos + num_encoder_tokens <= query_start[i]:
                    # 编码器输出已处理并存储在解码器的 KV 缓存中
                    continue

                start_idx = max(query_start[i] - start_pos, 0)
                end_idx = min(query_end[i] - start_pos, num_encoder_tokens)
                assert start_idx < end_idx
                curr_embeds_start, curr_embeds_end = (
                    pos_info.get_embeds_indices_in_range(start_idx, end_idx)
                )
                # 如果当前范围内没有嵌入，跳过收集
                if curr_embeds_start == curr_embeds_end:
                    continue

                mm_hash = mm_feature.identifier
                encoder_output = self.encoder_cache.encoder_outputs.get(mm_hash, None)
                assert encoder_output is not None, f"Encoder cache miss for {mm_hash}."

                if (is_embed := pos_info.is_embed) is not None:
                    is_embed = is_embed[start_idx:end_idx]
                    mm_embeds_item = encoder_output[curr_embeds_start:curr_embeds_end]
                else:
                    mm_embeds_item = encoder_output[start_idx:end_idx]

                req_start_pos = query_start_loc[i] + start_pos - query_start[i]
                is_mm_embed[req_start_pos + start_idx : req_start_pos + end_idx] = (
                    True if is_embed is None else is_embed
                )
                mm_embeds.append(mm_embeds_item)

        return mm_embeds, is_mm_embed

    @torch.inference_mode()
    def get_inputs_embeds(
        self,
        input_ids: torch.Tensor,
        mm_embeds: list[torch.Tensor],
        is_mm_embed: torch.Tensor,
    ) -> torch.Tensor:
        """合并文本 token IDs 和多模态嵌入为统一的输入嵌入。

        使用模型的 embed_input_ids 方法将文本嵌入和多模态嵌入合并。
        结果复制到预分配的缓冲区以支持 CUDA Graph。

        Args:
            input_ids: 文本 token IDs
            mm_embeds: 多模态嵌入列表
            is_mm_embed: 标记哪些位置是多模态嵌入

        Returns:
            合并后的输入嵌入张量
        """
        x = self.model.embed_input_ids(
            input_ids, multimodal_embeddings=mm_embeds, is_multimodal=is_mm_embed
        )
        # 复制到预分配的缓冲区以支持 CUDA Graph
        self.inputs_embeds[: x.shape[0]] = x
        return self.inputs_embeds
