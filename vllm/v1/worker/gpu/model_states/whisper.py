# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Whisper 模型状态模块 (Whisper Model State)

本模块实现了 Whisper 语音转录模型的模型状态。Whisper 是一个编码器-解码器架构模型，
与标准 Transformer 模型有以下关键区别：

1. 编码器-解码器架构：
   - 编码器处理音频输入，生成音频表示
   - 解码器使用交叉注意力 (cross-attention) 关注编码器输出

2. 输入处理方式不同：
   - 标准模型：编码器输出作为 inputs_embeds 与文本嵌入合并
   - Whisper：编码器输出通过 encoder_outputs 传递，由解码器的交叉注意力层消费

3. 交叉注意力 (Cross-Attention)：
   - 解码器需要知道编码器序列的长度 (encoder_seq_lens)
   - 编码器的 K/V 在第一步写入 KV 缓存，后续解码步骤直接从缓存读取

4. 只支持转录任务 (transcription)
"""
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.kv_cache_interface import CrossAttentionSpec, KVCacheConfig
from vllm.v1.worker.gpu.attn_utils import build_attn_metadata
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.mm.encoder_runner import EncoderRunner
from vllm.v1.worker.gpu.model_states.interface import (
    ModelSpecificAttnMetadata,
    ModelState,
)
from vllm.v1.worker.gpu.states import RequestState
from vllm.v1.worker.utils import AttentionGroup


@dataclass
class WhisperAttnMetadata(ModelSpecificAttnMetadata):
    """Whisper 模型的注意力元数据。

    包含编码器序列长度信息，用于交叉注意力计算。
    按 KV 缓存组索引组织，因为只有包含 CrossAttentionSpec 的组才需要此信息。

    属性：
        encoder_seq_lens: 编码器序列长度字典，key 为 KV 缓存组 ID，
                         value 为 (GPU 张量, CPU ndarray) 的元组
    """
    encoder_seq_lens: dict[int, tuple[torch.Tensor, np.ndarray]]

    def get_extra_common_attn_kwargs(
        self,
        kv_cache_group_id: int,
        num_reqs: int,
    ) -> dict[str, Any]:
        """返回交叉注意力需要的编码器序列长度。

        Args:
            kv_cache_group_id: KV 缓存组 ID
            num_reqs: 当前批次的请求数

        Returns:
            包含 encoder_seq_lens 的字典（如果该组有交叉注意力）
        """
        encoder_seq_lens = self.encoder_seq_lens.get(kv_cache_group_id)
        if encoder_seq_lens is None:
            return {}
        encoder_seq_lens_gpu, encoder_seq_lens_cpu = encoder_seq_lens
        return {
            "encoder_seq_lens": encoder_seq_lens_gpu[:num_reqs],
            "encoder_seq_lens_cpu": encoder_seq_lens_cpu[:num_reqs],
        }


class WhisperModelState(ModelState):
    """Whisper 语音转录模型的状态实现。

    与标准模型的关键区别：
    1. 不使用 inputs_embeds，而是通过 encoder_outputs 传递编码器输出
    2. 需要管理编码器序列长度信息（用于交叉注意力）
    3. 只支持转录任务
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
        encoder_cache: EncoderCache | None,
        device: torch.device,
    ) -> None:
        """初始化 Whisper 模型状态。

        Args:
            vllm_config: vLLM 全局配置
            model: PyTorch 模型实例
            encoder_cache: 多模态编码器缓存（Whisper 必须有）
            device: 计算设备
        """
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.scheduler_config = vllm_config.scheduler_config
        self.model = model
        self.max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        self.max_num_tokens = self.scheduler_config.max_num_batched_tokens
        self.max_model_len = self.model_config.max_model_len
        self.device = device

        # Whisper 必须有编码器缓存
        assert encoder_cache is not None
        self.encoder_cache = encoder_cache
        # 初始化编码器运行器
        self.encoder_runner = EncoderRunner(
            model=self.model,
            max_num_tokens=self.max_num_tokens,
            hidden_size=self.model_config.get_inputs_embeds_size(),
            encoder_cache=self.encoder_cache,
            dtype=self.model_config.dtype,
            device=self.device,
        )

        # 编码器最大序列长度（从 HF 配置获取）
        self.max_encoder_len = getattr(
            self.model_config.hf_config,
            "max_source_positions",
            self.max_model_len,
        )
        # GPU 上的编码器序列长度张量
        self.encoder_seq_lens_gpu = torch.zeros(
            self.max_num_reqs, dtype=torch.int32, device=self.device
        )

        # 编码器输出列表（每步更新）
        self.encoder_outputs: list[torch.Tensor] = []

    def get_supported_generation_tasks(self):
        """Whisper 只支持转录任务。"""
        return ("transcription",)

    def get_mm_embeddings(
        self,
        scheduled_encoder_inputs: dict[str, list[int]],
        input_batch: InputBatch,
        req_states: RequestState,
    ) -> None:
        """获取 Whisper 编码器的输出。

        与标准多模态模型不同，Whisper：
        1. 通过 encoder_outputs 而非 inputs_embeds 消费编码器输出
        2. 只有音频一种模态，因此 execute_mm_encoder 保持请求顺序
        3. 不需要存储到 encoder_cache：交叉注意力的 K/V 在第一步写入 KV 缓存，
           后续解码步骤使用缓存

        Args:
            scheduled_encoder_inputs: 调度的编码器输入
            input_batch: 输入批次数据
            req_states: 请求状态管理器

        Returns:
            None（Whisper 不返回 inputs_embeds）
        """
        # 确保编码器输入与 input_batch.req_ids 的顺序一致
        encoder_inputs: dict[str, list[int]] = {}
        for req_id in input_batch.req_ids:
            req_encoder_inputs = scheduled_encoder_inputs.get(req_id, [])
            if req_encoder_inputs:
                encoder_inputs[req_id] = req_encoder_inputs
        _, mm_kwargs = self.encoder_runner.prepare_mm_inputs(encoder_inputs)
        if mm_kwargs:
            # Whisper 通过 encoder_outputs 消费编码器输出，而非 inputs_embeds。
            # 单一模态（音频），所以 execute_mm_encoder 保持请求顺序；
            # 直接使用其返回值。
            # 无需存储到 encoder_cache：交叉注意力的 K/V 在第一步写入 KV 缓存；
            # 解码步骤使用缓存。
            self.encoder_outputs = self.encoder_runner.execute_mm_encoder(mm_kwargs)
        else:
            # 解码步骤：编码器 K/V 已在交叉注意力 KV 缓存中
            self.encoder_outputs = []
        return None

    def prepare_inputs(
        self, input_batch: InputBatch, req_states: RequestState
    ) -> dict[str, Any]:
        """准备 Whisper 解码器的输入。

        将编码器输出传递给解码器，然后清空以避免在后续步骤中重复使用。

        Args:
            input_batch: 输入批次数据
            req_states: 请求状态管理器

        Returns:
            包含 encoder_outputs 的字典
        """
        model_inputs = {"encoder_outputs": self.encoder_outputs}
        self.encoder_outputs = []
        return model_inputs

    def prepare_dummy_inputs(self, num_reqs: int, num_tokens: int) -> dict[str, Any]:
        """为 CUDA Graph 捕获准备虚拟输入。"""
        return {"encoder_outputs": []}

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
        """构建 Whisper 模型的注意力元数据。

        与标准模型相比，额外构建了编码器序列长度信息，
        用于交叉注意力计算。

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
            num_reqs = input_batch.num_reqs_after_padding
            num_tokens = input_batch.num_tokens_after_padding
        else:
            num_reqs = input_batch.num_reqs
            num_tokens = input_batch.num_tokens

        # 创建 Whisper 特有的注意力元数据（包含编码器序列长度）
        whisper_attn_metadata = WhisperAttnMetadata(
            self._get_encoder_seq_lens(input_batch.req_ids, attn_groups, for_capture)
        )

        query_start_loc_cpu = torch.from_numpy(input_batch.query_start_loc_np)
        max_query_len = input_batch.num_scheduled_tokens.max().item()
        seq_lens_cpu_upper_bound = input_batch.seq_lens_cpu_upper_bound
        if for_capture:
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
            model_specific_attn_metadata=whisper_attn_metadata,
            for_cudagraph_capture=for_capture,
        )
        return attn_metadata

    def _get_encoder_seq_lens(
        self,
        req_ids: list[str],
        attn_groups: list[list[AttentionGroup]],
        for_capture: bool,
    ) -> dict[int, tuple[torch.Tensor, np.ndarray]]:
        """获取每个请求的编码器序列长度。

        对于包含交叉注意力的 KV 缓存组，返回编码器序列长度。
        长度信息同时以 GPU 张量和 CPU ndarray 形式返回，
        因为不同位置可能需要不同的数据格式。

        Args:
            req_ids: 请求 ID 列表
            attn_groups: 注意力组信息
            for_capture: 是否用于 CUDA Graph 捕获

        Returns:
            编码器序列长度字典，key 为 KV 缓存组 ID
        """
        num_reqs = len(req_ids)
        encoder_seq_lens_np = np.zeros(num_reqs, dtype=np.int32)
        if not for_capture:
            # 正常执行期间，使用实际的编码器长度
            for i, req_id in enumerate(req_ids):
                mm_features = self.encoder_cache.mm_features.get(req_id, [])
                encoder_seq_lens_np[i] = sum(
                    feature.mm_position.get_num_embeds() for feature in mm_features
                )
        else:
            # CUDA Graph 捕获期间，使用最大编码器长度，以便
            # max_seqlen_k 为交叉注意力捕获正确的值
            encoder_seq_lens_np[:] = self.max_encoder_len

        # 将编码器序列长度复制到 GPU（非阻塞传输）
        self.encoder_seq_lens_gpu[:num_reqs].copy_(
            torch.from_numpy(encoder_seq_lens_np), non_blocking=True
        )
        self.encoder_seq_lens_gpu[num_reqs:].fill_(0)
        encoder_seq_lens_gpu = self.encoder_seq_lens_gpu[:num_reqs]

        # 只为包含交叉注意力的 KV 缓存组返回编码器序列长度
        seq_lens_by_group: dict[int, tuple[torch.Tensor, np.ndarray]] = {}
        for kv_cache_group_idx, groups in enumerate(attn_groups):
            has_cross_attn = any(
                isinstance(attn_group.kv_cache_spec, CrossAttentionSpec)
                for attn_group in groups
            )
            if has_cross_attn:
                seq_lens_by_group[kv_cache_group_idx] = (
                    encoder_seq_lens_gpu,
                    encoder_seq_lens_np,
                )
        return seq_lens_by_group
