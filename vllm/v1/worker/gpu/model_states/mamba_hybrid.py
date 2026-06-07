# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Mamba 混合模型状态模块 (Mamba Hybrid Model State)

本模块实现了 Mamba + Transformer 混合架构模型（如 Jamba）的模型状态。
这类模型同时包含标准 Transformer 注意力层和 Mamba/线性注意力层。

与标准 Transformer 模型相比，Mamba 混合模型有以下特殊需求：
1. 需要跟踪每个请求是否处于 prefill 阶段（Mamba 层需要此信息）
2. 需要管理推测解码 (speculative decoding) 相关的状态：
   - num_accepted_tokens: 每步实际接受的 token 数量
   - num_decode_draft_tokens: 草稿 token 数量
3. 使用专用的注意力元数据构建器（Mamba2AttentionMetadataBuilder、GDNAttentionMetadataBuilder）
"""
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder
from vllm.v1.attention.backends.mamba2_attn import Mamba2AttentionMetadataBuilder
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.attn_utils import build_attn_metadata
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.model_states.default import DefaultModelState
from vllm.v1.worker.gpu.model_states.interface import ModelSpecificAttnMetadata
from vllm.v1.worker.utils import AttentionGroup


@dataclass
class MambaHybridAttnMetadata(ModelSpecificAttnMetadata):
    """Mamba 混合模型的注意力元数据。

    属性：
        is_prefilling: 每个请求是否处于 prefill 阶段的布尔张量
        num_accepted_tokens: 每个请求接受的 token 数量（用于推测解码）
        num_decode_draft_tokens_cpu: 每个请求的草稿 token 数量（用于推测解码）
    """
    is_prefilling: torch.Tensor
    num_accepted_tokens: torch.Tensor | None = None
    num_decode_draft_tokens_cpu: torch.Tensor | None = None

    def get_extra_common_attn_kwargs(
        self,
        kv_cache_group_id: int,
        num_reqs: int,
    ) -> dict[str, Any]:
        """返回 Mamba 层需要的公共注意力参数。

        Mamba 层需要知道每个请求是否处于 prefill 阶段，
        因为 prefill 和 decode 阶段的处理逻辑不同。

        Args:
            kv_cache_group_id: KV 缓存组 ID
            num_reqs: 当前批次的请求数

        Returns:
            包含 is_prefilling 的字典
        """
        return {"is_prefilling": self.is_prefilling[:num_reqs]}

    def get_extra_attn_kwargs(
        self,
        attn_metadata_builder: Any,
        num_reqs: int,
    ) -> dict[str, Any]:
        """返回针对 Mamba2/GDN 注意力构建器的额外参数。

        仅当注意力构建器为 Mamba2 或 GDN 类型时才返回参数，
        因为这些参数仅对 Mamba 相关的注意力层有意义。

        Args:
            attn_metadata_builder: 注意力元数据构建器
            num_reqs: 当前批次的请求数

        Returns:
            包含推测解码相关参数的字典
        """
        if not isinstance(
            attn_metadata_builder,
            (Mamba2AttentionMetadataBuilder, GDNAttentionMetadataBuilder),
        ):
            return {}
        return {
            "num_accepted_tokens": None
            if self.num_accepted_tokens is None
            else self.num_accepted_tokens[:num_reqs],
            "num_decode_draft_tokens_cpu": None
            if self.num_decode_draft_tokens_cpu is None
            else self.num_decode_draft_tokens_cpu[:num_reqs],
        }


class MambaHybridModelState(DefaultModelState):
    """Mamba + Transformer 混合架构模型的状态实现。

    继承自 DefaultModelState，额外管理：
    1. 每个请求的 prefill/decode 状态
    2. 推测解码相关的 token 计数
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
        encoder_cache: EncoderCache | None,
        device: torch.device,
    ) -> None:
        """初始化 Mamba 混合模型状态。

        Args:
            vllm_config: vLLM 全局配置
            model: PyTorch 模型实例
            encoder_cache: 多模态编码器缓存
            device: 计算设备
        """
        super().__init__(vllm_config, model, encoder_cache, device)
        # 每个请求接受的 token 数量，初始为 1（非推测解码模式的默认值）
        self.num_accepted_tokens_gpu = torch.ones(
            self.max_num_reqs, dtype=torch.int32, device=self.device
        )

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
        """构建 Mamba 混合模型的注意力元数据。

        与默认模型相比，此方法额外构建：
        1. is_prefilling 标记：标识每个请求是否处于 prefill 阶段
        2. num_accepted_tokens: 推测解码中实际接受的 token 数量
        3. num_decode_draft_tokens_cpu: 草稿 token 数量（GDN 使用 -1 作为非解码行的哨兵值）

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
        query_start_loc_cpu = torch.from_numpy(input_batch.query_start_loc_np)
        max_query_len = input_batch.num_scheduled_tokens.max().item()

        # 构建 is_prefilling 标记数组
        is_prefilling = torch.zeros(num_reqs, dtype=torch.bool, device="cpu")
        is_prefilling[: input_batch.num_reqs] = torch.from_numpy(
            input_batch.is_prefilling_np
        )

        # 在 CUDA Graph 捕获期间，num_decode_draft_tokens_cpu 和 num_accepted_tokens
        # 由 attn_metadata_builder.build_for_cudagraph_capture 创建，
        # 因此仅在实际（非捕获）前向执行期间计算它们。
        num_accepted_tokens = None
        num_decode_draft_tokens_cpu = None
        if not for_capture:
            # 构建每个请求的 accepted tokens 数量
            num_accepted_tokens = self.num_accepted_tokens_gpu.new_ones(num_reqs)
            num_accepted_tokens[: input_batch.num_reqs] = self.num_accepted_tokens_gpu[
                input_batch.idx_mapping
            ]

            # GDN 使用 >= 0 来选择推测解码行，因此非解码行需要 -1 作为哨兵值，
            # 而不是原始的 0 草稿计数
            num_decode_draft_tokens_np = np.full(num_reqs, -1, dtype=np.int32)
            if input_batch.num_draft_tokens_per_req is not None:
                # 推测解码掩码：仅对有草稿 token 且不在 prefill 阶段的请求生效
                spec_decode_mask = (
                    input_batch.num_draft_tokens_per_req > 0
                ) & ~input_batch.is_prefilling_np
                num_decode_draft_tokens_np[: input_batch.num_reqs] = np.where(
                    spec_decode_mask,
                    input_batch.num_draft_tokens_per_req,
                    -1,
                )
            num_decode_draft_tokens_cpu = torch.from_numpy(num_decode_draft_tokens_np)

        # 创建 Mamba 混合注意力元数据
        mamba_attn_metadata = MambaHybridAttnMetadata(
            is_prefilling=is_prefilling,
            num_accepted_tokens=num_accepted_tokens,
            num_decode_draft_tokens_cpu=num_decode_draft_tokens_cpu,
        )
        return build_attn_metadata(
            attn_groups=attn_groups,
            num_reqs=num_reqs,
            num_tokens=num_tokens,
            query_start_loc_gpu=input_batch.query_start_loc,
            query_start_loc_cpu=query_start_loc_cpu,
            max_query_len=max_query_len,
            seq_lens=input_batch.seq_lens,
            max_seq_len=self.max_model_len,
            block_tables=block_tables,
            slot_mappings=slot_mappings,
            kv_cache_config=kv_cache_config,
            dcp_local_seq_lens=input_batch.dcp_local_seq_lens,
            model_specific_attn_metadata=mamba_attn_metadata,
            for_cudagraph_capture=for_capture,
        )

    def postprocess_state(
        self,
        input_batch: InputBatch,
        num_sampled: torch.Tensor,
    ) -> None:
        """更新每个请求的 accepted tokens 计数。

        分块 prefill 不会采样 token，因此 num_sampled 可能为 0。
        Mamba 将 num_accepted_tokens=1 视为非推测解码的中性值，
        所以使用 clamp(min=1) 确保最小值为 1。

        Args:
            input_batch: 输入批次数据
            num_sampled: 每个请求采样的 token 数量
        """
        # 分块 prefill 不采样 token，因此 num_sampled 可能为 0。
        # Mamba 将 num_accepted_tokens=1 视为非推测解码的中性值。
        self.num_accepted_tokens_gpu[input_batch.idx_mapping] = torch.clamp(
            num_sampled, min=1
        )
