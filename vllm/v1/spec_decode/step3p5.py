# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Step3.5 MTP (Multi-Token Prediction) 投机解码提议器模块

本模块实现了 Step3.5 模型的 MTP 投机解码提议器。
Step3.5 MTP 是一种基于 EAGLE 架构改进的投机解码方法，
主要特点包括：

1. 多 KV 缓存组支持：Step3.5 的 MTP draft 层可能跨越多个 KV 缓存组，
   每个组有独立的 block table 和 slot mapping
2. 逐层 draft-step 选择：不同 draft 层可以选择不同的 draft step
3. 每层独立的 lm_head：Step3.5 MTP 每个 MTP 层自带 lm_head，
   而非像 EAGLE 那样共享 target model 的 lm_head

与 EAGLE 的关系：
  继承自 EagleProposer，覆盖了 KV 缓存管理、注意力元数据构建、
  采样逻辑等关键方法以适配 Step3.5 的特殊架构。

数据流（与 EAGLE 类似但多了多 KV 组处理）：
  target_hidden_states -> 多组 KV 缓存管理 -> 逐层前向传播 -> 多步 draft token 生成
"""

from copy import copy

import torch

from vllm.config import VllmConfig, get_layers_from_vllm_config, replace
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.models.utils import get_draft_quant_config
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.eagle import EagleProposer
from vllm.v1.spec_decode.utils import PADDING_SLOT_ID
from vllm.v1.worker.utils import AttentionGroup


class Step3p5MTPProposer(EagleProposer):
    """Step3.5 MTP proposer with per-layer draft-step selection."""

    # 中文注释：Step3.5 MTP 投机解码提议器。
    # 继承自 EagleProposer，核心扩展点是支持多 KV 缓存组。
    # Step3.5 的 MTP 层可能分布在不同的 KV 缓存组中，每组有独立的
    # block table 和 slot mapping，需要分别管理。

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ):
        super().__init__(vllm_config, device, runner)
        # 中文注释：每个 KV 缓存组独立的 block table。
        # key 是 KV 缓存组 id (gid)，value 是该组的 block table tensor。
        # 与 EAGLE 只用一个 block table 不同，Step3.5 需要按组管理。
        self._per_group_block_tables: dict[int, torch.Tensor] = {}
        # 中文注释：每个 KV 缓存组独立的 slot mapping。
        # slot mapping 将逻辑位置映射到物理 KV cache slot。
        self._per_group_slot_mappings: dict[int, torch.Tensor] = {}
        # Slot-mapping buffers for non-primary KV cache groups (the primary
        # group reuses self._slot_mapping_buffer from the base class).
        # 中文注释：非主 KV 缓存组的 slot mapping 缓冲区。
        # 主组复用基类的 self._slot_mapping_buffer，其他组各自分配独立缓冲区。
        self._per_group_slot_mapping_buffers: dict[int, torch.Tensor] = {}

    def set_per_group_attn_metadata(
        self,
        gid: int,
        block_table: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        """中文注释：为指定 KV 缓存组设置注意力元数据（block table 和 slot mapping）。
        由 GPUModelRunner 在构建 batch 时调用，将每个 KV 缓存组的元数据注册到提议器。

        参数：
            gid: KV 缓存组 id
            block_table: 该组的 block table（逻辑 block -> 物理 block 映射）
            slot_mapping: 该组的 slot mapping（token 位置 -> 物理 KV cache slot）
        """
        self._per_group_block_tables[gid] = block_table
        self._per_group_slot_mappings[gid] = slot_mapping

    def _slot_mapping_buffer_for(self, gid: int) -> torch.Tensor:
        """中文注释：获取指定 KV 缓存组的 slot mapping 缓冲区。
        主组（kv_cache_gid）复用基类的缓冲区，其他组按需创建独立缓冲区。
        缓冲区采用延迟初始化策略，避免未使用的组浪费显存。

        参数：
            gid: KV 缓存组 id

        返回：
            该组对应的 slot mapping 缓冲区 tensor
        """
        if gid == self.kv_cache_gid:
            return self._slot_mapping_buffer
        buf = self._per_group_slot_mapping_buffers.get(gid)
        if buf is None:
            buf = torch.zeros(self.max_positions, dtype=torch.int64, device=self.device)
            self._per_group_slot_mapping_buffers[gid] = buf
        return buf

    def _get_slot_mapping(
        self,
        num_tokens: int,
        slot_mapping: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Per-layer slot_mapping with one buffer per KV cache group."""
        # 中文注释：为每层生成 slot mapping，按 KV 缓存组分组管理。
        # 与 EAGLE 只需一个全局 slot mapping 不同，Step3.5 的每层可能属于不同的 KV 缓存组，
        # 因此需要为每组维护独立的 slot mapping。
        #
        # 流程：
        # 1. 遍历所有 draft 注意力组
        # 2. 为每组获取或复制 slot mapping 到对应缓冲区
        # 3. 超出部分填充 PADDING_SLOT_ID
        # 4. 将每组的 slot mapping 关联到其包含的所有层
        per_layer: dict[str, torch.Tensor] = {}
        for attn_group in self.draft_attn_groups:
            gid = attn_group.kv_cache_group_id
            buf = self._slot_mapping_buffer_for(gid)
            source = self._per_group_slot_mappings.get(gid, slot_mapping)
            # 中文注释：如果 source 不为空且与缓冲区不是同一块内存，复制数据
            if source is not None and buf.data_ptr() != source.data_ptr():
                n = source.shape[0]
                buf[:n].copy_(source)
                # 中文注释：超出实际 token 数的部分填充 padding slot id
                if num_tokens > n:
                    buf[n:num_tokens].fill_(PADDING_SLOT_ID)
            view = buf[:num_tokens]
            for layer_name in attn_group.layer_names:
                per_layer[layer_name] = view
        return per_layer

    def _update_positions_dependent_metadata(
        self,
        positions: torch.Tensor,
        common_attn_metadata: CommonAttentionMetadata,
        batch_size: int,
        input_batch_size: int,
        block_size: int,
    ) -> torch.Tensor:
        """中文注释：更新与位置相关的元数据（positions、slot_mapping 等）。
        覆盖父类方法，在父类为主 KV 缓存组生成 slot_mapping 后，
        为其他非主 KV 缓存组各自重新计算 slot_mapping。

        流程：
        1. 调用父类方法更新 positions 并为主组生成 slot_mapping
        2. 遍历非主 KV 缓存组
        3. 用各组自己的 block table 计算 slot_mapping
        4. 处理超出 max_model_len 的位置，填充 PADDING_SLOT_ID

        参数：
            positions: 当前 token 的位置
            common_attn_metadata: 通用注意力元数据（会被原地修改）
            batch_size: 实际 batch 大小
            input_batch_size: 填充后的 batch 大小
            block_size: KV 缓存块大小

        返回：
            更新后的 positions
        """
        old_positions_1d = positions[0] if self.uses_mrope else positions
        # 中文注释：调用父类方法，更新 positions 并为主 KV 缓存组计算 slot_mapping
        positions = super()._update_positions_dependent_metadata(
            positions,
            common_attn_metadata,
            batch_size,
            input_batch_size,
            block_size,
        )
        # Parent already produced slot_mapping for the primary gid.
        self._per_group_slot_mappings[self.kv_cache_gid] = (
            common_attn_metadata.slot_mapping
        )
        # Recompute slot_mapping for the remaining gids using their own block tables.
        # 中文注释：为非主 KV 缓存组重新计算 slot_mapping。
        # 每个组的 block table 不同，因此相同的位置会映射到不同的物理 slot。
        # slot_mapping 的计算公式：block_id * block_size + (position % block_size)
        new_positions_1d = positions[0] if self.uses_mrope else positions
        exceeds = old_positions_1d + 1 >= self.max_model_len
        for attn_group in self.draft_attn_groups:
            gid = attn_group.kv_cache_group_id
            if gid == self.kv_cache_gid:
                continue
            block_table = self._per_group_block_tables.get(gid)
            if block_table is None:
                continue
            # 中文注释：通过 block table 查找每个位置对应的物理 block id
            n_blocks = block_table.shape[1]
            bn = (new_positions_1d // block_size).clamp(max=n_blocks - 1).to(torch.long)
            block_ids = block_table[:batch_size].gather(1, bn.unsqueeze(1)).squeeze(1)
            # 中文注释：计算 slot mapping = 物理 block 起始地址 + block 内偏移
            sm = block_ids * block_size + (new_positions_1d % block_size)
            # 中文注释：超出模型最大长度的位置填 padding
            sm.masked_fill_(exceeds, PADDING_SLOT_ID)
            buf = self._slot_mapping_buffer_for(gid)
            buf[:batch_size].copy_(sm)
            # 中文注释：填充多余的位置为 PADDING_SLOT_ID
            if input_batch_size > batch_size:
                buf[batch_size:input_batch_size].fill_(PADDING_SLOT_ID)
            self._per_group_slot_mappings[gid] = buf[:batch_size]
        return positions

    def build_per_group_and_layer_attn_metadata(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        draft_index: int = 0,
    ) -> tuple[list[object], dict[str, object]]:
        """中文注释：构建每个注意力组和每层的注意力元数据。
        覆盖父类方法以支持多 KV 缓存组：
        - 对于有独立 block table 的组，复制 common_attn_metadata 并替换为该组的元数据
        - 对于没有独立元数据的组，直接使用通用元数据

        流程：
        1. 遍历所有 draft 注意力组
        2. 对有独立 block table 的组，创建元数据副本并替换 block table 和 slot mapping
        3. 通过 metadata builder 构建该组的注意力元数据
        4. 将元数据关联到组内所有层

        参数：
            common_attn_metadata: 通用注意力元数据
            draft_index: draft step 索引

        返回：
            (per_group_attn_metadata, per_layer_attn_metadata)
            - per_group_attn_metadata: 每个注意力组的元数据列表
            - per_layer_attn_metadata: 每层名称到元数据的映射字典
        """
        per_group_attn_metadata: list[object] = []
        per_layer_attn_metadata: dict[str, object] = {}
        # The proposer always works in unpadded shape. Per-group block tables
        # registered via set_per_group_attn_metadata are stored at the model
        # runner's padded shape; slice them to match cm's num_reqs.
        num_reqs = common_attn_metadata.num_reqs
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        for attn_group in self.draft_attn_groups:
            gid = attn_group.kv_cache_group_id
            if gid in self._per_group_block_tables:
                # 中文注释：对于有独立 block table 的 KV 缓存组，
                # 复制 common_attn_metadata 并替换为该组的 block table 和 slot mapping。
                # 使用 copy() 创建浅副本，避免修改原始元数据。
                cm = copy(common_attn_metadata)
                cm.block_table_tensor = self._per_group_block_tables[gid][:num_reqs]
                if gid in self._per_group_slot_mappings:
                    sm = self._per_group_slot_mappings[gid]
                    if sm.shape[0] >= num_actual_tokens:
                        sm = sm[:num_actual_tokens]
                    cm.slot_mapping = sm
            else:
                # 中文注释：没有独立元数据的组，直接使用通用元数据
                cm = common_attn_metadata
            # 中文注释：通过 metadata builder 构建该组的注意力元数据
            attn_metadata = attn_group.get_metadata_builder().build_for_drafting(
                common_attn_metadata=cm,
                draft_index=draft_index,
            )
            per_group_attn_metadata.append(attn_metadata)
            for layer_name in attn_group.layer_names:
                per_layer_attn_metadata[layer_name] = attn_metadata
        return per_group_attn_metadata, per_layer_attn_metadata

    def _maybe_share_lm_head(self, target_language_model: torch.nn.Module) -> None:
        """Step3.5 MTP uses the lm_head stored in each MTP layer."""

        # The base MTP path shares target lm_head into shared_head.head.
        # Step3.5 checkpoints carry per-MTP-layer shared_head weights.
        # 中文注释：Step3.5 MTP 不需要共享 target model 的 lm_head。
        # 与 EAGLE 不同，Step3.5 的每个 MTP 层自带独立的 lm_head 权重，
        # checkpoint 中已包含这些权重，无需额外共享操作。
        return

    def _create_draft_vllm_config(self) -> VllmConfig:
        """中文注释：创建 draft model 的 VllmConfig。
        覆盖父类方法，使用 Step3.5 的 draft model 配置和量化配置。
        """
        base = super()._create_draft_vllm_config()
        return replace(
            base,
            model_config=self.draft_model_config,
            quant_config=get_draft_quant_config(base),
        )

    def validate_same_kv_cache_group(self, kv_cache_config: KVCacheConfig) -> None:
        """Step3.5 MTP draft layers may span multiple KV cache groups."""
        # 中文注释：覆盖父类的校验方法。
        # EAGLE 要求所有 draft 层在同一个 KV 缓存组中，但 Step3.5 的 MTP 层
        # 可能跨越多个 KV 缓存组，因此跳过此校验。
        return

    def initialize_attn_backend(
        self,
        kv_cache_config: KVCacheConfig,
        kernel_block_sizes: list[int] | None = None,
    ) -> None:
        """中文注释：初始化 draft model 的注意力后端。
        覆盖父类方法，支持 Step3.5 的多 KV 缓存组架构。

        流程：
        1. 获取所有注意力层的引用
        2. 构建 layer -> gid（KV 缓存组 id）和 layer -> KVCacheSpec 的映射
        3. 处理 KV 共享层（kv_sharing_target_layer_name）
        4. 按（后端类型, gid）分组创建 AttentionGroup
        5. 为每个组创建 metadata builder
        6. 设置主 KV 缓存组和 block_size

        参数：
            kv_cache_config: KV 缓存配置
            kernel_block_sizes: 每个 KV 缓存组的 kernel block 大小
        """
        all_attn_layers = get_layers_from_vllm_config(
            self.vllm_config,
            AttentionLayerBase,  # type: ignore[type-abstract]
        )

        # 中文注释：构建层到 KV 缓存组 id 和 KVCacheSpec 的映射
        layer_to_gid: dict[str, int] = {}
        layer_to_spec: dict[str, KVCacheSpec] = {}
        for gid, group in enumerate(kv_cache_config.kv_cache_groups):
            group_spec = group.kv_cache_spec
            for layer_name in group.layer_names:
                layer_to_gid[layer_name] = gid
                if isinstance(group_spec, UniformTypeKVCacheSpecs):
                    if layer_name in group_spec.kv_cache_specs:
                        layer_to_spec[layer_name] = group_spec.kv_cache_specs[
                            layer_name
                        ]
                    else:
                        # 中文注释：处理 KV 共享层。
                        # 如果当前层声明了 kv_sharing_target_layer_name，
                        # 则复用目标层的 KVCacheSpec，避免重复分配 KV 缓存。
                        target_layer_name = getattr(
                            all_attn_layers.get(layer_name),
                            "kv_sharing_target_layer_name",
                            None,
                        )
                        if (
                            target_layer_name
                            and target_layer_name in group_spec.kv_cache_specs
                        ):
                            layer_to_spec[layer_name] = group_spec.kv_cache_specs[
                                target_layer_name
                            ]
                        else:
                            layer_to_spec[layer_name] = group_spec
                else:
                    layer_to_spec[layer_name] = group_spec

        # 中文注释：按（注意力后端类型, KV 缓存组 id）分组创建 AttentionGroup。
        # 相同后端和相同 KV 组的层共享同一个 AttentionGroup，共用元数据构建器。
        attention_groups: dict[tuple[tuple[str, str], int], AttentionGroup] = {}
        for layer_name in sorted(self._draft_attn_layer_names):
            if layer_name not in layer_to_spec:
                continue
            attn_layer = all_attn_layers[layer_name]
            attn_backend = attn_layer.get_attn_backend()
            spec = layer_to_spec[layer_name]
            gid = layer_to_gid[layer_name]
            group_key = (attn_backend.full_cls_name(), gid)

            if group_key not in attention_groups:
                kernel_block_size = (
                    kernel_block_sizes[gid]
                    if kernel_block_sizes is not None and gid < len(kernel_block_sizes)
                    else None
                )
                attn_group = AttentionGroup(
                    backend=attn_backend,
                    layer_names=[layer_name],
                    kv_cache_spec=spec,
                    kv_cache_group_id=gid,
                )
                attn_group.create_metadata_builders(
                    self.vllm_config,
                    self.device,
                    kernel_block_size=kernel_block_size,
                )
                attention_groups[group_key] = attn_group
            else:
                attention_groups[group_key].layer_names.append(layer_name)

        # 中文注释：设置主 KV 缓存组和 block_size。
        # 主组是第一个 draft 注意力组所属的 KV 缓存组。
        self.draft_attn_groups = list(attention_groups.values())
        if self.draft_attn_groups:
            self.kv_cache_gid = self.draft_attn_groups[0].kv_cache_group_id
            self.block_size = (
                self.draft_attn_groups[0]
                .get_metadata_builder()
                .kv_cache_spec.block_size
            )
        else:
            self.kv_cache_gid = 0
            self.block_size = kv_cache_config.kv_cache_groups[
                0
            ].kv_cache_spec.block_size

    def _sample_draft_tokens_for_step(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        spec_step_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """中文注释：为指定 draft step 从 hidden states 中采样 draft token。
        覆盖父类方法，使用 Step3.5 的逐 step 计算 logits 接口。

        流程：
        1. 如果未启用概率性 draft 或全部是 greedy 采样：
           - 使用 argmax 获取最可能的 token（快速路径）
           - 如果启用 local argmax reduction，直接从模型获取 top token
        2. 否则，从 logits 中采样（支持 temperature、top-k、top-p 等）

        参数：
            hidden_states: draft model 的 hidden states
            sampling_metadata: 采样配置（temperature、top-k 等）
            spec_step_idx: 当前 draft step 索引

        返回：
            (draft_token_ids, draft_probs)
            - draft_token_ids: 采样得到的 draft token id
            - draft_probs: 概率分布（仅在概率性采样时非 None）
        """
        if not self._enable_probabilistic_draft_probs or sampling_metadata.all_greedy:
            if self.use_local_argmax_reduction:
                return self.model.get_top_tokens(hidden_states), None
            logits = self.model.compute_logits(
                hidden_states, spec_step_idx=spec_step_idx
            )
            return logits.argmax(dim=-1), None

        logits = self.model.compute_logits(hidden_states, spec_step_idx=spec_step_idx)
        return self._sample_from_logits(logits, sampling_metadata)

    def propose(
        self,
        target_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        next_token_ids: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        common_attn_metadata: CommonAttentionMetadata,
        sampling_metadata: SamplingMetadata,
        mm_embed_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        num_rejected_tokens_gpu: torch.Tensor | None = None,
        slot_mappings: dict[str, torch.Tensor]
        | list[dict[str, torch.Tensor]]
        | None = None,
    ) -> torch.Tensor:
        """中文注释：Step3.5 MTP 的主提议方法，生成多步 draft token。

        整体流程：
        1. set_inputs_first_pass: 准备第一轮输入（token ids、positions、hidden states）
        2. build_per_group_and_layer_attn_metadata: 构建多 KV 组的注意力元数据
        3. 第一轮前向传播：用 target hidden states 驱动 draft model，生成第一步 draft token
        4. 后续步循环：用上一步的 draft token 作为输入，逐层自回归生成更多 draft token
        5. 将所有步的 draft token 堆叠返回

        与 EAGLE 的关键区别：
        - 支持多 KV 缓存组（每组独立的 block table 和 slot mapping）
        - 使用 Step3.5 的逐 step 计算 logits 接口
        - 不共享 target model 的 lm_head

        参数：
            target_token_ids: target model 处理的 token id
            target_positions: target token 的位置
            target_hidden_states: target model 输出的 hidden states
            next_token_ids: target model 采样的下一个 token（bonus token）
            token_indices_to_sample: 采样索引
            common_attn_metadata: 通用注意力元数据
            sampling_metadata: 采样配置
            mm_embed_inputs: 多模态嵌入输入
            num_rejected_tokens_gpu: 被拒绝的 token 数量
            slot_mappings: slot mapping 字典

        返回：
            draft_token_ids: 形状为 [batch_size, num_speculative_tokens] 的 draft token tensor
        """
        self._last_draft_probs = None
        batch_size = common_attn_metadata.batch_size()

        # 中文注释：步骤 1 - 准备第一轮输入
        # 设置 target hidden states、positions、slot mapping 等元数据
        num_tokens, token_indices_to_sample, common_attn_metadata = (
            self.set_inputs_first_pass(
                target_token_ids=target_token_ids,
                next_token_ids=next_token_ids,
                target_positions=target_positions,
                target_hidden_states=target_hidden_states,
                token_indices_to_sample=token_indices_to_sample,
                cad=common_attn_metadata,
                num_rejected_tokens_gpu=num_rejected_tokens_gpu,
            )
        )

        # 中文注释：步骤 2 - 构建多 KV 缓存组的注意力元数据
        # Step3.5 可能有多个 KV 缓存组，每组需要独立的注意力元数据
        per_group_attn_metadata, per_layer_attn_metadata = (
            self.build_per_group_and_layer_attn_metadata(common_attn_metadata)
        )

        # 中文注释：步骤 3 - 确定 batch 执行模式和 padding 大小
        cudagraph_runtime_mode, num_input_tokens, num_tokens_across_dp = (
            self._determine_batch_execution_and_padding(num_tokens)
        )

        # 中文注释：步骤 4 - 构建模型输入并运行第一轮前向传播
        model_kwargs, slot_mapping_size = self.build_model_inputs_first_pass(
            num_tokens, num_input_tokens, mm_embed_inputs
        )
        model_kwargs["spec_step_idx"] = 0

        with set_forward_context(
            per_layer_attn_metadata,
            self.vllm_config,
            num_tokens=num_input_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            slot_mapping=self._get_slot_mapping(
                slot_mapping_size, common_attn_metadata.slot_mapping
            ),
        ):
            ret_hidden_states = self.model(**model_kwargs)
            if not self.model_returns_tuple():
                last_hidden_states = ret_hidden_states
                hidden_states = last_hidden_states
            else:
                last_hidden_states, hidden_states = ret_hidden_states

        # 中文注释：提取需要采样位置的 hidden states
        sample_hidden_states = last_hidden_states[token_indices_to_sample]

        # 中文注释：快速路径 - 如果只有 1 个 speculative token 或启用并行草稿，
        # 直接采样并返回，无需多步循环
        if self.num_speculative_tokens == 1 or self.parallel_drafting:
            draft_token_ids, draft_probs = self._sample_draft_tokens_for_step(
                sample_hidden_states, sampling_metadata, spec_step_idx=0
            )
            if draft_probs is not None:
                self._last_draft_probs = draft_probs.view(
                    -1, self.num_speculative_tokens, draft_probs.shape[-1]
                ).contiguous()
            return draft_token_ids.view(-1, self.num_speculative_tokens)

        # 中文注释：多步自回归路径 - 准备后续步的输入
        if self.uses_mrope:
            positions = self.mrope_positions[:, token_indices_to_sample]
        else:
            positions = self.positions[token_indices_to_sample]
        hidden_states = hidden_states[token_indices_to_sample]

        if self.constant_draft_positions:
            self.positions[:batch_size] = positions

        # 中文注释：采样第一步 draft token
        draft_token_ids, draft_probs = self._sample_draft_tokens_for_step(
            sample_hidden_states, sampling_metadata, spec_step_idx=0
        )
        draft_probs_list = None if draft_probs is None else [draft_probs]

        # 中文注释：校验注意力元数据类型是否支持多步投机
        if self.allowed_attn_types is not None:
            for group_md in per_group_attn_metadata:
                if not isinstance(group_md, self.allowed_attn_types):
                    raise ValueError(
                        f"Unsupported attention metadata type for speculative "
                        "decoding with num_speculative_tokens > 1: "
                        f"{type(group_md)}. Supported types are: "
                        f"{self.allowed_attn_types}"
                    )

        draft_token_ids_list = [draft_token_ids]

        # 中文注释：后续步使用 decode 模式（每请求 1 个 token）
        cudagraph_runtime_mode, input_batch_size, batch_size_across_dp = (
            self._determine_batch_execution_and_padding(batch_size)
        )

        # 中文注释：更新注意力元数据为 decode 模式
        common_attn_metadata.num_actual_tokens = batch_size
        common_attn_metadata.max_query_len = 1
        common_attn_metadata.query_start_loc = self.arange[: batch_size + 1]
        common_attn_metadata.query_start_loc_cpu = torch.from_numpy(
            self.token_arange_np[: batch_size + 1]
        ).clone()

        if self.num_speculative_tokens > 1 and num_rejected_tokens_gpu is not None:
            common_attn_metadata.seq_lens -= num_rejected_tokens_gpu
            common_attn_metadata._seq_lens_cpu = None
            common_attn_metadata._num_computed_tokens_cpu = None

        block_size = self.block_size
        assert block_size > 0, "block_size has not been initialized."
        # 中文注释：步骤 5 - 多步自回归循环，逐 token 生成 draft
        # 每一步将上一步的 draft token 作为输入，运行 draft model 前向传播，
        # 采样得到下一步的 draft token
        for token_index in range(self.num_speculative_tokens - 1):
            spec_step_idx = token_index + 1
            input_ids = draft_token_ids_list[-1].int()

            # 中文注释：更新 positions 和各 KV 组的 slot mapping
            if not self.constant_draft_positions:
                positions = self._update_positions_dependent_metadata(
                    positions,
                    common_attn_metadata,
                    batch_size,
                    input_batch_size,
                    block_size,
                )

            # 中文注释：重建注意力元数据（positions 变化后 slot mapping 需要更新）
            if not self.constant_draft_positions or token_index == 0:
                _, per_layer_attn_metadata = (
                    self.build_per_group_and_layer_attn_metadata(
                        common_attn_metadata, draft_index=spec_step_idx
                    )
                )

            self.input_ids[:batch_size] = input_ids
            self.hidden_states[:batch_size] = hidden_states
            if self.supports_mm_inputs:
                self.inputs_embeds[:batch_size] = self.model.embed_input_ids(input_ids)

                input_ids = None
                inputs_embeds = self.inputs_embeds[:input_batch_size]
            else:
                input_ids = self.input_ids[:input_batch_size]
                inputs_embeds = None

            model_kwargs = {
                "input_ids": input_ids,
                "positions": self._get_positions(input_batch_size),
                "inputs_embeds": inputs_embeds,
                "spec_step_idx": spec_step_idx,
            }
            if self.pass_hidden_states_to_model:
                model_kwargs["hidden_states"] = self.hidden_states[:input_batch_size]

            # 中文注释：运行 draft model 前向传播
            with set_forward_context(
                per_layer_attn_metadata,
                self.vllm_config,
                num_tokens=input_batch_size,
                num_tokens_across_dp=batch_size_across_dp,
                cudagraph_runtime_mode=cudagraph_runtime_mode,
                slot_mapping=self._get_slot_mapping(input_batch_size),
            ):
                ret_hidden_states = self.model(**model_kwargs)
                if not self.model_returns_tuple():
                    last_hidden_states = ret_hidden_states
                    hidden_states = ret_hidden_states
                else:
                    last_hidden_states, hidden_states = ret_hidden_states

            hidden_states = hidden_states[:batch_size]
            # 中文注释：从当前步的 hidden states 采样 draft token
            draft_token_ids, draft_probs = self._sample_draft_tokens_for_step(
                last_hidden_states[:batch_size],
                sampling_metadata,
                spec_step_idx=spec_step_idx,
            )
            if draft_probs is not None:
                assert draft_probs_list is not None
                draft_probs_list.append(draft_probs)
            draft_token_ids_list.append(draft_token_ids)

        # 中文注释：步骤 6 - 将所有步的 draft token 堆叠为 [batch_size, num_speculative_tokens]
        draft_token_ids = torch.stack(draft_token_ids_list, dim=1)
        if draft_probs_list is not None:
            self._last_draft_probs = torch.stack(draft_probs_list, dim=1).contiguous()
        return draft_token_ids
