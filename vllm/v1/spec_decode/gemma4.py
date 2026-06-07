# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gemma4 MTP (Multi-Token Prediction) proposer for speculative decoding.

The Gemma4 assistant model runs all decoder layers per draft step
(producing one token), and all its attention layers share KV cache
with the target model via cross-model KV sharing.
"""
# ==============================================================================
# Gemma4 推测解码提案器 (Proposer)
#
# 本模块实现了基于 Gemma4 MTP（Multi-Token Prediction）模型的推测解码提案器。
# 核心设计思路：
#   1. Gemma4 MTP 模型作为草稿模型（draft model），在每个解码步骤中预测一个
#      draft token，但它的所有注意力层与目标模型（target model）共享 KV cache。
#      这意味着不需要为草稿模型单独分配 KV cache 显存。
#   2. Gemma4 的特殊之处在于它有多种注意力类型（sliding attention 和 full attention），
#      各自的 head dimension 不同（256 vs 512），因此需要多个 KV cache group，
#      每个 group 使用不同的 block table。
#   3. 草稿模型的所有 draft step 都从同一个位置（目标模型的最后一个位置）开始预测，
#      即 positions 不会在 step 之间递增（constant_draft_positions=True）。
#   4. 支持 centroids（聚类中心）CUDA graph 加速采样，用于快速 greedy sampling。
#
# 整体流程：
#   load_model() -> 加载草稿模型并设置 KV sharing
#   -> propose() 被 GPUModelRunner 调用，执行草稿模型 forward
#   -> _greedy_sample() 从 hidden states 中采样 draft tokens
# ==============================================================================

from collections import defaultdict
from copy import copy

import torch
import torch.nn as nn

from vllm.config import VllmConfig, get_layers_from_vllm_config, replace
from vllm.logger import init_logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer
from vllm.v1.worker.utils import AttentionGroup

logger = init_logger(__name__)


# Gemma4Proposer：Gemma4 MTP 推测解码提案器的核心类。
# 继承自 SpecDecodeBaseProposer，重写了 KV sharing、多 group 注意力、
# centroids 采样等 Gemma4 特有的逻辑。
class Gemma4Proposer(SpecDecodeBaseProposer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ):
        super().__init__(
            vllm_config,
            device,
            # 中文注释：Gemma4 MTP 模型需要接收目标模型的 hidden states 作为输入，
            # 因此设置 pass_hidden_states_to_model=True。
            pass_hidden_states_to_model=True,
            runner=runner,
        )
        # All draft steps predict from the same position (the last
        # target-model position), so positions and seq_lens must not
        # advance between steps.
        # 中文注释：Gemma4 MTP 的所有 draft step 都从同一个位置（目标模型最后一个
        # 位置）开始预测，positions 在各 step 之间不会递增。这与 EAGLE 等模型不同，
        # EAGLE 的每个 draft step 会逐步推进位置。
        self.constant_draft_positions = True

        # Per-group block tables for multi-group KV cache models.
        # Populated by gpu_model_runner during _prepare_inputs.
        # 中文注释：Gemma4 有多个 KV cache group（sliding attention + full attention），
        # 每个 group 有自己的 block table。此字典按 group id 存储各自的 block table，
        # 由 GPUModelRunner 在准备输入时填充。
        self._per_group_block_tables: dict[int, torch.Tensor] = {}

        # Centroids CUDA graphs — populated in load_model if centroids
        # masking is active. _centroids_sizes is pre-sorted for fast
        # lookup in _greedy_sample.
        # 中文注释：centroids CUDA graph 相关数据结构，用于加速 greedy sampling。
        # _centroids_sizes 按大小排序，在 _greedy_sample 中查找最合适的预捕获图。
        # centroids 机制将 hidden states 通过聚类中心映射到词表，避免完整的 lm_head 计算。
        self._centroids_sizes: list[int] = []
        self._centroids_graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self._centroids_inputs: dict[int, torch.Tensor] = {}
        self._centroids_outputs: dict[int, torch.Tensor] = {}

    def set_per_group_block_table(self, gid: int, block_table: torch.Tensor) -> None:
        """设置指定 KV cache group 的 block table。

        Args:
            gid: KV cache group 的索引（0=sliding attention, 1=full attention 等）
            block_table: 该 group 对应的物理 block table tensor
        """
        # 中文注释：由 GPUModelRunner 在 _prepare_inputs 阶段调用，
        # 将每个 KV cache group 的 block table 传递给 proposer，
        # 以便后续构建注意力元数据时使用正确的 block table。
        self._per_group_block_tables[gid] = block_table

    def model_returns_tuple(self) -> bool:
        # forward() returns (draft_hidden_states, backbone_hidden_states).
        # The proposer uses draft_hidden_states for compute_logits and
        # backbone_hidden_states for the hidden-state feedback buffer.
        # 中文注释：Gemma4 MTP 的 forward 返回一个二元组：
        #   - draft_hidden_states：草稿模型自身的 hidden states，用于 compute_logits
        #   - backbone_hidden_states：目标模型的 hidden states，用于反馈给下一轮
        return True

    def build_per_group_and_layer_attn_metadata(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        draft_index: int = 0,
    ) -> tuple[list[object], dict[str, object]]:
        """Build attention metadata using the correct block table per group.

        Gemma4 has multiple KV cache groups (sliding vs full attention)
        with different block tables.  The base class receives a single
        common_attn_metadata whose block_table belongs to one group.
        We swap in the correct block table for each draft attention group.
        """
        # 中文注释：为每个 KV cache group 构建独立的注意力元数据。
        #
        # 流程说明：
        # 1. 遍历所有 draft attention group（例如 sliding attention group、full attention group）
        # 2. 对于每个 group，检查是否有专用的 block table（_per_group_block_tables）
        # 3. 如果有，复制 common_attn_metadata 并替换其 block_table_tensor；
        #    如果没有，直接使用原始 common_attn_metadata
        # 4. 调用该 group 的 metadata builder 构建 draft 阶段的注意力元数据
        # 5. 同时建立 layer_name -> attn_metadata 的映射，方便后续按层查找
        per_group_attn_metadata: list[object] = []
        per_layer_attn_metadata: dict[str, object] = {}
        for attn_group in self.draft_attn_groups:
            gid = attn_group.kv_cache_group_id
            if gid in self._per_group_block_tables:
                # 中文注释：浅拷贝 common_attn_metadata 并替换 block table，
                # 避免修改原始对象影响其他 group。
                cm = copy(common_attn_metadata)
                cm.block_table_tensor = self._per_group_block_tables[gid]
            else:
                cm = common_attn_metadata
            # 中文注释：使用 group 专属的 metadata builder 构建 draft 注意力元数据
            attn_metadata = attn_group.get_metadata_builder().build_for_drafting(
                common_attn_metadata=cm, draft_index=draft_index
            )
            per_group_attn_metadata.append(attn_metadata)
            for layer_name in attn_group.layer_names:
                per_layer_attn_metadata[layer_name] = attn_metadata
        return per_group_attn_metadata, per_layer_attn_metadata

    def _greedy_sample(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """从 hidden states 中进行 greedy sampling，返回预测的 token ids。

        如果启用了 centroids 机制，优先使用预捕获的 CUDA graph 加速采样；
        否则回退到基类的默认 greedy sampling。
        """
        # 中文注释：centroids 采样快速路径。
        # 流程：
        # 1. 如果 _centroids_sizes 非空（即 centroids 已启用），尝试找到
        #    一个 >= 当前 token 数 T 的预捕获 CUDA graph
        # 2. 将 hidden states 拷贝到静态输入 buffer，replay CUDA graph
        # 3. 返回 CUDA graph 的输出（截取前 T 个 token）
        # 4. 如果没有合适的 CUDA graph 大小，直接调用 get_top_tokens（无 CUDA graph）
        # 5. 如果 centroids 未启用，回退到基类的默认 _greedy_sample
        if self._centroids_sizes:
            T = hidden_states.shape[0]
            for size in self._centroids_sizes:
                if size >= T:
                    # 中文注释：将当前 batch 的 hidden states 拷贝到预分配的静态 buffer 中
                    self._centroids_inputs[size][:T].copy_(hidden_states)
                    # 中文注释：replay 预捕获的 CUDA graph，避免 kernel launch 开销
                    self._centroids_graphs[size].replay()
                    return self._centroids_outputs[size][:T].clone()
            # 中文注释：没有匹配的 CUDA graph 大小，直接计算
            return self.model.get_top_tokens(hidden_states)
        return super()._greedy_sample(hidden_states)

    def _setup_centroids_cuda_graphs(self) -> None:
        """Capture CUDA graphs for centroids get_top_tokens at key sizes."""
        # 中文注释：预捕获 centroids get_top_tokens 的 CUDA graph。
        #
        # 为什么需要 CUDA graph？
        #   centroids 采样通过 masked_embedding 将 hidden states 映射到词表中
        #   少量候选 token（聚类中心），然后做 argmax。每次推理时 token 数不同，
        #   但我们可以预捕获若干常用大小（1,2,4,8,16,32,64）的 CUDA graph，
        #   在 _greedy_sample 中选择最接近的图进行 replay，从而消除 kernel launch 开销。
        #
        # 流程：
        # 1. 获取 masked_embedding 层和 lm_head 权重
        # 2. 对每个目标大小，预热 3 次（warmup），然后捕获 CUDA graph
        # 3. 将 graph、静态输入、静态输出分别存储
        # 4. 对大小排序，便于 _greedy_sample 中快速查找
        masked_emb = self.model.masked_embedding
        lm_head_weight = self.model._get_full_lm_head_weight()

        for size in [1, 2, 4, 8, 16, 32, 64]:
            # 中文注释：创建静态输入 buffer，大小为 [size, hidden_size]
            static_input = torch.zeros(
                size,
                masked_emb.hidden_size,
                dtype=self.dtype,
                device=self.device,
            )
            # 中文注释：预热 3 次，确保所有 lazy initialization 完成
            for _ in range(3):
                masked_emb.get_top_tokens(static_input, lm_head_weight)
            torch.accelerator.synchronize()

            # 中文注释：捕获 CUDA graph，后续 replay 时直接复用
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                static_output = masked_emb.get_top_tokens(
                    static_input,
                    lm_head_weight,
                )
            self._centroids_graphs[size] = g
            self._centroids_inputs[size] = static_input
            self._centroids_outputs[size] = static_output

        # 中文注释：对捕获的图大小排序，_greedy_sample 中按顺序查找第一个 >= T 的大小
        self._centroids_sizes = sorted(self._centroids_graphs)
        logger.info(
            "Gemma4 MTP: captured centroids CUDA graphs for sizes %s.",
            self._centroids_sizes,
        )

    def _create_draft_vllm_config(self) -> VllmConfig:
        """Preserve the target's forced TRITON_ATTN backend for draft layers.

        Gemma4 forces TRITON_ATTN due to heterogeneous head dimensions
        (head_dim=256 sliding, global_head_dim=512 full). The base class
        resets attention_config.backend to None for draft models, causing
        sliding layers to fall back to FLASH_ATTN which cannot handle
        KV-shared cache. Override to carry the target's backend through.
        """
        # 中文注释：覆盖基类方法，将目标模型的 attention backend 传递给草稿模型配置。
        #
        # 为什么需要这样做？
        #   Gemma4 因为混合了不同 head dimension 的注意力层（sliding=256, full=512），
        #   强制使用 TRITON_ATTN 作为 attention backend。但基类在创建草稿模型配置时
        #   会将 backend 重置为 None，导致 sliding attention 层回退到 FLASH_ATTN，
        #   而 FLASH_ATTN 不支持 KV-shared cache（KV 共享缓存）。
        #   因此这里必须保留目标模型的 backend 设置。
        base = super()._create_draft_vllm_config()
        target_backend = self.vllm_config.attention_config.backend
        if target_backend is not None:
            base = replace(
                base,
                attention_config=replace(
                    base.attention_config,
                    backend=target_backend,
                ),
            )
        return base

    def _maybe_share_lm_head(self, target_language_model: nn.Module) -> None:
        """Gemma4 MTP always keeps its own draft-dim lm_head.

        The draft model's lm_head operates in draft hidden_size (e.g. 256),
        which differs from the target's backbone hidden_size (e.g. 1536).
        Sharing would break compute_logits (and centroids masking when
        use_ordered_embeddings is enabled).
        """
        # 中文注释：覆盖基类方法，不与目标模型共享 lm_head。
        #
        # 为什么不能共享？
        #   Gemma4 MTP 草稿模型的 hidden_size（如 256）与目标模型的 backbone
        #   hidden_size（如 1536）不同，lm_head 权重矩阵的维度也因此不同。
        #   如果共享 lm_head，维度不匹配会导致 compute_logits 报错。
        #   此外，centroids masking（use_ordered_embeddings）也依赖草稿模型
        #   自己的 lm_head 维度。
        logger.info(
            "Gemma4 MTP: keeping draft model's own lm_head (draft_dim != backbone_dim)."
        )

    def load_model(self, target_model: nn.Module) -> None:
        """加载草稿模型并初始化 KV sharing 和 centroids CUDA graph。

        流程：
        1. 记录目标模型的注意力层名称集合（用于后续区分草稿模型独有层）
        2. 调用基类 load_model 加载草稿模型
        3. 设置 Gemma4 特有的 KV sharing：将草稿模型的注意力层映射到目标模型
        4. 如果模型支持 centroids 采样，预捕获 CUDA graph

        Args:
            target_model: 目标模型的 nn.Module 实例
        """
        # 中文注释：步骤1 - 记录目标模型已有的注意力层名称，
        # 后续用于判断哪些层是草稿模型新增的。
        target_attn_layer_names = set(
            get_layers_from_vllm_config(
                self.vllm_config,
                AttentionLayerBase,  # type: ignore[type-abstract]
            ).keys()
        )

        # 中文注释：步骤2 - 调用基类的 load_model，加载草稿模型权重
        super().load_model(target_model)

        # 中文注释：步骤3 - 设置 KV sharing，将草稿层的 KV cache 映射到目标模型层
        self._setup_gemma4_kv_sharing(target_attn_layer_names)

        # 中文注释：步骤4 - 如果模型有 masked_embedding 层（centroids 机制），
        # 预捕获 CUDA graph 以加速 greedy sampling
        if getattr(self.model, "masked_embedding", None) is not None:
            self._setup_centroids_cuda_graphs()

    def validate_same_kv_cache_group(self, kv_cache_config: KVCacheConfig) -> None:
        """Draft layers span multiple KV cache groups (sliding + full
        attention with different head dimensions), so skip the base
        class single-group assertion."""
        # 中文注释：覆盖基类方法，跳过单 group 断言检查。
        # Gemma4 的草稿模型横跨多个 KV cache group（sliding + full attention），
        # 它们的 head dimension 不同（256 vs 512），因此不能用基类的单 group 假设。
        pass

    def initialize_attn_backend(
        self,
        kv_cache_config: KVCacheConfig,
        kernel_block_sizes: list[int] | None = None,
    ) -> None:
        """Create separate AttentionGroup objects per KV cache spec
        so that each head-dim variant gets its own metadata builder."""
        # 中文注释：初始化草稿模型的注意力后端。
        #
        # 与基类不同，Gemma4 有多种 head dimension 的注意力层（sliding=256, full=512），
        # 不能将所有层归为同一个 AttentionGroup。这里按 (backend, kv_cache_spec) 的组合
        # 将层分组，每组创建独立的 metadata builder。
        #
        # 流程：
        # 1. 获取所有注意力层，建立 layer_name -> group_id 和 layer_name -> spec 的映射
        # 2. 对于 KV sharing 的层，查找其共享目标的 spec
        # 3. 按 (backend_cls, spec) 分组，每组创建一个 AttentionGroup
        # 4. 设置默认的 kv_cache_gid 和 block_size

        # 中文注释：步骤1 - 获取所有注意力层信息
        all_attn_layers = get_layers_from_vllm_config(
            self.vllm_config,
            AttentionLayerBase,  # type: ignore[type-abstract]
        )

        # 中文注释：建立 layer -> group_id 和 layer -> kv_cache_spec 的映射
        layer_to_gid: dict[str, int] = {}
        layer_to_spec: dict[str, KVCacheSpec] = {}
        for gid, group in enumerate(kv_cache_config.kv_cache_groups):
            group_spec = group.kv_cache_spec
            for ln in group.layer_names:
                layer_to_gid[ln] = gid
                if isinstance(group_spec, UniformTypeKVCacheSpecs):
                    if ln in group_spec.kv_cache_specs:
                        layer_to_spec[ln] = group_spec.kv_cache_specs[ln]
                    else:
                        # 中文注释：该层可能通过 KV sharing 共享另一层的 spec，
                        # 需要查找共享目标层的 spec
                        tgt = getattr(
                            all_attn_layers.get(ln),
                            "kv_sharing_target_layer_name",
                            None,
                        )
                        if tgt and tgt in group_spec.kv_cache_specs:
                            layer_to_spec[ln] = group_spec.kv_cache_specs[tgt]
                        else:
                            layer_to_spec[ln] = group_spec
                else:
                    layer_to_spec[ln] = group_spec

        # 中文注释：步骤2 - 按 (backend, spec) 分组，每组创建一个 AttentionGroup
        # group_key = (backend类全名, kv_cache_spec)，相同 key 的层共享同一个 metadata builder
        attention_groups: dict[tuple[tuple[str, str], KVCacheSpec], AttentionGroup] = {}
        for layer_name in self._draft_attn_layer_names:
            if layer_name not in layer_to_spec:
                continue
            attn_layer = all_attn_layers[layer_name]
            attn_backend = attn_layer.get_attn_backend()
            spec = layer_to_spec[layer_name]
            gid = layer_to_gid[layer_name]
            group_key = (attn_backend.full_cls_name(), spec)

            if group_key not in attention_groups:
                # 中文注释：新 group，创建 AttentionGroup 并初始化 metadata builder
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
                # 中文注释：已有相同 group，将该层追加到已有 group 的 layer_names 中
                attention_groups[group_key].layer_names.append(layer_name)

        # 中文注释：步骤3 - 保存分组结果，设置默认 block_size
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
        logger.debug("Using block size %d for drafting layers", self.block_size)

    def _setup_gemma4_kv_sharing(
        self,
        target_attn_layer_names: set[str],
    ) -> None:
        """Wire draft layers to share KV with the target model.

        Each draft decoder layer is mapped to the last non-KV-shared
        target layer of the same attention type (sliding or full).
        """
        # 中文注释：设置 Gemma4 草稿模型与目标模型之间的 KV cache 共享。
        #
        # 核心思想：
        #   草稿模型的每个注意力层需要复用目标模型中同类型（sliding/full）的
        #   最后一个非 KV-sharing 层的 KV cache。这样草稿模型不需要额外分配
        #   KV cache 显存，直接读取目标模型已经计算好的 KV。
        #
        # 流程：
        # 1. 读取目标模型的 layer_types（每个层的注意力类型：sliding 或 full）
        # 2. 排除目标模型中已经通过 KV sharing 共享的层，只保留"真实"计算层
        # 3. 按注意力类型分组，记录每种类型的层索引列表
        # 4. 遍历草稿模型的每个层，找到同类型的目标模型层（取最后一个），
        #    设置 kv_sharing_target_layer_name 属性

        # 中文注释：读取草稿模型和目标模型的配置
        draft_config = self.speculative_config.draft_model_config.hf_config
        draft_text_config = draft_config.get_text_config()
        target_config = self.vllm_config.model_config.hf_config
        target_text_config = target_config.get_text_config()
        target_layer_types = getattr(target_text_config, "layer_types", [])

        if not (hasattr(self.model, "model") and hasattr(self.model.model, "layers")):
            return

        # 中文注释：计算目标模型中非 KV-sharing 的层数，
        # num_kv_shared_layers 是目标模型自身已经通过 KV sharing 复用的层数
        target_num_kv_shared = getattr(target_text_config, "num_kv_shared_layers", 0)
        num_non_shared = len(target_layer_types) - target_num_kv_shared
        # 中文注释：按注意力类型（sliding / full）分组，记录各类型的层索引
        type_to_target_indices: dict[str, list[int]] = defaultdict(list)
        for idx, lt in enumerate(target_layer_types[:num_non_shared]):
            type_to_target_indices[lt].append(idx)

        # 中文注释：推断目标模型层名的前缀（如 "model.layers"）
        target_prefix = "model.layers"
        for name in target_attn_layer_names:
            if ".layers." in name:
                target_prefix = name.split(".layers.")[0] + ".layers"
                break

        # 中文注释：遍历草稿模型的每个层，设置 KV sharing 目标
        draft_layer_types = getattr(draft_text_config, "layer_types", [])
        for draft_idx, layer in enumerate(self.model.model.layers):
            if not hasattr(layer, "self_attn"):
                continue
            attn = getattr(layer.self_attn, "attn", None)
            if attn is None:
                continue

            # 中文注释：确定当前草稿层的注意力类型
            draft_layer_type = (
                draft_layer_types[draft_idx]
                if draft_idx < len(draft_layer_types)
                else "full_attention"
            )
            # 中文注释：查找目标模型中同类型的层，取最后一个（最深层）作为共享目标
            candidates = type_to_target_indices.get(draft_layer_type, [])
            if not candidates:
                logger.warning(
                    "No target layer of type '%s' for draft layer %d",
                    draft_layer_type,
                    draft_idx,
                )
                continue

            # 中文注释：设置 KV sharing 目标层名，草稿层将复用该目标层的 KV cache
            target_idx = candidates[-1]
            target_layer_name = f"{target_prefix}.{target_idx}.self_attn.attn"
            attn.kv_sharing_target_layer_name = target_layer_name
            logger.info(
                "Gemma4 MTP: draft layer %d (%s) -> %s",
                draft_idx,
                draft_layer_type,
                target_layer_name,
            )
