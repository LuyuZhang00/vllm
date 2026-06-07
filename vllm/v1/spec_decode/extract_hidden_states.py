# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ==============================================================================
# ExtractHiddenStates 推测解码提案器 (Proposer)
#
# 本模块实现了一个特殊的"推测解码"提案器，其核心目的不是真正地推测未来 token，
# 而是在目标模型推理过程中提取并缓存中间层的 hidden states。
#
# 核心设计思路：
#   1. ExtractHiddenStatesModel 本身是一个轻量模型，它将目标模型的中间层
#      hidden states 写入 KV cache（通过 cache-only attention，不做实际计算）。
#   2. 这些缓存的 hidden states 可以用于后续的 KV transfer（跨节点 KV 缓存传输）
#      或其他需要中间层特征的场景。
#   3. 该 proposer 的 propose() 方法直接返回目标模型已采样的 token 作为 "draft" token，
#      因此 draft 总是与 target 匹配（100% 接受率），本质是一个 passthrough 操作。
#   4. 仅支持 num_speculative_tokens=1 的配置。
#
# 整体流程：
#   load_model() -> 加载 ExtractHiddenStatesModel，识别其注意力层
#   -> propose() 被 GPUModelRunner 调用，接收目标模型的 hidden states
#   -> 将 hidden states 通过模型 forward 写入 KV cache
#   -> 返回 sampled_token_ids 作为 "draft" tokens（始终与 target 匹配）
#
# 与 KV transfer 的关系：
#   缓存的 hidden states 通过 KV connector 机制在不同推理节点之间传输，
#   实现 disaggregated prefill/decode 架构中的 KV 缓存迁移。
# ==============================================================================

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from vllm.config import CUDAGraphMode, VllmConfig, get_layers_from_vllm_config
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.model_loader import get_model
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.v1.attention.backend import AttentionMetadataBuilder, CommonAttentionMetadata
from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher
from vllm.v1.utils import CpuGpuBuffer
from vllm.v1.worker.dp_utils import coordinate_batch_across_dp
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import KVCacheConfig

# 中文注释：padding slot 的特殊 ID，用于标识 KV cache 中不需要写入的位置。
# 在 slot mapping 中，-1 表示该 token 不需要写入 KV cache（padding 用途）。
PADDING_SLOT_ID = -1


# ExtractHiddenStatesProposer：提取目标模型中间层 hidden states 的提案器。
# 与 EAGLE、Gemma4 等真正的草稿模型不同，这个 proposer 的核心目的不是
# 推测未来 token，而是将目标模型的中间层 hidden states 缓存到 KV cache 中，
# 用于 KV transfer 等场景。
class ExtractHiddenStatesProposer:
    def __init__(self, vllm_config: VllmConfig, device):
        assert vllm_config.speculative_config is not None

        # 中文注释：仅支持 num_speculative_tokens=1，即只缓存一层 hidden state
        assert vllm_config.speculative_config.num_speculative_tokens == 1
        if vllm_config.speculative_config.disable_padded_drafter_batch:
            raise ValueError(
                "disable_padded_drafter_batch is not supported with "
                "extract_hidden_states method"
            )
        self.vllm_config = vllm_config
        self.device = device
        self.dtype = vllm_config.model_config.dtype
        # 中文注释：data parallel rank，用于多 DP 场景下的 batch 协调
        self.dp_rank = vllm_config.parallel_config.data_parallel_rank

        # Model and attention layer tracking (initialized in load_model)
        # 中文注释：以下属性在 load_model() 中初始化
        self.model: nn.Module | None = None
        # 中文注释：ExtractHiddenStatesModel 的注意力层名称列表（通常只有一个 cache-only 层）
        self.attn_layer_names: list[str] = []
        # 中文注释：注意力元数据构建器，用于为 cache-only attention 构建元数据
        self.attn_metadata_builder: AttentionMetadataBuilder | None = None
        # 中文注释：KV cache group 索引，标识该模型的注意力层属于哪个 KV cache group
        self.kv_cache_gid: int = -1

        # Maximum number of tokens for buffers
        # 中文注释：计算 buffer 的最大 token 数，等于 max_batched_tokens + max_num_seqs。
        # 额外的 max_num_seqs 是为了容纳 speculative decoding 带来的额外 token。
        max_batch_size = vllm_config.scheduler_config.max_num_seqs
        self.max_num_tokens = (
            vllm_config.scheduler_config.max_num_batched_tokens + max_batch_size
        )

        # 中文注释：备份的 next token ids（CPU/GPU 双缓冲），用于当采样结果无效时
        # 提供回退 token。例如请求被丢弃时，使用请求状态中最后一个 token 作为替代。
        self.backup_next_token_ids = CpuGpuBuffer(
            max_batch_size,
            dtype=torch.int32,
            pin_memory=is_pin_memory_available(),
            device=device,
            with_numpy=True,
        )

        # 中文注释：从 draft model 配置中读取需要提取的 hidden state 层 ID。
        # eagle_aux_hidden_state_layer_ids 指定了目标模型的哪些中间层的
        # hidden states 需要被提取和缓存。
        self.hf_config = vllm_config.speculative_config.draft_model_config.hf_config
        layer_ids = getattr(self.hf_config, "eagle_aux_hidden_state_layer_ids", None)
        if not layer_ids:
            raise ValueError(
                "eagle_aux_hidden_state_layer_ids must be set in the draft "
                "model config for extract_hidden_states method"
            )
        # 中文注释：hidden states 缓存 buffer，形状为 [max_tokens, num_layers, hidden_size]。
        # 用于存储目标模型中间层的 hidden states，在 propose() 中传给模型 forward。
        self.num_hidden_states = len(layer_ids)
        self.hidden_size = vllm_config.model_config.get_hidden_size()
        self.hidden_states = torch.zeros(
            (self.max_num_tokens, self.num_hidden_states, self.hidden_size),
            dtype=self.dtype,
            device=device,
        )
        # 中文注释：CUDA graph 调度器，决定是否使用 CUDA graph 以及使用哪种模式
        self.cudagraph_dispatcher = CudagraphDispatcher(self.vllm_config)

        # 中文注释：slot mapping 的 GPU buffer，用于 cache-only attention 层
        # 将 hidden states 写入 KV cache 时的地址映射
        self._slot_mapping_buffer = torch.zeros(
            self.max_num_tokens, dtype=torch.int64, device=device
        )

    def propose(
        self,
        sampled_token_ids: torch.Tensor,
        target_hidden_states: list[torch.Tensor],
        common_attn_metadata: CommonAttentionMetadata,
        slot_mappings: dict[str, torch.Tensor]
        | list[dict[str, torch.Tensor]]
        | None = None,
    ) -> torch.Tensor:
        """Propose draft tokens by calling the ExtractHiddenStatesModel model.

        The ExtractHiddenStatesModel caches the hidden states in the KV cache
        without performing actual attention computation. This allows us to
        extract and store hidden states for later use (e.g., KV transfer).

        This proposer doesn't actually perform speculation - it returns the
        sampled tokens as "draft" tokens, ensuring they always verify (match).
        The main purpose is to cache hidden states, not to speculate.

        Args:
            sampled_token_ids: Sampled token IDs from the target model
            target_hidden_states: List of hidden state tensors from target model
                                (one per aux hidden state layer)
            common_attn_metadata: Attention metadata
            slot_mappings: Slot mappings for KV cache (unused, provided for
                          interface compatibility)

        Returns:
            Tuple of:
                - Draft tokens matching sampled tokens, shape [batch_size, 1]
                - KV connector output (if KV transfer is active), else None
        """
        # 中文注释：propose() 是该提案器的核心方法，被 GPUModelRunner 在每轮解码时调用。
        #
        # 整体流程：
        #   1. 将目标模型的多层 hidden states 堆叠成 [num_tokens, num_layers, hidden_size]
        #   2. 拷贝到预分配的 buffer 中
        #   3. 构建 cache-only attention 的元数据
        #   4. 确定 batch 执行模式（是否使用 CUDA graph、是否需要 DP 协调）
        #   5. 在 forward context 下执行模型 forward，将 hidden states 写入 KV cache
        #   6. 返回 sampled_token_ids 作为 "draft" tokens（始终与 target 匹配）
        assert self.model is not None and isinstance(target_hidden_states, list)

        # target_hidden_states is a list of tensors (one per layer)
        # Each tensor has shape [num_tokens, hidden_size]
        # Stack to shape: [num_tokens, num_hidden_states, hidden_size]
        # 中文注释：将目标模型的多层 hidden states 从 list 合并为单个 tensor。
        # 例如如果有 2 个中间层，shape 从 [2, num_tokens, hidden_size] 变为
        # [num_tokens, 2, hidden_size]。
        stacked_hidden_states = torch.stack(target_hidden_states, dim=1)
        num_tokens = stacked_hidden_states.shape[0]

        # Copy hidden states to buffer
        # 中文注释：将堆叠后的 hidden states 拷贝到预分配的 GPU buffer 中
        self.hidden_states[:num_tokens] = stacked_hidden_states

        # 中文注释：构建 cache-only attention 的元数据。
        # cache-only attention 不做实际的 Q*K 计算，只将 hidden states 写入 KV cache。
        assert self.attn_metadata_builder is not None
        attn_metadata = self.attn_metadata_builder.build_for_drafting(
            common_attn_metadata=common_attn_metadata, draft_index=0
        )

        # We assume all cache-only layers belong to the same KV cache group,
        # thus using the same attention metadata.
        # 中文注释：为每个注意力层分配相同的元数据。
        # 假设所有 cache-only 层属于同一个 KV cache group。
        per_layer_attn_metadata = {}
        for layer_name in self.attn_layer_names:
            per_layer_attn_metadata[layer_name] = attn_metadata

        # 中文注释：确定 batch 执行模式：
        #   - cudagraph_runtime_mode：是否使用 CUDA graph（NONE / PIECEWISE / FULL）
        #   - num_input_tokens：经过 padding 后的 token 数（可能用于 CUDA graph 对齐）
        #   - num_tokens_across_dp：各 DP rank 的 token 数（用于跨 rank 协调）
        cudagraph_runtime_mode, num_input_tokens, num_tokens_across_dp = (
            self._determine_batch_execution_and_padding(num_tokens)
        )
        if num_tokens_across_dp is not None:
            num_tokens_across_dp[self.dp_rank] = num_input_tokens

        # 中文注释：在 forward context 下执行模型 forward。
        # set_forward_context 设置了注意力元数据、CUDA graph 模式、slot mapping 等，
        # 这些信息会在模型 forward 过程中被各层读取。
        # 模型 forward 将 hidden states 通过 cache-only attention 写入 KV cache。
        with set_forward_context(
            per_layer_attn_metadata,
            self.vllm_config,
            num_tokens=num_input_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            slot_mapping=self._get_slot_mapping(
                num_input_tokens, common_attn_metadata.slot_mapping
            ),
        ):
            self.model(
                hidden_states=self.hidden_states[:num_input_tokens],
            )

        # Return the sampled tokens as "draft" tokens
        # Shape: [batch_size, 1] to match num_speculative_tokens=1
        # On decode steps with spec tokens, sampled_token_ids may have
        # shape [batch_size, 2] (target + spec verification); slice to
        # return only the target-sampled column.
        # 中文注释：直接返回目标模型已采样的 token 作为 "draft" token。
        # 由于这不是真正的推测解码（只是缓存 hidden states），draft token
        # 总是与 target token 匹配，接受率为 100%。
        # 注意：在有 speculative token 验证的步骤中，sampled_token_ids 可能是
        # [batch_size, 2]（target + spec），只取第一列。
        return sampled_token_ids[:, :1]

    def _get_slot_mapping(
        self,
        num_tokens: int,
        slot_mapping: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return slot_mapping dict for cache-only attention layers.

        If slot_mapping is provided, copies it into the buffer first.
        """
        # 中文注释：为 cache-only attention 层构建 slot mapping 字典。
        #
        # slot mapping 的含义：
        #   slot mapping 将每个 token 的逻辑位置映射到 KV cache 中的物理 slot 位置。
        #   cache-only attention 层在 forward 时会根据 slot mapping 将 hidden states
        #   写入 KV cache 的对应位置，后续 KV transfer 可以从这些位置读取。
        #
        # 流程：
        #   1. 如果外部提供了 slot_mapping，拷贝到内部 buffer
        #   2. 如果 padding 后的 token 数 > 实际 token 数，多余部分填充 PADDING_SLOT_ID
        #   3. 返回 layer_name -> slot_mapping 的字典（所有层共享同一份 slot mapping）
        if slot_mapping is not None:
            num_actual = slot_mapping.shape[0]
            self._slot_mapping_buffer[:num_actual].copy_(slot_mapping)
            if num_tokens > num_actual:
                # 中文注释：padding 部分填充 -1，表示不需要写入 KV cache
                self._slot_mapping_buffer[num_actual:num_tokens].fill_(PADDING_SLOT_ID)

        # 中文注释：所有 cache-only 层共享同一份 slot mapping view
        view = self._slot_mapping_buffer[:num_tokens]
        return {name: view for name in self.attn_layer_names}

    def _determine_batch_execution_and_padding(
        self,
        num_tokens: int,
        use_cudagraphs: bool = True,
    ) -> tuple[CUDAGraphMode, int, torch.Tensor | None]:
        # 中文注释：确定 batch 的执行模式和 padding 策略。
        #
        # 这个方法决定三个关键参数：
        #   1. cudagraph_runtime_mode：使用哪种 CUDA graph 模式（NONE/PIECEWISE/FULL）
        #   2. num_input_tokens：经过 padding 后的 token 数（用于 CUDA graph 对齐或 DP 协调）
        #   3. num_tokens_across_dp：各 DP rank 的 token 数（仅在 DP > 1 时非 None）
        #
        # 流程：
        #   1. 通过 cudagraph_dispatcher 根据 num_tokens 选择合适的 CUDA graph 模式
        #   2. 如果使用 DP（data parallel），与所有 rank 协调 token 数，
        #      确保所有 rank 使用相同的 batch 大小（或至少兼容的 CUDA graph 模式）
        #   3. 如果 DP 协调后 padding 数变化，重新 dispatch 以更新 batch descriptor

        # 中文注释：步骤1 - 根据 token 数 dispatch 选择 CUDA graph 模式
        cudagraph_mode, batch_desc = self.cudagraph_dispatcher.dispatch(
            num_tokens,
            valid_modes=({CUDAGraphMode.NONE} if not use_cudagraphs else None),
        )
        num_tokens_padded = batch_desc.num_tokens

        # Extra coordination when running data-parallel since we need to
        # coordinate across ranks
        # TODO(Flechman): support DBO ubatching
        # 中文注释：步骤2 - DP 协调（仅在 data_parallel_size > 1 时执行）
        should_ubatch, num_tokens_across_dp = False, None
        if self.vllm_config.parallel_config.data_parallel_size > 1:
            # 中文注释：与所有 DP rank 同步，确定每个 rank 的 token 数和 CUDA graph 模式
            should_ubatch, num_tokens_across_dp, synced_cudagraph_mode = (
                coordinate_batch_across_dp(
                    num_tokens_unpadded=num_tokens,
                    parallel_config=self.vllm_config.parallel_config,
                    allow_microbatching=False,
                    num_tokens_padded=num_tokens_padded,
                    cudagraph_mode=cudagraph_mode.value,
                )
            )
            assert not should_ubatch, (
                "DBO ubatching not implemented for extract_hidden_states"
            )

            # Extract DP-synced values
            # 中文注释：步骤3 - 使用 DP 同步后的值重新 dispatch
            if num_tokens_across_dp is not None:
                dp_rank = self.dp_rank
                num_tokens_padded = int(num_tokens_across_dp[dp_rank].item())
                # Re-dispatch with DP padding so we have the correct
                # batch_descriptor
                cudagraph_mode, batch_desc = self.cudagraph_dispatcher.dispatch(
                    num_tokens_padded,
                    valid_modes={CUDAGraphMode(synced_cudagraph_mode)},
                )
                # Assert to make sure the agreed upon token count is correct
                # otherwise num_tokens_across_dp will no-longer be valid
                assert batch_desc.num_tokens == num_tokens_padded
                num_tokens_across_dp[dp_rank] = num_tokens_padded

        return cudagraph_mode, num_tokens_padded, num_tokens_across_dp

    def initialize_cudagraph_keys(self, cudagraph_mode: CUDAGraphMode) -> None:
        """Initialize cudagraph dispatcher keys.

        Only supports PIECEWISE cudagraphs (via mixed_mode).
        Should be called after adjust_cudagraph_sizes_for_spec_decode.
        """
        # 中文注释：初始化 CUDA graph dispatcher 的 key。
        #
        # CUDA graph dispatcher 需要预先知道可能的 batch 大小，以便预分配静态 buffer。
        # 这里根据目标模型的 CUDA graph 模式决定 proposer 使用哪种模式：
        #   - 如果目标模型使用 PIECEWISE 或 FULL CUDA graph，proposer 使用 PIECEWISE
        #   - 否则使用 NONE（eager 模式）
        #
        # 注意：此方法应在 adjust_cudagraph_sizes_for_spec_decode 之后调用，
        # 以确保 buffer 大小已正确调整。
        assert self.vllm_config.speculative_config is not None
        if (
            not self.vllm_config.speculative_config.enforce_eager
            and cudagraph_mode.mixed_mode()
            in [CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL]
        ):
            proposer_cudagraph_mode = CUDAGraphMode.PIECEWISE
        else:
            proposer_cudagraph_mode = CUDAGraphMode.NONE

        self.cudagraph_dispatcher.initialize_cudagraph_keys(proposer_cudagraph_mode)

    @torch.inference_mode()
    def dummy_run(
        self,
        num_tokens: int,
        use_cudagraphs: bool = True,
        is_graph_capturing: bool = False,
        slot_mappings: dict[str, torch.Tensor] | None = None,
    ) -> None:
        """执行一次 dummy forward pass，用于预热或 CUDA graph 捕获。

        在模型初始化和 CUDA graph 捕获阶段调用，使用零值 hidden states
        执行一次 forward，确保所有 lazy 初始化完成、CUDA graph 被正确捕获。

        Args:
            num_tokens: 本次 dummy run 的 token 数
            use_cudagraphs: 是否允许使用 CUDA graph
            is_graph_capturing: 当前是否处于 graph 捕获阶段
            slot_mappings: 可选的 slot mapping 字典
        """
        assert self.model is not None, "Model must be initialized before dummy_run"
        # 中文注释：确定 batch 执行模式和 padding 策略
        cudagraph_runtime_mode, num_input_tokens, num_tokens_across_dp = (
            self._determine_batch_execution_and_padding(
                num_tokens, use_cudagraphs=use_cudagraphs
            )
        )

        if num_tokens_across_dp is not None:
            num_tokens_across_dp[self.dp_rank] = num_input_tokens

        # Use our own slot mapping buffer during cudagraph capture.
        # 中文注释：在 CUDA graph 捕获时，使用内部的 slot mapping buffer，
        # 因为 CUDA graph 需要静态地址，不能使用动态变化的外部 slot mapping。
        if (
            self.attn_layer_names
            and slot_mappings is not None
            and self.attn_layer_names[0] in slot_mappings
        ):
            slot_mapping_dict = self._get_slot_mapping(num_input_tokens)
        else:
            slot_mapping_dict = slot_mappings or {}

        # 中文注释：在 forward context 下执行 dummy forward
        with set_forward_context(
            None,
            self.vllm_config,
            num_tokens=num_input_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            slot_mapping=slot_mapping_dict,
        ):
            self.model(
                hidden_states=self.hidden_states[:num_input_tokens],
            )

    def _build_attn_metadata_builder(
        self, draft_attn_layers: dict[str, AttentionLayerBase]
    ) -> AttentionMetadataBuilder:
        """Build the attention metadata builder from draft attention layers."""
        # 中文注释：从草稿模型的注意力层构建注意力元数据构建器。
        #
        # 流程：
        #   1. 从草稿注意力层中取出第一个层
        #   2. 获取该层的 attention backend（如 FlashAttention、Triton 等）
        #   3. 使用 backend 的 builder 类创建元数据构建器实例
        #   4. 构建器后续用于 build_for_drafting() 生成 cache-only attention 的元数据
        if not draft_attn_layers:
            raise ValueError("No attention layers found for ExtractHiddenStatesModel")
        layer = next(iter(draft_attn_layers.values()))
        attn_backend = layer.get_attn_backend()
        return attn_backend.get_builder_cls()(
            layer.get_kv_cache_spec(self.vllm_config),
            self.attn_layer_names,
            self.vllm_config,
            self.device,
        )

    def prepare_next_token_ids_padded(
        self,
        sampled_token_ids: torch.Tensor,
        requests: dict[str, CachedRequestState],
        gpu_input_batch: InputBatch,
        discard_request_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare next token IDs for speculative decoding.

        Since num_speculative_tokens == 1, sampled_token_ids has shape
        (batch_size, 1). For each request we either use the sampled token
        (if valid and not discarded) or a backup token from the request state.

        Args:
            sampled_token_ids: 目标模型采样的 token IDs，shape [batch_size, 1]
            requests: 所有请求的状态字典
            gpu_input_batch: GPU 上的输入 batch 信息
            discard_request_mask: 标记哪些请求需要被丢弃的 bool mask

        Returns:
            tuple: (next_token_ids, valid_sampled_tokens_count)
                - next_token_ids: 每个请求的下一个 token ID [batch_size]
                - valid_sampled_tokens_count: 每个请求的有效采样 token 数 [batch_size]
        """
        # 中文注释：为每个请求准备下一个 token ID。
        #
        # 流程：
        #   1. 预计算每个请求的备份 token（来自请求状态的最后一个 token）
        #   2. 对于每个请求，检查采样的 token 是否有效（在词表范围内且未被丢弃）
        #   3. 如果有效，使用采样的 token；否则使用备份 token
        #   4. 返回最终的 next_token_ids 和有效采样计数
        #
        # 为什么需要备份 token？
        #   某些请求可能因为采样失败（如 token ID 超出词表范围）或被标记为丢弃
        #   而无法使用采样结果，此时需要用请求中已有的最后一个 token 作为替代，
        #   保证 batch 中所有请求都有合法的 token ID。

        num_reqs = gpu_input_batch.num_reqs

        # Precompute backup token IDs for discarded requests.
        # 中文注释：为每个请求计算备份 token ID（请求状态中最后一个 token）
        num_reqs = gpu_input_batch.num_reqs
        for i in range(num_reqs):
            self.backup_next_token_ids.np[i] = requests[
                gpu_input_batch.req_ids[i]
            ].get_token_id(gpu_input_batch.num_tokens_no_spec[i] - 1)
        self.backup_next_token_ids.copy_to_gpu(num_reqs)
        backup_tokens_gpu = self.backup_next_token_ids.gpu[:num_reqs]

        assert discard_request_mask.dtype == torch.bool

        # With num_speculative_tokens == 1, there is exactly one token
        # 中文注释：提取采样结果的第一列（num_speculative_tokens=1 时只有一列）
        sampled = sampled_token_ids[:, 0]
        # 中文注释：检查采样 token 是否有效（在合法的词表 ID 范围内）
        is_valid = (sampled >= 0) & (sampled < gpu_input_batch.vocab_size)
        valid_sampled_tokens_count = is_valid.to(torch.int32)

        # 中文注释：决定使用采样 token 还是备份 token：
        #   - 使用采样 token 的条件：token 有效 且 请求未被丢弃
        #   - 否则使用备份 token
        use_sampled = is_valid & ~discard_request_mask[:num_reqs]
        next_token_ids = torch.where(
            use_sampled, sampled.to(torch.int32), backup_tokens_gpu
        )

        return next_token_ids, valid_sampled_tokens_count

    def load_model(self, target_model: nn.Module) -> None:
        """Load the ExtractHiddenStatesModel model.

        This method instantiates the ExtractHiddenStatesModel model which is used
        to cache hidden states during speculative decoding. The model uses
        cache-only attention (no computation, just caching KV states).

        Args:
            target_model: The target model (passed for compatibility with
                         EagleProposer interface, but not used here)
        """
        # 中文注释：加载 ExtractHiddenStatesModel 模型。
        #
        # 流程：
        #   1. 记录目标模型已有的注意力层名称（用于后续区分草稿模型新增层）
        #   2. 从 draft_model_config 加载草稿模型权重
        #   3. 通过集合差集识别草稿模型独有的注意力层（应恰好只有一个 cache-only 层）
        #   4. 为该 cache-only 层构建注意力元数据构建器
        #
        # ExtractHiddenStatesModel 的结构：
        #   该模型通常只有一个 cache-only attention 层，不执行实际的注意力计算，
        #   而是将输入的 hidden states 直接写入 KV cache，供后续 KV transfer 使用。

        # Get the target model's attention layers before loading draft model
        # 中文注释：步骤1 - 记录目标模型的注意力层名称集合
        target_attn_layer_names = set(
            get_layers_from_vllm_config(self.vllm_config, AttentionLayerBase).keys()  # type: ignore[type-abstract]
        )

        assert self.vllm_config.speculative_config is not None
        draft_model_config = self.vllm_config.speculative_config.draft_model_config
        from vllm.compilation.backends import set_model_tag

        # 中文注释：步骤2 - 使用 "extract_hidden_states" 标签加载草稿模型，
        # 该标签用于区分编译缓存中的不同模型
        with set_model_tag("extract_hidden_states"):
            self.model = get_model(
                vllm_config=self.vllm_config, model_config=draft_model_config
            )

        # Identify draft model's attention layers (difference from target)
        # 中文注释：步骤3 - 通过集合差集找到草稿模型独有的注意力层
        # （即不在目标模型中的层），这些是 cache-only attention 层
        all_attn_layers = get_layers_from_vllm_config(
            self.vllm_config,
            AttentionLayerBase,  # type: ignore[type-abstract]
        )
        draft_attn_layers = {
            name: layer
            for name, layer in all_attn_layers.items()
            if name not in target_attn_layer_names
        }
        self.attn_layer_names = list(draft_attn_layers.keys())
        # 中文注释：断言草稿模型恰好有一个 cache-only attention 层
        assert len(draft_attn_layers) == 1, (
            "ExtractHiddenStatesModel should have exactly one "
            f"attention layer, found {len(draft_attn_layers)}"
        )
        # 中文注释：步骤4 - 为 cache-only 层构建注意力元数据构建器
        self.attn_metadata_builder = self._build_attn_metadata_builder(
            draft_attn_layers
        )

    def validate_same_kv_cache_group(self, kv_cache_config: KVCacheConfig) -> None:
        """Validate all drafting layers belong to the same KV cache group
        and record the group index for common_attn_metadata selection."""
        # 中文注释：验证草稿模型的 cache-only 层属于某个 KV cache group，
        # 并记录该 group 的索引（kv_cache_gid）。
        #
        # 为什么需要这个验证？
        #   GPUModelRunner 在准备输入时需要知道使用哪个 KV cache group 的
        #   common_attn_metadata。这里通过查找 cache-only 层所属的 group，
        #   将 group 索引保存到 self.kv_cache_gid，后续通过该索引获取
        #   正确的 block table 和 slot mapping。
        assert len(self.attn_layer_names) == 1
        layer = self.attn_layer_names[0]
        for gid, group in enumerate(kv_cache_config.kv_cache_groups):
            if layer in group.layer_names:
                self.kv_cache_gid = gid
                return
        raise ValueError(f"Cache-only layer {layer!r} not in any KV cache group")
