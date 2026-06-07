# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
DFlash 投机解码提议器模块

DFlash 是一种基于交叉注意力（cross-attention）的投机解码方法。
与 EAGLE 等基于自回归 draft model 的方法不同，DFlash 的核心思想是：

1. 将 target model 的 hidden states 作为 K/V（上下文）
2. 将 next token embedding + mask token embedding 作为 Q（查询）
3. 通过一次前向传播并行生成所有 speculative token 的预测

关键特点：
- 并行草稿生成（parallel drafting）：所有 speculative token 在一次前向传播中生成，
  而非像 EAGLE 那样逐 token 自回归生成
- 交叉注意力机制：Q 来自查询 token，K/V 来自 target model 的 hidden states
- 支持非因果注意力：DFlash 默认使用非因果注意力模式
- 上下文 KV 预计算：先将 target hidden states 投影为 KV 并存入缓存，
  再用查询 token 进行注意力计算

数据流：
  target_hidden_states -> precompute_and_store_context_kv (K/V) -> 查询 token 注意力 -> draft tokens
"""

from dataclasses import replace
from typing import Any

import torch
from typing_extensions import override

from vllm.config import VllmConfig
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.triton_utils import triton
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer
from vllm.v1.spec_decode.utils import copy_and_expand_dflash_inputs_kernel

logger = init_logger(__name__)


class DFlashProposer(SpecDecodeBaseProposer):
    # 中文注释：DFlash 投机解码提议器。
    # 继承 SpecDecodeBaseProposer，使用交叉注意力机制并行生成 draft token。
    # 与 EAGLE 的逐 token 自回归不同，DFlash 在一次前向传播中同时预测所有 speculative token。

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ):
        assert vllm_config.speculative_config is not None
        assert vllm_config.speculative_config.method == "dflash"
        super().__init__(
            vllm_config=vllm_config,
            device=device,
            # 中文注释：DFlash 需要将 target model 的 hidden states 传给 draft model
            pass_hidden_states_to_model=True,
            runner=runner,
        )

        # Only next_token_ids and mask tokens are query tokens, all other context is K/V
        # 中文注释：DFlash 中的 token 分为两类：
        #   - Query tokens: next_token_ids（bonus token）+ mask tokens（speculative positions）
        #   - Context tokens: target model 的 hidden states，投影为 K/V
        # 每个请求有 1 个 bonus token + num_speculative_tokens 个 mask token 作为 query
        self.max_query_tokens = self.max_batch_size * (1 + self.num_speculative_tokens)
        # Positions covers both context states + query states
        self.max_positions = self.max_num_tokens + self.max_query_tokens

        # Separate context buffers to keep query buffer addresses stable for CUDA graphs
        # 中文注释：DFlash 将 context 和 query 的元数据存放在不同的 buffer 中。
        # 这样做的原因是 CUDA graph 要求输入 tensor 地址稳定，
        # 将不参与 CUDA graph 的 context 处理与参与 CUDA graph 的 query 处理分离。
        self._context_slot_mapping_buffer = torch.zeros(
            self.max_num_tokens,
            dtype=torch.int64,
            device=device,
        )
        self._slot_mapping_buffer = torch.zeros(
            self.max_query_tokens,
            dtype=torch.int64,
            device=device,
        )
        self._context_positions_buffer = torch.zeros(
            self.max_num_tokens,
            dtype=torch.int64,
            device=device,
        )
        self.positions = torch.zeros(
            self.max_query_tokens,
            dtype=torch.int64,
            device=device,
        )

        self.arange = torch.arange(
            self.max_positions + 1, device=device, dtype=torch.int32
        )

        # For DFlash we use the input embeddings to embed the mask token
        # 中文注释：DFlash 使用 input embeddings 来嵌入 mask token，
        # 不需要额外的 parallel_drafting_hidden_state_tensor
        self.parallel_drafting_hidden_state_tensor = None

        # 中文注释：是否使用因果注意力。DFlash 默认使用非因果注意力，
        # 允许 query token 看到所有 context token，而不论位置关系。
        self.dflash_causal = self.dflash_config.get("causal", False)

    @override
    def _create_draft_vllm_config(self) -> VllmConfig:
        """中文注释：创建 draft model 的配置。
        覆盖父类方法，将注意力模式设置为与 dflash_causal 配置一致。
        当 dflash_causal=False 时使用非因果注意力（允许 query 看到所有 context）。
        """
        base = super()._create_draft_vllm_config()
        return replace(
            base,
            attention_config=replace(
                base.attention_config,
                use_non_causal=not self.dflash_causal,
            ),
        )

    @override
    def _warn_if_multimodal(self):
        # Override to allow multimodal inputs since DFlash supports Qwen3.5 models
        # 中文注释：覆盖父类的多模态警告，因为 DFlash 支持多模态输入（如 Qwen3.5 模型）
        pass

    @override
    def set_inputs_first_pass(
        self,
        target_token_ids: torch.Tensor,
        next_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        cad: CommonAttentionMetadata,
        num_rejected_tokens_gpu: torch.Tensor | None,
    ) -> tuple[int, torch.Tensor, CommonAttentionMetadata]:
        """中文注释：设置 DFlash 第一轮前向传播的输入。

        DFlash 的交叉注意力机制：
        - K/V 来自 target model 的 hidden states（上下文 token）
        - Q 来自 query embeddings（bonus token + mask token）

        流程：
        1. 计算 query token 数量：每请求 1 个 bonus + num_speculative_tokens 个 mask
        2. 保存 target hidden states 供后续 KV 预计算使用
        3. 启动 Triton 融合 kernel，一次性生成 input_ids、positions、slot_mapping、
           token_indices_to_sample（避免多次 CPU-GPU 同步）
        4. 构建新的 CommonAttentionMetadata，反映 query token 的注意力配置

        参数：
            target_token_ids: target model 本轮处理的 token id
            next_token_ids: target model 采样得到的下一个 token（bonus token）
            target_positions: target token 的位置编码
            target_hidden_states: target model 输出的 hidden states
            token_indices_to_sample: 采样索引（DFlash 中会重新生成）
            cad: 通用注意力元数据
            num_rejected_tokens_gpu: 被拒绝的 token 数量（拒绝采样模式）

        返回：
            (num_query_total, token_indices_to_sample, new_cad)
        """
        # DFlash cross-attention: context K/V from target hidden states,
        # Q from query embeddings (bonus + mask tokens).
        batch_size = cad.batch_size()
        num_context = target_token_ids.shape[0]
        # 中文注释：每个请求的 query token 数 = 1 个 bonus token + N 个 mask token
        num_query_per_req = 1 + self.num_speculative_tokens
        num_query_total = batch_size * num_query_per_req

        # Store for build_model_inputs_first_pass to use
        # 中文注释：保存 context token 数量和 hidden states，
        # 供后续 build_model_inputs_first_pass 中的 KV 预计算使用
        self._dflash_num_context = num_context

        # We don't need to copy into a buffer here since the context preprocessing
        # does not run in a CUDA graph
        self._dflash_hidden_states = target_hidden_states

        token_indices_to_sample = torch.empty(
            batch_size * self.num_speculative_tokens,
            dtype=torch.int32,
            device=self.device,
        )

        # Launch fused triton kernel for input_ids, positions, slot_mapping,
        # and token_indices_to_sample
        # 中文注释：启动融合 Triton kernel，在一次 kernel 调用中完成：
        #   1. 将 next_token_ids 复制到 input_ids（作为 bonus token）
        #   2. 生成 context 和 query 的 positions
        #   3. 生成 context 和 query 的 slot_mapping（通过 block_table 计算物理地址）
        #   4. 生成 token_indices_to_sample（标记哪些位置需要采样）
        # 融合 kernel 避免了多次 GPU kernel 启动和潜在的 CPU-GPU 同步。
        max_ctx_per_req = cad.max_query_len
        max_tokens_per_req = max_ctx_per_req + num_query_per_req
        BLOCK_SIZE = min(256, triton.next_power_of_2(max_tokens_per_req))
        num_blocks = triton.cdiv(max_tokens_per_req, BLOCK_SIZE)
        grid = (batch_size, num_blocks)

        has_num_rejected = num_rejected_tokens_gpu is not None
        copy_and_expand_dflash_inputs_kernel[grid](
            # Inputs
            next_token_ids_ptr=next_token_ids,
            target_positions_ptr=target_positions,
            # Outputs
            out_input_ids_ptr=self.input_ids,
            out_context_positions_ptr=self._context_positions_buffer,
            out_query_positions_ptr=self.positions,
            out_context_slot_mapping_ptr=self._context_slot_mapping_buffer,
            out_query_slot_mapping_ptr=self._slot_mapping_buffer,
            out_token_indices_ptr=token_indices_to_sample,
            # Block table
            block_table_ptr=cad.block_table_tensor,
            block_table_stride=cad.block_table_tensor.stride(0),
            # Metadata
            query_start_loc_ptr=cad.query_start_loc,
            num_rejected_tokens_ptr=(
                num_rejected_tokens_gpu if has_num_rejected else 0
            ),
            # Scalars
            parallel_drafting_token_id=self.parallel_drafting_token_id,
            block_size=self.block_size,
            num_query_per_req=num_query_per_req,
            num_speculative_tokens=self.num_speculative_tokens,
            total_input_tokens=num_context,
            BLOCK_SIZE=BLOCK_SIZE,
            HAS_NUM_REJECTED=has_num_rejected,
        )

        query_slot_mapping = self._slot_mapping_buffer[:num_query_total]
        new_query_start_loc = self.arange[: batch_size + 1] * num_query_per_req

        # In padded mode, cad.seq_lens includes rejected tokens. Subtract
        # them so attention only sees the valid prefix of context states.
        # 中文注释：在 padded 模式下，seq_lens 包含了被拒绝的 token。
        # 减去它们以确保注意力只看到有效的 context 前缀。
        effective_seq_lens = cad.seq_lens
        if has_num_rejected:
            effective_seq_lens = effective_seq_lens - num_rejected_tokens_gpu

        # Skip num_rejected_tokens (GPU-only); overestimating is fine here.
        # 中文注释：构建新的 CommonAttentionMetadata，反映 query token 的注意力配置。
        # seq_lens 增加 num_query_per_req 以包含新加入的 query token。
        new_seq_lens_cpu_upper_bound = (
            cad.seq_lens_cpu_upper_bound + num_query_per_req
            if cad.seq_lens_cpu_upper_bound is not None
            else None
        )
        new_cad = CommonAttentionMetadata(
            query_start_loc=new_query_start_loc,
            seq_lens=effective_seq_lens + num_query_per_req,
            query_start_loc_cpu=(
                torch.from_numpy(self.token_arange_np[: batch_size + 1]).clone()
                * num_query_per_req
            ),
            _seq_lens_cpu=None,
            _num_computed_tokens_cpu=None,
            seq_lens_cpu_upper_bound=new_seq_lens_cpu_upper_bound,
            num_reqs=cad.num_reqs,
            num_actual_tokens=num_query_total,
            max_query_len=num_query_per_req,
            max_seq_len=cad.max_seq_len + num_query_per_req,
            block_table_tensor=cad.block_table_tensor,
            slot_mapping=query_slot_mapping,
            causal=self.dflash_causal,
        )

        return num_query_total, token_indices_to_sample, new_cad

    @override
    @torch.inference_mode()
    def dummy_run(
        self,
        num_tokens: int,
        use_cudagraphs: bool = True,
        is_graph_capturing: bool = False,
        slot_mappings: dict[str, torch.Tensor] | None = None,
    ) -> None:
        """
        Key differences to default dummy_run:
        - Only one forward pass due to parallel drafting
        - DFlash uses context states as unpadded metadata, so hidden_states will
        use the unpadded num_tokens instead of num_input_tokens
        - max_query_tokens is quite small, DFlash only sees spec tokens as queries
        - Multimodal inputs are not currently supported
        """
        # 中文注释：DFlash 的 dummy run（用于显存 profiling 和 CUDA graph 捕获）。
        # 与默认 dummy run 的区别：
        #   1. 只需一次前向传播（因为 DFlash 是并行草稿生成）
        #   2. context 预计算不走 CUDA graph，单独执行 KV 投影
        #   3. max_query_tokens 较小，DFlash 只把 spec token 作为 query

        # 中文注释：DFlash 中 query token 数量远小于 EAGLE，只需 batch_size * (1 + spec_tokens)
        num_query_tokens = min(num_tokens, self.max_query_tokens)
        cudagraph_runtime_mode, num_input_tokens, num_tokens_across_dp = (
            self._determine_batch_execution_and_padding(
                num_query_tokens, use_cudagraphs=use_cudagraphs
            )
        )

        # Slot mapping sized to num_input_tokens (query only), matching
        # the K/V tensor size from the model forward.  Context KVs are
        # pre-inserted separately and don't flow through the model.
        # 中文注释：slot mapping 只用于 query token（context KV 已通过预计算单独插入缓存）
        if (
            self._draft_attn_layer_names
            and slot_mappings is not None
            and next(iter(self._draft_attn_layer_names)) in slot_mappings
        ):
            slot_mapping_dict = self._get_slot_mapping(num_input_tokens)
        else:
            slot_mapping_dict = slot_mappings or {}

        # Context and query positions use separate buffers; no copy needed.
        context_positions = self._context_positions_buffer[:num_tokens]
        # Context states will be passed directly to the precomputation without
        # going through the buffer, since no CUDA graph is used for the precomputation.
        # For the dummy run, we use the dummy buffer.
        context_states = self.hidden_states[:num_tokens]

        # Run the KV projection (GEMM + norms + RoPE) for memory profiling,
        # 中文注释：步骤 1 - 预计算 context KV 并存入缓存。
        # 包括 GEMM 投影、LayerNorm、RoPE 位置编码等操作。
        # 这一步不走 CUDA graph，用于显存 profiling。
        self.model.precompute_and_store_context_kv(context_states, context_positions)
        # 中文注释：步骤 2 - 用 query token 运行前向传播，同样用于显存 profiling。
        with set_forward_context(
            None,
            self.vllm_config,
            num_tokens=num_input_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            slot_mapping=slot_mapping_dict,
        ):
            self.model(
                input_ids=self.input_ids[:num_input_tokens],
                positions=self._get_positions(num_input_tokens),
                inputs_embeds=None,
            )

    @override
    def build_model_inputs_first_pass(
        self,
        num_tokens: int,
        num_input_tokens: int,
        mm_embed_inputs: tuple[list[torch.Tensor], torch.Tensor] | None,
    ) -> tuple[dict[str, Any], int]:
        """中文注释：构建 DFlash 第一轮前向传播的模型输入。

        流程：
        1. 将 target hidden states 预计算为 KV 并直接插入缓存（通过 precompute_and_store_context_kv）
        2. 返回 query token 的输入字典（input_ids、positions）

        DFlash 的关键设计：context KV 的预计算独立于 query 的前向传播，
        预计算不走 CUDA graph（因为 context 长度可变），query 前向传播走 CUDA graph。

        参数：
            num_tokens: query token 总数
            num_input_tokens: 填充后的输入 token 数（用于 CUDA graph）
            mm_embed_inputs: 多模态嵌入输入（DFlash 中不使用）

        返回：
            (model_kwargs_dict, num_input_tokens)
        """
        # Context and query positions/slots were written to separate
        # buffers by the kernel — no copy needed.
        num_context = self._dflash_num_context

        # Pre-insert context KVs directly into cache
        # 中文注释：将 target model 的 hidden states 投影为 KV 并插入缓存。
        # 这一步完成了 DFlash 交叉注意力中的 K/V 准备：
        #   - hidden states -> Linear projection -> K, V
        #   - 应用 LayerNorm 和 RoPE 位置编码
        #   - 将 K, V 写入对应 slot 的 KV cache
        self.model.precompute_and_store_context_kv(
            self._dflash_hidden_states,  # Shape is already [num_context, hidden_size]
            self._context_positions_buffer[:num_context],
            self._context_slot_mapping_buffer[:num_context],
        )
        # 中文注释：返回 query token 的输入。query token 的 input_ids 由
        # Triton kernel 在 set_inputs_first_pass 中生成（包含 bonus token 和 mask token）。
        return (
            dict(
                input_ids=self.input_ids[:num_input_tokens],
                positions=self._get_positions(num_input_tokens),
                inputs_embeds=None,
            ),
            num_input_tokens,
        )

    @override
    def build_per_group_and_layer_attn_metadata(
        self, cad: CommonAttentionMetadata, draft_index: int = 0
    ) -> tuple[list[object], dict[str, object]]:
        """中文注释：构建每个注意力组和每层的注意力元数据。
        覆盖父类方法，增加了对非因果注意力的校验：
        当 DFlash 配置为非因果模式时，所有注意力层必须支持非因果注意力。
        """
        per_group, per_layer = super().build_per_group_and_layer_attn_metadata(
            cad, draft_index
        )
        if not self.dflash_causal:
            # Require all layers to support non-causal attention when required by DFlash
            for layer_name, attn_metadata in per_layer.items():
                assert getattr(attn_metadata, "causal", None) is False, (
                    f"Attention metadata for layer {layer_name} does not have"
                    " non-causal support, which is required for DFlash."
                    " Consider using a different attention backend, e.g FlashAttention."
                )
        return per_group, per_layer

    @override
    def _get_eagle3_use_aux_hidden_state_from_config(self):
        """中文注释：获取是否使用辅助 hidden state 的配置。
        DFlash 从自己的配置中读取，而非使用 EAGLE3 的默认值。
        """
        return self.dflash_config.get("use_aux_hidden_state", True)

    @property
    def dflash_config(self):
        """中文注释：获取 DFlash 特有配置。
        从 draft model 的 HuggingFace 配置中读取 dflash_config 字段。
        如果不存在则返回空字典，使用默认行为。
        """
        return getattr(self.draft_model_config.hf_config, "dflash_config", None) or {}
