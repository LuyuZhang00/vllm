# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
EAGLE 投机解码器模块。

本模块实现了 EAGLE（Extrapolation Algorithm for Greater Language-model Efficiency）
投机解码器的核心逻辑。

EAGLE 的工作原理：
1. 在目标模型的隐状态（hidden states）基础上，训练一个轻量级的草稿头（draft head）
2. 草稿头利用目标模型的 KV cache 和注意力机制进行自回归预测
3. 通过多步解码一次性生成多个候选 token
4. 使用拒绝采样验证这些候选 token

主要类：
- EagleSpeculator: EAGLE 投机解码器的主类，负责协调整个投机解码流程

主要流程（propose 方法）：
1. 准备输入数据（隐状态、温度、种子等）
2. 执行草稿预填充（draft prefill）：生成第一个候选 token
3. 执行多步解码（multi-step decode）：生成后续候选 token
4. 返回所有候选 token

性能优化：
- 使用 CUDA 图（CUDA Graph）减少 kernel launch 开销
- 支持数据并行（Data Parallel）
- 复用目标模型的注意力元数据
- 使用 Triton 内核进行高效的输入准备
"""

from typing import Any

import torch
import torch.nn as nn

from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.triton_utils import tl, triton
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.attn_utils import (
    build_attn_metadata,
    build_slot_mappings_by_layer,
    init_attn_backend,
)
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.cudagraph_utils import (
    BatchExecutionDescriptor,
    CapturedAttentionState,
    get_uniform_token_count,
)
from vllm.v1.worker.gpu.dp_utils import dispatch_cg_and_sync_dp
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
from vllm.v1.worker.gpu.spec_decode.eagle.cudagraph import (
    DecodeEagleCudaGraphManager,
    PrefillEagleCudaGraphManager,
)
from vllm.v1.worker.gpu.spec_decode.eagle.utils import load_eagle_model

logger = init_logger(__name__)


class EagleSpeculator:
    """
    EAGLE 投机解码器。

    实现了 EAGLE 投机解码算法，利用目标模型的隐状态进行快速的草稿生成。

    主要功能：
    1. 管理 EAGLE 草稿模型的生命周期（加载、初始化、推理）
    2. 协调草稿生成的多步解码流程
    3. 管理 CUDA 图的捕获和重放
    4. 处理多模态输入（如果支持）

    属性:
        vllm_config (VllmConfig): vLLM 全局配置。
        device (torch.device): 计算设备。
        speculative_config: 投机解码配置。
        method (str): 投机解码方法（"eagle", "eagle3", "mtp"）。
        num_speculative_steps (int): 投机解码步数。
        draft_model_config: 草稿模型配置。
        hidden_size (int): 隐状态维度（可能因 HC-multiplexing 而扩大）。
        vocab_size (int): 词汇表大小。
        model: EAGLE 草稿模型实例。
        input_buffers (InputBuffers): 输入缓冲区。
        hidden_states (torch.Tensor): 隐状态缓冲区。
        draft_tokens (torch.Tensor): 草稿 token 缓冲区。
        draft_logits (torch.Tensor | None): 草稿 logits 缓冲区（用于概率采样）。
    """

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        """
        初始化 EAGLE 投机解码器。

        参数:
            vllm_config (VllmConfig): vLLM 全局配置。
            device (torch.device): 计算设备。

        初始化内容:
            1. 提取配置信息（投机解码方法、步数、模型配置等）
            2. 计算隐状态维度（考虑 HC-multiplexing）
            3. 分配预填充缓冲区（隐状态、草稿 token、温度、种子等）
            4. 初始化 CUDA 图管理器（延迟到 init_cudagraph_manager）
        """
        self.vllm_config = vllm_config
        self.device = device

        self.speculative_config = vllm_config.speculative_config
        assert self.speculative_config is not None
        self.method = self.speculative_config.method
        self.num_speculative_steps = self.speculative_config.num_speculative_tokens
        self.draft_model_config = self.speculative_config.draft_model_config

        self.scheduler_config = vllm_config.scheduler_config
        self.max_num_reqs = self.scheduler_config.max_num_seqs
        self.max_num_tokens = self.scheduler_config.max_num_batched_tokens
        self.max_model_len = vllm_config.model_config.max_model_len
        # We need to get the hidden size from the draft model config because
        # the draft model's hidden size can be different from the target model's
        # hidden size (e.g., Llama 3.3 70B).
        # 从草稿模型配置获取隐状态维度，因为草稿模型的隐状态维度可能与目标模型不同
        self.hidden_size = self.draft_model_config.get_hidden_size()
        # Widen for HC-multiplexed residuals (e.g. DeepSeek V4 feeds the MTP
        # draft the target's pre-hc_head (T, hc_mult * hidden_size) residual).
        # Non-HC models default to hc_mult=1 and are unaffected.
        # 扩展隐状态维度以支持 HC-multiplexed 残差（如 DeepSeek V4）
        hc_mult = getattr(self.draft_model_config.hf_config, "hc_mult", 1)
        self.hidden_size = self.hidden_size * hc_mult
        self.vocab_size = self.draft_model_config.get_vocab_size()
        self.dtype = vllm_config.model_config.dtype
        self.use_fp64_gumbel = vllm_config.model_config.use_fp64_gumbel

        # DP configuration
        # 数据并行配置
        self.dp_size = vllm_config.parallel_config.data_parallel_size
        self.dp_rank = vllm_config.parallel_config.data_parallel_rank

        # 分配预填充缓冲区
        self.input_buffers = InputBuffers(
            max_num_reqs=self.max_num_reqs,
            max_num_tokens=self.max_num_tokens,
            device=device,
        )
        self.hidden_states = torch.zeros(
            self.max_num_tokens, self.hidden_size, dtype=self.dtype, device=device
        )
        self.idx_mapping = torch.zeros(
            self.max_num_reqs, dtype=torch.int32, device=device
        )
        self.temperature = torch.zeros(
            self.max_num_reqs, dtype=torch.float32, device=device
        )
        self.seeds = torch.zeros(self.max_num_reqs, dtype=torch.int64, device=device)
        self.draft_tokens = torch.zeros(
            self.max_num_reqs,
            self.num_speculative_steps,
            dtype=torch.int64,
            device=device,
        )
        self.current_draft_step = torch.tensor(0, dtype=torch.int64, device=device)
        self.last_token_indices = torch.zeros(
            self.max_num_reqs, dtype=torch.int64, device=device
        )
        self.arange = torch.arange(
            self.max_num_reqs + 1, dtype=torch.int32, device="cpu"
        )

        # 检查草稿模型是否支持多模态输入
        self.supports_mm_inputs = MULTIMODAL_REGISTRY.supports_multimodal_inputs(
            self.draft_model_config
        )
        if self.supports_mm_inputs:
            self.inputs_embeds = torch.zeros(
                self.max_num_tokens, self.hidden_size, dtype=self.dtype, device=device
            )

        # 草稿 logits 缓冲区（用于概率采样模式）
        self.draft_logits: torch.Tensor | None = None
        if self.speculative_config.draft_sample_method == "probabilistic":
            self.draft_logits = torch.zeros(
                self.max_num_reqs,
                self.num_speculative_steps,
                self.vocab_size,
                dtype=torch.float32,
                device=device,
            )

        # CUDA 图管理器（延迟初始化）
        self.prefill_cudagraph_manager: PrefillEagleCudaGraphManager | None = None
        self.decode_cudagraph_manager: DecodeEagleCudaGraphManager | None = None

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        """
        初始化 CUDA 图管理器。

        为草稿预填充和草稿解码分别创建 CUDA 图管理器。
        预填充和解码共享同一个 CUDA 图内存池，因为它们不会并发执行。

        参数:
            cudagraph_mode (CUDAGraphMode): CUDA 图模式。

        流程:
            1. 创建预填充 CUDA 图管理器
            2. 调整解码的 CUDA 图模式（PIECEWISE 不支持 eagle 草稿解码）
            3. 创建解码 CUDA 图管理器
            4. 共享内存池
        """
        cudagraph_mode = self.vllm_config.compilation_config.cudagraph_mode
        # Initialize cudagraph manager for draft prefill (draft position 0).
        self.prefill_cudagraph_manager = PrefillEagleCudaGraphManager(
            self.vllm_config,
            self.device,
            cudagraph_mode,
            self.num_speculative_steps + 1,
        )

        # PIECEWISE cudagraphs are not supported for eagle draft decodes.
        # PIECEWISE pads num_tokens to the next capture size without padding
        # num_reqs, which can cause attention backends to read past the
        # valid per-request metadata (e.g. FlashInfer's kv_indptr buffer).
        # PIECEWISE 模式不支持 eagle 草稿解码，因为它只填充 token 数量而不填充
        # 请求数量，可能导致注意力后端读取越界
        if cudagraph_mode.decode_mode() == CUDAGraphMode.FULL:
            cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY
        else:
            cudagraph_mode = CUDAGraphMode.NONE

        # Initialize cudagraph manager for draft decodes (draft positions > 0).
        self.decode_cudagraph_manager = DecodeEagleCudaGraphManager(
            self.vllm_config,
            self.device,
            cudagraph_mode,
            decode_query_len=1,
        )
        # Share a single pool between prefill and decode since they never
        # execute concurrently.
        # 预填充和解码共享同一个内存池，因为它们不会并发执行
        self.decode_cudagraph_manager.pool = self.prefill_cudagraph_manager.pool

    def load_model(self, target_model: nn.Module) -> None:
        """
        加载 EAGLE 草稿模型。

        加载草稿模型，并识别草稿模型独有的注意力层。

        参数:
            target_model (nn.Module): 目标模型实例。

        流程:
            1. 记录目标模型的注意力层名称
            2. 加载 EAGLE 草稿模型（与目标模型共享部分权重）
            3. 识别草稿模型独有的注意力层（用于单独初始化注意力后端）
        """
        target_attn_layer_names = get_layers_from_vllm_config(
            self.vllm_config,
            AttentionLayerBase,  # type: ignore[type-abstract]
        ).keys()

        self.model = load_eagle_model(target_model, self.vllm_config)

        all_attn_layers = get_layers_from_vllm_config(
            self.vllm_config,
            AttentionLayerBase,  # type: ignore[type-abstract]
        ).keys()
        self.draft_attn_layer_names = set(all_attn_layers) - set(
            target_attn_layer_names
        )

    def set_attn(
        self,
        model_state: ModelState,
        kv_cache_config: KVCacheConfig,
        block_tables: BlockTables,
    ) -> None:
        """
        设置注意力相关的配置和状态。

        初始化草稿模型的注意力后端和 block table。

        参数:
            model_state (ModelState): 模型状态。
            kv_cache_config (KVCacheConfig): KV cache 配置。
            block_tables (BlockTables): Block table 管理器。
        """
        self.model_state = model_state
        self.kv_cache_config = kv_cache_config
        self.attn_groups, _, _ = init_attn_backend(
            kv_cache_config,
            self.vllm_config,
            self.device,
            active_layer_names=self.draft_attn_layer_names,
        )
        self.block_tables = block_tables

    @torch.inference_mode()
    def run_model(
        self,
        num_tokens: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        运行 EAGLE 草稿模型的前向传播。

        执行草稿模型的前向计算，返回最后一层的隐状态和所有层的隐状态。

        参数:
            num_tokens (int): 输入 token 数量。
            attn_metadata (dict | None): 注意力元数据。
            slot_mappings (dict | None): slot 映射。
            num_tokens_across_dp (torch.Tensor | None): 数据并行各 rank 的 token 数量。
            cudagraph_runtime_mode (CUDAGraphMode): CUDA 图运行模式。
            mm_inputs (tuple | None): 多模态输入。

        返回:
            tuple[torch.Tensor, torch.Tensor]:
                - last_hidden_states: 最后一层的隐状态，形状 [num_tokens, hidden_size]。
                - hidden_states: 所有层的隐状态，形状 [num_tokens, hidden_size]。
        """
        batch_descriptor = BatchDescriptor(num_tokens=num_tokens)
        with set_forward_context(
            attn_metadata,
            self.vllm_config,
            num_tokens=num_tokens,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            num_tokens_across_dp=num_tokens_across_dp,
            slot_mapping=slot_mappings,
            batch_descriptor=batch_descriptor,
        ):
            inputs_embeds = None
            if self.supports_mm_inputs:
                # Merge multimodal embeddings with input ids.
                # 合并多模态嵌入和 input ids
                mm_embeds, is_mm_embed = mm_inputs or (None, None)
                num_input_tokens = (
                    is_mm_embed.shape[0] if is_mm_embed is not None else num_tokens
                )
                self.inputs_embeds[:num_input_tokens] = self.model.embed_input_ids(
                    self.input_buffers.input_ids[:num_input_tokens],
                    multimodal_embeddings=mm_embeds,
                    is_multimodal=is_mm_embed,
                )
                inputs_embeds = self.inputs_embeds[:num_tokens]

            ret_hidden_states = self.model(
                input_ids=self.input_buffers.input_ids[:num_tokens],
                positions=self.input_buffers.positions[:num_tokens],
                hidden_states=self.hidden_states[:num_tokens],
                inputs_embeds=inputs_embeds,
            )
        if self.method == "mtp":
            last_hidden_states = ret_hidden_states
            hidden_states = ret_hidden_states
        else:
            last_hidden_states, hidden_states = ret_hidden_states
        return last_hidden_states, hidden_states

    def _sample_draft(
        self,
        logits: torch.Tensor,
        idx_mapping: torch.Tensor,
        pos: torch.Tensor,
        draft_step: torch.Tensor,
        draft_logits: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        从草稿 logits 中采样 token。

        根据配置选择采样方式：
        - 概率采样（probabilistic）：使用 Gumbel-max 技巧进行可微分采样
        - 贪心采样：直接取 argmax

        参数:
            logits (torch.Tensor): 草稿模型输出的 logits。
            idx_mapping (torch.Tensor): 请求索引映射。
            pos (torch.Tensor): 位置索引。
            draft_step (torch.Tensor): 当前草稿步骤。
            draft_logits (torch.Tensor | None): 草稿 logits 缓冲区（用于存储处理后的 logits）。

        返回:
            torch.Tensor: 采样的 token ID，形状 [num_reqs]。
        """
        if draft_logits is not None:
            # NOTE(woosuk): We must add 1 to the positions to match the Gumbel noise
            # used for draft and target sampling.
            # 使用 Gumbel-max 技巧进行概率采样
            # 注意：位置需要加 1 以匹配草稿和目标采样使用的 Gumbel 噪声
            return gumbel_sample(
                logits,
                idx_mapping,
                self.temperature,
                self.seeds,
                pos + 1,
                apply_temperature=True,
                output_processed_logits=draft_logits,
                output_processed_logits_col=draft_step,
                use_fp64=self.use_fp64_gumbel,
            )
        else:
            # 贪心采样：直接取 argmax
            return logits.argmax(dim=-1)

    def prefill(
        self,
        num_reqs: int,
        num_tokens: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        """
        执行草稿预填充（draft prefill）。

        这是 EAGLE 草稿生成的第一步：
        1. 运行草稿模型的前向传播
        2. 从最后一个 token 的隐状态计算 logits
        3. 采样第一个草稿 token
        4. 更新隐状态和位置信息

        参数:
            num_reqs (int): 请求数量。
            num_tokens (int): token 数量。
            attn_metadata (dict | None): 注意力元数据。
            slot_mappings (dict | None): slot 映射。
            num_tokens_across_dp (torch.Tensor | None): 数据并行各 rank 的 token 数量。
            cudagraph_runtime_mode (CUDAGraphMode): CUDA 图运行模式。
            mm_inputs (tuple | None): 多模态输入。
        """
        last_token_indices = self.last_token_indices[:num_reqs]
        pos = self.input_buffers.positions[last_token_indices]
        idx_mapping = self.idx_mapping[:num_reqs]

        last_hidden_states, hidden_states = self.run_model(
            num_tokens,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp=num_tokens_across_dp,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            mm_inputs=mm_inputs,
        )
        # 取最后一个 token 的隐状态用于计算 logits
        sample_hidden_states = last_hidden_states[last_token_indices]
        logits = self.model.compute_logits(sample_hidden_states)

        # 采样第一个草稿 token
        self.draft_tokens[:num_reqs, 0] = self._sample_draft(
            logits,
            idx_mapping,
            pos,
            self.current_draft_step,
            self.draft_logits,
        )
        # 更新隐状态和位置信息
        self.hidden_states[:num_reqs] = hidden_states[last_token_indices]
        self.input_buffers.positions[:num_reqs] = pos

    def multi_step_decode(
        self,
        num_reqs: int,
        skip_attn: bool,
        batch_desc: BatchExecutionDescriptor,
        num_tokens_across_dp: torch.Tensor | None,
    ) -> None:
        """
        执行多步解码（multi-step decode）。

        这是 EAGLE 草稿生成的第二步，循环执行 num_speculative_steps - 1 次解码，
        每次生成一个草稿 token。

        参数:
            num_reqs (int): 请求数量。
            skip_attn (bool): 是否跳过注意力计算（用于 dummy run）。
            batch_desc (BatchExecutionDescriptor): 批次执行描述符。
            num_tokens_across_dp (torch.Tensor | None): 数据并行各 rank 的 token 数量。

        流程:
            对于每个草稿步骤（从 1 到 num_speculative_steps - 1）：
            1. 构建注意力元数据和 slot 映射
            2. 更新当前草稿步骤
            3. 生成草稿 token（通过 CUDA 图重放或普通前向传播）
        """
        positions = self.input_buffers.positions[:num_reqs]
        query_start_loc = self.input_buffers.query_start_loc[: num_reqs + 1]
        idx_mapping = self.idx_mapping[:num_reqs]

        for step in range(1, self.num_speculative_steps):
            attn_metadata = None
            slot_mappings_by_layer = None
            if not skip_attn:
                # Build attention metadata and slot mappings for each draft
                # decode step. It is necessary to rebuild the attention
                # metadata even when replaying the FULL graph so that any
                # attention metadata builder state is updated.
                # 为每个草稿解码步骤构建注意力元数据和 slot 映射
                # 即使在重放 FULL 图时也需要重建，以更新注意力元数据构建器的状态
                slot_mappings = self.block_tables.compute_slot_mappings(
                    idx_mapping,
                    query_start_loc,
                    positions,
                    batch_desc.num_tokens,
                )
                slot_mappings_by_layer = build_slot_mappings_by_layer(
                    slot_mappings, self.kv_cache_config
                )
                attn_metadata = self._build_draft_attn_metadata(
                    num_reqs=num_reqs,
                    num_reqs_padded=batch_desc.num_reqs or num_reqs,
                    num_tokens_padded=batch_desc.num_tokens,
                )

            # Update the current draft step.
            self.current_draft_step.fill_(step)

            # Generate draft tokens for the current step.
            if batch_desc.cg_mode == CUDAGraphMode.FULL:
                assert self.decode_cudagraph_manager is not None
                self.decode_cudagraph_manager.run_fullgraph(batch_desc)
            else:
                self.generate_draft(
                    num_reqs,
                    batch_desc.num_tokens,
                    attn_metadata,
                    slot_mappings_by_layer,
                    num_tokens_across_dp=num_tokens_across_dp,
                    cudagraph_runtime_mode=batch_desc.cg_mode,
                )

    def generate_draft(
        self,
        num_reqs: int,
        num_tokens_padded: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> None:
        """
        生成一个草稿 token（单步解码）。

        执行一次草稿模型的前向传播和采样，并更新下一步的输入。

        参数:
            num_reqs (int): 请求数量。
            num_tokens_padded (int): 填充后的 token 数量。
            attn_metadata (dict | None): 注意力元数据。
            slot_mappings (dict | None): slot 映射。
            num_tokens_across_dp (torch.Tensor | None): 数据并行各 rank 的 token 数量。
            cudagraph_runtime_mode (CUDAGraphMode): CUDA 图运行模式。

        流程:
            1. 运行草稿模型前向传播
            2. 计算 logits 并采样草稿 token
            3. 更新下一步的输入（input_ids, hidden_states, positions, seq_lens）
        """
        idx_mapping = self.idx_mapping[:num_reqs]
        positions = self.input_buffers.positions[:num_reqs]
        # Run the eagle model forward pass.
        last_hidden_states, hidden_states = self.run_model(
            num_tokens_padded,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            cudagraph_runtime_mode,
        )
        last_hidden_states = last_hidden_states[:num_reqs]

        # Sample the draft tokens.
        logits = self.model.compute_logits(last_hidden_states)
        draft_tokens = self._sample_draft(
            logits,
            idx_mapping,
            positions,
            self.current_draft_step,
            self.draft_logits,
        )

        # Update the inputs for the next step.
        # 更新下一步的输入
        update_eagle_draft_inputs(
            draft_tokens,
            self.current_draft_step,
            hidden_states,
            self.draft_tokens,
            self.hidden_states,
            self.input_buffers,
            num_reqs,
            self.max_model_len,
            self.num_speculative_steps,
        )

    def _build_draft_attn_metadata(
        self,
        num_reqs: int,
        num_reqs_padded: int,
        num_tokens_padded: int,
    ) -> dict[str, Any] | None:
        """
        构建草稿模型的注意力元数据。

        为草稿模型的解码步骤构建注意力元数据，包括 query_start_loc、
        seq_lens、block_tables 和 slot_mappings。

        参数:
            num_reqs (int): 实际请求数量。
            num_reqs_padded (int): 填充后的请求数量。
            num_tokens_padded (int): 填充后的 token 数量。

        返回:
            dict | None: 注意力元数据字典，如果没有草稿注意力层则返回 None。
        """
        if not self.draft_attn_layer_names:
            return None

        query_start_loc_cpu = torch.clamp(
            self.arange[: num_reqs_padded + 1], max=num_reqs
        )
        block_tables = [
            x[:num_reqs_padded] for x in self.block_tables.input_block_tables
        ]
        slot_mappings = self.block_tables.slot_mappings[:, :num_tokens_padded]
        attn_metadata = build_attn_metadata(
            attn_groups=self.attn_groups,
            num_reqs=num_reqs_padded,
            num_tokens=num_tokens_padded,
            query_start_loc_gpu=self.input_buffers.query_start_loc[
                : num_reqs_padded + 1
            ],
            query_start_loc_cpu=query_start_loc_cpu,
            max_query_len=1,
            seq_lens=self.input_buffers.seq_lens[:num_reqs_padded],
            max_seq_len=self.max_model_len,
            block_tables=block_tables,
            slot_mappings=slot_mappings,
            kv_cache_config=self.kv_cache_config,
        )
        return attn_metadata

    def capture(
        self,
        attn_states: dict[BatchExecutionDescriptor, CapturedAttentionState],
    ) -> None:
        """
        捕获 CUDA 图。

        为 EAGLE 草稿模型的前向传播捕获 CUDA 图，以减少 kernel launch 开销。

        参数:
            attn_states (dict): 注意力状态字典，包含预构建的注意力元数据。

        流程:
            1. 重置 last_token_indices 为 0（防止捕获时的越界访问）
            2. 捕获预填充 CUDA 图（模型前向 + compute_logits + sample）
            3. 如果 num_speculative_steps > 1，捕获解码 CUDA 图
        """
        logger.info("Capturing model for Eagle speculator...")
        # Reset indices to zeros to prevent stale values from prior
        # dummy runs to cause out-of-bounds indexing during capture.
        self.last_token_indices.zero_()

        # Capture the prefill routine (model forward + compute_logits +
        # sample).
        # For FULL graphs, the entire routine is recorded as one graph.
        # For PIECEWISE, only the model's compiled regions are captured
        # and the rest (compute_logits, gumbel_sample) runs eagerly.
        # 捕获预填充流程（模型前向 + compute_logits + sample）
        assert self.prefill_cudagraph_manager is not None
        self.prefill_cudagraph_manager.capture(
            self.prefill,
            attn_states,
            progress_bar_desc="Capturing eagle prefill CUDA graphs",
        )

        if self.num_speculative_steps == 1:
            return

        # Capture the decode draft generation routine (model forward +
        # compute_logits + sample + update_eagle_inputs) for a single
        # step.
        # 捕获解码草稿生成流程（模型前向 + compute_logits + sample + update_eagle_inputs）
        assert self.decode_cudagraph_manager is not None
        self.decode_cudagraph_manager.capture(
            self.generate_draft,
            self.model_state,
            self.input_buffers,
            self.block_tables,
            self.attn_groups,
            self.kv_cache_config,
            progress_bar_desc="Capturing eagle decode CUDA graphs",
        )

    @torch.inference_mode()
    def propose(
        self,
        input_batch: InputBatch,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        # [num_tokens, hidden_size]
        last_hidden_states: torch.Tensor,
        # num_layers x [num_tokens, hidden_size]
        aux_hidden_states: list[torch.Tensor] | None,
        # [num_reqs]
        num_sampled: torch.Tensor,
        # [num_reqs]
        num_rejected: torch.Tensor,
        # [max_num_reqs]
        last_sampled: torch.Tensor,
        # [max_num_reqs]
        next_prefill_tokens: torch.Tensor,
        # [max_num_reqs]
        temperature: torch.Tensor,
        # [max_num_reqs]
        seeds: torch.Tensor,
        num_tokens_across_dp: torch.Tensor | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
    ) -> torch.Tensor:
        """
        提议候选 token（主入口函数）。

        这是 EAGLE 投机解码的主要入口，协调整个草稿生成流程。

        参数:
            input_batch (InputBatch): 输入批次信息。
            attn_metadata (dict): 目标模型的注意力元数据。
            slot_mappings (dict): 目标模型的 slot 映射。
            last_hidden_states (torch.Tensor): 目标模型最后一层的隐状态。
            aux_hidden_states (list | None): 辅助隐状态（用于 EAGLE3）。
            num_sampled (torch.Tensor): 已采样的 token 数量。
            num_rejected (torch.Tensor): 被拒绝的 token 数量。
            last_sampled (torch.Tensor): 最后采样的 token。
            next_prefill_tokens (torch.Tensor): 下一个预填充 token。
            temperature (torch.Tensor): 温度参数。
            seeds (torch.Tensor): 随机种子。
            num_tokens_across_dp (torch.Tensor | None): 数据并行各 rank 的 token 数量。
            dummy_run (bool): 是否为 dummy run。
            skip_attn_for_dummy_run (bool): dummy run 时是否跳过注意力。
            mm_inputs (tuple | None): 多模态输入。
            is_profile (bool): 是否为性能分析模式。

        返回:
            torch.Tensor: 候选 token，形状 [num_reqs, num_speculative_steps]。

        流程:
            1. 准备输入数据（隐状态、温度、种子、输入 ID 等）
            2. 执行草稿预填充，生成第一个候选 token
            3. 如果 num_speculative_steps > 1：
               a. 准备解码输入
               b. 执行多步解码，生成后续候选 token
            4. 返回所有候选 token
        """
        num_tokens = input_batch.num_tokens_after_padding
        num_reqs = input_batch.num_reqs
        max_query_len = input_batch.num_scheduled_tokens.max()

        # NOTE(woosuk): To avoid CPU-GPU synchronization without CPU knowing the
        # number of rejected tokens, we maintain the size of eagle's input_ids and
        # hidden_states the same as the target model's. This means, we pad each
        # request's query length to include any rejected positions. By doing so,
        # we can also reuse the attention metadata (e.g., query_start_loc,
        # seq_lens) of the target model.
        # 为了避免 CPU-GPU 同步，我们保持 eagle 的 input_ids 和 hidden_states
        # 与目标模型相同的大小。这样可以复用目标模型的注意力元数据。
        if aux_hidden_states:
            assert self.method == "eagle3"
            hidden_states = self.model.combine_hidden_states(
                torch.cat(aux_hidden_states, dim=-1)
            )
        else:
            hidden_states = last_hidden_states
        self.hidden_states[:num_tokens].copy_(hidden_states)

        # Copy temperature, seeds, and idx mapping to the pre-allocated buffers.
        # NOTE(woosuk): For draft sampling, we only consider the temperature
        # and ignore the other sampling parameters such as top_k and top_p,
        # for simplicity and performance.
        # While this may slightly degrade the acceptance rate, it does not
        # affect the output distribution after rejection sampling.
        # 复制温度、种子和索引映射到预分配的缓冲区
        # 注意：草稿采样只考虑温度，忽略 top_k 和 top_p 等其他参数
        self.temperature.copy_(temperature)
        self.seeds.copy_(seeds)
        self.idx_mapping[:num_reqs].copy_(input_batch.idx_mapping)

        # Get the input ids and last token indices for the speculator.
        # 准备 EAGLE 的输入（input_ids、位置、last_token_indices 等）
        prepare_eagle_inputs(
            self.last_token_indices,
            self.current_draft_step,
            self.input_buffers,
            input_batch,
            num_sampled,
            num_rejected,
            last_sampled,
            next_prefill_tokens,
            self.max_num_reqs,
        )

        # When all requests are decoding (no true prefills), each has
        # num_speculative_steps + 1 tokens, enabling FULL graph replay.
        # 当所有请求都是解码时，每个请求有 num_speculative_steps + 1 个 token，
        # 可以启用 FULL 图重放
        uniform_token_count = get_uniform_token_count(
            num_reqs,
            # Use the actual number of tokens without padding added by
            # the target model during FULL cudagraph.
            input_batch.num_tokens,
            max_query_len,
        )
        prefill_batch_desc, num_tokens_across_dp = dispatch_cg_and_sync_dp(
            self.prefill_cudagraph_manager,
            num_reqs,
            num_tokens,
            uniform_token_count,
            dp_size=self.dp_size,
            dp_rank=self.dp_rank,
            need_eager=is_profile,
        )

        if prefill_batch_desc.cg_mode == CUDAGraphMode.FULL:
            # Replay the full graph for draft prefill.
            # 重放 FULL 图进行草稿预填充
            assert self.prefill_cudagraph_manager is not None
            self.prefill_cudagraph_manager.run_fullgraph(prefill_batch_desc)
        else:
            # The target model's attention metadata and slot mappings
            # can directly be used for draft prefill, because of the
            # identical batch shape and KV cache layout.
            # 目标模型的注意力元数据和 slot 映射可以直接用于草稿预填充
            self.prefill(
                num_reqs,
                prefill_batch_desc.num_tokens,
                attn_metadata,
                slot_mappings,
                num_tokens_across_dp=num_tokens_across_dp,
                cudagraph_runtime_mode=prefill_batch_desc.cg_mode,
                mm_inputs=mm_inputs,
            )

        if self.num_speculative_steps == 1:
            # Early exit.
            return self.draft_tokens[:num_reqs, :1]

        # Prepare the inputs for the decode steps.
        # 准备解码步骤的输入
        prepare_eagle_decode(
            self.draft_tokens[:num_reqs, 0],
            input_batch.seq_lens,
            num_rejected,
            self.input_buffers,
            self.max_model_len,
            self.max_num_reqs,
        )

        # Each request produces exactly 1 token per draft generation step,
        # enabling FULL graph replay.
        # 每个请求在每个草稿生成步骤产生恰好 1 个 token，可以启用 FULL 图重放
        decode_batch_desc, num_tokens_across_dp = dispatch_cg_and_sync_dp(
            self.decode_cudagraph_manager,
            num_reqs,
            num_reqs,
            uniform_token_count=1,
            dp_size=self.dp_size,
            dp_rank=self.dp_rank,
            need_eager=is_profile,
        )

        # Generate the remaining num_speculative_steps - 1 draft tokens.
        # 生成剩余的 num_speculative_steps - 1 个草稿 token
        self.multi_step_decode(
            num_reqs,
            dummy_run and skip_attn_for_dummy_run,
            decode_batch_desc,
            num_tokens_across_dp,
        )

        return self.draft_tokens[:num_reqs]


@triton.jit
def _prepare_eagle_inputs_kernel(
    last_token_indices_ptr,
    eagle_current_draft_step_ptr,
    eagle_input_ids_ptr,
    eagle_positions_ptr,
    eagle_query_start_loc_ptr,
    eagle_seq_lens_ptr,
    target_input_ids_ptr,
    target_positions_ptr,
    idx_mapping_ptr,
    last_sampled_ptr,
    next_prefill_tokens_ptr,
    num_sampled_ptr,
    num_rejected_ptr,
    query_start_loc_ptr,
    seq_lens_ptr,
    max_num_reqs,
    BLOCK_SIZE: tl.constexpr,
):
    """
    准备 EAGLE 输入的 Triton 内核。

    将目标模型的输入数据转换为 EAGLE 草稿模型所需的格式。

    主要操作：
    1. 将 input_ids 向右移动一位（因为 EAGLE 预测的是下一个 token）
    2. 在最后一个位置填入下一个 token（已采样的或预填充的）
    3. 复制位置信息和序列长度
    4. 计算 last_token_indices
    5. 为 CUDA 图填充 padding

    每个程序处理一个请求。

    参数:
        last_token_indices_ptr: 最后 token 索引输出指针。
        eagle_current_draft_step_ptr: 当前草稿步骤指针。
        eagle_input_ids_ptr: EAGLE input_ids 输出指针。
        eagle_positions_ptr: EAGLE 位置输出指针。
        eagle_query_start_loc_ptr: EAGLE query_start_loc 输出指针。
        eagle_seq_lens_ptr: EAGLE seq_lens 输出指针。
        target_input_ids_ptr: 目标模型 input_ids 输入指针。
        target_positions_ptr: 目标模型位置输入指针。
        idx_mapping_ptr: 请求索引映射指针。
        last_sampled_ptr: 最后采样的 token 指针。
        next_prefill_tokens_ptr: 下一个预填充 token 指针。
        num_sampled_ptr: 已采样数量指针。
        num_rejected_ptr: 被拒绝数量指针。
        query_start_loc_ptr: 目标模型 query_start_loc 指针。
        seq_lens_ptr: 目标模型 seq_lens 指针。
        max_num_reqs: 最大请求数量。
        BLOCK_SIZE: 块大小。
    """
    req_idx = tl.program_id(0)
    num_reqs = tl.num_programs(0)
    req_state_idx = tl.load(idx_mapping_ptr + req_idx)

    query_start = tl.load(query_start_loc_ptr + req_idx)
    query_end = tl.load(query_start_loc_ptr + req_idx + 1)
    query_len = query_end - query_start
    seq_len = tl.load(seq_lens_ptr + req_idx)

    # Get the true query length and next token after accounting for rejected tokens.
    # 考虑被拒绝的 token 后，获取真实的 query 长度
    num_rejected = tl.load(num_rejected_ptr + req_idx)
    query_len -= num_rejected

    num_sampled = tl.load(num_sampled_ptr + req_idx)
    if num_sampled > 0:
        next_token = tl.load(last_sampled_ptr + req_state_idx).to(tl.int32)
    else:
        # Chunked prefilling.
        # Get the next prefill token.
        # 分块预填充：获取下一个预填充 token
        next_token = tl.load(next_prefill_tokens_ptr + req_state_idx)

    # Shift target_input_ids by one.
    # 将 target_input_ids 向右移动一位（EAGLE 预测下一个 token）
    for i in range(1, query_len, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < query_len
        input_ids = tl.load(target_input_ids_ptr + query_start + block, mask=mask)
        tl.store(eagle_input_ids_ptr + query_start + block - 1, input_ids, mask=mask)

    last_token_index = query_start + query_len - 1
    tl.store(last_token_indices_ptr + req_idx, last_token_index)
    # 在最后一个位置填入下一个 token
    tl.store(eagle_input_ids_ptr + last_token_index, next_token)

    # Copy positions.
    # 复制位置信息
    for i in range(0, query_len, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < query_len
        target_pos = tl.load(target_positions_ptr + query_start + block, mask=mask)
        tl.store(eagle_positions_ptr + query_start + block, target_pos, mask=mask)

    # Copy query start locations.
    tl.store(eagle_query_start_loc_ptr + req_idx, query_start)
    # Copy sequence lengths.
    tl.store(eagle_seq_lens_ptr + req_idx, seq_len)
    if req_idx == (num_reqs - 1):
        # Reset the current draft step to 0.
        tl.store(eagle_current_draft_step_ptr, 0)
        # Pad query_start_loc for CUDA graphs.
        # 为 CUDA 图填充 query_start_loc
        for i in range(num_reqs, max_num_reqs + 1, BLOCK_SIZE):
            block = i + tl.arange(0, BLOCK_SIZE)
            mask = block < max_num_reqs + 1
            tl.store(eagle_query_start_loc_ptr + block, query_end, mask=mask)
        # Pad seq_lens for CUDA graphs.
        # 为 CUDA 图填充 seq_lens
        for i in range(num_reqs, max_num_reqs, BLOCK_SIZE):
            block = i + tl.arange(0, BLOCK_SIZE)
            mask = block < max_num_reqs
            tl.store(eagle_seq_lens_ptr + block, 0, mask=mask)
        # Pad last_token_indices for CUDA graphs.
        # 为 CUDA 图填充 last_token_indices
        for i in range(num_reqs, max_num_reqs, BLOCK_SIZE):
            block = i + tl.arange(0, BLOCK_SIZE)
            mask = block < max_num_reqs
            tl.store(last_token_indices_ptr + block, 0, mask=mask)


def prepare_eagle_inputs(
    # [num_reqs]
    last_token_indices: torch.Tensor,
    current_draft_step: torch.Tensor,
    input_buffers: InputBuffers,
    input_batch: InputBatch,
    # [num_reqs]
    num_sampled: torch.Tensor,
    # [num_reqs]
    num_rejected: torch.Tensor,
    # [max_num_reqs]
    last_sampled: torch.Tensor,
    # [max_num_reqs]
    next_prefill_tokens: torch.Tensor,
    max_num_reqs,
) -> torch.Tensor:
    """
    准备 EAGLE 的输入数据。

    调用 Triton 内核将目标模型的输入转换为 EAGLE 草稿模型所需的格式。

    参数:
        last_token_indices (torch.Tensor): 最后 token 索引输出。
        current_draft_step (torch.Tensor): 当前草稿步骤。
        input_buffers (InputBuffers): EAGLE 的输入缓冲区。
        input_batch (InputBatch): 目标模型的输入批次。
        num_sampled (torch.Tensor): 已采样数量。
        num_rejected (torch.Tensor): 被拒绝数量。
        last_sampled (torch.Tensor): 最后采样的 token。
        next_prefill_tokens (torch.Tensor): 下一个预填充 token。
        max_num_reqs: 最大请求数量。

    返回:
        torch.Tensor: 最后 token 索引。
    """
    num_reqs = input_batch.num_reqs
    _prepare_eagle_inputs_kernel[(num_reqs,)](
        last_token_indices,
        current_draft_step,
        input_buffers.input_ids,
        input_buffers.positions,
        input_buffers.query_start_loc,
        input_buffers.seq_lens,
        input_batch.input_ids,
        input_batch.positions,
        input_batch.idx_mapping,
        last_sampled,
        next_prefill_tokens,
        num_sampled,
        num_rejected,
        input_batch.query_start_loc,
        input_batch.seq_lens,
        max_num_reqs,
        BLOCK_SIZE=1024,
    )
    return last_token_indices


@triton.jit
def _prepare_eagle_decode_kernel(
    draft_tokens_ptr,
    draft_tokens_stride,
    target_seq_lens_ptr,
    num_rejected_ptr,
    input_ids_ptr,
    positions_ptr,
    query_start_loc_ptr,
    seq_lens_ptr,
    max_model_len,
    max_num_reqs,
    BLOCK_SIZE: tl.constexpr,
):
    """
    准备 EAGLE 解码输入的 Triton 内核。

    将草稿 token 转换为下一步解码的输入。

    主要操作：
    1. 将草稿 token 写入 input_ids
    2. 递增位置和序列长度（带 clamp 防止越界）
    3. 计算 query_start_loc（用于 CUDA 图 padding）

    每个程序处理一个请求（最后一个程序处理 padding）。

    参数:
        draft_tokens_ptr: 草稿 token 输入指针。
        draft_tokens_stride: 草稿 token 的行步长。
        target_seq_lens_ptr: 目标模型序列长度指针。
        num_rejected_ptr: 被拒绝数量指针。
        input_ids_ptr: EAGLE input_ids 输出指针。
        positions_ptr: EAGLE 位置输出指针。
        query_start_loc_ptr: EAGLE query_start_loc 输出指针。
        seq_lens_ptr: EAGLE seq_lens 输出指针。
        max_model_len: 最大模型长度。
        max_num_reqs: 最大请求数量。
        BLOCK_SIZE: 块大小。
    """
    req_idx = tl.program_id(0)
    num_reqs = tl.num_programs(0) - 1
    if req_idx == num_reqs:
        # Compute query_start_loc. Pad it with the last query_start_loc
        # for CUDA graphs.
        # 计算 query_start_loc，并为 CUDA 图填充 padding
        for i in range(0, max_num_reqs + 1, BLOCK_SIZE):
            block = i + tl.arange(0, BLOCK_SIZE)
            q = tl.where(block < num_reqs, block, num_reqs)
            mask = block < max_num_reqs + 1
            tl.store(query_start_loc_ptr + block, q, mask=mask)
        # Pad seq_lens for CUDA graphs.
        # 为 CUDA 图填充 seq_lens
        for i in range(req_idx, max_num_reqs, BLOCK_SIZE):
            block = i + tl.arange(0, BLOCK_SIZE)
            mask = block < max_num_reqs
            tl.store(seq_lens_ptr + block, 0, mask=mask)
        return

    # draft token -> input id.
    # 将草稿 token 转换为 input id
    draft_token = tl.load(draft_tokens_ptr + req_idx * draft_tokens_stride)
    tl.store(input_ids_ptr + req_idx, draft_token)

    # Compute position and seq_lens.
    # NOTE(woosuk): To prevent out-of-range access, we clamp these values
    # if they reach the max model length.
    # 计算位置和序列长度（带 clamp 防止越界）
    position = tl.load(positions_ptr + req_idx)
    position = tl.minimum(position + 1, max_model_len - 1)
    tl.store(positions_ptr + req_idx, position)

    target_seq_len = tl.load(target_seq_lens_ptr + req_idx)
    num_rejected = tl.load(num_rejected_ptr + req_idx)
    seq_len = target_seq_len - num_rejected
    seq_len = tl.minimum(seq_len + 1, max_model_len)
    tl.store(seq_lens_ptr + req_idx, seq_len)


def prepare_eagle_decode(
    draft_tokens: torch.Tensor,
    target_seq_lens: torch.Tensor,
    num_rejected: torch.Tensor,
    input_buffers: InputBuffers,
    max_model_len: int,
    max_num_reqs: int,
):
    """
    准备 EAGLE 解码输入。

    调用 Triton 内核将草稿 token 转换为下一步解码的输入。

    参数:
        draft_tokens (torch.Tensor): 草稿 token，形状 [num_reqs]。
        target_seq_lens (torch.Tensor): 目标模型序列长度。
        num_rejected (torch.Tensor): 被拒绝数量。
        input_buffers (InputBuffers): EAGLE 的输入缓冲区。
        max_model_len (int): 最大模型长度。
        max_num_reqs (int): 最大请求数量。
    """
    num_reqs = draft_tokens.shape[0]
    _prepare_eagle_decode_kernel[(num_reqs + 1,)](
        draft_tokens,
        draft_tokens.stride(0),
        target_seq_lens,
        num_rejected,
        input_buffers.input_ids,
        input_buffers.positions,
        input_buffers.query_start_loc,
        input_buffers.seq_lens,
        max_model_len,
        max_num_reqs,
        BLOCK_SIZE=1024,
    )


@triton.jit
def _update_eagle_draft_inputs_kernel(
    output_draft_tokens_ptr,
    output_draft_tokens_stride,
    next_input_hidden_states_ptr,
    next_input_hidden_states_stride,
    input_ids_ptr,
    positions_ptr,
    seq_lens_ptr,
    draft_tokens_ptr,
    current_draft_step_ptr,
    hidden_states_ptr,
    hidden_states_stride,
    hidden_size,
    max_model_len,
    num_speculative_steps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    更新 EAGLE 草稿输入的 Triton 内核。

    在每个草稿步骤完成后，更新下一步的输入数据。

    主要操作：
    1. 将采样的草稿 token 写入 draft_tokens 数组
    2. 如果不是最后一步，更新下一步的输入：
       a. 将采样的 token 写入 input_ids
       b. 复制隐状态到下一步的输入缓冲区
       c. 递增位置和序列长度

    每个程序处理一个请求。

    参数:
        output_draft_tokens_ptr: 输出的草稿 token 数组指针。
        output_draft_tokens_stride: 草稿 token 数组的行步长。
        next_input_hidden_states_ptr: 下一步输入隐状态指针。
        next_input_hidden_states_stride: 隐状态的行步长。
        input_ids_ptr: input_ids 指针。
        positions_ptr: 位置指针。
        seq_lens_ptr: seq_lens 指针。
        draft_tokens_ptr: 当前采样的草稿 token 指针。
        current_draft_step_ptr: 当前草稿步骤指针。
        hidden_states_ptr: 隐状态指针。
        hidden_states_stride: 隐状态的行步长。
        hidden_size: 隐状态维度。
        max_model_len: 最大模型长度。
        num_speculative_steps: 投机解码步数。
        BLOCK_SIZE: 块大小。
    """
    req_idx = tl.program_id(0)

    # Write the sampled draft token into self.draft_tokens[req_idx, step].
    # 将采样的草稿 token 写入 draft_tokens 数组
    draft_token = tl.load(draft_tokens_ptr + req_idx)
    step = tl.load(current_draft_step_ptr)
    tl.store(
        output_draft_tokens_ptr + req_idx * output_draft_tokens_stride + step,
        draft_token,
    )

    if step >= num_speculative_steps - 1:
        # This is the final step. Skip updating draft forward inputs.
        # 最后一步：跳过更新草稿前向输入
        return

    # Write the sampled draft token into the input ids tensor for the next
    # forward pass.
    # 将采样的草稿 token 写入下一步的 input_ids
    tl.store(input_ids_ptr + req_idx, draft_token)

    # Copy hidden states into the input hidden states tensor for the next
    # forward pass.
    # 复制隐状态到下一步的输入缓冲区
    for i in range(0, hidden_size, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < hidden_size
        hidden_states = tl.load(
            hidden_states_ptr + req_idx * hidden_states_stride + block,
            mask=mask,
        )
        tl.store(
            next_input_hidden_states_ptr
            + req_idx * next_input_hidden_states_stride
            + block,
            hidden_states,
            mask=mask,
        )

    # Increment position and seq_lens.
    # NOTE(woosuk): To prevent out-of-range access, we clamp these values
    # if they reach the max model length.
    # 递增位置和序列长度（带 clamp 防止越界）
    position = tl.load(positions_ptr + req_idx)
    position = tl.minimum(position + 1, max_model_len - 1)
    tl.store(positions_ptr + req_idx, position)

    seq_len = tl.load(seq_lens_ptr + req_idx)
    seq_len = tl.minimum(seq_len + 1, max_model_len)
    tl.store(seq_lens_ptr + req_idx, seq_len)


def update_eagle_draft_inputs(
    draft_tokens: torch.Tensor,
    current_draft_step: torch.Tensor,
    hidden_states: torch.Tensor,
    output_draft_tokens: torch.Tensor,
    next_input_hidden_states: torch.Tensor,
    input_buffers: InputBuffers,
    num_reqs: int,
    max_model_len: int,
    num_speculative_steps: int,
):
    """
    更新 EAGLE 草稿输入。

    调用 Triton 内核更新下一步的输入数据。

    参数:
        draft_tokens (torch.Tensor): 当前采样的草稿 token。
        current_draft_step (torch.Tensor): 当前草稿步骤。
        hidden_states (torch.Tensor): 当前的隐状态。
        output_draft_tokens (torch.Tensor): 输出的草稿 token 数组。
        next_input_hidden_states (torch.Tensor): 下一步的输入隐状态。
        input_buffers (InputBuffers): 输入缓冲区。
        num_reqs (int): 请求数量。
        max_model_len (int): 最大模型长度。
        num_speculative_steps (int): 投机解码步数。
    """
    _, hidden_size = hidden_states.shape
    _update_eagle_draft_inputs_kernel[(num_reqs,)](
        output_draft_tokens,
        output_draft_tokens.stride(0),
        next_input_hidden_states,
        next_input_hidden_states.stride(0),
        input_buffers.input_ids,
        input_buffers.positions,
        input_buffers.seq_lens,
        draft_tokens,
        current_draft_step,
        hidden_states,
        hidden_states.stride(0),
        hidden_size,
        max_model_len,
        num_speculative_steps,
        BLOCK_SIZE=1024,
    )
