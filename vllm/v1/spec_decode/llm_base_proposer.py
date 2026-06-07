# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
投机解码（Speculative Decoding）LLM 基础提议器模块。

本模块实现了基于 LLM 的投机解码提议器基类，用于生成多个候选 token
以加速推理过程。投机解码的核心思想是：使用一个较小的草稿模型（draft model）
快速生成多个候选 token，然后由目标模型并行验证这些 token，从而减少总的
推理步骤数。

主要功能：
1. 支持多种草稿模型架构（EAGLE3、DFlash、传统 draft model）
2. 支持并行草稿生成（parallel drafting）和自回归草稿生成
3. 集成 CUDA Graph 优化以提高推理性能
4. 支持多模态输入（文本、图像等）
5. 管理 KV cache 和注意力元数据
6. 处理 token 采样和概率计算

关键设计：
- 使用预分配的缓冲区（input_ids、positions、hidden_states）避免动态内存分配
- 支持 M-RoPE 和 XD-RoPE 等位置编码方式
- 通过 Triton 内核优化输入数据的准备和处理
- 实现了灵活的批次处理和填充策略

在推理流程中的作用：
1. 接收目标模型的隐藏状态和 token 信息
2. 使用草稿模型生成 num_speculative_tokens 个候选 token
3. 将候选 token 传递给验证器进行验证
4. 返回验证通过的 token 序列
"""
from importlib.util import find_spec
from typing import Any, cast

import numpy as np
import torch
import torch.nn as nn

from vllm.config import (
    CUDAGraphMode,
    VllmConfig,
    get_layers_from_vllm_config,
    replace,
)
from vllm.distributed.parallel_state import get_pp_group
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.model_loader import get_model
from vllm.model_executor.models import supports_multimodal
from vllm.model_executor.models.deepseek_eagle3 import Eagle3DeepseekV2ForCausalLM
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.model_executor.models.llama_eagle3 import Eagle3LlamaForCausalLM
from vllm.model_executor.models.qwen3_dflash import DFlashQwen3ForCausalLM
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.platforms import current_platform
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.attention.backends.triton_attn import TritonAttentionMetadata
from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher
from vllm.v1.kv_cache_interface import KVCacheConfig, UniformTypeKVCacheSpecs
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import _SAMPLING_EPS
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.spec_decode.utils import (
    PADDING_SLOT_ID,
    compute_new_slot_mapping,
    copy_and_expand_eagle_inputs_kernel,
    eagle_prepare_inputs_padded_kernel,
    eagle_prepare_next_token_padded_kernel,
    eagle_step_update_slot_mapping_and_metadata,
    extend_all_queries_by_N,
    next_power_of_2,
)
from vllm.v1.utils import CpuGpuBuffer
from vllm.v1.worker.dp_utils import coordinate_batch_across_dp
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch
from vllm.v1.worker.utils import AttentionGroup

# 初始化日志记录器
logger = init_logger(__name__)


class SpecDecodeBaseProposer:
    """
    投机解码基础提议器基类。

    这是所有基于 LLM 的投机解码提议器的基类，提供了核心的接口和通用实现。
    投机解码通过使用较小的草稿模型快速生成多个候选 token，然后由目标模型
    并行验证，从而减少推理步骤数，提高吞吐量。

    主要职责：
    1. 管理草稿模型的生命周期和配置
    2. 准备模型输入（input_ids、positions、hidden_states）
    3. 处理注意力元数据和 KV cache 映射
    4. 执行草稿模型的前向传播
    5. 采样候选 token 并返回给验证器

    支持的草稿模型类型：
    - EAGLE3：基于隐藏状态预测的草稿模型
    - DFlash：支持并行草稿生成的模型
    - 传统 draft model：独立的较小语言模型

    关键特性：
    - 支持 CUDA Graph 优化
    - 支持多模态输入
    - 支持多种位置编码（RoPE、M-RoPE、XD-RoPE）
    - 支持数据并行
    - 支持概率性草稿采样
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        pass_hidden_states_to_model: bool,
        runner=None,
    ):
        """
        初始化投机解码基础提议器。

        参数:
            vllm_config: vLLM 配置对象，包含模型、调度器、并行等所有配置
            device: 计算设备（CPU/GPU）
            pass_hidden_states_to_model: 是否将隐藏状态传递给草稿模型
            runner: GPU 模型运行器实例，用于执行前向传播
        """
        # 保存配置和设备信息
        self.vllm_config = vllm_config
        assert vllm_config.speculative_config is not None
        self.speculative_config = vllm_config.speculative_config
        self.draft_model_config = self.speculative_config.draft_model_config
        self.method = self.speculative_config.method
        self.pass_hidden_states_to_model = pass_hidden_states_to_model

        # 设置设备和数据类型
        self.device = device
        self.dtype = vllm_config.model_config.dtype
        self.max_model_len = vllm_config.model_config.max_model_len
        self.dp_rank = vllm_config.parallel_config.data_parallel_rank
        self.num_speculative_tokens = self.speculative_config.num_speculative_tokens

        # We need to get the hidden size from the draft model config because
        # the draft model's hidden size can be different from the target model's
        # hidden size (e.g., Llama 3.3 70B).
        # 中文注释：获取草稿模型的隐藏层大小，可能与目标模型不同
        self.hidden_size = self.draft_model_config.get_hidden_size()
        self.inputs_embeds_size = self.draft_model_config.get_inputs_embeds_size()

        # DeepSeek V4 MTP consumes the target's pre-hc_head residual stream,
        # shape (T, hc_mult * hidden_size). Expand the hidden_states buffer
        # so target_hidden_states fits; detect DeepseekV4 via draft hf_config.
        # 中文注释：处理 DeepSeek V4 MTP 模型的特殊隐藏层大小
        draft_hf_config = self.draft_model_config.hf_config
        if hasattr(draft_hf_config, "compress_ratios") and hasattr(
            draft_hf_config, "hc_mult"
        ):
            self.hidden_size = self.hidden_size * draft_hf_config.hc_mult

        # Unifying eagle, draft model, and parallel drafting support.
        # DFlash always uses parallel drafting (all tokens in one pass),
        # but has an additional slot for the next_token_id (does not shift like EAGLE)
        # 中文注释：配置并行草稿生成和额外槽位
        self.parallel_drafting: bool = self.speculative_config.parallel_drafting
        self.extra_slots_per_request = (
            1 if not self.parallel_drafting else self.num_speculative_tokens
        )
        self.net_num_new_slots_per_request = self.extra_slots_per_request - (
            1 if (self.pass_hidden_states_to_model and self.method != "dflash") else 0
        )
        self.needs_extra_input_slots = self.net_num_new_slots_per_request > 0

        # When True, all draft steps reuse the same position as the
        # first step instead of advancing by one each iteration.
        # Used by draft models with Q-only attention that share KV
        # with the target and always predict from the same position.
        # 中文注释：是否使用恒定位置（用于某些特殊草稿模型）
        self.constant_draft_positions: bool = False

        # 中文注释：并行草稿生成的 token ID 和隐藏状态张量
        self.parallel_drafting_token_id: int = 0
        self.parallel_drafting_hidden_state_tensor: torch.Tensor | None = None
        if self.parallel_drafting:
            self._init_parallel_drafting_params()
        # 中文注释：是否使用本地 argmax 归约（优化采样）
        self.use_local_argmax_reduction: bool = (
            self.speculative_config.use_local_argmax_reduction
        )

        # 中文注释：批次大小和 token 数量限制
        self.max_batch_size = vllm_config.scheduler_config.max_num_seqs
        self.max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.token_arange_np = np.arange(self.max_num_tokens, dtype=np.int32)

        # Can be specialized by methods like DFlash to reduce the limit
        # 中文注释：查询 token 和位置的最大数量
        self.max_query_tokens = self.max_num_tokens
        self.max_positions = self.max_num_tokens

        # Multi-modal data support
        # 中文注释：多模态支持检测
        self.mm_registry = MULTIMODAL_REGISTRY
        self.supports_mm_inputs = self.mm_registry.supports_multimodal_inputs(
            vllm_config.model_config
        )

        # 中文注释：草稿模型注意力组和 KV cache 组 ID
        self.draft_attn_groups: list[AttentionGroup] = []
        self.kv_cache_gid: int = -1
        # 中文注释：是否使用 Eagle3 辅助隐藏状态
        self.eagle3_use_aux_hidden_state: bool = (
            self._get_eagle3_use_aux_hidden_state_from_config()
        )

        # 中文注释：编译配置
        self.compilation_config = self.vllm_config.compilation_config

        # Cudagraph dispatcher for PIECEWISE-only dispatching in eagle.
        # Keys are initialized later via initialize_cudagraph_keys() called from
        # gpu_model_runner._check_and_update_cudagraph_mode after
        # adjust_cudagraph_sizes_for_spec_decode is called.
        # 中文注释：CUDA Graph 调度器，用于优化草稿模型的执行
        self.cudagraph_dispatcher = CudagraphDispatcher(self.vllm_config)

        # persistent buffers for cuda graph
        # 中文注释：预分配的持久化缓冲区，用于 CUDA Graph 优化
        # input_ids 缓冲区：存储输入 token ID
        self.input_ids = torch.zeros(
            self.max_num_tokens, dtype=torch.int32, device=device
        )
        # Use draft model's M-RoPE setting, not target model's
        # Draft models may be text-only even if target is multimodal
        # 中文注释：位置编码配置，根据草稿模型类型选择不同的位置编码方式
        self.uses_mrope = self.draft_model_config.uses_mrope
        self.uses_xdrope_dim = self.vllm_config.model_config.uses_xdrope_dim
        self.draft_uses_xdrope_dim = self.draft_model_config.uses_xdrope_dim
        if self.uses_mrope:
            # NOTE: `mrope_positions` is implemented with one additional dummy
            # position on purpose to make it non-contiguous so that it can work
            # with torch compile.
            # See detailed explanation in https://github.com/vllm-project/vllm/pull/12128#discussion_r1926431923

            # NOTE: When M-RoPE is enabled, position ids are 3D regardless of
            # the modality of inputs. For text-only inputs, each dimension has
            # identical position IDs, making M-RoPE functionally equivalent to
            # 1D-RoPE.
            # See page 5 of https://arxiv.org/abs/2409.12191
            # 中文注释：M-RoPE 位置编码缓冲区（3D 位置）
            self.mrope_positions = torch.zeros(
                (3, self.max_positions + 1), dtype=torch.int64, device=device
            )
        elif self.uses_xdrope_dim > 0 and self.draft_uses_xdrope_dim > 0:
            # 中文注释：XD-RoPE 位置编码缓冲区
            self.xdrope_positions = torch.zeros(
                (self.uses_xdrope_dim, self.max_positions + 1),
                dtype=torch.int64,
                device=device,
            )
        else:
            # RoPE need (max_num_tokens,)
            # 中文注释：标准 RoPE 位置编码缓冲区（1D 位置）
            self.positions = torch.zeros(
                self.max_positions,
                dtype=torch.int64,
                device=device,
            )
        # 中文注释：隐藏状态缓冲区，存储目标模型的隐藏状态输出
        self.hidden_states = torch.zeros(
            (self.max_num_tokens, self.hidden_size), dtype=self.dtype, device=device
        )

        # Will be set when we initialize the attention backend
        # 中文注释：注意力块大小，将在初始化注意力后端时设置
        self.block_size: int = -1

        # We need +1 here because the arange is used to set query_start_loc,
        # which has one more element than batch_size.
        # 中文注释：预分配的 arange 缓冲区，用于 query_start_loc 等计算
        max_num_slots_for_arange = max(self.max_batch_size + 1, self.max_num_tokens)
        self.arange = torch.arange(
            max_num_slots_for_arange, device=device, dtype=torch.int32
        )

        # 中文注释：检查额外输入槽位的兼容性
        if self.needs_extra_input_slots:
            self._raise_if_padded_drafter_batch_disabled()
            self._warn_if_multimodal()
            self._raise_if_mrope()

        # 中文注释：token 掩码缓冲区，用于跟踪被拒绝和被掩码的 token
        self.is_rejected_token_mask: torch.Tensor | None = None
        self.is_masked_token_mask: torch.Tensor | None = None
        if self.needs_extra_input_slots:
            # For draft models and parallel drafting, we need to keep track of
            # which tokens are rejected to update the slot mapping with padding slots.
            # 中文注释：被拒绝 token 的掩码（用于更新 slot mapping）
            self.is_rejected_token_mask = torch.zeros(
                (self.max_num_tokens,), dtype=torch.bool, device=device
            )
            # For parallel drafting, we also need to keep track of which tokens
            # are parallel-padding tokens used to sample at later positions.
            # We populate this tensor even when using draft models for simplicity.
            # 中文注释：被掩码 token 的掩码（用于并行草稿生成）
            self.is_masked_token_mask = torch.zeros(
                (self.max_num_tokens,), dtype=torch.bool, device=device
            )

        # 中文注释：输入嵌入缓冲区（用于多模态模型）
        self.inputs_embeds = torch.zeros(
            (self.max_num_tokens, self.inputs_embeds_size),
            dtype=self.dtype,
            device=device,
        )

        # 中文注释：备份下一个 token ID 的 CPU-GPU 缓冲区
        self.backup_next_token_ids = CpuGpuBuffer(
            self.max_batch_size,
            dtype=torch.int32,
            pin_memory=is_pin_memory_available(),
            device=device,
            with_numpy=True,
        )
        # 中文注释：是否启用概率性草稿采样
        self._enable_probabilistic_draft_probs = (
            self.speculative_config.rejection_sample_method == "standard"
            and self.speculative_config.draft_sample_method == "probabilistic"
        )
        # 中文注释：上一次草稿概率（用于概率性采样）
        self._last_draft_probs: torch.Tensor | None = None

        # 中文注释：slot mapping 缓冲区
        self._slot_mapping_buffer = torch.zeros(
            self.max_positions,
            dtype=torch.int64,
            device=device,
        )

        # Determine allowed attention backends once during initialization.
        # 中文注释：确定允许的注意力后端类型（主要用于 ROCm 平台）
        self.allowed_attn_types: tuple | None = None
        if current_platform.is_rocm():
            from vllm.models.deepseek_v4.amd.rocm import (
                DeepseekV4ROCMAiterMLASparseMetadata,
                DeepseekV4ROCMAiterSparseSWAMetadata,
            )
            from vllm.v1.attention.backends.mla.indexer import (
                DeepseekV32IndexerMetadata,
            )
            from vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse import (
                ROCMAiterMLASparseMetadata,
            )
            from vllm.v1.attention.backends.rocm_attn import RocmAttentionMetadata

            rocm_types = [
                TritonAttentionMetadata,
                RocmAttentionMetadata,
                ROCMAiterMLASparseMetadata,
                DeepseekV4ROCMAiterMLASparseMetadata,
                DeepseekV4ROCMAiterSparseSWAMetadata,
                DeepseekV32IndexerMetadata,
            ]
            # ROCM_AITER_FA is an optional backend
            # We check is_enabled() here to avoid importing the backend module during
            # auto-discovery when VLLM_ROCM_USE_AITER=0, which would trigger aiter
            # import and JIT compilation warnings. Explicit backend selection via
            # attention_config still works because the backend module is loaded
            # directly when selected, not through this auto-discovery path.
            # Check if backend module exists to allow explicit selection
            if find_spec(
                AttentionBackendEnum.ROCM_AITER_FA.get_path(include_classname=False)
            ):
                from vllm.v1.attention.backends.rocm_aiter_fa import (
                    AiterFlashAttentionMetadata,
                )

                rocm_types.append(AiterFlashAttentionMetadata)

            # TRITON_MLA backend support for MLA models (e.g., DeepSeek)
            from vllm.model_executor.layers.attention.mla_attention import (
                MLACommonMetadata,
            )

            rocm_types.append(MLACommonMetadata)

            # FlexAttention backend support
            from vllm.v1.attention.backends.flex_attention import FlexAttentionMetadata

            rocm_types.append(FlexAttentionMetadata)

            self.allowed_attn_types = tuple(rocm_types)

    def _raise_if_padded_drafter_batch_disabled(self):
        """
        检查填充批次是否被禁用。

        如果禁用了填充批次，则抛出 NotImplementedError 异常。
        投机解码的草稿模型和并行草稿生成只支持填充批次模式。

        抛出:
            NotImplementedError: 如果禁用了填充批次
        """
        if self.speculative_config.disable_padded_drafter_batch:
            raise NotImplementedError(
                "Speculative Decoding with draft models or parallel drafting only "
                "supports padded drafter batch. Please unset "
                "disable_padded_drafter_batch in the speculative_config."
            )

    def _warn_if_multimodal(self):
        """
        检查多模态支持并发出警告。

        如果模型支持多模态输入，发出警告，因为投机解码目前
        不完全支持多模态模型。
        """
        if self.supports_mm_inputs:
            logger.warning(
                "Speculative Decoding with draft models or parallel drafting "
                "does not fully support multimodal models yet. "
                "Proceeding with text-only speculative decoding."
            )

    def _raise_if_mrope(self):
        """
        检查 M-RoPE 支持并抛出异常。

        如果草稿模型使用 M-RoPE 位置编码，则抛出 NotImplementedError 异常。
        投机解码目前不支持 M-RoPE。

        抛出:
            NotImplementedError: 如果草稿模型使用 M-RoPE
        """
        if self.draft_model_config.uses_mrope:
            raise NotImplementedError(
                "Speculative Decoding with draft models or parallel drafting "
                "does not support M-RoPE yet"
            )

    def _init_parallel_drafting_params(self):
        """
        初始化并行草稿生成的参数。

        并行草稿生成需要：
        1. 掩码 token ID：用于填充被掩码的槽位
        2. 隐藏状态张量：用于 EAGLE + 并行草稿生成的掩码槽位

        从草稿模型配置中提取掩码 token ID，支持以下配置格式：
        - DFlash: dflash_config.mask_token_id
        - PARD: pard_token
        - PTD: ptd_token_id
        """
        # For parallel drafting, we need the token ID to use for masked slots
        # And for EAGLE + parallel drafting, we need the hidden state tensor to use
        # for those masked slots.

        # 中文注释：从草稿模型配置中提取掩码 token ID
        model_hf_config = self.draft_model_config.hf_config
        # DFlash stores mask_token_id in dflash_config
        dflash_config = getattr(model_hf_config, "dflash_config", None)
        if dflash_config and "mask_token_id" in dflash_config:
            self.parallel_drafting_token_id = dflash_config["mask_token_id"]
        elif hasattr(model_hf_config, "pard_token"):
            self.parallel_drafting_token_id = model_hf_config.pard_token
        elif hasattr(model_hf_config, "ptd_token_id"):
            self.parallel_drafting_token_id = model_hf_config.ptd_token_id
        else:
            raise ValueError(
                "For parallel drafting, the draft model config must have "
                "`pard_token`, `ptd_token_id`, or "
                "`dflash_config.mask_token_id` specified in its config.json."
            )

        # 中文注释：如果需要传递隐藏状态，创建隐藏状态张量缓冲区
        if self.pass_hidden_states_to_model:
            self.parallel_drafting_hidden_state_tensor = torch.empty(
                self.hidden_size, dtype=self.dtype, device=self.device
            )

    def _get_positions(self, num_tokens: int):
        """
        获取指定数量的位置编码。

        根据位置编码类型返回相应的位置编码张量切片。

        参数:
            num_tokens: 需要获取的位置数量

        返回:
            位置编码张量，形状根据位置编码类型而定：
            - M-RoPE: (3, num_tokens)
            - XD-RoPE: (uses_xdrope_dim, num_tokens)
            - 标准 RoPE: (num_tokens,)
        """
        if self.uses_mrope:
            return self.mrope_positions[:, :num_tokens]
        if self.uses_xdrope_dim > 0 and self.draft_uses_xdrope_dim > 0:
            return self.xdrope_positions[:, :num_tokens]
        return self.positions[:num_tokens]

    def _set_positions(self, num_tokens: int, positions: torch.Tensor):
        """
        设置指定数量的位置编码。

        将位置编码写入预分配的缓冲区。

        参数:
            num_tokens: 需要设置的位置数量
            positions: 位置编码张量
        """
        if self.uses_mrope:
            self.mrope_positions[:, :num_tokens] = positions
        elif self.uses_xdrope_dim > 0 and self.draft_uses_xdrope_dim > 0:
            self.xdrope_positions[:, :num_tokens] = positions
        else:
            # Convert M-RoPE positions if target model uses M-RoPE
            # but draft doesn't, For text inputs, all M-RoPE
            # dimensions are identical
            # 中文注释：如果目标模型使用 M-RoPE 但草稿模型不使用，
            # 则提取第一个维度的位置（文本输入时所有维度相同）
            if self.vllm_config.model_config.uses_mrope:
                positions = positions[0]
            self.positions[:num_tokens] = positions

    def _get_slot_mapping(
        self,
        num_tokens: int,
        slot_mapping: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return slot_mapping dict for EAGLE layers.

        If slot_mapping is provided, copies it into the buffer first.

        中文注释：获取 EAGLE 层的 slot mapping 字典。

        slot mapping 将逻辑 token 位置映射到物理 KV cache 位置。
        如果提供了 slot mapping，则先复制到缓冲区，然后用填充 ID 填充剩余部分。

        参数:
            num_tokens: 需要的 slot mapping 数量
            slot_mapping: 可选的外部 slot mapping 张量

        返回:
            字典，键为注意力层名称，值为 slot mapping 张量
        """
        # 中文注释：如果提供了外部 slot mapping，复制到缓冲区
        if slot_mapping is not None:
            num_actual = slot_mapping.shape[0]
            self._slot_mapping_buffer[:num_actual].copy_(slot_mapping)
            # 中文注释：用填充 ID 填充剩余部分
            if num_tokens > num_actual:
                self._slot_mapping_buffer[num_actual:num_tokens].fill_(PADDING_SLOT_ID)

        # 中文注释：返回所有草稿注意力层共享的 slot mapping 视图
        view = self._slot_mapping_buffer[:num_tokens]
        return {name: view for name in self._draft_attn_layer_names}

    def initialize_cudagraph_keys(self, cudagraph_mode: CUDAGraphMode) -> None:
        """Initialize cudagraph dispatcher keys for the drafter.

        Only supports PIECEWISE cudagraphs (via mixed_mode).
        This should be called after adjust_cudagraph_sizes_for_spec_decode.

        中文注释：初始化草稿模型的 CUDA Graph 调度器键。

        CUDA Graph 优化可以显著减少 kernel 启动开销，提高推理性能。
        此方法只支持 PIECEWISE 模式的 CUDA Graph。

        参数:
            cudagraph_mode: CUDA Graph 运行模式
        """
        # 中文注释：根据配置决定是否启用 CUDA Graph
        if (
            not self.speculative_config.enforce_eager
            and cudagraph_mode.mixed_mode()
            in [CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL]
        ):
            eagle_cudagraph_mode = CUDAGraphMode.PIECEWISE
        else:
            eagle_cudagraph_mode = CUDAGraphMode.NONE

        # 中文注释：初始化 CUDA Graph 调度器键
        self.cudagraph_dispatcher.initialize_cudagraph_keys(eagle_cudagraph_mode)

    def _greedy_sample(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Greedy-sample draft tokens from hidden states.

        中文注释：从隐藏状态中贪婪采样草稿 token。

        贪婪采样选择概率最高的 token 作为草稿 token。

        参数:
            hidden_states: 隐藏状态张量，形状 (num_tokens, hidden_size)

        返回:
            草稿 token ID 张量，形状 (num_tokens,)
        """
        # 中文注释：如果使用本地 argmax 归约，直接获取 top token
        if self.use_local_argmax_reduction:
            return self.model.get_top_tokens(hidden_states)
        # 中文注释：否则计算 logits 并取 argmax
        return self.model.compute_logits(hidden_states).argmax(dim=-1)

    def _sample_from_logits(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        从 logits 中采样草稿 token。

        根据配置选择贪婪采样或概率性采样。

        参数:
            logits: 模型输出的 logits 张量
            sampling_metadata: 采样元数据

        返回:
            元组 (draft_token_ids, draft_probs)
            - draft_token_ids: 采样的 token ID
            - draft_probs: 采样概率（仅概率性采样时返回）
        """
        # 中文注释：如果未启用概率性草稿概率，使用贪婪采样
        if not self._enable_probabilistic_draft_probs:
            return logits.argmax(dim=-1), None
        # 中文注释：如果所有请求都是贪婪采样，使用 argmax
        if sampling_metadata.all_greedy:
            return logits.argmax(dim=-1), None
        # 中文注释：使用概率性采样
        return compute_probs_and_sample_next_token(logits, sampling_metadata)

    def _sample_draft_tokens(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        从隐藏状态中采样草稿 token。

        根据配置选择贪婪采样或概率性采样。

        参数:
            hidden_states: 隐藏状态张量
            sampling_metadata: 采样元数据

        返回:
            元组 (draft_token_ids, draft_probs)
        """
        # 中文注释：如果未启用概率性草稿概率或所有请求都是贪婪采样，使用贪婪采样
        if not self._enable_probabilistic_draft_probs or sampling_metadata.all_greedy:
            return self._greedy_sample(hidden_states), None
        # 中文注释：计算 logits 并进行概率性采样
        logits = self.model.compute_logits(hidden_states)
        return self._sample_from_logits(logits, sampling_metadata)

    def take_last_draft_probs(self) -> torch.Tensor | None:
        """
        获取上一次草稿采样的概率。

        用于概率性草稿采样，返回上一次采样得到的概率分布。

        返回:
            草稿概率张量，形状 (batch_size, num_speculative_tokens, vocab_size)
            如果未启用概率性采样，返回 None
        """
        return self._last_draft_probs

    def propose(
        self,
        # [num_tokens]
        target_token_ids: torch.Tensor,
        # [num_tokens] or [3, num_tokens] when M-RoPE is enabled
        target_positions: torch.Tensor,
        # [num_tokens, hidden_size]
        target_hidden_states: torch.Tensor,
        # [batch_size]
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
        """
        提议方法：生成投机解码的候选 token 序列。

        这是投机解码的核心方法，负责：
        1. 准备草稿模型的输入（input_ids、positions、hidden_states）
        2. 执行草稿模型的前向传播
        3. 采样生成多个候选 token
        4. 返回给验证器进行验证

        流程概述：
        [步骤 1] 准备第一次前向传播的输入
        [步骤 2] 构建注意力元数据
        [步骤 3] 确定批次执行和填充策略
        [步骤 4] 构建模型输入
        [步骤 5] 执行第一次前向传播
        [步骤 6] 采样第一个草稿 token
        [步骤 7] 如果只有一个草稿 token 或使用并行草稿，直接返回
        [步骤 8] 循环生成剩余的草稿 token（自回归方式）

        参数:
            target_token_ids: 目标模型的输入 token ID，形状 (num_tokens,)
            target_positions: 目标模型的位置编码，形状 (num_tokens,) 或 (3, num_tokens)
            target_hidden_states: 目标模型的隐藏状态，形状 (num_tokens, hidden_size)
            next_token_ids: 目标模型采样的下一个 token ID，形状 (batch_size,)
            token_indices_to_sample: 需要采样的 token 索引，形状 (batch_size,)
            common_attn_metadata: 通用注意力元数据
            sampling_metadata: 采样元数据
            mm_embed_inputs: 多模态嵌入输入
            num_rejected_tokens_gpu: 被拒绝的 token 数量（GPU 张量）
            slot_mappings: slot mapping 字典或列表

        返回:
            draft_token_ids: 草稿 token ID 张量，形状 (batch_size, num_speculative_tokens)
        """
        # 中文注释：重置上一次的草稿概率
        self._last_draft_probs = None
        batch_size = common_attn_metadata.batch_size()

        # 中文注释：步骤 1 - 对于 EAGLE3 和 DFlash 模型，需要组合隐藏状态
        if self.method in ("eagle3", "dflash"):
            assert isinstance(
                self.model,
                (
                    Eagle3LlamaForCausalLM,
                    Eagle3DeepseekV2ForCausalLM,
                    DFlashQwen3ForCausalLM,
                ),
            )
            target_hidden_states = self.model.combine_hidden_states(
                target_hidden_states
            )
            assert target_hidden_states.shape[-1] == self.hidden_size

        # 中文注释：步骤 2 - 准备第一次前向传播的输入
        # 包括调整 input_ids、positions、hidden_states 的布局
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

        # 中文注释：步骤 3 - 构建注意力元数据
        # 包括每个注意力组和每个注意力层的元数据
        per_group_attn_metadata, per_layer_attn_metadata = (
            self.build_per_group_and_layer_attn_metadata(common_attn_metadata)
        )

        # 中文注释：步骤 4 - 确定批次执行模式和填充策略
        cudagraph_runtime_mode, num_input_tokens, num_tokens_across_dp = (
            self._determine_batch_execution_and_padding(num_tokens)
        )

        # 中文注释：步骤 5 - 构建模型输入（包括多模态嵌入）
        model_kwargs, slot_mapping_size = self.build_model_inputs_first_pass(
            num_tokens, num_input_tokens, mm_embed_inputs
        )

        # 中文注释：步骤 6 - 执行第一次前向传播
        # 使用 set_forward_context 设置前向传播上下文
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

        # 中文注释：提取需要采样的隐藏状态
        sample_hidden_states = last_hidden_states[token_indices_to_sample]

        # Early exit if there is only one draft token to be generated.
        # 中文注释：步骤 7 - 如果只有一个草稿 token 或使用并行草稿，直接采样并返回
        if self.num_speculative_tokens == 1 or self.parallel_drafting:
            draft_token_ids, draft_probs = self._sample_draft_tokens(
                sample_hidden_states, sampling_metadata
            )
            if draft_probs is not None:
                self._last_draft_probs = draft_probs.view(
                    -1, self.num_speculative_tokens, draft_probs.shape[-1]
                ).contiguous()
            return draft_token_ids.view(-1, self.num_speculative_tokens)

        # 中文注释：步骤 8 - 准备生成剩余的草稿 token
        # 提取位置编码和隐藏状态
        if self.uses_mrope:
            positions = self.mrope_positions[:, token_indices_to_sample]
        else:
            positions = self.positions[token_indices_to_sample]
        hidden_states = hidden_states[token_indices_to_sample]

        # 中文注释：如果使用恒定位置，将采样位置写入位置缓冲区开头
        if self.constant_draft_positions:
            # Write the sampling positions into the front of the
            # positions buffer so that subsequent loop iterations
            # (which read via _get_positions) use the correct values.
            self.positions[:batch_size] = positions

        # 中文注释：采样第一个草稿 token
        draft_token_ids, draft_probs = self._sample_draft_tokens(
            sample_hidden_states, sampling_metadata
        )
        draft_probs_list = None if draft_probs is None else [draft_probs]

        # 中文注释：检查注意力元数据类型是否支持
        if self.allowed_attn_types is not None:
            for group_md in per_group_attn_metadata:
                if not isinstance(group_md, self.allowed_attn_types):
                    raise ValueError(
                        f"Unsupported attention metadata type for speculative "
                        "decoding with num_speculative_tokens > 1: "
                        f"{type(group_md)}. Supported types are: "
                        f"{self.allowed_attn_types}"
                    )

        # Generate the remaining draft tokens.
        # 中文注释：开始循环生成剩余的草稿 token
        draft_token_ids_list = [draft_token_ids]

        # 中文注释：为后续迭代确定批次执行模式
        cudagraph_runtime_mode, input_batch_size, batch_size_across_dp = (
            self._determine_batch_execution_and_padding(batch_size)
        )

        # 中文注释：更新注意力元数据为单 token 查询模式
        common_attn_metadata.num_actual_tokens = batch_size
        common_attn_metadata.max_query_len = 1
        common_attn_metadata.query_start_loc = self.arange[: batch_size + 1]
        common_attn_metadata.query_start_loc_cpu = torch.from_numpy(
            self.token_arange_np[: batch_size + 1]
        ).clone()

        # In padded drafter batch, we need to adjust the sequence lengths
        # to remove the "padding" (i.e. rejected tokens).
        # Only apply this adjustment when we have rejected tokens
        # (i.e., not the first proposal).
        # 中文注释：如果有被拒绝的 token，需要将序列长度减去被拒绝的 token 数量。
        # 这是因为填充批次中被拒绝的 token 被当作 padding 保留在输入中，
        # 但实际有效序列长度应该排除这些已验证错误的 token。
        # 例如：seq_lens=[10,8,12], rejected=[2,0,1] -> seq_lens=[8,8,11]
        if self.num_speculative_tokens > 1 and num_rejected_tokens_gpu is not None:
            common_attn_metadata.seq_lens -= num_rejected_tokens_gpu
            # Invalidate the CPU-side shadows to avoid H<>D sync.
            # 中文注释：清除 CPU 侧的元数据缓存副本。
            # 因为 GPU 侧的 seq_lens 已被修改，如果保留旧的 CPU 副本，
            # 后续代码可能错误地使用过时的 CPU 值，或者触发不必要的 Host-Device 同步。
            # 设置为 None 强制后续访问时从 GPU 重新读取或重新计算。
            common_attn_metadata._seq_lens_cpu = None
            common_attn_metadata._num_computed_tokens_cpu = None

        # 中文注释：获取注意力块大小
        block_size = self.block_size
        assert block_size > 0, "block_size has not been initialized."

        # 中文注释：步骤 9 - 循环生成剩余的草稿 token（自回归方式）
        for token_index in range(self.num_speculative_tokens - 1):
            # Update the inputs.
            # cast to int32 is crucial when eagle model is compiled.
            # tensor.argmax() returns int64 by default.
            # 中文注释：准备输入 token ID（转换为 int32）
            input_ids = draft_token_ids_list[-1].int()

            # 中文注释：如果位置不是恒定的，更新位置和相关元数据
            if not self.constant_draft_positions:
                positions = self._update_positions_dependent_metadata(
                    positions,
                    common_attn_metadata,
                    batch_size,
                    input_batch_size,
                    block_size,
                )

            # Rebuild attention metadata. When draft positions are constant
            # (e.g. Gemma4 MTP), common_attn_metadata is invariant across
            # loop iterations so we build once and reuse.
            # 中文注释：重新构建注意力元数据
            if not self.constant_draft_positions or token_index == 0:
                _, per_layer_attn_metadata = (
                    self.build_per_group_and_layer_attn_metadata(
                        common_attn_metadata, draft_index=token_index + 1
                    )
                )

            # copy inputs to buffer for cudagraph
            # 中文注释：复制输入到 CUDA Graph 缓冲区
            self.input_ids[:batch_size] = input_ids
            self.hidden_states[:batch_size] = hidden_states
            # 中文注释：处理多模态输入
            if self.supports_mm_inputs:
                self.inputs_embeds[:batch_size] = self.model.embed_input_ids(input_ids)

                input_ids = None
                inputs_embeds = self.inputs_embeds[:input_batch_size]
            else:
                input_ids = self.input_ids[:input_batch_size]
                inputs_embeds = None

            # Run the model.
            # 中文注释：构建模型输入参数，包括 token ID、位置编码和可选的嵌入/隐藏状态
            model_kwargs = {
                "input_ids": input_ids,
                "positions": self._get_positions(input_batch_size),
                "inputs_embeds": inputs_embeds,
            }
            # 中文注释：EAGLE/MTP 等方法需要将 target 模型的隐藏状态传入 draft 模型
            if self.pass_hidden_states_to_model:
                model_kwargs["hidden_states"] = self.hidden_states[:input_batch_size]

            # 中文注释：在 set_forward_context 上下文中执行草稿模型的前向传播。
            # set_forward_context 会设置全局注意力元数据，使模型内各层能正确读写 KV cache。
            # per_layer_attn_metadata 将 layer_name 映射到对应的注意力元数据。
            with set_forward_context(
                per_layer_attn_metadata,
                self.vllm_config,
                num_tokens=input_batch_size,
                num_tokens_across_dp=batch_size_across_dp,
                cudagraph_runtime_mode=cudagraph_runtime_mode,
                slot_mapping=self._get_slot_mapping(input_batch_size),
            ):
                ret_hidden_states = self.model(**model_kwargs)
                # 中文注释：根据模型输出格式提取隐藏状态。
                # MTP/draft_model/dflash 方法返回 (last_hidden_states, hidden_states) 元组，
                # 其中 last_hidden_states 用于采样，hidden_states 传给下一次迭代。
                # EAGLE 方法直接返回单一隐藏状态张量。
                if not self.model_returns_tuple():
                    last_hidden_states = ret_hidden_states
                    hidden_states = ret_hidden_states
                else:
                    last_hidden_states, hidden_states = ret_hidden_states

            # 中文注释：提取隐藏状态并采样草稿 token。
            # 截取 batch_size 部分，排除填充的 token。
            # _sample_draft_tokens 内部会调用 lm_head + argmax/采样生成下一个 token。
            hidden_states = hidden_states[:batch_size]
            draft_token_ids, draft_probs = self._sample_draft_tokens(
                last_hidden_states[:batch_size], sampling_metadata
            )
            # 中文注释：保存草稿概率（如果启用概率性采样）
            if draft_probs is not None:
                assert draft_probs_list is not None
                draft_probs_list.append(draft_probs)
            draft_token_ids_list.append(draft_token_ids)

        # [batch_size, num_speculative_tokens]
        # 中文注释：步骤 10 - 将所有草稿 token 堆叠成最终结果。
        # draft_token_ids_list 包含 num_speculative_tokens 个 [batch_size] 张量，
        # 堆叠后得到 [batch_size, num_speculative_tokens] 的草稿 token 矩阵。
        # 每一行是一个请求的连续 num_speculative_tokens 个草稿 token。
        draft_token_ids = torch.stack(draft_token_ids_list, dim=1)
        if draft_probs_list is not None:
            # 中文注释：将草稿概率也堆叠并保存，供后续 rejection sampling 使用。
            # contiguous() 确保内存连续，便于后续传输和计算。
            self._last_draft_probs = torch.stack(draft_probs_list, dim=1).contiguous()
        return draft_token_ids

    def _update_positions_dependent_metadata(
        self,
        positions: torch.Tensor,
        common_attn_metadata,
        batch_size: int,
        input_batch_size: int,
        block_size: int,
    ) -> torch.Tensor:
        """Update positions, slot mappings, and sequence metadata for the
        next draft step. Returns the updated positions tensor.

        中文注释：更新位置、slot mapping 和序列元数据，为下一个草稿步骤做准备。

        在自回归生成草稿 token 的过程中，每一步都需要：
        1. 更新位置编码（前进一位）
        2. 根据新位置重新计算 slot mapping
        3. 更新序列长度元数据

        参数:
            positions: 当前位置编码张量
            common_attn_metadata: 通用注意力元数据
            batch_size: 实际批次大小
            input_batch_size: 输入批次大小（可能包含填充）
            block_size: 注意力块大小

        返回:
            更新后的位置编码张量
        """
        # 中文注释：提取一维位置编码（M-RoPE 取第一个维度）
        positions_1d = positions[0] if self.uses_mrope else positions
        # 中文注释：确定输出位置缓冲区
        if self.uses_mrope:
            out_pos = self.mrope_positions[0, :batch_size]
        elif self.uses_xdrope_dim > 0 and self.draft_uses_xdrope_dim > 0:
            out_pos = self.xdrope_positions[0, :batch_size]
        else:
            out_pos = self.positions[:batch_size]

        # 中文注释：使用 Triton 内核更新 slot mapping 和元数据
        # 这是性能关键路径，使用 GPU 内核避免 CPU-GPU 同步
        eagle_step_update_slot_mapping_and_metadata(
            positions_1d=positions_1d,
            block_table_tensor=common_attn_metadata.block_table_tensor,
            seq_lens=common_attn_metadata.seq_lens,
            block_size=block_size,
            max_model_len=self.max_model_len,
            out_clamped_positions=out_pos,
            out_slot_mapping=self._slot_mapping_buffer[:input_batch_size],
            input_batch_size=input_batch_size,
        )
        # 中文注释：更新注意力元数据的 slot mapping
        common_attn_metadata.slot_mapping = self._slot_mapping_buffer[:batch_size]

        # 中文注释：根据位置编码类型更新位置
        if self.uses_mrope:
            # M-RoPE：复制第一个维度到其他维度
            self.mrope_positions[1:, :batch_size] = self.mrope_positions[0, :batch_size]
            positions = self.mrope_positions[:, :batch_size]
        elif self.uses_xdrope_dim > 0 and self.draft_uses_xdrope_dim > 0:
            # XD-RoPE：复制第一个维度到其他维度
            self.xdrope_positions[1:, :batch_size] = self.xdrope_positions[
                0, :batch_size
            ]
            positions = self.xdrope_positions[0, :batch_size]
        else:
            # 标准 RoPE：直接使用更新后的位置
            positions = self.positions[:batch_size]

        # 中文注释：更新最大序列长度（限制不超过模型最大长度）
        common_attn_metadata.max_seq_len = min(
            common_attn_metadata.max_seq_len + 1,
            self.max_model_len,
        )

        # 中文注释：更新 CPU 侧的元数据副本
        if common_attn_metadata._seq_lens_cpu is not None:
            common_attn_metadata._seq_lens_cpu += 1
        if common_attn_metadata._num_computed_tokens_cpu is not None:
            common_attn_metadata._num_computed_tokens_cpu += 1
        if common_attn_metadata.seq_lens_cpu_upper_bound is not None:
            common_attn_metadata.seq_lens_cpu_upper_bound += 1

        return positions

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
        """
        设置第一次前向传播的输入。

        根据是否需要额外输入槽位，采用不同的处理路径：

        路径 1（默认 EAGLE 路径）：
        - 不需要额外输入槽位
        - 简单旋转 input_ids 并插入下一个 token
        - 位置编码保持不变

        路径 2（草稿模型或并行草稿路径）：
        - 需要额外输入槽位
        - 使用 Triton 内核复制和扩展输入
        - 重新计算 slot mapping
        - 更新注意力元数据

        参数:
            target_token_ids: 目标模型的输入 token ID
            next_token_ids: 目标模型采样的下一个 token ID
            target_positions: 目标模型的位置编码
            target_hidden_states: 目标模型的隐藏状态
            token_indices_to_sample: 需要采样的 token 索引
            cad: 通用注意力元数据
            num_rejected_tokens_gpu: 被拒绝的 token 数量

        返回:
            元组 (num_tokens, token_indices_to_sample, common_attn_metadata)
            - num_tokens: 输出 token 数量
            - token_indices_to_sample: 采样索引
            - common_attn_metadata: 更新后的注意力元数据
        """
        if not self.needs_extra_input_slots:
            # Default EAGLE pathway: no reshaping of input tensors needed.
            # Simply rotate the input ids and leave the positions unchanged,
            # Inserting the next token ids at the last slot in each request.
            # 中文注释：默认 EAGLE 路径 - 不需要额外槽位
            # 简单旋转 input_ids 并在每个请求的最后一个槽位插入下一个 token
            if token_indices_to_sample is None:
                token_indices_to_sample = cad.query_start_loc[1:] - 1

            num_tokens = target_token_ids.shape[0]
            # Shift the input ids by one token.
            # E.g., [a1, b1, b2, c1, c2, c3] -> [b1, b2, c1, c2, c3, c3]
            # 中文注释：向左旋转 input_ids 一位
            self.input_ids[: num_tokens - 1] = target_token_ids[1:]
            # Replace the last token with the next token.
            # E.g., [b1, b2, c1, c2, c3, c3] -> [a2, b2, b3, c2, c3, c4]
            # 中文注释：在每个请求的最后一个槽位插入下一个 token
            self.input_ids[token_indices_to_sample] = next_token_ids

            # copy inputs to buffer for cudagraph
            # 中文注释：复制位置编码到缓冲区
            if self.uses_xdrope_dim > 0 and self.draft_uses_xdrope_dim == 0:
                target_positions = target_positions[0]
            self._set_positions(num_tokens, target_positions)

            # 中文注释：复制隐藏状态到缓冲区
            self.hidden_states[:num_tokens] = target_hidden_states

            return num_tokens, token_indices_to_sample, cad
        else:
            # 中文注释：草稿模型或并行草稿路径 - 需要额外槽位
            assert self.is_rejected_token_mask is not None
            assert self.is_masked_token_mask is not None
            # 1.
            # Call a custom triton kernel to copy input_ids and positions
            # into the correct slots in the preallocated buffers self.input_ids,
            # self.positions.
            # 中文注释：步骤 1 - 使用 Triton 内核复制和扩展输入
            batch_size = cad.batch_size()
            # Since we might have to copy a lot of data for prefills, we select the
            # block size based on the max query length and limit to max 256 slots/block.
            # 中文注释：计算 Triton 内核的块大小和网格大小
            max_num_tokens_per_request = (
                cad.max_query_len + self.net_num_new_slots_per_request
            )
            BLOCK_SIZE_TOKENS = min(256, next_power_of_2(max_num_tokens_per_request))
            num_blocks = (
                max_num_tokens_per_request + BLOCK_SIZE_TOKENS - 1
            ) // BLOCK_SIZE_TOKENS
            total_num_input_tokens = target_token_ids.shape[0]
            total_num_output_tokens = total_num_input_tokens + (
                self.net_num_new_slots_per_request * batch_size
            )

            # 中文注释：创建采样索引缓冲区
            token_indices_to_sample = torch.empty(
                batch_size * self.extra_slots_per_request,
                dtype=torch.int32,
                device=self.device,
            )

            # Destination indices to write target_hidden_states into drafting buffer.
            # 中文注释：创建隐藏状态映射缓冲区
            out_hidden_state_mapping = torch.empty(
                total_num_input_tokens, dtype=torch.int32, device=self.device
            )

            # Kernel grid: one program per request (row)
            # 中文注释：设置 Triton 内核网格
            grid = (batch_size, num_blocks)
            query_start_loc = cad.query_start_loc
            query_end_loc = cad.query_start_loc[1:] - 1
            if num_rejected_tokens_gpu is not None:
                query_end_loc = query_end_loc - num_rejected_tokens_gpu

            # 中文注释：执行 Triton 内核，复制和扩展输入。
            # 该内核的核心功能：
            # 1. 将 target 模型的 token_id 和 position 复制到 draft 模型的输入缓冲区
            # 2. 在每个请求末尾插入 padding slot（用于被拒绝的 token 或新增的 next_token）
            # 3. 在正确位置插入 next_token（本轮采样的新 token）
            # 4. 生成 is_rejected_token_mask 和 is_masked_token_mask 用于后续过滤
            # 5. 计算 token_indices_to_sample（新 token 在扩展后序列中的位置索引）
            # 6. 计算 out_hidden_state_mapping（target 隐藏状态到 draft 缓冲区的映射）
            copy_and_expand_eagle_inputs_kernel[grid](
                # (Padded) Inputs from the target model
                target_token_ids_ptr=target_token_ids,
                target_positions_ptr=target_positions,
                next_token_ids_ptr=next_token_ids,  # sampled tokens, one per request
                # Outputs to the drafting buffers
                out_input_ids_ptr=self.input_ids,
                out_positions_ptr=self.positions,  # Doesn't support mrope for now
                out_is_rejected_token_mask_ptr=self.is_rejected_token_mask,
                out_is_masked_token_mask_ptr=self.is_masked_token_mask,
                out_new_token_indices_ptr=token_indices_to_sample,
                out_hidden_state_mapping_ptr=out_hidden_state_mapping,
                # Input metadata
                query_start_loc_ptr=query_start_loc,
                query_end_loc_ptr=query_end_loc,
                padding_token_id=0,
                parallel_drafting_token_id=self.parallel_drafting_token_id,
                # Sizing info
                # Note that we can deduce batch_size for free from the grid size
                total_input_tokens=total_num_input_tokens,
                num_padding_slots_per_request=self.extra_slots_per_request,
                shift_input_ids=self.pass_hidden_states_to_model,
                BLOCK_SIZE_TOKENS=BLOCK_SIZE_TOKENS,
            )

            # 中文注释：处理隐藏状态（如果需要）
            if self.pass_hidden_states_to_model:
                assert self.parallel_drafting_hidden_state_tensor is not None
                self.hidden_states[out_hidden_state_mapping] = target_hidden_states
                # Use torch.where to avoid DtoH sync from boolean indexing
                # 中文注释：使用 torch.where 避免 CPU-GPU 同步
                mask = self.is_masked_token_mask[:total_num_output_tokens]
                torch.where(
                    mask.unsqueeze(1),
                    self.parallel_drafting_hidden_state_tensor,
                    self.hidden_states[:total_num_output_tokens],
                    out=self.hidden_states[:total_num_output_tokens],
                )

            # 2.
            # Recompute the slot mapping based on the new positions and
            # rejection mask.
            # 中文注释：步骤 2 - 重新计算 slot mapping
            assert self.block_size > 0, "block_size has not been initialized."
            new_slot_mapping = compute_new_slot_mapping(
                cad=cad,
                new_positions=self.positions[:total_num_output_tokens],
                is_rejected_token_mask=self.is_rejected_token_mask[
                    :total_num_output_tokens
                ],
                block_size=self.block_size,
                num_new_tokens=self.net_num_new_slots_per_request,
                max_model_len=self.max_model_len,
            )

            # 3. Update the common attention metadata with the new (meta)data
            # 中文注释：步骤 3 - 更新注意力元数据
            new_cad = extend_all_queries_by_N(
                cad,
                N=self.net_num_new_slots_per_request,
                arange=self.arange,
                new_slot_mapping=new_slot_mapping,
            )

            return total_num_output_tokens, token_indices_to_sample, new_cad

    def build_model_inputs_first_pass(
        self,
        num_tokens: int,
        num_input_tokens: int,
        mm_embed_inputs: tuple[list[torch.Tensor], torch.Tensor] | None,
    ) -> tuple[dict[str, Any], int]:
        """
        构建第一次前向传播的模型输入。

        根据是否支持多模态输入，构建不同的模型输入参数。

        参数:
            num_tokens: 总 token 数量
            num_input_tokens: 实际输入 token 数量（可能包含填充）
            mm_embed_inputs: 多模态嵌入输入

        返回:
            元组 (model_kwargs, slot_mapping_size)
            - model_kwargs: 模型输入参数字典
            - slot_mapping_size: slot mapping 的大小
        """
        if self.supports_mm_inputs:
            # 中文注释：处理多模态输入
            mm_embeds, is_mm_embed = mm_embed_inputs or (None, None)

            # 中文注释：将 token ID 转换为嵌入向量
            self.inputs_embeds[:num_tokens] = self.model.embed_input_ids(
                self.input_ids[:num_tokens],
                multimodal_embeddings=mm_embeds,
                is_multimodal=is_mm_embed,
            )

            input_ids = None
            inputs_embeds = self.inputs_embeds[:num_input_tokens]
        else:
            # 中文注释：纯文本输入，直接使用 token ID
            input_ids = self.input_ids[:num_input_tokens]
            inputs_embeds = None

        model_kwargs = {
            "input_ids": input_ids,
            "positions": self._get_positions(num_input_tokens),
            "inputs_embeds": inputs_embeds,
        }
        # 中文注释：如果 draft 模型需要接收 target 模型的隐藏状态（如 EAGLE 方法），
        # 则将 hidden_states 也加入模型输入参数中。
        if self.pass_hidden_states_to_model:
            model_kwargs["hidden_states"] = self.hidden_states[:num_input_tokens]

        return model_kwargs, num_input_tokens

    # [流程说明] 为 draft 模型的各注意力层组构建注意力元数据。
    # 每个 AttentionGroup 对应一种注意力后端（如 FlashAttention），
    # 内部包含多个使用相同后端的 draft 层。此方法为每个 group 生成
    # 对应的 attn_metadata，并建立 layer_name -> attn_metadata 的映射。
    def build_per_group_and_layer_attn_metadata(
        self, common_attn_metadata: CommonAttentionMetadata, draft_index: int = 0
    ) -> tuple[list[object], dict[str, object]]:
        per_group_attn_metadata: list[object] = []
        per_layer_attn_metadata: dict[str, object] = {}
        for attn_group in self.draft_attn_groups:
            # 中文注释：调用注意力组的元数据构建器，为 draft 模型构建注意力元数据。
            # build_for_drafting 会根据 common_attn_metadata 和 draft_index
            # 生成适合 speculative decoding 的注意力元数据。
            attn_metadata = attn_group.get_metadata_builder().build_for_drafting(
                common_attn_metadata=common_attn_metadata, draft_index=draft_index
            )
            per_group_attn_metadata.append(attn_metadata)
            # 中文注释：将同一注意力组内所有层名映射到相同的 attn_metadata，
            # 这样模型 forward 时每层能通过 layer_name 查找对应的注意力元数据。
            for layer_name in attn_group.layer_names:
                per_layer_attn_metadata[layer_name] = attn_metadata
        return per_group_attn_metadata, per_layer_attn_metadata

    # [流程说明] 判断 draft 模型的 forward 是否返回元组。
    # MTP、draft_model、dflash 方法返回自定义输出对象，而非元组。
    def model_returns_tuple(self) -> bool:
        return self.method not in ("mtp", "draft_model", "dflash")

    # [流程说明] 在 CPU 上准备 speculative decoding 所需的 next_token_ids。
    # 这是 draft 模型前向传播的输入起点：每个请求需要一个 token 作为 draft 的起始 token。
    # 对于正常解码（有采样结果），直接取最后一个采样 token；
    # 对于 partial prefill（首步无采样结果），从请求状态中回退获取下一个待生成的 token。
    def prepare_next_token_ids_cpu(
        self,
        sampled_token_ids: list[list[int]],
        requests: dict[str, CachedRequestState],
        gpu_input_batch: InputBatch,
        num_scheduled_tokens: dict[str, int],
    ) -> torch.Tensor:
        """
        This function is used to prepare the inputs for speculative decoding.
        It calculates the next token ids for each request based on the sampled
        token ids from the CPU. If a request has no sampled token ids (e.g.,
        during the initial decoding steps), it falls back to using the request
        state to get the next token id.

        中文注释：在 CPU 上准备投机解码所需的 next_token_ids。

        这是 draft 模型前向传播的输入起点：每个请求需要一个 token 作为 draft 的起始 token。

        处理逻辑：
        1. 正常解码：取最后一个采样 token 作为下一个 token
        2. Partial prefill：从请求状态中回退获取下一个待生成的 token

        参数:
            sampled_token_ids: 每个请求的采样 token ID 列表
            requests: 请求状态字典
            gpu_input_batch: GPU 输入批次
            num_scheduled_tokens: 每个请求的调度 token 数量

        返回:
            next_token_ids: 下一个 token ID 张量，形状 (batch_size,)
        """
        req_ids = gpu_input_batch.req_ids
        next_token_ids: list[int] = []
        for i, token_ids in enumerate(sampled_token_ids):
            if token_ids:
                # Common case.
                # 中文注释：正常情况下，取最后一个采样到的 token 作为下一个 token。
                # 这是 target 模型本轮生成的最后一个 token，将作为 draft 的起始输入。
                next_token_id = token_ids[-1]
            else:
                # Partial prefill (rare case).
                # Get the next token id from the request state.
                # 中文注释：partial prefill 的少见场景——首轮 prefill 未生成任何 token。
                # 此时需要从请求状态中获取序列指定位置的 token 作为下一个输入。
                req_id = req_ids[i]
                req_state = requests[req_id]
                seq_len = req_state.num_computed_tokens + num_scheduled_tokens[req_id]
                next_token_id = req_state.get_token_id(seq_len)
            next_token_ids.append(next_token_id)
        # 中文注释：将 CPU 上计算的 next_token_ids 转为 GPU tensor，
        # 供后续 draft 模型前向传播使用。
        next_token_ids = torch.tensor(
            next_token_ids, dtype=torch.int32, device=self.input_ids.device
        )
        return next_token_ids

    def prepare_next_token_ids_padded(
        self,
        sampled_token_ids: torch.Tensor,
        requests: dict[str, CachedRequestState],
        gpu_input_batch: InputBatch,
        discard_request_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        This function is used to prepare the inputs for speculative decoding.
        It calculates the next token ids and the number of valid sampled tokens
        for each request, considering the "discarded" requests whose next token
        is not sampled and comes from `request.get_token_id()` instead. This is denoted
        the "backup" token id. It also counts rejected tokens via `sampled_token_ids`.

        中文注释：准备填充批次的下一个 token ID。

        处理投机解码中的"被丢弃"请求（discard_request_mask）：
        - 正常请求：从采样结果中获取下一个 token
        - 被丢弃的请求：从请求状态中获取备用 token

        同时统计每个请求的有效采样 token 数量。

        参数:
            sampled_token_ids: 采样的 token ID 张量，形状 (batch_size, num_tokens)
            requests: 请求状态字典
            gpu_input_batch: GPU 输入批次
            discard_request_mask: 被丢弃请求的掩码

        返回:
            元组 (next_token_ids, valid_sampled_tokens_count)
            - next_token_ids: 下一个 token ID，形状 (batch_size,)
            - valid_sampled_tokens_count: 有效采样 token 数量，形状 (batch_size,)
        """
        # Precompute backup token IDs for discarded requests.
        # 中文注释：预计算被丢弃请求的备用 token ID。
        # 在投机解码中，部分请求可能被"丢弃"（discard），即本轮不参与投机验证，
        # 其下一个 token 需要从请求的已知序列中获取（而非从采样结果中获取）。
        # get_token_id(seq_len - 1) 获取的是该请求在当前序列长度位置的 token。
        num_reqs = gpu_input_batch.num_reqs
        for i in range(num_reqs):
            self.backup_next_token_ids.np[i] = requests[
                gpu_input_batch.req_ids[i]
            ].get_token_id(gpu_input_batch.num_tokens_no_spec[i] - 1)
        self.backup_next_token_ids.copy_to_gpu(num_reqs)
        backup_tokens_gpu = self.backup_next_token_ids.gpu

        batch_size, num_tokens = sampled_token_ids.shape
        device = sampled_token_ids.device

        assert discard_request_mask.dtype == torch.bool
        assert backup_tokens_gpu.dtype == torch.int32

        # 中文注释：创建输出缓冲区
        next_token_ids = torch.empty(batch_size, dtype=torch.int32, device=device)
        valid_sampled_tokens_count = next_token_ids.new_empty(batch_size)

        # Kernel grid: one program per request (row)
        grid = (batch_size,)

        # Find the next power of 2 for block sizes
        # 中文注释：使用 Triton 内核高效处理填充批次。
        # 该内核并行处理每个请求：
        # 1. 遍历该请求的采样 token 序列
        # 2. 统计有效采样 token 数量（遇到无效值即停止）
        # 3. 取最后一个有效 token 作为 next_token_id
        # 4. 对于被丢弃的请求，使用 backup token 作为 next_token_id
        BLOCK_SIZE_TOKENS = next_power_of_2(num_tokens)
        eagle_prepare_next_token_padded_kernel[grid](
            sampled_token_ids,
            discard_request_mask,
            backup_tokens_gpu,
            next_token_ids,
            valid_sampled_tokens_count,
            gpu_input_batch.vocab_size,
            num_tokens,
            batch_size,
            sampled_token_ids.stride(0),
            BLOCK_SIZE_TOKENS=BLOCK_SIZE_TOKENS,
        )

        return next_token_ids, valid_sampled_tokens_count

    def prepare_inputs_padded(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        spec_decode_metadata: SpecDecodeMetadata,
        valid_sampled_tokens_count: torch.Tensor,
    ) -> tuple[CommonAttentionMetadata, torch.Tensor, torch.Tensor]:
        """
        This function is used to prepare the inputs for speculative decoding
        It updates the common_attn_metadata for speculative decoding,
        but does not consider the rejected tokens. Instead, all tokens
        are included as inputs to the speculator, with the rejected tokens
        used as padding and filtered out later by `token_indices_to_sample`.
        No blocking CPU operations should be introduced in this function.

        中文注释：准备填充批次的投机解码输入。

        此方法更新 common_attn_metadata 以适应投机解码，但不考虑被拒绝的 token。
        所有 token 都作为投机器的输入，被拒绝的 token 用作填充，
        稍后通过 token_indices_to_sample 过滤。

        关键点：此函数不应引入阻塞的 CPU 操作。

        参数:
            common_attn_metadata: 通用注意力元数据
            spec_decode_metadata: 投机解码元数据
            valid_sampled_tokens_count: 有效采样 token 数量

        返回:
            元组 (spec_common_attn_metadata, token_indices_to_sample, num_rejected_tokens_gpu)
            - spec_common_attn_metadata: 更新后的注意力元数据
            - token_indices_to_sample: 需要采样的 token 索引
            - num_rejected_tokens_gpu: 被拒绝的 token 数量
        """
        num_reqs = common_attn_metadata.num_reqs
        device = valid_sampled_tokens_count.device

        # 中文注释：创建输出缓冲区
        token_indices_to_sample = torch.empty(
            (num_reqs,), dtype=torch.int32, device=device
        )
        num_rejected_tokens_gpu = torch.empty(
            (num_reqs,), dtype=torch.int32, device=device
        )

        # 中文注释：使用 Triton 内核计算采样索引和被拒绝 token 数量。
        # 该内核并行处理每个请求：
        # 1. 从 cu_num_draft_tokens 获取该请求的草稿 token 总数
        # 2. 从 valid_sampled_tokens_count 获取有效采样 token 数量
        # 3. 计算 num_rejected = draft_count - (valid_count - 1)
        #    （valid_count 包含新采样的 1 个 token，所以减 1）
        # 4. 计算 token_indices_to_sample：新采样 token 在扩展序列中的位置索引
        grid = (num_reqs,)
        eagle_prepare_inputs_padded_kernel[grid](
            spec_decode_metadata.cu_num_draft_tokens,
            valid_sampled_tokens_count,
            common_attn_metadata.query_start_loc,
            token_indices_to_sample,
            num_rejected_tokens_gpu,
            num_reqs,
        )

        # 中文注释：计算新的查询长度
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        new_query_len_per_req = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]

        total_num_tokens = query_start_loc_cpu[-1].item()

        # 中文注释：创建投机解码专用的注意力元数据。
        # 填充批次模式下，query_start_loc 和 seq_lens 保持不变（因为 padding token 仍然占据位置），
        # 但 num_actual_tokens 更新为去除 padding 后的实际 token 总数，
        # max_query_len 更新为实际最大查询长度。
        # block_table_tensor 复用原始的（padding token 的 KV 在 block table 中仍然有效）。
        spec_common_attn_metadata = CommonAttentionMetadata(
            query_start_loc=common_attn_metadata.query_start_loc,
            seq_lens=common_attn_metadata.seq_lens,
            query_start_loc_cpu=query_start_loc_cpu,
            _seq_lens_cpu=common_attn_metadata._seq_lens_cpu,
            _num_computed_tokens_cpu=common_attn_metadata._num_computed_tokens_cpu,
            seq_lens_cpu_upper_bound=common_attn_metadata.seq_lens_cpu_upper_bound,
            num_reqs=common_attn_metadata.num_reqs,
            num_actual_tokens=total_num_tokens,
            max_query_len=new_query_len_per_req.max().item(),
            max_seq_len=common_attn_metadata.max_seq_len,
            block_table_tensor=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping[:total_num_tokens],
            causal=True,
            dcp_local_seq_lens=common_attn_metadata.dcp_local_seq_lens,
        )

        return (
            spec_common_attn_metadata,
            token_indices_to_sample,
            num_rejected_tokens_gpu,
        )

    def prepare_inputs(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        sampled_token_ids: list[list[int]],
        num_draft_tokens: list[int],
    ) -> tuple[CommonAttentionMetadata, torch.Tensor]:
        """
        This function is used to prepare the inputs for speculative decoding.
        It updates to the common_attn_metadata to account for the rejected
        tokens (and newly sampled tokens). It also returns the token indices
        of the tokens that should be fed to the speculator.

        中文注释：准备投机解码的输入（非填充批次版本）。

        此方法更新 common_attn_metadata 以考虑被拒绝的 token 和新采样的 token。
        同时返回应该传递给投机器的 token 索引。

        参数:
            common_attn_metadata: 通用注意力元数据
            sampled_token_ids: 每个请求的采样 token ID 列表
            num_draft_tokens: 每个请求的草稿 token 数量

        返回:
            元组 (spec_common_attn_metadata, token_indices)
            - spec_common_attn_metadata: 更新后的注意力元数据
            - token_indices: 需要传递给投机器的 token 索引
        """
        # E.g.
        #  common_attn_metadata.query_start_loc{_cpu}:
        #       [0, q1, q1 + q2, q1 + q2 + q3]
        #  common_attn_metadata.seq_lens{_cpu}: [s1, s2, s3]
        #  num_rejected_tokens: [n1, n2, n3]
        # This function computes the intermediate values:
        #  num_tokens_per_req: [q1 - n1, q2 - n2, q3 - n3]
        # And returns:
        #  common_attn_metadata.query_start_loc{_cpu}:
        #       [0, q1 - n1, q1 + q2 - n1 - n2, q1 + q2 + q3 - n1 - n2 - n3]
        #  common_attn_metadata.seq_lens{_cpu}:
        #       [s1 - n1 + 1, s2 - n2 + 1, s3 - n3 + 1]
        #  token_indices: [0, 1, ..., q1 - n1 - 1,
        #                 q1, q1 + 1, ..., q1 + q2 - n2 - 1,
        #                 q1 + q2, q1 + q2 + 1, ..., q1 + q2 + q3 - n3 - 1]

        # 中文注释：计算每个请求被拒绝的 token 数量。
        # num_draft_tokens[i] 是第 i 个请求的草稿 token 数量。
        # sampled_token_ids[i] 是第 i 个请求本轮采样到的 token（包含被接受的草稿 token + 新采样的 token）。
        # 如果一个请求有 n 个草稿 token，则期望采样到 n+1 个 token（n 个被验证 + 1 个新采样）。
        # 实际采样数少于 n+1 说明有部分草稿 token 被拒绝。
        num_rejected_tokens = [
            n + 1 - len(sampled_token_ids[i]) if n > 0 else 0
            for i, n in enumerate(num_draft_tokens)
        ]
        num_rejected_tokens = torch.tensor(num_rejected_tokens, dtype=torch.int32)

        device = common_attn_metadata.query_start_loc.device
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        # upper_bound - rejected = actual post-rejection seq_lens (no D2H sync).
        # 中文注释：计算新的序列长度（避免 CPU-GPU 同步）
        assert common_attn_metadata.seq_lens_cpu_upper_bound is not None
        new_seq_lens_cpu = (
            common_attn_metadata.seq_lens_cpu_upper_bound - num_rejected_tokens
        )

        # [0, q1, q1 + q2, q1 + q2 + q3] -> [q1, q2, q3]
        # 中文注释：计算每个请求的查询长度
        new_query_len_per_req = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
        # [q1, q2, q3] -> [q1 - n1, q2 - n2, q3 - n3]
        # 中文注释：计算每个请求的新 token 数量（减去被拒绝的）
        new_num_tokens_per_req = new_query_len_per_req - num_rejected_tokens
        new_num_tokens_per_req_np = new_num_tokens_per_req.numpy()

        # [q1 - n1, q2 - n2, q3 - n3] ->
        # [0, q1 - n1, q1 + q2 - n1 - n2, q1 + q2 + q3 - n1 - n2 - n3]
        # 中文注释：计算新的查询起始位置
        new_query_start_loc_cpu = torch.zeros(
            query_start_loc_cpu.shape,
            dtype=torch.int32,
            pin_memory=is_pin_memory_available(),
        )
        new_query_start_loc_np = new_query_start_loc_cpu.numpy()
        np.cumsum(new_num_tokens_per_req_np, out=new_query_start_loc_np[1:])

        total_num_tokens = new_query_start_loc_np[-1]
        # Example assuming num_tokens_per_req_np = [2, 4, 3]
        # this implies that `new_query_start_locs` is:
        # [0, 2, 6, 9] ->
        # [0, 0, 2, 2, 2, 2, 6, 6, 6]
        #  _r1_  ____r2____  ___r3__
        # 中文注释：扩展查询起始位置以匹配 token 模式
        new_query_start_locs_expanded = np.repeat(
            new_query_start_loc_np[:-1], new_num_tokens_per_req_np
        )
        # [0, 1, 2, 3, 4, 5, 6, 7, 8] ->
        # [0, 1, 0, 1, 2, 3, 0, 1, 2]
        #  _r1_  ____r2____  ___r3__
        # 中文注释：计算 token 偏移量
        token_offsets = (
            self.token_arange_np[:total_num_tokens] - new_query_start_locs_expanded
        )

        # Expand starting positions to match token pattern
        # [0, q1, q1 + q2] ->
        # [0, 0, q1, q1, q1, q1, q1 + q2, q1 + q2, q1 + q2]
        #  _r1_  _____r2_______  ___________r3____________
        # 中文注释：扩展旧的查询起始位置
        old_query_start_locs_expanded = np.repeat(
            query_start_loc_cpu[:-1].numpy(), new_num_tokens_per_req_np
        )
        # Final token indices are:
        # [0, 1,                                // req 1
        #  q1 + 0, q1 + 1, q1 + 2, q1 + 3,       // req 2
        #  q1 + q2 + 0, q1 + q2 + 1, q1 + q2 + 2] // req 3
        # 中文注释：计算最终的 token 索引
        token_indices_np = token_offsets + old_query_start_locs_expanded
        token_indices = torch.from_numpy(token_indices_np).to(device, non_blocking=True)

        # 中文注释：创建投机解码专用的注意力元数据。
        # 非填充模式下，query_start_loc 和 seq_lens 都需要重新计算：
        # - query_start_loc 反映去除被拒绝 token 后每个请求的实际起始位置
        # - seq_lens 反映去除被拒绝 token 后的实际序列长度
        # - slot_mapping 按照 token_indices 重新索引，只保留有效 token 的 slot
        # - causal=True 因为投机解码仍然使用因果注意力
        spec_common_attn_metadata = CommonAttentionMetadata(
            query_start_loc=new_query_start_loc_cpu.to(device, non_blocking=True),
            seq_lens=new_seq_lens_cpu.to(device, non_blocking=True),
            query_start_loc_cpu=new_query_start_loc_cpu,
            _seq_lens_cpu=new_seq_lens_cpu,
            _num_computed_tokens_cpu=common_attn_metadata._num_computed_tokens_cpu,
            seq_lens_cpu_upper_bound=new_seq_lens_cpu,
            num_reqs=common_attn_metadata.num_reqs,
            num_actual_tokens=total_num_tokens,
            max_query_len=new_query_len_per_req.max().item(),
            max_seq_len=new_seq_lens_cpu.max().item(),
            block_table_tensor=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping[token_indices],
            causal=True,
            dcp_local_seq_lens=common_attn_metadata.dcp_local_seq_lens,
        )

        return spec_common_attn_metadata, token_indices

    def get_model_name(self, model: nn.Module) -> str:
        """
        获取模型的类名称。

        参数:
            model: 模型实例

        返回:
            模型类的名称字符串
        """
        # 中文注释：处理多 GPU 情况（模型被 DataParallel 包装）
        if hasattr(model, "module"):  # multi-GPU
            model = model.module
        return model.__class__.__name__

    def _create_draft_vllm_config(self) -> VllmConfig:
        """Return a VllmConfig with kernel-level overrides for the proposer.
        Subclasses may override to apply additional config changes.

        中文注释：创建草稿模型的 vLLM 配置。

        基于基础配置，应用草稿模型特定的内核级覆盖：
        1. MoE 后端配置
        2. 注意力后端配置

        返回:
            草稿模型的 vLLM 配置对象
        """
        spec_cfg = self.speculative_config
        base = self.vllm_config

        # 中文注释：应用 MoE 后端配置（如果指定）
        if spec_cfg.moe_backend is not None:
            base = replace(
                base,
                kernel_config=replace(
                    base.kernel_config,
                    moe_backend=spec_cfg.moe_backend,
                ),
            )

        # Note (matt): Never inherit the attention backend from base, because there are
        # many opportunities for incompatibility, so we always independently autoselect
        # unless explicitly specified in the speculative config.
        # 中文注释：应用注意力后端配置（不继承基础配置，避免兼容性问题）
        base = replace(
            base,
            attention_config=replace(
                base.attention_config,
                backend=spec_cfg.attention_backend,
            ),
        )

        return base

    def _get_model(self) -> nn.Module:
        """
        Default method to call get_model(). Can be overridden by subclasses which
        need to customize model loading.

        中文注释：获取草稿模型实例。

        默认方法调用 get_model() 加载草稿模型。
        子类可以覆盖此方法以自定义模型加载。

        返回:
            草稿模型实例
        """
        from vllm.compilation.backends import set_model_tag

        # 中文注释：创建草稿模型的 vLLM 配置
        draft_vllm_config = self._create_draft_vllm_config()
        # 中文注释：设置模型标签为 "eagle_head"，用于编译优化
        with set_model_tag("eagle_head"):
            # 中文注释：加载草稿模型
            model = get_model(
                vllm_config=draft_vllm_config,
                model_config=self.speculative_config.draft_model_config,
                load_config=self.speculative_config.draft_load_config,
            )
        return model

    def load_model(self, target_model: nn.Module) -> None:
        """
        加载草稿模型并初始化注意力层。

        此方法：
        1. 获取目标模型的注意力层名称
        2. 加载草稿模型
        3. 初始化草稿模型的注意力层和 KV cache 配置

        参数:
            target_model: 目标模型实例
        """
        # 中文注释：获取目标模型的注意力层名称
        target_attn_layer_names = set(
            get_layers_from_vllm_config(
                self.vllm_config,
                AttentionLayerBase,  # type: ignore[type-abstract]
            ).keys()
        )

        self.model = self._get_model()

        # Find draft layers (attention layers added by draft model)
        all_attn_layers = get_layers_from_vllm_config(
            self.vllm_config,
            AttentionLayerBase,  # type: ignore[type-abstract]
        )
        # Filter to only layers that have KV cache specs.
        self._draft_attn_layer_names = {
            name
            for name in (set(all_attn_layers.keys()) - target_attn_layer_names)
            if all_attn_layers[name].get_kv_cache_spec(self.vllm_config) is not None
        }

        # 中文注释：检测草稿模型是否支持多模态输入。
        # 即使 target 模型支持多模态，draft 模型可能只支持纯文本。
        # 通过尝试调用 embed_input_ids 来检测，如果不支持则回退到纯文本模式。
        if self.supports_mm_inputs:
            # Even if the target model is multimodal, we can also use
            # text-only draft models
            try:
                dummy_input_ids = torch.tensor([[1]], device=self.input_ids.device)
                self.model.embed_input_ids(dummy_input_ids, multimodal_embeddings=None)
            except (NotImplementedError, AttributeError, TypeError):
                logger.warning(
                    "Draft model does not support multimodal inputs, "
                    "falling back to text-only mode"
                )
                self.supports_mm_inputs = False

        # 中文注释：处理多模态模型的 image_token_index 配置。
        # 不同的多模态架构将 image_token_index 存放在不同的配置字段中，
        # 这里需要根据 target 模型的类名将其映射到 draft 模型的配置中，
        # 以确保 draft 模型能正确识别图像 token。
        if supports_multimodal(target_model):
            # handle multimodality
            assert hasattr(target_model, "config")
            if self.get_model_name(target_model) in [
                "Cohere2VisionForConditionalGeneration",
                "Exaone4_5_ForConditionalGeneration",
                "GlmOcrForConditionalGeneration",
                "HunYuanVLForConditionalGeneration",
                "InternS2PreviewForConditionalGeneration",
                "MiMoV2OmniForCausalLM",
                "Qwen2_5_VLForConditionalGeneration",
                "Qwen3_5ForConditionalGeneration",
                "Qwen3_5MoeForConditionalGeneration",
                "Qwen3VLForConditionalGeneration",
                "Qwen3VLMoeForConditionalGeneration",
                "Gemma4ForConditionalGeneration",
                "Step3p7ForConditionalGeneration",
            ]:
                self.model.config.image_token_index = target_model.config.image_token_id
            elif self.get_model_name(target_model) == "PixtralForConditionalGeneration":
                self.model.config.image_token_index = (
                    target_model.config.vision_config.image_token_id
                )
            elif self.get_model_name(target_model) == "KimiK25ForConditionalGeneration":
                self.model.config.image_token_index = (
                    target_model.config.media_placeholder_token_id
                )
            else:
                self.model.config.image_token_index = (
                    target_model.config.image_token_index
                )
            target_language_model = cast(
                SupportsMultiModal, target_model
            ).get_language_model()
        else:
            target_language_model = target_model

        self._maybe_share_embeddings(target_language_model)
        self._maybe_share_lm_head(target_language_model)

        # 中文注释：并行草稿模式下初始化隐藏状态模板张量。
        # parallel_drafting_hidden_state_tensor 是一个预分配的张量，
        # 用于在并行草稿模式中为 padding 位置填充默认的隐藏状态。
        # EAGLE3 方法使用 mask_hidden 选择哪些隐藏状态维度是有效的，
        # 然后通过 combine_hidden_states 投影得到默认值。
        if (
            self.parallel_drafting
            and self.pass_hidden_states_to_model
            and self.parallel_drafting_hidden_state_tensor is not None
        ):
            flat_mask = self.model.mask_hidden.view(-1)
            if self.eagle3_use_aux_hidden_state:
                # EAGLE3: mask_hidden stores all aux hidden states,
                # project through combine_hidden_states
                self.parallel_drafting_hidden_state_tensor.copy_(
                    self.model.combine_hidden_states(flat_mask)
                )
            else:
                self.parallel_drafting_hidden_state_tensor.copy_(flat_mask)

    def _maybe_share_embeddings(self, target_language_model: nn.Module) -> None:
        """
        Some draft models may not have their own embedding layers, and some may
        have a duplicate copy of the target model's embedding layers. In these cases,
        we share the target model's embedding layers with the draft model to save
        memory.

        中文注释：尝试共享目标模型的嵌入层。

        某些草稿模型可能没有自己的嵌入层，或者有与目标模型相同的嵌入层副本。
        在这些情况下，我们共享目标模型的嵌入层以节省内存。

        参数:
            target_language_model: 目标语言模型
        """
        # 中文注释：只在单 pipeline 并行时共享嵌入层
        if get_pp_group().world_size == 1:
            inner_model = getattr(target_language_model, "model", None)
            if inner_model is None:
                raise AttributeError("Target model does not have 'model' attribute")
            if hasattr(inner_model, "embed_tokens"):
                target_embed_tokens = inner_model.embed_tokens
            elif hasattr(inner_model, "embedding"):
                target_embed_tokens = inner_model.embedding
            else:
                raise AttributeError(
                    "Target model does not have 'embed_tokens' or 'embedding' attribute"
                )

            share_embeddings = False
            if hasattr(self.model, "has_own_embed_tokens"):
                # EAGLE model
                # 中文注释：EAGLE 模型 - 检查是否有自己的嵌入层
                if not self.model.has_own_embed_tokens:
                    share_embeddings = True
                    logger.info(
                        "Detected EAGLE model without its own embed_tokens in the"
                        " checkpoint. Sharing target model embedding weights with the"
                        " draft model."
                    )
                elif (
                    isinstance(target_embed_tokens.weight, torch.Tensor)
                    and isinstance(self.model.model.embed_tokens.weight, torch.Tensor)
                    # TODO: Offload to CPU for comparison to avoid extra GPU memory
                    # usage in CI testing environments with limited GPU memory
                    and torch.equal(
                        target_embed_tokens.weight.cpu(),
                        self.model.model.embed_tokens.weight.cpu(),
                    )
                ):
                    share_embeddings = True
                    logger.info(
                        "Detected EAGLE model with embed_tokens identical to the target"
                        " model. Sharing target model embedding weights with the draft"
                        " model."
                    )
                else:
                    logger.info(
                        "Detected EAGLE model with distinct embed_tokens weights. "
                        "Keeping separate embedding weights from the target model."
                    )
            else:
                # MTP model
                # 中文注释：MTP 模型 - 默认共享嵌入层
                share_embeddings = True
                logger.info(
                    "Detected MTP model. "
                    "Sharing target model embedding weights with the draft model."
                )

            # 中文注释：如果需要共享，替换草稿模型的嵌入层
            if share_embeddings:
                if hasattr(self.model.model, "embed_tokens"):
                    del self.model.model.embed_tokens
                self.model.model.embed_tokens = target_embed_tokens
        else:
            logger.info(
                "The draft model's vocab embedding will be loaded separately"
                " from the target model."
            )

    def _maybe_share_lm_head(self, target_language_model: nn.Module) -> None:
        """
        Some draft models may not have their own LM head, and some may have a
        duplicate copy of the target model's LM head. In these cases, we share
        the target model's LM head with the draft model to save memory.

        中文注释：尝试共享目标模型的 LM head。

        某些草稿模型可能没有自己的 LM head，或者有与目标模型相同的 LM head 副本。
        在这些情况下，我们共享目标模型的 LM head 以节省内存。

        参数:
            target_language_model: 目标语言模型
        """
        share_lm_head = False
        if hasattr(self.model, "has_own_lm_head"):
            # EAGLE model
            # 中文注释：EAGLE 模型 - 检查是否有自己的 LM head
            if not self.model.has_own_lm_head:
                share_lm_head = True
                logger.info(
                    "Detected EAGLE model without its own lm_head in the checkpoint. "
                    "Sharing target model lm_head weights with the draft model."
                )
            elif (
                hasattr(target_language_model, "lm_head")
                and hasattr(target_language_model.lm_head, "weight")
                and hasattr(self.model.lm_head, "weight")
                and isinstance(target_language_model.lm_head.weight, torch.Tensor)
                and isinstance(self.model.lm_head.weight, torch.Tensor)
                # TODO: Offload to CPU for comparison to avoid extra GPU memory
                # usage in CI testing environments with limited GPU memory
                and torch.equal(
                    target_language_model.lm_head.weight.cpu(),
                    self.model.lm_head.weight.cpu(),
                )
            ):
                share_lm_head = True
                logger.info(
                    "Detected EAGLE model with lm_head identical to the target model. "
                    "Sharing target model lm_head weights with the draft model."
                )
            else:
                logger.info(
                    "Detected EAGLE model with distinct lm_head weights. "
                    "Keeping separate lm_head weights from the target model."
                )
        else:
            # MTP model
            # 中文注释：MTP 模型 - 默认共享 LM head
            share_lm_head = True
            logger.info(
                "Detected MTP model. "
                "Sharing target model lm_head weights with the draft model."
            )

        # 中文注释：如果需要共享，替换草稿模型的 LM head
        if share_lm_head and hasattr(target_language_model, "lm_head"):
            if hasattr(self.model, "lm_head"):
                del self.model.lm_head
            self.model.lm_head = target_language_model.lm_head

            # MTP models call compute_logits via shared_head.head (a
            # ParallelLMHead inside each MTP layer), not self.model.lm_head.
            # If the checkpoint omits a copy of the lm_head weights at the
            # MTP layer path, shared_head.head stays uninitialised and
            # produces NaN logits. Always share it explicitly.
            # 中文注释：MTP 模型通过 shared_head.head 计算 logits，需要显式共享
            inner = getattr(self.model, "model", None)
            layers = getattr(inner, "layers", None) if inner else None
            if layers is not None:
                items = layers.values() if isinstance(layers, nn.ModuleDict) else layers
                for layer in items:
                    sh = getattr(layer, "shared_head", None)
                    if sh is not None and hasattr(sh, "head"):
                        del sh.head
                        sh.head = target_language_model.lm_head
                        logger.info(
                            "Shared target model lm_head with MTP shared_head.head."
                        )

        # 中文注释：共享 topk_indices_buffer（如果存在）。
        # topk_indices_buffer 用于 MTP 模型中将 target 词表的 token 索引
        # 映射到 draft 模型的本地词表。共享此缓冲区可以避免重复分配显存，
        # 并确保 draft 和 target 模型使用相同的词表映射关系。
        if hasattr(target_language_model.model, "topk_indices_buffer"):
            if hasattr(self.model.model, "topk_indices_buffer"):
                del self.model.model.topk_indices_buffer
            self.model.model.topk_indices_buffer = (
                target_language_model.model.topk_indices_buffer
            )
            logger.info(
                "Detected MTP model with topk_indices_buffer. "
                "Sharing target model topk_indices_buffer with the draft model."
            )

        # 中文注释：检查本地 argmax 归约优化的兼容性。
        # use_local_argmax_reduction 是一种通信优化：draft 模型在本地对 logits
        # 取 argmax（或 top-k），只将结果（O(2*tp_size)）而非完整 logits（O(vocab_size)）
        # 通过 all-gather 传输，大幅减少 tensor parallel 的通信量。
        # 但这要求 draft 模型实现 get_top_tokens() 方法。
        if self.use_local_argmax_reduction:
            if not hasattr(self.model, "get_top_tokens"):
                raise ValueError(
                    "use_local_argmax_reduction is enabled but draft model "
                    f"{self.model.__class__.__name__} does not implement "
                    "get_top_tokens()."
                )
            # Warn if draft model has vocab remapping, which forces fallback
            # to the full-logits path (negating the optimization).
            if (
                hasattr(self.model, "draft_id_to_target_id")
                and self.model.draft_id_to_target_id is not None
            ):
                logger.warning(
                    "use_local_argmax_reduction is enabled but draft model "
                    "uses draft_id_to_target_id vocab remapping. The "
                    "optimization will be bypassed (falling back to full "
                    "logits gather + argmax)."
                )
            else:
                logger.info(
                    "Using local argmax reduction for draft token generation "
                    "(communication: O(2*tp_size) vs O(vocab_size))."
                )

    @torch.inference_mode()
    def dummy_run(
        self,
        num_tokens: int,
        use_cudagraphs: bool = True,
        is_graph_capturing: bool = False,
        slot_mappings: dict[str, torch.Tensor] | None = None,
    ) -> None:
        """
        执行虚拟前向传播（用于预热和 CUDA Graph 捕获）。

        此方法用于：
        1. 预热模型（分配内存、编译 kernel）
        2. 捕获 CUDA Graph（如果启用）
        3. 测试模型是否能正常运行

        参数:
            num_tokens: token 数量
            use_cudagraphs: 是否使用 CUDA Graph
            is_graph_capturing: 是否正在捕获 CUDA Graph
            slot_mappings: slot mapping 字典
        """
        # FIXME: when using tree-based specdec, adjust number of forward-passes
        # according to the depth of the tree.
        # 中文注释：确定是否只执行一次前向传播。
        # CUDA Graph 捕获时只需执行一次（捕获计算图），
        # 并行草稿模式下也只需一次（所有草稿 token 并行生成）。
        # 其他模式需要执行 num_speculative_tokens 次（自回归逐个生成草稿 token）。
        only_one_forward_pass = is_graph_capturing or self.parallel_drafting
        for fwd_idx in range(
            1 if only_one_forward_pass else self.num_speculative_tokens
        ):
            # 中文注释：第一次迭代时确定批次执行模式（CUDA Graph 模式和填充策略）。
            # 后续迭代复用相同的模式，避免重复调度。
            if fwd_idx <= 1:
                cudagraph_runtime_mode, num_input_tokens, num_tokens_across_dp = (
                    self._determine_batch_execution_and_padding(
                        num_tokens, use_cudagraphs=use_cudagraphs
                    )
                )

            # Make sure to use EAGLE's own buffer during cudagraph capture.
            # 中文注释：确保在 CUDA Graph 捕获时使用 EAGLE 的缓冲区
            if (
                self._draft_attn_layer_names
                and slot_mappings is not None
                and next(iter(self._draft_attn_layer_names)) in slot_mappings
            ):
                slot_mapping_dict = self._get_slot_mapping(num_input_tokens)
            else:
                slot_mapping_dict = slot_mappings or {}

            # 中文注释：设置前向传播上下文并执行虚拟前向传播
            with set_forward_context(
                None,
                self.vllm_config,
                num_tokens=num_input_tokens,
                num_tokens_across_dp=num_tokens_across_dp,
                cudagraph_runtime_mode=cudagraph_runtime_mode,
                slot_mapping=slot_mapping_dict,
            ):
                if self.supports_mm_inputs:
                    input_ids = None
                    inputs_embeds = self.inputs_embeds[:num_input_tokens]
                else:
                    input_ids = self.input_ids[:num_input_tokens]
                    inputs_embeds = None

                kwargs = dict(
                    input_ids=input_ids,
                    positions=self._get_positions(num_input_tokens),
                    inputs_embeds=inputs_embeds,
                )
                if self.pass_hidden_states_to_model:
                    kwargs["hidden_states"] = self.hidden_states[:num_input_tokens]
                self.model(**kwargs)

    def _get_eagle3_use_aux_hidden_state_from_config(self) -> bool:
        """
        Some eagle3 heads (e.g., nvidia/gpt-oss-120b-Eagle3-v2) do not use auxiliary
        hidden states and directly uses the last layer output just like eagle1.
        They might indicate this by setting "use_aux_hidden_state" to False
        inside the "eagle_config" dict of their hf_config.

        中文注释：从配置中获取是否使用 Eagle3 辅助隐藏状态。

        某些 Eagle3 头（如 nvidia/gpt-oss-120b-Eagle3-v2）不使用辅助隐藏状态，
        而是像 Eagle1 一样直接使用最后一层的输出。它们可能在 hf_config 的
        "eagle_config" 字典中设置 "use_aux_hidden_state" 为 False。

        返回:
            是否使用辅助隐藏状态
        """
        # 中文注释：只有 Eagle3 方法才支持辅助隐藏状态
        if self.method != "eagle3":
            return False
        # Assume that eagle3 heads use aux hidden states by default
        # 中文注释：默认假设 Eagle3 头使用辅助隐藏状态
        use_aux_hidden_state = True
        eagle_config = getattr(self.draft_model_config.hf_config, "eagle_config", None)
        if eagle_config is not None:
            use_aux_hidden_state = eagle_config.get("use_aux_hidden_state", True)
        return use_aux_hidden_state

    def validate_same_kv_cache_group(self, kv_cache_config: KVCacheConfig) -> None:
        """
        Validate that all drafting layers belong to the same KVCacheGroup.
        Need this assumption to ensure all drafting layers can use the
        same AttentionMetadata.
        May extend to multiple AttentionMetadata in the future.

        中文注释：验证所有草稿层是否属于同一个 KV cache 组。

        需要此假设以确保所有草稿层可以使用相同的 AttentionMetadata。
        未来可能会扩展到多个 AttentionMetadata。

        参数:
            kv_cache_config: KV cache 配置

        抛出:
            AssertionError: 如果草稿层不属于同一个 KV cache 组
        """
        # 中文注释：构建层名到 KV cache 组 ID 的映射
        kv_cache_groups: dict[str, int] = {}
        for id, kv_cache_group in enumerate(kv_cache_config.kv_cache_groups):
            for layer_name in kv_cache_group.layer_names:
                kv_cache_groups[layer_name] = id
        # 中文注释：验证所有草稿层属于同一个 KV cache 组
        assert (
            len(
                set(
                    [
                        kv_cache_groups[layer_name]
                        for layer_name in self._draft_attn_layer_names
                    ]
                )
            )
            == 1
        ), "All drafting layers should belong to the same kv cache group"

    def initialize_attn_backend(
        self,
        kv_cache_config: KVCacheConfig,
        kernel_block_sizes: list[int] | None = None,
    ) -> None:
        """
        Initialize AttentionGroups for draft layers using kv_cache_config.
        Called from the model runner's initialize_metadata_builders.

        中文注释：使用 kv_cache_config 初始化草稿层的注意力组。

        从模型运行器的 initialize_metadata_builders 调用。

        此方法：
        1. 验证所有草稿层属于同一个 KV cache 组
        2. 找到草稿层所属的 KV cache 组
        3. 为每个注意力后端创建注意力组
        4. 初始化元数据构建器

        参数:
            kv_cache_config: KV cache 配置
            kernel_block_sizes: 内核块大小列表
        """
        # 中文注释：获取所有注意力层
        all_attn_layers = get_layers_from_vllm_config(
            self.vllm_config,
            AttentionLayerBase,  # type: ignore[type-abstract]
        )

        # Find which kv_cache_group the draft layers belong to
        # 中文注释：验证并找到草稿层所属的 KV cache 组
        self.validate_same_kv_cache_group(kv_cache_config)
        kv_cache_spec = None
        for gid, group in enumerate(kv_cache_config.kv_cache_groups):
            if self._draft_attn_layer_names & set(group.layer_names):
                self.kv_cache_gid = gid
                kv_cache_spec = group.kv_cache_spec
                break

        # 中文注释：按注意力后端分组创建注意力组。
        # 不同的注意力层可能使用不同的后端（如 FlashAttention、FlashInfer 等），
        # 使用相同后端的层会被归入同一个 AttentionGroup。
        # 每个 AttentionGroup 内部维护一个 metadata_builder，
        # 用于根据 CommonAttentionMetadata 生成该后端特定的注意力元数据。
        attention_groups: dict[tuple[str, str], AttentionGroup] = {}
        if kv_cache_spec is not None:
            for layer_name in self._draft_attn_layer_names:
                attn_backend = all_attn_layers[layer_name].get_attn_backend()
                backend_key = attn_backend.full_cls_name()
                if backend_key not in attention_groups:
                    # 中文注释：获取该层的 KV cache 规格。
                    # UniformTypeKVCacheSpecs 表示不同层可能有不同的 KV cache 类型，
                    # 需要按层名查找对应的规格。
                    layer_kv_cache_spec = kv_cache_spec
                    if isinstance(layer_kv_cache_spec, UniformTypeKVCacheSpecs):
                        layer_kv_cache_spec = layer_kv_cache_spec.kv_cache_specs[
                            layer_name
                        ]

                    # 中文注释：获取内核块大小（如果有指定）。
                    # kernel_block_size 控制 Triton/CUDA 内核的线程块大小，
                    # 影响计算效率和显存使用。
                    kernel_block_size = (
                        kernel_block_sizes[self.kv_cache_gid]
                        if kernel_block_sizes is not None
                        and self.kv_cache_gid < len(kernel_block_sizes)
                        else None
                    )
                    # 中文注释：创建注意力组，将注意力后端、层名和 KV cache 规格关联起来
                    attn_group = AttentionGroup(
                        backend=attn_backend,
                        layer_names=[layer_name],
                        kv_cache_spec=layer_kv_cache_spec,
                        kv_cache_group_id=self.kv_cache_gid,
                    )
                    # 中文注释：为该注意力组创建元数据构建器。
                    # metadata_builder 负责在每次 forward 前根据调度结果生成注意力所需的元数据
                    # （如 block table、slot mapping、序列长度等）。
                    attn_group.create_metadata_builders(
                        self.vllm_config,
                        self.device,
                        kernel_block_size=kernel_block_size,
                    )
                    attention_groups[backend_key] = attn_group
                else:
                    # 中文注释：相同后端的层归入同一个注意力组
                    attention_groups[backend_key].layer_names.append(layer_name)

        # 中文注释：保存注意力组列表和 KV block 大小。
        # block_size 从第一个注意力组的元数据构建器中获取，
        # 用于后续的 slot mapping 计算和 KV cache 管理。
        self.draft_attn_groups = list(attention_groups.values())
        self.block_size = (
            self.draft_attn_groups[0].get_metadata_builder().kv_cache_spec.block_size
        )
        logger.debug("Using block size %d for drafting layers", self.block_size)

    def _determine_batch_execution_and_padding(
        self,
        num_tokens: int,
        use_cudagraphs: bool = True,
    ) -> tuple[CUDAGraphMode, int, torch.Tensor | None]:
        """
        确定批次执行模式和填充策略。

        此方法：
        1. 根据 token 数量和 CUDA Graph 配置，决定使用哪种执行模式
        2. 计算填充后的 token 数量
        3. 在数据并行情况下，协调各 rank 的批次大小

        参数:
            num_tokens: 原始 token 数量
            use_cudagraphs: 是否使用 CUDA Graph

        返回:
            元组 (cudagraph_mode, num_tokens_padded, num_tokens_across_dp)
            - cudagraph_mode: CUDA Graph 执行模式
            - num_tokens_padded: 填充后的 token 数量
            - num_tokens_across_dp: 数据并行各 rank 的 token 数量
        """
        # 中文注释：步骤 1 - 根据 token 数量调度 CUDA Graph 模式。
        # cudagraph_dispatcher 会根据 num_tokens 选择合适的 CUDA Graph 模式：
        # - CUDAGraphMode.NONE：不使用 CUDA Graph，直接 eager 执行
        # - CUDAGraphMode.PADDED：使用预捕获的 CUDA Graph，token 数量填充到固定大小
        # batch_desc 包含填充后的 token 数量和批次描述信息。
        cudagraph_mode, batch_desc = self.cudagraph_dispatcher.dispatch(
            num_tokens,
            valid_modes=({CUDAGraphMode.NONE} if not use_cudagraphs else None),
        )
        num_tokens_padded = batch_desc.num_tokens

        # Extra coordination when running data-parallel since we need to
        # coordinate across ranks
        # TODO(Flechman): support DBO ubatching
        # 中文注释：步骤 2 - 数据并行协调。
        # 当使用数据并行（DP > 1）时，所有 DP rank 必须使用相同的批次大小和 CUDA Graph 模式，
        # 否则会导致 NCCL 通信死锁。coordinate_batch_across_dp 会在所有 rank 之间同步，
        # 选择最大的 token 数量作为统一的批次大小。
        should_ubatch, num_tokens_across_dp = False, None
        if self.vllm_config.parallel_config.data_parallel_size > 1:
            should_ubatch, num_tokens_across_dp, synced_cudagraph_mode = (
                coordinate_batch_across_dp(
                    num_tokens_unpadded=num_tokens,
                    parallel_config=self.vllm_config.parallel_config,
                    allow_microbatching=False,
                    num_tokens_padded=num_tokens_padded,
                    cudagraph_mode=cudagraph_mode.value,
                )
            )
            assert not should_ubatch, "DBO ubatching not implemented for EAGLE"

            # Extract DP-synced values
            # 中文注释：步骤 3 - 提取数据并行同步后的值并重新调度。
            # 所有 DP rank 协商后确定了统一的 token 数量，
            # 需要重新调度以确保 CUDA Graph 模式与协商结果一致。
            if num_tokens_across_dp is not None:
                dp_rank = self.dp_rank
                num_tokens_padded = int(num_tokens_across_dp[dp_rank].item())
                # Re-dispatch with DP padding so we have the correct
                # batch_descriptor
                # 中文注释：使用 DP 协商后的 token 数量重新调度 CUDA Graph
                cudagraph_mode, batch_desc = self.cudagraph_dispatcher.dispatch(
                    num_tokens_padded,
                    valid_modes={CUDAGraphMode(synced_cudagraph_mode)},
                )
                # Assert to make sure the agreed upon token count is correct
                # otherwise num_tokens_across_dp will no-longer be valid
                # 中文注释：验证重新调度后的 token 数量与协商结果一致
                assert batch_desc.num_tokens == num_tokens_padded
                num_tokens_across_dp[dp_rank] = num_tokens_padded

        return cudagraph_mode, num_tokens_padded, num_tokens_across_dp


# NOTE(woosuk): Currently, the below code is not used and we always use argmax
# to sample the draft tokens. We will use this after we find a way to manage
# the draft prob tensor.
# Refer to https://github.com/vllm-project/vllm/pull/16899 for the details.
# FIXME(woosuk): The logic here is duplicated with the main sampling code.
# We should refactor this to reuse the same sampling implementation.
def compute_probs_and_sample_next_token(
    logits: torch.Tensor,
    sampling_metadata: SamplingMetadata,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    计算概率并采样下一个 token。

    此函数用于概率性草稿采样，根据温度参数采样草稿 token。

    参数:
        logits: 模型输出的 logits 张量
        sampling_metadata: 采样元数据

    返回:
        元组 (next_token_ids, probs)
        - next_token_ids: 采样的 token ID
        - probs: 采样概率
    """
    if sampling_metadata.all_greedy:
        # For greedy requests, draft_probs is not used in rejection sampling.
        # Therefore, we can just return the logits.
        # 中文注释：所有请求都是贪婪采样时，直接对 logits 取 argmax 即可。
        # 此时 rejection sampling 不需要概率信息，只需比较 draft 和 target 的 token 是否一致。
        probs = logits
        next_token_ids = logits.argmax(dim=-1)
        return next_token_ids, probs

    assert sampling_metadata.temperature is not None

    # Use epsilon comparison to detect greedy sampling (temperature ~ 0.0)
    # consistent with sampler.py's _SAMPLING_EPS threshold
    # 中文注释：应用温度缩放将 logits 转换为概率分布。
    # 温度越高分布越平坦（更随机），越低分布越尖锐（更确定）。
    # 对于混合批次中的贪婪请求（temperature ~ 0），用 1.0 替换避免除零，
    # 后续再用 argmax 覆盖其采样结果。
    temperature = sampling_metadata.temperature
    # Avoid division by zero if there are greedy requests.
    if not sampling_metadata.all_random:
        is_greedy = temperature < _SAMPLING_EPS
        temperature = torch.where(is_greedy, 1.0, temperature)
    logits.div_(temperature.view(-1, 1))
    probs = logits.softmax(dim=-1, dtype=torch.float32)

    # NOTE(woosuk): Currently, we ignore most of the sampling parameters in
    # generating the draft tokens. We only use the temperature. While this
    # could degrade the acceptance rate, it does not affect the distribution
    # of the generated tokens after rejection sampling.

    # TODO(woosuk): Consider seeds.
    # 中文注释：使用 Gumbel-max 技巧进行采样。
    # 这是一种无需 CPU-GPU 同步的高效采样方法：
    # 1. 从指数分布中采样 q ~ Exp(1)
    # 2. 计算 probs / q 并取 argmax
    # 数学上等价于从 categorical(probs) 中采样，但全部在 GPU 上完成。
    q = torch.empty_like(probs)
    q.exponential_()
    # NOTE(woosuk): We shouldn't use `probs.div_(q)` because the draft_probs
    # will be used later for rejection sampling.
    # 中文注释：不能用 in-place 除法，因为 probs 会被保存用于后续的 rejection sampling
    next_token_ids = probs.div(q).argmax(dim=-1).view(-1)
    # 中文注释：对于混合批次中的贪婪请求，用 argmax 覆盖其采样结果
    if not sampling_metadata.all_random:
        greedy_token_ids = probs.argmax(dim=-1)
        next_token_ids = torch.where(is_greedy, greedy_token_ids, next_token_ids)
    return next_token_ids, probs
