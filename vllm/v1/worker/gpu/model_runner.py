# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
NOTE: Coding style guide for this file:
This model runner is shared by all models: text and multimodal, generative
and embedding, public and private. As a result, this file must only contain
code that is common to every model. Model-specific behavior belongs in the
appropriate model-specific files.

In other words:
* Be paranoid about changing this file. It should remain stable.
* Be even more paranoid about adding new lines. It should remain minimal.

Even for shared features (for example, different parallelism modes), keep the
complexity out of this path. The less common the feature, the more it should be
hidden. Prefer utility functions defined elsewhere and call them from here,
instead of embedding feature-specific logic directly.
"""

# =============================================================================
# 模块概述 (Module Overview)
# =============================================================================
# 本文件是 vLLM v1 引擎中的 GPU 模型运行器 (GPU Model Runner)。
# 它是所有模型（文本模型、多模态模型、生成模型、嵌入模型）的共享执行核心。
#
# 核心职责包括：
# 1. 模型加载与初始化：加载预训练模型权重，初始化各种组件（采样器、KV 缓存等）
# 2. 前向推理执行 (execute_model)：管理完整的模型前向传播流程
# 3. 采样 (sample_tokens)：从模型输出中生成 token，支持标准采样和推测解码
# 4. CUDA 图管理：捕获和重放 CUDA 图以优化推理性能
# 5. KV 缓存管理：初始化和维护 KV 缓存，管理块表 (block tables)
# 6. 内存管理：配置 GPU 显存分配，处理请求状态
# 7. 并行策略支持：支持张量并行、流水线并行、数据并行
#
# 请求执行流程：
#   SchedulerOutput -> finish_requests -> add_requests -> update_requests
#   -> prepare_inputs -> prepare_attn -> execute_model -> sample_tokens
#   -> postprocess -> ModelRunnerOutput
# =============================================================================

import functools
import gc
import time
from copy import deepcopy
from typing import Any, NamedTuple

import numpy as np
import torch
import torch.nn as nn

from vllm.compilation.counter import compilation_counter
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed.parallel_state import (
    get_dcp_group,
    get_pp_group,
    prepare_communication_buffer_for_model,
)
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.logger import init_logger
from vllm.lora.layers import LoRAMapping
from vllm.model_executor.layers.mamba.ops.ssu_dispatch import (
    initialize_mamba_ssu_backend,
)
from vllm.model_executor.model_loader import get_model_loader
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors
from vllm.tasks import SupportedTask
from vllm.utils.math_utils import cdiv
from vllm.utils.mem_utils import DeviceMemoryProfiler, format_gib
from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig, MambaSpec
from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput
from vllm.v1.worker.cp_utils import check_attention_cp_compatibility
from vllm.v1.worker.gpu.async_utils import AsyncOutput, AsyncPoolingOutput
from vllm.v1.worker.gpu.attn_utils import (
    build_slot_mappings_by_layer,
    get_kv_cache_spec,
    init_attn_backend,
    init_kv_cache,
)
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.buffer_utils import async_copy_to_gpu
from vllm.v1.worker.gpu.cp_utils import prepare_dcp_local_seq_lens
from vllm.v1.worker.gpu.cudagraph_utils import (
    BatchExecutionDescriptor,
    ModelCudaGraphManager,
    get_uniform_token_count,
)
from vllm.v1.worker.gpu.dp_utils import dispatch_cg_and_sync_dp
from vllm.v1.worker.gpu.eplb_utils import EPLBController, step_eplb_after
from vllm.v1.worker.gpu.input_batch import (
    InputBatch,
    InputBuffers,
    combine_sampled_and_draft_tokens,
    expand_idx_mapping,
    get_num_sampled_and_rejected,
    post_update,
    post_update_pool,
    prepare_pos_seq_lens,
    prepare_prefill_inputs,
)
from vllm.v1.worker.gpu.kv_connector import (
    NO_OP_KV_CONNECTOR,
    KVConnector,
    get_kv_connector,
)
from vllm.v1.worker.gpu.lora_utils import LoraState
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.model_states import init_model_state
from vllm.v1.worker.gpu.pool.pooling_runner import PoolingRunner
from vllm.v1.worker.gpu.pp_utils import pp_broadcast, pp_receive
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.sample.prompt_logprob import PromptLogprobsWorker
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.shutdown import free_before_shutdown
from vllm.v1.worker.gpu.spec_decode import init_speculator
from vllm.v1.worker.gpu.spec_decode.eagle.eagle3_utils import (
    set_eagle3_aux_hidden_state_layers,
)
from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler
from vllm.v1.worker.gpu.spec_decode.utils import DraftTokensHandler
from vllm.v1.worker.gpu.states import RequestState
from vllm.v1.worker.gpu.structured_outputs import StructuredOutputsWorker
from vllm.v1.worker.lora_model_runner_mixin import LoRAModelRunnerMixin

logger = init_logger(__name__)


class GPUModelRunner(LoRAModelRunnerMixin):
    """
    GPU 模型运行器 (GPU Model Runner)

    这是 vLLM v1 引擎的核心执行组件，负责在 GPU 上运行模型的前向推理。
    它继承自 LoRAModelRunnerMixin 以支持 LoRA 适配器。

    主要职责：
    1. 管理模型的生命周期（加载、初始化、关闭）
    2. 处理调度器输出的请求批次，准备模型输入
    3. 执行模型前向传播（支持 CUDA 图、eager 模式等）
    4. 从隐藏状态中采样 token（支持标准采样和推测解码）
    5. 管理 KV 缓存、块表、注意力元数据
    6. 支持多种并行策略（PP、DP、DCP）
    7. 管理多模态输入的编码器缓存

    属性：
        vllm_config: 全局配置对象
        model: 加载的 PyTorch 模型
        sampler: 采样器，用于从 logits 中选择 token
        cudagraph_manager: CUDA 图管理器
        kv_caches: KV 缓存张量列表
        block_tables: 块表管理器，用于 KV 缓存的块级索引
        req_states: 请求状态管理器
        execute_model_state: execute_model 和 sample_tokens 之间传递的状态
    """

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        # -----------------------------------------------------------------
        # 初始化配置 (Configuration Initialization)
        # -----------------------------------------------------------------
        # 从全局配置对象中提取各子配置
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config      # 模型配置（架构、dtype 等）
        self.cache_config = vllm_config.cache_config       # 缓存配置（KV 缓存策略）
        self.compilation_config = vllm_config.compilation_config  # 编译配置（CUDA 图模式等）
        self.lora_config = vllm_config.lora_config         # LoRA 配置（低秩适配器）
        self.load_config = vllm_config.load_config         # 模型加载配置（量化、格式等）
        self.parallel_config = vllm_config.parallel_config # 并行配置（TP、PP、DP）
        self.scheduler_config = vllm_config.scheduler_config  # 调度器配置
        self.speculative_config = vllm_config.speculative_config  # 推测解码配置
        self.observability_config = vllm_config.observability_config  # 可观测性配置

        self.device = device  # GPU 设备
        self.dtype = self.model_config.dtype  # 模型计算精度（如 float16、bfloat16）

        # KV 缓存数据类型：默认与模型精度相同，支持量化 KV 缓存
        self.kv_cache_dtype = self.dtype
        if self.cache_config.cache_dtype != "auto":
            # Quantized KV cache.
            self.kv_cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[
                self.cache_config.cache_dtype
            ]

        # 模型基本参数
        self.vocab_size = self.model_config.get_vocab_size()  # 词表大小
        self.max_model_len = self.model_config.max_model_len  # 模型最大序列长度
        # 单批次最大 token 数
        self.max_num_tokens = self.scheduler_config.max_num_batched_tokens
        self.max_num_reqs = self.scheduler_config.max_num_seqs  # 单批次最大请求数
        self.is_encoder_decoder = self.model_config.is_encoder_decoder  # 是否为编码器-解码器模型

        # 异步调度：启用时 sample_tokens 可与下一批次的 prepare 重叠执行
        self.use_async_scheduling = self.scheduler_config.async_scheduling
        self.output_copy_stream = torch.cuda.Stream(self.device)  # 用于异步 D2H 拷贝的 CUDA 流

        # -----------------------------------------------------------------
        # 流水线并行 (Pipeline Parallelism)
        # -----------------------------------------------------------------
        self.use_pp = self.parallel_config.pipeline_parallel_size > 1
        self.is_first_pp_rank = get_pp_group().is_first_rank  # 是否为第一个 PP 阶段
        self.is_last_pp_rank = get_pp_group().is_last_rank    # 是否为最后一个 PP 阶段

        # 持久化缓冲区：非第一个 PP 阶段的中间张量
        # 用于在 PP 阶段之间传递隐藏状态
        self.intermediate_tensors: IntermediateTensors | None = None

        # -----------------------------------------------------------------
        # 数据并行 (Data Parallelism)
        # -----------------------------------------------------------------
        self.dp_size = self.parallel_config.data_parallel_size
        self.dp_rank = self.parallel_config.data_parallel_rank

        # -----------------------------------------------------------------
        # 解码上下文并行 (Decode Context Parallelism, DCP)
        # -----------------------------------------------------------------
        # DCP 将单个请求的 KV 缓存分片到不同 rank，实现长序列的分布式解码
        self.dcp_size = self.parallel_config.decode_context_parallel_size
        self.use_dcp = self.dcp_size > 1
        self.dcp_rank = get_dcp_group().rank_in_group if self.use_dcp else 0
        self.cp_interleave = self.parallel_config.cp_kv_cache_interleave_size

        # -----------------------------------------------------------------
        # 多模态支持 (Multimodal Support)
        # -----------------------------------------------------------------
        self.mm_registry = MULTIMODAL_REGISTRY
        self.supports_mm_inputs = self.mm_registry.supports_multimodal_inputs(
            self.model_config
        )
        # 编码器缓存：仅在第一个 PP 阶段和多模态模型时创建
        # 用于缓存视觉/音频编码器的输出，避免重复计算
        self.encoder_cache = None
        if self.supports_mm_inputs and self.is_first_pp_rank:
            self.encoder_cache = EncoderCache()

        # -----------------------------------------------------------------
        # 推测解码 (Speculative Decoding)
        # -----------------------------------------------------------------
        # 推测解码通过草稿模型提前预测多个 token，然后用目标模型验证
        # 可显著提升解码速度（2-3x）
        self.speculator = None
        self.num_speculative_steps = 0
        self.use_aux_hidden_state_outputs = False
        if self.speculative_config is not None:
            self.num_speculative_steps = self.speculative_config.num_speculative_tokens

            if self.is_last_pp_rank:
                self.speculator = init_speculator(self.vllm_config, self.device)

            if self.speculative_config.method == "eagle3":
                # EAGLE3 may require auxiliary hidden states from target model outputs.
                # EAGLE3 推测解码方法需要目标模型的辅助隐藏状态
                self.use_aux_hidden_state_outputs = True
                if self.use_pp:
                    raise ValueError("EAGLE3 with pipeline parallel is not supported.")

        # 草稿 token 传播器：用于推测解码 + 结构化输出的组合场景
        self.draft_tokens_handler = DraftTokensHandler(self.device)
        # 统一解码查询长度：1（实际 token）+ num_speculative_steps（草稿 token）
        self.uniform_decode_query_len = 1 + self.num_speculative_steps

        # -----------------------------------------------------------------
        # 池化模型 (Pooling Models)
        # -----------------------------------------------------------------
        # 池化模型（如嵌入模型）使用 PoolingRunner 而非采样器
        self.is_pooling_model = self.model_config.runner_type == "pooling"
        self.pooling_runner: PoolingRunner | None = None

        # -----------------------------------------------------------------
        # 请求状态和输入缓冲区 (Request States & Input Buffers)
        # -----------------------------------------------------------------
        # RequestState：管理所有活跃请求的内部状态
        #   - token IDs、序列长度、已计算 token 数等
        #   - 使用预分配的 numpy/pytorch 数组，避免运行时分配
        self.req_states = RequestState(
            max_num_reqs=self.max_num_reqs,
            max_model_len=self.max_model_len,
            max_num_batched_tokens=self.max_num_tokens,
            num_speculative_steps=self.num_speculative_steps,
            vocab_size=self.vocab_size,
            device=self.device,
        )
        # InputBuffers：预分配的 GPU 输入缓冲区
        #   - input_ids、positions、seq_lens 等
        #   - 避免每次推理时分配新的 GPU 张量
        self.input_buffers = InputBuffers(
            max_num_reqs=self.max_num_reqs,
            max_num_tokens=self.max_num_tokens,
            device=self.device,
        )

        # -----------------------------------------------------------------
        # 采样器和相关组件 (Sampler & Related Components)
        # -----------------------------------------------------------------
        # 仅在最后一个 PP 阶段且为生成模型时初始化采样器
        self.sampler: Sampler | None = None
        self.rejection_sampler: RejectionSampler | None = None
        self.prompt_logprobs_worker: PromptLogprobsWorker | None = None
        self.structured_outputs_worker: StructuredOutputsWorker | None = None
        if self.is_last_pp_rank and not self.is_pooling_model:
            # Initialize sampling-related workers.
            # These components are only set up on the last PP rank and
            # for generative (non-pooling) models.
            # 标准采样器：从 logits 中选择下一个 token
            self.sampler = Sampler(
                max_num_reqs=self.max_num_reqs,
                vocab_size=self.vocab_size,
                device=self.device,
                req_states=self.req_states,
                logprobs_mode=self.model_config.logprobs_mode,
                num_speculative_tokens=self.num_speculative_steps + 1,
                use_fp64_gumbel=self.model_config.use_fp64_gumbel,
            )
            if self.speculative_config is not None:
                # 拒绝采样器：用于推测解码，验证草稿 token 是否被接受
                self.rejection_sampler = RejectionSampler(
                    self.sampler,
                    self.speculative_config,
                    self.device,
                )
            # Prompt logprobs 工作器：计算 prompt 部分的对数概率
            self.prompt_logprobs_worker = PromptLogprobsWorker(self.max_num_reqs)
            # 结构化输出工作器：应用语法掩码限制采样空间
            self.structured_outputs_worker = StructuredOutputsWorker(
                max_num_logits=self.max_num_reqs * (self.num_speculative_steps + 1),
                vocab_size=self.vocab_size,
                device=self.device,
            )

        # -----------------------------------------------------------------
        # CUDA 图和 LoRA (CUDA Graphs & LoRA)
        # -----------------------------------------------------------------
        # 解码查询长度：每个请求在解码阶段的查询 token 数
        self.decode_query_len = self.num_speculative_steps + 1
        # CUDA 图管理器：在 initialize_kv_cache 中初始化
        self.cudagraph_manager: ModelCudaGraphManager | None = None
        # LoRA 状态管理器：跟踪每个请求使用的 LoRA 适配器
        self.lora_state = LoraState(max_num_reqs=self.max_num_reqs)
        # KV 连接器：用于跨节点的 KV 缓存传输（如预填充-解码分离架构）
        self.kv_connector: KVConnector = NO_OP_KV_CONNECTOR

        # execute_model 和 sample_tokens 之间传递的状态
        # 在 execute_model 中设置，在 sample_tokens 中消费
        self.execute_model_state: ExecuteModelState | None = None

        # 专家并行负载均衡器：管理 MoE 模型中专家的分配和迁移
        self.eplb = EPLBController(self.parallel_config, self.device)

    def update_max_model_len(self, max_model_len: int) -> None:
        """更新模型最大序列长度。当运行时动态调整时使用。"""
        self.max_model_len = max_model_len
        self.req_states.max_model_len = max_model_len

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        """获取当前模型支持的任务类型（如 generate、embed 等）。"""
        tasks: list[SupportedTask] = []
        if self.model_config.runner_type == "generate":
            tasks.extend(self.model_state.get_supported_generation_tasks())
        if self.is_pooling_model:
            # Do not rely on pooling_runner here, since this information is needed
            # on the first PP rank, while pooling_runner is only initialized
            # on the last PP rank.
            tasks.extend(PoolingRunner.get_supported_tasks(self.model))
        return tuple(tasks)

    def load_model(self, load_dummy_weights: bool = False, *args, **kwargs) -> None:
        """
        加载模型权重并初始化依赖模型的组件。

        执行流程：
        1. 使用模型加载器加载权重到 GPU
        2. 如果配置了 LoRA，叠加 LoRA 适配器
        3. 设置 EAGLE3 辅助隐藏状态层
        4. 加载推测解码的草稿模型
        5. 准备分布式通信缓冲区
        6. 初始化模型状态（input_processor 等）
        7. 为非第一个 PP 阶段创建中间张量缓冲区

        Args:
            load_dummy_weights: 是否加载虚拟权重（用于测试/调试）
        """
        time_before_load = time.perf_counter()
        if load_dummy_weights:
            self.load_config.load_format = "dummy"
        self.eplb.prepare_load()
        eplb_models_added = False
        with DeviceMemoryProfiler() as m:
            model_loader = get_model_loader(self.vllm_config.load_config)
            logger.info("Loading model from scratch...")

            self.model = model_loader.load_model(
                vllm_config=self.vllm_config, model_config=self.vllm_config.model_config
            )
            if self.lora_config:
                self.model = self.load_lora_model(
                    self.model, self.vllm_config, self.device
                )

            if self.use_aux_hidden_state_outputs:
                assert self.speculative_config is not None
                set_eagle3_aux_hidden_state_layers(self.model, self.speculative_config)
            if self.speculator is not None:
                self.speculator.load_model(self.model)
                eplb_models_added = self.eplb.maybe_register_speculator(
                    self.speculator, self.speculative_config, load_dummy_weights
                )
        time_after_load = time.perf_counter()

        self.model_memory_usage = m.consumed_memory
        logger.info(
            "Model loading took %s GiB and %.6f seconds",
            format_gib(m.consumed_memory),
            time_after_load - time_before_load,
        )

        if not load_dummy_weights:
            # 准备分布式通信缓冲区（如张量并行的 all-reduce 缓冲区）
            prepare_communication_buffer_for_model(self.model)
            if self.speculator is not None:
                prepare_communication_buffer_for_model(self.speculator.model)

        # Initialize the components that require the model.
        # 初始化依赖模型的组件（input_processor、encoder_runner 等）
        self.model_state = init_model_state(
            self.vllm_config, self.model, self.encoder_cache, self.device
        )
        if self.is_pooling_model and self.is_last_pp_rank:
            self.pooling_runner = PoolingRunner(self.model)
        eplb_models_added |= self.eplb.maybe_register_model(
            self.model,
            self.model_config,
            load_dummy_weights,
        )
        self.eplb.maybe_start_async_loop(eplb_models_added)

        if not self.is_first_pp_rank:
            # For non-first PP ranks, create intermediate tensors sized
            # for the max capture size so they can be sliced per batch.
            # Save as persistent member so runtime can copy received data
            # into the same addresses that the CUDA graphs captured.
            # 为非第一个 PP 阶段创建中间张量缓冲区
            # 大小为 max_num_tokens，以便在运行时按批次切片使用
            # 保存为持久化成员，确保运行时拷贝的数据地址与 CUDA 图捕获时一致
            self.intermediate_tensors = self.model.make_empty_intermediate_tensors(
                batch_size=self.max_num_tokens,
                dtype=self.model_config.dtype,
                device=self.device,
            )

    def get_model(self) -> nn.Module:
        """返回底层的 PyTorch 模型。"""
        return self.model

    def reload_weights(self, *args, **kwargs) -> None:
        """重新加载模型权重（不重新创建模型结构）。"""
        # TODO(Wentao): Use full version instead of import when fully migrated to v2
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner as GPUModelRunnerV1

        GPUModelRunnerV1.reload_weights(self, *args, **kwargs)  # type: ignore[arg-type]
        self.reset_encoder_cache()
        self.reset_mm_cache()

    def update_config(self, *args, **kwargs) -> None:
        """更新模型运行器的配置（如 LoRA 切换时）。"""
        # TODO(Wentao): Use full version instead of import when fully migrated to v2
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner as GPUModelRunnerV1

        GPUModelRunnerV1.update_config(self, *args, **kwargs)  # type: ignore[arg-type]

        # v2 reads config via self.vllm_config (e.g. in load_model), so keep it
        # in sync with the attributes the v1 helper just replaced.
        self.vllm_config.model_config = self.model_config
        self.vllm_config.load_config = self.load_config

    @functools.cached_property
    def main_stream(self) -> torch.cuda.Stream:
        # Cache the default CUDA stream to avoid lookup overhead.
        # 缓存默认 CUDA 流以避免重复查找开销
        return torch.cuda.current_stream(self.device)

    def get_kv_cache_spec(self):
        """获取 KV 缓存规范（每层的形状、dtype 等）。"""
        return get_kv_cache_spec(self.vllm_config)

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        """
        初始化 KV 缓存系统。

        这是模型运行器初始化的关键步骤，包括：
        1. 计算每个 KV 缓存组所需的最大块数
        2. 初始化注意力后端（FlashAttention、FlashInfer 等）
        3. 创建块表管理器 (BlockTables)
        4. 解析并初始化 CUDA 图模式
        5. 实际分配 KV 缓存张量

        Args:
            kv_cache_config: KV 缓存配置，描述缓存的分组和规格
        """
        kv_cache_config = deepcopy(kv_cache_config)
        self.kv_cache_config = kv_cache_config

        # 块表的最大模型长度：编码器-解码器模型可能需要更长的长度
        block_table_max_model_len = self.max_model_len
        if self.is_encoder_decoder:
            # Cross-attention block tables need to index encoder tokens
            # (e.g., Whisper ~1500), which can exceed decoder max_model_len.
            # 交叉注意力的块表需要索引编码器 token（如 Whisper ~1500），
            # 这可能超过解码器的 max_model_len。
            block_table_max_model_len = max(
                block_table_max_model_len,
                getattr(self.model_config.hf_config, "max_source_positions", 0),
            )

        # 计算每个 KV 缓存组的块大小和最大块数
        block_sizes = []
        max_num_blocks_per_group = []
        for kv_cache_group in kv_cache_config.kv_cache_groups:
            spec = kv_cache_group.kv_cache_spec
            block_sizes.append(spec.block_size)
            # When using DCP, each request's KV cache is sharded among different ranks.
            # As a result, one block on the current rank covers `block_size * cp_size`
            # tokens in the full, global (unsharded) sequence.
            max_num_blocks = cdiv(
                block_table_max_model_len, spec.block_size * self.dcp_size
            )
            # Align to a multiple of (128 / block_size) as required by some attention
            # backends such as TRTLLM (#39324)
            if spec.block_size <= 128:
                alignment = 128 // spec.block_size
                max_num_blocks = cdiv(max_num_blocks, alignment) * alignment
            # For Mamba/Hybrid Model, KVCaches need extra blocks for speculative tokens
            if isinstance(spec, MambaSpec):
                max_num_blocks = (
                    max_num_blocks if self.cache_config.enable_prefix_caching else 1
                ) + spec.num_speculative_blocks
            max_num_blocks_per_group.append(max_num_blocks)

        # 初始化注意力后端（FlashAttention、FlashInfer 等）
        # attn_groups: 注意力层的分组配置
        # attn_cg_support: 注意力后端对 CUDA 图的支持级别
        # kernel_block_sizes: 注意力内核使用的块大小
        self.attn_groups, attn_cg_support, kernel_block_sizes = init_attn_backend(
            self.kv_cache_config, self.vllm_config, self.device
        )
        # 创建块表管理器：管理 KV 缓存的块级索引
        self.block_tables = BlockTables(
            block_sizes=block_sizes,
            max_num_reqs=self.max_num_reqs,
            max_num_batched_tokens=self.max_num_tokens,
            max_num_blocks_per_group=max_num_blocks_per_group,
            device=self.device,
            kernel_block_sizes=kernel_block_sizes,
            cp_size=self.dcp_size,
            cp_rank=self.dcp_rank,
            cp_interleave=self.cp_interleave,
        )
        initialize_mamba_ssu_backend(
            self.vllm_config.mamba_config, self.kv_cache_config
        )
        # 解析 CUDA 图模式和批次大小
        # CUDA 图模式包括：NONE（禁用）、PIECEWISE（分段图）、FULL（完整图）
        cudagraph_mode = self.compilation_config.resolve_cudagraph_mode_and_sizes(
            attn_cg_support.min_cg_support,
            attn_cg_support.min_cg_attn_backend,
            self.uniform_decode_query_len,
            self.parallel_config.tensor_parallel_size,
            self.kv_cache_config,
            self.max_num_reqs,
        )
        # 初始化 CUDA 图管理器：负责捕获和重放 CUDA 图
        self.cudagraph_manager = ModelCudaGraphManager(
            self.vllm_config,
            self.device,
            cudagraph_mode,
            decode_query_len=self.decode_query_len,
        )
        if self.speculator is not None:
            self.speculator.init_cudagraph_manager(cudagraph_mode)

        check_attention_cp_compatibility(self.vllm_config)
        if self.speculator is not None:
            # HACK(woosuk)
            self.speculator.set_attn(
                self.model_state, self.kv_cache_config, self.block_tables
            )

        # 实际分配 KV 缓存张量
        # kv_caches: 扁平化的 KV 缓存张量列表
        # kv_caches_dict: 按层名索引的 KV 缓存字典（用于 KV 连接器）
        self.kv_caches: list[torch.Tensor] = []
        kv_caches_dict = init_kv_cache(
            self.kv_caches,
            self.compilation_config.static_forward_context,
            self.kv_cache_config,
            self.attn_groups,
            self.device,
            self.cache_config.cache_dtype,
            kernel_block_sizes,
            self.vllm_config,
        )
        # 初始化 KV 连接器：用于跨节点的 KV 缓存传输
        self.kv_connector = get_kv_connector(self.vllm_config, kv_caches_dict)

    @torch.inference_mode()
    @step_eplb_after(is_dummy=True)
    def _dummy_run(
        self,
        num_tokens: int,
        *args,
        skip_attn: bool = False,
        uniform_decode: bool = False,
        skip_eplb: bool = False,
        is_profile: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """
        执行虚拟前向传播 (Dummy Run)。

        用于以下场景：
        1. 内存分析 (Memory Profiling)：估算模型运行所需的 GPU 显存
        2. DP 同步：确保所有数据并行 rank 执行相同数量的虚拟步骤
        3. CUDA 图预热：在实际推理前预热模型执行路径

        执行流程：
        1. 创建虚拟的 SchedulerOutput（包含 num_tokens 个虚拟 token）
        2. 禁用 KV 连接器（虚拟运行不需要真实的 KV 传输）
        3. 调用 execute_model 执行虚拟前向传播
        4. 如果存在推测解码器，也执行虚拟的 propose 步骤以确保 DP/EP 同步

        Args:
            num_tokens: 虚拟运行的 token 数量
            skip_attn: 是否跳过注意力计算（仅在初始内存分析时使用）
            uniform_decode: 是否使用统一解码模式（每个请求相同数量的 token）
            is_profile: 是否为内存分析运行

        Returns:
            (hidden_states, sample_hidden_states) 的元组，非最后一个 PP 阶段返回 (None, None)
        """
        if skip_attn and not is_profile:
            raise ValueError(
                "skip_attn must only be True for initial memory profiling."
            )

        # Create a dummy scheduler output.
        num_reqs = min(num_tokens, self.max_num_reqs)
        if uniform_decode:
            # HACK(lucas): for now since the worker is shared between MRV1 and MRV2,
            # and for spec-decode with MTP we want to make sure the dummy runs use
            # 1+num_speculative_tokens we use max here, this will likely be eventually
            # changed in the worker: https://github.com/vllm-project/vllm/pull/35243
            num_tokens = max(num_tokens, self.decode_query_len)
            num_reqs = num_tokens // self.decode_query_len
            assert num_tokens % self.decode_query_len == 0
        num_tokens_per_request = [num_tokens // num_reqs] * num_reqs
        num_tokens_per_request[-1] += num_tokens % num_reqs

        assert sum(num_tokens_per_request) == num_tokens
        num_scheduled_tokens = {
            f"_dummy_req_{i}": n for i, n in enumerate(num_tokens_per_request)
        }
        dummy_scheduler_output = SchedulerOutput.make_empty()
        dummy_scheduler_output.total_num_scheduled_tokens = num_tokens
        dummy_scheduler_output.num_scheduled_tokens = num_scheduled_tokens

        # Disable any use of KVConnector for dummy runs.
        self.kv_connector.set_disabled(True)

        # Get the intermediate tensors for the dummy run.
        intermediate_tensors = None
        if not self.is_first_pp_rank:
            assert self.intermediate_tensors is not None
            intermediate_tensors = self.intermediate_tensors[:num_tokens]

        # Execute the model.
        self.execute_model(
            dummy_scheduler_output,
            intermediate_tensors=intermediate_tensors,
            dummy_run=True,
            skip_attn_for_dummy_run=skip_attn,
            is_profile=is_profile,
        )
        self.kv_connector.set_disabled(False)

        # Non-last PP ranks don't produce output for sampling.
        if not self.is_last_pp_rank:
            return None, None

        assert self.execute_model_state is not None
        input_batch = self.execute_model_state.input_batch
        attn_metadata = self.execute_model_state.attn_metadata
        slot_mappings_by_layer = self.execute_model_state.slot_mappings_by_layer
        hidden_states = self.execute_model_state.hidden_states
        aux_hidden_states = self.execute_model_state.aux_hidden_states
        self.execute_model_state = None

        # dummy run the eagle speculator's propose to ensure DP/EP sync.
        if self.speculator is not None:
            assert self.sampler is not None
            mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None
            if self.speculator.supports_mm_inputs:
                mm_inputs = (
                    [],
                    torch.zeros(
                        input_batch.num_tokens,
                        dtype=torch.bool,
                        device=self.device,
                    ),
                )

            # Let the target override the hidden state fed to the drafter
            # (e.g. DeepSeek V4 MTP needs the pre-hc_head residual). The
            # target returns a persistent buffer sized at max_num_batched_tokens;
            # slice to the active token count that propose() expects.
            spec_hidden_states = hidden_states
            if hasattr(self.model, "get_mtp_target_hidden_states"):
                pre_hc_hidden_states = self.model.get_mtp_target_hidden_states()
                spec_hidden_states = pre_hc_hidden_states[: hidden_states.shape[0]]  # type: ignore[union-attr]
            self.speculator.propose(
                input_batch=input_batch,
                attn_metadata=attn_metadata,
                slot_mappings=slot_mappings_by_layer,
                last_hidden_states=spec_hidden_states,
                aux_hidden_states=aux_hidden_states,
                num_sampled=torch.ones(
                    input_batch.num_reqs, dtype=torch.int32, device=self.device
                ),
                num_rejected=torch.zeros(
                    input_batch.num_reqs, dtype=torch.int32, device=self.device
                ),
                last_sampled=self.req_states.last_sampled_tokens,
                next_prefill_tokens=self.req_states.next_prefill_tokens,
                temperature=self.sampler.sampling_states.temperature.gpu,
                seeds=self.sampler.sampling_states.seeds.gpu,
                dummy_run=True,
                skip_attn_for_dummy_run=skip_attn,
                mm_inputs=mm_inputs,
                is_profile=is_profile,
            )

        assert hidden_states is not None  # Last PP rank always has hidden_states
        sample_hidden_states = hidden_states[input_batch.logits_indices]
        return hidden_states, sample_hidden_states

    @torch.inference_mode()
    def _dummy_sampler_run(self, hidden_states: torch.Tensor) -> None:
        """执行虚拟采样运行，用于内存分析时估算采样器的显存占用。"""
        num_reqs = hidden_states.shape[0]
        logits = self.model.compute_logits(hidden_states)
        dummy_input_batch = InputBatch.make_dummy(
            num_reqs, num_reqs, self.input_buffers
        )

        # NOTE(woosuk): During the initial memory profiling, the sampler may skip
        # top_k, top_p, and logprobs, using less GPU memory than what is possible
        # during actual execution.
        # 注意：在初始内存分析期间，采样器可能跳过 top_k、top_p 和 logprobs 计算，
        # 因此实际执行时可能使用更多 GPU 内存。
        assert self.sampler is not None
        self.sampler(logits, dummy_input_batch)

    @torch.inference_mode()
    def _dummy_pooler_run(self, hidden_states: torch.Tensor) -> None:
        """执行虚拟池化运行，用于内存分析时估算池化器的显存占用。"""
        assert self.pooling_runner is not None
        self.pooling_runner.dummy_pooler_run(hidden_states)

    @torch.inference_mode()
    def profile_run(self) -> None:
        """
        执行内存分析运行 (Profile Run)。

        通过执行虚拟前向传播来确定模型运行所需的 GPU 显存量。
        这个信息用于：
        1. 计算 KV 缓存可以使用的显存大小
        2. 决定可以同时处理多少个请求
        3. 验证显存分配是否合理

        流程：
        1. 执行 _dummy_run（跳过注意力计算以节省时间）
        2. 在最后一个 PP 阶段执行虚拟采样/池化
        3. 同步 GPU 并清理临时内存
        """
        hidden_states, sample_hidden_states = self._dummy_run(
            self.max_num_tokens, skip_attn=True, is_profile=True
        )

        # Only run sampler/pooler on last PP rank (non-last ranks return None).
        if self.is_last_pp_rank:
            assert sample_hidden_states is not None
            if self.pooling_runner is None:
                self._dummy_sampler_run(sample_hidden_states)
            else:
                self._dummy_pooler_run(hidden_states)

        torch.accelerator.synchronize()
        del hidden_states, sample_hidden_states
        gc.collect()

    def post_kv_cache_wake_up(self) -> None:
        """KV 缓存唤醒后初始化块表布局张量。"""
        self.block_tables.init_block_table_layout_tensors()

    def reset_mm_cache(self) -> None:
        """重置多模态缓存（视觉特征等）。"""
        if self.encoder_cache is not None:
            self.encoder_cache.reset_mm_cache()

    def reset_encoder_cache(self) -> None:
        """重置编码器缓存。"""
        if self.encoder_cache is not None:
            self.encoder_cache.reset_encoder_cache()

    def _get_num_input_tokens(self, num_scheduled_tokens: int) -> int:
        # SP is not supported yet.
        # 序列并行 (Sequence Parallelism) 尚未支持，直接返回原始 token 数
        return num_scheduled_tokens

    def profile_cudagraph_memory(self) -> int:
        # NOTE(woosuk): It is TBD whether we keep this API or not.
        return 0

    @torch.inference_mode()
    def capture_model(self) -> int:
        """
        捕获 CUDA 图 (CUDA Graph Capture)。

        CUDA 图通过将整个前向传播的 GPU 操作录制到一个图中，然后重放该图来避免
        每次推理时的内核启动开销。这对于小批次的解码阶段特别有效。

        工作原理：
        1. 清理 GPU 内存以获得准确的显存使用测量
        2. 使用虚拟 LoRA 设置（如果启用 LoRA）
        3. 调用 cudagraph_manager.capture() 录制 CUDA 图
           - 为不同批次大小录制多个图
           - 捕获所有 CUDA 操作（内核启动、内存拷贝等）
        4. 如果存在推测解码器，也为草稿模型捕获 CUDA 图
        5. 记录捕获时间和显存占用

        Returns:
            CUDA 图占用的显存大小（字节）
        """
        assert self.cudagraph_manager is not None
        if not self.cudagraph_manager.needs_capture():
            logger.warning(
                "Skipping CUDA graph capture. To turn on CUDA graph capture, "
                "ensure `cudagraph_mode` was not manually set to `NONE`"
            )
            return 0

        compilation_counter.num_gpu_runner_capture_triggers += 1

        start_time = time.perf_counter()
        gc.collect()
        torch.accelerator.empty_cache()
        start_free_gpu_memory = torch.cuda.mem_get_info()[0]

        with self.maybe_setup_dummy_loras(self.lora_config):
            captured_attn_states = self.cudagraph_manager.capture(
                self.model,
                self.model_state,
                self.input_buffers,
                self.intermediate_tensors,
                self.block_tables,
                self.attn_groups,
                self.kv_cache_config,
                has_lora=self.lora_config is not None,
                use_aux_hidden_state_outputs=self.use_aux_hidden_state_outputs,
            )
            if self.speculator is not None:
                self.speculator.capture(captured_attn_states)

        end_time = time.perf_counter()
        end_free_gpu_memory = torch.cuda.mem_get_info()[0]
        elapsed_time = end_time - start_time
        cuda_graph_size = start_free_gpu_memory - end_free_gpu_memory
        # This usually takes 5~20 seconds.
        logger.info(
            "Graph capturing finished in %.0f secs, took %.2f GiB",
            elapsed_time,
            cuda_graph_size / (1 << 30),
        )
        return cuda_graph_size

    def _remove_request(self, req_id: str) -> bool:
        """从所有组件中移除请求，清理相关状态。"""
        if not self.req_states.remove_request(req_id):
            return False
        if self.encoder_cache is not None:
            self.encoder_cache.remove_request(req_id)
        if self.prompt_logprobs_worker is not None:
            self.prompt_logprobs_worker.remove_request(req_id)
        self.lora_state.remove_request(req_id)
        return True

    def finish_requests(self, scheduler_output: SchedulerOutput) -> None:
        """
        处理已完成和被抢占的请求。

        将 finished_req_ids 和 preempted_req_ids 中的请求从运行器中移除，
        释放相关资源（编码器缓存、LoRA 映射等）。
        """
        finished_req_ids = scheduler_output.finished_req_ids
        preempted_req_ids = scheduler_output.preempted_req_ids
        if preempted_req_ids:
            finished_req_ids = finished_req_ids.union(preempted_req_ids)
        for req_id in finished_req_ids:
            self._remove_request(req_id)

    def free_states(self, scheduler_output: SchedulerOutput) -> None:
        """释放调度器标记的编码器缓存。"""
        if self.encoder_cache is not None:
            for mm_hash in scheduler_output.free_encoder_mm_hashes:
                self.encoder_cache.free_encoder_cache(mm_hash)

    def add_requests(self, scheduler_output: SchedulerOutput) -> None:
        """
        添加新请求到运行器。

        对于每个新请求：
        1. 在 RequestState 中注册请求
        2. 添加到编码器缓存（多模态）
        3. 设置块表（KV 缓存块映射）
        4. 配置 LoRA 适配器
        5. 在采样器中注册（最后一个 PP 阶段）
        6. 应用暂存的写入操作
        """
        for new_req_data in scheduler_output.scheduled_new_reqs:
            assert new_req_data.prompt_token_ids is not None
            assert new_req_data.prefill_token_ids is not None
            req_id = new_req_data.req_id

            # Streaming input update: request already exists from a prior
            # chunk. Remove old state so it can be cleanly re-added below
            # with the updated prompt_token_ids and mm_features.
            self._remove_request(req_id)

            prompt_len = len(new_req_data.prompt_token_ids)
            self.req_states.add_request(
                req_id=req_id,
                prompt_len=prompt_len,
                all_token_ids=new_req_data.prefill_token_ids,
                num_computed_tokens=new_req_data.num_computed_tokens,
            )
            req_index = self.req_states.req_id_to_index[req_id]

            if self.encoder_cache is not None:
                self.encoder_cache.add_request(req_id, new_req_data.mm_features)

            self.model_state.add_request(req_index, new_req_data)
            self.block_tables.append_block_ids(
                req_index, new_req_data.block_ids, overwrite=True
            )
            self.lora_state.add_request(req_id, req_index, new_req_data.lora_request)

            if self.is_last_pp_rank and new_req_data.sampling_params is not None:
                assert self.sampler is not None
                self.sampler.add_request(
                    req_index, prompt_len, new_req_data.sampling_params
                )
                assert self.prompt_logprobs_worker is not None
                self.prompt_logprobs_worker.add_request(
                    req_id, req_index, new_req_data.sampling_params
                )

        if scheduler_output.scheduled_new_reqs:
            self.req_states.apply_staged_writes()
            self.model_state.apply_staged_writes()
        if self.sampler is not None:
            self.sampler.apply_staged_writes()

    def update_requests(self, scheduler_output: SchedulerOutput) -> None:
        """
        更新已存在的请求状态。

        对于缓存的请求（已经在运行中的请求）：
        1. 更新已计算的 token 数量
        2. 添加新分配的 KV 缓存块到块表
        3. 更新预填充已计算 token 数（用于确定预填充进度）
        """
        # Add new blocks and update num_computed_tokens for the existing requests.
        reqs = scheduler_output.scheduled_cached_reqs
        num_computed_tokens_np = self.req_states.num_computed_tokens_np
        for req_id, num_computed_tokens, req_new_block_ids in zip(
            reqs.req_ids, reqs.num_computed_tokens, reqs.new_block_ids
        ):
            req_index = self.req_states.req_id_to_index[req_id]
            num_computed_tokens_np[req_index] = num_computed_tokens
            if req_new_block_ids is not None:
                self.block_tables.append_block_ids(
                    req_index, req_new_block_ids, overwrite=False
                )

        # Update num_computed_prefill_tokens.
        # 取 min(num_computed_tokens, prefill_len)，确保不超过预填充长度
        np.minimum(
            self.req_states.num_computed_tokens_np,
            self.req_states.prefill_len.np,
            out=self.req_states.num_computed_prefill_tokens,
        )

    def prepare_inputs(
        self, scheduler_output: SchedulerOutput, batch_desc: BatchExecutionDescriptor
    ) -> InputBatch:
        """
        准备模型前向传播的输入数据。

        这是模型执行前的核心准备工作，将调度器输出转换为模型可接受的输入格式。

        执行流程：
        1. 排序请求：按 token 数升序排列（decode 在前，prefill 在后）
        2. 构建索引映射：将请求 ID 映射到内部状态数组的索引
        3. 处理推测解码的草稿 token（如果存在）
        4. 计算 query_start_loc：每个请求在批次中的起始位置
        5. 准备 prefill token（从 all_token_ids 中读取）
        6. 计算位置 (positions) 和序列长度 (seq_lens)
        7. 准备 DCP 本地序列长度（如果启用 DCP）
        8. 合并采样 token 和草稿 token，获取 logits 索引

        Args:
            scheduler_output: 调度器输出，包含请求分配信息
            batch_desc: 批次执行描述符（CUDA 图模式、padding 信息等）

        Returns:
            InputBatch 对象，包含所有模型输入数据
        """
        num_tokens = scheduler_output.total_num_scheduled_tokens
        num_tokens_after_padding = batch_desc.num_tokens
        assert num_tokens > 0
        num_tokens_per_req = scheduler_output.num_scheduled_tokens
        num_reqs = len(num_tokens_per_req)

        # Decode first, then prefill.
        # batch_idx -> req_id
        req_ids = sorted(num_tokens_per_req, key=num_tokens_per_req.get)  # type: ignore[arg-type]
        numtoks_iter = map(num_tokens_per_req.get, req_ids)
        num_scheduled_tokens = np.fromiter(numtoks_iter, dtype=np.int32, count=num_reqs)

        idx_mapping_iter = map(self.req_states.req_id_to_index.get, req_ids)
        idx_mapping_np = np.fromiter(idx_mapping_iter, dtype=np.int32, count=num_reqs)
        idx_mapping = async_copy_to_gpu(idx_mapping_np, device=self.device)

        # Get the number of draft tokens for each request.
        draft_tokens = scheduler_output.scheduled_spec_decode_tokens
        num_draft_tokens_per_req = None
        if not draft_tokens:
            # No draft token scheduled (common case).
            total_num_draft_tokens = 0
            total_num_logits = num_reqs
            cu_num_logits_np = np.arange(num_reqs + 1, dtype=np.int32)
            cu_num_logits = torch.arange(
                num_reqs + 1, device=self.device, dtype=torch.int32
            )
            expanded_idx_mapping = idx_mapping
            expanded_local_pos = torch.zeros(
                num_reqs, dtype=torch.int32, device=self.device
            )
        else:
            num_draft_tokens_per_req = np.fromiter(
                (len(draft_tokens.get(req_id, ())) for req_id in req_ids),
                dtype=np.int32,
                count=num_reqs,
            )
            total_num_draft_tokens = int(num_draft_tokens_per_req.sum())
            total_num_logits = num_reqs + total_num_draft_tokens

            num_logits = num_draft_tokens_per_req + 1
            cu_num_logits_np = np.empty(num_reqs + 1, dtype=np.int32)
            cu_num_logits_np[0] = 0
            np.cumsum(num_logits, out=cu_num_logits_np[1:])
            cu_num_logits = async_copy_to_gpu(cu_num_logits_np, device=self.device)

            max_expand_len = self.num_speculative_steps + 1
            expanded_idx_mapping, expanded_local_pos = expand_idx_mapping(
                idx_mapping, total_num_logits, cu_num_logits, max_expand_len
            )

        # Get query_start_loc.
        # num_reqs_padded is None for PIECEWISE graphs (no request padding needed)
        num_reqs_padded = batch_desc.num_reqs or num_reqs
        query_start_loc_np = np.empty(self.max_num_reqs + 1, dtype=np.int32)
        query_start_loc_np[0] = 0
        np.cumsum(num_scheduled_tokens, out=query_start_loc_np[1 : num_reqs + 1])
        # Pad for full CUDA graph mode.
        # Some attention backends like FA3 require query_start_loc to be non-decreasing.
        query_start_loc_np[num_reqs + 1 :] = num_tokens
        async_copy_to_gpu(query_start_loc_np, out=self.input_buffers.query_start_loc)
        query_start_loc_np = query_start_loc_np[: num_reqs_padded + 1]
        query_start_loc = self.input_buffers.query_start_loc[: num_reqs_padded + 1]
        is_prefilling_np = self.req_states.is_prefilling(idx_mapping_np)

        # Get prefill tokens if any.
        if np.any(is_prefilling_np):
            prepare_prefill_inputs(
                self.input_buffers.input_ids,
                self.req_states.next_prefill_tokens,
                idx_mapping,
                query_start_loc,
                self.req_states.all_token_ids.gpu,
                self.req_states.prefill_len.gpu,
                self.req_states.num_computed_tokens.gpu,
            )

        # Prepare positions and seq_lens.
        prepare_pos_seq_lens(
            idx_mapping,
            query_start_loc,
            self.req_states.num_computed_tokens.gpu,
            self.input_buffers.positions,
            self.input_buffers.seq_lens,
        )
        seq_lens = self.input_buffers.seq_lens[:num_reqs_padded]

        dcp_local_seq_lens = None
        if self.use_dcp:
            # Prepare dcp local seq_lens.
            prepare_dcp_local_seq_lens(
                self.input_buffers.dcp_local_seq_lens,
                self.input_buffers.seq_lens,
                num_reqs,
                self.dcp_size,
                self.dcp_rank,
                self.cp_interleave,
            )
            dcp_local_seq_lens = self.input_buffers.dcp_local_seq_lens[:num_reqs_padded]

        # Some input token ids are directly read from the last sampled tokens
        # and draft tokens. Also, get the logits indices to sample tokens from.
        logits_indices = combine_sampled_and_draft_tokens(
            self.input_buffers.input_ids,
            idx_mapping,
            self.req_states.last_sampled_tokens,
            query_start_loc,
            seq_lens,
            self.req_states.prefill_len.gpu,
            self.req_states.draft_tokens,
            cu_num_logits,
            total_num_logits,
        )

        # CPU upper bound on seq_lens; padded entries left at zero.
        seq_lens_cpu_upper_bound_np = np.zeros(num_reqs_padded, dtype=np.int32)
        np.add(
            self.req_states.num_computed_tokens_np[idx_mapping_np],
            num_scheduled_tokens,
            out=seq_lens_cpu_upper_bound_np[:num_reqs],
        )
        seq_lens_cpu_upper_bound = torch.from_numpy(seq_lens_cpu_upper_bound_np)
        return InputBatch(
            req_ids=req_ids,
            num_reqs=num_reqs,
            num_reqs_after_padding=num_reqs_padded,
            idx_mapping=idx_mapping,
            idx_mapping_np=idx_mapping_np,
            expanded_idx_mapping=expanded_idx_mapping,
            expanded_local_pos=expanded_local_pos,
            num_scheduled_tokens=num_scheduled_tokens,
            num_tokens=num_tokens,
            num_tokens_after_padding=num_tokens_after_padding,
            num_draft_tokens=total_num_draft_tokens,
            num_draft_tokens_per_req=num_draft_tokens_per_req,
            query_start_loc=query_start_loc,
            query_start_loc_np=query_start_loc_np,
            seq_lens=seq_lens,
            seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
            dcp_local_seq_lens=dcp_local_seq_lens,
            is_prefilling_np=is_prefilling_np,
            input_ids=self.input_buffers.input_ids[:num_tokens_after_padding],
            positions=self.input_buffers.positions[:num_tokens_after_padding],
            logits_indices=logits_indices,
            cu_num_logits=cu_num_logits,
            cu_num_logits_np=cu_num_logits_np,
            has_structured_output_reqs=scheduler_output.has_structured_output_requests,
        )

    def prepare_attn(
        self, input_batch: InputBatch
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        """
        准备注意力机制所需的元数据。

        生成两个关键数据结构：
        1. Block Tables (块表)：每个请求的 KV 缓存块映射
           - 形状：[num_kv_cache_groups, num_reqs_padded, max_num_blocks]
           - 用于注意力内核定位 KV 缓存中的正确位置
        2. Slot Mappings (槽位映射)：每个 token 对应的 KV 缓存槽位
           - 形状：[num_kv_cache_groups, num_tokens_padded]
           - 用于将 K/V 写入正确的位置

        Args:
            input_batch: 输入批次数据

        Returns:
            (block_tables, slot_mappings) 元组
        """
        # Block tables: num_kv_cache_groups x [num_reqs_padded, max_num_blocks].
        block_tables = self.block_tables.gather_block_tables(
            input_batch.idx_mapping,
            num_reqs_padded=input_batch.num_reqs_after_padding,
        )
        # Slot mappings: [num_kv_cache_groups, num_tokens_padded].
        # Kernel pads beyond num_tokens with PAD_SLOT_ID.
        slot_mappings = self.block_tables.compute_slot_mappings(
            input_batch.idx_mapping,
            input_batch.query_start_loc,
            input_batch.positions,
            num_tokens_padded=input_batch.num_tokens_after_padding,
        )
        return block_tables, slot_mappings

    def prepare_dummy_attn(
        self, input_batch: InputBatch
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        """准备虚拟注意力元数据，用于 dummy run 和内存分析。"""
        block_tables = self.block_tables.get_dummy_block_tables(input_batch.num_reqs)
        slot_mappings = self.block_tables.get_dummy_slot_mappings(
            input_batch.num_tokens
        )
        return block_tables, slot_mappings

    def sample(
        self,
        hidden_states: torch.Tensor,
        input_batch: InputBatch,
        grammar_output: GrammarOutput | None,
    ) -> tuple[SamplerOutput, torch.Tensor, torch.Tensor]:
        """
        从模型输出中采样 token。

        这是生成流程的核心步骤，将隐藏状态转换为具体的 token。

        执行流程：
        1. 从隐藏状态中提取需要采样的位置（logits_indices）
        2. 计算 logits（词表上的概率分布）
        3. 如果有结构化输出约束，应用语法掩码限制采样空间
        4. 执行采样：
           - 无草稿 token 时：使用标准采样器
           - 有草稿 token 时：使用拒绝采样器（推测解码）
        5. 计算采样和拒绝的 token 数量

        Args:
            hidden_states: 模型输出的隐藏状态
            input_batch: 输入批次数据
            grammar_output: 结构化输出的语法约束（可选）

        Returns:
            (sampler_output, num_sampled, num_rejected) 元组
        """
        sample_hidden_states = hidden_states[input_batch.logits_indices]
        logits = self.model.compute_logits(sample_hidden_states)
        if grammar_output is not None:
            # Apply grammar bitmask to the logits in-place.
            assert self.structured_outputs_worker is not None
            self.structured_outputs_worker.apply_grammar_bitmask(
                logits,
                input_batch,
                grammar_output.structured_output_request_ids,
                grammar_output.grammar_bitmask,
            )

        if input_batch.num_draft_tokens == 0:
            # No draft tokens (common case).
            assert self.sampler is not None
            sampler_output = self.sampler(logits, input_batch)
        else:
            # Rejection sampling for spec decoding.
            assert self.rejection_sampler is not None
            assert self.speculator is not None
            sampler_output = self.rejection_sampler(
                logits,
                input_batch,
                # Draft logits are needed for probabilistic rejection sampling.
                self.speculator.draft_logits,
            )

        # Get the number of sampled and rejected tokens.
        # For chunked prefills, num_sampled and num_rejected are both 0.
        num_sampled, num_rejected = get_num_sampled_and_rejected(
            sampler_output.num_sampled,
            input_batch.seq_lens,
            input_batch.cu_num_logits,
            input_batch.idx_mapping,
            self.req_states.prefill_len.gpu,
        )
        return sampler_output, num_sampled, num_rejected

    def postprocess(
        self,
        input_batch: InputBatch,
        sampled_tokens: torch.Tensor,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
    ) -> None:
        """
        后处理采样结果，更新请求状态。

        在采样完成后更新以下状态：
        1. 已计算 token 数量（num_computed_tokens）
        2. 最后采样的 token（last_sampled_tokens）
        3. 采样惩罚的输出计数（用于频率/重复惩罚）
        4. 所有 token IDs 和总长度
        5. 模型特定的状态（通过 model_state.postprocess_state）

        Args:
            input_batch: 输入批次数据
            sampled_tokens: 采样得到的 token IDs
            num_sampled: 每个请求采样的 token 数量
            num_rejected: 每个请求被拒绝的 token 数量（推测解码）
        """
        # Update the number of computed tokens.
        if self.is_last_pp_rank:
            assert self.sampler is not None
            output_bin_counts = self.sampler.penalties_state.output_bin_counts
        else:
            output_bin_counts = None
        post_update(
            input_batch.idx_mapping,
            self.req_states.num_computed_tokens.gpu,
            self.req_states.last_sampled_tokens,
            output_bin_counts,
            sampled_tokens,
            num_sampled,
            num_rejected,
            input_batch.query_start_loc,
            self.req_states.all_token_ids.gpu,
            self.req_states.total_len.gpu,
        )

        self.model_state.postprocess_state(input_batch, num_sampled)

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
        intermediate_tensors: IntermediateTensors | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        is_profile: bool = False,
    ) -> ModelRunnerOutput | IntermediateTensors | None:
        """
        执行模型前向传播。

        这是 GPUModelRunner 的核心方法，负责完整的模型执行流程。

        执行流程：
        1. 更新请求状态（如果不是 dummy run）：
           - 完成/移除已完成的请求
           - 添加新请求
           - 更新已有请求的状态
           - 应用块表的暂存写入
        2. 同步数据并行 rank，确定批次描述符
        3. 准备输入数据：
           - 调用 prepare_inputs 构建 InputBatch
           - 调用 prepare_attn 构建注意力元数据
           - 激活 LoRA 适配器（如果配置）
        4. 准备多模态嵌入（如果是多模态模型）
        5. 执行模型前向传播：
           - FULL CUDA 图模式：使用 cudagraph_manager.run_fullgraph()
           - PIECEWISE/EAger 模式：直接调用 model()
        6. 处理输出：
           - 最后一个 PP 阶段：返回隐藏状态
           - 其他 PP 阶段：返回中间张量

        Args:
            scheduler_output: 调度器输出，包含请求分配信息
            intermediate_tensors: 来自前一个 PP 阶段的中间张量
            dummy_run: 是否为虚拟运行（用于 DP 同步或内存分析）
            skip_attn_for_dummy_run: 虚拟运行时是否跳过注意力
            is_profile: 是否为内存分析运行

        Returns:
            - 最后一个 PP 阶段返回 None（状态保存在 execute_model_state 中）
            - 非最后一个 PP 阶段返回 IntermediateTensors
            - 如果没有需要运行的 token，返回空输出
        """
        if not dummy_run:
            # 更新请求状态（实际推理时）
            # 1. 完成/移除已完成的请求
            # 2. 释放编码器缓存
            # 3. 添加新请求到运行器
            # 4. 更新已有请求的状态
            # 5. 应用块表的暂存写入
            self.finish_requests(scheduler_output)
            self.free_states(scheduler_output)
            self.add_requests(scheduler_output)
            self.update_requests(scheduler_output)
            self.block_tables.apply_staged_writes()
            if scheduler_output.total_num_scheduled_tokens == 0:
                # No need to run the model.
                # 没有需要运行的 token，直接返回空输出
                empty_output = self.kv_connector.no_forward(scheduler_output)
                return empty_output

        # Get batch descriptor and sync across DP ranks.
        # 获取批次描述符并同步所有数据并行 rank
        # 确保所有 DP rank 使用相同的 CUDA 图模式和批次大小
        num_reqs = len(scheduler_output.num_scheduled_tokens)
        num_toks = scheduler_output.total_num_scheduled_tokens
        max_query_len = max(scheduler_output.num_scheduled_tokens.values())
        uniform_tok_count = get_uniform_token_count(num_reqs, num_toks, max_query_len)

        skip_compiled = False
        if self.is_encoder_decoder and scheduler_output.scheduled_encoder_inputs:
            # Encoder-decoder models such as Whisper should run eager/non-compiled
            # when encoder inputs are scheduled, because this step updates
            # cross-attention cache with dynamic encoder outputs.
            skip_compiled = True

        batch_desc, num_tokens_across_dp = dispatch_cg_and_sync_dp(
            self.cudagraph_manager,
            num_reqs,
            num_toks,
            uniform_tok_count,
            self.dp_size,
            self.dp_rank,
            need_eager=is_profile or skip_compiled,
        )

        if batch_desc.num_tokens == 0:
            # All DP ranks have zero tokens to run.
            empty_output = self.kv_connector.no_forward(scheduler_output)
            return empty_output

        if not dummy_run:
            # Common case.
            # Prepare all the inputs and copy to the input buffers.
            input_batch = self.prepare_inputs(scheduler_output, batch_desc)
            block_tables, slot_mappings = self.prepare_attn(input_batch)

            if self.lora_config:
                # Activate LoRA adapters.
                lora_inputs = self.lora_state.make_lora_inputs(
                    input_batch.req_ids,
                    input_batch.idx_mapping_np,
                    input_batch.num_scheduled_tokens,
                )
                self._set_active_loras(*lora_inputs)
        else:
            # No actual tokens to run. A dummy run for DP or memory profiling.
            input_batch = InputBatch.make_dummy(
                batch_desc.num_reqs or num_reqs,
                batch_desc.num_tokens,
                self.input_buffers,
            )
            if not skip_attn_for_dummy_run:
                block_tables, slot_mappings = self.prepare_dummy_attn(input_batch)
            else:
                assert batch_desc.cg_mode != CUDAGraphMode.FULL, (
                    "Attention metadata must be prepared for dummy runs when using "
                    "FULL cudagraph mode."
                )
                block_tables = None
                slot_mappings = None
            if self.lora_config:
                # program a no-LoRA mapping here so kernels early-exit instead of
                # reading uninitialized metadata during dummy runs.
                # FIXME: Replace this with LoRA warmup:
                # https://github.com/vllm-project/vllm/pull/35536
                assert hasattr(self, "lora_manager")
                adapter_manager = self.lora_manager._adapter_manager
                adapter_manager.set_adapter_mapping(
                    LoRAMapping(
                        index_mapping=(0,) * input_batch.num_tokens_after_padding,
                        prompt_mapping=(0,) * input_batch.num_reqs,
                        is_prefill=True,
                    )
                )
                seen_wrappers: set[int] = set()
                for punica_wrapper in adapter_manager.punica_wrapper_mapping.values():
                    if id(punica_wrapper) in seen_wrappers:
                        continue
                    seen_wrappers.add(id(punica_wrapper))
                    for kernel_meta in (
                        punica_wrapper.token_mapping_meta,  # type: ignore[attr-defined]
                        punica_wrapper.prompt_mapping_meta,  # type: ignore[attr-defined]
                    ):
                        kernel_meta.no_lora_flag_cpu[0] = False
                        kernel_meta.num_active_loras_cpu[0] = 1

        attn_metadata = None
        slot_mappings_by_layer = None
        if not (dummy_run and skip_attn_for_dummy_run):
            assert slot_mappings is not None
            slot_mappings_by_layer = build_slot_mappings_by_layer(
                slot_mappings, self.kv_cache_config
            )
            assert block_tables is not None
            attn_metadata = self.model_state.prepare_attn(
                input_batch,
                batch_desc.cg_mode,
                block_tables,
                slot_mappings,
                self.attn_groups,
                self.kv_cache_config,
            )

        inputs_embeds = None
        if self.supports_mm_inputs and self.is_first_pp_rank:
            # Run MM encoder (if needed) and get multimodal embeddings.
            # Only first PP rank prepares multimodal embeddings.
            # NOTE(woosuk): We must call get_mm_embeddings even during dummy runs
            # to obtain inputs_embeds, because the compiled model expects this input.
            inputs_embeds = self.model_state.get_mm_embeddings(
                scheduler_output.scheduled_encoder_inputs,
                input_batch,
                self.req_states,
            )

        model_inputs = {
            "input_ids": input_batch.input_ids,
            "positions": input_batch.positions,
            "inputs_embeds": inputs_embeds,
            # NOTE: Values returned by `prepare_inputs` will override the default
            # values above.
            **self.model_state.prepare_inputs(input_batch, self.req_states),
        }
        if not self.is_first_pp_rank:
            # Update for non-first PP ranks.
            model_inputs["input_ids"] = None
            model_inputs["inputs_embeds"] = None

            # Prepare the intermediate tensors.
            assert intermediate_tensors is not None
            assert self.intermediate_tensors is not None
            n = input_batch.num_tokens_after_padding
            model_inputs["intermediate_tensors"] = IntermediateTensors(
                {
                    k: v[:n].copy_(intermediate_tensors.tensors[k][:n])
                    for k, v in self.intermediate_tensors.tensors.items()
                }
            )
            del intermediate_tensors

        # Run model.
        # 执行模型前向传播
        if batch_desc.cg_mode == CUDAGraphMode.FULL:
            # Use explicit cudagraph replay for FULL mode.
            # NOTE(woosuk): Here, we don't need to pass the input tensors,
            # because they are already copied to the CUDA graph input buffers.
            # FULL CUDA 图模式：重放预捕获的 CUDA 图
            # 输入张量已经在 prepare_inputs 时拷贝到 CUDA 图的输入缓冲区中
            assert self.cudagraph_manager is not None
            self.kv_connector.pre_forward(scheduler_output)
            model_output = self.cudagraph_manager.run_fullgraph(batch_desc)
        else:
            # For piecewise and eager mode, just call model().
            # PIECEWISE 或 EAger 模式：直接调用模型
            batch_descriptor = BatchDescriptor(
                num_tokens=input_batch.num_tokens_after_padding,
                has_lora=self.lora_config is not None,
            )

            with set_forward_context(
                attn_metadata,
                self.vllm_config,
                num_tokens=input_batch.num_tokens_after_padding,
                cudagraph_runtime_mode=batch_desc.cg_mode,
                num_tokens_across_dp=num_tokens_across_dp,
                batch_descriptor=batch_descriptor,
                slot_mapping=slot_mappings_by_layer,
                skip_compiled=skip_compiled,
            ):
                self.kv_connector.pre_forward(scheduler_output)
                model_output = self.model(**model_inputs)

        # 处理模型输出
        if self.is_last_pp_rank:
            # 最后一个 PP 阶段：输出是隐藏状态
            if self.use_aux_hidden_state_outputs:
                # EAGLE3 推测解码需要辅助隐藏状态
                assert isinstance(model_output, tuple)
                hidden_states, aux_hidden_states = model_output
            else:
                assert isinstance(model_output, torch.Tensor)
                hidden_states = model_output
                aux_hidden_states = None
            output_intermediate_tensors = None
        else:
            # 非最后一个 PP 阶段：输出是中间张量，需要传递给下一个阶段
            assert isinstance(model_output, IntermediateTensors)
            hidden_states = None
            aux_hidden_states = None
            output_intermediate_tensors = model_output

        # 保存 execute_model 状态，供 sample_tokens 使用
        finished_req_ids = scheduler_output.finished_req_ids
        self.execute_model_state = ExecuteModelState(
            input_batch=input_batch,
            attn_metadata=attn_metadata,
            slot_mappings_by_layer=slot_mappings_by_layer,
            hidden_states=hidden_states,
            aux_hidden_states=aux_hidden_states,
            finished_req_ids=finished_req_ids,
        )

        if not self.is_last_pp_rank:
            # Non-last PP rank: return IntermediateTensors for sending.
            return output_intermediate_tensors
        return None

    @torch.inference_mode()
    @step_eplb_after()
    def sample_tokens(
        self, grammar_output: GrammarOutput | None
    ) -> AsyncOutput | ModelRunnerOutput | None:
        """
        从 execute_model 的输出中采样 token 并构建最终输出。

        这个方法与 execute_model 分离，支持异步调度：
        execute_model 可以在 sample_tokens 完成之前开始处理下一批次。

        执行流程：
        1. 从 execute_model_state 中恢复上下文
        2. 非最后一个 PP 阶段：接收广播的采样结果并更新状态
        3. 最后一个 PP 阶段：
           a. 调用 sample() 执行采样
           b. 如果启用 PP，广播采样结果到其他 rank
           c. 计算 prompt logprobs（如果需要）
           d. 创建异步输出对象（AsyncOutput）
           e. 收集多模态嵌入（如果使用推测解码 + 多模态）
           f. 执行 postprocess 更新请求状态
           g. 如果存在推测解码器，调用 propose() 生成草稿 token
        4. 处理 KV 连接器的后向操作
        5. 返回输出（异步或同步模式）

        Args:
            grammar_output: 结构化输出的语法约束（可选）

        Returns:
            - 异步调度模式：返回 AsyncOutput（可延迟获取实际结果）
            - 同步调度模式：返回 ModelRunnerOutput
            - 如果 execute_model 失败，返回 None
        """
        if self.execute_model_state is None:
            # The prior execute_model call must have failed.
            return None

        input_batch = self.execute_model_state.input_batch
        attn_metadata = self.execute_model_state.attn_metadata
        slot_mappings_by_layer = self.execute_model_state.slot_mappings_by_layer
        hidden_states = self.execute_model_state.hidden_states
        aux_hidden_states = self.execute_model_state.aux_hidden_states
        finished_req_ids = self.execute_model_state.finished_req_ids
        self.execute_model_state = None

        if not self.is_last_pp_rank:
            # Non-last PP rank: hidden_states is None because this rank produced
            # IntermediateTensors instead of final hidden states. Receive the
            # sampled tokens broadcast from the last rank and update local state.
            sampled, num_sampled, num_rejected = pp_receive(
                input_batch.num_reqs, max_sample_len=self.num_speculative_steps + 1
            )
            self.postprocess(input_batch, sampled, num_sampled, num_rejected)

            # Post-step KV connector related operations.
            kv_connector_output = self.kv_connector.post_forward(finished_req_ids)
            return ModelRunnerOutput.with_kv_conn_output_only(kv_connector_output)

        # Last rank: sample tokens
        sampler_output, num_sampled, num_rejected = self.sample(
            hidden_states, input_batch, grammar_output
        )

        if self.use_pp:
            # Broadcast to non-last PP ranks (handles spec decode multi-token).
            pp_broadcast(sampler_output.sampled_token_ids, num_sampled, num_rejected)

        assert self.prompt_logprobs_worker is not None
        prompt_logprobs_dict = self.prompt_logprobs_worker.compute_prompt_logprobs(
            self.model.compute_logits,
            hidden_states,
            input_batch,
            self.req_states.all_token_ids.gpu,
            self.req_states.num_computed_tokens.gpu,
            self.req_states.prompt_len.np,
            self.req_states.prefill_len.np,
            self.req_states.num_computed_prefill_tokens,
        )

        # Prepare the model runner output.
        model_runner_output = ModelRunnerOutput(
            req_ids=input_batch.req_ids,
            # NOTE(woosuk): req_id_to_index is unused in this model runner.
            # Only for compatibility with the existing model runner and scheduler.
            req_id_to_index={req_id: i for i, req_id in enumerate(input_batch.req_ids)},
            sampled_token_ids=None,  # type: ignore
            prompt_logprobs_dict=prompt_logprobs_dict,  # type: ignore[arg-type]
        )
        # Start async output copy here so that it can overlap with speculator proposal.
        async_output = AsyncOutput(
            model_runner_output=model_runner_output,
            sampler_output=sampler_output,
            num_sampled_tokens=num_sampled,
            main_stream=self.main_stream,
            copy_stream=self.output_copy_stream,
        )

        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None
        if self.speculator is not None and self.speculator.supports_mm_inputs:
            # Get cached multimodal embeddings for draft forward.
            # NOTE: This is done here because postprocess updates
            # num_computed_prefill_tokens.
            prefill_lens = self.req_states.prefill_len.np[input_batch.idx_mapping_np]
            computed_prefill_lens = self.req_states.num_computed_prefill_tokens[
                input_batch.idx_mapping_np
            ]
            mm_inputs = self.model_state.encoder_runner.gather_mm_embeddings(
                input_batch.req_ids,
                input_batch.num_tokens,
                input_batch.num_scheduled_tokens,
                input_batch.query_start_loc_np,
                prefill_lens,
                computed_prefill_lens + 1,  # +1 to consider the skew in eagle
            )

        # Postprocess results and update request states.
        # NOTE: This is intentionally done after creating the AsyncOutput,
        # ensuring that `copy_event` is recorded before calling postprocess.
        # This sequencing may slightly reduce latency as async D2H copy does not
        # need to wait for the postprocess to finish.
        self.postprocess(
            input_batch, sampler_output.sampled_token_ids, num_sampled, num_rejected
        )

        if self.speculator is not None:
            assert self.sampler is not None
            # Let the target override the hidden state fed to the drafter
            # (e.g. DeepSeek V4 MTP needs the pre-hc_head residual). The
            # target returns a persistent buffer sized at max_num_batched_tokens;
            # slice to the active token count that propose() expects.
            spec_hidden_states = hidden_states
            if hasattr(self.model, "get_mtp_target_hidden_states"):
                pre_hc_hidden_states = self.model.get_mtp_target_hidden_states()
                spec_hidden_states = pre_hc_hidden_states[: hidden_states.shape[0]]  # type: ignore[union-attr]
            draft_tokens = self.speculator.propose(
                input_batch,
                attn_metadata,
                slot_mappings_by_layer,
                spec_hidden_states,
                aux_hidden_states,
                num_sampled,
                num_rejected,
                self.req_states.last_sampled_tokens,
                self.req_states.next_prefill_tokens,
                self.sampler.sampling_states.temperature.gpu,
                self.sampler.sampling_states.seeds.gpu,
                mm_inputs=mm_inputs,
            )
            self.req_states.draft_tokens[input_batch.idx_mapping] = draft_tokens
            self.draft_tokens_handler.set_draft_tokens(input_batch, draft_tokens)

        # Post-step KV connector related operations.
        kv_connector_output = self.kv_connector.post_forward(finished_req_ids)
        model_runner_output.kv_connector_output = kv_connector_output

        if self.use_async_scheduling:
            return async_output
        return async_output.get_output()

    def take_draft_token_ids(self) -> DraftTokenIds | None:
        """获取推测解码的草稿 token IDs。"""
        return self.draft_tokens_handler.get_draft_tokens()

    @torch.inference_mode()
    @step_eplb_after()
    def pool(self) -> AsyncPoolingOutput | ModelRunnerOutput | None:
        """
        执行池化操作（用于嵌入模型等池化任务）。

        与 sample_tokens 类似，但使用 PoolingRunner 而非采样器。
        适用于嵌入、分类、排序等任务。

        Returns:
            异步或同步的池化输出
        """
        if self.execute_model_state is None:
            # The prior execute_model call must have failed.
            return None

        input_batch = self.execute_model_state.input_batch
        hidden_states = self.execute_model_state.hidden_states
        finished_req_ids = self.execute_model_state.finished_req_ids
        self.execute_model_state = None

        # Post-step KV connector related operations.
        kv_connector_output = self.kv_connector.post_forward(finished_req_ids)

        if not self.is_last_pp_rank:
            self.postprocess_pool(input_batch)
            return ModelRunnerOutput.with_kv_conn_output_only(kv_connector_output)

        assert self.pooling_runner is not None
        pooler_output, is_valid = self.pooling_runner.pool(
            hidden_states, input_batch, self.req_states
        )

        # Build the model runner output.
        model_runner_output = ModelRunnerOutput(
            req_ids=input_batch.req_ids,
            req_id_to_index={req_id: i for i, req_id in enumerate(input_batch.req_ids)},
            kv_connector_output=kv_connector_output,
        )
        async_output = AsyncPoolingOutput(
            model_runner_output=model_runner_output,
            pooler_output=pooler_output,
            is_valid=is_valid,
            main_stream=self.main_stream,
            copy_stream=self.output_copy_stream,
        )

        self.postprocess_pool(input_batch)
        if self.use_async_scheduling:
            return async_output
        return async_output.get_output()

    def postprocess_pool(self, input_batch: InputBatch) -> None:
        """池化任务的后处理，更新已计算 token 数量。"""
        # Update the number of computed tokens.
        post_update_pool(
            input_batch.idx_mapping,
            self.req_states.num_computed_tokens.gpu,
            input_batch.query_start_loc,
        )

    def shutdown(self) -> None:
        """
        关闭模型运行器，释放所有 GPU 资源。

        释放的资源包括：
        - 模型权重
        - KV 缓存张量
        - 注意力组配置
        - CUDA 图
        - 其他工作区内存

        调用后会触发垃圾回收和 CUDA 缓存清理。
        """
        torch.accelerator.synchronize()
        if hasattr(self, "kv_caches"):
            self.kv_caches.clear()
        if hasattr(self, "attn_groups"):
            self.attn_groups.clear()
        if hasattr(self, "kv_cache_config"):
            del self.kv_cache_config
        free_before_shutdown(self.vllm_config)
        if hasattr(self, "model"):
            del self.model

        gc.collect()
        torch.accelerator.empty_cache()
        logger.debug("Cleaned up model weights, KV caches, and workspace")

    # =================================================================
    # 专家并行负载均衡 (Expert Parallelism Load Balancing, EPLB)
    # =================================================================
    # EPLB 用于 MoE (Mixture of Experts) 模型的动态负载均衡。
    # 当某些专家被频繁使用时，EPLB 会将它们迁移到更空闲的 GPU 上。
    ########### EPLB methods start ###########
    @property
    def eplb_state(self):
        return self.eplb.state

    @eplb_state.setter
    def eplb_state(self, state) -> None:
        self.eplb.state = state

    @property
    def eep_eplb_suppressed(self) -> bool:
        return self.eplb.suppressed

    @eep_eplb_suppressed.setter
    def eep_eplb_suppressed(self, suppressed: bool) -> None:
        self.eplb.suppressed = suppressed

    def setup_eplb_from_mapping(
        self,
        expanded_physical_to_logical: torch.Tensor,
        old_num_physical_experts: int,
    ) -> None:
        self.eplb.setup_from_mapping(
            self.model,
            self.model_config,
            expanded_physical_to_logical,
            old_num_physical_experts,
        )

    ########### EPLB methods end ###########


class ExecuteModelState(NamedTuple):
    """
    execute_model 和 sample_tokens 之间传递的状态。

    这个 NamedTuple 用于在 execute_model（前向传播）和 sample_tokens（采样）
    之间传递必要的上下文信息。这种设计支持异步调度，允许 execute_model 提前返回。

    属性：
        input_batch: 输入批次数据（包含请求 ID、索引映射等）
        attn_metadata: 注意力元数据（用于推测解码的草稿模型）
        slot_mappings_by_layer: 每层的槽位映射（用于 KV 缓存写入）
        hidden_states: 模型输出的隐藏状态（最后一个 PP 阶段）
        aux_hidden_states: 辅助隐藏状态（EAGLE3 推测解码使用）
        finished_req_ids: 本批次完成的请求 ID 集合
    """
    input_batch: InputBatch
    attn_metadata: dict[str, Any] | None
    slot_mappings_by_layer: dict[str, torch.Tensor] | None
    hidden_states: torch.Tensor | None
    aux_hidden_states: list[torch.Tensor] | None
    finished_req_ids: set[str]
