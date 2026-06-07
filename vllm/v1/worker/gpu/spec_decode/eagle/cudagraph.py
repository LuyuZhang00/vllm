# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
EAGLE CUDA 图管理器模块。

本模块实现了 EAGLE 投机解码器专用的 CUDA 图管理器。

CUDA 图（CUDA Graph）是一种性能优化技术，通过将一系列 GPU 操作录制为一个图，
然后可以高效地重放这个图，避免重复的 kernel launch 开销。

对于 EAGLE 投机解码，CUDA 图特别重要，因为：
1. 草稿模型的前向传播涉及多个小 kernel，launch 开销显著
2. 每次迭代的计算模式相同，适合图重放
3. 可以显著减少 CPU-GPU 同步开销

本模块提供两个管理器：
1. PrefillEagleCudaGraphManager: 管理草稿预填充的 CUDA 图
2. DecodeEagleCudaGraphManager: 管理草稿解码的 CUDA 图

注意：EAGLE 使用专用的 CUDA 图内存池，避免与主模型的 CUDA 图冲突。
"""

from collections.abc import Callable

import torch

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.cudagraph_utils import (
    BatchExecutionDescriptor,
    CapturedAttentionState,
    CudaGraphManager,
    prepare_inputs_to_capture,
)
from vllm.v1.worker.gpu.input_batch import InputBuffers
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.utils import AttentionGroup


class EagleCudaGraphManagerBase(CudaGraphManager):
    """
    EAGLE CUDA 图管理器基类。

    继承自通用的 CudaGraphManager，但使用专用的 CUDA 图内存池，
    避免与主模型的 CUDA 图发生内存冲突。

    专用内存池的必要性：
    - EAGLE 的内部分配（如 gumbel_sample 的临时缓冲区）可能与主模型冲突
    - 共享内存池可能导致地址重叠和数据损坏

    属性:
        pool: 专用的 CUDA 图内存池句柄。
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        cudagraph_mode: CUDAGraphMode,
        decode_query_len: int,
    ):
        """
        初始化 EAGLE CUDA 图管理器基类。

        参数:
            vllm_config (VllmConfig): vLLM 全局配置。
            device (torch.device): 计算设备。
            cudagraph_mode (CUDAGraphMode): CUDA 图模式。
            decode_query_len (int): 解码查询长度。
        """
        super().__init__(vllm_config, device, cudagraph_mode, decode_query_len)

        # Use a dedicated pool for Eagle to avoid memory overlap with the main
        # model's cudagraph. The base class uses a shared global pool, but Eagle's
        # internal allocations (e.g., gumbel_sample temporaries) can conflict with
        # the main model's allocations when sharing the same pool.
        # 使用专用内存池，避免与主模型的 CUDA 图发生内存冲突
        if cudagraph_mode:
            self.pool = torch.cuda.graph_pool_handle()


class PrefillEagleCudaGraphManager(EagleCudaGraphManagerBase):
    """
    EAGLE 预填充 CUDA 图管理器。

    管理草稿预填充（draft prefill）的 CUDA 图。
    预填充阶段处理所有请求的初始输入，生成第一个草稿 token。

    特点：
    - 使用目标模型预构建的注意力状态（从 target model capture 复用）
    - 支持变化的 token 数量（通过不同的 capture size）
    - 每个请求有 num_speculative_steps + 1 个 token
    """

    def capture(
        self,
        forward_fn: Callable,
        full_cg_attn_states: dict[BatchExecutionDescriptor, CapturedAttentionState],
        progress_bar_desc: str = "Capturing CUDA graphs",
    ) -> None:
        """
        捕获预填充 CUDA 图。

        使用目标模型预构建的注意力状态，为不同的批次大小捕获 CUDA 图。

        参数:
            forward_fn (Callable): 前向传播函数，签名 (num_reqs, num_tokens, attn_metadata, slot_mappings, num_tokens_across_dp, cg_mode)。
            full_cg_attn_states (dict): 目标模型预构建的注意力状态字典。
            progress_bar_desc (str): 进度条描述。

        流程:
            1. 对于每个捕获大小，创建前向函数闭包
            2. 从 full_cg_attn_states 中获取预构建的注意力元数据
            3. 调用基类的 capture 方法进行实际的图捕获
        """
        def create_forward_fn(
            desc: BatchExecutionDescriptor,
        ) -> tuple[Callable[[CUDAGraphMode], None], CapturedAttentionState]:
            num_tokens = desc.num_tokens
            num_reqs = desc.num_reqs or min(num_tokens, self.max_num_reqs)
            num_tokens_across_dp = (
                torch.full((self.dp_size,), num_tokens, dtype=torch.int32, device="cpu")
                if self.dp_size > 1
                else None
            )
            # 从预构建的状态中获取注意力元数据
            attn_state = full_cg_attn_states[desc]
            attn_metadata, slot_mappings = attn_state
            fwd = lambda cg_mode: forward_fn(
                num_reqs,
                num_tokens,
                attn_metadata,
                slot_mappings,
                num_tokens_across_dp,
                cg_mode,
            )
            return fwd, attn_state

        super().capture(create_forward_fn, progress_bar_desc)


class DecodeEagleCudaGraphManager(EagleCudaGraphManagerBase):
    """
    EAGLE 解码 CUDA 图管理器。

    管理草稿解码（draft decode）的 CUDA 图。
    解码阶段逐个生成草稿 token，每个请求每次生成 1 个 token。

    特点：
    - 自行构建注意力元数据（不复用目标模型的）
    - 每个请求固定 1 个 token（decode_query_len=1）
    - 支持 FULL 和 PIECEWISE 两种 CUDA 图模式
    """

    def capture(
        self,
        forward_fn: Callable,
        model_state: ModelState,
        input_buffers: InputBuffers,
        block_tables: BlockTables,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        progress_bar_desc: str = "Capturing CUDA graphs",
    ) -> None:
        """
        捕获解码 CUDA 图。

        自行构建注意力元数据，为不同的批次大小捕获 CUDA 图。

        参数:
            forward_fn (Callable): 前向传播函数。
            model_state (ModelState): 模型状态。
            input_buffers (InputBuffers): 输入缓冲区。
            block_tables (BlockTables): Block table 管理器。
            attn_groups (list): 注意力组。
            kv_cache_config (KVCacheConfig): KV cache 配置。
            progress_bar_desc (str): 进度条描述。

        流程:
            1. 对于每个捕获大小，创建前向函数闭包
            2. 调用 prepare_inputs_to_capture 构建注意力元数据
            3. 调用基类的 capture 方法进行实际的图捕获
        """
        def create_forward_fn(
            desc: BatchExecutionDescriptor,
        ) -> tuple[Callable[[CUDAGraphMode], None], CapturedAttentionState]:
            num_tokens = desc.num_tokens
            num_reqs = desc.num_reqs or min(num_tokens, self.max_num_reqs)
            num_tokens_across_dp = (
                torch.full((self.dp_size,), num_tokens, dtype=torch.int32, device="cpu")
                if self.dp_size > 1
                else None
            )
            # 自行构建注意力元数据（不复用目标模型的）
            attn_state = prepare_inputs_to_capture(
                num_reqs,
                num_tokens,
                model_state,
                input_buffers,
                block_tables,
                attn_groups,
                kv_cache_config,
                skip_attn=(desc.cg_mode == CUDAGraphMode.PIECEWISE),
            )
            attn_metadata, slot_mappings = attn_state

            fwd = lambda cg_mode: forward_fn(
                num_reqs,
                num_tokens,
                attn_metadata,
                slot_mappings,
                num_tokens_across_dp,
                cg_mode,
            )
            return fwd, attn_state

        super().capture(create_forward_fn, progress_bar_desc)
