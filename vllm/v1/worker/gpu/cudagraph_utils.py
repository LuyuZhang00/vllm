# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
CUDA Graph 管理工具模块。

本模块实现了 vLLM v1 引擎的 CUDA Graph 捕获和回放功能。
CUDA Graph 可以将一系列 CUDA 操作预录为一个图，后续只需回放该图即可，
避免了 CPU 端的内核启动开销，显著提升推理性能。

核心概念：
1. CUDA Graph 模式：
   - NONE: 不使用 CUDA Graph（eager 模式）
   - FULL: 完整 CUDA Graph，整个前向传播被捕获为一个图
   - PIECEWISE: 分段 CUDA Graph，支持将不同形状的子图分别捕获

2. 批次执行描述符（BatchExecutionDescriptor）：
   - 描述一个批次的形状（token 数、请求数）和 CUDA Graph 模式
   - 用于在捕获和运行时进行形状匹配

3. 统一 token 计数（uniform_token_count）：
   - 当批次中所有请求的 token 数相同时，记录该值
   - 用于匹配专用的解码 CUDA Graph

主要组件：
1. CudaGraphManager - 通用的 CUDA Graph 管理器
2. ModelCudaGraphManager - 模型专用的 CUDA Graph 管理器（管理隐藏状态）
3. CapturedAttentionState - 捕获的注意力状态
4. BatchExecutionDescriptor - 批次执行描述符
"""
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, NamedTuple

import torch
import torch.nn as nn
from tqdm import tqdm

from vllm.compilation.counter import compilation_counter
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed.parallel_state import (
    get_pp_group,
    graph_capture,
    is_global_first_rank,
)
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.offloader.base import get_offloader
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.cp_utils import prepare_dcp_local_seq_lens
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.utils import AttentionGroup

logger = init_logger(__name__)


class CapturedAttentionState(NamedTuple):
    """CUDA Graph 捕获时的注意力状态。

    属性:
        attn_metadata: 捕获时的注意力元数据（PIECEWISE 模式下为 None）
        slot_mappings: 每层的 slot 映射字典
    """
    attn_metadata: dict[str, Any] | None
    slot_mappings: dict[str, torch.Tensor]


@dataclass(frozen=True)
class BatchExecutionDescriptor:
    """描述批次形状和 CUDA Graph 模式的不可变数据类。

    用于在 CUDA Graph 捕获和运行时之间进行形状匹配。

    属性:
        cg_mode: CUDA Graph 模式（NONE/FULL/PIECEWISE）
        num_tokens: 批次中的 token 总数
        num_reqs: 批次中的请求数（PIECEWISE 模式下为 None，表示不需要请求填充）
        uniform_token_count: 统一的每请求 token 数（用于匹配专用解码图）
    """

    cg_mode: CUDAGraphMode
    num_tokens: int
    num_reqs: int | None  # None means no request padding is needed (PIECEWISE graphs)
    uniform_token_count: int | None = None


def _is_compatible(
    desc: BatchExecutionDescriptor,
    num_reqs: int,
    num_tokens: int,
    uniform_token_count: int | None,
) -> bool:
    """检查批次执行描述符是否与给定的批次参数兼容。

    兼容性条件：
    1. uniform_token_count 匹配（或描述符不关心此值）
    2. 请求数不超过描述符的最大请求数（或描述符不关心此值）
    3. token 数不超过描述符的最大 token 数

    参数:
        desc: 待检查的批次执行描述符
        num_reqs: 实际请求数
        num_tokens: 实际 token 数
        uniform_token_count: 实际的统一 token 计数

    返回:
        bool: 是否兼容
    """
    # desc.uniform_token_count=None (PIECEWISE) can handle any uniform_token_count
    # desc.num_reqs=None means no request padding needed (PIECEWISE)
    return (
        (
            desc.uniform_token_count is None
            or desc.uniform_token_count == uniform_token_count
        )
        and (desc.num_reqs is None or desc.num_reqs >= num_reqs)
        and desc.num_tokens >= num_tokens
    )


def get_uniform_token_count(
    num_reqs: int,
    num_tokens: int,
    max_query_len: int,
) -> int | None:
    """判断批次是否为"统一"批次，并返回统一的 token 计数。

    统一批次是指批次中所有请求拥有相同数量的 token。
    这在纯解码阶段很常见（每个请求恰好 1 个 token）。

    参数:
        num_reqs: 请求数量
        num_tokens: token 总数
        max_query_len: 最大查询长度

    返回:
        int | None: 统一的每请求 token 数，非统一时返回 None
    """
    if (max_query_len == num_tokens // num_reqs) and (
        num_tokens == max_query_len * num_reqs
    ):
        return max_query_len
    return None


class CudaGraphManager:
    """通用的 CUDA Graph 管理器。

    负责：
    1. 初始化候选的 CUDA Graph 描述符列表
    2. 捕获 CUDA Graph
    3. 根据批次形状调度合适的 CUDA Graph
    4. 回放 CUDA Graph

    属性:
        vllm_config: vLLM 全局配置
        device: 计算设备
        max_num_reqs: 最大请求数
        cudagraph_mode: CUDA Graph 模式
        decode_query_len: 解码阶段的查询长度（通常为 1 + num_spec_steps）
        graphs: 已捕获的 CUDA Graph 字典
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        cudagraph_mode: CUDAGraphMode,
        decode_query_len: int,
    ):
        self.vllm_config = vllm_config
        self.device = device
        self.max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        self.compilation_config = vllm_config.compilation_config
        assert self.compilation_config is not None
        self.cudagraph_mode = cudagraph_mode
        self.decode_query_len = decode_query_len

        self.dp_size = vllm_config.parallel_config.data_parallel_size
        self.tp_size = vllm_config.parallel_config.tensor_parallel_size
        self.is_first_pp_rank = get_pp_group().is_first_rank
        self.is_last_pp_rank = get_pp_group().is_last_rank

        self.graphs: dict[BatchExecutionDescriptor, torch.cuda.CUDAGraph] = {}
        self.pool = current_platform.get_global_graph_pool() if cudagraph_mode else None

        self._graphs_captured = False
        self._candidates: list[list[BatchExecutionDescriptor]] = []
        self._capture_descs: dict[CUDAGraphMode, list[BatchExecutionDescriptor]] = {}
        # 调整 CUDA Graph 捕获大小，使其为统一解码查询长度的倍数
        self.compilation_config.adjust_cudagraph_sizes_for_spec_decode(
            self.decode_query_len, self.tp_size
        )
        self._init_candidates()

    def _init_candidates(self) -> None:
        """构建按优先级排序的候选 CUDA Graph 描述符列表。

        对于每个 token 数，创建候选描述符列表。在运行时，通过匹配
        token 数快速找到兼容的 CUDA Graph。

        流程：
        1. 遍历所有需要捕获的大小
        2. 为解码模式（统一 token 计数）创建专用描述符
        3. 为混合模式（任意 token 分布）创建通用描述符
        4. 构建从 token 数到候选列表的映射
        """
        capture_sizes = self.compilation_config.cudagraph_capture_sizes
        if not (self.cudagraph_mode and capture_sizes):
            return

        capture_sizes = sorted(capture_sizes)
        max_decode_tokens = self.max_num_reqs * self.decode_query_len
        decode_mode = self.cudagraph_mode.decode_mode()
        mixed_mode = self.cudagraph_mode.mixed_mode()
        separate_decode_routine = self.cudagraph_mode.separate_routine()

        descs_by_token_count = defaultdict(list)
        descs_by_mode = defaultdict(list)

        for num_tokens in capture_sizes:
            # 如果需要独立的解码路径，捕获统一解码专用图
            #  (i.e. separate decode routine)
            if (
                separate_decode_routine
                and decode_mode
                and self.decode_query_len <= num_tokens <= max_decode_tokens
            ):
                desc = BatchExecutionDescriptor(
                    cg_mode=decode_mode,
                    num_tokens=num_tokens,
                    num_reqs=num_tokens // self.decode_query_len,
                    uniform_token_count=self.decode_query_len,
                )
                descs_by_mode[decode_mode].append(desc)
                descs_by_token_count[num_tokens].append(desc)

            if mixed_mode:
                # 对于 PIECEWISE 图，回放时没有请求数限制
                # （即不需要请求填充），所以设为 None
                num_reqs = (
                    min(num_tokens, self.max_num_reqs)
                    if mixed_mode == CUDAGraphMode.FULL
                    else None
                )
                desc = BatchExecutionDescriptor(
                    cg_mode=mixed_mode,
                    num_tokens=num_tokens,
                    num_reqs=num_reqs,
                )
                descs_by_mode[mixed_mode].append(desc)
                descs_by_token_count[num_tokens].append(desc)

        if not descs_by_token_count:
            return

        # 构建从 token 数到候选描述符列表的映射
        sorted_padded = sorted(descs_by_token_count.keys())
        self._candidates = [[] for _ in range(sorted_padded[-1] + 1)]

        current_range_start = 0
        for cg_size in sorted_padded:
            for i in range(current_range_start, cg_size + 1):
                self._candidates[i] = descs_by_token_count[cg_size]
            current_range_start = cg_size + 1

        # 按模式对描述符排序（从大到小）
        for mode, descs in descs_by_mode.items():
            descs.sort(key=lambda d: d.num_tokens, reverse=True)
            self._capture_descs[mode] = descs

    def needs_capture(self) -> bool:
        """检查是否需要捕获 CUDA Graph。"""
        return len(self._capture_descs) > 0

    @torch.inference_mode()
    def capture(
        self,
        create_forward_fn: Callable[
            [BatchExecutionDescriptor],
            tuple[Callable[[CUDAGraphMode], None], CapturedAttentionState],
        ],
        progress_bar_desc: str = "Capturing CUDA graphs",
    ) -> dict[BatchExecutionDescriptor, CapturedAttentionState]:
        """捕获 CUDA Graph。

        捕获顺序：先 PIECEWISE，再 FULL。PIECEWISE 有更大的激活张量，
        所以 FULL 的激活张量应该能适配图池中已分配的缓冲区。

        参数:
            create_forward_fn: 工厂函数，在图外部准备输入并返回
                (forward_fn, captured_attn_state) 元组
            progress_bar_desc: 进度条描述文本

        返回:
            dict: 描述符 -> 捕获的注意力状态
        """
        captured_attn_states: dict[
            BatchExecutionDescriptor, CapturedAttentionState
        ] = {}
        with graph_capture(device=self.device):
            # Capture in order: PIECEWISE first, then FULL. PIECEWISE has larger
            # activations so FULL activations should fit in already allocated
            # buffers in the graph pool.
            for mode in [CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL]:
                if mode not in self._capture_descs:
                    continue

                descs = self._capture_descs[mode]
                if is_global_first_rank():
                    descs = tqdm(descs, desc=f"{progress_bar_desc} ({mode.name})")
                for desc in descs:
                    # 在图外部准备输入并获取前向函数
                    forward_fn, attn_state = create_forward_fn(desc)

                    # 预热运行
                    forward_fn(CUDAGraphMode.NONE)

                    # 捕获
                    logger.debug(
                        "CG Capture: mode=%s, batch_desc=%s", desc.cg_mode.name, desc
                    )
                    if desc.cg_mode == CUDAGraphMode.PIECEWISE:
                        captured_attn_states[desc] = attn_state
                        forward_fn(CUDAGraphMode.PIECEWISE)
                    else:
                        # 使用全新的注意力状态捕获。预热时的注意力状态被丢弃，
                        # 因为某些后端（如 FlashMLA）执行延迟初始化，
                        # 这些初始化必须被捕获到图中。
                        forward_fn, attn_state = create_forward_fn(desc)
                        captured_attn_states[desc] = attn_state
                        assert desc not in self.graphs, (
                            f"Graph already captured for {desc}"
                        )
                        graph = torch.cuda.CUDAGraph()
                        # 在捕获前同步 offloader 的拷贝流。
                        # 确保捕获前的所有预取操作已完成。
                        get_offloader().sync_prev_onload()
                        with torch.cuda.graph(graph, self.pool):
                            forward_fn(CUDAGraphMode.NONE)
                            # 前向传播后加入 offloader 的拷贝流以避免
                            # 未加入流的错误。最后一层的 start_prefetch
                            # 会分叉 copy_stream，但 wait_prefetch 只在
                            # 下一次前向传播时发生。
                            get_offloader().join_after_forward()
                        self.graphs[desc] = graph
                        compilation_counter.num_cudagraph_captured += 1
        self._graphs_captured = True
        return captured_attn_states

    def dispatch(
        self,
        num_reqs: int,
        num_tokens: int,
        uniform_token_count: int | None,
    ) -> BatchExecutionDescriptor:
        """根据批次参数查找匹配的 CUDA Graph 描述符。

        从优先级排序的候选列表中找到第一个兼容的描述符。
        如果没有匹配，返回 NONE 模式（eager 执行）。

        参数:
            num_reqs: 请求数量
            num_tokens: token 数量
            uniform_token_count: 统一的每请求 token 数

        返回:
            BatchExecutionDescriptor: 匹配的描述符
        """
        if self._graphs_captured and 0 < num_tokens < len(self._candidates):
            for desc in self._candidates[num_tokens]:
                if _is_compatible(desc, num_reqs, num_tokens, uniform_token_count):
                    return desc
        return BatchExecutionDescriptor(
            cg_mode=CUDAGraphMode.NONE, num_tokens=num_tokens, num_reqs=num_reqs
        )

    def run_fullgraph(self, desc: BatchExecutionDescriptor):
        """回放捕获的 FULL CUDA Graph。

        在回放前同步 offloader——当从 eager/piecewise 切换到 full cudagraph
        时（如 prefill -> decode），这是必需的。之前的 eager 迭代的
        start_prefetch 可能已在 copy_stream 上排队了 H2D 拷贝，
        而图的捕获事件无法感知这些拷贝。不执行此同步可能导致回放时
        覆盖仍在传输中的静态缓冲区。

        参数:
            desc: 批次执行描述符
        """
        assert desc.cg_mode == CUDAGraphMode.FULL, (
            f"Expected FULL mode, got {desc.cg_mode}"
        )
        assert desc in self.graphs, f"No cudagraph for {desc}"
        # Sync offloader before replay - needed when transitioning from
        # eager/piecewise to full cudagraph (e.g., prefill → decode).
        # The previous eager iteration's start_prefetch may have queued
        # H2D copies on copy_stream that the graph's captured events
        # cannot see. Without this, replay could overwrite static buffers
        # while those copies are still in flight.
        get_offloader().sync_prev_onload()
        self.graphs[desc].replay()


class ModelCudaGraphManager(CudaGraphManager):
    """模型专用的 CUDA Graph 管理器。

    在 CudaGraphManager 基础上增加了模型特定的功能：
    1. 管理隐藏状态（hidden states）的存储和切片
    2. 管理辅助隐藏状态（auxiliary hidden states）
    3. 管理中间张量（用于流水线并行的非最后阶段）

    属性:
        hidden_states: FULL CUDA Graph 使用的隐藏状态缓冲区
        aux_hidden_states: 辅助隐藏状态缓冲区列表
        use_aux_hidden_state_outputs: 是否使用辅助隐藏状态输出
        intermediate_tensors: 流水线并行非最后阶段的中间张量
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        cudagraph_mode: CUDAGraphMode,
        decode_query_len: int,
    ):
        super().__init__(vllm_config, device, cudagraph_mode, decode_query_len)
        # Used for FULL CUDA graphs. PW CUDA graphs do not use these.
        self.hidden_states: torch.Tensor | None = None
        self.aux_hidden_states: list[torch.Tensor] = []
        self.use_aux_hidden_state_outputs = False
        self.intermediate_tensors: IntermediateTensors | None = None

    def capture(
        self,
        model: nn.Module,
        model_state: ModelState,
        input_buffers: InputBuffers,
        intermediate_tensors: IntermediateTensors | None,
        block_tables: BlockTables,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        has_lora: bool = False,
        use_aux_hidden_state_outputs: bool = False,
        progress_bar_desc: str = "Capturing CUDA graphs",
    ) -> dict[BatchExecutionDescriptor, CapturedAttentionState]:
        """为模型前向传播捕获 CUDA Graph。

        参数:
            model: 模型实例
            model_state: 模型状态
            input_buffers: 输入缓冲区
            intermediate_tensors: 流水线并行的中间张量
            block_tables: 块表管理器
            attn_groups: 注意力组列表
            kv_cache_config: KV 缓存配置
            has_lora: 是否使用 LoRA
            use_aux_hidden_state_outputs: 是否使用辅助隐藏状态
            progress_bar_desc: 进度条描述文本

        返回:
            dict: 描述符 -> 捕获的注意力状态
        """
        self.use_aux_hidden_state_outputs = use_aux_hidden_state_outputs

        def create_forward_fn(
            desc: BatchExecutionDescriptor,
        ) -> tuple[
            Callable[[CUDAGraphMode], None],
            CapturedAttentionState,
        ]:
            num_tokens = desc.num_tokens
            num_reqs = desc.num_reqs or min(num_tokens, self.max_num_reqs)
            num_tokens_across_dp = (
                torch.full((self.dp_size,), num_tokens, dtype=torch.int32, device="cpu")
                if self.dp_size > 1
                else None
            )

            # 准备模型输入
            model_inputs = {
                "input_ids": input_buffers.input_ids[:num_tokens],
                "positions": input_buffers.positions[:num_tokens],
                **model_state.prepare_dummy_inputs(num_reqs, num_tokens),
            }
            if not self.is_first_pp_rank:
                # 非第一个 PP 排名不需要 input_ids
                model_inputs["input_ids"] = None
                model_inputs["inputs_embeds"] = None
                assert intermediate_tensors is not None
                model_inputs["intermediate_tensors"] = intermediate_tensors[:num_tokens]

            attn_metadata, slot_mappings = prepare_inputs_to_capture(
                num_reqs,
                num_tokens,
                model_state,
                input_buffers,
                block_tables,
                attn_groups,
                kv_cache_config,
                skip_attn=(desc.cg_mode == CUDAGraphMode.PIECEWISE),
            )

            def forward_fn(cg_mode: CUDAGraphMode) -> None:
                batch_descriptor = None
                if cg_mode == CUDAGraphMode.PIECEWISE:
                    assert attn_metadata is None
                    batch_descriptor = BatchDescriptor(
                        num_tokens=num_tokens, has_lora=has_lora
                    )
                with set_forward_context(
                    attn_metadata,
                    self.vllm_config,
                    num_tokens=num_tokens,
                    cudagraph_runtime_mode=cg_mode,
                    num_tokens_across_dp=num_tokens_across_dp,
                    slot_mapping=slot_mappings,
                    batch_descriptor=batch_descriptor,
                ):
                    model_output = model(**model_inputs)

                if cg_mode == CUDAGraphMode.PIECEWISE:
                    # PW CUDA graph 内部处理模型输出。
                    # 无需跟踪隐藏状态。
                    return None

                if self.is_last_pp_rank:
                    # 最后一个 PP 排名（常见情况）
                    if self.use_aux_hidden_state_outputs:
                        hidden_states, aux_hidden_states = model_output
                    else:
                        hidden_states = model_output
                        aux_hidden_states = []
                    if self.hidden_states is None:
                        self.hidden_states = torch.empty_like(hidden_states)
                    self.hidden_states[:num_tokens] = hidden_states
                    if self.use_aux_hidden_state_outputs and not self.aux_hidden_states:
                        self.aux_hidden_states = [
                            torch.empty_like(x) for x in aux_hidden_states
                        ]
                    for i, aux in enumerate(aux_hidden_states):
                        self.aux_hidden_states[i][:num_tokens] = aux
                else:
                    # 非最后的 PP 排名
                    assert isinstance(model_output, IntermediateTensors)
                    intermediate_tensors = model_output
                    if self.intermediate_tensors is None:
                        self.intermediate_tensors = IntermediateTensors.empty_like(
                            intermediate_tensors
                        )
                    for k, v in intermediate_tensors.tensors.items():
                        self.intermediate_tensors[k][:num_tokens] = v

            return forward_fn, CapturedAttentionState(attn_metadata, slot_mappings)

        return super().capture(create_forward_fn, progress_bar_desc)

    def run_fullgraph(
        self, desc: BatchExecutionDescriptor
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]] | IntermediateTensors:
        """回放捕获的 FULL CUDA Graph 并返回隐藏状态。

        参数:
            desc: 批次执行描述符

        返回:
            模型输出（隐藏状态、隐藏状态+辅助状态、或中间张量）
        """
        super().run_fullgraph(desc)
        if not self.is_last_pp_rank:
            assert self.intermediate_tensors is not None
            return self.intermediate_tensors[: desc.num_tokens]

        assert self.hidden_states is not None
        hidden_states = self.hidden_states[: desc.num_tokens]
        if not self.use_aux_hidden_state_outputs:
            return hidden_states
        return hidden_states, [x[: desc.num_tokens] for x in self.aux_hidden_states]


def prepare_inputs_to_capture(
    num_reqs: int,
    num_tokens: int,
    model_state: ModelState,
    input_buffers: InputBuffers,
    block_tables: BlockTables,
    attn_groups: list[list[AttentionGroup]],
    kv_cache_config: KVCacheConfig,
    skip_attn: bool = False,
) -> CapturedAttentionState:
    """为 CUDA Graph 捕获准备输入数据。

    创建虚拟的输入批次、块表和 slot 映射，用于图捕获时的形状和地址固定。

    参数:
        num_reqs: 请求数量
        num_tokens: token 数量
        model_state: 模型状态
        input_buffers: 输入缓冲区
        block_tables: 块表管理器
        attn_groups: 注意力组列表
        kv_cache_config: KV 缓存配置
        skip_attn: 是否跳过注意力元数据准备

    返回:
        CapturedAttentionState: 捕获的注意力状态
    """
    input_batch = InputBatch.make_dummy(num_reqs, num_tokens, input_buffers)
    input_block_tables = block_tables.get_dummy_block_tables(num_reqs)
    slot_mappings = block_tables.get_dummy_slot_mappings(num_tokens)
    slot_mappings_by_layer = build_slot_mappings_by_layer(
        slot_mappings, kv_cache_config
    )

    # HACK(woosuk): Special handling for DCP.
    if block_tables.cp_size > 1:
        prepare_dcp_local_seq_lens(
            input_buffers.dcp_local_seq_lens,
            input_batch.seq_lens,
            num_reqs,
            block_tables.cp_size,
            block_tables.cp_rank,
            block_tables.cp_interleave,
        )
        input_batch.dcp_local_seq_lens = input_buffers.dcp_local_seq_lens[:num_reqs]

    attn_metadata = None
    if not skip_attn:
        attn_metadata = model_state.prepare_attn(
            input_batch,
            CUDAGraphMode.NONE,
            input_block_tables,
            slot_mappings,
            attn_groups,
            kv_cache_config,
            for_capture=True,
        )
    return CapturedAttentionState(attn_metadata, slot_mappings_by_layer)
