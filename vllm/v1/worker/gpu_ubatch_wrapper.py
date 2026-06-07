# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# =============================================================================
# 模块概述: UBatch Wrapper (micro-batch 包装器)
# =============================================================================
# 本模块实现了一个将单个大 batch 拆分成多个 micro-batch (ubatch) 并行执行的包装层。
#
# 核心设计思想:
#   1. 当一个 forward pass 包含大量 token 时，将其拆分成多个较小的 ubatch，
#      每个 ubatch 在独立的 CUDA stream 和线程上并行执行模型前向传播。
#   2. 这种拆分使得计算 (compute) 和通信 (communication, 如 MoE all-to-all)
#      可以重叠执行，从而提高 GPU 利用率和整体吞吐量。
#   3. 同时支持 CUDA graph 捕获：对首次执行的特定 token 数量形状进行 graph 捕获，
#      后续相同形状的执行可以直接 replay 已捕获的 graph，避免 kernel launch 开销。
#
# 主要组件:
#   - UBatchWrapper: 核心包装类，管理 ubatch 拆分、并行执行、CUDA graph 捕获/重放
#   - SMControlContextManager: SM (Streaming Multiprocessor) 资源分配管理，
#     在计算和通信之间动态分配 GPU 计算单元
#   - UbatchMetadata / CUDAGraphMetaData: 元数据结构体
#
# 与上层的交互:
#   - GPUModelRunner 调用模型 forward 时，如果启用了 ubatching，会经过此包装层
#   - ForwardContext 中的 ubatch_slices 决定了如何拆分 batch
# =============================================================================

import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

import vllm.envs as envs
from vllm.compilation.cuda_graph import CUDAGraphWrapper
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.distributed import get_ep_group
from vllm.distributed.device_communicators.pynccl_allocator import set_graph_pool_id
from vllm.forward_context import (
    DPMetadata,
    create_forward_context,
    get_forward_context,
    override_forward_context,
)
from vllm.logger import init_logger
from vllm.model_executor.offloader.base import get_offloader
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.utils.deep_gemm import set_num_sms as deep_gemm_set_num_sms
from vllm.utils.import_utils import has_deep_gemm
from vllm.utils.platform_utils import num_compute_units
from vllm.v1.worker.ubatching import UBatchContext, make_ubatch_contexts

logger = init_logger(__name__)


def _cat_ubatch_outputs(
    sorted_results: list,
) -> "torch.Tensor | tuple[torch.Tensor, ...]":
    """Concatenate per-ubatch model outputs along the batch dim.

    Most models return a single hidden-states tensor per ubatch. Target
    models running with auxiliary output (e.g. EAGLE3 speculative decoding,
    which collects aux hidden states for the drafter) return a tuple of
    tensors instead. Fan out over tuple components so `torch.cat` sees
    matching shapes and the caller receives the same structure the model
    produced for a single ubatch (#40769).
    """
    # 中文注释: 将多个 ubatch 的模型输出沿 batch 维度拼接起来。
    # 大多数模型每个 ubatch 返回单个 hidden-states tensor，直接 torch.cat 即可。
    # 但某些模型（如 EAGLE3 推测解码）会返回 tuple 形式的辅助输出，
    # 此时需要对 tuple 中的每个分量分别进行拼接，保持输出结构一致。
    if sorted_results and isinstance(sorted_results[0], tuple):
        return tuple(torch.cat(parts, dim=0) for parts in zip(*sorted_results))
    return torch.cat(sorted_results, dim=0)


@dataclass
class UbatchMetadata:
    """中文注释: 单个 ubatch (micro-batch) 的元数据。

    每个 ubatch 包含:
    - context: ubatch 上下文，管理该 ubatch 的 CUDA stream、forward context、
      线程同步等信息
    - input_ids / positions / inputs_embeds: 模型输入的切片数据，
      由原始 batch 按 token_slice 切分而来
    - intermediate_tensors: 中间张量（如 pipeline parallel 场景下的中间激活），
      按 token_slice 切分
    - num_tokens: 该 ubatch 中包含的 token 数量
    """
    context: UBatchContext
    input_ids: torch.Tensor
    positions: torch.Tensor
    inputs_embeds: torch.Tensor | None
    intermediate_tensors: IntermediateTensors | None
    num_tokens: int


@dataclass
class CUDAGraphMetaData:
    """中文注释: CUDA graph 捕获后的元数据。

    用于缓存已捕获的 CUDA graph 及其关联信息:
    - cudagraph: 已捕获的 CUDA graph 对象，后续可通过 replay() 快速重放
    - ubatch_metadata: 捕获时使用的 ubatch 元数据
    - outputs: 捕获时产生的输出 tensor 引用（CUDA graph 重放时直接使用此引用）
    """
    cudagraph: torch.cuda.CUDAGraph
    ubatch_metadata: UbatchMetadata
    outputs: Any | None = None


class SMControlContextManager:
    """中文注释: SM (Streaming Multiprocessor) 资源分配上下文管理器。

    背景: 在使用专家并行 (Expert Parallelism) 的 MoE 模型中，GPU 需要同时执行:
      - 计算 (compute): 矩阵乘法、注意力计算等
      - 通信 (communication): MoE 的 all-to-all 专家分发通信
    两者都使用 GPU 的 SM 资源。如果不加控制，通信 kernel 可能占用过多 SM，
    导致计算 kernel 无法充分利用 GPU。

    设计思路:
      - 进入上下文时: 将 SM 分为两部分，comm_sms 个 SM 用于通信，
        total_sms - comm_sms 个 SM 用于计算
      - 退出上下文时: 恢复所有 SM 用于计算或通信（不加限制）
    这样可以确保 ubatch 执行期间计算和通信都能获得合理的 SM 资源。
    """
    def __init__(
        self,
        comm_sms: int,
        set_comm_sms: Callable[[int], None],
        set_compute_sms: Callable[[int], None],
    ):
        """
        Context manager for controlling SM (Streaming Multiprocessor)
        allocation. Upon entering the context, it sets the number of SMs
        allocated for communication and computation to comm_sms and
        total_sms - comm_sms respectively. Upon exiting, it restores the
        allocation to use all available SMs (i.e. total_sms).

        Args:
            comm_sms (int): The number of SMs to allocate for communication.
                (The remainder will be used for computation.)
            set_comm_sms (Callable[[int], None]):
                A function that sets the number of SMs for communication.
            set_compute_sms (Callable[[int], None]):
                A function that sets the number of SMs for computation.
        """

        assert current_platform.is_cuda() or current_platform.is_rocm(), (
            "SM/CU control is supported on CUDA and ROCm platforms"
        )
        device = torch.accelerator.current_device_index()
        # 中文注释: 获取当前 GPU 设备的总 SM 数量
        total_sms = num_compute_units(device)

        assert comm_sms < total_sms
        self.total_sms = total_sms
        # 中文注释: 计算 SM = 总 SM - 通信 SM，确保计算有足够的资源
        self.compute_sms = total_sms - comm_sms
        self.comm_sms = comm_sms
        self.set_comm_sms = set_comm_sms
        self.set_compute_sms = set_compute_sms

    def __enter__(self):
        # 中文注释: 进入上下文时，按比例分配 SM 给通信和计算
        self.set_comm_sms(self.comm_sms)
        self.set_compute_sms(self.compute_sms)

    def __exit__(self, exc_type, exc_value, traceback):
        # 中文注释: 退出上下文时，恢复所有 SM 可用于任意用途
        self.set_comm_sms(self.total_sms)
        self.set_compute_sms(self.total_sms)


class UBatchWrapper:
    """中文注释: UBatch (micro-batch) 包装器，核心类。

    职责:
    1. 将一个大的 forward batch 拆分成多个较小的 ubatch 并行执行
    2. 管理 ubatch 的 CUDA stream、线程同步和 forward context
    3. 支持 CUDA graph 捕获和重放，提升重复执行效率
    4. 集成 SM 控制，协调计算和通信的 GPU 资源分配

    工作原理:
    - 当 ForwardContext 中存在 ubatch_slices 时，说明需要拆分执行
    - 每个 ubatch 在独立线程 + 独立 CUDA stream 上运行模型 forward
    - 通过 threading.Barrier 同步所有 ubatch 线程的就绪状态
    - 所有 ubatch 完成后，在主线程中拼接输出

    CUDA graph 支持:
    - 首次遇到某个 num_tokens 形状时，捕获 CUDA graph 并缓存
    - 后续相同形状的执行直接 replay 已捕获的 graph
    - graph 捕获在 SM 控制上下文中进行，确保计算/通信 SM 比例正确
    """
    def __init__(
        self,
        runnable: Callable,
        vllm_config: VllmConfig,
        runtime_mode: CUDAGraphMode,
        device: torch.cuda.device,
    ):
        # 中文注释: runnable 是实际执行模型 forward 的可调用对象（通常是 ModelRunner 的方法）
        self.runnable = runnable
        self.vllm_config = vllm_config
        self.compilation_config = vllm_config.compilation_config
        # 中文注释: 专用的通信 CUDA stream，用于 ubatch 间的通信操作（如 MoE all-to-all）
        self.comm_stream = torch.cuda.Stream(device=device)
        # 中文注释: 线程同步屏障，确保所有 ubatch 线程都初始化好 CUDA context 后再开始执行。
        # num_ubatches 个 ubatch 线程 + 主线程 = num_ubatches + 1
        self.ready_barrier = threading.Barrier(
            self.vllm_config.parallel_config.num_ubatches + 1
        )

        # 中文注释: 缓存已捕获的 CUDA graph，key 为 num_tokens（batch 中的总 token 数），
        # value 为 CUDA graph 及其元数据。相同 num_tokens 形状的后续执行可以直接 replay。
        self.cudagraphs: dict[int, CUDAGraphMetaData] = {}

        # 中文注释: 如果启用了 CUDA graph 模式（FULL 或 PIECEWISE），创建 CUDAGraphWrapper
        # 用于处理非 ubatch 路径（即不需要拆分时）的 CUDA graph 捕获/重放
        self.cudagraph_wrapper = None
        if runtime_mode is not CUDAGraphMode.NONE:
            self.cudagraph_wrapper = CUDAGraphWrapper(
                runnable, vllm_config, runtime_mode=runtime_mode
            )

        # 中文注释: 创建 SM 控制上下文管理器，用于在 ubatch 执行期间动态分配计算/通信 SM 资源
        self.sm_control = self._create_sm_control_context(vllm_config)
        self.device = device
        self.is_debugging_mode = envs.VLLM_LOGGING_LEVEL == "DEBUG"
        self._runnable_str = str(runnable) if self.is_debugging_mode else None

    @property
    def graph_pool(self):
        # 中文注释: 获取 CUDA graph 的内存池。
        # 如果存在 cudagraph_wrapper 则使用其 graph pool，否则使用平台默认的。
        # graph pool 是 CUDA 提供的专用内存区域，graph 捕获时的内存分配都在此池中进行，
        # 使得 replay 时可以快速复用相同的内存地址。
        if self.cudagraph_wrapper is not None:
            return self.cudagraph_wrapper.graph_pool
        return None

    def clear_graphs(self) -> None:
        # 中文注释: 清除所有已缓存的 CUDA graph。
        # 当模型权重更新或配置变化时需要调用此方法，因为旧的 graph 不再有效。
        self.cudagraphs.clear()
        if self.cudagraph_wrapper is not None:
            self.cudagraph_wrapper.clear_graphs()

    @staticmethod
    def _create_sm_control_context(vllm_config: VllmConfig):
        """中文注释: 创建 SM 控制上下文管理器。

        流程:
        1. 从环境变量读取通信 SM 数量 (VLLM_DBO_COMM_SMS)
        2. 如果启用了专家并行 (EP)，获取 DeepEP all2all 管理器，
           并将其最大 SM 使用数作为通信 SM 的上限
        3. 创建回调函数：通信 SM 数量由 all2all_manager 控制，
           计算 SM 数量由 DeepGEMM 控制
        4. 返回配置好的 SMControlContextManager 实例

        为什么需要 SM 控制:
        - MoE 模型的 all-to-all 通信和计算共享 GPU SM 资源
        - 如果不限制，通信 kernel 可能占用过多 SM，导致计算 kernel 延迟
        - 通过 SM 分配，可以让计算和通信各自获得稳定、合理的 SM 资源份额
        """
        comm_sms: int = envs.VLLM_DBO_COMM_SMS

        set_comm_sms = lambda sms: None
        if vllm_config.parallel_config.enable_expert_parallel:
            # Currently only DeepEP highthroughput supports SM control so this
            # only affects that case.
            ep_group = get_ep_group()
            device_communicator = ep_group.device_communicator
            all2all_manager = None
            if device_communicator is not None:
                all2all_manager = device_communicator.all2all_manager

            if all2all_manager is not None:
                max_sms_used = all2all_manager.max_sms_used()
                if max_sms_used is not None:
                    comm_sms = min(comm_sms, max_sms_used)

            if comm_sms > 0 and all2all_manager is not None:
                set_comm_sms = lambda sms: all2all_manager.set_num_sms(sms)

        # TODO(lucas): support other kernels besides DeepGEMM
        set_compute_sms = lambda sms: None
        if has_deep_gemm() and comm_sms > 0:
            set_compute_sms = lambda sms: deep_gemm_set_num_sms(sms)

        return SMControlContextManager(
            comm_sms=comm_sms,
            set_comm_sms=set_comm_sms,
            set_compute_sms=set_compute_sms,
        )

    def __getattr__(self, key: str):
        # allow accessing the attributes of the runnable.
        if hasattr(self.runnable, key):
            return getattr(self.runnable, key)
        if self.is_debugging_mode:
            raise AttributeError(
                f"Attribute {key} not exists in the runnable of "
                f"cudagraph wrapper: {self._runnable_str}"
            )
        raise AttributeError

    def unwrap(self) -> Callable:
        # in case we need to access the original runnable.
        return self.runnable

    def _capture_ubatches(self, ubatch_metadata, model) -> torch.Tensor:
        """
        Capture a cudagraph for a microbatched run.

        The logic here is somewhat complicated because we need to make sure that
        each of the ubatch threads initialize the cuda context before we start
        the graph capture.

        The flow is as follows:
        1. The main thread starts up each ubatch thread. Each thread will
        initialize its cuda context (torch.cuda.current_blas_handle())
        before going to sleep upon entering the ubatch_context.

        2. The main thread starts the graph capture and wakes up the first
        ubatch thread.

        3. Each ubatch thread runs the model to completion and returns the
        completed output tensors back to the main thread.

        4. The main thread stores the captured cudagraph along with its metadata
        and returns
        """

        @torch.inference_mode()
        def _capture_ubatch_thread(results, ubatch_metadata):
            torch.accelerator.set_device_index(self.device)
            ubatch_context = ubatch_metadata.context
            with torch.cuda.stream(ubatch_context.compute_stream):
                _ = torch.cuda.current_blas_handle()
            with torch.cuda.stream(ubatch_context.comm_stream):
                _ = torch.cuda.current_blas_handle()
            with ubatch_context:
                model_output = model(
                    input_ids=ubatch_metadata.input_ids,
                    positions=ubatch_metadata.positions,
                    intermediate_tensors=ubatch_metadata.intermediate_tensors,
                    inputs_embeds=ubatch_metadata.inputs_embeds,
                )

            results.append((ubatch_metadata.context.id, model_output))

        results: list[tuple[int, torch.Tensor]] = []
        compute_stream = ubatch_metadata[0].context.compute_stream
        num_tokens = ubatch_metadata[0].num_tokens + ubatch_metadata[1].num_tokens

        # Ubatches will manually manage the forward context, so we override
        # it to None here so we can have it restored correctly later
        with override_forward_context(None):
            ubatch_threads = []
            for metadata in ubatch_metadata:
                thread = threading.Thread(
                    target=_capture_ubatch_thread,
                    args=(
                        results,
                        metadata,
                    ),
                )
                ubatch_threads.append(thread)
                thread.start()
            self.ready_barrier.wait()  # Wait for both threads to be ready

            # Capture the cudagraph
            cudagraph_metadata = CUDAGraphMetaData(
                cudagraph=torch.cuda.CUDAGraph(),
                ubatch_metadata=ubatch_metadata,
            )
            if self.graph_pool is not None:
                set_graph_pool_id(self.graph_pool)
            else:
                set_graph_pool_id(current_platform.graph_pool_handle())

            # Sync offloader's copy stream before capture.
            # Ensure any pre-capture prefetches from offloader are complete.
            get_offloader().sync_prev_onload()

            with torch.cuda.graph(
                cudagraph_metadata.cudagraph,
                stream=compute_stream,
                pool=self.graph_pool,
            ):
                ubatch_metadata[0].context.cpu_wait_event.set()
                for thread in ubatch_threads:
                    thread.join()
                sorted_results = [value for position, value in sorted(results)]
                result = _cat_ubatch_outputs(sorted_results)
                cudagraph_metadata.outputs = result
                # Join offloader's copy stream after forward to avoid unjoined
                # stream error. The last layer's start_prefetch forks copy_stream,
                # but wait_prefetch only happens in the next forward pass.
                get_offloader().join_after_forward()
            self.cudagraphs[num_tokens] = cudagraph_metadata
        return cudagraph_metadata.outputs

    def _run_ubatches(self, ubatch_metadata, model) -> torch.Tensor:
        @torch.inference_mode()
        def _ubatch_thread(results, model, ubatch_metadata):
            with ubatch_metadata.context:
                model_output = model(
                    input_ids=ubatch_metadata.input_ids,
                    positions=ubatch_metadata.positions,
                    intermediate_tensors=ubatch_metadata.intermediate_tensors,
                    inputs_embeds=ubatch_metadata.inputs_embeds,
                )
            results.append((ubatch_metadata.context.id, model_output))

        results: list[tuple[int, torch.Tensor]] = []

        # Ubatch threads will manually manage the forward context, so we
        # override it to None here so we can have it restored correctly
        # after both threads have finished
        with override_forward_context(None):
            ubatch_threads = []
            for metadata in ubatch_metadata:
                thread = threading.Thread(
                    target=_ubatch_thread,
                    args=(
                        results,
                        model,
                        metadata,
                    ),
                )
                ubatch_threads.append(thread)
                thread.start()
            self.ready_barrier.wait()  # Wait for both threads to be ready
            ubatch_metadata[0].context.cpu_wait_event.set()
            for thread in ubatch_threads:
                thread.join()
        sorted_results = [value for position, value in sorted(results)]
        result = _cat_ubatch_outputs(sorted_results)
        return result

    def _make_ubatch_metadata(
        self,
        ubatch_slices,
        attn_metadata,
        slot_mapping,
        input_ids,
        positions,
        inputs_embeds,
        intermediate_tensors,
        compute_stream,
        dp_metadata,
        batch_descriptor,
        cudagraph_runtime_mode,
    ) -> list[UbatchMetadata]:
        # Create one forward context per ubatch
        forward_contexts = []
        # slot_mapping can be None, an empty dict (from create_forward_context
        # converting None to {}), or a list of dicts (one per ubatch)
        has_slot_mapping = slot_mapping and isinstance(slot_mapping, list)
        for i, ubatch_slice in enumerate(ubatch_slices):
            forward_contexts.append(
                create_forward_context(
                    attn_metadata[i] if attn_metadata is not None else None,
                    self.vllm_config,
                    dp_metadata=dp_metadata[i],
                    batch_descriptor=batch_descriptor,
                    cudagraph_runtime_mode=cudagraph_runtime_mode,
                    slot_mapping=slot_mapping[i] if has_slot_mapping else None,
                )
            )

        ubatch_ctxs = make_ubatch_contexts(
            num_micro_batches=len(ubatch_slices),
            comm_stream=self.comm_stream,
            compute_stream=compute_stream,
            forward_contexts=forward_contexts,
            ready_barrier=self.ready_barrier,
        )

        ubatch_metadata: list[UbatchMetadata] = []
        for i, ubatch_slice in enumerate(ubatch_slices):
            (
                sliced_input_ids,
                sliced_positions,
                sliced_inputs_embeds,
                sliced_intermediate_tensors,
            ) = self._slice_model_inputs(
                ubatch_slice.token_slice,
                input_ids,
                positions,
                inputs_embeds,
                intermediate_tensors,
            )
            ubatch_metadata.append(
                UbatchMetadata(
                    context=ubatch_ctxs[i],
                    input_ids=sliced_input_ids,
                    positions=sliced_positions,
                    inputs_embeds=sliced_inputs_embeds,
                    intermediate_tensors=sliced_intermediate_tensors,
                    num_tokens=ubatch_slice.token_slice.stop
                    - ubatch_slice.token_slice.start,
                )
            )

        return ubatch_metadata

    def _slice_model_inputs(
        self,
        tokens_slice: slice,
        input_ids,
        positions,
        inputs_embeds,
        intermediate_tensors,
    ):
        sliced_input_ids = input_ids[tokens_slice] if input_ids is not None else None
        # if we are using mrope. Mrope adds an additional dimension to the
        # positions tensor
        if positions.ndim == 2:
            sliced_positions = positions[:, tokens_slice]
        else:
            sliced_positions = positions[tokens_slice]
        sliced_inputs_embeds = (
            inputs_embeds[tokens_slice] if inputs_embeds is not None else None
        )
        sliced_intermediate_tensors = (
            intermediate_tensors[tokens_slice]
            if intermediate_tensors is not None
            else None
        )

        return (
            sliced_input_ids,
            sliced_positions,
            sliced_inputs_embeds,
            sliced_intermediate_tensors,
        )

    def __call__(self, *args, **kwargs):
        forward_context = get_forward_context()
        batch_descriptor = forward_context.batch_descriptor
        ubatch_slices = forward_context.ubatch_slices
        cudagraph_runtime_mode = forward_context.cudagraph_runtime_mode

        # If there's no ubatching, just run the runnable object
        if ubatch_slices is None:
            # This is to account for the case where ubatching was aborted.
            # When we capture full graphs we only capture one graph per shape,
            # meaning that if we have a ubatched  cudagraph for the current
            # num_tokens, we don't have a non-ubatched one. Without this
            # check, the cudagraph wrapper will try to capture a cudagraph
            # for this shape during a normal run.
            if cudagraph_runtime_mode is CUDAGraphMode.FULL:
                assert batch_descriptor is not None
                if batch_descriptor.num_tokens in self.cudagraphs:
                    cudagraph_runtime_mode = CUDAGraphMode.NONE

            if cudagraph_runtime_mode in (CUDAGraphMode.NONE, CUDAGraphMode.PIECEWISE):
                return self.runnable(*args, **kwargs)
            else:
                assert self.cudagraph_wrapper is not None
                return self.cudagraph_wrapper(*args, **kwargs)

        attn_metadata = forward_context.attn_metadata
        slot_mapping = forward_context.slot_mapping
        num_tokens = sum(ubatch_slice.num_tokens for ubatch_slice in ubatch_slices)
        input_ids = kwargs["input_ids"]
        positions = kwargs["positions"]
        intermediate_tensors = kwargs["intermediate_tensors"]
        inputs_embeds = kwargs["inputs_embeds"]
        compute_stream = torch.cuda.current_stream()

        dp_metadata = forward_context.dp_metadata

        # We shouldn't be here unless we are running with multiple DP ranks
        assert dp_metadata is not None
        ubatch_dp_metadata = []
        for ubatch_slice in ubatch_slices:
            dp_size = self.vllm_config.parallel_config.data_parallel_size
            ubatch_num_tokens_across_dp = torch.tensor(
                [ubatch_slice.num_tokens] * dp_size, device="cpu", dtype=torch.int32
            )
            ubatch_dp_metadata.append(
                DPMetadata.make(
                    self.vllm_config.parallel_config,
                    ubatch_slice.num_tokens,
                    ubatch_num_tokens_across_dp,
                )
            )

        if (
            num_tokens not in self.cudagraphs
            and cudagraph_runtime_mode is CUDAGraphMode.FULL
        ):
            ubatch_metadata = self._make_ubatch_metadata(
                ubatch_slices=ubatch_slices,
                attn_metadata=attn_metadata,
                slot_mapping=slot_mapping,
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                compute_stream=compute_stream,
                dp_metadata=ubatch_dp_metadata,
                batch_descriptor=batch_descriptor,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
            )
            with self.sm_control:
                return self._capture_ubatches(ubatch_metadata, self.runnable)
        elif (
            num_tokens in self.cudagraphs
            and cudagraph_runtime_mode is CUDAGraphMode.FULL
        ):
            cudagraph_metadata = self.cudagraphs[num_tokens]
            # Sync offloader before replay - ensures any external dependencies
            # from pre-capture prefetches are satisfied.
            get_offloader().sync_prev_onload()
            cudagraph_metadata.cudagraph.replay()
            return cudagraph_metadata.outputs
        else:
            ubatch_metadata = self._make_ubatch_metadata(
                ubatch_slices=ubatch_slices,
                attn_metadata=attn_metadata,
                slot_mapping=slot_mapping,
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                compute_stream=compute_stream,
                dp_metadata=ubatch_dp_metadata,
                batch_descriptor=batch_descriptor,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
            )
            with self.sm_control:
                return self._run_ubatches(ubatch_metadata, self.runnable)
