# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
CPU 模型运行器模块

=============================================================
【模块概述】
=============================================================
本模块实现了 CPU 后端的模型运行器（CPUModelRunner），它继承自
GPUModelRunner 并将所有 GPU 特有的操作替换为 CPU 兼容的实现。

=============================================================
【设计思路】
=============================================================
CPUModelRunner 的核心策略是"继承 + 替换"：
1. 继承 GPUModelRunner 的完整推理逻辑（调度、输入处理、采样等）
2. 在初始化时将所有 GPU 张量替换为 CPU 张量（_postprocess_tensors）
3. 将所有 CUDA Triton kernel 替换为 CPU Triton kernel（_postprocess_triton）
4. 重写所有使用 CUDA Stream/Event 的方法为 CPU 安全版本

这种设计避免了大量代码重复，同时确保 CPU 后端的功能完整性。

=============================================================
【关键技术细节】
=============================================================
1. _torch_cuda_wrapper：上下文管理器，在初始化期间临时替换 torch.cuda.Stream
   和 torch.Event 为空操作占位符，防止 GPUModelRunner.__init__() 中的
   CUDA 初始化代码在 CPU 环境下崩溃。

2. _set_torch_accelerator_to_noop：将 torch.accelerator.synchronize 和
   torch.accelerator.empty_cache 替换为空操作，因为 CPU 没有独立的加速器。

3. _postprocess_tensors：遍历所有属性，将 GPU 张量替换为对应的 CPU 张量。
   例如将 temperature（GPU）替换为 temperature_cpu_tensor（CPU）。

4. _postprocess_triton：将 GPU Triton kernel 函数引用替换为 CPU 版本，
   包括 block table 操作、投机解码相关的 kernel 等。

=============================================================
【CPU 后端的特殊处理】
=============================================================
1. 不支持 CUDA Graph：use_cuda_graph = False
2. 不支持级联注意力：cascade_attn_enabled = False
3. 不需要设备属性初始化：_init_device_properties 为空操作
4. 不需要设备同步：_sync_device 为空操作
5. 不需要清零 block IDs：CPU 注意力对无效位置分配 -INF
6. 不支持 dummy 权重加载（弹性 EP 扩展）
"""

from contextlib import contextmanager
from typing import Any

import torch
import torch.nn as nn

import vllm.utils.cpu_triton_utils as cpu_tl
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model
from vllm.tracing import instrument
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.utils import CpuGpuBuffer
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

logger = init_logger(__name__)


class CPUModelRunner(GPUModelRunner):
    """
    CPU 模型运行器。

    继承自 GPUModelRunner，将所有 GPU 特有操作替换为 CPU 兼容实现。
    支持完整的推理功能，包括 LoRA、投机解码（EAGLE）等。
    """

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        """
        初始化 CPU 模型运行器。

        ==========================================================
        【初始化流程】
        ==========================================================
        1. 禁用 torch.accelerator 的 GPU 操作（设为空操作）
        2. 在 CUDA 包装器下调用父类初始化（防止 CUDA API 调用崩溃）
        3. 验证设备为 CPU
        4. 禁用 CUDA Graph 和级联注意力
        5. 后处理：将 GPU 张量替换为 CPU 张量
        6. 后处理：将 GPU Triton kernel 替换为 CPU 版本

        参数：
            vllm_config: vLLM 全局配置
            device: 目标设备（必须为 torch.device("cpu")）
        """
        # 避免继承的方法调用加速器 API（如 CUDA 同步、缓存清理等）
        # avoid calling accelerator APIs for methods inherited from super class
        _set_torch_accelerator_to_noop()

        # 在 CUDA 包装器下初始化父类
        # 包装器会临时替换 torch.cuda.Stream 和 torch.Event 为占位符
        with _torch_cuda_wrapper():
            super().__init__(vllm_config, device)

        assert device == torch.device("cpu")
        # Note: speculative decoding is now supported on CPU with C++ native impls

        # CPU 后端不支持 CUDA Graph 和级联注意力
        self.use_cuda_graph = False
        self.cascade_attn_enabled = False

        # 后处理：替换张量和 Triton kernel
        self._postprocess_tensors()
        self._postprocess_triton()

    def _postprocess_tensors(self) -> None:
        """
        将 GPU 张量替换为 CPU 张量。

        ==========================================================
        【替换策略】
        ==========================================================
        1. 遍历 self 的所有属性，找到 CpuGpuBuffer 实例
           将其 .gpu 属性设为 .cpu 的引用（同一块内存）
        2. 遍历 input_batch 的所有属性，找到以 _cpu_tensor 结尾的张量
           将对应的设备张量（去掉 _cpu_tensor 后缀）替换为 CPU 张量
        3. 遍历 block_table 中的 CpuGpuBuffer 实例，同样替换

        这样所有后续代码引用 self.temperature 等属性时，
        实际使用的是 CPU 上的张量。
        """
        # Note: replace device tensors with cpu tensors
        def replace_tensor(obj: Any, cpu_attr_name: str, device_attr_name) -> None:
            """将对象的设备张量属性替换为对应的 CPU 张量。"""
            cpu_tensor = getattr(obj, cpu_attr_name, None)
            device_tensor = getattr(obj, device_attr_name, None)
            if isinstance(cpu_tensor, torch.Tensor) and isinstance(
                device_tensor, torch.Tensor
            ):
                setattr(obj, device_attr_name, cpu_tensor)

        # 替换 self 上的 CpuGpuBuffer
        for v in vars(self).values():
            if isinstance(v, CpuGpuBuffer):
                v.gpu = v.cpu

        # 替换 input_batch 上的 CPU/设备张量对
        # 例如：将 input_batch.temperature（原本是 GPU 张量）
        # 替换为 input_batch.temperature_cpu_tensor（CPU 张量）
        for k, v in vars(self.input_batch).items():
            if k.endswith("_cpu_tensor") and isinstance(v, torch.Tensor):
                replace_tensor(self.input_batch, k, k[:-11])

        # 替换 block_table 中的 CpuGpuBuffer
        for block_table in self.input_batch.block_table.block_tables:
            for v in vars(block_table).values():
                if isinstance(v, CpuGpuBuffer):
                    v.gpu = v.cpu

    def _postprocess_triton(self) -> None:
        """
        将 GPU Triton kernel 替换为 CPU 兼容版本。

        ==========================================================
        【替换的 kernel 列表】
        ==========================================================
        1. block_table 中的 compute_slot_mapping_kernel
           - 计算 KV cache 的 slot 映射
        2. 投机解码（EAGLE）相关的 kernel：
           - eagle_prepare_inputs_padded_kernel：准备 EAGLE 输入
           - eagle_prepare_next_token_padded_kernel：准备下一个 token
           - copy_and_expand_eagle_inputs_kernel：复制和扩展输入
           - eagle_step_slot_mapping_metadata_kernel：步骤 slot 映射
        3. 拒绝采样（Rejection Sampling）相关的 kernel：
           - rejection_greedy_sample_kernel：贪心采样
           - rejection_random_sample_kernel：随机采样
           - expand_kernel：张量扩展
           - sample_recovered_tokens_kernel：恢复 token 采样
        """
        import vllm.v1.worker.block_table

        vllm.v1.worker.block_table._compute_slot_mapping_kernel = (
            cpu_tl.compute_slot_mapping_kernel
        )

        # Speculative decoding fallbacks
        import vllm.v1.sample.rejection_sampler
        import vllm.v1.spec_decode.llm_base_proposer
        import vllm.v1.spec_decode.utils

        vllm.v1.spec_decode.llm_base_proposer.eagle_prepare_inputs_padded_kernel = (
            cpu_tl.eagle_prepare_inputs_padded_kernel
        )
        vllm.v1.spec_decode.llm_base_proposer.eagle_prepare_next_token_padded_kernel = (
            cpu_tl.eagle_prepare_next_token_padded_kernel
        )
        vllm.v1.spec_decode.llm_base_proposer.copy_and_expand_eagle_inputs_kernel = (
            cpu_tl.copy_and_expand_eagle_inputs_kernel
        )
        vllm.v1.spec_decode.utils.eagle_step_slot_mapping_metadata_kernel = (
            cpu_tl.eagle_step_slot_mapping_metadata_kernel
        )
        vllm.v1.sample.rejection_sampler.rejection_greedy_sample_kernel = (
            cpu_tl.rejection_greedy_sample_kernel
        )
        vllm.v1.sample.rejection_sampler.rejection_random_sample_kernel = (
            cpu_tl.rejection_random_sample_kernel
        )
        vllm.v1.sample.rejection_sampler.expand_kernel = cpu_tl.expand_kernel
        vllm.v1.sample.rejection_sampler.sample_recovered_tokens_kernel = (
            cpu_tl.sample_recovered_tokens_kernel
        )

    @instrument(span_name="Loading (CPU)")
    def load_model(self, load_dummy_weights: bool = False) -> None:
        """
        加载模型权重到 CPU 内存。

        ==========================================================
        【加载流程】
        ==========================================================
        1. 如果请求加载 dummy 权重（用于弹性 EP 扩展），抛出异常
           （CPU 后端不支持此功能）
        2. 使用 get_model() 加载模型
        3. 如果配置了 LoRA，加载 LoRA 适配器
        4. 如果配置了投机解码（drafter），加载 drafter 模型
        5. 设置 EAGLE3 辅助隐藏状态输出

        参数：
            load_dummy_weights: 是否加载 dummy 权重（CPU 不支持）
        """
        if load_dummy_weights:
            raise ValueError(
                "Loading dummy weights (needed for elastic EP scale-up) "
                "Is not supported by the CPU Model Runner."
            )
        logger.info("Starting to load model %s...", self.model_config.model)
        self.model = get_model(vllm_config=self.vllm_config)

        if self.lora_config:
            self.model = self.load_lora_model(self.model, self.vllm_config, self.device)

        if hasattr(self, "drafter"):
            logger.info_once("Loading drafter model...")
            self.drafter.load_model(self.model)

        self._setup_eagle3_aux_hidden_state_outputs()

    def get_model(self) -> nn.Module:
        """返回已加载的模型实例。"""
        return self.model

    @instrument(span_name="Warmup (CPU)")
    def warming_up_model(self) -> None:
        """
        预热模型，触发编译（torch.compile 等）。

        执行一次 profile_run() 以触发延迟编译，
        确保后续推理不会因为首次编译而卡顿。
        """
        logger.info("Warming up model for the compilation...")
        # Only generate graph for the generic shape
        with _set_global_compilation_settings(self.vllm_config):
            self.profile_run()
        logger.info("Warming up done.")

    def initialize_kv_cache(
        self,
        kv_cache_config: KVCacheConfig,
        is_profiling: bool = False,
    ) -> None:
        """
        初始化 KV Cache。

        调用父类的初始化方法，然后为投机解码输出额外的日志信息。

        参数：
            kv_cache_config: KV Cache 配置
            is_profiling: 是否处于性能分析模式
        """
        super().initialize_kv_cache(kv_cache_config, is_profiling)

        if self.speculative_config:
            if self.speculative_config.use_eagle():
                logger.info("EAGLE drafter KV cache initialized for CPU backend")
            elif self.speculative_config.uses_draft_model():
                logger.info("Draft model KV cache initialized for CPU backend")

    def _init_device_properties(self) -> None:
        """CPU 后端不需要初始化设备属性（如 GPU 内存大小等）。"""
        pass

    def _sync_device(self) -> None:
        """CPU 后端不需要设备同步（CPU 操作是同步的）。"""
        pass

    def _zero_block_ids(self, block_ids: list[int]) -> None:
        """
        CPU 后端不需要清零 block IDs。

        原因：CPU 注意力实现对无效位置分配 -INF 值，
        因此旧的 KV Cache 数据不会影响计算结果。
        """
        # CPU attention assigns -INF to logits at invalid positions,
        # so stale KV cache data never affects computation.
        pass

    # =========================================================================
    # CPU-safe overrides for speculative decoding methods
    # These methods override GPU-specific implementations that use CUDA streams
    # =========================================================================
    # 以下方法重写了 GPU 版本中使用 CUDA Stream/Event 的实现，
    # 提供 CPU 安全的版本（不需要异步复制和事件同步）

    def _copy_draft_token_ids_to_cpu(
        self, scheduler_output: "SchedulerOutput", zeros_only: bool = False
    ) -> None:
        """CPU-safe version: no async copy needed, tensors already on CPU."""
        """
        CPU 安全版本：将 draft token ID 复制到 CPU 缓冲区。

        GPU 版本使用 CUDA Stream 异步复制，CPU 版本直接同步复制。

        参数：
            scheduler_output: 调度器输出
            zeros_only: 如果为 True，只清零不复制实际数据
        """
        if self.use_async_scheduling and not (
            scheduler_output.has_structured_output_requests
            or self.input_batch.sampling_metadata.output_token_ids
        ):
            return
        self._draft_token_req_ids = self.input_batch.req_ids.copy()

        draft_token_ids: torch.Tensor = self._draft_token_ids
        if not torch.is_tensor(draft_token_ids):
            return

        num_reqs = draft_token_ids.shape[0]
        if self.draft_token_ids_cpu is not None:
            if not zeros_only:
                self.draft_token_ids_cpu[:num_reqs].copy_(draft_token_ids)
            else:
                self.draft_token_ids_cpu[:num_reqs] = 0

    def _get_draft_token_ids_cpu(self) -> tuple[list[list[int]], list[str]]:
        """CPU-safe version: no event synchronization needed."""
        """
        CPU 安全版本：获取 draft token ID。

        GPU 版本需要等待 CUDA Event 同步，CPU 版本直接读取。

        返回：
            (draft_token_ids, req_ids) 元组
        """
        if isinstance(self._draft_token_ids, list):
            return self._draft_token_ids, self.input_batch.req_ids
        req_ids = self._draft_token_req_ids
        if req_ids is None:
            return [], []
        if self.draft_token_ids_cpu is not None:
            return self.draft_token_ids_cpu[: len(req_ids)].tolist(), req_ids
        return [], []

    def _copy_valid_sampled_token_count(
        self, next_token_ids: torch.Tensor, valid_sampled_tokens_count: torch.Tensor
    ) -> None:
        """CPU-safe version: direct copy without CUDA streams."""
        """
        CPU 安全版本：复制有效采样 token 计数。

        GPU 版本使用 CUDA Stream 异步复制，CPU 版本直接同步复制。

        参数：
            next_token_ids: 下一个 token ID 张量
            valid_sampled_tokens_count: 有效采样 token 计数张量
        """
        if self.valid_sampled_token_count_cpu is None:
            return

        counts = valid_sampled_tokens_count
        counts_cpu = self.valid_sampled_token_count_cpu
        counts_cpu[: counts.shape[0]].copy_(counts)
        self.input_batch.prev_sampled_token_ids = next_token_ids.unsqueeze(1)

    def _get_valid_sampled_token_count(self) -> list[int]:
        """CPU-safe version: no event synchronization needed."""
        """
        CPU 安全版本：获取有效采样 token 计数。

        GPU 版本需要等待 CUDA Event 同步，CPU 版本直接读取。

        返回：
            每个请求的有效采样 token 计数列表
        """
        prev_sampled_token_ids = self.input_batch.prev_sampled_token_ids
        if prev_sampled_token_ids is None:
            return []

        counts_cpu = self.valid_sampled_token_count_cpu
        if counts_cpu is None:
            return []
        return counts_cpu[: prev_sampled_token_ids.shape[0]].tolist()

    def _to_list(self, sampled_token_ids: torch.Tensor) -> list[list[int]]:
        """CPU-safe version: direct tolist() without CUDA events."""
        """
        CPU 安全版本：将采样的 token ID 张量转换为 Python 列表。

        GPU 版本需要等待 CUDA Event 同步后才能安全读取数据，
        CPU 版本直接调用 tolist() 即可（CPU 操作是同步的）。
        """
        return sampled_token_ids.tolist()


@contextmanager
def _torch_cuda_wrapper():
    """
    CUDA 包装器上下文管理器。

    在进入上下文时，临时将 torch.Event 和 torch.cuda.Stream 替换为
    空操作占位符（_EventPlaceholder 和 _StreamPlaceholder）。
    这样 GPUModelRunner.__init__() 中使用这些 CUDA 组件的代码
    不会在 CPU 环境下崩溃。

    退出上下文时恢复原始的 torch.Event 和 torch.cuda.Stream。

    ==========================================================
    【为什么需要这个包装器？】
    ==========================================================
    GPUModelRunner 的 __init__() 中会创建 CUDA Stream 和 Event
    用于异步数据传输。在 CPU 环境下没有 CUDA 设备，这些调用会失败。
    通过临时替换为占位符，我们可以在不修改父类代码的情况下安全初始化。
    """

    class _EventPlaceholder:
        """torch.Event 的空操作占位符。"""

        def __init__(self, *args, **kwargs) -> None:
            self.record = lambda: None
            self.synchronize = lambda: None

    class _StreamPlaceholder:
        """torch.cuda.Stream 的空操作占位符。"""

        def __init__(self, *args, **kwargs) -> None:
            pass

    cuda_event = torch.Event
    cuda_stream = torch.cuda.Stream
    try:
        torch.Event = _EventPlaceholder
        torch.cuda.Stream = _StreamPlaceholder
        yield
    finally:
        torch.Event = cuda_event
        torch.cuda.Stream = cuda_stream


@contextmanager
def _set_global_compilation_settings(config: VllmConfig):
    """
    设置全局编译选项上下文管理器。

    在模型预热期间，根据配置调整 torch._inductor 的编译选项。
    特别是当启用 max_autotune 时，需要开启 freezing（参数冻结）
    以支持 MKLDNN 和 CPPGEMM 后端。

    参数：
        config: vLLM 全局配置
    """
    import torch._inductor.config as torch_inductor_config

    inductor_config = config.compilation_config.inductor_compile_config
    # Note: The MKLDNN and CPPGEMM backend requires freezing parameters.
    freezing_value = torch_inductor_config.freezing
    try:
        if inductor_config.get("max_autotune", False):
            torch_inductor_config.freezing = True
        yield
    finally:
        torch_inductor_config.freezing = freezing_value


def _set_torch_accelerator_to_noop() -> None:
    """
    将 torch.accelerator 的同步和缓存清理操作设为空操作。

    CPU 没有独立的加速器设备，因此 torch.accelerator.synchronize()
    和 torch.accelerator.empty_cache() 会失败或无意义。
    将它们替换为空函数以避免错误。
    """
    def noop(*args: Any, **kwargs: Any) -> None:
        pass

    torch.accelerator.synchronize = noop
    torch.accelerator.empty_cache = noop
