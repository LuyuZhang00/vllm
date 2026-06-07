# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# =============================================================================
# CUDA 图调度器模块 (CUDA Graph Dispatcher Module)
# =============================================================================
# 本模块实现了一个运行时 CUDA 图调度器，负责管理和调度 CUDA 计算图。
#
# 核心概念：
# 1. CUDA 图 (CUDA Graph)：一种将一系列 CUDA 操作录制到一个图中，然后
#    整体重放的技术。这样可以减少 CPU 端的调度开销，提高 GPU 利用率。
# 2. 两种 CUDA 图模式：
#    - FULL（完整模式）：将整个前向传播过程录制为一个完整的 CUDA 图。
#      适用于解码阶段，因为解码时每个请求的 token 数固定，batch 结构
#      可预测。优点是性能最优，缺点是只能处理特定 batch 结构。
#    - PIECEWISE（分段模式）：将前向传播过程分割成多段，每段分别录制
#      CUDA 图。适用于预填充阶段，因为预填充时不同请求的 token 数不同，
#      batch 结构不规则。优点是灵活性高，适用范围更广。
# 3. NONE（无图模式）：不使用 CUDA 图，直接执行普通操作。
#
# 调度流程：
# 1) 初始化阶段：根据配置计算所有可能的 batch 描述符 (BatchDescriptor)，
#    作为有效调度键存入 cudagraph_keys 字典。
# 2) 运行时调度阶段：每步前向传播时，根据当前 batch 的特征（token 数、
#    请求数、是否均匀解码等）调用 dispatch() 方法，返回最优的 CUDA 图模式
#    和对应的 batch 描述符。
# 3) 图捕获阶段：通过 get_capture_descs() 获取所有需要捕获的 batch 描述
#    列表，用于预先录制 CUDA 图。
# =============================================================================

from collections.abc import Set as AbstractSet
from dataclasses import replace
from itertools import product

from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import BatchDescriptor
from vllm.logger import init_logger
from vllm.lora.utils import get_captured_lora_counts

logger = init_logger(__name__)


class CudagraphDispatcher:
    """
    Runtime cudagraph dispatcher to dispatch keys for multiple set of
    cudagraphs.

    The dispatcher stores two sets of dispatch keys, one for PIECEWISE and one
    for FULL cudagraph runtime mode. The keys are initialized depending on
    attention support and what cudagraph mode is set in CompilationConfig. The
    keys stored in dispatcher are the only source of truth for valid
    cudagraphs that can be dispatched at runtime.

    At runtime, the dispatch method generates the runtime cudagraph mode (FULL,
    PIECEWISE, or NONE for no cudagraph) and the valid key (batch descriptor)
    based on the input key. After dispatching (communicated via forward
    context), the cudagraph wrappers will trust the dispatch key to either
    capture or replay (if the mode matches), or pass through to the underlying
    runnable without cudagraph (if the mode does not match or mode is NONE).
    """

    # =========================================================================
    # CudagraphDispatcher 类：CUDA 图调度器
    # =========================================================================
    # 这是整个 CUDA 图调度系统的核心类。它维护了所有有效的 CUDA 图调度键，
    # 并在运行时根据输入的 batch 特征决定使用哪种 CUDA 图模式。
    #
    # 主要职责：
    # 1. 维护有效的 CUDA 图调度键集合（cudagraph_keys）
    # 2. 在运行时根据 batch 特征进行调度决策
    # 3. 提供图捕获所需的描述符列表
    #
    # 内部数据结构：
    # - cudagraph_keys: dict[CUDAGraphMode, set[BatchDescriptor]]
    #   存储两种模式（PIECEWISE 和 FULL）下的所有有效调度键。
    #   这是运行时有效 CUDA 图的唯一权威来源。
    # - _bs_to_padded_graph_size: list[int]
    #   从实际 batch size 到填充后图大小的映射表。
    #   例如：如果捕获大小为 [8, 16, 32]，则 batch_size=5 会映射到 8，
    #   batch_size=10 会映射到 16，batch_size=20 会映射到 32。
    # =========================================================================

    def __init__(self, vllm_config: VllmConfig):
        # ---- 初始化方法 ----
        # 接收 VllmConfig 配置对象，初始化调度器的基本参数。
        # 此时尚未初始化 CUDA 图键，需要后续调用 initialize_cudagraph_keys()。
        self.vllm_config = vllm_config
        self.compilation_config = vllm_config.compilation_config

        # ---- 均匀解码查询长度 (uniform_decode_query_len) ----
        # 定义"均匀解码"模式下每个请求的 token 数量。
        # 1) 普通解码：每个请求恰好有 1 个 token（自回归生成）。
        # 2) 投机解码 (Speculative Decoding)：每个请求有
        #    1 + num_speculative_tokens 个 token。
        #    投机解码会一次猜测多个 token，然后验证。
        self.uniform_decode_query_len = (
            1
            if not self.vllm_config.speculative_config
            else 1 + self.vllm_config.speculative_config.num_speculative_tokens
        )

        # ---- CUDA 图调度键字典 ----
        # 存储有效的 CUDA 图调度键，按模式分类：
        # - CUDAGraphMode.PIECEWISE: 分段模式下的所有有效 BatchDescriptor
        # - CUDAGraphMode.FULL: 完整模式下的所有有效 BatchDescriptor
        # 这些键是运行时判断是否可以使用 CUDA 图的唯一依据。
        self.cudagraph_keys: dict[CUDAGraphMode, set[BatchDescriptor]] = {
            CUDAGraphMode.PIECEWISE: set(),
            CUDAGraphMode.FULL: set(),
        }

        from vllm.compilation.breakable_cudagraph import (
            is_breakable_cudagraph_enabled,
        )

        # ---- 配置校验 ----
        # 确保分段模式的前置条件满足：
        # 如果 cudagraph_mode 需要分段编译（PIECEWISE），则必须满足以下
        # 条件之一：
        # 1) attention 操作已配置在 splitting_ops 中（即注意力计算会被
        #    作为分段点）
        # 2) 或者启用了 breakable_cudagraph（可中断的 CUDA 图）
        # 否则，分段模式无法正确工作，因为不知道在哪里分割计算图。
        assert (
            not self.compilation_config.cudagraph_mode.requires_piecewise_compilation()
            or self.compilation_config.is_attention_compiled_piecewise()
            or is_breakable_cudagraph_enabled()
        ), (
            "Compilation mode should be CompilationMode.VLLM_COMPILE when "
            "cudagraph_mode piecewise cudagraphs is used, "
            "and attention should be in splitting_ops or "
            "inductor splitting should be used. "
            f"cudagraph_mode={self.compilation_config.cudagraph_mode}, "
            f"compilation_mode={self.compilation_config.mode}, "
            f"splitting_ops={self.compilation_config.splitting_ops}"
        )

        # ---- 键初始化标志 ----
        # 标记 CUDA 图键是否已经初始化完成。
        # 在 initialize_cudagraph_keys() 调用之前，调度器处于未初始化状态。
        self.keys_initialized = False

        # ---- LoRA 专用化配置 ----
        # 是否为不同的 LoRA 适配器数量分别捕获 CUDA 图。
        # 如果启用，则会为 0、1、2、4、8... 等不同的 active_lora 数量
        # 分别创建独立的 CUDA 图。这样可以让依赖 LoRA 数量的内核
        # （如 fused_moe_lora）得到正确的图捕获。
        self.specialize_lora_count = (
            self.vllm_config.lora_config.specialize_active_lora
            if self.vllm_config.lora_config is not None
            else False
        )

        # ---- 默认 CUDA 图模式 ----
        # 初始设置为 NONE（不使用 CUDA 图），直到 initialize_cudagraph_keys()
        # 被调用后才会更新为实际的模式。
        self.cudagraph_mode = CUDAGraphMode.NONE

    def _compute_bs_to_padded_graph_size(self) -> None:
        """Pre-compute the mapping from batch size to padded graph size."""

        # ---- 计算 batch size 到填充后图大小的映射 ----
        # 作用：将任意 batch size 映射到最近的、已捕获的图大小。
        #
        # 工作原理示例：
        # 假设 cudagraph_capture_sizes = [8, 16, 32], max_size = 32
        #
        # 则 _bs_to_padded_graph_size 数组为：
        #   [0, 8, 8, 8, 8, 8, 8, 8, 8,   # bs 0-8 -> 8
        #    16, 16, 16, 16, 16, 16, 16, 16, # bs 9-16 -> 16
        #    32, 32, 32, 32, 32, 32, 32, 32, # bs 17-24 -> 32
        #    32, 32, 32, 32, 32, 32, 32, 32] # bs 25-32 -> 32
        #
        # 注意：每个区间的第一个元素映射到区间起点（精确匹配），
        # 其余元素映射到区间终点（向上取整）。这样确保：
        # 1) 精确大小（如 bs=8）映射到自身
        # 2) 其他大小（如 bs=5）向上取整到最近的捕获大小
        max_size = self.compilation_config.max_cudagraph_capture_size
        capture_sizes = self.compilation_config.cudagraph_capture_sizes
        assert max_size is not None, (
            "Maximum cudagraph capture size must be set when cudagraphs are enabled."
        )
        assert capture_sizes is not None, (
            "Cudagraph capture sizes must be set when cudagraphs are enabled."
        )
        self._bs_to_padded_graph_size: list[int] = [0] * (max_size + 1)
        for end, start in zip(
            capture_sizes + [max_size + 1],
            [0] + capture_sizes,
        ):
            for bs in range(start, end):
                if bs == start:
                    self._bs_to_padded_graph_size[bs] = start
                else:
                    self._bs_to_padded_graph_size[bs] = end

        # ---- 校验 compile_sizes 与填充的兼容性 ----
        # 确保用户指定的 compile_sizes 在经过填充后不会改变。
        # 如果 compile_sizes 中的某个大小会被填充到不同的值，则抛出错误。
        # 这是因为 compile_sizes 指定的是需要精确编译的大小，
        # 如果填充改变了这些大小，就无法保证编译的正确性。
        # Only validate when cudagraphs are actually being used.
        if (
            self.compilation_config.compile_sizes
            and self.cudagraph_mode != CUDAGraphMode.NONE
        ):
            for size in self.compilation_config.compile_sizes:
                size = int(size)
                if size <= max_size:
                    padded = self._bs_to_padded_graph_size[size]
                    if padded != size:
                        raise ValueError(
                            f"compile_sizes contains {size} which would be "
                            f"padded to {padded}. All compile_sizes must be "
                            "values that won't be changed by cudagraph padding. "
                            "Use values from cudagraph_capture_sizes."
                        )

    def _get_lora_cases(self) -> list[int]:
        """
        Returns list of has_lora values for CUDA graph capture.
        This is the single source of truth for LoRA capture cases.
        """

        # ---- 获取 LoRA 捕获场景列表 ----
        # 确定需要为哪些 LoRA 配置捕获 CUDA 图。
        #
        # 三种情况：
        # 1) 无 LoRA 配置：只捕获 [0]（无 LoRA 的情况）
        # 2) 启用 LoRA 且 specialize_lora_count=True：
        #    捕获 [0, 1, 2, 4, 8, ...] 等不同 active_lora 数量的图。
        #    通过 get_captured_lora_counts() 获取具体数量列表。
        # 3) 启用 LoRA 且 specialize_lora_count=False：
        #    只捕获 [max_loras + 1]（即最大 LoRA 数量 + 1 的情况）。
        #    这样一张图可以处理所有 LoRA 配置。
        lora_config = self.vllm_config.lora_config
        if lora_config is None:
            # No LoRA configured - single case with no LoRA
            return [0]

        # LoRA is enabled - capture graphs based on cudagraph_specialize_lora
        if self.compilation_config.cudagraph_specialize_lora:
            captured_counts = get_captured_lora_counts(
                lora_config.max_loras, self.specialize_lora_count
            )
            # Specialize: capture separate graphs for with and without LoRA
            return [0] + captured_counts
        else:
            # No specialization: only capture graphs with LoRA active
            return [lora_config.max_loras + 1]

    def _create_padded_batch_descriptor(
        self,
        num_tokens: int,
        uniform_decode: bool,
        has_lora: bool,
        num_active_loras: int = 0,
    ) -> BatchDescriptor:
        # ---- 创建填充后的 BatchDescriptor ----
        # 根据输入的 token 数量和模式，创建填充后的 batch 描述符。
        #
        # 参数说明：
        # - num_tokens: 实际的 token 数量
        # - uniform_decode: 是否为均匀解码模式（所有请求 token 数相同）
        # - has_lora: 是否有 LoRA 适配器激活
        # - num_active_loras: 活跃的 LoRA 适配器数量
        #
        # 处理流程：
        # 1. 将 num_tokens 填充到最近的捕获大小
        # 2. 如果是均匀解码且当前模式支持 FULL，则：
        #    - 计算请求数 = min(填充后 token 数 / 查询长度, 最大序列数)
        #    - 确保填充后 token 数能被查询长度整除
        # 3. 否则（非均匀解码）：
        #    - 请求数 = min(填充后 token 数, 最大序列数)
        #    - uniform_decode 强制设为 False
        #
        # 返回：填充后的 BatchDescriptor 对象
        max_num_seqs = self.vllm_config.scheduler_config.max_num_seqs
        uniform_decode_query_len = self.uniform_decode_query_len
        num_tokens_padded = self._bs_to_padded_graph_size[num_tokens]

        if uniform_decode and self.cudagraph_mode.has_mode(CUDAGraphMode.FULL):
            num_reqs = min(num_tokens_padded // uniform_decode_query_len, max_num_seqs)
            assert num_tokens_padded % uniform_decode_query_len == 0
        else:
            uniform_decode = False
            num_reqs = min(num_tokens_padded, max_num_seqs)

        return BatchDescriptor(
            num_tokens=num_tokens_padded,
            num_reqs=num_reqs,
            uniform=uniform_decode,
            has_lora=has_lora,
            num_active_loras=num_active_loras,
        )

    def add_cudagraph_key(
        self, runtime_mode: CUDAGraphMode, batch_descriptor: BatchDescriptor
    ):
        # ---- 添加 CUDA 图调度键 ----
        # 将一个 BatchDescriptor 添加到指定模式的调度键集合中。
        #
        # 参数：
        # - runtime_mode: CUDA 图运行时模式（只能是 PIECEWISE 或 FULL）
        # - batch_descriptor: 要添加的 batch 描述符
        #
        # 注意：此方法会在 initialize_cudagraph_keys() 中被多次调用，
        # 用于构建完整的有效调度键集合。
        assert runtime_mode in [CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL], (
            f"Invalid cudagraph runtime mode for keys: {runtime_mode}"
        )
        self.cudagraph_keys[runtime_mode].add(batch_descriptor)

    def initialize_cudagraph_keys(
        self, cudagraph_mode: CUDAGraphMode, uniform_decode_query_len: int = 1
    ):
        # ---- 初始化 CUDA 图调度键 ----
        # 这是调度器的核心初始化方法，必须在 attention backend 初始化完成后调用。
        # 它会根据配置生成所有有效的 CUDA 图调度键。
        #
        # 初始化流程：
        # 1) 设置 cudagraph_mode（从配置解析后的实际模式）
        # 2) 如果模式为 NONE，则提前返回（不使用 CUDA 图）
        # 3) 计算 batch size 到填充图大小的映射表
        # 4) 获取 LoRA 捕获场景列表
        # 5) 生成混合模式（mixed_mode）的调度键：
        #    - 遍历所有捕获大小和 LoRA 场景的组合
        #    - 为每个组合创建填充后的 BatchDescriptor
        #    - 对于 PIECEWISE 模式，放松约束（num_reqs=None, uniform=False）
        # 6) 如果解码模式为 FULL 且需要分离例程：
        #    - 计算解码专用的捕获大小范围
        #    - 为解码阶段生成独立的 FULL 模式调度键
        # 7) 标记初始化完成
        # This should be called only after attention backend is initialized. So we can
        # get the correct cudagraph mode after backend support is resolved.
        self.cudagraph_mode = cudagraph_mode

        # Early exit if cudagraphs are disabled
        if cudagraph_mode == CUDAGraphMode.NONE:
            self.keys_initialized = True
            return

        self._compute_bs_to_padded_graph_size()

        # Get LoRA cases to capture
        lora_cases = self._get_lora_cases()
        self.captured_lora_counts = [
            lora_count for lora_count in lora_cases if lora_count
        ]

        # Note: we create all valid keys for cudagraph here but do not
        # guarantee all keys would be used. For example, if we allow lazy
        # capturing in future PR, some keys may never be triggered.

        # ---- 生成混合模式调度键 ----
        # mixed_mode() 返回模式中的第二个分量：
        # - FULL_DECODE_ONLY -> NONE（无混合模式）
        # - FULL_AND_PIECEWISE -> PIECEWISE
        # - 单一模式（如 FULL 或 PIECEWISE）-> 返回自身
        # 如果混合模式不为 NONE，则需要为混合模式生成调度键。
        if cudagraph_mode.mixed_mode() != CUDAGraphMode.NONE:
            assert self.compilation_config.cudagraph_capture_sizes is not None, (
                "Cudagraph capture sizes must be set when mixed mode is enabled."
            )
            # 使用 product 生成所有 (batch_size, lora_count) 组合
            for bs, num_active_loras in product(
                self.compilation_config.cudagraph_capture_sizes, lora_cases
            ):
                batch_desc = self._create_padded_batch_descriptor(
                    bs, False, num_active_loras > 0, num_active_loras
                )
                # Only relax for PIECEWISE mode. FULL mode needs exact num_reqs
                # because FA3's scheduler_metadata computation depends on it.
                # 对于 PIECEWISE 模式，放松 num_reqs 和 uniform 约束，
                # 使其可以处理任意数量的请求和非均匀 batch。
                # 对于 FULL 模式，必须保留精确的 num_reqs，
                # 因为 FA3（FlashAttention 3）的 scheduler_metadata 计算依赖于此。
                if cudagraph_mode.mixed_mode() == CUDAGraphMode.PIECEWISE:
                    batch_desc = replace(batch_desc, num_reqs=None, uniform=False)
                self.add_cudagraph_key(cudagraph_mode.mixed_mode(), batch_desc)

        # ---- 生成解码专用的 FULL 模式调度键 ----
        # 如果解码模式为 FULL 且需要分离例程（separate_routine），
        # 则需要为解码阶段单独生成 FULL 模式调度键。
        # 分离例程意味着预填充和解码使用不同的 CUDA 图模式：
        # - 预填充：使用 PIECEWISE 模式（灵活性高）
        # - 解码：使用 FULL 模式（性能最优）
        # if decode cudagraph mode is FULL, and we don't already have mixed
        # mode full cudagraphs then add them here.
        if (
            cudagraph_mode.decode_mode() == CUDAGraphMode.FULL
            and cudagraph_mode.separate_routine()
        ):
            # 计算解码阶段的最大 token 数：查询长度 * 最大序列数
            max_num_tokens = (
                uniform_decode_query_len
                * self.vllm_config.scheduler_config.max_num_seqs
            )
            assert self.compilation_config.cudagraph_capture_sizes is not None, (
                "Cudagraph capture sizes must be set when full mode is enabled."
            )
            # 过滤出适用于解码的捕获大小：
            # 1) 不超过最大 token 数
            # 2) 至少为查询长度（确保至少能处理一个请求）
            cudagraph_capture_sizes_for_decode = [
                x
                for x in self.compilation_config.cudagraph_capture_sizes
                if x <= max_num_tokens and x >= uniform_decode_query_len
            ]
            for bs, num_active_loras in product(
                cudagraph_capture_sizes_for_decode, lora_cases
            ):
                self.add_cudagraph_key(
                    CUDAGraphMode.FULL,
                    self._create_padded_batch_descriptor(
                        bs, True, num_active_loras > 0, num_active_loras
                    ),
                )

        self.keys_initialized = True

    def dispatch(
        self,
        num_tokens: int,
        uniform_decode: bool = False,
        has_lora: bool = False,
        num_active_loras: int = 0,
        valid_modes: AbstractSet[CUDAGraphMode] | None = None,
        invalid_modes: AbstractSet[CUDAGraphMode] | None = None,
    ) -> tuple[CUDAGraphMode, BatchDescriptor]:
        """
        Given conditions(e.g.,batch descriptor and if using piecewise only),
        dispatch to a cudagraph runtime mode and the valid batch descriptor.
        A new batch descriptor is returned as we might dispatch a uniform batch
        to a graph that supports a more general batch (uniform to non-uniform).

        Args:
            num_tokens: Number of tokens in the batch.
            uniform_decode: Whether the batch is uniform decode (i.e. uniform and query
                length is uniform_decode_query_len).
            has_lora: Whether LoRA is active.
            num_active_loras: Number of distinct active LoRA adapters.
            valid_modes: Set of cudagraph modes that are allowed. None means
                all modes are allowed.
            invalid_modes: Set of cudagraph modes to exclude. Subtracted from
                valid_modes to compute allowed modes. (e.g., {FULL} for
                features like cascade attention not supported by full
                cudagraphs). None means no modes are excluded.
        """

        # ---- 调度方法：运行时 CUDA 图模式选择 ----
        # 这是调度器的核心方法，在每次前向传播时被调用。
        # 它根据当前 batch 的特征，选择最优的 CUDA 图模式。
        #
        # 调度逻辑：
        # 1) 确定允许的模式集合：
        #    - 从 valid_modes 开始（默认为所有运行时模式）
        #    - 减去 invalid_modes（排除不支持的模式）
        # 2) 快速返回条件（返回 NONE 模式）：
        #    - 键未初始化
        #    - cudagraph_mode 为 NONE
        #    - max_size 未设置
        #    - num_tokens 超过最大捕获大小
        #    - 允许的模式只有 NONE
        # 3) 处理 LoRA 适配器数量：
        #    - 如果启用 LoRA 专用化，找到最小的 >= 当前数量的捕获数量
        #    - 如果未启用，使用 max_loras + 1
        # 4) 创建填充后的 BatchDescriptor
        # 5) 按优先级尝试匹配：
        #    - 优先尝试 FULL 模式（性能最优）
        #    - 然后尝试 PIECEWISE 模式（使用放松后的键）
        #    - 如果都匹配失败，返回 NONE 模式
        #
        # 返回值：(CUDA 图模式, BatchDescriptor) 元组
        allowed_modes = valid_modes or CUDAGraphMode.valid_runtime_modes()

        if invalid_modes:
            allowed_modes -= invalid_modes

        assert len(allowed_modes) >= 1, (
            f"No allowed cudagraph modes: valid_modes={valid_modes}, "
            f"invalid_modes={invalid_modes}"
        )
        max_size = self.compilation_config.max_cudagraph_capture_size

        # ---- 快速返回 NONE 模式的条件 ----
        if (
            not self.keys_initialized
            or self.cudagraph_mode == CUDAGraphMode.NONE
            or max_size is None
            or num_tokens > max_size
            or allowed_modes <= {CUDAGraphMode.NONE}
        ):
            return CUDAGraphMode.NONE, BatchDescriptor(num_tokens)

        # ---- 处理 LoRA 适配器数量 ----
        effective_num_active_loras = num_active_loras
        if has_lora and num_active_loras > 0:
            if self.specialize_lora_count:
                # Find the smallest captured `num_active_loras` that is >= the current
                # `num_active_loras`. This is because we only capture graphs for
                # a subset of possible `num_active_loras` values (powers of 2).
                # 使用二分查找找到最小的 >= 当前数量的捕获数量。
                # 例如：如果捕获数量为 [0, 1, 2, 4, 8]，当前为 3，则选择 4。
                import bisect

                idx = bisect.bisect_left(self.captured_lora_counts, num_active_loras)
                if idx < len(self.captured_lora_counts):
                    effective_num_active_loras = self.captured_lora_counts[idx]
            else:
                # When not specializing, graphs are captured only with max_loras + 1,
                # so we must use max_loras + 1 for dispatch to find a matching graph.
                # 未启用专用化时，所有图都按 max_loras + 1 捕获，
                # 所以调度时也必须使用 max_loras + 1。
                assert self.vllm_config.lora_config is not None, (
                    "LoRA config must be set when has_lora is True."
                )
                effective_num_active_loras = self.vllm_config.lora_config.max_loras + 1

        # ---- 创建填充后的 BatchDescriptor ----
        # 只有在分离例程模式下才使用 uniform_decode 标志。
        # 分离例程意味着预填充和解码使用不同的图，此时解码阶段
        # 通常是均匀的（每个请求 1 个 token）。
        normalized_uniform = uniform_decode and self.cudagraph_mode.separate_routine()
        batch_desc = self._create_padded_batch_descriptor(
            num_tokens, normalized_uniform, has_lora, effective_num_active_loras
        )

        # ---- 按优先级尝试匹配 CUDA 图 ----
        # 优先级：FULL > PIECEWISE > NONE
        # FULL 模式性能最优，因为它将整个前向传播录制为一个图。
        if CUDAGraphMode.FULL in allowed_modes:
            # check if key exists for full cudagraph
            batch_desc_to_check = batch_desc
            if batch_desc_to_check in self.cudagraph_keys[CUDAGraphMode.FULL]:
                return CUDAGraphMode.FULL, batch_desc_to_check

        if CUDAGraphMode.PIECEWISE in allowed_modes:
            # also check if the relaxed key exists for more "general"
            # piecewise cudagraph
            # PIECEWISE 模式使用放松后的键（num_reqs=None, uniform=False），
            # 这样一个图可以处理更多种类的 batch。
            batch_desc_to_check = replace(batch_desc, num_reqs=None, uniform=False)
            if batch_desc_to_check in self.cudagraph_keys[CUDAGraphMode.PIECEWISE]:
                return CUDAGraphMode.PIECEWISE, batch_desc_to_check

        # ---- 匹配失败，返回 NONE 模式 ----
        # 如果没有匹配的 CUDA 图，则回退到普通执行模式。
        assert CUDAGraphMode.NONE in allowed_modes, (
            f"No matching cudagraph found and NONE is not in "
            f"allowed_modes={allowed_modes}"
        )
        return CUDAGraphMode.NONE, BatchDescriptor(num_tokens)

    def get_capture_descs(self) -> list[tuple[CUDAGraphMode, list[BatchDescriptor]]]:
        """
        Returns capture descriptors for cudagraph capturing.

        Returns:
            List of (runtime_mode, batch_descriptors) tuples, ordered PIECEWISE
            first then FULL. Batch descriptors are sorted largest-first for
            memory efficiency.
        """

        # ---- 获取图捕获描述符 ----
        # 返回需要捕获的所有 CUDA 图描述符列表。
        # 这个方法在模型初始化阶段被调用，用于确定需要预先录制哪些 CUDA 图。
        #
        # 返回值格式：
        # [(PIECEWISE 模式, [描述符列表]), (FULL 模式, [描述符列表])]
        #
        # 排序策略：
        # - 先返回 PIECEWISE 模式的描述符，再返回 FULL 模式的
        # - 每个模式内按 (num_tokens, num_active_loras) 降序排列
        # - 从大到小捕获有利于内存效率（大图可以复用小图的内存）
        if not self.keys_initialized or self.cudagraph_mode == CUDAGraphMode.NONE:
            return []

        result = []
        # Return in order: PIECEWISE first, then FULL
        for mode in [CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL]:
            descs = list(self.cudagraph_keys[mode])
            if descs:
                # Sort by (num_tokens, num_active_loras) descending
                descs.sort(
                    key=lambda d: (d.num_tokens, d.num_active_loras),
                    reverse=True,
                )
                result.append((mode, descs))

        return result
