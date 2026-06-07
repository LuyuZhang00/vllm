# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""High-Performance Triton-only Attention layer."""

# =============================================================================
# 中文说明：vLLM V1 Triton 注意力后端模块
# =============================================================================
# 本文件实现了基于 Triton 的高性能注意力后端，是 vLLM V1 推理引擎中
# 注意力计算的纯 Triton 实现方案。它不依赖 FlashAttention 等第三方 CUDA
# 库，而是完全通过 Triton JIT 编译的 kernel 来完成注意力计算。
#
# 整体架构包含四个核心类：
#
# 1. TritonAttentionMetadata（注意力元数据数据类）
#    - 封装一次前向传播所需的所有注意力元数据
#    - 包含 query/seq 长度信息、block_table（页表）、slot_mapping（槽位映射）
#    - 包含并行分段 softmax 所需的中间缓冲区（用于 3D kernel 路径）
#    - 包含 cascade attention（级联注意力）所需的公共前缀信息
#
# 2. TritonAttentionMetadataBuilder（元数据构建器）
#    - 负责从 CommonAttentionMetadata 构建 Triton 特定的元数据对象
#    - 管理 2D/3D kernel 路径的选择阈值（seq_threshold_3D）
#    - 预分配并行 softmax 分段的 GPU 缓冲区
#    - 支持 CUDA Graph 捕获场景下的元数据构建
#
# 3. TritonAttentionBackend（后端注册类）
#    - 注册后端能力：支持的数据类型、KV cache 数据类型、block size 等
#    - 工厂方法：返回 Impl 类和 Builder 类
#    - 定义 KV cache 的 shape 和内存布局（NHD/HND）
#
# 4. TritonAttentionImpl（注意力计算实现类）
#    - forward(): decode 和 prefill 阶段的注意力前向传播
#      - 根据 batch 大小选择 2D kernel（序列数 >= 阈值）或 3D kernel
#      - 支持滑动窗口、ALiBi 位置编码、logits soft cap 等特性
#    - do_kv_cache_update(): 将当前 step 的 K/V 写入 KV cache
#      - 支持 FP8 量化和 per-token-head 量化模式
#    - _forward_encoder_attention(): encoder 注意力（双向，无 KV cache）
#
# 关键设计决策：
# - KV cache 采用分页管理（PagedAttention），通过 block_table 将逻辑地址
#   映射到物理显存，不同请求的 KV cache 不需要在物理上连续
# - 2D kernel 与 3D kernel 的选择：当 batch 序列数较少时，2D kernel 的
#   GPU SM 利用率不足，改用 3D kernel（按 Q block 维度展开）以获得更高并行度
# - KV cache 支持 NHD 和 HND 两种内存布局，通过 stride_order 控制
# - forward_includes_kv_cache_update = False，表示 KV cache 更新与注意力
#   计算是分离的两个步骤，由 GPUModelRunner 分别调用
#
# 数据流：
#   SchedulerOutput -> GPUModelRunner -> CommonAttentionMetadata
#   -> TritonAttentionMetadataBuilder.build() -> TritonAttentionMetadata
#   -> TritonAttentionImpl.do_kv_cache_update()  （写入 KV cache）
#   -> TritonAttentionImpl.forward()             （执行注意力计算）
# =============================================================================

from dataclasses import dataclass
from typing import ClassVar

import torch

import vllm.envs as envs
from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8StaticTensorSym,
)
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.utils.math_utils import next_power_of_2
from vllm.utils.torch_utils import async_tensor_h2d, is_quantized_kv_cache
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.utils import (
    get_kv_cache_layout,
    get_num_attention_heads_from_layers,
)
from vllm.v1.attention.ops.triton_prefill_attention import context_attention_fwd
from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
    triton_reshape_and_cache_flash,
    triton_reshape_and_cache_flash_per_token_head_quant,
)
from vllm.v1.attention.ops.triton_unified_attention import unified_attention
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVQuantMode,
    get_kv_quant_mode,
    kv_cache_uses_per_token_head_scales,
)

logger = init_logger(__name__)


# 中文注释：2D kernel 的最小启动网格大小。
# 2D kernel 的启动网格为 (num_q_blocks, num_heads_kv)，如果序列数太少
# 导致 num_q_blocks 不足，GPU SM 利用率会很低，此时应改用 3D kernel。
MIN_LAUNCH_GRID_SIZE_2D = 128  # Minimum launch grid size of 2D kernel
# 中文注释：并行分段 softmax 的分段数。
# 3D kernel 将 softmax 计算拆分为多个 segment 并行处理，
# 每个 segment 独立计算局部 max 和 expsum，最后通过归约得到全局结果。
NUM_PAR_SOFTMAX_SEGMENTS = 16  # Number of parallel tiled softmax segments


# 中文注释：Triton 注意力元数据数据类。
# 封装了一次前向传播中注意力计算所需的全部元数据。
# 这些元数据由 TritonAttentionMetadataBuilder 构建，传递给
# TritonAttentionImpl.forward() 和 do_kv_cache_update() 使用。
#
# 关键字段说明：
# - num_actual_tokens: 本 batch 中实际的 token 数（去除 padding）
# - block_table: 页表，将逻辑 block id 映射到物理显存 block 位置
# - slot_mapping: 每个 token 在 KV cache 中的物理槽位地址
# - seq_lens: 每个序列当前的总长度（已计算 + 待计算）
# - query_start_loc: 每个序列 query 在拼接张量中的起始位置
# - seq_threshold_3D: 选择 3D kernel 的序列数阈值
# - 三个 softmax_segm_* 缓冲区：3D kernel 并行分段 softmax 的中间结果
@dataclass
class TritonAttentionMetadata:
    # NOTE(sang): Definition of context_len, query_len, and seq_len.
    # |---------- N-1 iteration --------|
    # |---------------- N iteration ---------------------|
    # |- tokenA -|......................|-- newTokens ---|
    # |---------- context_len ----------|
    # |-------------------- seq_len ---------------------|
    #                                   |-- query_len ---|

    # --- 基本 token 和序列信息 ---
    num_actual_tokens: int  # Number of tokens excluding padding.
    # 中文注释：本 batch 中所有序列的最大 query 长度（不含已计算的 context 部分）
    max_query_len: int
    # 中文注释：每个序列的 query 在拼接 token 张量中的起始偏移（cumsum 格式）。
    # shape = [num_seqs + 1]，用于区分不同序列的 token 边界。
    query_start_loc: torch.Tensor
    # 中文注释：本 batch 中所有序列的最大总长度（context + query）
    max_seq_len: int
    # 中文注释：每个序列当前的总有效长度（context_len + query_len）。
    # 用于 KV cache 的边界检查和 mask 计算。
    seq_lens: torch.Tensor
    # 中文注释：页表，逻辑 block id 到物理显存 block id 的映射。
    # shape = [num_seqs, max_num_blocks_per_seq]。
    # PagedAttention 的核心：通过此表实现 KV cache 的非连续物理存储。
    block_table: torch.Tensor
    # 中文注释：每个 token 对应的 KV cache 物理槽位地址。
    # 由 (block_id * block_size + offset_in_block) 计算得到，
    # 用于 reshape_and_cache 将 K/V 写入正确的物理位置。
    slot_mapping: torch.Tensor

    # --- 3D kernel 并行分段 softmax 相关 ---
    # 中文注释：选择 3D kernel 的序列数阈值。当 batch 中序列数 < 此阈值时，
    # 2D kernel 的 GPU SM 利用率不足，改用 3D kernel 获得更高并行度。
    seq_threshold_3D: int
    # 中文注释：并行 softmax 的分段数，3D kernel 将 softmax 计算
    # 拆分为这么多独立 segment 并行处理。
    num_par_softmax_segments: int
    # 中文注释：3D kernel 并行分段 softmax 的中间输出缓冲区。
    # shape = [seq_threshold_3D, num_heads_q, num_par_softmax_segments, headdim_padded]
    softmax_segm_output: torch.Tensor
    # 中文注释：每个 segment 的局部 max 值，用于后续归约。
    softmax_segm_max: torch.Tensor
    # 中文注释：每个 segment 的局部 expsum 值，用于后续归约。
    softmax_segm_expsum: torch.Tensor

    # --- 级联注意力（Cascade Attention）相关 ---
    # 中文注释：是否使用级联注意力。当多个序列共享较长的公共前缀时，
    # 级联注意力可以避免重复计算公共前缀部分的 KV，提升效率。
    # For cascade attention.
    use_cascade: bool
    common_prefix_len: int
    cu_prefix_query_lens: torch.Tensor | None
    prefix_kv_lens: torch.Tensor | None
    suffix_kv_lens: torch.Tensor | None

    # --- 可选的 AOT 调度和多模态前缀相关 ---
    # Optional aot scheduling
    scheduler_metadata: torch.Tensor | None = None
    prefix_scheduler_metadata: torch.Tensor | None = None
    # 中文注释：多模态（mm = multimodal）前缀的 token 范围。
    # key 为序列索引，value 为该序列中多模态 token 的 (start, end) 区间列表。
    # 用于在注意力计算中对多模态 token 进行特殊处理。
    mm_prefix_range: dict[int, list[tuple[int, int]]] | None = None
    mm_prefix_range_tensor: torch.Tensor | None = None

    @staticmethod
    def compute_mm_prefix_range_tensor(
        mm_prefix_range: dict[int, list[tuple[int, int]]] | None,
        num_seqs: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        """Convert mm_prefix_range dict to padded tensor for Triton kernel.

        Returns shape: (num_seqs, max_ranges, 2) with 0-padding for empty ranges.
        Empty ranges have start==end==0, which kernel skips via is_valid check.
        """
        # 中文注释：将多模态前缀范围从 Python 字典转换为 GPU 上的填充张量。
        # Triton kernel 无法直接接受 Python 字典，因此需要：
        # 1. 将所有序列的范围列表补齐到相同长度（用 (0,0) 填充）
        # 2. 构建为 (num_seqs, max_ranges, 2) 的规则张量
        # 3. 在 CPU pinned memory 上构建后一次性 H2D 传输，避免多次小传输
        #
        # 多模态前缀（mm_prefix）的应用场景：
        # 当输入包含图像/视频等多模态 token 时，这些 token 的位置编码
        # 和注意力 mask 可能与文本 token 不同。通过记录每个序列中
        # 多模态 token 的 (start, end) 范围，kernel 可以在注意力计算中
        # 对这些 token 进行特殊处理（如调整 mask 或位置偏移）。
        if mm_prefix_range is None:
            return None

        # Collect ranges, using [(0,0)] for empty sequences to ensure uniform dims
        range_lists = [
            mm_prefix_range.get(i, [(0, 0)]) or [(0, 0)] for i in range(num_seqs)
        ]

        # Return None if all ranges are trivial (only (0,0) placeholders)
        if all(r == [(0, 0)] for r in range_lists):
            return None

        # Build on CPU first then move to GPU in a single H2D transfer
        max_ranges = max(len(r) for r in range_lists)
        # Pad all sequences to the same number of ranges
        padded = []
        for r in range_lists:
            padded_r = list(r) + [(0, 0)] * (max_ranges - len(r))
            padded.append(padded_r)
        # Build on pinned CPU memory so the H2D transfer is non-blocking.
        padded = async_tensor_h2d(padded, dtype=torch.int32, device=device)
        return padded.view(num_seqs, max_ranges, 2)


# 中文注释：Triton 注意力元数据构建器。
# 负责从通用的 CommonAttentionMetadata 构建 Triton 特定的元数据对象。
# 核心职责：
# 1. 管理 2D/3D kernel 路径的选择逻辑（基于序列数阈值）
# 2. 预分配并行分段 softmax 的 GPU 缓冲区（避免每次推理重新分配）
# 3. 处理级联注意力（cascade attention）的元数据
# 4. 支持 CUDA Graph 捕获场景
#
# _cudagraph_support = ALWAYS 表示此后端始终支持 CUDA Graph，
# 无需额外的图捕获兼容性检查。
class TritonAttentionMetadataBuilder(AttentionMetadataBuilder[TritonAttentionMetadata]):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.ALWAYS

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        # 中文注释：初始化 Triton 注意力元数据构建器。
        # 构建流程（按顺序）：
        # 1. 调用父类初始化，获取通用的 KV cache spec 信息
        # 2. 从模型配置中提取注意力头数、KV head 数、head 维度
        # 3. 检查是否启用 CUDA Graph（影响 2D/3D 阈值选择）
        # 4. 计算 2D/3D kernel 切换阈值并调整为 CUDA Graph 兼容大小
        # 5. 预分配 3D kernel 并行分段 softmax 所需的 GPU 缓冲区
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

        self.block_size = kv_cache_spec.block_size

        model_config = vllm_config.model_config
        # Compatible with models with non-uniform per-layer head counts.
        self.num_heads_q = get_num_attention_heads_from_layers(
            vllm_config, layer_names
        ) or model_config.get_num_attention_heads(vllm_config.parallel_config)
        self.num_heads_kv = model_config.get_num_kv_heads(vllm_config.parallel_config)
        self.headdim = model_config.get_head_size()

        # Check if CUDA Graphs are enabled for decode
        self.decode_cudagraph_enabled = (
            self.vllm_config.compilation_config.cudagraph_mode
            in (
                CUDAGraphMode.FULL_AND_PIECEWISE,
                CUDAGraphMode.FULL_DECODE_ONLY,
                CUDAGraphMode.FULL,
            )
        )

        # 中文注释：计算 2D kernel 和 3D kernel 的切换阈值。
        # 2D kernel 的启动网格为 (num_q_blocks, num_heads_kv)，
        # 其中 num_q_blocks 的下界是序列数。为了保证网格大小达到
        # MIN_LAUNCH_GRID_SIZE_2D，需要序列数 >= 128 / num_heads_kv。
        # 当 batch 序列数不足时，改用 3D kernel（按 Q block 维度展开）。
        # The launch grid for the 2D kernel is defined as (num_q_blocks, num_heads_kv).
        # A lower bound for num_q_blocks is the number of sequences.
        # To ensure the minimum launch grid size is achieved, the number of sequences
        # must be at least equal to the threshold below.
        # If this threshold is not reached (i.e., the batch size is not large enough),
        # the 3D kernel will be selected instead.
        self.seq_threshold_3D = MIN_LAUNCH_GRID_SIZE_2D // self.num_heads_kv

        # Modify the threshold if needed.
        if self.decode_cudagraph_enabled:
            capture_sizes = self.vllm_config.compilation_config.cudagraph_capture_sizes
            assert capture_sizes, "CUDA Graphs enabled but no capture sizes specified."

            # 中文注释：当启用 CUDA Graph 时，将阈值调整为最接近的 CUDA Graph
            # 捕获大小。这是因为 CUDA Graph 捕获时固定了启动参数，
            # 必须保证阈值与某个捕获大小一致，才能确保图的执行路径正确。
            # Select the CUDA Graph capture size closest to self.seq_threshold_3D
            # as threshold. This ensures that each captured graph covers the
            # correct execution path.
            self.seq_threshold_3D = min(
                capture_sizes,
                key=lambda x: abs(x - self.seq_threshold_3D),
            )

        self.num_par_softmax_segments = NUM_PAR_SOFTMAX_SEGMENTS
        # 中文注释：预分配 3D kernel 并行分段 softmax 所需的 GPU 缓冲区。
        # head_size 向上取整到 2 的幂次，以满足 Triton kernel 对齐要求。
        # 这些缓冲区在所有 batch 之间复用，避免每次推理的内存分配开销。
        headdim_padded = next_power_of_2(self.headdim)
        self.softmax_segm_output = torch.empty(
            (
                self.seq_threshold_3D,
                self.num_heads_q,
                self.num_par_softmax_segments,
                headdim_padded,
            ),
            dtype=torch.float32,
            device=device,
        )
        self.softmax_segm_max = torch.empty(
            (self.seq_threshold_3D, self.num_heads_q, self.num_par_softmax_segments),
            dtype=torch.float32,
            device=device,
        )
        self.softmax_segm_expsum = torch.empty(
            (self.seq_threshold_3D, self.num_heads_q, self.num_par_softmax_segments),
            dtype=torch.float32,
            device=device,
        )

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> TritonAttentionMetadata:
        """为 CUDA Graph 捕获构建特殊的元数据。

        CUDA Graph 捕获的特殊要求：
        - 捕获时固定的执行路径（kernel 选择、启动参数等不能变化）
        - 捕获时间应尽量短（避免长时间阻塞 GPU）
        - seq_lens 设为 1 可以大幅减少 kernel 执行的 token 数，
          从而缩短图捕获时间，同时不影响图的正确性（图只记录
          执行模式，实际执行时会使用真实的 seq_lens）
        """
        attn_metadata = self.build(0, common_attn_metadata)
        # When doing full graph capture, setting seq_lens to
        # max_model_len will cause graph capture to be extremely
        # slow, so here we set it to 1.
        attn_metadata.seq_lens.fill_(1)
        return attn_metadata

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> TritonAttentionMetadata:
        """核心构建方法：将通用注意力元数据转换为 Triton 特定的元数据。

        构建流程：
        1. 从 CommonAttentionMetadata 提取基本的 token/序列信息
           （这些信息已由 GPUModelRunner 预处理好，包含页表、slot mapping 等）
        2. 判断是否使用级联注意力（当 common_prefix_len > 0 时）
        3. 如果使用级联注意力，构建前缀/后缀的 KV 长度信息：
           - cu_prefix_query_lens: 前缀部分的 query 累积长度
           - prefix_kv_lens: 公共前缀的 KV 长度
           - suffix_kv_lens: 每个序列去掉公共前缀后的剩余 KV 长度
        4. 组装为 TritonAttentionMetadata 对象，包含：
           - 基本的 token/序列信息
           - 2D/3D kernel 切换阈值和并行 softmax 缓冲区
           - 级联注意力相关字段

        参数说明：
        - common_prefix_len: 多个序列共享的公共前缀长度（通常来自 prefix cache）
        - common_attn_metadata: 通用注意力元数据，包含页表、slot mapping 等
        - fast_build: 是否快速构建（跳过非必要的计算）
        """
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        max_query_len = common_attn_metadata.max_query_len

        max_seq_len = common_attn_metadata.max_seq_len
        query_start_loc = common_attn_metadata.query_start_loc
        seq_lens = common_attn_metadata.seq_lens
        block_table_tensor = common_attn_metadata.block_table_tensor
        slot_mapping = common_attn_metadata.slot_mapping

        use_cascade = common_prefix_len > 0

        if use_cascade:
            cu_prefix_query_lens = torch.tensor(
                [0, num_actual_tokens], dtype=torch.int32, device=self.device
            )
            prefix_kv_lens = torch.tensor(
                [common_prefix_len], dtype=torch.int32, device=self.device
            )
            suffix_kv_lens = common_attn_metadata.seq_lens.cpu() - common_prefix_len
            suffix_kv_lens = suffix_kv_lens.to(self.device)
        else:
            cu_prefix_query_lens = None
            prefix_kv_lens = None
            suffix_kv_lens = None
            prefix_scheduler_metadata = None

        attn_metadata = TritonAttentionMetadata(
            num_actual_tokens=num_actual_tokens,
            max_query_len=max_query_len,
            query_start_loc=query_start_loc,
            max_seq_len=max_seq_len,
            seq_lens=seq_lens,
            block_table=block_table_tensor,
            slot_mapping=slot_mapping,
            use_cascade=use_cascade,
            common_prefix_len=common_prefix_len,
            cu_prefix_query_lens=cu_prefix_query_lens,
            prefix_kv_lens=prefix_kv_lens,
            suffix_kv_lens=suffix_kv_lens,
            prefix_scheduler_metadata=prefix_scheduler_metadata,
            seq_threshold_3D=self.seq_threshold_3D,
            num_par_softmax_segments=self.num_par_softmax_segments,
            softmax_segm_output=self.softmax_segm_output,
            softmax_segm_max=self.softmax_segm_max,
            softmax_segm_expsum=self.softmax_segm_expsum,
        )
        return attn_metadata


# 中文注释：Triton 注意力后端注册类。
# 继承 AttentionBackend 抽象基类，声明 Triton 后端的能力和支持范围。
# vLLM 的后端选择机制会根据这些声明来决定是否可以使用此后端。
#
# 后端选择机制：
# 1. vLLM 根据硬件平台、模型配置、用户指定的后端名称来选择注意力后端
# 2. 每个后端通过类变量（supported_dtypes, supported_kv_cache_dtypes 等）
#    声明自己的能力
# 3. 选择时会检查后端是否支持当前的计算精度、KV cache 类型、block size 等
# 4. Triton 后端是"万能后备"——它支持所有平台、所有精度，但性能可能
#    不如针对特定硬件优化的后端（如 FlashAttention for CUDA）
#
# 核心设计：forward_includes_kv_cache_update = False
# 这意味着 KV cache 更新和注意力计算是分离的两个步骤：
#   Step 1: GPUModelRunner 调用 do_kv_cache_update() 将 K/V 写入 cache
#   Step 2: GPUModelRunner 调用 forward() 从 cache 读取 K/V 并计算注意力
# 这种分离设计的优势：
# - 两个步骤可以分别进行 CUDA Graph 捕获
# - KV cache 更新可以跨层共享（如 KV sharing）
# - 更灵活的内存管理（写入和读取可以交错进行）
class TritonAttentionBackend(AttentionBackend):
    # 中文注释：后端支持的计算数据类型。
    # 相比其他后端，Triton 额外支持 float32（精度最高但速度最慢）。
    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ]
    # 中文注释：后端支持的 KV cache 数据类型。
    # 除标准的 float16/bfloat16 外，还支持：
    # - fp8 系列：FP8 量化 KV cache，节省显存
    # - int8_per_token_head / fp8_per_token_head：per-token-per-head 量化，
    #   每个 token 的每个 head 有独立的 scale，精度更高但开销更大
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
        "fp8_e5m2",
        "int8_per_token_head",
        "fp8_per_token_head",
    ]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        # 中文注释：返回支持的 KV cache block 大小。
        # MultipleOf(16) 表示 block_size 必须是 16 的倍数。
        # Triton kernel 内部按 16 元素的 tile 进行加载，
        # 非 16 的倍数会导致越界或效率下降。
        return [MultipleOf(16)]

    @classmethod
    def supports_block_size(cls, block_size: int | None) -> bool:
        """检查是否支持指定的 block size。

        要求 block_size 是 16 的倍数，因为 Triton kernel 内部按 16 元素的
        tile 进行加载，非 16 的倍数会导致越界或效率下降。
        """
        if block_size is None:
            return True
        return block_size % 16 == 0

    # 中文注释：关键设计标志——KV cache 更新不在 forward() 中执行。
    # 设为 False 表示 GPUModelRunner 会先单独调用 do_kv_cache_update()
    # 将 K/V 写入 cache，然后再调用 forward() 执行注意力计算。
    # 这种分离设计使得两个步骤可以分别进行 CUDA Graph 捕获。
    forward_includes_kv_cache_update: bool = False

    @staticmethod
    def get_name() -> str:
        return "TRITON_ATTN"

    @classmethod
    def supports_batch_invariance(cls) -> bool:
        """检查是否支持 batch 不变性。

        batch 不变性意味着：无论 batch 大小如何变化，每个 token 的计算
        结果都是一样的。这对于 CUDA Graph 捕获非常重要，因为 Graph 捕获时
        的 batch 大小可能与实际执行时不同。

        Triton 后端支持 batch 不变性，因为它使用 PagedAttention，
        每个 token 的计算不依赖于 batch 中其他 token 的存在。
        """
        return True

    @staticmethod
    def get_impl_cls() -> type["TritonAttentionImpl"]:
        return TritonAttentionImpl

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        """定义 KV cache 的逻辑张量形状。

        返回形状：(num_blocks, 2, block_size, num_kv_heads, head_size)

        各维度含义：
        - num_blocks: 总的物理 block 数，由 KV cache manager 分配
        - 2: K 和 V 各占一半，通过 unbind(dim=1) 可分离为 key_cache 和 value_cache
        - block_size: 每个 block 包含的 token 数（通常为 16/32/64 等）
        - num_kv_heads: KV head 数（GQA 时通常小于 Q head 数）
        - head_size: 每个 head 的维度

        特殊情况：per-token-head 量化模式
        当使用 per-token-head 量化时，每个 head 需要额外存储一个 float32 scale 值。
        为了高效存储，scale 值被内联到 head 维度的末尾（padding 区域），
        使得 head_size 变为 head_size + sizeof(float32)/sizeof(cache_dtype)。
        后续通过 typed view 提取 data[:head_size] 和 scale[head_size:]。
        这种内联存储的优势：scale 与数据在内存中相邻，访问效率更高。

        PagedAttention 的核心思想：
        KV cache 不需要在物理显存上连续存放。每个 block 是独立的，
        通过 block_table 将逻辑 block id 映射到物理显存位置。
        这样不同请求的 KV cache 可以交错存放在显存中，提高显存利用率。
        """
        if block_size % 16 != 0:
            raise ValueError("Block size must be a multiple of 16.")
        if kv_cache_uses_per_token_head_scales(cache_dtype_str):
            # Pad head_size by sizeof(float32)/sizeof(cache_dtype) so
            # the per-head scale fits inline.  The backend extracts
            # data[:head_size] and scale[head_size:] via typed views.
            from vllm.utils.torch_utils import (
                STR_DTYPE_TO_TORCH_DTYPE,
                get_dtype_size,
            )

            cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_dtype_str]
            scale_pad = get_dtype_size(torch.float32) // get_dtype_size(cache_dtype)
            return (num_blocks, 2, block_size, num_kv_heads, head_size + scale_pad)
        return (num_blocks, 2, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        """返回 KV cache 的物理内存布局排列顺序。

        stride_order 是一个排列（permutation），指示如何从 get_kv_cache_shape()
        返回的逻辑形状变换到实际的物理内存布局。

        两种主要布局模式：
        1. NHD layout（默认）：
           - 逻辑形状: (num_blocks, 2, block_size, num_kv_heads, head_size)
           - 物理排列: Block -> KV_half -> Token -> Head -> Dim
           - 内存访问模式: 连续访问同一 token 的不同 head
           - 适用场景: batch 较大时，多个 head 的数据在同一 cache line 中

        2. HND layout：
           - 逻辑形状: (num_blocks, num_kv_heads, 2, block_size, head_size)
           - 物理排列: Block -> Head -> KV_half -> Token -> Dim
           - 内存访问模式: 连续访问同一 head 的不同 token
           - 适用场景: 长序列时，同一 head 的连续 token 数据在内存中相邻

        stride_order 的计算原理：
        - 输入: 逻辑形状的维度顺序 [0, 1, 2, 3, 4]
        - 输出: 物理内存中各维度的排列顺序
        - 例如 NHD 的 stride_order = (0, 1, 2, 3, 4) 表示逻辑和物理一致
        - HND 的 stride_order = (0, 1, 3, 2, 4) 表示将 block_size 和
          num_kv_heads 维度交换

        不同 layout 对 kernel 的内存访问模式和性能有显著影响，
        选择合适的 layout 可以显著提升 KV cache 的访问效率。
        """
        # `stride_order` indicates the permutation that gets
        # us from `get_kv_cache_shape` to the actual memory layout we want.
        cache_layout = get_kv_cache_layout()
        if cache_layout == "NHD" and include_num_layers_dimension:
            # (num_blocks, num_layers, 2, block_size, num_kv_heads, head_size)
            return (1, 0, 2, 3, 4, 5)
        elif cache_layout == "NHD":
            stride_order = (0, 1, 2, 3, 4)
        elif cache_layout == "HND" and include_num_layers_dimension:
            # (num_blocks, num_kv_heads, num_layers, 2, block_size, head_size)
            return (1, 4, 0, 2, 3, 5)
        elif cache_layout == "HND":
            stride_order = (0, 1, 3, 2, 4)
        else:
            raise ValueError(f"Unknown cache layout: {cache_layout}")
        return stride_order

    @staticmethod
    def use_cascade_attention(*args, **kwargs) -> bool:
        # 中文注释：Triton 后端当前不使用级联注意力优化。
        # 级联注意力（Cascade Attention）用于多个序列共享长公共前缀的场景，
        # 通过共享前缀的 KV 计算来减少重复工作。Triton 后端暂未实现此优化。
        return False

    @staticmethod
    def get_builder_cls() -> type["TritonAttentionMetadataBuilder"]:
        return TritonAttentionMetadataBuilder

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        """检查是否支持指定的 head size。

        要求 head_size >= 32，因为 Triton kernel 内部使用 32 元素的向量化
        加载指令，较小的 head size 会导致效率下降或无法正常工作。
        """
        return head_size >= 32

    @classmethod
    def supports_mm_prefix(cls) -> bool:
        """检查是否支持多模态前缀（mm_prefix）。

        多模态前缀用于处理包含图像/视频等多模态 token 的输入。
        这些 token 的位置编码和注意力 mask 可能与文本 token 不同。
        """
        return True

    @classmethod
    def supports_sink(cls) -> bool:
        """检查是否支持 sink token（流式 LLM 的起始 token）。

        Sink token 是 StreamingLLM 技术中的概念：
        - 在滑动窗口注意力中，保留序列开头的几个 token（sink token）
        - 这些 token 的 KV cache 不会被丢弃
        - 用于稳定注意力计算，避免信息丢失
        """
        return True

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        """检查是否支持指定的注意力类型。

        Triton 后端支持所有四种注意力类型：
        1. DECODER: 自回归解码器注意力（因果注意力 + KV cache）
        2. ENCODER: 编码器注意力（双向注意力，无 KV cache）
        3. ENCODER_ONLY: 纯编码器模型（如 BERT、ViT）
        4. ENCODER_DECODER: 编码器-解码器模型（如 T5、BART）
        """
        return attn_type in (
            AttentionType.DECODER,
            AttentionType.ENCODER,
            AttentionType.ENCODER_ONLY,
            AttentionType.ENCODER_DECODER,
        )

    @classmethod
    def supports_alibi_sqrt(cls) -> bool:
        """检查是否支持 ALiBi sqrt 变体。

        ALiBi sqrt 是 ALiBi 位置编码的一种变体，对 slopes 取平方根。
        用于某些模型（如 MPT）的位置编码。
        """
        return True

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        """检查是否支持指定的计算能力。

        Triton 后端支持所有计算能力，因为它是通过 Triton JIT 编译的，
        不依赖特定的 CUDA 计算能力。Triton 会根据目标硬件自动选择
        最优的指令集和优化策略。
        """
        return True


# 中文注释：Triton 注意力计算实现类。
# 继承 AttentionImpl，实现了基于 Triton kernel 的 PagedAttention 前向传播。
#
# 核心职责：
# 1. forward(): 执行注意力计算（decode + prefill 统一路径）
#    - 将 Q 与 KV cache 中的 K/V 进行注意力计算
#    - 支持 GQA（Grouped Query Attention）、滑动窗口、ALiBi 等特性
#    - 根据 batch 大小自动选择 2D 或 3D kernel
# 2. do_kv_cache_update(): 将当前 step 的 K/V 写入 KV cache
#    - 通过 slot_mapping 定位每个 token 在 cache 中的物理位置
#    - 支持 FP8 量化和 per-token-head 量化
# 3. _forward_encoder_attention(): encoder 自注意力（双向，无 KV cache）
# 4. do_rope_and_kv_cache_update(): 融合 RoPE 位置编码与 KV cache 更新
#    - 仅在 ROCm 平台可用（通过 aiter_ops 调用）
#
# 关键设计：
# - forward_includes_kv_cache_update = False（在 Backend 中设置），
#   意味着 KV cache 更新和注意力计算是分离的两个步骤
# - 支持 per-token-head 量化：每个 head 有独立的 scale，内联存储在
#   KV cache 的 head 维度 padding 中
# - 支持 fused output quantization：注意力输出在 kernel 内部直接量化为 FP8，
#   避免额外的量化 kernel 调用
#
# 执行路径选择逻辑：
# 当 attn_type 为 ENCODER 或 ENCODER_ONLY 时：
#   -> _forward_encoder_attention()（双向注意力，无 KV cache）
# 当使用 per-token-head 量化时：
#   -> do_kv_cache_update() 使用 triton_reshape_and_cache_flash_per_token_head_quant
#   -> forward() 使用 k_scale_cache/v_scale_cache
# 否则（标准路径）：
#   -> do_kv_cache_update() 使用 triton_reshape_and_cache_flash
#   -> forward() 使用 layer._k_scale/layer._v_scale
class TritonAttentionImpl(AttentionImpl):
    # Per-token-head quant: scale views carved from inline head padding.
    # 中文注释：per-token-head 量化的 scale 缓存视图。
    # 这些是从 KV cache 的 head 维度 padding 区域中创建的 strided view，
    # 避免了额外的 scale 存储开销。初始化后在所有 batch 间复用。
    _k_scale_cache: torch.Tensor | None = None
    _v_scale_cache: torch.Tensor | None = None

    def _ensure_scale_caches(self, kv_cache: torch.Tensor) -> None:
        """Extract per-head scale views from the padded head dimension.

        The KV cache shape is ``(num_blocks, 2, block_size, nkv, hs+pad)``
        where ``pad = sizeof(float32) / sizeof(cache_dtype)``.  The last
        ``pad`` elements of each head hold one float32 scale.  We create
        strided float32 views over those bytes.

        Scale shape: ``(num_blocks, block_size, num_kv_heads)``
        """
        # 中文注释：从 KV cache 的 head 维度 padding 区域提取 per-head scale 视图。
        #
        # 内存布局示意（以 fp8 为例，dtype_sz=1, scale_pad=4）：
        #   KV cache shape: (num_blocks, 2, block_size, nkv, hs+4)
        #   每个 head 的内存布局: [data_byte_0, data_byte_1, ..., data_byte_hs-1,
        #                         scale_byte_0, scale_byte_1, scale_byte_2, scale_byte_3]
        #
        # 实现原理：
        # 1. 获取 kv_cache 的底层 untyped_storage（共享同一块 GPU 显存）
        # 2. 创建一个 float32 类型的 tensor 引用同一块内存（base_f32）
        # 3. 通过 torch.as_strided 创建 strided view，使其恰好覆盖
        #    每个 head 末尾的 4 字节（一个 float32 scale 值）
        # 4. 分别为 K 和 V 创建独立的 scale 视图
        #
        # 这种内联存储方式的优势：
        # - scale 与数据在内存中相邻，访问效率更高
        # - strided view 的创建是零拷贝的，不占用额外显存
        # - scale 值随 KV cache 一起被管理（分配、释放、prefix cache 复用）
        # - 避免了额外的 scale 张量管理开销
        #
        # stride 计算说明（以 float32 为单位）：
        # - full_block_f32: 相邻 block 之间的 stride（2 * kv_half_bytes / 4）
        # - slot_f32: 同一 block 内相邻 slot 之间的 stride
        # - head_f32: 同一 slot 内相邻 head 之间的 stride
        # - scale_off_f32: 从 head 起始到 scale 位置的偏移
        if self._k_scale_cache is not None:
            return
        from vllm.utils.torch_utils import get_dtype_size

        num_blocks, _, block_size, nkv, padded_hs = kv_cache.shape
        dtype_sz = kv_cache.element_size()
        scale_pad = get_dtype_size(torch.float32) // dtype_sz  # e.g. 4
        hs = padded_hs - scale_pad

        raw = kv_cache.untyped_storage()
        base_f32 = torch.tensor([], dtype=torch.float32, device=kv_cache.device).set_(
            raw
        )

        # In the raw bytes, each (block, kv_half, slot, head) occupies
        # padded_hs * dtype_sz bytes.  The scale float32 sits at byte
        # offset hs * dtype_sz within that region.
        kv_half_bytes = block_size * nkv * padded_hs * dtype_sz
        full_block_f32 = 2 * kv_half_bytes // 4  # stride between blocks
        slot_f32 = nkv * padded_hs * dtype_sz // 4  # stride between slots
        head_f32 = padded_hs * dtype_sz // 4  # stride between heads
        scale_off_f32 = hs * dtype_sz // 4  # offset to scale within head

        # K scales: kv_half=0
        self._k_scale_cache = torch.as_strided(
            base_f32,
            size=(num_blocks, block_size, nkv),
            stride=(full_block_f32, slot_f32, head_f32),
            storage_offset=scale_off_f32,
        )
        self._k_scale_cache.fill_(1.0)

        # V scales: kv_half=1, offset by kv_half_bytes
        v_base_f32 = kv_half_bytes // 4
        self._v_scale_cache = torch.as_strided(
            base_f32,
            size=(num_blocks, block_size, nkv),
            stride=(full_block_f32, slot_f32, head_f32),
            storage_offset=v_base_f32 + scale_off_f32,
        )
        self._v_scale_cache.fill_(1.0)

    def fused_output_quant_supported(self, quant_key: QuantKey):
        """检查是否支持融合输出量化。

        融合输出量化的优势：
        当返回 True 时，注意力计算的输出会在 kernel 内部直接量化为 FP8，
        避免额外的量化 kernel 调用和显存读写。

        支持条件：
        - 目前仅支持 FP8 静态对称量化（kFp8StaticTensorSym）
        - 需要 Triton kernel 支持输出量化（通过 output_scale 参数）

        工作原理：
        - 在 unified_attention() 内部，计算完 attention_output 后
        - 直接应用量化公式：output = round(output / scale)
        - 将量化后的结果写入 output 张量
        - 避免了额外的量化 kernel 调用和显存读写
        """
        return quant_key == kFp8StaticTensorSym

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: int | None = None,
        sinks: torch.Tensor | None = None,
        use_alibi_sqrt: bool = False,
        chunk_lookback: int = -1,
    ) -> None:
        """初始化 Triton 注意力实现。

        构建流程（按顺序）：
        1. 保存基本注意力参数（head 数、维度、缩放因子等）
        2. 处理 ALiBi 位置编码的 slopes（如果启用）
        3. 配置滑动窗口参数（encoder 和 decoder 的窗口范围不同）
        4. 计算 GQA 的 query-per-kv 比率（每多少个 Q head 共享一个 KV head）
        5. 确定 KV 量化模式（fp8、per-token-head 等）
        6. 配置 Tensor Descriptor（用于 Intel XPU 的硬件 2D block 读取）

        参数说明：
        - num_heads: Q head 数量
        - head_size: 每个 head 的维度
        - scale: softmax 缩放因子（通常为 1/sqrt(head_size)）
        - num_kv_heads: KV head 数量（GQA 时小于 num_heads）
        - alibi_slopes: ALiBi 位置编码的 slopes，None 表示不使用 ALiBi
        - sliding_window: 滑动窗口大小，None 表示不使用滑动窗口
        - kv_cache_dtype: KV cache 的数据类型（如 "auto", "fp8" 等）
        - logits_soft_cap: logits soft cap 值，用于 Gemma 等模型
        - attn_type: 注意力类型（DECODER/ENCODER/ENCODER_ONLY/ENCODER_DECODER）
        - sinks: sink token 的注意力权重，用于 streaming LLM
        - use_alibi_sqrt: 是否对 ALiBi slopes 取平方根
        - chunk_lookback: 分块注意力的回看块数
        """
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        if sliding_window is None:
            self.sliding_window = (-1, -1)
        elif attn_type in (AttentionType.ENCODER, AttentionType.ENCODER_ONLY):
            self.sliding_window = (sliding_window - 1, sliding_window - 1)
        else:
            self.sliding_window = (sliding_window - 1, 0)
        self.kv_cache_dtype = kv_cache_dtype
        if logits_soft_cap is None:
            # In flash-attn, setting logits_soft_cap as 0 means no soft cap.
            logits_soft_cap = 0
        self.logits_soft_cap = logits_soft_cap
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name

        # 中文注释：GQA（Grouped Query Attention）的比率。
        # 每个 KV head 被多少个 Q head 共享。例如 LLaMA-70B 中
        # num_heads=64, num_kv_heads=8, 则 num_queries_per_kv=8。
        # 当比率为 1 时退化为标准 MHA。
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        self.attn_type = attn_type
        self.fp8_dtype = current_platform.fp8_dtype()

        self.sinks = sinks
        if sinks is not None:
            assert sinks.shape[0] == num_heads, (
                "Sinks must have the same number of heads as the number of "
                f"heads in the layer. Sinks shape: {sinks.shape}, "
                f"num_heads: {num_heads}."
            )
        self.use_alibi_sqrt = use_alibi_sqrt
        self.chunk_lookback = chunk_lookback
        self.supports_quant_query_input = current_platform.is_cuda()

        self._kv_quant_mode = get_kv_quant_mode(kv_cache_dtype)
        self._is_per_token_head_quant = self._kv_quant_mode.is_per_token_head

        # Enable tensor descriptors for Q/K/V load/store on platforms that
        # benefit from HW 2D block reads (Intel Xe2/Xe3).  The dead branch
        # is eliminated at Triton compile time, so other platforms see
        # zero cost when TD is off.
        #
        # ``VLLM_TRITON_ATTN_USE_TD`` is tri-state:
        #   - unset (None): auto-select (TD on for XPU, off elsewhere),
        #   - ``1``: force TD on regardless of platform,
        #   - ``0``: force TD off regardless of platform (useful for A/B).
        td_override = envs.VLLM_TRITON_ATTN_USE_TD
        if td_override is None:
            self.use_td = current_platform.is_xpu()
        else:
            self.use_td = td_override

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with Paged Attention impl. in Triton.

        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache: shape =
                [num_blocks, 2, block_size, num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        # 中文注释：Triton 注意力前向传播的核心方法。
        #
        # 整体流程：
        # 1. 前置检查：验证输出量化格式、处理 profiling run、验证不使用 cascade
        # 2. 路径分派：根据 attn_type 和量化模式选择不同的处理路径
        #    a. Encoder 注意力 -> _forward_encoder_attention()
        #    b. per-token-head 量化 -> 使用内联 scale 缓存
        #    c. 标准 FP8/auto 量化 -> 使用 layer._k_scale/layer._v_scale
        # 3. 准备 Q/K/V 缓存：处理 FP8 dtype 转换、scale 展开
        # 4. 提取元数据：从 attn_metadata 中获取序列长度、页表等
        # 5. 调用 unified_attention() Triton kernel 执行实际的注意力计算
        #
        # 性能注意事项（来自原代码注释）：
        # 使用 piece-wise CUDA graphs 时，此方法在 eager-mode PyTorch 中执行。
        # 因此需要特别注意 CPU 开销，例如 view 和 slice 操作即使不触发 GPU op
        # 也可能很慢。应尽量减少此方法中的 PyTorch 操作。
        if output_block_scale is not None:
            raise NotImplementedError(
                "fused block_scale output quantization is not yet supported"
                " for TritonAttentionImpl"
            )

        if attn_metadata is None:
            # Profiling run.
            return output.fill_(0)

        assert attn_metadata.use_cascade is False

        # IMPORTANT!
        # NOTE(woosuk): With piece-wise CUDA graphs, this method is executed in
        # eager-mode PyTorch. Thus, we need to be careful about any CPU overhead
        # in this method. For example, `view` and `slice` (or `[:n]`) operations
        # are surprisingly slow even in the case they do not invoke any GPU ops.
        # Minimize the PyTorch ops in this method as much as possible.
        # Whenever making a change in this method, please benchmark the
        # performance to make sure it does not introduce any overhead.

        num_actual_tokens = attn_metadata.num_actual_tokens

        # Handle encoder attention differently - no KV cache needed
        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            # For encoder attention,
            # we use direct Q, K, V tensors without caching
            return self._forward_encoder_attention(
                query[:num_actual_tokens],
                key[:num_actual_tokens],
                value[:num_actual_tokens],
                output[:num_actual_tokens],
                attn_metadata,
                layer,
            )

        # Per-token-head quantized KV cache: use separate scale caches.
        if self._is_per_token_head_quant:
            # 中文注释：per-token-head 量化路径。
            # 使用内联存储在 KV cache head padding 中的 scale 值。
            # 每个 token 的每个 head 有独立的 scale，精度更高。
            self._ensure_scale_caches(kv_cache)
            key_cache, value_cache = kv_cache.unbind(1)
            if key_cache.dtype == torch.uint8:
                key_cache = key_cache.view(self.fp8_dtype)
                value_cache = value_cache.view(self.fp8_dtype)
            q_descale = None
            k_descale = None
            v_descale = None
            k_scale_cache = self._k_scale_cache
            v_scale_cache = self._v_scale_cache
        # FP8 per-tensor / auto path (original flow).
        else:
            # 中文注释：标准 FP8 per-tensor 量化路径。
            # 使用 layer 级别的 _k_scale/_v_scale 进行反量化。
            # 这些 scale 是整个层共享的，精度略低但开销更小。
            key_cache, value_cache = kv_cache.unbind(1)
            if (
                is_quantized_kv_cache(self.kv_cache_dtype)
                and key_cache.dtype != self.fp8_dtype
            ):
                key_cache = key_cache.view(self.fp8_dtype)
                value_cache = value_cache.view(self.fp8_dtype)
            descale_shape = (
                attn_metadata.query_start_loc.shape[0] - 1,
                key_cache.shape[2],
            )
            q_descale = (
                layer._q_scale
                if (
                    self._kv_quant_mode == KVQuantMode.FP8_PER_TENSOR
                    and query.dtype == self.fp8_dtype
                )
                else None
            )
            k_descale = layer._k_scale.expand(descale_shape)
            v_descale = layer._v_scale.expand(descale_shape)
            k_scale_cache = None
            v_scale_cache = None

        # 中文注释：提取注意力计算所需的元数据字段。
        # - cu_seqlens_q: 每个序列的 query 起始位置（cumsum 格式）
        # - seqused_k: 每个序列的 KV 长度（用于 PagedAttention 的边界检查）
        # - max_seqlen_q: 最大 query 长度（用于 kernel 的 tile 大小选择）
        # - max_seqlen_k: 最大 KV 长度
        # - block_table: 页表，逻辑 block id -> 物理 block id
        cu_seqlens_q = attn_metadata.query_start_loc
        seqused_k = attn_metadata.seq_lens
        max_seqlen_q = attn_metadata.max_query_len
        max_seqlen_k = attn_metadata.max_seq_len
        block_table = attn_metadata.block_table

        # 中文注释：3D kernel 并行分段 softmax 相关的元数据。
        # 这些缓冲区由 Builder 预分配，在所有 batch 间复用。
        seq_threshold_3D = attn_metadata.seq_threshold_3D
        num_par_softmax_segments = attn_metadata.num_par_softmax_segments
        softmax_segm_output = attn_metadata.softmax_segm_output
        softmax_segm_max = attn_metadata.softmax_segm_max
        softmax_segm_expsum = attn_metadata.softmax_segm_expsum

        mm_prefix_range_tensor = attn_metadata.mm_prefix_range_tensor

        # 中文注释：调用 Triton kernel 执行实际的注意力计算。
        # unified_attention() 是核心 kernel，实现了 PagedAttention：
        # - 通过 block_table 从 KV cache 中读取 K/V
        # - 计算 Q*K^T 的注意力分数
        # - 应用 softmax（支持 2D/3D 两种路径）
        # - 计算 attention_output = softmax(scores) * V
        #
        # kernel 内部会根据 seq_threshold_3D 自动选择 2D 或 3D 路径：
        # - 2D kernel: 网格为 (num_q_blocks, num_heads_kv)，适合序列数较多的 batch
        # - 3D kernel: 网格为 (num_q_blocks, num_heads_kv, num_segments)，
        #   按 Q block 维度展开并行度，适合序列数较少的 batch
        unified_attention(
            q=query[:num_actual_tokens],
            k=key_cache,
            v=value_cache,
            out=output[:num_actual_tokens],
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            seqused_k=seqused_k,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=self.scale,
            causal=True,
            alibi_slopes=self.alibi_slopes,
            use_alibi_sqrt=self.use_alibi_sqrt,
            window_size=self.sliding_window,
            block_table=block_table,
            softcap=self.logits_soft_cap,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
            seq_threshold_3D=seq_threshold_3D,
            num_par_softmax_segments=num_par_softmax_segments,
            softmax_segm_output=softmax_segm_output,
            softmax_segm_max=softmax_segm_max,
            softmax_segm_expsum=softmax_segm_expsum,
            sinks=self.sinks,
            output_scale=output_scale,
            mm_prefix_range=mm_prefix_range_tensor,
            kv_quant_mode=self._kv_quant_mode,
            k_scale_cache=k_scale_cache,
            v_scale_cache=v_scale_cache,
            chunk_lookback=self.chunk_lookback,
            use_td=self.use_td,
        )

        return output

    def _forward_encoder_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        layer: torch.nn.Module,
    ) -> torch.Tensor:
        """Forward pass for encoder attention without KV cache.

        Args:
            query: shape = [num_encoder_tokens, num_heads, head_size]
            key: shape = [num_encoder_tokens, num_kv_heads, head_size]
            value: shape = [num_encoder_tokens, num_kv_heads, head_size]
            output: shape = [num_encoder_tokens, num_heads, head_size]
            attn_metadata: Encoder attention metadata
            layer: The attention layer
        """
        # 中文注释：Encoder 注意力前向传播。
        # 与 Decoder 注意力的关键区别：
        # 1. 不使用 KV cache：encoder 的 K/V 是当前输入的，不需要缓存历史
        # 2. 双向注意力（is_causal=False）：每个 token 可以看到所有其他 token
        # 3. 使用 context_attention_fwd 而非 unified_attention：
        #    前者直接在 Q/K/V 上计算，后者通过 block_table 从 KV cache 读取
        #
        # 应用场景：
        # - Transformer encoder 部分的自注意力（如 BERT、ViT）
        # - Encoder-decoder 模型中 encoder 部分的注意力（如 T5、BART）
        #
        # 注意：不支持量化 KV cache，因为 encoder 不使用 KV cache。
        if is_quantized_kv_cache(self.kv_cache_dtype):
            raise NotImplementedError(
                "quantized KV cache is not supported for encoder attention"
            )

        # Use encoder-specific metadata for sequence information
        query_start_loc = attn_metadata.query_start_loc
        seq_lens = attn_metadata.seq_lens
        max_query_len = attn_metadata.max_query_len

        # Call flash attention directly on Q, K, V tensors
        context_attention_fwd(
            q=query,
            k=key,
            v=value,
            o=output,
            b_start_loc=query_start_loc,
            b_seq_len=seq_lens,
            max_input_len=max_query_len,
            is_causal=False,  # Encoder attention is bidirectional
            softmax_scale=self.scale,
            sliding_window_q=self.sliding_window[0],
            sliding_window_k=self.sliding_window[1],
        )
        return output

    def do_kv_cache_update(
        self,
        layer: AttentionLayer,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ):
        """将当前 step 的 K/V 写入 KV cache。

        这是 KV cache 更新的核心方法，由 GPUModelRunner 在 forward() 之前调用。
        执行流程：
        1. 对于 encoder 注意力，直接返回（不使用 KV cache）
        2. 根据量化模式选择不同的写入路径：
           a. per-token-head 量化：使用 triton_reshape_and_cache_flash_per_token_head_quant
              - 同时写入数据和 scale（scale 内联在 head padding 中）
           b. 标准量化/auto：使用 triton_reshape_and_cache_flash
              - 数据写入 KV cache，scale 存储在 layer._k_scale/layer._v_scale
        3. 通过 slot_mapping 定位每个 token 在 KV cache 中的物理位置

        参数说明：
        - layer: 注意力层，包含量化 scale 等参数
        - key: 当前 step 的 K 张量，shape = [num_tokens, num_kv_heads, head_size]
        - value: 当前 step 的 V 张量，shape = [num_tokens, num_kv_heads, head_size]
        - kv_cache: KV cache 张量，shape = [num_blocks, 2, block_size, num_kv_heads, head_size]
        - slot_mapping: 每个 token 对应的 KV cache 物理槽位地址，
          由 (block_id * block_size + offset_in_block) 计算得到

        PagedAttention 的写入机制：
        - 每个 token 通过 slot_mapping 找到其在 KV cache 中的物理位置
        - 不需要连续写入，不同 token 可以写入不同的物理 block
        - 写入由 Triton kernel 完成，支持高效的并行写入
        """
        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            # For encoder attention,
            # we use direct Q, K, V tensors without caching
            return
        # Reshape the input keys and values and store them in the cache.
        if self._is_per_token_head_quant:
            # 中文注释：per-token-head 量化路径。
            # 同时写入数据和 scale，scale 内联存储在 head padding 中。
            # 这种方式的优势：scale 与数据在内存中相邻，访问效率更高。
            self._ensure_scale_caches(kv_cache)
            key_cache, value_cache = kv_cache.unbind(1)
            if key_cache.dtype == torch.uint8:
                key_cache = key_cache.view(self.fp8_dtype)
                value_cache = value_cache.view(self.fp8_dtype)
            triton_reshape_and_cache_flash_per_token_head_quant(
                key,
                value,
                key_cache,
                value_cache,
                self._k_scale_cache,
                self._v_scale_cache,
                slot_mapping,
            )
            return
        # For decoder and cross-attention, use KV cache as before.
        # 中文注释：标准量化路径。
        # 数据写入 KV cache，scale 存储在 layer 级别的属性中。
        key_cache, value_cache = kv_cache.unbind(1)
        if is_quantized_kv_cache(self.kv_cache_dtype):
            key_cache = key_cache.view(self.fp8_dtype)
            value_cache = value_cache.view(self.fp8_dtype)
        triton_reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
        )

    def fused_rope_kvcache_supported(self):
        """检查是否支持融合的 RoPE + KV cache 更新。

        融合 RoPE 的优势：
        将 RoPE 位置编码计算和 KV cache 写入合并到一个 kernel 中，
        避免了中间结果的显存读写，减少 kernel 启动开销。

        限制条件：
        - 仅在 ROCm 平台可用（通过 aiter_ops）
        - 不支持 per-token-head 量化（需要额外处理 scale）
        """
        if self._is_per_token_head_quant:
            return False
        return rocm_aiter_ops.is_enabled()

    def do_rope_and_kv_cache_update(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        is_neox: bool,
        kv_cache: torch.Tensor,
        layer_slot_mapping: torch.Tensor,
    ):
        """融合的 RoPE 位置编码 + KV cache 更新。

        将以下两个步骤合并到一个 kernel 中执行：
        1. 对 Q/K 应用 RoPE 位置编码
        2. 将 K/V 写入 KV cache

        这种融合方式的优势：
        - 减少中间结果的显存读写（RoPE 输出不需要先写入显存再读取）
        - 减少 kernel 启动开销（一个 kernel 替代两个）
        - 提高数据局部性（RoPE 和 cache 写入共享相同的内存访问模式）

        参数说明：
        - query: Q 张量（将被 RoPE 编码）
        - key: K 张量（将被 RoPE 编码并写入 cache）
        - value: V 张量（将被写入 cache）
        - positions: 每个 token 的位置索引
        - cos_sin_cache: 预计算的 cos/sin 值缓存
        - is_neox: 是否使用 NeoX 风格的 RoPE（GPT-NeoX/LLaMA 风格）
        - kv_cache: KV cache 张量
        - layer_slot_mapping: 每个 token 对应的 KV cache 物理槽位地址
        """
        key_cache, value_cache = kv_cache.unbind(1)
        flash_layout = True

        is_fp8_kv_cache = is_quantized_kv_cache(self.kv_cache_dtype)
        if is_fp8_kv_cache:
            key_cache = key_cache.view(self.fp8_dtype)
            value_cache = value_cache.view(self.fp8_dtype)

        rocm_aiter_ops.triton_rope_and_cache(
            query,
            key,
            value,
            positions,
            cos_sin_cache,
            is_neox,
            key_cache,
            value_cache,
            layer_slot_mapping,
            layer._k_scale,
            layer._v_scale,
            flash_layout,
            is_fp8_kv_cache,
        )
