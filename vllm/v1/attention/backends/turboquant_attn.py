# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TurboQuant attention backend for vLLM.

Prefill: Standard scaled dot-product attention on uncompressed K/V,
         then quantize K and store K+V into combined cache slot.
Decode:  Compute TQ attention scores from compressed cache,
         unpack FP16 values, softmax + weighted sum.

Cache layout (no leading 2 dimension):
  (num_blocks, block_size, num_kv_heads, slot_size)
  where slot_size = key_packed_size + value_fp16_size

Per-head per-position slot layout:
  [key_packed (kps bytes) | value_fp16 (D*2 bytes)]
  For turboquant_k3v4_nc head_dim=256: [100 bytes key | 512 bytes value] = 612
"""
# TurboQuant 注意力后端模块。
#
# TurboQuant 是一种 KV 缓存压缩技术，通过以下方式减少显存占用：
# 1. Key 压缩：使用 Hadamard 旋转 + Lloyd-Max 标量量化（MSE 量化）
#    - 先对 Key 向量应用 Hadamard 变换（正交变换，保持向量范数）
#    - 然后使用预计算的质心（centroids）进行标量量化
#    - 每个 key 值被映射到最近的质心索引（压缩存储）
# 2. Value 压缩：直接量化到低精度（如 4-bit）
#
# Prefill 流程：
# 1. 对未压缩的原始 Q/K/V 执行标准缩放点积注意力
# 2. 将 K/V 量化并存储到组合缓存槽中
#
# Decode 流程：
# 1. 从压缩缓存中读取量化数据
# 2. 在线反量化 K/V，计算注意力分数
# 3. 执行 softmax + 加权求和
#
# 缓存布局（无前导维度 2）：
#   (num_blocks, block_size, num_kv_heads, slot_size)
#   其中 slot_size = key_packed_size + value_fp16_size
#
# 每个头每个位置的槽布局：
#   [key_packed (kps bytes) | value_fp16 (D*2 bytes)]
#   例如 turboquant_k3v4_nc head_dim=256: [100 bytes key | 512 bytes value] = 612

import functools
import math
from dataclasses import dataclass
from typing import Any, ClassVar

import torch
import torch.nn.functional as F

from vllm.config import get_current_vllm_config
from vllm.config.cache import CacheDType
from vllm.model_executor.layers.quantization.turboquant.centroids import (
    get_centroids,
)
from vllm.triton_utils import triton
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadata,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.fa_utils import (
    get_flash_attn_version,
    is_flash_attn_varlen_func_available,
)
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.attention.ops.triton_turboquant_decode import (
    _tq_full_dequant_kv,
    _use_fp8_e4b15,
    triton_turboquant_decode_attention,
)
from vllm.v1.attention.ops.triton_turboquant_store import triton_turboquant_store
from vllm.v1.worker.workspace import (
    current_workspace_manager,
    is_workspace_manager_initialized,
)

_HAS_FLASH_ATTN = is_flash_attn_varlen_func_available()
if _HAS_FLASH_ATTN:
    from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func

# Continuation prefill: for small continuation chunks (q_len ≤ threshold),
# use the TQ decode kernel directly instead of full-dequant + flash_attn.
# do_kv_cache_update already stored all tokens to TQ cache, so the decode
# kernel can read them efficiently. This avoids O(cached_len) dequant work
# per continuation, eliminating the O(N²/chunk_size) collapse at long context.
# 延续 prefill 的阈值：对于较小的延续块（q_len ≤ 阈值），
# 直接使用 TQ decode 内核，而不是先完整反量化再用 flash_attn。
# do_kv_cache_update 已经将所有 token 存储到 TQ 缓存中，
# decode 内核可以直接高效读取。这避免了每次延续都需要 O(cached_len)
# 的反量化工作，消除了长上下文下 O(N^2/chunk_size) 的性能崩溃。
_CONTINUATION_DECODE_THRESHOLD = 128


def _build_hadamard(d: int, device_str: str) -> torch.Tensor:
    """Orthonormal Hadamard matrix (Sylvester construction), cached per (d, device).

    Precomputed D×D matrix enables matmul-based WHT — single cuBLAS GEMM
    instead of log2(D) butterfly kernel launches. 64KB for D=128.
    """
    # 正交归一化的 Hadamard 矩阵（Sylvester 构造法），按 (d, device) 缓存。
    #
    # 预计算的 D*D 矩阵支持基于矩阵乘法的 Walsh-Hadamard 变换 (WHT)，
    # 只需一次 cuBLAS GEMM 调用，而不是 log2(D) 次蝴蝶内核启动。
    # 对于 D=128 仅需 64KB 存储。
    #
    # Hadamard 矩阵的性质：
    # 1. 正交性：H * H^T = I（保持向量范数不变）
    # 2. 对称性：H = H^T（正变换和逆变换相同）
    # 3. 元素仅含 +1/-1（乘法变为加减法）
    # Normalize device string so "cuda" and "cuda:0" hit the same cache entry.
    return _build_hadamard_cached(d, str(torch.device(device_str)))


@functools.cache
def _build_hadamard_cached(d: int, device_str: str) -> torch.Tensor:
    """使用 Sylvester 构造法构建 Hadamard 矩阵并缓存。

    递归构造过程：从 1x1 矩阵 [[1]] 开始，每次将维度翻倍：
    H_{2n} = [[H_n, H_n], [H_n, -H_n]]
    最后除以 sqrt(d) 进行归一化。
    """
    H = torch.tensor([[1.0]])
    while H.shape[0] < d:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return (H / math.sqrt(d)).to(torch.device(device_str))


class TurboQuantAttentionBackend(AttentionBackend):
    """Attention backend using TurboQuant KV-cache compression.
    TurboQuant 注意力后端。使用 TurboQuant KV 缓存压缩技术。
    """

    accept_output_buffer: bool = True
    forward_includes_kv_cache_update: bool = False

    # 支持的数据类型
    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
        torch.bfloat16,
    ]
    # 支持的 KV 缓存数据类型（TurboQuant 的不同压缩配置）：
    # - turboquant_k8v4: Key 8-bit 量化 + Value 4-bit 量化
    # - turboquant_4bit_nc: Key 4-bit 非中心化量化 + Value 4-bit
    # - turboquant_k3v4_nc: Key 3-bit 非中心化量化 + Value 4-bit
    # - turboquant_3bit_nc: Key 3-bit 非中心化量化 + Value 3-bit
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "turboquant_k8v4",
        "turboquant_4bit_nc",
        "turboquant_k3v4_nc",
        "turboquant_3bit_nc",
    ]

    @staticmethod
    def get_name() -> str:
        return "TURBOQUANT"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [16, 32, 64, 128]

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER

    @classmethod
    def supports_per_head_quant_scales(cls) -> bool:
        return False

    @staticmethod
    def get_impl_cls() -> type["TurboQuantAttentionImpl"]:
        return TurboQuantAttentionImpl

    @staticmethod
    def get_builder_cls() -> type["TurboQuantMetadataBuilder"]:
        return TurboQuantMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "turboquant_4bit_nc",
    ) -> tuple[int, ...]:
        """Combined K+V cache shape — no leading 2 dimension.

        Standard attention backends use (2, num_blocks, block_size, num_kv_heads,
        head_dim) with a leading 2 to separate K and V. TurboQuant packs K+V
        into a single interleaved slot per head per position, so the cache is:

            (num_blocks, block_size, num_kv_heads, slot_size_aligned)

        Each slot = [key_packed | value_packed | padding].
        This is safe because TQ has its own get_kv_cache_shape override and
        never shares cache tensors with other backends. Layers that fall back
        to native dtype via kv_cache_dtype_skip_layers get their own
        standard-shaped cache allocation.

        head_size is the model's real head_dim. slot_size_aligned is computed
        from the TQ config to ensure correct cache allocation for all head dims.
        """
        from vllm.model_executor.layers.quantization.turboquant.config import (
            TurboQuantConfig,
        )

        tq_config = TurboQuantConfig.from_cache_dtype(cache_dtype_str, head_size)
        return (num_blocks, block_size, num_kv_heads, tq_config.slot_size_aligned)

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: CacheDType | None) -> bool:
        if kv_cache_dtype is None:
            return False
        return kv_cache_dtype.startswith("turboquant_")

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        # head_size from spec is effective_head_size (padded_slot//2),
        # not the model's actual head_dim. Accept any positive value.
        return head_size > 0


@dataclass
class TurboQuantMetadata(AttentionMetadata):
    """Metadata for TurboQuant attention.
    TurboQuant 注意力的元数据，包含注意力计算所需的所有信息。
    """

    seq_lens: torch.Tensor  # (num_reqs,) — 每个请求的总上下文长度
    slot_mapping: torch.Tensor  # (num_tokens,) — 每个 token 对应的缓存槽映射
    block_table: torch.Tensor  # (num_reqs, max_num_blocks) — 块表，用于从缓存中查找 KV
    query_start_loc: torch.Tensor  # (num_reqs + 1,) — 查询的累积序列长度（cu_seqlens）
    num_actual_tokens: int = 0  # 实际 token 数（不含填充）
    max_query_len: int = 0  # 批次中最长的查询长度
    max_seq_len: int = 0  # 批次中最长的上下文长度
    is_prefill: bool = False  # 是否为 prefill 阶段
    num_decodes: int = 0  # decode 请求数量（在批次前端）
    num_decode_tokens: int = 0  # decode 请求的 token 总数
    # CPU 端副本，用于 prefill 路径中的逐请求迭代，
    # 避免每步都进行 GPU->CPU 同步。
    query_start_loc_cpu: torch.Tensor | None = None
    seq_lens_cpu: torch.Tensor | None = None


class TurboQuantMetadataBuilder(AttentionMetadataBuilder[TurboQuantMetadata]):
    """Builds TurboQuantMetadata from scheduler output.
    从调度器输出构建 TurboQuantMetadata 元数据。
    """

    # CUDA Graph 支持模式：UNIFORM_BATCH 表示所有请求必须是同一类型
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        # reorder_batch_threshold=1 表示 decode 请求（query_len=1）排在批次前面
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=False)

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> TurboQuantMetadata:
        """为 CUDA Graph 捕获构建元数据。

        将 seq_lens 设为 1 以加速 CUDA Graph 捕获
        （实际 seq_lens 在回放时填充）。
        """
        attn_metadata = self.build(0, common_attn_metadata)
        # Set seq_lens to 1 so CUDA graph capture is fast
        # (real seq_lens are filled at replay time).
        attn_metadata.seq_lens.fill_(1)
        return attn_metadata

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        """Build TurboQuantMetadata from common attention metadata.
        从通用注意力元数据构建 TurboQuantMetadata。

        处理流程：
        1. 使用 split_decodes_and_prefills 将批次分为 decode 和 prefill
        2. decode 请求排在批次前端（由 reorder_batch 保证）
        3. 构建包含所有必要信息的 TurboQuantMetadata
        """
        cam = common_attn_metadata

        # With reorder_batch_threshold=1, the model runner guarantees
        # decodes come first in the batch. split_decodes_and_prefills
        # finds the boundary (operates on CPU tensors — no GPU sync).
        # reorder_batch_threshold=1 时，模型运行器保证 decode 请求在批次前端。
        # split_decodes_and_prefills 找到 decode 和 prefill 的分界点（CPU 操作，无 GPU 同步）。
        assert self.reorder_batch_threshold is not None
        num_decodes, num_prefills, num_decode_tokens, _ = split_decodes_and_prefills(
            cam, decode_threshold=self.reorder_batch_threshold
        )

        return TurboQuantMetadata(
            seq_lens=cam.seq_lens,
            slot_mapping=cam.slot_mapping,
            block_table=cam.block_table_tensor,
            query_start_loc=cam.query_start_loc,
            num_actual_tokens=cam.num_actual_tokens,
            max_query_len=cam.max_query_len,
            max_seq_len=cam.max_seq_len,
            is_prefill=(cam.max_query_len > 1),
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            query_start_loc_cpu=cam.query_start_loc_cpu,
            seq_lens_cpu=cam.seq_lens_cpu_upper_bound,
        )


class TurboQuantAttentionImpl(AttentionImpl["TurboQuantMetadata"]):
    """TurboQuant attention implementation.

    Vectorized PyTorch: batch quantize/store, vectorized bit-unpack
    decode with einsum scores and value gather.
    """
    # TurboQuant 注意力实现类。
    #
    # 整体架构：
    # 1. do_kv_cache_update: 将 K/V 量化后存入组合 TQ 缓存
    # 2. forward: 根据批次类型分发到不同的注意力计算路径
    #    - 纯 decode: 使用 Triton TQ decode 内核
    #    - 纯 prefill: 使用 FlashAttention 或 SDPA
    #    - 混合批次: 先 decode 后 prefill

    supports_quant_query_input: bool = False

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        **kwargs,
    ):
        """初始化 TurboQuantAttentionImpl。

        Args:
            num_heads: Q 的注意力头数
            head_size: 头维度
            scale: 注意力缩放因子
            num_kv_heads: KV 的头数（GQA 场景下可能少于 num_heads）
            alibi_slopes: ALiBi 注意力的斜率
            sliding_window: 滑动窗口大小
            kv_cache_dtype: KV 缓存数据类型（如 "turboquant_k3v4_nc"）
            logits_soft_cap: logits 软上限
            attn_type: 注意力类型（DECODER/ENCODER 等）
            kv_sharing_target_layer_name: KV 缓存共享的目标层名
        """
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.num_kv_groups = num_heads // self.num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype

        from vllm.model_executor.layers.quantization.turboquant.config import (
            TurboQuantConfig,
        )

        # 从缓存数据类型解析 TurboQuant 配置
        self.tq_config = TurboQuantConfig.from_cache_dtype(kv_cache_dtype, head_size)

        # Pre-compute kernel constants from config (avoid repeated arithmetic)
        # 从配置中预计算内核常量（避免重复运算）
        cfg = self.tq_config
        # MSE 量化 Key 的字节数
        self._mse_bytes = (
            math.ceil(head_size * cfg.key_mse_bits / 8)
            if not cfg.key_fp8
            else head_size
        )
        # Value 量化数据的字节数
        self._val_data_bytes = math.ceil(head_size * cfg.effective_value_quant_bits / 8)
        # 量化质心数量（FP8 模式下为 1，即不使用 MSE 量化）
        self._n_centroids = cfg.n_centroids if not cfg.key_fp8 else 1

        # Detect flash-attn version (FA2/3/4) for prefill paths.
        # 检测 FlashAttention 版本（FA2/3/4），用于 prefill 路径
        self.fa_version = get_flash_attn_version(head_size=head_size)

        # Fixed NUM_KV_SPLITS (grid dims must be constant for cudagraph,
        # and benchmarks show no regression vs dynamic in eager mode).
        # 固定 KV 分区数（CUDA Graph 要求 grid 维度为常量，
        # 基准测试显示与动态模式相比无性能退化）。
        vllm_config = get_current_vllm_config()
        self.max_num_kv_splits = (
            vllm_config.attention_config.tq_max_kv_splits_for_cuda_graph
        )

    def _flash_attn_varlen(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
    ) -> torch.Tensor:
        """调用 flash_attn_varlen_func 执行变长因果注意力。

        根据检测到的 FA 版本决定是否传递 fa_version 参数。
        """
        # fa_utils.get_flash_attn_version() returns None on backends that
        # should not pass an explicit fa_version kwarg.
        if self.fa_version is None:
            return flash_attn_varlen_func(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                softmax_scale=self.scale,
                causal=True,
            )
        return flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=self.scale,
            causal=True,
            fa_version=self.fa_version,
        )

    def _ensure_on_device(self, layer, device):
        """One-time derivation of TQ buffers (rotation matrix, midpoints).

        The Hadamard rotation is shared across all layers: random sign
        flips do not improve Lloyd-Max quantization quality because the
        quantizer is symmetric around zero (sign-flipping a coordinate
        maps it to the mirror centroid with identical distortion).
        """
        # 一次性推导 TQ 缓冲区（旋转矩阵、中点等）并确保在目标设备上。
        #
        # Hadamard 旋转矩阵在所有层之间共享：随机符号翻转不会改善
        # Lloyd-Max 量化质量，因为量化器围绕零对称（符号翻转一个坐标
        # 会映射到具有相同失真的镜像质心）。
        #
        # 设置的缓冲区：
        # 1. _tq_Pi / _tq_PiT: Hadamard 旋转矩阵及其转置（相同，因为对称）
        # 2. _tq_Pi_half: FP16 版本的旋转矩阵（用于延续 prefill 路径）
        # 3. _tq_centroids: Lloyd-Max 量化的质心
        # 4. _tq_midpoints: 相邻质心的中点（用于量化决策）
        if not hasattr(layer, "_tq_cached"):
            D = self.head_size

            # Pure Hadamard: orthonormal + symmetric (H = H^T), enabling
            # in-kernel butterfly fusion and trivial inverse for continuation.
            # 纯 Hadamard 矩阵：正交 + 对称（H = H^T），支持内核内
            # 蝴蝶融合和延续 prefill 的简单逆变换。
            H = _build_hadamard(D, str(device))
            layer._tq_PiT = H
            layer._tq_Pi = H
            # fp16 copy for rotation in continuation prefill path
            layer._tq_Pi_half = H.to(torch.float16)

            # Centroids for Lloyd-Max quantization.
            # Lloyd-Max 量化的质心：预计算的最优标量量化码本
            layer._tq_centroids = get_centroids(D, self.tq_config.centroid_bits).to(
                device=device, dtype=torch.float32
            )

            # 计算相邻质心的中点（用于判断量化到哪个质心）
            c_sorted, _ = layer._tq_centroids.sort()
            layer._tq_midpoints = (c_sorted[:-1] + c_sorted[1:]) / 2
            layer._tq_cached = True

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        """Store compressed K/V into the combined TQ cache.

        Called as a separate custom op (unified_kv_cache_update) BEFORE
        the attention forward, matching FlashAttention's split pattern.
        slot_mapping is already sliced to num_actual_tokens by the caller.
        """
        # 将压缩后的 K/V 存入组合 TQ 缓存。
        #
        # 作为独立的自定义操作（unified_kv_cache_update）在注意力前向传播
        # 之前调用，与 FlashAttention 的分离模式一致。
        # 调用者已将 slot_mapping 切片到 num_actual_tokens。
        #
        # 处理流程：
        # 1. 确保 TQ 缓冲区（旋转矩阵等）在目标设备上
        # 2. 将 key/value 重塑为 (N, num_kv_heads, head_size) 形状
        # 3. 调用 _store_kv 执行量化和存储
        N = slot_mapping.shape[0]
        if N <= 0:
            return

        device = key.device
        self._ensure_on_device(layer, device)

        k = key[:N].view(N, self.num_kv_heads, self.head_size)
        v = value[:N].view(N, self.num_kv_heads, self.head_size)
        self._store_kv(k, v, kv_cache, slot_mapping, layer)

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: "TurboQuantMetadata",
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """TurboQuant 注意力的前向传播。

        根据批次类型分发到不同路径：
        1. 纯 decode 批次：使用 TQ decode 内核（从压缩缓存读取）
        2. 纯 prefill 批次：使用 FlashAttention 或 SDPA（在原始 K/V 上计算）
        3. 混合批次：先处理 decode 请求，再处理 prefill 请求

        注意：KV 缓存已由 do_kv_cache_update 提前更新。
        """
        num_tokens = query.shape[0]

        if output is None:
            output = torch.zeros(
                num_tokens,
                self.num_heads * self.head_size,
                dtype=query.dtype,
                device=query.device,
            )

        if attn_metadata is None:
            return output.fill_(0)

        # Slice to actual tokens
        N = attn_metadata.num_actual_tokens
        if N <= 0:
            return output.fill_(0)

        q = query[:N].view(N, self.num_heads, self.head_size)

        # Get TQ buffers, ensure on device (one-time migration).
        # Use Any-typed alias for dynamic _tq_* attrs set by _ensure_on_device.
        tq_layer: Any = layer
        device = q.device
        self._ensure_on_device(tq_layer, device)
        Pi = tq_layer._tq_Pi
        PiT = tq_layer._tq_PiT
        centroids = tq_layer._tq_centroids

        # Compute attention (KV cache was already updated by do_kv_cache_update)
        # With reorder_batch_threshold=1, decodes come first in the batch.
        # num_decodes/num_decode_tokens from metadata give the split point.
        num_decodes = attn_metadata.num_decodes
        num_decode_tokens = attn_metadata.num_decode_tokens

        if not attn_metadata.is_prefill:
            # Pure decode batch — fast path
            attn_out = self._decode_attention(
                q, kv_cache, attn_metadata, Pi, centroids, PiT, layer
            )
        elif num_decodes == 0:
            # Pure prefill batch
            k = key[:N].view(N, self.num_kv_heads, self.head_size)
            v = value[:N].view(N, self.num_kv_heads, self.head_size)
            attn_out = self._prefill_attention(
                q,
                k,
                v,
                kv_cache,
                attn_metadata,
                Pi,
                centroids,
                PiT,
                layer=layer,
            )
        else:
            # Mixed batch: decodes first (guaranteed by reorder_batch).
            attn_out = torch.empty(
                N, self.num_heads, self.head_size, device=device, dtype=q.dtype
            )

            # --- Decode portion (first num_decodes requests) ---
            # Use full-batch max_seq_len as safe upper bound (no GPU sync).
            decode_meta = TurboQuantMetadata(
                seq_lens=attn_metadata.seq_lens[:num_decodes],
                slot_mapping=attn_metadata.slot_mapping[:num_decode_tokens],
                block_table=attn_metadata.block_table[:num_decodes],
                query_start_loc=attn_metadata.query_start_loc[: num_decodes + 1],
                num_actual_tokens=num_decode_tokens,
                max_query_len=1,
                max_seq_len=attn_metadata.max_seq_len,
                is_prefill=False,
            )
            attn_out[:num_decode_tokens] = self._decode_attention(
                q[:num_decode_tokens], kv_cache, decode_meta, Pi, centroids, PiT, layer
            )

            # --- Prefill portion (remaining requests) ---
            # CRITICAL: use prefill-specific max_seq_len so flash_attn's
            # fast path (max_query_len == max_seq_len) triggers for
            # first-chunk prefills. Using full-batch max_seq_len breaks
            # this because decode requests inflate max_seq_len.
            prefill_seq_lens = attn_metadata.seq_lens[num_decodes:]
            # Use the CPU-resident `seq_lens` upper-bound from the metadata
            # (populated in the builder) to compute the prefill sub-batch
            # max without a GPU→CPU sync.
            if attn_metadata.seq_lens_cpu is not None:
                prefill_max_seq = int(attn_metadata.seq_lens_cpu[num_decodes:].max())
            else:
                prefill_max_seq = attn_metadata.max_seq_len
            prefill_qsl = (
                attn_metadata.query_start_loc[num_decodes:] - num_decode_tokens
            )
            prefill_qsl_cpu = None
            if attn_metadata.query_start_loc_cpu is not None:
                prefill_qsl_cpu = (
                    attn_metadata.query_start_loc_cpu[num_decodes:] - num_decode_tokens
                )
            prefill_meta = TurboQuantMetadata(
                seq_lens=prefill_seq_lens,
                slot_mapping=attn_metadata.slot_mapping[num_decode_tokens:N],
                block_table=attn_metadata.block_table[num_decodes:],
                query_start_loc=prefill_qsl,
                num_actual_tokens=N - num_decode_tokens,
                max_query_len=attn_metadata.max_query_len,
                max_seq_len=prefill_max_seq,
                is_prefill=True,
                query_start_loc_cpu=prefill_qsl_cpu,
                seq_lens_cpu=attn_metadata.seq_lens_cpu[num_decodes:]
                if attn_metadata.seq_lens_cpu is not None
                else None,
            )
            k = key[:N].view(N, self.num_kv_heads, self.head_size)
            v = value[:N].view(N, self.num_kv_heads, self.head_size)
            attn_out[num_decode_tokens:] = self._prefill_attention(
                q[num_decode_tokens:],
                k[num_decode_tokens:],
                v[num_decode_tokens:],
                kv_cache,
                prefill_meta,
                Pi,
                centroids,
                PiT,
                layer=layer,
            )

        # Write into output buffer: attn_out is (N, Hq, D)
        # output may be 2D (N, Hq*D) or 3D (N, Hq, D)
        if output.ndim == 3:
            output[:N] = attn_out.to(output.dtype)
        else:
            output[:N] = attn_out.reshape(N, -1).to(output.dtype)
        return output

    # ------------------------------------------------------------------ #
    #  Store K/V into combined cache (vectorized)                         #
    #  将 K/V 存入组合缓存（向量化）                                       #
    # ------------------------------------------------------------------ #
    def _store_kv(
        self,
        key: torch.Tensor,  # (N, Hk, D)
        value: torch.Tensor,  # (N, Hk, D)
        kv_cache: torch.Tensor,  # (num_blocks, block_size, Hk, slot_size)
        slot_mapping: torch.Tensor,
        layer: Any,
    ):
        """Quantize + store via fused Triton kernel.
        通过融合的 Triton 内核执行量化和存储。

        处理流程：
        1. 对 Key 应用 Hadamard 旋转（MSE 量化模式）
        2. 使用 Lloyd-Max 质心对 Key 进行标量量化并打包
        3. 对 Value 进行低精度量化
        4. 将量化后的 Key + Value 写入组合缓存槽
        """
        triton_turboquant_store(
            key,
            value,
            kv_cache,
            slot_mapping,
            layer._tq_PiT,
            layer._tq_midpoints,
            mse_bits=self.tq_config.key_mse_bits,
            key_packed_size=self.tq_config.key_packed_size,
            value_quant_bits=self.tq_config.effective_value_quant_bits,
            key_fp8=self.tq_config.key_fp8,
        )

    # ------------------------------------------------------------------ #
    #  Prefill: SDPA on raw Q/K/V with causal mask                        #
    #  Prefill: 使用因果掩码在原始 Q/K/V 上执行缩放点积注意力                #
    # ------------------------------------------------------------------ #
    def _prefill_attention(
        self,
        query: torch.Tensor,  # (N, Hq, D)
        key: torch.Tensor,  # (N, Hk, D)
        value: torch.Tensor,  # (N, Hk, D)
        kv_cache: torch.Tensor,  # (num_blocks, block_size, Hk, slot_size)
        attn_metadata: TurboQuantMetadata,
        Pi: torch.Tensor,
        centroids: torch.Tensor,
        PiT: torch.Tensor | None = None,
        layer: Any = None,
    ) -> torch.Tensor:
        """对 prefill 请求执行注意力计算。

        Prefill 有两种子路径：
        1. 首块 prefill（q_len == seq_len）：所有 K/V 都在当前批次中，
           可以使用 FlashAttention 的快速路径。
        2. 延续 prefill（q_len < seq_len）：需要访问已缓存的 K/V，
           根据 q_len 大小选择 TQ decode 内核或反量化 + flash_attn。

        Args:
            query: 查询张量 (N, Hq, D)
            key: 键张量 (N, Hk, D)
            value: 值张量 (N, Hk, D)
            kv_cache: TQ 组合缓存
            attn_metadata: TurboQuant 元数据
            Pi: Hadamard 旋转矩阵
            centroids: Lloyd-Max 量化质心
            PiT: Hadamard 旋转矩阵的转置
            layer: 注意力层（用于访问 TQ 缓冲区）

        Returns:
            注意力输出 (N, Hq, D)
        """
        N, Hq, D = query.shape

        # Fast path: use flash_attn for first-chunk prefills (all K/V in batch).
        # max_query_len == max_seq_len means no request has prior cached KV.
        # Both are Python ints — no GPU sync.
        # 快速路径：对首块 prefill 使用 flash_attn（所有 K/V 都在当前批次中）。
        # max_query_len == max_seq_len 表示没有请求有之前的缓存 KV。
        # 两者都是 Python int，不需要 GPU 同步。
        if _HAS_FLASH_ATTN and attn_metadata.max_query_len == attn_metadata.max_seq_len:
            return self._flash_attn_varlen(
                q=query,
                k=key,
                v=value,
                cu_seqlens_q=attn_metadata.query_start_loc,
                cu_seqlens_k=attn_metadata.query_start_loc,
                max_seqlen_q=attn_metadata.max_query_len,
                max_seqlen_k=attn_metadata.max_query_len,
            )

        # Continuation or no flash_attn: per-request attention.
        # For continuation chunks (seq_len > q_len), we must attend to
        # previously cached K/V from the TQ cache, not just the current
        # chunk's raw K/V.
        Hk = key.shape[1]
        use_gqa = Hk < Hq
        query_start_loc = attn_metadata.query_start_loc
        num_reqs = query_start_loc.shape[0] - 1

        output = torch.zeros(N, Hq, D, device=query.device, dtype=query.dtype)

        # Prefer the CPU-resident copies from the metadata if populated —
        # otherwise `.tolist()` on GPU tensors forces a synchronizing copy.
        if attn_metadata.query_start_loc_cpu is not None:
            qsl = attn_metadata.query_start_loc_cpu.tolist()
        else:
            qsl = query_start_loc.tolist()
        if attn_metadata.seq_lens_cpu is not None:
            seq_lens_list = attn_metadata.seq_lens_cpu.tolist()
        else:
            seq_lens_list = attn_metadata.seq_lens.tolist()

        # Pre-allocate cu_seqlens for single-request flash_attn calls
        # to avoid per-request host→device tensor creation.
        if not hasattr(self, "_cu_2"):
            self._cu_2 = torch.zeros(2, device=query.device, dtype=torch.int32)
        # Cache arange on self (avoid per-call kernel launch).
        _max_seq = attn_metadata.max_seq_len
        _ac: torch.Tensor | None = getattr(self, "_arange_cache", None)
        if _ac is None or _ac.shape[0] <= _max_seq:
            _ac = torch.arange(
                0, _max_seq + 1, device=query.device, dtype=attn_metadata.seq_lens.dtype
            )
            self._arange_cache = _ac
        _arange_cache: torch.Tensor = _ac

        for i in range(num_reqs):
            q_start = qsl[i]
            q_end = qsl[i + 1]
            q_len = q_end - q_start
            if q_len <= 0:
                continue

            seq_len = seq_lens_list[i]
            q_seq = query[q_start:q_end]  # (q_len, Hq, D)
            k_seq = key[q_start:q_end]  # (q_len, Hk, D)
            v_seq = value[q_start:q_end]  # (q_len, Hk, D)

            if q_len == seq_len:
                # First-chunk prefill: all K/V are in the current batch.
                if _HAS_FLASH_ATTN:
                    # Assign to slice to avoid gpu/cpu sync.
                    self._cu_2[1:2] = q_len
                    cu = self._cu_2
                    out = self._flash_attn_varlen(
                        q=q_seq,
                        k=k_seq,
                        v=v_seq,
                        cu_seqlens_q=cu,
                        cu_seqlens_k=cu,
                        max_seqlen_q=q_len,
                        max_seqlen_k=q_len,
                    )
                else:
                    q_t = q_seq.transpose(0, 1).contiguous()
                    k_t = k_seq.transpose(0, 1).contiguous()
                    v_t = v_seq.transpose(0, 1).contiguous()
                    out = F.scaled_dot_product_attention(
                        q_t,
                        k_t,
                        v_t,
                        is_causal=True,
                        scale=self.scale,
                        enable_gqa=use_gqa,
                    ).transpose(0, 1)
                output[q_start:q_end] = out.to(query.dtype)
            else:
                # Continuation chunk: tokens already stored to TQ cache
                # by do_kv_cache_update. Use decode kernel directly to
                # avoid O(cached_len) full-dequant per continuation.
                # For large continuations, fall back to _continuation_prefill.
                cached_len = seq_len - q_len
                if q_len <= _CONTINUATION_DECODE_THRESHOLD:
                    # Fast path: treat each query as a decode request
                    # with incremental seq_lens for causal masking.
                    # Slice from pre-built arange (no kernel launch)
                    synth_seq_lens = _arange_cache[cached_len + 1 : seq_len + 1]
                    synth_bt = attn_metadata.block_table[i : i + 1].expand(q_len, -1)
                    out = triton_turboquant_decode_attention(
                        query=q_seq,
                        kv_cache=kv_cache,
                        block_table=synth_bt,
                        seq_lens=synth_seq_lens,
                        Pi=Pi,
                        centroids=centroids,
                        scale=self.scale,
                        mse_bits=self.tq_config.key_mse_bits,
                        key_packed_size=self.tq_config.key_packed_size,
                        value_quant_bits=(self.tq_config.effective_value_quant_bits),
                        key_fp8=self.tq_config.key_fp8,
                        norm_correction=self.tq_config.norm_correction,
                        PiT=PiT,
                    )
                else:
                    # Large continuation: dequant cached K/V and use
                    # flash_attn for better throughput.
                    out = self._continuation_prefill(
                        layer,
                        q_seq,
                        k_seq,
                        v_seq,
                        kv_cache,
                        attn_metadata.block_table[i : i + 1],
                        cached_len,
                        seq_len,
                        Pi,
                        centroids,
                    )
                output[q_start:q_end] = out.to(query.dtype)

        return output

    def _continuation_prefill(
        self,
        layer: Any,
        query: torch.Tensor,  # (q_len, Hq, D)
        key_chunk: torch.Tensor,  # (q_len, Hk, D)
        val_chunk: torch.Tensor,  # (q_len, Hk, D)
        kv_cache: torch.Tensor,  # (num_blocks, block_size, Hk, slot_size)
        block_table: torch.Tensor,  # (1, max_num_blocks)
        cached_len: int,
        seq_len: int,
        Pi: torch.Tensor,
        centroids: torch.Tensor,
    ) -> torch.Tensor:
        """Handle continuation chunk by dequanting cached K/V from TQ cache.

        Dequants previously cached K/V, concatenates with the current
        chunk's raw K/V, then runs flash_attn with causal masking.
        """
        q_len, Hq, D = query.shape
        Hk = key_chunk.shape[1]
        device = query.device
        block_size = kv_cache.shape[1]
        BLOCK_D = triton.next_power_of_2(D)

        mse_bytes = self._mse_bytes
        val_data_bytes = self._val_data_bytes

        # Dequant cached K/V from TQ cache
        # Allocate slightly over to align to block_size for the grid.
        # Reuse cached buffers to avoid per-call allocation (~16MB at 8K).
        alloc_len = math.ceil(cached_len / block_size) * block_size
        buf_shape = (1, Hk, alloc_len, D)
        # Use WorkspaceManager for dequant buffers.
        # Shared across all layers — saves 60× memory at long context.
        # Required for CUDA Graph capture (per-layer growth incompatible with CG).
        k_buf, v_buf = current_workspace_manager().get_simultaneous(
            (buf_shape, torch.float16),
            (buf_shape, torch.float16),
        )
        # Skip .zero_() — kernel writes all positions up to cached_len,
        # and we only read [:cached_len] afterwards.
        k_cached = k_buf[:, :, :alloc_len, :]
        v_cached = v_buf[:, :, :alloc_len, :]

        grid = (alloc_len, 1 * Hk)
        _tq_full_dequant_kv[grid](
            kv_cache,
            block_table,
            centroids,
            k_cached,
            v_cached,
            k_cached.stride(0),
            k_cached.stride(1),
            k_cached.stride(2),
            v_cached.stride(0),
            v_cached.stride(1),
            v_cached.stride(2),
            kv_cache.stride(0),
            kv_cache.stride(1),
            kv_cache.stride(2),
            block_table.stride(0),
            HEAD_DIM=D,
            BLOCK_SIZE=block_size,
            NUM_KV_HEADS=Hk,
            MSE_BYTES=mse_bytes,
            KPS=self.tq_config.key_packed_size,
            VQB=self.tq_config.effective_value_quant_bits,
            VAL_DATA_BYTES=val_data_bytes,
            MSE_BITS=self.tq_config.key_mse_bits,
            KEY_FP8=1 if self.tq_config.key_fp8 else 0,
            BLOCK_D=BLOCK_D,
            NORM_CORRECTION=1 if self.tq_config.norm_correction else 0,
            FP8_E4B15=_use_fp8_e4b15(device.index or 0),
            num_warps=4,
        )

        # Inverse-rotate MSE keys back to original space
        if not self.tq_config.key_fp8:
            # fp16 matmul for rotation (2× less bandwidth, uses fp16 tensor cores)
            Pi_half = layer._tq_Pi_half
            k_flat = k_cached[0, :, :cached_len, :].reshape(-1, D)
            k_flat = k_flat @ Pi_half
            k_cached_trim = k_flat.reshape(Hk, cached_len, D).transpose(
                0, 1
            )  # (cached_len, Hk, D) — already fp16
        else:
            k_cached_trim = k_cached[0, :, :cached_len, :].transpose(
                0, 1
            )  # (cached_len, Hk, D)

        # Skip .contiguous() — the copy into k_full/v_full handles layout
        v_cached_trim = v_cached[0, :, :cached_len, :].transpose(0, 1)

        # Concatenate cached + current chunk K/V (match query dtype)
        # Pre-allocate full K/V buffer, copy into slices (no cat alloc)
        qdtype = query.dtype
        k_full = torch.empty(seq_len, Hk, D, dtype=qdtype, device=device)
        v_full = torch.empty(seq_len, Hk, D, dtype=qdtype, device=device)
        k_full[:cached_len] = k_cached_trim.to(qdtype)
        k_full[cached_len:] = key_chunk
        v_full[:cached_len] = v_cached_trim.to(qdtype)
        v_full[cached_len:] = val_chunk

        # Attention: q_len queries attending to seq_len K/V with causal mask
        if _HAS_FLASH_ATTN:
            # Reuse pre-allocated cu_seqlens (avoid host→device transfer)
            if not hasattr(self, "_cu_2_q"):
                self._cu_2_q = torch.zeros(2, device=device, dtype=torch.int32)
                self._cu_2_k = torch.zeros(2, device=device, dtype=torch.int32)
            # Assigning to slice uses fill_ which avoids cpu/gpu sync.
            self._cu_2_q[1:2] = q_len
            self._cu_2_k[1:2] = seq_len
            cu_seqlens_q = self._cu_2_q
            cu_seqlens_k = self._cu_2_k
            return self._flash_attn_varlen(
                q=query,
                k=k_full,
                v=v_full,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=q_len,
                max_seqlen_k=seq_len,
            )
        else:
            # SDPA fallback: expand KV for GQA, build causal mask
            q_t = query.transpose(0, 1).unsqueeze(0)  # (1, Hq, q_len, D)
            k_t = k_full.transpose(0, 1).unsqueeze(0)  # (1, Hk, seq_len, D)
            v_t = v_full.transpose(0, 1).unsqueeze(0)  # (1, Hk, seq_len, D)
            # Build causal mask: query position p can attend to K position j
            # where j <= cached_len + p (p is 0-indexed within chunk)
            q_pos = torch.arange(q_len, device=device).unsqueeze(1) + cached_len
            k_pos = torch.arange(seq_len, device=device).unsqueeze(0)
            mask = k_pos <= q_pos  # (q_len, seq_len)
            out = F.scaled_dot_product_attention(
                q_t,
                k_t,
                v_t,
                attn_mask=mask,
                scale=self.scale,
                enable_gqa=(Hk < Hq),
            )  # (1, Hq, q_len, D)
            return out[0].transpose(0, 1)  # (q_len, Hq, D)

    # ------------------------------------------------------------------ #
    #  Decode: Triton TQ decode attention                                 #
    #  Decode: 使用 Triton TQ decode 注意力内核                            #
    # ------------------------------------------------------------------ #
    def _decode_attention(
        self,
        query: torch.Tensor,  # (B, Hq, D)
        kv_cache: torch.Tensor,  # (num_blocks, block_size, Hk, slot_size)
        attn_metadata: TurboQuantMetadata,
        Pi: torch.Tensor,
        centroids: torch.Tensor,
        PiT: torch.Tensor | None = None,
        layer: torch.nn.Module | None = None,
    ) -> torch.Tensor:
        """使用 Triton TQ decode 内核执行 decode 注意力。

        从压缩的 TQ 缓存中读取 K/V，在线反量化并计算注意力分数。
        使用分区内归约（partitioned reduction）来处理长序列。

        Args:
            query: 查询张量 (B, Hq, D)，每个请求一个查询 token
            kv_cache: TQ 组合缓存
            attn_metadata: TurboQuant 元数据
            Pi: Hadamard 旋转矩阵
            centroids: Lloyd-Max 量化质心
            PiT: Hadamard 旋转矩阵的转置
            layer: 注意力层

        Returns:
            注意力输出 (B, Hq, D)
        """
        # Acquire shared decode scratch buffers from WorkspaceManager.
        # Layers execute sequentially so one set of buffers is sufficient.
        # Falls back to kernel-internal allocation if workspace unavailable.
        # 从 WorkspaceManager 获取共享的 decode 临时缓冲区。
        # 层是顺序执行的，所以一组缓冲区就够了。
        # 如果工作空间不可用，回退到内核内部分配。
        B = query.shape[0]
        D = self.head_size
        S = self.max_num_kv_splits
        Hq = self.num_heads
        mid_o_buf = output_buf = lse_buf = None
        if is_workspace_manager_initialized():
            # output_buf in query dtype — matches the in-kernel fp16 cast in stage2.
            mid_o_buf, output_buf, lse_buf = (
                current_workspace_manager().get_simultaneous(
                    ((B, Hq, S, D + 1), torch.float32),
                    ((B, Hq, D), query.dtype),
                    ((B, Hq), torch.float32),
                )
            )

        result = triton_turboquant_decode_attention(
            query=query,
            kv_cache=kv_cache,
            block_table=attn_metadata.block_table,
            seq_lens=attn_metadata.seq_lens,
            Pi=Pi,
            centroids=centroids,
            scale=self.scale,
            mse_bits=self.tq_config.key_mse_bits,
            key_packed_size=self.tq_config.key_packed_size,
            value_quant_bits=self.tq_config.effective_value_quant_bits,
            key_fp8=self.tq_config.key_fp8,
            norm_correction=self.tq_config.norm_correction,
            PiT=PiT,
            mid_o_buf=mid_o_buf,
            output_buf=output_buf,
            lse_buf=lse_buf,
            buf_holder=layer,
            max_num_kv_splits=self.max_num_kv_splits,
        )
        return result
