# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# 本模块实现了基于 Triton 的 "reshape and cache" 操作，
# 用于将新计算的 KV 张量（来自 prefill 或 decode 阶段）
# 重塑并写入分页 KV 缓存中。
#
# 核心功能：
# 1. reshape_and_cache_kernel_flash - 标准 KV 缓存写入
#    将 key [num_tokens, num_heads, head_size] 和 value 张量
#    按照 slot_mapping 映射写入分页缓存：
#    key_cache [num_blocks, block_size, num_heads, head_size]
#    value_cache [num_blocks, block_size, num_heads, head_size]
#    支持两种内存布局：
#    - block-major（标准）：key_cache.ndim == 4
#    - head-major（5D）：key_cache.ndim == 5，用于 FlashAttention 的 K 布局
#
# 2. _reshape_cache_per_token_head - 逐 token-head 动态量化
#    每个 (token, head) 对独立计算 absmax 缩放因子，
#    将数据量化为 int8 或 fp8 并存入缓存，
#    缩放因子存入对应的 scale_cache 中。
#
# 3. reshape_and_cache_kernel_flash_diffkv - 差异化 KV 头维度
#    当 K 和 V 的头维度不同时（如 MLA 模式），
#    将 K 和 V 拼接后写入统一的 kv_cache 缓存。
#
# FP8 支持：
# - 自动检测 kv_cache_dtype 是否为量化类型（fp8_e4m3, fp8_e5m2 等）
# - 写入时进行 FP8 反量化缩放：value / scale
# - Triton 的 tl.store 会根据目标 dtype 自动进行隐式类型转换

import torch

from vllm.model_executor.layers.quantization.utils.quant_utils import (
    FP8_DTYPE,
    get_fp8_min_max,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import is_quantized_kv_cache

# FP8 数据类型的数值范围
FP8_MIN, FP8_MAX = get_fp8_min_max()

# 支持的原生（非量化）KV 缓存数据类型集合
_NATIVE_KV_CACHE_DTYPES = {"auto", "float16", "bfloat16", "float32", "half", "float"}


def _is_supported_kv_cache_dtype(kv_cache_dtype: str) -> bool:
    """检查 KV 缓存数据类型是否受支持（原生或量化类型）"""
    return kv_cache_dtype in _NATIVE_KV_CACHE_DTYPES or is_quantized_kv_cache(
        kv_cache_dtype
    )


@triton.jit
def reshape_and_cache_kernel_flash(
    key_ptr,  # [num_tokens, num_heads, head_size]
    value_ptr,  # [num_tokens, num_heads, head_size]
    key_cache_ptr,  # [num_blocks, block_size, num_heads, head_size]
    value_cache_ptr,  # [num_blocks, block_size, num_heads, head_size]
    slot_mapping_ptr,  # [num_tokens]
    k_scale,  # float32
    v_scale,  # float32
    # strides
    key_stride: tl.int64,
    value_stride: tl.int64,
    block_stride: tl.int64,
    head_stride: tl.int64,
    dim_stride_k: tl.int64,
    dim_stride_v: tl.int64,
    page_stride: tl.int64,
    num_heads: tl.constexpr,
    head_size: tl.constexpr,
    block_size: tl.constexpr,
    x: tl.constexpr,
    USE_HEAD_MAJOR_LAYOUT: tl.constexpr,
    # FP8 flags
    FP8_KV_CACHE: tl.constexpr,
    # tune parameters
    TILE_SIZE: tl.constexpr,
):
    """
    标准 KV 缓存写入 kernel。

    网格维度：(num_tokens, num_heads * head_size / TILE_SIZE)
    每个程序处理一个 token 的一部分（TILE_SIZE 个元素）。

    工作流程：
    1. 通过 slot_mapping 将 token 索引映射到缓存中的物理位置
    2. 计算 block_idx（页号）和 block_offset（页内偏移）
    3. 根据 USE_HEAD_MAJOR_LAYOUT 选择不同的地址计算方式
    4. 加载 key 和 value 数据
    5. 如果启用 FP8 量化，进行反量化缩放
    6. 写入 key_cache 和 value_cache
    """
    token_idx = tl.program_id(axis=0)
    slot_idx = tl.load(slot_mapping_ptr + token_idx).to(tl.int64)
    if slot_idx < 0:
        # Padding token that should be ignored.
        return

    # 将 slot 映射为页号和页内偏移
    block_idx = slot_idx // block_size
    block_offset = slot_idx % block_size

    # 第二个程序轴负责将 (num_heads * head_size) 分成 TILE_SIZE 大小的块
    tile_i = tl.program_id(axis=1)
    tile_offs = tl.arange(0, TILE_SIZE)
    tile_pos = tile_i * TILE_SIZE + tile_offs
    src_key_idx = token_idx * key_stride
    src_value_idx = token_idx * value_stride

    if USE_HEAD_MAJOR_LAYOUT:
        # Head-major 布局（5D K 缓存）：[Block, Head, Dim//x, Slot, x]
        # 用于 FlashAttention 的 K 布局，x 通常为 8（用于向量化访问）
        # 分解 tile 索引为 head 和 dim 坐标
        cur_head = tile_pos // head_size
        cur_dim = tile_pos % head_size
        # Value 地址计算（4D）：[Block, Head, Dim, Slot]
        tgt_idx_v = (
            block_idx * block_stride
            + cur_head * head_stride
            + cur_dim * dim_stride_v
            + block_offset * 1
        )
        # Key 地址计算（5D）：[Block, Head, Dim//8, Slot, 8]
        # 将 head_size 维度分成 Dim//x 块，每块包含 x 个元素
        tgt_idx_k = (
            block_idx * block_stride
            + cur_head * head_stride
            + (cur_dim // x) * dim_stride_k
            + block_offset * x
            + (cur_dim % x)
        )
    else:
        # 标准 block-major 布局（4D）：[Block, Slot, Head, Dim]
        cur_head = tile_pos // head_size
        cur_dim = tile_pos % head_size
        tgt_idx_k = (
            block_idx * block_stride
            + block_offset * page_stride
            + cur_head * head_stride
            + cur_dim
        )
        tgt_idx_v = tgt_idx_k

    # [TILE_SIZE] 加载 key 数据
    key_load = tl.load(
        key_ptr + src_key_idx + tile_pos, mask=tile_pos < (num_heads * head_size)
    )
    # FP8 反量化：如果数据已经是 FP8 则直接存储，否则除以缩放因子
    # tl.store 会根据目标 dtype.element_ty 自动进行隐式类型转换
    if FP8_KV_CACHE:
        key_tile = key_load if key_load.dtype.is_fp8() else key_load / tl.load(k_scale)
    else:
        key_tile = key_load

    # [TILE_SIZE] 加载 value 数据
    value_load = tl.load(
        value_ptr + src_value_idx + tile_pos, mask=tile_pos < (num_heads * head_size)
    )
    if FP8_KV_CACHE:
        if value_load.dtype.is_fp8():
            value_tile = value_load
        else:
            # tl.store will do the correct implicit cast to fp8,
            #  based on the value_cache_ptr.dtype.element_ty
            value_tile = value_load / tl.load(v_scale)
    else:
        value_tile = value_load

    tl.store(
        key_cache_ptr + tgt_idx_k,
        key_tile,
        mask=tile_pos < (num_heads * head_size),
    )
    tl.store(
        value_cache_ptr + tgt_idx_v,
        value_tile,
        mask=tile_pos < (num_heads * head_size),
    )
    return


# ---------------------------------------------------------------------------
# Per-token-head dynamic quantization kernel
# Grid: (num_tokens, NUM_KV_HEADS)
# Each program handles one (token, head) pair:
#   1. Loads K (or V) for that single head
#   2. Computes absmax across head_size → scale = absmax / QUANT_MAX
#   3. Quantizes and stores the data + per-head scale
#
# Parametrised by QUANT_MAX / QUANT_MIN so the same code path works
# for int8 (±127/128), fp8_e4m3 (±448), and other formats.
# ---------------------------------------------------------------------------
# 逐 token-head 动态量化 kernel
# 网格维度：(num_tokens, num_kv_heads)
# 每个程序处理一个 (token, head) 对：
# 1. 加载该 head 的 K 或 V 数据
# 2. 计算 absmax 得到缩放因子：scale = absmax / QUANT_MAX
# 3. 量化并存储数据和逐 head 的缩放因子
#
# 通过 QUANT_MAX / QUANT_MIN 参数化，同一代码路径支持：
# - int8：±127/128
# - fp8_e4m3：±448
# - 其他格式
# ---------------------------------------------------------------------------
@triton.jit
def _reshape_cache_per_token_head(
    key_ptr,  # [num_tokens, num_kv_heads, head_size]
    value_ptr,  # [num_tokens, num_kv_heads, head_size_v]
    key_cache_ptr,  # [num_blocks, block_size, num_kv_heads, head_size]
    value_cache_ptr,  # [num_blocks, block_size, num_kv_heads, head_size_v]
    k_scale_cache_ptr,  # [num_blocks, block_size, num_kv_heads] float32
    v_scale_cache_ptr,  # [num_blocks, block_size, num_kv_heads] float32
    slot_mapping_ptr,  # [num_tokens]
    stride_key_tok: tl.int64,
    stride_key_head: tl.int64,
    stride_val_tok: tl.int64,
    stride_val_head: tl.int64,
    stride_kc_blk: tl.int64,  # key_cache stride over blocks
    stride_kc_slot: tl.int64,  # key_cache stride over slots
    stride_kc_head: tl.int64,  # key_cache stride over heads
    stride_vc_blk: tl.int64,
    stride_vc_slot: tl.int64,
    stride_vc_head: tl.int64,
    stride_ks_blk: tl.int64,  # k_scale_cache stride[0] (blocks)
    stride_ks_slot: tl.int64,  # k_scale_cache stride[1] (slots)
    stride_ks_head: tl.int64,  # k_scale_cache stride[2] (heads)
    stride_vs_blk: tl.int64,  # v_scale_cache stride[0] (blocks)
    stride_vs_slot: tl.int64,  # v_scale_cache stride[1] (slots)
    stride_vs_head: tl.int64,  # v_scale_cache stride[2] (heads)
    block_size: tl.constexpr,
    head_size: tl.constexpr,
    head_size_v: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,  # next_power_of_2(max(head_size, head_size_v))
    QUANT_MAX: tl.constexpr = 127.0,
    QUANT_MIN: tl.constexpr = -128.0,
):
    tok = tl.program_id(0)
    head = tl.program_id(1)

    # 通过 slot_mapping 将 token 索引映射到缓存中的物理位置
    slot = tl.load(slot_mapping_ptr + tok).to(tl.int64)
    if slot < 0:
        return

    # 计算页号和页内偏移
    blk = slot // block_size
    slot_in_blk = slot % block_size

    dim_offs = tl.arange(0, HEAD_SIZE_PADDED)

    # ---- Key 处理：加载一个 head → 计算 absmax → 量化 → 存储 ----------
    k_mask = dim_offs < head_size
    # 加载该 (token, head) 的 K 数据
    k_h = tl.load(
        key_ptr + tok * stride_key_tok + head * stride_key_head + dim_offs,
        mask=k_mask,
        other=0.0,
    ).to(tl.float32)

    # 计算动态缩放因子：scale = absmax / QUANT_MAX，最小值 1e-6 防止除零
    k_scale = tl.maximum(tl.max(tl.abs(k_h)) / QUANT_MAX, 1e-6)
    # 存储缩放因子到 scale_cache
    tl.store(
        k_scale_cache_ptr
        + blk * stride_ks_blk
        + slot_in_blk * stride_ks_slot
        + head * stride_ks_head,
        k_scale,
    )

    # 量化：将值缩放到 [QUANT_MIN, QUANT_MAX] 范围
    k_q = tl.clamp(k_h * (1.0 / k_scale), QUANT_MIN, QUANT_MAX)
    tl.store(
        key_cache_ptr
        + blk * stride_kc_blk
        + slot_in_blk * stride_kc_slot
        + head * stride_kc_head
        + dim_offs,
        k_q,
        mask=k_mask,
    )

    # ---- Value 处理：与 Key 相同的逐 head 量化方法 ---------------------
    v_mask = dim_offs < head_size_v
    v_h = tl.load(
        value_ptr + tok * stride_val_tok + head * stride_val_head + dim_offs,
        mask=v_mask,
        other=0.0,
    ).to(tl.float32)

    v_scale = tl.maximum(tl.max(tl.abs(v_h)) / QUANT_MAX, 1e-6)
    tl.store(
        v_scale_cache_ptr
        + blk * stride_vs_blk
        + slot_in_blk * stride_vs_slot
        + head * stride_vs_head,
        v_scale,
    )

    v_q = tl.clamp(v_h * (1.0 / v_scale), QUANT_MIN, QUANT_MAX)
    tl.store(
        value_cache_ptr
        + blk * stride_vc_blk
        + slot_in_blk * stride_vc_slot
        + head * stride_vc_head
        + dim_offs,
        v_q,
        mask=v_mask,
    )


# 从缓存 torch dtype 到 (QUANT_MAX, QUANT_MIN) 的映射表
# 用于逐 token-head 量化 kernel 的参数选择
_PER_TOKEN_HEAD_QUANT_PARAMS: dict[torch.dtype, tuple[float, float]] = {
    torch.int8: (127.0, -128.0),
    FP8_DTYPE: (FP8_MAX, FP8_MIN),
}


def triton_reshape_and_cache_flash_per_token_head_quant(
    key: torch.Tensor,  # [num_tokens, num_kv_heads, head_size]
    value: torch.Tensor,  # [num_tokens, num_kv_heads, head_size_v]
    key_cache: torch.Tensor,  # [num_blocks, block_size, num_kv_heads, head_size]
    value_cache: torch.Tensor,  # [num_blocks, block_size, num_kv_heads, head_size_v]
    k_scale_cache: torch.Tensor,  # [num_blocks, block_size, num_kv_heads] float32
    v_scale_cache: torch.Tensor,  # [num_blocks, block_size, num_kv_heads] float32
    slot_mapping: torch.Tensor,  # [num_tokens]
):
    """Quantize key/value per (token, head) and write to paged cache.

    Computes one scale = absmax / QUANT_MAX per (token, head), stores
    quantized data in key_cache/value_cache, and stores the float32
    scale in k_scale_cache/v_scale_cache.

    The quantization range (QUANT_MAX, QUANT_MIN) is derived from the
    cache tensor dtype so the same code path works for int8 and fp8.

    逐 (token, head) 量化 Key/Value 并写入分页缓存。

    每个 (token, head) 对计算一个缩放因子 scale = absmax / QUANT_MAX，
    将量化后的数据存入 key_cache/value_cache，
    将 float32 缩放因子存入 k_scale_cache/v_scale_cache。

    量化范围 (QUANT_MAX, QUANT_MIN) 从缓存张量的 dtype 推导，
    因此同一代码路径支持 int8 和 fp8。
    """
    cache_dtype = key_cache.dtype
    quant_params = _PER_TOKEN_HEAD_QUANT_PARAMS.get(cache_dtype)
    if quant_params is None:
        raise ValueError(
            f"Per-token-head quantization not supported for cache dtype "
            f"{cache_dtype}.  Supported: {list(_PER_TOKEN_HEAD_QUANT_PARAMS)}"
        )
    quant_max, quant_min = quant_params

    num_tokens, num_kv_heads, head_size = key.shape
    head_size_v = value.shape[2]
    head_size_padded = triton.next_power_of_2(max(head_size, head_size_v))

    block_size = key_cache.shape[1]

    if current_platform.is_rocm() or current_platform.is_xpu():
        num_warps = 4
    else:
        num_warps = min(16, max(1, head_size_padded // 32))

    _reshape_cache_per_token_head[(num_tokens, num_kv_heads)](
        key_ptr=key,
        value_ptr=value,
        key_cache_ptr=key_cache,
        value_cache_ptr=value_cache,
        k_scale_cache_ptr=k_scale_cache,
        v_scale_cache_ptr=v_scale_cache,
        slot_mapping_ptr=slot_mapping,
        stride_key_tok=key.stride(0),
        stride_key_head=key.stride(1),
        stride_val_tok=value.stride(0),
        stride_val_head=value.stride(1),
        stride_kc_blk=key_cache.stride(0),
        stride_kc_slot=key_cache.stride(1),
        stride_kc_head=key_cache.stride(2),
        stride_vc_blk=value_cache.stride(0),
        stride_vc_slot=value_cache.stride(1),
        stride_vc_head=value_cache.stride(2),
        stride_ks_blk=k_scale_cache.stride(0),
        stride_ks_slot=k_scale_cache.stride(1),
        stride_ks_head=k_scale_cache.stride(2),
        stride_vs_blk=v_scale_cache.stride(0),
        stride_vs_slot=v_scale_cache.stride(1),
        stride_vs_head=v_scale_cache.stride(2),
        block_size=block_size,
        head_size=head_size,
        head_size_v=head_size_v,
        HEAD_SIZE_PADDED=head_size_padded,
        QUANT_MAX=quant_max,
        QUANT_MIN=quant_min,
        num_warps=num_warps,
    )


def triton_reshape_and_cache_flash(
    key: torch.Tensor,  # [num_tokens, num_heads, head_size]
    value: torch.Tensor,  # [num_tokens, num_heads, head_size]
    # [num_blocks, block_size, num_heads, head_size]
    key_cache: torch.Tensor,
    # [num_blocks, block_size, num_heads, head_size]
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,  # [num_tokens]
    kv_cache_dtype: str,  # "auto", "fp8"
    k_scale: torch.Tensor,  # float32
    v_scale: torch.Tensor,  # float32
):
    """标准 KV 缓存写入的 Python 封装函数。

    工作流程：
    1. 检测内存布局（head-major 5D 或 block-major 4D）
    2. 检查 KV 缓存数据类型是否受支持
    3. 如果是 FP8 量化类型，将缓存张量转换为正确的 FP8 dtype
    4. 根据平台和硬件能力选择启发式参数（TILE_SIZE, num_warps, num_stages）
    5. 启动 reshape_and_cache_kernel_flash Triton kernel
    """
    num_heads = key.shape[1]
    head_size = key.shape[2]

    # 检测内存布局：5D 缓存使用 head-major 布局
    use_head_major_layout = key_cache.ndim == 5
    if use_head_major_layout:
        block_size = key_cache.shape[3]
        x = key_cache.shape[4]
        head_stride = key_cache.stride(1)
        dim_stride_k = key_cache.stride(2)
        dim_stride_v = value_cache.stride(2)
    else:
        block_size = key_cache.shape[1]
        x = 1
        dim_stride_k = 0
        dim_stride_v = 0
        head_stride = key_cache.stride()[2]
    n = num_heads * head_size
    key_stride = key.stride()[0]
    value_stride = value.stride()[0]
    block_stride = key_cache.stride()[0]
    page_stride = key_cache.stride()[1]

    assert _is_supported_kv_cache_dtype(kv_cache_dtype), (
        f"unsupported kv_cache_dtype (str), got {kv_cache_dtype}."
    )
    kv_cache_torch_dtype = (
        current_platform.fp8_dtype()
        if is_quantized_kv_cache(kv_cache_dtype)
        else key_cache.dtype
    )

    if key_cache.dtype != kv_cache_torch_dtype and is_quantized_kv_cache(
        kv_cache_dtype
    ):
        # to avoid erounous implicit cast in triton kernel (tl.store to uint8)
        # (e.g. explicit cast to fp8e4m3fnuz is not supported in triton 3.4)
        key_cache = key_cache.view(kv_cache_torch_dtype)
        value_cache = value_cache.view(kv_cache_torch_dtype)
    assert kv_cache_dtype != torch.uint8, (
        "explicit fp8 cast and store to "
        "uint8 is not supported by triton reshape_and_cache_flash"
    )

    FP8_KV_CACHE = is_quantized_kv_cache(kv_cache_dtype)
    assert (not FP8_KV_CACHE) or kv_cache_torch_dtype in [
        torch.float8_e4m3fn,
        torch.float8_e5m2,
        torch.uint8,
        torch.float8_e4m3fnuz,
    ], (
        "unsupported dtype of KV cache tensor, got "
        "{kv_cache_torch_dtype}. Supported kv cache dtypes: fp8e4m3fn, "
        "fp8e5m2, uint8, bfloat16, float16, float32, fp8e4m3fnuz."
    )

    # 启发式参数选择（替代自动调优以减少编译时间）
    # TILE_SIZE: 每个程序处理的元素数量，取 2048 和 n 的 2 的幂次的较小值
    TILE_SIZE = min(2048, triton.next_power_of_2(n))
    if current_platform.is_rocm() or current_platform.is_xpu():
        num_stages = 4
        num_warps = 8
    else:  # cuda
        num_stages = 10
        num_warps = 16
        # 对于旧款 GPU（compute capability < 9，如 A100），减小 TILE_SIZE
        if torch.cuda.get_device_capability(key.device)[0] < 9:
            TILE_SIZE = min(512, TILE_SIZE)

    # TODO(ngl): maybe replace with static launch grid to avoid overhead if
    #   using cudagraphs
    grid = lambda meta: (
        slot_mapping.shape[0],
        triton.cdiv(n, meta["TILE_SIZE"]),
    )

    reshape_and_cache_kernel_flash[grid](
        key_ptr=key,
        value_ptr=value,
        key_cache_ptr=key_cache,
        value_cache_ptr=value_cache,
        slot_mapping_ptr=slot_mapping,
        k_scale=k_scale,
        v_scale=v_scale,
        # strides
        key_stride=key_stride,
        value_stride=value_stride,
        block_stride=block_stride,
        head_stride=head_stride,
        dim_stride_k=dim_stride_k,
        dim_stride_v=dim_stride_v,
        page_stride=page_stride,
        num_heads=num_heads,
        head_size=head_size,
        block_size=block_size,
        x=x,
        USE_HEAD_MAJOR_LAYOUT=use_head_major_layout,
        FP8_KV_CACHE=FP8_KV_CACHE,
        # autotune parameters
        TILE_SIZE=TILE_SIZE,
        num_warps=num_warps,
        num_stages=num_stages,
    )


@triton.jit
def reshape_and_cache_kernel_flash_diffkv(
    key_ptr,  # [num_tokens, num_heads, head_size]
    value_ptr,  # [num_tokens, num_heads, head_size_v]
    kv_cache_ptr,  # [num_blocks, block_size, num_heads, head_size + head_size_v]
    slot_mapping_ptr,  # [num_tokens]
    k_scale,  # float32
    v_scale,  # float32
    # strides
    key_stride: tl.int64,
    value_stride: tl.int64,
    block_stride: tl.int64,
    page_stride: tl.int64,
    num_heads: tl.constexpr,
    head_size_k: tl.constexpr,
    head_size_v: tl.constexpr,
    block_size: tl.constexpr,
    # FP8 flags
    FP8_KV_CACHE: tl.constexpr,
    # tune parameters
    TILE_SIZE: tl.constexpr,
):
    """
    差异化 KV 头维度的缓存写入 kernel（用于 MLA 等场景）。

    当 K 和 V 的头维度不同时（如 MLA 中 K 的 head_size=576, V 的 head_size=512），
    使用统一的 kv_cache 缓存，将 K 和 V 拼接存储：
    kv_cache [num_blocks, block_size, num_heads, head_size_k + head_size_v]

    网格维度：(num_tokens, num_heads)
    每个程序处理一个 (token, head) 对的全部 K 和 V 数据。
    """
    token_idx = tl.program_id(axis=0)
    slot_idx = tl.load(slot_mapping_ptr + token_idx).to(tl.int64)
    if slot_idx < 0:
        # Padding token that should be ignored.
        return

    tile_i = tl.program_id(axis=1)
    tile_offs = tl.arange(0, TILE_SIZE)

    block_idx = slot_idx // block_size
    block_offset = slot_idx % block_size

    src_key_idx = token_idx * key_stride + tile_i * head_size_k
    src_value_idx = token_idx * value_stride + tile_i * head_size_v

    tgt_idx = (
        block_idx * block_stride
        + block_offset * page_stride
        + tile_i * (head_size_k + head_size_v)
    )

    # [TILE_SIZE]
    key_load = tl.load(key_ptr + src_key_idx + tile_offs, mask=tile_offs < head_size_k)
    if FP8_KV_CACHE:
        # tl.store will do the correct implicit cast to fp8,
        # based on the key_cache_ptr.dtype.element_ty
        key_tile = key_load if key_load.dtype.is_fp8() else key_load / tl.load(k_scale)
    else:
        key_tile = key_load

    # [TILE_SIZE]
    value_load = tl.load(
        value_ptr + src_value_idx + tile_offs, mask=tile_offs < head_size_v
    )
    if FP8_KV_CACHE:
        if value_load.dtype.is_fp8():
            value_tile = value_load
        else:
            # tl.store will do the correct implicit cast to fp8,
            #  based on the value_cache_ptr.dtype.element_ty
            value_tile = value_load / tl.load(v_scale)
    else:
        value_tile = value_load

    tl.store(
        kv_cache_ptr + tgt_idx + tile_offs,
        key_tile,
        mask=tile_offs < head_size_k,
    )
    tl.store(
        kv_cache_ptr + tgt_idx + head_size_k + tile_offs,
        value_tile,
        mask=tile_offs < head_size_v,
    )
    return


def triton_reshape_and_cache_flash_diffkv(
    key: torch.Tensor,  # [num_tokens, num_heads, head_size]
    value: torch.Tensor,  # [num_tokens, num_heads, head_size_v]
    # [num_blocks, block_size, num_heads, head_size + head_size_v]
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,  # [num_tokens]
    kv_cache_dtype: str,  # "auto", "fp8"
    k_scale: torch.Tensor,  # float32
    v_scale: torch.Tensor,  # float32
):
    """差异化 KV 头维度缓存写入的 Python 封装函数。

    用于 K 和 V 头维度不同的场景（如 MLA），
    将 K 和 V 拼接后写入统一的 kv_cache 缓存。
    """
    num_heads = key.shape[1]
    head_size_k = key.shape[2]
    head_size_v = value.shape[2]
    block_size = kv_cache.shape[1]

    k_stride = key.stride()[0]
    v_stride = value.stride()[0]
    block_stride = kv_cache.stride()[0]
    page_stride = kv_cache.stride()[1]

    assert _is_supported_kv_cache_dtype(kv_cache_dtype), (
        f"unsupported kv_cache_dtype (str), got {kv_cache_dtype}."
    )
    kv_cache_torch_dtype = (
        current_platform.fp8_dtype()
        if is_quantized_kv_cache(kv_cache_dtype)
        else kv_cache.dtype
    )

    if kv_cache.dtype != kv_cache_torch_dtype and is_quantized_kv_cache(kv_cache_dtype):
        # to avoid erounous implicit cast in triton kernel (tl.store to uint8)
        # (e.g. explicit cast to fp8e4m3fnuz is not supported in triton 3.4)
        kv_cache = kv_cache.view(kv_cache_torch_dtype)
    assert kv_cache_dtype != torch.uint8, (
        "explicit fp8 cast and store to "
        "uint8 is not supported by triton reshape_and_cache_flash_diffkv"
    )

    FP8_KV_CACHE = is_quantized_kv_cache(kv_cache_dtype)
    assert (not FP8_KV_CACHE) or kv_cache_torch_dtype in [
        torch.float8_e4m3fn,
        torch.float8_e5m2,
        torch.uint8,
        torch.float8_e4m3fnuz,
    ], (
        "unsupported dtype of KV cache tensor, got "
        "{kv_cache_torch_dtype}. Supported kv cache dtypes: fp8e4m3fn, "
        "fp8e5m2, uint8, bfloat16, float16, float32, fp8e4m3fnuz."
    )

    # 启发式参数选择：TILE_SIZE 取 K 和 V 头维度的较大值的 2 的幂次
    TILE_SIZE = max(head_size_k, head_size_v)
    TILE_SIZE = triton.next_power_of_2(TILE_SIZE)
    if current_platform.is_rocm() or current_platform.is_xpu():
        num_stages = 4
        num_warps = 8
    else:  # cuda
        num_stages = 10
        num_warps = 16

    # TODO(ngl): maybe replace with static launch grid to avoid overhead if
    #   using cudagraphs
    grid = lambda meta: (
        slot_mapping.shape[0],
        num_heads,
    )

    reshape_and_cache_kernel_flash_diffkv[grid](
        key_ptr=key,
        value_ptr=value,
        kv_cache_ptr=kv_cache,
        slot_mapping_ptr=slot_mapping,
        k_scale=k_scale,
        v_scale=v_scale,
        # strides
        key_stride=k_stride,
        value_stride=v_stride,
        block_stride=block_stride,
        page_stride=page_stride,
        num_heads=num_heads,
        head_size_k=head_size_k,
        head_size_v=head_size_v,
        block_size=block_size,
        # FP8 flags
        FP8_KV_CACHE=FP8_KV_CACHE,
        # autotune parameters
        TILE_SIZE=TILE_SIZE,
        num_warps=num_warps,
        num_stages=num_stages,
    )
