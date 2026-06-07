# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Gumbel-Top-K 采样模块 (Gumbel-Top-K Sampling Module)

本模块实现了基于 Gumbel-Max 技术的采样算法，用于从 logits 分布中采样 token。

Gumbel-Max 采样的核心思想：
1. 对 logits 加上 Gumbel 噪声：logits + gumbel_noise
2. 对加噪后的 logits 取 argmax，等价于从 softmax(logits) 分布中采样

Gumbel 噪声的生成：gumbel_noise = -log(-log(u))，其中 u 是 (0,1) 均匀分布的随机数。

实现细节：
1. 温度缩放 (Temperature Scaling): logits = logits / temperature
2. 分块处理：将 vocab_size 分成多个 block 并行计算
3. 两阶段归约：先在每个 block 内找最大值，再跨 block 找全局最大值
4. 支持 FP32 和 FP64 两种精度的 Gumbel 噪声生成

为什么使用 Gumbel-Max 而不是直接 softmax + multinomial：
- 避免显式计算完整的 softmax 概率分布（节省内存）
- 数值稳定性更好
- 与 top-k/top-p 筛选自然兼容
"""
import torch

from vllm.triton_utils import HAS_TRITON, tl, triton

# 最小的正 fp32 规格化值。用于钳制均匀分布采样值，使得
# `log(u)` 不会产生 -inf（从而 `-log(-log(u))` 保持有限）。
#
# Triton 要求从 `@triton.jit` 函数访问的全局变量必须用
# `tl.constexpr(...)` 包装。只有在 Triton 可用时才能这样做——
# 在 CPU worker 路径上 `tl` 是一个占位符，其 `constexpr` 属性为 `None`，
# `tl.constexpr(...)` 会在导入时崩溃。
_FP32_TINY = (
    tl.constexpr(float.fromhex("0x1p-126")) if HAS_TRITON else float.fromhex("0x1p-126")
)


@triton.jit
def _temperature_kernel(
    logits_ptr,
    logits_stride,
    expanded_idx_mapping_ptr,
    temperature_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    """温度缩放 Triton 内核。

    对每个 token 的 logits 除以对应的温度参数。
    温度的作用：
    - temperature > 1: 使分布更平坦（更随机）
    - temperature < 1: 使分布更尖锐（更确定）
    - temperature = 0: 退化为贪心采样（argmax）
    - temperature = 1: 不做任何缩放

    Args:
        logits_ptr: logits 数据指针
        logits_stride: logits 的行步长
        expanded_idx_mapping_ptr: 扩展索引映射指针
        temperature_ptr: 温度参数指针
        vocab_size: 词表大小
        BLOCK_SIZE: 每个 block 处理的 token 数量（编译时常量）
    """
    token_idx = tl.program_id(0)
    req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx)
    temperature = tl.load(temperature_ptr + req_state_idx).to(tl.float32)
    if temperature == 0.0 or temperature == 1.0:
        # 提前返回以避免加载 logits（温度为 0 或 1 时无需缩放）
        return

    block_idx = tl.program_id(1)
    block = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block < vocab_size

    logits = tl.load(logits_ptr + token_idx * logits_stride + block, mask=mask)
    logits = logits.to(tl.float32)
    logits = logits / temperature
    tl.store(logits_ptr + token_idx * logits_stride + block, logits, mask=mask)


def apply_temperature(
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    temperature: torch.Tensor,
) -> None:
    """应用温度缩放到 logits。

    Args:
        logits: logits 张量 [num_tokens, vocab_size]
        expanded_idx_mapping: 扩展的请求索引映射 [num_tokens]
        temperature: 温度参数 [max_num_reqs]
    """
    num_tokens, vocab_size = logits.shape
    BLOCK_SIZE = 8192
    num_blocks = triton.cdiv(vocab_size, BLOCK_SIZE)
    _temperature_kernel[(num_tokens, num_blocks)](
        logits,
        logits.stride(0),
        expanded_idx_mapping,
        temperature,
        vocab_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )


@triton.jit
def tl_rand64(seed, offset, includes_zero: tl.constexpr):
    """生成 64 位精度的均匀随机数。

    使用 Triton 的 randint4x 生成 4 个 32 位随机数，组合成 64 位随机数，
    然后缩放到 [0, 1) 或 (0, 1) 范围。

    Args:
        seed: 随机种子
        offset: 偏移量
        includes_zero: 是否包含 0（False 时钳制为最小正 fp64 值）

    Returns:
        fp64 均匀随机数
    """
    lo, hi, _, _ = tl.randint4x(seed, offset)
    lo = lo.to(tl.uint32, bitcast=True).to(tl.uint64)
    hi = hi.to(tl.uint32, bitcast=True).to(tl.uint64)
    r = (hi << 32) | lo

    # 1 / 2**64
    scale = 5.421010862427522170037e-20
    u = r.to(tl.float64) * scale
    if not includes_zero:
        u = tl.maximum(u, 2.2250738585072014e-308)  # float64 tiny
    return u


@triton.jit
def gumbel_block_argmax(
    logits,
    block,
    mask,
    token_idx,
    expanded_idx_mapping_ptr,
    temp_ptr,
    seeds_ptr,
    pos_ptr,
    processed_logits_ptr,
    processed_logits_stride,
    processed_logits_col_ptr,
    vocab_size,
    APPLY_TEMPERATURE: tl.constexpr,
    USE_FP64: tl.constexpr,
):
    """在单个 block 内执行 Gumbel-Max 的 argmax 操作。

    处理流程：
    1. 如果需要温度缩放，将 logits 除以温度
    2. 如果需要，存储处理后的 logits（用于 logprobs 计算）
    3. 如果温度不为 0，生成 Gumbel 噪声并加到 logits 上
    4. 在 block 内找最大值及其索引

    Gumbel-Max 定理：argmax(logits + gumbel_noise) 等价于从 softmax(logits) 中采样。

    Args:
        logits: 当前 block 的 logits 数据
        block: 当前 block 的索引范围
        mask: 有效位置的掩码
        token_idx: token 的全局索引
        其他: 各种状态指针和配置常量

    Returns:
        (value, idx): block 内的最大值和对应的局部索引
    """
    req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx)
    temp = tl.load(temp_ptr + req_state_idx).to(tl.float32)
    if temp != 0.0 and APPLY_TEMPERATURE:
        # 应用温度缩放。
        # 注意：需要与 _temperature_kernel 的行为保持一致。
        # 例如，如果内核使用 tl.div_rn，这里也应该使用 tl.div_rn。
        logits = logits / temp

    if processed_logits_ptr is not None:
        # 存储应用温度后的 logits（用于后续 logprobs 计算）
        if processed_logits_col_ptr is not None:
            col = tl.load(processed_logits_col_ptr)
        else:
            col = 0
        tl.store(
            processed_logits_ptr
            + req_state_idx * processed_logits_stride
            + col * vocab_size
            + block,
            logits,
            mask=mask,
        )

    # fp32 是默认的归约数据类型；fp64 在 H100/Ada/Blackwell 上的吞吐量约为
    # fp32 的 1/32-1/64，但对 Gumbel-max 来说经验上没有区别。
    if USE_FP64:
        logits = logits.to(tl.float64)
    if temp != 0.0:
        # 计算 Gumbel 噪声的种子
        seed = tl.load(seeds_ptr + req_state_idx)
        pos = tl.load(pos_ptr + token_idx)
        gumbel_seed = tl.randint(seed, pos)

        if USE_FP64:
            u = tl_rand64(gumbel_seed, block, includes_zero=False)
        else:
            u = tl.rand(gumbel_seed, block)
            u = tl.maximum(u, _FP32_TINY)
        gumbel_noise = -tl.log(-tl.log(u))

        # 应用 Gumbel 噪声到 logits
        logits = tl.where(mask, logits + gumbel_noise, float("-inf"))

    value, idx = tl.max(logits, axis=0, return_indices=True)
    return value, idx


@triton.jit
def _gumbel_sample_kernel(
    local_argmax_ptr,
    local_argmax_stride,
    local_max_ptr,
    local_max_stride,
    processed_logits_ptr,
    processed_logits_stride,
    processed_logits_col_ptr,
    logits_ptr,
    logits_stride,
    expanded_idx_mapping_ptr,
    seeds_ptr,
    pos_ptr,
    temp_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
    APPLY_TEMPERATURE: tl.constexpr,
    USE_FP64: tl.constexpr,
):
    """Gumbel 采样 Triton 内核。

    每个 (token_idx, block_idx) 组合启动一个线程，在对应的 vocab block 内
    执行 Gumbel-Max 操作，找到局部最大值和对应的 token ID。

    Args:
        local_argmax_ptr: 局部 argmax 结果指针 [num_tokens, num_blocks]
        local_max_ptr: 局部最大值指针 [num_tokens, num_blocks]
        processed_logits_ptr: 处理后的 logits 指针（用于 logprobs，可选）
        logits_ptr: 原始 logits 指针
        其他: 各种状态指针和配置常量
    """
    token_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    block = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block < vocab_size
    logits = tl.load(
        logits_ptr + token_idx * logits_stride + block,
        mask=mask,
        other=float("-inf"),
    )
    logits = logits.to(tl.float32)

    value, idx = gumbel_block_argmax(
        logits,
        block,
        mask,
        token_idx,
        expanded_idx_mapping_ptr,
        temp_ptr,
        seeds_ptr,
        pos_ptr,
        processed_logits_ptr,
        processed_logits_stride,
        processed_logits_col_ptr,
        vocab_size,
        APPLY_TEMPERATURE=APPLY_TEMPERATURE,
        USE_FP64=USE_FP64,
    )
    token_id = block_idx * BLOCK_SIZE + idx
    tl.store(local_argmax_ptr + token_idx * local_argmax_stride + block_idx, token_id)
    tl.store(local_max_ptr + token_idx * local_max_stride + block_idx, value)


def gumbel_sample(
    logits: torch.Tensor,  # [num_tokens, vocab_size]
    expanded_idx_mapping: torch.Tensor,  # [num_tokens]
    temperature: torch.Tensor,  # [max_num_reqs]
    seed: torch.Tensor,  # [max_num_reqs]
    pos: torch.Tensor,  # [num_tokens]
    apply_temperature: bool,
    output_processed_logits: torch.Tensor | None = None,
    output_processed_logits_col: torch.Tensor | None = None,
    use_fp64: bool = False,
) -> torch.Tensor:
    """执行 Gumbel-Top-K 采样。

    算法流程：
    1. 将 vocab_size 分成多个 block（默认 BLOCK_SIZE=1024）
    2. 每个 block 内独立执行 Gumbel-Max 操作，找到局部最大值和对应的 token ID
    3. 在所有 block 的局部最大值中找全局最大值
    4. 全局最大值对应的 token ID 即为采样结果

    Args:
        logits: logits 张量 [num_tokens, vocab_size]
        expanded_idx_mapping: 扩展的请求索引映射 [num_tokens]
        temperature: 温度参数 [max_num_reqs]
        seed: 随机种子 [max_num_reqs]
        pos: 每个 token 的位置 [num_tokens]
        apply_temperature: 是否在此内核中应用温度（如果为 False，logits 应已预处理）
        output_processed_logits: 可选，存储处理后的 logits（用于 logprobs 计算）
        output_processed_logits_col: 可选，存储处理后的 logits 的列索引
        use_fp64: 是否使用 FP64 精度的 Gumbel 噪声

    Returns:
        采样的 token IDs [num_tokens]
    """
    num_tokens, vocab_size = logits.shape
    BLOCK_SIZE = 1024
    num_blocks = triton.cdiv(vocab_size, BLOCK_SIZE)
    local_argmax = logits.new_empty(num_tokens, num_blocks, dtype=torch.int64)
    local_max_dtype = torch.float64 if use_fp64 else torch.float32
    local_max = logits.new_empty(num_tokens, num_blocks, dtype=local_max_dtype)
    _gumbel_sample_kernel[(num_tokens, num_blocks)](
        local_argmax,
        local_argmax.stride(0),
        local_max,
        local_max.stride(0),
        output_processed_logits,
        output_processed_logits.stride(0) if output_processed_logits is not None else 0,
        output_processed_logits_col,
        logits,
        logits.stride(0),
        expanded_idx_mapping,
        seed,
        pos,
        temperature,
        vocab_size,
        BLOCK_SIZE=BLOCK_SIZE,
        APPLY_TEMPERATURE=apply_temperature,
        USE_FP64=use_fp64,
    )
    # 找到全局最大值所在的 block 索引
    # 注意：使用 int64 以便后续索引操作
    max_block_idx = local_max.argmax(dim=-1, keepdim=True)
    # 从对应的 block 中取出采样的 token ID
    sampled = local_argmax.gather(dim=-1, index=max_block_idx).view(-1)
    return sampled
