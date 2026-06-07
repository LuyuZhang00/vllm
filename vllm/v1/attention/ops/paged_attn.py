# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# 中文注释：本文件实现了 PagedAttention 的核心操作，即分页 KV cache 的读写接口。
# PagedAttention 是 vLLM 的核心创新之一，其核心思想是：
#   将 KV cache 划分为固定大小的 block（类似操作系统的内存页），
#   每个请求通过 block table（页表）将逻辑 token 位置映射到物理显存中的 block 位置。
# 这样不同请求的 KV cache 不需要在物理显存上连续存放，大大提高了显存利用率。
#
# 本文件提供两个核心操作：
#   1. split_kv_cache：将紧凑的 KV cache 张量拆分为 key_cache 和 value_cache，
#      并 reshape 为 attention kernel 期望的 5D/4D 格式。
#   2. write_to_paged_cache：将新计算的 K/V 向量写入到 KV cache 的指定物理位置
#     （通过 slot_mapping 定位）。

import torch

from vllm.platforms import current_platform

# 中文注释：根据平台选择底层操作库。CUDA/XPU 使用各自优化的 C++ 扩展。
if current_platform.is_cuda_alike():
    from vllm import _custom_ops as ops
elif current_platform.is_xpu():
    from vllm._xpu_ops import xpu_ops as ops  # type: ignore[no-redef]


class PagedAttention:
    # 中文注释：PagedAttention 操作类，封装了 KV cache 的拆分和写入操作。
    # 所有方法都是静态方法，无需实例化。
    @staticmethod
    def split_kv_cache(
        kv_cache: torch.Tensor,
        num_kv_heads: int,
        head_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 中文注释：将紧凑的 KV cache 张量拆分为 key_cache 和 value_cache。
        #
        # 【输入格式】
        #   kv_cache: [2, num_blocks, ...] 的紧凑张量，第 0 维区分 key(0) 和 value(1)。
        #   在 vLLM V1 中，KV cache 的物理布局为：
        #     key:   [num_blocks, num_kv_heads, head_size/x, block_size, x]
        #     value: [num_blocks, num_kv_heads, head_size, block_size]
        #   其中 x = 16 / element_size，用于 key cache 的 swizzle 优化
        #  （将 head_size 维度分块以提高 GPU 内存访问合并度）。
        #
        # 【输出格式】
        #   key_cache:   [num_blocks, num_kv_heads, head_size/x, block_size, x] (5D)
        #   value_cache: [num_blocks, num_kv_heads, head_size, block_size]       (4D)
        #
        # 【x 的含义】
        #   x = 16 / element_size 表示 16 字节能容纳的元素个数。
        #   对于 fp16/bf16 (2 bytes)：x = 8
        #   对于 fp32 (4 bytes)：x = 4
        #   这种分块方式使得 key cache 在 head_size 维度上按 16 字节对齐，
        #   提高 GPU 全局内存的合并访问 (coalesced access) 效率。
        x = 16 // kv_cache.element_size()
        num_blocks = kv_cache.shape[1]

        key_cache = kv_cache[0]
        key_cache = key_cache.view(num_blocks, num_kv_heads, head_size // x, -1, x)
        value_cache = kv_cache[1]
        value_cache = value_cache.view(num_blocks, num_kv_heads, head_size, -1)
        return key_cache, value_cache

    @staticmethod
    def write_to_paged_cache(
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        k_scale: torch.Tensor,
        v_scale: torch.Tensor,
    ) -> None:
        # 中文注释：将新计算的 K/V 向量写入到分页 KV cache 的指定物理位置。
        #
        # 【参数说明】
        #   key, value: 当前 step 新计算的 K/V 张量，shape 通常为 [num_tokens, num_kv_heads, head_size]
        #   key_cache, value_cache: 分页 KV cache，由 split_kv_cache 得到
        #   slot_mapping: 物理槽位映射表 [num_tokens]，每个元素是一个整数，
        #     表示该 token 的 KV 应写入 KV cache 中的第几个 slot。
        #     slot_mapping 由 KV cache manager 根据 block table 生成。
        #   kv_cache_dtype: KV cache 的数据类型字符串（如 "auto", "fp8" 等）
        #   k_scale, v_scale: FP8 量化的缩放因子
        #
        # 【实现】
        #   调用 C++ 扩展 ops.reshape_and_cache，它会：
        #     1. 将 key 从 [num_tokens, num_kv_heads, head_size] reshape 为 key_cache 的格式
        #     2. 将 value 从 [num_tokens, num_kv_heads, head_size] reshape 为 value_cache 的格式
        #     3. 根据 slot_mapping 将数据 scatter 写入到物理显存位置
        #   使用 C++ 实现是为了避免 Python 循环，利用 GPU 的并行写入能力。
        ops.reshape_and_cache(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping.flatten(),
            kv_cache_dtype,
            k_scale,
            v_scale,
        )
