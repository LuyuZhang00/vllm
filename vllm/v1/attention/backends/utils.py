# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# =============================================================================
# 中文说明：vLLM V1 Attention Backends 工具模块
# =============================================================================
# 本文件是 vLLM V1 推理引擎中 attention 后端的核心工具模块，提供了以下功能：
#
# 1. KV cache 布局管理：
#    - 支持 "NHD"（num_heads, head_dim）和 "HND"（head_dim, num_heads）两种布局
#    - 通过环境变量或代码覆盖来设置 KV cache 的内存布局
#
# 2. 每层注意力参数管理：
#    - 提取每个注意力层的超参数（滑动窗口大小、logits soft cap、缩放因子等）
#    - FlashInfer 等后端要求所有层共享相同参数，本模块负责验证和推断
#
# 3. 本地注意力（Local Attention）虚拟批次构建：
#    - 将长序列按 chunk 拆分为多个"虚拟批次项"，使得标准 FlashAttention
#      可以模拟局部注意力掩码，而无需自定义掩码矩阵
#
# 4. Batch 重排序（Decode/Prefill 分离）：
#    - 将混合 batch 重排序为 decode -> short_extend -> long_extend -> prefill
#    - 这使得注意力后端可以对不同类型的请求使用不同的计算路径
#
# 5. 投机解码（Speculative Decoding）支持：
#    - 为投机解码场景提供 query tensor 的 reshape 工具
#
# 6. Mamba 状态空间模型支持：
#    - 为 Mamba 内核提供 block table 的适配逻辑
# =============================================================================

import functools
from collections.abc import Callable
from dataclasses import dataclass, field, fields, make_dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Protocol,
    get_args,
)

import numpy as np
import torch
from typing_extensions import runtime_checkable

from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.utils.math_utils import cdiv
from vllm.v1.kv_cache_interface import KVCacheSpec, MambaSpec

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.worker.gpu_input_batch import InputBatch

import vllm.envs as envs
from vllm.distributed.kv_transfer.kv_connector.utils import (
    get_kv_connector_cache_layout,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    CommonAttentionMetadata,
    subclass_attention_backend,
)

logger = init_logger(__name__)

# 中文注释：KV cache 布局类型，决定 K/V 张量在显存中的维度排列顺序。
# "NHD" 表示 (num_tokens, num_heads, head_dim)，"HND" 表示 (num_tokens, head_dim, num_heads)。
# 不同布局对硬件的内存访问模式有影响，可能影响 kernel 性能。
KVCacheLayoutType = Literal["NHD", "HND"]

# 中文注释：全局 KV cache 布局覆盖变量。当非 None 时，优先级最高，
# 用于代码层面强制指定 KV cache 布局（通常用于测试或特殊场景）。
_KV_CACHE_LAYOUT_OVERRIDE: KVCacheLayoutType | None = None

# 中文注释：PAD_SLOT_ID = -1，表示无效/填充的 slot 位置。
# 在 slot mapping 中用于标记不需要写入 KV cache 的填充 token。
PAD_SLOT_ID = -1

# 中文注释：NULL_BLOCK_ID = 0，表示空/无效的 block ID。
# 在 block table 中用于标记未分配 KV block 的位置。
NULL_BLOCK_ID = 0


def is_valid_kv_cache_layout(value: str) -> bool:
    # 中文注释：检查给定字符串是否为合法的 KV cache 布局类型。
    # 合法值为 "NHD" 或 "HND"，通过 Literal 类型的 get_args 获取。
    return value in get_args(KVCacheLayoutType)


@functools.lru_cache
def get_kv_cache_layout():
    # 中文注释：获取当前的 KV cache 内存布局。结果会被 lru_cache 缓存，
    # 因此在运行期间只会计算一次（除非被 set_kv_cache_layout 显式清除缓存）。
    #
    # 优先级顺序（从高到低）：
    #   1. 代码中的 _KV_CACHE_LAYOUT_OVERRIDE（用于测试或特殊场景）
    #   2. 用户通过环境变量 VLLM_KV_CACHE_LAYOUT 指定
    #   3. 从 KV connector 的配置中获取默认值
    #
    # Format specified by the code.
    global _KV_CACHE_LAYOUT_OVERRIDE

    cache_layout: Literal["NHD", "HND"] | None = None
    if _KV_CACHE_LAYOUT_OVERRIDE is not None:
        cache_layout = _KV_CACHE_LAYOUT_OVERRIDE
        logger.info_once(
            "`_KV_CACHE_LAYOUT_OVERRIDE` variable detected. "
            "Setting KV cache layout to %s.",
            cache_layout,
        )
        return cache_layout

    # Format specified by the user.
    cache_layout = envs.VLLM_KV_CACHE_LAYOUT
    # When neither the user nor the override specified a layout, get default
    if cache_layout is None:
        cache_layout = get_kv_connector_cache_layout()
    else:
        assert is_valid_kv_cache_layout(cache_layout)
        logger.info_once(
            "`VLLM_KV_CACHE_LAYOUT` environment variable "
            "detected. Setting KV cache layout to %s.",
            cache_layout,
        )
    return cache_layout


def set_kv_cache_layout(cache_layout: KVCacheLayoutType | None):
    # 中文注释：设置全局 KV cache 布局覆盖值，并清除 get_kv_cache_layout 的缓存。
    # 调用后，下次调用 get_kv_cache_layout() 将返回新的布局值。
    global _KV_CACHE_LAYOUT_OVERRIDE
    _KV_CACHE_LAYOUT_OVERRIDE = cache_layout
    get_kv_cache_layout.cache_clear()


@dataclass
class PerLayerParameters:
    """
    Currently, FlashInfer backend only support models in which all layers share
    the same values for the following hyperparameters. Should not be used for
    trtllm-gen backend since it supports different values for the following
    hyperparameters.
    """

    # 中文注释：每层注意力的超参数数据类。
    # FlashInfer 后端要求所有注意力层共享相同的参数值（如窗口大小、缩放因子等），
    # trtllm-gen 后端则支持不同层使用不同参数。
    #
    # 关键属性说明：
    # - window_left: 滑动窗口注意力的左侧窗口大小。-1 表示无滑动窗口（全注意力）。
    #   例如 Llama-3 中 window_left=4095 表示每个 token 只关注前 4096 个 token。
    # - logits_soft_cap: logits 软截断阈值，用于 Gemma 等模型。
    #   当非 None 时，logits 会被 tanh 软截断到 [-soft_cap, soft_cap] 范围。
    # - sm_scale: softmax 缩放因子，通常为 1/sqrt(head_dim)。
    # - has_sinks: 是否使用了 attention sink（如 StreamingLLM 中保留初始 token 的策略）。
    # - has_same_window_lefts: 所有层的 window_left 是否相同（用于优化 plan 路径）。
    # - has_same_all_params: 所有层的所有参数是否完全相同（用于优化 plan 路径）。

    window_left: int
    logits_soft_cap: float | None
    sm_scale: float
    has_sinks: bool = False
    # has same params for all layers
    has_same_window_lefts: bool | None = field(default=None, compare=False)
    has_same_all_params: bool | None = field(default=None, compare=False)


def get_per_layer_parameters(
    vllm_config: VllmConfig, layer_names: list[str], cls_: type["AttentionImpl"]
) -> dict[str, PerLayerParameters]:
    """
    Scan layers in `layer_names` and determine some hyperparameters
    to use during `plan`.
    """

    # 中文注释：扫描指定的注意力层，提取每层的关键超参数。
    # 这些参数在 attention backend 的 plan 阶段使用，用于配置注意力 kernel。
    #
    # 算法流程：
    # 1. 从 vllm_config 中获取所有指定名称的注意力层实例
    # 2. 对每个层提取其滑动窗口大小、logits soft cap、缩放因子、sink 等参数
    # 3. 返回 {层名: PerLayerParameters} 的字典映射
    #
    # 参数：
    # - vllm_config: vLLM 全局配置对象
    # - layer_names: 需要扫描的注意力层名称列表
    # - cls_: 注意力实现类的类型，用于类型检查
    #
    # 返回：dict[str, PerLayerParameters]，key 为层名，value 为该层的超参数

    layers = get_layers_from_vllm_config(
        vllm_config,
        AttentionLayerBase,  # type: ignore[type-abstract]
        layer_names,
    )
    per_layer_params: dict[str, PerLayerParameters] = {}

    for key, layer in layers.items():
        impl = layer.impl
        assert isinstance(impl, cls_)

        # Infer hyperparameters from the attention layer
        window_size = getattr(impl, "sliding_window", None)
        window_left = window_size[0] if window_size is not None else -1
        logits_soft_cap = getattr(impl, "logits_soft_cap", None)
        sm_scale = impl.scale
        has_sinks = getattr(impl, "sinks", None) is not None

        per_layer_params[key] = PerLayerParameters(
            window_left, logits_soft_cap, sm_scale, has_sinks
        )

    return per_layer_params


def get_num_attention_heads_from_layers(
    vllm_config: VllmConfig, layer_names: list[str]
) -> int | None:
    """Per-TP-rank ``num_heads`` shared by the named Attention layers.

    Use in metadata builders whose plan-time allocations depend on the
    head count: the model-wide ``get_num_attention_heads()`` is wrong
    for models with non-uniform per-layer head counts. All layers in
    one attention group must agree on ``num_heads``; this is asserted.
    Returns ``None`` when no matching Attention layer is found.
    """
    # 中文注释：获取指定注意力层组中每层共享的注意力头数（per-TP-rank）。
    #
    # 为什么需要这个函数：
    # - 模型级别的 get_num_attention_heads() 对于非均匀注意力头数的模型（如 DeepSeek V3）
    #   是不准确的。不同注意力组（如 MLA 和 GQA）的头数可能不同。
    # - 在 metadata builder 的 plan 阶段，需要根据头数分配 buffer，
    #   因此必须使用正确组的头数。
    #
    # 算法流程：
    # 1. 从 vllm_config 获取所有指定名称的注意力层
    # 2. 提取每层的 num_heads
    # 3. 断言同一组内所有层的头数必须一致
    # 4. 返回头数，若无匹配层则返回 None
    attn_layers = get_layers_from_vllm_config(
        vllm_config,
        AttentionLayerBase,  # type: ignore[type-abstract]
        layer_names,
    )
    if not attn_layers:
        return None
    heads = {layer.impl.num_heads for layer in attn_layers.values()}
    assert len(heads) == 1, (
        f"All layers in one attention group must share num_heads; "
        f"got {heads} for {layer_names}."
    )
    return heads.pop()


def infer_global_hyperparameters(
    per_layer_params: dict[str, PerLayerParameters],
) -> PerLayerParameters:
    """
    Currently, FlashInfer backend other than trtllm-gen
    only support models in which all layers share
    the same values for the following hyperparameters:
    - `window_left`
    - `logits_soft_cap`
    - `sm_scale`

    So this function asserts that all layers share the same values for these
    hyperparameters and returns the global values.
    """

    # 中文注释：从每层参数中推断全局超参数。
    # FlashInfer 后端（非 trtllm-gen）要求所有注意力层共享相同的超参数值。
    # 此函数验证这一约束并返回全局参数。
    #
    # 算法流程：
    # 1. 断言至少存在一个注意力层
    # 2. 以第一层的参数作为参考基准
    # 3. 检查所有层的 window_left 是否相同 -> 设置 has_same_window_lefts
    # 4. 检查所有层的所有参数是否完全相同 -> 设置 has_same_all_params
    # 5. 返回全局参数（以第一层为基准）
    #
    # 这些标志位用于 attention backend 的 plan 阶段做优化：
    # - 如果所有参数相同，plan 只需执行一次，结果可以复用到所有层
    # - 如果只有 window_left 相同，可以部分复用 plan 结果

    assert len(per_layer_params) > 0, "No attention layers found in the model."

    param_sets = list(per_layer_params.values())
    global_params = param_sets[0]

    global_params.has_same_window_lefts = all(
        params.window_left == global_params.window_left for params in param_sets
    )
    global_params.has_same_all_params = all(
        params == global_params for params in param_sets
    )

    return global_params


# 中文注释：本地注意力（Local/Sliding Window Attention）虚拟批次构建算法。
# ===========================================================================
#
# 核心思想：
# 标准 FlashAttention 不支持自定义的注意力掩码（如滑动窗口掩码）。
# 但可以通过将序列拆分为多个"虚拟批次项"来模拟局部注意力。
# 每个虚拟批次项对应一个注意力窗口（chunk），FlashAttention 会自然地
# 只在每个虚拟批次项内部做全注意力，从而等效实现了滑动窗口掩码。
#
# 工作原理：
# 输入 `query_start_loc_np` 和 `seq_lens_np`，将序列按 attn_chunk_size 拆分为
# 多个局部注意力 block，每个 block 作为独立的"虚拟"批次项传给注意力 kernel。
#
# 以 3 个序列的 chunked prefill 为例：
#   q_seqlens  = [4, 10, 5]
#   kv_seqlens = [6, 17, 9]
#
# 普通注意力对 batch 0 (q_seqlens=4, kv_seqlens=6) 的注意力掩码为：
#   batch idx: 0 (q_seqlens = 4, kv_seqlens = 6)
#        k_toks >   0 1 2 3 4 5
#        q_toks v  _____________
#               0 | 1 1 1
#               1 | 1 1 1 1
#               2 | 1 1 1 1 1
#               3 | 1 1 1 1 1 1
#
# 局部注意力 (attn_chunk_size = 4) 的掩码为：
#   batch idx: 0  (q_seqlens = 4, kv_seqlens = 6, attn_chunk_size = 4)
#        k_toks >   0 1 2 3 4 5
#        q_toks v  _____________
#               0 | 1 1 1
#               1 | 1 1 1 1
#               2 |         1
#               3 |         1 1
#
# 拆分为虚拟批次后，batch 0 被分成两个虚拟批次项：
#   local-batch idx: 0 (q_seqlens = 2, kv_seqlens = 4)  (batch 0)
#        k_toks >   0 1 2 3
#        q_toks v  _____________
#               0 | 1 1 1
#               1 | 1 1 1 1
#   local-batch idx: 1 (q_seqlens = 2, kv_seqlens = 2) (batch 0)
#        k_toks >   4 5
#        q_toks v  _____________
#               2 | 1
#               3 | 1 1
#
# 示例输出：
#   attn_chunk_size = 4
#   query_start_loc_np = [0, 4, 14, 19] (q_seqlens = [4, 10, 5])
#   返回：
#                             __b0__  ______b1______  __b2__ < 原始 batch 索引
#   q_seqlens_local    = [   2,  2,  1,  4,  4,  1,  4,  1]
#   cu_seqlens_q_local = [0, 4,  6, 10, 14, 18, 19, 23, 24]
#   seqlens_k_local    = [   4,  2,  4,  4,  4,  1,  4,  1]
#   block_table_local  : shape[local_virtual_batches, pages_per_local_batch]
def make_local_attention_virtual_batches(
    attn_chunk_size: int,
    common_attn_metadata: CommonAttentionMetadata,
    block_size: int = 0,
) -> tuple[CommonAttentionMetadata, Callable[[torch.Tensor], torch.Tensor]]:
    # 中文注释：将混合 batch 中的序列按 attn_chunk_size 拆分为虚拟批次项，
    # 用于模拟滑动窗口/局部注意力。
    #
    # 算法流程（以 3 个序列为例，attn_chunk_size=4）：
    #
    # 步骤 1：计算每个序列需要拆分为多少个虚拟批次项（local_blocks）。
    #   - 处理序列起始位置不在 chunk 边界的情况（chunked prefill 中常见）
    #   - 第一个 block 可能只有部分 token（q_tokens_in_first_block）
    #   - 最后一个 block 可能不满一个 chunk
    #
    # 步骤 2：计算每个虚拟批次项的 query 序列长度（seqlens_q_local）。
    #   - 使用 batched arange 技巧将按序列的计数展开为按虚拟批次项的计数
    #   - 第一个和最后一个虚拟批次项可能是部分 block
    #
    # 步骤 3：计算每个虚拟批次项的 KV 序列长度（seqlens_k_local）。
    #   - 除了每个序列的最后一个 block，其余都是完整的 attn_chunk_size
    #
    # 步骤 4：为虚拟批次项构建新的 block_table。
    #   - 根据每个虚拟批次项对应的 KV 范围，从原始 block_table 中提取对应行
    #
    # 返回值：
    # - CommonAttentionMetadata：虚拟批次项的元数据（可直接传给注意力 kernel）
    # - Callable：用于在 block_table 更新时重建局部 block_table 的函数

    query_start_loc_np = common_attn_metadata.query_start_loc_cpu.numpy()
    seq_lens_np = common_attn_metadata.seq_lens_cpu.numpy()
    block_table = common_attn_metadata.block_table_tensor
    device = common_attn_metadata.query_start_loc.device

    q_seqlens = query_start_loc_np[1:] - query_start_loc_np[:-1]
    actual_batch_size = seq_lens_np.shape[0]

    # Handle if we are starting in the middle of a local attention block,
    #  we assume q_seqlens > 0 (for all elements), for each batch idx we compute
    #  the number of tokens that are not in the first local attention block and
    #  then we can simply use a cdiv for the rest.
    # For example if we have:
    #   attn_chunk_size = 4
    #   q_seqlens = [4, 10, 5]
    #   k_seqlens = [6, 17, 9]
    # Then we would get:
    #   new_tokens_in_first_block = [2, 1, 4]
    #   local_blocks = [2, 4, 2]

    # 中文注释：计算每个序列第一个虚拟批次项中的 query token 数。
    # 这处理了 chunked prefill 中序列起始位置不在 chunk 边界的情况。
    # 例如 attn_chunk_size=4, seq_len-q_seqlens=2 时，第一个 chunk 只剩 2 个位置。
    q_tokens_in_first_block = np.minimum(
        attn_chunk_size - ((seq_lens_np - q_seqlens) % attn_chunk_size), q_seqlens
    ).astype(np.int32)
    # 中文注释：最后一个虚拟批次项的 KV 长度（可能不满一个 chunk）。
    tokens_in_last_block = attn_chunk_size + (seq_lens_np % -attn_chunk_size)
    # 中文注释：每个序列需要拆分为多少个虚拟批次项。
    local_blocks = 1 + cdiv(q_seqlens - q_tokens_in_first_block, attn_chunk_size)

    # Once we know the number of local blocks we can compute the request spans
    #  for each batch idx, we can figure out the number of "virtual" requests we
    #  have to make,
    # For the above example we would get:
    #   seqlens_q_local = [2, 2, 1, 4, 4, 1, 4, 1]
    #
    # First Get batched arange. (E.g., [2, 4, 2] -> [0, 1, 0, 1, 2, 3, 0, 1])
    #   (TODO: make a utility to share this code with _prepare_inputs)
    # arange step 1. [2, 4, 2] -> [2, 6, 8]
    cu_num_blocks = np.cumsum(local_blocks)
    # 中文注释：虚拟批次项的总数，即所有序列拆分后的虚拟请求数。
    virtual_batches = cu_num_blocks[-1]
    # arange step 2. [2, 6, 8] -> [0, 0, 2, 2, 2, 2, 6, 6]
    block_offsets = np.repeat(cu_num_blocks - local_blocks, local_blocks)
    # arange step 3. [0, 1, 0, 1, 2, 3, 0, 1]
    arange = np.arange(virtual_batches, dtype=np.int32) - block_offsets
    # also compute reverse arange (i.e. [1, 0, 3, 2, 1, 0, 1, 0])
    rarange = np.repeat(local_blocks, local_blocks) - arange - 1
    # Then we can compute the seqlens_q_local, handling the fact that the
    #  first and last blocks could be partial

    # 中文注释：计算每个虚拟批次项的 query 序列长度。
    seqlens_q_local = np.repeat(q_seqlens - q_tokens_in_first_block, local_blocks)
    # set the first block since this may be a partial block
    seqlens_q_local[arange == 0] = q_tokens_in_first_block
    # set the remaining blocks
    seqlens_q_local[arange > 0] = np.minimum(
        seqlens_q_local - attn_chunk_size * (arange - 1), attn_chunk_size
    )[arange > 0]

    # convert from q_seqlens to cu_seqlens_q
    cu_seqlens_q_local = np.empty(virtual_batches + 1, dtype=np.int32)
    np.cumsum(seqlens_q_local, out=cu_seqlens_q_local[1:])
    cu_seqlens_q_local[0] = 0

    # compute the seqlens_k_local,
    #  basically a full local attention block for all but the last block in each
    #  batch
    # For our example this will be:
    #   seqlens_k_local = [4, 2, 4, 4, 4, 1, 4, 1]

    # 中文注释：计算每个虚拟批次项的 KV 序列长度。
    # 除了每个序列的最后一个 block 外，其余都是完整的 attn_chunk_size。
    seqlens_k_local = np.full(cu_num_blocks[-1], attn_chunk_size, dtype=np.int32)
    seqlens_k_local[cu_num_blocks - 1] = tokens_in_last_block
    # 中文注释：每个虚拟批次项中已计算的 token 数 = KV 长度 - query 长度。
    # 这些 token 的 KV 已在之前的迭代中计算过，不需要重新计算。
    num_computed_tokens_local = seqlens_k_local - seqlens_q_local

    # 中文注释：计算每个虚拟批次项在原始序列中的 KV 起始位置（绝对位置）。
    k_seqstarts_absolute = np.repeat(seq_lens_np, local_blocks) - (
        rarange * attn_chunk_size + np.repeat(tokens_in_last_block, local_blocks)
    )
    # For the example the local attention blocks start at:
    #                           _b0_  _____b1_____  _b2_
    #   k_seqstarts_absolute = [0, 4, 4, 8, 12, 16, 4, 8]

    # 中文注释：将绝对起始位置转换为 block table 中的 block 起始索引。
    block_starts = k_seqstarts_absolute // block_size
    assert attn_chunk_size % block_size == 0, (
        f"attn_chunk_size {attn_chunk_size} is not divisible by block_size {block_size}"
    )
    # 中文注释：每个虚拟批次项需要多少个 KV block（pages_per_local_batch = chunk_size / block_size）。
    pages_per_local_batch = attn_chunk_size // block_size

    # Create a block_table for the local attention blocks
    # For out example if we have a block-table like (assuming block_size=2):
    #   block_table = [
    #     [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9],  < batch 0
    #     [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],  < batch 1
    #     [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],  < batch 2
    #   ]
    # Then for the local batches we would want a block-table like
    #   block_table_local = [
    #     [  0,  1 ], < local-batch 0, (batch 0, starting from k[0])
    #     [  2,  3 ], < local-batch 1, (batch 0, starting from k[4])
    #     [ 12, 13 ], < local-batch 2, (batch 1, starting from k[4])
    #     [ 14, 15 ], < local-batch 3, (batch 1, starting from k[8])
    #     [ 16, 17 ], < local-batch 4, (batch 1, starting from k[12])
    #     [ 18, 19 ], < local-batch 5, (batch 1, starting from k[16])
    #     [ 22, 23 ], < local-batch 6, (batch 2, starting from k[4])
    #     [ 24, 25 ], < local-batch 7, (batch 2, starting from k[8])
    #   ]

    # 中文注释：为每个虚拟批次项构建对应的 block_table 行。
    # 通过计算每个虚拟批次项在原始 block_table 中的 block 索引范围，
    # 使用 fancy indexing 提取出对应的物理 block ID。
    block_indices = block_starts[:, None] + np.arange(
        pages_per_local_batch, dtype=np.int32
    )
    block_indices = block_indices.reshape(-1).clip(max=block_table.shape[1] - 1)
    batch_indices = np.repeat(
        np.arange(actual_batch_size, dtype=np.int32),
        local_blocks * pages_per_local_batch,
    )

    # NOTE: https://github.com/pytorch/pytorch/pull/160256 causes performance
    # regression when using numpy arrays (batch and block indices) to index into
    # torch tensor (block_table). As a workaround, convert numpy arrays to torch
    # tensor first, which recovers perf.
    # Upload the index tensors to the block_table's device up-front so that the
    # fancy indexing below doesn't implicitly force a synchronous H2D copy.
    batch_indices_torch = torch.from_numpy(batch_indices).to(device, non_blocking=True)
    block_indices_torch = torch.from_numpy(block_indices).to(device, non_blocking=True)

    # Save as a lambda so we can return this for update_block_table
    # 中文注释：保存一个 lambda 函数，用于在 block_table 更新时重建局部 block_table。
    # 这在每次迭代中 block_table 可能变化时很有用。
    make_block_table = lambda block_table: block_table[
        batch_indices_torch, block_indices_torch
    ].view(virtual_batches, -1)
    block_table_local = make_block_table(block_table)

    query_start_loc_cpu = torch.from_numpy(cu_seqlens_q_local)
    seq_lens_cpu = torch.from_numpy(seqlens_k_local)
    max_seq_len = int(seq_lens_cpu.max())

    # 中文注释：构建虚拟批次的 CommonAttentionMetadata 并返回。
    # 这个 metadata 可以直接传给注意力 kernel，kernel 会将每个虚拟批次项
    # 视为独立的请求，从而自然地实现了局部注意力的效果。
    return CommonAttentionMetadata(
        query_start_loc_cpu=query_start_loc_cpu,
        query_start_loc=query_start_loc_cpu.to(device=device, non_blocking=True),
        seq_lens=seq_lens_cpu.to(device=device, non_blocking=True),
        num_reqs=len(seq_lens_cpu),
        num_actual_tokens=common_attn_metadata.num_actual_tokens,
        max_query_len=seqlens_q_local.max(),
        max_seq_len=max_seq_len,
        block_table_tensor=block_table_local,
        slot_mapping=common_attn_metadata.slot_mapping,
        causal=True,
        seq_lens_cpu_upper_bound=common_attn_metadata.seq_lens_cpu_upper_bound,
        _seq_lens_cpu=seq_lens_cpu,
        _num_computed_tokens_cpu=torch.from_numpy(num_computed_tokens_local),
    ), make_block_table


def make_kv_sharing_fast_prefill_common_attn_metadata(
    common_attn_metadata: CommonAttentionMetadata,
) -> CommonAttentionMetadata:
    # 中文注释：为 KV 共享的快速 prefill 路径构建注意力元数据。
    #
    # 背景：在 KV 共享（KV Sharing）场景中，多个层共享同一份 KV cache。
    # 快速 prefill 路径的思路是：在 prefill 阶段，只计算 logits 需要的 token
    # 的注意力（而不是所有 query token），从而减少计算量。
    #
    # 适用场景：
    # - 当一个 batch 中同时包含 prefill 和 decode 请求时
    # - prefill 请求的 logits_indices 指定了哪些 token 需要计算 logits
    # - 对于 decode 请求，只有最后一个 token 需要计算 logits
    #
    # 算法流程：
    # 1. 如果所有请求都是 decode（max_query_len == 1），直接返回原始 metadata
    # 2. 从 logits_indices 中提取需要计算 logits 的 token 位置
    # 3. 用 bucketize 将这些 token 映射回各自的请求
    # 4. 统计每个请求有多少个需要计算 logits 的 token
    # 5. 构建新的 query_start_loc，只包含 logits 相关的 token
    # 6. 返回新的 CommonAttentionMetadata
    #
    # 这样注意力 kernel 只会对需要输出 logits 的 token 子集做计算，
    # 而不是对所有 query token 做全量计算，显著降低了计算量。

    if common_attn_metadata.max_query_len == 1:
        # All requests are decode (assume 1 token for now)
        # Skip computing fast prefill path
        return common_attn_metadata

    assert common_attn_metadata.logits_indices_padded is not None
    assert common_attn_metadata.num_logits_indices is not None

    logits_indices_padded = common_attn_metadata.logits_indices_padded
    num_logits_indices = common_attn_metadata.num_logits_indices
    # Get rid of CUDAGraph padding, if any
    logits_indices = logits_indices_padded[:num_logits_indices]
    num_reqs = common_attn_metadata.num_reqs
    query_start_loc = common_attn_metadata.query_start_loc
    # Example inputs
    # num_reqs: 3
    # generation_indices:  [14, 18, 19, 27]
    # query_start_loc: [0, 15, 20, 28]
    # seq_lens:        [41, 31, 40]

    # Find how many decode indices belong to each request
    # request_ids: [0, 1, 1, 2]
    request_ids = torch.bucketize(logits_indices, query_start_loc[1:], right=True)

    # Figure out how many tokens are in each request
    # num_decode_tokens: [1, 2, 1]
    # Avoid `torch.bincount` here — on CUDA it forces a sync to determine
    # the output size (even with `minlength`, the kernel must confirm no
    # value exceeds the bound). `scatter_add_` into a preallocated buffer
    # is equivalent and stays async.
    num_decode_tokens = torch.zeros(
        num_reqs, dtype=request_ids.dtype, device=request_ids.device
    )
    num_decode_tokens.scatter_add_(
        0, request_ids.to(num_decode_tokens.dtype), torch.ones_like(request_ids)
    )

    # Calculate new query_start_loc with tokens in generation_indices
    # decode_query_start_loc: [0, 1, 3, 4]
    decode_query_start_loc = torch.empty(
        num_reqs + 1, device=query_start_loc.device, dtype=query_start_loc.dtype
    )

    decode_query_start_loc[:1].fill_(0)  # Avoid sync from scalar assignment.
    decode_query_start_loc[1:] = torch.cumsum(num_decode_tokens, dim=0)
    decode_max_query_len = int(num_decode_tokens.max().item())
    total_num_decode_tokens = int(num_decode_tokens.sum().item())

    # 中文注释：构建新的 CommonAttentionMetadata，只包含需要计算 logits 的 token。
    # 注意：
    # - num_actual_tokens 变为 total_num_decode_tokens（只包含 logits 相关 token）
    # - max_query_len 变为 decode_max_query_len（每个请求最多需要计算 logits 的 token 数）
    # - slot_mapping 和 block_table 保持不变（KV cache 的物理映射不受影响）
    common_attn_metadata = CommonAttentionMetadata(
        query_start_loc=decode_query_start_loc,
        query_start_loc_cpu=decode_query_start_loc.to("cpu", non_blocking=True),
        seq_lens=common_attn_metadata.seq_lens,
        num_reqs=num_reqs,
        num_actual_tokens=total_num_decode_tokens,
        max_query_len=decode_max_query_len,
        max_seq_len=common_attn_metadata.max_seq_len,
        block_table_tensor=common_attn_metadata.block_table_tensor,
        slot_mapping=common_attn_metadata.slot_mapping,
        causal=True,
        seq_lens_cpu_upper_bound=common_attn_metadata.seq_lens_cpu_upper_bound,
        _seq_lens_cpu=common_attn_metadata._seq_lens_cpu,
        _num_computed_tokens_cpu=common_attn_metadata._num_computed_tokens_cpu,
    )
    return common_attn_metadata


def split_decodes_prefills_and_extends(
    common_attn_metadata: CommonAttentionMetadata,
    decode_threshold: int = 1,
) -> tuple[int, int, int, int, int, int]:
    """
    Assuming a reordered batch, finds the boundary between prefill and decode
    requests.

    Args:
        common_attn_metadata: CommonAttentionMetadata object containing the
            batch metadata.
        decode_threshold: The maximum query length to be considered a decode.

    Returns:
        num_decodes: The number of decode requests.
        num_extends: The number of extend requests.
        num_prefills: The number of prefill requests.
        num_decode_tokens: The number of tokens in the decode requests.
        num_extend_tokens: The number of tokens in the extend requests.
        num_prefill_tokens: The number of tokens in the prefill requests.
    """
    # 中文注释：在已重排序的 batch 中，将请求分为三类并统计数量和 token 数。
    #
    # 三类请求的定义：
    # 1. decode：query_len <= decode_threshold，且已完成 prefill（seq_len > query_len）
    #    这是标准的自回归解码阶段，每次只生成一个 token。
    #
    # 2. extend：query_len > decode_threshold，但 seq_len > query_len
    #    这是 chunked prefill 的中间阶段，已有一部分 KV cache 被计算，
    #    本轮需要继续计算剩余的 query token。
    #
    # 3. prefill：query_len > decode_threshold，且 seq_len == query_len
    #    这是首轮 prefill（first chunk），没有已缓存的 KV，
    #    本轮需要计算该请求的所有 token。
    #
    # 假设 batch 已按 decode -> extend -> prefill 的顺序重排，
    # 此函数通过线性扫描找到三类请求的边界。
    #
    # 算法流程：
    # 1. 如果所有请求的 max_query_len <= decode_threshold，全部是 decode
    # 2. 计算每个请求的 query_len
    # 3. 识别 is_prefill_or_extend（query_len > threshold）
    # 4. 识别 is_prefill（seq_len == query_len，即首轮 prefill）
    # 5. 通过 argmax 找到第一项 extend 和第一项 prefill 的位置
    # 6. 根据边界位置计算各类请求的数量和 token 数

    max_query_len = common_attn_metadata.max_query_len
    num_reqs = common_attn_metadata.num_reqs
    num_tokens = common_attn_metadata.num_actual_tokens
    query_start_loc = common_attn_metadata.query_start_loc_cpu
    # Upper bound is exact for prefill rows; decode rows still satisfy
    # seq_len > query_len under the optimistic bound, so `seq_lens ==
    # query_lens` identifies prefills correctly either way.
    assert common_attn_metadata.seq_lens_cpu_upper_bound is not None
    seq_lens = common_attn_metadata.seq_lens_cpu_upper_bound

    if max_query_len <= decode_threshold:
        return num_reqs, 0, 0, num_tokens, 0, 0

    query_lens = query_start_loc[1:] - query_start_loc[:-1]
    is_prefill_or_extend = query_lens > decode_threshold
    is_prefill = (seq_lens == query_lens) & is_prefill_or_extend
    first_extend = is_prefill_or_extend.int().argmax(dim=-1).item()
    first_prefill = is_prefill.int().argmax(dim=-1).item()
    num_decodes = first_extend
    num_decode_tokens = query_start_loc[first_extend].item()
    if not torch.any(is_prefill_or_extend):
        return (num_decodes, 0, 0, num_decode_tokens, 0, 0)

    num_prefills_or_extends = num_reqs - num_decodes
    num_prefill_or_extend_tokens = num_tokens - num_decode_tokens
    if not torch.any(is_prefill):
        return (
            num_decodes,
            num_prefills_or_extends,
            0,
            num_decode_tokens,
            num_prefill_or_extend_tokens,
            0,
        )

    num_extends = first_prefill - num_decodes
    num_prefills = num_reqs - first_prefill

    num_prefill_tokens = num_tokens - query_start_loc[first_prefill]
    num_extend_tokens = num_prefill_or_extend_tokens - num_prefill_tokens
    return (
        num_decodes,
        num_extends,
        num_prefills,
        num_decode_tokens,
        num_extend_tokens,
        num_prefill_tokens,
    )


def split_decodes_and_prefills(
    common_attn_metadata: CommonAttentionMetadata,
    decode_threshold: int = 1,
    require_uniform: bool = False,
    treat_short_extends_as_decodes: bool = True,
) -> tuple[int, int, int, int]:
    """
    Assuming a reordered batch, finds the boundary between prefill and decode
    requests.

    The batch is expected to be ordered as:
        decode → short_extend → long_extend → prefill

    Args:
        common_attn_metadata: CommonAttentionMetadata object containing the
            batch metadata.
        decode_threshold: The maximum query length to be considered a decode.
        require_uniform: If True, requires that all decode requests have the
            same query length. When set, some queries may be considered prefills
            even if they are <= decode_threshold, in order to ensure uniformity.
        treat_short_extends_as_decodes: If True (default), short extends
            (query_len <= threshold but still prefilling) are counted as
            decodes. If False, they are counted as prefills.

    Returns:
        num_decodes: The number of decode requests.
        num_prefills: The number of prefill requests.
        num_decode_tokens: The number of tokens in the decode requests.
        num_prefill_tokens: The number of tokens in the prefill requests.
    """
    # 中文注释：在已重排序的 batch 中，将请求分为 decode 和 prefill 两类。
    # 这是 split_decodes_prefills_and_extends 的简化版本，不区分 extend。
    #
    # 与 split_decodes_prefills_and_extends 的区别：
    # - 此函数将 extend 归入 prefill，只返回 decode vs prefill 的二分结果
    # - 支持 require_uniform 模式：要求所有 decode 请求的 query_len 相同
    #   （用于 CUDA Graph 全捕获场景，CG 要求 batch 内 decode 的 shape 一致）
    # - 支持 treat_short_extends_as_decodes 选项：
    #   - True（默认）：短 extend（query_len <= threshold 但仍在 prefilling）算作 decode
    #   - False：短 extend 算作 prefill
    #
    # 算法流程：
    # 1. 快速路径：如果所有请求都是 decode，直接返回
    # 2. 检查第一个请求是否为 decode（batch 已排序，第一个不是 decode 则没有 decode）
    # 3. 在 require_uniform 模式下，检查是否所有 decode 的 query_len 一致
    # 4. 通过 argmax 找到第一个 prefill 请求的位置
    # 5. 根据边界计算 decode 和 prefill 的数量及 token 数
    max_query_len = common_attn_metadata.max_query_len
    num_reqs = common_attn_metadata.num_reqs
    num_tokens = common_attn_metadata.num_actual_tokens
    query_start_loc = common_attn_metadata.query_start_loc_cpu

    if (
        max_query_len <= decode_threshold
        and (not require_uniform or decode_threshold <= 1)
        and treat_short_extends_as_decodes
    ):
        return num_reqs, 0, num_tokens, 0

    query_lens = query_start_loc[1:] - query_start_loc[:-1]
    if query_lens[0].item() > decode_threshold:
        # first request is not decode, so no decode requests
        return 0, num_reqs, 0, num_tokens

    if require_uniform:
        # check if we are in a padded uniform batch; this is used for full-CGs, some
        # requests may have a query length of 0 but since they are padding its fine
        # to treat them as decodes (ensures num_decodes matches the captured size)
        if torch.all((query_lens == query_lens[0]) | (query_lens == 0)):
            return num_reqs, 0, num_tokens, 0  # all decodes
        is_prefill = query_lens != query_lens[0]
    else:
        is_prefill = query_lens > decode_threshold

    if not treat_short_extends_as_decodes:
        assert common_attn_metadata.is_prefilling is not None
        is_prefill |= common_attn_metadata.is_prefilling

    if not torch.any(is_prefill):
        return num_reqs, 0, num_tokens, 0

    first_prefill = is_prefill.int().argmax(dim=-1).item()
    num_decodes = first_prefill
    num_prefills = num_reqs - num_decodes
    num_decode_tokens = query_start_loc[first_prefill].item()
    num_prefill_tokens = num_tokens - num_decode_tokens
    return (num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens)


def split_prefill_chunks(
    seq_lens_cpu: torch.Tensor, workspace_size: int, request_offset: int = 0
) -> list[tuple[int, int]]:
    """
    Split the prefill requests into chunks such that the total sequence length
    of each chunk is less than or equal to the workspace size.

    Args:
        seq_lens_cpu: The sequence lengths of the prefill requests on CPU.
        workspace_size: The maximum workspace size (in tokens) per chunk.
        request_offset: The offset to add to the request indices.
    Returns:
        A list of tuples of (reqs_start, reqs_end) representing chunk boundaries.
    """
    # 中文注释：将 prefill 请求按 workspace 大小拆分为多个 chunk。
    #
    # 背景：在 prefill 阶段，注意力计算需要的 workspace（临时显存）与
    # 序列长度相关。如果同时处理太多长序列，可能超出显存限制。
    # 此函数将 prefill 请求分组，使得每组的总序列长度不超过 workspace_size。
    #
    # 算法流程（贪心装箱）：
    # 1. 从第一个请求开始，逐个累加序列长度
    # 2. 当累加长度超过 workspace_size 时，结束当前 chunk，开始新 chunk
    # 3. 每个 chunk 的边界为 (reqs_start, reqs_end)
    #
    # 参数：
    # - seq_lens_cpu: prefill 请求的序列长度（CPU tensor）
    # - workspace_size: 每个 chunk 的最大 token 数
    # - request_offset: 请求索引的偏移量（用于全局索引对齐）
    #
    # 返回：list[tuple[int, int]]，每个元素为 (起始请求索引, 结束请求索引)

    chunk_bounds = []
    i, n = 0, len(seq_lens_cpu)
    assert torch.all(seq_lens_cpu <= workspace_size).item()

    while i < n:
        start, chunk_total = i, 0
        while i < n and (chunk_total + (s := seq_lens_cpu[i].item())) <= workspace_size:
            chunk_total += s
            i += 1
        chunk_bounds.append((start + request_offset, i + request_offset))
    return chunk_bounds


def reorder_batch_to_split_decodes_and_prefills(
    input_batch: "InputBatch",
    scheduler_output: "SchedulerOutput",
    decode_threshold: int = 1,
) -> bool:
    """
    Reorders the batch to split into prefill and decode requests; places all
    requests with <= decode_threshold tokens at the front of the batch.

    The batch is reordered into 4 regions:
        decode:        (num_scheduled <= threshold AND is not prefilling)
        short_extend:  (num_scheduled <= threshold AND is chunked prefilling)
        long_extend:   (num_scheduled > threshold AND is chunked prefilling)
        prefill:       (num_computed == 0)   # First chunks

    Returns:
        True if the batch was modified, False otherwise.
    """
    # 中文注释：将混合 batch 重排序，使 decode 请求排在前面，prefill 排在后面。
    # 这是 vLLM V1 推理引擎中的关键优化，允许注意力后端对 decode 和 prefill
    # 使用不同的计算路径（例如 decode 使用 CUDA Graph，prefill 使用普通 kernel）。
    #
    # 重排序后 batch 被分为 4 个区域：
    # 1. decode：已完成 prefill，本轮只需解码 1 个 token（num_scheduled <= threshold）
    # 2. short_extend：chunked prefill 的中间阶段，但本轮只需处理少量 token（<= threshold）
    # 3. long_extend：chunked prefill 的中间阶段，本轮需要处理较多 token（> threshold）
    # 4. prefill：首轮 prefill（num_computed == 0），没有已缓存的 KV
    #
    # 为什么要这样排序：
    # - decode 和 short_extend 的 query_len 小，可以使用相同的 CUDA Graph
    # - long_extend 和 prefill 的 query_len 大，需要使用不同的计算路径
    # - 同类型请求连续排列可以减少 kernel launch 的开销
    #
    # 算法流程：
    # 1. 对每个请求分类为 4 种类型之一（互斥）
    # 2. 计算目标排列顺序
    # 3. 如果当前排列已经是目标排列，直接返回 False（无需修改）
    # 4. 否则通过 swap_states 交换请求状态，实现重排序
    # 5. 使用循环交换算法，每个请求最多被交换一次

    num_reqs = len(input_batch.req_ids)
    num_scheduled_tokens = [
        scheduler_output.num_scheduled_tokens[id] for id in input_batch.req_ids
    ]
    num_scheduled_tokens_np = np.array(num_scheduled_tokens)
    num_computed_tokens_np = input_batch.num_computed_tokens_cpu[:num_reqs]
    num_prompt_tokens_np = input_batch.num_prompt_tokens[:num_reqs]

    has_context = num_computed_tokens_np > 0
    is_below_threshold = num_scheduled_tokens_np <= decode_threshold
    done_prefilling = num_computed_tokens_np >= num_prompt_tokens_np

    # Mutually exclusive categories (exactly one True per request):
    # 1. No context yet -> prefill
    # 2. Has context, above threshold -> long_extend
    # 3. Has context, below threshold, still prefilling -> short_extend
    # 4. Has context, below threshold, done prefilling -> decode
    is_pure_prefill = ~has_context
    is_long_extend = has_context & ~is_below_threshold
    is_short_extend = has_context & is_below_threshold & ~done_prefilling
    is_decode = has_context & is_below_threshold & done_prefilling

    # Desired order: decode → short_extend → long_extend → prefill
    req_regions = np.zeros(num_reqs, dtype=np.int32)  # 0 = decode by default
    req_regions[is_short_extend] = 1
    req_regions[is_long_extend] = 2
    req_regions[is_pure_prefill] = 3

    num_decodes = int(is_decode.sum())
    num_short_extends = int(is_short_extend.sum())
    num_long_extends = int(is_long_extend.sum())
    num_prefills = int(is_pure_prefill.sum())

    target_regions = np.repeat(
        [0, 1, 2, 3],
        [num_decodes, num_short_extends, num_long_extends, num_prefills],
    ).astype(np.int32)

    needs_swap = req_regions != target_regions

    if not needs_swap.any():
        return False

    # Extract indices that need swapping and sort by target region
    orig_indices = np.where(needs_swap)[0]
    sorted_order = np.argsort(req_regions[needs_swap], kind="stable")
    src_indices = orig_indices[sorted_order]

    src_dest_map = {int(src): int(dst) for src, dst in zip(src_indices, orig_indices)}

    for src in src_dest_map:
        dst = src_dest_map[src]
        while src != dst:
            input_batch.swap_states(src, dst)
            # Mark dst as done by updating its destination to itself
            next_dst = src_dest_map.get(dst, dst)
            src_dest_map[dst] = dst
            dst = next_dst

    return True


def reshape_query_for_spec_decode(query: torch.Tensor, batch_size: int) -> torch.Tensor:
    """
    Reshapes the query tensor for the specified batch size, so that
    it has shape (batch_size, seq_len, num_heads, head_dim).
    """
    # 中文注释：为投机解码（Speculative Decoding）场景 reshape query tensor。
    #
    # 投机解码中，每个请求一次生成多个候选 token（由 draft model 提供），
    # 因此 query 的 shape 为 (total_tokens, num_heads, head_dim)，其中
    # total_tokens = batch_size * seq_len（seq_len 为每个请求的候选 token 数）。
    #
    # 一些注意力后端（如 FlashInfer）需要 4D 输入 (batch_size, seq_len, num_heads, head_dim)，
    # 此函数将 3D 的 packed query reshape 为 4D。
    #
    # 参数：
    # - query: shape (total_tokens, num_heads, head_dim) 的 3D tensor
    # - batch_size: 请求批次大小
    #
    # 返回：shape (batch_size, seq_len, num_heads, head_dim) 的 4D tensor
    assert query.dim() == 3, f"query must be 3D, got {query.dim()}D"
    total_tokens = query.shape[0]
    num_heads = query.shape[1]
    head_dim = query.shape[2]
    assert total_tokens % batch_size == 0, (
        f"{total_tokens=} is not divisible by {batch_size=}"
    )
    seq_len = total_tokens // batch_size
    return query.view(batch_size, seq_len, num_heads, head_dim)


def reshape_attn_output_for_spec_decode(attn_output: torch.Tensor) -> torch.Tensor:
    """
    Reshapes the attention output tensor, so that
    the batch_size and seq_len dimensions are combined.
    """
    # 中文注释：为投机解码场景 reshape 注意力输出 tensor，将 batch_size 和 seq_len
    # 维度合并回 packed 格式。这是 reshape_query_for_spec_decode 的逆操作。
    #
    # 输入：(batch_size, seq_len, num_heads, head_dim) 或已 packed 的 3D tensor
    # 输出：(total_tokens, num_heads, head_dim)，其中 total_tokens = batch_size * seq_len
    if attn_output.dim() == 3:
        # Already in the correct shape
        return attn_output
    assert attn_output.dim() == 4, f"attn_output must be 4D, got {attn_output.dim()}D"
    total_tokens = attn_output.shape[0] * attn_output.shape[1]
    return attn_output.view(total_tokens, attn_output.shape[2], attn_output.shape[3])


def subclass_attention_metadata(
    name_prefix: str,
    metadata_cls: Any,
    fields: list[tuple[str, Any, Any]],
) -> Any:
    """
    Return a new subclass of `metadata_cls` with additional fields
    """
    # 中文注释：动态创建一个继承自 metadata_cls 的新 dataclass 子类，
    # 并添加额外的字段。用于在运行时扩展注意力元数据类，
    # 例如为 KV 共享快速 prefill 路径添加 logits_indices 等字段。
    #
    # 参数：
    # - name_prefix: 新类名的前缀
    # - metadata_cls: 要继承的基类
    # - fields: 要添加的额外字段列表，每个元素为 (字段名, 类型, 默认值)
    #
    # 返回：新的 dataclass 子类
    name: str = name_prefix + metadata_cls.__name__  # type: ignore
    Wrapped = make_dataclass(name, fields, bases=(metadata_cls,))
    return Wrapped


@runtime_checkable
class KVSharingFastPrefillMetadata(Protocol):
    # 中文注释：KV 共享快速 prefill 路径的元数据协议（Protocol）。
    # 定义了快速 prefill 路径需要的额外字段：
    # - logits_indices_padded: 需要计算 logits 的 token 索引（带 padding）
    # - num_logits_indices: 实际需要计算 logits 的 token 数量
    # 这些字段用于在注意力计算后只提取需要输出 logits 的 token 子集。
    logits_indices_padded: torch.Tensor | None = None
    num_logits_indices: int | None = None


def create_fast_prefill_custom_backend(
    prefix: str,
    underlying_attn_backend: type[AttentionBackend],
) -> type[AttentionBackend]:
    # 中文注释：为 KV 共享场景创建一个自定义的注意力后端。
    # 该后端在 build 阶段会先调用 make_kv_sharing_fast_prefill_common_attn_metadata
    # 将 CommonAttentionMetadata 裁剪为只包含 logits 相关的 token，
    # 然后调用底层后端的 build 方法构建注意力元数据。
    #
    # 这样在 KV 共享场景中，多层共享同一份 KV cache 时，
    # 只有需要输出 logits 的 token 子集会被送入注意力 kernel 计算，
    # 从而显著减少计算量。
    #
    # 参数：
    # - prefix: 自定义后端的名称前缀
    # - underlying_attn_backend: 底层注意力后端类（如 FlashAttention、FlashInfer）
    #
    # 返回：一个带有快速 prefill 优化的自定义注意力后端类

    underlying_builder = underlying_attn_backend.get_builder_cls()

    class FastPrefillAttentionBuilder(underlying_builder):  # type: ignore
        def build(
            self,
            common_prefix_len: int,
            common_attn_metadata: CommonAttentionMetadata,
            fast_build: bool = False,
        ) -> AttentionMetadata:
            # 中文注释：先将 common_attn_metadata 裁剪为只包含 logits 相关的 token，
            # 然后调用底层后端的 build 方法。
            new_common_attn_metadata = (
                make_kv_sharing_fast_prefill_common_attn_metadata(common_attn_metadata)
            )
            metadata = super().build(
                common_prefix_len, new_common_attn_metadata, fast_build
            )

            class KVSharingFastPrefillAttentionMetadata(
                metadata.__class__,  #  type: ignore
                KVSharingFastPrefillMetadata,
            ):
                def __init__(self, metadata, common_attn_metadata):
                    # Shallow copy all fields in metadata cls
                    for _field in fields(metadata.__class__):
                        setattr(self, _field.name, getattr(metadata, _field.name))

                    self.logits_indices_padded = (
                        common_attn_metadata.logits_indices_padded
                    )
                    self.num_logits_indices = common_attn_metadata.num_logits_indices

            return KVSharingFastPrefillAttentionMetadata(metadata, common_attn_metadata)

    attn_backend = subclass_attention_backend(
        name_prefix=prefix,
        attention_backend_cls=underlying_attn_backend,
        builder_cls=FastPrefillAttentionBuilder,
    )

    return attn_backend


def compute_causal_conv1d_metadata(
    query_start_loc_p_cpu: torch.Tensor,
    *,
    device: torch.device,
):
    # 中文注释：为 causal_conv1d kernel 计算元数据。
    # causal_conv1d 是 Mamba 等状态空间模型使用的因果 1D 卷积 kernel。
    #
    # 此函数在 CPU 上计算（使用 CPU tensor 避免 D2H 同步），
    # 然后将结果拷贝到目标设备上。
    #
    # 算法流程：
    # 1. 从 query_start_loc 计算每个请求的序列长度（seqlens）
    # 2. 对每个 BLOCK_M 值（目前只有 8），计算：
    #    - nums: 每个请求需要多少个 BLOCK_M 大小的 tile（向上取整）
    #    - mlist: 每个 tile 对应的请求索引（展开形式）
    #    - offsetlist: 每个 tile 在请求内的偏移量
    #    - batch_ptr: GPU 上的请求索引数组（用于 causal_conv1d kernel）
    #    - token_chunk_offset_ptr: GPU 上的偏移量数组
    # 3. 使用 PAD_SLOT_ID (-1) 填充未使用的位置
    #
    # 返回值：
    # - nums_dict: 包含每个 BLOCK_M 的元数据字典
    # - batch_ptr: GPU tensor，请求索引数组
    # - token_chunk_offset_ptr: GPU tensor，token chunk 偏移量数组

    # Needed for causal_conv1d. Use the CPU query_start_loc to avoid DtoH sync.
    assert query_start_loc_p_cpu.device.type == "cpu"
    seqlens = query_start_loc_p_cpu.diff()
    nums_dict = {}  # type: ignore
    batch_ptr = None
    token_chunk_offset_ptr = None
    for BLOCK_M in [8]:  # cover all BLOCK_M values
        # 中文注释：向上取整计算每个请求需要的 tile 数量。
        nums = -(-seqlens // BLOCK_M)
        nums_dict[BLOCK_M] = {}
        nums_dict[BLOCK_M]["nums"] = nums
        nums_dict[BLOCK_M]["tot"] = nums.sum().item()
        # 中文注释：展开形式的请求索引。例如 nums=[2,3] -> mlist=[0,0,1,1,1]
        mlist = torch.from_numpy(np.repeat(np.arange(len(nums)), nums))
        nums_dict[BLOCK_M]["mlist"] = mlist
        mlist_len = len(nums_dict[BLOCK_M]["mlist"])
        nums_dict[BLOCK_M]["mlist_len"] = mlist_len
        MAX_NUM_PROGRAMS = max(1024, mlist_len) * 2
        # 中文注释：每个 tile 在请求内的偏移量。例如 nums=[2,3] -> offsetlist=[0,1,0,1,2]
        offsetlist = []  # type: ignore
        for idx, num in enumerate(nums):
            offsetlist.extend(range(num))
        offsetlist = torch.tensor(offsetlist, dtype=torch.int32)
        nums_dict[BLOCK_M]["offsetlist"] = offsetlist

        if batch_ptr is None:
            # Update default value after class definition
            batch_ptr = torch.full(
                (MAX_NUM_PROGRAMS,), PAD_SLOT_ID, dtype=torch.int32, device=device
            )
            token_chunk_offset_ptr = torch.full(
                (MAX_NUM_PROGRAMS,), PAD_SLOT_ID, dtype=torch.int32, device=device
            )
        else:
            if batch_ptr.nelement() < MAX_NUM_PROGRAMS:
                batch_ptr.resize_(MAX_NUM_PROGRAMS).fill_(PAD_SLOT_ID)
                token_chunk_offset_ptr.resize_(  # type: ignore
                    MAX_NUM_PROGRAMS
                ).fill_(PAD_SLOT_ID)

        batch_ptr[0:mlist_len].copy_(mlist, non_blocking=True)
        token_chunk_offset_ptr[  # type: ignore
            0:mlist_len
        ].copy_(offsetlist, non_blocking=True)
        nums_dict[BLOCK_M]["batch_ptr"] = batch_ptr
        nums_dict[BLOCK_M]["token_chunk_offset_ptr"] = token_chunk_offset_ptr  # type: ignore

    return nums_dict, batch_ptr, token_chunk_offset_ptr


def get_dcp_local_seq_lens(
    seq_lens: torch.Tensor,
    dcp_size: int = 1,
    dcp_rank: int | None = None,
    cp_kv_cache_interleave_size: int = 1,
) -> torch.Tensor:
    """While using dcp, kv_cache size stored on each rank may be different,
    use this function to calculate split decode seq_lens of each dcp rank.
    Only consider dcp now, we can extend the case of cp based on this.
    """
    # 中文注释：在分布式上下文并行（DCP, Distributed Context Parallel）场景下，
    # 计算每个 DCP rank 本地存储的 KV cache 序列长度。
    #
    # 背景：DCP 将长序列的 KV cache 分散到多个 rank 上存储。
    # 由于 interleave 模式，不同 rank 存储的 token 数量可能不同。
    # 此函数计算每个 rank 实际存储的序列长度。
    #
    # 算法流程：
    # 1. 将序列长度按 interleave_size 和 dcp_size 分为 base 和 remainder
    # 2. base = floor(seq_len / interleave_size / dcp_size) * interleave_size
    #    这是每个 rank 至少存储的 token 数
    # 3. remainder = seq_len - base * dcp_size
    #    剩余的 token 按 interleave 模式分配给各 rank
    # 4. 每个 rank 的实际长度 = base + min(max(remainder - rank_offset * interleave_size, 0), interleave_size)
    #
    # 参数：
    # - seq_lens: 每个请求的原始序列长度
    # - dcp_size: DCP 的并行度（rank 数）
    # - dcp_rank: 当前 rank 的索引。None 时返回所有 rank 的结果
    # - cp_kv_cache_interleave_size: KV cache 的 interleave 大小
    #
    # 返回：每个请求在当前 rank（或所有 rank）上的本地序列长度

    num_requests = seq_lens.size(0)
    if dcp_rank is None:
        rank_offsets = (
            torch.arange(dcp_size, dtype=torch.int32, device=seq_lens.device)
            .unsqueeze(0)
            .repeat(num_requests, 1)
        )
    else:
        rank_offsets = torch.tensor(
            [[dcp_rank]], dtype=torch.int32, device=seq_lens.device
        )
    seq_lens_tiled = (
        seq_lens.to(torch.int32).unsqueeze(-1).repeat(1, rank_offsets.shape[1])
    )
    base = (
        seq_lens_tiled
        // cp_kv_cache_interleave_size
        // dcp_size
        * cp_kv_cache_interleave_size
    )
    remainder = seq_lens_tiled - base * dcp_size
    remainder = torch.clip(
        remainder - rank_offsets * cp_kv_cache_interleave_size,
        0,
        cp_kv_cache_interleave_size,
    )
    dcp_local_seq_lens = base + remainder
    return dcp_local_seq_lens.squeeze(1)


def mamba_get_block_table_tensor(
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    kv_cache_spec: KVCacheSpec,
    mamba_cache_mode: str,
) -> torch.Tensor:
    """
    Get the block table tensor for mamba kernels from the input
    common_attn_metadata.block_table_tensor given different mamba cache modes.

    - "all":   input  (#requests, cdiv(max_model_len, block_size)
                        + num_speculative_blocks);
               output (#requests, cdiv(max_model_len, block_size)
                        + num_speculative_blocks).

    - "none":  input  (#requests, 1 + num_speculative_blocks);
               output (#requests, 1 + num_speculative_blocks).

    - "align": input  (#requests, cdiv(max_model_len, block_size));
               output (#requests, 1 + num_speculative_blocks), which are the last
               1 + num_speculative_blocks of each request.
    """
    # 中文注释：为 Mamba kernel 获取 block table tensor。
    # Mamba 是一种状态空间模型（SSM），其"KV cache"实际上是状态矩阵（state），
    # 与 Transformer 的 KV cache 有不同的缓存管理需求。
    #
    # 支持三种缓存模式：
    # 1. "all"：保留所有历史状态（全量缓存），block_table 保持不变
    # 2. "none"：不保留历史状态（无缓存），block_table 保持不变
    # 3. "align"：只保留最近的状态块（对齐模式），需要从 block_table 中
    #    提取每个请求最后 1 + num_speculative_blocks 个 block
    #
    # 算法流程（"align" 模式）：
    # 1. 计算每个请求的起始 block 索引：start_index = (seq_len - 1) // block_size
    # 2. 计算需要提取的 block 偏移量：0, 1, ..., num_speculative_blocks
    # 3. 使用 torch.gather 从 block_table 中提取对应的 block ID
    #
    # 参数：
    # - block_table: 原始 block table tensor
    # - seq_lens: 每个请求的序列长度
    # - kv_cache_spec: KV cache 规格（包含 block_size、num_speculative_blocks 等）
    # - mamba_cache_mode: 缓存模式，"all"/"none"/"align"
    #
    # 返回：适配后的 block table tensor

    if mamba_cache_mode in ("all", "none"):
        return block_table
    else:
        assert isinstance(kv_cache_spec, MambaSpec)
        # NOTE: For 0-length requests in CUDA graph, use a start_index of 0
        # to handle the invalid block table.
        start_indices = torch.clamp(
            (seq_lens - 1) // kv_cache_spec.block_size,
            min=0,
        )
        # Use int32 for arithmetic to avoid dtype promotion overhead,
        # then convert to int64 for gather (which requires Long indices)
        offsets = torch.arange(
            1 + kv_cache_spec.num_speculative_blocks,
            device=block_table.device,
            dtype=torch.int32,
        )
        indices_to_gather = (start_indices.unsqueeze(1) + offsets).to(torch.int64)
        return torch.gather(block_table, 1, indices_to_gather)
