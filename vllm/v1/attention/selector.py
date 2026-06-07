# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# 整体说明：Attention 后端选择器
# ===========================================
# 本文件负责根据当前平台（CUDA/ROCm/XPU 等）和配置参数，选择合适的 Attention 后端实现。
# 核心流程如下：
#   1. get_attn_backend() 是主入口函数，被外部调用以获取 attention 后端类。
#   2. 它先从 vllm 全局配置中收集各种参数（head_size、dtype、block_size 等），
#      封装为 AttentionSelectorConfig（一个 NamedTuple，便于作为缓存 key）。
#   3. 然后调用 _cached_get_attn_backend()，该函数使用 @cache 装饰器进行结果缓存，
#      避免每次调用都重复进行平台检测和后端选择。
#   4. 在缓存函数内部，通过 current_platform.get_attn_backend_cls() 让平台对象
#      根据具体配置选择后端类路径，再通过 resolve_obj_by_qualname() 动态导入。
#   5. 如果选定的后端对 KV cache 布局有特殊要求（如 NHD 或 HND），会通过
#      set_kv_cache_layout() 设置全局布局。
#
# 此外还提供了 get_mamba_attn_backend() 用于选择 Mamba 类（非标准 attention）的后端。

from functools import cache
from typing import NamedTuple, cast, get_args

import torch

import vllm.envs as envs
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.utils.import_utils import resolve_obj_by_qualname
from vllm.v1.attention.backend import AttentionBackend, AttentionType
from vllm.v1.attention.backends.registry import (
    MambaAttentionBackendEnum,
)

logger = init_logger(__name__)


# 中文注释：Attention 后端选择器的配置参数集合。
# 使用 NamedTuple 而非普通 dict 的原因是：NamedTuple 是不可变的且可哈希的，
# 可以直接作为 @cache 装饰器的函数参数 key，从而实现高效的缓存查找。
# 这些参数共同决定了应该选择哪个 attention 后端实现：
#   - head_size: 每个注意力头的维度（如 64、128、256 等）
#   - dtype: 模型的浮点数据类型（如 float16、bfloat16）
#   - kv_cache_dtype: KV cache 的数据类型，可能与模型 dtype 不同（如使用 FP8 量化）
#   - block_size: KV cache 分页管理的块大小，由用户显式指定时才有意义
#   - use_mla: 是否使用 Multi-head Latent Attention（DeepSeek 风格的压缩 KV）
#   - has_sink: 是否使用 Sink Attention（保留初始 token 的注意力机制）
#   - use_sparse: 是否使用稀疏注意力
#   - use_mm_prefix: 是否使用多模态前缀
#   - use_per_head_quant_scales: 是否使用逐头量化缩放因子
#   - attn_type: 注意力类型，默认是 DECODER，也可以是 ENCODER 或 ENCODER_DECODER
#   - use_non_causal: 是否使用非因果注意力（如 ViT 中的双向注意力）
#   - use_batch_invariant: 是否要求批处理结果可复现（确定性模式）
#   - use_kv_connector: 是否使用 KV cache 传输功能（用于分离式 prefill/decode 架构）
class AttentionSelectorConfig(NamedTuple):
    head_size: int
    dtype: torch.dtype
    kv_cache_dtype: CacheDType | None
    block_size: int | None
    use_mla: bool = False
    has_sink: bool = False
    use_sparse: bool = False
    use_mm_prefix: bool = False
    use_per_head_quant_scales: bool = False
    attn_type: str = AttentionType.DECODER
    use_non_causal: bool = False
    use_batch_invariant: bool = False
    use_kv_connector: bool = False

    def __repr__(self):
        return (
            f"AttentionSelectorConfig(head_size={self.head_size}, "
            f"dtype={self.dtype}, "
            f"kv_cache_dtype={self.kv_cache_dtype}, "
            f"block_size={self.block_size}, "
            f"use_mla={self.use_mla}, "
            f"has_sink={self.has_sink}, "
            f"use_sparse={self.use_sparse}, "
            f"use_mm_prefix={self.use_mm_prefix}, "
            f"use_per_head_quant_scales={self.use_per_head_quant_scales}, "
            f"attn_type={self.attn_type}, "
            f"use_non_causal={self.use_non_causal}, "
            f"use_batch_invariant={self.use_batch_invariant}, "
            f"use_kv_connector={self.use_kv_connector})"
        )


# 中文注释：获取 Attention 后端类的主入口函数。
# 这是外部调用者（如 Model Runner 初始化 attention 层时）获取后端实现的入口。
# 执行流程：
#   1. 校验 kv_cache_dtype 是否合法（如果指定了的话）。
#   2. 从全局 vllm_config 中读取 cache_config 和 kv_transfer_config。
#   3. 如果用户显式指定了 block_size，则记录下来，否则为 None。
#   4. 将所有参数封装为 AttentionSelectorConfig。
#   5. 调用 _cached_get_attn_backend() 获取后端类（带缓存）。
# 返回值是 AttentionBackend 的子类（type，不是实例），由调用方后续实例化。
def get_attn_backend(
    head_size: int,
    dtype: torch.dtype,
    kv_cache_dtype: str | None,
    use_mla: bool = False,
    has_sink: bool = False,
    use_sparse: bool = False,
    use_mm_prefix: bool = False,
    use_per_head_quant_scales: bool = False,
    attn_type: str | None = None,
    num_heads: int | None = None,
) -> type[AttentionBackend]:
    """Selects which attention backend to use and lazily imports it."""

    if kv_cache_dtype is not None:
        valid_cache_dtypes = get_args(CacheDType)
        assert kv_cache_dtype in valid_cache_dtypes, (
            f"Invalid kv_cache_dtype: {kv_cache_dtype}. "
            f"Valid values are: {valid_cache_dtypes}"
        )

    from vllm.config import get_current_vllm_config

    vllm_config = get_current_vllm_config()

    cache_config = vllm_config.cache_config
    if cache_config is not None and cache_config.user_specified_block_size:
        block_size = cache_config.block_size
    else:
        block_size = None

    kv_transfer_config = vllm_config.kv_transfer_config
    use_kv_connector = (
        kv_transfer_config is not None and kv_transfer_config.is_kv_transfer_instance
    )

    attn_selector_config = AttentionSelectorConfig(
        head_size=head_size,
        dtype=dtype,
        kv_cache_dtype=cast(CacheDType | None, kv_cache_dtype),
        block_size=block_size,
        use_mla=use_mla,
        has_sink=has_sink,
        use_sparse=use_sparse,
        use_mm_prefix=use_mm_prefix,
        use_per_head_quant_scales=use_per_head_quant_scales,
        attn_type=attn_type or AttentionType.DECODER,
        use_non_causal=vllm_config.attention_config.use_non_causal,
        use_batch_invariant=envs.VLLM_BATCH_INVARIANT,
        use_kv_connector=use_kv_connector,
    )

    return _cached_get_attn_backend(
        backend=vllm_config.attention_config.backend,
        attn_selector_config=attn_selector_config,
        num_heads=num_heads,
    )


# 中文注释：带缓存的 attention 后端查找函数。
# 使用 @cache 装饰器，相同参数组合只会在第一次调用时执行实际的平台检测和后端选择逻辑，
# 后续调用直接返回缓存结果。这是性能关键路径，因为 attention 后端选择在模型初始化时
# 会被多个 attention 层反复调用（每个 transformer 层都有自己的 attention）。
# 内部流程：
#   1. 通过 current_platform.get_attn_backend_cls() 让当前平台根据配置
#      返回合适的后端类的全限定名（如 "vllm.v1.attention.backends.flash_attn.FlashAttentionBackend"）。
#   2. 通过 resolve_obj_by_qualname() 动态导入该类。
#   3. 如果后端要求特定的 KV cache 布局（如 NHD 或 HND），则设置全局布局。
@cache
def _cached_get_attn_backend(
    backend,
    attn_selector_config: AttentionSelectorConfig,
    num_heads: int | None = None,
) -> type[AttentionBackend]:
    from vllm.platforms import current_platform

    attention_cls = current_platform.get_attn_backend_cls(
        backend,
        attn_selector_config=attn_selector_config,
        num_heads=num_heads,
    )
    if not attention_cls:
        raise ValueError(
            f"Invalid attention backend for {current_platform.device_name}"
        )
    backend = resolve_obj_by_qualname(attention_cls)

    # 中文注释：根据选定后端的要求调整 KV cache 的内存布局。
    # 不同的 attention 后端可能对 KV cache 的维度排列有不同的要求：
    #   - NHD (NumHeads, HeadDim): 例如 FlashAttention 默认布局
    #   - HND (Heads, NumTokens, HeadDim): 例如某些 Triton 后端
    # 如果后端不要求特定布局（返回 None），则使用系统默认布局。
    required_layout = backend.get_required_kv_cache_layout()
    if required_layout is not None:
        from vllm.v1.attention.backends.utils import set_kv_cache_layout

        set_kv_cache_layout(required_layout)
        logger.info(
            "Using %s KV cache layout for %s backend.",
            required_layout,
            backend.get_name(),
        )

    return backend


# 中文注释：获取 Mamba 类注意力后端的入口函数。
# Mamba 是一种基于状态空间模型（SSM）的序列建模架构，与标准的 Transformer attention 不同。
# 此函数用于选择 Mamba1、Mamba2、ShortConv、Linear 或 GDN 等非标准注意力后端。
def get_mamba_attn_backend(
    mamba_type: MambaAttentionBackendEnum,
) -> type[AttentionBackend]:
    """Select which mamba attention backend to use and lazily import it."""
    return _cached_get_mamba_attn_backend(mamba_type)


# 中文注释：带缓存的 Mamba 后端查找函数。
# 与 _cached_get_attn_backend 类似，使用 @cache 避免重复的后端选择逻辑。
# 如果开启了批处理不变性模式（VLLM_BATCH_INVARIANT），还会检查后端是否支持该特性。
@cache
def _cached_get_mamba_attn_backend(
    mamba_type: MambaAttentionBackendEnum,
) -> type[AttentionBackend]:
    assert mamba_type and isinstance(mamba_type, MambaAttentionBackendEnum)

    mamba_attn_backend = mamba_type.get_class()
    if envs.VLLM_BATCH_INVARIANT and not mamba_attn_backend.supports_batch_invariance():
        raise RuntimeError(
            "VLLM batch_invariant mode is not supported for "
            f"{mamba_attn_backend.get_name()}."
        )
    return mamba_attn_backend
