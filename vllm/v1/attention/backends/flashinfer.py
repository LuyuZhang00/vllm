# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention layer with FlashInfer."""

# =============================================================================
# 中文说明：FlashInfer 注意力后端模块
# =============================================================================
# FlashInfer 是一个高性能的注意力计算库，提供了基于分页 KV cache（PagedKV）
# 的 prefill 和 decode 注意力核函数。它是 vLLM V1 中可插拔注意力后端之一。
#
# 本模块的核心职责：
# 1. FlashInferBackend：后端注册入口，声明支持的数据类型、head size、
#    GPU compute capability 等能力信息；提供 KV cache shape/stride 的定义。
# 2. FlashInferMetadataBuilder：元数据构建器，负责将 Scheduler 产出的
#    CommonAttentionMetadata 转换为 FlashInfer 核函数所需的 paged_kv_indptr、
#    paged_kv_indices、paged_kv_last_page_len 等结构，并调用 FlashInfer 的
#    plan() 方法预计算注意力执行计划。
# 3. FlashInferMetadata：存储当前 batch 的注意力元数据，包含 prefill/decode
#    的分发信息、slot_mapping、cascade attention 开关等。
# 4. FlashInferImpl：注意力计算实现类，其 forward() 方法根据 prefill/decode
#    路径分发到对应的 FlashInfer 核函数（native FI 或 TRTLLM）。
#
# FlashInfer 后端支持两种核函数路径：
# - Native FlashInfer 路径：使用 BatchPrefillWithPagedKVCacheWrapper 和
#   BatchDecodeWithPagedKVCacheWrapper，适用于通用场景。
# - TRTLLM 路径：使用 TensorRT-LLM 的 trtllm_batch_context_with_kv_cache
#   和 trtllm_batch_decode_with_kv_cache，在 SM100 (Blackwell) 上性能更优，
#   支持 FP8 query 量化、attention sinks 等高级特性。
#
# 数据流概述：
#   SchedulerOutput -> GPUModelRunner -> CommonAttentionMetadata
#     -> FlashInferMetadataBuilder.build() -> FlashInferMetadata
#     -> FlashInferImpl.forward()
#       -> 根据 prefill/decode 路径选择对应的 FlashInfer wrapper 执行注意力计算
#       -> 输出 shape: [num_tokens, num_heads * head_size]
# =============================================================================

from dataclasses import dataclass
from functools import partial
from typing import ClassVar

import numpy as np
import torch
from flashinfer import (
    BatchDecodeWithPagedKVCacheWrapper,
    BatchPrefillWithPagedKVCacheWrapper,
    BatchPrefillWithRaggedKVCacheWrapper,
    MultiLevelCascadeAttentionWrapper,
)
from flashinfer.decode import fast_decode_plan, trtllm_batch_decode_with_kv_cache
from flashinfer.prefill import trtllm_batch_context_with_kv_cache
from flashinfer.utils import FP4Tensor
from typing_extensions import override

from vllm import envs
from vllm.config import (
    CUDAGraphMode,
    VllmConfig,
    get_current_vllm_config_or_none,
)
from vllm.config.cache import CacheDType
from vllm.distributed.parallel_state import get_dcp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8StaticTensorSym,
    kNvfp4Dynamic,
)
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.triton_utils import tl, triton
from vllm.utils.flashinfer import (
    can_use_trtllm_attention,
    use_trtllm_attention,
)
from vllm.utils.math_utils import cdiv
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.utils.torch_utils import (
    canonicalize_singleton_dim_strides,
    is_quantized_kv_cache,
    is_strictly_contiguous,
    nvfp4_kv_cache_full_dim,
    nvfp4_kv_cache_split_views,
)
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.utils import (
    KVCacheLayoutType,
    get_dcp_local_seq_lens,
    get_kv_cache_layout,
    get_num_attention_heads_from_layers,
    get_per_layer_parameters,
    infer_global_hyperparameters,
    split_decodes_and_prefills,
)
from vllm.v1.attention.ops.common import cp_lse_ag_out_rs
from vllm.v1.attention.ops.dcp_alltoall import dcp_a2a_lse_reduce
from vllm.v1.attention.ops.merge_attn_states import merge_attn_states
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVQuantMode,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.utils import CpuGpuBuffer

# 中文注释：FlashInfer 工作区缓冲区大小（batch invariant 模式下）。
# FlashInfer 的注意力核函数需要一个工作区缓冲区用于临时存储。
# 在 batch invariant 模式下使用固定的 2GB 缓冲区，以确保不同 batch size
# 下的计算行为一致（用于调试和可复现性）。
FLASHINFER_WORKSPACE_BUFFER_SIZE_BATCH_INVARIANT = 2048 * 1024 * 1024

# 中文注释：平台相关的 FP8 数据类型，用于 KV cache 的 FP8 量化存储。
FP8_DTYPE = current_platform.fp8_dtype()
# 中文注释：FP4 数据类型，NVFP4 量化 KV cache 使用 uint8 存储。
FP4_DTYPE = torch.uint8

logger = init_logger(__name__)

# 中文注释：TRTLLM 核函数使用的全局工作区缓冲区（惰性初始化）。
# TRTLLM 的 prefill/decode 核函数需要一个大型工作区用于内部临时存储，
# 采用全局单例模式避免重复分配。
trtllm_gen_workspace_buffer = None


# 中文注释：获取 TRTLLM 核函数的工作区缓冲区。
# 采用惰性初始化策略：首次调用时分配，后续调用直接返回已分配的缓冲区。
# 缓冲区大小由环境变量 VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE 控制。
def _get_trtllm_gen_workspace_buffer():
    global trtllm_gen_workspace_buffer
    if trtllm_gen_workspace_buffer is None:
        trtllm_gen_workspace_buffer = torch.zeros(
            envs.VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE, dtype=torch.uint8, device="cuda"
        )
    return trtllm_gen_workspace_buffer


# 中文注释：Triton 内核函数，用于在 TRTLLM prefill 路径中将 FP8 格式的
# KV cache 反量化为 BF16/FP16。
# 当 query 不是 FP8 但 KV cache 是 FP8 时，TRTLLM prefill 注意力核函数
# 不支持这种混合精度组合，因此需要先将 KV cache 反量化。
# 每个 Triton 线程处理一个请求的一个 page，遍历所有 KV head 进行反量化。
@triton.jit
def _trtllm_prefill_attn_kvfp8_dequant(
    kv_cache_ptr,
    block_tables_prefill_ptr,
    block_table_stride,
    mock_kv_cache_ptr,
    k_scale_ptr,
    v_scale_ptr,
    src_stride_page,
    src_stride_kv,
    src_stride_head,
    DST_K_CACHE_STRIDE: tl.constexpr,
    DST_KV_CACHE_STRIDE: tl.constexpr,
    HEAD_STRIDE: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
):
    batch_idx = tl.program_id(0).to(tl.int64)
    mock_block_table_idx = tl.program_id(1).to(tl.int64)
    orig_page_num = tl.load(
        block_tables_prefill_ptr + batch_idx * block_table_stride + mock_block_table_idx
    ).to(tl.int64)
    if orig_page_num <= 0:
        return
    dequant_dtype = mock_kv_cache_ptr.dtype.element_ty

    k_scale_val = tl.load(k_scale_ptr)
    v_scale_val = tl.load(v_scale_ptr)

    mock_page_idx = batch_idx * block_table_stride + mock_block_table_idx + 1
    head_offsets = tl.arange(0, HEAD_STRIDE)

    for h in range(NUM_KV_HEADS):
        h_off = tl.cast(h, tl.int64)

        # Read K from source (supports non-contiguous page/kv/head strides)
        src_k = orig_page_num * src_stride_page + h_off * src_stride_head + head_offsets
        fp8_k = tl.load(kv_cache_ptr + src_k)
        dequant_k = (fp8_k.to(tl.float32) * k_scale_val).to(dequant_dtype)

        # Write K to contiguous mock cache
        dst_k = mock_page_idx * DST_KV_CACHE_STRIDE + h * HEAD_STRIDE + head_offsets
        tl.store(mock_kv_cache_ptr + dst_k, dequant_k)

        # Read V from source (offset by src_stride_kv for the V half)
        src_v = (
            orig_page_num * src_stride_page
            + src_stride_kv
            + h_off * src_stride_head
            + head_offsets
        )
        fp8_v = tl.load(kv_cache_ptr + src_v)
        dequant_v = (fp8_v.to(tl.float32) * v_scale_val).to(dequant_dtype)

        # Write V to contiguous mock cache
        dst_v = (
            mock_page_idx * DST_KV_CACHE_STRIDE
            + DST_K_CACHE_STRIDE
            + h * HEAD_STRIDE
            + head_offsets
        )
        tl.store(mock_kv_cache_ptr + dst_v, dequant_v)


# 中文注释：将 FP8 KV cache 反量化并重新组织为 TRTLLM prefill 所需的格式。
# 当 query 是 BF16/FP16 而 KV cache 是 FP8 时，TRTLLM prefill 核函数不支持
# 这种混合精度，因此需要：
# 1. 分配一个 mock_kv_cache，存放反量化后的 BF16/FP16 KV 数据
# 2. 生成 mock_block_table，将原始 page 映射到 mock_kv_cache 中的连续位置
# 3. 通过 Triton 内核并行完成反量化拷贝
# 参数：
#   kv_cache: 原始 FP8 KV cache，shape: [num_blocks, 2, num_kv_heads, block_size, head_size]
#   block_tables_prefill: prefill 请求的 block table，shape: [batch_size, max_pages]
#   k_scale, v_scale: K 和 V 的反量化缩放因子
#   dequant_dtype: 反量化目标类型（bf16 或 fp16）
# 返回：
#   (mock_kv_cache, mock_block_table): 反量化后的 KV cache 和对应的 block table
def trtllm_prefill_attn_kvfp8_dequant(
    kv_cache: torch.Tensor,
    block_tables_prefill: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    dequant_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, num_of_page_per_token = block_tables_prefill.shape
    s = kv_cache.shape
    assert s[1] == 2
    assert dequant_dtype in (torch.bfloat16, torch.float16)

    num_kv_heads, block_size, head_size = s[2], s[3], s[4]
    head_stride = block_size * head_size
    k_cache_stride = num_kv_heads * head_stride
    kv_cache_stride = k_cache_stride * s[1]

    strides = kv_cache.stride()
    assert strides[3] == head_size and strides[4] == 1, (
        "For kv cache layouts, (block_size, head_size) "
        f"dimensions must be contiguous, got strides {strides}"
    )

    new_s = (batch_size * num_of_page_per_token + 1, s[1], s[2], s[3], s[4])
    # mock kv cache contains just the pages needed by this prefill
    mock_kv_cache = torch.empty(new_s, dtype=dequant_dtype, device=kv_cache.device)
    # we simply sequentially index the pages needed by this prefill
    mock_block_table = torch.arange(
        start=1,
        end=batch_size * num_of_page_per_token + 1,
        dtype=torch.int32,
        device=block_tables_prefill.device,
    ).reshape(batch_size, num_of_page_per_token)
    grid = (batch_size, num_of_page_per_token)
    _trtllm_prefill_attn_kvfp8_dequant[grid](
        kv_cache,
        block_tables_prefill,
        num_of_page_per_token,
        mock_kv_cache,
        k_scale,
        v_scale,
        strides[0],
        strides[1],
        strides[2],
        k_cache_stride,
        kv_cache_stride,
        head_stride,
        num_kv_heads,
    )
    return mock_kv_cache, mock_block_table


# 中文注释：DCP（Decode Context Parallel）prefill 包装器。
# 用于支持上下文并行（Context Parallelism）场景下的 prefill 注意力计算。
# DCP 将长序列的 prefill 分散到多个 GPU 上并行执行，每个 GPU 只计算
# 部分 KV cache 对应的注意力，最后通过 all-gather 或 all-to-all 通信
# 合并结果。
# 内部维护两个 FlashInfer wrapper：
# - _context: 处理从 KV cache 中读取的历史 token 的注意力（非因果）
# - _new_tokens: 处理当前 batch 中新 token 之间的注意力（因果）
# 最终通过 merge_attn_states 合并两部分的输出和 LSE（log-sum-exp）。
class BatchDCPPrefillWrapper:
    def __init__(
        self,
        workspace_buffer: torch.Tensor | None = None,
        dcp_a2a: bool = False,
    ):
        if dcp_a2a:
            self._dcp_combine = partial(dcp_a2a_lse_reduce, is_lse_base_on_e=False)
        else:
            self._dcp_combine = partial(cp_lse_ag_out_rs, is_lse_base_on_e=False)
        self._context = BatchPrefillWithPagedKVCacheWrapper(
            workspace_buffer, get_kv_cache_layout()
        )
        self._new_tokens = BatchPrefillWithRaggedKVCacheWrapper(workspace_buffer)

    def plan(
        self,
        qo_indptr_cpu: torch.Tensor,
        paged_kv_indptr_cpu: torch.Tensor,
        paged_kv_indices: torch.Tensor,
        paged_kv_last_page_len_cpu: torch.Tensor,
        page_size: int,
        num_qo_heads: int,
        dcp_world_size: int,
        num_kv_heads: int,
        head_dim: int,
        sm_scale: float,
        window_left: int,
        logits_soft_cap: float | None,
        q_data_type: torch.dtype,
        kv_cache_dtype: torch.dtype,
        prefill_fixed_split_size: int,
        disable_split_kv: bool,
    ):
        """Plan the prefill operation with given parameters."""
        self._context.plan(
            qo_indptr=qo_indptr_cpu,
            paged_kv_indptr=paged_kv_indptr_cpu,
            paged_kv_indices=paged_kv_indices,
            paged_kv_last_page_len=paged_kv_last_page_len_cpu,
            num_qo_heads=num_qo_heads * dcp_world_size,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim,
            page_size=page_size,
            causal=False,  # This is context run
            sm_scale=sm_scale,
            window_left=window_left,
            logits_soft_cap=logits_soft_cap,
            q_data_type=q_data_type,
            kv_data_type=kv_cache_dtype,
            fixed_split_size=prefill_fixed_split_size,
            disable_split_kv=disable_split_kv,
        )
        self._new_tokens.plan(
            qo_indptr=qo_indptr_cpu,
            kv_indptr=qo_indptr_cpu,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim,
            head_dim_vo=head_dim,
            causal=True,  # This is newtokens run
            sm_scale=sm_scale,
            window_left=window_left,
            logits_soft_cap=logits_soft_cap,
            q_data_type=q_data_type,
        )

    # 中文注释：执行 DCP prefill 注意力计算。
    # 流程：
    # 1. all_gather：将本 GPU 的 query 广播到所有 DCP GPU，使每个 GPU 都能
    #    计算全局 query 对本地 KV cache 的注意力。
    # 2. context run：计算全局 query 对已缓存 KV 的注意力（非因果），
    #    返回输出和 LSE。
    # 3. dcp_combine：通过 all-gather/all-to-all 合并各 GPU 的 context 输出。
    # 4. new_tokens run：计算 query 对当前 batch 新 token 的注意力（因果）。
    # 5. merge_attn_states：合并 context 和 new_tokens 的输出，使用 LSE
    #    加权平均得到最终输出。
    def run(
        self,
        layer: torch.nn.Module,
        prefill_query: torch.Tensor,
        kv_cache_permute: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        out: torch.Tensor,
    ):
        prefill_query_across_dcp = get_dcp_group().all_gather(
            prefill_query.contiguous(), dim=1
        )
        output_context_tmp, lse_context_tmp = self._context.run(
            prefill_query_across_dcp,
            kv_cache_permute,
            k_scale=layer._k_scale_float,
            v_scale=layer._v_scale_float,
            return_lse=True,
        )
        output_context, lse_context = self._dcp_combine(
            output_context_tmp,
            lse_context_tmp,
            get_dcp_group(),
            return_lse=True,
        )
        lse_context = lse_context.transpose(0, 1).contiguous()

        output_query, lse_query = self._new_tokens.run(
            prefill_query,
            key,
            value,
            return_lse=True,
        )
        lse_query = lse_query.transpose(0, 1).contiguous()

        merge_attn_states(
            out,
            output_context,
            lse_context,
            output_query,
            lse_query,
        )
        return out


# 中文注释：FlashInfer 注意力后端注册类。
# 职责：
# 1. 声明后端名称 "FLASHINFER"，供调度器和模型运行器按名称查找。
# 2. 声明支持的数据类型（float16, bfloat16）和 KV cache 数据类型
#   （auto, float16, bfloat16, fp8, nvfp4 等）。
# 3. 提供 KV cache shape/stride 的计算方法，供 KV cache manager 分配显存。
# 4. 工厂方法：get_impl_cls() 返回 FlashInferImpl，
#    get_builder_cls() 返回 FlashInferMetadataBuilder。
# 5. 能力查询：支持的 head size、GPU compute capability、是否支持 sink 等。
class FlashInferBackend(AttentionBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
        "fp8_e5m2",
        "nvfp4",
    ]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        # Note: Not sure for all platforms, but on Blackwell,
        # only support a page size of 16, 32, 64.
        return [16, 32, 64]

    @staticmethod
    def get_name() -> str:
        return "FLASHINFER"

    @staticmethod
    def get_impl_cls() -> type["FlashInferImpl"]:
        return FlashInferImpl

    @staticmethod
    def get_builder_cls() -> type["FlashInferMetadataBuilder"]:
        return FlashInferMetadataBuilder

    # 中文注释：计算 KV cache 的逻辑形状。
    # FlashInfer 使用分页 KV cache，形状为：
    #   (num_blocks, 2, block_size, num_kv_heads, head_size)
    # 其中 2 表示 K 和 V 两个缓存。NVFP4 格式在最后一维打包了数据和缩放因子。
    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        if cache_dtype_str == "nvfp4":
            # Packed layout: fp4 data + fp8 block scales in last dim
            last_dim = nvfp4_kv_cache_full_dim(head_size)
            return (num_blocks, 2, block_size, num_kv_heads, last_dim)
        return (num_blocks, 2, block_size, num_kv_heads, head_size)

    # 中文注释：获取 KV cache 的 stride 排列顺序。
    # 不同的 KV cache 布局（NHD vs HND）对应不同的维度排列方式。
    # NHD: (num_blocks, 2, block_size, num_kv_heads, head_size)
    # HND: (num_blocks, 2, num_kv_heads, block_size, head_size)
    # stride_order 描述了从 get_kv_cache_shape 到实际内存布局的维度置换关系。
    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        # `stride_order` indicates the permutation that gets us from
        # `get_kv_cache_shape` to the actual memory layout we want.
        cache_layout = get_kv_cache_layout()
        if cache_layout == "NHD" and include_num_layers_dimension:
            # (num_blocks, num_layers, 2, block_size, num_kv_heads, head_size)
            return (1, 0, 2, 3, 4, 5)
        elif cache_layout == "NHD":
            stride_order = (0, 1, 2, 3, 4)
        elif cache_layout == "HND" and include_num_layers_dimension:
            # (num_blocks, 2, num_kv_heads, num_layers, block_size, head_size)
            return (1, 2, 4, 0, 3, 5)
        elif cache_layout == "HND":
            stride_order = (0, 1, 3, 2, 4)
        else:
            raise ValueError(f"Unknown cache layout format {cache_layout}.")
        return stride_order

    # 中文注释：将 KV cache 的字符串类型标识转换为 FlashInfer 所需的 torch.dtype。
    # FlashInfer 核函数使用原生的 torch dtype 来指定 KV cache 的数据类型。
    @staticmethod
    def get_dtype_for_flashinfer(kv_cache_dtype: str) -> torch.dtype:
        if kv_cache_dtype in ("fp8", "fp8_e4m3"):
            return torch.float8_e4m3fn
        elif kv_cache_dtype == "fp8_e5m2":
            return torch.float8_e5m2
        elif kv_cache_dtype == "nvfp4":
            return torch.uint8
        else:
            raise ValueError(f"Unrecognized dtype: {kv_cache_dtype}")

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        # https://github.com/flashinfer-ai/flashinfer/blob/3d55c71a62052c590c130897d3a3db49b14fcc34/include/flashinfer/utils.cuh#L157
        return [64, 128, 256, 512]

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability >= DeviceCapability(7, 5) and capability <= DeviceCapability(
            12, 1
        )

    @classmethod
    def supports_sink(cls) -> bool:
        """FlashInfer supports sinks when TRTLLM attention is available (SM100)."""
        from vllm.utils.flashinfer import (
            force_use_trtllm_attention,
            supports_trtllm_attention,
        )

        # Respect explicit disable flag (e.g.,
        # --attention-config.use_trtllm_attention=0)
        if force_use_trtllm_attention() is False:
            return False

        # Check if TRTLLM is supported on this platform
        return supports_trtllm_attention()

    @classmethod
    def get_required_kv_cache_layout(cls) -> KVCacheLayoutType | None:
        capability = current_platform.get_device_capability()
        if capability is not None and capability.major == 10:
            return "HND"
        return None

    # 中文注释：FlashInfer 后端不通过 forward() 方法更新 KV cache，
    # 而是由 AttentionImpl.do_kv_cache_update() 专门处理。
    forward_includes_kv_cache_update: bool = False


# 中文注释：FlashInfer 原生 prefill 路径的元数据。
# 包含一个已 plan 好的 BatchPrefillWithPagedKVCacheWrapper 或
# BatchDCPPrefillWrapper，可以直接调用 run() 执行 prefill 注意力计算。
@dataclass
class FIPrefill:
    """Metadata for the native FlashInfer prefill pathway (non-TRTLLM)."""

    wrapper: BatchPrefillWithPagedKVCacheWrapper | BatchDCPPrefillWrapper


# 中文注释：FlashInfer 原生 decode 路径的元数据。
# 包含一个已 plan 好的 BatchDecodeWithPagedKVCacheWrapper。
@dataclass
class FIDecode:
    """Metadata for the native FlashInfer decode pathway (non-TRTLLM)."""

    wrapper: BatchDecodeWithPagedKVCacheWrapper


# 中文注释：TRTLLM prefill 路径的元数据。
# 与 FlashInfer 原生路径不同，TRTLLM 路径直接使用 GPU 上的 block_tables
# 和 seq_lens，不需要在 CPU 上预计算 paged_kv_indptr/indices 等结构。
# 这种路径在 SM100 (Blackwell) GPU 上性能更优，支持 FP8 query 量化。
@dataclass
class TRTLLMPrefill:
    """Metadata for the TRTLLM prefill pathway."""

    block_tables: torch.Tensor
    """
    The slice of the block table tensor corresponding *only* to prefill requests.
    Shape: [num_prefills, max_num_blocks_per_seq]
    """

    seq_lens: torch.Tensor
    """
    The slice of the sequence lengths tensor corresponding *only* to prefill requests.
    Shape: [num_prefills]
    """

    cum_seq_lens_q: torch.Tensor
    cum_seq_lens_kv: torch.Tensor

    max_q_len: int
    """
    The maximum query length *among prefill requests*.
    """

    max_seq_len: int
    """The maximum sequence length for KV Cache."""


# 中文注释：TRTLLM decode 路径的元数据。
# 与 TRTLLMPrefill 类似，直接使用 GPU 上的 block_tables 和 seq_lens。
@dataclass
class TRTLLMDecode:
    """Metadata for the TRTLLM decode pathway."""

    block_tables: torch.Tensor
    """
    The slice of the block table tensor corresponding *only* to decode requests.
    Shape: [num_decodes, max_num_blocks_per_seq]
    """

    seq_lens: torch.Tensor
    """
    The slice of the sequence lengths tensor corresponding *only* to decode requests.
    Shape: [num_decodes]
    """

    max_seq_len: int
    """The maximum sequence length for KV Cache."""


# 中文注释：FlashInfer 注意力的元数据数据类。
# 这是 FlashInferMetadataBuilder.build() 的输出，FlashInferImpl.forward() 的输入。
# 它包含了当前 batch 执行一次注意力前向传播所需的全部信息。
# 核心字段：
# - num_actual_tokens: 有效 token 数（不含 padding）
# - slot_mapping: 将 token 写入 KV cache 的物理位置映射
# - num_decodes/num_prefills: decode 和 prefill 请求的数量
# - prefill/decode: 各路径的特定元数据（FI 原生或 TRTLLM）
# - use_cascade: 是否使用 cascade attention（共享前缀场景）
@dataclass
class FlashInferMetadata:
    num_actual_tokens: int
    """Total number of tokens in the batch (excluding padding)."""

    slot_mapping: torch.Tensor
    """Tensor for writing K/V to the cache. Shape: [num_actual_tokens]"""

    q_data_type: torch.dtype

    num_decodes: int
    num_decode_tokens: int
    num_prefills: int
    num_prefill_tokens: int

    prefill: FIPrefill | TRTLLMPrefill | None
    """
    Holds the metadata for the prefill portion of the batch.
    Will be `None` if `num_prefill_tokens == 0`.
    """

    decode: FIDecode | TRTLLMDecode | None
    """
    Holds the metadata for the decode portion of the batch.
    Will be `None` if `num_decode_tokens == 0`.
    """

    # --- Special Case: Cascade Attention ---

    use_cascade: bool
    """
    If True, the entire batch is a cascade attention call, and the
    `prefill` and `decode` fields will both be None.
    """

    cascade_wrapper: MultiLevelCascadeAttentionWrapper | None


# 中文注释：FlashInfer 注意力元数据构建器。
# 职责：将 GPUModelRunner 产出的 CommonAttentionMetadata 转换为
# FlashInferImpl.forward() 所需的 FlashInferMetadata。
# 核心流程（build 方法）：
# 1. 将 batch 中的请求分为 decode 和 prefill 两组
# 2. 根据 GPU 能力和配置选择核函数路径（native FI 或 TRTLLM）
# 3. 为 prefill 路径计算 paged_kv_indptr/indices/last_page_len 或
#    准备 TRTLLM 所需的 block_tables/seq_lens
# 4. 为 decode 路径准备对应的元数据
# 5. 调用 FlashInfer 的 plan() 方法预计算执行计划
# 关键设计：
# - 支持 CUDA Graph 捕获，为每个 batch size 预创建 decode wrapper
# - 支持 DCP（上下文并行），处理跨 GPU 的 KV cache 分片
# - 支持 cascade attention，利用共享前缀减少计算量
class FlashInferMetadataBuilder(AttentionMetadataBuilder[FlashInferMetadata]):
    # 中文注释：decode/encode 重排阈值。
    # 当请求的 query 长度 <= 此值时被视为 decode 请求，否则为 prefill 请求。
    # 值为 1 表示只有单 token 请求被视为 decode（标准连续批处理行为）。
    # 值大于 1 时支持投机解码场景，允许多 token 的请求也被视为 decode。
    reorder_batch_threshold: int = 1

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.cache_config = vllm_config.cache_config
        self.model_config = vllm_config.model_config
        self.attention_config = vllm_config.attention_config
        self._workspace_buffer = None
        self._prefill_wrapper: (
            BatchPrefillWithPagedKVCacheWrapper | BatchDCPPrefillWrapper | None
        ) = None  # Wrapper for prefill/append
        self._decode_wrapper = None  # Wrapper for decode (general shape)

        # 中文注释：batch invariant 模式下的固定分割大小配置。
        # FlashInfer 支持将长序列的注意力计算分割为多个固定大小的块，
        # 以减少内存碎片和提高 CUDA 利用率。在 batch invariant 模式下，
        # 使用固定大小以确保不同 batch 的计算行为完全一致。
        if envs.VLLM_BATCH_INVARIANT:
            self.decode_fixed_split_size = 2048
            self.prefill_fixed_split_size = 4096
            self.disable_split_kv = True
        else:
            self.decode_fixed_split_size = -1
            self.prefill_fixed_split_size = -1
            self.disable_split_kv = False

        self.compilation_config = vllm_config.compilation_config
        max_num_pages_per_req = cdiv(
            self.model_config.max_model_len, self.kv_cache_spec.block_size
        )
        max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        max_num_pages = max_num_reqs * max_num_pages_per_req
        speculative_config = vllm_config.speculative_config
        num_spec_tokens = (
            speculative_config.num_speculative_tokens
            if speculative_config is not None
            else 0
        )
        # 中文注释：CUDA Graph 支持判断。
        # 当启用 FULL 模式的 CUDA Graph 时，需要为每个 batch size 预创建
        # 一个 decode wrapper（FlashInfer 要求），以便在 CUDA Graph 捕获时
        # 固定 kernel 的执行计划。
        self.enable_cuda_graph = (
            self.compilation_config.cudagraph_mode.decode_mode() == CUDAGraphMode.FULL
        )
        if self.enable_cuda_graph:
            # For full cudagraph capture, one `decode_wrapper` for each batch
            # size is needed for FlashInfer.
            self._decode_wrappers_cudagraph: dict[
                int, BatchDecodeWithPagedKVCacheWrapper
            ] = {}
            # 中文注释：CUDA Graph 模式下的最大 batch size，
            # 考虑了投机解码的额外 token 数。
            self._decode_cudagraph_max_bs = (1 + num_spec_tokens) * max_num_reqs
            if self.compilation_config.max_cudagraph_capture_size is not None:
                self._decode_cudagraph_max_bs = min(
                    self._decode_cudagraph_max_bs,
                    self.compilation_config.max_cudagraph_capture_size,
                )
        # 中文注释：初始化 DCP（Decode Context Parallel）相关配置。
        # DCP 将 KV cache 按 head 分片到多个 GPU 上，每个 GPU 只存储
        # 部分 head 的 KV 数据，从而减少单 GPU 的显存占用。
        # dcp_world_size > 1 表示启用了 DCP。
        try:
            self.dcp_world_size = get_dcp_group().world_size
            self.dcp_rank = get_dcp_group().rank_in_group
            self.dcp_kv_cache_interleave_size = (
                vllm_config.parallel_config.dcp_kv_cache_interleave_size
            )
        except AssertionError:
            # DCP might not be initialized in testing
            self.dcp_world_size = 1
            self.dcp_rank = 0
            self.dcp_kv_cache_interleave_size = 1
        self.use_dcp = self.dcp_world_size > 1
        # 中文注释：DCP 通信后端选择。"a2a" 使用 all-to-all 通信，
        # 否则使用 all-gather + reduce-scatter。
        self.dcp_a2a = (
            self.use_dcp and vllm_config.parallel_config.dcp_comm_backend == "a2a"
        )

        # Compatible with models with non-uniform per-layer head counts.
        self.num_qo_heads = get_num_attention_heads_from_layers(
            vllm_config, layer_names
        ) or self.model_config.get_num_attention_heads(self.vllm_config.parallel_config)

        self.num_kv_heads = self.kv_cache_spec.num_kv_heads
        self.head_dim = self.kv_cache_spec.head_size
        self.page_size = self.kv_cache_spec.block_size

        # 中文注释：根据 KV cache 的量化模式确定实际使用的数据类型。
        # - KVQuantMode.NONE: 不量化，使用模型默认 dtype（bf16/fp16）
        # - FP8: 使用 float8_e4m3fn 或 float8_e5m2
        # - NVFP4: 使用 uint8 存储 FP4 数据和 FP8 缩放因子
        if self.kv_cache_spec.kv_quant_mode != KVQuantMode.NONE:
            self.cache_dtype = self.cache_config.cache_dtype
            # Cannot use self.kv_cache_spec.dtype here because kv_cache_spec
            # storage dtype may not be the same as the op dtype (uint8 vs fp8_e4m3)
            self.is_kvcache_nvfp4 = self.cache_dtype == "nvfp4"
            if self.is_kvcache_nvfp4:
                # For NVFP4, kv_cache_dtype stays as the string "nvfp4"
                # which is passed to FlashInferImpl
                self.kv_cache_dtype = self.cache_dtype
            else:
                self.kv_cache_dtype = FlashInferBackend.get_dtype_for_flashinfer(
                    self.cache_dtype
                )
        else:
            self.cache_dtype = "auto"
            self.is_kvcache_nvfp4 = False
            assert self.kv_cache_spec.dtype == self.model_config.dtype
            self.kv_cache_dtype = self.kv_cache_spec.dtype

        # 中文注释：确定 query 的数据类型。
        # 当 TRTLLM 注意力可用且未禁用 query 量化时，尝试使用与 KV cache
        # 相同的量化类型（如 FP8）作为 query 类型，以利用 TRTLLM 的
        # 混合精度加速。否则使用模型默认的 dtype（bf16/fp16）。
        # Use model dtype as q dtype when TRTLLM attn is not supported, or
        # --attention-config.disable_flashinfer_q_quantization is set to 1. Otherwise,
        # try to use fp8 q if kv cache is fp8, and will fall back to model dtype
        # if TRTLLM attention kernel is not used when building attn metadata
        can_use_trtllm = can_use_trtllm_attention(self.num_qo_heads, self.num_kv_heads)

        if (
            can_use_trtllm
            and not vllm_config.attention_config.disable_flashinfer_q_quantization
        ):
            if self.is_kvcache_nvfp4:
                # NVFP4 KV cache uses FP8 quantized queries
                self.q_data_type = FlashInferBackend.get_dtype_for_flashinfer(
                    "fp8_e4m3"
                )
            else:
                self.q_data_type = self.kv_cache_dtype
        else:
            self.q_data_type = self.model_config.dtype

        # 中文注释：decode 路径优先使用 TRTLLM 注意力核函数。
        # TRTLLM decode 支持 UNIFORM_BATCH 模式的 CUDA Graph，可以将
        # 不同 batch size 统一捕获为同一个 CUDA Graph，减少编译开销。
        # Prefer TRTLLM attention for decoding in all cases.
        # This allows us to use AttentionCGSupport.UNIFORM_BATCH mode.
        self.use_trtllm_decode_attention = can_use_trtllm
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=can_use_trtllm)

        self._cascade_wrapper = None  # Wrapper for cascade attention

        # Global hyperparameters shared by all attention layers
        # TODO: discard this for trtllm-gen backend
        self.global_hyperparameters = infer_global_hyperparameters(
            get_per_layer_parameters(vllm_config, layer_names, FlashInferImpl)
        )
        self.sm_scale = self.global_hyperparameters.sm_scale
        self.window_left = self.global_hyperparameters.window_left
        self.logits_soft_cap = self.global_hyperparameters.logits_soft_cap
        self.has_sinks = self.global_hyperparameters.has_sinks
        if self.has_sinks and not can_use_trtllm:
            raise NotImplementedError(
                "FlashInfer backend currently does not support attention "
                "sinks, please use trtllm on blackwell or flash attention on "
                "earlier GPUs."
            )
        # 中文注释：预分配持久化的 CPU/GPU 缓冲区。
        # FlashInfer 的 plan/execute 模式需要以下结构来描述分页 KV cache：
        # - paged_kv_indptr: 累积页数指针，shape [num_reqs + 1]
        #   paged_kv_indptr[i] 到 paged_kv_indptr[i+1] 之间的索引
        #   表示第 i 个请求的所有 page
        # - paged_kv_indices: 所有请求的 page id 扁平数组
        # - paged_kv_last_page_len: 每个请求最后一页的有效 token 数
        # 这些缓冲区在每次 build() 时被复用，避免频繁分配/释放内存。
        # Preparing persistent buffers
        # Since we do not have explicit synchronization in ModelRunnerV2, we do not pin
        # reused CPU buffers to avoid a race condition between step N async copies to
        # GPU and step N+1 buffer updates.
        self.pin_memory = (
            not vllm_config.use_v2_model_runner and is_pin_memory_available()
        )
        self.paged_kv_indptr = self._make_buffer(max_num_reqs + 1)
        self.paged_kv_indptr_cpu_buffer = torch.zeros_like(
            self.paged_kv_indptr.cpu, pin_memory=self.pin_memory
        )  # Extra buffer for mutable paged_kv_indptr.cpu in cuda graph mode
        self.paged_kv_indices = self._make_buffer(max_num_pages)
        self.paged_kv_last_page_len = self._make_buffer(max_num_reqs)

    # 中文注释：创建一个 CPU/GPU 双端缓冲区。
    # CpuGpuBuffer 同时维护 CPU 和 GPU 上的张量，并支持 numpy 视图，
    # 方便在 CPU 上用 numpy 高效计算，然后异步拷贝到 GPU。
    def _make_buffer(
        self, *size: int | torch.SymInt, dtype: torch.dtype = torch.int32
    ) -> CpuGpuBuffer:
        return CpuGpuBuffer(
            *size,
            dtype=dtype,
            device=self.device,
            pin_memory=self.pin_memory,
            with_numpy=True,
        )

    # 中文注释：查询 FlashInfer 后端对 CUDA Graph 的支持级别。
    # - UNIFORM_BATCH: 当 TRTLLM decode 可用时，支持任意 batch size 的
    #   统一 CUDA Graph 捕获，性能最优。
    # - UNIFORM_SINGLE_TOKEN_DECODE: 当 TRTLLM 不可用时，只能为每个
    #   batch size 单独捕获 CUDA Graph，且仅限单 token decode。
    @override  # type: ignore[misc]
    @classmethod
    def get_cudagraph_support(
        cls: type["FlashInferMetadataBuilder"],
        vllm_config: VllmConfig,
        kv_cache_spec: AttentionSpec,
    ) -> AttentionCGSupport:
        """Get the cudagraph support level for FlashInfer attention.

        This depends on whether we can use TRTLLM attention for decodes, since we can
        only do UNIFORM_SINGLE_TOKEN_DECODE if it is unavailable.
        To check this, we must call can_use_trtllm_attention with the number of KV
        heads from the kv_cache_spec. We check all available KV cache specs and
        only return UNIFORM_BATCH if all of them support TRTLLM attention.
        """
        # For UniformTypeKVCacheSpecs, check all contained specs
        kv_specs = (
            kv_cache_spec.kv_cache_specs.values()
            if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs)
            else [kv_cache_spec]
        )
        num_qo_heads = vllm_config.model_config.get_num_attention_heads(
            vllm_config.parallel_config
        )
        has_trtllm_support: bool = len(kv_specs) > 0
        for spec in kv_specs:
            if not isinstance(spec, AttentionSpec):
                # FlashInfer only applies to attention, so we don't consider other types
                # of KV spec (e.g. Mamba) here. This is mostly for type checking.
                continue
            if not can_use_trtllm_attention(
                num_qo_heads=num_qo_heads,
                num_kv_heads=spec.num_kv_heads,
            ):
                has_trtllm_support = False
                break

        if has_trtllm_support:
            return AttentionCGSupport.UNIFORM_BATCH
        else:
            return AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE

    # 中文注释：获取或创建 FlashInfer 工作区缓冲区。
    # FlashInfer 的注意力核函数需要一个临时工作区用于存储中间结果。
    # 在 batch invariant 模式下使用固定的 2GB 大小。
    def _get_workspace_buffer(self):
        if self._workspace_buffer is None:
            buffer_size = envs.VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE
            if envs.VLLM_BATCH_INVARIANT:
                buffer_size = FLASHINFER_WORKSPACE_BUFFER_SIZE_BATCH_INVARIANT
            self._workspace_buffer = torch.zeros(
                buffer_size, dtype=torch.uint8, device=self.device
            )
        return self._workspace_buffer

    def set_workspace_buffer(self, workspace_buffer: torch.Tensor):
        self._workspace_buffer = workspace_buffer

    # 中文注释：获取或创建 prefill 注意力包装器。
    # - DCP 模式：创建 BatchDCPPrefillWrapper，支持跨 GPU 的上下文并行。
    # - 普通模式：创建 BatchPrefillWithPagedKVCacheWrapper。
    # - NVFP4 KV cache：使用 trtllm-gen 后端（FlashAttention 不支持 NVFP4）。
    # 包装器在首次调用时惰性创建，后续复用。
    def _get_prefill_wrapper(
        self,
    ) -> BatchPrefillWithPagedKVCacheWrapper | BatchDCPPrefillWrapper:
        if self._prefill_wrapper is None:
            if self.use_dcp:
                self._prefill_wrapper = BatchDCPPrefillWrapper(
                    workspace_buffer=self._get_workspace_buffer(),
                    dcp_a2a=self.dcp_a2a,
                )
            else:
                # NVFP4 KV cache requires the trtllm-gen backend inside
                # the wrapper; fa2/fa3 do not support nvfp4.
                backend = "trtllm-gen" if self.is_kvcache_nvfp4 else "auto"
                self._prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(
                    self._get_workspace_buffer(),
                    get_kv_cache_layout(),
                    backend=backend,
                )
        assert self._prefill_wrapper is not None
        return self._prefill_wrapper

    # 中文注释：获取或创建 decode 注意力包装器。
    # 参数：
    #   batch_size: decode 请求的 token 总数（投机解码时可能 > 请求数）
    #   use_cudagraph: 是否使用 CUDA Graph 模式
    # CUDA Graph 模式下，为每个 batch size 缓存一个独立的 wrapper，
    # 因为 CUDA Graph 捕获要求固定的输入形状。
    # 非 CUDA Graph 模式下，复用同一个 wrapper。
    def _get_decode_wrapper(self, batch_size: int, use_cudagraph: bool = False):
        if use_cudagraph:
            decode_wrapper = self._decode_wrappers_cudagraph.get(batch_size, None)
        else:
            decode_wrapper = self._decode_wrapper

        if decode_wrapper is None:
            if use_cudagraph:
                paged_kv_indptr = self.paged_kv_indptr.gpu[: batch_size + 1]
                paged_kv_indices = self.paged_kv_indices.gpu
                paged_kv_last_page_len = self.paged_kv_last_page_len.gpu[:batch_size]
            else:
                paged_kv_indptr = None
                paged_kv_indices = None
                paged_kv_last_page_len = None
            # NVFP4 KV cache requires the trtllm-gen backend inside
            # the wrapper; fa2/fa3 do not support nvfp4.
            backend = "trtllm-gen" if self.is_kvcache_nvfp4 else "auto"
            decode_wrapper = BatchDecodeWithPagedKVCacheWrapper(
                self._get_workspace_buffer(),
                get_kv_cache_layout(),
                use_cuda_graph=use_cudagraph,
                paged_kv_indptr_buffer=paged_kv_indptr,
                paged_kv_indices_buffer=paged_kv_indices,
                paged_kv_last_page_len_buffer=paged_kv_last_page_len,
                # Tensor cores are enabled by default because the perf would be
                # at least as good as cuda cores for all attention ops in latest
                # gpus.
                use_tensor_cores=True,
                backend=backend,
            )

            # save the decode wrapper
            if use_cudagraph:
                self._decode_wrappers_cudagraph[batch_size] = decode_wrapper
            else:
                self._decode_wrapper = decode_wrapper

        return decode_wrapper

    # 中文注释：获取或创建 cascade attention 包装器。
    # Cascade attention 用于共享前缀场景：当多个请求有相同的系统提示时，
    # 共享前缀的 KV cache 只存储一份，所有请求共用，从而减少显存占用
    # 和重复计算。使用 2 级级联（共享前缀 + 各请求独有部分）。
    def _get_cascade_wrapper(self):
        if self._cascade_wrapper is None:
            self._cascade_wrapper = MultiLevelCascadeAttentionWrapper(
                2, self._get_workspace_buffer(), get_kv_cache_layout()
            )
        return self._cascade_wrapper

    # 中文注释：计算 FlashInfer 原生路径所需的分页 KV cache 元数据。
    # 将 block_table_tensor（二维，每行一个请求的 page 列表）转换为
    # FlashInfer 所需的三个一维结构：
    # 1. paged_kv_indptr: 累积页数指针，用于标识每个请求的 page 范围
    # 2. paged_kv_indices: 所有请求的 page id 扁平数组
    # 3. paged_kv_last_page_len: 每个请求最后一页的有效 token 数
    # 这些结构是 FlashInfer 的 paged KV attention 核函数的标准输入格式。
    def _compute_flashinfer_kv_metadata(
        self,
        num_blocks_np: np.ndarray,
        seq_lens_np: np.ndarray,
        block_table_tensor: torch.Tensor,
        num_reqs: int,
        page_size: int,
    ) -> torch.Tensor:
        """
        Compute paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len for FlashInfer
        attention.

        Results are stored in self.paged_kv_indptr,
        self.paged_kv_indices, self.paged_kv_last_page_len buffers.

        Returns paged_kv_indices, a GPU tensor with shape [num_actual_pages].
        """
        # write self.paged_kv_indptr_cpu inplace (0-index is always 0)
        np.cumsum(
            num_blocks_np,
            dtype=np.int32,
            out=self.paged_kv_indptr.np[1 : num_reqs + 1],
        )
        # NOTE(woosuk): Because self.paged_kv_indptr_cpu can be modified
        # after this line (e.g., for cuda graphs), we need to copy the data to
        # self.paged_kv_indptr_buffer to avoid race condition.
        self.paged_kv_indptr_cpu_buffer[: num_reqs + 1] = self.paged_kv_indptr.cpu[
            : num_reqs + 1
        ]
        paged_kv_indptr = self.paged_kv_indptr.gpu[: num_reqs + 1]
        paged_kv_indptr.copy_(
            self.paged_kv_indptr_cpu_buffer[: num_reqs + 1], non_blocking=True
        )

        # write self.paged_kv_indices inplace
        num_actual_pages = self.paged_kv_indptr.np[num_reqs]
        paged_kv_indices = self.paged_kv_indices.gpu[:num_actual_pages]
        _copy_page_indices_kernel[(num_reqs,)](
            paged_kv_indices,
            block_table_tensor,
            block_table_tensor.stride(0),
            paged_kv_indptr,
            BLOCK_SIZE=1024,
        )

        # write self.paged_kv_last_page_len_cpu inplace
        paged_kv_last_page_len_np = seq_lens_np % page_size
        self.paged_kv_last_page_len.np[:num_reqs] = np.where(
            (paged_kv_last_page_len_np == 0) & (seq_lens_np != 0),
            page_size,
            paged_kv_last_page_len_np,
        )
        self.paged_kv_last_page_len.gpu[:num_reqs].copy_(
            self.paged_kv_last_page_len.cpu[:num_reqs], non_blocking=True
        )
        return paged_kv_indices

    # 中文注释：构建 FlashInfer 注意力元数据。
    # 这是 FlashInferMetadataBuilder 的核心方法，由 GPUModelRunner 在每次
    # 模型前向传播前调用。它将通用的注意力元数据转换为 FlashInfer 核函数
    # 所需的特定格式。
    #
    # 参数：
    #   common_prefix_len: 共享前缀的长度（用于 cascade attention），
    #     > 0 时启用 cascade 模式
    #   common_attn_metadata: GPUModelRunner 构建的通用注意力元数据，
    #     包含 seq_lens、block_table、query_start_loc 等
    #   fast_build: 是否使用快速构建模式（跳过部分校验）
    #
    # 返回：FlashInferMetadata，包含 prefill/decode 路径的完整执行计划
    #
    # 算法流程：
    # Step 1: 决定 prefill/decode 使用的核函数路径（FI 原生 vs TRTLLM）
    # Step 2: 初始化输出元数据结构
    # Step 3: 处理 cascade attention（如果启用）
    # Step 4: 为 prefill 路径准备元数据并调用 plan()
    # Step 5: 为 decode 路径准备元数据并调用 plan()
    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> FlashInferMetadata:
        # 中文注释：获取 batch 基本信息并区分 decode/prefill 请求。
        # split_decodes_and_prefills 根据 query 长度将请求分为两组：
        # - decode 请求（query_len <= reorder_batch_threshold）排在前面
        # - prefill 请求（query_len > reorder_batch_threshold）排在后面
        # require_uniform=True 要求 decode 请求的 query 长度一致（TRTLLM 要求）。
        num_reqs = common_attn_metadata.num_reqs
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(
                common_attn_metadata,
                decode_threshold=self.reorder_batch_threshold,
                require_uniform=True,
            )
        )

        page_size = self.page_size
        max_seq_len = common_attn_metadata.max_seq_len
        seq_lens = common_attn_metadata.seq_lens
        block_table_tensor = common_attn_metadata.block_table_tensor
        qo_indptr = common_attn_metadata.query_start_loc
        qo_indptr_cpu = common_attn_metadata.query_start_loc_cpu

        # Step 1: 决定使用哪种核函数路径。
        # 三种决策维度：
        # - use_cascade: 是否使用 cascade attention（共享前缀长度 > 0）
        # - prefill_use_trtllm: prefill 是否使用 TRTLLM 核函数
        # - decode_use_trtllm: decode 是否使用 TRTLLM 核函数
        # TRTLLM 路径在 SM100+ GPU 上性能更优，但不支持所有场景（如 DCP）。
        # Step 1: Decide which dispatch modes to use:
        # - Cascade attention (distinct mode)
        # - Prefill (FI native or TRTLLM)
        # - Decode (FI native or TRTLLM)
        use_cascade = common_prefix_len > 0
        uses_spec_reorder = self.reorder_batch_threshold > 1
        prefill_use_trtllm = use_trtllm_attention(
            self.num_qo_heads,
            self.num_kv_heads,
            num_prefill_tokens,
            max_seq_len,
            self.dcp_world_size,
            self.cache_dtype,
            self.q_data_type,
            is_prefill=True,
            force_use_trtllm=self.attention_config.use_trtllm_attention,
            has_sinks=self.has_sinks,
            has_spec=uses_spec_reorder,
        )
        decode_use_trtllm = (
            self.use_trtllm_decode_attention and self.dcp_world_size <= 1
        )

        # 中文注释：判断是否所有注意力路径都使用 TRTLLM。
        # 只有当 prefill 和 decode 都使用 TRTLLM 时，才能跳过 CPU 端的
        # paged_kv_indices 计算，因为 TRTLLM 直接使用 GPU 上的 block_tables。
        all_uses_trtllm = (num_prefills == 0 or prefill_use_trtllm) and (
            num_decodes == 0 or decode_use_trtllm
        )

        if not all_uses_trtllm:
            if self.has_sinks:
                raise NotImplementedError(
                    "FlashInfer backend currently does not support attention "
                    "sinks, please use trtllm on blackwell or flash attention "
                    "on earlier GPUs."
                )

            if not self.global_hyperparameters.has_same_window_lefts:
                raise ValueError(
                    "Window left is not the same for all layers. "
                    "One potential fix is to set disable_sliding_window=True"
                )

            assert self.global_hyperparameters.has_same_all_params, (
                "FlashInfer backend currently only supports models in which "
                "all layers share the same values for the following "
                "hyperparameters: `window_left`, `logits_soft_cap`, "
                "`sm_scale`."
            )

            # The q quantization is not supported for non-trtllm attention,
            # fall back to model dtype.
            self.q_data_type = self.model_config.dtype

        # Step 2: 初始化输出元数据。
        # prefill/decode/cascade_wrapper 字段留空，后续根据实际 batch 内容填充。
        # Step 2: Initialize the output metadata
        # Leave prefill/decode/cascade_wrapper empty, to be populated
        # case by case depending on the batch contents and backend selection.
        attn_metadata = FlashInferMetadata(
            num_actual_tokens=num_actual_tokens,
            slot_mapping=common_attn_metadata.slot_mapping,
            q_data_type=self.q_data_type,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            use_cascade=use_cascade,
            prefill=None,
            decode=None,
            cascade_wrapper=None,
        )

        # 中文注释：判断是否需要 CPU 端的 seq_lens。
        # TRTLLM 路径直接使用 GPU 上的张量，不需要 CPU 同步。
        # 只有 DCP、cascade 或原生 FlashInfer 路径才需要 CPU 端数据。
        # Guard access to seq_lens_cpu, which may not always be needed
        # and can be expensive to retrieve in async mode.
        # When all attention (both prefill and decode) uses TRTLLM,
        # seq_lens_cpu is not needed since TRTLLM paths use GPU tensors
        # (block_tables, seq_lens) directly.
        needs_seq_lens_cpu = self.use_dcp or use_cascade or not all_uses_trtllm
        seq_lens_cpu = common_attn_metadata.seq_lens_cpu if needs_seq_lens_cpu else None
        seq_lens_np = seq_lens_cpu.numpy() if seq_lens_cpu is not None else None
        num_blocks_np = (
            (seq_lens_np + (page_size - 1)) // page_size
            if seq_lens_np is not None
            else None
        )

        # 中文注释：DCP 模式下调整 seq_lens。
        # DCP 将 KV cache 按 head 分片，每个 GPU 只存储部分 head。
        # 需要从总 seq_len 中减去当前 batch 的 query 长度（因为这些 token
        # 尚未写入 KV cache），然后按 DCP world size 切分到本地长度。
        # Adjust seq_lens_cpu for DCP
        if self.use_dcp:
            assert seq_lens_cpu is not None
            if num_prefills > 0:
                qo_indptr_prefill_cpu = (
                    qo_indptr_cpu[num_decodes:] - qo_indptr_cpu[num_decodes]
                )
                query_lens_prefill_cpu = (
                    qo_indptr_prefill_cpu[1:] - qo_indptr_prefill_cpu[:-1]
                )
                seq_lens_cpu[num_decodes:] = (
                    seq_lens_cpu[num_decodes:] - query_lens_prefill_cpu
                )

            seq_lens_cpu = get_dcp_local_seq_lens(
                seq_lens_cpu,
                self.dcp_world_size,
                self.dcp_rank,
                self.dcp_kv_cache_interleave_size,
            )

        # 中文注释：cascade attention 模式下调整 page 数量。
        # 共享前缀的 page 会被单独处理（作为级联的第一级），
        # 因此从每个请求的 page 数中减去共享前缀占用的 page 数。
        # Adjust num_block_np for cascade attention
        if use_cascade:
            assert num_blocks_np is not None
            assert common_prefix_len % page_size == 0
            num_common_kv_blocks = common_prefix_len // page_size
            num_blocks_np -= num_common_kv_blocks

        # 中文注释：计算 paged_kv_indices（仅 FlashInfer 原生路径需要）。
        # TRTLLM 路径直接使用 GPU 上的 block_tables，不需要这个中间结构。
        # Compute paged_kv_indices if necessary
        # paged_kv_indices is only needed for FlashInfer native paths;
        # TRTLLM paths use block_tables directly on GPU.
        needs_paged_kv_indices = use_cascade or not all_uses_trtllm
        if needs_paged_kv_indices:
            assert num_blocks_np is not None
            assert seq_lens_np is not None
            paged_kv_indices = self._compute_flashinfer_kv_metadata(
                num_blocks_np,
                seq_lens_np,
                block_table_tensor,
                num_reqs,
                page_size,
            )
        else:
            paged_kv_indices = None

        # Early-out for cascade attention
        if use_cascade:
            assert num_blocks_np is not None
            # Grab the blocks of the shared prefix from the first request.
            num_common_kv_blocks = common_prefix_len // page_size

            # Create CPU versions directly for cascade (no GPU versions needed)
            shared_qo_indptr_cpu = torch.tensor(
                [0, num_actual_tokens], dtype=torch.int32, device="cpu"
            )
            shared_kv_page_indptr_cpu = torch.tensor(
                [0, num_common_kv_blocks], dtype=torch.int32, device="cpu"
            )
            shared_kv_page_indices_cpu = block_table_tensor[0, :num_common_kv_blocks]
            shared_kv_last_page_len_cpu = torch.tensor(
                [page_size], dtype=torch.int32, device="cpu"
            )

            # Remove the blocks of the shared prefix from all requests.
            block_table_tensor = block_table_tensor[:, num_common_kv_blocks:]
            num_blocks_np -= num_common_kv_blocks

            assert paged_kv_indices is not None
            paged_kv_indptr_cpu = self.paged_kv_indptr.cpu[: 1 + num_reqs]
            paged_kv_last_page_len_cpu = self.paged_kv_last_page_len.cpu[:num_reqs]

            attn_metadata.cascade_wrapper = self._get_cascade_wrapper()
            attn_metadata.cascade_wrapper.plan(
                qo_indptr_arr=[shared_qo_indptr_cpu, qo_indptr_cpu],
                paged_kv_indptr_arr=[shared_kv_page_indptr_cpu, paged_kv_indptr_cpu],
                paged_kv_indices_arr=[shared_kv_page_indices_cpu, paged_kv_indices],
                paged_kv_last_page_len=[
                    shared_kv_last_page_len_cpu,
                    paged_kv_last_page_len_cpu,
                ],
                num_qo_heads=self.num_qo_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                page_size=self.page_size,
                causal=True,
                sm_scale=self.sm_scale,
                window_left=self.window_left,
                logits_soft_cap=self.logits_soft_cap,
                q_data_type=self.q_data_type,
                kv_data_type=self.kv_cache_dtype,
            )
            return attn_metadata

        # Step 3: Handle prefill and decode pathways case by case
        ## PREFILL PATHWAY
        if num_prefills > 0:
            # Slices for shared prefill metadata
            prefill_start = num_decodes
            qo_indptr_prefill_cpu = (
                qo_indptr_cpu[prefill_start:] - qo_indptr_cpu[prefill_start]
            )
            assert qo_indptr_prefill_cpu.shape[0] == num_prefills + 1

            if prefill_use_trtllm:
                # Create GPU versions
                qo_indptr_prefill_gpu = (
                    qo_indptr[prefill_start:] - qo_indptr[prefill_start]
                )
                # Compute cum_seq_lens_kv on GPU to avoid CPU sync.
                # This is the cumulative sum of the number of KV cache
                # blocks per prefill request.
                prefill_seq_lens = seq_lens[prefill_start:]
                num_blocks_per_req = (prefill_seq_lens + page_size - 1) // page_size
                paged_kv_indptr_prefill_gpu = self.paged_kv_indptr.gpu[
                    prefill_start : num_reqs + 1
                ]
                # Assign to slice to avoid cpu sync.
                paged_kv_indptr_prefill_gpu[:1] = 0
                torch.cumsum(
                    num_blocks_per_req,
                    dim=0,
                    out=paged_kv_indptr_prefill_gpu[1:],
                )
                # Compute max_q_len for prefill requests
                query_lens_prefill_cpu = (
                    qo_indptr_prefill_cpu[1:] - qo_indptr_prefill_cpu[:-1]
                )
                max_q_len_prefill = int(query_lens_prefill_cpu.max().item())
                attn_metadata.prefill = TRTLLMPrefill(
                    block_tables=block_table_tensor[prefill_start:],
                    seq_lens=seq_lens[prefill_start:],
                    cum_seq_lens_q=qo_indptr_prefill_gpu,
                    cum_seq_lens_kv=paged_kv_indptr_prefill_gpu,
                    max_q_len=max_q_len_prefill,
                    max_seq_len=max_seq_len,
                )
            else:
                prefill_wrapper = self._get_prefill_wrapper()
                # Slicing CPU buffers that are only needed for FI native prefills
                paged_kv_last_page_len_prefill_cpu = self.paged_kv_last_page_len.cpu[
                    prefill_start:num_reqs
                ]
                assert paged_kv_last_page_len_prefill_cpu.shape[0] == num_prefills
                paged_kv_indptr_prefill_cpu = self.paged_kv_indptr.cpu[
                    prefill_start : num_reqs + 1
                ]
                assert paged_kv_indptr_prefill_cpu.shape[0] == num_prefills + 1
                if self.use_dcp:
                    assert isinstance(prefill_wrapper, BatchDCPPrefillWrapper)
                    prefill_wrapper.plan(
                        qo_indptr_cpu=qo_indptr_prefill_cpu,
                        paged_kv_indptr_cpu=paged_kv_indptr_prefill_cpu,
                        paged_kv_indices=paged_kv_indices,
                        paged_kv_last_page_len_cpu=paged_kv_last_page_len_prefill_cpu,
                        page_size=self.page_size,
                        num_qo_heads=self.num_qo_heads,
                        dcp_world_size=self.dcp_world_size,
                        num_kv_heads=self.num_kv_heads,
                        head_dim=self.head_dim,
                        sm_scale=self.sm_scale,
                        window_left=self.window_left,
                        logits_soft_cap=self.logits_soft_cap,
                        q_data_type=self.q_data_type,
                        kv_cache_dtype=self.kv_cache_dtype,
                        prefill_fixed_split_size=self.prefill_fixed_split_size,
                        disable_split_kv=self.disable_split_kv,
                    )
                else:
                    assert isinstance(
                        prefill_wrapper,
                        BatchPrefillWithPagedKVCacheWrapper,
                    )
                    # NVFP4 trtllm kernel only supports FP8 output;
                    # use FP8 o_data_type so the wrapper matches the
                    # FP8 output buffer allocated in forward().
                    o_dtype = (
                        FP8_DTYPE if self.is_kvcache_nvfp4 else self.model_config.dtype
                    )
                    prefill_wrapper.plan(
                        qo_indptr=qo_indptr_prefill_cpu,
                        paged_kv_indptr=paged_kv_indptr_prefill_cpu,
                        paged_kv_indices=paged_kv_indices,
                        paged_kv_last_page_len=paged_kv_last_page_len_prefill_cpu,
                        num_qo_heads=self.num_qo_heads,
                        num_kv_heads=self.num_kv_heads,
                        head_dim_qk=self.head_dim,
                        page_size=self.page_size,
                        causal=True,
                        sm_scale=self.sm_scale,
                        window_left=self.window_left,
                        logits_soft_cap=self.logits_soft_cap,
                        q_data_type=self.q_data_type,
                        kv_data_type=self.kv_cache_dtype,
                        o_data_type=o_dtype,
                        fixed_split_size=self.prefill_fixed_split_size,
                        disable_split_kv=self.disable_split_kv,
                    )
                attn_metadata.prefill = FIPrefill(wrapper=prefill_wrapper)

        ## DECODE PATHWAY
        if num_decodes > 0:
            if decode_use_trtllm:
                assert num_decode_tokens % num_decodes == 0, (
                    "TRTLLM decode requires uniform query lengths per request. "
                    f"Got {num_decode_tokens=} and {num_decodes=}."
                )
                attn_metadata.decode = TRTLLMDecode(
                    block_tables=block_table_tensor[:num_decodes],
                    seq_lens=seq_lens[:num_decodes],
                    max_seq_len=max_seq_len,
                )
            else:
                assert seq_lens_cpu is not None
                pure_decode = num_prefills == 0
                use_cudagraph = (
                    self.enable_cuda_graph
                    and pure_decode
                    and num_decode_tokens <= self._decode_cudagraph_max_bs
                )
                num_input_tokens = num_decode_tokens

                decode_wrapper = self._get_decode_wrapper(
                    num_input_tokens, use_cudagraph
                )
                # Use the persistent buffer with padding length,
                # instead of the same address but chunked version
                # in atten_metadata when using cudagraph.
                # NVFP4 trtllm kernel only supports FP8 output;
                # use FP8 o_data_type so the wrapper matches the
                # FP8 output buffer allocated in forward().
                o_dtype = (
                    FP8_DTYPE if self.is_kvcache_nvfp4 else self.model_config.dtype
                )
                fast_plan_decode(
                    decode_wrapper,
                    indptr_cpu=self.paged_kv_indptr.cpu[: num_input_tokens + 1],
                    indices=paged_kv_indices,
                    last_page_len_cpu=self.paged_kv_last_page_len.cpu[
                        :num_input_tokens
                    ],
                    num_qo_heads=self.num_qo_heads * self.dcp_world_size,
                    num_kv_heads=self.num_kv_heads,
                    head_dim=self.head_dim,
                    page_size=self.page_size,
                    # Disable flashinfer's pos encoding and use vllm's rope.
                    pos_encoding_mode="NONE",
                    sm_scale=self.sm_scale,
                    window_left=self.window_left,
                    logits_soft_cap=self.logits_soft_cap,
                    q_data_type=self.q_data_type,
                    kv_data_type=self.kv_cache_dtype,
                    o_data_type=o_dtype,
                    fixed_split_size=self.decode_fixed_split_size,
                    disable_split_kv=self.disable_split_kv,
                )
                attn_metadata.decode = FIDecode(wrapper=decode_wrapper)
        return attn_metadata

    def use_cascade_attention(self, *args, **kwargs) -> bool:
        # 中文注释：判断是否使用 cascade attention（级联注意力）。
        # cascade attention 将 prefix 部分共享的 KV 计算与每个请求独有的部分分开处理，
        # 可以在多请求共享相同 prefix 时减少重复计算。
        # 当前由于 cascade wrapper 不支持 KV cache dtype 与 query dtype 不同的场景，
        # 以及该功能尚未完全调通，所以始终返回 False 禁用。
        if self.kv_cache_spec.dtype != self.vllm_config.model_config.dtype:
            # TODO: The cascade wrapper currently does not support setting
            # kv cache dtype to something different from query dtype.
            return False
        # TODO: Cascade attention doesn't work, disable it for now
        # return use_cascade_attention(*args, **kwargs)
        return False


# 中文注释：FlashInferImpl 是基于 FlashInfer 库的实际注意力计算实现类。
# 它继承自 AttentionImpl，负责在 forward 阶段执行 prefill 和 decode 两种注意力计算路径。
# 支持三种注意力后端：
#   1. FlashInfer 原生路径（FIPrefill / FIDecode）
#   2. TRT-LLM 路径（TRTLLMPrefill / TRTLLMDecode），支持 FP8/FP4 量化输出融合
#   3. Cascade attention 路径（级联注意力，当前已禁用）
# 还支持 DCP（Decode Context Parallel）分布式 decode 注意力。
class FlashInferImpl(AttentionImpl):
    # 中文注释：标记该注意力实现可以在 decode 阶段返回 log-sum-exp（LSE）值。
    # LSE 用于 cascade attention 或 DCP 模式下对多个注意力输出做加权合并。
    can_return_lse_for_decode: bool = True

    # 中文注释：初始化 FlashInferImpl 注意力计算模块。
    # 参数说明：
    #   - num_heads: Q 头数（query heads），GQA 模式下通常大于 num_kv_heads
    #   - head_size: 每个头的维度（如 128）
    #   - scale: 缩放因子，通常为 1/sqrt(head_size)
    #   - num_kv_heads: KV 头数，GQA 模式下小于 num_heads
    #   - alibi_slopes: ALiBi 位置编码的斜率，None 表示不使用 ALiBi
    #   - sliding_window: 滑动窗口注意力的窗口大小，None 表示全局注意力
    #   - kv_cache_dtype: KV cache 的数据类型字符串（如 "auto"、"fp8"、"nvfp4" 等）
    #   - logits_soft_cap: logits 软截断值（Gemma2 等模型使用）
    #   - attn_type: 注意力类型，当前仅支持 DECODER
    #   - kv_sharing_target_layer_name: KV cache 共享的目标层名称（跨层共享 KV）
    #   - sinks: 注意力 sink token 的嵌入向量（StreamingLLM 等场景使用）
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
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        # 中文注释：滑动窗口注意力的范围设置。
        # FlashInfer 使用 (left, right) 元组表示窗口，right=0 表示只看左侧 token。
        # (-1, -1) 表示不使用滑动窗口（全局注意力）。
        if sliding_window is None:
            self.sliding_window = (-1, -1)
        else:
            self.sliding_window = (sliding_window - 1, 0)
        # 中文注释：window_left 是 FlashInfer decode wrapper 需要的参数，
        # 表示每个 query token 最多能看到左侧多少个 KV token。-1 表示无限制。
        self.window_left = (
            self.sliding_window[0] if self.sliding_window is not None else -1
        )
        self.kv_cache_dtype = kv_cache_dtype
        # 中文注释：标记 KV cache 是否使用 NVFP4 量化格式。
        # NVFP4 是 NVIDIA 的 FP4 量化格式，将数据和 scale factor 打包在一起。
        self.is_kvcache_nvfp4 = kv_cache_dtype == "nvfp4"
        self.fp4_data_dim = head_size // 2 if self.is_kvcache_nvfp4 else 0
        self.logits_soft_cap = logits_soft_cap
        # 中文注释：KV cache 共享的目标层名称。
        # 如果非 None，表示该层不自己写入 KV cache，而是复用目标层的 KV cache。
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name

        # 中文注释：GQA（Grouped Query Attention）中每个 KV 头对应的 Q 头数。
        # 例如 num_heads=32, num_kv_heads=8 时，num_queries_per_kv=4，
        # 表示每 4 个 Q 头共享 1 个 KV 头。
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "Encoder self-attention and "
                "encoder/decoder cross-attention "
                "are not implemented for "
                "FlashInferImpl"
            )

        # 中文注释：注意力 sink tokens 的嵌入向量（用于 StreamingLLM 等场景）。
        # sink tokens 是序列开头的若干 token，即使在滑动窗口注意力中也会被保留，
        # 以防止注意力分布退化。形状为 [num_heads, head_size]。
        self.sinks: torch.Tensor | None = None
        if sinks is not None:
            if sinks.shape[0] != num_heads:
                raise ValueError(
                    "Sinks must have the same number of heads as the number of "
                    f"heads in the layer. Expected {num_heads}, but got "
                    f"{sinks.shape[0]}."
                )
            self.sinks = sinks

        # 中文注释：检查当前硬件和头数配置是否支持 TRT-LLM 注意力核函数。
        # TRT-LLM 注意力支持 FP8/FP4 输出量化融合，可减少显存带宽占用。
        self.support_trtllm_attn = can_use_trtllm_attention(num_heads, num_kv_heads)
        vllm_config = get_current_vllm_config_or_none()
        # 中文注释：是否支持将 query 也量化为 FP8 输入，用于 TRT-LLM 注意力核函数。
        # 量化 query 可以进一步提升性能，但需要在配置中启用。
        self.supports_quant_query_input = (
            self.support_trtllm_attn
            and vllm_config is not None
            and not vllm_config.attention_config.disable_flashinfer_q_quantization
        )
        # 中文注释：bmm1_scale 和 bmm2_scale 分别是 attention 计算中
        # Q*K^T 和 attn*V 两步矩阵乘法的缩放因子。
        # 对于量化 KV cache，需要在基础 scale 上乘以量化 scale。
        self.bmm1_scale: float | None = None
        self.bmm2_scale: float | None = None
        # 中文注释：输出量化 scale factor，仅在 TRT-LLM 路径中 attention+quant 融合时使用。
        self.o_sf_scale: float | None = None

        # Pre-allocated FP8 output buffer for NVFP4 without fused output quant.
        # 中文注释：当使用 NVFP4 KV cache 但没有启用输出量化融合时，
        # TRT-LLM 核函数只能输出 FP8 结果，需要先写入预分配的 FP8 缓冲区，
        # 再反量化为目标 dtype。此处预分配该缓冲区以避免动态分配开销。
        if self.is_kvcache_nvfp4 and vllm_config is not None:
            max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens
            self._nvfp4_fp8_out = torch.empty(
                (max_num_tokens, num_heads, head_size),
                dtype=FP8_DTYPE,
                device="cuda",
            )
        else:
            self._nvfp4_fp8_out = None

        # 中文注释：配置 DCP（Decode Context Parallel）模式下的注意力合并策略。
        # DCP 将 decode 阶段的 KV 分散到多个 GPU 上并行计算，然后合并结果。
        # "a2a"（all-to-all）通信后端使用 dcp_a2a_lse_reduce 进行高效合并；
        # 默认使用 cp_lse_ag_out_rs（基于 all-gather + reduce-scatter）。
        # 合并时需要使用 LSE（log-sum-exp）进行数值稳定的加权平均。
        dcp_a2a = (
            vllm_config is not None
            and vllm_config.parallel_config.decode_context_parallel_size > 1
            and vllm_config.parallel_config.dcp_comm_backend == "a2a"
        )
        if dcp_a2a:
            self.dcp_combine = partial(dcp_a2a_lse_reduce, is_lse_base_on_e=False)
        else:
            self.dcp_combine = partial(cp_lse_ag_out_rs, is_lse_base_on_e=False)

    # 中文注释：判断是否支持 attention 输出与量化的融合（fused output quantization）。
    # 当支持时，注意力计算的输出直接以量化格式（FP8/FP4）写出，
    # 省去了先输出 FP16/BF16 再量化的过程，减少显存带宽消耗。
    # 需要同时满足：TRT-LLM 注意力可用、KV cache 已量化、输出量化方案匹配。
    def fused_output_quant_supported(self, quant_key: QuantKey):
        return (
            self.support_trtllm_attn
            and is_quantized_kv_cache(self.kv_cache_dtype)
            and quant_key in (kFp8StaticTensorSym, kNvfp4Dynamic)
        )

    # FlashInfer requires attention sinks to be float32
    # 中文注释：模型权重加载完成后的后处理。
    # FlashInfer 的 sink tokens 必须是 float32 精度以保证数值稳定性。
    def process_weights_after_loading(self, act_dtype: torch.dtype):
        if self.sinks is not None and self.sinks.dtype != torch.float32:
            self.sinks = self.sinks.to(torch.float32)

    # 中文注释：FlashInfer 注意力的前向计算方法。
    # 整体流程：
    #   1. 校验 query dtype 与元数据一致
    #   2. 初始化/缓存 bmm1_scale、bmm2_scale（量化场景下需要包含量化 scale）
    #   3. 处理 attention+quant 融合的 scale 参数
    #   4. 裁剪 padding（CUDA graph 场景中输入可能被 pad）
    #   5. 将 KV cache 排列为 HND 布局（FlashInfer 要求）
    #   6. 分别执行 prefill 和 decode 路径的注意力计算
    #   7. 返回注意力输出（带 padding 的原始形状）
    #
    # 在 vLLM V1 的 continuous batching 中，一个 batch 里 decode 请求排在前面，
    # prefill 请求排在后面（由 SchedulerOutput 中的排序保证）。
    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashInferMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with FlashInfer.

        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache: KV cache tensor with different possible shapes:
                - NHD: [num_blocks, 2, block_size, num_kv_heads, head_size]
                - HND: [num_blocks, 2, num_kv_heads, block_size, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        if attn_metadata is None:
            # Profiling run.
            # 中文注释：profiling 阶段没有真实请求，直接返回零填充的输出。
            return output.fill_(0)

        # Ensure query dtype matches the expected dtype from attention metadata
        # 中文注释：校验 query 的 dtype 与注意力元数据中记录的一致。
        # 当使用 FP8 query 量化时，query 的 dtype 会是 FP8 而非 BF16/FP16。
        assert attn_metadata.q_data_type == query.dtype, (
            f"Query dtype mismatch: expected {attn_metadata.q_data_type}, "
            f"got {query.dtype}"
        )

        # 中文注释：初始化 bmm1_scale（Q*K^T 的缩放因子）。
        # 基础值是 attention scale（1/sqrt(head_size)）。
        # 当 KV cache 使用量化格式时，需要额外乘以 Q 和 K 的量化反量化 scale，
        # 以补偿量化引入的数值范围变化。
        if self.bmm1_scale is None:
            self.bmm1_scale = self.scale
            if is_quantized_kv_cache(self.kv_cache_dtype):
                self.bmm1_scale *= layer._q_scale_float * layer._k_scale_float

        # 中文注释：初始化 bmm2_scale（attn*V 的缩放因子）。
        # 默认为 1.0，当 KV cache 量化时需要乘以 V 的量化 scale。
        if self.bmm2_scale is None:
            self.bmm2_scale = 1.0
            if is_quantized_kv_cache(self.kv_cache_dtype):
                self.bmm2_scale *= layer._v_scale_float

        # 中文注释：判断 prefill 和 decode 是否使用 TRT-LLM 注意力核函数。
        # TRT-LLM 路径支持 FP8/FP4 输出量化融合，性能更优但要求更严格。
        prefill_use_trtllm = isinstance(attn_metadata.prefill, TRTLLMPrefill)
        decode_use_trtllm = isinstance(attn_metadata.decode, TRTLLMDecode)

        # The attn+quant fusion happens when output_scale is provided.
        # 中文注释：当 output_scale 非 None 时，表示 attention 输出需要直接量化。
        # 这要求 query 必须是 FP8 格式，且 prefill/decode 都使用 TRT-LLM 路径。
        # output_scale 用于 FP8 的静态/动态量化；output_block_scale 用于 FP4 的块量化。
        if output_scale is None:
            assert output_block_scale is None, (
                "output_block_scale is not supported when fusion has not happened"
            )
        else:
            assert attn_metadata.q_data_type == FP8_DTYPE, (
                "Query must be FP8 when attn+quant fusion happened."
            )
            assert (attn_metadata.num_prefills == 0 or prefill_use_trtllm) and (
                attn_metadata.num_decodes == 0 or decode_use_trtllm
            ), "Must use TRT-LLM attn"

            if output.dtype == FP8_DTYPE:
                assert output_block_scale is None, (
                    "output_block_scale should not be provided for fp8 output"
                )
            elif output.dtype == FP4_DTYPE:
                assert output_block_scale is not None, (
                    "output_block_scale is required for nvfp4 output"
                )
            else:
                raise ValueError(f"Unsupported output dtype: {output.dtype}")

            # TRTLLM attn kernel requires to scale to pass as a host scalar,
            # store the o scale as a host scalar in warmup run with cuda graph
            # not enabled
            if layer._o_scale_float is None:
                layer._o_scale_float = output_scale.cpu().item()
                if output.dtype == FP8_DTYPE:
                    self.bmm2_scale = self.bmm2_scale / layer._o_scale_float
                elif output.dtype == FP4_DTYPE:
                    self.o_sf_scale = layer._o_scale_float

        # IMPORTANT!
        # NOTE(woosuk): With piece-wise CUDA graphs, this method is executed in
        # eager-mode PyTorch. Thus, we need to be careful about any CPU overhead
        # in this method. For example, `view` and `slice` (or `[:n]`) operations
        # are surprisingly slow even in the case they do not invoke any GPU ops.
        # Minimize the PyTorch ops in this method as much as possible.
        # Whenever making a change in this method, please benchmark the
        # performance to make sure it does not introduce any overhead.

        # 中文注释：num_actual_tokens 是本轮 batch 中实际有效的 token 总数。
        # CUDA graph 场景中输入可能被 pad 到固定大小，实际 token 数可能小于 tensor 尺寸。
        num_actual_tokens = attn_metadata.num_actual_tokens

        # FlashInfer treats uint8 KV cache as NVFP4. vLLM stores FP8 KV cache
        # as uint8 bytes, so pass FP8 caches with their logical dtype.
        # 中文注释：FlashInfer 会将 uint8 类型的 KV cache 视为 NVFP4。
        # 但 vLLM 中 FP8 KV cache 也以 uint8 存储（因为 PyTorch 的 FP8 类型支持有限），
        # 所以需要将 uint8 重新 view 为正确的 FP8 dtype，避免 FlashInfer 误判。
        if not self.is_kvcache_nvfp4 and kv_cache.dtype == torch.uint8:
            fp8_view_dtype = None
            if self.kv_cache_dtype in ("fp8", "fp8_e4m3", torch.float8_e4m3fn):
                fp8_view_dtype = torch.float8_e4m3fn
            elif self.kv_cache_dtype in ("fp8_e5m2", torch.float8_e5m2):
                fp8_view_dtype = torch.float8_e5m2
            if fp8_view_dtype is not None:
                kv_cache = kv_cache.view(fp8_view_dtype)

        # Inputs and outputs may be padded for CUDA graphs
        # 中文注释：裁剪到实际 token 数，去掉 CUDA graph 引入的 padding。
        # output_padded 保留原始（带 padding 的）tensor 引用，最终返回时需要恢复为该形状。
        query = query[:num_actual_tokens]
        key = key[:num_actual_tokens]
        value = value[:num_actual_tokens]
        output_padded = output
        output = output[:num_actual_tokens]

        # 中文注释：级联注意力路径（当前已禁用，极少触发）。
        # cascade attention 将共享 prefix 的 KV 和每个请求独有的 KV 分开计算，
        # 适用于多请求共享相同长前缀的场景。直接返回并跳过后续的 prefill/decode 路径。
        if attn_metadata.use_cascade:
            # Cascade attention (rare case).
            assert attn_metadata.cascade_wrapper is not None
            output.copy_(attn_metadata.cascade_wrapper.run(query, kv_cache))
            return output

        # When using spec decoding, num_decodes can be < num_decode_tokens
        # because some decode requests may have more than one query token.
        # 中文注释：区分 batch 中的 decode token 数和 prefill token 数。
        # decode token 是每个请求新生成的 1 个 token（投机解码时可能是多个）；
        # prefill token 是 prompt 中待计算的新 token。
        # 在 V1 的 continuous batching 中，decode 请求排在 batch 前面，prefill 排在后面。
        num_decode_tokens = attn_metadata.num_decode_tokens
        num_prefill_tokens = attn_metadata.num_prefill_tokens

        # 中文注释：将 KV cache 重排为 FlashInfer 要求的 HND 布局。
        # HND = [num_blocks, 2, num_kv_heads, block_size, head_size]
        # 其中 dim=1 的 2 分别对应 K 和 V。FlashInfer 要求 HND 布局，
        # 而 vLLM 内部存储可能是 NHD 布局，通过 permute 转换（零拷贝，只改变 stride）。
        stride_order = FlashInferBackend.get_kv_cache_stride_order()
        kv_cache_permute = kv_cache.permute(*stride_order)  # HND and contiguous
        # Fix degenerate strides on any size-1 dimension (e.g. num_kv_heads=1
        # with TP=8).  PyTorch permits non-canonical strides on size-1 dims;
        # CUDA TMA requires ≥16-byte alignment on all non-outermost strides.
        # canonicalize_singleton_dim_strides patches metadata via as_strided —
        # zero-copy.  See vllm.utils.torch_utils.
        # 中文注释：修复 size-1 维度上的退化 stride。
        # PyTorch 允许 size-1 维度的 stride 为任意值，但 CUDA TMA 要求
        # 非最外层维度的 stride 至少 16 字节对齐。
        # canonicalize_singleton_dim_strides 通过 as_strided 零拷贝修复。
        fixed = canonicalize_singleton_dim_strides(kv_cache_permute)
        if fixed is not kv_cache_permute:
            logger.debug(
                "Canonicalized degenerate KV cache strides (FlashInfer): "
                "shape=%s, strides before=%s, strides after=%s",
                kv_cache_permute.shape,
                kv_cache_permute.stride(),
                fixed.stride(),
            )
        kv_cache_permute = fixed

        # For NVFP4, the kv_cache last dim is full_dim (data + scale packed).
        # Split into correctly-strided data and scale views.
        # 中文注释：NVFP4 量化格式下，KV cache 的最后一个维度同时包含了 FP4 数据和 scale factor。
        # 这里将其拆分为正确 stride 的 data 视图和 scale 视图，
        # 以便分别传给 TRT-LLM 核函数的 kv_cache 和 kv_cache_sf 参数。
        nvfp4_kv_data = None
        nvfp4_kv_block_scales = None
        if self.is_kvcache_nvfp4:
            nvfp4_kv_data, nvfp4_kv_block_scales = nvfp4_kv_cache_split_views(
                kv_cache_permute
            )

        # 中文注释：判断是否使用 DCP（Decode Context Parallel）分布式 decode。
        # DCP 将 KV cache 分散到多个 GPU 上，每个 GPU 只持有部分 KV，
        # 通过 all-gather 合并 query 或 all-to-all 合并注意力结果。
        use_dcp = self.dcp_world_size > 1

        # Regular attention (common case).
        # Decodes are at the front and prefills are at the back.
        # 中文注释：prefill 注意力计算路径。
        # 在 V1 的 continuous batching 中，batch 内 decode 请求在前、prefill 在后，
        # 所以 prefill 的 query 从 num_decode_tokens 位置开始。
        # prefill 有两种子路径：
        #   (a) FlashInfer 原生路径：使用 BatchPrefillWithPagedKVCacheWrapper
        #   (b) TRT-LLM 路径：使用 trtllm_batch_context_with_kv_cache，
        #       支持 FP8/FP4 输出量化融合
        if num_prefill_tokens > 0:
            prefill_query = query[num_decode_tokens:]
            assert prefill_query.shape[0] == num_prefill_tokens

            if not prefill_use_trtllm:
                # 中文注释：FlashInfer 原生 prefill 路径。
                # 使用 FlashInfer 的 BatchPrefillWithPagedKVCacheWrapper 计算注意力。
                # 该 wrapper 已在 MetadataBuilder 中通过 plan() 方法预设好
                # block table、seq lens 等元数据。
                assert isinstance(attn_metadata.prefill, FIPrefill)
                prefill_wrapper = attn_metadata.prefill.wrapper
                assert prefill_wrapper is not None
                if use_dcp:
                    # 中文注释：DCP 模式下的 prefill 注意力。
                    # 使用 BatchDCPPrefillWrapper，将 context 和 new_tokens 分开处理。
                    # context 部分是非因果的（可以看到所有 token），new_tokens 部分是因果的。
                    assert isinstance(prefill_wrapper, BatchDCPPrefillWrapper)
                    assert prefill_wrapper._context._window_left == self.window_left
                    assert prefill_wrapper._context._logits_soft_cap == (
                        self.logits_soft_cap or 0.0
                    )
                    assert prefill_wrapper._context._sm_scale == self.scale
                    assert not prefill_wrapper._context._causal
                    assert prefill_wrapper._new_tokens._window_left == self.window_left
                    assert prefill_wrapper._new_tokens._logits_soft_cap == (
                        self.logits_soft_cap or 0.0
                    )
                    assert prefill_wrapper._new_tokens._sm_scale == self.scale
                    assert prefill_wrapper._new_tokens._causal

                    prefill_wrapper.run(
                        layer,
                        prefill_query,
                        kv_cache_permute,
                        key[num_decode_tokens:],
                        value[num_decode_tokens:],
                        out=output[num_decode_tokens:],
                    )
                else:
                    # 中文注释：标准 FlashInfer prefill 路径（非 DCP）。
                    # 断言检查 wrapper 的参数与当前层一致（窗口大小、soft cap、scale 等），
                    # 确保 plan() 阶段设置的参数没有被篡改。
                    assert isinstance(
                        prefill_wrapper, BatchPrefillWithPagedKVCacheWrapper
                    )
                    assert prefill_wrapper._window_left == self.window_left
                    assert prefill_wrapper._logits_soft_cap == (
                        self.logits_soft_cap or 0.0
                    )
                    assert prefill_wrapper._sm_scale == self.scale
                    assert prefill_wrapper._causal

                    # 中文注释：NVFP4 场景下使用拆分后的 data 视图作为 kv_cache，
                    # 并传入对应的 block scale factor。
                    if self.is_kvcache_nvfp4:
                        kv_cache_permute = nvfp4_kv_data
                    kv_cache_sf = (
                        nvfp4_kv_block_scales if self.is_kvcache_nvfp4 else None
                    )

                    # NVFP4 trtllm kernel only supports FP8 output.
                    # Use a pre-allocated FP8 buffer and dequantize
                    # afterwards.
                    # 中文注释：NVFP4 的 FlashInfer 核函数只能输出 FP8 结果。
                    # 如果目标输出不是 FP8，需要先写入预分配的 FP8 缓冲区，
                    # 然后再反量化为目标 dtype（如 BF16/FP16）。
                    needs_fp8_out_prefill = (
                        self.is_kvcache_nvfp4 and output.dtype != FP8_DTYPE
                    )
                    if needs_fp8_out_prefill:
                        out_prefill = self._nvfp4_fp8_out[:num_prefill_tokens]
                    else:
                        out_prefill = output[num_decode_tokens:]

                    prefill_wrapper.run(
                        prefill_query,
                        kv_cache_permute,
                        k_scale=layer._k_scale_float,
                        v_scale=layer._v_scale_float,
                        out=out_prefill,
                        kv_cache_sf=kv_cache_sf,
                    )

                    if needs_fp8_out_prefill:
                        output[
                            num_decode_tokens : num_decode_tokens + num_prefill_tokens
                        ].copy_(out_prefill.to(output.dtype))
            else:
                # 中文注释：TRT-LLM prefill 注意力路径。
                # 使用 TRT-LLM 的 batch context attention 核函数，
                # 支持 FP8/FP4 输出量化融合以及 FP4 KV cache。
                # TRT-LLM 核函数对内存布局有严格要求，需要确保 query 是连续的。
                assert isinstance(attn_metadata.prefill, TRTLLMPrefill)
                # prefill_query may be non-contiguous or have degenerate strides
                # on size=1 dims. contiguous() ensures memory layout; then
                # canonicalize_singleton_dim_strides fixes any remaining
                # degenerate strides on size=1 dims for TMA alignment.
                prefill_query = prefill_query.contiguous()
                prefill_query = canonicalize_singleton_dim_strides(prefill_query)
                workspace_buffer = _get_trtllm_gen_workspace_buffer()
                # 中文注释：TRT-LLM 路径使用自己的 block_tables 和 seq_lens，
                # 这些在 MetadataBuilder 中专门为 TRT-LLM 准备。
                block_tables_prefill = attn_metadata.prefill.block_tables
                seq_lens_prefill = attn_metadata.prefill.seq_lens

                # This path needs to be enabled with VLLM_KV_CACHE_LAYOUT = HND
                # 中文注释：TRT-LLM 核函数要求所有输入 tensor 在内存中严格连续，
                # 且 KV cache 布局必须为 HND。这些 assert 在调试阶段帮助及早发现布局问题。
                assert get_kv_cache_layout() == "HND"
                assert is_strictly_contiguous(prefill_query)
                assert is_strictly_contiguous(workspace_buffer)
                assert is_strictly_contiguous(block_tables_prefill)
                assert is_strictly_contiguous(seq_lens_prefill)

                # 中文注释：根据输出 dtype 选择合适的输出 tensor。
                # FP4 输出需要使用 FP4Tensor 包装，包含 data 和 block scale 两部分。
                if output.dtype == FP4_DTYPE:
                    assert self.o_sf_scale is not None
                    out = FP4Tensor(
                        data=output[num_decode_tokens:],
                        scale=output_block_scale,
                        scale_start_index=num_decode_tokens,
                        original_shape=prefill_query.shape,
                    )
                else:
                    assert self.o_sf_scale is None
                    out = output[num_decode_tokens:]

                # NVFP4 trtllm kernel only supports FP8 output.
                # Use a pre-allocated FP8 buffer and dequantize afterwards.
                # 中文注释：NVFP4 核函数只能输出 FP8。如果目标不是 FP8，
                # 先写入预分配的 FP8 缓冲区，稍后再反量化。
                needs_fp8_out = self.is_kvcache_nvfp4 and output.dtype != FP8_DTYPE
                if needs_fp8_out:
                    out = self._nvfp4_fp8_out[:num_prefill_tokens]

                # 中文注释：根据 KV cache 类型选择 prefill 注意力的输入数据。
                # NVFP4：使用拆分后的 data/scale 视图，要求 query 为 FP8。
                # FP8 KV 但非 FP8 query：需要反量化 KV cache 为 BF16 来做 prefill，
                # 因为 TRT-LLM prefill 不支持 BF16 query + FP8 KV 的组合。
                # 其他情况：直接使用原始 kv_cache_permute。
                prefill_kv_block_scales = None
                if self.is_kvcache_nvfp4:
                    # NVFP4 trtllm-gen kernel requires FP8 query.
                    assert attn_metadata.q_data_type == FP8_DTYPE, (
                        "NVFP4 KV cache requires FP8 quantized queries for "
                        "trtllm-gen prefill. Set "
                        "disable_flashinfer_q_quantization=False."
                    )
                    mock_kv_cache = nvfp4_kv_data
                    mock_block_table = block_tables_prefill
                    prefill_kv_block_scales = nvfp4_kv_block_scales
                elif (
                    attn_metadata.q_data_type != FP8_DTYPE
                    and self.kv_cache_dtype.startswith("fp8")
                ):
                    # TRTLLM prefill attention does not support BF16 Q
                    # and fp8 kv cache. So to enable prefill attention
                    # with fp8 kv cache, we can construct a mock block
                    # and mock kv cache with BF16 KV involved in the prefill
                    #
                    kv_cache_permute = canonicalize_singleton_dim_strides(
                        kv_cache_permute
                    )
                    kv_strides = kv_cache_permute.stride()
                    assert (
                        kv_strides[-1] == 1
                        and kv_strides[-2] == kv_cache_permute.shape[-1]
                    ), (
                        "KV cache inner dims (block_size, head_size) must be "
                        f"contiguous, got strides {kv_strides}"
                    )
                    mock_kv_cache, mock_block_table = trtllm_prefill_attn_kvfp8_dequant(
                        kv_cache_permute,
                        block_tables_prefill,
                        layer._k_scale,
                        layer._v_scale,
                        attn_metadata.q_data_type,
                    )
                else:
                    mock_kv_cache = kv_cache_permute
                    mock_block_table = block_tables_prefill

                # 中文注释：调用 TRT-LLM 的 batch context attention 核函数。
                # 该核函数将所有 prefill 请求打包在一起，通过 cum_seq_lens_q/kv
                # 标记每个请求的边界，实现高效的批量 prefill 注意力计算。
                # 参数说明：
                #   - query: 所有 prefill 请求的 query 拼接
                #   - kv_cache / mock_kv_cache: KV cache（可能是反量化后的 BF16 版本）
                #   - workspace_buffer: TRT-LLM 需要的工作区缓冲区
                #   - block_tables: 分页 KV cache 的物理 block 映射表
                #   - seq_lens: 每个请求的 KV 序列长度
                #   - max_q_len / max_kv_len: 最大 query/KV 长度，用于内核优化
                #   - cum_seq_lens_q/kv: 累积序列长度，标记每个请求在拼接 tensor 中的起止位置
                #   - window_left: 滑动窗口左边界
                #   - sinks: attention sink tokens
                #   - o_sf_scale: 输出量化 scale（融合量化时使用）
                #   - out: 输出 tensor
                #   - kv_cache_sf: NVFP4 的 block scale factor
                trtllm_batch_context_with_kv_cache(
                    query=prefill_query,
                    kv_cache=mock_kv_cache,
                    workspace_buffer=workspace_buffer,
                    block_tables=mock_block_table,
                    seq_lens=seq_lens_prefill,
                    max_q_len=attn_metadata.prefill.max_q_len,
                    max_kv_len=attn_metadata.prefill.max_seq_len,
                    bmm1_scale=self.bmm1_scale,
                    bmm2_scale=self.bmm2_scale,
                    batch_size=attn_metadata.num_prefills,
                    cum_seq_lens_q=attn_metadata.prefill.cum_seq_lens_q,
                    cum_seq_lens_kv=attn_metadata.prefill.cum_seq_lens_kv,
                    window_left=self.window_left,
                    sinks=self.sinks,
                    o_sf_scale=self.o_sf_scale,
                    out=out,
                    kv_cache_sf=prefill_kv_block_scales,
                )

                if needs_fp8_out:
                    output[
                        num_decode_tokens : num_decode_tokens + num_prefill_tokens
                    ].copy_(out[:num_prefill_tokens].to(output.dtype))

        # 中文注释：decode 注意力计算路径。
        # decode 请求位于 batch 前面，所以 query 从位置 0 开始。
        # decode 同样有两种子路径：
        #   (a) FlashInfer 原生路径：使用 BatchDecodeWithPagedKVCacheWrapper
        #   (b) TRT-LLM 路径：使用 trtllm_batch_decode_with_kv_cache
        if num_decode_tokens > 0:
            decode_query = query[:num_decode_tokens]
            assert decode_query.shape[0] == num_decode_tokens

            if not decode_use_trtllm:
                # 中文注释：FlashInfer 原生 decode 路径。
                # 使用 FlashInfer 的 BatchDecodeWithPagedKVCacheWrapper。
                # decode 每个请求只需要计算 1 个新 token（投机解码时可能多个），
                # 但需要读取整个 KV cache 历史来做注意力。
                assert isinstance(attn_metadata.decode, FIDecode)
                decode_wrapper = attn_metadata.decode.wrapper
                assert decode_wrapper is not None
                assert decode_wrapper._window_left == self.window_left
                assert decode_wrapper._logits_soft_cap == (self.logits_soft_cap or 0.0)
                assert decode_wrapper._sm_scale == self.scale

                if self.is_kvcache_nvfp4:
                    kv_cache_permute = nvfp4_kv_data
                kv_cache_sf = nvfp4_kv_block_scales if self.is_kvcache_nvfp4 else None

                # NVFP4 kernel only supports FP8 output.
                # Use a pre-allocated FP8 buffer and dequantize afterwards.
                needs_fp8_out = self.is_kvcache_nvfp4 and output.dtype != FP8_DTYPE
                if needs_fp8_out:
                    out_decode = self._nvfp4_fp8_out[:num_decode_tokens]
                else:
                    out_decode = output[:num_decode_tokens]

                if use_dcp:
                    # 中文注释：DCP 模式下的 decode 注意力。
                    # 流程：(1) all-gather 收集所有 DCP worker 上的 query head
                    #       (2) 每个 worker 计算本地 KV 的注意力，同时返回 LSE
                    #       (3) 通过 dcp_combine 合并多个 worker 的输出
                    # LSE（log-sum-exp）用于数值稳定的加权合并。
                    decode_query = get_dcp_group().all_gather(
                        decode_query.contiguous(), dim=-2
                    )
                    output_tmp = torch.empty_like(decode_query)
                    lse = torch.empty(
                        (decode_query.size(0), decode_query.size(1)),
                        dtype=torch.float32,
                        device=decode_query.device,
                    )
                    decode_wrapper.run(
                        decode_query,
                        kv_cache_permute,
                        k_scale=layer._k_scale_float,
                        v_scale=layer._v_scale_float,
                        out=output_tmp,
                        lse=lse,
                        return_lse=True,
                        kv_cache_sf=kv_cache_sf,
                    )
                    output[:num_decode_tokens] = self.dcp_combine(
                        output_tmp,
                        lse,
                        get_dcp_group(),
                    )
                else:
                    decode_wrapper.run(
                        decode_query,
                        kv_cache_permute,
                        k_scale=layer._k_scale_float,
                        v_scale=layer._v_scale_float,
                        out=out_decode,
                        kv_cache_sf=kv_cache_sf,
                    )

                if needs_fp8_out:
                    output[:num_decode_tokens].copy_(out_decode.to(output.dtype))
            else:
                # 中文注释：TRT-LLM decode 注意力路径。
                # 使用 TRT-LLM 的 batch decode 核函数，支持 FP8/FP4 输出量化融合。
                # 与 prefill 路径类似，需要确保所有输入 tensor 严格连续。
                assert isinstance(attn_metadata.decode, TRTLLMDecode)
                # decode_query may be non-contiguous or have degenerate strides
                # on size=1 dims. contiguous() ensures memory layout; then
                # canonicalize_singleton_dim_strides fixes any remaining
                # degenerate strides on size=1 dims for TMA alignment.
                decode_query = decode_query.contiguous()
                decode_query = canonicalize_singleton_dim_strides(decode_query)
                workspace_buffer = _get_trtllm_gen_workspace_buffer()
                # 中文注释：TRT-LLM decode 路径使用的 block_tables 和 seq_lens。
                block_tables_decode = attn_metadata.decode.block_tables
                seq_lens_decode = attn_metadata.decode.seq_lens

                # This path needs to be enabled with VLLM_KV_CACHE_LAYOUT = HND
                # 中文注释：TRT-LLM decode 路径同样要求 HND 布局和严格连续的输入。
                assert get_kv_cache_layout() == "HND"
                assert is_strictly_contiguous(decode_query)
                assert is_strictly_contiguous(workspace_buffer)
                assert is_strictly_contiguous(block_tables_decode)
                assert is_strictly_contiguous(seq_lens_decode)
                kv_cache_permute = canonicalize_singleton_dim_strides(kv_cache_permute)
                kv_strides = kv_cache_permute.stride()
                assert (
                    kv_strides[-1] == 1 and kv_strides[-2] == kv_cache_permute.shape[-1]
                ), (
                    "KV cache inner dims (block_size, head_size) must be "
                    f"contiguous, got strides {kv_strides}"
                )

                if output.dtype == FP4_DTYPE:
                    assert self.o_sf_scale is not None
                    out = FP4Tensor(
                        data=output[:num_decode_tokens],
                        scale=output_block_scale,
                        scale_start_index=0,
                        original_shape=decode_query.shape,
                    )
                else:
                    assert self.o_sf_scale is None
                    out = output[:num_decode_tokens]

                # NVFP4 trtllm kernel only supports FP8 output.
                # Use a pre-allocated FP8 buffer and dequantize afterwards.
                needs_fp8_out = self.is_kvcache_nvfp4 and output.dtype != FP8_DTYPE
                if needs_fp8_out:
                    out = self._nvfp4_fp8_out[:num_decode_tokens]

                # 中文注释：计算每个 decode 请求的 query 长度。
                # 正常情况下每个 decode 请求只有 1 个 query token；
                # 投机解码时可能有多个。dummy_run 时 q_len=0 需要特殊处理为 1。
                if num_decode_tokens % attn_metadata.num_decodes != 0:
                    # This gets triggered when the dummy_run forces
                    # attention to be initialized with q_len = 0
                    q_len_per_req = 1
                else:
                    q_len_per_req = num_decode_tokens // attn_metadata.num_decodes

                # 中文注释：调用 TRT-LLM 的 batch decode 注意力核函数。
                # 与 prefill 不同，decode 使用 block_tables 而非 cum_seq_lens
                # 来索引 KV cache，因为每个 decode 请求的 KV 长度不同，
                # 使用分页 block table 可以更高效地随机访问。
                trtllm_batch_decode_with_kv_cache(
                    query=decode_query,
                    kv_cache=(
                        nvfp4_kv_data if self.is_kvcache_nvfp4 else kv_cache_permute
                    ),
                    workspace_buffer=workspace_buffer,
                    block_tables=block_tables_decode,
                    seq_lens=seq_lens_decode,
                    max_seq_len=attn_metadata.decode.max_seq_len,
                    bmm1_scale=self.bmm1_scale,
                    bmm2_scale=self.bmm2_scale,
                    window_left=self.window_left,
                    sinks=self.sinks,
                    o_sf_scale=self.o_sf_scale,
                    out=out,
                    q_len_per_req=q_len_per_req,
                    kv_cache_sf=(
                        nvfp4_kv_block_scales if self.is_kvcache_nvfp4 else None
                    ),
                )

                if needs_fp8_out:
                    output[:num_decode_tokens].copy_(out.to(output.dtype))
        # 中文注释：返回带原始 padding 的输出 tensor，保持与 CUDA graph 输入的形状一致。
        return output_padded

    # 中文注释：将当前 token 的 K/V 写入 KV cache。
    # 这是注意力计算的关键步骤：模型 forward 中产生的 key 和 value tensor
    # 需要按 slot_mapping 指定的位置写入分页 KV cache 的物理 block 中。
    # slot_mapping 将每个 token 的逻辑位置映射到 KV cache 的物理 slot，
    # 由 KV cache manager 在调度阶段生成。
    # 如果该层启用了 KV cache 共享（kv_sharing_target_layer_name 非 None），
    # 则跳过写入，因为目标层已经写过了。
    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        if self.kv_sharing_target_layer_name is None:
            # Reshape the input keys and values and store them in the cache.
            # Skip this if sharing KV cache with an earlier attention layer.
            # NOTE(woosuk): Here, key and value are padded while slot_mapping is
            # not padded. However, we don't need to do key[:num_actual_tokens]
            # and value[:num_actual_tokens] because the reshape_and_cache_flash
            # op uses the slot_mapping's shape to determine the number of
            # actual tokens.
            # 中文注释：将 key 和 value 通过 reshape_and_cache_flash 写入 KV cache。
            # k_cache 和 v_cache 分别是 KV cache 的 K 和 V 部分（dim=1 索引 0 和 1）。
            # reshape_and_cache_flash 会根据 slot_mapping 的长度确定实际 token 数，
            # 所以即使 key/value 有 padding 也不会写入多余数据。
            # 对于量化 KV cache（如 FP8），还会应用 k_scale/v_scale 进行量化。
            k_cache = kv_cache[:, 0]
            v_cache = kv_cache[:, 1]
            torch.ops._C_cache_ops.reshape_and_cache_flash(
                key,
                value,
                k_cache,
                v_cache,
                slot_mapping,
                self.kv_cache_dtype,
                layer._k_scale,
                layer._v_scale,
            )


# 中文注释：为 FlashInfer decode wrapper 提供优化的 plan 函数。
# FlashInfer 的 decode 注意力需要先通过 plan() 方法设置好元数据（block table、
# 序列长度等），然后才能调用 run() 执行注意力计算。
# fast_plan_decode 对 CUDA graph 场景做了专门优化：
#   - 首次调用时使用 FlashInfer 原始 plan() 进行 warmup（生成内部缓存的编译模块）
#   - 后续 CUDA graph replay 时使用 fast_decode_plan，避免不必要的设备间拷贝
# 这是性能关键路径，因为 decode 阶段每次 forward 都需要调用 plan。
def fast_plan_decode(
    self,  # decode wrapper
    indptr_cpu: torch.Tensor,
    indices: torch.Tensor,
    last_page_len_cpu: torch.Tensor,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
    pos_encoding_mode: str = "NONE",
    window_left: int = -1,
    logits_soft_cap: float | None = None,
    q_data_type: str | torch.dtype | None = "float16",
    kv_data_type: str | torch.dtype | None = None,
    o_data_type: str | torch.dtype | None = None,
    data_type: str | torch.dtype | None = None,
    sm_scale: float | None = None,
    rope_scale: float | None = None,
    rope_theta: float | None = None,
    non_blocking: bool = True,
    fixed_split_size: int = -1,
    disable_split_kv: bool = False,
) -> None:
    """
    A faster version of BatchDecodeWithPagedKVCacheWrapper::plan used for
    cudagraph capture/replay, while the no cudagraph version turns back
    to the original plan.
    using original plan after passing host-side buffers:
    - only host-to-device copy of indptr and last_page_len buffers
    Modifications for cudagraph:
    - only host-to-device copy of indptr and last_page_len buffers.
    - avoid device-to-device copy of indices buffer.

    Part of the code get inspiration from the original plan from FlashInfer repo
    and the implementation of fast_decode_plan for FlashInfer in SGlang repo.
    """
    # Warm up with the original plan if it is first call, and always run the
    # original plan if we run for dynamic shape. For fixed shape (cudagraph),
    # this warm up is to generate the _cached_module for the decode wrapper.
    # 中文注释：首次调用或非 CUDA graph 模式时，使用 FlashInfer 原始 plan()。
    # 原始 plan() 会编译并缓存内部的 Triton/CUDA 模块（_cached_module），
    # 后续 fast_decode_plan 才能复用这些编译结果。
    if not self.is_cuda_graph_enabled or getattr(self, "vllm_first_call", True):
        self.plan(
            indptr=indptr_cpu,
            indices=indices,
            last_page_len=last_page_len_cpu,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            page_size=page_size,
            pos_encoding_mode=pos_encoding_mode,
            window_left=window_left,
            logits_soft_cap=logits_soft_cap,
            q_data_type=q_data_type,
            kv_data_type=kv_data_type,
            o_data_type=o_data_type,
            data_type=data_type,
            sm_scale=sm_scale,
            rope_scale=rope_scale,
            rope_theta=rope_theta,
            non_blocking=non_blocking,
            block_tables=None,
            seq_lens=None,
            fixed_split_size=fixed_split_size,
            disable_split_kv=disable_split_kv,
        )
        self.vllm_first_call = False
        return

    assert self.is_cuda_graph_enabled, "Should be cudagraph only here"

    # 中文注释：CUDA graph 模式下使用优化的 fast_decode_plan。
    # 它只做必要的 host-to-device 拷贝（indptr 和 last_page_len），
    # 避免原始 plan() 中不必要的 device-to-device 拷贝（indices），
    # 从而降低 CUDA graph replay 时的 CPU 开销。
    fast_decode_plan(
        self,
        indptr=indptr_cpu,
        indices=indices,
        last_page_len=last_page_len_cpu,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        page_size=page_size,
        pos_encoding_mode=pos_encoding_mode,
        window_left=window_left,
        logits_soft_cap=logits_soft_cap,
        q_data_type=q_data_type,
        kv_data_type=kv_data_type,
        data_type=data_type,
        sm_scale=sm_scale,
        rope_scale=rope_scale,
        rope_theta=rope_theta,
        non_blocking=non_blocking,
        fixed_split_size=fixed_split_size,
        disable_split_kv=disable_split_kv,
    )


# 中文注释：Triton kernel，用于高效地将 block table 中的 page indices 拷贝到连续数组中。
# 在 FlashInfer 的 fast_decode_plan 中，需要将每个请求的物理 block id
# 从二维 block table 中提取出来，拼接成一维的 page_indices 数组。
# 使用 Triton kernel 可以在 GPU 上并行完成这一操作，避免 CPU 端的循环开销。
# 每个 Triton program 处理一个请求（req_idx），将该请求的 block ids
# 从 block_table 的对应行拷贝到 page_indices 的对应位置。
@triton.jit
def _copy_page_indices_kernel(
    page_indices,
    block_table,
    block_table_stride,
    cu_num_blocks,
    BLOCK_SIZE: tl.constexpr,
):
    # 中文注释：每个 program 处理一个请求。
    req_idx = tl.program_id(0)
    # 中文注释：计算当前请求在 block_table 中的行起始地址。
    row_ptr = block_table + req_idx * block_table_stride
    # 中文注释：从 cu_num_blocks 中读取当前请求的 block 起止索引。
    # cu_num_blocks 是累积 block 数组，[start, end) 标记每个请求的 block 范围。
    start_idx = tl.load(cu_num_blocks + req_idx)
    end_idx = tl.load(cu_num_blocks + req_idx + 1)
    num_blocks = end_idx - start_idx

    # 中文注释：分块加载并存储 block ids，每次处理 BLOCK_SIZE 个。
    # mask 处理最后一个不完整块的边界情况。
    offset = tl.arange(0, BLOCK_SIZE)
    for i in tl.range(0, num_blocks, BLOCK_SIZE):
        block_ids = tl.load(row_ptr + i + offset, mask=i + offset < num_blocks)
        tl.store(
            page_indices + start_idx + i + offset,
            block_ids,
            mask=i + offset < num_blocks,
        )
