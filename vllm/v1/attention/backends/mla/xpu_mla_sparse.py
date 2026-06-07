# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
XPU MLA 稀疏注意力后端模块。

本模块实现了基于 Intel XPU 的稀疏 MLA（Multi-Latent Attention）注意力后端，
用于 Intel GPU 上的 DeepSeek 稀疏模型推理。

核心特性：
1. 稀疏注意力：只关注索引器选出的 topk 个最重要的 KV token
2. 使用 Triton 内核将 per-request 索引转换为全局物理索引
3. 使用 XPU 特有的 BF16 MLA 稀疏内核（triton_bf16_mla_sparse_interface）
4. 不支持 FP8 KV 缓存
5. 不支持 CUDA Graph

与其他稀疏 MLA 后端的区别：
- 仅支持 Intel XPU
- 不支持 FP8 KV 缓存
- 使用 XPU 特有的 Triton 内核
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Optional

import numpy as np
import torch

from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.mla_attention import (
    get_mla_dims,
)
from vllm.utils.torch_utils import is_quantized_kv_cache
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionLayer,
    AttentionMetadata,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    SparseMLAAttentionImpl,
)
from vllm.v1.attention.backends.mla.flashmla_sparse import (
    triton_convert_req_index_to_global_index,
)
from vllm.v1.attention.ops.xpu_mla_sparse import triton_bf16_mla_sparse_interface
from vllm.v1.kv_cache_interface import AttentionSpec

if TYPE_CHECKING:
    from vllm.model_executor.models.deepseek_v2 import Indexer
logger = init_logger(__name__)


class XPUMLASparseBackend(AttentionBackend):
    """
    XPU MLA 稀疏注意力后端。

    定义了后端的基本属性和支持的配置。
    仅支持 Intel XPU，不支持 FP8 KV 缓存。
    """
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
    ]

    @staticmethod
    def get_name() -> str:
        return "XPU_MLA_SPARSE"

    @staticmethod
    def get_metadata_cls() -> type["XPUMLASparseMetadata"]:
        return XPUMLASparseMetadata

    @staticmethod
    def get_builder_cls() -> type["XPUMLASparseMetadataBuilder"]:
        return XPUMLASparseMetadataBuilder

    @staticmethod
    def get_impl_cls() -> type["XPUMLASparseImpl"]:
        return XPUMLASparseImpl

    @classmethod
    def is_mla(cls) -> bool:
        return True

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,  # 对于 MLA 假设为 1
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        return (num_blocks, block_size, head_size)

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        """支持的 head size：576。"""
        return [576]


@dataclass
class XPUMLASparseMetadata(AttentionMetadata):
    """
    XPU MLA 稀疏注意力的元数据。

    包含稀疏注意力所需的基本元数据。
    """
    num_reqs: int
    max_query_len: int
    max_seq_len: int

    num_actual_tokens: int  # 不含 padding 的实际 token 数量
    query_start_loc: torch.Tensor
    slot_mapping: torch.Tensor

    block_table: torch.Tensor
    req_id_per_token: torch.Tensor

    block_size: int = 1
    topk_tokens: int = 2048


@dataclass
class XPUMLASparseMetadataBuilder(AttentionMetadataBuilder[XPUMLASparseMetadata]):
    """
    XPU MLA 稀疏注意力的元数据构建器。

    负责构建 XPU 稀疏注意力内核所需的元数据。
    不支持 CUDA Graph。
    """
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.NEVER

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        self.kv_cache_spec = kv_cache_spec
        self.model_config = vllm_config.model_config
        parallel_config = vllm_config.parallel_config
        self.device = device
        max_num_batched_tokens = vllm_config.scheduler_config.max_num_batched_tokens

        self.num_heads = self.model_config.get_num_attention_heads(parallel_config)
        self.mla_dims = get_mla_dims(self.model_config)
        self.topk_tokens = vllm_config.model_config.hf_config.index_topk
        self.topk_tokens_tensor = torch.tensor(
            [self.topk_tokens], device=device, dtype=torch.int32
        )
        self.max_model_len_tensor = torch.tensor(
            [self.model_config.max_model_len], device=device, dtype=torch.int32
        )
        # 当 indices 不为 None 时，flash_mla_with_kvcache 会忽略此值
        self.dummy_block_table = torch.empty(
            (1, 1), dtype=torch.int32, device=self.device
        )

        self.req_id_per_token_buffer = torch.empty(
            (max_num_batched_tokens,),
            dtype=torch.int32,
            device=device,
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> XPUMLASparseMetadata:
        """
        构建 XPU MLA 稀疏注意力的元数据。

        流程：
        1. 构建 req_id_per_token 映射
        2. 零填充缓冲区
        3. 返回元数据
        """
        num_tokens = common_attn_metadata.num_actual_tokens
        starts = np.asarray(common_attn_metadata.query_start_loc_cpu, dtype=np.int32)
        seg_lengths = np.diff(starts)
        req_id_per_token = np.repeat(
            np.arange(seg_lengths.shape[0], dtype=np.int32), seg_lengths
        )
        # 为 CUDA Graph 零填充
        self.req_id_per_token_buffer.fill_(0)
        self.req_id_per_token_buffer[: req_id_per_token.shape[0]].copy_(
            torch.from_numpy(req_id_per_token), non_blocking=True
        )

        req_id_per_token = self.req_id_per_token_buffer[:num_tokens]

        metadata = XPUMLASparseMetadata(
            num_reqs=common_attn_metadata.num_reqs,
            max_query_len=common_attn_metadata.max_query_len,
            max_seq_len=common_attn_metadata.max_seq_len,
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            query_start_loc=common_attn_metadata.query_start_loc,
            slot_mapping=common_attn_metadata.slot_mapping,
            block_table=common_attn_metadata.block_table_tensor,
            req_id_per_token=req_id_per_token,
            block_size=self.kv_cache_spec.block_size,
            topk_tokens=self.topk_tokens,
        )
        return metadata


class XPUMLASparseImpl(SparseMLAAttentionImpl[XPUMLASparseMetadata]):
    """
    XPU MLA 稀疏注意力的具体实现。

    使用 XPU 特有的 BF16 MLA 稀疏内核执行注意力计算。
    不支持 FP8 KV 缓存。
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        # MLA 特有参数
        topk_indice_buffer: torch.Tensor | None = None,
        indexer: Optional["Indexer"] = None,
        **mla_args,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.kv_lora_rank: int = mla_args["kv_lora_rank"]
        self.softmax_scale = scale
        assert indexer is not None
        self.topk_indices_buffer: torch.Tensor | None = indexer.topk_indices_buffer

    def _forward_bf16_kv(
        self,
        q: torch.Tensor,  # [sq, heads, d_qk]
        kv_c_and_k_pe_cache: torch.Tensor,  # [blocks, heads, d_qk]
        topk_indices: torch.Tensor,  # [sq, topk]
        attn_metadata: XPUMLASparseMetadata,
    ) -> torch.Tensor:
        """
        BF16 KV 缓存的前向传播。

        流程：
        1. 展平 KV 缓存为 [num_blocks * block_size, 1, head_size]
        2. 重塑 topk_indices 为 [sq, 1, topk]
        3. 调用 triton_bf16_mla_sparse_interface 计算注意力
        4. 裁切输出回实际 head 数
        """
        num_tokens = q.shape[0]
        kv_c_and_k_pe_cache = kv_c_and_k_pe_cache.view(
            -1, 1, kv_c_and_k_pe_cache.shape[-1]
        )

        topk_indices = topk_indices.view(num_tokens, 1, -1)

        output, _, _ = triton_bf16_mla_sparse_interface(
            q,
            kv_c_and_k_pe_cache,
            topk_indices,
            sm_scale=self.softmax_scale,
        )

        return output[:, : self.num_heads, :]

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: XPUMLASparseMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        稀疏 MLA 注意力的主前向传播入口。

        对于稀疏 FlashMLA 内核，prefill 和 decode 都使用 MQA 576/512 方法。

        流程：
        1. 检查不支持 FP8 KV 缓存
        2. 如果 q 是元组，拼接为完整 q
        3. 获取 topk 索引并转换为全局物理索引
        4. 调用 _forward_bf16_kv 计算注意力
        """
        # NOTE(lucas): for the sparse FlashMLA kernels the kernels want to use
        # MQA 576/512 approach for both prefill and decode
        # 对于稀疏 FlashMLA 内核，prefill 和 decode 都使用 MQA 576/512 方法

        if is_quantized_kv_cache(self.kv_cache_dtype):
            raise NotImplementedError("FP8 kv is not supported with XPU MLA Sparse yet")

        # 如果 q 是元组（ql_nope, q_pe），拼接
        if isinstance(q, tuple):
            q = torch.cat(q, dim=-1)

        num_actual_toks = q.shape[0]

        assert self.topk_indices_buffer is not None
        topk_indices = self.topk_indices_buffer[:num_actual_toks]

        # 将 per-request 逻辑索引转换为全局物理索引
        topk_indices_global = triton_convert_req_index_to_global_index(
            attn_metadata.req_id_per_token,
            attn_metadata.block_table,
            topk_indices,
            BLOCK_SIZE=attn_metadata.block_size,
            NUM_TOPK_TOKENS=attn_metadata.topk_tokens,
        )

        attn_out = self._forward_bf16_kv(
            q, kv_c_and_k_pe_cache, topk_indices_global, attn_metadata
        )

        return attn_out, None
