# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
FlashInfer MLA 稀疏注意力后端模块。

本模块实现了基于 FlashInfer TRT-LLM MLA 内核的稀疏注意力后端，
主要用于 NVIDIA Blackwell（SM10.x）GPU 上的 DeepSeek V3.2 等稀疏模型。

核心特性：
1. 使用 FlashInfer 的 trtllm_batch_decode_with_kv_cache_mla 内核
2. 支持 sparse_mla_top_k 参数进行稀疏注意力
3. 支持 FP8 KV 缓存（fp8, fp8_e4m3）
4. 使用 Triton 内核将 per-request 索引转换为全局物理索引

与 FlashMLA 稀疏后端的区别：
- 使用 FlashInfer 而非 FlashMLA 内核
- 仅支持 Blackwell GPU（SM10.x）
- 需要 qk_nope_head_dim 在 [128, 192] 范围内
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import torch
from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla

from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.mla_attention import (
    get_mla_dims,
)
from vllm.platforms.interface import DeviceCapability
from vllm.utils.torch_utils import is_quantized_kv_cache
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionLayer,
    AttentionMetadata,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
    SparseMLAAttentionImpl,
)
from vllm.v1.attention.backends.mla.sparse_utils import (
    triton_convert_req_index_to_global_index,
)
from vllm.v1.attention.backends.utils import KVCacheLayoutType
from vllm.v1.kv_cache_interface import AttentionSpec

if TYPE_CHECKING:
    from vllm.model_executor.models.deepseek_v2 import Indexer

logger = init_logger(__name__)

FLASHINFER_MLA_SPARSE_WORKSPACE_BUFFER_SIZE = 128 * 1024 * 1024  # 128MB


class FlashInferMLASparseBackend(AttentionBackend):
    """
    FlashInfer MLA 稀疏注意力后端。

    使用 FlashInfer TRT-LLM MLA 内核和 sparse_mla_top_k 参数进行稀疏注意力。
    """
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
    ]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        """支持的块大小：32 和 64。"""
        return [32, 64]

    @staticmethod
    def get_name() -> str:
        return "FLASHINFER_MLA_SPARSE"

    @staticmethod
    def get_impl_cls() -> type["FlashInferMLASparseImpl"]:
        return FlashInferMLASparseImpl

    @staticmethod
    def get_builder_cls() -> type["FlashInferMLASparseMetadataBuilder"]:
        return FlashInferMLASparseMetadataBuilder

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        """支持的 head size：576。"""
        return [576]

    @classmethod
    def is_mla(cls) -> bool:
        return True

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        """FlashInfer 稀疏 MLA 目标 Blackwell（SM 10.x）。"""
        return capability.major == 10

    @classmethod
    def supports_combination(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: CacheDType | None,
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        device_capability: DeviceCapability,
    ) -> str | None:
        """检查是否支持给定的配置组合。"""
        # FlashInfer MLA 稀疏内核要求 qk_nope_head_dim 在 [128, 192] 范围内
        from vllm.config import get_current_vllm_config

        vllm_config = get_current_vllm_config()
        if vllm_config.model_config is not None:
            hf_text_config = vllm_config.model_config.hf_text_config
            qk_nope_head_dim = getattr(hf_text_config, "qk_nope_head_dim", 1)
            if qk_nope_head_dim not in [128, 192]:
                return (
                    "FlashInfer MLA Sparse kernel requires qk_nope_head_dim "
                    f"in [128, 192], but got {qk_nope_head_dim}"
                )
            # 检查 index_topk 是否存在（表示稀疏模型）
            if not hasattr(hf_text_config, "index_topk"):
                return "FlashInfer MLA Sparse requires model with index_topk config"
        return None

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
    def get_required_kv_cache_layout(cls) -> "KVCacheLayoutType | None":
        """要求 HND（Head-NumTokens-Dim）布局。"""
        return "HND"


@dataclass
class FlashInferMLASparseMetadata(AttentionMetadata):
    """
    FlashInfer MLA 稀疏注意力的元数据。

    包含稀疏注意力所需的基本元数据。
    """
    num_reqs: int
    max_query_len: int
    max_seq_len: int
    num_actual_tokens: int

    query_start_loc: torch.Tensor  # Query 起始位置
    slot_mapping: torch.Tensor  # Slot 映射
    block_table: torch.Tensor  # 块表
    req_id_per_token: torch.Tensor  # 每个 token 的请求 ID

    seq_lens: torch.Tensor  # 所有序列的序列长度（context + query）

    block_size: int = 64
    topk_tokens: int = 2048


class FlashInferMLASparseMetadataBuilder(
    AttentionMetadataBuilder[FlashInferMLASparseMetadata]
):
    """
    FlashInfer MLA 稀疏注意力的元数据构建器。

    负责构建 FlashInfer 内核所需的元数据。
    """
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        self.vllm_config = vllm_config
        self.layer_names = layer_names
        self.kv_cache_spec = kv_cache_spec
        self.model_config = vllm_config.model_config
        self.device = device

        self.mla_dims = get_mla_dims(self.model_config)
        self.topk_tokens = vllm_config.model_config.hf_config.index_topk

        self.req_id_per_token_buffer = torch.empty(
            (vllm_config.scheduler_config.max_num_batched_tokens,),
            dtype=torch.int32,
            device=device,
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> FlashInferMLASparseMetadata:
        """
        构建 FlashInfer MLA 稀疏注意力的元数据。

        流程：
        1. 构建 req_id_per_token 映射
        2. 零填充缓冲区以支持 CUDA Graph
        3. 返回元数据
        """
        cm = common_attn_metadata
        num_tokens = cm.num_actual_tokens

        # 构建 req_id_per_token 映射
        starts = np.asarray(cm.query_start_loc_cpu, dtype=np.int32)
        seg_lengths = np.diff(starts)
        req_id_per_token = np.repeat(
            np.arange(seg_lengths.shape[0], dtype=np.int32), seg_lengths
        )

        # 为 CUDA Graph 零填充
        self.req_id_per_token_buffer.fill_(0)
        self.req_id_per_token_buffer[: req_id_per_token.shape[0]].copy_(
            torch.from_numpy(req_id_per_token), non_blocking=True
        )
        req_id_per_token_tensor = self.req_id_per_token_buffer[:num_tokens]

        return FlashInferMLASparseMetadata(
            num_reqs=cm.num_reqs,
            max_query_len=cm.max_query_len,
            max_seq_len=cm.max_seq_len,
            num_actual_tokens=cm.num_actual_tokens,
            query_start_loc=cm.query_start_loc,
            slot_mapping=cm.slot_mapping,
            block_table=cm.block_table_tensor,
            req_id_per_token=req_id_per_token_tensor,
            seq_lens=cm.seq_lens,
            block_size=self.kv_cache_spec.block_size,
            topk_tokens=self.topk_tokens,
        )


# 全局 workspace 缓冲区（延迟初始化）
_fi_sparse_workspace: torch.Tensor | None = None


def _get_workspace_buffer(device: torch.device) -> torch.Tensor:
    """获取或创建 FlashInfer 稀疏 MLA 的 workspace 缓冲区。"""
    global _fi_sparse_workspace
    if _fi_sparse_workspace is None:
        _fi_sparse_workspace = torch.zeros(
            FLASHINFER_MLA_SPARSE_WORKSPACE_BUFFER_SIZE,
            dtype=torch.uint8,
            device=device,
        )
    return _fi_sparse_workspace


class FlashInferMLASparseImpl(SparseMLAAttentionImpl[FlashInferMLASparseMetadata]):
    """
    FlashInfer MLA 稀疏注意力的具体实现。

    使用 TRT-LLM MLA 内核和 sparse_mla_top_k 参数进行稀疏注意力计算。
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
        indexer: "Indexer | None" = None,
        **mla_args,
    ) -> None:
        unsupported_features = [alibi_slopes, sliding_window, logits_soft_cap]
        if any(unsupported_features):
            raise NotImplementedError(
                "FlashInferMLASparseImpl does not support one of the following: "
                "alibi_slopes, sliding_window, logits_soft_cap"
            )

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "Encoder self-attention and "
                "encoder/decoder cross-attention "
                "are not implemented for "
                "FlashInferMLASparseImpl"
            )

        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype

        # MLA 特有维度
        self.kv_lora_rank: int = mla_args["kv_lora_rank"]
        self.qk_nope_head_dim: int = mla_args["qk_nope_head_dim"]
        self.qk_rope_head_dim: int = mla_args["qk_rope_head_dim"]

        assert indexer is not None, "Indexer required for sparse MLA"
        self.topk_indices_buffer: torch.Tensor | None = indexer.topk_indices_buffer

        self._workspace_buffer: torch.Tensor | None = None
        self.bmm1_scale: float | None = None
        self.bmm2_scale: float | None = None

        # FP8 query 量化在使用 FP8 KV 缓存时是必需的，
        # 因为 TRTLLM-GEN 稀疏 MLA 内核要求 query 和 kv_cache 的 dtype 匹配
        #（不支持混合 bf16+fp8）。
        self.supports_quant_query_input = True

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: FlashInferMLASparseMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        稀疏 MLA 注意力的主前向传播入口。

        流程：
        1. 如果 q 是元组，拼接为完整 q
        2. 将 topk 逻辑索引转换为全局物理索引
        3. 计算 bmm1_scale 和 bmm2_scale（考虑 FP8 缩放）
        4. 调用 trtllm_batch_decode_with_kv_cache_mla 计算注意力
        """
        if isinstance(q, tuple):
            q = torch.cat(q, dim=-1)

        num_actual_toks = q.shape[0]

        assert self.topk_indices_buffer is not None
        topk_indices = self.topk_indices_buffer[:num_actual_toks]

        # 将逻辑索引转换为物理索引，并获取有效计数
        topk_indices_physical, seq_lens = triton_convert_req_index_to_global_index(
            attn_metadata.req_id_per_token[:num_actual_toks],
            attn_metadata.block_table,
            topk_indices,
            BLOCK_SIZE=attn_metadata.block_size,
            NUM_TOPK_TOKENS=topk_indices.shape[1],
            return_valid_counts=True,
        )

        if self._workspace_buffer is None:
            self._workspace_buffer = _get_workspace_buffer(q.device)

        # 计算缩放因子（考虑 FP8 缓存）
        if self.bmm1_scale is None:
            self.bmm1_scale = self.scale
            if is_quantized_kv_cache(self.kv_cache_dtype):
                self.bmm1_scale *= layer._q_scale_float * layer._k_scale_float
        if self.bmm2_scale is None:
            self.bmm2_scale = 1.0
            if is_quantized_kv_cache(self.kv_cache_dtype):
                self.bmm2_scale *= layer._k_scale_float

        o = trtllm_batch_decode_with_kv_cache_mla(
            query=q.unsqueeze(1),
            kv_cache=kv_c_and_k_pe_cache.unsqueeze(1),
            workspace_buffer=self._workspace_buffer,
            qk_nope_head_dim=self.qk_nope_head_dim,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            block_tables=topk_indices_physical.unsqueeze(1),
            seq_lens=seq_lens,
            max_seq_len=attn_metadata.topk_tokens,
            bmm1_scale=self.bmm1_scale,
            bmm2_scale=self.bmm2_scale,
            sparse_mla_top_k=attn_metadata.topk_tokens,
        )
        return o.view(-1, o.shape[-2], o.shape[-1]), None
