# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
FlashAttention MLA 注意力后端模块。

本模块实现了基于 FlashAttention v3 的 MLA（Multi-Latent Attention）注意力后端，
主要用于 NVIDIA Hopper（SM90）GPU 上的 DeepSeek 模型推理。

核心特性：
1. 使用 FlashAttention v3 的 MLA 专用内核（flash_attn_varlen_func）
2. 支持 CUDA Graph 加速
3. 支持 DCP（Distributed Context Parallelism）分布式上下文并行
4. 支持 FA3 调度器元数据预计算
5. 支持 batch invariance（批量不变性）

架构概述：
- FlashAttnMLABackend: 后端定义
- FlashAttnMLAMetadataBuilder: 元数据构建器
- FlashAttnMLAImpl: 注意力实现

与其他 MLA 后端的区别：
- 使用 FlashAttention v3 而非专用 MLA 内核
- 支持 CUDA Graph 的完整捕获
- 将 q_nope 作为 q_v 传递给 FlashAttention，实现 MLA 的低秩分解注意力
"""

from dataclasses import dataclass
from typing import ClassVar

import torch

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.mla_attention import (
    MLACommonBackend,
    MLACommonDecodeMetadata,
    MLACommonImpl,
    MLACommonMetadata,
    MLACommonMetadataBuilder,
    QueryLenSupport,
)
from vllm.platforms.interface import DeviceCapability
from vllm.utils.math_utils import round_up
from vllm.utils.torch_utils import is_quantized_kv_cache
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    AttentionLayer,
    AttentionType,
    MultipleOf,
)
from vllm.v1.attention.backends.fa_utils import (
    flash_attn_supports_mla,
    get_flash_attn_version,
)
from vllm.v1.kv_cache_interface import AttentionSpec
from vllm.vllm_flash_attn import (  # type: ignore[attr-defined]
    flash_attn_varlen_func,
    get_scheduler_metadata,
)

logger = init_logger(__name__)


class FlashAttnMLABackend(MLACommonBackend):
    """
    FlashAttention MLA 注意力后端。

    定义了后端的基本属性和支持的配置。
    仅支持 SM90（Hopper）GPU，且需要 FlashAttention 支持 MLA。
    """
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
    ]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        """支持的块大小：16 的倍数。"""
        return [MultipleOf(16)]

    @staticmethod
    def get_name() -> str:
        return "FLASH_ATTN_MLA"

    @classmethod
    def supports_batch_invariance(cls) -> bool:
        """支持批量不变性。"""
        return True

    @staticmethod
    def get_builder_cls() -> type["FlashAttnMLAMetadataBuilder"]:
        return FlashAttnMLAMetadataBuilder

    @staticmethod
    def get_impl_cls() -> type["FlashAttnMLAImpl"]:
        return FlashAttnMLAImpl

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        """仅支持 SM90（Hopper）GPU。"""
        return capability.major == 9

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
        """检查 FlashAttention 是否支持 MLA。"""
        if not flash_attn_supports_mla():
            return "FlashAttention MLA not supported on this device"
        return None


@dataclass
class FlashAttnMLADecodeMetadata(MLACommonDecodeMetadata):
    """
    FlashAttention MLA decode 元数据。

    包含 FlashAttention v3 所需的 decode 元数据。
    """
    query_start_loc: torch.Tensor  # query 起始位置
    max_query_len: int  # 最大 query 长度
    max_seq_len: int  # 最大序列长度
    scheduler_metadata: torch.Tensor | None = None  # FA3 调度器元数据
    max_num_splits: int = 0  # 最大分割数（用于 CUDA Graph）


@dataclass
class FlashAttnMLAMetadata(MLACommonMetadata[FlashAttnMLADecodeMetadata]):
    """FlashAttention MLA 的完整元数据。"""
    pass


class FlashAttnMLAMetadataBuilder(MLACommonMetadataBuilder[FlashAttnMLAMetadata]):
    """
    FlashAttention MLA 的元数据构建器。

    负责构建 FlashAttention v3 所需的元数据，包括：
    1. FA3 调度器元数据预计算
    2. CUDA Graph 兼容的缓冲区管理
    3. 支持将小 prefill 与 decode 一起处理（reorder_batch_threshold=512）
    """
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH
    query_len_support: ClassVar[QueryLenSupport] = QueryLenSupport.VARLEN
    reorder_batch_threshold: int = 512  # 将小 prefill 与 decode 一起处理

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        interleave_size = vllm_config.parallel_config.cp_kv_cache_interleave_size
        super().__init__(
            kv_cache_spec,
            layer_names,
            vllm_config,
            device,
            FlashAttnMLAMetadata,
            supports_dcp_with_varlen=(interleave_size == 1),
        )
        self.max_num_splits = 0  # 分割数无上限
        self.fa_aot_schedule = get_flash_attn_version() == 3

        self.use_full_cuda_graph = (
            self.compilation_config.cudagraph_mode.has_full_cudagraphs()
        )
        self.max_cudagraph_size = self.compilation_config.max_cudagraph_capture_size

        if self.use_full_cuda_graph and self.fa_aot_schedule:
            # FA3 scheduler_metadata 大小：1 + round_up(batch_size, 4) * 4
            # +1 是 tile_count_semaphore（同步用）。
            # 每 batch 元素的 4 个 slot（num_prepare_batch_vectors）是：
            #   prepare_varlen + dynamic_split + sort_batches + head_swizzle
            max_batch_size = max(
                vllm_config.scheduler_config.max_num_seqs,
                self.max_cudagraph_size or 0,
            )
            self.scheduler_metadata = torch.zeros(
                1 + round_up(max_batch_size, 4) * 4,
                dtype=torch.int32,
                device=self.device,
            )
            # 使用 cuda graph 时，需要设置分割数的上限，以便在捕获期间
            # 预分配足够大的中间缓冲区。
            self.max_num_splits = (
                vllm_config.attention_config.flash_attn_max_num_splits_for_cuda_graph
            )

        if envs.VLLM_BATCH_INVARIANT:
            self.max_num_splits = 1

    def _schedule_decode(
        self,
        num_reqs,
        cu_query_lens,
        max_query_len,
        seqlens,
        max_seq_len,
        causal,
        max_num_splits,
    ):
        """
        为 decode 计算 FA3 调度器元数据。

        仅在 FA3 可用时调用 get_scheduler_metadata。
        """
        if self.fa_aot_schedule:
            return get_scheduler_metadata(
                batch_size=num_reqs,
                max_seqlen_q=max_query_len,
                max_seqlen_k=max_seq_len,
                num_heads_q=self.num_heads * self.dcp_world_size,
                num_heads_kv=1,
                headdim=self.mla_dims.qk_rope_head_dim,
                cache_seqlens=seqlens,
                qkv_dtype=self.kv_cache_spec.dtype,
                headdim_v=self.mla_dims.kv_lora_rank,
                page_size=self.page_size,
                cu_seqlens_q=cu_query_lens,
                causal=causal,
                num_splits=max_num_splits,
            )
        return None

    def _build_decode(
        self,
        block_table_tensor: torch.Tensor,
        seq_lens_device: torch.Tensor,
        max_seq_len: int,
        query_start_loc_cpu: torch.Tensor,
        query_start_loc_device: torch.Tensor,
        num_decode_tokens: int,
        dcp_tot_seq_lens_device: torch.Tensor | None,
    ) -> FlashAttnMLADecodeMetadata:
        """
        构建 decode 部分的元数据。

        流程：
        1. 计算最大 query 长度
        2. 确定是否使用 CUDA Graph 以及分割数
        3. 计算 FA3 调度器元数据
        4. 如果使用 CUDA Graph，将调度器元数据复制到持久化缓冲区
        """
        query_lens_cpu = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
        max_query_len = query_lens_cpu.max().item()

        # 对于 Flash Attention MLA + 完整 cudagraph
        max_num_splits = 0
        if (
            self.use_full_cuda_graph
            and self.max_cudagraph_size is not None
            and num_decode_tokens <= self.max_cudagraph_size
        ):
            # NOTE(woosuk): Setting num_splits > 1 may increase the memory
            # usage, because the intermediate buffers of size [num_splits,
            # num_heads, num_tokens, head_size] are allocated. Therefore,
            # we only set num_splits when using cuda graphs.
            # 设置 num_splits > 1 可能增加内存使用，因为中间缓冲区会被分配。
            # 因此，只在使用 cuda graphs 时设置 num_splits。
            max_num_splits = self.max_num_splits

        if envs.VLLM_BATCH_INVARIANT:
            max_num_splits = 1

        scheduler_metadata = self._schedule_decode(
            num_reqs=seq_lens_device.shape[0],
            cu_query_lens=query_start_loc_device,
            max_query_len=max_query_len,
            seqlens=seq_lens_device,
            max_seq_len=max_seq_len,
            causal=True,
            max_num_splits=max_num_splits,
        )

        if self.use_full_cuda_graph and scheduler_metadata is not None:
            n = scheduler_metadata.shape[0]
            # 确保持久化缓冲区足够大
            assert n <= self.scheduler_metadata.shape[0], (
                f"Scheduler metadata size {n} exceeds buffer size "
                f"{self.scheduler_metadata.shape[0]}"
            )
            self.scheduler_metadata[:n] = scheduler_metadata
            # NOTE(woosuk): We should zero out the rest of the scheduler
            # metadata to guarantee the correctness. Otherwise, some thread
            # blocks may use the invalid scheduler metadata and overwrite the
            # output buffer.
            # 应该将调度器元数据的其余部分清零以保证正确性。
            self.scheduler_metadata[n:] = 0
            scheduler_metadata = self.scheduler_metadata[:n]

        metadata = FlashAttnMLADecodeMetadata(
            block_table=block_table_tensor,
            seq_lens=seq_lens_device,
            query_start_loc=query_start_loc_device,
            max_query_len=max_query_len,
            max_seq_len=max_seq_len,
            scheduler_metadata=scheduler_metadata,
            max_num_splits=max_num_splits,
            dcp_tot_seq_lens=dcp_tot_seq_lens_device,
        )
        return metadata


class FlashAttnMLAImpl(MLACommonImpl[FlashAttnMLAMetadata]):
    """
    FlashAttention MLA 注意力的具体实现。

    使用 FlashAttention v3 的 MLA 专用内核执行注意力计算。
    将 MLA 的 q_nope 作为 q_v 传递给 FlashAttention，实现低秩分解注意力。
    """
    can_return_lse_for_decode: bool = True

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
        **mla_args,
    ) -> None:
        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            alibi_slopes,
            sliding_window,
            kv_cache_dtype,
            logits_soft_cap,
            attn_type,
            kv_sharing_target_layer_name,
            **mla_args,
        )

        assert flash_attn_supports_mla(), "FlashAttnMLA is not supported on this device"

        unsupported_features = [alibi_slopes, sliding_window, logits_soft_cap]
        if any(unsupported_features):
            raise NotImplementedError(
                "FlashAttnMLAImpl does not support one of the following: "
                "alibi_slopes, sliding_window, logits_soft_cap"
            )

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "Encoder self-attention and "
                "encoder/decoder cross-attention "
                "are not implemented for "
                "FlashAttnMLAImpl"
            )

        if is_quantized_kv_cache(self.kv_cache_dtype):
            raise NotImplementedError(
                "FlashAttnMLA V1 with FP8 KV cache not yet supported"
            )

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: FlashAttnMLAMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Decode 注意力的前向传播。

        MLA 注意力计算流程：
        1. 将 q 拆分为 q_nope（无 RoPE 部分）和 q_pe（RoPE 部分）
        2. 将 KV 缓存拆分为 kv_c_cache（压缩 KV）和 k_pe_cache（RoPE K）
        3. 调用 flash_attn_varlen_func：
           - q = q_pe（RoPE 部分作为 Q）
           - k = k_pe_cache（RoPE 部分作为 K）
           - v = kv_c_cache（压缩 KV 作为 V）
           - q_v = q_nope（无 RoPE 部分作为 Q_V，用于 MLA 的低秩分解）
        4. 返回注意力输出和可选的 LSE
        """
        assert kv_c_and_k_pe_cache.numel() > 0
        assert attn_metadata.decode is not None

        if type(q) is tuple:
            q_nope, q_pe = q
        else:
            q_nope, q_pe = torch.split(
                q, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
            )

        if is_quantized_kv_cache(self.kv_cache_dtype):
            raise NotImplementedError("FP8 FlashAttention MLA not yet supported")

        kv_c_cache = kv_c_and_k_pe_cache[..., : self.kv_lora_rank]
        k_pe_cache = kv_c_and_k_pe_cache[..., self.kv_lora_rank :]

        # NOTE(matt): During CUDA graph capture, max_query_len can be 0, but the
        # kernel uses this to calculate grid dimensions. Ensure it's at least 1
        # to prevent invalid grid configuration during graph capture.
        # 在 CUDA Graph 捕获期间，max_query_len 可能为 0，但内核使用它来计算 grid 维度。
        # 确保至少为 1 以防止无效的 grid 配置。
        max_seqlen_q = max(attn_metadata.decode.max_query_len, 1)

        attn_out = flash_attn_varlen_func(
            q=q_pe,
            k=k_pe_cache.unsqueeze(-2),  # 添加 head 维度 1
            v=kv_c_cache.unsqueeze(-2),  # 添加 head 维度 1
            q_v=q_nope,  # MLA 的低秩分解：q_nope 作为 q_v
            max_seqlen_q=max_seqlen_q,
            cu_seqlens_q=attn_metadata.decode.query_start_loc,
            max_seqlen_k=attn_metadata.decode.max_seq_len,
            seqused_k=attn_metadata.decode.seq_lens,
            block_table=attn_metadata.decode.block_table,
            softmax_scale=self.scale,
            causal=True,
            return_softmax_lse=self.need_to_return_lse_for_decode,
            fa_version=3,  # 只支持版本 3
            scheduler_metadata=attn_metadata.decode.scheduler_metadata,
            num_splits=attn_metadata.decode.max_num_splits,
            cp_world_size=self.dcp_world_size,
            cp_rank=self.dcp_rank,
            cp_tot_seqused_k=attn_metadata.decode.dcp_tot_seq_lens,
        )

        if self.need_to_return_lse_for_decode:
            o, lse = attn_out
            # FA 返回 LSE 形状为 [H, B]，但 DCP 需要 [B, H]
            return o, lse.transpose(0, 1)  # [H, B] -> [B, H]
        else:
            o = attn_out
            return o, None
