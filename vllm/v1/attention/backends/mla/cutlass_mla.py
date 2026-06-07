# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
CUTLASS MLA 注意力后端模块。

本模块实现了基于 CUTLASS 的 MLA（Multi-Latent Attention）注意力后端，
专为 NVIDIA Blackwell（SM100）GPU 优化。

核心特性：
1. 使用 CUTLASS 的 SM100 MLA decode 内核（sm100_cutlass_mla_decode）
2. 支持 FP8 KV 缓存（fp8, fp8_e4m3）
3. 支持动态 workspace 大小调整
4. 支持 CUDA Graph（UNIFORM_SINGLE_TOKEN_DECODE 模式）
5. 可返回 LSE（LogSumExp）用于分布式归约
6. 强制 padding 到 MAX_HEADS=128 个头

与其他 MLA 后端的区别：
- 使用 CUTLASS 而非 FlashAttention/Triton
- 仅支持 Blackwell GPU（SM100）
- 固定块大小为 128
- 强制 padding 到 128 个头
"""

import os
from typing import ClassVar

import torch

import vllm._custom_ops as ops
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.mla_attention import (
    MLACommonBackend,
    MLACommonImpl,
    MLACommonMetadata,
    MLACommonMetadataBuilder,
)
from vllm.platforms.interface import DeviceCapability
from vllm.utils.platform_utils import num_compute_units
from vllm.utils.torch_utils import is_quantized_kv_cache
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    AttentionLayer,
    AttentionType,
    MultipleOf,
)

logger = init_logger(__name__)


class CutlassMLAMetadataBuilder(MLACommonMetadataBuilder[MLACommonMetadata]):
    """CUTLASS MLA 的元数据构建器。启用完整 CUDA Graph 支持用于 decode-only 捕获。"""
    _cudagraph_support: ClassVar[AttentionCGSupport] = (
        AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
    )


class CutlassMLABackend(MLACommonBackend):
    """
    CUTLASS MLA 注意力后端。

    定义了后端的基本属性和支持的配置。
    仅支持 Blackwell（SM100）GPU。
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
        """固定块大小为 128。"""
        return [128]

    @staticmethod
    def get_name() -> str:
        return "CUTLASS_MLA"

    @staticmethod
    def get_impl_cls() -> type["CutlassMLAImpl"]:
        return CutlassMLAImpl

    @staticmethod
    def get_builder_cls() -> type["CutlassMLAMetadataBuilder"]:
        return CutlassMLAMetadataBuilder

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        """仅支持 Blackwell（SM100）GPU。"""
        return capability.major == 10


class SM100Workspace:
    """
    SM100 CUTLASS MLA 的 workspace 管理器。

    管理 CUTLASS 内核所需的工作空间缓冲区，支持动态大小调整。

    属性：
    - _workspace_buf: 工作空间缓冲区
    - _block_size: 块大小（固定为 128）
    - _sm_count: SM 数量
    """

    def __init__(self, initial_workspace_size):
        self._workspace_buf = torch.empty(
            initial_workspace_size, device="cuda", dtype=torch.uint8
        )

        self._block_size = 128  # 强制为 128

        # 预计算 sm_count 以避免重复计算。使用设备 0 作为代理
        #（假设所有设备类似）。
        self._sm_count = num_compute_units(0)

    def get_buf(self):
        """返回工作空间缓冲区。"""
        return self._workspace_buf

    def ensure_size(self, attn_metadata: MLACommonMetadata, num_kv_splits: int):
        """
        确保工作空间缓冲区足够大。

        根据当前 batch 的大小和 KV split 数量计算所需的工作空间大小，
        如果当前缓冲区不够大则扩展。
        """
        batch_size = attn_metadata.num_reqs
        max_seq_len = attn_metadata.max_query_len

        workspace_size = ops.sm100_cutlass_mla_get_workspace_size(
            max_seq_len * self._block_size,
            batch_size,
            self._sm_count,
            num_kv_splits=num_kv_splits,
        )

        if self._workspace_buf.shape[0] < workspace_size:
            self._workspace_buf.resize_(workspace_size)


# 全局 workspace 实例（128MB 初始大小）
g_sm100_workspace = SM100Workspace(128 * 1024 * 1024)  # 128MB

MAX_HEADS = 128  # CUTLASS 内核要求的最大 head 数


class CutlassMLAImpl(MLACommonImpl[MLACommonMetadata]):
    """
    CUTLASS MLA 注意力的具体实现。

    使用 CUTLASS 的 SM100 MLA decode 内核执行注意力计算。
    强制 padding 到 MAX_HEADS=128 个头。
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
            q_pad_num_heads=MAX_HEADS,
            **mla_args,
        )

        unsupported_features = [alibi_slopes, sliding_window, logits_soft_cap]
        if any(unsupported_features):
            raise NotImplementedError(
                "CutlassMLAImpl does not support one of the following: "
                "alibi_slopes, sliding_window, logits_soft_cap"
            )

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "Encoder self-attention and "
                "encoder/decoder cross-attention "
                "are not implemented for "
                "CutlassMLAImpl"
            )

        # TODO: Currently, num_kv_splits is limited to 16 to avoid hanging
        #       issues. In case the code hangs, use:
        #       FORCE_NUM_KV_SPLITS=1
        # 目前 num_kv_splits 限制为 16 以避免挂起问题。
        force_num_kv_splits = os.environ.get("FORCE_NUM_KV_SPLITS", None)
        if force_num_kv_splits:
            logger.debug_once("Forcing num_kv_splits to %d", int(force_num_kv_splits))
            self._num_kv_splits = int(force_num_kv_splits)
        else:
            self._num_kv_splits = -1  # => 自动检测

        # 所有执行共享 workspace 缓冲区
        self._workspace = g_sm100_workspace

    def _sm100_cutlass_mla_decode(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        seq_lens: torch.Tensor,
        page_table: torch.Tensor,
        workspace: torch.Tensor,
        sm_scale: float,
        num_kv_splits: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        SM100 CUTLASS MLA decode 的核心调用。

        参数验证和内核调用。

        参数：
        - q_nope: 无 RoPE 部分的 query [B, H, D_latent]
        - q_pe: RoPE 部分的 query [B, H, D_rope]
        - kv_c_and_k_pe_cache: KV 缓存 [num_blocks, PAGE_SIZE, D_ckv]
        - seq_lens: 序列长度 [B]
        - page_table: 页表 [B, num_blocks]
        - workspace: 工作空间缓冲区
        - sm_scale: softmax 缩放因子
        - num_kv_splits: KV 分割数
        """
        assert q_nope.ndim == 3, f"q_nope must be a 3D tensor, but got {q_nope.ndim}"
        assert q_pe.ndim == 3, f"q_pe must be a 3D tensor, but got {q_pe.ndim}"
        assert kv_c_and_k_pe_cache.ndim == 3, (
            "kv_c_and_k_pe_cache must be a 3D tensor, but got {}".format(
                kv_c_and_k_pe_cache.ndim
            )
        )

        B_q, H, D_q_nope = q_nope.shape
        B_q_2, H_2, D_q_pe = q_pe.shape
        assert (B_q == B_q_2) and (H == H_2)

        _, PAGE_SIZE, D_ckv = kv_c_and_k_pe_cache.shape

        D_latent = 512
        D_rope = 64
        assert D_q_nope == D_latent
        assert D_q_pe == D_rope
        assert D_ckv == D_latent + D_rope

        MAX_HEADS = 128
        assert H <= MAX_HEADS, f"H must be <= {MAX_HEADS}, but got {H}"

        assert len(page_table.shape) == 2
        B_block_table, block_num = page_table.shape
        assert B_block_table == B_q
        assert block_num > 0, f"block num must be greater than 0, got {block_num}"
        assert block_num % (128 / PAGE_SIZE) == 0

        assert q_nope.dtype in (torch.float16, torch.bfloat16, torch.float8_e4m3fn), (
            f"q_nope.dtype needs to be fp16 or bf16 or e4m3 but got {q_nope.dtype}."
        )
        assert q_nope.dtype == q_pe.dtype == kv_c_and_k_pe_cache.dtype
        assert seq_lens.dtype == torch.int32, (
            f"seq_lens.dtype needs to be int32 but got {seq_lens.dtype}."
        )
        assert page_table.dtype == torch.int32, (
            f"page_table.dtype needs to be int32 but got {page_table.dtype}."
        )

        dtype = (
            torch.bfloat16
            if is_quantized_kv_cache(self.kv_cache_dtype)
            else q_nope.dtype
        )
        out = q_nope.new_empty((B_q, MAX_HEADS, D_latent), dtype=dtype)
        lse = (
            torch.empty((B_q, MAX_HEADS), dtype=torch.float32, device=q_nope.device)
            if self.need_to_return_lse_for_decode
            else torch.Tensor()
        )

        ops.sm100_cutlass_mla_decode(
            out,
            lse,
            q_nope,
            q_pe,
            kv_c_and_k_pe_cache,
            seq_lens,
            page_table,
            workspace,
            sm_scale,
            num_kv_splits,
        )

        if H < MAX_HEADS:
            # 提取输出的子集
            lse = lse[:, :H] if self.need_to_return_lse_for_decode else lse
            out = out[:, :H]

        return out, lse

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: MLACommonMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Decode 注意力的前向传播。

        流程：
        1. 检查是否使用了不支持的缩放
        2. 将 q 拆分为 q_nope 和 q_pe
        3. 确保 workspace 大小足够
        4. 调用 _sm100_cutlass_mla_decode 计算注意力
        5. 返回输出和可选的 LSE
        """
        assert kv_c_and_k_pe_cache.numel() > 0
        assert attn_metadata.decode is not None

        if layer._q_scale_float != 1.0 or layer._k_scale_float != 1.0:
            raise NotImplementedError(
                "CutlassMLAImpl does not support scaling for q and kv_latent yet"
            )

        if type(q) is tuple:
            q_nope, q_pe = q
        else:
            q_nope, q_pe = torch.split(
                q, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
            )

        # 调整 workspace 大小（如果需要）
        self._workspace.ensure_size(attn_metadata, self._num_kv_splits)

        # 运行 MLA
        o, lse = self._sm100_cutlass_mla_decode(
            q_nope,
            q_pe,
            kv_c_and_k_pe_cache,
            attn_metadata.decode.seq_lens,
            attn_metadata.decode.block_table,
            self._workspace.get_buf(),
            self.scale,
            self._num_kv_splits,
        )

        return o, (lse if self.need_to_return_lse_for_decode else None)
