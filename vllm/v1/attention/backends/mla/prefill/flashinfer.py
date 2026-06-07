# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashInfer backend for MLA prefill."""
# FlashInfer MLA prefill 后端实现。
#
# FlashInfer 是一个高性能的注意力内核库，专门针对 LLM 推理场景优化。
# 本模块使用 FlashInfer 的 BatchPrefillWithRaggedKVCacheWrapper 来实现
# MLA 的 prefill 注意力计算。
#
# 关键特性：
# 1. 仅支持 Blackwell (SM100) 架构的 GPU
# 2. 要求 DeepSeek R1 的 MLA 维度（qk_nope=128, qk_rope=64, v=128）
# 3. 使用 "NHD" 内存布局和 "cutlass" 后端
# 4. 支持分块上下文（chunked context）计算
# 5. LSE 输出需要从 (q_len, num_heads) 转置为 (num_heads, q_len)

from typing import TYPE_CHECKING

import torch

import vllm.envs as envs
from vllm.v1.attention.backends.mla.prefill.base import MLAPrefillBackend
from vllm.v1.attention.backends.utils import (
    PerLayerParameters,
    get_per_layer_parameters,
    infer_global_hyperparameters,
)
from vllm.v1.worker.workspace import current_workspace_manager

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.model_executor.layers.attention.mla_attention import (
        MLACommonPrefillMetadata,
    )
    from vllm.platforms.interface import DeviceCapability

try:
    from flashinfer import BatchPrefillWithRaggedKVCacheWrapper
except ImportError:
    BatchPrefillWithRaggedKVCacheWrapper = object  # type: ignore[misc,assignment]

# 默认的上下文分块数量。每个分块对应一个独立的 FlashInfer wrapper 实例。
# 分块数量越多，单次计算的序列越短，显存峰值越低。
_DEFAULT_NUM_CHUNKS = 32


class FlashInferPrefillBackend(MLAPrefillBackend):
    """FlashInfer backend for MLA prefill.
    基于 FlashInfer 库的 MLA prefill 后端。
    使用 BatchPrefillWithRaggedKVCacheWrapper 进行批量变长 prefill 计算。
    """

    # 要求模型具有 DeepSeek R1 的 MLA 维度配置
    requires_r1_mla_dimensions = True

    @staticmethod
    def get_name() -> str:
        """返回后端名称 "FLASHINFER"。"""
        return "FLASHINFER"

    @classmethod
    def supports_compute_capability(cls, device_capability: "DeviceCapability") -> bool:
        """仅支持 Blackwell 架构（SM100，计算能力主版本号为 10）。"""
        return device_capability.major == 10

    @classmethod
    def is_available(cls) -> bool:
        """检查 flashinfer 库是否已安装。"""
        try:
            from flashinfer import (
                BatchPrefillWithRaggedKVCacheWrapper,  # noqa: F401
            )

            return True
        except ImportError:
            return False

    def __init__(
        self,
        num_heads: int,
        scale: float,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        vllm_config: "VllmConfig",
    ) -> None:
        """初始化 FlashInferPrefillBackend。

        初始化过程：
        1. 调用父类构造函数保存 MLA 维度参数
        2. 初始化主 prefill wrapper 和分块 wrapper 列表（延迟到 prepare_metadata）
        3. 获取全局超参数缓存（延迟到首次使用时）
        4. 从 WorkspaceManager 获取 FlashInfer 工作空间缓冲区

        Args:
            num_heads: Q 的注意力头数
            scale: 注意力缩放因子
            kv_lora_rank: KV 的低秩压缩维度
            qk_nope_head_dim: 不带 RoPE 的 Q/K 头维度
            qk_rope_head_dim: 带 RoPE 的 Q/K 头维度
            v_head_dim: V 的头维度
            vllm_config: vLLM 全局配置
        """
        super().__init__(
            num_heads=num_heads,
            scale=scale,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            vllm_config=vllm_config,
        )

        # 主 prefill wrapper（用于新 token 的因果注意力）
        self._prefill_main: BatchPrefillWithRaggedKVCacheWrapper | None = None
        # 分块上下文 prefill wrapper 列表（每个分块一个）
        self._prefill_chunks: list[BatchPrefillWithRaggedKVCacheWrapper] = []
        # 全局注意力超参数（如 softmax 缩放、窗口大小、logits 软上限等）
        self._global_hyperparameters: PerLayerParameters | None = None
        # FlashInfer 工作空间缓冲区，用于存储中间结果
        (self._workspace_buffer,) = current_workspace_manager().get_simultaneous(
            ((envs.VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE,), torch.uint8),
        )

    def _ensure_chunks(
        self,
        num_chunks: int,
        workspace_buffer: torch.Tensor,
    ) -> None:
        """确保有足够的分块 wrapper 实例。

        如果当前分块数量不足，则创建新的 BatchPrefillWithRaggedKVCacheWrapper。

        Args:
            num_chunks: 需要的分块数量
            workspace_buffer: FlashInfer 工作空间缓冲区
        """
        if len(self._prefill_chunks) < num_chunks:
            for _ in range(len(self._prefill_chunks), num_chunks):
                self._prefill_chunks.append(
                    BatchPrefillWithRaggedKVCacheWrapper(
                        workspace_buffer, "NHD", backend="cutlass"
                    )
                )

    def _resolve_global_hyperparameters(self) -> PerLayerParameters:
        """解析并缓存全局注意力超参数。

        从模型的前向上下文中获取所有 MLA 注意力层的参数，
        然后推断出统一的全局超参数（如 softmax 缩放因子、
        滑动窗口大小、logits 软上限等）。

        Returns:
            全局注意力超参数
        """
        if self._global_hyperparameters is not None:
            return self._global_hyperparameters

        from vllm.model_executor.layers.attention.mla_attention import (
            MLAAttention,
            MLACommonImpl,
        )

        # 从静态前向上下文中获取所有 MLA 注意力层
        forward_context = self.vllm_config.compilation_config.static_forward_context
        layer_names = [
            name
            for name, layer in forward_context.items()
            if isinstance(layer, MLAAttention)
        ]

        # 推断全局超参数（所有层共享相同参数时才会成功）
        self._global_hyperparameters = infer_global_hyperparameters(
            get_per_layer_parameters(
                self.vllm_config,
                layer_names,
                MLACommonImpl,  # type: ignore[type-abstract]
            )
        )
        return self._global_hyperparameters

    def prepare_metadata(
        self,
        prefill_metadata: "MLACommonPrefillMetadata",
    ) -> None:
        """准备 FlashInfer 的注意力计算计划（plan）。

        FlashInfer 需要在实际计算前调用 plan() 方法来预处理元数据，
        包括设置 Q/K 的累积序列长度、头数量、头维度等参数。

        对于新 token 的因果注意力，使用 _prefill_main wrapper；
        对于上下文分块的非因果注意力，使用 _prefill_chunks 列表中的 wrapper。

        Args:
            prefill_metadata: MLA prefill 的通用元数据
        """
        global_hyperparameters = self._resolve_global_hyperparameters()
        qo_indptr = prefill_metadata.query_start_loc
        has_context = prefill_metadata.chunked_context is not None
        # 延迟初始化主 prefill wrapper
        if self._prefill_main is None:
            self._prefill_main = BatchPrefillWithRaggedKVCacheWrapper(
                self._workspace_buffer, "NHD", backend="cutlass"
            )
            self._ensure_chunks(_DEFAULT_NUM_CHUNKS, self._workspace_buffer)

        # 如果有上下文分块，确保有足够的 chunk wrapper
        if has_context:
            chunked_context = prefill_metadata.chunked_context
            assert chunked_context is not None
            num_chunks = chunked_context.cu_seq_lens.shape[0]
            self._ensure_chunks(num_chunks, self._workspace_buffer)

        num_qo_heads = self.num_heads
        num_kv_heads = num_qo_heads  # MLA 中 Q 和 KV 的头数相同

        # MLA 的 Q/K 总维度 = 不带 RoPE 的维度 + 带 RoPE 的维度
        head_dim_qk = self.qk_nope_head_dim + self.qk_rope_head_dim
        head_dim_vo = self.v_head_dim
        # MLA 中 K 和 Q 共享相同的累积序列长度（ragged KV cache 模式）
        kv_indptr = qo_indptr.clone()

        # 为主 prefill wrapper 制定注意力计算计划
        assert self._prefill_main is not None
        self._prefill_main.plan(
            qo_indptr=qo_indptr,
            kv_indptr=kv_indptr,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim_qk,
            head_dim_vo=head_dim_vo,
            causal=True,  # 新 token 使用因果掩码
            sm_scale=global_hyperparameters.sm_scale,
            window_left=global_hyperparameters.window_left,
            logits_soft_cap=global_hyperparameters.logits_soft_cap,
            q_data_type=prefill_metadata.q_data_type,
            o_data_type=prefill_metadata.output_dtype,
        )

        # 为每个上下文分块制定非因果注意力计算计划
        if has_context:
            chunked_context = prefill_metadata.chunked_context
            assert chunked_context is not None
            for i in range(num_chunks):
                kv_indptr_chunk = chunked_context.cu_seq_lens[i]

                self._prefill_chunks[i].plan(
                    qo_indptr=qo_indptr,
                    kv_indptr=kv_indptr_chunk,
                    num_qo_heads=num_qo_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim_qk=head_dim_qk,
                    head_dim_vo=head_dim_vo,
                    causal=False,  # 上下文分块不需要因果掩码
                    sm_scale=global_hyperparameters.sm_scale,
                    window_left=global_hyperparameters.window_left,
                    logits_soft_cap=global_hyperparameters.logits_soft_cap,
                    q_data_type=prefill_metadata.q_data_type,
                    o_data_type=prefill_metadata.output_dtype,
                )

    def run_prefill_new_tokens(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        return_softmax_lse: bool,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """对新 token 执行因果注意力 prefill。

        使用 _prefill_main wrapper 执行计划好的因果注意力计算。

        Args:
            q: 查询张量
            k: 键张量
            v: 值张量
            return_softmax_lse: 是否返回 lse

        Returns:
            注意力输出或 (输出, lse) 元组。
            注意：FlashInfer 返回的 lse 形状为 (q_len, num_heads)，
            需要转置为 (num_heads, q_len) 以保持一致性。
        """
        assert self._prefill_main is not None

        ret = self._prefill_main.run(
            q=q,
            k=k,
            v=v,
            return_lse=return_softmax_lse,
        )

        if isinstance(ret, tuple):
            # Convert from (q_len, num_heads) to (num_heads, q_len)
            # FlashInfer 输出 lse 为 (q_len, num_heads) 格式，
            # 需要转置为 (num_heads, q_len) 与其他后端保持一致
            return ret[0], ret[1].transpose(0, 1).contiguous()
        return ret

    def run_prefill_context_chunk(
        self,
        chunk_idx: int,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """对上下文的一个分块执行非因果注意力计算。

        使用 _prefill_chunks[chunk_idx] 对应的 wrapper 执行计算。

        Args:
            chunk_idx: 上下文分块索引
            q: 查询张量
            k: 键张量（从 KV 缓存中取出的当前分块）
            v: 值张量（从 KV 缓存中取出的当前分块）

        Returns:
            (注意力输出, softmax_lse) 元组，lse 形状为 (num_heads, q_len)
        """
        attn_out, lse = self._prefill_chunks[chunk_idx].run(
            q=q,
            k=k,
            v=v,
            return_lse=True,
        )

        # Convert from (q_len, num_heads) to (num_heads, q_len)
        return attn_out, lse.transpose(0, 1).contiguous()
