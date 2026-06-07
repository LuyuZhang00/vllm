# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TRT-LLM Ragged backend for MLA prefill."""
# TRT-LLM Ragged MLA prefill 后端实现。
#
# 本模块使用 TensorRT-LLM 的 ragged attention 实现来执行 MLA prefill。
# "Ragged" 意味着序列不需要填充到相同长度，每个序列可以有不同的长度。
# 该内核通过 flashinfer 库中的 trtllm_ragged_attention_deepseek 函数调用。
#
# 关键特性：
# 1. 仅支持 Blackwell (SM100) 架构的 GPU
# 2. 要求 DeepSeek R1 的 MLA 维度（qk_nope=128, qk_rope=64, v=128）
# 3. 使用 FlashInfer 的工作空间缓冲区管理内存
# 4. 需要预分配输出张量（out 参数）
# 5. LSE 输出需要从 (q_len, num_heads) 转置为 (num_heads, q_len)

from typing import TYPE_CHECKING

import torch

import vllm.envs as envs
from vllm.v1.attention.backends.mla.prefill.base import MLAPrefillBackend
from vllm.v1.worker.workspace import current_workspace_manager

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.model_executor.layers.attention.mla_attention import (
        MLACommonPrefillMetadata,
    )
    from vllm.platforms.interface import DeviceCapability


class TrtllmRaggedPrefillBackend(MLAPrefillBackend):
    """TRT-LLM Ragged backend for MLA prefill.
    基于 TRT-LLM ragged attention 的 MLA prefill 后端。
    使用 flashinfer 库中的 trtllm_ragged_attention_deepseek 内核。
    """

    # 要求模型具有 DeepSeek R1 的 MLA 维度配置
    requires_r1_mla_dimensions = True

    @staticmethod
    def get_name() -> str:
        """返回后端名称 "TRTLLM_RAGGED"。"""
        return "TRTLLM_RAGGED"

    @classmethod
    def supports_compute_capability(cls, device_capability: "DeviceCapability") -> bool:
        """仅支持 Blackwell 架构（SM100，计算能力主版本号为 10）。"""
        return device_capability.major == 10

    @classmethod
    def is_available(cls) -> bool:
        """检查 flashinfer 中的 trtllm_ragged_attention_deepseek 是否可用。"""
        try:
            from flashinfer.prefill import (
                trtllm_ragged_attention_deepseek,  # noqa: F401
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
        """初始化 TrtllmRaggedPrefillBackend。

        初始化过程：
        1. 调用父类构造函数保存 MLA 维度参数
        2. 从 WorkspaceManager 获取工作空间缓冲区

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
        # 获取 FlashInfer 工作空间缓冲区
        (self._workspace_buffer,) = current_workspace_manager().get_simultaneous(
            (
                (envs.VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE,),
                torch.uint8,
            ),
        )

    def prepare_metadata(
        self,
        prefill_metadata: "MLACommonPrefillMetadata",
    ) -> None:
        """准备元数据，计算每个请求的序列长度。

        从累积序列长度（query_start_loc）计算每个请求的实际序列长度。

        Args:
            prefill_metadata: MLA prefill 的通用元数据
        """
        super().prepare_metadata(prefill_metadata)
        # 从累积序列长度差分计算每个请求的序列长度
        self._query_seq_lens = (
            prefill_metadata.query_start_loc[1:] - prefill_metadata.query_start_loc[:-1]
        )

    def run_prefill_new_tokens(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        return_softmax_lse: bool,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """对新 token 执行因果注意力 prefill。

        使用 trtllm_ragged_attention_deepseek 内核。需要预分配输出张量，
        因为该内核是 out-of-place 操作。

        参数说明：
        - bmm1_scale: 第一次矩阵乘法的缩放因子（Q*K^T 的 softmax_scale）
        - bmm2_scale: 第二次矩阵乘法的缩放因子（通常为 1.0）
        - o_sf_scale: 输出的缩放因子（通常为 1.0）
        - window_left: 滑动窗口大小（-1 表示无窗口限制）
        - enable_pdl: 是否启用 PDL（Programmatic Dependent Launch）

        Args:
            q: 查询张量
            k: 键张量
            v: 值张量
            return_softmax_lse: 是否返回 lse

        Returns:
            注意力输出或 (输出, lse) 元组
        """
        from flashinfer.prefill import trtllm_ragged_attention_deepseek

        # 预分配输出张量
        out = torch.empty(
            q.shape[0],
            q.shape[1],
            v.shape[2],
            device=q.device,
            dtype=self._prefill_metadata.output_dtype,
        )

        ret = trtllm_ragged_attention_deepseek(
            query=q,
            key=k,
            value=v,
            workspace_buffer=self._workspace_buffer,
            seq_lens=self._query_seq_lens,
            max_q_len=self._prefill_metadata.max_query_len,
            max_kv_len=self._prefill_metadata.max_query_len,
            bmm1_scale=self.scale,
            bmm2_scale=1.0,
            o_sf_scale=1.0,
            batch_size=self._query_seq_lens.shape[0],
            window_left=-1,
            cum_seq_lens_q=self._prefill_metadata.query_start_loc,
            cum_seq_lens_kv=self._prefill_metadata.query_start_loc,
            enable_pdl=False,
            is_causal=True,
            return_lse=return_softmax_lse,
            out=out,
        )

        if isinstance(ret, tuple):
            # Convert from (q_len, num_heads) to (num_heads, q_len)
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

        与新 token 的 prefill 类似，但使用非因果掩码且 K/V 的序列长度
        来自分块上下文元数据。

        Args:
            chunk_idx: 上下文分块索引
            q: 查询张量
            k: 键张量
            v: 值张量

        Returns:
            (注意力输出, softmax_lse) 元组
        """
        from flashinfer.prefill import trtllm_ragged_attention_deepseek

        assert self._prefill_metadata.chunked_context is not None
        assert self._prefill_metadata.chunked_context.seq_lens[chunk_idx] is not None

        # 预分配输出张量
        out = torch.empty(
            q.shape[0],
            q.shape[1],
            v.shape[2],
            device=q.device,
            dtype=self._prefill_metadata.output_dtype,
        )

        attn_out, lse = trtllm_ragged_attention_deepseek(
            query=q,
            key=k,
            value=v,
            workspace_buffer=self._workspace_buffer,
            seq_lens=self._prefill_metadata.chunked_context.seq_lens[chunk_idx],
            max_q_len=self._prefill_metadata.max_query_len,
            max_kv_len=self._prefill_metadata.chunked_context.max_seq_lens[chunk_idx],
            bmm1_scale=self.scale,
            bmm2_scale=1.0,
            o_sf_scale=1.0,
            batch_size=self._prefill_metadata.chunked_context.seq_lens[chunk_idx].shape[
                0
            ],
            window_left=-1,
            cum_seq_lens_q=self._prefill_metadata.query_start_loc,
            cum_seq_lens_kv=self._prefill_metadata.chunked_context.cu_seq_lens[
                chunk_idx
            ],
            enable_pdl=False,
            is_causal=False,
            return_lse=True,
            out=out,
        )

        # Convert from (q_len, num_heads) to (num_heads, q_len)
        return attn_out, lse.transpose(0, 1).contiguous()
