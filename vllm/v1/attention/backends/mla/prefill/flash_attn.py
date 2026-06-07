# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashAttention backend for MLA prefill."""
# FlashAttention MLA prefill 后端实现。
#
# 本模块使用 FlashAttention 库（flash_attn_varlen_func）来实现 MLA 的
# prefill 注意力计算。FlashAttention 是一种 IO 感知的精确注意力算法，
# 通过分块计算和在线 softmax 技术，将注意力的显存复杂度从 O(N^2)
# 降低到 O(N)，同时保持数值精确性。
#
# 关键特性：
# 1. 支持不同 Q/K 和 V 的头维度（通过 V 填充实现）
# 2. 支持 vllm_flash_attn 和上游 flash_attn 两种变体
# 3. 支持 CUDA Graph 捕获模式下的确定性行为
# 4. 支持上下文分块计算（chunked context）用于长序列

import functools
from typing import TYPE_CHECKING

import torch

import vllm.envs as envs
from vllm.platforms import current_platform
from vllm.v1.attention.backends.fa_utils import (
    get_flash_attn_version,
    is_flash_attn_varlen_func_available,
)
from vllm.v1.attention.backends.mla.prefill.base import MLAPrefillBackend

if TYPE_CHECKING:
    from vllm.config import VllmConfig

if is_flash_attn_varlen_func_available():
    from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func
else:
    flash_attn_varlen_func = None  # type: ignore[assignment]


class FlashAttnPrefillBackend(MLAPrefillBackend):
    """FlashAttention backend for MLA prefill.
    基于 FlashAttention 库的 MLA prefill 后端。
    使用 flash_attn_varlen_func 处理变长序列的注意力计算。
    """

    @staticmethod
    def get_name() -> str:
        """返回后端名称 "FLASH_ATTN"。"""
        return "FLASH_ATTN"

    @classmethod
    def is_available(cls) -> bool:
        """检查 flash_attn_varlen_func 是否可用。
        只有当 flash_attn 库已安装且版本支持 varlen 函数时才返回 True。
        """
        return is_flash_attn_varlen_func_available()

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
        """初始化 FlashAttnPrefillBackend。

        主要初始化逻辑：
        1. 调用父类构造函数保存 MLA 维度参数
        2. 确定使用哪个版本的 flash_attn_varlen_func
        3. 判断是否需要对 V 进行填充（padding）
        4. 判断是否使用 vllm 的 FA 实现（区分 CUDA/XPU 与 ROCm）

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

        # Handle the differences between the flash_attn_varlen from
        # flash_attn and the one from vllm_flash_attn
        # 处理 flash_attn 库和 vllm_flash_attn 库之间的差异。
        # vllm_flash_attn 是 vLLM 内置的 FlashAttention 变体。
        assert flash_attn_varlen_func is not None, (
            "FlashAttnPrefillBackend requires flash_attn_varlen_func. "
            "Ensure FlashAttnPrefillBackend.is_available() is checked first."
        )
        qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.flash_attn_varlen_func = flash_attn_varlen_func
        self.vllm_flash_attn_version = get_flash_attn_version(head_size=qk_head_dim)
        if self.vllm_flash_attn_version is not None:
            # 如果使用 vllm_flash_attn，通过 functools.partial 预绑定 fa_version 参数
            self.flash_attn_varlen_func = functools.partial(
                flash_attn_varlen_func, fa_version=self.vllm_flash_attn_version
            )

        # Determine if we need to pad V
        # 判断是否需要对 V 进行零填充。
        # MLA 中 V 的头维度（v_head_dim）通常小于 Q/K 的头维度，
        # 但某些 FlashAttention 版本不支持不同的 Q/K 和 V 头维度。
        # 解决方案：用零将 V 填充到与 Q/K 相同的维度。
        # FA3 on Hopper (SM90) 和 FA4 原生支持不同的头维度，无需填充。
        device_capability = current_platform.get_device_capability()
        self.requires_v_padding = self.vllm_flash_attn_version is None or not (
            (
                self.vllm_flash_attn_version == 3
                and device_capability is not None
                and device_capability[0] == 9
            )
            or self.vllm_flash_attn_version == 4
        )

        # Track whether we're using vllm's FA or upstream (for ROCm)
        # 记录使用的是 vllm 的 FA 实现还是上游 FA（用于 ROCm）。
        # ROCm 使用上游 flash_attn，其参数名称有所不同。
        self._is_vllm_fa = current_platform.is_cuda() or current_platform.is_xpu()

    def _flash_attn_varlen_diff_headdims(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        return_softmax_lse: bool = False,
        softmax_scale: float | None = None,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """调用 flash_attn_varlen_func 并处理 Q/K 与 V 头维度不同的情况。

        这是 FlashAttn 后端的核心内部方法，处理以下差异：
        1. V 维度填充/反填充（当 V 头维度 < Q/K 头维度时）
        2. vllm FA 和上游 FA 的参数差异（return_softmax_lse vs return_attn_probs）
        3. 批次不变性模式（VLLM_BATCH_INVARIANT）下设置 num_splits=1

        Args:
            q: 查询张量
            k: 键张量
            v: 值张量
            return_softmax_lse: 是否返回 softmax 的 log-sum-exp 值
            softmax_scale: 注意力缩放因子，None 时使用默认值
            **kwargs: 传递给 flash_attn_varlen_func 的额外参数

        Returns:
            注意力输出张量，或 (输出, lse) 元组
        """
        maybe_padded_v = v
        # 如果需要填充 V：在最后一维补零，使 V 的头维度与 Q/K 对齐
        if self.requires_v_padding:
            maybe_padded_v = torch.nn.functional.pad(
                v, [0, q.shape[-1] - v.shape[-1]], value=0
            )

        # 根据 FA 实现选择正确的参数名
        if self._is_vllm_fa:
            kwargs["return_softmax_lse"] = return_softmax_lse
        else:
            # ROCm leverages the upstream flash_attn, which takes a parameter
            # called "return_attn_probs" instead of return_softmax_lse
            # ROCm 使用上游 flash_attn，参数名是 return_attn_probs
            kwargs["return_attn_probs"] = return_softmax_lse
        if envs.VLLM_BATCH_INVARIANT:
            # 批次不变性模式：固定 num_splits=1 以确保确定性输出
            kwargs["num_splits"] = 1

        attn_out = self.flash_attn_varlen_func(
            q=q,
            k=k,
            v=maybe_padded_v,
            softmax_scale=softmax_scale,
            **kwargs,
        )

        # Unpack the output if there are multiple results
        # 如果返回了多个结果（输出 + lse），解包
        lse = None
        if isinstance(attn_out, tuple):
            attn_out, lse = attn_out[0], attn_out[1]

        # Unpad output back to v_head_dim if we padded V
        # 如果之前填充了 V，现在裁剪输出回到原始 v_head_dim 维度
        if self.requires_v_padding:
            attn_out = attn_out[..., : v.shape[-1]]

        # Remain consistent with old `flash_attn_varlen_func` where there
        # is only one output tensor if `return_softmax_lse` is False.
        # 保持与旧版 flash_attn_varlen_func 的一致性：
        # 不需要 lse 时只返回输出张量
        if return_softmax_lse:
            return attn_out, lse
        return attn_out

    def run_prefill_new_tokens(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        return_softmax_lse: bool,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """对新 token 执行因果注意力 prefill。

        使用 query_start_loc 作为 Q 和 K 的累积序列长度（cu_seqlens），
        表示这是一个纯 prefill 场景（所有 K/V 都是当前批次中的新 token）。
        启用因果掩码（causal=True），确保每个 token 只能看到它之前的 token。

        Args:
            q: 查询张量
            k: 键张量
            v: 值张量
            return_softmax_lse: 是否返回 lse

        Returns:
            注意力输出或 (输出, lse) 元组
        """
        return self._flash_attn_varlen_diff_headdims(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=self._prefill_metadata.query_start_loc,
            cu_seqlens_k=self._prefill_metadata.query_start_loc,
            max_seqlen_q=self._prefill_metadata.max_query_len,
            max_seqlen_k=self._prefill_metadata.max_query_len,
            softmax_scale=self.scale,
            causal=True,
            return_softmax_lse=return_softmax_lse,
        )

    def run_prefill_context_chunk(
        self,
        chunk_idx: int,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """对上下文的一个分块执行非因果注意力计算。

        上下文分块是指将已经缓存的 KV 序列分成多个块来逐块计算注意力。
        注意：上下文分块使用非因果掩码（causal=False），因为当前 query
        需要看到整个上下文块中的所有 token。

        Args:
            chunk_idx: 上下文分块索引
            q: 查询张量
            k: 键张量（从缓存中取出的当前分块）
            v: 值张量（从缓存中取出的当前分块）

        Returns:
            (注意力输出, softmax_lse) 元组，用于后续跨分块合并
        """
        assert self._prefill_metadata.chunked_context is not None
        return self._flash_attn_varlen_diff_headdims(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=self._prefill_metadata.query_start_loc,
            cu_seqlens_k=self._prefill_metadata.chunked_context.cu_seq_lens[chunk_idx],
            max_seqlen_q=self._prefill_metadata.max_query_len,
            max_seqlen_k=self._prefill_metadata.chunked_context.max_seq_lens[chunk_idx],
            softmax_scale=self.scale,
            causal=False,  # Context is unmasked
            return_softmax_lse=True,
        )
