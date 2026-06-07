# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TokenSpeed CuTe DSL backend for MLA prefill."""
# TokenSpeed CuTe DSL MLA prefill 后端实现。
#
# TokenSpeed MLA 是一个使用 NVIDIA CuTe (Compositional Tensors) DSL 编写的
# 高性能 MLA prefill 内核。CuTe 是 CUTLASS 库中的张量抽象层，提供
# 细粒度的内存布局控制和高效的 GPU 计算。
#
# 关键特性：
# 1. 仅支持 Blackwell (SM100) 架构的 GPU
# 2. 要求 DeepSeek R1 的 MLA 维度（qk_nope=128, qk_rope=64, v=128）
# 3. 支持 BF16 和 FP8 两种 Q 数据类型的预热 JIT 编译
# 4. v 参数需要显式确保连续性（contiguous）
# 5. LSE 输出需要从 (q_len, num_heads) 转置为 (num_heads, q_len)
#
# 依赖：
# - tokenspeed_mla 包（通过 `uv pip install tokenspeed-mla` 安装）

from typing import TYPE_CHECKING

import torch

from vllm.v1.attention.backends.mla.prefill.base import MLAPrefillBackend

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.model_executor.layers.attention.mla_attention import (
        MLACommonPrefillMetadata,
    )
    from vllm.platforms.interface import DeviceCapability


class TokenspeedMLAPrefillBackend(MLAPrefillBackend):
    """TokenSpeed CuTe DSL backend for MLA prefill.
    基于 TokenSpeed CuTe DSL 的 MLA prefill 后端。
    使用 JIT 编译的内核进行高性能 MLA 注意力计算。
    """

    # 要求模型具有 DeepSeek R1 的 MLA 维度配置
    requires_r1_mla_dimensions = True

    @staticmethod
    def get_name() -> str:
        """返回后端名称 "TOKENSPEED_MLA"。"""
        return "TOKENSPEED_MLA"

    @classmethod
    def supports_compute_capability(cls, device_capability: "DeviceCapability") -> bool:
        """仅支持 Blackwell 架构（SM100，计算能力主版本号为 10）。"""
        return device_capability.major == 10

    # 安装提示信息，当用户显式选择此后端但未安装依赖时显示
    _INSTALL_HINT = (
        "tokenspeed_mla package is not installed. "
        "Install it with: `uv pip install tokenspeed-mla`"
    )

    @classmethod
    def is_available(cls) -> bool:
        """检查 tokenspeed_mla 包是否已安装。"""
        try:
            from tokenspeed_mla import (
                tokenspeed_mla_prefill,  # noqa: F401
            )

            return True
        except ImportError:
            return False

    @classmethod
    def validate_configuration(
        cls,
        device_capability,
        selector_config,
    ) -> list[str]:
        """验证配置并提供友好的安装提示。

        覆盖基类方法，将通用的 "required dependencies not available" 消息
        替换为具体的安装命令提示，帮助用户快速定位和解决问题。
        """
        # Replace the generic "required dependencies not available" message
        # from the base class with a specific install hint so users know
        # exactly which package to install when they explicitly select this
        # backend without having tokenspeed_mla installed.
        reasons = super().validate_configuration(device_capability, selector_config)
        return [
            cls._INSTALL_HINT if r == "required dependencies not available" else r
            for r in reasons
        ]

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
        """初始化 TokenspeedMLAPrefillBackend。

        初始化过程：
        1. 调用父类构造函数保存 MLA 维度参数
        2. 预热 JIT 编译 BF16 和 FP8 两种数据类型的 prefill 内核

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

        # Pre-JIT BF16 and FP8 prefill kernels. Idempotent — also called from
        # TokenspeedMLAImpl.__init__; second call is a no-op.
        # 预热 JIT 编译 BF16 和 FP8 两种数据类型的 prefill 内核。
        # 这是幂等操作 — 也可以从 TokenspeedMLAImpl.__init__ 调用，
        # 第二次调用是无操作（no-op）。
        from tokenspeed_mla import warmup_compile_prefill

        for q_dtype in (torch.bfloat16, torch.float8_e4m3fn):
            warmup_compile_prefill(
                q_dtype=q_dtype,
                d_qk=qk_nope_head_dim + qk_rope_head_dim,
                d_v=v_head_dim,
                enable_pdl=False,
            )

    def prepare_metadata(
        self,
        prefill_metadata: "MLACommonPrefillMetadata",
    ) -> None:
        """准备元数据，计算每个请求的序列长度。

        内核签名需要 seq_lens 参数（虽然实现中从未读取它，因为
        每个批次的实际长度是从 cum_seq_lens 的差分计算得到的）。
        这里计算它是为了与 trtllm_ragged 后端保持一致。

        注意：CUDA Graph 模式下的 query_start_loc 填充会导致末尾
        差分为 0，填充的批次在内核中是空操作（no-op）。

        Args:
            prefill_metadata: MLA prefill 的通用元数据
        """
        super().prepare_metadata(prefill_metadata)
        # Kernel signature requires `seq_lens` but the implementation never reads
        # it (per-batch lengths are derived from `cum_seq_lens` diffs); compute
        # for parity with trtllm_ragged. cuda-graph padding in
        # `query_start_loc` is saturated to `total_num_tokens`
        # (gpu_model_runner.py:1905), so trailing diffs are 0 and padded batches
        # are kernel no-ops — same reason trtllm passes the padded length as
        # batch_size directly.
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

        调用 tokenspeed_mla_prefill 内核进行计算。注意 v 张量可能是
        非连续的（来自 mla_attention.forward_mha 中的 split 操作），
        需要先调用 .contiguous() 确保内存连续性，避免内核中的隐式拷贝。

        Args:
            q: 查询张量
            k: 键张量
            v: 值张量（可能是非连续的）
            return_softmax_lse: 是否返回 lse

        Returns:
            注意力输出或 (输出, lse) 元组
        """
        from tokenspeed_mla import tokenspeed_mla_prefill

        # `v` arrives as the second half of `kv_nope.split(...)` in
        # mla_attention.forward_mha — a non-contiguous view of `kv_nope` along
        # dim=-1. The kernel does `v.reshape(1, total_kv, h_k, 1, d_v)` which
        # would silently copy on a non-contiguous tensor; force contiguity here
        # so the copy (if any) happens once outside the kernel call.
        # v 来自 mla_attention.forward_mha 中 kv_nope.split(...) 的第二半，
        # 是沿最后一维的非连续视图。内核会对 v 执行 reshape 操作，
        # 如果 v 不连续会导致隐式拷贝。这里提前确保连续性。
        v = v.contiguous()

        ret = tokenspeed_mla_prefill(
            query=q,
            key=k,
            value=v,
            seq_lens=self._query_seq_lens,
            cum_seq_lens=self._prefill_metadata.query_start_loc,
            max_seq_len=self._prefill_metadata.max_query_len,
            batch_size=self._query_seq_lens.shape[0],
            softmax_scale=self.scale,
            is_causal=True,
            return_lse=return_softmax_lse,
            enable_pdl=False,
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

        使用 tokenspeed_mla_prefill 内核，设置 is_causal=False 以允许
        查询看到整个上下文块中的所有 token。

        Args:
            chunk_idx: 上下文分块索引
            q: 查询张量
            k: 键张量
            v: 值张量

        Returns:
            (注意力输出, softmax_lse) 元组
        """
        from tokenspeed_mla import tokenspeed_mla_prefill

        assert self._prefill_metadata.chunked_context is not None
        chunked = self._prefill_metadata.chunked_context

        # See note in run_prefill_new_tokens — `v` is a split-view of `kv_nope`
        # in `_compute_prefill_context` and arrives non-contiguous.
        # 同 run_prefill_new_tokens 中的说明，v 可能是非连续的
        v = v.contiguous()

        attn_out, lse = tokenspeed_mla_prefill(
            query=q,
            key=k,
            value=v,
            seq_lens=chunked.seq_lens[chunk_idx],
            cum_seq_lens=chunked.cu_seq_lens[chunk_idx],
            max_seq_len=chunked.max_seq_lens[chunk_idx],
            batch_size=chunked.seq_lens[chunk_idx].shape[0],
            softmax_scale=self.scale,
            is_causal=False,
            return_lse=True,
            cum_seq_lens_q=self._prefill_metadata.query_start_loc,
            max_seq_len_q=self._prefill_metadata.max_query_len,
            enable_pdl=False,
        )

        # Convert from (q_len, num_heads) to (num_heads, q_len)
        return attn_out, lse.transpose(0, 1).contiguous()
