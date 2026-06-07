# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
TokenSpeed CuTe DSL MLA decode 后端模块（Blackwell GPU，仅 FP8 KV 缓存）。

本模块实现了基于 TokenSpeed CuTe DSL 的 MLA decode 注意力后端，
专为 NVIDIA Blackwell（SM10.x）GPU 上的 DeepSeek R1 模型优化。

核心特性：
1. 使用 TokenSpeed 的 CuTe DSL 内核（tokenspeed_mla_decode）
2. 仅支持 FP8 KV 缓存（fp8, fp8_e4m3）
3. 形状特化：仅支持 DeepSeek R1 的 MLA 维度
   （qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128）
4. 使用 FP8 query（supports_quant_query_input=True）
5. 支持 MTP（Multi-Token Prediction）多 token decode

与其他 MLA 后端的区别：
- 仅支持 FP8 KV 缓存
- 形状特化，性能最优但兼容性有限
- 需要安装 tokenspeed_mla 包
"""

from typing import ClassVar

import torch

from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.mla_attention import (
    MLACommonBackend,
    MLACommonImpl,
    MLACommonMetadata,
    MLACommonMetadataBuilder,
    QueryLenSupport,
)
from vllm.platforms.interface import DeviceCapability
from vllm.utils.torch_utils import is_quantized_kv_cache
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    AttentionLayer,
    AttentionType,
    MultipleOf,
)
from vllm.v1.attention.backends.utils import KVCacheLayoutType

logger = init_logger(__name__)

# tokenspeed_mla_decode 的 workspace 上界（每设备，延迟分配）：
#   num_sms * num_heads * MAX_Q_LEN * (kv_lora_rank + 1) * sizeof(float32)
# 匹配内核的 `get_workspace_size` 公式。MAX_Q_LEN=8 覆盖到
# EAGLE3 / MTP-2 spec decoding 的 query 长度；更大的 q_len 会触发
# 内核自己的 buffer 检查失败。
_TOKENSPEED_MAX_Q_LEN = 8

_g_workspace: dict[torch.device, torch.Tensor] = {}


def _get_workspace(
    device: torch.device, num_heads: int, kv_lora_rank: int
) -> torch.Tensor:
    """获取或创建 TokenSpeed MLA 的 workspace 缓冲区。"""
    from tokenspeed_mla import get_num_sm

    needed = (
        get_num_sm(device) * num_heads * _TOKENSPEED_MAX_Q_LEN * (kv_lora_rank + 1) * 4
    )
    existing = _g_workspace.get(device)
    if existing is None or existing.numel() < needed:
        _g_workspace[device] = torch.empty(needed, dtype=torch.int8, device=device)
    return _g_workspace[device]


class TokenspeedMLAMetadataBuilder(MLACommonMetadataBuilder[MLACommonMetadata]):
    """TokenSpeed MLA 的元数据构建器。"""
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH
    query_len_support: ClassVar[QueryLenSupport] = QueryLenSupport.UNIFORM


class TokenspeedMLABackend(MLACommonBackend):
    """
    TokenSpeed MLA 注意力后端。

    定义了后端的基本属性和支持的配置。
    仅支持 FP8 KV 缓存和 Blackwell GPU。
    """
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "fp8",
        "fp8_e4m3",
    ]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        """支持的块大小：32 和 64。"""
        return [32, 64]

    @staticmethod
    def get_name() -> str:
        return "TOKENSPEED_MLA"

    @staticmethod
    def get_impl_cls() -> type["TokenspeedMLAImpl"]:
        return TokenspeedMLAImpl

    @staticmethod
    def get_builder_cls() -> type["TokenspeedMLAMetadataBuilder"]:
        return TokenspeedMLAMetadataBuilder

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        """仅支持 Blackwell（SM10.x）GPU。"""
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
        # 在前面提供清晰的安装提示，而不是让 ModuleNotFoundError
        # 在第一次请求时 deep inside `forward_mqa` 中触发。
        try:
            import tokenspeed_mla  # noqa: F401
        except ImportError:
            return (
                "tokenspeed_mla package is not installed. "
                "Install it with: `uv pip install tokenspeed-mla`"
            )

        # tokenspeed_mla CuTe DSL 内核形状特化为 DeepSeek R1 MLA 维度
        #（qk_nope=128, qk_rope=64, v=128）。拒绝其他任何维度。
        from vllm.config import get_current_vllm_config

        vllm_config = get_current_vllm_config()
        if vllm_config.model_config is not None:
            hf_text_config = vllm_config.model_config.hf_text_config
            qk_nope_head_dim = getattr(hf_text_config, "qk_nope_head_dim", 0)
            qk_rope_head_dim = getattr(hf_text_config, "qk_rope_head_dim", 0)
            v_head_dim = getattr(hf_text_config, "v_head_dim", 0)
            if qk_nope_head_dim != 128 or qk_rope_head_dim != 64 or v_head_dim != 128:
                return (
                    "tokenspeed_mla requires DeepSeek R1 MLA dimensions "
                    "(qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128), "
                    f"got ({qk_nope_head_dim}, {qk_rope_head_dim}, {v_head_dim})"
                )
        return None

    @classmethod
    def get_required_kv_cache_layout(cls) -> "KVCacheLayoutType | None":
        """要求 HND（Head-NumTokens-Dim）布局。"""
        return "HND"


class TokenspeedMLAImpl(MLACommonImpl[MLACommonMetadata]):
    """
    TokenSpeed MLA 注意力的具体实现。

    使用 TokenSpeed 的 CuTe DSL 内核执行 FP8 decode 注意力计算。
    形状特化为 DeepSeek R1 的 MLA 维度。
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

        unsupported_features = [alibi_slopes, sliding_window, logits_soft_cap]
        if any(unsupported_features):
            raise NotImplementedError(
                "TokenspeedMLAImpl does not support one of the following: "
                "alibi_slopes, sliding_window, logits_soft_cap"
            )

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "Encoder self-attention and "
                "encoder/decoder cross-attention "
                "are not implemented for "
                "TokenspeedMLAImpl"
            )

        if not is_quantized_kv_cache(self.kv_cache_dtype):
            raise NotImplementedError(
                "TokenspeedMLAImpl requires an FP8 KV cache "
                "(--kv-cache-dtype fp8 or fp8_e4m3); "
                f"got kv_cache_dtype={self.kv_cache_dtype!r}."
            )

        # 延迟分配 workspace — __init__ 在 worker 设置设备之前运行；
        # 我们在 forward 时看到输入 tensor 时才确定设备。
        self._workspace_buffer: torch.Tensor | None = None
        self.softmax_scale: float | None = None
        self.output_scale: float | None = None

        # 在此预 JIT BF16 和 FP8 prefill 内核 — decode 实现总是在
        # 选择 tokenspeed 时运行，prefill 后端可能不运行（用户可以
        # 与 flash_attn / trtllm 配对）。幂等操作。
        from tokenspeed_mla import warmup_compile_prefill

        for q_dtype in (torch.bfloat16, torch.float8_e4m3fn):
            warmup_compile_prefill(
                q_dtype=q_dtype,
                d_qk=self.qk_nope_head_dim + self.qk_rope_head_dim,
                d_v=self.v_head_dim,
                enable_pdl=False,
            )

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
        1. 如果 q 是元组，拼接为完整 q
        2. 验证 q 已被量化为 FP8（supports_quant_query_input=True）
        3. 重塑 q 为 (num_decodes, q_len, num_heads, head_dim)
        4. 计算 softmax_scale 和 output_scale
        5. 调用 tokenspeed_mla_decode 计算注意力
        6. 展平输出并返回
        """
        from tokenspeed_mla import tokenspeed_mla_decode

        assert kv_c_and_k_pe_cache.numel() > 0
        assert attn_metadata.decode is not None

        if isinstance(q, tuple):
            q_nope, q_pe = q
            q = torch.cat([q_nope, q_pe], dim=-1)

        # supports_quant_query_input=True（在 MLACommonImpl 中设置）告诉
        # 管道在上游通过 _decode_concat_quant_fp8_op 拼接+FP8 量化 Q。
        # 内核形状特化为 FP8 Q + FP8 KV，所以这里任何其他值意味着
        # 上游量化没运行，内核将产生垃圾。
        assert q.dtype == torch.float8_e4m3fn, (
            f"TokenspeedMLAImpl expected FP8 query (supports_quant_query_input=True), "
            f"got {q.dtype}. Pipeline isinstance(q, tuple)={isinstance(q, tuple)}, "
            f"q_scale={layer._q_scale_float}, k_scale={layer._k_scale_float}."
        )

        # tokenspeed_mla_decode 期望 query 形状为
        # (num_decodes, q_len_per_request, num_heads, head_dim)。
        if attn_metadata.num_decode_tokens % attn_metadata.num_decodes != 0:
            logger.warning_once(
                """TokenspeedMLAImpl got a query of uneven length.
                This usually indicates an issue in batch reordering
                or incorrect setup in dummy_run."""
            )
            q = q.unsqueeze(1)
        else:
            q = q.view(attn_metadata.num_decodes, -1, q.shape[-2], q.shape[-1])

        if self.softmax_scale is None:
            # FP8 KV 缓存对此后端是强制的，所以 q_scale/k_scale 总是适用。
            # softmax_scale 是 bmm1；output_scale 是 bmm2 — 两者都需要
            # 以从 FP8 KV 缓存恢复正确的注意力输出
            #（V 存储为 V_real/k_scale）。
            self.softmax_scale = (
                self.scale * layer._q_scale_float * layer._k_scale_float
            )
            self.output_scale = layer._k_scale_float

        if self._workspace_buffer is None:
            self._workspace_buffer = _get_workspace(
                q.device, self.num_heads, self.kv_lora_rank
            )

        # vLLM kv_c_and_k_pe_cache 已经是 (num_blocks, block_size, head_size)。
        # tokenspeed_mla_decode 需要 3D — 直接传递（不像 trtllm 那样 unsqueeze）。
        o = tokenspeed_mla_decode(
            query=q,
            kv_cache=kv_c_and_k_pe_cache,
            workspace_buffer=self._workspace_buffer,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            block_tables=attn_metadata.decode.block_table,
            seq_lens=attn_metadata.decode.seq_lens,
            max_seq_len=attn_metadata.max_seq_len,
            softmax_scale=self.softmax_scale,
            output_scale=self.output_scale,
            enable_pdl=False,
        )

        # 展平输出以获得一致的形状
        o = o.view(-1, o.shape[-2], o.shape[-1])

        # tokenspeed_mla_decode 不返回 LSE。
        return o, None
