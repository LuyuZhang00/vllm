# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention layer with PagedAttention and Triton prefix prefill."""
"""
ROCm 注意力后端（ROCm Attention Backend）。

本模块实现了在 AMD GPU（ROCm 平台）上运行的注意力计算后端，适用于
AMD MI200、MI300 系列 GPU。采用分页注意力（PagedAttention）机制管理
KV 缓存，并使用 Triton kernel 进行前缀预填充。

核心组件：
1. RocmAttentionBackend：后端能力声明，包括支持的数据类型、头大小、
   KV 缓存形状（[2, num_blocks, block_size, num_kv_heads, head_size]）
2. RocmAttentionMetadata：注意力元数据，存储序列信息、块表、级联注意力参数等
3. RocmAttentionMetadataBuilder：元数据构建器，支持 CUDA Graph 捕获优化
4. RocmAttentionImpl：前向计算实现，调用 chunked_prefill_paged_decode kernel

与标准 FlashAttention 后端的主要差异：
- KV 缓存布局不同：ROCm 使用 [2, num_blocks, block_size, num_kv_heads, head_size]
  而标准 FlashAttn 使用 [num_blocks, block_size, num_kv_heads, head_size]（分离 K/V）
- 不支持级联注意力（cascade attention）
- 不支持 KV 连接器（KV connector），因为 KV 缓存布局不兼容
- 不支持注意力汇聚（attention sink）
- 支持 ROCm 特有的融合 RoPE + KV 缓存更新 kernel（通过 AITER ops）

执行流程：
1. 编码器注意力：直接调用 Triton prefill kernel，无需 KV 缓存
2. 解码器注意力：
   a. 拆分 KV 缓存为 key_cache 和 value_cache
   b. 如果有新的 key/value，写入 KV 缓存（PagedAttention.write_to_paged_cache）
   c. 调用 chunked_prefill_paged_decode 执行融合的 prefill+decode 计算
"""

from dataclasses import dataclass
from typing import ClassVar

import torch

from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8StaticTensorSym,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import is_quantized_kv_cache
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.ops.chunked_prefill_paged_decode import (
    chunked_prefill_paged_decode,
    has_native_kv_cache_layout,
)
from vllm.v1.attention.ops.paged_attn import PagedAttention
from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
    triton_reshape_and_cache_flash,
)
from vllm.v1.kv_cache_interface import AttentionSpec

logger = init_logger(__name__)


@dataclass
class RocmAttentionMetadata:
    """
    ROCm 注意力元数据数据类。

    注意力计算中关键概念说明（见下方 ASCII 图）：
    - context_len（上下文长度）：之前迭代已经计算过的 token 数量
    - query_len（查询长度）：当前迭代新加入的 token 数量
    - seq_len（序列长度）：总序列长度 = context_len + query_len

    |---------- N-1 iteration --------|
    |---------------- N iteration ---------------------|
    |- tokenA -|......................|-- newTokens ---|
    |---------- context_len ----------|
    |-------------------- seq_len ---------------------|
                                  |-- query_len ---|

    属性说明：
    - num_actual_tokens: 实际有效 token 数（不含 padding）
    - max_query_len: 当前批次中最大查询长度
    - query_start_loc: 每个请求查询的起始位置（累积和）
    - max_seq_len: 当前批次中最大序列长度
    - seq_lens: 每个请求的总序列长度
    - block_table: 逻辑块到物理块的映射表
    - slot_mapping: token 到 KV 缓存槽位的映射
    - use_cascade: 是否使用级联注意力优化
    - common_prefix_len: 共享前缀长度（级联注意力使用）
    - cu_prefix_query_lens: 前缀查询长度的累积和
    - prefix_kv_lens: 前缀 KV 长度
    - suffix_kv_lens: 后缀 KV 长度
    - scheduler_metadata: 可选的提前调度元数据
    - prefix_scheduler_metadata: 前缀调度元数据
    - causal: 是否使用因果注意力掩码
    """

    # NOTE(sang): Definition of context_len, query_len, and seq_len.
    # |---------- N-1 iteration --------|
    # |---------------- N iteration ---------------------|
    # |- tokenA -|......................|-- newTokens ---|
    # |---------- context_len ----------|
    # |-------------------- seq_len ---------------------|
    #                                   |-- query_len ---|

    num_actual_tokens: int  # Number of tokens excluding padding.
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor

    # For cascade attention.
    # 级联注意力相关字段
    use_cascade: bool
    common_prefix_len: int
    cu_prefix_query_lens: torch.Tensor | None
    prefix_kv_lens: torch.Tensor | None
    suffix_kv_lens: torch.Tensor | None

    # Optional aot scheduling
    # 可选的提前调度（Ahead-of-Time scheduling）元数据
    scheduler_metadata: torch.Tensor | None = None
    prefix_scheduler_metadata: torch.Tensor | None = None

    # DFlash drafting sets this to False via CommonAttentionMetadata.
    # 是否使用因果注意力，DFlash 草稿模型会设为 False
    causal: bool = True


class RocmAttentionMetadataBuilder(AttentionMetadataBuilder[RocmAttentionMetadata]):
    """
    ROCm 注意力元数据构建器。

    支持 CUDA Graph 捕获优化：
    - _cudagraph_support = ALWAYS 表示始终支持 CUDA Graph
    - build_for_cudagraph_capture 方法在捕获 CUDA Graph 时特殊处理
      seq_lens 和 query_start_loc 以避免性能问题

    级联注意力处理：
    当多个请求共享相同前缀时，可以将共享前缀的 KV 缓存复用，
    减少重复计算。但在 ROCm 后端当前禁用此功能。
    """

    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.ALWAYS

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

        self.block_size = kv_cache_spec.block_size

        model_config = vllm_config.model_config
        # 获取模型的注意力头参数
        self.num_heads_q = model_config.get_num_attention_heads(
            vllm_config.parallel_config
        )
        self.num_heads_kv = model_config.get_num_kv_heads(vllm_config.parallel_config)
        self.headdim = model_config.get_head_size()

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> RocmAttentionMetadata:
        """
        为 CUDA Graph 捕获构建特殊的元数据。

        CUDA Graph 捕获时需要固定大小的张量，因此：
        1. 将 seq_lens 填充为 1（而非 max_model_len），避免图捕获极慢
        2. 将 query_start_loc 置零，避免 prefix_prefill kernel 中的非法内存访问
        """
        attn_metadata = self.build(0, common_attn_metadata)
        # When doing full graph capture, setting seq_lens to
        # max_model_len will cause graph capture to be extremely
        # slow, so here we set it to 1.
        attn_metadata.seq_lens.fill_(1)

        # Here we set the query start locs to 0. This is to
        # cover up an invalid memory access in the prefix_prefil kernel
        # that we run into during graph capture (#25985)
        common_attn_metadata.query_start_loc.zero_()
        common_attn_metadata.query_start_loc_cpu.zero_()

        return attn_metadata

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> RocmAttentionMetadata:
        """
        构建 ROCm 注意力元数据。

        处理流程：
        1. 从通用元数据中提取基本信息
        2. 判断是否使用级联注意力（common_prefix_len > 0）
        3. 如果使用级联注意力，构建前缀/后缀相关的元数据
        4. 组装并返回 RocmAttentionMetadata
        """
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        max_query_len = common_attn_metadata.max_query_len

        max_seq_len = common_attn_metadata.max_seq_len
        query_start_loc = common_attn_metadata.query_start_loc
        seq_lens = common_attn_metadata.seq_lens
        block_table_tensor = common_attn_metadata.block_table_tensor
        slot_mapping = common_attn_metadata.slot_mapping

        use_cascade = common_prefix_len > 0

        if use_cascade:
            # 级联注意力：所有请求共享 common_prefix_len 长度的前缀
            cu_prefix_query_lens = torch.tensor(
                [0, num_actual_tokens], dtype=torch.int32, device=self.device
            )
            prefix_kv_lens = torch.tensor(
                [common_prefix_len], dtype=torch.int32, device=self.device
            )
            # 后缀 KV 长度 = 总序列长度 - 共享前缀长度
            suffix_kv_lens = common_attn_metadata.seq_lens.cpu() - common_prefix_len
            suffix_kv_lens = suffix_kv_lens.to(self.device)
        else:
            cu_prefix_query_lens = None
            prefix_kv_lens = None
            suffix_kv_lens = None
            prefix_scheduler_metadata = None

        attn_metadata = RocmAttentionMetadata(
            num_actual_tokens=num_actual_tokens,
            max_query_len=max_query_len,
            query_start_loc=query_start_loc,
            max_seq_len=max_seq_len,
            seq_lens=seq_lens,
            block_table=block_table_tensor,
            slot_mapping=slot_mapping,
            use_cascade=use_cascade,
            common_prefix_len=common_prefix_len,
            cu_prefix_query_lens=cu_prefix_query_lens,
            prefix_kv_lens=prefix_kv_lens,
            suffix_kv_lens=suffix_kv_lens,
            prefix_scheduler_metadata=prefix_scheduler_metadata,
            causal=common_attn_metadata.causal,
        )
        return attn_metadata


class RocmAttentionBackend(AttentionBackend):
    """
    ROCm 注意力后端类。

    声明 AMD GPU 上注意力计算的能力和支持范围：
    - 支持 fp16、bf16、fp32 数据类型
    - 支持多种 KV 缓存类型（包括 FP8 量化）
    - 块大小必须为 16 的倍数（ROCm 原生 kernel 仅支持 16 和 32，
      其他大小通过 Triton kernel 动态路由处理）
    - 不支持级联注意力、注意力汇聚、KV 连接器
    - 支持非因果注意力（用于编码器）
    """

    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
        "fp8_e5m2",
    ]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        # ROCM paged attention native C++ kernel only supports block sizes 16 and 32
        # due to shared memory (LDS) constraints on AMD GPUs.
        # See csrc/rocm/attention.cu CALL_CUSTOM_LAUNCHER_BLK macro.
        # However, vLLM allows support for any multiple of 16 via the Triton path.
        # As addressed in PR: https://github.com/vllm-project/vllm/pull/31380,
        # non-standard models (like qwen3-next with block_size 544, or qwen3_5
        # with 784 and 1056) are dynamically routed to our optimized Triton kernel
        # in `do_kv_cache_update`.
        # ROCm 原生 C++ kernel 仅支持 16 和 32 的块大小（受 LDS 共享内存限制），
        # 但通过 Triton 路径可以支持任意 16 的倍数
        return [MultipleOf(16)]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        """返回支持的注意力头大小。"""
        return [32, 64, 80, 96, 128, 160, 192, 224, 256]

    @classmethod
    def supports_mm_prefix(cls) -> bool:
        """支持多模态前缀（multimodal prefix）。"""
        return True

    @classmethod
    def supports_sink(cls) -> bool:
        # ROCM custom attention kernel does not support sinks.
        # Callink this backend with sinks will cause it to fall back to the Triton
        # kernel, which is less efficient than the proper triton backends.
        # ROCm 自定义 kernel 不支持注意力汇聚（attention sink），
        # 使用 sink 会回退到效率较低的 Triton kernel
        return False

    @classmethod
    def supports_non_causal(cls) -> bool:
        """支持非因果注意力（用于编码器处理）。"""
        return True

    @classmethod
    def supports_kv_connector(cls) -> bool:
        # ROCM_ATTN uses (2, num_blocks, ...) KV cache layout which is
        # incompatible with KV connectors that require blocks-first layout.
        # ROCm 使用 (2, num_blocks, ...) 布局，与要求 blocks-first 布局的
        # KV 连接器不兼容
        return False

    # forward 方法不包含 KV 缓存更新，由 do_kv_cache_update 单独处理
    forward_includes_kv_cache_update: bool = False

    @staticmethod
    def get_name() -> str:
        """返回后端名称标识。"""
        return "ROCM_ATTN"

    @staticmethod
    def get_impl_cls() -> type["RocmAttentionImpl"]:
        """返回 ROCm 注意力实现类。"""
        return RocmAttentionImpl

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        """ENCODER_DECODER is not supported because
        chunked_prefill_paged_decode's prefill kernel (context_attention_fwd)
        assumes self-attention semantics: it treats passed K/V as new tokens
        to mix with cached K/V. For cross-attention layers the encoder K/V
        are already fully cached, so mixing them again produces incorrect
        results when max_query_len > 1 (e.g. beam search).
        """
        # 不支持 ENCODER_DECODER 类型，因为 chunked_prefill_paged_decode 的
        # prefill kernel 假设自注意力语义，会将传入的 K/V 与缓存的 K/V 混合，
        # 对于交叉注意力会导致结果错误
        return attn_type in (
            AttentionType.DECODER,
            AttentionType.ENCODER,
            AttentionType.ENCODER_ONLY,
        )

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        """
        返回 KV 缓存形状。

        ROCm 布局：[2, num_blocks, block_size, num_kv_heads, head_size]
        - 维度 0 (2): key 和 value 各一个
        - 维度 1: 缓存块数量
        - 维度 2: 每块中的 token 数
        - 维度 3: KV 头数量
        - 维度 4: 头维度

        注意：这与标准 FlashAttn 的布局不同，
        标准 FlashAttn 为 [num_blocks, block_size, num_kv_heads, head_size]
        且 key 和 value 分开存储。
        """
        if block_size % 16 != 0:
            raise ValueError("Block size must be a multiple of 16.")
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def use_cascade_attention(*args, **kwargs) -> bool:
        """ROCm 后端不支持级联注意力。"""
        return False

    @staticmethod
    def get_builder_cls() -> type["RocmAttentionMetadataBuilder"]:
        """返回 ROCm 元数据构建器类。"""
        return RocmAttentionMetadataBuilder


class RocmAttentionImpl(AttentionImpl):
    """
    ROCm 注意力前向计算实现类。

    实现了在 AMD GPU 上的注意力计算，主要特性：
    1. 支持融合输出量化（FP8 静态张量对称量化）
    2. 编码器注意力使用 Triton prefill kernel
    3. 解码器注意力使用 chunked_prefill_paged_decode 融合 kernel
    4. 支持 FP8 KV 缓存量化
    5. 支持 ALiBi 位置编码和滑动窗口
    6. 支持融合 RoPE + KV 缓存更新（通过 ROCm AITER ops）

    性能注意事项：
    - 此方法在 eager 模式 PyTorch 下执行（非 CUDA Graph 模式）
    - view 和 slice 操作虽然不涉及 GPU 计算，但 Python 层面开销不可忽视
    - 尽量减少 PyTorch 操作以降低 CPU 开销
    """

    def fused_output_quant_supported(self, quant_key: QuantKey):
        """
        检查是否支持融合输出量化。

        ROCm 后端仅支持 FP8 静态张量对称量化（kFp8StaticTensorSym）。
        """
        return quant_key == kFp8StaticTensorSym

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
        """
        初始化 ROCm 注意力实现。

        参数说明：
        - num_heads: 查询头数量
        - head_size: 每个头的维度
        - scale: 注意力缩放因子
        - num_kv_heads: KV 头数量（GQA 时可能少于 num_heads）
        - alibi_slopes: ALiBi 位置编码斜率
        - sliding_window: 滑动窗口大小
        - kv_cache_dtype: KV 缓存数据类型
        - logits_soft_cap: logits 软封顶值（0 表示不使用）
        - attn_type: 注意力类型
        - kv_sharing_target_layer_name: KV 共享目标层名
        - sinks: 注意力汇聚张量
        """
        self.attn_type = attn_type
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        if sliding_window is None:
            self.sliding_window = (-1, -1)
        else:
            self.sliding_window = (sliding_window - 1, 0)
        self.kv_cache_dtype = kv_cache_dtype
        if logits_soft_cap is None:
            # In flash-attn, setting logits_soft_cap as 0 means no soft cap.
            # 在 flash-attn 中，logits_soft_cap=0 表示不使用软封顶
            logits_soft_cap = 0
        self.logits_soft_cap = logits_soft_cap
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name

        # 每个 KV 头服务的查询头数量（GQA 比率）
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        # FP8 数据类型（用于 KV 缓存量化）
        self.fp8_dtype = current_platform.fp8_dtype()

        # 注意力汇聚（Attention Sink）
        self.sinks = sinks
        if sinks is not None:
            assert sinks.shape[0] == num_heads, (
                "Sinks must have the same number of heads as the number of "
                f"heads in the layer. Sinks shape: {sinks.shape}, "
                f"num_heads: {num_heads}."
            )

    def _forward_encoder_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: RocmAttentionMetadata,
        layer: torch.nn.Module,
    ) -> torch.Tensor:
        """Forward pass for encoder attention without KV cache.

        Args:
            query: shape = [num_encoder_tokens, num_heads, head_size]
            key: shape = [num_encoder_tokens, num_kv_heads, head_size]
            value: shape = [num_encoder_tokens, num_kv_heads, head_size]
            output: shape = [num_encoder_tokens, num_heads, head_size]
            attn_metadata: Encoder attention metadata
            layer: The attention layer
        """
        # 编码器注意力：不使用 KV 缓存，直接对 Q、K、V 计算
        # For encoder attention, process FP8 quantization if needed
        if is_quantized_kv_cache(self.kv_cache_dtype):
            raise NotImplementedError(
                "quantization is not supported for encoder attention"
            )

        # Use encoder-specific metadata for sequence information
        query_start_loc = attn_metadata.query_start_loc
        seq_lens = attn_metadata.seq_lens
        max_query_len = attn_metadata.max_query_len

        # Call flash attention directly on Q, K, V tensors
        # 使用 Triton prefill attention kernel 直接计算
        from vllm.v1.attention.ops.triton_prefill_attention import context_attention_fwd

        context_attention_fwd(
            q=query,
            k=key,
            v=value,
            o=output,
            b_start_loc=query_start_loc,
            b_seq_len=seq_lens,
            max_input_len=max_query_len,
            is_causal=False,  # 编码器不使用因果掩码
            softmax_scale=self.scale,
            sliding_window_q=self.sliding_window[0],
            sliding_window_k=self.sliding_window[1],
        )
        return output

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: RocmAttentionMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with FlashAttention.

        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache: shape =
                [2, num_blocks, block_size, num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        # ROCm 后端暂不支持融合 block_scale 输出量化
        if output_block_scale is not None:
            raise NotImplementedError(
                "fused block_scale output quantization is not yet supported"
                " for RocmAttentionImpl"
            )

        if attn_metadata is None:
            # Profiling run.
            # 预热阶段，返回零输出
            return output.fill_(0)

        # ROCm 后端不支持级联注意力，此处断言确认
        assert attn_metadata.use_cascade is False

        # IMPORTANT!
        # NOTE(woosuk): With piece-wise CUDA graphs, this method is executed in
        # eager-mode PyTorch. Thus, we need to be careful about any CPU overhead
        # in this method. For example, `view` and `slice` (or `[:n]`) operations
        # are surprisingly slow even in the case they do not invoke any GPU ops.
        # Minimize the PyTorch ops in this method as much as possible.
        # Whenever making a change in this method, please benchmark the
        # performance to make sure it does not introduce any overhead.

        num_actual_tokens = attn_metadata.num_actual_tokens

        # 编码器注意力走单独的路径，不需要 KV 缓存
        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            return self._forward_encoder_attention(
                query[:num_actual_tokens],
                key[:num_actual_tokens],
                value[:num_actual_tokens],
                output[:num_actual_tokens],
                attn_metadata,
                layer,
            )

        # 解码器注意力：使用分页 KV 缓存
        # 将 KV 缓存拆分为 key_cache 和 value_cache
        key_cache, value_cache = PagedAttention.split_kv_cache(
            kv_cache, self.num_kv_heads, self.head_size
        )

        # FP8 KV 缓存：将缓存视图转换为 FP8 数据类型
        if is_quantized_kv_cache(self.kv_cache_dtype):
            key_cache = key_cache.view(self.fp8_dtype)
            value_cache = value_cache.view(self.fp8_dtype)
            assert layer._q_scale_float == 1.0, (
                "A non 1.0 q_scale is not currently supported."
            )

        cu_seqlens_q = attn_metadata.query_start_loc
        seqused_k = attn_metadata.seq_lens
        max_seqlen_q = attn_metadata.max_query_len
        max_seqlen_k = attn_metadata.max_seq_len
        block_table = attn_metadata.block_table

        # Compute attention and update output up to `num_actual_tokens`.
        # 调用融合的 chunked prefill + paged decode kernel
        # 该 kernel 同时处理 prefill（新 token 的注意力计算）和
        # decode（从 KV 缓存中读取历史 token）
        chunked_prefill_paged_decode(
            query=query[:num_actual_tokens],
            key=key[:num_actual_tokens] if key is not None else None,
            value=value[:num_actual_tokens] if value is not None else None,
            output=output[:num_actual_tokens],
            kv_cache_dtype=self.kv_cache_dtype,
            key_cache=key_cache,
            value_cache=value_cache,
            block_table=block_table,
            query_start_loc=cu_seqlens_q,
            seq_lens=seqused_k,
            max_seq_len=max_seqlen_k,
            max_query_len=max_seqlen_q,
            k_scale=layer._k_scale,
            v_scale=layer._v_scale,
            alibi_slopes=self.alibi_slopes,
            sliding_window=self.sliding_window[0],
            sm_scale=self.scale,
            output_scale=output_scale,
            sinks=self.sinks,
            causal=attn_metadata.causal,
        )

        return output

    def do_kv_cache_update(
        self,
        layer: AttentionLayer,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ):
        """
        更新 KV 缓存。

        将当前步骤的 key 和 value 写入到 KV 缓存的对应物理槽位。

        根据块大小和缓存布局选择不同的写入路径：
        1. 块大小为 16 或 32 且使用原生布局 -> 使用 vLLM 原生 HIP C++ kernel
        2. 其他块大小或混合布局 -> 使用步长感知的 Triton kernel
           （原生 kernel 假设连续块存储，会写入错误的缓存位置）
        """
        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            # 编码器注意力不需要更新 KV 缓存
            return
        key_cache, value_cache = PagedAttention.split_kv_cache(
            kv_cache, self.num_kv_heads, self.head_size
        )

        # Reshape the input keys and values and store them in the cache.
        # Get the actual block_size from value_cache
        # value_cache shape: [num_blocks, num_heads, head_size, block_size]
        block_size = value_cache.shape[3]
        has_native_layout = has_native_kv_cache_layout(key_cache, value_cache)

        if block_size in (16, 32) and has_native_layout:
            # Normal 16, 32 with contiguous blocks: use vLLM native HIP C++ logic.
            # 标准块大小 + 连续存储：使用高效的 C++ kernel
            PagedAttention.write_to_paged_cache(
                key,
                value,
                key_cache,
                value_cache,
                slot_mapping,
                self.kv_cache_dtype,
                layer._k_scale,
                layer._v_scale,
            )
        else:
            # Non-standard blocks and hybrid attention/Mamba layouts need the
            # stride-aware Triton writer. The native reshape_and_cache kernel
            # assumes contiguous block storage and writes to the wrong hybrid
            # cache blocks.
            # 非标准块大小或混合布局：使用步长感知的 Triton kernel
            triton_reshape_and_cache_flash(
                key,
                value,
                key_cache,
                value_cache,
                slot_mapping,
                self.kv_cache_dtype,
                layer._k_scale,
                layer._v_scale,
            )

    def fused_rope_kvcache_supported(self):
        """
        检查是否支持融合 RoPE + KV 缓存更新。

        需要 ROCm AITER ops 启用。该优化将 RoPE 位置编码计算与
        KV 缓存更新融合为一个 kernel，减少内存访问次数。
        """
        return rocm_aiter_ops.is_enabled()

    def do_rope_and_kv_cache_update(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        is_neox: bool,
        kv_cache: torch.Tensor,
        layer_slot_mapping: torch.Tensor,
    ):
        """
        融合执行 RoPE 位置编码和 KV 缓存更新。

        将以下两个操作融合为一个 kernel：
        1. 对 query 和 key 应用 RoPE（Rotary Position Embedding）位置编码
        2. 将编码后的 key 和 value 写入 KV 缓存

        参数：
            layer: 注意力层
            query: 查询张量
            key: 键张量
            value: 值张量
            positions: 位置索引
            cos_sin_cache: 预计算的 cos/sin 缓存
            is_neox: 是否使用 NeoX 风格的 RoPE（不同的维度拆分方式）
            kv_cache: KV 缓存张量
            layer_slot_mapping: 槽位映射
        """
        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            return
        key_cache, value_cache = PagedAttention.split_kv_cache(
            kv_cache,
            layer.num_kv_heads,  # type: ignore[attr-defined]
            layer.head_size,  # type: ignore[attr-defined]
        )
        flash_layout = False

        is_fp8_kv_cache = is_quantized_kv_cache(self.kv_cache_dtype)
        if is_fp8_kv_cache:
            key_cache = key_cache.view(self.fp8_dtype)
            value_cache = value_cache.view(self.fp8_dtype)

        # 调用 ROCm AITER 的 Triton 融合 kernel
        rocm_aiter_ops.triton_rope_and_cache(
            query,
            key,
            value,
            positions,
            cos_sin_cache,
            is_neox,
            key_cache,
            value_cache,
            layer_slot_mapping,
            layer._k_scale,
            layer._v_scale,
            flash_layout,
            is_fp8_kv_cache,
        )
