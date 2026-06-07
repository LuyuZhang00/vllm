# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
线性注意力后端（Linear Attention Backend）。

本模块实现了线性注意力（Linear Attention）后端。线性注意力是传统
Softmax 注意力的一种高效替代方案，通过将 softmax 核函数替换为
可分解的核函数，将注意力计算的复杂度从 O(n^2) 降低到 O(n)。

线性注意力的核心思想：
1. 标准注意力：Attn(Q,K,V) = softmax(QK^T / sqrt(d)) V  -- O(n^2)
2. 线性注意力：Attn(Q,K,V) = phi(Q) (phi(K)^T V)          -- O(n)
   其中 phi 是一个逐元素的特征映射函数（如 elu+1、ReLU 等）

线性注意力的优势：
1. 线性复杂度 O(n)：适合超长序列处理
2. 固定大小的隐状态：可以流式处理，内存效率高
3. RNN 形式：decode 阶段可以像 RNN 一样逐 token 更新状态

与 Mamba SSM 的异同：
- 相同点：都使用隐状态存储历史信息，都是 O(n) 复杂度
- 不同点：线性注意力使用注意力矩阵的低秩近似，
  而 Mamba 使用连续时间状态空间模型的离散化

本模块的元数据相对简单，不继承 BaseMambaAttentionMetadata，
因为线性注意力不需要复杂的分块（chunking）机制。
"""

from dataclasses import dataclass

import torch

from vllm.config import VllmConfig
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.utils import (
    mamba_get_block_table_tensor,
    split_decodes_and_prefills,
)
from vllm.v1.kv_cache_interface import AttentionSpec, MambaSpec


class LinearAttentionBackend(AttentionBackend):
    """
    线性注意力后端类。

    声明线性注意力模型的后端能力。
    标记为 is_ssm=True，因为线性注意力也使用隐状态机制。
    """

    @staticmethod
    def get_name() -> str:
        """返回后端名称标识。"""
        return "LINEAR_ATTN"

    @staticmethod
    def get_builder_cls() -> type["LinearAttentionMetadataBuilder"]:
        """返回线性注意力元数据构建器类。"""
        return LinearAttentionMetadataBuilder

    @classmethod
    def is_ssm(cls) -> bool:
        """
        标识此为状态空间模型（SSM）后端。

        线性注意力使用隐状态存储历史信息，
        与 Mamba 等 SSM 模型共享相同的状态管理接口。
        """
        return True


@dataclass
class LinearAttentionMetadata:
    """
    线性注意力元数据数据类。

    线性注意力的元数据比 Mamba 类模型简单，因为：
    1. 不需要分块（chunking）元数据
    2. 不需要 causal_conv1d 元数据
    3. 不需要复杂的推测解码支持

    属性说明：
    - num_prefills: prefill 请求数量
    - num_prefill_tokens: prefill token 总数
    - num_decodes: decode 请求数量
    - num_decode_tokens: decode token 总数
    - query_start_loc: 每个请求的查询起始位置（累积和）
    - seq_lens: 每个请求的总序列长度
    - state_indices_tensor: 隐状态索引张量，shape=[batch]
      将每个请求映射到其隐状态在缓存中的存储位置
    """

    num_prefills: int
    num_prefill_tokens: int
    num_decodes: int
    num_decode_tokens: int
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor

    state_indices_tensor: torch.Tensor  # shape: [batch,]


class LinearAttentionMetadataBuilder(AttentionMetadataBuilder[LinearAttentionMetadata]):
    """
    线性注意力元数据构建器。

    构建流程：
    1. 计算隐状态索引（state_indices_tensor）
    2. 将请求拆分为 decode 和 prefill 两组
    3. 组装 LinearAttentionMetadata

    CUDA Graph 支持：
    - _cudagraph_support = UNIFORM_SINGLE_TOKEN_DECODE
    - 仅支持单 token decode 的均匀批次 CUDA Graph
    - 不支持多 token decode（无推测解码支持）

    重排序策略：
    - reorder_batch_threshold = 1
    - query 长度为 1 的视为 decode，其余为 prefill
    """

    reorder_batch_threshold: int = 1

    _cudagraph_support = AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        assert isinstance(kv_cache_spec, MambaSpec)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> LinearAttentionMetadata:
        """
        构建线性注意力元数据。

        处理流程：
        1. 从通用元数据中提取查询位置和序列长度
        2. 计算隐状态索引（取 block_table 的第一列）
        3. 按 query 长度拆分为 decode 和 prefill 两组
        4. 组装并返回 LinearAttentionMetadata
        """
        query_start_loc = common_attn_metadata.query_start_loc
        seq_lens = common_attn_metadata.seq_lens

        # 计算隐状态索引：从块表中提取每个请求的隐状态位置
        # mamba_get_block_table_tensor 返回 [batch, num_blocks] 形状的张量
        # [:, 0] 取第一列，即每个请求的隐状态块索引
        state_indices_tensor = mamba_get_block_table_tensor(
            common_attn_metadata.block_table_tensor,
            common_attn_metadata.seq_lens,
            self.kv_cache_spec,
            self.vllm_config.cache_config.mamba_cache_mode,
        )[:, 0]

        # 按 query 长度拆分为 decode（长度为 1）和 prefill（长度 > 1）
        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(
                common_attn_metadata, decode_threshold=self.reorder_batch_threshold
            )
        )

        attn_metadata = LinearAttentionMetadata(
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            query_start_loc=query_start_loc,
            seq_lens=seq_lens,
            state_indices_tensor=state_indices_tensor,
        )
        return attn_metadata
