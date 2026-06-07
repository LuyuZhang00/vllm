# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Mamba1 注意力后端（Mamba1 Attention Backend）。

本模块实现了 Mamba1 状态空间模型的注意力后端。Mamba1 是 Gu & Dao 提出的
选择性状态空间模型（Selective State Space Model），是第一个成功应用于
大规模语言模型的 SSM 架构。

Mamba1 的核心特性：
1. 选择性机制（Selectivity）：通过输入依赖的参数（B, C, Delta）动态调整
   状态转移矩阵，使模型能够选择性地记忆或遗忘信息
2. 硬件感知算法：使用并行扫描（parallel scan）替代传统的递推计算，
   在 GPU 上实现高效训练
3. 因果卷积前置：在状态更新前使用短因果卷积进行局部特征提取

Mamba1 vs Transformer 的关键差异：
- 复杂度：Mamba1 为 O(n)，Transformer 为 O(n^2)
- 状态：Mamba1 使用固定大小的隐状态，Transformer 使用 KV 缓存
- 训练：Mamba1 使用并行扫描，Transformer 使用矩阵乘法
- 推理：Mamba1 逐 token 更新状态，Transformer 需要访问全部历史

本模块的元数据构建逻辑：
1. 调用基类 _compute_common_metadata 计算公共元数据
2. 如果是 "all" 缓存模式且有 prefill 请求，计算分块元数据
3. 分块元数据用于 Mamba1 的并行扫描算法

缓存模式说明：
- "all" 模式：所有隐状态存储在分页块表中，支持前缀缓存
  使用 block_size 大小的块对齐分块
- 其他模式：使用简化的状态索引映射
"""

from dataclasses import dataclass, replace
from typing import Any

from vllm.v1.attention.backend import AttentionBackend, CommonAttentionMetadata
from vllm.v1.attention.backends.mamba_attn import (
    BaseMambaAttentionMetadata,
    BaseMambaAttentionMetadataBuilder,
)


class Mamba1AttentionBackend(AttentionBackend):
    """
    Mamba1 注意力后端类。

    声明 Mamba1 SSM 模型的后端能力。
    标记为 is_ssm=True 以表明这是状态空间模型后端。
    """

    @staticmethod
    def get_name() -> str:
        """返回后端名称标识。"""
        return "MAMBA1_ATTN"

    @staticmethod
    def get_builder_cls() -> type["Mamba1AttentionMetadataBuilder"]:
        """返回 Mamba1 元数据构建器类。"""
        return Mamba1AttentionMetadataBuilder

    @classmethod
    def is_ssm(cls) -> bool:
        """
        标识此为状态空间模型（SSM）后端。

        Mamba1 是经典的 SSM 架构，使用隐状态而非 KV 缓存存储历史信息。
        """
        return True


@dataclass
class Mamba1AttentionMetadata(BaseMambaAttentionMetadata):
    """
    Mamba1 注意力元数据数据类。

    直接继承 BaseMambaAttentionMetadata，不添加额外字段。

    Mamba1 不需要 Mamba2 特有的字段（如 prep_initial_states、
    chunk_size、seq_idx_p），因为 Mamba1 的分块逻辑由基类处理。
    """
    pass


class Mamba1AttentionMetadataBuilder(
    BaseMambaAttentionMetadataBuilder[Mamba1AttentionMetadata]
):
    """
    Mamba1 注意力元数据构建器。

    继承自 BaseMambaAttentionMetadataBuilder，添加 Mamba1 特有的
    分块元数据计算逻辑。

    build 方法处理流程：
    1. 调用 _compute_common_metadata 计算公共元数据
      （包括 decode/prefill 拆分、隐状态索引、推测解码处理等）
    2. 如果是 "all" 缓存模式且有 prefill 请求：
       a. 调用 _build_chunk_metadata_tensors 计算分块元数据
       b. 使用 block_size 作为分块大小（对齐到块边界）
       c. 将分块元数据添加到元数据对象中
    3. 返回最终的元数据对象

    分块元数据的作用：
    - cu_chunk_seqlen_p：分块累积序列长度，用于并行扫描算法
    - last_chunk_indices_p：每个序列最后一个块的索引
    """

    metadata_cls = Mamba1AttentionMetadata

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
        **kwargs: Any,
    ) -> Mamba1AttentionMetadata:
        """
        构建 Mamba1 注意力元数据。

        参数：
            common_prefix_len: 共享前缀长度（未使用，Mamba1 不支持级联注意力）
            common_attn_metadata: 调度器输出的通用注意力元数据
            fast_build: 是否快速构建（未使用）

        返回：
            Mamba1AttentionMetadata 对象
        """
        # 第一步：计算公共元数据（decode/prefill 拆分、隐状态索引等）
        common = self._compute_common_metadata(common_attn_metadata)

        if (
            common.num_prefills > 0
            and self.vllm_config.cache_config.mamba_cache_mode == "all"
        ):
            # 第二步：仅在 "all" 缓存模式下计算分块元数据
            # block_size 作为分块大小，确保与 KV 缓存块边界对齐
            cu_chunk_seqlen_p, _, last_chunk_indices_p = (
                self._build_chunk_metadata_tensors(
                    self.kv_cache_spec.block_size,
                    common,
                    common_attn_metadata,
                )
            )
            # 使用 dataclasses.replace 不可变地更新元数据
            return replace(
                common,
                cu_chunk_seqlen_p=cu_chunk_seqlen_p,
                last_chunk_indices_p=last_chunk_indices_p,
            )

        return common
