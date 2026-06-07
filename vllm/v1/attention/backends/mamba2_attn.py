# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Mamba2 注意力后端（Mamba2 Attention Backend）。

本模块实现了 Mamba2 状态空间模型的注意力后端。Mamba2 是 Mamba1 的改进版本，
由 Dao & Gu 在 "Transformers are SSMs" 论文中提出，主要改进包括：

1. 结构化状态空间对偶性（SSD）：揭示了 SSM 和注意力机制之间的数学等价性
2. 更高效的分块算法：使用 SSD 算法替代并行扫描，进一步提升计算效率
3. 更大的状态维度：支持更大的隐状态，增强模型的记忆容量
4. 多头结构：类似 Transformer 的多头注意力，支持 GQA（分组查询注意力）

Mamba2 vs Mamba1 的关键差异：
- 算法：Mamba2 使用 SSD（Structured State-space Duality），Mamba1 使用并行扫描
- 分块大小：Mamba2 使用模型配置的 chunk_size（通常为 256 或 64），
  Mamba1 使用 block_size
- 初始状态：Mamba2 需要显式处理初始状态（prep_initial_states 标志）
- 序列索引：Mamba2 需要 seq_idx_p 张量来标识每个 token 所属的序列

本模块的关键组件：
1. compute_varlen_chunk_metadata：计算变长分块元数据的独立函数
   - 供测试和其他调用者使用，避免重复实现
   - 处理序列边界和物理块边界的对齐
2. Mamba2AttentionMetadataBuilder：元数据构建器
   - 从模型配置获取 chunk_size
   - 为 prefill 计算分块元数据和序列索引
   - 检查是否有初始状态需要准备

SSD 分块处理流程：
1. 将每个序列的 token 按 chunk_size 分块
2. 每个块不超过序列边界和物理块边界
3. 输出 cu_chunk_seqlens（累积长度）、last_chunk_indices（最后块索引）、
   seq_idx_chunks（块到序列的映射）
"""

import itertools
from dataclasses import dataclass, replace
from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.v1.attention.backend import (
    AttentionBackend,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.mamba_attn import (
    BaseMambaAttentionMetadata,
    BaseMambaAttentionMetadataBuilder,
)
from vllm.v1.kv_cache_interface import AttentionSpec


def compute_varlen_chunk_metadata(
    query_start_loc: torch.Tensor,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build chunk-aligned, variable-length metadata used by Mamba2 SSD kernels.

    Given per-sequence cumulative token starts `query_start_loc` of shape [B+1]
    and a physical `chunk_size`, returns three tensors on the same device:
      - cu_chunk_seqlens:  (nchunks+1,) int32   exclusive prefix-sum of
        logical-chunk lengths (each logical chunk never crosses a sequence or
        physical-chunk boundary).
      - last_chunk_indices: (B,)       int32   index of the last logical chunk
        for each sequence (=-1 for empty sequences).
      - seq_idx_chunks:     (nchunks,) int32   sequence index for each logical
        chunk in order.

    This is intentionally lightweight and CPU-side; it mirrors the metadata
    produced by the V1 Mamba2 meta-data builder and is exported so tests
    (and other callers) can avoid duplicating the logic.
    """
    # 构建 Mamba2 SSD kernel 使用的分块对齐变长元数据。
    #
    # 输入：query_start_loc [B+1]，每个序列的累积 token 起始位置
    #       chunk_size：物理块大小
    #
    # 输出：三个张量
    #   cu_chunk_seqlens: (nchunks+1,) int32 - 逻辑块长度的排他前缀和
    #     每个逻辑块不跨越序列边界或物理块边界
    #   last_chunk_indices: (B,) int32 - 每个序列最后一个逻辑块的索引
    #     空序列的索引为 -1
    #   seq_idx_chunks: (nchunks,) int32 - 每个逻辑块所属的序列索引
    #
    # 此函数在 CPU 上执行，轻量级设计；供测试和其他调用者使用。

    assert query_start_loc.ndim == 1, "query_start_loc must be 1-D [B+1]"
    assert int(query_start_loc[0].item()) == 0, "query_start_loc[0] must be 0"
    device = query_start_loc.device

    qsl64 = query_start_loc.to(torch.int64)
    starts = qsl64[:-1].tolist()
    ends = qsl64[1:].tolist()
    total = int(qsl64[-1].item())

    chunk_lens: list[int] = []
    seq_idx_chunks: list[int] = []
    last_chunk_indices: list[int] = [-1] * len(starts)

    for b, (s, e) in enumerate(zip(starts, ends)):
        if e <= s:
            # empty sequence
            # 空序列跳过
            continue
        pos = s
        while pos < e:
            # split at both sequence boundaries and physical chunk boundaries
            # 在序列边界和物理块边界处分割
            # room: 当前物理块中剩余的空间
            room = chunk_size - (pos % chunk_size)
            # take: 实际取的 token 数（不超过剩余空间和序列剩余长度）
            take = min(room, e - pos)
            chunk_lens.append(int(take))
            seq_idx_chunks.append(b)
            last_chunk_indices[b] = len(chunk_lens) - 1
            pos += take

    # Exclusive prefix sum over logical-chunk lengths
    # 计算逻辑块长度的排他前缀和
    if chunk_lens:
        cu_chunk_seqlens = torch.tensor(
            [0] + list(itertools.accumulate(chunk_lens)),
            device=device,
            dtype=torch.int32,
        )
        # Final boundary must equal total tokens
        # 最终边界必须等于总 token 数
        assert int(cu_chunk_seqlens[-1].item()) == total
    else:
        cu_chunk_seqlens = torch.tensor([0], device=device, dtype=torch.int32)

    last_chunk_indices_t = (
        torch.tensor(last_chunk_indices, device=device, dtype=torch.int32)
        if len(starts) > 0
        else torch.empty((0,), device=device, dtype=torch.int32)
    )
    seq_idx_chunks_t = torch.tensor(seq_idx_chunks, device=device, dtype=torch.int32)
    return cu_chunk_seqlens, last_chunk_indices_t, seq_idx_chunks_t


class Mamba2AttentionBackend(AttentionBackend):
    """
    Mamba2 注意力后端类。

    声明 Mamba2 SSM 模型的后端能力。
    标记为 is_ssm=True 以表明这是状态空间模型后端。
    """

    @staticmethod
    def get_name() -> str:
        """返回后端名称标识。"""
        return "MAMBA2_ATTN"

    @staticmethod
    def get_builder_cls() -> type["Mamba2AttentionMetadataBuilder"]:
        """返回 Mamba2 元数据构建器类。"""
        return Mamba2AttentionMetadataBuilder

    @classmethod
    def is_ssm(cls) -> bool:
        """
        标识此为状态空间模型（SSM）后端。

        Mamba2 是改进版 SSM 架构，结合了 SSM 和注意力机制的数学对偶性。
        """
        return True


@dataclass
class Mamba2AttentionMetadata(BaseMambaAttentionMetadata):
    """
    Mamba2 注意力元数据数据类。

    在基类 BaseMambaAttentionMetadata 的基础上添加 Mamba2 特有的字段：

    - prep_initial_states: 是否需要准备初始状态
      当 prefill 请求有先前计算的隐状态（prefix caching 场景）时为 True。
      此标志告诉 Mamba2 kernel 在计算前先加载并准备初始状态。

    - chunk_size: SSD 算法的分块大小
      从模型配置中读取（通常为 256 或 64）。
      决定了 SSD kernel 每次处理的 token 数量。

    - seq_idx_p: prefill 请求中每个 token 所属的序列索引
      shape=[total_prefill_tokens]，用于变长批次中标识 token 归属。
      与 cu_chunk_seqlen_p 配合使用，支持 SSD kernel 的变长处理。
    """

    # 是否需要准备初始状态（有 prefix caching 的 prefill 时为 True）
    prep_initial_states: bool = False
    # SSD 算法的分块大小
    chunk_size: int = 0

    # Chunk-related metadata (only for prefill)
    # 分块相关元数据（仅 prefill 使用）
    # 每个 prefill token 所属的序列索引
    seq_idx_p: torch.Tensor | None = None


class Mamba2AttentionMetadataBuilder(
    BaseMambaAttentionMetadataBuilder[Mamba2AttentionMetadata]
):
    """
    Mamba2 注意力元数据构建器。

    在基类 BaseMambaAttentionMetadataBuilder 的基础上：
    1. 从模型配置获取 chunk_size（SSD 分块大小）
    2. 为 prefill 计算分块元数据和序列索引
    3. 检查是否有初始状态需要准备（prep_initial_states）

    build 方法处理流程：
    1. 调用 _compute_common_metadata 计算公共元数据
    2. 如果有 prefill 请求：
       a. 检查是否有初始状态（has_initial_states_p）
       b. 调用 _build_chunk_metadata_tensors 计算分块元数据
       c. 设置 prep_initial_states 标志
    3. 使用 replace 不可变地更新元数据对象

    chunk_size 说明：
    - 从 vllm_config.model_config.get_mamba_chunk_size() 获取
    - 通常为 256（Mamba2 标准配置）
    - 必须在模型配置中设置，否则断言失败
    - 决定 SSD kernel 的分块粒度，影响计算效率和内存使用
    """

    metadata_cls = Mamba2AttentionMetadata

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        # 从模型配置获取 Mamba2 的分块大小
        chunk_size = vllm_config.model_config.get_mamba_chunk_size()
        assert chunk_size is not None, (
            "chunk_size needs to be set in the model config for Mamba2 models"
        )
        self.chunk_size: int = chunk_size

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
        **kwargs: Any,
    ) -> Mamba2AttentionMetadata:
        """
        构建 Mamba2 注意力元数据。

        参数：
            common_prefix_len: 共享前缀长度（未使用）
            common_attn_metadata: 调度器输出的通用注意力元数据
            fast_build: 是否快速构建（未使用）
            **kwargs: 额外参数，可能包含：
                - num_accepted_tokens: 推测解码接受的 token 数
                - prev_last_scheduled_idx: 上一步的调度索引

        返回：
            Mamba2AttentionMetadata 对象
        """
        # 第一步：计算公共元数据
        common = self._compute_common_metadata(
            common_attn_metadata,
            num_accepted_tokens=kwargs.get("num_accepted_tokens"),
            prev_last_scheduled_idx=kwargs.get("prev_last_scheduled_idx"),
        )

        seq_idx_p = None
        cu_chunk_seqlen_p = None
        last_chunk_indices_p = None
        prep_initial_states = False

        # Compute seq_idx for prefill only
        # 仅在有 prefill 请求时计算分块元数据
        if common.num_prefills > 0:
            # 检查是否有初始状态需要准备
            # 当任一 prefill 请求有先前的隐状态时，设置为 True
            prep_initial_states = (
                torch.any(common.has_initial_states_p).item()
                if common.has_initial_states_p is not None
                else False
            )

            # 计算分块元数据：
            # - cu_chunk_seqlen_p：分块累积序列长度
            # - seq_idx_p：每个 token 所属的序列索引
            # - last_chunk_indices_p：每个序列最后一个块的索引
            cu_chunk_seqlen_p, seq_idx_p, last_chunk_indices_p = (
                self._build_chunk_metadata_tensors(
                    self.chunk_size,
                    common,
                    common_attn_metadata,
                )
            )

        # 使用 replace 不可变地更新元数据对象
        return replace(
            common,
            prep_initial_states=prep_initial_states,
            chunk_size=self.chunk_size,
            seq_idx_p=seq_idx_p,
            cu_chunk_seqlen_p=cu_chunk_seqlen_p,
            last_chunk_indices_p=last_chunk_indices_p,
        )
