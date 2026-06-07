# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
晚期交互 (Late Interaction) 模块 (vllm/v1/pool/late_interaction.py)

本模块实现了晚期交互模型（如 ColBERT）的支持功能。

晚期交互模型概述：
- 传统嵌入模型：将整个文档编码为单个向量，用点积计算相似度
- 晚期交互模型：将查询和文档分别编码为 token 级别的向量序列，
  然后通过 MaxSim（最大相似度）操作计算相关性

MaxSim 计算过程：
1. 查询 Q 编码为 [q1, q2, ..., qm]（m 个 token 向量）
2. 文档 D 编码为 [d1, d2, ..., dn]（n 个 token 吽量）
3. 计算相似度矩阵 S[i][j] = dot(qi, dj)
4. 对每个查询 token qi，取其与所有文档 token 的最大相似度：max_j(S[i][j])
5. 将所有查询 token 的最大相似度求和：score = sum_i(max_j(S[i][j]))

模块提供两种晚期交互模式：
1. cache_query: 缓存查询嵌入，在进程本地内存中存储
   - 相同 query_key 的请求固定到同一引擎（通过哈希分片）
   - 避免重复计算查询嵌入
2. score_doc: 计算文档嵌入并使用缓存的查询嵌入计算分数

辅助函数：
- get_late_interaction_engine_index: 根据 query_key 计算引擎索引
- build_late_interaction_query_params: 构建查询参数
- build_late_interaction_doc_params: 构建文档参数
- compute_maxsim_score_batched: 批量计算 MaxSim 分数
"""

import zlib
from collections.abc import Sequence

import torch

from vllm.pooling_params import LateInteractionParams, PoolingParams

# 两种晚期交互模式
LATE_INTERACTION_MODE_CACHE_QUERY = "cache_query"   # 缓存查询嵌入
LATE_INTERACTION_MODE_SCORE_DOC = "score_doc"       # 计算文档分数


def get_late_interaction_engine_index(
    pooling_params: PoolingParams | None,
    num_engines: int,
) -> int | None:
    """
    根据请求的 query_key 计算应路由到的引擎索引。

    设计目的：
    - 查询嵌入缓存在进程本地内存中
    - 相同 query_key 的请求必须路由到同一引擎，才能命中缓存
    - 使用 CRC32 哈希实现一致性路由

    Args:
        pooling_params: 池化参数
        num_engines: 引擎总数

    Returns:
        引擎索引，如果不是晚期交互模式则返回 None
    """
    if pooling_params is None or pooling_params.late_interaction_params is None:
        return None

    late_interaction_params = pooling_params.late_interaction_params
    mode = late_interaction_params.mode
    if mode not in (
        LATE_INTERACTION_MODE_CACHE_QUERY,
        LATE_INTERACTION_MODE_SCORE_DOC,
    ):
        return None

    query_key = late_interaction_params.query_key
    if not isinstance(query_key, str) or not query_key:
        return None

    # 查询嵌入缓存在进程本地内存中，
    # 将共享相同 query_key 的请求固定到同一引擎。
    return zlib.crc32(query_key.encode("utf-8")) % num_engines


def build_late_interaction_query_params(
    query_key: str,
    query_uses: int,
) -> LateInteractionParams:
    """
    构建晚期交互的查询参数。

    用于发起查询嵌入计算的请求。query_uses 指定查询嵌入将被使用多少次
    （对应后续的文档打分请求数）。

    Args:
        query_key: 查询的唯一标识符
        query_uses: 查询嵌入的使用次数

    Returns:
        LateInteractionParams 对象
    """
    return LateInteractionParams(
        mode=LATE_INTERACTION_MODE_CACHE_QUERY,
        query_key=query_key,
        query_uses=max(1, int(query_uses)),
    )


def build_late_interaction_doc_params(
    query_key: str,
) -> LateInteractionParams:
    """
    构建晚期交互的文档参数。

    用于发起文档嵌入计算和 MaxSim 分数计算的请求。
    会使用之前缓存的相同 query_key 的查询嵌入。

    Args:
        query_key: 查询的唯一标识符（必须与之前的查询请求匹配）

    Returns:
        LateInteractionParams 对象
    """
    return LateInteractionParams(
        mode=LATE_INTERACTION_MODE_SCORE_DOC,
        query_key=query_key,
    )


def compute_maxsim_score_batched(
    q_embs: Sequence[torch.Tensor],
    d_embs: Sequence[torch.Tensor],
    max_batch_size: int = 64,
    max_score_matrix_elements: int = 64_000_000,
) -> list[torch.Tensor]:
    """
    批量计算多个查询/文档对的 MaxSim 分数。

    MaxSim 计算过程：
    1. 对每个查询 token qi，计算其与所有文档 token 的点积相似度
    2. 取每个查询 token 的最大相似度
    3. 将所有查询 token 的最大相似度求和

    分批处理策略：
    - 每批最多 max_batch_size 个对
    - 分数矩阵元素数不超过 max_score_matrix_elements（避免内存溢出）
    - 如果当前批次超出限制，逐步缩小批次大小

    Args:
        q_embs: 查询嵌入序列，每个形状为 (num_query_tokens, dim)
        d_embs: 文档嵌入序列，每个形状为 (num_doc_tokens, dim)
        max_batch_size: 每批最大对数
        max_score_matrix_elements: 分数矩阵的最大元素数

    Returns:
        每对的 MaxSim 分数列表

    Raises:
        ValueError: 输入长度不匹配、维度不匹配、不在同一设备等
    """
    if len(q_embs) != len(d_embs):
        raise ValueError("q_embs and d_embs must have the same length")

    num_pairs = len(q_embs)
    if num_pairs == 0:
        return []

    if max_batch_size <= 0:
        raise ValueError("max_batch_size must be greater than 0")
    if max_score_matrix_elements <= 0:
        raise ValueError("max_score_matrix_elements must be greater than 0")

    for q_emb, d_emb in zip(q_embs, d_embs):
        if q_emb.ndim != 2 or d_emb.ndim != 2:
            raise ValueError("Each embedding tensor must be 2-D")
        if q_emb.shape[1] != d_emb.shape[1]:
            raise ValueError("Query and document embeddings must have same dim")
        if q_emb.device != d_emb.device:
            raise ValueError("Query and document embeddings must be on same device")

    scores: list[torch.Tensor] = []
    start = 0
    while start < num_pairs:
        end = min(start + max_batch_size, num_pairs)
        max_q = max(int(x.shape[0]) for x in q_embs[start:end])
        max_d = max(int(x.shape[0]) for x in d_embs[start:end])

        # 保持分数矩阵在限制范围内，避免过大的内存分配。
        while (
            end - start > 1
            and (end - start) * max_q * max_d > max_score_matrix_elements
        ):
            end -= 1
            max_q = max(int(x.shape[0]) for x in q_embs[start:end])
            max_d = max(int(x.shape[0]) for x in d_embs[start:end])

        batch_q = q_embs[start:end]
        batch_d = d_embs[start:end]
        batch_size = end - start
        device = batch_q[0].device
        dim = int(batch_q[0].shape[1])

        # 创建填充后的批次张量
        q_batch = torch.zeros(
            (batch_size, max_q, dim), dtype=torch.float32, device=device
        )
        d_batch = torch.zeros(
            (batch_size, max_d, dim), dtype=torch.float32, device=device
        )
        q_mask = torch.zeros((batch_size, max_q), dtype=torch.bool, device=device)
        d_mask = torch.zeros((batch_size, max_d), dtype=torch.bool, device=device)

        # 将数据拷贝到填充张量中
        for i, (q_emb, d_emb) in enumerate(zip(batch_q, batch_d)):
            q_len = int(q_emb.shape[0])
            d_len = int(d_emb.shape[0])
            q_batch[i, :q_len] = q_emb.to(device=device, dtype=torch.float32)
            d_batch[i, :d_len] = d_emb.to(device=device, dtype=torch.float32)
            q_mask[i, :q_len] = True
            d_mask[i, :d_len] = True

        # 计算 token 级别相似度矩阵：(batch, max_q, max_d)
        token_scores = torch.bmm(q_batch, d_batch.transpose(1, 2))
        # 将填充位置的相似度设为 -inf（排除在 max 之外）
        token_scores.masked_fill_(~d_mask.unsqueeze(1), float("-inf"))
        # 对每个查询 token，取与所有文档 token 的最大相似度
        max_per_query = token_scores.amax(dim=-1)
        # 将填充的查询 token 相似度设为 0
        max_per_query.masked_fill_(~q_mask, 0.0)
        # 求和得到最终分数
        batch_scores = max_per_query.sum(dim=-1)
        scores.extend(batch_scores.unbind(0))
        start = end

    return scores
