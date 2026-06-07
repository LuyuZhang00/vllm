# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
后期交互评分运行器模块 (Late Interaction Runner Module)

本模块实现了后期交互 (Late Interaction) 评分的 worker 端状态管理和后处理。
后期交互是一种用于信息检索的评分方法，典型代表是 ColBERT 模型。

后期交互评分的工作原理：
1. Query 请求：编码查询文本，得到每个 token 的嵌入向量
2. Document 请求：编码文档文本，得到每个 token 的嵌入向量
3. 评分：计算 query 和 document 之间的 MaxSim 分数
   - 对于 query 中的每个 token，找到 document 中最相似的 token
   - 将所有最大相似度求平均

两种模式：
1. CACHE_QUERY: 缓存 query 的 token 嵌入，供后续 document 评分使用
2. SCORE_DOC: 使用缓存的 query 嵌入计算 document 的分数

缓存管理：
- query_cache: 存储 query 的 token 嵌入
- query_uses: 跟踪每个 query 还有多少 document 需要使用
- doc_query_keys: 记录每个 document 请求对应的 query key

当所有使用某个 query 的 document 完成后，自动释放 query 缓存。
"""
from collections.abc import Iterable

import torch

from vllm.pooling_params import PoolingParams
from vllm.v1.outputs import PoolerOutput
from vllm.v1.pool.late_interaction import (
    LATE_INTERACTION_MODE_CACHE_QUERY,
    LATE_INTERACTION_MODE_SCORE_DOC,
    compute_maxsim_score_batched,
)


class LateInteractionRunner:
    """Worker 端的后期交互评分状态管理和后处理。

    管理 query 缓存和 document-query 映射关系，
    处理后期交互评分的后处理逻辑。
    """

    def __init__(self) -> None:
        # query_key -> query 的 token 嵌入
        self._query_cache: dict[str, torch.Tensor] = {}
        # query_key -> 剩余需要使用此 query 的 document 数量
        self._query_uses: dict[str, int] = {}
        # document 请求 ID -> query key
        self._doc_query_keys: dict[str, str] = {}

    def clear(self) -> None:
        """清除所有缓存状态。"""
        self._query_cache.clear()
        self._query_uses.clear()
        self._doc_query_keys.clear()

    def register_request(
        self, req_id: str, pooling_params: PoolingParams | None
    ) -> None:
        """注册新请求的后期交互元数据。

        Args:
            req_id: 请求 ID
            pooling_params: 池化参数
        """
        mode, query_key, _ = self._parse_late_interaction_meta(pooling_params)
        if mode == LATE_INTERACTION_MODE_SCORE_DOC and query_key is not None:
            self._doc_query_keys[req_id] = query_key
        else:
            self._doc_query_keys.pop(req_id, None)

    def on_requests_finished(self, finished_req_ids: Iterable[str]) -> None:
        """处理完成的请求，释放 query 缓存引用。

        Args:
            finished_req_ids: 完成的请求 ID 列表
        """
        for req_id in finished_req_ids:
            query_key = self._doc_query_keys.pop(req_id, None)
            if query_key is not None:
                self._release_query_use(query_key)

    def postprocess_pooler_output(
        self,
        raw_pooler_output: PoolerOutput,
        pooling_params: list[PoolingParams],
        req_ids: list[str],
        finished_mask: list[bool],
    ) -> PoolerOutput:
        """后处理池化输出，执行后期交互评分。

        处理流程：
        1. 遍历所有完成的请求
        2. CACHE_QUERY 模式：缓存 query 的 token 嵌入
        3. SCORE_DOC 模式：使用缓存的 query 嵌入计算 MaxSim 分数
        4. 批量计算所有 document 的分数

        Args:
            raw_pooler_output: 原始池化输出
            pooling_params: 每个请求的池化参数
            req_ids: 请求 ID 列表
            finished_mask: 完成状态掩码

        Returns:
            处理后的池化输出
        """
        if not isinstance(raw_pooler_output, list):
            return raw_pooler_output

        num_reqs = len(pooling_params)
        if len(raw_pooler_output) != num_reqs:
            raise ValueError(
                "raw_pooler_output and pooling_params must have the same length."
            )
        if len(req_ids) != num_reqs:
            raise ValueError("req_ids and pooling_params must have the same length.")
        if len(finished_mask) != num_reqs:
            raise ValueError(
                "finished_mask and pooling_params must have the same length."
            )

        if not any(finished_mask):
            return raw_pooler_output
        if not any(p.late_interaction_params is not None for p in pooling_params):
            return raw_pooler_output

        outputs: list[torch.Tensor | None] = list(raw_pooler_output)
        score_indices: list[int] = []
        score_req_ids: list[str] = []
        score_query_keys: list[str] = []
        score_queries: list[torch.Tensor] = []
        score_docs: list[torch.Tensor] = []
        for i, (req_id, output, params, finished) in enumerate(
            zip(req_ids, outputs, pooling_params, finished_mask)
        ):
            if not finished or output is None:
                continue

            mode, query_key, query_uses = self._parse_late_interaction_meta(params)
            if mode is None:
                continue

            assert query_key is not None
            if mode == LATE_INTERACTION_MODE_CACHE_QUERY:
                assert query_uses is not None
                # output 可能是当前步骤隐藏状态缓冲区的视图，
                # 因此在跨调度步骤存储前需要克隆
                self._query_cache[query_key] = output.clone()
                self._query_uses[query_key] = query_uses
                outputs[i] = torch.zeros((), device=output.device, dtype=torch.float32)
                continue

            if mode == LATE_INTERACTION_MODE_SCORE_DOC:
                query_output = self._query_cache.get(query_key)
                if query_output is None:
                    raise ValueError(
                        "late-interaction query cache miss for key "
                        f"{query_key!r}. Ensure query requests are executed "
                        "before their paired document requests."
                    )

                score_indices.append(i)
                score_req_ids.append(req_id)
                score_query_keys.append(query_key)
                score_queries.append(query_output)
                score_docs.append(output)
                continue

            raise ValueError(f"Unsupported late-interaction mode: {mode!r}")

        if score_indices:
            # 批量计算所有 document 的 MaxSim 分数
            score_values = compute_maxsim_score_batched(score_queries, score_docs)
            for i, req_id, query_key, score in zip(
                score_indices, score_req_ids, score_query_keys, score_values
            ):
                outputs[i] = score
                self._doc_query_keys.pop(req_id, None)
                self._release_query_use(query_key)

        return outputs

    def _release_query_use(self, query_key: str) -> None:
        """释放 query 缓存的一个引用。

        当引用计数降为 0 时，清除 query 缓存。

        Args:
            query_key: query 的唯一标识符
        """
        remaining = self._query_uses.get(query_key, 1) - 1
        if remaining <= 0:
            self._query_uses.pop(query_key, None)
            self._query_cache.pop(query_key, None)
        else:
            self._query_uses[query_key] = remaining

    @staticmethod
    def _parse_late_interaction_meta(
        pooling_params: PoolingParams | None,
    ) -> tuple[str | None, str | None, int | None]:
        """解析后期交互元数据。

        Args:
            pooling_params: 池化参数

        Returns:
            (mode, query_key, query_uses) 元组
        """
        if pooling_params is None or pooling_params.late_interaction_params is None:
            return None, None, None

        late_interaction_params = pooling_params.late_interaction_params
        mode = late_interaction_params.mode

        query_key = late_interaction_params.query_key
        if not isinstance(query_key, str) or not query_key:
            raise ValueError(
                "late-interaction request is missing a valid query key in "
                "pooling_params.late_interaction_params."
            )

        if mode == LATE_INTERACTION_MODE_CACHE_QUERY:
            query_uses_raw = late_interaction_params.query_uses
            if query_uses_raw is None:
                query_uses_raw = 1
            try:
                query_uses = max(1, int(query_uses_raw))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "late-interaction query uses must be an integer value."
                ) from exc
            return mode, query_key, query_uses

        return mode, query_key, None
