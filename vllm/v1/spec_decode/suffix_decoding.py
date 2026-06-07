# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Suffix Decoding 投机解码提议器模块

本模块实现了基于后缀匹配的投机解码（Suffix Decoding）提议器。
参考论文：https://arxiv.org/pdf/2411.04975

核心思想：
  不同于基于 draft model 的投机解码（如 EAGLE、Medusa），Suffix Decoding 利用
  后缀树（suffix tree）在已有的请求输出历史中查找匹配模式，直接复用历史 token
  作为 draft token。这种方法无需额外的 draft model，开销极低。

工作流程：
  1. 每个请求开始时，构建其 prompt 的后缀树（由 ArcticInference 库管理）
  2. 每个解码步，将新采样的 token 追加到请求的响应历史中
  3. 用最近 max_tree_depth 个 token 作为模式串，在后缀树中查找匹配
  4. 匹配到的后续 token 即为 draft token，数量由 max_spec_factor 和 min_token_prob 控制

依赖：需要安装 arctic_inference 库（snowflakedb/ArcticInference）
"""

import torch

from vllm.config import VllmConfig
from vllm.v1.worker.gpu_input_batch import InputBatch


class SuffixDecodingProposer:
    """
    Speculative decoding proposer for Suffix Decoding (https://arxiv.org/pdf/2411.04975).
    This class imports and uses the official implementation from Arctic Inference
    (https://github.com/snowflakedb/ArcticInference).
    """

    # 中文注释：基于后缀匹配的投机解码提议器。
    # 与基于 draft model 的方法不同，它不使用额外的小模型来预测 token，
    # 而是在请求输出历史中通过后缀树模式匹配来推测后续 token。
    # 优势：无额外模型开销；劣势：对全新内容（无历史匹配）无加速效果。

    def __init__(self, vllm_config: VllmConfig):
        """中文注释：初始化 Suffix Decoding 提议器。

        参数：
            vllm_config: vLLM 全局配置，包含投机解码配置和模型配置

        配置项说明：
            num_speculative_tokens: 每步最多推测的 token 数上限
            suffix_decoding_max_tree_depth: 后缀树最大深度，即匹配时使用的历史 token 数
            suffix_decoding_max_spec_factor: 推测因子，控制推测长度与匹配长度的比例
            suffix_decoding_min_token_prob: 最小 token 概率阈值，低于此阈值的匹配被丢弃
        """
        config = vllm_config.speculative_config
        assert config is not None, "Speculative config must be set"
        self.num_speculative_tokens = config.num_speculative_tokens
        self.max_tree_depth = config.suffix_decoding_max_tree_depth
        self.max_spec_factor = config.suffix_decoding_max_spec_factor
        self.min_token_prob = config.suffix_decoding_min_token_prob
        self.max_model_len = vllm_config.model_config.max_model_len

        # Lazy import to avoid error when Suffix Decoding is not used.
        from arctic_inference.suffix_decoding import SuffixDecodingCache

        # Initialize and empty cache. This object will take care of caching request
        # outputs, evicting old requests, and manages the per-prompt suffix trees.
        # 中文注释：初始化后缀解码缓存。
        # SuffixDecodingCache 负责：
        #   1. 为每个请求维护独立的后缀树
        #   2. 缓存请求输出历史，供后续匹配使用
        #   3. 自动淘汰旧请求以控制内存使用
        self.suffix_cache = SuffixDecodingCache(
            max_tree_depth=config.suffix_decoding_max_tree_depth,
            max_cached_requests=config.suffix_decoding_max_cached_requests,
        )

    def propose(
        self,
        input_batch: InputBatch,
        sampled_token_ids: list[list[int]],
        slot_mappings: dict[str, torch.Tensor]
        | list[dict[str, torch.Tensor]]
        | None = None,  # unused
    ) -> list[list[int]]:
        """
        Propose speculative tokens for each request in the input batch. Suffix Decoding
        will speculate a dynamic number of tokens for each request every decoding step,
        so each entry in the returned list may have different lengths.
        """
        # 中文注释：为输入批次中的每个请求生成投机 token。
        # Suffix Decoding 的特点是每个请求推测的 token 数量可能不同，
        # 因为匹配到的后缀长度取决于历史模式的匹配程度。

        # 中文注释：结果列表，每个元素是一个请求的 draft token id 列表
        draft_token_ids: list[list[int]] = []
        for i, sampled_ids in enumerate(sampled_token_ids):
            # 中文注释：步骤 1 - 跳过 partial prefill（尚未完成 prefill 的请求）
            if not sampled_ids:
                # Skip speculative decoding for partial prefills.
                draft_token_ids.append([])
                continue

            req_id = input_batch.req_ids[i]
            num_tokens = input_batch.num_tokens_no_spec[i]
            # 中文注释：步骤 2 - 跳过已达最大模型长度的请求，无法再推测
            if num_tokens >= self.max_model_len:
                # Skip requests that have already reached the max model length.
                draft_token_ids.append([])
                continue

            index = input_batch.req_id_to_index[req_id]
            # 中文注释：步骤 3 - 如果是新请求，初始化后缀树。
            # 首先检查是否在活跃请求中；如果不在但有缓存，先驱逐缓存再重新构建。
            if req_id not in self.suffix_cache.active_requests:
                if req_id in self.suffix_cache.cached_requests:
                    # Reset the suffix cache for this request.
                    self.suffix_cache.evict_cached_response(req_id)
                num_prompt_tokens = input_batch.num_prompt_tokens[index]
                prompt_token_ids = input_batch.token_ids_cpu[index, :num_prompt_tokens]
                # Start a new request, this will build the suffix tree for that prompt.
                # 中文注释：用 prompt token 构建后缀树，后续匹配将在此树上进行
                self.suffix_cache.start_request(req_id, prompt_token_ids)

            # Append the newly sampled ids to the suffix cache for this request.
            # 中文注释：步骤 4 - 将本轮新采样的 token 追加到请求的响应历史中，
            # 丰富后缀树的匹配数据源
            self.suffix_cache.add_active_response(req_id, sampled_ids)

            # Suffix decoding only uses the most recent tokens up to max_tree_depth, so
            # we extract the pattern from the end of the input.
            # 中文注释：步骤 5 - 提取最近 max_tree_depth 个 token 作为匹配模式串。
            # 只用最近的 token 是因为越近的上下文对预测未来越有价值。
            start = max(0, num_tokens - self.max_tree_depth)
            pattern = input_batch.token_ids_cpu[i, start:num_tokens]
            # 中文注释：步骤 6 - 在后缀树中查找匹配，返回 draft token。
            # max_spec_tokens 取 num_speculative_tokens 和剩余空间的较小值。
            # max_spec_factor 控制推测长度与匹配长度的比例上限。
            # min_token_prob 过滤低概率匹配，避免低质量 draft。
            draft = self.suffix_cache.speculate(
                req_id,
                pattern,
                max_spec_tokens=min(
                    self.num_speculative_tokens, self.max_model_len - num_tokens - 1
                ),
                max_spec_factor=self.max_spec_factor,
                min_token_prob=self.min_token_prob,
            )

            draft_token_ids.append(draft.token_ids)

        # Stop requests that were not seen in the input batch.
        # 中文注释：步骤 7 - 清理不在当前批次中的请求。
        # 对于已完成或被移除的请求，停止其后缀树追踪以释放资源。
        for req_id in (
            self.suffix_cache.active_requests - input_batch.req_id_to_index.keys()
        ):
            self.suffix_cache.stop_request(req_id)

        return draft_token_ids

    def load_model(self, *args, **kwargs):
        # No model to load.
        # 中文注释：Suffix Decoding 不需要加载额外的 draft model，
        # 因为它完全基于后缀树模式匹配来推测 token，无需神经网络推理。
        pass
