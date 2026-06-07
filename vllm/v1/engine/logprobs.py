# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
对数概率（Logprobs）处理模块。

本模块负责处理和管理模型输出的对数概率信息，包括：

1. 采样对数概率（Sample Logprobs）：
   - 每个生成步骤中，被采样 token 及其 top-k 候选 token 的对数概率
   - 用于展示模型在每个位置的概率分布

2. 提示对数概率（Prompt Logprobs）：
   - 输入 prompt 中每个 token 的对数概率
   - 用于评估模型对输入的理解程度

主要组件：
- LogprobsProcessor: 核心处理器，负责：
  1. 从 EngineCore 接收原始的对数概率张量
  2. 将张量数据转换为 Python 可用的数据结构
  3. 处理 token 的反标记化（将 token ID 转换为文本）
  4. 修复 UTF-8 多字节字符的解码问题
  5. 管理提示对数概率的聚合和弹出

数据流：
EngineCore -> LogprobsTensors/LogprobsLists -> LogprobsProcessor -> SampleLogprobs/PromptLogprobs
"""
import itertools
from collections.abc import Iterable
from dataclasses import dataclass

from vllm.logger import init_logger
from vllm.logprobs import (
    FlatLogprobs,
    PromptLogprobs,
    SampleLogprobs,
    append_logprobs_for_next_position,
    create_prompt_logprobs,
    create_sample_logprobs,
)
from vllm.tokenizers.detokenizer_utils import (
    TokenizerLike,
    convert_ids_list_to_tokens,
)
from vllm.v1.engine import EngineCoreOutput, EngineCoreRequest
from vllm.v1.outputs import LogprobsLists, LogprobsTensors

logger = init_logger(__name__)

# 用于生成无限 None 迭代器，当 tokenizer 不可用时使用
NONES = itertools.repeat(None)


@dataclass
class LogprobsProcessor:
    """
    对数概率处理器。

    负责处理来自 EngineCore 的对数概率数据，将其转换为前端可用的格式。
    每个请求对应一个 LogprobsProcessor 实例。

    处理流程：
    1. 从 EngineCoreOutput 接收原始张量数据
    2. 将 numpy/torch 张量转换为 Python 列表
    3. 对 token ID 进行反标记化
    4. 修复 UTF-8 多字节字符的边界解码问题
    5. 累积到输出数据结构中
    """

    # Tokenizer for this request,
    # None if detokenization is disabled.
    # 本请求使用的分词器，如果禁用了反标记化则为 None
    tokenizer: TokenizerLike | None

    # Logprobs for this request
    # 本请求的对数概率数据
    # 采样对数概率（每个生成步骤的 top-k 对数概率）
    logprobs: SampleLogprobs | None
    # 提示对数概率（prompt 中每个 token 的对数概率）
    prompt_logprobs: PromptLogprobs | None
    # 累积对数概率（所有已采样 token 的对数概率之和）
    cumulative_logprob: float | None
    # 每个位置返回的候选 token 数量（不含采样 token 本身）
    num_logprobs: int | None
    # prompt 每个位置返回的候选 token 数量
    num_prompt_logprobs: int | None

    @classmethod
    def from_new_request(
        cls,
        tokenizer: TokenizerLike | None,
        request: EngineCoreRequest,
    ) -> "LogprobsProcessor":
        """
        工厂方法：从新请求创建 LogprobsProcessor 实例。

        根据请求的采样参数决定是否启用对数概率记录：
        - num_logprobs: 采样对数概率，None 表示禁用
        - num_prompt_logprobs: 提示对数概率，None 表示禁用

        参数:
            tokenizer: 分词器实例
            request: 引擎核心请求对象

        返回:
            新创建的 LogprobsProcessor 实例
        """
        sampling_params = request.sampling_params
        assert sampling_params is not None
        num_logprobs = sampling_params.num_logprobs
        num_prompt_logprobs = sampling_params.prompt_logprobs
        return cls(
            tokenizer=tokenizer,
            cumulative_logprob=(None if num_logprobs is None else 0.0),
            logprobs=(
                None
                if num_logprobs is None
                else create_sample_logprobs(sampling_params.flat_logprobs)
            ),
            prompt_logprobs=(
                None
                if num_prompt_logprobs is None
                else create_prompt_logprobs(sampling_params.flat_logprobs)
            ),
            num_prompt_logprobs=num_prompt_logprobs,
            num_logprobs=num_logprobs,
        )

    def _update_sample_logprobs(self, logprobs_lists: LogprobsLists) -> None:
        """Update with sample logprobs from EngineCore.

        Outer lists are only of len > 1 if EngineCore made
        >1 tokens in prior step (e.g. in spec decoding).

        Args:
          logprobs_lists: the lists of logprob tokens, logprobs, and ranks.

        """
        """
        使用来自 EngineCore 的采样对数概率更新处理器状态。

        处理步骤：
        1. 遍历每个生成步骤的对数概率数据
        2. 将 numpy 数组转换为 Python 列表
        3. 对 token ID 进行反标记化
        4. 修复 UTF-8 多字节字符的解码问题
        5. 累积对数概率
        6. 追加到输出数据结构

        注意：外层列表长度 > 1 的情况发生在 EngineCore 一次生成多个 token 时
        （例如投机解码中的多 token 生成）。

        参数:
            logprobs_lists: 包含 token IDs、对数概率和排名的列表
        """

        assert self.num_logprobs is not None
        assert self.logprobs is not None
        assert self.cumulative_logprob is not None

        token_ids_lst, logprobs_lst, ranks_lst, _ = logprobs_lists

        for rank_np, logprobs_np, token_ids_np in zip(
            ranks_lst, logprobs_lst, token_ids_lst
        ):
            # 将 numpy 数组转换为 Python 列表
            rank = rank_np.tolist()
            logprobs = logprobs_np.tolist()
            token_ids = token_ids_np.tolist()
            # Detokenize (non-incrementally).
            # 非增量反标记化：将 token ID 转换为文本字符串
            decoded_tokens: list[str] | Iterable[None]
            if self.tokenizer is None:
                # 没有 tokenizer，使用无限 None 迭代器
                decoded_tokens = NONES
            else:
                # 将 token ID 列表转换为 token 字符串列表
                decoded_tokens_list = convert_ids_list_to_tokens(
                    self.tokenizer, token_ids
                )
                # 获取前序采样 token 的上下文，用于修复 UTF-8 解码
                context_token_ids = self._get_sampled_context_ids(self.logprobs)
                # 验证并修复包含替换字符的 token
                decoded_tokens = self._verify_tokens(
                    decoded_tokens_list=decoded_tokens_list,
                    tokens=token_ids,
                    context_token_ids=context_token_ids,
                )

            # Sampler puts the sampled logprob in first.
            # 采样器将被采样 token 的对数概率放在列表第一位
            sampled_token_logprob = logprobs[0]
            # 累积对数概率
            self.cumulative_logprob += sampled_token_logprob

            # Update with the Logprob container for this pos.
            # 将当前位置的对数概率数据追加到输出容器中
            append_logprobs_for_next_position(
                self.logprobs,
                token_ids,
                logprobs,
                decoded_tokens,
                rank,
                self.num_logprobs,
            )

    def _update_prompt_logprobs(
        self,
        prompt_logprobs_tensors: LogprobsTensors,
    ) -> None:
        """Update with prompt logprobs from EngineCore.

        Args:
          prompt_logprobs_tensors: tuple containing the prompt logprobs
                                   tensors.

        """
        """
        使用来自 EngineCore 的提示对数概率更新处理器状态。

        处理步骤：
        1. 从张量中恢复形状信息
        2. 批量反标记化所有 token
        3. 将张量数据转换为 Python 数据结构
        4. 逐位置处理对数概率，包括 UTF-8 修复
        5. 追加到输出数据结构

        参数:
            prompt_logprobs_tensors: 包含 token IDs、对数概率、排名的元组
        """

        # Prompt logprobs are enabled.
        assert self.num_prompt_logprobs is not None
        assert self.prompt_logprobs is not None

        token_ids, logprobs, ranks, _ = prompt_logprobs_tensors

        # Recover shapes.
        # 恢复张量形状：[prompt_token数量, logprobs数量]
        num_prompt_tokens, num_logprobs = logprobs.shape

        # Detokenize non-incrementally.
        # Output is flat: [num_tok, num_lps] -> [num_tok * num_lps]
        # 非增量批量反标记化：将二维展平为一维后解码
        all_decoded_tokens: list[str] | None = (
            None
            if self.tokenizer is None
            else convert_ids_list_to_tokens(
                self.tokenizer, token_ids.flatten().tolist()
            )
        )

        # Pythonize the torch tensors.
        # 将 PyTorch 张量转换为 Python 列表
        prompt_token_ranks = ranks.tolist()
        prompt_logprobs = logprobs.tolist()
        token_ids_list = token_ids.tolist()

        # Make Logprob for each position.
        # 逐位置处理对数概率
        for pos in range(num_prompt_tokens):
            # Handle flattening and UTF-8 correction per position
            # 计算展平后的偏移量，用于提取当前 position 的解码结果
            offset = pos * num_logprobs
            offset_end = offset + num_logprobs

            decoded_tokens_for_pos: list[str] | Iterable[None]
            if all_decoded_tokens is None:
                decoded_tokens_for_pos = NONES
            else:
                # Extract decoded tokens for this position
                # 提取当前位置的解码结果
                decoded_tokens_slice = all_decoded_tokens[offset:offset_end]
                # Context: preceding prompt tokens accumulated in
                # self.prompt_logprobs from previous loop iterations.
                # 获取前序 prompt token 作为上下文（用于 UTF-8 修复）
                context_token_ids = self._get_sampled_context_ids(self.prompt_logprobs)
                # Apply UTF-8 correction within this position's token boundaries
                # 应用 UTF-8 修复
                decoded_tokens_for_pos = self._verify_tokens(
                    decoded_tokens_list=decoded_tokens_slice,
                    tokens=token_ids_list[pos],
                    context_token_ids=context_token_ids,
                )

            # Update with the Logprob container for this pos.
            # 将当前位置的对数概率追加到输出容器
            append_logprobs_for_next_position(
                self.prompt_logprobs,
                token_ids_list[pos],
                prompt_logprobs[pos],
                decoded_tokens_for_pos,
                prompt_token_ranks[pos],
                self.num_prompt_logprobs,
            )

    def pop_prompt_logprobs(self) -> PromptLogprobs | None:
        """Pop and return all request prompt logprobs

        The logprobs processor aggregates prompt chunk logprobs
        over one or more prefill chunks. This method returns
        all prompt logprobs at once and then forgets them.
        Ensures correct RequestOutputKind.DELTA semantics
        wherein all prompt logprobs are returned at once at
        the end of prefill.

        Returns:
          None if prompt logprobs are disabled for this request.
          List of all prompt logprobs, otherwise.
        """
        """
        弹出并返回所有提示对数概率。

        提示对数概率可能跨越多个 prefill chunk 累积而来。
        此方法一次性返回所有提示对数概率，然后清空内部状态。
        这确保了 RequestOutputKind.DELTA 语义的正确性，
        即所有提示对数概率在 prefill 结束时一次性返回。

        返回:
            如果提示对数概率被禁用，返回 None。
            否则返回所有提示对数概率列表。
        """
        plp = self.prompt_logprobs
        if plp:
            self.prompt_logprobs = []
        return plp

    @staticmethod
    def _get_sampled_context_ids(
        logprobs_source: SampleLogprobs | PromptLogprobs | None,
        max_context: int = 4,
    ) -> list[int]:
        """Extract recent sampled token IDs from a logprobs source.

        The sampled (or prompt) token at each position is the first
        entry, since it is always inserted first by
        append_logprobs_for_next_position.

        Args:
            logprobs_source: The logprobs container to extract from.
            max_context: Maximum number of preceding tokens to return.
                4 is sufficient for any UTF-8 multi-byte sequence.

        Returns:
            List of sampled token IDs, oldest first, most recent last.
        """
        """
        从对数概率数据源中提取最近的采样 token ID。

        每个位置的采样 token（或 prompt token）是该位置对数概率字典中的
        第一个条目，因为它总是由 append_logprobs_for_next_position 最先插入。

        这些上下文 token ID 用于修复 UTF-8 多字节字符的解码问题。
        UTF-8 字符最多使用 4 个字节，因此 max_context=4 足够覆盖任何情况。

        参数:
            logprobs_source: 要提取的数据源（SampleLogprobs 或 PromptLogprobs）
            max_context: 返回的最大前序 token 数量，默认 4

        返回:
            采样 token ID 列表，按从旧到新的顺序排列
        """
        if not logprobs_source:
            return []

        n = len(logprobs_source)
        start = max(0, n - max_context)

        # Efficient path for FlatLogprobs: access token_ids directly.
        # FlatLogprobs 的高效路径：直接访问 token_ids 数组
        if isinstance(logprobs_source, FlatLogprobs):
            return [
                logprobs_source.token_ids[logprobs_source.start_indices[i]]
                for i in range(start, n)
                if logprobs_source.start_indices[i] < logprobs_source.end_indices[i]
            ]

        # list[dict] path
        # 标准路径：从字典列表中提取第一个 key（即采样 token ID）
        result: list[int] = []
        for i in range(start, n):
            entry = logprobs_source[i]
            if entry is not None:
                result.append(next(iter(entry)))
        return result

    def _correct_decoded_token(
        self, token_id: int, context_token_ids: list[int]
    ) -> str:
        """Correct a decoded token that contains the replacement character.

        When byte-fallback tokenization splits multi-byte UTF-8
        characters across tokens, individual token decoding produces
        the replacement character U+FFFD. This method uses preceding
        sampled tokens as context to reconstruct the correct text.

        Args:
            token_id: The single token ID to correct.
            context_token_ids: Preceding sampled token IDs in sequential
                order (oldest first). These are the actual tokens in
                the generated sequence, NOT top-k alternatives.

        Returns:
            The corrected decoded string, or empty string if the byte
            sequence is genuinely incomplete at this point.
        """
        """
        修复包含替换字符（U+FFFD）的解码 token。

        背景：当字节回退（byte-fallback）分词器将多字节 UTF-8 字符拆分到
        多个 token 时，单独解码每个 token 会产生 Unicode 替换字符 U+FFFD。
        此方法使用前序采样 token 作为上下文来重建正确的文本。

        算法：
        1. 从 1 到 max_ctx 逐步增加上下文长度
        2. 将上下文 token 与当前 token 一起解码
        3. 如果解码结果不再以替换字符结尾，说明找到了完整的 UTF-8 序列
        4. 从完整解码结果中去除前序干净 token 的部分，得到当前 token 的正确文本

        参数:
            token_id: 需要修复的单个 token ID
            context_token_ids: 前序采样 token ID 列表（从旧到新），
                这些是实际生成序列中的 token，不是 top-k 替代 token

        返回:
            修复后的解码字符串。如果字节序列在此处确实不完整，返回空字符串。
        """
        assert self.tokenizer is not None

        max_ctx = min(len(context_token_ids), 4)

        for num_ctx in range(1, max_ctx + 1):
            # 取最后 num_ctx 个上下文 token
            context = context_token_ids[-num_ctx:]
            # 将上下文 token 与当前 token 一起解码
            full_decoded = self.tokenizer.decode(context + [token_id])

            if full_decoded.endswith("�"):
                # 仍然以替换字符结尾，说明还需要更多上下文
                continue

            # Find the boundary between "clean" context tokens and
            # byte-fallback tokens that are part of the same incomplete
            # sequence. Byte-fallback context tokens returned "" when
            # they were processed, so their text must be attributed to
            # this completing token.
            # 找到"干净"上下文 token 和属于同一不完整序列的字节回退 token 之间的边界
            # 字节回退的上下文 token 在之前处理时返回了空字符串，
            # 因此它们的文本应归属于当前这个完成序列的 token
            clean_end = len(context)
            for j in range(len(context) - 1, -1, -1):
                if self.tokenizer.decode([context[j]]).endswith("�"):
                    clean_end = j
                else:
                    break

            # Decode only the clean (non-byte-fallback) prefix.
            # 只解码干净的（非字节回退的）前缀部分
            if clean_end > 0:
                clean_prefix = self.tokenizer.decode(context[:clean_end])
            else:
                clean_prefix = ""

            if full_decoded.startswith(clean_prefix):
                # 从完整解码结果中去除干净前缀，得到当前 token 的正确文本
                return full_decoded[len(clean_prefix) :]

            # Tokenizer normalization may cause prefix mismatch.
            # Find the longest common prefix between them.
            # Tokenizer 规范化可能导致前缀不匹配，
            # 找到最长公共前缀来处理这种情况
            common_len = 0
            for a, b in zip(clean_prefix, full_decoded):
                if a != b:
                    break
                common_len += 1
            return full_decoded[common_len:]

        # 所有上下文长度尝试都失败，返回空字符串
        return ""

    def _verify_tokens(
        self,
        decoded_tokens_list: list[str],
        tokens: list[int],
        context_token_ids: list[int] | None = None,
    ) -> list[str]:
        """Verify and correct decoded tokens with replacement characters.

        Args:
            decoded_tokens_list: Decoded token strings to verify.
            tokens: Token IDs corresponding to decoded_tokens_list.
                These are alternatives at the SAME position (e.g.
                [sampled, top1, top2]), NOT sequential tokens.
            context_token_ids: Preceding sampled token IDs providing
                sequential context. If None, extracted from
                self.logprobs.
        """
        """
        验证并修复包含替换字符的解码 token。

        遍历解码后的 token 列表，检查是否包含 Unicode 替换字符（U+FFFD）。
        如果包含，使用前序上下文 token 尝试修复。

        注意：tokens 参数中的 token 是同一位置的替代 token
        （例如 [采样token, top1, top2]），不是序列中的连续 token。

        参数:
            decoded_tokens_list: 已解码的 token 字符串列表
            tokens: 对应的 token ID 列表（同一位置的替代 token）
            context_token_ids: 前序采样 token ID 列表，提供序列上下文。
                如果为 None，从 self.logprobs 中提取。

        返回:
            修复后的解码 token 字符串列表
        """
        if context_token_ids is None:
            context_token_ids = self._get_sampled_context_ids(self.logprobs)

        corrected_decoded_token_map = dict()
        for idx, text in enumerate(decoded_tokens_list):
            if text.endswith("�"):
                # Replacement char at the end means a potential
                # unfinished byte sequence from byte-fallback
                # tokenization. Correct each token independently
                # using only the sequential context.
                # 末尾的替换字符意味着可能是字节回退分词器产生的未完成字节序列。
                # 使用序列上下文独立修复每个 token。
                corrected_decoded_token_map[idx] = self._correct_decoded_token(
                    tokens[idx], context_token_ids
                )

        # 将修复后的结果写回列表
        for idx, text in corrected_decoded_token_map.items():
            decoded_tokens_list[idx] = text

        return decoded_tokens_list

    def update_from_output(self, output: EngineCoreOutput) -> None:
        """
        从 EngineCoreOutput 更新对数概率处理器状态。

        这是外部调用的主入口方法，根据输出中包含的数据类型
        分别处理采样对数概率和提示对数概率。

        参数:
            output: EngineCore 的输出对象，可能包含：
                - new_logprobs: 新的采样对数概率
                - new_prompt_logprobs_tensors: 新的提示对数概率张量
        """
        if output.new_logprobs is not None:
            self._update_sample_logprobs(output.new_logprobs)
        if output.new_prompt_logprobs_tensors is not None:
            self._update_prompt_logprobs(output.new_prompt_logprobs_tensors)
