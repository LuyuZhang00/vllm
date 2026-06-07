# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
解码器（Detokenizer）模块。

本模块实现了增量式反标记化（incremental detokenization）逻辑，将模型输出的
token ID 序列逐步转换为可读的文本字符串。

主要包含以下组件：
1. IncrementalDetokenizer - 基类，定义反标记化的基本接口
2. BaseIncrementalDetokenizer - 抽象基类，实现共享的反标记化逻辑
   （包括停止字符串检测、缓冲区管理等）
3. FastIncrementalDetokenizer - 基于 tokenizers 库的快速解码器
   （使用 Rust 实现的 DecodeStream，性能最优）
4. SlowIncrementalDetokenizer - 基于 Python 的慢速解码器
   （兼容不支持快速解码的 tokenizer）

工作流程：
1. 每当 EngineCore 产生新的 token ID 时，调用 update() 方法
2. update() 将新 token 反标记化为文本，并检查是否匹配停止字符串
3. 外部通过 get_next_output_text() 获取已解码的文本输出
   （支持增量模式和全量模式）
"""
from abc import ABC, abstractmethod

import tokenizers
import tokenizers.decoders
from packaging import version
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast

from vllm.logger import init_logger
from vllm.tokenizers import TokenizerLike
from vllm.tokenizers.detokenizer_utils import (
    convert_prompt_ids_to_tokens,
    detokenize_incrementally,
)
from vllm.utils import length_from_prompt_token_ids_or_embeds
from vllm.v1.engine import EngineCoreRequest

logger = init_logger(__name__)

# Only tokenizers >= 0.22.0 supports DecodeStream with native prefill
# (ids parameter) used for FastIncrementalDetokenizer.
# 只有 tokenizers >= 0.22.0 才支持带有原生预填充（ids 参数）的 DecodeStream，
# 这是 FastIncrementalDetokenizer 所需的功能。
USE_FAST_DETOKENIZER = version.parse(tokenizers.__version__) >= version.parse("0.22.0")

# Error string from https://github.com/huggingface/tokenizers/blob/909fdde2a4ffedd9295206f705eb612be2a91b12/tokenizers/src/tokenizer/mod.rs#L1042
# 当 tokenizers 库内部遇到无效前缀时抛出的错误消息前缀，
# 用于捕获并恢复 DecodeStream 的状态损坏问题。
INVALID_PREFIX_ERR_MSG = "Invalid prefix encountered"


class IncrementalDetokenizer:
    """
    增量式反标记化器的基类。

    提供最基本的接口定义，包括：
    - 维护输出 token ID 列表
    - 提供 token 数量查询
    - update() 方法：接收新 token 并返回反标记化结果
    - get_next_output_text() 方法：获取已解码的文本

    当没有 tokenizer 可用时（反标记化被禁用），直接使用此类，
    所有方法返回空值或默认值。
    """

    def __init__(self):
        # 存储所有已生成的 token ID（对于子类可能包含 prompt token ID）
        self.token_ids: list[int] = []

    @property
    def output_token_ids(self) -> list[int]:
        """返回输出 token ID 列表。"""
        return self.token_ids

    def num_output_tokens(self) -> int:
        """返回已生成的输出 token 数量。"""
        return len(self.token_ids)

    def update(self, new_token_ids: list[int], stop_terminated: bool) -> str | None:
        """
        更新反标记化状态。

        参数:
            new_token_ids: 新生成的 token ID 列表
            stop_terminated: 是否因匹配停止字符串而终止

        返回:
            基类实现中始终返回 None（不做任何反标记化）。
            子类会返回匹配的停止字符串或 None。
        """
        self.token_ids.extend(new_token_ids)
        return None

    def get_next_output_text(self, finished: bool, delta: bool) -> str:
        """
        获取已解码的输出文本。

        参数:
            finished: 请求是否已完成
            delta: 是否只返回自上次调用以来的增量文本

        返回:
            基类实现中始终返回空字符串。
            子类会返回实际的解码文本。
        """
        return ""

    @classmethod
    def from_new_request(
        cls,
        tokenizer: TokenizerLike | None,
        request: EngineCoreRequest,
    ) -> "IncrementalDetokenizer":
        """
        工厂方法：根据请求创建合适的反标记化器实例。

        选择策略：
        1. 如果没有 tokenizer，返回不进行反标记化的基类实例
        2. 如果有快速 tokenizer（tokenizers >= 0.22.0 且为
           PreTrainedTokenizerFast），使用 FastIncrementalDetokenizer
        3. 否则使用 SlowIncrementalDetokenizer

        参数:
            tokenizer: 分词器实例，None 表示禁用反标记化
            request: 引擎核心请求对象

        返回:
            创建的反标记化器实例
        """
        assert request.sampling_params is not None

        if tokenizer is None:
            # No tokenizer => skipping detokenization.
            # 没有 tokenizer，跳过反标记化
            return IncrementalDetokenizer()

        if USE_FAST_DETOKENIZER and isinstance(tokenizer, PreTrainedTokenizerFast):
            # Fast tokenizer => use tokenizers library DecodeStream.
            # 快速 tokenizer，使用 tokenizers 库的 DecodeStream（Rust 实现）
            return FastIncrementalDetokenizer(tokenizer, request)

        # Fall back to slow python-based incremental detokenization.
        # 回退到慢速的基于 Python 的增量反标记化
        return SlowIncrementalDetokenizer(tokenizer, request)


class BaseIncrementalDetokenizer(IncrementalDetokenizer, ABC):
    """
    增量式反标记化器的抽象基类。

    实现了所有具体反标记化器共享的核心逻辑：
    1. 停止字符串管理 - 配置和检测停止条件
    2. 最小 token 数量 - 支持 min_tokens 参数
    3. 停止字符串缓冲 - 在流式输出时暂存可能包含停止字符串的文本
    4. 增量文本输出 - 支持 delta 模式的文本输出

    子类需要实现 decode_next() 方法来完成具体的单 token 解码逻辑。
    """

    def __init__(self, request: EngineCoreRequest):
        super().__init__()

        # Stop strings
        # 解析并规范化停止字符串列表
        params = request.sampling_params
        assert params is not None
        if params.stop is None:
            self.stop = []
        elif isinstance(params.stop, str):
            self.stop = [params.stop]
        else:
            self.stop = params.stop
        # 最小生成 token 数量，在此之前不检测停止字符串
        self.min_tokens = params.min_tokens
        # 是否在输出中包含停止字符串本身
        self.include_stop_str_in_output = params.include_stop_str_in_output

        # Number of chars to hold back when stop strings are to be excluded
        # from streamed output.
        # 当需要从流式输出中排除停止字符串时，需要暂存的字符数。
        # 等于最长停止字符串长度减 1，确保不会提前输出部分匹配的停止字符串。
        if self.stop and not self.include_stop_str_in_output:
            self.stop_buffer_length = max(len(s) for s in self.stop) - 1
        else:
            self.stop_buffer_length = 0
        # 记录上次输出文本的位置，用于增量输出
        self._last_output_text_offset: int = 0

        # Generation data
        # 累积的完整输出文本
        self.output_text = ""

    def update(self, new_token_ids: list[int], stop_terminated: bool) -> str | None:
        """
        更新请求状态，执行以下步骤：
            1) 将新 token ID 增量反标记化为文本
            2) 检查是否匹配停止条件

        参数:
            new_token_ids: 新生成的 token ID 列表
            stop_terminated: 是否因停止字符串匹配而终止

        返回:
            匹配到的停止字符串，如果没有匹配则返回 None
        """
        if not new_token_ids:
            # Skip detokenization if no new token ids.
            # 没有新 token，跳过反标记化
            return None

        if stop_terminated and not self.include_stop_str_in_output:
            # If stop-terminated, exclude last token from detokenization
            # based on include_stop_str_in_output parameter.
            # 如果因停止字符串终止且不需要在输出中包含停止字符串，
            # 则跳过最后一个 token 的反标记化（该 token 通常是停止字符串的一部分）
            skipped_stop_token_id = new_token_ids[-1]
            new_token_ids = new_token_ids[:-1]
        else:
            skipped_stop_token_id = None

        # 1) Detokenize the new token ids incrementally.
        # 第一步：增量反标记化新 token ID
        stop_check_offset = len(self.output_text)
        for new_token_id in new_token_ids:
            self.token_ids.append(new_token_id)
            self.output_text += self.decode_next(new_token_id)
            # Support min_tokens, see https://github.com/vllm-project/vllm/pull/22014
            # 支持 min_tokens：在达到最小 token 数之前，更新停止检查偏移量，
            # 这样就不会检测到过早出现的停止字符串
            if self.min_tokens and self.num_output_tokens() <= self.min_tokens:
                stop_check_offset = len(self.output_text)

        if skipped_stop_token_id is not None:
            # Cleanup after skipping detokenization.
            # 跳过反标记化的 token 仍然需要记录到 token_ids 中
            self.token_ids.append(skipped_stop_token_id)

        # 2) Evaluate stop strings.
        # 第二步：检测停止字符串
        stop_string = None
        if self.stop and self.num_output_tokens() > self.min_tokens:
            stop = check_stop_strings(
                output_text=self.output_text,
                new_char_count=len(self.output_text) - stop_check_offset,
                stop=self.stop,
                include_in_output=self.include_stop_str_in_output,
            )
            if stop is not None:
                stop_string, truncate_to = stop
                if truncate_to != -1:
                    # 如果需要截断（不包含停止字符串在输出中），
                    # 将输出文本截断到停止字符串的起始位置
                    self.output_text = self.output_text[:truncate_to]

        return stop_string

    @abstractmethod
    def decode_next(self, next_token_id: int) -> str:
        """
        将单个 token ID 解码为文本字符串。

        这是抽象方法，由子类实现具体的解码逻辑。

        参数:
            next_token_id: 要解码的 token ID

        返回:
            解码后的文本字符串
        """
        raise NotImplementedError

    def get_next_output_text(self, finished: bool, delta: bool) -> str:
        """If delta is True, only new text since the last call to
        this method is returned"""
        """
        获取已解码的输出文本。

        参数:
            finished: 请求是否已完成。如果完成则返回全部文本，
                     否则可能暂存末尾部分以避免输出部分停止字符串。
            delta: 如果为 True，只返回自上次调用以来新增的文本。

        返回:
            解码后的文本字符串。
        """
        # We return the full output text if the sequence is finished.
        # 如果序列已完成，不需要缓冲；否则根据 stop_buffer_length 暂存末尾文本
        buffer_length = 0 if finished else self.stop_buffer_length
        if not delta:
            # 全量模式：返回从开头到（减去缓冲区长度）的文本
            if not buffer_length:
                return self.output_text
            return self.output_text[:-buffer_length]

        # 增量模式：只返回自上次调用以来新增的文本
        length = len(self.output_text) - buffer_length
        last_offset = self._last_output_text_offset
        if last_offset < length:
            self._last_output_text_offset = length
            return self.output_text[last_offset:length]
        return ""


class FastIncrementalDetokenizer(BaseIncrementalDetokenizer):
    """
    基于 tokenizers 库的快速增量反标记化器。

    使用 HuggingFace tokenizers 库提供的 Rust 实现的 DecodeStream，
    性能显著优于纯 Python 实现。

    主要特点：
    1. 使用 DecodeStream 进行增量解码，避免重复解码整个序列
    2. 支持原生预填充（native prefill），用 prompt token 初始化解码流
    3. 处理特殊 token 之间的空格问题
    4. 内置错误恢复机制（处理无效前缀和溢出错误）
    """

    def __init__(self, tokenizer: PreTrainedTokenizerFast, request: EngineCoreRequest):
        super().__init__(request)

        sampling_params = request.sampling_params
        assert sampling_params is not None

        self.request_id = request.request_id
        # 是否跳过特殊 token（如 [CLS], [SEP] 等）
        self.skip_special_tokens = sampling_params.skip_special_tokens

        # 获取底层的 tokenizers.Tokenizer 对象（Rust 实现）
        self.tokenizer: Tokenizer = tokenizer._tokenizer

        # Use native prefill to prime the decode stream with prompt tokens.
        # Look up DecodeStream on the module so backend patches (e.g. the
        # fastokens shim that replaces ``tokenizers.decoders.DecodeStream``)
        # are honored regardless of import order.
        # 使用原生预填充功能，将 prompt token IDs 传入 DecodeStream
        # 这样可以正确初始化解码器状态，确保后续生成的 token 能被正确解码
        self.stream = tokenizers.decoders.DecodeStream(
            ids=request.prompt_token_ids,
            skip_special_tokens=self.skip_special_tokens,
        )

        # 是否在特殊 token 之间添加空格
        self.spaces_between_special_tokens = (
            sampling_params.skip_special_tokens
            or sampling_params.spaces_between_special_tokens
        )

        if not self.spaces_between_special_tokens:
            # Store dict of added token ids so that we can suppress
            # the spaces between them.
            # 存储添加的特殊 token ID 字典，用于在连续特殊 token 之间抑制空格
            added_token_ids = getattr(self.tokenizer, "added_token_ids", None)
            if added_token_ids is None:
                self.tokenizer.added_token_ids = added_token_ids = {
                    tid: tok.content
                    for tid, tok in self.tokenizer.get_added_tokens_decoder().items()
                }

            if added_token_ids:
                self.last_special = False
                self.added_token_ids = added_token_ids
            else:
                # No added tokens.
                # 没有添加的特殊 token，启用空格模式
                self.spaces_between_special_tokens = True

    def decode_next(self, next_token_id: int) -> str:
        """
        将单个 token ID 解码为文本。

        使用 DecodeStream 的 step 方法进行增量解码，
        并处理特殊 token 之间的空格问题。

        参数:
            next_token_id: 要解码的 token ID

        返回:
            解码后的文本字符串
        """
        # 调用受保护的 step 方法，处理可能的异常
        token = self._protected_step(next_token_id)

        if not self.spaces_between_special_tokens:
            # 检查当前 token 是否为特殊 token
            special_token = self.added_token_ids.get(next_token_id)
            is_special = special_token is not None
            if is_special and self.last_special:
                # Return raw token string without any prefixed spaces.
                # 连续两个特殊 token 时，直接使用原始 token 字符串，
                # 去除 DecodeStream 可能添加的前缀空格
                token = special_token
            self.last_special = is_special

        return token or ""

    def _protected_step(self, next_token_id: int) -> str | None:
        """
        受保护的 DecodeStream.step() 调用，包含异常处理和恢复逻辑。

        处理以下异常情况：
        1. OverflowError/TypeError - 罕见的 token ID 溢出问题
        2. "Invalid prefix" 错误 - tokenizer 产生非单调、无效 UTF-8 输出
           时会破坏 DecodeStream 的内部状态，需要重置流

        参数:
            next_token_id: 要解码的 token ID

        返回:
            解码后的文本，或 None（解码失败时）
        """
        try:
            token = self.stream.step(self.tokenizer, next_token_id)
        except (OverflowError, TypeError):
            # Handle rare observed overflow, still to be diagnosed.
            # See https://github.com/vllm-project/vllm/issues/21951.
            # 处理罕见的溢出错误（原因尚待诊断）
            logger.exception("Encountered invalid token id: %r", next_token_id)
            token = None
        except Exception as e:
            if not str(e).startswith(INVALID_PREFIX_ERR_MSG):
                raise e
            # Recover from edge case where tokenizer can produce non-monotonic,
            # invalid UTF-8 output, which breaks the internal state of
            # tokenizers' DecodeStream.
            # See https://github.com/vllm-project/vllm/issues/17448.
            # 恢复 tokenizer 产生非单调、无效 UTF-8 输出的边界情况，
            # 这种情况会破坏 DecodeStream 的内部状态，需要重置流
            logger.warning(
                "Encountered invalid prefix detokenization error"
                " for request %s, resetting decode stream.",
                self.request_id,
            )
            # 重置 DecodeStream（不传入 prompt IDs，从当前 token 重新开始）
            self.stream = tokenizers.decoders.DecodeStream(
                skip_special_tokens=self.skip_special_tokens
            )
            token = self.stream.step(self.tokenizer, next_token_id)
        return token


class SlowIncrementalDetokenizer(BaseIncrementalDetokenizer):
    """
    基于 Python 的慢速增量反标记化器。

    当 tokenizers 库版本过低或 tokenizer 不是 PreTrainedTokenizerFast 时使用。
    使用 vllm 自带的 detokenize_incrementally() 函数实现增量解码。

    主要特点：
    1. 使用 prefix_offset 和 read_offset 追踪解码位置
    2. 维护完整的 token 列表用于增量解码
    3. 需要单独管理 prompt token ID（不支持原生预填充）
    4. 兼容所有类型的 tokenizer
    """

    def __init__(self, tokenizer: TokenizerLike, request: EngineCoreRequest):
        super().__init__(request)

        self.tokenizer = tokenizer
        params = request.sampling_params
        assert params is not None

        # 计算 prompt 长度（支持 prompt_embeds 的情况）
        self.prompt_len = length_from_prompt_token_ids_or_embeds(
            request.prompt_token_ids, request.prompt_embeds
        )

        # Metadata for incremental detokenization.
        # 增量反标记化的元数据：tokens 存储已解码的 token 列表，
        # prefix_offset 和 read_offset 用于追踪增量解码位置
        if request.prompt_token_ids is not None:
            self.tokens, self.prefix_offset, self.read_offset = (
                convert_prompt_ids_to_tokens(
                    tokenizer=tokenizer,
                    prompt_ids=request.prompt_token_ids,
                    skip_special_tokens=params.skip_special_tokens,
                )
            )
        else:
            # Prompt embedding requests cannot be detokenized, in general.
            # 使用 prompt embeddings 的请求一般无法反标记化
            self.tokens = [""] * self.prompt_len
            self.prefix_offset = 0
            self.read_offset = 0

        # 将 prompt token ID 添加到 token_ids 列表中，
        # 用于 detokenize_incrementally() 函数
        self.token_ids.extend(request.prompt_token_ids or [0] * self.prompt_len)

        self.skip_special_tokens = params.skip_special_tokens
        self.spaces_between_special_tokens = params.spaces_between_special_tokens

    @property
    def output_token_ids(self) -> list[int]:
        """
        返回输出 token ID 列表（不包含 prompt token ID）。

        对于 SlowIncrementalDetokenizer，token_ids 包含了 prompt token，
        因此需要跳过前面的 prompt 部分。
        """
        if self.prompt_len:
            return self.token_ids[self.prompt_len :]
        return self.token_ids

    def num_output_tokens(self) -> int:
        """
        返回已生成的输出 token 数量（不包含 prompt token）。
        """
        return len(self.token_ids) - self.prompt_len

    def decode_next(self, next_token_id: int) -> str:
        """
        将单个 token ID 增量解码为文本。

        使用 vllm 的 detokenize_incrementally() 函数，
        该函数会考虑前序 token 的影响（例如多字节 UTF-8 字符的处理）。

        参数:
            next_token_id: 要解码的 token ID

        返回:
            解码后的文本增量
        """
        new_tokens, decoded_text, prefix_offset, read_offset = detokenize_incrementally(
            tokenizer=self.tokenizer,
            all_input_ids=self.token_ids,
            prev_tokens=self.tokens,
            prefix_offset=self.prefix_offset,
            read_offset=self.read_offset,
            skip_special_tokens=self.skip_special_tokens,
            spaces_between_special_tokens=self.spaces_between_special_tokens,
        )

        # 更新内部状态
        self.tokens.extend(new_tokens)
        self.prefix_offset = prefix_offset
        self.read_offset = read_offset

        return decoded_text


def check_stop_strings(
    output_text: str,
    new_char_count: int,
    stop: list[str],
    include_in_output: bool,
) -> tuple[str, int] | None:
    """Check if any stop strings are matched and truncate sequence
    output text accordingly.

    Returns tuple (stop_string, offset) if matched or else None.

    Where stop_string is the matched stop string and offset is the
    length to which output_text should be truncated, or -1 for no
    truncation.
    """
    """
    检查输出文本中是否匹配到任何停止字符串，并相应地截断文本。

    参数:
        output_text: 完整的输出文本
        new_char_count: 新增的字符数量（用于避免重复搜索已搜索过的文本）
        stop: 停止字符串列表
        include_in_output: 是否在输出中包含停止字符串

    返回:
        如果匹配到停止字符串，返回 (停止字符串, 截断位置) 的元组：
        - 截断位置为 -1 表示不需要截断（停止字符串在最末尾）
        - 否则截断位置为文本应截断到的索引
        如果没有匹配到，返回 None
    """
    if not new_char_count or not stop:
        return None

    for stop_str in stop:
        stop_string_len = len(stop_str)
        # Avoid searching already-searched text.
        # 计算搜索起始位置，避免重复搜索已搜索过的文本。
        # 从 new_char_count + stop_string_len - 1 之前开始搜索，
        # 以确保能捕获跨越新旧文本边界的停止字符串。
        stop_index = output_text.find(stop_str, 1 - new_char_count - stop_string_len)
        if stop_index == -1:
            continue

        if include_in_output:
            # Truncate to end of stop string.
            # 需要在输出中包含停止字符串，截断位置设为停止字符串末尾
            stop_index += stop_string_len
            if stop_index >= len(output_text):
                # No truncation required.
                # 停止字符串已在文本末尾，不需要截断
                return stop_str, -1

        # Truncate the output text to either the beginning
        # or end of the stop string.
        # 截断输出文本到停止字符串的起始位置（不包含停止字符串）
        return stop_str, stop_index
    return None
