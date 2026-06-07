# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2025-present the Outlines developers
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# =============================================================================
# Outlines 结构化输出后端 (Outlines Backend)
# =============================================================================
# Outlines（基于 outlines_core 库）是 vLLM 支持的结构化输出后端之一。
# outlines_core 是一个基于正则表达式自动机（regex-automata）的高效引导解码库。
#
# 本模块实现了基于 Outlines 的结构化输出后端，支持以下约束类型：
# 1. JSON Schema：通过 JSON 模式构建正则表达式来约束输出
# 2. 正则表达式（REGEX）：直接使用正则表达式约束输出
# 3. 选项列表（CHOICE）：将选项转换为正则表达式 alternation
#
# 不支持的约束类型：
# - EBNF 语法（GRAMMAR）
# - JSON Object（通用 JSON 对象）
# - 结构化标签（STRUCTURAL_TAG）
#
# 核心组件：
# - OutlinesBackend：引擎级别的后端，负责编译正则表达式和分配位掩码
# - OutlinesGrammar：请求级别的语法对象，通过 Guide 管理 FSM 状态
#
# 工作原理：
# 1. 将 JSON Schema 转换为正则表达式（通过 outlines_core.json_schema）
# 2. 使用正则表达式构建 Index（编译后的自动机）
# 3. 通过 Guide 在运行时跟踪 FSM 状态并生成位掩码
# 4. 支持缓存机制以避免重复编译相同的正则表达式
#
# 特殊注意事项：
# - 正则表达式不支持反向引用、look-around 断言、Unicode 词边界等高级特性
# - 正则表达式必须具有锚定的通用起始状态（universal start state）
# - DFA 接受信号会延迟一步，以确保 EOS token 可以被正确发出
# =============================================================================

from __future__ import annotations

import ast
import importlib
import json
import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from regex import escape as regex_escape

from vllm.sampling_params import SamplingParams
from vllm.utils.import_utils import LazyLoader
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.v1.structured_output.backend_types import (
    StructuredOutputBackend,
    StructuredOutputGrammar,
    StructuredOutputOptions,
)
from vllm.v1.structured_output.utils import (
    OutlinesVocabulary,
    get_outlines_cache,
    get_outlines_vocabulary,
)

if TYPE_CHECKING:
    import outlines_core as oc
    import outlines_core.json_schema as json_schema
else:
    oc = LazyLoader("oc", globals(), "outlines_core")
    json_schema = LazyLoader("json_schema", globals(), "outlines_core.json_schema")

# Python 3.11+ sre_parse and sre_constants
# are deprecated, so we must import them from re
if sys.version_info >= (3, 11):
    # Hack to get around pre-commit regex module rule
    # because going through re is the only way to get sre_parse
    # and sre_constants in Python 3.11+
    _re = importlib.import_module("re")
    sre_parse = _re._parser
    sre_constants = _re._constants
else:
    import sre_constants
    import sre_parse


@dataclass
class OutlinesBackend(StructuredOutputBackend):
    """Outlines 引擎级别后端。

    # 负责：
    # 1. 初始化 Outlines 词汇表和缓存
    # 2. 编译正则表达式为 Index（自动机）
    # 3. 创建 Guide 对象用于运行时 FSM 跟踪
    # 4. 分配位掩码内存
    #
    # 缓存机制：
    # - 使用 vocabulary 哈希 + 正则表达式字符串作为缓存键
    # - 避免相同正则表达式的重复编译
    """

    def __post_init__(self):
        # 获取 Outlines 词汇表对象（包含字节到 token ID 的映射）
        self.vocabulary = get_outlines_vocabulary(self.tokenizer)
        # 获取缓存实例（内存 LRU 或磁盘缓存）
        self.cache = get_outlines_cache()

    def _compile_index(
        self, regex_string: str, vocabulary: OutlinesVocabulary
    ) -> oc.Index:
        """编译正则表达式为 Index（自动机）并缓存结果。

        # 处理流程：
        # 1. 使用词汇表哈希 + 正则表达式作为缓存键
        # 2. 如果缓存命中，直接返回缓存的 Index
        # 3. 如果缓存未命中，编译正则表达式为 Index
        # 4. 将编译结果存入缓存
        #
        # Index 是 outlines_core 的核心数据结构，它将正则表达式
        # 编译为确定性有限自动机（DFA），用于高效的 token 级匹配。
        """
        cache_key = f"{vocabulary._hash}_{regex_string}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        index = oc.Index(regex_string, vocabulary.inner)
        self.cache[cache_key] = index

        return index

    def compile_grammar(
        self, request_type: StructuredOutputOptions, grammar_spec: str
    ) -> StructuredOutputGrammar:
        """编译语法规范为 OutlinesGrammar 对象。

        # 处理流程：
        # 1. 根据请求类型将语法规范转换为正则表达式：
        #    - JSON: 通过 outlines_core.json_schema 构建正则表达式
        #    - REGEX: 直接使用提供的正则表达式
        #    - CHOICE: 将选项列表转换为正则表达式 alternation
        # 2. 编译正则表达式为 Index（自动机）
        # 3. 创建 Guide 对象用于运行时 FSM 跟踪
        # 4. 配置 max_rollback_tokens 以支持投机解码
        """
        if request_type == StructuredOutputOptions.JSON:
            # 将 JSON Schema 转换为等价的正则表达式
            regex = json_schema.build_regex_from_schema(grammar_spec)
        elif request_type == StructuredOutputOptions.REGEX:
            regex = grammar_spec
        elif request_type == StructuredOutputOptions.CHOICE:
            # 将选项列表转换为正则表达式：(choice1|choice2|...)
            choices = ast.literal_eval(grammar_spec)
            choices = [regex_escape(c) for c in choices]
            regex = "(" + "|".join(choices) + ")"
        else:
            raise ValueError(
                f"Invalid request type for Outlines backend ({request_type!s})"
            )
        # 编译正则表达式为 Index（DFA 自动机）
        index = self._compile_index(regex, self.vocabulary)
        # 获取投机解码的回滚 token 数
        max_rollback_tokens = (
            self.vllm_config.speculative_config.num_speculative_tokens
            if self.vllm_config.speculative_config is not None
            else 0
        )
        # 创建 Guide 对象，用于运行时跟踪 FSM 状态
        return OutlinesGrammar(
            vocab_size=self.vocab_size,
            guide=oc.Guide(index, max_rollback=max_rollback_tokens),
        )

    def allocate_token_bitmask(self, max_num_seqs: int) -> torch.Tensor:
        """分配 token 位掩码张量。

        # 位掩码形状为 (max_num_seqs, ceil(vocab_size / 32))，
        # 每个 token 用一个 bit 表示（int32 的 32 个 bit 可表示 32 个 token）。
        # 初始值为 -1（全 1），表示默认允许所有 token。
        # 使用 pin_memory 优化 CPU 到 GPU 的异步传输。
        """
        return torch.full(
            (max_num_seqs, (self.vocab_size + 31) // 32),
            -1,
            dtype=torch.int32,
            pin_memory=is_pin_memory_available(),
        )

    def destroy(self):
        """Outlines 后端无需特殊清理。"""
        pass


@dataclass
class OutlinesGrammar(StructuredOutputGrammar):
    """Outlines 请求级别语法对象。

    # 每个需要结构化输出的请求创建一个 OutlinesGrammar 实例。
    # 它封装了 outlines_core 的 Guide 对象，用于：
    # 1. 跟踪请求的 DFA 状态
    # 2. 接受或拒绝 token（推进或拒绝 FSM）
    # 3. 生成下一步允许的 token 位掩码
    # 4. 支持回滚操作（用于投机解码）
    #
    # 特殊设计：
    # - DFA 接受信号延迟一步：outlines_core 在 DFA 接受时就报告完成，
    #   但 vLLM 需要在 EOS token 发出后才标记终止。
    #   因此使用 _prev_finished 延迟一步报告终止状态。
    """

    vocab_size: int
    # Guide 实例，用于运行时 DFA 状态跟踪
    guide: oc.Guide = field(hash=False)
    # 已处理的 token 数量计数器
    num_processed_tokens: int = field(
        default_factory=lambda: 0, repr=False, hash=False, init=False
    )

    # outlines_core signals done on DFA accept; vLLM expects done after EOS.
    # We delay the finished flag by one step so EOS can still be emitted.
    # DFA 接受信号延迟一步：记录上一步的完成状态，
    # 确保 EOS token 可以在终止信号之前被正确发出。
    _prev_finished: bool = field(default=False, init=False, repr=False, hash=False)

    def accept_tokens(self, request_id: str, tokens: list[int]) -> bool:
        """接受 token 列表并推进 FSM。

        # 处理逻辑：
        # 1. 先调用 accepts_tokens 检查所有 token 是否能被接受
        # 2. 如果能接受，逐个推进 FSM 状态
        # 3. 注意：accepts_tokens 只检查当前 token 是否可接受，
        #    而 advance 还会检查推进后的下一个状态是否是死状态。
        #    如果下一个状态是死状态，advance 会失败。
        # 4. 更新已处理 token 计数

        Returns True if the FSM was advanced successfully.
        Returns False if the FSM failed to advance.
        """
        if self.guide.accepts_tokens(tokens):
            # Advance can fail when the next state reached after advancing with
            # the current tokens is a dead state. This is because Guide.accepts_tokens()
            # only checks whether the current tokens can be accepted,
            # whereas guide.advance() additionally checks the next state
            # after all tokens are accepted.
            # We need to be aware that the FSM must be prepared without dead states.
            for t in tokens:
                self.guide.advance(t)
                self.num_processed_tokens += 1
            return True
        return False

    def rollback(self, num_tokens: int) -> None:
        """回滚 FSM 状态。

        # 回滚指定数量的 token，恢复 Guide 到之前的状态。
        # 同时更新已处理 token 计数器。
        # 用于投机解码中当草稿 token 被拒绝时。
        """
        self.guide.rollback_state(num_tokens)
        self.num_processed_tokens -= num_tokens

    def validate_tokens(self, tokens: list[int]) -> list[int]:
        """验证 token 列表是否被 FSM 按顺序接受，但不推进 FSM。

        # 逐个检查 token 是否能被当前状态接受：
        # 1. 将已验证的 token 追加到临时列表
        # 2. 调用 accepts_tokens 检查是否可接受
        # 3. 一旦遇到不被接受的 token，停止并返回已接受的前缀
        #
        # 主要用于投机解码中验证草稿 token 是否符合语法。
        """
        accepted: list[int] = []
        for tok in tokens:
            accepted.append(tok)
            if not self.guide.accepts_tokens(accepted):
                accepted.pop()
                break
        return accepted

    def fill_bitmask(self, bitmask: torch.Tensor, idx: int) -> None:
        """填充位掩码。

        # 调用 Guide 的 write_mask_into 方法，将当前步骤允许的 token
        # 写入到位掩码的指定索引位置。
        # write_mask_into 直接操作底层内存，效率较高。
        """
        mask = bitmask[idx]
        self.guide.write_mask_into(mask.data_ptr(), mask.numel(), mask.element_size())

    def is_terminated(self) -> bool:
        """检查语法是否已终止。

        # 特殊设计：延迟一步报告终止状态。
        # outlines_core 在 DFA 接受时就报告完成（is_finished()），
        # 但 vLLM 需要确保 EOS token 可以在终止信号之前被发出。
        # 因此返回上一步的完成状态（_prev_finished），而非当前状态。
        #
        # 这样做的好处：
        # - 当 DFA 接受时，is_finished() 返回 True
        # - 但此时返回 False（因为 _prev_finished 还是 False）
        # - 下一次调用时，返回 True（此时 EOS 已经可以发出）
        """
        curr = self.guide.is_finished()
        prev = self._prev_finished
        self._prev_finished = curr
        return prev

    def reset(self):
        """重置语法状态到初始状态。"""
        self.num_processed_tokens = 0
        self._prev_finished = False
        self.guide.reset()


def validate_structured_output_request_outlines(params: SamplingParams):
    """验证 Outlines 后端的结构化输出请求。

    # 验证流程：
    # 1. 正则表达式：验证正则表达式是否可构建为自动机
    # 2. JSON 模式：将 JSON Schema 转换为正则表达式后验证
    # 3. 选项列表：将选项转换为正则表达式后验证
    # 4. EBNF 语法：不支持，抛出 ValueError
    #
    # 验证的目的是确保正则表达式不包含 outlines_core 不支持的高级特性
    # （如反向引用、look-around 断言等）。
    """
    if params.structured_outputs is None:
        return

    so_params = params.structured_outputs

    if so_params.regex:
        validate_regex_is_buildable(so_params.regex)
    elif so_params.json:
        if isinstance(so_params.json, str):
            try:
                # make sure schema is valid json
                json.loads(so_params.json)
                schema = so_params.json
            except json.JSONDecodeError as e:
                raise ValueError("Invalid JSON grammar specification.") from e
        else:
            try:
                schema = json.dumps(so_params.json)
            except Exception as e:
                raise ValueError(
                    f"Error serializing structured outputs jsonschema: {e}"
                ) from e
        # 将 JSON Schema 转换为正则表达式后验证
        pattern = json_schema.build_regex_from_schema(schema)
        validate_regex_is_buildable(pattern)
    elif so_params.choice:
        # 将选项列表转换为正则表达式 alternation
        choices = [regex_escape(str(choice)) for choice in so_params.choice]
        regex = "(" + "|".join(choices) + ")"
        validate_regex_is_buildable(regex)
    elif so_params.grammar:
        raise ValueError(
            "Outlines structured outputs backend "
            "does not support grammar specifications"
        )


def _prefix_needs_context(parsed) -> bool:
    """检查正则表达式的前缀是否需要上下文（look-around/anchor）。

    # 判断正则表达式在匹配任何字符之前是否包含需要上下文的元素：
    # - look-around 断言（(?=...), (?!...) 等）
    # - 锚点（^, $, \\b 等）
    #
    # 如果前缀需要上下文，则该正则表达式不具有"锚定的通用起始状态"，
    # outlines_core 的自动机无法正确处理。
    #
    # Returns:
    #     True 如果前缀包含 look-around 或锚点，False 否则
    """

    def subpattern_consumes(parsed) -> bool:
        """检查子模式是否能消耗至少一个字符。

        # 判断逻辑：
        # - 字面量、字符类、通配符（.）总是消耗字符
        # - 量词（*, +, ?）如果最大重复次数不为 0 且内部模式消耗字符
        # - 分支（|）如果任何一个分支消耗字符
        # - 分组递归检查内部模式
        """
        tokens = parsed.data if hasattr(parsed, "data") else parsed
        for ttype, tval in tokens:
            # literal, character class, or dot always consumes
            if ttype in (sre_parse.LITERAL, sre_parse.IN, sre_parse.ANY):
                return True
            # quantified subpattern: check inner pattern
            elif ttype == sre_parse.MAX_REPEAT:
                _, mx, sub = tval
                if mx != 0 and subpattern_consumes(sub):
                    return True
            # alternation: if any branch consumes, the whole does
            elif ttype == sre_parse.BRANCH:
                _, branches = tval
                if any(subpattern_consumes(br) for br in branches):
                    return True
            # grouped subpattern: recurse into its contents
            elif ttype == sre_parse.SUBPATTERN and subpattern_consumes(tval[3]):
                return True
        # No consumers, return False
        return False

    tokens = parsed.data if hasattr(parsed, "data") else parsed
    for ttype, tval in tokens:
        # Direct anchors or look-around
        if ttype == sre_parse.AT or ttype in (
            sre_constants.ASSERT,
            sre_constants.ASSERT_NOT,
        ):
            return True

        # Nested subpattern: check
        if ttype == sre_parse.SUBPATTERN:
            # tval: (group, add_flags, del_flags, subpattern)
            if _prefix_needs_context(tval[3]):
                return True
            if subpattern_consumes(tval[3]):
                return False

        # if any branch has a prefix anchor => True,
        # else if at least one branch consumes => prefix ends => False
        elif ttype == sre_parse.BRANCH:
            saw_consumer = False
            for br in tval[1]:
                if _prefix_needs_context(br):
                    return True
                if subpattern_consumes(br):
                    saw_consumer = True
            if saw_consumer:
                return False

        # Immediate consumer tokens
        elif ttype in (sre_parse.LITERAL, sre_parse.IN, sre_parse.ANY):
            return False

        # if subpattern has anchor => True, if it can consume => stop
        elif ttype == sre_parse.MAX_REPEAT:
            if _prefix_needs_context(tval[2]):
                return True
            if subpattern_consumes(tval[2]):
                return False

    return False


def _check_unsupported(parsed) -> None:
    """检查正则表达式是否包含 regex-automata 不支持的特性。

    # 不支持的特性包括：
    # 1. 反向引用（backreference）：如 \\1, \\2 等引用捕获组
    # 2. look-around 断言：如 (?=...), (?!...), (?<=...), (?<!...)
    # 3. Unicode 词边界：如 \\b, \\B
    #
    # 这些特性需要更复杂的自动机实现，regex-automata 不支持。
    # 如果检测到这些特性，抛出 ValueError。
    """
    tokens = parsed.data if hasattr(parsed, "data") else parsed
    for ttype, tval in tokens:
        # backreference
        if ttype in (sre_parse.GROUPREF, sre_parse.GROUPREF_EXISTS):
            raise ValueError("Backreferences are unsupported.")

        # look-around assertion
        elif ttype in (sre_constants.ASSERT, sre_constants.ASSERT_NOT):
            raise ValueError("Look-Around assertion are unsupported.")

        # unicode word boundaries
        elif ttype == sre_parse.AT:
            if tval in (sre_constants.AT_BOUNDARY, sre_constants.AT_NON_BOUNDARY):
                raise ValueError("Unicode word boundaries are unsupported.")

        elif ttype == sre_parse.BRANCH:
            # tval is (None, branches)
            for branch in tval[1]:
                _check_unsupported(branch)

        # tval is (min, max, subpattern)
        elif ttype == sre_parse.MAX_REPEAT:
            _check_unsupported(tval[2])


def validate_regex_is_buildable(pattern: str) -> None:
    """验证正则表达式是否可以被 outlines_core 的 regex-automata 引擎构建。

    # 验证流程：
    # 1. 使用 Python 的 sre_parse 解析正则表达式
    # 2. 调用 _check_unsupported 检查不支持的特性
    # 3. 调用 _prefix_needs_context 检查是否具有锚定的通用起始状态
    #
    # "通用起始状态"（universal start state）是指自动机可以在不需要
    # 任何上下文的情况下开始匹配。这意味着正则表达式不能以锚点
    # （如 ^）或 look-around 断言开头。
    #
    # 参考：https://docs.rs/regex-automata/latest/regex_automata/dfa/trait.Automaton.html#method.universal_start_state
    """
    try:
        parsed = sre_parse.parse(pattern)

    except sre_constants.error as e:
        raise ValueError(f"Error parsing regex: {e}") from e

    try:
        _check_unsupported(parsed)
    except ValueError as e:
        raise ValueError(
            f"Regex uses unsupported feature for structured outputs: {e}. "
            "Only basic matching constructs are supported—lookarounds, "
            "backreferences, and unicode boundaries are not."
        ) from e

    if _prefix_needs_context(parsed):
        raise ValueError(
            "Regex does not have a anchored universal start state"
            "This means that the Regex uses anchors (^) or look-arounds "
            "in a way which requires context before any token is matched."
            "structured outputs needs regexes that can match without needing "
            "that context. Try rewriting the pattern without using these "
            f"constructs. Pattern:\n{pattern}"
        )
