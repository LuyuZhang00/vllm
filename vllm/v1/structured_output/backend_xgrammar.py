# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# =============================================================================
# XGrammar 结构化输出后端 (XGrammar Backend)
# =============================================================================
# XGrammar 是一个高效的语法引导解码库，由 MLC AI 开发。
# 本模块实现了基于 XGrammar 的结构化输出后端，支持以下约束类型：
#
# 1. JSON Schema：通过 JSON 模式约束输出格式
# 2. JSON Object：约束输出为任意 JSON 对象
# 3. 正则表达式（REGEX）：通过正则表达式约束输出
# 4. EBNF 语法（GRAMMAR）：通过扩展巴科斯-瑙尔范式约束输出
# 5. 结构化标签（STRUCTURAL_TAG）：约束带标签的结构化输出
#
# 核心组件：
# - XgrammarBackend：引擎级别的后端，负责编译语法和分配位掩码
# - XgrammarGrammar：请求级别的语法对象，管理单个请求的 FSM 状态
#
# XGrammar 使用有限状态机（FSM）来跟踪语法状态，
# 并通过位掩码（bitmask）来指示每个步骤允许的 token。
# =============================================================================

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

import vllm.envs
from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams
from vllm.utils.import_utils import LazyLoader
from vllm.utils.mistral import is_mistral_tokenizer
from vllm.v1.structured_output.backend_types import (
    StructuredOutputBackend,
    StructuredOutputGrammar,
    StructuredOutputOptions,
)
from vllm.v1.structured_output.utils import (
    choice_as_grammar,
    convert_lark_to_ebnf,
    grammar_is_likely_lark,
)

if TYPE_CHECKING:
    import xgrammar as xgr
else:
    xgr = LazyLoader("xgr", globals(), "xgrammar")

logger = init_logger(__name__)


@dataclass
class XgrammarBackend(StructuredOutputBackend):
    """XGrammar 引擎级别后端。

    # 负责：
    # 1. 初始化 XGrammar 编译器和分词器信息
    # 2. 编译各种类型的语法规范
    # 3. 分配位掩码内存
    """

    def __post_init__(self):
        # 是否禁用任意空白字符（控制 JSON 中空白的灵活度）
        self.disable_any_whitespace = (
            self.vllm_config.structured_outputs_config.disable_any_whitespace
        )

        if is_mistral_tokenizer(self.tokenizer):
            # NOTE: ideally, xgrammar should handle this accordingly.
            # refer to https://github.com/mlc-ai/xgrammar/blob/d77c0a0173ef14779c918e3be7966ba852f7910f/python/xgrammar/tokenizer_info.py#L98
            #
            # Mistral 分词器需要特殊处理
            stop_token_ids = [self.tokenizer.eos_token_id]

            # not self.tokenizer.vocab_size as self.tokenizer.vocab
            # collapses all decoded errors into a single token.
            # 使用 len(vocab) 而非 vocab_size，因为后者会将所有解码错误合并为单个 token
            self.vocab_size = len(self.tokenizer.vocab)
            # 为 Mistral 分词器手动构建 TokenizerInfo
            tokenizer_info = xgr.TokenizerInfo(  # type: ignore
                encoded_vocab=self.tokenizer.vocab,
                # NOTE: https://github.com/mlc-ai/xgrammar/blob/5e141f6ff1ca02bc31f9e512e68b61f2a8ae88e5/tests/python/test_tokenizer_info.py#L43 # noqa: E501
                # 根据分词器类型选择词汇表类型
                vocab_type=xgr.VocabType.RAW
                if self.tokenizer.is_tekken
                else xgr.VocabType.BYTE_FALLBACK,
                vocab_size=self.vocab_size,
                stop_token_ids=stop_token_ids,
                add_prefix_space=True,
            )
        else:
            # 非 Mistral 分词器：从 HuggingFace 分词器自动获取 TokenizerInfo
            tokenizer_info = xgr.TokenizerInfo.from_huggingface(
                self.tokenizer,
                vocab_size=self.vocab_size,
            )
        # 初始化语法编译器
        # - max_threads=8：最多使用 8 个线程进行并行编译
        # - cache_enabled=True：启用编译缓存
        # - cache_limit_bytes：缓存大小限制（通过环境变量配置）
        self.compiler = xgr.GrammarCompiler(
            tokenizer_info,
            max_threads=8,
            cache_enabled=True,
            cache_limit_bytes=vllm.envs.VLLM_XGRAMMAR_CACHE_MB * 1024 * 1024,
        )

        # 获取投机解码的最大 token 数，用于配置 GrammarMatcher 的回滚能力
        self.num_speculative_tokens = 0
        if self.vllm_config.speculative_config is not None:
            self.num_speculative_tokens = (
                self.vllm_config.speculative_config.num_speculative_tokens
            )

    def compile_grammar(
        self, request_type: StructuredOutputOptions, grammar_spec: str
    ) -> StructuredOutputGrammar:
        """编译语法规范为 StructuredOutputGrammar 对象。

        # 根据请求类型选择不同的编译方式：
        # - JSON: 编译 JSON 模式
        # - JSON_OBJECT: 编译为通用 JSON 对象模式
        # - GRAMMAR: 编译 EBNF 语法
        # - REGEX: 编译正则表达式
        # - STRUCTURAL_TAG: 编译结构化标签
        """
        if request_type == StructuredOutputOptions.JSON:
            ctx = self.compiler.compile_json_schema(
                grammar_spec, any_whitespace=not self.disable_any_whitespace
            )
        elif request_type == StructuredOutputOptions.JSON_OBJECT:
            ctx = self.compiler.compile_json_schema(
                '{"type": "object"}', any_whitespace=not self.disable_any_whitespace
            )
        elif request_type == StructuredOutputOptions.GRAMMAR:
            ctx = self.compiler.compile_grammar(grammar_spec)
        elif request_type == StructuredOutputOptions.REGEX:
            ctx = self.compiler.compile_regex(grammar_spec)
        elif request_type == StructuredOutputOptions.STRUCTURAL_TAG:
            s_tag = json.loads(grammar_spec)
            if "structures" in s_tag:
                # Falling back to deprecated method of compiling structural tag
                # 使用已废弃的方法编译结构化标签（向后兼容）
                tags = [
                    xgr.StructuralTagItem(
                        begin=s["begin"],
                        schema=json.dumps(s["schema"]),
                        end=s["end"],
                    )
                    for s in s_tag["structures"]
                ]
                ctx = self.compiler.compile_structural_tag(tags, s_tag["triggers"])
            else:
                ctx = self.compiler.compile_structural_tag(grammar_spec)
        else:
            logger.error(
                "Validation should have already occurred. Please file an issue."
            )
            raise ValueError(
                f"grammar is not of valid supported types. ({request_type!s})"
            )

        # 创建 GrammarMatcher 用于在运行时匹配和推进语法状态
        # max_rollback_tokens 支持投机解码的回滚
        return XgrammarGrammar(
            matcher=xgr.GrammarMatcher(
                ctx,
                max_rollback_tokens=self.num_speculative_tokens,
            ),
            vocab_size=self.vocab_size,
            ctx=ctx,
        )

    def allocate_token_bitmask(self, max_num_seqs: int):
        """分配 token 位掩码张量。

        # 分配一个形状为 (max_num_seqs, vocab_size) 的位掩码，
        # 用于标记每个序列中每个 token 是否被语法允许。
        """
        return xgr.allocate_token_bitmask(max_num_seqs, self.vocab_size)

    def destroy(self):
        """销毁编译器，释放资源。"""
        del self.compiler


@dataclass
class XgrammarGrammar(StructuredOutputGrammar):
    """XGrammar 请求级别语法对象。

    # 每个需要结构化输出的请求都会创建一个 XgrammarGrammar 实例。
    # 它封装了 XGrammar 的 GrammarMatcher，用于：
    # 1. 跟踪请求的语法状态（FSM 状态）
    # 2. 接受或拒绝 token（推进或拒绝 FSM）
    # 3. 生成下一步允许的 token 位掩码
    # 4. 支持回滚操作（用于投机解码）
    #
    # NOTE: This would be a generic-enough class for
    # supporting different backends, in the future.
    # For now, just xgrammar.
    #
    # https://xgrammar.mlc.ai/docs/api/python/index.html#xgrammar.GrammarMatcher.find_jump_forward_string
    # for jump-forward decoding
    """

    vocab_size: int
    # GrammarMatcher 实例，用于运行时语法匹配
    matcher: xgr.GrammarMatcher = field(hash=False)
    # 编译后的语法上下文
    ctx: xgr.CompiledGrammar = field(hash=False)
    # 已处理的 token 数量计数器
    num_processed_tokens: int = field(
        default_factory=lambda: 0, repr=False, hash=False, init=False
    )
    # 语法是否已终止（例如匹配到 EOS 或完成 JSON 结构）
    _is_terminated: bool = field(default=False, repr=False, hash=False)

    def accept_tokens(self, request_id: str, tokens: list[int]) -> bool:
        """Accepts a list of tokens and advances the FSM.

        # 接受 token 列表并推进有限状态机（FSM）。
        #
        # 处理逻辑：
        # 1. 如果已终止，直接返回 False
        # 2. 逐个接受 token 并推进 FSM
        # 3. 如果任何 token 被拒绝，记录错误并返回 False
        # 4. 更新已处理 token 计数
        # 5. 检查 FSM 是否已终止

        Returns True if the FSM was advanced successfully.
        Returns False if the FSM failed to advance.
        """
        if self._is_terminated:
            return False
        for token in tokens:
            if not self.matcher.accept_token(token):
                logger.error(
                    "Failed to advance FSM for request %s "
                    "for tokens %s. Please file an issue.",
                    request_id,
                    token,
                )
                return False
            self.num_processed_tokens += 1
        self._is_terminated = self.matcher.is_terminated()
        return True

    def validate_tokens(self, tokens: list[int]) -> list[int]:
        """Checks if the list of tokens are accepted by the FSM in sequence.
        Will not advance the FSM.

        # 验证 token 列表是否被 FSM 按顺序接受，但不推进 FSM。
        # 主要用于投机解码中验证草稿 token 是否符合语法。
        #
        # 处理逻辑：
        # 1. 逐个尝试接受 token
        # 2. 一旦遇到不被接受的 token，停止
        # 3. 回滚 FSM 到初始状态（因为只是验证，不应改变状态）
        # 4. 返回被接受的 token 前缀列表

        Returns the prefix list of tokens that are accepted by the FSM.
        """
        accepted_tokens = []
        for token in tokens:
            if self.matcher.accept_token(token):
                accepted_tokens.append(token)
            else:
                break
        if len(accepted_tokens) > 0:
            # Rollback the FSM to the initial state
            # 回滚 FSM 到初始状态
            self.matcher.rollback(len(accepted_tokens))
        return accepted_tokens

    def rollback(self, num_tokens: int) -> None:
        """回滚 FSM 状态。

        # 回滚指定数量的 token，恢复 FSM 到之前的状态。
        # 同时更新已处理 token 计数器和终止状态。
        # 用于投机解码中当草稿 token 被拒绝时回滚状态。
        """
        self.matcher.rollback(num_tokens)
        self.num_processed_tokens -= num_tokens
        self._is_terminated = self.matcher.is_terminated()

    def fill_bitmask(self, bitmask: torch.Tensor, idx: int) -> None:
        """填充位掩码。

        # 在位掩码的指定索引位置填充当前步骤允许的 token。
        # 位掩码中的每个 bit 对应词表中的一个 token：
        # - 1 表示允许
        # - 0 表示禁止
        """
        self.matcher.fill_next_token_bitmask(bitmask, idx)

    def is_terminated(self) -> bool:
        """检查语法是否已终止。"""
        return self._is_terminated

    def reset(self):
        """重置语法状态到初始状态。"""
        self.num_processed_tokens = 0
        self.matcher.reset()


# cf https://github.com/mlc-ai/xgrammar/blob/a32ac892676d2eedc0327416105b9b06edfb94b2/cpp/json_schema_converter.cc
# XGrammar 支持的字符串格式列表
STRING_SUPPORTED_FORMATS = {
    "email",
    "date",
    "time",
    "date-time",
    "duration",
    "ipv4",
    "ipv6",
    "hostname",
    "uuid",
    "uri",
    "uri-reference",
    "uri-template",
    "json-pointer",
    "relative-json-pointer",
}


def has_xgrammar_unsupported_json_features(schema: dict[str, Any]) -> bool:
    """Check if JSON schema contains features unsupported by xgrammar.

    # 检查 JSON 模式是否包含 XGrammar 不支持的特性。
    #
    # 不支持的特性包括：
    # 1. 数值类型的 multipleOf 约束
    # 2. 数组的 uniqueItems、contains、minContains、maxContains 约束
    # 3. 字符串的不支持格式（不在 STRING_SUPPORTED_FORMATS 中的）
    # 4. 对象的 patternProperties、propertyNames 约束
    #
    # 递归检查所有嵌套对象和数组。
    """

    def check_object(obj: dict[str, Any]) -> bool:
        if not isinstance(obj, dict):
            return False

        # Check for numeric ranges
        # 检查数值类型的 multipleOf 约束
        if obj.get("type") in ("integer", "number") and ("multipleOf" in obj):
            return True

        # Check for array unsupported keywords
        # 检查数组的不支持关键字
        if obj.get("type") == "array" and any(
            key in obj
            for key in ("uniqueItems", "contains", "minContains", "maxContains")
        ):
            return True

        # Unsupported keywords for strings
        # 检查字符串的不支持格式
        if (
            obj.get("type") == "string"
            and "format" in obj
            and obj["format"] not in STRING_SUPPORTED_FORMATS
        ):
            return True

        # Unsupported keywords for objects
        # 检查对象的不支持关键字
        if obj.get("type") == "object" and any(
            key in obj for key in ("patternProperties", "propertyNames")
        ):
            return True

        # Recursively check all nested objects and arrays
        # 递归检查所有嵌套对象和数组
        for value in obj.values():
            if isinstance(value, dict):
                if check_object(value):
                    return True
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict) and check_object(item):
                        return True

        return False

    return check_object(schema)


def validate_xgrammar_grammar(sampling_params: SamplingParams) -> None:
    """Validate that the request is supported by structured output.

    # 验证请求是否被 XGrammar 后端支持。
    #
    # 验证流程：
    # 1. 正则表达式：尝试编译为语法
    # 2. 选项列表：转换为 EBNF 语法并验证
    # 3. JSON 模式：检查是否包含不支持的特性，并尝试编译
    # 4. EBNF 语法：如果是 Lark 格式先转换为 EBNF，然后验证
    # 5. 结构化标签：验证 JSON 格式和标签定义
    #
    # Raises ValueError if the request is not supported.
    """
    if sampling_params.structured_outputs is None:
        return

    so_params = sampling_params.structured_outputs

    # 验证正则表达式
    if so_params.regex:
        try:
            xgr.Grammar.from_regex(so_params.regex)
        except Exception as err:
            raise ValueError(
                f"Failed to transform regex into a grammar: {err}"
            ) from err

    # 验证选项列表：转换为 EBNF 语法
    if so_params.choice:
        choice_grammar = choice_as_grammar(so_params.choice)
        try:
            xgr.Grammar.from_ebnf(choice_grammar)
        except Exception as err:
            raise ValueError(
                f"Failed to transform choices into a grammar: {err}"
            ) from err
        # 将 choice 转换为 grammar 格式存储
        so_params.choice = None
        so_params.grammar = choice_grammar
        return

    # 验证 JSON 模式
    if so_params.json:
        if isinstance(so_params.json, str):
            try:
                schema = json.loads(so_params.json)
            except json.JSONDecodeError as e:
                raise ValueError("Invalid JSON grammar specification.") from e
        else:
            schema = so_params.json

        if has_xgrammar_unsupported_json_features(schema):
            raise ValueError(
                "The provided JSON schema contains features not supported by xgrammar."
            )

        try:
            xgr.Grammar.from_json_schema(schema)
        except Exception as err:
            raise ValueError(
                f"Failed to transform json schema into a grammar: {err}"
            ) from err
        return

    # 验证 EBNF/Lark 语法
    if so_params.grammar:
        if grammar_is_likely_lark(so_params.grammar):
            # xgrammar supports EBNF grammars only
            # XGrammar 只支持 EBNF 语法，如果是 Lark 格式则先转换
            try:
                so_params.grammar = convert_lark_to_ebnf(so_params.grammar)
            except ValueError as e:
                raise ValueError(
                    "Failed to convert the grammar from Lark to EBNF. "
                ) from e

        # Test parsing EBNF grammar, possibly already converted from Lark
        # 测试解析 EBNF 语法
        try:
            # parse the grammar, but we aren't compiling it.
            xgr.Grammar.from_ebnf(so_params.grammar)
        except Exception as e:
            raise ValueError("Invalid grammar specification.") from e
        return

    # 验证结构化标签
    if so_params.structural_tag:
        try:
            s_tag = json.loads(so_params.structural_tag)

            # Using the deprecated method of compiling structural tag
            # 使用已废弃的方法编译结构化标签（向后兼容）
            if "structures" in s_tag:
                tags = [
                    xgr.StructuralTagItem(
                        begin=s["begin"],
                        schema=json.dumps(s["schema"]),
                        end=s["end"],
                    )
                    for s in s_tag["structures"]
                ]
                xgr.Grammar.from_structural_tag(tags, s_tag["triggers"])
            else:
                xgr.Grammar.from_structural_tag(so_params.structural_tag)
        except Exception as e:
            raise ValueError("Invalid structural tag specification.") from e
