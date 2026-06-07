# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# =============================================================================
# Guidance 结构化输出后端 (Guidance Backend)
# =============================================================================
# Guidance（基于 llguidance 库）是 vLLM 支持的结构化输出后端之一。
# llguidance 是一个高性能的语法引导解码库，支持多种约束类型。
#
# 本模块实现了基于 Guidance/llguidance 的结构化输出后端，支持：
# 1. JSON Schema：通过 JSON 模式约束输出格式
# 2. JSON Object：约束输出为任意 JSON 对象
# 3. 正则表达式（REGEX）：通过正则表达式约束输出
# 4. EBNF 语法（GRAMMAR）：通过语法约束输出
# 5. 选项列表（CHOICE）：约束输出为预定义选项之一
# 6. 结构化标签（STRUCTURAL_TAG）：约束带标签的结构化输出
#
# 核心组件：
# - GuidanceBackend：引擎级别的后端，负责编译语法和分配位掩码
# - GuidanceGrammar：请求级别的语法对象，管理单个请求的匹配器状态
#
# 特殊处理：
# - additionalProperties：自动为 JSON 模式添加 additionalProperties: false
# - Mistral 分词器：使用专用的 llg_tokenizer 接口
# =============================================================================

import copy
import json
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams
from vllm.utils.import_utils import LazyLoader
from vllm.utils.mistral import is_mistral_tokenizer
from vllm.v1.structured_output.backend_types import (
    StructuredOutputBackend,
    StructuredOutputGrammar,
    StructuredOutputOptions,
)
from vllm.v1.structured_output.request import get_structured_output_key

if TYPE_CHECKING:
    import llguidance
    import llguidance.hf as llguidance_hf
    import llguidance.torch as llguidance_torch
else:
    llguidance = LazyLoader("llguidance", globals(), "llguidance")
    llguidance_hf = LazyLoader("llguidance.hf", globals(), "llguidance.hf")
    llguidance_torch = LazyLoader("llguidance.torch", globals(), "llguidance.torch")

logger = init_logger(__name__)


def _walk_json_for_additional_properties(data: object):
    """递归遍历 JSON 模式，为缺少 additionalProperties 的对象添加默认值。

    # 遍历逻辑：
    # 1. 对于字典（对象）：
    #    - 先递归处理所有值
    #    - 如果有 properties 或 patternProperties 但没有 additionalProperties，
    #      添加 additionalProperties: false
    # 2. 对于列表（数组）：
    #    - 递归处理每个元素
    #
    # 这确保了 JSON 输出不会包含模式中未定义的额外属性。
    """
    if isinstance(data, dict):
        for value in data.values():
            _walk_json_for_additional_properties(value)
        if "additionalProperties" not in data and (
            "properties" in data or "patternProperties" in data
        ):
            data["additionalProperties"] = False
    elif isinstance(data, list):
        for item in data:
            _walk_json_for_additional_properties(item)


def has_guidance_unsupported_json_features(schema: dict[str, Any]) -> bool:
    """Check if JSON schema contains features unsupported by guidance/llguidance.

    # 检查 JSON 模式是否包含 llguidance 不支持的特性。
    #
    # 当前不支持的特性：
    # - patternProperties：正则表达式属性名匹配
    #
    # 递归检查所有嵌套对象和数组。
    """

    def check_object(obj: dict[str, Any]) -> bool:
        if not isinstance(obj, dict):
            return False

        # patternProperties is not supported by llguidance
        if "patternProperties" in obj:
            return True

        # Recursively check all nested objects and arrays
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


def process_for_additional_properties(
    guide_json: str | dict[str, Any],
) -> dict[str, Any]:
    """处理 JSON 模式，为对象添加 additionalProperties: false。

    # 如果输入是字符串，先解析为字典。
    # 如果输入是字典，深拷贝后处理（避免修改原始数据）。
    # 然后调用 _walk_json_for_additional_properties 递归处理。
    """
    if isinstance(guide_json, str):
        guide_json_obj = json.loads(guide_json)
    else:
        # copy for modifications
        guide_json_obj = copy.deepcopy(guide_json)
    _walk_json_for_additional_properties(guide_json_obj)
    return guide_json_obj


@dataclass
class GuidanceBackend(StructuredOutputBackend):
    """Guidance 引擎级别后端。

    # 负责：
    # 1. 初始化 llguidance 的分词器封装
    # 2. 编译各种类型的语法规范
    # 3. 分配位掩码内存
    """

    def __post_init__(self):
        # 是否禁用任意空白字符
        self.disable_any_whitespace = (
            self.vllm_config.structured_outputs_config.disable_any_whitespace
        )
        # 是否禁用 additionalProperties（自动为 JSON 对象添加）
        self.disable_additional_properties = (
            self.vllm_config.structured_outputs_config.disable_additional_properties
        )

        # 为不同类型的分词器创建 llguidance 的分词器封装
        if is_mistral_tokenizer(self.tokenizer):
            # Mistral 分词器使用专用的 llg_tokenizer 接口
            self.ll_tokenizer = self.tokenizer.llg_tokenizer
        else:
            # 其他分词器使用 from_huggingface 转换
            self.ll_tokenizer = llguidance_hf.from_tokenizer(
                self.tokenizer, max(self.vocab_size, len(self.tokenizer))
            )

    def compile_grammar(
        self, request_type: StructuredOutputOptions, grammar_spec: str
    ) -> StructuredOutputGrammar:
        """编译语法规范为 GuidanceGrammar 对象。

        # 处理流程：
        # 1. 将语法规范序列化为 llguidance 可识别的格式
        # 2. 创建 LLMatcher 匹配器实例
        # 3. 包装为 GuidanceGrammar 对象并检查错误
        """
        # 将语法规范序列化为 llguidance 格式
        self.serialized_grammar = serialize_guidance_grammar(
            request_type,
            grammar_spec,
            self.disable_any_whitespace,
            self.disable_additional_properties,
        )

        # 创建 LLMatcher 匹配器
        # log_level 通过环境变量 LLGUIDANCE_LOG_LEVEL 控制
        ll_matcher = llguidance.LLMatcher(
            self.ll_tokenizer,
            self.serialized_grammar,
            log_level=int(os.environ.get("LLGUIDANCE_LOG_LEVEL", "1")),
        )

        r = GuidanceGrammar(
            ll_matcher=ll_matcher,
            ll_tokenizer=self.ll_tokenizer,
            vocab_size=self.vocab_size,
        )

        r.check_error()
        return r

    def allocate_token_bitmask(self, max_num_seqs: int):
        """分配 token 位掩码张量。"""
        return llguidance_torch.allocate_token_bitmask(
            max_num_seqs, self.ll_tokenizer.vocab_size
        )

    def destroy(self):
        """Guidance 后端无需特殊清理。"""
        pass


@dataclass
class GuidanceGrammar(StructuredOutputGrammar):
    """Guidance 请求级别语法对象。

    # 每个需要结构化输出的请求创建一个 GuidanceGrammar 实例。
    # 封装了 llguidance 的 LLMatcher，用于：
    # 1. 跟踪请求的语法匹配状态
    # 2. 接受或拒绝 token
    # 3. 生成下一步允许的 token 位掩码
    # 4. 支持回滚操作
    # 5. 处理 EOS token 和终止状态
    """

    ll_matcher: llguidance.LLMatcher
    ll_tokenizer: llguidance.LLTokenizer
    vocab_size: int
    # 是否已打印过错误信息（避免重复打印）
    printed_error: bool = False
    # 语法是否已终止
    terminated: bool = False
    # 回滚滞后量：处理 EOS token 时的特殊偏移
    rollback_lag: int = 0

    def check_error(self):
        """检查并记录 LLMatcher 的错误。

        # 只打印一次错误信息，避免日志重复。
        """
        if not self.printed_error:
            err = self.ll_matcher.get_error()
            if err:
                self.printed_error = True
                logger.warning("LLMatcher error: %s", err)

    def accept_tokens(self, request_id: str, tokens: list[int]) -> bool:
        """Accepts a list of tokens and advances the parser.

        # 接受 token 列表并推进解析器。
        #
        # 处理逻辑：
        # 1. 检测 EOS token：如果包含 EOS 且匹配器已停止，设置终止状态
        # 2. 如果匹配器已停止，直接返回 True（不再处理 token）
        # 3. 调用 consume_tokens 消费 token 并推进解析器
        # 4. 检查是否有错误发生
        #
        # 返回 True 表示解析器成功推进，False 表示推进失败。

        Returns True if the parser was advanced successfully.
        Returns False if the parser failed to advance.
        """

        if self.ll_tokenizer.eos_token in tokens:
            if self.ll_matcher.is_stopped() and not self.terminated:
                # 设置回滚滞后量为 1，用于 EOS token 的特殊处理
                self.rollback_lag = 1
            self.terminated = True

        if self.ll_matcher.is_stopped():
            return True

        # TODO - Add jump decoding support in the future:
        # self.ll_matcher.compute_ff_bytes() - this should always work
        # self.ll_matcher.compute_ff_tokens() - this only works for
        #   "canonical" tokenizers
        # For conversion between the two, see
        # https://github.com/guidance-ai/llguidance/blob/main/docs/fast_forward.md
        #
        # 未来计划支持跳转解码（jump-forward decoding）以加速推理。

        r = self.ll_matcher.consume_tokens(tokens)

        self.check_error()

        return r

    def validate_tokens(self, tokens: list[int]) -> list[int]:
        """Checks if the list of tokens are accepted by the parser in sequence.
        Will not advance the parser.

        # 验证 token 列表是否被解析器按顺序接受，但不推进解析器。
        # 主要用于投机解码中验证草稿 token。
        #
        # 返回被接受的 token 前缀列表。

        Returns the prefix list of tokens that are accepted by the parser.
        """
        if len(tokens) == 0:
            return []
        if self.ll_matcher.is_stopped():
            return []

        num_tokens = self.ll_matcher.validate_tokens(tokens)

        self.check_error()

        return tokens[:num_tokens]

    def rollback(self, num_tokens: int) -> None:
        """回滚解析器状态。

        # 回滚指定数量的 token，恢复解析器到之前的状态。
        # 注意：回滚时考虑 rollback_lag（EOS token 的特殊偏移）。
        # 回滚后重置终止状态。
        """
        if num_tokens > 0:
            self.ll_matcher.rollback(num_tokens - self.rollback_lag)
            self.terminated = False
            self.rollback_lag = 0
            self.check_error()

    def fill_bitmask(self, bitmask: torch.Tensor, idx: int) -> None:
        """填充位掩码。

        # 调用 llguidance 的 torch 接口填充位掩码。
        # 如果匹配器已停止或处于错误状态，自动返回 [EOS] 掩码。
        """
        # this will automatically return [EOS] mask if the matcher is stopped
        # or otherwise in an error state
        llguidance_torch.fill_next_token_bitmask(self.ll_matcher, bitmask, idx)
        self.check_error()

    def is_terminated(self) -> bool:
        """检查语法是否已终止。"""
        return self.terminated

    def reset(self):
        """重置匹配器到初始状态。"""
        # This method may be not needed anymore? TODO
        self.ll_matcher.reset()


def serialize_guidance_grammar(
    request_type: StructuredOutputOptions,
    grammar_spec: str | dict[str, Any],
    disable_any_whitespace: bool = False,
    disable_additional_properties: bool = False,
) -> str:
    """将语法规范序列化为 llguidance 可识别的格式。

    # 根据请求类型选择不同的序列化方式：
    #
    # 1. JSON / JSON_OBJECT：
    #    - 调用 LLMatcher.grammar_from_json_schema 转换
    #    - 可选处理 additionalProperties 和空白字符灵活性
    #
    # 2. REGEX / GRAMMAR / CHOICE：
    #    - 使用 llguidance.grammar_from 统一转换
    #
    # 3. STRUCTURAL_TAG：
    #    - 解析 JSON 格式的标签定义
    #    - 为每个结构创建 StructTag 对象
    #    - 使用 StructTag.to_grammar 转换为语法
    """
    def _process_schema(
        grammar_spec: str | dict[str, Any],
    ) -> str:
        """处理 JSON 模式，应用 additionalProperties 和空白字符设置。"""
        if disable_additional_properties:
            grammar_spec = process_for_additional_properties(grammar_spec)
        return llguidance.LLMatcher.grammar_from_json_schema(
            grammar_spec,
            defaults={
                "whitespace_flexible": not disable_any_whitespace,
            },
        )

    if request_type == StructuredOutputOptions.JSON:
        return _process_schema(grammar_spec)
    elif request_type == StructuredOutputOptions.JSON_OBJECT:
        return llguidance.LLMatcher.grammar_from_json_schema(
            '{"type": "object"}',
            defaults={
                "whitespace_flexible": not disable_any_whitespace,
            },
        )
    else:
        if request_type == StructuredOutputOptions.REGEX:
            tp = "regex"
        elif request_type == StructuredOutputOptions.GRAMMAR:
            tp = "grammar"
        elif request_type == StructuredOutputOptions.CHOICE:
            tp = "choice"
        elif request_type == StructuredOutputOptions.STRUCTURAL_TAG:
            # 处理结构化标签
            if isinstance(grammar_spec, str):
                s_tag = json.loads(grammar_spec)
            else:
                s_tag = grammar_spec
            triggers: list[str] = s_tag["triggers"]
            tags: list[llguidance.StructTag] = []
            for s in s_tag["structures"]:
                begin: str = s["begin"]
                # 查找与 begin 匹配的 trigger
                trig = next((t for t in triggers if begin.startswith(t)), None)
                if trig is None:
                    raise ValueError(
                        f"Trigger {begin} not found in triggers {triggers}"
                    )
                tags.append(
                    llguidance.StructTag(
                        trigger=trig,
                        begin=s["begin"],
                        grammar=_process_schema(s["schema"]),
                        end=s["end"],
                    )
                )
            if not tags:
                raise ValueError("No structural tags found in the grammar spec.")
            return llguidance.StructTag.to_grammar(tags)
        else:
            logger.error(
                "Validation should have already occurred. Please file an issue."
            )
            raise ValueError(
                f"grammar is not of valid supported types. ({request_type!s})"
            )
        return llguidance.grammar_from(tp, grammar_spec)


def validate_guidance_grammar(
    sampling_params: SamplingParams, tokenizer: llguidance.LLTokenizer | None = None
) -> None:
    """验证 Guidance 语法是否有效。

    # 处理流程：
    # 1. 检查是否启用了结构化输出
    # 2. 获取结构化输出键（类型 + 规范）
    # 3. 将语法序列化为 llguidance 格式
    # 4. 使用 LLMatcher.validate_grammar 验证语法
    # 5. 如果有错误，抛出 ValueError
    """
    # if structured output is not enabled, there is nothing to validate
    if sampling_params.structured_outputs is None:
        return
    tp, grm = get_structured_output_key(sampling_params.structured_outputs)
    guidance_grm = serialize_guidance_grammar(tp, grm)
    err = llguidance.LLMatcher.validate_grammar(guidance_grm, tokenizer)
    if err:
        raise ValueError(f"Grammar error: {err}")
