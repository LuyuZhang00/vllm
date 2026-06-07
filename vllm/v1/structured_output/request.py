# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# =============================================================================
# 结构化输出请求 (Structured Output Request)
# =============================================================================
# 本模块定义了结构化输出请求的数据结构和辅助函数。
#
# 核心组件：
# 1. StructuredOutputRequest：结构化输出请求的数据类
#    - 存储请求的结构化输出参数
#    - 管理语法对象（支持异步编译的 Future 模式）
#    - 缓存推理解析器实例
#    - 跟踪推理阶段的结束状态
#
# 2. get_structured_output_key：将结构化输出参数转换为统一的键格式
#    - 根据参数类型返回 (StructuredOutputOptions, 规范字符串) 的元组
#    - 用于语法编译的缓存键和后端分发
#
# 语法编译的异步模式：
# - 语法编译可能耗时较长，因此使用 Future 模式支持异步
# - grammar 属性的 getter 会检查 Future 是否完成
# - is_grammar_ready 属性用于快速检查编译是否完成
# =============================================================================

import dataclasses
import functools
import json
from concurrent.futures import Future
from concurrent.futures._base import TimeoutError
from typing import TYPE_CHECKING, Any, cast

from vllm.sampling_params import SamplingParams, StructuredOutputsParams
from vllm.v1.structured_output.backend_types import (
    StructuredOutputGrammar,
    StructuredOutputKey,
    StructuredOutputOptions,
)

if TYPE_CHECKING:
    from vllm.reasoning import ReasoningParser


@dataclasses.dataclass
class StructuredOutputRequest:
    """结构化输出请求数据类。

    # 每个需要结构化输出的请求都会创建一个 StructuredOutputRequest 实例。
    # 它存储在 Request 对象的 structured_output_request 属性中。
    #
    # 主要字段：
    # - params：结构化输出参数（JSON/正则/语法/选项等）
    # - _grammar：语法对象，支持 Future 异步模式
    # - reasoning_ended：推理阶段是否已结束
    # - reasoner：推理解析器实例（请求级别缓存）
    """

    params: StructuredOutputsParams
    # 语法对象，可以是 Future（异步编译中）、已完成的 Grammar 或 None
    _grammar: Future[StructuredOutputGrammar] | StructuredOutputGrammar | None = None
    # 推理阶段是否已结束（None 表示尚未检测）
    reasoning_ended: bool | None = None
    # 推理解析器的额外参数
    reasoning_parser_kwargs: dict[str, Any] | None = None
    # Cached per request; do not share reasoning parsers across requests because
    # their behavior can depend on reasoning_parser_kwargs.
    # 请求级别的推理解析器缓存。不跨请求共享，因为行为可能依赖于请求参数。
    reasoner: "ReasoningParser | None" = None

    @staticmethod
    def from_sampling_params(
        sampling_params: SamplingParams | None,
    ) -> "StructuredOutputRequest | None":
        """从采样参数创建结构化输出请求。

        # 工厂方法，根据采样参数创建请求实例。
        # 如果没有结构化输出参数或所有约束都为 None，返回 None。

        Args:
            sampling_params: 采样参数，包含结构化输出配置

        Returns:
            StructuredOutputRequest 实例，或 None（无结构化输出需求）
        """
        if sampling_params is None:
            return None
        params = sampling_params.structured_outputs
        if not params or params.all_constraints_none():
            return None
        return StructuredOutputRequest(params=params)

    def _check_grammar_completion(self) -> bool:
        """检查语法编译是否已完成。

        # 此方法实现了异步语法编译的非阻塞检查机制：
        #
        # 处理流程：
        # 1. 检查语法对象是否为 Future 类型（表示异步编译中）
        # 2. 如果是 Future，尝试在 100 微秒内获取结果：
        #    a. 如果在超时前完成，将 Future 替换为实际的 Grammar 对象
        #    b. 更新请求状态为 WAITING（表示可以被调度）
        #    c. 如果超时，返回 False（编译仍在进行中）
        # 3. 如果不是 Future（已完成或未设置），返回 True
        #
        # 设计原因：
        # - 使用 100 微秒超时而非阻塞等待，避免阻塞引擎主循环
        # - 这样调度器可以在语法编译完成前继续处理其他请求
        # - 请求在语法编译完成前保持在 WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR 状态
        """
        # NOTE: We have to lazy import to gate circular imports
        from vllm.v1.request import RequestStatus

        if isinstance(self._grammar, Future):
            try:
                # We will check whether the future is ready within 100 us
                # 尝试在 100 微秒内获取结果
                self._grammar = self._grammar.result(timeout=0.0001)
                self.status = RequestStatus.WAITING
            except TimeoutError:
                return False
        return True

    @property
    def is_grammar_ready(self) -> bool:
        """检查语法是否已准备就绪。

        # 快速检查语法编译是否完成，不阻塞调用者。
        # 用于调度器决定是否可以调度该请求。
        """
        return self._check_grammar_completion()

    @property
    def grammar(self) -> StructuredOutputGrammar | None:
        """获取语法对象。

        # 如果语法编译完成，返回 Grammar 对象。
        # 如果仍在编译中，返回 None。
        """
        completed = self._check_grammar_completion()
        return (
            cast(StructuredOutputGrammar | None, self._grammar) if completed else None
        )

    @grammar.setter
    def grammar(
        self, grammar: StructuredOutputGrammar | Future[StructuredOutputGrammar]
    ) -> None:
        """设置语法对象。

        # 接受 Grammar 对象或 Future 对象。
        # Future 对象表示语法正在异步编译中。
        """
        self._grammar = grammar

    @functools.cached_property
    def structured_output_key(self) -> StructuredOutputKey:
        """获取结构化输出键。

        # 使用缓存属性，每个请求只计算一次。
        # 键格式为 (StructuredOutputOptions, 规范字符串)。
        """
        return get_structured_output_key(self.params)


def get_structured_output_key(params: StructuredOutputsParams) -> StructuredOutputKey:
    """将结构化输出参数转换为统一的键格式。

    # 此函数将各种结构化输出参数统一转换为 (类型, 规范) 的元组格式。
    # 这个键用于：
    # 1. 语法编译的缓存键（相同规范的请求可以复用编译结果）
    # 2. 后端分发（根据类型选择不同的编译方式）
    #
    # 按优先级检查参数类型，返回 (类型枚举, 规范字符串) 的元组：
    #
    # 优先级顺序（先检查到的优先）：
    # 1. json：JSON 模式 -> (JSON, schema_str)
    #    - 如果 json 是字典，序列化为 JSON 字符串
    # 2. json_object：JSON 对象 -> (JSON_OBJECT, "")
    #    - 空字符串表示通用 JSON 对象约束
    # 3. regex：正则表达式 -> (REGEX, pattern)
    # 4. choice：选项列表 -> (CHOICE, json_str)
    #    - 如果 choice 不是字符串，序列化为 JSON 字符串
    # 5. grammar：EBNF 语法 -> (GRAMMAR, grammar_str)
    # 6. structural_tag：结构化标签 -> (STRUCTURAL_TAG, tag_str)
    #
    # 如果没有任何有效参数，抛出 ValueError。
    """
    if params.json is not None:
        if not isinstance(params.json, str):
            json_str = json.dumps(params.json)
        else:
            json_str = params.json
        return StructuredOutputOptions.JSON, json_str
    if params.json_object:
        return StructuredOutputOptions.JSON_OBJECT, ""
    if params.regex is not None:
        return StructuredOutputOptions.REGEX, params.regex
    if params.choice is not None:
        if not isinstance(params.choice, str):
            json_str = json.dumps(params.choice)
        else:
            json_str = params.choice
        return StructuredOutputOptions.CHOICE, json_str
    if params.grammar is not None:
        return StructuredOutputOptions.GRAMMAR, params.grammar
    if params.structural_tag is not None:
        return StructuredOutputOptions.STRUCTURAL_TAG, params.structural_tag
    raise ValueError("No valid structured output parameter found")
