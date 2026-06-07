# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# =============================================================================
# 结构化输出类型定义 (Structured Output Type Definitions)
# =============================================================================
# 本模块定义了结构化输出功能的核心抽象类型：
#
# 1. StructuredOutputOptions (枚举)：定义支持的结构化输出类型
#    - JSON：JSON 模式约束
#    - JSON_OBJECT：通用 JSON 对象约束
#    - REGEX：正则表达式约束
#    - GRAMMAR：EBNF 语法约束
#    - CHOICE：选项列表约束
#    - STRUCTURAL_TAG：结构化标签约束
#
# 2. StructuredOutputGrammar (抽象类)：请求级别的语法接口
#    - 每个需要结构化输出的请求对应一个 Grammar 实例
#    - 管理单个请求的语法状态机（FSM）状态
#    - 提供 token 接受/验证/回滚/位掩码填充等操作
#
# 3. StructuredOutputBackend (抽象类)：引擎级别的后端接口
#    - 整个引擎共享一个 Backend 实例
#    - 负责编译语法规范和分配位掩码内存
#
# 后端实现类（如 XgrammarBackend、GuidanceBackend 等）继承自这些抽象类。
# =============================================================================

import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from vllm.config import VllmConfig
    from vllm.tokenizers import TokenizerLike
else:
    VllmConfig = object
    TokenizerLike = object


class StructuredOutputOptions(enum.Enum):
    """结构化输出类型枚举。

    # 定义了 vLLM 支持的所有结构化输出约束类型：
    JSON = enum.auto()           # JSON 模式约束（指定 schema）
    JSON_OBJECT = enum.auto()    # 通用 JSON 对象约束（无 schema）
    REGEX = enum.auto()          # 正则表达式约束
    GRAMMAR = enum.auto()        # EBNF 语法约束
    CHOICE = enum.auto()         # 选项列表约束（如 ["yes", "no"]）
    STRUCTURAL_TAG = enum.auto() # 结构化标签约束（带标签的分阶段输出）
    """
    JSON = enum.auto()
    JSON_OBJECT = enum.auto()
    REGEX = enum.auto()
    GRAMMAR = enum.auto()
    CHOICE = enum.auto()
    STRUCTURAL_TAG = enum.auto()


# 结构化输出键的类型：(输出类型, 语法规范字符串)
# 用于唯一标识一个结构化输出请求的约束
StructuredOutputKey = tuple[StructuredOutputOptions, str]


class StructuredOutputGrammar(ABC):
    """Request-level backend for structured output requests.

    # 请求级别的结构化输出语法抽象类。
    #
    # 每个需要结构化输出的请求会创建一个 Grammar 实例。
    # 该实例维护请求级别的语法状态机（FSM）状态，
    # 并提供以下核心操作：
    #
    # 1. accept_tokens：接受 token 并推进 FSM
    # 2. validate_tokens：验证 token 是否符合语法（不推进 FSM）
    # 3. rollback：回滚 FSM 状态
    # 4. fill_bitmask：填充下一步允许的 token 位掩码
    # 5. is_terminated：检查语法是否已完成
    # 6. reset：重置 FSM 到初始状态
    """

    @abstractmethod
    def accept_tokens(self, request_id: str, tokens: list[int]) -> bool:
        """
        Determines whether the provided tokens are accepted for the
        given request.

        # 判断提供的 token 是否被接受并推进 FSM。
        # 如果所有 token 都被接受，返回 True 并更新内部状态。
        # 如果任何 token 不被接受，返回 False。
        #
        # 使用场景：
        # - 在解码阶段，每生成一个 token 后调用此方法推进 FSM
        # - 在投机解码中，验证草稿 token 后调用此方法推进 FSM
        # - 在位掩码生成中，模拟投机 token 的接受以生成正确的位掩码

        Args:
            request_id (str): The unique identifier for the request.
            tokens (list[int]): A list of token IDs to evaluate.

        Returns:
            bool: True if the tokens are accepted, False otherwise.
        """

    @abstractmethod
    def validate_tokens(self, tokens: list[int]) -> list[int]:
        """
        Validates the provided tokens against the grammar.
        Will not advance the FSM.

        # 验证 token 列表是否符合语法，但不推进 FSM。
        # 返回被接受的 token 前缀列表。
        #
        # 使用场景：
        # - 投机解码中验证草稿 token 是否符合语法约束
        # - 在验证过程中不修改 FSM 状态，确保可以安全地多次调用
        #
        # 返回值说明：
        # - 返回的列表是输入 token 的前缀
        # - 如果所有 token 都被接受，返回完整的 token 列表
        # - 如果没有任何 token 被接受，返回空列表

        Args:
            tokens (list[int]): A list of token IDs to validate.

        Returns:
            list[int]: A list of accepted token IDs. Will be a prefix
                of the input tokens, and empty if none are accepted.
        """

    @abstractmethod
    def rollback(self, num_tokens: int) -> None:
        """
        Rolls back the state of the grammar by a specified number of tokens.
        Will also revert counters for the number of processed tokens.

        # 回滚语法状态指定数量的 token。
        # 同时恢复已处理 token 的计数器。
        #
        # 使用场景：
        # - 投机解码中当草稿 token 被拒绝时，回滚 FSM 到接受前的状态
        # - 位掩码生成中，模拟投机 token 接受后回滚（因为实际接受尚未确定）
        #
        # 实现要求：
        # - 回滚操作必须是精确的，恢复到接受指定数量 token 之前的状态
        # - 某些后端（如 Guidance）需要特殊处理 EOS token 的回滚偏移

        Args:
            num_tokens (int): The number of tokens to roll back.
        """

    @abstractmethod
    def fill_bitmask(self, bitmask: "torch.Tensor", batch_index: int) -> None:
        """
        Fills the bitmask for a specific batch index.

        # 在位掩码的指定批次索引位置填充当前步骤允许的 token。
        # 位掩码中 bit 为 1 表示允许该 token，为 0 表示禁止。
        #
        # 位掩码格式：
        # - 形状为 (batch_size, ceil(vocab_size / 32))
        # - 每个 int32 值的 32 个 bit 对应 32 个 token
        # - bit 为 1 表示允许，0 表示禁止
        #
        # 使用场景：
        # - 在每次解码迭代中，为所有需要结构化输出的请求填充位掩码
        # - 位掩码随后被传递给 GPU 模型运行器，用于修改 logit 值

        Args:
            bitmask (torch.Tensor): The bitmask to fill
            batch_index (int): The index in the bitmask to fill
        """

    @abstractmethod
    def is_terminated(self) -> bool:
        """
        Checks whether the structured output process has terminated.

        # 检查结构化输出过程是否已终止。
        # 例如：JSON 结构已完成、匹配到 EOS token 等。

        Returns:
            bool: True if the process is terminated, False otherwise.
        """

    @abstractmethod
    def reset(self):
        """
        Resets the state of the structured output grammar.

        # 重置结构化输出语法的状态到初始状态。
        """


@dataclass
class StructuredOutputBackend(ABC):
    """Engine-level backend for structured output requests.

    # 引擎级别的结构化输出后端抽象类。
    #
    # 整个引擎共享一个 Backend 实例。
    # 负责：
    # 1. 编译语法规范为 Grammar 对象
    # 2. 分配位掩码内存
    # 3. 管理后端资源的生命周期
    #
    # 具体后端实现包括：
    # - XgrammarBackend：基于 XGrammar 库
    # - GuidanceBackend：基于 llguidance 库
    # - OutlinesBackend：基于 outlines_core 库
    # - LMFormatEnforcerBackend：基于 lm-format-enforcer 库
    """

    vllm_config: VllmConfig
    tokenizer: TokenizerLike
    vocab_size: int

    @abstractmethod
    def compile_grammar(
        self, request_type: StructuredOutputOptions, grammar_spec: str
    ) -> StructuredOutputGrammar:
        """
        Compiles a grammar specification into a structured output grammar.

        # 将语法规范编译为 StructuredOutputGrammar 对象。
        #
        # 编译过程可能涉及：
        # - 将 JSON 模式转换为有限状态机
        # - 将正则表达式编译为 NFA/DFA
        # - 将 EBNF 语法解析为语法图
        # - 构建结构化标签的匹配器

        Args:
            request_type (StructuredOutputOptions): The type of structured
                output request.
            grammar_spec (str): The grammar specification to compile.

        Returns:
            StructuredOutputGrammar: The compiled structured output grammar.
        """

    @abstractmethod
    def allocate_token_bitmask(self, max_num_seqs: int) -> "torch.Tensor":
        """
        Allocates a token bitmask for the specified maximum number of sequences.

        # 为指定的最大序列数分配 token 位掩码。
        #
        # 位掩码形状通常为 (max_num_seqs * (1 + num_spec_tokens), vocab_size)，
        # 其中 num_spec_tokens 是投机解码的 token 数。

        Args:
            max_num_seqs (int): The maximum number of sequences for which
                to allocate the bitmask.
        """

    @abstractmethod
    def destroy(self):
        """
        Backend-specific cleanup.

        # 后端特定的清理操作，释放资源。
        """
