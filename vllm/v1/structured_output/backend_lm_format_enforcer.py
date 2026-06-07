# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# =============================================================================
# LM Format Enforcer 结构化输出后端 (LM Format Enforcer Backend)
# =============================================================================
# LM Format Enforcer 是 vLLM 支持的结构化输出后端之一。
# 它通过字符级别的解析器来约束模型输出，支持：
#
# 1. JSON Schema：通过 JsonSchemaParser 约束 JSON 输出
# 2. JSON Object：约束输出为任意 JSON 对象
# 3. 正则表达式（REGEX）：通过 RegexParser 约束输出
# 4. 选项列表（CHOICE）：通过 UnionParser + StringParser 约束输出
#
# 注意：不支持 EBNF 语法和结构化标签，也不支持投机解码。
#
# 工作原理：
# 1. 字符级解析器（CharacterLevelParser）在字符级别跟踪输出状态
# 2. TokenEnforcer 将字符级约束转换为 token 级约束
# 3. 通过 get_allowed_tokens 获取当前允许的 token 列表
# 4. 使用位掩码将允许的 token 信息传递给采样过程
# =============================================================================

import ast
import json
from dataclasses import dataclass, field
from functools import lru_cache
from typing import TYPE_CHECKING

import torch
from transformers import PreTrainedTokenizerBase

from vllm.sampling_params import SamplingParams
from vllm.utils.import_utils import LazyLoader
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.v1.structured_output.backend_types import (
    StructuredOutputBackend,
    StructuredOutputGrammar,
    StructuredOutputOptions,
)

if TYPE_CHECKING:
    import lmformatenforcer
    import lmformatenforcer.integrations.vllm as lmfe_vllm
else:
    lmformatenforcer = LazyLoader("lmformatenforcer", globals(), "lmformatenforcer")
    lmfe_vllm = LazyLoader(
        "lmformatenforcer.integrations.vllm",
        globals(),
        "lmformatenforcer.integrations.vllm",
    )


@lru_cache
def _cached_build_vllm_token_enforcer_tokenizer_data(
    tokenizer: PreTrainedTokenizerBase, vocab_size: int
) -> "lmfe_vllm.TokenEnforcerTokenizerData":
    """构建并缓存 vLLM Token Enforcer 的分词器数据。

    # 使用 LRU 缓存避免重复构建。
    # 分词器数据包含了将字符级约束转换为 token 级约束所需的信息。
    # use_bitmask=True 表示使用位掩码格式输出允许的 token。
    """
    return lmfe_vllm.build_vllm_token_enforcer_tokenizer_data(
        tokenizer, use_bitmask=True, vocab_size=vocab_size
    )


@dataclass
class LMFormatEnforcerGrammar(StructuredOutputGrammar):
    """LM Format Enforcer 请求级别语法对象。

    # 每个需要结构化输出的请求创建一个实例。
    # 通过维护已生成的 token 前缀来跟踪输出状态，
    # 并使用 TokenEnforcer 来确定每步允许的 token。
    """

    # TokenEnforcer 实例，用于计算允许的 token
    token_enforcer: lmformatenforcer.TokenEnforcer
    # 当前已生成的 token 前缀列表
    current_tokens_prefix: list[int] = field(default_factory=list)

    def accept_tokens(self, request_id: str, tokens: list[int]) -> bool:
        """接受 token 列表并更新内部状态。

        # 处理流程：
        # 1. 记录当前前缀长度（用于失败时回滚）
        # 2. 逐个检查 token 是否被当前前缀允许：
        #    a. 调用 get_allowed_tokens 获取当前前缀下允许的 token 集合
        #    b. 调用 is_token_allowed 检查 token 是否在允许集合中
        # 3. 如果任何 token 不被允许：
        #    a. 回滚已追加的部分 token（确保原子性）
        #    b. 返回 False
        # 4. 所有 token 都被接受则追加到前缀并返回 True
        #
        # 注意：LM Format Enforcer 的 token 约束是基于字符级解析器的，
        # 每次调用 get_allowed_tokens 都会重新计算允许的 token 集合。
        """
        original_len = len(self.current_tokens_prefix)
        for token in tokens:
            if not self.token_enforcer.get_allowed_tokens(
                self.current_tokens_prefix
            ).is_token_allowed(token):
                # Rollback partial updates to ensure atomicity.
                # 回滚部分更新以确保原子性
                del self.current_tokens_prefix[original_len:]
                return False
            self.current_tokens_prefix.append(token)
        return True

    def validate_tokens(self, tokens: list[int]) -> list[int]:
        """验证 token 列表是否符合语法，但不修改内部状态。

        # 验证流程：
        # 1. 遍历 token 列表，逐个验证
        # 2. 对于每个 token，构建临时前缀（当前前缀 + 已验证 token）
        # 3. 调用 get_allowed_tokens 获取临时前缀下允许的 token
        # 4. 检查当前 token 是否在允许集合中
        # 5. 一旦遇到不被接受的 token，停止并返回已接受的前缀
        #
        # 注意：此方法不修改内部状态（current_tokens_prefix），
        # 因此可以安全地用于投机解码的验证。
        """
        for prefix_length in range(len(tokens)):
            prefix = tokens[:prefix_length]
            next_token = tokens[prefix_length]
            if not self.token_enforcer.get_allowed_tokens(
                self.current_tokens_prefix + prefix
            ).is_token_allowed(next_token):
                break
        else:
            return tokens

        return tokens[:prefix_length]

    def rollback(self, num_tokens: int) -> None:
        """回滚指定数量的 token。

        # 通过截断 current_tokens_prefix 列表实现回滚。
        """
        self.current_tokens_prefix = self.current_tokens_prefix[:-num_tokens]

    def fill_bitmask(self, bitmask: torch.Tensor, batch_index: int) -> None:
        """填充位掩码。

        # 获取当前前缀下允许的 token 列表，
        # 并将其转换为位掩码格式写入指定位置。
        """
        allowed_tokens = self.token_enforcer.get_allowed_tokens(
            self.current_tokens_prefix
        )
        bitmask[batch_index] = allowed_tokens.allowed_tokens

    def is_terminated(self) -> bool:
        """检查语法是否已终止。

        # 如果前缀的最后一个 token 是 EOS token，则认为已终止。
        """
        # We are considered terminated if the prefix ends with eos_token_id
        return_value = (
            len(self.current_tokens_prefix) > 0
            and self.current_tokens_prefix[-1] == self.token_enforcer.eos_token_id
        )
        return return_value

    def reset(self):
        """重置语法状态到初始状态。"""
        self.current_tokens_prefix = []


@dataclass
class LMFormatEnforcerBackend(StructuredOutputBackend):
    """LM Format Enforcer 引擎级别后端。

    # 负责：
    # 1. 构建并缓存分词器数据
    # 2. 编译各种类型的语法规范
    # 3. 分配位掩码内存
    #
    # 注意：不支持投机解码（会抛出 ValueError）。
    """

    def __post_init__(self):
        # 构建并缓存分词器数据（LRU 缓存，相同分词器只构建一次）
        self.tokenizer_data = _cached_build_vllm_token_enforcer_tokenizer_data(
            self.tokenizer, self.vocab_size
        )

    def compile_grammar(
        self, request_type: StructuredOutputOptions, grammar_spec: str
    ) -> StructuredOutputGrammar:
        """编译语法规范为 LMFormatEnforcerGrammar 对象。

        # 处理流程：
        # 1. 根据请求类型创建字符级解析器（CharacterLevelParser）：
        #    - JSON: JsonSchemaParser，解析 JSON Schema 并约束输出
        #    - JSON_OBJECT: JsonSchemaParser(None)，约束为任意 JSON 对象
        #    - REGEX: RegexParser，使用正则表达式约束输出
        #    - CHOICE: UnionParser + StringParser，约束为预定义选项之一
        # 2. 检查是否启用投机解码（不支持，会抛出异常）
        # 3. 创建 TokenEnforcer，将字符级约束转换为 token 级约束
        #
        # LM Format Enforcer 的工作原理：
        # - 字符级解析器在字符级别跟踪输出状态
        # - TokenEnforcer 将字符级约束转换为 token 级约束
        # - 对于每个 token 前缀，TokenEnforcer 计算哪些 token 是合法的
        # - 这种方法灵活但效率较低（需要为每个 token 重新计算）
        """
        character_level_parser: lmformatenforcer.CharacterLevelParser
        if request_type == StructuredOutputOptions.JSON:
            spec_dict = json.loads(grammar_spec)
            character_level_parser = lmformatenforcer.JsonSchemaParser(spec_dict)
        elif request_type == StructuredOutputOptions.JSON_OBJECT:
            character_level_parser = lmformatenforcer.JsonSchemaParser(None)
        elif request_type == StructuredOutputOptions.REGEX:
            character_level_parser = lmformatenforcer.RegexParser(grammar_spec)
        elif request_type == StructuredOutputOptions.CHOICE:
            # 使用 ast.literal_eval 解析选项列表字符串
            choices = ast.literal_eval(grammar_spec)
            character_level_parser = lmformatenforcer.UnionParser(
                [lmformatenforcer.StringParser(choice) for choice in choices]
            )
        else:
            raise ValueError(
                f"Invalid request type for LM Format Enforcer backend({request_type!s})"
            )
        # 检查是否启用了投机解码
        max_rollback_tokens = (
            self.vllm_config.speculative_config.num_speculative_tokens
            if self.vllm_config.speculative_config is not None
            else 0
        )

        # LM Format Enforcer 不支持投机解码，因为它没有高效的回滚机制
        if max_rollback_tokens > 0:
            raise ValueError(
                "LM Format Enforcer backend does not support speculative tokens"
            )

        # 创建 TokenEnforcer，将字符级解析器转换为 token 级约束
        # tokenizer_data 包含了将字符级约束映射到 token 级所需的信息
        token_enforcer = lmformatenforcer.TokenEnforcer(
            tokenizer_data=self.tokenizer_data,
            parser=character_level_parser,
        )
        return LMFormatEnforcerGrammar(token_enforcer)

    def allocate_token_bitmask(self, max_num_seqs: int) -> torch.Tensor:
        """分配 token 位掩码张量。

        # 形状为 (max_num_seqs, ceil(vocab_size / 32))，
        # 每个 token 用一个 bit 表示，初始值为 -1（全 1，允许所有 token）。
        # 使用 pin_memory 优化 CPU 到 GPU 的传输（如果可用）。
        """
        return torch.full(
            (max_num_seqs, (self.vocab_size + 31) // 32),
            -1,
            dtype=torch.int32,
            pin_memory=is_pin_memory_available(),
        )

    def destroy(self):
        """LM Format Enforcer 后端无需特殊清理。"""
        pass


def validate_structured_output_request_lm_format_enforcer(params: SamplingParams):
    """验证 LM Format Enforcer 后端的结构化输出请求。

    # 验证流程：
    # 1. 正则表达式：直接通过（无需额外验证）
    # 2. JSON 模式：验证 JSON 格式的有效性
    # 3. 选项列表：直接通过
    # 4. EBNF 语法：不支持，抛出 ValueError
    # 5. 结构化标签：不支持（未列出）
    """
    if params.structured_outputs is None:
        return

    so_params = params.structured_outputs

    # 正则表达式直接通过验证
    if so_params.regex:
        return
    elif so_params.json:
        if isinstance(so_params.json, str):
            try:
                # make sure schema is valid json
                # 确保 JSON 模式是有效的 JSON 字符串
                json.loads(so_params.json)
            except json.JSONDecodeError as e:
                raise ValueError("Invalid JSON grammar specification.") from e
        else:
            try:
                json.dumps(so_params.json)
            except Exception as e:
                raise ValueError(
                    f"Error serializing structured outputs jsonschema: {e}"
                ) from e
        return
    # 选项列表直接通过验证
    elif so_params.choice:
        return
    # EBNF 语法不被 LM Format Enforcer 支持
    elif so_params.grammar:
        raise ValueError(
            "LM Format Enforcer structured outputs backend "
            "does not support grammar specifications"
        )
