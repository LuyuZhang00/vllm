# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# =============================================================================
# 结构化输出工具函数 (Structured Output Utilities)
# =============================================================================
# 本模块提供结构化输出功能所需的工具函数和辅助类，包括：
#
# 1. 语法位掩码应用：apply_grammar_bitmask() 将语法约束应用到模型输出的 logit 上
# 2. Outlines 词汇表管理：OutlinesVocabulary 封装 outlines_core 的词汇表对象
# 3. 缓存管理：get_outlines_cache() 管理语法索引的缓存
# 4. 词汇表简化：_reduced_vocabulary() 将分词器词汇表转换为字节级映射
# 5. 语法格式转换：Lark -> EBNF 格式转换、选项列表 -> EBNF 语法转换
# 6. 语法格式检测：grammar_is_likely_lark() 检测语法是否为 Lark 格式
# =============================================================================

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import tempfile
from typing import TYPE_CHECKING

import numpy as np
import regex as re
import torch
from cachetools import LRUCache

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.utils.import_utils import LazyLoader
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput

if TYPE_CHECKING:
    import outlines_core as oc
    import transformers.convert_slow_tokenizer as convert_slow_tokenizer
    import transformers.file_utils as file_utils
    import xgrammar as xgr

    from vllm.tokenizers import TokenizerLike
    from vllm.v1.worker.gpu_input_batch import InputBatch
else:
    xgr = LazyLoader("xgr", globals(), "xgrammar")
    oc = LazyLoader("oc", globals(), "outlines_core")
    file_utils = LazyLoader("file_utils", globals(), "transformers.file_utils")
    convert_slow_tokenizer = LazyLoader(
        "convert_slow_tokenizer", globals(), "transformers.convert_slow_tokenizer"
    )


logger = init_logger(__name__)

# 全局缓存实例，用于 outlines 索引缓存
CACHE = None


def apply_grammar_bitmask(
    scheduler_output: SchedulerOutput,
    grammar_output: GrammarOutput,
    input_batch: InputBatch,
    logits: torch.Tensor,
) -> None:
    """将语法位掩码应用到模型输出的 logit 上。

    # 此函数在 GPU 模型运行器的前向传播后被调用，
    # 根据语法约束修改 logit 值，使不符合语法的 token 的 logit 被设为 -inf。
    # 这样在采样时，被禁止的 token 就不会被选中。
    #
    # 核心挑战：
    # - 调度器传来的位掩码只包含有结构化输出的请求（压缩格式）
    # - GPU 运行器批次中的请求顺序可能与位掩码中的顺序不同
    # - 投机解码会在 logit 维度引入额外的偏移
    #
    # 处理流程（步骤编号）：
    # 1. 接收调度器传来的压缩位掩码（numpy 数组格式，序列化效率高）
    # 2. 计算每个请求在 logit 张量中的实际索引（考虑投机解码偏移）
    # 3. 创建与 logit 张量相同大小的排序位掩码，填充默认值 -1（全允许）
    # 4. 将压缩位掩码重新排列到正确的位置
    # 5. 将位掩码异步复制到 GPU 设备
    # 6. 调用 xgrammar 的 apply_token_bitmask_inplace 原地修改 logit
    #
    # 优化细节：
    # - 如果所有请求都有结构化输出（位掩码完全对齐），跳过索引传递
    # - 使用 pin_memory 和 non_blocking 传输避免 GPU 同步等待
    # - CPU 路径需要处理 float32 类型兼容性问题

    Args:
        scheduler_output (SchedulerOutput): The result of engine scheduling.
        grammar_output (GrammarOutput): 结构化输出的语法输出，包含位掩码和请求ID列表
        input_batch (InputBatch): The input of model runner.
        logits (torch.Tensor): The output logits of model forward.
    """
    # Serialization of np.ndarray is much more efficient than a tensor,
    # so we receive it in that format.
    # 使用 numpy 数组格式接收位掩码，因为序列化效率比 tensor 高得多
    grammar_bitmask = grammar_output.grammar_bitmask

    # We receive the structured output bitmask from the scheduler,
    # compacted to contain bitmasks only for structured output requests.
    # The order of the requests in the bitmask is not guaranteed to be the
    # same as the order of the requests in the gpu runner's batch. We need
    # to sort the bitmask to match the order of the requests used here.
    #
    # 位掩码中请求的顺序可能与 GPU 运行器批次中的顺序不同，
    # 需要重新排序以匹配。

    # Get the batch indices of the structured output requests.
    # Keep track of the number of speculative tokens scheduled for every
    # request in the batch, as the logit indices are offset by this amount.
    # 获取每个结构化输出请求在批次中的索引。
    # 跟踪每个请求的投机 token 数量，因为 logit 索引会因此偏移。
    #
    # 步骤 2：计算每个请求在 logit 张量中的实际索引。
    # 当启用投机解码时，每个请求会在 logit 维度占用多个位置：
    # - 1 个主 token 位置
    # - N 个投机 token 位置（N = num_speculative_tokens）
    # 因此需要累计偏移量来计算正确的 logit 索引。
    struct_out_req_batch_indices: dict[str, int] = {}
    cumulative_offset = 0
    spec_tokens = scheduler_output.scheduled_spec_decode_tokens
    struct_out_req_ids = set(grammar_output.structured_output_request_ids)
    for batch_index, req_id in enumerate(input_batch.req_ids):
        logit_index = batch_index + cumulative_offset
        cumulative_offset += len(spec_tokens.get(req_id, ()))
        if req_id in struct_out_req_ids:
            struct_out_req_batch_indices[req_id] = logit_index

    out_indices = []

    # Reorder the bitmask to match the order of the requests in the batch.
    # 重新排序位掩码以匹配批次中请求的顺序
    #
    # 步骤 3-4：创建排序后的位掩码并将压缩位掩码展开到正确位置。
    # - 创建与 logit 张量相同大小的位掩码，填充默认值 -1（全 1，允许所有 token）
    # - 遍历压缩位掩码中的每个请求，将其复制到对应的 logit 索引位置
    # - 对于投机解码，每个请求需要复制 1 + num_spec_tokens 行位掩码
    sorted_bitmask = np.full(
        shape=(logits.shape[0], grammar_bitmask.shape[1]),
        fill_value=-1,
        dtype=grammar_bitmask.dtype,
    )
    cumulative_index = 0
    for req_id in grammar_output.structured_output_request_ids:
        num_spec_tokens = len(spec_tokens.get(req_id, ()))
        if (logit_idx := struct_out_req_batch_indices.get(req_id)) is not None:
            # 将每个请求的位掩码（包括投机 token 的位掩码）复制到正确位置
            for i in range(1 + num_spec_tokens):
                bitmask_index = logit_idx + i
                sorted_bitmask[bitmask_index] = grammar_bitmask[cumulative_index + i]
                out_indices.append(bitmask_index)
        cumulative_index += 1 + num_spec_tokens

    # Copy async to device as tensor.
    # 异步将位掩码复制到 GPU 设备
    grammar_bitmask = torch.from_numpy(sorted_bitmask).to(
        logits.device, non_blocking=True
    )

    # If the length of out indices and the logits have the same shape
    # we don't need to pass indices to the kernel,
    # since the bitmask is already aligned with the logits.
    # 如果输出索引数量等于 logit 数量，说明位掩码已完全对齐，无需传递索引
    skip_out_indices = len(out_indices) == logits.shape[0]

    if not logits.is_cpu:
        # GPU 路径：使用异步传输优化
        index_tensor = None
        if not skip_out_indices:
            # xgrammar expects a python list of indices but it will actually work with
            # a tensor. If we copy the tensor ourselves here we can do it in a
            # non_blocking manner and there should be no cpu sync within xgrammar.
            # 将索引列表转为 tensor 并异步传输到 GPU
            pin_memory = is_pin_memory_available()
            index_tensor = torch.tensor(
                out_indices, dtype=torch.int32, device="cpu", pin_memory=pin_memory
            )
            index_tensor = index_tensor.to(logits.device, non_blocking=True)

        # 调用 xgrammar 的原地操作函数，将位掩码应用到 logit 上
        xgr.apply_token_bitmask_inplace(logits, grammar_bitmask, indices=index_tensor)
        return

    # CPU case, use list for indices.
    # CPU 路径：直接使用 Python 列表作为索引
    indices = None if skip_out_indices else out_indices
    # Handle dtype conversion for CPU (older xgrammar CPU kernels require float32)
    # See: https://github.com/vllm-project/vllm/issues/31901
    # CPU 上旧版 xgrammar 内核需要 float32 类型
    if logits.dtype != torch.float32:
        # Convert to float32, apply bitmask, then convert back
        logits_fp32 = logits.to(torch.float32)
        xgr.apply_token_bitmask_inplace(logits_fp32, grammar_bitmask, indices=indices)
        # Copy the modified values back to the original tensor
        logits.copy_(logits_fp32.to(logits.dtype))
    else:
        xgr.apply_token_bitmask_inplace(logits, grammar_bitmask, indices=indices)


class OutlinesVocabulary:
    """
    Wrapper class for `outlines_core.Vocabulary`,
    which allows us to store a hash with the vocabulary

    # Outlines 词汇表的封装类
    # 在原始 Vocabulary 对象基础上增加哈希值缓存，
    # 用于将词汇表作为缓存键使用
    """

    def __init__(self, vocabulary: oc.Vocabulary) -> None:
        # Actual vocabulary object
        self.inner = vocabulary
        # Have to do abs(hash()) because python hashes can
        # be negative, and we are using hash as a cache key.
        # 使用 SHA-256 计算词汇表的哈希值作为缓存键
        # Python 的 hash() 可能为负数，所以使用 SHA-256 确保正值
        hex_str = hashlib.sha256(vocabulary.__repr__().encode("utf-8")).hexdigest()
        hash_int = int(hex_str, 16)
        self._hash = hash_int


def get_outlines_cache_path() -> str:
    """Get the context object that contains previously-computed return values

    # 获取 outlines 缓存目录路径
    # 按优先级查找：
    # 1. OUTLINES_CACHE_DIR 环境变量
    # 2. XDG_CACHE_HOME 环境变量下的 .cache/outlines
    # 3. 用户主目录下的 ~/.cache/outlines
    # 4. 临时目录下的 .cache/outlines（容器环境下回退）
    """
    outlines_cache_dir = os.getenv("OUTLINES_CACHE_DIR")
    xdg_cache_home = os.getenv("XDG_CACHE_HOME")
    home_dir = os.path.expanduser("~")

    if outlines_cache_dir:
        # OUTLINES_CACHE_DIR takes precedence
        return outlines_cache_dir
    if xdg_cache_home:
        return os.path.join(xdg_cache_home, ".cache", "outlines")
    # If homedir is "/", we may be inside a container, and thus writing to
    # root would be problematic, so we fall back to using a tempfile.
    # Also validate the path exists, since os.path.expanduser does
    # not guarantee existence.
    if os.path.isdir(home_dir) and home_dir != "/":
        # Default Unix fallback: ~/.cache/outlines
        return os.path.join(home_dir, ".cache", "outlines")

    # home_dir may be / inside a docker container without existing user
    tempdir = tempfile.gettempdir()
    return os.path.join(tempdir, ".cache", "outlines")


def get_outlines_cache():
    """Get the Cache instance to be used for index caching

    # 获取 outlines 索引缓存实例
    # 根据配置返回不同类型的缓存：
    # - 如果启用了 VLLM_V1_USE_OUTLINES_CACHE，使用 diskcache（无界磁盘缓存）
    # - 否则使用内存中的 LRU 缓存（最多 128 条）
    """

    cache_dir = get_outlines_cache_path()
    if envs.VLLM_V1_USE_OUTLINES_CACHE:
        from diskcache import Cache

        logger.warning(
            "Enabling outlines cache. This is an unbounded on-disk "
            "cache. It may consume a lot of disk space and should "
            "not be used with untrusted clients."
        )
        cache = Cache(cache_dir, eviction_policy="none", cull_limit=0)
        outlines_version = importlib.metadata.version("outlines_core")

        # 如果版本不匹配，清除缓存以避免兼容性问题
        cached_version = cache.get("__version__", None)
        if cached_version != outlines_version:
            cache.clear()
        cache.set("__version__", outlines_version)
        return cache

    return LRUCache(maxsize=128)


# 匹配 Llama 风格的字节 token，如 <0x48>
re_llama_byte_token = re.compile(r"^<0x[0-9A-F]{2}>$")
# 匹配包含 Unicode 替换字符的序列
re_replacement_seq = re.compile(r"^.{0,6}�+.{0,6}$")


def _reduced_vocabulary(tokenizer: TokenizerLike) -> dict[bytes, list[int]]:
    """Create a map from vocabulary tokens to lists of equivalent token ids.

    # 将分词器的词汇表简化为字节到 token ID 列表的映射。
    #
    # 处理逻辑：
    # 1. 遍历分词器的所有非特殊 token
    # 2. 将每个 token 转换为字符串
    # 3. 处理各种特殊情况：
    #    - BPE 分词器中以 bytes 存储的 token
    #    - 包含无效 UTF-8 序列的 token（Llama 风格 <0xXX> 和 GPT2 风格）
    # 4. 将有效 token 编码为 UTF-8 字节并建立映射
    #
    # Returns:
    #     A Dict of token bytes -> equivalent token ids
    """
    eos_token_id = tokenizer.eos_token_id

    # 构建 Unicode 字符到字节的反向映射（用于 GPT2 风格分词器）
    unicode_to_bytes = {
        v: k for k, v in convert_slow_tokenizer.bytes_to_unicode().items()
    }

    def convert_token_to_string(token: str) -> str:
        """将 token 转换为对应的字符串表示。"""
        string = tokenizer.convert_tokens_to_string([token])

        # A hack to handle missing spaces to HF's Llama tokenizers
        # 处理 Llama 分词器缺少空格的问题
        if (
            type(token) is str
            and token.startswith(file_utils.SPIECE_UNDERLINE)
            or token == "<0x20>"
        ):
            return " " + string

        return string

    vocabulary: dict[bytes, list[int]] = {}
    empty_token_ids: list[int] = []
    for token, token_idx in tokenizer.get_vocab().items():
        # 跳过特殊 token（如 [CLS], [SEP], <eos> 等）
        if token in tokenizer.all_special_tokens:
            continue

        token_str = convert_token_to_string(token)
        if token_str:
            if isinstance(token, (bytes, bytearray)):
                # For BPE tokenizers where tokens are stored as bytes.
                # BPE 分词器中 token 以 bytes 形式存储

                # safe to ignore since token_str is of type (bytearray, bytes)
                # by this point.
                token_bytes = bytes(token_str)  # type: ignore[arg-type]

            elif (token_str == "�" and token != "�") or (
                "�" in token_str and not re_replacement_seq.match(token_str)
            ):
                # Handle tokens with invalid UTF-8 sequences.
                # 处理包含无效 UTF-8 序列的 token。
                # 当 token 无法正确解码为 UTF-8 时，需要特殊处理：
                if re_llama_byte_token.match(token):
                    # Llama-like tokenizers use <0xXX> for incomplete sequences.
                    # Llama 风格分词器使用 <0xXX> 格式表示单字节。
                    # 例如 <0x48> 表示字节 0x48（字符 'H'）
                    token_bytes = bytes([int(token[3:5], 16)])
                else:
                    # GPT2 tokenizers: map each byte back using unicode_to_bytes
                    # GPT2 分词器：通过 unicode_to_bytes 映射将每个字符转回字节。
                    # GPT2 使用一种特殊的 Unicode 映射来表示字节序列，
                    # 这里需要反向映射回原始字节。
                    byte_vals = [unicode_to_bytes.get(c) for c in token]
                    if None in byte_vals:
                        raise RuntimeError(
                            f"Cannot convert token `{token}`"
                            f" ({token_idx}) to bytes: {token_str}"
                        )
                    # safe to ignore, since if None in byte_vals,
                    # an error is thrown.
                    token_bytes = bytes(byte_vals)  # type: ignore[arg-type]
            else:
                # 正常的 UTF-8 token，直接编码为字节
                token_bytes = token_str.encode("utf-8")

            if token_idx != eos_token_id:
                # 将字节映射到 token ID 列表。
                # 多个 token 可能映射到相同的字节序列（同义 token），
                # 因此使用列表存储所有对应的 token ID。
                vocabulary.setdefault(token_bytes, []).append(token_idx)
        else:
            # 空字符串 token 单独收集
            empty_token_ids.append(token_idx)

    return vocabulary


def get_outlines_vocabulary(tokenizer: TokenizerLike) -> oc.Vocabulary:
    """Get the `Vocabulary` object for a given tokenizer.

    # 获取分词器对应的 Outlines Vocabulary 对象。
    # 使用缓存机制，每个分词器只创建一次 Vocabulary 对象。
    """
    if hasattr(tokenizer, "_outlines_vocabulary"):
        return tokenizer._outlines_vocabulary  # type: ignore

    reduced_vocab = _reduced_vocabulary(tokenizer)
    vocabulary = OutlinesVocabulary(
        oc.Vocabulary(tokenizer.eos_token_id, reduced_vocab)
    )
    # 将创建的词汇表缓存到分词器对象上
    tokenizer._outlines_vocabulary = vocabulary  # type: ignore

    return vocabulary


def grammar_is_likely_lark(grammar_str: str) -> bool:
    """Check if grammar appears to use Lark syntax.

    # 检测语法字符串是否使用 Lark 格式。
    #
    # 检测方法：
    # - 遍历每一行，跳过注释和空行
    # - 如果找到 EBNF 风格的 "::=" 操作符，返回 False
    # - 否则认为是 Lark 格式
    #
    # Lark 格式示例：rule: 'abc'
    # EBNF 格式示例：rule ::= 'abc'

    Args:
        grammar_str: Input grammar string

    Returns:
        bool: True if grammar appears to be in Lark format, False otherwise

    Examples:
        >>> grammar_is_likely_lark("rule: 'abc'")
        True
        >>> grammar_is_likely_lark("rule ::= 'abc'")
        False
    """
    if not grammar_str or not isinstance(grammar_str, str):
        return False

    for line in grammar_str.split("\n"):
        # Remove both comment styles
        # 移除 # 和 // 两种注释风格
        line = re.sub(r"(#|//).*$", "", line).strip()
        if not line:
            continue

        # Look for EBNF rule definition
        if "::=" in line:
            return False

    return True


def convert_lark_to_ebnf(grammar_str: str) -> str:
    """Convert a Lark grammar string to EBNF format.

    # 将 Lark 格式的语法转换为 EBNF（扩展巴科斯-瑙尔范式）格式。
    #
    # 转换流程：
    # 1. 第一遍扫描：识别所有规则定义，确定根规则
    #    - 第一个定义的规则默认为根规则
    #    - 如果存在名为 "start" 的规则，则将其作为根规则
    # 2. 添加根规则：root ::= <first_rule>
    # 3. 第二遍扫描：处理规则定义和备选项
    #    - 将单引号字符串转换为双引号
    #    - 提取规则引用
    #    - 用 "|" 连接备选项
    # 4. 验证所有引用的规则都已定义
    #
    # EBNF 参考: https://github.com/ggerganov/llama.cpp/blob/master/grammars/README.md
    # Lark 语法参考: https://lark-parser.readthedocs.io/en/latest/grammar.html

    Args:
        grammar_str: Input grammar in Lark format

    Returns:
        str: Converted grammar in EBNF format

    Examples:
        >>> print(convert_lark_to_ebnf("rule: 'hello'"))
        root ::= rule
        rule ::= "hello"
    """
    if not isinstance(grammar_str, str):
        raise ValueError(f"Grammar must be a string, got {type(grammar_str)}")
    if not grammar_str.strip():
        raise ValueError("Grammar string cannot be empty")

    # 已定义的规则集合
    defined_rules = set()
    # 被引用的规则集合
    referenced_rules = set()
    output_lines = []

    def clean_line(line: str) -> str:
        """Remove comments and whitespace from line.
        # 移除注释和首尾空白
        """
        return re.sub(r"(#|//).*$", "", line).strip()

    def check_quotes(text: str, rule_name: str, line_num: int) -> None:
        """Validate quote matching in text.
        # 验证文本中引号是否匹配
        """
        if text.count("'") % 2 != 0 or text.count('"') % 2 != 0:
            raise ValueError(f"Mismatched quotes in {rule_name} on line {line_num}")

    def extract_references(text: str) -> set[str]:
        """Extract rule references from text.
        # 从文本中提取规则引用（排除引号内的字符串和特殊字符）
        """
        # Remove quoted strings and special characters
        text = re.sub(r'"[^"]*"', "", text)
        text = re.sub(r"[+*?()|\[\]{}]", " ", text)
        return set(re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", text))

    # First pass: Find root rule and validate rule definitions
    # 第一遍扫描：查找根规则并验证规则定义
    lines = [clean_line(line) for line in grammar_str.split("\n")]
    first_rule = None

    for line_num, line in enumerate(lines, 1):
        if not line or line.startswith("|"):
            continue

        if ":" in line:
            try:
                name = line.split(":", 1)[0].strip().strip("?")
                defined_rules.add(name)
                if first_rule is None:
                    first_rule = name
                # 如果存在 "start" 规则，优先作为根规则
                if name == "start":
                    first_rule = "start"
            except IndexError as e:
                raise ValueError(
                    f"Invalid rule format on line {line_num}. "
                    "Expected 'rule_name: definition'"
                ) from e

    if not defined_rules:
        raise ValueError("No valid rules found in grammar")

    # Add root rule
    # 添加根规则
    output_lines.append(f"root ::= {first_rule}")

    # Second pass: Process rule definitions and alternatives
    # 第二遍扫描：处理规则定义和备选项
    current_rule = None
    current_definition = []

    for line_num, line in enumerate(lines, 1):
        if not line:
            continue

        try:
            if ":" in line and not line.startswith("|"):
                # Save previous rule if exists
                # 保存前一个规则（如果存在）
                if current_rule:
                    output_lines.append(
                        f"{current_rule} ::= {' | '.join(current_definition)}"
                    )

                # Process new rule
                # 处理新规则
                name, definition = line.split(":", 1)
                current_rule = name.strip().strip("?")

                check_quotes(definition, f"rule '{current_rule}'", line_num)
                # 将单引号字符串转为双引号
                definition = re.sub(r"'([^']*)'", r'"\1"', definition)
                referenced_rules.update(extract_references(definition))
                current_definition = [definition.strip()]

            elif line.startswith("|"):
                # 处理备选项（以 | 开头的行）
                if not current_rule:
                    raise ValueError(
                        f"Alternative '|' on line {line_num} "
                        "without a preceding rule definition"
                    )

                alt_def = line[1:].strip()
                check_quotes(
                    alt_def, f"alternative for rule '{current_rule}'", line_num
                )
                alt_def = re.sub(r"'([^']*)'", r'"\1"', alt_def)
                referenced_rules.update(extract_references(alt_def))
                current_definition.append(alt_def)

        except ValueError as e:
            raise ValueError(f"Error on line {line_num}: {str(e)}") from e

    # Add final rule if exists
    # 添加最后一个规则
    if current_rule:
        output_lines.append(f"{current_rule} ::= {' | '.join(current_definition)}")

    # Validate all rules are defined
    # 验证所有引用的规则都已定义
    undefined_rules = referenced_rules - defined_rules - {"root"}
    if undefined_rules:
        raise ValueError(
            f"Referenced rules are not defined: {', '.join(sorted(undefined_rules))}"
        )

    return "\n".join(output_lines)


def choice_as_grammar(choice: list[str]) -> str:
    """将选项列表转换为 EBNF 语法。

    # 将字符串选项列表转换为 EBNF 格式的语法。
    # 例如：["yes", "no"] -> 'root ::= "yes" | "no"'
    """
    def escape_ebnf_string(s: str) -> str:
        """Escape special characters in a EBNF string.
        # 转义 EBNF 字符串中的特殊字符（双引号和反斜杠）
        """
        # Escape double quotes and backslashes
        return re.sub(r'(["\\])', r"\\\1", s)

    escaped_choices = (escape_ebnf_string(c) for c in choice)
    grammar = "root ::= " + " | ".join(f'"{c}"' for c in escaped_choices)
    return grammar
