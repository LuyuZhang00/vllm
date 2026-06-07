# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
禁用词过滤模块 (Bad Words Filtering)

本模块实现了基于 token 序列的禁用词过滤功能，用于阻止模型生成
指定的不安全、不当或被禁止的文本内容。

工作原理：
1. 用户通过 API 指定一组"禁用词"（bad_words），每个禁用词是
   一个 token ID 序列，例如 [101, 2003, 102] 对应 "is" 这个词。
2. 在每一步生成时，检查当前已生成的 token 序列是否匹配某个
   禁用词的前缀。
3. 如果匹配到了禁用词的前 n-1 个 token，则将第 n 个 token 的
   logits 设为负无穷，阻止其被采样到。

示例：
  禁用词 = [[10, 20, 30]]  （即 token 序列 [10, 20, 30] 被禁止）
  已生成序列 = [..., 10, 20]
  → 匹配到前缀 [10, 20]，将 token 30 的 logits 设为 -inf

两种使用场景：
1. 普通生成 (apply_bad_words): 每个 batch 元素独立处理
2. 投机解码 (apply_bad_words_with_drafts): 需要处理草稿 token 的额外行
"""

import torch

# 最小的 logit 值，用于屏蔽被禁用的 token
_SMALLEST_LOGIT = float("-inf")


def _apply_bad_words_single_batch(
    logits: torch.Tensor,
    bad_words_token_ids: list[list[int]],
    past_tokens_ids: list[int],
) -> None:
    """
    对单个 batch 元素应用禁用词过滤。

    遍历所有禁用词，检查已生成序列是否匹配某个禁用词的前缀。
    如果匹配，则将该禁用词最后一个 token 的 logits 设为 -inf。

    Args:
        logits: [vocab_size] 的 logits 张量（单个 batch 元素）
        bad_words_token_ids: 禁用词列表，每个元素是一个 token ID 序列
            例如 [[10, 20, 30], [5, 6]] 表示禁止生成 [10,20,30] 和 [5,6]
        past_tokens_ids: 当前已生成的 token ID 序列（不含 prompt）
    """
    for bad_word_ids in bad_words_token_ids:
        # 如果禁用词比已生成序列还长，跳过（不可能匹配）
        if len(bad_word_ids) > len(past_tokens_ids) + 1:
            continue

        # 计算前缀长度（禁用词长度减 1）
        prefix_length = len(bad_word_ids) - 1
        # 获取禁用词的最后一个 token（需要被屏蔽的 token）
        last_token_id = bad_word_ids[-1]
        # 获取已生成序列末尾的 prefix_length 个 token 作为实际前缀
        actual_prefix = past_tokens_ids[-prefix_length:] if prefix_length > 0 else []
        # 获取禁用词的前 prefix_length 个 token 作为期望前缀
        expected_prefix = bad_word_ids[:prefix_length]

        assert len(actual_prefix) == len(expected_prefix)

        # 如果实际前缀匹配期望前缀，屏蔽最后一个 token
        if actual_prefix == expected_prefix:
            # 使用切片赋值避免 CPU->GPU 同步
            logits[last_token_id : last_token_id + 1] = _SMALLEST_LOGIT


def apply_bad_words(
    logits: torch.Tensor,
    bad_words_token_ids: dict[int, list[list[int]]],
    past_tokens_ids: list[list[int]],
) -> None:
    """
    对整个 batch 应用禁用词过滤。

    Args:
        logits: [batch_size, vocab_size] 的 logits 张量
        bad_words_token_ids: 字典，键为 batch 索引，值为该请求的禁用词列表
            例如 {0: [[10, 20, 30]], 2: [[5, 6]]}
            表示 batch[0] 禁止 [10,20,30]，batch[2] 禁止 [5,6]
        past_tokens_ids: 列表，每个元素对应一个 batch 的已生成 token 序列
    """
    for i, bad_words_ids in bad_words_token_ids.items():
        _apply_bad_words_single_batch(logits[i], bad_words_ids, past_tokens_ids[i])


def apply_bad_words_with_drafts(
    logits: torch.Tensor,
    bad_words_token_ids: dict[int, list[list[int]]],
    past_tokens_ids: list[list[int]],
    num_draft_tokens: list[int],
) -> None:
    """
    在投机解码（speculative decoding）场景下应用禁用词过滤。

    投机解码中，模型会一次生成多个候选 token（草稿 token），
    因此 logits 张量的行数 = batch_size * (1 + num_draft_tokens)。
    例如 batch_size=2，每个请求有 3 个草稿 token 时，logits 有 8 行：
      - 行 0-3: 请求 0 的原始 + 3 个草稿
      - 行 4-7: 请求 1 的原始 + 3 个草稿

    Args:
        logits: [total_rows, vocab_size] 的 logits 张量
            total_rows = sum(1 + num_draft_tokens[i] for all i)
        bad_words_token_ids: 字典，键为 batch 索引，值为禁用词列表
        past_tokens_ids: 列表，每个元素对应一行的已生成 token 序列
        num_draft_tokens: 列表，每个元素为对应 batch 的草稿 token 数量
    """
    start_idx = 0
    remaining = len(bad_words_token_ids)
    for i, n in enumerate(num_draft_tokens):
        if (bad_words_ids := bad_words_token_ids.get(i)) is not None:
            # 对该请求的所有行（原始 + 草稿）应用禁用词过滤
            for draft_idx in range(start_idx, start_idx + n):
                _apply_bad_words_single_batch(
                    logits[draft_idx],
                    bad_words_ids,
                    past_tokens_ids[draft_idx],
                )
            remaining -= 1
            if not remaining:
                break
        start_idx += n
