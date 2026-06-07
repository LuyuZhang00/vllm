# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
GPU-accelerated N-gram proposer using fully async PyTorch tensor operations.

This version uses a fully vectorized approach with unfold and argmax for
finding the first match across all sequences in parallel.
"""

# 中文注释：本模块实现了基于 GPU 加速的 N-gram 推测解码（speculative decoding）proposer。
#
# 【模块功能概述】
# 本模块是 vLLM V1 推测解码框架中的 N-gram proposer 实现。与传统的基于 CPU 的 n-gram
# 匹配不同，本模块将所有核心计算（滑动窗口匹配、候选 token 提取）完全放在 GPU 上执行，
# 避免了 CPU-GPU 同步开销，适合高吞吐场景。
#
# 【核心算法流程】
# 1. 取当前序列末尾长度为 n 的 token 子串（suffix n-gram）作为搜索模式
# 2. 在该序列的历史 token 中，用滑动窗口（unfold）找到第一个匹配位置
# 3. 匹配位置之后的 k 个 token 即为 draft token 候选
# 4. 如果多个 n-gram 长度都有匹配，选择最长的 n-gram 以获得更准确的预测
# 5. 无效位置用 -1 标记，最终统计每个请求的有效 draft token 数量
#
# 【关键类】
# - NgramGPUKernel: 核心 GPU 计算内核，支持 torch.compile 优化
# - NgramProposerGPU: 封装层，负责输入准备（将新采样 token 写入 token_ids 缓冲区）
#   和调用 NgramGPUKernel
# - update_scheduler_for_invalid_drafts: 根据 GPU 计算结果，修剪 scheduler 中
#   超出有效范围的 draft token
# - update_ngram_gpu_tensors_incremental: 增量更新 GPU 上的 token_ids 和序列长度
#   张量，处理新增/删除/重排序请求
#
# 【在推测解码链路中的位置】
# Scheduler 调度请求 -> Model Runner 执行一步推理，采样得到 token ->
# NgramProposerGPU.propose() 基于历史 token 提出 draft token ->
# Model Runner 对 draft token 做验证（verify） ->
# 接受正确的 draft token，拒绝错误的

import torch
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import (
    CompilationConfig,
    CompilationMode,
    CUDAGraphMode,
    VllmConfig,
)
from vllm.forward_context import set_forward_context
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.utils import record_function_or_nullcontext
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch


@support_torch_compile()
class NgramGPUKernel(nn.Module):
    """GPU-accelerated N-gram proposer using fully async tensor operations."""

    # 中文注释：NgramGPUKernel 是推测解码中 N-gram 提案的核心 GPU 计算内核。
    # 它继承 nn.Module 以便支持 torch.compile 优化编译。
    # 所有计算（滑动窗口展开、匹配比较、候选提取）都在 GPU tensor 上完成，
    # 不涉及任何 CPU 同步操作，从而实现完全异步的推测 token 提案。
    #
    # 【设计要点】
    # - 使用 torch.unfold 生成滑动窗口视图（O(1) 内存），避免显式循环
    # - 通过向量化 argmax 找到最早匹配位置，避免数据依赖分支
    # - 对多个 n-gram 长度逐一尝试，选择最长的有效匹配以提高预测准确性
    # - 支持 torch.compile 编译优化，减少 Python 开销

    def __init__(
        self, vllm_config: VllmConfig, prefix: str = "", device: torch.device = "cuda"
    ):
        # 中文注释：初始化 NgramGPUKernel。
        # 从 vllm_config 中读取推测解码相关的超参数，并分配必要的 buffer。
        super().__init__()

        assert vllm_config.speculative_config is not None
        assert vllm_config.speculative_config.prompt_lookup_min is not None
        assert vllm_config.speculative_config.prompt_lookup_max is not None

        # 中文注释：min_n / max_n 定义了搜索 n-gram 时尝试的长度范围。
        # 例如 min_n=2, max_n=5 表示依次尝试长度为 2, 3, 4, 5 的后缀 n-gram，
        # 选择能匹配到的最长 n-gram 对应的后续 token 作为 draft 候选。
        self.min_n = vllm_config.speculative_config.prompt_lookup_min
        self.max_n = vllm_config.speculative_config.prompt_lookup_max
        # 中文注释：k 是每个请求要提出的 draft token 数量。
        # 推测解码会尝试验证这 k 个 token，全部接受则一步生成 k+1 个 token。
        self.k = vllm_config.speculative_config.num_speculative_tokens
        self.max_model_len = vllm_config.model_config.max_model_len
        self.max_num_seqs = vllm_config.scheduler_config.max_num_seqs
        self.device = device

    def _find_first_and_extract_all_n_parallel(
        self,
        token_ids: torch.Tensor,
        seq_lengths: torch.Tensor,
        min_ngram_len: int,
        max_ngram_len: int,
        num_draft_tokens: int,
    ) -> torch.Tensor:
        # 中文注释：核心匹配方法——在所有序列上并行查找后缀 n-gram 的历史匹配，
        # 并提取匹配位置之后的 token 作为 draft 候选。
        #
        # 【算法流程】
        # 1. 对每个 n-gram 长度 n (从 min_n 到 max_n)：
        #    a. 用 torch.unfold 将 token_ids 展开为滑动窗口 [batch, num_windows, n]（O(1) 视图）
        #    b. 提取每个序列末尾的 n 个 token 作为搜索模式（suffix）
        #    c. 比较所有滑动窗口与 suffix 是否完全匹配，得到 [batch, num_windows] 的 bool 矩阵
        #    d. 用 argmax 找到每个序列中最早匹配的窗口位置
        #    e. 匹配位置必须留有至少 1 个后续 token（否则无法提取 draft）
        # 2. 在所有 n-gram 长度中选择最长的有效匹配（更长的上下文 -> 更准确的预测）
        # 3. 从匹配位置之后提取 k 个 token 作为 draft，超出序列长度的位置标记为 -1
        #
        # 【返回值】
        # 返回 [batch_size, k] 的 tensor，值为 token ID 或 -1（无效/无匹配）。
        """
        Find suffix n-gram matches and extract following tokens.
        Searches for the earliest prior occurrence of the trailing n-gram,
        tries multiple lengths, and picks the longest valid match.

        Args:
            token_ids: Token IDs for each sequence
            seq_lengths: Actual length of each sequence (excluding padding)
            min_ngram_len: Minimum n-gram size to search for (e.g., 2)
            max_ngram_len: Maximum n-gram size to search for (e.g., 5)
            num_draft_tokens: Number of tokens to extract after match (k)

        Returns:
            Draft token predictions; -1 means invalid/no match.
        """
        batch_size = token_ids.shape[0]
        max_seq_len = token_ids.shape[1]
        device = token_ids.device
        # 中文注释：需要尝试的 n-gram 长度种类数（例如 min_n=2, max_n=5 则有 4 种）。
        num_ngram_sizes = max_ngram_len - min_ngram_len + 1

        # All n-gram sizes to try.
        ngram_lengths = torch.arange(min_ngram_len, max_ngram_len + 1, device=device)
        # 中文注释：batch 维度的索引，用于后续 advanced indexing。
        batch_indices = torch.arange(batch_size, device=device)

        # Earliest match per (sequence, ngram_len); -1 means no match.
        # 中文注释：记录每个序列在每种 n-gram 长度下的最早匹配位置。
        # 值为 -1 表示该长度下未找到匹配。
        first_match_positions = torch.full(
            (batch_size, num_ngram_sizes), -1, dtype=torch.long, device=device
        )

        # 中文注释：遍历每种 n-gram 长度，在所有序列上并行执行滑动窗口匹配。
        for i, ngram_len in enumerate(range(min_ngram_len, max_ngram_len + 1)):
            # Sliding windows of size ngram_len; unfold is O(1) view.
            # 中文注释：使用 torch.unfold 生成滑动窗口视图 [batch, num_windows, ngram_len]。
            # unfold 返回的是原始 tensor 的视图，不额外分配内存（O(1)）。
            search_windows = token_ids.unfold(1, ngram_len, 1)
            num_windows = search_windows.shape[1]

            # Trailing suffix (last ngram_len tokens) for each sequence.
            # 中文注释：提取每个序列末尾的 ngram_len 个 token 作为搜索模式（suffix）。
            # 这就是要在历史 token 中查找的模式。
            suffix_starts = seq_lengths - ngram_len
            suffix_indices = suffix_starts.unsqueeze(1) + torch.arange(
                ngram_len, device=device
            )
            suffix = torch.gather(token_ids, 1, suffix_indices.clamp(min=0))

            # Window matches for each sequence.
            # 中文注释：比较所有滑动窗口与 suffix 是否完全匹配。
            # 结果为 [batch, num_windows] 的 bool tensor。
            matches = (search_windows == suffix.unsqueeze(1)).all(dim=-1)

            # Match must leave room for at least one draft token.
            # 中文注释：匹配位置之后必须至少还有 1 个 token 才能作为 draft 候选。
            # 因此只保留位置 <= seq_len - ngram_len - 1 的匹配。
            max_valid_suffix_start = seq_lengths - ngram_len - 1
            window_positions = torch.arange(num_windows, device=device)
            valid_mask = window_positions <= max_valid_suffix_start.unsqueeze(1)
            final_matches = matches & valid_mask

            # Find earliest match (argmax=0 when empty; verify with has_match).
            # 中文注释：用 argmax 找到每个序列中最早（最靠前）的匹配位置。
            # 注意：当无匹配时 argmax 返回 0，需要通过 has_match 来判断是否真有匹配。
            first_match_idx = torch.argmax(final_matches.int(), dim=1)
            has_match = final_matches[batch_indices, first_match_idx]

            # Store valid match positions (window index = position).
            # 中文注释：将匹配位置存入 first_match_positions；无匹配则为 -1。
            first_match_positions[:, i] = torch.where(has_match, first_match_idx, -1)

        # Select the longest n-gram with a match.
        # 中文注释：从所有 n-gram 长度中选择最长的有效匹配。
        # 技巧：先 flip 再 argmax 可以在向量化条件下选择"最后一个 True"（即最长 n-gram）。
        best_ngram_idx = (first_match_positions >= 0).int().flip(dims=[1]).argmax(dim=1)
        best_ngram_idx = num_ngram_sizes - 1 - best_ngram_idx  # Flip back

        # Match position for the best n-gram.
        # 中文注释：获取每个序列所选最佳 n-gram 长度对应的匹配位置。
        best_match_pos = first_match_positions[batch_indices, best_ngram_idx]

        # Avoid data-dependent branching.
        # 中文注释：记录每个序列是否有任何长度的匹配（避免数据依赖分支，利于 GPU 执行）。
        has_any_match = best_match_pos >= 0

        # Length of the best matching n-gram.
        best_ngram_lengths = ngram_lengths[best_ngram_idx]

        # Start position right after the matched suffix.
        # 中文注释：draft token 的起始位置 = 匹配位置 + 匹配的 n-gram 长度。
        # 即从历史中匹配到的后缀之后的 token 开始提取。
        draft_start = torch.where(
            has_any_match,
            best_match_pos + best_ngram_lengths,
            torch.zeros_like(best_match_pos),
        )
        # 中文注释：每个序列在 draft_start 之后还剩余多少 token 可供提取。
        tokens_available = seq_lengths - draft_start

        # Gather indices for draft tokens.
        # 中文注释：构造 draft token 的索引 [batch, k]，从 draft_start 开始连续取 k 个位置。
        draft_indices = draft_start.unsqueeze(1) + torch.arange(
            num_draft_tokens, device=device
        )
        draft_indices = draft_indices.clamp(min=0, max=max_seq_len - 1)

        # Extract draft tokens; gather always runs.
        # 中文注释：用 gather 从 token_ids 中提取 draft token（始终执行，通过后续 mask 处理无效位置）。
        draft_tokens = torch.gather(token_ids, 1, draft_indices)

        # Mask positions beyond available tokens.
        # 中文注释：如果某个位置超出了序列实际长度，标记为 -1（无效）。
        position_indices = torch.arange(num_draft_tokens, device=device).unsqueeze(0)
        valid_positions = position_indices < tokens_available.unsqueeze(1)

        draft_tokens = torch.where(
            valid_positions,
            draft_tokens,
            torch.full_like(draft_tokens, -1),
        )

        # If no match, mask all positions.
        # 中文注释：如果该序列完全没有匹配，所有 draft token 都标记为 -1。
        draft_tokens = torch.where(
            has_any_match.unsqueeze(1),
            draft_tokens,
            torch.full_like(draft_tokens, -1),
        )

        return draft_tokens

    def forward(
        self,
        num_tokens_no_spec: torch.Tensor,
        token_ids_gpu: torch.Tensor,
        combined_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 中文注释：NgramGPUKernel 的前向传播入口（被 torch.compile 优化）。
        #
        # 【流程】
        # 1. 分配 draft_tokens 输出 buffer（值为 -1 表示无效）
        # 2. 调用 _find_first_and_extract_all_n_parallel 执行核心匹配逻辑
        # 3. 用 combined_mask 过滤掉不需要推测解码的序列
        # 4. 统计每个序列中前导连续有效 draft token 数量（num_valid_draft_tokens）
        #
        # 【num_valid_draft_tokens 的计算方式】
        # 利用 cumsum 技巧：对 is_valid 做前缀和，如果位置 i 处的前缀和 == i+1，
        # 说明前 i+1 个 token 都是有效的。对所有满足条件的位置计数即可得到前导连续有效数。
        """
        Forward pass for N-gram proposal using GPU tensor operations.

        Args:
            num_tokens_no_spec: Number of tokens for each sequence [batch_size]
            token_ids_gpu: Token IDs [batch_size, max_len]
            combined_mask: Whether each sequence is valid for spec decode [batch_size]

        Returns:
            draft_tokens: [batch_size, k] on GPU
            num_valid_draft_tokens: [batch_size] int32 on GPU, count of
                leading valid (non -1) tokens per request.
        """

        device = token_ids_gpu.device

        # Infer batch size to preserve dynamic shape.
        # 中文注释：从输入 tensor 推断 batch size，以支持动态 batch 大小。
        actual_batch_size = token_ids_gpu.shape[0]

        # Allocate in forward so torch.compile can optimize.
        # NOTE(patchy): Do NOT pre-allocate this as a buffer
        #               it breaks torch.compile
        # 中文注释：在 forward 中分配输出 buffer，初始值全为 -1（无效标记）。
        # 注意：不能预分配为类属性 buffer，否则会破坏 torch.compile 的优化。
        draft_tokens = torch.full(
            (actual_batch_size, self.k), -1, dtype=torch.int32, device=device
        )

        # 中文注释：调用核心匹配算法，获取所有序列的 draft token 候选。
        results = self._find_first_and_extract_all_n_parallel(
            token_ids_gpu,
            num_tokens_no_spec,
            min_ngram_len=self.min_n,
            max_ngram_len=self.max_n,
            num_draft_tokens=self.k,
        )

        # 中文注释：用 combined_mask 过滤掉不需要推测解码的序列（如新请求、被丢弃请求等）。
        draft_tokens = torch.where(combined_mask.unsqueeze(1), results, -1)

        # Count leading contiguous valid (non -1) tokens per request.
        # 中文注释：统计每个请求中前导连续有效 draft token 的数量。
        # 这个数量决定了推测解码验证阶段实际需要验证多少个 token。
        # 【cumsum 技巧】
        # 对 [True, True, False, True] 做 cumsum 得到 [1, 2, 2, 3]，
        # 比较 positions [1, 2, 3, 4]，只有 [True, True, False, False]，
        # sum = 2，即前导连续有效数为 2。
        is_valid = draft_tokens != -1  # [batch, k]
        cum_valid = is_valid.int().cumsum(dim=1)  # [batch, k]
        positions = torch.arange(1, self.k + 1, device=device).unsqueeze(0)
        num_valid_draft_tokens = (cum_valid == positions).int().sum(dim=1)

        return draft_tokens, num_valid_draft_tokens

    def load_model(self, *args, **kwargs):
        """No model to load for N-gram proposer."""
        pass


class NgramProposerGPU:
    # 中文注释：NgramProposerGPU 是 GPU 加速 N-gram proposer 的封装层。
    #
    # 【职责】
    # 1. 管理 NgramGPUKernel 的生命周期（初始化、dummy warmup、设备放置）
    # 2. 在 propose() 中将新采样的 token 写入 token_ids 缓冲区（scatter 操作）
    # 3. 计算临时的序列长度，然后调用 kernel 执行匹配
    # 4. 提供 update_token_ids_ngram() 用于增量更新 token_ids 和序列长度
    #
    # 【与 Scheduler / Model Runner 的交互】
    # - Model Runner 采样得到 token 后，调用 propose() 获取 draft token
    # - draft token 传给 Scheduler 的 scheduled_spec_decode_tokens
    # - 下一步由 Model Runner 对 draft token 执行 verify（验证）

    def __init__(self, vllm_config: VllmConfig, device: torch.device, runner=None):
        assert vllm_config.speculative_config is not None
        assert vllm_config.speculative_config.prompt_lookup_min is not None
        assert vllm_config.speculative_config.prompt_lookup_max is not None

        # 中文注释：为 NgramGPUKernel 创建独立的编译配置。
        # 使用 VLLM_COMPILE 模式 + Inductor 后端，启用 max_autotune 和 aggressive_fusion
        # 以获得最佳性能。禁用 CUDA Graph（cudagraph_mode=NONE）因为 kernel 的 batch size 动态变化。
        compilation_config = CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            custom_ops=["none"],
            splitting_ops=[],
            compile_sizes=[],
            inductor_compile_config={
                "enable_auto_functionalized_v2": False,
                "max_autotune": True,
                "aggressive_fusion": True,
                "triton.autotune_pointwise": True,
                "coordinate_descent_tuning": True,
                "use_mixed_mm": False,
            },
            cudagraph_mode=CUDAGraphMode.NONE,
        )
        model_config = vllm_config.model_config
        speculative_config = vllm_config.speculative_config
        scheduler_config = vllm_config.scheduler_config

        self.vllm_config = VllmConfig(
            compilation_config=compilation_config,
            model_config=model_config,
            speculative_config=speculative_config,
            scheduler_config=scheduler_config,
        )

        self.min_n = vllm_config.speculative_config.prompt_lookup_min
        self.max_n = vllm_config.speculative_config.prompt_lookup_max
        self.k = vllm_config.speculative_config.num_speculative_tokens
        self.max_model_len = vllm_config.model_config.max_model_len
        self.max_num_seqs = vllm_config.scheduler_config.max_num_seqs
        self.device = device

        # 中文注释：创建 NgramGPUKernel 实例并放置到目标设备。
        # kernel.eval() 设置为推理模式（禁用 dropout 等训练行为）。
        self.kernel = NgramGPUKernel(
            vllm_config=self.vllm_config, prefix="ngram_gpu_kernel", device=device
        )
        self.kernel.to(device)
        self.kernel.eval()

        # 中文注释：执行 dummy warmup，触发 torch.compile 的编译和缓存。
        # 这样后续真实推理时不会遇到编译延迟。
        self._dummy_run()

    def _dummy_run(self):
        # 中文注释：用随机数据预热 kernel，触发 torch.compile 编译。
        # 运行 3 次以覆盖 torch.compile 的不同编译阶段（warmup、compile、cache）。
        token_ids, num_tokens, sampled_flags, valid_mask = self._generate_dummy_data(
            batch_size=self.max_num_seqs,
            max_seq_len=self.max_model_len,
            pattern_len=self.k,
            device=self.device,
        )

        combined_mask = sampled_flags & valid_mask & (num_tokens >= self.min_n)

        for _ in range(3):
            with set_forward_context(None, self.vllm_config):
                _, _ = self.kernel(num_tokens, token_ids, combined_mask)

    def _generate_dummy_data(
        self,
        batch_size: int,
        max_seq_len: int,
        pattern_len: int,
        device: str = "cuda",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate random test data with n-gram repetitions.

        Args:
            batch_size: Number of sequences in the batch
            max_seq_len: Maximum sequence length
            pattern_len: Length of patterns to inject for matching
            device: Device to place tensors on

        Returns:
            token_ids: [batch_size, max_seq_len] tensor
            num_tokens: [batch_size] tensor
            sampled_flags: [batch_size] bool tensor
            valid_mask: [batch_size] bool tensor
        """
        token_ids = torch.zeros(
            batch_size,
            max_seq_len,
            dtype=torch.int32,
            device=device,
        )

        num_tokens = torch.randint(
            pattern_len, max_seq_len, (batch_size,), dtype=torch.int32, device=device
        )

        sampled_flags = torch.ones(batch_size, dtype=torch.bool, device=device)
        valid_mask = torch.ones(batch_size, dtype=torch.bool, device=device)

        return token_ids, num_tokens, sampled_flags, valid_mask

    def propose(
        self,
        num_tokens_no_spec: torch.Tensor,  # [batch_size]
        token_ids_gpu: torch.Tensor,  # [batch_size, max_len]
        valid_sampled_token_ids_gpu: torch.Tensor,  # [batch_size, num_spec_tokens + 1]
        valid_sampled_tokens_count: torch.Tensor,  # [batch_size]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 中文注释：N-gram proposer 的主入口方法。
        #
        # 【流程】
        # 1. 将本轮新采样的 token（valid_sampled_token_ids_gpu）scatter 写入
        #    token_ids_gpu 的对应位置（num_tokens_no_spec 偏移处）
        # 2. 计算临时的更新后序列长度 = num_tokens_no_spec + valid_sampled_tokens_count
        # 3. 构造 combined_mask，排除不需要推测解码的序列
        # 4. 调用 kernel.forward() 执行 GPU 上的 n-gram 匹配
        # 5. 返回 draft_tokens [batch, k] 和 num_valid_draft_tokens [batch]
        #
        # 【注意】
        # token_ids_gpu 会被原地修改（scatter_），调用方需要确保这是可接受的。
        """
        Propose draft tokens using GPU-accelerated n-gram matching.

        Scatter sampled tokens into `token_ids_gpu`, compute temporary
        updated lengths, then run the kernel.

        Args:
            num_tokens_no_spec: Number of tokens per sequence (read-only)
            token_ids_gpu: Token IDs tensor (modified in-place with new tokens)
            valid_sampled_token_ids_gpu: Newly sampled tokens to scatter
            valid_sampled_tokens_count: Count of valid tokens per sequence

        Returns:
            draft_tokens: Proposed draft token IDs [batch_size, k]
            num_valid_draft_tokens: Count of leading valid draft tokens
                per request [batch_size]
        """
        assert token_ids_gpu.device == self.device
        assert num_tokens_no_spec.device == self.device

        batch_size = num_tokens_no_spec.shape[0]
        max_seq_len = token_ids_gpu.shape[1]
        max_new_tokens = valid_sampled_token_ids_gpu.shape[1]  # num_spec_tokens + 1

        # Scatter newly sampled tokens into token_ids_gpu.
        # 中文注释：将本轮新采样的 token 写入 token_ids_gpu 的对应位置。
        # 写入位置 = num_tokens_no_spec + offset（offset 为 0, 1, 2, ...）。
        # 使用 scatter_ 原地写入，避免分配新 tensor。
        offsets = torch.arange(max_new_tokens, device=self.device)
        write_positions = num_tokens_no_spec.unsqueeze(1) + offsets.unsqueeze(0)
        # 中文注释：构造写入 mask，三重条件：
        # 1. offset < 该请求的有效 token 数（valid_write_mask）
        # 2. token ID 不为 -1（有效 token）
        # 3. 写入位置不超过 max_seq_len（in_bounds）
        valid_write_mask = offsets.unsqueeze(0) < valid_sampled_tokens_count.unsqueeze(
            1
        )
        in_bounds = write_positions < max_seq_len
        scatter_mask = (
            valid_write_mask & (valid_sampled_token_ids_gpu != -1) & in_bounds
        )

        # 中文注释：对不满足 mask 条件的位置，保留 token_ids_gpu 中的原始值（避免覆盖）。
        write_positions_long = write_positions.clamp(max=max_seq_len - 1).long()
        existing_values = token_ids_gpu.gather(1, write_positions_long)

        tokens_cast = valid_sampled_token_ids_gpu.to(token_ids_gpu.dtype)
        tokens_to_scatter = torch.where(
            scatter_mask,
            tokens_cast,
            existing_values,
        )
        token_ids_gpu.scatter_(1, write_positions_long, tokens_to_scatter)

        # 中文注释：计算临时的更新后序列长度（包含本轮新采样的 token）。
        # 这个长度用于告诉 kernel 搜索 token_ids 的有效范围。
        num_tokens_tmp = (num_tokens_no_spec + valid_sampled_tokens_count).to(
            torch.int32
        )

        # Compute validity masks.
        # 中文注释：构造综合 mask，排除以下序列：
        # 1. 本轮没有采样到有效 token 的序列（sampled_flags）
        # 2. 其他无效条件（valid_mask，目前恒为 True，预留扩展）
        # 3. 序列长度不足 min_n（无法形成最短 n-gram 模式）
        sampled_flags = valid_sampled_tokens_count > 0
        valid_mask = torch.ones(batch_size, dtype=torch.bool, device=self.device)

        # 中文注释：set_forward_context 为 torch.compile 提供必要的前向上下文。
        with set_forward_context(None, self.vllm_config):
            combined_mask = sampled_flags & valid_mask & (num_tokens_tmp >= self.min_n)

            # 中文注释：调用 GPU kernel 执行 n-gram 匹配，获取 draft token 和有效数量。
            with record_function_or_nullcontext("ngram_proposer_gpu: kernel"):
                draft_tokens, num_valid_draft_tokens = self.kernel(
                    num_tokens_tmp,
                    token_ids_gpu,
                    combined_mask,
                )

            return draft_tokens, num_valid_draft_tokens

    def update_token_ids_ngram(
        self,
        sampled_token_ids: torch.Tensor | list[list[int]],
        gpu_input_batch: InputBatch,
        token_ids_gpu: torch.Tensor,
        num_tokens_no_spec: torch.Tensor,
        discard_request_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 中文注释：在设备端准备推测解码的输入——处理新采样 token、被丢弃请求、被拒绝 token。
        #
        # 【流程】
        # 1. 如果 sampled_token_ids 是不规则的 list[list[int]]，填充为统一长度的 tensor
        # 2. 备份每个序列最后一个有效 token（用于回退，当无有效采样 token 时使用）
        # 3. 将被丢弃请求（discard_request_mask）的采样 token 标记为 -1
        # 4. 统计每个请求的有效采样 token 数量
        # 5. 提取每个请求的最后一个有效 token 作为 next_token_id
        #
        # 【返回值】
        # - next_token_ids: 每个请求的下一个 token ID（用于更新 KV cache 等）
        # - valid_sampled_tokens_count: 每个请求的有效采样 token 数
        # - valid_sampled_token_ids_gpu: 处理后的采样 token tensor
        """
        Prepare speculative decoding inputs on device:
        compute next token ids and valid counts, honoring discarded requests
        and rejected tokens, without CPU-GPU sync.
        """
        num_reqs = gpu_input_batch.num_reqs

        # 中文注释：处理不规则的 list[list[int]] 输入（当 disable_padded_drafter_batch=True 时）。
        # 各子列表长度可能不同（被丢弃的请求为空列表），需要填充为统一长度后再转为 tensor。
        if isinstance(sampled_token_ids, list):
            # When disable_padded_drafter_batch=True, sampled_token_ids is
            # an irregular list[list[int]] where sublists may have different
            # lengths (including empty lists for discarded requests).
            # Pad all sublists to the same length with -1 before converting
            # to tensor.
            max_len = max(
                (len(sublist) for sublist in sampled_token_ids),
                default=0,
            )
            # Ensure at least length 1 for tensor creation
            max_len = max(max_len, 1)
            padded_list = [
                sublist + [-1] * (max_len - len(sublist))
                for sublist in sampled_token_ids
            ]
            sampled_token_ids = torch.tensor(
                padded_list, dtype=torch.int32, device=self.device
            )
        assert isinstance(sampled_token_ids, torch.Tensor), (
            "sampled_token_ids should be a torch.Tensor for ngram_gpu"
        )

        # Backup last valid token before speculative tokens.
        # 中文注释：备份每个序列在推测 token 之前的最后一个 token。
        # 当某个请求本轮没有有效采样 token 时，使用这个备份值作为回退。
        backup_indices = (num_tokens_no_spec[:num_reqs] - 1).clamp(min=0).long()
        backup_next_token_ids = torch.gather(
            token_ids_gpu[:num_reqs], dim=1, index=backup_indices.unsqueeze(1)
        ).squeeze(1)

        valid_sampled_token_ids_gpu = sampled_token_ids.clone()
        # Invalidate sampled tokens for discarded requests.
        # 中文注释：将被丢弃请求的所有采样 token 标记为 -1（无效）。
        discard_mask_expanded = discard_request_mask[:num_reqs].unsqueeze(1)
        valid_sampled_token_ids_gpu.masked_fill_(discard_mask_expanded, -1)

        # Mask valid tokens within each request.
        # 中文注释：构建有效 token mask，排除 -1（无效）和超出词表范围的 token。
        valid_mask = (valid_sampled_token_ids_gpu != -1) & (
            valid_sampled_token_ids_gpu < gpu_input_batch.vocab_size
        )

        # Count valid tokens per request.
        # 中文注释：统计每个请求中有效采样 token 的数量。
        valid_sampled_tokens_count = valid_mask.sum(dim=1).to(torch.int32)

        # Rightmost valid index per row.
        # 中文注释：找到每行最右侧有效 token 的索引（用于提取最后一个有效 token）。
        last_valid_indices = valid_sampled_tokens_count - 1
        last_valid_indices_safe = torch.clamp(last_valid_indices, min=0)

        # Last valid token from each row; undefined if none.
        selected_tokens = torch.gather(
            valid_sampled_token_ids_gpu, 1, last_valid_indices_safe.unsqueeze(1)
        ).squeeze(1)

        # Use last token if valid; otherwise fallback to backup.
        # 中文注释：如果该请求有有效采样 token，使用最后一个；否则回退到备份值。
        next_token_ids = torch.where(
            last_valid_indices != -1,
            selected_tokens,
            backup_next_token_ids,
        )

        return next_token_ids, valid_sampled_tokens_count, valid_sampled_token_ids_gpu

    def load_model(self, *args, **kwargs):
        self.kernel.load_model(*args, **kwargs)


def update_scheduler_for_invalid_drafts(
    num_valid_draft_tokens_event: torch.cuda.Event,
    num_valid_draft_tokens_cpu: torch.Tensor,
    scheduler_output: "SchedulerOutput",
    req_id_to_index: dict[str, int],
) -> None:
    """Trim invalid speculative slots using per-request valid draft counts.

    Args:
        num_valid_draft_tokens_event: Event for async D2H completion.
        num_valid_draft_tokens_cpu: CPU buffer of valid draft counts.
        scheduler_output: Scheduler metadata to update in-place.
        req_id_to_index: Request-id to batch-index mapping.
    """
    # 中文注释：根据 GPU 计算出的每个请求有效 draft token 数，修剪 SchedulerOutput 中
    # 超出有效范围的 draft token。
    #
    # 【背景】
    # N-gram proposer 可能为某些请求提出少于 k 个有效 draft token（如历史中只有短匹配）。
    # 但 Scheduler 已经按 k 个 draft token 分配了 KV cache block 和调度名额。
    # 此函数需要：
    # 1. 同步等待 GPU->CPU 的 num_valid_draft_tokens 传输完成
    # 2. 对每个请求，截断超出有效数量的 draft token
    # 3. 更新 SchedulerOutput 中的 token 计数（total_num_scheduled_tokens、
    #    num_scheduled_tokens），避免后续 KV cache 管理出现错误
    req_data = scheduler_output.scheduled_cached_reqs
    # 中文注释：同步等待 GPU -> CPU 的异步传输完成。
    # 这是整个函数中唯一的同步点，确保 CPU 上的 num_valid_draft_tokens_cpu 数据有效。
    num_valid_draft_tokens_event.synchronize()

    # 中文注释：遍历所有已调度的请求，修剪超出有效范围的 draft token。
    for req_id in req_data.req_ids:
        # 中文注释：获取该请求在 batch 中的索引，用于从 GPU 结果中查找有效 draft 数。
        req_index = req_id_to_index.get(req_id)
        if req_index is None:
            continue

        spec_token_ids = scheduler_output.scheduled_spec_decode_tokens.get(req_id)
        if spec_token_ids is None:
            continue

        # 中文注释：scheduled_k 是 Scheduler 已分配的 draft token 数，
        # valid_k 是 GPU 实际计算出的有效 draft token 数。
        scheduled_k = len(spec_token_ids)

        valid_k = int(num_valid_draft_tokens_cpu[req_index].item())
        valid_k = max(0, min(valid_k, scheduled_k))

        # 中文注释：计算需要裁剪的 token 数，并更新 SchedulerOutput 的全局和逐请求计数。
        tokens_to_trim = scheduled_k - valid_k
        scheduler_output.total_num_scheduled_tokens -= tokens_to_trim
        scheduler_output.num_scheduled_tokens[req_id] -= tokens_to_trim

        # 中文注释：如果有效 draft 数为 0，完全移除该请求的推测 token；
        # 否则截断到有效长度。
        if valid_k == 0:
            scheduler_output.scheduled_spec_decode_tokens.pop(req_id, None)
        else:
            scheduler_output.scheduled_spec_decode_tokens[req_id] = spec_token_ids[
                :valid_k
            ]


def update_ngram_gpu_tensors_incremental(
    input_batch: InputBatch,
    token_ids_gpu_tensor: torch.Tensor,
    num_tokens_no_spec_gpu: torch.Tensor,
    new_reqs: list[CachedRequestState],
    device: torch.device,
    _pinned_idx_buf: torch.Tensor,
    _pinned_val_buf: torch.Tensor,
) -> None:
    """Incrementally update token_ids_gpu_tensor and num_tokens_no_spec_gpu
    for ngram GPU proposer.
    """
    # 中文注释：增量更新 N-gram proposer 在 GPU 上维护的 token_ids 和序列长度张量。
    #
    # 【背景】
    # N-gram proposer 需要在 GPU 上保存所有活跃请求的完整 token 历史（用于滑动窗口匹配）。
    # 每个调度步骤，请求集合可能发生变化（新增、移除、重排序），此函数负责增量同步。
    #
    # 【三种更新场景】
    # 1. 首次运行（prev_req_id_to_index is None）：全量拷贝所有活跃请求的 token_ids
    # 2. 请求重排序：当 batch 中请求的位置发生变化时，交换 GPU tensor 中对应行
    # 3. 新增/恢复请求：将新请求的 token_ids 从 CPU 拷贝到 GPU
    #
    # 【性能优化】
    # - 使用预分配的 pinned memory buffer（_pinned_idx_buf, _pinned_val_buf）避免每步分配
    # - 使用 non_blocking 传输减少 CPU-GPU 同步等待
    # - 只更新发生变化的请求，避免全量拷贝
    prev_req_id_to_index = input_batch.prev_req_id_to_index
    curr_req_id_to_index = input_batch.req_id_to_index

    if not curr_req_id_to_index:
        return

    active_indices = list(curr_req_id_to_index.values())
    n_active = len(active_indices)

    # Use resident pinned buffers to avoid per-call allocation.
    # 中文注释：使用预分配的 pinned memory buffer 构建活跃请求索引，
    # 然后异步传输到 GPU，用于后续的 index_copy_ 等操作。
    active_idx_cpu = _pinned_idx_buf[:n_active]
    active_idx_cpu.copy_(torch.as_tensor(active_indices, dtype=torch.long))

    active_idx_gpu = active_idx_cpu.to(device=device, non_blocking=True)

    new_req_ids = {req.req_id for req in new_reqs}

    # First run, no previous state.
    # 中文注释：首次运行时没有历史状态，需要全量拷贝所有活跃请求的 token_ids 到 GPU。
    if prev_req_id_to_index is None:
        for idx in active_indices:
            num_tokens = input_batch.num_tokens_no_spec[idx]
            if num_tokens > 0:
                token_ids_gpu_tensor[idx, :num_tokens].copy_(
                    input_batch.token_ids_cpu_tensor[idx, :num_tokens],
                    non_blocking=True,
                )

        _sync_num_tokens(
            input_batch,
            num_tokens_no_spec_gpu,
            active_idx_cpu,
            active_idx_gpu,
            n_active,
            device,
            _pinned_val_buf,
        )
        return

    # Detect index changes for reorder.
    # 中文注释：检测请求在 batch 中的位置是否发生变化（重排序）。
    # 当 Scheduler 从 waiting 队列取出新请求或移除完成请求时，batch 中的请求顺序可能改变。
    reorder_src: list[int] = []
    reorder_dst: list[int] = []

    for req_id, curr_idx in curr_req_id_to_index.items():
        if req_id in new_req_ids:
            continue
        prev_idx = prev_req_id_to_index.get(req_id)
        if prev_idx is not None and prev_idx != curr_idx:
            reorder_src.append(prev_idx)
            reorder_dst.append(curr_idx)

    # 中文注释：对位置发生变化的请求，通过 clone + index_copy 交换 GPU tensor 中的行。
    # 使用 clone 创建临时副本避免原地写入的数据竞争。
    if reorder_src:
        src_tensor = async_tensor_h2d(reorder_src, dtype=torch.long, device=device)
        dst_tensor = async_tensor_h2d(reorder_dst, dtype=torch.long, device=device)

        temp_token_ids = token_ids_gpu_tensor[src_tensor].clone()
        temp_num_tokens = num_tokens_no_spec_gpu[src_tensor].clone()

        token_ids_gpu_tensor[dst_tensor] = temp_token_ids
        num_tokens_no_spec_gpu[dst_tensor] = temp_num_tokens

    # Full copy for new/resumed requests.
    # 中文注释：对新增或恢复的请求，从 CPU 全量拷贝 token_ids 到 GPU 对应行。
    for req_state in new_reqs:
        new_req_idx = curr_req_id_to_index.get(req_state.req_id)
        if new_req_idx is None:
            continue

        num_tokens = input_batch.num_tokens_no_spec[new_req_idx]
        if num_tokens > 0:
            token_ids_gpu_tensor[new_req_idx, :num_tokens].copy_(
                input_batch.token_ids_cpu_tensor[new_req_idx, :num_tokens],
                non_blocking=True,
            )

    # Always batch-sync sequence lengths from CPU for ALL active requests.
    # 中文注释：始终同步所有活跃请求的序列长度到 GPU。
    # 序列长度可能因 decode 步进而变化，需要每步更新。
    _sync_num_tokens(
        input_batch,
        num_tokens_no_spec_gpu,
        active_idx_cpu,
        active_idx_gpu,
        n_active,
        device,
        _pinned_val_buf,
    )


def _sync_num_tokens(
    input_batch: InputBatch,
    num_tokens_no_spec_gpu: torch.Tensor,
    active_idx_cpu: torch.Tensor,
    active_idx_gpu: torch.Tensor,
    n_active: int,
    device: torch.device,
    _pinned_val_buf: torch.Tensor,
) -> None:
    """Batch-sync GPU sequence lengths from CPU source of truth.

    Inputs:
        input_batch: Batch container with CPU length tensor.
        num_tokens_no_spec_gpu: Destination GPU length tensor.
        active_idx_cpu: Active request indices on CPU.
        active_idx_gpu: Active request indices on GPU.
        n_active: Number of active requests.
        device: Target CUDA device.
        _pinned_val_buf: Resident pinned int32 staging buffer.
    Outputs:
        None (updates num_tokens_no_spec_gpu in-place).
    """
    # 中文注释：批量同步 CPU 上的序列长度到 GPU。
    #
    # 【流程】
    # 1. 从 CPU tensor 中按 active_idx_cpu 索引选出对应的序列长度值
    # 2. 写入预分配的 pinned memory buffer（vals）
    # 3. 异步传输到 GPU
    # 4. 用 index_copy_ 写入 num_tokens_no_spec_gpu 的对应位置
    #
    # 【为什么用 pinned memory】
    # Pinned memory（页锁定内存）允许 DMA 异步传输，CPU 不需要等待传输完成即可继续执行，
    # 从而实现 CPU-GPU 计算重叠。
    src_cpu = input_batch.num_tokens_no_spec_cpu_tensor
    # 中文注释：从 CPU 源 tensor 中按索引选取活跃请求的序列长度，写入 pinned buffer。
    vals = _pinned_val_buf[:n_active]
    vals.copy_(src_cpu.index_select(0, active_idx_cpu))

    # 中文注释：将 pinned buffer 中的值异步传输到 GPU，并写入 num_tokens_no_spec_gpu 的对应行。
    num_tokens_no_spec_gpu.index_copy_(
        0,
        active_idx_gpu,
        vals.to(device=device, non_blocking=True),
    )


def copy_num_valid_draft_tokens(
    num_valid_draft_tokens_cpu: torch.Tensor,
    num_valid_draft_tokens_copy_stream: torch.cuda.Stream,
    num_valid_draft_tokens_event: torch.cuda.Event,
    num_valid_draft_tokens: torch.Tensor | None,
    batch_size: int,
) -> None:
    """
    Async D2H copy of per-request valid draft counts.
    """
    # 中文注释：异步将 GPU 上的 num_valid_draft_tokens（每个请求的有效 draft token 数）
    # 拷贝到 CPU pinned buffer。
    #
    # 【为什么需要异步 D2H 拷贝】
    # num_valid_draft_tokens 是 kernel 的输出（在 GPU 上），但 Scheduler 在 CPU 上运行，
    # 需要用这个值来修剪超出有效范围的 draft token。
    # 使用独立的 CUDA stream + Event 实现异步传输：
    # 1. 在 copy_stream 上执行 D2H 拷贝（不阻塞默认 stream 的计算）
    # 2. 记录 Event
    # 3. 后续 update_scheduler_for_invalid_drafts() 中通过 event.synchronize() 等待完成
    #
    # 【为什么用单独的 stream】
    # 避免 D2H 拷贝阻塞默认 stream 上的模型前向计算，实现计算与传输的重叠。
    if num_valid_draft_tokens is None:
        return

    # 中文注释：计算实际需要拷贝的请求数（取 batch_size 和 tensor 实际大小的较小值）。
    num_reqs_to_copy = min(batch_size, num_valid_draft_tokens.shape[0])
    if num_reqs_to_copy <= 0:
        return

    # 中文注释：在独立的 copy stream 上执行异步 D2H 拷贝。
    # wait_stream 确保 copy stream 等待默认 stream 上 kernel 计算完成后再开始拷贝。
    default_stream = torch.cuda.current_stream()
    with torch.cuda.stream(num_valid_draft_tokens_copy_stream):
        num_valid_draft_tokens_copy_stream.wait_stream(default_stream)
        num_valid_draft_tokens_cpu[:num_reqs_to_copy].copy_(
            num_valid_draft_tokens[:num_reqs_to_copy], non_blocking=True
        )
        # 中文注释：记录 Event，后续通过 event.synchronize() 等待拷贝完成。
        num_valid_draft_tokens_event.record()
