# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA graph manager for vision encoder budget-batch execution."""
"""
视觉编码器 CUDA Graph 管理器模块

=============================================================
【模块概述】
=============================================================
本模块实现了视觉编码器（Vision Encoder）的 CUDA Graph 捕获和回放机制。
在多模态模型（如 LLaVA、Qwen-VL 等）中，视觉编码器负责将图像/视频
转换为 token 序列。通过预捕获 CUDA Graph，可以显著减少编码器的推理开销。

=============================================================
【核心设计思想】
=============================================================
1. 基于预算（Budget-based）的 CUDA Graph：
   - 预定义一组 token 预算（如 128, 256, 512, 1024...）
   - 对每个预算值捕获一个 CUDA Graph，该 Graph 能处理
     总输出 token 数不超过该预算的任意图像组合
   - 推理时选择最小的满足需求的预算

2. 贪心装箱（Greedy Packing）算法：
   - 将图像按输出 token 数从小到大排序
   - 贪心地将图像打包到批次中，直到超过预算或批次大小限制
   - 通过交换论证可知，最小优先排序能最小化 eager 回退次数

3. 数据并行（Data Parallel）支持：
   - 当 TP > 1 且使用 data 并行模式时
   - 将图像分配给不同的 TP rank，每个 rank 独立执行编码器
   - 通过 all_gather 收集所有 rank 的输出

=============================================================
【CUDA Graph 捕获/回放流程】
=============================================================
捕获阶段（capture）：
1. 对每个预算值，准备输入缓冲区
2. 执行一次前向传播以分配所有内部张量
3. 捕获 CUDA Graph

回放阶段（execute）：
1. 将实际输入数据复制到预分配的输入缓冲区
2. 调用 graph.replay() 执行捕获的计算图
3. 从输出缓冲区读取结果

=============================================================
【性能优化】
=============================================================
1. 避免动态内存分配：所有缓冲区在捕获时预分配
2. 减少 kernel launch 开销：CUDA Graph 将多个 kernel 合并为一个
3. 零拷贝回放：输入缓冲区原地更新，避免重新分配
"""

from dataclasses import dataclass
from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import (
    SupportsEncoderCudaGraph,
)
from vllm.model_executor.models.utils import scatter_output_slices
from vllm.model_executor.models.vision import get_load_balance_assignment
from vllm.v1.worker.encoder_cudagraph_defs import (
    EncoderCudaGraphConfig,
    EncoderItemSpec,
)

logger = init_logger(__name__)


@dataclass
class BudgetGraphMetadata:
    """Metadata for a single budget graph.

    CUDA graph replay pattern:
    * Copy precomputed values into input_buffers
    * Replay graph
    * Read encoder outputs from output_buffer
    """
    """
    单个预算级别的 CUDA Graph 元数据。

    ==========================================================
    【CUDA Graph 回放模式】
    ==========================================================
    1. 将预计算的值复制到 input_buffers（原地更新）
    2. 调用 graph.replay() 重放计算图
    3. 从 output_buffer 读取编码器输出

    属性说明：
        token_budget: 此 Graph 支持的最大 token 预算
        max_batch_size: 单个批次中最大图像/视频数
        max_frames_per_batch: 单个批次中最大帧数（视频场景）
        graph: 捕获的 CUDA Graph 对象
        input_buffers: Graph 使用的输入缓冲区字典（如 embedding、序列元数据等）
                      回放前，管理器会原地更新这些缓冲区。
                      默认在复制实际值前先清零，模型特定的填充行为由
                      EncoderCudaGraphConfig.padding_logics 提供。
        output_buffer: Graph 写入的输出缓冲区，回放后从中读取结果
    """

    token_budget: int
    max_batch_size: int  # Max number of images/videos per batch
    max_frames_per_batch: int  # Max total frames per batch (for video)
    graph: torch.cuda.CUDAGraph
    # Buffers recorded into the CUDA graph (e.g. embeddings, sequence metadata).
    # Before replay the manager updates these in-place. By default buffers are
    # zeroed before slice-copying the actual values; model-specific padding
    # behavior is provided by EncoderCudaGraphConfig.padding_logics.
    input_buffers: dict[str, torch.Tensor]
    # Output written by graph, read after replay
    output_buffer: torch.Tensor


class EncoderCudaGraphManager:
    """Budget-based CUDA graph capture/replay for vision encoders."""
    """
    视觉编码器的预算级 CUDA Graph 管理器。

    ==========================================================
    【类职责】
    ==========================================================
    1. 确定 token 预算和最大批次大小
    2. 捕获不同预算级别的 CUDA Graph
    3. 在推理时使用贪心装箱算法将图像打包到批次中
    4. 执行 CUDA Graph 回放或 eager 前向传播（回退）
    5. 支持数据并行模式下的分布式编码器执行
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        dtype: torch.dtype,
        model: SupportsEncoderCudaGraph,
    ):
        """Initialize CUDA graph manager with provided token budgets
        and max batch size."""
        """
        初始化 CUDA Graph 管理器。

        ==========================================================
        【初始化流程】
        ==========================================================
        1. 确定 token 预算列表（用户指定或自动推断）
        2. 确定最大批次大小（用户指定或自动推断）
        3. 验证不变量：max_batch_size <= min(token_budgets)
           （确保每个图像至少能分配到 1 个输出 token 位置）
        4. 确定最大帧数（视频场景）
        5. 判断是否使用数据并行模式
        6. 初始化统计计数器

        参数说明：
            vllm_config: vLLM 全局配置
            device: 目标设备
            dtype: 数据类型（如 float16, bfloat16）
            model: 支持 CUDA Graph 的视觉编码器模型
        """
        self.vllm_config = vllm_config
        self.device = device
        self.dtype = dtype
        self.model = model
        self.config: EncoderCudaGraphConfig = model.get_encoder_cudagraph_config()

        comp_config = vllm_config.compilation_config
        user_budgets = comp_config.encoder_cudagraph_token_budgets
        user_max_vision_items = comp_config.encoder_cudagraph_max_vision_items_per_batch
        user_max_frames = comp_config.encoder_cudagraph_max_frames_per_batch

        multimodal_config = vllm_config.model_config.multimodal_config

        # 不变量（Invariant）：max_batch_size <= min_token_budget。
        # 这保证了 per_image_output = budget // max_batch_size >= 1
        # 对于每个捕获的预算都成立，避免 CUDA Graph 捕获时的空张量 reshape 崩溃。
        # Invariant: max_batch_size <= min_token_budget.
        # This ensures per_image_output = budget // max_batch_size >= 1
        # for every captured budget, preventing reshape crashes on empty
        # tensors during CUDA graph capture. Validated/enforced below for
        # each configuration path.
        if user_budgets and user_max_vision_items > 0:
            # 用户完全指定了预算和最大视觉项数，验证不变量
            # Fully user-specified: validate the invariant.
            self.token_budgets = sorted(user_budgets)
            self.max_batch_size = user_max_vision_items
            min_tok = min(self.token_budgets)
            if self.max_batch_size > min_tok:
                raise ValueError(
                    f"encoder_cudagraph_max_vision_items_per_batch "
                    f"({self.max_batch_size}) must be <= smallest token "
                    f"budget ({min_tok}). With budgets="
                    f"{self.token_budgets}, per_image_output = "
                    f"{min_tok} // {self.max_batch_size} = "
                    f"{min_tok // self.max_batch_size}, which would cause "
                    f"a capture failure. Either increase the smallest "
                    f"budget or decrease max_vision_items_per_batch."
                )
        else:
            # 根据模型配置自动推断缺失的值
            # Auto-infer missing values from model.
            min_budget, max_budget = model.get_encoder_cudagraph_budget_range(
                vllm_config
            )
            if min_budget <= 0 or max_budget <= 0:
                raise ValueError(
                    f"Invalid encoder cudagraph budget range: "
                    f"min_budget={min_budget}, max_budget={max_budget}. "
                    f"Both must be positive."
                )
            if min_budget > max_budget:
                raise ValueError(
                    f"Invalid encoder cudagraph budget range: "
                    f"min_budget={min_budget} > max_budget={max_budget}."
                )

            if user_max_vision_items > 0:
                # 用户仅提供了 max_vision_items，调整自动推断的预算
                # 使 min(budgets) >= max_batch_size
                # User provided max_vision_items only; adjust auto-inferred
                # budgets so min(budgets) >= max_batch_size.
                self.max_batch_size = user_max_vision_items
                effective_min = max(min_budget, user_max_vision_items)
                self.token_budgets = self._generate_budgets(effective_min, max_budget)
            elif user_budgets:
                # 用户仅提供了预算，将自动推断的 max_batch_size 限制
                # 为不超过 min(user_budgets)
                # User provided budgets only; cap auto-inferred
                # max_batch_size to min(user_budgets).
                self.token_budgets = sorted(user_budgets)
                self.max_batch_size = min(
                    max_budget // min_budget,
                    min(self.token_budgets),
                )
            else:
                # 完全自动推断
                # Fully auto-inferred.
                self.token_budgets = self._generate_budgets(min_budget, max_budget)
                self.max_batch_size = min(
                    max_budget // min_budget,
                    min(self.token_budgets),
                )

        assert multimodal_config is not None
        if multimodal_config.get_limit_per_prompt("video") == 0:
            # 不支持视频输入
            self.max_frames_per_batch = 0
        elif user_max_frames is not None:
            self.max_frames_per_batch = user_max_frames
        else:
            # 使用模型配置中的默认值
            # Set it to the model-specific value from config.
            max_frames_per_video = self.config.max_frames_per_video
            self.max_frames_per_batch = self.max_batch_size * max_frames_per_video

        # 判断是否使用数据并行模式
        # 当 mm_encoder_tp_mode 为 "data" 且 TP > 1 时启用
        mm_config = vllm_config.model_config.multimodal_config
        self.use_dp = (
            mm_config is not None
            and mm_config.mm_encoder_tp_mode == "data"
            and vllm_config.parallel_config.tensor_parallel_size > 1
        )

        # 预算 -> CUDA Graph 元数据的映射
        self.budget_graphs: dict[int, BudgetGraphMetadata] = {}
        # CUDA Graph 命中/未命中计数器（用于性能统计）
        self.graph_hits = 0
        self.graph_misses = 0
        # 每处理多少个请求输出一次统计日志
        self.log_stats_interval = 100

        logger.info(
            "EncoderCudaGraphManager initialized with "
            "budgets=%s, max_batch_size=%d, max_frames_per_batch=%s, use_dp=%s",
            self.token_budgets,
            self.max_batch_size,
            self.max_frames_per_batch,
            self.use_dp,
        )

    @staticmethod
    def _generate_budgets(min_budget: int, max_budget: int) -> list[int]:
        """Generate power-of-2 token budgets from min_budget to max_budget."""
        """
        生成从 min_budget 到 max_budget 的 2 的幂次预算列表。

        例如：min_budget=128, max_budget=1024 -> [128, 256, 512, 1024]
        如果 max_budget 不是 2 的幂，会额外追加以确保覆盖。

        参数：
            min_budget: 最小预算
            max_budget: 最大预算

        返回：
            排序后的预算列表
        """
        budgets: list[int] = []
        b = min_budget
        while b <= max_budget:
            budgets.append(b)
            b *= 2
        # 确保 max_budget 一定被包含（如果不是 2 的幂的边界）
        # Always include max_budget if it's not already a power-of-2 boundary
        if not budgets or budgets[-1] < max_budget:
            budgets.append(max_budget)
        return budgets

    def supports_modality(self, modality: str) -> bool:
        """Check if a modality is supported by this manager."""
        """检查此管理器是否支持指定的模态（如 "image"、"video"）。"""
        return modality in self.config.modalities

    def capture(self):
        """Capture CUDA graphs for all token budgets."""
        """
        捕获所有预算级别的 CUDA Graph。

        遍历 token_budgets 列表，为每个预算值捕获一个 CUDA Graph。
        此方法应在模型初始化阶段调用一次。
        """
        for token_budget in self.token_budgets:
            self._capture_budget_graph(token_budget)

        logger.info(
            "Encoder CUDA graph capture complete. Captured %d budget graphs.",
            len(self.budget_graphs),
        )

    def _capture_budget_graph(self, token_budget: int):
        """Capture CUDA graph for a single token budget."""
        """
        为单个 token 预算捕获 CUDA Graph。

        ==========================================================
        【捕获流程】
        ==========================================================
        1. 调用模型的 prepare_encoder_cudagraph_capture_inputs()
           准备预分配的输入缓冲区
        2. 执行一次 eager 前向传播（warmup），确保所有内部张量已分配
        3. 创建 CUDA Graph 对象
        4. 在 CUDA Graph 上下文中再次执行前向传播来捕获计算图
        5. 将捕获的 Graph 和缓冲区存储到 budget_graphs 字典中

        参数：
            token_budget: 此 Graph 的 token 预算值
        """
        logger.debug(
            "Capturing encoder cudagraph for budget=%d, max_batch_size=%d, "
            "max_frames_per_batch=%d",
            token_budget,
            self.max_batch_size,
            self.max_frames_per_batch,
        )

        # 准备捕获用的输入数据
        capture_inputs = self.model.prepare_encoder_cudagraph_capture_inputs(
            token_budget,
            self.max_batch_size,
            self.max_frames_per_batch,
            self.device,
            self.dtype,
        )

        values = capture_inputs.values

        # Warmup 前向传播：执行一次以确保所有内部张量/缓冲区已初始化
        with torch.inference_mode():
            output = self.model.encoder_cudagraph_forward({**values})
            output_buffer = torch.empty_like(output)

        # 捕获 CUDA Graph：在 graph 上下文中执行前向传播
        # 所有 CUDA 操作会被记录到 graph 对象中
        graph = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(graph):
            output = self.model.encoder_cudagraph_forward({**values})
            output_buffer.copy_(output)

        self.budget_graphs[token_budget] = BudgetGraphMetadata(
            token_budget=token_budget,
            max_batch_size=self.max_batch_size,
            max_frames_per_batch=self.max_frames_per_batch,
            graph=graph,
            input_buffers=values,
            output_buffer=output_buffer,
        )

    def _find_smallest_fitting_budget_given_tokens(
        self, total_tokens: int
    ) -> int | None:
        """Find smallest budget >= total_tokens.

        Returns:
            Token budget if found, None if no fitting budget.
        """
        """
        查找能容纳指定 token 数量的最小预算。

        使用二分查找或顺序遍历（预算列表已排序），
        找到第一个 >= total_tokens 的预算值。

        参数：
            total_tokens: 需要容纳的 token 总数

        返回：
            匹配的 token 预算，如果没有找到则返回 None
        """
        for budget in self.token_budgets:
            if budget >= total_tokens:
                return budget
        return None

    def _get_item_specs(self, mm_kwargs: dict[str, Any]) -> list[EncoderItemSpec]:
        """Get item specs from the model."""
        """从模型获取每个输入项（图像/视频）的规格描述。"""
        return self.model.get_encoder_cudagraph_item_specs(mm_kwargs)

    def _get_per_item_out_tokens(self, mm_kwargs: dict[str, Any]) -> list[int]:
        """Get per-item output token counts as plain ints."""
        """获取每个输入项的输出 token 数量（纯整数列表）。"""
        return [spec.output_tokens for spec in self._get_item_specs(mm_kwargs)]

    @staticmethod
    def _copy_padded_buffer(
        dst: torch.Tensor,
        src: torch.Tensor,
    ) -> None:
        """
        将 src 复制到 dst，先清零 dst 再复制。
        如果 src 比 dst 短，多余位置保持零值（即 padding）。
        """
        dst.zero_()
        dst[: src.shape[0]].copy_(src)

    def _run_budget_graph(
        self,
        mm_kwargs: dict[str, Any],
        token_budget: int,
    ) -> torch.Tensor | None:
        """Execute budget graph.

        Args:
            mm_kwargs: Multimodal inputs for the batch.
            token_budget: Token budget to use.

        Returns:
            Encoder outputs, or None if graph not captured.
        """
        """
        执行指定预算的 CUDA Graph 回放。

        ==========================================================
        【回放流程】
        ==========================================================
        1. 检查该预算是否有已捕获的 Graph
        2. 调用模型的 prepare_encoder_cudagraph_replay_buffers()
           准备回放用的输入数据
        3. 将输入数据复制到 Graph 的输入缓冲区（按 buffer_keys 遍历）
        4. 调用 graph.replay() 重放计算图
        5. 返回输出缓冲区

        参数：
            mm_kwargs: 多模态输入数据
            token_budget: 使用的 token 预算

        返回：
            编码器输出张量，如果该预算没有已捕获的 Graph 则返回 None
        """
        num_items = len(self._get_item_specs(mm_kwargs))
        if token_budget not in self.budget_graphs:
            self.graph_misses += num_items
            return None

        graph_meta = self.budget_graphs[token_budget]

        # 准备回放用的缓冲区数据
        replay = self.model.prepare_encoder_cudagraph_replay_buffers(
            mm_kwargs,
            self.max_batch_size,
            self.max_frames_per_batch,
        )

        # 将元数据缓冲区复制到 Graph 的输入缓冲区
        # 使用 config 中定义的 buffer_keys 来确定需要更新哪些缓冲区
        # Copy metadata buffers using keys from config.buffer_keys.
        for key in self.config.buffer_keys:
            src = replay.values.get(key)
            if src is None:
                continue
            buf = graph_meta.input_buffers[key]
            if src.ndim == 0:
                # 标量值直接复制
                buf.copy_(src)
            else:
                # 使用填充逻辑复制（默认清零后复制，模型可自定义）
                padding_logic = self.config.padding_logics.get(
                    key, self._copy_padded_buffer
                )
                padding_logic(buf, src)

        # 重放 CUDA Graph
        graph_meta.graph.replay()

        self.graph_hits += num_items
        return graph_meta.output_buffer

    def _execute_local(
        self,
        mm_kwargs: dict[str, Any],
    ) -> list[torch.Tensor]:
        """Execute encoder on local inputs using greedy-packed CUDA graphs.

        Sort images by output token count (smallest first), then greedily pack
        as many images as possible into each batch while staying within
        max_budget tokens and max_batch_size. Once a batch is finalised (next
        image would overflow either constraint), find the smallest fitting
        budget once for that batch.

        By exchange argument, greedy smallest-first packing minimises eager
        fallbacks -- any other ordering yields a higher token sum in some batch,
        making that batch more likely to exceed the budget.

        Stats note:
          graph_hits  -- counted inside _run_budget_graph after successful replay.
          graph_misses -- counted here for single-image batches where the image
                         exceeds max_budget. Batches split due to max_batch_size
                         always satisfy total_tokens <= max_budget and therefore
                         always find a valid budget (no miss).
        """
        """
        在本地（单 rank）执行编码器，使用贪心装箱的 CUDA Graph。

        ==========================================================
        【贪心装箱算法】
        ==========================================================
        1. 将所有图像按输出 token 数从小到大排序
        2. 贪心地将图像打包到当前批次：
           - 如果加入当前图像后不超过 max_budget 且不超过 max_batch_size，
             将其加入当前批次
           - 否则，完成当前批次并开始新批次
        3. 对每个完成的批次，查找最小的满足条件的预算
        4. 如果预算存在，使用 CUDA Graph 回放；否则 eager 回退

        ==========================================================
        【为什么最小优先排序最优？】
        ==========================================================
        通过交换论证（exchange argument）可以证明：
        最小优先排序最小化了需要 eager 回退的次数。
        任何其他排序都会在某些批次中产生更高的 token 总和，
        使得该批次更容易超过预算。

        参数：
            mm_kwargs: 多模态输入数据

        返回：
            按原始顺序排列的编码器输出列表
        """
        item_specs = self._get_item_specs(mm_kwargs)
        num_items = len(item_specs)
        max_budget = self.token_budgets[-1]

        per_item_out_tokens = [spec.output_tokens for spec in item_specs]

        # 按输出 token 数升序排序
        # Sort ascending by output token count (smallest first)
        sorted_indices = sorted(range(num_items), key=lambda i: per_item_out_tokens[i])

        # 贪心装箱：维护当前批次的索引列表和 token 总数
        # Greedy pack against max_budget and max_batch_size.
        # _find_smallest_fitting_budget_given_tokens is called once per
        # finalised batch, not per image.
        batches: list[tuple[list[int], int | None]] = []
        current_batch: list[int] = []
        current_batch_tokens = 0

        for orig_idx in sorted_indices:
            item_tokens = per_item_out_tokens[orig_idx]
            if (
                current_batch_tokens + item_tokens <= max_budget
                and len(current_batch) < self.max_batch_size
            ):
                # 可以加入当前批次
                current_batch.append(orig_idx)
                current_batch_tokens += item_tokens
            else:
                # 当前批次已满，完成它并开始新批次
                if current_batch:
                    batches.append(
                        (
                            current_batch,
                            self._find_smallest_fitting_budget_given_tokens(
                                current_batch_tokens
                            ),
                        )
                    )
                current_batch = [orig_idx]
                current_batch_tokens = item_tokens

        # 处理最后一个批次
        if current_batch:
            batches.append(
                (
                    current_batch,
                    self._find_smallest_fitting_budget_given_tokens(
                        current_batch_tokens
                    ),
                )
            )

        # outputs_by_orig_idx 将原始图像索引映射到输出张量
        # 由于贪心装箱会改变图像顺序，需要在返回前恢复原始顺序
        # outputs_by_orig_idx maps each original image index to its output
        # tensor. Needed because greedy packing reorders images; we restore
        # the original order before returning.
        outputs_by_orig_idx: dict[int, torch.Tensor] = {}

        for batch_orig_indices, token_budget in batches:
            batch_mm_kwargs = self.model.select_encoder_cudagraph_items(
                mm_kwargs, batch_orig_indices
            )
            batch_out_tokens = sum(per_item_out_tokens[i] for i in batch_orig_indices)

            if token_budget is None:
                # 单个超大图像：其 token 数超过了最大预算
                # 需要 eager 回退执行
                # Single oversized image: item_tokens > max_budget.
                # graph_misses counted here for this eager fallback.
                logger.debug(
                    "Encoder CUDA graph fallback to eager: no budget for "
                    "%d tokens from %d images",
                    batch_out_tokens,
                    len(batch_orig_indices),
                )
                self.graph_misses += len(batch_orig_indices)
                with torch.inference_mode():
                    raw = self.model.encoder_eager_forward(batch_mm_kwargs)
                scatter_output_slices(
                    raw,
                    batch_orig_indices,
                    per_item_out_tokens,
                    outputs_by_orig_idx,
                )
            else:
                # 使用 CUDA Graph 回放
                logger.debug(
                    "Encoder CUDA graph: batch_size=%d, tokens=%d, "
                    "budget=%d, waste=%.1f%%",
                    len(batch_orig_indices),
                    batch_out_tokens,
                    token_budget,
                    (token_budget - batch_out_tokens) / token_budget * 100,
                )

                # graph_hits 在 _run_budget_graph 内部的 replay 成功后计数
                # graph_hits counted inside _run_budget_graph after replay.
                output = self._run_budget_graph(batch_mm_kwargs, token_budget)
                assert output is not None
                self.model.postprocess_encoder_output(
                    output,
                    batch_orig_indices,
                    per_item_out_tokens,
                    outputs_by_orig_idx,
                    clone=True,
                    batch_mm_kwargs=batch_mm_kwargs,
                )

        # 按原始批次顺序返回结果（调用方会将输出映射到 token 位置）
        # Return in original batch order (caller maps outputs to token positions)
        return [outputs_by_orig_idx[i] for i in range(num_items)]

    def _dp_shard(
        self,
        mm_kwargs: dict[str, Any],
        per_item_out_tokens: list[int],
    ) -> tuple[dict[str, Any], list[int], list[int], int]:
        """Distribute items across TP ranks for data-parallel execution.

        Uses get_load_balance_assignment() to balance load by input size,
        then select_encoder_cudagraph_items() to extract each rank's inputs.

        Returns:
            local_mm_kwargs: Inputs for this rank.
            image_rank_assignment: Flattened assignment order across all ranks.
            images_per_rank: Number of items per rank.
            max_output_tokens_per_rank: Max output tokens across all ranks
                (for padding during all_gather).
        """
        """
        将输入项分配给各个 TP rank 以进行数据并行执行。

        ==========================================================
        【数据并行分配策略】
        ==========================================================
        1. 使用 get_load_balance_assignment() 按输入大小均衡负载
        2. 每个 rank 只处理分配给它的图像子集
        3. 使用 select_encoder_cudagraph_items() 提取每个 rank 的输入

        ==========================================================
        【为什么按输入大小均衡？】
        ==========================================================
        不同图像的输入大小（patch 数量）可能差异很大。
        按输入大小分配可以确保每个 rank 的计算量大致相等，
        避免某些 rank 过载而其他 rank 空闲。

        参数：
            mm_kwargs: 所有多模态输入数据
            per_item_out_tokens: 每个输入项的输出 token 数

        返回：
            local_mm_kwargs: 本 rank 的输入数据
            image_rank_assignment: 所有 rank 的分配顺序（展平）
            images_per_rank: 每个 rank 分配的图像数
            max_output_tokens_per_rank: 所有 rank 中最大的输出 token 数
                                       （用于 all_gather 时的 padding）
        """
        tp_size = get_tensor_model_parallel_world_size()
        current_rank = get_tensor_model_parallel_rank()

        item_specs = self._get_item_specs(mm_kwargs)
        per_item_input_sizes = [spec.input_size for spec in item_specs]

        # 获取负载均衡分配方案
        (image_rank_assignment, images_per_rank, input_patches_per_rank) = (
            get_load_balance_assignment(per_item_input_sizes, tp_size)
        )

        # 计算每个 rank 的图像起始索引（前缀和）
        cum_images_per_rank = [0]
        for count in images_per_rank:
            cum_images_per_rank.append(cum_images_per_rank[-1] + count)

        # 提取当前 rank 对应的图像索引
        local_indices = image_rank_assignment[
            cum_images_per_rank[current_rank] : cum_images_per_rank[current_rank + 1]
        ]

        if len(local_indices) > 0:
            local_mm_kwargs = self.model.select_encoder_cudagraph_items(
                mm_kwargs, local_indices
            )
        else:
            local_mm_kwargs = self.model.select_encoder_cudagraph_items(mm_kwargs, [])

        # 计算所有 rank 中最大的输出 token 数，用于 all_gather 时的 padding
        max_output_tokens_per_rank = (
            max(
                sum(
                    per_item_out_tokens[i]
                    for i in image_rank_assignment[
                        cum_images_per_rank[r] : cum_images_per_rank[r + 1]
                    ]
                )
                for r in range(tp_size)
            )
            if len(per_item_out_tokens) > 0
            else 0
        )

        return (
            local_mm_kwargs,
            image_rank_assignment,
            images_per_rank,
            max_output_tokens_per_rank,
        )

    def _dp_gather(
        self,
        local_outputs: list[torch.Tensor],
        per_item_out_tokens: list[int],
        image_rank_assignment: list[int],
        images_per_rank: list[int],
        max_output_tokens_per_rank: int,
    ) -> list[torch.Tensor]:
        """Gather outputs from all TP ranks and reorder to original sequence.

        Assumes 2D output tensors [tokens, hidden]. Follows the same
        pad -> all_gather -> unpad -> reorder algorithm as
        run_dp_sharded_mrope_vision_model() in the eager path.
        """
        """
        从所有 TP rank 收集输出并恢复原始顺序。

        ==========================================================
        【收集流程】
        ==========================================================
        1. 将本 rank 的输出拼接为连续张量
        2. Padding 到 max_output_tokens_per_rank 以对齐长度
        3. 使用 all_gather 从所有 rank 收集数据
        4. Unpad：从收集结果中提取每个 rank 的实际输出
        5. Reorder：将输出按原始图像顺序重新排列

        假设输出张量为 2D：[tokens, hidden_size]
        算法与 eager 路径中的 run_dp_sharded_mrope_vision_model() 一致

        参数：
            local_outputs: 本 rank 的编码器输出列表
            per_item_out_tokens: 每个输入项的输出 token 数
            image_rank_assignment: 所有 rank 的图像分配顺序
            images_per_rank: 每个 rank 分配的图像数
            max_output_tokens_per_rank: 最大的 rank 输出 token 数

        返回：
            按原始顺序排列的所有编码器输出
        """
        hidden_size = self.config.out_hidden_size
        tp_size = len(images_per_rank)

        if len(local_outputs) > 0:
            local_concat = torch.cat(local_outputs, dim=0)
        else:
            local_concat = torch.empty(
                (0, hidden_size), device=self.device, dtype=self.dtype
            )

        # Padding 到 max_output_tokens_per_rank 以便 all_gather 对齐
        # Pad to max_output_tokens_per_rank for all_gather
        current_len = local_concat.shape[0]
        if current_len < max_output_tokens_per_rank:
            padding = torch.empty(
                (max_output_tokens_per_rank - current_len, hidden_size),
                dtype=self.dtype,
                device=self.device,
            )
            local_padded = torch.cat([local_concat, padding], dim=0)
        else:
            local_padded = local_concat

        # 从所有 rank 收集数据
        gathered = tensor_model_parallel_all_gather(local_padded, dim=0)

        # Unpad：从收集结果中提取每个 rank 的实际输出（去除 padding）
        # Unpad each rank's contribution
        rank_outputs: list[torch.Tensor] = []
        current_idx = 0
        for rank in range(tp_size):
            start = rank * max_output_tokens_per_rank
            rank_count = images_per_rank[rank]
            rank_indices = image_rank_assignment[current_idx : current_idx + rank_count]
            rank_tokens = sum(per_item_out_tokens[i] for i in rank_indices)
            current_idx += rank_count
            rank_outputs.append(gathered[start : start + rank_tokens])

        # Reorder：将各 rank 的输出按原始图像顺序重新排列
        # Reorder to original sequence
        total_items = len(per_item_out_tokens)
        result: list[torch.Tensor | None] = [None] * total_items
        current_idx = 0
        for rank in range(tp_size):
            count = images_per_rank[rank]
            if count > 0:
                rank_items = image_rank_assignment[current_idx : current_idx + count]
                scatter_output_slices(
                    rank_outputs[rank],
                    rank_items,
                    per_item_out_tokens,
                    result,
                )
                current_idx += count

        return [t for t in result if t is not None]

    def execute(
        self,
        mm_kwargs: dict[str, Any],
    ) -> list[torch.Tensor]:
        """Execute encoder using CUDA graph with optional DP.

        Args:
            mm_kwargs: Multimodal keyword arguments containing the
                input tensor and grid dimensions.

        Returns:
            List of encoder outputs (one per item).
        """
        """
        使用 CUDA Graph（可选数据并行）执行编码器。

        ==========================================================
        【执行流程】
        ==========================================================
        1. 如果启用数据并行（use_dp=True）：
           a. 通过 _dp_shard() 将图像分配给各 rank
           b. 每个 rank 使用 _execute_local() 执行本地编码器
           c. 通过 _dp_gather() 收集并重排结果
        2. 如果未启用数据并行：
           直接调用 _execute_local() 执行
        3. 定期输出性能统计日志

        参数：
            mm_kwargs: 多模态输入关键字参数

        返回：
            每个输入项的编码器输出张量列表
        """
        if self.use_dp:
            per_item_out_tokens = self._get_per_item_out_tokens(mm_kwargs)

            (
                local_mm_kwargs,
                image_rank_assignment,
                images_per_rank,
                max_output_tokens_per_rank,
            ) = self._dp_shard(mm_kwargs, per_item_out_tokens)

            local_outputs = self._execute_local(local_mm_kwargs)

            result = self._dp_gather(
                local_outputs,
                per_item_out_tokens,
                image_rank_assignment,
                images_per_rank,
                max_output_tokens_per_rank,
            )
        else:
            result = self._execute_local(mm_kwargs)

        # 定期输出累计统计日志
        # Log cumulative stats periodically
        stats = self.get_cumulative_stats()
        total_requests = self.graph_hits + self.graph_misses
        if total_requests > 0 and total_requests % self.log_stats_interval == 0:
            logger.debug(
                "Encoder CUDA graph cumulative stats: "
                "hits=%d, misses=%d, hit_rate=%.1f%%",
                stats["graph_hits"],
                stats["graph_misses"],
                stats["hit_rate"] * 100,
            )

        return result

    def get_cumulative_stats(self) -> dict[str, Any]:
        """Get cumulative CUDA graph statistics."""
        """
        获取累计的 CUDA Graph 统计信息。

        返回包含以下键的字典：
        - graph_hits: CUDA Graph 回放成功的次数
        - graph_misses: 未命中（需要 eager 回退）的次数
        - hit_rate: 命中率（0.0 ~ 1.0）
        - num_budgets: 已捕获的预算数量
        - token_budgets: 预算值列表
        """
        total_requests = self.graph_hits + self.graph_misses
        hit_rate = self.graph_hits / total_requests if total_requests > 0 else 0.0

        return {
            "graph_hits": self.graph_hits,
            "graph_misses": self.graph_misses,
            "hit_rate": hit_rate,
            "num_budgets": len(self.budget_graphs),
            "token_budgets": self.token_budgets,
        }
