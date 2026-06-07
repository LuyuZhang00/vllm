# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
池化元数据模块 (vllm/v1/pool/metadata.py)

本模块定义了池化操作所需的元数据结构。

池化（Pooling）概述：
- 池化是将变长的 token 序列转换为固定大小向量的操作
- 常见的池化方式：取第一个 token（BOS）、取最后一个 token（EOS）、取平均等
- vLLM v1 中，池化元数据与调度输出一起传递给模型运行器

核心组件：
1. PoolingCursor: 池化游标
   - 记录每个序列在隐藏状态张量中的首/尾 token 位置
   - 用于从扁平化的隐藏状态张量中提取各序列的池化结果
   - 支持部分 prefill（chunked prefill）的检测

2. PoolingStates: 池化状态
   - 用于 chunked prefill 场景，缓存中间隐藏状态
   - 在整个 prefill 完成后才执行池化

3. PoolingMetadata: 池化元数据
   - 包含所有池化操作所需的信息
   - prompt_lens: 每个序列的 prompt 长度
   - prompt_token_ids: 每个序列的 token IDs
   - pooling_params: 每个序列的池化参数
   - pooling_states: 每个序列的池化状态
   - pooling_cursor: 池化游标（在 build_pooling_cursor 中构建）
"""

from dataclasses import dataclass

import numpy as np
import torch

from vllm.pooling_params import PoolingParams
from vllm.tasks import PoolingTask
from vllm.utils.platform_utils import is_pin_memory_available

pin_memory = is_pin_memory_available()


@dataclass
class PoolingCursor:
    """
    池化游标，记录每个序列在隐藏状态张量中的位置信息。

    用于从扁平化的隐藏状态张量中高效提取各序列的首/尾 token 表示。

    属性：
        first_token_indices_gpu: 每个序列第一个 token 在隐藏状态中的索引（GPU）
        last_token_indices_gpu: 每个序列最后一个 token 在隐藏状态中的索引（GPU）
        prompt_lens_cpu: 每个序列的 prompt 长度（CPU）
        seq_lens_cpu: 每个序列的总长度（CPU）
        num_scheduled_tokens_cpu: 本次调度中每个序列的 token 数（CPU）
    """
    first_token_indices_gpu: torch.Tensor
    last_token_indices_gpu: torch.Tensor
    prompt_lens_cpu: torch.Tensor
    seq_lens_cpu: torch.Tensor
    num_scheduled_tokens_cpu: torch.Tensor

    def __getitem__(self, indices: slice) -> "PoolingCursor":
        """支持切片索引，返回新的 PoolingCursor 子集。"""
        return PoolingCursor(
            first_token_indices_gpu=self.first_token_indices_gpu[indices],
            last_token_indices_gpu=self.last_token_indices_gpu[indices],
            prompt_lens_cpu=self.prompt_lens_cpu[indices],
            seq_lens_cpu=self.seq_lens_cpu[indices],
            num_scheduled_tokens_cpu=self.num_scheduled_tokens_cpu[indices],
        )

    def is_partial_prefill(self) -> bool:
        """
        检查是否是部分 prefill（chunked prefill）。

        当调度的 token 数不等于 prompt 长度时，说明是 chunked prefill 的一部分。

        Returns:
            True 表示是部分 prefill
        """
        return not torch.all(self.prompt_lens_cpu == self.num_scheduled_tokens_cpu)

    def is_finished(self) -> torch.Tensor:
        """
        检查每个序列是否已完成（prompt 处理完毕）。

        当 seq_lens == prompt_lens 时，序列的 prompt 已完全处理。

        Returns:
            布尔张量，True 表示对应序列已完成
        """
        return self.prompt_lens_cpu == self.seq_lens_cpu


class PoolingStates:
    """
    池化状态，用于 chunked prefill 场景。

    在 chunked prefill 中，prompt 被分成多个 chunk 处理。
    每个 chunk 的隐藏状态需要缓存起来，在整个 prompt 处理完毕后
    再执行池化操作。

    属性：
        hidden_states_cache: 缓存的隐藏状态列表
    """

    def __init__(self) -> None:
        # 用于 chunked prefill 场景，缓存中间隐藏状态
        self.hidden_states_cache: list[torch.Tensor] = []

    def clean(self) -> None:
        """清空缓存的隐藏状态。"""
        self.hidden_states_cache.clear()


@dataclass
class PoolingMetadata:
    """
    池化操作的元数据，包含执行池化所需的所有信息。

    属性：
        prompt_lens: 每个序列的 prompt 长度（CPU 张量）
        prompt_token_ids: 每个序列的 token IDs（模型设备张量）
        prompt_token_ids_cpu: 每个序列的 token IDs（CPU 张量）
        pooling_params: 每个序列的池化参数（包含任务类型等）
        pooling_states: 每个序列的池化状态（用于 chunked prefill）
        pooling_cursor: 池化游标（在 build_pooling_cursor 中初始化）
    """

    prompt_lens: torch.Tensor  # CPU Tensor
    prompt_token_ids: torch.Tensor | None  # Model-device tensor
    prompt_token_ids_cpu: torch.Tensor | None  # CPU tensor
    pooling_params: list[PoolingParams]
    pooling_states: list[PoolingStates]
    pooling_cursor: PoolingCursor | None = None

    def __post_init__(self) -> None:
        """初始化后验证：每个 pooling_param 必须有 task。"""
        pooling_params = self.pooling_params

        tasks: list[PoolingTask] = [
            task
            for pooling_param in pooling_params
            if (task := pooling_param.task) is not None
        ]
        if len(pooling_params) != len(tasks):
            raise ValueError(
                "Every pooling param must have a task set, but got "
                f"{len(tasks)} tasks for {len(pooling_params)} pooling params"
            )

        self.tasks = tasks

    def __getitem__(self, indices: slice) -> "PoolingMetadata":
        """支持切片索引，返回新的 PoolingMetadata 子集。"""
        return PoolingMetadata(
            prompt_lens=self.prompt_lens[indices],
            prompt_token_ids=None
            if self.prompt_token_ids is None
            else self.prompt_token_ids[indices],
            prompt_token_ids_cpu=None
            if self.prompt_token_ids_cpu is None
            else self.prompt_token_ids_cpu[indices],
            pooling_params=self.pooling_params[indices],
            pooling_states=self.pooling_states[indices],
            pooling_cursor=None
            if self.pooling_cursor is None
            else self.pooling_cursor[indices],
        )

    def _get_prompt_token_ids(
        self,
        prompt_token_ids: torch.Tensor | None,
    ) -> list[torch.Tensor]:
        """
        提取每个序列的 prompt token IDs 列表。

        Args:
            prompt_token_ids: prompt token IDs 张量

        Returns:
            每个序列的 token IDs 张量列表

        Raises:
            ValueError: 如果 prompt_token_ids 未设置
        """
        if prompt_token_ids is None:
            raise ValueError(
                "prompt_token_ids is required but was not set. "
                "Please set `requires_token_ids=True` in `get_pooling_updates`"
            )
        return [prompt_token_ids[i, :num] for i, num in enumerate(self.prompt_lens)]

    def get_prompt_token_ids(self) -> list[torch.Tensor]:
        """获取模型设备上的 prompt token IDs 列表。"""
        return self._get_prompt_token_ids(self.prompt_token_ids)

    def get_prompt_token_ids_cpu(self) -> list[torch.Tensor]:
        """获取 CPU 上的 prompt token IDs 列表。"""
        return self._get_prompt_token_ids(self.prompt_token_ids_cpu)

    def get_pooling_cursor(self) -> PoolingCursor:
        """
        获取池化游标。

        Returns:
            PoolingCursor 对象

        Raises:
            RuntimeError: 如果游标未初始化
        """
        pooling_cursor = self.pooling_cursor
        if pooling_cursor is None:
            raise RuntimeError(
                "pooling_cursor has not been initialized. "
                "Call `build_pooling_cursor` before accessing it"
            )

        return pooling_cursor

    def build_pooling_cursor(
        self,
        num_scheduled_tokens_np: np.ndarray,
        seq_lens_cpu: torch.Tensor,
        device: torch.device,
        query_start_loc_gpu: torch.Tensor | None = None,
    ) -> None:
        """
        构建池化游标。

        计算每个序列在隐藏状态张量中的首/尾 token 位置。
        这些位置用于从扁平化的隐藏状态中提取各序列表示。

        计算过程：
        1. 计算每个序列的调度 token 数的累积和
        2. 首 token 索引 = cumsum[:-1]（每个序列的起始位置）
        3. 尾 token 索引 = cumsum[1:] - 1（每个序列的结束位置）

        Args:
            num_scheduled_tokens_np: 每个序列的调度 token 数（numpy 数组）
            seq_lens_cpu: 每个序列的总长度（CPU 张量）
            device: 目标设备
            query_start_loc_gpu: 可选的预计算查询起始位置（GPU 张量）
        """
        n_seq = len(num_scheduled_tokens_np)
        prompt_lens = self.prompt_lens

        if len(prompt_lens) != n_seq:
            raise ValueError(
                f"prompt_lens length ({len(prompt_lens)}) does not match "
                f"the number of sequences ({n_seq})"
            )

        num_scheduled_tokens_cpu = torch.from_numpy(num_scheduled_tokens_np)
        if query_start_loc_gpu is None:
            # 计算累积和作为位置索引
            cumsum = torch.zeros(
                n_seq + 1, dtype=torch.int64, pin_memory=pin_memory, device="cpu"
            )
            torch.cumsum(num_scheduled_tokens_cpu, dim=0, out=cumsum[1:])
            cumsum = cumsum.to(device, non_blocking=True)
        else:
            if query_start_loc_gpu.shape[0] != n_seq + 1:
                raise ValueError(
                    "query_start_loc_gpu length does not match "
                    f"the number of sequences: {query_start_loc_gpu.shape[0]} "
                    f"!= {n_seq + 1}."
                )
            if query_start_loc_gpu.device != device:
                raise ValueError(
                    "query_start_loc_gpu must be on the same device as the "
                    f"hidden states: {query_start_loc_gpu.device} != {device}."
                )
            cumsum = query_start_loc_gpu
        # 构建池化游标
        self.pooling_cursor = PoolingCursor(
            first_token_indices_gpu=cumsum[:n_seq],      # 每个序列的首 token 位置
            last_token_indices_gpu=cumsum[1:] - 1,       # 每个序列的尾 token 位置
            prompt_lens_cpu=prompt_lens,                  # prompt 长度
            seq_lens_cpu=seq_lens_cpu,                    # 序列总长度
            num_scheduled_tokens_cpu=num_scheduled_tokens_cpu,  # 调度 token 数
        )
