# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
投机解码通用工具模块。

本模块提供了投机解码过程中使用的通用数据结构和工具函数。
主要包括草稿 token 的处理和异步数据传输功能。

在投机解码中，草稿模型生成的候选 token 需要被正确地管理和传递，
本模块中的 DraftTokensHandler 类负责：
1. 缓存草稿 token 的请求 ID 列表
2. 异步地将草稿 token 从 GPU 拷贝到 CPU（用于结构化输出验证）
3. 提供草稿 token 的异步获取接口
"""

import numpy as np
import torch

from vllm.v1.outputs import DraftTokenIds
from vllm.v1.worker.gpu.async_utils import async_copy_to_np
from vllm.v1.worker.gpu.input_batch import InputBatch


class DraftTokensHandler:
    """
    草稿 token 处理器。

    负责管理投机解码过程中生成的草稿 token 的缓存和异步传输。
    主要用于：
    1. 存储草稿模型生成的候选 token
    2. 在需要时（如结构化输出场景）异步将 token 从 GPU 拷贝到 CPU
    3. 提供统一的接口供调度器获取草稿 token 信息

    属性:
        device (torch.device | None): 计算设备。
        copy_stream (torch.cuda.Stream): 用于异步数据拷贝的 CUDA 流。
        copy_event (torch.cuda.Event): 用于同步拷贝完成的 CUDA 事件。
        req_ids (list[str]): 当前批次的请求 ID 列表。
        draft_tokens_np (np.ndarray | None): 草稿 token 的 numpy 数组（GPU 拷贝后的结果）。
        num_draft_tokens (int): 每个请求的草稿 token 数量。
    """

    def __init__(self, device: torch.device | None = None):
        """
        初始化草稿 token 处理器。

        参数:
            device (torch.device | None): 计算设备，默认为 None。
        """
        self.device = device
        # 创建专用的 CUDA 流，用于异步数据拷贝，避免阻塞主计算流
        self.copy_stream = torch.cuda.Stream(device)
        # CUDA 事件，用于标记拷贝操作完成
        self.copy_event = torch.cuda.Event()

        self.req_ids: list[str] = []
        self.draft_tokens_np: np.ndarray | None = None
        self.num_draft_tokens: int = 0

    def set_draft_tokens(
        self, input_batch: InputBatch, draft_tokens: torch.Tensor
    ) -> None:
        """
        设置当前批次的草稿 token。

        将草稿模型生成的 token 缓存起来。如果存在结构化输出请求，
        则异步地将草稿 token 从 GPU 拷贝到 CPU，以便后续进行语法验证。

        参数:
            input_batch (InputBatch): 输入批次信息，包含请求 ID 和结构化输出标志。
            draft_tokens (torch.Tensor): 草稿模型生成的候选 token，形状为 [num_reqs, num_speculative_steps]。

        流程:
            1. 记录当前批次的请求 ID 和草稿 token 数量
            2. 检查是否需要进行结构化输出验证
            3. 如果不需要验证，将 draft_tokens_np 设为 None（跳过异步拷贝）
            4. 如果需要验证，在异步流中将草稿 token 拷贝到 CPU numpy 数组
        """
        self.req_ids = input_batch.req_ids
        self.num_draft_tokens = draft_tokens.shape[1]
        if not input_batch.has_structured_output_reqs:
            # No draft token validation needs to be performed by
            # the scheduler for this batch.
            self.draft_tokens_np = None
            return

        # For spec decoding + structured outputs, we must transfer the
        # draft tokens back to the scheduler for grammar validation.
        # 获取当前 CUDA 流，等待其完成后再进行拷贝
        current_stream = torch.cuda.current_stream(self.device)
        self.copy_stream.wait_stream(current_stream)
        with torch.cuda.stream(self.copy_stream):
            # 异步将 GPU tensor 拷贝到 CPU numpy 数组
            self.draft_tokens_np = async_copy_to_np(draft_tokens)
            # 记录拷贝完成事件
            self.copy_event.record()

    def get_draft_tokens(self) -> DraftTokenIds | None:
        """
        获取草稿 token 信息。

        返回一个 DraftTokenIds 对象，包含请求 ID 和对应的草稿 token 列表。
        如果存在结构化输出请求，会等待异步拷贝完成后返回实际的草稿 token；
        否则返回占位符（-1），表示不需要验证。

        返回:
            DraftTokenIds | None: 包含请求 ID 和草稿 token 的对象，如果没有草稿 token 则返回 None。

        流程:
            1. 如果 draft_tokens_np 不为 None，等待拷贝完成并转换为列表
            2. 否则生成占位符列表（-1），表示跳过验证
        """
        if self.draft_tokens_np is not None:
            # 等待异步拷贝完成
            self.copy_event.synchronize()
            draft_token_ids = self.draft_tokens_np.tolist()
        else:
            # This case only happens when async scheduling is disabled.
            # 生成占位符，表示不需要验证草稿 token
            draft_token_ids = [[-1] * self.num_draft_tokens for _ in self.req_ids]
        return DraftTokenIds(self.req_ids, draft_token_ids)
