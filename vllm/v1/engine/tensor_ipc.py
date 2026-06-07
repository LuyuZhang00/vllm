# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
基于 torch.multiprocessing.Queue 的张量 IPC 传输模块。

本模块实现了通过 torch.multiprocessing.Queue 在进程间共享张量的逻辑，
主要用于 API 服务器与引擎核心之间的通信（例如传输多模态输入张量）。

架构概述：
1. 发送端（TensorIpcSender）：
   - 将张量移到共享内存（share_memory_）
   - 通过 multiprocessing.Queue 发送 TensorIpcData
   - 返回元数据句柄（用于 msgpack 序列化）

2. 接收端（TensorIpcReceiver）：
   - 从 Queue 中接收 TensorIpcData
   - 使用排空-缓冲模式处理乱序到达的张量
   - 根据句柄中的元数据查找对应的张量

数据流：
发送端张量 -> share_memory_() -> Queue -> 接收端缓冲区 -> 返回给调用者

设计要点：
1. 零拷贝传输：使用共享内存，避免张量数据的序列化/反序列化
2. 支持多发送者：通过 sender_id 区分不同的发送进程
3. 乱序处理：接收端缓冲不匹配的张量，支持消息乱序到达
4. 超时保护：Queue 的 get/put 操作都有超时限制
"""

import dataclasses
import uuid
from collections import defaultdict
from dataclasses import field
from multiprocessing.queues import Queue as MPQueue
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.v1.serial_utils import OOBTensorConsumer

logger = init_logger(__name__)

# 张量 IPC 队列类型别名
TensorIpcQueue = MPQueue


@dataclasses.dataclass
class TensorIpcData:
    """
    通过 torch.multiprocessing.Queue 发送的数据结构，支持零拷贝 IPC。

    包含发送者标识、消息标识、张量标识和实际的张量数据。
    张量在共享内存中（GPU 或 CPU），实现高效的进程间通信。

    字段说明：
    - sender_id: 发送者唯一标识（8字符十六进制字符串）
    - message_id: 消息编号（同一发送者内递增）
    - tensor_id: 张量编号（同一消息内递增，支持一条消息包含多个张量）
    - tensor: 实际的张量数据（在共享内存中）
    """

    sender_id: str
    message_id: int
    tensor_id: int
    tensor: torch.Tensor


class TensorIpcSender(OOBTensorConsumer):
    """
    张量 IPC 发送端。

    通过 torch.multiprocessing.Queue 发送张量数据。
    使用单个队列，目标为 rank 0（在 TP>1/PP>1 时，只有 rank 0
    消费多模态张量）。注意：不支持 DP>1。

    实现了 OOBTensorConsumer 接口，可以与 msgpack 序列化器集成。

    工作流程：
    1. new_message() - 标记新消息开始
    2. __call__(tensor) - 发送张量，返回元数据句柄
    3. 句柄通过 msgpack 序列化后随主消息一起发送
    4. 接收端通过句柄从 Queue 中找回对应的张量
    """

    def __init__(self, queue: TensorIpcQueue):
        """
        初始化发送端。

        参数:
            queue: 用于发送张量的 multiprocessing.Queue
        """
        self.queue = queue
        # 张量 ID 计数器（同一消息内递增）
        self._tensor_id_counter = 0
        # 消息 ID 计数器（全局递增）
        self._message_counter = 0
        # 发送者唯一标识（8字符十六进制字符串）
        self._sender_id = uuid.uuid4().hex[:8]

    def set_target_engine(self, target_engine: int) -> None:
        """
        设置目标引擎。

        TensorIpcSender 只支持单个队列（目标为 rank 0），
        不支持指定其他目标引擎。

        参数:
            target_engine: 目标引擎索引

        异常:
            IndexError: 如果 target_engine != 0
        """
        if target_engine != 0:
            raise IndexError(
                "TensorIpcSender only supports a single queue; "
                f"got target engine {target_engine}"
            )

    def new_message(self) -> None:
        """
        标记新消息开始。

        递增消息 ID 计数器，重置张量 ID 计数器。
        每次发送新的主消息（如包含多模态输入的请求）前调用。
        """
        self._message_counter += 1
        self._tensor_id_counter = 0

    def __call__(self, tensor: torch.Tensor) -> dict[str, Any] | None:
        """Send tensor via queue, return its handle. Returns None if failed."""
        """
        发送张量并返回元数据句柄。

        步骤：
        1. 确保张量在共享内存中（调用 share_memory_()）
        2. 构建包含发送者、消息、张量标识的元数据
        3. 创建 TensorIpcData 并通过 Queue 发送
        4. 返回元数据作为句柄（用于 msgpack 序列化）

        参数:
            tensor: 要发送的 PyTorch 张量

        返回:
            元数据字典（sender_id, message_id, tensor_id），
            发送失败时返回 None（此时调用者应回退到标准序列化）
        """
        try:
            # Move tensor to shared memory for IPC
            # This is required for proper inter-process communication
            # 将张量移到共享内存，这是进程间通信所必需的
            if not tensor.is_shared():
                tensor = tensor.share_memory_()

            metadata = {
                "sender_id": self._sender_id,
                "message_id": self._message_counter,
                "tensor_id": self._tensor_id_counter,
            }

            self._tensor_id_counter += 1

            ipc_data = TensorIpcData(**metadata, tensor=tensor)  # type: ignore[arg-type]

            # Use a timeout to avoid blocking indefinitely
            # 使用超时避免无限阻塞（10秒超时）
            self.queue.put(ipc_data, timeout=10.0)

            logger.debug(
                "Sent tensor %s for (shape=%s, device=%s) "
                "via IPC queue (shared memory)",
                metadata,
                tensor.shape,
                tensor.device,
            )

            return metadata
        except Exception as e:
            logger.warning(
                "Failed to send tensor via IPC queue: %s. "
                "Falling back to standard serialization.",
                e,
            )
            return None


@dataclasses.dataclass
class _Sender:
    """
    单个发送者的内部状态。

    用于跟踪每个发送者的当前消息 ID 和已接收的张量缓冲区。

    字段说明：
    - current_message_id: 当前正在处理的消息 ID
    - tensors: 按消息 ID 和张量 ID 组织的张量缓冲区
      结构为 {message_id: {tensor_id: tensor}}
    """
    current_message_id: int = -1
    tensors: dict[int, dict[int, torch.Tensor]] = field(default_factory=dict)


class TensorIpcReceiver:
    """
    张量 IPC 接收端。

    从 torch.multiprocessing.Queue 接收张量数据。
    使用排空-缓冲模式（drain-and-buffer pattern）处理乱序到达的张量。

    工作原理：
    1. 当请求特定张量时，先检查缓冲区
    2. 如果缓冲区没有，从 Queue 中排空所有可用的张量
    3. 将排空的张量存入缓冲区，返回请求的张量
    4. 处理消息过期（stale message）的情况

    设计优势：
    - 支持多个发送者（通过 sender_id 区分）
    - 支持消息乱序到达（通过缓冲区暂存）
    - 自动清理过期消息（避免内存泄漏）
    """

    def __init__(self, queue: TensorIpcQueue):
        """
        初始化接收端。

        参数:
            queue: 用于接收张量的 multiprocessing.Queue
        """
        self.queue = queue
        # 按 sender_id 组织的发送者状态字典
        self._tensor_buffers = defaultdict[str, _Sender](_Sender)

    def __call__(
        self, dtype: str, shape: tuple[int, ...], meta: dict[str, Any]
    ) -> torch.Tensor:
        """Retrieve a tensor from torch.multiprocessing.Queue.

        Uses a drain-and-buffer pattern: drains all available tensors from
        the queue, buffering them, until the requested tensor is found.
        Works for CUDA and CPU.
        """
        """
        从 Queue 中检索指定的张量。

        使用排空-缓冲模式：
        1. 首先检查缓冲区中是否已有请求的张量
        2. 如果没有，从 Queue 中读取一个张量并存入缓冲区
        3. 重复直到找到请求的张量
        4. 清理过期消息的张量以避免内存泄漏

        参数:
            dtype: 期望的数据类型（未使用，仅为接口兼容）
            shape: 期望的张量形状（未使用，仅为接口兼容）
            meta: 元数据字典，包含 sender_id, message_id, tensor_id

        返回:
            请求的 PyTorch 张量

        异常:
            超时（10秒未收到数据）或其他队列错误时会抛出异常
        """

        # Create lookup key from handle
        # 从句柄中提取查找键
        sender_id: str = meta["sender_id"]
        message_id: int = meta["message_id"]
        tensor_id: int = meta["tensor_id"]

        # Drain all available tensors. We save them regardless if this is
        # the one we're waiting for as they may arrive out of order from
        # multiple producers.
        # 排空所有可用张量。无论是否是当前等待的张量都保存，
        # 因为多个生产者可能导致消息乱序到达。
        while True:
            # 首先检查缓冲区
            sender = self._tensor_buffers.get(sender_id)
            if sender is not None:
                tensors = sender.tensors
                # 尝试从缓冲区中弹出目标张量
                tensor = tensors.get(message_id, {}).pop(tensor_id, None)
                if tensor is not None:
                    # 找到了目标张量
                    if sender.current_message_id != message_id:
                        # 当前消息 ID 不匹配，需要清理过期消息
                        while tensors and (mid := next(iter(tensors))) < message_id:
                            if sender.tensors.pop(mid):
                                logger.warning(
                                    "Discarding %d stale tensors from sender %s",
                                    sender_id,
                                )
                        sender.current_message_id = message_id
                    logger.debug(
                        "Received tensor %s from sender %s for (shape=%s, device=%s) "
                        "via IPC queue (shared memory)",
                        (message_id, tensor_id),
                        sender_id,
                        tensor.shape,
                        tensor.device,
                    )
                    return tensor

            # 缓冲区中没有目标张量，从 Queue 中读取
            ipc_data: TensorIpcData = self.queue.get(timeout=10.0)

            # Store tensor
            # 将收到的张量存入缓冲区
            sender = self._tensor_buffers[ipc_data.sender_id]
            if sender.current_message_id > ipc_data.message_id:
                # 收到的张量属于已过期的消息，忽略
                logger.warning(
                    "Ignoring stale tensor from sender %s", ipc_data.sender_id
                )
                continue

            # 按 message_id 和 tensor_id 组织存储
            sender.tensors.setdefault(ipc_data.message_id, {})[ipc_data.tensor_id] = (
                ipc_data.tensor
            )
