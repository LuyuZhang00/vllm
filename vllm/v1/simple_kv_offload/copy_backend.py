# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DMA copy backend for GPU<->CPU block transfers.

DMA复制后端，用于GPU<->CPU块传输。
使用cuMemcpyBatchAsync/hipMemcpyBatchAsync API在后台线程中执行批量内存拷贝。

主要特点：
1. 使用后台线程执行批量内存拷贝，避免阻塞主线程
2. 支持CUDA和ROCm平台
3. 使用CUDA事件跟踪传输完成状态
4. 通过队列实现线程间通信
"""

from __future__ import annotations

import queue
import threading

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.v1.simple_kv_offload.cuda_mem_ops import (
    BatchMemcpyParams,
    build_params,
    copy_blocks,
)

logger = init_logger(__name__)


class DmaCopyBackend:
    """cuMemcpyBatchAsync copy backend (background thread).

    cuMemcpyBatchAsync复制后端（后台线程）。

    工作流程：
    1. 初始化时构建存储和加载的批量复制参数
    2. 启动后台线程监听复制请求队列
    3. 当收到复制请求时，调用copy_blocks执行实际的内存拷贝
    4. 记录CUDA事件用于跟踪传输完成状态
    """

    def __init__(self) -> None:
        """初始化DMA复制后端。

        创建后端实例，但不立即初始化。
        需要调用init()方法完成初始化。
        """
        self._store_params: BatchMemcpyParams | None = None  # 存储操作的批量复制参数
        self._load_params: BatchMemcpyParams | None = None  # 加载操作的批量复制参数
        self._load_stream: torch.cuda.Stream | None = None  # 加载CUDA流
        self._store_stream: torch.cuda.Stream | None = None  # 存储CUDA流
        self._queue: queue.SimpleQueue | None = None  # 复制请求队列
        self._thread: threading.Thread | None = None  # 后台工作线程
        self._shutdown: bool = False  # 关闭标志

    def init(
        self,
        gpu_caches: dict[str, torch.Tensor],
        cpu_caches: dict[str, torch.Tensor],
        device: torch.device,
        load_stream: torch.cuda.Stream,
        store_stream: torch.cuda.Stream,
    ) -> None:
        """初始化DMA复制后端。

        构建批量复制参数并启动后台工作线程。

        初始化流程：
        1. 保存CUDA流引用
        2. 构建存储操作（GPU->CPU）的批量复制参数
        3. 构建加载操作（CPU->GPU）的批量复制参数
        4. 创建复制请求队列
        5. 启动后台工作线程

        参数：
        - gpu_caches: GPU KV缓存张量字典
        - cpu_caches: CPU KV缓存张量字典
        - device: GPU设备
        - load_stream: 加载CUDA流
        - store_stream: 存储CUDA流
        """
        self._load_stream = load_stream
        self._store_stream = store_stream

        # 构建批量复制参数
        self._store_params = build_params(gpu_caches, cpu_caches, store_stream)
        self._load_params = build_params(cpu_caches, gpu_caches, load_stream)

        # 创建队列和后台线程
        self._queue = queue.SimpleQueue()
        self._thread = threading.Thread(
            target=self._copy_loop,
            args=(self._queue, device, load_stream, store_stream),
            daemon=True,  # 守护线程，主线程退出时自动终止
        )
        self._thread.start()

    def launch_copy(
        self,
        src_blocks: list[int],
        dst_blocks: list[int],
        is_store: bool,
        event_idx: int,
        events_list: list[tuple[int, torch.Event]],
    ) -> None:
        """启动批量复制操作。

        将复制请求放入队列，由后台线程异步执行。

        参数：
        - src_blocks: 源块ID列表
        - dst_blocks: 目标块ID列表
        - is_store: True表示GPU->CPU存储，False表示CPU->GPU加载
        - event_idx: 事件索引，用于跟踪完成状态
        - events_list: 事件列表，用于记录CUDA事件
        """
        params = self._store_params if is_store else self._load_params
        assert params is not None and self._queue is not None
        # 将复制请求放入队列
        self._queue.put(
            (src_blocks, dst_blocks, params, is_store, event_idx, events_list)
        )

    def shutdown(self) -> None:
        """关闭DMA复制后端。

        发送关闭信号并等待后台线程结束。
        """
        if self._shutdown:
            return
        self._shutdown = True
        if self._queue is not None:
            # 发送None作为关闭信号
            self._queue.put(None)
        if self._thread is not None:
            # 等待线程结束，超时5秒
            self._thread.join(timeout=5.0)

    @staticmethod
    def _copy_loop(
        q: queue.SimpleQueue,
        device: torch.device,
        load_stream: torch.cuda.Stream,
        store_stream: torch.cuda.Stream,
    ) -> None:
        """后台复制循环。

        持续监听队列，执行批量内存拷贝操作。

        工作流程：
        1. 设置当前线程的CUDA设备
        2. 循环从队列获取复制请求
        3. 如果收到None（关闭信号），退出循环
        4. 调用copy_blocks执行实际的内存拷贝
        5. 记录CUDA事件用于跟踪完成状态
        6. 将事件添加到事件列表

        参数：
        - q: 复制请求队列
        - device: GPU设备
        - load_stream: 加载CUDA流
        - store_stream: 存储CUDA流
        """
        current_platform.set_device(device)
        while True:
            # 从队列获取请求（阻塞）
            item = q.get()
            if item is None:
                return  # 收到关闭信号，退出
            src_blocks, dst_blocks, params, is_store, event_idx, events_list = item
            # 执行批量内存拷贝
            copy_blocks(src_blocks, dst_blocks, params)
            stream = store_stream if is_store else load_stream
            # 记录CUDA事件
            event = torch.Event()
            event.record(stream)
            # 将事件添加到列表
            events_list.append((event_idx, event))
