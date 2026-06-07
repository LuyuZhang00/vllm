# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
CPU 工作器模块 (vllm/v1/worker/cpu/__init__.py)

本模块实现了 CPU 后端的模型运行环境。其核心思路是通过猴子补丁 (monkey-patching) 的方式，
将原本为 GPU (CUDA) 设计的 torch 和 vllm 内部 API 替换为 CPU 兼容的空操作 (no-op) 或
简化实现，从而使得 GPU 模型运行器 (GPUModelRunner) 的代码能够在 CPU 环境下直接复用，
而无需大规模修改。

工作原理概述：
1. 替换 CUDA Event/Stream 相关 API 为空操作占位符 (_EventPlaceholder, _StreamPlaceholder)，
   因为 CPU 推理不需要 CUDA 流同步。
2. 替换 torch.accelerator.synchronize / empty_cache 为空操作。
3. 替换 async_tensor_h2d 为简单的 torch.tensor 创建（CPU 上无需异步主机到设备拷贝）。
4. 替换 GPU 版本的 UvaBuffer 为 CPU 版本（CPU 上不支持 UVA，但提供相同接口以便兼容）。

这样做的好处是最大化代码复用，让 CPU 后端只需维护少量差异代码。
"""

# isort: skip_file
# ruff: noqa: E402
# mypy: disable-error-code="misc, assignment"

from typing import Any

# Patch torch APIs
# 补丁 torch API：将 CUDA 相关的 Event、Stream 等替换为空操作实现
import torch


def noop(*args: Any, **kwargs: Any) -> None:
    """
    通用空操作函数，用于替换所有不需要在 CPU 上执行的 CUDA 操作。
    例如 synchronize、empty_cache 等。
    """
    pass


class _EventPlaceholder:
    """
    CUDA Event 的 CPU 占位符实现。

    在 GPU 上，Event 用于记录 CUDA 流中的时间点，用于流间同步。
    在 CPU 上不需要这些功能，因此所有方法都替换为空操作。
    """

    def __init__(self, *args, **kwargs) -> None:
        # record: 在 CUDA 流中记录事件点 -> CPU 上无操作
        self.record = noop
        # synchronize: 等待事件完成 -> CPU 上无操作
        self.synchronize = noop


class _StreamPlaceholder:
    """
    CUDA Stream 的 CPU 占位符实现。

    在 GPU 上，Stream 是异步执行的命令队列。在 CPU 上所有计算都是同步的，
    因此 Stream 概念不需要。提供上下文管理器接口以支持 `with` 语法。
    """

    def __init__(self, *args, **kwargs) -> None:
        # wait_stream: 等待另一个流完成 -> CPU 上无操作
        self.wait_stream = noop

    def __enter__(self, *args, **kwargs):
        # 进入流上下文 -> CPU 上直接返回自身即可
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 退出流上下文 -> CPU 上无操作
        pass


# 以下补丁替换所有 CUDA 相关的 torch API 为空操作版本
# 1. Event 和 Stream 替换为占位符
torch.Event = _EventPlaceholder
torch.cuda.Event = _EventPlaceholder
torch.cuda.Stream = _StreamPlaceholder
# 2. 流设置和获取替换为空操作
torch.cuda.set_stream = noop
torch.cuda.current_stream = lambda *args, **kwargs: _StreamPlaceholder()
# 3. 设备同步和缓存清理替换为空操作（CPU 不需要这些）
torch.accelerator.synchronize = noop
torch.accelerator.empty_cache = noop

# Patch vLLM torch utils
# 补丁 vLLM 的 torch 工具函数
import vllm.utils.torch_utils as torch_utils


def async_tensor_h2d(
    data: list,
    dtype: torch.dtype,
    device: str | torch.device,
    pin_memory: bool = False,
) -> torch.Tensor:
    """
    替换原始的 async_tensor_h2d 函数。

    原始实现将数据从主机 (host) 异步传输到设备 (device)。
    在 CPU 后端中，设备就是 CPU 本身，所以直接创建 CPU 张量即可。
    忽略 pin_memory 参数（pin memory 是 CUDA 优化技术，对 CPU 无意义）。
    """
    return torch.tensor(data, dtype=dtype, device="cpu")


# 用 CPU 版本替换 vllm 的异步 h2d 传输函数
torch_utils.async_tensor_h2d = async_tensor_h2d

# Patch model runner APIs
# 补丁模型运行器 API：用 CPU 版本的 buffer_utils 替换 GPU 版本
import vllm.v1.worker.gpu.buffer_utils as gpu_buffer_utils
import vllm.v1.worker.cpu.buffer_utils as cpu_buffer_utils

# 用 CPU 版本的 UvaBuffer 替换 GPU 版本
# GPU 版本使用 CUDA Unified Virtual Addressing (UVA)，
# CPU 版本使用普通的 torch.zeros 创建缓冲区
gpu_buffer_utils.UvaBuffer = cpu_buffer_utils.UvaBuffer
