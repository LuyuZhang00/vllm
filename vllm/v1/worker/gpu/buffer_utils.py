# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
缓冲区工具模块。

本模块提供高效的 CPU-GPU 数据传输和缓冲区管理工具，是 vLLM v1 引擎
性能优化的关键组件。

核心概念：
1. UVA（Unified Virtual Addressing）：统一虚拟寻址，允许 CPU 和 GPU
   共享同一块内存，减少显式的数据拷贝。

2. 暂存写入（Staged Write）：将多个小的随机写入操作暂存在 CPU 端，
   然后通过单个 Triton 内核批量提交到 GPU，避免大量小的 CUDA 内核调用。

主要组件：
1. async_copy_to_gpu - 将 CPU 张量异步拷贝到 GPU
2. UvaBuffer - 单个 UVA 缓冲区
3. UvaBufferPool - UVA 缓冲区池，支持并发操作
4. UvaBackedTensor - 以 UVA 缓冲区为后端的张量
5. StagedWriteTensor - 支持暂存写入的 GPU 张量
"""
from collections.abc import Iterable, Sequence
from functools import partial

import numpy as np
import torch

from vllm.triton_utils import tl, triton
from vllm.utils.platform_utils import is_uva_available
from vllm.utils.torch_utils import (
    async_tensor_h2d,
    get_accelerator_view_from_cpu_tensor,
)


def async_copy_to_gpu(
    x: torch.Tensor | np.ndarray,
    out: torch.Tensor | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """将 CPU 张量或 NumPy 数组异步拷贝到 GPU。

    直接拷贝到 GPU——显式调用 pin_memory() 在高并发下会因 CUDA 驱动
    竞争导致偶发卡顿。驱动程序无需手动固定内存即可高效处理传输。

    参数:
        x: CPU 上的张量或 NumPy 数组
        out: 可选的输出 GPU 张量（如果提供则复用）
        device: 目标设备（当 out 为 None 时必须提供）

    返回:
        torch.Tensor: GPU 上的张量
    """
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    assert x.is_cpu

    if out is None:
        assert device is not None
        out = torch.empty_like(x, device=device)

    # Copy directly to GPU — explicit pin_memory() causes sporadic stalls
    # under high concurrency due to CUDA driver contention. The driver
    # handles the transfer efficiently without manual pinning.
    return out.copy_(x, non_blocking=True)


class UvaBuffer:
    """单个 UVA（Unified Virtual Addressing）缓冲区。

    UVA 缓冲区同时拥有 CPU 视图和 GPU 视图。CPU 端的修改对 GPU 立即可见，
    反之亦然。这避免了显式的 CPU<->GPU 数据拷贝。

    属性:
        cpu: pinned memory 上的 CPU 张量（源数据写入此处）
        np: CPU 张量的 NumPy 视图（方便 numpy 操作）
        uva: GPU 端的 UVA 视图（GPU 可直接访问此视图）
    """

    def __init__(self, size: int | Sequence[int], dtype: torch.dtype):
        if not is_uva_available():
            raise RuntimeError("UVA is not available")
        # 必须使用 pinned memory 才能创建 UVA 视图
        self.cpu = torch.zeros(size, dtype=dtype, device="cpu", pin_memory=True)
        self.np = self.cpu.numpy()
        self.uva = get_accelerator_view_from_cpu_tensor(self.cpu)


class UvaBufferPool:
    """UVA 缓冲区池，支持并发的数据传输。

    通过维护多个 UVA 缓冲区实现"生产者-消费者"模式的并发。
    当一个缓冲区正在被 GPU 读取时，CPU 可以写入另一个缓冲区。

    属性:
        size: 每个缓冲区的大小
        dtype: 数据类型
        max_concurrency: 最大并发数（缓冲区数量）
    """

    def __init__(
        self,
        size: int | Sequence[int],
        dtype: torch.dtype,
        max_concurrency: int = 2,
    ):
        self.size = size
        self.dtype = dtype
        self.max_concurrency = max_concurrency

        # UVA buffers for concurrency
        self._uva_bufs = [UvaBuffer(size, dtype) for _ in range(max_concurrency)]
        # Current buffer index
        self._curr = 0

    def copy_to_uva(self, x: torch.Tensor | np.ndarray | list) -> torch.Tensor:
        """将数据拷贝到 UVA 缓冲区（CPU->CPU 操作）。

        使用轮询策略选择下一个缓冲区，实现并发操作。

        参数:
            x: 源数据（张量、NumPy 数组或列表）

        返回:
            torch.Tensor: UVA 视图（GPU 可直接访问）
        """
        # Round robin to the next buffer.
        self._curr = (self._curr + 1) % self.max_concurrency
        buf = self._uva_bufs[self._curr]
        # CPU-to-CPU copy
        dst = buf.cpu if isinstance(x, torch.Tensor) else buf.np
        n = len(x)
        dst[:n] = x
        return buf.uva[:n]

    def copy_to_gpu(
        self,
        x: torch.Tensor | np.ndarray,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """将数据通过 UVA 拷贝到 GPU。

        先拷贝到 UVA 缓冲区（CPU->CPU），然后从 UVA 视图拷贝到 GPU。

        参数:
            x: 源数据
            out: 可选的输出 GPU 张量

        返回:
            torch.Tensor: GPU 上的张量
        """
        uva = self.copy_to_uva(x)
        # CPU-to-GPU copy
        return uva.clone() if out is None else out.copy_(uva, non_blocking=True)


class UvaBackedTensor:
    """以 UVA 缓冲区为后端的张量。

    维护一个 CPU 端的"真实数据"张量和一个 UVA 缓冲区池。
    CPU 端的数据更新后，通过 copy_to_uva() 同步到 GPU 可见的视图。

    属性:
        dtype: 数据类型
        cpu: CPU 端的真实数据张量（pinned memory）
        np: CPU 张量的 NumPy 视图
        gpu: 当前 GPU 可见的 UVA 视图
    """

    def __init__(
        self, size: int | Sequence[int], dtype: torch.dtype, max_concurrency: int = 2
    ):
        self.dtype = dtype

        # Source of truth
        self.cpu = torch.zeros(size, dtype=dtype, device="cpu", pin_memory=False)
        self.np = self.cpu.numpy()

        # Buffers for concurrency
        self.pool = UvaBufferPool(size, dtype, max_concurrency)
        self.gpu = self.pool.copy_to_uva(self.np)

    def copy_to_uva(self, n: int | None = None) -> torch.Tensor:
        """将 CPU 数据同步到 UVA 缓冲区。

        参数:
            n: 可选，仅同步前 n 个元素

        返回:
            torch.Tensor: 更新后的 UVA 视图
        """
        # CPU-to-CPU copy
        self.gpu = self.pool.copy_to_uva(self.np[:n] if n is not None else self.np)
        return self.gpu


class StagedWriteTensor:
    """支持暂存写入的 GPU 张量。

    核心思想：将多个小的随机写入操作暂存在 CPU 端的列表中，
    然后通过单个 Triton 内核（_apply_write_kernel）批量提交到 GPU。
    这比逐个调用小的 CUDA 内核高效得多。

    支持的写入模式：
    1. stage_write(index, start, x): 向指定行的指定起始位置写入多个值
    2. stage_write_elem(index, x): 向指定行的第 0 列写入单个值
    3. apply_write(): 批量提交所有暂存的写入操作

    支持的数据类型: int32, int64, float32

    属性:
        gpu: GPU 端的目标张量
        num_rows: 行数
        max_concurrency: 最大并发数
    """

    def __init__(
        self,
        size: int | Sequence[int],
        dtype: torch.dtype,
        device: torch.device,
        max_concurrency: int = 2,
        uva_instead_of_gpu: bool = False,
    ):
        supported_dtypes = [torch.int32, torch.int64, torch.float32]
        if dtype not in supported_dtypes:
            raise ValueError(
                f"Unsupported dtype {dtype}: should be one of {supported_dtypes}"
            )
        self.num_rows = size if isinstance(size, int) else size[0]
        self.dtype = dtype
        self.device = device
        self.max_concurrency = max_concurrency

        if not uva_instead_of_gpu:
            # 创建 GPU 张量（默认）
            self.gpu = torch.zeros(size, dtype=dtype, device=device)
        else:
            # 对于大型但不频繁访问的张量，使用 UVA 代替 GPU 以节省显存
            self._uva_buf = UvaBuffer(size, dtype)
            self.gpu = self._uva_buf.uva

        # 暂存写入操作的缓冲区
        self._staged_write_indices: list[int] = []   # 目标行索引
        self._staged_write_starts: list[int] = []    # 行内起始偏移
        self._staged_write_contents: list[int | float] = []  # 写入内容（展平）
        self._staged_write_cu_lens: list[int] = []   # 累积长度前缀和

        new_buffer = partial(UvaBufferPool, max_concurrency=max_concurrency)

        # UVA 缓冲区，用于将写入元数据传输到 GPU
        self.write_indices = new_buffer(self.num_rows, dtype=torch.int32)
        self.write_starts = new_buffer(self.num_rows, dtype=torch.int32)
        self.write_cu_lens = new_buffer(self.num_rows, dtype=torch.int32)

    def stage_write(
        self, index: int, start: int, x: Iterable[int] | Iterable[float]
    ) -> None:
        """暂存一次行写入操作。

        将要写入的数据记录到 CPU 端的列表中，不立即执行 GPU 写入。

        参数:
            index: 目标行索引
            start: 行内的起始偏移位置
            x: 要写入的数据（可迭代对象）
        """
        assert index >= 0
        assert start >= 0
        if not x:
            return
        self._staged_write_indices.append(index)
        self._staged_write_starts.append(start)
        self._staged_write_contents.extend(x)
        self._staged_write_cu_lens.append(len(self._staged_write_contents))

    def stage_write_elem(self, index: int, x: int) -> None:
        """暂存一次单元素写入操作（写入到行的第 0 列）。

        参数:
            index: 目标行索引
            x: 要写入的单个值
        """
        assert index >= 0
        self._staged_write_indices.append(index)
        self._staged_write_starts.append(0)
        self._staged_write_contents.append(x)
        self._staged_write_cu_lens.append(len(self._staged_write_contents))

    def apply_write(self) -> None:
        """将所有暂存的写入操作批量提交到 GPU。

        通过 Triton 内核 _apply_write_kernel 将所有暂存的写入操作
        一次性应用到 GPU 张量上。这比逐个执行小的 CUDA 内核高效得多。

        流程：
        1. 将写入元数据（索引、起始位置、累积长度）拷贝到 UVA 缓冲区
        2. 将写入内容异步拷贝到 GPU
        3. 启动 Triton 内核执行批量写入
        4. 清空暂存缓冲区
        """
        n = len(self._staged_write_indices)
        if n == 0:
            return

        indices_uva = self.write_indices.copy_to_uva(self._staged_write_indices)
        starts_uva = self.write_starts.copy_to_uva(self._staged_write_starts)
        cu_lens_uva = self.write_cu_lens.copy_to_uva(self._staged_write_cu_lens)

        # Special handling for write_contents
        write_contents = async_tensor_h2d(
            self._staged_write_contents, self.dtype, self.device
        )

        # Write diffs to the GPU buffer
        _apply_write_kernel[(n,)](
            self.gpu,
            self.gpu.stride(0),
            indices_uva,
            starts_uva,
            write_contents,
            cu_lens_uva,
            BLOCK_SIZE=1024,
        )
        # Clear the staged writes
        self.clear_staged_writes()

    def clear_staged_writes(self) -> None:
        """清空所有暂存的写入操作。"""
        self._staged_write_indices.clear()
        self._staged_write_starts.clear()
        self._staged_write_contents.clear()
        self._staged_write_cu_lens.clear()


@triton.jit
def _apply_write_kernel(
    output_ptr,
    output_stride,
    write_indices_ptr,
    write_starts_ptr,
    write_contents_ptr,
    write_cu_lens_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    """批量应用暂存写入操作的 Triton 内核。

    每个 Triton 程序处理一次暂存的写入操作：
    1. 读取目标行索引和行内起始偏移
    2. 从累积长度数组计算出写入内容的范围
    3. 将写入内容从暂存缓冲区拷贝到目标张量的指定位置

    参数:
        output_ptr: 输出张量的指针
        output_stride: 输出张量的行步进
        write_indices_ptr: 写入行索引数组的指针
        write_starts_ptr: 行内起始偏移数组的指针
        write_contents_ptr: 写入内容数组的指针
        write_cu_lens_ptr: 累积长度数组的指针
        BLOCK_SIZE: Triton 块大小
    """
    pid = tl.program_id(0)
    row_idx = tl.load(write_indices_ptr + pid)
    start_idx = tl.load(write_starts_ptr + pid)

    cu_start = tl.load(write_cu_lens_ptr + pid - 1) if pid > 0 else 0
    cu_end = tl.load(write_cu_lens_ptr + pid)
    content_len = cu_end - cu_start

    for i in range(0, content_len, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < content_len
        content = tl.load(write_contents_ptr + cu_start + block, mask=mask)
        tl.store(
            output_ptr + row_idx * output_stride + start_idx + block, content, mask=mask
        )
