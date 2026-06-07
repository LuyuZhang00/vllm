# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
CPU 缓冲区工具模块 (vllm/v1/worker/cpu/buffer_utils.py)

本模块提供 CPU 后端的缓冲区实现，作为 GPU 版本 UvaBuffer 的替代。

UVA (Unified Virtual Addressing) 背景：
- 在 CUDA 中，UVA 允许 CPU 和 GPU 共享统一的虚拟地址空间，
  使得 CPU 可以直接通过指针访问 GPU 内存。
- 在 CPU 后端中，不存在 GPU 设备，因此 UVA 的概念简化为普通的 CPU 内存分配。
- 本模块的 UvaBuffer 保持与 GPU 版本相同的接口（cpu、np、uva 三个属性），
  以确保上层代码无需修改即可在 CPU 环境下运行。

接口说明：
- cpu: CPU 上的 torch.Tensor，用于存储实际数据
- np: 对应的 numpy 数组视图，用于高效的数据操作
- uva: 指向同一 CPU 张量（CPU 上无 UVA 区分，直接指向 cpu 属性）
"""

from collections.abc import Sequence

import torch

from vllm.utils.platform_utils import is_uva_available


class UvaBuffer:
    """
    CPU 后端的 UVA 缓冲区实现。

    与 GPU 版本的 UvaBuffer 保持相同接口，但底层使用普通 CPU 内存。
    这样上层代码（如 buffer_utils）可以直接替换使用，无需感知底层设备差异。

    属性：
        cpu (torch.Tensor): CPU 上的张量，dtype 和 size 由调用者指定。
        np (numpy.ndarray): cpu 张量的 numpy 视图，共享底层内存，无数据拷贝。
        uva (torch.Tensor): 与 cpu 指向同一张量。在 GPU 版本中，uva 指向
            CUDA 统一内存；在 CPU 版本中，直接指向 cpu 张量。
    """

    def __init__(self, size: int | Sequence[int], dtype: torch.dtype):
        # 检查 UVA 可用性（CPU 后端中通常始终可用）
        if not is_uva_available():
            raise RuntimeError("UVA is not available")
        # 创建 CPU 上的零初始化张量，作为数据存储
        self.cpu = torch.zeros(size, dtype=dtype, device="cpu")
        # 获取 numpy 数组视图，便于高效数值操作（无数据拷贝）
        self.np = self.cpu.numpy()
        # 在 CPU 后端中，uva 直接指向 cpu 张量
        # （GPU 版本中 uva 指向 CUDA 统一内存区域）
        self.uva = self.cpu
