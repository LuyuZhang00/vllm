# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
CPU 加载/存储规格模块 (vllm/v1/kv_offload/cpu/common.py)

本模块定义了 CPU 内存作为 KV 缓存卸载目标时的加载/存储规格。

CPULoadStoreSpec 的作用：
- 继承自 BlockIDsLoadStoreSpec 基类
- 通过 block_ids 指定要加载/存储的 CPU 缓存块 ID
- medium() 方法返回 "CPU" 字符串，标识数据传输的目标介质
- 在调度器与工作器之间传递时，用于描述"哪些块需要从/到 CPU 内存传输"

使用场景：
1. prepare_load 时：创建 CPULoadStoreSpec 指定要从 CPU 加载的块 ID
2. prepare_store 时：创建 CPULoadStoreSpec 指定要存储到 CPU 的块 ID
3. 工作器接收到 spec 后，根据 block_ids 计算内存地址并执行 DMA 传输
"""

from vllm.v1.kv_offload.base import BlockIDsLoadStoreSpec


class CPULoadStoreSpec(BlockIDsLoadStoreSpec):
    """
    CPU 内存的 KV 块加载/存储规格。

    描述 KV 缓存块在 CPU 内存中的位置，用于指导数据传输操作。
    通过 block_ids 列表指定需要操作的 CPU 缓存块编号。
    """

    @staticmethod
    def medium() -> str:
        """
        返回存储介质标识符。

        Returns:
            "CPU" 字符串，标识此规格对应的存储介质为 CPU 内存。
            用于日志记录、事件通知等场景中区分不同的卸载目标。
        """
        return "CPU"
