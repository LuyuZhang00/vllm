# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
CPU 卸载规格模块 (vllm/v1/kv_offload/cpu/spec.py)

本模块定义了 CPU KV 缓存卸载的配置规格，连接调度器侧（Manager）和工作器侧（Handlers）。

CPUOffloadingSpec 的职责：
1. 计算 CPU 可卸载的缓存块数（基于配置的 CPU 内存大小）
2. 创建调度器侧的 OffloadingManager（负责缓存管理、淘汰决策）
3. 创建工作器侧的 OffloadingHandlers（负责实际数据传输）
4. 通过 get_handlers 方法向工作器暴露双向传输处理器

配置参数（通过 kv_connector_extra_config 传入）：
- cpu_bytes_to_use: 可用于 CPU 卸载的内存大小（字节）
- eviction_policy: 淘汰策略（"lru" 或 "arc"）
- store_threshold: 卸载阈值（块需要在 lookup 中出现此次数后才允许卸载）
- max_tracker_size: 引用计数跟踪器的最大容量

块大小计算：
- kv_bytes_per_block = 总 GPU KV 字节数 / GPU 块数 * world_size
- kv_bytes_per_offloaded_block = kv_bytes_per_block * block_size_factor
- num_blocks = cpu_bytes_to_use / kv_bytes_per_offloaded_block
"""

from collections.abc import Iterator

from vllm.config import VllmConfig
from vllm.platforms import current_platform
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.kv_offload.base import (
    CanonicalKVCaches,
    GPULoadStoreSpec,
    LoadStoreSpec,
    OffloadingManager,
    OffloadingSpec,
)
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from vllm.v1.kv_offload.cpu.gpu_worker import CpuGpuOffloadingHandlers
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.kv_offload.worker.worker import OffloadingHandler


class CPUOffloadingSpec(OffloadingSpec):
    """
    CPU KV 缓存卸载规格。

    定义 CPU 卸载的配置、容量计算，并提供创建管理器和处理器的方法。

    在初始化时计算：
    - 每个卸载块的 KV 字节数
    - CPU 可容纳的卸载块总数
    - 每个工作器的 CPU 页大小
    """

    def __init__(self, vllm_config: VllmConfig, kv_cache_config: KVCacheConfig):
        super().__init__(vllm_config, kv_cache_config)

        # 从配置中获取 CPU 可用字节数
        cpu_bytes_to_use = self.extra_config.get("cpu_bytes_to_use")
        if not cpu_bytes_to_use:
            raise Exception(
                "cpu_bytes_to_use must be specified in kv_connector_extra_config"
            )

        # 计算每个卸载块的 KV 字节数
        assert kv_cache_config is not None
        if kv_cache_config.num_blocks > 0:
            # 总 GPU KV 字节数 / GPU 块数 = 每块的 KV 字节数
            # 乘以 world_size 是因为分布式部署时每块需要存储所有工作器的数据
            total_gpu_kv_bytes = sum(t.size for t in kv_cache_config.kv_cache_tensors)
            kv_bytes_per_block = (
                total_gpu_kv_bytes // kv_cache_config.num_blocks
            ) * vllm_config.parallel_config.world_size
        else:
            kv_bytes_per_block = 0

        # 卸载块大小 = GPU 块大小 * block_size_factor
        # block_size_factor 通常大于 1，因为 CPU 块可以更大以提高传输效率
        kv_bytes_per_offloaded_block = kv_bytes_per_block * self.block_size_factor
        # 计算 CPU 可容纳的卸载块数
        self.num_blocks = (
            int(cpu_bytes_to_use) // kv_bytes_per_offloaded_block
            if kv_bytes_per_offloaded_block > 0
            else 0
        )
        world_size = vllm_config.parallel_config.world_size
        # 每个工作器在每个块上的页大小（字节）
        self.cpu_page_size_per_worker: int = (
            kv_bytes_per_offloaded_block // world_size if world_size > 0 else 0
        )

        # 调度器侧的管理器（懒初始化）
        self._manager: OffloadingManager | None = None

        # 工作器侧的处理器（懒初始化）
        self._handlers: CpuGpuOffloadingHandlers | None = None

        # 淘汰策略，默认 LRU
        self.eviction_policy: str = self.extra_config.get("eviction_policy", "lru")

    def get_manager(self) -> OffloadingManager:
        """
        获取或创建调度器侧的卸载管理器。

        懒初始化 CPUOffloadingManager，配置从 extra_config 中读取。

        Returns:
            OffloadingManager 实例
        """
        if not self._manager:
            kv_events_config = self.vllm_config.kv_events_config
            enable_events = (
                kv_events_config is not None and kv_events_config.enable_kv_cache_events
            )

            # store_threshold: 块需要在 lookup() 中出现多少次后才允许 CPU 卸载。
            # 值 < 2 禁用过滤（阈值 1 等于无过滤；0 是默认值）。
            store_threshold = int(self.extra_config.get("store_threshold", 0))

            # 内部跟踪器 LRU 表的最大条目数。
            max_tracker_size = int(self.extra_config.get("max_tracker_size", 64_000))

            self._manager = CPUOffloadingManager(
                num_blocks=self.num_blocks,
                cache_policy=self.eviction_policy,  # type: ignore[arg-type]
                enable_events=enable_events,
                store_threshold=store_threshold,
                max_tracker_size=max_tracker_size,
            )
        return self._manager

    def create_handlers(self, kv_caches: CanonicalKVCaches) -> CpuGpuOffloadingHandlers:
        """
        创建工作器侧的传输处理器。

        Args:
            kv_caches: KV 缓存规范信息

        Returns:
            CpuGpuOffloadingHandlers 实例
        """
        return CpuGpuOffloadingHandlers(
            kv_caches=kv_caches,
            block_size_factor=self.block_size_factor,
            num_cpu_blocks=self.num_blocks,
        )

    def get_handlers(
        self, kv_caches: CanonicalKVCaches
    ) -> Iterator[tuple[type[LoadStoreSpec], type[LoadStoreSpec], OffloadingHandler]]:
        """
        获取工作器侧的传输处理器迭代器。

        懒初始化处理器，并 yield 两个方向的处理器：
        1. (GPULoadStoreSpec, CPULoadStoreSpec, gpu_to_cpu_handler) - GPU->CPU 卸载
        2. (CPULoadStoreSpec, GPULoadStoreSpec, cpu_to_gpu_handler) - CPU->GPU 加载

        Args:
            kv_caches: KV 缓存规范信息

        Yields:
            (源 LoadStoreSpec 类型, 目标 LoadStoreSpec 类型, 处理器) 三元组

        Raises:
            Exception: 如果不在 CUDA 平台上运行
        """
        if not self._handlers:
            if not current_platform.is_cuda_alike():
                raise Exception(
                    "CPU Offloading is currently only supported on CUDA-alike GPUs"
                )
            self._handlers = self.create_handlers(kv_caches)

        assert self._handlers is not None
        # GPU -> CPU 方向（卸载）
        yield GPULoadStoreSpec, CPULoadStoreSpec, self._handlers.gpu_to_cpu_handler
        # CPU -> GPU 方向（加载）
        yield CPULoadStoreSpec, GPULoadStoreSpec, self._handlers.cpu_to_gpu_handler
