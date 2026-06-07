# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
TieringOffloadingSpec: Spec for multi-tier KV cache offloading.

This spec creates a TieringOffloadingManager with a CPU primary tier
and configurable secondary tiers (e.g., Storage, Network).

Configuration via kv_connector_extra_config:
  - cpu_bytes_to_use: (required) Bytes to allocate for CPU primary tier
  - block_size: (optional) Block size for offloaded blocks (default: GPU block size)
  - eviction_policy: (optional) Primary tier eviction policy: "lru" or
    "arc" (default: "lru")
  - secondary_tiers: (optional) List of secondary tier configurations
    Each secondary tier config is a dict with:
      - type: (required) Type of secondary tier (e.g., "example", "storage", "network")
      - Additional tier-specific parameters are passed directly to the tier
        constructor. See each tier's documentation for supported parameters.

Example configuration:
{
    "cpu_bytes_to_use": 10737418240,  # 10 GB
    "block_size": 16,
    "eviction_policy": "lru",
    "secondary_tiers": [
        {
            "type": "example",
            "custom_param": 67
        }
    ]
}

=== 中文说明：多层级 KV Cache 卸载规格文件 ===

本文件定义了 TieringOffloadingSpec，用于配置多层级（multi-tier）KV cache 卸载机制。
多层级卸载的核心思想是：将 KV cache 数据按照访问频率和容量需求，
存储在不同速度和容量的存储介质上（GPU -> CPU -> 远程存储/磁盘等）。

整体架构概述（自上而下）：
  1. GPU 层：速度最快，容量最小，是模型推理时直接读写 KV cache 的地方。
  2. CPU 主层级（Primary Tier）：速度中等，容量较大，作为 GPU 和更低层级之间的桥梁。
     CPU 主层级可以直接与 GPU 交换数据（通过 DMA/mmap），是所有卸载操作的入口。
  3. 次层级（Secondary Tier）：速度最慢，容量最大（如远程存储、磁盘等）。
     次层级不能直接访问 GPU 内存，所有数据必须经由 CPU 主层级中转。

数据流转路径：
  - 卸载（Offload）：GPU -> CPU -> 次层级（当 GPU 显存不足时触发）
  - 加载（Load）：次层级 -> CPU -> GPU（当需要重新计算时触发）

通过 kv_connector_extra_config 字典进行配置，支持以下字段：
  - cpu_bytes_to_use（必需）：CPU 主层级分配的字节数
  - block_size（可选）：卸载 block 的大小，默认使用 GPU block 大小
  - eviction_policy（可选）：主层级淘汰策略，支持 "lru"（最近最少使用）或
    "arc"（自适应替换缓存），默认为 "lru"
  - secondary_tiers（可选）：次层级配置列表，每个元素是一个字典，包含：
      - type（必需）：次层级类型（如 "example", "storage", "network"）
      - 其他参数：直接传递给对应次层级的构造函数

================================================================================
中文详细说明：本文件在 vLLM v1 多层 KV cache 卸载架构中的角色
================================================================================

【1. 文件定位】
本文件是多层卸载系统的"配置规格层"（Spec），位于调用链的最上游。
它负责将用户的 JSON 配置翻译成可运行的卸载管理器实例。
在 vLLM v1 的组件层次中，Spec 类扮演"装配者"的角色：
  - 它不负责运行时的 block 调度（那是 TieringOffloadingManager 的职责）
  - 它不负责具体的 I/O 操作（那是各二级层管理器的职责）
  - 它只负责一次性地把所有组件正确地创建并组装起来

【2. 在 vLLM v1 启动流程中的位置】
vLLM 启动时，KV offload 相关的初始化顺序如下：
  (1) EngineCore 读取 VllmConfig，其中 kv_connector_extra_config 包含卸载配置。
  (2) 根据配置选择 OffloadingSpec 的具体子类（CPUOffloadingSpec 或本类）。
  (3) 调用 spec.get_manager() 创建 OffloadingManager（Scheduler 进程使用）。
  (4) 调用 spec.get_handlers() 创建数据搬运 handler（Worker 进程使用）。
  (5) Scheduler 在每个 engine step 中通过 manager 的 lookup/prepare_load/prepare_store
      等方法来查询和搬运 KV cache block。

【3. 与父类 CPUOffloadingSpec 的分工】
  CPUOffloadingSpec（父类）负责：
    - 解析 cpu_bytes_to_use、block_size、eviction_policy 等基础配置
    - 计算 CPU 可容纳的 block 数量（num_blocks）
    - 提供 get_handlers() 方法，为 GPU worker 创建 CPU<->GPU 数据搬运 handler
    - 创建 CPUOffloadingManager（单层级模式）

  TieringOffloadingSpec（本类）负责：
    - 继承父类的全部能力
    - 新增解析 secondary_tiers 配置列表
    - 创建 CPUPrimaryTierOffloadingManager（带 mmap 共享内存的一级层管理器）
    - 通过 SecondaryTierFactory 工厂创建所有二级层实例
    - 组装 TieringOffloadingManager（多层级管理器，协调一级层和二级层）
    - 重写 create_handlers()，使用 mmap 共享内存实现 scheduler-worker 通信

【4. mmap 共享内存机制详解】
本类中涉及两处 mmap 创建，它们共享同一块物理内存但用途不同：
  (1) Scheduler 侧 mmap（rank=None）：
      在 get_manager() 中创建，用于 CPUPrimaryTierOffloadingManager 获取 memoryview。
      这个 memoryview 会被传递给所有二级层，作为二级层读写一级层数据的通道。
  (2) Worker 侧 mmap（rank=具体 GPU 索引）：
      在 create_handlers() 中创建，用于该 GPU worker 执行 DMA 拷贝。
      由于使用相同的 instance_id 和 total_size_bytes，worker mmap 和 scheduler mmap
      映射到同一块物理内存，从而实现进程间零拷贝数据共享。

【5. 多层卸载的运行时数据流】
  卸载路径（GPU 计算完毕 -> 存储到 offload 层）：
    (1) Scheduler 调用 manager.prepare_store() -> 一级层分配 slot
    (2) Worker 通过 handler 执行 GPU->CPU DMA 传输
    (3) Scheduler 调用 manager.complete_store() -> 触发一级层到所有二级层的级联
    (4) 二级层异步读取一级层的 mmap 数据并持久化

  加载路径（从 offload 层恢复到 GPU）：
    (1) Scheduler 调用 manager.lookup() -> 一级层未命中 -> 查询二级层
    (2) 二级层命中 -> 发起提升（promote）：二级层数据写入一级层 slot
    (3) 提升完成后，Scheduler 调用 manager.prepare_load() -> 一级层准备数据
    (4) Worker 通过 handler 执行 CPU->GPU DMA 传输
    (5) Scheduler 调用 manager.complete_load() -> 释放一级层 block 的引用计数
"""


# 中文注释：导入依赖模块
# - torch：PyTorch，用于获取当前 GPU 设备索引（torch.accelerator.current_device_index()）
# - override：typing_extensions 提供的装饰器，标记方法为父类方法的重写，便于静态检查
# - VllmConfig：vLLM 全局配置对象，包含 parallel_config、instance_id 等
# - init_logger：vLLM 的日志初始化工具
# - KVCacheConfig：KV cache 配置接口，描述 block 数量、大小等
# - CanonicalKVCaches：标准化的 KV cache 张量集合（各层的 K/V tensor）
# - OffloadingManager：卸载管理器的抽象基类，定义 lookup/prepare_load/prepare_store 等接口
# - CpuGpuOffloadingHandlers：CPU-GPU 之间数据搬运的 handler（含 DMA 拷贝操作）
# - SharedOffloadRegion：共享内存映射区域，CPU 与 GPU worker 通过 mmap 共享数据
# - CPUOffloadingSpec：CPU 单层级卸载的规格基类，本类在此基础上扩展多层级支持
# - SecondaryTierFactory：二级层工厂，根据配置创建不同类型的二级层实例
# - CPUPrimaryTierOffloadingManager：CPU 主层级卸载管理器，带 mmap 共享内存支持
# - TieringOffloadingManager：多层级卸载管理器，协调主层级和二级层之间的数据流转

import torch
from typing_extensions import override

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.kv_offload.base import CanonicalKVCaches, OffloadingManager
from vllm.v1.kv_offload.cpu.gpu_worker import CpuGpuOffloadingHandlers
from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion
from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec
from vllm.v1.kv_offload.tiering.factory import SecondaryTierFactory
from vllm.v1.kv_offload.tiering.manager import (
    CPUPrimaryTierOffloadingManager,
    TieringOffloadingManager,
)

logger = init_logger(__name__)


# 中文注释：TieringOffloadingSpec 是多层级 KV cache 卸载的配置规格类。
# 它继承自 CPUOffloadingSpec（CPU 单层级卸载），在此基础上增加了二级层的支持。
#
# 继承关系：
#   OffloadingSpec（抽象基类）-> CPUOffloadingSpec（父类）-> TieringOffloadingSpec（本类）
#
# 父类 CPUOffloadingSpec 提供了：
#   (1) CPU 主层级的基本配置解析（cpu_bytes_to_use、block_size、eviction_policy 等）
#   (2) 计算 CPU 可容纳的 block 数量（num_blocks）和页大小（cpu_page_size_per_worker）
#   (3) 创建 CPU <-> GPU 数据搬运 handler 的能力（get_handlers() 方法）
#   (4) CPUOffloadingManager 的创建（单层级模式下的默认管理器）
#
# 本类 TieringOffloadingSpec 新增了：
#   (1) 解析二级层配置列表（secondary_tiers），支持多个二级层
#   (2) 创建带 mmap 共享内存的 CPUPrimaryTierOffloadingManager（替代父类的 CPUOffloadingManager）
#   (3) 通过 SecondaryTierFactory 工厂创建所有二级层实例（延迟加载）
#   (4) 组装 TieringOffloadingManager（多层级管理器，协调主层级和二级层）
#   (5) 重写 create_handlers()，使用 mmap 共享内存实现 scheduler-worker 零拷贝通信
#
# 类设计的关键点：
#   1. 主层级（CPU）是所有数据流转的必经之路：GPU <-> CPU <-> 二级层
#   2. 二级层不能直接访问 GPU，必须经由 CPU 中转
#   3. 主层级使用 mmap 共享内存，确保 scheduler 和 worker 可以高效通信
#   4. 本类是"装配者"角色：只负责一次性创建和组装组件，不参与运行时调度

class TieringOffloadingSpec(CPUOffloadingSpec):
    """
    Spec for multi-tier KV cache offloading.

    Creates a TieringOffloadingManager with:
    - Primary tier: CPU (LRU or ARC eviction policy)
    - Secondary tiers: Configurable via extra_config

    The CPU primary tier has direct GPU access and serves as the gateway for
    all GPU↔offload operations. Secondary tiers cannot directly access GPU
    memory and must transfer data through the primary tier.
    """

    def __init__(self, vllm_config: VllmConfig, kv_cache_config: KVCacheConfig):
        # 中文注释：__init__ 负责解析配置并保存状态，不创建管理器实例。
        # 管理器的创建是懒加载的，推迟到首次调用 get_manager() 时进行。
        #
        # 初始化流程分三步：
        #   步骤 1：调用父类构造函数，完成 CPU 单层级的基本初始化。
        #     父类会解析 cpu_bytes_to_use、block_size、eviction_policy 等配置，
        #     并计算 num_blocks（CPU 可容纳的 block 数量）和
        #     cpu_page_size_per_worker（每个 worker 的 CPU 页大小）。
        super().__init__(vllm_config, kv_cache_config)
        # Redeclare for mypy: parent sets this but `--follow-imports skip` hides it
        self._manager: OffloadingManager | None = None

        # 步骤 2：从 extra_config 中解析二级层配置列表。
        # secondary_tiers 是一个列表，每个元素是一个描述二级层的字典。
        # 例如：[{"type": "fs", "base_dir": "/data"}, {"type": "example"}]
        # 如果用户未配置 secondary_tiers，默认为空列表（即只有 CPU 主层级，无二级层）。
        # Parse secondary tier configurations
        self.secondary_tier_configs = self.extra_config.get("secondary_tiers", [])
        if not isinstance(self.secondary_tier_configs, list):
            raise ValueError("secondary_tiers must be a list of tier configurations")

        # 步骤 3：初始化 scheduler 侧的 mmap 共享内存区域引用。
        # 此处只声明类型，实际的 mmap 在 get_manager() 中创建。
        # rank=None 表示这是 scheduler 侧的映射（不关联到特定 GPU worker）。
        # 该 mmap 区域用于 scheduler 和 worker 之间共享 CPU 卸载缓冲区的数据。
        # Scheduler-side mmap (rank=None); kept for cleanup
        self._scheduler_mmap: SharedOffloadRegion | None = None

    # 中文注释：get_manager() 是获取多层级卸载管理器的核心方法。
    # 采用懒加载模式（lazy initialization），首次调用时创建管理器，后续直接返回缓存实例。
    #
    # 这个方法完成了多层级卸载系统的完整初始化流程（共 4 步）：
    #   步骤 1：创建 scheduler 侧的 mmap 共享内存区域（SharedOffloadRegion）
    #   步骤 2：创建 CPU 主层级管理器（CPUPrimaryTierOffloadingManager）
    #   步骤 3：通过工厂模式创建所有二级层（SecondaryTierFactory.create_secondary_tier）
    #   步骤 4：组装多层级管理器（TieringOffloadingManager）
    #
    # 调用时机：
    #   - Scheduler 进程启动时，EngineCore 调用 spec.get_manager() 获取管理器
    #   - 管理器随后被注入到 Scheduler 中，在每个 engine step 中被调用
    #
    # 返回值：
    #   TieringOffloadingManager 实例，实现了 OffloadingManager 接口

    @override
    def get_manager(self) -> OffloadingManager:
        """
        Get the TieringOffloadingManager.

        Creates a TieringOffloadingManager with:
        - Primary tier: CPU (LRU or ARC)
        - Secondary tiers: As configured in extra_config

        Returns:
            TieringOffloadingManager instance
        """
        if not self._manager:
            # 中文注释：检查是否启用 KV cache 事件通知。
            # KV cache 事件用于监控和调试，记录 KV cache 的分配、释放、命中等操作。
            # 事件数据可通过 vLLM 的事件系统导出到外部监控系统。
            kv_events_config = self.vllm_config.kv_events_config
            enable_events = (
                kv_events_config is not None and kv_events_config.enable_kv_cache_events
            )

            # Create scheduler-side SharedOffloadRegion (rank=None) so the
            # primary tier can eagerly create a memoryview over _base.
            #
            # 中文注释：步骤 1/4 - 创建 scheduler 侧的共享内存映射区域（SharedOffloadRegion）。
            #
            # SharedOffloadRegion 使用 mmap 在 CPU 内存中创建一块共享区域，
            # scheduler 进程和各个 GPU worker 进程都可以访问这块区域。
            # 这是实现进程间零拷贝数据共享的关键基础设施。
            #
            # 参数说明：
            #   - instance_id：实例标识，用于区分不同的 vLLM 实例（防止多个实例冲突）
            #   - total_size_bytes：总大小 = 每个 worker 的 CPU 页大小 * world_size * block 数量
            #     （world_size 是 GPU 数量，每个 GPU 的 worker 都需要独立的页空间）
            #   - num_blocks：KV cache block 总数（由 cpu_bytes_to_use / kv_bytes_per_block 计算得出）
            #   - rank=None：表示这是 scheduler 侧的映射（不关联到特定 GPU）
            #   - num_workers：GPU worker 数量（即 world_size）
            #   - cpu_page_size：每个 worker 的 CPU 页大小（kv_bytes_per_block / world_size）
            #
            # 为什么 scheduler 也需要 mmap？
            # 因为 CPUPrimaryTierOffloadingManager 运行在 scheduler 进程中，
            # 它需要通过 mmap 获取底层共享内存的 memoryview，才能将其传递给二级层。
            world_size = self.vllm_config.parallel_config.world_size
            scheduler_mmap = SharedOffloadRegion(
                instance_id=self.vllm_config.instance_id,
                total_size_bytes=self.cpu_page_size_per_worker
                * world_size
                * self.num_blocks,
                num_blocks=self.num_blocks,
                rank=None,
                num_workers=world_size,
                cpu_page_size=self.cpu_page_size_per_worker,
            )
            self._scheduler_mmap = scheduler_mmap

            # Create primary tier (CPU-based)
            #
            # 中文注释：步骤 2/4 - 创建 CPU 主层级卸载管理器（CPUPrimaryTierOffloadingManager）。
            #
            # CPUPrimaryTierOffloadingManager 继承自 CPUOffloadingManager，增加了：
            #   - mmap 共享内存支持（通过 mmap_region 参数）
            #   - memoryview 创建能力（get_kv_memoryview() 方法）
            #   - 面向二级层的 read/write 语义别名（避免与 load/store 混淆）
            #
            # 主层级是多层级架构的核心组件，负责：
            #   (a) 管理 CPU 内存中的 KV cache block（分配、淘汰、释放）
            #   (b) 执行淘汰策略（LRU 或 ARC）来决定当 CPU 内存满时淘汰哪些 block
            #   (c) 作为 GPU 和二级层之间的数据中转站
            #   (d) 通过 mmap 共享区域与 GPU worker 高效通信（零拷贝）
            #
            # 参数说明：
            #   - num_blocks：CPU 侧可容纳的 KV cache block 数量
            #     （由父类 CPUOffloadingSpec 根据 cpu_bytes_to_use 计算得出）
            #   - cache_policy：淘汰策略（"lru" 或 "arc"）
            #     LRU = 最近最少使用，ARC = 自适应替换缓存（更智能但开销更大）
            #   - enable_events：是否启用事件通知
            #   - mmap_region：步骤 1 创建的共享内存映射区域
            assert len(self.gpu_block_size) == 1
            primary_tier = CPUPrimaryTierOffloadingManager(
                num_blocks=self.num_blocks,
                cache_policy=self.eviction_policy,  # type: ignore[arg-type]
                enable_events=enable_events,
                mmap_region=scheduler_mmap,
            )

            # Create secondary tiers
            #
            # 中文注释：步骤 3/4 - 通过工厂模式创建所有二级层（Secondary Tier）。
            #
            # 二级层用于在 CPU 主层级容量不足时，将不常用的 KV cache block
            # 卸载到更慢但容量更大的存储（如本地磁盘、远程存储等）。
            #
            # 创建过程：
            #   (a) 获取主层级的 KV 内存 memoryview（primary_kv_view）
            #       这个 memoryview 是二级层读写一级层数据的唯一通道。
            #   (b) 遍历 secondary_tier_configs 配置列表
            #   (c) 使用 SecondaryTierFactory.create_secondary_tier() 根据配置创建二级层实例
            #       工厂会根据配置中的 "type" 字段（如 "fs"、"example"）选择对应实现。
            #   (d) 工厂内部使用延迟加载（importlib），首次创建时才导入实现模块。
            #
            # 注意：二级层通过 primary_kv_view 与主层级共享数据视图。
            # 由于底层是 mmap 共享内存，二级层可以直接通过 view[block_id] 读写
            # 主层级管理的 CPU 内存区域，无需额外的数据拷贝。
            primary_kv_view = primary_tier.get_kv_memoryview()
            secondary_tiers = []
            for i, tier_config in enumerate(self.secondary_tier_configs):
                try:
                    tier = SecondaryTierFactory.create_secondary_tier(
                        tier_config, primary_kv_view, self
                    )
                    secondary_tiers.append(tier)
                    logger.info(
                        "Created secondary tier #%d (%s)",
                        i,
                        tier.tier_type,
                    )
                except Exception as e:
                    logger.error(
                        "Failed to create secondary tier from config %s: %s",
                        tier_config,
                        e,
                    )
                    raise

            # Create TieringOffloadingManager. GPU↔CPU transfers use the inherited
            # get_handlers(); secondary tier transfers are handled by the
            # secondary tier managers and need no additional handlers here.
            #
            # 中文注释：步骤 4/4 - 创建多层级卸载管理器（TieringOffloadingManager）。
            #
            # TieringOffloadingManager 是整个多层级卸载系统的顶层协调者，运行在 Scheduler 进程中。
            # 它实现了 OffloadingManager 接口，Scheduler 在每个 engine step 中通过
            # lookup/prepare_load/prepare_store/complete_load/complete_store 等方法与之交互。
            #
            # 管理器负责：
            #   (a) 协调主层级和二级层之间的数据流转（级联和提升）
            #   (b) 决定何时将 block 从主层级级联到二级层（complete_store 时自动触发）
            #   (c) 决定何时从二级层提升 block 回主层级（lookup 命中时自动触发）
            #   (d) 管理整个生命周期（分配、淘汰、释放、引用计数保护）
            #   (e) 批量处理待提交的提升请求（减少 I/O 开销）
            #
            # 数据传输路径说明：
            #   - GPU <-> CPU 的数据搬运：使用父类 CPUOffloadingSpec.get_handlers()
            #     创建的 CpuGpuOffloadingHandlers，包含 DMA 拷贝操作。
            #     这部分在 Worker 进程中执行。
            #   - CPU <-> 二级层的数据搬运：由各二级层管理器自行处理
            #     （通过各自的内部机制，如文件 I/O、网络传输等）。
            #     这部分在 Scheduler 进程中通过异步任务管理。
            tiering_manager = TieringOffloadingManager(
                primary_tier=primary_tier,
                secondary_tiers=secondary_tiers,
                enable_events=enable_events,
            )
            # 中文注释：store_threshold 参数在多层级模式下不支持。
            # store_threshold 用于控制 block 被 lookup 多少次后才存储到 offload 目标，
            # 在单层级 CPU 模式下可以使用，但多层级模式有自己的分层淘汰策略，
            # 因此强制禁用此参数以避免策略冲突。
            if int(self.extra_config.get("store_threshold", 0)) >= 2:
                raise ValueError(
                    "store_threshold is not supported for TieringOffloadingSpec"
                )
            self._manager = tiering_manager

            logger.info(
                "Created TieringOffloadingManager with primary tier "
                "(%s, %s blocks) and %s secondary tier(s)",
                self.eviction_policy,
                self.num_blocks,
                len(secondary_tiers),
            )

        return self._manager

    # 中文注释：create_handlers() 为当前 GPU worker 创建 CPU-GPU 数据搬运的 handler。
    #
    # 该方法在每个 GPU worker 进程中被调用（不是在 scheduler 进程中），因此：
    #   - 使用 torch.accelerator.current_device_index() 获取当前 GPU 的 rank
    #   - 创建该 worker 专属的 SharedOffloadRegion（rank=当前 GPU 索引）
    #
    # 与 get_manager() 中创建的 scheduler 侧 mmap（rank=None）的关系：
    #   - scheduler 侧 mmap（rank=None）：在 get_manager() 中创建，用于主层级管理器获取 memoryview
    #   - worker 侧 mmap（rank=具体 GPU 索引）：在本方法中创建，用于该 worker 执行 GPU DMA 传输
    #   - 两者使用相同的 instance_id 和 total_size_bytes，因此映射到同一块物理内存
    #   - 这就是 scheduler 和 worker 之间零拷贝通信的基础
    #
    # 返回的 CpuGpuOffloadingHandlers 封装了：
    #   (1) kv_caches：GPU KV cache 张量引用（各层的 K/V tensor）
    #   (2) block_size_factor：block 大小因子（卸载 block 与 GPU block 的大小比）
    #   (3) num_cpu_blocks：CPU 侧可容纳的 block 数量
    #   (4) mmap_region：mmap 共享内存区域（worker 侧映射）
    # 这些信息使得 worker 可以在 GPU 和 CPU 之间高效地拷贝 KV cache block 数据。
    #
    # 调用时机：
    #   Worker 进程启动时，调用 spec.get_handlers(kv_caches) 获取 handler。
    #   handler 随后被注入到 OffloadingHandler 中，在每个 engine step 中被调度执行。

    @override
    def create_handlers(self, kv_caches: CanonicalKVCaches) -> CpuGpuOffloadingHandlers:
        world_size = self.vllm_config.parallel_config.world_size
        rank = torch.accelerator.current_device_index()
        # 中文注释：创建 worker 侧的 mmap 共享内存区域。
        # 参数与 scheduler 侧完全一致（instance_id、total_size_bytes、num_blocks），
        # 唯一不同的是 rank=具体 GPU 索引（而非 None）。
        # 由于 mmap 使用相同的路径/标识，worker 映射和 scheduler 映射指向同一块物理内存。
        worker_mmap = SharedOffloadRegion(
            instance_id=self.vllm_config.instance_id,
            total_size_bytes=self.cpu_page_size_per_worker
            * world_size
            * self.num_blocks,
            num_blocks=self.num_blocks,
            rank=rank,
            num_workers=world_size,
            cpu_page_size=self.cpu_page_size_per_worker,
        )
        # 中文注释：构造 CpuGpuOffloadingHandlers，封装 GPU<->CPU 数据搬运所需的所有信息。
        # Worker 通过 handler.gpu_to_cpu_handler 和 handler.cpu_to_gpu_handler
        # 分别执行 GPU->CPU（卸载）和 CPU->GPU（加载）的 DMA 传输。
        return CpuGpuOffloadingHandlers(
            kv_caches=kv_caches,
            block_size_factor=self.block_size_factor,
            num_cpu_blocks=self.num_blocks,
            mmap_region=worker_mmap,
        )
