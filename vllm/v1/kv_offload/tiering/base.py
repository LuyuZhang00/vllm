# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Abstract interfaces and data types for the secondary tiering layer.
"""
# =============================================================================
# 二级存储层（Secondary Tier）的抽象接口和数据类型定义
#
# 本模块定义了 vLLM v1 多级 KV cache 卸载架构中"二级存储层"的核心抽象。
# 在多级卸载架构中，数据流如下：
#   - 存储（Store/级联）：GPU -> CPU（一级层/Primary） -> 二级层（Secondary）
#   - 加载（Load/提升）：二级层 -> CPU（一级层） -> GPU
#
# 二级层不能直接访问 GPU 显存，所有数据必须经过 CPU 一级层中转。
# 二级层的典型实现包括：文件系统（fs）、网络存储、NVMe SSD 等。
#
# 关键设计：
#   1. 所有方法都在 Scheduler 进程中运行，必须轻量且非阻塞。
#   2. submit_load()/submit_store() 提交异步传输任务，
#      get_finished_jobs() 轮询任务完成情况。
#   3. 每个二级层通过 OffloadKey（block hash + group index）标识 block。
# =============================================================================
#
# ========================= 文件功能概述 =========================
#
# 本文件是 vLLM v1 多级 KV cache 卸载架构的"二级存储层"抽象基类定义文件。
# 它不包含具体实现，只定义了接口契约和数据类型，供具体二级层实现继承。
#
# 文件包含以下核心组件：
#
# 【1. 数据类型】
#   - JobId (int): 异步传输任务的唯一标识符。
#   - JobMetadata (dataclass): 描述一个 in-flight 传输任务的元数据，
#     包含任务 ID、涉及的 block 键集合、一级层物理 block ID、传输方向等。
#   - JobResult (dataclass): 传输任务完成结果，包含 job_id 和 success 标志。
#
# 【2. 抽象基类】
#   - SecondaryTierManager (ABC): 二级存储层管理器的抽象接口，
#     定义了 lookup/submit_store/submit_load/get_finished_jobs 等核心方法。
#
# 【3. 多级卸载架构总览】
#   vLLM v1 采用三级 KV cache 存储架构：
#     第一级：GPU 显存（HBM）—— 用于当前推理计算
#     第二级：CPU 内存（Primary tier）—— 一级缓存，可直接被 GPU 访问
#     第三级：二级存储（Secondary tier）—— 长期存储，如文件系统、NVMe、网络存储
#
#   数据流向：
#     存储（Cascade）：GPU -> CPU(Primary) -> Secondary
#       当 GPU 产生新的 KV block 时，先写入 CPU 一级层，再级联到二级层。
#     加载（Promotion）：Secondary -> CPU(Primary) -> GPU
#       当 GPU 需要某个 block 但二级层有缓存时，先提升到 CPU 一级层，
#       再从一级层加载到 GPU。
#
# 【4. 异步任务模型】
#   所有数据传输都是异步的：
#     1. 调用 submit_store() 或 submit_load() 提交传输任务，立即返回。
#     2. 后台线程/进程执行实际数据传输。
#     3. 调用 get_finished_jobs() 轮询已完成的任务。
#     4. 框架根据 JobResult 释放资源、更新状态。
#
# 【5. 设计约束】
#   - 所有方法在 Scheduler 进程中执行，必须非阻塞。
#   - 二级层不能直接访问 GPU 显存，必须经由 CPU 一级层中转。
#   - block 通过 OffloadKey（block hash + group index）唯一标识。
#   - 一级层 block 通过 ref_cnt（引用计数）保护不被淘汰。
# =============================================================================

from abc import ABC, abstractmethod
from collections.abc import Collection, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from vllm.v1.kv_offload.base import OffloadKey, ReqContext, RequestOffloadingContext

if TYPE_CHECKING:
    from vllm.v1.kv_offload.base import OffloadingSpec

# Type alias for job IDs used in async transfer tracking
# 中文注释：异步传输任务的唯一标识符类型，用于追踪每个 in-flight 的数据传输任务。
JobId = int


@dataclass
class JobMetadata:
    """Metadata for an in-flight async transfer job."""
    # 中文注释：异步传输任务的元数据，描述一个正在进行的数据传输任务。
    # 包含了该任务所需的所有信息：任务 ID、涉及的 block 键集合、
    # 一级层中的物理 block ID、传输方向（提升/级联）以及请求上下文。
    # 这些元数据在任务提交时创建，在任务完成时通过 get_finished_jobs()
    # 返回的 JobResult 关联回来。
    #
    # 【创建时机】
    #   - submit_store()：框架在级联存储时创建，block_ids 来自一级层分配。
    #   - submit_load()：框架在提升加载时创建，block_ids 来自一级层准备好的 slot。
    #
    # 【生命周期】
    #   1. 框架调用 submit_store/submit_load 时传入 JobMetadata。
    #   2. 二级层实现将其保存，用于后台传输线程。
    #   3. 传输完成后，二级层通过 get_finished_jobs() 返回 JobResult。
    #   4. 框架通过 job_id 关联 JobMetadata 和 JobResult，执行资源清理。

    job_id: JobId
    # 中文注释：该传输任务涉及的所有 block 的 OffloadKey 集合。
    # 用于在任务完成后执行引用计数释放、资源清理等操作。
    keys: Collection[OffloadKey]
    # 中文注释：一级层（CPU primary tier）中对应的物理 block ID 数组。
    # 对于存储（级联）操作：从这些 block 读取数据到二级层；
    # 对于加载（提升）操作：将数据写入这些 block。
    # 这些 block ID 由一级层的 prepare_read/prepare_write 分配，
    # 并在传输期间通过 ref_cnt 保护其不被淘汰。
    block_ids: np.ndarray
    # 中文注释：传输方向标志。
    # True 表示提升（promotion）：二级层 -> 一级层（用于后续 GPU 加载）
    # False 表示级联（cascade）：一级层 -> 二级层（用于持久化保存）
    is_promotion: bool
    # 中文注释：请求上下文，包含请求 ID 和 KV 传输参数。
    # 用于在任务完成后将完成事件关联到正确的请求。
    req_context: ReqContext


@dataclass
class JobResult:
    """Result of an async transfer job (successful or failed)."""
    # 中文注释：异步传输任务的完成结果。
    # 由二级层的 get_finished_jobs() 返回，框架根据此结果
    # 决定是提交成功（释放引用计数、使 block 可用）还是处理失败。
    #
    # 【使用场景】
    #   - 级联（store）成功：释放一级层 block 的引用计数，标记 block
    #     已持久化到二级层，后续可从二级层恢复。
    #   - 级联（store）失败：释放引用计数，但不标记为已持久化。
    #   - 提升（load）成功：标记一级层 block 数据已就绪，可供 GPU 加载。
    #   - 提升（load）失败：标记 block 需要从 GPU 重新计算。

    job_id: JobId
    # 中文注释：任务是否成功完成。
    # 成功时框架会正确释放资源；失败时框架会进行相应的清理。
    success: bool


class SecondaryTierManager(ABC):
    """
    Abstract interface for managing a single non-primary offloading tier.

    Secondary tiers cannot directly access GPU memory. All data transfers
    must go through the CPU (primary) tier:
      - Store: GPU → CPU (primary) → secondary  (cascade)
      - Load:  secondary → CPU (primary) → GPU  (promotion)

    IMPORTANT: All methods run in the Scheduler process and must be
    lightweight and non-blocking. submit_load() and submit_store() submit
    async jobs; get_finished_jobs() polls for completion.
    """
    # 中文注释：二级存储层管理器的抽象基类。
    #
    # 在多级 KV cache 卸载架构中，SecondaryTierManager 管理一个非主级（non-primary）
    # 的存储层。典型实现包括：文件系统、远程存储、NVMe SSD 等。
    #
    # 核心约束：
    #   1. 二级层不能直接访问 GPU 显存，所有数据必须经由 CPU 一级层中转。
    #   2. 数据流方向：
    #      - Store（级联）：GPU -> CPU(Primary) -> Secondary
    #      - Load（提升）：Secondary -> CPU(Primary) -> GPU
    #   3. 所有方法在 Scheduler 进程中执行，必须非阻塞。
    #
    # 【生命周期】
    #   1. Scheduler 通过 lookup() 查询 block 是否存在于二级层。
    #   2. 命中时，调用 submit_load() 发起异步提升（promotion）。
    #   3. 新 block 从 GPU 存入一级层后，框架调用 submit_store() 级联到二级层。
    #   4. 通过 get_finished_jobs() 轮询任务完成，框架据此释放资源。
    #
    # 【子类实现要求】
    #   必须实现以下抽象方法：
    #     - lookup(): 查询 block 是否存在
    #     - submit_store(): 提交级联存储任务
    #     - submit_load(): 提交提升加载任务
    #     - get_finished_jobs(): 返回已完成的任务
    #     - on_new_request(): 新请求到达时的回调
    #
    #   可选覆盖：
    #     - touch(): 更新 block 访问时间（用于淘汰策略）
    #     - on_request_finished(): 请求完成时的清理回调
    #     - shutdown(): 释放资源

    def __init__(
        self,
        offloading_spec: "OffloadingSpec",
        primary_kv_view: memoryview,
        tier_type: str,
    ) -> None:
        """
        Args:
            offloading_spec: Offloading configuration.
            primary_kv_view: Memoryview of the primary tier's CPU KV cache.
            tier_type: Tier type identifier, set by SecondaryTierFactory
                from the registered tier type.
        """
        # 中文注释：保存卸载配置引用，后续用于获取 block 大小、hash 规则等参数。
        self._offloading_spec = offloading_spec
        # 中文注释：一级层 CPU KV cache 的 memoryview 视图。
        # 二级层通过此视图直接读写一级层的 block 数据（通过 block_ids 索引）。
        # 这是二级层与一级层交换数据的唯一通道。
        self._primary_kv_view: memoryview = primary_kv_view
        # 中文注释：二级层类型标识符（如 "fs"、"example"），由工厂注册时设定。
        self.tier_type = tier_type

    @abstractmethod
    def lookup(self, key: OffloadKey, req_context: ReqContext) -> bool | None:
        """
        Check whether a block exists in this secondary tier.

        Args:
            key: Offload key to look up.
            req_context: per-request context (e.g. kv_transfer_params).

        Returns:
            True if the block is present and ready,
            False if not found,
            or None if the block is being transferred (retry later).
        """
        # 中文注释：查询指定 block 是否存在于当前二级层中。
        #
        # 返回值语义：
        #   True  — block 存在且可用，可以立即发起提升（promote）。
        #   False — block 不存在，需要从 GPU 重新计算后级联存储。
        #   None  — block 正在传输中（in-flight），调用方应稍后重试。
        #           这种"透明重试"机制避免了重复提交传输任务。
        #
        # 调用时机：Scheduler 在 lookup 阶段调用，先查一级层，一级层未命中时再查二级层。
        pass

    @abstractmethod
    def submit_store(self, job_metadata: JobMetadata) -> None:
        """
        Submit an async job to store blocks from the primary tier to this
        secondary tier.

        This method must be lightweight and non-blocking: allocate metadata
        and submit the transfer, but do NOT perform the data copy on the
        calling thread.

        Preconditions (guaranteed by the framework):
          - ``job_metadata.block_ids`` are valid primary-tier slots, pinned
            (ref-counted) for the duration of the transfer.

        The implementation is responsible for:
          1. Filtering out blocks already present in this tier
          2. Evicting blocks if capacity is needed
          3. Allocating space in this tier
          4. Submitting the async transfer (read from primary via block_ids)

        Report completion via ``get_finished_jobs()``.

        Args:
            job_metadata: Job metadata including job_id, keys, and block_ids
                          identifying the primary-tier slots to read from.
        """
        # 中文注释：提交异步级联（cascade）存储任务 —— 将 block 从一级层复制到二级层。
        #
        # 数据流：CPU Primary（通过 block_ids 索引读取） -> Secondary
        #
        # 实现者职责：
        #   1. 过滤掉二级层中已存在的 block（避免重复存储）。
        #   2. 容量不足时淘汰旧 block。
        #   3. 在二级层分配存储空间。
        #   4. 提交异步数据传输（从一级层的 memoryview 读取）。
        #   5. 传输完成后通过 get_finished_jobs() 报告结果。
        #
        # 前置条件（由框架保证）：
        #   job_metadata.block_ids 指向的一级层 slot 已被引用计数保护（pinned），
        #   在传输完成前不会被淘汰。
        pass

    @abstractmethod
    def submit_load(self, job_metadata: JobMetadata) -> None:
        """
        Submit an async job to load blocks from this secondary tier to the
        primary tier.

        This method must be lightweight and non-blocking: mark blocks as
        in-flight and submit the transfer, but do NOT perform the data copy
        on the calling thread.

        Preconditions (guaranteed by the framework):
          - ``job_metadata.block_ids`` are allocated primary-tier slots
            ready to receive data.

        The implementation must copy data from this tier into the
        primary-tier slots identified by ``block_ids``.

        Report completion via ``get_finished_jobs()``.

        Args:
            job_metadata: Job metadata including job_id, keys, and block_ids
                          identifying the primary-tier slots to write into.
        """
        # 中文注释：提交异步提升（promotion）加载任务 —— 将 block 从二级层复制回一级层。
        #
        # 数据流：Secondary -> CPU Primary（通过 block_ids 索引写入）
        #
        # 这是"提升"路径的关键步骤：当 GPU 需要某个 block 的 KV 数据，
        # 而该数据只存在于二级层时，需要先提升到一级层，再从一级层加载到 GPU。
        #
        # 实现者职责：
        #   1. 从二级层读取对应 block 的数据。
        #   2. 写入一级层通过 block_ids 指定的 slot。
        #   3. 完成后通过 get_finished_jobs() 报告结果。
        #
        # 前置条件（由框架保证）：
        #   job_metadata.block_ids 指向的一级层 slot 已分配完成，准备好接收数据。
        pass

    @abstractmethod
    def get_finished_jobs(self) -> Iterable[JobResult]:
        """
        Return all jobs (loads and stores) that completed since the last call.

        The framework uses these results to release resources and finalize
        transfers.

        Returns:
            Iterable of JobResult objects for jobs finished since the
            last call.
        """
        # 中文注释：获取自上次调用以来完成的所有异步传输任务。
        #
        # 框架每步（engine step）至少调用一次此方法来轮询任务完成情况。
        # 返回的 JobResult 包含 job_id 和 success 标志，框架据此：
        #   - 对于级联（store）完成：释放一级层 block 的引用计数。
        #   - 对于提升（load）完成：标记一级层 block 可用，供 GPU 加载。
        #
        # 调用时机：由 TieringOffloadingManager._process_finished_jobs() 调用。
        pass

    def touch(self, keys: Collection[OffloadKey], req_context: ReqContext):
        """
        Mark blocks as recently used for eviction policy.

        Args:
            keys: Offload keys to mark as recently used.
            req_context: Per-request context.
        """
        # 中文注释：标记指定 block 为"最近使用"，用于淘汰策略（如 LRU）。
        # 默认空实现，子类可覆盖以维护自己的淘汰顺序。
        # 调用时机：GPU prefix cache 命中时，虽然不需要从二级层加载数据，
        # 但仍需更新这些 block 的访问时间以避免被淘汰。
        return

    @abstractmethod
    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext:
        """
        Called when a new request is first seen by the scheduler.

        Returns a RequestOffloadingContext expressing this tier's preference
        for how blocks should be offloaded for this request.

        Args:
            req_context: Per-request context.
        """
        # 中文注释：当 Scheduler 首次看到一个新请求时调用此方法。
        #
        # 二级层通过返回 RequestOffloadingContext 来表达对这个请求的卸载偏好：
        #   - BLOCK_LEVEL：只卸载新计算的 block（跳过 prefix cache 已命中的 block）。
        #   - REQUEST_LEVEL：卸载该请求的所有 block（包括 prefix 命中的 block）。
        #
        # REQUEST_LEVEL 适用于需要完整 KV 上下文的场景（如某些网络存储层）。
        pass

    def on_request_finished(self, req_context: ReqContext) -> None:
        """
        Called when a request has finished.

        Args:
            req_context: per-request context.
        """
        # 中文注释：请求完成时调用，用于清理该请求在二级层中的临时状态。
        # 默认空实现，子类可覆盖以执行资源释放。
        return

    def shutdown(self) -> None:
        """Release resources held by this tier (threads, connections, etc.)."""
        # 中文注释：关闭二级层并释放所有资源（如线程、文件句柄、网络连接等）。
        return
