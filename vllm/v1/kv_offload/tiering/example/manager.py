# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
ExampleSecondaryTierManager: A simple in-memory secondary tier.

This implementation provides a minimal secondary tier that stores blocks
in memory (using a dictionary) with immediate completion. It serves as a
reference for writing new tiers and is useful for testing the
TieringOffloadingManager without requiring actual storage or network backends.
"""

# =============================================================================
# ExampleSecondaryTierManager：一个简单的内存二级存储层示例
#
# 本模块实现了 SecondaryTierManager 的一个最简示例，用于演示如何编写
# 自定义二级存储层。它使用 Python 字典作为存储后端，所有操作同步完成
# （无异步 I/O），非常适合用于单元测试和架构理解。
#
# ==================== 多级 KV cache 卸载架构总览 ====================
#
# vLLM v1 采用多级（tiered）KV cache 卸载架构：
#
#   第 0 级：GPU 显存  ——  模型推理时直接访问，速度最快但容量最小
#   第 1 级：CPU 内存（Primary Tier）  ——  通过 mmap 管理，GPU 可通过 DMA 读写
#   第 2 级及以后：Secondary Tier  ——  如本示例（内存字典）、文件系统、NVMe、
#                                     远程存储等，速度较慢但容量大或可持久化
#
# 数据流方向：
#   存储（Store/Cascade）：GPU -> CPU(Primary) -> Secondary
#     1. GPU 计算完某个 request 的 KV cache
#     2. GPU->CPU DMA 传输到一级层（由 CPUOffloadingManager 处理）
#     3. CPU->Secondary 异步传输到二级层（由本类处理）
#
#   加载（Load/Promotion）：Secondary -> CPU(Primary) -> GPU
#     1. Scheduler 查询某个 block，一级层未命中但二级层命中
#     2. 发起提升（promotion）：二级层->CPU 一级层（由本类处理）
#     3. CPU->GPU DMA 加载到显存（由 CPUOffloadingManager 处理）
#
# ==================== 本文件在架构中的位置 ====================
#
# TieringOffloadingManager（编排器）持有：
#   - 一个 CPUPrimaryTierOffloadingManager（一级层）
#   - 零或多个 SecondaryTierManager（二级层）
#
# Scheduler 通过 TieringOffloadingManager 的 lookup()/prepare_load()/
# prepare_store()/complete_store() 等方法与多级存储交互。
# TieringOffloadingManager 内部再将二级层操作委托给本类（或其他
# SecondaryTierManager 实现）。
#
# ==================== 本示例的特点 ====================
#
# 1. 存储介质：Python 字典（key -> True），仅记录 block 是否存在。
# 2. 同步完成：submit_store/submit_load 立即完成，无异步 I/O。
#    真实实现（如文件系统、网络存储）应在线程中执行异步传输。
# 3. 无淘汰策略：不维护 LRU、不检查容量。真实实现需要管理容量和淘汰。
# 4. 无数据传输：不实际复制 KV 数据。真实实现需要通过 _primary_kv_view
#    读写一级层的 block 数据。
#
# ==================== 如何编写自定义二级层 ====================
#
# 步骤 1：继承 SecondaryTierManager
# 步骤 2：实现 lookup() —— 检查 block 是否存在于你的存储中
# 步骤 3：实现 submit_store() —— 异步将 block 从一级层写入你的存储
# 步骤 4：实现 submit_load() —— 异步将 block 从你的存储读出到一级层
# 步骤 5：实现 get_finished_jobs() —— 返回已完成的异步任务结果
# 步骤 6：实现 on_new_request() —— 返回该请求的卸载策略偏好
# 步骤 7（可选）：实现 touch()、on_request_finished()、shutdown()
#
# ==================== 核心类与方法一览 ====================
#
# 类：ExampleSecondaryTierManager（继承自 SecondaryTierManager）
#   核心方法：
#     - lookup()          查询 block 是否存在于二级层
#     - submit_store()    提交级联存储任务（CPU 一级层 -> 二级层）
#     - submit_load()     提交提升加载任务（二级层 -> CPU 一级层）
#     - get_finished_jobs() 轮询已完成的异步任务
#     - on_new_request()  返回新请求的卸载策略偏好
#     - get_num_blocks()  返回当前存储的 block 数量
#
# 关键数据结构：
#     - self.blocks         已存储 block 的存在性集合（dict[OffloadKey, bool]）
#     - self.completed_jobs 已完成任务的结果队列（list[JobResult]）
# =============================================================================

# ==================== 导入模块说明 ====================
# 1. logging —— 日志记录，用于输出初始化信息和调试信息
# 2. collections.abc.Iterable —— 用于类型标注 get_finished_jobs() 的返回值
# 3. TYPE_CHECKING —— 仅在类型检查时导入，避免运行时循环导入
#
# 关键类型说明：
#   OffloadKey —— 卸载块的唯一标识符，包含 block hash 等信息
#   ReqContext —— 请求上下文，包含请求级别的元数据
#   RequestOffloadingContext —— 请求卸载上下文，返回给框架的策略偏好
#   JobMetadata —— 传输任务的元数据，包含 job_id、keys、block_ids 等
#   JobResult —— 传输任务的结果，包含 job_id 和 success 标志
#   SecondaryTierManager —— 二级存储层管理器的抽象基类
#   OffloadingSpec —— 卸载规格配置，包含 block 大小、hash 规则等

import logging
from collections.abc import Iterable
from typing import TYPE_CHECKING

from vllm.v1.kv_offload.base import OffloadKey, ReqContext, RequestOffloadingContext
from vllm.v1.kv_offload.tiering.base import (
    JobMetadata,
    JobResult,
    SecondaryTierManager,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from vllm.v1.kv_offload.base import OffloadingSpec


class ExampleSecondaryTierManager(SecondaryTierManager):
    """
    A simple in-memory secondary tier.

    This implementation:
    - Stores blocks in a dictionary (key -> True)
    - Completes transfers immediately (synchronous)
    """
    # 中文注释：示例二级存储层管理器，继承自 SecondaryTierManager 抽象基类。
    #
    # 这是最简单的二级层实现，用于演示和测试。它将所有 block 存储在
    # Python 字典中，所有操作同步完成（不涉及真实 I/O）。
    #
    # 真实的二级层实现（如文件系统、远程存储）需要注意：
    #   1. submit_store() 和 submit_load() 必须是非阻塞的，
    #      应在后台线程中执行实际的数据搬运。
    #   2. 需要通过 self._primary_kv_view[block_id] 来读写一级层的数据。
    #   3. 需要实现容量管理和淘汰策略（如 LRU）。
    #   4. get_finished_jobs() 应返回自上次调用以来完成的所有任务。
    #
    # ==================== 类的核心职责 ====================
    #
    # 本类的核心职责是管理二级存储层中的 KV cache block，具体包括：
    #
    # 1. 查询管理（lookup）：
    #    - 回答"某个 block 是否在二级层中"的查询
    #    - 供 TieringOffloadingManager 在一级层未命中时调用
    #
    # 2. 级联存储管理（submit_store）：
    #    - 接收从一级层（CPU 内存）级联过来的 block
    #    - 在真实实现中应异步复制 KV 数据到二级存储
    #    - 完成后通过 get_finished_jobs() 报告结果
    #
    # 3. 提升加载管理（submit_load）：
    #    - 将 block 从二级层提升回一级层（CPU 内存）
    #    - 在真实实现中应异步从二级存储读取数据并写入一级层
    #    - 完成后通过 get_finished_jobs() 报告结果
    #
    # 4. 任务完成报告（get_finished_jobs）：
    #    - 返回自上次调用以来完成的所有异步传输任务
    #    - 框架根据结果执行资源清理（释放引用计数等）
    #
    # ==================== 与框架的交互流程 ====================
    #
    # 场景 A：级联存储（block 从 CPU 一级层 -> 二级层）
    #   Step 1: GPU 计算完 KV cache，通过 DMA 传到 CPU 一级层
    #   Step 2: TieringOffloadingManager.complete_store() 被调用
    #   Step 3: 框架为每个二级层调用 submit_store()
    #   Step 4: 本类异步执行存储（示例中同步完成）
    #   Step 5: get_finished_jobs() 返回成功结果
    #   Step 6: 框架调用 primary.complete_read() 释放一级层引用
    #
    # 场景 B：提升加载（block 从二级层 -> CPU 一级层 -> GPU）
    #   Step 1: Scheduler 查询 block，一级层未命中但二级层命中
    #   Step 2: TieringOffloadingManager._initiate_promotion() 在一级层分配 slot
    #   Step 3: _flush_pending_promotions() 调用 submit_load()
    #   Step 4: 本类异步执行加载（示例中同步完成）
    #   Step 5: get_finished_jobs() 返回成功结果
    #   Step 6: 框架调用 primary.complete_write() 使一级层 slot 可用
    #   Step 7: 后续 prepare_load() 将数据从一级层加载到 GPU

    def __init__(
        self,
        offloading_spec: "OffloadingSpec",
        primary_kv_view: memoryview,
        tier_type: str,
        custom_param: int = 0,
    ):
        """
        Initialize the example secondary tier.

        Args:
            custom_param: Dummy parameter demonstrating custom args.
        """
        # 中文注释：构造函数，初始化二级存储层管理器。
        #
        # ==================== 参数说明 ====================
        #
        # offloading_spec: OffloadingSpec
        #   卸载规格配置对象，包含：
        #   - block_size: 每个 KV block 包含的 token 数
        #   - hash_algorithm: block 内容的哈希算法
        #   - 其他与卸载相关的配置
        #
        # primary_kv_view: memoryview
        #   一级层（CPU 内存）KV cache 的 memoryview 视图。
        #   这是二级层读写一级层数据的唯一通道：
        #   - 级联存储时：通过 primary_kv_view[block_id] 读取一级层数据
        #   - 提升加载时：通过 primary_kv_view[block_id] 写入一级层数据
        #   注意：此示例中未实际使用（不复制真实数据）。
        #
        # tier_type: str
        #   二级层的类型标识符，用于框架区分不同的二级层实现。
        #   例如 "filesystem"、"nvme"、"remote_storage" 等。
        #
        # custom_param: int
        #   自定义参数示例，演示如何向二级层传递配置参数。
        #   真实实现可使用此类参数配置存储路径、并发数等。
        #
        # ==================== 初始化流程 ====================
        #
        # 1. 调用父类构造函数，保存 offloading_spec、primary_kv_view、tier_type
        # 2. 初始化 self.blocks 字典（存储 block 存在性信息）
        # 3. 初始化 self.completed_jobs 列表（存储已完成的任务结果）

        # 中文注释：调用父类构造函数，保存 offloading_spec（包含 block 大小、
        # hash 规则等配置）、primary_kv_view（一级层 KV cache 的 memoryview，
        # 是二级层读写一级层数据的唯一通道）和 tier_type（类型标识符）。
        super().__init__(
            offloading_spec=offloading_spec,
            primary_kv_view=primary_kv_view,
            tier_type=tier_type,
        )

        logger.info(
            "ExampleSecondaryTierManager initialized with custom_param=%d", custom_param
        )

        # key -> True (only care about presence)
        # 中文注释：存储所有已级联到本二级层的 block 的 OffloadKey。
        # 字典值统一为 True，仅用于判断 block 是否存在（presence set）。
        # 真实实现中此处应存储实际的 KV 数据或文件路径等信息。
        # OffloadKey 是 block 的唯一标识符，通常包含 block hash 等信息，
        # 确保不同请求的相同内容 block 可以复用（prefix caching 的基础）。
        self.blocks: dict[OffloadKey, bool] = {}

        # Completed jobs waiting to be retrieved by get_finished_jobs()
        # 中文注释：已完成的异步传输任务列表。由于本示例是同步完成的，
        # submit_store/submit_load 会立即将 JobResult 追加到此列表。
        # get_finished_jobs() 会一次性取出并清空此列表，返回给上层框架。
        # 真实的异步实现中，后台线程完成传输后会将结果追加到此列表。
        #
        # JobResult 包含：
        #   - job_id: 任务的唯一标识符
        #   - success: 任务是否成功完成
        #   - is_promotion: 是否为提升任务（由框架在创建 JobMetadata 时设置）
        self.completed_jobs: list[JobResult] = []

    def lookup(self, key: OffloadKey, req_context: ReqContext) -> bool | None:
        """
        Check whether a block exists in this secondary tier.

        Args:
            key: Offload key to look up.
            req_context: Per-request context.

        Returns:
            True if the block is present, False if not found.
        """
        # 中文注释：查询指定 block 是否存在于本二级层中。
        #
        # ==================== 调用时机 ====================
        #
        # 此方法由 TieringOffloadingManager.lookup() 在一级层未命中时调用。
        # 完整的查询链路：
        #   1. Scheduler 需要某个 block 的 KV cache
        #   2. TieringOffloadingManager.lookup() 先查一级层（CPU 内存）
        #   3. 一级层未命中时，遍历所有二级层调用 lookup()
        #   4. 如果某个二级层返回 True，框架发起提升（promotion）
        #
        # ==================== 参数说明 ====================
        #
        # key: OffloadKey
        #   block 的唯一标识符，包含 block hash 等信息。
        #   不同请求的相同内容 block 会映射到相同的 OffloadKey，
        #   这是 prefix caching（前缀缓存）的基础。
        #
        # req_context: ReqContext
        #   请求上下文，包含请求级别的元数据。
        #   本示例中未使用，但真实实现可能需要它来：
        #   - 追踪每个请求的 block 访问模式
        #   - 实现请求级别的缓存策略
        #   - 记录访问统计信息
        #
        # ==================== 返回值语义 ====================
        #
        # （由 SecondaryTierManager 接口定义）
        #   True  — block 存在且可用，TieringOffloadingManager 会发起提升。
        #   False — block 不存在，TieringOffloadingManager 继续查下一个二级层。
        #   None  — block 正在传输中（in-flight），调用方应稍后重试。
        #
        # 本示例中不会返回 None，因为所有操作都是同步完成的。
        # 真实的异步实现中，如果 block 正在从远程存储加载中，应返回 None。
        # 这种"三态返回值"设计允许框架区分"不存在"和"正在加载中"两种状态，
        # 从而决定是发起新的加载还是等待现有加载完成。
        #
        # ==================== 性能考量 ====================
        #
        # 由于本示例使用字典存储，查找时间复杂度为 O(1)。
        # 真实实现中，查找可能涉及：
        #   - 文件系统：检查文件是否存在（可能有 I/O 开销）
        #   - 远程存储：发送查询请求（网络延迟）
        #   - NVMe：查询索引结构（通常较快）
        return key in self.blocks

    def submit_store(self, job_metadata: JobMetadata) -> None:
        """
        Submit a job to store blocks from primary tier to this tier.

        Args:
            job_metadata: Job metadata including job_id, keys, and
                          spec for reading blocks from the primary tier.
        """
        # 中文注释：提交级联（cascade）存储任务 —— 将 block 从一级层复制到本二级层。
        #
        # ==================== 调用时机 ====================
        #
        # 当 GPU->CPU 传输完成后（TieringOffloadingManager.complete_store()），
        # 框架会为每个二级层调用此方法，将新计算的 KV block 级联到所有二级层。
        #
        # 完整的级联存储流程：
        #   1. GPU 计算完某个 request 的 KV cache
        #   2. GPU->CPU DMA 传输到一级层（由 CPUOffloadingManager 处理）
        #   3. TieringOffloadingManager.complete_store() 被调用
        #   4. 框架为每个二级层调用 submit_store()
        #   5. 本类异步执行存储（示例中同步完成）
        #   6. get_finished_jobs() 返回成功结果
        #   7. 框架调用 primary.complete_read() 释放一级层引用
        #
        # ==================== 数据流 ====================
        #
        # CPU Primary -> Secondary
        # 在真实实现中：
        #   - 通过 self._primary_kv_view[block_id] 读取一级层数据
        #   - 将数据写入二级层存储（文件系统、远程存储等）
        #
        # ==================== 参数说明 ====================
        #
        # job_metadata: JobMetadata
        #   传输任务的元数据，包含：
        #   - job_id: 任务的唯一标识符
        #   - keys: List[OffloadKey]，待存储 block 的唯一标识符列表
        #   - block_ids: List[int]，一级层中对应的 block 位置索引
        #   - is_promotion: 是否为提升任务（此处为 False）
        #
        # ==================== 本示例的简化处理 ====================
        #
        # 1. 不实际复制 KV 数据（真实实现应通过 self._primary_kv_view[block_id]
        #    读取一级层数据，然后写入二级层存储）。
        # 2. 同步完成（真实实现应在线程中异步执行传输，完成后通过
        #    get_finished_jobs() 报告结果）。
        # 3. 不检查容量、不淘汰旧 block（真实实现需要 LRU 等淘汰策略）。
        #
        # ==================== 前置条件 ====================
        #
        # （由框架保证）
        # job_metadata.block_ids 指向的一级层 slot 已被 prepare_read()
        # 增加了引用计数（ref_cnt），在传输完成前不会被淘汰。
        # 这确保了在异步传输过程中，一级层的数据不会被其他任务覆盖。

        keys = job_metadata.keys
        block_ids = job_metadata.block_ids

        assert len(keys) == len(block_ids), (
            f"Length mismatch: {len(keys)} keys but {len(block_ids)} block_ids"
        )

        # 中文注释：将所有 block 标记为已存储。本示例仅记录 key 的存在性。
        # 真实实现中此处应执行实际的数据复制操作。
        for key in keys:
            self.blocks[key] = True
        # 中文注释：同步完成——立即将成功结果追加到完成队列。
        # get_finished_jobs() 将在下一个 engine step 被调用时取出此结果。
        # TieringOffloadingManager 收到成功结果后会调用
        # primary_tier.complete_read() 释放一级层 block 的引用计数。
        self.completed_jobs.append(JobResult(job_id=job_metadata.job_id, success=True))

    def submit_load(self, job_metadata: JobMetadata) -> None:
        """
        Submit a job to load blocks from this tier to primary tier.

        Args:
            job_metadata: Job metadata including job_id, keys, and
                          spec for writing blocks into the primary tier.
        """
        # 中文注释：提交提升（promotion）加载任务 —— 将 block 从本二级层复制回一级层。
        #
        # ==================== 调用时机 ====================
        #
        # 当 Scheduler 查询某个 block，一级层未命中但本二级层命中时，
        # TieringOffloadingManager._initiate_promotion() 会先在一级层分配 slot，
        # 然后在 engine step 结束时通过 _flush_pending_promotions() 批量调用此方法。
        #
        # 完整的提升加载流程：
        #   1. Scheduler 需要某个 block 的 KV cache
        #   2. TieringOffloadingManager.lookup() 查一级层未命中
        #   3. 遍历二级层，本类 lookup() 返回 True（block 存在）
        #   4. _initiate_promotion() 在一级层分配 slot（prepare_write()）
        #   5. _flush_pending_promotions() 调用 submit_load()
        #   6. 本类异步执行加载（示例中同步完成）
        #   7. get_finished_jobs() 返回成功结果
        #   8. 框架调用 primary.complete_write() 使一级层 slot 可用
        #   9. 后续 prepare_load() 将数据从一级层加载到 GPU
        #
        # ==================== 数据流 ====================
        #
        # Secondary -> CPU Primary
        # 在真实实现中：
        #   - 从二级层存储读取 KV 数据
        #   - 通过 self._primary_kv_view[block_id] 写入一级层 slot
        #
        # ==================== 参数说明 ====================
        #
        # job_metadata: JobMetadata
        #   传输任务的元数据，包含：
        #   - job_id: 任务的唯一标识符
        #   - keys: List[OffloadKey]，待加载 block 的唯一标识符列表
        #   - block_ids: List[int]，一级层中已分配的 block 位置索引
        #   - is_promotion: 是否为提升任务（此处为 True）
        #
        # ==================== 本示例的简化处理 ====================
        #
        # 1. 不实际复制 KV 数据（真实实现应从二级层存储读取数据，
        #    通过 self._primary_kv_view[block_id] 写入一级层 slot）。
        # 2. 同步完成（真实实现应在线程中异步执行传输）。
        # 3. 如果 block 不存在则报告失败（真实实现可能需要重试或报错）。
        #
        # ==================== 前置条件 ====================
        #
        # （由框架保证）
        # job_metadata.block_ids 指向的一级层 slot 已由 prepare_write() 分配，
        # 处于 in-flight 状态（ref_cnt=-1），准备好接收数据。
        # 这确保了在异步传输过程中，一级层的 slot 不会被其他任务占用。

        keys = job_metadata.keys
        block_ids = job_metadata.block_ids

        assert len(keys) == len(block_ids), (
            f"Length mismatch: {len(keys)} keys but {len(block_ids)} block_ids"
        )

        # 中文注释：检查所有待加载的 block 是否都存在于本二级层中。
        # 如果有任何一个 block 不存在，整个任务报告失败。
        # 这是一种"原子性"保证：要么所有 block 都成功加载，要么全部失败。
        # 真实实现中，如果部分 block 缺失，可能需要：
        #   - 报告失败并让框架重试
        #   - 从其他二级层或源头重新获取缺失的 block
        for key in keys:
            if key not in self.blocks:
                self.completed_jobs.append(
                    JobResult(job_id=job_metadata.job_id, success=False)
                )
                return

        # 中文注释：所有 block 都存在，同步完成——立即将成功结果追加到完成队列。
        # TieringOffloadingManager 收到成功结果后会调用
        # primary_tier.complete_write() 使一级层 slot 可用，
        # 后续 prepare_load() 即可将数据从一级层加载到 GPU。
        self.completed_jobs.append(JobResult(job_id=job_metadata.job_id, success=True))

    def get_finished_jobs(self) -> Iterable[JobResult]:
        """
        Poll for finished jobs.

        Returns:
            Iterable of JobResult objects for all jobs that have
            finished since the last call.
        """
        # 中文注释：轮询自上次调用以来完成的所有异步传输任务。
        #
        # ==================== 调用时机 ====================
        #
        # 此方法由 TieringOffloadingManager._process_finished_jobs() 在每个
        # engine step 中调用（通过 _maybe_process_finished_jobs() 门控，每步最多
        # 执行一次）。这种设计避免了在同一 engine step 中多次轮询，
        # 减少了不必要的开销。
        #
        # ==================== 返回值处理 ====================
        #
        # 框架根据返回的 JobResult 执行资源清理：
        #
        # 情况 1：提升完成（is_promotion=True）
        #   - 调用 primary.complete_write()
        #   - 使一级层 slot 可用（从 in-flight 状态变为正常状态）
        #   - 后续 prepare_load() 即可将数据从一级层加载到 GPU
        #
        # 情况 2：级联完成（is_promotion=False）
        #   - 调用 primary.complete_read()
        #   - 释放一级层 block 的引用计数
        #   - 恢复淘汰能力（如果 ref_cnt 降为 0，block 可被淘汰）
        #
        # 情况 3：任务失败（success=False）
        #   - 框架可能记录错误日志
        #   - 对于提升任务，已分配的一级层 slot 可能需要回收
        #
        # ==================== 实现要点 ====================
        #
        # 1. 必须返回"自上次调用以来"新完成的任务（不是全部已完成任务）。
        #    这是因为框架需要知道"新完成"的任务，而不是重复处理已完成的任务。
        #
        # 2. 取出后应清空内部列表，避免下次重复返回。
        #    本示例使用"取出并替换"模式实现（原子性好，实现简单）。
        #
        # 3. 在真实的异步实现中，后台线程完成传输后会将 JobResult 追加到
        #    self.completed_jobs，本方法取出并清空，实现"生产者-消费者"模式。
        #
        # ==================== 线程安全考量 ====================
        #
        # 在真实的多线程实现中，需要注意：
        #   - self.completed_jobs 可能被后台线程和主线程同时访问
        #   - 需要使用锁或其他同步机制保护
        #   - 本示例由于是同步完成的，不存在线程安全问题
        result = self.completed_jobs
        self.completed_jobs = []
        return result

    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext:
        # 中文注释：当新请求到达时，返回该请求的卸载策略偏好。
        #
        # ==================== 调用时机 ====================
        #
        # 此方法由 TieringOffloadingManager 在新请求到达时调用，
        # 用于获取该请求在二级层的卸载策略偏好。
        #
        # ==================== 参数说明 ====================
        #
        # req_context: ReqContext
        #   请求上下文，包含请求级别的元数据，例如：
        #   - request_id: 请求的唯一标识符
        #   - 其他请求相关的配置和状态
        #
        # ==================== 返回值说明 ====================
        #
        # 返回 RequestOffloadingContext 对象，包含该请求的卸载策略偏好，
        # 例如：
        #   - 是否启用级联存储（cascade）
        #   - 是否启用提升加载（promotion）
        #   - 优先级设置
        #   - 其他自定义策略参数
        #
        # 本示例返回默认的 RequestOffloadingContext（所有策略使用默认值）。
        # 真实实现可能根据请求类型、优先级等因素返回不同的策略。
        return RequestOffloadingContext()

    def get_num_blocks(self) -> int:
        """Get the number of blocks currently stored in this tier."""
        # 中文注释：返回当前存储在本二级层中的 block 数量。
        #
        # ==================== 用途 ====================
        #
        # 此方法主要用于：
        # 1. 监控和统计：了解二级层的存储使用情况
        # 2. 调试：验证 block 是否正确存储和删除
        # 3. 容量管理：在真实实现中，可用于判断是否需要淘汰旧 block
        #
        # ==================== 实现说明 ====================
        #
        # 本示例直接返回 self.blocks 字典的长度。
        # 由于 self.blocks 使用 OffloadKey 作为 key，每个 key 对应一个 block，
        # 因此字典长度即为存储的 block 数量。
        #
        # 注意：本示例不实现 block 删除逻辑（on_request_finished() 未实现），
        # 因此 self.blocks 的大小会持续增长。
        # 真实实现需要在请求完成时清理不再需要的 block，
        # 并在容量不足时淘汰旧 block（如 LRU 策略）。
        return len(self.blocks)
