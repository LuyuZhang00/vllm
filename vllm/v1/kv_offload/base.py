# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Core abstractions for KV cache offloading in vLLM v1.

KV cache offloading 的核心抽象模块。

本模块定义了 vLLM v1 中 KV cache 卸载（offloading）的核心接口和数据结构。
KV cache offloading 的核心思想是：当 GPU 显存不足时，将不活跃的 KV cache
block 从 GPU 移动到更廉价的存储介质（如 CPU 内存、磁盘等），需要时再加载回来。

整体架构分为三层：
1. OffloadingSpec（规范层）—— 定义 offloading 的配置参数和存储规格，
   在初始化时决定 block 大小、hash block 大小等关键参数。
2. OffloadingManager（管理层）—— 运行在 Scheduler 进程中，负责追踪哪些
   block 已被卸载、管理 LRU 淘汰策略、协调 load/store 操作的生命周期。
3. OffloadingHandler（执行层）—— 运行在 Worker 进程中，负责实际的
   数据搬运（GPU -> CPU / CPU -> GPU 等）。

数据流：
  Scheduler 调用 Manager 的 lookup/prepare_load/prepare_store 等方法获取元数据，
  然后将 LoadStoreSpec 传递给 Worker，Worker 通过 Handler 执行实际的数据传输。
"""

from abc import ABC, abstractmethod
from collections.abc import Collection, Iterable, Iterator, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, NewType

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.v1.core.kv_cache_utils import resolve_kv_cache_block_sizes

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig
    # OffloadingHandler 实际运行在 Worker 进程中，负责执行具体的
    # 数据搬运操作（如 GPU->CPU 的 store，CPU->GPU 的 load）。
    from vllm.v1.kv_offload.worker.worker import OffloadingHandler

# `OffloadKey` identifies an offloaded block. It combines a block hash with
# its KV cache group index, encoded as raw bytes to avoid tuple GC overhead.
# Use the helper functions below to construct / decompose keys.
# 中文注释：OffloadKey 是已卸载 KV block 的唯一标识符。
# 它由 block hash（块内容哈希）和 KV cache group index（KV 缓存组索引）
# 拼接而成的原始字节序列。
# 设计为 bytes 类型而非 tuple，是为了避免 Python tuple 的 GC 开销，
# 因为 OffloadKey 会被大量创建和用作字典键。
# 使用下方的辅助函数 make_offload_key / get_offload_block_hash /
# get_offload_group_idx 来构造和拆解 key。
OffloadKey = NewType("OffloadKey", bytes)

logger = init_logger(__name__)


def make_offload_key(block_hash: bytes, group_idx: int) -> OffloadKey:
    """Pack a block hash and group index into an `OffloadKey`."""
    # 中文注释：将 block hash 和 KV cache group index 打包成一个 OffloadKey。
    # group_idx 以 4 字节大端无符号整数形式追加到 block_hash 末尾。
    # 这样 key 的前 N-4 字节是 block hash，后 4 字节是 group index。
    return OffloadKey(block_hash + group_idx.to_bytes(4, "big", signed=False))


def get_offload_block_hash(key: OffloadKey) -> bytes:
    """Extract the block hash from an `OffloadKey`."""
    # 中文注释：从 OffloadKey 中提取 block hash（即去掉末尾 4 字节 group index）。
    return key[:-4]


def get_offload_group_idx(key: OffloadKey) -> int:
    """Extract the group index from an `OffloadKey`."""
    # 中文注释：从 OffloadKey 中提取 group index（末尾 4 字节大端无符号整数）。
    return int.from_bytes(key[-4:], "big", signed=False)


@dataclass
class ReqContext:
    """中文注释：请求级别的上下文信息，用于在 offloading 操作中标识和关联请求。

    每个请求在 Scheduler 中被处理时，都会携带一个 ReqContext 对象，
    用于在 Manager 和 Handler 之间传递请求标识和额外参数。

    字段说明：
        req_id: 请求的唯一标识符。
        kv_transfer_params: KV 传输相关的可选参数（如分布式 KV 传输的目标地址等），
            由具体的 offloading connector 实现决定其含义。
    """
    req_id: str
    kv_transfer_params: dict[str, Any] | None = None


class OffloadPolicy(Enum):
    """中文注释：卸载策略枚举，决定哪些 KV block 需要被卸载到低层存储。

    两种策略的区别在于是否卸载 prefix cache 命中的 block：
    - BLOCK_LEVEL: 只卸载新计算的 block（decode 阶段新生成的 KV），
      已经在 offload 存储中的 prefix-hit block 不重复卸载。
    - REQUEST_LEVEL: 卸载请求的所有 block（包括 prefix-hit 的），
      适用于需要完整 KV 上下文的存储层（如分布式 KV 传输）。
    """
    # Offload only newly-computed blocks as they arrive; prefix-hit
    # blocks (already offloaded by a prior request) are skipped.
    BLOCK_LEVEL = "block_level"
    # Offload all blocks for the request, including prefix hits.
    # Used by tiers that need the complete KV context for a request.
    REQUEST_LEVEL = "request_level"


@dataclass
class RequestOffloadingContext:
    """中文注释：每个请求关联的卸载上下文，记录该请求的卸载策略。

    由 OffloadingManager.on_new_request() 返回，Scheduler 根据此上下文
    决定如何为该请求执行 load/store 操作。

    字段说明：
        policy: 该请求适用的卸载策略（BLOCK_LEVEL 或 REQUEST_LEVEL）。
    """
    policy: OffloadPolicy = OffloadPolicy.BLOCK_LEVEL


class LoadStoreSpec(ABC):
    """
    Abstract metadata that encapsulates information allowing a worker
    to load, and optionally also to store, blocks of KV data.
    """
    # 中文注释：LoadStoreSpec 是 load/store 操作的抽象元数据基类。
    # 它封装了 Worker 执行实际数据搬运所需的全部信息（如目标存储介质、
    # block 地址等），但不包含具体的数据搬运逻辑。
    #
    # 设计模式：
    #   Manager（运行在 Scheduler 进程）生成 LoadStoreSpec，
    #   通过 IPC 传递给 Worker 进程，Worker 根据 spec 类型
    #   找到对应的 OffloadingHandler 执行实际搬运。
    #
    # 子类需要实现 medium() 方法，返回存储介质的字符串标识
    # （如 "GPU"、"CPU"、"DISK" 等），用于匹配对应的 Handler。

    @staticmethod
    @abstractmethod
    def medium() -> str:
        """
        Returns a string representation of the medium type
        this store/load targets.
        """
        pass


@dataclass
class PrepareStoreOutput:
    """中文注释：prepare_store() 的返回值，包含 store 操作的完整信息。

    字段说明：
        keys_to_store: 需要执行存储操作的 block key 列表。
            这些 key 在 prepare_store 调用后已被保护（不会被 LRU 淘汰），
            直到 complete_store() 被调用。
        store_spec: 描述数据应存储到哪里的元数据（如目标存储介质、地址等）。
            Worker 会根据此 spec 找到对应的 OffloadingHandler 执行写入。
        evicted_keys: 因为存储空间不足而被淘汰的旧 block key 列表。
            被淘汰的 block 数据将被丢弃，Scheduler 需要感知这一点
            以便在必要时重新计算这些 block。
    """
    keys_to_store: list[OffloadKey]
    store_spec: LoadStoreSpec
    evicted_keys: list[OffloadKey]


@dataclass
class OffloadingEvent:
    """中文注释：卸载事件，用于通知外部组件 offloading 状态的变化。

    Manager 在完成 store/remove 操作后会生成事件，通过 take_events()
    接口暴露给 Scheduler 或其他监控组件。

    字段说明：
        keys: 发生变化的 block key 列表。
        medium: 发生变化的存储介质标识（如 "GPU"、"CPU"、"DISK"）。
        removed: True 表示 block 被移除（淘汰或删除），
            False 表示 block 被存储（新的 offload 写入）。
    """
    keys: list[OffloadKey]
    medium: str
    # True if blocks are removed, False if stored
    removed: bool


"""
OffloadingManager class for managing KV data offloading in vLLM v1

This class runs in the scheduler, tracks which blocks are offloaded
and their address.

The class provides the following primitives:
    lookup() - check whether a single block is offloaded and ready.
    prepare_load() - prepare given blocks to be read.
        The given blocks will be protected from eviction.
        This function returns a LoadSpec which encapsulates
        information required for performing the load.
    touch() - marks the give blocks as recently used. Can be used
        to track block's LRU. This function is separated from the
        prepare_load function to allow setting block recency even
        for blocks which do not need reading from the cache, such as
        blocks that are cached by the GPU prefix cache.
    complete_load() - mark blocks which were previously prepared to be
        loaded as done loading. This is to re-allow their eviction.
    prepare_store() - prepare the given blocks to be written.
        Returns a StoreSpec encapsulating offloading information,
        as well as a list of blocks that were evicted as a result.
    complete_store() - marks a previous store as completed.
        Following this call, the given blocks will become loadable.
"""
# 中文注释：OffloadingManager 是 KV offloading 的核心管理层，运行在 Scheduler 进程中。
#
# 它的职责是：
#   1. 追踪哪些 KV block 已被卸载到低层存储，以及它们在低层存储中的位置。
#   2. 管理 LRU（最近最少使用）淘汰策略，当低层存储满时决定淘汰哪些 block。
#   3. 协调 load/store 操作的生命周期（prepare -> execute -> complete）。
#
# 核心生命周期流程：
#   【Load 流程】（从低层存储加载 KV 到 GPU）：
#     1) lookup(key)        —— 检查 block 是否在低层存储中可用
#     2) touch(keys)        —— 更新 block 的 LRU 时间戳（即使不 load 也要更新）
#     3) prepare_load(keys) —— 准备加载：锁定这些 block 不被淘汰，返回 LoadStoreSpec
#     4) Worker 执行实际数据搬运（根据 LoadStoreSpec）
#     5) complete_load(keys) —— 完成加载：解除锁定，允许再次被淘汰
#
#   【Store 流程】（将 GPU KV 卸载到低层存储）：
#     1) prepare_store(keys) —— 准备存储：锁定 block、检查空间、必要时淘汰旧 block
#     2) Worker 执行实际数据搬运（根据 LoadStoreSpec）
#     3) complete_store(keys) —— 完成存储：block 变为可 load 状态
#
# Manager 与 Handler 的分离设计：
#   Manager 负责元数据管理和调度决策（运行在 Scheduler 进程），
#   Handler 负责实际数据搬运（运行在 Worker 进程），
#   两者通过 LoadStoreSpec 作为桥梁传递信息。


class OffloadingManager(ABC):
    @abstractmethod
    def lookup(self, key: OffloadKey, req_context: ReqContext) -> bool | None:
        """
        Checks whether a single block is offloaded and ready to be read.

        Args:
            key: the key identifying the block to lookup.
            req_context: per-request context (e.g. kv_transfer_params).

        Returns:
            True if the block is offloaded and ready, False if not,
            or None if the lookup should be retried later.
            Returning None will delay the request handling by the vLLM
            scheduler.
        """
        # 中文注释：查询单个 block 是否已被卸载到低层存储且可读。
        #
        # 返回值语义：
        #   True  —— block 已在低层存储中，可以执行 load 操作。
        #   False —— block 不在低层存储中，需要重新计算或从其他来源获取。
        #   None  —— 查询暂时无法确定（如异步传输尚未完成），
        #            Scheduler 会将该请求延迟处理，稍后重试。
        #
        # 在主链路中的作用：
        #   Scheduler 在调度请求时，对 prefix cache 命中的 block 调用 lookup()，
        #   判断哪些 block 可以从 offload 存储直接加载，哪些需要重新计算。
        pass

    @abstractmethod
    def prepare_load(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
    ) -> LoadStoreSpec:
        """
        Prepare the given blocks to be read.
        The given blocks will be protected from eviction until
        complete_load is called.
        It assumes all given blocks are offloaded.

        Args:
            keys: the keys identifying the blocks.
            req_context: per-request context (e.g. kv_transfer_params).

        Returns:
            A LoadStoreSpec that can be used by a worker to locate and load
            the actual offloaded KV data.
        """
        # 中文注释：准备从低层存储加载指定的 KV block。
        #
        # 调用前假设：所有传入的 key 对应的 block 都已在低层存储中（通过 lookup 确认）。
        # 调用后的效果：
        #   1) 这些 block 被"锁定"，不会被 LRU 淘汰，直到 complete_load() 被调用。
        #   2) 返回一个 LoadStoreSpec，包含 Worker 执行实际数据搬运所需的全部信息
        #      （如源存储介质、目标 block 地址、数据布局等）。
        #
        # 在主链路中的作用：
        #   Scheduler 将返回的 LoadStoreSpec 通过 IPC 传递给 Worker，
        #   Worker 根据 spec 类型找到对应的 OffloadingHandler，
        #   执行 CPU->GPU 或 DISK->GPU 的数据传输。
        pass

    def touch(self, keys: Collection[OffloadKey], req_context: ReqContext):
        """
        Mark the given blocks as recently used.
        This could in practice mean moving them to the end of an LRU list.

        Args:
            keys: the keys identifying the blocks.
            req_context: per-request context (e.g. kv_transfer_params).
        """
        # 中文注释：将指定 block 标记为"最近使用"，更新其 LRU 时间戳。
        #
        # 为什么 touch() 与 prepare_load() 分离？
        #   因为某些 block 可能已经被 GPU 的 prefix cache 缓存，不需要实际从
        #   低层存储 load，但仍然需要更新其在低层存储中的 LRU 位置，防止被淘汰。
        #   所以 touch() 只更新时间戳，不触发数据搬运。
        return

    def complete_load(self, keys: Collection[OffloadKey], req_context: ReqContext):
        """
        Marks previous blocks that were prepared to load as done loading.

        Args:
            keys: the keys identifying the blocks.
            req_context: per-request context (e.g. kv_transfer_params).
        """
        # 中文注释：标记之前通过 prepare_load() 准备的 block 已完成加载。
        # 调用后，这些 block 解除"锁定"，重新进入 LRU 淘汰池。
        # 这是 load 生命周期的最后一步，确保 block 在加载过程中不会被淘汰。
        return

    @abstractmethod
    def prepare_store(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
    ) -> PrepareStoreOutput | None:
        """
        Prepare the given blocks to be offloaded.
        The given blocks will be protected from eviction until
        complete_store is called.

        Args:
            keys: the keys identifying the blocks.
            req_context: per-request context (e.g. kv_transfer_params).

        Returns:
            A PrepareStoreOutput indicating which blocks need storing,
            where to store them (LoadStoreSpec), and list of blocks that
            were evicted as a result.
            None is returned if the blocks cannot be stored.
        """
        pass

    def complete_store(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
        success: bool = True,
    ):
        """
        Marks blocks which were previously prepared to be stored, as stored.
        Following this call, the blocks become loadable.
        If success is False, blocks that were not marked as stored will be
        removed.

        Args:
            keys: the keys identifying the blocks.
            req_context: per-request context (e.g. kv_transfer_params).
            success: whether the blocks were stored successfully.
        """
        return

    @abstractmethod
    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext:
        """
        Called when a new request is first seen by the scheduler.

        Returns a RequestOffloadingContext indicating how this request's
        blocks should be offloaded.

        Args:
            req_context: per-request context.
        """
        pass

    def on_request_finished(self, req_context: ReqContext) -> None:
        """
        Called when a request has finished.

        Args:
            req_context: per-request context.
        """
        return

    def take_events(self) -> Iterable[OffloadingEvent]:
        """
        Take the offloading events from the manager.

        Yields:
            New OffloadingEvents collected since the last call.
        """
        return ()

    def reset_cache(self) -> None:
        """Evict all tracked blocks and reset internal state."""
        return

    def shutdown(self) -> None:
        """Shutdown the manager and release any resources."""
        return


class BlockIDsLoadStoreSpec(LoadStoreSpec, ABC):
    """
    Spec for loading/storing KV blocks from given block numbers.
    """

    def __init__(self, block_ids: list[int]):
        self.block_ids = np.array(block_ids, dtype=np.int64)

    def __repr__(self) -> str:
        return repr(self.block_ids)


class GPULoadStoreSpec(BlockIDsLoadStoreSpec):
    """
    Spec for loading/storing a KV block to GPU memory.

    If there are multiple KV groups, the blocks are expected to be
    ordered by the group index.
    In that case, group_sizes[i] determines the number of blocks
    per the i-th KV group, and thus sum(group_sizes) == len(block_ids).
    group_sizes=None indicates a single KV group.

    If block_indices is given, each group (determined by group_sizes) of block IDs
    will correspond to logically contiguous blocks, e.g. blocks 5-10 of a some request.
    block_indices[i] will represent the block index of the first block in group #i.
    Thus, len(block_indices) == len(group_sizes) = number of KV cache groups.
    This information is required in order to support off/loading from offloaded blocks
    which are larger than GPU blocks.
    In such cases, the first GPU block per each group may be unaligned to the offloaded
    block size, and so knowing block_indices[i] allows the worker to correctly
    skip part of the first matching offloaded block.
    """

    def __init__(
        self,
        block_ids: list[int],
        group_sizes: Sequence[int],
        block_indices: Sequence[int],
    ):
        super().__init__(block_ids)
        assert sum(group_sizes) == len(block_ids)
        assert len(block_indices) == len(group_sizes)
        self.group_sizes: Sequence[int] = group_sizes
        self.block_indices: Sequence[int] = block_indices

    @staticmethod
    def medium() -> str:
        return "GPU"


@dataclass
class CanonicalKVCacheTensor:
    """
    A canonicalized KV cache tensor whose first dimension is num_blocks.

    For attention backends where the raw tensor has num_blocks at a
    non-leading physical dimension (e.g. FlashAttention's
    (2, num_blocks, ...) layout), the tensor is split so that each
    resulting CanonicalKVCacheTensor starts with (num_blocks, ...).
    """

    # The KV cache tensor with shape (num_blocks, ...)
    tensor: torch.Tensor
    # The (possibly padded) page size per block in bytes
    page_size_bytes: int


@dataclass
class CanonicalKVCacheRef:
    """
    Per-layer (or group of layers) reference to a specific (by index)
    CanonicalKVCacheTensor and records the un-padded page size used by that layer.
    """

    # Index into the list of CanonicalKVCacheTensor objects
    tensor_idx: int
    # The un-padded page size per block in bytes
    page_size_bytes: int


@dataclass
class CanonicalKVCaches:
    """
    Canonicalized block-level representation of the KV caches.

    Composed of:
        - Unique list of KV cache data tensors,
          each with shape (num_blocks, page_size_in_bytes) and int8 dtype.
        - Per-group data references of the tensors.
          i.e. how each KV cache group maps to the tensors.
    """

    # Ordered list of unique block tensors, each with shape
    # (num_blocks, ...).
    tensors: list[CanonicalKVCacheTensor]
    # Per-KV-cache-group list of data references that map each layer
    # in the group to the appropriate entry in the tensors list.
    group_data_refs: list[list[CanonicalKVCacheRef]]


class OffloadingSpec(ABC):
    """Spec for an offloading connector"""

    def __init__(self, vllm_config: "VllmConfig", kv_cache_config: "KVCacheConfig"):
        logger.warning(
            "Initializing OffloadingSpec. This API is experimental and "
            "subject to change in the future as we iterate the design."
        )
        self.vllm_config = vllm_config
        self.kv_cache_config = kv_cache_config

        kv_transfer_config = vllm_config.kv_transfer_config
        assert kv_transfer_config is not None
        self.extra_config = kv_transfer_config.kv_connector_extra_config

        # When True, only prompt (prefill) blocks are offloaded; decode-phase
        # blocks (KV generated after the prompt) are skipped. Useful when prior
        # turns' generated tokens are dropped before the next turn (e.g.
        # reasoning models that strip thinking).
        self.offload_prompt_only: bool = bool(
            self.extra_config.get("offload_prompt_only", True)
        )

        parallel_config = vllm_config.parallel_config
        context_parallel_factor = (
            parallel_config.decode_context_parallel_size
            * parallel_config.prefill_context_parallel_size
        )

        # gpu block size per group
        self.gpu_block_size: tuple[int, ...] = tuple(
            kv_cache_group.kv_cache_spec.block_size * context_parallel_factor
            for kv_cache_group in kv_cache_config.kv_cache_groups
        )

        # hash_block_size must match what the scheduler uses for
        # Request.block_hashes (resolved via resolve_kv_cache_block_sizes).
        _, self.hash_block_size = resolve_kv_cache_block_sizes(
            kv_cache_config, vllm_config
        )

        for block_size in self.gpu_block_size:
            assert block_size % self.hash_block_size == 0, (
                f"gpu_block_size={block_size} not divisible by "
                f"hash_block_size={self.hash_block_size}. "
                f"Hybrid models (e.g. Mamba+Attention) need "
                f"--enable-prefix-caching to align block sizes."
            )

        # offloaded_block_size / gpu_block_size
        self.block_size_factor: int = 1

        offloaded_block_size = self.extra_config.get("block_size")
        if offloaded_block_size is not None:
            offloaded_block_size_int = int(offloaded_block_size)
            gpu_block_sizes = set(self.gpu_block_size)
            assert len(gpu_block_sizes) == 1, (
                "If 'block_size' is specified in kv_connector_extra_config, "
                "there must be at least one KV cache group, "
                "and all groups must have the same block size."
            )
            gpu_block_size = gpu_block_sizes.pop()

            assert offloaded_block_size_int % gpu_block_size == 0
            self.block_size_factor = offloaded_block_size_int // gpu_block_size

    @abstractmethod
    def get_manager(self) -> OffloadingManager:
        """
        Get an OffloadingManager that will be used
        by the scheduler-side offloading connector to track
        offloaded blocks and manage evictions.
        """
        pass

    @abstractmethod
    def get_handlers(
        self, kv_caches: CanonicalKVCaches
    ) -> Iterator[tuple[type[LoadStoreSpec], type[LoadStoreSpec], "OffloadingHandler"]]:
        """
        Get offloading handlers along with their respective src and dst types.

        Args:
            kv_caches: Canonicalized KV caches.

        Yields:
            Tuples of (src_type, dst_type, offloading_handler).
        """
        pass
