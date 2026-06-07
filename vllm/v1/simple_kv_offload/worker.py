# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-side handler for SimpleCPUOffloadConnector.

工作节点端处理器，负责执行KV缓存的CPU卸载传输操作。
主要职责：
1. 注册GPU和CPU KV缓存张量
2. 管理异步CUDA流用于数据传输
3. 执行实际的GPU<->CPU数据拷贝
4. 跟踪传输事件并报告完成状态
"""

from typing import TYPE_CHECKING

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.v1.simple_kv_offload.copy_backend import DmaCopyBackend
from vllm.v1.simple_kv_offload.cuda_mem_ops import pin_tensor
from vllm.v1.simple_kv_offload.metadata import (
    SimpleCPUOffloadMetadata,
    SimpleCPUOffloadWorkerMetadata,
)

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import KVCacheConfig

logger = init_logger(__name__)


class SimpleCPUOffloadWorker:
    """Worker-side handler for CPU offloading transfers.

    CPU卸载传输的工作节点端处理器。
    负责在工作节点（GPU进程）中执行实际的KV缓存数据传输。

    核心功能：
    1. 初始化时注册GPU KV缓存并分配对应的CPU pinned内存
    2. 每步接收调度器的元数据，包含需要加载/存储的块ID映射
    3. 使用DMA后端在后台线程中执行异步数据传输
    4. 通过CUDA事件跟踪传输完成状态
    5. 向调度器报告完成的传输事件
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: "KVCacheConfig | None",
        cpu_capacity_bytes: int,
    ):
        """
        初始化CPU卸载工作节点。

        参数：
        - vllm_config: vLLM全局配置
        - kv_cache_config: KV缓存配置
        - cpu_capacity_bytes: CPU内存容量（字节）
        """
        self.vllm_config = vllm_config
        self.kv_cache_config = kv_cache_config
        self.cpu_capacity_bytes = cpu_capacity_bytes

        # GPU和CPU KV缓存张量字典，键为层名，值为对应的张量
        self.gpu_kv_caches: dict[str, torch.Tensor] | None = None
        self.cpu_kv_caches: dict[str, torch.Tensor] | None = None
        self.device: torch.device | None = None  # GPU设备
        self.num_cpu_blocks: int = 0  # CPU块数量

        # CUDA streams for the async transfers
        # CUDA流，用于异步传输
        self.load_stream: torch.cuda.Stream | None = None  # CPU->GPU加载流
        self.store_stream: torch.cuda.Stream | None = None  # GPU->CPU存储流

        # DMA复制后端，使用后台线程执行批量内存拷贝
        self._backend = DmaCopyBackend()

        # Ordered (event_idx, Event). Events pre-allocated on main thread.
        # 有序的(event_idx, Event)列表。事件在主线程中预分配。
        self._load_events: list[tuple[int, torch.Event]] = []  # 加载事件列表
        self._store_events: list[tuple[int, torch.Event]] = []  # 存储事件列表
        # High-water marks: highest event_idx completed per stream.
        # When the event list is empty, the hwm covers all prior events.
        # 高水位标记：每个流完成的最高event_idx。
        # 当事件列表为空时，hwm覆盖所有先前的事件。
        self._load_hwm: int = -1  # 加载流的高水位标记
        self._store_hwm: int = -1  # 存储流的高水位标记

        # Metadata for the current step
        # 当前步骤的元数据
        self._connector_metadata: SimpleCPUOffloadMetadata | None = None

        # Pending event index sets, populated in bind_connector_metadata
        # 待处理的事件索引集合，在bind_connector_metadata中填充
        self._pending_load_event_indices: set[int] = set()
        self._pending_store_event_indices: set[int] = set()
        # Completed store events to report via build_connector_worker_meta
        # 已完成的存储事件，通过build_connector_worker_meta报告
        self._completed_store_events: dict[int, int] = {}

    def register_kv_caches(
        self,
        kv_caches: dict[str, torch.Tensor],
    ) -> None:
        """Register GPU KV caches and allocate pinned CPU tensors.
        The worker will infer the underlying raw storage from the kv_caches.

        注册GPU KV缓存并分配pinned CPU张量。
        工作节点将从kv_caches推断底层原始存储。

        注册流程：
        1. 解析GPU KV缓存，去重共享存储的层
        2. 为每个唯一的存储创建[num_blocks, block_bytes]的int8视图
        3. 计算每个块的字节数，确定CPU块数量
        4. 分配pinned CPU内存（使用cudaHostRegister避免PyTorch的2次幂对齐浪费）
        5. 创建低优先级CUDA流，让KV缓存I/O让步于计算流
        6. 初始化DMA复制后端

        Args:
            kv_caches: Per-layer GPU KV caches. Values are either a single
                tensor (attention layers) or a list of tensors (Mamba layers
                in hybrid models). All values are included for offloading
                by resolving to their underlying raw storage.

        参数：
        - kv_caches: 每层的GPU KV缓存。值可以是单个张量（注意力层）
          或张量列表（混合模型中的Mamba层）。
          所有值都通过解析底层原始存储包含在卸载中。
        """
        if not kv_caches:
            logger.warning("No KV caches to offload.")
            return

        # Resolve each entry to a representative tensor for storage
        # deduplication. For attention layers the value is already a tensor;
        # for Mamba layers it is a list of tensors that all share the same
        # underlying raw storage, so we take the first one.
        # 将每个条目解析为代表性张量，用于存储去重。
        # 对于注意力层，值已经是张量；
        # 对于Mamba层，它是共享同一底层存储的张量列表，所以我们取第一个。
        def _repr_tensor(v: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
            assert isinstance(v, torch.Tensor | list)
            return v if isinstance(v, torch.Tensor) else v[0]

        any_tensor = _repr_tensor(next(iter(kv_caches.values())))
        self.device = any_tensor.device

        assert self.kv_cache_config is not None
        num_blocks = self.kv_cache_config.num_blocks

        # Deduplicate: multiple layers may share the same backing storage.
        # 去重：多个层可能共享相同的后备存储
        seen_ptrs: dict[int, tuple[str, torch.Tensor]] = {}
        for name, value in kv_caches.items():
            tensor = _repr_tensor(value)
            ptr = tensor.untyped_storage().data_ptr()
            if ptr not in seen_ptrs:
                seen_ptrs[ptr] = (name, tensor)

        # Build [num_blocks, block_bytes] int8 views from each unique
        # storage so that stride(0) gives block_bytes for the copy op.
        #
        # The physical layout varies across attention backends:
        #   FlashAttn/ROCm:  (2, num_blocks, ...) -> K/V outermost, 2 segments
        #   FlashInfer/MLA:  (num_blocks, ...)    -> blocks outermost, 1 segment
        # We derive page_size_bytes = storage.nbytes() // num_blocks, then
        # classify dims: any dim whose byte-stride exceeds page_size_bytes
        # must be an outer segment dim (e.g. the K/V dim of size 2). A less
        # hacky way is to update the interface with the layout.
        #
        # 从每个唯一的存储构建[num_blocks, block_bytes]的int8视图，
        # 使得stride(0)给出block_bytes用于复制操作。
        #
        # 物理布局因注意力后端而异：
        #   FlashAttn/ROCm:  (2, num_blocks, ...) -> K/V最外层，2个段
        #   FlashInfer/MLA:  (num_blocks, ...)    -> blocks最外层，1个段
        # 我们推导page_size_bytes = storage.nbytes() // num_blocks，
        # 然后分类维度：任何字节步长超过page_size_bytes的维度
        # 必须是外部段维度（例如大小为2的K/V维度）。
        unique_gpu_caches: dict[str, torch.Tensor] = {}
        for name, tensor in seen_ptrs.values():
            storage = tensor.untyped_storage()
            raw = torch.empty(0, dtype=torch.int8, device=self.device).set_(
                storage, 0, (storage.nbytes(),)
            )
            el = tensor.element_size()
            page_size_bytes = storage.nbytes() // num_blocks
            outer_dims = [
                d for d in range(tensor.ndim) if tensor.stride(d) * el > page_size_bytes
            ]
            if not outer_dims:
                unique_gpu_caches[name] = raw.view(num_blocks, -1)
            else:
                seg_stride = tensor.stride(outer_dims[0]) * el
                for idx in range(tensor.shape[outer_dims[0]]):
                    offset = idx * seg_stride
                    chunk = raw[offset : offset + seg_stride]
                    unique_gpu_caches[f"{name}.{idx}"] = chunk.view(num_blocks, -1)

        # Compute per-tensor bytes_per_block. Tensors may have different
        # page_size_bytes (e.g., UniformTypeKVCacheSpecs with varying head_size).
        # 计算每个张量的bytes_per_block。张量可能有不同的
        # page_size_bytes（例如，UniformTypeKVCacheSpecs有不同的head_size）。
        per_tensor_bpb = [
            t.stride(0) * t.element_size() for t in unique_gpu_caches.values()
        ]
        total_bytes_per_block = sum(per_tensor_bpb)

        # 根据CPU容量和每块字节数计算CPU块数量
        self.num_cpu_blocks = max(1, self.cpu_capacity_bytes // total_bytes_per_block)

        logger.info(
            "SimpleCPUOffloadWorker: %d unique GPU KV tensors, "
            "allocating %d CPU blocks (%.2f GB)",
            len(unique_gpu_caches),
            self.num_cpu_blocks,
            (self.num_cpu_blocks * total_bytes_per_block) / (1024**3),
        )

        # 检查是否可用pinned memory
        pin_memory = is_pin_memory_available()
        if not pin_memory:
            logger.warning(
                "Pinned memory not available. CPU offload performance may be degraded."
            )

        self.gpu_kv_caches = unique_gpu_caches
        self.cpu_kv_caches = {}
        for name, gpu_tensor in unique_gpu_caches.items():
            cpu_shape = (self.num_cpu_blocks,) + gpu_tensor.shape[1:]
            # Allocate non-pinned first, then pin via cudaHostRegister to
            # bypass PyTorch's CUDACachingHostAllocator which rounds up to
            # the next power of 2 (e.g. 100 GB -> 128 GB).
            # 先分配非pinned内存，然后通过cudaHostRegister固定，
            # 绕过PyTorch的CUDACachingHostAllocator（它会向上舍入到2的幂，
            # 例如100 GB -> 128 GB）。
            tensor = torch.zeros(cpu_shape, dtype=gpu_tensor.dtype, device="cpu")
            if pin_memory:
                pin_tensor(tensor)
            self.cpu_kv_caches[name] = tensor

        # Use lowest priority so KV cache I/O yields to compute streams.
        # 使用最低优先级，让KV缓存I/O让步于计算流
        low_pri, _ = torch.cuda.Stream.priority_range()
        self.load_stream = torch.cuda.Stream(priority=low_pri)
        self.store_stream = torch.cuda.Stream(priority=low_pri)

        # Initialize copy backend with caches and streams.
        # 使用缓存和流初始化复制后端
        self._backend.init(
            self.gpu_kv_caches,
            self.cpu_kv_caches,
            self.device,
            self.load_stream,
            self.store_stream,
        )

    def bind_connector_metadata(self, metadata: SimpleCPUOffloadMetadata) -> None:
        """绑定连接器元数据。

        每步调用一次，将调度器生成的元数据传递给工作节点。
        元数据包含需要执行的加载和存储操作的事件索引。

        参数：
        - metadata: 包含加载/存储操作详情的元数据
        """
        self._connector_metadata = metadata
        if metadata.load_event >= 0:
            self._pending_load_event_indices.add(metadata.load_event)
        if metadata.store_event >= 0:
            self._pending_store_event_indices.add(metadata.store_event)

    def clear_connector_metadata(self) -> None:
        """清除连接器元数据。

        在步骤完成后调用，清理当前步骤的元数据。
        """
        self._connector_metadata = None

    def start_load_kv(self) -> None:
        """启动KV缓存加载。

        注意：我们延迟启动加载和存储到get_finished()，
        它在模型执行后运行。这样可以将CPU端的块复制操作开销（~5ms）
        隐藏在GPU计算之后。
        """
        # NOTE: we defer launching both load and store to get_finished(),
        # which runs after model execution. This hides the CPU-side
        # block copy op overhead (~5ms) behind GPU compute.
        pass

    def wait_for_save(self) -> None:
        """等待存储完成。

        当前实现为空，因为存储是异步的，通过事件机制跟踪完成状态。
        """
        pass

    def get_finished(
        self,
        finished_req_ids: set[str],
    ) -> tuple[set[str] | None, set[str] | None]:
        """Submit transfers and report completed events to the scheduler.

        提交传输并向调度器报告完成事件。

        此方法在模型执行后调用，负责：
        1. 启动本步骤的加载和存储传输
        2. 轮询已完成的传输事件
        3. 返回已完成的加载请求ID集合

        调度器只为已确认计算完成的块调度存储，
        因此我们可以立即启动加载和存储——无需延迟或跨流同步。

        Returns:
            tuple of (finished_sending, finished_recving).
            - finished_sending: always None (stores use worker metadata).
            - finished_recving: req_ids whose loads have completed.

        返回：
        - (finished_sending, finished_recving) 元组
          - finished_sending: 始终为None（存储使用工作节点元数据）
          - finished_recving: 加载已完成的请求ID集合
        """
        # (1) Submit transfers
        # (1) 提交传输
        metadata = self._connector_metadata
        if metadata is not None:
            # Launch loads (CPU->GPU).
            # 启动加载（CPU->GPU）
            if metadata.load_cpu_blocks:
                self._backend.launch_copy(
                    metadata.load_cpu_blocks,
                    metadata.load_gpu_blocks,
                    is_store=False,
                    event_idx=metadata.load_event,
                    events_list=self._load_events,
                )
            # Launch stores (GPU->CPU).
            # 启动存储（GPU->CPU）
            if metadata.store_gpu_blocks:
                self._backend.launch_copy(
                    metadata.store_gpu_blocks,
                    metadata.store_cpu_blocks,
                    is_store=True,
                    event_idx=metadata.store_event,
                    events_list=self._store_events,
                )

        # (2) Track completed transfer events
        # (2) 跟踪已完成的传输事件
        finished_recving: set[str] = set()

        # 检查加载事件完成情况
        if self._pending_load_event_indices:
            load_wm = self._poll_stream_events(is_store=False)
            for j in [j for j in self._pending_load_event_indices if j <= load_wm]:
                self._pending_load_event_indices.discard(j)
                req_ids = (
                    metadata.load_event_to_reqs.get(j) if metadata is not None else None
                )
                if req_ids:
                    finished_recving.update(req_ids)

        # 检查存储事件完成情况
        if self._pending_store_event_indices:
            store_wm = self._poll_stream_events(is_store=True)
            for j in [j for j in self._pending_store_event_indices if j <= store_wm]:
                self._pending_store_event_indices.discard(j)
                # 记录完成的存储事件，用于后续报告给调度器
                self._completed_store_events[j] = 1

        return None, finished_recving or None

    def build_connector_worker_meta(self) -> SimpleCPUOffloadWorkerMetadata | None:
        """Return completed store events since the last call.

        返回自上次调用以来完成的存储事件。

        工作节点通过此方法向调度器报告已完成的存储事件。
        返回后会清空已完成事件集合。

        返回：
        - SimpleCPUOffloadWorkerMetadata: 包含已完成存储事件的元数据
        - None: 如果没有已完成的事件
        """
        if not self._completed_store_events:
            return None
        meta = SimpleCPUOffloadWorkerMetadata(
            completed_store_events=self._completed_store_events,
        )
        self._completed_store_events = {}
        return meta

    def handle_preemptions(
        self, kv_connector_metadata: SimpleCPUOffloadMetadata
    ) -> None:
        """Sync all in-flight transfers before preempted blocks are reused.

        在被抢占的块被重用之前，同步所有飞行中的传输。

        当发生请求抢占时，需要确保所有传输完成，
        因为被抢占的块可能会被重新分配给其他请求。

        参数：
        - kv_connector_metadata: 连接器元数据，包含need_flush标志
        """
        if not kv_connector_metadata.need_flush:
            return
        self._flush_and_sync_all()

    def _flush_and_sync_all(self) -> None:
        """Synchronize all in-flight transfer events.

        同步所有飞行中的传输事件。

        等待所有加载和存储事件完成，并更新高水位标记。
        这是一个阻塞操作，确保在继续之前所有传输都已完成。
        """
        for event_idx, event in self._load_events:
            event.synchronize()
            self._load_hwm = event_idx
        self._load_events.clear()

        for event_idx, event in self._store_events:
            event.synchronize()
            self._store_hwm = event_idx
        self._store_events.clear()

    def _poll_stream_events(self, is_store: bool) -> int:
        """Non-blocking poll for completed events and return the high-water mark.

        非阻塞轮询已完成的事件并返回高水位标记。

        工作原理：
        1. 检查事件列表中的第一个事件
        2. 如果已完成，更新高水位标记并移除该事件
        3. 继续检查下一个事件，直到遇到未完成的事件
        4. 返回当前的高水位标记

        参数：
        - is_store: True表示轮询存储事件，False表示轮询加载事件

        返回：
        - 当前的高水位标记（已完成的最高事件索引）
        """
        events = self._store_events if is_store else self._load_events
        hwm = self._store_hwm if is_store else self._load_hwm
        while events:
            event_idx, event = events[0]
            if not event.query():
                break
            hwm = event_idx
            events.pop(0)
        if is_store:
            self._store_hwm = hwm
        else:
            self._load_hwm = hwm
        return hwm
