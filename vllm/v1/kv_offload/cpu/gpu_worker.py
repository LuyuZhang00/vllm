# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
GPU <-> CPU 数据传输模块 (vllm/v1/kv_offload/cpu/gpu_worker.py)

本模块实现了 GPU 和 CPU 之间 KV 缓存块的批量异步数据传输。

核心组件：
1. Transfer 数据类：描述一次传输任务（包含流、事件、传输大小等）
2. compute_sub_block_ptrs 函数：计算子块的内存地址指针
3. pin_mmap_region 函数：将 mmap 区域注册为 CUDA 锁页内存
4. SingleDirectionOffloadingHandler：单方向传输处理器（GPU->CPU 或 CPU->GPU）
5. CpuGpuOffloadingHandlers：双向传输处理器集合

数据传输流程：
1. 调度器调用 prepare_store/prepare_load 创建传输规格 (TransferSpec)
2. TransferSpec 包含源和目标的 BlockIDsLoadStoreSpec（指定块 ID）
3. 工作器调用 transfer_async 启动异步传输：
   a. 根据块 ID 计算源和目标的内存地址指针
   b. 从 CUDA 流池中获取或创建一个流
   c. 在流上记录开始事件
   d. 调用 ops.swap_blocks_batch 执行批量块拷贝
   e. 在流上记录结束事件
4. 工作器定期调用 get_finished 检查已完成的传输
5. 工作器调用 wait 等待特定传输完成（如需要同步）

性能优化：
- 使用 CUDA 流池避免频繁创建/销毁流
- 使用事件池避免频繁创建/销毁事件
- 批量传输减少内核启动开销
- 锁页内存 (pinned memory) 加速 CPU <-> GPU DMA 传输
- CPU->GPU 传输使用 CU_MEMCPY_SRC_ACCESS_ORDER_ANY 优化源读取流水线
"""

import time
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.v1.kv_offload.base import (
    BlockIDsLoadStoreSpec,
    CanonicalKVCacheRef,
    CanonicalKVCaches,
    GPULoadStoreSpec,
)
from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion
from vllm.v1.kv_offload.worker.worker import (
    OffloadingHandler,
    TransferResult,
    TransferSpec,
)

logger = init_logger(__name__)


@dataclass
class Transfer:
    """
    描述一次正在进行的数据传输任务。

    属性：
        job_id: 传输任务的唯一标识符
        stream: 执行传输的 CUDA 流
        start_event: 传输开始时记录的 CUDA 事件（用于计时）
        end_event: 传输结束时记录的 CUDA 事件（用于完成检测和计时）
        num_bytes: 本次传输的总字节数
    """
    job_id: int
    stream: torch.cuda.Stream
    start_event: torch.Event
    end_event: torch.Event
    num_bytes: int


def compute_sub_block_ptrs(
    block_ids: np.ndarray,
    block_size_factor: int,
    output: np.ndarray,
    tensor: torch.Tensor,
    skip_count: int = 0,
):
    """
    计算给定块 ID 对应的子块内存字节指针。

    背景：GPU 和 CPU 的缓存块大小可能不同。
    - GPU 块较小（gpu_page_size），CPU 块较大（cpu_page_size = gpu_page_size * block_size_factor）
    - 一个 CPU 块包含 block_size_factor 个 GPU 大小的子块
    - 传输时需要将 GPU 块映射到对应的 CPU 子块位置

    指针计算公式：
        子块 j 在块 b 中的地址 = base_ptr + b * row_stride + j * sub_block_size
        其中 sub_block_size = tensor.shape[1] // block_size_factor（即 GPU 页大小）

    特殊情况处理：
    - 当 block_size_factor == 1 时（1:1 映射），走快速路径
    - 当 row_stride != block_size_factor * sub_block_size 时（如非连续的 CPU 张量），
      需要正确的步幅计算

    Args:
        block_ids: 块 ID 数组（张量的原生粒度）
        block_size_factor: 每个块包含的子块数量
        output: 预分配的 int64 数组，用于写入计算结果指针
        tensor: 源或目标张量
        skip_count: 第一个块中需要跳过的子块数（用于部分块传输）
    """
    assert skip_count < block_size_factor

    num_sub_blocks = len(output)
    # 获取张量的基地址指针
    base_ptr = tensor.data_ptr()
    # 获取行步幅（每行的字节偏移）
    row_stride = tensor.stride(0)

    if block_size_factor == 1:
        # 快速路径：1:1 映射，无需子块扩展
        # 直接计算每个块的地址 = 基地址 + 块ID * 行步幅
        output[:] = base_ptr + block_ids[:num_sub_blocks] * row_stride
        return

    # 向量化扩展路径（block_size_factor > 1 的情况）
    # 验证张量列数能被 block_size_factor 整除
    assert tensor.shape[1] % block_size_factor == 0
    # 计算每个子块的字节大小
    sub_block_size = tensor.shape[1] // block_size_factor
    # 生成子块偏移数组：[0, sub_block_size, 2*sub_block_size, ...]
    sub_offsets = np.arange(block_size_factor, dtype=np.int64) * sub_block_size
    # 广播计算所有子块的指针地址：
    # (num_blocks, 1) + (1, block_size_factor) -> (num_blocks, block_size_factor)
    all_ptrs = (
        base_ptr + block_ids.astype(np.int64)[:, np.newaxis] * row_stride
    ) + sub_offsets[np.newaxis, :]
    # 展平为一维数组，并应用 skip_count 和截断
    flat = all_ptrs.ravel()
    output[:] = flat[skip_count : skip_count + num_sub_blocks]


def pin_mmap_region(region: SharedOffloadRegion) -> None:
    """
    将整个 mmap 区域注册为 CUDA 锁页内存 (pinned memory)。

    通过 cudaHostRegister 将已有的 mmap 内存区域注册为 CUDA 可访问的锁页内存。
    锁页内存的优势：
    1. DMA 传输不需要经过操作系统页面调度，速度更快
    2. 支持异步传输，可以与计算重叠
    3. GPU 可以直接通过 PCIe 访问锁页内存

    如果注册失败（如驱动不支持），传输仍然可以工作，但可能较慢（非锁页 DMA）。

    Args:
        region: 要注册的共享卸载内存区域
    """
    rank = region.rank

    base_ptr = region._base.data_ptr()
    # 调用 cudaHostRegister 将 mmap 内存注册为 CUDA 锁页内存
    # flags=0 表示默认行为
    result = torch.cuda.cudart().cudaHostRegister(base_ptr, region.total_size_bytes, 0)
    if result.value != 0:
        logger.warning(
            "cudaHostRegister failed for rank=%d (code=%d) — "
            "transfers will still work but may be slower (unpinned DMA)",
            rank,
            result,
        )
    else:
        logger.debug(
            "cudaHostRegister rank=%d %.2f GB",
            rank,
            region.total_size_bytes / 1e9,
        )
        region.is_pinned = True


class SingleDirectionOffloadingHandler(OffloadingHandler):
    """
    单方向数据传输处理器。

    处理 GPU -> CPU 或 CPU -> GPU 的单方向 KV 缓存块传输。
    特性：
    1. 传输保证按提交顺序执行
    2. 每次传输使用独立的 CUDA 流
    3. 当前传输的流会等待前一个传输的流完成后再开始执行
    4. 使用对象池复用 CUDA 流和事件，减少创建开销

    传输类型：
    - GPU -> CPU：用于将 KV 缓存从 GPU 卸载到 CPU 内存（offload/store）
    - CPU -> GPU：用于将 KV 缓存从 CPU 加载回 GPU（load）
    """

    def __init__(
        self,
        gpu_tensors: list[torch.Tensor],
        cpu_tensors: list[torch.Tensor],
        block_size_factor: int,
        kv_cache_groups_data_refs: list[list[CanonicalKVCacheRef]],
        gpu_to_cpu: bool,
        mmap_region: SharedOffloadRegion | None = None,
    ):
        """
        初始化单方向传输处理器。

        Args:
            gpu_tensors: GPU KV 缓存张量列表。
                每个张量形状为 (num_gpu_blocks, gpu_page_size_bytes)，dtype 为 int8。
            cpu_tensors: CPU KV 缓存张量列表。
                每个张量形状为 (num_cpu_blocks, cpu_page_size_bytes)，dtype 为 int8。
                顺序应与 gpu_tensors 匹配。
            block_size_factor: CPU 块与 GPU 块的大小比例。
                cpu_page_size = gpu_page_size * block_size_factor。
            kv_cache_groups_data_refs: 每个 KV 缓存组的 CanonicalKVCacheRef 列表。
                用于混合注意力模型（HMA）中不同注意力头组的缓存。
            gpu_to_cpu: 传输方向。True 表示 GPU->CPU，False 表示 CPU->GPU。
            mmap_region: 可选的共享 mmap 内存区域（用于多进程共享卸载）。
        """
        assert len(gpu_tensors) == len(cpu_tensors)
        assert len(gpu_tensors) > 0

        # 验证输入张量的形状和类型符合预期
        for gpu_tensor, cpu_tensor in zip(gpu_tensors, cpu_tensors):
            assert gpu_tensor.dtype == torch.int8
            assert gpu_tensor.ndim == 2
            assert gpu_tensor.is_cuda
            assert cpu_tensor.dtype == torch.int8
            assert cpu_tensor.ndim == 2
            assert cpu_tensor.device.type == "cpu"
            _, gpu_page_size = gpu_tensor.shape
            _, cpu_page_size = cpu_tensor.shape
            # 验证 CPU 页大小是 GPU 页大小的 block_size_factor 倍
            assert cpu_page_size == gpu_page_size * block_size_factor

        # 根据传输方向设置源和目标张量
        self.src_tensors: list[torch.Tensor] = (
            gpu_tensors if gpu_to_cpu else cpu_tensors
        )
        self.dst_tensors: list[torch.Tensor] = (
            cpu_tensors if gpu_to_cpu else gpu_tensors
        )
        self.gpu_to_cpu: bool = gpu_to_cpu
        self.kv_cache_groups_data_refs = kv_cache_groups_data_refs

        # 设置源和目标的块大小因子
        # GPU -> CPU：源是 GPU（因子=1），目标是 CPU（因子=block_size_factor）
        # CPU -> GPU：源是 CPU（因子=block_size_factor），目标是 GPU（因子=1）
        self.src_block_size_factor = 1 if self.gpu_to_cpu else block_size_factor
        self.dst_block_size_factor = block_size_factor if self.gpu_to_cpu else 1

        # 传输类型标识，用于日志和结果报告
        self.transfer_type = ("GPU", "CPU") if self.gpu_to_cpu else ("CPU", "GPU")
        # mmap 区域引用（gpu_to_cpu 处理器负责清理）
        self._mmap_region = mmap_region
        # job_id -> 结束事件的映射，用于 wait() 查询
        self._transfer_events: dict[int, torch.Event] = {}
        # 传输任务队列（FIFO），用于 get_finished() 按序检查
        self._transfers: deque[Transfer] = deque()
        # CUDA 流对象池，避免频繁创建/销毁
        self._stream_pool: list[torch.cuda.Stream] = []
        # CUDA 事件对象池，避免频繁创建/销毁
        self._event_pool: list[torch.Event] = []

    def transfer_async(self, job_id: int, transfer_spec: TransferSpec) -> bool:
        """
        异步启动一次数据传输。

        根据传输规格（源/目标块 ID），计算内存地址，在 CUDA 流上启动批量传输。

        传输规格说明：
        - TransferSpec 包含 (src_spec, dst_spec) 两个 BlockIDsLoadStoreSpec
        - 每个 spec 包含 block_ids（块 ID 数组）
        - 块 ID 可能来自不同的 KV 缓存组（用于混合注意力模型）

        边界情况处理：
        - 第一个和最后一个 CPU 块可能只部分填充（partial block）
        - 需要通过 skip_count 跳过部分子块
        - 多个 KV 缓存组各自独立处理 partial block

        Args:
            job_id: 本次传输的唯一任务 ID
            transfer_spec: 传输规格，包含源和目标的块 ID

        Returns:
            True 表示传输已成功启动（始终返回 True）
        """
        src_spec, dst_spec = transfer_spec
        assert isinstance(src_spec, BlockIDsLoadStoreSpec)
        assert isinstance(dst_spec, BlockIDsLoadStoreSpec)

        src_blocks = src_spec.block_ids
        dst_blocks = dst_spec.block_ids
        assert src_blocks.ndim == 1
        assert dst_blocks.ndim == 1

        num_src_blocks = len(src_blocks)
        num_dst_blocks = len(dst_blocks)

        # 传输有两种类型：
        # 1. GPU -> CPU（offload）
        # 2. CPU -> GPU（load）
        #
        # 传输的是 CPU 块，但第一个和最后一个 CPU 块可能只匹配
        # 较小（字节层面）的 GPU 块子集。
        # 此时需要跳过一些 GPU 大小的子块，从第一个 CPU 块的中间开始读写。
        # 如果有多个 KV 缓存组（混合注意力模型使用 HMA 时），
        # 每个组可能各自有 partial first/last CPU 块。
        # group_sizes 参数编码每个组在 GPU dst_blocks 中的大小。
        # 如果 group_sizes 为 None，假设所有块属于单个组。
        # logical_offset 参数将每组块映射到请求中的逻辑偏移（以 GPU 块为单位），
        # 用于在匹配的第一个 CPU 块中找到正确的起始位置。

        # 从 GPU spec 中提取 group_sizes
        gpu_spec = src_spec if self.gpu_to_cpu else dst_spec
        assert isinstance(gpu_spec, GPULoadStoreSpec)
        group_sizes = gpu_spec.group_sizes
        assert len(group_sizes) == len(self.kv_cache_groups_data_refs)

        # 从 GPU spec 中提取块索引
        block_indices = gpu_spec.block_indices
        assert len(block_indices) == len(self.kv_cache_groups_data_refs)

        # 计算总拷贝操作数
        num_copy_ops = 0
        for group_size, group_data_refs in zip(
            group_sizes, self.kv_cache_groups_data_refs
        ):
            num_copy_ops += group_size * len(group_data_refs)

        # 预分配源指针、目标指针和大小数组
        all_src = np.empty(num_copy_ops, dtype=np.int64)
        all_dst = np.empty(num_copy_ops, dtype=np.int64)
        all_sizes = np.empty(num_copy_ops, dtype=np.int64)

        src_offset = 0
        dst_offset = 0
        op_idx = 0
        # 统计传输的总字节数
        num_transfer_bytes = 0
        for group_size, block_idx, group_data_refs in zip(
            group_sizes, block_indices, self.kv_cache_groups_data_refs
        ):
            if group_size == 0:
                continue

            # 计算需要跳过的逻辑块数（用于 partial block 处理）
            src_logical_blocks_to_skip = block_idx % self.src_block_size_factor
            dst_logical_blocks_to_skip = block_idx % self.dst_block_size_factor
            src_logical_blocks_count = group_size + src_logical_blocks_to_skip
            dst_logical_blocks_count = group_size + dst_logical_blocks_to_skip

            # 计算实际需要的块数（向上取整）
            dst_blocks_count = cdiv(
                dst_logical_blocks_count, self.dst_block_size_factor
            )
            dst_end_offset = dst_offset + dst_blocks_count
            assert dst_end_offset <= num_dst_blocks

            src_blocks_count = cdiv(
                src_logical_blocks_count, self.src_block_size_factor
            )
            src_end_offset = src_offset + src_blocks_count
            assert src_end_offset <= num_src_blocks

            # 提取当前组的源和目标块
            group_src = src_blocks[src_offset:src_end_offset]
            group_dst = dst_blocks[dst_offset:dst_end_offset]

            for data_ref in group_data_refs:
                t_idx = data_ref.tensor_idx
                end_idx = op_idx + group_size

                # 计算源子块的内存地址指针
                compute_sub_block_ptrs(
                    group_src,
                    self.src_block_size_factor,
                    all_src[op_idx:end_idx],
                    self.src_tensors[t_idx],
                    skip_count=src_logical_blocks_to_skip,
                )
                # 计算目标子块的内存地址指针
                compute_sub_block_ptrs(
                    group_dst,
                    self.dst_block_size_factor,
                    all_dst[op_idx:end_idx],
                    self.dst_tensors[t_idx],
                    skip_count=dst_logical_blocks_to_skip,
                )

                # 记录每个拷贝操作的字节大小
                all_sizes[op_idx:end_idx] = data_ref.page_size_bytes
                num_transfer_bytes += group_size * data_ref.page_size_bytes
                op_idx = end_idx

            src_offset = src_end_offset
            dst_offset = dst_end_offset

        # 验证所有块都已处理
        assert src_offset == num_src_blocks
        assert dst_offset == num_dst_blocks
        assert op_idx == num_copy_ops

        # 将 numpy 数组转换为 torch 张量，用于 CUDA 内核调用
        batch_src = torch.from_numpy(all_src)
        batch_dst = torch.from_numpy(all_dst)
        batch_sizes = torch.from_numpy(all_sizes)

        # 从对象池获取或创建 CUDA 流和事件
        stream = self._stream_pool.pop() if self._stream_pool else torch.cuda.Stream()
        start_event = (
            self._event_pool.pop()
            if self._event_pool
            else torch.Event(enable_timing=True)
        )
        end_event = (
            self._event_pool.pop()
            if self._event_pool
            else torch.Event(enable_timing=True)
        )

        if self.gpu_to_cpu:
            # GPU -> CPU：需要等待模型计算完成后再开始卸载
            # 避免读取正在被计算使用的 KV 缓存数据
            stream.wait_stream(torch.cuda.current_stream())
        if self._transfers:
            last_transfer: Transfer = self._transfers[-1]
            last_event = last_transfer.end_event
            # 确保当前传输在前一个传输完成后再开始
            # 保持传输的顺序性
            stream.wait_event(last_event)
        # CPU->GPU 从主机锁页内存读取，该内存不会被其他 GPU 流并发写入，
        # 因此 CU_MEMCPY_SRC_ACCESS_ORDER_ANY 是安全的，允许驱动流水线化源读取。
        # GPU->CPU 从活动的 GPU KV 缓存读取，计算流会持续写入；
        # 必须保持 STREAM 顺序，以便源读取受传输流的 wait_stream(compute) 屏障约束。
        is_src_access_order_any = not self.gpu_to_cpu
        with torch.cuda.stream(stream):
            start_event.record(stream)
            if num_copy_ops > 0:
                # 调用自定义批量块拷贝操作
                ops.swap_blocks_batch(
                    batch_src,
                    batch_dst,
                    batch_sizes,
                    is_src_access_order_any=is_src_access_order_any,
                )
            end_event.record(stream)

        # 记录传输信息
        self._transfer_events[job_id] = end_event
        self._transfers.append(
            Transfer(
                job_id=job_id,
                stream=stream,
                start_event=start_event,
                end_event=end_event,
                num_bytes=num_transfer_bytes,
            )
        )

        # 传输已成功启动
        return True

    def get_finished(self) -> list[TransferResult]:
        """
        检查并收集已完成的传输任务。

        从传输队列头部开始检查，如果传输的结束事件已完成（query() 返回 True），
        则将其从队列中移除并收集结果。已用完的流和事件会被归还到对象池中复用。

        Returns:
            已完成的传输结果列表，每个结果包含 job_id、成功状态、传输大小和耗时。
        """
        results: list[TransferResult] = []
        while self._transfers and self._transfers[0].end_event.query():
            transfer = self._transfers.popleft()
            # 计算传输耗时（elapsed_time 返回毫秒，转换为秒）
            transfer_time = (
                transfer.start_event.elapsed_time(transfer.end_event) * 1e-3
            )  # elapsed_time is in milliseconds
            result = TransferResult(
                job_id=transfer.job_id,
                success=True,
                transfer_size=transfer.num_bytes,
                transfer_time=transfer_time,
                transfer_type=self.transfer_type,
            )

            results.append(result)
            # 归还流和事件到对象池，供后续传输复用
            self._stream_pool.append(transfer.stream)
            self._event_pool.append(transfer.end_event)
            self._event_pool.append(transfer.start_event)
            del self._transfer_events[transfer.job_id]
        return results

    def wait(self, job_ids: set[int]):
        """
        等待指定的传输任务完成。

        通过同步对应的结束事件来阻塞等待。
        用于需要确保数据传输完成后再进行后续操作的场景。

        Args:
            job_ids: 需要等待完成的传输任务 ID 集合
        """
        for job_id in job_ids:
            event = self._transfer_events.get(job_id)
            if event is not None:
                event.synchronize()

    def shutdown(self) -> None:
        """
        关闭传输处理器。

        执行清理操作：
        1. 等待所有正在进行的传输完成
        2. 清理所有状态和对象池
        3. 如果拥有 mmap 区域，执行清理释放资源
        """
        while self._transfers:
            transfer = self._transfers.popleft()
            transfer.end_event.synchronize()
        self._transfer_events.clear()
        self._stream_pool.clear()
        self._event_pool.clear()
        self.src_tensors.clear()
        self.dst_tensors.clear()
        if self._mmap_region is not None:
            self._mmap_region.cleanup()
            self._mmap_region = None


class CpuGpuOffloadingHandlers:
    """
    GPU <-> CPU 双向数据传输处理器集合。

    管理 GPU 和 CPU 之间的双向 KV 缓存传输：
    - gpu_to_cpu_handler: GPU -> CPU 方向（卸载/offload）
    - cpu_to_gpu_handler: CPU -> GPU 方向（加载/load）

    构造时负责：
    1. 分配或映射 CPU 端的 KV 缓存内存
    2. 如果有 mmap 区域，将其注册为 CUDA 锁页内存
    3. 创建两个方向的 SingleDirectionOffloadingHandler
    """

    def __init__(
        self,
        kv_caches: CanonicalKVCaches,
        block_size_factor: int,
        num_cpu_blocks: int,
        mmap_region: SharedOffloadRegion | None = None,
    ):
        pin_memory = is_pin_memory_available()
        logger.info("Allocating %d CPU tensors...", len(kv_caches.tensors))
        self._mmap_region = mmap_region
        # 如果有 mmap 区域且支持锁页内存，将其注册为 CUDA 锁页内存
        if mmap_region is not None and pin_memory:
            pin_mmap_region(mmap_region)

        gpu_tensors: list[torch.Tensor] = []
        cpu_tensors: list[torch.Tensor] = []
        for kv_cache_tensor in kv_caches.tensors:
            gpu_page_size_bytes = kv_cache_tensor.page_size_bytes
            # 将 GPU KV 缓存张量重塑为 (num_blocks, page_size_bytes) 的 int8 视图
            gpu_tensor = kv_cache_tensor.tensor.view(torch.int8).view(
                (-1, gpu_page_size_bytes)
            )
            # CPU 页大小 = GPU 页大小 * block_size_factor
            cpu_page_size_bytes = gpu_page_size_bytes * block_size_factor

            if mmap_region is not None:
                # 使用共享 mmap 区域创建 CPU 缓存视图
                cpu_tensor = mmap_region.create_next_view(cpu_page_size_bytes)
            else:
                # 分配独立的 CPU 缓存内存
                t0 = time.monotonic()
                cpu_tensor = torch.zeros(
                    (num_cpu_blocks, cpu_page_size_bytes),
                    dtype=torch.int8,
                    device="cpu",
                    pin_memory=pin_memory,
                )
                logger.debug(
                    "torch.zeros pinned tensor %d×%d (%.2f GB): %.3f s",
                    num_cpu_blocks,
                    cpu_page_size_bytes,
                    num_cpu_blocks * cpu_page_size_bytes / 1e9,
                    time.monotonic() - t0,
                )

            gpu_tensors.append(gpu_tensor)
            cpu_tensors.append(cpu_tensor)

        # 创建 GPU -> CPU 方向的传输处理器（负责 mmap 区域的清理）
        self.gpu_to_cpu_handler = SingleDirectionOffloadingHandler(
            gpu_tensors=gpu_tensors,
            cpu_tensors=cpu_tensors,
            block_size_factor=block_size_factor,
            kv_cache_groups_data_refs=kv_caches.group_data_refs,
            gpu_to_cpu=True,
            mmap_region=mmap_region,
        )

        # 创建 CPU -> GPU 方向的传输处理器
        self.cpu_to_gpu_handler = SingleDirectionOffloadingHandler(
            gpu_tensors=gpu_tensors,
            cpu_tensors=cpu_tensors,
            block_size_factor=block_size_factor,
            kv_cache_groups_data_refs=kv_caches.group_data_refs,
            gpu_to_cpu=False,
        )
