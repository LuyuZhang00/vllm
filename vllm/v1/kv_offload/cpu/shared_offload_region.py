# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
共享卸载内存区域模块 (vllm/v1/kv_offload/cpu/shared_offload_region.py)

本模块实现了基于 mmap 的多进程共享内存区域，用于 KV 缓存卸载。

共享内存的设计目标：
- 多个工作器进程共享同一块 CPU 内存区域来存储卸载的 KV 缓存
- 避免每个进程独立分配内存，节省系统内存
- 通过内存映射文件实现进程间零拷贝数据共享

内存布局：
```
worker0_block0 | worker1_block0 | ... | worker{M-1}_block0
worker0_block1 | worker1_block1 | ... | worker{M-1}_block1
...
```
- 每行（block 行）包含所有工作器在该块上的数据
- 行步幅 = cpu_page_size * num_workers
- 每个工作器占据每个行中的一个固定偏移位置

协调机制：
1. 第一个打开文件的工作器（O_EXCL 成功）成为创建者，负责 ftruncate 设置文件大小
2. 其他工作器等待文件达到预期大小后再 mmap
3. 使用 MADV_POPULATE_WRITE 预分配页面，避免 page fault 延迟

关键特性：
- 使用 /dev/shm（tmpfs）作为存储后端，保证是内存文件系统
- 支持注册为 CUDA 锁页内存以加速 DMA 传输
- cleanup 时由创建者负责删除文件
"""

import mmap
import os
import time

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


def _wait_for_file_size(fd: int, expected_size: int, timeout: float = 30.0) -> None:
    """
    自旋等待文件达到预期大小。

    非创建者的工作器需要等待创建者完成 ftruncate 后才能 mmap。
    每 5ms 检查一次文件大小，超时后抛出异常。

    Args:
        fd: 文件描述符
        expected_size: 期望的文件大小（字节）
        timeout: 超时时间（秒），默认 30 秒

    Raises:
        TimeoutError: 超时后文件仍未达到预期大小
    """
    deadline = time.monotonic() + timeout
    while True:
        if os.fstat(fd).st_size >= expected_size:
            return
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"Timed out waiting for mmap file to reach {expected_size} bytes"
            )
        time.sleep(0.005)


class SharedOffloadRegion:
    """
    基于 mmap 的多进程共享内存区域。

    为一个 vLLM 实例的所有工作器提供单一的、共享的内存映射区域。
    工作器通过文件系统协调：
    1. 第一个以 O_EXCL 标志打开文件的工作器成为创建者，调用 ftruncate 设置大小
    2. 其他工作器打开已存在的文件，等待文件达到预期大小
    3. 所有工作器 mmap 整个文件

    文件路径：/dev/shm/vllm_offload_{instance_id}.mmap

    参数：
        instance_id: vLLM 实例的唯一标识符
        total_size_bytes: 共享区域的总字节数
        num_blocks: KV 缓存块的数量
        rank: 当前工作器的编号（None 表示无多工作器）
        num_workers: 工作器总数
        cpu_page_size: 每个工作器在每个块上的数据大小（字节）
    """

    def __init__(
        self,
        instance_id: str,
        total_size_bytes: int,
        num_blocks: int,
        rank: int | None,
        num_workers: int,
        cpu_page_size: int,
    ) -> None:
        # 系统页面大小（通常 4KB）
        self.page_size = mmap.PAGESIZE

        self.total_size_bytes = total_size_bytes
        # mmap 文件路径，使用 /dev/shm（Linux 共享内存文件系统）
        self.mmap_path = f"/dev/shm/vllm_offload_{instance_id}.mmap"
        # 标记当前进程是否是文件的创建者
        self._creator = False
        self.num_blocks = num_blocks
        self.rank = rank
        # 交错布局的行步幅：一行 = 所有工作器在一个块上的数据
        self._row_stride = cpu_page_size * num_workers
        if rank is not None:
            # 当前工作器在每个块行中的字节偏移
            self._worker_offset = rank * cpu_page_size
            # 当前工作器在每个行中的独占区域上界
            self._worker_area_end = (rank + 1) * cpu_page_size
        try:
            # 独占创建 —— 只有一个工作器能成功
            self.fd: int | None = os.open(
                self.mmap_path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600
            )
            # 设置文件大小
            os.ftruncate(self.fd, self.total_size_bytes)
            self._creator = True
            logger.info(
                "Created mmap file %s (%.2f GB)",
                self.mmap_path,
                self.total_size_bytes / 1e9,
            )
        except FileExistsError:
            # 文件已存在，说明其他工作器已创建
            self.fd = os.open(self.mmap_path, os.O_RDWR)
            # 等待文件达到预期大小
            _wait_for_file_size(self.fd, self.total_size_bytes)
            logger.info("Opened existing mmap file %s", self.mmap_path)

        # 创建 mmap 对象，使用 MAP_SHARED 实现多进程共享
        self.mmap_obj: mmap.mmap | None = mmap.mmap(
            self.fd,
            self.total_size_bytes,
            flags=mmap.MAP_SHARED,
            prot=mmap.PROT_READ | mmap.PROT_WRITE,
        )

        # MADV_POPULATE_WRITE 在 Linux 5.14 中添加（值为 23）。
        # 此 madvise 标志会预分配页面，避免后续写入时的 page fault。
        _MADV_POPULATE_WRITE = getattr(mmap, "MADV_POPULATE_WRITE", 23)
        if rank is not None:
            # 仅预分配当前工作器的页面（每个块行中一个槽位）
            worker_offset = rank * cpu_page_size
            _t0 = time.perf_counter()
            page_size = self.page_size
            for block in range(num_blocks):
                raw_offset = block * self._row_stride + worker_offset
                # 对齐到页面边界
                aligned_offset = (raw_offset // page_size) * page_size
                end = raw_offset + cpu_page_size
                aligned_length = end - aligned_offset
                self.mmap_obj.madvise(
                    _MADV_POPULATE_WRITE, aligned_offset, aligned_length
                )
            logger.debug(
                "MADV_POPULATE_WRITE loop: %d blocks in %.3f s",
                num_blocks,
                time.perf_counter() - _t0,
            )
        else:
            # 无 rank —— 一次性预分配整个共享区域
            _t0 = time.perf_counter()
            self.mmap_obj.madvise(_MADV_POPULATE_WRITE, 0, self.total_size_bytes)
            logger.debug(
                "MADV_POPULATE_WRITE entire region: %.3f s", time.perf_counter() - _t0
            )

        # 创建基础 int8 张量，引用 mmap 内存
        self._base = torch.frombuffer(memoryview(self.mmap_obj), dtype=torch.int8)
        # 所有视图张量的列表
        self._views: list[torch.Tensor] = []
        # 是否已注册为 CUDA 锁页内存
        self.is_pinned: bool = False

    def create_next_view(self, tensor_page_size: int) -> torch.Tensor:
        """
        为当前工作器创建下一个 KV 缓存张量的跨步视图。

        必须为每个规范张量调用一次。完整的 mmap 内存布局如下：

            worker0_block0 | worker1_block0 | ... | worker{M-1}_block0
            worker0_block1 | worker1_block1 | ... | worker{M-1}_block1
            ...

        每个 worker_block 单元大小为 cpu_page_size 字节，包含该工作器和块的
        所有规范张量数据（拼接存储）：
            [ tensor0_data | tensor1_data | ... | tensor{L-1}_data ]

        相邻行之间的步幅为 row_stride = cpu_page_size * M。

        返回一个 int8 张量，形状为 (num_blocks, tensor_page_size)，
        步幅为 (row_stride, 1)。使用 int8 保持步幅 == 字节数，
        这样 swap_blocks 的地址算术无需任何 dtype 转换。

        Args:
            tensor_page_size: 此张量每个块的字节数

        Returns:
            跨步 int8 张量视图，形状 (num_blocks, tensor_page_size)
        """
        assert self.rank is not None
        new_offset = self._worker_offset + tensor_page_size
        assert new_offset <= self._worker_area_end, (
            f"Worker offset {new_offset} exceeds worker area end "
            f"{self._worker_area_end} (overflowed by "
            f"{new_offset - self._worker_area_end} bytes)"
        )
        # 创建跨步视图，步幅为 (row_stride, 1)，偏移为当前工作器偏移
        worker_layer_view = torch.as_strided(
            self._base,
            size=(self.num_blocks, tensor_page_size),
            stride=(self._row_stride, 1),
            storage_offset=self._worker_offset,
        )
        # 更新偏移，供下一个张量使用
        self._worker_offset = new_offset
        self._views.append(worker_layer_view)
        return worker_layer_view

    def create_kv_memoryview(self) -> memoryview:
        """
        返回整个 KV 缓冲区的零拷贝 memoryview。

        形状：(num_blocks, row_stride_bytes)。二级分层通过 view[b] 访问块 b。

        用于将 KV 数据暴露给其他分层存储后端（如文件系统），
        无需数据拷贝即可读取/写入。

        Returns:
            整个 KV 缓冲区的 memoryview 视图
        """
        kv_tensor = self._base.view(self.num_blocks, self._row_stride)
        np_arr = kv_tensor.numpy()
        # 验证是零拷贝视图，而非数据拷贝
        assert np_arr.ctypes.data == self._base.data_ptr(), (
            "view()/numpy() created a copy instead of sharing the mmap buffer; "
            "secondary tiers require zero-copy access to primary KV data"
        )
        return memoryview(np_arr)

    def cleanup(self) -> None:
        """
        清理共享内存区域。

        执行顺序：
        1. 如果已注册为 CUDA 锁页内存，先取消注册
        2. 释放所有视图张量（每个视图持有 _base 的引用和直接 StorageImpl 引用）
        3. 释放基础张量
        4. 关闭 mmap 对象
        5. 关闭文件描述符
        6. 如果是创建者，删除 mmap 文件

        注意：必须先释放视图再释放 _base，因为视图持有 _base 的引用计数。
        """
        if self.is_pinned and self._base is not None:
            base_ptr = self._base.data_ptr()
            result = torch.cuda.cudart().cudaHostUnregister(base_ptr)
            if result.value != 0:
                logger.warning(
                    "cudaHostUnregister failed for rank=%d (code=%d)", self.rank, result
                )
            self.is_pinned = False
        # 先释放视图再释放 _base：每个视图持有 _base 引用和直接 StorageImpl 引用。
        # 先释放视图让两个引用计数都下降，这样 StorageImpl（持有 mmap_obj buffer export）
        # 在 mmap_obj.close() 之前被释放。
        if self._views is not None:
            self._views.clear()
        self._base = None
        if self.mmap_obj:
            try:
                self.mmap_obj.close()
            except Exception:
                logger.warning("Failed to close mmap_obj", exc_info=True)
            self.mmap_obj = None
        if self.fd is not None:
            try:
                os.close(self.fd)
            except Exception:
                logger.warning("Failed to close fd %s", self.fd, exc_info=True)
            self.fd = None
        # 创建者负责删除 mmap 文件
        if self._creator and getattr(self, "mmap_path", None):
            try:
                os.unlink(self.mmap_path)
                logger.info("Removed mmap file %s", self.mmap_path)
            except Exception:
                logger.warning(
                    "Failed to unlink path %s", self.mmap_path, exc_info=True
                )
            self._creator = False
