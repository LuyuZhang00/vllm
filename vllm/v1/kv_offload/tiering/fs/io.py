# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
KV cache block 的磁盘 I/O 操作模块。

本模块提供 KV cache block 在文件系统二级层的读写底层操作：
  - store_block(): 将一个 KV block 从内存（一级层 memoryview）写入磁盘文件。
  - load_block():  将一个 KV block 从磁盘文件读回内存（一级层 memoryview）。

设计要点：
  1. 原子写入（Atomic Write）：store_block 先写入临时文件，再 os.replace
     原子替换到目标路径。这样即使写入过程中进程崩溃，也不会产生损坏的
     半写文件，保证了数据一致性。
  2. O_DIRECT 标志：使用 Linux 的 O_DIRECT 绕过内核页缓存（page cache），
     直接在用户态 buffer 和 磁盘之间传输数据。这对于大块连续数据（KV cache）
     可以避免双重缓存（double buffering）问题，提高 I/O 效率。
  3. 容错清理：读写失败时会清理临时文件或损坏文件，避免磁盘空间泄漏。
  4. 线程安全：通过线程本地存储为临时文件生成唯一后缀，避免多线程竞争。

这两个函数作为回调（callback）传入 DualQueueThreadPool，在 I/O 线程中执行。

模块导出接口：
  - store_block(dest_path, buffer, offset, block_size) -> None
  - load_block(source_path, view, offset, block_size) -> None

使用示例（伪代码）：

  # 存储 KV block 到磁盘
  store_block(dest_path="/data/kv_cache/abc/ab_g0/block_42",
              buffer=cpu_kv_cache_memoryview,
              offset=block_id * block_size,
              block_size=block_size_bytes)

  # 从磁盘加载 KV block
  load_block(source_path="/data/kv_cache/abc/ab_g0/block_42",
             view=cpu_kv_cache_memoryview,
             offset=block_id * block_size,
             block_size=block_size_bytes)
"""

# ===========================================================================
# 文件概述说明
# ===========================================================================
# 本文件是 vLLM KV cache 分层存储（tiering）架构中的底层 I/O 模块。
#
# 在 vLLM 的 KV cache 三级存储架构中：
#   - 第一级（GPU 显存）：由 KV cache manager 直接管理，速度最快但容量有限。
#   - 第二级（CPU 内存）：由 block pool 管理，容量较大但需要 CPU-GPU 数据传输。
#   - 第三级（磁盘/SSD）：由文件系统层管理，容量最大但 I/O 延迟最高。
#
# 本文件的职责是实现第二级和第三级之间的数据传输：
#   - 当 KV cache 被逐出（evict）到磁盘时，调用 store_block() 将内存中的
#     block 数据持久化到磁盘文件。
#   - 当 KV cache 需要被提升（promote）回内存时，调用 load_block() 将磁盘
#     文件中的 block 数据读取回内存。
#
# 调用链路：
#   FileBasedTieringBackend
#     -> DualQueueThreadPool (I/O 线程池)
#       -> store_block() / load_block() (本文件中的函数)
#
# 设计约束：
#   1. 函数签名固定：本模块的函数必须符合 DualQueueThreadPool 要求的
#      callback 签名，即 (path, buffer, offset, block_size) -> None。
#   2. O_DIRECT 兼容：使用 O_DIRECT 时，buffer 地址和写入大小必须对齐到
#      文件系统逻辑块大小（通常为 512 字节或 4096 字节）。
#   3. 线程安全：多个 I/O 线程可能并发调用这些函数，必须保证无共享状态。
# ===========================================================================

# ===========================================================================
# 模块依赖说明
# ===========================================================================
# 本模块使用的标准库模块：
#   - logging: 日志记录，用于报告清理失败等警告信息。
#   - os: 底层文件操作（open, write, readv, replace, remove, makedirs）。
#   - random: 生成随机数，用于临时文件后缀。
#   - threading: 线程本地存储（thread-local），确保多线程环境下临时文件名唯一。
# ===========================================================================

import logging
import os
import random
import threading

# ===========================================================================
# 模块级日志记录器
# ===========================================================================
# 中文注释：获取本模块的日志记录器。
# 所有日志消息都会记录到 vllm.v1.kv_offload.tiering.fs.io 这个命名空间下。
# 在生产环境中，日志级别通常为 INFO 或 WARNING，用于记录 I/O 错误和警告。
logger = logging.getLogger(__name__)

# ===========================================================================
# 常量定义：O_DIRECT 标志
# ===========================================================================
# O_DIRECT is Linux-specific and not available on macOS
# 中文注释：Linux 特有的 O_DIRECT 标志，用于绕过内核页缓存。
# 在 macOS 上不可用，这里通过 getattr 提供默认值 0 以保证跨平台兼容。
#
# O_DIRECT 的作用：
#   - 普通文件 I/O 会经过内核页缓存（page cache），数据先从磁盘读到内核缓冲区，
#     再拷贝到用户空间缓冲区（双重拷贝）。
#   - O_DIRECT 绕过页缓存，直接在用户态 buffer 和磁盘之间传输数据。
#   - 对于 KV cache 这种大块数据（通常 1MB+），使用 O_DIRECT 可以：
#     a) 避免页缓存污染，不占用宝贵的内核内存。
#     b) 减少一次内存拷贝，提高 I/O 吞吐量。
#     c) 避免双重缓存（double buffering）——数据同时存在于用户态和内核态。
#
# 注意事项：
#   - 使用 O_DIRECT 时，buffer 地址和 I/O 大小必须对齐到文件系统逻辑块大小。
#     如果不对齐，会导致 EINVAL 错误或数据损坏。
#   - 在非 Linux 系统上，O_DIRECT 被设为 0，相当于不使用该标志。
# ===========================================================================
O_DIRECT = getattr(os, "O_DIRECT", 0)

# ===========================================================================
# 模块级状态：线程本地存储
# ===========================================================================
# Thread-local storage for unique temporary file suffixes
# 中文注释：线程本地存储，用于为每个 I/O 线程生成唯一的临时文件后缀。
# 这样不同线程写入同一目标文件时不会产生临时文件名冲突。
#
# 为什么使用线程本地存储：
#   - 当多个 I/O 线程并发执行 store_block() 时，如果临时文件名不唯一，
#     可能导致一个线程覆盖另一个线程的临时文件。
#   - 每个线程有自己的后缀，确保临时文件名在同一目标路径下唯一。
#   - 使用 threading.local() 而非全局锁，避免锁竞争影响 I/O 性能。
# ===========================================================================
_thread_local = threading.local()


# ===========================================================================
# 辅助函数 1：生成临时文件后缀
# ===========================================================================
def _get_tmp_suffix() -> str:
    """Generate a thread-local unique suffix for temporary files."""
    # 中文注释：为临时文件生成线程本地的唯一后缀。
    #
    # 工作原理：
    #   - 每个线程首次调用时，生成一个随机的 63 位整数后缀（如 "_123456789.tmp"）。
    #   - 后续调用直接从线程本地缓存中读取，无需重新生成。
    #   - 不同线程的后缀相互独立，避免多线程并发写入时临时文件名冲突。
    #
    # 为什么需要唯一后缀：
    #   当多个 I/O 线程同时写入同一个目标文件的不同 block 时，
    #   临时文件名必须唯一，否则会互相覆盖导致数据丢失。
    #
    # 性能优化：
    #   使用 try-except 而非 hasattr 来检测属性是否存在。
    #   在 Python 中，try-except 的异常处理比 hasattr 更高效，
    #   因为大多数情况下属性已经存在，不会触发异常。
    try:
        return _thread_local.tmp_suffix
    except AttributeError:
        _thread_local.tmp_suffix = f"_{random.randint(0, 2**63 - 1)}.tmp"
        return _thread_local.tmp_suffix


# ===========================================================================
# 辅助函数 2：确保目录存在
# ===========================================================================
def _ensure_dirs(path: str) -> None:
    """Create parent directories of *path* if they don't exist."""
    # 中文注释：确保目标文件的父目录存在。
    # 文件系统二级层采用分层目录结构（如 <base>_r<rank>/<hhh>/<hh>_g<group_idx>/），
    # 首次写入某个 block 时需要自动创建中间目录。
    #
    # 目录结构示例：
    #   假设 base_dir = /data/kv_cache，rank = 0，group_idx = 1
    #   则目标路径可能是：/data/kv_cache_r0/abc/ab_g1/block_42
    #   其中：
    #     /data/kv_cache_r0/  - 按 rank 分隔的根目录
    #     /abc/               - 三级哈希目录（由 block hash 的高 12 位生成）
    #     /ab_g1/             - 二级哈希目录 + group 索引
    #     /block_42           - 实际的 block 文件
    #
    # 使用 exist_ok=True 避免目录已存在时报错，提高并发安全性。
    os.makedirs(os.path.dirname(path), exist_ok=True)


# ===========================================================================
# 核心函数 1：store_block - 将 KV block 写入磁盘
# ===========================================================================
def store_block(
    dest_path: str,
    buffer: memoryview,
    offset: int,
    block_size: int,
) -> None:
    """
    Store callback: Writes to a temp file then atomically replaces the destination.
    """
    # 中文注释：将一个 KV block 从内存写入磁盘文件（级联存储路径）。
    #
    # 执行流程：
    #   1. 检查目标文件是否已存在（避免重复写入，实现幂等性）。
    #   2. 生成唯一的临时文件路径（<dest_path>.<random>.tmp）。
    #   3. 确保父目录存在。
    #   4. 将 memoryview 中的 block 数据（按 offset 和 block_size 切片）
    #      写入临时文件，使用 O_DIRECT 绕过内核页缓存。
    #   5. 原子替换：os.replace 将临时文件移动到目标路径。
    #      这保证了目标文件要么是旧的完整内容，要么是新的完整内容，
    #      不会出现半写（partial write）的不一致状态。
    #   6. 如果写入失败，清理临时文件后重新抛出异常。
    #
    # 参数说明：
    #   dest_path:  目标文件路径，由 FileMapper.get_file_name() 生成。
    #   buffer:     一级层 CPU KV cache 的 memoryview（多维，itemsize > 1）。
    #   offset:     起始字节偏移，等于 block_id * block_size。
    #   block_size: 一个 KV block 的字节数。
    #
    # 返回值：无（None）
    #
    # 异常处理：
    #   - 如果写入过程中发生任何异常，会清理临时文件后重新抛出。
    #   - 调用者（DualQueueThreadPool）会捕获异常并通知上层。

    # Check if block already exists to avoid redundant writes
    # 中文注释：如果目标文件已存在，说明该 block 之前已成功存储，直接跳过。
    # 这实现了幂等性——重复调用不会产生副作用。
    #
    # 为什么需要幂等性：
    #   在分层存储系统中，同一个 block 可能被多次请求存储（例如重试场景），
    #   幂等性确保重复存储不会导致错误或数据不一致。
    if os.path.exists(dest_path):
        return

    # 中文注释：生成唯一的临时文件路径。
    # 临时文件名 = 目标路径 + 线程唯一后缀（如 ".123456789.tmp"）。
    # 这确保即使多个线程同时写入同一个目标文件，临时文件也不会冲突。
    tmp_path = dest_path + _get_tmp_suffix()

    # Ensure parent directories exist
    # 中文注释：确保目标文件的父目录存在。
    # 在分层目录结构中，某些目录可能尚未创建，需要动态创建。
    _ensure_dirs(dest_path)

    # Write block atomically. Cast to a flat byte view so the slice uses byte
    # indices; the raw memoryview may be multi-dimensional with itemsize > 1.
    # 中文注释：将 memoryview 转换为一维字节视图后切片。
    # 因为 KV cache 的 memoryview 通常是多维的（如 [layer, head, seq, dim]），
    # 其 itemsize 可能大于 1 字节，直接切片会产生错误的字节偏移。
    # cast("B") 将其强制视为字节数组，确保 offset 和 block_size 以字节为单位。
    view_slice = buffer.cast("B")[offset : offset + block_size]

    try:
        # 中文注释：打开临时文件用于写入。
        # 文件打开标志说明：
        #   - O_CREAT: 如果文件不存在则创建。
        #   - O_EXCL: 与 O_CREAT 配合，确保文件必须是新创建的（防止覆盖已有文件）。
        #   - O_WRONLY: 只写模式。
        #   - O_TRUNC: 如果文件已存在则截断（但 O_EXCL 已确保不会发生）。
        #   - O_DIRECT: 绕过内核页缓存，直接写入磁盘。
        # 权限 0o644: 所有者可读写，组用户和其他用户只读。
        fd = os.open(
            tmp_path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_TRUNC | O_DIRECT,
            0o644,
        )
        try:
            # 中文注释：将 view_slice 数据写入临时文件。
            # os.write 返回实际写入的字节数，需要校验是否全部写入。
            written = os.write(fd, view_slice)
            if written < len(view_slice):
                raise OSError(
                    f"Short write: expected {len(view_slice)} bytes, wrote {written}"
                )
        finally:
            # 中文注释：确保文件描述符被关闭，即使写入失败也会执行。
            # 文件描述符泄漏会导致进程的文件描述符耗尽。
            os.close(fd)

        # 中文注释：原子替换——将临时文件移动到目标路径。
        # os.replace 在同一文件系统上是原子操作（rename 系统调用），
        # 保证读取方永远不会看到半写入的文件。
        #
        # 为什么使用原子替换：
        #   如果直接写入目标文件，写入过程中进程崩溃或断电会导致文件损坏。
        #   使用临时文件 + 原子替换，要么看到旧文件，要么看到完整的新文件。
        os.replace(tmp_path, dest_path)

    except Exception:
        # 中文注释：写入失败时清理临时文件，避免磁盘空间泄漏。
        # 使用 try-except 包裹 os.remove，因为清理操作本身也可能失败
        # （例如文件已被其他进程删除）。
        try:
            os.remove(tmp_path)
        except OSError as cleanup_exc:
            logger.warning("Failed to remove temp file %s: %s", tmp_path, cleanup_exc)
        # 中文注释：重新抛出原始异常，让调用者知道写入失败。
        raise


# ===========================================================================
# 核心函数 2：load_block - 从磁盘读取 KV block
# ===========================================================================
def load_block(
    source_path: str,
    view: memoryview,
    offset: int,
    block_size: int,
) -> None:
    """
    Load callback: read one KV block from disk. Remove the file on failure.
    """
    # 中文注释：从磁盘文件读取一个 KV block 到内存（提升加载路径）。
    #
    # 执行流程：
    #   1. 将 memoryview 转换为一维字节视图并切片，确定写入目标位置。
    #   2. 以只读 + O_DIRECT 模式打开源文件。
    #   3. 使用 os.readv（scatter-gather I/O）将文件内容直接读入
    #      一级层 memoryview 的对应偏移位置，避免额外的内存拷贝。
    #   4. 校验读取字节数，如果短读（short read）则抛出异常。
    #   5. 读取失败时删除源文件（可能是损坏的 block），防止后续重复读取
    #      损坏数据，并让上层重新计算该 block。
    #
    # 参数说明：
    #   source_path: 源文件路径，由 FileMapper.get_file_name() 生成。
    #   view:        一级层 CPU KV cache 的 memoryview，数据将被写入其中。
    #   offset:      写入起始字节偏移，等于 block_id * block_size。
    #   block_size:  一个 KV block 的字节数。
    #
    # 返回值：无（None）
    #
    # 异常处理：
    #   - 如果读取失败，会删除源文件并重新抛出异常。
    #   - 删除源文件是为了防止后续再次读取损坏的数据。
    #   - 调用者（DualQueueThreadPool）会捕获异常并通知上层。

    # 中文注释：文件描述符初始化为 None，用于 finally 块中确保关闭。
    fd: int | None = None

    # 中文注释：同 store_block，将多维 memoryview 转为一维字节切片，
    # 确保 offset 以字节为单位正确索引。
    # view_slice 指向 view 中从 offset 开始的 block_size 字节区域。
    view_slice = view.cast("B")[offset : offset + block_size]

    try:
        # 中文注释：以只读 + O_DIRECT 模式打开源文件。
        # 使用 O_DIRECT 绕过内核页缓存，直接从磁盘读取到用户态 buffer。
        # 这避免了数据在内核缓冲区和用户缓冲区之间的双重拷贝。
        fd = os.open(source_path, os.O_RDONLY | O_DIRECT)

        # 中文注释：使用 readv（scatter-gather read）将文件数据读入
        # memoryview 切片。readv 支持直接读入多个不连续的缓冲区，
        # 这里只传入一个缓冲区，效果等同于单次 read 但避免了额外拷贝。
        #
        # 为什么使用 readv 而非 read：
        #   - readv 可以直接读入 memoryview 对应的内存区域，无需额外拷贝。
        #   - read 返回 bytes 对象，需要额外的内存拷贝才能写入 memoryview。
        #   - 对于大块 KV cache 数据，减少一次拷贝可以显著提高性能。
        bytes_read = os.readv(fd, [view_slice])

        # 中文注释：校验读取的字节数是否等于预期的 block_size。
        # 短读（short read）可能发生在：
        #   - 文件被截断（如写入过程中崩溃）。
        #   - 文件系统错误。
        #   - O_DIRECT 对齐问题。
        if bytes_read < block_size:
            raise OSError(f"Short read: expected {block_size} bytes, read {bytes_read}")

    except Exception:
        # 中文注释：读取失败时删除源文件。
        # 这是一种容错策略：文件可能已损坏（如磁盘错误、写入中断等），
        # 删除后上层调用者会重新从 GPU 计算并存储该 block。
        #
        # 为什么删除而非保留：
        #   - 如果保留损坏的文件，后续每次读取都会失败，浪费 I/O 资源。
        #   - 删除后，上层可以重新计算该 block 并生成新的正确文件。
        #   - 这是一种"快速失败"（fail-fast）策略，避免系统陷入反复重试。
        try:
            os.remove(source_path)
        except OSError as cleanup_exc:
            logger.warning(
                "Failed to remove unreadable file %s: %s", source_path, cleanup_exc
            )
        # 中文注释：重新抛出原始异常，让调用者知道读取失败。
        raise

    finally:
        # 中文注释：确保文件描述符被关闭，无论读取成功还是失败。
        # 文件描述符泄漏会导致进程的文件描述符耗尽，影响系统稳定性。
        if fd is not None:
            os.close(fd)
