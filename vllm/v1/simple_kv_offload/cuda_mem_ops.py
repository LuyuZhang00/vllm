# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Low-level CUDA/HIP memory helpers: pinning and batch DMA transfers.

底层CUDA/HIP内存辅助工具：内存锁定和批量DMA传输。

本模块提供以下功能：
1. pin_tensor(): 将CPU张量固定到物理内存，避免页面交换
2. BatchMemcpyParams: 批量内存拷贝的参数结构
3. build_params(): 构建批量内存拷贝参数
4. copy_blocks(): 执行批量块拷贝

支持CUDA和ROCm两种平台：
- CUDA: 使用cuMemcpyBatchAsync API
- ROCm: 使用hipMemcpyBatchAsync API（需要ROCm 7.1+）
"""

import ctypes
from typing import Any, NamedTuple

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)


def pin_tensor(tensor: torch.Tensor) -> None:
    """Pin a CPU tensor via cudaHostRegister.

    通过cudaHostRegister固定CPU张量到物理内存。

    这绕过了PyTorch的CUDACachingHostAllocator，它会将每个pin_memory=True的分配
    向上舍入到2的幂（例如100 GB变成128 GB）。

    固定内存的好处：
    1. 避免页面交换，确保数据始终在物理内存中
    2. 启用DMA传输，CPU和GPU之间可以直接传输数据
    3. 提高异步内存拷贝的性能

    参数：
    - tensor: 要固定的CPU张量
    """
    err = torch.cuda.cudart().cudaHostRegister(tensor.data_ptr(), tensor.nbytes, 0)
    if err.value != 0:
        raise RuntimeError(f"cudaHostRegister failed: {err}")


class _CUmemLocation(ctypes.Structure):
    """CUDA内存位置结构体。

    用于指定内存操作的源和目标位置类型。
    """
    _fields_ = [("type", ctypes.c_uint), ("id", ctypes.c_int)]


class _CUmemcpyAttributes(ctypes.Structure):
    """CUDA内存拷贝属性结构体。

    用于配置批量内存拷贝的行为：
    - srcAccessOrder: 源访问顺序（流顺序或任意顺序）
    - srcLocHint: 源位置提示
    - dstLocHint: 目标位置提示
    - flags: 操作标志
    """
    _fields_ = [
        ("srcAccessOrder", ctypes.c_uint),
        ("srcLocHint", _CUmemLocation),
        ("dstLocHint", _CUmemLocation),
        ("flags", ctypes.c_uint),
    ]


# 批量内存拷贝函数类型定义
# 参数说明：
# 1. dsts: 目标地址数组
# 2. srcs: 源地址数组
# 3. sizes: 大小数组
# 4. count: 拷贝数量
# 5. attrs: 属性数组
# 6. attrIdxs: 属性索引数组
# 7. numAttrs: 属性数量
# 8. failIdx: 失败索引输出
# 9. stream: CUDA流
_BATCH_MEMCPY_FUNC_TYPE = ctypes.CFUNCTYPE(
    ctypes.c_uint,  # CUresult / hipError_t
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_void_p,
    ctypes.c_void_p,
)

# Resolved lazily on first use.
# 延迟解析，首次使用时初始化
_batch_memcpy_fn: Any = None


def _resolve_batch_memcpy():
    """Resolve the platform batch-memcpy entry point (one-time).

    解析平台批量内存拷贝入口点（一次性）。

    支持两种平台：
    1. CUDA: 通过cuGetProcAddress获取cuMemcpyBatchAsync（使用srcAccessOrder=STREAM）
    2. ROCm: 从libamdhip64获取hipMemcpyBatchAsync（需要ROCm 7.1+）

    注意：ROCm 7.2.1或7.2.2拒绝任何带有numAttrs > 0的调用，
    因此我们使用numAttrs=0调用。

    Raises ``RuntimeError`` if the symbol is unavailable (older CUDA
    driver, ROCm < 7.1, unusual install). The connector requires the
    batch API.

    如果符号不可用（较旧的CUDA驱动、ROCm < 7.1、异常安装），则抛出RuntimeError。
    连接器需要批量API。
    """
    if current_platform.is_rocm():
        try:
            lib = ctypes.CDLL("libamdhip64.so", mode=ctypes.RTLD_GLOBAL)
            fn = lib.hipMemcpyBatchAsync
        except (OSError, AttributeError) as e:
            raise RuntimeError(
                "hipMemcpyBatchAsync is unavailable in this ROCm install; "
                "SimpleCPUOffloadConnector requires ROCm 7.1+."
            ) from e
        fn.restype = ctypes.c_uint
        fn.argtypes = [
            ctypes.c_void_p,  # dsts
            ctypes.c_void_p,  # srcs
            ctypes.c_void_p,  # sizes
            ctypes.c_size_t,  # count
            ctypes.c_void_p,  # attrs
            ctypes.c_void_p,  # attrIdxs
            ctypes.c_size_t,  # numAttrs
            ctypes.c_void_p,  # failIdx
            ctypes.c_void_p,  # stream
        ]
        return fn

    from cuda.bindings import driver as drv

    err, ptr, _ = drv.cuGetProcAddress(b"cuMemcpyBatchAsync", 12080, 0)
    if err != drv.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuGetProcAddress(cuMemcpyBatchAsync) failed: {err}")
    return _BATCH_MEMCPY_FUNC_TYPE(ptr)


class BatchMemcpyParams(NamedTuple):
    """批量内存拷贝参数。

    包含执行批量内存拷贝所需的所有参数：
    - 源和目标的基础地址数组
    - 每层的字节数
    - 层数
    - CUDA属性（仅CUDA使用）
    - 流句柄

    这些参数在初始化时构建一次，然后重复用于多次拷贝操作。
    """
    src_bases: np.ndarray  # [num_layers] uint64 — 每层的数据指针
    dst_bases: np.ndarray  # [num_layers] uint64 — 每层的数据指针
    bpb: np.ndarray  # [num_layers] uint64 — 每块的字节数
    num_layers: int  # 层数
    # CUDA only: one attributes entry with srcAccessOrder=ANY. Unused on
    # ROCm (7.2.1 or 7.2.2) because the current runtime rejects numAttrs > 0.
    # 仅CUDA：一个属性条目，srcAccessOrder=ANY。
    # ROCm（7.2.1或7.2.2）不使用，因为当前运行时拒绝numAttrs > 0。
    attrs: _CUmemcpyAttributes
    attrs_idx: ctypes.c_size_t  # 属性索引
    # NOTE: cuMemcpyBatchAsync_v2() removed fail_idx field, but we use
    # cuMemcpyBatchAsync() with fail_idx for backward compatibility
    # 注意：cuMemcpyBatchAsync_v2()移除了fail_idx字段，但我们使用
    # 带有fail_idx的cuMemcpyBatchAsync()以保持向后兼容性
    fail_idx: ctypes.c_size_t  # 失败索引输出
    stream_handle: int  # 原始cudaStream_t / CUstream句柄


def build_params(
    src_caches: dict[str, torch.Tensor],
    dst_caches: dict[str, torch.Tensor],
    stream: torch.cuda.Stream,
) -> BatchMemcpyParams:
    """构建批量内存拷贝参数。

    从源和目标缓存张量字典构建批量内存拷贝参数。
    这些参数在初始化时构建一次，然后重复用于多次拷贝操作。

    构建流程：
    1. 解析平台批量内存拷贝函数（首次调用时）
    2. 验证源和目标缓存的键相同
    3. 提取每层的数据指针和每块字节数
    4. 设置CUDA属性（srcAccessOrder=ANY）
    5. 构建并返回BatchMemcpyParams

    参数：
    - src_caches: 源缓存张量字典
    - dst_caches: 目标缓存张量字典
    - stream: CUDA流

    返回：
    - BatchMemcpyParams: 批量内存拷贝参数
    """
    global _batch_memcpy_fn
    if _batch_memcpy_fn is None:
        _batch_memcpy_fn = _resolve_batch_memcpy()

    assert list(src_caches.keys()) == list(dst_caches.keys())
    src_tensors = list(src_caches.values())
    dst_tensors = list(dst_caches.values())

    src_bases, dst_bases, bpb = [], [], []
    for s, d in zip(src_tensors, dst_tensors):
        s_bpb = s.stride(0) * s.element_size()
        assert s_bpb == d.stride(0) * d.element_size()
        src_bases.append(s.data_ptr())
        dst_bases.append(d.data_ptr())
        bpb.append(s_bpb)

    # ``srcAccessOrder=3`` == CU_MEMCPY_SRC_ACCESS_ORDER_ANY /
    # hipMemcpySrcAccessOrderAny. See
    # https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1g6f1ff58e3065df3eb4b573dba77ad31f  # noqa: E501
    # srcAccessOrder=3 表示 CU_MEMCPY_SRC_ACCESS_ORDER_ANY /
    # hipMemcpySrcAccessOrderAny，允许硬件优化内存访问模式
    attrs = _CUmemcpyAttributes(srcAccessOrder=3)

    return BatchMemcpyParams(
        src_bases=np.array(src_bases, dtype=np.uint64),
        dst_bases=np.array(dst_bases, dtype=np.uint64),
        bpb=np.array(bpb, dtype=np.uint64),
        num_layers=len(src_tensors),
        attrs=attrs,
        attrs_idx=ctypes.c_size_t(0),
        fail_idx=ctypes.c_size_t(0),
        stream_handle=stream.cuda_stream,
    )


def copy_blocks(
    src_block_ids: list[int],
    dst_block_ids: list[int],
    params: BatchMemcpyParams,
) -> None:
    """Copy blocks via cuMemcpyBatchAsync / hipMemcpyBatchAsync.

    通过cuMemcpyBatchAsync / hipMemcpyBatchAsync批量拷贝块。

    执行流程：
    1. 将块ID转换为numpy数组
    2. 计算所有源和目标地址（每层每块的地址）
    3. 构建大小数组
    4. 调用平台批量内存拷贝函数
    5. 检查错误

    参数：
    - src_block_ids: 源块ID列表
    - dst_block_ids: 目标块ID列表
    - params: 批量内存拷贝参数

    计算地址的公式：
    - 对于层l和块b，源地址 = src_bases[l] + src_block_ids[b] * bpb[l]
    - 对于层l和块b，目标地址 = dst_bases[l] + dst_block_ids[b] * bpb[l]
    """
    n = len(src_block_ids)
    if n == 0:
        return

    src_ids = np.array(src_block_ids, dtype=np.uint64)
    dst_ids = np.array(dst_block_ids, dtype=np.uint64)

    # 计算所有源地址：每层的基础地址 + 块ID * 每块字节数
    src_all = (
        params.src_bases[:, None] + src_ids[None, :] * params.bpb[:, None]
    ).ravel()
    # 计算所有目标地址：每层的基础地址 + 块ID * 每块字节数
    dst_all = (
        params.dst_bases[:, None] + dst_ids[None, :] * params.bpb[:, None]
    ).ravel()
    # 每个拷贝的大小：重复每块字节数n次（每个块一次）
    sz_all = np.repeat(params.bpb, n)
    total = n * params.num_layers  # 总拷贝数量

    # ROCm 7.2.1/7.2.2 rejects any call with numAttrs>0 (hipMemcpyBatchAsync
    # hipamd/src/hip_memory.cpp:2819-2822); CUDA uses one attrs entry so
    # srcAccessOrder is honored. attrs / attrsIdxs are ignored when
    # numAttrs==0, so we pass the same values from both paths.
    # ROCm 7.2.1/7.2.2拒绝任何带有numAttrs>0的调用（hipMemcpyBatchAsync
    # hipamd/src/hip_memory.cpp:2819-2822）；CUDA使用一个attrs条目，
    # 因此srcAccessOrder会被遵循。当numAttrs==0时attrs/attrsIdxs被忽略，
    # 所以我们从两个路径传递相同的值。
    num_attrs = 0 if current_platform.is_rocm() else 1
    err = _batch_memcpy_fn(
        dst_all.ctypes.data,
        src_all.ctypes.data,
        sz_all.ctypes.data,
        total,
        ctypes.addressof(params.attrs),
        ctypes.byref(params.attrs_idx),
        num_attrs,
        ctypes.byref(params.fail_idx),
        params.stream_handle,
    )
    if err != 0:
        raise RuntimeError(
            f"batch memcpy failed: err={err} failIdx={params.fail_idx.value}"
        )
