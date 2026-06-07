// A CUDAPluggableAllocator based on cumem* APIs.
// Important: allocation size, CUdeviceptr and CUmemGenericAllocationHandle*
// need to be unsigned long long

// =============================================================================
// 文件概述：基于 CUDA Virtual Memory Management (VMM) API 的可插拔内存分配器
// =============================================================================
//
// 本文件实现了一个基于 CUDA cumem* 系列 API 的显存分配器，用于 vLLM 的 KV cache 管理。
//
// 核心设计思路：
//   传统的 cudaMalloc/cudaFree 分配的显存无法精细控制，而 CUDA VMM API 提供了
//   三阶段的显存管理能力：
//     (1) cuMemAddressReserve  —— 预留虚拟地址空间（类似 mmap 的匿名映射）
//     (2) cuMemCreate          —— 创建物理显存块（CUmemGenericAllocationHandle）
//     (3) cuMemMap             —— 将物理显存块映射到虚拟地址空间
//     (4) cuMemSetAccess       —— 设置访问权限
//
// 这种分层管理方式的优势：
//   - 物理显存和虚拟地址解耦，支持延迟映射和灵活的内存布局
//   - 可以跨进程共享显存（通过 fabric handle 或 POSIX fd）
//   - 可以在不释放虚拟地址的情况下更换物理显存块
//   - 支持 GPUDirect RDMA，适用于多节点通信场景
//
// 工作流程：
//   Python 层通过 init_module() 注册 malloc/free 回调函数。
//   当 CUDAPluggableAllocator 需要分配显存时，调用 my_malloc()：
//     1. 预留虚拟地址空间
//     2. 通过 Python 回调将 (device, size, addr, handle) 信息传递给 Python 层
//     3. Python 层记录元数据后，回调通知 C++ 层完成 cuMemCreate + cuMemMap
//     4. 返回可用的设备指针
//   释放时，调用 my_free()：
//     1. 通过 Python 回调获取之前记录的元数据
//     2. 执行 cuMemUnmap + cuMemRelease 释放物理显存
//     3. 释放虚拟地址空间
//
// ROCm (AMD GPU) 特殊处理：
//   ROCm 的 VMM API 对单次 cuMemCreate 的大小有限制，因此采用分块（chunk）策略：
//   将大块显存拆分为多个小块分别创建和映射，分块大小可通过环境变量
//   VLLM_ROCM_SLEEP_MEM_CHUNK_SIZE 配置（默认 256MB）。
// =============================================================================

#include <iostream>

#include "cumem_allocator_compat.h"

// 中文注释：PyArg_ParseTuple 的格式字符串，用于从 Python 元组中解析参数。
// 非 ROCm 路径：四个 unsigned long long（device, size, d_mem, p_memHandle）
// ROCm 路径：三个 unsigned long long + 一个 PyObject*（handle 列表）
#ifndef USE_ROCM
static const char* PYARGS_PARSE = "KKKK";
#else
  #include <cstdlib>
  #include <cerrno>
  #include <climits>

// Default chunk size 256MB for ROCm. Can be overridden at runtime by the
// environment variable VLLM_ROCM_SLEEP_MEM_CHUNK_SIZE, specified in megabytes
// (MB). The env value is parsed with strtoull as an integer number of MB
// (decimal or 0x hex). The parsed MB value is converted to bytes. If
// parsing fails, the value is 0, or the multiplication would overflow,
// the default (256MB) is used.

// 中文注释：ROCm 默认分块大小为 256MB。由于 ROCm 的 VMM API 对单次 cuMemCreate
// 的大小有限制，需要将大块显存拆分为多个小块。分块大小可通过环境变量
// VLLM_ROCM_SLEEP_MEM_CHUNK_SIZE（单位 MB）在运行时调整。
static const unsigned long long DEFAULT_MEMCREATE_CHUNK_SIZE =
    (256ULL * 1024ULL * 1024ULL);

// 中文注释：从环境变量读取 ROCm 分块大小配置（单位 MB），失败则使用默认值 256MB。
static unsigned long long get_memcreate_chunk_size() {
  const char* env = getenv("VLLM_ROCM_SLEEP_MEM_CHUNK_SIZE");
  if (!env) return DEFAULT_MEMCREATE_CHUNK_SIZE;
  char* endptr = nullptr;
  errno = 0;
  unsigned long long val_mb = strtoull(env, &endptr, 0);
  if (endptr == env || errno != 0) {
    // parsing failed, fallback to default
    return DEFAULT_MEMCREATE_CHUNK_SIZE;
  }
  if (val_mb == 0) return DEFAULT_MEMCREATE_CHUNK_SIZE;

  const unsigned long long MB = 1024ULL * 1024ULL;
  // guard against overflow when converting MB -> bytes
  if (val_mb > (ULLONG_MAX / MB)) {
    return DEFAULT_MEMCREATE_CHUNK_SIZE;
  }
  return val_mb * MB;
}

static inline unsigned long long my_min(unsigned long long a,
                                        unsigned long long b) {
  return a < b ? a : b;
}

static const char* PYARGS_PARSE = "KKKO";
#endif

extern "C" {

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <sys/types.h>

// 中文注释：全局错误信息缓冲区和错误码，用于 CUDA API 调用失败时保存错误信息。
// 这些全局变量使得错误处理可以通过 error_code 检查而不需要每次都判断返回值。
char error_msg[10240];  // 10KB buffer to store error messages
CUresult no_error = CUresult(0);
CUresult error_code = no_error;  // store error code

// 中文注释：CUDA 错误检查宏。如果 CUDA API 调用失败，将错误码保存到全局变量
// error_code，并将可读的错误信息写入 error_msg 缓冲区并输出到 stderr。
// 使用 do-while(0) 模式确保宏在 if/else 等语句中安全使用。
#define CUDA_CHECK(condition)                                           \
  do {                                                                  \
    CUresult error = condition;                                         \
    if (error != 0) {                                                   \
      error_code = error;                                               \
      char* error_string;                                               \
      cuGetErrorString(error, (const char**)&error_string);             \
      snprintf(error_msg, sizeof(error_msg), "CUDA Error: %s at %s:%d", \
               error_string, __FILE__, __LINE__);                       \
      std::cerr << error_msg << std::endl;                              \
    }                                                                   \
  } while (0)

// Global references to Python callables
// NOTE: this is borrowed reference, so we don't need to DECREF them.
// This brings the limitation that the allocator needs to be singleton.

// 中文注释：全局 Python 回调函数指针，由 init_module() 设置。
// 这些是 borrowed reference（借用引用），不需要手动 DECREF，
// 但要求 Python 层必须保持这些回调对象的生命周期，且本模块只能是单例。
static PyObject* g_python_malloc_callback = nullptr;
static PyObject* g_python_free_callback = nullptr;

// ---------------------------------------------------------------------------
// Helper functions:

// 中文注释：确保当前线程已设置 CUDA 设备上下文。
// CUDA VMM API 要求调用时必须有有效的设备上下文。
// 如果当前线程没有上下文，通过 cuDevicePrimaryCtxRetain 获取主上下文并设置。
void ensure_context(unsigned long long device) {
  CUcontext pctx;
  CUDA_CHECK(cuCtxGetCurrent(&pctx));
  if (!pctx) {
    // Ensure device context.
    CUDA_CHECK(cuDevicePrimaryCtxRetain(&pctx, device));
    CUDA_CHECK(cuCtxSetCurrent(pctx));
  }
}

// 中文注释：创建物理显存块并映射到已预留的虚拟地址空间。
// 这是 CUDA VMM 三阶段分配流程的第二、三步（cuMemCreate + cuMemMap）。
//
// 参数说明：
//   device      —— GPU 设备 ID
//   size        —— 要映射的总大小（字节），已按粒度对齐
//   d_mem       —— 由 cuMemAddressReserve 预留的虚拟地址
//   p_memHandle —— cuMemCreate 返回的物理显存句柄
//     非 ROCm：单个句柄指针
//     ROCm：句柄指针数组（分块分配）
//
// 流程：
//   1. 设置显存分配属性（pinned memory, device location, 无压缩）
//   2. （仅 NVIDIA）检查并启用 GPUDirect RDMA 和 fabric handle 支持
//   3. 调用 cuMemCreate 创建物理显存块
//   4. 调用 cuMemMap 将物理显存块映射到虚拟地址 d_mem
//   5. 调用 cuMemSetAccess 设置读写权限
void create_and_map(unsigned long long device, ssize_t size, CUdeviceptr d_mem,
#ifndef USE_ROCM
                    CUmemGenericAllocationHandle* p_memHandle) {
#else
                    CUmemGenericAllocationHandle** p_memHandle,
                    unsigned long long* chunk_sizes, size_t num_chunks) {
#endif
  ensure_context(device);
  // Define memory allocation properties
  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = device;
  prop.allocFlags.compressionType = CU_MEM_ALLOCATION_COMP_NONE;

#ifndef USE_ROCM
  // 中文注释：检查设备是否支持 GPUDirect RDMA（GPU 直接远程内存访问）。
  // 启用此标志后，分配的显存可以被网络设备直接读写，适用于多节点推理场景。
  int flag = 0;
  CUresult rdma_result = cuDeviceGetAttribute(
      &flag, CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED,
      device);
  if (rdma_result == CUDA_SUCCESS &&
      flag) {  // support GPUDirect RDMA if possible
    prop.allocFlags.gpuDirectRDMACapable = 1;
  }
  // 中文注释：检查设备是否支持 fabric handle（NVLink fabric 句柄）。
  // fabric handle 允许在多 GPU 系统中跨 GPU 共享显存，适用于张量并行等场景。
  // 如果不支持，后续会回退到 POSIX 文件描述符方式。
  int fab_flag = 0;
  CUresult fab_result = cuDeviceGetAttribute(
      &fab_flag, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, device);
  if (fab_result == CUDA_SUCCESS &&
      fab_flag) {  // support fabric handle if possible
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;
  }
#endif

#ifndef USE_ROCM
  // 中文注释：[NVIDIA 路径] 调用 cuMemCreate 创建物理显存块。
  // 如果 fabric handle 分配失败（例如没有多节点 NVLink），自动回退到 POSIX 文件描述符。
  // Allocate memory using cuMemCreate
  CUresult ret = (CUresult)cuMemCreate(p_memHandle, size, &prop, 0);
  if (ret) {
    if (fab_flag &&
        (ret == CUDA_ERROR_NOT_PERMITTED || ret == CUDA_ERROR_NOT_SUPPORTED)) {
      // Fabric allocation may fail without multi-node nvlink,
      // fallback to POSIX file descriptor
      prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
      CUDA_CHECK(cuMemCreate(p_memHandle, size, &prop, 0));
    } else {
      CUDA_CHECK(ret);
    }
  }
  if (error_code != 0) {
    return;
  }
  // 中文注释：将物理显存块映射到之前预留的虚拟地址 d_mem。
  // 此时 d_mem 地址开始指向实际的 GPU 显存。
  CUDA_CHECK(cuMemMap(d_mem, size, 0, *p_memHandle, 0));
  if (error_code != 0) {
    return;
  }
#else
  // 中文注释：[ROCm 路径] 由于 ROCm 的 VMM API 对单次 cuMemCreate 大小有限制，
  // 需要分块创建物理显存块。如果中途失败，需要清理已创建的句柄。
  for (auto i = 0; i < num_chunks; ++i) {
    CUDA_CHECK(cuMemCreate(p_memHandle[i], chunk_sizes[i], &prop, 0));
    if (error_code != 0) {
      // Clean up previously created handles
      for (auto j = 0; j < i; ++j) {
        cuMemRelease(*(p_memHandle[j]));
      }
      return;
    }
  }
  // 中文注释：[ROCm 路径] 将各分块依次映射到连续的虚拟地址空间。
  // 每个分块映射到 d_mem + offset 的位置，形成连续的虚拟地址布局。
  unsigned long long allocated_size = 0;
  for (auto i = 0; i < num_chunks; ++i) {
    void* map_addr = (void*)((uintptr_t)d_mem + allocated_size);
    CUDA_CHECK(cuMemMap(map_addr, chunk_sizes[i], 0, *(p_memHandle[i]), 0));
    if (error_code != 0) {
      // unmap previously mapped chunks
      unsigned long long unmapped_size = 0;
      for (auto j = 0; j < i; ++j) {
        void* unmap_addr = (void*)((uintptr_t)d_mem + unmapped_size);
        cuMemUnmap(unmap_addr, chunk_sizes[j]);
        unmapped_size += chunk_sizes[j];
      }
      // release all created handles
      for (auto j = 0; j < num_chunks; ++j) {
        cuMemRelease(*(p_memHandle[j]));
      }
      return;
    }
    allocated_size += chunk_sizes[i];
  }
#endif

  CUmemAccessDesc accessDesc = {};
  accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  accessDesc.location.id = device;
  accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

  CUDA_CHECK(cuMemSetAccess(d_mem, size, &accessDesc, 1));
  if (error_code != 0) {
    return;
  }
  // std::cout << "create_and_map: device=" << device << ", size=" << size << ",
  // d_mem=" << d_mem << ", p_memHandle=" << p_memHandle << std::endl;
}

void unmap_and_release(unsigned long long device, ssize_t size,
                       CUdeviceptr d_mem,
#ifndef USE_ROCM
                       CUmemGenericAllocationHandle* p_memHandle) {
#else
                       CUmemGenericAllocationHandle** p_memHandle,
                       unsigned long long* chunk_sizes, size_t num_chunks) {
#endif
  // std::cout << "unmap_and_release: device=" << device << ", size=" << size <<
  // ", d_mem=" << d_mem << ", p_memHandle=" << p_memHandle << std::endl;
  ensure_context(device);
#ifndef USE_ROCM
  CUDA_CHECK(cuMemUnmap(d_mem, size));
  if (error_code != 0) {
    return;
  }
  CUDA_CHECK(cuMemRelease(*p_memHandle));
  if (error_code != 0) {
    return;
  }
#else
  unsigned long long allocated_size = 0;
  CUresult first_error = no_error;

  for (auto i = 0; i < num_chunks; ++i) {
    void* map_addr = (void*)((uintptr_t)d_mem + allocated_size);
    CUresult status = cuMemUnmap(map_addr, chunk_sizes[i]);
    if (status != no_error && first_error == no_error) {
      first_error = status;
    }
    allocated_size += chunk_sizes[i];
  }

  for (auto i = 0; i < num_chunks; ++i) {
    CUresult status = cuMemRelease(*(p_memHandle[i]));
    if (status != no_error && first_error == no_error) {
      first_error = status;
    }
  }

  if (first_error != no_error) {
    CUDA_CHECK(first_error);
  }
#endif
}

PyObject* create_tuple_from_c_integers(unsigned long long a,
                                       unsigned long long b,
                                       unsigned long long c,
                                       unsigned long long d) {
  // Create a new tuple of size 4
  PyObject* tuple = PyTuple_New(4);
  if (!tuple) {
    return NULL;  // Return NULL on failure
  }

  // Convert integers to Python objects and set them in the tuple
  PyTuple_SetItem(
      tuple, 0,
      PyLong_FromUnsignedLongLong(a));  // Steals reference to the PyLong
  PyTuple_SetItem(tuple, 1, PyLong_FromUnsignedLongLong(b));
  PyTuple_SetItem(tuple, 2, PyLong_FromUnsignedLongLong(c));
  PyTuple_SetItem(tuple, 3, PyLong_FromUnsignedLongLong(d));

  // Note: PyTuple_SetItem "steals" a reference to each object,
  // so we do not need to Py_DECREF the PyLong objects explicitly.

  return tuple;  // Return the created tuple
}

PyObject* create_tuple_from_c_mixed(unsigned long long a, unsigned long long b,
                                    unsigned long long c,
                                    CUmemGenericAllocationHandle** vec,
                                    unsigned long long* chunk_sizes,
                                    size_t num_chunks) {
  PyObject* tuple = PyTuple_New(4);
  if (!tuple) {
    return NULL;
  }

  // PyObject* list = PyList_New(vec.size());
  PyObject* list = PyList_New(num_chunks);
  for (auto i = 0; i < num_chunks; ++i) {
    PyObject* addr_size_pair = PyTuple_New(2);
    PyObject* addr = PyLong_FromUnsignedLongLong((unsigned long long)(vec[i]));
    PyObject* size =
        PyLong_FromUnsignedLongLong((unsigned long long)(chunk_sizes[i]));
    PyTuple_SetItem(addr_size_pair, 0, addr);
    PyTuple_SetItem(addr_size_pair, 1, size);
    PyList_SetItem(list, i, addr_size_pair);
  }

  PyTuple_SetItem(tuple, 0, PyLong_FromUnsignedLongLong(a));
  PyTuple_SetItem(tuple, 1, PyLong_FromUnsignedLongLong(b));
  PyTuple_SetItem(tuple, 2, PyLong_FromUnsignedLongLong(c));
  PyTuple_SetItem(tuple, 3, list);

  return tuple;
}

// ---------------------------------------------------------------------------
// Our exported C functions that call Python:

// use CUstream instead of cudaStream_t, to avoid including cuda_runtime_api.h
void* my_malloc(ssize_t size, int device, CUstream stream) {
  ensure_context(device);

  // first allocation, align the size, and reserve an address, and also allocate
  // a CUmemGenericAllocationHandle

  // Define memory allocation properties
  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = device;
  prop.allocFlags.compressionType = CU_MEM_ALLOCATION_COMP_NONE;

  // Check if the allocation is supported
  size_t granularity;
  CUDA_CHECK(cuMemGetAllocationGranularity(&granularity, &prop,
                                           CU_MEM_ALLOC_GRANULARITY_MINIMUM));
  if (error_code != 0) {
    return nullptr;
  }
  size_t alignedSize = ((size + granularity - 1) / granularity) * granularity;

  CUdeviceptr d_mem;
#ifndef USE_ROCM
  CUDA_CHECK(cuMemAddressReserve(&d_mem, alignedSize, 0, 0, 0));
  if (error_code != 0) {
    return nullptr;
  }
#else
  CUDA_CHECK(cuMemAddressReserve(&d_mem, alignedSize, granularity, 0, 0));
  if (error_code != 0) {
    return nullptr;
  }
#endif

#ifndef USE_ROCM
  // allocate the CUmemGenericAllocationHandle
  CUmemGenericAllocationHandle* p_memHandle =
      (CUmemGenericAllocationHandle*)malloc(
          sizeof(CUmemGenericAllocationHandle));
#else
  // Make sure chunk size is aligned with hardware granularity. The base
  // chunk size can be configured via environment variable
  // ``VLLM_ROCM_SLEEP_MEM_CHUNK_SIZE``; otherwise
  // DEFAULT_MEMCREATE_CHUNK_SIZE is used.
  size_t base_chunk = (size_t)get_memcreate_chunk_size();
  size_t aligned_chunk_size =
      ((base_chunk + granularity - 1) / granularity) * granularity;
  size_t num_chunks =
      (alignedSize + aligned_chunk_size - 1) / aligned_chunk_size;
  CUmemGenericAllocationHandle** p_memHandle =
      (CUmemGenericAllocationHandle**)malloc(
          num_chunks * sizeof(CUmemGenericAllocationHandle*));
  unsigned long long* chunk_sizes =
      (unsigned long long*)malloc(num_chunks * sizeof(unsigned long long));
  for (auto i = 0; i < num_chunks; ++i) {
    p_memHandle[i] = (CUmemGenericAllocationHandle*)malloc(
        sizeof(CUmemGenericAllocationHandle));
    if (p_memHandle[i] == nullptr) {
      std::cerr << "ERROR: malloc failed for p_memHandle[" << i << "].\n";
      for (auto j = 0; j < i; ++j) {
        free(p_memHandle[j]);
      }
      free(p_memHandle);
      free(chunk_sizes);
      return nullptr;
    }
    chunk_sizes[i] = (unsigned long long)my_min(
        (unsigned long long)(alignedSize - i * aligned_chunk_size),
        (unsigned long long)aligned_chunk_size);
  }
#endif

  if (!g_python_malloc_callback) {
    std::cerr << "ERROR: g_python_malloc_callback not set.\n";
    return nullptr;
  }

  // Acquire GIL (not in stable ABI officially, but often works)
  PyGILState_STATE gstate = PyGILState_Ensure();

#ifndef USE_ROCM
  PyObject* arg_tuple = create_tuple_from_c_integers(
      (unsigned long long)device, (unsigned long long)alignedSize,
      (unsigned long long)d_mem, (unsigned long long)p_memHandle);
#else
  PyObject* arg_tuple = create_tuple_from_c_mixed(
      (unsigned long long)device, (unsigned long long)alignedSize,
      (unsigned long long)d_mem, p_memHandle, chunk_sizes, num_chunks);
#endif

  // Call g_python_malloc_callback
  PyObject* py_result =
      PyObject_CallFunctionObjArgs(g_python_malloc_callback, arg_tuple, NULL);
  Py_DECREF(arg_tuple);

  if (!py_result) {
    PyErr_Print();
    PyGILState_Release(gstate);
    return nullptr;
  }

  PyGILState_Release(gstate);

  // do the final mapping
#ifndef USE_ROCM
  create_and_map(device, alignedSize, d_mem, p_memHandle);
#else
  create_and_map(device, alignedSize, d_mem, p_memHandle, chunk_sizes,
                 num_chunks);
  free(chunk_sizes);
#endif

  if (error_code != 0) {
    // free address and the handle
    CUDA_CHECK(cuMemAddressFree(d_mem, alignedSize));
#ifndef USE_ROCM
    free(p_memHandle);
#else
    for (size_t i = 0; i < num_chunks; ++i) {
      free(p_memHandle[i]);
    }
    free(p_memHandle);
#endif
    return nullptr;
  }

  return (void*)d_mem;
}

// use CUstream instead of cudaStream_t, to avoid including cuda_runtime_api.h
void my_free(void* ptr, ssize_t size, int device, CUstream stream) {
  // get memory handle from the pointer
  if (!g_python_free_callback) {
    std::cerr << "ERROR: g_python_free_callback not set.\n";
    return;
  }

  // Acquire GIL (not in stable ABI officially, but often works)
  PyGILState_STATE gstate = PyGILState_Ensure();

  PyObject* py_ptr =
      PyLong_FromUnsignedLongLong(reinterpret_cast<unsigned long long>(ptr));

  PyObject* py_result =
      PyObject_CallFunctionObjArgs(g_python_free_callback, py_ptr, NULL);

  if (!py_result || !PyTuple_Check(py_result) || PyTuple_Size(py_result) != 4) {
    PyErr_SetString(PyExc_TypeError, "Expected a tuple of size 4");
    Py_XDECREF(py_result);
    Py_XDECREF(py_ptr);
    return;
  }

  unsigned long long recv_device, recv_size;
  unsigned long long recv_d_mem;
#ifndef USE_ROCM
  unsigned long long recv_p_memHandle;
#else
  PyObject* recv_p_memHandle;
#endif
  // Unpack the tuple into four C integers
  if (!PyArg_ParseTuple(py_result, PYARGS_PARSE, &recv_device, &recv_size,
                        &recv_d_mem, &recv_p_memHandle)) {
    // PyArg_ParseTuple sets an error if it fails
    Py_XDECREF(py_result);
    Py_XDECREF(py_ptr);
    return;
  }

  // For ROCm, copy the Python list of (addr,size) pairs into C arrays while
  // holding the GIL. Then release the GIL and call the unmap/release helper
  // using the copied arrays. This avoids calling PyList_* APIs without the
  // GIL (which is undefined behavior and can crash when called from other
  // threads).
  CUdeviceptr d_mem = (CUdeviceptr)recv_d_mem;
#ifdef USE_ROCM
  Py_ssize_t num_chunks = PyList_Size(recv_p_memHandle);
  CUmemGenericAllocationHandle** p_memHandle =
      (CUmemGenericAllocationHandle**)malloc(
          num_chunks * sizeof(CUmemGenericAllocationHandle*));
  if (p_memHandle == nullptr) {
    Py_DECREF(py_ptr);
    Py_DECREF(py_result);
    PyGILState_Release(gstate);
    std::cerr << "ERROR: malloc failed for p_memHandle in my_free."
              << std::endl;
    return;
  }
  unsigned long long* chunk_sizes =
      (unsigned long long*)malloc(num_chunks * sizeof(unsigned long long));
  if (chunk_sizes == nullptr) {
    free(p_memHandle);
    Py_DECREF(py_ptr);
    Py_DECREF(py_result);
    PyGILState_Release(gstate);
    std::cerr << "ERROR: malloc failed for chunk_sizes in my_free."
              << std::endl;
    return;
  }
  for (Py_ssize_t i = 0; i < num_chunks; ++i) {
    PyObject* item = PyList_GetItem(recv_p_memHandle, i);
    PyObject* addr_py = PyTuple_GetItem(item, 0);
    PyObject* size_py = PyTuple_GetItem(item, 1);
    p_memHandle[i] =
        (CUmemGenericAllocationHandle*)PyLong_AsUnsignedLongLong(addr_py);
    chunk_sizes[i] = (unsigned long long)PyLong_AsUnsignedLongLong(size_py);
  }

  // Drop temporary Python refs, then release the GIL before calling into
  // non-Python APIs.
  Py_DECREF(py_ptr);
  Py_DECREF(py_result);
  PyGILState_Release(gstate);

  unmap_and_release(device, size, d_mem, p_memHandle, chunk_sizes, num_chunks);
#else
  // Non-ROCm path: simple integer handle already extracted; drop temporary
  // Python refs while still holding the GIL, then release it.
  Py_DECREF(py_ptr);
  Py_DECREF(py_result);
  PyGILState_Release(gstate);

  CUmemGenericAllocationHandle* p_memHandle =
      (CUmemGenericAllocationHandle*)recv_p_memHandle;
  unmap_and_release(device, size, d_mem, p_memHandle);
#endif

  // free address and the handle
  CUDA_CHECK(cuMemAddressFree(d_mem, size));
#ifndef USE_ROCM
  free(p_memHandle);
#else
  for (auto i = 0; i < num_chunks; ++i) {
    free(p_memHandle[i]);
  }
  free(p_memHandle);
  free(chunk_sizes);
#endif
}

// ---------------------------------------------------------------------------
// Python extension boilerplate:

// Python-exposed function: init_module(python_malloc, python_free)
static PyObject* py_init_module(PyObject* self, PyObject* args) {
  PyObject* malloc_callback = nullptr;
  PyObject* free_callback = nullptr;

  if (!PyArg_ParseTuple(args, "OO", &malloc_callback, &free_callback)) {
    return nullptr;
  }

  if (!PyCallable_Check(malloc_callback) || !PyCallable_Check(free_callback)) {
    PyErr_SetString(PyExc_TypeError, "Both arguments must be callables");
    return nullptr;
  }

  // Save the Python callables
  // This module does not handle GC of these objects, so they must be kept alive
  // outside of this module.
  g_python_malloc_callback = malloc_callback;
  g_python_free_callback = free_callback;

  Py_RETURN_NONE;
}

static PyObject* python_unmap_and_release(PyObject* self, PyObject* args) {
  if (!args || !PyTuple_Check(args) || PyTuple_Size(args) != 4) {
    PyErr_SetString(PyExc_TypeError, "Expected a tuple of size 4");
    return nullptr;
  }

  unsigned long long recv_device, recv_size;
  unsigned long long recv_d_mem;
#ifndef USE_ROCM
  unsigned long long recv_p_memHandle;
#else
  PyObject* recv_p_memHandle;
#endif
  // Unpack the tuple into four C integers
  if (!PyArg_ParseTuple(args, PYARGS_PARSE, &recv_device, &recv_size,
                        &recv_d_mem, &recv_p_memHandle)) {
    // PyArg_ParseTuple sets an error if it fails
    return nullptr;
  }

  CUdeviceptr d_mem_ptr = (CUdeviceptr)recv_d_mem;
#ifndef USE_ROCM
  CUmemGenericAllocationHandle* p_memHandle =
      (CUmemGenericAllocationHandle*)recv_p_memHandle;

  unmap_and_release(recv_device, recv_size, d_mem_ptr, p_memHandle);
#else
  if (!PyList_Check(recv_p_memHandle)) {
    PyErr_SetString(PyExc_TypeError,
                    "Expected a list for the 4th argument on ROCm");
    return nullptr;
  }
  Py_ssize_t num_chunks = PyList_Size(recv_p_memHandle);
  if (num_chunks < 0) {
    return nullptr;  // PyList_Size sets an exception on error.
  }
  CUmemGenericAllocationHandle** p_memHandle =
      (CUmemGenericAllocationHandle**)malloc(
          num_chunks * sizeof(CUmemGenericAllocationHandle*));
  if (p_memHandle == nullptr) {
    PyErr_SetString(PyExc_MemoryError, "malloc failed for p_memHandle");
    return nullptr;
  }
  unsigned long long* chunk_sizes =
      (unsigned long long*)malloc(num_chunks * sizeof(unsigned long long));
  if (chunk_sizes == nullptr) {
    free(p_memHandle);
    PyErr_SetString(PyExc_MemoryError, "malloc failed for chunk_sizes");
    return nullptr;
  }
  for (Py_ssize_t i = 0; i < num_chunks; ++i) {
    PyObject* item = PyList_GetItem(recv_p_memHandle, i);
    if (item == nullptr || !PyTuple_Check(item) || PyTuple_Size(item) != 2) {
      free(p_memHandle);
      free(chunk_sizes);
      PyErr_SetString(
          PyExc_TypeError,
          "List items must be tuples of size 2 (handle_addr, size)");
      return nullptr;
    }
    PyObject* addr_py = PyTuple_GetItem(item, 0);
    PyObject* size_py = PyTuple_GetItem(item, 1);
    if (addr_py == nullptr || size_py == nullptr) {
      free(p_memHandle);
      free(chunk_sizes);
      return nullptr;  // PyTuple_GetItem sets an exception
    }
    p_memHandle[i] =
        (CUmemGenericAllocationHandle*)PyLong_AsUnsignedLongLong(addr_py);
    if (PyErr_Occurred()) {
      free(p_memHandle);
      free(chunk_sizes);
      return nullptr;
    }
    chunk_sizes[i] = (unsigned long long)PyLong_AsUnsignedLongLong(size_py);
    if (PyErr_Occurred()) {
      free(p_memHandle);
      free(chunk_sizes);
      return nullptr;
    }
  }

  unmap_and_release(recv_device, recv_size, d_mem_ptr, p_memHandle, chunk_sizes,
                    num_chunks);

  free(p_memHandle);
  free(chunk_sizes);
#endif

  if (error_code != 0) {
    error_code = no_error;
    PyErr_SetString(PyExc_RuntimeError, error_msg);
    return nullptr;
  }

  Py_RETURN_NONE;
}

static PyObject* python_create_and_map(PyObject* self, PyObject* args) {
  if (!args || !PyTuple_Check(args) || PyTuple_Size(args) != 4) {
    PyErr_SetString(PyExc_TypeError, "Expected a tuple of size 4");
    return nullptr;
  }

  unsigned long long recv_device, recv_size;
  unsigned long long recv_d_mem;
#ifndef USE_ROCM
  unsigned long long recv_p_memHandle;
#else
  PyObject* recv_p_memHandle;
#endif
  // Unpack the tuple into four C integers
  if (!PyArg_ParseTuple(args, PYARGS_PARSE, &recv_device, &recv_size,
                        &recv_d_mem, &recv_p_memHandle)) {
    // PyArg_ParseTuple sets an error if it fails
    return nullptr;
  }

  CUdeviceptr d_mem_ptr = (CUdeviceptr)recv_d_mem;
#ifndef USE_ROCM
  CUmemGenericAllocationHandle* p_memHandle =
      (CUmemGenericAllocationHandle*)recv_p_memHandle;

  create_and_map(recv_device, recv_size, d_mem_ptr, p_memHandle);
#else
  Py_ssize_t num_chunks = PyList_Size(recv_p_memHandle);
  CUmemGenericAllocationHandle** p_memHandle =
      (CUmemGenericAllocationHandle**)malloc(
          num_chunks * sizeof(CUmemGenericAllocationHandle*));
  if (p_memHandle == nullptr) {
    PyErr_SetString(PyExc_MemoryError, "malloc failed for p_memHandle");
    return nullptr;
  }
  unsigned long long* chunk_sizes =
      (unsigned long long*)malloc(num_chunks * sizeof(unsigned long long));
  if (chunk_sizes == nullptr) {
    free(p_memHandle);
    PyErr_SetString(PyExc_MemoryError, "malloc failed for chunk_sizes");
    return nullptr;
  }
  for (auto i = 0; i < num_chunks; ++i) {
    PyObject* item = PyList_GetItem(recv_p_memHandle, i);
    PyObject* addr_py = PyTuple_GetItem(item, 0);
    PyObject* size_py = PyTuple_GetItem(item, 1);
    p_memHandle[i] =
        (CUmemGenericAllocationHandle*)PyLong_AsUnsignedLongLong(addr_py);
    chunk_sizes[i] = PyLong_AsUnsignedLongLong(size_py);
  }

  create_and_map(recv_device, recv_size, d_mem_ptr, p_memHandle, chunk_sizes,
                 num_chunks);

  free(p_memHandle);
  free(chunk_sizes);
#endif

  if (error_code != 0) {
    error_code = no_error;
    PyErr_SetString(PyExc_RuntimeError, error_msg);
    return nullptr;
  }

  Py_RETURN_NONE;
}

static PyMethodDef module_methods[] = {
    {"init_module", (PyCFunction)py_init_module, METH_VARARGS,
     "Initialize module with python_malloc and python_free callables."},
    {"python_create_and_map", (PyCFunction)python_create_and_map, METH_VARARGS,
     "Create and map memory on the device."},
    {"python_unmap_and_release", (PyCFunction)python_unmap_and_release,
     METH_VARARGS, "Unmap and release memory on the device."},
    {NULL, NULL, 0, NULL}  // sentinel
};

static struct PyModuleDef cumem_allocator_module = {
    PyModuleDef_HEAD_INIT, "cumem_allocator",
    "cumem-based allocator for CUDAPluggableAllocator", -1, module_methods};

PyMODINIT_FUNC PyInit_cumem_allocator(void) {
  // Initialize the module
  PyObject* module = PyModule_Create(&cumem_allocator_module);
  if (!module) {
    return NULL;
  }
  return module;
}
}  // extern "C"
