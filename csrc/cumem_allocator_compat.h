#pragma once

/*
 * =============================================================================
 * 文件功能概述（中文）
 * =============================================================================
 * 本文件是 CUDA Virtual Memory Management (VMM) API 的跨平台兼容层。
 * 在 ROCm (AMD GPU) 平台上，将 NVIDIA 的 cuMem* 系列 API 映射到
 * 对应的 hipMem* API，使 cumem_allocator.cpp 中的代码无需 #ifdef 即可
 * 在两种平台上编译运行。
 *
 * 【提供内容】
 *   1. 类型别名：CUdeviceptr、CUresult、CUcontext 等 -> HIP 对应类型
 *   2. 宏定义：CU_MEM_ALLOCATION_TYPE_PINNED 等 -> HIP 常量
 *   3. 函数包装：cuMemCreate、cuMemMap、cuMemSetAccess 等 -> hipMem* 调用
 *   4. 错误处理：cuGetErrorString -> hipGetErrorString
 *
 * 【设计目的】
 *   让 cumem_allocator.cpp 中使用 CUDA VMM API 编写的代码
 *   在 ROCm 上也能正常编译和运行，无需维护两套代码。
 * =============================================================================
 */

#ifdef USE_ROCM
////////////////////////////////////////
// For compatibility with CUDA and ROCm
////////////////////////////////////////
  #include <hip/hip_runtime_api.h>

extern "C" {
  #ifndef CUDA_SUCCESS
    #define CUDA_SUCCESS hipSuccess
  #endif  // CUDA_SUCCESS

// 【类型映射】将 CUDA Driver API 类型映射到 HIP 对应类型
// https://rocm.docs.amd.com/projects/HIPIFY/en/latest/tables/CUDA_Driver_API_functions_supported_by_HIP.html
typedef unsigned long long CUdevice;
typedef hipDeviceptr_t CUdeviceptr;    // GPU 设备指针
typedef hipError_t CUresult;           // API 返回错误码
typedef hipCtx_t CUcontext;            // GPU 上下文句柄
typedef hipStream_t CUstream;          // CUDA/HIP 流
typedef hipMemGenericAllocationHandle_t CUmemGenericAllocationHandle;  // VMM 物理显存句柄
typedef hipMemAllocationGranularity_flags CUmemAllocationGranularity_flags;
typedef hipMemAllocationProp CUmemAllocationProp;    // 显存分配属性
typedef hipMemAccessDesc CUmemAccessDesc;            // 显存访问权限描述符

  #define CU_MEM_ALLOCATION_TYPE_PINNED hipMemAllocationTypePinned
  #define CU_MEM_LOCATION_TYPE_DEVICE hipMemLocationTypeDevice
  #define CU_MEM_ACCESS_FLAGS_PROT_READWRITE hipMemAccessFlagsProtReadWrite
  #define CU_MEM_ALLOC_GRANULARITY_MINIMUM hipMemAllocationGranularityMinimum

  // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html
  #define CU_MEM_ALLOCATION_COMP_NONE 0x0

// 【常量映射】将 CUDA VMM API 的常量映射到 HIP 对应值

// 【函数包装】以下函数将 CUDA Driver API 调用映射到 HIP 对应实现。
// 每个函数的语义和参数与 CUDA 版本完全一致，只是底层调用 HIP API。

// Error Handling
// https://docs.nvidia.com/cuda/archive/11.4.4/cuda-driver-api/group__CUDA__ERROR.html
CUresult cuGetErrorString(CUresult hipError, const char** pStr) {
  *pStr = hipGetErrorString(hipError);
  return CUDA_SUCCESS;
}

// Context Management
// https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html
CUresult cuCtxGetCurrent(CUcontext* ctx) {
  // This API is deprecated on the AMD platform, only for equivalent cuCtx
  // driver API on the NVIDIA platform.
  return hipCtxGetCurrent(ctx);
}

CUresult cuCtxSetCurrent(CUcontext ctx) {
  // This API is deprecated on the AMD platform, only for equivalent cuCtx
  // driver API on the NVIDIA platform.
  return hipCtxSetCurrent(ctx);
}

// Primary Context Management
// https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__PRIMARY__CTX.html
CUresult cuDevicePrimaryCtxRetain(CUcontext* ctx, CUdevice dev) {
  return hipDevicePrimaryCtxRetain(ctx, dev);
}

// Virtual Memory Management
// https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__VA.html
CUresult cuMemAddressFree(CUdeviceptr ptr, size_t size) {
  return hipMemAddressFree(ptr, size);
}

CUresult cuMemAddressReserve(CUdeviceptr* ptr, size_t size, size_t alignment,
                             CUdeviceptr addr, unsigned long long flags) {
  return hipMemAddressReserve(ptr, size, alignment, addr, flags);
}

CUresult cuMemCreate(CUmemGenericAllocationHandle* handle, size_t size,
                     const CUmemAllocationProp* prop,
                     unsigned long long flags) {
  return hipMemCreate(handle, size, prop, flags);
}

CUresult cuMemGetAllocationGranularity(
    size_t* granularity, const CUmemAllocationProp* prop,
    CUmemAllocationGranularity_flags option) {
  return hipMemGetAllocationGranularity(granularity, prop, option);
}

CUresult cuMemMap(CUdeviceptr dptr, size_t size, size_t offset,
                  CUmemGenericAllocationHandle handle,
                  unsigned long long flags) {
  return hipMemMap(dptr, size, offset, handle, flags);
}

CUresult cuMemRelease(CUmemGenericAllocationHandle handle) {
  return hipMemRelease(handle);
}

CUresult cuMemSetAccess(CUdeviceptr ptr, size_t size,
                        const CUmemAccessDesc* desc, size_t count) {
  return hipMemSetAccess(ptr, size, desc, count);
}

CUresult cuMemUnmap(CUdeviceptr ptr, size_t size) {
  return hipMemUnmap(ptr, size);
}
}  // extern "C"

#else
////////////////////////////////////////
// Import CUDA headers for NVIDIA GPUs
////////////////////////////////////////
  #include <cuda_runtime_api.h>
  #include <cuda.h>
#endif
