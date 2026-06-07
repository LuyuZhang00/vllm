#pragma once

// =============================================================================
// 中文注释：CUDA 工具函数头文件
//
// 【模块功能概述】
//   本文件是 vLLM C++/CUDA 扩展层的基础工具头文件，提供以下核心功能：
//     1. 跨平台编译宏定义 —— 统一 CUDA(NVCC)、HIP(AMD ROCm)、普通 C++ 编译器
//        之间的函数修饰符差异，使同一份代码可以在不同 GPU 后端上编译。
//     2. CUDA 运行时错误检查宏 —— 封装 cudaError_t 检查逻辑，简化内核调用
//        后的错误处理，避免遗漏错误导致静默失败。
//     3. 设备属性查询接口 —— 提供查询 GPU 设备能力（如共享内存大小等）的函数。
//     4. 通用数学工具函数 —— 如整数向上取整除法（ceil_div），在 CUDA kernel
//        和 Host 端均可使用。
//
// 【文件在 vLLM 中的位置】
//   vLLM 的 C++ 扩展（如 attention kernel、量化 kernel、token embedding 等）
//   都会间接包含此头文件，使用其中的宏和工具函数。
// =============================================================================

#include <stdio.h>

// -----------------------------------------------------------------------------
// 中文注释：跨平台函数修饰符宏定义
//
// 【设计背景】
//   NVIDIA CUDA 使用 NVCC 编译器，AMD ROCm 使用 HIPCC 编译器，二者对函数执行
//   位置的修饰符语法略有不同。同时，纯 CPU 端代码不使用这些修饰符。为了在一份
//   代码中同时支持三种场景，这里通过预处理宏进行统一抽象：
//
//   - HOST_DEVICE_INLINE：同时在 CPU（Host）和 GPU（Device）上可调用的内联函数
//   - DEVICE_INLINE：仅在 GPU（Device）上可调用的内联函数
//   - HOST_INLINE：仅在 CPU（Host）上可调用的内联函数
//
// 【三种编译环境的差异】
//   (1) HIPCC（AMD GPU）：__host__ / __device__ 已足够，HIP 编译器默认内联
//   (2) NVCC / NVHPC（NVIDIA GPU）：需要显式加 __forceinline__ 强制内联
//   (3) 普通 C++ 编译器（无 GPU）：退化为标准 inline
// -----------------------------------------------------------------------------

#if defined(__HIPCC__)
  #define HOST_DEVICE_INLINE __host__ __device__
  #define DEVICE_INLINE __device__
  #define HOST_INLINE __host__
#elif defined(__CUDACC__) || defined(_NVHPC_CUDA)
  #define HOST_DEVICE_INLINE __host__ __device__ __forceinline__
  #define DEVICE_INLINE __device__ __forceinline__
  #define HOST_INLINE __host__ __forceinline__
#else
  #define HOST_DEVICE_INLINE inline
  #define DEVICE_INLINE inline
  #define HOST_INLINE inline
#endif

// -----------------------------------------------------------------------------
// 中文注释：CUDA 运行时错误检查宏
//
// 【功能说明】
//   对任意 CUDA API 调用（cudaMalloc、cudaMemcpy、cudaMemset 等）的返回值
//   进行统一检查。如果调用失败（返回值 != cudaSuccess），则打印详细的错误信息
//   （包含文件名、行号、错误描述），然后立即终止进程。
//
// 【使用示例】
//   CUDA_CHECK(cudaMalloc(&ptr, size));
//   CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
//
// 【设计要点】
//   - 使用 do { ... } while(0) 惯用法，确保宏在 if/else 等语句中安全展开。
//   - 采用 printf 而非 std::cerr，因为此宏可能在 CUDA kernel 的 host 端包装
//     函数中使用，printf 在多线程/多进程环境下更为可靠。
//   - 出错即 exit(EXIT_FAILURE)，因为 CUDA 错误通常是不可恢复的致命错误。
// -----------------------------------------------------------------------------
#define CUDA_CHECK(cmd)                                             \
  do {                                                              \
    cudaError_t e = cmd;                                            \
    if (e != cudaSuccess) {                                         \
      printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__, \
             cudaGetErrorString(e));                                \
      exit(EXIT_FAILURE);                                           \
    }                                                               \
  } while (0)

// 中文注释：查询指定 GPU 设备的某个属性值。
// 参数 attribute：CUDA 设备属性枚举值（如 cudaDevAttrMaxThreadsPerBlock 等）。
// 参数 device_id：GPU 设备编号（从 0 开始）。
// 返回值：该属性的整数值。此函数的实现在对应的 .cu 文件中，通过
//   cudaDeviceGetAttribute() 获取结果。
int64_t get_device_attribute(int64_t attribute, int64_t device_id);

// 中文注释：查询指定 GPU 设备上每个 block 可使用的最大共享内存（Shared Memory）字节数。
// 这个值在 vLLM 中用于判断 attention kernel 等是否可以在该设备上运行。
// 参数 device_id：GPU 设备编号（从 0 开始）。
// 返回值：最大共享内存大小（字节）。内部调用 get_device_attribute，
//   传入 cudaDevAttrMaxSharedMemoryPerBlock 属性。
int64_t get_max_shared_memory_per_block_device_attribute(int64_t device_id);

// -----------------------------------------------------------------------------
// 中文注释：cuda_utils 命名空间
//
// 包含在 CUDA kernel 和 Host 端均可使用的通用数学/工具函数。
// -----------------------------------------------------------------------------
namespace cuda_utils {

// 中文注释：整数向上取整除法（Ceiling Division）。
//
// 【功能】
//   计算 ceil(a / b)，即 a 除以 b 并向上取整。
//   例如：ceil_div(10, 3) = 4, ceil_div(9, 3) = 3, ceil_div(0, 3) = 0。
//
// 【使用场景】
//   在 vLLM 中广泛用于：
//     - 计算所需的 block/thread 数量（如 (num_elements + block_size - 1) / block_size）
//     - KV cache 中计算需要多少个 block 来存放给定数量的 token
//     - 量化 kernel 中计算处理所需的 group 数
//
// 【模板约束】
//   仅对整数类型启用（通过 std::enable_if_t + std::is_integral_v），避免浮点数误用。
//
// 【HOST_DEVICE_INLINE】
//   使用前文定义的宏，使此函数同时可在 CPU 和 GPU 上调用，且强制内联。
//   constexpr 保证编译期常量参数时可被编译器优化。
// -----------------------------------------------------------------------------
template <typename T>
HOST_DEVICE_INLINE constexpr std::enable_if_t<std::is_integral_v<T>, T>
ceil_div(T a, T b) {
  return (a + b - 1) / b;
}

};  // namespace cuda_utils