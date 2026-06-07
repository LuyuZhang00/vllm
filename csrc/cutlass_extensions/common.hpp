#pragma once

// 中文注释：本文件是 cutlass_extensions 的公共头文件，提供以下核心功能：
//   1. CUTLASS 错误检查宏 (CUTLASS_CHECK)
//   2. CUDA 共享内存查询工具函数
//   3. SM（Streaming Multiprocessor）版本号查询
//   4. 按 CUDA 计算能力（Compute Capability）分段的 kernel 保护模板，
//      用于在编译期将 kernel 限定在特定 GPU 架构上执行，
//      避免在不支持的架构上编译，从而减小二进制体积。

#include "cutlass/cutlass.h"
#include <climits>
#include "cuda_runtime.h"
#include <cstdio>
#include <cstdlib>

#include <torch/headeronly/util/shim_utils.h>

/**
 * Helper function for checking CUTLASS errors
 */
#define CUTLASS_CHECK(status)                           \
  {                                                     \
    cutlass::Status error = status;                     \
    STD_TORCH_CHECK(error == cutlass::Status::kSuccess, \
                    cutlassGetStatusString(error));     \
  }

// 中文注释：查询指定 GPU 设备上单个 block 可选择使用的最大共享内存大小（字节）。
// 这个值用于判断 kernel 是否可以通过 cudaFuncSetAttribute 请求额外的共享内存。
inline int get_cuda_max_shared_memory_per_block_opt_in(int const device) {
  int max_shared_mem_per_block_opt_in = 0;
  cudaDeviceGetAttribute(&max_shared_mem_per_block_opt_in,
                         cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
  return max_shared_mem_per_block_opt_in;
}

// 中文注释：获取当前 GPU 的 SM 版本号，格式为 major * 10 + minor（如 sm80 返回 80）。
int32_t get_sm_version_num();

/**
 * A wrapper for a kernel that is used to guard against compilation on
 * architectures that will never use the kernel. The purpose of this is to
 * reduce the size of the compiled binary.
 * __CUDA_ARCH__ is not defined in host code, so this lets us smuggle the ifdef
 * into code that will be executed on the device where it is defined.
 */

// 中文注释：以下是一系列 CUDA 架构守卫模板，每个模板通过 __CUDA_ARCH__ 宏在编译期
// 限定 kernel 只能在特定 SM 版本范围内运行。如果在不匹配的架构上调用，会触发 trap 中断。
// 这样做的好处是：
//   1. 避免编译出在不支持的 GPU 上运行的代码，减少二进制大小
//   2. 在运行时提供明确的错误提示
//
// SM 版本对照：
//   sm75 = Turing (RTX 20xx), sm80 = Ampere (A100), sm89 = Ada Lovelace (RTX 4090),
//   sm90 = Hopper (H100), sm100-sm120 = Blackwell, sm120 = RTX 5090

// 中文注释：限定 kernel 仅在 sm75（Turing）到 sm80（不含）之间运行。
template <typename Kernel>
struct enable_sm75_to_sm80 : Kernel {
  template <typename... Args>
  CUTLASS_DEVICE static void invoke(Args&&... args) {
#if defined __CUDA_ARCH__
  #if __CUDA_ARCH__ >= 750 && __CUDA_ARCH__ < 800
    Kernel::invoke(std::forward<Args>(args)...);
  #else
    printf("This kernel only supports sm[75, 80).\n");
    asm("trap;");
  #endif
#endif
  }
};

// 中文注释：限定 kernel 仅在 sm80（Ampere）到 sm89（Ada Lovelace，不含）之间运行。
template <typename Kernel>
struct enable_sm80_to_sm89 : Kernel {
  template <typename... Args>
  CUTLASS_DEVICE static void invoke(Args&&... args) {
#if defined __CUDA_ARCH__
  #if __CUDA_ARCH__ >= 800 && __CUDA_ARCH__ < 890
    Kernel::invoke(std::forward<Args>(args)...);
  #else
    printf("This kernel only supports sm[80, 89).\n");
    asm("trap;");
  #endif
#endif
  }
};

// 中文注释：限定 kernel 仅在 sm89（Ada Lovelace）到 sm90（Hopper，不含）之间运行。
template <typename Kernel>
struct enable_sm89_to_sm90 : Kernel {
  template <typename... Args>
  CUTLASS_DEVICE static void invoke(Args&&... args) {
#if defined __CUDA_ARCH__
  #if __CUDA_ARCH__ >= 890 && __CUDA_ARCH__ < 900
    Kernel::invoke(std::forward<Args>(args)...);
  #else
    printf("This kernel only supports sm[89, 90).\n");
    asm("trap;");
  #endif
#endif
  }
};

// 中文注释：限定 kernel 仅在 sm90（Hopper）及以上架构运行。
template <typename Kernel>
struct enable_sm90_or_later : Kernel {
  template <typename... Args>
  CUTLASS_DEVICE void operator()(Args&&... args) {
#if defined __CUDA_ARCH__
  #if __CUDA_ARCH__ >= 900
    Kernel::operator()(std::forward<Args>(args)...);
  #else
    printf("This kernel only supports sm >= 90.\n");
    asm("trap;");
  #endif
#endif
  }
};

// 中文注释：限定 kernel 仅在 sm100 到 sm120（不含）之间运行（Blackwell 系列）。
template <typename Kernel>
struct enable_sm100_to_sm120 : Kernel {
  template <typename... Args>
  CUTLASS_DEVICE void operator()(Args&&... args) {
#if defined __CUDA_ARCH__
  #if (__CUDA_ARCH__ >= 1000 && __CUDA_ARCH__ < 1200)
    Kernel::operator()(std::forward<Args>(args)...);
  #else
    printf("This kernel only supports sm[100, 120).\n");
    asm("trap;");
  #endif
#endif
  }
};

// 中文注释：限定 kernel 仅在 sm120（Blackwell RTX 5090 等）上运行。
template <typename Kernel>
struct enable_sm120_only : Kernel {
  template <typename... Args>
  CUTLASS_DEVICE void operator()(Args&&... args) {
#if defined __CUDA_ARCH__
  #if __CUDA_ARCH__ == 1200
    Kernel::operator()(std::forward<Args>(args)...);
  #else
    printf("This kernel only supports sm120a.\n");
    asm("trap;");
  #endif
#endif
  }
};

// SM12x family includes SM120 (RTX 5090) and SM121 (DGX Spark GB10)
// 中文注释：限定 kernel 仅在 SM12x 家族架构上运行，包括 sm120（RTX 5090）和 sm121（DGX Spark GB10）。
template <typename Kernel>
struct enable_sm120_family : Kernel {
  template <typename... Args>
  CUTLASS_DEVICE void operator()(Args&&... args) {
#if defined __CUDA_ARCH__
  #if (__CUDA_ARCH__ >= 1200 && __CUDA_ARCH__ < 1300)
    Kernel::operator()(std::forward<Args>(args)...);
  #else
    printf("This kernel only supports sm120f.\n");
    asm("trap;");
  #endif
#endif
  }
};
