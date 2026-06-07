#pragma once

// =============================================================================
// 中文注释: Marlin 量化 GEMM 公共头文件
// =============================================================================
// 本文件定义了 Marlin 量化矩阵乘法 kernel 的基础常量、数据结构和异步拷贝辅助函数。
//
// Marlin 是一种高性能的量化 GEMM (通用矩阵乘法) 实现，特点:
//   1. 支持 4-bit/8-bit 权重量化 (W4A16, W8A16, W4A8 等)
//   2. 使用 Tensor Core 进行矩阵乘法加速
//   3. 通过多级异步流水线 (cp_async) 隐藏内存延迟
//   4. 支持 grouped quantization 和 activation reordering
//
// 关键常量说明:
//   - default_threads = 256: 每个 thread block 8 个 warp (8*32=256)
//     每个 SM 有 4 个 scheduler，8 warp 允许每个 scheduler 调度 2 个 warp，实现延迟隐藏
//   - tile_size = 16: Tensor Core 的最小 tile 大小 (16x16)
//   - pipe_stages = 4: 异步加载流水线的阶段数，需要 4 个阶段才能完全隐藏内存延迟
//   - min_thread_n/k = 64: 每个线程处理的最小 N/K 维度
//   - max_thread_n = 256: 每个线程处理的最大 N 维度
// =============================================================================

#ifndef _marlin_cuh
  #define _marlin_cuh
  // These torch headers are only needed by non-stable callers (e.g. ops.cu).
  // Guard them so that stable ABI targets can still include marlin.cuh
  // for Vec, constants, and cp_async helpers without pulling in torch/all.h.
  #ifndef TORCH_TARGET_VERSION
    #include <torch/all.h>
    #include <ATen/cuda/CUDAContext.h>
    #include <c10/cuda/CUDAGuard.h>
  #endif
  #include <cuda.h>
  #include <cuda_fp16.h>
  #include <cuda_runtime.h>
  #include <iostream>

  #ifndef MARLIN_NAMESPACE_NAME
    #define MARLIN_NAMESPACE_NAME marlin
  #endif

namespace MARLIN_NAMESPACE_NAME {

// Marlin params

// 8 warps are a good choice since every SM has 4 schedulers and having more
// than 1 warp per schedule allows some more latency hiding. At the same time,
// we want relatively few warps to have many registers per warp and small tiles.
// 中文注释: 默认线程数 256 = 8 warps * 32 threads/warp
// 选择 8 warp 的原因: 每个 SM 有 4 个 warp scheduler，8 warp 让每个 scheduler
// 调度 2 个 warp，当一个 warp 等待内存时另一个可以执行，实现延迟隐藏
static constexpr int default_threads = 256;

// 中文注释: 异步流水线阶段数。4 阶段可以完全隐藏全局内存到共享内存的加载延迟
static constexpr int pipe_stages =
    4;  // 4 pipeline stages fit into shared memory

// 中文注释: N/K 维度的最小/最大 tile 大小 (单位: 16 元素)
static constexpr int min_thread_n = 64;
static constexpr int min_thread_k = 64;
static constexpr int max_thread_n = 256;

// 中文注释: Tensor Core tile 大小，Tensor Core 的 MMA 指令最小操作 16x16 的矩阵块
static constexpr int tile_size = 16;
static constexpr int max_par = 16;

// Repack params
static constexpr int repack_stages = 8;

static constexpr int repack_threads = 256;

// 中文注释: 权重 repack 时的 tile 尺寸
// tile_n_size = tile_k_size * 4 是因为 4-bit 量化时每个 int32 存储 8 个权重，
// 而 tile_k_size=16 对应 16 个 K 维度元素
static constexpr int tile_k_size = tile_size;
static constexpr int tile_n_size = tile_k_size * 4;

// Helpers
// 中文注释: 固定长度向量模板，用于寄存器中存储多个元素
// 例如 Vec<int, 4> 表示 4 个 int，对应一个 int4 (128-bit)
// 在 Marlin kernel 中用于寄存器级别的数据搬运和存储
template <typename T, int n>
struct Vec {
  T elems[n];
  __device__ T& operator[](int i) { return elems[i]; }
};

using I4 = Vec<int, 4>;

// 中文注释: 向上取整的整数除法
constexpr int div_ceil(int a, int b) { return (a + b - 1) / b; }

// 中文注释: SM < 80 (Ampere 之前) 的异步拷贝退化实现
// 在旧架构上，cp.async 指令不可用，退化为普通的同步内存拷贝
// 虽然性能不如真正的异步拷贝，但保持了相同的接口
  #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800

__device__ inline void cp_async1_ca_pred(void* smem_ptr, const void* glob_ptr,
                                         bool pred = true) {
  if (pred) {
    reinterpret_cast<int32_t*>(smem_ptr)[0] =
        reinterpret_cast<const int32_t*>(glob_ptr)[0];
  }
}

__device__ inline void cp_async2_ca_pred(void* smem_ptr, const void* glob_ptr,
                                         bool pred = true) {
  if (pred) {
    reinterpret_cast<int64_t*>(smem_ptr)[0] =
        reinterpret_cast<const int64_t*>(glob_ptr)[0];
  }
}

__device__ inline void cp_async4_ca_pred(void* smem_ptr, const void* glob_ptr,
                                         bool pred = true) {
  if (pred) {
    reinterpret_cast<int4*>(smem_ptr)[0] =
        reinterpret_cast<const int4*>(glob_ptr)[0];
  }
}

__device__ inline void cp_async4_pred(void* smem_ptr, const void* glob_ptr,
                                      bool pred = true) {
  if (pred) {
    reinterpret_cast<int4*>(smem_ptr)[0] =
        reinterpret_cast<const int4*>(glob_ptr)[0];
  }
}

__device__ inline void cp_async4(void* smem_ptr, const void* glob_ptr) {
  reinterpret_cast<int4*>(smem_ptr)[0] =
      reinterpret_cast<const int4*>(glob_ptr)[0];
}

__device__ inline void cp_async_fence() {}

template <int n>
__device__ inline void cp_async_wait() {}

  #else

// 中文注释: SM >= 80 (Ampere+) 的硬件异步拷贝指令实现
// cp.async 是 Ampere 引入的指令，允许从全局内存异步拷贝到共享内存，
// 不占用计算单元，从而实现计算与加载的重叠 (pipeline)
//
// 命名规则:
//   cp_async1/2/4: 拷贝 1/2/4 个 int4 (4/8/16 字节)
//   ca: cache all (缓存所有层级)
//   cg: cache global (只缓存全局内存层级，绕过 L1)
//   pred: 支持谓词 (条件执行)，用于边界检查
//
// 使用模式:
//   1. cp_async4(smem, gmem): 发起异步拷贝
//   2. cp_async_fence(): 提交当前所有异步拷贝为一个 group
//   3. cp_async_wait<N>(): 等待前 N 个 group 完成，然后可以安全读取共享内存
__device__ inline void cp_async1_ca_pred(void* smem_ptr, const void* glob_ptr,
                                         bool pred = true) {
  const int BYTES = 4;
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
      "{\n"
      "   .reg .pred p;\n"
      "   setp.ne.b32 p, %0, 0;\n"
      "   @p cp.async.ca.shared.global [%1], [%2], %3;\n"
      "}\n" ::"r"((int)pred),
      "r"(smem), "l"(glob_ptr), "n"(BYTES));
}

__device__ inline void cp_async2_ca_pred(void* smem_ptr, const void* glob_ptr,
                                         bool pred = true) {
  const int BYTES = 8;
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
      "{\n"
      "   .reg .pred p;\n"
      "   setp.ne.b32 p, %0, 0;\n"
      "   @p cp.async.ca.shared.global [%1], [%2], %3;\n"
      "}\n" ::"r"((int)pred),
      "r"(smem), "l"(glob_ptr), "n"(BYTES));
}

__device__ inline void cp_async4_ca_pred(void* smem_ptr, const void* glob_ptr,
                                         bool pred = true) {
  const int BYTES = 16;
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
      "{\n"
      "   .reg .pred p;\n"
      "   setp.ne.b32 p, %0, 0;\n"
      "   @p cp.async.ca.shared.global [%1], [%2], %3;\n"
      "}\n" ::"r"((int)pred),
      "r"(smem), "l"(glob_ptr), "n"(BYTES));
}

__device__ inline void cp_async4_pred(void* smem_ptr, const void* glob_ptr,
                                      bool pred = true) {
  const int BYTES = 16;
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
      "{\n"
      "   .reg .pred p;\n"
      "   setp.ne.b32 p, %0, 0;\n"
      "   @p cp.async.cg.shared.global [%1], [%2], %3;\n"
      "}\n" ::"r"((int)pred),
      "r"(smem), "l"(glob_ptr), "n"(BYTES));
}

__device__ inline void cp_async4(void* smem_ptr, const void* glob_ptr) {
  const int BYTES = 16;
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
      "{\n"
      "   cp.async.cg.shared.global [%0], [%1], %2;\n"
      "}\n" ::"r"(smem),
      "l"(glob_ptr), "n"(BYTES));
}

// 中文注释: 提交异步拷贝组，将之前所有 cp_async 操作打包为一个 group
__device__ inline void cp_async_fence() {
  asm volatile("cp.async.commit_group;\n" ::);
}

// 中文注释: 等待前 n 个异步拷贝组完成
// n=0 表示等待所有组完成 (cp_async_wait<0>)
// n=1 表示只等待最早的 1 个组，允许后续组继续执行 (用于 pipeline)
template <int n>
__device__ inline void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(n));
}

  #endif

}  // namespace MARLIN_NAMESPACE_NAME

#endif