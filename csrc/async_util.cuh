/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * =============================================================================
 * 文件功能概述（中文）
 * =============================================================================
 * 本文件封装了 CUDA 异步内存拷贝原语（cp.async），用于在 GPU kernel 中
 * 高效地将数据从全局显存（global memory）异步加载到共享内存（shared memory）。
 *
 * 【核心概念】
 *   - cp.async 是 Ampere (SM80+) 架构引入的异步拷贝指令，允许 warp 在
 *     等待数据传输的同时继续执行其他指令，从而隐藏全局内存访问延迟。
 *   - cg (cache global)：数据不经过 L1 cache，直接写入共享内存。
 *   - ca (cache all)：数据经过 L1/L2 cache 层级。
 *
 * 【提供的函数】
 *   1. cp_async_shared_global_16_cg：固定 16 字节的异步拷贝（cg 模式）
 *   2. cp_async_shared_global_ca：可变大小（4/8/16 字节）的异步拷贝（ca 模式）
 *   3. cp_async_commit_group：提交异步拷贝组（将之前发出的 cp.async 打包）
 *   4. cp_async_wait_group<N>：等待异步拷贝组中最多 N 个操作完成
 *
 * 【平台兼容性】
 *   - NVIDIA SM80+：使用 PTX 内联汇编调用原生 cp.async 指令
 *   - NVIDIA SM < 80：回退到同步的显存读写（功能正确但无异步加速）
 *   - AMD ROCm：使用同步方式（HIP 不支持 cp.async）
 *
 * 【使用场景】
 *   在 attention kernel、MLA decode 等需要大量全局内存加载到共享内存的场景中，
 *   使用这些原语可以将数据加载与计算重叠执行，显著提升 kernel 吞吐。
 * =============================================================================
 */

#pragma once

namespace vllm {
namespace cuda_async {

// 【cp_async_shared_global_16_cg】固定 16 字节的异步拷贝（cache global 模式）
// 将 glob_ptr 指向的 16 字节数据异步拷贝到 smem_ptr（共享内存地址）。
// SM80+：使用 cp.async.cg 指令，数据不经过 L1 cache。
// SM < 80 / ROCm：回退到同步的 int4（16 字节）读写。
__device__ __forceinline__ void cp_async_shared_global_16_cg(
    void* smem_ptr, const void* glob_ptr) {
#if defined(USE_ROCM)
  *reinterpret_cast<int4*>(smem_ptr) = *reinterpret_cast<const int4*>(glob_ptr);
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
               :
               : "r"(smem), "l"(glob_ptr));
#elif defined(__CUDA_ARCH__)
  *reinterpret_cast<int4*>(smem_ptr) = *reinterpret_cast<const int4*>(glob_ptr);
#else
  (void)smem_ptr;
  (void)glob_ptr;
#endif
}

// 【cp_async_shared_global_ca】可变大小的异步拷贝（cache all 模式）
// 支持 4/8/16 字节三种大小，数据经过 L1/L2 cache 层级。
// SM80+：根据 size_bytes 选择对应的 cp.async.ca 指令（4/8/16 字节）。
// SM < 80 / ROCm：回退到同步读写。
__device__ __forceinline__ void cp_async_shared_global_ca(void* smem_ptr,
                                                          const void* glob_ptr,
                                                          int size_bytes) {
#if defined(USE_ROCM)
  if (size_bytes == 4) {
    *reinterpret_cast<uint32_t*>(smem_ptr) =
        *reinterpret_cast<const uint32_t*>(glob_ptr);
  } else if (size_bytes == 8) {
    *reinterpret_cast<uint64_t*>(smem_ptr) =
        *reinterpret_cast<const uint64_t*>(glob_ptr);
  } else {
    *reinterpret_cast<int4*>(smem_ptr) =
        *reinterpret_cast<const int4*>(glob_ptr);
  }
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  if (size_bytes == 4) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"
                 :
                 : "r"(smem), "l"(glob_ptr));
  } else if (size_bytes == 8) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n"
                 :
                 : "r"(smem), "l"(glob_ptr));
  } else {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                 :
                 : "r"(smem), "l"(glob_ptr));
  }
#elif defined(__CUDA_ARCH__)
  if (size_bytes == 4) {
    *reinterpret_cast<uint32_t*>(smem_ptr) =
        *reinterpret_cast<const uint32_t*>(glob_ptr);
  } else if (size_bytes == 8) {
    *reinterpret_cast<uint64_t*>(smem_ptr) =
        *reinterpret_cast<const uint64_t*>(glob_ptr);
  } else {
    *reinterpret_cast<int4*>(smem_ptr) =
        *reinterpret_cast<const int4*>(glob_ptr);
  }
#else
  (void)smem_ptr;
  (void)glob_ptr;
  (void)size_bytes;
#endif
}

// 【cp_async_commit_group】提交异步拷贝组
// 将之前发出的所有 cp.async 操作打包为一个"组"，后续可以用 cp_async_wait_group
// 等待该组中的操作完成。SM < 80 和 ROCm 上为空操作。
__device__ __forceinline__ void cp_async_commit_group() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800 && !defined(USE_ROCM)
  asm volatile("cp.async.commit_group;\n" ::);
#endif
}

// 【cp_async_wait_group<N>】等待异步拷贝组完成
// 等待当前未完成的异步拷贝操作中最多 N 个操作完成。
// 例如 wait_group<0> 等待所有操作完成，wait_group<1> 允许最多 1 个未完成。
// SM < 80 和 ROCm 上为空操作。
template <int n>
__device__ __forceinline__ void cp_async_wait_group() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800 && !defined(USE_ROCM)
  asm volatile("cp.async.wait_group %0;\n" : : "n"(n));
#endif
}

}  // namespace cuda_async
}  // namespace vllm
