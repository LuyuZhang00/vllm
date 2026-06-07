#pragma once
// ============================================================================
// 【模块概述】custom_all_reduce.cuh — 自定义 AllReduce 实现（基于 CUDA IPC）
// ============================================================================
// 本文件实现了一个基于 CUDA IPC（进程间通信）的自定义 AllReduce 操作，
// 用于替代 NCCL 在多 GPU 间高效执行张量归约。
//
// 核心设计思路：
//   1. 利用 CUDA IPC 共享内存，直接在各 GPU 之间进行 P2P 数据读写，
//      避免了 NCCL 的额外开销，特别适合中小规模张量。
//   2. 提供两种 AllReduce 算法：
//      - 一阶段（1-stage）：每个 GPU 直接读取所有其他 GPU 的数据并累加归约。
//      - 两阶段（2-stage）：Reduce-Scatter + All-Gather，通信量更均衡。
//   3. 使用 flag 计数器进行跨 GPU 同步，确保所有 GPU 在正确时机开始/结束操作。
//   4. 完全兼容 CUDA Graph 捕获，支持图重放时自动解析 IPC 指针。
//
// 典型使用场景：
//   vLLM 在张量并行（Tensor Parallel）推理时，各 GPU 分别计算各自的注意力输出，
//   之后需要 AllReduce 合并结果。本模块在 NVLink/xGMI 全连接拓扑下，
//   对中小规模张量（几十 KB 到几百 KB）比 NCCL 更快。
// ============================================================================

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// ROCm 平台兼容：将 HIP 的 bfloat16 类型映射到 CUDA 风格的 nv_bfloat16
#if defined(USE_ROCM)
typedef __hip_bfloat16 nv_bfloat16;
#endif

#include <iostream>
#include <array>
#include <limits>
#include <map>
#include <unordered_map>
#include <vector>
#include <cstdlib>
#include <cstring>

namespace vllm {

// 【错误检查宏】封装 CUDA API 调用，失败时打印错误信息并终止进程。
#define CUDACHECK(cmd)                                              \
  do {                                                              \
    cudaError_t e = cmd;                                            \
    if (e != cudaSuccess) {                                         \
      printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__, \
             cudaGetErrorString(e));                                \
      exit(EXIT_FAILURE);                                           \
    }                                                               \
  } while (0)

// 【AllReduce Kernel 最大 block 数量】
// AllReduce kernel 中使用的 block 数量上限。实验表明使用有限的 SM 数量
// （而非所有 SM）反而能获得更好的性能，因为过多 SM 会导致 NVLink 总线争用。
constexpr int kMaxBlocks = 36;

// 【默认 block 数量限制】
// 经过在 A100、A10、A30、T4、V100 等多种 GPU 上的网格搜索测试得出的默认值。
// ROCm 平台由于架构差异，使用更保守的 16 个 block。
#ifndef USE_ROCM
const int defaultBlockLimit = 36;
CUpointer_attribute rangeStartAddrAttr = CU_POINTER_ATTRIBUTE_RANGE_START_ADDR;
#else
const int defaultBlockLimit = 16;
hipPointer_attribute rangeStartAddrAttr =
    HIP_POINTER_ATTRIBUTE_RANGE_START_ADDR;
#endif

// 【同步标志类型】使用 uint32_t 作为跨 GPU 同步的计数器类型。
// 计数器可能会溢出，但无符号整数溢出是 C++ 标准保证的定义行为，不影响正确性。
using FlagType = uint32_t;

// 【Signal 同步结构体】用于跨 GPU 的 AllReduce 同步。
// 设计要点：
//   需要两组计数器（start 和 end）分别用于"操作开始"和"操作结束"两个同步点。
//   原因：GPU A 可能已到达第二个同步点，而 GPU B 还在第一个同步点等待。
//   如果只用一组计数器，GPU A 可能写入 counter+1，而 GPU B 还在等待 counter，
//   导致数据竞争。使用交替计数器数组可以避免此问题。
//   _flag 数组存储每个 block 当前的递增标志值，用于确定下一次同步的目标值。
struct Signal {
  alignas(128) FlagType start[kMaxBlocks][8];  // 操作开始同步标志
  alignas(128) FlagType end[kMaxBlocks][8];    // 操作结束同步标志
  alignas(128) FlagType _flag[kMaxBlocks];     // 每个 block 的递增标志计数
};

// 【RankData 结构体】存储所有 rank（GPU）上同一 buffer 的指针列表。
// 在 AllReduce 中，每个 GPU 需要知道其他 GPU 上对应 buffer 的地址，
// 以便直接通过 NVLink/xGMI 读取数据。最多支持 8 GPU。
struct __align__(16) RankData {
  const void* ptrs[8];  // 每个 rank 上对应 buffer 的设备指针
};

// 【RankSignals 结构体】存储所有 rank 的 Signal 指针，用于跨 GPU 同步。
struct __align__(16) RankSignals {
  Signal* signals[8];  // 每个 rank 的同步信号缓冲区指针
};

// 【对齐数组模板】类似 std::array，但对齐到 T 元素大小 * 数组长度。
// 对齐的目的是确保可以使用 128-bit 宽的 load/store 指令（ld.128/st.128），
// 这样每个线程一次可以读写 16 字节，最大化显存带宽利用率。
template <typename T, int sz>
struct __align__(alignof(T) * sz) array_t {
  T data[sz];
  using type = T;
  static constexpr int size = sz;
};

// 【打包类型模板】将数据类型封装为 16 字节的打包类型，用于高效的 load/store。
// 设计目标：让编译器生成 ld.128 和 st.128 指令，每次读写 16 字节。
// P（Packed）类型：用于内存读写，例如 half 为 array_t<half, 8>（8个half=16字节）。
// A（Accumulator）类型：用于归约计算时的中间累加，始终用 float 保持精度。
template <typename T>
struct packed_t {
  // the (P)acked type for load/store
  using P = array_t<T, 16 / sizeof(T)>;
  // the (A)ccumulator type for reduction
  using A = array_t<float, 16 / sizeof(T)>;
};

// 【设备内联宏】强制编译器将函数内联到设备代码中，减少函数调用开销。
#define DINLINE __device__ __forceinline__

// ====================================================================
// 【标量类型转换函数】用于 half/bfloat16 与 float 之间的相互转换。
// AllReduce 归约过程中，为了保证精度，累加操作在 float 精度下进行，
// 最终结果再转回原始类型。
// ====================================================================
DINLINE float upcast_s(half val) { return __half2float(val); }

template <typename T>
DINLINE T downcast_s(float val);
template <>
DINLINE half downcast_s(float val) {
  return __float2half(val);
}

// 【标量加法函数】针对不同类型的带赋值加法（a += b）。
// 注意：在 PyTorch 编译环境下，half 和 bfloat16 的 + 运算符被禁用，
// 因此直接调用 CUDA intrinsics（__hadd）来实现加法。
DINLINE half& assign_add(half& a, half b) {
  a = __hadd(a, b);
  return a;
}
DINLINE float& assign_add(float& a, float b) { return a += b; }

// bfloat16 类型需要 SM >= 80（A100 及以上）才原生支持
#if (__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))
DINLINE float upcast_s(nv_bfloat16 val) { return __bfloat162float(val); }
template <>
DINLINE nv_bfloat16 downcast_s(float val) {
  return __float2bfloat16(val);
}
DINLINE nv_bfloat16& assign_add(nv_bfloat16& a, nv_bfloat16 b) {
  a = __hadd(a, b);
  return a;
}
#endif

// 【打包加法】对两个 array_t 向量逐元素执行累加（a += b）。
// 使用 #pragma unroll 让编译器展开循环，消除循环控制开销。
template <typename T, int N>
DINLINE array_t<T, N>& packed_assign_add(array_t<T, N>& a, array_t<T, N> b) {
#pragma unroll
  for (int i = 0; i < N; i++) {
    assign_add(a.data[i], b.data[i]);
  }
  return a;
}

// 【向上类型转换】将 array_t<T, N> 转换为 array_t<float, N>。
// 如果 T 已经是 float，则直接返回，不做无意义转换。
// 归约累加在 float 精度下进行，避免 half/bfloat16 精度损失。
template <typename T, int N>
DINLINE array_t<float, N> upcast(array_t<T, N> val) {
  if constexpr (std::is_same<T, float>::value) {
    return val;
  } else {
    array_t<float, N> out;
#pragma unroll
    for (int i = 0; i < N; i++) {
      out.data[i] = upcast_s(val.data[i]);
    }
    return out;
  }
}

// 【向下类型转换】将 array_t<float, N> 转换回目标类型 array_t<O, N>。
// 归约完成后将 float 结果转回原始精度类型。
template <typename O>
DINLINE O downcast(array_t<float, O::size> val) {
  if constexpr (std::is_same<typename O::type, float>::value) {
    return val;
  } else {
    O out;
#pragma unroll
    for (int i = 0; i < O::size; i++) {
      out.data[i] = downcast_s<typename O::type>(val.data[i]);
    }
    return out;
  }
}

// ====================================================================
// 【CUDA 内存屏障原语】（仅 CUDA 平台，非 ROCm）
// 这些函数通过 PTX 内联汇编实现跨 GPU 的内存可见性控制。
//
// 关键概念：
//   - release（释放语义）：确保此 store 之前的所有读写操作对其他 GPU 可见。
//   - acquire（获取语义）：确保此 load 之后的所有读写操作看到最新的数据。
//   - volatile（易失语义）：禁止编译器优化（如缓存、重排序），但不保证
//     跨 GPU 的内存可见性，适用于同步对性能要求不高或仅需本地同步的场景。
//
// SM >= 700（Volta 架构及以上）使用原生的 st.release/ld.acquire 指令，
// 更低版本回退到 membar + volatile 方案。
// ====================================================================
#if !defined(USE_ROCM)

// 【带 release 语义的存储】写入标志值，保证此前的写操作对其他 GPU 可见。
static DINLINE void st_flag_release(FlagType* flag_addr, FlagType flag) {
  #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
  asm volatile("st.release.sys.global.u32 [%1], %0;" ::"r"(flag),
               "l"(flag_addr));
  #else
  asm volatile("membar.sys; st.volatile.global.u32 [%1], %0;" ::"r"(flag),
               "l"(flag_addr));
  #endif
}

// 【带 acquire 语义的加载】读取标志值，保证此后的读操作能看到最新的数据。
static DINLINE FlagType ld_flag_acquire(FlagType* flag_addr) {
  FlagType flag;
  #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];"
               : "=r"(flag)
               : "l"(flag_addr));
  #else
  asm volatile("ld.volatile.global.u32 %0, [%1]; membar.gl;"
               : "=r"(flag)
               : "l"(flag_addr));
  #endif
  return flag;
}

// 【volatile 存储】不带 release 语义的存储，仅保证编译器不优化掉此写操作。
// 用于不需要跨 GPU 内存可见性保证的场景（如纯同步计数器）。
static DINLINE void st_flag_volatile(FlagType* flag_addr, FlagType flag) {
  asm volatile("st.volatile.global.u32 [%1], %0;" ::"r"(flag), "l"(flag_addr));
}

// 【volatile 加载】不带 acquire 语义的加载，仅保证编译器不优化掉此读操作。
static DINLINE FlagType ld_flag_volatile(FlagType* flag_addr) {
  FlagType flag;
  asm volatile("ld.volatile.global.u32 %0, [%1];"
               : "=r"(flag)
               : "l"(flag_addr));
  return flag;
}

// ====================================================================
// 【barrier_at_start】AllReduce 的第一个同步屏障（CUDA 平台版本）
// 作用：确保所有 GPU 上的所有 block 都到达此同步点后，才开始归约计算。
//
// 同步机制：
//   1. 每个 block 根据上次的标志值计算新的期望值 flag。
//   2. 前 ngpus 个线程（每个线程对应一个 peer GPU）将自己的 flag 写入
//      peer GPU 的 start 计数器（volatile 写，不需要 release 语义，
//      因为这是第一个屏障，不需要保证之前的内存可见性）。
//   3. 每个线程等待对应的 peer GPU 写入匹配的 flag 值。
//   4. 全部匹配后，block 内所有线程同步（__syncthreads）。
//   5. 由线程 0 更新本地的递增标志，为下次同步做准备。
// ====================================================================
template <int ngpus>
DINLINE void barrier_at_start(const RankSignals& sg, Signal* self_sg,
                              int rank) {
  uint32_t flag = self_sg->_flag[blockIdx.x] + 1;
  if (threadIdx.x < ngpus) {
    auto peer_counter_ptr = &sg.signals[threadIdx.x]->start[blockIdx.x][rank];
    auto self_counter_ptr = &self_sg->start[blockIdx.x][threadIdx.x];
    // Write the expected counter value to peer and wait for correct value
    // from peer.
    st_flag_volatile(peer_counter_ptr, flag);
    while (ld_flag_volatile(self_counter_ptr) != flag);
  }
  __syncthreads();
  // use one thread to update flag
  if (threadIdx.x == 0) self_sg->_flag[blockIdx.x] = flag;
}

// ====================================================================
// 【barrier_at_end】AllReduce 的结束同步屏障（CUDA 平台版本）
// 作用：确保所有 GPU 上的所有 block 都完成归约计算后，才退出 kernel。
//
// 关键设计：final_sync 模板参数控制是否使用 release/acquire 语义。
//   - 非最终同步（final_sync=false）：使用 release/acquire 语义，
//     确保归约结果对其他 GPU 可见，用于两阶段算法的中间同步。
//   - 最终同步（final_sync=true）：使用 volatile 语义即可，
//     因为 kernel 结束后结果会被后续操作正确读取，无需额外保证。
// ====================================================================
template <int ngpus, bool final_sync = false>
DINLINE void barrier_at_end(const RankSignals& sg, Signal* self_sg, int rank) {
  __syncthreads();
  uint32_t flag = self_sg->_flag[blockIdx.x] + 1;
  if (threadIdx.x < ngpus) {
    auto peer_counter_ptr = &sg.signals[threadIdx.x]->end[blockIdx.x][rank];
    auto self_counter_ptr = &self_sg->end[blockIdx.x][threadIdx.x];
    // Write the expected counter value to peer and wait for correct value from
    // peer.
    if constexpr (!final_sync) {
      st_flag_release(peer_counter_ptr, flag);
      while (ld_flag_acquire(self_counter_ptr) != flag);
    } else {
      st_flag_volatile(peer_counter_ptr, flag);
      while (ld_flag_volatile(self_counter_ptr) != flag);
    }
  }
  if constexpr (!final_sync) __syncthreads();

  // use one thread to update flag
  if (threadIdx.x == 0) self_sg->_flag[blockIdx.x] = flag;
}

#else
// ====================================================================
// 【ROCm 平台的同步屏障】使用 HIP 的原子操作实现，语义与 CUDA 版本相同。
// 区别：ROCm 使用 __scoped_atomic_store_n/__scoped_atomic_load_n 替代
// PTX 内联汇编，并使用 __MEMORY_SCOPE_SYSTEM 实现跨设备可见性。
// ====================================================================

template <int ngpus>
DINLINE void barrier_at_start(const RankSignals& sg, Signal* self_sg,
                              int rank) {
  uint32_t flag = self_sg->_flag[blockIdx.x] + 1;
  if (threadIdx.x < ngpus) {
    // simultaneously write to the corresponding flag of all ranks.
    // Latency = 1 p2p write
    __scoped_atomic_store_n(&sg.signals[threadIdx.x]->start[blockIdx.x][rank],
                            flag, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
    // wait until we got true from all ranks
    while (__scoped_atomic_load_n(&self_sg->start[blockIdx.x][threadIdx.x],
                                  __ATOMIC_RELAXED,
                                  __MEMORY_SCOPE_DEVICE) < flag);
  }
  __syncthreads();
  // use one thread to update flag
  if (threadIdx.x == 0) self_sg->_flag[blockIdx.x] = flag;
}

template <int ngpus, bool final_sync = false>
DINLINE void barrier_at_end(const RankSignals& sg, Signal* self_sg, int rank) {
  __syncthreads();
  uint32_t flag = self_sg->_flag[blockIdx.x] + 1;
  if (threadIdx.x < ngpus) {
    // simultaneously write to the corresponding flag of all ranks.
    // Latency = 1 p2p write
    __scoped_atomic_store_n(&sg.signals[threadIdx.x]->end[blockIdx.x][rank],
                            flag,
                            final_sync ? __ATOMIC_RELAXED : __ATOMIC_RELEASE,
                            __MEMORY_SCOPE_SYSTEM);
    // wait until we got true from all ranks
    while (
        __scoped_atomic_load_n(&self_sg->end[blockIdx.x][threadIdx.x],
                               final_sync ? __ATOMIC_RELAXED : __ATOMIC_ACQUIRE,
                               __MEMORY_SCOPE_DEVICE) < flag);
  }
  if constexpr (!final_sync) __syncthreads();
  // use one thread to update flag
  if (threadIdx.x == 0) self_sg->_flag[blockIdx.x] = flag;
}

#endif

// ====================================================================
// 【packed_reduce】核心归约函数 — 将所有 rank 在 idx 位置的数据相加。
//
// 流程：
//   1. 将第一个 rank 的数据提升为 float 精度（upcast）。
//   2. 依次将其他 rank 的数据提升并累加到 float 累加器中。
//   3. 将累加结果降回原始精度类型（downcast）。
//
// 使用 float 累加器可以避免 half/bfloat16 累加时的精度损失。
// 这是 AllReduce 中"Reduce"步骤的核心计算。
// ====================================================================
template <typename P, int ngpus, typename A>
DINLINE P packed_reduce(const P* ptrs[], int idx) {
  A tmp = upcast(ptrs[0][idx]);
#pragma unroll
  for (int i = 1; i < ngpus; i++) {
    packed_assign_add(tmp, upcast(ptrs[i][idx]));
  }
  return downcast<P>(tmp);
}

// ====================================================================
// 【一阶段 AllReduce Kernel】cross_device_reduce_1stage
//
// 算法思路（每一步的所有 GPU 同时执行）：
//   1. 【同步】所有 GPU 的所有 block 通过 barrier_at_start 同步，
//      确保每个 GPU 的输入数据已准备就绪。
//   2. 【并行归约】每个线程处理若干元素：直接从所有 rank 的 GPU 内存中
//      读取对应位置的数据，累加后写入本 GPU 的 result 输出。
//      由于所有 rank 的累加顺序一致，保证结果 bit-wise 相同。
//   3. 【同步】通过 barrier_at_end（final_sync=true）确认所有 block
//      完成计算后退出。
//
// 适用场景：GPU 数量少（2 个）或数据量小（通信量不是瓶颈时），
// 因为每个 GPU 需要读取所有 rank 的完整数据，通信量 = (N-1) * data_size。
// ====================================================================
template <typename T, int ngpus>
__global__ void __launch_bounds__(512, 1)
    cross_device_reduce_1stage(RankData* _dp, RankSignals sg, Signal* self_sg,
                               T* __restrict__ result, int rank, int size) {
  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;
  // note: we don't reorder the address so the accumulation order is the same
  // for all ranks, ensuring bitwise identical results
  auto dp = *_dp;
  barrier_at_start<ngpus>(sg, self_sg, rank);
  // do the actual reduction
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size;
       idx += gridDim.x * blockDim.x) {
    ((P*)result)[idx] = packed_reduce<P, ngpus, A>((const P**)&dp.ptrs[0], idx);
  }
  barrier_at_end<ngpus, true>(sg, self_sg, rank);
}

// 【获取临时缓冲区指针】每个 rank 的 Signal 结构后面紧跟一段临时缓冲区，
// 用于两阶段算法中存放 Reduce-Scatter 的中间结果。
// 布局：| -- sizeof(Signal) -- | ------ 临时数据区 ----- |
template <typename P>
DINLINE P* get_tmp_buf(Signal* sg) {
  return (P*)(((Signal*)sg) + 1);
}

// ====================================================================
// 【两阶段 AllReduce Kernel】cross_device_reduce_2stage
//
// 算法思路（Reduce-Scatter + All-Gather）：
//
// 【阶段 1：Reduce-Scatter】
//   将输入数据等分为 ngpus 份。每个 GPU 负责计算其中一个分片的归约结果：
//   - GPU i 负责分片 i：读取所有 rank 的分片 i 数据，累加后写入临时缓冲区。
//   - 通信量：每个 GPU 只读取 (N-1) * (size/N) 的数据，
//     比一阶段算法的 (N-1) * size 少 N 倍。
//
// 【阶段 2：All-Gather】
//   所有 GPU 通过 barrier_at_end 同步（确保 Reduce-Scatter 结果可见）后，
//   每个 GPU 从所有其他 rank 的临时缓冲区中收集各分片的归约结果，
//   组装成完整的 AllReduce 输出。
//
// 重要：两个阶段中线程的 tid 必须一致！
//   跨 GPU 的可见性保证仅在相同 tid 的线程之间有效。
//   例如线程 i 在阶段 1 计算 start+i 的归约，
//   那么线程 i 在阶段 2 必须负责收集 start+i 的结果。
//
// 适用场景：GPU 数量 >= 4 且数据量较大时，
// 因为每轮通信量减少 N 倍，可以更好地利用 NVLink 带宽。
// ====================================================================
template <typename T, int ngpus>
__global__ void __launch_bounds__(512, 1)
    cross_device_reduce_2stage(RankData* _dp, RankSignals sg, Signal* self_sg,
                               T* __restrict__ result, int rank, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;
  // 将数据等分为 ngpus 份，当前 rank 负责第 rank 份的 Reduce-Scatter
  int part = size / ngpus;
  int start = rank * part;
  int end = rank == ngpus - 1 ? size : start + part;
  // largest_part: 最后一个分片可能包含余数，是最大的分片大小
  int largest_part = part + size % ngpus;
  const P* ptrs[ngpus];   // 所有 rank 上输入数据的指针
  P* tmps[ngpus];          // 所有 rank 上临时缓冲区的指针
#pragma unroll
  for (int i = 0; i < ngpus; i++) {
    int target = (rank + i) % ngpus;
    ptrs[i] = (const P*)_dp->ptrs[target];
    tmps[i] = get_tmp_buf<P>(sg.signals[target]);
  }
  auto tmp_out = tmps[0];  // 本 rank 的临时输出缓冲区
  barrier_at_start<ngpus>(sg, self_sg, rank);

  // stage 1: reduce scatter
  // 每个 GPU 只负责自己那份分片的归约，通信量为 (N-1)*part
  for (int idx = start + tid; idx < end; idx += stride) {
    tmp_out[idx - start] = packed_reduce<P, ngpus, A>(ptrs, idx);
  }
  barrier_at_end<ngpus>(sg, self_sg, rank);

  // stage 2: allgather. Note: it's important to match the tid between
  // the two stages, because visibility across devices is only guaranteed
  // between threads that have the same tid. If thread i computes the sum of
  // start + i in the first stage, then thread i also gathers start + i from
  // all ranks.

  // 从所有 rank 的临时缓冲区收集各分片结果，组装完整的 AllReduce 输出
  for (int idx = tid; idx < largest_part; idx += stride) {
#pragma unroll
    for (int i = 0; i < ngpus; i++) {
      int gather_from_rank = ((rank + i) % ngpus);
      // 最后一个 rank 的分片可能更大（包含余数），需要检查边界
      if (gather_from_rank == ngpus - 1 || idx < part) {
        int dst_idx = gather_from_rank * part + idx;
        ((P*)result)[dst_idx] = tmps[i][idx];
      }
    }
  }
}

// 【IPC 键类型】用于在 std::map 中索引已打开的 IPC 句柄。
using IPC_KEY = std::array<uint8_t, sizeof(cudaIpcMemHandle_t)>;
static_assert(sizeof(IPC_KEY) == sizeof(cudaIpcMemHandle_t));
static_assert(alignof(IPC_KEY) == alignof(cudaIpcMemHandle_t));

// ====================================================================
// 【CustomAllreduce 类】自定义 AllReduce 操作的核心管理类
//
// 职责：
//   1. 管理跨 GPU 的 IPC 共享内存（通过 cudaIpcMemHandle）。
//   2. 维护每个 buffer 在所有 rank 上的对等指针（RankData）。
//   3. 提供 AllReduce 接口，自动选择一阶段或两阶段算法。
//   4. 完全兼容 CUDA Graph：支持图捕获阶段预分配指针槽位，
//      图重放前再填入实际的 IPC 指针。
//
// 内存模型：此类不拥有任何设备内存，所有缓冲区由构造函数的调用方提供。
// Signal 缓冲区的布局：
//   | -- sizeof(Signal) -- | ------ 临时数据区（用于两阶段算法）----- |
// ====================================================================
class CustomAllreduce {
 public:
  int rank_;           // 当前 GPU 的 rank 编号（0 ~ world_size_-1）
  int world_size_;     // 参与 AllReduce 的 GPU 总数
  // Full NVLink or xGMI connection between GPUs.
  bool fully_connected_;  // 是否所有 GPU 间都有 NVLink/xGMI 全连接

  RankSignals sg_;     // 所有 rank 的同步信号缓冲区指针
  // Stores a map from a pointer to its peer pointers from all ranks.
  std::unordered_map<void*, RankData*> buffers_;
  // 【buffers_ 映射表】key = 本 rank 上的 buffer 地址，
  // value = 指向设备内存中 RankData 的指针（包含所有 rank 的对等指针）。
  // AllReduce 时通过此表快速查找对应的跨 rank 指针列表。
  Signal* self_sg_;    // 当前 rank 的同步信号缓冲区

  // 【CUDA Graph 支持的 RankData 管理】
  // 问题：CUDA Graph 要求 kernel 参数在捕获阶段固定，但跨 rank 的 IPC 指针
  // 在捕获阶段尚不可知（需要先完成 IPC handle 交换）。
  //
  // 解决方案：
  //   1. 捕获阶段：预分配 RankData 槽位（d_rank_data_base_），
  //      将输入 buffer 地址记录到 graph_unreg_buffers_。
  //   2. 捕获完成后：通过 get_graph_buffer_ipc_meta 获取 IPC handle。
  //   3. 在 Python 层 all-gather 所有 rank 的 IPC handle。
  //   4. 调用 register_graph_buffers 打开 IPC handle，将实际对等指针
  //      填入预先分配的 RankData 槽位中。
  RankData *d_rank_data_base_, *d_rank_data_end_;
  std::vector<void*> graph_unreg_buffers_;  // 图捕获期间注册的 buffer 地址
  // a map from IPC handles to opened IPC pointers
  std::map<IPC_KEY, char*> ipc_handles_;  // 已打开的 IPC handle 缓存（避免重复打开）

  /**
   * Signals are an array of ipc-enabled buffers from all ranks.
   * For each of the buffer, the layout is as follows:
   * | -- sizeof(Signal) -- | ------ a few MB ----- |
   * The first section is for allreduce synchronization, and the second
   * section is for storing the intermediate results required by some
   * allreduce algos.
   *
   * Note: this class does not own any device memory. Any required buffers
   * are passed in from the constructor.
   */
  // 【构造函数】初始化 CustomAllreduce 实例。
  // 参数说明：
  //   signals: 各 rank 的同步信号缓冲区指针数组（通过 IPC 共享）。
  //   rank_data: 预分配的设备内存，用于存放 RankData（对等指针列表）。
  //   rank_data_sz: rank_data 的总字节数，可容纳 rank_data_sz/sizeof(RankData) 个条目。
  //   rank: 当前 GPU 的 rank 编号。
  //   world_size: 参与 AllReduce 的 GPU 总数。
  //   fully_connected: 是否所有 GPU 间有 NVLink/xGMI 全连接（影响算法选择）。
  CustomAllreduce(Signal** signals, void* rank_data, size_t rank_data_sz,
                  int rank, int world_size, bool fully_connected = true)
      : rank_(rank),
        world_size_(world_size),
        fully_connected_(fully_connected),
        self_sg_(signals[rank]),
        d_rank_data_base_(reinterpret_cast<RankData*>(rank_data)),
        d_rank_data_end_(d_rank_data_base_ + rank_data_sz / sizeof(RankData)) {
    for (int i = 0; i < world_size_; i++) {
      sg_.signals[i] = signals[i];
    }
  }

  // 【打开 IPC 句柄】将 CUDA IPC handle 转换为可直接访问的设备指针。
  // 使用缓存机制（ipc_handles_ 避免重复打开同一个 handle，
  // 因为 cudaIpcOpenMemHandle 有引用计数，重复打开会增加开销。
  char* open_ipc_handle(const void* ipc_handle) {
    auto [it, new_handle] =
        ipc_handles_.insert({*((IPC_KEY*)ipc_handle), nullptr});
    if (new_handle) {
      char* ipc_ptr;
      CUDACHECK(cudaIpcOpenMemHandle((void**)&ipc_ptr,
                                     *((const cudaIpcMemHandle_t*)ipc_handle),
                                     cudaIpcMemLazyEnablePeerAccess));
      it->second = ipc_ptr;
    }
    return it->second;
  }

  // 【获取图 buffer 的 IPC 元数据】
  // 在 CUDA Graph 捕获完成后调用，获取所有注册 buffer 的 IPC handle 和偏移量。
  //
  // 流程：
  //   1. 遍历 graph_unreg_buffers_ 中记录的所有 buffer 地址。
  //   2. 通过 cuPointerGetAttribute 获取每个地址所属分配的基地址（base_ptr）。
  //      这是必须的，因为 IPC handle 只能共享整个分配的基地址，
  //      不能共享分配内部的某个偏移地址。
  //   3. 对基地址调用 cudaIpcGetMemHandle 获取 IPC handle。
  //   4. 记录 buffer 地址相对于基地址的偏移量，供对端 rank 计算实际指针。
  //
  // 返回值：(handles 字符串, offsets 向量)，
  //   handles 是连续存放的所有 IPC handle，用于 all-gather 交换。
  std::pair<std::string, std::vector<int64_t>> get_graph_buffer_ipc_meta() {
    auto num_buffers = graph_unreg_buffers_.size();
    auto handle_sz = sizeof(cudaIpcMemHandle_t);
    std::string handles(handle_sz * num_buffers, static_cast<char>(0));
    std::vector<int64_t> offsets(num_buffers);
    for (int i = 0; i < num_buffers; i++) {
      auto ptr = graph_unreg_buffers_[i];
      void* base_ptr;
      // note: must share the base address of each allocation, or we get wrong
      // address
      if (cuPointerGetAttribute(&base_ptr, rangeStartAddrAttr,
                                (CUdeviceptr)ptr) != CUDA_SUCCESS)
        throw std::runtime_error("failed to get pointer attr");
      CUDACHECK(cudaIpcGetMemHandle(
          (cudaIpcMemHandle_t*)&handles[i * handle_sz], base_ptr));
      offsets[i] = ((char*)ptr) - ((char*)base_ptr);
    }
    return std::make_pair(handles, offsets);
  }

  // 【检查 RankData 缓冲区容量】确保还有足够的空闲槽位容纳 num 个新条目。
  void check_rank_data_capacity(size_t num = 1) {
    if (d_rank_data_base_ + num > d_rank_data_end_)
      throw std::runtime_error(
          "Rank data buffer is overflowed by " +
          std::to_string(d_rank_data_base_ + num - d_rank_data_end_));
  }

  /**
   * Register already-shared IPC pointers.
   */
  // 【注册已交换的 IPC buffer】将一组已通过 IPC 交换的对等指针注册到 AllReduce 系统。
  // 参数 ptrs: 长度为 world_size 的数组，ptrs[i] 是 rank i 上对应 buffer 的设备指针。
  // 流程：
  //   1. 将所有 rank 的指针打包成 RankData。
  //   2. 拷贝到设备内存中的下一个空闲槽位。
  //   3. 在 buffers_ 映射表中记录：本 rank 的 buffer 地址 -> 设备上的 RankData。
  //      AllReduce kernel 执行时通过此表查找所有 rank 的对等指针。
  void register_buffer(void** ptrs) {
    check_rank_data_capacity();
    RankData data;
    for (int i = 0; i < world_size_; i++) {
      data.ptrs[i] = ptrs[i];
    }
    auto d_data = d_rank_data_base_++;
    CUDACHECK(
        cudaMemcpy(d_data, &data, sizeof(RankData), cudaMemcpyHostToDevice));
    buffers_[ptrs[rank_]] = d_data;
  }

  // Note: when registering graph buffers, we intentionally choose to not
  // deduplicate the addresses. That means if the allocator reuses some
  // addresses, they will be registered again. This is to account for the
  // remote possibility of different allocation patterns between ranks. For
  // example, rank 1 may get the same input address for the second allreduce,
  // but rank 2 got a different address. IPC handles have internal reference
  // counting mechanism so overhead should be small.
  // 【注册 CUDA Graph 的 buffer】
  // 在所有 rank 交换完 IPC handle 后调用，完成图 buffer 的最终注册。
  //
  // 流程：
  //   1. 遍历 graph_unreg_buffers_ 中的所有 buffer。
  //   2. 对每个 buffer，从各 rank 传入的 handles 中提取对应的 IPC handle。
  //   3. 调用 open_ipc_handle 将 IPC handle 转换为设备指针，
  //      再加上偏移量得到实际的 buffer 地址。
  //   4. 本 rank 的 buffer 直接使用本地地址。
  //   5. 将所有 RankData 批量拷贝到设备内存。
  //   6. 清空 graph_unreg_buffers_，表示所有图 buffer 已注册完毕。
  //
  // 注意：不进行地址去重。如果内存分配器复用了某地址，会再次注册。
  // 这是为了应对不同 rank 间分配模式可能不同的边缘情况。
  // IPC handle 有内部引用计数，重复注册的开销很小。
  void register_graph_buffers(
      const std::vector<std::string>& handles,
      const std::vector<std::vector<int64_t>>& offsets) {
    auto num_buffers = graph_unreg_buffers_.size();
    check_rank_data_capacity(num_buffers);
    std::vector<RankData> rank_data(num_buffers);
    for (int i = 0; i < num_buffers; i++) {
      auto self_ptr = graph_unreg_buffers_[i];
      auto& rd = rank_data[i];
      for (int j = 0; j < world_size_; j++) {
        if (j != rank_) {
          char* handle =
              open_ipc_handle(&handles[j][i * sizeof(cudaIpcMemHandle_t)]);
          handle += offsets[j][i];
          rd.ptrs[j] = handle;
        } else {
          rd.ptrs[j] = self_ptr;
        }
      }
    }
    CUDACHECK(cudaMemcpy(d_rank_data_base_, rank_data.data(),
                         sizeof(RankData) * num_buffers,
                         cudaMemcpyHostToDevice));
    d_rank_data_base_ += num_buffers;
    graph_unreg_buffers_.clear();
  }

  /**
   * Performs allreduce, assuming input has already been registered.
   *
   * Block and grid default configs are results after careful grid search.
   * Using 36 blocks give the best or close to the best runtime on the devices
   * I tried: A100, A10, A30, T4, V100. You'll notice that NCCL kernels also
   * only take a small amount of SMs. Not quite sure the underlying reason,
   * but my guess is that too many SMs will cause contention on NVLink bus.
   */
  // ====================================================================
  // 【allreduce】执行 AllReduce 操作的主入口函数。
  //
  // 参数说明：
  //   stream: CUDA 流，kernel 在此流上异步执行。
  //   input: 输入张量的设备指针（必须已通过 register_buffer 或图捕获注册）。
  //   output: 输出张量的设备指针，存放 AllReduce 的结果。
  //   size: 输入张量的元素个数（必须是打包类型大小的整数倍）。
  //   threads: 每个 block 的线程数（默认 512，经网格搜索确定）。
  //   block_limit: 使用的 block 数量上限（默认 36，NVLink 总线争用限制）。
  //
  // 【核心流程】
  //   1. 验证：检查 size 是否是打包大小的倍数，block_limit 是否合法。
  //   2. 查找对等指针：
  //      - 如果当前处于 CUDA Graph 捕获状态，预分配一个 RankData 槽位，
  //        并将 input 记录到 graph_unreg_buffers_，后续再填入实际指针。
  //      - 否则，从 buffers_ 映射表中查找已注册的对等指针。
  //   3. 计算 kernel 配置：将 size 除以打包大小得到打包后的元素数，
  //      根据元素数和 block_limit 确定实际 block 数。
  //   4. 选择算法：根据环境变量 VLLM_CUSTOM_ALLREDUCE_ALGO 或启发式规则
  //      选择一阶段或两阶段算法。
  //   5. 启动 kernel。
  //
  // 【算法选择启发式】（当未设置环境变量时）
  //   - 2 GPU：始终使用一阶段算法（通信量差异不大，一阶段更简单）。
  //   - 4 GPU 全连接且数据 < 512KB：一阶段。
  //   - 8 GPU 全连接且数据 < 256KB：一阶段。
  //   - 其他情况：两阶段算法（通信量更均衡）。
  // ====================================================================
  template <typename T>
  void allreduce(cudaStream_t stream, T* input, T* output, int size,
                 int threads = 512, int block_limit = defaultBlockLimit) {
    auto d = packed_t<T>::P::size;
    if (size % d != 0)
      throw std::runtime_error(
          "custom allreduce currently requires input length to be multiple "
          "of " +
          std::to_string(d));
    if (block_limit > kMaxBlocks)
      throw std::runtime_error("max supported block limit is " +
                               std::to_string(kMaxBlocks) + ". Got " +
                               std::to_string(block_limit));

    RankData* ptrs;
    cudaStreamCaptureStatus status;
    CUDACHECK(cudaStreamIsCapturing(stream, &status));
    if (status == cudaStreamCaptureStatusActive) {
      // 【CUDA Graph 捕获模式】预分配 RankData 槽位，指针稍后填入
      ptrs = d_rank_data_base_ + graph_unreg_buffers_.size();
      graph_unreg_buffers_.push_back(input);
    } else {
      // 【正常执行模式】从已注册的 buffer 映射表中查找对等指针
      auto it = buffers_.find(input);
      if (it == buffers_.end())
        throw std::runtime_error(
            "buffer address " +
            std::to_string(reinterpret_cast<uint64_t>(input)) +
            " is not registered!");
      ptrs = it->second;
    }

    size /= d;  // 转换为打包类型的元素数
    auto bytes = size * sizeof(typename packed_t<T>::P);  // 实际数据字节数
    int blocks = std::min(block_limit, (size + threads - 1) / threads);

    // Check environment variable once
    // 【算法选择】可通过 VLLM_CUSTOM_ALLREDUCE_ALGO 环境变量强制指定算法
    const char* env_algo = std::getenv("VLLM_CUSTOM_ALLREDUCE_ALGO");
    bool force_1stage = false;
    bool force_2stage = false;
    if (env_algo != nullptr) {
      if (std::strcmp(env_algo, "1stage") == 0 ||
          std::strcmp(env_algo, "oneshot") == 0) {
        force_1stage = true;
      } else if (std::strcmp(env_algo, "2stage") == 0 ||
                 std::strcmp(env_algo, "twoshot") == 0) {
        force_2stage = true;
      } else {
        throw std::runtime_error(
            "Invalid VLLM_CUSTOM_ALLREDUCE_ALGO: " + std::string(env_algo) +
            ". Valid values: 1stage, oneshot, 2stage, twoshot");
      }
    }

// 【kernel 启动宏】简化 kernel 调用代码
#define KL(ngpus, name)                                                       \
  name<T, ngpus><<<blocks, threads, 0, stream>>>(ptrs, sg_, self_sg_, output, \
                                                 rank_, size);
// 【分支宏】根据 GPU 数量和数据大小选择算法
#define REDUCE_CASE(ngpus)                              \
  case ngpus: {                                         \
    if (force_1stage) {                                 \
      KL(ngpus, cross_device_reduce_1stage);            \
    } else if (force_2stage) {                          \
      KL(ngpus, cross_device_reduce_2stage);            \
    } else {                                            \
      if (world_size_ == 2) {                           \
        KL(ngpus, cross_device_reduce_1stage);          \
      } else if (fully_connected_) {                    \
        if ((world_size_ <= 4 && bytes < 512 * 1024) || \
            (world_size_ <= 8 && bytes < 256 * 1024)) { \
          KL(ngpus, cross_device_reduce_1stage);        \
        } else {                                        \
          KL(ngpus, cross_device_reduce_2stage);        \
        }                                               \
      }                                                 \
    }                                                   \
    break;                                              \
  }

    switch (world_size_) {
      REDUCE_CASE(2)
      REDUCE_CASE(4)
      REDUCE_CASE(6)
      REDUCE_CASE(8)
      default:
        throw std::runtime_error(
            "custom allreduce only supports num gpus in (2,4,6,8). Actual "
            "num "
            "gpus = " +
            std::to_string(world_size_));
    }
#undef REDUCE_CASE
#undef KL
  }

  ~CustomAllreduce() {
    for (auto [_, ptr] : ipc_handles_) {
      CUDACHECK(cudaIpcCloseMemHandle(ptr));
    }
  }
};

/**
 * To inspect PTX/SASS, copy paste this header file to compiler explorer and
 add a template instantiation:
 * template void vllm::CustomAllreduce::allreduce<half>(cudaStream_t, half *,
 half *, int, int, int);
*/
}  // namespace vllm