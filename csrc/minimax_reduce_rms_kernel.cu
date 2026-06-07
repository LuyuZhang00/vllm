
/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
 * 中文注释 - 文件功能概述：
 * ============================================================================
 * 本文件实现了 MiniMax 模型专用的 "AllReduce + RMSNorm" 融合 CUDA kernel。
 *
 * 核心功能：
 *   在张量并行 (TP) 场景下，将 RMSNorm 与跨 GPU 的 AllReduce 融合在同一个
 *   kernel 中执行，避免 AllReduce 和 RMSNorm 分开执行时的额外显存读写开销。
 *
 * 整体算法流程（以单个 token 为例）：
 *   1. 每个 rank 读取本地的输入向量 x_local，计算局部方差和 sum(x_local^2)。
 *   2. 通过 Lamport 协议（一种基于标志位的无锁 AllReduce 方式）在所有 rank
 *      之间交换局部方差和，得到全局方差和 sum_all = sum_over_ranks(sum(x_r^2))。
 *   3. 计算 RMSNorm 输出：output = x_local * rsqrt(sum_all / (hidden_dim * nranks) + eps) * gamma。
 *
 * 本文件包含两个 CUDA kernel：
 *   (A) minimax_reduce_rms_kernel_lamport:
 *       标量版本，每个线程处理 kElemsPerAccess<DType> 个元素（float4 加载），
 *       适用于单矩阵（如 Q 或 K 单独）的 RMSNorm。
 *   (B) minimax_reduce_qk_rms_kernel_lamport_float4:
 *       float4 优化版本，一次循环同时处理 4 行（4 个 token），且同时处理 Q 和 K
 *       两个矩阵，实现更好的内存合并访问和更高的吞吐量。
 *
 * Lamport 无锁通信协议：
 *   使用三缓冲区（triple buffering）和标志位实现跨 rank 的无锁数据交换。
 *   每个 rank 将局部结果写入所有其他 rank 的缓冲区中自己的槽位，
 *   然后轮询读取其他 rank 写入本 rank 缓冲区中的数据。
 *   通过 flag 的模 3 切换缓冲区偏移，实现流水线式的无锁通信。
 * ============================================================================
 */

#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <torch/cuda.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "cuda_compat.h"
#include "cuda_utils.h"
#include "core/registration.h"
#include "minimax_reduce_rms_kernel.h"

#include <algorithm>

#define FINAL_MASK 0xffffffff
// 中文注释：一个 warp 中的线程数，CUDA 中 warp 大小固定为 32
#define MINIMAX_REDUCE_RMS_WARP_SIZE 32

namespace vllm {
namespace tensorrt_llm {

/*
 * 中文注释 - LamportComm 结构体：
 * Lamport 无锁跨 rank 通信协议的设备端管理器。
 *
 * 工作原理：
 *   - 使用三缓冲区（triple buffering）避免读写冲突：flag_value % 3 决定当前
 *     写入的缓冲区偏移，(flag_value + 2) % 3 决定需要清除的旧缓冲区偏移。
 *   - 每个 rank 有 NRanks 个数据缓冲区指针 data_bufs[r]，指向 rank r 的共享
 *     内存区域，允许本 rank 直接写入其他 rank 的缓冲区。
 *   - counter_ptr 用于跟踪所有 block 是否完成当前迭代（barrier 同步）。
 *   - flag_ptr 控制缓冲区轮转，clear_ptr 记录需要清除的数据大小。
 *
 * 生命周期：
 *   构造 -> 各 block 完成计算并写入数据 -> update() 中 block 0 等待所有 block
 *   完成后翻转 flag 并清除旧缓冲区。
 */
template <int NRanks>
struct LamportComm {
  // 中文注释：构造函数，初始化 Lamport 通信所需的缓冲区指针。
  // workspace 布局：
  //   workspace[0..NRanks-1]         : 输入数据缓冲区指针（未使用）
  //   workspace[NRanks..2*NRanks-1]  : 输出数据缓冲区指针（未使用）
  //   workspace[2*NRanks..3*NRanks-1]: 每个 rank 的三缓冲区基地址
  //   workspace[NRanks*3]            : 共享计数器和标志位（counter, flag）
  //   workspace[NRanks*3+1]          : clear 指针和 comm_size
  __device__ __forceinline__ LamportComm(void** workspace, int rank) {
    counter_ptr = &reinterpret_cast<int*>(workspace[NRanks * 3])[0];
    flag_ptr = &reinterpret_cast<int*>(workspace[NRanks * 3])[2];
    clear_ptr = &reinterpret_cast<int64_t*>(workspace[NRanks * 3 + 1])[0];
    flag_value = *flag_ptr;
    auto comm_size = reinterpret_cast<int64_t*>(workspace[NRanks * 3 + 1])[1];
    clear_size = *clear_ptr;
    // 中文注释：data_offset 指向当前轮次的写入缓冲区，clear_offset 指向上一轮的
    // 缓冲区（需要清除以供下一轮使用），两者相差 2（模 3）保证不冲突。
    int data_offset = flag_value % 3;
    int clear_offset = (flag_value + 2) % 3;
    for (int r = 0; r < NRanks; ++r) {
      data_bufs[r] = reinterpret_cast<uint8_t*>(workspace[2 * NRanks + r]) +
                     data_offset * comm_size;
    }
    clear_buf = reinterpret_cast<uint8_t*>(workspace[2 * NRanks + rank]) +
                clear_offset * comm_size;
    __syncthreads();
    // 中文注释：每个 block 的 thread 0 原子递增计数器，用于后续 barrier 同步。
    if (threadIdx.x == 0) {
      atomicAdd(counter_ptr, 1);
    }
  }

  // 中文注释：update 由 block 0 的 thread 0 调用，执行三缓冲区轮转。
  // 流程：等待所有 block 完成（counter == gridDim.x）-> 翻转 flag ->
  // 更新 clear_size -> 重置 counter 为 0，为下一轮迭代做准备。
  __device__ __forceinline__ void update(int64_t new_clear_size) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
      while (*reinterpret_cast<int volatile*>(counter_ptr) != gridDim.x) {
      }
      *flag_ptr = (flag_value + 1) % 3;
      *clear_ptr = new_clear_size;
      *counter_ptr = 0;
    }
  }

  int* counter_ptr;
  int* flag_ptr;
  int64_t* clear_ptr;
  uint8_t* data_bufs[NRanks];
  uint8_t* clear_buf;
  int64_t clear_size;
  int flag_value;
};

/*
 * 中文注释 - 负零检测工具函数：
 * 在 Lamport 协议中，负零（-0.0）被用作"数据尚未就绪"的哨兵值。
 * 因为正常的浮点计算结果不会产生负零（-0.0 只在特定除法或初始化时出现），
 * 所以可以用它来区分"已写入的有效数据"和"未写入的缓冲区"。
 * is_neg_zero: 检测 float/float4 是否为负零。
 * get_neg_zero: 构造一个全负零的 float4 向量，用于清除缓冲区。
 */
__device__ __forceinline__ bool is_neg_zero(float v) {
  return *reinterpret_cast<uint32_t*>(&v) == 0x80000000;
}

__device__ __forceinline__ bool is_neg_zero(float4 v) {
  return is_neg_zero(v.x) || is_neg_zero(v.y) || is_neg_zero(v.z) ||
         is_neg_zero(v.w);
}

__device__ __forceinline__ float4 get_neg_zero() {
  float4 vec;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    reinterpret_cast<uint32_t*>(&vec)[i] = 0x80000000;
  }
  return vec;
}

/*
 * 中文注释 - RMSNorm 的 rsqrt 计算：
 * RMSNorm 的归一化因子为 rsqrt(mean(x^2) + eps) = rsqrt(sum(x^2)/Dim + eps)。
 * 这里 v 传入的是已经跨 rank 求和后的全局 sum(x^2)，Dim 是完整的 hidden_dim
 * （即 nranks * per_rank_dim），kInvDim = 1.0/Dim 用于将求和转为均值。
 * 模板参数 Dim 在编译期确定，避免运行时除法。
 */
template <int Dim>
__device__ __forceinline__ float rms_rsqrt(float& v, float eps) {
  constexpr float kInvDim = 1.0F / static_cast<float>(Dim);
  v = rsqrtf((v * kInvDim) + eps);
  return v;
}

// 中文注释：float4 向量化版本，同时对 4 个 token 的方差和计算 rsqrt。
template <int Dim>
__device__ __forceinline__ float4 rms_rsqrt(float4& v, float eps) {
  constexpr float kInvDim = 1.0F / static_cast<float>(Dim);
  v.x = rsqrtf((v.x * kInvDim) + eps);
  v.y = rsqrtf((v.y * kInvDim) + eps);
  v.z = rsqrtf((v.z * kInvDim) + eps);
  v.w = rsqrtf((v.w * kInvDim) + eps);
  return v;
}
/*
 * 中文注释 - volatile 全局内存加载：
 * 使用 PTX 内联汇编的 ld.volatile 指令从全局内存加载数据。
 * "volatile" 语义确保编译器不会缓存该加载结果，每次都从显存重新读取。
 * 这在 Lamport 轮询等待场景中至关重要：线程需要不断检查其他 rank 是否
 * 已写入数据，如果加载被缓存，可能永远看不到最新写入的值。
 * float4 版本一次加载 128 位（4 个 float），float 版本加载 32 位。
 */
__device__ __forceinline__ float4 ld_global_volatile(float4* addr) {
  float4 val;
  asm volatile("ld.volatile.global.v4.f32 {%0, %1, %2, %3}, [%4];"
               : "=f"(val.x), "=f"(val.y), "=f"(val.z), "=f"(val.w)
               : "l"(addr));
  return val;
}

__device__ __forceinline__ float ld_global_volatile(float* addr) {
  float val;
  asm volatile("ld.volatile.global.f32 %0, [%1];" : "=f"(val) : "l"(addr));
  return val;
}

/*
 * 中文注释 - warpReduceSumV2：标量 warp 内归约求和。
 * 在一个 warp 内通过 __shfl_xor_sync 进行 butterfly 归约，
 * 将 NUM 个通道的值分别在 32 个线程间求和。
 * 仅用于标量 kernel（非 float4 版本）。
 */
// Used by the scalar (non-float4) kernel only
template <typename T, int NUM>
__inline__ __device__ T warpReduceSumV2(T* val) {
#pragma unroll
  for (int i = 0; i < NUM; i++) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
      val[i] += __shfl_xor_sync(FINAL_MASK, val[i], mask, 32);
  }
  return (T)(0.0f);
}

/*
 * 中文注释 - blockReduceSumV2：block 级别归约求和。
 * 两阶段归约：
 *   1. 每个 warp 内先通过 warpReduceSumV2 归约到 warp 部分和。
 *   2. 每个 warp 的 lane 0 将部分和写入 shared memory。
 *   3. 第一个 warp 读取所有 warp 的部分和，再次 warp 内归约得到 block 总和。
 * 使用 shared[NUM][33] 布局，第二维 33 是为了避免 bank conflict。
 */
template <typename T, int NUM>
__inline__ __device__ T blockReduceSumV2(T* val) {
  static __shared__ T shared[NUM][33];
  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;

  warpReduceSumV2<T, NUM>(val);

  if (lane == 0) {
#pragma unroll
    for (int i = 0; i < NUM; i++) {
      shared[i][wid] = val[i];
    }
  }

  __syncthreads();

  bool is_mask = threadIdx.x < (blockDim.x / 32.f);
#pragma unroll
  for (int i = 0; i < NUM; i++) {
    val[i] = is_mask ? shared[i][lane] : (T)(0.0f);
  }
  warpReduceSumV2<T, NUM>(val);
  return (T)0.0f;
}

/*
 * 中文注释 - local_warp_reduce_sum_array：float4 版本的 warp 内归约。
 * 对 ArraySize 个 float 通道分别在 kNumThreads 个线程间进行 butterfly 归约。
 * kNumThreads 可以小于 32（如只用 NRanks 个线程做跨 rank 归约），
 * active_mask 控制参与归约的线程掩码，未参与的线程自动贡献 0。
 */
// for float4 version
template <uint32_t kNumThreads, typename T, int ArraySize = 4>
__device__ __forceinline__ void local_warp_reduce_sum_array(
    T* value_ptr, uint32_t active_mask = 0xffffffffu) {
  static_assert(kNumThreads >= 1 &&
                kNumThreads <= MINIMAX_REDUCE_RMS_WARP_SIZE);
#pragma unroll
  for (int i = 0; i < ArraySize; ++i) {
#pragma unroll
    for (int mask = kNumThreads / 2; mask > 0; mask >>= 1) {
      value_ptr[i] += __shfl_xor_sync(active_mask, value_ptr[i], mask,
                                      MINIMAX_REDUCE_RMS_WARP_SIZE);
    }
  }
}

// 中文注释：计算大于等于 val 的最小 2 的幂次，用于 warp 归约时确定有效线程数。
constexpr int next_pow2(int val) {
  int result = 1;
  while (result < val) {
    result <<= 1;
  }
  return result;
}

// ---------------------------------------------------------------------------

/*
 * 中文注释 - IndexHelper 类：
 * 负责计算当前线程在全局数据中的索引。根据 CUDA 架构版本选择不同的索引策略：
 *   - SM 90+（Hopper）：使用 cooperative_groups 的 cluster 索引，一个 cluster
 *     处理一个 token，cluster 内线程并行处理该 token 的不同元素。
 *   - SM < 90：使用传统的 blockIdx.x 和 threadIdx.x 映射。
 *
 * 关键成员变量：
 *   token_id          : 当前线程负责处理的 token 编号
 *   access_id_in_token: token 内的线程偏移（即负责第几个 float4 访问）
 *   access_id         : 全局访问索引 = token_id * hidden_dim/elems_per_access + access_id_in_token
 *   access_stride     : 一次循环迭代后跳到下一个 token 的步长
 *   tot_access        : 总共需要访问的 float4 数量
 */
template <typename DType>
class IndexHelper {
 public:
  __device__ __forceinline__ IndexHelper(MiniMaxReduceRMSParams const& params) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    namespace cg = cooperative_groups;
    cg::cluster_group cluster = cg::this_cluster();
    cg::grid_group grid = cg::this_grid();
    token_id = grid.cluster_rank();
    access_id_in_token = cluster.thread_rank();
    token_stride = grid.num_clusters();
#else
    token_id = blockIdx.x;
    access_id_in_token = threadIdx.x;
    token_stride = gridDim.x;
#endif
    access_id = token_id * params.hidden_dim / kElemsPerAccess<DType> +
                access_id_in_token;
    access_stride = token_stride * params.hidden_dim / kElemsPerAccess<DType>;
    tot_access = params.size_q / kElemsPerAccess<DType>;
  }

  int token_id;
  int access_id_in_token;
  int token_stride;
  int access_id;
  int access_stride;
  int tot_access;
};

/**
* this kernel is used to for minimax attention module
* input tensor [total_tokens, hidden_dim / tp_size], fp32
* rms weight [hidden_dim / tp_size], bf16
step 1: reduce from single rank to get the variance sum (reduce(input^2,
dim=-1)) step 2: reduce from all ranks to get the variance sum
(all_reduce(variance_sum)) step 3: calculate the rms norm (input *
rsqrt(variance + eps)) in this case, max hidden_dim is 6144 (float data), for
each token, we only need 6144 / 4 / tp_size = (1536 / tp_size) threads so we can
assume cluster size is 1 (tp_size >= 2)
 */
/*
 * 中文注释 - minimax_reduce_rms_kernel_lamport（标量版本 kernel）：
 * =========================================================================
 * 这是 AllReduce + RMSNorm 融合的主 kernel（标量处理，一个循环处理一个 token）。
 *
 * 每个 block 处理一个 token，block 内的线程并行处理该 token 的 hidden_dim 个元素。
 *
 * 执行流程：
 *   Step 1 - 局部方差计算：
 *     每个线程用 float4 加载 kElemsPerAccess 个元素，计算每个元素的平方并累加，
 *     然后通过 blockReduceSumV2 在整个 block 内求和，得到本 rank 对该 token 的
 *     局部方差和 sum(x_local^2)。
 *
 *   Step 2 - 跨 rank AllReduce（Lamport 协议）：
 *     每个 block 的 thread 0 将局部方差和写入所有其他 rank 的通信缓冲区中
 *     自己的槽位（push），然后轮询读取所有其他 rank 写入本 rank 缓冲区的
 *     局部方差和（pull），通过 volatile 加载确保读到最新值。
 *     所有 rank 的局部方差和相加得到全局方差和。
 *
 *   Step 3 - RMSNorm 归一化：
 *     用全局方差和计算 rsqrt(sum_all / (hidden_dim * NRanks) + eps)，
 *     然后将原始输入值乘以该归一化因子和可学习的 gamma 权重，写回输出。
 *
 *   Step 4 - 清理：
 *     将上一轮使用的缓冲区用负零清除，为下一轮迭代做准备。
 *     调用 comm.update() 翻转三缓冲区的标志位。
 * =========================================================================
 */
template <typename DType, int NRanks>
__global__ void __launch_bounds__(1024)
    minimax_reduce_rms_kernel_lamport(MiniMaxReduceRMSParams params) {
  // 中文注释：通过 IndexHelper 计算当前线程的索引信息
  IndexHelper<DType> index_helper(params);
  int token_id = index_helper.token_id;
  int access_id_in_token = index_helper.access_id_in_token;
  int token_stride = index_helper.token_stride;
  int access_id = index_helper.access_id;
  int access_stride = index_helper.access_stride;
  int tot_access = index_helper.tot_access;
  int tot_tokens = params.size_q / params.hidden_dim;
  // 中文注释：clear_vec 是全负零的 float4，用于清除上一轮的通信缓冲区
  float4 clear_vec = get_neg_zero();

  // 中文注释：初始化 Lamport 通信上下文，建立跨 rank 的缓冲区指针
  LamportComm<NRanks> comm(params.workspace, params.rank);
  int clear_access = comm.clear_size / kElemsPerAccess<DType>;
  // 中文注释：SM 90+ (Hopper) 使用 grid dependency control 等待前置 kernel 完成
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
#endif
  // 中文注释：主循环 - 遍历当前线程负责的所有 token（可能有多个 token 需要处理）
  for (int idx = access_id; idx < tot_access;
       idx += access_stride, token_id += token_stride) {
    // ========== Step 1: 计算本 rank 的局部方差和 ==========
    alignas(16) DType vals[kElemsPerAccess<DType>];
    float sum_variance = 0.F;
    // 中文注释：从全局内存加载当前 token 对应的 kElemsPerAccess 个元素
    *reinterpret_cast<float4*>(vals) =
        reinterpret_cast<float4*>(params.allreduce_in)[idx];
    // 中文注释：计算每个元素的平方并累加，得到线程级别的部分方差和
#pragma unroll
    for (int i = 0; i < kElemsPerAccess<DType>; ++i) {
      sum_variance += static_cast<float>(vals[i]) * static_cast<float>(vals[i]);
    }
    // 中文注释：block 内归约求和，得到该 token 在本 rank 上的完整方差和
    blockReduceSumV2<float, 1>(&sum_variance);
    // 中文注释：清除负零哨兵值（归约过程中可能意外产生）
    if (is_neg_zero(sum_variance)) {
      sum_variance = 0.F;
    }
    // ========== Step 2: 跨 rank AllReduce（Lamport push） ==========
    // 中文注释：thread 0 将本 rank 的局部方差和写入所有其他 rank 的缓冲区
    // 写入位置：comm.data_bufs[r] 的 [rank * tot_tokens + token_id] 槽位
    if (threadIdx.x == 0) {
      for (int r = 0; r < NRanks; ++r) {
        reinterpret_cast<float*>(
            comm.data_bufs[r])[(params.rank * tot_tokens) + token_id] =
            (sum_variance);
      }
    }

    // ========== Step 2 续: 跨 rank AllReduce（Lamport pull + 轮询等待） ==========
    // 中文注释：轮询等待所有 rank 将它们的局部方差和写入本 rank 的缓冲区。
    // 使用 volatile 加载确保每次都从显存读取最新值。
    // is_neg_zero 检测：负零表示数据尚未就绪，非负零表示有效数据已写入。
    bool done = false;
    float vars_all_ranks[NRanks];
    while (!done) {
      done = true;
#pragma unroll
      for (int r = 0; r < NRanks; ++r) {
        vars_all_ranks[r] = ld_global_volatile(&reinterpret_cast<float*>(
            comm.data_bufs[params.rank])[(r * tot_tokens) + token_id]);
        done &= !is_neg_zero(vars_all_ranks[r]);
      }
    }
    // 中文注释：汇总所有 rank 的局部方差和，得到全局方差和
    sum_variance = 0.F;
#pragma unroll
    for (int r = 0; r < NRanks; ++r) {
      sum_variance += vars_all_ranks[r];
    }

    // ========== Step 3: RMSNorm 归一化 ==========
    // 中文注释：加载 RMSNorm 的 gamma 权重（每个元素对应一个可学习参数）
    DType norm_weight[kElemsPerAccess<DType>];
    *reinterpret_cast<typename ElemsPerAccess<DType>::vec_type*>(norm_weight) =
        reinterpret_cast<typename ElemsPerAccess<DType>::vec_type*>(
            params.rms_gamma)[access_id_in_token];

    // 中文注释：计算最终输出 = x * rsqrt(sum_all / (hidden_dim * NRanks) + eps) * gamma
    // 其中 hidden_dim * NRanks 是完整的（未切分的）隐藏维度，
    // rsqrt 中的除法将方差和转为均值后再加 eps 取倒数平方根。
#pragma unroll
    for (int i = 0; i < kElemsPerAccess<DType>; ++i) {
      vals[i] = static_cast<DType>(
          static_cast<float>(vals[i]) *
          rsqrtf(
              (sum_variance / static_cast<float>(params.hidden_dim) / NRanks) +
              params.rms_eps) *
          static_cast<float>(norm_weight[i]));
    }

    // 中文注释：将归一化结果写回全局输出
    reinterpret_cast<float4*>(params.rms_norm_out)[idx] =
        *reinterpret_cast<float4*>(vals);
  }
  // ========== Step 4: 清理上一轮的通信缓冲区 ==========
  // 中文注释：用负零填充上一轮使用的缓冲区，为下一轮 Lamport 通信做准备
  for (int idx = access_id; idx < clear_access; idx += access_stride) {
    reinterpret_cast<float4*>(comm.clear_buf)[idx] = clear_vec;
  }
  // 中文注释：翻转三缓冲区标志位，推进到下一轮
  comm.update(params.size_q * NRanks);
  // 中文注释：SM 90+ 通知依赖的后续 kernel 可以开始执行
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

/**
 * Float4 variant: process 4 rows at once, allreduce variance sums as float4 for
 * better memory coalescing. sum_variance is always float; applies to all DTypes
 * (half, bf16, float). When tot_tokens % 4 != 0, the last group pads rows with
 * zeros; padded rows are not written to rms_norm_out. IsQK: when true, process
 * Q+K in one loop with doubled comm buffer; when false, single-matrix (Q only).
 */
/*
 * 中文注释 - minimax_reduce_qk_rms_kernel_lamport_float4（float4 优化版本）：
 * =========================================================================
 * 这是 AllReduce + RMSNorm 融合的优化版本，同时处理 Q 和 K 两个矩阵。
 *
 * 与标量版本的关键区别：
 *   1. 每次循环处理 4 行（4 个 token）而不是 1 行，提升数据复用率。
 *   2. 使用 float4 向量化进行 AllReduce 通信，提升内存带宽利用率。
 *   3. 同一个 kernel 内同时处理 Q 和 K 两个矩阵，减少 kernel 启动开销。
 *
 * Block 内线程分配：
 *   前 NumWarpQ 个 warp 负责 Q 矩阵的处理，
 *   后 NumWarpK 个 warp 负责 K 矩阵的处理。
 *   Q 和 K 各自独立计算方差和、独立做跨 rank 归约，最后独立写回。
 *
 * 执行流程（以 Q 为例，K 同理）：
 *   1. 每组 4 行的线程分别加载 float4 数据，计算每行的平方和。
 *   2. warp 内 butterfly 归约得到每个 warp 对 4 行的方差和。
 *   3. shared memory 二次归约得到 block 级别的方差和。
 *   4. NRanks 个 leader 线程通过 Lamport 协议做跨 rank 归约。
 *   5. 计算 rsqrt，广播到所有线程，完成 RMSNorm 归一化并写回。
 * =========================================================================
 */
template <typename DType, int NRanks, int OriginQDim, int OriginKDim>
__global__ void __launch_bounds__(1024)
    minimax_reduce_qk_rms_kernel_lamport_float4(MiniMaxReduceRMSParams params) {
  // 中文注释：编译期确定每个 rank 上 Q/K 的维度大小
  // Compile-time per-rank dimensions
  constexpr int RankQDim = OriginQDim / NRanks;
  constexpr int RankKDim = OriginKDim / NRanks;
  // 中文注释：覆盖一行 Q/K 所需的 float4 访问次数（即每行需要多少个线程）
  // Threads needed to cover one row of Q / K with float4 accesses
  constexpr int ThreadsPerRowQ = RankQDim / kElemsPerAccess<DType>;
  constexpr int ThreadsPerRowK = RankKDim / kElemsPerAccess<DType>;
  // 中文注释：覆盖一行 Q/K 需要的 warp 数量（向上取整到 warp 对齐）
  // Number of warps dedicated to Q / K
  constexpr int NumWarpQ = (ThreadsPerRowQ + MINIMAX_REDUCE_RMS_WARP_SIZE - 1) /
                           MINIMAX_REDUCE_RMS_WARP_SIZE;
  constexpr int NumWarpK = (ThreadsPerRowK + MINIMAX_REDUCE_RMS_WARP_SIZE - 1) /
                           MINIMAX_REDUCE_RMS_WARP_SIZE;

  int tot_tokens = params.size_q / RankQDim;
  // 中文注释：总 token 数除以 4 得到"组数"，每组处理 4 个 token；最后不足 4 个的用零填充
  int tot_groups = (tot_tokens + 3) / 4;  // ceiling; last group may be partial

  // Memory strides for strided qkv tensors (elements -> float4-access units)
  int access_stride_q = (params.stride_q > 0 ? params.stride_q : RankQDim) /
                        kElemsPerAccess<DType>;
  int access_stride_k = (params.stride_k > 0 ? params.stride_k : RankKDim) /
                        kElemsPerAccess<DType>;
  // Output strides: default to contiguous (hidden_dim / hidden_dim_k)
  int access_stride_q_out =
      (params.stride_q_out > 0 ? params.stride_q_out : params.hidden_dim) /
      kElemsPerAccess<DType>;
  int access_stride_k_out =
      (params.stride_k_out > 0 ? params.stride_k_out : params.hidden_dim_k) /
      kElemsPerAccess<DType>;

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  namespace cg = cooperative_groups;
  cg::cluster_group cluster = cg::this_cluster();
  cg::grid_group grid = cg::this_grid();
  int group_id = grid.cluster_rank();
  int access_id_in_token = cluster.thread_rank();
  int group_stride = grid.num_clusters();
#else
  int group_id = blockIdx.x;
  int access_id_in_token = threadIdx.x;
  int group_stride = gridDim.x;
#endif

  bool is_q = (access_id_in_token < NumWarpQ * MINIMAX_REDUCE_RMS_WARP_SIZE);
  int k_thread_idx =
      access_id_in_token - (NumWarpQ * MINIMAX_REDUCE_RMS_WARP_SIZE);
  bool is_valid_q = (access_id_in_token < ThreadsPerRowQ);
  bool is_valid_k = (k_thread_idx >= 0 && k_thread_idx < ThreadsPerRowK);
  float4 clear_vec = get_neg_zero();

  // Shared memory for two-level block reduction and scale broadcast
  __shared__ float block_reduce_sum[4][MINIMAX_REDUCE_RMS_WARP_SIZE + 1];
  __shared__ float global_scale_q[4];
  __shared__ float global_scale_k[4];

  LamportComm<NRanks> comm(params.workspace, params.rank);

  DType norm_weight[kElemsPerAccess<DType>]{};
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
#endif
  if (is_q) {
    if (is_valid_q) {
      *reinterpret_cast<typename ElemsPerAccess<DType>::vec_type*>(
          norm_weight) =
          reinterpret_cast<typename ElemsPerAccess<DType>::vec_type const*>(
              params.rms_gamma)[access_id_in_token];
    }
  } else {
    if (is_valid_k) {
      *reinterpret_cast<typename ElemsPerAccess<DType>::vec_type*>(
          norm_weight) =
          reinterpret_cast<typename ElemsPerAccess<DType>::vec_type const*>(
              params.rms_gamma_k)[k_thread_idx];
    }
  }

  // Main loop: process one group of 4 tokens per iteration.
  for (int g = group_id; g < tot_groups; g += group_stride) {
    alignas(16) DType vals[4][kElemsPerAccess<DType>]{};
    float warp_sum_variance[4]{0.F, 0.F, 0.F, 0.F};

    if (is_q) {
#pragma unroll
      for (int row = 0; row < 4; ++row) {
        int token_r = g * 4 + row;
        if (token_r >= tot_tokens || !is_valid_q) {
          continue;
        }
        int idx_r = token_r * access_stride_q + access_id_in_token;
        *reinterpret_cast<float4*>(&vals[row][0]) =
            reinterpret_cast<float4 const*>(params.allreduce_in)[idx_r];
#pragma unroll
        for (int i = 0; i < kElemsPerAccess<DType>; ++i) {
          float x = static_cast<float>(vals[row][i]);
          warp_sum_variance[row] += x * x;
        }
      }
    } else {
#pragma unroll
      for (int row = 0; row < 4; ++row) {
        int token_r = g * 4 + row;
        if (token_r >= tot_tokens || !is_valid_k) {
          continue;
        }
        int idx_r = token_r * access_stride_k + k_thread_idx;
        *reinterpret_cast<float4*>(&vals[row][0]) =
            reinterpret_cast<float4 const*>(params.allreduce_in_k)[idx_r];
#pragma unroll
        for (int i = 0; i < kElemsPerAccess<DType>; ++i) {
          float x = static_cast<float>(vals[row][i]);
          warp_sum_variance[row] += x * x;
        }
      }
    }

    local_warp_reduce_sum_array<MINIMAX_REDUCE_RMS_WARP_SIZE, float, 4>(
        warp_sum_variance);
    // Warp lane 0 writes its warp's partial sum to shared memory
    int lane = threadIdx.x & (MINIMAX_REDUCE_RMS_WARP_SIZE - 1);
    if (lane == 0) {
#pragma unroll
      for (int t = 0; t < 4; ++t) {
        block_reduce_sum[t][threadIdx.x / MINIMAX_REDUCE_RMS_WARP_SIZE] =
            warp_sum_variance[t];
      }
    }
    __syncthreads();

    int tid = threadIdx.x;

    if (tid < MINIMAX_REDUCE_RMS_WARP_SIZE) {
      constexpr int kNumWarpQPow2 =
          (next_pow2(NumWarpQ) > NRanks) ? next_pow2(NumWarpQ) : NRanks;
      float local_sum[4];
#pragma unroll
      for (int t = 0; t < 4; ++t) {
        local_sum[t] = (tid < NumWarpQ) ? block_reduce_sum[t][tid] : 0.F;
      }
      // After this, all kNumWarpQPow2 lanes (including tid 0..NRanks-1) have
      // the total Q sum-of-squares for all 4 tokens.
      local_warp_reduce_sum_array<kNumWarpQPow2, float, 4>(local_sum);

      if (tid < NRanks) {
#pragma unroll
        for (int t = 0; t < 4; ++t) {
          if (is_neg_zero(local_sum[t])) {
            local_sum[t] = 0.F;
          }
        }
        // Parallel push: thread tid writes this rank's Q sum to rank tid's buf
        reinterpret_cast<float4*>(
            comm.data_bufs[tid])[(params.rank * tot_groups * 2) + (2 * g)] =
            *reinterpret_cast<float4*>(local_sum);

        // Parallel pull: thread tid reads rank tid's contribution from
        // this rank's (params.rank's) buffer
        bool done = false;
        float4 var_all_ranks;
        while (!done) {
          done = true;
          var_all_ranks = ld_global_volatile(&reinterpret_cast<float4*>(
              comm.data_bufs[params.rank])[(tid * tot_groups * 2) + (2 * g)]);
          done &= !is_neg_zero(var_all_ranks);
        }

        // Warp-level allreduce: each of the NRanks threads holds one rank's
        // partial sum; after this all NRanks threads have the global total.
        constexpr uint32_t kQActiveMask = (1u << NRanks) - 1u;
        local_warp_reduce_sum_array<NRanks, float, 4>(
            reinterpret_cast<float*>(&var_all_ranks), kQActiveMask);

        // Thread 0 computes rsqrt with compile-time Dim and writes to smem
        if (tid == 0) {
          *reinterpret_cast<float4*>(global_scale_q) =
              rms_rsqrt<OriginQDim>(var_all_ranks, params.rms_eps);
        }
      }
    } else if (tid >= MINIMAX_REDUCE_RMS_WARP_SIZE * NumWarpQ &&
               tid < MINIMAX_REDUCE_RMS_WARP_SIZE * (NumWarpQ + 1)) {
      // --- K leader warp ---
      constexpr int kNumWarpKPow2 =
          (next_pow2(NumWarpK) > NRanks) ? next_pow2(NumWarpK) : NRanks;
      float local_sum[4];
#pragma unroll
      for (int t = 0; t < 4; ++t) {
        local_sum[t] = (k_thread_idx < NumWarpK)
                           ? block_reduce_sum[t][NumWarpQ + k_thread_idx]
                           : 0.F;
      }
      local_warp_reduce_sum_array<kNumWarpKPow2, float, 4>(local_sum);

      if (k_thread_idx < NRanks) {
#pragma unroll
        for (int t = 0; t < 4; ++t) {
          if (is_neg_zero(local_sum[t])) {
            local_sum[t] = 0.F;
          }
        }
        reinterpret_cast<float4*>(
            comm.data_bufs[k_thread_idx])[(params.rank * tot_groups * 2) +
                                          (2 * g + 1)] =
            *reinterpret_cast<float4*>(local_sum);

        bool done = false;
        float4 var_all_ranks;
        while (!done) {
          done = true;
          var_all_ranks = ld_global_volatile(&reinterpret_cast<float4*>(
              comm.data_bufs[params.rank])[(k_thread_idx * tot_groups * 2) +
                                           (2 * g + 1)]);
          done &= !is_neg_zero(var_all_ranks);
        }

        constexpr uint32_t kKActiveMask = (1u << NRanks) - 1u;
        local_warp_reduce_sum_array<NRanks, float, 4>(
            reinterpret_cast<float*>(&var_all_ranks), kKActiveMask);

        if (k_thread_idx == 0) {
          *reinterpret_cast<float4*>(global_scale_k) =
              rms_rsqrt<OriginKDim>(var_all_ranks, params.rms_eps);
        }
      }
    }
    __syncthreads();

    if (is_q) {
#pragma unroll
      for (int t = 0; t < 4; ++t) {
        warp_sum_variance[t] = global_scale_q[t];
      }
#pragma unroll
      for (int r = 0; r < 4; ++r) {
#pragma unroll
        for (int i = 0; i < kElemsPerAccess<DType>; ++i) {
          vals[r][i] = static_cast<DType>(static_cast<float>(vals[r][i]) *
                                          warp_sum_variance[r] *
                                          static_cast<float>(norm_weight[i]));
        }
        int token_r = g * 4 + r;
        if (token_r >= tot_tokens || !is_valid_q) {
          continue;
        }
        int idx_out = token_r * access_stride_q_out + access_id_in_token;
        reinterpret_cast<float4*>(params.rms_norm_out)[idx_out] =
            *reinterpret_cast<float4*>(&vals[r][0]);
      }
    } else {
#pragma unroll
      for (int t = 0; t < 4; ++t) {
        warp_sum_variance[t] = global_scale_k[t];
      }
#pragma unroll
      for (int r = 0; r < 4; ++r) {
#pragma unroll
        for (int i = 0; i < kElemsPerAccess<DType>; ++i) {
          vals[r][i] = static_cast<DType>(static_cast<float>(vals[r][i]) *
                                          warp_sum_variance[r] *
                                          static_cast<float>(norm_weight[i]));
        }
        int token_r = g * 4 + r;
        if (token_r >= tot_tokens || !is_valid_k) {
          continue;
        }
        int idx_out = token_r * access_stride_k_out + k_thread_idx;
        reinterpret_cast<float4*>(params.rms_norm_out_k)[idx_out] =
            *reinterpret_cast<float4*>(&vals[r][0]);
      }
    }
  }  // end group loop
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif

  int clear_access = static_cast<int>(comm.clear_size / kElemsPerAccess<DType>);
  int clear_stride = group_stride * blockDim.x;
  for (int idx = group_id * blockDim.x + threadIdx.x; idx < clear_access;
       idx += clear_stride) {
    reinterpret_cast<float4*>(comm.clear_buf)[idx] = clear_vec;
  }

  comm.update(static_cast<int64_t>(2) * tot_groups * kElemsPerAccess<DType> *
              NRanks);
}

int get_sm_count() {
  static int sm_count = 0;
  if (sm_count == 0) {
    int device_id;
    CUDA_CHECK(cudaGetDevice(&device_id));
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, device_id);
    sm_count = device_prop.multiProcessorCount;
  }
  return sm_count;
}

inline int getSMVersion(bool queryRealSmArch = false) {
  int device{-1};
  CUDA_CHECK(cudaGetDevice(&device));
  int sm_major = 0;
  int sm_minor = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&sm_major,
                                    cudaDevAttrComputeCapabilityMajor, device));
  CUDA_CHECK(cudaDeviceGetAttribute(&sm_minor,
                                    cudaDevAttrComputeCapabilityMinor, device));
  int sm = sm_major * 10 + sm_minor;
  if (sm == 121 && !queryRealSmArch) {
    return 120;
  }
  return sm;
}

template <typename KernelFunc>
int get_max_active_blocks(KernelFunc kernel, int block_size,
                          int dynamic_smem = 0) {
  int max_active = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &max_active, kernel, block_size, dynamic_smem));
  return std::max(max_active, 1);
}

template <typename DType, int NRanks>
void minimax_reduce_rms_kernel_launcher(MiniMaxReduceRMSParams const& params) {
  static int SM = getSMVersion();
  int token_num = params.size_q / params.hidden_dim;
  int sm_count = get_sm_count();
  int cluster_size = 1;
  int cluster_num = token_num;
  int threads_per_token = params.hidden_dim / kElemsPerAccess<DType>;
  int block_size = threads_per_token;

  int max_blocks_per_sm = get_max_active_blocks(
      minimax_reduce_rms_kernel_lamport<DType, NRanks>, block_size);
  int max_grid = max_blocks_per_sm * sm_count;

  int grid_size =
      (std::min(max_grid, cluster_num * cluster_size) / cluster_size) *
      cluster_size;

  cudaLaunchConfig_t cfg;
  cfg.gridDim = grid_size;
  cfg.blockDim = block_size;
  cfg.dynamicSmemBytes = 0;
  cfg.stream = params.stream;

  cudaLaunchAttribute attribute[2];
  attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attribute[0].val.programmaticStreamSerializationAllowed = 1;
  attribute[1].id = cudaLaunchAttributeClusterDimension;
  attribute[1].val.clusterDim.x = cluster_size;
  attribute[1].val.clusterDim.y = 1;
  attribute[1].val.clusterDim.z = 1;
  cfg.attrs = attribute;
  cfg.numAttrs = SM >= 90 ? 2 : 0;

  CUDA_CHECK(cudaLaunchKernelEx(
      &cfg, minimax_reduce_rms_kernel_lamport<DType, NRanks>, params));
}

template <typename DType, int NRanks, int OriginQDim, int OriginKDim>
void minimax_reduce_rms_kernel_launcher_float4(
    MiniMaxReduceRMSParams const& params) {
  TORCH_CHECK(params.size_q % params.hidden_dim == 0);
  TORCH_CHECK(params.hidden_dim % kElemsPerAccess<DType> == 0);
  if (params.stride_q > 0) {
    TORCH_CHECK(params.stride_q % kElemsPerAccess<DType> == 0);
  }
  TORCH_CHECK(params.allreduce_in_k != nullptr,
              "float4 QK kernel requires K input");
  TORCH_CHECK(params.hidden_dim >= params.hidden_dim_k);
  TORCH_CHECK(params.size_k % params.hidden_dim_k == 0);
  TORCH_CHECK(params.hidden_dim_k % kElemsPerAccess<DType> == 0);
  TORCH_CHECK(params.size_q / params.hidden_dim ==
              params.size_k / params.hidden_dim_k);
  if (params.stride_k > 0) {
    TORCH_CHECK(params.stride_k % kElemsPerAccess<DType> == 0);
  }

  int token_num = params.size_q / params.hidden_dim;
  int tot_groups = (token_num + 3) / 4;
  if (tot_groups == 0) {
    return;
  }

  static int SM = getSMVersion();
  int sm_count = get_sm_count();
  int cluster_size = 1;
  int cluster_num = tot_groups;

  int access_per_row_q = params.hidden_dim / kElemsPerAccess<DType>;
  int access_per_row_k = params.hidden_dim_k / kElemsPerAccess<DType>;

  // Round each section up to a warp boundary
  auto divUp = [](int a, int b) { return (a + b - 1) / b * b; };
  int block_size = divUp(access_per_row_q, MINIMAX_REDUCE_RMS_WARP_SIZE) +
                   divUp(access_per_row_k, MINIMAX_REDUCE_RMS_WARP_SIZE);

  auto kfn =
      minimax_reduce_qk_rms_kernel_lamport_float4<DType, NRanks, OriginQDim,
                                                  OriginKDim>;

  int max_blocks_per_sm = get_max_active_blocks(kfn, block_size);
  int max_grid = max_blocks_per_sm * sm_count;
  int grid_size =
      (std::min(max_grid, cluster_num * cluster_size) / cluster_size) *
      cluster_size;

  cudaLaunchConfig_t cfg;
  cfg.gridDim = grid_size;
  cfg.blockDim = block_size;
  cfg.dynamicSmemBytes = 0;
  cfg.stream = params.stream;

  cudaLaunchAttribute attribute[2];
  attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attribute[0].val.programmaticStreamSerializationAllowed = 1;
  attribute[1].id = cudaLaunchAttributeClusterDimension;
  attribute[1].val.clusterDim.x = cluster_size;
  attribute[1].val.clusterDim.y = 1;
  attribute[1].val.clusterDim.z = 1;
  cfg.attrs = attribute;
  cfg.numAttrs = SM >= 90 ? 2 : 0;

  CUDA_CHECK(cudaLaunchKernelEx(&cfg, kfn, params));
}

template <int NRanks>
void dispatch_dtype(MiniMaxReduceRMSParams const& params) {
  // Use the optimized QK float4 kernel when:
  //  - K input is present, AND
  //  - the full (NRanks * per-rank) dimensions match the MiniMax M2 shape.
  // Otherwise fall back to the scalar kernel.
  bool use_float4 = (params.allreduce_in_k != nullptr) &&
                    (params.hidden_dim * params.nranks == 6144) &&
                    (params.hidden_dim_k * params.nranks == 1024);

  if (params.dtype == at::ScalarType::Half) {
    if (use_float4) {
      minimax_reduce_rms_kernel_launcher_float4<half, NRanks, 6144, 1024>(
          params);
    } else {
      minimax_reduce_rms_kernel_launcher<half, NRanks>(params);
    }
  } else if (params.dtype == at::ScalarType::BFloat16) {
    if (use_float4) {
      minimax_reduce_rms_kernel_launcher_float4<__nv_bfloat16, NRanks, 6144,
                                                1024>(params);
    } else {
      minimax_reduce_rms_kernel_launcher<__nv_bfloat16, NRanks>(params);
    }
  } else if (params.dtype == at::ScalarType::Float) {
    if (use_float4) {
      minimax_reduce_rms_kernel_launcher_float4<float, NRanks, 6144, 1024>(
          params);
    } else {
      minimax_reduce_rms_kernel_launcher<float, NRanks>(params);
    }
  } else {
    TORCH_CHECK(false, "Unsupported data type for minimax_reduce_rms_op");
  }
}

void minimax_reduce_rms_op(MiniMaxReduceRMSParams const& params) {
  if (params.nranks == 2) {
    dispatch_dtype<2>(params);
  } else if (params.nranks == 4) {
    dispatch_dtype<4>(params);
  } else if (params.nranks == 8) {
    dispatch_dtype<8>(params);
  } else if (params.nranks == 16) {
    dispatch_dtype<16>(params);
  } else {
    TORCH_CHECK(false, "minimax_reduce_rms_op: unsupported ranks number!");
  }
}
}  // namespace tensorrt_llm
}  // namespace vllm

torch::Tensor minimax_allreduce_rms(torch::Tensor const& input,
                                    torch::Tensor const& norm_weight,
                                    torch::Tensor workspace, int64_t const rank,
                                    int64_t const nranks, double const eps) {
  auto allreduce_params = vllm::tensorrt_llm::MiniMaxReduceRMSParams();

  allreduce_params.nranks = static_cast<int>(nranks);
  allreduce_params.rank = static_cast<int>(rank);
  allreduce_params.dtype = input.scalar_type();
  allreduce_params.size_q = static_cast<int>(input.numel());
  allreduce_params.hidden_dim = static_cast<int>(input.size(-1));
  allreduce_params.stride_q = allreduce_params.hidden_dim;
  allreduce_params.workspace =
      reinterpret_cast<void**>(workspace.mutable_data_ptr());
  allreduce_params.allreduce_in = input.data_ptr();
  allreduce_params.rms_gamma = norm_weight.data_ptr();
  allreduce_params.rms_eps = static_cast<float>(eps);
  allreduce_params.stream = at::cuda::getCurrentCUDAStream(input.get_device());

  torch::Tensor rms_norm_out = torch::empty_like(input);
  allreduce_params.rms_norm_out = rms_norm_out.mutable_data_ptr();

  vllm::tensorrt_llm::minimax_reduce_rms_op(allreduce_params);

  return rms_norm_out;
}

std::tuple<torch::Tensor, torch::Tensor> minimax_allreduce_rms_qk(
    torch::Tensor qkv, torch::Tensor const& norm_weight_q,
    torch::Tensor const& norm_weight_k, torch::Tensor workspace,
    int64_t const q_size, int64_t const kv_size, int64_t const rank,
    int64_t const nranks, double const eps) {
  TORCH_CHECK(qkv.dim() == 2, "minimax_allreduce_rms_qk: qkv must be 2D");
  TORCH_CHECK(qkv.is_contiguous(),
              "minimax_allreduce_rms_qk: qkv must be contiguous");
  int64_t qkv_dim = qkv.size(-1);
  TORCH_CHECK(qkv_dim == q_size + 2 * kv_size,
              "minimax_allreduce_rms_qk: qkv last dim must equal "
              "q_size + 2 * kv_size");
  TORCH_CHECK(rank < nranks,
              "minimax_allreduce_rms_qk: rank must be less than nranks");

  int64_t num_tokens = qkv.size(0);
  int elem_bytes = qkv.element_size();

  torch::Tensor q_out = torch::empty({num_tokens, q_size}, qkv.options());
  torch::Tensor k_out = torch::empty({num_tokens, kv_size}, qkv.options());

  auto params = vllm::tensorrt_llm::MiniMaxReduceRMSParams();
  params.nranks = static_cast<int>(nranks);
  params.rank = static_cast<int>(rank);
  params.dtype = qkv.scalar_type();
  params.size_q = static_cast<int>(num_tokens * q_size);
  params.hidden_dim = static_cast<int>(q_size);
  params.size_k = static_cast<int>(num_tokens * kv_size);
  params.hidden_dim_k = static_cast<int>(kv_size);
  params.stride_q = static_cast<int>(qkv_dim);
  params.stride_k = static_cast<int>(qkv_dim);
  params.stride_q_out = 0;  // q_out is contiguous; kernel uses hidden_dim
  params.stride_k_out = 0;  // k_out is contiguous; kernel uses hidden_dim_k
  params.workspace = reinterpret_cast<void**>(workspace.mutable_data_ptr());

  uint8_t* base = static_cast<uint8_t*>(qkv.data_ptr());
  params.allreduce_in = base;
  params.allreduce_in_k = base + q_size * elem_bytes;
  params.rms_gamma = norm_weight_q.data_ptr();
  params.rms_gamma_k = norm_weight_k.data_ptr();
  params.rms_eps = static_cast<float>(eps);
  params.stream = at::cuda::getCurrentCUDAStream(qkv.get_device());

  params.rms_norm_out = q_out.mutable_data_ptr();
  params.rms_norm_out_k = k_out.mutable_data_ptr();

  vllm::tensorrt_llm::minimax_reduce_rms_op(params);
  return {q_out, k_out};
}
