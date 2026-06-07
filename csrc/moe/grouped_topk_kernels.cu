/*
 * Adapted from
 * https://github.com/NVIDIA/TensorRT-LLM/blob/v1.3.0rc2/cpp/tensorrt_llm/kernels/noAuxTcKernels.cu
 * Copyright (c) 2025, The vLLM team.
 * SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// =============================================================================
// 中文注释：Grouped Top-K 路由算子实现
//
// 本文件实现了 DeepSeek V3 / Kimi K2 等 MoE 模型使用的"分组 Top-K"路由算法。
// 与标准 top-k 不同，grouped top-k 先将专家分成 n_group 组，每组计算一个
// "组得分"（组内 top-2 专家得分之和），然后选出得分最高的 topk_group 个组，
// 最终只在这些被选中的组内做 top-k 选择。
//
// 算法流程（以 DeepSeek V3 为例，256 专家，8 组，每组 32 专家）：
// 1. 对每个 token 的 256 个专家得分做 sigmoid/scoring 激活 + bias 偏移。
// 2. 每个 warp 负责一个专家组，计算组内 top-2 得分之和作为组得分。
// 3. warp 0 从 8 个组得分中选出 topk_group 个组。
// 4. 在被选中的组内，合并所有候选专家，选出最终的 topk 个专家。
// 5. 对选中专家的权重做归一化和缩放。
//
// 支持两种 kernel 路径：
// - grouped_topk_fused_kernel：通用路径，一个 warp 对应一个专家组，
//   使用 WarpSelect 进行高效的在线 top-k 选择。
// - grouped_topk_fused_small_expert_count_kernel：针对小专家数（<=512）的
//   优化路径，使用 reduceTopK 进行 warp 级归约，支持无分组和有分组两种模式。
//
// 支持的模型配置：
// - DeepSeek V3: 256 experts, 8 groups, topk_group=4, topk=8
// - Kimi K2: 384 experts
// - Nemotron: 512 experts, 1 group, topk=22
//
// 依赖文件：moeTopKFuncs.cuh（提供 warp 级 reduceTopK 原语）
// =============================================================================
#include "moeTopKFuncs.cuh"
#include <c10/cuda/CUDAStream.h>
#include <torch/all.h>
#include <cmath>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda/std/limits>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

namespace vllm {
namespace moe {

constexpr unsigned FULL_WARP_MASK = 0xffffffff;
static constexpr int WARP_SIZE = 32;
static constexpr int NumNemotronExperts = 512;
static constexpr int NumKimiK2Experts = 384;
static constexpr int NumDeepseekExperts = 256;
static constexpr int MaxSupportedExpertCount =
    std::max({NumNemotronExperts, NumKimiK2Experts, NumDeepseekExperts});
static constexpr int MaxNumExpertsUnit = 128;
static constexpr int NumTopGroupScores = 2;
static constexpr int DefaultMaxNumTopExperts = 8;
static constexpr int MaxSupportedTopExperts = 22;
static constexpr int MaxNumTopGroups = 4;

// 中文注释：warp_topk 命名空间 —— 提供 warp 级别的 Top-K 排序/选择原语。
// 包含 BitonicSort（双调排序）、BitonicMerge（双调归并）、WarpSort（排序基类）、
// WarpSelect（在线 Top-K 选择）等工具类。
// 核心思想：利用 warp 内 __shfl_xor_sync 进行无 shared memory 的高效排序。
namespace warp_topk {

template <int size, typename T>
__host__ __device__ constexpr T round_up_to_multiple_of(T len) {
  if (len == 0) {
    return 0;
  }
  return ((len - 1) / size + 1) * size;
}

template <typename T>
constexpr __host__ __device__ bool isPowerOf2(T v) {
  return (v && !(v & (v - 1)));
}

template <bool greater, typename T>
__forceinline__ __device__ bool is_better_than(T val, T baseline) {
  return (val > baseline && greater) || (val < baseline && !greater);
}

template <bool greater, typename T, typename idxT>
__forceinline__ __device__ bool is_better_than(T val, T baseline, idxT index,
                                               idxT baseline_index) {
  bool res = (val > baseline && greater) || (val < baseline && !greater);
  if (val == baseline) {
    res = (index < baseline_index && greater) ||
          (index < baseline_index && !greater);
  }
  return res;
}

// 中文注释：BitonicMerge —— 双调归并网络。
// 输入一个双调序列（先递增后递减，或先递减后递增），输出单调序列。
// 递归地将序列拆分为两半，每半内部已经是单调的，通过比较-交换操作归并。
// size >= 2*WARP_SIZE 时使用寄存器数组存储，否则使用 warp shuffle。
template <int size, bool ascending, bool reverse, typename T, typename idxT,
          bool is_stable>
struct BitonicMerge {
  // input should be a bitonic sequence, and sort it to be a monotonic sequence
  __device__ static void merge(T* __restrict__ val_arr,
                               idxT* __restrict__ idx_arr) {
    static_assert(isPowerOf2(size));
    static_assert(size >= 2 * WARP_SIZE);
    constexpr int arr_len = size / WARP_SIZE;

    constexpr int stride = arr_len / 2;
    for (int i = 0; i < stride; ++i) {
      int const other_i = i + stride;
      T& val = val_arr[i];
      T& other_val = val_arr[other_i];
      bool is_better;
      if constexpr (is_stable) {
        is_better = is_better_than<ascending>(val, other_val, idx_arr[i],
                                              idx_arr[other_i]);
      } else {
        is_better = is_better_than<ascending>(val, other_val);
      }

      if (is_better) {
        T tmp = val;
        val = other_val;
        other_val = tmp;

        idxT tmp2 = idx_arr[i];
        idx_arr[i] = idx_arr[other_i];
        idx_arr[other_i] = tmp2;
      }
    }

    BitonicMerge<size / 2, ascending, reverse, T, idxT, is_stable>::merge(
        val_arr, idx_arr);
    BitonicMerge<size / 2, ascending, reverse, T, idxT, is_stable>::merge(
        val_arr + arr_len / 2, idx_arr + arr_len / 2);
  }
};

// 中文注释：BitonicSort —— 双调排序网络。
// 将无序序列排序为单调序列。先递归地将序列分为两半，分别排成递增和递减，
// 形成双调序列后用 BitonicMerge 归并。
template <int size, bool ascending, typename T, typename idxT, bool is_stable>
struct BitonicSort {
  __device__ static void sort(T* __restrict__ val_arr,
                              idxT* __restrict__ idx_arr) {
    static_assert(isPowerOf2(size));
    static_assert(size >= 2 * WARP_SIZE);
    constexpr int arr_len = size / WARP_SIZE;

    BitonicSort<size / 2, true, T, idxT, is_stable>::sort(val_arr, idx_arr);
    BitonicSort<size / 2, false, T, idxT, is_stable>::sort(
        val_arr + arr_len / 2, idx_arr + arr_len / 2);
    BitonicMerge<size, ascending, ascending, T, idxT, is_stable>::merge(
        val_arr, idx_arr);
  }
};

template <bool ascending, typename T, typename idxT, bool is_stable>
struct BitonicSort<32, ascending, T, idxT, is_stable> {
  __device__ static void sort(T* __restrict__ val_arr,
                              idxT* __restrict__ idx_arr) {
    int const lane = threadIdx.x % WARP_SIZE;

    // ascending doesn't matter before merging since all we need is a bitonic
    // sequence
    for (int stage = 0; stage < 4; ++stage) {
      for (int stride = (1 << stage); stride > 0; stride /= 2) {
        bool reverse = (lane >> stage) & 2;
        bool is_second = lane & stride;

        T other = __shfl_xor_sync(FULL_WARP_MASK, *val_arr, stride);
        idxT other_idx = __shfl_xor_sync(FULL_WARP_MASK, *idx_arr, stride);

        bool is_better;
        if constexpr (is_stable) {
          if constexpr (ascending) {
            is_better = ((*val_arr > other) ||
                         ((*val_arr == other) && (*idx_arr < other_idx))) !=
                        (reverse != is_second);
          } else {
            is_better = ((*val_arr > other) ||
                         ((*val_arr == other) && (*idx_arr > other_idx))) !=
                        (reverse != is_second);
          }
        } else {
          is_better = (*val_arr != other &&
                       (*val_arr > other) != (reverse != is_second));
        }
        if (is_better) {
          *val_arr = other;
          *idx_arr = other_idx;
        }
      }
    }

    BitonicMerge<32, ascending, ascending, T, idxT, is_stable>::merge(val_arr,
                                                                      idx_arr);
  }
};

template <bool ascending, bool reverse, typename T, typename idxT,
          bool is_stable>
struct BitonicMerge<32, ascending, reverse, T, idxT, is_stable> {
  __device__ static void merge(T* __restrict__ val_arr,
                               idxT* __restrict__ idx_arr) {
    int const lane = threadIdx.x % WARP_SIZE;
    for (int stride = WARP_SIZE / 2; stride > 0; stride /= 2) {
      bool is_second = lane & stride;
      T& val = *val_arr;
      T other = __shfl_xor_sync(FULL_WARP_MASK, val, stride);
      idxT& idx = *idx_arr;
      idxT other_idx = __shfl_xor_sync(FULL_WARP_MASK, idx, stride);

      bool is_better;
      if constexpr (is_stable) {
        if constexpr (ascending) {
          is_better = ((*val_arr > other) ||
                       ((*val_arr == other) && (*idx_arr < other_idx))) ==
                      (reverse != is_second);  // for min
        } else {
          is_better = ((*val_arr > other) ||
                       ((*val_arr == other) && (*idx_arr > other_idx))) ==
                      (reverse != is_second);  // for max
        }
      } else {
        is_better =
            (val != other && ((val > other) == (ascending != is_second)));
      }

      if (is_better) {
        val = other;
        idx = other_idx;
      }
    }
  }
};

// 中文注释：WarpSort —— Warp 级排序基类。
// 每个线程持有 capacity/WARP_SIZE 个元素，通过 warp shuffle 进行排序。
// 提供 load_sorted（加载已排序数据并合并）和 dump（输出排序结果）接口。
template <int capacity, bool greater, typename T, typename idxT, bool is_stable>
class WarpSort {
 public:
  __device__ WarpSort(idxT k, T dummy)
      : lane_(threadIdx.x % WARP_SIZE), k_(k), dummy_(dummy) {
    static_assert(capacity >= WARP_SIZE && isPowerOf2(capacity));

    for (int i = 0; i < max_arr_len_; ++i) {
      val_arr_[i] = dummy_;
      idx_arr_[i] = 0;
    }
  }

  // load and merge k sorted values
  __device__ void load_sorted(T const* __restrict__ in,
                              idxT const* __restrict__ in_idx, idxT start) {
    idxT idx = start + WARP_SIZE - 1 - lane_;
    for (int i = max_arr_len_ - 1; i >= 0; --i, idx += WARP_SIZE) {
      if (idx < start + k_) {
        T t = in[idx];
        bool is_better;
        if constexpr (is_stable) {
          is_better =
              is_better_than<greater>(t, val_arr_[i], in_idx[idx], idx_arr_[i]);
        } else {
          is_better = is_better_than<greater>(t, val_arr_[i]);
        }
        if (is_better) {
          val_arr_[i] = t;
          idx_arr_[i] = in_idx[idx];
        }
      }
    }

    BitonicMerge<capacity, greater, !greater, T, idxT, is_stable>::merge(
        val_arr_, idx_arr_);
  }

  __device__ void dump(T* __restrict__ out, idxT* __restrict__ out_idx) const {
    for (int i = 0; i < max_arr_len_; ++i) {
      idxT out_i = i * WARP_SIZE + lane_;
      if (out_i < k_) {
        out[out_i] = val_arr_[i];
        out_idx[out_i] = idx_arr_[i];
      }
    }
  }

  __device__ void dumpIdx(idxT* __restrict__ out_idx) const {
    for (int i = 0; i < max_arr_len_; ++i) {
      idxT out_i = i * WARP_SIZE + lane_;
      if (out_i < k_) {
        out_idx[out_i] = idx_arr_[i];
      }
    }
  }

  // Accessors for per-lane selected value/index.
  // NOTE: For the common case `capacity == WARP_SIZE`, `max_arr_len_ == 1`
  // and callers should use `i == 0`.
  __device__ __forceinline__ idxT get_idx(int i = 0) const {
    return idx_arr_[i];
  }

  __device__ __forceinline__ T get_val(int i = 0) const { return val_arr_[i]; }

 protected:
  static constexpr int max_arr_len_ = capacity / WARP_SIZE;

  T val_arr_[max_arr_len_];
  idxT idx_arr_[max_arr_len_];

  int const lane_;
  idxT const k_;
  T const dummy_;

};  // end class WarpSort

// 中文注释：WarpSelect —— Warp 级在线 Top-K 选择器（继承自 WarpSort）。
// 与 WarpSort 不同，WarpSelect 不需要一次性加载所有数据，而是通过 add() 方法
// 逐步添加候选元素，内部维护一个大小为 K 的"堆"（实际上是排序数组）。
// 当候选数量超过 K 时，自动丢弃得分最低的元素。
// 使用 shared memory 作为临时缓冲区，当缓冲区满时与寄存器数组合并。
template <int capacity, bool greater, typename T, typename idxT, bool is_stable>
class WarpSelect : public WarpSort<capacity, greater, T, idxT, is_stable> {
 public:
  __device__ WarpSelect(idxT k, T dummy)
      : WarpSort<capacity, greater, T, idxT, is_stable>(k, dummy),
        k_th_(dummy),
        k_th_idx_(0),
        k_th_lane_((k - 1) % WARP_SIZE) {
    extern __shared__ char smem_buf[];  // extern __shared__ T smem_buf[];

    int const num_of_warp = blockDim.x / WARP_SIZE;
    int const warp_id = threadIdx.x / WARP_SIZE;
    val_smem_ = reinterpret_cast<T*>(smem_buf);
    val_smem_ += warp_id * WARP_SIZE;
    idx_smem_ = reinterpret_cast<idxT*>(
        smem_buf +
        round_up_to_multiple_of<256>(num_of_warp * sizeof(T) * WARP_SIZE));
    idx_smem_ += warp_id * WARP_SIZE;
  }

  __device__ void add(T const* in, idxT start, idxT end) {
    idxT const end_for_fullwarp =
        round_up_to_multiple_of<WARP_SIZE>(end - start) + start;
    for (idxT i = start + lane_; i < end_for_fullwarp; i += WARP_SIZE) {
      T val = (i < end) ? in[i] : dummy_;
      add(val, i);
    }
  }

  __device__ void add(T val, idxT idx) {
    bool do_add;
    if constexpr (is_stable) {
      do_add = is_better_than<greater>(val, k_th_, idx, k_th_idx_);
    } else {
      do_add = is_better_than<greater>(val, k_th_);
    }

    uint32_t mask = __ballot_sync(FULL_WARP_MASK, do_add);
    if (mask == 0) {
      return;
    }

    int pos = smem_buf_len_ + __popc(mask & ((0x1u << lane_) - 1));
    if (do_add && pos < WARP_SIZE) {
      val_smem_[pos] = val;
      idx_smem_[pos] = idx;
      do_add = false;
    }
    smem_buf_len_ += __popc(mask);
    if (smem_buf_len_ >= WARP_SIZE) {
      __syncwarp();
      merge_buf_(val_smem_[lane_], idx_smem_[lane_]);
      smem_buf_len_ -= WARP_SIZE;
    }
    if (do_add) {
      pos -= WARP_SIZE;
      val_smem_[pos] = val;
      idx_smem_[pos] = idx;
    }
    __syncwarp();
  }

  __device__ void done() {
    if (smem_buf_len_) {
      T val = (lane_ < smem_buf_len_) ? val_smem_[lane_] : dummy_;
      idxT idx = (lane_ < smem_buf_len_) ? idx_smem_[lane_] : 0;
      merge_buf_(val, idx);
    }
  }

 private:
  __device__ void set_k_th_() {
    k_th_ = __shfl_sync(FULL_WARP_MASK, val_arr_[max_arr_len_ - 1], k_th_lane_);
    if constexpr (is_stable) {
      k_th_idx_ =
          __shfl_sync(FULL_WARP_MASK, idx_arr_[max_arr_len_ - 1], k_th_lane_);
    }
  }

  __device__ void merge_buf_(T val, idxT idx) {
    BitonicSort<WARP_SIZE, greater, T, idxT, is_stable>::sort(&val, &idx);

    T& old = val_arr_[max_arr_len_ - 1];

    bool is_better;
    if constexpr (is_stable) {
      is_better =
          is_better_than<greater>(val, old, idx, idx_arr_[max_arr_len_ - 1]);
    } else {
      is_better = is_better_than<greater>(val, old);
    }

    if (is_better) {
      old = val;
      idx_arr_[max_arr_len_ - 1] = idx;
    }

    BitonicMerge<capacity, greater, !greater, T, idxT, is_stable>::merge(
        val_arr_, idx_arr_);

    set_k_th_();
  }

  using WarpSort<capacity, greater, T, idxT, is_stable>::max_arr_len_;
  using WarpSort<capacity, greater, T, idxT, is_stable>::val_arr_;
  using WarpSort<capacity, greater, T, idxT, is_stable>::idx_arr_;
  using WarpSort<capacity, greater, T, idxT, is_stable>::lane_;
  using WarpSort<capacity, greater, T, idxT, is_stable>::k_;
  using WarpSort<capacity, greater, T, idxT, is_stable>::dummy_;

  T* val_smem_;
  idxT* idx_smem_;
  int smem_buf_len_ = 0;

  T k_th_;
  idxT k_th_idx_;
  int const k_th_lane_;
};  // end class WarpSelect
}  // namespace warp_topk

template <typename T_OUT, typename T_IN>
__device__ inline T_OUT cuda_cast(T_IN val) {
  return val;
}

template <>
__device__ inline float cuda_cast<float, __nv_bfloat16>(__nv_bfloat16 val) {
  return __bfloat162float(val);
}

template <typename T>
__device__ inline T neg_inf() {
  // cuda::std::numeric_limits<T>::infinity() returns `0` for [T=bf16 or fp16]
  // so we need to cast from fp32
  return cuda_cast<T, float>(-cuda::std::numeric_limits<float>::infinity());
}

template <typename T>
__device__ inline bool is_finite(const T val) {
#if (__CUDACC_VER_MAJOR__ * 10000 + __CUDACC_VER_MINOR__ * 100 >= 120800)
  return cuda::std::isfinite(val);
#else
  return isfinite(cuda_cast<float, T>(val));
#endif
}

// Scoring function enums
enum ScoringFunc {
  SCORING_NONE = 0,    // no activation function
  SCORING_SIGMOID = 1  // apply sigmoid
};

// Efficient sigmoid approximation from TensorRT-LLM
__device__ inline float sigmoid_accurate(float x) {
  return 0.5f * tanhf(0.5f * x) + 0.5f;
}

template <typename T>
__device__ inline T apply_sigmoid(T val) {
  float f = cuda_cast<float, T>(val);
  return cuda_cast<T, float>(sigmoid_accurate(f));
}

template <ScoringFunc SF, typename T>
__device__ inline T apply_scoring(T val) {
  if constexpr (SF == SCORING_NONE) {
    return val;
  } else if constexpr (SF == SCORING_SIGMOID) {
    return apply_sigmoid(val);
  } else {
    static_assert(SF == SCORING_NONE || SF == SCORING_SIGMOID,
                  "Unsupported ScoringFunc in apply_scoring");
    return val;
  }
}

// 中文注释：topk_with_k2 —— 单个 warp 内计算组内 top-2 专家得分之和。
// 每个 warp 负责一个专家组（如 32 个专家），计算组内得分最高的 2 个专家的
// 得分之和，作为该组的"组得分"。
// 算法流程：
// 1. 每个线程遍历负责的专家，做 scoring 激活 + bias，找到局部 top-2。
// 2. 通过 warp reduce 找到全局 top-1。
// 3. 如果只有一个线程持有 top-1，则将该线程的 top-1 替换为 top-2，
//    再做一次 warp reduce 找到全局 top-2。
// 4. 输出 top-1 + top-2 作为组得分。
template <typename T, typename BiasT, ScoringFunc SF>
__device__ void topk_with_k2(T* output, T const* input, BiasT const* bias,
                             cg::thread_block_tile<32> const& tile,
                             int32_t const lane_id,
                             int const num_experts_per_group) {
  // Get the top2 per thread
  T largest = neg_inf<T>();
  T second_largest = neg_inf<T>();

  if (num_experts_per_group > WARP_SIZE) {
    for (int i = lane_id; i < num_experts_per_group; i += WARP_SIZE) {
      T value = apply_scoring<SF>(input[i]);
      value = value + static_cast<T>(bias[i]);

      if (value > largest) {
        second_largest = largest;
        largest = value;
      } else if (value > second_largest) {
        second_largest = value;
      }
    }
  } else {
    for (int i = lane_id; i < num_experts_per_group; i += WARP_SIZE) {
      T value = apply_scoring<SF>(input[i]);
      value = value + static_cast<T>(bias[i]);
      largest = value;
    }
  }
  // Get the top2 warpwise
  T max1 = cg::reduce(tile, largest, cg::greater<T>());

  T max2 = max1;
  bool equal_to_max1 = (max1 == largest);

  int count_max1 = __popc(__ballot_sync(FULL_WARP_MASK, equal_to_max1));

  if (count_max1 == 1) {
    largest = (largest == max1) ? second_largest : largest;
    max2 = cg::reduce(tile, largest, cg::greater<T>());
  }

  if (lane_id == 0) {
    *output = max1 + max2;
  }
}

// 中文注释：grouped_topk_fused_kernel —— 通用的分组 Top-K 路由 kernel。
// 适用于专家数较多、每组专家数超过 warp 大小的场景。
//
// 线程组织：一个 block 处理一个 token，一个 warp 处理一个专家组。
// block 大小 = n_group * WARP_SIZE（如 8 组 * 32 = 256 线程）。
//
// 算法流程：
// phase 1（所有 warp 并行）：
//   每个 warp 调用 topk_with_k2 计算该组的 top-2 得分之和，写入 shared memory。
// phase 2（仅 warp 0 执行）：
//   a. 从 n_group 个组得分中选出 topk_group 个组（使用 WarpSelect）。
//   b. 在被选中的组内，遍历所有专家，做 scoring + bias，添加到 expert_sel。
//   c. 从合并的候选中选出最终的 topk 个专家。
//   d. 计算无偏权重（只用 sigmoid 值，不含 bias），可选归一化。
//   e. 乘以 routed_scaling_factor，写入输出。
template <typename T, typename BiasT, typename IdxT, ScoringFunc SF>
__global__ void grouped_topk_fused_kernel(
    T* scores, float* topk_values, IdxT* topk_indices, BiasT const* bias,
    int64_t const num_tokens, int64_t const num_experts, int64_t const n_group,
    int64_t const topk_group, int64_t const topk, bool renormalize,
    double routed_scaling_factor) {
  int32_t const token_id = static_cast<int32_t>(blockIdx.x);
  if (token_id >= num_tokens) {
    return;
  }

  int32_t const warp_id = threadIdx.x / WARP_SIZE;
  int32_t const lane_id = threadIdx.x % WARP_SIZE;

  int32_t const n_group_i32 = static_cast<int32_t>(n_group);
  int32_t const topk_group_i32 = static_cast<int32_t>(topk_group);
  int32_t const topk_i32 = static_cast<int32_t>(topk);
  int32_t const num_experts_i32 = static_cast<int32_t>(num_experts);

  int32_t const num_warps = blockDim.x / WARP_SIZE;
  if (warp_id >= n_group_i32 || num_warps < n_group_i32) {
    return;
  }

  int32_t const num_experts_per_group = num_experts_i32 / n_group_i32;

  T* scores_token = scores + static_cast<int64_t>(token_id) * num_experts;

  cg::thread_block block = cg::this_thread_block();
  cg::thread_block_tile<32> tile = cg::tiled_partition<32>(block);

  extern __shared__ char smem_buf[];
  // warpSelect internal staging buffer layout
  size_t const val_bytes =
      static_cast<size_t>(num_warps) * WARP_SIZE * sizeof(T);
  size_t const val_bytes_aligned =
      warp_topk::round_up_to_multiple_of<256>(val_bytes);
  size_t const idx_bytes =
      static_cast<size_t>(num_warps) * WARP_SIZE * sizeof(int32_t);
  size_t const internal_bytes = val_bytes_aligned + idx_bytes;

  // user-managed shared memory starts after warpSelect internal staging.
  uintptr_t ptr_u = reinterpret_cast<uintptr_t>(smem_buf + internal_bytes);
  ptr_u = (ptr_u + 15) & ~static_cast<uintptr_t>(15);  // align to 16B
  T* s_group_scores = reinterpret_cast<T*>(ptr_u);

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");  // I think all prolog can be put before
                                         // acqbulk because it's ptr arithmetic
#endif

  // phase 1: per-group scan
  int32_t const group_offset = warp_id * num_experts_per_group;
  topk_with_k2<T, BiasT, SF>(s_group_scores + warp_id,
                             scores_token + group_offset, bias + group_offset,
                             tile, lane_id, num_experts_per_group);

  __syncthreads();

  // phase 2: warp0 selects groups + merges candidates to final topk
  if (warp_id != 0) {
    return;
  }

  topk_values += static_cast<int64_t>(token_id) * topk;
  topk_indices += static_cast<int64_t>(token_id) * topk;

  // select topk_group groups by group score
  warp_topk::WarpSelect</*capability*/ WARP_SIZE, /*greater*/ true, T, int32_t,
                        /* is_stable */ true>
      group_sel(static_cast<int32_t>(topk_group_i32), neg_inf<T>());

  // all lanes must participate in WarpSelect::add().
  T gscore = (lane_id < n_group_i32) ? s_group_scores[lane_id] : neg_inf<T>();
  group_sel.add(gscore, lane_id);
  group_sel.done();

  // proceed only if the k-th selected group score is not -inf
  bool proceed = false;
  if (topk_group_i32 > 0) {
    int const kth_lane = topk_group_i32 - 1;
    // broadcast the k-th selected group score to all lanes
    T kth_val = __shfl_sync(FULL_WARP_MASK, group_sel.get_val(0), kth_lane);
    proceed = (kth_val != neg_inf<T>());
  }

  if (!proceed) {
    for (int i = lane_id; i < topk_i32; i += WARP_SIZE) {
      topk_indices[i] = static_cast<IdxT>(i);
      topk_values[i] = 1.0f / static_cast<float>(topk_i32);
    }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    asm volatile("griddepcontrol.launch_dependents;");
#endif
    return;
  }

  // merge per-group topk candidates for selected groups, then select topk
  warp_topk::WarpSelect</*capability*/ WARP_SIZE, /*greater*/ true, T, int32_t,
                        /* is_stable */ true>
      expert_sel(static_cast<int32_t>(topk_i32), neg_inf<T>());

  // selected group ids reside in lanes [0, topk_group)
  int32_t sel_gid_lane = (lane_id < topk_group_i32) ? group_sel.get_idx(0) : 0;

  // add candidates from selected groups to expert_sel
  for (int32_t g = 0; g < topk_group_i32; ++g) {
    int32_t gid = __shfl_sync(FULL_WARP_MASK, sel_gid_lane, g);
    int32_t const offset = gid * num_experts_per_group;
    int32_t const align_num_experts_per_group =
        warp_topk::round_up_to_multiple_of<WARP_SIZE>(num_experts_per_group);
    for (int32_t i = lane_id; i < align_num_experts_per_group; i += WARP_SIZE) {
      // all lanes must call `add()` the same number of times.
      T cand = neg_inf<T>();
      int32_t idx = 0;
      if (i < num_experts_per_group) {
        idx = offset + i;
        T input = scores_token[idx];
        if (is_finite(input)) {
          T score = apply_scoring<SF>(input);
          cand = score + static_cast<T>(bias[idx]);
        }
      }
      expert_sel.add(cand, idx);
    }
  }
  expert_sel.done();

  // compute unbiased routing weights + optional renorm.
  float lane_unbiased = 0.0f;
  IdxT lane_idx = 0;
  if (lane_id < topk_i32) {
    lane_idx = static_cast<IdxT>(expert_sel.get_idx(0));
    T in = scores_token[static_cast<int32_t>(lane_idx)];
    lane_unbiased = cuda_cast<float, T>(apply_scoring<SF>(in));
  }

  float topk_sum = 1e-20f;
  if (renormalize) {
    topk_sum += cg::reduce(tile, lane_unbiased, cg::plus<float>());
  }

  float scale = static_cast<float>(routed_scaling_factor);
  if (renormalize) {
    scale /= topk_sum;
  }

  if (lane_id < topk_i32) {
    topk_indices[lane_id] = lane_idx;
    topk_values[lane_id] = lane_unbiased * scale;
  }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

// 中文注释：grouped_topk_fused_small_expert_count_kernel
// —— 针对小专家数（<=512）优化的分组 Top-K 路由 kernel。
// 与 grouped_topk_fused_kernel 不同，此 kernel 使用 reduceTopK 进行 warp 级归约，
// 而非 WarpSelect，避免了 shared memory 的使用。
//
// 支持两种模式：
// 1. UseGroups=true：有分组模式，先计算组得分，选组，再在组内选专家。
// 2. UseGroups=false：无分组模式，直接从所有专家中选 topk。
//
// 模板参数 MaxNumExperts 决定 shared memory 大小和线程数：
// - MaxNumExpertsUnit (128), NumDeepseekExperts (256), NumKimiK2Experts (384), NumNemotronExperts (512)
// 这些是编译期常量，确保 shared memory 布局在编译时确定。
//
// 线程组织：一个 block 处理一个 token，线程数 = MaxNumExperts。
// 每个 warp 代表一个专家组（或一组专家）。
template <typename T, typename BiasT, typename IdxT, ScoringFunc SF,
          int MaxNumExperts, bool UseGroups,
          int MaxNumTopExperts = DefaultMaxNumTopExperts>
__global__ void grouped_topk_fused_small_expert_count_kernel(
    T* scores, float* topkValues, IdxT* topkIndices, BiasT const* routingBias,
    int64_t const numTokens, int64_t const numGroup, int64_t const topkGroup,
    int64_t const topk, int64_t const numExperts,
    int64_t const numExpertsPerGroup, bool const renormalize,
    double const routedScalingFactor) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
#endif
  // declare shared memory structure
  // number of experts is bounded by number of threads
  __shared__ float __attribute((aligned(128))) smemScoreSigmoid[MaxNumExperts];
  __shared__ float __attribute((aligned(128))) smemScoreBias[MaxNumExperts];
  // number of expert groups is bounded by number of warps
  int constexpr NumWarps = MaxNumExperts / WARP_SIZE;
  __shared__ float __attribute((aligned(128))) smemGroupScores[NumWarps];

  // needed for warp reduce
  auto block = cg::this_thread_block();
  auto warp = cg::tiled_partition<WARP_SIZE>(block);

  // for the final reduction of weight norm, only some lanes need to participate
  int32_t laneIdx = threadIdx.x % WARP_SIZE;
  int32_t warpIdx = __shfl_sync(0xffffffff, threadIdx.x / WARP_SIZE, 0);

  if constexpr (UseGroups) {
    if (warpIdx >= numGroup) {
      return;
    }
  }
  // note that for invalid scores, we simply use a negative value:
  // they work well even with the compacted format used in topK, and
  // sigmoid / bias activated scores cannot be negative
  const float invalidScoreFloat = float{-INFINITY};

  // load bias already; each warp represents one expert group
  auto threadExpert = threadIdx.x;
  bool expertSelected = threadExpert < numExperts;
  if constexpr (UseGroups) {
    threadExpert = warpIdx * numExpertsPerGroup + laneIdx;
    expertSelected = laneIdx < numExpertsPerGroup;
  }

  auto scoreIdx = int64_t{blockIdx.x} * int64_t{numExperts} + threadExpert;
  auto biasVal = expertSelected ? static_cast<float>(routingBias[threadExpert])
                                : invalidScoreFloat;
  topkValues += blockIdx.x * topk;
  topkIndices += blockIdx.x * topk;

  // get our assigned thread score; each warp represents one expert group
  float score =
      expertSelected ? static_cast<float>(scores[scoreIdx]) : invalidScoreFloat;
  auto scoreSigmoid = apply_scoring<SF>(score);
  // write the sigmoid score to shared for later use
  if (expertSelected) {
    smemScoreSigmoid[threadExpert] = scoreSigmoid;
  }

  // get the score with bias
  // note that with invalid values, because sigmoid is < 1 and bias is -1,
  // we must get a negative value, which is smaller than any valid value
  auto scoreBias = float{scoreSigmoid + float{biasVal}};

  if (expertSelected) {
    smemScoreBias[threadExpert] = scoreBias;
  }

  // registers for top group score reduction
  float topExpGroupScores[NumTopGroupScores];
  [[maybe_unused]] int32_t topExpGroupIdx[NumTopGroupScores];
  float topGroups[MaxNumTopGroups];  // bound of numGroup
  int32_t topGroupIdx[MaxNumTopGroups];
  float expertScoreGroup[MaxNumTopGroups];
  int32_t expertIdxGroup[MaxNumTopGroups];
  float topScores[MaxNumTopExperts];  // bound of topk
  int32_t topExperts[MaxNumTopExperts];

  if constexpr (UseGroups) {
    reduce_topk::reduceTopK(warp, topExpGroupScores, topExpGroupIdx, scoreBias,
                            threadExpert,
                            /* minValue */ invalidScoreFloat);

    // get the final group score and write it to shared
    if (warp.thread_rank() == 0) {
      auto groupScore = topExpGroupScores[0] + topExpGroupScores[1];
      smemGroupScores[warpIdx] = groupScore;
    }
  }

  // make group scores available to all warps
  __syncthreads();

  if constexpr (UseGroups) {
    if (warpIdx == 0) {
      // a single warp performs the selection of top groups, and goes on to
      // select the final experts
      float groupScore =
          laneIdx < numGroup ? smemGroupScores[laneIdx] : invalidScoreFloat;

      reduce_topk::reduceTopK(warp, topGroups, topGroupIdx, groupScore, laneIdx,
                              /* minValue */ invalidScoreFloat);
      // final expert selection: get relevant indexes and scores from shared
#pragma unroll
      for (int ii = 0; ii < MaxNumTopGroups; ++ii) {  // bound of numGroup
        auto groupIdx = topGroupIdx[ii];
        expertIdxGroup[ii] = groupIdx * numExpertsPerGroup + laneIdx;

        expertScoreGroup[ii] = (ii < topkGroup) && expertSelected
                                   ? smemScoreBias[expertIdxGroup[ii]]
                                   : invalidScoreFloat;
      }

      reduce_topk::reduceTopK(warp, topScores, topExperts, expertScoreGroup,
                              expertIdxGroup, /* minValue */ invalidScoreFloat,
                              topk);
    }
  } else if constexpr (MaxNumExperts > MaxNumExpertsUnit) {
    // without groups, and the expert number is larger than MaxNumExpertsUnit,
    // we need to use multiple warps to calculate the intermediate topk results

    int constexpr NumExpertWarps = (MaxNumExperts - 1) / MaxNumExpertsUnit + 1;
    int constexpr NumInterTopK = NumExpertWarps * MaxNumTopExperts;
    __shared__ float
        __attribute((aligned(128))) smemInterTopScores[NumInterTopK];
    __shared__ int32_t
        __attribute((aligned(128))) smemInterTopExperts[NumInterTopK];
    if (warpIdx < NumExpertWarps) {
      int offset = warpIdx * WARP_SIZE * MaxNumTopGroups;
#pragma unroll
      for (int ii = 0; ii < MaxNumTopGroups; ++ii) {
        auto expertIdx = ii * WARP_SIZE + laneIdx;
        expertIdxGroup[ii] = offset + expertIdx;
        expertScoreGroup[ii] = offset + expertIdx < numExperts
                                   ? smemScoreBias[offset + expertIdx]
                                   : invalidScoreFloat;
      }
      reduce_topk::reduceTopK(warp, topScores, topExperts, expertScoreGroup,
                              expertIdxGroup,
                              /* minValue */ invalidScoreFloat, topk);

      if (laneIdx < topk) {
        smemInterTopScores[warpIdx * MaxNumTopExperts + laneIdx] =
            topScores[laneIdx];
        smemInterTopExperts[warpIdx * MaxNumTopExperts + laneIdx] =
            topExperts[laneIdx];
      } else if (laneIdx >= topk && laneIdx < MaxNumTopExperts) {
        smemInterTopScores[warpIdx * MaxNumTopExperts + laneIdx] =
            invalidScoreFloat;
        smemInterTopExperts[warpIdx * MaxNumTopExperts + laneIdx] =
            MaxNumExperts - 1;
      }
    }
    __syncthreads();
    if (warpIdx == 0) {
      int constexpr NumInterTopKPerThread = (NumInterTopK - 1) / WARP_SIZE + 1;
      float intermediateScore[NumInterTopKPerThread];
      int32_t intermediateExpert[NumInterTopKPerThread];
      for (int i = laneIdx; i < NumInterTopKPerThread * WARP_SIZE;
           i += WARP_SIZE) {
        int ii = i / WARP_SIZE;
        if (i < NumInterTopK) {
          intermediateScore[ii] = smemInterTopScores[i];
          intermediateExpert[ii] = smemInterTopExperts[i];
        } else {
          intermediateScore[ii] = invalidScoreFloat;
          intermediateExpert[ii] = MaxNumExperts - 1;
        }
      }
      reduce_topk::reduceTopK(warp, topScores, topExperts, intermediateScore,
                              intermediateExpert,
                              /* minValue */ invalidScoreFloat, topk);
    }
  } else {
    // without groups, and the expert number is smaller than MaxNumExpertsUnit
    // each thread just takes `MaxNumTopGroups` experts
    if (warpIdx == 0) {
#pragma unroll
      for (int ii = 0; ii < MaxNumTopGroups; ++ii) {
        auto expertIdx = ii * WARP_SIZE + laneIdx;
        expertIdxGroup[ii] = expertIdx;
        expertScoreGroup[ii] = expertIdx < numExperts ? smemScoreBias[expertIdx]
                                                      : invalidScoreFloat;
      }
      reduce_topk::reduceTopK(warp, topScores, topExperts, expertScoreGroup,
                              expertIdxGroup,
                              /* minValue */ invalidScoreFloat, topk);
    }
  }

  if (warpIdx == 0) {
    // determine our lane's expert index and write to output
    int32_t expertIdx =
        laneIdx < topk ? topExperts[laneIdx] : MaxNumExperts - 1;
    float scoreNorm = laneIdx < topk ? smemScoreSigmoid[expertIdx] : 0.F;
    float finalScore = static_cast<float>(scoreNorm * routedScalingFactor);
    // norm the value
    if (renormalize) {
      auto redNorm = cg::reduce(warp, scoreNorm, cg::plus<float>{});
      finalScore /= (redNorm + 1e-20);
    }
    // store the topk scores and experts to output
    if (laneIdx < topk) {
      topkValues[laneIdx] = finalScore;
      topkIndices[laneIdx] = expertIdx;
    }
  }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaTriggerProgrammaticLaunchCompletion();
#endif
}

// 中文注释：invokeNoAuxTc —— 分组 Top-K 路由的 kernel 调度入口。
// 根据专家数量和分组配置，选择最优的 kernel 实现：
// 1. 小专家数（<=128 或 <=256/384/512）且条件满足时，使用
//    grouped_topk_fused_small_expert_count_kernel（共享内存归约，更快）。
// 2. 否则使用 grouped_topk_fused_kernel（WarpSelect，在线选择，更通用）。
//
// 支持 CUDA Graph 的 Programmatic Dependent Launch (PDL) 优化：
// 通过 cudaLaunchAttributeProgrammaticStreamSerialization 实现 kernel 间依赖。
template <typename T, typename BiasT, typename IdxT, ScoringFunc SF>
void invokeNoAuxTc(T* scores, float* topk_values, IdxT* topk_indices,
                   BiasT const* bias, int64_t const num_tokens,
                   int64_t const num_experts, int64_t const n_group,
                   int64_t const topk_group, int64_t const topk,
                   bool const renormalize, double const routed_scaling_factor,
                   bool enable_pdl = false, cudaStream_t const stream = 0) {
  cudaLaunchConfig_t config;
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = enable_pdl;
  config.numAttrs = 1;
  config.attrs = attrs;

  // Check if we can use the optimized
  // grouped_topk_fused_small_expert_count_kernel
  bool const is_single_group =
      (n_group == 1) && (topk_group == 1) &&
      (num_experts <= MaxSupportedExpertCount) &&
      (topk <= DefaultMaxNumTopExperts || topk == MaxSupportedTopExperts);

  int64_t const experts_per_group = num_experts / n_group;
  bool const is_multi_group =
      (n_group > 1) && (num_experts <= NumDeepseekExperts) &&
      (experts_per_group <= WARP_SIZE) &&
      (experts_per_group * topk_group <= MaxNumExpertsUnit) &&
      (topk <= DefaultMaxNumTopExperts) && (topk_group <= MaxNumTopGroups);

  if (is_single_group || is_multi_group) {
    auto* kernel_instance =
        &grouped_topk_fused_small_expert_count_kernel<T, BiasT, IdxT, SF,
                                                      NumDeepseekExperts, true>;
    int num_threads = NumDeepseekExperts;
    if (is_single_group) {
      // Special case for Nemotron, which selects top 22 from 512 experts, and 1
      // group only.
      if (num_experts == NumNemotronExperts && n_group == 1 &&
          topk == MaxSupportedTopExperts) {
        kernel_instance = &grouped_topk_fused_small_expert_count_kernel<
            T, BiasT, IdxT, SF, NumNemotronExperts, false,
            MaxSupportedTopExperts>;
        num_threads = NumNemotronExperts;
      } else if (num_experts > NumKimiK2Experts &&
                 num_experts <= MaxSupportedExpertCount) {
        kernel_instance = &grouped_topk_fused_small_expert_count_kernel<
            T, BiasT, IdxT, SF, MaxSupportedExpertCount, false>;
        num_threads = MaxSupportedExpertCount;
      } else if (num_experts > MaxNumExpertsUnit &&
                 num_experts <= NumKimiK2Experts) {
        kernel_instance = &grouped_topk_fused_small_expert_count_kernel<
            T, BiasT, IdxT, SF, NumKimiK2Experts, false>;
        num_threads = NumKimiK2Experts;
      } else {
        kernel_instance = &grouped_topk_fused_small_expert_count_kernel<
            T, BiasT, IdxT, SF, MaxNumExpertsUnit, false>;
        num_threads = MaxNumExpertsUnit;
      }
    }
    config.gridDim = num_tokens;
    config.blockDim = num_threads;
    config.dynamicSmemBytes = 0;
    cudaLaunchKernelEx(&config, kernel_instance, scores, topk_values,
                       topk_indices, bias, num_tokens, n_group, topk_group,
                       topk, num_experts, num_experts / n_group, renormalize,
                       routed_scaling_factor);
  } else {
    auto* kernel_instance = &grouped_topk_fused_kernel<T, BiasT, IdxT, SF>;
    // One block per token; one warp per group.
    config.gridDim = static_cast<uint32_t>(num_tokens);
    config.blockDim = static_cast<uint32_t>(n_group) * WARP_SIZE;
    // Dynamic shared memory: WarpSelect staging + per-group topk buffers.
    int32_t const num_warps = static_cast<int32_t>(n_group);
    size_t const val_bytes =
        static_cast<size_t>(num_warps) * WARP_SIZE * sizeof(T);
    size_t const val_bytes_aligned =
        warp_topk::round_up_to_multiple_of<256>(val_bytes);
    size_t const idx_bytes =
        static_cast<size_t>(num_warps) * WARP_SIZE * sizeof(int32_t);
    size_t const internal_bytes = val_bytes_aligned + idx_bytes;
    size_t const extra_bytes = 16 + static_cast<size_t>(n_group) * sizeof(T);
    config.dynamicSmemBytes = internal_bytes + extra_bytes;
    cudaLaunchKernelEx(&config, kernel_instance, scores, topk_values,
                       topk_indices, bias, num_tokens, num_experts, n_group,
                       topk_group, topk, renormalize, routed_scaling_factor);
  }
}

#define INSTANTIATE_NOAUX_TC(T, BiasT, IdxT, SF)                             \
  template void invokeNoAuxTc<T, BiasT, IdxT, SF>(                           \
      T * scores, float* topk_values, IdxT* topk_indices, BiasT const* bias, \
      int64_t const num_tokens, int64_t const num_experts,                   \
      int64_t const n_group, int64_t const topk_group, int64_t const topk,   \
      bool const renormalize, double const routed_scaling_factor,            \
      bool enable_pdl, cudaStream_t const stream);

INSTANTIATE_NOAUX_TC(float, float, int32_t, SCORING_SIGMOID);
INSTANTIATE_NOAUX_TC(float, half, int32_t, SCORING_SIGMOID);
INSTANTIATE_NOAUX_TC(float, __nv_bfloat16, int32_t, SCORING_SIGMOID);
INSTANTIATE_NOAUX_TC(half, float, int32_t, SCORING_SIGMOID);
INSTANTIATE_NOAUX_TC(half, half, int32_t, SCORING_SIGMOID);
INSTANTIATE_NOAUX_TC(half, __nv_bfloat16, int32_t, SCORING_SIGMOID);
INSTANTIATE_NOAUX_TC(__nv_bfloat16, float, int32_t, SCORING_SIGMOID);
INSTANTIATE_NOAUX_TC(__nv_bfloat16, half, int32_t, SCORING_SIGMOID);
INSTANTIATE_NOAUX_TC(__nv_bfloat16, __nv_bfloat16, int32_t, SCORING_SIGMOID);
INSTANTIATE_NOAUX_TC(float, float, int32_t, SCORING_NONE);
INSTANTIATE_NOAUX_TC(float, half, int32_t, SCORING_NONE);
INSTANTIATE_NOAUX_TC(float, __nv_bfloat16, int32_t, SCORING_NONE);
INSTANTIATE_NOAUX_TC(half, float, int32_t, SCORING_NONE);
INSTANTIATE_NOAUX_TC(half, half, int32_t, SCORING_NONE);
INSTANTIATE_NOAUX_TC(half, __nv_bfloat16, int32_t, SCORING_NONE);
INSTANTIATE_NOAUX_TC(__nv_bfloat16, float, int32_t, SCORING_NONE);
INSTANTIATE_NOAUX_TC(__nv_bfloat16, half, int32_t, SCORING_NONE);
INSTANTIATE_NOAUX_TC(__nv_bfloat16, __nv_bfloat16, int32_t, SCORING_NONE);
}  // end namespace moe
}  // namespace vllm

// 中文注释：grouped_topk —— Python 层调用的分组 Top-K 路由入口函数。
// 输入：
//   scores: [num_tokens, num_experts] — 每个 token 对每个专家的原始得分
//   bias: [num_experts] — 每个专家的校正偏置
//   n_group: 专家组数（如 DeepSeek V3 为 8）
//   topk_group: 选中的专家组数（如 DeepSeek V3 为 4）
//   topk: 最终选出的专家数（如 DeepSeek V3 为 8）
//   scoring_func: 激活函数类型（0=无，1=sigmoid）
// 输出：
//   topk_values: [num_tokens, topk] — 选中专家的路由权重
//   topk_indices: [num_tokens, topk] — 选中专家的索引
//
// 流程：根据 scores/bias 的数据类型（fp16/fp32/bf16）和 scoring_func
// 分发到对应的 invokeNoAuxTc 模板实例。
std::tuple<torch::Tensor, torch::Tensor> grouped_topk(
    torch::Tensor const& scores, int64_t n_group, int64_t topk_group,
    int64_t topk, bool renormalize, double routed_scaling_factor,
    torch::Tensor const& bias, int64_t scoring_func = 0) {
  auto data_type = scores.scalar_type();
  auto bias_type = bias.scalar_type();
  auto input_size = scores.sizes();
  int64_t num_tokens = input_size[0];
  int64_t num_experts = input_size[1];
  TORCH_CHECK(input_size.size() == 2, "scores must be a 2D Tensor");
  TORCH_CHECK(n_group > 0, "n_group must be positive");
  TORCH_CHECK(topk > 0, "topk must be positive");
  TORCH_CHECK(topk_group > 0, "topk_group must be positive");
  TORCH_CHECK(topk_group <= n_group, "topk_group must be <= n_group");
  TORCH_CHECK(num_experts % n_group == 0,
              "num_experts should be divisible by n_group");
  TORCH_CHECK(n_group <= 32,
              "n_group should be smaller than or equal to 32 for now");
  TORCH_CHECK(topk <= 32, "topk should be smaller than or equal to 32 for now");
  TORCH_CHECK(topk <= topk_group * (num_experts / n_group),
              "topk must be <= topk_group * (num_experts / n_group)");
  TORCH_CHECK(scoring_func == vllm::moe::SCORING_NONE ||
                  scoring_func == vllm::moe::SCORING_SIGMOID,
              "scoring_func must be SCORING_NONE (0) or SCORING_SIGMOID (1)");

  // Always output float32 for topk_values (eliminates Python-side conversion)
  torch::Tensor topk_values = torch::empty(
      {num_tokens, topk}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
  torch::Tensor topk_indices = torch::empty(
      {num_tokens, topk}, torch::dtype(torch::kInt32).device(torch::kCUDA));

  auto stream = c10::cuda::getCurrentCUDAStream(scores.get_device());
  auto const sf = static_cast<vllm::moe::ScoringFunc>(scoring_func);

#define LAUNCH_KERNEL_SF(T, BiasT, IdxT)                                      \
  do {                                                                        \
    switch (sf) {                                                             \
      case vllm::moe::SCORING_NONE:                                           \
        vllm::moe::invokeNoAuxTc<T, BiasT, IdxT, vllm::moe::SCORING_NONE>(    \
            reinterpret_cast<T*>(scores.mutable_data_ptr()),                  \
            reinterpret_cast<float*>(topk_values.mutable_data_ptr()),         \
            reinterpret_cast<IdxT*>(topk_indices.mutable_data_ptr()),         \
            reinterpret_cast<BiasT const*>(bias.data_ptr()), num_tokens,      \
            num_experts, n_group, topk_group, topk, renormalize,              \
            routed_scaling_factor, false, stream);                            \
        break;                                                                \
      case vllm::moe::SCORING_SIGMOID:                                        \
        vllm::moe::invokeNoAuxTc<T, BiasT, IdxT, vllm::moe::SCORING_SIGMOID>( \
            reinterpret_cast<T*>(scores.mutable_data_ptr()),                  \
            reinterpret_cast<float*>(topk_values.mutable_data_ptr()),         \
            reinterpret_cast<IdxT*>(topk_indices.mutable_data_ptr()),         \
            reinterpret_cast<BiasT const*>(bias.data_ptr()), num_tokens,      \
            num_experts, n_group, topk_group, topk, renormalize,              \
            routed_scaling_factor, false, stream);                            \
        break;                                                                \
      default:                                                                \
        throw std::invalid_argument("Unsupported scoring_func");              \
        break;                                                                \
    }                                                                         \
  } while (0)

#define LAUNCH_KERNEL(T, IdxT)                                         \
  do {                                                                 \
    switch (bias_type) {                                               \
      case torch::kFloat16:                                            \
        LAUNCH_KERNEL_SF(T, half, IdxT);                               \
        break;                                                         \
      case torch::kFloat32:                                            \
        LAUNCH_KERNEL_SF(T, float, IdxT);                              \
        break;                                                         \
      case torch::kBFloat16:                                           \
        LAUNCH_KERNEL_SF(T, __nv_bfloat16, IdxT);                      \
        break;                                                         \
      default:                                                         \
        throw std::invalid_argument(                                   \
            "Invalid bias dtype, only supports float16, float32, and " \
            "bfloat16");                                               \
        break;                                                         \
    }                                                                  \
  } while (0)

  switch (data_type) {
    case torch::kFloat16:
      // Handle Float16
      LAUNCH_KERNEL(half, int32_t);
      break;
    case torch::kFloat32:
      // Handle Float32
      LAUNCH_KERNEL(float, int32_t);
      break;
    case torch::kBFloat16:
      // Handle BFloat16
      LAUNCH_KERNEL(__nv_bfloat16, int32_t);
      break;
    default:
      // Handle other data types
      throw std::invalid_argument(
          "Invalid dtype, only supports float16, float32, and bfloat16");
      break;
  }
#undef LAUNCH_KERNEL
#undef LAUNCH_KERNEL_SF
  return {topk_values, topk_indices};
}
