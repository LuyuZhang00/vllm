/*
 * Modified by Neural Magic
 * Copyright (C) Marlin.2024 Elias Frantar
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * Adapted from https://github.com/IST-DASLab/marlin
 */

// =============================================================================
// 中文注释: Marlin 量化 GEMM 主入口文件
// =============================================================================
// 本文件是 Marlin 量化矩阵乘法的 Python/C++ 绑定层和 kernel 调度逻辑。
//
// 整体流程:
//   1. marlin_gemm(): Python 绑定入口，参数校验和数据准备
//   2. marlin_mm(): 核心调度函数，选择最优 kernel 配置并启动
//   3. permute_cols_kernel(): 列重排 kernel (activation reordering)
//   4. determine_exec_config(): 自动选择最优的线程配置
//   5. get_marlin_kernel(): 根据参数选择具体的 Marlin kernel 实例
//
// 支持的量化格式:
//   - W4A16: 4-bit 权重 + 16-bit 激活 (FP16/BF16)
//   - W8A16: 8-bit 权重 + 16-bit 激活
//   - W4A8: 4-bit 权重 + 8-bit 激活 (FP8/INT8)
//   - 支持 grouped quantization (per-group scale/zero-point)
//   - 支持 activation reordering (act_order) 提升量化精度
//
// 关键设计:
//   - 自动配置选择: 根据矩阵形状和 GPU 能力自动选择最优的
//     thread_m_blocks, thread_n_blocks, thread_k_blocks 配置
//   - 分块处理: 当 M 维度较大时，拆分为多个 block 并行处理
//   - 列重排: 支持 act_order 模式，通过 permute_cols_kernel 对 A 矩阵列重排
//   - SM 版本适配: Turing (SM75) 使用 2 阶段 pipeline，Ampere+ 使用 4 阶段
// =============================================================================

#ifndef MARLIN_NAMESPACE_NAME
  #define MARLIN_NAMESPACE_NAME marlin
#endif

#include "kernel.h"
#include "core/registration.h"

#define STATIC_ASSERT_SCALAR_TYPE_VALID(scalar_t)               \
  static_assert(std::is_same<scalar_t, half>::value ||          \
                    std::is_same<scalar_t, nv_bfloat16>::value, \
                "only float16 and bfloat16 is supported");

namespace marlin {

__global__ void MarlinDefault(MARLIN_KERNEL_PARAMS){};

using MarlinFuncPtr = void (*)(MARLIN_KERNEL_PARAMS);

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 750

__global__ void permute_cols_kernel(int4 const* __restrict__ a_int4_ptr,
                                    int const* __restrict__ perm_int_ptr,
                                    int4* __restrict__ out_int4_ptr, int size_m,
                                    int size_k, int lda, int block_rows) {}

}  // namespace marlin

torch::Tensor marlin_gemm(
    torch::Tensor& a, std::optional<torch::Tensor> c_or_none,
    torch::Tensor& b_q_weight,
    std::optional<torch::Tensor> const& b_bias_or_none, torch::Tensor& b_scales,
    std::optional<torch::Tensor> const& b_zeros_or_none,
    std::optional<torch::Tensor> const& g_idx_or_none,
    std::optional<torch::Tensor> const& perm_or_none, torch::Tensor& workspace,
    vllm::ScalarTypeId const& b_type_id, int64_t size_m, int64_t size_n,
    int64_t size_k, bool is_k_full, bool use_atomic_add, bool use_fp32_reduce,
    bool is_zp_float) {
  TORCH_CHECK_NOT_IMPLEMENTED(false,
                              "marlin_gemm(..) requires CUDA_ARCH >= 7.5");
  return torch::empty({1, 1});
}

#else

// For a given "a" of size [M,K] performs a permutation of the K columns based
// on the given "perm" indices.
// 中文注释: 列重排 kernel (Activation Reordering)
// 功能: 对输入矩阵 A 的 K 维度列进行重排，使得同一量化 group 的列连续存放
// 这是 act_order (activation ordering) 优化的关键步骤
//
// 为什么需要列重排:
//   - 在 grouped quantization 中，每个 group 有独立的 scale/zero-point
//   - 如果同一 group 的 K 维度元素在内存中不连续，kernel 需要频繁跳转访问 scale
//   - 通过重排使同一 group 的列连续，kernel 可以顺序访问 scale，提升效率
//
// 实现方式:
//   - 每个 thread block 处理 block_rows 行
//   - 每个线程并行处理 K 维度的多个元素
//   - perm_int_ptr[k] 存储第 k 列应该从原始矩阵的哪一列读取
__global__ void permute_cols_kernel(int4 const* __restrict__ a_int4_ptr,
                                    int const* __restrict__ perm_int_ptr,
                                    int4* __restrict__ out_int4_ptr, int size_m,
                                    int size_k, int lda, int block_rows) {
  auto start_row = block_rows * blockIdx.x;
  int finish_row = start_row + block_rows;
  if (finish_row > size_m) {
    finish_row = size_m;
  }
  int cur_block_rows = finish_row - start_row;

  int input_row_stride = lda * sizeof(half) / 16;
  int output_row_stride = size_k * sizeof(half) / 16;

  auto permute_row = [&](int row) {
    int iters = size_k / default_threads;
    int rest = size_k % default_threads;

    int input_offset = row * input_row_stride;
    int output_offset = row * output_row_stride;

    half const* a_row_half =
        reinterpret_cast<half const*>(a_int4_ptr + input_offset);
    half* out_half = reinterpret_cast<half*>(out_int4_ptr + output_offset);

    int base_k = 0;

    for (int i = 0; i < iters; i++) {
      auto cur_k = base_k + threadIdx.x;
      int src_pos = perm_int_ptr[cur_k];

      out_half[cur_k] = a_row_half[src_pos];

      base_k += default_threads;
    }

    if (rest) {
      if (threadIdx.x < rest) {
        auto cur_k = base_k + threadIdx.x;
        int src_pos = perm_int_ptr[cur_k];

        out_half[cur_k] = a_row_half[src_pos];
      }
    }
  };

  for (int i = 0; i < cur_block_rows; i++) {
    int cur_row = start_row + i;
    if (cur_row < size_m) {
      permute_row(cur_row);
    }
  }
}

// 中文注释: 线程配置结构体
// thread_k: 每个线程处理的 K 维度大小 (以 16 元素为单位)
// thread_n: 每个线程处理的 N 维度大小
// num_threads: thread block 中的线程数
typedef struct {
  int thread_k;
  int thread_n;
  int num_threads;
} thread_config_t;

// 中文注释: 小 batch (M <= 1) 的线程配置候选列表
// 优先使用较大的 tile (128x128) 以充分利用 Tensor Core
// 当共享内存不够时退化为较小的 tile
thread_config_t small_batch_thread_configs[] = {
    // Ordered by priority

    // thread_k, thread_n, num_threads
    {128, 128, 256},
    {64, 128, 128},
    {128, 64, 128}};

// 中文注释: 大 batch (M > 1) 的线程配置候选列表
// 使用更大的 N 维度 tile (256) 来提升吞吐量
thread_config_t large_batch_thread_configs[] = {
    // Ordered by priority

    // thread_k, thread_n, num_threads
    {64, 256, 256},
    {64, 128, 128},
    {128, 64, 128}};

// 中文注释: 执行配置结构体
// blocks_per_sm: 每个 SM 上运行的 thread block 数量
// tb_cfg: 线程配置 (thread_k, thread_n, num_threads)
typedef struct {
  int blocks_per_sm;
  thread_config_t tb_cfg;
} exec_config_t;

// 中文注释: 计算 scale 在共享内存中所需的缓存大小
// 用于判断当前配置是否能放入共享内存
// 参数说明:
//   - group_size == -1: 每个 N 列一个 scale (per-channel)
//   - group_size == 0: 使用 act_order，最坏情况是 32 个 group
//   - group_size > 0: 标准的 per-group quantization
int get_scales_cache_size(thread_config_t const& th_config, int prob_m,
                          int prob_n, int prob_k, int num_bits, int group_size,
                          bool has_act_order, bool is_k_full, int stages) {
  bool cache_scales_chunk = has_act_order && !is_k_full;

  int tb_n = th_config.thread_n;
  int tb_k = th_config.thread_k;

  // Get max scale groups per thread-block
  int tb_groups;
  if (group_size == -1) {
    tb_groups = 1;
  } else if (group_size == 0) {
    tb_groups = div_ceil(tb_k, 32);  // Worst case is 32 group size
  } else {
    tb_groups = div_ceil(tb_k, group_size);
  }

  if (cache_scales_chunk) {
    int load_groups =
        tb_groups * stages * 2;          // Chunk size is 2x pipeline over dim K
    load_groups = max(load_groups, 32);  // We load at least 32 scale groups
    return load_groups * tb_n * 2;
  } else {
    int tb_scales = tb_groups * tb_n * 2;

    return tb_scales * stages;
  }
}

int get_kernel_cache_size(thread_config_t const& th_config, int thread_m_blocks,
                          int prob_m, int prob_n, int prob_k, int num_bits,
                          int group_size, bool has_act_order, bool is_k_full,
                          int has_zp, bool is_zp_float, bool is_a_8bit,
                          int stages) {
  int pack_factor = 32 / num_bits;

  // Get B size
  int tb_k = th_config.thread_k;
  int tb_n = th_config.thread_n;
  int tb_m = thread_m_blocks * 16;
  int sh_a_size = stages * (tb_m * tb_k) * (is_a_8bit ? 1 : 2);
  int sh_b_size = stages * (tb_k * tb_n / pack_factor) * 4;
  int sh_red_size = tb_m * (tb_n + 8) * 2;
  int sh_bias_size = tb_n * 2;
  int tmp_size =
      (sh_b_size > sh_red_size ? sh_red_size : sh_b_size) + sh_bias_size;
  tmp_size = max(max(sh_b_size, sh_red_size), tmp_size);

  int sh_s_size =
      get_scales_cache_size(th_config, prob_m, prob_n, prob_k, num_bits,
                            group_size, has_act_order, is_k_full, stages);
  int sh_g_idx_size = has_act_order && !is_k_full ? stages * tb_k / 4 : 0;
  int sh_zp_size = 0;
  if (has_zp) {
    if (is_zp_float)
      sh_zp_size = sh_s_size;
    else if (num_bits == 4)
      sh_zp_size = sh_s_size / 4;
    else if (num_bits == 8)
      sh_zp_size = sh_s_size / 2;
  }

  int total_size =
      tmp_size + sh_a_size + sh_s_size + sh_zp_size + sh_g_idx_size;

  return total_size;
}

bool is_valid_config(thread_config_t const& th_config, int thread_m_blocks,
                     int prob_m, int prob_n, int prob_k, int num_bits,
                     int group_size, bool has_act_order, bool is_k_full,
                     int has_zp, bool is_zp_float, bool is_a_8bit, int stages,
                     int max_shared_mem) {
  // Sanity
  if (th_config.thread_k == -1 || th_config.thread_n == -1 ||
      th_config.num_threads == -1) {
    return false;
  }

  // Verify K/N are divisible by thread K/N
  if (prob_k % th_config.thread_k != 0 || prob_n % th_config.thread_n != 0) {
    return false;
  }

  // Verify min for thread K/N
  if (th_config.thread_n < min_thread_n || th_config.thread_k < min_thread_k) {
    return false;
  }

  // num_threads must be at least 128 (= 4 warps)
  if (th_config.num_threads < 128) {
    return false;
  }

  // Check that pipeline fits into cache
  int cache_size = get_kernel_cache_size(
      th_config, thread_m_blocks, prob_m, prob_n, prob_k, num_bits, group_size,
      has_act_order, is_k_full, has_zp, is_zp_float, is_a_8bit, stages);
  return cache_size <= max_shared_mem;
}

// 中文注释: 根据模板参数选择具体的 Marlin kernel 实例
// 通过 kernel_selector.h 中的宏展开，将所有可能的模板组合编译为具体的 kernel 函数
// 然后返回匹配当前参数的函数指针
// 如果没有匹配的 kernel，返回 MarlinDefault (空函数)
MarlinFuncPtr get_marlin_kernel(
    const vllm::ScalarType a_type, const vllm::ScalarType b_type,
    const vllm::ScalarType c_type, const vllm::ScalarType s_type,
    int thread_m_blocks, int thread_n_blocks, int thread_k_blocks,
    bool m_block_size_8, bool has_act_order, bool has_zp, int group_blocks,
    int threads, bool is_zp_float, int stages) {
  int num_bits = b_type.size_bits();
  auto kernel = MarlinDefault;

  #include "kernel_selector.h"

  return kernel;
}

exec_config_t determine_exec_config(
    const vllm::ScalarType& a_type, const vllm::ScalarType& b_type,
    const vllm::ScalarType& c_type, const vllm::ScalarType& s_type, int prob_m,
    int prob_n, int prob_k, int thread_m_blocks, bool m_block_size_8,
    int num_bits, int group_size, bool has_act_order, bool is_k_full,
    bool has_zp, bool is_zp_float, int is_a_8bit, int stages,
    int max_shared_mem, int sms) {
  exec_config_t exec_cfg = exec_config_t{1, thread_config_t{-1, -1, -1}};
  thread_config_t* thread_configs = thread_m_blocks > 1
                                        ? large_batch_thread_configs
                                        : small_batch_thread_configs;
  int thread_configs_size =
      thread_m_blocks > 1
          ? sizeof(large_batch_thread_configs) / sizeof(thread_config_t)
          : sizeof(small_batch_thread_configs) / sizeof(thread_config_t);

  for (int i = 0; i < thread_configs_size; i++) {
    thread_config_t th_config = thread_configs[i];

    if (!is_valid_config(th_config, thread_m_blocks, prob_m, prob_n, prob_k,
                         num_bits, group_size, has_act_order, is_k_full, has_zp,
                         is_zp_float, is_a_8bit, stages,
                         max_shared_mem - 512)) {
      continue;
    }

    int cache_size = get_kernel_cache_size(th_config, thread_m_blocks, prob_m,
                                           prob_n, prob_k, num_bits, group_size,
                                           has_act_order, is_k_full, has_zp,
                                           is_zp_float, is_a_8bit, stages);

    int group_blocks = 0;
    if (!has_act_order) {
      group_blocks = group_size == -1 ? -1 : group_size / 16;
    }

    auto kernel =
        get_marlin_kernel(a_type, b_type, c_type, s_type, thread_m_blocks,
                          th_config.thread_n / 16, th_config.thread_k / 16,
                          m_block_size_8, has_act_order, has_zp, group_blocks,
                          th_config.num_threads, is_zp_float, stages);

    if (kernel == MarlinDefault) continue;

    return {1, th_config};
  }

  return exec_cfg;
}

// 中文注释: Marlin 矩阵乘法核心调度函数
//
// 功能: 执行 C = A @ B_dequant 的量化矩阵乘法
// 流程:
//   1. 如果有 act_order，先调用 permute_cols_kernel 对 A 的列重排
//   2. 获取 GPU 的共享内存大小和计算能力
//   3. 当 M 较大时，将 M 维度拆分为多个 block 并行处理
//   4. 对每个 M block:
//      a. determine_exec_config 选择最优线程配置
//      b. get_marlin_kernel 获取对应的 kernel 函数指针
//      c. 启动 kernel 执行矩阵乘法
//
// M 维度分块策略:
//   - max_thread_m_blocks = 4: 每个 thread block 最多处理 4*16=64 行
//   - 当 M > 64 时，启动多个 thread block 并行处理不同行
//   - 使用 locks 数组进行跨 block 同步 (用于全局归约)
//
// SM 版本适配:
//   - SM75 (Turing): 2 阶段 pipeline，只支持 FP16/INT8 激活
//   - SM80+ (Ampere): 4 阶段 pipeline，支持所有数据类型
//   - SM89 (Ada): 额外支持 FP8 激活 (W4A8-FP8)
void marlin_mm(const void* A, const void* B, void* C, void* C_tmp, void* b_bias,
               void* a_s, void* b_s, void* g_s, void* zp, void* g_idx,
               void* perm, void* a_tmp, int prob_m, int prob_n, int prob_k,
               int lda, void* workspace, vllm::ScalarType const& a_type,
               vllm::ScalarType const& b_type, vllm::ScalarType const& c_type,
               vllm::ScalarType const& s_type, bool has_bias,
               bool has_act_order, bool is_k_full, bool has_zp, int num_groups,
               int group_size, int dev, cudaStream_t stream, int thread_k_init,
               int thread_n_init, int sms, bool use_atomic_add,
               bool use_fp32_reduce, bool is_zp_float) {
  bool is_a_8bit = a_type.size_bits() == 8;
  TORCH_CHECK(prob_m > 0 && prob_n > 0 && prob_k > 0, "Invalid MNK = [", prob_m,
              ", ", prob_n, ", ", prob_k, "]");

  int group_blocks = 0;
  if (has_act_order) {
    if (is_k_full) {
      TORCH_CHECK(group_size != -1);
      group_blocks = group_size / 16;
      TORCH_CHECK(prob_k % group_blocks == 0, "prob_k = ", prob_k,
                  " is not divisible by group_blocks = ", group_blocks);
    } else {
      TORCH_CHECK(group_size == 0);
      group_blocks = 0;
    }
  } else {
    if (group_size == -1) {
      group_blocks = -1;
    } else {
      group_blocks = group_size / 16;
      TORCH_CHECK(prob_k % group_blocks == 0, "prob_k = ", prob_k,
                  " is not divisible by group_blocks = ", group_blocks);
    }
  }

  int num_bits = b_type.size_bits();
  const int4* A_ptr = (const int4*)A;
  const int4* B_ptr = (const int4*)B;
  int4* C_ptr = (int4*)C;
  int4* C_tmp_ptr = (int4*)C_tmp;

  const int4* bias_ptr = (const int4*)b_bias;
  const float* a_s_ptr = (const float*)a_s;
  const int4* b_s_ptr = (const int4*)b_s;
  const float* g_s_ptr = (const float*)g_s;

  const int4* zp_ptr = (const int4*)zp;
  const int* g_idx_ptr = (const int*)g_idx;
  const int* perm_ptr = (const int*)perm;
  int4* a_tmp_ptr = (int4*)a_tmp;
  int* locks = (int*)workspace;

  if (has_act_order) {
    // Permute A columns
    int block_rows = div_ceil(prob_m, sms);
    // avoid ">>>" being formatted to "> > >"
    // clang-format off
    permute_cols_kernel<<<sms, default_threads, 0, stream>>>(
        A_ptr, perm_ptr, a_tmp_ptr, prob_m, prob_k, lda, block_rows);
    // clang-format on
    A_ptr = a_tmp_ptr;
    lda = prob_k;

    // If we have a full K, then we can run the non-act-order version of Marlin
    // (since the weight rows are reordered by increasing group ids, and by
    // having a full K, we have full original groups)
    if (is_k_full) has_act_order = false;
  }

  int max_shared_mem = 0;
  cudaDeviceGetAttribute(&max_shared_mem,
                         cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  TORCH_CHECK(max_shared_mem > 0);

  int major_capability, minor_capability;
  cudaDeviceGetAttribute(&major_capability, cudaDevAttrComputeCapabilityMajor,
                         dev);
  cudaDeviceGetAttribute(&minor_capability, cudaDevAttrComputeCapabilityMinor,
                         dev);
  TORCH_CHECK(major_capability * 10 + minor_capability >= 75,
              "marlin kernel only support Turing or newer GPUs.");
  int stages = 4;
  if (major_capability == 7 && minor_capability == 5) {
    stages = 2;
    TORCH_CHECK(a_type == vllm::kFloat16 || a_type == vllm::kS8,
                "Turing only support FP16 or INT8 activation.");
  }
  if (a_type == vllm::kFE4M3fn) {
    TORCH_CHECK(major_capability * 10 + minor_capability >= 89,
                "FP8 only support Ada Lovelace or newer GPUs.");
    TORCH_CHECK(
        major_capability * 10 + minor_capability == 89 ||
            major_capability == 12,
        "Marlin W4A8-FP8 only support SM89 or SM12x device (It is slower than "
        "Marlin W4A16 on other devices).");
  }

  int max_par = 16;
  if (prob_n <= 4096) max_par = 16 * 8;
  int max_shared_mem_new = max_shared_mem;
  int rest_m = prob_m;
  int max_thread_m_blocks = 4;
  while (rest_m) {
    int par_count = rest_m / (max_thread_m_blocks * 16);
    if (par_count > max_par) par_count = max_par;
    int prob_m_split =
        par_count > 0 ? (par_count * (max_thread_m_blocks * 16)) : rest_m;

    int thread_k = thread_k_init;
    int thread_n = thread_n_init;

    int thread_m_blocks = min(div_ceil(prob_m_split, 16), max_thread_m_blocks);
    int m_block_size_8 = prob_m_split <= 8 && a_type.size_bits() == 16;

    // Set thread config
    exec_config_t exec_cfg;
    thread_config_t thread_tfg;
    if (thread_k != -1 && thread_n != -1) {
      thread_tfg = thread_config_t{thread_k, thread_n, default_threads};
      exec_cfg = exec_config_t{1, thread_tfg};
      TORCH_CHECK(prob_n % thread_n == 0, "prob_n = ", prob_n,
                  " is not divisible by thread_n = ", thread_n);
      TORCH_CHECK(prob_k % thread_k == 0, "prob_k = ", prob_k,
                  " is not divisible by thread_k = ", thread_k);
    } else {
      // Auto config
      exec_cfg = determine_exec_config(
          a_type, b_type, c_type, s_type, prob_m_split, prob_n, prob_k,
          thread_m_blocks, m_block_size_8, num_bits, group_size, has_act_order,
          is_k_full, has_zp, is_zp_float, is_a_8bit, stages, max_shared_mem,
          sms);
      thread_tfg = exec_cfg.tb_cfg;
      if (thread_tfg.thread_n != -1) {
        if (prob_n / thread_tfg.thread_n *
                div_ceil(prob_m_split, thread_m_blocks * 16) * 4 <=
            sms) {
          if (is_valid_config({128, 64, 128}, thread_m_blocks, prob_m_split,
                              prob_n, prob_k, num_bits, group_size,
                              has_act_order, is_k_full, has_zp, is_zp_float,
                              is_a_8bit, stages, max_shared_mem_new)) {
            thread_tfg = {128, 64, 128};
            exec_cfg = {1, thread_tfg};
          }
        }
      }

      if (thread_tfg.thread_k == -1 && max_thread_m_blocks > 1) {
        max_thread_m_blocks--;
        continue;
      }
    }

    int num_threads = thread_tfg.num_threads;
    thread_k = thread_tfg.thread_k;
    thread_n = thread_tfg.thread_n;
    int blocks = sms * exec_cfg.blocks_per_sm;
    if (exec_cfg.blocks_per_sm > 1)
      max_shared_mem_new = max_shared_mem / exec_cfg.blocks_per_sm - 1024;

    int thread_k_blocks = thread_k / 16;
    int thread_n_blocks = thread_n / 16;

    TORCH_CHECK(
        is_valid_config(thread_tfg, thread_m_blocks, prob_m_split, prob_n,
                        prob_k, num_bits, group_size, has_act_order, is_k_full,
                        has_zp, is_zp_float, is_a_8bit, stages,
                        max_shared_mem_new),
        "Invalid thread config: thread_m_blocks = ", thread_m_blocks,
        ", thread_k = ", thread_tfg.thread_k,
        ", thread_n = ", thread_tfg.thread_n,
        ", num_threads = ", thread_tfg.num_threads, " for MKN = [", prob_m,
        ", ", prob_k, ", ", prob_n, "] and num_bits = ", num_bits,
        ", prob_m_split = ", prob_m_split, ", group_size = ", group_size,
        ", has_act_order = ", has_act_order, ", is_k_full = ", is_k_full,
        ", has_zp = ", has_zp, ", is_zp_float = ", is_zp_float,
        ", stages = ", stages, ", max_shared_mem_new = ", max_shared_mem_new);

    auto kernel = get_marlin_kernel(
        a_type, b_type, c_type, s_type, thread_m_blocks, thread_n_blocks,
        thread_k_blocks, m_block_size_8, has_act_order, has_zp, group_blocks,
        num_threads, is_zp_float, stages);

    if (kernel == MarlinDefault) {
      TORCH_CHECK(false, "Unsupported shapes: MNK = [", prob_m, ", ", prob_n,
                  ", ", prob_k, "]", ", has_act_order = ", has_act_order,
                  ", num_groups = ", num_groups, ", group_size = ", group_size,
                  ", prob_m_split = ", prob_m_split,
                  ", thread_m_blocks = ", thread_m_blocks,
                  ", thread_n_blocks = ", thread_n_blocks,
                  ", thread_k_blocks = ", thread_k_blocks,
                  ", num_threads = ", num_threads, ", num_bits = ", num_bits);
    }

    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         max_shared_mem_new);

    bool part_use_atomic_add =
        use_atomic_add && div_ceil(prob_m_split, 64) * prob_n <= 2048;

    // avoid ">>>" being formatted to "> > >"
    // clang-format off
    kernel<<<blocks, num_threads, max_shared_mem_new, stream>>>(
        A_ptr, B_ptr, C_ptr, C_tmp_ptr, bias_ptr, a_s_ptr, b_s_ptr, g_s_ptr, zp_ptr,
        g_idx_ptr, num_groups,
        prob_m_split, prob_n, prob_k, lda, locks, has_bias, part_use_atomic_add,
        use_fp32_reduce, max_shared_mem_new);
    // clang-format on

    bool is_a_8bit = a_type.size_bits() == 8;
    A_ptr += prob_m_split * (lda / (is_a_8bit ? 16 : 8));
    a_s_ptr += prob_m_split;
    C_ptr += prob_m_split * (prob_n / 8);
    rest_m -= prob_m_split;
  }
}

}  // namespace marlin

// 中文注释: Marlin GEMM 的 Python 绑定入口函数
// 从 PyTorch 张量中提取数据指针和参数，进行校验后调用 marlin_mm
//
// 参数说明:
//   a: 输入激活矩阵 [M, K]，支持 FP16/BF16/FP8/INT8
//   c_or_none: 可选的输出矩阵 [M, N]，如果为 None 则自动分配
//   b_q_weight: 量化权重矩阵，packed 格式
//   b_bias_or_none: 可选的偏置向量 [N]
//   b_scales: 权重量化缩放因子 [num_groups, N]
//   a_scales_or_none: 激活缩放因子 [M] (仅 8-bit 激活时使用)
//   global_scale_or_none: 全局缩放因子 (仅 NVFP4 格式)
//   b_zeros_or_none: 零点 (非对称量化时使用)
//   g_idx_or_none: 分组索引 (act_order 时使用)
//   perm_or_none: 列重排索引 (act_order 时使用)
//   workspace: 工作空间，用于跨 block 同步
//   b_type_id: 权重数据类型 ID
//   size_m/n/k: 矩阵乘法维度
//   is_k_full: K 维度是否完整 (影响 act_order 的处理方式)
//   use_atomic_add: 是否使用 atomicAdd 归约
//   use_fp32_reduce: 是否使用 FP32 全局归约
//   is_zp_float: 零点是否为 float 类型
torch::Tensor marlin_gemm(
    torch::Tensor& a, std::optional<torch::Tensor> c_or_none,
    torch::Tensor& b_q_weight,
    std::optional<torch::Tensor> const& b_bias_or_none, torch::Tensor& b_scales,
    std::optional<torch::Tensor> const& a_scales_or_none,
    std::optional<torch::Tensor> const& global_scale_or_none,
    std::optional<torch::Tensor> const& b_zeros_or_none,
    std::optional<torch::Tensor> const& g_idx_or_none,
    std::optional<torch::Tensor> const& perm_or_none, torch::Tensor& workspace,
    vllm::ScalarTypeId const& b_type_id, int64_t size_m, int64_t size_n,
    int64_t size_k, bool is_k_full, bool use_atomic_add, bool use_fp32_reduce,
    bool is_zp_float) {
  vllm::ScalarTypeId a_type_id, c_type_id, s_type_id;

  auto c_dtype = a.dtype();
  if (a.scalar_type() == at::ScalarType::Half) {
    a_type_id = vllm::kFloat16.id();
    c_type_id = vllm::kFloat16.id();
  } else if (a.scalar_type() == at::ScalarType::BFloat16) {
    a_type_id = vllm::kBFloat16.id();
    c_type_id = vllm::kBFloat16.id();
  } else {
    c_dtype = b_scales.dtype();
    if (b_scales.scalar_type() == at::ScalarType::Half) {
      c_type_id = vllm::kFloat16.id();
    } else if (b_scales.scalar_type() == at::ScalarType::BFloat16) {
      c_type_id = vllm::kBFloat16.id();
    } else {
      c_type_id = vllm::kBFloat16.id();

      TORCH_CHECK(c_or_none.has_value(), "c must be passed for W4A8-FP4");
      torch::Tensor c = c_or_none.value();
      c_dtype = c.dtype();

      if (c.scalar_type() == at::ScalarType::Half) {
        c_type_id = vllm::kFloat16.id();
      } else if (c.scalar_type() == at::ScalarType::BFloat16) {
        c_type_id = vllm::kBFloat16.id();
      } else {
        TORCH_CHECK(false, "unsupported c dtype");
      }
    }

    if (a.scalar_type() == at::ScalarType::Float8_e4m3fn) {
      a_type_id = vllm::kFE4M3fn.id();
    } else if (a.scalar_type() == at::ScalarType::Char) {
      a_type_id = vllm::kS8.id();
    } else {
      TORCH_CHECK(false, "unsupported `a` scalar_type");
    }
  }

  s_type_id = c_type_id;
  if (b_type_id == vllm::kFE2M1f.id()) {
    if (b_scales.scalar_type() == at::ScalarType::Float8_e4m3fn) {
      s_type_id = vllm::kFE4M3fn.id();
    } else if (b_scales.scalar_type() == at::ScalarType::Float8_e8m0fnu) {
      s_type_id = vllm::kFE8M0fnu.id();
    } else {
      TORCH_CHECK(false,
                  "When b_type = float4_e2m1f, b_scale scalar type must be",
                  "float8_e4m3fn (for NVFP4) or float8_e8m0fnu (for MXFP4).");
    }
  } else if (b_type_id == vllm::kFE4M3fn.id() &&
             b_scales.scalar_type() == at::ScalarType::Float8_e8m0fnu) {
    s_type_id = vllm::kFE8M0fnu.id();
  }

  vllm::ScalarType a_type = vllm::ScalarType::from_id(a_type_id);
  vllm::ScalarType b_type = vllm::ScalarType::from_id(b_type_id);
  vllm::ScalarType c_type = vllm::ScalarType::from_id(c_type_id);
  vllm::ScalarType s_type = vllm::ScalarType::from_id(s_type_id);

  int pack_factor = 32 / b_type.size_bits();

  // Verify A
  TORCH_CHECK(a.size(0) == size_m, "Shape mismatch: a.size(0) = ", a.size(0),
              ", size_m = ", size_m);
  TORCH_CHECK(a.size(1) == size_k, "Shape mismatch: a.size(1) = ", a.size(1),
              ", size_k = ", size_k);

  // Verify B
  TORCH_CHECK(
      size_k % MARLIN_NAMESPACE_NAME::tile_size == 0, "size_k = ", size_k,
      " is not divisible by tile_size = ", MARLIN_NAMESPACE_NAME::tile_size);
  TORCH_CHECK((size_k / MARLIN_NAMESPACE_NAME::tile_size) == b_q_weight.size(0),
              "Shape mismatch: b_q_weight.size(0) = ", b_q_weight.size(0),
              ", size_k = ", size_k,
              ", tile_size = ", MARLIN_NAMESPACE_NAME::tile_size);
  TORCH_CHECK(
      b_q_weight.size(1) % MARLIN_NAMESPACE_NAME::tile_size == 0,
      "b_q_weight.size(1) = ", b_q_weight.size(1),
      " is not divisible by tile_size = ", MARLIN_NAMESPACE_NAME::tile_size);
  int actual_size_n =
      (b_q_weight.size(1) / MARLIN_NAMESPACE_NAME::tile_size) * pack_factor;
  TORCH_CHECK(size_n == actual_size_n, "size_n = ", size_n,
              ", actual_size_n = ", actual_size_n);

  // Verify device and strides
  TORCH_CHECK(a.device().is_cuda(), "A is not on GPU");
  TORCH_CHECK(a.stride(1) == 1, "A.stride(1) is not 1");
  // We use int4 (16 bytes) to load A, so A must aligned to 16 bytes
  TORCH_CHECK(a.stride(0) % 8 == 0, "A.stride(0) must divisible by 8");
  TORCH_CHECK(((uint64_t)a.data_ptr()) % 16 == 0, "A must aligned to 16 bytes");

  TORCH_CHECK(b_q_weight.device().is_cuda(), "b_q_weight is not on GPU");
  TORCH_CHECK(b_q_weight.is_contiguous(), "b_q_weight is not contiguous");

  TORCH_CHECK(b_scales.device().is_cuda(), "b_scales is not on GPU");
  TORCH_CHECK(b_scales.is_contiguous(), "b_scales is not contiguous");

  torch::Tensor a_scales;
  auto options = torch::TensorOptions().dtype(c_dtype).device(a.device());
  auto options_fp32 =
      torch::TensorOptions().dtype(at::kFloat).device(a.device());

  if (a_scales_or_none.has_value()) {
    a_scales = a_scales_or_none.value();
    TORCH_CHECK(a_type.size_bits() == 8,
                "a_scales can only be used for 8bit activation.");
  } else {
    a_scales = torch::empty({0}, options_fp32);
    TORCH_CHECK(a_type.size_bits() != 8,
                "the a_scales parameter must be passed for 8bit activation.");
  }

  // thread_k: `k` size of a thread_tile in `weights` (can usually be left as
  // auto -1)
  int thread_k = -1;
  // thread_n: `n` size of a thread_tile in `weights` (can usually be left as
  // auto -1)
  int thread_n = -1;
  // sms: number of SMs to use for the kernel
  int sms = -1;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, a.get_device());

  // Alloc buffers
  const at::cuda::OptionalCUDAGuard device_guard(device_of(a));
  torch::Tensor c;
  if (c_or_none.has_value()) {
    c = c_or_none.value();
    TORCH_CHECK(c.device().is_cuda(), "c is not on GPU");
    TORCH_CHECK(c.is_contiguous(), "c is not contiguous");
    TORCH_CHECK(c.size(0) == size_m, "Shape mismatch: c.size(0) = ", c.size(0),
                ", size_m = ", size_m);
    TORCH_CHECK(c.size(1) == size_n, "Shape mismatch: c.size(1) = ", c.size(1),
                ", size_n = ", size_n);
  } else {
    c = torch::empty({size_m, size_n}, options);
  }
  if (size_m == 0) return c;

  // Alloc C tmp buffer that is going to be used for the global reduce
  torch::Tensor c_tmp;
  if (use_fp32_reduce) {
    int max_m_block_size = (size_m + 16 - 1) / 16 * 16;
    max_m_block_size = min(max_m_block_size, 64);
    int max_c_tmp_size =
        sms * max_m_block_size * MARLIN_NAMESPACE_NAME::max_thread_n;
    c_tmp = torch::empty({max_c_tmp_size}, options_fp32);
  } else {
    c_tmp = torch::empty({0}, options_fp32);
  }

  // Detect groupsize and act_order
  int num_groups = -1;
  int group_size = -1;

  int rank = b_scales.sizes().size();
  TORCH_CHECK(rank == 2, "b_scales rank = ", rank, " is not 2");
  TORCH_CHECK(b_scales.size(1) == size_n, "b_scales dim 1 = ", b_scales.size(1),
              " is not size_n = ", size_n);
  num_groups = b_scales.size(0);

  torch::Tensor g_idx, perm, a_tmp;
  if (g_idx_or_none.has_value() && perm_or_none.has_value()) {
    g_idx = g_idx_or_none.value();
    perm = perm_or_none.value();

    TORCH_CHECK(g_idx.device().is_cuda(), "g_idx is not on GPU");
    TORCH_CHECK(g_idx.is_contiguous(), "g_idx is not contiguous");
    TORCH_CHECK(perm.device().is_cuda(), "perm is not on GPU");
    TORCH_CHECK(perm.is_contiguous(), "perm is not contiguous");

    // Verify g_idx and perm
    TORCH_CHECK((g_idx.size(-1) == 0 && perm.size(-1) == 0) ||
                    (g_idx.size(-1) == size_k && perm.size(-1) == size_k),
                "Unexpected g_idx.size(-1) = ", g_idx.size(-1),
                " and perm.size(-1) = ", perm.size(-1),
                ", where size_k = ", size_k);
  } else {
    g_idx = torch::empty({0}, options);
    perm = torch::empty({0}, options);
    a_tmp = torch::empty({0}, options);
  }
  bool has_act_order = g_idx.size(-1) > 0 && perm.size(-1) > 0;

  if (has_act_order) {
    a_tmp = torch::empty({size_m, size_k}, options);
    if (is_k_full) {
      TORCH_CHECK(num_groups > 1, "For act_order, num_groups must be > 1");
      TORCH_CHECK(size_k % num_groups == 0, "size_k = ", size_k,
                  ", is not divisible by num_groups = ", num_groups);
      group_size = size_k / num_groups;
    } else {
      group_size = 0;
    }

  } else {
    a_tmp = torch::empty({0}, options);
    if (num_groups > 1) {
      TORCH_CHECK(
          size_k % num_groups == 0, "size_k = ", size_k,
          ", is not divisible by b_scales.size(0) = ", b_scales.size(0));
      group_size = size_k / num_groups;
    } else {
      group_size = -1;
    }
  }

  torch::Tensor global_scale;
  if (global_scale_or_none.has_value()) {
    global_scale = global_scale_or_none.value();
    TORCH_CHECK(b_type == vllm::kFE2M1f && s_type == vllm::kFE4M3fn,
                "global_scale can only be used for nvfp4 format.");
  } else {
    global_scale = torch::empty({0}, options_fp32);
    TORCH_CHECK(!(b_type == vllm::kFE2M1f && s_type == vllm::kFE4M3fn),
                "the global_scale parameter must be passed for nvfp4 format.");
  }

  bool has_bias = b_bias_or_none.has_value();
  torch::Tensor b_bias;
  if (has_bias) {
    b_bias = b_bias_or_none.value();
    TORCH_CHECK(b_bias.device().is_cuda(), "b_bias is not on GPU");
    TORCH_CHECK(b_bias.is_contiguous(), "b_bias is not contiguous");
    TORCH_CHECK(b_bias.size(0) == size_n, "b_bias.size(0) != size_n");
    TORCH_CHECK(b_bias.stride(0) == 1, "b_bias.stride(0) != 1");
  } else {
    b_bias = torch::empty({0}, options);
  }

  torch::Tensor b_zeros;
  if (b_zeros_or_none.has_value()) {
    b_zeros = b_zeros_or_none.value();
    TORCH_CHECK(b_zeros.device().is_cuda(), "b_zeros is not on GPU");
    TORCH_CHECK(b_zeros.is_contiguous(), "b_zeros is not contiguous");
  } else {
    b_zeros = torch::empty({0}, options);
  }
  bool has_zp = b_zeros.size(-1) > 0;
  if (has_zp) {
    TORCH_CHECK(
        b_type == vllm::kU4 || b_type == vllm::kU8,
        "b_type must be u4 or u8 when has_zp = True. Got = ", b_type.str());
  } else {
    TORCH_CHECK(b_type == vllm::kU4B8 || b_type == vllm::kU8B128 ||
                    b_type == vllm::kS4 || b_type == vllm::kS8 ||
                    b_type == vllm::kFE4M3fn || b_type == vllm::kFE2M1f,
                "b_type must be uint4b8, uint8b128, int4, int8, "
                "float8_e4m3fn or float4_e2m1f when has_zp = False. Got = ",
                b_type.str());
  }

  if (has_zp && is_zp_float) {
    TORCH_CHECK(a.scalar_type() == at::ScalarType::Half,
                "Computation type must be float16 (half) when using float zero "
                "points.");
  }

  // Verify b_zeros
  if (has_zp) {
    int rank = b_zeros.sizes().size();
    TORCH_CHECK(rank == 2, "b_zeros rank = ", rank, " is not 2");
    if (is_zp_float) {
      TORCH_CHECK(b_zeros.size(1) == size_n,
                  "b_zeros dim 1 = ", b_zeros.size(1),
                  " is not size_n = ", size_n);
      TORCH_CHECK(num_groups == b_zeros.size(0),
                  "b_zeros dim 0 = ", b_zeros.size(0),
                  " is not num_groups = ", num_groups);
      TORCH_CHECK(num_groups != -1, "num_groups must be != -1");
    } else {
      TORCH_CHECK(b_zeros.size(0) == num_groups,
                  "b_zeros dim 0 = ", b_zeros.size(0),
                  " is not num_groups = ", num_groups);
      TORCH_CHECK(b_zeros.size(1) == size_n / pack_factor,
                  "b_zeros dim 1 = ", b_zeros.size(1),
                  " is not size_n / pack_factor = ", size_n / pack_factor);
    }
  }

  // Verify workspace size
  TORCH_CHECK(size_n % MARLIN_NAMESPACE_NAME::min_thread_n == 0,
              "size_n = ", size_n, ", is not divisible by min_thread_n = ",
              MARLIN_NAMESPACE_NAME::min_thread_n);

  int min_workspace_size = sms;
  TORCH_CHECK(workspace.numel() >= min_workspace_size,
              "workspace.numel = ", workspace.numel(),
              " is below min_workspace_size = ", min_workspace_size);

  int dev = a.get_device();

  TORCH_CHECK(a_scales.scalar_type() == at::ScalarType::Float,
              "scalar type of a_scales must be float");
  TORCH_CHECK(global_scale.scalar_type() == at::ScalarType::Float,
              "scalar type of global_scale must be float");
  if (a_type.size_bits() == 16) {
    TORCH_CHECK(
        a.scalar_type() == c.scalar_type(),
        "scalar type of a must be the same with c for 16 bit activation");
  }

  marlin::marlin_mm(
      a.data_ptr(), b_q_weight.data_ptr(), c.data_ptr(), c_tmp.data_ptr(),
      b_bias.data_ptr(), a_scales.data_ptr(), b_scales.data_ptr(),
      global_scale.data_ptr(), b_zeros.data_ptr(), g_idx.data_ptr(),
      perm.data_ptr(), a_tmp.data_ptr(), size_m, size_n, size_k, a.stride(0),
      workspace.data_ptr(), a_type, b_type, c_type, s_type, has_bias,
      has_act_order, is_k_full, has_zp, num_groups, group_size, dev,
      at::cuda::getCurrentCUDAStream(dev), thread_k, thread_n, sms,
      use_atomic_add, use_fp32_reduce, is_zp_float);

  return c;
}

#endif

TORCH_LIBRARY_IMPL_EXPAND(TORCH_EXTENSION_NAME, CUDA, m) {
  m.impl("marlin_gemm", &marlin_gemm);
}
