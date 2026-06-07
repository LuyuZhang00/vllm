
// =============================================================================
// 中文注释: Marlin Kernel 参数定义和声明
// =============================================================================
// 本文件定义了 Marlin kernel 的统一参数列表宏和 kernel 函数声明。
//
// MARLIN_KERNEL_PARAMS 宏定义了所有 Marlin kernel 变体共享的参数:
//   - A: 输入激活矩阵 (fp16/bf16/fp8/int8)，形状 [m, k]
//   - B: 量化权重矩阵 (4bit/8bit packed)，形状 [k/pack_factor, n]
//   - C: 输出矩阵 (fp16/bf16)，形状 [m, n]
//   - C_tmp: FP32 临时输出缓冲区，用于跨 thread block 的全局归约
//   - b_bias_ptr: 可选的偏置向量
//   - a_scales_ptr: 激活的缩放因子 (仅 8-bit 激活时使用)
//   - scales_ptr: 权重量化缩放因子，形状 [k/groupsize, n]
//   - global_scale_ptr: 全局缩放因子 (仅 NVFP4 格式使用)
//   - zp_ptr: 零点 (zero point)，用于非对称量化
//   - g_idx: 分组索引 (activation reordering 时使用)
//   - num_groups: 每个输出通道的缩放组数
//   - prob_m/n/k: 矩阵乘法维度 M, N, K
//   - lda: A 矩阵的 leading dimension (stride)
//   - locks: 全局锁数组，用于跨 thread block 同步
//   - has_bias: 是否有偏置
//   - use_atomic_add: 是否使用 atomicAdd 进行归约
//   - use_fp32_reduce: 是否使用 FP32 全局归约
//   - max_shared_mem: 可用的最大共享内存大小
//
// Marlin kernel 模板参数:
//   - a_type_id/b_type_id/c_type_id/s_type_id: 数据类型 ID
//   - threads: 线程数 (通常 256 = 8 warps)
//   - thread_m_blocks: M 维度的 16x16 block 数 (batch 大小)
//   - thread_n_blocks: N 维度的 16x16 block 数 (输出维度)
//   - thread_k_blocks: K 维度的 16x16 block 数 (归约维度)
//   - m_block_size_8: 是否使用 8 元素的 M block (小 batch 优化)
//   - stages: 异步流水线阶段数
//   - group_blocks: 量化分组大小 (以 16x16 block 为单位)
//   - is_zp_float: 零点是否为 float16 类型
// =============================================================================

#ifndef MARLIN_NAMESPACE_NAME
  #define MARLIN_NAMESPACE_NAME marlin
#endif

#include "marlin.cuh"
#include "marlin_dtypes.cuh"
#include "core/scalar_type.hpp"

#define MARLIN_KERNEL_PARAMS                                                   \
  const int4 *__restrict__ A, const int4 *__restrict__ B,                      \
      int4 *__restrict__ C, int4 *__restrict__ C_tmp,                          \
      const int4 *__restrict__ b_bias_ptr,                                     \
      const float *__restrict__ a_scales_ptr,                                  \
      const int4 *__restrict__ scales_ptr,                                     \
      const float *__restrict__ global_scale_ptr,                              \
      const int4 *__restrict__ zp_ptr, const int *__restrict__ g_idx,          \
      int num_groups, int prob_m, int prob_n, int prob_k, int lda, int *locks, \
      bool has_bias, bool use_atomic_add, bool use_fp32_reduce,                \
      int max_shared_mem

namespace MARLIN_NAMESPACE_NAME {
template <const vllm::ScalarTypeId a_type_id,  // A ScalarType id
          const vllm::ScalarTypeId b_type_id,  // B ScalarType id
          const vllm::ScalarTypeId c_type_id,  // C ScalarType id
          const vllm::ScalarTypeId s_type_id,  // B_SCALE ScalarType id
          const int threads,          // number of threads in a threadblock
          const int thread_m_blocks,  // number of 16x16 blocks in the m
                                      // dimension (batchsize) of the
                                      // threadblock
          const int thread_n_blocks,  // same for n dimension (output)
          const int thread_k_blocks,  // same for k dimension (reduction)
          const bool m_block_size_8,  // whether m_block_size == 8
                                      // only works when thread_m_blocks == 1
          const int stages,  // number of stages for the async global->shared
                             // fetch pipeline
          const int group_blocks,  // number of consecutive 16x16 blocks
                                   // with a separate quantization scale
          const bool is_zp_float   // is zero point of float16 type?
          >
__global__ void Marlin(MARLIN_KERNEL_PARAMS);

}
