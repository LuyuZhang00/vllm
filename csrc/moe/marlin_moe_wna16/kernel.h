// =============================================================================
// 中文注释：Marlin MoE WNA16 量化 GEMM kernel 头文件
//
// 本文件声明了 Marlin MoE WNA16 量化矩阵乘法 kernel 的模板和参数。
//
// Marlin 是一种高性能的量化矩阵乘法实现，支持：
// - 4-bit / 2-bit 权重量化（WNA16）
// - 分组量化（group_blocks 控制量化组大小）
// - 零点支持（int 或 float16 类型）
// - 异步全局内存到共享内存的流水线预取（stages 控制流水线级数）
//
// MARLIN_KERNEL_PARAMS 宏定义了 kernel 的所有参数：
// - A: 激活矩阵（int4 打包）
// - B: 量化权重矩阵（int4 打包）
// - C/C_tmp: 输出矩阵 / 临时输出矩阵
// - scales_ptr/zp_ptr: 量化缩放因子和零点
// - sorted_token_ids_ptr/expert_ids_ptr: MoE 排序后的 token 和专家 ID
// - topk_weights_ptr: top-k 权重（可选乘到输出上）
// - locks: 用于跨 block 同步的锁数组
// =============================================================================

#ifndef MARLIN_NAMESPACE_NAME
  #define MARLIN_NAMESPACE_NAME marlin_moe_wna16
#endif

#include "quantization/marlin/marlin.cuh"
#include "quantization/marlin/marlin_dtypes.cuh"
#include "core/scalar_type.hpp"

#define MARLIN_KERNEL_PARAMS                                          \
  const int4 *__restrict__ A, const int4 *__restrict__ B,             \
      int4 *__restrict__ C, int4 *__restrict__ C_tmp,                 \
      const int4 *__restrict__ b_bias_ptr,                            \
      const float *__restrict__ a_scales_ptr,                         \
      const int4 *__restrict__ scales_ptr,                            \
      const float *__restrict__ global_scale_ptr,                     \
      const int4 *__restrict__ zp_ptr, const int *__restrict__ g_idx, \
      const int32_t *__restrict__ sorted_token_ids_ptr,               \
      const int32_t *__restrict__ expert_ids_ptr,                     \
      const int32_t *__restrict__ num_tokens_past_padded_ptr,         \
      const float *__restrict__ topk_weights_ptr, int top_k,          \
      bool mul_topk_weights, int num_groups, int prob_m, int prob_n,  \
      int prob_k, int *locks, bool has_bias, bool use_atomic_add,     \
      bool use_fp32_reduce

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
