// =============================================================================
// 中文注释：MoE Token 排列/反排列 kernel 头文件
//
// 本头文件声明了 MoE token 排列和反排列操作所需的函数和类：
//
// 1. CubKeyValueSorter —— CUB radix sort 封装类。
//    用于将 (expert_id, token_index) 对排序，使得同一专家的 token 连续存放。
//    内部维护了 keys_out/values_out 缓冲区和 num_bits_ 参数。
//
// 2. sortAndScanExpert —— 排列 + 前缀和计算函数。
//    使用 CubKeyValueSorter 对 token 按专家排序，然后计算每个专家的
//    token 起始偏移（expert_first_token_offset）。
//
// 3. expandInputRowsKernelLauncher —— token 排列 kernel。
//    将输入 token 按排序后的顺序复制到输出缓冲区。
//    同时记录逆排列索引（inv_permuted_idx），用于后续 unpermute。
//
// 4. finalizeMoeRoutingKernelLauncher —— token 反排列 + 加权求和 kernel。
//    将专家输出还原到原始 token 顺序，并按 topk_weights 加权求和。
//
// 5. preprocessTopkIdLauncher —— 专家 ID 预处理函数。
//    当使用 expert parallelism 时，将全局专家 ID 映射为本地专家 ID。
//
// 实际 kernel 实现在 moe_permute_unpermute_kernel.inl 文件中。
// =============================================================================

#pragma once
// reference from tensorrt_llm moe kernel implementation archive in
// https://github.com/BBuf/tensorrt-llm-moe/tree/master

#include <c10/core/ScalarType.h>
#include <torch/all.h>
#include "dispatch.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/util_type.cuh>
#include "cutlass/numeric_size.h"
#include "cutlass/array.h"

template <typename T>
inline T* get_ptr(torch::Tensor& t) {
  return reinterpret_cast<T*>(t.data_ptr());
}

template <typename T>
inline const T* get_ptr(const torch::Tensor& t) {
  return reinterpret_cast<const T*>(t.data_ptr());
}

class CubKeyValueSorter {
 public:
  CubKeyValueSorter();

  CubKeyValueSorter(int const num_experts);

  void updateNumExperts(int const num_experts);

  static size_t getWorkspaceSize(size_t const num_key_value_pairs,
                                 int const num_experts);

  void run(void* workspace, size_t const workspace_size, int const* keys_in,
           int* keys_out, int const* values_in, int* values_out,
           size_t const num_key_value_pairs, cudaStream_t stream);

 private:
  static int expertsToBits(int experts);
  int num_experts_;
  int num_bits_;
};

void computeExpertFirstTokenOffset(int const* sorted_indices,
                                   int const total_indices,
                                   int const num_experts,
                                   int64_t* expert_first_token_offset,
                                   cudaStream_t stream);

void sortAndScanExpert(const int* expert_for_source_row, const int* source_rows,
                       int* permuted_experts, int* permuted_rows,
                       int64_t* expert_first_token_offset, int num_rows,
                       int num_experts, int num_experts_per_node, int k,
                       CubKeyValueSorter& sorter, void* sorter_ws,
                       cudaStream_t stream);

template <typename T>
void expandInputRowsKernelLauncher(
    T const* unpermuted_input, T* permuted_output,
    int const* expanded_dest_row_to_expanded_source_row,
    int* expanded_source_row_to_expanded_dest_row, int* permuted_idx,
    int64_t const* expert_first_token_offset, int64_t const num_rows,
    int64_t const* num_valid_tokens_ptr, int64_t const cols, int const k,
    int num_local_experts, cudaStream_t stream);

template <class T, class OutputType>
void finalizeMoeRoutingKernelLauncher(
    T const* expanded_permuted_rows, OutputType* reduced_unpermuted_output,
    float const* scales, int const* expanded_source_row_to_expanded_dest_row,
    int64_t const num_rows, int64_t const cols, int64_t const k,
    int64_t const* num_valid_ptr, cudaStream_t stream);

void preprocessTopkIdLauncher(int* topk_id_ptr, int size,
                              const int* expert_map_ptr, int num_experts,
                              cudaStream_t stream);

#include "moe_permute_unpermute_kernel.inl"
