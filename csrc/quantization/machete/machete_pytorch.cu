#include "machete_mm_launcher.cuh"
#include "machete_prepack_launcher.cuh"
#include "core/scalar_type.hpp"

#include "core/registration.h"

// =============================================================================
// 中文注释: Machete 量化 GEMM 的 Python/C++ 绑定层
// =============================================================================
// Machete 是另一种高性能量化 GEMM 实现，与 Marlin 类似但有不同的优化策略。
// 本文件提供 Machete 的 Python 绑定接口。
//
// 主要功能:
//   1. supported_schedules(): 查询当前硬件支持的调度策略列表
//   2. mm(): 执行量化矩阵乘法 C = A @ B_dequant
//   3. prepack_B(): 预处理权重矩阵 B 为 Machete 的内部格式
//
// 与 Marlin 的区别:
//   - Machete 使用 CUTLASS 库的模板化 GEMM 实现
//   - 支持更多的量化格式和缩放策略 (group/channel/token scales)
//   - 通过 schedule 参数选择不同的 tile/mma 配置
//   - 需要预先 pack 权重矩阵
//
// 支持的缩放策略:
//   - group_scales: per-group 缩放因子 (grouped quantization)
//   - group_zeros: per-group 零点 (非对称量化)
//   - channel_scales: per-channel 缩放因子 (用于权重的每列缩放)
//   - token_scales: per-token 缩放因子 (用于激活的每行缩放)
// =============================================================================

namespace machete {

using namespace vllm;

std::vector<std::string> supported_schedules(
    at::ScalarType a_type, int64_t b_type_id,
    std::optional<at::ScalarType> maybe_group_scales_type,
    std::optional<at::ScalarType> maybe_group_zeros_type,
    std::optional<at::ScalarType> maybe_channel_scales_type,
    std::optional<at::ScalarType> maybe_token_scales_type,
    std::optional<at::ScalarType> maybe_out_type) {
  ScalarType const b_type = ScalarType::from_id(b_type_id);
  return supported_schedules_dispatch({
      .a_type = a_type,
      .b_type = b_type,
      .maybe_group_scales_type = maybe_group_scales_type,
      .maybe_group_zeros_type = maybe_group_zeros_type,
      .maybe_channel_scales_type = maybe_channel_scales_type,
      .maybe_token_scales_type = maybe_token_scales_type,
      .maybe_out_type = maybe_out_type,
  });
}

torch::Tensor mm(torch::Tensor const& A, torch::Tensor const& B,
                 int64_t b_type_id,
                 std::optional<at::ScalarType> const& maybe_out_type,
                 std::optional<torch::Tensor> const& maybe_group_scales,
                 std::optional<torch::Tensor> const& maybe_group_zeros,
                 std::optional<int64_t> maybe_group_size,
                 std::optional<torch::Tensor> const& maybe_channel_scales,
                 std::optional<torch::Tensor> const& maybe_token_scales,
                 std::optional<std::string> maybe_schedule) {
  ScalarType const b_type = ScalarType::from_id(b_type_id);
  return mm_dispatch({.A = A,
                      .B = B,
                      .b_type = b_type,
                      .maybe_out_type = maybe_out_type,
                      .maybe_group_scales = maybe_group_scales,
                      .maybe_group_zeros = maybe_group_zeros,
                      .maybe_group_size = maybe_group_size,
                      .maybe_channel_scales = maybe_channel_scales,
                      .maybe_token_scales = maybe_token_scales,
                      .maybe_schedule = maybe_schedule});
}

torch::Tensor prepack_B(
    torch::Tensor const& B, at::ScalarType const& a_type, int64_t b_type_id,
    std::optional<at::ScalarType> const& maybe_group_scales_type) {
  ScalarType const b_type = ScalarType::from_id(b_type_id);
  return prepack_B_dispatch(
      {.B = B,
       .a_type = a_type,
       .b_type = b_type,
       .maybe_group_scales_type = maybe_group_scales_type});
}

TORCH_LIBRARY_IMPL_EXPAND(TORCH_EXTENSION_NAME, CUDA, m) {
  m.impl("machete_prepack_B", &prepack_B);
  m.impl("machete_mm", &mm);
}

// use CatchAll since supported_schedules has no tensor arguments
TORCH_LIBRARY_IMPL(TORCH_EXTENSION_NAME, CatchAll, m) {
  m.impl("machete_supported_schedules", &supported_schedules);
}

};  // namespace machete
