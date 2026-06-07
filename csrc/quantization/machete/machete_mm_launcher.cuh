#pragma once

// =============================================================================
// 中文注释: Machete 量化 GEMM 启动器
// =============================================================================
// 本文件是 Machete 量化矩阵乘法的调度和启动层。
//
// 核心功能:
//   1. MMArgs: 矩阵乘法参数结构体，封装所有输入参数
//   2. SupportedSchedulesArgs: 查询支持的调度策略的参数结构体
//   3. run_impl<MacheteKernel>: 模板化的 kernel 执行函数
//      - 从 PyTorch 张量提取数据指针
//      - 设置 CUDA stream
//      - 调用 MacheteKernel::run 执行实际的矩阵乘法
//   4. mm_dispatch: 根据数据类型分发到具体的 kernel 实例
//
// Machete 使用 CUTLASS 库的模板化 GEMM 框架，通过不同的模板参数组合
// 实现对不同量化格式、tile 大小、MMA 指令的支持。
// =============================================================================

#include <torch/all.h>
#include <Python.h>

#include "machete_mm_kernel.cuh"
#include "cutlass_extensions/torch_utils.hpp"
#include "core/scalar_type.hpp"

namespace machete {

// 中文注释: 矩阵乘法参数结构体
// A: 输入激活矩阵 [M, K]
// B: 预 pack 过的量化权重矩阵
// b_type: 权重量化类型
// maybe_out_type: 可选的输出类型
// maybe_group_scales/zeros: 可选的 per-group 缩放因子和零点
// maybe_channel_scales: 可选的 per-channel 缩放因子
// maybe_token_scales: 可选的 per-token 缩放因子
// maybe_schedule: 可选的调度策略名称
struct MMArgs {
  torch::Tensor const& A;
  torch::Tensor const& B;
  vllm::ScalarType const& b_type;
  std::optional<at::ScalarType> const& maybe_out_type;
  std::optional<torch::Tensor> const& maybe_group_scales;
  std::optional<torch::Tensor> const& maybe_group_zeros;
  std::optional<int64_t> maybe_group_size;
  std::optional<torch::Tensor> const& maybe_channel_scales;
  std::optional<torch::Tensor> const& maybe_token_scales;
  std::optional<std::string> maybe_schedule;
};

struct SupportedSchedulesArgs {
  at::ScalarType a_type;
  vllm::ScalarType b_type;
  std::optional<at::ScalarType> maybe_group_scales_type;
  std::optional<at::ScalarType> maybe_group_zeros_type;
  std::optional<at::ScalarType> maybe_channel_scales_type;
  std::optional<at::ScalarType> maybe_token_scales_type;
  std::optional<at::ScalarType> maybe_out_type;
};

torch::Tensor mm_dispatch(MMArgs args);

std::vector<std::string> supported_schedules_dispatch(
    SupportedSchedulesArgs args);

template <typename MacheteKernel>
torch::Tensor run_impl(MMArgs args) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(args.A));

  auto device = args.A.device();
  auto stream = at::cuda::getCurrentCUDAStream(device.index());

  int M = args.A.size(0);
  int N = args.B.size(1);
  int K = args.A.size(1);

  // Allocate output
  torch::Tensor D = torch::empty(
      {M, N},
      torch::TensorOptions()
          .dtype(equivalent_scalar_type_v<typename MacheteKernel::ElementD>)
          .device(device));

  auto arguments = MacheteKernel::create_arguments(
      stream,  //
      args.A, args.B, D, args.maybe_group_scales, args.maybe_group_zeros,
      args.maybe_group_size, args.maybe_channel_scales,
      args.maybe_token_scales);
  TORCH_CHECK(MacheteKernel::can_implement(arguments),
              "Machete kernel cannot be run with these arguments");

  size_t workspace_size = MacheteKernel::get_workspace_size(arguments);
  torch::Tensor workspace = torch::empty(
      workspace_size, torch::TensorOptions().dtype(torch::kU8).device(device));

  MacheteKernel::run(arguments, workspace.mutable_data_ptr(), stream);

  return D;
};

};  // namespace machete