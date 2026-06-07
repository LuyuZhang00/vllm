#pragma once

// =============================================================================
// 中文注释: Machete 权重预打包启动器
// =============================================================================
// 本文件实现了 Machete 的权重预打包 (prepack) 功能。
//
// 为什么需要预打包:
//   - Machete 使用 CUTLASS 的 GEMM 框架，要求权重按特定的内存布局存储
//   - 预打包将原始量化权重转换为 CUTLASS 所需的内部布局
//   - 预打包在模型加载时一次性完成，运行时直接使用打包后的权重
//
// 预打包流程:
//   1. 从 PyTorch 张量提取权重数据指针
//   2. 转置权重矩阵 (从 (packed_K, N) 到 (N, packed_K))
//   3. 调用预打包 kernel 将权重按 PPBlockShape_NK 的 tile 大小重新排列
//   4. 返回打包后的权重张量
//
// PrepackBArgs 结构体:
//   B: 待打包的量化权重张量
//   a_type: 激活数据类型 (用于确定打包策略)
//   b_type: 权重量化类型
//   maybe_group_scales_type: 可选的 group 缩放因子类型
// =============================================================================

#include "machete_prepack_kernel.cuh"
#include "cutlass_extensions/torch_utils.hpp"
#include "core/scalar_type.hpp"

namespace machete {

// 中文注释: 预打包参数结构体
struct PrepackBArgs {
  torch::Tensor const& B;
  at::ScalarType a_type;
  vllm::ScalarType b_type;
  std::optional<at::ScalarType> maybe_group_scales_type;
};

template <typename PrepackedLayoutB>
torch::Tensor prepack_impl(torch::Tensor const B) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(B));
  using ElementB = typename PrepackedLayoutB::ElementB;
  using PPBlockShape_NK = typename PrepackedLayoutB::PPBlockShape_NK;

  auto device = B.device();
  auto stream = at::cuda::getCurrentCUDAStream(device.index());
  auto B_ptr = static_cast<ElementB const*>(B.const_data_ptr());
  // elements per storage item for B
  auto eles_per_storage =
      (B.dtype().itemsize() * 8) / cute::sizeof_bits_v<ElementB>;

  // torch B passed in is/should be (packed_K,N), the kernel expects (N,K,L) (to
  // match cutlass using (N,K,L) for B), so we transpose B to (N,packed_K,L)
  auto Bt_packed = B.t();

  TORCH_CHECK(
      (B.size(0) * eles_per_storage) % size<1>(PPBlockShape_NK{}) == 0,
      "B.shape[0] (in terms of unpacked elements) must be a multiple of ",
      size<1>(PPBlockShape_NK{}));
  TORCH_CHECK(B.size(1) % size<0>(PPBlockShape_NK{}) == 0,
              "B.shape[1] must be a multiple of ", size<0>(PPBlockShape_NK{}));

  using StrideB = cutlass::detail::TagToStrideB_t<cutlass::layout::ColumnMajor>;
  auto const l_Bt_packed = make_cute_layout<StrideB>(Bt_packed, "B");

  // convert (N,packed_K,L) layout to (N,K,L) layout
  //  in effect we want to do: blocked_product(layout_Bt_packed,
  //      make_ordered_layout(make_shape(_1{}, eles_per_storage, _1{}),
  //                          Step<_1, _0, _2>{}));
  // but blocked_product does not support dynamic strides so we implement the
  // equivalent manually,
  //   new_shape = (N, packed_K, L) * (1, eles_per_storage, 1) -> (N, K, L)
  //   new_stride = (s0, s1, s2) * (eles_per_storage, 1, eles_per_storage)
  //                 when s1 == 1
  TORCH_CHECK(stride<1>(l_Bt_packed) == 1);
  // clang-format off
  auto const layout_Bt = make_layout(
      transform_with_idx(l_Bt_packed.shape(), [&](auto ele, auto idx) {
        return idx == 1 ? ele * eles_per_storage : ele;
      }), 
      transform_with_idx(l_Bt_packed.stride(), [&](auto ele, auto idx) {
        return idx != 1 ? ele * eles_per_storage : ele;
      }));
  // clang-format on

  // Allocate output
  torch::Tensor D = torch::empty_like(B, {}, at::MemoryFormat::Contiguous);

  prepack_B_template<PrepackedLayoutB>(
      stream, B_ptr, layout_Bt, static_cast<ElementB*>(D.mutable_data_ptr()));

  return D;
};

torch::Tensor prepack_B_dispatch(PrepackBArgs args);

};  // namespace machete