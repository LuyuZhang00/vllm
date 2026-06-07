#pragma once

// 中文注释：本文件是 PyTorch（libtorch）与 CUTLASS/CuTe 之间的桥接工具。
// 主要功能：
//   1. 处理 stable ABI / unstable ABI 的兼容性（通过 TORCH_TARGET_VERSION 宏切换）
//   2. 提供 make_cute_layout 函数，将 PyTorch Tensor 的 shape/stride 转换为 CuTe Layout
//   3. 提供 Torch 类型与 CUTLASS 类型之间的映射（equivalent_cutlass_type / equivalent_scalar_type）
//   4. 提供 CuTe 辅助工具（transform_with_idx, make_shape_from_idx）
//
// 这些工具使得 CUTLASS kernel 可以直接从 PyTorch Tensor 中获取正确的布局信息，
// 无需手动转换。

#include "torch_utils.h"

// This header is shared between _C (unstable ABI, used by machete) and
// _C_stable_libtorch (stable ABI, used by W4A8/sparse). TORCH_TARGET_VERSION
// is defined only for the stable target, so we switch includes and types
// accordingly. TorchTensor (not Tensor) avoids ambiguity with cute::Tensor.
#ifdef TORCH_TARGET_VERSION
  #include <torch/csrc/stable/tensor.h>
  #include <torch/headeronly/util/BFloat16.h>
  #include <torch/headeronly/util/Half.h>
using TorchTensor = torch::stable::Tensor;
#else
using TorchTensor = torch::Tensor;
#endif

#include "cute/layout.hpp"
#include "cutlass/layout/matrix.h"
#include "cutlass/bfloat16.h"
#include "cutlass/half.h"

using ColumnMajor = typename cutlass::layout::ColumnMajor;
using RowMajor = typename cutlass::layout::RowMajor;

namespace cute {

namespace detail {

template <class T, class F, class G, int... I>
CUTE_HOST_DEVICE constexpr auto tapply_with_idx(T&& t, F&& f, G&& g,
                                                seq<I...>) {
  return g(f(cute::get<I>(static_cast<T&&>(t)), I)...);
}

template <class F, int... I>
CUTE_HOST_DEVICE constexpr auto make_shape_from_idx(F&& f, seq<I...>) {
  return make_shape(f(I)...);
}

};  // namespace detail

// 中文注释：对 tuple 中的每个元素应用变换函数 f(element, index)，
// 返回变换后的 tuple。支持嵌套 tuple 的递归处理。
// 用于在 make_cute_layout 中同时处理 shape 和 stride 的每个维度。
template <class T, class F>
CUTE_HOST_DEVICE constexpr auto transform_with_idx(T const& t, F&& f) {
  if constexpr (cute::is_tuple<T>::value) {
    return detail::tapply_with_idx(
        t, f, [](auto const&... a) { return cute::make_tuple(a...); },
        tuple_seq<T>{});
  } else {
    return f(t);
  }

  CUTE_GCC_UNREACHABLE;
}

// calls: make_shape(f(0), f(1), ..., f(N-1))
// 中文注释：通过索引函数 f 构造 CuTe Shape。make_shape(f(0), f(1), ..., f(N-1))。
template <int N, class F>
CUTE_HOST_DEVICE constexpr auto make_shape_from_idx(F&& f) {
  return detail::make_shape_from_idx(f, make_seq<N>{});
}

};  // namespace cute

// Make a layout from a tensor with `rank(Stride{})`, where the shape is the
// shape of the passed in tensor and the strides are of type `Stride` and
// contain the strides of the passed in tensor, checking that any static strides
// in `Stride{}` match the strides of the passed in tensor.
// If `tensor.dim() < rank(Stride{})`, the shape is padded with 1s and the extra
// strides are set to be 0 or 1.
// 中文注释：将 PyTorch Tensor 转换为 CuTe Layout。
// 这是连接 PyTorch 张量系统和 CUTLASS 计算核心的关键函数。
//
// 工作流程：
//   1. 从 Stride 模板参数确定目标 layout 的维度数
//   2. 遍历每个维度，提取 tensor 的 stride，并与 Stride 中的静态值进行校验
//   3. 对于 size 为 1 的维度，将 stride 设为 0（便于 CuTe/TMA 优化维度折叠）
//   4. 如果 tensor 维度少于 Stride 维度，用 1 填充 shape，0/1 填充 stride
//   5. 最终构造 CuTe Layout(shape, stride) 返回
//
// 使用示例：
//   make_cute_layout<Stride<int, int, int>>(tensor) -- 动态 stride
//   make_cute_layout<Stride<_1, int, int>>(tensor)  -- 第一维 stride 必须为 1（列主序）
template <typename Stride>
static inline auto make_cute_layout(TorchTensor const& tensor,
                                    std::string_view name = "tensor") {
  TORCH_UTILS_CHECK(tensor.dim() <= rank(Stride{}));
  auto stride = cute::transform_with_idx(Stride{}, [&](auto const& stride_ele,
                                                       auto const& idx) {
    using StrideEle = std::decay_t<decltype(stride_ele)>;

    if (idx < tensor.dim()) {
      if constexpr (cute::is_static_v<StrideEle>) {
        TORCH_UTILS_CHECK(StrideEle::value == tensor.stride(idx), "Expected ",
                          name, ".stride(", idx, ") to be ", StrideEle::value);
        return StrideEle{};
      } else {
        if (tensor.size(idx) == 1) {
          // use 0 stride for dim with size 1, this is easier for
          // cute/cutlass to optimize (helps the TMA code flatten dims)
          return StrideEle{0};
        } else {
          return tensor.stride(idx);
        }
      }
    } else {
      // Extra strides are assumed to be 0 or 1
      if constexpr (cute::is_static_v<StrideEle>) {
        static_assert(StrideEle::value == 0 || StrideEle::value == 1);
      }
      return StrideEle{};
    }
  });

  auto shape = cute::make_shape_from_idx<rank(Stride{})>([&](auto const& idx) {
    if (idx < tensor.dim())
      return tensor.size(idx);
    else
      return int64_t(1);
  });

  return make_layout(shape, stride);
}

// 中文注释：可选版本的 make_cute_layout。当 tensor 存在时转换为 CuTe Layout，
// 当 tensor 为 std::nullopt 时返回空的 optional。用于处理 bias 等可选参数。
template <typename Stride>
static inline auto maybe_make_cute_layout(
    std::optional<TorchTensor> const& tensor,
    std::string_view name = "tensor") {
  using Layout = decltype(make_cute_layout<Stride>(*tensor));

  if (tensor) {
    return std::optional<Layout>{make_cute_layout<Stride>(*tensor, name)};
  } else {
    return std::optional<Layout>{};
  }
}

//
//  Torch Type to Cutlass Type (equivalent_cutlass_type)
//

// 中文注释：Torch 类型到 CUTLASS 类型的映射。
// PyTorch 和 CUTLASS 各有自己的类型系统（如 torch::Half vs cutlass::half_t），
// 此模板在编译期建立两者之间的对应关系，使得 kernel 可以统一使用 CUTLASS 类型。
//   - torch::headeronly::Half     -> cutlass::half_t
//   - torch::headeronly::BFloat16 -> cutlass::bfloat16_t
//   - 其他类型默认保持不变
template <typename T>
struct equivalent_cutlass_type {
  using type = T;
};

template <typename T>
using equivalent_cutlass_type_t = typename equivalent_cutlass_type<T>::type;

template <>
struct equivalent_cutlass_type<torch::headeronly::Half> {
  using type = cutlass::half_t;
};

template <>
struct equivalent_cutlass_type<torch::headeronly::BFloat16> {
  using type = cutlass::bfloat16_t;
};

//
// equivalent_scalar_t (basically inverse of equivalent_cutlass_type)
//

// Return a `torch::headeronly::CppTypeToScalarType<T>` compatible type, i.e.
// get the C++ type equivalent to T, e.g.: `cutlass::half_t -> Half`
// 中文注释：CUTLASS 类型到 Torch 类型的反向映射（equivalent_cutlass_type 的逆映射）。
//   - cutlass::half_t     -> torch::headeronly::Half
//   - cutlass::bfloat16_t -> torch::headeronly::BFloat16
// 用于在 kernel 结束后将结果类型映射回 PyTorch 可识别的类型。
template <typename T>
struct equivalent_scalar_type {
  using type = T;
};

template <typename T>
using equivalent_scalar_type_t = typename equivalent_scalar_type<T>::type;

template <>
struct equivalent_scalar_type<cutlass::half_t> {
  using type = torch::headeronly::Half;
};

template <>
struct equivalent_scalar_type<cutlass::bfloat16_t> {
  using type = torch::headeronly::BFloat16;
};

// get equivalent torch::headeronly::ScalarType tag from compile time type
// 中文注释：编译期获取 CUTLASS 类型对应的 PyTorch ScalarType 枚举值。
// 例如 equivalent_scalar_type_v<cutlass::half_t> 等价于 torch::kHalf。
// 用于在 C++ 层面设置 PyTorch Tensor 的 dtype。
template <typename T>
static inline constexpr torch::headeronly::ScalarType equivalent_scalar_type_v =
    torch::headeronly::CppTypeToScalarType<equivalent_scalar_type_t<T>>::value;
