// =============================================================================
// 中文注释：MoE 数据类型分发宏
//
// 本文件定义了 MoE kernel 中用于数据类型分发的宏和模板：
//
// 1. MOE_DISPATCH(TYPE, ...) —— 根据 PyTorch 的 ScalarType 分发到对应的
//    CUDA 数据类型（float, half, bfloat16, fp8_e5m2, fp8_e4m3, uint8）。
//    用于 moe_permute/unpermute kernel 中支持多种输入数据类型。
//
// 2. ScalarType2CudaType<> —— 模板特化，将 PyTorch ScalarType 映射为 CUDA 类型。
// =============================================================================

#pragma once
#include <cuda_fp8.h>
#define MOE_SWITCH(TYPE, ...)                                     \
  at::ScalarType _st = ::detail::scalar_type(TYPE);               \
  switch (_st) {                                                  \
    __VA_ARGS__                                                   \
    default:                                                      \
      TORCH_CHECK(false, "[moe permute]data type dispatch fail!") \
  }

#define MOE_DISPATCH_CASE(enum_type, ...)                  \
  case enum_type: {                                        \
    using scalar_t = ScalarType2CudaType<enum_type>::type; \
    __VA_ARGS__();                                         \
    break;                                                 \
  }
#define MOE_DISPATCH_FLOAT_CASE(...)                            \
  MOE_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__)         \
  MOE_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)          \
  MOE_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__)      \
  MOE_DISPATCH_CASE(at::ScalarType::Float8_e5m2, __VA_ARGS__)   \
  MOE_DISPATCH_CASE(at::ScalarType::Float8_e4m3fn, __VA_ARGS__) \
  MOE_DISPATCH_CASE(at::ScalarType::Byte, __VA_ARGS__)

#define MOE_DISPATCH(TYPE, ...) \
  MOE_SWITCH(TYPE, MOE_DISPATCH_FLOAT_CASE(__VA_ARGS__))

template <at::ScalarType type>
struct ScalarType2CudaType;

template <>
struct ScalarType2CudaType<at::ScalarType::Float> {
  using type = float;
};
template <>
struct ScalarType2CudaType<at::ScalarType::Half> {
  using type = half;
};
template <>
struct ScalarType2CudaType<at::ScalarType::BFloat16> {
  using type = __nv_bfloat16;
};
// uint8 for packed fp4
template <>
struct ScalarType2CudaType<at::ScalarType::Byte> {
  using type = uint8_t;
};

// #if __CUDA_ARCH__ >= 890
// fp8
template <>
struct ScalarType2CudaType<at::ScalarType::Float8_e5m2> {
  using type = __nv_fp8_e5m2;
};
template <>
struct ScalarType2CudaType<at::ScalarType::Float8_e4m3fn> {
  using type = __nv_fp8_e4m3;
};
// #endif