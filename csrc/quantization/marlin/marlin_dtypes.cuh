
// =============================================================================
// 中文注释: Marlin 数据类型定义
// =============================================================================
// 本文件为每种标量类型定义了 Marlin kernel 所需的所有类型别名和辅助函数。
//
// MarlinScalarType 模板类为每种数据类型提供:
//   - scalar_t/scalar_t2/scalar_t4: 标量和向量化类型
//   - FragA/FragB/FragC: Tensor Core MMA 指令所需的 fragment 类型
//     - FragA: A 矩阵 fragment (16x16 子块)
//     - FragB: B 矩阵 fragment (8x16 子块)
//     - FragC: C 矩阵累加器 fragment (16x8 子块)
//   - FragS: 缩放因子 fragment
//   - FragZP: 零点 fragment
//   - num2float/float2num: 类型转换辅助函数
//
// Fragment 布局遵循 NVIDIA PTX 文档:
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#matrix-fragments-for-mma-m16n8k16-with-floating-point-type
//
// 支持的数据类型:
//   - FP16 (half): 基础类型，所有 SM 支持
//   - BF16 (nv_bfloat16): SM80+ 支持
//   - FP8 (Float8_e4m3fn): SM89+ 支持 (W4A8-FP8)
//   - INT8 (int8_t): SM75+ 支持
// =============================================================================

#ifndef _data_types_cuh
#define _data_types_cuh
#include "marlin.cuh"
#include "core/scalar_type.hpp"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#ifndef MARLIN_NAMESPACE_NAME
  #define MARLIN_NAMESPACE_NAME marlin
#endif

namespace MARLIN_NAMESPACE_NAME {

// 中文注释: MarlinScalarType 主模板 (未特化版本，编译时会报错)
template <long scalar_type_id>
class MarlinScalarType {};

// 中文注释: FP16 类型特化
// half2 是 CUDA 的 2 元素 half 向量类型，用于向量化运算
// FragA = Vec<half2, 4> = 4 个 half2 = 8 个 half = 128 bit
// FragB = Vec<half2, 2> = 2 个 half2 = 4 个 half = 64 bit
// FragC = Vec<float, 4> = 4 个 float = 128 bit (FP32 累加器)
template <>
class MarlinScalarType<vllm::kFloat16.id()> {
 public:
  using scalar_t = half;
  using scalar_t2 = half2;
  using scalar_t4 = half2;
  using scalar_32bit_t = half2;

  // Matrix fragments for tensor core instructions; their precise layout is
  // documented here:
  // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#matrix-fragments-for-mma-m16n8k16-with-floating-point-type
  using FragA = Vec<half2, 4>;
  using FragB = Vec<half2, 2>;
  using FragC = Vec<float, 4>;
  using FragS = Vec<half2, 1>;
  using FragS0 = Vec<__nv_fp8x2_e4m3, 1>;
  using FragZP = Vec<half2, 4>;

  static __device__ float inline num2float(const half x) {
    return __half2float(x);
  }

  static __device__ half2 inline num2num2(const half x) {
    return __half2half2(x);
  }

  static __device__ half2 inline nums2num2(const half x1, const half x2) {
    return __halves2half2(x1, x2);
  }

  static __host__ __device__ half inline float2num(const float x) {
    return __float2half(x);
  }

  static __host__ __device__ float2 inline num22float2(const half2 x) {
    return __half22float2(x);
  }
};

template <>
class MarlinScalarType<vllm::kBFloat16.id()> {
 public:
  using scalar_t = nv_bfloat16;
  using scalar_t2 = nv_bfloat162;
  using scalar_t4 = nv_bfloat162;
  using scalar_32bit_t = nv_bfloat162;

  using FragA = Vec<nv_bfloat162, 4>;
  using FragB = Vec<nv_bfloat162, 2>;
  using FragC = Vec<float, 4>;
  using FragS = Vec<nv_bfloat162, 1>;
  using FragS0 = Vec<__nv_fp8x2_e4m3, 1>;
  using FragZP = Vec<nv_bfloat162, 4>;

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
  static __device__ float inline num2float(const nv_bfloat16 x) {
    return __bfloat162float(x);
  }

  static __device__ nv_bfloat162 inline num2num2(const nv_bfloat16 x) {
    return __bfloat162bfloat162(x);
  }

  static __device__ nv_bfloat162 inline nums2num2(const nv_bfloat16 x1,
                                                  const nv_bfloat16 x2) {
    return __halves2bfloat162(x1, x2);
  }

  static __host__ __device__ nv_bfloat16 inline float2num(const float x) {
    return __float2bfloat16(x);
  }

  static __host__ __device__ float2 inline num22float2(const nv_bfloat162 x) {
    return __bfloat1622float2(x);
  }
#endif
};

template <>
class MarlinScalarType<vllm::kFE4M3fn.id()> {
 public:
  using scalar_t = __nv_fp8_e4m3;
  using scalar_t2 = __nv_fp8x2_e4m3;
  using scalar_t4 = __nv_fp8x4_e4m3;
  using scalar_32bit_t = __nv_fp8x4_e4m3;

  using FragA = Vec<__nv_fp8x4_e4m3, 4>;
  using FragB = Vec<__nv_fp8x4_e4m3, 2>;
  using FragC = Vec<float, 4>;
  using FragZP = Vec<__nv_fp8x2_e4m3, 4>;

  static __host__ __device__
      float2 inline num22float2(const __nv_fp8x2_e4m3 x) {
    return (float2)x;
  }
};

template <>
class MarlinScalarType<vllm::kS8.id()> {
 public:
  using scalar_t = int8_t;
  using scalar_t2 = int16_t;
  using scalar_t4 = int32_t;
  using scalar_32bit_t = int32_t;

  using FragA = Vec<int32_t, 4>;
  using FragB = Vec<int32_t, 2>;
  using FragC = Vec<float, 4>;
  using FragZP = Vec<int16_t, 4>;
};

template <typename scalar_t>
class MarlinScalarType2 {};

template <>
class MarlinScalarType2<half> : public MarlinScalarType<vllm::kFloat16.id()> {};

template <>
class MarlinScalarType2<nv_bfloat16>
    : public MarlinScalarType<vllm::kBFloat16.id()> {};

template <>
class MarlinScalarType2<__nv_fp8_e4m3>
    : public MarlinScalarType<vllm::kFE4M3fn.id()> {};

template <>
class MarlinScalarType2<int8_t> : public MarlinScalarType<vllm::kS8.id()> {};

}  // namespace MARLIN_NAMESPACE_NAME

#endif
