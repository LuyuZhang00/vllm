#pragma once

// =============================================================================
// 中文注释: NVIDIA GPU FP8 量化/反量化工具
// =============================================================================
// 本文件实现了 NVIDIA GPU 上 FP8 数据类型的类型转换工具函数。
//
// 核心功能:
//   1. vec_conversion<Tout, Tin>(): 无缩放因子的向量化类型转换
//      - 支持 FP8 <-> half/bf16/float 之间的相互转换
//      - 支持打包格式: fp8x2 (uint16), fp8x4 (uint32), fp8x8 (uint2)
//      - 利用 NVIDIA 硬件指令 __nv_cvt_* 实现高效转换
//
//   2. scaled_vec_conversion<Tout, Tin>(): 带缩放因子的向量化类型转换
//      - 量化: HP / scale -> FP8
//      - 反量化: FP8 * scale -> HP
//      - 用于动态量化场景，每个 group/channel 有独立的缩放因子
//
//   3. convert/scaled_convert: 模板分发接口
//      - 根据 Fp8KVCacheDataType 枚举选择 E4M3 或 E5M2 格式
//      - 用于 KV cache 的 FP8 存储
//
//   4. DISPATCH_BY_KV_CACHE_DTYPE: 宏分发器
//      - 根据源数据类型和 KV cache 数据类型自动选择正确的模板实例化
//
// 支持的数据类型组合:
//   输入: float, half (uint16), bfloat16, float2, Float4_, Float8_
//   输出: 同上 + uint8_t (FP8 标量), uint16_t (FP8x2), uint32_t (FP8x4)
//
// 硬件要求:
//   - FP8 转换需要 SM >= 80 (Ampere/SM80+ 的 __nv_fp8 类型)
//   - 部分模板特化在 SM < 80 时会被禁用 (通过 __CUDA_ARCH__ 检查)
// =============================================================================

#include "../../../../attention/attention_dtypes.h"
#include <torch/headeronly/core/ScalarType.h>
#include <assert.h>
#include <float.h>
#include <stdint.h>
#include <type_traits>

namespace vllm {
#ifndef USE_ROCM

namespace fp8 {
  #ifdef ENABLE_FP8

// 中文注释: vec_conversion 默认模板 -- 同类型转换直接返回 (零开销)
template <typename Tout, typename Tin>
__inline__ __device__ Tout vec_conversion(
    const Tin& x, const __nv_fp8_interpretation_t fp8_type = __NV_E4M3) {
  return x;
}

// float -> c10::Float8_e4m3fn
// 中文注释: float 到 FP8 的标量转换
// - SM >= 80: 使用硬件指令 __nv_cvt_float_to_fp8，带饱和截断 (__NV_SATFINITE)
// - SM < 80: 退化为 PyTorch 的软件实现
// __NV_SATFINITE 表示超出范围的值被截断到 FP8 的最大/最小可表示值
template <>
__inline__ __device__ c10::Float8_e4m3fn
vec_conversion<c10::Float8_e4m3fn, float>(
    const float& a, const __nv_fp8_interpretation_t fp8_type) {
    #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  return static_cast<c10::Float8_e4m3fn>(a);
    #else
  return c10::Float8_e4m3fn(__nv_cvt_float_to_fp8(a, __NV_SATFINITE, fp8_type),
                            c10::Float8_e4m3fn::from_bits());
    #endif
}

    #if 0  // Disable the following code to reduce the binary size.
// fp8 -> half
template <>
__inline__ __device__ uint16_t vec_conversion<uint16_t, uint8_t>(
    const uint8_t &a, const __nv_fp8_interpretation_t fp8_type) {
  __half_raw res = __nv_cvt_fp8_to_halfraw(a, fp8_type);
  return res.x;
}

// fp8x2 -> half2
template <>
__inline__ __device__ uint32_t vec_conversion<uint32_t, uint16_t>(
    const uint16_t &a, const __nv_fp8_interpretation_t fp8_type) {
  union {
    uint16_t u16[2];
    uint32_t u32;
  } tmp;
  __half2_raw res = __nv_cvt_fp8x2_to_halfraw2(a, fp8_type);
  tmp.u16[0] = res.x;
  tmp.u16[1] = res.y;
  return tmp.u32;
}

// fp8x4 -> half2x2
template <>
__inline__ __device__ uint2 vec_conversion<uint2, uint32_t>(
    const uint32_t &a, const __nv_fp8_interpretation_t fp8_type) {
  union {
    uint2 u32x2;
    uint32_t u32[2];
  } tmp;
  tmp.u32[0] = vec_conversion<uint32_t, uint16_t>((uint16_t)a, fp8_type);
  tmp.u32[1] =
      vec_conversion<uint32_t, uint16_t>((uint16_t)(a >> 16U), fp8_type);
  return tmp.u32x2;
}

// fp8x8 -> half2x4
template <>
__inline__ __device__ uint4 vec_conversion<uint4, uint2>(
    const uint2 &a, const __nv_fp8_interpretation_t fp8_type) {
  union {
    uint4 u64x2;
    uint2 u64[2];
  } tmp;
  tmp.u64[0] = vec_conversion<uint2, uint32_t>(a.x, fp8_type);
  tmp.u64[1] = vec_conversion<uint2, uint32_t>(a.y, fp8_type);
  return tmp.u64x2;
}

// fp8 -> __nv_bfloat16
template <>
__inline__ __device__ __nv_bfloat16 vec_conversion<__nv_bfloat16, uint8_t>(
    const uint8_t &a, const __nv_fp8_interpretation_t fp8_type) {
  // Note there is no direct convert function from fp8 to bf16.
  // fp8 -> half
  __half_raw res = __nv_cvt_fp8_to_halfraw(a, fp8_type);
  // half -> float -> bf16
  float tmp = half_to_float(res.x);
  return __float2bfloat16(tmp);
}

// fp8x2 -> __nv_bfloat162
template <>
__inline__ __device__ __nv_bfloat162 vec_conversion<__nv_bfloat162, uint16_t>(
    const uint16_t &a, const __nv_fp8_interpretation_t fp8_type) {
  __nv_bfloat162 res;
  res.x = vec_conversion<__nv_bfloat16, uint8_t>((uint8_t)a, fp8_type);
  res.y = vec_conversion<__nv_bfloat16, uint8_t>((uint8_t)(a >> 8U), fp8_type);
  return res;
}

// fp8x4 -> bf16_4_t
template <>
__inline__ __device__ bf16_4_t vec_conversion<bf16_4_t, uint32_t>(
    const uint32_t &a, const __nv_fp8_interpretation_t fp8_type) {
  bf16_4_t res;
  res.x = vec_conversion<__nv_bfloat162, uint16_t>((uint16_t)a, fp8_type);
  res.y =
      vec_conversion<__nv_bfloat162, uint16_t>((uint16_t)(a >> 16U), fp8_type);
  return res;
}

// fp8x8 -> bf16_8_t
template <>
__inline__ __device__ bf16_8_t vec_conversion<bf16_8_t, uint2>(
    const uint2 &a, const __nv_fp8_interpretation_t fp8_type) {
  bf16_4_t tmp1, tmp2;
  tmp1 = vec_conversion<bf16_4_t, uint32_t>(a.x, fp8_type);
  tmp2 = vec_conversion<bf16_4_t, uint32_t>(a.y, fp8_type);
  bf16_8_t res;
  res.x = tmp1.x;
  res.y = tmp1.y;
  res.z = tmp2.x;
  res.w = tmp2.y;
  return res;
}

// fp8 -> float
template <>
__inline__ __device__ float
vec_conversion<float, uint8_t>(const uint8_t &a,
                               const __nv_fp8_interpretation_t fp8_type) {
  // fp8 -> half
  uint16_t tmp = vec_conversion<uint16_t, uint8_t>(a, fp8_type);
  // half -> float
  return half_to_float(tmp);
}

// fp8x2 -> float2
template <>
__inline__ __device__ float2 vec_conversion<float2, uint16_t>(
    const uint16_t &a, const __nv_fp8_interpretation_t fp8_type) {
  // fp8x2 -> half2
  uint32_t tmp = vec_conversion<uint32_t, uint16_t>(a, fp8_type);
  // half2 -> float2
  return half2_to_float2(tmp);
}

// fp8x4 -> float4
template <>
__inline__ __device__ Float4_ vec_conversion<Float4_, uint32_t>(
    const uint32_t &a, const __nv_fp8_interpretation_t fp8_type) {
  Float4_ res;
  res.x = vec_conversion<float2, uint16_t>((uint16_t)a, fp8_type);
  res.y = vec_conversion<float2, uint16_t>((uint16_t)(a >> 16U), fp8_type);
  return res;
}

// fp8x8 -> float8
template <>
__inline__ __device__ Float8_ vec_conversion<Float8_, uint2>(
    const uint2 &a, const __nv_fp8_interpretation_t fp8_type) {
  Float4_ tmp1, tmp2;
  tmp1 = vec_conversion<Float4_, uint32_t>(a.x, fp8_type);
  tmp2 = vec_conversion<Float4_, uint32_t>(a.y, fp8_type);
  Float8_ res;
  res.x = tmp1.x;
  res.y = tmp1.y;
  res.z = tmp2.x;
  res.w = tmp2.y;
  return res;
}

// half -> fp8
template <>
__inline__ __device__ uint8_t vec_conversion<uint8_t, uint16_t>(
    const uint16_t &a, const __nv_fp8_interpretation_t fp8_type) {
  __half_raw tmp;
  tmp.x = a;
  __nv_fp8_storage_t res =
      __nv_cvt_halfraw_to_fp8(tmp, __NV_SATFINITE, fp8_type);
  return (uint8_t)res;
}

// bf16 -> fp8
template <>
__inline__ __device__ uint8_t vec_conversion<uint8_t, __nv_bfloat16>(
    const __nv_bfloat16 &a, const __nv_fp8_interpretation_t fp8_type) {
      #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  assert(false);
      #else
  __nv_fp8_storage_t res = __nv_cvt_bfloat16raw_to_fp8(
      __nv_bfloat16_raw(a), __NV_SATFINITE, fp8_type);
  return (uint8_t)res;
      #endif
}

// float -> fp8
template <>
__inline__ __device__ uint8_t vec_conversion<uint8_t, float>(
    const float &a, const __nv_fp8_interpretation_t fp8_type) {
  __nv_fp8_storage_t res = __nv_cvt_float_to_fp8(a, __NV_SATFINITE, fp8_type);
  return (uint8_t)res;
}

// fp8x4 -> float4
template <>
__inline__ __device__ float4 vec_conversion<float4, uint32_t>(
    const uint32_t &a, const __nv_fp8_interpretation_t fp8_type) {
  Float4_ tmp = vec_conversion<Float4_, uint32_t>(a, fp8_type);
  float4 res = make_float4(tmp.x.x, tmp.x.y, tmp.y.x, tmp.y.y);
  return res;
}

template <>
__inline__ __device__ uint32_t vec_conversion<uint32_t, float2>(
    const float2 &a, const __nv_fp8_interpretation_t fp8_type) {
  union {
    half2 float16;
    uint32_t uint32;
  };

  float16 = __float22half2_rn(a);
  return uint32;
}

template <>
__inline__ __device__ uint2 vec_conversion<uint2, Float4_>(
    const Float4_ &a, const __nv_fp8_interpretation_t fp8_type) {
  uint2 b;
  float2 val;
  val.x = a.x.x;
  val.y = a.x.y;
  b.x = vec_conversion<uint32_t, float2>(val, fp8_type);

  val.x = a.y.x;
  val.y = a.y.y;
  b.y = vec_conversion<uint32_t, float2>(val, fp8_type);

  return b;
}

template <>
__inline__ __device__ float4 vec_conversion<float4, Float4_>(
    const Float4_ &a, const __nv_fp8_interpretation_t fp8_type) {
  float4 b;
  b.x = a.x.x;
  b.y = a.x.y;
  b.z = a.y.x;
  b.w = a.y.y;
  return b;
}

template <>
__inline__ __device__ uint4 vec_conversion<uint4, Float8_>(
    const Float8_ &a, const __nv_fp8_interpretation_t fp8_type) {
  uint4 b;
  b.x = vec_conversion<uint32_t, float2>(a.x, fp8_type);
  b.y = vec_conversion<uint32_t, float2>(a.y, fp8_type);
  b.z = vec_conversion<uint32_t, float2>(a.z, fp8_type);
  b.w = vec_conversion<uint32_t, float2>(a.w, fp8_type);
  return b;
}

template <>
__inline__ __device__ __nv_bfloat162 vec_conversion<__nv_bfloat162, float2>(
    const float2 &a, const __nv_fp8_interpretation_t fp8_type) {
  __nv_bfloat162 b;
  from_float(b, a);
  return b;
}

template <>
__inline__ __device__ bf16_4_t vec_conversion<bf16_4_t, Float4_>(
    const Float4_ &a, const __nv_fp8_interpretation_t fp8_type) {
  bf16_4_t b;
  from_float(b, a);
  return b;
}

template <>
__inline__ __device__ bf16_8_t vec_conversion<bf16_8_t, Float8_>(
    const Float8_ &a, const __nv_fp8_interpretation_t fp8_type) {
  bf16_8_t b;
  from_float(b, a);
  return b;
}
    #endif

/* Scaled and vectorized conversions, for data exchange between high and low
   precision domains Convention of the scale in API, e.g: FP8_data =
   Quantization( High_Precision_data / scale ) s.t. Quantize(HP / scale) => FP8
     Dequant(FP8) * scale =>  HP
 */

// =============================================================================
// 中文注释: 带缩放因子的向量化类型转换接口
// =============================================================================
// 约定:
//   量化: FP8 = Quantize(HP / scale) -- 高精度除以缩放因子后量化
//   反量化: HP = Dequant(FP8) * scale -- FP8 反量化后乘以缩放因子
//
// 默认模板: 同类型直接返回 (用于 kAuto 模式，即不做 FP8 转换)
// =============================================================================

template <typename Tout, typename Tin>
__inline__ __device__ Tout scaled_vec_conversion(
    const Tin& x, const float scale, const __nv_fp8_interpretation_t fp8_type) {
  return x;
}

// fp8 -> half
template <>
__inline__ __device__ uint16_t scaled_vec_conversion<uint16_t, uint8_t>(
    const uint8_t& a, const float scale,
    const __nv_fp8_interpretation_t fp8_type) {
  __half_raw tmp = __nv_cvt_fp8_to_halfraw(a, fp8_type);
  return float_to_half(half_to_float(tmp.x) * scale);
}

// fp8x2 -> half2
template <>
__inline__ __device__ uint32_t scaled_vec_conversion<uint32_t, uint16_t>(
    const uint16_t& a, const float scale,
    const __nv_fp8_interpretation_t fp8_type) {
  union {
    uint16_t u16[2];
    uint32_t u32;
  } tmp;
  __half2_raw res = __nv_cvt_fp8x2_to_halfraw2(a, fp8_type);
  tmp.u16[0] = float_to_half(half_to_float(res.x) * scale);
  tmp.u16[1] = float_to_half(half_to_float(res.y) * scale);
  return tmp.u32;
}

// fp8x4 -> half2x2
template <>
__inline__ __device__ uint2 scaled_vec_conversion<uint2, uint32_t>(
    const uint32_t& a, const float scale,
    const __nv_fp8_interpretation_t fp8_type) {
  union {
    uint2 u32x2;
    uint32_t u32[2];
  } tmp;
  tmp.u32[0] =
      scaled_vec_conversion<uint32_t, uint16_t>((uint16_t)a, scale, fp8_type);
  tmp.u32[1] = scaled_vec_conversion<uint32_t, uint16_t>((uint16_t)(a >> 16U),
                                                         scale, fp8_type);
  return tmp.u32x2;
}

// fp8x8 -> half2x4
template <>
__inline__ __device__ uint4
scaled_vec_conversion<uint4, uint2>(const uint2& a, const float scale,
                                    const __nv_fp8_interpretation_t fp8_type) {
  union {
    uint4 u64x2;
    uint2 u64[2];
  } tmp;
  tmp.u64[0] = scaled_vec_conversion<uint2, uint32_t>(a.x, scale, fp8_type);
  tmp.u64[1] = scaled_vec_conversion<uint2, uint32_t>(a.y, scale, fp8_type);
  return tmp.u64x2;
}

// fp8 -> __nv_bfloat16
template <>
__inline__ __device__ __nv_bfloat16
scaled_vec_conversion<__nv_bfloat16, uint8_t>(
    const uint8_t& a, const float scale,
    const __nv_fp8_interpretation_t fp8_type) {
  // Note there is no direct convert function from fp8 to bf16.
  // fp8 -> half
  __half_raw res = __nv_cvt_fp8_to_halfraw(a, fp8_type);
  // half -> float -> bf16
  float tmp = half_to_float(res.x);
  return __float2bfloat16(tmp * scale);
}

// fp8x2 -> __nv_bfloat162
template <>
__inline__ __device__ __nv_bfloat162
scaled_vec_conversion<__nv_bfloat162, uint16_t>(
    const uint16_t& a, const float scale,
    const __nv_fp8_interpretation_t fp8_type) {
  __nv_bfloat162 res;
  res.x = scaled_vec_conversion<__nv_bfloat16, uint8_t>((uint8_t)a, scale,
                                                        fp8_type);
  res.y = scaled_vec_conversion<__nv_bfloat16, uint8_t>((uint8_t)(a >> 8U),
                                                        scale, fp8_type);
  return res;
}

// fp8x4 -> bf16_4_t
template <>
__inline__ __device__ bf16_4_t scaled_vec_conversion<bf16_4_t, uint32_t>(
    const uint32_t& a, const float scale,
    const __nv_fp8_interpretation_t fp8_type) {
  bf16_4_t res;
  res.x = scaled_vec_conversion<__nv_bfloat162, uint16_t>((uint16_t)a, scale,
                                                          fp8_type);
  res.y = scaled_vec_conversion<__nv_bfloat162, uint16_t>((uint16_t)(a >> 16U),
                                                          scale, fp8_type);
  return res;
}

// fp8x8 -> bf16_8_t
template <>
__inline__ __device__ bf16_8_t scaled_vec_conversion<bf16_8_t, uint2>(
    const uint2& a, const float scale,
    const __nv_fp8_interpretation_t fp8_type) {
  bf16_4_t tmp1, tmp2;
  tmp1 = scaled_vec_conversion<bf16_4_t, uint32_t>(a.x, scale, fp8_type);
  tmp2 = scaled_vec_conversion<bf16_4_t, uint32_t>(a.y, scale, fp8_type);
  bf16_8_t res;
  res.x = tmp1.x;
  res.y = tmp1.y;
  res.z = tmp2.x;
  res.w = tmp2.y;
  return res;
}

// fp8 -> float
template <>
__inline__ __device__ float scaled_vec_conversion<float, uint8_t>(
    const uint8_t& a, const float scale,
    const __nv_fp8_interpretation_t fp8_type) {
  // fp8 -> half
  __half_raw res = __nv_cvt_fp8_to_halfraw(a, fp8_type);
  uint16_t tmp = res.x;

  // half -> float
  return half_to_float(tmp) * scale;
}

// fp8x2 -> float2
template <>
__inline__ __device__ float2 scaled_vec_conversion<float2, uint16_t>(
    const uint16_t& a, const float scale,
    const __nv_fp8_interpretation_t fp8_type) {
  // fp8x2 -> half2
  uint32_t tmp = scaled_vec_conversion<uint32_t, uint16_t>(a, scale, fp8_type);
  // half2 -> float2
  return half2_to_float2(tmp);
}

// fp8x4 -> float4
template <>
__inline__ __device__ Float4_ scaled_vec_conversion<Float4_, uint32_t>(
    const uint32_t& a, const float scale,
    const __nv_fp8_interpretation_t fp8_type) {
  Float4_ res;
  res.x = scaled_vec_conversion<float2, uint16_t>((uint16_t)a, scale, fp8_type);
  res.y = scaled_vec_conversion<float2, uint16_t>((uint16_t)(a >> 16U), scale,
                                                  fp8_type);
  return res;
}

// fp8x8 -> float8
template <>
__inline__ __device__ Float8_ scaled_vec_conversion<Float8_, uint2>(
    const uint2& a, const float scale,
    const __nv_fp8_interpretation_t fp8_type) {
  Float4_ tmp1, tmp2;
  tmp1 = scaled_vec_conversion<Float4_, uint32_t>(a.x, scale, fp8_type);
  tmp2 = scaled_vec_conversion<Float4_, uint32_t>(a.y, scale, fp8_type);
  Float8_ res;
  res.x = tmp1.x;
  res.y = tmp1.y;
  res.z = tmp2.x;
  res.w = tmp2.y;
  return res;
}

// half -> fp8
template <>
__inline__ __device__ uint8_t scaled_vec_conversion<uint8_t, uint16_t>(
    const uint16_t& a, const float scale,
    const __nv_fp8_interpretation_t fp8_type) {
  __nv_fp8_storage_t res =
      __nv_cvt_float_to_fp8(half_to_float(a) / scale, __NV_SATFINITE, fp8_type);
  return (uint8_t)res;
}

// bf16 -> fp8
template <>
__inline__ __device__ uint8_t scaled_vec_conversion<uint8_t, __nv_bfloat16>(
    const __nv_bfloat16& a, const float scale,
    const __nv_fp8_interpretation_t fp8_type) {
    #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  assert(false);
    #else
  __nv_fp8_storage_t res = __nv_cvt_float_to_fp8(__bfloat162float(a) / scale,
                                                 __NV_SATFINITE, fp8_type);
  return (uint8_t)res;
    #endif
  __builtin_unreachable();  // Suppress missing return statement warning
}

// float -> fp8
template <>
__inline__ __device__ uint8_t scaled_vec_conversion<uint8_t, float>(
    const float& a, const float scale,
    const __nv_fp8_interpretation_t fp8_type) {
  __nv_fp8_storage_t res =
      __nv_cvt_float_to_fp8(a / scale, __NV_SATFINITE, fp8_type);
  return (uint8_t)res;
}

// fp8x4 -> float4
template <>
__inline__ __device__ float4 scaled_vec_conversion<float4, uint32_t>(
    const uint32_t& a, const float scale,
    const __nv_fp8_interpretation_t fp8_type) {
  Float4_ tmp = scaled_vec_conversion<Float4_, uint32_t>(a, scale, fp8_type);
  float4 res = make_float4(tmp.x.x, tmp.x.y, tmp.y.x, tmp.y.y);
  return res;
}
  #endif  // ENABLE_FP8

// 中文注释: 根据 KV cache 数据类型分发的无缩放转换接口
// 用于 KV cache 使用 FP8 存储时的数据类型转换
// 参数 kv_dt 决定使用 E4M3 还是 E5M2 格式
// 注意: 当前代码被 #if 0 禁用以减小二进制大小
template <typename Tout, typename Tin, Fp8KVCacheDataType kv_dt>
__inline__ __device__ Tout convert(const Tin& x) {
  #if 0  // Disable the following code to reduce the binary size.
  if constexpr (kv_dt == Fp8KVCacheDataType::kFp8E4M3) {
    return vec_conversion<Tout, Tin>(x, __NV_E4M3);
  } else if constexpr (kv_dt == Fp8KVCacheDataType::kFp8E5M2) {
    return vec_conversion<Tout, Tin>(x, __NV_E5M2);
  }
  #endif
  assert(false);
  __builtin_unreachable();  // Suppress missing return statement warning
}

// 中文注释: 根据 KV cache 数据类型分发的带缩放转换接口
// 用于 KV cache 使用 FP8 + per-token 缩放因子的场景
// 典型流程: 写入 KV cache 时量化 (HP / scale -> FP8)，读取时反量化 (FP8 * scale -> HP)
template <typename Tout, typename Tin, Fp8KVCacheDataType kv_dt>
__inline__ __device__ Tout scaled_convert(const Tin& x, const float scale) {
  #ifdef ENABLE_FP8
  if constexpr (kv_dt == Fp8KVCacheDataType::kFp8E4M3) {
    return scaled_vec_conversion<Tout, Tin>(x, scale, __NV_E4M3);
  } else if constexpr (kv_dt == Fp8KVCacheDataType::kFp8E5M2) {
    return scaled_vec_conversion<Tout, Tin>(x, scale, __NV_E5M2);
  }
  #endif
  assert(false);
  __builtin_unreachable();  // Suppress missing return statement warning
}

  // The following macro is used to dispatch the conversion function based on
  // the data type of the key and value cache. The FN is a macro that calls a
  // function with template<typename scalar_t, typename cache_t,
  // Fp8KVCacheDataType kv_dt>.
  // 中文注释: KV cache 数据类型分发宏
  // 根据源数据类型 (float/half/bf16) 和 KV cache 数据类型 (Auto/E4M3/E5M2)
  // 自动选择正确的模板实例化
  // 使用方式: DISPATCH_BY_KV_CACHE_DTYPE(src_dtype, kv_dtype, my_kernel_func)
  // 其中 my_kernel_func 必须是一个接受 <scalar_t, cache_t, kv_dt> 模板参数的宏
  #define DISPATCH_BY_KV_CACHE_DTYPE(SRC_DTYPE, KV_DTYPE, FN)                  \
    vllm::Fp8KVCacheDataType KV_CACHE_DTYPE =                                  \
        vllm::get_fp8_kv_cache_data_type(KV_DTYPE);                            \
    if (KV_CACHE_DTYPE == vllm::Fp8KVCacheDataType::kAuto) {                   \
      if (SRC_DTYPE == torch::headeronly::ScalarType::Float) {                 \
        FN(float, float, vllm::Fp8KVCacheDataType::kAuto);                     \
      } else if (SRC_DTYPE == torch::headeronly::ScalarType::Half) {           \
        FN(uint16_t, uint16_t, vllm::Fp8KVCacheDataType::kAuto);               \
      } else if (SRC_DTYPE == torch::headeronly::ScalarType::BFloat16) {       \
        FN(__nv_bfloat16, __nv_bfloat16, vllm::Fp8KVCacheDataType::kAuto);     \
      } else {                                                                 \
        STD_TORCH_CHECK(false,                                                 \
                        "Unsupported input type of kv cache: ", SRC_DTYPE);    \
      }                                                                        \
    } else if (KV_CACHE_DTYPE == vllm::Fp8KVCacheDataType::kFp8E4M3) {         \
      if (SRC_DTYPE == torch::headeronly::ScalarType::Float) {                 \
        FN(float, uint8_t, vllm::Fp8KVCacheDataType::kFp8E4M3);                \
      } else if (SRC_DTYPE == torch::headeronly::ScalarType::Half) {           \
        FN(uint16_t, uint8_t, vllm::Fp8KVCacheDataType::kFp8E4M3);             \
      } else if (SRC_DTYPE == torch::headeronly::ScalarType::BFloat16) {       \
        FN(__nv_bfloat16, uint8_t, vllm::Fp8KVCacheDataType::kFp8E4M3);        \
      } else {                                                                 \
        STD_TORCH_CHECK(false,                                                 \
                        "Unsupported input type of kv cache: ", SRC_DTYPE);    \
      }                                                                        \
    } else if (KV_CACHE_DTYPE == vllm::Fp8KVCacheDataType::kFp8E5M2) {         \
      if (SRC_DTYPE == torch::headeronly::ScalarType::Float) {                 \
        FN(float, uint8_t, vllm::Fp8KVCacheDataType::kFp8E5M2);                \
      } else if (SRC_DTYPE == torch::headeronly::ScalarType::Half) {           \
        FN(uint16_t, uint8_t, vllm::Fp8KVCacheDataType::kFp8E5M2);             \
      } else if (SRC_DTYPE == torch::headeronly::ScalarType::BFloat16) {       \
        FN(__nv_bfloat16, uint8_t, vllm::Fp8KVCacheDataType::kFp8E5M2);        \
      } else {                                                                 \
        STD_TORCH_CHECK(false,                                                 \
                        "Unsupported input type of kv cache: ", SRC_DTYPE);    \
      }                                                                        \
    } else {                                                                   \
      STD_TORCH_CHECK(false, "Unsupported data type of kv cache: ", KV_DTYPE); \
    }

}  // namespace fp8
#endif  // not USE_ROCM
}  // namespace vllm
