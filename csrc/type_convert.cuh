#pragma once

/*
 * =============================================================================
 * 文件功能概述（中文）
 * =============================================================================
 * 本文件提供了 PyTorch 数据类型与 CUDA/HIP 原生类型之间的转换工具，
 * 以及用于向量化内存访问的 FP16/BF16 向量 POD 结构体。
 *
 * 【核心组件】
 *   1. _typeConvert<T> 模板结构体：
 *      将 PyTorch 类型（c10::Half、c10::BFloat16、float）映射到
 *      CUDA/HIP 原生类型（__half、__nv_bfloat16、float），并提供
 *      标量和打包（packed）类型的相互转换函数。
 *
 *   2. _f16Vec<scalar_t, width> 结构体：
 *      16 字节对齐的向量 POD 类型，用于 fused_add_rms_norm 等 kernel 中
 *      的向量化内存访问。支持 +=、*=、sum_squares 等运算，
 *      内部使用 128-bit（ld.128/st.128）指令以最大化显存带宽。
 *
 * 【设计背景】
 *   CUDA/HIP 对 half/bfloat16 的类型转换运算符实现不一致，
 *   无法通过通用的类型强制转换实现跨平台兼容。
 *   因此需要这些 converter 结构体封装平台特定的转换 intrinsics。
 * =============================================================================
 */

#include <torch/headeronly/util/BFloat16.h>
#include <torch/headeronly/util/Half.h>

#ifndef USE_ROCM
  #include <cuda.h>
  #include <cuda_bf16.h>
  #include <cuda_fp16.h>
#else
  #include <hip/hip_bf16.h>
  #include <hip/hip_fp16.h>

using __nv_bfloat16 = __hip_bfloat16;
using __nv_bfloat162 = __hip_bfloat162;
#endif

namespace vllm {
/* Converter structs for the conversion from torch types to HIP/CUDA types,
   and the associated type conversions within HIP/CUDA. These helpers need
   to be implemented for now because the relevant type conversion
   operators/constructors are not consistently implemented by HIP/CUDA, so
   a generic conversion via type casts cannot be implemented.

   Each struct should have the member static constexpr bool `exists`:
   If false, the optimized kernel is not used for the corresponding torch type.
   If true, the struct should be fully defined as shown in the examples below.
 */
// 【_typeConvert<T>】类型转换器模板
// 默认情况下 exists = false，表示该类型不支持优化的 kernel 路径。
// 每个特化版本需要定义：
//   - exists = true：表示支持
//   - hip_type：CUDA/HIP 原生标量类型
//   - packed_hip_type：打包类型（如 half2、bfloat162）
//   - packed_hip_type4：128 位打包类型（仅 float 特化提供）
//   - convert() 系列静态函数：实现标量和打包类型之间的相互转换
template <typename torch_type>
struct _typeConvert {
  static constexpr bool exists = false;
};

// 【float 特化】float 无需转换，直接透传。
// packed_hip_type = float2（64 位），packed_hip_type4 = float4（128 位）。
template <>
struct _typeConvert<float> {
  static constexpr bool exists = true;
  using hip_type = float;
  using packed_hip_type = float2;
  using packed_hip_type4 = float4;  // For 128-bit vectorization

  __device__ static __forceinline__ float convert(hip_type x) { return x; }
  __device__ static __forceinline__ float2 convert(packed_hip_type x) {
    return x;
  }
  __device__ static __forceinline__ float4 convert(packed_hip_type4 x) {
    return x;
  }
};

// 【c10::Half 特化】PyTorch Half -> CUDA __half 类型转换。
// 仅在 ROCm 或 CUDA >= 12.0 时启用（CUDA < 12.0 打包类型转换有问题）。
// 提供 half<->float 的标量和打包类型（half2<->float2）转换函数。
#if defined(USE_ROCM) || (defined(CUDA_VERSION) && (CUDA_VERSION >= 12000))
// CUDA < 12.0 runs into issues with packed type conversion
template <>
struct _typeConvert<c10::Half> {
  static constexpr bool exists = true;
  using hip_type = __half;
  using packed_hip_type = __half2;

  __device__ static __forceinline__ float convert(hip_type x) {
    return __half2float(x);
  }
  __device__ static __forceinline__ float2 convert(packed_hip_type x) {
    return __half22float2(x);
  }
  __device__ static __forceinline__ hip_type convert(float x) {
    return __float2half_rn(x);
  }
  __device__ static __forceinline__ packed_hip_type convert(float2 x) {
    return __float22half2_rn(x);
  }
};

// 【c10::BFloat16 特化】PyTorch BFloat16 -> CUDA __nv_bfloat16 类型转换。
// 仅在 SM >= 80（A100 及以上）或 ROCm 7.0+ 时启用，
// 因为更早的 GPU 架构不原生支持 bfloat16。
  #if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800) || defined(USE_ROCM)
// CUDA_ARCH < 800 does not have BF16 support
// ROCm 7.0+ supports bfloat16
template <>
struct _typeConvert<c10::BFloat16> {
  static constexpr bool exists = true;
  using hip_type = __nv_bfloat16;
  using packed_hip_type = __nv_bfloat162;

  __device__ static __forceinline__ float convert(hip_type x) {
    return __bfloat162float(x);
  }
  __device__ static __forceinline__ float2 convert(packed_hip_type x) {
    return __bfloat1622float2(x);
  }
  __device__ static __forceinline__ hip_type convert(float x) {
    return __float2bfloat16(x);
  }
  __device__ static __forceinline__ packed_hip_type convert(float2 x) {
    return __float22bfloat162_rn(x);
  }
};
  #endif  // (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800) ||
          // defined(USE_ROCM)
#endif    // defined(USE_ROCM) || (defined(CUDA_VERSION) && (CUDA_VERSION >=
          // 12000))

/* Vector POD struct to generate vectorized and packed FP16/BF16 ops
   for appropriate specializations of fused_add_rms_norm_kernel.
   Only functions that are necessary in that kernel are implemented.
   Alignment to 16 bytes is required to use 128-bit global memory ops.
 */
// 【_f16Vec<scalar_t, width>】向量化 FP16/BF16 运算的 POD 结构体
// 用于 fused_add_rms_norm 等 kernel 中的高效向量化内存访问。
//
// 设计要点：
//   - 对齐到 16 字节（alignas(16)），以便使用 128-bit 的 ld.128/st.128 指令
//   - width 必须是 2 的幂，以便利用打包类型（half2/bfloat162）进行 SIMD 运算
//   - 当 width 为偶数时，使用打包的 half2/bfloat162 运算，吞吐翻倍
//   - 当 width 为奇数时，退化为标量运算
//
// 提供的运算：
//   - operator+=：向量逐元素加法（用于残差连接）
//   - operator*=：向量逐元素乘法（用于缩放）
//   - operator*=(float)：向量标量乘法（用于 RMS Norm 的缩放）
//   - sum_squares()：计算向量元素的平方和（用于 RMS Norm 的均方根计算）
template <typename scalar_t, int width>
struct alignas(16) _f16Vec {
  /* Not theoretically necessary that width is a power of 2 but should
     almost always be the case for optimization purposes */
  static_assert(width > 0 && (width & (width - 1)) == 0,
                "Width is not a positive power of 2!");
  using Converter = _typeConvert<scalar_t>;
  using T1 = typename Converter::hip_type;
  using T2 = typename Converter::packed_hip_type;
  T1 data[width];

  __device__ _f16Vec& operator+=(const _f16Vec<scalar_t, width>& other) {
    if constexpr (width % 2 == 0) {
#pragma unroll
      for (int i = 0; i < width; i += 2) {
        if constexpr (std::is_same_v<T2, float2>) {
          data[i] += other.data[i];
          data[i + 1] += other.data[i + 1];
        } else {
          T2 temp{data[i], data[i + 1]};
          temp += T2{other.data[i], other.data[i + 1]};
          data[i] = temp.x;
          data[i + 1] = temp.y;
        }
      }
    } else {
#pragma unroll
      for (int i = 0; i < width; ++i) data[i] += other.data[i];
    }
    return *this;
  }

  __device__ _f16Vec& operator*=(const _f16Vec<scalar_t, width>& other) {
    if constexpr (width % 2 == 0) {
#pragma unroll
      for (int i = 0; i < width; i += 2) {
        if constexpr (std::is_same_v<T2, float2>) {
          data[i] *= other.data[i];
          data[i + 1] *= other.data[i + 1];
        } else {
          T2 temp{data[i], data[i + 1]};
          temp *= T2{other.data[i], other.data[i + 1]};
          data[i] = temp.x;
          data[i + 1] = temp.y;
        }
      }
    } else {
#pragma unroll
      for (int i = 0; i < width; ++i) data[i] *= other.data[i];
    }
    return *this;
  }

  __device__ _f16Vec& operator*=(const float scale) {
    if constexpr (width % 2 == 0) {
#pragma unroll
      for (int i = 0; i < width; i += 2) {
        float2 temp_f = Converter::convert(T2{data[i], data[i + 1]});
        temp_f.x *= scale;
        temp_f.y *= scale;
        T2 temp = Converter::convert(temp_f);
        data[i] = temp.x;
        data[i + 1] = temp.y;
      }
    } else {
#pragma unroll
      for (int i = 0; i < width; ++i) {
        float temp = Converter::convert(data[i]) * scale;
        data[i] = Converter::convert(temp);
      }
    }
    return *this;
  }

  __device__ float sum_squares() const {
    float result = 0.0f;
    if constexpr (width % 2 == 0) {
#pragma unroll
      for (int i = 0; i < width; i += 2) {
        float2 z = Converter::convert(T2{data[i], data[i + 1]});
        result += z.x * z.x + z.y * z.y;
      }
    } else {
#pragma unroll
      for (int i = 0; i < width; ++i) {
        float x = Converter::convert(data[i]);
        result += x * x;
      }
    }
    return result;
  }
};
}  // namespace vllm
