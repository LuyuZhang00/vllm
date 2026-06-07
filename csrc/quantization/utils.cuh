#pragma once

// =============================================================================
// 中文注释: 量化公共工具头文件
// =============================================================================
// 本文件提供量化操作的基础工具，被 w8a8、fused_kernels 等多个量化模块共享引用。
//
// 核心功能:
//   1. quant_type_max: 获取各量化类型的最大可表示值
//      - Float8_e4m3fn: 448.0 (NVIDIA OCP FP8 标准)
//      - Float8_e4m3fnuz: 240.0 (AMD ROCm FP8 变体，使用 224.0 避免精度问题)
//      - int8_t: 127 (标准有符号 8 位整数)
//
//   2. min_scaling_factor: 获取安全的最小缩放因子
//      - 防止除零错误 (当输入全为 0 时)
//      - FP8: 1.0 / (max_val * 512.0)
//      - INT8: float epsilon
//
// 设计原因:
//   - FP8 的 e4m3 格式精度有限，使用 224.0 而非 240.0 作为 max 可以
//     避免动态量化时的溢出/精度损失问题
//   - 缩放因子下限确保即使在极端输入下也不会产生 NaN/Inf
// =============================================================================

/**
 * Quantization utilities including:
 *   Adjusted maximum values for qtypes.
 *   Minimum scaling factors for qtypes.
 */

#include <cmath>
#include <torch/headeronly/macros/Macros.h>

#ifndef USE_ROCM
  #include <torch/headeronly/util/Float8_e4m3fn.h>
  #define MAYBE_HOST_DEVICE C10_HOST_DEVICE
#else
  #include <torch/headeronly/util/Float8_e4m3fn.h>
  #include <torch/headeronly/util/Float8_e4m3fnuz.h>
  // ROCm doesn't seem to need C10_HOST_DEVICE for static constexpr
  #define MAYBE_HOST_DEVICE
#endif

// 中文注释: 量化类型最大值模板
// 默认实现: 直接使用类型自身的 numeric_limits::max()
// 适用于 Float8_e4m3fn (448.0) 和 int8_t (127)
template <typename T,
          typename = std::enable_if_t<
              std::is_same_v<T, torch::headeronly::Float8_e4m3fn> ||
              std::is_same_v<T, torch::headeronly::Float8_e4m3fnuz> ||
              std::is_same_v<T, int8_t>>>
struct quant_type_max {
  static constexpr T val() { return std::numeric_limits<T>::max(); }
};

// Using the default max value from pytorch (240.0 0x7F) will cause accuracy
// issues when running dynamic quantization. Here use 224.0 0x7E for rocm.
// 中文注释: ROCm FP8 特化 -- 使用 224.0 (0x7E) 而非标准的 240.0 (0x7F)
// 原因: ROCm 的 FNUZ 变体在动态量化中使用 240.0 会导致精度问题
// 224.0 提供了更好的数值稳定性
template <>
struct quant_type_max<torch::headeronly::Float8_e4m3fnuz> {
  static constexpr torch::headeronly::Float8_e4m3fnuz val() {
    return torch::headeronly::Float8_e4m3fnuz(
        0x7E, torch::headeronly::Float8_e4m3fnuz::from_bits());
  }
};

// 中文注释: 便捷访问接口 quant_type_max_v<T>，用于 kernel 中快速获取量化范围
// 例如: float scale = max_val / quant_type_max_v<fp8_type>
template <typename T>
MAYBE_HOST_DEVICE static constexpr T quant_type_max_v =
    quant_type_max<T>::val();

// 中文注释: 最小安全缩放因子模板
// 作用: 防止量化过程中除以过小的缩放因子导致数值溢出
// 计算公式: 1.0 / (quant_max * 512.0)，确保 scale 不会太小
// 使用场景: 当输入张量的最大绝对值非常小时，限制缩放因子下限
template <typename T,
          typename = std::enable_if_t<
              std::is_same_v<T, torch::headeronly::Float8_e4m3fn> ||
              std::is_same_v<T, torch::headeronly::Float8_e4m3fnuz> ||
              std::is_same_v<T, int8_t>>>
struct min_scaling_factor {
  C10_DEVICE C10_ALWAYS_INLINE static float val() {
    return 1.0f / (quant_type_max_v<T> * 512.0f);
  }
};

// 中文注释: INT8 特化 -- 使用 float epsilon 作为最小缩放因子
// INT8 范围较小 ([-127, 127])，使用 epsilon 即可提供足够的数值保护
template <>
struct min_scaling_factor<int8_t> {
  C10_DEVICE C10_ALWAYS_INLINE static float val() {
    return std::numeric_limits<float>::epsilon();
  }
};