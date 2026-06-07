#pragma once

// Shared TORCH_UTILS_CHECK across both libtorch stable and unstable source
// files. Keep this header free of CUTLASS/CUTE so attention/quant headers can
// use it.
//
// If TORCH_TARGET_VERSION is defined, we are building _C_stable_libtorch.so so
// use STD_TORCH_CHECK via header-only.
// Otherwise, use TORCH_CHECK via torch/all.h.

/*
 * =============================================================================
 * 文件功能概述（中文）
 * =============================================================================
 * 本文件提供跨 libtorch stable/unstable ABI 的条件检查宏。
 *
 * 【设计背景】
 *   vLLM 的 C++ 扩展分为两个构建目标：
 *     1. _C.so（unstable ABI）：使用完整的 torch/all.h，功能更丰富但 ABI 不稳定
 *     2. _C_stable_libtorch.so（stable ABI）：使用 header-only 的 torch 接口，
 *        ABI 稳定但功能有限
 *
 *   两种构建使用不同的断言宏：
 *     - unstable：TORCH_CHECK（来自 torch/all.h）
 *     - stable：STD_TORCH_CHECK（来自 torch/headeronly）
 *
 *   本文件通过 TORCH_UTILS_CHECK 宏统一两种构建的断言接口，
 *   使上层代码（attention、quantization 等）无需关心构建目标的差异。
 * =============================================================================
 */

#ifdef TORCH_TARGET_VERSION
  #include <torch/headeronly/util/Exception.h>
  #define TORCH_UTILS_CHECK STD_TORCH_CHECK
#else
  #include <torch/all.h>
  #define TORCH_UTILS_CHECK TORCH_CHECK
#endif
