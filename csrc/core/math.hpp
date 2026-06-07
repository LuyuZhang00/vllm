#pragma once

#include <climits>
#include <iostream>

// 中文注释：模块功能概述
// =========================================================================
// math.hpp 提供常用的数学工具函数
//
// 这些函数都是编译期可计算的（constexpr），用于：
// 1. 内存对齐：确保数据地址或大小满足对齐要求
// 2. 分块计算：将数据分成固定大小的块
// 3. 性能优化：使用位运算替代乘除法
//
// 典型应用场景：
// - KV cache block 大小对齐
// - 线程块大小计算
// - 内存分配对齐
// =========================================================================

// 中文注释：计算大于等于 num 的最小 2 的幂次
// 例如：next_pow_2(5) = 8, next_pow_2(8) = 8, next_pow_2(9) = 16
//
// 实现原理：
// 1. num - 1：将目标值减1，得到需要填充的位数
// 2. __builtin_clz：计算前导零的数量（count leading zeros）
// 3. CHAR_BIT * sizeof(num) - __builtin_clz：计算最高有效位的位置
// 4. 1 << position：将1左移到该位置，得到2的幂次
//
// 使用场景：
// - 计算线程块大小（必须是2的幂次）
// - 内存分配对齐（某些硬件要求对齐到2的幂次）
inline constexpr uint32_t next_pow_2(uint32_t const num) {
  if (num <= 1) return num;
  return 1 << (CHAR_BIT * sizeof(num) - __builtin_clz(num - 1));
}

// 中文注释：向上取整的除法（ceiling division）
// 计算 ceil(a / b)，即不小于 a/b 的最小整数
//
// 实现原理：
// (a + b - 1) / b 等价于 ceil(a / b)
// 例如：div_ceil(10, 3) = 4, div_ceil(9, 3) = 3
//
// 使用场景：
// - 计算需要多少个 block 来存储 N 个元素
// - 计算需要多少个线程块来处理 N 个任务
template <typename A, typename B>
static inline constexpr auto div_ceil(A a, B b) {
  return (a + b - 1) / b;
}

// 中文注释：向下对齐到 b 的倍数
// 将 a 向下舍入到最接近的 b 的倍数
// 例如：round_to_previous_multiple_of(10, 3) = 9
//       round_to_previous_multiple_of(9, 3) = 9
//
// 使用场景：
// - 内存地址对齐（确保地址是某个值的倍数）
// - 数据分块（确保块大小是对齐的）
template <typename T>
inline constexpr T round_to_previous_multiple_of(T a, T b) {
  return a % b == 0 ? a : (a / b) * b;
}

// 中文注释：向上对齐到 b 的倍数
// 将 a 向上舍入到最接近的 b 的倍数
// 例如：round_to_next_multiple_of(10, 3) = 12
//       round_to_next_multiple_of(9, 3) = 9
//
// 使用场景：
// - 确保缓冲区大小足够（向上对齐到页大小）
// - 计算需要分配的内存大小
template <typename T>
inline constexpr T round_to_next_multiple_of(T a, T b) {
  return a % b == 0 ? a : ((a / b) + 1) * b;
}
