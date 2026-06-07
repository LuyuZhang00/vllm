#pragma once

// =============================================================================
// 中文注释: Machete 交错布局工具
// =============================================================================
// 本文件提供了权重数据交错 (interleave) 布局的计算工具。
//
// 为什么要交错:
//   - 量化权重在内存中按 packed 格式存储 (如 4-bit: 每 int32 存 8 个值)
//   - Tensor Core 的 MMA 指令需要特定的数据布局
//   - 交错布局使得在寄存器中反量化时可以使用更高效的操作
//
// 交错布局的含义:
//   - bit_stride: 相邻元素之间的位间隔
//   - blk_bit_width: 每个 block 的位宽度 (通常是 32，即一个 int32)
//   - 返回一个 Layout 对象，描述元素在 block 内的排列方式
//
// 示例:
//   - T=uint8, bit_stride=8, blk=32: 每个 int32 存 4 个 uint8，无交错 (4:1)
//   - T=uint8, bit_stride=16, blk=32: 每 16-bit 一个元素，2x2 交错
//   - T=uint4, bit_stride=8, blk=32: 每 8-bit 2 个 uint4，4x2 交错
//   - T=uint4, bit_stride=16, blk=32: 每 16-bit 4 个 uint4，2x4 交错
// =============================================================================

#include "cutlass/cutlass.h"
#include "cute/layout.hpp"

namespace machete {

using namespace cute;

// get an interleaved block layout where each element consecutive element has a
// stride of bit_stride and the block width is blk_bit_width,
// examples:
//  size_bits<T> = 8, bit_stride = 8,  blk_bit_width = 32 -> 4:1
//  size_bits<T> = 8, bit_stride = 16, blk_bit_width = 32 -> (2, 2):(2, 1)
//  size_bits<T> = 4, bit_stride = 8,  blk_bit_width = 32 -> (4, 2):(2, 1)
//  size_bits<T> = 4, bit_stride = 16, blk_bit_width = 32 -> (2, 4):(4, 1)
// 中文注释: 计算交错布局的模板函数
// 根据元素类型 T、位间隔 bit_stride 和 block 位宽度 blk_bit_width，
// 返回一个 CUTE Layout 对象描述交错后的内存布局
template <typename T, int bit_stride, int blk_bit_width>
CUTE_HOST_DEVICE static constexpr auto get_interleaved_blk_layout() {
  static_assert(blk_bit_width % bit_stride == 0);
  static_assert(bit_stride % cute::sizeof_bits_v<T> == 0);

  constexpr auto elems_per_blk = blk_bit_width / cute::sizeof_bits_v<T>;

  if constexpr (cute::sizeof_bits_v<T> == bit_stride) {
    // identity layout
    return Layout<Shape<Int<elems_per_blk>>>{};
  } else {
    constexpr auto elems_per_stride = bit_stride / cute::sizeof_bits_v<T>;
    constexpr auto num_strides = elems_per_blk / elems_per_stride;
    return Layout<Shape<Int<num_strides>, Int<elems_per_stride>>,
                  Stride<Int<elems_per_stride>, Int<1>>>{};
  }
}

};  // namespace machete
