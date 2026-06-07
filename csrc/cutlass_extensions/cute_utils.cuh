#pragma once

// 中文注释：本文件提供 CuTe（CUTE Tensor Extension）相关的辅助工具，包含三大类功能：
//   1. 布局（Layout）操作工具：维度重排、恒等布局判断
//   2. 指针工具：处理 sub-byte 类型（如 4-bit）的逻辑指针
//   3. 杂项工具：自动向量化内存拷贝优化
// 这些工具被 vLLM 的 CUTLASS 扩展广泛使用，用于高效地操作张量的内存布局。

#include <cute/tensor.hpp>
namespace cute {

////////////////////////////////////////////////////////////////////
// layout utils
////////////////////////////////////////////////////////////////////

// Permute layout based on indices, example:
//   permute_layout<1, 0>(layout) will swap the two dimensions
//   permute_layout<0, 2, 1>(layout) will swap the last two dimensions
// 中文注释：按给定的索引序列重排 Layout 的维度。
// 例如 permute_layout<1, 0>(layout) 交换两个维度，permute_layout<0, 2, 1>(layout) 交换后两个维度。
// 这在 GEMM 中调整矩阵的行/列存储顺序时非常有用。
template <size_t... I, typename Layout>
CUTE_HOST_DEVICE static constexpr auto permute_layout(Layout l) {
  static_assert(rank(l) == sizeof...(I), "Invalid permutation, rank mismatch");
  return cute::make_layout(cute::get<I>(l)...);
}

// is the layout f(x) = x
// 中文注释：判断一个 Layout 是否是恒等映射（即 f(x) = x，步长为 1 的连续布局）。
// 恒等布局意味着数据在内存中是紧密连续存放的，无需做任何重排。
// 对于 void 类型（表示无需布局转换的情况），也返回 true。
template <typename Layout>
CUTE_HOST_DEVICE static constexpr bool is_identity_layout() {
  if constexpr (std::is_same_v<Layout, void>) {
    return true;
  } else {
    constexpr auto coalesced_layout = coalesce(Layout{});
    if constexpr (rank(coalesced_layout) == 1 &&
                  stride<0>(coalesced_layout) == 1) {
      return true;
    }
    return false;
  }
}

////////////////////////////////////////////////////////////////////
// Pointer utils
////////////////////////////////////////////////////////////////////

// 中文注释：对于 sub-byte 类型（如 4-bit 整数，sizeof_bits < 8），普通的 char* 指针
// 无法直接索引到单个元素。此函数返回一个 subbyte_iterator，使其可以像普通指针一样
// 按元素索引。对于 byte 及更大的类型，直接返回原始指针。
template <class PointerType>
static constexpr auto get_logical_ptr(PointerType* ptr) {
  if constexpr (cute::sizeof_bits_v<PointerType> < 8) {
    return cute::subbyte_iterator<PointerType>(ptr);
  } else {
    return ptr;
  }
}

////////////////////////////////////////////////////////////////////
// Misc utils
////////////////////////////////////////////////////////////////////

// 中文注释：创建一个自动向量化的拷贝操作对象。根据数据类型 T 和元素数量 Elements
// 计算总 bit 数，并选择最大的对齐粒度（128/64/32/16/8 bit）来执行内存拷贝。
// 更大的对齐粒度意味着可以用更少的指令完成拷贝，提升带宽利用率。
// 例如：对于 half_t（16bit）x 8 个元素 = 128 bit，会使用 128-bit 对齐的拷贝。
template <typename T, typename Elements>
CUTE_HOST_DEVICE static constexpr auto create_auto_vectorizing_copy() {
  constexpr auto bits = sizeof_bits_v<T> * Elements{};
  if constexpr (bits % 128 == 0) {
    return AutoVectorizingCopyWithAssumedAlignment<128>{};
  } else if constexpr (bits % 64 == 0) {
    return AutoVectorizingCopyWithAssumedAlignment<64>{};
  } else if constexpr (bits % 32 == 0) {
    return AutoVectorizingCopyWithAssumedAlignment<32>{};
  } else if constexpr (bits % 16 == 0) {
    return AutoVectorizingCopyWithAssumedAlignment<16>{};
  } else {
    return AutoVectorizingCopyWithAssumedAlignment<8>{};
  }
}

};  // namespace cute
