#pragma once

// 中文注释：本文件定义了 vLLM 专用的量化数据类型，主要用于 GPTQ 等量化算法。
// GPTQ（全精度量化训练后量化）使用对称量化，需要带有偏置（bias）的无符号整数类型。
// 本文件定义了两个关键类型：
//   1. vllm_uint4b8_t：4-bit 无符号整数，偏置为 8（即存储值 = 真实值 + 8）
//      用于 GPTQ 4-bit 量化，真实值范围 [-8, 7]，存储值范围 [0, 15]
//   2. vllm_uint8b128_t：8-bit 无符号整数，偏置为 128（即存储值 = 真实值 + 128）
//      用于 GPTQ 8-bit 量化，真实值范围 [-128, 127]，存储值范围 [0, 255]
//
// 这些类型继承自 CUTLASS 的 integer_subbyte，使其可以在 CUTLASS 的 GEMM 和 epilogue 中使用。

#include "cutlass/integer_subbyte.h"

namespace cutlass {

///////////////////////////////////////////////////////////////////////////////////////////////////

// 中文注释：带偏置的 sub-byte 整数类型基类模板。
// Bits: 位宽（4 或 8），Bias: 偏置值，Signed: 是否有符号。
// 继承自 CUTLASS 的 integer_subbyte，使得该类型可以直接用于 CUTLASS 的类型系统中。
template <int Bits, int Bias, bool Signed = false>
struct vllm_biased_integer_subbyte : public integer_subbyte<Bits, Signed> {
  using Base = integer_subbyte<Bits, Signed>;

  using Storage = typename Base::Storage;
  using xint_t = typename Base::xint_t;

  using Base::bits_mask_;
  using Base::sign_mask_;
  using Base::storage;

  //
  // Methods
  //

  /// No operation
  vllm_biased_integer_subbyte() = default;

  /// Conversion from integer type
  CUTLASS_HOST_DEVICE explicit vllm_biased_integer_subbyte(int value)
      : Base(value) {}
  CUTLASS_HOST_DEVICE explicit vllm_biased_integer_subbyte(unsigned value)
      : Base(value) {}
  CUTLASS_HOST_DEVICE explicit vllm_biased_integer_subbyte(double value)
      : Base(value) {}
};
///////////////////////////////////////////////////////////////////////////////////////////////////

// "GPTQ" types, i.e. symmetric quantization
// 中文注释：GPTQ 量化使用的具体类型定义。
//   - vllm_uint4b8_t (u4b8)：4-bit 量化，偏置 8。用于 4-bit 权重量化（如 GPTQ-4bit）。
//   - vllm_uint8b128_t (u8b128)：8-bit 量化，偏置 128。用于 8-bit 权重量化。
using vllm_uint4b8_t = vllm_biased_integer_subbyte<4, 8>;      // u4b8
using vllm_uint8b128_t = vllm_biased_integer_subbyte<8, 128>;  // u8b128

///////////////////////////////////////////////////////////////////////////////////////////////////

// 中文注释：特化 CUTLASS 的 sizeof_bits 模板，使 CUTLASS 能正确识别这些自定义类型的位宽。
// 这是 CUTLASS 类型系统的要求，让它知道 vllm_biased_integer_subbyte<4, ...> 占 4 bit。
template <int Bits, int Bias, bool Signed>
struct sizeof_bits<vllm_biased_integer_subbyte<Bits, Bias, Signed>> {
  static constexpr int value = Bits;
};

///////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace cutlass
