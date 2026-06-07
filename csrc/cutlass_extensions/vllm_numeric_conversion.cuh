#pragma once

// 中文注释：本文件是 vLLM 的数值类型转换核心文件，扩展了 CUTLASS 的 numeric_conversion.h。
// 主要功能：
//   1. 为 vLLM 自定义的 GPTQ 量化类型（vllm_uint4b8_t, vllm_uint8b128_t）提供高效的
//      GPU 向量化类型转换，支持转换到 int8, fp8_e4m3, half, bfloat16, float 等目标类型。
//   2. 提供 InterleavedNumericArrayConverter（交错数组转换器），针对 sub-byte 类型
//      采用交错布局（interleaved layout）可以更高效地从 32-bit 寄存器中提取 sub-byte 元素。
//   3. 所有转换都使用内联 PTX 汇编（prmt, lop3 等指令）实现，
//      在单个 warp 内将 32-bit 寄存器中打包的多个 sub-byte 值一次性转换，
//      避免逐元素处理，大幅提升吞吐量。
//
// 核心转换算法概述（以 4-bit -> FP16 为例）：
//   - 存储的 4-bit 值带有偏置 8，即 stored = (x + 8)
//   - 利用 FP16 的指数位，构造一个"魔数"浮点值
//   - 通过 FP16 减法去除偏置，得到正确的浮点结果
//   - 整个过程用 prmt + lop3 + hfma 等少量指令完成

#include "cutlass/numeric_conversion.h"
#include "cutlass_extensions/vllm_custom_types.cuh"
#include "cutlass_extensions/cute_utils.cuh"
#include "cutlass_extensions/vllm_type_utils.cuh"

// this file extends:
//   https://github.com/NVIDIA/cutlass/blob/cutlass-3.5.0/include/cutlass/numeric_conversion.h
// with vllm specific type conversions, namely: vllm_uint4b8_t, vllm_uint8b128_t
// as well as adds interleaved numeric array converters for specific types.
// (interleaved numeric array converters can be more efficient for subbyte
// types)

namespace cutlass {

// InterleavedNumericArrayConverter is like NumericArrayConverter but also
// deinterleaves converted elements based on IlvBlkLayout, interleaving can
// make subbyte converts more efficient by allowing for efficient extraction
// of subbyte elements from a 32bit register.
// 中文注释：交错数值数组转换器的主模板。
// 与标准 NumericArrayConverter 不同，它在转换的同时还根据 IlvBlkLayout 进行"去交错"操作。
// 交错布局（interleaved layout）的优势在于：对于 sub-byte 类型（如 4-bit），
// 从 32-bit 寄存器中提取元素时，交错排列可以让 prmt 等位操作指令一次性处理更多元素。
// 主模板是 fallback 实现，当没有找到匹配的特化时会打印错误信息。
template <typename IlvBlkLayout, typename T, typename S, int N,
          FloatRoundStyle Round = FloatRoundStyle::round_to_nearest,
          class Enable = void>
struct InterleavedNumericArrayConverter {
  using Converter = NumericArrayConverter<T, S, N, Round>;

  using result_type = typename Converter::result_type;
  using source_type = typename Converter::source_type;

  CUTLASS_DEVICE
  static result_type convert(source_type const& source) {
    if (cute::elect_one_sync()) {
      if constexpr (std::is_same_v<IlvBlkLayout, void>) {
        printf(
            "Convert %s <= %s (N = %d, IlvBlkLayout = void), not implemented\n",
            nameof_v<T>, nameof_v<S>, N);
      } else {
        printf(
            "Convert %s <= %s (N = %d, size(IlvBlkLayout{}) = %d), not "
            "implemented\n",
            nameof_v<T>, nameof_v<S>, N, size(IlvBlkLayout{}));
      }
      __brkpt();
    }
    return {};
  }

  CUTLASS_DEVICE
  result_type operator()(source_type const& s) const { return convert(s); }
};

// 中文注释：当交错布局是恒等映射（identity layout）时的特化。
// 此时不需要去交错操作，直接委托给标准的 NumericArrayConverter 即可。
// 这是一个优化：当布局本身就是连续的，交错转换退化为普通转换。
template <typename IlvBlkLayout, typename T, typename S, int N,
          FloatRoundStyle Round>
struct InterleavedNumericArrayConverter<
    IlvBlkLayout, T, S, N, Round,
    std::enable_if_t<is_identity_layout<IlvBlkLayout>()>> {
  using Converter = NumericArrayConverter<T, S, N, Round>;

  using result_type = typename Converter::result_type;
  using source_type = typename Converter::source_type;

  CUTLASS_DEVICE
  static result_type convert(source_type const& source) {
    return Converter::convert(source);
  }

  CUTLASS_DEVICE
  result_type operator()(source_type const& s) const { return convert(s); }
};

// 中文注释：基于 32-bit 寄存器的打包数组转换器。
// 核心思想：将多个 sub-byte 元素（如 8 个 4-bit 值）打包在 32-bit 寄存器中，
// 通过单个 32-bit 操作一次性转换所有元素，而不是逐元素处理。
//
// 模板参数：
//   - RegConvert32bit: 32-bit 寄存器级别的转换器，负责单个 32-bit 寄存器的转换
//   - T: 目标元素类型（如 half_t, bfloat16_t, int8_t）
//   - S: 源元素类型（如 vllm_uint4b8_t, vllm_uint8b128_t）
//   - N: 要转换的元素总数
//
// 工作流程：
//   1. 根据源类型位宽计算每个 32-bit 寄存器能容纳多少个源元素
//   2. 使用 VectorizedConverter 将 N 个元素分成 8/4/2 个一组进行批量转换
//   3. 每组通过 packed_convert 调用 RegConvert32bit 完成寄存器级转换
template <typename RegConvert32bit, typename T, typename S, int N>
struct ArrayConverterPacked32Bit {
  using result_type = Array<T, N>;
  using source_type = Array<S, N>;

  using result_packed_8_t = Array<T, 8>;
  using result_packed_4_t = Array<T, 4>;
  using result_packed_2_t = Array<T, 2>;
  using src_packed_8_t = Array<S, 8>;
  using src_packed_4_t = Array<S, 4>;
  using src_packed_2_t = Array<S, 2>;

  static_assert(N % 2 == 0, "N must be a multiple of 2");
  static_assert(cutlass::sizeof_bits_v<S> >= 4);  // TODO: add 16 packed sources
  static_assert(32 % cutlass::sizeof_bits_v<S> == 0);
  static constexpr auto src_elems_per_32bit_reg =
      32 / cutlass::sizeof_bits_v<S>;

  // Maybe not Valid. ScalarConverter will not actually work unless
  // NumericConverter<T, S, Round> is implemented. However it won't be used
  // anyways since we assert N % 2 == 0, just here for compliance with
  // VectorizedConverter.
  using ScalarConverter = NumericConverter<T, S>;

  // 中文注释：将打包的源数据类型转换为 uint32_t 寄存器数组。
  // 根据 PackedSrc 的大小（1/2/4/8 字节），选择合适的转换方式。
  // 这是将任意源类型"标准化"为 32-bit 寄存器的关键步骤。
  template <typename PackedSrc>
  CUTLASS_DEVICE static auto to_regs(PackedSrc const& src) {
    if constexpr (sizeof(PackedSrc) == 1) {
      return Array<uint32_t, 1>{reinterpret_cast<uint8_t const&>(src)};
    } else if constexpr (sizeof(PackedSrc) == 2) {
      return Array<uint32_t, 1>{reinterpret_cast<uint16_t const&>(src)};
    } else if constexpr (sizeof(PackedSrc) == 4) {
      return Array<uint32_t, 1>{reinterpret_cast<uint32_t const&>(src)};
    } else {
      static_assert(sizeof(PackedSrc) == 8);
      return reinterpret_cast<Array<uint32_t, 2> const&>(src);
    }
  }

  // The core converter uses bit tricks to construct a known FP16 number, then
  // does a subtraction in FP16 for the final result.
  // 中文注释：核心的打包转换函数。将打包的源数据（如 8 个 4-bit 值打包在 32-bit 中）
  // 转换为打包的目标类型（如 8 个 FP16 值）。
  // 实际转换由 RegConvert32bit 完成，此函数负责类型分发和断言检查。
  template <typename PackedResultType, typename PackedSrcType>
  CUTLASS_DEVICE static PackedResultType packed_convert(
      PackedSrcType const& source) {
    static_assert(PackedSrcType::kElements == PackedResultType::kElements);
    static_assert(PackedResultType::kElements == 2 ||
                      PackedResultType::kElements == 4 ||
                      PackedResultType::kElements == 8,
                  "Invalid PackedResultType must be 2, 4 or 8.");
    static_assert(std::is_same_v<typename PackedSrcType::Element, S>);
    static_assert(std::is_same_v<typename PackedResultType::Element, T>);

    return RegConvert32bit::template convert<PackedResultType>(to_regs(source));
  }

  friend class detail::VectorizedConverter;

 public:
  // 中文注释：主转换入口。根据源类型每个 32-bit 寄存器容纳的元素数，
  // 选择最优的分组策略进行向量化转换：
  //   - >= 8 个元素/寄存器：优先按 8 个一组转换，剩余按 4 和 2 处理
  //   - >= 4 个元素/寄存器：按 4 个一组转换
  //   - 其他：按 2 个一组转换
  CUTLASS_DEVICE static result_type convert(source_type const& source) {
    result_type result;
    using ConverterType =
        ArrayConverterPacked32Bit<RegConvert32bit,
                                  typename result_type::Element,
                                  typename source_type::Element, N>;

    if constexpr (src_elems_per_32bit_reg >= 8) {
      detail::VectorizedConverter::convert<
          ConverterType, result_packed_8_t, src_packed_8_t, result_packed_4_t,
          src_packed_4_t, result_packed_2_t, src_packed_2_t>(result, source);
    } else if constexpr (src_elems_per_32bit_reg >= 4) {
      detail::VectorizedConverter::convert<ConverterType, result_packed_4_t,
                                           src_packed_4_t, result_packed_2_t,
                                           src_packed_2_t>(result, source);
    } else {
      detail::VectorizedConverter::convert<ConverterType, result_packed_2_t,
                                           src_packed_2_t>(result, source);
    }

    return result;
  }
};

// Convert 8 4bit values packed into a 32bit register to 8 8bit values packed
// into 2 32bit register.
// 中文注释：使用查找表（LUT）将 32-bit 寄存器中打包的 8 个 4-bit 值转换为
// 2 个 32-bit 寄存器中的 8 个 8-bit 值。
//
// 算法流程：
//   1. 16 个模板参数 LUT0-LUT15 构成一个 16 项的查找表，索引 0-15 对应 4-bit 值
//   2. 将查找表分为高半部分（LUT8-15）和低半部分（LUT0-7），分别打包到 LOW/HIGH 常量中
//   3. 从源 32-bit 寄存器中提取高位（high_bit = bit3），用于选择高半或低半查找表
//   4. 提取低 3 位（lut_idx = bit[2:0]）作为查找表索引
//   5. 使用 PTX prmt.b32 指令（byte permute）在两个 32-bit 值之间按字节索引重排，
//      实现查找表查找。两次 prmt 分别查找低候选和高候选，再用 high_bit 选择最终结果。
//   6. 循环 2 次处理 32-bit 中的两个 16-bit 半字。
//
// 该函数是类型特定转换器（如 int8、fp8、half、bfloat16）的基础构建块。
// 不同目标类型通过传入不同的 LUT 值来实现不同的转换映射。
template <uint8_t LUT0, uint8_t LUT1, uint8_t LUT2, uint8_t LUT3,    //
          uint8_t LUT4, uint8_t LUT5, uint8_t LUT6, uint8_t LUT7,    //
          uint8_t LUT8, uint8_t LUT9, uint8_t LUT10, uint8_t LUT11,  //
          uint8_t LUT12, uint8_t LUT13, uint8_t LUT14, uint8_t LUT15>
CUTLASS_DEVICE cutlass::AlignedArray<uint32_t, 2> lut_4bit_to_8bit_convert(
    uint32_t src) {
  cutlass::AlignedArray<uint32_t, 2> r;
  // Determines if the value is in the top half of the LUT if set or
  //  (i.e. LUT[8:15]) in the bottom half (i.e. LUT[0:7]) if not set. Then move
  //  into bit position 0x4 of each nibble so when or'd with final_prmt_base it
  //  selects the correct candidate. When elements in final_prmt_base
  //  are >= 0x4, the high candidate is selected (i.e. LUT[8:15]), when elements
  //  are  < 0x4, the low candidate is selected (i.e. LUT[0:7])
  uint32_t high_bit = (src & 0x88888888) >> 1;

  // `high_bit` is OR'd with 0x31203120 to find the correct value in the LUT
  // (selects correct high or low candidate)
  const uint32_t final_prmt_base = 0x32103210;

  // Ignore the high bit when indexing into LUT, for each 4bit value
  //  we index into both the high and low candidates then use
  //  high_bit | final_prmt_base to select the correct candidate
  uint32_t lut_idx = (src & 0x77777777);

  auto pack = [](uint8_t a, uint8_t b, uint8_t c, uint8_t d) {
    return uint32_t(a) | (uint32_t(b) << 8) | (uint32_t(c) << 16) |
           (uint32_t(d) << 24);
  };

  static constexpr uint32_t LOW_0 = pack(LUT0, LUT1, LUT2, LUT3);
  static constexpr uint32_t LOW_1 = pack(LUT4, LUT5, LUT6, LUT7);
  static constexpr uint32_t HIGH_0 = pack(LUT8, LUT9, LUT10, LUT11);
  static constexpr uint32_t HIGH_1 = pack(LUT12, LUT13, LUT14, LUT15);

  CUTLASS_PRAGMA_UNROLL
  for (int ii = 0; ii < 2; ++ii, lut_idx >>= 16, high_bit >>= 16) {
    uint32_t final_prmt_idx = final_prmt_base | high_bit;

    // This uses a look up table to convert packed int4s to packed int8s,
    // using the int4 value as the index to prmt. It first select both the
    // high and low candidates, then uses the high bit (i.e. `high_bit`) to
    // select the correct candidate.
    asm volatile(
        "{\n"
        "  .reg .b32 low, high;\n"
        "  prmt.b32 low, %1, %2, %5;\n"
        "  prmt.b32 high, %3, %4, %5;\n"
        "  prmt.b32 %0, low, high, %6;\n"
        "}\n"
        : "=r"(r[ii])
        : "n"(LOW_0), "n"(LOW_1), "n"(HIGH_0), "n"(HIGH_1), "r"(lut_idx),
          "r"(final_prmt_idx));
  }

  return r;
};

// for Array<int8_t, N> <= Array<vllm_uint4b8_t, N>
// 中文注释：vllm_uint4b8_t (4-bit GPTQ，偏置 8) -> int8_t 的数组转换器。
// 使用 lut_4bit_to_8bit_convert，LUT 映射为：
//   4-bit 值 0-7  -> int8 的 0-7（正数部分）
//   4-bit 值 8-15 -> int8 的 -8 到 -1（负数部分，补码表示）
// 这对应 GPTQ 对称量化的真实值范围 [-8, 7]。
template <FloatRoundStyle Round, int N>
struct NumericArrayConverter<int8_t, vllm_uint4b8_t, N, Round> {
  using result_type = Array<int8_t, N>;
  using source_type = Array<vllm_uint4b8_t, N>;

  static FloatRoundStyle const round_style = Round;

 private:
  struct RegConvert {
    template <typename PackedResultType>
    CUTLASS_DEVICE static PackedResultType convert(Array<uint32_t, 1> src_) {
      // [-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7] as int8s
      auto r = lut_4bit_to_8bit_convert<0xF8, 0xF9, 0xFA, 0xFB,  //
                                        0xFC, 0xFD, 0xFE, 0xFF,  //
                                        0x00, 0x01, 0x02, 0x03,  //
                                        0x04, 0x05, 0x06, 0x07>(src_[0]);
      return reinterpret_cast<PackedResultType&>(r);
    };
  };

 public:
  CUTLASS_DEVICE
  static result_type convert(source_type const& source) {
    return ArrayConverterPacked32Bit<RegConvert, typename result_type::Element,
                                     typename source_type::Element,
                                     N>::convert(source);
  }

  CUTLASS_DEVICE
  result_type operator()(source_type const& s) const { return convert(s); }
};

// for Array<cutlass::float_e4m3_t, N> <= Array<vllm_uint4b8_t, N>
// 中文注释：vllm_uint4b8_t (4-bit GPTQ) -> FP8 E4M3 的数组转换器。
// LUT 将 4-bit 整数值映射到 FP8 E4M3 编码。用于支持 FP8 量化的混合精度 GEMM。
template <FloatRoundStyle Round, int N>
struct NumericArrayConverter<cutlass::float_e4m3_t, vllm_uint4b8_t, N, Round> {
  using result_type = Array<cutlass::float_e4m3_t, N>;
  using source_type = Array<vllm_uint4b8_t, N>;

  static FloatRoundStyle const round_style = Round;

 private:
  struct RegConvert {
    template <typename PackedResultType>
    CUTLASS_DEVICE static PackedResultType convert(Array<uint32_t, 1> src_) {
      // [-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7] as fp8s
      auto r = lut_4bit_to_8bit_convert<0xD0, 0xCE, 0xCC, 0xCA,  //
                                        0xC8, 0xC4, 0xC0, 0xB8,  //
                                        0x00, 0x38, 0x40, 0x44,  //
                                        0x48, 0x4A, 0x4C, 0x4E>(src_[0]);
      return reinterpret_cast<PackedResultType&>(r);
    };
  };

 public:
  CUTLASS_DEVICE
  static result_type convert(source_type const& source) {
    return ArrayConverterPacked32Bit<RegConvert, typename result_type::Element,
                                     typename source_type::Element,
                                     N>::convert(source);
  }

  CUTLASS_DEVICE
  result_type operator()(source_type const& s) const { return convert(s); }
};

// for Array<cutlass::half_t, N> <= Array<vllm_uint4b8_t, N>
// 中文注释：vllm_uint4b8_t (4-bit GPTQ，偏置 8) -> FP16 的数组转换器。
// 这是最常用的转换之一，用于将 GPTQ 4-bit 量化的权重在 GEMM 计算时解量化为 FP16。
//
// 转换算法（高效位操作法）：
//   1. 使用 prmt 指令将 32-bit 中 8 个 4-bit 值分拆到 4 个 FP16 槽中
//      每个槽包含低 4 位的原始值
//   2. 使用 lop3（AND + XOR 组合指令）设置 FP16 指数位为 0x6400（= 1024），
//      同时清除高位 nibble，构造形如 {1024 + (x+8)} 的 FP16 值
//   3. 使用 hfma（half-precision FMA）执行：
//      result = scale * constructed_val + bias
//      其中 scale = {1/16, 1}, bias = {-72, -1032}
//      这等价于：result = {x1, x0}，即去除了偏置 8 的真实值
//
// 这种方法比逐元素转换快得多，因为所有操作都在寄存器级别进行，没有分支。
template <FloatRoundStyle Round, int N>
struct NumericArrayConverter<cutlass::half_t, vllm_uint4b8_t, N, Round> {
  using result_type = Array<cutlass::half_t, N>;
  using source_type = Array<vllm_uint4b8_t, N>;

  struct RegConvert {
    template <typename PackedResultType>
    CUTLASS_DEVICE static PackedResultType convert(Array<uint32_t, 1> src_) {
      uint32_t src = src_[0];
      using RegArray =
          cutlass::AlignedArray<uint32_t, PackedResultType::kElements / 2,
                                sizeof(PackedResultType)>;
      RegArray r;

      // Below constructs the following temporary:
      // fp16s_01 = {0x00, i4_01, 0x00, i4_01}
      // fp16s_23 = {0x00, i4_23, 0x00, i4_23}
      // fp16s_45 = {0x00, i4_45, 0x00, i4_45}
      // fp16s_67 = {0x00, i4_67, 0x00, i4_67}
      // We use inline asm instead of __byte_perm intrinsic since we don't want
      // the documented (& 0x7) on the index. NVCC might be able to optimize it
      // out since the index is a constexpr, but we choose to be safe about it
      // here.
      uint32_t prmt_indices[4] = {0x4040, 0x4141, 0x4242, 0x4343};
      static_assert(RegArray::kElements <= 4,
                    "Too many inputs for F16 -> I4 vector converter");
      CUTLASS_PRAGMA_UNROLL
      for (int ii = 0; ii < RegArray::kElements; ++ii) {
        asm volatile(
            "{\n"
            "  prmt.b32 %0, %1, %2, %3;\n"
            "}\n"
            : "=r"(r[ii])
            : "r"(src), "n"(0), "r"(prmt_indices[ii]));
      }

      // Since the stored 4bit values are biased by 8 we get stored_val = (x+8)
      //  we are trying to construct x and a fp16 value
      // The below XOR does the following:
      //  1) Sets the exponent bits of the FP16 to the correct value for the
      //  FP16 magic_num. We will be constructing {1024+16*(x1+8), 1024+(x0+8)},
      //  where x1 in the high nibble and x0 is the low nibble then using hfma
      //  to subtract 1032 from that
      // The AND does the following:
      //  1) Clear the set bits for the int4 we will ignore.
      // We use lop3 so that we can use 1 instruction for AND and XOR.
      static constexpr uint32_t xor_mask = 0x64006400;
      static constexpr uint32_t and_mask = 0xFFF0FF0F;
      static constexpr uint32_t immLut = (0xf0 & 0xcc) ^ 0xaa;

      // For each operand, computes:
      // r[i] = (r[i] & and_mask) ^ xor_mask
      CUTLASS_PRAGMA_UNROLL
      for (int ii = 0; ii < RegArray::kElements; ++ii) {
        asm volatile(
            "{\n"
            "  lop3.b32 %0, %0, %1, %2, %3;\n"
            "}\n"
            : "+r"(r[ii])
            : "n"(and_mask), "n"(xor_mask), "n"(immLut));
      }

      // We will issue 2 hfmas that do the following:
      // {x1, x0} = {1024+16*(x1+8), 1024+(x0+8)} * {1/16, 1} - {72, 1032}
      //          = {x1 + 1152, x0 + 1032} * {1/16, 1} - {72, 1032}
      static constexpr uint32_t hfma_bias_rep = 0xD480E408;   // {72, 1032}
      static constexpr uint32_t hfma_scale_rep = 0x2C003C00;  // {1 / 16, 1}

      const half2& hfma_bias = reinterpret_cast<const half2&>(hfma_bias_rep);
      const half2& hfma_scale = reinterpret_cast<const half2&>(hfma_scale_rep);
      CUTLASS_PRAGMA_UNROLL
      for (int ii = 0; ii < RegArray::kElements; ++ii) {
        half2& fp16x2_val = reinterpret_cast<__half2&>(r[ii]);
        fp16x2_val = __hfma2(hfma_scale, fp16x2_val, hfma_bias);
      }

      return reinterpret_cast<PackedResultType&>(r);
    };
  };

 public:
  CUTLASS_DEVICE
  static result_type convert(source_type const& source) {
    return ArrayConverterPacked32Bit<RegConvert, typename result_type::Element,
                                     typename source_type::Element,
                                     N>::convert(source);
  }

  CUTLASS_DEVICE
  result_type operator()(source_type const& s) const { return convert(s); }
};

// for Array<cutlass::half_t, N> <= Array<vllm_uint4b8_t, N>
//   for IlvdLayout: (2, 4):(4, 1)
// 中文注释：vllm_uint4b8_t -> FP16 的交错布局转换器，交错布局为 (2,4):(4,1)。
// 交错布局将 8 个元素分为两组：低 nibble（bit[2:0]）和高 nibble（bit[6:4]），
// 分别处理后再合并。这种布局在某些 CUTLASS GEMM 内核中使用，
// 可以更好地利用寄存器位域，提高 sub-byte 类型的处理效率。
template <FloatRoundStyle Round, int N>
struct InterleavedNumericArrayConverter<Layout<Shape<_2, _4>, Stride<_4, _1>>,
                                        cutlass::half_t, vllm_uint4b8_t, N,
                                        Round, void> {
  using IlvdLayout = Layout<Shape<_2, _4>, Stride<_4, _1>>;
  static_assert(N % size(IlvdLayout{}) == 0);

  using result_type = Array<cutlass::half_t, N>;
  using source_type = Array<vllm_uint4b8_t, N>;

  static FloatRoundStyle const round_style = Round;

 private:
  struct RegConvert {
    template <typename PackedResultType>
    CUTLASS_DEVICE static PackedResultType convert(Array<uint32_t, 1> src_) {
      uint32_t src = src_[0];
      using RegArray =
          cutlass::AlignedArray<uint32_t, PackedResultType::kElements / 2,
                                sizeof(PackedResultType)>;
      RegArray r;

      static_assert(PackedResultType::kElements <= size(IlvdLayout{}));
      static constexpr uint32_t xor_mask = 0x64006400;

      for (int ii = 0; ii < RegArray::kElements; ii += 2) {
        auto src_ = src >> (4 * (ii));
        r[ii + 0] = src_;
        r[ii + 1] = src_;

        static constexpr uint32_t and_xor_imm_lut = (0xf0 & 0xcc) ^ 0xaa;

        static constexpr uint32_t low_nib_mask = 0x000F000F;
        static constexpr uint32_t high_nib_mask = 0x00F000F0;

        asm volatile(
            "{\n"
            "  lop3.b32 %0, %0, %1, %2, %3;\n"
            "}\n"
            : "+r"(r[ii + 0])
            : "n"(low_nib_mask), "n"(xor_mask), "n"(and_xor_imm_lut));

        asm volatile(
            "{\n"
            "  lop3.b32 %0, %0, %1, %2, %3;\n"
            "}\n"
            : "+r"(r[ii + 1])
            : "n"(high_nib_mask), "n"(xor_mask), "n"(and_xor_imm_lut));

        // For low nibble:
        //  {x1, x0} = {1024+(x1+8), 1024+(x0+8)} * {1, 1} - {1032, 1032}
        // For high nibble:
        //  {x1, x0} = {1024+16*(x1+8), 1024+16*(x0+8)} * {1/16, 1/16}
        //             - {72, 72}
        static constexpr uint32_t low_nib_bias = 0x64086408;    // {1032, 1032}
        static constexpr uint32_t high_nib_scale = 0x2C002C00;  // {1/16, 1/16}
        static constexpr uint32_t high_nib_bias = 0xD480D480;   // {-72, -72}

        {
          half2& fp16x2_val = reinterpret_cast<__half2&>(r[ii + 0]);
          fp16x2_val =
              __hsub2(fp16x2_val, reinterpret_cast<const half2&>(low_nib_bias));
        }

        {
          half2& fp16x2_val = reinterpret_cast<__half2&>(r[ii + 1]);
          fp16x2_val = __hfma2(fp16x2_val,
                               reinterpret_cast<const half2&>(high_nib_scale),
                               reinterpret_cast<const half2&>(high_nib_bias));
        }
      }

      return reinterpret_cast<PackedResultType&>(r);
    };
  };

 public:
  CUTLASS_DEVICE
  static result_type convert(source_type const& source) {
    return ArrayConverterPacked32Bit<RegConvert, typename result_type::Element,
                                     typename source_type::Element,
                                     N>::convert(source);
  }

  CUTLASS_DEVICE
  result_type operator()(source_type const& s) const { return convert(s); }
};

// for Array<cutlass::half_t, N> <= Array<uint4_t, N>
//   for IlvdLayout: (2, 4):(4, 1)
// 中文注释：标准 CUTLASS uint4_t -> FP16 的交错布局转换器。
// 与上面 vllm_uint4b8_t 版本的区别在于偏置值不同：
//   - vllm_uint4b8_t 偏置为 8，真实值范围 [-8, 7]
//   - uint4_t 无偏置，值范围 [0, 15]
// 因此 bias 常量不同（1024 vs 1032）。
template <FloatRoundStyle Round, int N>
struct InterleavedNumericArrayConverter<Layout<Shape<_2, _4>, Stride<_4, _1>>,
                                        cutlass::half_t, uint4_t, N, Round,
                                        void> {
  using IlvdLayout = Layout<Shape<_2, _4>, Stride<_4, _1>>;
  static_assert(N % size(IlvdLayout{}) == 0);

  using result_type = Array<cutlass::half_t, N>;
  using source_type = Array<uint4_t, N>;

  static FloatRoundStyle const round_style = Round;

 private:
  struct RegConvert {
    template <typename PackedResultType>
    CUTLASS_DEVICE static PackedResultType convert(Array<uint32_t, 1> src_) {
      uint32_t src = src_[0];
      using RegArray =
          cutlass::AlignedArray<uint32_t, PackedResultType::kElements / 2,
                                sizeof(PackedResultType)>;
      RegArray r;

      static_assert(PackedResultType::kElements <= size(IlvdLayout{}));
      static constexpr uint32_t xor_mask = 0x64006400;

      for (int ii = 0; ii < RegArray::kElements; ii += 2) {
        auto src_ = src >> (4 * (ii));
        r[ii + 0] = src_;
        r[ii + 1] = src_;

        static constexpr uint32_t and_xor_imm_lut = (0xf0 & 0xcc) ^ 0xaa;

        static constexpr uint32_t low_nib_mask = 0x000F000F;
        static constexpr uint32_t high_nib_mask = 0x00F000F0;

        asm volatile(
            "{\n"
            "  lop3.b32 %0, %0, %1, %2, %3;\n"
            "}\n"
            : "+r"(r[ii + 0])
            : "n"(low_nib_mask), "n"(xor_mask), "n"(and_xor_imm_lut));

        asm volatile(
            "{\n"
            "  lop3.b32 %0, %0, %1, %2, %3;\n"
            "}\n"
            : "+r"(r[ii + 1])
            : "n"(high_nib_mask), "n"(xor_mask), "n"(and_xor_imm_lut));

        // For low nibble:
        //  {x1, x0} = {1024+x1, 1024+x0} - {1024, 1024}
        // For high nibble:
        //  {x1, x0} = {1024+16*x1, 1024+16*x0} * {1/16, 1/16} - {64, 64}
        static constexpr uint32_t low_nib_bias = 0x64006400;    // {1024, 1024}
        static constexpr uint32_t high_nib_scale = 0x2C002C00;  // {1/16, 1/16}
        static constexpr uint32_t high_nib_bias = 0xD400D400;   // {-64, -64}

        {
          half2& fp16x2_val = reinterpret_cast<__half2&>(r[ii + 0]);
          fp16x2_val =
              __hsub2(fp16x2_val, reinterpret_cast<const half2&>(low_nib_bias));
        }

        {
          half2& fp16x2_val = reinterpret_cast<__half2&>(r[ii + 1]);
          fp16x2_val = __hfma2(fp16x2_val,
                               reinterpret_cast<const half2&>(high_nib_scale),
                               reinterpret_cast<const half2&>(high_nib_bias));
        }
      }

      return reinterpret_cast<PackedResultType&>(r);
    };
  };

 public:
  CUTLASS_DEVICE
  static result_type convert(source_type const& source) {
    return ArrayConverterPacked32Bit<RegConvert, typename result_type::Element,
                                     typename source_type::Element,
                                     N>::convert(source);
  }

  CUTLASS_DEVICE
  result_type operator()(source_type const& s) const { return convert(s); }
};

// for Array<cutlass::half_t, N> <= Array<vllm_uint8b128_t, N>
// 中文注释：vllm_uint8b128_t (8-bit GPTQ，偏置 128) -> FP16 的数组转换器。
// 转换算法：
//   1. 使用 prmt 指令将每个 8-bit 值扩展到 FP16 的低字节
//   2. 设置 FP16 的高字节为 0x64（即指数部分，构造起始值 2^13 = 8192 的 FP16）
//   3. 减去偏置 (8388608 + 128) = 2^23 + 128，得到正确的 FP16 值
// 这里利用了 IEEE 754 浮点数的位表示特性来高效完成整数到浮点的转换。
template <FloatRoundStyle Round, int N>
struct NumericArrayConverter<cutlass::half_t, vllm_uint8b128_t, N, Round> {
  using result_type = Array<cutlass::half_t, N>;
  using source_type = Array<vllm_uint8b128_t, N>;

  struct RegConvert {
    template <typename PackedResultType>
    CUTLASS_DEVICE static PackedResultType convert(Array<uint32_t, 1> src_) {
      uint32_t src = src_[0];
      // Hold output FP16s in reg. We need 1 reg for every 2 elements
      using RegArray =
          cutlass::AlignedArray<uint32_t, PackedResultType::kElements / 2,
                                sizeof(PackedResultType)>;
      RegArray r;

      uint32_t const prmt_indices[2] = {0x5150, 0x5352};
      static constexpr uint32_t start_byte_for_fp16 = 0x64646464;

      for (int ii = 0; ii < RegArray::kElements; ++ii) {
        asm volatile("prmt.b32 %0,%1,%2,%3;\n"
                     : "=r"(r[ii])
                     : "r"(src), "n"(start_byte_for_fp16),
                       "r"(prmt_indices[ii]));
      }

      // -128 is folded into bias subtraction, i.e. the 0x80 in the low bytes
      static constexpr uint32_t bias_rep = 0x64806480;
      const half2& bias = reinterpret_cast<const half2&>(bias_rep);
      CUTLASS_PRAGMA_UNROLL
      for (int ii = 0; ii < RegArray::kElements; ++ii) {
        half2& fp16x2_val = reinterpret_cast<__half2&>(r[ii]);
        fp16x2_val = __hsub2(fp16x2_val, bias);
      }

      return reinterpret_cast<PackedResultType&>(r);
    };
  };

 public:
  CUTLASS_DEVICE
  static result_type convert(source_type const& source) {
    return ArrayConverterPacked32Bit<RegConvert, typename result_type::Element,
                                     typename source_type::Element,
                                     N>::convert(source);
  }

  CUTLASS_DEVICE
  result_type operator()(source_type const& s) const { return convert(s); }
};

// for Array<cutlass::float, N> <= Array<vllm_uint8b128_t, N>
// 中文注释：vllm_uint8b128_t (8-bit GPTQ，偏置 128) -> FP32 的数组转换器。
// 转换算法（"magic number"法）：
//   1. 使用 __byte_perm 将每个 8-bit 值放入 FP32 的最低字节，
//      同时将高 3 字节设为 0x4B（构造 FP32 的指数部分，对应 2^23 = 8388608）
//   2. 从结果中减去 8388608.0 + 128.0，去除魔数和偏置
//   3. 最终得到正确的 FP32 值
// 每次循环处理一个 32-bit 寄存器（包含 4 个 8-bit 值 -> 4 个 FP32 值）。
template <FloatRoundStyle Round, int N>
struct NumericArrayConverter<float, vllm_uint8b128_t, N, Round> {
  using result_type = Array<float, N>;
  using source_type = Array<vllm_uint8b128_t, N>;
  static FloatRoundStyle const round_style = Round;

 private:
  struct RegConvert {
    template <typename PackedResultType>
    CUTLASS_DEVICE static PackedResultType convert(Array<uint32_t, 1> src_) {
      uint32_t src = src_[0];
      PackedResultType r;

      // __byte_perm simulates the add.u32 0x4B000000 to every u8 element of
      // u8x4 source and stores the result in r (without introducing extra
      // cvt.u32.u8 instruction)
      uint32_t const prmt_indices[4] = {0x7650, 0x7651, 0x7652, 0x7653};
      uint32_t* result_as_int = reinterpret_cast<uint32_t*>(&r);
      for (int ii = 0; ii < PackedResultType::kElements; ++ii) {
        result_as_int[ii] = __byte_perm(src, 0x4B000000, prmt_indices[ii]);
        // Subtract the magic number 0x4B000000 from tmp in floating-point
        // arithmetic to obtain final result
        r[ii] -= (8388608.f + 128.f);  // fold in -128 bias
      }

      return r;
    };
  };

 public:
  CUTLASS_DEVICE
  static result_type convert(source_type const& source) {
    return ArrayConverterPacked32Bit<RegConvert, typename result_type::Element,
                                     typename source_type::Element,
                                     N>::convert(source);
  }

  CUTLASS_DEVICE
  result_type operator()(source_type const& s) const { return convert(s); }
};

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)

// 中文注释：以下 BF16 转换器仅在 SM80+（Ampere 及以上）架构上可用，
// 因为 bfloat16 的原生硬件支持从 SM80（A100）开始。

// for Array<cutlass::bfloat16_t, N> <= Array<vllm_uint4b8_t, N>
// 中文注释：vllm_uint4b8_t (4-bit GPTQ，偏置 8) -> BF16 的数组转换器。
// 转换算法与 FP16 版本类似，但使用 BF16 的指数范围：
//   1. 使用 prmt 提取每个 4-bit 值
//   2. 使用 lop3 设置 BF16 指数位为 0x4300（对应 128.0），
//      构造 {128 + (x+8)} 的 BF16 值
//   3. 减去偏置 136 = 128 + 8，得到正确的 BF16 值
template <FloatRoundStyle Round, int N>
struct NumericArrayConverter<cutlass::bfloat16_t, vllm_uint4b8_t, N, Round> {
  using result_type = Array<cutlass::bfloat16_t, N>;
  using source_type = Array<vllm_uint4b8_t, N>;

  static FloatRoundStyle const round_style = Round;

 private:
  struct RegConvert {
    template <typename PackedResultType>
    CUTLASS_DEVICE static PackedResultType convert(Array<uint32_t, 1> src_) {
      uint32_t src_reg = src_[0];
      // Hold output BF16s in reg. We need 1 reg for every 2 elements
      using RegArray =
          cutlass::AlignedArray<uint32_t, PackedResultType::kElements / 2,
                                sizeof(PackedResultType)>;
      RegArray r;
      uint32_t src_reg_shifted = src_reg >> 4;

      // Below constructs the following temporary:
      uint32_t const prmt_indices[4] = {0xF4F0, 0xF5F1, 0xF6F2, 0xF7F3};
      static_assert(RegArray::kElements <= 4,
                    "Too many inputs for uint4b8_t -> BF16 vector converter");
      CUTLASS_PRAGMA_UNROLL
      for (int ii = 0; ii < RegArray::kElements; ++ii) {
        asm volatile(
            "{\n"
            "  prmt.b32 %0, %1, %2, %3;\n"
            "}\n"
            : "=r"(r[ii])
            : "r"(src_reg), "r"(src_reg_shifted), "r"(prmt_indices[ii]));
      }

      // Since the stored 4bit values are biased by 8 we get stored_val = (x+8)
      //  we are trying to construct x and a BF16 value
      // The below XOR does the following:
      //  1) Sets the exponent bits of the BF16 to the correct value for the
      //  BF16 magic_num. We will be constructing {128 + (x1+8), 128 + (x0+8)}
      //  and subtracting 136 to get {x1, x0}
      static constexpr uint32_t xor_mask = 0x43004300;
      static constexpr uint32_t and_mask = 0x000F000F;
      static constexpr uint32_t immLut = (0xf0 & 0xcc) ^ 0xaa;

      // For each operand, computes:
      // r[i] = (r[i] & and_mask) ^ xor_mask
      CUTLASS_PRAGMA_UNROLL
      for (int ii = 0; ii < RegArray::kElements; ++ii) {
        asm volatile(
            "{\n"
            "  lop3.b32 %0, %0, %1, %2, %3;\n"
            "}\n"
            : "+r"(r[ii])
            : "n"(and_mask), "n"(xor_mask), "n"(immLut));
      }

      // We will issue 2 bfmas that do the following:
      // high BF16:
      // hi_bf16 - 136, lo_bf16 - 136

      // This is the BF16 {136, 136} represented as an integer.
      static constexpr uint32_t bias_rep = 0x43084308;
      const __nv_bfloat162& bias =
          reinterpret_cast<const __nv_bfloat162&>(bias_rep);

      CUTLASS_PRAGMA_UNROLL
      for (int ii = 0; ii < RegArray::kElements; ++ii) {
        __nv_bfloat162& bf16x2_val = reinterpret_cast<__nv_bfloat162&>(r[ii]);
        bf16x2_val = __hsub2(bf16x2_val, bias);
      }

      return reinterpret_cast<PackedResultType&>(r);
    }
  };

 public:
  CUTLASS_DEVICE
  static result_type convert(source_type const& source) {
    return ArrayConverterPacked32Bit<RegConvert, typename result_type::Element,
                                     typename source_type::Element,
                                     N>::convert(source);
  }

  CUTLASS_DEVICE
  result_type operator()(source_type const& s) const { return convert(s); }
};

// for Array<cutlass::bfloat16_t, N> <= Array<vllm_uint4b8_t, N>
//   for IlvdLayout: (2, 4):(4, 1)
// 中文注释：vllm_uint4b8_t -> BF16 的交错布局转换器。
// 与 FP16 交错版本不同，BF16 的尾数（mantissa）只有 7 bit，无法容纳两个 4-bit nibble。
// 因此每次只能处理一个 nibble，逐个转换再组合。
template <FloatRoundStyle Round, int N>
struct InterleavedNumericArrayConverter<Layout<Shape<_2, _4>, Stride<_4, _1>>,
                                        cutlass::bfloat16_t, vllm_uint4b8_t, N,
                                        Round, void> {
  using IlvdLayout = Layout<Shape<_2, _4>, Stride<_4, _1>>;
  static_assert(N % size(IlvdLayout{}) == 0);

  using result_type = Array<cutlass::bfloat16_t, N>;
  using source_type = Array<vllm_uint4b8_t, N>;

 private:
  struct RegConvert {
    template <typename PackedResultType>
    CUTLASS_DEVICE static PackedResultType convert(Array<uint32_t, 1> src_) {
      uint32_t src = src_[0];
      using RegArray =
          cutlass::AlignedArray<uint32_t, PackedResultType::kElements / 2,
                                sizeof(PackedResultType)>;
      RegArray r;

      static_assert(PackedResultType::kElements <= size(IlvdLayout{}));
      static constexpr uint32_t or_mask = 0x43004300;

      // Unlike float16 where the mantissa is large enough to contain 2
      // nibbles, bfloat16 can only fit one, so we can only convert one
      // nibble at a time
      for (int ii = 0; ii < RegArray::kElements; ++ii) {
        r[ii] = src >> (4 * ii);

        static constexpr uint32_t and_or_imm_lut = (0xf0 & 0xcc) | 0xaa;
        static constexpr uint32_t low_nib_mask = 0x000F000F;

        asm volatile(
            "{\n"
            "  lop3.b32 %0, %0, %1, %2, %3;\n"
            "}\n"
            : "+r"(r[ii + 0])
            : "n"(low_nib_mask), "n"(or_mask), "n"(and_or_imm_lut));

        // For low nibble:
        //  {x1, x0} = {128+(x1+8), 128+(x0+8)} * {1, 1} - {136, 136}
        static constexpr uint32_t low_nib_bias = 0x43084308;  // {136, 136}

        {
          __nv_bfloat162& fp16x2_val = reinterpret_cast<__nv_bfloat162&>(r[ii]);
          fp16x2_val =
              __hsub2(fp16x2_val,
                      reinterpret_cast<const __nv_bfloat162&>(low_nib_bias));
        }
      }

      return reinterpret_cast<PackedResultType&>(r);
    };
  };

 public:
  CUTLASS_DEVICE
  static result_type convert(source_type const& source) {
    return ArrayConverterPacked32Bit<RegConvert, typename result_type::Element,
                                     typename source_type::Element,
                                     N>::convert(source);
  }

  CUTLASS_DEVICE
  result_type operator()(source_type const& s) const { return convert(s); }
};

// for Array<cutlass::bfloat16_t, N> <= Array<uint4_t, N>
//   for IlvdLayout: (2, 4):(4, 1)
// 中文注释：标准 CUTLASS uint4_t -> BF16 的交错布局转换器。
// 与 vllm_uint4b8_t 版本类似，但无偏置（值范围 [0, 15]）。
template <FloatRoundStyle Round, int N>
struct InterleavedNumericArrayConverter<Layout<Shape<_2, _4>, Stride<_4, _1>>,
                                        cutlass::bfloat16_t, uint4_t, N, Round,
                                        void> {
  using IlvdLayout = Layout<Shape<_2, _4>, Stride<_4, _1>>;
  static_assert(N % size(IlvdLayout{}) == 0);

  using result_type = Array<cutlass::bfloat16_t, N>;
  using source_type = Array<uint4_t, N>;

 private:
  struct RegConvert {
    template <typename PackedResultType>
    CUTLASS_DEVICE static PackedResultType convert(Array<uint32_t, 1> src_) {
      uint32_t src = src_[0];
      using RegArray =
          cutlass::AlignedArray<uint32_t, PackedResultType::kElements / 2,
                                sizeof(PackedResultType)>;
      RegArray r;

      static_assert(PackedResultType::kElements <= size(IlvdLayout{}));
      static constexpr uint32_t or_mask = 0x43004300;

      // Unlike float16 where the mantissa is large enough to contain 2
      // nibbles, bfloat16 can only fit one, so we can only convert one
      // nibble at a time
      for (int ii = 0; ii < RegArray::kElements; ++ii) {
        r[ii] = src >> (4 * ii);

        static constexpr uint32_t and_or_imm_lut = (0xf0 & 0xcc) | 0xaa;
        static constexpr uint32_t low_nib_mask = 0x000F000F;

        asm volatile(
            "{\n"
            "  lop3.b32 %0, %0, %1, %2, %3;\n"
            "}\n"
            : "+r"(r[ii])
            : "n"(low_nib_mask), "n"(or_mask), "n"(and_or_imm_lut));

        // For low nibble:
        //  {x1, x0} = {128 + x1, 128 + x0} * {1, 1} - {128, 128}
        static constexpr uint32_t low_nib_bias = 0x43004300;  // {128, 128}

        {
          __nv_bfloat162& fp16x2_val = reinterpret_cast<__nv_bfloat162&>(r[ii]);
          fp16x2_val =
              __hsub2(fp16x2_val,
                      reinterpret_cast<const __nv_bfloat162&>(low_nib_bias));
        }
      }

      return reinterpret_cast<PackedResultType&>(r);
    };
  };

 public:
  CUTLASS_DEVICE
  static result_type convert(source_type const& source) {
    return ArrayConverterPacked32Bit<RegConvert, typename result_type::Element,
                                     typename source_type::Element,
                                     N>::convert(source);
  }

  CUTLASS_DEVICE
  result_type operator()(source_type const& s) const { return convert(s); }
};

// for Array<cutlass::bfloat16_t, N> <= Array<vllm_uint8b128_t, N>
// 中文注释：vllm_uint8b128_t (8-bit GPTQ，偏置 128) -> BF16 的数组转换器。
// 由于 BF16 没有直接的 8-bit -> BF16 的高效指令路径，
// 此实现采用两步法：先将 uint8 转为 FP32，再将 FP32 截断为 BF16。
// 虽然不是最优路径，但保证了正确性，且中间 FP32 步骤本身也是向量化的。
template <FloatRoundStyle Round, int N>
struct NumericArrayConverter<cutlass::bfloat16_t, vllm_uint8b128_t, N, Round> {
  using result_type = Array<cutlass::bfloat16_t, N>;
  using source_type = Array<vllm_uint8b128_t, N>;
  static FloatRoundStyle const round_style = Round;

 private:
  using result_packed_4_t = Array<cutlass::bfloat16_t, 4>;
  using result_packed_2_t = Array<cutlass::bfloat16_t, 2>;
  using src_packed_4_t = Array<vllm_uint8b128_t, 4>;
  using src_packed_2_t = Array<vllm_uint8b128_t, 2>;

  // Not Valid, not supported, only here to satisfy the interface and to avoid
  //  a compile error. ScalarConverter will not actually work until
  //  NumericConverter<cutlass::bfloat16_t, vllm_uint8b128_t, Round> is
  //  implemented
  using ScalarConverter =
      NumericConverter<cutlass::bfloat16_t, vllm_uint8b128_t, Round>;

  template <typename PackedResultType, typename PackedSrcType>
  CUTLASS_DEVICE static PackedResultType packed_convert(
      PackedSrcType const& source) {
    static_assert(
        (platform::is_same<PackedSrcType, src_packed_2_t>::value &&
         platform::is_same<PackedResultType, result_packed_2_t>::value) ||
            (platform::is_same<PackedSrcType, src_packed_4_t>::value &&
             platform::is_same<PackedResultType, result_packed_4_t>::value),
        "Invalid PackedSrcType/PackedResultType must be 2 or 4 to use private "
        "convert dispatch.");

    NumericArrayConverter<float, vllm_uint8b128_t, PackedResultType::kElements,
                          Round>
        convert_uint8_to_f32;
    Array<float, PackedResultType::kElements> tmp =
        convert_uint8_to_f32(source);
    NumericArrayConverter<cutlass::bfloat16_t, float,
                          PackedResultType::kElements, Round>
        convert_f32_to_bf16_;
    return convert_f32_to_bf16_(tmp);
  }

  friend class detail::VectorizedConverter;

 public:
  CUTLASS_DEVICE
  static result_type convert(source_type const& source) {
    result_type result;
    using ConverterType =
        NumericArrayConverter<typename result_type::Element,
                              typename source_type::Element, N, Round>;
    detail::VectorizedConverter::convert<ConverterType, result_packed_4_t,
                                         src_packed_4_t, result_packed_2_t,
                                         src_packed_2_t>(result, source);

    return result;
  }

  CUTLASS_DEVICE
  result_type operator()(source_type const& s) const { return convert(s); }
};

#endif

// for Array<int8_t, N> <= Array<cutlass::half_t, N>
//   FastFP16toINT8 from https://arxiv.org/pdf/2406.09904
// 中文注释：FP16 -> int8 的快速向量化转换器。
// 算法来源：https://arxiv.org/pdf/2406.09904（FastFP16toINT8）
//
// 算法流程：
//   1. 给每个 FP16 值加上魔数偏置 0x6480（= 25728.0），
//      使得 FP16 的尾数位恰好包含原始整数值
//   2. 使用 prmt 指令从 4 个 FP16 值（分布在两个 32-bit 寄存器中）
//      提取低字节，组成 4 个 uint8 值
//   3. XOR 0x80 将 uint8 转为 int8（uint8 和 int8 的区别仅在最高位解释）
//
// 该方法将 4 个 FP16 -> int8 的转换压缩到仅 3 条指令（add + prmt + xor），
// 比逐元素的 cvt 快约 4 倍。
template <FloatRoundStyle Round, int N>
struct NumericArrayConverter<int8_t, cutlass::half_t, N, Round> {
  using result_type = Array<int8_t, N>;
  using source_type = Array<cutlass::half_t, N>;

  struct RegConvert {
    // FastFP16toINT8 from https://arxiv.org/pdf/2406.09904
    template <typename PackedResultType, int src_regs>
    CUTLASS_DEVICE static PackedResultType convert(
        Array<uint32_t, src_regs> src) {
      // Hold output int8s in reg. We need 1 reg for every 4 elements
      using RegArray = cutlass::AlignedArray<
          uint32_t, std::max(PackedResultType::kElements / 4, size_t(1))>;
      RegArray r;

      static constexpr uint32_t MAGIC_BIAS_ = 0x64806480;
      auto MAGIC_BIAS = *reinterpret_cast<const half2*>(&MAGIC_BIAS_);

      *reinterpret_cast<half2*>(&src[0]) =
          __hadd2(*reinterpret_cast<half2*>(&src[0]), MAGIC_BIAS);

      if constexpr (src_regs > 1) {
        *reinterpret_cast<half2*>(&src[1]) =
            __hadd2(*reinterpret_cast<half2*>(&src[1]), MAGIC_BIAS);
      }

      static_assert(PackedResultType::kElements <= 4);
      uint32_t uint8s;
      static constexpr uint32_t MASK_0246 = 0x6420;
      static constexpr uint32_t UINT8s_TO_INT8s_MASK = 0x80808080;
      asm volatile("prmt.b32 %0,%1,%2,%3;\n"
                   : "=r"(uint8s)
                   : "r"(src[0]), "r"((src_regs > 1) ? src[1] : src[0]),
                     "n"(MASK_0246));

      uint32_t int8s = (uint8s ^ UINT8s_TO_INT8s_MASK);

      return reinterpret_cast<PackedResultType&>(int8s);
    };
  };

 public:
  CUTLASS_DEVICE
  static result_type convert(source_type const& source) {
    return ArrayConverterPacked32Bit<RegConvert, typename result_type::Element,
                                     typename source_type::Element,
                                     N>::convert(source);
  }

  CUTLASS_DEVICE
  result_type operator()(source_type const& s) const { return convert(s); }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
