#pragma once

#include <cstdint>
#include <string>
#include <tuple>
#include <utility>
#include <variant>

// For STD_TORCH_CHECK
#include <torch/headeronly/util/Exception.h>

namespace vllm {

// 中文注释：模块功能概述
// =========================================================================
// ScalarType 是 vLLM 的核心标量类型系统，用于表示各种浮点和整数数据类型，
// 特别是能够表示子字节（sub-byte）数据类型（如 4-bit、2-bit 等），
// 这是 PyTorch 的 torch.dtype 目前不支持的功能。
//
// 主要用途：
// 1. 量化（Quantization）：支持各种量化方案（INT4、INT8、FP8、FP4 等）
// 2. 类型安全：在编译期进行类型检查和转换
// 3. 序列化：通过 id() 方法生成唯一标识符，用于模板特化
//
// 设计特点：
// - 使用位域紧凑存储类型信息（exponent、mantissa、signed 等）
// - 支持编译期计算（constexpr）
// - 提供 IEEE 754 标准和非标准浮点类型的表示
// - 支持 NaN、Infinity 等特殊值的多种表示方式
//
// Python 端对应的类型定义在: vllm/scalar_type.py
// 两边的定义需要保持同步。
// =========================================================================
class ScalarType {
 public:
  // 中文注释：NaN（Not a Number）表示方式枚举
  // 不同的浮点格式对 NaN 有不同的编码方式：
  // - NAN_NONE: 不支持 NaN（某些量化类型）
  // - NAN_IEEE_754: 标准 IEEE 754 方式（指数全1，尾数非全0）
  // - NAN_EXTD_RANGE_MAX_MIN: 扩展范围方式（指数全1，尾数全1）
  //   这种方式牺牲 NaN 表示，换取更大的数值范围
  enum NanRepr : uint8_t {
    NAN_NONE = 0,                // nans are not supported
    NAN_IEEE_754 = 1,            // nans are: exp all 1s, mantissa not all 0s
    NAN_EXTD_RANGE_MAX_MIN = 2,  // nans are: exp all 1s, mantissa all 1s

    NAN_REPR_ID_MAX
  };

  // 中文注释：构造函数
  // 参数说明：
  // - exponent: 指数位数（对于整数类型为0）
  // - mantissa: 尾数位数（对于整数类型，表示除符号位外的位数）
  // - signed_: 是否有符号（即是否有符号位）
  // - bias: 偏置值（存储值 = 实际值 + bias），用于量化类型
  // - finite_values_only: 是否只包含有限值（不含 +/-inf）
  // - nan_repr: NaN 表示方式
  constexpr ScalarType(uint8_t exponent, uint8_t mantissa, bool signed_,
                       int32_t bias, bool finite_values_only = false,
                       NanRepr nan_repr = NAN_IEEE_754)
      : exponent(exponent),
        mantissa(mantissa),
        signed_(signed_),
        bias(bias),
        finite_values_only(finite_values_only),
        nan_repr(nan_repr) {};

  // 中文注释：创建有符号整数类型的工厂方法
  // 例如：int_(8) 创建 INT8 类型，int_(4) 创建 INT4 类型
  // size_bits: 总位数（包含符号位）
  // bias: 偏置值，用于量化（如 uint(8, 128) 表示零点偏移为128的UINT8）
  static constexpr ScalarType int_(uint8_t size_bits, int32_t bias = 0) {
    return ScalarType(0, size_bits - 1, true, bias);
  }

  // 中文注释：创建无符号整数类型的工厂方法
  // 例如：uint(8) 创建 UINT8 类型
  // 无符号类型没有符号位，所有位都用于表示数值
  static constexpr ScalarType uint(uint8_t size_bits, int32_t bias = 0) {
    return ScalarType(0, size_bits, false, bias);
  }

  // 中文注释：创建符合 IEEE 754 标准的浮点类型工厂方法
  // 例如：float_IEEE754(5, 2) 创建 FP16（E5M2）格式
  //       float_IEEE754(8, 7) 创建 BF16（E8M7）格式
  // 标准 IEEE 754 类型支持 NaN 和 Infinity
  static constexpr ScalarType float_IEEE754(uint8_t exponent,
                                            uint8_t mantissa) {
    STD_TORCH_CHECK(mantissa > 0 && exponent > 0);
    return ScalarType(exponent, mantissa, true, 0, false, NAN_IEEE_754);
  }

  // 中文注释：创建非 IEEE 754 标准的浮点类型工厂方法
  // 用于创建特殊的量化浮点类型（如 FP8 E4M3FN）
  // 这些类型通常牺牲 NaN/Infinity 表示来换取更大的数值范围
  // finite_values_only: 是否只包含有限值
  // nan_repr: NaN 的表示方式
  static constexpr ScalarType float_(uint8_t exponent, uint8_t mantissa,
                                     bool finite_values_only,
                                     NanRepr nan_repr) {
    STD_TORCH_CHECK(nan_repr < NAN_REPR_ID_MAX, "Invalid NanRepr");
    STD_TORCH_CHECK(mantissa > 0 && exponent > 0);
    STD_TORCH_CHECK(
        nan_repr != NAN_IEEE_754,
        "use `float_IEEE754` constructor for floating point types that "
        "follow IEEE 754 conventions");
    return ScalarType(exponent, mantissa, true, 0, finite_values_only,
                      nan_repr);
  }

  // 中文注释：核心成员变量
  // =========================================================================
  // 这些成员变量定义了标量类型的完整特征：
  //
  // 浮点类型布局：[sign] [exponent] [mantissa]
  //   例如 FP16 (E5M2)：1位符号 + 5位指数 + 2位尾数 = 8位
  //
  // 整数类型布局：[sign] [value bits]
  //   例如 INT8：1位符号 + 7位数值 = 8位
  //   例如 UINT8：0位符号 + 8位数值 = 8位
  // =========================================================================

  // 指数位数（对于整数类型为0）
  uint8_t const exponent;  // size of the exponent field (0 for integer types)

  // 尾数位数（对于整数类型，表示除符号位外的数值位数）
  uint8_t const mantissa;  // size of the mantissa field (size of the integer
                           // excluding the sign bit for integer types)

  // 是否有符号位
  bool const signed_;  // flag if the type supports negative numbers (i.e. has a
                       // sign bit)

  // 偏置值：存储值 = 实际值 + bias
  // 用于量化类型，例如 UINT8 的零点偏移（zero point）
  int32_t const bias;  // stored values equal value + bias,
                       // used for quantized type

  // 中文注释：以下成员变量仅用于浮点类型
  // =========================================================================

  // 是否只包含有限值（不含 +/-inf），某些量化浮点类型使用此特性换取更大范围
  bool const finite_values_only;  // i.e. no +/-inf if true

  // NaN 的表示方式（不适用于整数类型）
  NanRepr const nan_repr;         // how NaNs are represented
                                  // (not applicable for integer types)

  // 中文注释：类型标识符类型
  // Id 用于在编译期生成类型的唯一标识符
  // 这是 C++17 模板特化的 workaround（C++20 可以直接传递字面量类作为模板参数）
  using Id = int64_t;

 private:
  // 中文注释：辅助函数 - 获取成员变量在 ID 中占用的位数
  // bool 类型占用 1 位，其他类型占用 sizeof(T) * 8 位
  template <typename T_>
  static constexpr size_t member_id_field_width() {
    using T = std::decay_t<T_>;
    return std::is_same_v<T, bool> ? 1 : sizeof(T) * 8;
  }

  // 中文注释：辅助函数 - 递归地对所有成员变量应用函数 f（折叠操作）
  // 这是一个编译期的 reduce/fold 操作，用于遍历所有成员变量
  template <typename Fn, typename Init, typename Member, typename... Rest>
  static constexpr auto reduce_members_helper(Fn f, Init val, Member member,
                                              Rest... rest) {
    auto new_val = f(val, member);
    if constexpr (sizeof...(rest) > 0) {
      return reduce_members_helper(f, new_val, rest...);
    } else {
      return new_val;
    };
  }

  // 中文注释：对实例的成员变量应用折叠操作
  // 成员变量顺序必须与构造函数参数顺序一致（用于 from_id 反序列化）
  template <typename Fn, typename Init>
  constexpr auto reduce_members(Fn f, Init init) const {
    // Should be in constructor order for `from_id`
    return reduce_members_helper(f, init, exponent, mantissa, signed_, bias,
                                 finite_values_only, nan_repr);
  };

  // 中文注释：对成员变量类型应用折叠操作（静态版本，不需要实例）
  template <typename Fn, typename Init>
  static constexpr auto reduce_member_types(Fn f, Init init) {
    constexpr auto dummy_type = ScalarType(0, 0, false, 0, false, NAN_NONE);
    return dummy_type.reduce_members(f, init);
  };

  // 中文注释：计算 ID 所需的总位数
  static constexpr auto id_size_bits() {
    return reduce_member_types(
        [](int acc, auto member) -> int {
          return acc + member_id_field_width<decltype(member)>();
        },
        0);
  }

 public:
  // 中文注释：生成类型的唯一标识符
  // 将所有成员变量打包成一个 int64_t，用于 C++17 模板特化
  // 位布局：[exponent][mantissa][signed_][bias][finite_values_only][nan_repr]
  // 示例：FP16 (E5M2) 的 ID 包含 exponent=5, mantissa=2, signed=true 等信息
  //
  // 使用场景：
  // - 模板特化：在编译期确定类型，生成高效的特化代码
  // - 序列化：将类型信息传递给 Python 端
  constexpr Id id() const {
    static_assert(id_size_bits() <= sizeof(Id) * 8,
                  "ScalarType id is too large to be stored");

    auto or_and_advance = [](std::pair<Id, uint32_t> result,
                             auto member) -> std::pair<Id, uint32_t> {
      auto [id, bit_offset] = result;
      auto constexpr bits = member_id_field_width<decltype(member)>();
      return {id | (int64_t(member) & ((uint64_t(1) << bits) - 1))
                       << bit_offset,
              bit_offset + bits};
    };
    return reduce_members(or_and_advance, std::pair<Id, uint32_t>{}).first;
  }

  // 中文注释：从 ID 反序列化创建 ScalarType
  // 与 id() 方法相反，从打包的 int64_t 中提取所有成员变量
  // 这是 id() 的逆操作，用于从模板参数恢复类型信息
  static constexpr ScalarType from_id(Id id) {
    auto extract_and_advance = [id](auto result, auto member) {
      using T = decltype(member);
      auto [tuple, bit_offset] = result;
      auto constexpr bits = member_id_field_width<T>();
      auto extracted_val = static_cast<T>((int64_t(id) >> bit_offset) &
                                          ((uint64_t(1) << bits) - 1));
      auto new_tuple = std::tuple_cat(tuple, std::make_tuple(extracted_val));
      return std::pair<decltype(new_tuple), int>{new_tuple, bit_offset + bits};
    };

    auto [tuple_args, _] = reduce_member_types(extract_and_advance,
                                               std::pair<std::tuple<>, int>{});
    return std::apply([](auto... args) { return ScalarType(args...); },
                      tuple_args);
  }

  // 中文注释：查询类型的总位数
  // 计算方式：尾数位 + 指数位 + 符号位（如果有）
  // 例如：FP16 (E5M2) = 2 + 5 + 1 = 8 位
  //       INT8 = 7 + 0 + 1 = 8 位
  constexpr int64_t size_bits() const {
    return mantissa + exponent + is_signed();
  }

  // 中文注释：类型查询方法
  // 这些方法提供了类型的特征信息，所有方法都是编译期可计算的（constexpr）
  constexpr bool is_signed() const { return signed_; }

  // 是否为整数类型（指数位数为0）
  constexpr bool is_integer() const { return exponent == 0; }

  // 是否为浮点类型（指数位数大于0）
  constexpr bool is_floating_point() const { return exponent > 0; }

  // 是否为标准 IEEE 754 浮点类型
  // 条件：是浮点 + 支持无限值 + NaN 使用 IEEE 754 编码
  constexpr bool is_ieee_754() const {
    return is_floating_point() && finite_values_only == false &&
           nan_repr == NAN_IEEE_754;
  }

  // 是否支持 NaN
  constexpr bool has_nans() const {
    return is_floating_point() && nan_repr != NAN_NONE;
  }

  // 是否支持 Infinity
  constexpr bool has_infs() const {
    return is_floating_point() && finite_values_only == false;
  }

  // 是否有偏置（用于量化类型）
  constexpr bool has_bias() const { return bias != 0; }

 private:
  // 中文注释：计算浮点类型的最大值
  // 通过位操作将类型的特征转换为 double 表示的最大值
  //
  // 计算步骤：
  // 1. 计算最大尾数值（如果使用 EXTENDED_RANGE，需要减1）
  // 2. 计算最大指数值（根据 NaN 表示方式调整）
  // 3. 计算指数偏置（标准为 2^(e-1) - 1）
  // 4. 将尾数和指数组合成 double 的位模式
  //
  // 注意：这里假设使用标准指数偏置，非标准偏置（如 float8_e4m3b11fnuz）
  // 暂不支持，待有需求时再扩展
  double _floating_point_max() const {
    STD_TORCH_CHECK(mantissa <= 52 && exponent <= 11,
                    "Cannot represent max/min as a double for type ", str());

    uint64_t max_mantissa = (uint64_t(1) << mantissa) - 1;
    if (nan_repr == NAN_EXTD_RANGE_MAX_MIN) {
      max_mantissa -= 1;
    }

    uint64_t max_exponent = (uint64_t(1) << exponent) - 2;
    if (nan_repr == NAN_EXTD_RANGE_MAX_MIN || nan_repr == NAN_NONE) {
      STD_TORCH_CHECK(exponent < 11,
                      "Cannot represent max/min as a double for type ", str());
      max_exponent += 1;
    }

    // adjust the exponent to match that of a double
    //  for now we assume the exponent bias is the standard 2^(e-1) -1, (where e
    //  is the exponent bits), there is some precedent for non-standard biases,
    //  example `float8_e4m3b11fnuz` here: https://github.com/jax-ml/ml_dtypes
    //  but to avoid premature over complication we are just assuming the
    //  standard exponent bias until there is a need to support non-standard
    //  biases
    uint64_t exponent_bias = (uint64_t(1) << (exponent - 1)) - 1;
    uint64_t exponent_bias_double = (uint64_t(1) << 10) - 1;  // double e = 11

    uint64_t max_exponent_double =
        max_exponent - exponent_bias + exponent_bias_double;

    // shift the mantissa into the position for a double and
    // the exponent
    uint64_t double_raw =
        (max_mantissa << (52 - mantissa)) | (max_exponent_double << 52);

    return *reinterpret_cast<double*>(&double_raw);
  }

  // 中文注释：获取原始最大值（未考虑 bias）
  // 浮点类型返回 double，整数类型返回 int64_t
  constexpr std::variant<int64_t, double> _raw_max() const {
    if (is_floating_point()) {
      return {_floating_point_max()};
    } else {
      STD_TORCH_CHECK(size_bits() < 64 || size_bits() == 64 && is_signed(),
                      "Cannot represent max as a int64_t");
      return {(int64_t(1) << mantissa) - 1};
    }
  }

  // 中文注释：获取原始最小值（未考虑 bias）
  // 浮点类型通过设置符号位得到负最大值
  // 整数类型使用算术右移构造最小值
  constexpr std::variant<int64_t, double> _raw_min() const {
    if (is_floating_point()) {
      STD_TORCH_CHECK(
          is_signed(),
          "We currently assume all floating point types are signed");
      constexpr uint64_t sign_bit_double = (uint64_t(1) << 63);

      double max = _floating_point_max();
      uint64_t max_raw = *reinterpret_cast<uint64_t*>(&max);
      uint64_t min_raw = max_raw | sign_bit_double;
      return {*reinterpret_cast<double*>(&min_raw)};
    } else {
      STD_TORCH_CHECK(!is_signed() || size_bits() <= 64,
                      "Cannot represent min as a int64_t");
      if (is_signed()) {
        // set the top bit to 1 (i.e. INT64_MIN) and the rest to 0
        // then perform an arithmetic shift right to set all the bits above
        // (size_bits() - 1) to 1
        return {INT64_MIN >> (64 - size_bits())};
      } else {
        return {int64_t(0)};
      }
    }
  }

 public:
  // 中文注释：获取类型可表示的最大值（考虑 bias）
  // 公式：max = raw_max - bias
  // 用于量化/反量化时确定数值范围
  constexpr std::variant<int64_t, double> max() const {
    return std::visit(
        [this](auto x) -> std::variant<int64_t, double> { return {x - bias}; },
        _raw_max());
  }

  // 中文注释：获取类型可表示的最小值（考虑 bias）
  // 公式：min = raw_min - bias
  constexpr std::variant<int64_t, double> min() const {
    return std::visit(
        [this](auto x) -> std::variant<int64_t, double> { return {x - bias}; },
        _raw_min());
  }

  // 中文注释：生成类型的字符串表示
  // 命名规范参考 https://github.com/jax-ml/ml_dtypes
  //
  // 浮点类型格式：float<size>_e<exp>m<mant>[flags]
  //   flags: f=finite only, n=non-standard NaN
  //   示例：float8_e4m3fn, float8_e5m2, bfloat16 (float16_e8m7)
  //
  // 整数类型格式：[u]int<size>[b<bias>]
  //   示例：int8, uint8, uint4b8
  std::string str() const {
    /* naming generally follows: https://github.com/jax-ml/ml_dtypes
     * for floating point types (leading f) the scheme is:
     *  `float<size_bits>_e<exponent_bits>m<mantissa_bits>[flags]`
     *  flags:
     *  - no-flags: means it follows IEEE 754 conventions
     *  - f: means finite values only (no infinities)
     *  - n: means nans are supported (non-standard encoding)
     * for integer types the scheme is:
     *  `[u]int<size_bits>[b<bias>]`
     *  - if bias is not present it means its zero
     */
    if (is_floating_point()) {
      auto ret = "float" + std::to_string(size_bits()) + "_e" +
                 std::to_string(exponent) + "m" + std::to_string(mantissa);
      if (!is_ieee_754()) {
        if (finite_values_only) {
          ret += "f";
        }
        if (nan_repr != NAN_NONE) {
          ret += "n";
        }
      }
      return ret;
    } else {
      auto ret = ((is_signed()) ? "int" : "uint") + std::to_string(size_bits());
      if (has_bias()) {
        ret += "b" + std::to_string(bias);
      }
      return ret;
    }
  }

  // 中文注释：相等比较运算符
  // 所有成员变量都相等时，两个 ScalarType 相等
  constexpr bool operator==(ScalarType const& other) const {
    return mantissa == other.mantissa && exponent == other.exponent &&
           bias == other.bias && signed_ == other.signed_ &&
           finite_values_only == other.finite_values_only &&
           nan_repr == other.nan_repr;
  }
};

using ScalarTypeId = ScalarType::Id;

// 中文注释：预定义的标量类型常量
// =========================================================================
// 这些常量定义了 vLLM 支持的所有常见数据类型
// 命名风格参考：https://github.com/pytorch/pytorch/blob/6d9f74f0af54751311f0dd71f7e5c01a93260ab3/torch/csrc/api/include/torch/types.h
//
// 分类：
// 1. 整数类型（INT4/8, UINT4/8）
// 2. 量化浮点类型（FP4, FP6, FP8 等）
// 3. 标准浮点类型（FP16, BF16）
// =========================================================================

// 中文注释：Rust 风格命名（k前缀）
// 这些是 vLLM 内部使用的类型别名
static inline constexpr auto kS4 = ScalarType::int_(4);
static inline constexpr auto kU4 = ScalarType::uint(4);
static inline constexpr auto kU4B8 = ScalarType::uint(4, 8);
static inline constexpr auto kS8 = ScalarType::int_(8);
static inline constexpr auto kU8 = ScalarType::uint(8);
static inline constexpr auto kU8B128 = ScalarType::uint(8, 128);

// 中文注释：量化浮点类型
// 这些类型常用于模型量化（如 GPTQ、AWQ、FP8 量化等）
static inline constexpr auto kFE2M1f =
    ScalarType::float_(2, 1, true, ScalarType::NAN_NONE);
static inline constexpr auto kFE3M2f =
    ScalarType::float_(3, 2, true, ScalarType::NAN_NONE);
static inline constexpr auto kFE4M3fn =
    ScalarType::float_(4, 3, true, ScalarType::NAN_EXTD_RANGE_MAX_MIN);
static inline constexpr auto kFE8M0fnu =
    ScalarType(8, 0, false, 0, true, ScalarType::NAN_EXTD_RANGE_MAX_MIN);
static inline constexpr auto kFE5M2 = ScalarType::float_IEEE754(5, 2);
static inline constexpr auto kFE8M7 = ScalarType::float_IEEE754(8, 7);
static inline constexpr auto kFE5M10 = ScalarType::float_IEEE754(5, 10);

// 中文注释：固定宽度风格命名
// 提供更直观的类型名称，与 PyTorch 的命名风格保持一致
static inline constexpr auto kInt4 = kS4;
static inline constexpr auto kUint4 = kU4;
static inline constexpr auto kUint4b8 = kU4B8;
static inline constexpr auto kInt8 = kS8;
static inline constexpr auto kUint8 = kU8;
static inline constexpr auto kUint8b128 = kU8B128;

static inline constexpr auto kFloat4_e2m1f = kFE2M1f;
static inline constexpr auto kFloat6_e3m2f = kFE3M2f;
static inline constexpr auto kFloat8_e4m3fn = kFE4M3fn;
static inline constexpr auto kFloat8_e5m2 = kFE5M2;
static inline constexpr auto kFloat16_e8m7 = kFE8M7;
static inline constexpr auto kFloat16_e5m10 = kFE5M10;

// 中文注释：常用类型别名
// kHalf/kFloat16: 标准 FP16（E5M10）
// kBFloat16: Brain Floating Point（E8M7），训练中常用
static inline constexpr auto kHalf = kFE5M10;
static inline constexpr auto kFloat16 = kHalf;
static inline constexpr auto kBFloat16 = kFE8M7;

// 中文注释：预计算的类型 ID，用于模板特化
static inline constexpr auto kFloat16Id = kFloat16.id();
};  // namespace vllm
