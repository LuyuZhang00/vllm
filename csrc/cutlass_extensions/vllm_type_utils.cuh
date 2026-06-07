// 中文注释：本文件提供编译期类型名称查询工具。
// 通过 nameof<T>::value 和 nameof_v<T> 可以在编译期获取类型的字符串名称，
// 主要用于错误信息打印和调试日志中，使类型信息可读。
// 支持 CUTLASS 标准类型（half, bfloat16, float, int8 等）以及
// vLLM 自定义的量化类型（vllm_uint4b8_t, vllm_uint8b128_t）。

#include "cutlass/bfloat16.h"
#include "cutlass/half.h"
#include "cuda_bf16.h"

#include "cutlass_extensions/vllm_custom_types.cuh"

namespace cutlass {

// 中文注释：nameof 模板结构体，通过模板特化为每个类型提供编译期字符串名称。
// 默认值为 "unknown"，特定类型通过 NAMEOF_TYPE 宏进行特化。
template <typename T>
struct nameof {
  static constexpr char const* value = "unknown";
};

// 中文注释：nameof_v 是 nameof<T>::value 的便捷别名，简化使用方式。
template <typename T>
inline constexpr auto nameof_v = nameof<T>::value;

// 中文注释：NAMEOF_TYPE 宏用于快速为指定类型 T 特化 nameof 模板，
// 使其 value 返回类型名的字符串字面量。
#define NAMEOF_TYPE(T)                       \
  template <>                                \
  struct nameof<T> {                         \
    static constexpr char const* value = #T; \
  };

NAMEOF_TYPE(float_e4m3_t)
NAMEOF_TYPE(float_e5m2_t)
NAMEOF_TYPE(half_t)
NAMEOF_TYPE(nv_bfloat16)
NAMEOF_TYPE(bfloat16_t)
NAMEOF_TYPE(float)

NAMEOF_TYPE(int4b_t)
NAMEOF_TYPE(int8_t)
NAMEOF_TYPE(int32_t)
NAMEOF_TYPE(int64_t)

NAMEOF_TYPE(vllm_uint4b8_t)
NAMEOF_TYPE(uint4b_t)
NAMEOF_TYPE(uint8_t)
NAMEOF_TYPE(vllm_uint8b128_t)
NAMEOF_TYPE(uint32_t)
NAMEOF_TYPE(uint64_t)

};  // namespace cutlass