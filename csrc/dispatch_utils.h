/*
 * Adapted from
 * https://github.com/pytorch/pytorch/blob/v2.0.1/aten/src/ATen/Dispatch.h
 */

/*
 * 中文注释：模块功能概述
 * =========================================================================
 * 本文件是 vLLM C++/CUDA 扩展中的类型分发（dispatch）工具头文件。
 *
 * 核心作用：
 *   提供一组宏，用于在 C++ 模板实例化时根据运行时 tensor 的数据类型
 *   （如 float、half、bfloat16、fp8 等）自动分发到对应的模板特化代码路径。
 *
 * 背景知识：
 *   PyTorch 的 AT_DISPATCH_SWITCH / AT_DISPATCH_CASE 宏可以将运行时的
 *   at::ScalarType 映射到编译期的类型参数（通常命名为 scalar_t），
 *   从而让开发者只需编写一次模板代码，运行时会自动根据 tensor 类型
 *   调用对应的特化版本。
 *
 * 本文件在此基础上进行了以下定制：
 *   1. 定义 vLLM 特有的类型分发组合（如仅半精度、仅 FP8、量化类型等）
 *   2. 为 FP8 类型定义特殊的类型别名宏（fp8_t 而非 scalar_t），
 *      以便在嵌套分发场景中区分不同层级的类型参数
 *   3. 提供 ROCm 平台的 FP8 兼容性处理（同时支持 fn 和 fnuz 两种变体）
 *   4. 提供向量大小、布尔值、分组大小等非类型参数的编译期分发宏
 *
 * 设计目的：
 *   在 kernel 启动时将运行时的类型/参数选择转化为编译期常量，
 *   从而让编译器能够进行常量折叠、循环展开等优化，提升 kernel 性能。
 * =========================================================================
 */

#pragma once

#include <torch/all.h>

// Need a special dispatch case macro since we will nest the FP8 dispatch.
// Instead of the usual 'scalar_t', this names the dispatched type 'fp8_t'.

/*
 * 中文注释：AT_DISPATCH_FP8_CASE —— FP8 类型的专用分发 case 宏
 *
 * 背景：
 *   PyTorch 标准的 AT_DISPATCH_CASE 会将匹配到的类型绑定到模板参数名 scalar_t。
 *   但在 vLLM 的某些 kernel 中，可能需要同时分发两种不同的类型（嵌套分发），
 *   例如外层是浮点类型 scalar_t，内层是 FP8 类型。
 *   如果两层都用 scalar_t 作为参数名，就会产生命名冲突。
 *
 * 解决方案：
 *   此宏使用 AT_PRIVATE_CASE_TYPE_USING_HINT 将 FP8 类型绑定到自定义名称 fp8_t，
 *   而非默认的 scalar_t，从而支持嵌套分发场景。
 *
 * 使用场景：
 *   当 kernel 需要同时处理 FP8 量化权重和非 FP8 的激活值时，
 *   可以用外层 VLLM_DISPATCH_FLOATING_TYPES 分发激活值类型为 scalar_t，
 *   内层 VLLM_DISPATCH_FP8_TYPES 分发权重类型为 fp8_t。
 */
#define AT_DISPATCH_FP8_CASE(enum_type, ...) \
  AT_PRIVATE_CASE_TYPE_USING_HINT(enum_type, fp8_t, __VA_ARGS__)

/*
 * 中文注释：VLLM_DISPATCH_CASE_FLOATING_TYPES —— 浮点类型 case 宏
 * 覆盖 vLLM 支持的三种主要浮点类型：
 *   - Float (float32)：单精度浮点
 *   - Half (float16)：半精度浮点
 *   - BFloat16：Brain 浮点 16 位，训练和推理中广泛使用
 */
#define VLLM_DISPATCH_CASE_FLOATING_TYPES(...)         \
  AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)  \
  AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__)

/*
 * 中文注释：VLLM_DISPATCH_FLOATING_TYPES —— 浮点类型分发入口宏
 * 根据运行时 tensor 的 ScalarType 分发到对应的模板特化。
 * 调用后在 __VA_ARGS__ 中可使用 scalar_t 作为编译期类型参数。
 *
 * 使用示例：
 *   VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "my_kernel", [&] {
 *       my_kernel_func<scalar_t>(input.data_ptr<scalar_t>(), ...);
 *   });
 */
#define VLLM_DISPATCH_FLOATING_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(TYPE, NAME, VLLM_DISPATCH_CASE_FLOATING_TYPES(__VA_ARGS__))

/*
 * 中文注释：VLLM_DISPATCH_CASE_HALF_TYPES —— 仅半精度类型 case 宏
 * 只覆盖 Half (float16) 和 BFloat16 两种半精度类型，不包含 float32。
 * 用于那些只在半精度下有意义的 kernel（如 FP16/BF16 专用的量化、融合操作等）。
 */
#define VLLM_DISPATCH_CASE_HALF_TYPES(...)            \
  AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__)

/*
 * 中文注释：VLLM_DISPATCH_HALF_TYPES —— 半精度类型分发入口宏
 * 仅在 tensor 为 float16 或 bfloat16 时分发执行，
 * 其他类型会触发 PyTorch 默认的错误处理（抛出异常）。
 */
#define VLLM_DISPATCH_HALF_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(TYPE, NAME, VLLM_DISPATCH_CASE_HALF_TYPES(__VA_ARGS__))

// ROCm devices might use either fn or fnuz, so set up dispatch table for both.
// A host-based check at runtime will create a preferred FP8 type for ROCm
// such that the correct kernel is dispatched.

/*
 * 中文注释：ROCm 平台的 FP8 / 量化类型分发
 * =========================================================================
 * 背景：
 *   AMD ROCm 平台上，FP8 数据类型存在两种变体：
 *   - Float8_e4m3fn：标准的 FP8 E4M3 格式（尾数4位，指数3位）
 *   - Float8_e4m3fnuz：ROCm 特有的 "fnuz" 变体（NaN/Inf 处理方式不同）
 *
 * 策略：
 *   在编译期通过条件编译 #ifdef USE_ROCM 同时支持两种变体，
 *   运行时由 host 端检查选择正确的 FP8 类型，确保 kernel 能正确分发。
 *
 * QUANT_TYPES（量化类型）额外包含 Char (int8)：
 *   除了 FP8 外，还支持 INT8 量化，用于 weight-only quantization 场景。
 * =========================================================================
 */
#ifdef USE_ROCM
  #define VLLM_DISPATCH_CASE_FP8_TYPES(...)                          \
    AT_DISPATCH_FP8_CASE(at::ScalarType::Float8_e4m3fn, __VA_ARGS__) \
    AT_DISPATCH_FP8_CASE(at::ScalarType::Float8_e4m3fnuz, __VA_ARGS__)

  #define VLLM_DISPATCH_CASE_QUANT_TYPES(...)                      \
    AT_DISPATCH_CASE(at::ScalarType::Float8_e4m3fn, __VA_ARGS__)   \
    AT_DISPATCH_CASE(at::ScalarType::Float8_e4m3fnuz, __VA_ARGS__) \
    AT_DISPATCH_CASE(at::ScalarType::Char, __VA_ARGS__)
#else
  #define VLLM_DISPATCH_CASE_FP8_TYPES(...) \
    AT_DISPATCH_FP8_CASE(at::ScalarType::Float8_e4m3fn, __VA_ARGS__)

  #define VLLM_DISPATCH_CASE_QUANT_TYPES(...)                    \
    AT_DISPATCH_CASE(at::ScalarType::Float8_e4m3fn, __VA_ARGS__) \
    AT_DISPATCH_CASE(at::ScalarType::Char, __VA_ARGS__)
#endif

// When using this dispatch macro, the type is 'fp8_t' not 'scalar_t'.
// See AT_DISPATCH_FP8_CASE above.

/*
 * 中文注释：VLLM_DISPATCH_FP8_TYPES —— FP8 类型分发入口宏
 * 将运行时 tensor 类型分发到 FP8 的模板特化代码。
 * 分发后在 lambda 中可使用 fp8_t 作为编译期类型参数。
 *
 * 典型使用场景：
 *   FP8 量化的线性层（如 FP8 GEMM）中，对权重 tensor 调用此宏，
 *   在内部用 fp8_t 访问权重数据。
 */
#define VLLM_DISPATCH_FP8_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(TYPE, NAME, VLLM_DISPATCH_CASE_FP8_TYPES(__VA_ARGS__))

/*
 * 中文注释：VLLM_DISPATCH_QUANT_TYPES —— 量化类型分发入口宏
 * 覆盖所有支持的量化类型（FP8 + INT8），用于通用量化 kernel。
 * 使用 scalar_t 作为类型参数名。
 */
#define VLLM_DISPATCH_QUANT_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(TYPE, NAME, VLLM_DISPATCH_CASE_QUANT_TYPES(__VA_ARGS__))

/*
 * 中文注释：VLLM_DISPATCH_CASE_FLOATING_AND_BYTE_TYPES —— 浮点 + 字节类型 case 宏
 * 在三种浮点类型基础上，额外覆盖 Byte (uint8) 类型。
 * 用于需要同时处理浮点激活值和字节级数据（如 mask、索引等）的 kernel。
 */
#define VLLM_DISPATCH_CASE_FLOATING_AND_BYTE_TYPES(...)   \
  AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__)    \
  AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)     \
  AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::Byte, __VA_ARGS__)

/*
 * 中文注释：VLLM_DISPATCH_FLOATING_AND_BYTE_TYPES —— 浮点 + 字节类型分发入口宏
 * 对应 case 宏的入口，适用于需要处理混合类型输入的 kernel。
 */
#define VLLM_DISPATCH_FLOATING_AND_BYTE_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(TYPE, NAME,                               \
                     VLLM_DISPATCH_CASE_FLOATING_AND_BYTE_TYPES(__VA_ARGS__))

/*
 * 中文注释：VLLM_DISPATCH_CASE_INTEGRAL_TYPES —— 有符号整数类型 case 宏
 * 覆盖常见的有符号整数类型：
 *   - Byte (uint8)、Char (int8)、Short (int16)、Int (int32)、Long (int64)
 * 用于需要整数运算的 kernel（如索引计算、token ID 处理等）。
 */
#define VLLM_DISPATCH_CASE_INTEGRAL_TYPES(...)         \
  AT_DISPATCH_CASE(at::ScalarType::Byte, __VA_ARGS__)  \
  AT_DISPATCH_CASE(at::ScalarType::Char, __VA_ARGS__)  \
  AT_DISPATCH_CASE(at::ScalarType::Short, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::Int, __VA_ARGS__)   \
  AT_DISPATCH_CASE(at::ScalarType::Long, __VA_ARGS__)

/*
 * 中文注释：VLLM_DISPATCH_CASE_INTEGRAL_AND_UNSIGNED_TYPES —— 整数 + 无符号整数 case 宏
 * 在有符号整数基础上，额外覆盖无符号整数类型：
 *   - UInt16、UInt32、UInt64
 * 用于需要处理无符号索引或位操作的 kernel。
 */
#define VLLM_DISPATCH_CASE_INTEGRAL_AND_UNSIGNED_TYPES(...) \
  AT_DISPATCH_CASE(at::ScalarType::Byte, __VA_ARGS__)       \
  AT_DISPATCH_CASE(at::ScalarType::Char, __VA_ARGS__)       \
  AT_DISPATCH_CASE(at::ScalarType::Short, __VA_ARGS__)      \
  AT_DISPATCH_CASE(at::ScalarType::Int, __VA_ARGS__)        \
  AT_DISPATCH_CASE(at::ScalarType::Long, __VA_ARGS__)       \
  AT_DISPATCH_CASE(at::ScalarType::UInt16, __VA_ARGS__)     \
  AT_DISPATCH_CASE(at::ScalarType::UInt32, __VA_ARGS__)     \
  AT_DISPATCH_CASE(at::ScalarType::UInt64, __VA_ARGS__)

/*
 * 中文注释：VLLM_DISPATCH_INTEGRAL_TYPES —— 整数类型分发入口宏
 */
#define VLLM_DISPATCH_INTEGRAL_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(TYPE, NAME, VLLM_DISPATCH_CASE_INTEGRAL_TYPES(__VA_ARGS__))

/*
 * 中文注释：VLLM_DISPATCH_INTEGRAL_AND_UNSIGNED_TYPES —— 整数 + 无符号整数分发入口宏
 */
#define VLLM_DISPATCH_INTEGRAL_AND_UNSIGNED_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(                                              \
      TYPE, NAME, VLLM_DISPATCH_CASE_INTEGRAL_AND_UNSIGNED_TYPES(__VA_ARGS__))

/*
 * 中文注释：VLLM_DISPATCH_VEC_SIZE —— 向量大小编译期分发宏
 * =========================================================================
 * 功能：
 *   将运行时的向量加载大小（VEC_SIZE）转化为编译期常量 vec_size，
 *   从而让编译器为每种向量大小生成优化的 SIMD/SIMT 代码。
 *
 * 支持的向量大小：
 *   16、8、4、2、1（默认 fallback）
 *
 * 设计原理：
 *   GPU kernel 中的向量化内存访问（vectorized load/store）要求在编译期
 *   确定向量宽度，以便编译器生成对应的 LDG.128/64/32 等指令。
 *   此宏通过 switch-case 将运行时值映射到 constexpr 常量，
 *   使每个分支内的代码都能被编译器优化。
 *
 * 使用场景：
 *   量化反量化 kernel、融合算子中需要根据数据类型选择向量宽度时。
 *   例如：float32 可用 vec_size=4（128位），float16 可用 vec_size=8（128位）。
 * =========================================================================
 */
#define VLLM_DISPATCH_VEC_SIZE(VEC_SIZE, ...) \
  switch (VEC_SIZE) {                         \
    case 16: {                                \
      constexpr int vec_size = 16;            \
      __VA_ARGS__();                          \
      break;                                  \
    }                                         \
    case 8: {                                 \
      constexpr int vec_size = 8;             \
      __VA_ARGS__();                          \
      break;                                  \
    }                                         \
    case 4: {                                 \
      constexpr int vec_size = 4;             \
      __VA_ARGS__();                          \
      break;                                  \
    }                                         \
    case 2: {                                 \
      constexpr int vec_size = 2;             \
      __VA_ARGS__();                          \
      break;                                  \
    }                                         \
    default: {                                \
      constexpr int vec_size = 1;             \
      __VA_ARGS__();                          \
      break;                                  \
    }                                         \
  }

/*
 * 中文注释：VLLM_DISPATCH_BOOL —— 布尔值编译期分发宏
 * =========================================================================
 * 功能：
 *   将运行时的布尔表达式转化为编译期常量，使编译器能根据 true/false
 *   两个分支分别优化（如消除死代码、展开条件分支等）。
 *
 * 参数说明：
 *   - expr：运行时的布尔表达式（如 config.use_flash_attn）
 *   - const_expr：在 lambda 内可用的 constexpr bool 变量名
 *   - ...：接受 lambda 的代码块
 *
 * 使用场景：
 *   kernel 中的条件逻辑（如是否启用某种优化、是否应用 mask 等）
 *   可通过此宏在编译期确定，避免运行时分支开销。
 * =========================================================================
 */
#define VLLM_DISPATCH_BOOL(expr, const_expr, ...) \
  if (expr) {                                     \
    constexpr bool const_expr = true;             \
    __VA_ARGS__();                                \
  } else {                                        \
    constexpr bool const_expr = false;            \
    __VA_ARGS__();                                \
  }

/*
 * 中文注释：VLLM_DISPATCH_GROUP_SIZE —— 量化分组大小编译期分发宏
 * =========================================================================
 * 功能：
 *   将运行时的量化分组大小（group_size）转化为编译期常量 const_group_size，
 *   用于 per-group quantization 场景下的 kernel 优化。
 *
 * 支持的分组大小：
 *   - 128：每 128 个元素共享一组量化参数（scale/zero_point）
 *   - 64：更细粒度的分组，精度更高但开销更大
 *
 * 使用场景：
 *   FP8/INT8 per-group 量化的反量化 kernel 中，需要在编译期知道
 *   分组大小以便展开循环和优化内存访问模式。
 * =========================================================================
 */
#define VLLM_DISPATCH_GROUP_SIZE(group_size, const_group_size, ...) \
  if (group_size == 128) {                                          \
    constexpr int const_group_size = 128;                           \
    __VA_ARGS__();                                                  \
  } else if (group_size == 64) {                                    \
    constexpr int const_group_size = 64;                            \
    __VA_ARGS__();                                                  \
  }

/*
 * 中文注释：VLLM_DISPATCH_RANK234 —— Tensor 维度编译期分发宏
 * =========================================================================
 * 功能：
 *   将运行时的 tensor 维度数（rank）转化为编译期常量 tensor_rank，
 *   支持 2D、3D、4D tensor 的分发。
 *
 * 支持的维度：
 *   - 2D：如 (batch, seq_len) 的 token ID tensor
 *   - 3D：如 (batch, seq_len, hidden_dim) 的激活值 tensor
 *   - 4D：如 (batch, channels, height, width) 的图像特征 tensor
 *
 * 设计原理：
 *   许多 CUDA kernel 需要根据 tensor 维度数选择不同的索引计算方式，
 *   在编译期确定维度数可以避免运行时条件判断，并让编译器进行
 *   更激进的优化（如常量折叠、消除未使用的维度索引等）。
 *
 * 错误处理：
 *   不支持的维度数会触发 TORCH_CHECK 断言失败，输出错误信息。
 * =========================================================================
 */
#define VLLM_DISPATCH_RANK234(NUM_DIMS, ...)                                   \
  switch (NUM_DIMS) {                                                          \
    case 2: {                                                                  \
      constexpr int tensor_rank = 2;                                           \
      __VA_ARGS__();                                                           \
      break;                                                                   \
    }                                                                          \
    case 3: {                                                                  \
      constexpr int tensor_rank = 3;                                           \
      __VA_ARGS__();                                                           \
      break;                                                                   \
    }                                                                          \
    case 4: {                                                                  \
      constexpr int tensor_rank = 4;                                           \
      __VA_ARGS__();                                                           \
      break;                                                                   \
    }                                                                          \
    default:                                                                   \
      TORCH_CHECK(false, "Expects rank 2, 3 or 4 tensors but got ", NUM_DIMS); \
  }
