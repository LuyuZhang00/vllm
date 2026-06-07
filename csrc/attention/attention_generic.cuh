/*
 * 中文注释：本文件是 vLLM PagedAttention 向量运算的通用抽象层。
 * 它定义了 Vec、FloatVec 模板结构体和 mul/sum/dot/zero 等模板函数的通用接口，
 * 由各数据类型的具体实现文件（dtype_float16.cuh、dtype_bfloat16.cuh 等）来特化。
 *
 * 整体架构说明：
 *   1. Vec<T, VEC_SIZE> — 用于存储 Q/K/V 元素的向量类型，T 是标量类型
 *      （如 uint16_t 表示 FP16），VEC_SIZE 是向量元素个数。
 *   2. FloatVec<T> — 用于存储 FP32 累加结果的向量类型，保证计算精度。
 *   3. 各特化实现提供：向量加法(add)、向量乘法(mul)、融合乘加(fma)、
 *      元素求和(sum)、类型转换(from_float/to_float)等操作。
 *
 * 这些向量化运算在 PagedAttention kernel 中被大量使用，目的是：
 *   - 利用 GPU 的宽向量 load/store（如 128-bit）提升显存带宽利用率
 *   - 利用向量化算术指令（如 f16x2）提升计算吞吐
 *
 * 源自 NVIDIA FasterTransformer 项目，经过 vLLM 团队修改。
 */
/*
 * Adapted from
 * https://github.com/NVIDIA/FasterTransformer/blob/release/v5.3_tag/src/fastertransformer/kernels/decoder_masked_multihead_attention_utils.h
 * Copyright (c) 2023, The vLLM team.
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <stdint.h>

namespace vllm {

// A vector type to store Q, K, V elements.
// 中文注释：通用向量类型模板，用于存储 Q、K、V 的元素。
// 具体的类型映射（如 T=uint16_t, VEC_SIZE=4 对应 uint2）在各 dtype 文件中特化。
template <typename T, int VEC_SIZE>
struct Vec {};

// A vector type to store FP32 accumulators.
// 中文注释：FP32 累加器向量类型模板。Attention 计算中为保证精度，
// 需要将中间结果以 FP32 累加，此类型用于存储 FP32 版本的向量。
template <typename T>
struct FloatVec {};

// Template vector operations.
// 中文注释：声明向量乘法模板函数，具体实现由各 dtype 文件特化提供。
template <typename Acc, typename A, typename B>
inline __device__ Acc mul(A a, B b);

// 中文注释：声明向量元素求和模板函数，将向量中所有分量累加为一个 float 标量。
template <typename T>
inline __device__ float sum(T v);

// 中文注释：向量点积 = 求和(向量逐元素乘积)。这是 Attention 中 Q*K^T 的基本操作。
template <typename T>
inline __device__ float dot(T a, T b) {
  return sum(mul<T, T, T>(a, b));
}

// 中文注释：带累加器类型 A 的点积版本，乘法结果先提升到累加器类型再求和，
// 用于需要更高精度的场景（如半精度输入但 FP32 累加）。
template <typename A, typename T>
inline __device__ float dot(T a, T b) {
  return sum(mul<A, T, T>(a, b));
}

// 中文注释：将变量清零的通用模板。通过 union 将变量按 32-bit word 逐个置零。
// 这种方式比直接赋值零值更安全，因为不依赖于类型特定的零值构造。
template <typename T>
inline __device__ void zero(T& dst) {
  constexpr int WORDS = sizeof(T) / 4;
  union {
    T raw;
    uint32_t words[WORDS];
  } tmp;

#pragma unroll
  for (int ii = 0; ii < WORDS; ++ii) {
    tmp.words[ii] = 0u;
  }
  dst = tmp.raw;
}

}  // namespace vllm
