/*
 * 中文注释：FP32 数据类型的向量运算实现。
 *
 * 本文件的核心作用：
 *   1. 定义 Float4_ 和 Float8_ 自定义累加器类型，分别包含 4 个和 8 个 float 分量，
 *      用于半精度（FP16/BF16）Attention 计算时的 FP32 累加，避免精度损失。
 *   2. 特化 Vec<float, N> —— FP32 的 Q/K/V 向量类型映射。
 *   3. 特化 FloatVec<T> —— FP32 累加器向量类型映射。
 *   4. 实现 FP32 向量的 add、mul、fma、sum、dot、from_float、to_float、zero 操作。
 *
 * 被 dtype_float16.cuh 和 dtype_bfloat16.cuh 依赖，因为半精度的 FloatVec
 * 累加器类型需要用到此文件中定义的 Float4_、Float8_。
 *
 * 源自 NVIDIA FasterTransformer 项目，经过 vLLM 团队修改。
 */
/*
 * Adapted from
 * https://github.com/NVIDIA/FasterTransformer/blob/release/v5.3_tag/src/fastertransformer/kernels/decoder_masked_multihead_attention/decoder_masked_multihead_attention_template.hpp
 * and
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

#include "attention_generic.cuh"

#include <stdint.h>

namespace vllm {

// Define custom FP32 vector data types.
// 中文注释：自定义 FP32 累加器向量类型。
// Float4_ 包含 4 个 float 分量（以 2 个 float2 存储），对应半精度 4 元素向量的累加结果。
// Float8_ 包含 8 个 float 分量（以 4 个 float2 存储），对应半精度 8 元素向量的累加结果。
// 在 PagedAttention 中，Q*K^T 和 softmax*V 的累加都使用 FP32 以保证数值精度。
struct Float4_ {
  float2 x;
  float2 y;
};

struct Float8_ {
  float2 x;
  float2 y;
  float2 z;
  float2 w;
};

// FP32 vector types for Q, K, V.
// 中文注释：FP32 标量类型对应的 Vec 特化，用于 FP32 模式的 Q/K/V 存储。
template <>
struct Vec<float, 1> {
  using Type = float;
};
template <>
struct Vec<float, 2> {
  using Type = float2;
};
template <>
struct Vec<float, 4> {
  using Type = float4;
};

// FP32 accumulator vector types corresponding to Vec.
// 中文注释：FP32 累加器类型特化。对于 FP32 输入，累加器类型与输入类型相同。
template <>
struct FloatVec<float> {
  using Type = float;
};
template <>
struct FloatVec<float2> {
  using Type = float2;
};
template <>
struct FloatVec<float4> {
  using Type = float4;
};

// Vector addition.
// 中文注释：FP32 向量逐元素加法，支持 float/float2/float4 三种宽度。
inline __device__ float add(float a, float b) { return a + b; }

inline __device__ float2 add(float2 a, float2 b) {
  float2 c;
  c.x = add(a.x, b.x);
  c.y = add(a.y, b.y);
  return c;
}

inline __device__ float4 add(float4 a, float4 b) {
  float4 c;
  c.x = add(a.x, b.x);
  c.y = add(a.y, b.y);
  c.z = add(a.z, b.z);
  c.w = add(a.w, b.w);
  return c;
}

// Vector multiplication.
// 中文注释：FP32 向量逐元素乘法。支持标量*标量、向量*向量、标量*向量三种形式。
// 标量*向量形式在 Attention 中用于将 softmax 权重乘以 V 向量（广播乘法）。
template <>
inline __device__ float mul<float, float>(float a, float b) {
  return a * b;
}

template <>
inline __device__ float2 mul(float2 a, float2 b) {
  float2 c;
  c.x = a.x * b.x;
  c.y = a.y * b.y;
  return c;
}

template <>
inline __device__ float2 mul(float a, float2 b) {
  float2 c;
  c.x = a * b.x;
  c.y = a * b.y;
  return c;
}

template <>
inline __device__ float4 mul(float4 a, float4 b) {
  float4 c;
  c.x = a.x * b.x;
  c.y = a.y * b.y;
  c.z = a.z * b.z;
  c.w = a.w * b.w;
  return c;
}

template <>
inline __device__ float4 mul(float a, float4 b) {
  float4 c;
  c.x = a * b.x;
  c.y = a * b.y;
  c.z = a * b.z;
  c.w = a * b.w;
  return c;
}

// Vector fused multiply-add.
// 中文注释：融合乘加 (Fused Multiply-Add) 操作，计算 a*b + c。
// 这是 Attention kernel 中最核心的操作：在计算 Q*K^T 的累积点积时，
// 每次循环迭代执行 vec_q[i]*vec_k[i] + acc，一条 FMA 指令完成乘法和加法。
// 支持 float、float2、float4、Float4_、Float8_ 各种向量宽度。
inline __device__ float fma(float a, float b, float c) { return a * b + c; }

inline __device__ float2 fma(float2 a, float2 b, float2 c) {
  float2 d;
  d.x = fma(a.x, b.x, c.x);
  d.y = fma(a.y, b.y, c.y);
  return d;
}

inline __device__ float2 fma(float a, float2 b, float2 c) {
  float2 d;
  d.x = fma(a, b.x, c.x);
  d.y = fma(a, b.y, c.y);
  return d;
}

inline __device__ float4 fma(float4 a, float4 b, float4 c) {
  float4 d;
  d.x = fma(a.x, b.x, c.x);
  d.y = fma(a.y, b.y, c.y);
  d.z = fma(a.z, b.z, c.z);
  d.w = fma(a.w, b.w, c.w);
  return d;
}

inline __device__ float4 fma(float a, float4 b, float4 c) {
  float4 d;
  d.x = fma(a, b.x, c.x);
  d.y = fma(a, b.y, c.y);
  d.z = fma(a, b.z, c.z);
  d.w = fma(a, b.w, c.w);
  return d;
}

inline __device__ Float4_ fma(float a, Float4_ b, Float4_ c) {
  Float4_ d;
  d.x = fma(a, b.x, c.x);
  d.y = fma(a, b.y, c.y);
  return d;
}

inline __device__ Float8_ fma(float a, Float8_ b, Float8_ c) {
  Float8_ d;
  d.x = fma(a, b.x, c.x);
  d.y = fma(a, b.y, c.y);
  d.z = fma(a, b.z, c.z);
  d.w = fma(a, b.w, c.w);
  return d;
}

// Vector sum.
// 中文注释：向量元素水平求和，将向量所有分量累加为一个标量 float。
// 用于 dot() 函数中完成点积的最后一步归约（将 mul/fma 的向量结果归约为标量）。
template <>
inline __device__ float sum(float v) {
  return v;
}

template <>
inline __device__ float sum(float2 v) {
  return v.x + v.y;
}

template <>
inline __device__ float sum(float4 v) {
  return v.x + v.y + v.z + v.w;
}

template <>
inline __device__ float sum(Float4_ v) {
  return v.x.x + v.x.y + v.y.x + v.y.y;
}

template <>
inline __device__ float sum(Float8_ v) {
  return v.x.x + v.x.y + v.y.x + v.y.y + v.z.x + v.z.y + v.w.x + v.w.y;
}

// Vector dot product.
// 中文注释：向量点积运算，计算两个向量的内积（对应元素相乘再求和）。
// 在 PagedAttention 中用于计算 Q 和 K 的点积（Q*K^T），是 Attention score 的核心计算。
// 实现上先用 mul/fma 做向量化乘法累加，再用 sum 归约为标量。
inline __device__ float dot(float a, float b) { return a * b; }

inline __device__ float dot(float2 a, float2 b) {
  float2 c = mul<float2, float2, float2>(a, b);
  return c.x + c.y;
}

inline __device__ float dot(Float4_ a, Float4_ b) {
  float2 acc = mul<float2, float2, float2>(a.x, b.x);
  acc = fma(a.y, b.y, acc);
  return acc.x + acc.y;
}

inline __device__ float dot(Float8_ a, Float8_ b) {
  float2 acc = mul<float2, float2, float2>(a.x, b.x);
  acc = fma(a.y, b.y, acc);
  acc = fma(a.z, b.z, acc);
  acc = fma(a.w, b.w, acc);
  return acc.x + acc.y;
}

// From float to float.
// 中文注释：FP32 -> FP32 的类型转换（恒等操作）。统一接口，使半精度文件可以
// 通过 from_float/to_float 在 FP32 和低精度之间自由转换。
inline __device__ void from_float(float& dst, float src) { dst = src; }

inline __device__ void from_float(float2& dst, float2 src) { dst = src; }

inline __device__ void from_float(float4& dst, float4 src) { dst = src; }

// From float to float.
inline __device__ float to_float(float u) { return u; }

inline __device__ float2 to_float(float2 u) { return u; }

inline __device__ float4 to_float(float4 u) { return u; }

inline __device__ Float4_ to_float(Float4_ u) { return u; }

inline __device__ Float8_ to_float(Float8_ u) { return u; }

// Zero-out a variable.
// 中文注释：将 FP32 变量清零。
inline __device__ void zero(float& dst) { dst = 0.f; }

}  // namespace vllm
