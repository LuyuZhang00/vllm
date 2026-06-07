/*
 * 中文注释：本文件是 PagedAttention 数据类型支持的统一头文件。
 * 它将所有支持的数据类型实现汇总在一起：
 *   - attention_generic.cuh：通用向量运算接口（Vec、FloatVec、mul、sum、dot 等）
 *   - dtype_float16.cuh：FP16 (half) 类型的向量运算特化
 *   - dtype_float32.cuh：FP32 类型的向量运算特化（同时定义 Float4_、Float8_ 累加器类型）
 *   - dtype_bfloat16.cuh：BF16 类型的向量运算特化
 *   - dtype_fp8.cuh：FP8 类型的向量定义（用于 KV cache 量化场景）
 *
 * 使用时只需 #include "attention_dtypes.h"，即可获得所有数据类型的 Attention 向量运算支持。
 */
#pragma once

#include "attention_generic.cuh"
#include "dtype_float16.cuh"
#include "dtype_float32.cuh"
#include "dtype_bfloat16.cuh"
#include "dtype_fp8.cuh"
