/*
 * Adapted from SGLang's sgl-kernel implementation, which was adapted from
 * https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/kernels/dsv3MinLatencyKernels/dsv3RouterGemm.cu
 * https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/thop/dsv3RouterGemmOp.cpp
 *
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
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

// =============================================================================
// 中文注释：DeepSeek V3 Router GEMM 工具函数
//
// 本文件提供 DSV3 Router GEMM 所需的辅助函数。
// getSMVersion() 获取当前 GPU 的 SM 版本号（如 SM90 返回 90），
// 用于在运行时判断是否支持特定的 PTX 指令（如 griddepcontrol）。
// =============================================================================

#pragma once

#include <ATen/cuda/CUDAContext.h>

#include <cstdlib>
#include <mutex>

inline int getSMVersion() {
  auto* props = at::cuda::getCurrentDeviceProperties();
  return props->major * 10 + props->minor;
}
