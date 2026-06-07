#pragma once

// =============================================================================
// 中文注释: FP8 量化公共工具头文件
// =============================================================================
// 本文件提供 FP8 量化/反量化的核心工具函数，是 FP8 量化模块的基础层。
//
// 核心功能:
//   1. is_fp8_ocp(): 检测当前平台是否支持 OCP (Open Compute Project) FP8 标准
//      - NVIDIA GPU: 始终返回 true (从 Hopper 架构开始支持)
//      - AMD GPU: 检查是否为 gfx94x 架构 (MI300 系列)
//
//   2. atomicMaxFloat(): 浮点数原子最大值操作
//      - 利用 int/uint 的原子操作实现浮点原子 max
//      - 正数用 atomicMax，负数用 atomicMin (因为负数的位表示与大小顺序相反)
//
//   3. scaled_fp8_conversion(): 带缩放因子的 FP8 量化核心函数
//      - is_scale_inverted=true: val * scale (scale = 1/原始scale，用乘法代替除法)
//      - is_scale_inverted=false: val / scale (传统除法)
//      - 饱和截断到 [-quant_max, quant_max] 范围
//      - NVIDIA: 使用硬件 cvt 指令 (__nv_cvt_float_to_fp8)
//      - AMD: 使用 HIP 的 cvt 指令
//
// 设计说明:
//   - 量化公式: fp8_val = clamp(val / scale, -fp8_max, fp8_max)
//   - 反量化公式: real_val = fp8_val * scale
//   - 使用 is_scale_inverted 优化: 当 scale 已经预计算为 1/scale 时，
//     用乘法代替除法，乘法在 GPU 上更快
// =============================================================================

#include "libtorch_stable/quantization/vectorization.cuh"
#include "../../utils.cuh"

#include <cmath>

// This header is shared between _C and _C_stable_libtorch targets.
// torch_utils.h provides get_device_prop(). We need to pass USE_CUDA
// to the .so to expose some of the shims used by torch_utils.h. For now
// this is only done for _C_stable_libtorch and not for _C, so we use the
// non stable at::cuda::getCurrentDeviceProperties for _C for now.
// 中文注释: 平台适配层
// - _C_stable_libtorch 目标: 使用自定义的 torch_utils.h 获取设备属性
// - _C 目标 (CUDA): 使用 at::cuda::getCurrentDeviceProperties
// - _C 目标 (ROCm): 使用 ATen 的 HIPContext
#ifdef TORCH_TARGET_VERSION
  #include "../../../libtorch_stable/torch_utils.h"
#else
  #ifdef USE_ROCM
    #include <ATen/hip/HIPContext.h>
  #endif
#endif

// 中文注释: 根据平台选择对应的 FP8 量化工具实现
// - NVIDIA: nvidia/quant_utils.cuh (使用 CUDA __nv_fp8 类型和硬件 cvt 指令)
// - AMD: amd/quant_utils.cuh (使用 HIP __hip_fp8 类型)
// 两者提供相同的接口 (vec_conversion, scaled_vec_conversion)，但底层实现不同
#ifndef USE_ROCM
  #include "nvidia/quant_utils.cuh"
#else
  #include "amd/quant_utils.cuh"
#endif

// Determines the preferred FP8 type for the current platform.
// Note that for CUDA this just returns true,
// but on ROCm it will check device props.
// 中文注释: 检测当前平台是否使用 OCP FP8 标准
// - NVIDIA (CUDA): 始终返回 true，NVIDIA 统一使用 OCP FP8 (Float8_e4m3fn)
// - AMD (ROCm): 检查 GPU 架构
//   - gfx94x (MI300 系列): 返回 false，使用 FNUZ 变体 (Float8_e4m3fnuz)
//   - 其他架构 (如 gfx95x): 返回 true，使用 OCP 标准
// 这个函数决定了量化时选择 Float8_e4m3fn 还是 Float8_e4m3fnuz 类型
static bool is_fp8_ocp() {
#ifndef USE_ROCM
  return true;
#else
  #ifdef TORCH_TARGET_VERSION
  auto* dprops = get_device_prop();
  #else
  auto* dprops = at::cuda::getCurrentDeviceProperties();
  #endif
  std::string device_arch = dprops->gcnArchName;
  size_t substring = device_arch.find("gfx94");
  return substring == std::string::npos;
#endif
}

namespace vllm {

// 中文注释: 浮点数原子最大值操作
// 用途: 在 kernel 中并行计算量化缩放因子时，需要求全局最大绝对值
// 实现原理:
//   - 正数 (value >= 0): 直接用 atomicMax 对 int 表示做比较
//   - 负数 (value < 0): 用 atomicMin 对 unsigned int 表示做比较
//     (IEEE 754 浮点数的位表示中，负数的大小顺序与无符号整数相反)
__device__ __forceinline__ float atomicMaxFloat(float* addr, float value) {
  float old;
  old = (value >= 0)
            ? __int_as_float(atomicMax((int*)addr, __float_as_int(value)))
            : __uint_as_float(
                  atomicMin((unsigned int*)addr, __float_as_uint(value)));

  return old;
}

// 中文注释: 带缩放因子的 FP8 量化核心函数
// 这是整个 FP8 量化模块最核心的转换函数
//
// 参数:
//   val: 待量化的浮点值
//   scale: 缩放因子
//   is_scale_inverted: 是否为反转的缩放因子 (即 1/scale)
//
// 计算流程:
//   1. 根据 is_scale_inverted 决定用乘法还是除法
//      - true:  x = val * scale (scale 已预计算为 1/original_scale)
//      - false: x = val / scale (传统方式)
//   2. 饱和截断: x = clamp(x, -fp8_max, fp8_max)
//   3. 调用平台特定的硬件转换指令:
//      - NVIDIA: __nv_cvt_float_to_fp8 (SM >= 80)
//      - AMD: __hip_cvt_float_to_fp8
//
// 使用场景:
//   - 激活量化: activation_kernels.cu 中的 act_and_mul_quant_kernel
//   - 权重量化: 各种 W8A8 kernel
//   - KV cache 量化: attention 层的 FP8 KV cache
template <bool is_scale_inverted, typename fp8_type>
__device__ __forceinline__ fp8_type scaled_fp8_conversion(float const val,
                                                          float const scale) {
  float x = 0.0f;
  if constexpr (is_scale_inverted) {
    x = val * scale;
  } else {
    x = val / scale;
  }

  float r =
      fmaxf(-quant_type_max_v<fp8_type>, fminf(x, quant_type_max_v<fp8_type>));
#ifndef USE_ROCM
  // Use hardware cvt instruction for fp8 on nvidia
  // Currently only support fp8_type = c10::Float8_e4m3fn
  return fp8::vec_conversion<fp8_type, float>(r);
#else
  // Use hardware cvt instruction for fp8 on rocm
  return fp8::cvt_c10<fp8_type>(r);
#endif
}

}  // namespace vllm
