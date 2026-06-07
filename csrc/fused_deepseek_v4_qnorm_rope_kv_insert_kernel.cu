/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 *
 * Horizontally-fused DeepseekV4-MLA kernel:
 *   - Q side:  per-head RMSNorm (no weight) + GPT-J RoPE on last ROPE_DIM
 *   - KV side: GPT-J RoPE on last ROPE_DIM + UE8M0 FP8 quant on NoPE + paged
 *              cache insert
 *
 * Structured after `applyMLARopeAndAssignQKVKernelGeneration` in
 * TensorRT-LLM's mlaKernels.cu: one kernel, one grid, with head-slot
 * dispatch choosing Q vs KV work per warp.  The per-warp RMSNorm/RoPE
 * skeleton is adapted from vllm-deepseek_v4's existing
 * `fusedQKNormRopeKernel` (csrc/fused_qknorm_rope_kernel.cu).
 *
 * Assumptions (hard-coded for DeepseekV4 attention):
 *   HEAD_DIM  = 512
 *   ROPE_DIM  = 64   (RoPE applied to dims [NOPE_DIM, HEAD_DIM))
 *   NOPE_DIM  = 448
 *   QUANT_BLOCK = 64 (UE8M0 FP8 quant block)
 *   FP8_MAX   = 448.0f
 *   is_neox=false (GPT-J interleaved pairs)
 *   cos_sin_cache layout [max_pos, rope_dim] = cos || sin (cos first, sin
 *     second along last dim; each half is rope_dim/2 = 32 values)
 *
 * Cache layout per paged-cache block (block_size tokens):
 *   [0,            bs*576):          token data, 448 fp8 + 128 bf16 each
 *   [bs*576,       bs*576 + bs*8):   UE8M0 scales, 7 real + 1 pad per token
 *
 *
 * ============================================================================
 * 文件功能概述（中文）
 * ============================================================================
 * 本文件实现了 DeepseekV4 模型的 MLA（Multi-head Latent Attention）中
 * 一个高度融合的 CUDA kernel，将以下多个操作合并到一次 kernel launch 中：
 *
 *   1. Q 侧处理（每个 Q head 独立执行）：
 *      (a) RMSNorm（无权重版本，即仅做归一化不乘 gamma）
 *      (b) GPT-J 风格的旋转位置编码（RoPE），仅作用于最后 ROPE_DIM=64 维
 *
 *   2. KV 侧处理（所有 Q head 共享一份 KV）：
 *      (a) GPT-J 风格的 RoPE，同样仅作用于最后 ROPE_DIM=64 维
 *      (b) 对 NoPE 部分（前 448 维）做 UE8M0 FP8 量化
 *      (c) 将量化后的 KV 写入分页 KV cache
 *
 * 设计思路：
 *   - 采用"水平融合"（horizontally-fused）策略：一个 kernel、一个 grid，
 *     通过 warp 级别的 slot 分发来决定每个 warp 执行 Q 还是 KV 工作。
 *   - 每个 token 对应 (num_heads_q_padded + 1) 个 slot：
 *       * slot < num_heads_q        -> 处理真实的 Q head（RMSNorm + RoPE）
 *       * num_heads_q <= slot < padded -> 填充 Q slot（写零，供 FlashMLA 读取）
 *       * slot == padded              -> 处理 KV（RoPE + FP8 量化 + 写入 cache）
 *   - 这种设计将多个小 kernel 合并为一个大 kernel，减少了 kernel launch 开销
 *     和全局内存读写次数，显著提升 MLA 预处理阶段的吞吐。
 *
 * 参考实现：
 *   - TensorRT-LLM 的 applyMLARopeAndAssignQKVKernelGeneration（mlaKernels.cu）
 *   - vllm-deepseek_v4 的 fusedQKNormRopeKernel（csrc/fused_qknorm_rope_kernel.cu）
 *
 * 关键常量假设（硬编码为 DeepseekV4 注意力配置）：
 *   HEAD_DIM  = 512   -- 每个 head 的总维度
 *   ROPE_DIM  = 64    -- 应用 RoPE 的维度范围 [NOPE_DIM, HEAD_DIM)
 *   NOPE_DIM  = 448   -- 不应用 RoPE 的维度（前 448 维，将被 FP8 量化）
 *   QUANT_BLOCK = 64  -- UE8M0 FP8 量化的分组大小
 *   FP8_MAX   = 448.0f -- FP8 E4M3 格式的最大可表示值
 *   is_neox=false     -- 使用 GPT-J 的交错配对方式（非 NeoX 风格）
 *
 * cos_sin_cache 布局：[max_pos, rope_dim] = cos || sin
 *   （cos 在前、sin 在后，各占 rope_dim/2 = 32 个 float 值）
 *
 * 分页 KV cache 的每个 block 布局（block_size 个 token）：
 *   [0,            bs*576):          token 数据区，每个 token 448 字节 fp8 + 128 字节 bf16
 *   [bs*576,       bs*576 + bs*8):   UE8M0 scale 区，每个 token 7 个真实 scale + 1 个 padding
 * ============================================================================
 */

#include <cmath>
#ifndef USE_ROCM
  #include <cuda_fp8.h>
#else
  #include <hip/hip_fp8.h>
#endif
#include <cuda_runtime.h>
#include <type_traits>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/cuda.h>

#include "cuda_compat.h"
#include "dispatch_utils.h"
#include "type_convert.cuh"

#ifndef FINAL_MASK
  #ifdef USE_ROCM
    #define FINAL_MASK 0xffffffffffffffffULL
  #else
    #define FINAL_MASK 0xffffffffu
  #endif
#endif

#ifdef USE_ROCM
// ROCm-compatible FP8 conversion helpers
__device__ __forceinline__ uint8_t rocm_cvt_float_to_fp8_e4m3(float val) {
  #if defined(HIP_FP8_TYPE_OCP)
  __hip_fp8_e4m3 fp8_val(val);
  #else
  __hip_fp8_e4m3_fnuz fp8_val(val);
  #endif
  return reinterpret_cast<uint8_t&>(fp8_val);
}
#endif

namespace vllm {
namespace deepseek_v4_fused_ops {

namespace {
inline int getSMVersion() {
  auto* props = at::cuda::getCurrentDeviceProperties();
  return props->major * 10 + props->minor;
}
}  // namespace

// ────────────────────────────────────────────────────────────────────────────
// Constants
// ────────────────────────────────────────────────────────────────────────────

// 中文注释：DeepseekV4 MLA 注意力的核心维度常量
// kHeadDim = 512：每个注意力头的总维度，包含 NoPE 和 RoPE 两部分
// kRopeDim = 64：应用旋转位置编码的维度区间（后 64 维）
// kNopeDim = 448：不应用 RoPE 的维度（前 448 细），将被量化为 FP8 存储
constexpr int kHeadDim = 512;
constexpr int kRopeDim = 64;
constexpr int kNopeDim = kHeadDim - kRopeDim;  // 448

// 中文注释：UE8M0 FP8 量化的分组参数
// kQuantBlock = 64：每个量化分组包含 64 个元素
// kNumQuantBlocks = 7：NoPE 部分（448 维）分为 7 个量化分组
// kScaleBytesPerToken = 8：每个 token 需要 8 字节存放 scale（7 个真实 scale + 1 字节 padding）
constexpr int kQuantBlock = 64;
constexpr int kNumQuantBlocks = kNopeDim / kQuantBlock;   // 7
constexpr int kScaleBytesPerToken = kNumQuantBlocks + 1;  // 8 (7 real + 1 pad)

// 中文注释：每个 token 在 KV cache 中的数据区字节数
// = 448 字节（NoPE 部分，FP8 每元素 1 字节）+ 128 字节（RoPE 部分，BF16 每元素 2 字节）
// = 576 字节
constexpr int kTokenDataBytes = kNopeDim + kRopeDim * 2;  // 448 + 128 = 576

// 中文注释：FP8 E4M3 格式的最大可表示值，用于量化时的缩放基准
constexpr float kFp8Max = 448.0f;

#ifndef USE_ROCM
// When num_tokens is less than this threshold,
// run the reduced grid variant on cuda
constexpr float NUM_TOKEN_CUTOFF = 1024;
#endif

// Per-warp layout:  32 lanes × 16 elems/lane = 512 elems = HEAD_DIM.
// 中文注释：每个 warp 的工作划分方式
// - 一个 warp 有 32 个 lane（线程），每个 lane 负责处理 16 个元素
// - 32 × 16 = 512 = kHeadDim，恰好覆盖一个完整的 head 维度
// - 这样每个 warp 恰好处理一个 (token, head) 对的完整 512 维数据
constexpr int kNumLanes = 32;
constexpr int kElemsPerLane = kHeadDim / kNumLanes;  // 16

// ────────────────────────────────────────────────────────────────────────────
// Small inline helpers
// ────────────────────────────────────────────────────────────────────────────

// 中文注释：在 4 个连续 lane 之间做绝对值最大值归约
// 用途：FP8 量化时，每个量化分组（64 个元素）由 4 个 lane 各处理 16 个元素，
//       需要先求出这 4 个 lane 持有的绝对值最大值，以确定量化的缩放因子。
// 工作原理：通过 __shfl_xor_sync 在 lane 0-3、4-7、8-11、... 各组内
//           做 butterfly 归约，两步完成 4-lane 的 max 归约。
__device__ __forceinline__ float warp4MaxAbs(float val) {
  // Reduce absolute max across 4 consecutive lanes (lane id & 3 group).
  float peer = __shfl_xor_sync(FINAL_MASK, val, 1);
  val = fmaxf(val, peer);
  peer = __shfl_xor_sync(FINAL_MASK, val, 2);
  val = fmaxf(val, peer);
  return val;
}

// 中文注释：在整个 warp（32 个 lane）内做求和归约
// 用途：RMSNorm 计算时，需要求每个 head 的所有元素的平方和，
//       然后用 rsqrt(平方和/N + eps) 得到归一化因子。
// 工作原理：标准的 butterfly reduction，通过 __shfl_xor_sync 依次
//           在 mask=16,8,4,2,1 上做 pairwise 加法，5 步完成 32-lane 归约。
template <typename T>
__device__ __forceinline__ float warpSum(float val) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    val += __shfl_xor_sync(FINAL_MASK, val, mask, 32);
  }
  return val;
}

// ────────────────────────────────────────────────────────────────────────────
// Per-slot inner pipeline
// ────────────────────────────────────────────────────────────────────────────
// Shared by both kernel variants: 1 CTA per (token, head) pair vs. 1 CTA per
// token.  Templated on `kNumHeadsQPadded` so the KV-sentinel comparison and
// q_out stride fold to compile-time constants.
//
// Slot layout (per token):
//   slot < num_heads_q                          → live-Q   (RMSNorm + RoPE,
//                                                           read q_in →
//                                                           write q_out)
//   num_heads_q <= slot < kNumHeadsQPadded      → pad-Q    (zero-fill q_out;
//                                                           v0/v1 unused)
//   slot == kNumHeadsQPadded                    → KV       (RoPE + UE8M0 quant
//                                                           + paged-cache
//                                                           insert)
//
// 中文注释：每个 slot 的核心处理流水线（设备端内联函数）
// ============================================================================
// 本函数是融合 kernel 的核心计算逻辑，被两种 kernel 变体共同调用：
//   - 标准 kernel：每个 warp 处理一个 (token, slot) 对
//   - ReducedGrid kernel：每个 warp 处理一个 token 的多个 slot（循环迭代）
//
// 参数 kNumHeadsQPadded 是模板参数（编译期常量），使得 slot 类型判断和
// q_out 的 stride 计算在编译期即可折叠为常量操作，避免运行时分支开销。
//
// 每个 token 的 slot 分配逻辑：
//   slot < num_heads_q          -> live-Q：执行 RMSNorm + RoPE，从 q_in 读取，写入 q_out
//   num_heads_q <= slot < padded -> pad-Q：在 q_out 对应位置写零（FlashMLA 需要读取这些位置）
//   slot == padded               -> KV：执行 RoPE + UE8M0 FP8 量化 + 写入分页 KV cache
//
// 整体处理流程（每个 warp 处理一个 slot 时）：
//   步骤 1：判断 slot 类型（live-Q / pad-Q / KV）
//   步骤 2：如果是 pad-Q，直接写零并返回
//   步骤 3：将 bf16 输入解码为 16 个 fp32 寄存器（每个 lane 16 个元素）
//   步骤 4：如果是 live-Q，执行 RMSNorm（无权重版本）
//   步骤 5：如果当前 lane 负责 RoPE 维度区间，执行 GPT-J 风格的 RoPE
//   步骤 6：Q 分支 -> 将 fp32 结果转回 bf16 写入 q_out
//           KV 分支 -> 计算量化参数，对 NoPE 维度做 FP8 量化写入 cache，
//                     对 RoPE 维度保持 bf16 写入 cache
// ============================================================================
template <typename scalar_t_in, int kNumHeadsQPadded>
__device__ __forceinline__ void processDeepseekV4Slot(
    uint4 v0, uint4 v1, int const tokenIdx, int const slotIdx,
    int const dim_base, int const laneId, int const num_heads_q,
    float const eps, scalar_t_in* __restrict__ q_out,
    uint8_t* __restrict__ k_cache, int64_t const* __restrict__ slot_mapping,
    int64_t const* __restrict__ position_ids,
    float const* __restrict__ cos_sin_cache, int const cache_block_size,
    int const kv_block_stride) {
  // 中文注释：类型转换器，用于 bf16 <-> fp32 的相互转换
  using Converter = vllm::_typeConvert<scalar_t_in>;

  // 中文注释：根据 slot 索引判断当前 warp 要处理的是哪种类型的工作
  // - isKV == true：当前 slot 是 KV slot，需要处理 KV 的 RoPE + 量化 + cache 写入
  // - isPadQ == true：当前 slot 是填充 Q slot，只需写零
  // - 两者都为 false：当前 slot 是真实的 Q head，需要做 RMSNorm + RoPE
  bool const isKV = (slotIdx == kNumHeadsQPadded);
  bool const isPadQ = !isKV && (slotIdx >= num_heads_q);

  // ── Pad-Q branch: write 32 B of zeros and exit. ─────────────────────────
  // FlashMLA reads these slots; bf16 +0.0 is bit pattern 0x0000, so a uint4
  // zero literal is correct.  Matches the live-Q branch's vectorized store.
  // 中文注释：Pad-Q 分支 -- 当 Q head 数量不足 padded 数量时，多余的位置需要填零。
  // FlashMLA 内部会读取 q_out 的全部 padded 维度，因此这些位置必须有合法值。
  // bf16 的 +0.0 对应的位模式恰好是 0x0000，所以用 uint4 零值直接写入即可。
  // 每个 slot 写入 2 × 16 字节 = 32 字节，覆盖该 lane 负责的 16 个 bf16 元素。
  if (isPadQ) {
    scalar_t_in* dst =
        q_out +
        (static_cast<int64_t>(tokenIdx) * kNumHeadsQPadded + slotIdx) *
            kHeadDim +
        dim_base;
    uint4 const zero4 = {0u, 0u, 0u, 0u};
    *reinterpret_cast<uint4*>(dst) = zero4;
    *reinterpret_cast<uint4*>(dst + 8) = zero4;
    return;
  }

  // ── Decode the bf16 → 16 fp32 registers ─────────────────────────────
  // 中文注释：将 bf16 输入解码为 16 个 fp32 寄存器
  // v0 和 v1 各包含 8 个 bf16 值（uint4 = 16 字节 = 8 × bf16），共 16 个元素。
  // 使用 Converter 做 bf16 -> fp32 的精度提升，后续所有计算在 fp32 精度下进行。
  float elements[kElemsPerLane];
  {
    typename Converter::packed_hip_type const* p0 =
        reinterpret_cast<typename Converter::packed_hip_type const*>(&v0);
    typename Converter::packed_hip_type const* p1 =
        reinterpret_cast<typename Converter::packed_hip_type const*>(&v1);
#pragma unroll
    for (int i = 0; i < 4; i++) {
      float2 f2 = Converter::convert(p0[i]);
      elements[2 * i] = f2.x;
      elements[2 * i + 1] = f2.y;
    }
#pragma unroll
    for (int i = 0; i < 4; i++) {
      float2 f2 = Converter::convert(p1[i]);
      elements[8 + 2 * i] = f2.x;
      elements[8 + 2 * i + 1] = f2.y;
    }
  }

  // ── Q branch: RMSNorm (no weight) ───────────────────────────────────
  // 中文注释：Q 分支的 RMSNorm 计算（仅对 live-Q slot 执行）
  // RMSNorm（Root Mean Square Normalization）公式：
  //   RMSNorm(x) = x / sqrt(mean(x^2) + eps)
  // 这里是"无权重"版本，即不乘以可学习的 gamma 参数（直接用归一化后的值）。
  //
  // 计算步骤：
  //   1. 每个 lane 计算自己负责的 16 个元素的平方和
  //   2. 通过 warpSum 在整个 warp（32 个 lane）间归约，得到完整的 kHeadDim 个元素的平方和
  //   3. 计算 rms_rcp = rsqrt(平方和 / kHeadDim + eps)，即 1/RMS
  //   4. 将每个元素乘以 rms_rcp，完成归一化
  if (!isKV) {
    float sumOfSquares = 0.0f;
#pragma unroll
    for (int i = 0; i < kElemsPerLane; i++) {
      sumOfSquares += elements[i] * elements[i];
    }
    sumOfSquares = warpSum<float>(sumOfSquares);
    float const rms_rcp =
        rsqrtf(sumOfSquares / static_cast<float>(kHeadDim) + eps);
#pragma unroll
    for (int i = 0; i < kElemsPerLane; i++) {
      elements[i] = elements[i] * rms_rcp;
    }
  }

  // ── GPT-J RoPE on dims [NOPE_DIM, HEAD_DIM) ─────────────────────────────
  // All math in fp32.  cos_sin_cache is loaded as fp32 (its native storage).
  // 中文注释：GPT-J 风格的旋转位置编码（RoPE）
  // ========================================================================
  // RoPE 仅应用于后 kRopeDim=64 维（即 dim 范围 [448, 512)）。
  // 前 kNopeDim=448 维（NoPE 部分）不参与 RoPE，直接保持原值。
  //
  // GPT-J 风格的 RoPE 采用交错配对方式（而非 NeoX 风格的前后两半配对）：
  //   对于维度 [x_0, x_1, x_2, x_3, ...]，配对方式为 (x_0,x_1), (x_2,x_3), ...
  //   旋转公式：x_even' = x_even * cos - x_odd * sin
  //             x_odd'  = x_even * sin + x_odd * cos
  //
  // cos_sin_cache 的布局：[max_pos, rope_dim]，其中 rope_dim=64
  //   前 32 个 float 是 cos 值，后 32 个 float 是 sin 值
  //   每个位置 position_ids[tokenIdx] 对应一组 cos/sin 值
  //
  // 只有 dim_base >= kNopeDim 的 lane 才处理 RoPE 维度，其他 lane 跳过此步骤。
  bool const is_rope_lane = dim_base >= kNopeDim;
  if (is_rope_lane) {
    int64_t const pos = position_ids[tokenIdx];
    constexpr int kHalfRope = kRopeDim / 2;
    float const* cos_ptr = cos_sin_cache + pos * kRopeDim;
    float const* sin_ptr = cos_ptr + kHalfRope;

    int const rope_local_base = dim_base - kNopeDim;
    int const half_base = rope_local_base >> 1;

    // Load phase: 4 vectorized LDGs issue back-to-back.
    // 中文注释：向量化加载 cos/sin 值
    // 每个 lane 负责 16 个元素，即 8 对 (even, odd)，需要 8 个 cos 和 8 个 sin。
    // 使用 4 次 float4（每次 4 个 float）的向量化加载，充分利用内存带宽。
    float4 const c0 = *reinterpret_cast<float4 const*>(cos_ptr + half_base);
    float4 const c1 = *reinterpret_cast<float4 const*>(cos_ptr + half_base + 4);
    float4 const s0 = *reinterpret_cast<float4 const*>(sin_ptr + half_base);
    float4 const s1 = *reinterpret_cast<float4 const*>(sin_ptr + half_base + 4);
    float const cos_arr[8] = {c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w};
    float const sin_arr[8] = {s0.x, s0.y, s0.z, s0.w, s1.x, s1.y, s1.z, s1.w};

#pragma unroll
    // 中文注释：执行 GPT-J 风格的 2D 旋转
    // 对 8 对 (even, odd) 元素分别应用旋转矩阵：
    //   [x_even']   [cos  -sin] [x_even]
    //   [x_odd' ] = [sin   cos] [x_odd ]
    for (int p = 0; p < kElemsPerLane / 2; p++) {
      float const x_even = elements[2 * p];
      float const x_odd = elements[2 * p + 1];
      elements[2 * p] = x_even * cos_arr[p] - x_odd * sin_arr[p];
      elements[2 * p + 1] = x_even * sin_arr[p] + x_odd * cos_arr[p];
    }
  }

  // ═══════════════════════════════════════════════════════════════════
  // Q / KV branch dispatch. Restructured as if/else (no early `return`)
  // so every code path lands at the same exit point — callers own PDL
  // triggering and per-iteration buffer rotation.
  // ═══════════════════════════════════════════════════════════════════
  // 中文注释：Q 与 KV 的分支调度
  // 在完成 RMSNorm（仅 Q）和 RoPE（Q 和 KV 共用）之后，Q 和 KV 的后续处理路径分离：
  //   - Q 分支：将 fp32 结果转回 bf16，写入 q_out 的对应位置
  //   - KV 分支：对 NoPE 维度做 FP8 量化，对 RoPE 维度保持 bf16，写入分页 KV cache
  //
  // 注意：这里用 if/else 而非 early return，确保所有代码路径在同一退出点结束，
  // 便于调用方统一管理 PDL（Programmatic Dependent Launch）触发和 buffer 轮换。
  if (!isKV) {
    // ── Live-Q: cast back to bf16 and store into the padded q_out. ─────
    // 中文注释：Live-Q 分支 -- 将 fp32 结果转回 bf16 并写入 q_out
    // q_out 的形状为 [N, kNumHeadsQPadded, 512]，是一个 padded 的输出张量。
    // 每个 lane 将自己负责的 16 个 fp32 值转换为 bf16，然后通过两次 uint4
    // 向量化写入（每次 8 个 bf16 = 16 字节）存入 q_out 的对应位置。
    uint4 out0, out1;
    typename Converter::packed_hip_type* po0 =
        reinterpret_cast<typename Converter::packed_hip_type*>(&out0);
    typename Converter::packed_hip_type* po1 =
        reinterpret_cast<typename Converter::packed_hip_type*>(&out1);
#pragma unroll
    for (int i = 0; i < 4; i++) {
      po0[i] =
          Converter::convert(make_float2(elements[2 * i], elements[2 * i + 1]));
    }
#pragma unroll
    for (int i = 0; i < 4; i++) {
      po1[i] = Converter::convert(
          make_float2(elements[8 + 2 * i], elements[8 + 2 * i + 1]));
    }
    scalar_t_in* dst =
        q_out +
        (static_cast<int64_t>(tokenIdx) * kNumHeadsQPadded + slotIdx) *
            kHeadDim +
        dim_base;
    *reinterpret_cast<uint4*>(dst) = out0;
    *reinterpret_cast<uint4*>(dst + 8) = out1;
  } else {
    // ── KV: FP8 quant on NoPE + bf16 store on RoPE + cache insert.
    // 中文注释：KV 分支 -- RoPE 后处理 + UE8M0 FP8 量化 + 分页 cache 写入
    // ========================================================================
    // 这是本 kernel 最复杂的分支，完成以下工作：
    //
    // 步骤 1：通过 slot_mapping 获取当前 token 在分页 KV cache 中的物理位置
    //   slot_id -> block_idx（哪个 block）+ pos_in_block（block 内的第几个 token）
    //
    // 步骤 2：将 fp32 元素转回 bf16 再转回 fp32（截断精度，模拟 bf16 的舍入行为），
    //         确保量化前的值与实际存储后重新加载的值一致
    //
    // 步骤 3：计算 UE8M0 FP8 量化的缩放因子
    //   - 每 64 个元素（kQuantBlock）为一组，共享一个 scale
    //   - scale 的计算：absmax = 组内绝对值最大值
    //                   exponent = ceil(log2(absmax / 448.0))
    //                   inv_scale = 2^(-exponent)
    //   - UE8M0 格式的 scale 仅存储指数部分（8 位无符号整数 = 127 + 真实指数）
    //
    // 步骤 4：根据维度类型分两条路径写入 KV cache：
    //   - NoPE 维度（前 448 维）：量化为 FP8 E4M3，每元素 1 字节，存入 token 数据区前半部分
    //   - RoPE 维度（后 64 维）：保持 bf16，每元素 2 字节，存入 token 数据区后半部分
    //   - scale 值存入 block 末尾的 scale 区域
    //
    // 分页 KV cache 的 block 内存布局：
    //   [0, block_size * 576)                          : token 数据区
    //     每个 token 占 576 字节 = 448(fp8 NoPE) + 128(bf16 RoPE)
    //   [block_size * 576, block_size * 576 + block_size * 8) : scale 区
    //     每个 token 占 8 字节 = 7 个真实 scale + 1 字节 padding
    // ========================================================================
    int64_t const slot_id = slot_mapping[tokenIdx];
    if (slot_id >= 0) {
      // 中文注释：计算当前 token 在分页 KV cache 中的物理地址
      // slot_id 是全局的线性 slot 编号，通过除法和取模得到：
      //   block_idx = slot_id / cache_block_size  -> 第几个 block
      //   pos_in_block = slot_id % cache_block_size -> block 内的第几个 token
      int64_t const block_idx = slot_id / cache_block_size;
      int64_t const pos_in_block = slot_id % cache_block_size;
      // 中文注释：block_base 是当前 block 在 k_cache 中的起始地址
      // k_cache 的形状为 [num_blocks, kv_block_stride]，
      // kv_block_stride 是每个 block 的字节数（数据区 + scale 区）
      uint8_t* block_base =
          k_cache + block_idx * static_cast<int64_t>(kv_block_stride);
      // 中文注释：token_fp8_ptr 指向当前 token 数据区的起始位置
      // 该位置起的前 448 字节存放 NoPE 部分的 FP8 值
      uint8_t* token_fp8_ptr = block_base + pos_in_block * kTokenDataBytes;
      // 中文注释：token_bf16_ptr 指向当前 token 数据区中 RoPE 部分的起始位置
      // 从 FP8 区之后开始，存放 64 个 bf16 值（128 字节）
      uint8_t* token_bf16_ptr = token_fp8_ptr + kNopeDim;
      // 中文注释：token_scale_ptr 指向当前 token 的 UE8M0 scale 存储位置
      // scale 区在 block 数据区之后，每个 token 占 8 字节（7 个真实 scale + 1 padding）
      uint8_t* token_scale_ptr =
          block_base +
          static_cast<int64_t>(cache_block_size) * kTokenDataBytes +
          pos_in_block * kScaleBytesPerToken;

#pragma unroll
      // 中文注释：bf16 精度截断（round-trip）
      // 将 fp32 -> bf16 -> fp32，模拟 bf16 存储后再加载的舍入行为。
      // 这一步确保后续 FP8 量化的输入值与实际 KV cache 中存储后读出的值一致，
      // 避免量化误差与 bf16 舍入误差的叠加。
      for (int i = 0; i < kElemsPerLane; i++) {
        elements[i] = Converter::convert(Converter::convert(elements[i]));
      }

      // 中文注释：计算 UE8M0 FP8 量化的缩放因子
      // 每个量化分组（64 个元素）由 4 个 lane 各处理 16 个元素。
      //
      // 步骤 1：每个 lane 计算自己 16 个元素的局部绝对值最大值
      float local_absmax = 0.0f;
#pragma unroll
      for (int i = 0; i < kElemsPerLane; i++) {
        local_absmax = fmaxf(local_absmax, fabsf(elements[i]));
      }
      // 步骤 2：在 4 个 lane 间归约，得到分组的全局绝对值最大值
      // 下界 1e-4f 防止除零（当所有元素为零时）
      float const absmax = fmaxf(warp4MaxAbs(local_absmax), 1e-4f);
      // 步骤 3：计算 UE8M0 格式的指数
      // UE8M0 是一种"microscaling"浮点格式，scale 仅存储指数部分。
      // exponent = ceil(log2(absmax / 448.0))，确保缩放后的值不超过 FP8 E4M3 的范围。
      float const exponent = ceilf(log2f(absmax / kFp8Max));
      // 步骤 4：计算反向缩放因子 inv_scale = 2^(-exponent)
      // 用于将原始值乘以 inv_scale 映射到 FP8 的可表示范围内
      float const inv_scale = exp2f(-exponent);

      if (!is_rope_lane) {
        // 中文注释：NoPE 维度的 FP8 量化和写入
        // 当前 lane 负责的 16 个元素属于 NoPE 部分（前 448 维），需要量化为 FP8 E4M3。
        //
        // 量化流程：
        //   1. 每个元素乘以 inv_scale 缩放到 FP8 可表示范围
        //   2. 钳位（clamp）到 [-448.0, 448.0]，使用饱和模式（SATFINITE）
        //   3. 转换为 FP8 E4M3 格式的 uint8 存储
        uint8_t out_bytes[kElemsPerLane];
#pragma unroll
        for (int i = 0; i < kElemsPerLane; i++) {
          float scaled = elements[i] * inv_scale;
          scaled = fminf(fmaxf(scaled, -kFp8Max), kFp8Max);
#ifndef USE_ROCM
          __nv_fp8_storage_t s =
              __nv_cvt_float_to_fp8(scaled, __NV_SATFINITE, __NV_E4M3);
          out_bytes[i] = static_cast<uint8_t>(s);
#else
          out_bytes[i] = rocm_cvt_float_to_fp8_e4m3(scaled);
#endif
        }
        // 中文注释：将量化后的 16 字节（16 个 FP8 值）向量化写入 KV cache 的 FP8 数据区
        *reinterpret_cast<uint4*>(token_fp8_ptr + dim_base) =
            *reinterpret_cast<uint4 const*>(out_bytes);

        // 中文注释：写入 UE8M0 scale 值
        // 每 4 个 lane 共享一个 scale（对应一个 64 元素的量化分组）。
        // 只有每组的第 0 个 lane（laneId & 3 == 0）负责写入 scale。
        // scale 存储格式：uint8 = exponent + 127.0（即 IEEE 754 单精度浮点的指数偏移编码）。
        // 每个 token 共 7 个真实 scale（对应 7 个量化分组）+ 1 个 padding 字节。
        if ((laneId & 3) == 0) {
          int const q_block_idx = laneId >> 2;
          float encoded = fmaxf(fminf(exponent + 127.0f, 255.0f), 0.0f);
          token_scale_ptr[q_block_idx] = static_cast<uint8_t>(encoded);
        }
        // 中文注释：lane 0 负责写入 scale 区的最后一个 padding 字节（设为 0）
        if (laneId == 0) {
          token_scale_ptr[kNumQuantBlocks] = 0;
        }
      } else {
        // 中文注释：RoPE 维度的 bf16 存储
        // RoPE 部分（后 64 维）不进行 FP8 量化，保持 bf16 精度直接存入 KV cache。
        // 这是因为 RoPE 编码后的维度对精度更敏感，且数据量较小（64 维 × 2 字节 = 128 字节），
        // 使用 bf16 存储可以在精度和存储效率之间取得较好的平衡。
        uint4 out0, out1;
        typename Converter::packed_hip_type* po0 =
            reinterpret_cast<typename Converter::packed_hip_type*>(&out0);
        typename Converter::packed_hip_type* po1 =
            reinterpret_cast<typename Converter::packed_hip_type*>(&out1);
#pragma unroll
        for (int i = 0; i < 4; i++) {
          po0[i] = Converter::convert(
              make_float2(elements[2 * i], elements[2 * i + 1]));
        }
#pragma unroll
        for (int i = 0; i < 4; i++) {
          po1[i] = Converter::convert(
              make_float2(elements[8 + 2 * i], elements[8 + 2 * i + 1]));
        }
        // 中文注释：计算 RoPE 部分在 token 数据区中的偏移并写入
        // rope_local_base = dim_base - kNopeDim，即 RoPE 维度内的局部偏移
        // bf16_dst 指向 token_bf16_ptr（NoPE FP8 区之后）的对应位置
        int const rope_local_base = dim_base - kNopeDim;
        scalar_t_in* bf16_dst =
            reinterpret_cast<scalar_t_in*>(token_bf16_ptr) + rope_local_base;
        *reinterpret_cast<uint4*>(bf16_dst) = out0;
        *reinterpret_cast<uint4*>(bf16_dst + 8) = out1;
      }
    }
  }
}

// ────────────────────────────────────────────────────────────────────────────
// Kernel
// ────────────────────────────────────────────────────────────────────────────
//
// Grid: 1D, gridDim.x = ceil(num_tokens_full * (kNumHeadsQPadded + 1) /
// warps_per_block) Block: blockDim.x = 256 threads (8 warps per block) Each
// warp handles one (token, head_slot) pair.
//   slot < num_heads_q                              → live-Q branch
//                                                     (RMSNorm + RoPE,
//                                                      read q_in → write q_out)
//   num_heads_q <= slot < kNumHeadsQPadded          → pad-Q branch
//                                                     (zero-fill q_out)
//   slot == kNumHeadsQPadded                        → KV branch
//                                                     (RoPE + UE8M0 quant +
//                                                      paged-cache insert)
//
// `kNumHeadsQPadded` is a template parameter (compile-time constant) so the
// divisions in the grid math and the KV-sentinel comparison fold to fast
// constant operations.  The launch wrapper dispatches the runtime value to
// the matching instantiation.
//
// With DP padding, q/kv/position_ids can have more rows than slot_mapping.
// The live-Q and pad-Q branches cover all `num_tokens_full` rows (downstream
// attention uses them).  The KV branch only inserts the first
// `num_tokens_insert` tokens (= slot_mapping length) into the paged cache.
//
// 中文注释：标准融合 kernel（每个 warp 处理一个 (token, slot) 对）
// ============================================================================
// 网格配置：
//   - Grid：1D，gridDim.x = ceil(num_tokens_full * (padded + 1) / warps_per_block)
//   - Block：256 线程 = 8 个 warp
//   - 每个 warp 处理一个 (token, slot) 对
//
// 线程映射逻辑：
//   globalWarpIdx = blockIdx.x * warpsPerBlock + warpId
//   tokenIdx = globalWarpIdx / kTotalSlotsPerToken  -> 当前处理第几个 token
//   slotIdx  = globalWarpIdx % kTotalSlotsPerToken  -> 当前处理该 token 的第几个 slot
//
// slot 的含义（与 processDeepseekV4Slot 中的定义一致）：
//   slot < num_heads_q          -> live-Q（RMSNorm + RoPE）
//   num_heads_q <= slot < padded -> pad-Q（写零）
//   slot == padded               -> KV（RoPE + FP8 量化 + cache 写入）
//
// kNumHeadsQPadded 作为模板参数，使得 grid 数学运算和 KV sentinel 比较
// 在编译期即可折叠为常量操作，避免运行时除法开销。
//
// DP padding 说明：
//   当使用数据并行（DP）padding 时，q/kv/position_ids 的行数可能大于 slot_mapping 的长度。
//   live-Q 和 pad-Q 分支处理所有 num_tokens_full 行（下游 attention 需要使用这些值），
//   而 KV 分支只处理前 num_tokens_insert 行（= slot_mapping 的长度），
//   因为 DP-padded 的 token 没有对应的 KV cache slot。
//
template <typename scalar_t_in, int kNumHeadsQPadded>
__global__ void fusedDeepseekV4QNormRopeKVRopeQuantInsertKernel(
    scalar_t_in const* __restrict__ q_in,      // [N, num_heads_q,      512]
    scalar_t_in* __restrict__ q_out,           // [N, kNumHeadsQPadded, 512]
    scalar_t_in const* __restrict__ kv_in,     // [N, 512] bf16
    uint8_t* __restrict__ k_cache,             // [num_blocks, block_stride]
    int64_t const* __restrict__ slot_mapping,  // [num_tokens_insert] i64
    int64_t const* __restrict__ position_ids,  // [N] i64
    float const* __restrict__ cos_sin_cache,   // [max_pos, 64] fp32
    float const eps,
    int const num_tokens_full,    // = q.size(0) = kv.size(0)
    int const num_tokens_insert,  // = slot_mapping.size(0), ≤ num_tokens_full
    int const num_heads_q,        // live Q heads (input layout)
    int const cache_block_size,   // tokens per paged-cache block
    int const kv_block_stride) {  // bytes per paged-cache block
#if (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 800) && !defined(USE_ROCM)
  // BF16 _typeConvert specialization is unavailable on pre-Ampere.  The
  // DeepseekV4 kernel only runs with bf16 inputs in practice, so compile a
  // no-op stub for sm_70/sm_75 to keep multi-arch builds happy.
  if constexpr (std::is_same_v<scalar_t_in, c10::BFloat16>) {
    return;
  } else {
#endif
    // 中文注释：计算当前 warp 的全局索引，并从中解码出 token 和 slot 编号
    int const warpsPerBlock = blockDim.x / 32;
    int const warpId = threadIdx.x / 32;
    int const laneId = threadIdx.x % 32;
    int const globalWarpIdx = blockIdx.x * warpsPerBlock + warpId;

    // 中文注释：每个 token 对应 (kNumHeadsQPadded + 1) 个 slot
    // 通过整数除法和取模将线性的 globalWarpIdx 映射到 (tokenIdx, slotIdx) 二维坐标
    constexpr int kTotalSlotsPerToken = kNumHeadsQPadded + 1;
    int const tokenIdx = globalWarpIdx / kTotalSlotsPerToken;
    int const slotIdx = globalWarpIdx % kTotalSlotsPerToken;
    if (tokenIdx >= num_tokens_full) return;

    // 中文注释：判断当前 slot 的类型
    bool const isKV = (slotIdx == kNumHeadsQPadded);
    bool const isPadQ = !isKV && (slotIdx >= num_heads_q);
    // KV branch: skip DP-padded tokens (no slot reserved for them).
    // 中文注释：KV 分支跳过 DP-padded 的 token（这些 token 没有对应的 slot_mapping 条目）
    if (isKV && tokenIdx >= num_tokens_insert) return;

    // PDL: wait for predecessor kernel (upstream q/kv producer) to signal
    // before touching any global memory.  No-op when PDL is not enabled on
    // the launch.  The CUDA runtime wrapper emits the griddepcontrol.wait
    // PTX with the required memory clobber internally.
    // 中文注释：PDL（Programmatic Dependent Launch）同步点
    // 在 SM90+（Hopper 及以上）硬件上，等待上游 kernel（Q/KV 的生产者）完成信号，
    // 确保在访问全局内存之前上游数据已就绪。在非 SM90 硬件上此调用为空操作。
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    cudaGridDependencySynchronize();
#endif

    // Dim range this lane owns within the 512-wide head.
    // 中文注释：计算当前 lane 在 512 维 head 中负责的维度起始位置
    // 每个 lane 负责连续的 16 个元素，dim_base 的范围为 [0, 16, 32, ..., 496]
    int const dim_base = laneId * kElemsPerLane;  // in [0, 512) step 16

    // Load only for live-Q and KV slots; pad-Q skips the read (q_in beyond
    // num_heads_q is out of bounds) and the helper zero-fills its output.
    // 中文注释：从全局内存加载当前 lane 负责的 16 个 bf16 元素（32 字节）
    // pad-Q slot 跳过加载（因为 q_in 在 num_heads_q 之外是越界的），
    // processDeepseekV4Slot 会直接对 pad-Q 写零。
    // live-Q 从 q_in 加载，KV 从 kv_in 加载，通过 isKV 区分数据源。
    uint4 v0, v1;
    if (!isPadQ) {
      // 中文注释：根据 slot 类型选择数据源并加载
      scalar_t_in const* src_ptr;
      if (isKV) {
        // 中文注释：KV slot -- 从 kv_in 加载当前 token 的 512 维 KV 数据
        // kv_in 形状为 [N, 512]，每个 token 占一行
        src_ptr = kv_in + static_cast<int64_t>(tokenIdx) * kHeadDim + dim_base;
      } else {
        // 中文注释：live-Q slot -- 从 q_in 加载当前 token 的第 slotIdx 个 Q head
        // q_in 形状为 [N, num_heads_q, 512]，行优先布局
        int64_t const q_row_offset =
            (static_cast<int64_t>(tokenIdx) * num_heads_q + slotIdx) *
                kHeadDim +
            dim_base;
        src_ptr = q_in + q_row_offset;
      }
      // 中文注释：向量化加载 32 字节（2 × uint4 = 2 × 16 字节 = 16 个 bf16 值）
      v0 = *reinterpret_cast<uint4 const*>(src_ptr);
      v1 = *reinterpret_cast<uint4 const*>(src_ptr + 8);
    }

    // 中文注释：调用核心处理函数，完成 RMSNorm/RoPE/量化/cache 写入
    processDeepseekV4Slot<scalar_t_in, kNumHeadsQPadded>(
        v0, v1, tokenIdx, slotIdx, dim_base, laneId, num_heads_q, eps, q_out,
        k_cache, slot_mapping, position_ids, cos_sin_cache, cache_block_size,
        kv_block_stride);

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    // 中文注释：PDL 完成触发 -- 通知下游 kernel 本 kernel 的工作已完成
    cudaTriggerProgrammaticLaunchCompletion();
#endif
#if (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 800) && !defined(USE_ROCM)
  }
#endif
}

// ────────────────────────────────────────────────────────────────────────────
// Kernel
// ────────────────────────────────────────────────────────────────────────────
//
// Grid: 1D, gridDim.x = num_tokens_full
// Block: blockDim.x = 256 threads (8 warps per block) Each
// warp handles one token, iterating over each head.
// Q branch (RMSNorm + RoPE, in place) head_slot == num_heads_q
// KV branch (RoPE + UE8M0 quant + insert)
//
// 中文注释：ReducedGrid 变体 kernel（每个 warp 处理一个 token 的多个 slot）
// ============================================================================
// 与标准 kernel 的区别：
//   - 标准 kernel：gridDim = ceil(total_warps / warps_per_block)，
//     每个 warp 处理一个 (token, slot) 对。当 token 数较少时，grid 较小，
//     但每个 warp 只处理一个 slot，GPU 利用率可能不足。
//   - ReducedGrid kernel：gridDim = num_tokens_full，每个 block 处理一个 token，
//     block 内的 8 个 warp 通过循环迭代处理该 token 的所有 slot。
//     当 token 数较多时，这避免了 grid 过大导致的调度开销。
//
// 选择策略（在 launch wrapper 中）：
//   当 num_tokens_full < 1024 时使用标准 kernel（grid 较小，需要更多并行度），
//   否则使用 ReducedGrid kernel（token 足够多，每个 block 内循环更高效）。
//
// 流水线优化：
//   ReducedGrid kernel 内部使用了 double-buffering（双缓冲）策略：
//   在处理当前 slot 的同时，预取下一个 slot 的数据，隐藏内存延迟。
//   这通过 load_slot lambda 和 buffer rotation 实现。
//
template <typename scalar_t_in, int kNumHeadsQPadded>
__global__ void fusedDeepseekV4QNormRopeKVRopeQuantInsertKernelReducedGrid(
    scalar_t_in const* __restrict__ q_in, scalar_t_in* __restrict__ q_out,
    scalar_t_in const* __restrict__ kv_in, uint8_t* __restrict__ k_cache,
    int64_t const* __restrict__ slot_mapping,
    int64_t const* __restrict__ position_ids,
    float const* __restrict__ cos_sin_cache, float const eps,
    int const num_tokens_full, int const num_tokens_insert,
    int const num_heads_q, int const cache_block_size,
    int const kv_block_stride) {
#if (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 800) && !defined(USE_ROCM)
  if constexpr (std::is_same_v<scalar_t_in, c10::BFloat16>) {
    return;
  } else {
#endif
    int const warpsPerBlock = blockDim.x / 32;
    int const warpId = threadIdx.x / 32;
    int const laneId = threadIdx.x % 32;

    // 中文注释：ReducedGrid 的线程映射 -- 每个 block 处理一个 token
    int const tokenIdx = blockIdx.x;
    if (tokenIdx >= num_tokens_full) return;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    cudaGridDependencySynchronize();
#endif

    int const dim_base = laneId * kElemsPerLane;  // in [0, 512) step 16
    // Slot enumeration: live-Q + pad-Q + (KV if this token has a slot).
    // 中文注释：计算当前 token 需要处理的 slot 总数
    // - 如果 tokenIdx >= num_tokens_insert（DP-padded token），不处理 KV slot
    // - 否则，处理所有 live-Q + pad-Q + KV = kNumHeadsQPadded + 1 个 slot
    int const slot_end = (tokenIdx >= num_tokens_insert)
                             ? kNumHeadsQPadded
                             : (kNumHeadsQPadded + 1);

    // 中文注释：数据加载 lambda 函数
    // 根据 slot 编号 s 从对应的输入张量加载 16 个 bf16 元素（32 字节）。
    // pad-Q slot 跳过加载（q_in 在 num_heads_q 之外越界），
    // live-Q 从 q_in 加载，KV 从 kv_in 加载。
    auto load_slot = [&](int s, uint4& va, uint4& vb) {
      // pad-Q slots skip the load — q_in beyond num_heads_q is OOB.
      if (s >= num_heads_q && s < kNumHeadsQPadded) return;
      scalar_t_in const* src;
      if (s == kNumHeadsQPadded) {
        src = kv_in + static_cast<int64_t>(tokenIdx) * kHeadDim + dim_base;
      } else {
        src = q_in +
              (static_cast<int64_t>(tokenIdx) * num_heads_q +
               static_cast<int64_t>(s)) *
                  kHeadDim +
              dim_base;
      }
      va = *reinterpret_cast<uint4 const*>(src);
      vb = *reinterpret_cast<uint4 const*>(src + 8);
    };

    // 中文注释：主循环 -- 每个 warp 从自己负责的起始 slot 开始，
    // 以 warpsPerBlock 为步长迭代处理多个 slot。
    // 例如 block 内有 8 个 warp（warpsPerBlock=8），slot_end=65（64个Q head + 1个KV）：
    //   warp 0 处理 slot: 0, 8, 16, 24, 32, 40, 48, 56, 64
    //   warp 1 处理 slot: 1, 9, 17, 25, 33, 41, 49, 57
    //   ...以此类推
    if (warpId < slot_end) {
      int curr_slot = warpId;
      uint4 v0_curr, v1_curr;
      // 中文注释：加载第一个 slot 的数据
      load_slot(curr_slot, v0_curr, v1_curr);

      while (curr_slot < slot_end) {
        int const next_slot = curr_slot + warpsPerBlock;
        bool const has_next = (next_slot < slot_end);

        // Prefetch src for the next slot
        // 中文注释：预取下一个 slot 的数据（double-buffering 流水线）
        // 在处理当前 slot 的同时，提前发起下一个 slot 的全局内存加载，
        // 以隐藏内存访问延迟。
        uint4 v0_next, v1_next;
        if (has_next) {
          load_slot(next_slot, v0_next, v1_next);
        }

        // 中文注释：处理当前 slot（RMSNorm/RoPE/量化/cache 写入）
        processDeepseekV4Slot<scalar_t_in, kNumHeadsQPadded>(
            v0_curr, v1_curr, tokenIdx, curr_slot, dim_base, laneId,
            num_heads_q, eps, q_out, k_cache, slot_mapping, position_ids,
            cos_sin_cache, cache_block_size, kv_block_stride);

        // ── Buffer rotation: hand the prefetched LDGs to the next iter.
        // 中文注释：buffer 轮换 -- 将预取的数据传递给下一次迭代
        v0_curr = v0_next;
        v1_curr = v1_next;
        curr_slot = next_slot;
      }  // while
    }  // if (warpId < slot_end)

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    cudaTriggerProgrammaticLaunchCompletion();
#endif
#if (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 800) && !defined(USE_ROCM)
  }
#endif
}

// ────────────────────────────────────────────────────────────────────────────
// Launch wrapper
// ────────────────────────────────────────────────────────────────────────────
// 中文注释：kernel 启动包装函数（模板实例化层）
// ============================================================================
// 本函数负责根据运行时参数选择合适的 kernel 变体并启动：
//   1. 计算 grid 大小（标准 kernel 需要的 warp 总数）
//   2. 配置 PDL（Programmatic Dependent Launch）属性（SM90+）
//   3. 根据 token 数量选择标准 kernel 或 ReducedGrid kernel
//   4. 通过 cudaLaunchKernelEx 启动 kernel（CUDA）或标准 <<<>>> 语法（ROCm）
//
// 选择逻辑：
//   - num_tokens_full < 1024：使用标准 kernel（每个 warp 处理一个 (token, slot) 对）
//     此时 token 较少，需要更多的并行 warp 来填满 GPU
//   - num_tokens_full >= 1024：使用 ReducedGrid kernel（每个 block 处理一个 token）
//     此时 token 足够多，block 内循环迭代更高效
// ============================================================================
template <typename scalar_t_in, int kNumHeadsQPadded>
static void launchFusedDeepseekV4Templated(
    scalar_t_in const* q_in, scalar_t_in* q_out, scalar_t_in const* kv_in,
    uint8_t* k_cache, int64_t const* slot_mapping, int64_t const* position_ids,
    float const* cos_sin_cache, float const eps, int const num_tokens_full,
    int const num_tokens_insert, int const num_heads_q,
    int const cache_block_size, int const kv_block_stride,
    cudaStream_t stream) {
  constexpr int kBlockSize = 256;
  constexpr int kWarpsPerBlock = kBlockSize / 32;
  // 中文注释：计算标准 kernel 所需的 warp 总数
  // 每个 token 需要 (kNumHeadsQPadded + 1) 个 warp（Q heads + pad + KV）
  int64_t const total_warps =
      static_cast<int64_t>(num_tokens_full) * (kNumHeadsQPadded + 1);
  int const grid =
      static_cast<int>((total_warps + kWarpsPerBlock - 1) / kWarpsPerBlock);

  // PDL: enable programmatic stream serialization whenever the hardware
  // supports it (SM90+).  On pre-Hopper GPUs the attribute is unavailable,
  // so leave numAttrs = 0 and launch as a regular kernel.
#ifndef USE_ROCM
  static int const sm_version = getSMVersion();
  // Host-side guard: the device kernel body is compiled as a no-op for
  // bf16 on pre-Ampere (sm_70/sm_75) because _typeConvert<BFloat16> is
  // unavailable there.  Refuse the launch loudly instead of silently
  // skipping the work.
  // 中文注释：硬件版本检查 -- 要求 Ampere（sm_80）或更新的 GPU
  TORCH_CHECK(
      sm_version >= 80,
      "fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert requires sm_80+ "
      "(Ampere or newer); got sm_",
      sm_version);
  // 中文注释：配置 CUDA kernel 启动参数
  cudaLaunchConfig_t config;
  config.gridDim = dim3(grid);
  config.blockDim = dim3(kBlockSize);
  config.dynamicSmemBytes = 0;
  config.stream = stream;
  // 中文注释：配置 PDL（Programmatic Dependent Launch）属性
  // cudaLaunchAttributeProgrammaticStreamSerialization 允许本 kernel 与上游 kernel
  // 通过程序化方式（而非硬件事件）进行流序列化，减少 kernel 间的同步开销。
  // 仅在 SM90+（Hopper）上启用，其他架构 numAttrs=0 作为普通 kernel 启动。
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = 1;
  config.attrs = attrs;
  config.numAttrs = (sm_version >= 90) ? 1 : 0;

  // 中文注释：根据 token 数量选择 kernel 变体
  if (num_tokens_full < NUM_TOKEN_CUTOFF) {
    // 中文注释：token 数量较少时，使用标准 kernel（更多并行 warp）
    cudaLaunchKernelEx(
        &config,
        fusedDeepseekV4QNormRopeKVRopeQuantInsertKernel<scalar_t_in,
                                                        kNumHeadsQPadded>,
        q_in, q_out, kv_in, k_cache, slot_mapping, position_ids, cos_sin_cache,
        eps, num_tokens_full, num_tokens_insert, num_heads_q, cache_block_size,
        kv_block_stride);
  } else {
    // 中文注释：token 数量较多时，使用 ReducedGrid kernel（每个 block 处理一个 token）
    config.gridDim = dim3(num_tokens_full);
    cudaLaunchKernelEx(
        &config,
        fusedDeepseekV4QNormRopeKVRopeQuantInsertKernelReducedGrid<
            scalar_t_in, kNumHeadsQPadded>,
        q_in, q_out, kv_in, k_cache, slot_mapping, position_ids, cos_sin_cache,
        eps, num_tokens_full, num_tokens_insert, num_heads_q, cache_block_size,
        kv_block_stride);
  }
#else
  // ROCm: use standard kernel launch syntax (no PDL/stream serialization)
  // clang-format off
  // 中文注释：ROCm 平台使用标准 <<<>>> 语法启动 kernel（不支持 PDL）
  fusedDeepseekV4QNormRopeKVRopeQuantInsertKernel<scalar_t_in, kNumHeadsQPadded>
      <<<grid, kBlockSize, 0, stream>>>(
          q_in, q_out, kv_in, k_cache, slot_mapping, position_ids,
          cos_sin_cache, eps, num_tokens_full, num_tokens_insert, num_heads_q,
          cache_block_size, kv_block_stride);
#endif
}

// Runtime dispatch into one of the precompiled `kNumHeadsQPadded`
// instantiations.  Supported padded head counts: 8, 16, 32, 64, 128.
// 中文注释：运行时分发函数 -- 将 num_heads_q_padded 的运行时值分发到预编译的模板实例
// ============================================================================
// 由于 kNumHeadsQPadded 是模板参数（编译期常量），无法直接用运行时值实例化。
// 因此通过 switch-case 宏展开，将常见的 padded head 数量（8, 16, 32, 64, 128）
// 预编译为对应的模板实例，运行时根据实际值选择匹配的实例。
//
// 这种"模板分发"模式是 CUDA 编程中的常见优化技巧：
//   - 避免运行时分支：模板参数在编译期确定，编译器可以做更多优化
//   - 减少寄存器压力：编译器可以精确计算每个实例所需的寄存器数量
//   - 代价是编译时间和二进制大小增加（每个实例生成独立的 kernel 代码）
// ============================================================================
template <typename scalar_t_in>
void launchFusedDeepseekV4QNormRopeKVRopeQuantInsert(
    scalar_t_in const* q_in, scalar_t_in* q_out, scalar_t_in const* kv_in,
    uint8_t* k_cache, int64_t const* slot_mapping,
    int64_t const* position_ids, float const* cos_sin_cache, float const eps,
    int const num_tokens_full, int const num_tokens_insert,
    int const num_heads_q, int const num_heads_q_padded,
    int const cache_block_size, int const kv_block_stride,
    cudaStream_t stream) {
#define DISPATCH(N)                                                         \
  case N:                                                                   \
    launchFusedDeepseekV4Templated<scalar_t_in, N>(                         \
        q_in, q_out, kv_in, k_cache, slot_mapping, position_ids,            \
        cos_sin_cache, eps, num_tokens_full, num_tokens_insert, num_heads_q, \
        cache_block_size, kv_block_stride, stream);                         \
    return;

  switch (num_heads_q_padded) {
    DISPATCH(8)
    DISPATCH(16)
    DISPATCH(32)
    DISPATCH(64)
    DISPATCH(128)
    default:
      TORCH_CHECK(false,
                  "fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert: "
                  "unsupported num_heads_q_padded=",
                  num_heads_q_padded,
                  " (compiled instantiations: 8, 16, 32, 64, 128).");
  }
#undef DISPATCH
}

}  // namespace deepseek_v4_fused_ops
}  // namespace vllm

// ────────────────────────────────────────────────────────────────────────────
// Torch op wrapper
// ────────────────────────────────────────────────────────────────────────────
// 中文注释：PyTorch 自定义算子包装函数
// ============================================================================
// 本函数是 Python 层调用本 CUDA kernel 的入口点，负责：
//   1. 验证所有输入张量的合法性（设备、形状、数据类型、连续性）
//   2. 提取运行时参数（token 数量、head 数量、block 大小等）
//   3. 分配输出张量 q_out（形状为 [N, kNumHeadsQPadded, 512]，bf16）
//   4. 通过 VLLM_DISPATCH_HALF_TYPES 宏分发到对应数据类型的 kernel 启动函数
//
// 输入张量说明：
//   q_in:          [N, num_heads_q, 512] bf16 -- 原始 Q 向量（未归一化、未编码）
//   kv:            [N, 512] bf16              -- 原始 KV 向量（单 head，MLA 的 latent KV）
//   k_cache:       [num_blocks, block_bytes] uint8 -- 分页 KV cache（字节级寻址）
//   slot_mapping:  [num_tokens_insert] int64  -- 每个 token 在 KV cache 中的全局 slot 编号
//   position_ids:  [N] int64                 -- 每个 token 的位置编号（用于查 cos/sin 表）
//   cos_sin_cache: [max_pos, 64] fp32        -- 预计算的 RoPE cos/sin 查找表
//   q_head_padded: int                       -- Q head 的 padded 数量（供 FlashMLA 使用）
//   eps:           float                     -- RMSNorm 的 epsilon 值（防止除零）
//   cache_block_size: int                    -- 分页 KV cache 每个 block 的 token 数
//
// 输出：
//   q_out: [N, q_head_padded, 512] bf16 -- 归一化 + RoPE 编码后的 Q 向量（含 padding）
// ============================================================================
torch::Tensor fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
    torch::Tensor const& q_in,           // [N, num_heads_q, 512] bf16
    torch::Tensor const& kv,             // [N, 512] bf16 (read-only)
    torch::Tensor& k_cache,              // [num_blocks, block_bytes] uint8
    torch::Tensor const& slot_mapping,   // [N] int64
    torch::Tensor const& position_ids,   // [N] int64
    torch::Tensor const& cos_sin_cache,  // [max_pos, rope_dim] bf16
    int64_t q_head_padded,               // padded Q head count for output
    double eps, int64_t cache_block_size) {
  // 中文注释：输入张量合法性检查
  // 检查所有张量是否在 CUDA 设备上、是否连续、数据类型和形状是否正确
  TORCH_CHECK(q_in.is_cuda() && q_in.is_contiguous(),
              "q_in must be contiguous CUDA");
  TORCH_CHECK(kv.is_cuda() && kv.is_contiguous(), "kv must be contiguous CUDA");
  TORCH_CHECK(k_cache.is_cuda(), "k_cache must be CUDA");
  TORCH_CHECK(slot_mapping.is_cuda() && slot_mapping.dtype() == torch::kInt64,
              "slot_mapping must be int64 CUDA");
  TORCH_CHECK(position_ids.is_cuda() && position_ids.dtype() == torch::kInt64,
              "position_ids must be int64 CUDA");
  TORCH_CHECK(cos_sin_cache.is_cuda(), "cos_sin_cache must be CUDA");
  TORCH_CHECK(q_in.dim() == 3 && q_in.size(2) == 512,
              "q_in shape [N, num_heads_q, 512]");
  TORCH_CHECK(kv.dim() == 2 && kv.size(1) == 512, "kv shape [N, 512]");
  TORCH_CHECK(q_in.dtype() == kv.dtype(), "q_in and kv dtype must match");
  TORCH_CHECK(q_head_padded >= q_in.size(1),
              "q_head_padded must be >= q_in.size(1) (num_heads_q)");
  TORCH_CHECK(k_cache.dtype() == torch::kUInt8, "k_cache must be uint8");
  TORCH_CHECK(cos_sin_cache.dim() == 2 && cos_sin_cache.size(1) == 64,
              "cos_sin_cache shape [max_pos, 64]");
  TORCH_CHECK(cos_sin_cache.dtype() == torch::kFloat32,
              "cos_sin_cache must be float32");

  // With DP padding, slot_mapping can be shorter than q/kv/positions.
  // Q-norm+RoPE runs on all q.size(0) rows (downstream attention uses them);
  // KV quant+insert runs only on the first slot_mapping.size(0) rows.
  // 中文注释：提取运行时参数
  // num_tokens_full = q_in 的行数 = kv 的行数 = position_ids 的长度
  //   （包含 DP padding 的所有 token）
  // num_tokens_insert = slot_mapping 的长度
  //   （仅包含需要写入 KV cache 的真实 token，不含 DP padding）
  int const num_tokens_full = static_cast<int>(q_in.size(0));
  int const num_tokens_insert = static_cast<int>(slot_mapping.size(0));
  TORCH_CHECK(static_cast<int>(kv.size(0)) == num_tokens_full &&
                  static_cast<int>(position_ids.size(0)) == num_tokens_full,
              "q/kv/position_ids row counts must match");
  TORCH_CHECK(num_tokens_insert <= num_tokens_full,
              "slot_mapping must not exceed q row count");
  int const num_heads_q = static_cast<int>(q_in.size(1));
  int const num_heads_q_padded = static_cast<int>(q_head_padded);
  int const cache_block_size_i = static_cast<int>(cache_block_size);
  // 中文注释：kv_block_stride 是 k_cache 张量在第 0 维上的 stride（字节数），
  // 即每个 KV cache block 占用的总字节数（数据区 + scale 区）
  int const kv_block_stride = static_cast<int>(k_cache.stride(0));

  // 中文注释：设置 CUDA 设备上下文（确保在正确的 GPU 上分配内存和执行）
  at::cuda::OptionalCUDAGuard device_guard(device_of(q_in));
  auto stream = at::cuda::getCurrentCUDAStream();

  // Allocate the padded q output.  The kernel writes every element (live
  // region gets RMSNorm+RoPE; pad region gets zeros), so `empty` is safe.
  // 中文注释：分配输出张量 q_out
  // 形状为 [N, q_head_padded, 512]，与 q_in 相同的数据类型（bf16）。
  // 使用 torch::empty（未初始化）是安全的，因为 kernel 会写入每一个元素：
  //   - live-Q 区域：写入 RMSNorm + RoPE 后的值
  //   - pad-Q 区域：写入零值
  torch::Tensor q_out = torch::empty(
      {q_in.size(0), q_head_padded, q_in.size(2)}, q_in.options());

  // 中文注释：通过宏分发到对应数据类型的 kernel 启动函数
  // VLLM_DISPATCH_HALF_TYPES 会根据 q_in 的标量类型（bf16 或 fp16）
  // 实例化 launchFusedDeepseekV4QNormRopeKVRopeQuantInsert 模板函数。
  // 实际上 DeepseekV4 只使用 bf16，但宏保持了通用性。
  VLLM_DISPATCH_HALF_TYPES(
      q_in.scalar_type(), "fused_deepseek_v4_qnorm_rope_kv_insert", [&] {
        using qkv_scalar_t = scalar_t;
        vllm::deepseek_v4_fused_ops::
            launchFusedDeepseekV4QNormRopeKVRopeQuantInsert<qkv_scalar_t>(
                reinterpret_cast<qkv_scalar_t const*>(q_in.data_ptr()),
                reinterpret_cast<qkv_scalar_t*>(q_out.data_ptr()),
                reinterpret_cast<qkv_scalar_t const*>(kv.data_ptr()),
                reinterpret_cast<uint8_t*>(k_cache.data_ptr()),
                reinterpret_cast<int64_t const*>(slot_mapping.data_ptr()),
                reinterpret_cast<int64_t const*>(position_ids.data_ptr()),
                cos_sin_cache.data_ptr<float>(), static_cast<float>(eps),
                num_tokens_full, num_tokens_insert, num_heads_q,
                num_heads_q_padded, cache_block_size_i, kv_block_stride,
                stream);
      });
  // 中文注释：返回处理后的 Q 向量，供下游 FlashMLA 等注意力算子使用
  return q_out;
}
