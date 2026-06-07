/*
 * Persistent TopK Scheduler for DSA Indexer
 */

// ============================================================================
// 文件概述：Persistent TopK 调度器（CUDA 实现）
// ============================================================================
// 本文件实现了高性能的 TopK 选择算法，用于 DSA（Dynamic Sparse Attention）
// 索引器中，从 logits 向量中快速选出概率最高的 K 个 token 索引。
//
// 核心设计思想：
//   采用"基数排序选择"（Radix Select）策略，将浮点数转换为有序整数后，
//   通过逐字节构建直方图 + 后缀和（suffix sum）定位第 K 大元素所在的
//   "桶"（bin），然后收集该桶及以上桶中的元素，最终得到 TopK 索引。
//
// 三条执行路径（根据序列长度自适应选择）：
//   1. Decode 路径（seq_len <= 8192）：
//      使用 2048-bin FP16 直方图，一次扫描通常即可定位阈值桶，
//      适用于解码阶段常见的短序列。
//   2. Medium 路径（8192 < seq_len <= 65536）：
//      使用 256-bin FP16 粗粒度直方图 + 4 轮 FP32 基数细化（refinement），
//      适用于中等长度的序列（如 prompt 较长时的 prefill）。
//   3. Large 路径（seq_len > 65536）：
//      多个 CTA 协作处理同一行（row），每个 CTA 负责一个 chunk，
//      通过全局内存中的直方图和屏障（barrier）进行 CTA 间同步，
//      4 轮 radix select 后收集 TopK 索引。
//
// 辅助功能：
//   - FilteredTopKUnifiedKernel：BS>32 时使用的独立 kernel，
//     采用双缓冲（double buffering）和向量化加载优化。
//   - convert_to_uint32_v2 / convert_to_uint8：将浮点数转换为有序整数，
//     保持大小关系不变，使基数排序可以直接基于整数比较。
// ============================================================================

#ifndef PERSISTENT_TOPK_CUH_
#define PERSISTENT_TOPK_CUH_

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cstdint>

namespace vllm {
namespace persistent {

// ============================================================================
// Constants
// ============================================================================

// 中文注释：每个 CUDA block 的线程数，所有 kernel 统一使用 1024 线程。
constexpr int kThreadsPerBlock = 1024;

// 中文注释：基数（radix）的值，即每个字节 8 位对应 256 个桶（bin）。
// 所有路径的直方图都以 256 为基数，每次细化一个字节（8 bit）。
constexpr int RADIX = 256;

// Medium path: all shared state in dynamic smem (no static __shared__,
// which would inflate the kernel's smem footprint and kill occupancy
// for the decode/trivial paths).

// 中文注释：Medium 路径的共享内存布局常量。
// 使用动态共享内存（dynamic smem）而非静态 __shared__，避免在
// decode/trivial 路径中占用额外的共享内存、降低占用率（occupancy）。
//
// kMediumHistBytes: 双缓冲直方图所需的字节数（2 个缓冲区 x 384 个 int）。
//   额外的 128 是为了后缀和计算时的对齐空间。
constexpr size_t kMediumHistBytes = 2 * (RADIX + 128) * sizeof(int);  // 3072
// 中文注释：标量变量所需的字节数（output_count, threshold_bin,
//   buffered_count[2], final_k 共 5 个 int）。
constexpr size_t kMediumScalarsBytes = 5 * sizeof(int);               // 20
// 中文注释：头部区域大小（直方图 + 标量），按 128 字节对齐以满足合并访问要求。
constexpr size_t kMediumHeaderSize =
    (kMediumHistBytes + kMediumScalarsBytes + 127) & ~size_t(127);  // 3200
// 中文注释：每个缓冲区最多缓存的候选元素数量。当阈值桶中的元素过多时，
// 超出此数量的元素将被丢弃（极端情况下可能丢失精度，但概率极低）。
constexpr int MAX_BUFFERED_ITEMS = 4096;
// 中文注释：Medium 路径总共享内存大小 = 头部 + 两个候选缓冲区。
constexpr size_t kSmemMedium =
    kMediumHeaderSize + 2 * MAX_BUFFERED_ITEMS * sizeof(int);  // 35968
// 中文注释：序列长度阈值，低于此值走 Decode 路径，高于此值走 Medium 或 Large 路径。
constexpr uint32_t RADIX_THRESHOLD = 32768;

// Decode path constants

// 中文注释：Decode 路径使用 2048 个 bin 的直方图。
// 2048 = 2^11，对应 FP16 表示的高 11 位，提供更细粒度的分桶。
constexpr int kDecodeBins = 2048;
// 中文注释：当 seq_len <= 8192 时走 Decode 路径（2048-bin 直方图）。
// 8192 / 2048 = 4 个元素/bin（平均），一次扫描通常足够定位阈值桶。
constexpr uint32_t HIST2048_THRESHOLD = 8192;

// Large path: fixed shared memory for histograms + scalars

// 中文注释：Large 路径的固定共享内存大小。
// 包含：256 个 local_histogram + 256 个 suffix_sum + 5 个标量，
// 按 16 字节对齐。剩余的共享内存用于存放转换后的有序 uint32 数据。
constexpr size_t kFixedSmemLarge =
    ((RADIX + RADIX + 5) * sizeof(uint32_t) + 15) & ~size_t(15);

// ============================================================================
// Common helpers
// ============================================================================

// 中文注释：将 float32 转换为保持大小关系的 uint32。
// 原理：IEEE 754 浮点数中，正数的二进制表示天然有序（指数 + 尾数），
// 但负数的有序性与正数相反。因此：
//   - 正数：设置最高位为 1（bits | 0x80000000），使其大于所有原始负数。
//   - 负数：按位取反（~bits），使得更小的负数映射为更大的 uint32。
// 转换后，浮点数的大小关系完全映射为 uint32 的大小关系，
// 可以直接用整数比较进行基数排序。
__device__ __forceinline__ auto convert_to_uint32_v2(float x) -> uint32_t {
  uint32_t bits = __float_as_uint(x);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

// 中文注释：将 float32 转换为保持大小关系的 uint8（取 FP16 的高 8 位）。
// 流程：float32 -> FP16 -> 提取有序的 uint16 -> 右移 8 位取高字节。
// 结果是 256 个 bin 的粗粒度键，用于 Medium 路径的初始直方图构建。
// 精度损失由后续 4 轮 radix refinement 补偿。
__device__ __forceinline__ auto convert_to_uint8(float x) -> uint8_t {
  __half h = __float2half_rn(x);
  uint16_t bits = __half_as_ushort(h);
  uint16_t key = (bits & 0x8000) ? static_cast<uint16_t>(~bits)
                                 : static_cast<uint16_t>(bits | 0x8000);
  return static_cast<uint8_t>(key >> 8);
}

// ============================================================================
// Vectorized load helpers
// ============================================================================

// 中文注释：无条件向量化加载 4 个 float（float4）。
// 使用 PTX 内联汇编实现，.cg 提示仅在全局内存级别缓存（不经过 L1）。
// 适用于地址已对齐且不会越界的场景，性能优于逐元素加载。
__device__ __forceinline__ void load_float4(const float* ptr, float& v0,
                                            float& v1, float& v2, float& v3) {
  uint32_t r0, r1, r2, r3;
  asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "l"(ptr));
  v0 = __uint_as_float(r0);
  v1 = __uint_as_float(r1);
  v2 = __uint_as_float(r2);
  v3 = __uint_as_float(r3);
}

// 中文注释：带谓词的标量加载（每元素独立判断是否越界）。
// 当地址未对齐或存在越界风险时使用。越界位置填充 -inf（0xFF800000），
// 确保这些无效元素在排序中排在最前面，不会干扰 TopK 结果。
// 使用 PTX 谓词指令 @pr 实现条件加载，避免分支发散。
__device__ __forceinline__ void load_float4_predicated(const float* ptr,
                                                       int base, int seq_len,
                                                       float& v0, float& v1,
                                                       float& v2, float& v3) {
  uint32_t r0, r1, r2, r3;
  int p0 = (base < seq_len);
  int p1 = (base + 1 < seq_len);
  int p2 = (base + 2 < seq_len);
  int p3 = (base + 3 < seq_len);
  asm volatile(
      "{\n"
      "  .reg .pred pr0, pr1, pr2, pr3;\n"
      "  setp.ne.u32 pr0, %4, 0;\n"
      "  setp.ne.u32 pr1, %5, 0;\n"
      "  setp.ne.u32 pr2, %6, 0;\n"
      "  setp.ne.u32 pr3, %7, 0;\n"
      "  mov.u32 %0, 0xFF800000;\n"
      "  mov.u32 %1, 0xFF800000;\n"
      "  mov.u32 %2, 0xFF800000;\n"
      "  mov.u32 %3, 0xFF800000;\n"
      "  @pr0 ld.global.cg.u32 %0, [%8];\n"
      "  @pr1 ld.global.cg.u32 %1, [%8+4];\n"
      "  @pr2 ld.global.cg.u32 %2, [%8+8];\n"
      "  @pr3 ld.global.cg.u32 %3, [%8+12];\n"
      "}\n"
      : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
      : "r"(p0), "r"(p1), "r"(p2), "r"(p3), "l"(ptr));
  v0 = __uint_as_float(r0);
  v1 = __uint_as_float(r1);
  v2 = __uint_as_float(r2);
  v3 = __uint_as_float(r3);
}

// ============================================================================
// Large path: inter-CTA coordination state (one per group)
// ============================================================================

// 中文注释：Large 路径中多个 CTA 协作处理同一行时的共享协调状态。
// 每个 CTA 组（group）对应一个 RadixRowState 实例，存放在全局内存中。
//
// histogram[3][256]：三重缓冲（triple-buffered）直方图。
//   使用 3 个缓冲区可以在当前轮写入的同时读取上一轮的结果，
//   避免 CTA 间的写后读冲突。
// remaining_k：当前轮还需选出的元素数量。
// prefix：已确定的前缀（高位匹配值），用于过滤不相关的元素。
// arrival_counter：到达屏障的 CTA 计数器，用于 CTA 间同步。
// output_counter：输出位置的原子计数器，协调多 CTA 的写入位置。
struct RadixRowState {
  uint32_t histogram[3][256];  // Triple-buffered histograms
  uint32_t remaining_k;
  uint32_t prefix;
  int arrival_counter;
  int output_counter;
};

// ============================================================================
// Kernel parameters
// ============================================================================

// 中文注释：Persistent TopK kernel 的参数结构体。
// 所有参数通过一个结构体传递，减少 kernel 启动时的参数拷贝开销。
struct PersistentTopKParams {
  const float* __restrict__ input;      // [num_rows, stride] 输入 logits 矩阵
  int32_t* __restrict__ output;         // [num_rows, top_k] 输出索引矩阵
  const int32_t* __restrict__ lengths;  // [num_rows] 每行的有效长度
  RadixRowState* row_states;            // large path: per-group state（全局内存中的协调状态）
  uint32_t num_rows;      // 总行数（即 batch 中的请求数）
  uint32_t stride;        // 输入矩阵的行步长（每行占用的 float 数量，可能含 padding）
  uint32_t top_k;         // 实际的 K 值，决定输出每行的宽度
  uint32_t chunk_size;    // large path: 每个 CTA 负责的元素数量
  uint32_t ctas_per_group;  // 1=medium/decode（单 CTA 处理整行），>1=large（多 CTA 协作）
  uint32_t max_seq_len;     // 所有行中最大的 seq_len，用于非 CTA-0 线程的提前退出优化
};

// ============================================================================
// Decode path: 2048-bin histogram for short sequences (seq_len <= 8192)
// Uses 11-bit half-precision bins for fine granularity.
// One histogram pass typically suffices since 8192/2048 = 4 elements/bin avg.
// ============================================================================

// 中文注释：将 float32 转换为 11-bit 的有序 bin 索引（0~2047）。
// 流程：float32 -> FP16 -> 提取有序 uint16 -> 右移 5 位取高 11 位。
// 与 convert_to_uint8 的原理相同，但精度更高（11 位 vs 8 位），
// 使得 2048 个 bin 的粒度足够细，8192 个元素平均每个 bin 仅 4 个，
// 一次直方图扫描通常就能精确定位第 K 大元素所在的 bin。
__device__ __forceinline__ uint32_t decode_bin(float x) {
  __half hx = __float2half(x);
  uint16_t bits = __half_as_ushort(hx);
  uint16_t key = (bits & 0x8000) ? static_cast<uint16_t>(~bits)
                                 : static_cast<uint16_t>(bits | 0x8000);
  return key >> 5;
}

// 中文注释：Decode 路径的 TopK kernel 函数（seq_len <= 8192）。
// 整体流程分为三个阶段：
//   阶段 1：构建 2048-bin 直方图
//     - 使用 float4 向量化加载，将每个元素映射到 2048 个 bin 之一
//     - 通过 atomicAdd 累加每个 bin 的计数
//     - 使用 CUB BlockScan 计算后缀和，定位阈值 bin
//   阶段 2：收集高于阈值的元素
//     - 使用 warp 级别的 __ballot_sync 和 __popc 进行高效的 warp 聚合 atomicAdd
//     - 高于阈值 bin 的元素直接写入输出
//     - 等于阈值 bin 的元素缓存到 buffer 中
//   阶段 3：延迟细化（罕见路径）
//     - 当阈值桶中的元素不足以填满剩余的 TopK 时触发
//     - 对缓冲区中的元素进行 4 轮 FP32 基数细化，逐字节精确定位
template <int TopK>
__device__ __noinline__ void histogram_2048_topk(
    const float* __restrict__ logits, int32_t* __restrict__ output_indices,
    int32_t seq_len) {
  extern __shared__ int decode_smem[];
  const int tx = threadIdx.x;
  const int lane = tx & 31;  // 当前线程在其 warp 内的 lane 编号（0~31）

  // ---- Layout constants ----
  // 中文注释：共享内存布局设计（总共 8192 个 int = 32KB）：
  //   [0, 2048)       : 2048-bin 直方图（阶段 1）
  //   [2048, 768)     : 后缀和临时空间（与直方图共享，阶段 1 后不再需要）
  //   [768, 768+DBUF) : 候选缓冲区 0（阶段 2 收集阈值桶元素）
  //   [768+DBUF, SBASE): 候选缓冲区 1（细化阶段使用）
  //   [SBASE, SBASE+8): 标量变量（阈值、输出计数、refine 计数等）
  constexpr int SBASE = 8192 - 8;           // 8184（标量区域起始位置）
  constexpr int RHIST = RADIX + 128;        // 384（细化直方图大小，含对齐空间）
  constexpr int BOFF = 2 * RHIST;           // 768（候选缓冲区起始偏移）
  constexpr int DBUF = (SBASE - BOFF) / 2;  // 3708（每个候选缓冲区的最大容量）
  constexpr int MAX_ITEMS_PER_THREAD =
      (HIST2048_THRESHOLD + kThreadsPerBlock - 1) / kThreadsPerBlock;

  // 中文注释：标量区域的偏移索引。
  //   sTHR: 阈值 bin 编号（哪个 bin 是第 K 大元素所在的分界线）
  //   sOUT: 已输出的元素计数
  //   sREF: 细化阶段的阈值 bin
  //   sFIN: 细化阶段剩余需要填充的槽位数
  //   sBUF0/sBUF1: 两个候选缓冲区的元素计数
  enum : int { sTHR = 0, sOUT = 1, sREF = 2, sFIN = 3, sBUF0 = 4, sBUF1 = 5 };

  // ---- Initialize scalars (prevents stale data from prior rows) ----
  // 中文注释：初始化标量区域，防止上一行遗留的脏数据影响当前行的计算。
  if (tx < 8) {
    decode_smem[SBASE + tx] = 0;
  }

  // ---- Phase 1: Build 2048-bin histogram with float4 vectorized loads ----
  // 中文注释：阶段 1 —— 构建 2048-bin 直方图。
  // 流程：
  //   1) 初始化直方图为 0
  //   2) 每个线程使用 float4 向量化加载（一次读 4 个 float），
  //      将每个 float 转换为 11-bit bin 索引（decode_bin）
  //   3) 将 bin 索引存入寄存器数组 reg_bins（供阶段 2 复用，避免二次读取）
  //   4) 使用 atomicAdd 累加直方图
  int* histo = decode_smem;
  uint16_t reg_bins[MAX_ITEMS_PER_THREAD];  // 中文注释：寄存器中缓存的 bin 索引，供阶段 2 复用
  int nitems = 0;

  for (int i = tx; i < kDecodeBins; i += kThreadsPerBlock) {
    histo[i] = 0;
  }
  __syncthreads();

  const int n_vec = (seq_len + 3) >> 2;
  const bool row_aligned = ((reinterpret_cast<uintptr_t>(logits) & 15) == 0);

  for (int i = tx; i < n_vec; i += kThreadsPerBlock) {
    const int base = i << 2;
    float v0, v1, v2, v3;

    if (row_aligned && base + 3 < seq_len) {
      load_float4(logits + base, v0, v1, v2, v3);
    } else {
      load_float4_predicated(logits + base, base, seq_len, v0, v1, v2, v3);
    }

    const uint16_t b0 = static_cast<uint16_t>(decode_bin(v0));
    const uint16_t b1 = static_cast<uint16_t>(decode_bin(v1));
    const uint16_t b2 = static_cast<uint16_t>(decode_bin(v2));
    const uint16_t b3 = static_cast<uint16_t>(decode_bin(v3));
    reg_bins[nitems++] = b0;
    reg_bins[nitems++] = b1;
    reg_bins[nitems++] = b2;
    reg_bins[nitems++] = b3;
    atomicAdd(&histo[b0], 1);
    atomicAdd(&histo[b1], 1);
    atomicAdd(&histo[b2], 1);
    atomicAdd(&histo[b3], 1);
  }
  __syncthreads();

  // ---- CUB suffix sum ----
  // 中文注释：使用 CUB BlockScan 计算直方图的后缀和（suffix sum）。
  // 后缀和 suffix[i] = sum(histogram[j] for j >= i)，表示 bin i 及以上桶中的元素总数。
  // 优化技巧：将相邻两个 bin 配对（pair），1024 个线程处理 2048 个 bin，
  // 每个线程负责一对，减少 BlockScan 的参与线程数。
  using BlockScanT = cub::BlockScan<int, kThreadsPerBlock>;
  const int h0 = histo[2 * tx];
  const int pair_sum = h0 + histo[2 * tx + 1];

  auto& scan_storage = *reinterpret_cast<typename BlockScanT::TempStorage*>(
      decode_smem + kDecodeBins);

  int pair_prefix, total;
  BlockScanT(scan_storage).ExclusiveSum(pair_sum, pair_prefix, total);

  // Find threshold bin purely from registers
  // 中文注释：从后缀和中定位阈值 bin。
  // 条件：suffix[threshold] >= TopK 且 suffix[threshold+1] < TopK，
  // 即第 TopK 大的元素恰好落在 threshold 这个 bin 中。
  const int pair_suffix = total - pair_prefix;

  // 中文注释：检查配对中的第一个 bin（偶数 bin）
  if (pair_suffix >= TopK && (pair_suffix - h0) < TopK) {
    decode_smem[SBASE + sTHR] = 2 * tx;
  }
  {
    // 中文注释：检查配对中的第二个 bin（奇数 bin）
    const int right_suf = pair_suffix - h0;
    const int next_suf = pair_suffix - pair_sum;
    if (right_suf >= TopK && next_suf < TopK) {
      decode_smem[SBASE + sTHR] = 2 * tx + 1;
    }
  }
  __syncthreads();

  const int threshold = decode_smem[SBASE + sTHR];

  // ---- Phase 2: Collection with warp-aggregated atomicAdds ----
  // 中文注释：阶段 2 —— 收集 TopK 元素的索引。
  // 核心思想：
  //   - bin > threshold 的元素必然属于 TopK，直接写入输出数组。
  //   - bin == threshold 的元素是候选者（可能属于 TopK），先缓存到 buffer。
  //   - bin < threshold 的元素必然不属于 TopK，直接跳过。
  // 效率优化：使用 warp 级别的 __ballot_sync 聚合同一 warp 中满足条件的线程，
  // 然后只由 lane 0 执行一次 atomicAdd 获取写入基地址，
  // 再通过 __shfl_sync 广播给所有线程，避免每个线程单独 atomicAdd。
  int* bufs[2] = {decode_smem + BOFF, decode_smem + BOFF + DBUF};
  const int sOUT_abs = SBASE + sOUT;
  const int sBUF0_abs = SBASE + sBUF0;

  {
    const uint32_t uthr = static_cast<uint32_t>(threshold);
    int item = 0;
    const int n_vec_iters = (n_vec + kThreadsPerBlock - 1) / kThreadsPerBlock;

    for (int iter = 0; iter < n_vec_iters; iter++) {
      const int i = tx + iter * kThreadsPerBlock;
      const bool vec_valid = (i < n_vec);
      const int base_idx = i << 2;

#pragma unroll 4
      for (int sub = 0; sub < 4; sub++) {
        const int elem_idx = base_idx + sub;
        uint32_t bin = 0;
        if (vec_valid) bin = reg_bins[item++];
        const bool is_above = vec_valid && (bin > uthr);
        const bool is_equal = vec_valid && (bin == uthr);

        const uint32_t above_mask = __ballot_sync(0xffffffff, is_above);
        if (above_mask) {
          const int above_count = __popc(above_mask);
          const int above_rank = __popc(above_mask & ((1u << lane) - 1));
          int above_base;
          if (lane == 0) {
            above_base = atomicAdd(&decode_smem[sOUT_abs], above_count);
          }
          above_base = __shfl_sync(0xffffffff, above_base, 0);
          if (is_above) {
            output_indices[above_base + above_rank] = elem_idx;
          }
        }

        const uint32_t equal_mask = __ballot_sync(0xffffffff, is_equal);
        if (equal_mask) {
          const int equal_count = __popc(equal_mask);
          const int equal_rank = __popc(equal_mask & ((1u << lane) - 1));
          int equal_base;
          if (lane == 0) {
            equal_base = atomicAdd(&decode_smem[sBUF0_abs], equal_count);
          }
          equal_base = __shfl_sync(0xffffffff, equal_base, 0);
          if (is_equal && __builtin_expect(equal_base + equal_rank < DBUF, 1)) {
            bufs[0][equal_base + equal_rank] = elem_idx;
          }
        }
      }
    }
  }
  __syncthreads();

  // 中文注释：计算还差多少个元素才能凑满 TopK。
  int remaining_k = TopK - decode_smem[SBASE + sOUT];
  if (remaining_k <= 0) return;

  // If all buffered elements fit, output them all (common for short seqs)
  // 中文注释：快速路径——如果缓冲区中的候选元素数量 <= remaining_k，
  // 说明所有候选元素都属于 TopK，直接全部输出即可（短序列的常见情况）。
  const int raw_buf0 = decode_smem[SBASE + sBUF0];
  if (raw_buf0 <= remaining_k) {
    const int nb = (raw_buf0 < DBUF) ? raw_buf0 : DBUF;
    const int base = decode_smem[SBASE + sOUT];
    for (int i = tx; i < nb; i += kThreadsPerBlock) {
      output_indices[base + i] = bufs[0][i];
    }
    __syncthreads();
    return;
  }

  // ---- Phase 3: Deferred refinement (rare path) ----
  // 中文注释：阶段 3 —— 延迟细化（罕见路径）。
  // 当阈值桶中的候选元素数量 > remaining_k 时，需要进一步区分这些元素。
  // 方法：对候选元素的 FP32 有序表示逐字节进行 4 轮基数细化（radix refinement）。
  // 每轮：构建 256-bin 直方图 -> 计算后缀和 -> 定位新的阈值 bin -> 收集/过滤。
  // 双缓冲 refine[2] 交替使用，避免写后读冲突。
  int* refine[2] = {decode_smem, decode_smem + RHIST};
  const int num_buf0 = (raw_buf0 < DBUF) ? raw_buf0 : DBUF;

  for (int i = tx; i < RHIST; i += kThreadsPerBlock) {
    refine[0][i] = 0;
  }
  __syncthreads();

  for (int i = tx; i < num_buf0; i += kThreadsPerBlock) {
    const uint32_t fp32 = convert_to_uint32_v2(logits[bufs[0][i]]);
    atomicAdd(&refine[0][(fp32 >> 24) & 0xFF], 1);
  }
  __syncthreads();

  // 中文注释：计算后缀和的 lambda 函数。
  // 使用 Hillis-Steele 并行前缀和算法的变体（后缀和版本）。
  // 8 轮迭代，每轮步长翻倍（1, 2, 4, ..., 128），双缓冲交替读写。
  // 复杂度 O(log N) 轮同步，每轮 O(N) 工作量。
  auto compute_suffix_sum = [&]() {
#pragma unroll 8
    for (int i = 0; i < 8; ++i) {
      if (tx < RADIX) {
        const int stride = 1 << i;
        const int s = i & 1;      // 源缓冲区索引
        const int d = s ^ 1;      // 目标缓冲区索引
        int value = refine[s][tx];
        if (tx < RADIX - stride) value += refine[s][tx + stride];
        refine[d][tx] = value;
      }
      __syncthreads();
    }
  };

#pragma unroll 4
  // 中文注释：4 轮基数细化主循环。
  // 每轮处理 FP32 有序表示的一个字节（从高位到低位：byte3, byte2, byte1, byte0）。
  // 双缓冲策略：src=当前轮读取的缓冲区，dst=下一轮写入的缓冲区。
  for (int pass = 0; pass < 4; ++pass) {
    const int src = pass & 1;
    const int dst = src ^ 1;

    const int raw_buf = decode_smem[SBASE + sBUF0 + src];
    const int num_buffered = (raw_buf < DBUF) ? raw_buf : DBUF;

    // 中文注释：计算当前轮候选元素的后缀和，定位新的阈值 bin。
    compute_suffix_sum();

    // 中文注释：找到新的阈值 bin —— 即 remaining_k 个最大元素所在的 bin。
    if (tx < RADIX && refine[0][tx] > remaining_k &&
        refine[0][tx + 1] <= remaining_k) {
      decode_smem[SBASE + sREF] = tx;
      decode_smem[SBASE + sBUF0 + dst] = 0;
      decode_smem[SBASE + sFIN] = remaining_k - refine[0][tx + 1];
    }
    __syncthreads();

    const int ref_thr = decode_smem[SBASE + sREF];
    remaining_k -= refine[0][ref_thr + 1];
    const int bit_offset = 24 - pass * 8;  // 中文注释：当前轮处理的字节位偏移

    // 中文注释：如果 remaining_k 已降为 0，说明所有 TopK 元素已在之前轮次确定，
    // 只需收集本轮中 > ref_thr 的元素即可。
    if (remaining_k == 0) {
      for (int i = tx; i < num_buffered; i += kThreadsPerBlock) {
        const int idx = bufs[src][i];
        const uint32_t fp32 = convert_to_uint32_v2(logits[idx]);
        if (((fp32 >> bit_offset) & 0xFF) > static_cast<uint32_t>(ref_thr)) {
          const int pos = atomicAdd(&decode_smem[SBASE + sOUT], 1);
          output_indices[pos] = idx;
        }
      }
      __syncthreads();
      break;
    }

    // 中文注释：remaining_k > 0，需要继续细化，重置直方图。
    __syncthreads();
    if (tx < RADIX + 1) refine[0][tx] = 0;
    __syncthreads();

    for (int i = tx; i < num_buffered; i += kThreadsPerBlock) {
      const int idx = bufs[src][i];
      const float logit_val = logits[idx];
      const uint32_t fp32 = convert_to_uint32_v2(logit_val);
      const int bin = (fp32 >> bit_offset) & 0xFF;

      if (bin > ref_thr) {
        // 中文注释：bin > 阈值，该元素必然属于 TopK，直接输出。
        const int pos = atomicAdd(&decode_smem[SBASE + sOUT], 1);
        output_indices[pos] = idx;
      } else if (bin == ref_thr) {
        // 中文注释：bin == 阈值，该元素可能属于 TopK。
        if (pass == 3) {
          // 中文注释：最后一轮（处理最低字节），无法再细化，
          // 直接将剩余槽位分配给等于阈值的元素（任意顺序均可）。
          const int slot = atomicAdd(&decode_smem[SBASE + sFIN], -1);
          if (slot > 0) output_indices[TopK - slot] = idx;
        } else {
          // 中文注释：非最后一轮，将等于阈值的元素存入另一个缓冲区，
          // 供下一轮继续细化。
          const int bp = atomicAdd(&decode_smem[SBASE + sBUF0 + dst], 1);
          if (__builtin_expect(bp < DBUF, 1)) {
            bufs[dst][bp] = idx;
            const int nbo = bit_offset - 8;
            // 中文注释：同时构建下一轮的直方图（基于下一个字节）。
            atomicAdd(&refine[0][(fp32 >> nbo) & 0xFF], 1);
          }
        }
      }
    }
    __syncthreads();
  }
}

// ============================================================================
// Medium path: coarse FP16 histogram + 4-pass FP32 radix refinement
// For sequences 8K < seq_len <= 64K.
// ============================================================================

// Adapted from:
// https://github.com/sgl-project/sglang/blob/v0.5.8/sgl-kernel/csrc/elementwise/topk.cu#L87
// by: DarkSharpness
// which at the same time is an optimized topk kernel copied from tilelang
// kernel

// 中文注释：Medium 路径的 TopK kernel 函数（8192 < seq_len <= 65536）。
// 与 Decode 路径的主要区别：
//   1) 使用 256-bin FP16 粗粒度直方图（而非 2048-bin），初始桶更粗。
//   2) 需要 4 轮 FP32 基数细化（radix refinement）来精确定位。
//   3) 使用动态共享内存（dynamic smem），避免在非 Medium 路径中浪费共享内存。
//
// 整体流程：
//   阶段 1：构建 256-bin FP16 直方图，计算后缀和，定位阈值 bin。
//   阶段 2：收集 > 阈值的元素到输出，= 阈值的元素到缓冲区，
//           同时为细化阶段构建下一轮直方图。
//   阶段 3：4 轮基数细化，每轮处理 FP32 有序表示的一个字节，
//           逐步缩小候选集，直到 remaining_k 降为 0。
template <int TopK>
__device__ __noinline__ void histogram_256_topk(
    const float* __restrict__ logits, int* __restrict__ output_indices,
    int logits_offset, int seq_len) {
  // All shared state lives in dynamic shared memory to avoid static
  // 中文注释：所有共享状态都使用动态共享内存，避免静态 __shared__ 变量
  // 在 kernel 启动时就占用共享内存，影响 decode/trivial 路径的占用率。
  extern __shared__ char medium_smem[];

  // 中文注释：共享内存布局——
  //   [0, kMediumHistBytes)           : 双缓冲直方图（2 x 384 个 int）
  //   [kMediumHistBytes, +20)        : 标量变量（output_count, threshold_bin, buffered_count[2], final_k）
  //   [kMediumHeaderSize, +)         : 双缓冲候选索引数组（2 x 4096 个 int）
  int (*shared_histogram)[RADIX + 128] =
      reinterpret_cast<int (*)[RADIX + 128]>(medium_smem);
  int* medium_scalars = reinterpret_cast<int*>(medium_smem + kMediumHistBytes);
  int& shared_output_count = medium_scalars[0];      // 中文注释：已输出的元素计数
  int& shared_threshold_bin = medium_scalars[1];      // 中文注释：当前阈值 bin 编号
  int* shared_buffered_count = &medium_scalars[2];    // 中文注释：两个缓冲区的元素计数（双缓冲）
  int& shared_final_k = medium_scalars[4];            // 中文注释：最后一轮剩余的槽位数
  int (*buffered_indices)[MAX_BUFFERED_ITEMS] =
      reinterpret_cast<int (*)[MAX_BUFFERED_ITEMS]>(medium_smem +
                                                    kMediumHeaderSize);

  const int thread_id = threadIdx.x;
  int remaining_k = TopK;

  // 中文注释：阶段 1 —— 构建 256-bin FP16 粗粒度直方图。
  // 每个线程遍历自己负责的元素，将 float32 转换为 8-bit bin 索引，
  // 并通过 atomicAdd 累加直方图。
  if (thread_id < RADIX + 1) {
    shared_histogram[0][thread_id] = 0;
  }
  __syncthreads();

  for (int idx = thread_id; idx < seq_len; idx += kThreadsPerBlock) {
    const auto bin = convert_to_uint8(logits[idx + logits_offset]);
    atomicAdd(&shared_histogram[0][bin], 1);
  }
  __syncthreads();

  // 中文注释：计算后缀和的 lambda 函数（与 Decode 路径的 Hillis-Steele 变体相同）。
  // 8 轮迭代，步长从 1 递增到 128，双缓冲交替读写。
  // 结果：shared_histogram[0][i] = sum(histogram[j] for j >= i)。
  auto compute_cumulative_sum = [&]() {
#pragma unroll 8
    for (int i = 0; i < 8; ++i) {
      if (__builtin_expect(thread_id < RADIX, 1)) {
        const int stride = 1 << i;
        const int src_buffer = i & 1;
        const int dst_buffer = src_buffer ^ 1;
        int value = shared_histogram[src_buffer][thread_id];
        if (thread_id < RADIX - stride) {
          value += shared_histogram[src_buffer][thread_id + stride];
        }
        shared_histogram[dst_buffer][thread_id] = value;
      }
      __syncthreads();
    }
  };

  // 中文注释：计算后缀和后定位阈值 bin。
  // 条件：suffix[threshold] > TopK 且 suffix[threshold+1] <= TopK。
  compute_cumulative_sum();

  if (thread_id < RADIX && shared_histogram[0][thread_id] > remaining_k &&
      shared_histogram[0][thread_id + 1] <= remaining_k) {
    shared_threshold_bin = thread_id;
    shared_buffered_count[0] = 0;
    shared_output_count = 0;
  }
  __syncthreads();

  const int threshold_bin = shared_threshold_bin;
  remaining_k -= shared_histogram[0][threshold_bin + 1];

  // 中文注释：快速路径——如果阈值桶以上的元素已经 >= TopK，
  // 说明无需细化，直接收集所有 > 阈值 bin 的元素即可。
  if (remaining_k == 0) {
    for (int idx = thread_id; idx < seq_len; idx += kThreadsPerBlock) {
      const int bin = convert_to_uint8(logits[idx + logits_offset]);
      if (bin > threshold_bin) {
        const int output_pos = atomicAdd(&shared_output_count, 1);
        output_indices[output_pos] = idx;
      }
    }
    __syncthreads();
    return;
  }

  // 中文注释：阶段 2 —— 收集 + 为细化阶段构建直方图。
  // 重新遍历所有元素：
  //   - bin > 阈值：直接输出（必然属于 TopK）
  //   - bin == 阈值：存入缓冲区（候选者），同时提取 FP32 有序表示的最高字节
  //     构建细化阶段的直方图，避免后续再遍历原始数据。
  //   - bin < 阈值：跳过（必然不属于 TopK）
  __syncthreads();
  if (thread_id < RADIX + 1) {
    shared_histogram[0][thread_id] = 0;
  }
  __syncthreads();

  for (int idx = thread_id; idx < seq_len; idx += kThreadsPerBlock) {
    const float logit_value = logits[idx + logits_offset];
    const int bin = convert_to_uint8(logit_value);
    if (bin > threshold_bin) {
      const int output_pos = atomicAdd(&shared_output_count, 1);
      output_indices[output_pos] = idx;
    } else if (bin == threshold_bin) {
      const int buffer_pos = atomicAdd(&shared_buffered_count[0], 1);
      if (__builtin_expect(buffer_pos < MAX_BUFFERED_ITEMS, 1)) {
        buffered_indices[0][buffer_pos] = idx;
        const uint32_t fp32_bits = convert_to_uint32_v2(logit_value);
        const int next_bin = (fp32_bits >> 24) & 0xFF;
        atomicAdd(&shared_histogram[0][next_bin], 1);
      }
    }
  }
  __syncthreads();

#pragma unroll 4
  // 中文注释：阶段 3 —— 4 轮基数细化主循环。
  // 与 Decode 路径的细化逻辑完全一致，但使用双缓冲的候选索引数组。
  // 每轮处理 FP32 有序表示的一个字节（从最高字节 byte3 到最低字节 byte0）。
  for (int pass = 0; pass < 4; ++pass) {
    const int src_buffer = pass % 2;       // 中文注释：当前轮读取的缓冲区索引
    const int dst_buffer = src_buffer ^ 1; // 中文注释：下一轮写入的缓冲区索引
    const int raw_buffered = shared_buffered_count[src_buffer];
    const int num_buffered =
        (raw_buffered < MAX_BUFFERED_ITEMS) ? raw_buffered : MAX_BUFFERED_ITEMS;

    compute_cumulative_sum();

    // 中文注释：定位新的阈值 bin。
    if (thread_id < RADIX && shared_histogram[0][thread_id] > remaining_k &&
        shared_histogram[0][thread_id + 1] <= remaining_k) {
      shared_threshold_bin = thread_id;
      shared_buffered_count[dst_buffer] = 0;
      shared_final_k = remaining_k - shared_histogram[0][thread_id + 1];
    }
    __syncthreads();

    const int threshold_bin = shared_threshold_bin;
    remaining_k -= shared_histogram[0][threshold_bin + 1];
    const int bit_offset = 24 - pass * 8;  // 中文注释：当前轮处理的字节位偏移

    // 中文注释：remaining_k 降为 0，只需收集 > 阈值的元素。
    if (remaining_k == 0) {
      for (int i = thread_id; i < num_buffered; i += kThreadsPerBlock) {
        const int idx = buffered_indices[src_buffer][i];
        const uint32_t fp32_bits =
            convert_to_uint32_v2(logits[idx + logits_offset]);
        const int bin = (fp32_bits >> bit_offset) & 0xFF;
        if (bin > threshold_bin) {
          const int output_pos = atomicAdd(&shared_output_count, 1);
          output_indices[output_pos] = idx;
        }
      }
      __syncthreads();
      break;
    }

    // 中文注释：remaining_k > 0，重置直方图，继续下一轮细化。
    __syncthreads();
    if (thread_id < RADIX + 1) {
      shared_histogram[0][thread_id] = 0;
    }
    __syncthreads();

    for (int i = thread_id; i < num_buffered; i += kThreadsPerBlock) {
      const int idx = buffered_indices[src_buffer][i];
      const float logit_value = logits[idx + logits_offset];
      const uint32_t fp32_bits = convert_to_uint32_v2(logit_value);
      const int bin = (fp32_bits >> bit_offset) & 0xFF;
      if (bin > threshold_bin) {
        // 中文注释：bin > 阈值，必然属于 TopK，直接输出。
        const int output_pos = atomicAdd(&shared_output_count, 1);
        output_indices[output_pos] = idx;
      } else if (bin == threshold_bin) {
        // 中文注释：bin == 阈值，候选者。
        if (pass == 3) {
          // 中文注释：最后一轮，无法再细化，直接分配剩余槽位。
          const int slot = atomicAdd(&shared_final_k, -1);
          if (slot > 0) {
            output_indices[TopK - slot] = idx;
          }
        } else {
          // 中文注释：非最后一轮，存入另一个缓冲区，供下一轮细化。
          const int buffer_pos =
              atomicAdd(&shared_buffered_count[dst_buffer], 1);
          if (__builtin_expect(buffer_pos < MAX_BUFFERED_ITEMS, 1)) {
            buffered_indices[dst_buffer][buffer_pos] = idx;
            const int next_bit_offset = bit_offset - 8;
            const int next_bin = (fp32_bits >> next_bit_offset) & 0xFF;
            // 中文注释：同时构建下一轮的直方图（基于下一个字节）。
            atomicAdd(&shared_histogram[0][next_bin], 1);
          }
        }
      }
    }
    __syncthreads();
  }
}

// ============================================================================
// Inter-CTA sync primitives
// ============================================================================

// 中文注释：CTA 间同步原语。
// CUDA 的 __syncthreads() 只能同步同一 block 内的线程。
// 当多个 CTA 协作处理同一行时（Large 路径），需要通过全局内存实现 CTA 间同步。
// 这里使用 acquire/release 语义的内存操作实现轻量级屏障：
//   - ld_acquire: 带 acquire 语义的加载，确保后续读取看到屏障之前的数据。
//   - red_release: 带 release 语义的原子加，确保之前写入对其他 CTA 可见。
//   - st_release: 带 release 语义的存储，用于初始化/重置计数器。
//   - wait_ge: 自旋等待直到计数器 >= 目标值（仅 lane 0 自旋，其余线程等待 __syncthreads）。

// 中文注释：带 acquire 语义的全局内存加载。
// Volta 及以上架构使用原生 acquire 指令；旧架构退化为普通加载（.cg）。
__device__ __forceinline__ int ld_acquire(int* ptr) {
  int state = 0;
#if (__CUDA_ARCH__ >= 700)
  asm volatile("ld.global.acquire.gpu.b32 %0, [%1];\n"
               : "=r"(state)
               : "l"(ptr));
#else
  asm volatile("ld.cg.global.b32 %0, [%1];\n" : "=r"(state) : "l"(ptr));
#endif
  return state;
}

// 中文注释：带 release 语义的原子加。
// 先执行 fence 保证之前的写入全局可见，再执行原子加作为到达信号。
__device__ __forceinline__ void red_release(int* ptr, int val) {
#if (__CUDA_ARCH__ >= 700)
  asm volatile("fence.acq_rel.gpu;\n");
  asm volatile("red.relaxed.gpu.global.add.s32 [%0], %1;\n"
               :
               : "l"(ptr), "r"(val));
#else
  __threadfence();
  atomicAdd(ptr, val);
#endif
}

// 中文注释：带 release 语义的全局内存存储。
// 用于将计数器重置为特定值（如 output_counter = 0），确保其他 CTA 看到重置后的值。
__device__ __forceinline__ void st_release(int* ptr, int val) {
#if (__CUDA_ARCH__ >= 700)
  asm volatile("fence.acq_rel.gpu;\n");
  asm volatile("st.release.gpu.global.b32 [%0], %1;\n" : : "l"(ptr), "r"(val));
#else
  __threadfence();
  atomicExch(ptr, val);
#endif
}

// 中文注释：自旋等待屏障函数。
// 仅由每个 warp 的 lane 0 执行自旋等待（减少全局内存流量），
// 其余线程通过 __syncthreads() 等待 lane 0 完成。
// 当 arrival_counter >= target_val 时，说明所有 CTA 都已到达。
__device__ __forceinline__ void wait_ge(int* ptr, int target_val,
                                        int thread_idx) {
  if (thread_idx == 0) {
#pragma unroll 1
    while (ld_acquire(ptr) < target_val) {
    }
  }
  __syncthreads();
}

// ============================================================================
// Large path: multi-CTA radix select for sequences > 64K
//
// Each row is processed by a group of CTAs. Each CTA loads its chunk into
// shared memory as ordered uint32, then participates in 4 rounds of
// coordinated radix select via global-memory histograms and barriers.
// ============================================================================

// ============================================================================
// Multi-CTA cooperative RadixTopK for a single large row.
// Adapted from https://github.com/flashinfer-ai/flashinfer/pull/2215
// ============================================================================

// 中文注释：Large 路径的多 CTA 协作 RadixTopK 函数。
// 当序列长度 > 65536 时，单个 CTA 无法在共享内存中容纳所有数据，
// 因此将一行的数据分成多个 chunk，由多个 CTA 并行处理。
//
// 整体流程（三个阶段）：
//   阶段 1：加载（Load）
//     每个 CTA 将自己负责的 chunk 从全局内存加载到共享内存，
//     同时将 float32 转换为有序 uint32（convert_to_uint32_v2）。
//   阶段 2：4 轮基数选择（Radix Select）
//     所有 CTA 协作进行 4 轮基数细化：
//       a) 每个 CTA 在本地直方图中统计自己 chunk 的桶分布
//       b) 将本地直方图累加到全局直方图（全局内存，原子操作）
//       c) 所有 CTA 到达屏障后，计算全局后缀和
//       d) 定位新的阈值桶，更新 prefix 和 remaining_k
//   阶段 3：收集（Collect）
//     每个 CTA 收集自己 chunk 中 > pivot 的元素索引到输出数组，
//     然后收集 == pivot 的元素填满剩余槽位。
//
// 参数说明：
//   row_input: 当前行的 logits 起始地址
//   row_output: 当前行的输出索引起始地址
//   seq_len: 当前行的有效长度
//   my_chunk_start: 当前 CTA 负责的 chunk 起始偏移
//   chunk_size: 每个 CTA 负责的元素数量
//   local_histogram: 本地直方图（共享内存，每个 CTA 独立）
//   suffix_sum: 后缀和数组（共享内存）
//   shared_scalars: 共享标量（prefix, remaining_k, threshold, final_k）
//   shared_ordered: 转换后的有序 uint32 数据（共享内存）
//   state: CTA 间协调状态（全局内存，RadixRowState）
//   cta_in_group: 当前 CTA 在组内的编号
//   ctas_per_group: 每组的 CTA 数量
//   barrier_phase: 屏障阶段计数器（用于计算到达目标值）
//   iter: 当前迭代编号（用于计算全局轮次）
//   tx: 线程在 block 内的编号
template <int TopK, uint32_t VEC_SIZE>
__device__ void radix_topk(const float* __restrict__ row_input,
                           int32_t* __restrict__ row_output, uint32_t seq_len,
                           uint32_t my_chunk_start, uint32_t chunk_size,
                           uint32_t* local_histogram, uint32_t* suffix_sum,
                           uint32_t* shared_scalars, uint32_t* shared_ordered,
                           RadixRowState* state, uint32_t cta_in_group,
                           uint32_t ctas_per_group, int& barrier_phase,
                           uint32_t iter, uint32_t tx) {
  const uint32_t my_chunk_end = (my_chunk_start + chunk_size < seq_len)
                                    ? my_chunk_start + chunk_size
                                    : seq_len;
  const uint32_t actual_chunk_size =
      (my_chunk_start < seq_len) ? (my_chunk_end - my_chunk_start) : 0;

  // -- Stage 1: Load chunk to shared memory as ordered uint32 --
  // 中文注释：阶段 1 —— 将当前 CTA 负责的 chunk 加载到共享内存。
  // 同时将 float32 转换为有序 uint32，使后续的整数比较等价于浮点比较。
  // 使用向量化加载（float4/float2）提高内存带宽利用率。
  {
    const uint32_t aligned_size = (actual_chunk_size / VEC_SIZE) * VEC_SIZE;

    for (uint32_t i = tx * VEC_SIZE; i < aligned_size;
         i += kThreadsPerBlock * VEC_SIZE) {
      const float* src = row_input + my_chunk_start + i;
      if constexpr (VEC_SIZE == 4) {
        float4 v = *reinterpret_cast<const float4*>(src);
        shared_ordered[i] = convert_to_uint32_v2(v.x);
        shared_ordered[i + 1] = convert_to_uint32_v2(v.y);
        shared_ordered[i + 2] = convert_to_uint32_v2(v.z);
        shared_ordered[i + 3] = convert_to_uint32_v2(v.w);
      } else if constexpr (VEC_SIZE == 2) {
        float2 v = *reinterpret_cast<const float2*>(src);
        shared_ordered[i] = convert_to_uint32_v2(v.x);
        shared_ordered[i + 1] = convert_to_uint32_v2(v.y);
      } else {
        shared_ordered[i] = convert_to_uint32_v2(*src);
      }
    }
    for (uint32_t i = aligned_size + tx; i < actual_chunk_size;
         i += kThreadsPerBlock) {
      shared_ordered[i] = convert_to_uint32_v2(row_input[my_chunk_start + i]);
    }
  }
  __syncthreads();

  // -- Init radix select state --
  // 中文注释：初始化基数选择状态。
  // prefix = 0：当前已确定的前缀为空（还没有任何高位匹配）。
  // remaining_k = TopK：还需要选出 K 个元素。
  if (tx == 0) {
    shared_scalars[0] = 0;     // prefix
    shared_scalars[1] = TopK;  // remaining_k
  }
  __syncthreads();

  // -- Initial barrier --
  // 中文注释：初始屏障——等待所有 CTA 完成数据加载。
  // 每个 CTA 的 lane 0 原子加 1 到 arrival_counter，
  // 然后自旋等待直到所有 ctas_per_group 个 CTA 都到达。
  if (tx == 0) {
    red_release(&state->arrival_counter, 1);
  }
  wait_ge(&state->arrival_counter,
          (barrier_phase + 1) * static_cast<int>(ctas_per_group), tx);
  barrier_phase++;
  __syncthreads();

  // 中文注释：由组内第一个 CTA 初始化输出计数器。
  if (cta_in_group == 0 && tx == 0) {
    st_release(&state->output_counter, 0);
  }

  // -- Stage 2: 4 rounds of radix select --
  for (uint32_t round = 0; round < 4; round++) {
    const uint32_t global_round = iter * 4 + round;
    const uint32_t shift = 24 - round * 8;
    const uint32_t prefix = shared_scalars[0];
    const uint32_t remaining_k = shared_scalars[1];

    uint32_t* current_hist = state->histogram[global_round % 3];
    uint32_t* next_hist = state->histogram[(global_round + 1) % 3];

    for (uint32_t i = tx; i < RADIX; i += kThreadsPerBlock) {
      local_histogram[i] = 0;
    }
    __syncthreads();

    for (uint32_t i = tx; i < actual_chunk_size; i += kThreadsPerBlock) {
      uint32_t ordered = shared_ordered[i];
      uint32_t mask = (round == 0) ? 0u : (~0u << (32 - round * 8));
      if ((ordered & mask) == prefix) {
        uint32_t bucket = (ordered >> shift) & 0xFF;
        atomicAdd(&local_histogram[bucket], 1);
      }
    }
    __syncthreads();

    for (uint32_t i = tx; i < RADIX; i += kThreadsPerBlock) {
      if (local_histogram[i] > 0) {
        atomicAdd(&current_hist[i], local_histogram[i]);
      }
    }

    if (cta_in_group == 0) {
      for (uint32_t i = tx; i < RADIX; i += kThreadsPerBlock) {
        next_hist[i] = 0;
      }
    }

    if (tx == 0) {
      red_release(&state->arrival_counter, 1);
    }
    wait_ge(&state->arrival_counter,
            (barrier_phase + 1) * static_cast<int>(ctas_per_group), tx);
    barrier_phase++;
    __syncthreads();

    for (uint32_t i = tx; i < RADIX; i += kThreadsPerBlock) {
      suffix_sum[i] = current_hist[i];
    }
    __syncthreads();

    for (uint32_t stride = 1; stride < RADIX; stride *= 2) {
      uint32_t val = 0;
      if (tx < RADIX) {
        val = suffix_sum[tx];
        if (tx + stride < RADIX) val += suffix_sum[tx + stride];
      }
      __syncthreads();
      if (tx < RADIX) suffix_sum[tx] = val;
      __syncthreads();
    }

    if (tx == 0) {
      shared_scalars[2] = 0;
      shared_scalars[3] = remaining_k;
    }
    __syncthreads();

    if (tx < RADIX) {
      uint32_t count_ge = suffix_sum[tx];
      uint32_t count_gt = (tx + 1 < RADIX) ? suffix_sum[tx + 1] : 0;
      if (count_ge >= remaining_k && count_gt < remaining_k) {
        shared_scalars[2] = tx;
        shared_scalars[3] = remaining_k - count_gt;
      }
    }
    __syncthreads();

    if (tx == 0) {
      shared_scalars[0] = prefix | (shared_scalars[2] << shift);
      shared_scalars[1] = shared_scalars[3];
    }
    __syncthreads();
  }  // end 4 radix rounds

  // -- Count local > pivot elements --
  const uint32_t ordered_pivot = shared_scalars[0];

  if (tx == 0) suffix_sum[0] = 0;
  __syncthreads();

  uint32_t my_gt_count = 0;
  for (uint32_t i = tx; i < actual_chunk_size; i += kThreadsPerBlock) {
    if (shared_ordered[i] > ordered_pivot) my_gt_count++;
  }
  for (int offset = 16; offset > 0; offset /= 2) {
    my_gt_count += __shfl_down_sync(0xffffffff, my_gt_count, offset);
  }
  if (tx % 32 == 0 && my_gt_count > 0) {
    atomicAdd(&suffix_sum[0], my_gt_count);
  }
  __syncthreads();
  const uint32_t local_gt_count = suffix_sum[0];

  // -- Stage 3: Collect top-k indices --
  if (tx == 0) {
    local_histogram[0] = 0;
    if (local_gt_count > 0) {
      local_histogram[1] =
          atomicAdd(&state->output_counter, static_cast<int>(local_gt_count));
    }
  }
  __syncthreads();

  for (uint32_t i = tx; i < actual_chunk_size; i += kThreadsPerBlock) {
    if (shared_ordered[i] > ordered_pivot) {
      uint32_t local_pos = atomicAdd(&local_histogram[0], 1);
      int pos = static_cast<int>(local_histogram[1]) + local_pos;
      row_output[pos] = static_cast<int32_t>(my_chunk_start + i);
    }
  }

  if (tx == 0) {
    red_release(&state->arrival_counter, 1);
  }
  wait_ge(&state->arrival_counter,
          (barrier_phase + 1) * static_cast<int>(ctas_per_group), tx);
  barrier_phase++;
  __syncthreads();

  for (uint32_t i = tx; i < actual_chunk_size; i += kThreadsPerBlock) {
    if (shared_ordered[i] == ordered_pivot) {
      int pos = atomicAdd(&state->output_counter, 1);
      if (pos < TopK) {
        row_output[pos] = static_cast<int32_t>(my_chunk_start + i);
      }
    }
  }
}

// ============================================================================
// Persistent kernel — BS≤32, decode/medium/large paths with RadixTopK
// BS>32 uses standalone histogram_256_buffered_topk (separate kernel,
// see filtered_topk.cuh)
// ============================================================================

template <int TopK = 2048, uint32_t VEC_SIZE = 1>
__global__ void __launch_bounds__(kThreadsPerBlock, 2)
    persistent_topk_kernel(PersistentTopKParams params) {
  const uint32_t tx = threadIdx.x;
  extern __shared__ uint8_t smem_raw[];

  // ========================================================================
  // Group mode: multi-CTA groups with static round-robin row assignment.
  // Non-large rows: CTA-0 handles trivial/decode/medium.
  // Large rows: all CTAs in the group cooperate via RadixTopK.
  // ========================================================================
  const uint32_t ctas_per_group = params.ctas_per_group;
  const uint32_t group_id = blockIdx.x / ctas_per_group;
  const uint32_t cta_in_group = blockIdx.x % ctas_per_group;
  const uint32_t num_groups = gridDim.x / ctas_per_group;
  const uint32_t chunk_size = params.chunk_size;

  if (blockIdx.x >= num_groups * ctas_per_group) return;

  // Early exit: non-CTA-0 threads are never needed if no large rows exist
  if (cta_in_group != 0 && params.max_seq_len <= RADIX_THRESHOLD) return;

  uint32_t* local_histogram = reinterpret_cast<uint32_t*>(smem_raw);
  uint32_t* suffix_sum = local_histogram + RADIX;
  uint32_t* shared_scalars = suffix_sum + RADIX;
  uint32_t* shared_ordered =
      reinterpret_cast<uint32_t*>(smem_raw + kFixedSmemLarge);

  // RadixRowState for multi-CTA cooperative radix.
  // Zero-initialization is done host-side via cudaMemsetAsync in topk.cu
  // before launch — that gives a stream-ordered happens-before edge for all
  // CTAs, which the previous in-kernel init (CTA-0 only + intra-CTA
  // __syncthreads) did not provide and which manifested as a race against
  // CTA-1+'s first red_release on arrival_counter.
  RadixRowState* state = &params.row_states[group_id];

  int barrier_phase = 0;
  const uint32_t total_iters = (params.num_rows + num_groups - 1) / num_groups;

  for (uint32_t iter = 0; iter < total_iters; iter++) {
    // Static round-robin: all CTAs in the group implicitly agree on the row
    uint32_t row_idx = group_id + iter * num_groups;
    if (row_idx >= params.num_rows) break;

    const uint32_t seq_len = params.lengths[row_idx];
    int32_t* row_output = params.output + row_idx * params.top_k;
    const float* row_input = params.input + row_idx * params.stride;

    if (seq_len <= RADIX_THRESHOLD) {
      if (cta_in_group == 0) {
        if (seq_len <= static_cast<uint32_t>(TopK)) {
          // Trivial case: seq_len <= TopK
          for (uint32_t i = tx; i < static_cast<uint32_t>(TopK);
               i += kThreadsPerBlock) {
            row_output[i] = (i < seq_len) ? static_cast<int32_t>(i) : -1;
          }
        } else if (seq_len <= static_cast<uint32_t>(HIST2048_THRESHOLD)) {
          histogram_2048_topk<TopK>(row_input, row_output, seq_len);
        } else {
          histogram_256_topk<TopK>(row_input, row_output, 0, seq_len);
        }
      }
      continue;
    }

    const uint32_t my_chunk_start = cta_in_group * chunk_size;
    radix_topk<TopK, VEC_SIZE>(
        row_input, row_output, seq_len, my_chunk_start, chunk_size,
        local_histogram, suffix_sum, shared_scalars, shared_ordered, state,
        cta_in_group, ctas_per_group, barrier_phase, iter, tx);
  }
}

}  // namespace persistent

// ============================================================================
// FlashInfer FilteredTopK (BS>32 dispatch) — float32 only.
// Extracted from flashinfer_topk.cuh. Lives in namespace vllm (not persistent).
// Adapted from https://github.com/flashinfer-ai/flashinfer/pull/2215
// ============================================================================

#define FLASHINFER_CUDA_CALL(func, ...) \
  {                                     \
    cudaError_t e = (func);             \
    if (e != cudaSuccess) {             \
      return e;                         \
    }                                   \
  }

#define FLASHINFER_INLINE inline __attribute__((always_inline)) __device__

template <typename T, size_t N>
struct vec_t {
  T data[N];

  FLASHINFER_INLINE T& operator[](size_t i) { return data[i]; }
  FLASHINFER_INLINE const T& operator[](size_t i) const { return data[i]; }

  FLASHINFER_INLINE void cast_load(const T* ptr) {
#pragma unroll
    for (size_t i = 0; i < N; ++i) {
      data[i] = ptr[i];
    }
  }

  FLASHINFER_INLINE void cast_store(T* ptr) const {
#pragma unroll
    for (size_t i = 0; i < N; ++i) {
      ptr[i] = data[i];
    }
  }
};
#undef FLASHINFER_INLINE

// FilteredTopK traits for different data types
template <typename DType>
struct FilteredTopKTraits;

// Specialization for float (32-bit): coarse histogram uses FP16 high 8 bits, 4
// refinement rounds
template <>
struct FilteredTopKTraits<float> {
  using OrderedType = uint32_t;
  static constexpr int NUM_REFINE_ROUNDS = 4;
  static constexpr int FIRST_REFINE_SHIFT = 24;

  __device__ __forceinline__ static uint8_t ToCoarseKey(float x) {
    // Convert to FP16 representation and extract high 8 bits
    __half h = __float2half_rn(x);
    uint16_t bits = __half_as_ushort(h);
    uint16_t key = (bits & 0x8000) ? static_cast<uint16_t>(~bits)
                                   : static_cast<uint16_t>(bits | 0x8000);
    return static_cast<uint8_t>(key >> 8);
  }

  __device__ __forceinline__ static OrderedType ToOrdered(float x) {
    uint32_t bits = __float_as_uint(x);
    return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
  }
};

constexpr uint32_t FILTERED_TOPK_BLOCK_THREADS = 1024;
constexpr uint32_t FILTERED_TOPK_SMEM_INPUT_SIZE =
    16 * 1024;  // 16K indices per buffer
constexpr size_t FILTERED_TOPK_SMEM_DYNAMIC =
    sizeof(int) * 2 * FILTERED_TOPK_SMEM_INPUT_SIZE;  // 128KB

/*!
 * \brief Filtered Top-K kernel for ragged sequences.
 *
 * \tparam DType Data type (float, half, nv_bfloat16)
 * \tparam IdType Index type (int32_t)
 * \tparam VEC_SIZE Vector size for input loads (1, 2, 4, or 8)
 */
template <typename DType, typename IdType, int VEC_SIZE, uint32_t MAX_K = 2048>
__global__ void __launch_bounds__(FILTERED_TOPK_BLOCK_THREADS)
    FilteredTopKUnifiedKernel(const DType* __restrict__ input,
                              IdType* __restrict__ output,
                              const IdType* __restrict__ lengths,
                              uint32_t num_rows, uint32_t top_k,
                              uint32_t max_len) {
  constexpr uint32_t BLOCK_SIZE = FILTERED_TOPK_BLOCK_THREADS;
  constexpr int RADIX = 256;
  constexpr int SMEM_INPUT_SIZE = FILTERED_TOPK_SMEM_INPUT_SIZE;

  const uint32_t bid = blockIdx.x;
  const int tx = threadIdx.x;

  if (bid >= num_rows) return;

  const int length =
      (lengths != nullptr) ? lengths[bid] : static_cast<int>(max_len);
  const DType* score = input + bid * max_len;
  IdType* dst = output + bid * top_k;

  // Trivial case: length <= top_k
  if (length <= static_cast<int>(top_k)) {
    for (int i = tx; i < static_cast<int>(top_k); i += BLOCK_SIZE) {
      dst[i] = (i < length) ? static_cast<IdType>(i) : static_cast<IdType>(-1);
    }
    return;
  }

  // Static shared memory
  alignas(128) __shared__ int s_histogram_buf[2][RADIX + 128];
  alignas(128) __shared__ int s_counter;
  alignas(128) __shared__ int s_threshold_bin_id;
  alignas(128) __shared__ int s_num_input[2];
  alignas(128) __shared__ int s_indices[MAX_K];

  auto& s_histogram = s_histogram_buf[0];

  // Dynamic shared memory for input double buffer
  extern __shared__ int s_input_idx[][SMEM_INPUT_SIZE];

  using Traits = FilteredTopKTraits<DType>;
  int topk = top_k;

  // Stage 1: 8-bit coarse histogram with vectorized loads
  if (tx < RADIX + 1) s_histogram[tx] = 0;
  __syncthreads();

  vec_t<DType, VEC_SIZE> score_vec;

  const int aligned_length = (length / VEC_SIZE) * VEC_SIZE;
#pragma unroll 2
  for (int base = tx * VEC_SIZE; base < aligned_length;
       base += BLOCK_SIZE * VEC_SIZE) {
    score_vec.cast_load(&score[base]);
#pragma unroll
    for (int j = 0; j < VEC_SIZE; ++j) {
      const auto bin = Traits::ToCoarseKey(score_vec[j]);
      atomicAdd(&s_histogram[bin], 1);
    }
  }
  // Handle tail
  for (int i = aligned_length + tx; i < length; i += BLOCK_SIZE) {
    const auto bin = Traits::ToCoarseKey(score[i]);
    atomicAdd(&s_histogram[bin], 1);
  }
  __syncthreads();

  // Suffix sum
  const auto run_cumsum = [&]() {
#pragma unroll 8
    for (int i = 0; i < 8; ++i) {
      if (tx < RADIX) {
        const auto j = 1 << i;
        const auto k = i & 1;
        auto value = s_histogram_buf[k][tx];
        if (tx < RADIX - j) {
          value += s_histogram_buf[k][tx + j];
        }
        s_histogram_buf[k ^ 1][tx] = value;
      }
      __syncthreads();
    }
  };

  run_cumsum();
  if (tx < RADIX && s_histogram[tx] > topk && s_histogram[tx + 1] <= topk) {
    s_threshold_bin_id = tx;
    s_num_input[0] = 0;
    s_counter = 0;
  }
  __syncthreads();

  const auto threshold_bin = s_threshold_bin_id;
  topk -= s_histogram[threshold_bin + 1];

  constexpr int NUM_ROUNDS = Traits::NUM_REFINE_ROUNDS;
  constexpr int FIRST_SHIFT = Traits::FIRST_REFINE_SHIFT;

  if (topk == 0) {
    // Collect indices where bin > threshold
#pragma unroll 2
    for (int base = tx * VEC_SIZE; base < aligned_length;
         base += BLOCK_SIZE * VEC_SIZE) {
      score_vec.cast_load(&score[base]);
#pragma unroll
      for (int j = 0; j < VEC_SIZE; ++j) {
        const auto bin = static_cast<int>(Traits::ToCoarseKey(score_vec[j]));
        if (bin > threshold_bin) {
          const auto pos = atomicAdd(&s_counter, 1);
          s_indices[pos] = base + j;
        }
      }
    }
    // Handle tail
    for (int i = aligned_length + tx; i < length; i += BLOCK_SIZE) {
      const auto bin = static_cast<int>(Traits::ToCoarseKey(score[i]));
      if (bin > threshold_bin) {
        const auto pos = atomicAdd(&s_counter, 1);
        s_indices[pos] = i;
      }
    }
    __syncthreads();
  } else {
    __syncthreads();
    if (tx < RADIX + 1) s_histogram[tx] = 0;
    __syncthreads();

    // Filter + histogram for refinement
    auto filter_and_add_to_histogram = [&](auto raw_input, int index) {
      const auto bin = static_cast<int>(Traits::ToCoarseKey(raw_input));
      if (bin > threshold_bin) {
        const auto pos = atomicAdd(&s_counter, 1);
        s_indices[pos] = index;
      } else if (bin == threshold_bin) {
        const auto pos = atomicAdd(&s_num_input[0], 1);
        if (__builtin_expect(pos < SMEM_INPUT_SIZE, 1)) {
          s_input_idx[0][pos] = index;
          const auto ordered = Traits::ToOrdered(raw_input);
          const auto sub_bin = (ordered >> FIRST_SHIFT) & 0xFF;
          atomicAdd(&s_histogram[sub_bin], 1);
        }
      }
    };
#pragma unroll 2
    for (int base = tx * VEC_SIZE; base < aligned_length;
         base += BLOCK_SIZE * VEC_SIZE) {
      score_vec.cast_load(&score[base]);
#pragma unroll
      for (int j = 0; j < VEC_SIZE; ++j) {
        filter_and_add_to_histogram(score_vec[j], base + j);
      }
    }
    // Handle tail
    for (int i = aligned_length + tx; i < length; i += BLOCK_SIZE) {
      filter_and_add_to_histogram(score[i], i);
    }
    __syncthreads();

    // Stage 2: refine with 8bit radix passes
#pragma unroll
    for (int round = 0; round < NUM_ROUNDS; ++round) {
      __shared__ int s_last_remain;
      const auto r_idx = round % 2;

      const auto _raw_num_input = s_num_input[r_idx];
      const auto num_input =
          (_raw_num_input < SMEM_INPUT_SIZE) ? _raw_num_input : SMEM_INPUT_SIZE;

      run_cumsum();
      if (tx < RADIX && s_histogram[tx] > topk && s_histogram[tx + 1] <= topk) {
        s_threshold_bin_id = tx;
        s_num_input[r_idx ^ 1] = 0;
        s_last_remain = topk - s_histogram[tx + 1];
      }
      __syncthreads();

      const auto threshold = s_threshold_bin_id;
      topk -= s_histogram[threshold + 1];

      const int offset = FIRST_SHIFT - round * 8;
      const bool is_last_round = (round == NUM_ROUNDS - 1);

      if (topk == 0) {
        for (int i = tx; i < num_input; i += BLOCK_SIZE) {
          const auto idx = s_input_idx[r_idx][i];
          const auto bin = (Traits::ToOrdered(score[idx]) >> offset) & 0xFF;
          if (static_cast<int>(bin) > threshold) {
            const auto pos = atomicAdd(&s_counter, 1);
            s_indices[pos] = idx;
          }
        }
        __syncthreads();
        break;
      } else {
        __syncthreads();
        if (tx < RADIX + 1) s_histogram[tx] = 0;
        __syncthreads();
        for (int i = tx; i < num_input; i += BLOCK_SIZE) {
          const auto idx = s_input_idx[r_idx][i];
          const auto raw_input = score[idx];
          const auto bin = (Traits::ToOrdered(raw_input) >> offset) & 0xFF;
          if (static_cast<int>(bin) > threshold) {
            const auto pos = atomicAdd(&s_counter, 1);
            s_indices[pos] = idx;
          } else if (static_cast<int>(bin) == threshold) {
            if (is_last_round) {
              const auto pos = atomicAdd(&s_last_remain, -1);
              if (pos > 0) {
                s_indices[top_k - pos] = idx;
              }
            } else {
              const auto pos = atomicAdd(&s_num_input[r_idx ^ 1], 1);
              if (__builtin_expect(pos < SMEM_INPUT_SIZE, 1)) {
                s_input_idx[r_idx ^ 1][pos] = idx;
                const auto bin32 = Traits::ToOrdered(raw_input);
                const auto sub_bin = (bin32 >> (offset - 8)) & 0xFF;
                atomicAdd(&s_histogram[sub_bin], 1);
              }
            }
          }
        }
        __syncthreads();
      }
    }
  }

  // Output phase - mode-specific
#pragma unroll 2
  for (int base = tx; base < static_cast<int>(top_k); base += BLOCK_SIZE) {
    const int idx = s_indices[base];
    dst[base] = static_cast<IdType>(idx);
  }
}

// Helper to compute GCD for VEC_SIZE selection
constexpr uint32_t gcd(uint32_t a, uint32_t b) {
  while (b != 0) {
    uint32_t t = b;
    b = a % b;
    a = t;
  }
  return a;
}

// Compute optimal VEC_SIZE based on max_len and dtype
// Returns 1, 2, 4, or 8
template <typename DType>
constexpr int ComputeFilteredTopKVecSize(uint32_t max_len) {
  constexpr int MAX_VEC = 16 / sizeof(DType);  // 4 for float32, 8 for fp16/bf16
  // Use GCD to find largest power-of-2 divisor
  const uint32_t g = gcd(max_len, static_cast<uint32_t>(MAX_VEC));
  return static_cast<int>(g);
}

template <typename DType, typename IdType, uint32_t MAX_K = 2048>
cudaError_t FilteredTopKRaggedTransform(const DType* input,
                                        IdType* output_indices,
                                        const IdType* lengths,
                                        uint32_t num_rows, uint32_t top_k_val,
                                        uint32_t max_len,
                                        cudaStream_t stream = 0) {
  constexpr size_t smem_size = FILTERED_TOPK_SMEM_DYNAMIC;
  constexpr int MAX_VEC = 16 / sizeof(DType);

  dim3 grid(num_rows);
  dim3 block(FILTERED_TOPK_BLOCK_THREADS);
  void* args[] = {&input,    &output_indices, &lengths,
                  &num_rows, &top_k_val,      &max_len};

  const int vec_size = ComputeFilteredTopKVecSize<DType>(max_len);

#define DISPATCH_VEC_SIZE(VS)                                               \
  if (vec_size == VS) {                                                     \
    auto kernel = FilteredTopKUnifiedKernel<DType, IdType, VS, MAX_K>;      \
    FLASHINFER_CUDA_CALL(cudaFuncSetAttribute(                              \
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));   \
    FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, grid, block, args, \
                                          smem_size, stream));              \
    return cudaSuccess;                                                     \
  }

  DISPATCH_VEC_SIZE(1)
  DISPATCH_VEC_SIZE(2)
  DISPATCH_VEC_SIZE(4)
  if constexpr (MAX_VEC >= 8) {
    DISPATCH_VEC_SIZE(8)
  }
#undef DISPATCH_VEC_SIZE

  return cudaSuccess;
}

}  // namespace vllm

#endif  // PERSISTENT_TOPK_CUH_
