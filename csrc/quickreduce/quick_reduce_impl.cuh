// =============================================================================
// 中文注释：QuickReduce AllReduce 核心实现 (quick_reduce_impl.cuh)
// =============================================================================
// 本文件实现了 QuickReduce 的核心通信逻辑，包括：
//
// 1. 编解码器（Codec）：定义数据如何序列化/反序列化用于跨 GPU 通信
//    - CodecFP: 全精度（fp16/bf16），直接传输原始数据
//    - CodecQ4: 4-bit 量化，将 fp16 数据压缩为 4-bit 整数，带宽节省约 4x
//    - CodecQ6: 6-bit 量化，介于 Q4 和 Q8 之间的精度/带宽权衡
//    - CodecQ8: 8-bit 量化，将 fp16 数据压缩为 8-bit 整数，带宽节省约 2x
//
// 2. AllReduceTwoshot: 两阶段 AllReduce 算法实现
//    - Phase-1A: 每个 rank 将自己的数据分段发送给负责该段的 rank
//    - Phase-1B: 每个 rank 归约收到的所有分段数据（scatter-reduce）
//    - Phase-2A: 每个 rank 将归约结果广播给所有其他 rank
//    - Phase-2B: 每个 rank 收集所有段的最终结果（all-gather）
//
// 优化亮点：
//    - 支持量化通信：在传输前量化、接收后反量化，减少跨 GPU 带宽需求
//    - 使用 AMD buffer resource 进行高效的内存访问
//    - 使用原子标志位进行 rank 间同步，避免昂贵的全局 barrier
//    - 对 bf16 输入可选择先转为 half 再计算（cast_bf2half），利用 half 的硬件指令优势
// =============================================================================

#pragma once

#include <hip/hip_runtime.h>
#include "base.h"

namespace quickreduce {

// 中文注释：编解码器基类。所有 Codec（FP/Q4/Q6/Q8）的公共基类
// 管理线程编号、rank 编号和量化线程组的 leader 线程
// set_fp16_ovfl(true) 确保 fp16 运算溢出时饱和而非产生 NaN
struct CodecBase {
  const int thread;
  const int rank;
  const int group_leader;
  __quickreduce_device_inline__ CodecBase(int thread, int rank)
      : thread(thread),
        rank(rank),
        group_leader((threadIdx.x / kThreadGroupSize) * kThreadGroupSize) {
    set_fp16_ovfl(true);
  }
};

// 中文注释：全精度编解码器（CodecFP）
// 直接传输原始 fp16/bf16 数据，不进行量化
// 优点：无精度损失；缺点：带宽消耗最大
// 数据布局：每个 rank 的数据按 tile 粒度连续存放
// Default full precision codec.
template <typename T, int world_size>
struct CodecFP : public CodecBase {
  static constexpr int kWorldSize = world_size;
  static constexpr int kRankAtoms = kAtoms / kWorldSize;

  // Codec tile size process by this workgroup.
  // Each thread processes atoms of f16x8_t (16B).
  // 中文注释：单个 rank 在一个 tile 中传输的字节数
  // = 256 线程 * (8/world_size) atoms/线程 * 16 字节/atom
  // 例如 world_size=4 时，每个 rank 传 8KB
  static constexpr int kRankTransmittedTileSize =
      kBlockSize * kRankAtoms * sizeof(int32x4_t);
  static_assert(kRankTransmittedTileSize % 16 == 0,
                "kRankTransmittedTileSize must be 16B aligned.");

  // Total tile size for the collective communication.
  // 中文注释：所有 rank 的一个 tile 的总传输字节数 = 单 rank 大小 * world_size
  static constexpr int kTransmittedTileSize =
      kRankTransmittedTileSize * kWorldSize;

  __quickreduce_device_inline__ CodecFP(int thread, int rank)
      : CodecBase(thread, rank) {}

  // 中文注释：将本地数据发送到通信缓冲区。使用 nontemporal store 避免污染 L2 cache
  // 数据按线程交错布局写入：第 i 个 atom 写到 send_buffer[thread + i * kAtomStride]
  __quickreduce_device_inline__ void send(int32x4_t* __restrict__ send_buffer,
                                          const int32x4_t* __restrict__ data) {
    for (int i = 0; i < kRankAtoms; i++) {
      __builtin_nontemporal_store(data[i], send_buffer + thread);
      send_buffer += kAtomStride;
    }
  }

  // 中文注释：从通信缓冲区接收数据。使用 nontemporal load 读取，不会缓存到 L2
  // recv_buffer 是一个指针的指针，调用后会自动推进到下一个 atom 的位置
  __quickreduce_device_inline__ void recv(int32x4_t** __restrict__ recv_buffer,
                                          int32x4_t* __restrict__ data) {
    for (int i = 0; i < kRankAtoms; i++) {
      data[i] = __builtin_nontemporal_load(*recv_buffer + thread);
      *recv_buffer += kAtomStride;
    }
  }
};

// Int4 symmetric quantization codec.
// We quantize the FP16 data to block-scaled Int4 in blocks of 4 *
// kThreadGroupSize.
// 中文注释：4-bit 对称量化编解码器（CodecQ4）
// 将 fp16/bf16 数据量化为 4-bit 整数进行传输，接收后反量化还原
// 量化策略：
//   - 每 32 个值（8 线程 * 4 值/线程）共享一个 fp16 scale
//   - scale = abs_max * (-1/8)，然后 encoding_scale = 1/scale
//   - 量化公式：q = clamp(round(x * encoding_scale), -8, +7) + 8
//   - 反量化公式：x = (q - 8) * decoding_scale
// 带宽节省：相比 fp16，约节省 4x 带宽（4-bit 数据 + 少量 scale 开销）
template <typename T, int world_size>
struct CodecQ4 : public CodecBase {
  static constexpr int kWorldSize = world_size;

  // Codec tile size process by this workgroup.
  // Each threads processes a fragment of fp16x8_t (16B),
  // into a int4x8_t (4B) and a fp16 scale shared among 32 values.
  // 中文注释：Q4 量化后的 tile 内存布局：
  //   - 前 1024 字节：量化后的 4-bit 数据（256 线程 * 4 字节/线程）
  //   - 1024~1152 字节：32 个 fp16 scale（256 线程 / 8 线程每组 = 32 个 scale）
  //   - 每个 tile stride = 1152 字节
  static constexpr int kRankAtoms = kAtoms / kWorldSize;
  static constexpr int kRankTileStride = 1152;
  static constexpr int kRankTileScaleOffset = 1024;
  static constexpr int kRankTransmittedTileSize = kRankTileStride * kRankAtoms;
  static_assert(kRankTransmittedTileSize % 16 == 0,
                "kRankTransmittedTileSize must be 16B aligned.");

  static constexpr int kRankBufferTileStride =
      kRankTileStride / sizeof(int32x4_t);

  // Total tile size for the collective communication.
  static constexpr int kTransmittedTileSize =
      kRankTransmittedTileSize * kWorldSize;

  // Constants configuration
  // 中文注释：Q4 量化的常量配置。根据数据类型（half 或 bf16）选择不同的位模式
  //   - kScaleFactor: 缩放因子 = -1/8.0，将 abs_max 映射到 [-8, +7] 范围
  //   - kScaleEpsilon: 极小值 1e-7，防止除零
  //   - kRangeMin/kRangeMax: 量化范围 [-8, +7]（4-bit 有符号整数）
  //   - kRangeBias: 偏移量 +8，将有符号值转为无符号存储 [0, 15]

  // {-1/8.0h, -1/8.0h}, f16x2_t
  static constexpr int kScaleFactor =
      std::is_same<T, half>::value ? 0xB000B000 : 0xBE00BE00;

  // {1e-7, 1e-7}, f16x2_t
  static constexpr int kScaleEpsilon =
      std::is_same<T, half>::value ? 0x00010001 : 0x33D733D7;

  // {-8, -8}, f16x2_t
  static constexpr int kRangeMin =
      std::is_same<T, half>::value ? 0xC800C800 : 0xC100C100;

  // {+7, +7}, f16x2_t
  static constexpr int kRangeMax =
      std::is_same<T, half>::value ? 0x47004700 : 0x40E040E0;

  // {+8, +8}, int16x2_t
  static constexpr int kRangeBias = 0x00080008;

  __quickreduce_device_inline__ CodecQ4(int thread, int rank)
      : CodecBase(thread, rank) {}

  // 中文注释：Q4 量化发送函数。将 fp16/bf16 数据量化为 4-bit 后写入通信缓冲区
  // 量化流程（对每个 atom，即 8 个 fp16 值）：
  //   步骤 1: 计算线程组内 32 个值的绝对值最大值（group_abs_max）
  //   步骤 2: 计算 decoding_scale = abs_max * (-1/8)
  //           计算 encoding_scale = 1 / (decoding_scale + epsilon)
  //   步骤 3: 量化 q = clamp(round(x * encoding_scale), -8, +7)
  //   步骤 4: 偏移 q += 8，转为无符号 [0, 15]
  //   步骤 5: 将 8 个 4-bit 值打包成一个 int32（低 4 位有效）
  //   步骤 6: 每个线程写 4 字节量化数据；只有 group_leader 写 scale
  __quickreduce_device_inline__ void send(int32x4_t* __restrict__ send_buffer,
                                          const int32x4_t* __restrict__ data) {
    for (int k = 0; k < kRankAtoms; k++) {
      int32x4_t const atom = data[k];

      // Compute the absolute maximum of the atom in the thread group
      // In 2 blocks of values, upper/lower halves of the f16x2_t
      int wblockmax = group_abs_max<T>(atom);

      // Derive scales
      int decoding_scale;
      int encoding_scale;
      decoding_scale = packed_mul<T>(wblockmax, kScaleFactor);
      encoding_scale = packed_add<T>(decoding_scale, kScaleEpsilon);
      encoding_scale = packed_rcp<T>(encoding_scale);

      // Apply scales to get quantized values
      int32x4_t w;
      for (int i = 0; i < 4; i++) {
        w[i] = packed_mul<T>(atom[i], encoding_scale);
        w[i] = packed_max<T>(w[i], kRangeMin);
        w[i] = packed_min<T>(w[i], kRangeMax);
      }

      // Convert from f16x2_t to uint16x2_t
      int32x4_t q;
      {
        int16_t* qi = reinterpret_cast<int16_t*>(&q);
        T* wh = reinterpret_cast<T*>(&w);
        for (int i = 0; i < 8; i++) qi[i] = (int16_t)rintf(T2float_cast(wh[i]));

        for (int i = 0; i < 4; i++) {
          q[i] = packed_add<int16_t>(q[i], kRangeBias);
        }
      }

      // Pack 8 x q4 into int32_t
      int qw = q[0] | (q[1] << 4) | (q[2] << 8) | (q[3] << 12);

      // Write quantized atom to send_buffer
      // note: only the group leader stores the scale
      uint8_t* atom_ptr =
          reinterpret_cast<uint8_t*>(send_buffer + k * kRankBufferTileStride);
      int32_t* qw_ptr = reinterpret_cast<int32_t*>(atom_ptr) + thread;
      int* qs_ptr = reinterpret_cast<int*>(atom_ptr + kRankTileScaleOffset) +
                    (thread / 8);

      __builtin_nontemporal_store(qw, qw_ptr);
      if (threadIdx.x == group_leader) {
        __builtin_nontemporal_store(decoding_scale, qs_ptr);
      }
    }
  }

  // 中文注释：Q4 反量化接收函数。从通信缓冲区读取 4-bit 量化数据并还原为 fp16/bf16
  // 反量化流程（对每个 atom）：
  //   步骤 1: 从通信缓冲区读取打包的 4-bit 数据（4 字节）和对应的 scale
  //   步骤 2: 解包每个 4-bit 值（通过位移和掩码提取）
  //   步骤 3: 对于 half 类型，使用巧妙的浮点位模式技巧：
  //           将 4-bit 值嵌入到 fp16 的尾数位，加上 {1024, 1024} 使其成为合法 fp16，
  //           再减去 {-1032, -1032} 得到原始整数值，避免了整数到浮点的转换指令
  //   步骤 4: 应用 decoding_scale 还原为原始 fp16/bf16 值
  __quickreduce_device_inline__ void recv(int32x4_t** __restrict__ recv_buffer,
                                          int32x4_t* __restrict__ data) {
    for (int k = 0; k < kRankAtoms; k++) {
      // Directly read quantized atom from recv_buffer
      uint8_t* atom_ptr = reinterpret_cast<uint8_t*>(*recv_buffer);
      int32_t* qw_ptr = reinterpret_cast<int32_t*>(atom_ptr) + thread;
      int* qs_ptr = reinterpret_cast<int*>(atom_ptr + kRankTileScaleOffset) +
                    (thread / 8);

      int32_t qw = __builtin_nontemporal_load(qw_ptr);
      int qs = __builtin_nontemporal_load(qs_ptr);

      *recv_buffer += kRankBufferTileStride;

      // Unpack q4 into f16x8_t
      int32x4_t w;
      {
        static constexpr uint kMask000F = 0x000F000F;
        static constexpr uint kHalf2_1024 =
            0x64006400;  // {1024.0, 1024.0}, fp16x2_t
        static uint constexpr kHalf2_1032 =
            0xE408E408;  // {-1032.0, -1032.0}, fp16x2_t

        for (int i = 0; i < 4; i++) {
          if constexpr (std::is_same<T, half>::value) {
            int32_t q4 = ((qw >> (i * 4)) & kMask000F) | kHalf2_1024;
            w[i] = packed_add<half>(q4, kHalf2_1032);
          } else {
            int32_t int16_2 = (qw >> (i * 4)) & kMask000F;
            int16_t low = static_cast<int16_t>(int16_2 & 0xFFFF);
            int16_t high = static_cast<int16_t>((int16_2 >> 16) & 0xFFFF);
            nv_bfloat16 bf_low = __float2bfloat16(static_cast<float>(low));
            nv_bfloat16 bf_high = __float2bfloat16(static_cast<float>(high));
            nv_bfloat162 bf2 = __halves2bfloat162(bf_low, bf_high);
            int32_t packed_bf16 = *reinterpret_cast<int32_t*>(&bf2);
            w[i] = packed_add<nv_bfloat16>(packed_bf16, kRangeMin);
          }
        }
      }

      // Apply decoding scales
      for (int i = 0; i < 4; i++) {
        w[i] = packed_mul<T>(w[i], qs);
      }

      data[k] = w;
    }
  }
};

// Int6 symmetric quantization codec.
// We quantize the FP16 data to block-scaled Int6 in blocks of 4 *
// kThreadGroupSize.
// 中文注释：6-bit 对称量化编解码器（CodecQ6）
// 将 fp16/bf16 数据量化为 6-bit 整数，是 Q4 和 Q8 之间的折中方案
// 数据布局：每个 atom 的 6-bit 数据分为两部分存储：
//   - 低 4 位（q4）: 存放在前 1024 字节区域（与 Q4 布局相同）
//   - 高 2 位（q2）: 存放在 1024~1536 字节区域
//   - scale: 存放在 1536~1664 字节区域
// 量化范围：[-32, +31]，比 Q4 的 [-8, +7] 精度更高
template <typename T, int world_size>
struct CodecQ6 : public CodecBase {
  static constexpr int kWorldSize = world_size;

  // Codec tile size process by this workgroup.
  // Each threads processes a fragment of fp16x8_t (16B),
  // into a int6x8_t (4B + 2B) and a fp16 scale shared among 32 values.
  static constexpr int kRankAtoms = kAtoms / kWorldSize;
  static constexpr int kRankTileStride = 1664;
  static constexpr int kRankTileQ2Offset = 1024;
  static constexpr int kRankTileScaleOffset = 1536;
  static constexpr int kRankTransmittedTileSize = kRankTileStride * kRankAtoms;
  static_assert(kRankTransmittedTileSize % 16 == 0,
                "kRankTransmittedTileSize must be 16B aligned.");

  static constexpr int kRankBufferTileStride =
      kRankTileStride / sizeof(int32x4_t);

  // Total tile size for the collective communication.
  static constexpr int kTransmittedTileSize =
      kRankTransmittedTileSize * kWorldSize;

  // Constants configuration

  // {-1/32.0h, -1/32.0h}, fp16x2_t
  static constexpr int kScaleFactor =
      std::is_same<T, half>::value ? 0xA800A800 : 0xBD00BD00;

  // {1e-7, 1e-7}, fp16x2_t
  static constexpr int kScaleEpsilon =
      std::is_same<T, half>::value ? 0x00010001 : 0x33D733D7;

  // {-32, -32}, fp16x2_t
  static constexpr int kRangeMin =
      std::is_same<T, half>::value ? 0xD000D000 : 0xC200C200;

  // {+31, +31}, fp16x2_t
  static constexpr int kRangeMax =
      std::is_same<T, half>::value ? 0x4FC04FC0 : 0x41F841F8;

  // {+32, +32}, int16x2_t
  static constexpr int kRangeBias = 0x00200020;

  __quickreduce_device_inline__ CodecQ6(int thread, int rank)
      : CodecBase(thread, rank) {}

  __quickreduce_device_inline__ void send(int32x4_t* __restrict__ send_buffer,
                                          const int32x4_t* __restrict__ data) {
    for (int k = 0; k < kRankAtoms; k++) {
      int32x4_t const atom = data[k];

      // Compute the absolute maximum of the atom in the thread group
      // In 2 blocks of values, upper/lower halves of the f16x2_t
      int wblockmax = group_abs_max<T>(atom);

      // Derive scales
      int decoding_scale;
      int encoding_scale;
      decoding_scale = packed_mul<T>(wblockmax, kScaleFactor);
      encoding_scale = packed_add<T>(decoding_scale, kScaleEpsilon);
      encoding_scale = packed_rcp<T>(encoding_scale);

      // Apply scales to get quantized values
      int32x4_t w;
      for (int i = 0; i < 4; i++) {
        w[i] = packed_mul<T>(atom[i], encoding_scale);
        w[i] = packed_max<T>(w[i], kRangeMin);
        w[i] = packed_min<T>(w[i], kRangeMax);
      }

      // Convert from f16x2_t to uint16x2_t
      int32x4_t q;
      {
        int16_t* qi = reinterpret_cast<int16_t*>(&q);
        T* wh = reinterpret_cast<T*>(&w);
        for (int i = 0; i < 8; i++) qi[i] = (int16_t)rintf(T2float_cast(wh[i]));

        for (int i = 0; i < 4; i++) {
          q[i] = packed_add<int16_t>(q[i], kRangeBias);
        }
      }

      // Pack 8 x q6 into int32_t + int16_t
      uint32_t q4w;
      uint16_t q2w = 0;
      q4w = (q[0] & 0x000F000F) | ((q[1] & 0x000F000F) << 4) |
            ((q[2] & 0x000F000F) << 8) | ((q[3] & 0x000F000F) << 12);
      {
        int16_t* tw = reinterpret_cast<int16_t*>(&q);
#pragma unroll
        for (int i = 0; i < 8; i++) {
          q2w |= (tw[i] >> 4) << (i * 2);
        }
      }
      // Write quantized atom to send_buffer
      // note: only the group leader stores the scale
      uint8_t* atom_ptr =
          reinterpret_cast<uint8_t*>(send_buffer + k * kRankBufferTileStride);
      uint32_t* q4w_ptr = reinterpret_cast<uint32_t*>(atom_ptr) + thread;
      uint16_t* q2w_ptr =
          reinterpret_cast<uint16_t*>(atom_ptr + kRankTileQ2Offset) + thread;
      int* qs_ptr = reinterpret_cast<int*>(atom_ptr + kRankTileScaleOffset) +
                    (thread / 8);

      __builtin_nontemporal_store(q4w, q4w_ptr);
      __builtin_nontemporal_store(q2w, q2w_ptr);
      if (threadIdx.x == group_leader) {
        __builtin_nontemporal_store(decoding_scale, qs_ptr);
      }
    }
  }

  __quickreduce_device_inline__ void recv(int32x4_t** __restrict__ recv_buffer,
                                          int32x4_t* __restrict__ data) {
    for (int k = 0; k < kRankAtoms; k++) {
      // Directly read quantized atom from recv_buffer
      uint8_t* atom_ptr = reinterpret_cast<uint8_t*>(*recv_buffer);
      uint32_t* q4w_ptr = reinterpret_cast<uint32_t*>(atom_ptr) + thread;
      uint16_t* q2w_ptr =
          reinterpret_cast<uint16_t*>(atom_ptr + kRankTileQ2Offset) + thread;
      int* qs_ptr = reinterpret_cast<int*>(atom_ptr + kRankTileScaleOffset) +
                    (thread / 8);

      uint32_t q4w = __builtin_nontemporal_load(q4w_ptr);
      uint16_t q2w = __builtin_nontemporal_load(q2w_ptr);
      int qs = __builtin_nontemporal_load(qs_ptr);

      *recv_buffer += kRankBufferTileStride;

      // Unpack q6 into fp16x8_t
      int32x4_t w;
      {
        static uint constexpr kMask000F = 0x000F000F;
        static uint constexpr kHalf2_1024 =
            0x64006400;  // {1024.0, 1024.0}, fp16x2_t
        static uint constexpr kHalf2_1056 =
            0xE420E420;  // {-1056.0, -1056.0}, fp16x2_t

#pragma unroll
        for (int i = 0; i < 4; i++) {
          int32_t q4 = q4w & kMask000F;
          int32_t q2 = (q2w & 0x3) | ((q2w & 0xC) << 14);
          q4w >>= 4;
          q2w >>= 4;
          if constexpr (std::is_same<T, half>::value) {
            int32_t q6 = q4 | (q2 << 4) | kHalf2_1024;
            asm volatile("v_pk_add_f16 %0, %1, %2"
                         : "=v"(w[i])
                         : "v"(q6), "v"(kHalf2_1056));
          } else {
            int32_t int16_2 = q4 | (q2 << 4);
            int16_t low = static_cast<int16_t>(int16_2 & 0xFFFF);
            int16_t high = static_cast<int16_t>((int16_2 >> 16) & 0xFFFF);

            nv_bfloat16 bf_low = __float2bfloat16(static_cast<float>(low));
            nv_bfloat16 bf_high = __float2bfloat16(static_cast<float>(high));
            nv_bfloat162 bf2 = __halves2bfloat162(bf_low, bf_high);
            int32_t packed_bf16 = *reinterpret_cast<int32_t*>(&bf2);
            w[i] = packed_add<nv_bfloat16>(packed_bf16, kRangeMin);
          }
        }
      }

      // Apply decoding scales
      for (int i = 0; i < 4; i++) {
        w[i] = packed_mul<T>(w[i], qs);
      }

      // That's pretty much it...
      data[k] = w;
    }
  }
};

// Int8 symmetric quantization codec.
// We quantize the FP16 data to block-scaled Int8 in blocks of 4 *
// kThreadGroupSize.
// 中文注释：8-bit 对称量化编解码器（CodecQ8）
// 将 fp16/bf16 数据量化为 8-bit 整数，精度最高但带宽节省最少（约 2x）
// 量化范围：[-128, +127]，量化误差最小
// 数据布局：
//   - 前 2048 字节：8-bit 量化数据（256 线程 * 8 字节/线程）
//   - 2048~2176 字节：32 个 fp16 scale
template <typename T, int world_size>
struct CodecQ8 : public CodecBase {
  static constexpr int kWorldSize = world_size;

  // Codec tile size process by this workgroup.
  // Each threads processes a fragment of f16x8_t (16B),
  // into a int8x8_t (8B) and a f16 scale shared among 32 values.
  static constexpr int kRankAtoms = kAtoms / kWorldSize;
  static constexpr int kRankTileStride = 2176;
  static constexpr int kRankTileScaleOffset = 2048;
  static constexpr int kRankTransmittedTileSize = kRankTileStride * kRankAtoms;
  static_assert(kRankTransmittedTileSize % 16 == 0,
                "kRankTileSize must be 16B aligned.");

  static constexpr int kRankBufferTileStride =
      kRankTileStride / sizeof(int32x4_t);

  // Total tile size for the collective communication.
  static constexpr int kTransmittedTileSize =
      kRankTransmittedTileSize * kWorldSize;

  // Constants configuration

  // {-1/128.0h, -1/128.0h}, f16x2_t
  static constexpr int kScaleFactor =
      std::is_same<T, half>::value ? 0xA000A000 : 0xBC00BC00;

  // {1e-7, 1e-7}, f16x2_t
  static constexpr int kScaleEpsilon =
      std::is_same<T, half>::value ? 0x00010001 : 0x33D733D7;

  // {-128, -128}, f16x2_t
  static constexpr int kRangeMin =
      std::is_same<T, half>::value ? 0xD800D800 : 0xC300C300;
  // {+127, +127}, f16x2_t
  static constexpr int kRangeMax =
      std::is_same<T, half>::value ? 0x57F057F0 : 0x42FE42FE;

  // {+128, +128}, int16x2_t
  static constexpr int kRangeBias = 0x00800080;

  __quickreduce_device_inline__ CodecQ8(int thread, int rank)
      : CodecBase(thread, rank) {}

  __quickreduce_device_inline__ void send(int32x4_t* __restrict__ send_buffer,
                                          int32x4_t const* __restrict__ data) {
    for (int k = 0; k < kRankAtoms; k++) {
      int32x4_t const atom = data[k];
      // Compute the absolute maximum of the atom in the thread group
      // In 2 blocks of values, upper/lower halves of the f16x2_t
      int wblockmax = group_abs_max<T>(atom);

      // Derive scales
      int decoding_scale;
      int encoding_scale;
      decoding_scale = packed_mul<T>(wblockmax, kScaleFactor);
      encoding_scale = packed_add<T>(decoding_scale, kScaleEpsilon);
      encoding_scale = packed_rcp<T>(encoding_scale);

      // Apply scales to get quantized values
      int32x4_t w;
      for (int i = 0; i < 4; i++) {
        w[i] = packed_mul<T>(atom[i], encoding_scale);
        w[i] = packed_max<T>(w[i], kRangeMin);
        w[i] = packed_min<T>(w[i], kRangeMax);
      }

      // Convert from f16x2_t to uint16x2_t
      int32x4_t q;
      {
        int16_t* qi = reinterpret_cast<int16_t*>(&q);
        T* wh = reinterpret_cast<T*>(&w);
        for (int i = 0; i < 8; i++) qi[i] = (int16_t)rintf(T2float_cast(wh[i]));

        for (int i = 0; i < 4; i++) {
          q[i] = packed_add<int16_t>(q[i], kRangeBias);
        }
      }

      // Pack 8 x q8 into int32x2_t
      int32x2_t qw;
      qw[0] = q[0] | (q[1] << 8);
      qw[1] = q[2] | (q[3] << 8);

      // Write quantized atom to send_buffer
      // note: only the group leader stores the scale
      uint8_t* atom_ptr =
          reinterpret_cast<uint8_t*>(send_buffer + k * kRankBufferTileStride);
      int32x2_t* qw_ptr = reinterpret_cast<int32x2_t*>(atom_ptr) + thread;
      int* qs_ptr = reinterpret_cast<int*>(atom_ptr + kRankTileScaleOffset) +
                    (thread / 8);

      __builtin_nontemporal_store(qw, qw_ptr);
      if (threadIdx.x == group_leader) {
        __builtin_nontemporal_store(decoding_scale, qs_ptr);
      }
    }
  }

  __quickreduce_device_inline__ void recv(int32x4_t** __restrict__ recv_buffer,
                                          int32x4_t* __restrict__ data) {
    for (int k = 0; k < kRankAtoms; k++) {
      // Directly read quantized atom from recv_buffer
      uint8_t* atom_ptr = reinterpret_cast<uint8_t*>(*recv_buffer);
      int32x2_t* qw_ptr = reinterpret_cast<int32x2_t*>(atom_ptr) + thread;
      int* qs_ptr = reinterpret_cast<int*>(atom_ptr + kRankTileScaleOffset) +
                    (thread / 8);

      int32x2_t qw = __builtin_nontemporal_load(qw_ptr);
      int qs = __builtin_nontemporal_load(qs_ptr);

      *recv_buffer += kRankBufferTileStride;

      // Unpack q8 into fp16x8_t
      int32x4_t w;
      {
        static uint constexpr kMask00FF = 0x00FF00FF;

        // {1024.0, 1024.0}, fp16x2_t
        static uint constexpr kHalf2_1024 = 0x64006400;

        // {-1152.0, -1152.0}, fp16x2_t
        static uint constexpr kHalf2_1152 = 0xE480E480;

#pragma unroll
        for (int i = 0; i < 4; i++) {
          if constexpr (std::is_same<T, half>::value) {
            int32_t q8 =
                ((qw[i / 2] >> ((i % 2) * 8)) & kMask00FF) | kHalf2_1024;
            w[i] = packed_add<half>(q8, kHalf2_1152);
          } else {
            int32_t int16_2 = (qw[i / 2] >> ((i % 2) * 8)) & kMask00FF;
            int16_t low = static_cast<int16_t>(int16_2 & 0xFFFF);
            int16_t high = static_cast<int16_t>((int16_2 >> 16) & 0xFFFF);
            nv_bfloat16 bf_low = __float2bfloat16(static_cast<float>(low));
            nv_bfloat16 bf_high = __float2bfloat16(static_cast<float>(high));
            nv_bfloat162 bf2 = __halves2bfloat162(bf_low, bf_high);
            int32_t packed_bf16 = *reinterpret_cast<int32_t*>(&bf2);
            w[i] = packed_add<nv_bfloat16>(packed_bf16, kRangeMin);
          }
        }
      }

      // Apply decoding scales
      for (int i = 0; i < 4; i++) {
        w[i] = packed_mul<T>(w[i], qs);
      }

      data[k] = w;
    }
  }
};

// 中文注释：两阶段 AllReduce 算法实现（AllReduceTwoshot）
// ================================================================
// 这是 QuickReduce 的核心算法。传统的 Ring AllReduce 需要 2*(P-1) 步
// （P 为 GPU 数量），而 Two-Shot 只需要 2 步：
//
// 第一步（Scatter-Reduce）：
//   1A: 每个 rank 将自己的数据分成 P 段，发送给对应的 rank
//   1B: 每个 rank 收集所有 rank 发来的同一段数据，进行本地归约（求和）
//
// 第二步（All-Gather）：
//   2A: 每个 rank 将归约后的结果广播给所有其他 rank
//   2B: 每个 rank 收集所有段的最终归约结果，拼接成完整的输出
//
// 模板参数：
//   - T: 数据类型（half 或 nv_bfloat16）
//   - Codec: 编解码器类型（CodecFP/Q4/Q6/Q8），决定是否使用量化通信
//   - cast_bf2half: 是否将 bf16 输入转为 half 计算，利用 half 的硬件指令优势
// ================================================================
// Twoshot All Reduce
template <typename T, class Codec, bool cast_bf2half>
struct AllReduceTwoshot {
  static_assert(sizeof(T) == 2);

  static constexpr int kWorldSize = Codec::kWorldSize;

  // 中文注释：AllReduce 主函数。每个线程块处理一个 tile 的数据
  // 参数说明：
  //   - input: 本 rank 的输入数据指针
  //   - output: 输出数据指针（AllReduce 结果写入此处）
  //   - N: 元素总数
  //   - block: 当前处理的 tile 编号（用于处理超过最大 block 数的大张量）
  //   - rank: 当前 GPU 的 rank 编号
  //   - buffer_list: 所有 rank 的通信缓冲区指针列表（通过 IPC 共享）
  //   - data_offset: 通信缓冲区中数据区域的起始偏移
  //   - flag_color: 当前轮次的标志颜色（用于区分不同 AllReduce 调用）
  //   - data_size_per_phase: 每个阶段的数据区大小
  __device__ static void run(
      T const* __restrict__ input, T* __restrict__ output,
      uint32_t const N,                    // number of elements
      int const block,                     // block index
      int const rank,                      // rank index
      uint8_t** __restrict__ buffer_list,  // communication buffers
      uint32_t const data_offset,          // offset to start of the data buffer
      uint32_t flag_color, int64_t data_size_per_phase) {
    // Topology
    // 中文注释：计算线程的全局编号（0~255）
    int thread = threadIdx.x + threadIdx.y * kWavefront;
    // 中文注释：获取当前 rank 的通信缓冲区（用于接收其他 rank 写入的数据）
    uint8_t* rank_buffer = buffer_list[rank];
    // 中文注释：创建编解码器实例，用于数据的序列化/反序列化
    Codec codec(thread, rank);
    int block_id = blockIdx.x;
    // --------------------------------------------------------
    // Read input into registers
    // 中文注释：将输入数据从 global memory 加载到寄存器
    // tA 数组存储当前线程负责的所有 atom（8 个 int32x4_t = 128 字节 = 64 个 fp16 值）
    int32x4_t tA[kAtoms];

    // 中文注释：创建 buffer resource 描述符，用于高效的 raw buffer load
    BufferResource src_buffer(const_cast<T*>(input), N * sizeof(T));
    // 中文注释：计算当前线程在 tile 内的起始偏移
    // 布局：block * kTileSize 定位到当前 tile，thread * 16B 定位到当前线程的 atom
    uint32_t src_offset = block * kTileSize + thread * sizeof(int32x4_t);

    // 中文注释：循环加载 kAtoms=8 个 atom 到寄存器，步长为 kAtomStride=256
    // 这样相邻线程加载的是相邻的 16 字节数据，实现了内存合并访问（coalesced access）
    for (int i = 0; i < kAtoms; i++) {
      tA[i] = buffer_load_dwordx4(src_buffer.descriptor, src_offset, 0, 0);
      src_offset += kAtomStride * sizeof(int32x4_t);
      // 中文注释：如果启用了 bf16->half 转换（cast_bf2half），在此处执行
      // 这是因为 AMD GPU 对 half 类型有更丰富的硬件指令支持（如 v_pk_add_f16）
      // 转换开销会被硬件指令的加速所抵消
      if constexpr (cast_bf2half) {
        const nv_bfloat162* bf_buf =
            reinterpret_cast<const nv_bfloat162*>(&tA[i]);
        half2 half_buf[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          float2 f = __bfloat1622float2(bf_buf[j]);
          half_buf[j] = __float22half2_rn(f);
        }
        tA[i] = *reinterpret_cast<const int32x4_t*>(half_buf);
      }
    }

    // --------------------------------------------------------
    // Phase-1A: Write segment data into the communication buffer of the target
    // rank responsible for this segment.
    // 中文注释：=== 第一阶段 A：Scatter（数据分发）===
    //
    // 通信缓冲区内存布局：
    //   Phase-1 数据区：data_offset 起始，每个 block 一个 tile，每个 tile 内按 rank 排列
    //   Phase-1 标志区：从 0 开始，每个 block 有 world_size * sizeof(uint32_t) 字节
    //   Phase-2 数据区：Phase-1 数据区之后
    //   Phase-2 标志区：Phase-1 标志区之后
    //
    // 数据分发逻辑：
    //   本 rank 将自己的数据分成 world_size 段，第 r 段发送到 rank r 的通信缓冲区
    //   这样每个 rank 收集到的是来自所有 rank 的同一段数据
    uint32_t comm_data0_offset =
        data_offset + block_id * Codec::kTransmittedTileSize;
    uint32_t comm_data1_offset = data_size_per_phase + comm_data0_offset;

    uint32_t comm_flags0_offset = block_id * (kWorldSize * sizeof(uint32_t));
    uint32_t comm_flags1_offset = (data_offset / 2) + comm_flags0_offset;

    // 中文注释：将第 r 段数据发送到 rank r 的通信缓冲区中"本 rank 对应的位置"
    // codec.send 会根据 Codec 类型决定是直接传输还是量化后传输
    for (int r = 0; r < kWorldSize; r++) {
      int32x4_t* send_buffer =
          reinterpret_cast<int32x4_t*>(buffer_list[r] + comm_data0_offset +
                                       rank * Codec::kRankTransmittedTileSize);
      codec.send(send_buffer, &tA[r * Codec::kRankAtoms]);
    }

    // 中文注释：同步后设置标志位，通知其他 rank 数据已写入完成
    // 只有前 world_size 个线程（每个对应一个目标 rank）负责写标志
    __syncthreads();
    if (thread < kWorldSize) {
      int r = thread;
      uint32_t* flag_ptr = reinterpret_cast<uint32_t*>(
          buffer_list[r] + comm_flags0_offset + rank * sizeof(uint32_t));
      set_sync_flag(flag_ptr, flag_color);
    }
    // --------------------------------------------------------
    // Phase-1B: Reduce the segment data from the communication buffers.
    // 中文注释：=== 第一阶段 B：Reduce（数据归约）===
    //
    // 当前 rank 负责归约自己对应的那段数据。流程：
    //   1. 从通信缓冲区中逐个读取每个 rank 写入的该段数据
    //   2. 每读到一个 rank 的数据，就累加到 tR（reduce accumulator）中
    //   3. 等待所有 rank 的数据都归约完成后，tR 就是该段的总和
    //
    // tR 初始化为 0，通过 packed_assign_add 逐元素累加
    int32x4_t tR[Codec::kRankAtoms] = {};
    {
      // Read the data from the communication buffer.
      int32x4_t* recv_buffer =
          reinterpret_cast<int32x4_t*>(rank_buffer + comm_data0_offset);
      uint32_t* flag_ptr =
          reinterpret_cast<uint32_t*>(rank_buffer + comm_flags0_offset);

      for (int r = 0; r < kWorldSize; r++) {
        // 中文注释：等待 rank r 的数据写入完成（flag 被设置）
        // 只有 thread 0 进行忙等待，然后通过 __syncthreads 广播给整个 block
        // Wait for the flags to be set.
        if (thread == 0) {
          wait_sync_flag(&flag_ptr[r], flag_color);
        }
        __syncthreads();

        // 中文注释：从通信缓冲区接收（反量化）rank r 的数据到 tA 临时缓冲区
        // note: we reuse tA as temp buffer here
        codec.recv(&recv_buffer, tA);

        // 中文注释：将 rank r 的数据累加到归约结果 tR 中
        for (int i = 0; i < Codec::kRankAtoms; i++) {
          packed_assign_add<T>(&tR[i], &tA[i]);
        }
      }
    }

    // Phase-2: Write the reduced segment to every other rank
    // 中文注释：=== 第二阶段 A：Broadcast（广播归约结果）===
    //
    // 当前 rank 将自己归约好的段数据发送给所有其他 rank
    // 每个 rank 都在做同样的事情：广播自己归约的那一段
    for (int r = 0; r < kWorldSize; r++) {
      int32x4_t* send_buffer =
          reinterpret_cast<int32x4_t*>(buffer_list[r] + comm_data1_offset +
                                       rank * Codec::kRankTransmittedTileSize);
      codec.send(send_buffer, tR);
    }

    // 中文注释：同步后设置 Phase-2 的标志位
    __syncthreads();
    if (thread < kWorldSize) {
      int r = thread;
      uint32_t* flag_ptr = reinterpret_cast<uint32_t*>(
          buffer_list[r] + comm_flags1_offset + rank * sizeof(uint32_t));
      set_sync_flag(flag_ptr, flag_color);
    }

    // Phase-2: Read the gather segments from the rank's communication buffer.
    // 中文注释：=== 第二阶段 B：Gather（收集所有段）===
    //
    // 当前 rank 从通信缓冲区中收集所有 rank 广播的归约结果
    // 第 r 个 rank 的归约结果对应输出的第 r 段，写入 tA[r * kRankAtoms]
    // 最终 tA 包含完整的 AllReduce 输出数据
    {
      // Read the data from the communication buffer.
      int32x4_t* recv_buffer =
          reinterpret_cast<int32x4_t*>(rank_buffer + comm_data1_offset);
      uint32_t* flag_ptr =
          reinterpret_cast<uint32_t*>(rank_buffer + comm_flags1_offset);

      for (int r = 0; r < kWorldSize; r++) {
        // Wait for the flags to be set.
        if (thread == 0) {
          wait_sync_flag(&flag_ptr[r], flag_color);
        }
        __syncthreads();

        // Gather all reduced and final rank segments into tA.
        codec.recv(&recv_buffer, &tA[r * Codec::kRankAtoms]);
      }
    }

    // --------------------------------------------------------
    // Write the result to output.
    // 中文注释：=== 写回输出 ===
    //
    // 将寄存器中的 AllReduce 结果写回 global memory 的 output 指针
    // 如果之前做了 bf16->half 转换，此处需要将 half 转回 bf16
    // 使用 buffer_store_dwordx4 进行高效的 128 位写入
    BufferResource dst_buffer(output, N * sizeof(T));
    uint32_t dst_offset = block * kTileSize + thread * sizeof(int32x4_t);

    for (int i = 0; i < kAtoms; i++) {
      if constexpr (cast_bf2half) {
        // 中文注释：half -> bf16 转换：通过 float 中转
        const half2* half_buf = reinterpret_cast<const half2*>(&tA[i]);
        nv_bfloat162 bf16_buf[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          float2 f = __half22float2(half_buf[j]);
          bf16_buf[j] = __float22bfloat162_rn(f);
        }
        buffer_store_dwordx4(*reinterpret_cast<const int32x4_t*>(bf16_buf),
                             dst_buffer.descriptor, dst_offset, 0, 0);
      } else {
        buffer_store_dwordx4(tA[i], dst_buffer.descriptor, dst_offset, 0, 0);
      }
      dst_offset += kAtomStride * sizeof(int32x4_t);
    }
  }
};

}  // namespace quickreduce