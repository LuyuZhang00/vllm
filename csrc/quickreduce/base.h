// =============================================================================
// 中文注释：QuickReduce 基础工具头文件 (base.h)
// =============================================================================
// 本文件是 QuickReduce 库的底层基础设施层，专门为 AMD GPU (HIP/CDNA) 架构优化。
// QuickReduce 是一个高性能的跨 GPU 通信库，用于实现 AllReduce 等集合通信操作。
//
// 本文件主要提供以下功能：
//   1. 常量定义：线程块大小、wavefront 大小、tile 尺寸等硬件相关参数
//   2. 缓冲区资源描述符：用于 AMD GPU 的 raw buffer load/store 指令
//   3. 向量化打包运算：将两个 fp16/bf16 值打包在一个 int32 中进行高效运算
//   4. 原子同步原语：基于 acquire-release 语义的标志位同步机制
//   5. 分组归约：在量化过程中计算线程组内的绝对值最大值
//
// 优化策略：
//   - 使用 AMD GPU 的 packed fp16 指令（v_pk_add_f16 等）一次处理两个值
//   - 使用 buffer load/store 指令替代普通 global memory 访问，提高带宽利用率
//   - 使用 nontemporal store 减少对 L2 cache 的污染
// =============================================================================

#pragma once

#include <cstdint>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>

// 中文注释：内联宏定义，确保编译器将这些函数内联到调用点，消除函数调用开销
#define __quickreduce_device_inline__ __device__ __forceinline__

// 中文注释：Two-shot AllReduce 的 launch bounds：每个 block 256 线程，每个 CU 最多驻留 4 个 block
// 这有助于编译器优化寄存器分配和 occupancy
#define __quickreduce_launch_bounds_two_shot__ __launch_bounds__(256, 4)

// 中文注释：One-shot AllReduce 的 launch bounds：每个 block 512 线程，每个 CU 最多驻留 4 个 block
#define __quickreduce_launch_bounds_one_shot__ __launch_bounds__(512, 4)

namespace quickreduce {

// 中文注释：将 HIP 的 bf16 类型别名为 nv_bfloat16，保持与 NVIDIA 风格 API 的兼容性
typedef __hip_bfloat16 nv_bfloat16;
typedef __hip_bfloat162 nv_bfloat162;

// 中文注释：定义向量化的 int32 类型，用于一次加载/存储多个数据
// int32x2_t: 打包 2 个 int32（8 字节），对应 4 个 fp16 值
// int32x4_t: 打包 4 个 int32（16 字节），对应 8 个 fp16 值，即一个 "atom"
using int32x2_t = __attribute__((__vector_size__(2 * sizeof(int)))) int;
using int32x4_t = __attribute__((__vector_size__(4 * sizeof(int)))) int;

// 中文注释：为 AMD GPU 的向量化内存访问指令（mubuf）设置 acquire-release 语义
// 不同 CDNA 架构使用不同的控制位：
//   - CDNA3 (gfx942): 使用 scope bits sc0, sc1
//   - CDNA1/CDNA2 (gfx908/gfx90a): 使用 glc (globally coherent) 位
// Setup acquire-release semantics for vector memory reads (mubuf instruction)
// as per architecture.
#if defined(__gfx942__)
// CDNA3: Scope bits sc0, sc1
  #define MUBUF_ACQUIRE 16
  #define MUBUF_RELEASE 16
#elif (defined(__gfx908__) || defined(__gfx90a__))
// CDNA1 and CDNA2 - glc bit
  #define MUBUF_ACQUIRE 1
  #define MUBUF_RELEASE 0
#endif

// 中文注释：{-1, -1} 的 fp16x2 打包表示，用于 packed_sub 中的 FMA 替代减法
// MI300 缺少 packed fp16 减法指令，所以用 a + (-1)*b 的 FMA 方式实现
static constexpr int kNegOne = 0xBC00BC00;  // {-1, -1}, fp16x2_t

// 中文注释：每个线程处理的 "atom" 数量。每个 atom 是一个 int32x4_t（16 字节 = 8 个 fp16 值）
// 因此每个线程处理 8 * 8 = 64 个 fp16 值 = 128 字节
// Number of atoms (4xf16x2_t) processed by a single thread
static constexpr int kAtoms = 8;

// 中文注释：每个 workgroup（线程块）的线程数。选择 256 是因为这是 AMD CDNA 架构的高效配置
// We use a workgroup of 256 threads
static constexpr int kBlockSize = 256;

// 中文注释：atom 的步长，等于线程数。在线程交错布局中，相邻 atom 之间间隔 kBlockSize 个位置
static constexpr int kAtomStride = kBlockSize;

// Size and atom stride of source/destination data that the block will
// process.
// Workgroup scope = Tile = (256 threads x 8 atoms x 16B)
// 中文注释：一个 tile 的总字节数 = 256 线程 * 8 atoms/线程 * 16 字节/atom = 32KB
// 这是单个 workgroup 一次处理的数据量，是 AllReduce 中数据分块的基本单位
static constexpr int kTileSize = kBlockSize * kAtoms * sizeof(int32x4_t);

// Max number of blocks. 304 CUs on MI300
// 中文注释：最大 block 数量限制。MI300 有 304 个 CU，每个 CU 可以运行 4 个 block
static constexpr int kMaxNumBlocks = 304 * 4;

// Standard CDNA wavefront size.
// 中文注释：CDNA 架构的标准 wavefront 大小（类似 NVIDIA 的 warp size = 32）
static constexpr int kWavefront = 64;

// 256 thread, 4 wavefronts.
// 中文注释：Two-shot kernel 的线程块维度：{64, 4, 1} = 256 线程
// 第一维 x=64 是一个 wavefront 的大小，第二维 y=4 表示 4 个 wavefront
static dim3 constexpr kBlockTwoShot = {kWavefront, kBlockSize / kWavefront, 1};

// Number of threads in a group for quantization
// It corresponds to 32 F16 elements in quantization block
// 中文注释：量化中线程组的大小。每 8 个线程为一组，处理 32 个 fp16 元素
// （8 线程 * 4 个 int32x4_t/线程 = 32 个 fp16 值），共享一个量化 scale
static constexpr int kThreadGroupSize = 8;

// Methods
// 中文注释：向上整除函数，用于计算需要多少个 block 来容纳 N 个元素
__quickreduce_device_inline__ __host__ unsigned long divceil(unsigned long x,
                                                             unsigned long y) {
  return ((x + y - 1) / y);
}

// 中文注释：AMD GPU 缓冲区资源描述符（Buffer Resource）
// 这是一个 128 位的硬件描述符，用于 AMD 的 raw buffer load/store 指令
// 它将内存地址、大小范围和格式配置打包在一起，使 GPU 能够高效访问 global memory
// 相比普通指针，buffer resource 支持边界检查和更高的访问带宽
union BufferResource {
  __quickreduce_device_inline__ constexpr BufferResource()
      : config(0x00020000U) {}

  __quickreduce_device_inline__ constexpr BufferResource(void* buffer_address,
                                                         uint32_t buffer_size)
      : address(buffer_address), range(buffer_size), config(0x00020000U) {}

  int32x4_t descriptor;
  struct {
    void* address;  // 8B, out of which first 48b is address, and 16b is stride
    // (unused)
    uint32_t range;   // Byte range for the buffer resource
    uint32_t config;  // Constant, DFMT=32b
  };
};

// 中文注释：AMD GPU 的 raw buffer load 指令，一次加载 4 个 dword（16 字节 = 128 位）
// 使用 buffer resource 描述符进行地址计算，比普通 global load 更高效
__quickreduce_device_inline__ static int32x4_t buffer_load_dwordx4(
    int32x4_t srsrc, int32_t voffset, int32_t soffset,
    int32_t aux) __asm("llvm.amdgcn.raw.buffer.load.v4i32");

// 中文注释：AMD GPU 的 raw buffer store 指令，一次存储 4 个 dword（16 字节 = 128 位）
__quickreduce_device_inline__ static void buffer_store_dwordx4(
    int32x4_t data, int32x4_t srsrc, int32_t voffset, int32_t soffset,
    int32_t aux) __asm("llvm.amdgcn.raw.buffer.store.v4i32");

// 中文注释：设置 fp16 溢出行为。仅在 gfx942 (CDNA3) 上有效
// 当设置为 true 时，fp16 运算遇到溢出会饱和到最大值而不是产生 NaN
__quickreduce_device_inline__ static void set_fp16_ovfl(bool const value) {
#if defined(__gfx942__)
  if (value) {
    asm volatile("s_setreg_imm32_b32 0xdc1, 1;" ::);
  } else {
    asm volatile("s_setreg_imm32_b32 0xdc1, 0;" ::);
  }
#endif
}
// 中文注释：bf162 和 int 之间的类型双关联盟，用于在 bf16x2 和 int32 之间无开销转换
union bf162_int_union {
  int i;
  nv_bfloat162 bf2;
};

// 中文注释：向量化打包加法赋值：A += B，其中 A 和 B 各包含 4 个 int32
// 每个 int32 打包了 2 个 fp16/bf16 值，所以一次调用处理 8 个值
// 这是 AllReduce 中归约操作的核心原语
template <typename T>
__quickreduce_device_inline__ void packed_assign_add(int32x4_t* A,
                                                     int32x4_t* B);

template <>
__quickreduce_device_inline__ void packed_assign_add<half>(int32x4_t* A,
                                                           int32x4_t* B) {
  int32x4_t& tR_fragment = A[0];
  int32x4_t& tA_fragment = B[0];

  asm volatile("v_pk_add_f16 %0, %1, %2"
               : "=v"(tR_fragment[0])
               : "v"(tR_fragment[0]), "v"(tA_fragment[0]));
  asm volatile("v_pk_add_f16 %0, %1, %2"
               : "=v"(tR_fragment[1])
               : "v"(tR_fragment[1]), "v"(tA_fragment[1]));
  asm volatile("v_pk_add_f16 %0, %1, %2"
               : "=v"(tR_fragment[2])
               : "v"(tR_fragment[2]), "v"(tA_fragment[2]));
  asm volatile("v_pk_add_f16 %0, %1, %2"
               : "=v"(tR_fragment[3])
               : "v"(tR_fragment[3]), "v"(tA_fragment[3]));
}

template <>
__quickreduce_device_inline__ void packed_assign_add<nv_bfloat16>(
    int32x4_t* A, int32x4_t* B) {
  nv_bfloat162* tA = reinterpret_cast<nv_bfloat162*>(A);
  nv_bfloat162* tB = reinterpret_cast<nv_bfloat162*>(B);
#pragma unroll
  for (int i = 0; i < 4; i++) {
    tA[i] = __hadd2(tA[i], tB[i]);
  }
}

// 中文注释：打包取最大值。比较两个 int32 中各自打包的 2 个 fp16/bf16 值，返回逐元素最大值
template <typename T>
__quickreduce_device_inline__ int packed_max(int a, int b);

template <>
__quickreduce_device_inline__ int packed_max<half>(int a, int b) {
  int result;
  asm volatile("v_pk_max_f16 %0, %1, %2" : "=v"(result) : "v"(a), "v"(b));
  return result;
}

template <>
__quickreduce_device_inline__ int packed_max<nv_bfloat16>(int a, int b) {
  bf162_int_union A, B, R;
  A.i = a;
  B.i = b;
  R.bf2 = __hmax2(A.bf2, B.bf2);
  return R.i;
}

// 中文注释：打包取最小值，与 packed_max 对称
template <typename T>
__quickreduce_device_inline__ int packed_min(int a, int b);

template <>
__quickreduce_device_inline__ int packed_min<half>(int a, int b) {
  int result;
  asm volatile("v_pk_min_f16 %0, %1, %2" : "=v"(result) : "v"(a), "v"(b));
  return result;
}

template <>
__quickreduce_device_inline__ int packed_min<nv_bfloat16>(int a, int b) {
  bf162_int_union A, B, R;
  A.i = a;
  B.i = b;
  R.bf2 = __hmin2(A.bf2, B.bf2);
  return R.i;
}

// 中文注释：打包绝对值最大值。比较两个值的绝对值大小，返回绝对值更大的那个原始值
// 用于量化中确定 scale factor
template <typename T>
__quickreduce_device_inline__ int packed_abs_max(int a, int b);

template <>
__quickreduce_device_inline__ int packed_abs_max<half>(int a, int b) {
  half2 wmaxh2 = __builtin_bit_cast(half2, a);
  half2 wminh2 = __builtin_bit_cast(half2, b);
  half2 wblockmaxh2;

  wblockmaxh2.x =
      __hgt(__habs(wmaxh2.x), __habs(wminh2.x)) ? wmaxh2.x : wminh2.x;
  wblockmaxh2.y =
      __hgt(__habs(wmaxh2.y), __habs(wminh2.y)) ? wmaxh2.y : wminh2.y;
  return __builtin_bit_cast(int, wblockmaxh2);
}

template <>
__quickreduce_device_inline__ int packed_abs_max<nv_bfloat16>(int a, int b) {
  bf162_int_union A, B, R;
  A.i = a;
  B.i = b;
  R.bf2.x = __hgt(__habs(A.bf2.x), __habs(B.bf2.x)) ? A.bf2.x : B.bf2.x;
  R.bf2.y = __hgt(__habs(A.bf2.y), __habs(B.bf2.y)) ? A.bf2.y : B.bf2.y;
  return R.i;
}

// 中文注释：打包加法。将两个 int32 中各自打包的 2 个 fp16/bf16/int16 值逐元素相加
// 注意：int16_t 特化使用整数加法，其他使用浮点加法
template <typename T>
__quickreduce_device_inline__ int packed_add(int a, int b);

template <>
__quickreduce_device_inline__ int packed_add<half>(int a, int b) {
  int result;
  asm volatile("v_pk_add_f16 %0, %1, %2" : "=v"(result) : "v"(a), "v"(b));
  return result;
}

template <>
__quickreduce_device_inline__ int packed_add<nv_bfloat16>(int a, int b) {
  bf162_int_union A, B, R;
  A.i = a;
  B.i = b;
  R.bf2 = __hadd2(A.bf2, B.bf2);
  return R.i;
}

template <>
__quickreduce_device_inline__ int packed_add<int16_t>(int a, int b) {
  int result;
  asm volatile("v_pk_add_i16 %0, %1, %2" : "=v"(result) : "v"(a), "v"(b));
  return result;
}

// 中文注释：打包减法。对于 half 类型，MI300 缺少 packed fp16 减法指令，
// 使用 FMA 指令实现：a - b = a + (-1)*b
template <typename T>
__quickreduce_device_inline__ int packed_sub(int a, int b);

template <>
__quickreduce_device_inline__ int packed_sub<half>(int a, int b) {
  int result;

  // MI300 lacks packed fp16 sub instruction. So we do -1 * min + max
  asm volatile("v_pk_fma_f16 %0, %1, %2 %3"
               : "=v"(result)
               : "v"(kNegOne), "v"(b), "v"(a));
  return result;
}

template <>
__quickreduce_device_inline__ int packed_sub<nv_bfloat16>(int a, int b) {
  bf162_int_union A, B, R;
  A.i = a;
  B.i = b;
  R.bf2 = __hsub2(A.bf2, B.bf2);
  return R.i;
}

// 中文注释：打包乘法。将两个 int32 中各自打包的 2 个 fp16/bf16 值逐元素相乘
template <typename T>
__quickreduce_device_inline__ int packed_mul(int a, int b);

template <>
__quickreduce_device_inline__ int packed_mul<half>(int a, int b) {
  int result;
  asm volatile("v_pk_mul_f16 %0, %1, %2" : "=v"(result) : "v"(a), "v"(b));
  return result;
}

template <>
__quickreduce_device_inline__ int packed_mul<nv_bfloat16>(int a, int b) {
  nv_bfloat162* tA = reinterpret_cast<nv_bfloat162*>(&a);
  nv_bfloat162* tB = reinterpret_cast<nv_bfloat162*>(&b);
  nv_bfloat162 tR = __hmul2(*tA, *tB);
  return *(reinterpret_cast<int*>(&tR));
}

// 中文注释：打包倒数。计算打包的 2 个 fp16/bf16 值各自的 1/x
// 在量化中用于计算编码 scale（encoding_scale = 1 / decoding_scale）
template <typename T>
__quickreduce_device_inline__ int packed_rcp(int a);

template <>
__quickreduce_device_inline__ int packed_rcp<half>(int a) {
  return __builtin_bit_cast(int, h2rcp(__builtin_bit_cast(half2, a)));
}

template <>
__quickreduce_device_inline__ int packed_rcp<nv_bfloat16>(int a) {
  bf162_int_union A, R;
  A.i = a;
  R.bf2 = h2rcp(A.bf2);
  return R.i;
}

// 中文注释：类型转换辅助函数，将 fp16/bf16 转换为 float
// 在量化过程中需要将 fp16 转为 float 进行四舍五入后再转回整数
// changes dtype
__quickreduce_device_inline__ float T2float_cast(half a) {
  return __half2float(a);
}

__quickreduce_device_inline__ float T2float_cast(nv_bfloat16 a) {
  return __bfloat162float(a);
}

// 中文注释：在线程组内计算绝对值最大值（用于量化 scale 计算）
// 流程：
//   1. 每个线程在自己的 4 个 int32（8 个 fp16 值）中找到最大值和最小值
//   2. 通过 shfl_down 在 kThreadGroupSize=8 个线程间进行 warp-level 归约
//   3. 从最大值和最小值中选出绝对值最大的那个（可能是正数或负数）
//   4. 通过 __shfl 广播给组内所有线程，确保使用同一个 scale
// 这样每 8 个线程（32 个 fp16 值）共享一个量化 scale
template <typename T>
__quickreduce_device_inline__ int group_abs_max(int32x4_t atom) {
  const int group_leader = (threadIdx.x / kThreadGroupSize) * kThreadGroupSize;

  int wmax, wmin, wblockmax;
  int a, b;
  a = packed_max<T>(atom[0], atom[1]);
  b = packed_max<T>(atom[2], atom[3]);

  wmax = packed_max<T>(a, b);

  a = packed_min<T>(atom[0], atom[1]);
  b = packed_min<T>(atom[2], atom[3]);

  wmin = packed_min<T>(a, b);

  // Reduce the max among a group of threads
  // Note: This is basically 2 blocks of values setup as the
  // upper/lower halves of the f16x2_t
  for (int i = 1; i < kThreadGroupSize; i <<= 1) {
    int x = __shfl_down(wmax, i);
    wmax = packed_max<T>(wmax, x);

    int y = __shfl_down(wmin, i);
    wmin = packed_min<T>(wmin, y);
  }
  wblockmax = packed_abs_max<T>(wmax, wmin);
  // Share with the cohort
  wblockmax = __shfl(wblockmax, group_leader);
  return wblockmax;
}

// 中文注释：同步标志位写入函数。使用 release 语义确保之前的数据写入对其他 GPU 可见
// flag_color 用于区分不同轮次的 AllReduce 操作，避免新旧数据的竞态条件
__quickreduce_device_inline__ void set_sync_flag(uint32_t* flag_ptr,
                                                 uint32_t flag) {
  __atomic_store_n(flag_ptr, flag, __ATOMIC_RELEASE);
}

// 中文注释：同步标志位等待函数。使用 relaxed 语义进行忙等待（spin-wait）
// 直到目标 flag 值匹配当前的 flag_color，说明对方 rank 已完成数据写入
__quickreduce_device_inline__ void wait_sync_flag(uint32_t* flag_ptr,
                                                  uint32_t flag) {
  while (__atomic_load_n(flag_ptr, __ATOMIC_RELAXED) != flag) {
  }
}

}  // namespace quickreduce