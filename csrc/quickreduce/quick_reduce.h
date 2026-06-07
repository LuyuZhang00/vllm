// =============================================================================
// 中文注释：QuickReduce 对外接口头文件 (quick_reduce.h)
// =============================================================================
// 本文件是 QuickReduce 库的最高层接口，提供：
//
// 1. DeviceComms 结构体：管理跨 GPU 的通信基础设施
//    - 通过 HIP IPC（Inter-Process Communication）实现 GPU 间内存共享
//    - 管理通信缓冲区的分配、IPC handle 的创建和交换
//    - 提供 allreduce 模板函数作为最终的 AllReduce 入口
//
// 2. allreduce_prototype_twoshot kernel：
//    - CUDA kernel 的入口函数，处理大张量的分块调度
//    - 当张量超过单个 kernel launch 的处理能力时，通过 while 循环处理多个 tile
//
// 3. TWOSHOT_DISPATCH 宏：
//    - 根据 world_size 和 Codec 类型分发到具体的 kernel 实例化
//    - 支持 world_size = 2, 4, 8 的配置
//
// 使用流程：
//   1. 创建 DeviceComms 实例
//   2. 调用 init() 分配通信缓冲区
//   3. 交换 IPC handles（需要外部协调）
//   4. 调用 open_ipc_handles() 建立跨 GPU 的内存访问
//   5. 调用 allreduce() 执行 AllReduce 操作
// =============================================================================

#pragma once

#include <vector>
#include <hip/hip_runtime.h>
#include "quick_reduce_impl.cuh"

// 中文注释：HIP 错误检查宏，在每个 HIP API 调用后检查返回值
// 如果发生错误，打印详细的错误信息（文件名、行号、错误描述）并抛出异常
#define HIP_CHECK(err)                                                     \
  do {                                                                     \
    hipError_t err_ = (err);                                               \
    if (err_ != hipSuccess) {                                              \
      std::printf("HIP error %d at %s:%d. %s\n", err_, __FILE__, __LINE__, \
                  hipGetErrorString(err_));                                \
      throw std::runtime_error("HIP error");                               \
    }                                                                      \
  } while (0)

namespace quickreduce {
// 中文注释：函数指针类型，用于跨语言（如 Python ctypes）调用
using fptr_t = int64_t;
static_assert(sizeof(void*) == sizeof(fptr_t));

// 中文注释：Two-Shot AllReduce 的 CUDA kernel 入口函数
// 当数据量超过一个 kernel launch 能处理的 tile 数量时（num_blocks > grid），
// 通过 while 循环让每个 block 处理多个 tile（grid-stride loop 模式）
// 这样即使数据量非常大，也只需要一次 kernel launch
//
// 每处理完一个 tile，flag_color 递增，用于区分不同 tile 的同步标志
template <typename AllReduceKernel, typename T>
__global__ __quickreduce_launch_bounds_two_shot__ static void
allreduce_prototype_twoshot(T const* A, T* B, uint32_t N, uint32_t num_blocks,
                            int rank, uint8_t** dbuffer_list,
                            uint32_t data_offset, uint32_t flag_color,
                            int64_t data_size_per_phase) {
  int block = blockIdx.x;
  int grid = gridDim.x;

  while (block < num_blocks) {
    AllReduceKernel::run(A, B, N, block, rank, dbuffer_list, data_offset,
                         flag_color, data_size_per_phase);
    block += grid;
    flag_color++;
  }
}

// 中文注释：Two-Shot kernel 分发宏
// 根据 world_size（GPU 数量）选择对应的 Codec 实例化并启动 kernel
// 支持 world_size = 2, 4, 8 三种配置
// Codec 模板参数决定了通信时是否使用量化（FP/Q4/Q6/Q8）
#define TWOSHOT_DISPATCH(__codec)                                           \
  if (world_size == 2) {                                                    \
    using LineCodec = __codec<T, 2>;                                        \
    using AllReduceKernel = AllReduceTwoshot<T, LineCodec, cast_bf2half>;   \
    hipLaunchKernelGGL((allreduce_prototype_twoshot<AllReduceKernel, T>),   \
                       dim3(grid), dim3(kBlockTwoShot), 0, stream, A, B, N, \
                       num_blocks, rank, dbuffer_list, data_offset,         \
                       flag_color, this->kMaxProblemSize);                  \
  } else if (world_size == 4) {                                             \
    using LineCodec = __codec<T, 4>;                                        \
    using AllReduceKernel = AllReduceTwoshot<T, LineCodec, cast_bf2half>;   \
    hipLaunchKernelGGL((allreduce_prototype_twoshot<AllReduceKernel, T>),   \
                       dim3(grid), dim3(kBlockTwoShot), 0, stream, A, B, N, \
                       num_blocks, rank, dbuffer_list, data_offset,         \
                       flag_color, this->kMaxProblemSize);                  \
  } else if (world_size == 8) {                                             \
    using LineCodec = __codec<T, 8>;                                        \
    using AllReduceKernel = AllReduceTwoshot<T, LineCodec, cast_bf2half>;   \
    hipLaunchKernelGGL((allreduce_prototype_twoshot<AllReduceKernel, T>),   \
                       dim3(grid), dim3(kBlockTwoShot), 0, stream, A, B, N, \
                       num_blocks, rank, dbuffer_list, data_offset,         \
                       flag_color, this->kMaxProblemSize);                  \
  }

// 中文注释：量化级别枚举，决定 AllReduce 通信时使用的数据精度
//   - F16: 不量化，直接传输 fp16/bf16（最高精度，最大带宽）
//   - INT8: 8-bit 量化（精度和带宽的平衡）
//   - INT6: 6-bit 量化
//   - INT4: 4-bit 量化（最低精度，最省带宽）
enum QuickReduceQuantLevel {
  F16 = 0,
  INT8 = 1,
  INT6 = 2,
  INT4 = 3,
};

// 中文注释：DeviceComms - 跨 GPU 通信管理器
// ================================================================
// 这是 QuickReduce 库的核心管理类，负责：
//
// 1. 通信缓冲区管理：
//    - 分配一块大的 GPU 内存作为通信缓冲区
//    - 缓冲区分为"标志区"和"数据区"两部分
//    - 标志区用于 rank 间的同步信号
//    - 数据区用于存放 AllReduce 的中间数据
//
// 2. IPC 内存共享：
//    - 通过 HIP IPC handle 实现跨进程的 GPU 内存共享
//    - 每个 rank 创建自己的 IPC handle，然后与其他 rank 交换
//    - 交换完成后，所有 rank 都可以直接访问其他 rank 的通信缓冲区
//
// 3. AllReduce 调度：
//    - 根据量化级别选择合适的 Codec
//    - 计算 grid 大小和 block 数量
//    - 启动 AllReduce kernel
// ================================================================
struct DeviceComms {
  // Max problem size is 2GB (in bytes) or half of uint32_t max value.
  // 中文注释：单次 AllReduce 支持的最大数据量（2GB）
  // 限制来源于通信缓冲区的偏移量使用 uint32_t 存储
  int64_t kMaxProblemSize =
      static_cast<int64_t>(std::numeric_limits<int32_t>::max()) + 1;

  // Max TP-8
  // 中文注释：最大支持的 GPU 数量（张量并行度上限为 8）
  static int constexpr kMaxWorldSize = 8;

  bool initialized = false;  // 中文注释：是否已完成初始化
  // 中文注释：标志颜色（flag color），每轮 AllReduce 递增，用于区分不同轮次
  // 这避免了上一轮的旧标志位干扰当前轮次的同步
  uint32_t flag_color = 1;
  int world_size;  // 中文注释：参与 AllReduce 的 GPU 总数
  int rank;        // 中文注释：当前 GPU 的 rank 编号（0 ~ world_size-1）

  uint8_t* dbuffer;              // 中文注释：本 rank 的通信缓冲区（GPU 内存）
  uint8_t** dbuffer_list;        // 中文注释：所有 rank 的通信缓冲区指针列表（GPU 上）
  hipIpcMemHandle_t buffer_ipc_handle;                    // 中文注释：本 rank 的 IPC handle
  std::vector<hipIpcMemHandle_t> all_buffer_ipc_handles;  // 中文注释：所有 rank 的 IPC handle
  std::vector<uint8_t*> buffer_list;   // 中文注释：所有 rank 的缓冲区指针（CPU 端副本）
  uint32_t data_offset;               // 中文注释：数据区在缓冲区中的起始偏移

  DeviceComms() : initialized(false), world_size(1), rank(0) {}
  ~DeviceComms() { destroy(); }

  // 中文注释：初始化通信基础设施
  // 流程：
  //   1. 计算通信缓冲区总大小 = 标志区 + 数据区
  //      - 标志区：2 个阶段 * world_size 个 rank * kMaxNumBlocks 个 block * 4 字节
  //      - 数据区：2 个阶段 * kMaxProblemSize 字节（最坏情况为 F16 不压缩）
  //   2. 分配 uncached GPU 内存（hipDeviceMallocUncached）
  //      使用 uncached 内存避免跨 GPU 访问时的 cache 一致性问题
  //   3. 清零标志区
  //   4. 创建 IPC handle 用于跨进程共享
  void init(int world_size, int rank,
            std::optional<int64_t> max_problem_size = std::nullopt) {
    destroy();
    this->world_size = world_size;
    this->rank = rank;
    if (max_problem_size.has_value() && max_problem_size.value() > 0) {
      this->kMaxProblemSize = max_problem_size.value();
    }
    // Allocate buffer size for worst case: F16 2-stage buffer.
    uint32_t flags_buffer_size =
        2 * world_size * kMaxNumBlocks * sizeof(uint32_t);
    static int64_t data_buffer_size = 2 * this->kMaxProblemSize;
    int64_t total_buffer_size = flags_buffer_size + data_buffer_size;
    data_offset = flags_buffer_size;
    HIP_CHECK(hipExtMallocWithFlags((void**)&dbuffer, total_buffer_size,
                                    hipDeviceMallocUncached));

    // Clear the flags buffer.
    HIP_CHECK(hipMemset(dbuffer, 0, flags_buffer_size));

    // Device-side list of IPC buffers.
    buffer_list.resize(world_size);
    HIP_CHECK(hipMalloc(&dbuffer_list, world_size * sizeof(uint8_t*)));

    // Create IPC handles for rank's communication buffer.
    all_buffer_ipc_handles.resize(world_size);
    HIP_CHECK(hipIpcGetMemHandle(&buffer_ipc_handle, dbuffer));

    initialized = true;
  }
  int get_world_size() { return world_size; }
  int get_rank() { return rank; }
  bool status() { return initialized; }
  hipIpcMemHandle_t const get_handle() { return buffer_ipc_handle; }

  void destroy() {
    if (initialized) {
      for (int i = 0; i < world_size; i++) {
        if (i != rank) {
          HIP_CHECK(hipIpcCloseMemHandle(dbuffer_list[i]));
        }
      }

      HIP_CHECK(hipFree(dbuffer));
      HIP_CHECK(hipFree(dbuffer_list));

      initialized = false;
    }
  }

  // 中文注释：打开所有 rank 的 IPC handle，建立跨 GPU 的内存访问
  // 流程：
  //   1. 收集所有 rank 的 IPC handle（由外部协调完成交换）
  //   2. 对于其他 rank，调用 hipIpcOpenMemHandle 打开共享内存
  //      hipIpcMemLazyEnablePeerAccess 表示延迟启用 P2P 访问
  //   3. 对于自己的 rank，直接使用本地指针
  //   4. 将 buffer_list 拷贝到 GPU 上的 dbuffer_list，供 kernel 使用
  void open_ipc_handles(std::vector<hipIpcMemHandle_t> const& ipc_handles) {
    assert(ipc_handles.size() == all_buffer_ipc_handles.size());
    for (int i = 0; i < world_size; i++) {
      all_buffer_ipc_handles[i] = ipc_handles[i];
    }

    // Open device memory access to the IPC communication buffers.
    // Note: For our own rank, we do not need to open a handle.
    for (int i = 0; i < world_size; i++) {
      if (i != rank) {
        HIP_CHECK(hipIpcOpenMemHandle((void**)&buffer_list[i],
                                      all_buffer_ipc_handles[i],
                                      hipIpcMemLazyEnablePeerAccess));
      } else {
        buffer_list[i] = dbuffer;
      }
    }

    // 中文注释：将 CPU 端的 buffer_list 拷贝到 GPU 端的 dbuffer_list
    // kernel 通过 dbuffer_list 访问所有 rank 的通信缓冲区
    HIP_CHECK(hipMemcpy(dbuffer_list, buffer_list.data(),
                        world_size * sizeof(uint8_t*), hipMemcpyHostToDevice));
  }

  // 中文注释：执行 AllReduce 操作的主入口函数
  // 模板参数：
  //   - T: 数据类型（half 或 nv_bfloat16）
  //   - cast_bf2half: 是否将 bf16 转为 half 计算
  // 参数：
  //   - A: 输入数据指针
  //   - B: 输出数据指针
  //   - N: 元素数量
  //   - quant_level: 量化级别（0=F16, 1=INT8, 2=INT6, 3=INT4）
  //   - stream: HIP 流
  //
  // 执行流程：
  //   1. 验证 world_size 是否受支持（2/4/8）
  //   2. 计算需要的 block 数量和 grid 大小
  //   3. 根据量化级别选择对应的 Codec，启动 kernel
  //   4. 递增 flag_color 为下一次 AllReduce 做准备
  template <typename T, bool cast_bf2half>
  void allreduce(T const* A, T* B, uint32_t N, int quant_level,
                 hipStream_t stream) {
    if (world_size != 2 && world_size != 4 && world_size != 8) {
      throw std::runtime_error("All Reduce not supported for world_size = " +
                               std::to_string(world_size));
    }

    // Configuration.
    // 中文注释：计算消息大小、tile 数量和 grid 大小
    // grid = min(最大block数, 所需tile数)，多余的 tile 通过 grid-stride loop 处理
    uint32_t msg_size = N * sizeof(T);
    uint32_t num_blocks = divceil(msg_size, kTileSize);
    uint32_t grid = min(kMaxNumBlocks, num_blocks);
    // 中文注释：根据量化级别分发到对应的 Codec
    // CodecFP: 全精度无压缩
    // CodecQ8/Q6/Q4: 逐级压缩，减少通信带宽但增加量化误差
    auto quant_level_ = static_cast<QuickReduceQuantLevel>(quant_level);
    switch (quant_level_) {
      case QuickReduceQuantLevel::INT8:
        TWOSHOT_DISPATCH(CodecQ8)
        break;
      case QuickReduceQuantLevel::INT6:
        TWOSHOT_DISPATCH(CodecQ6)
        break;
      case QuickReduceQuantLevel::INT4:
        TWOSHOT_DISPATCH(CodecQ4)
        break;
      default:
        TWOSHOT_DISPATCH(CodecFP)
        break;
    }
    HIP_CHECK(cudaGetLastError());
    // Rotate the flag color.
    // 中文注释：递增 flag_color，确保下一次 AllReduce 的标志位不会与本次冲突
    // 递增量取决于处理了多少个 tile（每个 tile 需要独立的 flag_color）
    flag_color += divceil(N, grid);
  }
};

}  // namespace quickreduce