// ============================================================================
// 【模块概述】custom_all_reduce.cu — 自定义 AllReduce 的 Python 绑定层（Host 端实现）
// ============================================================================
//
// 本文件是自定义 AllReduce 操作的 Host 端入口，负责：
//   1. 初始化和管理 CustomAllreduce 实例（通过 CUDA IPC 建立跨 GPU 共享内存）
//   2. 提供 Python 可调用的 C++ 函数（通过 PyBind 绑定）
//   3. 管理 IPC 共享缓冲区的生命周期（分配、注册、释放）
//   4. 支持 CUDA Graph 场景下的 IPC 指针解析
//
// 整体调用链路：
//   Python 端 --> PyBind 绑定的 C 函数 --> CustomAllreduce 类方法 --> CUDA Kernel
//
// 关键设计：
//   - 使用 "fake pointer"（fptr_t = int64_t）在 Python 和 C++ 之间传递指针，
//     因为 Python 无法直接处理 C++ 指针类型。
//   - 通过 CUDA IPC MemHandle 实现跨进程的 GPU 内存共享，
//     使得不同进程中的 GPU 可以直接读取彼此的显存数据。
//   - 所有缓冲区由调用方分配并传入，本模块不拥有设备内存的所有权。
// ============================================================================

#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/all.h>

#include "custom_all_reduce.cuh"

// Fake pointer type, must match fptr_t type in ops.h.
// We use this type alias to indicate when pointers are passed in as int64_t.
// 【指针类型别名】将 C++ 指针以 int64_t 形式传递给 Python，用于跨语言边界传递 GPU 内存地址。
using fptr_t = int64_t;
static_assert(sizeof(void*) == sizeof(fptr_t));

// ============================================================================
// 【初始化函数】init_custom_ar — 创建 CustomAllreduce 实例
// ============================================================================
// 该函数是 AllReduce 操作的初始化入口，在 Python 端调用后获得一个"句柄"（fptr_t），
// 后续所有 AllReduce 操作都通过该句柄进行。
//
// 参数说明：
//   fake_ipc_ptrs  : 各 rank（GPU）的 Signal 同步缓冲区的 IPC 指针列表，
//                    通过 CUDA IPC 在进程间共享，用于跨 GPU 同步。
//   rank_data      : 预分配的 GPU 张量，用于存储各 rank 的 buffer 指针元数据（RankData），
//                    是 AllReduce kernel 的输入参数。
//   rank           : 当前 GPU 的编号（0 到 world_size-1）
//   fully_connected: 是否所有 GPU 之间都有 NVLink/xGMI 全连接，
//                    影响算法选择（一阶段 vs 两阶段）
//
// 返回值：指向新创建的 CustomAllreduce 对象的"假指针"（fptr_t），
//         后续通过 reinterpret_cast 转回 C++ 对象指针使用。
// ============================================================================
fptr_t init_custom_ar(const std::vector<fptr_t>& fake_ipc_ptrs,
                      torch::Tensor& rank_data, int64_t rank,
                      bool fully_connected) {
  // 【参数校验】
  // - 最多支持 8 GPU（受 RankData 中 ptrs[8] 数组大小限制）
  // - 当前仅支持偶数 GPU 数量（两阶段算法需要均匀分区）
  // - rank 必须在有效范围内
  int world_size = fake_ipc_ptrs.size();
  if (world_size > 8)
    throw std::invalid_argument("world size > 8 is not supported");
  if (world_size % 2 != 0)
    throw std::invalid_argument("Odd num gpus is not supported for now");
  if (rank < 0 || rank >= world_size)
    throw std::invalid_argument("invalid rank passed in");

  // 将 Python 传入的"假指针"转换为 C++ Signal 指针数组
  vllm::Signal* ipc_ptrs[8];
  for (int i = 0; i < world_size; i++) {
    ipc_ptrs[i] = reinterpret_cast<vllm::Signal*>(fake_ipc_ptrs[i]);
  }
  // 在堆上创建 CustomAllreduce 对象并返回其地址（作为 fptr_t 句柄）
  return (fptr_t) new vllm::CustomAllreduce(ipc_ptrs, rank_data.data_ptr(),
                                            rank_data.numel(), rank, world_size,
                                            fully_connected);
}

/**
 * Make sure tensor t's data lies completely within ((char)t.data_ptr()) +
 * t.numel() * t.element_size(). This is slightly weaker than t.is_contiguous()
 * because it allows transpose of contiguous slice (i.e. slicing the first
 * dimension). Currently, we require this because stride information is not
 * passed into the kernels and we treat input tensors as flat.
 *
 * Examples
 * A = torch.zeros(3, 3, 3)
 * 1. A: OK
 * 2. A[1:]: OK
 * 3. A.permute(2, 0, 1): OK
 * 4. A[1:].permute(2, 0, 1): OK
 * 5. A[None].expand(2, -1, -1, -1): Not OK
 * 6. A[:, 1:, 1:]: Not OK
 */
// 【弱连续性检查】验证张量的数据在内存中是"扁平"连续的。
// 这比 PyTorch 的 is_contiguous() 更宽松：允许转置连续切片等操作，
// 但不允许有"空洞"的切片（如 A[:, 1:, 1:]）。
// AllReduce kernel 将张量视为一维数组处理，不感知 stride 信息，
// 因此要求输入张量的物理内存布局是连续的。
bool _is_weak_contiguous(torch::Tensor& t) {
  return t.is_contiguous() ||
         (t.storage().nbytes() - t.storage_offset() * t.element_size() ==
          t.numel() * t.element_size());
}

/**
 * Performs an out-of-place allreduce and stores result in out.
 *
 * If _reg_buffer is null, assumes inp.data_ptr() is already IPC-registered.
 * Otherwise, _reg_buffer is assumed to be IPC-registered and inp is first
 * copied into _reg_buffer.
 */
// ============================================================================
// 【核心函数】all_reduce — 执行一次 AllReduce 操作
// ============================================================================
// 这是 Python 端调用 AllReduce 的主入口。流程如下：
//
//   步骤 1: 参数校验（类型匹配、元素数匹配、内存连续性）
//   步骤 2: 如果提供了已注册的 IPC 缓冲区（_reg_buffer），
//           先将输入数据异步拷贝到该缓冲区（cudaMemcpyAsync）；
//           否则直接使用输入数据的指针（假设已注册）。
//   步骤 3: 根据数据类型（float32/float16/bfloat16）分发到对应的
//           CustomAllreduce::allreduce<T> 模板实例，启动 CUDA kernel。
//
// 参数说明：
//   _fa              : init_custom_ar 返回的 CustomAllreduce 对象句柄
//   inp              : 输入张量（本 rank 的局部数据）
//   out              : 输出张量（存放 AllReduce 归约结果）
//   _reg_buffer      : 已 IPC 注册的缓冲区指针，0 表示 inp 已注册
//   reg_buffer_sz_bytes: 注册缓冲区的大小（字节）
// ============================================================================
void all_reduce(fptr_t _fa, torch::Tensor& inp, torch::Tensor& out,
                fptr_t _reg_buffer, int64_t reg_buffer_sz_bytes) {
  // 将 fptr_t 句柄转回 CustomAllreduce 对象指针
  auto fa = reinterpret_cast<vllm::CustomAllreduce*>(_fa);
  // 设置 CUDA 设备上下文，确保操作在正确的 GPU 上执行
  const at::cuda::OptionalCUDAGuard device_guard(device_of(inp));
  auto stream = c10::cuda::getCurrentCUDAStream().stream();

  // 【步骤 1: 参数校验】
  TORCH_CHECK_EQ(inp.scalar_type(), out.scalar_type());
  TORCH_CHECK_EQ(inp.numel(), out.numel());
  TORCH_CHECK(_is_weak_contiguous(out));
  TORCH_CHECK(_is_weak_contiguous(inp));
  auto input_size = inp.numel() * inp.element_size();

  // 【步骤 2: 数据准备】
  // 如果提供了已注册的 IPC 缓冲区，先将输入数据拷贝进去；
  // 否则直接使用输入指针（要求输入张量的内存已经通过 IPC 注册过）。
  auto reg_buffer = reinterpret_cast<void*>(_reg_buffer);
  if (reg_buffer) {
    TORCH_CHECK_LE(input_size, reg_buffer_sz_bytes);
    AT_CUDA_CHECK(cudaMemcpyAsync(reg_buffer, inp.data_ptr(), input_size,
                                  cudaMemcpyDeviceToDevice, stream));
  } else {
    reg_buffer = inp.data_ptr();
  }
  // 【步骤 3: 根据数据类型启动 AllReduce kernel】
  // CustomAllreduce::allreduce<T> 会根据 world_size 和数据量自动选择
  // 一阶段或两阶段算法，并启动对应的 CUDA kernel。
  switch (out.scalar_type()) {
    case at::ScalarType::Float: {
      fa->allreduce<float>(stream, reinterpret_cast<float*>(reg_buffer),
                           reinterpret_cast<float*>(out.data_ptr()),
                           out.numel());
      break;
    }
    case at::ScalarType::Half: {
      fa->allreduce<half>(stream, reinterpret_cast<half*>(reg_buffer),
                          reinterpret_cast<half*>(out.data_ptr()), out.numel());
      break;
    }
#if (__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))
    // bfloat16 需要 SM >= 80（A100 及以上）才支持
    case at::ScalarType::BFloat16: {
      fa->allreduce<nv_bfloat16>(
          stream, reinterpret_cast<nv_bfloat16*>(reg_buffer),
          reinterpret_cast<nv_bfloat16*>(out.data_ptr()), out.numel());
      break;
    }
#endif
    default:
      throw std::runtime_error(
          "custom allreduce only supports float32, float16 and bfloat16");
  }
}

// 【销毁函数】释放 CustomAllreduce 对象，关闭所有已打开的 IPC 句柄。
void dispose(fptr_t _fa) {
  delete reinterpret_cast<vllm::CustomAllreduce*>(_fa);
}

// 【元数据大小】返回 Signal 同步结构体的字节大小，用于 Python 端分配 IPC 共享内存。
int64_t meta_size() { return sizeof(vllm::Signal); }

// ============================================================================
// 【缓冲区注册】register_buffer — 将一组 IPC 指针注册为 AllReduce 可用的缓冲区
// ============================================================================
// 注册后，AllReduce kernel 可以通过 RankData 中存储的指针直接访问
// 其他 rank 上对应位置的 GPU 内存。这是实现跨 GPU 数据读取的关键步骤。
//
// 参数：
//   _fa           : CustomAllreduce 对象句柄
//   fake_ipc_ptrs : 各 rank 上同一缓冲区的 IPC 指针（长度必须等于 world_size）
// ============================================================================
void register_buffer(fptr_t _fa, const std::vector<fptr_t>& fake_ipc_ptrs) {
  auto fa = reinterpret_cast<vllm::CustomAllreduce*>(_fa);
  TORCH_CHECK(fake_ipc_ptrs.size() == fa->world_size_);
  void* ipc_ptrs[8];
  for (int i = 0; i < fake_ipc_ptrs.size(); i++) {
    ipc_ptrs[i] = reinterpret_cast<void*>(fake_ipc_ptrs[i]);
  }
  // 委托给 CustomAllreduce::register_buffer，将指针列表拷贝到 GPU 显存中的 RankData
  fa->register_buffer(ipc_ptrs);
}

// ============================================================================
// 【CUDA Graph 支持】获取 CUDA Graph 捕获期间使用的缓冲区的 IPC 元数据
// ============================================================================
// 在 CUDA Graph 捕获时，AllReduce kernel 的参数（peer 指针）尚未确定。
// 此函数获取这些缓冲区的 IPC MemHandle 和偏移量，
// 以便在各 rank 之间交换后，填入正确的 peer 指针。
//
// 返回值：(handles, offsets)
//   - handles: 所有 graph 缓冲区的 IPC MemHandle 的字节序列
//   - offsets: 各缓冲区相对于其分配基地址的偏移量
// ============================================================================
// Use vector<int64_t> to represent byte data for python binding compatibility.
std::tuple<std::vector<int64_t>, std::vector<int64_t>>
get_graph_buffer_ipc_meta(fptr_t _fa) {
  auto fa = reinterpret_cast<vllm::CustomAllreduce*>(_fa);
  auto [handle, offsets] = fa->get_graph_buffer_ipc_meta();
  std::vector<int64_t> bytes(handle.begin(), handle.end());
  return std::make_tuple(bytes, offsets);
}

// ============================================================================
// 【CUDA Graph 支持】注册 CUDA Graph 使用的缓冲区
// ============================================================================
// 在各 rank 交换 IPC Handle 之后，调用此函数将远端 rank 的指针填入
// RankData 中，使得 CUDA Graph 重放时 AllReduce kernel 能正确访问
// 所有 rank 的数据。
//
// 参数：
//   _fa     : CustomAllreduce 对象句柄
//   handles : 各 rank 的 graph 缓冲区 IPC Handle（二维字节数组）
//   offsets : 各缓冲区相对于基地址的偏移量
// ============================================================================
// Use vector<int64_t> to represent byte data for python binding compatibility.
void register_graph_buffers(fptr_t _fa,
                            const std::vector<std::vector<int64_t>>& handles,
                            const std::vector<std::vector<int64_t>>& offsets) {
  auto fa = reinterpret_cast<vllm::CustomAllreduce*>(_fa);
  std::vector<std::string> bytes;
  bytes.reserve(handles.size());
  for (int i = 0; i < handles.size(); i++) {
    bytes.emplace_back(handles[i].begin(), handles[i].end());
  }
  bytes.reserve(handles.size());
  fa->register_graph_buffers(bytes, offsets);
}

// ============================================================================
// 【共享内存分配】allocate_shared_buffer_and_handle — 分配 IPC 共享缓冲区
// ============================================================================
// 分配一块可通过 CUDA IPC 跨进程共享的 GPU 内存，并返回其 IPC MemHandle。
//
// 流程：
//   1. 在当前 GPU 上分配 size 字节的显存
//   2. 清零并同步（确保内存可访问）
//   3. 通过 cudaIpcGetMemHandle 生成 IPC 句柄
//   4. 返回 (缓冲区指针, IPC 句柄张量)
//
// ROCm 特殊处理：MI200 等 AMD GPU 需要使用 uncached 内存标志
// 以确保跨 GPU 信号传递的正确性。
// ============================================================================
std::tuple<fptr_t, torch::Tensor> allocate_shared_buffer_and_handle(
    int64_t size) {
  auto device_index = c10::cuda::current_device();
  at::DeviceGuard device_guard(at::Device(at::DeviceType::CUDA, device_index));
  void* buffer;
  cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
  auto stream = c10::cuda::getCurrentCUDAStream().stream();
  AT_CUDA_CHECK(cudaThreadExchangeStreamCaptureMode(&mode));

  // Allocate buffer
#if defined(USE_ROCM)
  // data buffers need to be "uncached" for signal on MI200
  AT_CUDA_CHECK(
      hipExtMallocWithFlags((void**)&buffer, size, hipDeviceMallocUncached));
#else
  AT_CUDA_CHECK(cudaMalloc((void**)&buffer, size));
#endif
  AT_CUDA_CHECK(cudaMemsetAsync(buffer, 0, size, stream));
  AT_CUDA_CHECK(cudaStreamSynchronize(stream));
  AT_CUDA_CHECK(cudaThreadExchangeStreamCaptureMode(&mode));

  // Create IPC memhandle for the allocated buffer.
  // Will use it in open_mem_handle.
  auto options =
      torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU);
  auto handle =
      torch::empty({static_cast<int64_t>(sizeof(cudaIpcMemHandle_t))}, options);
  AT_CUDA_CHECK(
      cudaIpcGetMemHandle((cudaIpcMemHandle_t*)handle.data_ptr(), buffer));

  return std::make_tuple(reinterpret_cast<fptr_t>(buffer), handle);
}

// 【打开 IPC 句柄】将其他进程通过 IPC 传递的 MemHandle 打开为本地可访问的设备指针。
// 打开后的指针可以直接通过 NVLink/xGMI 读取远端 GPU 的显存数据。
fptr_t open_mem_handle(torch::Tensor& mem_handle) {
  void* ipc_ptr;
  AT_CUDA_CHECK(cudaIpcOpenMemHandle(
      (void**)&ipc_ptr, *((const cudaIpcMemHandle_t*)mem_handle.data_ptr()),
      cudaIpcMemLazyEnablePeerAccess));
  return reinterpret_cast<fptr_t>(ipc_ptr);
}

// 【释放共享缓冲区】释放之前分配的 IPC 共享 GPU 内存。
void free_shared_buffer(fptr_t buffer) {
  AT_CUDA_CHECK(cudaFree(reinterpret_cast<void*>(buffer)));
}
