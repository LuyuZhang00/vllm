#pragma once

// =============================================================================
// 文件功能概述：
// ops.h 是 vLLM C++/CUDA 算子库的核心头文件，声明了 vLLM 推理引擎在 GPU 上
// 所需的各类底层运算函数。这些函数覆盖了以下几大类操作：
//
//   1. 张量工具：弱引用张量创建、CPU-GPU 张量转换
//   2. 归一化算子：RMS Norm 及其融合版本
//   3. 位置编码算子：Rotary Position Embedding (RoPE)
//   4. 激活函数：SiLU、GELU 及其与乘法的融合变体
//   5. 量化算子：INT8 静态/动态量化、分块量化
//   6. 注意力算子：MLA Decode (DeepSeek V4)
//   7. MoE 算子：动态 4-bit INT MoE CPU 计算
//   8. 通信算子：自定义 AllReduce（NCCL/RCCL）、跨进程共享内存管理
//
// 这些算子的实现分布在 csrc/ 目录下的各 .cu / .cpp 文件中，并通过
// PyTorch 的 torch::library 机制注册到 Python 端，供 vLLM 的 CUDA 或 CPU
// 推理路径调用。
// =============================================================================

#include <optional>
#include <string>
#include <torch/library.h>
#include <tuple>

#include "core/scalar_type.hpp"

#include <vector>

// weak_ref_tensor: 创建一个"弱引用"张量，与原张量共享底层数据指针但不增加引用计数。
// 用途：在 CUDA graph 捕获等场景中，需要确保多个张量视图指向同一块显存，
//       同时避免 PyTorch 引用计数机制干扰 CUDA graph 的内存管理。
// 流程：
//   1. 校验输入张量必须在 CUDA 设备上
//   2. 获取原始数据指针、形状、步长、dtype 等元信息
//   3. 通过 torch::from_blob 用相同指针创建新张量（该张量不拥有内存）
torch::Tensor weak_ref_tensor(torch::Tensor& tensor) {
  // Ensure tensor is on CUDA
  if (!tensor.is_cuda()) {
    throw std::runtime_error("Tensor must be on CUDA device");
  }

  // Get the raw data pointer
  void* data_ptr = tensor.data_ptr();

  // Get tensor sizes and strides
  std::vector<int64_t> sizes = tensor.sizes().vec();
  std::vector<int64_t> strides = tensor.strides().vec();

  // Get tensor options (dtype, device)
  auto options = tensor.options();

  // Create a new tensor from the raw data pointer
  auto new_tensor = torch::from_blob(data_ptr, sizes, strides, options);

  return new_tensor;
}

// rms_norm and fused_add_rms_norm declarations also exist in
// csrc/libtorch_stable/ops.h (torch::stable ABI for CUDA). They remain here
// because the CPU build still uses these torch::Tensor declarations.

// rms_norm: 执行 Root Mean Square Layer Normalization（RMS 归一化）。
// 这是 LLaMA、Qwen 等现代 LLM 中广泛使用的归一化方式，比 LayerNorm 更高效。
// 公式：out = input * weight / sqrt(mean(input^2) + epsilon)
// 参数：
//   out    - 输出张量（与 input 形状相同）
//   input  - 输入张量，形状 [tokens, hidden_dim]
//   weight - 可学习的缩放参数，形状 [hidden_dim]
//   epsilon - 防止除零的极小值
void rms_norm(torch::Tensor& out, torch::Tensor& input, torch::Tensor& weight,
              double epsilon);

// fused_add_rms_norm: 融合了"残差加法"和"RMS 归一化"的算子。
// 在 Transformer 的每个 block 中，子层输出通常为 x + SubLayer(x)，
// 然后对结果做 RMS Norm。此算子将这两步融合为一次 kernel 调用，
// 减少一次显存读写，显著提升性能。
// 计算过程：
//   1. input = input + residual   （原地更新 input 作为残差结果）
//   2. residual = input           （保存残差供下一层使用）
//   3. input = rms_norm(input, weight, epsilon)
// 参数：
//   input   - 输入张量，同时作为残差结果的输出（原地修改）
//   residual - 残差张量，被更新为 input 的原始值
//   weight  - RMS Norm 的缩放参数
//   epsilon - 防止除零的极小值
void fused_add_rms_norm(torch::Tensor& input, torch::Tensor& residual,
                        torch::Tensor& weight, double epsilon);

// fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert:
// 针对 DeepSeek V4 MoE 模型的超融合算子，将以下 5 步操作合并为单次 kernel 调用：
//   1. 对 Query 做 RMS Norm（qnorm）
//   2. 对 Query 应用 RoPE 位置编码
//   3. 对 Key 应用 RoPE 位置编码
//   4. 对 Key/Value 做量化（用于 KV cache 压缩存储）
//   5. 将量化后的 KV 插入 KV cache（通过 slot_mapping 写入对应位置）
//
// 这种超融合设计大幅减少了 DeepSeek V4 模型在 prefill 阶段的显存访问次数。
// 参数：
//   q_in           - 原始 Query 输入 [num_tokens, num_q_heads, head_dim]
//   kv             - 原始 Key/Value 输入 [num_tokens, num_kv_heads, head_dim]
//   k_cache        - KV cache 的物理存储张量（原地写入）
//   slot_mapping   - 每个 token 在 KV cache 中的物理槽位索引
//   position_ids   - 每个 token 的绝对位置 ID
//   cos_sin_cache  - 预计算的 cos/sin 值用于 RoPE
//   q_head_padded  - Q head 数量的对齐值（用于硬件优化）
//   eps            - RMS Norm 的 epsilon
//   cache_block_size - KV cache 的 block 大小（PagedAttention 的页大小）
torch::Tensor fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
    torch::Tensor const& q_in, torch::Tensor const& kv, torch::Tensor& k_cache,
    torch::Tensor const& slot_mapping, torch::Tensor const& position_ids,
    torch::Tensor const& cos_sin_cache, int64_t q_head_padded, double eps,
    int64_t cache_block_size);

// silu_and_mul_per_block_quant: 融合 SiLU 激活 + 逐元素乘法 + 分块量化。
// 在 MoE（Mixture of Experts）的 Expert FFN 中，SwiGLU 激活函数为：
//   output = SiLU(x1) * x2
// 其中 x1, x2 是 input 沿最后一维拆分的两半。
// 此算子在计算完 SwiGLU 后，立即对输出做分块量化（per-block quantization），
// 将结果量化为 INT8/FP8 并输出对应的 scale，避免额外的量化 kernel 调用。
// 参数：
//   out    - 量化后的输出张量
//   input  - 输入张量，最后一维大小为 2 * hidden_dim
//   scales - 输出的量化 scale 张量
//   group_size - 量化分组大小（每 group_size 个元素共享一个 scale）
//   scale_ub   - 可选的 scale 上界（用于截断异常值）
//   is_scale_transposed - scale 是否转置存储
void silu_and_mul_per_block_quant(torch::Tensor& out,
                                  torch::Tensor const& input,
                                  torch::Tensor& scales, int64_t group_size,
                                  std::optional<torch::Tensor> scale_ub,
                                  bool is_scale_transposed);

// rotary_embedding also exist in csrc/libtorch_stable/ops.h (torch::stable
// ABI for CUDA). It remains here because the CPU build still uses these
// torch::Tensor declarations.

// rotary_embedding: 实现 Rotary Position Embedding（RoPE）。
// RoPE 是 LLaMA、Qwen、Mistral 等现代 LLM 的标准位置编码方式，
// 通过将位置信息编码为旋转矩阵，使 attention 天然具有相对位置感知能力。
// 计算过程：
//   1. 从 cos_sin_cache 中按 position_ids 查找对应的 cos/sin 值
//   2. 将 query（和可选的 key）的最后一维拆分为两半
//   3. 对两半应用 2D 旋转：[x1, x2] -> [x1*cos - x2*sin, x1*sin + x2*cos]
// 参数：
//   positions      - 每个 token 的位置 ID [num_tokens]
//   query          - Query 张量，原地旋转更新
//   key            - 可选的 Key 张量，原地旋转更新
//   head_size      - 每个注意力头的维度
//   cos_sin_cache  - 预计算的 cos/sin 缓存表 [max_position, rope_dim]
//   is_neox        - 是否使用 GPT-NeoX 风格的 RoPE（影响拆分维度顺序）
//   rope_dim_offset - RoPE 维度偏移（支持部分维度不做 RoPE 的场景）
//   inverse        - 是否应用逆旋转（用于某些解码场景）
void rotary_embedding(torch::Tensor& positions, torch::Tensor& query,
                      std::optional<torch::Tensor> key, int64_t head_size,
                      torch::Tensor& cos_sin_cache, bool is_neox,
                      int64_t rope_dim_offset, bool inverse);

// silu_and_mul: 计算 SwiGLU 激活函数：out = SiLU(x1) * x2
// 其中 input 沿最后一维拆分为 x1 和 x2。
// SiLU(x) = x * sigmoid(x)，也称为 Swish 激活函数。
// 这是 LLaMA、Qwen 等模型 FFN 层的核心激活函数。
void silu_and_mul(torch::Tensor& out, torch::Tensor& input);

// silu_and_mul_clamp: silu_and_mul 的带截断版本。
// 在计算 SiLU(x1) * x2 后，对结果做 clamp(-limit, limit) 截断，
// 防止极端值导致数值溢出，常用于训练不稳定的场景。
void silu_and_mul_clamp(torch::Tensor& out, torch::Tensor& input, double limit);

// silu_and_mul_quant: 融合 SiLU*Mul + 量化的算子。
// 计算完 SwiGLU 后，立即使用给定的 scale 做量化输出，
// 用于 MoE Expert 的输出直接量化存储，减少额外 kernel 调用。
void silu_and_mul_quant(torch::Tensor& out, torch::Tensor& input,
                        torch::Tensor& scale);

// persistent_masked_m_silu_mul_quant: 针对 MoE 的持久化融合 kernel。
// 在 MoE 中，不同 Expert 被分配到不同数量的 token（由 routing 决定），
// 此算子将 masked SiLU*Mul 和量化融合为单个 persistent kernel：
//   1. 根据 counts 确定每个 Expert 处理的有效 token 数
//   2. 对每个 Expert 的有效 token 执行 SiLU(x1)*x2
//   3. 对结果做分组量化，输出量化值 y_q 和对应 scale y_s
// 使用 persistent kernel 设计可以避免 kernel 启动开销并提高 GPU 利用率。
// 参数：
//   input  - 输入 [E, T, 2*H]，E=Expert数, T=最大token数, H=隐藏维度
//   counts - 每个 Expert 处理的有效 token 数 [E]
//   y_q    - 量化输出 [E, T, H]
//   y_s    - 量化 scale [E, T, H//group_size]
//   use_ue8m0 - 是否使用 UE8M0 格式的 scale（用于 FP8 量化）
void persistent_masked_m_silu_mul_quant(
    const at::Tensor& input,   // (E, T, 2*H)
    const at::Tensor& counts,  // (E)
    at::Tensor& y_q,           // (E, T, H) [OUT]
    at::Tensor& y_s,           // (E, T, H//group_size) [OUT]
    bool use_ue8m0);

// gelu_and_mul: GELU 激活与逐元素乘法的融合算子。
// 计算 out = GELU(x1) * x2，其中 input 沿最后一维拆分为 x1, x2。
// 使用近似 GELU 公式：GELU(x) = x * 0.5 * (1 + erf(x/sqrt(2)))
// 常用于部分 Transformer 模型的 FFN 层（如 GPT-2、BLOOM 等）。
void gelu_and_mul(torch::Tensor& out, torch::Tensor& input);

// gelu_tanh_and_mul: 使用 tanh 近似的 GELU + 逐元素乘法融合算子。
// 计算 out = GELU_tanh(x1) * x2。
// tanh 近似公式：GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))
// 毉标准 erf 版 GELU 更快，但精度略有损失。
void gelu_tanh_and_mul(torch::Tensor& out, torch::Tensor& input);

// gelu_new: 新版 GELU 激活函数实现。
// 用于需要精确 GELU 计算的场景。
void gelu_new(torch::Tensor& out, torch::Tensor& input);

// gelu_fast: 快速 GELU 激活函数实现。
// 使用更快的近似计算，牺牲少量精度换取更高吞吐。
void gelu_fast(torch::Tensor& out, torch::Tensor& input);

// gelu_quick: 极速 GELU 激活函数实现。
// 使用最简单的近似公式，适用于对延迟极度敏感的场景。
void gelu_quick(torch::Tensor& out, torch::Tensor& input);

// cutlass_mla_decode: 使用 CUTLASS 库实现的 Multi-head Latent Attention (MLA) Decode。
// MLA 是 DeepSeek V2/V3 模型的核心注意力机制，其特点是将 KV cache 压缩为
// 低秩表示（latent），大幅减少推理时的 KV cache 显存占用。
// 与标准 MHA 不同，MLA 将 Key 分为两部分：
//   - q_nope：不含位置信息的部分（对应压缩后的 latent）
//   - q_pe：包含位置信息的部分（对应 RoPE 编码）
// Decode 阶段每次只为一个 token 计算 attention，因此使用 page_table
// 进行 PagedAttention 的地址查找。
// 参数：
//   out                  - 输出张量
//   q_nope               - Query 的非位置部分
//   q_pe                 - Query 的位置编码部分
//   kv_c_and_k_pe_cache  - 合并存储的 KV 压缩缓存和 K_PE 缓存
//   seq_lens             - 每个请求的有效序列长度
//   page_table           - PagedAttention 的页表，逻辑块到物理块的映射
//   scale                - attention 缩放因子（通常为 1/sqrt(head_dim)）
void cutlass_mla_decode(torch::Tensor const& out, torch::Tensor const& q_nope,
                        torch::Tensor const& q_pe,
                        torch::Tensor const& kv_c_and_k_pe_cache,
                        torch::Tensor const& seq_lens,
                        torch::Tensor const& page_table, double scale);

// get_cuda_view_from_cpu_tensor: 从 CPU 张量获取 CUDA 设备上的视图。
// 用于需要在 GPU 上直接访问 CPU 张量数据的场景（如 pinned memory）。
// 要求 cpu_tensor 必须是 pinned memory（通过 pin_memory() 分配），
// 这样 GPU 可以通过 DMA 直接读取，无需显式拷贝。
torch::Tensor get_cuda_view_from_cpu_tensor(torch::Tensor& cpu_tensor);

// static_scaled_int8_quant: 静态 scale 的 INT8 量化。
// 使用预先确定的固定 scale（来自校准数据集）将 FP16/BF16 输入量化为 INT8。
// 公式：out = round(input / scale) + azp
// 其中 azp (asymmetric zero point) 是可选的非对称量化零点。
// 静态量化的 scale 在部署前确定，运行时不再变化，适合 latency 敏感场景。
void static_scaled_int8_quant(torch::Tensor& out, torch::Tensor const& input,
                              torch::Tensor const& scale,
                              std::optional<torch::Tensor> const& azp);

// dynamic_scaled_int8_quant: 动态 scale 的 INT8 量化。
// 运行时根据每组数据的实际范围动态计算 scale，精度更高但有额外开销。
// 公式：scale = max(|input|) / 127，out = round(input / scale) + azp
// 动态量化在每次前向传播时重新计算 scale，适合精度要求高的场景。
void dynamic_scaled_int8_quant(torch::Tensor& out, torch::Tensor const& input,
                               torch::Tensor& scales,
                               std::optional<torch::Tensor> const& azp);

// dynamic_4bit_int_moe_cpu: 在 CPU 上执行动态 4-bit 量化的 MoE 计算。
// 用于 CPU offloading 场景，当 GPU 显存不足时将部分 Expert 计算卸载到 CPU。
// 流程：
//   1. 根据 topk_ids 选择每个 token 对应的 Expert
//   2. 从 w13_packed 和 w2_packed 中解包 4-bit 权重
//   3. 动态计算量化 scale
//   4. 执行 Expert 的 FFN 计算（gate_proj + up_proj -> 激活 -> down_proj）
//   5. 用 topk_weights 加权聚合各 Expert 的输出
// 参数：
//   x        - 输入张量 [num_tokens, hidden_dim]
//   topk_ids - 每个 token 选择的 top-K Expert ID [num_tokens, top_k]
//   topk_weights - 每个 Expert 的路由权重 [num_tokens, top_k]
//   w13_packed   - Expert 1 和 3（gate_proj 和 up_proj）的 4-bit 打包权重
//   w2_packed    - Expert 2（down_proj）的 4-bit 打包权重
//   H        - 隐藏维度
//   I        - FFN 中间维度
//   I2       - FFN 中间维度的第二部分
//   group_size - 量化分组大小
//   apply_router_weight_on_input - 是否将路由权重应用到输入端
//   activation_kind - 激活函数类型（0=SiLU, 1=GELU 等）
torch::Tensor dynamic_4bit_int_moe_cpu(
    torch::Tensor x, torch::Tensor topk_ids, torch::Tensor topk_weights,
    torch::Tensor w13_packed, torch::Tensor w2_packed, int64_t H, int64_t I,
    int64_t I2, int64_t group_size, bool apply_router_weight_on_input,
    int64_t activation_kind);

// =============================================================================
// 以下为自定义 AllReduce 通信算子的声明，用于 Tensor Parallel 场景下的
// 高性能跨 GPU 通信。该实现绕过 NCCL 的通用路径，直接使用 CUDA IPC
// 共享内存实现 all-reduce，以降低小 tensor 通信的延迟。
// =============================================================================

// fptr_t: 功能指针类型，指向 C++ 端的 CustomAllreduce 对象。
// Python 端通过此指针调用底层 C++ 对象的方法。
using fptr_t = int64_t;

// init_custom_ar: 初始化自定义 AllReduce 通信器。
// 流程：
//   1. 通过 CUDA IPC 打开其他 rank 的共享内存句柄
//   2. 在各 rank 间建立对等通信所需的内存映射
//   3. 创建 CustomAllreduce 对象并返回其功能指针
// 参数：
//   fake_ipc_ptrs  - 其他 rank 通过 IPC 共享的内存指针列表
//   rank_data      - 当前 rank 的元数据
//   rank           - 当前进程的 rank 编号
//   fully_connected - 是否为全连接拓扑（所有 rank 两两直连）
fptr_t init_custom_ar(const std::vector<int64_t>& fake_ipc_ptrs,
                      torch::Tensor& rank_data, int64_t rank,
                      bool fully_connected);

// all_reduce: 执行自定义 AllReduce 操作。
// 将所有 rank 的输入张量求和，并将结果写入输出张量。
// 使用 NCCL ring/tree 算法的替代方案，对小 tensor 更高效。
// 参数：
//   _fa             - CustomAllreduce 对象的功能指针
//   inp             - 输入张量（当前 rank 的本地数据）
//   out             - 输出张量（存放 all-reduce 结果）
//   reg_buffer      - 注册的共享缓冲区指针
//   reg_buffer_sz_bytes - 共享缓冲区大小（字节）
void all_reduce(fptr_t _fa, torch::Tensor& inp, torch::Tensor& out,
                fptr_t reg_buffer, int64_t reg_buffer_sz_bytes);

// dispose: 释放自定义 AllReduce 通信器，清理所有关联的共享内存资源。
void dispose(fptr_t _fa);

// meta_size: 返回 CustomAllreduce 对象所需的元数据大小（字节）。
// 用于分配 rank_data 缓冲区时确定大小。
int64_t meta_size();

// register_buffer: 注册 IPC 共享缓冲区，使通信器能够访问其他 rank 的内存。
void register_buffer(fptr_t _fa, const std::vector<int64_t>& fake_ipc_ptrs);

// get_graph_buffer_ipc_meta: 获取 CUDA graph 缓冲区的 IPC 元信息。
// 用于在多 GPU 间共享 CUDA graph 的执行状态。
// 返回：(handles, offsets) - IPC 句柄列表和内存偏移列表。
std::tuple<std::vector<int64_t>, std::vector<int64_t>>
get_graph_buffer_ipc_meta(fptr_t _fa);

// register_graph_buffers: 注册 CUDA graph 的共享缓冲区。
// 在使用 CUDA graph 加速推理时，需要在各 rank 间注册 graph 使用的缓冲区。
void register_graph_buffers(fptr_t _fa,
                            const std::vector<std::vector<int64_t>>& handles,
                            const std::vector<std::vector<int64_t>>& offsets);

// allocate_shared_buffer_and_handle: 分配一块 CUDA IPC 共享内存，并返回其句柄。
// 返回：(buffer_ptr, handle_tensor) - 缓冲区指针和用于 IPC 传递的句柄张量。
std::tuple<int64_t, torch::Tensor> allocate_shared_buffer_and_handle(
    int64_t size);

// open_mem_handle: 打开由其他进程通过 IPC 传递的内存句柄。
// 返回该内存在当前进程中的地址。
int64_t open_mem_handle(torch::Tensor& mem_handle);

// free_shared_buffer: 释放之前分配的共享内存缓冲区。
void free_shared_buffer(int64_t buffer);

// =============================================================================
// ROCm (AMD GPU) 专用：自定义量化 AllReduce 通信算子。
// 在 AMD GPU 上，使用 quantized all-reduce 算法，通过量化传输数据来减少
// 跨 GPU 通信的带宽需求，同时保持足够的数值精度。
// =============================================================================

#ifdef USE_ROCM

// init_custom_qr: 初始化 ROCm 平台的量化 AllReduce 通信器。
// 参数：
//   rank        - 当前进程的 rank 编号
//   world_size  - 总进程数（GPU 数量）
//   qr_max_size - 可选的最大量化 reduce 缓冲区大小
fptr_t init_custom_qr(int64_t rank, int64_t world_size,
                      std::optional<int64_t> qr_max_size = std::nullopt);

// qr_destroy: 销毁量化 AllReduce 通信器，释放所有资源。
void qr_destroy(fptr_t _fa);

// qr_get_handle: 获取通信器的 IPC 句柄，用于在进程间共享。
torch::Tensor qr_get_handle(fptr_t _fa);

// qr_open_handles: 打开其他 rank 的 IPC 句柄，建立跨进程通信连接。
void qr_open_handles(fptr_t _fa, const std::vector<torch::Tensor>& handles);

// qr_all_reduce: 执行量化 AllReduce 操作。
// 使用量化技术（如 INT8/FP8）压缩中间传输数据，减少带宽占用。
// 参数：
//   _fa          - 量化 AllReduce 通信器的功能指针
//   inp          - 输入张量
//   out          - 输出张量（存放 all-reduce 结果）
//   quant_level  - 量化级别（控制压缩率和精度的平衡）
//   cast_bf2half - 是否将 BF16 转为 FP16 以兼容某些硬件
void qr_all_reduce(fptr_t _fa, torch::Tensor& inp, torch::Tensor& out,
                   int64_t quant_level, bool cast_bf2half = false);

// qr_max_size: 返回量化 reduce 缓冲区的最大大小。
int64_t qr_max_size();

#endif

// =============================================================================
// CUDA (NVIDIA GPU) 专用：融合 AllReduce + RMS Norm 算子。
// 在 Tensor Parallel 场景中，注意力层的 Q/K/V 计算后需要 all-reduce，
// 然后对结果做 RMS Norm。此融合算子将通信和归一化合并，减少 kernel 启动次数。
// =============================================================================

#ifndef USE_ROCM

// minimax_allreduce_rms: 融合 AllReduce + RMS Norm。
// 流程：
//   1. 对所有 rank 的输入执行 all-reduce（求和）
//   2. 对 all-reduce 结果执行 RMS Norm
// 这将两个独立的 kernel 合并为一个，减少了一次显存读写。
// 参数：
//   input       - 当前 rank 的输入张量
//   norm_weight - RMS Norm 的可学习缩放参数
//   workspace   - AllReduce 使用的工作空间缓冲区
//   rank        - 当前进程的 rank 编号
//   nranks      - 总进程数
//   eps         - RMS Norm 的 epsilon 值
torch::Tensor minimax_allreduce_rms(torch::Tensor const& input,
                                    torch::Tensor const& norm_weight,
                                    torch::Tensor workspace, int64_t const rank,
                                    int64_t const nranks, double const eps);

// minimax_allreduce_rms_qk: 融合 AllReduce + 双 RMS Norm（Q 和 K）。
// 在注意力计算中，Q 和 K 分别需要做 RMS Norm，此算子将 all-reduce 和
// 两次 RMS Norm 融合为单个 kernel 调用，进一步减少显存访问。
// 流程：
//   1. 对 all-reduce 后的 QKV 数据按 q_size/kv_size 拆分
//   2. 对 Q 部分执行 RMS Norm（使用 norm_weight_q）
//   3. 对 K 部分执行 RMS Norm（使用 norm_weight_k）
//   4. 返回归一化后的 Q 和 K
// 参数：
//   qkv          - 融合的 QKV 张量
//   norm_weight_q - Q 的 RMS Norm 缩放参数
//   norm_weight_k - K 的 RMS Norm 缩放参数
//   workspace    - AllReduce 工作空间
//   q_size       - Q 部分的大小
//   kv_size      - K 部分的大小
//   rank         - 当前 rank 编号
//   nranks       - 总进程数
//   eps          - RMS Norm epsilon
std::tuple<torch::Tensor, torch::Tensor> minimax_allreduce_rms_qk(
    torch::Tensor qkv, torch::Tensor const& norm_weight_q,
    torch::Tensor const& norm_weight_k, torch::Tensor workspace,
    int64_t const q_size, int64_t const kv_size, int64_t const rank,
    int64_t const nranks, double const eps);

#endif
