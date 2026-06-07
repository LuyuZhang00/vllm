#pragma once

// =============================================================================
// 中文注释：KV Cache 操作模块头文件
// =============================================================================
// 本文件声明了 vLLM 推理引擎中与 KV Cache（键值缓存）相关的核心 C++ 操作函数。
//
// 在 LLM 推理过程中，为了加速自回归生成，需要将已计算过的 Key 和 Value 张量
// 缓存下来，避免重复计算。这些缓存数据存储在 GPU 显存中，采用分页管理（类似
// 操作系统的虚拟内存分页），每个"页"称为一个 KV Block。
//
// 本模块提供的核心功能包括：
//   1. swap_blocks / swap_blocks_batch：
//      KV Block 的搬运操作，用于在 GPU 显存与 CPU 内存之间交换数据，
//      典型场景是 preemption（抢占）时将低优先级请求的 KV cache 换出到 CPU，
//      或在恢复时换入到 GPU。
//
//   2. reshape_and_cache / reshape_and_cache_flash：
//      将新计算的 K/V 张量 reshape 并写入 KV cache 的指定 slot 中。
//      slot_mapping 将 token 级别的逻辑位置映射到 KV cache 中的物理存储位置。
//
//   3. concat_and_cache_mla / concat_and_cache_mla_rope_fused：
//      MLA（Multi-head Latent Attention）架构专用的 KV 缓存写入操作，
//      将压缩后的 kv_c 和旋转位置编码 k_pe 拼接后写入缓存。
//
//   4. gather_and_maybe_dequant_cache / cp_gather_cache：
//      从 KV cache 中按 block table 收集指定序列的 KV 数据，
//      可选地进行反量化（如 FP8 -> BF16），用于 chunked prefill 等场景。
//
//   5. cp_gather_and_upconvert_fp8_kv_cache：
//      专门用于 FP8 格式 KV cache 的收集与向上转换（FP8 -> BF16）。
//
//   6. indexer_k_quant_and_cache / cp_gather_indexer_k_quant_cache：
//      Indexer 量化相关的 K cache 写入和读取操作，
//      用于对 K 进行分块量化后存入缓存，以及从缓存中收集并反量化。
//
//   7. concat_mla_q：
//      MLA 架构中将 query 的 nope（无位置编码）部分和 rope（旋转位置编码）
//      部分拼接在一起。
//
// 设计要点：
//   - 所有操作都通过 slot_mapping 或 block_table 将逻辑地址映射到物理地址，
//     这是 PagedAttention 分页缓存机制的核心。
//   - 支持多种 KV cache 数据类型（FP16, BF16, FP8 等），通过 kv_cache_dtype
//     字符串参数和 scale 参数控制量化/反量化。
//   - 这些函数的 CUDA kernel 实现位于同目录下的 .cu 文件中。
// =============================================================================

#include <torch/all.h>
#include <c10/util/Optional.h>

#include <map>
#include <vector>

// =============================================================================
// 1. KV Block 交换操作（用于 Preemption / 换入换出）
// =============================================================================

// 中文注释：将 src 张量中的 KV block 批量拷贝到 dst 张量中。
// src 和 dst 可以分别位于 GPU 或 CPU 上，实现 GPU<->CPU 之间的 block 数据交换。
// 典型场景：当 GPU 显存不足时，将低优先级请求的 KV cache 换出（evict）到 CPU 内存，
// 待需要时再换入（restore）回 GPU。
//
// 参数说明：
//   src: 源 KV cache 张量，可以位于 GPU 或 CPU
//   dst: 目标 KV cache 张量，可以位于 GPU 或 CPU
//   block_size_in_bytes: 每个 block 的字节大小，用于确定拷贝粒度
//   block_mapping: 一个二维张量，每行 [src_block_idx, dst_block_idx]，
//                  表示源 block 索引到目标 block 索引的映射关系
void swap_blocks(torch::Tensor& src, torch::Tensor& dst,
                 int64_t block_size_in_bytes,
                 const torch::Tensor& block_mapping);

// 中文注释：批量版本的 block 交换操作，支持多个 block 的并行拷贝。
// 相比 swap_blocks，此函数接受预计算好的源/目标指针数组和大小数组，
// 避免了在 kernel 内部进行地址计算，效率更高。
//
// 参数说明：
//   src_ptrs: 源地址指针数组（每个元素是一个 block 的起始地址）
//   dst_ptrs: 目标地址指针数组（每个元素是一个 block 的起始地址）
//   sizes: 每个 block 需要拷贝的字节数
//   is_src_access_order_any: 源数据的访问顺序是否任意（影响内存访问优化策略）。
//                            如果为 true，表示源 block 的访问顺序不连续，
//                            需要使用更保守的内存访问模式以避免性能问题。
void swap_blocks_batch(const torch::Tensor& src_ptrs,
                       const torch::Tensor& dst_ptrs,
                       const torch::Tensor& sizes,
                       bool is_src_access_order_any);

// =============================================================================
// 2. KV Cache 写入操作（将新计算的 K/V 存入缓存）
// =============================================================================

// 中文注释：将当前 step 新计算的 Key 和 Value 张量 reshape 并写入 KV cache 中。
// 这是 vLLM KV cache 写入的核心操作，在模型 forward 每一步生成新 token 后调用。
//
// 工作流程：
//   1) 读取 slot_mapping 获取每个 token 对应的 KV cache 物理存储位置（slot）。
//   2) 将 key/value 从 [num_tokens, num_heads, head_dim] 的形状
//      reshape 为 key_cache/value_cache 所需的块内布局。
//   3) 将 reshape 后的数据写入 key_cache/value_cache 的对应 slot 中。
//
// PagedAttention 的核心思想：每个请求的 KV cache 不需要在物理显存上连续存放，
// 而是通过 slot_mapping 将 token 映射到分散的物理 slot 中。
// 这样可以避免显存碎片化，提高显存利用率。
//
// 参数说明：
//   key: 当前 step 新计算的 Key 张量，形状 [num_tokens, num_heads, head_dim]
//   value: 当前 step 新计算的 Value 张量，形状 [num_tokens, num_heads, head_dim]
//   key_cache: 全局 Key cache 池，形状 [num_blocks, block_size, num_heads, head_dim]
//   value_cache: 全局 Value cache 池，形状 [num_blocks, block_size, num_heads, head_dim]
//   slot_mapping: 每个 token 对应的物理 slot 索引，形状 [num_tokens]
//                 slot = block_id * block_size + offset_in_block
//   kv_cache_dtype: KV cache 的数据类型字符串（如 "auto", "fp8" 等）
//   k_scale: Key 的量化缩放因子
//   v_scale: Value 的量化缩放因子
void reshape_and_cache(torch::Tensor& key, torch::Tensor& value,
                       torch::Tensor& key_cache, torch::Tensor& value_cache,
                       torch::Tensor& slot_mapping,
                       const std::string& kv_cache_dtype,
                       torch::Tensor& k_scale, torch::Tensor& v_scale);

void reshape_and_cache_flash(torch::Tensor& key, torch::Tensor& value,
                             torch::Tensor& key_cache,
                             torch::Tensor& value_cache,
                             torch::Tensor& slot_mapping,
                             const std::string& kv_cache_dtype,
                             torch::Tensor& k_scale, torch::Tensor& v_scale);

void concat_and_cache_mla(torch::Tensor& kv_c, torch::Tensor& k_pe,
                          torch::Tensor& kv_cache, torch::Tensor& slot_mapping,
                          const std::string& kv_cache_dtype,
                          torch::Tensor& scale);

// NOTE: k_pe and kv_c order is flipped compared to concat_and_cache_mla
void concat_and_cache_mla_rope_fused(
    torch::Tensor& positions, torch::Tensor& q_pe, torch::Tensor& k_pe,
    torch::Tensor& kv_c, torch::Tensor& rope_cos_sin_cache, bool rope_is_neox,
    torch::Tensor& kv_cache_slot_mapping, torch::Tensor& kv_cache,
    const std::string& kv_cache_dtype, torch::Tensor& kv_cache_quant_scale);

// Just for unittest
void convert_fp8(torch::Tensor& dst_cache, torch::Tensor& src_cache,
                 const double scale, const std::string& kv_cache_dtype);

void gather_and_maybe_dequant_cache(
    torch::Tensor const& src_cache,     // [NUM_BLOCKS, BLOCK_SIZE, ENTRIES...]
    torch::Tensor const& dst,           // [TOT_TOKENS, ENTRIES...]
    torch::Tensor const& block_table,   // [BATCH, BLOCK_INDICES]
    torch::Tensor const& cu_seq_lens,   // [BATCH+1]
    torch::Tensor const& token_to_seq,  // [MAX_TOKEN_ACROSS_CHUNKS]
    int64_t num_tokens, const std::string& kv_cache_dtype,
    torch::Tensor const& scale,
    std::optional<torch::Tensor> seq_starts = std::nullopt);

// TODO(hc): cp_gather_cache need support scaled kvcahe in the future.
void cp_gather_cache(
    torch::Tensor const& src_cache,    // [NUM_BLOCKS, BLOCK_SIZE, ENTRIES...]
    torch::Tensor const& dst,          // [TOT_TOKENS, ENTRIES...]
    torch::Tensor const& block_table,  // [BATCH, BLOCK_INDICES]
    torch::Tensor const& cu_seq_lens,  // [BATCH+1]
    int64_t batch_size, std::optional<torch::Tensor> seq_starts = std::nullopt);

// Gather and upconvert FP8 KV cache to BF16 workspace
void cp_gather_and_upconvert_fp8_kv_cache(
    torch::Tensor const& src_cache,         // [NUM_BLOCKS, BLOCK_SIZE, 656]
    torch::Tensor const& dst,               // [TOT_TOKENS, 576]
    torch::Tensor const& block_table,       // [BATCH, BLOCK_INDICES]
    torch::Tensor const& seq_lens,          // [BATCH]
    torch::Tensor const& workspace_starts,  // [BATCH]
    int64_t batch_size);

// Indexer K quantization and cache function
void indexer_k_quant_and_cache(
    torch::Tensor& k,             // [num_tokens, head_dim]
    torch::Tensor& kv_cache,      // [num_blocks, block_size, cache_stride]
    torch::Tensor& slot_mapping,  // [num_tokens]
    int64_t quant_block_size,     // quantization block size
    const std::string& scale_fmt);

// Concatenate query nope and rope for MLA/DSA attention
void concat_mla_q(
    torch::Tensor& ql_nope,  // [num_tokens, num_heads, nope_dim]
    torch::Tensor& q_pe,     // [num_tokens, num_heads, rope_dim]
    torch::Tensor& q_out);   // [num_tokens, num_heads, nope_dim + rope_dim]

// Extract function to gather quantized K cache
void cp_gather_indexer_k_quant_cache(
    const torch::Tensor& kv_cache,  // [num_blocks, block_size, cache_stride]
    torch::Tensor& dst_k,           // [num_tokens, head_dim]
    torch::Tensor& dst_scale,  // [num_tokens, head_dim / quant_block_size * 4]
    const torch::Tensor& block_table,   // [batch_size, num_blocks]
    const torch::Tensor& cu_seq_lens);  // [batch_size + 1]
