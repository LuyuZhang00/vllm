// =============================================================================
// 中文注释：MoE (Mixture of Experts) CUDA 算子的 Torch 绑定注册文件
//
// 本文件是 MoE 模块的入口，将所有 CUDA kernel 注册为 PyTorch 自定义算子，
// 使得 Python 层可以通过 torch.ops.xxx 方式调用这些高性能 GPU kernel。
//
// 注册的主要算子包括：
// 1. topk_softmax / topk_sigmoid / topk_softplus_sqrt —— 路由(Routing)算子，
//    对 gating output 做激活函数 + top-k 选择，输出每个 token 选中的专家及其权重。
// 2. moe_sum —— 将多个专家的输出按权重求和，得到最终 MoE 层输出。
// 3. moe_align_block_size —— 将 token 按专家分组并对齐到 block_size 边界，
//    这是分块 MoE GEMM (如 Triton fused_moe) 的前置步骤。
// 4. moe_wna16_gemm / moe_wna16_marlin_gemm —— 量化 MoE GEMM 算子，
//    支持 WNA16 (Weight-Only N-bit with 16-bit activation) 量化推理。
// 5. moe_permute / moe_unpermute —— MoE token 排列/反排列算子，
//    将 token 按专家排序以便连续计算，计算完成后还原原始顺序。
// 6. grouped_topk —— DeepSeek V3 等模型使用的分组 top-k 路由算子。
// 7. dsv3_router_gemm —— DeepSeek V3 专用的 router GEMM 优化算子。
// 8. shuffle_rows —— 行重排辅助算子。
// =============================================================================

#include "core/registration.h"
#include "moe_ops.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, m) {
  // 中文注释：topk_softmax —— 对 gating_output [num_tokens, num_experts] 做 softmax 激活，
  // 然后选出 top-k 个专家，输出 topk_weights（权重）和 topk_indices（专家索引）。
  // 适用于标准 MoE 路由（如 Mixtral）。
  // Apply topk softmax to the gating outputs.
  m.def(
      "topk_softmax(Tensor! topk_weights, Tensor! topk_indices, Tensor! "
      "token_expert_indices, Tensor gating_output, bool renormalize, Tensor? "
      "bias) -> ()");
  m.impl("topk_softmax", torch::kCUDA, &topk_softmax);

  // 中文注释：topk_sigmoid —— 对 gating_output 做 sigmoid 激活后选 top-k 专家。
  // 适用于使用 sigmoid 作为路由激活函数的模型（如 DBRX）。
  // Apply topk sigmoid to the gating outputs.
  m.def(
      "topk_sigmoid(Tensor! topk_weights, Tensor! topk_indices, Tensor! "
      "token_expert_indices, Tensor gating_output, bool renormalize, Tensor? "
      "bias) -> ()");
  m.impl("topk_sigmoid", torch::kCUDA, &topk_sigmoid);

  // 中文注释：topk_softplus_sqrt —— 对 gating_output 做 softplus + sqrt 激活后选 top-k 专家。
  // 适用于 DeepSeek V2/V3 等使用 softplus-sqrt 路由激活函数的模型。
  // 支持 correction_bias（校正偏置）和 hash MoE 模式（通过 tid2eid 查找表直接获取专家索引）。
  m.def(
      "topk_softplus_sqrt(Tensor! topk_weights, Tensor! topk_indices, Tensor! "
      "token_expert_indices, Tensor gating_output, bool renormalize, float "
      "routed_scaling_factor, Tensor? "
      "bias, Tensor? input_ids, Tensor? tid2eid) -> ()");
  m.impl("topk_softplus_sqrt", torch::kCUDA, &topk_softplus_sqrt);

  // 中文注释：moe_sum —— 将多个专家的输出按 top-k 权重加权求和。
  // 输入形状 [num_tokens, topk, hidden_size]，输出形状 [num_tokens, hidden_size]。
  // 这是 MoE 层最后一步：将各专家的部分结果聚合为最终输出。
  // Calculate the result of moe by summing up the partial results
  // from all selected experts.
  m.def("moe_sum(Tensor input, Tensor! output) -> ()");
  m.impl("moe_sum", torch::kCUDA, &moe_sum);

  // 中文注释：moe_align_block_size —— MoE token 分组对齐算子。
  // 核心功能：将每个 token 按其选中的专家分组，并将每组的 token 数量向上对齐到 block_size 的倍数。
  // 这样做的目的是让后续的分块 MoE GEMM（如 Triton fused_moe kernel）可以按固定 block 大小处理，
  // 避免边界检查开销，同时通过 padding 保证每个 block 都是满的。
  // 输出：sorted_token_ids（排序后的 token 索引）、experts_ids（每个 block 对应的专家 ID）、
  //       num_tokens_post_pad（padding 后的总 token 数）。
  // Aligning the number of tokens to be processed by each expert such
  // that it is divisible by the block size.
  m.def(
      "moe_align_block_size(Tensor topk_ids, int num_experts,"
      "                     int block_size, Tensor! sorted_token_ids,"
      "                     Tensor! experts_ids,"
      "                     Tensor! num_tokens_post_pad,"
      "                     Tensor? maybe_expert_map) -> ()");
  m.impl("moe_align_block_size", torch::kCUDA, &moe_align_block_size);

  // 中文注释：batched_moe_align_block_size —— 批量版本的 token 分组对齐算子。
  // 用于 multi-step scheduling 等场景，一次处理多个 micro-batch 的 token 分配。
  // Aligning the number of tokens to be processed by each expert such
  // that it is divisible by the block size, but for the batched case.
  m.def(
      "batched_moe_align_block_size(int max_tokens_per_batch,"
      "                     int block_size, Tensor expert_num_tokens,"
      "                     Tensor! sorted_token_ids,"
      "                     Tensor! experts_ids,"
      "                     Tensor! num_tokens_post_pad) -> ()");
  m.impl("batched_moe_align_block_size", torch::kCUDA,
         &batched_moe_align_block_size);

  // 中文注释：moe_lora_align_block_size —— 带 LoRA 适配器的 MoE token 分组对齐算子。
  // 在 MoE + LoRA 场景下，不同 token 可能使用不同的 LoRA 适配器，
  // 本算子将 token 按 (LoRA ID, Expert ID) 双重分组并对齐到 block_size，
  // 使得后续 kernel 可以同时处理 LoRA 和非 LoRA token。
  // Aligning the number of tokens to be processed by each expert such
  // that it is divisible by the block size.
  m.def(
      "moe_lora_align_block_size(Tensor topk_ids,"
      "                     Tensor token_lora_mapping,"
      "                     int num_experts,"
      "                     int block_size, int max_loras, "
      "                     int max_num_tokens_padded, "
      "                     int max_num_m_blocks, "
      "                     Tensor !sorted_token_ids,"
      "                     Tensor !experts_ids,"
      "                     Tensor !num_tokens_post_pad,"
      "                     Tensor !adapter_enabled,"
      "                     Tensor !lora_ids,"
      "                     Tensor? maybe_expert_map) -> () ");
  m.impl("moe_lora_align_block_size", torch::kCUDA, &moe_lora_align_block_size);

#ifndef USE_ROCM
  m.def(
      "moe_wna16_gemm(Tensor input, Tensor! output, Tensor b_qweight, "
      "Tensor b_scales, Tensor? b_qzeros, "
      "Tensor? topk_weights, Tensor sorted_token_ids, "
      "Tensor expert_ids, Tensor num_tokens_post_pad, "
      "int top_k, int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, "
      "int bit) -> Tensor");

  m.impl("moe_wna16_gemm", torch::kCUDA, &moe_wna16_gemm);

  m.def(
      "moe_wna16_marlin_gemm(Tensor! a, Tensor? c_or_none,"
      "Tensor! b_q_weight, Tensor? b_bias_or_none,"
      "Tensor! b_scales, Tensor? a_scales, Tensor? global_scale, Tensor? "
      "b_zeros_or_none,"
      "Tensor? g_idx_or_none, Tensor? perm_or_none, Tensor! workspace,"
      "Tensor sorted_token_ids,"
      "Tensor! expert_ids, Tensor! num_tokens_past_padded,"
      "Tensor! topk_weights, int moe_block_size, int top_k, "
      "bool mul_topk_weights, int b_type_id,"
      "int size_m, int size_n, int size_k,"
      "bool is_full_k, bool use_atomic_add,"
      "bool use_fp32_reduce, bool is_zp_float,"
      "int thread_k, int thread_n, int blocks_per_sm) -> Tensor");

  m.def(
      "moe_permute(Tensor input, Tensor topk_ids,"
      "Tensor token_expert_indices, Tensor? expert_map, int n_expert,"
      "int n_local_expert,"
      "int topk, Tensor! permuted_input, Tensor! "
      "expert_first_token_offset, Tensor! inv_permuted_idx, Tensor! "
      "permuted_idx)->()");

  m.def(
      "moe_permute_with_scratch(Tensor input, Tensor topk_ids,"
      "Tensor token_expert_indices, Tensor? expert_map, int n_expert,"
      "int n_local_expert,"
      "int topk, Tensor! permuted_input, Tensor! "
      "expert_first_token_offset, Tensor! inv_permuted_idx, Tensor! "
      "permuted_idx, Tensor! sort_workspace, Tensor! permuted_experts_id, "
      "Tensor! sorted_row_idx, Tensor! topk_ids_for_sort)->()");

  m.def(
      "moe_unpermute(Tensor permuted_hidden_states, Tensor topk_weights,"
      "Tensor inv_permuted_idx, Tensor? expert_first_token_offset, "
      "int topk, Tensor! hidden_states)->()");

  m.def("moe_permute_unpermute_supported() -> bool");
  m.def(
      "moe_permute_sort_workspace_size(int num_expanded_rows, int n_expert) -> "
      "int");
  m.impl("moe_permute_unpermute_supported", &moe_permute_unpermute_supported);
  m.impl("moe_permute_sort_workspace_size", &moe_permute_sort_workspace_size);

  // Row shuffle for MoE
  m.def(
      "shuffle_rows(Tensor input_tensor, Tensor dst2src_map, Tensor! "
      "output_tensor) -> ()");
  m.impl("shuffle_rows", torch::kCUDA, &shuffle_rows);

  // Apply grouped topk routing to select experts.
  m.def(
      "grouped_topk(Tensor scores, int n_group, int "
      "topk_group, int topk, bool renormalize, float "
      "routed_scaling_factor, Tensor bias, int scoring_func) -> (Tensor, "
      "Tensor)");
  m.impl("grouped_topk", torch::kCUDA, &grouped_topk);

  // DeepSeek V3 optimized router GEMM for SM90+
  m.def("dsv3_router_gemm(Tensor! output, Tensor mat_a, Tensor mat_b) -> ()");
  // conditionally compiled so impl registration is in source file
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
