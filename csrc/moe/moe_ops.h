// =============================================================================
// 中文注释：MoE CUDA 算子头文件
//
// 本头文件声明了 vLLM MoE 模块中所有 CUDA kernel 的 C++ 函数签名。
// 这些函数在对应的 .cu 文件中实现，并通过 torch_bindings.cpp 注册为 PyTorch 自定义算子。
//
// 主要算子分类：
// 1. 路由(Routing)算子：topk_softmax, topk_sigmoid, topk_softplus_sqrt
//    —— 对模型输出的 gating logits 做激活函数并选择 top-k 专家。
// 2. Token 分组对齐算子：moe_align_block_size, batched_moe_align_block_size,
//    moe_lora_align_block_size
//    —— 将 token 按专家分组并对齐到 block 边界，供分块 GEMM kernel 使用。
// 3. 量化 GEMM 算子：moe_wna16_gemm
//    —— WNA16 (Weight-Only N-bit with 16-bit activation) 量化 MoE 矩阵乘法。
// 4. 排列/反排列算子：moe_permute, moe_unpermute, shuffle_rows
//    —— 将 token 按专家排序以便批量计算，计算完成后还原原始顺序。
// 5. 分组路由算子：grouped_topk
//    —— DeepSeek V3 等模型使用的分组 top-k 路由算法。
// 6. 专用 Router GEMM：dsv3_router_gemm
//    —— DeepSeek V3 专用的 router 矩阵乘法优化算子。
// 7. 求和算子：moe_sum
//    —— 将多个专家的输出按权重加权求和。
// =============================================================================

#pragma once

#include <torch/all.h>

void topk_softmax(torch::Tensor& topk_weights, torch::Tensor& topk_indices,
                  torch::Tensor& token_expert_indices,
                  torch::Tensor& gating_output, bool renormalize,
                  std::optional<torch::Tensor> bias);

void topk_sigmoid(torch::Tensor& topk_weights, torch::Tensor& topk_indices,
                  torch::Tensor& token_expert_indices,
                  torch::Tensor& gating_output, bool renormalize,
                  std::optional<torch::Tensor> bias);

void topk_softplus_sqrt(torch::Tensor& topk_weights,
                        torch::Tensor& topk_indices,
                        torch::Tensor& token_expert_indices,
                        torch::Tensor& gating_output, bool renormalize,
                        double routed_scaling_factor,
                        const c10::optional<torch::Tensor>& correction_bias,
                        const c10::optional<torch::Tensor>& input_ids,
                        const c10::optional<torch::Tensor>& tid2eid);

void moe_sum(torch::Tensor& input, torch::Tensor& output);

void moe_align_block_size(torch::Tensor topk_ids, int64_t num_experts,
                          int64_t block_size, torch::Tensor sorted_token_ids,
                          torch::Tensor experts_ids,
                          torch::Tensor num_tokens_post_pad,
                          std::optional<torch::Tensor> maybe_expert_map);

void batched_moe_align_block_size(int64_t max_tokens_per_batch,
                                  int64_t block_size,
                                  torch::Tensor const& expert_num_tokens,
                                  torch::Tensor sorted_ids,
                                  torch::Tensor expert_ids,
                                  torch::Tensor num_tokens_post_pad);

void moe_lora_align_block_size(
    torch::Tensor topk_ids, torch::Tensor token_lora_mapping,
    int64_t num_experts, int64_t block_size, int64_t max_loras,
    int64_t max_num_tokens_padded, int64_t max_num_m_blocks,
    torch::Tensor sorted_token_ids, torch::Tensor expert_ids,
    torch::Tensor num_tokens_post_pad, torch::Tensor adapter_enabled,
    torch::Tensor lora_ids, std::optional<torch::Tensor> maybe_expert_map);
#ifndef USE_ROCM
torch::Tensor moe_wna16_gemm(torch::Tensor input, torch::Tensor output,
                             torch::Tensor b_qweight, torch::Tensor b_scales,
                             std::optional<torch::Tensor> b_qzeros,
                             std::optional<torch::Tensor> topk_weights,
                             torch::Tensor sorted_token_ids,
                             torch::Tensor expert_ids,
                             torch::Tensor num_tokens_post_pad, int64_t top_k,
                             int64_t BLOCK_SIZE_M, int64_t BLOCK_SIZE_N,
                             int64_t BLOCK_SIZE_K, int64_t bit);

std::tuple<torch::Tensor, torch::Tensor> grouped_topk(
    torch::Tensor const& scores, int64_t n_group, int64_t topk_group,
    int64_t topk, bool renormalize, double routed_scaling_factor,
    torch::Tensor const& bias, int64_t scoring_func);
#endif

bool moe_permute_unpermute_supported();

int64_t moe_permute_sort_workspace_size(int64_t num_expanded_rows,
                                        int64_t num_experts);

void shuffle_rows(const torch::Tensor& input_tensor,
                  const torch::Tensor& dst2src_map,
                  torch::Tensor& output_tensor);

#ifndef USE_ROCM
// DeepSeek V3 optimized router GEMM kernel for SM90+
// Computes output = mat_a @ mat_b.T where:
//   mat_a: [num_tokens, hidden_dim] in bf16
//   mat_b: [num_experts, hidden_dim] in bf16
//   output: [num_tokens, num_experts] in bf16 or fp32
// Supports num_tokens in [1, 16], num_experts in {256, 384}, hidden_dim = 7168
void dsv3_router_gemm(torch::Tensor& output, const torch::Tensor& mat_a,
                      const torch::Tensor& mat_b);
#endif
