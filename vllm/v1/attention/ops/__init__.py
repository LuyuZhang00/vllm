# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# 本包（ops）包含 vLLM v1 引擎中注意力机制的底层算子实现，
# 主要基于 Triton 编写高性能 GPU kernel。
#
# 核心模块概览：
# 1. triton_decode_attention.py  - 解码阶段（decode）的 Triton 注意力 kernel，
#    采用分块 KV 分割（split-KV）策略实现高效解码注意力计算。
# 2. triton_prefill_attention.py - 预填充阶段（prefill）的 Triton 注意力 kernel，
#    支持因果掩码和滑动窗口注意力。
# 3. triton_reshape_and_cache_flash.py - 将新的 KV 张量重塑并写入分页 KV 缓存，
#    支持 FP8 量化和不同内存布局（head-major / block-major）。
# 4. triton_attention_helpers.py - 统一注意力 kernel 的共享辅助函数，
#    包括在线 softmax、掩码构建、ALiBi 偏置等。
# 5. chunked_prefill_paged_decode.py - 分块预填充 + 分页解码的混合注意力，
#    主要用于 ROCm 平台的自定义 paged attention。
# 6. dcp_alltoall.py - 解码上下文并行（DCP）的 All-to-All 通信后端，
#    通过交换部分注意力输出和 LSE 值实现跨 rank 合并。
# 7. triton_merge_attn_states.py - 合并分裂 KV（split-KV）的局部注意力结果，
#    使用 LSE 加权方式确保数值稳定性。
# 8. vit_attn_wrappers.py - Vision Transformer 的注意力包装器，
#    提供 FlashAttention、Triton、PyTorch SDPA、FlashInfer 等多种后端。
