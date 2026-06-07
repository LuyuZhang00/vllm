# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# MLA（Multi-Latent Attention）注意力后端实现包
#
# 本包包含多种 MLA 注意力后端的具体实现，用于支持 DeepSeek 系列模型的高效推理。
# MLA 是一种创新的注意力机制，通过低秩分解（Low-Rank Decomposition）将 KV 缓存
# 压缩到低维潜空间（kv_lora_rank），从而大幅降低内存占用。
#
# 包含以下后端实现：
# 1. flashmla_sparse.py      - FlashMLA 稀疏注意力（NVIDIA GPU，支持 FP8 缓存）
# 2. flashinfer_mla_sparse.py - FlashInfer MLA 稀疏注意力（Blackwell GPU）
# 3. rocm_aiter_mla.py       - ROCm AITER MLA 注意力（AMD GPU）
# 4. rocm_aiter_mla_sparse.py - ROCm AITER MLA 稀疏注意力（AMD GPU）
# 5. flashattn_mla.py        - FlashAttention MLA 注意力（Hopper GPU）
# 6. triton_mla.py           - Triton MLA 注意力（通用 GPU，Triton 实现）
# 7. cutlass_mla.py          - CUTLASS MLA 注意力（Blackwell GPU）
# 8. tokenspeed_mla.py       - TokenSpeed MLA 注意力（Blackwell GPU，FP8 缓存专用）
# 9. xpu_mla_sparse.py       - XPU MLA 稀疏注意力（Intel XPU）
# 10. sparse_swa.py           - 稀疏滑动窗口注意力（DeepSeekV4 SWA 层）
# 11. indexer.py              - MLA 索引器（为稀疏注意力计算 topk 索引）
# 12. compressor_utils.py     - 压缩工具函数（KV 缓存压缩相关的 slot mapping）
# 13. sparse_utils.py         - 稀疏工具函数（索引转换等通用工具）
