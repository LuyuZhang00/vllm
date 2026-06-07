# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
vLLM v1 注意力机制包（attention package）。

本包是 vLLM v1 引擎中注意力计算的核心模块，提供统一的注意力后端抽象接口，
以及多种注意力后端的具体实现（FlashAttention、Triton、CPU、ROCm 等）。

整体架构说明：
1. AttentionBackend（注意力后端）：定义后端的能力查询接口（支持的数据类型、
   头大小、CUDA Graph 支持等），是所有后端实现的基类。
2. AttentionMetadataBuilder（元数据构建器）：每一步推理前，根据调度器输出的
   通用元数据（CommonAttentionMetadata）构建特定后端所需的注意力元数据。
3. AttentionImpl（注意力实现）：包含具体的 forward 前向计算逻辑，执行
   query、key、value 之间的注意力运算。

请求流程：
  调度器 Scheduler -> CommonAttentionMetadata
    -> AttentionMetadataBuilder.build() -> 后端特定的 Metadata
      -> AttentionImpl.forward() -> 输出张量

子包 backends/ 包含各硬件平台和算法的具体实现：
- flash_attn.py：标准 FlashAttention 后端（NVIDIA GPU）
- flash_attn_diffkv.py：支持 K/V 不同头大小的 FlashAttention 变体
- triton_attn.py：基于 Triton 的注意力后端
- cpu_attn.py：CPU 平台注意力后端
- rocm_attn.py：AMD ROCm 平台注意力后端
- mamba_attn.py：Mamba 类 SSM 模型的公共注意力基类
- mamba1_attn.py / mamba2_attn.py：Mamba1 和 Mamba2 的具体实现
- gdn_attn.py：GatedDeltaNet 注意力后端
- linear_attn.py：线性注意力后端
- short_conv_attn.py：短卷积注意力后端
"""
