# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
vLLM v1 注意力后端（attention backends）子包。

本子包包含所有具体注意力后端的实现。每个后端由三个核心组件构成：
1. XxxBackend 类：继承 AttentionBackend，声明后端支持的能力（数据类型、
   头大小、块大小、注意力类型等），并提供元数据构建器和实现类的工厂方法。
2. XxxMetadataBuilder 类：继承 AttentionMetadataBuilder，负责在每步推理前
   从通用元数据构建后端特定的注意力元数据（如块表、槽映射、调度信息等）。
3. XxxImpl 类：继承 AttentionImpl，实现具体的 forward 前向注意力计算。

当前支持的后端列表：
- flash_attn / flash_attn_diffkv：NVIDIA GPU 上的 FlashAttention（含 K/V 异构头大小变体）
- triton_attn：基于 Triton 自定义 kernel 的注意力
- cpu_attn：CPU 平台（x86/ARM/RISC-V 等架构）的注意力实现
- rocm_attn：AMD ROCm 平台（MI200/MI300 等 GPU）的注意力实现
- mamba_attn：Mamba 类 SSM 模型的公共基类（含 chunk 分块、prefix caching 等逻辑）
- mamba1_attn / mamba2_attn：Mamba1 和 Mamba2 状态空间模型的具体后端
- gdn_attn：GatedDeltaNet（门控差分网络）注意力后端
- linear_attn：线性注意力（Linear Attention）后端
- short_conv_attn：短卷积（Short Convolution）注意力后端
"""
