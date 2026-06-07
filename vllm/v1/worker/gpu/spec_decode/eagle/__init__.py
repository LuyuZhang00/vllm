# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
EAGLE 投机解码模块。

EAGLE（Extrapolation Algorithm for Greater Language-model Efficiency）是一种高效的
投机解码方法。其核心思想是：

1. 在目标模型的基础上训练一个轻量级的"草稿头"（draft head），该草稿头利用目标模型
   的隐状态（hidden states）进行自回归式的 token 预测。
2. 与传统的独立草稿模型不同，EAGLE 草稿头直接复用目标模型的 KV cache 和注意力机制，
   从而大大减少了额外的计算开销。
3. 通过多步解码（multi-step decode），EAGLE 可以一次性生成多个候选 token，
   进一步提升吞吐量。

本子模块包含以下组件：
- speculator.py: EAGLE 投机解码器的主要实现
- cudagraph.py: EAGLE 专用的 CUDA 图管理器，用于优化执行性能
- eagle3_utils.py: EAGLE3 变体的辅助工具函数
- utils.py: 通用的 EAGLE 工具函数，包括模型加载和权重共享
"""
