# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
vLLM v1 采样操作模块 (Sampling Operations Module)

本模块包含 vLLM v1 引擎中用于文本生成采样阶段的各类操作算子。
采样是语言模型推理的最后一步：模型输出 logits 向量后，需要通过
一系列采样策略从中选择最终的 token。

本模块提供的核心功能包括：
1. Top-K / Top-P 采样 (topk_topp_sampler.py)
   - Top-K: 仅保留概率最高的 K 个 token 进行采样
   - Top-P (核采样): 保留累积概率达到 P 的最小 token 集合进行采样
   - 支持多种后端: PyTorch 原生、FlashInfer、Triton GPU 内核、XPU 内核、ROCm aiter

2. Triton Top-K / Top-P 实现 (topk_topp_triton.py)
   - 基于论文 "Qrita" 的高性能 GPU 采样算法
   - 使用基于枢轴(pivot)的截断和选择策略
   - 避免对整个词表进行排序，显著提升性能

3. 禁用词过滤 (bad_words.py)
   - 根据已生成的 token 历史，屏蔽特定的禁用词序列
   - 防止模型生成不当或禁止的内容

4. 对数概率计算 (logprobs.py)
   - 计算采样后的 logprobs 统计信息
   - 用于 API 返回 top-k logprobs 等功能

5. 采样惩罚 (penalties.py)
   - 存在惩罚 (presence penalty): 对已出现过的 token 施加惩罚
   - 频率惩罚 (frequency penalty): 根据 token 出现次数施加递增惩罚
   - 重复惩罚 (repetition penalty): 对重复 token 施加乘性惩罚
"""
