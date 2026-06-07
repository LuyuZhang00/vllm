# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
采样模块 (Sampling Module)

本包实现了 vLLM v1 引擎的采样功能，负责从模型输出的 logits 中选择 token。

核心组件：
1. sampler: 采样器主类，协调所有采样操作
2. states: 采样参数状态管理（温度、top-k、top-p、min-p、种子等）
3. gumbel: Gumbel-Max 采样算法实现
4. min_p: Min-P 采样过滤
5. penalties: 重复/频率/存在惩罚
6. bad_words: 禁用词过滤
7. logit_bias: Logit 偏置（允许的 token IDs、logit 偏置、最小 token 数）
8. logprob: Log 概率计算
9. prompt_logprob: Prompt log 概率计算
10. output: 采样器输出数据结构
"""
