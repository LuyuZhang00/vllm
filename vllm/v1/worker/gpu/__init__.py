# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# vLLM v1 GPU Worker 子包
# =========================
# 本包包含 vLLM v1 引擎中 GPU Worker 的各种辅助模块，提供以下功能：
#
# 1. lora_utils.py   —— LoRA（低秩适配）状态管理，用于跟踪每个请求绑定的 LoRA 适配器，
#                      并生成模型前向传播所需的 LoRA 映射输入。
# 2. pp_utils.py     —— 流水线并行（Pipeline Parallelism）工具，负责在 PP 组的最后一个
#                      rank 与前面 rank 之间广播采样结果。
# 3. shutdown.py     —— 关闭清理工具，在 Worker 关闭前释放全局资源（RoPE 字典、
#                      编译上下文、工作区管理器等），防止内存泄漏。
# 4. states.py       —— 请求状态管理，维护每个请求的 token ID、长度、已计算 token 数、
#                      采样结果等核心运行时状态。
# 5. structured_outputs.py —— 结构化输出工具，使用 Triton 内核将语法规则掩码（grammar
#                      bitmask）应用到 logits 上，实现 JSON schema 等结构化生成约束。
# 6. warmup.py       —— 内核预热，通过模拟 prefill 和 decode 迭代触发 Triton JIT 编译，
#                      避免首次推理时的编译延迟。
