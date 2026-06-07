# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""采样包：从模型输出中采样下一个token。

本包包含以下主要模块:
1. sampler.py - 主采样器，负责从logits中采样下一个token
2. rejection_sampler.py - 拒绝采样器，用于投机解码(speculative decoding)
3. thinking_budget_state.py - 思考预算状态管理，用于控制推理模型的思考token数量
4. metadata.py - 采样元数据，存储采样所需的参数和配置
5. logits_processor/ - logits处理器子包，包含内置处理器和自定义处理器接口
"""
