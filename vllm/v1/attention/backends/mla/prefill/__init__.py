# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# MLA (Multi-head Latent Attention) prefill 后端包的初始化文件。
#
# 本模块是 MLA prefill 注意力后端的统一入口，负责导出所有公共接口。
# MLA 是 DeepSeek 系列模型使用的高效注意力机制，通过低秩联合压缩
# KV 缓存来减少显存占用。
#
# 包含的公共接口如下：
# 1. MLAPrefillBackend      — 所有 MLA prefill 后端的抽象基类
# 2. MLAPrefillBackendEnum  — 所有可用后端的枚举（FlashAttn、FlashInfer 等）
# 3. get_mla_prefill_backend — 根据设备能力和配置自动选择最佳后端
# 4. register_mla_prefill_backend — 注册/覆盖自定义 MLA prefill 后端的装饰器
from vllm.v1.attention.backends.mla.prefill.base import MLAPrefillBackend
from vllm.v1.attention.backends.mla.prefill.registry import (
    MLAPrefillBackendEnum,
    register_mla_prefill_backend,
)
from vllm.v1.attention.backends.mla.prefill.selector import get_mla_prefill_backend

__all__ = [
    "MLAPrefillBackend",
    "MLAPrefillBackendEnum",
    "get_mla_prefill_backend",
    "register_mla_prefill_backend",
]
