# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Registry for MLA prefill backends.

This module provides an enumeration of all available MLA prefill backends
and utilities for loading and registering them.
"""
# MLA prefill 后端注册表模块。
#
# 本模块提供了 MLA prefill 后端的注册和管理机制，包括：
# 1. MLAPrefillBackendEnum — 枚举所有内置的 MLA prefill 后端
# 2. register_mla_prefill_backend — 注册/覆盖后端的装饰器函数
# 3. 后端覆盖机制 — 允许用户替换内置后端的实现或添加自定义后端
#
# 后端选择流程：
# 用户可以通过 AttentionConfig 显式指定后端，也可以让系统根据
# 设备能力和模型配置自动选择最佳后端（参见 selector.py）。
#
# 内置后端：
# - FLASH_ATTN: 基于 FlashAttention 库（最广泛支持）
# - FLASHINFER: 基于 FlashInfer 库（仅 Blackwell GPU）
# - TRTLLM_RAGGED: 基于 TensorRT-LLM 的 ragged attention（仅 Blackwell）
# - TOKENSPEED_MLA: 基于 TokenSpeed CuTe DSL（仅 Blackwell）
# - CUSTOM: 占位符，用于注册第三方/自定义后端

from collections.abc import Callable
from enum import Enum, EnumMeta
from typing import TYPE_CHECKING

from vllm.utils.import_utils import resolve_obj_by_qualname

if TYPE_CHECKING:
    from vllm.v1.attention.backends.mla.prefill.base import MLAPrefillBackend


class _MLAPrefillBackendEnumMeta(EnumMeta):
    """Metaclass for MLAPrefillBackendEnum to provide better error messages.
    MLAPrefillBackendEnum 的元类，提供更友好的错误消息。

    当用户通过字符串名称访问不存在的后端时，会列出所有有效的后端名称，
    而不是抛出晦涩的 KeyError。
    """

    def __getitem__(cls, name: str):
        try:
            return super().__getitem__(name)
        except KeyError:
            members = cls.__members__.keys()
            valid_backends = ", ".join(members)
            raise ValueError(
                f"Unknown MLA prefill backend: '{name}'. "
                f"Valid options are: {valid_backends}"
            ) from None


class MLAPrefillBackendEnum(Enum, metaclass=_MLAPrefillBackendEnumMeta):
    """Enumeration of all supported MLA prefill backends.
    所有支持的 MLA prefill 后端的枚举类。

    每个枚举成员的值是对应后端实现类的完全限定路径字符串。
    系统通过 resolve_obj_by_qualname 延迟加载这些类。
    """

    # 1. FlashAttention 后端 — 最广泛支持的后端，支持 Hopper 和 Blackwell
    FLASH_ATTN = (
        "vllm.v1.attention.backends.mla.prefill.flash_attn.FlashAttnPrefillBackend"
    )
    # 2. FlashInfer 后端 — 仅支持 Blackwell (SM100) GPU
    FLASHINFER = (
        "vllm.v1.attention.backends.mla.prefill.flashinfer.FlashInferPrefillBackend"
    )
    # 3. TRT-LLM Ragged 后端 — 仅支持 Blackwell，使用 flashinfer 中的
    #    trtllm_ragged_attention_deepseek 内核
    TRTLLM_RAGGED = (
        "vllm.v1.attention.backends.mla.prefill.trtllm_ragged."
        "TrtllmRaggedPrefillBackend"
    )
    # 4. TokenSpeed MLA 后端 — 仅支持 Blackwell，使用 CuTe DSL 编写的内核
    TOKENSPEED_MLA = (
        "vllm.v1.attention.backends.mla.prefill.tokenspeed_mla."
        "TokenspeedMLAPrefillBackend"
    )
    # Placeholder for third-party/custom backends - must be registered before use
    # 5. 自定义后端占位符 — 使用前必须通过 register_mla_prefill_backend 注册
    #    设置为 None 以避免与空字符串值的其他后端产生别名冲突
    CUSTOM = None

    def get_path(self) -> str:
        """Get the class path for this backend (respects overrides).
        获取后端实现类的完全限定路径（考虑覆盖）。

        如果该后端被 register_mla_prefill_backend 覆盖过，
        则返回覆盖后的路径；否则返回枚举成员的默认路径。

        Returns:
            后端类的完全限定路径字符串

        Raises:
            ValueError: 如果 CUSTOM 后端未注册就被使用
        """
        path = _MLA_PREFILL_OVERRIDES.get(self, self.value)
        if not path:
            raise ValueError(
                f"MLA prefill backend {self.name} must be registered before "
                f"use. Use register_mla_prefill_backend("
                f"MLAPrefillBackendEnum.{self.name}, "
                f"'your.module.YourClass')"
            )
        return path

    def get_class(self) -> "type[MLAPrefillBackend]":
        """Get the backend class (respects overrides).
        获取后端实现类（考虑覆盖）。

        通过 resolve_obj_by_qualname 动态导入并返回后端类。

        Returns:
            后端实现类

        Raises:
            ImportError: 如果后端类无法导入
            ValueError: 如果 CUSTOM 后端未注册
        """
        return resolve_obj_by_qualname(self.get_path())

    def is_overridden(self) -> bool:
        """Check if this backend has been overridden.
        检查该后端是否已被覆盖。

        Returns:
            是否被覆盖
        """
        return self in _MLA_PREFILL_OVERRIDES

    def clear_override(self) -> None:
        """Clear any override for this backend, reverting to the default.
        清除该后端的覆盖，恢复为默认实现。
        """
        _MLA_PREFILL_OVERRIDES.pop(self, None)


# 全局覆盖字典：存储被 register_mla_prefill_backend 覆盖的后端路径。
# 键为枚举成员，值为覆盖后的类路径字符串。
_MLA_PREFILL_OVERRIDES: dict[MLAPrefillBackendEnum, str] = {}


def register_mla_prefill_backend(
    backend: MLAPrefillBackendEnum,
    class_path: str | None = None,
) -> Callable[[type], type]:
    """Register or override an MLA prefill backend implementation.
    注册或覆盖 MLA prefill 后端实现。

    此函数既可以用作装饰器（不传 class_path），也可以直接调用
    （传入 class_path）。支持以下用法：

    用法 1 — 装饰器方式覆盖已有后端：
        @register_mla_prefill_backend(MLAPrefillBackendEnum.FLASH_ATTN)
        class MyCustomFlashAttn(MLAPrefillBackend):
            ...

    用法 2 — 装饰器方式注册自定义后端：
        @register_mla_prefill_backend(MLAPrefillBackendEnum.CUSTOM)
        class MyCustomPrefillBackend(MLAPrefillBackend):
            ...

    用法 3 — 直接注册（不使用装饰器）：
        register_mla_prefill_backend(
            MLAPrefillBackendEnum.CUSTOM,
            "my.module.MyCustomPrefillBackend"
        )

    Args:
        backend: 要注册/覆盖的 MLAPrefillBackendEnum 成员
        class_path: 可选的类路径字符串。如果不提供且用作装饰器，
            将从类自动生成路径。

    Returns:
        如果 class_path 为 None，返回装饰器函数；否则返回无操作的 lambda。
    """

    def decorator(cls: type) -> type:
        _MLA_PREFILL_OVERRIDES[backend] = f"{cls.__module__}.{cls.__qualname__}"
        return cls

    if class_path is not None:
        _MLA_PREFILL_OVERRIDES[backend] = class_path
        return lambda x: x

    return decorator
