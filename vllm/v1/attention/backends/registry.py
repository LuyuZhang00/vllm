# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention backend registry"""
#
# 整体说明：Attention 后端注册中心
# ===========================================
# 本文件实现了 vLLM V1 中所有 attention 后端的注册和管理机制。
# 核心设计思路：
#   1. 使用 Enum 枚举类（AttentionBackendEnum / MambaAttentionBackendEnum）
#      来列出所有已知的 attention 后端，每个枚举成员的值是该后端类的全限定路径。
#   2. 支持运行时覆盖（override）：通过 register_backend() 函数，用户可以替换
#      任意后端的实现，或者注册自定义的第三方后端（CUSTOM 类型）。
#   3. 覆盖信息存储在模块级字典 _ATTN_OVERRIDES 和 _MAMBA_ATTN_OVERRIDES 中，
#      当调用 get_class() 或 get_path() 时优先使用覆盖值。
#   4. 使用自定义元类 _AttentionBackendEnumMeta 来提供更友好的错误信息
#      （当用户拼错后端名称时，会列出所有有效选项）。
#
# 使用场景：
#   - selector.py 中的 _cached_get_attn_backend() 最终会调用此处的 get_class()
#     来获取实际的后端类。
#   - 第三方插件可以通过 register_backend() 装饰器将自己的实现注册为某个后端。
#   - CUSTOM 枚举成员是专门留给第三方/自定义后端的占位符。

from collections.abc import Callable
from enum import Enum, EnumMeta
from typing import TYPE_CHECKING, cast

from vllm.logger import init_logger
from vllm.utils.import_utils import resolve_obj_by_qualname

if TYPE_CHECKING:
    from vllm.v1.attention.backend import AttentionBackend

logger = init_logger(__name__)


# 中文注释：自定义枚举元类，用于改善错误提示信息。
# 当用户通过字符串名称查找后端（如 AttentionBackendEnum["FLASH_ATTN"]）时，
# 如果名称拼写错误，标准 Enum 只会抛出 KeyError，不够友好。
# 这个元类重写了 __getitem__，在 KeyError 时列出所有可用的后端名称，
# 帮助用户快速定位正确的后端名称。
class _AttentionBackendEnumMeta(EnumMeta):
    """Metaclass for AttentionBackendEnum to provide better error messages."""

    def __getitem__(cls, name: str):
        """Get backend by name with helpful error messages."""
        try:
            return super().__getitem__(name)
        except KeyError:
            members = cast("dict[str, Enum]", cls.__members__).keys()
            valid_backends = ", ".join(members)
            raise ValueError(
                f"Unknown attention backend: '{name}'. "
                f"Valid options are: {valid_backends}"
            ) from None


# 中文注释：标准 Attention 后端枚举类。
# 枚举成员涵盖了 vLLM V1 支持的所有 attention 后端实现，大致分为以下几类：
#   - FlashAttention 系列：FLASH_ATTN（通用）、FLASH_ATTN_DIFFKV（K/V头维度不同）、
#     ROCM_AITER_FA（ROCm 平台 AITER 优化版本）
#   - FlashInfer 系列：FLASHINFER（通用）、FLASHINFER_MLA（MLA 版本）
#   - Triton 系列：TRITON_ATTN（通用）、TRITON_MLA（MLA 版本）
#   - ROCm 专用：ROCM_ATTN、ROCM_AITER_MLA 等
#   - MLA（Multi-head Latent Attention）系列：用于 DeepSeek 等模型的压缩 KV attention
#   - 特殊类型：NO_ATTENTION（不需要 attention 的层）、FLEX_ATTENTION（灵活 attention）、
#     TORCH_SDPA（仅用于 ViT）、CPU_ATTN（CPU 上的 attention）
#   - CUSTOM：第三方/自定义后端的占位符，使用前必须通过 register_backend() 注册。
#
# 每个枚举成员的值是该后端类的全限定路径字符串（模块路径.类名），
# 但这个值可以被 register_backend() 运行时覆盖。
# 要获取实际的后端类（考虑覆盖），应使用 get_class() 方法。
class AttentionBackendEnum(Enum, metaclass=_AttentionBackendEnumMeta):
    """Enumeration of all supported attention backends.

    The enum value is the default class path, but this can be overridden
    at runtime using register_backend().

    To get the actual backend class (respecting overrides), use:
        backend.get_class()
    """

    FLASH_ATTN = "vllm.v1.attention.backends.flash_attn.FlashAttentionBackend"
    FLASH_ATTN_DIFFKV = (
        "vllm.v1.attention.backends.flash_attn_diffkv.FlashAttentionDiffKVBackend"
    )
    TRITON_ATTN = "vllm.v1.attention.backends.triton_attn.TritonAttentionBackend"
    ROCM_ATTN = "vllm.v1.attention.backends.rocm_attn.RocmAttentionBackend"
    ROCM_AITER_MLA = "vllm.v1.attention.backends.mla.rocm_aiter_mla.AiterMLABackend"
    ROCM_AITER_TRITON_MLA = (
        "vllm.v1.attention.backends.mla.aiter_triton_mla.AiterTritonMLABackend"
    )
    ROCM_AITER_FA = (
        "vllm.v1.attention.backends.rocm_aiter_fa.AiterFlashAttentionBackend"
    )
    ROCM_AITER_MLA_SPARSE = (
        "vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse.ROCMAiterMLASparseBackend"
    )
    XPU_MLA_SPARSE = "vllm.v1.attention.backends.mla.xpu_mla_sparse.XPUMLASparseBackend"
    TORCH_SDPA = ""  # this tag is only used for ViT
    FLASHINFER = "vllm.v1.attention.backends.flashinfer.FlashInferBackend"
    FLASHINFER_MLA = (
        "vllm.v1.attention.backends.mla.flashinfer_mla.FlashInferMLABackend"
    )
    TOKENSPEED_MLA = (
        "vllm.v1.attention.backends.mla.tokenspeed_mla.TokenspeedMLABackend"
    )
    FLASHINFER_MLA_SPARSE = (
        "vllm.v1.attention.backends.mla.flashinfer_mla_sparse."
        "FlashInferMLASparseBackend"
    )
    TRITON_MLA = "vllm.v1.attention.backends.mla.triton_mla.TritonMLABackend"
    CUTLASS_MLA = "vllm.v1.attention.backends.mla.cutlass_mla.CutlassMLABackend"
    FLASHMLA = "vllm.v1.attention.backends.mla.flashmla.FlashMLABackend"
    FLASHMLA_SPARSE = (
        "vllm.v1.attention.backends.mla.flashmla_sparse.FlashMLASparseBackend"
    )
    FLASH_ATTN_MLA = "vllm.v1.attention.backends.mla.flashattn_mla.FlashAttnMLABackend"
    NO_ATTENTION = "vllm.v1.attention.backends.no_attention.NoAttentionBackend"
    FLEX_ATTENTION = "vllm.v1.attention.backends.flex_attention.FlexAttentionBackend"
    ROCM_AITER_UNIFIED_ATTN = (
        "vllm.v1.attention.backends.rocm_aiter_unified_attn."
        "RocmAiterUnifiedAttentionBackend"
    )
    CPU_ATTN = "vllm.v1.attention.backends.cpu_attn.CPUAttentionBackend"
    TURBOQUANT = "vllm.v1.attention.backends.turboquant_attn.TurboQuantAttentionBackend"
    # Placeholder for third-party/custom backends - must be registered before use
    # set to None to avoid alias with other backend, whose value is an empty string
    CUSTOM = None

    # 中文注释：获取后端类的全限定路径字符串。
    # 如果该后端有运行时覆盖（通过 register_backend 注册），则返回覆盖路径；
    # 否则返回枚举成员的默认值（即类定义时写死的路径）。
    # 如果路径为空（如 CUSTOM 未注册、或 TORCH_SDPA），则抛出 ValueError。
    # include_classname 参数控制是否包含类名本身：
    #   - True: 返回 "vllm.v1.attention.backends.flash_attn.FlashAttentionBackend"
    #   - False: 返回 "vllm.v1.attention.backends.flash_attn"（仅模块路径）
    def get_path(self, include_classname: bool = True) -> str:
        """Get the class path for this backend (respects overrides).

        Returns:
            The fully qualified class path string

        Raises:
            ValueError: If Backend.CUSTOM is used without being registered
        """
        path = _ATTN_OVERRIDES.get(self, self.value)
        if not path:
            raise ValueError(
                f"Backend {self.name} must be registered before use. "
                f"Use register_backend(Backend.{self.name}, 'your.module.YourClass')"
            )
        if not include_classname:
            path = path.rsplit(".", 1)[0]
        return path

    # 中文注释：获取后端类的实际 Python class 对象。
    # 先通过 get_path() 获取全限定路径，再通过 resolve_obj_by_qualname()
    # 动态导入并返回该类。这是实际获取后端类的最终方法。
    def get_class(self) -> "type[AttentionBackend]":
        """Get the backend class (respects overrides).

        Returns:
            The backend class

        Raises:
            ImportError: If the backend class cannot be imported
            ValueError: If Backend.CUSTOM is used without being registered
        """
        return resolve_obj_by_qualname(self.get_path())

    # 中文注释：检查该后端是否已被运行时覆盖。
    def is_overridden(self) -> bool:
        """Check if this backend has been overridden.

        Returns:
            True if the backend has a registered override
        """
        return self in _ATTN_OVERRIDES

    # 中文注释：清除该后端的运行时覆盖，恢复为枚举定义时的默认值。
    def clear_override(self) -> None:
        """Clear any override for this backend, reverting to the default."""
        _ATTN_OVERRIDES.pop(self, None)


# 中文注释：Mamba 类注意力后端的枚举类。
# Mamba 架构（包括其变体）不属于标准 Transformer attention，但 vLLM 将其统一纳入
# attention 后端框架管理。包含以下类型：
#   - MAMBA1: Mamba 第一代 SSM 后端
#   - MAMBA2: Mamba 第二代（如 Mamba-2 / Jamba）
#   - SHORT_CONV: 短卷积注意力后端
#   - LINEAR: 线性注意力后端
#   - GDN_ATTN: GDN（Gated Delta Network）注意力后端
#   - CUSTOM: 自定义后端占位符
# 与 AttentionBackendEnum 类似，每个成员的值可以被 register_backend() 覆盖。
class MambaAttentionBackendEnum(Enum, metaclass=_AttentionBackendEnumMeta):
    """Enumeration of all supported mamba attention backends.

    The enum value is the default class path, but this can be overridden
    at runtime using register_backend().

    To get the actual backend class (respecting overrides), use:
        backend.get_class()
    """

    MAMBA1 = "vllm.v1.attention.backends.mamba1_attn.Mamba1AttentionBackend"
    MAMBA2 = "vllm.v1.attention.backends.mamba2_attn.Mamba2AttentionBackend"
    SHORT_CONV = "vllm.v1.attention.backends.short_conv_attn.ShortConvAttentionBackend"
    LINEAR = "vllm.v1.attention.backends.linear_attn.LinearAttentionBackend"
    GDN_ATTN = "vllm.v1.attention.backends.gdn_attn.GDNAttentionBackend"
    # Placeholder for third-party/custom backends - must be registered before use
    # set to None to avoid alias with other backend, whose value is an empty string
    CUSTOM = None

    # 中文注释：获取 Mamba 后端类的全限定路径字符串。
    # 逻辑与 AttentionBackendEnum.get_path() 完全对称，但查询的是 _MAMBA_ATTN_OVERRIDES 字典。
    def get_path(self, include_classname: bool = True) -> str:
        """Get the class path for this backend (respects overrides).

        Returns:
            The fully qualified class path string

        Raises:
            ValueError: If Backend.CUSTOM is used without being registered
        """
        path = _MAMBA_ATTN_OVERRIDES.get(self, self.value)
        if not path:
            raise ValueError(
                f"Backend {self.name} must be registered before use. "
                f"Use register_backend(Backend.{self.name}, 'your.module.YourClass')"
            )
        if not include_classname:
            path = path.rsplit(".", 1)[0]
        return path

    # 中文注释：获取 Mamba 后端类的实际 Python class 对象。
    def get_class(self) -> "type[AttentionBackend]":
        """Get the backend class (respects overrides).

        Returns:
            The backend class

        Raises:
            ImportError: If the backend class cannot be imported
            ValueError: If Backend.CUSTOM is used without being registered
        """
        return resolve_obj_by_qualname(self.get_path())

    # 中文注释：检查该 Mamba 后端是否已被运行时覆盖。
    def is_overridden(self) -> bool:
        """Check if this backend has been overridden.

        Returns:
            True if the backend has a registered override
        """
        return self in _MAMBA_ATTN_OVERRIDES

    # 中文注释：清除该 Mamba 后端的运行时覆盖，恢复默认。
    def clear_override(self) -> None:
        """Clear any override for this backend, reverting to the default."""
        _MAMBA_ATTN_OVERRIDES.pop(self, None)


# 中文注释：存储 attention 后端运行时覆盖映射的全局字典。
# 键是枚举成员（如 AttentionBackendEnum.FLASH_ATTN），值是覆盖后的类路径字符串。
# 当调用 get_path() 或 get_class() 时，优先从这个字典中查找覆盖值。
_ATTN_OVERRIDES: dict[AttentionBackendEnum, str] = {}
# 中文注释：存储 Mamba 后端运行时覆盖映射的全局字典。
_MAMBA_ATTN_OVERRIDES: dict[MambaAttentionBackendEnum, str] = {}


# 中文注释：注册或覆盖一个 attention 后端实现的核心函数。
# 这个函数有两种使用方式：
#   方式一：作为装饰器使用（不传 class_path 参数）
#     @register_backend(AttentionBackendEnum.FLASH_ATTN)
#     class MyCustomFlashAttn: ...
#     这时装饰器会自动从被装饰类的 __module__ 和 __qualname__ 生成路径。
#
#   方式二：直接调用注册（传入 class_path 字符串）
#     register_backend(AttentionBackendEnum.CUSTOM, "my.module.MyCustomBackend")
#     这时直接使用提供的路径字符串。
#
# 参数说明：
#   - backend: 要注册/覆盖的后端枚举成员
#   - class_path: 可选的类路径字符串，不传时由装饰器自动推导
#   - is_mamba: 是否是 Mamba 后端（决定写入哪个字典）
#
# 典型使用场景：
#   - 第三方插件将自己的 attention 实现注册为 CUSTOM 后端
#   - 覆盖已有的 FLASH_ATTN 等后端以替换为自定义实现
def register_backend(
    backend: AttentionBackendEnum | MambaAttentionBackendEnum,
    class_path: str | None = None,
    is_mamba: bool = False,
) -> Callable[[type], type]:
    """Register or override a backend implementation.

    Args:
        backend: The AttentionBackendEnum member to register
        class_path: Optional class path. If not provided and used as
            decorator, will be auto-generated from the class.

    Returns:
        Decorator function if class_path is None, otherwise a no-op

    Examples:
        # Override an existing attention backend
        @register_backend(AttentionBackendEnum.FLASH_ATTN)
        class MyCustomFlashAttn:
            ...

        # Override an existing mamba attention backend
        @register_backend(MambaAttentionBackendEnum.LINEAR, is_mamba=True)
        class MyCustomMambaAttn:
            ...

        # Register a custom third-party attention backend
        @register_backend(AttentionBackendEnum.CUSTOM)
        class MyCustomBackend:
            ...

        # Direct registration
        register_backend(
            AttentionBackendEnum.CUSTOM,
            "my.module.MyCustomBackend"
        )
    """

    # 中文注释：装饰器内部函数，将类的全限定路径写入对应的覆盖字典。
    def decorator(cls: type) -> type:
        if is_mamba:
            _MAMBA_ATTN_OVERRIDES[backend] = f"{cls.__module__}.{cls.__qualname__}"  # type: ignore[index]
        else:
            _ATTN_OVERRIDES[backend] = f"{cls.__module__}.{cls.__qualname__}"  # type: ignore[index]
        return cls

    # 中文注释：如果传入了 class_path，说明是直接注册模式（非装饰器模式），
    # 直接将路径写入覆盖字典，返回一个不做任何修改的 identity 装饰器。
    if class_path is not None:
        if is_mamba:
            _MAMBA_ATTN_OVERRIDES[backend] = class_path  # type: ignore[index]
        else:
            _ATTN_OVERRIDES[backend] = class_path  # type: ignore[index]
        return lambda x: x

    return decorator
