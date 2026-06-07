# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# 整体说明：FlashAttention 工具函数与平台适配层
# ===========================================
# 本文件负责：
#   1. 根据当前运行平台（CUDA / ROCm / XPU），导入对应的 flash_attn_varlen_func
#      和 reshape_and_cache_flash 等底层算子。这些算子是 attention 计算的核心基础设施。
#   2. 提供 get_flash_attn_version() 函数，根据 GPU 架构（SM 版本）、模型配置
#      （head_size、是否使用 ALiBi 等）自动选择 FlashAttention 的版本（FA2 / FA3 / FA4）。
#   3. 提供若干能力查询函数（如 flash_attn_supports_sinks()、flash_attn_supports_mla()），
#       供上层 attention 后端判断当前环境是否支持特定特性。
#
# 平台适配策略：
#   - CUDA: 使用 vllm 自己编译的 vllm_flash_attn 包（包含 FA2/FA3/FA4）。
#   - XPU (Intel GPU): 使用 xpu_ops 中的封装。
#   - ROCm (AMD GPU): 优先使用上游 flash-attn 包，如果未安装则提供抛出异常的桩函数。
#
# FlashAttention 版本选择逻辑（重要）：
#   FA2: 最基础版本，兼容性最广，所有 NVIDIA GPU 都支持。
#   FA3: 专为 Hopper (SM90, H100) 架构优化，支持 scheduler metadata。
#   FA4: 专为 Blackwell (SM100+, B200) 架构优化，支持更多特性。
#   选择时会考虑：GPU 架构 -> 用户配置覆盖 -> 不兼容特性回退 -> 环境变量约束。

from typing import Any

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

# 中文注释：ROCm 平台的 flash-attn 可用性标志。
# 在模块初始化时设置，之后不再修改。避免重复的 import 尝试。
_ROCM_FLASH_ATTN_AVAILABLE = False

# 中文注释：根据当前平台导入底层算子，包括三个核心函数：
#   - flash_attn_varlen_func: 变长 FlashAttention 计算（支持不同序列长度的 batch）
#   - reshape_and_cache_flash: 将新计算的 K/V 写入 KV cache（reshape 并缓存）
#   - get_scheduler_metadata: FlashAttention 3 引入的 scheduler 元数据（用于优化调度）
#
# 各平台来源不同：
#   CUDA: vllm 自己编译的 vllm_flash_attn 包（含 FA2/FA3/FA4 支持）
#   XPU: Intel 提供的 xpu_ops 封装
#   ROCm: 上游 flash-attn 包（如果可用），否则使用桩函数
if current_platform.is_cuda():
    from vllm._custom_ops import reshape_and_cache_flash
    from vllm.vllm_flash_attn import (  # type: ignore[attr-defined]
        flash_attn_varlen_func,
        get_scheduler_metadata,
    )

elif current_platform.is_xpu():
    from vllm import _custom_ops as ops
    from vllm._xpu_ops import xpu_ops

    reshape_and_cache_flash = ops.reshape_and_cache_flash
    flash_attn_varlen_func = xpu_ops.flash_attn_varlen_func  # type: ignore[assignment]
    get_scheduler_metadata = xpu_ops.get_scheduler_metadata  # type: ignore[assignment]
elif current_platform.is_rocm():
    try:
        from flash_attn import flash_attn_varlen_func  # type: ignore[no-redef]

        # Mark that upstream flash-attn is available on ROCm
        _ROCM_FLASH_ATTN_AVAILABLE = True
    except ImportError:

        # 中文注释：ROCm 平台未安装上游 flash-attn 时，提供桩函数。
        # 调用时会直接抛出 ImportError，提示用户安装依赖。
        def flash_attn_varlen_func(*args: Any, **kwargs: Any) -> Any:  # type: ignore[no-redef,misc]
            raise ImportError(
                "ROCm platform requires upstream flash-attn "
                "to be installed. Please install flash-attn first."
            )

    # ROCm doesn't use scheduler metadata (FA3 feature), provide stub
    # 中文注释：ROCm 不使用 FA3 的 scheduler metadata 特性，提供空桩函数。
    def get_scheduler_metadata(*args: Any, **kwargs: Any) -> None:  # type: ignore[misc]
        return None

    # ROCm uses the C++ custom op for reshape_and_cache
    from vllm import _custom_ops as ops

    reshape_and_cache_flash = ops.reshape_and_cache_flash


# 中文注释：选择 FlashAttention 版本的核心函数。
# 这是 vLLM 中一个关键的决策函数，决定每个 attention 层使用哪个版本的 FlashAttention。
#
# 返回值：2、3、4 表示 FA 版本号，None 表示该平台不使用 vllm_flash_attn（如 ROCm）。
#
# 版本选择的整体流程（按优先级从高到低）：
#   第一步：根据 GPU 架构初步选择默认版本
#     - XPU: 固定返回 2（Intel GPU 只支持 FA2）
#     - ROCm: 返回 None（ROCm 使用上游 flash-attn，不走 vllm_flash_attn 体系）
#     - SM90 (Hopper/H100): 优先 FA3
#     - SM100 (Blackwell/B200): 优先 FA4
#     - 其他: 回退到 FA2
#
#   第二步：检查用户显式配置覆盖
#     - 如果 vllm_config.attention_config.flash_attn_version 有值，使用用户的配置
#
#   第三步：不兼容特性回退
#     - Blackwell 上不允许 FA3 -> 回退到 FA4 或 FA2
#     - ALiBi 位置编码不兼容 FA3/FA4 -> 回退到 FA2
#     - FA3 在 SM90 上不支持某些 head_size（>256）或 diff-KV with sinks -> 升级到 FA4
#     - 批处理不变性模式（VLLM_BATCH_INVARIANT）不兼容 FA4 -> 回退到 FA2
#     - FA4 在 Blackwell 上有 TMEM 容量限制，head_size > 128 时回退（head_size=192 是 MLA 例外）
#
#   第四步：验证最终选择的版本是否可用
def get_flash_attn_version(
    requires_alibi: bool = False,
    head_size: int | None = None,
    head_size_v: int | None = None,
    has_sinks: bool = False,
) -> int | None:
    if current_platform.is_xpu():
        return 2
    if current_platform.is_rocm():
        # ROCm doesn't use vllm_flash_attn; return None to skip fa_version arg
        return None
    try:
        from vllm.vllm_flash_attn.flash_attn_interface import (
            fa_version_unsupported_reason,
            is_fa_version_supported,
        )

        device_capability = current_platform.get_device_capability()

        assert device_capability is not None

        # 1. default version depending on platform
        # 中文注释：第一步 - 根据 GPU 计算能力（SM 版本）选择默认的 FA 版本。
        if device_capability.major == 9 and is_fa_version_supported(3):
            # Hopper (SM90): prefer FA3
            fa_version = 3
        elif device_capability.major == 10 and is_fa_version_supported(4):
            # Blackwell (SM100+, restrict to SM100 for now): prefer FA4
            fa_version = 4
        else:
            # Fallback to FA2
            fa_version = 2

        # 2. override if passed by environment or config
        # 中文注释：第二步 - 检查用户是否在配置中显式指定了 FA 版本。
        from vllm.config import get_current_vllm_config_or_none

        vllm_config = get_current_vllm_config_or_none()
        if (
            vllm_config is not None
            and vllm_config.attention_config.flash_attn_version is not None
        ):
            fa_version = vllm_config.attention_config.flash_attn_version

        # 3. fallback for unsupported combinations
        # 中文注释：第三步 - 处理不兼容的特性组合，进行版本回退。
        # 中文注释：FA3 不支持 Blackwell (SM100+) 架构，回退到 FA4 或 FA2。
        if device_capability.major >= 10 and fa_version == 3:
            logger.warning_once(
                "Cannot use FA version 3 on Blackwell platform, "
                "defaulting to FA version 4 if supported, otherwise FA2."
            )
            fa_version = 4 if is_fa_version_supported(4) else 2

        # 中文注释：ALiBi 位置编码与 FA3/FA4 不兼容，回退到 FA2。
        if requires_alibi and fa_version == 3:
            logger.warning_once(
                "Cannot use FA version 3 with ALiBi, defaulting to FA version 2."
            )
            fa_version = 2

        if requires_alibi and fa_version == 4:
            logger.warning_once(
                "Cannot use FA version 4 with ALiBi, defaulting to FA version 2."
            )
            fa_version = 2

        # Some FA3 unsupported SM90 cases can use FA4 when available.
        # 中文注释：在 SM90 上，某些 FA3 不支持的场景（如 head_size > 256、diff-KV with sinks）
        # 可以升级到 FA4 来获得支持。
        if (
            fa_version == 3
            and device_capability.major == 9
            and is_fa_version_supported(4)
        ):
            upgrade_reason = None
            if head_size is not None and head_size > 256:
                upgrade_reason = f"FA3 does not support head_size={head_size} on SM90"
            elif (
                has_sinks
                and head_size is not None
                and head_size_v is not None
                and head_size != head_size_v
            ):
                upgrade_reason = "Diff-KV with sinks"
            if upgrade_reason:
                logger.info_once(
                    "%s: upgrading FlashAttention 3 -> 4",
                    upgrade_reason,
                    scope="local",
                )
                fa_version = 4

        # FA4 currently uses batch-shape-dependent scheduling
        # heuristics on SM100+, which breaks batch invariance.
        # 中文注释：FA4 的调度启发式依赖 batch 形状，无法保证批处理不变性，
        # 因此在 VLLM_BATCH_INVARIANT 模式下回退到 FA2。
        if envs.VLLM_BATCH_INVARIANT and fa_version == 4:
            logger.warning_once(
                "Cannot use FA version 4 with batch invariance, "
                "defaulting to FA version 2.",
            )
            fa_version = 2

        # FA4 on SM100 (Blackwell) has TMEM capacity limits that restrict
        # supported head dimensions.
        # See: https://github.com/Dao-AILab/flash-attention/issues/1959
        # Exception: hdim 192 is supported for MLA's diff-headdim case
        # (qk=192, v=128), added upstream in commits 1a15733e/1b36ab19.
        # 中文注释：FA4 在 Blackwell (SM100+) 上受 TMEM 容量限制，
        # 不支持 head_size > 128 的情况（MLA 的 head_size=192 是已知例外）。
        if (
            fa_version == 4
            and device_capability.major >= 10
            and head_size is not None
            and head_size > 128
            and head_size != 192
        ):
            logger.warning_once(
                "FA4 on Blackwell does not support head_size=%d due to TMEM "
                "capacity limits, defaulting to FA version 2.",
                head_size,
            )
            fa_version = 2

        # 中文注释：最终验证选定的 FA 版本是否在当前环境中可用。
        if not is_fa_version_supported(fa_version):
            logger.error(
                "Cannot use FA version %d is not supported due to %s",
                fa_version,
                fa_version_unsupported_reason(fa_version),
            )

        assert is_fa_version_supported(fa_version)
        return fa_version
    except (ImportError, AssertionError):
        return None


# 中文注释：检查指定的 FlashAttention 版本是否在当前环境中可用。
# 通过调用 vllm_flash_attn 包内部的 is_fa_version_supported 函数来判断。
# 如果 vllm_flash_attn 包未安装（ImportError），则返回 False。
def is_fa_version_supported(fa_version: int) -> bool:
    try:
        from vllm.vllm_flash_attn.flash_attn_interface import (
            is_fa_version_supported as _is_fa_version_supported,
        )

        return _is_fa_version_supported(fa_version)
    except ImportError:
        return False


# 中文注释：检查 FlashAttention 是否支持量化查询输入。
# XPU 平台不支持，其他平台均支持。
def flash_attn_supports_quant_query_input() -> bool:
    return not current_platform.is_xpu()


# 中文注释：检查 FlashAttention 是否支持 Sink Attention 特性。
# Sink Attention 保留初始 token 的注意力权重，防止注意力分散。
# FA3 和 FA4 支持此特性，XPU 平台也支持，FA2 不支持。
def flash_attn_supports_sinks() -> bool:
    if current_platform.is_xpu():
        return True
    return get_flash_attn_version() in (3, 4)


# 中文注释：检查当前平台是否支持 MLA（Multi-head Latent Attention）的 FlashAttention 实现。
# MLA 是 DeepSeek 提出的压缩 KV attention 机制，具有非标准的头维度（如 qk=576, v=512）。
# 目前仅在 CUDA 平台的 SM90 (Hopper) 架构上，且 FA3 可用时才支持。
# FA4 的 CuteDSL 后端目前不支持 MLA 的非标准头维度（受 TMEM 容量限制）。
def flash_attn_supports_mla():
    from vllm.platforms import current_platform

    if current_platform.is_cuda():
        try:
            from vllm.vllm_flash_attn.flash_attn_interface import (
                is_fa_version_supported,
            )

            return is_fa_version_supported(
                3
            ) and current_platform.is_device_capability_family(90)

            # NOTE(Lucas): FA4 CuteDSL does NOT currently support MLA's non-standard
            # head dimensions (576 for qk, 512 for v) due to TMEM capacity limits.

        except (ImportError, AssertionError):
            pass
    return False


# 中文注释：检查模块级导入的 flash_attn_varlen_func 是否为可用的实际实现（而非桩函数）。
# 各平台来源：
#   - CUDA: vllm.vllm_flash_attn.flash_attn_varlen_func（始终可用）
#   - XPU: xpu_ops.flash_attn_varlen_func（始终可用）
#   - ROCm: 上游 flash_attn.flash_attn_varlen_func（取决于是否安装了上游包）
# 注意：这与 AITER flash attention 后端（rocm_aiter_fa.py）中的 flash_attn_varlen_func
# 是不同的。AITER 的使用条件由 _aiter_ops.is_aiter_found_and_supported() 单独处理。
def is_flash_attn_varlen_func_available() -> bool:
    """Check if flash_attn_varlen_func is available.

    This function determines whether the flash_attn_varlen_func imported at module
    level is a working implementation or a stub.

    Platform-specific sources:
    - CUDA: vllm.vllm_flash_attn.flash_attn_varlen_func
    - XPU: xpu_ops.flash_attn_varlen_func
    - ROCm: upstream flash_attn.flash_attn_varlen_func (if available)

    Note: This is separate from the AITER flash attention backend (rocm_aiter_fa.py)
    which uses rocm_aiter_ops.flash_attn_varlen_func. The condition to use AITER is
    handled separately via _aiter_ops.is_aiter_found_and_supported().

    Returns:
        bool: True if a working flash_attn_varlen_func implementation is available.
    """
    if current_platform.is_cuda() or current_platform.is_xpu():
        # CUDA and XPU always have flash_attn_varlen_func available
        return True

    if current_platform.is_rocm():
        # Use the flag set during module import to check if
        # upstream flash-attn was successfully imported
        return _ROCM_FLASH_ATTN_AVAILABLE

    return False
