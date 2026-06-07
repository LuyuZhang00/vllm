# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# =============================================================================
# Ray 环境变量传播工具模块
# =============================================================================
# 本模块负责将驱动进程（driver process）的环境变量传播到 Ray worker 进程中。
#
# 背景说明：
#   在 vLLM 使用 Ray 作为分布式执行后端时，模型推理分布在多个 Ray worker
#   进程中执行。这些 worker 进程需要继承驱动进程中的某些环境变量（如模型路径、
#   缓存目录、日志级别等），但同时需要排除一些不适合传播的变量。
#
# 需要排除的变量类型：
#   1. worker 专属变量（worker_specific_vars）：每个 worker 有各自不同的值，
#      如 CUDA_VISIBLE_DEVICES（每个 worker 看到不同的 GPU）
#   2. Ray 自身管理的变量（RAY_NON_CARRY_OVER_ENV_VARS）：这些变量由 Ray
#      框架自行管理，不应从驱动进程覆盖到 worker
#
# 使用场景：
#   在 RayExecutor 初始化 Ray worker 时调用本模块的函数，确保 worker 能获得
#   正确的环境配置
# =============================================================================

import os

# 导入 Ray 不应从驱动进程传播到 worker 的环境变量集合
# 这些变量由 Ray 框架内部管理，如果手动设置可能导致行为异常
from vllm.ray.ray_env import RAY_NON_CARRY_OVER_ENV_VARS


def get_driver_env_vars(
    worker_specific_vars: set[str],
) -> dict[str, str]:
    """Return driver env vars to propagate to Ray workers.

    Returns everything from ``os.environ`` except ``worker_specific_vars``
    and user-configured exclusions (``RAY_NON_CARRY_OVER_ENV_VARS``).
    """
    # ------------------------------------------------------------------
    # 构建需要排除的环境变量集合
    # ------------------------------------------------------------------
    # 将 worker 专属变量和 Ray 管理的变量合并为一个排除集合
    # 使用集合的 | 运算符合并两个 set，实现高效的 O(1) 查找
    exclude_vars = worker_specific_vars | RAY_NON_CARRY_OVER_ENV_VARS

    # ------------------------------------------------------------------
    # 从驱动进程的环境变量中过滤并返回需要传播的变量
    # ------------------------------------------------------------------
    # 遍历当前进程（驱动进程）的所有环境变量
    # 排除掉不需要传播的变量，返回剩余变量的字典副本
    # 这些变量将被传递给 Ray worker 的运行时环境
    return {key: value for key, value in os.environ.items() if key not in exclude_vars}
