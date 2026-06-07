# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Worker 关闭前的资源清理模块。
# =============================
# 在 vLLM Worker 进程即将退出时，需要释放一些全局持有的资源引用，
# 以确保 Python 垃圾回收器能够正确回收相关对象，防止内存泄漏。
#
# 本模块的 free_before_shutdown() 函数负责清理以下资源：
#   1. KV Cache 配置中的 GPU 块数量引用。
#   2. 编译配置中的静态前向上下文（包含对模型层的引用）。
#   3. 全局旋转位置编码（RoPE）字典，该字典缓存了 RoPE 实例。
#   4. 工作区管理器（Workspace Manager），管理编译时的工作区分配。

from vllm.config import VllmConfig
from vllm.logger import init_logger

logger = init_logger(__name__)


def free_before_shutdown(vllm_config: VllmConfig) -> None:
    """
    在 Worker 关闭前释放全局资源引用，防止内存泄漏。

    此函数通过延迟导入（lazy import）获取全局单例对象的引用，
    然后清除它们持有的数据，使得 Python 垃圾回收器能够回收
    这些对象及其关联的 GPU/CPU 内存。

    清理操作包括：
      1. 将 cache_config.num_gpu_blocks 置为 None，释放 KV Cache 配置引用。
      2. 清空 compilation_config.static_forward_context，该字典持有模型层引用。
      3. 清空全局 RoPE 字典 _ROPE_DICT，释放缓存的旋转位置编码实例。
      4. 重置工作区管理器，释放编译时分配的工作区。

    参数：
      vllm_config: vLLM 全局配置对象，包含 cache、compilation 等子配置。
    """
    # 延迟导入，避免循环依赖，并确保获取的是全局单例。
    from vllm.model_executor.layers.rotary_embedding import _ROPE_DICT
    from vllm.v1.worker.workspace import reset_workspace_manager

    # 1. 释放 KV Cache 配置中的 GPU 块数量引用。
    cache_config = vllm_config.cache_config
    cache_config.num_gpu_blocks = None

    # 2. 清空编译配置中的静态前向上下文。
    #    该字典在模型编译阶段被填充，包含对各模型层的引用。
    #    清空它可以让 GC 回收模型层对象。
    compilation_config = vllm_config.compilation_config
    compilation_config.static_forward_context.clear()

    # 3. 清空全局 RoPE 字典，释放缓存的旋转位置编码实例。
    _ROPE_DICT.clear()

    # 4. 重置工作区管理器，释放编译时分配的工作区内存。
    reset_workspace_manager()
