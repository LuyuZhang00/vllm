# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Prometheus 集成模块
#
# 本模块负责 Prometheus 指标系统在 vLLM 中的初始化和管理。
# 主要功能包括：
#
# 1. 多进程 Prometheus 支持：vLLM 使用多进程架构（调度器、模型运行器、执行器
#    等在不同进程中运行），因此需要设置 PROMETHEUS_MULTIPROC_DIR 环境变量，
#    让 prometheus_client 库能够在多进程间共享指标数据。
#
# 2. 注册表管理：根据是否启用多进程模式，选择合适的 Prometheus 注册表。
#    多进程模式使用 CollectorRegistry + MultiProcessCollector，
#    单进程模式使用默认的全局 REGISTRY。
#
# 3. 指标生命周期管理：提供注销和关闭指标的函数，确保进程退出时正确清理资源，
#    避免残留的指标数据影响下次运行。

import os
import tempfile

from prometheus_client import REGISTRY, CollectorRegistry, multiprocess

from vllm.logger import init_logger

logger = init_logger(__name__)

# 全局临时目录引用，用于 Prometheus 多进程指标共享。
# 使用 TemporaryDirectory 可以在进程退出时自动清理临时目录。
_prometheus_multiproc_dir: tempfile.TemporaryDirectory | None = None


def setup_multiprocess_prometheus():
    """设置 Prometheus 多进程共享目录。

    当 vLLM 以多进程模式运行时，多个进程需要共享 Prometheus 指标数据。
    本函数检查 PROMETHEUS_MULTIPROC_DIR 环境变量是否已设置：
    - 如果未设置：创建一个临时目录并设置该环境变量，
      prometheus_client 库会自动使用此目录来存储跨进程的指标数据。
    - 如果已设置（用户手动配置）：发出警告，因为用户需要确保
      每次 vLLM 运行之间清理该目录，否则会出现不准确的指标数据。

    注意：全局 TemporaryDirectory 对象会在 Python 解释器退出时自动清理。
    """
    global _prometheus_multiproc_dir

    if "PROMETHEUS_MULTIPROC_DIR" not in os.environ:
        # Make TemporaryDirectory for prometheus multiprocessing
        # Note: global TemporaryDirectory will be automatically
        # cleaned up upon exit.
        _prometheus_multiproc_dir = tempfile.TemporaryDirectory()
        os.environ["PROMETHEUS_MULTIPROC_DIR"] = _prometheus_multiproc_dir.name
        logger.debug(
            "Created PROMETHEUS_MULTIPROC_DIR at %s", _prometheus_multiproc_dir.name
        )
    else:
        logger.warning(
            "Found PROMETHEUS_MULTIPROC_DIR was set by user. "
            "This directory must be wiped between vLLM runs or "
            "you will find inaccurate metrics. Unset the variable "
            "and vLLM will properly handle cleanup."
        )


def get_prometheus_registry() -> CollectorRegistry:
    """获取适合当前环境的 Prometheus 注册表。

    根据是否设置了 PROMETHEUS_MULTIPROC_DIR 环境变量来决定使用哪种注册表：

    1. 多进程模式（设置了 PROMETHEUS_MULTIPROC_DIR）：
       创建一个新的 CollectorRegistry，并附加 MultiProcessCollector，
       以收集来自所有子进程的指标数据。

    2. 单进程模式（未设置）：
       直接返回全局默认的 REGISTRY。

    Returns:
        CollectorRegistry: 适合当前部署模式的 Prometheus 注册表实例。
    """
    if os.getenv("PROMETHEUS_MULTIPROC_DIR") is not None:
        logger.debug("Using multiprocess registry for prometheus metrics")
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
        return registry

    return REGISTRY


def unregister_vllm_metrics():
    """注销所有已注册的 vLLM 指标收集器。

    在以下场景中需要调用此函数：
    1. 测试和 CI/CD 环境：测试之间需要清理指标，避免重复注册导致错误。
    2. 多进程模式：需要从全局注册表中注销指标，防止内存泄漏和指标重复。

    本函数遍历全局注册表中的所有收集器，找到名称中包含 "vllm" 的收集器并注销。
    """
    registry = REGISTRY
    # Unregister any existing vLLM collectors
    for collector in list(registry._collector_to_names):
        if hasattr(collector, "_name") and "vllm" in collector._name:
            registry.unregister(collector)


def shutdown_prometheus():
    """关闭 Prometheus 指标系统。

    在进程退出前调用，用于标记当前进程的指标数据为"死亡"状态。
    这对于多进程模式特别重要：
    - 在多进程模式下，每个进程将自己的指标数据写入共享目录。
    - 当进程退出时，如果不标记为死亡，其残留的指标数据会继续被收集，
      导致指标不准确（如计数器重复累加）。
    - 调用 mark_process_dead 会告诉 MultiProcessCollector
      忽略该进程的指标文件。
    """
    path = _prometheus_multiproc_dir
    if path is None:
        return
    try:
        pid = os.getpid()
        multiprocess.mark_process_dead(pid, path)
        logger.debug("Marked Prometheus metrics for process %d as dead", pid)
    except Exception as e:
        logger.error("Error during metrics cleanup: %s", str(e))
