# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# =============================================================================
# vLLM v1 执行器（Executor）包
# =============================================================================
# 本包是 vLLM v1 引擎的执行器模块，负责在底层设备（GPU 等）上实际运行模型。
#
# 主要功能：
#   1. 定义执行器的抽象基类 Executor，规范所有执行器必须实现的接口
#   2. 提供单进程执行器 UniProcExecutor，用于单 GPU 或张量并行度为 1 的场景
#   3. 提供多进程执行器（MultiprocExecutor），用于多 GPU 张量并行场景
#      （该类通过延迟导入避免在不需要时加载多进程依赖）
#   4. 提供 Ray 分布式执行器（RayExecutor），用于跨节点分布式推理
#
# 执行器在 vLLM v1 架构中的位置：
#   Scheduler（调度器） -> Executor（执行器） -> Worker（工作进程） -> GPUModelRunner（模型运行器）
#
# 包结构：
#   - abstract.py          -- 定义执行器抽象基类 Executor，包含所有执行器共有的接口方法
#   - uniproc_executor.py  -- 单进程执行器，在当前进程中直接运行模型，适用于 TP=1 场景
#   - multiproc_executor.py-- 多进程执行器，每个 GPU 对应一个子进程，适用于 TP>1 场景
#   - ray_executor.py      -- Ray 分布式执行器，利用 Ray 框架管理跨节点的工作进程
#   - ray_env_utils.py     -- Ray 环境变量工具，负责将驱动进程的环境变量传播到 Ray worker
#   - vllm_net_devices.py  -- RDMA 网络设备管理，实现 GPU 到网卡的 PCIe 地址映射
#
# 典型使用流程：
#   1. EngineCore 根据并行配置选择合适的执行器类型
#   2. 执行器初始化时创建 Worker 并分配 GPU 设备
#   3. 调度器每轮迭代后，将 SchedulerOutput 发送给执行器
#   4. 执行器协调各 Worker 执行模型前向计算
#   5. 计算结果返回给 EngineCore 进行后续处理
# =============================================================================

# 导入抽象执行器基类，所有具体执行器都继承自该类
from .abstract import Executor

# 导入单进程执行器，适用于张量并行度为 1 的场景（单 GPU 推理）
from .uniproc_executor import UniProcExecutor

# __all__ 定义了该包对外暴露的公共接口
# 注意：MultiprocExecutor 和 RayExecutor 不在此列表中，
# 因为它们通过 Executor.get_class() 工厂方法根据配置动态选择，
# 而非由外部代码直接导入
__all__ = ["Executor", "UniProcExecutor"]
