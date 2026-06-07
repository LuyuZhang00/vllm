# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# =============================================================================
# 中文注释：单进程执行器模块 (UniProcExecutor)
# =============================================================================
#
# 【模块职责】
#   本模块实现了 UniProcExecutor（单进程执行器）和
#   ExecutorWithExternalLauncher（外部启动器执行器）。
#   它们是 vLLM V1 引擎中最简单的 Executor 实现，适用于单 GPU 推理场景。
#
# 【核心设计理念】
#   与 MultiprocExecutor（多进程执行器）不同，UniProcExecutor 只在当前
#   进程中创建一个 Worker 实例，不需要进程间通信（IPC），因此具有更低的
#   延迟和更简单的调试体验。
#
# 【与 MultiprocExecutor 的关键区别】
#   1. 进程模型：
#      - UniProcExecutor: 单进程，Worker 与 Executor 在同一进程中运行
#      - MultiprocExecutor: 多进程，每个 GPU 一个独立子进程
#
#   2. 通信开销：
#      - UniProcExecutor: 无 IPC 开销，直接函数调用
#      - MultiprocExecutor: 通过共享内存消息队列（shared memory MQ）通信
#
#   3. 张量并行：
#      - UniProcExecutor: 不支持张量并行（TP=1），只有一个 Worker
#      - MultiprocExecutor: 支持张量并行（TP>1），多个 Worker 协同工作
#
#   4. 适用场景：
#      - UniProcExecutor: 单 GPU 推理、开发调试、小模型推理
#      - MultiprocExecutor: 多 GPU 生产环境、大模型张量并行
#
# 【在 vLLM V1 请求处理链路中的位置】
#   EngineCore（引擎核心）
#     -> Scheduler 产出 SchedulerOutput
#       -> UniProcExecutor.collective_rpc()
#         -> 直接调用 driver_worker 的方法（无 IPC）
#           -> GPUModelRunner 执行模型前向推理
#             -> 返回 ModelRunnerOutput
#
# 【类层次结构】
#   Executor（抽象基类，定义在 abstract.py）
#     └── UniProcExecutor（本文件）
#           └── ExecutorWithExternalLauncher（本文件）
#
# =============================================================================

import os
from collections.abc import Callable
from concurrent.futures import Future
from functools import cached_property
from multiprocessing import Lock
from typing import Any

import torch
import torch.distributed as dist

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.network_utils import get_distributed_init_method, get_ip, get_open_port
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.executor.abstract import Executor
from vllm.v1.executor.vllm_net_devices import set_worker_net_device
from vllm.v1.outputs import AsyncModelRunnerOutput, DraftTokenIds, ModelRunnerOutput
from vllm.v1.serial_utils import run_method
from vllm.v1.worker.worker_base import WorkerWrapperBase

logger = init_logger(__name__)


# =============================================================================
# 中文注释：AsyncOutputFuture 类
# =============================================================================
# 【功能说明】
#   AsyncOutputFuture 是对 Python 标准库 concurrent.futures.Future 的扩展，
#   专门用于包装异步模型运行输出（AsyncModelRunnerOutput）。
#
# 【设计动机】
#   在异步调度模式下，execute_model() 可能返回一个尚未完成的异步输出
#   （AsyncModelRunnerOutput），该输出需要在 GPU 计算完成后才能获取结果。
#   AsyncOutputFuture 将这个异步输出包装为标准的 Future 接口，使得
#   调用者可以通过统一的 future.result() 接口获取结果。
#
# 【工作流程】
#   1. 创建时：接收 AsyncModelRunnerOutput 和 single_value 标志
#   2. 调用 result() 时：
#      a. 如果结果尚未就绪（!super().done()），调用 async_output.get_output()
#         获取实际输出（这是一个阻塞调用，等待 GPU 计算完成）
#      b. 将结果设置到 Future 中（set_result 或 set_exception）
#      c. 返回结果
#
# 【参数说明】
#   - async_output: 异步模型运行输出，封装了 GPU 异步计算的结果
#   - single_value: 是否返回单个值（True）还是列表（False）
#     - True: 直接返回 output（用于 UniProcExecutor，只有一个 Worker）
#     - False: 返回 [output]（用于与多 Worker 接口兼容）
#
# 【与标准 Future 的区别】
#   - 不支持 timeout 参数（抛出 RuntimeError）
#   - 结果获取是惰性的：首次调用 result() 时才触发实际计算
# =============================================================================
class AsyncOutputFuture(Future):
    def __init__(self, async_output: AsyncModelRunnerOutput, single_value: bool):
        self.async_output = async_output
        self.single_value = single_value
        super().__init__()

    def result(self, timeout=None):
        if timeout is not None:
            raise RuntimeError("timeout not implemented")

        if not super().done():
            try:
                output = self.async_output.get_output()
                self.set_result(output if self.single_value else [output])
            except Exception as e:
                self.set_exception(e)
        return super().result()


# =============================================================================
# 中文注释：UniProcExecutor 类
# =============================================================================
# 【功能说明】
#   UniProcExecutor 是 vLLM V1 引擎的单进程执行器实现。
#   它在当前进程中创建一个 Worker 实例，直接调用 Worker 的方法执行
#   模型推理，无需任何进程间通信机制。
#
# 【适用场景】
#   1. 单 GPU 推理（TP=1，PP=1）
#   2. 开发和调试阶段（单进程更易于使用调试器）
#   3. 小模型推理（不需要多 GPU 加速）
#   4. 通过分布式后端配置 "uni" 显式选择
#
# 【核心属性】
#   - driver_worker: WorkerWrapperBase 实例，封装了实际的 GPU Worker
#
# 【核心方法】
#   - _init_executor(): 初始化 Worker，加载模型
#   - collective_rpc(): 直接调用 Worker 方法（无 IPC 开销）
#   - execute_model(): 执行模型推理
#   - sample_tokens(): 执行 token 采样
#   - check_health(): 健康检查（始终返回成功）
#   - shutdown(): 关闭 Worker
#
# 【与 MultiprocExecutor 的对比】
#   ┌─────────────────┬────────────────────┬────────────────────┐
#   │     特性         │  UniProcExecutor   │ MultiprocExecutor  │
#   ├─────────────────┼────────────────────┼────────────────────┤
#   │ Worker 数量      │ 1 个               │ N 个（每 GPU 一个）│
#   │ 进程模型         │ 单进程              │ 多进程              │
#   │ 通信方式         │ 直接函数调用         │ 共享内存消息队列    │
#   │ 张量并行         │ 不支持（TP=1）      │ 支持（TP>1）        │
#   │ 流水线并行       │ 不支持              │ 支持（PP>1）        │
#   │ IPC 开销         │ 无                  │ 有（序列化/反序列化）│
#   │ 调试难度         │ 低                  │ 高                  │
#   │ 生产适用性       │ 低（单 GPU）        │ 高（多 GPU）        │
#   └─────────────────┴────────────────────┴────────────────────┘
# =============================================================================
class UniProcExecutor(Executor):
    def _init_executor(self) -> None:
        """Initialize the worker and load the model."""
        # 中文注释：初始化执行器
        #
        # 【功能说明】
        #   这是 UniProcExecutor 的核心初始化方法，在 Executor.__init__() 中
        #   被调用。它完成以下工作：
        #   1. 创建 Worker 包装器实例
        #   2. 获取分布式初始化参数
        #   3. 初始化 Worker 并加载模型
        #
        # 【初始化流程详解】
        #   步骤 1: 创建 WorkerWrapperBase 实例
        #     - WorkerWrapperBase 是 Worker 的通用包装器
        #     - rpc_rank=0 表示这是唯一的 Worker（单进程模式）
        #
        #   步骤 2: 获取分布式参数
        #     - 调用 _distributed_args() 获取分布式初始化方法、rank、local_rank
        #     - 即使是单 GPU，也需要初始化分布式环境（用于某些 CUDA 操作）
        #
        #   步骤 3: 构建 Worker 初始化参数
        #     - vllm_config: 全局配置
        #     - local_rank: 本地 GPU 编号
        #     - rank: 全局 rank（单进程模式下为 0）
        #     - distributed_init_method: 分布式初始化方法（单机用 TCP）
        #     - is_driver_worker=True: 标记为驱动 Worker（负责收集和返回结果）
        #     - shared_worker_lock: 共享锁（单进程模式下只有一个 Worker）
        #
        #   步骤 4: 设置网络设备环境变量
        #     - 如果配置了 VLLM_GPU_NIC_PCIE_MAPPING，设置对应的网络设备
        #
        #   步骤 5: 初始化 Worker
        #     - init_worker(): 创建 Worker 实例（如 GPUWorker）
        #     - init_device(): 初始化 GPU 设备
        #     - load_model(): 加载模型权重到 GPU
        #
        #   步骤 6: 更新块大小
        #     - 根据注意力后端的要求，更新调度器的块大小配置
        #
        # 【为什么需要 Lock？】
        #   shared_worker_lock 参数在单进程模式下看似多余，但它是为了
        #   与 MultiprocExecutor 保持接口一致。WorkerWrapperBase 的接口
        #   要求传入此参数。
        self.driver_worker = WorkerWrapperBase(rpc_rank=0)
        distributed_init_method, rank, local_rank = self._distributed_args()
        kwargs = dict(
            vllm_config=self.vllm_config,
            local_rank=local_rank,
            rank=rank,
            distributed_init_method=distributed_init_method,
            is_driver_worker=True,
            shared_worker_lock=Lock(),
        )

        # Set net device env vars for the worker if VLLM_GPU_NIC_PCIE_MAPPING is set
        set_worker_net_device(local_rank, self.vllm_config)

        self.driver_worker.init_worker(all_kwargs=[kwargs])
        self.driver_worker.init_device()

        # 中文注释：加载模型权重
        # 如果启用了弹性 EP 扩容模式，使用特殊的加载方式；
        # 否则使用标准的模型加载流程
        if envs.VLLM_ELASTIC_EP_SCALE_UP_LAUNCH:
            self.driver_worker.elastic_ep_execute("load_model")
        else:
            self.driver_worker.load_model()
        # 中文注释：根据注意力后端的要求更新块大小
        # 不同的注意力后端（如 FlashAttention、Triton）对块大小有不同的要求
        current_platform.update_block_size_for_backend(self.vllm_config)

    def _distributed_args(self) -> tuple[str, int, int]:
        """Return (distributed_init_method, rank, local_rank)."""
        # 中文注释：获取分布式初始化参数
        #
        # 【功能说明】
        #   即使在单 GPU 模式下，vLLM 仍然需要初始化 PyTorch 的分布式环境
        #   （因为某些 CUDA 操作和通信原语依赖于此）。
        #   此方法为单进程模式生成最小化的分布式初始化参数。
        #
        # 【返回值说明】
        #   - distributed_init_method: 分布式初始化方法
        #     使用 TCP 方式（"tcp://ip:port"），适用于单机单进程场景
        #   - rank: 全局 rank，单进程模式下固定为 0
        #   - local_rank: 本地 GPU 编号，从设备字符串中解析
        #     例如 "cuda:0" -> local_rank=0, "cuda:1" -> local_rank=1
        #
        # 【设备字符串解析逻辑】
        #   device_config.device 的字符串形式可能是：
        #   - "cuda:0" -> 分割后得到 ["cuda", "0"] -> local_rank=0
        #   - "cuda"   -> 分割后得到 ["cuda"] -> local_rank=0（默认）
        #   - "cpu"    -> 分割后得到 ["cpu"] -> local_rank=0（默认）
        distributed_init_method = get_distributed_init_method(get_ip(), get_open_port())
        # set local rank as the device index if specified
        device_info = self.vllm_config.device_config.device.__str__().split(":")
        local_rank = int(device_info[1]) if len(device_info) > 1 else 0
        return distributed_init_method, 0, local_rank

    @cached_property
    def max_concurrent_batches(self) -> int:
        # 中文注释：最大并发批次数
        #
        # 【功能说明】
        #   返回执行器支持的最大并发批次数。
        #   - 异步调度模式 (async_scheduling=True): 返回 2，允许流水线执行
        #     （一个批次在 GPU 上执行时，可以准备下一个批次）
        #   - 同步调度模式 (async_scheduling=False): 返回 1，严格串行执行
        #
        # 【使用 @cached_property 的原因】
        #   调度模式在运行期间不会改变，因此只需计算一次并缓存结果。
        return 2 if self.scheduler_config.async_scheduling else 1

    def collective_rpc(  # type: ignore[override]
        self,
        method: str | Callable,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
        non_block: bool = False,
        single_value: bool = False,
    ) -> Any:
        # 中文注释：集体远程过程调用（RPC）
        #
        # 【功能说明】
        #   这是 UniProcExecutor 的核心方法，负责将 RPC 调用分发到 Worker。
        #   与 MultiprocExecutor 不同，这里不需要通过 IPC 机制通信，
        #   而是直接在当前进程中调用 Worker 的方法。
        #
        # 【参数说明】
        #   - method: 要调用的 Worker 方法名（如 "execute_model"）或可调用对象
        #   - timeout: 超时时间（未实现，保留接口兼容性）
        #   - args: 位置参数元组
        #   - kwargs: 关键字参数字典
        #   - non_block: 是否非阻塞模式
        #     - False: 同步执行，直接返回结果
        #     - True: 异步执行，返回 Future 对象
        #   - single_value: 是否返回单个值
        #     - True: 直接返回 result（用于只有一个 Worker 的场景）
        #     - False: 返回 [result]（用于与多 Worker 接口兼容）
        #
        # 【执行流程】
        #   1. 同步模式 (non_block=False):
        #      - 直接调用 run_method(self.driver_worker, method, args, kwargs)
        #      - 返回结果（单值或列表形式）
        #
        #   2. 异步模式 (non_block=True):
        #      - 调用 run_method 获取结果
        #      - 如果结果是 AsyncModelRunnerOutput：
        #        包装为 AsyncOutputFuture 返回（惰性获取结果）
        #      - 如果结果是普通值：
        #        包装为已完成的 Future 返回
        #      - 如果发生异常：
        #        包装为带异常的 Future 返回
        #
        # 【与 MultiprocExecutor 的对比】
        #   MultiprocExecutor 的 collective_rpc 需要：
        #   1. 将 method/args/kwargs 序列化
        #   2. 通过共享内存消息队列发送到各 Worker 子进程
        #   3. Worker 子进程反序列化并执行
        #   4. 将结果序列化后通过响应队列返回
        #
        #   UniProcExecutor 直接调用 run_method()，没有任何序列化/IPC 开销。
        if kwargs is None:
            kwargs = {}

        if not non_block:
            # 中文注释：同步模式 - 直接调用 Worker 方法并返回结果
            result = run_method(self.driver_worker, method, args, kwargs)
            return result if single_value else [result]

        # 中文注释：异步模式 - 返回 Future 对象
        try:
            result = run_method(self.driver_worker, method, args, kwargs)
            if isinstance(result, AsyncModelRunnerOutput):
                # 中文注释：如果结果是异步模型输出，包装为 AsyncOutputFuture
                # AsyncOutputFuture 会在首次调用 result() 时触发实际的 GPU 同步
                return AsyncOutputFuture(result, single_value)
            # 中文注释：普通结果，直接包装为已完成的 Future
            future = Future[Any]()
            future.set_result(result if single_value else [result])
        except Exception as e:
            # 中文注释：异常情况，创建带异常的 Future
            future = Future[Any]()
            future.set_exception(e)
        return future

    def execute_model(  # type: ignore[override]
        self, scheduler_output: SchedulerOutput, non_block: bool = False
    ) -> ModelRunnerOutput | None | Future[ModelRunnerOutput | None]:
        # 中文注释：执行模型推理
        #
        # 【功能说明】
        #   这是 UniProcExecutor 执行模型推理的入口方法。
        #   它将 SchedulerOutput 传递给 Worker 的 execute_model 方法，
        #   触发模型的前向推理。
        #
        # 【执行流程】
        #   1. 通过 collective_rpc 调用 Worker 的 "execute_model" 方法
        #   2. 将 scheduler_output 作为参数传递（包含本轮要执行的批次信息）
        #   3. single_value=True 表示直接返回结果（不是列表）
        #   4. 在非阻塞模式下，如果任务已完成且有异常，立即抛出
        #
        # 【参数说明】
        #   - scheduler_output: 调度器输出，包含：
        #     - 要处理的请求列表及其 token 信息
        #     - KV 缓存块的分配/释放信息
        #     - 运行的批次元数据
        #   - non_block: 是否非阻塞执行
        #     - False: 同步等待结果
        #     - True: 立即返回 Future
        #
        # 【返回值】
        #   - 同步模式: ModelRunnerOutput 或 None
        #   - 异步模式: Future[ModelRunnerOutput | None]
        #
        # 【异常处理（非阻塞模式）】
        #   在非阻塞模式下，如果 execute_model 立即完成（例如在 CUDA Graph
        #   命中时），会检查是否有异常并立即抛出，而不是等到调用者获取结果时。
        #   这样可以让错误更早暴露，便于调试。
        output = self.collective_rpc(
            "execute_model",
            args=(scheduler_output,),
            non_block=non_block,
            single_value=True,
        )
        # In non-blocking mode, surface any exception as early as possible.
        if non_block and output.done():
            # Raise the exception in-line if the task failed.
            output.result()
        return output

    def sample_tokens(  # type: ignore[override]
        self, grammar_output: GrammarOutput | None, non_block: bool = False
    ) -> ModelRunnerOutput | None | Future[ModelRunnerOutput | None]:
        # 中文注释：执行 token 采样
        #
        # 【功能说明】
        #   在 V1 引擎中，模型推理和 token 采样是分离的两个步骤：
        #   1. execute_model(): 执行模型前向推理，计算 logits
        #   2. sample_tokens(): 对 logits 进行采样，得到最终生成的 token
        #
        #   这种分离允许更灵活的调度策略（如异步调度）。
        #
        # 【参数说明】
        #   - grammar_output: 语法引导输出（用于结构化生成/JSON 格式约束）
        #     如果为 None 则不使用语法约束
        #   - non_block: 是否非阻塞执行
        #
        # 【返回值】
        #   - 同步模式: ModelRunnerOutput（包含采样后的 token ID）
        #   - 异步模式: Future[ModelRunnerOutput]
        return self.collective_rpc(
            "sample_tokens",
            args=(grammar_output,),
            non_block=non_block,
            single_value=True,
        )

    def take_draft_token_ids(self) -> DraftTokenIds | None:
        # 中文注释：获取推测解码的草稿 token ID
        #
        # 【功能说明】
        #   在推测解码（Speculative Decoding）场景中：
        #   1. 草稿模型（draft model）先快速生成多个候选 token
        #   2. 目标模型（target model）并行验证这些候选 token
        #   3. 选择第一个不匹配位置之前的 token 作为最终输出
        #
        #   此方法从 Worker 获取草稿模型生成的候选 token ID。
        #
        # 【返回值】
        #   - DraftTokenIds: 包含请求 ID 和对应的草稿 token ID 列表
        #   - None: 如果没有使用推测解码，或没有草稿 token
        return self.collective_rpc("take_draft_token_ids", single_value=True)

    def check_health(self) -> None:
        # UniProcExecutor will always be healthy as long as
        # it's running.
        # 中文注释：健康检查
        #
        # 【功能说明】
        #   UniProcExecutor 的健康检查始终通过（直接返回）。
        #   原因：如果 UniProcExecutor 进程本身还在运行，说明一切正常。
        #   如果进程崩溃了，这个方法根本不会被调用到。
        #
        #   相比之下，MultiprocExecutor 需要检查各子进程是否存活、
        #   IPC 通道是否正常等。
        return

    def shutdown(self) -> None:
        # 中文注释：关闭执行器
        #
        # 【功能说明】
        #   关闭 Worker，释放 GPU 资源。
        #   使用 walrus 操作符 (:=) 同时检查和赋值 worker 变量。
        #   如果 driver_worker 存在，调用其 shutdown() 方法。
        if worker := self.driver_worker:
            worker.shutdown()

    @classmethod
    def supports_async_scheduling(cls) -> bool:
        # 中文注释：声明支持异步调度
        #
        # 【功能说明】
        #   返回 True 表示 UniProcExecutor 支持异步调度。
        #   异步调度允许在等待当前批次执行结果的同时，提前准备下一个批次，
        #   从而提高 GPU 利用率和整体吞吐量。
        #
        # 【异步调度的工作原理】
        #   1. 调度器准备批次 A，调用 execute_model(non_block=True)
        #   2. 不等待批次 A 完成，立即准备批次 B
        #   3. 当批次 A 的结果返回时，处理结果并继续
        #   这需要 max_concurrent_batches >= 2 才能生效。
        return True


# =============================================================================
# 中文注释：ExecutorWithExternalLauncher 类
# =============================================================================
# 【功能说明】
#   ExecutorWithExternalLauncher 是 UniProcExecutor 的子类，专为使用
#   外部启动器（如 torchrun）进行离线推理而设计。
#
# 【设计动机】
#   在某些场景下，用户希望使用 torchrun 等标准分布式启动器来运行
#   vLLM 的离线推理。这种情况下：
#   - 每个进程由 torchrun 启动，各自管理一个 GPU
#   - 每个进程独立运行一个完整的 EngineCore + Executor + Worker
#   - 所有进程通过 PyTorch 的分布式通信（NCCL）进行张量并行
#
# 【核心思想】
#   虽然是张量并行推理，但每个 Executor 只创建一个 Worker。
#   用户使用 torchrun 启动多个独立的引擎实例，这些实例协同工作
#   处理相同的 prompts。当调度是确定性的，所有引擎会生成相同的输出，
#   它们之间不需要同步状态。
#
# 【使用示例】
#   参见 examples/features/torchrun/torchrun_example_offline.py
#
# 【与 UniProcExecutor 的区别】
#   1. 分布式初始化：
#      - UniProcExecutor: 自动获取 IP/端口，使用 TCP 初始化
#      - ExecutorWithExternalLauncher: 使用 "env://" 方法，依赖 torchrun
#        设置的环境变量（RANK, LOCAL_RANK, MASTER_ADDR, MASTER_PORT）
#
#   2. 多进程模式：
#      - UniProcExecutor: 禁用多进程（VLLM_ENABLE_V1_MULTIPROCESSING=0）
#      - ExecutorWithExternalLauncher: 同样禁用，以确保确定性执行
#
#   3. 可用内存探测：
#      - UniProcExecutor: 直接返回本地 GPU 的可用内存
#      - ExecutorWithExternalLauncher: 取所有 rank 的最小值，确保一致性
#
# 【参考链接】
#   - 设计动机: https://github.com/vllm-project/vllm/issues/11400
# =============================================================================
class ExecutorWithExternalLauncher(UniProcExecutor):
    """An executor that uses external launchers to launch engines,
    specially designed for torchrun-compatible launchers, for
    offline inference with tensor parallelism.

    see https://github.com/vllm-project/vllm/issues/11400 for
    the motivation, and examples/features/torchrun/torchrun_example_offline.py
    for the usage example.

    The key idea: although it is tensor-parallel inference, we only
    create one worker per executor, users will launch multiple
    engines with torchrun-compatible launchers, and all these engines
    work together to process the same prompts. When scheduling is
    deterministic, all the engines will generate the same outputs,
    and they don't need to synchronize the states with each other.
    """

    def _init_executor(self) -> None:
        """Initialize the worker and load the model."""
        # 中文注释：初始化执行器（带外部启动器检查）
        #
        # 【功能说明】
        #   在调用父类的 _init_executor() 之前，先检查是否禁用了多进程模式。
        #   这是因为 ExecutorWithExternalLauncher 需要确定性执行，
        #   而多进程模式可能引入非确定性行为。
        #
        # 【断言检查】
        #   VLLM_ENABLE_V1_MULTIPROCESSING 必须为 False（0），
        #   否则抛出 AssertionError，提示用户设置环境变量。
        assert not envs.VLLM_ENABLE_V1_MULTIPROCESSING, (
            "To get deterministic execution, "
            "please set VLLM_ENABLE_V1_MULTIPROCESSING=0"
        )
        super()._init_executor()

    def _distributed_args(self) -> tuple[str, int, int]:
        # 中文注释：获取分布式初始化参数（从环境变量）
        #
        # 【功能说明】
        #   与 UniProcExecutor 不同，这里不自动生成 IP/端口，
        #   而是从 torchrun 设置的环境变量中读取分布式参数。
        #
        # 【依赖的环境变量】
        #   torchrun 会自动设置以下环境变量：
        #   - RANK: 当前进程的全局 rank
        #   - LOCAL_RANK: 当前进程的本地 rank（对应 GPU 编号）
        #   - MASTER_ADDR: 主节点的 IP 地址
        #   - MASTER_PORT: 主节点的端口号
        #
        # 【分布式初始化方法】
        #   使用 "env://" 方法，PyTorch 会从上述环境变量中读取配置。
        #   这是 torchrun 启动器的标准做法。
        # engines are launched in torchrun-compatible launchers
        # so we can use the env:// method.
        # required env vars:
        # - RANK
        # - LOCAL_RANK
        # - MASTER_ADDR
        # - MASTER_PORT
        distributed_init_method = "env://"
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        return distributed_init_method, rank, local_rank

    def determine_available_memory(self) -> list[int]:  # in bytes
        # 中文注释：探测可用显存（跨 rank 取最小值）
        #
        # 【功能说明】
        #   在张量并行场景中，所有 rank 的可用显存应该一致（因为它们
        #   加载相同的模型权重）。但由于内存碎片等原因，不同 rank 报告的
        #   可用内存可能略有差异。
        #
        #   为了确保所有 rank 使用相同的 KV 缓存块数量，这里取所有 rank
        #   报告的可用内存的最小值。
        #
        # 【执行流程】
        #   1. 调用父类的 determine_available_memory() 获取本地 GPU 的可用内存
        #   2. 使用 PyTorch 的 all_reduce 操作（MIN 归约）获取所有 rank 的最小值
        #   3. 返回统一的最小可用内存值
        #
        # 【为什么使用 CPU 张量和 CPU 通信组】
        #   - 使用 CPU 张量 (device="cpu") 避免占用 GPU 显存
        #   - 使用 CPU 通信组 (cpu_group) 避免与 GPU 通信竞争带宽
        #   - 这只是一个很小的标量值，不需要 GPU 加速
        # we need to get the min across all ranks.
        memory = super().determine_available_memory()
        from vllm.distributed.parallel_state import get_world_group

        cpu_group = get_world_group().cpu_group
        memory_tensor = torch.tensor([memory], device="cpu", dtype=torch.int64)
        dist.all_reduce(memory_tensor, group=cpu_group, op=dist.ReduceOp.MIN)
        return [memory_tensor.item()]
