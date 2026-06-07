# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Must be imported firstly
# 注意：必须首先导入此模块，它初始化了 CPU 共享内存（SHM）的底层环境
import vllm.v1.worker.cpu.shm  # noqa # isort: skip

import math
import os
import sys
from typing import Any

import psutil
import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.platforms import CpuArchEnum, current_platform
from vllm.profiler.wrapper import TorchProfilerWrapper
from vllm.utils.cpu_resource_utils import (
    get_allowed_cpu_list,
    get_memory_node_info,
    get_visible_memory_node,
)
from vllm.utils.mem_utils import format_gib
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.worker.cpu_model_runner import CPUModelRunner
from vllm.v1.worker.gpu_worker import Worker, init_worker_distributed_environment
from vllm.v1.worker.worker_base import CompilationTimes

logger = init_logger(__name__)


class CPUWorker(Worker):
    """
    CPU 后端的工作进程。

    ==========================================================
    【类职责】
    ==========================================================
    CPUWorker 继承自 GPUWorker，但将所有 GPU 特有逻辑替换为 CPU 实现：
    1. 设备初始化：绑定 NUMA 内存节点，设置 CPU 线程亲和性
    2. 内存管理：基于 CPU 内存（而非 GPU 显存）管理 KV Cache
    3. 模型运行：使用 CPUModelRunner 执行推理
    4. 性能分析：使用 Torch CPU Profiler（而非 CUDA Profiler）

    ==========================================================
    【CPU 内存管理策略】
    ==========================================================
    与 GPU 不同，CPU 没有独立的显存。CPUWorker 使用以下策略：
    1. 通过 --gpu-memory-utilization 参数（复用 GPU 的参数名）
       指定 KV Cache 可使用的 CPU 内存比例
    2. 通过 --kv-cache-memory-bytes 参数显式指定 KV Cache 的字节数
    3. 内存绑定到特定 NUMA 节点以优化访问延迟

    ==========================================================
    【NUMA 优化】
    ==========================================================
    在多 socket 系统上，CPU Worker 会绑定到特定的 NUMA 节点：
    1. 使用 get_visible_memory_node() 获取可用的内存节点
    2. 使用 get_allowed_cpu_list() 获取允许使用的 CPU 核心
    3. 调用 torch.ops._C.init_cpu_memory_env() 初始化内存环境
    4. 所有后续的内存分配都在该 NUMA 节点上进行

    ==========================================================
    【不支持的功能】
    ==========================================================
    1. Sleep/Wake Up 模式：CPU 内存无法像 GPU 显存那样释放
    2. Custom All Reduce：CPU 后端使用标准的 all_reduce 实现
    3. Dummy 权重加载：不支持弹性 EP 扩展
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ):
        """
        初始化 CPU Worker。

        ==========================================================
        【初始化流程】
        ==========================================================
        1. 获取可用的 NUMA 内存节点和 CPU 核心列表
        2. 验证内存节点是否可用（如果不匹配则警告）
        3. 初始化 CPU 内存环境（绑定到 NUMA 节点）
        4. 计算请求的 CPU 内存量并验证可用性
        5. 调用父类初始化
        6. 禁用 custom all_reduce
        7. 初始化 Torch Profiler（如果配置了的话）

        参数：
            vllm_config: vLLM 全局配置
            local_rank: 本地进程排名
            rank: 全局进程排名
            distributed_init_method: 分布式初始化方法（如 "tcp://host:port"）
            is_driver_worker: 是否为驱动（主）工作进程
        """
        # TODO: use numactl for process setup
        # TODO: optimize for `interleaved` policy
        # 绑定内存节点
        allowed_memory_nodes = get_visible_memory_node()
        allowed_cpu_list = get_allowed_cpu_list()
        cpu_core = allowed_cpu_list[0]

        # 验证内存节点是否在允许列表中
        # TODO: some CI hosts are not correctly set, change to assertion
        # after fix
        if cpu_core.numa_node not in allowed_memory_nodes:
            logger.warning(
                "Node %s is not in available memory nodes %s.",
                cpu_core.numa_node,
                allowed_memory_nodes,
            )

        # 初始化 CPU 内存环境，绑定到指定的 NUMA 节点
        torch.ops._C.init_cpu_memory_env([cpu_core.numa_node])

        # 计算请求的 CPU 内存量
        # 注意：--gpu-memory-utilization 在 CPU 后端控制的是 CPU 内存比例
        memory_status = get_memory_node_info(cpu_core.numa_node)
        memory_fraction = vllm_config.cache_config.gpu_memory_utilization
        self.requested_cpu_memory = math.ceil(
            memory_status.total_memory * memory_fraction
        )
        available_memory = memory_status.available_memory

        # 验证请求的内存量不超过可用内存
        if (
            vllm_config.cache_config.kv_cache_memory_bytes is None
            and self.requested_cpu_memory > available_memory
        ):
            raise ValueError(
                f"Available memory on node {cpu_core.numa_node} "
                f"({format_gib(available_memory)}/"
                f"{format_gib(memory_status.total_memory)} GiB) on startup "
                f"is less than desired CPU memory utilization "
                f"({vllm_config.cache_config.gpu_memory_utilization}, "
                f"{format_gib(self.requested_cpu_memory)} GiB). "
                "On the CPU backend, the `--gpu-memory-utilization` flag "
                "controls the fraction of CPU memory reserved (despite its "
                "name). To resolve: decrease `--gpu-memory-utilization` "
                "(e.g. `--gpu-memory-utilization 0.5`) "
                "or reduce CPU memory used by other processes."
            )

        super().__init__(
            vllm_config,
            local_rank,
            rank,
            distributed_init_method,
            is_driver_worker=is_driver_worker,
        )

        # CPU 后端禁用 custom all_reduce（使用标准实现）
        self.parallel_config.disable_custom_all_reduce = True

        # Torch Profiler：用于性能分析，通过 profiler_config 启用和配置
        self.profiler: Any | None = None
        profiler_config = vllm_config.profiler_config
        if profiler_config.profiler == "torch":
            worker_name = f"{vllm_config.instance_id}-rank-{self.rank}"
            self.profiler = TorchProfilerWrapper(
                profiler_config,
                worker_name=worker_name,
                local_rank=self.local_rank,
                activities=["CPU"],
            )

    def init_device(self):
        """
        初始化 CPU 设备。

        ==========================================================
        【初始化流程】
        ==========================================================
        1. 设置设备为 CPU
        2. 检查关键性能库是否已预加载（libtcmalloc、libiomp）
        3. 替换 torch.set_num_threads 为禁止操作（防止覆盖线程绑定）
        4. 初始化分布式环境
        5. 设置随机种子
        6. 创建 CPUModelRunner 实例
        """
        self.device = torch.device("cpu")

        # 检查关键性能库是否在 LD_PRELOAD 中
        def check_preloaded_libs(name: str):
            ld_preload_list = os.environ.get("LD_PRELOAD", "")
            if name not in ld_preload_list:
                logger.warning(
                    "%s is not found in LD_PRELOAD. "
                    "For best performance, please follow the section "
                    "`set LD_PRELOAD` in "
                    "https://docs.vllm.ai/en/latest/getting_started/installation/cpu/ "
                    "to setup required pre-loaded libraries.",
                    name,
                )

        if sys.platform.startswith("linux"):
            # libtcmalloc：高性能内存分配器，减少内存碎片和锁竞争
            check_preloaded_libs("libtcmalloc")
            if current_platform.get_cpu_architecture() == CpuArchEnum.X86:
                # libiomp：Intel OpenMP 运行时，优化多线程并行性能
                check_preloaded_libs("libiomp")

        # 禁止在线程绑定后再次调用 torch.set_num_threads
        # 因为这可能会破坏已建立的线程亲和性设置
        def skip_set_num_threads(x: int):
            logger.warning(
                "CPU backend doesn't allow to use "
                "`torch.set_num_threads` after the thread binding, skip it."
            )

        torch.set_num_threads = skip_set_num_threads

        # 设置分布式标识符，用于创建 allreduce 共享内存
        # Note: unique identifier for creating allreduce shared memory
        os.environ["VLLM_DIST_IDENT"] = self.distributed_init_method.split(":")[-1]
        # 初始化分布式环境（进程组等）
        init_worker_distributed_environment(
            self.vllm_config,
            self.rank,
            self.distributed_init_method,
            self.local_rank,
            current_platform.dist_backend,
        )
        # 设置随机种子以确保可复现性
        set_random_seed(self.model_config.seed)

        # 根据配置选择模型运行器版本
        if self.use_v2_model_runner:
            from vllm.v1.worker.cpu.model_runner import (
                CPUModelRunner as CPUModelRunnerV2,
            )

            self.model_runner: CPUModelRunner = CPUModelRunnerV2(  # type: ignore
                self.vllm_config, self.device
            )
        else:
            self.model_runner = CPUModelRunner(self.vllm_config, torch.device("cpu"))

    def sleep(self, level: int = 1) -> None:
        """
        CPU 后端不支持 sleep 模式。

        GPU 的 sleep 模式会释放显存以节省资源，但 CPU 内存管理机制不同，
        无法像 GPU 那样高效地释放和重新分配内存。
        """
        logger.warning("sleep mode is not supported on CPU, ignore it.")
        pass

    def wake_up(self, tags: list[str] | None = None) -> None:
        """
        CPU 后端不支持 wake_up 模式。

        与 sleep() 配对使用，CPU 后端无需此功能。
        """
        logger.warning("sleep mode is not supported on CPU, ignore it.")
        pass

    def determine_available_memory(self) -> int:
        """
        确定可用于 KV Cache 的 CPU 内存量。

        ==========================================================
        【内存计算策略】
        ==========================================================
        有两种模式：

        模式 1 - 显式指定（--kv-cache-memory-bytes）：
        - 直接使用用户指定的字节数
        - 验证不超过可用内存

        模式 2 - 自动计算：
        - KV 内存 = 请求的总 CPU 内存 - 当前进程已使用的内存
        - 请求的总 CPU 内存 = 总节点内存 * gpu_memory_utilization
        - 当前进程内存通过 psutil 获取 RSS（常驻内存集）

        返回：
            可用于 KV Cache 的字节数
        """
        # 先预热模型（触发编译），然后测量可用内存
        self.model_runner.warming_up_model()

        allowed_cpu_list = get_allowed_cpu_list()
        cpu_core = allowed_cpu_list[0]

        memory_status = get_memory_node_info(cpu_core.numa_node)
        available_memory = memory_status.available_memory
        explicit_kv_cache_size = self.cache_config.kv_cache_memory_bytes

        kv_cache_size = None
        msg = None
        if explicit_kv_cache_size is not None:
            # 模式 1：用户显式指定了 KV Cache 大小
            if explicit_kv_cache_size > available_memory:
                raise ValueError(
                    f"Available memory on node {cpu_core.numa_node} "
                    f"({format_gib(available_memory)}/"
                    f"{format_gib(memory_status.total_memory)} GiB) on kv cache"
                    f" allocation is less than requested memory for kv "
                    f"({format_gib(explicit_kv_cache_size)} GiB). "
                    "Decrease --kv-cache-memory-bytes, VLLM_CPU_KVCACHE_SPACE, "
                    "or reduce CPU memory used by other processes."
                )
            kv_cache_size = explicit_kv_cache_size
            msg = (
                f"Explicitly set ({format_gib(kv_cache_size)}/"
                f"{format_gib(memory_status.total_memory)}) GiB for KV cache "
                f"on node {cpu_core.numa_node}."
            )
        else:
            # 模式 2：自动计算 KV Cache 大小
            # 获取当前进程的 RSS 内存使用量
            consumed_memory = psutil.Process(os.getpid()).memory_info().rss
            requested_memory_for_kv = int(self.requested_cpu_memory - consumed_memory)
            if (
                requested_memory_for_kv <= 0
                or requested_memory_for_kv > available_memory
            ):
                raise ValueError(
                    f"Available memory on node {cpu_core.numa_node} "
                    f"({format_gib(available_memory)}/"
                    f"{format_gib(memory_status.total_memory)} GiB) on kv cache"
                    f" allocation is less than requested memory for kv "
                    f"({format_gib(requested_memory_for_kv)}/"
                    f"{format_gib(self.requested_cpu_memory)} GiB). "
                    "Reduce CPU memory used by other processes."
                )
            kv_cache_size = requested_memory_for_kv
            msg = (
                f"Auto set ({format_gib(kv_cache_size)}/"
                f"{format_gib(memory_status.total_memory)}) GiB for KV cache "
                f"on node {cpu_core.numa_node}, with "
                f"{format_gib(self.requested_cpu_memory)} GiB requested memory"
                f" for the worker. {format_gib(consumed_memory)} GiB"
                f" memory was consumed by non-kv usages."
            )

        logger.info(msg)

        return kv_cache_size

    def compile_or_warm_up_model(self) -> CompilationTimes:
        """
        编译或预热模型。

        注意：模型已经在 determine_available_memory() 中通过
        warming_up_model() 触发了编译。这里只对没有 KV Cache 的
        模型（如纯编码器模型）再次预热。

        重置随机种子以确保模型初始化和性能分析不会影响随机状态。

        返回：
            编译时间统计（语言模型和编码器的编译时间）
        """
        # Note: the model has been compiled in determine_available_memory(),
        # Only compile here for models without kv cache
        if len(self.model_runner.kv_caches) == 0:
            self.model_runner.warming_up_model()
        # 重置随机种子，确保模型初始化和性能分析不影响后续推理的随机状态
        # Reset the seed to ensure that the random state is not affected by
        # the model initialization and profiling.
        set_random_seed(self.model_config.seed)
        return CompilationTimes(
            language_model=self.compilation_config.compilation_time,
            encoder=self.compilation_config.encoder_compilation_time,
        )

    def profile(self, is_start: bool = True, profile_prefix: str | None = None):
        """
        启动或停止 Torch CPU Profiler。

        用于性能分析，可以收集 CPU 上的计算时间、内存分配等信息。
        分析结果可通过 TensorBoard 或 Chrome trace viewer 查看。

        参数：
            is_start: True 表示启动分析，False 表示停止分析
            profile_prefix: 分析文件的前缀名（可选）
        """
        if self.profiler is None:
            raise RuntimeError("Profiler is not enabled.")
        if is_start:
            self.profiler.start()
        else:
            self.profiler.stop()
