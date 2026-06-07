# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# =============================================================================
# parallel.py - 并行配置模块
# =============================================================================
# 本模块定义了 vLLM 分布式推理的核心并行配置，包括：
# 1. 张量并行 (Tensor Parallelism, TP) - 将模型层切分到多个 GPU
# 2. 流水线并行 (Pipeline Parallelism, PP) - 将模型按层切分到不同阶段
# 3. 数据并行 (Data Parallelism, DP) - 多个副本同时处理不同请求
# 4. 专家并行 (Expert Parallelism, EP) - MoE 模型的专家分布到不同设备
# 5. 上下文并行 (Context Parallelism, CP) - 长序列的 KV 缓存切分
#
# 并行层级关系：
#   world_size = TP × PP × PCP (预填充上下文并行)
#   world_size_across_dp = world_size × DP
#   总 GPU 数 = world_size_across_dp (在多节点场景下)
# =============================================================================

import os
import socket
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal, overload

import regex as re
import torch
from pydantic import Field, field_validator, model_validator
from torch.distributed import ProcessGroup, ReduceOp, Store
from typing_extensions import Self

import vllm.envs as envs
from vllm.config.utils import config
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.network_utils import get_open_ports_list

if TYPE_CHECKING:
    from ray.runtime_env import RuntimeEnv
    from ray.util.placement_group import PlacementGroup

    from vllm.v1.executor import Executor
else:
    RuntimeEnv = Any
    PlacementGroup = Any
    Executor = Any

logger = init_logger(__name__)
# NUMA CPU 集合的正则表达式，用于验证 numactl --physcpubind 参数格式
# 格式示例: "0-3" 或 "0,2,4-7" 或 "0-3,8-11"
_NUMACTL_CPUSET_PATTERN = re.compile(r"^\d+(?:-\d+)?(?:,\d+(?:-\d+)?)*$")

# =============================================================================
# 类型别名定义 - 用于限制配置字段的合法取值
# =============================================================================

# 专家放置策略：决定 MoE 专家如何分配到不同 rank
# - "linear": 连续放置，如 4 专家 2 rank → rank0:[0,1], rank1:[2,3]
# - "round_robin": 轮询放置，如 4 专家 2 rank → rank0:[0,2], rank1:[1,3]
ExpertPlacementStrategy = Literal["linear", "round_robin"]

# 分布式执行器后端：决定 worker 进程的管理方式
# - "ray": 使用 Ray 框架管理分布式 worker
# - "mp": 使用 Python multiprocessing 管理（单节点推荐）
# - "uni": 单进程模式，无分布式
# - "external_launcher": 外部启动器模式（如 Slurm、Kubernetes）
DistributedExecutorBackend = Literal["ray", "mp", "uni", "external_launcher"]

# 数据并行后端：决定数据并行的通信方式
# - "ray": 通过 Ray 进行数据并行协调
# - "mp": 通过 multiprocessing 进行数据并行协调
DataParallelBackend = Literal["ray", "mp"]

# EPLB 策略选项
EPLBPolicyOption = Literal["default"]

# 解码上下文并行 (DCP) 通信后端
# - "ag_rs": AllGather + ReduceScatter（默认，兼容现有行为）
# - "a2a": All-to-All 交换部分输出和 LSE，然后用 Triton kernel 合并
DCPCommBackend = Literal["ag_rs", "a2a"]

# EPLB 通信后端：专家权重迁移的通信方式
# - "torch_nccl": 使用 torch.distributed NCCL 后端（GPU 通信）
# - "torch_gloo": 使用 torch.distributed Gloo 后端（CPU 通信）
# - "nixl": 使用 NIXL/RIXL 进行 staged send/recv
# - "pynccl": 使用 PyNccl send/recv
EPLBCommunicatorBackend = Literal["torch_nccl", "torch_gloo", "nixl", "pynccl"]

# All-to-All 通信后端：MoE 专家并行的 token 分发/汇聚方式
# 这些后端决定了 token 如何在专家之间分发以及结果如何汇聚
All2AllBackend = Literal[
    "naive",           # 简单实现
    "pplx",            # PPLX 实现（已废弃）
    "deepep_high_throughput",  # DeepEP 高吞吐 kernel
    "deepep_low_latency",      # DeepEP 低延迟 kernel
    "mori_high_throughput",    # MoRI 高吞吐（多节点）
    "mori_low_latency",        # MoRI 低延迟（多节点）
    "nixl_ep",                 # NIXL-EP kernel
    "allgather_reducescatter", # 基于 AllGather + ReduceScatter 的实现（默认）
    "flashinfer_all2allv",     # FlashInfer 双边 kernel（临时别名）
    "flashinfer_nvlink_two_sided",   # FlashInfer NVLink 双边 kernel
    "flashinfer_nvlink_one_sided",   # FlashInfer NVLink 单边 kernel
]


# =============================================================================
# EPLBConfig - 专家并行负载均衡配置
# =============================================================================
# EPLB (Expert Parallel Load Balancing) 用于解决 MoE 模型中专家负载不均的问题。
#
# 工作原理：
# 1. 监控阶段：在滑动窗口内记录每个专家的 token 处理量
# 2. 决策阶段：根据负载统计决定是否需要重新排列专家
# 3. 迁移阶段：将过载专家的权重迁移到空闲设备
#
# 典型场景：某些热门专家（如通用知识专家）被频繁激活，导致其所在 GPU
# 成为瓶颈，EPLB 会将冗余副本或重新分配以均衡负载。
# =============================================================================

@config
class EPLBConfig:
    """Configuration for Expert Parallel Load Balancing (EP).
    专家并行负载均衡配置。"""

    # ① 滑动窗口大小：用于统计专家负载的时间窗口（单位：step）
    # 窗口越大，统计越平滑，但对负载变化的响应越慢
    window_size: int = Field(default=1000, gt=0)
    """Window size for expert load recording.
    专家负载记录的滑动窗口大小。"""

    # ② 专家重排列间隔（单位：step）
    # 每隔多少个 step 检查一次是否需要重新排列专家
    # 如果大于 window_size，则只使用最近 window_size 的指标
    step_interval: int = Field(default=3000, gt=0)
    """
    Interval for rearranging experts in expert parallelism.
    专家并行中重新排列专家的间隔步数。

    Note that if this is greater than the EPLB window size, only the metrics
    of the last `lb_window_size` steps will be used for rearranging experts.
    注意：如果此值大于 EPLB 窗口大小，只有最近 window_size 步的指标会被使用。
    """

    # ③ 冗余专家数量：在每个设备上额外放置的专家副本数
    # 冗余专家可以吸收突发流量，但会增加内存占用
    num_redundant_experts: int = Field(default=0, ge=0)
    """Number of redundant experts to use for expert parallelism.
    专家并行中使用的冗余专家数量。"""

    # ④ 是否记录负载均衡度日志
    # 默认关闭，因为这会引入额外的通信开销
    log_balancedness: bool = False
    """
    Log the balancedness each step of expert parallelism.
    记录专家并行每一步的负载均衡度。
    This is turned off by default since it will cause communication overhead.
    默认关闭，因为会导致通信开销。
    """
    # ⑤ 负载均衡度日志记录间隔
    log_balancedness_interval: int = Field(default=1, gt=0)
    """
    Interval for logging the balancedness.
    负载均衡度日志记录间隔。
    """
    # ⑥ 是否使用异步（非阻塞）EPLB
    # 异步 EPLB 在后台进行权重迁移，不阻塞推理
    use_async: bool = True
    """
    Whether to use non-blocking EPLB.
    是否使用非阻塞的 EPLB。
    """

    # ⑦ EPLB 策略类型
    policy: EPLBPolicyOption = "default"
    """The policy type for expert parallel load balancing (EPLB).
    专家并行负载均衡的策略类型。"""

    # ⑧ 专家权重通信后端
    # 决定专家权重在 GPU 之间迁移时使用的通信方式
    # - None: 自动选择（异步模式用 torch_gloo，同步模式用 torch_nccl）
    communicator: EPLBCommunicatorBackend | None = None
    """
    Backend for EPLB expert weight communication:
    EPLB 专家权重通信后端：
    - "torch_nccl": Use torch.distributed on the device process group
      使用 torch.distributed 设备进程组（GPU 通信）
    - "torch_gloo": Use torch.distributed gloo with CPU staging
      使用 torch.distributed Gloo 后端，通过 CPU 中转
    - "nixl": Use NIXL/ RIXL with staged send/recv buffers
      使用 NIXL/RIXL 进行 staged send/recv
    - "pynccl": Use PyNccl send/recv
      使用 PyNccl send/recv
    - None: Auto-select backend ("torch_gloo" for async, "torch_nccl" for sync)
      自动选择后端（异步用 torch_gloo，同步用 torch_nccl）
    """

    @model_validator(mode="after")
    def _validate_eplb_config(self) -> Self:
        """验证 EPLB 配置的合法性。"""
        # 异步 EPLB 目前只支持 default 策略
        if self.use_async and self.policy != "default":
            raise ValueError("Async EPLB is only supported with the default policy.")
        # 如果启用了负载均衡度日志，间隔必须大于 0
        if self.log_balancedness and self.log_balancedness_interval <= 0:
            raise ValueError("log_balancedness_interval must be greater than 0.")
        return self


# =============================================================================
# ParallelConfig - 核心并行配置类
# =============================================================================
# 这是 vLLM 分布式推理的核心配置类，管理所有并行维度的配置。
#
# 并行维度总览：
# ┌─────────────────────────────────────────────────────────────────┐
# │                    总 GPU 数 (world_size_across_dp)              │
# │  ┌──────────────────────────────────────────┐  ┌─────────────┐  │
# │  │         world_size (TP × PP × PCP)        │  │     DP      │  │
# │  │  ┌──────┐  ┌──────┐  ┌────────────────┐  │  │  数据并行   │  │
# │  │  │  TP  │  │  PP  │  │      PCP       │  │  │             │  │
# │  │  │张量  │  │流水线│  │ 预填充上下文   │  │  │             │  │
# │  │  │并行  │  │并行  │  │    并行        │  │  │             │  │
# │  │  └──────┘  └──────┘  └────────────────┘  │  └─────────────┘  │
# │  └──────────────────────────────────────────┘                   │
# └─────────────────────────────────────────────────────────────────┘
#
# MoE 模型额外支持：
#   - EP (Expert Parallelism): 专家并行，将不同专家分配到不同 GPU
#   - EPLB (Expert Parallel Load Balancing): 专家负载均衡
# =============================================================================

@config
class ParallelConfig:
    """Configuration for the distributed execution.
    分布式执行配置。"""

    # =========================================================================
    # 第一部分：核心并行维度参数
    # =========================================================================

    # ① 流水线并行大小 (PP)
    # 将模型按层切分为多个阶段，每个阶段在不同 GPU 上执行
    # 例如：PP=2，模型 32 层 → GPU0 处理 0-15 层，GPU1 处理 16-31 层
    pipeline_parallel_size: int = 1
    """Number of pipeline parallel groups.
    流水线并行组数量。"""

    # ② 张量并行大小 (TP)
    # 将每一层的参数（权重矩阵）切分到多个 GPU
    # 例如：TP=2，Linear 层的权重矩阵按列切分到 2 个 GPU
    tensor_parallel_size: int = 1
    """Number of tensor parallel groups.
    张量并行组数量。"""

    # ③ 预填充上下文并行大小 (PCP)
    # 在预填充阶段将长序列的 KV 缓存切分到多个 GPU
    # 用于处理超长上下文（如 128K+ tokens）
    prefill_context_parallel_size: int = 1
    """Number of prefill context parallel groups.
    预填充上下文并行组数量。"""

    # ④ 数据并行大小 (DP)
    # 模型的多个副本同时处理不同请求
    # MoE 层会根据 TP × DP 的乘积进行切分
    data_parallel_size: int = 1
    """Number of data parallel groups. MoE layers will be sharded according to
    the product of the tensor parallel size and data parallel size.
    数据并行组数量。MoE 层将根据张量并行大小和数据并行大小的乘积进行切分。"""

    # ⑤ 本地数据并行大小
    # 当前节点内的数据并行副本数（跨节点 DP 时使用）
    data_parallel_size_local: int = 1
    """Number of local data parallel groups.
    本地数据并行组数量。"""

    # ⑥ 数据并行 rank（全局）
    data_parallel_rank: int = 0
    """Rank of the data parallel group.
    数据并行组的 rank。"""

    # ⑦ 数据并行本地 rank（仅在 SPMD 模式下设置）
    data_parallel_rank_local: int | None = None
    """Local rank of the data parallel group, set only in SPMD mode.
    数据并行组的本地 rank，仅在 SPMD 模式下设置。"""

    # =========================================================================
    # 第二部分：数据并行网络配置
    # =========================================================================

    # ⑧ 数据并行主节点 IP
    data_parallel_master_ip: str = "127.0.0.1"
    """IP of the data parallel master.
    数据并行主节点的 IP 地址。"""

    # ⑨ 数据并行 RPC 端口
    data_parallel_rpc_port: int = 29550
    """Port for data parallel messaging.
    数据并行消息传递的 RPC 端口。"""

    # ⑩ 数据并行主节点端口
    data_parallel_master_port: int = 29500
    """Port of the data parallel master.
    数据并行主节点的端口。"""

    # ⑪ 数据并行后端
    data_parallel_backend: DataParallelBackend = "mp"
    """Backend to use for data parallel, either "mp" or "ray".
    数据并行使用的后端，可选 "mp"（multiprocessing）或 "ray"。"""

    # =========================================================================
    # 第三部分：数据并行负载均衡模式
    # =========================================================================

    # ⑫ 外部负载均衡模式
    # 适用于 Kubernetes 中 "一个 Pod 一个 rank" 的宽 EP 部署
    # 此模式下，vLLM 不做内部负载均衡，由外部 LB（如 K8s Service）负责
    data_parallel_external_lb: bool = False
    """Whether to use "external" DP LB mode. Applies only to online serving
    and when data_parallel_size > 0. This is useful for a "one-pod-per-rank"
    wide-EP setup in Kubernetes. Supported only for MoE deployments; non-MoE
    models should use independent vLLM instances without --data-parallel-*
    arguments. Set implicitly when --data-parallel-rank is provided explicitly
    to vllm serve.
    是否使用"外部"数据并行负载均衡模式。仅适用于在线服务且 data_parallel_size > 0 时。
    适用于 Kubernetes 中"一个 Pod 一个 rank"的宽 EP 部署。仅支持 MoE 模型。"""

    # ⑬ 混合负载均衡模式
    # 节点内 vLLM 负责本地 DP 负载均衡，外部 LB 负责跨节点均衡
    data_parallel_hybrid_lb: bool = False
    """Whether to use "hybrid" DP LB mode. Applies only to online serving
    and when data_parallel_size > 0. Enables running an AsyncLLM
    and API server on a "per-node" basis where vLLM load balances
    between local data parallel ranks, but an external LB balances
    between vLLM nodes/replicas. Set explicitly in conjunction with
    --data-parallel-start-rank.
    是否使用"混合"数据并行负载均衡模式。节点内 vLLM 做本地 DP 负载均衡，
    外部 LB 做跨节点/副本的负载均衡。"""

    # =========================================================================
    # 第四部分：MoE（混合专家）相关配置
    # =========================================================================

    # ⑭ 是否为 MoE 模型
    is_moe_model: bool | None = None
    """Whether the deployed model is MoE (if known).
    部署的模型是否为 MoE 模型（如果已知）。"""

    # ⑮ 是否启用专家并行 (EP)
    # 启用后，MoE 层的专家将分布在不同 GPU 上，而非复制到所有 GPU
    enable_expert_parallel: bool = False
    """Use expert parallelism instead of tensor parallelism for MoE layers.
    对 MoE 层使用专家并行而非张量并行。"""

    # ⑯ 是否启用专家权重过滤
    # 启用后，每个 rank 只加载自己负责的专家权重，减少磁盘 I/O
    enable_ep_weight_filter: bool = False
    """Skip non-local expert weights during model loading when expert
    parallelism is active. Each rank only reads its own expert shard from
    disk, which can drastically reduce storage I/O for MoE models with
    per-expert weight tensors (e.g. DeepSeek, Mixtral, Kimi-K2.5). Has no
    effect on 3D fused-expert checkpoints (e.g. GPT-OSS) or non-MoE models.
    当专家并行激活时，跳过非本地专家权重的加载。每个 rank 只从磁盘读取
    自己负责的专家分片，可以大幅减少 MoE 模型的存储 I/O。"""

    # ⑰ 是否启用专家并行负载均衡 (EPLB)
    enable_eplb: bool = False
    """Enable expert parallelism load balancing for MoE layers.
    启用 MoE 层的专家并行负载均衡。"""

    # ⑱ EPLB 配置对象
    eplb_config: EPLBConfig = Field(default_factory=EPLBConfig)
    """Expert parallelism configuration.
    专家并行负载均衡配置。"""

    # ⑲ 专家放置策略
    expert_placement_strategy: ExpertPlacementStrategy = "linear"
    """The expert placement strategy for MoE layers:
    MoE 层的专家放置策略：

    - "linear": Experts are placed in a contiguous manner.
      连续放置。例如 4 专家 2 rank → rank0:[0,1], rank1:[2,3]
    - "round_robin": Experts are placed in a round-robin manner.
      轮询放置。例如 4 专家 2 rank → rank0:[0,2], rank1:[1,3]
      This strategy can help improve load balancing for grouped expert
      models with no redundant experts.
      此策略有助于改善分组专家模型的负载均衡，无需冗余专家。"""

    # ⑳ All-to-All 通信后端
    # 决定 MoE 专家并行中 token 如何在专家之间分发和汇聚
    all2all_backend: All2AllBackend = "allgather_reducescatter"
    """All2All backend for MoE expert parallel communication. Available options:
    MoE 专家并行通信的 All2All 后端。可选项：

    - "allgather_reducescatter": All2all based on allgather and reducescatter
      基于 AllGather + ReduceScatter 的实现（默认）
    - "deepep_high_throughput": Use deepep high-throughput kernels
      使用 DeepEP 高吞吐 kernel
    - "deepep_low_latency": Use deepep low-latency kernels
      使用 DeepEP 低延迟 kernel
    - "mori_high_throughput": MoRI EP with InterNodeV1 for multi-node
      MoRI EP 高吞吐（多节点）
    - "mori_low_latency": MoRI EP with InterNodeV1LL for multi-node
      MoRI EP 低延迟（多节点）
    - "nixl_ep": Use nixl-ep kernels
      使用 NIXL-EP kernel
    - "flashinfer_nvlink_two_sided": Use flashinfer two-sided kernels for mnnvl
      使用 FlashInfer 双边 kernel（MNNVL）
    - "flashinfer_nvlink_one_sided": Use flashinfer high-throughput a2a kernels
      使用 FlashInfer 高吞吐 A2A kernel"""

    # =========================================================================
    # 第五部分：模型加载与执行器配置
    # =========================================================================

    # ㉑ 最大并行加载 worker 数
    # 用于避免大模型在张量并行加载时 OOM
    max_parallel_loading_workers: int | None = None
    """Maximum number of parallel loading workers when loading model
    sequentially in multiple batches. To avoid RAM OOM when using tensor
    parallel and large models.
    顺序加载模型时的最大并行加载 worker 数。用于避免使用张量并行和大模型时 RAM OOM。"""

    # ㉒ 是否禁用自定义 AllReduce kernel
    # 自定义 AllReduce 比 NCCL 更高效，但某些场景不支持
    disable_custom_all_reduce: bool = False
    """Disable the custom all-reduce kernel and fall back to NCCL.
    禁用自定义 AllReduce kernel，回退到 NCCL。"""

    # ㉓ 是否启用弹性专家并行
    # 允许在运行时动态调整 EP 的 GPU 数量
    enable_elastic_ep: bool = False
    """Enable elastic expert parallelism with stateless NCCL groups for DP/EP.
    启用弹性专家并行，使用无状态 NCCL 组用于 DP/EP。"""

    # =========================================================================
    # 第六部分：双批次重叠 (DBO) 配置
    # =========================================================================
    # DBO (Dual Batch Overlap) 通过将一个批次拆分为两个 micro-batch，
    # 使得计算和通信可以重叠执行，提高 GPU 利用率。

    # ㉔ 是否启用双批次重叠
    enable_dbo: bool = False
    """Enable dual batch overlap for the model executor.
    为模型执行器启用双批次重叠。"""

    # ㉕ micro-batch 大小
    ubatch_size: int = 0
    """Number of ubatch size.
    micro-batch 大小。"""

    # ㉖ 纯解码批次的 DBO 阈值
    # 如果请求的 token 数超过此阈值，使用 microbatching
    dbo_decode_token_threshold: int = 32
    """The threshold for dual batch overlap for batches only containing decodes.
    If the number of tokens in the request is greater than this threshold,
    microbatching will be used. Otherwise, the request will be processed in a
    single batch.
    纯解码批次的双批次重叠阈值。如果请求的 token 数超过此阈值，
    将使用 microbatching。否则，请求将在单个批次中处理。"""

    # ㉗ 包含预填充的批次的 DBO 阈值
    dbo_prefill_token_threshold: int = 512  # TODO(lucas): tune
    """The threshold for dual batch overlap for batches that contain one or more
    prefills. If the number of tokens in the request is greater than this
    threshold, microbatching will be used. Otherwise, the request will be
    processed in a single batch.
    包含预填充的批次的双批次重叠阈值。如果请求的 token 数超过此阈值，
    将使用 microbatching。否则，请求将在单个批次中处理。"""

    # =========================================================================
    # 第七部分：通信与同步配置
    # =========================================================================

    # ㉘ 是否禁用 NCCL 用于 DP 同步
    # 异步调度时默认使用 Gloo（CPU 通信），同步调度时使用 NCCL（GPU 通信）
    disable_nccl_for_dp_synchronization: bool | None = None
    """Forces the dp synchronization logic in vllm/v1/worker/dp_utils.py
    to use Gloo instead of NCCL for its all reduce.

    Defaults to True when async scheduling is enabled, False otherwise.
    强制 DP 同步逻辑使用 Gloo 而非 NCCL 进行 all reduce。
    异步调度启用时默认为 True，否则为 False。"""

    # =========================================================================
    # 第八部分：Ray 相关配置
    # =========================================================================

    # ㉙ 是否使用 Nsight profiling 分析 Ray worker
    ray_workers_use_nsight: bool = False
    """Whether to profile Ray workers with nsight.
    是否使用 Nsight 分析 Ray worker。"""

    # ㉚ Ray 运行时环境
    ray_runtime_env: RuntimeEnv | None = None
    """Ray runtime environment to pass to distributed workers.
    传递给分布式 worker 的 Ray 运行时环境。"""

    # ㉛ Ray placement group
    placement_group: PlacementGroup | None = None
    """ray distributed model workers placement group.
    Ray 分布式模型 worker 的 placement group。"""

    # =========================================================================
    # 第九部分：分布式执行器后端配置
    # =========================================================================

    # ㉜ 分布式执行器后端
    # 决定如何管理分布式 worker 进程
    distributed_executor_backend: (
        str | DistributedExecutorBackend | type[Executor] | None
    ) = None
    """
    Backend to use for distributed model workers, either "ray" or "mp"
    (multiprocessing). If the product of pipeline_parallel_size and tensor_parallel_size
    is less than or equal to the number of GPUs available, "mp" will be used to
    keep processing on a single host. Otherwise, an error will be raised. To use "mp"
    you must also set nnodes, and to use "ray" you must manually set
    distributed_executor_backend to "ray".

    分布式模型 worker 使用的后端，可选 "ray" 或 "mp"（multiprocessing）。
    如果 PP × TP ≤ 可用 GPU 数，将使用 "mp" 在单机上处理。
    否则会报错。使用 "mp" 需要设置 nnodes，使用 "ray" 需要手动设置。

    Note:
        TPU 平台只支持 Ray 进行分布式推理。
    """

    # =========================================================================
    # 第十部分：Worker 类配置
    # =========================================================================

    # ㉝ Worker 类名
    worker_cls: str = "auto"
    """The full name of the worker class to use. If "auto", the worker class
    will be determined based on the platform.
    使用的 worker 类全名。如果为 "auto"，将根据平台自动确定。"""

    # ㉞ 投机解码 worker 类名
    sd_worker_cls: str = "auto"
    """The full name of the worker class to use for speculative decoding.
    If "auto", the worker class will be determined based on the platform.
    用于投机解码的 worker 类全名。如果为 "auto"，将根据平台自动确定。"""

    # ㉟ Worker 扩展类名
    # 用于向 worker 类注入新属性和方法，供 collective_rpc 调用使用
    worker_extension_cls: str = ""
    """The full name of the worker extension class to use. The worker extension
    class is dynamically inherited by the worker class. This is used to inject
    new attributes and methods to the worker class for use in collective_rpc
    calls.
    使用的 worker 扩展类全名。worker 扩展类会动态继承 worker 类，
    用于向 worker 类注入新属性和方法，供 collective_rpc 调用使用。"""

    # =========================================================================
    # 第十一部分：多节点分布式配置
    # =========================================================================

    # ㊱ 主节点地址（多节点 mp 模式）
    master_addr: str = "127.0.0.1"
    """distributed master address for multi-node distributed
    inference when distributed_executor_backend is mp.
    多节点分布式推理的主节点地址（当 distributed_executor_backend 为 mp 时）。"""

    # ㊲ 主节点端口（多节点 mp 模式）
    master_port: int = 29501
    """distributed master port for multi-node distributed
    inference when distributed_executor_backend is mp.
    多节点分布式推理的主节点端口。"""

    # ㊳ 节点 rank（多节点 mp 模式）
    node_rank: int = 0
    """distributed node rank for multi-node distributed
    inference when distributed_executor_backend is mp.
    多节点分布式推理的节点 rank。"""

    # ㊴ 节点数量（多节点 mp 模式）
    nnodes: int = 1
    """num of nodes for multi-node distributed
    inference when distributed_executor_backend is mp.
    多节点分布式推理的节点数量。"""

    # =========================================================================
    # 第十二部分：NUMA 绑定配置
    # =========================================================================
    # NUMA (Non-Uniform Memory Access) 绑定可以将 GPU worker 进程绑定到
    # 特定的 NUMA 节点，减少跨 NUMA 节点的内存访问延迟。

    # ㊵ 是否启用 NUMA 绑定
    numa_bind: bool = False
    """Enable NUMA binding for GPU worker subprocesses.

    By default, workers are pinned to their GPU's NUMA-local CPUs and
    memory; on PCT-capable Xeons they also auto-bind to the SKU's
    PCT priority cores.

    启用 GPU worker 子进程的 NUMA 绑定。
    默认情况下，worker 会被绑定到其 GPU 的 NUMA 本地 CPU 和内存。
    """

    # �NUMA 节点绑定列表
    numa_bind_nodes: list[int] | None = None
    """NUMA node to bind each GPU worker to.

    Specify one NUMA node per visible GPU, for example `[0, 0, 1, 1]`
    for a 4-GPU system with GPUs 0-1 on NUMA node 0 and GPUs 2-3 on
    NUMA node 1. If unset and `numa_bind=True`, vLLM auto-detects the
    GPU-to-NUMA topology.

    绑定每个 GPU worker 到的 NUMA 节点。
    每个可见 GPU 指定一个 NUMA 节点，例如 `[0, 0, 1, 1]`。
    如果未设置且 numa_bind=True，vLLM 会自动检测 GPU 到 NUMA 的拓扑。"""

    # ㊷ CPU 列表绑定
    numa_bind_cpus: list[str] | None = None
    """Optional CPU lists to bind each GPU worker to.

    Specify one CPU list per visible GPU, for example
    `["0-3", "4-7", "8-11", "12-15"]`. When set, vLLM uses
    `numactl --physcpubind` instead of `--cpunodebind`.

    可选的 CPU 列表，用于绑定每个 GPU worker。
    每个可见 GPU 指定一个 CPU 列表，例如 `["0-3", "4-7", "8-11", "12-15"]`。
    设置后，vLLM 使用 `numactl --physcpubind` 而非 `--cpunodebind`。"""

    # =========================================================================
    # 第十三部分：分布式超时配置
    # =========================================================================

    # ㊸ 分布式操作超时时间（秒）
    distributed_timeout_seconds: int | None = None
    """Timeout in seconds for distributed operations (e.g., init_process_group).
    If set, this value is passed to torch.distributed.init_process_group as the
    timeout parameter. If None, PyTorch's default timeout is used (600s for NCCL).
    Increase this for multi-node setups where model downloads may be slow.

    分布式操作的超时时间（秒）。如果设置，此值会传递给
    torch.distributed.init_process_group 作为 timeout 参数。
    如果为 None，使用 PyTorch 默认超时（NCCL 为 600 秒）。"""

    # ㊹ CPU 通信组超时时间（秒）
    cpu_distributed_timeout_seconds: int | None = None
    """Timeout (in seconds) for cpu communication groups. If None, PyTorch's
    default timeout is used (1800s for gloo).

    CPU 通信组的超时时间（秒）。如果为 None，使用 PyTorch 默认超时（Gloo 为 1800 秒）。"""

    # =========================================================================
    # 第十四部分：运行时状态字段（非用户配置）
    # =========================================================================

    # ㊺ world_size = TP × PP × PCP
    # 决定需要创建多少个 worker
    world_size: int = Field(init=False)
    """world_size is TPxPP, it affects the number of workers we create.
    world_size = TP × PP，决定需要创建多少个 worker。"""

    # ㊻ 全局 rank
    rank: int = 0
    """Global rank in distributed setup.
    分布式设置中的全局 rank。"""

    # ㊼ 数据并行主节点端口列表（内部使用）
    _data_parallel_master_port_list: list[int] = Field(default_factory=list)
    """List of open port auto-queried for data parallel messaging.
    Set to be private as it's not intended to be configured by users.
    自动查询的用于数据并行消息传递的开放端口列表。
    设置为私有，因为不打算由用户配置。"""

    # ㊽ 协调 TCPStore 端口（内部使用）
    _coord_store_port: int = 0
    """Port of the coordination TCPStore. Can be set by the API server; workers
    connect as clients to exchange self-picked group ports at runtime.
    协调 TCPStore 的端口。可由 API 服务器设置；worker 作为客户端连接，
    在运行时交换自选的组端口。"""

    # =========================================================================
    # 第十五部分：解码上下文并行 (DCP) 配置
    # =========================================================================
    # DCP (Decode Context Parallel) 在解码阶段复用 TP 组的 GPU，
    # 将长序列的 KV 缓存切分到多个 GPU 上，降低单 GPU 的 KV 缓存压力。

    # ㊾ 解码上下文并行大小
    # DCP 不改变 world_size，而是复用 TP 组的 GPU
    # tp_size 必须能被 dcp_size 整除
    decode_context_parallel_size: int = 1
    """Number of decode context parallel groups, because the world size does
    not change by dcp, it simply reuse the GPUs of TP group, and tp_size
    needs to be divisible by dcp_size.
    解码上下文并行组数量。DCP 不改变 world_size，而是复用 TP 组的 GPU，
    tp_size 需要能被 dcp_size 整除。"""

    # ㊿ DCP KV 缓存交错大小（已废弃，使用 cp_kv_cache_interleave_size）
    dcp_kv_cache_interleave_size: int = 1
    """
    Interleave size of kv_cache storage while using DCP.
    使用 DCP 时 KV 缓存存储的交错大小。
    dcp_kv_cache_interleave_size has been replaced by cp_kv_cache_interleave_size,
    and will be deprecated when PCP is fully supported.
    已被 cp_kv_cache_interleave_size 替换，将在 PCP 完全支持后废弃。
    """

    # 51 DCP 通信后端
    dcp_comm_backend: DCPCommBackend = "ag_rs"
    """Communication backend for Decode Context Parallel (DCP).
    解码上下文并行 (DCP) 的通信后端：
    - "ag_rs": AllGather + ReduceScatter (default, existing behavior)
      AllGather + ReduceScatter（默认，现有行为）
    - "a2a": All-to-All exchange of partial outputs + LSE, then
      combine with Triton kernel. Reduces NCCL calls from 3 to 2
      per layer for MLA models.
      All-to-All 交换部分输出和 LSE，然后用 Triton kernel 合并。
      对 MLA 模型每层减少 NCCL 调用从 3 次到 2 次。"""

    # 52 上下文并行 KV 缓存交错大小
    # 决定 KV 缓存在 CP rank 之间的分布方式
    cp_kv_cache_interleave_size: int = 1
    """Interleave size of kv_cache storage while using DCP or PCP.
    使用 DCP 或 PCP 时 KV 缓存存储的交错大小。

    For `total_cp_rank = pcp_rank * dcp_world_size + dcp_rank`,
        and `total_cp_world_size = pcp_world_size * dcp_world_size`.
    store interleave_size tokens on total_cp_rank i,
    then store next interleave_size tokens on total_cp_rank i+1.

    Interleave_size=1: token-level alignment, where token `i` is stored on
        total_cp_rank `i % total_cp_world_size`.
    交错大小=1：token 级别对齐，token `i` 存储在 total_cp_rank `i % total_cp_world_size`。

    Interleave_size=block_size: block-level alignment, where tokens are
        first populated to the preceding ranks.
    交错大小=block_size：block 级别对齐，token 优先填充到前面的 rank。"""

    # 53 数据并行索引（不用于 torch 进程组）
    data_parallel_index: int = Field(init=False)
    """Equal to the data parallel rank but not used for torch process groups
    and not overridden for dense models.
    等于数据并行 rank，但不用于 torch 进程组，且对 dense 模型不会被覆盖。"""

    # 54 API 进程数量（内部配置，用于 API 服务器扩展）
    _api_process_count: int = Field(default=1, gt=0)
    """
    The number of API processes initialized.
    初始化的 API 进程数量。

    Note:
        This is an internal config that is only valid for and
        should only be set by API server scale-out.
        这是一个内部配置，仅对 API 服务器扩展有效，且应只由 API 服务器扩展设置。
    """

    # 55 API 进程 rank（内部配置）
    _api_process_rank: int = Field(default=0, ge=-1)
    """
    The rank of this API process, or `-1` for engine core processes
    under API server scale-out.
    此 API 进程的 rank，或 `-1` 表示 API 服务器扩展下的引擎核心进程。

    Note:
        This is an internal config that is only valid for and
        should only be set by API server scale-out.
        这是一个内部配置，仅对 API 服务器扩展有效，且应只由 API 服务器扩展设置。
    """

    # =========================================================================
    # 字段验证器 (Field Validators)
    # =========================================================================

    @field_validator("disable_nccl_for_dp_synchronization", mode="wrap")
    @classmethod
    def _skip_none_validation(cls, value: Any, handler: Callable) -> Any:
        """Skip validation if the value is `None` when initialisation is delayed.
        当初始化延迟时，如果值为 None 则跳过验证。"""
        return None if value is None else handler(value)

    @field_validator("numa_bind_nodes")
    @classmethod
    def _validate_numa_bind_nodes(cls, value: list[int] | None) -> list[int] | None:
        """验证 NUMA 节点绑定列表的合法性。
        - 不能为 None（已处理）
        - 不能为空列表
        - 节点编号必须为非负整数
        """
        if value is None:
            return None
        if not value:
            raise ValueError("numa_bind_nodes must not be empty.")
        if any(node < 0 for node in value):
            raise ValueError("numa_bind_nodes must contain non-negative integers.")
        return value

    @field_validator("numa_bind_cpus")
    @classmethod
    def _validate_numa_bind_cpus(cls, value: list[str] | None) -> list[str] | None:
        """验证 NUMA CPU 绑定列表的合法性。
        - 不能为 None（已处理）
        - 不能为空列表
        - 每个条目必须符合 numactl CPU 列表语法（如 "0-3" 或 "0,2,4-7"）
        - 范围必须是升序的
        """
        if value is None:
            return None
        if not value:
            raise ValueError("numa_bind_cpus must not be empty.")

        for cpuset in value:
            if not cpuset:
                raise ValueError("numa_bind_cpus entries must not be empty.")
            if not _NUMACTL_CPUSET_PATTERN.fullmatch(cpuset):
                raise ValueError(
                    "numa_bind_cpus entries must use numactl CPU list syntax, "
                    "for example '0-3' or '0,2,4-7'."
                )
            for part in cpuset.split(","):
                if "-" not in part:
                    continue
                start_str, end_str = part.split("-", 1)
                if int(start_str) > int(end_str):
                    raise ValueError(
                        f"numa_bind_cpus ranges must be ascending, but got '{cpuset}'."
                    )
        return value

    # =========================================================================
    # 模型验证器 (Model Validators)
    # =========================================================================

    @model_validator(mode="after")
    def _validate_parallel_config(self) -> Self:
        """验证并行配置的合法性。
        检查各并行参数之间的约束关系和平台兼容性。
        """
        # 验证 API 进程 rank 的合法性
        if self._api_process_rank >= self._api_process_count:
            raise ValueError(
                "Invalid value of `_api_process_rank`. "
                f"Expected to be `-1` or `[0, {self._api_process_count})`, "
                f"but found: {self._api_process_rank}"
            )

        # 检查已废弃的 all2all 后端，自动回退到默认实现
        if self.all2all_backend in ["pplx", "naive"]:
            logger.warning(
                "The '%s' all2all backend has been removed. "
                "Falling back to 'allgather_reducescatter'.",
                self.all2all_backend,
            )
            self.all2all_backend = "allgather_reducescatter"

        # 本地 DP 大小不能超过全局 DP 大小
        if self.data_parallel_size_local > self.data_parallel_size:
            raise ValueError(
                f"data_parallel_size_local ({self.data_parallel_size_local}) "
                f"must be <= data_parallel_size ({self.data_parallel_size})"
            )

        # 外部负载均衡模式需要 DP > 1
        if self.data_parallel_size <= 1 and self.data_parallel_external_lb:
            raise ValueError(
                "data_parallel_external_lb can only be set when data_parallel_size > 1"
            )

        # NUMA 绑定参数需要先启用 numa_bind
        if not self.numa_bind and (
            self.numa_bind_nodes is not None or self.numa_bind_cpus is not None
        ):
            raise ValueError(
                "numa_bind_nodes and numa_bind_cpus require numa_bind=True."
            )

        # EPLB 相关验证
        if self.enable_eplb:
            # EPLB 目前只支持 CUDA/ROCm 设备
            if not current_platform.is_cuda_alike():
                raise ValueError(
                    "Expert parallelism load balancing is only supported on "
                    "CUDA devices or ROCm devices now."
                )
            # EPLB 需要先启用专家并行
            if not self.enable_expert_parallel:
                raise ValueError("enable_expert_parallel must be True to use EPLB.")
            # EPLB 需要至少 TP > 1 或 DP > 1
            if self.tensor_parallel_size * self.data_parallel_size <= 1:
                raise ValueError(
                    "EPLB requires tensor_parallel_size or data_parallel_size "
                    f"to be greater than 1, but got "
                    f"TP={self.tensor_parallel_size},DP={self.data_parallel_size}."
                )
        else:
            # 未启用 EPLB 时，不能设置冗余专家
            if self.eplb_config.num_redundant_experts != 0:
                raise ValueError(
                    "num_redundant_experts is set to "
                    f"{self.eplb_config.num_redundant_experts} but EPLB is not "
                    "enabled. Either enable EPLB or unset "
                    "num_redundant_experts."
                )

        # DCP 验证：tp_size 必须能被 dcp_size 整除
        # 因为 DCP 不改变 world_size，而是将一个 TP 组拆分为多个 DCP 组
        # 例如：TP=8, DCP=4 → 2 个 DCP 组，每组 4 个 GPU
        if self.tensor_parallel_size % self.decode_context_parallel_size != 0:
            raise ValueError(
                f"tp_size={self.tensor_parallel_size} must be divisible by"
                f"dcp_size={self.decode_context_parallel_size}."
            )

        # a2a 通信后端需要 DCP > 1
        if self.dcp_comm_backend == "a2a" and self.decode_context_parallel_size <= 1:
            raise ValueError(
                "dcp_comm_backend='a2a' requires decode_context_parallel_size > 1."
            )

        return self

    # =========================================================================
    # 属性 (Properties)
    # =========================================================================

    @property
    def world_size_across_dp(self) -> int:
        """world_size_across_dp is TPxPPxDP, it is the size of the world
        including data parallelism.
        包含数据并行的世界大小 = TP × PP × PCP × DP。
        这是整个分布式系统的总 GPU 数。"""
        return self.world_size * self.data_parallel_size

    @property
    def use_ubatching(self) -> bool:
        """是否使用 micro-batching。
        当启用 DBO 或 ubatch_size > 1 时使用。"""
        return self.enable_dbo or self.ubatch_size > 1

    @property
    def num_ubatches(self) -> int:
        """micro-batch 的数量。
        DBO 模式下固定为 2，否则为 ubatch_size。"""
        return 2 if self.enable_dbo else self.ubatch_size

    @property
    def local_engines_only(self) -> bool:
        """
        Client manages local+remote EngineCores in pure internal LB case.
        Client manages local EngineCores in hybrid and external LB case.

        客户端是否只管理本地 EngineCore。
        在纯内部 LB 模式下，客户端管理本地+远程 EngineCore。
        在混合和外部 LB 模式下，客户端只管理本地 EngineCore。
        """
        return self.data_parallel_external_lb or self.data_parallel_hybrid_lb

    # =========================================================================
    # 端口管理方法
    # =========================================================================

    def get_next_dp_init_port(self) -> int:
        """
        We might need to initialize process groups in multiple
        processes that is related to data parallelism,
        e.g. both in the worker and in the engine, which
        can live in different processes. To avoid port conflicts, we
        pop a new port from the prepared port list each time we need to
        initialize a new process group related to data parallelism.

        获取下一个数据并行初始化端口。
        我们可能需要在多个进程中初始化与数据并行相关的进程组
        （例如 worker 和 engine 中都需要），它们可能在不同的进程中。
        为了避免端口冲突，每次需要初始化新的数据并行进程组时，
        从准备好的端口列表中弹出一个新端口。
        """
        if self._data_parallel_master_port_list:
            answer = self._data_parallel_master_port_list.pop()
        else:
            answer = self.data_parallel_master_port
            self.data_parallel_master_port += 1

        return answer

    def _pick_stateless_dp_port(self) -> tuple[int, socket.socket | None]:
        """Return ``(port, listen_socket)`` for DP group init.

        With a coord store, rank 0 binds a socket and publishes the port;
        others read it.  Without one, pops a pre-allocated port and
        returns ``listen_socket=None``.

        为 DP 组初始化选择端口并返回 (port, listen_socket)。

        工作流程：
        1. 如果没有协调存储（_coord_store_port=0）：
           - 直接从预分配端口列表中获取端口
           - 返回 listen_socket=None
        2. 如果有协调存储：
           - rank 0: 绑定一个随机端口，将端口号发布到协调存储
           - 其他 rank: 从协调存储中读取 rank 0 发布的端口号
        """
        if not self._coord_store_port:
            return self.get_next_dp_init_port(), None

        from vllm.distributed.utils import get_cached_tcp_store_client

        store = get_cached_tcp_store_client(
            self.data_parallel_master_ip, self._coord_store_port
        )

        key = "dp_master_port"
        if self.data_parallel_rank == 0:
            # rank 0 负责选择端口并绑定监听
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind((self.data_parallel_master_ip, 0))
            s.listen()
            port = s.getsockname()[1]
            store.set(key, str(port).encode())
            return port, s
        else:
            # 其他 rank 从协调存储中读取端口号
            return int(store.get(key).decode()), None

    @overload
    def stateless_init_dp_group(
        self, return_store: Literal[False] = ...
    ) -> ProcessGroup: ...
    @overload
    def stateless_init_dp_group(
        self, return_store: Literal[True] = ...
    ) -> tuple[ProcessGroup, Store]: ...
    def stateless_init_dp_group(
        self, return_store: bool = False
    ) -> ProcessGroup | tuple[ProcessGroup, Store]:
        """无状态初始化数据并行进程组。

        无状态初始化意味着不依赖 torch.distributed 的全局状态，
        每个进程独立加入组，适用于弹性 EP 和动态扩缩容场景。

        工作流程：
        1. 选择一个可用端口（通过 _pick_stateless_dp_port）
        2. 使用 Gloo 后端初始化进程组（因为 engine 进程可能没有 CUDA 设备）
        3. 如果遇到端口冲突（EADDRINUSE），最多重试 5 次

        为什么使用 Gloo 而非 NCCL：
        - engine 进程可能没有 CUDA 设备
        - Gloo 支持 CPU 通信，更适合跨节点协调
        - 这里只需要简单的元数据交换，不需要 GPU 通信性能
        """
        # NOTE: In high-concurrency scenarios multiple processes
        # can pick the same (currently free) port through a race
        # condition when calling `get_open_port()`. When the first
        # process binds the port the others will subsequently fail
        # with `torch.distributed.DistNetworkError: EADDRINUSE`.
        # To make the initialization more robust we retry a few times
        # with a fresh port whenever this specific error is observed.
        from torch.distributed import DistNetworkError

        from vllm.distributed.utils import (
            stateless_init_torch_distributed_process_group,
        )

        max_retries = 5
        last_exc: Exception | None = None
        for _ in range(max_retries):
            try:
                port, listen_socket = self._pick_stateless_dp_port()
                # use gloo since the engine process might not have cuda device
                return stateless_init_torch_distributed_process_group(
                    self.data_parallel_master_ip,
                    port,
                    self.data_parallel_rank,
                    self.data_parallel_size,
                    backend="gloo",
                    return_store=return_store,
                    listen_socket=listen_socket,
                )
            except DistNetworkError as e:
                # We only want to retry when the root cause is EADDRINUSE.
                if "EADDRINUSE" in str(e):
                    logger.warning("Address already in use. Retrying with a new port.")
                    last_exc = e
                    continue  # try again with a new port
                raise e

        # If we get here all retries have failed.
        assert last_exc is not None
        raise last_exc

    # =========================================================================
    # MoE 相关属性
    # =========================================================================

    # The all_reduce at the end of attention (during o_proj) means that
    # inputs are replicated across each rank of the tensor parallel group.
    # If using expert-parallelism with DeepEP All2All ops, replicated
    # tokens results in useless duplicate computation and communication.
    #
    # In this case, ensure the input to the experts is sequence parallel
    # to avoid the excess work.
    #
    # 背景说明：
    # 注意力层的 o_proj 末尾有 all_reduce 操作，这意味着输入在 TP 组的每个 rank 上是复制的。
    # 如果使用 DeepEP All2All 操作进行专家并行，复制的 token 会导致无用的重复计算和通信。
    # 因此，在这种情况下，需要确保专家的输入是序列并行的，以避免多余的工作。
    @property
    def use_sequence_parallel_moe(self) -> bool:
        """是否使用序列并行的 MoE。
        当满足以下条件时返回 True：
        1. 使用支持序列并行的 all2all 后端
        2. 启用了专家并行
        3. TP > 1（需要在 TP 组内分发 token）
        4. DP > 1（需要在 DP 组间分发 token）
        """
        return (
            self.all2all_backend
            in (
                "allgather_reducescatter",
                "deepep_high_throughput",
                "deepep_low_latency",
                "mori_high_throughput",
                "mori_low_latency",
                "nixl_ep",
            )
            and self.enable_expert_parallel
            and self.tensor_parallel_size > 1
            and self.data_parallel_size > 1
        )

    @property
    def use_batched_dp_moe(self) -> bool:
        """是否使用批量 DP MoE。
        当使用低延迟 all2all 后端且 DP > 1 时返回 True。
        批量 DP MoE 可以将多个 DP rank 的 token 打包在一起进行 all2all 通信，
        减少通信次数，提高效率。
        """
        return (
            self.all2all_backend
            in (
                "deepep_low_latency",
                "nixl_ep",
            )
            and self.enable_expert_parallel
            and self.data_parallel_size > 1
        )

    # =========================================================================
    # 节点拓扑相关属性
    # =========================================================================

    @property
    def node_rank_within_dp(self) -> int:
        """当前节点在其 DP 组内的节点 rank。
        例如：4 节点，每节点 2 DP rank → node_rank_within_dp = node_rank % 2"""
        return self.node_rank % self.nnodes_within_dp

    @property
    def nnodes_within_dp(self) -> int:
        """每个 DP 组占用的节点数。
        计算公式：总节点数 / DP 组数（按节点划分）"""
        if self.nnodes == 1:
            return 1
        data_parallel_node_size = (
            self.data_parallel_size // self.data_parallel_size_local
        )
        return self.nnodes // data_parallel_node_size

    @property
    def local_world_size(self) -> int:
        """当前节点内的 world_size（GPU 数）。
        计算公式：总 world_size / 每个 DP 组的节点数"""
        return self.world_size // self.nnodes_within_dp

    # =========================================================================
    # 静态方法：DP 组同步操作
    # =========================================================================

    @staticmethod
    def has_unfinished_dp(dp_group: ProcessGroup, has_unfinished: bool) -> bool:
        """检查 DP 组中是否有任何 rank 有未完成的工作。

        使用 MAX all_reduce 实现逻辑 OR 操作：
        - 每个 rank 将自己的 has_unfinished 状态（0 或 1）放入 tensor
        - MAX all_reduce 后，只要有任意 rank 有未完成工作，结果就为 1

        示例：
        - rank 0: has_unfinished=True (1)
        - rank 1: has_unfinished=False (0)
        - MAX 结果: 1 → True（有未完成的工作）
        """
        tensor = torch.tensor([has_unfinished], dtype=torch.int32, device="cpu")
        # dp rank 0: has_unfinished_seqs=True
        # dp rank 1: has_unfinished_seqs=False
        # aggregated: has_unfinished_seqs=True
        # so this is an OR operation, i.e. MAX in integers
        torch.distributed.all_reduce(tensor, op=ReduceOp.MAX, group=dp_group)
        aggregated_has_unfinished = bool(tensor.item())
        return aggregated_has_unfinished

    @staticmethod
    def sync_dp_state(
        dp_group: ProcessGroup, has_unfinished: bool, pending_pause: bool
    ) -> tuple[bool, bool]:
        """Combined all-reduce for DP state synchronization.
        DP 状态同步的组合 all_reduce。

        Uses a single SUM all-reduce on a 2-element tensor:
        使用单次 SUM all_reduce 操作 2 元素 tensor：
          [0] = 1 if this rank has unfinished work, else 0.
                [0] = 1 表示此 rank 有未完成的工作，否则为 0。
                SUM > 0 ≡ logical OR across ranks → any rank has work.
                SUM > 0 等价于跨 rank 的逻辑 OR → 任意 rank 有工作。
          [1] = 1 if this rank has a pending pause request, else 0.
                [1] = 1 表示此 rank 有待处理的暂停请求，否则为 0。
                SUM == dp_size ≡ all ranks reached pause consensus.
                SUM == dp_size 等价于所有 rank 达成暂停共识。

        has_unfinished_global is true if any rank has unfinished work,
        or if some ranks are waiting for a pause consensus.
        has_unfinished_global 为 true 当任意 rank 有未完成的工作，
        或者某些 rank 正在等待暂停共识。

        Returns:
            (has_unfinished_global, pause_consensus)
            (全局是否有未完成工作, 是否达成暂停共识)
        """
        tensor = torch.tensor(
            [int(has_unfinished), int(pending_pause)], dtype=torch.int32, device="cpu"
        )
        torch.distributed.all_reduce(tensor, op=ReduceOp.SUM, group=dp_group)
        dp_size = dp_group.size()
        pause_count = tensor[1].item()
        # has_unfinished_global = 任意 rank 有工作 OR 暂停请求未完全同步
        has_unfinished_global = tensor[0].item() > 0 or pause_count % dp_size != 0
        return has_unfinished_global, pause_count == dp_size

    @staticmethod
    def sync_kv_cache_memory_size(dp_group: ProcessGroup, kv_cache_memory: int) -> int:
        """同步 DP 组中所有 rank 的 KV 缓存内存大小，取最小值。

        工作流程：
        1. 将 KV 缓存内存大小放入 tensor（-1 替换为 int64 最大值）
        2. 使用 MIN all_reduce 获取所有 rank 中的最小值
        3. 返回最小值作为统一的 KV 缓存内存大小

        为什么取最小值：
        - 确保所有 DP rank 使用相同的 KV 缓存大小
        - 取最小值可以避免某些 rank 因内存不足而 OOM
        - -1 表示"使用所有可用内存"，替换为最大值以参与 MIN 比较

        为什么不能用 broadcast：
        - 无状态 DP 组依赖全局 rank，broadcast 可能不正确
        """
        if kv_cache_memory == -1:
            kv_cache_memory = torch.iinfo(torch.int64).max
        tensor = torch.tensor([kv_cache_memory], dtype=torch.int64, device="cpu")
        # we cannot use broadcast for stateless dp group since it depends
        # on global rank
        torch.distributed.all_reduce(tensor, op=ReduceOp.MIN, group=dp_group)
        return tensor.item()

    def compute_hash(self):
        """
        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.

        This hash is also used for DP worker configuration validation
        to prevent hangs from mismatched collective communication patterns.

        计算配置哈希值，用于唯一标识影响计算图结构的所有配置。

        哈希范围：从 input ids/embeddings 到 final hidden states 的计算图结构。
        排除项：input ids/embeddings 之前和 final hidden states 之后的部分。

        此哈希也用于 DP worker 配置验证，防止因集合通信模式不匹配导致的挂起。

        忽略的因素包括：
        1. 派生/运行时拓扑、网络或启动细节（如 rank、端口、节点信息）
        2. 不影响计算图结构的执行器配置（如 Ray 配置、worker 类名）
        3. NUMA 绑定配置（只影响主机端内存局部性，不影响集合通信语义）
        """
        ignored_factors = {
            # Derived/runtime topology, networking, or launch details
            # 派生/运行时拓扑、网络或启动细节
            "data_parallel_rank",
            "data_parallel_rank_local",
            "data_parallel_size_local",
            "data_parallel_index",
            "data_parallel_backend",
            "data_parallel_external_lb",
            "data_parallel_hybrid_lb",
            "data_parallel_master_ip",
            "data_parallel_master_port",
            "_data_parallel_master_port_list",
            "data_parallel_rpc_port",
            "rank",
            "master_addr",
            "master_port",
            "node_rank",
            "nnodes",
            "max_parallel_loading_workers",
            "disable_custom_all_reduce",
            "ray_workers_use_nsight",
            "ray_runtime_env",
            "placement_group",
            "distributed_executor_backend",
            "worker_cls",
            "sd_worker_cls",
            "worker_extension_cls",
            "_api_process_count",
            "_api_process_rank",
            # NUMA binding is per-rank host-side memory locality; it does
            # not affect collective-communication semantics. When numa_bind
            # is enabled with auto-detection, each DP rank stores its own
            # NUMA node in numa_bind_nodes (see vllm/utils/numa_utils.py
            # `_get_numa_node`), which would otherwise diverge the DP hash.
            # NUMA 绑定是 per-rank 的主机端内存局部性配置；
            # 它不影响集合通信语义。
            # 当启用 numa_bind 自动检测时，每个 DP rank 在 numa_bind_nodes
            # 中存储自己的 NUMA 节点，这会导致 DP 哈希不一致。
            "numa_bind",
            "numa_bind_nodes",
            "numa_bind_cpus",
        }

        from vllm.config.utils import get_hash_factors, hash_factors

        factors = get_hash_factors(self, ignored_factors)
        return hash_factors(factors)

    def __post_init__(self) -> None:
        """初始化后处理：计算 world_size、验证参数、选择分布式后端。

        这是 ParallelConfig 初始化的核心方法，执行以下步骤：
        1. 计算 world_size = TP × PP × PCP
        2. 验证弹性 EP 配置
        3. 初始化数据并行相关参数
        4. 自动选择分布式执行器后端
        5. 配置 EPLB 通信后端
        """
        # Continue with the rest of the initialization
        # 步骤 1：计算 world_size = TP × PP × PCP
        self.world_size = (
            self.pipeline_parallel_size
            * self.tensor_parallel_size
            * self.prefill_context_parallel_size
        )

        # 外部启动器模式下，world_size 需要乘以 DP
        if self.distributed_executor_backend == "external_launcher":
            logger.info("Using external launcher for distributed inference.")
            self.world_size *= self.data_parallel_size

        # 步骤 2：验证弹性 EP 配置
        if self.enable_elastic_ep:
            if not self.enable_eplb:
                raise ValueError("Elastic EP is only supported with enable_eplb=True.")
            if self.pipeline_parallel_size > 1:
                raise ValueError(
                    "Elastic EP is not supported with pipeline parallelism "
                    f"(pipeline_parallel_size={self.pipeline_parallel_size})."
                )
            if self.data_parallel_external_lb or self.data_parallel_hybrid_lb:
                raise NotImplementedError(
                    "Elastic EP is not compatible with data_parallel_external_lb "
                    "or data_parallel_hybrid_lb. Elastic EP relies on a single API "
                    "server and core client to coordinate scale up/down."
                )

        # 步骤 3：初始化数据并行相关参数
        if self.data_parallel_size > 1 or self.data_parallel_size_local == 0:
            # Data parallel was specified in the engine args.
            # 数据并行已在引擎参数中指定
            if self.distributed_executor_backend == "external_launcher":
                # For external launcher,
                # we need to set the data parallel rank automatically
                # 外部启动器模式下，自动设置数据并行 rank
                self.data_parallel_rank = int(os.environ["RANK"]) // (
                    self.world_size // self.data_parallel_size
                )
                logger.info(
                    "Set data_parallel_rank to %d automatically.",
                    self.data_parallel_rank,
                )
            if not self.enable_elastic_ep:
                # 非弹性 EP 模式下，预分配端口列表
                if not self._data_parallel_master_port_list:
                    self._data_parallel_master_port_list = get_open_ports_list(5)
                self.data_parallel_master_port = (
                    self._data_parallel_master_port_list.pop()
                )

            # 验证 DP rank 范围
            if not (0 <= self.data_parallel_rank < self.data_parallel_size):
                raise ValueError(
                    f"data_parallel_rank ({self.data_parallel_rank})"
                    f" must be in the range [0, {self.data_parallel_size})"
                )
        else:
            # Otherwise fall back to env vars (e.g. for offline SPMD case).
            # 否则回退到环境变量（例如离线 SPMD 场景）
            self.data_parallel_size = envs.VLLM_DP_SIZE
            self.data_parallel_rank = envs.VLLM_DP_RANK
            self.data_parallel_rank_local = envs.VLLM_DP_RANK_LOCAL
            self.data_parallel_master_ip = envs.VLLM_DP_MASTER_IP
            self.data_parallel_master_port = envs.VLLM_DP_MASTER_PORT

            # 离线 DP 模式只支持 MoE 模型
            if self.data_parallel_size > 1 and self.is_moe_model is False:
                raise ValueError(
                    "Offline data parallel mode is not supported/useful"
                    " for dense models."
                )

        self.data_parallel_index = self.data_parallel_rank

        # 外部启动器模式下禁用 V1 多进程
        if self.distributed_executor_backend == "external_launcher":
            os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
            logger.info("Disabling V1 multiprocessing for external launcher.")

        # 步骤 4：自动选择分布式执行器后端
        if self.distributed_executor_backend is None and self.world_size_across_dp > 1:
            # We use multiprocessing by default if world_size fits on the
            # current node and we aren't in a ray placement group.
            # 如果 world_size 适合当前节点且不在 Ray placement group 中，
            # 默认使用 multiprocessing。

            from vllm.v1.executor import ray_utils

            backend: DistributedExecutorBackend = "mp"
            ray_found = ray_utils.ray_is_available()

            # 后端选择逻辑（按优先级）：
            # 1. TPU + SPMD → uni
            # 2. CUDA 多节点 → mp
            # 3. CUDA 单节点但 GPU 不够 → 报错
            # 4. DP 后端为 ray → ray
            # 5. Ray 可用且在 placement group 中 → ray
            # 6. 默认 → mp
            if current_platform.is_tpu() and envs.VLLM_XLA_USE_SPMD:
                backend = "uni"
            elif current_platform.is_cuda() and self.nnodes > 1:
                backend = "mp"
            elif (
                current_platform.is_cuda()
                and current_platform.device_count() < self.world_size
            ):
                gpu_count = current_platform.device_count()
                raise ValueError(
                    f"World size ({self.world_size}) is larger than the number of "
                    f"available GPUs ({gpu_count}) in this node. If this is "
                    "intentional and you are using:\n"
                    "- ray, set '--distributed-executor-backend ray'.\n"
                    "- multiprocessing, set '--nnodes' appropriately."
                )
            elif self.data_parallel_backend == "ray":
                logger.info(
                    "Using ray distributed inference because "
                    "data_parallel_backend is ray"
                )
                backend = "ray"
            elif ray_found:
                if self.placement_group:
                    backend = "ray"
                else:
                    from ray import is_initialized as ray_is_initialized

                    if ray_is_initialized():
                        from ray.util import get_current_placement_group

                        if get_current_placement_group():
                            backend = "ray"
            self.distributed_executor_backend = backend
            logger.debug("Defaulting to use %s for distributed inference", backend)

        # world_size=1 时使用 uni（单进程）模式
        if self.distributed_executor_backend is None and self.world_size == 1:
            self.distributed_executor_backend = "uni"

        # max_parallel_loading_workers 暂不支持
        if self.max_parallel_loading_workers is not None:
            logger.warning(
                "max_parallel_loading_workers is currently "
                "not supported and will be ignored."
            )
        # 多节点只支持 mp、uni、external_launcher 后端
        allowed_backends = ("mp", "uni", "external_launcher")
        if (
            self.distributed_executor_backend not in allowed_backends
            and self.nnodes > 1
        ):
            raise ValueError(
                "nnodes > 1 can only be set when distributed executor "
                "backend is mp, uni or external_launcher."
            )

        # 步骤 5：配置 EPLB 通信后端
        if self.enable_eplb and self.eplb_config.communicator is None:
            if self.enable_elastic_ep:
                # Elastic EP requires stateless mode
                # (torch.distributed.batch_isend_irecv doesn't
                # support stateless mode), so we use PyNCCL backend
                # 弹性 EP 需要无状态模式，使用 PyNCCL 后端
                self.eplb_config.communicator = "pynccl"
            else:
                # Avoid torch_nccl: NCCL is fundamentally incompatible
                # with async EPLB due to multi-stream conflicts, and
                # batched isend/irecv hangs under high load.
                # See https://github.com/pytorch/pytorch/issues/174288
                # Prefer nixl when available; fall back to torch_gloo.
                # 避免使用 torch_nccl：NCCL 与异步 EPLB 不兼容
                # （多流冲突），batched isend/irecv 在高负载下会挂起。
                # 优先使用 nixl（如果可用），否则回退到 torch_gloo。
                from vllm.distributed.nixl_utils import is_nixl_available

                if is_nixl_available():
                    self.eplb_config.communicator = "nixl"
                else:
                    self.eplb_config.communicator = "torch_gloo"

    # =========================================================================
    # Ray 相关属性
    # =========================================================================

    @property
    def use_ray(self) -> bool:
        """是否使用 Ray 作为分布式执行器。
        当分布式执行器后端为 "ray" 或自定义 Executor 类使用 Ray 时返回 True。"""
        return self.distributed_executor_backend == "ray" or (
            isinstance(self.distributed_executor_backend, type)
            and getattr(self.distributed_executor_backend, "uses_ray", False)
        )

    @model_validator(mode="after")
    def _verify_args(self) -> Self:
        """验证参数并进行最终配置调整。

        执行以下验证和调整：
        1. 验证分布式执行器后端的合法性
        2. 验证 Ray 可用性（如果使用 Ray）
        3. 根据平台和配置禁用自定义 AllReduce
        4. 验证 Nsight profiling 的使用条件
        """
        # Lazy import to avoid circular import
        from vllm.v1.executor import Executor

        # Enable batch invariance settings if requested
        # 如果请求了批次不变性设置，禁用自定义 AllReduce
        if envs.VLLM_BATCH_INVARIANT:
            self.disable_custom_all_reduce = True

        # 验证分布式执行器后端类型
        if (
            self.distributed_executor_backend is not None
            and not isinstance(self.distributed_executor_backend, str)
            and not (
                isinstance(self.distributed_executor_backend, type)
                and issubclass(self.distributed_executor_backend, Executor)
            )
        ):
            raise ValueError(
                "Unrecognized distributed executor backend "
                f"{self.distributed_executor_backend}. Supported "
                "values are 'ray', 'mp' 'uni', 'external_launcher', "
                " custom Executor subclass or its import path."
            )
        # 如果使用 Ray，验证 Ray 可用性
        if self.use_ray:
            from vllm.v1.executor import ray_utils

            ray_utils.assert_ray_available()

        # 禁用自定义 AllReduce 的条件：
        # 1. 当前平台不支持自定义 AllReduce
        # 2. 多节点部署（自定义 AllReduce 只支持单节点）
        if not current_platform.use_custom_allreduce():
            self.disable_custom_all_reduce = True
            logger.debug(
                "Disabled the custom all-reduce kernel because it is not "
                "supported on current platform."
            )
        if self.nnodes > 1:
            self.disable_custom_all_reduce = True
            logger.debug(
                "Disabled the custom all-reduce since we are running on multi-node."
            )
        # Nsight profiling 只能在 Ray worker 上使用
        if self.ray_workers_use_nsight and not self.use_ray:
            raise ValueError(
                "Unable to use nsight profiling unless workers run with Ray."
            )

        return self
