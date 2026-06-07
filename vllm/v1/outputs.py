# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
输出数据结构模块 (Output Data Structures)
==========================================

本模块定义了 vLLM v1 引擎中模型运行器 (ModelRunner) 产生输出后，
在各个进程之间传递所需的所有数据结构。

主要数据流：
    1. 模型前向推理完成后，GPUModelRunner 生成 SamplerOutput
       (包含采样得到的 token IDs 和 logprobs 张量)。
    2. SamplerOutput 被转换为 ModelRunnerOutput (使用 Python list
       而非 torch.Tensor，以降低序列化开销)。
    3. ModelRunnerOutput 通过 IPC (进程间通信) 发送给调度器进程。
    4. 调度器据此更新请求状态，返回新 token 给用户。

核心类说明：
    - LogprobsLists / LogprobsTensors: 存储 log 概率数据的两种形式
      (NumPy 列表 vs PyTorch 张量)，用于推理过程和 IPC 传输。
    - RoutedExpertsTensors / RoutedExpertsLists: MoE 模型的路由专家
      信息，记录每个 token 被分配到哪些专家。
    - SamplerOutput: 模型前向推理的直接输出，包含采样结果。
    - ModelRunnerOutput: 经过转换后用于 IPC 传输的输出数据。
    - AsyncModelRunnerOutput: 异步调度场景下的输出包装器。
    - KVConnectorOutput: KV 缓存传输连接器的输出状态。
    - ECConnectorOutput: 专家缓存传输连接器的输出状态。
    - DraftTokenIds: 投机解码 (speculative decoding) 中草稿模型
      生成的候选 token。
"""

from abc import ABC, abstractmethod
from copy import copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NamedTuple, TypeAlias

import numpy as np
import torch

from vllm.compilation.cuda_graph import CUDAGraphStat
from vllm.v1.core.sched.output import SchedulerOutput

if TYPE_CHECKING:
    from vllm.distributed.kv_events import KVConnectorKVEvents
    from vllm.distributed.kv_transfer.kv_connector.v1.base import (
        KVConnectorWorkerMetadata,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorStats
else:
    # TYPE_CHECKING 为 False 时 (运行时)，使用 object 作为占位类型，
    # 避免引入不必要的运行时依赖。
    KVConnectorStats = object
    KVConnectorWorkerMetadata = object
    KVConnectorKVEvents = object


class LogprobsLists(NamedTuple):
    """
    以 NumPy 数组形式存储的 log 概率数据。

    这是 log 概率的"列表"表示，主要用于 ModelRunnerOutput 的 IPC 传输。
    相比 torch.Tensor，NumPy 数组在进程间序列化时开销更低。

    字段说明：
        1. logprob_token_ids: 每个生成位置的候选 token IDs。
           形状为 [总生成位置数, max_num_logprobs + 1]，
           其中 +1 包含采样选中的 token。
        2. logprobs: 对应的 log 概率值，形状同上。
        3. sampled_token_ranks: 被采样选中 token 在候选中的排名。
           形状为 [总生成位置数]，通常为 0 (最高概率)。
        4. cu_num_generated_tokens: 累积生成 token 数列表，用于在
           投机解码场景下按请求切片。因为投机解码中不同请求可能生成
           不同数量的 token，需要此字段来定位每个请求的起始索引。
    """
    # [num_reqs x num_generated_tokens, max_num_logprobs + 1]
    logprob_token_ids: np.ndarray
    # [num_reqs x num_generated_tokens, max_num_logprobs + 1]
    logprobs: np.ndarray
    # [num_reqs x num_generated_tokens]
    sampled_token_ranks: np.ndarray
    # [num_reqs]
    # Used for slicing the logprobs in cases like speculative
    # decoding where the number of generated tokens may be
    # different for each request.
    cu_num_generated_tokens: list[int] | None = None

    def slice_request(self, req_idx: int, num_positions: int):
        """
        按请求索引切片 log 概率数据。

        用于从批量数据中提取单个请求的 log 概率信息。

        参数：
            req_idx: 请求索引。
            num_positions: 该请求的生成位置数量。

        返回：
            LogprobsLists: 包含该请求 log 概率数据的新 NamedTuple。

        处理逻辑：
            1. 如果存在 cu_num_generated_tokens (投机解码场景)，
               则使用累积数组查找实际的起始索引。
            2. 否则直接使用 req_idx 作为起始索引。
            3. 切片返回该请求对应的子数据，cu_num_generated_tokens
               设为 None (单个请求不需要)。
        """
        if self.cu_num_generated_tokens is not None:
            req_idx = self.cu_num_generated_tokens[req_idx]
        end_idx = req_idx + num_positions
        return LogprobsLists(
            self.logprob_token_ids[req_idx:end_idx],
            self.logprobs[req_idx:end_idx],
            self.sampled_token_ranks[req_idx:end_idx],
            None,
        )


class LogprobsTensors(NamedTuple):
    """
    以 PyTorch 张量形式存储的 log 概率数据。

    这是 log 概率的"张量"表示，主要用于 GPU 侧计算过程。
    与 LogprobsLists 的区别在于底层存储格式 (torch.Tensor vs np.ndarray)。

    字段说明：
        1. logprob_token_ids: 每个生成位置的候选 token IDs 张量。
           形状为 [总生成位置数, max_num_logprobs + 1]。
        2. logprobs: 对应的 log 概率值张量，形状同上。
        3. selected_token_ranks: 被采样选中 token 的排名张量。
           形状为 [总生成位置数]。
        4. cu_num_generated_tokens: 累积生成 token 数，用于投机解码
           场景下的按请求切片。
    """
    # [num_reqs x num_generated_tokens, max_num_logprobs + 1]
    logprob_token_ids: torch.Tensor
    # [num_reqs x num_generated_tokens, max_num_logprobs + 1]
    logprobs: torch.Tensor
    # [num_reqs x num_generated_tokens]
    selected_token_ranks: torch.Tensor
    # [num_reqs]
    cu_num_generated_tokens: list[int] | None = None

    def tolists(self, cu_num_generated_tokens: list[int] | None = None):
        """
        将 GPU 张量转换为 CPU NumPy 数组列表形式。

        将所有张量从 GPU 移动到 CPU 并转为 NumPy 数组，
        生成 LogprobsLists 实例。这是 IPC 传输前的必要步骤。

        参数：
            cu_num_generated_tokens: 可选的累积生成 token 数，
                如果为 None 则使用自身的值。

        返回：
            LogprobsLists: CPU 上的 NumPy 数组版本。
        """
        return LogprobsLists(
            self.logprob_token_ids.cpu().numpy(),
            self.logprobs.cpu().numpy(),
            self.selected_token_ranks.cpu().numpy(),
            cu_num_generated_tokens
            if cu_num_generated_tokens is not None
            else self.cu_num_generated_tokens,
        )

    def to_cpu_nonblocking(self) -> "LogprobsTensors":
        """
        非阻塞地将张量从 GPU 复制到 CPU。

        使用 non_blocking=True 实现 GPU 到 CPU 的异步数据传输，
        可以与后续的 GPU 计算重叠执行，提高整体吞吐量。

        如果张量已在 CPU 上，则直接返回自身，不做额外操作。

        返回：
            LogprobsTensors: CPU 上的新张量组 (或自身)。
        """
        if self.logprob_token_ids.device.type == "cpu":
            return self
        return LogprobsTensors(
            self.logprob_token_ids.to("cpu", non_blocking=True),
            self.logprobs.to("cpu", non_blocking=True),
            self.selected_token_ranks.to("cpu", non_blocking=True),
            self.cu_num_generated_tokens,
        )

    def filter(self, mask: torch.Tensor) -> "LogprobsTensors":
        """Filter the logprobs tensors with the given bool mask."""
        """
        使用布尔掩码过滤 log 概率张量。

        用于从批量数据中筛选出满足条件的位置。
        不支持与 cu_num_generated_tokens 同时使用，
        因为过滤会改变位置索引，破坏累积数组的一致性。

        参数：
            mask: 布尔张量，形状为 [总生成位置数]，
                True 表示保留该位置。

        返回：
            LogprobsTensors: 过滤后的新张量组。
        """
        assert self.cu_num_generated_tokens is None, (
            "filter can't be used with cu_num_generated_tokens"
        )
        return LogprobsTensors(
            self.logprob_token_ids[mask],
            self.logprobs[mask],
            self.selected_token_ranks[mask],
        )

    @staticmethod
    def empty_cpu(
        num_positions: int, num_tokens_per_position: int
    ) -> "LogprobsTensors":
        """Create empty LogprobsTensors on CPU."""
        """
        在 CPU 上创建空的 LogprobsTensors 实例。

        用于预分配缓冲区，避免在推理循环中反复分配内存。
        创建的张量内容未初始化，需要在使用前填充数据。

        参数：
            num_positions: 生成位置的数量。
            num_tokens_per_position: 每个位置的候选 token 数
                (即 max_num_logprobs + 1)。

        返回：
            LogprobsTensors: CPU 上的空张量组。
        """

        logprob_token_ids = torch.empty(
            (num_positions, num_tokens_per_position), dtype=torch.int32, device="cpu"
        )
        logprobs = torch.empty_like(logprob_token_ids, dtype=torch.float32)
        selected_token_ranks = torch.empty(
            num_positions, dtype=torch.int32, device="cpu"
        )
        return LogprobsTensors(
            logprob_token_ids=logprob_token_ids,
            logprobs=logprobs,
            selected_token_ranks=selected_token_ranks,
        )


class RoutedExpertsTensors(NamedTuple):
    """Device-side snapshot of routed experts data, pending async D2H.

    Produced by :class:`GPUModelRunner` at the end of each async-scheduled
    step. The copy stream waits on the default stream, then issues
    non-blocking D2H via :meth:`to_cpu_nonblocking` into a pinned CPU
    buffer; :class:`AsyncGPUModelRunnerOutput.get_output` synchronizes
    the copy before the scheduler reads it.

    Sliced to ``total_num_scheduled_tokens`` (step-level, across all
    requests — NOT per-request). Both ``routing_data`` and
    ``slot_mapping`` must be private clones when sourced from shared
    capturer / prepare-input buffers, so the next forward pass /
    ``_prepare_inputs`` on the default stream does not race with a
    D2H still pending on the copy stream.
    """
    """
    GPU 侧的路由专家数据快照，等待异步 D2H (Device-to-Host) 传输。

    这是 MoE (Mixture of Experts) 模型中的关键数据结构。
    在每个异步调度步骤结束时由 GPUModelRunner 产生。

    数据流：
        1. GPUModelRunner 完成前向推理后，记录每个 token 被分配到
           哪些专家。
        2. 复制流 (copy stream) 等待默认流完成后，发起非阻塞 D2H
           传输到 pinned CPU 缓冲区。
        3. AsyncGPUModelRunnerOutput.get_output() 在调度器读取前
           同步传输完成。

    重要说明：
        - 数据维度是按"步骤"聚合的 (跨所有请求的总 token 数)，
          而不是按单个请求。
        - routing_data 和 slot_mapping 必须是私有副本，因为原始
          缓冲区可能被下一次前向推理覆盖，导致数据竞争。
    """

    # (num_scheduled_tokens, num_layers, num_experts_per_tok)
    # 每个 token 在每层被路由到的专家 ID
    routing_data: torch.Tensor
    # (num_scheduled_tokens,)
    # 每个 token 对应的物理 KV 缓存槽位索引
    slot_mapping: torch.Tensor

    def to_cpu_nonblocking(self) -> "RoutedExpertsTensors":
        """Issue non-blocking D2H on the current stream.

        NOTE: ``non_blocking=True`` only delivers true overlap when the
        CPU target is pinned. The current fallback here allocates a
        new pageable CPU tensor per call, which silently degrades to a
        synchronous copy; acceptable because the sync happens on the
        dedicated copy stream, not the default stream.
        """
        """
        在当前流上发起非阻塞的 GPU 到 CPU 数据传输。

        重要说明：
            - non_blocking=True 仅在目标 CPU 内存为 pinned memory
              时才能实现真正的异步传输。
            - 当前实现每次调用会分配新的 pageable CPU 张量，
              实际上会退化为同步拷贝。但由于同步发生在专用的
              复制流上 (而非默认流)，因此不会阻塞 GPU 计算。

        返回：
            RoutedExpertsTensors: CPU 上的新张量组 (或自身)。
        """
        if self.routing_data.device.type == "cpu":
            return self
        return RoutedExpertsTensors(
            self.routing_data.to("cpu", non_blocking=True),
            self.slot_mapping.to("cpu", non_blocking=True),
        )

    def tolists(self) -> "RoutedExpertsLists":
        """Convert to the numpy-backed form consumed by the scheduler.

        ``.cpu()`` is a no-op when the tensor is already on CPU, so this
        is cheap for the post-D2H case; for raw device tensors it will
        synchronously block, which is only reached in tests.
        """
        """
        将张量转换为 NumPy 数组形式，供调度器消费。

        .cpu() 在张量已在 CPU 上时为空操作，因此 D2H 传输后的调用
        开销很低；对于仍在 GPU 上的原始张量 (仅在测试中出现)，
        会同步阻塞。

        返回：
            RoutedExpertsLists: CPU 上的 NumPy 数组版本。
        """
        return RoutedExpertsLists(
            self.routing_data.cpu().numpy(),
            self.slot_mapping.cpu().numpy(),
        )


class RoutedExpertsLists(NamedTuple):
    """CPU-side routed experts, the form :meth:`RoutedExpertsManager.store_batch`
    consumes.

    Batched per scheduler step: the leading dim is the number of tokens
    scheduled across all requests in this step (``total_num_scheduled_tokens``),
    not per-request tokens. ``slot_mapping[i]`` tells the scheduler which
    physical KV-cache slot row ``i`` of ``routing_data`` belongs to.
    """
    """
    CPU 侧的路由专家数据，由 RoutedExpertsManager.store_batch() 消费。

    这是 RoutedExpertsTensors 传输到 CPU 后的最终形式。
    调度器使用此数据将路由信息写入对应的 KV 缓存槽位缓冲区。

    数据组织方式：
        - 按调度步骤批量处理，首维度是该步骤中所有请求的总 token 数
          (total_num_scheduled_tokens)，而非单个请求的 token 数。
        - slot_mapping[i] 告诉调度器 routing_data 的第 i 行数据
          对应哪个物理 KV 缓存槽位。
        - 调度器通过 slot_buffer[slot_mapping] = routing_data
          将路由信息持久化到槽位缓冲区中。
    """

    # (num_scheduled_tokens, num_layers, num_experts_per_tok)
    # 每个 token 在每层被路由到的专家 ID (NumPy 数组)
    routing_data: np.ndarray
    # (num_scheduled_tokens,)
    # 每个 token 对应的物理 KV 缓存槽位索引 (NumPy 数组)
    slot_mapping: np.ndarray


# [num_reqs, <dynamic>]
# The shape of each element depends on the pooler used
# 池化器 (pooler) 输出的类型别名。
# 形状取决于所使用的池化器类型，可能是：
#   - 单个张量 (所有请求共用一个输出)
#   - 张量列表 (每个请求一个输出)
#   - 可选张量列表 (部分请求可能没有输出，如嵌入模型)
PoolerOutput: TypeAlias = torch.Tensor | list[torch.Tensor] | list[torch.Tensor | None]


@dataclass
class SamplerOutput:
    """
    采样器 (Sampler) 的直接输出。

    由 GPUModelRunner 在每步推理后产生，包含采样得到的 token IDs
    和对应的 log 概率张量。这是 GPU 侧的原始输出，后续会被转换为
    ModelRunnerOutput 用于 IPC 传输。

    字段说明：
        1. sampled_token_ids: 采样得到的 token IDs 张量。
           形状为 [num_reqs, max_num_generated_tokens]。
           不同请求可能生成不同数量的 token (投机解码)，
           不足的部分用 PLACEHOLDER_TOKEN_ID (-1) 填充。
        2. logprobs_tensors: 对应的 log 概率张量，可能为 None
           (当用户未请求 log 概率时)。
    """
    # [num_reqs, max_num_generated_tokens]
    # Different requests can have different number of generated tokens.
    # All requests are padded to max_num_generated_tokens.
    # PLACEHOLDER_TOKEN_ID (-1 by default) is used for padding.
    sampled_token_ids: torch.Tensor
    logprobs_tensors: LogprobsTensors | None


@dataclass
class KVConnectorOutput:
    """
    KV 缓存传输连接器的输出状态。

    在分布式推理场景 (如 PD 分离 - Prefill/Decode 分离部署) 中，
    KV 缓存需要在不同节点之间传输。此数据结构记录传输的状态信息。

    字段说明：
        1. finished_sending: 已完成 KV 缓存发送的请求 ID 集合。
        2. finished_recving: 已完成 KV 缓存接收的请求 ID 集合。
        3. kv_connector_stats: KV 连接器的性能统计信息。
        4. kv_cache_events: KV 缓存事件，用于事件驱动的缓存管理。
        5. kv_connector_worker_meta: KV 连接器工作节点的元数据。
        6. invalid_block_ids: 加载失败的外部 KV 缓存块 ID 集合。
           引用这些块的请求需要重新调度以重新计算。
        7. expected_finished_count: 每个请求预期的发送/接收完成
           通知数量。用于 Nixl 等基于握手的连接器更新
           KVOutputAggregator。通常是连接器发现后的静态配置。
    """
    # [req_ids]
    finished_sending: set[str] | None = None
    finished_recving: set[str] | None = None
    kv_connector_stats: KVConnectorStats | None = None
    kv_cache_events: KVConnectorKVEvents | None = None
    kv_connector_worker_meta: KVConnectorWorkerMetadata | None = None
    # IDs of externally computed KV blocks that failed to load.
    # Requests referencing these blocks should be rescheduled to recompute them
    invalid_block_ids: set[int] = field(default_factory=set)
    # Configuration describing how many finished sending/receiving
    # notifications should be expected for each request. This allows
    # handshake-based connectors like Nixl to update the KVOutputAggregator.
    # It captures a static setup info and should almost always remain constant
    # for a given connector after discovery. Default value entails no change.
    expected_finished_count: int = 0

    def is_empty(self):
        """
        检查此输出是否为空 (无任何有意义的状态信息)。

        所有字段都为 None 或空集合时返回 True。
        用于优化：空输出可以跳过序列化和传输。

        返回：
            bool: 是否为空。
        """
        return (
            not self.finished_sending
            and not self.finished_recving
            and not self.kv_connector_stats
            and not self.kv_cache_events
            and not self.invalid_block_ids
            and not self.kv_connector_worker_meta
        )


@dataclass
class ECConnectorOutput:
    """
    专家缓存 (Expert Cache) 传输连接器的输出状态。

    类似于 KVConnectorOutput，但用于 MoE 模型中专家缓存的传输。
    在专家缓存需要在不同节点之间同步时使用。

    字段说明：
        1. finished_sending: 已完成专家缓存发送的多模态哈希集合。
        2. finished_recving: 已完成专家缓存接收的多模态哈希集合。
    """
    # [mm_hash]
    finished_sending: set[str] | None = None
    finished_recving: set[str] | None = None


# ModelRunnerOutput is serialized and sent to the scheduler process.
# This is expensive for torch.Tensor so prefer to use list instead.
# ModelRunnerOutput 是序列化后发送给调度器进程的输出数据。
# 使用 torch.Tensor 进行序列化开销较大，因此优先使用 Python list。
@dataclass
class ModelRunnerOutput:
    """
    模型运行器的输出数据，通过 IPC 发送给调度器进程。

    这是 v1 引擎中最核心的输出数据结构。与 SamplerOutput 的区别是：
    SamplerOutput 是 GPU 侧的原始输出 (使用 torch.Tensor)，
    而 ModelRunnerOutput 将数据转换为 Python 原生类型 (list, dict)
    以降低 IPC 序列化开销。

    数据流：
        1. GPUModelRunner 产生 SamplerOutput (GPU 张量)。
        2. 输出处理器将其转换为 ModelRunnerOutput (Python list)。
        3. 通过 ZMQ IPC 发送给调度器进程。
        4. 调度器据此更新请求状态并返回新 token。

    字段说明：
        1. req_ids: 本步骤处理的请求 ID 列表。
        2. req_id_to_index: 请求 ID 到索引的映射字典。
        3. sampled_token_ids: 每个请求的采样 token IDs (list 形式)。
        4. logprobs: 生成 token 的 log 概率数据。
        5. prompt_logprobs_dict: 提示词 token 的 log 概率数据
           (仅当用户请求 prompt_logprobs 时)。
        6. pooler_output: 池化器输出 (用于嵌入模型等)。
        7. kv_connector_output: KV 缓存传输状态。
        8. ec_connector_output: 专家缓存传输状态。
        9. num_nans_in_logits: logits 中 NaN 的数量 (调试用)。
        10. cudagraph_stats: CUDA 图执行统计信息。
        11. routed_experts: MoE 路由专家数据。
    """
    # [num_reqs]
    req_ids: list[str]
    # req_id -> index
    req_id_to_index: dict[str, int]

    # num_reqs x num_generated_tokens
    # num_generated_tokens is the number of tokens
    # generated in the current step. It can be different for
    # each request due to speculative/jump decoding.
    sampled_token_ids: list[list[int]] = field(default_factory=list)

    # [num_reqs, max_num_logprobs + 1]
    # [num_reqs, max_num_logprobs + 1]
    # [num_reqs]
    logprobs: LogprobsLists | None = None

    # req_id -> (token_ids, logprobs, ranks)
    # [prompt_len, num_prompt_logprobs]
    # [prompt_len, num_prompt_logprobs]
    # [prompt_len]
    prompt_logprobs_dict: dict[str, LogprobsTensors | None] = field(
        default_factory=dict
    )

    # [num_reqs, hidden_size]
    pooler_output: list[torch.Tensor | None] | None = None

    kv_connector_output: KVConnectorOutput | None = None

    ec_connector_output: ECConnectorOutput | None = None

    # req_id -> num_nans_in_logits
    num_nans_in_logits: dict[str, int] | None = None

    # information related to cudagraph execution
    cudagraph_stats: CUDAGraphStat | None = None

    # Per-step routed experts data captured by the worker.
    # ``routing_data`` shape: (num_scheduled_tokens, num_layers,
    #                         num_experts_per_tok); expert IDs as uint8/uint16.
    # ``slot_mapping`` shape: (num_scheduled_tokens,); physical KV-cache
    #                         slot for each row of routing_data.
    # ``num_scheduled_tokens`` is step-level (total across all requests
    # in this step), not per-request. The scheduler persists this into
    # its slot buffer via ``slot_buffer[slot_mapping] = routing_data``.
    # ``None`` when ``enable_return_routed_experts`` is off.
    routed_experts: RoutedExpertsLists | None = None

    @staticmethod
    def with_kv_conn_output_only(
        kv_connector_output: KVConnectorOutput | None,
    ) -> "ModelRunnerOutput":
        """Return ModelRunnerOutput containing the provided KVConnectorOutput,
        otherwise empty. Returns None if kv_connector_output is passed as None.
        """
        """
        创建仅包含 KVConnectorOutput 的 ModelRunnerOutput。

        这是一个工厂方法，用于在仅需要传输 KV 连接器状态时
        创建轻量级输出，避免构造完整的 ModelRunnerOutput。

        参数：
            kv_connector_output: KV 连接器输出，为 None 或空时
                返回空的 ModelRunnerOutput 单例。

        返回：
            ModelRunnerOutput: 包含 KV 连接器输出的实例，
            或 EMPTY_MODEL_RUNNER_OUTPUT 单例。
        """
        if kv_connector_output is None or kv_connector_output.is_empty():
            return EMPTY_MODEL_RUNNER_OUTPUT
        output = copy(EMPTY_MODEL_RUNNER_OUTPUT)
        output.kv_connector_output = kv_connector_output
        return output


# ModelRunnerOutput wrapper for async scheduling.
# 异步调度场景下的 ModelRunnerOutput 包装器。
class AsyncModelRunnerOutput(ABC):
    """
    异步模型运行器输出的抽象基类。

    在异步调度模式下，模型前向推理和结果处理是解耦的。
    此类包装了延迟就绪的 ModelRunnerOutput，调用者通过
    get_output() 等待结果就绪。

    典型使用场景：
        GPUModelRunner 完成推理后，返回 AsyncModelRunnerOutput。
        调度器在需要时调用 get_output()，此时才会执行：
        1. GPU 到 CPU 的数据传输
        2. 同步等待所有异步操作完成
        3. 返回最终的 ModelRunnerOutput
    """

    @abstractmethod
    def get_output(self) -> ModelRunnerOutput:
        """Get the ModelRunnerOutput for this async output.

        This is a blocking call that waits until the results are ready, which
        might involve copying device tensors to the host.
        This method should only be called once per AsyncModelRunnerOutput.
        """
        pass


@dataclass
class DraftTokenIds:
    """
    投机解码 (Speculative Decoding) 中草稿模型的输出。

    在投机解码流程中，草稿模型 (draft model) 快速生成多个候选
    token，然后由目标模型 (target model) 并行验证。此数据结构
    存储草稿模型的输出。

    字段说明：
        1. req_ids: 请求 ID 列表。
        2. draft_token_ids: 每个请求的草稿 token IDs 列表。
           内部列表长度可能不同 (取决于草稿模型的预测)。
    """
    # [num_reqs]
    req_ids: list[str]
    # num_reqs x num_draft_tokens
    draft_token_ids: list[list[int]]


def make_empty_encoder_model_runner_output(
    scheduler_output: "SchedulerOutput",
) -> ModelRunnerOutput:
    """
    Create a ModelRunnerOutput stub that contains the correct
    per-request bookkeeping but no generated data yet.
    """
    """
    为编码器 (encoder) 模型创建空的 ModelRunnerOutput 存根。

    编码器模型 (如嵌入模型) 的处理流程与解码器不同：
    前向推理可能需要分多步完成 (例如处理长输入)。
    此函数创建一个包含正确请求簿记信息但尚无生成数据的存根。

    用途：
        在编码器模型完成部分推理步骤后，返回此存根给调度器，
        让调度器知道哪些请求已处理但尚未生成输出。

    参数：
        scheduler_output: 调度器输出，包含本步骤的调度信息。

    返回：
        ModelRunnerOutput: 包含请求簿记但无生成数据的存根，
        或 EMPTY_MODEL_RUNNER_OUTPUT 单例 (无调度 token 时)。
    """
    if not scheduler_output.num_scheduled_tokens:
        return EMPTY_MODEL_RUNNER_OUTPUT

    # Convert to list so we get a deterministic, indexable sequence
    req_ids: list[str] = list(scheduler_output.num_scheduled_tokens.keys())

    # Give every request its own contiguous index
    req_id_to_index: dict[str, int] = {rid: idx for idx, rid in enumerate(req_ids)}

    # No tokens generated yet ⇒ one empty list per request
    # 每个请求分配一个包含单个 0 的列表作为占位符
    sampled_token_ids: list[list[int]] = [[0] for _ in req_ids]

    # Pooler outputs are not available yet ⇒ use None placeholders
    # 池化器输出尚未就绪，使用 None 作为占位符
    pooler_output: list[torch.Tensor | None] = [None for _ in req_ids]

    return ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index=req_id_to_index,
        sampled_token_ids=sampled_token_ids,
        pooler_output=pooler_output,
    )


# 全局单例：空的 ModelRunnerOutput，用于无数据时避免重复创建。
EMPTY_MODEL_RUNNER_OUTPUT = ModelRunnerOutput(req_ids=[], req_id_to_index={})
