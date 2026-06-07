# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
vLLM v1 引擎核心数据结构定义模块。

本模块定义了 vLLM v1 引擎中所有核心数据结构和枚举类型，
是引擎各组件之间通信的基础。

主要数据结构分为以下几类：

1. 请求相关：
   - EngineCoreRequest: 引擎核心请求，包含完整的请求信息
     （token IDs、采样参数、多模态输入等）
   - EngineCoreRequestType: 请求类型枚举（ADD、ABORT 等）

2. 输出相关：
   - EngineCoreOutput: 单个请求的输出（新 token、对数概率、完成原因等）
   - EngineCoreOutputs: 批量输出（包含多个请求的输出和调度统计）

3. 事件相关：
   - EngineCoreEvent: 引擎核心事件（排队、调度、抢占）
   - EngineCoreEventType: 事件类型枚举

4. 完成原因：
   - FinishReason: 请求完成原因枚举（stop、length、abort 等）

5. 配置相关：
   - EngineCoreReadyResponse: 引擎启动完成的响应
   - ReconfigureDistributedRequest: 分布式重配置请求

6. 工具/实用：
   - UtilityOutput: 工具调用的输出

7. 暂停模式：
   - PauseMode: 暂停生成模式（abort/wait/keep）

8. EEP（弹性引擎池）通知：
   - EEPNotificationType: 通知类型枚举

设计要点：
1. 使用 msgspec.Struct 实现高效序列化（比 dataclass 更快）
2. 使用 enum.IntEnum 实现紧凑的整数序列化
3. array_like=True 启用类数组序列化优化
4. omit_defaults=True 跳过默认值以减少序列化大小
5. gc=False 禁用垃圾回收跟踪以提升性能
"""

import enum
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import msgspec
import numpy as np
import torch

from vllm.lora.request import LoRARequest
from vllm.multimodal.inputs import MultiModalFeatureSpec
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams
from vllm.v1.metrics.stats import PrefillStats, SchedulerStats
from vllm.v1.outputs import LogprobsLists, LogprobsTensors
from vllm.v1.serial_utils import UtilityResult

# Type for pause_generation mode parameter.
# - "abort": Abort all in-flight requests immediately (default).
# - "wait": Wait for in-flight requests to complete before pausing.
# - "keep": Freeze requests in queue; they resume on resume_generation().
# 暂停生成模式参数的类型定义：
# - "abort": 立即中止所有正在处理的请求（默认行为）
# - "wait": 等待正在处理的请求完成后再暂停
# - "keep": 冻结队列中的请求，恢复时继续执行
PauseMode = Literal["abort", "wait", "keep"]

# These are possible values of RequestOutput.finish_reason,
# so form part of the external API.
# 请求完成原因的字符串表示，这些值是外部 API 的一部分，
# 用于 RequestOutput.finish_reason 字段。
FINISH_REASON_STRINGS = ("stop", "length", "abort", "error", "repetition")

# EEP 通知的特殊调用 ID，-1 表示这不是一个普通的工具调用
EEP_NOTIFICATION_CALL_ID = -1


class EEPNotificationType(enum.Enum):
    """
    弹性引擎池（Elastic Engine Pool, EEP）通知类型。

    用于在引擎重配置过程中，通知前端（API 服务器）引擎状态的变化。

    通知流程：
    1. NEW_CORE_ENGINES_INIT_READY - 新的核心引擎已初始化
    2. NEW_CORE_ENGINES_WEIGHTS_INIT_READY - 新引擎的权重已加载完成
    3. RECONFIGURE_FINISHED - 重配置完成
    4. SHUTDOWN_COMPLETE - 关闭完成
    """
    NEW_CORE_ENGINES_INIT_READY = "NEW_CORE_ENGINES_INIT_READY"
    NEW_CORE_ENGINES_WEIGHTS_INIT_READY = "NEW_CORE_ENGINES_WEIGHTS_INIT_READY"
    RECONFIGURE_FINISHED = "RECONFIGURE_FINISHED"
    SHUTDOWN_COMPLETE = "SHUTDOWN_COMPLETE"


class FinishReason(enum.IntEnum):
    """
    Reason a request finished - stop, length, abort, error, or repetition.

    Int rather than Str for more compact serialization.

    stop - a stop string was emitted
    length - max_tokens was consumed, or max_model_len was reached
    abort - aborted by client
    error - retryable request-level internal error (e.g., KV load failure).
            Invariant: always converted to 500 Internal Server Error.
    repetition - repetitive token pattern detected (hallucination)

    """
    """
    请求完成原因枚举。

    使用 IntEnum 而非字符串枚举，以实现更紧凑的序列化。

    各值含义：
    - STOP (0): 匹配到停止字符串，正常结束
    - LENGTH (1): 达到 max_tokens 或 max_model_len 限制
    - ABORT (2): 被客户端主动中止
    - ERROR (3): 可重试的请求级内部错误（如 KV 缓存加载失败），
                 最终会转换为 HTTP 500 Internal Server Error
    - REPETITION (4): 检测到重复 token 模式（幻觉），提前终止
    """

    STOP = 0
    LENGTH = 1
    ABORT = 2
    ERROR = 3
    REPETITION = 4

    def __str__(self):
        return FINISH_REASON_STRINGS[self.value]


@dataclass
class EngineCoreReadyResponse:
    """Sent from EngineCore to each frontend at the end of engine startup.

    Contains post-initialization config that may differ from the original
    values (e.g. max_model_len after KV cache auto-fitting).
    """
    """
    引擎核心就绪响应。

    在引擎启动完成时，由 EngineCore 发送给每个前端（API 服务器）。
    包含初始化后的配置信息，这些值可能与初始配置不同
    （例如，KV 缓存自动适配后的 max_model_len）。

    字段说明：
    - max_model_len: 模型最大序列长度（可能因 KV 缓存适配而调整）
    - num_gpu_blocks: GPU 上的 KV 缓存块数量
    - dp_stats_address: 数据并统计信息的地址（DP 模式下使用）
    - dtype: 模型数据类型（如 "float16", "bfloat16"）
    - vllm_version: vLLM 版本号
    """

    max_model_len: int
    num_gpu_blocks: int
    dp_stats_address: str | None
    dtype: str
    vllm_version: str


class EngineCoreRequest(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,
):  # type: ignore[call-arg]
    """
    引擎核心请求数据结构。

    这是 vLLM v1 引擎中请求的标准表示，包含处理请求所需的所有信息。
    使用 msgspec.Struct 实现高效序列化，适合通过 IPC 传输。

    字段说明：
    - request_id: 请求的唯一标识符（内部使用）
    - prompt_token_ids: 输入 prompt 的 token ID 列表
    - mm_features: 多模态特征规范列表
    - sampling_params: 采样参数（用于生成任务）
    - pooling_params: 池化参数（用于嵌入任务）
    - arrival_time: 请求到达时间戳
    - lora_request: LoRA 适配器请求
    - cache_salt: 缓存盐值（用于前缀缓存的隔离）
    - data_parallel_rank: 数据并行的 rank（DP 模式下使用）
    - prompt_embeds: 预计算的 prompt 嵌入（与 prompt_token_ids 二选一）
    - prompt_is_token_ids: 混合模式的位置掩码（True=token ID, False=嵌入）
    - client_index: 客户端索引（用于前端扩缩容时路由输出）
    - current_wave: 当前请求波次（DP 模式下处理竞态条件）
    - priority: 请求优先级
    - trace_headers: 追踪头信息
    - resumable: 请求是否可恢复（用于中断后继续）
    - external_req_id: 用户提供的原始请求 ID
    - reasoning_ended: 推理是否已结束
    - reasoning_parser_kwargs: 推理解析器的额外参数
    - abort_immediately: 是否立即中止（用于 KV 传输清理）
    """
    request_id: str
    prompt_token_ids: list[int] | None
    mm_features: list[MultiModalFeatureSpec] | None
    sampling_params: SamplingParams | None
    pooling_params: PoolingParams | None
    arrival_time: float
    lora_request: LoRARequest | None
    cache_salt: str | None
    data_parallel_rank: int | None
    prompt_embeds: torch.Tensor | None = None

    # Per-position mask for mixed-mode inputs (e.g chat completion with
    # prompt_embeds content parts). `True` means the position is a real
    # token ID; `False` means the position uses a pre-computed entry from
    # `prompt_embeds`. `None` for pure-tokens and pure-embeds requests.
    # 混合模式输入的逐位置掩码（例如带有 prompt_embeds 内容部分的聊天补全）。
    # True 表示该位置是真实的 token ID；
    # False 表示该位置使用 prompt_embeds 中的预计算条目。
    # 对于纯 token 或纯嵌入请求，此字段为 None。
    prompt_is_token_ids: list[bool] | None = None

    # Index of the the client, used to ensure outputs are sent back to the same
    # client for this request when scaling out the front-end.
    # 客户端索引，用于在前端扩缩容时确保输出被发送回发起请求的同一客户端。
    client_index: int = 0

    # Used in DP case to indicate which wave of requests this is expected to
    # belong to, to cover a race condition where the request is sent before
    # a wave finished notification is received.
    # 在 DP 模式下使用，指示此请求预期属于哪个请求波次，
    # 用于覆盖请求在波次完成通知到达之前发送的竞态条件。
    current_wave: int = 0
    priority: int = 0

    trace_headers: Mapping[str, str] | None = None
    resumable: bool = False

    # The user-provided request ID. This field is set internally,
    # copied from the provided request_id that's originally assigned
    # to the request_id field, see InputProcessor.assign_request_id().
    # Used in outputs and to support abort(req_id, internal=False).
    # 用户提供的原始请求 ID。此字段在内部设置，
    # 从最初分配给 request_id 字段的 request_id 复制而来。
    # 用于输出和 abort(req_id, internal=False) 操作。
    external_req_id: str | None = None

    reasoning_ended: bool | None = None
    reasoning_parser_kwargs: dict[str, Any] | None = None

    # If True, the request should be added to the scheduler's waiting queue
    # and immediately aborted, so connector-side cleanup runs via the standard
    # request_finished hook. Used to free P-side prefill blocks when a
    # KV-transfer request is rejected on the D node before engine admission.
    # 如果为 True，请求应被添加到调度器的等待队列并立即中止，
    # 这样连接器侧的清理可以通过标准的 request_finished 钩子运行。
    # 用于在 KV 传输请求在 D 节点上被拒绝时，释放 P 侧的预填充块。
    abort_immediately: bool = False

    @property
    def params(self) -> SamplingParams | PoolingParams:
        """Return the processed params (sampling or pooling)."""
        """返回处理后的参数（采样参数或池化参数）。

        优先返回采样参数，如果不存在则返回池化参数。
        两种参数互斥：一个请求只能使用其中一种。

        返回:
            SamplingParams 或 PoolingParams 实例

        异常:
            AssertionError: 如果两者都为 None
        """
        if self.sampling_params is not None:
            return self.sampling_params
        assert self.pooling_params is not None
        return self.pooling_params


class EngineCoreEventType(enum.IntEnum):
    """The type of engine core request event."""
    """
    引擎核心请求事件类型。

    用于追踪请求在引擎中的生命周期：
    - QUEUED (1): 请求进入等待队列
    - SCHEDULED (2): 请求被调度执行
    - PREEMPTED (3): 请求被抢占（释放 KV 缓存以腾出空间）
    """

    QUEUED = 1
    SCHEDULED = 2
    PREEMPTED = 3


class EngineCoreEvent(msgspec.Struct):
    """A timestamped engine core event associated with a request.

    The timestamp is a monotonic timestamps and is used for by the engine
    frontend to calculate intervals between engine core events. These
    timestamps should not be compared with timestamps from other processes.
    """
    """
    带时间戳的引擎核心事件。

    与请求关联的事件记录，用于性能分析和监控。
    使用单调时钟（monotonic clock），确保时间戳只增不减。

    注意：这些时间戳只能在同一进程内比较，
    不应与其他进程的时间戳进行比较。

    字段说明：
    - type: 事件类型（QUEUED/SCHEDULED/PREEMPTED）
    - timestamp: 单调时间戳
    """

    type: EngineCoreEventType
    timestamp: float

    @classmethod
    def new_event(
        cls, event_type: EngineCoreEventType, timestamp: float | None = None
    ) -> "EngineCoreEvent":
        """
        创建新的事件实例。

        参数:
            event_type: 事件类型
            timestamp: 时间戳，如果为 None 则使用当前单调时间

        返回:
            新创建的 EngineCoreEvent 实例
        """
        timestamp = time.monotonic() if timestamp is None else timestamp
        return cls(event_type, timestamp)


class EngineCoreOutput(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,
):  # type: ignore[call-arg]
    """
    引擎核心输出数据结构。

    表示单个请求在一个迭代步骤中的输出。
    使用 msgspec.Struct 实现高效序列化。

    字段说明：
    - request_id: 请求 ID
    - new_token_ids: 新生成的 token ID 列表
    - new_logprobs: 新的采样对数概率
    - new_prompt_logprobs_tensors: 新的提示对数概率张量
    - pooling_output: 池化输出（用于嵌入任务）
    - finish_reason: 完成原因（None 表示请求仍在进行中）
    - stop_reason: 停止原因（停止字符串或停止 token ID）
    - events: 引擎核心事件列表（用于性能追踪）
    - kv_transfer_params: KV 传输参数（用于分离式预填充/解码）
    - trace_headers: 追踪头信息
    - prefill_stats: 预填充统计信息
    - routed_experts: 路由的专家索引（MoE 模型使用）
    - num_nans_in_logits: logits 中的 NaN 数量（>0 表示输出损坏）
    """
    request_id: str
    new_token_ids: list[int]

    new_logprobs: LogprobsLists | None = None
    new_prompt_logprobs_tensors: LogprobsTensors | None = None

    pooling_output: torch.Tensor | None = None

    finish_reason: FinishReason | None = None
    stop_reason: int | str | None = None
    events: list[EngineCoreEvent] | None = None
    kv_transfer_params: dict[str, Any] | None = None

    trace_headers: Mapping[str, str] | None = None

    prefill_stats: PrefillStats | None = None

    routed_experts: np.ndarray | None = None
    # The number of NaNs in logits.
    # A value greater than 0 indicates that the output is corrupted.
    # logits 中的 NaN 数量。大于 0 表示输出已损坏。
    num_nans_in_logits: int = 0

    @property
    def finished(self) -> bool:
        """返回请求是否已完成（finish_reason 不为 None 表示已完成）。"""
        return self.finish_reason is not None


class UtilityOutput(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    gc=False,
):  # type: ignore[call-arg]
    """
    工具调用输出数据结构。

    用于返回引擎内部工具调用（utility call）的结果。
    工具调用是引擎核心支持的一种特殊操作，如模型信息查询等。

    字段说明：
    - call_id: 调用 ID（用于匹配请求和响应）
    - failure_message: 失败消息（非 None 表示调用失败，result 应为 None）
    - result: 调用结果（None 表示调用失败）
    """
    call_id: int

    # Non-None implies the call failed, result should be None.
    # 非 None 表示调用失败，此时 result 应为 None。
    failure_message: str | None = None
    result: UtilityResult | None = None


class EngineCoreOutputs(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,
):  # type: ignore[call-arg]
    """
    引擎核心批量输出数据结构。

    包含一个迭代步骤中所有请求的输出以及调度统计信息。
    这是引擎核心发送给前端的主要数据包。

    字段说明：
    - engine_index: 引擎索引（多引擎模式下使用）
    - outputs: 本迭代的请求输出列表
    - scheduler_stats: 调度统计信息
    - timestamp: 输出产生的时间戳
    - utility_output: 工具调用输出（如果有）
    - finished_requests: 已完成请求的 ID 集合
    - wave_complete: DP 模式下，当前波次完成的标记
    - start_wave: DP 模式下，需要启动新波次的标记
    """

    # NOTE(Nick): We could consider ways to make this more compact,
    # e.g. columnwise layout
    # 注意：可以考虑更紧凑的布局方式，例如列式布局

    engine_index: int = 0

    # [num_reqs]
    # 本迭代中所有请求的输出列表
    outputs: list[EngineCoreOutput] = []
    scheduler_stats: SchedulerStats | None = None
    timestamp: float = 0.0

    utility_output: UtilityOutput | None = None
    finished_requests: set[str] | None = None

    # In DP case, used to signal that the current wave of requests
    # has finished and the engines are paused.
    # 在 DP 模式下，用于标记当前请求波次已完成且引擎已暂停。
    wave_complete: int | None = None
    # In DP case, used to signal that a request was received for an
    # "old" wave, so the next wave needs to be started in other engines.
    # 在 DP 模式下，用于标记收到了属于"旧"波次的请求，
    # 需要在其他引擎中启动下一个波次。
    start_wave: int | None = None

    def __post_init__(self):
        """初始化后处理：如果时间戳为默认值（0.0），设置为当前单调时间。"""
        if self.timestamp == 0.0:
            self.timestamp = time.monotonic()


class EngineCoreRequestType(enum.Enum):
    """
    Request types defined as hex byte strings, so it can be sent over sockets
    without separate encoding step.
    """
    """
    引擎核心请求类型枚举。

    使用十六进制字节字符串定义，以便可以直接通过 socket 发送，
    无需额外的编码步骤。

    各类型说明：
    - ADD (0x00): 添加新请求
    - ABORT (0x01): 中止请求
    - START_DP_WAVE (0x02): 启动数据并行波次
    - UTILITY (0x03): 工具调用
    - EXECUTOR_FAILED (0x04): 执行器失败标记（内部使用）
    - WAKEUP (0x05): 唤醒标记（用于关闭期间唤醒阻塞的 get()）
    """

    ADD = b"\x00"
    ABORT = b"\x01"
    START_DP_WAVE = b"\x02"
    UTILITY = b"\x03"
    # Sentinel used within EngineCoreProc.
    # 用于 EngineCoreProc 内部的哨兵值，标记执行器已失败。
    EXECUTOR_FAILED = b"\x04"
    # Sentinel to wake up input_queue.get() during shutdown.
    # 用于在关闭期间唤醒 input_queue.get() 的哨兵值。
    WAKEUP = b"\x05"


class ReconfigureDistributedRequest(msgspec.Struct):
    """
    分布式重配置请求数据结构。

    用于在运行时调整数据并行（DP）配置。
    当需要改变 DP 大小或重新分配 rank 时使用。

    字段说明：
    - new_data_parallel_size: 新的数据并行大小
    - new_data_parallel_rank: 新的数据并行 rank
    - new_data_parallel_rank_local: 新的本地数据并行 rank
    - new_data_parallel_master_ip: 数据并行主节点 IP
    - new_data_parallel_master_port: 数据并行主节点端口
    - new_data_parallel_master_port_list: 数据并行主节点端口列表
    - coord_store_port: 协调存储端口
    """
    new_data_parallel_size: int
    new_data_parallel_rank: int
    new_data_parallel_rank_local: int
    new_data_parallel_master_ip: str
    new_data_parallel_master_port: int
    new_data_parallel_master_port_list: list[int]
    coord_store_port: int


class ReconfigureRankType(enum.IntEnum):
    """
    Rank type for reconfiguring distributed request.
    """
    """
    分布式重配置的 rank 类型枚举。

    用于指定在重配置过程中如何处理当前 rank：

    - KEEP_CURRENT_RANK (-1): 保持当前 rank 不变，
      仅更新配置参数
    - SHUTDOWN_CURRENT_RANK (-2): 关闭当前 rank，
      该 rank 对应的进程将退出
    """

    KEEP_CURRENT_RANK = -1
    SHUTDOWN_CURRENT_RANK = -2
