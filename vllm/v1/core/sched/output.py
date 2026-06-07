# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
调度输出模块 (vllm/v1/core/sched/output.py)

本模块定义了调度器输出的数据结构，是调度器与模型运行器之间的"契约"。

核心数据结构：
1. NewRequestData: 首次调度的新请求数据
   - 包含请求的完整信息（token IDs、参数、块 ID 等）
   - 工作器会缓存这些数据，后续步骤不需要重传

2. CachedRequestData: 已缓存请求的增量数据
   - 仅包含需要更新的差异信息（新块 ID、新 token 数等）
   - 通过 resumbed_req_ids 区分"恢复"和"追加"两种块 ID 更新方式
   - 大幅减少调度器与工作器之间的通信开销

3. SchedulerOutput: 调度输出（每次调度步骤生成一个）
   - 包含新请求和已缓存请求的数据
   - 包含调度 token 数、投机解码 token、编码器输入等
   - 包含已完成和已抢占的请求 ID
   - 包含 KV 连接器和 EC 连接器的元数据

4. GrammarOutput: 语法输出
   - 用于结构化输出（JSON schema、正则表达式等）
   - 包含语法验证的位掩码

数据流：
1. 调度器生成 SchedulerOutput
2. SchedulerOutput 通过 IPC 发送到工作器进程
3. 工作器使用 NewRequestData 初始化新请求
4. 工作器使用 CachedRequestData 更新已缓存请求
5. 工作器执行模型前向传播
6. 模型输出通过 ModelRunnerOutput 返回调度器
"""

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
    import torch

    from vllm.distributed.ec_transfer.ec_connector.base import ECConnectorMetadata
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
    from vllm.lora.request import LoRARequest
    from vllm.multimodal.inputs import MultiModalFeatureSpec
    from vllm.pooling_params import PoolingParams
    from vllm.sampling_params import SamplingParams
    from vllm.v1.request import Request
else:
    ECConnectorMetadata = object
    KVConnectorMetadata = object
    LoRARequest = object
    MultiModalFeatureSpec = object
    PoolingParams = object
    SamplingParams = object
    Request = object


@dataclass
class NewRequestData:
    """
    首次调度的新请求数据。

    当请求第一次被调度时，其完整数据通过此类传递给工作器。
    工作器会缓存这些数据，后续调度步骤仅发送增量更新。

    属性：
        req_id: 请求的唯一标识符
        prompt_token_ids: prompt 的 token ID 列表
        mm_features: 多模态特征规格列表
        sampling_params: 采样参数（温度、top_p 等）
        pooling_params: 池化参数（用于嵌入/重排序模型）
        block_ids: KV 缓存块 ID（每个注意力层组一个列表）
        num_computed_tokens: 已计算的 token 数（用于前缀缓存命中）
        lora_request: LoRA 适配器请求
        prompt_embeds: 预计算的 prompt 嵌入（可选）
        prompt_is_token_ids: 指示 prompt 是否为 token IDs（可选）
        prefill_token_ids: 仅用于 v2 模型运行器
    """
    req_id: str
    prompt_token_ids: list[int] | None
    mm_features: list[MultiModalFeatureSpec]
    sampling_params: SamplingParams | None
    pooling_params: PoolingParams | None
    block_ids: tuple[list[int], ...]
    num_computed_tokens: int
    lora_request: LoRARequest | None
    prompt_embeds: "torch.Tensor | None" = None
    prompt_is_token_ids: list[bool] | None = None

    # 仅用于 v2 模型运行器
    prefill_token_ids: list[int] | None = None

    @classmethod
    def from_request(
        cls,
        request: Request,
        block_ids: tuple[list[int], ...],
        prefill_token_ids: list[int] | None = None,
    ) -> "NewRequestData":
        """从 Request 对象创建 NewRequestData。

        Args:
            request: 源请求对象
            block_ids: 分配的 KV 缓存块 ID
            prefill_token_ids: 可选的预填充 token ID

        Returns:
            NewRequestData 实例
        """
        return cls(
            req_id=request.request_id,
            prompt_token_ids=request.prompt_token_ids,
            mm_features=request.mm_features,
            sampling_params=request.sampling_params,
            pooling_params=request.pooling_params,
            block_ids=block_ids,
            num_computed_tokens=request.num_computed_tokens,
            lora_request=request.lora_request,
            prompt_embeds=request.prompt_embeds,
            prompt_is_token_ids=request.prompt_is_token_ids,
            prefill_token_ids=prefill_token_ids,
        )

    def __repr__(self) -> str:
        prompt_embeds_shape = (
            self.prompt_embeds.shape if self.prompt_embeds is not None else None
        )
        return (
            f"NewRequestData("
            f"req_id={self.req_id},"
            f"prompt_token_ids={self.prompt_token_ids},"
            f"prefill_token_ids={self.prefill_token_ids},"
            f"mm_features={self.mm_features},"
            f"sampling_params={self.sampling_params},"
            f"block_ids={self.block_ids},"
            f"num_computed_tokens={self.num_computed_tokens},"
            f"lora_request={self.lora_request},"
            f"prompt_embeds_shape={prompt_embeds_shape}"
            ")"
        )

    def anon_repr(self) -> str:
        """返回 prompt 数据混淆后的表示（用于日志记录，避免泄露用户数据）。"""
        prompt_token_ids_len = (
            len(self.prompt_token_ids) if self.prompt_token_ids is not None else None
        )
        prompt_embeds_shape = (
            self.prompt_embeds.shape if self.prompt_embeds is not None else None
        )
        prefill_token_ids_len = (
            len(self.prefill_token_ids) if self.prefill_token_ids is not None else None
        )
        return (
            f"NewRequestData("
            f"req_id={self.req_id},"
            f"prompt_token_ids_len={prompt_token_ids_len},"
            f"prefill_token_ids_len={prefill_token_ids_len},"
            f"mm_features={self.mm_features},"
            f"sampling_params={self.sampling_params},"
            f"block_ids={self.block_ids},"
            f"num_computed_tokens={self.num_computed_tokens},"
            f"lora_request={self.lora_request},"
            f"prompt_embeds_shape={prompt_embeds_shape}"
            ")"
        )


@dataclass
class CachedRequestData:
    """
    已缓存请求的增量更新数据。

    与 NewRequestData 不同，此类不包含请求的完整数据。
    工作器已缓存了请求的初始数据，这里只传递每次调度步骤的差异信息。

    设计目的：最小化调度器与工作器之间的通信开销。

    属性：
        req_ids: 本次调度涉及的请求 ID 列表
        resumed_req_ids: 恢复的请求 ID 集合
            - 不在集合中的请求：new_block_ids 追加到现有块 ID
            - 在集合中的请求：new_block_ids 替换现有块 ID（请求被抢占后恢复）
        new_token_ids: 新生成的 token ID（仅用于流水线并行）
        all_token_ids: 未在上一步调度的请求的完整 token ID（用于连接器）
        new_block_ids: 新分配的 KV 缓存块 ID
        num_computed_tokens: 已计算的 token 数
        num_output_tokens: 已输出的 token 数
    """
    req_ids: list[str]
    # 对于不在 resumed_req_ids 中的请求，new_block_ids 将追加到现有块 ID。
    # 对于在集合中的请求，new_block_ids 将替换现有块 ID。
    resumed_req_ids: set[str]
    # 注意 (woosuk): new_token_ids 仅用于流水线并行。
    # 未使用 PP 时，new_token_ids 为空。
    new_token_ids: list[list[int]]
    # 对于上一步未调度的请求，将 token ID 传播到连接器。
    # 不包含上一步已调度的请求。
    all_token_ids: dict[str, list[int]]
    new_block_ids: list[tuple[list[int], ...] | None]
    num_computed_tokens: list[int]
    num_output_tokens: list[int]

    def anon_repr(self) -> str:
        """返回 token ID 混淆后的版本（用于日志记录）。"""
        new_token_ids_lens = [len(toks) for toks in self.new_token_ids]
        all_token_ids_lens = {
            req_id: len(toks) for req_id, toks in self.all_token_ids.items()
        }
        return (
            f"CachedRequestData("
            f"req_ids={self.req_ids},"
            f"resumed_req_ids={self.resumed_req_ids},"
            f"new_token_ids_lens={new_token_ids_lens},"
            f"all_token_ids_lens={all_token_ids_lens},"
            f"new_block_ids={self.new_block_ids},"
            f"num_computed_tokens={self.num_computed_tokens},"
            f"num_output_tokens={self.num_output_tokens}"
            f")"
        )

    def __repr__(self) -> str:
        return self.anon_repr()

    @property
    def num_reqs(self) -> int:
        """请求总数。"""
        return len(self.req_ids)

    @cached_property
    def _req_id_to_num_output_tokens(self) -> dict[str, int]:
        """请求 ID 到输出 token 数的缓存映射，O(1) 查找。

        此缓存属性是安全的，因为 CachedRequestData 实例在每个调度迭代中
        创建，在迭代详情计算期间不会被修改。
        """
        return dict(zip(self.req_ids, self.num_output_tokens))

    def is_context_phase(self, req_id: str) -> bool:
        """检查请求是否处于上下文（prefill）阶段。

        当输出 token 数为 0 时，请求处于上下文阶段。

        Args:
            req_id: 请求 ID

        Returns:
            True 表示请求处于上下文阶段
        """
        num_output_tokens = self._req_id_to_num_output_tokens.get(req_id)
        return num_output_tokens is not None and num_output_tokens == 0

    @classmethod
    def make_empty(cls) -> "CachedRequestData":
        """创建空的 CachedRequestData 实例。"""
        return cls(
            req_ids=[],
            resumed_req_ids=set(),
            new_token_ids=[],
            all_token_ids={},
            new_block_ids=[],
            num_computed_tokens=[],
            num_output_tokens=[],
        )


@dataclass
class SchedulerOutput:
    """
    调度器输出，每个调度步骤生成一个。

    这是调度器与模型运行器之间的主要通信数据结构。
    包含模型运行器执行一次前向传播所需的所有信息。

    属性：
        scheduled_new_reqs: 首次调度的新请求列表
        scheduled_cached_reqs: 已缓存请求的增量数据
        num_scheduled_tokens: 每个请求的调度 token 数
        total_num_scheduled_tokens: 所有请求的总调度 token 数
        scheduled_spec_decode_tokens: 投机解码 token（请求 ID -> token ID 列表）
        scheduled_encoder_inputs: 需要处理的编码器输入（请求 ID -> 输入索引列表）
        num_common_prefix_blocks: 每个 KV 缓存组的公共前缀块数（用于级联注意力）
        finished_req_ids: 上一步和当前步之间完成的请求 ID
        free_encoder_mm_hashes: 需要从编码器缓存释放的 mm_hash 列表
        preempted_req_ids: 本步被抢占的请求 ID（仅用于 v2 模型运行器）
        has_structured_output_requests: 是否有使用结构化输出的请求
        pending_structured_output_tokens: 是否有待处理的结构化输出 token
        num_invalid_spec_tokens: 无效投机 token 数（用于调整接受率计算）
        kv_connector_metadata: KV 连接器元数据
        ec_connector_metadata: EC 连接器元数据
        new_block_ids_to_zero: 需要清零的新分配块 ID
    """

    # 首次调度的新请求列表。
    # 工作器缓存请求数据，因此不需要每步重传。
    scheduled_new_reqs: list[NewRequestData]
    # 已调度过的请求的增量数据。
    # 由于数据已缓存在工作器中，仅发送差异以最小化通信开销。
    scheduled_cached_reqs: CachedRequestData

    # 请求 ID -> 调度 token 数
    num_scheduled_tokens: dict[str, int]
    # 所有请求的总调度 token 数
    total_num_scheduled_tokens: int
    # 请求 ID -> 投机解码 token ID 列表
    # 如果请求没有投机解码 token，不包含在字典中。
    scheduled_spec_decode_tokens: dict[str, list[int]]
    # 请求 ID -> 需要处理的编码器输入索引
    # 例如，如果请求有 [0, 1]，表示视觉编码器需要处理该请求的第 0 和第 1 张图像。
    scheduled_encoder_inputs: dict[str, list[int]]
    # 每个 KV 缓存组的公共前缀块数。
    # 可用于级联注意力 (cascade attention)。
    num_common_prefix_blocks: list[int]

    # 上一步和当前步之间完成的请求 ID。
    # 用于通知工作器已完成的请求，以便释放缓存状态。
    finished_req_ids: set[str]
    # 需要从编码器缓存释放的 mm_hash 字符串列表。
    free_encoder_mm_hashes: list[str]

    # 本步被抢占的请求 ID。
    # 仅用于 v2 模型运行器。
    preempted_req_ids: set[str] | None = None

    # 已调度的请求是否使用结构化输出。
    # 仅在异步调度情况下设置。
    has_structured_output_requests: bool = False

    # 已调度的请求是否拥有执行语法位掩码计算所需的所有输出 token。
    pending_structured_output_tokens: bool = False

    # 用于调整接受率计算。
    num_invalid_spec_tokens: dict[str, int] | None = None

    # KV 缓存连接器元数据
    kv_connector_metadata: KVConnectorMetadata | None = None

    # EC 缓存连接器元数据
    ec_connector_metadata: ECConnectorMetadata | None = None

    # 在此调度步骤中从池中新分配的块 ID。
    # 工作器在使用前将对应的 GPU 内存清零，
    # 防止过时的 NaN/数据破坏注意力或 SSM 计算。
    new_block_ids_to_zero: list[int] | None = None

    @classmethod
    def make_empty(cls) -> "SchedulerOutput":
        """创建空的 SchedulerOutput 实例。"""
        return cls(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=CachedRequestData.make_empty(),
            num_scheduled_tokens={},
            total_num_scheduled_tokens=0,
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=[],
            finished_req_ids=set(),
            free_encoder_mm_hashes=[],
        )


@dataclass
class GrammarOutput:
    """
    语法输出，用于结构化输出的语法验证。

    包含语法验证的位掩码，用于约束模型只能生成符合语法的 token。

    属性：
        structured_output_request_ids: 使用结构化输出的请求 ID 列表
        grammar_bitmask: 语法位掩码，按 structured_output_request_ids 顺序排列。
            每个位表示对应 token 是否符合语法约束。
    """
    # 结构化输出请求的 ID 列表
    structured_output_request_ids: list[str]
    # 按 structured_output_request_ids 顺序排列的位掩码
    grammar_bitmask: "npt.NDArray[np.int32]"
