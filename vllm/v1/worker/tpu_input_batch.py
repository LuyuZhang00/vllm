# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Datastructures defining a TPU input batch
"""
TPU 输入批次管理模块

=============================================================
【模块概述】
=============================================================
本模块定义了 TPU 后端的 InputBatch 类，用于管理一个推理批次（batch）
中所有请求的状态数据。它是 TPU 模型运行器的核心数据结构，负责：
1. 存储和管理批次中所有请求的 token ID 序列
2. 维护采样参数（temperature、top_p、top_k、min_p 等）
3. 管理 block table（KV Cache 物理块映射）
4. 支持 LoRA 适配器的请求级映射
5. 支持请求的添加、移除和状态压缩（condense）

=============================================================
【核心设计思想】
=============================================================
1. 结构化批处理（Structured Batching）：
   - 所有请求的同类数据存储在连续的张量/数组中（SoA 布局），
     而非为每个请求单独存储（AoS 布局）。
   - 例如，所有请求的 temperature 存储在一个一维张量中，
     通过 req_index 索引访问。这种布局对 TPU/GPU 向量化计算更友好。

2. CPU-Device 双缓冲（Double Buffering）：
   - 每个采样参数都有两个版本：
     - `xxx_cpu_tensor` / `xxx_cpu`：CPU 端缓冲，支持 pin_memory 以便
       高效传输到设备。
     - `xxx`：设备端张量，供模型推理使用。
   - 调度阶段在 CPU 端修改数据，推理前传输到设备。

3. 压缩（Condense）机制：
   - 当请求从批次中移除时，会产生空洞（空索引）。
   - condense() 方法将后面的请求移动到空位，保持批次紧凑。
   - 空洞索引按降序排列，便于从后往前高效填充。
"""

from typing import cast

import numpy as np
import torch

from vllm.lora.request import LoRARequest
from vllm.sampling_params import SamplingType
from vllm.utils import length_from_prompt_token_ids_or_embeds
from vllm.utils.collection_utils import swap_dict_values
from vllm.v1.outputs import LogprobsTensors
from vllm.v1.worker.block_table import MultiGroupBlockTable
from vllm.v1.worker.gpu_input_batch import CachedRequestState

# 采样参数的极小阈值，用于判断 min_p 是否大于零（考虑浮点精度）
_SAMPLING_EPS = 1e-5


class InputBatch:
    """
    TPU 输入批次管理器

    ==========================================================
    【类职责】
    ==========================================================
    管理一个推理批次中所有请求的状态，包括：
    1. 请求标识（req_id -> req_index 的映射）
    2. Token ID 序列（prompt + output）
    3. 采样参数（temperature, top_p, top_k, min_p, penalties 等）
    4. Block table（逻辑块到物理块的映射）
    5. LoRA 适配器映射
    6. 日志概率（logprobs）和 logit 偏置

    ==========================================================
    【关键数据布局】
    ==========================================================
    - req_index：请求在批次中的索引（0 到 max_num_reqs - 1）
    - _req_ids[req_index]：该索引处的请求 ID
    - req_id_to_index[req_id]：请求 ID 到索引的反向映射
    - token_ids_cpu[req_index, :]：该请求的完整 token 序列
    - temperature_cpu[req_index]：该请求的温度参数
    - 以此类推...

    ==========================================================
    【生命周期】
    ==========================================================
    1. add_request()：将新请求加入批次的指定位置
    2. remove_request()：标记请求位置为空（产生空洞）
    3. condense()：压缩批次，消除空洞
    4. swap_states()：交换两个索引位置的状态（用于排序等场景）
    """

    def __init__(
        self,
        max_num_reqs: int,
        max_model_len: int,
        max_num_batched_tokens: int,
        device: torch.device,
        pin_memory: bool,
        vocab_size: int,
        block_sizes: list[int],  # The block_size of each kv cache group
        kernel_block_sizes: list[int],
    ):
        """
        初始化输入批次管理器。

        参数说明：
            max_num_reqs: 批次中最大请求数
            max_model_len: 模型支持的最大序列长度
            max_num_batched_tokens: 批次中最大 token 总数
            device: 目标设备（TPU/CPU）
            pin_memory: 是否将 CPU 缓冲区固定在内存中（便于 DMA 传输）
            vocab_size: 词表大小，用于 pad token 和 top_k 限制
            block_sizes: 每个 KV Cache 组的块大小列表
            kernel_block_sizes: Attention kernel 使用的块大小列表
        """
        self.max_num_reqs = max_num_reqs
        self.max_model_len = max_model_len
        self.max_num_batched_tokens = max_num_batched_tokens
        self.device = device
        self.pin_memory = pin_memory
        self.vocab_size = vocab_size

        # 请求 ID 管理：
        # _req_ids 列表中 None 表示该位置为空（请求已移除或从未使用）
        self._req_ids: list[str | None] = []
        # 请求 ID 到批次索引的映射，用于 O(1) 查找请求位置
        self.req_id_to_index: dict[str, int] = {}

        # ====================================================
        # Token ID 存储区域
        # ====================================================
        # CPU 端的 token ID 缓冲区，形状 [max_num_reqs, max_model_len]
        # 注意：当 max_model_len 很大时，这个缓冲区可能占用大量 CPU 内存
        # 此缓冲区不直接传输到设备，因此不需要 pin_memory
        # TODO(woosuk): This buffer could be too large if max_model_len is big.
        # Find a way to reduce the CPU memory usage.
        # This buffer is not directly transferred to the GPU, so it does not
        # need to be pinned.
        self.token_ids_cpu_tensor = torch.zeros(
            (max_num_reqs, max_model_len),
            device="cpu",
            dtype=torch.int32,
            pin_memory=False,
        )
        # 使用 numpy 视图以便高效地按索引写入单个请求的 token
        self.token_ids_cpu = self.token_ids_cpu_tensor.numpy()
        # 每个请求的非投机解码 token 数（prompt + 已生成的 output）
        self.num_tokens_no_spec = np.zeros(max_num_reqs, dtype=np.int32)
        # 每个请求的 prompt token 数
        self.num_prompt_tokens = np.zeros(max_num_reqs, dtype=np.int32)
        # 每个请求已计算（已缓存 KV）的 token 数
        self.num_computed_tokens_cpu_tensor = torch.zeros(
            (max_num_reqs,),
            device="cpu",
            dtype=torch.int32,
            pin_memory=pin_memory,
        )
        self.num_computed_tokens_cpu = self.num_computed_tokens_cpu_tensor.numpy()

        # ====================================================
        # Block Table（块表）：逻辑块 -> 物理块映射
        # ====================================================
        # 支持多组 KV Cache（如 attention sinks + sliding window）
        self.block_table = MultiGroupBlockTable(
            max_num_reqs=max_num_reqs,
            max_model_len=max_model_len,
            max_num_batched_tokens=max_num_batched_tokens,
            pin_memory=pin_memory,
            device=device,
            block_sizes=block_sizes,
            kernel_block_sizes=kernel_block_sizes,
        )

        # ====================================================
        # 采样参数区域（CPU-Device 双缓冲模式）
        # ====================================================
        # 每个采样参数都有三份：
        #   1. xxx（设备端张量）：用于设备上的采样计算
        #   2. xxx_cpu_tensor（CPU 端张量）：支持 pin_memory 的缓冲
        #   3. xxx_cpu（numpy 视图）：便于 Python 端逐元素修改
        # xxx_reqs 集合：记录使用了非默认值的请求 ID，用于快速判断
        # 批次中是否需要应用该采样策略

        # --- Temperature（温度）---
        # 用于控制采样随机性。值越小越确定，0.0 表示贪心解码
        self.temperature = torch.empty(
            (max_num_reqs,), dtype=torch.float32, device=device
        )
        self.temperature_cpu_tensor = torch.empty(
            (max_num_reqs,), dtype=torch.float32, device="cpu", pin_memory=pin_memory
        )
        self.temperature_cpu = self.temperature_cpu_tensor.numpy()
        # 贪心解码的请求集合
        self.greedy_reqs: set[str] = set()
        # 随机采样的请求集合
        self.random_reqs: set[str] = set()

        # --- Top-P（核采样）---
        # 仅从概率累积达到 top_p 的 token 中采样
        self.top_p = torch.empty((max_num_reqs,), dtype=torch.float32, device=device)
        self.top_p_cpu_tensor = torch.empty(
            (max_num_reqs,), dtype=torch.float32, device="cpu", pin_memory=pin_memory
        )
        self.top_p_cpu = self.top_p_cpu_tensor.numpy()
        self.top_p_reqs: set[str] = set()

        # --- Top-K ---
        # 仅从概率最高的 K 个 token 中采样
        self.top_k = torch.empty((max_num_reqs,), dtype=torch.int32, device=device)
        self.top_k_cpu_tensor = torch.empty(
            (max_num_reqs,), dtype=torch.int32, device="cpu", pin_memory=pin_memory
        )
        self.top_k_cpu = self.top_k_cpu_tensor.numpy()
        self.top_k_reqs: set[str] = set()

        # --- Min-P ---
        # 仅从概率不低于最高概率 * min_p 的 token 中采样
        self.min_p = torch.empty((max_num_reqs,), dtype=torch.float32, device=device)
        self.min_p_cpu_tensor = torch.empty(
            (max_num_reqs,), dtype=torch.float32, device="cpu", pin_memory=pin_memory
        )
        self.min_p_cpu = self.min_p_cpu_tensor.numpy()
        self.min_p_reqs: set[str] = set()

        # --- Frequency Penalty（频率惩罚）---
        # 根据 token 出现频率进行惩罚，出现越多惩罚越大
        self.frequency_penalties = torch.empty(
            (max_num_reqs,), dtype=torch.float, device=device
        )
        self.frequency_penalties_cpu_tensor = torch.empty(
            (max_num_reqs,), dtype=torch.float, device="cpu", pin_memory=pin_memory
        )
        self.frequency_penalties_cpu = self.frequency_penalties_cpu_tensor.numpy()
        self.frequency_penalties_reqs: set[str] = set()

        # --- Presence Penalty（存在惩罚）---
        # 只要 token 出现过就施加固定惩罚，鼓励生成新 token
        self.presence_penalties = torch.empty(
            (max_num_reqs,), dtype=torch.float, device=device
        )
        self.presence_penalties_cpu_tensor = torch.empty(
            (max_num_reqs,), dtype=torch.float, device="cpu", pin_memory=pin_memory
        )
        self.presence_penalties_cpu = self.presence_penalties_cpu_tensor.numpy()
        self.presence_penalties_reqs: set[str] = set()

        # --- Repetition Penalty（重复惩罚）---
        # 对已出现的 token 施加乘性惩罚，值 > 1.0 时抑制重复
        self.repetition_penalties = torch.empty(
            (max_num_reqs,), dtype=torch.float, device=device
        )
        self.repetition_penalties_cpu_tensor = torch.empty(
            (max_num_reqs,), dtype=torch.float, device="cpu", pin_memory=pin_memory
        )
        self.repetition_penalties_cpu = self.repetition_penalties_cpu_tensor.numpy()
        self.repetition_penalties_reqs: set[str] = set()

        # ====================================================
        # Min Tokens 约束
        # ====================================================
        # req_index -> (min_tokens, stop_token_ids)
        # 确保模型至少生成 min_tokens 个 token 后才允许停止
        # stop_token_ids 记录该请求的停止 token 集合
        self.min_tokens: dict[int, tuple[int, set[int]]] = {}

        # ====================================================
        # LoRA 适配器映射
        # ====================================================
        # 请求级别的 LoRA ID 映射（req_index -> lora_int_id）
        self.request_lora_mapping = np.zeros((self.max_num_reqs,), dtype=np.int64)
        # LoRA ID -> 使用该 LoRA 的请求 ID 集合
        self.lora_id_to_request_ids: dict[int, set[str]] = {}
        # LoRA ID -> LoRA 请求对象（包含权重路径等信息）
        self.lora_id_to_lora_request: dict[int, LoRARequest] = {}

        # ====================================================
        # 随机数生成器
        # ====================================================
        # req_index -> torch.Generator
        # 每个请求可以有自己独立的随机数生成器（用于可复现采样）
        # 注意：没有自定义生成器的请求不应包含在此字典中
        self.generators: dict[int, torch.Generator] = {}

        # ====================================================
        # Logprobs（对数概率）相关
        # ====================================================
        # 请求 ID -> 需要返回的 logprobs 数量
        self.num_logprobs: dict[str, int] = {}

        # 用于在多个 prefill 步骤中累积 prompt logprobs 的张量块
        # 键为请求 ID，值为正在累积的 LogprobsTensors
        self.in_progress_prompt_logprobs_cpu: dict[str, LogprobsTensors] = {}

        # ====================================================
        # Logit Bias 和 Token 限制
        # ====================================================
        # 每个请求的 logit 偏置（req_index -> {token_id: bias}）
        self.logit_bias: list[dict[int, float] | None] = [None] * max_num_reqs
        # 使用了 allowed_token_ids 限制的请求 ID 集合
        self.has_allowed_token_ids: set[str] = set()
        # Token 允许掩码：True 表示该 token 被禁止（填 -inf），False 表示允许
        # 注意：这里使用 masked_fill_ 来设置 -inf，所以 False 才是"允许"
        self.allowed_token_ids_mask: torch.Tensor | None = None
        self.allowed_token_ids_mask_cpu_tensor: torch.Tensor | None = None

        # ====================================================
        # Bad Words（禁止词）相关
        # ====================================================
        # req_index -> 禁止的 token 序列列表
        self.bad_words_token_ids: dict[int, list[list[int]]] = {}

        # 每个请求的输出 token ID 历史（用于状态恢复等）
        self.req_output_token_ids: list[list[int] | None] = []

    @property
    def req_ids(self) -> list[str]:
        """获取请求 ID 列表（去除 None 占位符）。

        注意：None 元素仅在状态更新过程中暂时存在，
        正常使用时不应有 None 值。
        """
        # None elements should only be present transiently
        # while performing state updates to the batch.
        return cast(list[str], self._req_ids)

    def add_request(
        self,
        request: "CachedRequestState",
        req_index: int | None = None,
    ) -> None:
        """
        将一个缓存的请求状态添加到批次中。

        ==========================================================
        【流程说明】
        ==========================================================
        1. 确定请求在批次中的索引位置（新的追加到末尾，或复用已有位置）
        2. 复制 prompt token 和已生成的 output token 到 token_ids_cpu
        3. 设置采样参数（temperature, top_p, top_k 等）
        4. 处理 LoRA 映射、logprobs、logit_bias 等附加信息

        参数说明：
            request: 缓存的请求状态，包含所有请求信息
            req_index: 目标索引位置。None 表示追加到末尾。
        """
        if req_index is None:
            req_index = self.num_reqs
        assert req_index < self.max_num_reqs

        req_id = request.req_id
        # 在列表末尾追加或复用已有位置
        if req_index == len(self._req_ids):
            self._req_ids.append(req_id)
            self.req_output_token_ids.append(request.output_token_ids)
        else:
            self._req_ids[req_index] = req_id
            self.req_output_token_ids[req_index] = request.output_token_ids

        self.req_id_to_index[req_id] = req_index

        # 复制 prompt token ID 和 output token ID 到 CPU 缓冲区
        # prompt_token_ids 和 prompt_embeds 互斥，用辅助函数获取长度
        num_prompt_tokens = length_from_prompt_token_ids_or_embeds(
            request.prompt_token_ids, request.prompt_embeds
        )
        # TODO: copy prompt_embeds
        self.num_prompt_tokens[req_index] = num_prompt_tokens
        self.token_ids_cpu[req_index, :num_prompt_tokens] = request.prompt_token_ids
        start_idx = num_prompt_tokens
        end_idx = start_idx + len(request.output_token_ids)
        self.token_ids_cpu[req_index, start_idx:end_idx] = request.output_token_ids
        # 记录不含投机解码 token 的总 token 数
        self.num_tokens_no_spec[req_index] = request.num_tokens

        # 设置已计算的 token 数和 block table
        self.num_computed_tokens_cpu[req_index] = request.num_computed_tokens
        self.block_table.add_row(request.block_ids, req_index)

        # ====================================================
        # 处理采样参数
        # ====================================================
        sampling_params = request.sampling_params
        assert sampling_params is not None, "pooling requests not supported yet"
        if sampling_params.sampling_type == SamplingType.GREEDY:
            # 贪心解码：temperature 设为 0 以避免后续除零错误
            # Should avoid division by zero later when apply_temperature.
            self.temperature_cpu[req_index] = 0.0
            self.greedy_reqs.add(req_id)
        else:
            self.temperature_cpu[req_index] = sampling_params.temperature
            self.random_reqs.add(req_id)

        self.top_p_cpu[req_index] = sampling_params.top_p
        if sampling_params.top_p < 1:
            self.top_p_reqs.add(req_id)
        top_k = sampling_params.top_k
        if 0 < top_k < self.vocab_size:
            self.top_k_reqs.add(req_id)
        else:
            top_k = self.vocab_size
        self.top_k_cpu[req_index] = top_k
        self.min_p_cpu[req_index] = sampling_params.min_p
        self.frequency_penalties_cpu[req_index] = sampling_params.frequency_penalty
        if sampling_params.min_p > _SAMPLING_EPS:
            self.min_p_reqs.add(req_id)
        if sampling_params.frequency_penalty != 0.0:
            self.frequency_penalties_reqs.add(req_id)
        self.presence_penalties_cpu[req_index] = sampling_params.presence_penalty
        if sampling_params.presence_penalty != 0.0:
            self.presence_penalties_reqs.add(req_id)
        self.repetition_penalties_cpu[req_index] = sampling_params.repetition_penalty
        if sampling_params.repetition_penalty != 1.0:
            self.repetition_penalties_reqs.add(req_id)
        if sampling_params.min_tokens:
            self.min_tokens[req_index] = (
                sampling_params.min_tokens,
                sampling_params.all_stop_token_ids,
            )

        # NOTE(woosuk): self.generators should not include the requests that
        # do not have their own generator.
        if request.generator is not None:
            self.generators[req_index] = request.generator

        # 处理 logprobs、logit_bias、allowed_token_ids、bad_words 等
        if sampling_params.logprobs is not None:
            self.num_logprobs[req_id] = sampling_params.logprobs
        if sampling_params.logit_bias is not None:
            self.logit_bias[req_index] = sampling_params.logit_bias

        if sampling_params.allowed_token_ids:
            self.has_allowed_token_ids.add(req_id)
            if self.allowed_token_ids_mask_cpu_tensor is None:
                # 延迟分配此大张量（max_num_reqs x vocab_size），仅在需要时分配
                # Lazy allocation for this tensor, which can be large.
                # False means we don't fill with -inf.
                self.allowed_token_ids_mask = torch.zeros(
                    self.max_num_reqs,
                    self.vocab_size,
                    dtype=torch.bool,
                    device=self.device,
                )
                self.allowed_token_ids_mask_cpu_tensor = torch.zeros(
                    self.max_num_reqs, self.vocab_size, dtype=torch.bool, device="cpu"
                )
            self.allowed_token_ids_mask_cpu_tensor[req_index] = True
            # 将允许的 token 位置设为 False（不填充 -inf）
            # False means we don't fill with -inf.
            self.allowed_token_ids_mask_cpu_tensor[req_index][
                sampling_params.allowed_token_ids
            ] = False

        if sampling_params.bad_words_token_ids:
            self.bad_words_token_ids[req_index] = sampling_params.bad_words_token_ids

        # ====================================================
        # 处理 LoRA 适配器映射
        # ====================================================
        if request.lora_request:
            lora_id = request.lora_request.lora_int_id
            if lora_id not in self.lora_id_to_request_ids:
                self.lora_id_to_request_ids[lora_id] = set()

            self.request_lora_mapping[req_index] = lora_id
            self.lora_id_to_request_ids[lora_id].add(request.req_id)
            self.lora_id_to_lora_request[lora_id] = request.lora_request
        else:
            # No LoRA
            self.request_lora_mapping[req_index] = 0

    def remove_request(self, req_id: str) -> int | None:
        """
        从批次中移除一个请求。

        注意：调用此方法后必须调用 condense() 来消除空洞。
        否则批次中会存在 None 占位符，影响后续操作。

        参数：
            req_id: 要移除的请求 ID

        返回：
            被移除请求在批次中的索引，如果请求不存在则返回 None
        """
        """This method must always be followed by a call to condense()."""

        req_index = self.req_id_to_index.pop(req_id, None)
        if req_index is None:
            return None
        # 将位置标记为空
        self._req_ids[req_index] = None
        self.req_output_token_ids[req_index] = None

        # 从各类采样请求集合中移除
        self.greedy_reqs.discard(req_id)
        self.random_reqs.discard(req_id)
        self.top_p_reqs.discard(req_id)
        self.top_k_reqs.discard(req_id)
        self.min_p_reqs.discard(req_id)
        self.min_tokens.pop(req_index, None)
        self.frequency_penalties_reqs.discard(req_id)
        self.presence_penalties_reqs.discard(req_id)
        self.repetition_penalties_reqs.discard(req_id)
        self.generators.pop(req_index, None)
        self.num_logprobs.pop(req_id, None)
        self.in_progress_prompt_logprobs_cpu.pop(req_id, None)

        # 清理 LoRA 映射
        lora_id = self.request_lora_mapping[req_index]
        if lora_id != 0:
            self.lora_id_to_request_ids[lora_id].discard(req_id)
            if len(self.lora_id_to_request_ids[lora_id]) == 0:
                self.lora_id_to_request_ids.pop(lora_id)
                self.lora_id_to_lora_request.pop(lora_id)
            self.request_lora_mapping[req_index] = 0

        self.logit_bias[req_index] = None
        self.has_allowed_token_ids.discard(req_id)
        if self.allowed_token_ids_mask_cpu_tensor is not None:
            # 重置掩码（全部设为 False = 不填充 -inf）
            # False means we don't fill with -inf.
            self.allowed_token_ids_mask_cpu_tensor[req_index].fill_(False)
        self.bad_words_token_ids.pop(req_index, None)
        return req_index

    def swap_states(self, i1: int, i2: int) -> None:
        """
        交换批次中两个索引位置的所有状态。

        这个操作用于重排序请求（例如按优先级排序），
        交换后两个位置的请求 ID 和所有关联数据互换。

        参数：
            i1, i2: 要交换的两个批次索引
        """
        old_id_i1 = self._req_ids[i1]
        old_id_i2 = self._req_ids[i2]
        self._req_ids[i1], self._req_ids[i2] = self._req_ids[i2], self._req_ids[i1]  # noqa
        self.req_output_token_ids[i1], self.req_output_token_ids[i2] = (
            self.req_output_token_ids[i2],
            self.req_output_token_ids[i1],
        )
        assert old_id_i1 is not None and old_id_i2 is not None
        self.req_id_to_index[old_id_i1], self.req_id_to_index[old_id_i2] = (
            self.req_id_to_index[old_id_i2],
            self.req_id_to_index[old_id_i1],
        )
        self.num_tokens_no_spec[i1], self.num_tokens_no_spec[i2] = (
            self.num_tokens_no_spec[i2],
            self.num_tokens_no_spec[i1],
        )
        self.num_prompt_tokens[i1], self.num_prompt_tokens[i2] = (
            self.num_prompt_tokens[i2],
            self.num_prompt_tokens[i1],
        )
        self.num_computed_tokens_cpu[i1], self.num_computed_tokens_cpu[i2] = (
            self.num_computed_tokens_cpu[i2],
            self.num_computed_tokens_cpu[i1],
        )
        self.temperature_cpu[i1], self.temperature_cpu[i2] = (
            self.temperature_cpu[i2],
            self.temperature_cpu[i1],
        )
        self.top_p_cpu[i1], self.top_p_cpu[i2] = self.top_p_cpu[i2], self.top_p_cpu[i1]
        self.top_k_cpu[i1], self.top_k_cpu[i2] = self.top_k_cpu[i2], self.top_k_cpu[i1]
        self.frequency_penalties_cpu[i1], self.frequency_penalties_cpu[i2] = (
            self.frequency_penalties_cpu[i2],
            self.frequency_penalties_cpu[i1],
        )
        self.presence_penalties_cpu[i1], self.presence_penalties_cpu[i2] = (
            self.presence_penalties_cpu[i2],
            self.presence_penalties_cpu[i1],
        )
        self.repetition_penalties_cpu[i1], self.repetition_penalties_cpu[i2] = (
            self.repetition_penalties_cpu[i2],
            self.repetition_penalties_cpu[i1],
        )
        self.min_p_cpu[i1], self.min_p_cpu[i2] = self.min_p_cpu[i2], self.min_p_cpu[i1]

        # 注意：不能直接用 Python 解包交换 numpy 数组行
        # 因为 numpy 切片返回的是视图，会导致数据损坏
        # NOTE: the following is unsafe
        # self.token_ids_cpu[i1, ...], self.token_ids_cpu[i2, ...], =\
        #     self.token_ids_cpu[i2, ...], self.token_ids_cpu[i1, ...]
        # instead, we need to temporarily copy the data for one of the indices
        # TODO(lucas): optimize this by only copying valid indices
        tmp = self.token_ids_cpu[i1, ...].copy()
        self.token_ids_cpu[i1, ...] = self.token_ids_cpu[i2, ...]
        self.token_ids_cpu[i2, ...] = tmp

        # 交换字典中的值（generators, min_tokens, bad_words）
        swap_dict_values(self.generators, i1, i2)
        swap_dict_values(self.min_tokens, i1, i2)
        swap_dict_values(self.bad_words_token_ids, i1, i2)

        self.request_lora_mapping[i1], self.request_lora_mapping[i2] = (
            self.request_lora_mapping[i2],
            self.request_lora_mapping[i1],
        )
        self.logit_bias[i1], self.logit_bias[i2] = (
            self.logit_bias[i2],
            self.logit_bias[i1],
        )

        if self.allowed_token_ids_mask_cpu_tensor is not None:
            (
                self.allowed_token_ids_mask_cpu_tensor[i1],
                self.allowed_token_ids_mask_cpu_tensor[i2],
            ) = (
                self.allowed_token_ids_mask_cpu_tensor[i2],
                self.allowed_token_ids_mask_cpu_tensor[i1],
            )
        self.block_table.swap_row(i1, i2)

    def condense(self, empty_req_indices: list[int]) -> None:
        """Move non-empty requests down into lower, empty indices.

        Args:
          empty_req_indices: empty batch indices, sorted descending.
        """
        """
        压缩批次，消除空洞。

        将非空请求向下移动到较低的空索引位置，使批次保持紧凑。
        这是为了避免批次中存在大量空洞导致资源浪费。

        ==========================================================
        【算法说明】
        ==========================================================
        1. 传入的 empty_req_indices 按降序排列
        2. 使用双指针：last_req_index 指向最后一个非空请求，
           empty_index 指向当前最小的空位
        3. 将 last_req_index 处的状态移动到 empty_index 处
        4. 重复直到所有空位都被填充或所有非空请求都已处理

        参数：
            empty_req_indices: 空批次索引列表，按降序排列
        """
        num_reqs = self.num_reqs
        if num_reqs == 0:
            # 批次中没有请求，清空所有状态
            # The batched states are empty.
            self._req_ids.clear()
            self.req_output_token_ids.clear()
            return

        # NOTE(woosuk): This function assumes that the empty_req_indices
        # is sorted in descending order.
        last_req_index = num_reqs + len(empty_req_indices) - 1
        while empty_req_indices:
            # 从后往前找到最大的非空索引
            while last_req_index in empty_req_indices:
                last_req_index -= 1

            # 弹出最小的空索引
            empty_index = empty_req_indices.pop()
            if empty_index >= last_req_index:
                break

            # 将 last_req_index 处的完整状态移动到 empty_index 处
            req_id = self._req_ids[last_req_index]
            output_token_ids = self.req_output_token_ids[last_req_index]
            assert req_id is not None
            self._req_ids[empty_index] = req_id
            self._req_ids[last_req_index] = None
            self.req_output_token_ids[empty_index] = output_token_ids
            self.req_output_token_ids[last_req_index] = None
            self.req_id_to_index[req_id] = empty_index

            # 迁移 token 数据和其他采样参数
            num_tokens = self.num_tokens_no_spec[last_req_index]
            self.token_ids_cpu[empty_index, :num_tokens] = self.token_ids_cpu[
                last_req_index, :num_tokens
            ]
            self.num_tokens_no_spec[empty_index] = self.num_tokens_no_spec[
                last_req_index
            ]
            self.num_prompt_tokens[empty_index] = self.num_prompt_tokens[last_req_index]
            self.num_computed_tokens_cpu[empty_index] = self.num_computed_tokens_cpu[
                last_req_index
            ]
            self.block_table.move_row(last_req_index, empty_index)
            self.temperature_cpu[empty_index] = self.temperature_cpu[last_req_index]
            self.top_p_cpu[empty_index] = self.top_p_cpu[last_req_index]
            self.top_k_cpu[empty_index] = self.top_k_cpu[last_req_index]
            self.frequency_penalties_cpu[empty_index] = self.frequency_penalties_cpu[
                last_req_index
            ]
            self.presence_penalties_cpu[empty_index] = self.presence_penalties_cpu[
                last_req_index
            ]
            self.repetition_penalties_cpu[empty_index] = self.repetition_penalties_cpu[
                last_req_index
            ]
            self.min_p_cpu[empty_index] = self.min_p_cpu[last_req_index]
            generator = self.generators.pop(last_req_index, None)
            if generator is not None:
                self.generators[empty_index] = generator

            min_token = self.min_tokens.pop(last_req_index, None)
            if min_token is not None:
                self.min_tokens[empty_index] = min_token

            self.request_lora_mapping[empty_index] = self.request_lora_mapping[
                last_req_index
            ]

            self.logit_bias[empty_index] = self.logit_bias[last_req_index]

            if self.allowed_token_ids_mask_cpu_tensor is not None:
                self.allowed_token_ids_mask_cpu_tensor[empty_index] = (
                    self.allowed_token_ids_mask_cpu_tensor[last_req_index]
                )

            bad_words_token_ids = self.bad_words_token_ids.pop(last_req_index, None)
            if bad_words_token_ids is not None:
                self.bad_words_token_ids[empty_index] = bad_words_token_ids
            # 递减 last_req_index，因为该位置现在已空
            # Decrement last_req_index since it is now empty.
            last_req_index -= 1

        # 截断列表，移除尾部多余的元素
        # Trim lists to the batch size.
        del self._req_ids[self.num_reqs :]
        del self.req_output_token_ids[self.num_reqs :]

    def _make_prompt_token_ids_tensor(self) -> torch.Tensor:
        """
        生成 prompt token ID 张量，用于发送给模型。

        创建一个形状为 [num_reqs, max_prompt_len] 的张量，
        其中 max_prompt_len 是当前批次中所有请求的 prompt 长度最大值。
        超出实际 prompt 长度的位置用 vocab_size 填充（作为 padding）。

        返回：
            在目标设备上的 prompt token ID 张量
        """
        max_prompt_len = self.num_prompt_tokens[: self.num_reqs].max()
        prompt_token_ids_cpu_tensor = torch.empty(
            (self.num_reqs, max_prompt_len),
            device="cpu",
            dtype=torch.int64,
            pin_memory=self.pin_memory,
        )
        prompt_token_ids = prompt_token_ids_cpu_tensor.numpy()
        prompt_token_ids[:] = self.token_ids_cpu[: self.num_reqs, :max_prompt_len]
        # 用 vocab_size 作为填充值（vocab_size 不是合法的 token ID）
        # Use the value of vocab_size as a pad since we don't have a
        # token_id of this value.
        for i in range(self.num_reqs):
            prompt_token_ids[i, self.num_prompt_tokens[i] :] = self.vocab_size
        return prompt_token_ids_cpu_tensor.to(device=self.device, non_blocking=True)

    def make_lora_inputs(
        self, num_scheduled_tokens: np.ndarray, num_sampled_tokens: np.ndarray
    ) -> tuple[tuple[int, ...], tuple[int, ...], set[LoRARequest]]:
        """
        Given the num_scheduled_tokens for each request in the batch, return
        datastructures used to activate the current LoRAs.
        Returns:
            1. prompt_lora_mapping: A tuple of size self.num_reqs where,
               prompt_lora_mapping[i] is the LoRA id to use for the ith prompt.
            2. token_lora_mapping: A tuple of size np.sum(num_scheduled_tokens)
               where, token_lora_mapping[i] is the LoRA id to use for ith token.
            3. lora_requests: Set of relevant LoRA requests.
        """
        """
        构建 LoRA 适配器的输入映射数据结构。

        为当前批次中的每个请求和每个 token 确定应该使用哪个 LoRA 适配器。
        这些映射会被传递给 LoRA 层，以便在前向传播时应用正确的适配器权重。

        参数：
            num_scheduled_tokens: 每个请求在本迭代中计划处理的 token 数
            num_sampled_tokens: 每个请求在本迭代中采样的 token 数

        返回：
            1. prompt_lora_mapping: 大小为 num_reqs 的元组，
               prompt_lora_mapping[i] 是第 i 个请求使用的 LoRA ID
            2. token_lora_mapping: 大小为所有 token 总数的元组，
               每个 token 都标注了对应的 LoRA ID
            3. lora_requests: 当前活跃的 LoRA 请求集合
        """

        req_lora_mapping = self.request_lora_mapping[: self.num_reqs]
        prompt_lora_mapping = tuple(req_lora_mapping)
        token_lora_mapping = tuple(req_lora_mapping.repeat(num_scheduled_tokens))
        active_lora_requests: set[LoRARequest] = set(
            self.lora_id_to_lora_request.values()
        )

        return prompt_lora_mapping, token_lora_mapping, active_lora_requests

    @property
    def num_reqs(self) -> int:
        """当前批次中的请求数量（不含空洞）。"""
        return len(self.req_id_to_index)

    @property
    def all_greedy(self) -> bool:
        """如果批次中所有请求都是贪心解码，则返回 True。"""
        return len(self.random_reqs) == 0

    @property
    def all_random(self) -> bool:
        """如果批次中所有请求都是随机采样，则返回 True。"""
        return len(self.greedy_reqs) == 0

    @property
    def no_top_p(self) -> bool:
        """如果批次中没有请求使用 top_p 采样，则返回 True。"""
        return len(self.top_p_reqs) == 0

    @property
    def no_top_k(self) -> bool:
        """如果批次中没有请求使用 top_k 采样，则返回 True。"""
        return len(self.top_k_reqs) == 0

    @property
    def no_min_p(self) -> bool:
        """如果批次中没有请求使用 min_p 采样，则返回 True。"""
        return len(self.min_p_reqs) == 0

    @property
    def no_penalties(self) -> bool:
        """如果批次中没有任何请求使用惩罚参数，则返回 True。"""
        return (
            len(self.presence_penalties_reqs) == 0
            and len(self.frequency_penalties_reqs) == 0
            and len(self.repetition_penalties_reqs) == 0
        )

    @property
    def max_num_logprobs(self) -> int | None:
        """返回批次中所有请求所需的最大 logprobs 数量。"""
        return max(self.num_logprobs.values()) if self.num_logprobs else None

    @property
    def no_allowed_token_ids(self) -> bool:
        """如果批次中没有请求使用 allowed_token_ids 限制，则返回 True。"""
        return len(self.has_allowed_token_ids) == 0
