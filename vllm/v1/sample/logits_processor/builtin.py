# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""内置logits处理器模块。

本模块实现了vLLM内置的三个logits处理器:

1. MinPLogitsProcessor (min_p处理器):
    - 基于概率阈值过滤低概率token
    - 计算方式: 将概率低于 max_prob * min_p 的token设为-inf
    - 是argmax不变的（不影响贪心采样结果）
    - 适用于: 生成更高质量的随机采样结果

2. LogitBiasLogitsProcessor (logit偏置处理器):
    - 对指定token的logits添加偏置值
    - 正偏置增加token被选中的概率，负偏置降低
    - 不是argmax不变的（可能改变贪心采样结果）
    - 适用于: 引导模型生成特定token或避免特定token

3. MinTokensLogitsProcessor (最少token数处理器):
    - 在生成指定数量的token之前，禁止输出停止token（如EOS）
    - 不是argmax不变的（通过抑制停止token改变贪心采样结果）
    - 适用于: 确保模型生成至少指定数量的token
    - 支持投机解码模式

辅助函数:
    process_dict_updates: 通用的batch状态更新工具函数
"""

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, TypeVar

import numpy as np
import torch

from vllm import SamplingParams
from vllm.v1.sample.logits_processor.interface import (
    BatchUpdate,
    LogitsProcessor,
    MoveDirectionality,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig

T = TypeVar("T")


class MinPLogitsProcessor(LogitsProcessor):
    """Min-P logits处理器: 基于概率阈值过滤低概率token。

    工作原理:
    1. 将logits转换为概率分布 (softmax)
    2. 找出每个序列的最大概率 max_prob
    3. 计算阈值: threshold = max_prob * min_p
    4. 将概率低于阈值的token的logits设为 -inf

    效果:
    - min_p = 0.0: 不过滤任何token（禁用）
    - min_p = 0.5: 过滤掉概率低于最大概率50%的token
    - min_p = 1.0: 只保留概率等于最大概率的token（类似贪心）

    该处理器是argmax不变的，因为最大概率的token永远不会被过滤掉。
    因此它仅影响随机采样，不影响贪心采样。

    属性:
        min_p_count: 当前batch中使用min_p的请求数量
        min_p_cpu_tensor: CPU上的min_p值张量（使用pin memory加速传输）
        min_p_cpu: min_p_cpu_tensor的numpy视图（用于高效的CPU端更新）
        min_p_device: GPU上的min_p值张量
        min_p: 当前batch大小的min_p切片
        use_double_tensor: 是否使用双张量模式（CPU+GPU）
    """

    def __init__(
        self, vllm_config: "VllmConfig", device: torch.device, is_pin_memory: bool
    ):
        """
        初始化Min-P处理器。

        参数:
            vllm_config: vLLM配置
            device: 计算设备
            is_pin_memory: 是否使用pin memory
        """
        max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        self.min_p_count: int = 0

        # 在CPU上预分配min_p张量（使用pin memory加速CPU->GPU传输）
        self.min_p_cpu_tensor = torch.zeros(
            (max_num_reqs,), dtype=torch.float32, device="cpu", pin_memory=is_pin_memory
        )
        self.min_p_cpu = self.min_p_cpu_tensor.numpy()

        # 判断是否需要双张量模式（CPU+GPU）
        self.use_double_tensor = torch.device(device).type != "cpu"

        if self.use_double_tensor:
            # 在GPU上预分配设备张量
            self.min_p_device: torch.Tensor = torch.empty(
                (max_num_reqs,), dtype=torch.float32, device=device
            )
        else:
            self.min_p_device = self.min_p_cpu_tensor
        # 当前batch大小的设备张量切片
        self.min_p: torch.Tensor = self.min_p_device[:0]

    def is_argmax_invariant(self) -> bool:
        """Min-p永远不会影响贪心采样。

        因为最大概率的token永远不会被min_p过滤掉。

        返回:
            True（argmax不变）
        """
        return True

    def get_min_p_by_index(self, index: int) -> float:
        """获取指定索引的min_p值。

        参数:
            index: 请求索引

        返回:
            该请求的min_p值
        """
        return float(self.min_p_cpu[index])

    def update_state(self, batch_update: BatchUpdate | None):
        """更新batch状态。

        处理请求的添加、删除和移动，更新min_p值。

        参数:
            batch_update: batch状态更新信息
        """
        if not batch_update:
            return

        needs_update = False
        # 处理添加的请求
        for index, params, _, _ in batch_update.added:
            min_p = params.min_p
            min_p_before = self.min_p_cpu[index]
            if min_p_before != min_p:
                needs_update = True
                self.min_p_cpu[index] = min_p
                if min_p and not min_p_before:
                    self.min_p_count += 1
                elif not min_p and min_p_before:
                    self.min_p_count -= 1

        if self.min_p_count:
            # 处理删除的请求
            if batch_update.removed:
                needs_update = True
                for index in batch_update.removed:
                    if self.min_p_cpu[index]:
                        self.min_p_cpu[index] = 0
                        self.min_p_count -= 1

            # 处理移动的请求（单向移动和交换）
            for adx, bdx, direct in batch_update.moved:
                min_p_a, min_p_b = self.min_p_cpu[adx], self.min_p_cpu[bdx]
                if min_p_a != min_p_b:
                    needs_update = True
                    self.min_p_cpu[bdx] = min_p_a
                    if direct == MoveDirectionality.SWAP:
                        self.min_p_cpu[adx] = min_p_b
                if direct == MoveDirectionality.UNIDIRECTIONAL:
                    if min_p_a:
                        self.min_p_cpu[adx] = 0
                    if min_p_b:
                        self.min_p_count -= 1

        # 如果需要更新，同步到GPU
        size = batch_update.batch_size
        if self.min_p_count and (needs_update or self.min_p.shape[0] != size):
            self.min_p = self.min_p_device[:size]
            if self.use_double_tensor:
                # 非阻塞复制到GPU
                self.min_p.copy_(self.min_p_cpu_tensor[:size], non_blocking=True)
            self.min_p.unsqueeze_(1)  # 扩展维度用于广播

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        """应用min-p过滤到logits。

        步骤:
        1. 将logits转换为概率分布
        2. 找出每个序列的最大概率
        3. 计算调整后的min_p阈值 = max_prob * min_p
        4. 将低于阈值的token的logits设为-inf

        参数:
            logits: 输入logits张量 [batch_size, vocab_size]

        返回:
            过滤后的logits张量（原地修改）
        """
        if not self.min_p_count:
            return logits

        # 将logits转换为概率分布
        probability_values = torch.nn.functional.softmax(logits, dim=-1)
        # 计算每个序列的最大概率
        max_probabilities = torch.amax(probability_values, dim=-1, keepdim=True)
        # 调整min_p阈值
        adjusted_min_p = max_probabilities.mul_(self.min_p)
        # 识别无效token（概率低于阈值）
        invalid_token_mask = probability_values < adjusted_min_p
        # 使用布尔索引应用掩码
        logits.masked_fill_(invalid_token_mask, -float("inf"))
        return logits


class LogitBiasLogitsProcessor(LogitsProcessor):
    """Logit偏置处理器: 对指定token的logits添加偏置值。

    工作原理:
    - 用户可以通过sampling_params.logit_bias指定每个token的偏置值
    - 偏置值会直接加到对应token的logits上
    - 正偏置增加token被选中的概率，负偏置降低

    示例:
        logit_bias = {100: 5.0, 200: -3.0}
        - token 100的logits增加5.0
        - token 200的logits减少3.0

    该处理器不是argmax不变的，因为偏置可能改变最大logit对应的token。
    因此它会影响贪心采样结果。

    属性:
        device: 计算设备
        pin_memory: 是否使用pin memory
        biases: 请求索引到token偏置字典的映射
        bias_tensor: 偏置值张量（GPU上）
        logits_slice: (请求索引张量, token ID张量) 用于高级索引
    """

    def __init__(self, _, device: torch.device, is_pin_memory: bool):
        """
        初始化Logit偏置处理器。

        参数:
            _: vLLM配置（未使用）
            device: 计算设备
            is_pin_memory: 是否使用pin memory
        """
        self.device = device
        self.pin_memory = is_pin_memory
        # 请求索引 -> {token_id: bias_value}
        self.biases: dict[int, dict[int, float]] = {}

        self.bias_tensor: torch.Tensor = torch.tensor(())
        self.logits_slice = (
            self._device_tensor([], torch.int32),
            self._device_tensor([], torch.int32),
        )

    def is_argmax_invariant(self) -> bool:
        """Logit偏置可以重新平衡token概率，改变贪心采样中argmax的结果。

        返回:
            False（非argmax不变）
        """
        return False

    def update_state(self, batch_update: BatchUpdate | None):
        """更新batch状态。

        处理请求的添加、删除和移动，更新偏置信息。

        参数:
            batch_update: batch状态更新信息
        """
        needs_update = process_dict_updates(
            self.biases, batch_update, lambda params, _, __: params.logit_bias or None
        )

        # 如果需要更新，重建张量
        if needs_update:
            reqs: list[int] = []
            tok_ids: list[int] = []
            biases: list[float] = []
            for req, lb in self.biases.items():
                reqs.extend([req] * len(lb))
                tok_ids.extend(lb.keys())
                biases.extend(lb.values())

            self.bias_tensor = self._device_tensor(biases, torch.float32)
            self.logits_slice = (
                self._device_tensor(reqs, torch.int32),
                self._device_tensor(tok_ids, torch.int32),
            )

    def _device_tensor(self, data: list, dtype: torch.dtype) -> torch.Tensor:
        """创建设备张量（通过CPU pin memory非阻塞传输到GPU）。

        参数:
            data: 数据列表
            dtype: 张量数据类型

        返回:
            设备上的张量
        """
        return torch.tensor(
            data, device="cpu", dtype=dtype, pin_memory=self.pin_memory
        ).to(device=self.device, non_blocking=True)

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        """应用logit偏置到logits。

        使用高级索引将偏置值加到指定位置。

        参数:
            logits: 输入logits张量 [batch_size, vocab_size]

        返回:
            应用偏置后的logits张量（原地修改）
        """
        if self.biases:
            logits[self.logits_slice] += self.bias_tensor
        return logits


class MinTokensLogitsProcessor(LogitsProcessor):
    """最少token数处理器: 在生成指定数量的token之前禁止停止token。

    工作原理:
    - 对于设置了min_tokens的请求，在生成的token数量达到min_tokens之前，
      将所有停止token（包括EOS）的logits设为-inf
    - 当生成的token数量达到min_tokens后，不再抑制停止token

    该处理器不是argmax不变的，因为它通过抑制停止token来改变贪心采样结果。

    属性:
        device: 计算设备
        pin_memory: 是否使用pin memory
        min_toks: 请求索引到(min_tokens, output_token_ids, stop_token_ids)的映射
        logits_slice: (请求索引张量, 停止token ID张量) 用于高级索引
        neg_inf_tensor: 负无穷张量
    """

    def __init__(
        self, vllm_config: "VllmConfig", device: torch.device, is_pin_memory: bool
    ):
        """
        初始化最少token数处理器。

        参数:
            vllm_config: vLLM配置
            device: 计算设备
            is_pin_memory: 是否使用pin memory
        """
        # 请求索引 -> (min_tokens, output_token_ids, stop_token_ids)
        self.device = device
        self.pin_memory = is_pin_memory
        self.min_toks: dict[int, tuple[int, Sequence[int], set[int]]] = {}

        # (req_idx_tensor, eos_tok_id_tensor)
        self.logits_slice: tuple[torch.Tensor, torch.Tensor] = (
            self._device_tensor([], torch.int32),
            self._device_tensor([], torch.int32),
        )

        self.neg_inf_tensor = torch.tensor(
            -float("inf"), dtype=torch.float32, device=self.device
        )

    def is_argmax_invariant(self) -> bool:
        """通过审查停止token，min-tokens可以改变贪心采样中argmax操作的结果。

        返回:
            False（非argmax不变）
        """
        return False

    @staticmethod
    def add_request(
        params: SamplingParams, _: list[int] | None, output_tok_ids: list[int]
    ) -> tuple[int, Sequence[int], set[int]] | None:
        """为新请求创建状态。

        如果请求没有设置min_tokens，或已生成的token数量已达到min_tokens，
        返回None（不需要应用该处理器）。

        参数:
            params: 采样参数
            _: prompt token IDs（未使用）
            output_tok_ids: 已生成的输出token IDs

        返回:
            (min_tokens, output_token_ids, stop_token_ids) 或 None
        """
        min_tokens = params.min_tokens
        if not min_tokens or len(output_tok_ids) >= min_tokens:
            return None
        return min_tokens, output_tok_ids, params.all_stop_token_ids

    def update_state(self, batch_update: BatchUpdate | None):
        """更新batch状态。

        处理请求的添加、删除和移动，检查哪些请求已达到min_tokens。

        参数:
            batch_update: batch状态更新信息
        """
        needs_update = process_dict_updates(
            self.min_toks, batch_update, self.add_request
        )
        if self.min_toks:
            # 检查是否有请求已达到min_tokens
            to_remove = tuple(
                index
                for index, (min_toks, out_tok_ids, _) in self.min_toks.items()
                if len(out_tok_ids) >= min_toks
            )
            if to_remove:
                needs_update = True
                for index in to_remove:
                    del self.min_toks[index]

        # 如果需要更新，重建张量
        if needs_update:
            reqs: list[int] = []
            tok_ids: list[int] = []
            for req, (_, _, stop_tok_ids) in self.min_toks.items():
                reqs.extend([req] * len(stop_tok_ids))
                tok_ids.extend(stop_tok_ids)

            self.logits_slice = (
                self._device_tensor(reqs, torch.int32),
                self._device_tensor(tok_ids, torch.int32),
            )

    def _device_tensor(self, data: list, dtype: torch.dtype) -> torch.Tensor:
        """创建设备张量（通过CPU pin memory非阻塞传输到GPU）。

        参数:
            data: 数据列表
            dtype: 张量数据类型

        返回:
            设备上的张量
        """
        return torch.tensor(
            data, device="cpu", dtype=dtype, pin_memory=self.pin_memory
        ).to(device=self.device, non_blocking=True)

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        """应用最少token数限制到logits。

        对于未达到min_tokens的请求，将其所有停止token的logits设为-inf。

        参数:
            logits: 输入logits张量 [batch_size, vocab_size]

        返回:
            处理后的logits张量（原地修改）
        """
        if self.min_toks:
            # 抑制未达到最小长度的请求的EOS token
            logits.index_put_(self.logits_slice, self.neg_inf_tensor)
        return logits

    def apply_with_spec_decode(
        self,
        logits: torch.Tensor,
        num_draft_tokens: list[int],
    ) -> torch.Tensor:
        """投机解码版本的apply()。

        优先级: min_tokens > stop_token_ids / EOS

        例如: num_draft_tokens = [2, 3, 1]
          -> logits形状 [6, V]，cumsum = [0, 2, 5, 6]
          -> 请求0拥有行0-1，请求1拥有行2-4，请求2拥有行5

        在投机解码中，每个请求可能有多个草稿token，需要对每个草稿位置
        独立判断是否需要抑制停止token。

        参数:
            logits: 输入logits张量 [total_tokens, vocab_size]
            num_draft_tokens: 每个请求的草稿token数量列表

        返回:
            处理后的logits张量（原地修改）
        """
        if not self.min_toks:
            return logits

        num_draft_arr = np.array(num_draft_tokens, dtype=np.int64)
        cumsum = np.concatenate([[0], np.cumsum(num_draft_arr)])

        entries = [
            (req_idx, min_tok, len(out_tok_ids), list(stop_tok_ids))
            for req_idx, (min_tok, out_tok_ids, stop_tok_ids) in self.min_toks.items()
            if stop_tok_ids
        ]

        if not entries:
            return logits

        all_rows: list[np.ndarray] = []  # 需要掩码的行索引
        all_toks: list[np.ndarray] = []  # 对应的停止token IDs

        for req_idx, min_tok, current_len, stop_toks in entries:
            remaining = min_tok - current_len
            # 需要掩码停止token的前导草稿位置数量
            n_mask = int(min(max(remaining, 0), num_draft_arr[req_idx]))

            if n_mask > 0:
                offset = cumsum[req_idx]
                row_indices = np.arange(offset, offset + n_mask, dtype=np.int64)
                n_stop = len(stop_toks)
                all_rows.append(np.repeat(row_indices, n_stop))
                all_toks.append(np.tile(stop_toks, n_mask))

        if all_rows:
            rows_arr = np.concatenate(all_rows)
            toks_arr = np.concatenate(all_toks)
            # (row_indices, token_indices) 用于index_put_设置-inf
            logits_slice = (
                torch.from_numpy(rows_arr).to(self.device, non_blocking=True),
                torch.from_numpy(toks_arr).to(self.device, non_blocking=True),
            )
            logits.index_put_(logits_slice, self.neg_inf_tensor)

        return logits


def process_dict_updates(
    req_entries: dict[int, T],
    batch_update: BatchUpdate | None,
    new_state: Callable[[SamplingParams, list[int] | None, list[int]], T | None],
) -> bool:
    """通用的batch状态更新工具函数。

    用于稀疏LogitsProcessor的dict状态更新。处理请求的添加、删除和移动。

    更新顺序（与BatchUpdate约定一致）:
    1. 处理添加的请求: 调用new_state创建新状态
    2. 处理删除的请求: 从字典中移除
    3. 处理移动的请求: 单向移动或交换

    参数:
        req_entries: 请求索引到状态的映射字典
        batch_update: batch状态更新信息
        new_state: 创建新状态的回调函数
            参数: (SamplingParams, prompt_token_ids, output_token_ids)
            返回: 新状态或None（如果不需要跟踪）

    返回:
        是否有任何更新
    """
    if not batch_update:
        # 无事可做
        return False

    updated = False
    # 处理添加的请求
    for index, params, prompt_tok_ids, output_tok_ids in batch_update.added:
        if (state := new_state(params, prompt_tok_ids, output_tok_ids)) is not None:
            req_entries[index] = state
            updated = True
        elif req_entries.pop(index, None) is not None:
            updated = True

    if req_entries:
        # 处理删除的请求
        for index in batch_update.removed:
            if req_entries.pop(index, None):
                updated = True

        # 处理移动的请求: 单向移动(a->b)和交换(a<->b)
        for a_index, b_index, direct in batch_update.moved:
            a_entry = req_entries.pop(a_index, None)
            b_entry = req_entries.pop(b_index, None)
            if a_entry is not None:
                req_entries[b_index] = a_entry
                updated = True
            if b_entry is not None:
                updated = True
                if direct == MoveDirectionality.SWAP:
                    req_entries[a_index] = b_entry

    return updated
