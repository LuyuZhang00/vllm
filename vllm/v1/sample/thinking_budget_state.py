# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""思考预算状态管理模块。

本模块实现了对推理模型（如DeepSeek-R1等）"思考token"数量的预算控制。

背景:
    推理模型在生成答案之前，通常会先进行一段"思考"过程，生成中间推理token。
    这些思考token由特殊标记）包围。
    用户可以通过设置thinking_token_budget来限制思考token的数量，
    避免模型过度思考，节省计算资源和时间。

主要功能:
    1. 跟踪每个请求的思考状态（是否在思考中、已生成的思考token数量等）
    2. 当思考token数量超过预算时，强制输出思考结束token
    3. 支持投机解码场景下的思考预算控制

工作流程:
    1. 初始化: 从prompt中检测是否已经进入思考模式
    2. 同步: 每步更新batch中请求的状态（添加、删除、移动）
    3. 更新: 根据新生成的token更新思考状态
    4. 应用: 当预算耗尽时，在logits中强制设置思考结束token的概率为极高值

类:
    ThinkingBudgetStateHolder: 思考预算状态持有者，管理所有请求的思考状态
"""

from typing import TYPE_CHECKING, Any

import torch

from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.sample.logits_processor.interface import (
    BatchUpdate,
    MoveDirectionality,
)

if TYPE_CHECKING:
    from vllm.config.reasoning import ReasoningConfig


def maybe_create_thinking_budget_state_holder(
    reasoning_config: "ReasoningConfig | None",
    max_num_seqs: int,
    num_spec_tokens: int,
    device: torch.device,
    is_pin_memory: bool,
) -> "ThinkingBudgetStateHolder | None":
    """根据推理配置创建思考预算状态持有者。

    如果没有提供推理配置（reasoning_config为None），则返回None，表示不启用思考预算控制。

    参数:
        reasoning_config: 推理配置，包含思考开始/结束token IDs等
        max_num_seqs: 最大并发序列数
        num_spec_tokens: 投机解码的token数量
        device: 计算设备
        is_pin_memory: 是否使用pin memory

    返回:
        ThinkingBudgetStateHolder实例，或None（如果未启用推理）
    """
    if reasoning_config is None:
        return None
    return ThinkingBudgetStateHolder(
        reasoning_config, max_num_seqs, num_spec_tokens, device, is_pin_memory
    )


class ThinkingBudgetStateHolder:
    """思考预算状态持有者：跟踪思考部分并在预算超限时强制输出结束token。

    该类管理每个请求的思考状态，包括:
    - 是否正在思考模式中
    - 已生成的思考token数量
    - 剩余的思考预算
    - 需要强制输出的思考结束token位置

    属性:
        think_start_token_ids: 思考开始token IDs序列（如[151648]表示<think>）
        think_end_token_ids: 思考结束token IDs序列（如[151649]表示</think>）
        in_spec_mode: 是否处于投机解码模式
        num_spec_tokens: 投机解码的token数量
        is_enabled: 是否启用思考预算控制
        _state: 每个请求的思考状态字典 {req_index: state_dict}
        cu_num_tokens: 每个请求的累积token数量
        _mask_capacity: logits掩码容量
    """

    think_start_token_ids: list[int]
    think_end_token_ids: list[int]

    def __init__(
        self,
        reasoning_config: "ReasoningConfig | None",
        max_num_seqs: int,
        num_spec_tokens: int,
        device: torch.device,
        is_pin_memory: bool,
    ):
        """
        初始化思考预算状态持有者。

        参数:
            reasoning_config: 推理配置，包含思考开始/结束token IDs
            max_num_seqs: 最大并发序列数
            num_spec_tokens: 投机解码的token数量
            device: 计算设备
            is_pin_memory: 是否使用pin memory（此处保留用于API一致性）
        """
        _ = is_pin_memory  # API parity with logits processors
        max_num_reqs = max_num_seqs
        self.in_spec_mode = num_spec_tokens > 0
        self.num_spec_tokens = num_spec_tokens

        # 没有单独的enable标志: 非None的reasoning_config就是开关
        self.is_enabled = reasoning_config is not None

        if reasoning_config is None:
            self.think_start_token_ids = []
            self.think_end_token_ids = []
        else:
            rs = reasoning_config.reasoning_start_token_ids
            re = reasoning_config.reasoning_end_token_ids
            self.think_start_token_ids = rs if rs else []
            self.think_end_token_ids = re if re else []

        self.device = device
        # 每个请求的思考状态 {req_index: state_dict}
        self._state: dict[int, dict[str, Any]] = {}
        # 每个请求的累积token数量
        self.cu_num_tokens: dict[int, int] = {}

        # 计算logits掩码容量
        if self.num_spec_tokens > 0:
            self._mask_capacity = max_num_reqs * (self.num_spec_tokens + 1)
        else:
            self._mask_capacity = max_num_reqs

    def has_tracked_requests(self) -> bool:
        """检查是否有被跟踪的请求。

        当sync_batch为某个thinking_token_budget行设置了状态时返回True。
        用于决定采样是否需要输出token行和投机组合;
        与仅仅持有holder实例不同（推理可能开启但当前batch中没有预算请求）。

        返回:
            是否有被跟踪的请求
        """
        return bool(self._state)

    def sync_batch(self, batch_update: BatchUpdate | None) -> None:
        """同步batch状态变更：添加/删除/移动请求的状态。

        仅更新每请求的状态（不调用_update_think_state）。

        参数:
            batch_update: batch状态更新信息，包含添加、删除、移动的请求
        """
        if not self.is_enabled or not batch_update:
            return

        # 处理删除的请求
        for index in batch_update.removed:
            self._state.pop(index, None)

        # 处理添加的请求
        for index, params, prompt_tok_ids, output_tok_ids in batch_update.added:
            thinking_token_budget = params.thinking_token_budget
            if thinking_token_budget is not None:
                # 如果请求设置了思考token预算，初始化其状态
                self._state[index] = self._init_state_entry(
                    prompt_tok_ids, thinking_token_budget
                )
                self._state[index]["output_tok_ids"] = output_tok_ids
                self._state[index]["spec_token_ids"] = []
            else:
                # 如果没有设置预算，移除其状态
                self._state.pop(index, None)

        # 处理移动的请求（交换或单向移动）
        for i1, i2, direction in batch_update.moved:
            if direction == MoveDirectionality.SWAP:
                # 交换操作: 交换两个位置的状态
                state1 = self._state.get(i1)
                state2 = self._state.get(i2)
                if state1 is not None:
                    self._state[i2] = state1
                if state2 is not None:
                    self._state[i1] = state2
            else:
                # 单向移动: 将状态从i1移动到i2
                state = self._state.pop(i1, None)
                if state is not None:
                    self._state[i2] = state

    def update_state(
        self,
        output_token_ids: list[list[int]],
        spec_token_ids: list[list[int]] | None,
        repeat_indices: torch.Tensor | None = None,
    ) -> None:
        """更新思考状态：从采样行刷新输出/投机token并重新计算思考状态。

        参数:
            output_token_ids: 每个请求的输出token ID列表
            spec_token_ids: 每个请求的投机token ID列表（可选）
            repeat_indices: 重复索引（用于投机解码，将batch索引映射到token级别）
        """
        if not self.is_enabled or not self._state:
            return

        spec_lists = spec_token_ids or []
        # 如果有repeat_indices，找出每个请求的最后一个行索引
        last_row_for_req: dict[int, int] | None = None
        if repeat_indices is not None:
            last_row_for_req = {}
            rpt = repeat_indices.cpu().tolist()
            for batch_row, req_i in enumerate(rpt):
                last_row_for_req[req_i] = batch_row

        for seq_idx, state in list(self._state.items()):
            # 更新output_tok_ids
            if last_row_for_req is not None:
                output_row: int | None = last_row_for_req.get(seq_idx)
                if output_row is None or output_row >= len(output_token_ids):
                    continue
                state["output_tok_ids"] = output_token_ids[output_row]
            elif seq_idx >= len(output_token_ids):
                continue
            else:
                state["output_tok_ids"] = output_token_ids[seq_idx]

            # 更新spec_token_ids
            if seq_idx < len(spec_lists):
                state["spec_token_ids"] = list(spec_lists[seq_idx])
            else:
                state["spec_token_ids"] = []

            state["in_spec_mode"] = self.in_spec_mode
            state["force_index"] = []

            if len(state["output_tok_ids"]) > 0:
                spec_len = len(state["spec_token_ids"])
                # 仅当有投机token时才剥离草稿后缀;
                # `[:-0]`会清空整个列表（Python将stop index 0视为"直到空"）
                if spec_len > 0 and len(state["output_tok_ids"]) >= spec_len:
                    state["output_tok_ids"] = state["output_tok_ids"][:-spec_len]

            # 重新计算思考状态
            self._update_think_state(state)

    def apply_to_logits(
        self,
        logits: torch.Tensor,
        predict_bonus_token: bool,
        spec_token_ids: list[list[int]] | None,
    ) -> torch.Tensor:
        """对logits应用强制结束思考token的处理。

        当思考预算耗尽时，将思考结束token的logits值设为极大值，
        强制模型在下一步输出思考结束token。

        参数:
            logits: 输入logits张量 [batch_size, vocab_size]
            predict_bonus_token: 是否预测bonus token
            spec_token_ids: 投机token IDs列表

        返回:
            处理后的logits张量
        """
        if not self.is_enabled or not self._state:
            return logits
        spec_lists = spec_token_ids or []
        return self._apply_forcing_to_logits(logits, predict_bonus_token, spec_lists)

    @staticmethod
    def _find_last_sequence_index(target_list: list[int], token_ids: list[int]) -> int:
        """在目标列表中查找token_ids序列最后一次出现的位置。

        从后向前搜索，返回序列起始位置的索引。

        参数:
            target_list: 目标列表（prompt或output tokens）
            token_ids: 要查找的token ID序列

        返回:
            序列最后一次出现的起始索引，未找到返回-1
        """
        if not token_ids:
            return -1
        for i in range(len(target_list) - len(token_ids), -1, -1):
            if target_list[i : i + len(token_ids)] == token_ids:
                return i
        return -1

    def _init_state_entry(
        self, prompt_tok_ids: list[int] | None, thinking_token_budget: int
    ) -> dict[str, Any]:
        """初始化请求的思考状态。

        分析prompt中是否已包含思考开始/结束token，设置初始状态。

        参数:
            prompt_tok_ids: prompt的token IDs，可能为None
            thinking_token_budget: 思考token预算数量

        返回:
            初始状态字典
        """
        if prompt_tok_ids is None:
            # 没有prompt信息，使用默认值
            last_start = -1
            last_end = -1
            in_think = False
            think_count = 0
            start_thinking = -1
            countdown = thinking_token_budget
            continue_thinking = False
            in_end = False
        else:
            start_thinking = -1
            countdown = thinking_token_budget
            continue_thinking = False
            in_end = False
            # 查找prompt中最后的思考开始和结束token位置
            last_start = self._find_last_sequence_index(
                prompt_tok_ids, self.think_start_token_ids
            )
            last_end = self._find_last_sequence_index(
                prompt_tok_ids, self.think_end_token_ids
            )
            # 如果开始token在结束token之后，说明正在思考模式中
            in_think = last_start > last_end
            # 加载指标如思考计数、开始思考位置
            # 如果请求已经在思考模式中
            if in_think:
                think_count = len(prompt_tok_ids) - (
                    last_start + len(self.think_start_token_ids)
                )
                start_thinking = len(prompt_tok_ids) - think_count - 1
                countdown -= think_count
                continue_thinking = True
                # 检查token是否在prompt中已经耗尽
                token_exhausted = thinking_token_budget - think_count
                in_end = token_exhausted <= 0
            else:
                think_count = 0

        return {
            "in_think": in_think,           # 是否在思考模式中
            "in_end": in_end,               # 是否在结束模式中（预算耗尽，需要输出结束token）
            "check_count_down": countdown,  # 剩余思考预算倒计时
            "think_count": think_count,     # 已生成的思考token数量
            "end_count": 0,                 # 已输出的结束token数量
            "prompt_tok_ids": prompt_tok_ids,  # prompt token IDs
            "output_tok_ids": [],           # 输出token IDs
            "thinking_token_budget": thinking_token_budget,  # 思考token预算
            "prev_output_length": 0,        # 上一步的输出长度
            "spec_token_ids": [],           # 投机token IDs
            "force_index": [],              # 需要强制输出的位置索引
            "start_thinking": start_thinking,  # 思考开始位置
            "end_thinking": -1,             # 思考结束位置
            "in_spec_mode": False,          # 是否在投机模式中
            "bonus_token_forced": False,    # 是否已强制输出bonus token
            "continue_thinking": continue_thinking,  # 是否继续思考（从prompt延续）
        }

    def _update_think_state(self, state: dict[str, Any]) -> None:
        """更新单个请求的思考状态。

        根据新生成的token，更新思考模式状态、倒计时，并决定是否需要
        强制输出思考结束token。

        参数:
            state: 请求的思考状态字典
        """
        if state.get("thinking_token_budget", -1) == -1:
            return
        if len(self.think_end_token_ids) == 0:
            state["thinking_token_budget"] = -1
            state["in_end"] = False
            state["force_index"] = []
            return

        # 查找思考开始位置（如果尚未找到）
        if state["start_thinking"] == -1:
            start_thinking = self._find_last_sequence_index(
                state.get("output_tok_ids", []), self.think_start_token_ids
            )
            state["start_thinking"] = start_thinking
        # 查找思考结束位置（如果尚未找到）
        if state["end_thinking"] == -1:
            end_thinking = self._find_last_sequence_index(
                state.get("output_tok_ids", []), self.think_end_token_ids
            )
            state["end_thinking"] = end_thinking

        if state["start_thinking"] == -1:
            return

        # 计算本步采样的token数量
        if state["continue_thinking"]:
            sampled_tokens_from_previous_step = len(
                state.get("output_tok_ids", [])
            ) - state.get("prev_output_length", 0)
        else:
            if state["prev_output_length"] == 0:
                sampled_tokens_from_previous_step = len(
                    state.get("output_tok_ids", [])
                ) - len(self.think_start_token_ids)
            else:
                sampled_tokens_from_previous_step = (
                    len(state.get("output_tok_ids", [])) - state["prev_output_length"]
                )

        # 更新倒计时
        current_step_countdown = (
            state["check_count_down"] - sampled_tokens_from_previous_step
        )
        predicted_countdown = current_step_countdown - len(state["spec_token_ids"]) - 1

        # 仅当倒计时到0或更少且处于"思考中"模式时才继续处理
        if (
            not state.get("in_end", False)
            and predicted_countdown >= 0
            and state["start_thinking"] > -1
        ):
            state["check_count_down"] = current_step_countdown
            state["prev_output_length"] = len(state.get("output_tok_ids", []))
            return

        output = state.get("output_tok_ids", [])
        if not output:
            # 当在初始化时设置了in_end（budget=0，prompt已在思考中）时，
            # 必须强制第一个生成的token为结束token;
            # 否则apply()看到in_end=True但force_index=[]会允许额外的思考token。
            if state.get("in_end", False):
                state["force_index"] = [0]
            return

        # 跟踪上一步输出长度用于增量处理
        prev_length = state.get("prev_output_length", 0)
        current_length = len(output)

        if current_length <= prev_length:
            # 输出没有增长（可能被拒绝采样器拒绝了）
            if state.get("in_end", False):
                remaining_budget = state["thinking_token_budget"] - state["think_count"]
                spec_len = len(state["spec_token_ids"])
                if spec_len > 0:
                    if 0 < remaining_budget < spec_len:
                        state["force_index"] = [remaining_budget]
                    elif remaining_budget <= 0:
                        state["force_index"] = [0]
                    else:
                        state["force_index"] = [spec_len]
                else:
                    state["force_index"] = [0]
            return

        state["prev_output_length"] = current_length

        start_len = len(self.think_start_token_ids)
        absolute_start_pos = state["start_thinking"]

        if state["continue_thinking"] and state["end_thinking"] > -1:
            absolute_end_pos = state["end_thinking"] + len(
                state.get("prompt_tok_ids") or []
            )
        else:
            absolute_end_pos = state["end_thinking"]

        # 更新状态（基于最近的序列）
        # 这是结束模式但拒绝采样器在结束token之前拒绝了token的情况，
        # 需要回到思考模式等待下一个结束token
        # 例如: 999是结束token [2,4,5,999] -> [3,-1,-1,-1]
        if state["in_end"] and state["end_count"] == 0:
            new_tokens = output[prev_length:]
            stopping_thinking = (
                self.think_end_token_ids[state["end_count"]] in new_tokens
            )
            if not stopping_thinking:
                state["in_think"] = True
                state["in_end"] = False
                state["end_count"] = 0
                state["bonus_token_forced"] = False

        if not state["in_end"]:
            if absolute_start_pos >= 0 and absolute_end_pos >= 0:
                # 情况: ...<end>...<start>... - 进入思考模式
                if absolute_start_pos > absolute_end_pos:
                    new_think_count = current_length - (absolute_start_pos + start_len)
                    state["in_think"] = True
                    state["think_count"] = new_think_count
                else:
                    # 情况: ...<start>...<end>... - 退出思考模式
                    state["in_think"] = False
                    state["think_count"] = 0

            elif absolute_start_pos >= 0 and not state["continue_thinking"]:
                # 找到思考开始标记 - 进入思考模式
                new_think_count = current_length - (absolute_start_pos + start_len)
                state["in_think"] = True
                state["think_count"] = new_think_count

            elif absolute_end_pos >= 0:
                # 找到思考结束标记 - 退出思考模式
                state["in_think"] = False
                state["think_count"] = 0

            elif state["in_think"]:
                # 继续思考模式，增加新token的计数
                prompt_tok_ids = state.get("prompt_tok_ids") or []
                think_tokens_in_prompt = len(prompt_tok_ids) - (
                    absolute_start_pos + start_len
                )
                state["think_count"] = (
                    len(state["output_tok_ids"]) + think_tokens_in_prompt
                )

            if state["in_think"]:
                remaining_budget = max(
                    0, state["thinking_token_budget"] - state["think_count"]
                )
                state["check_count_down"] = remaining_budget
            else:
                state["check_count_down"] = state["thinking_token_budget"]

            total_thinking_tokens = (
                state["think_count"] + len(state["spec_token_ids"]) + 1
            )
            # 检查是否需要转换到结束模式
            # 如果思考token数量超过预算，需要转换到结束模式
            if (
                state["in_think"]
                and total_thinking_tokens > state["thinking_token_budget"]
            ):
                # 计算force_index: 投机token中开始强制的位置。
                # 如果已经超出预算（不含投机token），从位置0开始强制。
                # 从预算被超出的位置开始强制。
                state["in_think"] = False
                state["in_end"] = True
                state["end_count"] = 0
                state["check_count_down"] = state["thinking_token_budget"]
                remaining_budget = state["thinking_token_budget"] - state["think_count"]
                spec_len = len(state["spec_token_ids"])
                if 0 < remaining_budget < spec_len:
                    state["force_index"] = [remaining_budget]

                elif remaining_budget <= 0:
                    state["force_index"] = [0]

                else:
                    # remaining_budget >= spec_len: 所有投机token都在预算内;
                    # 强制bonus token位置
                    state["force_index"] = [len(state["spec_token_ids"])]

        else:
            # 在结束模式中，逐步输出思考结束token
            state["force_index"] = []
            if len(state["spec_token_ids"]) > 0:
                for i, token_id in enumerate(state["spec_token_ids"]):
                    if state["end_count"] + 1 < len(self.think_end_token_ids):
                        if token_id == self.think_end_token_ids[state["end_count"] + 1]:
                            state["end_count"] += 1
                        else:
                            state["end_count"] += 1
                            state["force_index"] = [i]
                            break
                    else:
                        state["end_count"] += 1
                if len(state["force_index"]) == 0:
                    state["end_count"] += 1
                    state["force_index"] = [len(state["spec_token_ids"])]
            else:
                state["end_count"] += 1
                state["force_index"] = [0]
            # 如果所有结束token都已输出，退出结束模式
            if state["end_count"] >= len(self.think_end_token_ids):
                state.update(
                    {
                        "in_end": False,
                        "end_count": 0,
                        "check_count_down": state["thinking_token_budget"],
                    }
                )

    def _apply_forcing_to_logits(
        self,
        logits: torch.Tensor,
        predict_bonus_token: bool,
        spec_token_ids_for_layout: list[list[int]],
    ) -> torch.Tensor:
        """将强制结束思考token的处理应用到logits。

        当需要强制输出思考结束token时，在对应的logits位置设置极大的值（1e9），
        使该token在采样时几乎必然被选中。

        参数:
            logits: 输入logits张量
            predict_bonus_token: 是否预测bonus token
            spec_token_ids_for_layout: 投机token IDs列表（用于计算布局偏移）

        返回:
            处理后的logits张量
        """
        cumulative_total = 0
        self.cu_num_tokens.clear()

        n_layout = len(spec_token_ids_for_layout)
        if self._state:
            n_layout = max(n_layout, max(self._state.keys()) + 1)

        # 计算每个请求在logits张量中的起始位置
        for index in range(n_layout):
            self.cu_num_tokens[index] = cumulative_total
            spec_tokens = (
                spec_token_ids_for_layout[index]
                if index < len(spec_token_ids_for_layout)
                else []
            )
            if self.in_spec_mode:
                cumulative_total += len(spec_tokens) if not predict_bonus_token else 1
            else:
                cumulative_total += 1

        # 在CPU上构建活跃索引和强制token列表，避免每次迭代的标量同步写入GPU张量
        active_indices_cpu: list[int] = []
        force_tokens_cpu: list[int] = []

        for seq_idx in sorted(self._state.keys()):
            if seq_idx not in self.cu_num_tokens:
                continue
            state = self._state[seq_idx]
            if state.get("in_end", False):
                # logits处理器在投机模式下被调用两次:
                # 一次用于bonus token logits，一次用于target logits
                # 如果force_index是bonus token索引，则将其改为0
                if predict_bonus_token:
                    if state.get("force_index") and state["force_index"][0] < len(
                        state["spec_token_ids"]
                    ):
                        continue
                    else:
                        state["force_index"] = [0]
                # 继续强制输出结束思考token
                if state["end_count"] > 0:
                    state["bonus_token_forced"] = False
                if state and not state["bonus_token_forced"]:
                    force_index = state.get("force_index", [])
                    if len(force_index) == 0:
                        continue
                    end_count = state.get("end_count", 0)
                    for force_idx in force_index:
                        if end_count < len(self.think_end_token_ids):
                            mask_idx = self.cu_num_tokens[seq_idx] + force_idx
                            if (
                                mask_idx < self._mask_capacity
                                and mask_idx < logits.shape[0]
                            ):
                                active_indices_cpu.append(mask_idx)
                                force_tokens_cpu.append(
                                    self.think_end_token_ids[end_count]
                                )
                            if predict_bonus_token:
                                if state["end_count"] > 0:
                                    state["bonus_token_forced"] = False
                                    state["force_index"] = []
                                else:
                                    state["bonus_token_forced"] = True

        if active_indices_cpu:
            device = logits.device
            # 异步传输到GPU，避免CPU-GPU同步
            active_indices = async_tensor_h2d(
                active_indices_cpu, dtype=torch.long, device=device
            )
            force_tokens = async_tensor_h2d(
                force_tokens_cpu, dtype=torch.long, device=device
            )
            # 避免CPU->GPU同步
            fill = logits.new_full((len(active_indices_cpu),), 1e9)
            # 在指定位置设置极大值，强制采样该token
            logits.index_put_((active_indices, force_tokens), fill)

        return logits
