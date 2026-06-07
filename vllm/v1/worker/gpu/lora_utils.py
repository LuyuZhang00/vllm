# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
LoRA（低秩适配）状态管理工具模块。

本模块负责跟踪和管理每个请求关联的 LoRA 适配器，并在每一步迭代中
生成模型前向传播所需的 LoRA 输入映射。

工作流程概述：
  1. 当新请求到达时，通过 add_request() 记录该请求使用的 LoRA 适配器。
  2. 每步调度时，通过 make_lora_inputs() 将请求级别的 LoRA 信息展开为
     token 级别的映射，供模型的 LoRA 层使用。
  3. 请求结束时，通过 remove_request() 清理状态。
"""

import numpy as np

from vllm.lora.request import LoRARequest

# 特殊 ID 值，表示该请求没有使用任何 LoRA 适配器。
NO_LORA_ID = 0


class LoraState:
    """
    LoRA 状态管理器。

    职责：
      - 维护一个固定大小的数组 lora_ids，索引对应请求槽位（request slot），
        值为该请求使用的 LoRA 适配器的整数 ID。
      - 维护 req_id 到 LoRARequest 的映射字典，用于在需要时查找完整的
        LoRA 请求信息。

    属性：
      lora_ids (np.ndarray): 形状为 (max_num_reqs,) 的 int32 数组，
          每个元素存储对应请求槽位的 LoRA ID，未使用的槽位填充 NO_LORA_ID。
      lora_requests (dict[str, LoRARequest]): 请求 ID 到 LoRARequest 对象的映射，
          仅存储当前活跃的使用了 LoRA 的请求。
    """

    def __init__(self, max_num_reqs: int):
        """
        初始化 LoRA 状态管理器。

        参数：
          max_num_reqs: 最大并发请求数，决定 lora_ids 数组的大小。
        """
        # 创建固定大小的 LoRA ID 数组，初始全部填充为 NO_LORA_ID（无 LoRA）。
        self.lora_ids = np.zeros(max_num_reqs, dtype=np.int32)
        self.lora_ids.fill(NO_LORA_ID)
        # 请求 ID -> LoRARequest 对象的映射字典。
        self.lora_requests: dict[str, LoRARequest] = {}

    def add_request(
        self, req_id: str, req_index: int, lora_request: LoRARequest | None
    ) -> None:
        """
        注册一个新请求的 LoRA 信息。

        当新请求被调度到 GPU Worker 时调用此方法，将请求的 LoRA ID 写入
        对应的槽位。

        参数：
          req_id:      请求的唯一标识符。
          req_index:   请求在批处理中的槽位索引（对应 lora_ids 数组的下标）。
          lora_request: LoRARequest 对象，如果请求不使用 LoRA 则为 None。
        """
        if lora_request is not None:
            # 请求使用了 LoRA，记录 LoRARequest 对象并设置对应的 LoRA ID。
            self.lora_requests[req_id] = lora_request
            self.lora_ids[req_index] = lora_request.lora_int_id
        else:
            # 请求不使用 LoRA，设置为 NO_LORA_ID。
            self.lora_ids[req_index] = NO_LORA_ID

    def remove_request(self, req_id: str) -> None:
        """
        移除一个已结束请求的 LoRA 信息。

        当请求完成或被取消时调用，从映射字典中删除对应的条目。
        注意：lora_ids 数组中的值会在该槽位被新请求复用时被覆盖，
        因此此处无需显式清除。

        参数：
          req_id: 要移除的请求的唯一标识符。
        """
        self.lora_requests.pop(req_id, None)

    def make_lora_inputs(
        self,
        req_ids: list[str],
        idx_mapping: np.ndarray,
        num_scheduled_tokens: np.ndarray,
    ) -> tuple[tuple[int, ...], tuple[int, ...], set[LoRARequest]]:
        """
        生成当前步的 LoRA 模型输入。

        将请求级别的 LoRA ID 信息展开为 token 级别的映射，供模型中的
        LoRA 层在前向传播时使用。

        参数：
          req_ids:            当前步调度的请求 ID 列表。
          idx_mapping:        请求 ID 到批处理槽位索引的映射数组。
          num_scheduled_tokens: 每个请求在当前步调度的 token 数量。

        返回值，包含三个元素的元组：
          1. prompt_lora_mapping: 每个请求对应的 LoRA ID 元组（请求级别），
             用于 prompt 级别的 LoRA 路由。
          2. token_lora_mapping: 每个 token 对应的 LoRA ID 元组（token 级别），
             通过将每个请求的 LoRA ID 按其 token 数量重复展开得到。
          3. active_lora_requests: 当前步活跃的 LoRARequest 对象集合，
             用于确保对应的 LoRA 适配器权重已加载到 GPU。
        """
        # 通过 idx_mapping 索引获取当前步各请求的 LoRA ID。
        lora_ids = self.lora_ids[idx_mapping]

        # prompt 级别的 LoRA 映射：每个请求一个 LoRA ID。
        prompt_lora_mapping = tuple(lora_ids)

        # token 级别的 LoRA 映射：将每个请求的 LoRA ID 按其调度的 token 数重复。
        # 例如请求 A 有 3 个 token、LoRA ID=1，请求 B 有 2 个 token、LoRA ID=2，
        # 则结果为 (1, 1, 1, 2, 2)。
        token_lora_mapping = tuple(lora_ids.repeat(num_scheduled_tokens))

        # 收集当前步活跃的所有 LoRARequest 对象。
        active_lora_requests: set[LoRARequest] = set()
        for req_id in req_ids:
            lora_request = self.lora_requests.get(req_id)
            if lora_request is not None:
                active_lora_requests.add(lora_request)
        return prompt_lora_mapping, token_lora_mapping, active_lora_requests
