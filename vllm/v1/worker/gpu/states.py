# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# 请求状态管理模块。
# ==================
# 本模块定义了 RequestState 类，用于维护 GPU Worker 中每个请求的核心运行时状态。
# 这些状态包括：
#   1. 请求 ID 与批处理槽位索引之间的双向映射。
#   2. 每个请求的所有 token ID 序列（包括 prompt 和已生成的 output）。
#   3. 每个请求的 prompt 长度、prefill 长度和总长度。
#   4. 每个请求已计算的 token 数量（用于跟踪 prefill/decode 进度）。
#   5. 上一步采样的 token（用于 decode 阶段的输入）。
#   6. 投机解码（speculative decoding）的草稿 token。
#   7. 下一步 prefill 需要的 token。
#
# 为了节省 GPU 内存，本模块大量使用了 UVA（Unified Virtual Addressing）
# 和分阶段写入（Staged Write）技术：
#   - UVA：将张量分配在 CPU 锁页内存中，通过 UVA 映射到 GPU 地址空间，
#     适用于大型但访问频率较低的张量（如 all_token_ids）。
#   - Staged Write：先在 CPU 端暂存数据，然后批量拷贝到 GPU，
#     减少频繁的小规模 Host-Device 传输开销。

import numpy as np
import torch

from vllm.v1.worker.gpu.buffer_utils import StagedWriteTensor, UvaBackedTensor


class RequestState:
    """
    请求状态管理器，维护 GPU Worker 中所有请求的运行时状态。

    本类使用固定大小的数组（而非动态字典）来存储每个请求的状态，
    以实现高效的向量化操作和 GPU 访问。请求通过槽位索引（index）
    进行标识，支持快速的添加、删除和查询操作。

    属性：
      max_num_reqs:              最大并发请求数。
      max_model_len:             模型支持的最大序列长度。
      max_num_batched_tokens:    单步最大批处理 token 数。
      num_speculative_steps:     投机解码的步数（0 表示不使用投机解码）。
      vocab_size:                词表大小。
      device:                    计算设备（GPU）。
      req_id_to_index:           请求 ID -> 槽位索引的映射字典。
      index_to_req_id:           槽位索引 -> 请求 ID 的映射字典。
      free_indices:              空闲槽位索引列表，用于快速分配新请求。
      all_token_ids:             每个请求的所有 token ID（UVA 存储）。
      prompt_len:                每个请求的 prompt 长度（UVA 存储）。
      prefill_len:               每个请求的 prefill 长度（UVA 存储）。
      total_len:                 每个请求的总长度（GPU 存储，分阶段写入）。
      num_computed_prefill_tokens: 每个请求已计算的 prefill token 数（NumPy 数组）。
      num_computed_tokens:       每个请求已计算的 token 总数（GPU 存储）。
      num_computed_tokens_np:    num_computed_tokens 的 CPU 镜像（乐观上界）。
      last_sampled_tokens:       上一步采样的 token（GPU 存储）。
      draft_tokens:              投机解码的草稿 token（GPU 存储）。
      next_prefill_tokens:       下一步 prefill 需要的 token（GPU 存储）。
    """

    def __init__(
        self,
        max_num_reqs: int,
        max_model_len: int,
        max_num_batched_tokens: int,
        num_speculative_steps: int,
        vocab_size: int,
        device: torch.device,
    ):
        """
        初始化请求状态管理器。

        参数：
          max_num_reqs:            最大并发请求数，决定所有数组的第 0 维大小。
          max_model_len:           模型支持的最大序列长度。
          max_num_batched_tokens:  单步最大批处理 token 数。
          num_speculative_steps:   投机解码的步数。
          vocab_size:              词表大小。
          device:                  计算设备。
        """
        self.max_num_reqs = max_num_reqs
        self.max_model_len = max_model_len
        self.max_num_batched_tokens = max_num_batched_tokens
        self.num_speculative_steps = num_speculative_steps
        self.vocab_size = vocab_size
        self.device = device

        # 请求 ID 与槽位索引之间的双向映射。
        self.req_id_to_index: dict[str, int] = {}
        self.index_to_req_id: dict[int, str] = {}
        # 空闲槽位索引列表，初始时所有槽位都可用。
        self.free_indices = list(range(max_num_reqs))

        # 所有请求的 token ID 序列，形状为 (max_num_reqs, max_model_len)。
        # NOTE(woosuk): 该张量可能非常大（数 GB），使用 UVA 而非 GPU 内存以节省显存。
        self.all_token_ids = StagedWriteTensor(
            (self.max_num_reqs, self.max_model_len),
            dtype=torch.int32,
            device=device,
            uva_instead_of_gpu=True,
        )

        # NOTE(woosuk): 区分 prompt_len 和 prefill_len 的重要性：
        #   - prompt_len: 用户提供的原始 prompt 中的 token 数量。
        #   - prefill_len: 实际送入模型进行 prefill 的 token 数量，
        #     可能包含 prompt 和额外的部分输出 token（如 preemption 恢复场景）。
        #   因此 prefill_len >= prompt_len。
        #   区分这两个值非常重要，因为某些功能（如 prompt logprobs、频率惩罚）
        #   需要区分 prompt token 和 output token。
        self.prompt_len = UvaBackedTensor(self.max_num_reqs, dtype=torch.int32)
        self.prefill_len = UvaBackedTensor(self.max_num_reqs, dtype=torch.int32)

        # total_len = prompt_len + output_len。随着请求推进，该值不断增长。
        self.total_len = StagedWriteTensor(
            self.max_num_reqs, dtype=torch.int32, device=device
        )

        # 已计算的 prefill token 数量（CPU NumPy 数组，用于快速判断是否仍在 prefill 阶段）。
        self.num_computed_prefill_tokens = np.zeros(self.max_num_reqs, dtype=np.int32)
        # 已计算的 token 总数（GPU 张量，用于模型前向传播中的位置计算等）。
        self.num_computed_tokens = StagedWriteTensor(
            self.max_num_reqs, dtype=torch.int32, device=device
        )
        # num_computed_tokens 的 CPU 镜像，是 GPU 值的乐观上界。
        # 用于在 CPU 端快速查询，避免 GPU-CPU 同步。
        self.num_computed_tokens_np = np.zeros(self.max_num_reqs, dtype=np.int32)

        # 上一步采样的 token ID，形状为 (max_num_reqs, 1)。
        # 在 decode 阶段作为下一步的输入 token。
        self.last_sampled_tokens = torch.zeros(
            self.max_num_reqs, 1, dtype=torch.int64, device=device
        )

        # 投机解码的草稿 token，形状为 (max_num_reqs, num_speculative_steps)。
        # 不使用投机解码时 num_speculative_steps=0，此张量为空。
        self.draft_tokens = torch.zeros(
            self.max_num_reqs,
            self.num_speculative_steps,
            dtype=torch.int64,
            device=device,
        )

        # 下一步 prefill 需要处理的 token，形状为 (max_num_reqs,)。
        self.next_prefill_tokens = torch.zeros(
            self.max_num_reqs, dtype=torch.int32, device=device
        )

    @property
    def num_reqs(self) -> int:
        """当前活跃的请求数量。"""
        return len(self.req_id_to_index)

    def add_request(
        self,
        req_id: str,
        prompt_len: int,
        all_token_ids: list[int],
        num_computed_tokens: int,
    ) -> None:
        """
        添加一个新请求的状态信息。

        从空闲槽位列表中分配一个索引，初始化该请求的所有状态字段。
        对于从 preemption 恢复或 PD 分离场景，num_computed_tokens 可能 > 0，
        表示部分 token 已经被计算过。

        参数：
          req_id:              请求的唯一标识符。
          prompt_len:          prompt 的 token 数量。
          all_token_ids:       该请求当前所有的 token ID 列表
                               （包括 prompt 和已生成的 output）。
          num_computed_tokens: 已经被 KV Cache 计算过的 token 数量。
        """
        # 从空闲列表中分配一个槽位索引。
        assert len(self.free_indices) > 0, "No free indices"
        req_idx = self.free_indices.pop()
        self.req_id_to_index[req_id] = req_idx
        self.index_to_req_id[req_idx] = req_id

        # 设置 prompt 长度和 prefill 长度。
        self.prompt_len.np[req_idx] = prompt_len
        prefill_len = len(all_token_ids)
        assert prefill_len >= prompt_len, (
            f"prefill_len {prefill_len} < prompt_len {prompt_len}"
        )
        self.prefill_len.np[req_idx] = prefill_len

        # 设置总长度、所有 token ID、已计算 token 数等。
        self.total_len.stage_write_elem(req_idx, prefill_len)
        self.all_token_ids.stage_write(req_idx, 0, all_token_ids)
        self.num_computed_prefill_tokens[req_idx] = num_computed_tokens
        self.num_computed_tokens_np[req_idx] = num_computed_tokens
        self.num_computed_tokens.stage_write_elem(req_idx, num_computed_tokens)

        # 对于从 preemption 恢复或 PD 分离的请求（num_computed_tokens > 0 且 <= prefill_len），
        # 设置 last_sampled_tokens 为最后一个已计算的 token，这样第一步 decode
        # 能获得正确的输入 token ID。
        # 对于全新的 prefill 请求（num_computed_tokens == 0），last_sampled_tokens
        # 不会被读取，因此跳过写入。
        # 使用切片赋值而非标量索引，以避免 host/device 同步。
        if 0 < num_computed_tokens <= prefill_len:
            self.last_sampled_tokens[req_idx : req_idx + 1] = all_token_ids[
                num_computed_tokens - 1
            ]
        # 清零草稿 token。
        self.draft_tokens[req_idx].zero_()

    def apply_staged_writes(self) -> None:
        """
        将所有暂存在 CPU 端的数据批量拷贝到 GPU。

        在每步迭代开始前调用，确保 GPU 上的数据与 CPU 端的最新状态同步。
        包括：prompt_len、prefill_len（UVA 拷贝），total_len、all_token_ids、
        num_computed_tokens（分阶段写入）。
        """
        self.prompt_len.copy_to_uva()
        self.prefill_len.copy_to_uva()
        self.total_len.apply_write()
        self.all_token_ids.apply_write()
        self.num_computed_tokens.apply_write()

    def remove_request(self, req_id: str) -> bool:
        """
        移除一个已结束的请求，释放其占用的槽位。

        参数：
          req_id: 要移除的请求的唯一标识符。

        返回值：
          True 如果成功移除，False 如果请求不存在。
        """
        req_idx = self.req_id_to_index.pop(req_id, None)
        if req_idx is None:
            # 请求不存在。
            return False
        self.index_to_req_id.pop(req_idx, None)
        # 将槽位索引归还到空闲列表。
        self.free_indices.append(req_idx)
        return True

    def is_prefilling(self, idx_mapping_np: np.ndarray) -> np.ndarray:
        """
        判断给定请求是否仍在 prefill 阶段。

        通过比较已计算的 prefill token 数与 prefill 长度来判断：
        如果 num_computed_prefill_tokens < prefill_len，则该请求仍在 prefill。

        参数：
          idx_mapping_np: 请求槽位索引的 NumPy 数组。

        返回值：
          布尔类型的 NumPy 数组，True 表示对应的请求仍在 prefill 阶段。
        """
        return (
            self.num_computed_prefill_tokens[idx_mapping_np]
            < self.prefill_len.np[idx_mapping_np]
        )
