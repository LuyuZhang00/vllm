# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
多维旋转位置编码模块 (Multi-Dimensional Rotary Position Embedding Module)

本模块实现了多维旋转位置编码 (RoPE) 的状态管理，支持两种变体：

1. M-RoPE (Multi-dimensional RoPE):
   - 3 个维度，每个维度有独立的位置编码
   - 解码阶段使用 position delta（位置增量）
   - 用于支持多模态输入的位置编码（如图像+文本）
   - 对于纯文本输入，3 个维度的位置 ID 相同，等价于 1D RoPE

2. XD-RoPE (eXtended Dimension RoPE):
   - 3 或 4 个维度
   - delta 始终为 0（解码阶段所有维度使用原始位置）
   - 用于需要更灵活位置编码的模型

关键设计：
- 使用 StagedWriteTensor 暂存 prefill 阶段的位置编码（CPU -> GPU 批量传输）
- 使用 Triton 内核在 GPU 上高效计算位置编码
- positions 张量故意多分配一个位置以使其不连续，兼容 torch.compile

参考文献：
- M-RoPE: https://arxiv.org/abs/2409.12191
- 非连续张量设计: https://github.com/vllm-project/vllm/pull/12128#discussion_r1926431923
"""
from typing import cast

import torch
import torch.nn as nn

from vllm.config import ModelConfig
from vllm.model_executor.models.interfaces import SupportsMRoPE, SupportsXDRoPE
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.buffer_utils import StagedWriteTensor, UvaBackedTensor


class RopeState:
    """多维旋转位置编码的状态管理器。

    管理 M-RoPE 和 XD-RoPE 的位置编码计算和存储。

    属性：
        num_dims: 位置编码的维度数（M-RoPE: 3, XD-RoPE: 3 或 4）
        has_delta: 是否使用位置增量（M-RoPE: True, XD-RoPE: False）
        prefill_positions: prefill 阶段的位置编码（暂存 -> GPU）
        positions: 当前步骤的位置编码（GPU 张量）
        prefill_delta: prefill 阶段的位置增量
    """

    def __init__(
        self,
        num_dims: int,
        has_delta: bool,
        max_num_reqs: int,
        max_num_tokens: int,
        max_model_len: int,
        device: torch.device,
    ):
        """初始化 RoPE 状态。

        Args:
            num_dims: 位置编码的维度数
            has_delta: 是否使用位置增量
            max_num_reqs: 最大请求数
            max_num_tokens: 最大 token 数
            max_model_len: 最大模型长度
            device: 计算设备
        """
        self.num_dims = num_dims
        self.has_delta = has_delta
        self.max_num_reqs = max_num_reqs
        self.max_num_tokens = max_num_tokens
        self.max_model_len = max_model_len
        self.device = device

        # 注意：此张量可能非常大（例如几 GB），浪费大量 CPU 内存
        self.prefill_positions = StagedWriteTensor(
            (max_num_reqs * num_dims, max_model_len),
            dtype=torch.int32,
            device=device,
            uva_instead_of_gpu=True,
        )
        # 注意：故意多分配一个位置使其不连续，以便与 torch.compile 兼容
        self.positions = torch.zeros(
            (num_dims, max_num_tokens + 1), dtype=torch.int64, device=device
        )

        # 位置增量：M-RoPE 非零，XD-RoPE 始终为 0
        self.prefill_delta = UvaBackedTensor(max_num_reqs, dtype=torch.int32)

    def init_prefill_positions(
        self,
        req_idx: int,
        model: nn.Module,
        prefill_token_ids: list[int],
        mm_features: list,
    ) -> None:
        """初始化请求的 prefill 阶段位置编码。

        根据模型类型（M-RoPE 或 XD-RoPE）计算位置编码。

        Args:
            req_idx: 请求在批次中的索引
            model: 模型实例
            prefill_token_ids: prefill 的 token IDs
            mm_features: 多模态特征列表
        """
        if self.has_delta:
            # M-RoPE: 3 维位置编码 + 位置增量
            mrope_model = cast(SupportsMRoPE, model)
            prefill_positions, delta = mrope_model.get_mrope_input_positions(
                prefill_token_ids, mm_features
            )
            self.prefill_delta.np[req_idx] = delta
        else:
            # XD-RoPE: 多维位置编码，无位置增量
            xdrope_model = cast(SupportsXDRoPE, model)
            prefill_positions = xdrope_model.get_xdrope_input_positions(
                prefill_token_ids, mm_features
            )

        # 暂存每个维度的位置编码
        for i in range(self.num_dims):
            pos = prefill_positions[i].tolist()
            self.prefill_positions.stage_write(self.num_dims * req_idx + i, 0, pos)

    def apply_staged_writes(self) -> None:
        """将暂存的位置编码数据批量应用到 GPU。"""
        self.prefill_positions.apply_write()
        if self.has_delta:
            self.prefill_delta.copy_to_uva()

    def get_positions(self, num_tokens: int) -> torch.Tensor:
        """获取当前步骤的位置编码。

        Args:
            num_tokens: token 数量

        Returns:
            位置编码张量 [num_dims, num_tokens]
        """
        return self.positions[:, :num_tokens]

    def prepare_positions(
        self,
        idx_mapping: torch.Tensor,
        query_start_loc: torch.Tensor,
        prefill_lens: torch.Tensor,
        num_computed_tokens: torch.Tensor,
    ) -> None:
        """使用 Triton 内核在 GPU 上计算位置编码。

        对于每个请求：
        - prefill 阶段：从暂存的位置编码中读取
        - 解码阶段：使用原始位置 + delta（M-RoPE）或原始位置（XD-RoPE）

        Args:
            idx_mapping: 请求索引映射
            query_start_loc: 每个请求的 query 起始位置
            prefill_lens: prefill 长度
            num_computed_tokens: 已计算的 token 数量
        """
        num_reqs = idx_mapping.shape[0]
        _prepare_rope_positions_kernel[(num_reqs,)](
            self.positions,
            self.positions.stride(0),
            self.prefill_positions.gpu,
            self.num_dims * self.max_model_len,
            self.max_model_len,
            self.prefill_delta.gpu,
            idx_mapping,
            query_start_loc,
            prefill_lens,
            num_computed_tokens,
            BLOCK_SIZE=1024,
            NUM_DIMS=self.num_dims,
        )


def get_rope_state(
    model_config: ModelConfig,
    model: nn.Module,
    max_num_reqs: int,
    max_num_tokens: int,
    max_model_len: int,
    device: torch.device,
) -> RopeState | None:
    """如果模型使用多维 RoPE，则创建 RopeState。

    根据模型配置判断是否需要多维 RoPE：
    1. uses_mrope: 使用 M-RoPE（3 维，有 delta）
    2. uses_xdrope_dim > 0: 使用 XD-RoPE（3 或 4 维，无 delta）
    3. 其他: 不需要多维 RoPE，返回 None

    Args:
        model_config: 模型配置
        model: 模型实例
        max_num_reqs: 最大请求数
        max_num_tokens: 最大 token 数
        max_model_len: 最大模型长度
        device: 计算设备

    Returns:
        RopeState 实例，或 None（如果模型不使用多维 RoPE）
    """
    if model_config.uses_mrope:
        assert isinstance(model, SupportsMRoPE)
        return RopeState(
            num_dims=3,
            has_delta=True,
            max_num_reqs=max_num_reqs,
            max_num_tokens=max_num_tokens,
            max_model_len=max_model_len,
            device=device,
        )
    if model_config.uses_xdrope_dim > 0:
        assert isinstance(model, SupportsXDRoPE)
        return RopeState(
            num_dims=model_config.uses_xdrope_dim,
            has_delta=False,
            max_num_reqs=max_num_reqs,
            max_num_tokens=max_num_tokens,
            max_model_len=max_model_len,
            device=device,
        )
    return None


@triton.jit
def _prepare_rope_positions_kernel(
    positions_ptr,
    positions_stride,
    prefill_positions_ptr,
    prefill_positions_stride0,
    prefill_positions_stride1,
    prefill_delta_ptr,
    idx_mapping_ptr,
    query_start_loc_ptr,
    prefill_lens_ptr,
    num_computed_tokens_ptr,
    BLOCK_SIZE: tl.constexpr,
    NUM_DIMS: tl.constexpr,
):
    """RoPE 位置编码准备 Triton 内核。

    为每个请求计算多维位置编码：
    - prefill 阶段：从暂存的位置编码中读取（支持图像等多模态输入的复杂位置编码）
    - 解码阶段：使用原始位置 + delta（M-RoPE）或原始位置（XD-RoPE）

    Args:
        positions_ptr: 输出位置编码指针 [num_dims, max_num_tokens + 1]
        positions_stride: 输出位置编码的步长
        prefill_positions_ptr: 暂存的 prefill 位置编码指针
        prefill_positions_stride0: prefill 位置编码的第一维步长
        prefill_positions_stride1: prefill 位置编码的第二维步长
        prefill_delta_ptr: 位置增量指针
        idx_mapping_ptr: 请求索引映射指针
        query_start_loc_ptr: query 起始位置指针
        prefill_lens_ptr: prefill 长度指针
        num_computed_tokens_ptr: 已计算 token 数量指针
        BLOCK_SIZE: 每个 block 处理的 token 数量（编译时常量）
        NUM_DIMS: 位置编码的维度数（编译时常量）
    """
    batch_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + batch_idx)

    prefill_len = tl.load(prefill_lens_ptr + req_state_idx)
    num_computed = tl.load(num_computed_tokens_ptr + req_state_idx)
    is_prefill = num_computed < prefill_len

    query_start = tl.load(query_start_loc_ptr + batch_idx)
    query_end = tl.load(query_start_loc_ptr + batch_idx + 1)
    query_len = query_end - query_start

    delta = tl.load(prefill_delta_ptr + req_state_idx)

    for i in range(0, query_len, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < query_len
        orig_pos = num_computed + block

        for j in tl.static_range(NUM_DIMS):
            if is_prefill:
                # prefill 阶段：从暂存的位置编码中读取
                pos = tl.load(
                    prefill_positions_ptr
                    + req_state_idx * prefill_positions_stride0
                    + j * prefill_positions_stride1
                    + orig_pos,
                    mask=mask,
                )
            else:
                # 解码阶段：使用原始位置 + delta
                pos = orig_pos + delta
            tl.store(
                positions_ptr + j * positions_stride + query_start + block,
                pos,
                mask=mask,
            )
