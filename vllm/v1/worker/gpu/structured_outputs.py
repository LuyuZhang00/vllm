# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# 结构化输出（Structured Outputs）工具模块。
# ==========================================
# 本模块实现了将语法规则约束（如 JSON Schema）应用到模型输出 logits 上的功能。
# 核心思路是：通过 grammar bitmask 将不合法的 token 对应的 logits 设置为 -inf，
# 从而在采样时强制模型只选择合法的 token。
#
# 工作流程：
#   1. 调度器（Scheduler）根据请求的结构化输出配置，生成 grammar bitmask。
#   2. 每步推理后，将 bitmask 异步拷贝到 GPU。
#   3. 使用 Triton 内核将 bitmask 应用到 logits 张量上，
#      将被掩码（不允许）的 token 的 logits 设置为 -inf。
#
# 性能优化：
#   - 使用独立的 CUDA stream（copy_stream）进行异步数据拷贝，
#     与当前 stream 的计算重叠，减少等待时间。
#   - 使用 Triton JIT 编译的内核进行高效的位掩码解包和 logits 修改。

import numpy as np
import torch

from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import cdiv
from vllm.v1.worker.gpu.buffer_utils import async_copy_to_gpu
from vllm.v1.worker.gpu.input_batch import InputBatch


class StructuredOutputsWorker:
    """
    结构化输出 Worker，负责将语法约束掩码应用到模型输出的 logits 上。

    本类管理一个独立的 CUDA stream 用于异步数据拷贝，并预分配 GPU 缓冲区
    以避免每步重复分配内存。

    属性：
      logits_indices: GPU 上的 int32 张量，存储 bitmask 对应的 logits 行索引。
      grammar_bitmask: GPU 上的 int32 张量，形状为 (max_num_logits, ceil(vocab_size/32))，
                       存储压缩的位掩码。每个 int32 元素的每一位代表一个 token 是否合法。
      device:          计算设备。
      copy_stream:     用于异步拷贝 bitmask 和 indices 的独立 CUDA stream。
    """

    def __init__(self, max_num_logits: int, vocab_size: int, device: torch.device):
        """
        初始化结构化输出 Worker。

        参数：
          max_num_logits: 最大 logits 数量（通常等于 max_num_batched_tokens），
                          用于预分配缓冲区大小。
          vocab_size:     词表大小，决定 bitmask 的宽度。
          device:         计算设备（GPU）。
        """
        # 预分配 logits 索引缓冲区。
        self.logits_indices = torch.zeros(
            max_num_logits, dtype=torch.int32, device=device
        )
        # 预分配 bitmask 缓冲区。每 32 个 token 用一个 int32 表示。
        self.grammar_bitmask = torch.zeros(
            (max_num_logits, cdiv(vocab_size, 32)), dtype=torch.int32, device=device
        )
        self.device = device
        # 创建独立的 CUDA stream 用于异步数据拷贝。
        self.copy_stream = torch.cuda.Stream()

    def apply_grammar_bitmask(
        self,
        logits: torch.Tensor,
        input_batch: InputBatch,
        grammar_req_ids: list[str],
        grammar_bitmask: np.ndarray,
    ) -> None:
        """
        将语法约束掩码应用到 logits 张量上。

        该方法执行以下步骤：
          1. 在 copy_stream 上异步将 bitmask 从 CPU 拷贝到 GPU。
          2. 构建 bitmask 行索引到 logits 行索引的映射。
          3. 在 copy_stream 上异步将映射拷贝到 GPU。
          4. 等待拷贝完成后，启动 Triton 内核将掩码应用到 logits。

        参数：
          logits:          模型输出的 logits 张量，形状为 (total_logits, vocab_size)。
          input_batch:     当前批处理的输入信息，包含请求 ID 和 logits 索引。
          grammar_req_ids: 需要应用语法约束的请求 ID 列表。
          grammar_bitmask: CPU 上的 bitmask NumPy 数组，
                           形状为 (num_grammar_reqs, ceil(vocab_size/32))，
                           dtype 为 int32。值为 1 的位表示对应的 token 合法。
        """
        # 如果没有需要处理的请求，直接返回。
        if not grammar_req_ids:
            return

        # 步骤 1：在 copy_stream 上异步将 bitmask 拷贝到 GPU。
        with torch.cuda.stream(self.copy_stream):
            bitmask = async_copy_to_gpu(
                grammar_bitmask, out=self.grammar_bitmask[: grammar_bitmask.shape[0]]
            )

        # 步骤 2：构建 bitmask 行 -> logits 行的映射。
        # grammar_bitmask 的每一行对应一个请求的约束掩码，
        # 而 logits 张量中每个请求可能有多个 token 的 logits（多步采样）。
        # 需要将 bitmask 的每一行映射到对应的 logits 行范围。
        mapping: list[int] = []
        req_ids = input_batch.req_ids
        cu_num_logits = input_batch.cu_num_logits_np.tolist()
        req_id_to_idx = {req_id: i for i, req_id in enumerate(req_ids)}
        for grammar_req_id in grammar_req_ids:
            req_idx = req_id_to_idx[grammar_req_id]
            logits_start_idx = cu_num_logits[req_idx]
            logits_end_idx = cu_num_logits[req_idx + 1]
            mapping.extend(range(logits_start_idx, logits_end_idx))

        # 步骤 3：在 copy_stream 上异步将映射拷贝到 GPU。
        with torch.cuda.stream(self.copy_stream):
            logits_indices = torch.tensor(
                mapping, dtype=torch.int32, device="cpu", pin_memory=True
            )
            logits_indices = self.logits_indices[: len(mapping)].copy_(
                logits_indices, non_blocking=True
            )

        # 步骤 4：等待 copy_stream 上的所有异步拷贝完成。
        current_stream = torch.cuda.current_stream()
        current_stream.wait_stream(self.copy_stream)

        # 步骤 5：启动 Triton 内核，将 bitmask 应用到 logits。
        num_masks = bitmask.shape[0]
        assert num_masks == len(mapping)
        vocab_size = logits.shape[-1]
        BLOCK_SIZE = 8192
        grid = (num_masks, triton.cdiv(vocab_size, BLOCK_SIZE))
        _apply_grammar_bitmask_kernel[grid](
            logits,
            logits.stride(0),
            logits_indices,
            bitmask,
            bitmask.stride(0),
            vocab_size,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        # 步骤 6：确保 copy_stream 等待当前 stream 完成后再复用缓冲区。
        # 这样可以避免 copy_stream 在内核仍在读取缓冲区时覆盖数据。
        self.copy_stream.wait_stream(current_stream)


# 以下 Triton 内核改编自 xgrammar 项目：
# https://github.com/mlc-ai/xgrammar/blob/main/python/xgrammar/kernels/apply_token_bitmask_inplace_triton.py
@triton.jit
def _apply_grammar_bitmask_kernel(
    logits_ptr,
    logits_stride,
    logits_indices_ptr,
    bitmask_ptr,
    bitmask_stride,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton JIT 内核：将压缩的位掩码应用到 logits 张量。

    每个线程块处理一个 bitmask 行的一个 BLOCK_SIZE 大小的 vocab 分片。
    工作流程：
      1. 从 logits_indices_ptr 加载当前 bitmask 对应的 logits 行索引。
      2. 从 bitmask_ptr 加载压缩的位掩码数据（每 32 个 token 压缩为一个 int32）。
      3. 解包位掩码：将每个 int32 拆分为 32 个布尔值，表示对应 token 是否被允许。
         注意：bitmask 中值为 1 的位表示 token 合法，取反后得到被禁止的 token。
      4. 将被禁止的 token 对应的 logits 设置为 -inf。

    参数：
      logits_ptr:        logits 张量的指针。
      logits_stride:     logits 张量的行步长（即 vocab_size）。
      logits_indices_ptr: bitmask 到 logits 行的映射指针。
      bitmask_ptr:       bitmask 张量的指针。
      bitmask_stride:    bitmask 张量的行步长。
      vocab_size:        词表大小。
      BLOCK_SIZE:        每个线程块处理的 vocab 分片大小（编译时常量）。
    """
    # 确定当前线程块处理的 bitmask 行索引。
    bitmask_idx = tl.program_id(0)
    # 加载对应的 logits 行索引。
    logits_idx = tl.load(logits_indices_ptr + bitmask_idx)

    # 加载当前 BLOCK 对应的压缩位掩码数据。
    block_id = tl.program_id(1)
    bitmask_offset = (block_id * BLOCK_SIZE) // 32 + tl.arange(0, BLOCK_SIZE // 32)
    packed_bitmask = tl.load(
        bitmask_ptr + bitmask_idx * bitmask_stride + bitmask_offset,
        mask=bitmask_offset < bitmask_stride,
    )
    # 解包位掩码：将每个 int32 拆分为 32 个布尔值。
    # 然后取反（== 0），得到被禁止的 token 位置。
    bitmask = ((packed_bitmask[:, None] >> (tl.arange(0, 32)[None, :])) & 1) == 0
    bitmask = bitmask.reshape(BLOCK_SIZE)

    # 将被禁止的 token 对应的 logits 设置为 -inf。
    block_offset = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    tl.store(
        logits_ptr + logits_idx * logits_stride + block_offset,
        -float("inf"),
        mask=bitmask & (block_offset < vocab_size),
    )
