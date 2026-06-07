# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# 内核预热（Kernel Warmup）模块。
# ==============================
# 本模块负责在 vLLM 启动时通过模拟推理迭代来触发 Triton JIT 编译，
# 从而避免首次真实推理时的编译延迟。
#
# 预热流程概述：
#   1. 模拟一次 prefill 迭代：构造多个请求，每个请求有 (2 + num_spec_steps) 个
#      prompt token，触发 prefill 阶段的所有 Triton 内核编译。
#   2. 模拟一次 decode 迭代：让所有请求各生成 (1 + num_spec_steps) 个 token，
#      触发 decode 阶段的 Triton 内核编译。
#   3. 清理：通过发送 finished_req_ids 清理所有预热请求的状态。
#
# 为什么需要预热？
#   Triton 使用 JIT（即时编译）方式编译 GPU 内核。首次调用时需要编译，
#   耗时可能长达数秒。通过在启动时预先编译这些内核，可以确保首次真实
#   推理的延迟与后续推理一致。

from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from vllm import PoolingParams, SamplingParams
from vllm.utils.math_utils import cdiv
from vllm.v1.core.sched.output import (
    CachedRequestData,
    GrammarOutput,
    NewRequestData,
    SchedulerOutput,
)
from vllm.v1.request import Request
from vllm.v1.worker.gpu.model_runner import GPUModelRunner


@torch.inference_mode()
def warmup_kernels(
    model_runner: GPUModelRunner,
    worker_execute_model: Callable[[SchedulerOutput], Any],
    worker_sample_tokens: Callable[[GrammarOutput | None], Any],
) -> None:
    """
    执行两次模拟推理迭代（prefill + decode）以预热 Triton JIT 内核。

    必须通过传入的 worker 的 execute_model 进行调用，以确保流水线并行（PP）
    的协调正确。

    预热策略：
      - 第一次迭代模拟 prefill：每个请求有 (2 + num_spec_steps) 个 prompt token。
      - 第二次迭代模拟 decode：所有请求各生成 (1 + num_spec_steps) 个 token。
      - 使用 for_sampler_warmup() 生成的 SamplingParams，覆盖所有采样特性
        （temperature、top_k、top_p、repetition_penalty 等），以确保相关的
        Triton 内核都被编译。
      - 如果是 pooling 模型（如 embedding 模型），则使用 PoolingParams 替代。

    参数：
      model_runner:         GPU 模型运行器，提供配置信息和 KV Cache 管理。
      worker_execute_model: 执行模型前向传播的回调函数。
      worker_sample_tokens: 执行 token 采样的回调函数。
    """
    num_spec_steps = model_runner.num_speculative_steps
    # 使用 2 + num_spec_steps 个 token 作为 prompt 长度，确保 prefill 批处理的
    # 每请求 query 长度超过 decode_query_len（= 1 + num_spec_steps），
    # 防止被错误分类为统一的 decode 批处理。
    prompt_len = 2 + num_spec_steps
    prompt_token_ids = list(range(prompt_len))
    # prefill 后，decode 阶段生成 1 个已验证 token + num_spec_steps 个草稿 token。
    decode_len = prompt_len + 1 + num_spec_steps

    # 获取 KV Cache 组配置。
    kv_cache_groups = model_runner.kv_cache_config.kv_cache_groups
    num_kv_cache_groups = len(kv_cache_groups)

    # 计算每个请求在 prefill 和 decode 阶段需要的 KV Cache 块数量。
    group_block_sizes = [g.kv_cache_spec.block_size for g in kv_cache_groups]
    prefill_block_counts = [cdiv(prompt_len, bs) for bs in group_block_sizes]
    decode_block_counts = [cdiv(decode_len, bs) for bs in group_block_sizes]
    # decode 比 prefill 多需要的块数量。
    decode_block_deltas = [
        d - p for d, p in zip(decode_block_counts, prefill_block_counts)
    ]
    max_blocks_per_req = sum(decode_block_counts)

    # 计算预热请求数量，受以下条件约束：
    #   1. 不超过 max_num_seqs（最大并发序列数）。
    #   2. 不超过 max_num_batched_tokens / max(prompt_len, 1 + num_spec_steps)。
    #   3. KV Cache 块数量足够（预留 1 个 null block）。
    num_reqs = min(
        model_runner.scheduler_config.max_num_seqs,
        model_runner.scheduler_config.max_num_batched_tokens
        // max(prompt_len, 1 + num_spec_steps),
        # 预留 block 0（null block），确保有足够的块。
        max(1, (model_runner.kv_cache_config.num_blocks - 1) // max_blocks_per_req),
    )

    # 生成预热请求 ID。
    req_ids = [f"_warmup_{i}_" for i in range(num_reqs)]

    # 根据模型类型选择采样/池化参数。
    # for_sampler_warmup() 会返回覆盖所有采样特性的参数。
    if model_runner.is_pooling_model:
        sampling_params = None
        pooling_params = PoolingParams()
    else:
        sampling_params = SamplingParams.for_sampler_warmup()
        pooling_params = None

    # 为每个请求的每个 KV Cache 组分配独立的块 ID。
    # 从 1 开始分配（0 保留为 null block）。
    next_block_id = 1

    def _alloc_blocks(num_blocks: int) -> list[int]:
        nonlocal next_block_id
        return list(range(next_block_id, next_block_id := next_block_id + num_blocks))

    # ============================
    # 步骤 1：模拟 Prefill 阶段
    # ============================
    # 为每个请求创建 NewRequestData，包含分配的 KV Cache 块和 prompt token。
    new_reqs = [
        NewRequestData.from_request(
            Request(req_ids[i], prompt_token_ids, sampling_params, pooling_params),
            block_ids=tuple(_alloc_blocks(n) for n in prefill_block_counts),
            prefill_token_ids=prompt_token_ids,
        )
        for i in range(num_reqs)
    ]

    # 构造 SchedulerOutput，模拟调度器的 prefill 输出。
    prefill_output = SchedulerOutput.make_empty()
    prefill_output.scheduled_new_reqs = new_reqs
    prefill_output.num_scheduled_tokens = {rid: prompt_len for rid in req_ids}
    prefill_output.total_num_scheduled_tokens = prompt_len * num_reqs
    prefill_output.num_common_prefix_blocks = [0] * num_kv_cache_groups

    # 预热期间禁用 KV connector，避免不必要的跨节点通信。
    model_runner.kv_connector.set_disabled(True)
    worker_execute_model(prefill_output)

    if not model_runner.is_pooling_model:
        # 对于非 pooling 模型，还需要预热 sampler 和执行 decode 步骤。

        grammar_output = None
        if model_runner.is_last_pp_rank:
            # 构造一个 GrammarOutput 来预热结构化输出 bitmask 内核。
            # 使用全 1 的 bitmask（所有 token 都合法），仅用于触发内核编译。
            vocab_size = model_runner.model_config.get_vocab_size()
            bitmask_width = (vocab_size + 31) // 32
            grammar_bitmask = np.full(
                (len(req_ids), bitmask_width), fill_value=-1, dtype=np.int32
            )
            grammar_output = GrammarOutput(
                structured_output_request_ids=req_ids, grammar_bitmask=grammar_bitmask
            )

        # 预热 sampler（包括结构化输出内核）。
        worker_sample_tokens(grammar_output)

        # ============================
        # 步骤 2：模拟 Decode 阶段
        # ============================
        # 构造 CachedRequestData，模拟调度器的 decode 输出。
        cached_req_data = CachedRequestData.make_empty()
        cached_req_data.req_ids = list(req_ids)
        cached_req_data.num_computed_tokens = [prompt_len] * num_reqs
        cached_req_data.num_output_tokens = [1] * num_reqs
        # 如果 decode 阶段需要额外的 KV Cache 块，为每个请求分配。
        new_block = any(decode_block_deltas)
        cached_req_data.new_block_ids = [
            tuple(_alloc_blocks(n) for n in decode_block_deltas) if new_block else None
            for _ in range(num_reqs)
        ]

        decode_output = SchedulerOutput.make_empty()
        decode_output.scheduled_cached_reqs = cached_req_data
        decode_output.num_scheduled_tokens = {
            req_id: 1 + num_spec_steps for req_id in req_ids
        }
        # 如果使用投机解码，为每个请求添加草稿 token（全部使用 0，仅用于预热）。
        if num_spec_steps > 0:
            decode_output.scheduled_spec_decode_tokens = {
                req_id: [0] * num_spec_steps for req_id in req_ids
            }
        decode_output.total_num_scheduled_tokens = sum(
            decode_output.num_scheduled_tokens.values()
        )
        decode_output.num_common_prefix_blocks = [0] * num_kv_cache_groups

        worker_execute_model(decode_output)
        worker_sample_tokens(None)

    # ============================
    # 步骤 3：清理预热请求
    # ============================
    # 通过发送包含 finished_req_ids 的 SchedulerOutput 来清理所有预热请求的状态。
    cleanup_output = SchedulerOutput.make_empty()
    cleanup_output.finished_req_ids = set(req_ids)
    worker_execute_model(cleanup_output)
    # 恢复 KV connector 的正常状态。
    model_runner.kv_connector.set_disabled(False)
    # 同步设备，确保所有预热操作完成。
    torch.accelerator.synchronize()
