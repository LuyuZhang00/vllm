# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Backend for GatedDeltaNet attention."""
"""
GatedDeltaNet（门控差分网络）注意力后端。

GatedDeltaNet 是一种基于线性注意力的高效序列模型，结合了：
1. 门控机制（Gating）：控制信息流动，类似 LSTM 的遗忘门
2. 差分注意力（DeltaNet）：使用 delta 规则更新状态，增强记忆能力
3. 因果卷积（Causal Conv1d）：在状态更新前进行局部特征提取

与传统 Transformer 注意力相比，GDN 具有：
- 线性复杂度 O(n)（而非 O(n^2)），适合超长序列
- 固定大小的隐状态（类似 RNN），内存效率高
- 支持流式推理（streaming inference）

本模块实现 GDN 的注意力后端，包括：
- GDNAttentionBackend：后端能力声明
- GDNAttentionMetadata：注意力元数据
- GDNAttentionMetadataBuilder：元数据构建器，处理推测解码和前缀缓存

特殊处理：
1. 推测解码支持：将请求分为 spec decode 和 non-spec 两组分别处理
2. FLA 分块：使用 Flash Linear Attention 的分块机制进行 prefill
3. CUDA Graph 支持：统一批次（uniform batch）模式
4. 多种 prefill 后端：triton、flashinfer、cutedsl

执行流程：
1. 推测解码判断：根据 num_decode_draft_tokens_cpu 判断是否有推测解码序列
2. 请求分组：将 spec decode 和 non-spec 请求分别处理
3. FLA 分块元数据：为 prefill 计算分块索引和偏移
4. 隐状态索引：计算每个请求的隐状态存储位置
5. CUDA Graph padding：统一张量形状以支持 CUDA Graph
"""

from dataclasses import dataclass
from typing import Literal

import torch

from vllm.config import VllmConfig
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.utils import (
    NULL_BLOCK_ID,
    compute_causal_conv1d_metadata,
    mamba_get_block_table_tensor,
    split_decodes_and_prefills,
)
from vllm.v1.kv_cache_interface import AttentionSpec, MambaSpec


class GDNAttentionBackend(AttentionBackend):
    """
    GatedDeltaNet 注意力后端类。

    声明 GDN 模型的后端能力。GDN 是一种 SSM（状态空间模型），
    标记为 is_ssm=True 以区别于传统 Transformer 注意力。
    """

    @staticmethod
    def get_name() -> str:
        """返回后端名称标识。"""
        return "GDN_ATTN"

    @staticmethod
    def get_builder_cls() -> type["GDNAttentionMetadataBuilder"]:
        """返回 GDN 元数据构建器类。"""
        return GDNAttentionMetadataBuilder

    @classmethod
    def is_ssm(cls) -> bool:
        """
        标识此为状态空间模型（SSM）后端。

        SSM 后端与传统注意力后端的区别：
        1. 使用隐状态（而非 KV 缓存）存储历史信息
        2. 支持前缀缓存的隐状态恢复
        3. 推测解码处理方式不同
        """
        return True


@dataclass
class GDNAttentionMetadata:
    """
    GatedDeltaNet 注意力元数据数据类。

    存储 GDN 模型单步推理所需的全部元数据。

    基本计数信息：
    - num_prefills: non-spec prefill 请求数
    - num_prefill_tokens: non-spec prefill token 数
    - num_decodes: non-spec decode 请求数
    - num_decode_tokens: non-spec decode token 数
    - num_spec_decodes: 推测解码请求数
    - num_spec_decode_tokens: 推测解码 token 数
    - num_actual_tokens: 实际总 token 数（含 padding）

    状态相关：
    - has_initial_state: 每个 prefill 请求是否有初始状态

    查询位置（分为 spec 和 non-spec 两组）：
    - spec_query_start_loc: 推测解码请求的查询起始位置
    - non_spec_query_start_loc: 非推测请求的查询起始位置

    隐状态索引：
    - spec_state_indices_tensor: 推测解码请求的隐状态索引
    - non_spec_state_indices_tensor: 非推测请求的隐状态索引

    推测解码掩码：
    - spec_sequence_masks: 推测解码序列的布尔掩码
    - spec_token_indx: 推测 token 的全局索引
    - non_spec_token_indx: 非推测 token 的全局索引
    - num_accepted_tokens: 每个推测序列接受的 token 数

    FLA 分块元数据（prefill 专用）：
    - chunk_indices: 分块索引
    - chunk_offsets: 分块偏移

    causal_conv1d 元数据（Triton 实现）：
    - nums_dict, batch_ptr, token_chunk_offset_ptr
    """

    num_prefills: int
    num_prefill_tokens: int
    num_decodes: int
    num_decode_tokens: int
    num_spec_decodes: int
    num_spec_decode_tokens: int
    num_actual_tokens: int

    has_initial_state: torch.Tensor | None = None

    spec_query_start_loc: torch.Tensor | None = None  # shape: [num_spec_decodes + 1,]
    non_spec_query_start_loc: torch.Tensor | None = (
        None  # shape: [batch - num_spec_decodes + 1,]
    )

    spec_state_indices_tensor: torch.Tensor | None = None  # shape: [batch, num_spec]
    non_spec_state_indices_tensor: torch.Tensor | None = (
        None  # shape: [batch - num_spec_decodes,]
    )
    spec_sequence_masks: torch.Tensor | None = None  # shape: [batch,]
    spec_token_indx: torch.Tensor | None = None
    non_spec_token_indx: torch.Tensor | None = None

    num_accepted_tokens: torch.Tensor | None = None  # shape: [batch,]

    # Pre-computed FLA chunk metadata (avoids GPU->CPU sync in prepare_chunk_indices)
    # 预计算的 FLA 分块元数据，避免 GPU->CPU 同步
    chunk_indices: torch.Tensor | None = None
    chunk_offsets: torch.Tensor | None = None

    # The following attributes are for triton implementation of causal_conv1d
    # 以下属性用于 causal_conv1d 的 Triton 实现
    nums_dict: dict | None = None
    batch_ptr: torch.Tensor | None = None
    token_chunk_offset_ptr: torch.Tensor | None = None


class GDNAttentionMetadataBuilder(AttentionMetadataBuilder[GDNAttentionMetadata]):
    """
    GatedDeltaNet 注意力元数据构建器。

    处理 GDN 模型特有的元数据构建逻辑，包括：
    1. 推测解码序列的分离和处理
    2. FLA（Flash Linear Attention）分块元数据的计算
    3. 隐状态索引的计算
    4. CUDA Graph padding

    推测解码处理策略：
    - 当存在推测解码序列时，将请求分为 spec 和 non-spec 两组
    - non-spec decode 在有 spec decode 时被重分类为 prefill
      （因为 prefill kernel 可以正确处理带初始状态的单 token 序列）
    - 两组使用独立的 query_start_loc 和 state_indices_tensor

    FLA 分块：
    - Prefill 使用 Flash Linear Attention 的分块机制
    - 支持三种 prefill 后端：triton、flashinfer、cutedsl
    - 分块大小由 FLA_CHUNK_SIZE 决定
    """

    _cudagraph_support = AttentionCGSupport.UNIFORM_BATCH

    reorder_batch_threshold: int = 1

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        assert isinstance(kv_cache_spec, MambaSpec)
        self.vllm_config = vllm_config
        self.compilation_config = vllm_config.compilation_config
        self.speculative_config = vllm_config.speculative_config
        self.kv_cache_spec = kv_cache_spec
        # 确定 GDN prefill 使用的后端实现
        from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
            _resolve_gdn_prefill_backend,
        )

        self.gdn_prefill_backend: Literal["triton", "flashinfer", "cutedsl"]
        _, self.gdn_prefill_backend = _resolve_gdn_prefill_backend(vllm_config)

        if self.speculative_config:
            assert self.speculative_config.num_speculative_tokens is not None
            self.num_spec: int = self.speculative_config.num_speculative_tokens
        else:
            self.num_spec = 0
        self.use_spec_decode: bool = self.num_spec > 0
        self._init_reorder_batch_threshold(1, self.use_spec_decode)

        # 是否使用全图 CUDA Graph
        self.use_full_cuda_graph: bool = (
            self.compilation_config.cudagraph_mode.has_full_cudagraphs()
        )

        # CUDA Graph 捕获的最大批次大小
        # 推测解码时需要更大的空间（每个序列可能有 1 + num_spec 个 token）
        self.decode_cudagraph_max_bs: int = (
            self.vllm_config.scheduler_config.max_num_seqs * (self.num_spec + 1)
        )
        if self.compilation_config.max_cudagraph_capture_size is not None:
            self.decode_cudagraph_max_bs = min(
                self.decode_cudagraph_max_bs,
                self.compilation_config.max_cudagraph_capture_size,
            )

        # 预分配 CUDA Graph 所需的固定大小缓冲区
        # 推测解码相关的状态索引张量
        self.spec_state_indices_tensor: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs, self.num_spec + 1),
            dtype=torch.int32,
            device=device,
        )
        # 非推测解码的状态索引张量
        self.non_spec_state_indices_tensor: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs,),
            dtype=torch.int32,
            device=device,
        )
        # 推测解码序列掩码
        self.spec_sequence_masks: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs,),
            dtype=torch.bool,
            device=device,
        )
        # 推测 token 和非推测 token 的索引张量
        self.spec_token_indx: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs * (self.num_spec + 1),),
            dtype=torch.int32,
            device=device,
        )
        self.non_spec_token_indx: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs * (self.num_spec + 1),),
            dtype=torch.int32,
            device=device,
        )
        # 查询起始位置张量
        self.spec_query_start_loc: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs + 1,),
            dtype=torch.int32,
            device=device,
        )
        self.non_spec_query_start_loc: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs + 1,),
            dtype=torch.int32,
            device=device,
        )
        # 接受 token 数张量
        self.num_accepted_tokens: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs,),
            dtype=torch.int32,
            device=device,
        )

    def build(  # type: ignore[override]
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        num_accepted_tokens: torch.Tensor | None = None,
        num_decode_draft_tokens_cpu: torch.Tensor | None = None,
        fast_build: bool = False,
    ) -> GDNAttentionMetadata:
        """
        构建 GDN 注意力元数据。

        处理流程：
        1. 计算隐状态索引（block table tensor）
        2. 判断是否有推测解码序列
        3. 根据是否有推测解码，选择不同的处理路径
        4. 计算 FLA 分块元数据（仅 prefill）
        5. 计算 causal_conv1d 元数据（仅 prefill）
        6. 处理 CUDA Graph padding

        参数：
            common_prefix_len: 共享前缀长度
            common_attn_metadata: 通用注意力元数据
            num_accepted_tokens: 推测解码接受的 token 数
            num_decode_draft_tokens_cpu: CPU 上的 draft token 数（>=0 表示推测序列）
            fast_build: 是否快速构建
        """
        m = common_attn_metadata

        query_start_loc = m.query_start_loc
        query_start_loc_cpu = m.query_start_loc_cpu
        context_lens_tensor = m.compute_num_computed_tokens()
        nums_dict, batch_ptr, token_chunk_offset_ptr = None, None, None
        # 计算隐状态索引
        block_table_tensor = mamba_get_block_table_tensor(
            m.block_table_tensor,
            m.seq_lens,
            self.kv_cache_spec,
            self.vllm_config.cache_config.mamba_cache_mode,
        )

        # 判断是否有推测解码序列
        # num_decode_draft_tokens_cpu >= 0 表示该序列是推测解码序列
        spec_sequence_masks_cpu: torch.Tensor | None = None
        if (
            not self.use_spec_decode
            or num_decode_draft_tokens_cpu is None
            or num_decode_draft_tokens_cpu[num_decode_draft_tokens_cpu >= 0]
            .sum()
            .item()
            == 0
        ):
            # 无推测解码序列
            spec_sequence_masks = None
            num_spec_decodes = 0
        else:
            # 有推测解码序列
            spec_sequence_masks_cpu = num_decode_draft_tokens_cpu >= 0
            num_spec_decodes = spec_sequence_masks_cpu.sum().item()
            if num_spec_decodes == 0:
                spec_sequence_masks = None
                spec_sequence_masks_cpu = None
            else:
                spec_sequence_masks = spec_sequence_masks_cpu.to(
                    query_start_loc.device, non_blocking=True
                )

        if spec_sequence_masks is None:
            # 无推测解码：使用标准的 decode/prefill 拆分
            num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
                split_decodes_and_prefills(m, decode_threshold=1)
            )
            num_spec_decode_tokens = 0
            spec_token_indx = None
            non_spec_token_indx = None
            spec_state_indices_tensor = None
            non_spec_state_indices_tensor = block_table_tensor[:, 0]
            spec_query_start_loc = None
            non_spec_query_start_loc = query_start_loc
            non_spec_query_start_loc_cpu = query_start_loc_cpu
            num_accepted_tokens = None
        else:
            # 有推测解码：将请求分为 spec 和 non-spec 两组
            query_lens = query_start_loc[1:] - query_start_loc[:-1]
            assert spec_sequence_masks_cpu is not None
            query_lens_cpu = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]

            # Use CPU tensors to avoid CPU-GPU sync
            # 使用 CPU 张量避免 CPU-GPU 同步
            non_spec_query_lens_cpu = query_lens_cpu[~spec_sequence_masks_cpu]
            num_decodes = (non_spec_query_lens_cpu == 1).sum().item()
            # Exclude zero-length padded sequences from prefill count.
            # 排除零长度的填充序列
            num_zero_len = (non_spec_query_lens_cpu == 0).sum().item()
            num_prefills = non_spec_query_lens_cpu.size(0) - num_decodes - num_zero_len
            num_decode_tokens = num_decodes
            num_prefill_tokens = (
                non_spec_query_lens_cpu.sum().item() - num_decode_tokens
            )
            num_spec_decode_tokens = (
                query_lens_cpu.sum().item() - num_prefill_tokens - num_decode_tokens
            )

            # num_decodes and num_spec_decodes are mutually exclusive.
            # Reclassify non-spec decodes as prefills when spec decodes
            # exist — the prefill kernel handles 1-token sequences with
            # initial state correctly, producing identical results.
            # 当 spec decode 存在时，将 non-spec decode 重分类为 prefill，
            # 因为 prefill kernel 可以正确处理带初始状态的单 token 序列
            if num_decodes > 0 and num_spec_decodes > 0:
                num_prefills += num_decodes
                num_prefill_tokens += num_decode_tokens
                num_decodes = 0
                num_decode_tokens = 0

            if num_prefills == 0 and num_decodes == 0:
                # 仅有推测解码序列
                spec_token_size = min(
                    num_spec_decodes * (self.num_spec + 1),
                    query_start_loc_cpu[-1].item(),
                )
                spec_token_indx = torch.arange(
                    spec_token_size,
                    dtype=torch.int32,
                    device=query_start_loc.device,
                )
                non_spec_token_indx = torch.empty(
                    0, dtype=torch.int32, device=query_start_loc.device
                )
                # Filter by spec_sequence_masks to exclude padded sequences
                # 过滤掉填充序列
                spec_state_indices_tensor = block_table_tensor[
                    spec_sequence_masks_cpu, : self.num_spec + 1
                ]
                non_spec_state_indices_tensor = None
                # Padded sequences are always at the back, so the first
                # num_spec_decodes + 1 entries of query_start_loc already
                # contain the correct cumulative token counts.
                # 填充序列在末尾，前 num_spec_decodes+1 个条目已包含正确的累积计数
                spec_query_start_loc = query_start_loc[: num_spec_decodes + 1]
                non_spec_query_start_loc = None
                non_spec_query_start_loc_cpu = None
            else:
                # 同时有推测解码和非推测序列
                # 使用排序将 token 按 spec/non-spec 分组
                spec_token_masks = torch.repeat_interleave(
                    spec_sequence_masks,
                    query_lens,
                    output_size=query_start_loc_cpu[-1].item(),
                )
                index = torch.argsort(spec_token_masks, stable=True)
                num_non_spec_tokens = num_prefill_tokens + num_decode_tokens
                non_spec_token_indx = index[:num_non_spec_tokens]
                spec_token_indx = index[num_non_spec_tokens:]

                spec_state_indices_tensor = block_table_tensor[
                    spec_sequence_masks_cpu, : self.num_spec + 1
                ]
                non_spec_state_indices_tensor = block_table_tensor[
                    ~spec_sequence_masks_cpu, 0
                ]

                # 分别计算 spec 和 non-spec 的查询起始位置
                spec_query_start_loc = torch.zeros(
                    num_spec_decodes + 1,
                    dtype=torch.int32,
                    device=query_start_loc.device,
                )
                torch.cumsum(
                    query_lens[spec_sequence_masks_cpu],
                    dim=0,
                    out=spec_query_start_loc[1:],
                )
                non_spec_query_start_loc = torch.zeros(
                    query_lens.size(0) - num_spec_decodes + 1,
                    dtype=torch.int32,
                    device=query_start_loc.device,
                )
                torch.cumsum(
                    query_lens[~spec_sequence_masks_cpu],
                    dim=0,
                    out=non_spec_query_start_loc[1:],
                )
                non_spec_query_start_loc_cpu = torch.zeros(
                    query_lens_cpu.size(0) - num_spec_decodes + 1,
                    dtype=torch.int32,
                )
                torch.cumsum(
                    query_lens_cpu[~spec_sequence_masks_cpu],
                    dim=0,
                    out=non_spec_query_start_loc_cpu[1:],
                )

            assert num_accepted_tokens is not None
            num_accepted_tokens = num_accepted_tokens[spec_sequence_masks_cpu]

        # 计算 FLA 分块元数据（仅 prefill 需要）
        chunk_indices: torch.Tensor | None = None
        chunk_offsets: torch.Tensor | None = None
        if num_prefills > 0:
            from vllm.model_executor.layers.fla.ops.utils import FLA_CHUNK_SIZE

            if self.gdn_prefill_backend == "cutedsl":
                # 使用 CuteDSL 后端
                from vllm.model_executor.layers.mamba.ops.gdn_chunk_cutedsl import (
                    prepare_metadata_cutedsl,
                )

                assert non_spec_query_start_loc is not None
                assert non_spec_query_start_loc_cpu is not None
                total_tokens = int(non_spec_query_start_loc_cpu[-1].item())
                chunk_indices, chunk_offsets = prepare_metadata_cutedsl(
                    non_spec_query_start_loc,
                    total_tokens,
                    FLA_CHUNK_SIZE,
                )
            else:
                # 使用 Triton 或 FlashInfer 后端
                gpu_device = query_start_loc.device
                # Only prefill batches use FLA chunk ops.
                # Pre-compute on CPU and async-copy to GPU to avoid
                # GPU->CPU sync (.tolist()) in prepare_chunk_indices.
                # 在 CPU 上预计算，然后异步复制到 GPU，避免 GPU->CPU 同步
                from vllm.model_executor.layers.fla.ops.index import (
                    prepare_chunk_indices,
                    prepare_chunk_offsets,
                )

                assert non_spec_query_start_loc_cpu is not None
                chunk_indices = prepare_chunk_indices(
                    non_spec_query_start_loc_cpu, FLA_CHUNK_SIZE
                ).to(device=gpu_device, non_blocking=True)
                chunk_offsets = prepare_chunk_offsets(
                    non_spec_query_start_loc_cpu, FLA_CHUNK_SIZE
                ).to(device=gpu_device, non_blocking=True)

        if num_prefills > 0:
            # 判断 prefill 请求是否有初始状态
            has_initial_state = context_lens_tensor > 0
            if spec_sequence_masks_cpu is not None:
                has_initial_state = has_initial_state[~spec_sequence_masks_cpu]
                assert non_spec_query_start_loc_cpu is not None
            # 计算 causal_conv1d 的元数据
            nums_dict, batch_ptr, token_chunk_offset_ptr = (
                compute_causal_conv1d_metadata(
                    non_spec_query_start_loc_cpu,
                    device=query_start_loc.device,
                )
            )
        else:
            has_initial_state = None

        # Function code counted on either presency non-spec decode or spec decode,
        # but not both.
        # 断言：decode 和 spec_decode 互斥（不能同时存在）
        assert not (num_decodes > 0 and num_spec_decodes > 0), (
            f"num_decodes: {num_decodes}, num_spec_decodes: {num_spec_decodes}"
        )

        # Prepare tensors for cudagraph
        # Note: m.num_actual_tokens is already padded by the model runner for CUDAGraph
        # 为 CUDA Graph 准备张量（m.num_actual_tokens 已被 model runner 填充）
        batch_size = m.num_actual_tokens

        if (
            self.use_full_cuda_graph
            and num_prefills == 0
            and num_decodes == 0
            and num_spec_decodes <= self.decode_cudagraph_max_bs
            and num_spec_decode_tokens <= self.decode_cudagraph_max_bs
        ):
            # 纯推测解码的 CUDA Graph padding
            assert spec_sequence_masks is not None
            self.spec_state_indices_tensor[:num_spec_decodes].copy_(
                spec_state_indices_tensor, non_blocking=True
            )
            spec_state_indices_tensor = self.spec_state_indices_tensor[:batch_size]
            spec_state_indices_tensor[num_spec_decodes:].fill_(NULL_BLOCK_ID)

            self.spec_sequence_masks[:num_spec_decodes].copy_(
                spec_sequence_masks[:num_spec_decodes], non_blocking=True
            )
            spec_sequence_masks = self.spec_sequence_masks[:batch_size]
            spec_sequence_masks[num_spec_decodes:].fill_(False)

            assert non_spec_token_indx is not None and spec_token_indx is not None
            self.non_spec_token_indx[: non_spec_token_indx.size(0)].copy_(
                non_spec_token_indx, non_blocking=True
            )
            non_spec_token_indx = self.non_spec_token_indx[
                : non_spec_token_indx.size(0)
            ]

            self.spec_token_indx[: spec_token_indx.size(0)].copy_(
                spec_token_indx, non_blocking=True
            )
            spec_token_indx = self.spec_token_indx[: spec_token_indx.size(0)]

            self.spec_query_start_loc[: num_spec_decodes + 1].copy_(
                spec_query_start_loc, non_blocking=True
            )
            spec_num_query_tokens = spec_query_start_loc[-1]  # type: ignore[index]
            spec_query_start_loc = self.spec_query_start_loc[: batch_size + 1]
            spec_query_start_loc[num_spec_decodes + 1 :].fill_(spec_num_query_tokens)

            self.num_accepted_tokens[:num_spec_decodes].copy_(
                num_accepted_tokens, non_blocking=True
            )
            num_accepted_tokens = self.num_accepted_tokens[:batch_size]
            num_accepted_tokens[num_spec_decodes:].fill_(1)

        if (
            self.use_full_cuda_graph
            and num_prefills == 0
            and num_spec_decodes == 0
            and num_decodes <= self.decode_cudagraph_max_bs
        ):
            # 纯 non-spec decode 的 CUDA Graph padding
            self.non_spec_state_indices_tensor[:num_decodes].copy_(
                non_spec_state_indices_tensor, non_blocking=True
            )
            non_spec_state_indices_tensor = self.non_spec_state_indices_tensor[
                :batch_size
            ]
            non_spec_state_indices_tensor[num_decodes:].fill_(NULL_BLOCK_ID)

            self.non_spec_query_start_loc[: num_decodes + 1].copy_(
                non_spec_query_start_loc, non_blocking=True
            )
            non_spec_num_query_tokens = non_spec_query_start_loc[-1]  # type: ignore[index]
            non_spec_query_start_loc = self.non_spec_query_start_loc[:batch_size + 1]
            non_spec_query_start_loc[num_decodes + 1 :].fill_(non_spec_num_query_tokens)

        # 组装最终的注意力元数据
        attn_metadata = GDNAttentionMetadata(
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_spec_decodes=num_spec_decodes,
            num_spec_decode_tokens=num_spec_decode_tokens,
            num_actual_tokens=m.num_actual_tokens,
            has_initial_state=has_initial_state,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            spec_query_start_loc=spec_query_start_loc,
            non_spec_query_start_loc=non_spec_query_start_loc,
            spec_state_indices_tensor=spec_state_indices_tensor,
            non_spec_state_indices_tensor=non_spec_state_indices_tensor,
            spec_sequence_masks=spec_sequence_masks,
            spec_token_indx=spec_token_indx,
            non_spec_token_indx=non_spec_token_indx,
            num_accepted_tokens=num_accepted_tokens,
            nums_dict=nums_dict,
            batch_ptr=batch_ptr,
            token_chunk_offset_ptr=token_chunk_offset_ptr,
        )
        return attn_metadata

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ):
        """
        为 CUDA Graph 全图捕获构建元数据。

        当前仅支持 decode-only 的全图捕获。
        确保请求数和 token 数不超过 CUDA Graph 捕获的最大批次大小。
        """
        m = common_attn_metadata

        assert (
            m.num_reqs <= self.decode_cudagraph_max_bs
            and m.num_actual_tokens <= self.decode_cudagraph_max_bs
        ), (
            f"GDN only supports decode-only full CUDAGraph capture. "
            f"Make sure batch size ({m.num_reqs}) <= "
            f"cudagraph capture sizes ({self.decode_cudagraph_max_bs}), "
            f"and number of tokens ({m.num_actual_tokens}) <= "
            f"cudagraph capture sizes ({self.decode_cudagraph_max_bs})."
        )

        num_accepted_tokens = torch.diff(m.query_start_loc)
        num_decode_draft_tokens_cpu = (num_accepted_tokens - 1).cpu()

        return self.build(0, m, num_accepted_tokens, num_decode_draft_tokens_cpu)
