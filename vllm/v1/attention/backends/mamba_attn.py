# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Mamba 类 SSM（状态空间模型）注意力的公共基类模块。

本模块提供了 Mamba1、Mamba2、ShortConv 等基于状态空间模型（SSM）的注意力后端
所需的公共基础设施。SSM 模型与传统 Transformer 注意力模型的关键区别在于：

1. 状态空间模型使用隐状态（hidden state）而非 KV 缓存来存储历史信息
2. 处理分为 prefill（预填充）和 decode（解码）两个阶段：
   - Prefill 阶段：处理较长的 token 序列，使用分块（chunk）方式处理
   - Decode 阶段：每次处理一个或少量 token，更新隐状态
3. 支持 prefix caching（前缀缓存）：可以缓存和复用已计算的隐状态
4. 支持推测解码（speculative decoding）：通过接受多个 token 提高吞吐量

核心组件：
- BaseMambaAttentionMetadata：所有 Mamba 后端共享的元数据数据类
- BaseMambaAttentionMetadataBuilder：元数据构建器基类，实现公共的元数据计算逻辑

元数据构建流程：
1. 从 CommonAttentionMetadata 中提取序列信息
2. 将请求拆分为 decode 和 prefill 两组
3. 计算隐状态索引（state_indices_tensor）
4. 为 prefill 计算分块元数据（chunk metadata）
5. 为推测解码处理多 token 接受情况
6. 处理 CUDA Graph 捕获的 padding

缓存模式说明：
- "all" 模式：使用分页块表（block_table_tensor）直接索引隐状态
- 其他模式：使用 mamba_get_block_table_tensor 计算特定的索引映射
"""

import abc
from dataclasses import dataclass, replace
from typing import Any, ClassVar, TypeVar

import torch

from vllm.config import VllmConfig
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.attention.backend import (
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

# 类型变量，用于泛型元数据构建器
M = TypeVar("M", bound="BaseMambaAttentionMetadata")


@dataclass
class BaseMambaAttentionMetadata:
    """
    Mamba 类 SSM 注意力的公共元数据数据类。

    存储 Mamba1、Mamba2 等 SSM 模型在单步推理中所需的全部元数据。

    基本信息：
    - num_prefills: prefill 请求数量
    - num_prefill_tokens: prefill token 总数
    - num_decodes: decode 请求数量
    - num_decode_tokens: decode token 总数
    - num_reqs: 总请求数量

    Prefill 专用张量（无 prefill 请求时为 None）：
    - has_initial_states_p: 每个 prefill 请求是否有初始状态（即是否为续写）
    - query_start_loc_p: prefill 请求的查询起始位置
    - num_computed_tokens_p: 每个 prefill 请求已计算的 token 数
    - state_indices_tensor_p: prefill 请求的隐状态索引

    Decode 专用张量（无 decode 请求时为 None）：
    - state_indices_tensor_d: decode 请求的隐状态索引
    - query_start_loc_d: decode 请求的查询起始位置
    - num_accepted_tokens: 推测解码中每个序列接受的 token 数（含 bonus token，最小为 1）

    Prefix caching 专用张量（"all" 模式）：
    - block_idx_last_scheduled_token: 最后一个调度 token 的块索引
    - block_idx_first_scheduled_token_p: prefill 请求第一个调度 token 的块索引
    - block_idx_last_computed_token: 最后一个已计算 token 的块索引
    - block_idx_last_scheduled_token_prev_step: 上一步最后一个调度 token 的块索引

    Prefix caching "align" 模式：
    - seq_lens: 每个请求的序列长度

    分块元数据（prefill 专用）：
    - cu_chunk_seqlen_p: 分块累积序列长度，shape=[nchunks+1]
      第 i 个块包含 tokens [cu_chunk_seqlen_p[i], cu_chunk_seqlen_p[i+1])
    - last_chunk_indices_p: 每个序列最后一个块的索引，shape=[batch]

    causal_conv1d 相关（Triton 实现）：
    - nums_dict: 各种计数字典
    - batch_ptr: 批次指针
    - token_chunk_offset_ptr: token 块偏移指针
    """

    num_prefills: int
    num_prefill_tokens: int
    num_decodes: int
    num_decode_tokens: int
    num_reqs: int

    # The following tensors only contain prefill requests and will be None if
    # the batch has no prefill requests.
    # 以下张量仅包含 prefill 请求，批次中无 prefill 时为 None
    has_initial_states_p: torch.Tensor | None
    query_start_loc_p: torch.Tensor | None
    num_computed_tokens_p: torch.Tensor | None
    state_indices_tensor_p: torch.Tensor | None

    # The following tensors are used for decode requests and
    # speculative decoding compatibility, and will be None if the batch
    # has no decode requests.
    # 以下张量用于 decode 请求和推测解码兼容，批次中无 decode 时为 None
    state_indices_tensor_d: torch.Tensor | None
    query_start_loc_d: torch.Tensor | None  # shape: [num_decodes + 1,]

    # Number of accepted tokens for each spec sequence (for loading correct checkpoint)
    # Includes the bonus token (so minimum is 1)
    # 推测解码中每个序列接受的 token 数（含 bonus token，最小为 1）
    num_accepted_tokens: torch.Tensor | None  # shape: [batch,]

    # The following tensors are only used for prefix caching in all mode and
    # are None if disabled
    # 以下张量仅用于 "all" 模式的 prefix caching，禁用时为 None
    block_idx_last_scheduled_token: torch.Tensor | None
    block_idx_first_scheduled_token_p: torch.Tensor | None
    block_idx_last_computed_token: torch.Tensor | None
    block_idx_last_scheduled_token_prev_step: torch.Tensor | None

    # The following tensor is only used for prefix caching in align mode
    # 以下张量仅用于 "align" 模式的 prefix caching
    seq_lens: torch.Tensor

    # cu_chunk_seqlen_p is a tensor of shape (nchunks+1,) that contains, for
    # each chunk, its offsets into the varlen sequence dimension. It is defined
    # such that the i-th chunk contains tokens from cu_chunk_seqlen_p[i] to
    # cu_chunk_seqlen_p[i+1].
    # 分块累积序列长度，用于 SSD kernel 的分块计算
    cu_chunk_seqlen_p: torch.Tensor | None = None
    # last_chunk_indices_p is a tensor of shape (batch,) that contains the
    # index of the last chunk for every sequence in the (prefill) batch.
    # 每个 prefill 序列最后一个块的索引
    last_chunk_indices_p: torch.Tensor | None = None

    # The following attributes are for triton implementation of causal_conv1d
    # 以下属性用于 causal_conv1d 的 Triton 实现
    nums_dict: dict | None = None
    batch_ptr: torch.Tensor | None = None
    token_chunk_offset_ptr: torch.Tensor | None = None


class BaseMambaAttentionMetadataBuilder(AttentionMetadataBuilder[M], abc.ABC):
    """
    Mamba 类 SSM 注意力元数据构建器基类。

    提供 Mamba1、Mamba2、ShortConv 等后端共享的元数据构建逻辑。

    主要职责：
    1. 区分 decode 和 prefill 请求，进行批次重排序
    2. 计算隐状态索引（state_indices_tensor）
    3. 为 prefill 计算分块元数据（chunk metadata）
    4. 处理推测解码（speculative decoding）的多 token 情况
    5. 管理 CUDA Graph 捕获时的 padding

    CUDA Graph 支持：
    - _cudagraph_support = UNIFORM_BATCH，仅支持均匀批次的 CUDA Graph
    - decode 阶段可以使用全图捕获（full cudagraph capture）
    - prefill 阶段暂不支持全图捕获

    缓存模式：
    - "all" 模式：所有隐状态存储在分页块表中，支持 prefix caching
    - 其他模式：使用简化的状态索引映射

    推测解码支持：
    - 当启用推测解码时，decode 请求可能包含多个 token（1 + num_spec_tokens）
    - 需要存储 num_accepted_tokens 来正确加载检查点
    - 推测解码模式下禁用 update_block_table 功能
    """

    metadata_cls: type[M]
    # decode 请求的重排序阈值：query 长度 <= 此值的视为 decode
    reorder_batch_threshold: int = 1
    # CUDA Graph 支持级别：仅均匀批次
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH

    # Will be disabled if speculative decoding is used
    # 是否支持块表更新功能，推测解码时禁用
    supports_update_block_table: bool = True

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

        # Enable speculative decoding support
        # 获取推测解码配置
        self.speculative_config = vllm_config.speculative_config
        self.compilation_config = vllm_config.compilation_config
        self.num_spec_tokens: int = vllm_config.num_speculative_tokens
        self.use_spec_decode = self.num_spec_tokens > 0

        assert isinstance(kv_cache_spec, MambaSpec)
        scheduler_config = vllm_config.scheduler_config
        # CUDA Graph 捕获的最大批次大小
        self.decode_cudagraph_max_bs: int = scheduler_config.max_num_seqs
        if self.compilation_config.max_cudagraph_capture_size is not None:
            self.decode_cudagraph_max_bs = min(
                self.decode_cudagraph_max_bs,
                self.compilation_config.max_cudagraph_capture_size,
            )

        if self.vllm_config.cache_config.mamba_cache_mode == "all":
            # "all" 模式：使用分页块表，预分配最大块数的索引张量
            max_num_blocks = (
                cdiv(
                    self.vllm_config.model_config.max_model_len,
                    kv_cache_spec.block_size,
                )
                + kv_cache_spec.num_speculative_blocks
            )
            # TODO: reduce this size as needed for decode-only cudagraph capture
            self.state_indices_tensor_d: torch.Tensor = torch.empty(
                (
                    self.decode_cudagraph_max_bs,
                    max_num_blocks,
                ),
                dtype=torch.int32,
                device=device,
            )
            # 各种 prefix caching 相关的预分配张量
            self.block_idx_last_scheduled_token: torch.Tensor = torch.empty(
                (self.decode_cudagraph_max_bs,),
                dtype=torch.int32,
                device=device,
            )
            self.block_idx_last_computed_token: torch.Tensor = torch.empty(
                (self.decode_cudagraph_max_bs,),
                dtype=torch.int32,
                device=device,
            )
            if self.use_spec_decode:
                self.block_idx_last_scheduled_token_prev_step: torch.Tensor = (
                    torch.empty(
                        (self.decode_cudagraph_max_bs,),
                        dtype=torch.int32,
                        device=device,
                    )
                )
        else:
            # 非 "all" 模式：简化的状态索引
            self.state_indices_tensor_d = torch.empty(
                (self.decode_cudagraph_max_bs, 1 + self.num_spec_tokens),
                dtype=torch.int32,
                device=device,
            )

        # For speculative decoding, we need to store the following buffers
        # for CUDA graph capture during decode
        # 推测解码：预分配接受 token 数的缓冲区
        if self.num_spec_tokens > 0:
            self.decode_num_accepted_tokens: torch.Tensor = torch.empty(
                (self.decode_cudagraph_max_bs,),
                dtype=torch.int32,
                device=device,
            )

        self._init_reorder_batch_threshold(1, self.use_spec_decode)
        if self.use_spec_decode:
            # 推测解码模式下禁用块表更新
            self.supports_update_block_table = False

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> M:
        """
        为 CUDA Graph 全图捕获构建元数据。

        当前仅支持 decode-only 的全图捕获。确保：
        1. 所有请求的 query 长度 <= 1 + num_spec_tokens（即都是 decode）
        2. 请求数量 <= decode_cudagraph_max_bs
        """
        m = common_attn_metadata

        assert (
            m.max_query_len <= 1 + self.num_spec_tokens
            and m.num_reqs <= self.decode_cudagraph_max_bs
        ), (
            "Mamba only supports decode-only full CUDAGraph capture. "
            "Make sure all cudagraph capture sizes <= max_num_seq."
        )

        assert m.max_query_len == 1 + self.num_spec_tokens  # decode-only

        num_accepted_tokens = None
        if self.num_spec_tokens > 0:
            # 从 query_start_loc 计算每个请求接受的 token 数
            num_accepted_tokens = torch.diff(m.query_start_loc)

        prev_last_scheduled_idx = None
        if (
            self.use_spec_decode
            and self.vllm_config.cache_config.mamba_cache_mode == "all"
        ):
            # "all" 模式 + 推测解码：需要初始化上一步的调度索引
            prev_last_scheduled_idx = torch.zeros(
                (m.num_reqs,),
                dtype=torch.int32,
                device=m.query_start_loc.device,
            )

        return self.build(
            0,
            m,
            num_accepted_tokens=num_accepted_tokens,
            prev_last_scheduled_idx=prev_last_scheduled_idx,
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
        *,
        num_accepted_tokens: torch.Tensor | None = None,
        prev_last_scheduled_idx: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> M:
        """
        Default build implementation for Mamba-like attention backends.
        Subclasses (e.g., Mamba2) can override to add additional metadata.
        """
        # 默认的 build 实现，子类（如 Mamba2）可以重写以添加额外元数据
        return self._compute_common_metadata(
            common_attn_metadata,
            num_accepted_tokens=num_accepted_tokens,
            prev_last_scheduled_idx=prev_last_scheduled_idx,
        )

    def _compute_chunk_metadata(
        self,
        chunk_size: int,
        num_prefills: int,
        num_computed_tokens_p_cpu: torch.Tensor,
        query_start_loc_p_cpu: torch.Tensor,
    ) -> tuple[list[int], list[int], list[int]]:
        """
        Compute chunk-specific metadata for Mamba models.

        The code below carefully constructs the chunks such that:
        1. Chunks contain tokens from a *single* sequence only.
        2. For every sequence, we are guaranteed that we can
           retrieve the mamba state *every* chunk_size tokens.
        Constraint (1) dramatically simplifies the mamba kernels.
        Constraint (2) dramatically simplifies the implementation
        of prefix caching for mamba (wip). We need to take care
        of the interaction with chunked prefill in order to
        satisfy constraint (2).
        """
        # 为 Mamba 模型计算分块元数据。
        #
        # 分块约束：
        # 1. 每个块只包含单个序列的 token（简化 kernel 实现）
        # 2. 每隔 chunk_size 个 token 可以检索 Mamba 状态（简化 prefix caching）
        #
        # 对于未对齐的已计算 token，第一个块用于补齐对齐：
        # 例如已计算 100 个 token，chunk_size=64，则第一个块包含
        # tokens [100, 128)（28 个 token），之后的块按 chunk_size 对齐
        cu_chunk_seqlen = []
        seq_idx = []
        last_chunk_indices = []
        seqlen_pos = 0

        for req_idx in range(num_prefills):
            this_num_computed = num_computed_tokens_p_cpu[req_idx].item()
            this_new_tokens = (
                query_start_loc_p_cpu[req_idx + 1].item()
                - query_start_loc_p_cpu[req_idx].item()
            )

            # if computed tokens are not chunk-aligned, use the first
            # chunk to finish it off
            # 如果已计算的 token 未对齐到 chunk 边界，用第一个块补齐
            if this_num_computed % chunk_size != 0:
                seq_idx.append(req_idx)
                cu_chunk_seqlen.append(seqlen_pos)
                # how many tokens to finish the chunk?
                # 需要多少 token 来完成当前块
                chunk_len = (
                    cdiv(this_num_computed, chunk_size) * chunk_size - this_num_computed
                )
                # we can only use at most this_new_tokens
                # 最多只能使用 this_new_tokens 个 token
                chunk_len = min(chunk_len, this_new_tokens)
                seqlen_pos += chunk_len
                this_new_tokens -= chunk_len

            # 将剩余 token 按 chunk_size 分块
            n_chunks = cdiv(this_new_tokens, chunk_size)
            for chunk in range(n_chunks):
                seq_idx.append(req_idx)
                cu_chunk_seqlen.append(seqlen_pos)
                chunk_len = min(chunk_size, this_new_tokens)
                seqlen_pos += chunk_len
                this_new_tokens -= chunk_len

            assert this_new_tokens == 0
            last_chunk_indices.append(len(cu_chunk_seqlen) - 1)

        cu_chunk_seqlen.append(seqlen_pos)

        return cu_chunk_seqlen, seq_idx, last_chunk_indices

    def _build_chunk_metadata_tensors(
        self,
        chunk_size: int,
        common: M,
        common_attn_metadata: CommonAttentionMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute chunk metadata and return as device tensors.
        Returns (cu_chunk_seqlen_p, seq_idx_p, last_chunk_indices_p).
        """
        # 计算分块元数据并返回 GPU 张量
        num_reqs = common.num_reqs
        num_prefills = common.num_prefills
        num_decode_tokens = common.num_decode_tokens

        # Derive prefill context lengths from CPU data only.
        # `seq_lens_cpu_upper_bound` is precise for prefill rows in all modes
        # (including async spec decode), so this avoids the D2H sync that
        # `compute_num_computed_tokens().cpu()` would force.
        # 仅使用 CPU 数据推导 prefill 上下文长度，避免 D2H 同步
        seq_lens_cpu = common_attn_metadata.seq_lens_cpu_upper_bound
        assert seq_lens_cpu is not None
        query_start_loc_p_cpu = (
            common_attn_metadata.query_start_loc_cpu[-num_prefills - 1 :]
            - num_decode_tokens
        )
        prefill_query_lens_cpu = query_start_loc_p_cpu[1:] - query_start_loc_p_cpu[:-1]
        num_computed_tokens_p_cpu = (
            seq_lens_cpu[num_reqs - num_prefills : num_reqs] - prefill_query_lens_cpu
        )

        cu_chunk_seqlen, seq_idx, last_chunk_indices = self._compute_chunk_metadata(
            chunk_size,
            num_prefills,
            num_computed_tokens_p_cpu,
            query_start_loc_p_cpu,
        )

        device = common_attn_metadata.query_start_loc.device
        # Build on pinned CPU and upload non-blocking to avoid the synchronous
        # H2D copy that `torch.as_tensor(list, device=cuda)` would force.
        # 在 pinned CPU 上构建，然后异步上传到 GPU，避免同步 H2D 拷贝
        cu_chunk_seqlen_p = async_tensor_h2d(
            cu_chunk_seqlen, dtype=torch.int32, device=device
        )
        seq_idx_p = async_tensor_h2d(seq_idx, dtype=torch.int32, device=device)
        last_chunk_indices_p = async_tensor_h2d(
            last_chunk_indices, dtype=torch.int32, device=device
        )
        return cu_chunk_seqlen_p, seq_idx_p, last_chunk_indices_p

    def _compute_prefix_caching_block_indices(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        mamba_block_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        计算 prefix caching 所需的块索引。

        对于 "all" 缓存模式，需要知道：
        1. block_idx_last_computed_token：最后已计算 token 的块索引
        2. block_idx_first_scheduled_token：第一个待调度 token 的块索引
        3. block_idx_last_scheduled_token：最后一个待调度 token 的块索引

        这些索引用于确定哪些块需要从缓存中恢复，哪些需要新计算。
        """
        num_computed_tokens = common_attn_metadata.compute_num_computed_tokens()
        # Block index of the last computed token
        block_idx_last_computed_token = cdiv(num_computed_tokens, mamba_block_size) - 1
        # which is <= block index for the first scheduled token
        block_idx_first_scheduled_token = (
            cdiv(num_computed_tokens + 1, mamba_block_size) - 1
        )
        # which is <= block index of the last scheduled token
        block_idx_last_scheduled_token = (
            cdiv(common_attn_metadata.seq_lens, mamba_block_size) - 1
        )
        # -1 in case it's non-computed and causes later issues with indexing
        # 下限裁剪为 0，防止负索引
        block_idx_last_computed_token = torch.clamp(
            block_idx_last_computed_token, min=0
        )
        # -1 in the case we have a padded request (0 seq-len)
        block_idx_last_scheduled_token = torch.clamp(
            block_idx_last_scheduled_token, min=0
        )

        return (
            block_idx_last_computed_token,
            block_idx_first_scheduled_token,
            block_idx_last_scheduled_token,
        )

    def _compute_common_metadata(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        *,
        num_accepted_tokens: torch.Tensor | None = None,
        prev_last_scheduled_idx: torch.Tensor | None = None,
    ) -> M:
        """
        Compute metadata common to both Mamba1 and Mamba2.

        核心元数据计算方法，Mamba1 和 Mamba2 共享此逻辑。

        处理流程：
        1. 将请求拆分为 decode 和 prefill 两组
        2. 处理推测解码中的单 token prefill（视为 decode）
        3. 计算隐状态索引
        4. 为 prefill 计算初始状态标记和分块元数据
        5. 处理 CUDA Graph 捕获的 padding
        """
        num_reqs = common_attn_metadata.num_reqs

        # Treat multi-token queries as decode requests when
        # speculative decoding is enabled. Otherwise, use the
        # default decode threshold to prevent misclassification
        # of prefill queries as decode requests.
        # 推测解码时，多 token 查询视为 decode；否则使用默认阈值
        decode_threshold = (
            self.reorder_batch_threshold if num_accepted_tokens is not None else 1
        )

        # FULL-CG dispatch is shape-based, so one-token prefills with
        # prior Mamba state can replay a decode graph while `is_prefilling`
        # is still true. Treat them as decode/update rows. This is required
        # for NIXL disagg's h(N-1)->N recompute path and for sporadic
        # final single-token prefill chunks that land in a `uniform` FULL-CG
        # batch. Relies on `reorder` putting short extends before pure prefills.
        # 全图 CUDA Graph 基于形状调度，因此有先前状态的单 token prefill
        # 可以回放 decode 图。将它们重新分类为 decode 行。
        is_prefilling = common_attn_metadata.is_prefilling
        assert is_prefilling is not None
        seq_lens_cpu = common_attn_metadata.seq_lens_cpu_upper_bound
        assert seq_lens_cpu is not None
        query_lens_cpu = torch.diff(common_attn_metadata.query_start_loc_cpu)
        single_token_prefill_rows = is_prefilling & (query_lens_cpu == 1)
        # First-token prefills have no prior Mamba state and must stay prefills.
        # 首个 token 的 prefill 没有先前状态，必须保持为 prefill
        has_prior_state = seq_lens_cpu > 1
        prefill_to_decode = single_token_prefill_rows & has_prior_state
        if torch.any(prefill_to_decode).item():
            is_prefilling = is_prefilling.clone()
            is_prefilling[prefill_to_decode] = False
            common_attn_metadata = common_attn_metadata.replace(
                is_prefilling=is_prefilling
            )

        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(
                common_attn_metadata,
                decode_threshold=decode_threshold,
                treat_short_extends_as_decodes=False,
            )
        )

        # Need flags to indicate if there are initial states
        # 初始化各种标志和张量
        has_initial_states_p = None
        query_start_loc_p = None
        query_start_loc_d = None
        num_computed_tokens = None
        num_computed_tokens_p = None

        # for prefix caching
        block_idx_first_scheduled_token = None
        block_idx_first_scheduled_token_p = None
        block_idx_last_computed_token = None
        block_idx_last_scheduled_token = None
        block_idx_last_scheduled_token_prev_step = None

        # for causal_conv1d
        nums_dict, batch_ptr, token_chunk_offset_ptr = None, None, None

        if self.vllm_config.cache_config.mamba_cache_mode == "all":
            # "all" 模式：使用分页块表作为状态索引
            num_computed_tokens = common_attn_metadata.compute_num_computed_tokens()

            # Return a tensor of shape (#requests, #max blocks)
            state_indices_tensor = common_attn_metadata.block_table_tensor
            # Additional cache-related variables:
            mamba_block_size = self.kv_cache_spec.block_size
            (
                block_idx_last_computed_token,
                block_idx_first_scheduled_token,
                block_idx_last_scheduled_token,
            ) = self._compute_prefix_caching_block_indices(
                common_attn_metadata, mamba_block_size
            )
            if self.use_spec_decode and prev_last_scheduled_idx is not None:
                # 推测解码：使用上一步的调度索引，回退到基于计算 token 的索引
                fallback = torch.clamp(
                    (num_computed_tokens - 1) // mamba_block_size, min=0
                )
                block_idx_last_scheduled_token_prev_step = torch.where(
                    prev_last_scheduled_idx >= 0,
                    prev_last_scheduled_idx,
                    fallback,
                )
        else:
            # 非 "all" 模式：使用 mamba_get_block_table_tensor 计算索引
            state_indices_tensor = mamba_get_block_table_tensor(
                common_attn_metadata.block_table_tensor,
                common_attn_metadata.seq_lens,
                self.kv_cache_spec,
                self.vllm_config.cache_config.mamba_cache_mode,
            )

        # 确保 state_indices_tensor 至少是 2 维
        if state_indices_tensor.dim() == 1:
            state_indices_tensor = state_indices_tensor.unsqueeze(-1)

        # 将状态索引拆分为 decode 和 prefill 两组
        state_indices_tensor_d, state_indices_tensor_p = torch.split(
            state_indices_tensor,
            [num_decodes, num_prefills],
            dim=0,
        )
        if self.vllm_config.cache_config.mamba_cache_mode != "all":
            # 非 "all" 模式：截取所需列数
            state_indices_tensor_d = state_indices_tensor_d[
                :, : 1 + self.num_spec_tokens
            ]
            state_indices_tensor_p = state_indices_tensor_p[:, 0]

        # Sometimes even with specdec enabled we get single-token prefill chunks that
        # should be treated as decodes but don't have num_accepted_tokens set.
        # These should be fine to process as non-spec decodes since there's only
        # one token, so no risk of placing accepted tokens in the wrong slot.
        # 有时推测解码模式下的单 token prefill 没有设置 num_accepted_tokens，
        # 可以安全地作为非推测 decode 处理
        if num_decodes > 0 and self.use_spec_decode and num_accepted_tokens is not None:
            query_start_loc_d = common_attn_metadata.query_start_loc[: num_decodes + 1]
            num_accepted_tokens = num_accepted_tokens[:num_decodes]

        if num_prefills > 0:
            if num_computed_tokens is None:
                num_computed_tokens = common_attn_metadata.compute_num_computed_tokens()

            query_start_loc_p_cpu = (
                common_attn_metadata.query_start_loc_cpu[-num_prefills - 1 :]
                - num_decode_tokens
            )
            query_start_loc_p = (
                common_attn_metadata.query_start_loc[-num_prefills - 1 :]
                - num_decode_tokens
            )
            # 判断每个 prefill 是否有初始状态（已计算 > 0 个 token）
            has_initial_states_p = (
                num_computed_tokens[num_reqs - num_prefills : num_reqs] > 0
            )

            # 计算 causal_conv1d 的元数据（Triton 实现需要）
            nums_dict, batch_ptr, token_chunk_offset_ptr = (
                compute_causal_conv1d_metadata(
                    query_start_loc_p_cpu,
                    device=common_attn_metadata.query_start_loc.device,
                )
            )

            if self.vllm_config.cache_config.mamba_cache_mode == "all":
                assert num_computed_tokens is not None
                num_computed_tokens_p = num_computed_tokens[
                    num_reqs - num_prefills : num_reqs
                ]
                assert block_idx_first_scheduled_token is not None
                block_idx_first_scheduled_token_p = block_idx_first_scheduled_token[
                    num_reqs - num_prefills : num_reqs
                ]

        # 组装元数据对象
        metadata = self.metadata_cls(
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            query_start_loc_p=query_start_loc_p,
            has_initial_states_p=has_initial_states_p,
            state_indices_tensor_p=state_indices_tensor_p,
            state_indices_tensor_d=state_indices_tensor_d,
            num_accepted_tokens=num_accepted_tokens,
            query_start_loc_d=query_start_loc_d,
            block_idx_last_scheduled_token=block_idx_last_scheduled_token,
            block_idx_first_scheduled_token_p=block_idx_first_scheduled_token_p,
            block_idx_last_computed_token=block_idx_last_computed_token,
            block_idx_last_scheduled_token_prev_step=(
                block_idx_last_scheduled_token_prev_step
            ),
            num_computed_tokens_p=num_computed_tokens_p,
            num_reqs=num_reqs,
            seq_lens=common_attn_metadata.seq_lens,
            nums_dict=nums_dict,
            batch_ptr=batch_ptr,
            token_chunk_offset_ptr=token_chunk_offset_ptr,
        )

        return self._update_metadata_for_cudagraph_capture(metadata)

    def _update_metadata_for_cudagraph_capture(
        self,
        metadata: M,
    ) -> M:
        """
        Update the metadata for cudagraph capture.
        Currently, only decode is supported for full cudagraphs with Mamba.

        为 CUDA Graph 捕获更新元数据。

        当满足以下条件时，将元数据张量复制到预分配的固定大小缓冲区中，
        并用 NULL_BLOCK_ID 或 0 填充多余位置：
        1. 无 prefill 请求
        2. decode 请求数 <= decode_cudagraph_max_bs
        3. 启用了全图 CUDA Graph

        这确保 CUDA Graph 的输入张量形状在不同推理步骤间保持一致。
        """
        state_indices_tensor_d = metadata.state_indices_tensor_d
        query_start_loc_d = metadata.query_start_loc_d
        num_accepted_tokens = metadata.num_accepted_tokens
        block_idx_last_scheduled_token = metadata.block_idx_last_scheduled_token
        block_idx_last_computed_token = metadata.block_idx_last_computed_token
        block_idx_last_scheduled_token_prev_step = (
            metadata.block_idx_last_scheduled_token_prev_step
        )
        if (
            metadata.num_prefills == 0
            and metadata.num_decodes <= self.decode_cudagraph_max_bs
            and self.compilation_config.cudagraph_mode.has_full_cudagraphs()
        ):
            padded_bs = metadata.num_reqs
            # 将实际数据复制到预分配缓冲区的前部
            self.state_indices_tensor_d[: metadata.num_decodes].copy_(
                state_indices_tensor_d, non_blocking=True
            )
            state_indices_tensor_d = self.state_indices_tensor_d[:padded_bs]
            # 用 NULL_BLOCK_ID 填充多余位置
            state_indices_tensor_d[metadata.num_decodes :] = NULL_BLOCK_ID

            if self.use_spec_decode and num_accepted_tokens is not None:
                assert query_start_loc_d is not None
                query_start_loc_d = query_start_loc_d[: padded_bs + 1]
                self.decode_num_accepted_tokens[: metadata.num_decodes].copy_(
                    num_accepted_tokens, non_blocking=True
                )
                num_accepted_tokens = self.decode_num_accepted_tokens[:padded_bs]
                num_accepted_tokens[metadata.num_decodes :] = (
                    1  # pad with 1st slot index
                )

            if self.vllm_config.cache_config.mamba_cache_mode == "all":
                assert block_idx_last_scheduled_token is not None
                assert block_idx_last_computed_token is not None
                self.block_idx_last_scheduled_token[: metadata.num_decodes].copy_(
                    block_idx_last_scheduled_token[: metadata.num_decodes],
                    non_blocking=True,
                )
                block_idx_last_scheduled_token = self.block_idx_last_scheduled_token[
                    :padded_bs
                ]
                block_idx_last_scheduled_token[metadata.num_decodes :] = 0

                self.block_idx_last_computed_token[: metadata.num_decodes].copy_(
                    block_idx_last_computed_token[: metadata.num_decodes],
                    non_blocking=True,
                )
                block_idx_last_computed_token = self.block_idx_last_computed_token[
                    :padded_bs
                ]
                block_idx_last_computed_token[metadata.num_decodes :] = 0

                if (
                    self.use_spec_decode
                    and block_idx_last_scheduled_token_prev_step is not None
                ):
                    self.block_idx_last_scheduled_token_prev_step[
                        : metadata.num_decodes
                    ].copy_(
                        block_idx_last_scheduled_token_prev_step[
                            : metadata.num_decodes
                        ],
                        non_blocking=True,
                    )
                    block_idx_last_scheduled_token_prev_step = (
                        self.block_idx_last_scheduled_token_prev_step[:padded_bs]
                    )
                    block_idx_last_scheduled_token_prev_step[metadata.num_decodes :] = 0

        # 使用 dataclasses.replace 创建新的元数据对象（不可变更新）
        return replace(
            metadata,
            state_indices_tensor_d=state_indices_tensor_d,
            query_start_loc_d=query_start_loc_d,
            num_accepted_tokens=num_accepted_tokens,
            block_idx_last_scheduled_token=block_idx_last_scheduled_token,
            block_idx_last_computed_token=block_idx_last_computed_token,
            block_idx_last_scheduled_token_prev_step=(
                block_idx_last_scheduled_token_prev_step
            ),
        )

    def update_block_table(
        self,
        metadata: M,
        blk_table: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> M:
        """
        更新块表并重新计算状态索引。

        在增量解码过程中，当块表发生变化时（如新块分配），
        需要重新计算隐状态索引张量。

        参数：
            metadata: 当前的注意力元数据
            blk_table: 新的块表
            slot_mapping: 槽位映射

        返回：
            更新后的元数据对象
        """
        state_indices_tensor = mamba_get_block_table_tensor(
            blk_table,
            metadata.seq_lens,
            self.kv_cache_spec,
            self.vllm_config.cache_config.mamba_cache_mode,
        )
        if state_indices_tensor.dim() == 1:
            state_indices_tensor = state_indices_tensor.unsqueeze(-1)

        assert (
            metadata.num_prefills + metadata.num_decodes
            == state_indices_tensor.shape[0]
        ), (
            "Mismatch in number of requests when updating block table."
            f" Expected {metadata.num_prefills + metadata.num_decodes}, "
            f"got {state_indices_tensor.shape[0]}."
        )

        # 拆分为 decode 和 prefill 两组
        state_indices_tensor_d, state_indices_tensor_p = torch.split(
            state_indices_tensor,
            [metadata.num_decodes, metadata.num_prefills],
            dim=0,
        )
        if self.vllm_config.cache_config.mamba_cache_mode != "all":
            state_indices_tensor_d = state_indices_tensor_d[
                :, : 1 + self.num_spec_tokens
            ]
            state_indices_tensor_p = state_indices_tensor_p[:, 0]

        new_metadata = replace(
            metadata,
            state_indices_tensor_d=state_indices_tensor_d,
            state_indices_tensor_p=state_indices_tensor_p,
        )

        return self._update_metadata_for_cudagraph_capture(new_metadata)
