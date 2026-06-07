# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
注意力机制工具模块。

本模块负责 vLLM v1 引擎中注意力相关的初始化和管理，包括：
1. KV 缓存规格（KVCacheSpec）的发现和收集
2. 注意力后端（Attention Backend）的初始化和分组
3. KV 缓存的分配、形状重塑和绑定
4. 注意力元数据（Attention Metadata）的构建
5. Slot 映射的计算

核心流程：
  a. 初始化阶段：发现模型中的注意力层 -> 按后端和缓存规格分组 -> 初始化后端
  b. 分配阶段：为每个注意力组分配 KV 缓存张量 -> 重塑为目标形状 -> 绑定到模型
  c. 推理阶段：构建注意力元数据 -> 计算 slot 映射 -> 传递给注意力内核
"""
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, cast

import torch

from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    CommonAttentionMetadata,
)
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheConfig,
    KVCacheSpec,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.worker.gpu.model_states.interface import ModelSpecificAttnMetadata
from vllm.v1.worker.utils import (
    AttentionGroup,
    add_kv_sharing_layers_to_kv_cache_groups,
    bind_kv_cache,
    prepare_kernel_block_sizes,
)


@dataclass(frozen=True)
class AttentionCGSupportInfo:
    """CUDA Graph 对注意力后端的支持信息。

    属性:
        min_cg_support: 所有注意力后端中最低的 CUDA Graph 支持级别
        min_cg_attn_backend: 具有最低支持级别的后端名称（用于日志和调试）
    """
    min_cg_support: AttentionCGSupport = AttentionCGSupport.ALWAYS
    min_cg_attn_backend: str | None = None


def get_kv_cache_spec(vllm_config: VllmConfig) -> dict[str, KVCacheSpec]:
    """获取模型中所有需要 KV 缓存的注意力层的缓存规格。

    遍历模型中所有继承自 AttentionLayerBase 的层，收集每个层的 KV 缓存规格。
    跳过以下类型的层：
    - KV 共享层（kv_sharing_target_layer_name 不为空的层）
    - 不需要 KV 缓存的层（如纯编码器注意力）

    参数:
        vllm_config: vLLM 全局配置对象

    返回:
        dict[str, KVCacheSpec]: 层名到 KV 缓存规格的映射
    """
    kv_cache_spec: dict[str, KVCacheSpec] = {}
    layer_type = cast(type[Any], AttentionLayerBase)
    attn_layers = get_layers_from_vllm_config(vllm_config, layer_type)
    for layer_name, attn_module in attn_layers.items():
        # 跳过 KV 共享层——它们将使用目标层的 KV 缓存
        if getattr(attn_module, "kv_sharing_target_layer_name", None):
            # This layer will use KV cache of the sharing target layer.
            continue
        # 跳过不需要 KV 缓存的模块（如纯编码器注意力）
        if spec := attn_module.get_kv_cache_spec(vllm_config):
            kv_cache_spec[layer_name] = spec
    return kv_cache_spec


def get_shared_kv_cache_layers(vllm_config: VllmConfig):
    """获取所有 KV 共享层的映射关系。

    KV 共享是指某些注意力层复用其他层的 KV 缓存，而非独立分配。
    这在某些模型架构中用于减少内存占用。

    参数:
        vllm_config: vLLM 全局配置对象

    返回:
        dict[str, str]: 共享层名 -> 目标层名 的映射
    """
    attn_layers = get_layers_from_vllm_config(vllm_config, Attention)
    return {
        layer_name: kv_tgt_layer
        for layer_name, attn_module in attn_layers.items()
        if (kv_tgt_layer := attn_module.kv_sharing_target_layer_name)
    }


def init_attn_backend(
    kv_cache_config: KVCacheConfig,
    vllm_config: VllmConfig,
    device: torch.device,
    active_layer_names: set[str] | None = None,
) -> tuple[list[list[AttentionGroup]], AttentionCGSupportInfo, list[int]]:
    """初始化注意力后端，分为三个阶段。

    阶段 1：发现注意力组
    - 遍历每个 KV 缓存组中的注意力层
    - 按（后端类名, 缓存规格）对层进行分组，相同规格的层共享元数据构建器

    阶段 2：选择内核块大小
    - 为每个 KV 缓存组选择一个所有后端都支持的内核块大小

    阶段 3：创建元数据构建器并确定 CUDA Graph 支持
    - 为每个注意力组创建元数据构建器
    - 分享工作区缓冲区以减少内存分配
    - 检查每个后端的 CUDA Graph 支持级别

    参数:
        kv_cache_config: KV 缓存配置
        vllm_config: vLLM 全局配置
        device: 计算设备
        active_layer_names: 可选，仅初始化指定的活跃层

    返回:
        (attn_groups, attn_cg_support_info, kernel_block_sizes):
        - 注意力组列表、CUDA Graph 支持信息、内核块大小列表
    """
    # Phase 1: discover attention groups for each kv cache group.
    attn_groups: list[list[AttentionGroup]] = []

    # 将 KV 共享层添加到其目标层所在的 KV 缓存组中，
    # 以便在 Phase 1 中一起被发现和处理
    add_kv_sharing_layers_to_kv_cache_groups(
        get_shared_kv_cache_layers(vllm_config), kv_cache_config.kv_cache_groups
    )

    # Phase 1: discover attention groups for each kv cache group.
    for kv_cache_group_id, kv_cache_group_spec in enumerate(
        kv_cache_config.kv_cache_groups
    ):
        layer_names = kv_cache_group_spec.layer_names
        if active_layer_names is not None:
            layer_names = list(active_layer_names.intersection(layer_names))

        layer_type = cast(type[Any], AttentionLayerBase)
        attn_layers = get_layers_from_vllm_config(vllm_config, layer_type, layer_names)

        # group_map: 按 (后端全限定名, 缓存规格) 分组
        group_map: dict[tuple[tuple[str, str], KVCacheSpec], AttentionGroup] = {}
        # group_order: 记录分组的创建顺序，保持层的排列顺序
        group_order: list[tuple[tuple[str, str], KVCacheSpec]] = []

        for layer_name in layer_names:
            attn_backend = attn_layers[layer_name].get_attn_backend()

            layer_kv_cache_spec: KVCacheSpec = kv_cache_group_spec.kv_cache_spec
            if isinstance(layer_kv_cache_spec, UniformTypeKVCacheSpecs):
                layer_kv_cache_spec = layer_kv_cache_spec.kv_cache_specs[layer_name]

            key = (attn_backend.full_cls_name(), layer_kv_cache_spec)
            if key not in group_map:
                group_map[key] = AttentionGroup(
                    attn_backend, [layer_name], layer_kv_cache_spec, kv_cache_group_id
                )
                group_order.append(key)
            else:
                group_map[key].layer_names.append(layer_name)

        attn_groups.append([group_map[key] for key in group_order])

    # Phase 2: pick a kernel block size per kv cache group that is supported
    # by all backends within that group.
    kernel_block_sizes = prepare_kernel_block_sizes(kv_cache_config, attn_groups)

    # Phase 3: create metadata builders and determine cudagraph support.
    attn_backend_workspace: torch.Tensor | None = None
    min_cg_support = AttentionCGSupport.ALWAYS
    min_cg_attn_backend = None
    for kv_cache_group_id, groups in enumerate(attn_groups):
        kernel_block_size = None
        if kv_cache_group_id < len(kernel_block_sizes):
            kernel_block_size = kernel_block_sizes[kv_cache_group_id]
        for group in groups:
            group.create_metadata_builders(
                vllm_config=vllm_config,
                device=device,
                kernel_block_size=kernel_block_size,
                num_metadata_builders=1,
            )
            builder = group.get_metadata_builder(0)
            # 多个后端共享同一个工作区缓冲区，减少 GPU 内存分配
            if attn_backend_workspace is None:
                if hasattr(builder, "_get_workspace_buffer"):
                    attn_backend_workspace = builder._get_workspace_buffer()
            else:
                if hasattr(builder, "set_workspace_buffer"):
                    builder.set_workspace_buffer(attn_backend_workspace)
            # 检查该注意力后端对 CUDA Graph 的支持情况
            cg_support = builder.get_cudagraph_support(
                vllm_config,
                cast(AttentionSpec, group.kv_cache_spec),
            )
            # 记录所有后端中最低的 CUDA Graph 支持级别
            if cg_support.value < min_cg_support.value:
                min_cg_support = cg_support
                min_cg_attn_backend = group.backend.__name__

    attn_cg_support_info = AttentionCGSupportInfo(
        min_cg_support=min_cg_support, min_cg_attn_backend=min_cg_attn_backend
    )
    return attn_groups, attn_cg_support_info, kernel_block_sizes


def _allocate_kv_cache(
    kv_cache_config: KVCacheConfig, shared_layers: dict[str, str], device: torch.device
):
    """为所有 KV 缓存张量分配原始 GPU 内存。

    每个 KVCacheTensor 代表一个连续的 GPU 内存块，可能被多个层共享。
    分配的是 int8 类型的零初始化张量，后续会根据需要重塑为目标形状。

    参数:
        kv_cache_config: KV 缓存配置
        shared_layers: KV 共享层映射
        device: 计算设备

    返回:
        dict[str, torch.Tensor]: 层名 -> 原始 KV 缓存张量 的映射
    """
    kv_cache_raw_tensors: dict[str, torch.Tensor] = {}
    for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
        tensor = torch.zeros(kv_cache_tensor.size, dtype=torch.int8, device=device)
        for layer_name in kv_cache_tensor.shared_by:
            kv_cache_raw_tensors[layer_name] = tensor

    # 验证所有需要缓存的层都已正确初始化
    layer_names = set()
    for group in kv_cache_config.kv_cache_groups:
        for layer_name in group.layer_names:
            layer_names.add(layer_name)
    assert layer_names == (kv_cache_raw_tensors.keys() | shared_layers.keys()), (
        "Some layers are not correctly initialized"
    )
    return kv_cache_raw_tensors


def _reshape_kv_cache(
    attn_groups: Sequence[AttentionGroup],
    kv_cache_raw_tensors: dict[str, torch.Tensor],
    cache_dtype: str,
    kernel_block_sizes: list[int],
    shared_kv_cache_layers: dict[str, str],
) -> dict[str, Any]:
    """将原始 KV 缓存张量重塑为注意力后端所需的形状。

    根据不同类型的缓存规格进行处理：
    - AttentionSpec: 重塑为注意力后端期望的多维张量（如 [num_blocks, 2, num_heads, head_size]）
    - MambaSpec: 为 Mamba 状态创建多个步进张量视图
    - 压缩规格（如 DeepSeek V4）: 使用 storage_block_size 而非 block_size

    参数:
        attn_groups: 注意力组列表（已展平）
        kv_cache_raw_tensors: 原始 KV 缓存张量字典
        cache_dtype: 缓存数据类型的字符串表示
        kernel_block_sizes: 每个缓存组的内核块大小
        shared_kv_cache_layers: KV 共享层映射

    返回:
        dict[str, Any]: 层名 -> 重塑后的 KV 缓存张量 的映射
    """
    kv_caches: dict[str, Any] = {}
    has_attn, has_mamba = False, False

    for group in attn_groups:
        if group.kv_cache_group_id >= len(kernel_block_sizes):
            continue

        kv_cache_spec = group.kv_cache_spec
        if kv_cache_spec.storage_block_size != kv_cache_spec.block_size:
            # 对于应用压缩的规格（如 DeepSeek V4），使用 storage_block_size
            # 作为内核块大小
            kernel_block_size = kv_cache_spec.storage_block_size
        else:
            kernel_block_size = kernel_block_sizes[group.kv_cache_group_id]

        for layer_name in group.layer_names:
            if layer_name in shared_kv_cache_layers:
                # 共享层——张量将在后面别名到其目标层
                continue

            kv_raw_tensor = kv_cache_raw_tensors[layer_name]
            assert kv_raw_tensor.numel() % kv_cache_spec.page_size_bytes == 0
            num_blocks = kv_raw_tensor.numel() // kv_cache_spec.page_size_bytes

            if isinstance(kv_cache_spec, AttentionSpec):
                has_attn = True
                # 使用 storage_block_size：对于未压缩的规格等于 block_size，
                # 对于压缩规格（DeepSeek V4）则更小
                num_blocks_per_kv_block = (
                    kv_cache_spec.storage_block_size // kernel_block_size
                )
                kernel_num_blocks = num_blocks * num_blocks_per_kv_block
                kv_cache_shape = group.backend.get_kv_cache_shape(
                    kernel_num_blocks,
                    kernel_block_size,
                    kv_cache_spec.num_kv_heads,
                    kv_cache_spec.head_size,
                    cache_dtype_str=cache_dtype,
                )

                # 获取后端期望的 KV 缓存维度排列顺序
                # FIXME(woosuk): Add kv_cache_stride_order to all attention backends.
                try:
                    kv_cache_stride_order = group.backend.get_kv_cache_stride_order()
                    assert len(kv_cache_stride_order) == len(kv_cache_shape)
                except (AttributeError, NotImplementedError):
                    kv_cache_stride_order = tuple(range(len(kv_cache_shape)))

                # 按后端期望的顺序重排维度
                kv_cache_shape = tuple(kv_cache_shape[i] for i in kv_cache_stride_order)
                inv_order = [
                    kv_cache_stride_order.index(i)
                    for i in range(len(kv_cache_stride_order))
                ]

                dtype = kv_cache_spec.dtype
                kv_tensor = kv_raw_tensor.view(dtype)
                if kv_cache_spec.page_size_padded is not None:
                    # 使用步进视图处理包含 padding 的 page_size_bytes。
                    # 这与 gpu_model_runner.py 中 MambaSpec 的处理模式相同。
                    # NOTE: 假设 kv_cache_shape[0] == num_blocks
                    # （即第一个物理维度是块索引），这对所有当前后端成立
                    # （MLA, FlashAttention, TritonAttention 等）。
                    dtype_size = get_dtype_size(dtype)
                    page_stride = kv_cache_spec.page_size_bytes // dtype_size
                    strides = list(torch.empty(kv_cache_shape).stride())
                    strides[inv_order[0]] = page_stride
                    kv_cache = torch.as_strided(
                        kv_tensor,
                        size=kv_cache_shape,
                        stride=tuple(strides),
                    )
                else:
                    # 无 padding——安全使用连续视图
                    kv_cache = kv_tensor.view(kv_cache_shape)
                kv_caches[layer_name] = kv_cache.permute(*inv_order)

            elif isinstance(kv_cache_spec, MambaSpec):
                has_mamba = True
                # Mamba 状态由多个不同形状和类型的张量组成
                state_tensors = []
                storage_offset_bytes = 0
                for shape, dtype in zip(kv_cache_spec.shapes, kv_cache_spec.dtypes):
                    dtype_size = get_dtype_size(dtype)
                    num_element_per_page = kv_cache_spec.page_size_bytes // dtype_size
                    target_shape = (num_blocks, *shape)
                    stride = torch.empty(target_shape).stride()
                    target_stride = (num_element_per_page, *stride[1:])
                    assert storage_offset_bytes % dtype_size == 0
                    tensor = torch.as_strided(
                        kv_raw_tensor.view(dtype),
                        size=target_shape,
                        stride=target_stride,
                        storage_offset=storage_offset_bytes // dtype_size,
                    )
                    state_tensors.append(tensor)
                    storage_offset_bytes += stride[0] * dtype_size
                kv_caches[layer_name] = state_tensors
            else:
                raise NotImplementedError(
                    f"Unsupported KV cache spec type: {type(kv_cache_spec)}"
                )

    # 对于混合架构（同时包含注意力和 Mamba 层），需要更新注意力层的布局
    if has_attn and has_mamba:
        _update_hybrid_attention_layout(
            attn_groups=attn_groups,
            kv_caches=kv_caches,
            kernel_block_sizes=kernel_block_sizes,
            cache_dtype=cache_dtype,
        )

    # 将共享层的 KV 缓存映射到其目标层
    for layer_name, target_layer_name in shared_kv_cache_layers.items():
        kv_caches[layer_name] = kv_caches[target_layer_name]

    return kv_caches


def _update_hybrid_attention_layout(
    attn_groups: Iterable[AttentionGroup],
    kv_caches: dict[str, Any],
    kernel_block_sizes: list[int],
    cache_dtype: str,
) -> None:
    """更新混合架构中注意力层的 KV 缓存布局。

    在混合架构（如 Jamba = Mamba + Attention）中，Mamba 和注意力层
    共享同一块 KV 缓存内存。由于 Mamba 层使用不同的存储布局，
    需要调整注意力层的步进（stride）以适配共享内存布局。

    参数:
        attn_groups: 注意力组列表
        kv_caches: KV 缓存字典
        kernel_block_sizes: 内核块大小列表
        cache_dtype: 缓存数据类型
    """
    for group in attn_groups:
        if group.kv_cache_group_id >= len(kernel_block_sizes):
            continue

        kv_cache_spec = group.kv_cache_spec
        if not isinstance(kv_cache_spec, AttentionSpec):
            continue
        block_dim = group.backend.get_kv_cache_block_dim(
            kernel_block_sizes[group.kv_cache_group_id],
            kv_cache_spec.num_kv_heads,
            kv_cache_spec.head_size,
            cache_dtype_str=cache_dtype,
        )
        # 如果 KV 缓存布局的第一个维度已经是 num_blocks，无需调整
        if block_dim == 0:
            continue

        assert block_dim == 1, (
            "Expected the dim `num_blocks` at the second dim when updating"
            " the kvcache's layout of full attention layer"
        )

        for layer_name in group.layer_names:
            if layer_name not in kv_caches:
                # 共享层——将在本次处理后别名到目标层
                continue

            kv_cache = kv_caches[layer_name]
            if kv_cache.shape[0] == 2:
                assert kv_cache.shape[1] != 2, (
                    f"Cannot determine layout for tensor of shape {kv_cache.shape}"
                )
                hidden_size = kv_cache.shape[2:].numel()
                # 通过 as_strided_ 原地修改步进，使第一个维度（2=K/V）
                # 每次跳过整个 hidden_size
                kv_cache.as_strided_(
                    size=kv_cache.shape,
                    stride=(
                        hidden_size,
                        2 * hidden_size,
                        *kv_cache.stride()[2:],
                    ),
                )


def init_kv_cache(
    runner_kv_caches: list[torch.Tensor | list[torch.Tensor]],
    forward_context: dict[str, Any],
    kv_cache_config: KVCacheConfig,
    attn_groups: list[list[AttentionGroup]],
    device: torch.device,
    cache_dtype: str,
    kernel_block_sizes: list[int],
    vllm_config: VllmConfig,
) -> dict[str, Any]:
    """初始化 KV 缓存的完整流程。

    执行步骤：
    1. 获取共享 KV 缓存层的映射关系
    2. 分配原始 KV 缓存张量（GPU 内存）
    3. 将原始张量重塑为各注意力后端期望的形状
    4. 将缓存绑定到模型的注意力层

    参数:
        runner_kv_caches: 模型运行器的 KV 缓存列表
        forward_context: 前向传播上下文
        kv_cache_config: KV 缓存配置
        attn_groups: 注意力组列表
        device: 计算设备
        cache_dtype: 缓存数据类型
        kernel_block_sizes: 内核块大小列表
        vllm_config: vLLM 全局配置

    返回:
        dict[str, Any]: 层名 -> KV 缓存张量 的映射
    """
    shared_kv_cache_layers = get_shared_kv_cache_layers(vllm_config)
    kv_cache_raw_tensors = _allocate_kv_cache(
        kv_cache_config, shared_kv_cache_layers, device
    )
    flattened_attn_groups = list(group for groups in attn_groups for group in groups)
    kv_caches = _reshape_kv_cache(
        attn_groups=flattened_attn_groups,
        kv_cache_raw_tensors=kv_cache_raw_tensors,
        kernel_block_sizes=kernel_block_sizes,
        cache_dtype=cache_dtype,
        shared_kv_cache_layers=shared_kv_cache_layers,
    )
    bind_kv_cache(kv_caches, forward_context, runner_kv_caches)
    return kv_caches


def build_slot_mappings_by_layer(
    slot_mappings: torch.Tensor, kv_cache_config: KVCacheConfig
) -> dict[str, torch.Tensor]:
    """为每个注意力层构建 slot 映射。

    Slot 映射将 token 的逻辑位置映射到 KV 缓存中的物理存储位置。
    同一个 KV 缓存组中的所有层共享相同的 slot 映射。

    参数:
        slot_mappings: 按 KV 缓存组组织的 slot 映射 [num_groups, num_tokens]
        kv_cache_config: KV 缓存配置

    返回:
        dict[str, torch.Tensor]: 层名 -> slot 映射 的字典
    """
    slot_mappings_by_layer: dict[str, torch.Tensor] = {}
    kv_cache_groups = kv_cache_config.kv_cache_groups
    for slot_mapping, kv_cache_group in zip(slot_mappings, kv_cache_groups):
        for layer_name in kv_cache_group.layer_names:
            slot_mappings_by_layer[layer_name] = slot_mapping
    return slot_mappings_by_layer


def build_attn_metadata(
    attn_groups: list[list[AttentionGroup]],
    num_reqs: int,
    num_tokens: int,
    query_start_loc_gpu: torch.Tensor,
    query_start_loc_cpu: torch.Tensor,
    max_query_len: int,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    block_tables: Sequence[torch.Tensor],
    slot_mappings: torch.Tensor,
    kv_cache_config: KVCacheConfig,
    seq_lens_cpu_upper_bound: torch.Tensor | None = None,
    dcp_local_seq_lens: torch.Tensor | None = None,
    positions: torch.Tensor | None = None,
    model_specific_attn_metadata: ModelSpecificAttnMetadata | None = None,
    for_cudagraph_capture: bool = False,
) -> dict[str, Any]:
    """为所有注意力层构建注意力元数据。

    这是注意力计算的核心准备工作。为每个 KV 缓存组创建
    CommonAttentionMetadata，然后通过每个注意力组的元数据构建器
    生成后端特定的注意力元数据。

    参数:
        attn_groups: 注意力组列表
        num_reqs: 当前批次中的请求数量
        num_tokens: 当前批次中的 token 数量
        query_start_loc_gpu: 查询起始位置（GPU）
        query_start_loc_cpu: 查询起始位置（CPU）
        max_query_len: 最大查询长度
        seq_lens: 每个请求的序列长度
        max_seq_len: 最大序列长度
        block_tables: 每个 KV 缓存组的块表
        slot_mappings: 每个 KV 缓存组的 slot 映射
        kv_cache_config: KV 缓存配置
        seq_lens_cpu_upper_bound: CPU 端的序列长度上界估计
        dcp_local_seq_lens: DCP（分布式上下文并行）本地序列长度
        positions: token 位置编码
        model_specific_attn_metadata: 模型特定的注意力元数据
        for_cudagraph_capture: 是否用于 CUDA Graph 捕获

    返回:
        dict[str, Any]: 层名 -> 注意力元数据 的字典
    """
    # 截取实际请求范围（避免使用 padding 部分的数据）
    seq_lens = seq_lens[:num_reqs]
    if dcp_local_seq_lens is not None:
        dcp_local_seq_lens = dcp_local_seq_lens[:num_reqs]
    if seq_lens_cpu_upper_bound is not None:
        seq_lens_cpu_upper_bound = seq_lens_cpu_upper_bound[:num_reqs]

    attn_metadata: dict[str, Any] = {}
    num_kv_cache_groups = len(kv_cache_config.kv_cache_groups)
    for i in range(num_kv_cache_groups):
        block_table = block_tables[i]
        slot_mapping = slot_mappings[i]

        # 获取模型特定的额外参数
        common_attn_metadata_extra_kwargs = (
            model_specific_attn_metadata.get_extra_common_attn_kwargs(i, num_reqs)
            if model_specific_attn_metadata is not None
            else {}
        )
        # 创建通用注意力元数据
        common_attn_metadata = CommonAttentionMetadata(
            query_start_loc=query_start_loc_gpu,
            query_start_loc_cpu=query_start_loc_cpu,
            seq_lens=seq_lens,
            seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
            max_seq_len=max_seq_len,
            num_reqs=num_reqs,
            num_actual_tokens=num_tokens,
            max_query_len=max_query_len,
            block_table_tensor=block_table,
            slot_mapping=slot_mapping,
            causal=True,
            dcp_local_seq_lens=dcp_local_seq_lens,
            positions=positions,
            **common_attn_metadata_extra_kwargs,
        )

        for attn_group in attn_groups[i]:
            attn_metadata_builder = attn_group.get_metadata_builder(0)
            if for_cudagraph_capture:
                # CUDA Graph 捕获模式：构建简化的元数据
                metadata = attn_metadata_builder.build_for_cudagraph_capture(
                    common_attn_metadata
                )
            else:
                # 正常推理模式：构建完整的注意力元数据
                attn_metadata_extra_kwargs = (
                    model_specific_attn_metadata.get_extra_attn_kwargs(
                        attn_metadata_builder,
                        num_reqs,
                    )
                    if model_specific_attn_metadata is not None
                    else {}
                )
                metadata = attn_metadata_builder.build(
                    common_prefix_len=0,
                    common_attn_metadata=common_attn_metadata,
                    **attn_metadata_extra_kwargs,
                )
            # 将元数据分配给该组中的所有层
            for layer_name in attn_group.layer_names:
                attn_metadata[layer_name] = metadata
    return attn_metadata
