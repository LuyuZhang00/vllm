# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
EAGLE 通用工具模块。

本模块提供了 EAGLE 投机解码器的通用工具函数，主要包括：
1. EAGLE 草稿模型的加载
2. 权重共享机制（embedding 和 lm_head）

权重共享是 EAGLE 的一个重要优化：
- 草稿模型复用目标模型的 embedding 层和 lm_head 层
- 这样可以减少内存占用，并保持词表的一致性
- 对于 MTP（Multi-Token Prediction）模型，还会共享 topk_indices_buffer
"""

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.distributed.parallel_state import get_pp_group
from vllm.model_executor.model_loader import get_model


def _should_share(eagle: nn.Module, flag: str, draft, target) -> bool:
    """
    判断是否应该共享权重。

    根据以下条件判断是否共享：
    1. 草稿模型没有自己的副本（has_own_xxx 为 False）
    2. 或者草稿模型的副本与目标模型相同

    参数:
        eagle (nn.Module): EAGLE 模型实例。
        flag (str): 共享标志名称（如 "has_own_embed_tokens", "has_own_lm_head"）。
        draft: 草稿模型的权重（如 embedding 或 lm_head）。
        target: 目标模型的权重。

    返回:
        bool: 是否应该共享权重。

    注意：
        - torch.equal 在 GPU 上会分配一个与输入大小相同的 bool mask
        - 当 GPU 内存不足时，回退到 CPU 比较
    """
    """Share when the draft has no own copy, or its copy matches the target."""

    if not getattr(eagle, flag, False) or draft is None:
        return True
    if target is None:
        return False
    # torch.equal on GPU allocates a bool mask the size of the input.
    # Use the faster GPU path when there is plenty of headroom;
    # otherwise compare on CPU.
    w = draft.weight
    if w.is_cuda and torch.cuda.mem_get_info(w.device)[0] < w.numel() * 2:
        return torch.equal(w.cpu(), target.weight.cpu())
    return torch.equal(w, target.weight)


def load_eagle_model(target_model: nn.Module, vllm_config: VllmConfig) -> nn.Module:
    """
    加载 EAGLE 草稿模型。

    加载草稿模型并与目标模型共享权重（embedding 和 lm_head）。

    参数:
        target_model (nn.Module): 目标模型实例。
        vllm_config (VllmConfig): vLLM 全局配置。

    返回:
        nn.Module: 加载完成的 EAGLE 草稿模型。

    流程:
        1. 使用 set_model_tag("eagle_head") 标记模型加载（用于编译优化）
        2. 加载草稿模型
        3. 如果不是 pipeline parallel，共享 embedding 权重
        4. 共享 lm_head 权重
        5. 对于 MTP 模型，修复每层的 shared_head.head
        6. 共享 topk_indices_buffer（如果存在）
    """
    from vllm.compilation.backends import set_model_tag

    speculative_config = vllm_config.speculative_config
    assert speculative_config is not None
    draft_model_config = speculative_config.draft_model_config
    # 使用 eagle_head 标记，用于编译优化
    with set_model_tag("eagle_head"):
        eagle_model = get_model(
            vllm_config=vllm_config, model_config=draft_model_config
        )

    # 获取目标模型和草稿模型的内部模型
    target_language_model = (
        target_model.get_language_model()
        if hasattr(target_model, "get_language_model")
        else target_model
    )
    target_inner = target_language_model.model
    draft_inner = eagle_model.model

    # Skip embedding sharing under PP — each rank owns its own embedding.
    # 在 pipeline parallel 下跳过 embedding 共享，每个 rank 拥有自己的 embedding
    if get_pp_group().world_size == 1:
        target_embed = getattr(target_inner, "embed_tokens", None) or getattr(
            target_inner, "embedding", None
        )
        draft_embed = getattr(draft_inner, "embed_tokens", None)
        if target_embed is not None and _should_share(
            eagle_model, "has_own_embed_tokens", draft_embed, target_embed
        ):
            if draft_embed is not None:
                del draft_inner.embed_tokens
            draft_inner.embed_tokens = target_embed

    # 共享 lm_head 权重
    target_lm_head = getattr(target_model, "lm_head", None)
    draft_lm_head = getattr(eagle_model, "lm_head", None)
    if target_lm_head is not None and _should_share(
        eagle_model, "has_own_lm_head", draft_lm_head, target_lm_head
    ):
        if draft_lm_head is not None:
            del eagle_model.lm_head
        eagle_model.lm_head = target_lm_head

        # MTP layers route logits through layer.shared_head.head, not
        # eagle_model.lm_head, so the per-layer copies need fixing up too.
        # MTP 层通过 layer.shared_head.head 路由 logits，需要修复每层的副本
        layers = getattr(draft_inner, "layers", None)
        if layers is not None:
            items = layers.values() if isinstance(layers, nn.ModuleDict) else layers
            for layer in items:
                sh = getattr(layer, "shared_head", None)
                if sh is not None and hasattr(sh, "head"):
                    del sh.head
                    sh.head = target_lm_head

    # MTP also shares a topk_indices_buffer between target and draft.
    # MTP 还在目标和草稿之间共享 topk_indices_buffer
    if hasattr(target_inner, "topk_indices_buffer"):
        if hasattr(draft_inner, "topk_indices_buffer"):
            del draft_inner.topk_indices_buffer
        draft_inner.topk_indices_buffer = target_inner.topk_indices_buffer

    return eagle_model
