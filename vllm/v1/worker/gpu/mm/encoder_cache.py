# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
多模态编码器缓存模块 (Multi-Modal Encoder Cache Module)

本模块实现了多模态编码器输出的缓存管理。在多模态推理中（如图像-文本模型），
编码器（如视觉 Transformer）的计算开销很大。通过缓存编码器输出，可以避免
对相同多模态输入重复编码。

缓存结构：
1. mm_features: 请求级缓存，存储每个请求的多模态特征规格
   - key: 请求 ID (req_id)
   - value: 多模态特征规格列表 (list[MultiModalFeatureSpec])

2. encoder_outputs: 编码器输出缓存，存储编码器的计算结果
   - key: 多模态输入的唯一标识符 (mm_hash)
   - value: 编码器输出张量 (torch.Tensor)

生命周期：
1. 请求到来时，通过 add_request 注册多模态特征
2. 编码器执行后，输出通过 mm_hash 缓存
3. 后续步骤中，相同 mm_hash 的输入可以直接使用缓存的输出
4. 请求完成时，通过 remove_request 清理特征
"""
import torch

from vllm.multimodal.inputs import MultiModalFeatureSpec


class EncoderCache:
    """多模态编码器输出缓存。

    管理两个层次的缓存：
    1. 请求级：每个请求的多模态特征规格
    2. 全局级：编码器输出的 hash 缓存（跨请求共享）
    """

    def __init__(self):
        # 请求 ID -> 多模态特征规格列表
        self.mm_features: dict[str, list[MultiModalFeatureSpec]] = {}
        # 多模态输入 hash -> 编码器输出张量
        self.encoder_outputs: dict[str, torch.Tensor] = {}

    def add_request(
        self, req_id: str, mm_features: list[MultiModalFeatureSpec]
    ) -> None:
        """注册请求的多模态特征。

        Args:
            req_id: 请求 ID
            mm_features: 多模态特征规格列表
        """
        self.mm_features[req_id] = mm_features

    def remove_request(self, req_id: str) -> None:
        """移除请求的多模态特征。

        Args:
            req_id: 请求 ID
        """
        self.mm_features.pop(req_id, None)

    def reset_mm_cache(self) -> None:
        """
        Clear the multi-modal cache that was used during profiling,
        but no longer needed during inference.
        """
        # TODO: Implement MM budget for encoder dummy run
        pass

    def reset_encoder_cache(self) -> None:
        """Clear the GPU-side encoder cache storing vision embeddings.

        This should be called when model weights are updated to ensure
        stale embeddings computed with old weights are not reused.
        """
        self.encoder_outputs.clear()

    def free_encoder_cache(self, mm_hash: str) -> None:
        """释放特定多模态输入的编码器输出缓存。

        Args:
            mm_hash: 多模态输入的唯一标识符
        """
        self.encoder_outputs.pop(mm_hash, None)
