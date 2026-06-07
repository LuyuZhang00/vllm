# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
短卷积注意力后端（Short Convolution Attention Backend）。

本模块实现了基于短卷积（Short Convolution）的注意力后端。短卷积是一种
用于状态空间模型（SSM）的局部特征提取机制，在状态更新之前对输入进行
一维因果卷积操作。

短卷积在 SSM 模型中的作用：
1. 局部上下文建模：通过小窗口的卷积捕捉相邻 token 之间的局部依赖关系
2. 通道混合：在不同特征通道之间进行信息交换
3. 非线性增强：为线性 SSM 引入非线性变换能力

短卷积 vs 因果卷积（Causal Conv1d）：
- 短卷积：窗口较小（通常 4-16），计算简单
- 因果卷积：较长的卷积窗口，用于 Mamba 等模型的初始特征提取

本模块非常简洁，完全依赖 BaseMambaAttentionMetadata 和
BaseMambaAttentionMetadataBuilder 基类提供的公共基础设施：
- 元数据类型：直接复用 BaseMambaAttentionMetadata
- 元数据构建：使用基类的 _compute_common_metadata 方法

这是因为短卷积 SSM 模型的状态管理逻辑与 Mamba1 完全相同，
无需额外的元数据字段或构建逻辑。
"""

from dataclasses import dataclass

from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.attention.backends.mamba_attn import (
    BaseMambaAttentionMetadata,
    BaseMambaAttentionMetadataBuilder,
)


class ShortConvAttentionBackend(AttentionBackend):
    """
    短卷积注意力后端类。

    声明短卷积 SSM 模型的后端能力。
    标记为 is_ssm=True 以表明这是状态空间模型后端。
    """

    @staticmethod
    def get_name() -> str:
        """返回后端名称标识。"""
        return "SHORT_CONV_ATTN"

    @staticmethod
    def get_builder_cls() -> type["ShortConvAttentionMetadataBuilder"]:
        """返回短卷积元数据构建器类。"""
        return ShortConvAttentionMetadataBuilder

    @classmethod
    def is_ssm(cls) -> bool:
        """
        标识此为状态空间模型（SSM）后端。

        SSM 后端使用隐状态（而非 KV 缓存）存储历史信息，
        支持前缀缓存的隐状态恢复和推测解码。
        """
        return True


@dataclass
class ShortConvAttentionMetadata(BaseMambaAttentionMetadata):
    """
    短卷积注意力元数据数据类。

    直接继承 BaseMambaAttentionMetadata，不添加额外字段。

    这是因为短卷积 SSM 模型的状态管理逻辑与 Mamba1 完全相同：
    1. 使用相同的隐状态索引（state_indices_tensor）
    2. 使用相同的分块元数据（chunk metadata）
    3. 使用相同的 decode/prefill 拆分逻辑
    """
    pass


class ShortConvAttentionMetadataBuilder(
    BaseMambaAttentionMetadataBuilder[ShortConvAttentionMetadata]
):
    """
    短卷积注意力元数据构建器。

    直接继承 BaseMambaAttentionMetadataBuilder，使用基类的全部构建逻辑。

    metadata_cls 设置为 ShortConvAttentionMetadata，
    使基类 _compute_common_metadata 方法能正确实例化子类元数据对象。
    """

    metadata_cls = ShortConvAttentionMetadata
