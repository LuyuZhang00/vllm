# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Logits处理器接口定义模块。

本模块定义了logits处理器的抽象基类和相关数据结构。

主要类:
1. LogitsProcessor (ABC):
    - logits处理器的抽象基类
    - 所有内置和自定义logits处理器必须继承此类
    - 定义了四个必须实现的方法:
        a) __init__: 初始化处理器
        b) apply: 将处理器应用到logits张量
        c) is_argmax_invariant: 判断是否影响贪心采样
        d) update_state: 更新batch状态

2. BatchUpdate:
    - 不可变数据类，表示batch状态的变化信息
    - 包含添加、删除、移动的请求信息
    - 传递给LogitsProcessor.update_state()方法

3. MoveDirectionality (Enum):
    - 请求移动的方向性枚举
    - UNIDIRECTIONAL: 单向移动 (a -> b)
    - SWAP: 双向交换 (a <-> b)

类型别名:
    RemovedRequest: 被移除请求的batch索引
    AddedRequest: 新添加请求的信息元组
    MovedRequest: 移动请求的信息元组
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING

import torch

from vllm import SamplingParams

if TYPE_CHECKING:
    from vllm.config import VllmConfig


class MoveDirectionality(Enum):
    """请求移动的方向性枚举。

    用于描述batch中请求位置变化的方向。
    """
    # 单向移动: 请求从索引i1移动到i2
    UNIDIRECTIONAL = auto()
    # 双向交换: 索引i1和i2的请求互换位置
    SWAP = auto()


# 被移除请求的batch索引
RemovedRequest = int

# 新添加请求的信息元组: (索引, 采样参数, prompt token IDs, 输出token IDs)
# - 索引: 请求在batch中的位置
# - 采样参数: 该请求的采样配置
# - prompt token IDs: 提示的token IDs（可能为None）
# - 输出token IDs: 到目前为止生成的输出token IDs（是对运行列表的引用）
AddedRequest = tuple[int, SamplingParams, list[int] | None, list[int]]

# 移动请求的信息元组: (索引1, 索引2, 方向性)
# - 索引1: 源位置
# - 索引2: 目标位置
# - 方向性: 单向移动或双向交换
MovedRequest = tuple[int, int, MoveDirectionality]


@dataclass(frozen=True)
class BatchUpdate:
    """持久化batch状态变化信息，用于logits处理器。

    该数据类封装了batch中请求的添加、删除和移动信息，
    在每个解码步骤中传递给每个LogitsProcessor的update_state()方法。

    关键假设:
    - `added` 中的 `output_tok_ids` 列表是对请求运行输出token列表的引用;
      通过此引用，logits处理器始终看到最新的输出token列表。

    注意事项:
    - 添加或移动的请求可能替换具有相同索引的现有请求。
    - 操作应按以下顺序处理: removed, added, moved

    属性:
        batch_size: 当前batch中的请求数量
        removed: 被移除请求的索引序列
        added: 新添加请求的信息序列
        moved: 移动请求的信息序列
    """

    batch_size: int  # 当前batch中的请求数量

    # 请求的添加、删除和移动元数据
    #
    # 关键假设: `added` 中每个元组的 `output_tok_ids` 列表
    # 是对请求运行输出token列表的引用; 通过此引用，logits处理器
    # 始终看到最新的输出token列表。
    #
    # 注意:
    # * 添加或移动的请求可能替换具有相同索引的现有请求。
    # * 操作应按以下顺序处理: removed, added, moved
    removed: Sequence[RemovedRequest]
    added: Sequence[AddedRequest]
    moved: Sequence[MovedRequest]


class LogitsProcessor(ABC):
    """Logits处理器抽象基类。

    所有内置和自定义logits处理器必须继承此类并实现所有抽象方法。

    生命周期:
    1. __init__: 引擎启动时初始化一次
    2. update_state: 每个解步开始时调用，更新batch状态
    3. apply: 采样前调用，将处理器逻辑应用到logits

    关键设计决策:
    - is_argmax_invariant()决定处理器在采样流水线中的应用位置:
        * True (argmax不变): 仅在随机采样前应用（在温度缩放后）
        * False (非argmax不变): 在所有采样前应用（在温度缩放前）
    """

    @classmethod
    def validate_params(cls, sampling_params: SamplingParams):
        """验证采样参数对该logits处理器是否有效。

        对于无效参数应抛出ValueError。

        参数:
            sampling_params: 要验证的采样参数

        返回:
            None（默认不验证）
        """
        return None

    @abstractmethod
    def __init__(
        self, vllm_config: "VllmConfig", device: torch.device, is_pin_memory: bool
    ) -> None:
        """初始化logits处理器。

        参数:
            vllm_config: vLLM全局配置
            device: 计算设备（如cuda:0）
            is_pin_memory: 是否使用pin memory加速CPU-GPU传输
        """
        raise NotImplementedError

    @abstractmethod
    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        """将logits处理器应用到batch logits张量。

        更新后的张量必须返回，但可以原地修改。

        参数:
            logits: 输入logits张量 [batch_size, vocab_size]

        返回:
            处理后的logits张量
        """
        raise NotImplementedError

    @abstractmethod
    def is_argmax_invariant(self) -> bool:
        """判断该logits处理器是否对贪心采样中的argmax计算无影响。

        如果返回True，该处理器仅在随机采样路径中应用（温度缩放后、Top-K/Top-P前）。
        如果返回False，该处理器在所有采样路径中应用（温度缩放前）。

        注意: 对于给定LogitsProcessor子类的不同实例，此方法的返回值可能不同，
        取决于子类的实现。

        返回:
            True如果不影响argmax结果，False如果可能影响
        """
        raise NotImplementedError

    @abstractmethod
    def update_state(
        self,
        batch_update: "BatchUpdate | None",
    ) -> None:
        """在每次前向传播之前调用，当有新的输出token时更新状态。

        该方法负责:
        1. 处理新添加的请求（创建处理器状态）
        2. 处理被删除的请求（清理处理器状态）
        3. 处理被移动的请求（更新索引映射）
        4. 根据新的输出token更新内部状态

        参数:
            batch_update: batch状态变化信息，如果batch组成没有变化则为None
        """
        raise NotImplementedError
