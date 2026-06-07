# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Logits处理器子包。

本子包提供了logits处理器的接口定义、内置实现和状态管理。

主要组件:
1. interface.py - LogitsProcessor抽象基类和BatchUpdate数据结构
2. builtin.py - 内置logits处理器（MinP、LogitBias、MinTokens）
3. state.py - LogitsProcessors容器和BatchUpdateBuilder

使用流程:
    1. 通过build_logitsprocs()构建LogitsProcessors实例
    2. 在每个解码步骤中，通过BatchUpdateBuilder跟踪batch状态变化
    3. 将BatchUpdate传递给每个LogitsProcessor的update_state()方法
    4. 在采样时调用LogitsProcessor的apply()方法处理logits

Logits处理器分类:
    - argmax不变的处理器: 不影响贪心采样结果（如min_p），仅在随机采样时应用
    - 非argmax不变的处理器: 可能影响贪心采样结果（如min_tokens、logit_bias），在所有采样前应用

插件系统:
    通过entry_points机制支持自定义logits处理器插件:
    - 入口点组: vllm.logits_processors
    - 支持完全限定类名(FQCN)加载: module_path:ClassName
"""

import importlib
import inspect
import itertools
from abc import abstractmethod
from collections.abc import Sequence
from functools import lru_cache, partial
from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.logits_process import LogitsProcessor as RequestLogitsProcessor
from vllm.sampling_params import SamplingParams
from vllm.utils.torch_utils import guard_cuda_initialization
from vllm.v1.sample.logits_processor.builtin import (
    LogitBiasLogitsProcessor,
    MinPLogitsProcessor,
    MinTokensLogitsProcessor,
    process_dict_updates,
)
from vllm.v1.sample.logits_processor.interface import (
    BatchUpdate,
    LogitsProcessor,
    MoveDirectionality,
)
from vllm.v1.sample.logits_processor.state import BatchUpdateBuilder, LogitsProcessors

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)

# 错误信息: 用户尝试使用池化模型和自定义logits处理器时
STR_POOLING_REJECTS_LOGITSPROCS = (
    "Pooling models do not support custom logits processors."
)

# 错误信息: 用户尝试在投机解码启用时使用自定义logits处理器时
STR_SPEC_DEC_REJECTS_LOGITSPROCS = (
    "Custom logits processors are not supported when speculative decoding is enabled."
)

# logits处理器插件的入口点组名
LOGITSPROCS_GROUP = "vllm.logits_processors"

# 内置logits处理器列表
# 注意: MinTokensLogitsProcessor必须在最后，因为它需要看到其他处理器的状态
BUILTIN_LOGITS_PROCESSORS: list[type[LogitsProcessor]] = [
    MinTokensLogitsProcessor,
    LogitBiasLogitsProcessor,
    MinPLogitsProcessor,
]


def _load_logitsprocs_plugins() -> list[type[LogitsProcessor]]:
    """加载所有已安装的logits处理器插件。

    通过Python的entry_points机制发现和加载插件。
    插件需要在setup.py/pyproject.toml中注册vllm.logits_processors入口点。

    返回:
        已加载的LogitsProcessor类列表
    """
    from importlib.metadata import entry_points

    installed_logitsprocs_plugins = entry_points(group=LOGITSPROCS_GROUP)
    if len(installed_logitsprocs_plugins) == 0:
        logger.debug("No logitsprocs plugins installed (group %s).", LOGITSPROCS_GROUP)
        return []

    # 加载logits处理器插件
    logger.debug("Loading installed logitsprocs plugins (group %s):", LOGITSPROCS_GROUP)
    classes: list[type[LogitsProcessor]] = []
    for entrypoint in installed_logitsprocs_plugins:
        try:
            logger.debug(
                "- Loading logitproc plugin entrypoint=%s target=%s",
                entrypoint.name,
                entrypoint.value,
            )
            with guard_cuda_initialization():
                classes.append(entrypoint.load())
        except Exception as e:
            logger.error("Failed to load LogitsProcessor plugin %s: %s", entrypoint, e)
            raise RuntimeError(
                f"Failed to load LogitsProcessor plugin {entrypoint}"
            ) from e
    return classes


def _load_logitsprocs_by_fqcns(
    logits_processors: Sequence[str | type[LogitsProcessor]] | None,
) -> list[type[LogitsProcessor]]:
    """通过完全限定类名(FQCN)加载logits处理器类型。

    将混合的logits处理器类型和FQCN字符串列表转换为纯类型列表。
    FQCN语法: <module>:<type>，例如 x.y.z:CustomLogitProc

    已加载的logits处理器类型必须是LogitsProcessor的子类。

    参数:
        logits_processors: 可能混合的logits处理器类型和FQCN字符串列表

    返回:
        logits处理器类型列表
    """
    if not logits_processors:
        return []

    logger.debug(
        "%s additional custom logits processors specified, checking whether "
        "they need to be loaded.",
        len(logits_processors),
    )

    classes: list[type[LogitsProcessor]] = []
    for ldx, logitproc in enumerate(logits_processors):
        if isinstance(logitproc, type):
            # 已经是类型，直接验证和添加
            logger.debug(" - Already-loaded logit processor: %s", logitproc.__name__)
            if not issubclass(logitproc, LogitsProcessor):
                raise ValueError(
                    f"{logitproc.__name__} is not a subclass of LogitsProcessor"
                )
            classes.append(logitproc)
            continue

        # 从FQCN字符串加载
        logger.debug("- Loading logits processor %s", logitproc)
        module_path, qualname = logitproc.split(":")

        try:
            # 加载模块
            with guard_cuda_initialization():
                module = importlib.import_module(module_path)
        except Exception as e:
            logger.error(
                "Failed to load %sth LogitsProcessor plugin %s: %s",
                ldx,
                logitproc,
                e,
            )
            raise RuntimeError(
                f"Failed to load {ldx}th LogitsProcessor plugin {logitproc}"
            ) from e

        # 沿着点分名称遍历获取logits处理器类
        obj = module
        for attr in qualname.split("."):
            obj = getattr(obj, attr)
        if not isinstance(obj, type):
            raise ValueError("Loaded logit processor must be a type.")
        if not issubclass(obj, LogitsProcessor):
            raise ValueError(f"{obj.__name__} must be a subclass of LogitsProcessor")
        classes.append(obj)

    return classes


def _load_custom_logitsprocs(
    logits_processors: Sequence[str | type[LogitsProcessor]] | None,
) -> list[type[LogitsProcessor]]:
    """加载所有自定义logits处理器。

    加载顺序:
    1. 首先加载所有已安装的logits处理器插件
    2. 然后加载用户在初始化时传递的自定义logits处理器

    参数:
        logits_processors: 可能混合的logits处理器类型和FQCN字符串列表

    返回:
        所有已加载的logits处理器类型列表
    """
    from vllm.platforms import current_platform

    if current_platform.is_tpu():
        # TODO(andy) - vLLM V1 on TPU does not support custom logitsprocs
        return []

    return _load_logitsprocs_plugins() + _load_logitsprocs_by_fqcns(logits_processors)


def build_logitsprocs(
    vllm_config: "VllmConfig",
    device: torch.device,
    is_pin_memory: bool,
    is_pooling_model: bool,
    custom_logitsprocs: Sequence[str | type[LogitsProcessor]] = (),
) -> LogitsProcessors:
    """构建LogitsProcessors实例。

    根据配置和模型类型，构建并初始化所有logits处理器。

    参数:
        vllm_config: vLLM配置
        device: 计算设备
        is_pin_memory: 是否使用pin memory
        is_pooling_model: 是否是池化模型
        custom_logitsprocs: 自定义logits处理器列表

    返回:
        初始化后的LogitsProcessors实例

    异常:
        ValueError: 池化模型使用自定义处理器，或投机解码时使用自定义处理器
    """
    if is_pooling_model:
        if custom_logitsprocs:
            raise ValueError(STR_POOLING_REJECTS_LOGITSPROCS)
        logger.debug(
            "Skipping logits processor loading because pooling models"
            " do not support logits processors."
        )
        return LogitsProcessors()

    # 检查是否启用了投机解码
    if vllm_config.speculative_config:
        if custom_logitsprocs:
            raise ValueError(STR_SPEC_DEC_REJECTS_LOGITSPROCS)
        logger.warning(
            "min_p and logit_bias parameters won't work with speculative decoding."
        )
        return LogitsProcessors(
            [MinTokensLogitsProcessor(vllm_config, device, is_pin_memory)]
        )

    # 加载自定义logits处理器
    custom_logitsprocs_classes = _load_custom_logitsprocs(custom_logitsprocs)
    # 构建完整的处理器列表（内置 + 自定义）
    return LogitsProcessors(
        ctor(vllm_config, device, is_pin_memory)
        for ctor in itertools.chain(
            BUILTIN_LOGITS_PROCESSORS, custom_logitsprocs_classes
        )
    )


# 缓存的自定义logits处理器加载函数
cached_load_custom_logitsprocs = lru_cache(_load_custom_logitsprocs)


def validate_logits_processors_parameters(
    logits_processors: Sequence[str | type[LogitsProcessor]] | None,
    sampling_params: SamplingParams,
):
    """验证logits处理器参数。

    加载所有自定义logits处理器，并对每个处理器调用validate_params方法。

    参数:
        logits_processors: logits处理器列表（类型或FQCN字符串）
        sampling_params: 采样参数
    """
    logits_processors = (
        tuple(logits_processors) if logits_processors is not None else None
    )
    for logits_procs in cached_load_custom_logitsprocs(logits_processors):
        logits_procs.validate_params(sampling_params)


class AdapterLogitsProcessor(LogitsProcessor):
    """每请求logits处理器的适配器包装器。

    要包装特定的每请求logits处理器:
    1. 继承 `AdapterLogitsProcessor`
    2. 实现 `self.is_argmax_invariant()` 基类方法
    3. 实现 `self.new_req_logits_processor(params)` 方法

    通常不需要重写 `self.__init__(vllm_config, device, is_pin_memory)`。
    但是，要实现自定义构造函数行为 - 特别是任何操作或存储
    `vllm_config`、`device` 或 `is_pin_memory` 的逻辑 -
    必须重写 `self.__init__(vllm_config, device, is_pin_memory)`
    并在重写中调用 `super().__init__(vllm_config, device, is_pin_memory)`

    工作原理:
    - 维护一个请求索引到logits处理器状态的映射 (req_info)
    - 状态表示为partial[Tensor]，包含预填充了output token ids和
      prompt token ids（如果需要）的请求级logits处理器
    - partial携带对output token ids的*引用*，因此始终操作当前最新的列表

    属性:
        req_info: 请求索引到logits处理器partial的映射
    """

    def __init__(
        self, vllm_config: "VllmConfig", device: torch.device, is_pin_memory: bool
    ):
        """子类必须调用 `super().__init__(vllm_config, device, is_pin_memory)`。

        子类构造函数可能会发现使用 `vllm_config`、`device` 和 `is_pin_memory`
        参数很有用。但是无论是否使用这些参数，vLLM logits处理器接口要求所有三个参数都存在。
        """
        # 请求索引 -> logits处理器状态的映射
        #
        # 状态表示为partial[Tensor]，包含:
        # - 预填充了output token ids参数的请求级logits处理器
        # - 如果需要，还预填充了prompt token ids参数
        #
        # 注意partial携带对output token ids的*引用*，
        # 因此始终操作当前最新的列表，而不是创建partial时的列表。
        self.req_info: dict[int, partial[torch.Tensor]] = {}

    @abstractmethod
    def new_req_logits_processor(
        self,
        params: SamplingParams,
    ) -> RequestLogitsProcessor | None:
        """消费请求信息; 返回每请求logits处理器。

        如果logits处理器不需要应用于请求，返回None。

        参数:
            params: 请求采样参数

        返回:
            None（如果logits处理器不应应用于请求）; 否则返回
            RequestLogitsProcessor实例
        """
        raise NotImplementedError

    def _new_state(
        self,
        params: SamplingParams,
        prompt_ids: list[int] | None,
        output_ids: list[int],
    ) -> partial[torch.Tensor] | None:
        """为新请求返回状态表示。

        如果logits处理器不适用于该请求，返回None。

        参数:
            params: 请求采样参数
            prompt_ids: 请求的prompt token IDs
            output_ids: 该请求到目前为止的解码token

        返回:
            logits处理器的partial[Tensor]，或None
        """
        if req_lp := self.new_req_logits_processor(params):
            if len(inspect.signature(req_lp).parameters) == 3:
                if prompt_ids is None:
                    raise ValueError(
                        "Prompt token ids are required for this "
                        "logits processor but were not provided."
                    )
                args = [prompt_ids, output_ids]
            else:
                args = [output_ids]
            return partial(req_lp, *args)
        return None

    def update_state(self, batch_update: BatchUpdate | None):
        """更新batch状态。

        使用process_dict_updates工具函数处理请求的添加、删除和移动。

        参数:
            batch_update: batch状态更新信息
        """
        process_dict_updates(
            self.req_info,
            batch_update,
            self._new_state,
        )

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        """将每请求logits处理器应用到logits张量的对应行。

        参数:
            logits: 输入logits张量 [batch_size, vocab_size]

        返回:
            处理后的logits张量
        """
        if self.req_info:
            # 将每请求logits处理器应用到logits张量的对应行
            for req_idx, req_lp in self.req_info.items():
                req_logits = logits[req_idx]
                new_logits = req_lp(req_logits)
                if new_logits is not req_logits:
                    # 如果需要，原地修改logits张量行
                    logits[req_idx] = new_logits
        return logits


__all__ = [
    "LogitsProcessor",
    "LogitBiasLogitsProcessor",
    "MinPLogitsProcessor",
    "MinTokensLogitsProcessor",
    "BatchUpdate",
    "BatchUpdateBuilder",
    "MoveDirectionality",
    "LogitsProcessors",
    "build_logitsprocs",
    "STR_POOLING_REJECTS_LOGITSPROCS",
    "LOGITSPROCS_GROUP",
    "AdapterLogitsProcessor",
]
