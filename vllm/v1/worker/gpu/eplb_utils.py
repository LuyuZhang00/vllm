# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
专家负载均衡（Expert Parallel Load Balancing, EPLB）工具模块。

本模块实现了 MoE（Mixture of Experts）模型的专家负载均衡功能。
在 MoE 模型中，不同专家的负载可能不均衡，EPLB 通过动态调整
专家到 GPU 的映射来优化负载分布。

核心概念：
1. EPLB（Expert Parallel Load Balancing）：专家并行负载均衡
   - 监控每个专家的负载
   - 动态重映射专家到 GPU 的分配
   - 在推理过程中自动执行，无需中断

2. 弹性 EP（Elastic Expert Parallelism）：
   - 支持在运行时调整专家并行的规模
   - 当前不支持与草稿模型同时使用

3. 装饰器模式：
   - step_eplb_after 装饰器在模型运行器方法完成后自动触发 EPLB 步进

主要组件：
1. EPLBController - EPLB 控制器，管理 EPLB 状态和生命周期
2. step_eplb_after - 装饰器，自动触发 EPLB 步进
"""

from collections.abc import Callable
from functools import wraps
from typing import Any

import torch
import torch.nn as nn

from vllm.distributed.eplb.eplb_state import EplbState
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import (
    SupportsMultiModal,
    is_mixture_of_experts,
)

logger = init_logger(__name__)


def _unwrap_moe(model: nn.Module) -> nn.Module:
    """从 VLM 包装器中提取 MoE 语言模型。

    VLM 包装器（如 KimiK25ForConditionalGeneration）将 MoE 语言模型
    放在 `.language_model` 下，但自身不实现 MixtureOfExperts 接口。
    此函数镜像 V1 路径的行为（参见 vllm/v1/worker/gpu_model_runner.py,
    PR #39805）。

    参数:
        model: 模型实例

    返回:
        nn.Module: MoE 语言模型（如果是 VLM 且包含 MoE）
    """
    # VLM wrappers (e.g. KimiK25ForConditionalGeneration) hold the MoE
    # language model under `.language_model` but don't implement
    # MixtureOfExperts themselves. Mirror the V1 path
    # (see vllm/v1/worker/gpu_model_runner.py, PR #39805).
    if not is_mixture_of_experts(model) and isinstance(model, SupportsMultiModal):
        return model.get_language_model()
    return model


def step_eplb_after(*, is_dummy: bool = False) -> Callable:
    """装饰器：在模型运行器方法成功完成后触发 EPLB 步进。

    使用方法：
        @step_eplb_after()
        def execute_model(self, scheduler_output):
            ...

    参数:
        is_dummy: 是否为虚拟步进（用于预热或 profile）

    返回:
        装饰器函数
    """

    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(self: Any, *args, **kwargs) -> Any:
            result = fn(self, *args, **kwargs)
            if kwargs.get("skip_eplb", False):
                return result

            is_profile = kwargs.get("is_profile", False) if is_dummy else False
            self.eplb.step(is_dummy=is_dummy, is_profile=is_profile)
            return result

        return wrapper

    return decorator


class EPLBController:
    """EPLB 控制器，管理专家负载均衡的完整生命周期。

    生命周期：
    1. prepare_load() - 准备加载阶段，初始化 EPLB 状态
    2. maybe_register_model() - 注册主模型（如果是 MoE 模型）
    3. maybe_register_speculator() - 注册推测解码的草稿模型
    4. maybe_start_async_loop() - 启动异步 EPLB 循环
    5. step() - 每次推理迭代后调用，推进 EPLB 状态
    6. setup_from_mapping() - 从预定义映射初始化（用于恢复/迁移）

    属性:
        parallel_config: 并行配置
        device: 计算设备
        state: EPLB 状态（None 表示未启用）
        suppressed: 是否被抑制
        _has_registered_models: 是否已注册模型
    """

    def __init__(self, parallel_config: Any, device: torch.device):
        self.parallel_config = parallel_config
        self.device = device
        self.state: EplbState | None = None
        self.suppressed = False
        self._has_registered_models = False

    def prepare_load(self) -> None:
        """准备加载阶段，初始化 EPLB 状态。

        如果配置中启用了 EPLB，创建 EplbState 实例。
        """
        self.state = None
        self._has_registered_models = False
        if self.parallel_config.enable_eplb:
            self.state = EplbState(self.parallel_config, self.device)

    def maybe_register_speculator(
        self,
        speculator: Any | None,
        speculative_config: Any | None,
        load_dummy_weights: bool,
    ) -> bool:
        """尝试注册推测解码的草稿模型。

        仅当草稿模型是 MoE 模型时才注册。
        弹性 EP 不支持与草稿模型同时使用。

        参数:
            speculator: 推测解码器实例
            speculative_config: 推测解码配置
            load_dummy_weights: 是否加载虚拟权重

        返回:
            bool: 是否成功注册
        """
        # if speculator is a moe model, add it to eplb
        if (
            speculator is None
            or not hasattr(speculator, "model")
            or not self.parallel_config.enable_eplb
            or load_dummy_weights
        ):
            return False

        draft_model = speculator.model
        if not is_mixture_of_experts(draft_model):
            return False

        assert not self.parallel_config.enable_elastic_ep, (
            "Elastic EP is not supported with draft model."
        )
        assert speculative_config is not None
        assert speculative_config.draft_model_config is not None
        assert self.state is not None
        self.state.add_model(
            draft_model,
            speculative_config.draft_model_config,
        )
        self._has_registered_models = True
        return True

    def maybe_register_model(
        self,
        model: nn.Module,
        model_config: Any,
        load_dummy_weights: bool,
    ) -> bool:
        """尝试注册主模型。

        如果模型是 VLM 包装器，先提取其中的 MoE 语言模型。

        参数:
            model: 模型实例
            model_config: 模型配置
            load_dummy_weights: 是否加载虚拟权重

        返回:
            bool: 是否成功注册
        """
        if not self.parallel_config.enable_eplb or load_dummy_weights:
            return False

        model = _unwrap_moe(model)
        if not is_mixture_of_experts(model):
            return False

        logger.info_once("EPLB is enabled for model %s.", model_config.model)
        assert self.state is not None
        self.state.add_model(model, model_config)
        self._has_registered_models = True
        return True

    def maybe_start_async_loop(self, eplb_models_added: bool) -> None:
        """如果已注册模型且状态支持异步模式，启动异步 EPLB 循环。

        参数:
            eplb_models_added: 是否已添加 EPLB 模型
        """
        if eplb_models_added and self.state is not None and self.state.is_async:
            self.state.start_async_loop()

    def step(
        self,
        is_dummy: bool = False,
        is_profile: bool = False,
    ) -> None:
        """推进 EPLB 状态。

        在每次推理迭代后调用，触发负载均衡决策。

        参数:
            is_dummy: 是否为虚拟步进
            is_profile: 是否为 profile 步进
        """
        if (
            not self.parallel_config.enable_eplb
            or self.suppressed
            or self.state is None
            or not self._has_registered_models
        ):
            return

        self.state.step(
            is_dummy,
            is_profile,
            log_stats=self.parallel_config.eplb_config.log_balancedness,
        )

    def setup_from_mapping(
        self,
        model: nn.Module,
        model_config: Any,
        expanded_physical_to_logical: torch.Tensor,
        old_num_physical_experts: int,
    ) -> None:
        """从预定义的映射关系初始化 EPLB 状态。

        用于从之前的运行中恢复，或者在模型迁移场景下使用。

        参数:
            model: 模型实例
            model_config: 模型配置
            expanded_physical_to_logical: 物理专家到逻辑专家的映射
            old_num_physical_experts: 之前的物理专家数量
        """
        model = _unwrap_moe(model)
        assert is_mixture_of_experts(model)

        self.state = EplbState.from_mapping(
            model=model,
            model_config=model_config,
            device=self.device,
            parallel_config=self.parallel_config,
            expanded_physical_to_logical=expanded_physical_to_logical,
            num_valid_physical_experts=old_num_physical_experts,
        )
        self._has_registered_models = True
