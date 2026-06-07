# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# =============================================================================
# custom_class_proposer.py - 自定义推测解码 Proposer 的动态加载器
#
# 【模块功能概述】
# 本模块提供了一个工厂函数，用于动态加载和实例化用户自定义的推测解码 Proposer。
# 用户可以通过 speculative_config.model 字段指定一个完整的 Python 类路径
# （如 "my_module.MyCustomProposer"），本模块会：
#   1. 解析模块路径和类名
#   2. 动态导入模块
#   3. 实例化用户自定义的 Proposer 类
#   4. 校验其实例必须包含 callable 的 propose() 方法
#
# 【使用场景】
# 当内置的推测解码策略（EAGLE、Draft Model、MTP 等）不能满足需求时，
# 用户可以实现自定义的 Proposer 类，通过配置项指定类路径即可加载。
# 自定义 Proposer 必须：
#   - 接受 VllmConfig 作为构造函数参数
#   - 实现 propose() 方法（用于生成候选 token）
#
# 【在推测解码链路中的位置】
#   用户配置 speculative_config.model = "my_module.MyProposer"
#   -> create_custom_proposer() 动态加载并实例化
#   -> 返回的实例替代内置 Proposer，由 Engine/Runner 调用其 propose() 方法
# =============================================================================

import importlib

from vllm.config import VllmConfig
from vllm.logger import init_logger

logger = init_logger(__name__)


def create_custom_proposer(vllm_config: VllmConfig):
    """Load and instantiate a user-provided proposer class.

    The class path is read from ``speculative_config.model``
    (e.g., ``"my_module.MyCustomProposer"``).  The class is
    imported, instantiated with *vllm_config*, and returned
    directly so the caller can use it without any wrapper.

    The returned object must expose a callable ``propose`` method.

    # 中文注释：动态加载并实例化用户自定义的推测解码 Proposer 类。
    #
    # 【整体流程】
    # 1. 从 speculative_config.model 读取用户指定的完整类路径
    #    （如 "my_module.MyCustomProposer"）
    # 2. 解析出模块路径和类名
    # 3. 动态导入模块并获取类对象
    # 4. 使用 VllmConfig 实例化该类
    # 5. 校验实例必须包含 callable 的 propose() 方法
    # 6. 返回实例，调用方直接使用其 propose() 方法生成候选 token
    #
    # 【参数】
    # - vllm_config: 全局配置对象，其中 speculative_config.model 指定类路径
    #
    # 【返回值】
    # 用户自定义 Proposer 的实例，必须暴露 callable 的 propose 方法。
    #
    # 【异常说明】
    # - ValueError: 类路径不包含 "."（不是完整的模块路径）
    # - ImportError: 模块无法导入
    # - AttributeError: 模块中找不到指定的类，或类缺少 propose 方法
    # - RuntimeError: 类实例化失败（构造函数签名不匹配）
    """

    # ---- 步骤 1：校验配置 ----
    assert vllm_config.speculative_config is not None
    spec_config = vllm_config.speculative_config

    # 从配置中读取用户指定的类路径，如 "my_module.sub_module.MyProposer"
    backend = spec_config.model
    assert backend is not None

    # ---- 步骤 2：解析模块路径和类名 ----
    # 类路径必须包含 "."，形如 "module.ClassName" 或 "package.module.ClassName"
    if "." not in backend:
        raise ValueError(
            f"Invalid custom proposer module path '{backend}'. "
            "It must be a full module path (e.g., 'module.MyProposerClass')."
        )

    # 从右侧分割，确保多级包路径（如 a.b.c.MyClass）正确解析
    # rsplit(".", 1) -> ("a.b.c", "MyClass")
    module_path, class_name = backend.rsplit(".", 1)

    # ---- 步骤 3：动态导入模块 ----
    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise ImportError(
            f"Cannot import module '{module_path}' for custom proposer '{backend}': {e}"
        ) from e

    # ---- 步骤 4：获取类对象并校验存在性 ----
    user_class = getattr(module, class_name, None)
    if user_class is None:
        raise AttributeError(
            f"Module '{module_path}' has no attribute '{class_name}' "
            f"(speculative_config.model='{backend}')"
        )

    # ---- 步骤 5：实例化用户自定义类 ----
    # 用户类的构造函数必须接受 VllmConfig 作为参数
    try:
        instance = user_class(vllm_config)
    except Exception as e:
        raise RuntimeError(
            f"Failed to instantiate custom proposer class '{backend}': {e}. "
            "The class constructor must accept VllmConfig as argument."
        ) from e

    # ---- 步骤 6：校验实例必须有 callable 的 propose 方法 ----
    # propose() 是 Proposer 的核心接口，由 Engine/Runner 调用以生成候选 token
    if not hasattr(instance, "propose"):
        raise AttributeError(
            f"Custom proposer class '{backend}' must have a 'propose' method."
        )
    if not callable(instance.propose):
        raise AttributeError(
            f"Custom proposer class '{backend}' has a 'propose' attribute "
            "but it is not callable."
        )

    # ---- 步骤 7：记录日志并返回实例 ----
    logger.info(
        "Loaded custom proposer class '%s' with num_speculative_tokens=%d",
        backend,
        spec_config.num_speculative_tokens,
    )

    return instance
