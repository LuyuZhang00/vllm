# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# 中文注释：本文件是 vLLM V1 KV cache offload 的工厂模块。
# ================================================================
# 核心职责：
#   1. 提供 OffloadingSpecFactory 工厂类，用于创建 KV cache 卸载（offload）的规格对象。
#   2. 通过"注册表 + 懒加载"模式，将 spec 名称映射到对应的类，避免在模块导入时加载所有实现。
#   3. 支持两种 spec 来源：
#      - 已注册的内置 spec（如 CPUOffloadingSpec、TieringOffloadingSpec）
#      - 用户通过 extra_config 指定的自定义 spec（通过 spec_module_path 动态加载）
#
# 设计模式说明：
#   - 工厂模式 (Factory Pattern)：统一创建 OffloadingSpec 实例的入口。
#   - 懒加载 (Lazy Loading)：register_spec 不直接导入模块，而是注册一个 loader 函数，
#     只有在实际需要创建 spec 时才 import 模块，减少启动时的导入开销。
#   - 注册表模式 (Registry Pattern)：用 _registry 字典维护 name -> loader 的映射关系。
#
# 典型调用链路：
#   用户请求 -> VllmConfig.kv_transfer_config -> OffloadingSpecFactory.create_spec()
#   -> 根据 spec_name 从注册表或用户指定模块加载类 -> 实例化 OffloadingSpec
# ================================================================

import importlib
from collections.abc import Callable
from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import OffloadingSpec

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig

logger = init_logger(__name__)


class OffloadingSpecFactory:
    """
    中文注释：KV cache offload spec 的工厂类。
    ================================================================
    职责：
      1. 维护一个 spec 注册表 (_registry)，将 spec 名称映射到懒加载函数。
      2. 提供 register_spec() 用于注册新的 spec 实现。
      3. 提供 create_spec() 用于根据配置创建 OffloadingSpec 实例。

    注册表的工作原理：
      - _registry 的值不是类本身，而是一个无参的 Callable（loader 函数）。
      - 当调用 loader() 时，才会真正 import 模块并获取类对象。
      - 这样做可以避免在模块顶部 import 所有 spec 实现，加快启动速度。
    ================================================================
    """

    # 中文注释：spec 注册表，key 是 spec 名称（如 "CPUOffloadingSpec"），
    # value 是一个无参函数，调用时才真正 import 模块并返回类对象。
    _registry: dict[str, Callable[[], type[OffloadingSpec]]] = {}

    @classmethod
    def register_spec(cls, name: str, module_path: str, class_name: str) -> None:
        """Register a spec with a lazy-loading module and class name."""
        # 中文注释：注册一个 offload spec 到工厂的注册表中。
        # ================================================================
        # 参数说明：
        #   - name: spec 的唯一名称，作为注册表的 key。
        #   - module_path: Python 模块路径，如 "vllm.v1.kv_offload.cpu.spec"。
        #   - class_name: 模块中的类名，如 "CPUOffloadingSpec"。
        #
        # 执行流程：
        #   1. 检查 name 是否已注册，已注册则抛出异常防止重复注册。
        #   2. 定义一个 loader 闭包函数，该函数在被调用时：
        #      a. 通过 importlib.import_module() 动态导入 module_path 对应的模块。
        #      b. 通过 getattr() 从模块中获取 class_name 对应的类。
        #      c. 返回该类。
        #   3. 将 loader 函数注册到 _registry[name] 中。
        #
        # 为什么用闭包而非直接存类？
        #   因为我们希望延迟导入（lazy loading），避免在注册时就 import 整个模块。
        #   这对于可选依赖（如 CPU offload 或 tiering 相关的库）尤其重要，
        #   可以在未安装对应依赖时也不会导致导入失败。
        # ================================================================
        if name in cls._registry:
            raise ValueError(f"Connector '{name}' is already registered.")

        def loader() -> type[OffloadingSpec]:
            module = importlib.import_module(module_path)
            return getattr(module, class_name)

        cls._registry[name] = loader

    @classmethod
    def create_spec(
        cls,
        config: "VllmConfig",
        kv_cache_config: "KVCacheConfig",
    ) -> OffloadingSpec:
        # 中文注释：根据运行时配置创建并返回一个 OffloadingSpec 实例。
        # ================================================================
        # 执行流程（带序号）：
        #   (1) 从 VllmConfig 中获取 kv_transfer_config，该配置包含 KV 传输/卸载的参数。
        #   (2) 从 kv_transfer_config 的 extra_config 中读取 "spec_name"，
        #       默认值为 "CPUOffloadingSpec"（即 CPU 卸载是最常用的场景）。
        #   (3) 如果 spec_name 已在注册表中：
        #       a. 调用注册表中的 loader 函数，懒加载获取 spec 类。
        #   (4) 如果 spec_name 不在注册表中：
        #       a. 从 extra_config 中读取 "spec_module_path"（用户自定义模块路径）。
        #       b. 如果没有指定 module_path，则抛出 ValueError。
        #       c. 通过 importlib 动态导入模块，并通过 getattr 获取类。
        #   (5) 验证获取到的类确实是 OffloadingSpec 的子类。
        #   (6) 实例化 spec 类，传入 config 和 kv_cache_config，返回实例。
        #
        # 参数说明：
        #   - config: vLLM 的全局配置对象，包含模型、KV transfer 等配置。
        #   - kv_cache_config: KV cache 的配置对象，描述 KV cache 的结构和布局。
        #
        # 返回值：
        #   - OffloadingSpec 实例，描述了 KV cache 如何被卸载到外部存储（如 CPU 内存、
        #     远程存储等）。该实例随后会被 KV cache 管理系统用来决定哪些 block 需要
        #     被卸载、如何组织卸载数据等。
        # ================================================================
        kv_transfer_config = config.kv_transfer_config
        assert kv_transfer_config is not None
        extra_config = kv_transfer_config.kv_connector_extra_config
        spec_name = extra_config.get("spec_name", "CPUOffloadingSpec")
        if spec_name in cls._registry:
            spec_cls = cls._registry[spec_name]()
        else:
            spec_module_path = extra_config.get("spec_module_path")
            if spec_module_path is None:
                raise ValueError(f"Unsupported spec type: {spec_name}")
            spec_module = importlib.import_module(spec_module_path)
            spec_cls = getattr(spec_module, spec_name)
        assert issubclass(spec_cls, OffloadingSpec)
        logger.info("Creating offloading spec with name: %s", spec_name)
        return spec_cls(config, kv_cache_config)


# 中文注释：在此处注册内置的 offload spec 实现。
# ================================================================
# 已注册的 spec：
#   1. "CPUOffloadingSpec" —— 将 KV cache 从 GPU 卸载到 CPU 内存。
#      模块路径：vllm.v1.kv_offload.cpu.spec
#      适用场景：GPU 显存不足时，将不活跃的 KV cache block 卸载到 CPU 内存，
#      需要时再加载回 GPU。这是最常见的 offload 场景。
#
#   2. "TieringOffloadingSpec" —— 多层级存储的 KV cache 卸载。
#      模块路径：vllm.v1.kv_offload.tiering.spec
#      适用场景：支持多级存储（如 GPU -> CPU -> 远程存储）的分层卸载策略，
#      适用于大规模推理场景中需要更精细的存储管理。
#
# 注意：这里使用 register_spec 而非直接导入，是因为这些 spec 实现可能依赖
# 可选的第三方库。使用懒加载可以确保只有在实际需要时才尝试导入，
# 避免在不需要这些功能时因缺少依赖而启动失败。
# ================================================================
OffloadingSpecFactory.register_spec(
    "CPUOffloadingSpec", "vllm.v1.kv_offload.cpu.spec", "CPUOffloadingSpec"
)
OffloadingSpecFactory.register_spec(
    "TieringOffloadingSpec",
    "vllm.v1.kv_offload.tiering.spec",
    "TieringOffloadingSpec",
)
