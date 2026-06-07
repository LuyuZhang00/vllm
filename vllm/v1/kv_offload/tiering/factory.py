# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# =============================================================================
# 二级存储层工厂（Secondary Tier Factory）
#
# 本模块实现了二级存储层的注册与创建机制，采用工厂模式 + 延迟加载策略。
#
# 在 vLLM v1 的多级 KV cache 卸载架构中，存储分为三层：
#   GPU 层 -> CPU 主层级（Primary Tier） -> 二级层（Secondary Tier）
#
# 其中二级层是最底下的存储层，典型的实现包括：
#   - 文件系统（fs）：将 KV cache 持久化到本地磁盘。
#   - 网络存储 / 远程存储：将 KV cache 存储到远程服务器。
#   - NVMe SSD：利用高速 SSD 作为二级缓存。
#   - example：示例实现，用于测试和演示。
#
# 本文件的角色是"工厂"——它不关心二级层的具体实现细节，
# 只负责根据用户配置（tier_config）找到正确的二级层类并实例化。
#
# 设计目的：
#   1. 解耦：二级层的具体实现（如文件系统、网络存储）与核心框架解耦。
#      框架只依赖 SecondaryTierManager 抽象接口（定义在 base.py 中），
#      具体实现通过注册机制接入。
#   2. 延迟加载：通过 importlib 在首次创建时才加载模块，避免启动时导入
#      所有可能的二级层实现（有些可能依赖未安装的包，如 fsspec、boto3 等）。
#   3. 可扩展：第三方可以通过 register_tier() 注册自定义二级层实现，
#      无需修改框架代码。
#
# 已注册的二级层类型：
#   - "example"：示例实现，用于测试和演示。
#   - "fs"：文件系统实现，将 KV cache 持久化到本地磁盘。
#
# 使用流程（按调用顺序）：
#   1. Python 模块加载时（import factory），模块底部的 register_tier()
#      调用将已知的二级层类型注册到工厂的 _registry 字典中。
#   2. 用户在 vLLM 配置中指定二级层（通过 kv_connector_extra_config），
#      例如 {"secondary_tiers": [{"type": "fs", "base_dir": "/tmp/kv"}]}。
#   3. TieringOffloadingSpec.get_manager() 调用 create_secondary_tier()，
#      根据 "type" 字段查找注册表，延迟加载对应模块并实例化二级层管理器。
#   4. 实例化后的 SecondaryTierManager 被注入到 TieringOffloadingManager 中，
#      后续由 Scheduler 调用其 lookup/submit_load/submit_store 等方法。
#
# 关键类和方法：
#   - SecondaryTierFactory：工厂类，维护注册表和创建方法。
#     - register_tier()：注册一个二级层类型（只记录路径，不导入模块）。
#     - create_secondary_tier()：根据配置创建二级层实例。
# =============================================================================

import importlib
from collections.abc import Callable
from typing import TYPE_CHECKING

from vllm.v1.kv_offload.tiering.base import SecondaryTierManager

# 中文注释：OffloadingSpec 是卸载配置的抽象基类，定义在 vllm.v1.kv_offload.base 中。
# 仅在类型检查时导入，避免循环依赖（因为 OffloadingSpec 所在模块可能间接引用本模块）。
if TYPE_CHECKING:
    from vllm.v1.kv_offload.base import OffloadingSpec


class SecondaryTierFactory:
    """二级存储层工厂，负责注册和创建二级层实例。

    中文注释：SecondaryTierFactory 是二级层的"注册中心"，采用经典工厂模式。
    它维护一个注册表（_registry），将二级层类型的字符串标识映射到延迟加载函数。
    当用户配置指定某种二级层类型时，工厂负责：
      1. 查找注册表，确认该类型已注册。
      2. 调用延迟加载器，首次导入实现模块并获取类对象。
      3. 用用户配置参数实例化对应的 SecondaryTierManager 子类。

    这种设计使得二级层的实现可以完全独立于框架，
    第三方只需调用 register_tier() 即可接入自定义实现。
    """

    # 中文注释：注册表，存储 tier_type -> 延迟加载函数 的映射。
    # 值是无参函数（loader），调用时才真正 import 模块并返回类对象。
    # 这种延迟加载方式避免了在 import 本模块时就加载所有二级层的依赖。
    #
    # 例如，注册 "fs" 类型后，_registry["fs"] 是一个函数，
    # 调用 _registry["fs"]() 才会真正 import vllm.v1.kv_offload.tiering.fs.manager
    # 并返回 FileSystemTierManager 类。
    _registry: dict[str, Callable[[], type[SecondaryTierManager]]] = {}

    @classmethod
    def register_tier(cls, tier_type: str, module_path: str, class_name: str) -> None:
        """注册一个新的二级层类型到工厂注册表。

        中文注释：此方法在模块加载时被调用，将二级层类型信息记录到注册表中。
        注册时只记录模块路径和类名，不立即导入模块（延迟加载策略）。
        当用户配置中指定该类型时，才通过 loader() 函数导入模块并获取类。

        注册流程：
          1. 检查 tier_type 是否已注册（不允许重复注册）。
          2. 创建一个闭包 loader()，它封装了 importlib.import_module() 调用。
             闭包捕获了 module_path 和 class_name，后续调用时才执行导入。
          3. 将 loader 存入 _registry[tier_type]。

        Args:
            tier_type: 二级层类型的标识字符串（如 "fs"、"example"）。
                此字符串与用户配置中的 "type" 字段对应。
            module_path: 实现该类型的 Python 模块的完整路径
                （如 "vllm.v1.kv_offload.tiering.fs.manager"）。
            class_name: 模块中 SecondaryTierManager 子类的类名
                （如 "FileSystemTierManager"）。
        """
        # 中文注释：注册一个新的二级层类型。
        #
        # 注册时只记录模块路径和类名，不立即导入模块（延迟加载）。
        # 当用户配置中指定该类型时，才通过 loader() 导入模块并获取类。
        #
        # Args:
        #   tier_type: 二级层类型的标识字符串（如 "fs"、"example"）。
        #   module_path: 实现该类型的 Python 模块路径（如 "vllm.v1.kv_offload.tiering.fs.manager"）。
        #   class_name: 模块中 SecondaryTierManager 子类的类名。
        if tier_type in cls._registry:
            raise ValueError(f"Tier '{tier_type}' is already registered.")

        def loader() -> type[SecondaryTierManager]:
            # 中文注释：延迟加载器——首次调用时才导入模块并获取类对象。
            # importlib.import_module() 会触发模块的完整导入过程，
            # 包括执行模块顶层代码。getattr() 从模块中按名称获取类。
            module = importlib.import_module(module_path)
            return getattr(module, class_name)

        cls._registry[tier_type] = loader

    @classmethod
    def create_secondary_tier(
        cls,
        tier_config: dict,
        primary_kv_view: memoryview,
        offloading_spec: "OffloadingSpec",
    ) -> SecondaryTierManager:
        """根据用户配置创建一个二级层实例。

        中文注释：这是工厂的核心创建方法，由 TieringOffloadingSpec.get_manager()
        在初始化多层卸载管理器时调用。

        创建流程（按顺序）：
          1. 复制配置字典，避免修改原始数据。
          2. 从配置中弹出 "type" 字段，确定二级层类型（如 "fs"、"example"）。
          3. 在注册表中查找该类型，确认已注册且有对应的延迟加载器。
          4. 调用延迟加载器（首次调用时会触发模块导入），获取二级层类。
          5. 用 offloading_spec、primary_kv_view 和剩余的配置参数
             实例化二级层管理器。

        调用链路：
          用户配置 -> TieringOffloadingSpec.get_manager()
                   -> SecondaryTierFactory.create_secondary_tier()
                   -> 延迟加载模块 -> 实例化 SecondaryTierManager 子类
                   -> 返回给 TieringOffloadingManager 持有

        Args:
            tier_config: 用户配置字典，必须包含 "type" 字段，
                其余字段作为构造函数的额外关键字参数传递。
                例如 {"type": "fs", "base_dir": "/tmp/kv_cache"}。
            primary_kv_view: 一级层（CPU Primary Tier）KV cache 的 memoryview。
                二级层通过此视图直接读写一级层的 block 数据，
                这是二级层与一级层交换数据的唯一通道。
            offloading_spec: 卸载配置对象，包含 block 大小、hash 规则等
                全局参数。所有二级层共享同一份 offloading_spec。

        Returns:
            创建好的 SecondaryTierManager 实例，可直接用于
            lookup/submit_load/submit_store 等操作。

        Raises:
            ValueError: 如果配置中缺少 "type" 字段，
                或指定的类型未在注册表中注册。
        """
        # 中文注释：根据用户配置创建一个二级层实例。
        #
        # 创建流程：
        #   1. 从 tier_config 中提取 "type" 字段，确定二级层类型。
        #   2. 查找注册表，获取对应的延迟加载器。
        #   3. 加载器调用后获得二级层类。
        #   4. 用剩余的配置参数实例化二级层管理器。
        #
        # Args:
        #   tier_config: 用户配置字典，必须包含 "type" 字段，
        #     其余字段作为构造函数的额外参数传递。
        #   primary_kv_view: 一级层 CPU KV cache 的 memoryview，
        #     二级层通过此视图与一级层交换数据。
        #   offloading_spec: 卸载配置，包含 block 大小、hash 规则等。
        #
        # Returns:
        #   创建好的 SecondaryTierManager 实例。
        config = tier_config.copy()

        tier_type = config.pop("type", None)
        if not tier_type:
            raise ValueError("Secondary tier configuration must include 'type'")

        if tier_type not in cls._registry:
            raise ValueError(
                f"Unknown secondary tier type: {tier_type!r}. "
                f"Supported types: {list(cls._registry)}"
            )

        # 中文注释：调用延迟加载器获取二级层类，然后用配置参数实例化。
        # cls._registry[tier_type]() 触发 importlib.import_module()，
        # 首次调用后 Python 会缓存已导入的模块，后续调用不会重复导入。
        tier_cls = cls._registry[tier_type]()
        return tier_cls(
            offloading_spec=offloading_spec,
            primary_kv_view=primary_kv_view,
            tier_type=tier_type,
            **config,
        )


# =============================================================================
# 模块级注册：在工厂模块被导入时自动注册已知的二级层类型。
#
# 中文注释：以下代码在 Python import 本模块时自动执行。
# 这是"自注册"模式——每个已知的二级层类型在模块加载时就注册到工厂中，
# 无需外部代码显式调用 register_tier()。
#
# 注意：这里只是注册（记录路径和类名），不会实际导入二级层的实现模块。
# 实际导入发生在 create_secondary_tier() 被调用时（延迟加载）。
# 这样做的好处是：
#   1. 用户只需要安装自己需要的二级层的依赖（如 fsspec），
#      不需要安装所有二级层的依赖。
#   2. 启动速度更快，不会因为导入大量模块而变慢。
# =============================================================================

# 中文注释：注册示例二级层（用于测试和演示）。
# ExampleSecondaryTierManager 是一个简单的内存实现，不依赖外部存储。
# 通常在单元测试中使用，验证多层卸载的整体流程。
SecondaryTierFactory.register_tier(
    "example",
    "vllm.v1.kv_offload.tiering.example.manager",
    "ExampleSecondaryTierManager",
)

# 中文注释：注册文件系统二级层（将 KV cache 持久化到本地磁盘）。
# FileSystemTierManager 使用本地文件系统存储 KV cache block，
# 适用于单机场景下的大容量 KV cache 持久化。
SecondaryTierFactory.register_tier(
    "fs",
    "vllm.v1.kv_offload.tiering.fs.manager",
    "FileSystemTierManager",
)
