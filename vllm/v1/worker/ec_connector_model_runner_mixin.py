# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Define EC connector functionality mixin for model runners.
"""

# =============================================================================
# 中文注释：EC (Expert Cache) Connector Model Runner Mixin 模块概述
# =============================================================================
# 本模块为 Model Runner（如 GPUModelRunner）提供 Expert Cache 连接器的功能混入。
#
# 背景与作用：
#   在 MoE（Mixture of Experts）模型的分布式推理中，不同 GPU 可能持有不同的
#   expert 权重。当一个请求需要访问的 expert 不在当前 GPU 上时，需要通过
#   EC Connector（Expert Cache 连接器）进行跨设备的 expert 权重或 KV cache 传输。
#   本 mixin 封装了与 EC Connector 交互的通用逻辑，使 Model Runner 无需关心
#   底层传输细节。
#
# 核心功能：
#   1. maybe_save_ec_to_connector: 将 encoder 计算结果保存到 EC connector，
#      供其他设备或后续请求复用（producer 角色）。
#   2. maybe_get_ec_connector_output: 在模型 forward 期间，通过上下文管理器
#      协调 EC connector 的加载（consumer 角色）和完成状态清理。
#   3. _get_ec_connector_output: 内部上下文管理器，封装 EC connector 的完整
#      生命周期：绑定元数据 -> 加载缓存 -> 执行 forward -> 收集完成状态 -> 清理。
#
# 设计模式：
#   采用 Mixin 模式，将 EC connector 相关逻辑从 GPUModelRunner 中解耦，
#   保持 Model Runner 的职责单一。通过 has_ec_transfer() 检查是否有可用的
#   EC connector，实现可选依赖，不影响没有部署 EC 的环境。
# =============================================================================

from collections.abc import Generator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from typing import TYPE_CHECKING

import torch

from vllm.distributed.ec_transfer import get_ec_transfer, has_ec_transfer
from vllm.distributed.ec_transfer.ec_connector.base import ECConnectorBase
from vllm.logger import init_logger
from vllm.v1.outputs import ECConnectorOutput

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

logger = init_logger(__name__)


# 中文注释：EC Connector Model Runner Mixin 类
# 该类以静态方法的形式提供 EC connector 相关功能，供 Model Runner 继承使用。
# 之所以采用静态方法，是因为这些操作不依赖 Model Runner 的实例状态，
# 仅需要 encoder_cache 和 scheduler_output 等参数。
class ECConnectorModelRunnerMixin:
    @staticmethod
    def maybe_save_ec_to_connector(
        encoder_cache: dict[str, torch.Tensor],
        mm_hash: str,
    ):
        """
        中文注释：将 encoder cache 保存到 EC connector（producer 角色）。

        流程说明：
          1. 检查当前环境是否配置了 EC transfer（通过 has_ec_transfer()）。
             如果未配置，直接返回，不影响正常推理流程。
          2. 获取全局 EC connector 实例。
          3. 调用 connector.save_caches() 将 encoder 计算产生的 cache
             保存起来。这些 cache 可以通过 mm_hash（multimodal hash）标识，
             供后续请求或其他设备复用，避免重复计算。

        参数说明：
          - encoder_cache: 编码器的中间计算结果缓存，通常是 vision encoder
            等模块的输出 KV cache，key 为标识字符串，value 为 tensor。
          - mm_hash: multimodal 输入的唯一哈希标识，用于在 EC connector 中
            作为缓存 key，实现缓存的精确匹配和复用。
        """
        if not has_ec_transfer():
            logger.debug("Not have ec transfer please check")
            return
        connector = get_ec_transfer()
        connector.save_caches(encoder_cache=encoder_cache, mm_hash=mm_hash)

    @staticmethod
    def maybe_get_ec_connector_output(
        scheduler_output: "SchedulerOutput",
        encoder_cache: dict[str, torch.Tensor],
        **kwargs,
    ) -> AbstractContextManager[ECConnectorOutput | None]:
        """
        中文注释：获取 EC connector 的输出上下文管理器（入口方法）。

        流程说明：
          1. 检查当前环境是否配置了 EC transfer。
          2. 如果已配置：返回一个真正的上下文管理器 _get_ec_connector_output，
             该管理器会在 enter 时初始化 EC connector 状态，在 exit 时收集
             完成状态并清理。
          3. 如果未配置：返回 nullcontext()，即一个什么都不做的空上下文管理器，
             这样调用方可以用统一的 with 语法使用，无需条件判断。

        设计意图：
          这是一种"可选依赖"的优雅实现方式。Model Runner 的 execute_model 方法
          中只需写：
              with self.maybe_get_ec_connector_output(...) as ec_output:
                  # forward pass
          无论有没有 EC connector，代码结构保持一致。
        """
        return (
            ECConnectorModelRunnerMixin._get_ec_connector_output(
                scheduler_output, encoder_cache, **kwargs
            )
            if has_ec_transfer()
            else nullcontext()
        )

    # This context manager must be used within an active forward context.
    # It encapsulates the entire EC connector lifecycle within execute_model
    @staticmethod
    @contextmanager
    def _get_ec_connector_output(
        scheduler_output: "SchedulerOutput",
        encoder_cache: dict[str, torch.Tensor],
        **kwargs,
    ) -> Generator[ECConnectorOutput, None, None]:
        output = ECConnectorOutput()

        ec_connector = get_ec_transfer()
        assert isinstance(ec_connector, ECConnectorBase)
        assert scheduler_output.ec_connector_metadata is not None
        ec_connector.bind_connector_metadata(scheduler_output.ec_connector_metadata)

        # Load caches for consumer or both roles
        if ec_connector.is_consumer:
            ec_connector.start_load_caches(encoder_cache, **kwargs)

        try:
            yield output
        finally:
            output.finished_sending, output.finished_recving = (
                ec_connector.get_finished(scheduler_output.finished_req_ids)
            )

            ec_connector.clear_connector_metadata()
