# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
KV 缓存连接器模块。

本模块实现了 vLLM v1 引擎中的 KV 缓存传输连接器功能。
KV 连接器用于在不同的推理节点之间传输 KV 缓存，支持以下场景：
1. 分布式推理中的 KV 缓存共享
2. 预填充-解码分离架构（PD 分离）
3. KV 缓存的跨节点迁移

核心概念：
1. KVTransfer：底层的 KV 缓存传输组，负责实际的数据传输
2. KVConnector：GPUModelRunner 使用的接口，封装了传输逻辑
3. ActiveKVConnector：活跃的连接器实现，执行实际的传输操作
4. NO_OP_KV_CONNECTOR：空操作连接器，当不需要传输时使用

工作流程：
1. pre_forward: 在模型前向传播前，处理预取和元数据绑定
2. 模型前向传播执行
3. post_forward: 在模型前向传播后，等待保存完成并收集结果
"""
from typing import TYPE_CHECKING

import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer import (
    get_kv_transfer_group,
    has_kv_transfer_group,
    kv_transfer_state,
)
from vllm.distributed.kv_transfer.kv_connector.utils import copy_kv_blocks
from vllm.forward_context import (
    get_forward_context,
    is_forward_context_available,
    set_forward_context,
)
from vllm.v1.outputs import (
    EMPTY_MODEL_RUNNER_OUTPUT,
    KVConnectorOutput,
    ModelRunnerOutput,
)

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput


class KVConnector:
    """KV 连接器接口，供 GPUModelRunner 使用。

    所有方法都有默认的空操作实现，子类可以覆盖以实现实际功能。
    """

    def pre_forward(self, scheduler_output: "SchedulerOutput") -> None:
        """在模型前向传播前调用。

        参数:
            scheduler_output: 调度器输出
        """
        pass

    def post_forward(
        self, finished_req_ids: set[str], wait_for_save: bool = True
    ) -> KVConnectorOutput | None:
        """在模型前向传播后调用。

        参数:
            finished_req_ids: 已完成的请求 ID 集合
            wait_for_save: 是否等待保存完成

        返回:
            KVConnectorOutput | None: KV 连接器输出
        """
        return None

    def no_forward(self, scheduler_output: "SchedulerOutput") -> ModelRunnerOutput:
        """当没有实际的前向传播时调用（仅有 KV 缓存操作）。

        参数:
            scheduler_output: 调度器输出

        返回:
            ModelRunnerOutput: 空的模型输出
        """
        return EMPTY_MODEL_RUNNER_OUTPUT

    def set_disabled(self, disabled: bool) -> None:
        """设置连接器的禁用状态。

        参数:
            disabled: 是否禁用
        """
        pass


class ActiveKVConnector(KVConnector):
    """活跃的 KV 连接器实现。

    该连接器执行实际的 KV 缓存传输操作，包括：
    - 注册 KV 缓存到传输组
    - 处理预取和抢占
    - 等待保存完成
    - 收集传输结果

    属性:
        vllm_config: vLLM 全局配置
        kv_connector: 底层的 KV 传输组实例
    """

    def __init__(
        self, vllm_config: VllmConfig, kv_caches_dict: dict[str, torch.Tensor]
    ):
        self.vllm_config = vllm_config
        self.kv_connector = get_kv_transfer_group()
        # 将 KV 缓存注册到 KV 连接器
        # TODO: support cross_layers_kv_cache
        # (see https://github.com/vllm-project/vllm/pull/27743)
        self.kv_connector.register_kv_caches(kv_caches_dict)
        self.kv_connector.set_host_xfer_buffer_ops(copy_kv_blocks)

        self._disabled = False

    def pre_forward(self, scheduler_output: "SchedulerOutput") -> None:
        """在模型前向传播前，处理 KV 缓存的预取。

        流程：
        1. 处理预取（preemption）
        2. 绑定连接器元数据
        3. 开始加载 KV 缓存

        参数:
            scheduler_output: 调度器输出
        """
        if self._disabled:
            return

        kv_connector_metadata = scheduler_output.kv_connector_metadata
        assert kv_connector_metadata is not None
        self.kv_connector.handle_preemptions(kv_connector_metadata)
        self.kv_connector.bind_connector_metadata(kv_connector_metadata)

        # TODO: sort out KV Connectors' use of forward_context
        if is_forward_context_available():
            self.kv_connector.start_load_kv(get_forward_context())
        else:
            with set_forward_context(None, self.vllm_config):
                self.kv_connector.start_load_kv(get_forward_context())

    def post_forward(
        self, finished_req_ids: set[str], wait_for_save: bool = True
    ) -> KVConnectorOutput | None:
        """在模型前向传播后，等待保存完成并收集结果。

        流程：
        1. 等待 KV 缓存保存完成
        2. 获取已完成的发送/接收请求
        3. 获取加载错误的块 ID
        4. 收集统计信息和事件
        5. 清除连接器元数据

        参数:
            finished_req_ids: 已完成的请求 ID 集合
            wait_for_save: 是否等待保存完成

        返回:
            KVConnectorOutput | None: KV 连接器输出
        """
        if self._disabled:
            return None

        output = KVConnectorOutput()
        if wait_for_save:
            self.kv_connector.wait_for_save()
        output.finished_sending, output.finished_recving = (
            self.kv_connector.get_finished(finished_req_ids)
        )
        output.invalid_block_ids = self.kv_connector.get_block_ids_with_load_errors()
        output.kv_connector_stats = self.kv_connector.get_kv_connector_stats()
        output.kv_cache_events = self.kv_connector.get_kv_connector_kv_cache_events()
        output.kv_connector_worker_meta = (
            self.kv_connector.build_connector_worker_meta()
        )
        self.kv_connector.clear_connector_metadata()
        return output

    def no_forward(self, scheduler_output: "SchedulerOutput") -> ModelRunnerOutput:
        """当没有实际的前向传播时调用。

        在这种情况下，只执行 KV 缓存的预取和保存操作，
        不执行模型的前向传播。

        参数:
            scheduler_output: 调度器输出

        返回:
            ModelRunnerOutput: 仅包含 KV 连接器输出的模型输出
        """
        if self._disabled:
            return EMPTY_MODEL_RUNNER_OUTPUT

        self.pre_forward(scheduler_output)
        finished_req_ids = scheduler_output.finished_req_ids
        kv_connector_output = self.post_forward(finished_req_ids, wait_for_save=False)
        return ModelRunnerOutput.with_kv_conn_output_only(kv_connector_output)

    def set_disabled(self, disabled: bool) -> None:
        """设置连接器的禁用状态。

        禁用时，确保层级连接器钩子不会被调用。

        参数:
            disabled: 是否禁用
        """
        # Ensure that layer-wise connector hooks aren't called when disabled.
        kv_transfer_state._KV_CONNECTOR_AGENT = None if disabled else self.kv_connector
        self._disabled = disabled


# 全局的空操作连接器实例，当不需要 KV 传输时使用
NO_OP_KV_CONNECTOR = KVConnector()


def get_kv_connector(
    vllm_config: VllmConfig, kv_caches_dict: dict[str, torch.Tensor]
) -> KVConnector:
    """获取 KV 连接器实例。

    如果没有配置 KV 传输组，返回空操作连接器；
    否则返回活跃的 KV 连接器。

    参数:
        vllm_config: vLLM 全局配置
        kv_caches_dict: KV 缓存字典

    返回:
        KVConnector: KV 连接器实例
    """
    if not has_kv_transfer_group():
        # No-op connector.
        return NO_OP_KV_CONNECTOR

    return ActiveKVConnector(vllm_config, kv_caches_dict)
