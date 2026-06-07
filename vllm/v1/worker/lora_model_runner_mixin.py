# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Define LoRA functionality mixin for model runners.

LoRA（Low-Rank Adaptation）模型运行器混入模块：
    本模块为 vLLM V1 引擎的 ModelRunner（如 GPUModelRunner）提供 LoRA 功能支持。
    采用 Mixin 模式，将 LoRA 相关逻辑从主 ModelRunner 中解耦，便于维护和扩展。

核心职责：
    1. LoRA 模型加载：在 ModelRunner 初始化时加载 LoRA 管理器，将原始模型转换为支持 LoRA 的模型。
    2. 活跃 LoRA 设置：在每次 forward 之前，根据当前批次中的请求，设置哪些 LoRA 适配器处于活跃状态。
    3. LoRA 权重管理：提供添加、移除、pin 住、列举 LoRA 适配器的接口。
    4. CUDA Graph 预热：通过 dummy LoRA 请求，在 CUDA Graph 捕获阶段预热各种 LoRA 配置，
       确保运行时不同 LoRA 组合都能命中已捕获的 CUDA Graph，避免运行时重新编译。

与 LoRAMapping 的关系：
    LoRAMapping 将批次中每个 token 映射到对应的 LoRA 适配器 ID。
    这样在模型 forward 时，每个 token 的 K/V 计算可以使用对应的 LoRA 权重。
    映射分为 prompt 级别（per-request）和 token 级别（per-token），分别用于不同的计算路径。
"""

from contextlib import contextmanager
from typing import TypeAlias

import numpy as np
import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.config.lora import LoRAConfig
from vllm.logger import init_logger
from vllm.lora.layers import LoRAMapping, LoRAMappingType
from vllm.lora.request import LoRARequest
from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager
from vllm.model_executor.models import supports_lora
from vllm.v1.worker.gpu_input_batch import InputBatch as GPUInputBatch
from vllm.v1.worker.tpu_input_batch import InputBatch as TPUInputBatch

# 中文注释：InputBatch 类型别名，兼容 GPU 和 TPU 两种设备的输入批次数据结构。
# GPU 和 TPU 的 InputBatch 都实现了 make_lora_inputs() 方法，用于生成 LoRA 映射。
InputBatch: TypeAlias = TPUInputBatch | GPUInputBatch

logger = init_logger(__name__)


# Defined as a mixin for GPUModelRunner
# 中文注释：LoRAModelRunnerMixin 是一个混入类，为 ModelRunner 提供完整的 LoRA 生命周期管理。
# 设计思路：
#   - 使用 Mixin 模式而非继承，是因为 LoRA 功能是可选的，不应当强制所有 ModelRunner 继承。
#   - ModelRunner 只需在类声明中混入此 Mixin，即可获得所有 LoRA 功能。
#   - 内部依赖 LRUCacheWorkerLoRAManager 来管理 LoRA 权重的加载/卸载/缓存。
#
# 主要方法分类：
#   1. 模型加载：load_lora_model() -- 将原始模型包装为支持 LoRA 的模型
#   2. 活跃 LoRA 设置：set_active_loras() / _set_active_loras() -- 每次 forward 前调用
#   3. CUDA Graph 预热：maybe_setup_dummy_loras() / maybe_select_dummy_loras() / maybe_dummy_run_with_lora()
#   4. LoRA 权重管理：add_lora() / remove_lora() / pin_lora() / list_loras()
class LoRAModelRunnerMixin:
    def load_lora_model(
        self,
        model: nn.Module,
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> nn.Module:
        """加载并初始化 LoRA 模型。

        流程说明：
            步骤1：检查模型是否支持 LoRA（通过 @supports_lora 装饰器注册）。
                   不支持 LoRA 的模型无法使用此功能。
            步骤2：创建 LRUCacheWorkerLoRAManager 实例，它负责：
                   - 管理 LoRA 权重的 LRU 缓存（当缓存满时淘汰最久未使用的 LoRA）
                   - 从磁盘加载 LoRA 权重到 GPU 显存
                   - 提供 dummy LoRA 创建能力（用于预热）
            步骤3：调用 create_lora_manager() 将原始模型包装为支持 LoRA 的模型。
                   这会将模型中的 Linear 层替换为 LoRA 兼容的层（如 ColumnParallelLinearWithLoRA），
                   使得模型在 forward 时能够根据活跃的 LoRA 适配器动态调整权重。

        Args:
            model: 原始 PyTorch 模型。
            vllm_config: vLLM 全局配置，包含 LoRA 配置信息。
            device: 计算设备（如 cuda:0）。

        Returns:
            包装后的支持 LoRA 的模型。
        """
        if not supports_lora(model):
            raise ValueError(f"{model.__class__.__name__} does not support LoRA yet.")

        # Add LoRA Manager to the Model Runner
        # 中文注释：创建 LoRA 工作管理器，它是 LoRA 权重缓存和加载的核心组件。
        # LRUCacheWorkerLoRAManager 使用 LRU 策略管理 LoRA 权重的 GPU 显存占用。
        self.lora_manager = LRUCacheWorkerLoRAManager(
            vllm_config,
            device,
            model.embedding_modules,
        )
        # 中文注释：将原始模型包装为支持 LoRA 的模型。
        # 这一步会替换模型中的注意力层和线性层，使其支持 LoRA adapter 的动态切换。
        return self.lora_manager.create_lora_manager(model, vllm_config)

    def _set_active_loras(
        self,
        prompt_lora_mapping: tuple[int, ...],
        token_lora_mapping: tuple[int, ...],
        lora_requests: set[LoRARequest],
        mapping_type: LoRAMappingType = LoRAMappingType.LANGUAGE,
    ) -> None:
        """内部方法：设置当前批次中活跃的 LoRA 适配器。

        流程说明：
            步骤1：确保 LoRA 功能已启用（lora_manager 已初始化）。
            步骤2：构建 LoRAMapping 对象，它包含两个关键映射：
                   - prompt_lora_mapping：prompt 级别的映射，长度为采样 token 总数，
                     将每个采样位置映射到对应的 LoRA ID。
                   - token_lora_mapping：token 级别的映射，长度为调度 token 总数，
                     将每个 token 映射到对应的 LoRA ID。
                   LoRA ID 为 -1 表示该 token 不使用任何 LoRA（使用基础模型权重）。
                   LoRA ID 为 0 表示该位置未分配 LoRA。
                   LoRA ID >= 1 表示使用对应 ID 的 LoRA 适配器。
            步骤3：通过 lora_manager.set_active_adapters() 将映射设置到 LoRA 算子中，
                   这样后续的模型 forward 就能根据映射自动选择正确的 LoRA 权重。

        关于 is_prefill 参数：
            在非 CUDA 平台（如 TPU），is_prefill=True 始终使用 SGMV kernel，
            因为这些平台不区分 prefill 和 decode 使用不同的 kernel。
            在 CUDA 平台上，prefill 和 decode 使用相同的 kernel，此标志被忽略。

        Args:
            prompt_lora_mapping: prompt 级别的 LoRA 映射，长度 = sum(num_sampled_tokens)。
            token_lora_mapping: token 级别的 LoRA 映射，长度 = sum(num_scheduled_tokens)。
            lora_requests: 当前批次中活跃的 LoRA 请求集合。
            mapping_type: 映射类型（LANGUAGE 或其他模态）。
        """
        self._ensure_lora_enabled()

        # Set is_prefill to True, so we always use the SGMV kernels on
        # non-cuda platforms.
        # On cuda platforms we use the same kernels for prefill and
        # decode and this flag is generally ignored.
        lora_mapping = LoRAMapping(
            token_lora_mapping,
            prompt_lora_mapping,
            is_prefill=True,
            type=mapping_type,
        )
        self.lora_manager.set_active_adapters(lora_requests, lora_mapping)

    def _ensure_lora_enabled(self) -> None:
        """检查 LoRA 功能是否已启用。

        如果 lora_manager 属性不存在，说明 ModelRunner 初始化时未启用 LoRA，
        此时抛出异常提示用户需要通过 --enable-lora 启动参数来启用 LoRA 功能。
        """
        if not hasattr(self, "lora_manager"):
            raise RuntimeError("LoRA is not enabled. Use --enable-lora to enable LoRA.")

    def set_active_loras(
        self,
        input_batch: InputBatch,
        num_scheduled_tokens: np.ndarray,
        num_sampled_tokens: np.ndarray | None = None,
        mapping_type: LoRAMappingType = LoRAMappingType.LANGUAGE,
    ) -> None:
        """根据当前输入批次设置活跃的 LoRA 适配器。

        这是在每次模型 forward 之前调用的核心方法。它从 InputBatch 中提取
        当前批次的 LoRA 信息，并将其设置到 LoRA 算子中。

        流程说明：
            步骤1：如果未提供 num_sampled_tokens，默认每个请求采样 1 个 token（decode 场景）。
            步骤2：调用 input_batch.make_lora_inputs() 从输入批次中生成：
                   - prompt_lora_mapping：prompt 级别的 LoRA 映射
                   - token_lora_mapping：token 级别的 LoRA 映射
                   - lora_requests：当前批次中活跃的 LoRA 请求集合
                   make_lora_inputs 内部会遍历批次中所有请求，根据每个请求关联的
                   LoRA ID 生成对应的映射数组。
            步骤3：调用 _set_active_loras() 将映射设置到 LoRA 算子。

        Args:
            input_batch: 当前输入批次，包含所有请求的状态和 LoRA 信息。
            num_scheduled_tokens: 每个请求本轮调度的 token 数量，shape 为 (num_reqs,)。
            num_sampled_tokens: 每个请求本轮采样的 token 数量，shape 为 (num_reqs,)。
                在 decode 阶段通常每个请求采样 1 个 token。
            mapping_type: 映射类型，区分语言模型和其他模态。
        """
        if num_sampled_tokens is None:
            num_sampled_tokens = np.ones_like(num_scheduled_tokens, dtype=np.int32)

        prompt_lora_mapping: tuple[int, ...]  # of size np.sum(num_sampled_tokens)
        token_lora_mapping: tuple[int, ...]  # of size np.sum(num_scheduled_tokens)
        lora_requests: set[LoRARequest]
        prompt_lora_mapping, token_lora_mapping, lora_requests = (
            input_batch.make_lora_inputs(num_scheduled_tokens, num_sampled_tokens)
        )
        return self._set_active_loras(
            prompt_lora_mapping, token_lora_mapping, lora_requests, mapping_type
        )

    @contextmanager
    def maybe_setup_dummy_loras(
        self, lora_config: LoRAConfig | None, remove_lora: bool = True
    ):
        """上下文管理器：设置 dummy LoRA 用于 CUDA Graph 预热和捕获。

        背景说明：
            CUDA Graph 捕获需要在实际推理之前完成，且捕获时的内存布局必须与
            运行时一致。因此需要使用 dummy LoRA 来预热 LoRA 相关的算子和缓冲区，
            确保 CUDA Graph 中记录的 kernel 执行序列在运行时可以正确回放。

        流程说明：
            步骤1：如果 lora_config 为 None（未启用 LoRA），直接 yield，不做任何操作。
            步骤2：确定预热使用的 LoRA rank。取 max_lora_rank 和 8 中的较小值，
                   这是因为 CUDA Graph 的 key 与 rank 无关，只需要验证不同 rank
                   值下 kernel 能正常工作即可。使用较小的 rank 可以减少预热时的显存开销。
            步骤3：创建 max_loras 个 dummy LoRA 请求。这些请求的 lora_path 不指向
                   真实文件，因为它们只用于预热，不会真正加载权重。
            步骤4：进入 dummy_lora_cache 上下文，该上下文会临时修改 LoRA 管理器的行为，
                   使其在添加 LoRA 时不从磁盘加载权重，而是创建随机权重的 dummy LoRA。
            步骤5：为每个 dummy LoRA 请求添加 dummy LoRA 适配器。
            步骤6：yield，将控制权交给调用方执行 CUDA Graph 捕获。
            步骤7：退出时，如果 remove_lora=True，移除所有已添加的 dummy LoRA 适配器，
                   清理预热状态。

        Args:
            lora_config: LoRA 配置，如果为 None 则跳过所有 LoRA 设置。
            remove_lora: 退出时是否移除所有 LoRA 适配器。
        """
        if lora_config is None:
            yield
        else:
            # __enter__ code
            assert self.lora_manager is not None, "LoRA is not enabled"

            num_loras = lora_config.max_loras
            # 中文注释：预热 rank 取 max_lora_rank 和 8 的较小值。
            # CUDA Graph 的 key 不依赖具体 rank，只需确保不同 rank 的 kernel 能正常运行。
            # 使用较小的 rank 可以减少预热时的显存和计算开销。
            lora_warmup_rank: int = (
                lora_config.max_lora_rank if lora_config.max_lora_rank < 8 else 8
            )
            lora_warmup_rank = self.lora_manager.get_dummy_lora_warmup_rank(
                lora_warmup_rank
            )
            # Make dummy lora requests
            # 中文注释：创建 max_loras 个 dummy LoRA 请求，用于预热。
            # 这些请求的 lora_path 是无效路径，因为 dummy LoRA 不会从磁盘加载权重。
            lora_requests: set[LoRARequest] = {
                LoRARequest(
                    lora_name=f"warmup_{lora_id}",
                    lora_int_id=lora_id,
                    lora_path="/not/a/real/path",
                )
                for lora_id in range(1, num_loras + 1)
            }

            with self.lora_manager.dummy_lora_cache():
                # Add the dummy LoRAs here so _set_active_loras doesn't try to
                # load from disk.
                # 中文注释：在 dummy_lora_cache 上下文中添加 dummy LoRA，
                # 此时 LoRA 管理器会创建随机权重的 LoRA 适配器而非从磁盘加载。
                for lr in lora_requests:
                    self.lora_manager.add_dummy_lora(lr, rank=lora_warmup_rank)

                yield

            # __exit__ code
            # 中文注释：预热完成后清理所有 dummy LoRA 适配器，释放占用的显存。
            if remove_lora:
                self.lora_manager.remove_all_adapters()

    @contextmanager
    def maybe_select_dummy_loras(
        self,
        lora_config: LoRAConfig | None,
        num_scheduled_tokens: np.ndarray,
        mapping_type: LoRAMappingType = LoRAMappingType.LANGUAGE,
        num_sampled_tokens: np.ndarray | None = None,
        num_active_loras: int = 0,
    ):
        """
        Context manager to select dummy LoRAs for capture/warmup.

        中文注释：上下文管理器，为 CUDA Graph 捕获/预热选择 dummy LoRA 映射。

        背景说明：
            vLLM 使用 CUDA Graph 来加速模型推理。CUDA Graph 捕获时，LoRA 的
            活跃适配器数量会影响 kernel 的执行路径和缓冲区大小。因此需要在
            捕获阶段模拟不同数量的活跃 LoRA，以确保运行时各种 LoRA 组合
            都能命中已捕获的 CUDA Graph。

        关于 num_active_loras 的三种情况：
            - 0：不使用任何 LoRA，所有 token 映射到 LoRA ID 0。
            - 1 ~ max_loras：使用指定数量的活跃 LoRA 适配器。
            - > max_loras（如 max_loras + 1）：使用 max_loras 个适配器，
              并包含无 LoRA 的 token（映射到 -1）。这是最复杂的场景，
              需要预热时覆盖以确保 CUDA Graph 的 key 正确。

        流程说明：
            步骤1：如果未启用 LoRA（lora_config 为 None），直接 yield。
            步骤2：确定 effective_num_loras（实际使用的 LoRA 数量）和
                   是否包含无 LoRA token（include_no_lora）。
            步骤3：构建 prompt_lora_mapping，将请求循环分配到不同的 LoRA ID。
                   这模拟了最坏情况——每个活跃 LoRA 都被分配到至少一个请求。
            步骤4：通过 np.repeat 将 prompt 映射扩展为 sample 和 token 级别映射。
                   - sample_lora_mapping：每个采样位置一个映射值
                   - token_lora_mapping：每个调度 token 一个映射值
            步骤5：创建 dummy LoRA 请求并调用 _set_active_loras() 设置映射。
            步骤6：yield，将控制权交给调用方执行 CUDA Graph 捕获。

        Args:
            lora_config: LoRA configuration, or None if LoRA is disabled.
            num_scheduled_tokens: Array of scheduled token counts per request.
            num_sampled_tokens: Array of sampled token counts per request.
            num_active_loras: Number of distinct active LoRAs to use.
                - 0: No LoRA active (set up zero mappings).
                - >0: Use exactly this many distinct LoRAs.
        """
        if num_sampled_tokens is None:
            num_sampled_tokens = np.ones_like(num_scheduled_tokens, dtype=np.int32)

        # Skip LoRA setup entirely only if no LoRA config
        if lora_config is None:
            yield
        else:
            # __enter__ code
            assert self.lora_manager is not None, "LoRA is not enabled"

            num_reqs = len(num_scheduled_tokens)
            max_loras = lora_config.max_loras

            # Determine how many distinct LoRAs to use and whether to include
            # no-LoRA tokens (-1 entries).
            # When num_active_loras > max_loras (e.g., max_loras + 1), we need
            # to include -1 entries to simulate batches with both LoRA and
            # no-LoRA tokens. This ensures prepare_tensors computes the correct
            # num_active_loras that matches the cudagraph capture key.
            # 中文注释：根据 num_active_loras 确定实际使用的 LoRA 数量和是否包含无 LoRA token。
            # 当 num_active_loras > max_loras 时，需要模拟包含 LoRA 和非 LoRA token 的混合批次。
            # 这确保 CUDA Graph 捕获时的 key 与运行时一致。
            if num_active_loras == 0:
                # No LoRA active - use 0 mappings like the original code
                effective_num_loras = 0
                include_no_lora = False
            elif num_active_loras > max_loras:
                # num_active_loras > max_loras means we want max_loras adapters
                # PLUS no-LoRA tokens (-1). This is the max_loras + 1 case.
                effective_num_loras = max_loras
                include_no_lora = True
            else:
                # Specific number of active LoRAs requested
                effective_num_loras = min(num_active_loras, max_loras)
                include_no_lora = False

            # Make prompt lora mapping
            # Assign LoRA IDs cyclically to simulate a worst-case scenario.
            # LoRA IDs are 1-indexed (1 to max_loras) as required by LoRARequest.
            # convert_mapping() will convert these to 0-indexed slot indices.
            # 中文注释：构建 prompt 级别的 LoRA 映射。
            # 采用循环分配策略，确保每个活跃的 LoRA 都被分配到至少一个请求，
            # 从而覆盖最坏情况下的 CUDA Graph 捕获场景。
            if effective_num_loras > 0:
                if include_no_lora:
                    # Include -1 (no-LoRA) entries by cycling through
                    # -1, 1, 2, ..., effective_num_loras
                    # This ensures prepare_tensors sees both LoRA and no-LoRA
                    # tokens, computing num_active_loras = effective_num_loras+1
                    cycle_values = np.array(
                        list(range(1, effective_num_loras + 1)),
                        dtype=np.int32,
                    )
                    prompt_lora_mapping = cycle_values[
                        np.arange(num_reqs, dtype=np.int32) % len(cycle_values)
                    ]
                else:
                    # Use 1 to effective_num_loras (1-indexed lora IDs)
                    prompt_lora_mapping = (
                        np.arange(num_reqs, dtype=np.int32) % effective_num_loras
                    ) + 1
            else:
                # No LoRA active - use 0 for all tokens (original behavior)
                prompt_lora_mapping = np.zeros(num_reqs, dtype=np.int32)

            # Make sample lora mapping
            # 中文注释：将 prompt 级别映射按采样 token 数展开为 sample 级别映射。
            sample_lora_mapping = np.repeat(prompt_lora_mapping, num_sampled_tokens)

            # Make token lora mapping
            # 中文注释：将 prompt 级别映射按调度 token 数展开为 token 级别映射。
            token_lora_mapping = np.repeat(prompt_lora_mapping, num_scheduled_tokens)

            # Make dummy lora requests (only for the active LoRAs)
            # 中文注释：仅为实际活跃的 LoRA 创建 dummy 请求。
            lora_requests: set[LoRARequest] = {
                LoRARequest(
                    lora_name=f"warmup_{lora_id}",
                    lora_int_id=lora_id,
                    lora_path="/not/a/real/path",
                )
                for lora_id in range(1, effective_num_loras + 1)
            }

            self._set_active_loras(
                tuple(sample_lora_mapping),
                tuple(token_lora_mapping),
                lora_requests,
                mapping_type,
            )

            yield

    @contextmanager
    def maybe_dummy_run_with_lora(
        self,
        lora_config: LoRAConfig | None,
        num_scheduled_tokens: np.ndarray,
        num_sampled_tokens: np.ndarray,
        remove_lora: bool = True,
        num_active_loras: int = 0,
        mapping_type: LoRAMappingType = LoRAMappingType.LANGUAGE,
    ):
        """
        Context manager for dummy runs with LoRA.

        中文注释：组合两个 LoRA 预热步骤的上下文管理器，用于 CUDA Graph 捕获的完整 dummy run。

        这个方法将两个关键步骤组合在一起：
            1. maybe_setup_dummy_loras：创建并加载 dummy LoRA 适配器到 LoRA 管理器。
            2. maybe_select_dummy_loras：设置 dummy LoRA 映射，模拟运行时的批次配置。

        为什么需要组合使用：
            - setup_dummy_loras 负责"创建"dummy LoRA 适配器（分配显存、初始化权重）。
            - select_dummy_loras 负责"选择"哪些 dummy LoRA 在当前批次中活跃。
            - 两者必须同时生效，才能在 CUDA Graph 捕获时模拟真实的 LoRA 推理场景。

        使用场景：
            在 GPUModelRunner 的 CUDA Graph 捕获阶段，需要对每种可能的
            (batch_size, num_active_loras) 组合进行预热。此方法提供了一种
            简洁的方式来完成这一过程。

        Args:
            lora_config: LoRA configuration.
            num_scheduled_tokens: Array of scheduled token counts per request.
            num_sampled_tokens: Array of sampled token counts per request.
            remove_lora: Whether to remove LoRAs after the context exits.
            num_active_loras: Number of distinct active LoRAs to use.
                LoRA is activated when num_active_loras > 0.
        """
        with (
            self.maybe_setup_dummy_loras(lora_config, remove_lora),
            self.maybe_select_dummy_loras(
                lora_config,
                num_scheduled_tokens,
                mapping_type,
                num_sampled_tokens,
                num_active_loras,
            ),
        ):
            yield

    # =========================================================================
    # LoRA 权重管理方法
    # 以下方法提供 LoRA 适配器的增删查管理接口。
    # 这些方法通过 lora_manager（LRUCacheWorkerLoRAManager）操作底层的 LoRA 权重。
    # =========================================================================

    def maybe_remove_all_loras(self, lora_config: LoRAConfig | None):
        """如果启用了 LoRA，则移除所有已加载的 LoRA 适配器。

        在模型卸载或引擎关闭时调用，用于清理 LoRA 资源。
        """
        if lora_config is None:
            return
        self.lora_manager.remove_all_adapters()

    def add_lora(self, lora_request: LoRARequest) -> bool:
        """添加一个 LoRA 适配器到缓存中。

        当新请求携带 LoRA 时，Engine 会调用此方法将对应的 LoRA 权重加载到 GPU 显存。
        如果 LoRA 权重已在缓存中（LRU 命中），则无需重新加载。

        Args:
            lora_request: LoRA 请求，包含 LoRA 名称、ID 和权重路径。

        Returns:
            True 如果成功添加，False 如果适配器已存在。
        """
        self._ensure_lora_enabled()
        return self.lora_manager.add_adapter(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        """从缓存中移除指定 ID 的 LoRA 适配器。

        当某个 LoRA 不再被任何活跃请求使用时，可以调用此方法释放其占用的显存。

        Args:
            lora_id: 要移除的 LoRA 适配器 ID。

        Returns:
            True 如果成功移除，False 如果适配器不存在。
        """
        self._ensure_lora_enabled()
        return self.lora_manager.remove_adapter(lora_id)

    def pin_lora(self, lora_id: int) -> bool:
        """Pin 住一个 LoRA 适配器，防止其被 LRU 淘汰。

        对于高频使用的 LoRA，可以 pin 住以确保其始终保留在 GPU 显存中，
        避免反复加载带来的开销。

        Args:
            lora_id: 要 pin 住的 LoRA 适配器 ID。

        Returns:
            True 如果成功 pin 住。
        """
        self._ensure_lora_enabled()
        return self.lora_manager.pin_adapter(lora_id)

    def list_loras(self) -> set[int]:
        """列出当前缓存中所有已加载的 LoRA 适配器 ID。

        Returns:
            当前缓存中的 LoRA 适配器 ID 集合。
        """
        self._ensure_lora_enabled()
        return self.lora_manager.list_adapters()
