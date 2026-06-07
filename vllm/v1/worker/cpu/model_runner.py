# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
CPU 模型运行器模块 (vllm/v1/worker/cpu/model_runner.py)

本模块实现了 CPU 后端的模型运行器，继承自 GPU 模型运行器 (GPUModelRunner)。

设计思路：
- CPUModelRunner 继承 GPUModelRunner 的全部功能，只覆盖预热方法 warming_up_model。
- GPU 版本的预热需要编译 CUDA 图（CUDA Graph）并为多种 batch size 形状生成执行图。
- CPU 版本只需要为通用形状（generic shape）生成计算图即可，无需多种 batch size 的优化。
- 通过只覆盖 warming_up_model 方法，最大限度复用 GPU 模型运行器的代码。

注意：
- warming_up_model 在模型权重加载完成后调用，用于触发 JIT 编译或图捕获，
  确保首次实际推理时不会因编译而延迟。
"""

from vllm.logger import init_logger
from vllm.v1.worker.gpu.model_runner import GPUModelRunner

logger = init_logger(__name__)


class CPUModelRunner(GPUModelRunner):
    """
    CPU 后端的模型运行器。

    继承 GPUModelRunner 的所有功能，仅覆盖模型预热方法。
    GPU 版本需要为多种 batch size 形状预编译 CUDA Graph；
    CPU 版本只需为通用形状预编译一次即可。
    """

    # TBD: Whether need to move this to Worker?
    # 待定：是否需要将此方法移到 Worker 层？
    def warming_up_model(self) -> None:
        """
        模型预热方法。

        在模型权重加载完成后调用，触发一次 profile_run 来完成：
        1. JIT 编译（如 torch.compile 生成的内核）
        2. 内存分配优化
        3. 任何延迟初始化操作

        与 GPU 版本不同，CPU 版本只需为通用形状生成计算图，
        不需要为多种 batch size 预编译 CUDA Graph。
        """
        logger.info("Warming up model for the compilation...")
        # 仅生成通用形状的计算图（CPU 上不需要多种 batch size 的优化）
        self.profile_run()
        logger.info("Warming up done.")
