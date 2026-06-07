# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
异步输出处理工具模块。

本模块提供将模型推理结果从 GPU 异步拷贝到 CPU 的功能，避免在主 CUDA 流上
进行阻塞式的数据传输，从而提升整体吞吐量。

主要包含以下组件：
1. AsyncOutput - 文本生成模型的异步输出处理器
2. AsyncPoolingOutput - 池化模型（如嵌入模型）的异步输出处理器
3. async_copy_to_gpu - 辅助函数，将张量异步拷贝到 GPU
"""
import contextlib

import numpy as np
import torch

from vllm.v1.outputs import AsyncModelRunnerOutput, LogprobsTensors, ModelRunnerOutput
from vllm.v1.worker.gpu.sample.output import SamplerOutput


class AsyncOutput(AsyncModelRunnerOutput):
    """文本生成模型的异步输出处理器。

    该类负责将模型前向传播和采样器产生的 GPU 张量异步拷贝到 CPU，
    使用独立的 CUDA 拷贝流（copy_stream）来避免阻塞主计算流。

    工作流程：
    1. 在 copy_stream 上发起非阻塞的 GPU->CPU 拷贝操作
    2. 记录一个 CUDA 事件（copy_event）标记拷贝完成点
    3. 当需要使用输出时，调用 get_output() 等待拷贝完成

    属性:
        model_runner_output: 模型运行器的基础输出（包含请求 ID、已完成请求等）
        sampler_output: 采样器输出（包含采样 token ID、logprobs 等）
        num_sampled_tokens: 每个请求实际采样的 token 数量
        copy_event: CUDA 事件，用于同步拷贝操作
        sampled_token_ids: 已拷贝到 CPU 的采样 token ID（numpy 数组）
        logprobs_tensors: 已拷贝到 CPU 的 logprobs 张量
        num_nans: logits 中 NaN 的数量统计
        num_sampled_tokens_np: 已拷贝到 CPU 的采样 token 数量（numpy 数组）
    """

    def __init__(
        self,
        model_runner_output: ModelRunnerOutput,
        sampler_output: SamplerOutput,
        num_sampled_tokens: torch.Tensor,
        main_stream: torch.cuda.Stream,
        copy_stream: torch.cuda.Stream,
    ):
        # NOTE(woosuk): 必须保留对 GPU 张量的引用，
        # 因为拷贝操作在与张量创建时不同的 CUDA 流上执行。
        # 如果提前释放引用，可能导致数据丢失。
        self.model_runner_output = model_runner_output
        self.sampler_output = sampler_output
        self.num_sampled_tokens = num_sampled_tokens
        # 创建 CUDA 事件，用于后续同步等待拷贝完成
        self.copy_event = torch.cuda.Event()

        with stream(copy_stream, main_stream):
            # 等待主计算流完成后再开始拷贝，确保数据一致性
            copy_stream.wait_stream(main_stream)

            # 异步拷贝采样 token IDs 到 CPU（转为 numpy 数组以节省内存）
            self.sampled_token_ids = async_copy_to_np(sampler_output.sampled_token_ids)
            # 异步拷贝 logprobs 张量到 CPU
            self.logprobs_tensors: LogprobsTensors | None = None
            if sampler_output.logprobs_tensors is not None:
                self.logprobs_tensors = (
                    sampler_output.logprobs_tensors.to_cpu_nonblocking()
                )
            # 异步拷贝 NaN 计数到 CPU
            self.num_nans: np.ndarray | None = None
            if sampler_output.num_nans is not None:
                self.num_nans = async_copy_to_np(sampler_output.num_nans)
            # 异步拷贝采样 token 数量到 CPU
            self.num_sampled_tokens_np = async_copy_to_np(num_sampled_tokens)
            # 异步拷贝 prompt logprobs 字典到 CPU
            self.prompt_logprobs_dict = {
                k: v.to_cpu_nonblocking() if v is not None else None
                for k, v in self.model_runner_output.prompt_logprobs_dict.items()
            }
            # 记录拷贝完成事件
            self.copy_event.record(copy_stream)

    def get_output(self) -> ModelRunnerOutput:
        """等待异步拷贝完成后，组装并返回最终的 ModelRunnerOutput。

        该方法执行以下操作：
        1. 同步等待 GPU->CPU 拷贝完成
        2. 将 numpy 数组转换回 Python 列表格式
        3. 裁剪掉多余的 padding token
        4. 组装完整的输出结构

        返回:
            ModelRunnerOutput: 包含所有推理结果的输出对象
        """
        # 等待所有异步拷贝操作完成
        self.copy_event.synchronize()

        # NOTE(woosuk): 以下代码确保与现有模型运行器的兼容性。
        # 未来应将数据结构保持为 NumPy 数组，而不是 Python 列表，
        # 以减少内存拷贝和提升性能。
        sampled_token_ids: list[list[int]] = self.sampled_token_ids.tolist()
        num_sampled_tokens: list[int] = self.num_sampled_tokens_np.tolist()
        # 裁剪每个请求中多余的 padding token
        for token_ids, num_tokens in zip(sampled_token_ids, num_sampled_tokens):
            del token_ids[num_tokens:]
        self.model_runner_output.sampled_token_ids = sampled_token_ids

        # 组装 NaN 计数信息
        if self.num_nans is not None:
            self.model_runner_output.num_nans_in_logits = dict(
                zip(self.model_runner_output.req_ids, self.num_nans.tolist())
            )

        # 组装 logprobs 信息
        if self.logprobs_tensors is not None:
            self.model_runner_output.logprobs = self.logprobs_tensors.tolists()
        self.model_runner_output.prompt_logprobs_dict = self.prompt_logprobs_dict
        return self.model_runner_output


class AsyncPoolingOutput(AsyncModelRunnerOutput):
    """池化模型（如嵌入模型、重排序模型）的异步输出处理器。

    与 AsyncOutput 类似，但处理的是池化层的输出向量而非采样的 token。
    适用于嵌入（embedding）、分类、重排序等不需要逐 token 采样的场景。

    属性:
        model_runner_output: 模型运行器的基础输出
        pooler_output: 池化层产生的嵌入/分类向量（GPU 张量）
        is_valid: 每个请求的输出是否有效的标志（GPU 张量）
        copy_event: CUDA 事件，用于同步拷贝操作
    """

    def __init__(
        self,
        model_runner_output: ModelRunnerOutput,
        pooler_output: torch.Tensor,
        is_valid: torch.Tensor | None,
        main_stream: torch.cuda.Stream,
        copy_stream: torch.cuda.Stream,
    ):
        self.model_runner_output = model_runner_output
        self.pooler_output = pooler_output
        self.is_valid = is_valid
        self.copy_event = torch.cuda.Event()

        with stream(copy_stream, main_stream):
            # 等待主计算流完成
            copy_stream.wait_stream(main_stream)
            # 异步拷贝池化输出到 CPU
            self.pooler_output_cpu = self.pooler_output.to("cpu", non_blocking=True)
            # 异步拷贝有效性标志到 CPU
            if self.is_valid is not None:
                self.is_valid_cpu = self.is_valid.to("cpu", non_blocking=True)
            else:
                self.is_valid_cpu = None
            # 记录拷贝完成事件
            self.copy_event.record(copy_stream)

    def get_output(self -> ModelRunnerOutput:
        """等待异步拷贝完成后，组装并返回最终的 ModelRunnerOutput。

        返回:
            ModelRunnerOutput: 包含池化输出的输出对象。无效请求的输出被置为 None。
        """
        # 将池化输出按第一个维度拆分为列表
        pooler_output = list(self.pooler_output_cpu.unbind(dim=0))
        # 等待拷贝完成
        self.copy_event.synchronize()
        # 将无效请求的输出置为 None
        if self.is_valid_cpu is not None:
            is_valid_cpu = self.is_valid_cpu.tolist()
            for i, is_valid in enumerate(is_valid_cpu):
                if not is_valid:
                    pooler_output[i] = None
        self.model_runner_output.pooler_output = pooler_output
        return self.model_runner_output


def async_copy_to_np(x: torch.Tensor) -> np.ndarray:
    """将 GPU 张量异步拷贝到 CPU 并转换为 NumPy 数组。

    这是一种高效的 GPU->CPU 数据传输方式：
    1. 非阻塞地将数据从 GPU 拷贝到 CPU 内存
    2. 通过 numpy() 将 pinned memory 的张量直接转为 numpy 视图（零拷贝）

    参数:
        x: GPU 上的 PyTorch 张量

    返回:
        np.ndarray: CPU 上的 NumPy 数组（共享 pinned memory）
    """
    return x.to("cpu", non_blocking=True).numpy()


@contextlib.contextmanager
def stream(to_stream: torch.cuda.Stream, from_stream: torch.cuda.Stream):
    """轻量级的 CUDA 流上下文管理器。

    这是 torch.cuda.stream() 的轻量替代版本，避免了当前流和设备的查找开销。
    在上下文管理器内部，CUDA 操作会在 to_stream 上执行；
    退出时恢复到 from_stream。

    参数:
        to_stream: 要切换到的目标 CUDA 流
        from_stream: 上下文结束时恢复的原始 CUDA 流
    """
    try:
        torch.cuda.set_stream(to_stream)
        yield
    finally:
        torch.cuda.set_stream(from_stream)
