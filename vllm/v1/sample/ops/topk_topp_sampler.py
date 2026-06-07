# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Top-K / Top-P 采样器模块

本模块实现了 Top-K 和 Top-P (核采样) 两种主流的采样策略，用于从语言模型
输出的 logits 中选择下一个生成的 token。

算法原理：
1. Top-K 采样：只保留 logits 最高的 K 个 token，将其余 token 的 logits
   设为负无穷，然后在剩余的 K 个 token 中按概率随机采样。
   例如 k=50 时，只在概率最高的 50 个 token 中选择。

2. Top-P (Nucleus) 采样：将 token 按概率从高到低排序，保留累积概率
   刚好超过 P 的最小 token 集合，然后在其中随机采样。
   例如 p=0.9 时，保留累积概率达到 90% 的最小 token 集合。

3. 两者可以组合使用：先应用 Top-K 缩小候选集，再在候选集上应用 Top-P。

采样技术（拒绝采样 / Gumbel-max-trick）：
  本模块使用的随机采样方法等价于 torch.multinomial，但避免了 CPU-GPU
  同步开销。核心思想是：对概率分布 p 采样等价于 argmax(log(p) - log(E))
  其中 E 服从指数分布。即 probs.div(q).argmax() 等价于多项式采样。

支持的硬件后端：
  - forward_native: PyTorch 原生实现，适用于所有平台
  - forward_cuda: 使用 FlashInfer 库的 CUDA 优化实现
  - forward_cpu: CPU 平台特化实现
  - forward_hip: ROCm (AMD GPU) 平台的 aiter 实现
  - forward_xpu: Intel XPU 平台的专用内核实现
"""

import torch
import torch.nn as nn
from packaging import version

from vllm import envs
from vllm._aiter_ops import rocm_aiter_ops
from vllm.config.model import LogprobsMode
from vllm.logger import init_logger
from vllm.platforms import CpuArchEnum, current_platform
from vllm.triton_utils import HAS_TRITON

if HAS_TRITON:
    from vllm.v1.sample.ops.topk_topp_triton import apply_top_k_top_p_triton

logger = init_logger(__name__)


class TopKTopPSampler(nn.Module):
    """
    Top-K / Top-P 采样器模块。

    该模块执行可选的 Top-K 和 Top-P 过滤操作，然后对 logits 进行加权
    随机采样以选择下一个 token。

    注意：实现可能会原地修改 logits 张量。

    属性:
        logprobs_mode: 控制是否以及如何返回 logits/logprobs 信息。
            - "raw_logprobs": 默认模式，不返回处理后的 logits/logprobs
            - "processed_logits": 返回经过 top-k/top-p 处理后的 logits
            - "processed_logprobs": 返回处理后的 logprobs
            当需要返回中间 logits/logprobs 时，FlashInfer 优化不可用。
    """

    def __init__(self, logprobs_mode: LogprobsMode = "raw_logprobs") -> None:
        super().__init__()
        self.logprobs_mode = logprobs_mode
        # FlashInfer 优化不适用于需要返回中间 logprobs/logits 的情况
        # （即经过 top_k/top_p 处理后的结果）
        if (
            logprobs_mode not in ("processed_logits", "processed_logprobs")
            and current_platform.is_cuda()
        ):
            # 根据环境变量和 GPU 能力选择是否使用 FlashInfer 后端
            if envs.VLLM_USE_FLASHINFER_SAMPLER:
                from vllm.v1.attention.backends.flashinfer import FlashInferBackend

                capability = current_platform.get_device_capability()
                assert capability is not None
                if FlashInferBackend.supports_compute_capability(capability):
                    # GPU 支持 FlashInfer，使用优化的 CUDA 实现
                    logger.info_once(
                        "Using FlashInfer for top-p & top-k sampling.",
                        scope="global",
                    )
                    self.forward = self.forward_cuda
                elif envs.is_set("VLLM_USE_FLASHINFER_SAMPLER"):
                    # 用户显式启用了 FlashInfer 但 GPU 不支持，报错
                    capability_str = capability.as_version_str()
                    raise RuntimeError(
                        "FlashInfer does not support compute capability "
                        f"{capability_str}, unset VLLM_USE_FLASHINFER_SAMPLER=1."
                    )
                else:
                    # 默认启用路径：GPU 不支持 FlashInfer 时静默回退到 PyTorch 原生实现
                    # 不会导致服务器启动失败
                    logger.warning_once(
                        "FlashInfer top-p/top-k sampling not supported on "
                        "compute capability %s; falling back to PyTorch-native "
                        "sampler. Set VLLM_USE_FLASHINFER_SAMPLER=0 to silence.",
                        capability.as_version_str(),
                    )
                    self.forward = self.forward_native
            else:
                # 用户显式设置 VLLM_USE_FLASHINFER_SAMPLER=0 禁用 FlashInfer
                logger.info_once(
                    "FlashInfer top-p/top-k sampling disabled via "
                    "VLLM_USE_FLASHINFER_SAMPLER=0; using PyTorch-native sampler."
                )
                self.forward = self.forward_native

        elif current_platform.is_cpu():
            # CPU 平台：根据架构选择实现
            arch = current_platform.get_cpu_architecture()
            # POWERPC 和 RISCV 架构回退到原生实现
            # PowerPC 上 argmax 在 torch.compile 下产生错误输出
            # PR: https://github.com/vllm-project/vllm/pull/26987
            if arch in (CpuArchEnum.RISCV, CpuArchEnum.POWERPC):
                self.forward = self.forward_native
            else:
                self.forward = self.forward_cpu
        elif current_platform.is_xpu():
            # Intel XPU 平台：根据环境变量决定是否使用专用内核
            if envs.VLLM_XPU_USE_SAMPLER_KERNEL:
                self.forward = self.forward_xpu
            else:
                self.forward = self.forward_native
        elif (
            logprobs_mode not in ("processed_logits", "processed_logprobs")
            and rocm_aiter_ops.is_enabled()
        ):
            # ROCm (AMD GPU) 平台：尝试使用 aiter 采样优化库
            try:
                import aiter.ops.sampling  # noqa: F401

                self.aiter_ops = torch.ops.aiter
                logger.info_once(
                    "Using aiter sampler on ROCm (lazy import, sampling-only)."
                )
                self.forward = self.forward_hip
            except ImportError:
                logger.warning_once(
                    "aiter.ops.sampling is not available on ROCm. "
                    "Falling back to forward_native implementation."
                )
                self.forward = self.forward_native
        else:
            # 其他平台：使用 PyTorch 原生实现作为兜底
            self.forward = self.forward_native

    def forward_native(
        self,
        logits: torch.Tensor,
        generators: dict[int, torch.Generator],
        k: torch.Tensor | None,
        p: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        PyTorch 原生的 Top-K / Top-P 采样实现。

        流程：
        1. 应用 Top-K / Top-P 掩码，将不在候选集内的 logits 设为 -inf
        2. 根据 logprobs_mode 决定是否保存处理后的 logits/logprobs
        3. 将 logits 转换为概率分布 (softmax)
        4. 使用随机采样选择下一个 token

        Args:
            logits: [batch_size, vocab_size] 的 logits 张量，可能被原地修改
            generators: 每个请求独立的随机数生成器字典，键为 batch 索引
            k: [batch_size] 的 Top-K 值张量，None 表示不使用 Top-K
            p: [batch_size] 的 Top-P 值张量，None 表示不使用 Top-P

        Returns:
            (next_token_ids, logits_to_return) 元组：
            - next_token_ids: [batch_size] 的采样结果 token ID
            - logits_to_return: 处理后的 logits/logprobs（如果需要），否则为 None
        """
        logits = apply_top_k_top_p(logits, k, p)
        logits_to_return = None
        if self.logprobs_mode == "processed_logits":
            logits_to_return = logits
        elif self.logprobs_mode == "processed_logprobs":
            logits_to_return = logits.log_softmax(dim=-1, dtype=torch.float32)
        probs = logits.softmax(dim=-1, dtype=torch.float32)
        return random_sample(probs, generators), logits_to_return

    def forward_cuda(
        self,
        logits: torch.Tensor,
        generators: dict[int, torch.Generator],
        k: torch.Tensor | None,
        p: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """FlashInfer 优化的 CUDA 采样实现。

        使用 FlashInfer 库进行 Top-K / Top-P 采样，通过拒绝采样避免排序操作，
        比原生 PyTorch 实现更快。

        回退条件：
        - 没有 Top-K/Top-P 过滤时（k 和 p 均为 None）
        - 存在每请求独立的随机数生成器时（FlashInfer 0.2.3+ 不支持）
        """
        # 当 FlashInfer 无需工作（无 top-k / top-p 过滤）或存在每请求生成器时回退
        if (k is None and p is None) or generators:
            if generators:
                logger.debug_once(
                    "FlashInfer 0.2.3+ does not support "
                    "per-request generators. Falling back to "
                    "PyTorch-native implementation."
                )
            return self.forward_native(logits, generators, k, p)
        assert self.logprobs_mode not in ("processed_logits", "processed_logprobs"), (
            "FlashInfer does not support returning logits/logprobs"
        )
        # FlashInfer 采样函数要求 logits 在内存中连续
        # 在 flex_attn/triton_attn fp32 推理中，logits 可能不连续
        # （因为 logits_processor 中的切片操作）
        return flashinfer_sample(logits.contiguous(), k, p, generators), None

    def forward_cpu(
        self,
        logits: torch.Tensor,
        generators: dict[int, torch.Generator],
        k: torch.Tensor | None,
        p: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        CPU 平台专用的 Top-K / Top-P 采样实现。

        与 forward_native 类似，但在无自定义生成器时使用 torch.compile 优化的
        随机采样函数，以获得更好的 CPU 性能。

        Args:
            logits: [batch_size, vocab_size] 的 logits 张量
            generators: 每个请求独立的随机数生成器字典
            k: Top-K 值张量，None 表示不使用
            p: Top-P 值张量，None 表示不使用

        Returns:
            (next_token_ids, logits_to_return) 元组
        """
        logits = apply_top_k_top_p(logits, k, p)
        logits_to_return = None
        if self.logprobs_mode == "processed_logits":
            logits_to_return = logits
        elif self.logprobs_mode == "processed_logprobs":
            logits_to_return = logits.log_softmax(dim=-1, dtype=torch.float32)

        # 当没有为每个请求单独设置生成器时，使用编译优化的随机采样
        if len(generators) != logits.shape[0]:
            return compiled_random_sample(logits), logits_to_return

        # 有自定义生成器时使用 Gumbel-max-trick 方法采样
        probs = logits.softmax(dim=-1, dtype=torch.float32)
        q = torch.empty_like(probs)
        q.exponential_()
        for i, generator in generators.items():
            q[i].exponential_(generator=generator)

        return probs.div_(q).argmax(dim=-1).view(-1), logits_to_return

    def forward_hip(
        self,
        logits: torch.Tensor,
        generators: dict[int, torch.Generator],
        k: torch.Tensor | None,
        p: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """ROCm/aiter 平台优化的采样实现（结构与 forward_cuda 类似）。

        使用 AMD aiter 库进行高效的 Top-K / Top-P 采样。
        """
        if (k is None and p is None) or generators:
            if generators:
                logger.warning_once(
                    "aiter sampler does not support per-request generators; "
                    "falling back to PyTorch-native."
                )
            return self.forward_native(logits, generators, k, p)
        assert self.logprobs_mode not in (
            "processed_logits",
            "processed_logprobs",
        ), "aiter sampler does not support returning logits/logprobs."
        return self.aiter_sample(logits, k, p, generators), None

    def aiter_sample(
        self,
        logits: torch.Tensor,
        k: torch.Tensor | None,
        p: torch.Tensor | None,
        generators: dict[int, torch.Generator],
    ) -> torch.Tensor:
        """使用 aiter 操作库进行采样。

        根据 k 和 p 的组合情况选择不同的采样路径：
        1. 同时有 Top-K 和 Top-P：调用 top_k_top_p_sampling_from_probs
        2. 仅有 Top-P：调用 top_p_sampling_from_probs
        3. 仅有 Top-K：先对概率进行 Top-K 重归一化，再用 torch.multinomial 采样
        """
        use_top_k = k is not None
        use_top_p = p is not None
        # 同时有 Top-K 和 Top-P 的路径
        if use_top_p and use_top_k:
            probs = logits.softmax(dim=-1, dtype=torch.float32).contiguous()
            next_token_ids = self.aiter_ops.top_k_top_p_sampling_from_probs(
                probs,
                None,
                *_to_tensor_scalar_tuple(k),
                *_to_tensor_scalar_tuple(p),
                deterministic=True,
            )
            return next_token_ids.view(-1)
        # 仅有 Top-P 的路径
        elif use_top_p:
            probs = logits.softmax(dim=-1, dtype=torch.float32).contiguous()
            next_token_ids = self.aiter_ops.top_p_sampling_from_probs(
                probs, None, *_to_tensor_scalar_tuple(p), deterministic=True
            )
            return next_token_ids.view(-1)
        # 仅有 Top-K 的路径
        elif use_top_k:
            probs = logits.softmax(dim=-1, dtype=torch.float32).contiguous()
            renorm_probs = self.aiter_ops.top_k_renorm_probs(
                probs, *_to_tensor_scalar_tuple(k)
            )
            return torch.multinomial(renorm_probs, num_samples=1).view(-1)
        raise RuntimeError("aiter_sample was called with no active top-k or top-p.")

    def forward_xpu(
        self,
        logits: torch.Tensor,
        generators: dict[int, torch.Generator],
        k: torch.Tensor | None,
        p: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Intel XPU 平台专用的采样实现。

        使用 vLLM 自定义的 XPU 采样内核，该内核在设备端完成所有采样逻辑，
        包括 Top-K/Top-P 过滤和随机采样。

        注意：XPU 内核不支持每请求独立的生成器。
        """
        if generators:
            logger.warning_once(
                "xpu kernel topk_topp_sampler does not support "
                "per-request generators. Falling back to "
                "PyTorch-native implementation."
            )
            return self.forward_native(logits, generators, k, p)
        random_sampled = torch.empty(
            logits.shape[0], dtype=torch.int64, device=logits.device
        )
        logits_to_return = None
        if (
            self.logprobs_mode == "processed_logits"
            or self.logprobs_mode == "processed_logprobs"
        ):
            logits_to_return = torch.empty_like(logits)

        assert len(generators) != logits.shape[0], (
            "xpu kernel topk_topp_sampler does not support batch-wise generators."
        )
        generator = torch.xpu.default_generators[logits.device.index]

        state = generator.get_state()
        seed, offset = state.view(torch.int64)
        seeds = torch.tensor(
            [seed, offset], dtype=torch.int64, device=torch.device("cpu")
        )
        # XPU 内核期望 k 为 int64 (Long) 类型，但输入 batch 中 top_k 存储为 int32
        # 在此处进行类型转换以避免 dtype 不匹配
        if k is not None:
            k = k.to(torch.int64)
        torch.ops.vllm.xpu_topk_topp_sampler(
            random_sampled, logits_to_return, logits, k, p, self.logprobs_mode, seeds
        )
        # 自定义 XPU 采样内核在内部消费了随机数，因此需要推进默认生成器的偏移量
        # 以保持未来随机数生成的确定性
        # PyTorch 要求：offset 必须是 4 的倍数
        offset = (offset + logits.numel() + 3) // 4 * 4
        state.view(torch.int64)[1] = offset
        generator.set_state(state)
        return random_sampled, logits_to_return


# 注意：这是对 PyTorch 问题的临时解决方案
# 参见: https://github.com/pytorch/pytorch/pull/151218
@torch.compile(dynamic=True)
def compiled_random_sample(logits: torch.Tensor) -> torch.Tensor:
    """
    使用 torch.compile 优化的随机采样函数。

    使用 Gumbel-max-trick 实现高效的多项式采样：
    对于概率分布 probs，argmax(log(probs) - log(E)) 等价于按概率采样，
    其中 E ~ Exp(1)。等价于 probs.div(q).argmax()。

    Args:
        logits: [batch_size, vocab_size] 的 logits 张量

    Returns:
        [batch_size] 的采样结果 token ID
    """
    probs = logits.softmax(dim=-1, dtype=torch.float32)
    q = torch.empty_like(probs)
    q.exponential_()
    return probs.div(q).argmax(dim=-1).view(-1)


def apply_top_k_top_p(
    logits: torch.Tensor, k: torch.Tensor | None, p: torch.Tensor | None
) -> torch.Tensor:
    """
    对 logits 应用 Top-K 和/或 Top-P 掩码。

    根据平台和 batch 大小选择最优的实现：
    - CPU 平台：优先使用 Triton，否则回退到 PyTorch
    - GPU 平台且 batch_size >= 8：使用 Triton 实现（性能更优）
    - GPU 平台且 batch_size < 8：使用 PyTorch sort 实现（避免 Triton 启动开销）

    Args:
        logits: [batch_size, vocab_size] 的 logits 张量
        k: [batch_size] 的 Top-K 值，None 表示不使用 Top-K
        p: [batch_size] 的 Top-P 值，None 表示不使用 Top-P

    Returns:
        经过 Top-K/Top-P 掩码处理后的 logits 张量（可能原地修改）
    """
    if p is None and k is None:
        return logits

    if current_platform.is_cpu():
        if HAS_TRITON:
            return apply_top_k_top_p_triton(logits, k, p)
        return apply_top_k_top_p_pytorch(logits, k, p, allow_cpu_sync=True)

    if HAS_TRITON and logits.shape[0] >= 8:
        return apply_top_k_top_p_triton(logits, k, p)

    # 对小 batch 使用 PyTorch sort 实现
    return apply_top_k_top_p_pytorch(logits, k, p)


def apply_top_k_top_p_pytorch(
    logits: torch.Tensor,
    k: torch.Tensor | None,
    p: torch.Tensor | None,
    allow_cpu_sync: bool = False,
) -> torch.Tensor:
    """使用 PyTorch sort 实现 Top-K 和 Top-P 掩码。

    算法流程：
    1. 对 logits 按升序排序（获得排序后的 logits 和对应的原始索引）
    2. 应用 Top-K 掩码：将排名在 K 之后的 token 设为 -inf
    3. 应用 Top-P 掩码：计算累积概率，将超出阈值的 token 设为 -inf
    4. 使用 scatter 将排序后的结果恢复到原始顺序

    注意：如果使用 Top-P，此函数会对 logits 排序，对于大 batch 可能较慢。
    logits 张量可能被原地修改。
    """
    if p is None:
        if k is None:
            return logits

        if allow_cpu_sync:
            # 仅 Top-K 时避免对整个词表排序，使用更高效的实现
            return apply_top_k_only(logits, k)

    # 对 logits 沿最后一个维度升序排序
    logits_sort, logits_idx = logits.sort(dim=-1, descending=False)

    if k is not None:
        # 应用 Top-K 掩码
        # top_k_mask 计算每个 batch 元素中第 k 大的值的位置
        top_k_mask = logits_sort.size(1) - k.to(torch.long)  # shape: B
        # 获取第 k 大的 logits 值作为阈值
        top_k_mask = logits_sort.gather(1, top_k_mask.unsqueeze(dim=1))
        # 将小于该阈值的 logits 设为 -inf
        top_k_mask = logits_sort < top_k_mask
        logits_sort.masked_fill_(top_k_mask, -float("inf"))

    if p is not None:
        # 应用 Top-P 掩码
        # 先将 logits 转换为概率，再计算累积和
        probs_sort = logits_sort.softmax(dim=-1)
        probs_sum = torch.cumsum(probs_sort, dim=-1, out=probs_sort)
        # 累积概率 <= 1-p 的 token 被掩码掉（保留累积概率刚好超过 p 的最小集合）
        top_p_mask = probs_sum <= 1 - p.unsqueeze(dim=1)
        # 至少保留一个 token（最后一个不掩码）
        top_p_mask[:, -1] = False
        logits_sort.masked_fill_(top_p_mask, -float("inf"))

    # 使用 scatter 将排序后的 logits 恢复到原始顺序
    return logits.scatter_(dim=-1, index=logits_idx, src=logits_sort)


def apply_top_k_only(logits: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """
    仅对 logits 应用 Top-K 掩码（不使用 Top-P）。

    此实现不涉及对整个词表排序，使用 torch.topk 获取前 k 个最大值。
    但需要注意：它涉及 GPU->CPU 同步（获取 max_top_k 值），
    可能对异步调度性能产生不利影响。

    Args:
        logits: [batch_size, vocab_size] 的 logits 张量
        k: [batch_size] 的 Top-K 值

    Returns:
        经过 Top-K 掩码处理后的 logits 张量（原地修改）
    """
    # 标记不需要 Top-K 过滤的行（k == vocab_size 表示保留全部）
    no_top_k_mask = k == logits.shape[1]
    # 将不需要 Top-K 的行的 k 设为 1，以便后续 gather 操作正常执行
    k = k.masked_fill(no_top_k_mask, 1)
    max_top_k = k.max()
    # topk.values 张量形状为 [batch_size, max_top_k]
    # 将 k 转换为 0-based 索引（范围 [0, max_top_k)）
    k_index = k.sub_(1).unsqueeze(1)
    # 获取每个 batch 中第 k 大的 logits 值作为阈值
    top_k_mask = logits.topk(max_top_k, dim=1).values.gather(1, k_index.long())
    # 处理不需要 Top-K 的行，设为 -inf 使其不会被掩码
    top_k_mask.masked_fill_(no_top_k_mask.unsqueeze(1), -float("inf"))
    # 将小于阈值的 logits 设为 -inf
    return logits.masked_fill_(logits < top_k_mask, -float("inf"))


def random_sample(
    probs: torch.Tensor,
    generators: dict[int, torch.Generator],
) -> torch.Tensor:
    """从概率分布中随机采样。

    使用 Gumbel-max-trick 替代 torch.multinomial，因为 torch.multinomial
    会导致 CPU-GPU 同步。

    技术原理：对于概率分布 probs，argmax(log(probs) - log(E)) 等价于
    按概率多项式采样，其中 E ~ Exp(1)。即 probs.div(q).argmax()，
    其中 q 服从指数分布。

    Args:
        probs: [batch_size, vocab_size] 的概率分布张量
        generators: 每个请求独立的随机数生成器字典，键为 batch 索引

    Returns:
        [batch_size] 的采样结果 token ID
    """
    q = torch.empty_like(probs)
    # 注意(woosuk): 为了批量处理没有自定义种子的请求（大多数情况），
    # 我们首先假设所有请求都没有自己的种子，然后为有自定义种子的请求覆盖。
    if len(generators) != probs.shape[0]:
        q.exponential_()
    if generators:
        # TODO(woosuk): 这里逐个处理请求可能较慢，需要优化
        for i, generator in generators.items():
            q[i].exponential_(generator=generator)
    return probs.div_(q).argmax(dim=-1).view(-1)


def flashinfer_sample(
    logits: torch.Tensor,
    k: torch.Tensor | None,
    p: torch.Tensor | None,
    generators: dict[int, torch.Generator],
) -> torch.Tensor:
    """使用 FlashInfer 库进行采样。

    统计上等价于 random_sample 函数，但通过拒绝采样避免了对 logits 排序，
    因此更快。

    注意：此函数的输出不一定与 random_sample 函数完全相同，
    仅保证统计等价性（即相同分布下的独立采样）。

    Args:
        logits: [batch_size, vocab_size] 的 logits 张量（要求内存连续）
        k: [batch_size] 的 Top-K 值，None 表示不使用
        p: [batch_size] 的 Top-P 值，None 表示不使用
        generators: 每个请求独立的随机数生成器字典（当前未使用）

    Returns:
        [batch_size] 的采样结果 token ID
    """
    import flashinfer

    if version.parse(flashinfer.__version__) < version.parse("0.2.3"):
        raise ImportError(
            "FlashInfer version >= 0.2.3 required for top-k and top-p sampling. "
        )

    assert not (k is None and p is None)
    if k is None:
        # 仅有 Top-P 的路径
        probs = logits.softmax(dim=-1, dtype=torch.float32)
        next_token_ids = flashinfer.sampling.top_p_sampling_from_probs(
            probs, p, deterministic=True
        )
    elif p is None:
        # 仅有 Top-K 的路径
        probs = logits.softmax(dim=-1, dtype=torch.float32)
        next_token_ids = flashinfer.sampling.top_k_sampling_from_probs(
            probs, k, deterministic=True
        )
    else:
        # 同时有 Top-K 和 Top-P 的路径
        next_token_ids = flashinfer.sampling.top_k_top_p_sampling_from_logits(
            logits, k, p, deterministic=True
        )

    return next_token_ids.view(-1)


def _to_tensor_scalar_tuple(x):
    """
    将参数转换为 (tensor, scalar) 元组格式，以适配 aiter 操作的接口要求。

    如果输入已经是 Tensor，则返回 (tensor, 0)；
    如果输入是标量，则返回 (None, scalar)。

    Args:
        x: Tensor 或标量值

    Returns:
        (tensor, scalar) 元组
    """
    if isinstance(x, torch.Tensor):
        return (x, 0)
    else:
        return (None, x)
