# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# =============================================================================
# 中文注释 - 模块功能概述：
# 本模块实现了微批次（micro-batch）的同步调度机制，也称为"双缓冲"（Double Buffering, DBO）。
# 核心思想是：将一个大批次拆分为多个微批次（默认 2 个），利用多个 CUDA stream
# 计算与通信（如 AllReduce、AllGather 等集合通信）可以重叠执行的特性，
# 让一个微批次在 compute stream 上做计算的同时，另一个微批次在 comm stream 上做通信，
# 从而隐藏通信延迟，提升 GPU 利用率和整体吞吐量。
#
# 同步机制设计：
# - 每个微批次运行在独立的 Python 线程中，通过 threading.Event 进行 CPU 端的交替执行控制。
# - 任意时刻只有一个线程在运行（保证正确性），通过 yield 机制让出 CPU 给另一个微批次。
# - GPU 端通过 torch.Event（CUDA event）实现跨 stream 的依赖管理：
#   compute_done_event 确保通信开始前计算完成，comm_done_event 确保计算开始前通信完成。
#
# 典型使用场景：
# - 张量并行（Tensor Parallelism）下，每层 Transformer 的 AllReduce 通信
#   可以与下一层的计算重叠。
# - MoE 模型中 Expert Parallelism 的 AllToAll 通信与计算重叠。
# =============================================================================

import threading

import torch

from vllm import forward_context
from vllm.forward_context import ForwardContext
from vllm.logger import init_logger
from vllm.utils.torch_utils import current_stream

logger = init_logger(__name__)

# 中文注释：线程 ID 到微批次上下文 ID 的映射表。
# 每个微批次线程启动后会在此注册自己的线程 ID，用于后续通过当前线程 ID 查找对应的上下文。
_THREAD_ID_TO_CONTEXT: dict = {}
# Here we hardcode the number of microbatches to 2 for default.
# 中文注释：默认的微批次数量，设为 2 即"双缓冲"模式。
_NUM_UBATCHES: int = 2
# 中文注释：全局上下文列表，按微批次 ID 索引。
# 线程通过 _THREAD_ID_TO_CONTEXT 查到自己的 ID 后，再从此列表获取对应的 UBatchContext。
_CURRENT_CONTEXTS: list["UBatchContext | None"] = []


class UBatchContext:
    """
    Context manager for micro-batching synchronization using threading events.
    """

    # 中文注释：UBatchContext 是微批次同步的核心上下文管理器。
    # 每个微批次拥有一个独立的 UBatchContext 实例，管理该微批次的：
    #   1. CUDA stream（计算流和通信流）
    #   2. CPU 端线程同步事件（用于控制微批次之间的 CPU 交替执行）
    #   3. GPU 端 CUDA 事件（用于跨 stream 的依赖同步）
    #   4. 前向传播上下文（ForwardContext，包含当前 batch 的元数据）
    #
    # 使用方式：作为上下文管理器（with 语句）使用，
    # 进入时注册线程并等待所有微批次就绪，退出时通知下一个微批次可以运行。

    def __init__(
        self,
        id: int,
        comm_stream: torch.cuda.Stream,
        compute_stream: torch.cuda.Stream,
        forward_context: ForwardContext,
        ready_barrier: threading.Barrier,
        cpu_wait_event: threading.Event,
        cpu_signal_event: threading.Event,
        gpu_comm_done_event: torch.Event,
        gpu_compute_done_event: torch.Event,
        schedule: str = "default",
    ):
        # 中文注释：微批次的唯一标识（0, 1, 2, ...）。
        self.id = id
        # 中文注释：通信流，用于执行 AllReduce/AllGather/AllToAll 等集合通信操作。
        self.comm_stream = comm_stream
        # 中文注释：计算流，用于执行矩阵乘法、attention 等计算操作。
        self.compute_stream = compute_stream
        # 中文注释：该微批次对应的前向传播上下文，包含 batch 中所有请求的元数据。
        self.forward_context = forward_context
        # 中文注释：所有微批次线程的就绪屏障。所有线程都到达此屏障后才一起开始执行，
        # 确保所有微批次的上下文都已正确初始化。
        self.ready_barrier = ready_barrier
        # 中文注释：当前微批次的 CPU 等待事件。当前线程会在此事件上阻塞，
        # 直到上一个微批次发出信号表示它已让出执行权。
        self.cpu_wait_event = cpu_wait_event
        # 中文注释：当前微批次的 CPU 信号事件。当前线程完成工作后设置此事件，
        # 唤醒下一个微批次的线程开始执行。注意：cpu_signal_event 指向的是下一个微批次的等待事件。
        self.cpu_signal_event = cpu_signal_event
        # 中文注释：当前活跃的 CUDA stream，默认为计算流。
        self.current_stream = compute_stream
        # 中文注释：GPU 端通信完成事件，记录在 comm_stream 上，用于通知 compute_stream 通信已完成。
        self.gpu_comm_done_event = gpu_comm_done_event
        # 中文注释：GPU 端计算完成事件，记录在 compute_stream 上，用于通知 comm_stream 计算已完成。
        self.gpu_compute_done_event = gpu_compute_done_event
        # 中文注释：调度策略，默认为 "default"，可扩展为其他调度模式。
        self.schedule = schedule
        # 中文注释：可选的接收钩子函数，用于在微批次切换时执行延迟的通信接收操作。
        # 通常在上一个微批次退出时注册，在下一个微批次进入时执行。
        self.recv_hook = None

    def __enter__(self):
        # 中文注释：进入上下文管理器时的初始化流程：
        # 步骤 1：将当前线程 ID 注册到全局映射表，后续可通过线程 ID 查找此微批次。
        # 步骤 2：将自身放入全局上下文列表的对应位置。
        # 步骤 3：等待所有微批次线程都就绪（barrier 同步），确保所有 ForwardContext 都已准备好。
        # 步骤 4：等待 CPU 信号，只有当上一个微批次发出信号后才真正开始执行。
        #         这保证了任意时刻只有一个微批次线程在运行。
        # 步骤 5：恢复全局前向传播上下文（因为上一个微批次退出时可能覆盖了它）。
        # 步骤 6：切换到计算流，准备开始计算。
        global _CURRENT_CONTEXTS, _THREAD_ID_TO_CONTEXT
        _THREAD_ID_TO_CONTEXT[threading.get_ident()] = self.id
        _CURRENT_CONTEXTS[self.id] = self
        # _NUM_UBATCHES is set in make_ubatch_contexts
        self.ready_barrier.wait()

        self.cpu_wait_event.wait()
        self.cpu_wait_event.clear()
        self._restore_context()
        # Assume we want to start on the compute stream
        self.update_stream(self.compute_stream)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 中文注释：退出上下文管理器时的清理流程：
        # 步骤 1：从全局上下文列表中移除自身。
        # 步骤 2：从线程映射表中删除当前线程的注册。
        # 步骤 3：执行可能存在的接收钩子（如延迟的通信接收操作）。
        # 步骤 4：设置 cpu_signal_event，唤醒下一个微批次的线程开始执行。
        #         这是微批次交替执行的关键——当前微批次完成后主动让出 CPU。
        # 步骤 5：清除自身的等待事件（为下一轮迭代做准备）。
        global _CURRENT_CONTEXTS, _THREAD_ID_TO_CONTEXT
        _CURRENT_CONTEXTS[self.id] = None
        del _THREAD_ID_TO_CONTEXT[threading.get_ident()]
        self.maybe_run_recv_hook()
        self.cpu_signal_event.set()
        self.cpu_wait_event.clear()
        return False

    def _restore_context(self):
        # 中文注释：恢复全局前向传播上下文。
        # 为什么需要恢复：因为多个微批次线程共享同一个进程空间，
        # 而 forward_context._forward_context 是模块级全局变量，
        # 上一个微批次退出时会将其设为自己的上下文，所以当前微批次进入时需要重新设置。
        forward_context._forward_context = self.forward_context

    def update_stream(self, stream):
        # 中文注释：切换当前微批次活跃的 CUDA stream。
        # 如果目标 stream 与当前 CUDA 设备的 stream 不一致，则调用 set_stream 切换。
        # 这确保后续的 CUDA 操作（kernel launch、memory copy 等）都在正确的 stream 上执行。
        self.current_stream = stream
        if current_stream() != self.current_stream:
            torch.cuda.set_stream(self.current_stream)

    def _signal_comm_done(self):
        # 中文注释：在通信流上记录一个 CUDA 事件，表示通信操作已完成。
        # 其他 stream 可以通过 wait_event 等待此事件来确保通信完成后再继续。
        self.gpu_comm_done_event.record(self.comm_stream)

    def _signal_compute_done(self):
        # 中文注释：在计算流上记录一个 CUDA 事件，表示计算操作已完成。
        # 通信流可以等待此事件，确保计算完成后再开始通信（如 AllReduce 的输入已准备好）。
        self.gpu_compute_done_event.record(self.compute_stream)

    def _wait_compute_done(self):
        # 中文注释：让通信流等待计算完成。确保 compute_stream 上的所有操作执行完毕后，
        # comm_stream 上的操作才开始。这是 stream 间依赖管理的标准模式。
        self.comm_stream.wait_event(self.gpu_compute_done_event)

    def _wait_comm_done(self):
        # 中文注释：让计算流等待通信完成。确保 comm_stream 上的所有操作执行完毕后，
        # compute_stream 上的操作才开始。
        self.compute_stream.wait_event(self.gpu_comm_done_event)

    def _cpu_yield(self):
        # 中文注释：CPU 端的让出操作，这是微批次交替执行的核心机制。
        # 设计原则：任意时刻只能有一个微批次线程在运行，通过事件机制实现"乒乓"交替。
        #
        # 流程：
        # 步骤 1：断言检查——确保当前线程是唯一活跃的线程（正确性保障）。
        # 步骤 2：设置 cpu_signal_event，唤醒下一个微批次的线程。
        # 步骤 3：在 cpu_wait_event 上阻塞，等待上一个微批次完成后唤醒自己。
        # 步骤 4：被唤醒后清除等待事件，恢复全局前向传播上下文。
        #
        # 注意：GPU 上的操作可能仍在异步执行（通过 CUDA event 保证跨 stream 依赖），
        # 这里只控制 CPU 端的交替，不阻塞 GPU 的流水线。
        # It is critical for correctness that only one thread is running
        # at a time. These asserts just make sure that this is the only
        # thread running before waking the other one up and going to sleep
        assert forward_context._forward_context == self.forward_context
        assert current_stream() == self.current_stream
        assert not self.cpu_wait_event.is_set()

        self.cpu_signal_event.set()
        self.cpu_wait_event.wait()
        self.cpu_wait_event.clear()
        self._restore_context()

    def switch_to_comm(self):
        # 中文注释：切换到通信流（无同步）。CPU 立即切换 stream，不做 GPU 端等待。
        self.update_stream(self.comm_stream)

    def switch_to_compute(self):
        # 中文注释：切换到计算流（无同步）。CPU 立即切换 stream，不做 GPU 端等待。
        self.update_stream(self.compute_stream)

    def switch_to_comm_sync(self):
        # 中文注释：同步地从计算流切换到通信流。
        # 步骤 1：在计算流上记录计算完成事件。
        # 步骤 2：切换到通信流。
        # 步骤 3：让通信流等待计算完成事件，确保计算完成后再开始通信。
        self._signal_compute_done()
        self.update_stream(self.comm_stream)
        self._wait_compute_done()

    def switch_to_compute_sync(self):
        # 中文注释：同步地从通信流切换到计算流。
        # 步骤 1：在通信流上记录通信完成事件。
        # 步骤 2：切换到计算流。
        # 步骤 3：让计算流等待通信完成事件，确保通信完成后再开始计算。
        self._signal_comm_done()
        self.update_stream(self.compute_stream)
        self._wait_comm_done()

    def maybe_run_recv_hook(self):
        # 中文注释：执行延迟的接收钩子（如果存在）。
        # 接收钩子通常由 dbo_register_recv_hook 注册，用于在微批次切换时
        # 触发异步通信接收操作。执行后清除钩子，避免重复执行。
        if self.recv_hook is not None:
            self.recv_hook()
            self.recv_hook = None

    def yield_(self):
        # 中文注释：简单的 CPU 让出操作，不做 stream 切换。
        # 保存当前 stream -> 让出 CPU -> 被唤醒后恢复 stream。
        # 适用于：当前微批次完成某阶段工作，需要让另一个微批次运行，
        # 但不需要切换 stream 的场景。
        self.current_stream = current_stream()
        self._cpu_yield()
        self.update_stream(self.current_stream)

    def yield_and_switch_from_compute_to_comm(self):
        # 中文注释：从计算流让出 CPU 并切换到通信流。
        # 典型场景：微批次完成了一层 Transformer 的计算，需要让出 CPU 给另一个微批次，
        # 回来后要执行 AllReduce 通信操作。
        #
        # 流程：
        # 步骤 1：断言当前在计算流上。
        # 步骤 2：在计算流上记录计算完成事件（让通信流知道计算已完成）。
        # 步骤 3：CPU 让出，等待另一个微批次完成。
        # 步骤 4：被唤醒后，断言 current_stream 仍为计算流。
        # 步骤 5：切换到通信流。
        # 步骤 6：通信流等待计算完成事件（GPU 端同步）。
        assert current_stream() == self.compute_stream
        self._signal_compute_done()
        self._cpu_yield()
        assert self.current_stream == self.compute_stream
        self.update_stream(self.comm_stream)
        self._wait_compute_done()

    def yield_and_switch_from_comm_to_compute(self):
        # 中文注释：从通信流让出 CPU 并切换到计算流。
        # 典型场景：微批次完成了 AllReduce 通信，需要让出 CPU 给另一个微批次，
        # 回来后要执行下一层的计算。
        #
        # 流程与 yield_and_switch_from_compute_to_comm 对称：
        # 步骤 1：断言当前在通信流上。
        # 步骤 2：在通信流上记录通信完成事件。
        # 步骤 3：CPU 让出。
        # 步骤 4：被唤醒后切换到计算流。
        # 步骤 5：计算流等待通信完成事件。
        assert current_stream() == self.comm_stream
        self._signal_comm_done()
        self._cpu_yield()
        assert self.current_stream == self.comm_stream
        self.update_stream(self.compute_stream)
        self._wait_comm_done()


def dbo_enabled() -> bool:
    # 中文注释：检查当前是否启用了双缓冲（DBO）模式。
    # 如果有微批次线程注册过（_THREAD_ID_TO_CONTEXT 非空），则 DBO 已启用。
    # 用于在模型执行路径中判断是否需要进行微批次同步操作。
    return len(_THREAD_ID_TO_CONTEXT) > 0


def dbo_current_ubatch_id() -> int:
    # 中文注释：获取当前线程对应的微批次 ID。
    # 如果 DBO 未启用，返回 0（单微批次模式，ID 固定为 0）。
    # 如果 DBO 已启用，通过线程 ID 查找对应的微批次 ID。
    if len(_THREAD_ID_TO_CONTEXT) == 0:
        return 0
    return _THREAD_ID_TO_CONTEXT[threading.get_ident()]


def _register_ubatch_function(func):
    # 中文注释：将 UBatchContext 的实例方法包装为全局函数。
    # 为什么需要包装：模型执行代码（如 attention、MLP 层）不方便直接持有 UBatchContext 引用，
    # 但可以通过调用全局函数（如 dbo_yield()）来间接操作当前线程的上下文。
    # 包装后的函数会自动查找当前线程的上下文，然后调用对应的方法。
    # 如果 DBO 未启用（_THREAD_ID_TO_CONTEXT 为空），则什么都不做（no-op）。
    def wrapper(*args, **kwargs):
        if len(_THREAD_ID_TO_CONTEXT) > 0:
            ctx_idx = _THREAD_ID_TO_CONTEXT[threading.get_ident()]
            ctx = _CURRENT_CONTEXTS[ctx_idx]
            func(ctx, *args, **kwargs)

    return wrapper


# 中文注释：以下是对 UBatchContext 方法的全局函数包装。
# 这些函数在模型执行路径中被调用，用于控制微批次的同步和 stream 切换。
# 当 DBO 未启用时，这些函数都是 no-op，不影响正常（非 DBO）的执行路径。

dbo_maybe_run_recv_hook = _register_ubatch_function(UBatchContext.maybe_run_recv_hook)
dbo_yield = _register_ubatch_function(UBatchContext.yield_)
dbo_yield_and_switch_from_compute_to_comm = _register_ubatch_function(
    UBatchContext.yield_and_switch_from_compute_to_comm
)
dbo_yield_and_switch_from_comm_to_compute = _register_ubatch_function(
    UBatchContext.yield_and_switch_from_comm_to_compute
)
dbo_switch_to_comm = _register_ubatch_function(UBatchContext.switch_to_comm)
dbo_switch_to_compute = _register_ubatch_function(UBatchContext.switch_to_compute)
dbo_switch_to_comm_sync = _register_ubatch_function(UBatchContext.switch_to_comm_sync)
dbo_switch_to_compute_sync = _register_ubatch_function(
    UBatchContext.switch_to_compute_sync
)


def dbo_register_recv_hook(recv_hook):
    # 中文注释：为下一个微批次注册接收钩子。
    # 为什么注册到"下一个"微批次：当前微批次在注册钩子后会退出（yield），
    # 下一个微批次进入时会执行此钩子（通过 maybe_run_recv_hook）。
    # 这实现了跨微批次的延迟通信接收——当前微批次发出请求，
    # 但实际接收操作延迟到下一个微批次的上下文中执行。
    # 使用环形索引：(ctx_idx + 1) % _NUM_UBATCHES 实现循环。
    if len(_THREAD_ID_TO_CONTEXT) > 0:
        ctx_idx = _THREAD_ID_TO_CONTEXT[threading.get_ident()]
        next_ctx = _CURRENT_CONTEXTS[(ctx_idx + 1) % _NUM_UBATCHES]
        next_ctx.recv_hook = recv_hook


def dbo_get_previous_event(func, *args, **kwargs):
    # 中文注释：在当前微批次的计算流上执行一个可调用对象（通常是 CUDA 操作）。
    # 用途：在微批次的计算流上记录或等待事件，用于跨微批次的 GPU 端同步。
    # 通过 torch.cuda.stream 上下文管理器确保操作在正确的 stream 上执行。
    if len(_THREAD_ID_TO_CONTEXT) > 0:
        ctx_idx = _THREAD_ID_TO_CONTEXT[threading.get_ident()]
        ctx = _CURRENT_CONTEXTS[ctx_idx]
        # execute callable on the ubatch compute stream to record/wait events there
        with torch.cuda.stream(ctx.compute_stream):
            return func(*args, **kwargs)


def make_ubatch_contexts(
    num_micro_batches: int,
    compute_stream: torch.cuda.Stream,
    comm_stream: torch.cuda.Stream,
    forward_contexts: list[ForwardContext],
    ready_barrier: threading.Barrier,
    schedule: str = "default",
) -> list[UBatchContext]:
    # 中文注释：工厂函数，创建所有微批次的 UBatchContext 实例。
    #
    # 参数说明：
    #   num_micro_batches: 微批次数量（必须 > 1，否则没有重叠的意义）。
    #   compute_stream: 所有微批次共享的计算 CUDA stream。
    #   comm_stream: 所有微批次共享的通信 CUDA stream。
    #   forward_contexts: 每个微批次对应的前向传播上下文列表。
    #   ready_barrier: 所有微批次线程的就绪屏障，确保同步启动。
    #   schedule: 调度策略名称。
    #
    # 关键设计：
    #   - CPU 事件采用环形连接：微批次 i 的 cpu_signal_event 指向微批次 (i+1) % N 的 cpu_wait_event。
    #     这样微批次 i 退出时唤醒微批次 (i+1)，形成循环交替执行的模式。
    #   - GPU 事件：每个微批次独立拥有 comm_done 和 compute_done 事件，
    #     用于在同一微批次的 compute stream 和 comm stream 之间建立依赖。
    global _NUM_UBATCHES, _CURRENT_CONTEXTS
    assert num_micro_batches > 1, "num_micro_batches must be greater than 1"

    _NUM_UBATCHES = num_micro_batches
    # Ensure the global context list is large enough
    if len(_CURRENT_CONTEXTS) < num_micro_batches:
        _CURRENT_CONTEXTS.extend([None] * (num_micro_batches - len(_CURRENT_CONTEXTS)))

    """
    Create a context manager for micro-batching synchronization.
    """
    # 中文注释：创建 CPU 端的同步事件和 GPU 端的 CUDA 事件。
    cpu_events = [threading.Event() for _ in range(num_micro_batches)]
    gpu_comm_done_events = [torch.Event() for _ in range(num_micro_batches)]
    gpu_compute_done_events = [torch.Event() for _ in range(num_micro_batches)]

    ctxs = []
    for i in range(num_micro_batches):
        # 中文注释：为每个微批次创建上下文，核心是 CPU 事件的环形连接：
        # 微批次 i 的 cpu_wait_event = cpu_events[i]（自己等待的事件）
        # 微批次 i 的 cpu_signal_event = cpu_events[(i+1) % N]（唤醒下一个微批次的事件）
        # 这样形成 0 -> 1 -> 0 -> 1 -> ... 的交替执行链。
        ctx = UBatchContext(
            id=i,
            compute_stream=compute_stream,
            comm_stream=comm_stream,
            forward_context=forward_contexts[i],
            ready_barrier=ready_barrier,
            cpu_wait_event=cpu_events[i],
            cpu_signal_event=cpu_events[(i + 1) % num_micro_batches],
            gpu_comm_done_event=gpu_comm_done_events[i],
            gpu_compute_done_event=gpu_compute_done_events[i],
            schedule=schedule,
        )
        ctxs.append(ctx)

    return ctxs
