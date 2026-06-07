# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# =============================================================================
# workspace.py -- GPU 工作空间（Workspace）管理模块
# =============================================================================
# 中文注释：本模块负责管理模型推理过程中所需的 GPU 临时工作空间。
#
# 背景说明：
#   模型前向传播（forward pass）中，许多算子（如 MoE 的专家路由、
#   attention 的中间缓冲区等）需要临时的 GPU 张量作为工作缓冲区。
#   每次推理都动态分配/释放这些缓冲区会产生显著的 CUDA 分配开销。
#
# 解决方案：
#   采用"预分配 + 懒扩展 + 锁定"的策略：
#   1. 预分配（pre-allocation）：在模型初始化/warmup 阶段，根据实际需求
#      预分配一块足够大的 GPU 缓冲区作为工作空间。
#   2. 懒扩展（lazy grow）：如果后续需要更大的空间，自动扩展工作空间。
#   3. 锁定（lock）：warmup 完成后锁定工作空间大小，确保热路径中不会
#      触发意外的 GPU 内存分配，保证推理性能稳定。
#
# UBatch 支持：
#   工作空间支持多个 ubatch slot（微批次槽位），每个 ubatch 拥有独立的
#   工作空间，以支持 DBO（Disaggregated Batch Overlap）等高级调度策略。
#
# 全局单例模式：
#   使用模块级全局变量 _manager 管理唯一的 WorkspaceManager 实例，
#   通过 init_workspace_manager() 初始化，其他模块通过
#   current_workspace_manager() 获取实例。
# =============================================================================

import inspect
import os
from itertools import accumulate
from math import prod

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.utils.math_utils import round_up
from vllm.v1.worker.ubatching import dbo_current_ubatch_id

logger = init_logger(__name__)


def _compute_bytes(shape: tuple[int, ...], dtype: torch.dtype) -> int:
    """计算给定 shape 和 dtype 的张量所占用的字节数。"""
    return prod(shape) * dtype.itemsize


# Constants
_MB = 1024**2
_GiB = 1024**3

# 中文注释：全局工作空间管理器实例，采用模块级单例模式。
# 通过 init_workspace_manager() 初始化，通过 current_workspace_manager() 获取。
_manager: "WorkspaceManager | None" = None


# 中文注释：WorkspaceManager 是工作空间管理器的核心类。
#
# 核心职责：
#   1. 为每个 ubatch slot 维护一个 GPU 张量作为工作空间缓冲区。
#   2. 提供 get_simultaneous() 方法，在单次分配中获取多个张量视图（view），
#      避免多次独立分配的开销。
#   3. 支持"锁定"机制：锁定后禁止扩展工作空间，确保推理热路径中无意外分配。
#
# 设计思路：
#   工作空间以原始字节（uint8）形式分配，然后通过 view + reshape 切分出
#   不同形状和类型的张量子视图。这种"一次分配、多次视图"的方式比
#   多次独立分配更高效，且所有子视图共享同一块物理显存。
class WorkspaceManager:
    """Manager for workspace allocation.

    Manages one workspace buffer per active ubatch slot.
    Can be locked to prevent further growth during execution.
    """

    def __init__(self, device: torch.device, num_ubatches: int | None = None):
        # 中文注释：初始化工作空间管理器。
        # device: 工作空间分配所在的 GPU 设备。
        # num_ubatches: ubatch 槽位数量，默认为 1（不使用 DBO 时只需 1 个）。
        #   当启用 DBO（Disaggregated Batch Overlap）时，可能需要多个 ubatch slot，
        #   每个 slot 有独立的工作空间以避免数据竞争。
        self._device = device
        # Cache num ubatches at init based on configuration (default to 1)
        self._num_ubatches = num_ubatches if num_ubatches is not None else 1
        # 中文注释：当前各 ubatch slot 的工作空间张量列表。
        # 初始均为 None，在首次使用时懒分配（lazy allocation）。
        self._current_workspaces: list[torch.Tensor | None] = [
            None
        ] * self._num_ubatches
        # 中文注释：锁定标志。锁定后禁止扩展工作空间大小，
        # 任何尝试分配更大空间的操作都会触发 AssertionError。
        self._locked: bool = False

    @staticmethod
    def _workspace_size_bytes(workspace: torch.Tensor | None) -> int:
        """Get size of workspace in bytes."""
        # 中文注释：获取工作空间张量的字节大小。
        # 如果工作空间尚未分配（None），返回 0。
        if workspace is None:
            return 0
        return workspace.numel() * workspace.element_size()

    def lock(self) -> None:
        """Lock the workspace to prevent further growth.

        After locking, any attempt to allocate a larger workspace will raise
        an assertion error. This ensures workspace size is fixed during execution.
        """
        # 中文注释：锁定工作空间。
        # 通常在模型 warmup/profiling 完成后调用。锁定后，工作空间大小固定不变，
        # 这样在实际推理的热路径中不会触发任何 GPU 内存分配操作，
        # 从而保证推理延迟的稳定性和可预测性。
        self._locked = True
        if envs.VLLM_DEBUG_WORKSPACE:
            logger.info(
                "[WORKSPACE DEBUG] Workspace locked. Current sizes: %s",
                [
                    self._workspace_size_bytes(ws) / _MB
                    for ws in self._current_workspaces
                    if ws is not None
                ],
            )

    def unlock(self) -> None:
        """Unlock the workspace to allow growth.

        This is used during elastic EP scaling when the workspace size
        needs to grow due to changes in the number of experts.
        """
        # 中文注释：解锁工作空间，允许其扩展。
        # 使用场景：弹性专家并行（Elastic EP）缩放时，专家数量发生变化，
        # 工作空间需要扩大以容纳更多的专家中间结果。缩放完成后应重新锁定。
        self._locked = False
        if envs.VLLM_DEBUG_WORKSPACE:
            logger.info(
                "[WORKSPACE DEBUG] Workspace unlocked. Current sizes: %s",
                [
                    self._workspace_size_bytes(ws) / _MB
                    for ws in self._current_workspaces
                    if ws is not None
                ],
            )

    def is_locked(self) -> bool:
        """Check if workspace is locked."""
        return self._locked

    # 中文注释：get_simultaneous 是工作空间管理器最核心的公开方法。
    #
    # 功能：从单次分配的工作空间中同时获取多个不同形状/类型的张量视图。
    #
    # 工作流程（分步骤说明）：
    #   步骤 1：计算每个请求的张量所需的原始字节数。
    #   步骤 2：将每个张量的字节数向上对齐到 256 字节边界。
    #           对齐的原因：GPU 内存访问通常要求地址对齐，256 字节对齐
    #           可以确保后续的 view/reshape 操作不会因为地址未对齐而出错，
    #           同时也有利于 GPU 的内存访问效率。
    #   步骤 3：计算所有张量的总字节数，确保工作空间足够大。
    #   步骤 4：计算每个张量在工作空间中的起始偏移量（cumulative offset）。
    #   步骤 5：调用 _ensure_workspace_size() 确保工作空间已分配且足够大。
    #   步骤 6：通过切片 + view + reshape 从工作空间中创建各张量的视图。
    #           这些视图共享同一块物理显存，零额外分配开销。
    #
    # 返回：一个 torch.Tensor 列表，每个元素对应一个请求的张量视图。
    def get_simultaneous(
        self, *shapes_and_dtypes: tuple[tuple[int, ...], torch.dtype]
    ) -> list[torch.Tensor]:
        """Get multiple workspace tensors simultaneously from a single allocation.

        Args:
            *shapes_and_dtypes: One or more (shape, dtype) tuples.

        Returns:
            List of tensor views into the workspace buffer, one per shape/dtype pair.
        """
        # 步骤 1：计算每个张量的实际字节数
        actual_bytes = [_compute_bytes(s, d) for s, d in shapes_and_dtypes]
        # 步骤 2：向上对齐到 256 字节边界，保证 GPU 内存访问对齐
        aligned_bytes = [round_up(actual, 256) for actual in actual_bytes]
        # 步骤 3：计算总需求字节数
        total_bytes = sum(aligned_bytes)

        # Calculate cumulative offsets using itertools.accumulate
        # 步骤 4：计算累积偏移量，用于确定每个张量在工作空间中的起始位置
        # 例如：[0, 256, 512, ...] 表示第 0 个张量从偏移 0 开始，
        #       第 1 个张量从偏移 256 开始，依此类推。
        offsets = list(accumulate([0] + aligned_bytes[:-1]))

        # 步骤 5：确保当前 ubatch 的工作空间已分配且足够容纳所有张量
        current_workspace = self._ensure_workspace_size(total_bytes)

        # 步骤 6：从工作空间中切分出各张量的视图
        # 每个张量是原始工作空间 buffer 的一个切片视图（view），
        # 先按字节切片，再 view 为目标 dtype，最后 reshape 为目标 shape。
        # 由于是 view 而非 copy，所有张量共享同一块物理显存。
        return [
            current_workspace[offsets[i] : offsets[i] + actual_bytes[i]]
            .view(shapes_and_dtypes[i][1])
            .reshape(shapes_and_dtypes[i][0])
            for i in range(len(shapes_and_dtypes))
        ]

    # 中文注释：_ensure_workspace_size 是工作空间分配的核心内部方法。
    #
    # 功能：确保当前 ubatch 的工作空间已分配且大小足够，然后返回该工作空间张量。
    #
    # 工作流程（分步骤说明）：
    #   步骤 1：获取当前 ubatch 的 ID（通过 ubatching 模块获取）。
    #   步骤 2：获取该 ubatch 当前的工作空间张量及其大小。
    #   步骤 3：如果当前空间不足（current_size < required_bytes）：
    #     步骤 3a：检查是否已锁定。如果已锁定，抛出 AssertionError，
    #              因为锁定状态下不允许扩展（这说明 warmup 不充分）。
    #     步骤 3b：仅扩展当前 ubatch 的工作空间，不扩展其他 ubatch 的。
    #              原因：其他 ubatch 可能仍持有旧工作空间张量的 view，
    #              如果强制扩展会导致旧 view 悬空（dangling view），
    #              引发 DBO 内存泄漏。其他 ubatch 会在下次调用
    #              get_simultaneous() 时懒扩展。
    #     步骤 3c：释放旧工作空间，调用 torch.accelerator.empty_cache()
    #              将释放的显存归还给 CUDA 缓存分配器，以便后续分配更大
    #              的连续块。如果不调用 empty_cache，每次扩展都可能在
    #              预留显存中留下碎片，导致峰值显存偏高。
    #     步骤 3d：分配新的、更大的工作空间张量。
    #   步骤 4：返回工作空间张量。
    def _ensure_workspace_size(self, required_bytes: int) -> torch.Tensor:
        """Ensure workspace is allocated and large enough, return current workspace.

        Args:
            required_bytes: The number of bytes required.

        Returns:
            The current workspace tensor.
        """
        # 步骤 1：获取当前 ubatch ID
        ubatch_id = dbo_current_ubatch_id()
        # 步骤 2：获取当前 ubatch 的工作空间及其大小
        current_workspace = self._current_workspaces[ubatch_id]
        current_size = self._workspace_size_bytes(current_workspace)

        if current_size < required_bytes:

            # 中文注释：辅助函数，用于获取调用栈中第一个非 WorkspaceManager 的调用位置。
            # 这在错误报告和调试日志中非常有用，可以定位是哪个模块/函数
            # 触发了工作空间扩展。
            def get_caller_info() -> str:
                """Find first frame outside WorkspaceManager."""
                curr_frame = inspect.currentframe()
                if curr_frame is None:
                    return "unknown"
                # Walk up the stack skipping WorkspaceManager frames
                curr_frame = curr_frame.f_back
                while curr_frame is not None:
                    # TODO: This only catches instance methods (self), missing
                    # classmethods and staticmethods. Once Python 3.11+ is the
                    # minimum supported version, use co_qualname instead:
                    #   qualname = curr_frame.f_code.co_qualname
                    #   if qualname.startswith("WorkspaceManager."):
                    if isinstance(curr_frame.f_locals.get("self"), WorkspaceManager):
                        curr_frame = curr_frame.f_back
                        continue
                    filename = os.path.basename(curr_frame.f_code.co_filename)
                    return (
                        f"{filename}:{curr_frame.f_lineno}:{curr_frame.f_code.co_name}"
                    )
                return "unknown"

            # 步骤 3a：检查锁定状态。
            # 如果工作空间已锁定但仍然需要扩展，说明 warmup 阶段未能充分
            # 预估工作空间大小，此时抛出明确的错误信息。
            if self._locked:
                raise AssertionError(
                    f"Workspace is locked but allocation from '{get_caller_info()}' "
                    f"requires {required_bytes / _MB:.2f} MB, current size is "
                    f"{current_size / _MB:.2f} MB. "
                    "Workspace growth is not allowed after locking."
                )

            # Only resize the requesting ubatch's workspace.  Other
            # ubatches resize lazily on their next get_simultaneous call.
            # Resizing all ubatches here would orphan the other ubatch's
            # old tensor when it still holds views into it (DBO leak).
            # 步骤 3b：先释放当前 ubatch 的旧工作空间。
            # 仅释放当前 ubatch 的空间，其他 ubatch 的工作空间保持不变。
            # 原因：其他 ubatch 可能持有旧张量的 view（切片视图），
            # 如果在这里强制扩展其他 ubatch 的空间，会导致那些 view
            # 悬空，造成 DBO 内存泄漏。
            self._current_workspaces[ubatch_id] = None
            del current_workspace
            # Release the freed segment back to CUDA so the caching
            # allocator can reuse the GPU memory for the larger
            # allocation below. Without this, each resize may leave a
            # dead segment in reserved memory which can cause higher peak
            # memory usage.
            # 步骤 3c：调用 empty_cache() 将释放的显存归还给 CUDA 缓存分配器。
            # 这样后续的 torch.empty 可以复用这块显存来分配更大的连续块。
            # 如果不调用 empty_cache，每次扩展都可能在 CUDA 预留显存中
            # 留下无法利用的碎片，导致峰值显存使用偏高。
            torch.accelerator.empty_cache()
            # 步骤 3d：分配新的工作空间。使用 dtype=torch.uint8 作为原始字节缓冲区，
            # 后续通过 get_simultaneous() 中的 view/reshape 操作来切分出
            # 不同形状和类型的张量子视图。
            self._current_workspaces[ubatch_id] = torch.empty(
                (required_bytes,), dtype=torch.uint8, device=self._device
            )
            current_workspace = self._current_workspaces[ubatch_id]

            if envs.VLLM_DEBUG_WORKSPACE:
                logger.info(
                    "[WORKSPACE DEBUG] Resized workspace from '%s': %.2f MB -> "
                    "%.2f MB (ubatch %d)",
                    get_caller_info(),
                    current_size / _MB,
                    required_bytes / _MB,
                    ubatch_id,
                )

        return current_workspace


# =============================================================================
# 模块级便捷函数（Module-level convenience functions）
# =============================================================================
# 中文注释：以下函数是对全局 _manager 单例的便捷封装，
# 供外部模块（如 GPUModelRunner）调用，无需直接访问内部变量。

def is_workspace_manager_initialized() -> bool:
    """Check if workspace manager has been initialized.

    Returns:
        True if workspace manager is initialized, False otherwise.
    """
    return _manager is not None


def current_workspace_manager() -> "WorkspaceManager":
    """Get the current workspace manager instance.

    Raises:
        AssertionError: If workspace manager has not been initialized.
    """
    assert _manager is not None, (
        "WorkspaceManager not initialized. Call init_workspace_manager() "
        "with a device before using workspace functions."
    )
    return _manager


# 中文注释：初始化全局工作空间管理器。
# 典型调用时机：GPUModelRunner.__init__() 中，在模型加载完成后调用。
# 初始化后，其他模块即可通过 current_workspace_manager() 获取管理器实例，
# 并通过 get_simultaneous() 获取工作空间张量。
def init_workspace_manager(
    device: torch.device, num_ubatches: int | None = None
) -> None:
    """Initialize the workspace manager with a device.

    Must be called before using any workspace functions. Typically called
    from GPUModelRunner.__init__.

    Args:
        device: The device to allocate workspace on.
        num_ubatches: Number of workspace ubatch slots. Defaults to 1.
    """
    global _manager
    if _manager is not None:
        logger.warning(
            "WorkspaceManager already initialized on device %s, "
            "reinitializing on device %s",
            _manager._device,
            device,
        )
    _manager = WorkspaceManager(device, num_ubatches)


# 中文注释：锁定工作空间，禁止进一步扩展。
# 典型调用时机：模型 warmup/profiling 阶段结束后。
# 锁定后，推理热路径中所有 get_simultaneous() 调用必须在预分配的
# 大小范围内，否则会抛出 AssertionError。
# 这种设计确保了推理过程中不会发生意外的 GPU 内存分配，
# 从而保证推理延迟的稳定性和可预测性。
def lock_workspace() -> None:
    """Lock the workspace to prevent further growth.

    After calling this function, any attempt to allocate a workspace larger
    than the current size will raise an AssertionError. This ensures that
    workspace size is fixed during execution and prevents unexpected memory
    allocations in the hot path.

    Example:
        # During initialization
        init_workspace_manager(device)
        reserve_workspace(shape1, dtype1)
        reserve_workspace(shape2, dtype2)

        # Lock after warmup/profiling
        lock_workspace()

        # Now all get_workspace calls must fit in pre-allocated size
    """
    current_workspace_manager().lock()


# 中文注释：解锁工作空间，允许其扩展。
# 使用场景：弹性专家并行（Elastic EP）缩放时，专家数量发生变化，
# 需要更大的工作空间来容纳新增专家的中间结果。
# 缩放完成后应重新调用 lock_workspace() 以恢复锁定状态。
def unlock_workspace() -> None:
    """Unlock the workspace to allow growth.

    This is used during elastic EP scaling when the workspace size
    needs to grow due to changes in the number of experts.
    After scaling operations complete, lock_workspace() should be
    called again to prevent unexpected allocations.
    """
    current_workspace_manager().unlock()


# 中文注释：重置全局工作空间管理器为未初始化状态。
# 主要用于测试场景，允许测试代码在每个测试用例之间干净地重新初始化管理器。
def reset_workspace_manager() -> None:
    """Reset the workspace manager to uninitialized state.

    This is primarily intended for testing purposes to allow tests
    to reinitialize the workspace manager cleanly.
    """
    global _manager
    _manager = None
