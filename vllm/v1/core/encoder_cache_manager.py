# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
编码器缓存管理器模块 (vllm/v1/core/encoder_cache_manager.py)

本模块实现了多模态模型中编码器输出的缓存管理。

背景：
- 多模态模型（如 LLaVA）在处理图像等输入时，需要先通过编码器提取特征
- 编码器计算通常非常耗时（尤其是视觉编码器）
- 相同的图像可能出现在不同请求中，缓存编码器输出可以避免重复计算

缓存管理的核心设计：
1. 引用计数：每个缓存条目跟踪引用它的请求集合
2. 延迟淘汰：当引用计数降为 0 时，不立即释放，而是标记为可释放
3. 按需淘汰：只有当需要空间时，才真正淘汰可释放的条目
4. LRU 淘汰：优先淘汰最早标记为可释放的条目

缓存状态：
- cached: mm_hash -> 引用该条目的请求 ID 集合（非空表示正在使用）
- freeable: mm_hash -> 编码器嵌入数（有序字典，可释放的条目）
- freed: 最近被淘汰的 mm_hash 列表（用于通知工作器清理）

编码器缓存管理器和编码器-解码器缓存管理器：
- EncoderCacheManager: 用于多模态模型（如 LLaVA），支持完整的缓存功能
- EncoderDecoderCacheManager: 用于编码器-解码器模型（如 T5），简化版本
  目前仅用于调度目的，最终会合并到 EncoderCacheManager
"""

from collections import OrderedDict
from collections.abc import Mapping
from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.v1.request import Request

if TYPE_CHECKING:
    from vllm.config import SchedulerConfig

logger = init_logger(__name__)


class EncoderCacheManager:
    """管理 vLLM V1 中多模态模型编码器输出的缓存。

    处理多模态编码器输出（如图像的视觉嵌入）的生命周期。
    提供内存感知的缓存机制，避免在请求处理的不同阶段重复计算编码器输出。

    特别适用于：
    - 视觉-语言模型（如 LLaVA）缓存图像编码器输出
    - 任何编码器计算昂贵且可缓存的多模态模型

    缓存以单个多模态输入项为粒度操作，实现精细的内存管理。

    缓存功能：
    - 共享：通过哈希值识别相同的多模态数据，在不同请求间共享嵌入
    - 淘汰：在分配时如果没有空闲空间，淘汰最旧的无引用条目

    注意：EncoderCacheManager 在多模态嵌入的粒度上操作，
    而非编码器 token 的粒度。这意味着嵌入之间的文本 token
    不参与缓存大小和空闲槽位的计算。

    Args:
        cache_size: 缓存大小限制，以输入序列的编码器嵌入数为单位。

    Attributes:
        cache_size: 总缓存容量（编码器嵌入数）
        num_free_slots: 当前可用容量（编码器嵌入数）
        num_freeable_slots: 可立即回收的容量（通过淘汰零引用条目）
        cached: mm_hash -> 引用该条目的请求 ID 集合
        freeable: 可释放条目列表（mm_hash -> 编码器嵌入数）
        freed: 自上次 get_freed_mm_hashes() 调用以来被淘汰的 mm_hash 列表
    """

    def __init__(self, cache_size: int):
        self.cache_size = cache_size
        self.num_free_slots = cache_size
        self.num_freeable_slots = cache_size

        # mm_hash -> 引用该多模态数据的请求 ID 集合
        self.cached: dict[str, set[str]] = {}

        # mm_hash -> 该多模态数据的编码器嵌入数（有序字典，按插入顺序）
        self.freeable: OrderedDict[str, int] = OrderedDict()
        # 最近被淘汰的 mm_hash 列表
        self.freed: list[str] = []

    def reset(self) -> None:
        """重置编码器缓存到初始状态。

        清除所有缓存的编码器输出并重置容量跟踪。
        当模型权重更新时调用，以使旧的嵌入失效。
        """
        self.cached.clear()
        self.freeable.clear()
        self.freed.clear()
        self.num_free_slots = self.cache_size
        self.num_freeable_slots = self.cache_size

    def check_and_update_cache(self, request: Request, input_id: int) -> bool:
        """检查特定多模态输入的编码器输出是否已缓存。

        如果已缓存，更新 cached 以将请求 ID 添加到引用集合中。
        如果之前无引用，更新 freeable 和 num_freeable_slots。

        Args:
            request: 包含多模态输入的请求
            input_id: 请求中多模态输入的索引

        Returns:
            True 表示该输入的编码器输出已缓存
        """
        mm_hash = request.mm_features[input_id].identifier
        # 完全未缓存
        if mm_hash not in self.cached:
            return False

        # 已缓存但当前无请求引用
        if not self.cached[mm_hash]:
            num_encoder_embeds = self.freeable.pop(mm_hash)
            self.num_freeable_slots -= num_encoder_embeds

        self.cached[mm_hash].add(request.request_id)
        return True

    def can_allocate(
        self,
        request: Request,
        input_id: int,
        encoder_compute_budget: int,
        num_embeds_to_schedule: int,
    ) -> bool:
        """检查是否有足够的缓存空间用于多模态输入。

        分配逻辑：
        1. 检查计算预算是否足够
        2. 如果 num_free_slots 足够，直接返回 True
        3. 如果 num_free_slots 不足但 num_freeable_slots 足够，
           按 LRU 顺序淘汰可释放条目直到空间足够
        4. 如果 free + freeable 都不够，返回 False

        Args:
            request: 包含多模态输入的请求
            input_id: 多模态输入在请求中的索引
            encoder_compute_budget: 允许计算的编码器嵌入数上限
            num_embeds_to_schedule: 已计划分配缓存空间的嵌入数

        Returns:
            True 表示有足够的容量（可能经过淘汰后）

        Note: 此方法不分配物理内存，仅更新管理器的状态。
        """
        num_embeds = request.get_num_encoder_embeds(input_id)

        # 计算预算不足
        if num_embeds > encoder_compute_budget:
            return False

        num_embeds += num_embeds_to_schedule

        # 空闲槽位足够
        if num_embeds <= self.num_free_slots:
            return True

        # 可回收槽位不足
        if num_embeds > self.num_freeable_slots:
            return False

        # 空闲不足但可回收足够，执行淘汰
        # 注意：淘汰在此处发生，但物理内存在调度器输出通知模型运行器后才释放。
        while num_embeds > self.num_free_slots:
            mm_hash, num_free_embeds = self.freeable.popitem(last=False)
            del self.cached[mm_hash]
            self.freed.append(mm_hash)
            self.num_free_slots += num_free_embeds
        return True

    def allocate(self, request: Request, input_id: int) -> None:
        """为多模态输入的编码器输出分配缓存空间。

        此方法仅更新管理器的簿记信息，实际的编码器输出存储发生在模型运行器中。

        Note:
            此方法假设 can_allocate() 已对同一输入返回 True。
        """

        mm_hash = request.mm_features[input_id].identifier
        request_id = request.request_id
        if mm_hash not in self.cached:
            self.cached[mm_hash] = set()

        num_encoder_embeds = request.get_num_encoder_embeds(input_id)

        # 编码器缓存应始终有足够的空间，因为淘汰在 can_allocate() 中执行
        assert self.num_free_slots >= num_encoder_embeds
        assert self.num_freeable_slots >= num_encoder_embeds

        self.cached[mm_hash].add(request_id)
        self.num_free_slots -= num_encoder_embeds
        self.num_freeable_slots -= num_encoder_embeds

    def get_cached_input_ids(self, request: Request) -> set[int]:
        """获取请求的所有已缓存多模态输入 ID。

        返回其 mm_hash 存在于缓存映射中的输入 ID 集合。
        包括当前无引用的条目（在 freeable 中）。

        Returns:
            已缓存的输入 ID 集合
        """
        return {
            input_id
            for input_id in range(len(request.mm_features))
            if request.mm_features[input_id].identifier in self.cached
        }

    def free_encoder_input(self, request: Request, input_id: int) -> None:
        """释放请求对编码器输入的引用。

        当对应 mm_hash 的引用集合变空时，将条目添加到 freeable
        并增加 num_freeable_slots。

        条目不会被物理释放，直到容量被需要时（由 can_allocate 触发）。
        """
        req_id = request.request_id
        mm_hash = request.mm_features[input_id].identifier
        # mm_hash 不在缓存中或请求 ID 集合为空
        if not self.cached.get(mm_hash, None):
            return
        self.cached[mm_hash].discard(req_id)
        if not self.cached[mm_hash]:
            num_encoder_embeds = request.get_num_encoder_embeds(input_id)
            self.freeable[mm_hash] = num_encoder_embeds
            self.num_freeable_slots += num_encoder_embeds

    def free(self, request: Request) -> None:
        """释放请求持有的所有编码器输入缓存引用。

        对每个缓存的输入 ID 调用 free_encoder_input。
        数据保留在内存中直到未来的分配尝试触发淘汰。

        通常在请求完成、取消或中止时调用。
        """
        input_ids = self.get_cached_input_ids(request)
        for input_id in input_ids:
            self.free_encoder_input(request, input_id)

    def get_freed_mm_hashes(self) -> list[str]:
        """获取并清空最近被淘汰的编码器缓存条目列表。

        Returns:
            自上次调用以来被淘汰的 mm_hash 字符串列表，
            供调度器通知工作器哪些编码器输出可以移除。
        """
        freed = self.freed
        self.freed = []
        return freed


def compute_mm_encoder_budget(
    scheduler_config: "SchedulerConfig",
    mm_max_toks_per_item: Mapping[str, int],
) -> tuple[int, int]:
    """基于模型和调度器配置计算编码器缓存预算。

    Args:
        scheduler_config: 调度器配置
        mm_max_toks_per_item: 每种非文本模态的每项最大 token 数

    Returns:
        - 编码器执行的计算预算（以输入序列 token 数为单位）
        - 编码器缓存大小的空间预算（以输入序列 token 数为单位）
    """

    if not mm_max_toks_per_item:
        logger.warning(
            "All non-text modalities supported by the model have been "
            "explicitly disabled via limit_mm_per_prompt. Encoder cache will "
            "not be initialized."
        )
        return 0, 0

    max_tokens_per_mm_item = max(mm_max_toks_per_item.values())

    if (
        scheduler_config.disable_chunked_mm_input
        and max_tokens_per_mm_item > scheduler_config.max_num_batched_tokens
    ):
        raise ValueError(
            "Chunked MM input disabled but max_tokens_per_mm_item "
            f"({max_tokens_per_mm_item}) is larger than max_num_batched_tokens"
            f" ({scheduler_config.max_num_batched_tokens}). Please increase "
            "max_num_batched_tokens."
        )

    encoder_compute_budget = max(
        scheduler_config.max_num_encoder_input_tokens, max_tokens_per_mm_item
    )
    encoder_cache_size = max(
        scheduler_config.encoder_cache_size, max_tokens_per_mm_item
    )

    return encoder_compute_budget, encoder_cache_size


# 注意 (NickLucche): 编码器-解码器模型的临时实现，仅将管理器用于调度目的。
# 编码器-解码器模型最终将利用缓存，届时此类将合并到 EncoderCacheManager 中，
# 因为与多模态模型的差异正在缩小。
class EncoderDecoderCacheManager(EncoderCacheManager):
    """
    编码器-解码器模型的缓存管理器。

    简化版本，目前仅用于调度目的：
    - 不支持缓存复用（check_and_update_cache 始终返回 False）
    - 简化的分配/释放逻辑
    - 使用 allocated/to_free 列表代替 cached/freeable 字典

    与 EncoderCacheManager 的区别：
    - 编码器输出不会在请求间共享
    - 释放操作在模型执行前进行（通过 get_freed_mm_hashes）
    - to_free 缓冲区确保释放在模型执行后通知
    """

    def __init__(self, cache_size: int):
        self.cache_size = cache_size
        self.num_free_slots = cache_size
        # 已分配的 mm_hash 列表
        self.allocated: list[str] = []
        # 等待释放的 mm_hash 列表
        self.to_free: list[str] = []

    def reset(self) -> None:
        """重置编码器缓存到初始状态。"""
        self.num_free_slots = self.cache_size
        self.allocated.clear()
        self.to_free.clear()

    def check_and_update_cache(self, request: Request, input_id: int) -> bool:
        # 编码器-解码器模型不支持缓存复用
        return False

    def can_allocate(
        self,
        request: Request,
        input_id: int,
        encoder_compute_budget: int,
        num_embeds_to_schedule: int,
    ) -> bool:
        num_encoder_embeds = request.get_num_encoder_embeds(input_id)
        # 计算预算不足
        if num_encoder_embeds > encoder_compute_budget:
            return False

        num_encoder_embeds += num_embeds_to_schedule
        # 检查空闲槽位是否足够
        return num_encoder_embeds <= self.num_free_slots

    def allocate(self, request: Request, input_id: int) -> None:
        num_encoder_embeds = request.get_num_encoder_embeds(input_id)
        self.num_free_slots -= num_encoder_embeds

        mm_hash = request.mm_features[input_id].identifier
        self.allocated.append(mm_hash)

    def free(self, request: Request) -> None:
        for input_id in range(len(request.mm_features)):
            self.free_encoder_input(request, input_id)

    def get_cached_input_ids(self, request: Request) -> set[int]:
        return set(range(len(request.mm_features)))

    def get_freed_mm_hashes(self) -> list[str]:
        # 编码器缓存未用于编解码模型，可在此释放条目。
        # 实际释放在 runner 中、模型执行之前进行。
        # 因此 freeable 充当缓冲区，仅在模型执行后释放条目，
        # 模拟 EncoderCacheManager 的状态转换。
        to_free = self.to_free
        self.to_free = self.allocated
        self.allocated = []
        return to_free

    def free_encoder_input(self, request: Request, input_id: int) -> None:
        num_encoder_embeds = request.get_num_encoder_embeds(input_id)
        self.num_free_slots += num_encoder_embeds
