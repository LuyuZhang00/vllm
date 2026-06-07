# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
调度器配置模块 (Scheduler Configuration Module)

==============================================================================
【模块概述】
本模块定义了 vLLM 调度器 (Scheduler) 的所有配置项。
调度器是 vLLM 推理引擎的核心组件之一，负责决定每个迭代 (iteration) 中
哪些请求被处理、处理多少 token。

【核心职责】
1. 控制批处理大小：通过 max_num_batched_tokens 和 max_num_seqs 限制
   每个迭代中处理的 token 总数和序列总数
2. 管理分块预填充 (Chunked Prefill)：将长 prompt 分块处理，避免单次
   预填充占用过多资源
3. 调度策略选择：支持 FCFS（先来先服务）和 Priority（优先级）两种策略
4. 多模态输入处理：为多模态模型配置编码器缓存和计算预算
5. 异步调度：支持异步调度以提高 GPU 利用率

【关键概念】
- Batched Tokens (批处理 token 数)：单次迭代中所有请求的 token 总和
- Max Num Seqs (最大序列数)：单次迭代中最多同时处理的请求数量
- Chunked Prefill (分块预填充)：将长 prompt 拆分为多个块逐步处理
- Partial Prefill (部分预填充)：分块预填充时，允许同时处理多个部分预填充请求
==============================================================================
"""

from collections.abc import Callable
from dataclasses import InitVar
from typing import TYPE_CHECKING, Any, ClassVar, Literal, cast

from pydantic import Field, field_validator
from typing_extensions import Self

from vllm.config.utils import config
from vllm.logger import init_logger
from vllm.utils.hashing import safe_hash
from vllm.utils.import_utils import resolve_obj_by_qualname

if TYPE_CHECKING:
    from vllm.v1.core.sched.interface import SchedulerInterface

logger = init_logger(__name__)

# 运行器类型：
# - "generate"：标准文本生成模式（最常用）
# - "pooling"：池化模式，用于获取 embedding 向量（如嵌入模型）
# - "draft"：草稿模式，用于投机解码 (Speculative Decoding) 的草稿模型
RunnerType = Literal["generate", "pooling", "draft"]

# 调度策略类型：
# - "fcfs"：First Come First Served，先来先服务，按请求到达顺序处理
# - "priority"：优先级调度，按用户指定的优先级处理（值越小优先级越高），
#              优先级相同时按到达时间排序
SchedulerPolicy = Literal["fcfs", "priority"]


@config
class SchedulerConfig:
    """
    调度器配置类 (Scheduler Configuration)

    【类概述】
    本类封装了 vLLM 调度器的所有配置参数。调度器在每个推理迭代中决定：
    1. 哪些请求被调度执行
    2. 每个请求处理多少 token
    3. 如何在多个请求之间分配计算资源

    【配置项分类】
    - 批处理控制：max_num_batched_tokens, max_num_scheduled_tokens, max_num_seqs
    - 分块预填充：enable_chunked_prefill, max_num_partial_prefills,
                  max_long_partial_prefills, long_prefill_token_threshold
    - 调度策略：policy (fcfs/priority)
    - 多模态支持：is_multimodal_model, max_num_encoder_input_tokens,
                  encoder_cache_size, disable_chunked_mm_input
    - 调度器选择：scheduler_cls, async_scheduling
    - 流式输出：stream_interval

    【使用场景】
    - 在线服务 (Online Serving)：通过 API 服务器接收请求时使用
    - 离线推理 (Offline Inference)：通过 LLM 类进行批量推理时使用
    - 基准测试 (Benchmarking)：通过 vllm bench 命令进行性能测试时使用
    """

    # =========================================================================
    # 【InitVar 参数说明】
    # InitVar 是 dataclass 的特殊字段，只在 __init__ 时传递，不存储为实例属性。
    # 这些值来自 ModelConfig，在 __post_init__ 中用于验证和设置默认值。
    # =========================================================================

    max_model_len: InitVar[int]
    """模型支持的最大序列长度（包括 prompt 和生成的文本）。

    【说明】
    - 此值存储在 ModelConfig 中，此处仅用于提供默认值和验证其他属性
    - 例如：如果 max_model_len=4096，则任何超过 4096 token 的请求都会被拒绝
    - 此值影响 KV 缓存的大小分配和内存规划
    """

    is_encoder_decoder: InitVar[bool]
    """是否为编码器-解码器模型（Encoder-Decoder Model）。

    【说明】
    - 此值存储在 ModelConfig 中，此处用于禁用不兼容的功能
    - 编码器-解码器模型（如 T5、BART）不支持分块预填充和前缀缓存
    - 常见的仅解码器模型（如 LLaMA、GPT）此处为 False
    """

    # =========================================================================
    # 【类变量：默认值常量】
    # 这些是类级别的常量，用于提供配置项的默认值
    # =========================================================================

    # 默认最大批处理 token 数：2048
    # 这是一个保守的默认值，实际使用中通常由 EngineArgs.create_engine_config 设置
    DEFAULT_MAX_NUM_BATCHED_TOKENS: ClassVar[int] = 2048

    # 批量数据并行 (Batched DP) 时的默认最大批处理 token 数：256
    # 数据并行时每个 DP rank 处理的 token 数较少
    DEFAULT_MAX_NUM_BATCHED_TOKENS_FOR_BATCHED_DP: ClassVar[int] = 256

    # 默认最大序列数：128
    # 单次迭代中最多同时处理 128 个请求
    DEFAULT_MAX_NUM_SEQS: ClassVar[int] = 128

    # =========================================================================
    # 【实例配置字段】
    # =========================================================================

    runner_type: RunnerType = "generate"
    """运行器类型 (Runner Type)。

    【可选值】
    - "generate"：标准文本生成模式（默认），用于自回归语言模型
    - "pooling"：池化模式，用于获取文本嵌入向量（如 sentence-transformers）
    - "draft"：草稿模式，用于投机解码中的草稿模型

    【影响】
    此字段决定了使用哪种 Worker 和 ModelRunner 来执行模型推理。
    """

    max_num_batched_tokens: int = Field(default=DEFAULT_MAX_NUM_BATCHED_TOKENS, ge=1)
    """单次迭代中可处理的最大 token 数量（批处理 token 上限）。

    【作用】
    这是调度器最重要的参数之一，直接控制：
    1. GPU 显存占用：更多 token 意味着更大的激活张量
    2. 吞吐量：较大的值可以提高 GPU 利用率
    3. 延迟：较大的值可能增加单次迭代的延迟

    【典型值】
    - 8B 模型、80GB 显存：通常设置为 8192-16384
    - 70B 模型、多 GPU：通常设置为 4096-8192
    - 受限显存：可能需要降低到 2048 或更低

    【注意】
    - 默认值 2048 主要用于测试，生产环境应根据硬件调整
    - 实际使用中由 EngineArgs.create_engine_config 自动计算
    - 必须 >= max_num_seqs（每个序列至少 1 个 token）
    """

    max_num_scheduled_tokens: int | None = None
    """调度器单次迭代实际调度的最大 token 数量。

    【与 max_num_batched_tokens 的区别】
    - max_num_batched_tokens：模型实际处理的 token 上限
    - max_num_scheduled_tokens：调度器发出的 token 上限

    【使用场景】
    在投机解码 (Speculative Decoding) 中，调度器发出 N 个 token，
    但模型可能需要额外处理草稿模型的 token，因此：
    max_num_scheduled_tokens < max_num_batched_tokens

    【默认行为】
    如果为 None，则等于 max_num_batched_tokens。
    """

    max_num_seqs: int = Field(default=DEFAULT_MAX_NUM_SEQS, ge=1)
    """单次迭代中可处理的最大序列（请求）数量。

    【作用】
    限制同时处理的请求数量，用于：
    1. 控制调度开销：更多请求意味着更多的调度逻辑开销
    2. 保证公平性：防止单次迭代占用过多资源
    3. 内存控制：每个请求都需要独立的 KV 缓存空间

    【典型值】
    - 在线服务：128-256（平衡延迟和吞吐量）
    - 离线批处理：可以设置更高（如 512）

    【注意】
    - 默认值 128 主要用于测试
    - 必须 <= max_num_batched_tokens
    """

    # =========================================================================
    # 【分块预填充相关配置】
    # 分块预填充 (Chunked Prefill) 是一项重要优化：
    # 将长 prompt 分成多个块，与解码请求混合处理，提高 GPU 利用率
    # =========================================================================

    max_num_partial_prefills: int = Field(default=1, ge=1)
    """分块预填充时，最多允许同时进行部分预填充的序列数量。

    【作用】
    当 enable_chunked_prefill=True 时，长 prompt 会被分成多个块处理。
    此参数限制同时处理的部分预填充请求数量。

    【示例】
    如果 max_num_partial_prefills=2，有两个长 prompt 请求 A 和 B：
    - 迭代 1：处理 A 的第 1 块 + B 的第 1 块
    - 迭代 2：处理 A 的第 2 块 + B 的第 2 块

    【性能影响】
    - 较大的值：更高的 GPU 利用率，但可能增加调度复杂度
    - 较小的值（如 1）：更简单的调度逻辑，但可能降低 GPU 利用率
    """

    max_long_partial_prefills: int = Field(default=1, ge=1)
    """分块预填充时，最多允许同时处理的"长"prompt 数量。

    【"长"prompt 的定义】
    prompt 长度 > long_prefill_token_threshold 的请求被认为是"长"prompt。

    【作用】
    限制同时处理的长 prompt 数量，允许短 prompt 插队处理：
    - 如果 max_long_partial_prefills < max_num_partial_prefills，
      短 prompt 可以在长 prompt 的间隙中被处理
    - 这可以降低短请求的延迟（TTFT - Time To First Token）

    【示例】
    设 max_num_partial_prefills=3, max_long_partial_prefills=1：
    - 长 prompt A 正在处理中
    - 短 prompt B 和 C 到达
    - 下一个迭代：A 的第 2 块 + B + C（短 prompt 插队）
    """

    long_prefill_token_threshold: int = 0
    """判定"长"prompt 的 token 阈值。

    【作用】
    prompt 长度超过此阈值的请求被归类为"长"prompt，
    受 max_long_partial_prefills 限制。

    【默认行为】
    - 默认值为 0，表示所有 prompt 都被视为"长"prompt
    - 当 max_num_partial_prefills > 1 时，会自动设置为
      max_model_len * 0.04（即模型最大长度的 4%）

    【示例】
    如果 max_model_len=4096，阈值自动设为 163：
    - prompt 长度 100：短 prompt，可以插队
    - prompt 长度 200：长 prompt，受 max_long_partial_prefills 限制
    """

    enable_chunked_prefill: bool = True
    """是否启用分块预填充 (Chunked Prefill)。

    【什么是分块预填充】
    传统方式：一次性处理整个 prompt，可能导致：
    - GPU 显存不足（prompt 太长）
    - 解码请求被阻塞（需要等待长 prompt 完成）

    分块预填充：将长 prompt 分成多个块，与解码请求交替处理：
    - 迭代 1：处理 prompt 的前 1024 个 token + 其他请求的解码
    - 迭代 2：处理 prompt 的下 1024 个 token + 其他请求的解码
    - ...直到 prompt 完全处理

    【优势】
    1. 更好的延迟：解码请求不需要等待长 prompt 完成
    2. 更高的吞吐量：GPU 始终有工作可做
    3. 更稳定的显存使用：避免显存峰值

    【注意】
    - 默认启用（True）
    - 编码器-解码器模型不支持，会自动禁用
    - 实际使用中由 EngineArgs.create_engine_config 根据模型特性决定
    """

    # =========================================================================
    # 【多模态模型相关配置】
    # 用于支持视觉、音频等多模态输入的模型
    # =========================================================================

    is_multimodal_model: bool = False
    """是否为多模态模型。

    【作用】
    标识当前模型是否支持多模态输入（如图像、音频）。
    多模态模型需要额外的编码器来处理非文本输入。

    【影响】
    - 启用多模态相关的缓存和调度逻辑
    - 配置编码器计算预算和缓存大小
    """

    # TODO (ywang96): Make this configurable.
    # 【待改进】此字段当前不可由用户配置，未来版本将支持
    max_num_encoder_input_tokens: int = Field(init=False)
    """多模态编码器的计算预算（仅用于 V1 引擎）。

    【作用】
    限制编码器单次处理的最大 token 数量，用于：
    1. 控制编码器的计算开销
    2. 防止编码器占用过多 GPU 资源

    【默认行为】
    - 等于 max_num_batched_tokens
    - 如果最大多模态嵌入大小超过此值，会被覆盖

    【注意】
    此字段当前不可配置，由系统自动设置。
    """

    # TODO (ywang96): Make this configurable.
    # 【待改进】此字段当前不可由用户配置，未来版本将支持
    encoder_cache_size: int = Field(init=False)
    """多模态编码器的缓存大小（仅用于 V1 引擎）。

    【作用】
    缓存编码器的输出结果，避免重复计算：
    - 相同的图像输入只需要编码一次
    - 缓存可以显著提高多模态请求的处理效率

    【默认行为】
    - 等于 max_num_batched_tokens
    - 如果最大多模态嵌入大小超过此值，会被覆盖

    【注意】
    此字段当前不可配置，由系统自动设置。
    """

    # =========================================================================
    # 【调度策略配置】
    # =========================================================================

    policy: SchedulerPolicy = "fcfs"
    """调度策略 (Scheduling Policy)。

    【可选策略】

    1. "fcfs" (First Come First Served) - 先来先服务（默认）
       - 按请求到达顺序处理
       - 简单、公平、可预测
       - 适合大多数场景

    2. "priority" - 优先级调度
       - 按用户指定的优先级处理（值越小优先级越高）
       - 优先级相同时按到达时间排序
       - 适合需要差异化服务的场景（如 VIP 用户、紧急请求）

    【使用示例】
    ```python
    # FCFS 策略（默认）
    config = SchedulerConfig(policy="fcfs")

    # 优先级策略
    config = SchedulerConfig(policy="priority")
    # 请求通过 priority 参数指定优先级
    ```
    """

    # =========================================================================
    # 【多模态输入分块配置】
    # =========================================================================

    disable_chunked_mm_input: bool = False
    """是否禁用多模态输入的分块处理（仅用于 V1 引擎）。

    【问题背景】
    当启用分块预填充时，一个混合 prompt（文本+图像）可能被部分调度：
    - 原始 prompt：TTTT IIIIIIIIII（4 个文本 token + 10 个图像 token）
    - 如果只能调度 9 个 token：TTTT IIIII（剩余 IIIII）

    【禁用分块的效果】
    当 disable_chunked_mm_input=True 时：
    - 迭代 1：只调度 TTTT（文本部分）
    - 迭代 2：调度 IIIIIIIIII（完整的图像部分）

    【优势】
    确保多模态输入（如图像）被完整处理，避免：
    - 图像 token 被拆分导致的语义损失
    - 编码器需要处理不完整的输入

    【注意】
    - 默认为 False（允许分块处理多模态输入）
    - 某些模型可能需要设为 True 以确保正确性
    """

    # =========================================================================
    # 【调度器类配置】
    # =========================================================================

    # scheduler class or path. "vllm.v1.core.sched.scheduler.Scheduler"
    # (default) or "mod.custom_class".
    scheduler_cls: str | type[object] | None = None
    """自定义调度器类。

    【作用】
    允许用户指定自定义的调度器实现，用于：
    1. 实验新的调度算法
    2. 针对特定场景优化调度策略
    3. 集成外部调度系统

    【使用方式】
    - None（默认）：使用系统默认调度器
    - 类型为 type[object]：直接传入调度器类
    - 类型为 str：传入类的完整路径，如 "module.submodule.Scheduler"

    【默认调度器】
    - 同步调度：vllm.v1.core.sched.scheduler.Scheduler
    - 异步调度：vllm.v1.core.sched.async_scheduler.AsyncScheduler

    【注意】
    自定义调度器接口尚未公开，兼容性可能无法保证。
    """

    disable_hybrid_kv_cache_manager: bool | None = None
    """是否禁用混合 KV 缓存管理器。

    【什么是混合 KV 缓存管理器】
    某些模型（如 Mistral、Gemma 2）同时使用：
    - 全注意力 (Full Attention)：关注所有 token
    - 滑动窗口注意力 (Sliding Window Attention)：只关注最近的 N 个 token

    混合 KV 缓存管理器会为不同类型的注意力层分配不同大小的 KV 缓存：
    - 全注意力层：需要完整的 KV 缓存
    - 滑动窗口层：只需要窗口大小的 KV 缓存

    【配置选项】
    - None（默认）：根据环境和启动配置自动决定
    - True：禁用混合管理器，所有注意力层使用相同大小的 KV 缓存
    - False：强制启用混合管理器

    【何时禁用】
    当遇到兼容性问题或调试时，可以设为 True 禁用混合管理器。
    """

    scheduler_reserve_full_isl: bool = True
    """是否在调度时预留完整输入序列长度 (ISL) 的 KV 缓存空间。

    【什么是 ISL 预留】
    ISL = Input Sequence Length，即输入序列的总长度。

    当 enable_chunked_prefill=True 时：
    - False：只检查当前块是否能放入 KV 缓存
    - True（默认）：检查整个 prompt 是否能放入 KV 缓存

    【为什么需要预留】
    如果不预留完整 ISL：
    - 可能接受过多请求（over-admission）
    - 后续块可能无法分配 KV 缓存
    - 导致 KV 缓存抖动 (thrashing)，影响性能

    【示例】
    设 KV 缓存可容纳 10000 个 token：
    - 请求 A 的 ISL = 8000
    - 请求 B 的 ISL = 6000

    如果 scheduler_reserve_full_isl=True：
    - 接受 A 后，剩余 2000 空间，拒绝 B
    - 避免后续无法分配的问题

    如果 scheduler_reserve_full_isl=False：
    - 可能接受 A 和 B
    - 后续可能出现分配失败
    """

    async_scheduling: bool | None = None
    """是否启用异步调度。

    【什么是异步调度】
    传统同步调度：
    - GPU 计算 → CPU 调度 → GPU 计算 → ...
    - CPU 调度期间 GPU 空闲

    异步调度：
    - GPU 计算的同时，CPU 提前调度下一批请求
    - GPU 完成计算后立即开始下一批，无空闲时间

    【优势】
    1. 更高的 GPU 利用率：减少 GPU 空闲时间
    2. 更低的延迟：请求不需要等待调度完成
    3. 更高的吞吐量：每秒处理更多请求

    【配置选项】
    - None（默认）：根据系统配置自动决定
    - True：强制启用异步调度
    - False：强制禁用异步调度

    【注意】
    异步调度需要额外的同步机制，可能增加代码复杂度。
    """

    stream_interval: int = Field(default=1, ge=1)
    """流式输出的 token 间隔（缓冲区大小）。

    【作用】
    控制流式输出时，每生成多少个 token 就发送一次给客户端。

    【配置效果】
    - stream_interval=1（默认）：每生成 1 个 token 立即发送
      优点：最流畅的流式体验
      缺点：较高的主机开销（频繁的 IPC 通信）

    - stream_interval=10：每生成 10 个 token 批量发送
      优点：减少主机开销，可能提高吞吐量
      缺点：流式体验不够流畅，有 10 个 token 的延迟

    【使用场景】
    - 对话式应用：stream_interval=1（实时感强）
    - 批量处理：stream_interval=10 或更高（减少开销）
    """

    # =========================================================================
    # 【工厂方法】
    # =========================================================================

    @staticmethod
    def default_factory(**kwargs):
        """
        创建 SchedulerConfig 的工厂方法，为 InitVar 参数提供默认值。

        【作用】
        当不提供 max_model_len 或 is_encoder_decoder 时，使用安全的默认值：
        - max_model_len = 8192（常见的默认序列长度）
        - is_encoder_decoder = False（假设是仅解码器模型）

        【使用场景】
        在测试或不需要完整 ModelConfig 的场景下快速创建配置：
        ```python
        config = SchedulerConfig.default_factory()
        config = SchedulerConfig.default_factory(max_num_batched_tokens=4096)
        ```
        """
        if "max_model_len" not in kwargs:
            kwargs["max_model_len"] = 8192
        if "is_encoder_decoder" not in kwargs:
            kwargs["is_encoder_decoder"] = False
        return SchedulerConfig(**kwargs)

    def get_scheduler_cls(self) -> type["SchedulerInterface"]:
        """
        获取调度器类。

        【返回值】
        返回调度器类的类型，用于实例化调度器。

        【调度器选择逻辑】（按优先级）

        1. 如果 scheduler_cls 已指定：
           - 直接使用用户指定的自定义调度器
           - 支持传入类对象或类路径字符串

        2. 如果 scheduler_cls 为 None（默认）：
           a. 如果 async_scheduling=True：
              → 返回 AsyncScheduler（异步调度器）
              → 位于 vllm.v1.core.sched.async_scheduler

           b. 如果 async_scheduling=False 或 None：
              → 返回 Scheduler（同步调度器）
              → 位于 vllm.v1.core.sched.scheduler

        【注意】
        自定义调度器接口尚未公开，使用时会打印警告信息。
        """
        if self.scheduler_cls is None:
            if self.async_scheduling:
                from vllm.v1.core.sched.async_scheduler import AsyncScheduler

                return AsyncScheduler
            from vllm.v1.core.sched.scheduler import Scheduler

            return Scheduler

        # This warning can be removed once the Scheduler interface is
        # finalized and we can maintain support for scheduler classes that
        # implement it
        logger.warning_once(
            "Using custom scheduler class %s. This scheduler interface is "
            "not public and compatibility may not be maintained.",
            self.scheduler_cls,  # type: ignore[arg-type]
        )
        if not isinstance(self.scheduler_cls, str):
            return cast(type["SchedulerInterface"], self.scheduler_cls)
        return resolve_obj_by_qualname(self.scheduler_cls)

    def compute_hash(self) -> str:
        """
        计算配置哈希值。

        【作用】
        生成一个唯一标识此配置的哈希字符串，用于：
        1. torch.compile 缓存：判断配置是否变化，决定是否需要重新编译
        2. CUDA Graph 缓存：判断是否需要重新捕获 CUDA Graph
        3. LoRA 缓存：判断是否需要重新创建 LoRA 缓冲区

        【哈希因子】
        当前包含的因子：
        - max_num_batched_tokens

        【为什么需要包含 max_num_batched_tokens】
        1. LoRA 会基于 max_num_batched_tokens 创建静态缓冲区，
           缓冲区的大小和步长会被 torch.compile 图显式捕获
        2. Inductor 根据数据大小决定使用 32 位还是 64 位索引整数，
           max_num_batched_tokens 会影响这个决定

        【警告】
        每当添加新字段时，如果它影响计算图结构，
        必须将其包含在 factors 列表中。
        """
        factors: list[Any] = []

        # max_num_batched_tokens need to be included in the hash due
        # to two reasons:
        # 1. LoRA creates static buffers based on max_num_batched_tokens.
        #   The tensor sizes and strides get captured in the torch.compile
        #   graph explicitly.
        # 2. Inductor decides whether using 32-bit or 64-bit indexing integer
        #   based on the data sizes. `max_num_batched_tokens` has an
        #   impact on that. For more details, please check
        #   https://github.com/vllm-project/vllm/issues/29585
        factors.append(self.max_num_batched_tokens)

        hash_str = safe_hash(str(factors).encode(), usedforsecurity=False).hexdigest()
        return hash_str

    @field_validator("scheduler_cls", "async_scheduling", mode="wrap")
    @classmethod
    def _skip_none_validation(cls, value: Any, handler: Callable) -> Any:
        """
        跳过 None 值的验证（当初始化被延迟时）。

        【作用】
        对于 scheduler_cls 和 async_scheduling 字段：
        - 如果值为 None，直接返回 None，不进行验证
        - 如果值不为 None，调用正常的验证处理器

        【使用场景】
        这些字段可能在初始化时未被设置（延迟初始化），
        None 是一个有效的初始值，不应该触发验证错误。
        """
        return None if value is None else handler(value)

    def __post_init__(self, max_model_len: int, is_encoder_decoder: bool) -> None:
        """
        初始化后处理方法。

        【执行时机】
        在 dataclass 的 __init__ 完成后自动调用。

        【处理流程】

        1. 【编码器-解码器模型特殊处理】
           如果是编码器-解码器模型（如 T5、BART）：
           - 禁用多模态输入分块 (disable_chunked_mm_input = True)
           - 禁用分块预填充 (enable_chunked_prefill = False)
           - 重置长预填充阈值 (long_prefill_token_threshold = 0)
           - 原因：这些模型的架构不支持分块处理

        2. 【设置编码器配置】
           - max_num_encoder_input_tokens = max_num_batched_tokens
           - encoder_cache_size = max_num_batched_tokens
           - 原因：编码器的计算预算和缓存大小应与批处理大小一致

        3. 【记录分块预填充状态】
           如果启用分块预填充，记录日志

        4. 【设置长预填充阈值】
           如果 max_num_partial_prefills > 1 且阈值为 0：
           - 自动设置为 max_model_len * 0.04
           - 即模型最大长度的 4%

        5. 【验证配置】
           调用 verify_max_model_len 验证配置的有效性
        """
        if is_encoder_decoder:
            # Chunked prefill should be disabled for encoder-decoder models.
            self.disable_chunked_mm_input = True
            self.enable_chunked_prefill = False
            self.long_prefill_token_threshold = 0
            logger.info(
                "Encoder-decoder models do not support chunked prefill nor"
                " prefix caching; disabling both."
            )

        self.max_num_encoder_input_tokens = self.max_num_batched_tokens
        self.encoder_cache_size = self.max_num_batched_tokens

        if self.enable_chunked_prefill:
            logger.info_once(
                "Chunked prefill is enabled with max_num_batched_tokens=%d.",
                self.max_num_batched_tokens,
            )

        if self.max_num_partial_prefills > 1:
            if self.long_prefill_token_threshold == 0:
                self.long_prefill_token_threshold = int(max_model_len * 0.04)

            logger.info(
                "Concurrent partial prefills enabled with "
                "max_num_partial_prefills=%d, max_long_partial_prefills=%d, "
                "long_prefill_token_threshold=%d",
                self.max_num_partial_prefills,
                self.max_long_partial_prefills,
                self.long_prefill_token_threshold,
            )

        self.verify_max_model_len(max_model_len)

    def verify_max_model_len(self, max_model_len: int) -> Self:
        """
        验证调度器配置的有效性。

        【验证规则】

        规则 1：max_num_batched_tokens >= max_model_len（除非启用分块预填充）
        - 原因：如果不启用分块预fill，批处理大小必须能容纳完整序列
        - 违反后果：有效限制最大序列长度，超出的请求会被拒绝
        - 解决方案：增加 max_num_batched_tokens 或减少 max_model_len

        规则 2：max_num_batched_tokens >= max_num_seqs
        - 原因：每个序列至少需要 1 个 token 的空间
        - 违反后果：无法为每个序列分配最少的 token 空间

        规则 3（警告）：max_num_batched_tokens <= max_num_seqs * max_model_len
        - 原因：如果批处理大小超过所有序列都能达到最大长度的总和，
          可能导致资源浪费或意外行为
        - 这是一个警告，不会阻止启动

        规则 4：max_num_partial_prefills > 1 时必须启用分块预fill
        - 原因：部分预fill是分块预fill的子功能
        - 违反后果：配置逻辑矛盾

        规则 5：long_prefill_token_threshold <= max_model_len
        - 原因：阈值不应超过模型支持的最大长度
        - 违反后果：所有请求都被认为是"长"请求，阈值失去意义

        规则 6：max_long_partial_prefills <= max_num_partial_prefills
        - 原因：长请求的部分预fill数不应超过总的部分预fill数
        - 违反后果：配置逻辑矛盾

        【返回值】
        返回 self，支持链式调用。
        """
        if (
            self.max_num_batched_tokens < max_model_len
            and not self.enable_chunked_prefill
        ):
            raise ValueError(
                f"max_num_batched_tokens ({self.max_num_batched_tokens}) is "
                f"smaller than max_model_len ({max_model_len}). "
                "This effectively limits the maximum sequence length to "
                "max_num_batched_tokens and makes vLLM reject longer "
                "sequences. Please increase max_num_batched_tokens or "
                "decrease max_model_len."
            )

        if self.max_num_batched_tokens < self.max_num_seqs:
            raise ValueError(
                f"max_num_batched_tokens ({self.max_num_batched_tokens}) must "
                "be greater than or equal to max_num_seqs "
                f"({self.max_num_seqs})."
            )

        if self.max_num_batched_tokens > self.max_num_seqs * max_model_len:
            logger.warning(
                "max_num_batched_tokens (%d) exceeds max_num_seqs "
                "* max_model_len (%d). This may lead to unexpected behavior.",
                self.max_num_batched_tokens,
                self.max_num_seqs * max_model_len,
            )

        if self.max_num_partial_prefills > 1:
            if not self.enable_chunked_prefill:
                raise ValueError(
                    "Chunked prefill must be enabled to set "
                    "max_num_partial_prefills > 1."
                )

            if self.long_prefill_token_threshold > max_model_len:
                raise ValueError(
                    "long_prefill_token_threshold "
                    f"({self.long_prefill_token_threshold}) cannot be greater "
                    f"than the max_model_len ({max_model_len})."
                )

        if self.max_long_partial_prefills > self.max_num_partial_prefills:
            raise ValueError(
                f"{self.max_long_partial_prefills=} must be less than or equal to "
                f"{self.max_num_partial_prefills=}."
            )

        return self
