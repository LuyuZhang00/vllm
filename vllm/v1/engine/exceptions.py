# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
异常类定义模块。

本模块定义了 vLLM v1 引擎使用的核心异常类型：

1. EngineGenerateError - 生成过程中的可恢复错误
2. EngineDeadError - 引擎核心进程死亡的不可恢复错误

异常层次设计原则：
- 可恢复错误（EngineGenerateError）：
  * 表示单个请求的生成失败
  * 调用者可以捕获此异常并重试或返回错误响应
  * 不影响引擎整体状态

- 不可恢复错误（EngineDeadError）：
  * 表示引擎核心进程已崩溃或无法继续工作
  * 通常由底层 ZMQ 通信错误、进程异常退出等触发
  * 调用者收到此异常后应停止所有请求并考虑重启引擎
  * 通过 suppress_context 参数可简化堆栈跟踪，避免显示无关的 ZMQ 错误
"""


class EngineGenerateError(Exception):
    """Raised when a AsyncLLM.generate() fails. Recoverable."""
    """
    引擎生成错误（可恢复）。

    当 AsyncLLM.generate() 方法执行失败时抛出。
    这是一个可恢复的错误，表示单个请求的生成过程中出现了问题，
    但引擎核心仍然正常运行。

    常见场景：
    - 请求参数无效
    - 模型推理过程中的临时错误
    - 资源不足但引擎仍可服务其他请求

    使用方式：
        try:
            async for output in llm.generate(request):
                ...
        except EngineGenerateError as e:
            # 处理单个请求的错误，引擎仍可继续使用
            logger.error(f"Generation failed: {e}")
    """

    pass


class EngineDeadError(Exception):
    """Raised when the EngineCore dies. Unrecoverable."""
    """
    引擎死亡错误（不可恢复）。

    当引擎核心（EngineCore）进程死亡或无法继续工作时抛出。
    这是一个不可恢复的错误，表明整个引擎已不可用。

    常见触发场景：
    1. EngineCore 进程崩溃或被终止
    2. ZMQ 通信通道断开
    3. GPU 显存溢出导致 CUDA 错误
    4. 关键内部组件异常

    特殊设计：
    - 使用自定义的错误消息，引导用户查看上方的堆栈跟踪
    - suppress_context 参数用于在使用 LLMEngine 时简化堆栈，
      隐藏不相关的 ZMQ 错误信息，使根因更加清晰

    使用方式：
        try:
            await llm.generate(request)
        except EngineDeadError:
            # 引擎已死亡，需要重启
            logger.critical("Engine is dead, restarting...")
            await restart_engine()
    """

    def __init__(self, *args, suppress_context: bool = False, **kwargs):
        # 统一的错误消息，引导用户查看堆栈跟踪以找到根因
        ENGINE_DEAD_MESSAGE = "EngineCore encountered an issue. See stack trace (above) for the root cause."  # noqa: E501

        super().__init__(ENGINE_DEAD_MESSAGE, *args, **kwargs)
        # Make stack trace clearer when using with LLMEngine by
        # silencing irrelevant ZMQError.
        # 当使用 LLMEngine 时，通过抑制异常上下文来使堆栈跟踪更清晰，
        # 隐藏不相关的 ZMQ 错误信息
        self.__suppress_context__ = suppress_context
