# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# =============================================================================
# 结构化输出管理器 (StructuredOutputManager)
# =============================================================================
# 本模块是 vLLM v1 引擎中结构化输出功能的核心管理层。
# 结构化输出允许用户约束模型的输出格式，使其符合特定的 JSON 模式、
# 正则表达式、EBNF 语法规则、选项列表或结构化标签。
#
# 主要职责：
# 1. 初始化和管理结构化输出后端（xgrammar、guidance、outlines、lm-format-enforcer）
# 2. 编译语法规范并为每个请求创建语法对象
# 3. 生成并管理语法位掩码（bitmask），用于在解码阶段约束 token 采样
# 4. 与推理模式（thinking/reasoning）集成，决定何时启用结构化约束
# 5. 管理线程池以支持异步语法编译和并行位掩码填充
#
# 请求流程：
# 1. 请求到达时，grammar_init() 初始化后端并提交语法编译任务
# 2. 编译完成后，grammar_bitmask() 为当前批次生成位掩码
# 3. 位掩码传递给 GPU 模型运行器，在采样时过滤不允许的 token
# =============================================================================

import itertools
import multiprocessing
from collections.abc import Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.reasoning import ReasoningParserManager
from vllm.tokenizers import cached_tokenizer_from_config
from vllm.utils.import_utils import LazyLoader
from vllm.v1.structured_output.backend_guidance import GuidanceBackend
from vllm.v1.structured_output.backend_types import (
    StructuredOutputBackend,
    StructuredOutputGrammar,
    StructuredOutputOptions,
)
from vllm.v1.structured_output.backend_xgrammar import XgrammarBackend

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
    import torch

    from vllm.reasoning import ReasoningParser
    from vllm.v1.request import Request
else:
    torch = LazyLoader("torch", globals(), "torch")


logger = init_logger(__name__)


class StructuredOutputManager:
    """Engine-level manager for structured output requests.

    # 结构化输出管理器：引擎级别的结构化输出请求管理类
    # 负责协调语法编译、位掩码生成以及与推理模式的集成
    """

    def __init__(self, vllm_config: VllmConfig):
        self.backend: StructuredOutputBackend | None = None
        # We only store the class of the reasoner in the manager.
        # The parser instance is request-scoped because some reasoning parsers
        # depend on per-request chat-template kwargs.
        # 只存储推理解析器的类引用，不存储实例。
        # 解析器实例是请求级别的，因为某些推理解析器依赖于每个请求的 chat-template 参数。
        self.reasoner_cls: type[ReasoningParser] | None = None
        self.vllm_config = vllm_config

        # When in external_launcher mode, async grammar compilation causes deadlocks
        # due to external_launcher mode having a scheduler for each TP rank.
        # Async grammar compilation causes the
        # WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR → WAITING transition to
        # happen at different times on different TP ranks,
        # breaking the determinism assumption that external_launcher relies on.
        #
        # 在 external_launcher 模式下，异步语法编译会导致死锁。
        # 原因是 external_launcher 模式为每个张量并行（TP）rank 都有独立的调度器，
        # 异步编译会导致 WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR → WAITING 的状态转换
        # 在不同 TP rank 上发生的时间不同，破坏了 external_launcher 所依赖的确定性假设。
        self._use_async_grammar_compilation = (
            vllm_config.parallel_config.distributed_executor_backend
            != "external_launcher"
        )

        self._grammar_bitmask: torch.Tensor | None = None
        # 全掩码值（-1 的 int32 表示），用于标记不需要约束的位置
        self._full_mask = torch.tensor(-1, dtype=torch.int32)

        # 并行填充位掩码的阈值和批次大小配置
        # 当批次大小超过此阈值时，使用多线程并行填充位掩码
        max_batch_size = self.vllm_config.scheduler_config.max_num_seqs
        self.fill_bitmask_parallel_threshold = 128
        if self.fill_bitmask_parallel_threshold < max_batch_size:
            # 并行填充时每个子批次的大小
            self.fill_bitmask_parallel_batch_size = 16
            # Use:
            # - at least 1 CPU
            # - at most half the number of CPUs or 8, whichever is less
            # 线程池大小：至少1个线程，最多为CPU数的一半或8个（取较小值）
            max_workers = max(1, min(multiprocessing.cpu_count() // 2, 8))
            self.executor_for_fillmask = ThreadPoolExecutor(max_workers=max_workers)

        if not self.vllm_config.model_config.skip_tokenizer_init:
            # The default max_workers if not specified is the number of
            # CPUs * 5, which is way too high since these tasks are CPU-bound,
            # not I/O bound. We also know we would never dominate CPU usage
            # with just grammar compilation, so we set it to half the number
            # of CPUs.
            # 语法编译线程池的大小设置为 CPU 数的一半。
            # 默认值（CPU数*5）过高，因为语法编译是 CPU 密集型任务而非 I/O 密集型。
            max_workers = max(1, (multiprocessing.cpu_count() + 1) // 2)
            self.executor = ThreadPoolExecutor(max_workers=max_workers)
            # 加载分词器，用于语法编译时的 token 化
            self.tokenizer = cached_tokenizer_from_config(
                model_config=self.vllm_config.model_config
            )
            # 加载推理解析器插件（如果配置了自定义插件路径）
            reasoning_parser_plugin = (
                self.vllm_config.structured_outputs_config.reasoning_parser_plugin
            )
            if reasoning_parser_plugin and len(reasoning_parser_plugin) > 3:
                ReasoningParserManager.import_reasoning_parser(reasoning_parser_plugin)

            # 获取推理解析器类，用于在推理（thinking）模式下判断何时结束推理阶段
            reasoning_parser = (
                self.vllm_config.structured_outputs_config.reasoning_parser
            )
            if reasoning_parser:
                self.reasoner_cls = ReasoningParserManager.get_reasoning_parser(
                    reasoning_parser
                )

        # 是否在推理（thinking）阶段也启用结构化输出约束
        # 如果为 True，则模型在推理阶段也会被语法约束
        self.enable_in_reasoning = (
            self.vllm_config.structured_outputs_config.enable_in_reasoning
        )

    def _get_reasoner(self, request: "Request") -> "ReasoningParser | None":
        """获取请求关联的推理解析器实例。

        # 使用延迟初始化策略，首次调用时根据请求参数创建解析器实例。
        # 每个请求维护独立的解析器实例，因为不同请求可能有不同的模板参数。
        """
        structured_req = request.structured_output_request
        if structured_req is None or self.reasoner_cls is None:
            return None

        if structured_req.reasoner is None:
            # Lazily build the request-local parser so the structured-output
            # gate observes the same template kwargs used by the frontend.
            # 延迟构建请求级别的解析器，使结构化输出门控使用与前端相同的模板参数。
            parser_kwargs = structured_req.reasoning_parser_kwargs or {}
            structured_req.reasoner = self.reasoner_cls(
                tokenizer=self.tokenizer,
                **parser_kwargs,
            )
        return structured_req.reasoner

    def grammar_init(self, request: "Request") -> None:
        """初始化请求的语法编译任务。

        # 此方法在请求首次需要结构化输出时被调用，执行以下步骤：
        # 1. 检查请求是否需要结构化输出
        # 2. 首次调用时初始化对应的后端（xgrammar/guidance/outlines/lm-format-enforcer）
        # 3. 提交语法编译任务（异步或同步）
        # 4. 将编译结果（Future 或已完成的 Grammar）存储在请求中
        """
        if request.structured_output_request is None:
            return

        if TYPE_CHECKING:
            assert (
                request.sampling_params is not None
                and request.sampling_params.structured_outputs is not None
            )

        # Initialize the backend the first time it is needed.
        #
        # NOTE: We only support a single backend. We do NOT support different
        # backends on a per-request basis in V1 (for now, anyway...).
        # _backend is set in Processor._validate_structured_output
        #
        # 首次需要时初始化后端。
        # 注意：V1 中只支持单一后端，不支持不同请求使用不同后端。
        # _backend 在 Processor._validate_structured_output 中设置。
        if self.backend is None:
            assert request.sampling_params is not None
            backend = request.sampling_params.structured_outputs._backend
            vocab_size = self.vllm_config.model_config.get_vocab_size()
            # 根据配置的后端类型创建对应的后端实例
            if backend == "xgrammar":
                self.backend = XgrammarBackend(
                    self.vllm_config,
                    tokenizer=self.tokenizer,
                    vocab_size=vocab_size,
                )
            elif backend == "guidance":
                self.backend = GuidanceBackend(
                    self.vllm_config,
                    tokenizer=self.tokenizer,
                    vocab_size=vocab_size,
                )
            elif backend == "outlines":
                from vllm.v1.structured_output.backend_outlines import OutlinesBackend

                self.backend = OutlinesBackend(
                    self.vllm_config,
                    tokenizer=self.tokenizer,
                    vocab_size=vocab_size,
                )
            elif backend == "lm-format-enforcer":
                from vllm.v1.structured_output.backend_lm_format_enforcer import (  # noqa: E501
                    LMFormatEnforcerBackend,
                )

                self.backend = LMFormatEnforcerBackend(
                    self.vllm_config,
                    tokenizer=self.tokenizer,
                    vocab_size=vocab_size,
                )
            else:
                raise ValueError(f"Unsupported structured output backend: {backend}")

        # 根据配置决定使用异步还是同步方式编译语法
        # 异步编译可以避免阻塞引擎主循环
        if self._use_async_grammar_compilation:
            grammar = self.executor.submit(self._create_grammar, request)
        else:
            grammar = self._create_grammar(request)  # type: ignore[assignment]
        request.structured_output_request.grammar = grammar  # type: ignore[assignment]

    def _create_grammar(self, request: "Request") -> StructuredOutputGrammar:
        """创建语法对象。

        # 根据请求的结构化输出键（类型 + 规范）编译语法。
        # 请求在引擎核心客户端中已经过验证，因此此处的请求类型是受支持的。
        """
        key = request.structured_output_request.structured_output_key  # type: ignore[union-attr]

        # Note that the request was validated in the engine core client,
        # so at this point we know it is a supported type of request.
        #
        # TODO: we still need to handle xgrammar compilation failures,
        # though it should be unlikely as we test that up front as well.
        request_type, grammar_spec = key

        assert self.backend is not None
        return self.backend.compile_grammar(request_type, grammar_spec)

    def _fill_bitmasks(
        self, batch: Iterable[tuple[StructuredOutputGrammar, int, bool]]
    ) -> None:
        """为批次中的请求填充语法位掩码。

        # 对于需要应用位掩码且未终止的语法，调用语法对象的 fill_bitmask 方法；
        # 对于不需要约束的位置，填充全掩码（-1）表示允许所有 token。
        """
        assert self._grammar_bitmask is not None
        for grammar, index, apply_bitmask in batch:
            if apply_bitmask and not grammar.is_terminated():
                grammar.fill_bitmask(self._grammar_bitmask, index)
            else:
                # Note that for thinking support, we will need to
                # reset the relevant part of the bitmask for consequent
                # requests here.
                # 对于不需要约束的位置（如推理阶段），填充全掩码允许所有 token。
                self._grammar_bitmask[index].fill_(self._full_mask)

    def _async_submit_fill_bitmask(
        self, batch: list[tuple[StructuredOutputGrammar, int, bool]]
    ) -> Future:
        """异步提交位掩码填充任务到线程池。"""
        return self.executor_for_fillmask.submit(self._fill_bitmasks, batch)

    def grammar_bitmask(
        self,
        requests: dict[str, "Request"],
        structured_output_request_ids: list[str],
        scheduled_spec_decode_tokens: dict[str, list[int]],
    ) -> "npt.NDArray[np.int32] | None":
        """为当前批次生成结构化输出的语法位掩码。

        # 此方法在调度器每次迭代时被调用，为所有需要结构化输出的请求生成位掩码。
        #
        # 位掩码的工作原理：
        # - 每个请求对应位掩码中的一行或多行
        # - 每行是一个整数数组，每个 bit 对应词表中的一个 token
        # - bit 为 1 表示该 token 被语法允许，为 0 表示被禁止
        # - 在解码时，被禁止的 token 的 logit 会被设为 -inf
        #
        # 当启用投机解码（speculative decoding）时，每个请求需要多个位掩码行：
        # 一个用于当前 token，其余用于每个投机 token 位置。
        #
        # 处理流程（步骤编号）：
        # 1. 检查是否有结构化输出请求，没有则返回 None
        # 2. 获取投机解码的最大 token 数（用于计算每个请求需要的行数）
        # 3. 首次调用时分配位掩码张量（复用后续迭代）
        # 4. 遍历所有结构化输出请求，为每个请求填充位掩码：
        #    a. 判断是否应填充位掩码（考虑推理模式）
        #    b. 对于投机解码，需要为每个投机 token 位置填充位掩码
        #    c. 填充后需要回滚语法状态（因为投机 token 可能被拒绝）
        # 5. 使用并行或串行方式填充位掩码（根据批次大小选择）
        # 6. 将位掩码转换为 numpy 数组返回（序列化效率更高）
        #
        # 两种填充策略：
        # - 并行填充：当批次大小超过阈值且未启用投机解码时使用
        #   将批次分成子批次，使用线程池并行填充
        # - 串行填充：小批次或启用投机解码时使用
        #   逐个请求填充，同时处理投机 token 的语法状态推进和回滚
        #
        # Returns:
        #     numpy 数组形式的位掩码，或 None（如果没有结构化输出请求）
        """
        # Prepare the structured output bitmask for this batch.
        if not structured_output_request_ids:
            return None

        # 获取投机解码的最大 token 数，用于计算每个请求需要的位掩码行数
        max_num_spec_tokens = 0
        if self.vllm_config.speculative_config is not None:
            max_num_spec_tokens = (
                self.vllm_config.speculative_config.num_speculative_tokens
            )

        if self._grammar_bitmask is None:
            assert self.backend is not None
            max_batch_size = self.vllm_config.scheduler_config.max_num_seqs

            # Allocate a bitmask for each token needing to be checked:
            # one for each speculative position, and one more for the
            # bonus token / non-speculative token.
            # 为需要检查的每个 token 位置分配位掩码：
            # 每个投机位置一个，再加一个用于 bonus token 或非投机 token。
            self._grammar_bitmask = self.backend.allocate_token_bitmask(
                max_batch_size * (1 + max_num_spec_tokens)
            )

        # Generate a batched bitmask for all structured output requests.
        # When speculative decoding is enabled, we need to include multiple
        # masks for each request, one for each possible bonus token position.
        # These are stored inline in the tensor and unpacked by the gpu runner.
        cumulative_index = 0

        # Optimized parallel filling of bitmasks for
        # non-spec, large-batch-size cases
        # 当批次较大且未启用投机解码时，使用并行填充优化
        if (
            len(structured_output_request_ids) > self.fill_bitmask_parallel_threshold
            and max_num_spec_tokens == 0
        ):
            # 步骤 4a：并行填充策略
            # 将请求分成多个子批次，使用线程池并行填充位掩码。
            # 这种策略在大批量场景下可以显著降低 CPU 开销。
            promises = []
            batch = []
            for req_id in structured_output_request_ids:
                request = requests[req_id]
                structured_output_request = request.structured_output_request
                if TYPE_CHECKING:
                    assert structured_output_request is not None
                    assert structured_output_request.grammar is not None
                grammar = structured_output_request.grammar

                # 判断是否应为此请求填充位掩码（考虑推理模式）
                apply_bitmask = self.should_fill_bitmask(request)
                batch.append((grammar, cumulative_index, apply_bitmask))
                # 按固定批次大小提交并行任务
                if len(batch) == self.fill_bitmask_parallel_batch_size:
                    promises.append(self._async_submit_fill_bitmask(batch))
                    batch = []

                cumulative_index += 1
            if batch:
                promises.append(self._async_submit_fill_bitmask(batch))

            # Wait for all bitmask filling tasks to complete.
            # 等待所有并行位掩码填充任务完成
            for promise in promises:
                promise.result()
        else:
            # Fallback to serial filling of bitmasks for small-batch-size cases
            # 小批次或启用投机解码时使用串行填充
            #
            # 步骤 4b：串行填充策略
            # 对于每个请求，需要处理投机 token 的语法状态推进：
            # 1. 先为每个投机 token 位置填充位掩码
            # 2. 接受投机 token 并推进语法状态（模拟投机解码的接受）
            # 3. 为主 token 位置填充位掩码
            # 4. 最后回滚语法状态（因为投机 token 可能被拒绝）
            #
            # 这种"先推进后回滚"的策略确保位掩码反映了
            # 如果投机 token 被接受后的语法状态。
            for req_id in structured_output_request_ids:
                request = requests[req_id]
                structured_output_request = request.structured_output_request

                if TYPE_CHECKING:
                    assert structured_output_request is not None
                    assert structured_output_request.grammar is not None
                grammar = structured_output_request.grammar
                apply_bitmask = self.should_fill_bitmask(request)

                state_advancements = 0
                req_tokens = scheduled_spec_decode_tokens.get(req_id, ())
                # 遍历投机 token 和主 token（-1 作为分隔标记）
                # itertools.chain 将投机 token 列表和 [-1] 连接起来
                for token in itertools.chain(req_tokens, (-1,)):
                    self._fill_bitmasks(((grammar, cumulative_index, apply_bitmask),))
                    if token == -1:
                        # Stop advancing the grammar once we hit a padding token.
                        # 遇到填充标记（-1）后停止推进语法状态，
                        # 因为 -1 是主 token 的分隔标记，不是实际的投机 token。
                        apply_bitmask = False
                    if apply_bitmask and not grammar.is_terminated():
                        # 接受投机 token 并推进语法状态
                        # 这模拟了投机解码中草稿 token 被接受的场景
                        accepted = grammar.accept_tokens(req_id, [token])
                        assert accepted, (token, req_id, scheduled_spec_decode_tokens)
                        state_advancements += 1
                    cumulative_index += 1
                if state_advancements > 0:
                    # 回滚语法状态，因为投机 token 可能被拒绝
                    # 实际的接受/拒绝在投机解码验证阶段决定
                    grammar.rollback(state_advancements)

        bitmask_tensor = self._grammar_bitmask
        if cumulative_index < bitmask_tensor.shape[0]:
            bitmask_tensor = bitmask_tensor[:cumulative_index]

        # After finishing with the xgrammar operations, we convert to
        # np.ndarray, because that is much more efficient for serialization
        # and deserialization when sending this to the GPU workers.
        # 转换为 numpy 数组，因为序列化/反序列化比 tensor 更高效
        return bitmask_tensor.numpy()

    def should_fill_bitmask(self, request: "Request") -> bool:
        """判断是否应为请求填充语法位掩码。

        # 此方法决定了当前步骤是否应该为该请求生成语法位掩码。
        # 位掩码用于在采样时过滤不允许的 token。
        #
        # 判断逻辑（按优先级）：
        # 1. 如果有推理解析器（推理模式）：
        #    a. 如果启用了推理阶段约束（enable_in_reasoning），始终返回 True
        #    b. 如果推理尚未结束（reasoning_ended=None），延迟检测推理状态
        #    c. 返回推理是否已结束的判断结果
        # 2. 如果没有推理解析器（非推理模式），返回 True（始终启用约束）
        #
        # 与 should_advance 的区别：
        # - should_advance 决定是否推进 FSM 状态
        # - should_fill_bitmask 决定是否生成位掩码
        # - 在推理阶段，可能不需要推进 FSM，但仍需要生成全允许的位掩码
        #   以确保采样不受约束
        """
        # NOTE (Hanchen) if enable_in_reasoning is True, it means that
        # the model needs to be constrained in reasoning. So we should always
        # enable the bitmask filling.
        reasoner = self._get_reasoner(request)
        if reasoner is not None:
            if self.enable_in_reasoning:
                return True
            assert request.structured_output_request is not None
            if request.structured_output_request.reasoning_ended is None:
                # This should be removed here, but since `openai_gptoss`
                # is an independent code path, it is kept for now.
                # After unifying the `openai_gptoss` and non-`openai_gptoss` styles,
                # it can be removed.
                # 延迟检测推理是否已结束
                # 使用 prompt_token_ids 进行初始检测（首次调用时）
                request.structured_output_request.reasoning_ended = (
                    reasoner.is_reasoning_end(request.prompt_token_ids or [])
                )
            return request.structured_output_request.reasoning_ended
        return True

    def should_advance(self, request: "Request") -> bool:
        """判断是否应推进语法状态机（FSM）。

        # 此方法决定了当前步骤是否应该推进语法 FSM。
        # 它在推理（thinking/reasoning）模式下尤为重要。
        #
        # 决策逻辑（按优先级）：
        # 1. 如果请求不使用结构化输出，返回 False
        # 2. 如果没有推理解析器（非推理模式），返回 True（始终推进）
        # 3. 如果启用了推理阶段约束（enable_in_reasoning），返回 True
        # 4. 如果推理已结束（reasoning_ended=True），返回 True
        # 5. 检查推理是否在当前步骤中结束：
        #    a. 对于 JSON/regex/choice/grammar：延迟到下一步再推进
        #       原因：在关闭边界 token 上推进可能接受仍属于推理流的 token
        #    b. 对于结构化标签（STRUCTURAL_TAG）：可以在同一步骤内推进
        #       原因：结构化标签建模了分阶段输出（如 thinking tag -> answer tag），
        #       投机解码必须在转换后立即对草稿 token 运行 grammar.validate_tokens
        # 6. 其他情况返回 False
        #
        # 设计原因：
        # - 推理模式下，模型先输出推理过程（thinking），再输出最终答案
        # - 推理过程不应被结构化约束（可能是自由文本）
        # - 只有最终答案部分才需要应用结构化约束
        # - 因此需要准确判断推理何时结束，以决定何时开始约束
        """
        if not request.use_structured_output:
            return False

        # To determine whether we can advance the FSM.
        # Supports thinking usage where we skip the reasoning components.
        if TYPE_CHECKING:
            assert request.structured_output_request is not None
            assert request.structured_output_request.grammar is not None
        # by default, we should always advance
        # for cases that don't use thinking mode.
        reasoner = self._get_reasoner(request)
        if reasoner is None:
            return True

        # if the model needs structured in reasoning, we should advance
        if self.enable_in_reasoning:
            return True

        structured_req = request.structured_output_request
        if structured_req.reasoning_ended:
            return True

        # Check if reasoning ends in *this* step
        # 检查推理是否在当前步骤中结束
        # 使用流式检测：只检查最近生成的 token 序列
        delta_from = request.num_computed_tokens - request.num_output_placeholders
        all_token_ids = request.all_token_ids
        start = (
            delta_from if delta_from >= 0 else max(len(all_token_ids) + delta_from, 0)
        )
        if reasoner.is_reasoning_end_streaming(
            all_token_ids, itertools.islice(all_token_ids, start, None)
        ):
            structured_req.reasoning_ended = True

            # Reasoning just ended this step. Defer FSM advance until the next
            # pass (see reasoning_ended check above) for JSON/regex/choice/grammar:
            # advancing on the closing boundary token can accept tokens that still
            # belong to the reasoning stream. Structural tags are the only safe
            # same-step exception: they model phased output (e.g. thinking tag ->
            # answer tag), and speculative decoding must run grammar.validate_tokens
            # on draft tokens produced immediately after that transition.
            #
            # 推理在当前步骤结束。对于 JSON/regex/choice/grammar 类型，
            # 延迟到下一步再推进 FSM，因为在关闭边界 token 上推进可能接受
            # 仍属于推理流的 token。结构化标签是唯一的同步骤例外：
            # 它们建模了分阶段输出（如 thinking tag -> answer tag），
            # 投机解码必须在转换后立即对草稿 token 运行 grammar.validate_tokens。
            if (
                self.vllm_config.speculative_config is not None
                and structured_req.structured_output_key[0]
                == StructuredOutputOptions.STRUCTURAL_TAG
            ):
                return True

        return False

    def clear_backend(self) -> None:
        """清理后端资源。"""
        if self.backend is not None:
            self.backend.destroy()
