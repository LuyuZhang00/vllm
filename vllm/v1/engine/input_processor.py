# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# =============================================================================
# 模块概述: vLLM v1 引擎的输入处理器 (Input Processor)
# =============================================================================
# 本模块负责将用户传入的原始 prompt 转换为引擎核心可调度的 EngineCoreRequest。
#
# 核心组件:
#   - InputProcessor: 输入处理的主类，串联参数校验、分词、多模态特征提取等步骤
#   - InputPreprocessor: 底层预处理器，负责实际的分词和输入格式转换
#
# 处理流程 (process_inputs):
#   1. 参数校验: 验证 SamplingParams / PoolingParams 的合法性
#   2. LoRA 校验: 验证 LoRA 适配器配置
#   3. 输入预处理: 将 prompt 转换为 token IDs 或 prompt_embeds
#      - 文本输入: 通过 tokenizer 分词
#      - 多模态输入: 提取图像/音频特征，生成 mm_features
#      - Encoder-Decoder 模型: 拆分为 encoder 和 decoder 两部分
#   4. 平台验证: 调用平台特定的验证逻辑
#   5. 输入校验: 检查 prompt 长度、词汇表范围等
#   6. 构建 EngineCoreRequest: 组装所有信息为最终请求对象
#
# 多模态处理:
#   - MultiModalFeatureSpec: 多模态特征规范，包含数据、模态类型、位置、哈希等
#   - mm_hashes: 用于多模态缓存的唯一标识，相同哈希的特征可以复用
#   - argsort_mm_positions: 按位置排序多模态特征，确保正确的序列顺序
#
# 设计要点:
#   - process_inputs 在 Input 守护线程中调用，与 GPU 模型前向计算并行执行
#   - 多模态特征通过哈希进行缓存，避免相同图片的重复编码
#   - 请求 ID 在此处分配，添加随机后缀确保唯一性
# =============================================================================

import time
from collections.abc import Mapping
from typing import Any, Literal

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.inputs import (
    EngineInput,
    PromptType,
    SingletonInput,
    split_enc_dec_input,
)
from vllm.inputs.preprocess import InputPreprocessor
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.multimodal.encoder_budget import MultiModalBudget
from vllm.multimodal.inputs import MultiModalFeatureSpec
from vllm.multimodal.utils import argsort_mm_positions
from vllm.platforms import current_platform
from vllm.pooling_params import PoolingParams
from vllm.renderers import BaseRenderer, renderer_from_config
from vllm.sampling_params import SamplingParams
from vllm.tasks import GENERATION_TASKS, POOLING_TASKS, SupportedTask
from vllm.tokenizers import TokenizerLike
from vllm.utils import length_from_prompt_token_ids_or_embeds, random_uuid
from vllm.utils.jsontree import json_iter_leaves
from vllm.v1.engine import EngineCoreRequest

logger = init_logger(__name__)


class InputProcessor:
    # InputProcessor 是 vLLM 引擎的输入处理入口，负责将用户传入的原始 prompt
    # 转换为引擎核心可调度的 EngineCoreRequest 对象。
    # 它串联了参数校验、分词、多模态特征提取、缓存管理等关键步骤。

    def __init__(
        self,
        vllm_config: VllmConfig,
        renderer: BaseRenderer | None = None,
        *,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
    ) -> None:
        self.vllm_config = vllm_config
        self.model_config = model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.scheduler_config = vllm_config.scheduler_config
        self.speculative_config = vllm_config.speculative_config
        self.structured_outputs_config = vllm_config.structured_outputs_config
        self.observability_config = vllm_config.observability_config
        self.use_v2_model_runner = vllm_config.use_v2_model_runner

        self.generation_config_fields = model_config.try_get_generation_config()

        self.renderer = renderer or renderer_from_config(vllm_config)

        self.supports_mm_inputs = mm_registry.supports_multimodal_inputs(model_config)
        self.mm_encoder_cache_size = 0
        self.skip_prompt_length_check = False
        if self.supports_mm_inputs:
            # MultiModalBudget 根据模型配置和多模态注册表，计算编码器缓存的大小。
            # 编码器缓存用于存储多模态编码器输出（如图像 embedding），
            # 这样相同多模态输入可复用编码结果，避免重复计算。
            mm_budget = MultiModalBudget(vllm_config, mm_registry)
            self.mm_encoder_cache_size = mm_budget.encoder_cache_size
            self.skip_prompt_length_check = (
                mm_budget.processor.info.skip_prompt_length_check
            )
            mm_budget.reset_cache()  # Not used anymore

        self.input_preprocessor = InputPreprocessor(
            vllm_config,
            renderer=renderer,
            mm_registry=mm_registry,
        )

    @property
    def tokenizer(self) -> TokenizerLike | None:
        return self.renderer.tokenizer

    def get_tokenizer(self) -> TokenizerLike:
        return self.renderer.get_tokenizer()

    def _validate_params(
        self,
        params: SamplingParams | PoolingParams,
        supported_tasks: tuple[SupportedTask, ...],
    ) -> None:
        """Raise `ValueError` if SamplingParams or PoolingParams is not valid."""
        if isinstance(params, SamplingParams):
            supported_generation_tasks = [
                task for task in supported_tasks if task in GENERATION_TASKS
            ]
            if not supported_generation_tasks:
                raise ValueError("This model does not support generation")

            params.verify(
                self.model_config,
                self.speculative_config,
                self.structured_outputs_config,
                self.tokenizer,
            )

            if params.thinking_token_budget is not None:
                if (
                    self.vllm_config.reasoning_config is None
                    or not self.vllm_config.reasoning_config.enabled
                ):
                    raise ValueError(
                        "thinking_token_budget is set but reasoning_config is "
                        "not configured. Please set --reasoning-parser "
                        "and/or --reasoning-config to use thinking_token_budget."
                    )
                if self.use_v2_model_runner:
                    raise ValueError(
                        "thinking_token_budget is not yet supported by the V2 "
                        "model runner. Run vLLM with VLLM_USE_V2_MODEL_RUNNER=0 "
                        "to use thinking_token_budget."
                    )
        elif isinstance(params, PoolingParams):
            supported_pooling_tasks = [
                task for task in supported_tasks if task in POOLING_TASKS
            ]
            if not supported_pooling_tasks:
                raise ValueError("This model does not support pooling")

            if params.task is None:
                if "token_embed" in supported_pooling_tasks:
                    params.task = "token_embed"
                elif "token_classify" in supported_pooling_tasks:
                    params.task = "token_classify"
                elif "plugin" in supported_pooling_tasks:
                    params.task = "plugin"

            if params.task not in supported_pooling_tasks:
                raise ValueError(
                    f"Unsupported task: {params.task!r} "
                    f"Supported tasks: {supported_pooling_tasks}"
                )

            params.verify(self.model_config)
        else:
            raise TypeError(
                f"params must be either SamplingParams or PoolingParams, "
                f"but got {type(params).__name__}"
            )

    def _validate_lora(self, lora_request: LoRARequest | None) -> None:
        if lora_request is None:
            return

        # LoRA request passed in while LoRA is not enabled
        if not self.lora_config:
            raise ValueError(
                f"Got lora_request {lora_request} but LoRA is not enabled!"
            )

        if self.tokenizer is not None:
            logger.warning_once(
                "vLLM has deprecated support for supporting different "
                "tokenizers for different LoRAs. By default, vLLM uses base "
                "model's tokenizer. If you are using a LoRA "
                "with its own tokenizer, consider specifying `--tokenizer "
                "[lora_path]` to use the LoRA tokenizer."
            )

    def _get_mm_identifier(
        self,
        mm_hash: str,
        lora_request: LoRARequest | None,
    ) -> str:
        """
        When enable_tower_connector_lora is True, multi-modal embeddings
        vary depending on the LoRA request. Therefore, the mm_hash must be
        generated based on the LoRA request to prevent incorrect cache hits.
        """
        if (
            lora_request is None
            or self.lora_config is None
            or not self.lora_config.enable_tower_connector_lora
        ):
            return mm_hash
        return f"{lora_request.lora_name}:{mm_hash}"

    def inject_into_mm_cache(
        self,
        mm_hashes: dict[str, list[str]],
        mm_kwargs: dict[str, list],
    ) -> None:
        # 将已经由外部（如前端）预处理完成的多模态 kwargs 注入缓存。
        # 场景：当前端已通过 HF 处理器完成了多模态张量的预处理并传输到后端时，
        # 需要手动将结果写入缓存，以保证缓存命中率统计的准确性，
        # 并避免后续相同多模态输入的重复处理开销。

        """Inject pre-processed mm_kwargs into the processor cache.

        Call this when mm_kwargs have already been through the HF processor
        externally (e.g. by a frontend that transfers pre-processed tensors
        to the backend).  This ensures MM cache hit rate metrics are reported
        accurately and avoids redundant processing on subsequent requests
        with the same images.

        Uses ``get_and_update_item()`` with an empty prompt_updates list,
        since token expansion has already been handled externally.
        """
        cache = self.renderer.mm_processor_cache
        if cache is None:
            return
        try:
            for modality, hashes in mm_hashes.items():
                items = mm_kwargs.get(modality, [])
                for i, mm_hash in enumerate(hashes):
                    if i < len(items) and items[i] is not None:
                        # Insert into cache via get_and_update_item.
                        # Use the returned item (may be an address for SHM
                        # cache or the original item for LRU cache).
                        items[i], _ = cache.get_and_update_item(
                            (items[i], []),
                            mm_hash,
                        )
            # Update cache stats to reflect the externally processed items
            self.renderer.update_mm_cache_stats()
        except Exception:
            logger.warning(
                "Failed to inject mm_kwargs into processor cache",
                exc_info=True,
            )

    @staticmethod
    def assign_request_id(request: EngineCoreRequest):
        """Replace the externally supplied request ID with an internal request ID
        that adds 8 random characters in order to ensure uniqueness.
        """
        if request.external_req_id is not None:
            raise ValueError(
                "The external_req_id field should not be set on EngineCoreRequests"
                " passed to vLLM; use the request_id field."
            )
        request.external_req_id = request.request_id
        if envs.VLLM_DISABLE_REQUEST_ID_RANDOMIZATION:
            logger.warning_once(
                "VLLM_DISABLE_REQUEST_ID_RANDOMIZATION is set and will be "
                "removed in a future release. Duplicate externally-provided "
                "request IDs may cause failures and/or subtle correctness errors."
            )
        else:
            request.request_id = f"{request.external_req_id}-{random_uuid():.8}"

    def process_inputs(
        self,
        request_id: str,
        prompt: PromptType | EngineInput,
        params: SamplingParams | PoolingParams,
        supported_tasks: tuple[SupportedTask, ...],
        arrival_time: float | None = None,
        lora_request: LoRARequest | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
        trace_headers: Mapping[str, str] | None = None,
        priority: int = 0,
        data_parallel_rank: int | None = None,
        resumable: bool = False,
    ) -> EngineCoreRequest:
        # 输入预处理主流程，将用户 prompt 转换为引擎核心可调度的 EngineCoreRequest。
        #
        # 完整流水线:
        #   步骤 1 - 参数校验:
        #     验证 SamplingParams / PoolingParams 的合法性，
        #     包括模型是否支持生成/池化任务、参数值范围等。
        #
        #   步骤 2 - LoRA 校验:
        #     验证 LoRA 适配器配置是否与引擎配置一致。
        #
        #   步骤 3 - 输入预处理:
        #     根据输入类型选择不同的处理路径:
        #     - EngineInput (已处理): 直接使用，跳过分词
        #     - PromptType (原始输入): 通过 InputPreprocessor 分词/预处理
        #     对于 Encoder-Decoder 模型，将输入拆分为 encoder 和 decoder 两部分，
        #     因为编码器和解码器有各自独立的序列长度限制和输入格式要求。
        #
        #   步骤 4 - 平台验证:
        #     调用平台特定的验证逻辑（如 CUDA 设备检查）。
        #
        #   步骤 5 - 采样参数处理:
        #     克隆 SamplingParams 以避免修改原始对象，
        #     设置默认 max_tokens、更新 generation config 等。
        #
        #   步骤 6 - 多模态特征处理:
        #     对于多模态输入（图像、音频等）:
        #     - 按位置排序多模态特征（argsort_mm_positions）
        #     - 生成唯一标识符（mm_hash）用于缓存
        #     - 构建 MultiModalFeatureSpec 列表
        #
        #   步骤 7 - 输入校验:
        #     检查 prompt 长度是否超过 max_model_len，
        #     检查 token ID 是否在词汇表范围内。
        #
        #   步骤 8 - 构建 EngineCoreRequest:
        #     组装所有信息为最终请求对象，包括 token IDs、多模态特征、
        #     采样参数、到达时间、优先级等。
        # 输入预处理主流程，将用户 prompt 转换为引擎核心可调度的 EngineCoreRequest。
        # 流水线步骤：
        #   1. 参数校验（SamplingParams / PoolingParams 合法性、LoRA 配置）
        #   2. 对 Encoder-Decoder 模型，将输入拆分为 encoder 和 decoder 两部分
        #      （split_enc_dec_input），分别校验，因为编码器和解码器有各自独立的
        #      序列长度限制和输入格式要求
        #   3. 分词 / 处理 prompt_embeds
        #   4. 处理多模态特征：按位置排序、合并、生成唯一标识符
        #   5. 构建并返回 EngineCoreRequest 对象
        self._validate_params(params, supported_tasks)
        self._validate_lora(lora_request)

        parallel_config = self.vllm_config.parallel_config
        dp_size = parallel_config.data_parallel_size
        dp_local_size = parallel_config.data_parallel_size_local
        num_ranks = dp_local_size if parallel_config.local_engines_only else dp_size
        if data_parallel_rank is not None and not (0 <= data_parallel_rank < num_ranks):
            raise ValueError(
                f"data_parallel_rank {data_parallel_rank} "
                f"is out of range [0, {num_ranks})."
            )

        if isinstance(prompt, dict) and "type" in prompt:
            if tokenization_kwargs:
                logger.warning_once(
                    "Passing tokenization_kwargs to InputProcessor is deprecated "
                    "and will be removed in v0.18. You should instead pass "
                    "them to Renderer.render_cmpl() or Renderer.render_chat()."
                )

            if arrival_time is None:
                arrival_time = prompt.get("arrival_time", time.time())  # type: ignore[assignment]

            processed_inputs: EngineInput = prompt  # type: ignore[assignment]
        else:
            logger.warning_once(
                "Passing raw prompts to InputProcessor is deprecated "
                "and will be removed in v0.18. You should instead pass "
                "the outputs of Renderer.render_cmpl() or Renderer.render_chat()."
            )

            if arrival_time is None:
                arrival_time = time.time()

            processed_inputs = self.input_preprocessor.preprocess(
                prompt,
                tokenization_kwargs=tokenization_kwargs,
            )

        current_platform.validate_request(processed_inputs, params)

        encoder_inputs, decoder_inputs = split_enc_dec_input(processed_inputs)
        self._validate_model_inputs(encoder_inputs, decoder_inputs)

        # Mypy can be conservative for TypedDict unions; normalize access.
        if decoder_inputs["type"] == "embeds":
            prompt_embeds = decoder_inputs["prompt_embeds"]
            prompt_token_ids = decoder_inputs.get("prompt_token_ids")
            prompt_is_token_ids = decoder_inputs.get("is_token_ids")
        else:
            prompt_token_ids = decoder_inputs["prompt_token_ids"]
            prompt_embeds = None
            prompt_is_token_ids = None

        sampling_params = None
        pooling_params = None
        if isinstance(params, SamplingParams):
            # TODO: can we avoid cloning here in multiproc case?
            sampling_params = params.clone()
            # If unset max tokens, then generate up to the max_model_len.
            if sampling_params.max_tokens is None:
                seq_len = length_from_prompt_token_ids_or_embeds(
                    prompt_token_ids, prompt_embeds
                )
                sampling_params.max_tokens = self.model_config.max_model_len - seq_len

            sampling_params.update_from_generation_config(
                self.generation_config_fields,
                self.renderer.get_eos_token_id(),
            )
            if self.tokenizer is not None:
                sampling_params.update_from_tokenizer(self.tokenizer)
        else:
            pooling_params = params.clone()

        # Multimodal related.
        mm_features: list[MultiModalFeatureSpec] | None = None

        if decoder_inputs["type"] == "multimodal":
            decoder_mm_inputs = decoder_inputs["mm_kwargs"]
            decoder_mm_positions = decoder_inputs["mm_placeholders"]
            decoder_mm_hashes = decoder_inputs["mm_hashes"]

            if not all(
                isinstance(leaf, str) for leaf in json_iter_leaves(decoder_mm_hashes)
            ):
                raise ValueError(
                    f"mm_hashes must contain only strings, got: {decoder_mm_hashes}. "
                    "This is likely due to an incorrect custom implementation of "
                    "MultiModalProcessor.apply method."
                )

            # Merge and flatten multimodal placeholders, hashes and inputs
            # from dictionaries to lists, and sort them by each item's position
            # in the input sequence.
            sorted_mm_idxs = argsort_mm_positions(decoder_mm_positions)

            mm_features = []
            for modality, idx in sorted_mm_idxs:
                base_mm_hash = decoder_mm_hashes[modality][idx]
                mm_features.append(
                    MultiModalFeatureSpec(
                        data=decoder_mm_inputs[modality][idx],
                        modality=modality,
                        identifier=self._get_mm_identifier(
                            base_mm_hash,
                            lora_request,
                        ),
                        mm_position=decoder_mm_positions[modality][idx],
                        mm_hash=base_mm_hash,
                    )
                )

        return EngineCoreRequest(
            request_id=request_id,
            prompt_token_ids=prompt_token_ids,
            prompt_embeds=prompt_embeds,
            prompt_is_token_ids=prompt_is_token_ids,
            mm_features=mm_features,
            sampling_params=sampling_params,
            pooling_params=pooling_params,
            arrival_time=arrival_time,
            lora_request=lora_request,
            cache_salt=decoder_inputs.get("cache_salt"),
            priority=priority,
            data_parallel_rank=data_parallel_rank,
            trace_headers=trace_headers,
            resumable=resumable,
        )

    def _validate_prompt_len(
        self,
        prompt_len: int,
        prompt_type: Literal["encoder", "decoder"],
    ):
        if self.skip_prompt_length_check:
            return

        if prompt_len == 0 and prompt_type == "decoder":
            raise ValueError(f"The {prompt_type} prompt cannot be empty")

        model_config = self.model_config
        max_prompt_len = (
            model_config.max_model_len
            if prompt_type == "decoder"
            else self.mm_encoder_cache_size
        )
        if prompt_len > max_prompt_len:
            if self.supports_mm_inputs:
                suggestion = (
                    "Make sure that `max_model_len` is no smaller than the "
                    "number of text tokens plus multimodal tokens. For image "
                    "inputs, the number of image tokens depends on the number "
                    "of images, and possibly their aspect ratios as well."
                )
            else:
                suggestion = (
                    "Make sure that `max_model_len` is no smaller than the "
                    "number of text tokens."
                )

            raise ValueError(
                f"The {prompt_type} prompt (length {prompt_len}) is "
                f"longer than the maximum model length of {max_prompt_len}. "
                f"{suggestion}"
            )
        elif prompt_len == max_prompt_len and model_config.runner_type == "generate":
            suggestion = (
                "Make sure that `max_model_len` is no smaller than the "
                "number of text tokens (prompt + requested output tokens)."
            )
            raise ValueError(
                f"The {prompt_type} prompt (length {prompt_len}) plus the number of "
                f"requested output tokens (at least 1) is longer than the maximum "
                f"model length of {max_prompt_len}. {suggestion}"
            )

    def _validate_model_input(
        self,
        prompt_input: SingletonInput,
        prompt_type: Literal["encoder", "decoder"],
    ) -> None:
        # 校验单个模型输入（encoder 或 decoder）的合法性，包含三方面：
        #   1. prompt 长度校验：确保不超过 max_model_len（decoder）
        #      或 mm_encoder_cache_size（encoder，即多模态编码器缓存上限）
        #   2. 多模态嵌入大小校验：确保每个模态项的 embedding 数量不超过预分配的
        #      编码器缓存，避免运行时 OOM
        #   3. 词汇表校验：取分词器 vocab size 和模型 vocab size 的较大值作为
        #      有效词汇表范围，兼容 Qwen3 等两者不一致的模型
        model_config = self.model_config
        tokenizer = self.tokenizer

        prompt_ids = (
            None
            if prompt_input["type"] == "embeds"
            else prompt_input["prompt_token_ids"]
        )
        prompt_embeds = (
            prompt_input["prompt_embeds"] if prompt_input["type"] == "embeds" else None
        )

        prompt_len = length_from_prompt_token_ids_or_embeds(prompt_ids, prompt_embeds)
        self._validate_prompt_len(prompt_len, prompt_type)

        if prompt_input["type"] == "multimodal":
            decoder_mm_positions = prompt_input["mm_placeholders"]
            for modality, mm_positions in decoder_mm_positions.items():
                for mm_position in mm_positions:
                    num_embeds = mm_position.get_num_embeds()
                    if num_embeds > self.mm_encoder_cache_size:
                        raise ValueError(
                            f"The {prompt_type} prompt contains a(n) {modality} item "
                            f"with {num_embeds} embedding tokens, which exceeds the "
                            f"pre-allocated encoder cache size "
                            f"{self.mm_encoder_cache_size}. Please reduce the input "
                            f"size or increase the encoder cache size "
                            f"by setting --limit-mm-per-prompt at startup."
                        )

        if prompt_ids and tokenizer is not None:
            max_input_id = max(prompt_ids, default=0)

            # NOTE: tokenizer.max_token_id is the tokenizer’s vocab size while
            # self.model_config.get_vocab_size() is the model’s vocab size.
            # For Qwen3 models, the language model has extra tokens that do
            # not exist in the tokenizer, and vice versa for multimodal
            # placeholder tokens in some multimodal models.
            # See https://github.com/QwenLM/Qwen3/issues/29#issuecomment-1933720399 # noqa: E501
            # and https://github.com/vllm-project/vllm/pull/22471#discussion_r2312251421 # noqa: E501

            # Here we take the max of the two to determine if a token id is
            # truly out-of-vocabulary.
            model_vocab_size = model_config.get_vocab_size()
            if max_input_id > max(tokenizer.max_token_id, model_vocab_size - 1):
                raise ValueError(f"Token id {max_input_id} is out of vocabulary")

    def _validate_model_inputs(
        self,
        encoder_input: SingletonInput | None,
        decoder_input: SingletonInput,
    ):
        if encoder_input is not None:
            self._validate_model_input(encoder_input, prompt_type="encoder")

        self._validate_model_input(decoder_input, prompt_type="decoder")
