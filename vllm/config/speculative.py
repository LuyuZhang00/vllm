# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# =============================================================================
# 文件: speculative.py
# 功能: 投机解码(Speculative Decoding)的配置模块
#
# 【投机解码概述】
# 投机解码是一种加速LLM推理的技术。核心思想是:
#   1. 使用一个较小的"草稿模型"(draft model)快速生成多个候选token
#   2. 使用目标模型(target model)并行验证这些候选token
#   3. 接受正确的token，拒绝错误的token
# 这样可以在保证输出质量的前提下，显著提升推理速度。
#
# 【支持的投机方法】
# - ngram: 基于N-gram的提示查找，无需额外模型
# - medusa: Medusa方法，使用多个预测头
# - mlp_speculator: MLP投机器
# - draft_model: 使用独立的草稿模型
# - eagle/eagle3: EAGLE系列方法，使用辅助隐藏状态
# - mtp: Multi-Token Prediction，多token预测
# - suffix: 后缀解码方法
# - dflash: DFlash方法，支持并行草稿生成
# - custom_class: 自定义投机类
#
# 【配置流程】
# 1. 用户指定投机方法(method)和相关参数
# 2. __post_init__ 根据method初始化草稿模型配置
# 3. 验证并设置张量并行大小
# 4. 创建草稿模型的并行配置
# =============================================================================

import copy
from typing import TYPE_CHECKING, Any, Literal, get_args

from pydantic import Field, SkipValidation, field_validator, model_validator
from typing_extensions import Self

from vllm.config import LoadConfig
from vllm.config.kernel import MoEBackend
from vllm.config.model import ModelConfig
from vllm.config.parallel import ParallelConfig
from vllm.config.utils import config
from vllm.logger import init_logger
from vllm.transformers_utils.config import get_hf_text_config
from vllm.utils.hashing import safe_hash
from vllm.utils.import_utils import LazyLoader, has_arctic_inference
from vllm.v1.attention.backends.registry import AttentionBackendEnum

if TYPE_CHECKING:
    from transformers import PretrainedConfig

    import vllm.model_executor.layers.quantization as me_quant
else:
    # 延迟加载量化模块，避免循环导入
    PretrainedConfig = Any

    me_quant = LazyLoader(
        "model_executor", globals(), "vllm.model_executor.layers.quantization"
    )

logger = init_logger(__name__)

# =============================================================================
# 类型定义: 投机解码方法的类型字面量
# =============================================================================

# MTP (Multi-Token Prediction) 模型类型列表
# 这些是支持多token预测的模型架构，用于投机解码
MTPModelTypes = Literal[
    "deepseek_mtp",       # DeepSeek系列MTP
    "mimo_mtp",           # MiMo系列MTP
    "mimo_v2_mtp",        # MiMo V2系列MTP
    "glm4_moe_mtp",       # GLM4 MoE MTP
    "glm4_moe_lite_mtp",  # GLM4 MoE Lite MTP
    "glm_ocr_mtp",        # GLM OCR MTP
    "ernie_mtp",          # ERNIE MTP
    "nemotron_h_mtp",     # Nemotron H MTP
    "exaone_moe_mtp",     # EXAONE MoE MTP
    "exaone4_5_mtp",      # EXAONE 4.5 MTP
    "qwen3_next_mtp",     # Qwen3 Next MTP
    "qwen3_5_mtp",        # Qwen3.5 MTP
    "longcat_flash_mtp",  # LongCat Flash MTP
    "mtp",                # 通用MTP类型
    "pangu_ultra_moe_mtp",# Pangu Ultra MoE MTP
    "step3p5_mtp",        # Step3.5 MTP
    "hy_v3_mtp",          # HY V3 MTP
    "gemma4_mtp",         # Gemma4 MTP
]

# N-gram GPU方法类型
NgramGPUTypes = Literal["ngram_gpu"]

# DFlash方法类型
DFlashModelTypes = Literal["dflash"]

# EAGLE系列模型类型 (包含MTP和DFlash)
# EAGLE是一种使用辅助隐藏状态进行投机的方法
EagleModelTypes = Literal[
    "eagle", "eagle3", "extract_hidden_states", MTPModelTypes, DFlashModelTypes
]

# 所有支持的投机方法
SpeculativeMethod = Literal[
    "ngram",           # N-gram提示查找
    "medusa",          # Medusa多头预测
    "mlp_speculator",  # MLP投机器
    "draft_model",     # 独立草稿模型
    "suffix",          # 后缀解码
    "custom_class",    # 自定义投机类
    EagleModelTypes,   # EAGLE系列方法
    NgramGPUTypes,     # N-gram GPU方法
]

# 拒绝采样方法
# - standard: 标准概率拒绝采样
# - synthetic: 合成拒绝采样，使用衰减概率
RejectionSampleMethod = Literal["standard", "synthetic"]

# 草稿采样方法
# - greedy: 贪婪采样，总是选择概率最高的token
# - probabilistic: 概率采样，从分布中随机采样
DraftSampleMethod = Literal["greedy", "probabilistic"]


@config
class SpeculativeConfig:
    """投机解码配置类。

    【类功能概述】
    该类负责配置投机解码(Speculative Decoding)的所有参数。
    投机解码通过以下流程加速推理:
    1. 草稿模型(或方法)快速生成多个候选token
    2. 目标模型并行验证这些候选token
    3. 根据验证结果接受或拒绝token

    【主要配置项】
    - method: 投机方法类型 (ngram/medusa/eagle/mtp等)
    - model: 草稿模型名称或路径
    - num_speculative_tokens: 每次投机的token数量
    - draft_tensor_parallel_size: 草稿模型的张量并行度

    【初始化流程】
    1. __post_init__ 根据用户参数推断投机方法
    2. 根据method类型初始化草稿模型配置
    3. 验证张量并行配置
    4. 创建草稿模型的并行配置
    """

    # =========================================================================
    # 基础控制参数
    # =========================================================================

    enforce_eager: bool | None = None
    """是否强制使用eager模式执行。
    覆盖目标模型的默认enforce_eager设置。
    某些投机方法(如DeepSeek V32 MTP)不支持CUDA Graph，需要设置为True。"""

    # General speculative decoding control
    num_speculative_tokens: int = Field(default=None, gt=0)  # type: ignore[assignment]
    """投机token数量。
    每次投机解码尝试生成的候选token数量。
    如果草稿模型配置中有n_predict参数，会使用该值作为默认值。
    较大的值可以提高吞吐量，但也会增加验证开销。"""

    model: str | None = None
    """草稿模型的名称或路径。
    可以是:
    - HuggingFace模型名称 (如 "yuhuili/EAGLE-LLaMA3-Instruct-8B")
    - 本地模型路径
    - 特殊值: "ngram", "suffix", "extract_hidden_states"
    - 自定义类路径 (如 "my_module.MyProposer")"""

    method: SpeculativeMethod | None = None
    """投机解码方法名称。
    支持的方法:
    - ngram: 基于N-gram的提示查找，无需额外模型
    - medusa: Medusa多头预测方法
    - mlp_speculator: MLP投机器
    - draft_model: 使用独立的草稿模型
    - eagle/eagle3: EAGLE系列方法，使用辅助隐藏状态
    - mtp: Multi-Token Prediction，多token预测
    - suffix: 后缀解码方法
    - dflash: DFlash方法，支持并行草稿生成
    - custom_class: 自定义投机类

    如果用户提供了model参数，会尝试自动检测方法类型。
    如果未提供model，则必须显式指定method。"""

    draft_tensor_parallel_size: int | None = Field(default=None, ge=1)
    """草稿模型的张量并行度。
    限制条件:
    - 只能是1或与目标模型相同的张量并行度
    - mlp_speculator方法强制为1
    - 默认使用目标模型的张量并行度"""

    tensor_parallel_size: int | None = None
    """【已弃用】请使用draft_tensor_parallel_size。
    此参数仅用于在用户错误使用时发出警告。"""

    # =========================================================================
    # 草稿模型配置参数
    # =========================================================================

    # Draft model configuration
    quantization: me_quant.QuantizationMethods | str | None = None
    """草稿模型的量化方法。
    如果为None，假设模型权重未量化。
    仅在使用基于草稿模型的投机方法时生效。
    示例: "fp8", "awq", "gptq"等"""

    moe_backend: MoEBackend | None = None
    """草稿模型的MoE后端。
    当为None时，草稿模型继承目标模型的MoE后端设置。
    适用场景: 目标模型使用量化MoE，但草稿模型使用未量化MoE。"""

    attention_backend: AttentionBackendEnum | None = None
    """草稿模型的注意力后端。
    当为None时，自动选择后端。
    适用场景: DFlash需要非因果注意力后端(如FLASH_ATTN)。"""

    max_model_len: int | None = Field(default=None, ge=1)
    """草稿模型的最大序列长度。
    用于测试某些序列是否可以跳过投机。
    如果未指定，使用草稿模型配置中的默认值。"""

    revision: str | None = None
    """草稿模型的特定版本。
    可以是分支名、标签名或提交ID。
    如果未指定，使用默认版本。"""

    code_revision: str | None = None
    """草稿模型代码的特定版本(来自Hugging Face Hub)。
    可以是分支名、标签名或提交ID。
    如果未指定，使用默认版本。"""

    # =========================================================================
    # 高级控制参数
    # =========================================================================

    # Advanced control
    disable_padded_drafter_batch: bool = False
    """是否禁用投机解码的输入填充。
    当设置为True时，投机输入批次可以包含不同长度的序列。
    这可能只被某些注意力后端支持。
    目前仅影响EAGLE方法。"""

    use_local_argmax_reduction: bool = False
    """是否使用词汇表并行的本地argmax归约。
    当设置为True时，使用本地argmax代替全收集完整logits。
    这将通信量从O(vocab_size)减少到O(2 * tp_size)每token。
    仅适用于非树投机中的贪婪草稿选择。"""

    # =========================================================================
    # Ngram投机器配置
    # =========================================================================

    # Ngram proposer configuration
    prompt_lookup_max: int | None = Field(default=None, ge=1)
    """Ngram投机器的最大token窗口大小。
    当method设置为ngram时必需。
    控制在提示中查找匹配n-gram的最大长度。"""

    prompt_lookup_min: int | None = Field(default=None, ge=1)
    """Ngram投机器的最小token窗口大小。
    如果未提供，默认为1。
    控制在提示中查找匹配n-gram的最小长度。"""

    # =========================================================================
    # 替代草稿策略
    # =========================================================================

    # Alternative drafting strategies
    parallel_drafting: bool = False
    """是否启用并行草稿生成。
    当设置为True时，所有投机token并行生成而不是顺序生成。
    这可以提高性能，但要求投机模型被训练支持并行草稿。
    仅兼容EAGLE和draft_model方法。"""

    # =========================================================================
    # 引擎传入的必需配置参数
    # =========================================================================

    # required configuration params passed from engine
    target_model_config: SkipValidation[ModelConfig] = None  # type: ignore
    """目标模型的配置。
    由引擎在初始化时传入，用于配置草稿模型。"""

    target_parallel_config: SkipValidation[ParallelConfig] = None  # type: ignore
    """目标模型的并行配置。
    由引擎在初始化时传入，用于创建草稿模型的并行配置。"""

    # =========================================================================
    # 后初始化阶段生成的参数
    # =========================================================================

    # params generated in the post-init stage
    draft_model_config: SkipValidation[ModelConfig] = None  # type: ignore
    """草稿模型的配置。
    在__post_init__中根据method和model参数初始化。"""

    draft_parallel_config: SkipValidation[ParallelConfig] = None  # type: ignore
    """草稿模型的并行配置。
    在__post_init__中根据目标模型的并行配置创建。"""

    # =========================================================================
    # 后缀解码配置
    # =========================================================================

    # Suffix decoding configuration
    suffix_decoding_max_tree_depth: int = 24
    """后缀解码的最大树深度。
    树深度限制了前缀匹配和投机长度的总和。
    较大的值可以提高匹配机会，但会增加内存使用。"""

    suffix_decoding_max_cached_requests: int = 10000
    """全局后缀树中缓存的最大请求数。
    如果超过此数量，将按FIFO顺序触发驱逐。
    如果设置为0，全局后缀树被禁用，过去响应不会被缓存(提示树仍然使用)。"""

    suffix_decoding_max_spec_factor: float = 1.0
    """后缀解码的最大投机因子。
    投机因子根据前缀匹配长度控制投机长度:
    max_spec_tokens = max_spec_factor * prefix_match_length。"""

    suffix_decoding_min_token_prob: float = 0.1
    """后缀解码的最小token概率。
    只会投机估计概率(基于频率计数)大于或等于此值的token。"""

    # =========================================================================
    # 拒绝采样配置
    # =========================================================================

    draft_load_config: LoadConfig | None = None
    """草稿模型的加载配置。
    如果未指定，使用目标模型的加载配置。"""

    rejection_sample_method: RejectionSampleMethod = "standard"
    """拒绝采样方法。
    - standard: 标准概率拒绝采样(使用或不使用缓存的草稿logits)
    - synthetic: 合成拒绝采样，使用衰减概率校准到synthetic_acceptance_rate"""

    synthetic_acceptance_rates: list[float] | None = None
    """合成拒绝采样的每位置无条件接受率。
    位置i的条目是前i+1个草稿token都被接受的边际概率。
    要求:
    - 长度必须为num_speculative_tokens
    - 每个条目在[0, 1]范围内
    - 必须单调非递增
    仅当rejection_sample_method='synthetic'时有效。
    与synthetic_acceptance_length互斥。"""

    synthetic_acceptance_length: float | None = None
    """合成拒绝采样的目标平均接受长度。
    范围: [1, num_speculative_tokens + 1]
    内部会转换为synthetic_acceptance_rates。
    仅当rejection_sample_method='synthetic'时有效。
    与synthetic_acceptance_rates互斥。"""

    @staticmethod
    def _acceptance_length_to_rates(length: float, n: int) -> list[float]:
        """将平均接受长度转换为每位置无条件接受率。

        【算法说明】
        使用最小方差调度(minimum-variance schedule)将平均接受长度转换为
        每位置的无条件接受率。

        【转换逻辑】
        1. 计算期望接受的草稿token数: num_drafts = length - 1
        2. 整数部分设为1.0 (完全接受)
        3. 小数部分设为该值 (部分接受)
        4. 剩余部分设为0.0 (不接受)

        【示例】
        - length=3.5, n=5 -> [1.0, 1.0, 0.5, 0.0, 0.0]
        - length=2.0, n=4 -> [1.0, 0.0, 0.0, 0.0]

        Args:
            length: 平均接受长度，范围[1, n+1]
            n: 投机token数量

        Returns:
            每位置无条件接受率列表
        """
        num_drafts = length - 1  # 期望接受的草稿token数
        num_full = int(num_drafts)  # 完全接受的位置数
        return (
            [1.0] * num_full + [num_drafts - num_full] + [0.0] * (n - num_full - 1)
        )[:n]

    @staticmethod
    def _resolve_synthetic_acceptance_rates(
        n: int,
        rates: list[float] | None,
        length: float | None,
    ) -> list[float]:
        """解析合成拒绝采样的接受率。

        【功能说明】
        从rates或length中恰好一个参数解析每位置无条件接受率。
        验证范围、长度和单调性。

        【验证规则】
        1. rates和length必须恰好提供一个
        2. 如果提供rates:
           - 长度必须为n
           - 每个条目必须在[0, 1]范围内
           - 必须单调非递增
        3. 如果提供length:
           - 必须在[1, n+1]范围内

        Args:
            n: 投机token数量
            rates: 每位置接受率列表，或None
            length: 平均接受长度，或None

        Returns:
            每位置无条件接受率列表

        Raises:
            ValueError: 参数验证失败时
        """
        if (rates is None) == (length is None):
            raise ValueError(
                "rejection_sample_method='synthetic' requires exactly one of "
                "synthetic_acceptance_rates or synthetic_acceptance_length."
            )
        if rates is not None:
            if len(rates) != n:
                raise ValueError(
                    f"synthetic_acceptance_rates must have length {n}, got {rates}."
                )
            if not all(0.0 <= r <= 1.0 for r in rates):
                raise ValueError(
                    f"synthetic_acceptance_rates entries must be in [0, 1], "
                    f"got {rates}."
                )
            if any(rates[i] > rates[i - 1] for i in range(1, n)):
                raise ValueError(
                    f"synthetic_acceptance_rates must be non-increasing, got {rates}."
                )
            return list(rates)
        assert length is not None
        if not 1.0 <= length <= float(n + 1):
            raise ValueError(
                f"synthetic_acceptance_length must be in [1, {n + 1}], got {length}."
            )
        return SpeculativeConfig._acceptance_length_to_rates(length, n)

    draft_sample_method: DraftSampleMethod = "greedy"
    """草稿模型的采样方法。

    - greedy: 贪婪采样
      总是选择概率最高的token(argmax)。
      在拒绝采样期间，草稿概率被视为one-hot分布。
      优点: 简单高效，内存占用少。
      缺点: 可能错过概率较低但正确的token。

    - probabilistic: 概率采样
      从草稿分布中随机采样token。
      在拒绝采样期间，使用完整的草稿logits进行概率比测试。
      优点: 更准确地模拟目标模型分布。
      缺点: 需要额外的GPU内存存储完整logits。"""

    # =========================================================================
    # 核心方法: 计算配置哈希值
    # =========================================================================

    def compute_hash(self) -> str:
        """计算投机配置的哈希值。

        【功能说明】
        提供一个唯一标识所有影响计算图结构配置的哈希值。
        计算图: 从输入ids/embeddings到最终隐藏状态的路径。

        【重要提示】
        每当向此配置添加新字段时，请确保:
        - 如果该字段影响计算图结构，将其加入factors列表
        - 不影响计算图的字段不需要加入(如超参数、日志配置等)

        【影响计算图的因素】
        1. 是否使用辅助隐藏状态 (eagle3/extract_hidden_states/dflash)
        2. 辅助隐藏状态的层ID (影响模型结构)

        Returns:
            配置哈希值的十六进制字符串
        """
        factors: list[Any] = []
        # Eagle3和extract_hidden_states影响计算图，因为它们除了最终隐藏状态外
        # 还返回中间隐藏状态
        uses_aux_hidden_states = self.method in (
            "eagle3",
            "extract_hidden_states",
            "dflash",
        )
        factors.append(uses_aux_hidden_states)

        # 使用的具体层也影响计算图
        if uses_aux_hidden_states and self.draft_model_config is not None:
            layer_ids = getattr(
                self.draft_model_config.hf_config,
                "eagle_aux_hidden_state_layer_ids",
                None,
            )
            if layer_ids is not None:
                # 转换为元组使其可哈希
                factors.append(tuple(layer_ids))

        hash_str = safe_hash(str(factors).encode(), usedforsecurity=False).hexdigest()
        return hash_str

    @staticmethod
    def hf_config_override(hf_config: PretrainedConfig) -> PretrainedConfig:
        """覆盖HuggingFace配置以支持MTP(多token预测)模型。

        【功能说明】
        此静态方法用于将原始模型配置转换为MTP模型配置。
        它处理各种模型架构，将其转换为对应的MTP变体。

        【处理流程】
        1. 识别原始模型类型
        2. 将model_type修改为对应的MTP类型
        3. 设置n_predict参数(预测token数)
        4. 更新architectures为MTP模型架构

        【支持的模型转换】
        - DeepSeek系列: deepseek_v3/v32/v4 -> deepseek_mtp
        - MiMo系列: MiMoForCausalLM -> mimo_mtp
        - GLM系列: Glm4Moe -> glm4_moe_mtp
        - ERNIE系列: ernie4_5_moe -> ernie_mtp
        - Qwen系列: qwen3_next -> qwen3_next_mtp
        - 其他: exaone, step3p5, gemma4等

        Args:
            hf_config: 原始HuggingFace模型配置

        Returns:
            修改后的MTP模型配置
        """
        initial_architecture = hf_config.architectures[0]

        # =========================================================================
        # DeepSeek系列模型转换
        # =========================================================================
        if hf_config.model_type in (
            "deepseek_v3",
            "deepseek_v32",
            "glm_moe_dsa",
        ):
            hf_config.model_type = "deepseek_mtp"
        if hf_config.model_type == "deepseek_mtp":
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update(
                {"n_predict": n_predict, "architectures": ["DeepSeekMTPModel"]}
            )
        if hf_config.model_type == "deepseek_v4":
            hf_config.model_type = "deepseek_mtp"
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update(
                {"n_predict": n_predict, "architectures": ["DeepSeekV4MTPModel"]}
            )

        # =========================================================================
        # Pangu系列模型转换
        # =========================================================================
        if hf_config.model_type in ("pangu_ultra_moe"):
            hf_config.model_type = "pangu_ultra_moe_mtp"
        if hf_config.model_type == "pangu_ultra_moe_mtp":
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update(
                {"n_predict": n_predict, "architectures": ["OpenPanguMTPModel"]}
            )

        # =========================================================================
        # MiMo系列模型转换
        # =========================================================================
        if hf_config.architectures[0] == "MiMoForCausalLM":
            hf_config.model_type = "mimo_mtp"
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update(
                {
                    "num_hidden_layers": 0,
                    "n_predict": n_predict,
                    "architectures": ["MiMoMTPModel"],
                }
            )

        if (arch := hf_config.architectures[0]) in (
            "MiMoV2ForCausalLM",
            "MiMoV2OmniForCausalLM",
        ):
            from vllm.model_executor.models.mimo_v2_mtp import (
                _MIMO_V2_PRO_NUM_MTP_LAYERS,
            )

            mtp_arch_maps = {
                "MiMoV2ForCausalLM": "MiMoV2MTPModel",
                "MiMoV2OmniForCausalLM": "MiMoV2OmniMTPModel",
            }

            hf_config.model_type = "mimo_v2_mtp"
            # vLLM currently supports only the first MiMo-V2 MTP layer.
            n_predict = _MIMO_V2_PRO_NUM_MTP_LAYERS
            hf_config.update(
                {
                    "num_hidden_layers": 0,
                    "n_predict": n_predict,
                    "num_nextn_predict_layers": n_predict,
                    "architectures": [mtp_arch_maps[arch]],
                }
            )

        if hf_config.architectures[0] == "MiMoV2FlashForCausalLM":
            from vllm.model_executor.models.mimo_v2_mtp import (
                _MIMO_V2_FLASH_NUM_MTP_LAYERS,
            )

            hf_config.model_type = "mimo_v2_mtp"
            # vLLM currently supports only the first MiMo-V2 MTP layer.
            n_predict = _MIMO_V2_FLASH_NUM_MTP_LAYERS
            hf_config.update(
                {
                    "num_hidden_layers": 0,
                    "n_predict": n_predict,
                    "num_nextn_predict_layers": n_predict,
                    "architectures": ["MiMoV2MTPModel"],
                }
            )

        # =========================================================================
        # GLM系列模型转换
        # =========================================================================
        if hf_config.architectures[0] == "Glm4MoeForCausalLM":
            hf_config.model_type = "glm4_moe_mtp"
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update(
                {
                    "n_predict": n_predict,
                    "architectures": ["Glm4MoeMTPModel"],
                }
            )

        if hf_config.architectures[0] == "Glm4MoeLiteForCausalLM":
            hf_config.model_type = "glm4_moe_lite_mtp"
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update(
                {
                    "num_hidden_layers": 0,
                    "n_predict": n_predict,
                    "architectures": ["Glm4MoeLiteMTPModel"],
                }
            )

        if hf_config.architectures[0] == "GlmOcrForConditionalGeneration":
            hf_config.model_type = "glm_ocr_mtp"
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update(
                {
                    "num_hidden_layers": 0,
                    "n_predict": n_predict,
                    "architectures": ["GlmOcrMTPModel"],
                }
            )

        # =========================================================================
        # ERNIE系列模型转换
        # =========================================================================
        if hf_config.model_type == "ernie4_5_moe":
            hf_config.model_type = "ernie_mtp"
        if hf_config.model_type == "ernie_mtp":
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update(
                {"n_predict": n_predict, "architectures": ["ErnieMTPModel"]}
            )

        # =========================================================================
        # Nemotron系列模型转换
        # =========================================================================
        if hf_config.architectures[0] == "NemotronH_Super_Omni_Reasoning_V3":
            # 提升VLM的text_config，以便下面的MTP检测能正确触发
            hf_config = hf_config.text_config

        if (
            hf_config.model_type in {"nemotron_h", "nemotron_h_puzzle"}
            and hasattr(hf_config, "num_nextn_predict_layers")
            and hf_config.num_nextn_predict_layers > 0
        ):
            # 检查是否为MTP变体
            hf_config.model_type = "nemotron_h_mtp"
        if hf_config.model_type == "nemotron_h_mtp":
            n_predict = getattr(hf_config, "num_nextn_predict_layers", 1)
            hf_config.update(
                {"n_predict": n_predict, "architectures": ["NemotronHMTPModel"]}
            )

        # =========================================================================
        # Qwen系列模型转换
        # =========================================================================
        if hf_config.model_type == "qwen3_next":
            hf_config.model_type = "qwen3_next_mtp"
        if hf_config.model_type == "qwen3_next_mtp":
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update(
                {"n_predict": n_predict, "architectures": ["Qwen3NextMTP"]}
            )

        # =========================================================================
        # EXAONE系列模型转换
        # =========================================================================
        if hf_config.model_type == "exaone_moe":
            hf_config.model_type = "exaone_moe_mtp"
        if hf_config.model_type == "exaone_moe_mtp":
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update(
                {"n_predict": n_predict, "architectures": ["ExaoneMoeMTP"]}
            )
        if "exaone4_5" in hf_config.model_type:
            hf_config.model_type = "exaone4_5_mtp"
        if hf_config.model_type == "exaone4_5_mtp":
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update(
                {"n_predict": n_predict, "architectures": ["Exaone4_5_MTP"]}
            )

        # =========================================================================
        # Qwen3.5系列模型转换
        # =========================================================================
        if hf_config.model_type in ("qwen3_5", "qwen3_5_moe"):
            is_moe = hf_config.model_type == "qwen3_5_moe"
            hf_config.model_type = "qwen3_5_mtp"
            n_predict = getattr(hf_config, "mtp_num_hidden_layers", None)
            hf_config.update(
                {
                    "n_predict": n_predict,
                    "architectures": ["Qwen3_5MoeMTP" if is_moe else "Qwen3_5MTP"],
                }
            )
        if hf_config.model_type == "intern_s2_preview":
            text_config = getattr(hf_config, "text_config", None)
            is_moe = getattr(text_config, "model_type", None) == "qwen3_5_moe_text"
            hf_config.model_type = "qwen3_5_mtp"
            n_predict = getattr(text_config, "mtp_num_hidden_layers", None)
            hf_config.update(
                {
                    "n_predict": n_predict,
                    "architectures": ["Qwen3_5MoeMTP" if is_moe else "Qwen3_5MTP"],
                }
            )

        # =========================================================================
        # LongCat系列模型转换
        # =========================================================================
        if hf_config.model_type == "longcat_flash":
            hf_config.model_type = "longcat_flash_mtp"
            n_predict = getattr(hf_config, "num_nextn_predict_layers", 1)
            hf_config.update(
                {"n_predict": n_predict, "architectures": ["LongCatFlashMTPModel"]}
            )

        # =========================================================================
        # Step系列模型转换
        # =========================================================================
        if hf_config.model_type in ("step3p5", "step3p7") or hf_config.architectures[
            0
        ] in ("Step3p5ForCausalLM", "Step3p7ForConditionalGeneration"):
            quantization_config = getattr(hf_config, "quantization_config", None)
            hf_config = getattr(hf_config, "text_config", hf_config)
            if (
                quantization_config is not None
                and getattr(hf_config, "quantization_config", None) is None
            ):
                hf_config.update({"quantization_config": quantization_config})
            hf_config.model_type = "step3p5_mtp"
            n_predict = getattr(hf_config, "num_nextn_predict_layers", 1)
            hf_config.update({"n_predict": n_predict, "architectures": ["Step3p5MTP"]})

        # =========================================================================
        # Mistral系列模型转换
        # =========================================================================
        if initial_architecture == "MistralLarge3ForCausalLM":
            hf_config.update({"architectures": ["EagleMistralLarge3ForCausalLM"]})

        # =========================================================================
        # HY系列模型转换
        # =========================================================================
        if hf_config.model_type == "hy_v3":
            hf_config.model_type = "hy_v3_mtp"
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update(
                {"n_predict": n_predict, "architectures": ["HYV3MTPModel"]}
            )

        # =========================================================================
        # Gemma4系列模型转换
        # =========================================================================
        if hf_config.model_type == "gemma4_assistant":
            hf_config.model_type = "gemma4_mtp"
            text_config = getattr(hf_config, "text_config", hf_config)
            # The assistant runs all decoder layers in a single forward
            # call to produce one draft token, so n_predict=1.
            # num_kv_shared_layers must be 0: cross-model KV sharing is
            # set up by the proposer after model construction.
            if hasattr(text_config, "num_kv_shared_layers"):
                text_config.num_kv_shared_layers = 0
            hf_config.update({"n_predict": 1, "architectures": ["Gemma4MTPModel"]})

        return hf_config

    def __post_init__(self):
        """投机配置的后初始化方法。

        【功能说明】
        在dataclass初始化后自动调用，负责:
        1. 推断投机方法类型(method)
        2. 根据method初始化草稿模型配置
        3. 验证和设置各种参数
        4. 创建草稿模型的并行配置

        【初始化流程详解】

        第一步: 推断投机方法类型
        ─────────────────────────────────────────────────────
        - 如果model包含"."且不是URL/HF仓库，视为自定义类路径
        - 如果method为None:
          - model="ngram" -> method="ngram"
          - 其他情况 -> method="draft_model"

        第二步: 处理MTP方法的弃用警告
        ─────────────────────────────────────────────────────
        - 如果method是旧的MTP类型名称(如"deepseek_mtp")，转换为"mtp"

        第三步: 处理未指定model但指定了num_speculative_tokens的情况
        ─────────────────────────────────────────────────────
        - mtp: 使用目标模型作为草稿模型
        - ngram/ngram_gpu: 使用特殊标识
        - suffix: 使用特殊标识
        - extract_hidden_states: 使用特殊标识
        - custom_class: 必须已指定model

        第四步: 根据method类型初始化草稿模型配置
        ─────────────────────────────────────────────────────
        - ngram/ngram_gpu: 设置查找窗口参数
        - suffix: 验证后缀解码参数
        - custom_class: 警告实验性功能
        - extract_hidden_states: 创建ExtractHiddenStates配置
        - 其他(如eagle/medusa/draft_model): 创建草稿ModelConfig

        第五步: 自动检测投机方法
        ─────────────────────────────────────────────────────
        - 根据模型名称或配置自动识别方法类型
        - 设置EAGLE配置、并行草稿等高级选项

        第六步: 验证和设置张量并行配置
        ─────────────────────────────────────────────────────
        - 验证draft_tensor_parallel_size的有效性
        - 设置draft_model_config.max_model_len
        - 创建draft_parallel_config

        Returns:
            self: 配置实例
        """

        # =========================================================================
        # 第一步: 推断投机方法类型
        # =========================================================================
        # Note: "method" is a new parameter that helps to extend the
        # configuration of non-model-based proposers, and the "model" parameter
        # will be used to set the draft model, eagle head, or additional weight
        # when needed. If users do not specify "method", the speculative method
        # will be detected automatically if possible. If the speculative method
        # can not be detected, it will be considered as the "draft_model" by
        # default.

        # infer method from user args
        # Check if the model field contains a custom module path (e.g., 'pkg.Mod')
        if (
            self.model is not None
            and "." in self.model
            and not self.model.startswith(("http://", "https://", "file://"))
            and "/" not in self.model  # not a HuggingFace repo (org/model)
        ):
            # Treat as a custom class path
            self.method = "custom_class"
        elif self.method is None:
            if self.model in ("ngram", "[ngram]"):
                self.method = "ngram"
            else:
                self.method = "draft_model"

        # =========================================================================
        # 第二步: 处理MTP方法的弃用警告
        # =========================================================================
        if self.method in get_args(MTPModelTypes) and self.method != "mtp":
            logger.warning(
                "method `%s` is deprecated and replaced with mtp.", self.method
            )
            self.method = "mtp"

        # =========================================================================
        # 第三步: 处理未指定model但指定了num_speculative_tokens的情况
        # =========================================================================
        if self.model is None and self.num_speculative_tokens is not None:
            if self.method == "mtp":
                # MTP方法: 使用目标模型作为草稿模型
                if self.target_model_config is None:
                    raise ValueError("target_model_config must be present for mtp")
                if self.target_model_config.hf_text_config.model_type == "deepseek_v32":
                    # FIXME(luccafong): cudagraph with v32 MTP is not supported,
                    # remove this when the issue is fixed.
                    self.enforce_eager = True
                # use the draft model from the same model:
                self.model = self.target_model_config.model
                # Align the quantization of draft model for cases such as
                # --quantization fp8 with a bf16 checkpoint.
                if not self.quantization:
                    self.quantization = self.target_model_config.quantization
            elif self.method in ("ngram", "[ngram]"):
                # Ngram方法: 使用特殊标识
                self.model = "ngram"
            elif self.method == "ngram_gpu":
                # Ngram GPU方法: 使用特殊标识
                self.model = "ngram_gpu"
            elif self.method == "suffix":
                # 后缀解码方法: 使用特殊标识
                self.model = "suffix"
            elif self.method == "extract_hidden_states":
                # 提取隐藏状态方法: 使用特殊标识
                self.model = "extract_hidden_states"
            elif self.method == "custom_class":
                # method was set explicitly, but model should already contain the
                # custom module path. If not, this is a configuration error.
                if self.model is None:
                    raise ValueError(
                        "method='custom_class' requires 'model' to contain the "
                        "custom proposer module path (e.g., 'my_module.MyProposer')."
                    )
            else:
                raise ValueError(
                    "num_speculative_tokens was provided but without speculative model."
                )

        # =========================================================================
        # 第四步: 根据method类型初始化草稿模型配置
        # =========================================================================

        # 统一ngram方法的表示
        if self.method in ("ngram", "[ngram]"):
            self.method = "ngram"

        if self.method in ("ngram", "ngram_gpu"):
            # ─────────────────────────────────────────────────────────────────
            # Ngram方法配置
            # ─────────────────────────────────────────────────────────────────
            # Set default values if not provided
            if self.prompt_lookup_min is None and self.prompt_lookup_max is None:
                # TODO(woosuk): Tune these values. They are arbitrarily chosen.
                self.prompt_lookup_min = 5
                self.prompt_lookup_max = 5
            elif self.prompt_lookup_min is None:
                if self.prompt_lookup_max is None:
                    raise ValueError(
                        "Either prompt_lookup_max or prompt_lookup_min must be "
                        "provided when using the ngram method."
                    )
                self.prompt_lookup_min = self.prompt_lookup_max
            elif self.prompt_lookup_max is None:
                if self.prompt_lookup_min is None:
                    raise ValueError(
                        "Either prompt_lookup_max or prompt_lookup_min must be "
                        "provided when using the ngram method."
                    )
                self.prompt_lookup_max = self.prompt_lookup_min

            # Validate values
            if self.prompt_lookup_min > self.prompt_lookup_max:
                raise ValueError(
                    f"prompt_lookup_min={self.prompt_lookup_min} must "
                    f"be <= prompt_lookup_max={self.prompt_lookup_max}"
                )

            # TODO: current we still need extract vocab_size from target model
            # config, in future, we may try refactor it out, and set
            # draft related config as None here.
            # Ngram方法不需要独立的草稿模型配置，复用目标模型配置
            self.draft_model_config = self.target_model_config
            self.draft_parallel_config = self.target_parallel_config

        elif self.method == "suffix":
            # ─────────────────────────────────────────────────────────────────
            # 后缀解码方法配置
            # ─────────────────────────────────────────────────────────────────
            self._validate_suffix_decoding()

        elif self.method == "custom_class":
            # ─────────────────────────────────────────────────────────────────
            # 自定义类方法配置
            # ─────────────────────────────────────────────────────────────────
            # Custom class proposer does not need a draft model.
            # It will dynamically load the user-provided class at runtime.
            logger.warning_once(
                "Using a custom class-based proposer backend. This is an "
                "experimental feature and the proposer interface is subject to "
                "breaking changes in future vLLM releases."
            )
            self.prompt_lookup_max = 0
            self.prompt_lookup_min = 0
            self.draft_model_config = self.target_model_config
            self.draft_parallel_config = self.target_parallel_config

        elif self.method == "extract_hidden_states":
            # ─────────────────────────────────────────────────────────────────
            # 提取隐藏状态方法配置
            # ─────────────────────────────────────────────────────────────────
            from vllm.transformers_utils.configs.extract_hidden_states import (
                ExtractHiddenStatesConfig,
            )

            # ExtractHiddenStatesModel is instantiated manually in load_model()
            # We just need to store the target model config for KV cache shape info
            self.model = "extract_hidden_states"
            self.prompt_lookup_max = 0
            self.prompt_lookup_min = 0

            # 获取HuggingFace配置
            if hasattr(self.draft_model_config, "hf_config"):
                hf_config = self.draft_model_config.hf_config.to_dict()
            elif (
                isinstance(self.draft_model_config, dict)
                and "hf_config" in self.draft_model_config
            ):
                hf_config = self.draft_model_config["hf_config"]
            else:
                hf_config = {}

            # 创建草稿模型配置
            self.draft_model_config = copy.copy(self.target_model_config)
            self.draft_model_config.hf_config = ExtractHiddenStatesConfig(
                self.draft_model_config.hf_config, **hf_config
            )
            self.update_arch_()
            self.draft_parallel_config = self.target_parallel_config

        else:
            # ─────────────────────────────────────────────────────────────────
            # 基于模型的投机方法配置 (eagle/medusa/mlp_speculator/draft_model等)
            # ─────────────────────────────────────────────────────────────────
            self.prompt_lookup_max = 0
            self.prompt_lookup_min = 0

            if self.model is not None:
                # 创建草稿模型的ModelConfig
                # 继承目标模型的tokenizer、dtype等配置
                self.draft_model_config = ModelConfig(
                    model=self.model,
                    runner="draft",
                    tokenizer=self.target_model_config.tokenizer,
                    tokenizer_mode=self.target_model_config.tokenizer_mode,
                    trust_remote_code=self.target_model_config.trust_remote_code,
                    allowed_local_media_path=self.target_model_config.allowed_local_media_path,
                    allowed_media_domains=self.target_model_config.allowed_media_domains,
                    dtype=self.target_model_config.dtype,
                    seed=self.target_model_config.seed,
                    revision=self.revision,
                    code_revision=self.code_revision,
                    tokenizer_revision=self.target_model_config.tokenizer_revision,
                    max_model_len=self.max_model_len,  # type: ignore[arg-type]
                    spec_target_max_model_len=self.target_model_config.max_model_len,
                    quantization=self.quantization,
                    enforce_eager=self.target_model_config.enforce_eager,
                    max_logprobs=self.target_model_config.max_logprobs,
                    hf_overrides=SpeculativeConfig.hf_config_override,
                    config_format=self.target_model_config.config_format,
                )

                # =========================================================================
                # 第五步: 自动检测投机方法
                # =========================================================================
                # Automatically detect the method
                if self.method in ("eagle", "eagle3", "dflash"):
                    # 已经明确指定的方法，无需检测
                    pass
                # examples:
                # yuhuili/EAGLE-LLaMA3-Instruct-8B
                # yuhuili/EAGLE3-LLaMA3.1-Instruct-8B
                # AngelSlim/Qwen3-8B_eagle3
                elif "eagle-" in self.draft_model_config.model.lower():
                    # 模型名称包含"eagle-"，识别为EAGLE方法
                    self.method = "eagle"
                elif "eagle3" in self.draft_model_config.model.lower():
                    # 模型名称包含"eagle3"，识别为EAGLE3方法
                    self.method = "eagle3"
                elif "dflash" in self.draft_model_config.model.lower():
                    # 模型名称包含"dflash"，识别为DFlash方法
                    self.method = "dflash"
                elif self.draft_model_config.hf_config.model_type == "medusa":
                    # 模型类型为medusa，识别为Medusa方法
                    self.method = "medusa"
                elif self.draft_model_config.hf_config.model_type == "mlp_speculator":
                    # 模型类型为mlp_speculator，识别为MLP投机器方法
                    self.method = "mlp_speculator"
                elif self.draft_model_config.hf_config.model_type in get_args(
                    MTPModelTypes
                ):
                    # 模型类型为MTP类型，识别为MTP方法
                    self.method = "mtp"
                    if (
                        self.num_speculative_tokens > 1
                        and self.draft_model_config.hf_config.model_type
                        != "step3p5_mtp"
                    ):
                        logger.warning(
                            "Enabling num_speculative_tokens > 1 will run "
                            "multiple times of forward on same MTP layer"
                            ",which may result in lower acceptance rate"
                        )
                elif self.method == "draft_model":
                    # 默认的draft_model方法，无需特殊处理
                    pass
                else:
                    raise NotImplementedError(
                        f"Unsupported speculative method: '{self.method}'"
                    )

                # =========================================================================
                # EAGLE方法的特殊配置
                # =========================================================================
                # Replace hf_config for EAGLE draft_model
                if self.method in ("eagle", "eagle3", "dflash"):
                    from vllm.transformers_utils.configs.eagle import EAGLEConfig
                    from vllm.transformers_utils.configs.speculators import (
                        SpeculatorsConfig,
                    )

                    if isinstance(
                        self.draft_model_config.hf_config,
                        (EAGLEConfig, SpeculatorsConfig),
                    ):
                        pass
                    else:
                        # 创建EAGLE配置
                        eagle_config = EAGLEConfig(
                            self.draft_model_config.hf_config,
                            method=self.method,
                            model_type="eagle",
                        )
                        self.draft_model_config.hf_config = eagle_config
                        self.update_arch_()

                # DFlash方法强制启用并行草稿
                if self.method == "dflash":
                    self.parallel_drafting = True

                # 设置lookahead token数
                if self.num_speculative_tokens is not None and hasattr(
                    self.draft_model_config.hf_config, "num_lookahead_tokens"
                ):
                    self.draft_model_config.hf_config.num_lookahead_tokens = (
                        self.num_speculative_tokens
                    )

                # =========================================================================
                # 处理n_predict参数
                # =========================================================================
                n_predict = getattr(
                    self.draft_model_config.hf_config, "n_predict", None
                )
                if n_predict is not None:
                    if self.num_speculative_tokens is None:
                        # Default to max value defined in draft model config.
                        self.num_speculative_tokens = n_predict
                    elif (
                        self.num_speculative_tokens > n_predict
                        and self.num_speculative_tokens % n_predict != 0
                    ):
                        # Ensure divisibility for MTP module reuse.
                        raise ValueError(
                            f"num_speculative_tokens:{self.num_speculative_tokens}"
                            f" must be divisible by {n_predict=}"
                        )

                # 验证num_speculative_tokens已设置
                if self.num_speculative_tokens is None:
                    raise ValueError(
                        "A speculative model was provided, but "
                        "`num_speculative_tokens` was not provided"
                    )

                # =========================================================================
                # 第六步: 验证和设置张量并行配置
                # =========================================================================
                self.draft_tensor_parallel_size = (
                    SpeculativeConfig._verify_and_get_draft_tp(
                        self.target_parallel_config,
                        self.draft_tensor_parallel_size,
                        self.draft_model_config.hf_config,
                    )
                )

                # 设置草稿模型的最大序列长度
                self.draft_model_config.max_model_len = (
                    SpeculativeConfig._maybe_override_draft_max_model_len(
                        self.max_model_len,
                        self.draft_model_config.max_model_len,
                        self.target_model_config.max_model_len,
                    )
                )

                # 创建草稿模型的并行配置
                self.draft_parallel_config = (
                    SpeculativeConfig.create_draft_parallel_config(
                        self.target_parallel_config, self.draft_tensor_parallel_size
                    )
                )
        return self

    def _validate_suffix_decoding(self):
        """验证后缀解码配置参数。

        【功能说明】
        验证后缀解码方法的所有配置参数是否有效。

        【验证内容】
        1. 检查arctic-inference依赖是否已安装
        2. 设置默认的num_speculative_tokens
        3. 验证各参数的取值范围

        【参数约束】
        - suffix_decoding_max_tree_depth: >= 1
        - suffix_decoding_max_cached_requests: >= 0
        - suffix_decoding_max_spec_factor: >= 0
        - suffix_decoding_min_token_prob: [0, 1]

        Raises:
            ImportError: arctic-inference未安装
            ValueError: 参数验证失败
        """
        if not has_arctic_inference():
            raise ImportError(
                "Arctic Inference is required for suffix decoding. "
                "Install via `pip install arctic-inference==0.1.1`."
            )
        if self.num_speculative_tokens is None:
            # Suffix decoding decides the actual number of speculative tokens
            # dynamically and treats num_speculative_tokens as a maximum limit.
            self.num_speculative_tokens = self.suffix_decoding_max_tree_depth
            logger.warning(
                "Defaulted num_speculative_tokens to %s for suffix decoding.",
                self.num_speculative_tokens,
            )
        # Validate values
        if self.suffix_decoding_max_tree_depth < 1:
            raise ValueError(
                f"suffix_decoding_max_tree_depth="
                f"{self.suffix_decoding_max_tree_depth} must be >= 1"
            )
        if self.suffix_decoding_max_cached_requests < 0:
            raise ValueError(
                f"suffix_decoding_max_cached_requests="
                f"{self.suffix_decoding_max_cached_requests} must be >= 0"
            )
        if self.suffix_decoding_max_spec_factor < 0:
            raise ValueError(
                f"suffix_decoding_max_spec_factor="
                f"{self.suffix_decoding_max_spec_factor} must be >= 0"
            )
        if not 0 <= self.suffix_decoding_min_token_prob <= 1:
            raise ValueError(
                f"suffix_decoding_min_token_prob="
                f"{self.suffix_decoding_min_token_prob} must be in [0, 1]"
            )

    @staticmethod
    def _maybe_override_draft_max_model_len(
        speculative_max_model_len: int | None,
        draft_max_model_len: int,
        target_max_model_len: int,
    ) -> int:
        """确定草稿模型的最大序列长度。

        【功能说明】
        确定草稿模型可以处理的最大序列长度。
        这是必要的，以确保序列不会超过草稿模型或目标模型的容量。

        【选择逻辑】
        1. 如果指定了speculative_max_model_len，使用该值
        2. 否则，取draft_max_model_len和target_max_model_len的较小值

        【约束条件】
        - speculative_max_model_len不能大于draft_max_model_len
        - speculative_max_model_len不能大于target_max_model_len

        Args:
            speculative_max_model_len: 用户指定的投机最大序列长度，或None
            draft_max_model_len: 草稿模型的最大序列长度
            target_max_model_len: 目标模型的最大序列长度

        Returns:
            草稿模型的最大序列长度

        Raises:
            ValueError: speculative_max_model_len超过限制
        """

        if speculative_max_model_len is not None:
            if speculative_max_model_len > draft_max_model_len:
                raise ValueError(
                    f"{speculative_max_model_len=} cannot be "
                    f"larger than {draft_max_model_len=}"
                )

            if speculative_max_model_len > target_max_model_len:
                raise ValueError(
                    f"{speculative_max_model_len=} cannot be "
                    f"larger than {target_max_model_len=}"
                )

            return speculative_max_model_len

        result = min(
            draft_max_model_len,
            target_max_model_len,
        )
        if result != draft_max_model_len:
            logger.info(
                "Overriding draft model max model len from %d to %d",
                draft_max_model_len,
                result,
            )
        return result

    @staticmethod
    def _verify_and_get_draft_tp(
        target_parallel_config: ParallelConfig,
        speculative_draft_tensor_parallel_size: int | None,
        draft_hf_config: PretrainedConfig,
    ) -> int:
        """验证并获取草稿模型的张量并行大小。

        【功能说明】
        验证和调整草稿模型的张量并行大小。

        【处理逻辑】
        1. 如果未指定speculative_draft_tensor_parallel_size:
           - mlp_speculator方法: 强制为1(不支持tp>1)
           - 其他方法: 使用目标模型的张量并行大小
        2. 如果已指定:
           - 只能是1或与目标模型相同的张量并行大小

        Args:
            target_parallel_config: 目标模型的并行配置
            speculative_draft_tensor_parallel_size: 用户指定的草稿模型张量并行大小
            draft_hf_config: 草稿模型的HuggingFace配置

        Returns:
            验证后的草稿模型张量并行大小

        Raises:
            ValueError: 指定的值无效
        """
        # If speculative_draft_tensor_parallel_size is unset then set it
        # appropriately else verify that it is set correctly.
        if speculative_draft_tensor_parallel_size is None:
            if draft_hf_config.model_type == "mlp_speculator":
                speculative_draft_tensor_parallel_size = 1
                if target_parallel_config.tensor_parallel_size > 1:
                    logger.warning(
                        "%s cannot currently be run with tp>1; "
                        "setting speculative_draft_tensor_parallel_size=1",
                        draft_hf_config.model_type,
                    )
            else:
                speculative_draft_tensor_parallel_size = (
                    target_parallel_config.tensor_parallel_size
                )
        elif speculative_draft_tensor_parallel_size not in (
            1,
            target_parallel_config.tensor_parallel_size,
        ):
            raise ValueError(
                f"{speculative_draft_tensor_parallel_size=} cannot be "
                f"other value than 1 or target model tensor_parallel_size"
            )
        return speculative_draft_tensor_parallel_size

    def update_arch_(self):
        """更新草稿模型的架构相关字段。

        【功能说明】
        当EagleConfig或ExtractHiddenStatesConfig更新了架构后，
        需要同步更新草稿模型配置中的所有架构相关字段。

        【更新内容】
        1. hf_text_config: HuggingFace文本配置
        2. model_arch_config: 模型架构配置
        3. _model_info: 模型信息
        4. _architecture: 架构名称
        """
        self.draft_model_config.hf_text_config = get_hf_text_config(
            self.draft_model_config.hf_config
        )
        self.draft_model_config.model_arch_config = (
            self.draft_model_config.get_model_arch_config()
        )
        model_info, arch = self.draft_model_config.registry.inspect_model_cls(
            self.draft_model_config.architectures,
            self.draft_model_config,
        )
        self.draft_model_config._model_info = model_info
        self.draft_model_config._architecture = arch

    @staticmethod
    def create_draft_parallel_config(
        target_parallel_config: ParallelConfig,
        speculative_draft_tensor_parallel_size: int,
    ) -> ParallelConfig:
        """创建草稿模型的并行配置。

        【功能说明】
        为草稿worker创建并行配置。
        这主要是目标并行配置的副本，但使用不同的张量并行大小。

        【配置继承】
        从目标模型继承:
        - pipeline_parallel_size: 流水线并行大小
        - distributed_executor_backend: 分布式执行器后端
        - max_parallel_loading_workers: 最大并行加载worker数
        - disable_custom_all_reduce: 是否禁用自定义all-reduce
        - ray_workers_use_nsight: 是否使用Nsight
        - placement_group: 放置组

        Args:
            target_parallel_config: 目标模型的并行配置
            speculative_draft_tensor_parallel_size: 草稿模型的张量并行大小

        Returns:
            草稿模型的并行配置
        """
        draft_parallel_config = ParallelConfig(
            pipeline_parallel_size=target_parallel_config.pipeline_parallel_size,
            tensor_parallel_size=speculative_draft_tensor_parallel_size,
            distributed_executor_backend=target_parallel_config.distributed_executor_backend,
            max_parallel_loading_workers=target_parallel_config.max_parallel_loading_workers,
            disable_custom_all_reduce=target_parallel_config.disable_custom_all_reduce,
            ray_workers_use_nsight=target_parallel_config.ray_workers_use_nsight,
            placement_group=target_parallel_config.placement_group,
        )

        return draft_parallel_config

    @field_validator("attention_backend", mode="before")
    @classmethod
    def _parse_attention_backend(cls, value: Any) -> Any:
        """解析注意力后端字符串。

        【功能说明】
        将字符串形式的注意力后端转换为枚举类型。
        - "auto" -> None (自动选择)
        - 其他字符串 -> 对应的AttentionBackendEnum值

        Args:
            value: 输入值

        Returns:
            解析后的注意力后端
        """
        if isinstance(value, str):
            if value.lower() == "auto":
                return None
            return AttentionBackendEnum[value.upper()]
        return value

    @model_validator(mode="after")
    def _verify_args(self) -> Self:
        """验证所有配置参数。

        【功能说明】
        在模型初始化后验证所有参数的有效性。

        【验证内容】
        1. 检查tensor_parallel_size是否误用
        2. 验证num_speculative_tokens已设置
        3. 验证num_speculative_tokens > 0
        4. 处理合成拒绝采样配置
        5. 验证草稿模型与并行配置兼容
        6. 验证词汇表大小一致

        Returns:
            self: 配置实例

        Raises:
            ValueError: 参数验证失败
        """
        if self.tensor_parallel_size is not None:
            raise ValueError(
                "'tensor_parallel_size' is not a valid argument in the "
                "speculative_config. Please pass 'draft_tensor_parallel_size' instead."
            )

        if self.num_speculative_tokens is None:
            raise ValueError(
                "num_speculative_tokens must be provided with "
                "speculative model unless the draft model config contains an "
                "n_predict parameter."
            )

        if self.num_speculative_tokens <= 0:
            raise ValueError(
                "Expected num_speculative_tokens to be greater "
                f"than zero ({self.num_speculative_tokens})."
            )

        if self.rejection_sample_method == "synthetic":
            # Consolidate to per-position rates
            self.synthetic_acceptance_rates = self._resolve_synthetic_acceptance_rates(
                self.num_speculative_tokens,
                self.synthetic_acceptance_rates,
                self.synthetic_acceptance_length,
            )
            self.synthetic_acceptance_length = None
        elif (
            self.synthetic_acceptance_rates is not None
            or self.synthetic_acceptance_length is not None
        ):
            raise ValueError(
                "synthetic_acceptance_rates / synthetic_acceptance_length "
                "are only valid with rejection_sample_method='synthetic'."
            )

        if self.draft_model_config:
            self.draft_model_config.verify_with_parallel_config(
                self.draft_parallel_config
            )

        self.verify_equal_vocab_size_if_draft_model()
        return self

    def verify_equal_vocab_size_if_draft_model(self):
        """验证草稿模型和目标模型的词汇表大小一致。

        【功能说明】
        当使用draft_model方法时，确保草稿模型和目标模型具有相同的词汇表大小。
        使用不同tokenizer的模型会导致投机解码期间的越界错误。

        【验证逻辑】
        - 仅在method="draft_model"时验证
        - 比较target_vocab_size和draft_vocab_size
        - 如果不相等，抛出ValueError

        Raises:
            ValueError: 词汇表大小不一致
        """
        if (
            self.method == "draft_model"
            and self.target_model_config is not None
            and self.draft_model_config is not None
        ):
            target_vocab_size = self.target_model_config.get_vocab_size()
            draft_vocab_size = self.draft_model_config.get_vocab_size()
            if target_vocab_size != draft_vocab_size:
                raise ValueError(
                    f"Target and draft model should have the same vocabulary size. "
                    f"Target model vocab_size={target_vocab_size}. "
                    f"Draft model vocab_size={draft_vocab_size}. "
                    f"Using models with different tokenizers can cause out-of-bounds "
                    f"errors during speculative decoding."
                )

    @property
    def max_num_new_slots_for_drafting(self) -> int:
        """计算草稿时可能添加到批次的最大新槽数。

        【功能说明】
        计算在进行投机解码时，每个请求可能需要的最大新槽数。
        这用于预分配批次空间。

        【计算逻辑】
        1. 串行非草稿模型方法: 0 (不需要额外槽位)
        2. 并行草稿: num_speculative_tokens - 1 (每个masked token一个槽位)
        3. 草稿模型: +1 (每个请求一个额外槽位)

        Returns:
            每个请求的最大新槽数
        """
        slots_per_req = 0  # for serial non-draft-model methods, no change needed
        if self.parallel_drafting:
            # For parallel drafting, we need one new slot per 'masked' token
            slots_per_req = self.num_speculative_tokens - 1
        if self.uses_draft_model():
            # For draft model-based speculation, we need one new slot per request
            # Since we do not slice the draft tokens
            slots_per_req += 1
        return slots_per_req

    def use_gemma4_mtp(self) -> bool:
        """检查是否使用Gemma4 MTP方法。

        【功能说明】
        判断当前配置是否使用Gemma4的MTP(多token预测)方法。

        Returns:
            True如果使用Gemma4 MTP，否则False
        """
        return (
            self.method == "mtp"
            and self.draft_model_config is not None
            and getattr(self.draft_model_config.hf_config, "model_type", None)
            == "gemma4_mtp"
        )

    def use_step3p5_mtp(self) -> bool:
        """检查是否使用Step3.5 MTP方法。

        【功能说明】
        判断当前配置是否使用Step3.5的MTP(多token预测)方法。

        Returns:
            True如果使用Step3.5 MTP，否则False
        """
        return (
            self.method == "mtp"
            and self.draft_model_config is not None
            and getattr(self.draft_model_config.hf_config, "model_type", None)
            == "step3p5_mtp"
        )

    def use_eagle(self) -> bool:
        """检查是否使用EAGLE系列方法。

        【功能说明】
        判断当前配置是否使用EAGLE、EAGLE3、MTP或DFlash方法。

        Returns:
            True如果使用EAGLE系列方法，否则False
        """
        return self.method in ("eagle", "eagle3", "mtp", "dflash")

    def use_dflash(self) -> bool:
        """检查是否使用DFlash方法。

        【功能说明】
        判断当前配置是否使用DFlash方法。

        Returns:
            True如果使用DFlash，否则False
        """
        return self.method == "dflash"

    def uses_draft_model(self) -> bool:
        """检查是否使用独立草稿模型方法。

        【功能说明】
        判断当前配置是否使用draft_model方法。

        Returns:
            True如果使用draft_model，否则False
        """
        return self.method == "draft_model"

    def uses_extract_hidden_states(self) -> bool:
        """检查是否使用提取隐藏状态方法。

        【功能说明】
        判断当前配置是否使用extract_hidden_states方法。

        Returns:
            True如果使用extract_hidden_states，否则False
        """
        return self.method == "extract_hidden_states"

    def use_ngram_gpu(self) -> bool:
        """检查是否使用Ngram GPU方法。

        【功能说明】
        判断当前配置是否使用ngram_gpu方法。

        Returns:
            True如果使用ngram_gpu，否则False
        """
        return self.method == "ngram_gpu"

    def __repr__(self) -> str:
        """返回配置的字符串表示。

        【功能说明】
        返回SpeculativeConfig的可读字符串表示，用于调试和日志。

        【格式】
        SpeculativeConfig(method=..., model=..., num_spec_tokens=...)

        Returns:
            配置的字符串表示
        """
        method = self.method
        model = (
            None
            if method
            in (
                "ngram",
                "suffix",
                "extract_hidden_states",
                "custom_class",
            )
            else self.draft_model_config.model
        )
        num_spec_tokens = self.num_speculative_tokens
        return f"SpeculativeConfig({method=}, {model=}, {num_spec_tokens=})"
