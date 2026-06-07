# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Analytic flops/memory estimation module for transformer components,
to help derive MFU (Model Flops Utilization) stats for a running model.
"""

# 性能指标估算模块
#
# 本模块基于模型配置解析，分析估算 Transformer 各组件的 FLOPs（浮点运算数）
# 和内存带宽（读写字节数），用于计算 MFU（Model Flops Utilization，模型算力利用率）。
#
# MFU 是衡量推理效率的关键指标：
#   MFU = 实际 FLOPs / 理论峰值 FLOPs
#
# 本模块的核心设计：
#
# 1. 组件化架构：将模型分解为三个主要组件：
#    - Attention（注意力层）：QKV 投影、注意力计算、输出投影
#    - FFN（前馈网络层）：Dense FFN、MoE 路由专家、MoE 共享专家
#    - Unembed（反嵌入层）：logits 计算
#
# 2. 解析器链模式（Parser Chain）：
#    每个组件有一条解析器链，按顺序解析 VllmConfig 中的各字段。
#    解析器之间可能互相覆盖结果（如量化配置覆盖默认的权重字节大小）。
#
# 3. 统一接口：
#    每个组件实现 get_num_flops_breakdown()、get_read_bytes_breakdown()、
#    get_write_bytes_breakdown() 三个方法，返回分项明细。
#
# 使用流程：
#   1. ModelMetrics.__init__() 解析 VllmConfig，实例化各组件指标
#   2. get_step_perf_stats_per_gpu() 根据 SchedulerOutput 构建 ExecutionContext
#   3. 调用各组件的方法计算 FLOPs 和内存带宽
#   4. 汇总为 PerfStats 返回
#
# 支持的并行策略：
# - TP（Tensor Parallelism）：注意力头和 FFN 中间维度的切分
# - PP（Pipeline Parallelism）：层数的切分
# - EP（Expert Parallelism）：MoE 专家的切分
# - DP（Data Parallelism）：通过 ffn_tp_size/ffn_ep_size 间接影响

import json
import time
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any, Protocol

import prometheus_client
import torch
from pydantic import BaseModel, Field, ValidationError, model_validator
from typing_extensions import Self

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.utils.torch_utils import (
    STR_DTYPE_TO_TORCH_DTYPE,
    get_dtype_size,
    get_kv_cache_torch_dtype,
)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.metrics.utils import create_metric_per_engine

logger = init_logger(__name__)


class InvalidComponent(Exception):
    """
    Custom exception to indicate that a certain ComponentMetric is not
    applicable to the given VllmConfig.
    """

    pass


# 量化方法到有效权重字节大小的映射表。
# 由 AttentionQuantizationConfigParser 和 FfnQuantizationConfigParser 共同使用，
# 用于确定权重的字节大小，从而计算 FLOPs 和内存带宽。
#
# 注意：GPTQ 和 BitsAndBytes 等方法支持可变位宽（如 4-bit 和 8-bit），
# 这里默认使用 4-bit（0.5 字节），因为这是最常见的配置。
#
# 字节大小说明：
# - 1 字节：FP8 量化（每个权重占 8 位）
# - 0.5 字节：FP4/INT4 量化（每个权重占 4 位）
# - 1 字节：INT8 量化（如 experts_int8）
_QUANT_WEIGHT_BYTE_SIZE: dict[str, float] = {
    # FP8 methods (1 byte per weight)
    "fp8": 1,
    "fbgemm_fp8": 1,
    "ptpc_fp8": 1,
    "fp_quant": 1,
    "modelopt": 1,
    "modelopt_mxfp8": 1,
    # FP4 / INT4 methods (0.5 bytes per weight)
    "mxfp4": 0.5,
    "awq": 0.5,
    "awq_marlin": 0.5,
    "gptq": 0.5,
    "gptq_marlin": 0.5,
    "bitsandbytes": 0.5,
    "modelopt_fp4": 0.5,
    "petit_nvfp4": 0.5,
    "gguf": 0.5,
    "compressed-tensors": 0.5,
    "torchao": 0.5,
    "quark": 0.5,
    "moe_wna16": 0.5,
    "inc": 0.5,
    "experts_int8": 1,
}


#### Basic Data Types ####
# 基础数据类型


@dataclass
class DebugPerfStats:
    """用于调试指标计算的统计数据。

    当环境变量 VLLM_DEBUG_MFU_METRICS 启用时，记录详细的计算信息，
    包括计算耗时、请求分类、以及各组件的 FLOPs 和内存带宽明细。

    Attributes:
        calc_duration: 计算这些统计所花费的时间（秒）。
        num_prefill_requests: 预填充请求数。
        num_decode_requests: 解码请求数。
        context_breakdown: 执行上下文的详细分解（来自 ExecutionContext 的字典）。
        num_flops_per_gpu_breakdown: 每 GPU 的 FLOPs 分项明细。
        num_read_bytes_per_gpu_breakdown: 每 GPU 的内存读取字节分项明细。
        num_write_bytes_per_gpu_breakdown: 每 GPU 的内存写字节分项明细。
    """

    ## Stats for debugging the metrics calculation
    calc_duration: float = 0.0  # time spent calculating these stats
    num_prefill_requests: int = 0
    num_decode_requests: int = 0
    context_breakdown: dict[str, int] | None = None
    num_flops_per_gpu_breakdown: dict[str, int] | None = None
    num_read_bytes_per_gpu_breakdown: dict[str, int] | None = None
    num_write_bytes_per_gpu_breakdown: dict[str, int] | None = None


@dataclass
class PerfStats:
    """单步（step）的性能统计数据。

    每次调度迭代计算一次，包含该步骤中每 GPU 的估算 FLOPs 和内存带宽。
    这些数据用于：
    1. 日志输出：计算平均 TFLOPS 和 GB/s
    2. Prometheus 上报：累积计数器
    3. MFU 计算：与理论峰值比较

    Attributes:
        num_flops_per_gpu: 每 GPU 的估算浮点运算数。
        num_read_bytes_per_gpu: 每 GPU 的估算内存读取字节数。
        num_write_bytes_per_gpu: 每 GPU 的估算内存写字节数。
        debug_stats: 调试统计数据（仅在 VLLM_DEBUG_MFU_METRICS 启用时填充）。
    """

    num_flops_per_gpu: int = 0
    num_read_bytes_per_gpu: int = 0
    num_write_bytes_per_gpu: int = 0
    debug_stats: DebugPerfStats | None = None


@dataclass
class ExecutionContext:
    """请求批次的执行上下文。

    聚合一批请求的统计数据，分别跟踪预填充（Prefill）和解码（Decode）阶段。
    这些统计数据是计算 FLOPs 和内存带宽的输入。

    预填充阶段：处理输入 prompt 的所有 token，计算量与 token 数和上下文长度相关。
    解码阶段：逐 token 生成，计算量较小但需要读取完整的 KV Cache。

    关键统计量：
    - total_num_tokens()：所有请求的 token 总数，影响权重读取量
    - total_token_context_product()：num_tokens * context_len 的总和，
      影响注意力计算的 FLOPs 和 KV Cache 读取量
    - num_logits_tokens()：需要计算 logits 的 token 数，
      影响反嵌入层的计算量

    Example:
        一个批次包含一个完整预填充（2048 tokens）和一个解码（1 token, 8192 上下文）：
        ctx = ExecutionContext()
        ctx.add(2048, 2048, is_prefill=True)
        ctx.add(1, 8192, is_prefill=False)
    """

    # 预填充阶段统计
    num_prefill_requests: int = 0
    prefill_num_tokens: int = 0  # sum of num_tokens for prefill requests
    prefill_context_len: int = 0  # sum of context_len for prefill requests
    prefill_token_context_product: int = 0  # sum of (num_tokens * context_len)

    # 解码阶段统计
    num_decode_requests: int = 0
    decode_num_tokens: int = 0  # sum of num_tokens for decode requests
    decode_context_len: int = 0  # sum of context_len for decode requests
    decode_token_context_product: int = 0  # sum of (num_tokens * context_len)

    def add(self, num_tokens: int, context_len: int, is_prefill: bool) -> None:
        """添加一个请求的统计到批次上下文中。

        Args:
            num_tokens: 该请求在本次迭代中处理的 token 数。
                预填充可能为数百到数千，解码通常为 1。
            context_len: 该请求的总上下文长度。
                等于 num_computed_tokens + num_tokens。
            is_prefill: 是否为预填充阶段。
        """
        if is_prefill:
            self.num_prefill_requests += 1
            self.prefill_num_tokens += num_tokens
            self.prefill_context_len += context_len
            self.prefill_token_context_product += num_tokens * context_len
        else:
            self.num_decode_requests += 1
            self.decode_num_tokens += num_tokens
            self.decode_context_len += context_len
            self.decode_token_context_product += num_tokens * context_len

    def total_num_tokens(self) -> int:
        """批次中所有请求的 token 总数。"""
        return self.prefill_num_tokens + self.decode_num_tokens

    def total_token_context_product(self) -> int:
        """所有请求的 (num_tokens * context_len) 总和。

        用于计算注意力机制中的 Q*K^T 和 A*V 的 FLOPs，
        以及 KV Cache 的读取量。
        """
        return self.prefill_token_context_product + self.decode_token_context_product

    def num_logits_tokens(self) -> int:
        """需要计算 logits（反嵌入）的 token 数。

        计算规则：
        - 预填充：每个请求只需对最后一个 token 计算 logits
          （因为前面的 token 在预填充中不需要输出）
        - 解码：每个 token 都需要计算 logits
        """
        return self.num_prefill_requests + self.decode_num_tokens

    @classmethod
    def from_single_request(
        cls, num_tokens: int, context_len: int, is_prefill: bool
    ) -> "ExecutionContext":
        """从单个请求创建 ExecutionContext（主要用于测试）。

        Args:
            num_tokens: 请求数。
            context_len: 上下文长度。
            is_prefill: 是否为预填充。

        Returns:
            包含单个请求统计的 ExecutionContext。
        """
        ctx = cls()
        ctx.add(num_tokens, context_len, is_prefill)
        return ctx


class ParsedArgs:
    """解析器链的参数容器。

    提供点号（dot notation）语法糖来访问和更新解析参数。
    与普通字典相比，使用更直观。

    Example:
        args = ParsedArgs()
        args.x = 3
        args.y = args.x + 1
    """

    def __getattr__(self, name: str) -> Any:
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        object.__setattr__(self, name, value)

    def model_dump(self) -> dict[str, Any]:
        return vars(self).copy()


#### Abstract ####
# 抽象基类和协议


class Parser(Protocol):
    """解析器协议。

    解析器负责从 VllmConfig 中提取特定字段并更新 ParsedArgs。
    如果解析器发现当前配置不适用（如缺少必要的配置字段），
    应该静默跳过而不报错。
    """

    def parse(self, args: ParsedArgs, vllm_config: VllmConfig) -> ParsedArgs:
        """
        Parse the vllm config and update the current ParsedArgs and pass it on.
        If the parser isn't applicable to the vllm_config, it will do nothing.
        """
        ...


class ParserChain:
    """解析器链：按顺序应用一系列解析器。

    解析器链是策略模式的应用，将配置解析分解为多个独立的步骤。
    每个解析器负责解析特定的配置字段，后续的解析器可能覆盖
    前面解析器的结果（如量化配置覆盖默认的权重字节大小）。

    使用场景：
    - BaseConfigParser -> BaseAttentionConfigParser -> AttentionQuantizationConfigParser
      依次解析基础配置、注意力配置、量化配置
    - BaseConfigParser -> FfnParallelParser -> BaseFfnConfigParser -> ...
      依次解析基础配置、并行策略、FFN 配置、MoE 配置等

    Attributes:
        parsers: 解析器列表。
    """

    def __init__(self, *parsers: Parser) -> None:
        self.parsers = list(parsers)

    def add_parser(self, parser: Parser) -> None:
        """向链中追加一个解析器。"""
        self.parsers.append(parser)

    def parse(self, vllm_config: VllmConfig) -> ParsedArgs:
        """依次执行所有解析器，返回最终的解析结果。

        Args:
            vllm_config: vLLM 的完整配置对象。

        Returns:
            包含所有解析字段的 ParsedArgs 实例。
        """
        args = ParsedArgs()
        for parser in self.parsers:
            args = parser.parse(args, vllm_config)
        return args


# 组件指标注册表：在 ComponentMetrics 的子类定义时自动注册。
# 键为组件类型名称（如 "attn"、"ffn"、"unembed"），
# 值为对应的 ComponentMetrics 子类。
_COMPONENT_METRICS_REGISTRY: dict[str, type["ComponentMetrics"]] = {}


class ComponentMetrics(BaseModel, ABC):
    """组件指标的抽象基类。

    每个具体的 ComponentMetrics 子类关联：
    1. 字段：通过 Pydantic 模型定义和验证所需的配置字段
    2. 解析器：通过 ParserChain 从 VllmConfig 解析字段值
    3. 指标方法：根据 ExecutionContext 计算 FLOPs 和内存带宽

    子类自动注册到 _COMPONENT_METRICS_REGISTRY 中。
    ModelMetrics 通过遍历注册表来实例化所有组件指标。
    """

    @classmethod
    @abstractmethod
    def component_type(cls) -> str: ...

    @classmethod
    @abstractmethod
    def get_parser(cls) -> ParserChain:
        """
        Return a ParserChain that provides values for all required fields.
        The returned parser chain must populate ParsedArgs with values for every
        field defined on this ComponentMetrics class. Missing fields will cause
        a ValidationError when from_vllm_config() is called.
        See individual Parser docstrings for which args they provide, and field
        comments on ComponentMetrics subclasses for which parser provides each field.
        """
        ...

    def __init_subclass__(cls):
        """子类定义时自动注册到组件指标注册表。"""
        _COMPONENT_METRICS_REGISTRY[cls.component_type()] = cls

    @classmethod
    def from_vllm_config(cls, vllm_config: VllmConfig) -> Self:
        """从 VllmConfig 实例化组件指标。

        流程：
        1. 获取该组件的解析器链
        2. 执行解析器链，从 VllmConfig 提取字段值
        3. 使用 Pydantic 的 model_validate 验证并创建实例
        4. 如果验证失败，抛出 InvalidComponent 异常

        Args:
            vllm_config: vLLM 的完整配置对象。

        Returns:
            解析后的组件指标实例。

        Raises:
            InvalidComponent: 如果配置不适用于该组件。
        """

        parser = cls.get_parser()
        parsed_args = parser.parse(vllm_config)
        try:
            return cls.model_validate(parsed_args.model_dump())
        except ValidationError as e:
            raise InvalidComponent(f"Invalid {cls.component_type()} config: {e}") from e

    @classmethod
    def registered_metrics(cls) -> Iterable[type["ComponentMetrics"]]:
        """返回所有已注册的组件指标类。"""
        return iter(_COMPONENT_METRICS_REGISTRY.values())

    @abstractmethod
    def get_num_flops_breakdown(
        self, ctx: ExecutionContext, per_gpu: bool = True
    ) -> dict[str, int]: ...

    @abstractmethod
    def get_read_bytes_breakdown(
        self, ctx: ExecutionContext, per_gpu: bool = True
    ) -> dict[str, int]: ...

    @abstractmethod
    def get_write_bytes_breakdown(
        self, ctx: ExecutionContext, per_gpu: bool = True
    ) -> dict[str, int]: ...

    def get_num_flops(self, ctx: ExecutionContext, per_gpu: bool = True) -> int:
        """计算该组件的总 FLOPs。"""
        return sum(self.get_num_flops_breakdown(ctx, per_gpu).values())

    def get_read_bytes(self, ctx: ExecutionContext, per_gpu: bool = True) -> int:
        """计算该组件的总内存读取字节数。"""
        return sum(self.get_read_bytes_breakdown(ctx, per_gpu).values())

    def get_write_bytes(self, ctx: ExecutionContext, per_gpu: bool = True) -> int:
        """计算该组件的总内存写字节数。"""
        return sum(self.get_write_bytes_breakdown(ctx, per_gpu).values())


#### parsers ####
# 配置解析器


class BaseConfigParser(Parser):
    """基础模型配置解析器。

    从 VllmConfig 中提取模型的基本架构参数。
    这些参数被所有组件指标共享。

    提供的字段：
    - vocab_size：词表大小
    - hidden_size：隐藏层维度
    - num_attention_heads：注意力头总数（未除以 TP）
    - num_hidden_layers：Transformer 层数
    - weight_byte_size：权重数据类型的字节大小（默认基于模型 dtype）
    - activation_byte_size：激活值的字节大小（硬编码为 2，即 bf16）
    - dp_size：数据并行大小
    - tp_size：张量并行大小
    - pp_size：流水线并行大小
    - enable_ep：是否启用专家并行
    """

    def parse(self, args: ParsedArgs, vllm_config: VllmConfig) -> ParsedArgs:
        model_config = vllm_config.model_config

        args.vocab_size = model_config.get_vocab_size()
        args.hidden_size = model_config.get_hidden_size()
        # NOTE: model_config.get_attention_heads() divide by TP
        # so we access field manually here to get total num_heads
        args.num_attention_heads = get_required(
            model_config.hf_text_config, "num_attention_heads"
        )
        args.num_hidden_layers = get_required(
            model_config.hf_text_config, "num_hidden_layers"
        )

        model_dtype = vllm_config.model_config.dtype

        if isinstance(model_dtype, torch.dtype):
            torch_dtype = model_dtype
        elif isinstance(model_dtype, str) and model_dtype in STR_DTYPE_TO_TORCH_DTYPE:
            torch_dtype = STR_DTYPE_TO_TORCH_DTYPE[model_dtype]
        else:
            # FIXME: handle this better
            logger.warning(
                "Unknown model_dtype %s, defaulting to bfloat16",
                model_dtype,
            )
            torch_dtype = torch.bfloat16

        args.weight_byte_size = get_dtype_size(torch_dtype)

        # FIXME: handle this better by parsing whether activations use
        # bf16, fp32, etc...
        args.activation_byte_size = 2

        args.dp_size = vllm_config.parallel_config.data_parallel_size
        args.tp_size = vllm_config.parallel_config.tensor_parallel_size
        args.pp_size = vllm_config.parallel_config.pipeline_parallel_size
        args.enable_ep = vllm_config.parallel_config.enable_expert_parallel

        return args


#### Attention ####
# 注意力层指标


class BaseAttentionConfigParser(Parser):
    """注意力特定配置解析器。

    提取注意力层特有的配置参数。

    提供的字段：
    - num_key_value_heads：KV 头数（用于 GQA/MQA 模型）
    - head_dim：每个注意力头的维度
    - cache_byte_size：KV Cache 的字节大小（取决于缓存数据类型）
    """

    def parse(self, args: ParsedArgs, vllm_config: VllmConfig) -> ParsedArgs:
        model_config = vllm_config.model_config

        args.num_key_value_heads = model_config.get_total_num_kv_heads()
        args.head_dim = model_config.get_head_size()

        model_dtype = vllm_config.model_config.dtype
        cache_dtype = vllm_config.cache_config.cache_dtype

        kv_cache_torch_dtype = get_kv_cache_torch_dtype(cache_dtype, model_dtype)
        args.cache_byte_size = get_dtype_size(kv_cache_torch_dtype)

        return args


class AttentionQuantizationConfigParser(Parser):
    """注意力层量化配置解析器。

    如果模型使用了量化，覆盖默认的 weight_byte_size。
    不同的量化方法对应不同的权重字节大小（参见 _QUANT_WEIGHT_BYTE_SIZE）。

    覆盖的字段：
    - weight_byte_size：从默认的模型 dtype 字节大小覆盖为量化后的字节大小
    """

    def parse(self, args: ParsedArgs, vllm_config: VllmConfig) -> ParsedArgs:
        cfg = vllm_config.quant_config

        if cfg is None:
            return args

        quant_method = cfg.get_name()
        if quant_method in _QUANT_WEIGHT_BYTE_SIZE:
            args.weight_byte_size = _QUANT_WEIGHT_BYTE_SIZE[quant_method]
        else:
            raise InvalidComponent(
                f"Unsupported quantization method for attention metrics: {quant_method}"
            )

        return args


class AttentionMetrics(ComponentMetrics):
    """注意力层的性能指标。

    计算 Transformer 注意力层的 FLOPs 和内存带宽。
    注意力层包含以下子操作：
    1. QKV 投影：将隐藏状态分别投影为 Q、K、V
    2. 注意力计算：Q*K^T（注意力分数）和 A*V（注意力加权）
    3. 输出投影：将注意力输出投影回隐藏维度

    字段来源说明：
    - num_hidden_layers, hidden_size, num_attention_heads,
      activation_byte_size, tp_size, pp_size: 来自 BaseConfigParser
    - num_key_value_heads, head_dim, cache_byte_size: 来自 BaseAttentionConfigParser
    - weight_byte_size: 来自 BaseConfigParser，可被 AttentionQuantizationConfigParser 覆盖

    TODO: 区分不同类型的注意力层（如 SWA、MLA 等）的情况。
    """

    # From BaseConfigParser
    num_hidden_layers: int = Field(..., gt=0)
    hidden_size: int = Field(..., gt=0)
    num_attention_heads: int = Field(..., gt=0)
    activation_byte_size: int = Field(..., gt=0)
    tp_size: int = Field(..., gt=0)
    pp_size: int = Field(..., gt=0)

    # From BaseAttentionConfigParser
    num_key_value_heads: int = Field(..., gt=0)
    head_dim: int = Field(..., gt=0)
    cache_byte_size: int = Field(..., gt=0)

    # From BaseConfig Parser, overridden by AttentionQuantizationConfigParser
    weight_byte_size: int | float = Field(..., gt=0)

    # TODO: discern cases where we have mixture of different attention layer types
    # such as SWA, MLA, etc.

    @classmethod
    def component_type(cls) -> str:
        return "attn"

    @classmethod
    def get_parser(cls) -> ParserChain:
        return ParserChain(
            BaseConfigParser(),
            BaseAttentionConfigParser(),
            AttentionQuantizationConfigParser(),
        )

    def get_num_flops_breakdown(
        self, ctx: ExecutionContext, per_gpu: bool = True
    ) -> dict[str, int]:
        """计算注意力层的 FLOPs 分项明细。

        变量说明：
        - L: 注意力层数
        - D: 隐藏维度 (hidden_size)
        - q: 查询头数 (num_attention_heads)
        - kv: KV 头数 (num_key_value_heads)，GQA/MQA 时小于 q
        - d: 每头维度 (head_dim)
        - T: 总 token 数
        - TC: num_tokens * context_len 的总和

        FLOPs 计算公式（每个 GEMM 操作的 FLOPs = 2 * M * N * K）：
        1. qkv_proj: QKV 投影的 FLOPs
           = 2 * T * D * (q + 2*kv) * d * L
           输入 T*D，输出 (q+2*kv)*d
        2. attn_qk: Q*K^T 注意力分数计算
           = 2 * q * TC * d * L
           每个头：query_len * context_len * head_dim
        3. attn_av: A*V 注意力加权
           = 2 * q * TC * d * L
           与 Q*K^T 对称
        4. out_proj: 输出投影
           = 2 * T * D * q * d * L
           输入 q*d，输出 D

        当 per_gpu=True 时，根据并行策略调整：
        - PP：层数除以 pp_size
        - TP：头数除以 tp_size（向上取整到至少 1）
        """
        L, D, q, kv, d = (
            self.num_hidden_layers,
            self.hidden_size,
            self.num_attention_heads,
            self.num_key_value_heads,
            self.head_dim,
        )
        T = ctx.total_num_tokens()
        TC = ctx.total_token_context_product()

        if per_gpu:
            L //= self.pp_size
            # tensor parallel along heads
            q = max(1, q // self.tp_size)
            kv = max(1, kv // self.tp_size)

        return {
            "qkv_proj": 2 * T * D * (q + 2 * kv) * d * L,
            "attn_qk": 2 * q * TC * d * L,
            "attn_av": 2 * q * TC * d * L,
            "out_proj": 2 * T * D * q * d * L,
        }

    def get_read_bytes_breakdown(
        self, ctx: ExecutionContext, per_gpu: bool = True
    ) -> dict[str, int]:
        """计算注意力层的内存读取字节分项明细。

        读取内容包括：
        1. qkv_input: QKV 投影的输入激活值
        2. qkv_weight: QKV 投影的权重
        3. attn_input: 注意力计算的输入（Q、K、V）
           - 预填充：读取 Q、K、V 激活值（activation_byte_size）
           - 解码：读取 Q 激活值 + 从 KV Cache 读取 K、V（cache_byte_size）
        4. out_input: 输出投影的输入激活值
        5. out_weight: 输出投影的权重

        注意：预填充和解码阶段的注意力输入读取模式不同：
        - 预填充：Q、K、V 都是新计算的激活值
        - 解码：只有 Q 是新计算的，K、V 从缓存中读取
        """
        L, D, q, kv, d = (
            self.num_hidden_layers,
            self.hidden_size,
            self.num_attention_heads,
            self.num_key_value_heads,
            self.head_dim,
        )
        T = ctx.total_num_tokens()

        if per_gpu:
            L //= self.pp_size
            # tensor parallel along heads
            q = max(1, q // self.tp_size)
            kv = max(1, kv // self.tp_size)

        read_bytes = {}

        read_bytes["qkv_input"] = T * D * self.activation_byte_size * L
        read_bytes["qkv_weight"] = int(D * (q + 2 * kv) * d * self.weight_byte_size * L)

        # Attention input reads differ between prefill and decode
        # Prefill: read Q, K, V activations (all in activation_byte_size)
        if ctx.prefill_num_tokens > 0:
            read_bytes["attn_input"] = (
                (ctx.prefill_num_tokens * q + 2 * ctx.prefill_context_len * kv)
                * d
                * self.activation_byte_size
                * L
            )

        # Decode: read Q activations + read K, V from cache (in cache_byte_size)
        if ctx.decode_num_tokens > 0:
            read_bytes["attn_input"] = read_bytes.get("attn_input", 0) + (
                ctx.decode_num_tokens * q * d * self.activation_byte_size * L
                + 2 * ctx.decode_context_len * kv * d * self.cache_byte_size * L
            )

        read_bytes["out_input"] = T * q * d * self.activation_byte_size * L
        read_bytes["out_weight"] = int(q * d * D * self.weight_byte_size * L)

        return read_bytes

    def get_write_bytes_breakdown(
        self, ctx: ExecutionContext, per_gpu: bool = True
    ) -> dict[str, int]:
        """Calculate write memory traffic for attention layers."""
        """计算注意力层的内存写字节分项明细。

        写入内容包括：
        1. qkv_output: QKV 投影的输出激活值
        2. kv_cache: 写入 KV Cache 的 K 和 V 值
        3. out_output: 输出投影的输出激活值
        """
        L, D, q, kv, d = (
            self.num_hidden_layers,
            self.hidden_size,
            self.num_attention_heads,
            self.num_key_value_heads,
            self.head_dim,
        )
        T = ctx.total_num_tokens()

        if per_gpu:
            L //= self.pp_size
            # tensor parallel along heads
            q = max(1, q // self.tp_size)
            kv = max(1, kv // self.tp_size)

        return {
            "qkv_output": T * (q + 2 * kv) * d * self.activation_byte_size * L,
            "kv_cache": 2 * T * kv * d * self.cache_byte_size * L,
            "out_output": T * D * self.activation_byte_size * L,
        }


#### Ffn ####
# 前馈网络层指标


class BaseFfnConfigParser(Parser):
    """FFN 和 MoE 配置解析器。

    从模型配置中提取前馈网络（FFN）和混合专家（MoE）相关参数。

    提供的字段：
    - intermediate_size：FFN 中间层维度（默认为 hidden_size * 4）
    - num_experts：专家总数（MoE 模型）
    - num_experts_per_tok：每个 token 激活的专家数（MoE top-k）
    - moe_intermediate_size：MoE 专家的中间层维度
    - num_shared_experts：共享专家数
    - num_moe_layers：MoE 层数（默认所有层都是 MoE）
    """

    def parse(self, args: ParsedArgs, vllm_config: VllmConfig) -> ParsedArgs:
        cfg = vllm_config.model_config.hf_config
        if hasattr(cfg, "text_config") and cfg.text_config is not None:
            cfg = cfg.text_config

        args.intermediate_size = getattr(cfg, "intermediate_size", args.hidden_size * 4)

        # Try different naming conventions.
        args.num_experts = vllm_config.model_config.get_num_experts()
        args.num_experts_per_tok = getattr_from_list(
            cfg, ["num_experts_per_tok", "moe_topk"], 0
        )
        args.moe_intermediate_size = getattr_from_list(
            cfg, ["moe_intermediate_size", "intermediate_size"], 0
        )
        args.num_shared_experts = getattr_from_list(
            cfg, ["n_shared_experts", "num_shared_experts"], 0
        )

        is_moe = args.num_experts != 0
        # Assume all MoE layers by default
        args.num_moe_layers = args.num_hidden_layers if is_moe else 0

        return args


class FfnParallelParser(Parser):
    """FFN 并行策略解析器。

    计算 FFN 的实际张量并行大小和专家并行大小。
    注意：FFN 的 TP 大小不等于全局 TP 参数。

    例如：DP2TP4 配置下：
    - 未启用 EP：FFN 使用 TP8（dp_size * tp_size = 2 * 4）
    - 启用 EP：FFN 使用 EP8（dp_size * tp_size = 2 * 4）

    提供的字段：
    - ffn_tp_size: FFN 的张量并行大小
    - ffn_ep_size: FFN 的专家并行大小
    """

    def parse(self, args: ParsedArgs, vllm_config: VllmConfig) -> ParsedArgs:
        # NOTE: ffn tp_size does not equal the tp_size parameter directly.
        # e.g.) If we use DP2TP4, ffn will use TP8 (or EP8 if EP is enabled.)
        if args.enable_ep:
            ffn_tp_size, ffn_ep_size = 1, args.dp_size * args.tp_size
        else:
            ffn_tp_size, ffn_ep_size = args.dp_size * args.tp_size, 1

        args.ffn_tp_size = ffn_tp_size
        args.ffn_ep_size = ffn_ep_size

        return args


class InterleaveMoeLayerStepParser(Parser):
    """交错 MoE 层步长解析器（用于 Llama4 等模型）。

    Llama4 等模型使用交错的 MoE 结构，即并非所有层都是 MoE 层，
    而是每隔 interleave_moe_layer_step 层才有一个 MoE 层。

    例如：interleave_moe_layer_step=2, num_hidden_layers=8
    MoE 层索引：1, 3, 5, 7（即第 2, 4, 6, 8 层）
    num_moe_layers = 4

    覆盖的字段：
    - num_moe_layers: MoE 层的实际数量
    """

    def parse(self, args: ParsedArgs, vllm_config: VllmConfig) -> ParsedArgs:
        cfg = vllm_config.model_config.hf_config
        if hasattr(cfg, "text_config") and cfg.text_config is not None:
            cfg = cfg.text_config

        if (
            hasattr(cfg, "interleave_moe_layer_step")
            and cfg.interleave_moe_layer_step > 0
        ):
            args.num_moe_layers = len(
                [
                    layer
                    for layer in range(args.num_hidden_layers)
                    if (layer + 1) % cfg.interleave_moe_layer_step == 0
                ]
            )

        return args


class MoeLayerFreqParser(Parser):
    """MoE 层频率解析器（用于 DeepSeek 等模型）。

    DeepSeek 等模型使用 first_k_dense_replace 和 moe_layer_freq
    来定义 MoE 层的分布：
    - 前 first_k_dense_replace 层是稠密层
    - 之后每隔 moe_layer_freq 层是 MoE 层

    例如：first_k_dense_replace=3, moe_layer_freq=2, num_hidden_layers=10
    MoE 层索引：3, 5, 7, 9
    num_moe_layers = 4

    覆盖的字段：
    - num_moe_layers: MoE 层的实际数量
    """

    def parse(self, args: ParsedArgs, vllm_config: VllmConfig) -> ParsedArgs:
        cfg = vllm_config.model_config.hf_config
        if hasattr(cfg, "text_config") and cfg.text_config is not None:
            cfg = cfg.text_config

        if hasattr(cfg, "moe_layer_freq") and hasattr(cfg, "first_k_dense_replace"):
            args.num_moe_layers = len(
                [
                    layer
                    for layer in range(args.num_hidden_layers)
                    if layer >= cfg.first_k_dense_replace
                    and layer % cfg.moe_layer_freq == 0
                ]
            )

        return args


class FfnQuantizationConfigParser(Parser):
    """FFN 层量化配置解析器。

    如果模型使用了量化，覆盖默认的 weight_byte_size。
    与注意力层的量化解析器逻辑相同。

    覆盖的字段：
    - weight_byte_size: 从默认的模型 dtype 字节大小覆盖为量化后的字节大小
    """

    def parse(self, args: ParsedArgs, vllm_config: VllmConfig) -> ParsedArgs:
        cfg = vllm_config.quant_config

        if cfg is None:
            return args

        quant_method = cfg.get_name()
        if quant_method in _QUANT_WEIGHT_BYTE_SIZE:
            args.weight_byte_size = _QUANT_WEIGHT_BYTE_SIZE[quant_method]
        else:
            raise InvalidComponent(
                f"Unsupported quantization method for FFN metrics: {quant_method}"
            )

        return args


class FfnMetrics(ComponentMetrics):
    """前馈网络（FFN）层的性能指标。

    计算 Transformer FFN 层的 FLOPs 和内存带宽。
    FFN 层分为三种类型：

    1. 稠密 FFN（Dense FFN）：
       使用 SwiGLU 激活函数，包含三个线性层：
       - up_proj: 上投影 (D -> DI)
       - gate_proj: 门控投影 (D -> DI)
       - down_proj: 下投影 (DI -> D)
       其中 D = hidden_size, DI = intermediate_size

    2. MoE 路由专家（Routed Experts）：
       每个 token 只激活 E 个专家（top-k 路由）。
       每个专家的结构与稠密 FFN 相同。
       计算量 = 稠密 FFN * E（考虑负载均衡）

    3. MoE 共享专家（Shared Experts）：
       S 个共享专家对所有 token 都执行。
       计算量 = 稠密 FFN * S

    字段来源说明：
    - num_hidden_layers, hidden_size, activation_byte_size, pp_size:
      来自 BaseConfigParser
    - ffn_tp_size, ffn_ep_size: 来自 FfnParallelParser
    - intermediate_size, num_experts, num_experts_per_tok,
      moe_intermediate_size, num_shared_experts: 来自 BaseFfnConfigParser
    - num_moe_layers: 来自 BaseConfigParser，可被 InterleaveMoeLayerStep
      或 MoeLayerFreq 解析器覆盖
    - weight_byte_size: 来自 BaseConfigParser，可被 FfnQuantizationConfigParser 覆盖
    """

    # From BaseConfigParser
    num_hidden_layers: int = Field(..., gt=0)
    hidden_size: int = Field(..., gt=0)
    activation_byte_size: int = Field(..., gt=0)
    pp_size: int = Field(..., gt=0)

    # From FfnParallelParser
    ffn_tp_size: int = Field(..., gt=0)
    ffn_ep_size: int = Field(..., gt=0)

    # From BaseFfnConfigParser
    intermediate_size: int = Field(..., gt=0)
    num_experts: int = Field(0)
    num_experts_per_tok: int = Field(1)
    moe_intermediate_size: int = Field(0)
    num_shared_experts: int = Field(0)

    # From BaseConfigParser, can be overridden InterleaveMoeLayerStep or MoeLayerFreq
    num_moe_layers: int = Field(..., ge=0)

    # FIXME: might have to make this more granular
    # (i.e. dense_weight_byte_size, moe_routed_weight_byte_size,
    # moe_shared_weight_byte_size)
    # since it can differ from byte size of other components (e.g. attn)
    # and can differ even from each other.

    # From BaseConfigParser, can be overridden by FfnQuantizationConfigParser
    weight_byte_size: int | float = Field(..., gt=0)

    @model_validator(mode="after")
    def validate_moe_fields(self) -> Self:
        """验证 MoE 相关字段在 num_moe_layers > 0 时已正确设置。"""
        if self.num_moe_layers > 0:
            assert self.num_experts, f"{self.num_experts=}"
            assert self.num_experts_per_tok, f"{self.num_experts_per_tok=}"
            assert self.moe_intermediate_size, f"{self.moe_intermediate_size=}"
        return self

    @classmethod
    def component_type(cls) -> str:
        return "ffn"

    @classmethod
    def get_parser(cls) -> ParserChain:
        return ParserChain(
            BaseConfigParser(),
            FfnParallelParser(),
            BaseFfnConfigParser(),
            InterleaveMoeLayerStepParser(),
            MoeLayerFreqParser(),
            FfnQuantizationConfigParser(),
        )

    def get_num_flops_breakdown(
        self, ctx: ExecutionContext, per_gpu: bool = True
    ) -> dict[str, int]:
        """Calculate flops breakdown for FFN layers."""
        """计算 FFN 层的 FLOPs 分项明细。

        变量说明：
        - L: 总层数
        - D: 隐藏维度 (hidden_size)
        - DI: 中间维度 (intermediate_size)
        - Lm: MoE 层数
        - E: 每个 token 激活的专家数 (num_experts_per_tok)
        - MI: MoE 中间维度 (moe_intermediate_size)
        - S: 共享专家数 (num_shared_experts)
        - T: 总 token 数
        - Ld: 稠密层数 = L - Lm

        FLOPs 计算（SwiGLU 包含 3 个线性层：up, gate, down）：
        1. dense_ffn: 稠密 FFN 的 FLOPs
           = 2 * D * 3 * DI * T * Ld
           每层有 3 个 GEMM，每个 GEMM 的 FLOPs = 2 * 输入维度 * 输出维度 * token 数

        2. routed_ffn: MoE 路由专家的 FLOPs
           = 2 * D * 3 * MI * num_activated_tokens * Lm
           num_activated_tokens = T * E（每个 token 激活 E 个专家）

        3. shared_ffn: MoE 共享专家的 FLOPs
           = 2 * D * 3 * MI * S * T * Lm
           S 个共享专家对所有 token 都执行

        当 per_gpu=True 时：
        - Ld 和 Lm 除以 pp_size（流水线并行）
        - DI 和 MI 除以 ffn_tp_size（张量并行）
        - num_activated_tokens 除以 ffn_ep_size（专家并行）
        """
        L, D, DI = self.num_hidden_layers, self.hidden_size, self.intermediate_size
        Lm, E, MI, S = (
            self.num_moe_layers,
            self.num_experts_per_tok,
            self.moe_intermediate_size,
            self.num_shared_experts,
        )
        T = ctx.total_num_tokens()

        Ld = L - Lm

        num_activated_tokens = T * E if E else 0

        if per_gpu:
            Ld //= self.pp_size
            Lm //= self.pp_size

            DI //= self.ffn_tp_size
            if MI is not None:
                MI //= self.ffn_tp_size
            if E:
                num_activated_tokens //= self.ffn_ep_size

        flops = {}

        # Dense FFN layers (SwiGLU: 3 linear layers: up, gate, down)
        if Ld:
            flops["dense_ffn"] = 2 * D * 3 * DI * T * Ld

        # MoE routed experts (each token activates E experts)
        if Lm and E:
            flops["routed_ffn"] = 2 * D * 3 * MI * num_activated_tokens * Lm

        # MoE shared experts (all S shared experts run for every token)
        if Lm and S:
            flops["shared_ffn"] = 2 * D * 3 * MI * S * T * Lm

        return flops

    def get_read_bytes_breakdown(
        self, ctx: ExecutionContext, per_gpu: bool = True
    ) -> dict[str, int]:
        """Calculate read memory traffic for FFN layers."""
        """计算 FFN 层的内存读取字节分项明细。

        读取内容分为三组：

        1. 稠密 FFN 层（3 个 GEMM）：
           - dense_up_gate_input: up_proj 和 gate_proj 的输入激活值
           - dense_up_gate_weights: up_proj 和 gate_proj 的权重（2 个 GEMM）
           - dense_silu_input: SiLU 激活函数的输入
           - dense_down_input: down_proj 的输入激活值
           - dense_down_weights: down_proj 的权重

        2. MoE 路由专家：
           - routed_up_gate_input: 路由专家的 up/gate 输入
           - routed_up_gate_weights: 路由专家的 up/gate 权重
             （按激活的专家数计算，假设完美负载均衡）
           - routed_silu_input: 路由专家的 SiLU 输入
           - routed_down_input/down_weights: 路由专家的 down 层

        3. MoE 共享专家：
           - shared_*: 与路由专家类似，但乘以共享专家数 S
        """
        L, D, DI = self.num_hidden_layers, self.hidden_size, self.intermediate_size
        Lm, E, MI, S = (
            self.num_moe_layers,
            self.num_experts_per_tok,
            self.moe_intermediate_size,
            self.num_shared_experts,
        )
        T = ctx.total_num_tokens()
        num_experts = self.num_experts

        Ld = L - Lm

        num_activated_tokens = T * E if E else 0

        if per_gpu:
            Ld //= self.pp_size
            Lm //= self.pp_size

            DI //= self.ffn_tp_size
            if MI is not None:
                MI //= self.ffn_tp_size
            if E:
                num_activated_tokens //= self.ffn_ep_size
            if num_experts is not None:
                num_experts //= self.ffn_ep_size

        read_bytes = {}

        # Dense FFN layers (3 GEMMs: up, gate, down projections + SiLU activation)
        if Ld:
            read_bytes["dense_up_gate_input"] = int(
                T * D * self.activation_byte_size * Ld
            )
            read_bytes["dense_up_gate_weights"] = int(
                2 * D * DI * self.weight_byte_size * Ld
            )
            read_bytes["dense_silu_input"] = int(
                2 * T * DI * self.activation_byte_size * Ld
            )
            read_bytes["dense_down_input"] = int(
                T * DI * self.activation_byte_size * Ld
            )
            read_bytes["dense_down_weights"] = int(D * DI * self.weight_byte_size * Ld)

        if Lm:
            # MoE routed expert reads
            if E:
                # FIXME: Assume perfect load balancing for now.
                num_activated_experts = min(num_activated_tokens, num_experts)

                read_bytes["routed_up_gate_input"] = int(
                    num_activated_tokens * D * self.activation_byte_size * Lm
                )
                read_bytes["routed_up_gate_weights"] = int(
                    2 * D * MI * num_activated_experts * self.weight_byte_size * Lm
                )
                read_bytes["routed_silu_input"] = int(
                    2 * num_activated_tokens * MI * self.activation_byte_size * Lm
                )
                read_bytes["routed_down_input"] = int(
                    num_activated_tokens * MI * self.activation_byte_size * Lm
                )
                read_bytes["routed_down_weights"] = int(
                    D * MI * num_activated_experts * self.weight_byte_size * Lm
                )

            # MoE shared expert reads
            if S:
                read_bytes["shared_up_gate_input"] = int(
                    T * D * self.activation_byte_size * Lm
                )
                read_bytes["shared_up_gate_weights"] = int(
                    2 * D * MI * S * self.weight_byte_size * Lm
                )
                read_bytes["shared_silu_input"] = int(
                    2 * T * MI * S * self.activation_byte_size * Lm
                )
                read_bytes["shared_down_input"] = int(
                    T * MI * self.activation_byte_size * Lm
                )
                read_bytes["shared_down_weights"] = int(
                    D * MI * S * self.weight_byte_size * Lm
                )

        return read_bytes

    def get_write_bytes_breakdown(
        self, ctx: ExecutionContext, per_gpu: bool = True
    ) -> dict[str, int]:
        """Calculate write memory traffic for FFN layers."""
        """计算 FFN 层的内存写字节分项明细。

        写入内容分为三组：
        1. 稠密 FFN：up/gate 输出、SiLU 输出、down 输出
        2. MoE 路由专家：与稠密 FFN 类似，token 数为 num_activated_tokens
        3. MoE 共享专家：与稠密 FFN 类似，乘以共享专家数 S
        """
        L, D, DI = self.num_hidden_layers, self.hidden_size, self.intermediate_size
        Lm, E, MI, S = (
            self.num_moe_layers,
            self.num_experts_per_tok,
            self.moe_intermediate_size,
            self.num_shared_experts,
        )
        T = ctx.total_num_tokens()

        Ld = L - Lm

        num_activated_tokens = T * E if E else 0

        if per_gpu:
            Ld //= self.pp_size
            Lm //= self.pp_size

            DI //= self.ffn_tp_size
            if MI is not None:
                MI //= self.ffn_tp_size
            if E:
                num_activated_tokens //= self.ffn_ep_size

        write_bytes = {}

        # Dense FFN layers
        if Ld:
            write_bytes["dense_up_gate_output"] = int(
                2 * T * DI * self.activation_byte_size * Ld
            )
            write_bytes["dense_silu_output"] = int(
                T * DI * self.activation_byte_size * Ld
            )
            write_bytes["dense_down_output"] = int(
                T * D * self.activation_byte_size * Ld
            )

        # MoE outputs
        if Lm:
            if E:
                write_bytes["routed_up_gate_output"] = int(
                    2 * num_activated_tokens * MI * self.activation_byte_size * Lm
                )
                write_bytes["routed_silu_output"] = int(
                    num_activated_tokens * MI * self.activation_byte_size * Lm
                )
                write_bytes["routed_down_output"] = int(
                    num_activated_tokens * D * self.activation_byte_size * Lm
                )
            if S:
                write_bytes["shared_up_gate_output"] = int(
                    2 * T * S * MI * self.activation_byte_size * Lm
                )
                write_bytes["shared_silu_output"] = int(
                    T * S * MI * self.activation_byte_size * Lm
                )
                write_bytes["shared_down_output"] = int(
                    T * S * D * self.activation_byte_size * Lm
                )

        return write_bytes


#### Unembed ####
# 反嵌入层指标


class UnembedMetrics(ComponentMetrics):
    """反嵌入（Unembedding）层的性能指标。

    反嵌入层将隐藏状态投影为词表大小的 logits 向量，
    用于预测下一个 token。

    这是一个简单的线性变换：logits = hidden_states @ W^T
    其中 W 是词嵌入矩阵的转置（通常与嵌入层共享权重）。

    字段来源：全部来自 BaseConfigParser。

    注意：只有需要计算 logits 的 token 才会经过此层。
    - 预填充：每个请求只需对最后一个 token 计算 logits
    - 解码：每个 token 都需要计算 logits
    """

    # From BaseConfigParser
    hidden_size: int = Field(..., gt=0)
    vocab_size: int = Field(..., gt=0)
    weight_byte_size: int = Field(..., gt=0)
    activation_byte_size: int = Field(..., gt=0)

    tp_size: int

    @classmethod
    def component_type(cls) -> str:
        return "unembed"

    @classmethod
    def get_parser(cls) -> ParserChain:
        return ParserChain(
            BaseConfigParser(),
        )

    def get_num_flops_breakdown(
        self, ctx: ExecutionContext, per_gpu: bool = True
    ) -> dict[str, int]:
        """Calculate flops breakdown for unembedding layer."""
        """计算反嵌入层的 FLOPs。

        FLOPs = 2 * T * D * V
        其中 T = 需要 logits 的 token 数，D = 隐藏维度，V = 词表大小。
        """
        D, V = self.hidden_size, self.vocab_size
        T = ctx.num_logits_tokens()

        if per_gpu:
            V //= self.tp_size

        return {
            "unembed": 2 * T * D * V,
        }

    def get_read_bytes_breakdown(
        self, ctx: ExecutionContext, per_gpu: bool = True
    ) -> dict[str, int]:
        """Calculate read memory traffic for unembedding layer."""
        """计算反嵌入层的内存读取。

        读取内容：
        - input: 输入激活值 (T * D * activation_byte_size)
        - weight: 权重矩阵 (D * V * weight_byte_size)
        """
        D, V = self.hidden_size, self.vocab_size
        T = ctx.num_logits_tokens()

        if per_gpu:
            V //= self.tp_size

        return {
            "input": T * D * self.activation_byte_size,
            "weight": D * V * self.weight_byte_size,
        }

    def get_write_bytes_breakdown(
        self, ctx: ExecutionContext, per_gpu: bool = True
    ) -> dict[str, int]:
        """Calculate write memory traffic for unembedding layer."""
        """计算反嵌入层的内存写入。

        写入内容：
        - output: 输出 logits (T * V * activation_byte_size)
        """
        V = self.vocab_size
        T = ctx.num_logits_tokens()

        if per_gpu:
            V //= self.tp_size

        return {
            "output": T * V * self.activation_byte_size,
        }


#### ModelMetrics ####
# 模型级指标聚合


class ModelMetrics:
    """模型级性能指标聚合器。

    解析 VllmConfig 并实例化所有组件指标（Attention、FFN、Unembed）。
    提供统一的接口来计算模型整体的 FLOPs 和内存带宽。

    使用流程：
    1. __init__()：遍历所有已注册的 ComponentMetrics 子类，
       尝试从 VllmConfig 实例化。如果某个组件的配置不适用（如非 MoE 模型
       不需要 MoE 相关字段），会捕获 InvalidComponent 异常并跳过。
    2. is_enabled()：检查是否至少有一个组件指标被成功实例化。
    3. get_step_perf_stats_per_gpu()：根据 SchedulerOutput 计算
       当前步骤的性能统计。

    Attributes:
        vllm_config: vLLM 配置对象。
        metrics: 成功实例化的组件指标列表。
    """

    def __init__(self, vllm_config: VllmConfig) -> None:
        """
        Parse vllm_config to instantiate metrics for each component.
        is_enabled() will return False if no component metrics could be instantiated.
        """

        self.vllm_config = vllm_config

        self.metrics: list[ComponentMetrics] = []
        for metric_cls in ComponentMetrics.registered_metrics():
            try:
                metric = metric_cls.from_vllm_config(vllm_config)
                self.metrics.append(metric)
                logger.info(
                    "Instantiated ComponentMetrics [%s] with (%s)",
                    metric.component_type(),
                    str(metric),
                )
            except InvalidComponent as e:
                logger.debug(
                    "Failed to instantiate %s from %s",
                    metric_cls.component_type(),
                    str(e),
                )

    def is_enabled(self) -> bool:
        """检查是否至少有一个组件指标被成功实例化。"""
        return len(self.metrics) > 0

    def get_num_flops(self, ctx: ExecutionContext, per_gpu: bool = True) -> int:
        """计算所有组件的总 FLOPs。"""
        return sum(metric.get_num_flops(ctx, per_gpu) for metric in self.metrics)

    def get_read_bytes(self, ctx: ExecutionContext, per_gpu: bool = True) -> int:
        """计算所有组件的总内存读取字节数。"""
        return sum(metric.get_read_bytes(ctx, per_gpu) for metric in self.metrics)

    def get_write_bytes(self, ctx: ExecutionContext, per_gpu: bool = True) -> int:
        """计算所有组件的总内存写字节数。"""
        return sum(metric.get_write_bytes(ctx, per_gpu) for metric in self.metrics)

    def get_num_flops_breakdown(
        self, ctx: ExecutionContext, per_gpu: bool = True
    ) -> dict[str, int]:
        """计算所有组件的 FLOPs 分项明细。

        返回的字典键带有组件前缀，如 "attn.qkv_proj"、"ffn.dense_ffn"。
        """
        total = {}
        for metric in self.metrics:
            breakdown = metric.get_num_flops_breakdown(ctx, per_gpu)
            component = metric.component_type()
            prefixed = {f"{component}.{key}": val for key, val in breakdown.items()}
            total.update(prefixed)
        return total

    def get_read_bytes_breakdown(
        self, ctx: ExecutionContext, per_gpu: bool = True
    ) -> dict[str, int]:
        """计算所有组件的内存读取字节分项明细。"""
        total = {}
        for metric in self.metrics:
            breakdown = metric.get_read_bytes_breakdown(ctx, per_gpu)
            component = metric.component_type()
            prefixed = {f"{component}.{key}": val for key, val in breakdown.items()}
            total.update(prefixed)
        return total

    def get_write_bytes_breakdown(
        self, ctx: ExecutionContext, per_gpu: bool = True
    ) -> dict[str, int]:
        """计算所有组件的内存写字节分项明细。"""
        total = {}
        for metric in self.metrics:
            breakdown = metric.get_write_bytes_breakdown(ctx, per_gpu)
            component = metric.component_type()
            prefixed = {f"{component}.{key}": val for key, val in breakdown.items()}
            total.update(prefixed)
        return total

    def get_step_perf_stats_per_gpu(
        self, scheduler_output: SchedulerOutput
    ) -> PerfStats:
        """根据调度器输出计算当前步骤的每 GPU 性能统计。

        处理流程：
        1. 构建 ExecutionContext：
           a. 遍历新请求（scheduled_new_reqs）：这些是预填充阶段
           b. 遍历缓存请求（scheduled_cached_reqs）：通常是解码阶段
              （token 数 > 1 时为分块预填充）
        2. 调用各组件方法计算 FLOPs 和内存带宽
        3. 汇总为 PerfStats

        Args:
            scheduler_output: 调度器的输出，包含当前步骤调度的请求信息。

        Returns:
            当前步骤的性能统计数据。
        """

        t0 = time.monotonic()

        # Build a single batch context
        ctx = ExecutionContext()

        # Process new requests (these are in prefill phase)
        for new_req in scheduler_output.scheduled_new_reqs:
            req_id = new_req.req_id
            num_tokens = scheduler_output.num_scheduled_tokens.get(req_id, 0)
            if num_tokens == 0:
                continue

            # For new requests, context_len = num_computed_tokens + num_tokens
            # num_computed_tokens represents previously computed tokens in the sequence
            context_len = new_req.num_computed_tokens + num_tokens
            ctx.add(num_tokens, context_len, is_prefill=True)

        # Process cached requests (continuing requests)
        cached_reqs = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(cached_reqs.req_ids):
            num_tokens = scheduler_output.num_scheduled_tokens.get(req_id, 0)
            if num_tokens == 0:
                continue

            # For cached requests, we have the current num_computed_tokens
            num_computed_tokens = cached_reqs.num_computed_tokens[i]
            context_len = num_computed_tokens + num_tokens

            # Cached requests are typically in decode phase (num_tokens == 1)
            # unless they're doing chunked prefill (num_tokens > 1)
            is_prefill = num_tokens > 1
            ctx.add(num_tokens, context_len, is_prefill)

        num_flops_breakdown = self.get_num_flops_breakdown(ctx, True)
        read_bytes_breakdown = self.get_read_bytes_breakdown(ctx, True)
        write_bytes_breakdown = self.get_write_bytes_breakdown(ctx, True)
        perf_stats = PerfStats(
            sum(num_flops_breakdown.values()),
            sum(read_bytes_breakdown.values()),
            sum(write_bytes_breakdown.values()),
        )

        if envs.VLLM_DEBUG_MFU_METRICS:
            perf_stats.debug_stats = DebugPerfStats(
                time.monotonic() - t0,
                ctx.num_prefill_requests,
                ctx.num_decode_requests,
                asdict(ctx),
                num_flops_breakdown,
                read_bytes_breakdown,
                write_bytes_breakdown,
            )

        return perf_stats


#### Logging ####
# 日志记录


class PerfMetricsDebugLogging:
    """MFU 调试日志记录器。

    当 VLLM_DEBUG_MFU_METRICS 环境变量启用时，累积记录详细的 MFU 计算信息，
    包括计算耗时、请求分类、以及各组件的 FLOPs 和内存带宽明细。

    数据在每次 log() 调用后重置，按日志输出间隔聚合。
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """重置所有累积的调试统计数据。"""
        self.total_calc_duration: float = 0.0
        self.total_num_prefill_requests: int = 0
        self.total_num_decode_requests: int = 0
        self.total_num_batches: int = 0
        self.total_context_breakdown: dict[str, int] = {}
        self.total_num_flops_per_gpu_breakdown: dict[str, int] = {}
        self.total_read_bytes_per_gpu_breakdown: dict[str, int] = {}
        self.total_write_bytes_per_gpu_breakdown: dict[str, int] = {}

    def observe(self, debug_stats: DebugPerfStats) -> None:
        """累积一次步骤的调试统计。

        Args:
            debug_stats: 单步的调试统计数据。
        """
        self.total_calc_duration += debug_stats.calc_duration
        self.total_num_prefill_requests += debug_stats.num_prefill_requests
        self.total_num_decode_requests += debug_stats.num_decode_requests
        self.total_num_batches += 1

        for dst, src in zip(
            [
                self.total_context_breakdown,
                self.total_num_flops_per_gpu_breakdown,
                self.total_read_bytes_per_gpu_breakdown,
                self.total_write_bytes_per_gpu_breakdown,
            ],
            [
                debug_stats.context_breakdown,
                debug_stats.num_flops_per_gpu_breakdown,
                debug_stats.num_read_bytes_per_gpu_breakdown,
                debug_stats.num_write_bytes_per_gpu_breakdown,
            ],
        ):
            assert isinstance(src, dict)
            for key, val in src.items():
                dst[key] = dst.get(key, 0) + val

    def log(self, log_fn, log_prefix: str, delta_time: float):
        """输出格式化的 MFU 调试日志。

        将累积的统计数据转换为人类可读的格式：
        - FLOPs 转换为 TF（TeraFLOPs）
        - 字节数转换为 GB
        - 计算 MFU 计算的开销占比

        Args:
            log_fn: 日志输出函数。
            log_prefix: 日志前缀。
            delta_time: 日志输出间隔时间（秒）。
        """
        # pretty print breakdowns
        total_num_flops_per_gpu_breakdown = {
            k: f"{v / 1e12:.1f}TF"
            for k, v in self.total_num_flops_per_gpu_breakdown.items()
        }
        total_read_bytes_per_gpu_breakdown = {
            k: f"{v / 1e9:.1f}GB"
            for k, v in self.total_read_bytes_per_gpu_breakdown.items()
        }
        total_write_bytes_per_gpu_breakdown = {
            k: f"{v / 1e9:.1f}GB"
            for k, v in self.total_write_bytes_per_gpu_breakdown.items()
        }

        logger.debug(
            "%sMFU details: %s",
            log_prefix,
            json.dumps(
                {
                    "prefill_reqs": self.total_num_prefill_requests,
                    "decode_reqs": self.total_num_decode_requests,
                    "num_batches": self.total_num_batches,
                    "context_breakdown": self.total_context_breakdown,
                    "flops_breakdown": total_num_flops_per_gpu_breakdown,
                    "num_read_bytes_breakdown": total_read_bytes_per_gpu_breakdown,
                    "num_write_bytes_breakdown": (total_write_bytes_per_gpu_breakdown),
                    "duration": f"{delta_time:.1f}s",
                    "mfu_calc_overhead": (
                        f"{self.total_calc_duration / delta_time:.1%}"
                    ),
                },
                indent=2,
            ),
        )


class PerfMetricsLogging:
    """MFU 性能指标日志记录器。

    累积每个步骤的性能统计，在日志输出时计算并打印平均 TFLOPS 和 GB/s。

    使用流程：
    1. __init__()：初始化，设置是否启用调试日志
    2. observe()：每个步骤调用，累积 FLOPs 和内存带宽
    3. log()：定期调用（如每 N 秒），计算并输出平均值，然后重置

    输出格式：
    MFU: {avg_tflops} TF/s/GPU {avg_gbps} GB/s/GPU

    Attributes:
        vllm_config: vLLM 配置对象。
        pp_size: 流水线并行大小。
        debug_logging: 调试日志记录器（仅在 VLLM_DEBUG_MFU_METRICS 启用时创建）。
        last_log_time: 上次日志输出的时间。
        total_num_flops_per_gpu: 累积的每 GPU FLOPs。
        total_read_bytes_per_gpu: 累积的每 GPU 内存读取字节数。
        total_write_bytes_per_gpu: 累积的每 GPU 内存写字节数。
    """

    def __init__(self, vllm_config: VllmConfig):
        self.vllm_config = vllm_config
        self.pp_size = vllm_config.parallel_config.pipeline_parallel_size

        self.debug_logging: PerfMetricsDebugLogging | None = None
        if envs.VLLM_DEBUG_MFU_METRICS:
            self.debug_logging = PerfMetricsDebugLogging()

        self.reset()

    def reset(self):
        """重置所有累积的统计数据和时间戳。"""
        self.last_log_time = time.monotonic()

        self.total_num_flops_per_gpu: int = 0
        self.total_read_bytes_per_gpu: int = 0
        self.total_write_bytes_per_gpu: int = 0

        if self.debug_logging:
            self.debug_logging.reset()

    def observe(self, perf_stats: PerfStats) -> None:
        """累积一步的性能统计。

        Args:
            perf_stats: 单步的性能统计数据。
        """
        self.total_num_flops_per_gpu += perf_stats.num_flops_per_gpu
        self.total_read_bytes_per_gpu += perf_stats.num_read_bytes_per_gpu
        self.total_write_bytes_per_gpu += perf_stats.num_write_bytes_per_gpu

        if self.debug_logging:
            assert perf_stats.debug_stats is not None
            self.debug_logging.observe(perf_stats.debug_stats)

    def log(self, log_fn=logger.info, log_prefix: str = "") -> None:
        """计算并输出平均 MFU 指标，然后重置统计数据。

        计算公式：
        - avg_tflops_per_gpu = total_flops / delta_time / 1e12
        - avg_gbps_per_gpu = (total_read + total_write) / delta_time / 1e9

        如果没有任何累积数据（全部为 0），则跳过输出。
        """
        if not (
            self.total_num_flops_per_gpu
            or self.total_read_bytes_per_gpu
            or self.total_write_bytes_per_gpu
        ):
            return

        now = time.monotonic()
        delta_time = now - self.last_log_time

        if delta_time <= 0.0:
            avg_tflops_per_gpu = 0.0
            avg_gbps_per_gpu = 0.0
        else:
            avg_tflops_per_gpu = self.total_num_flops_per_gpu / delta_time / 1e12
            avg_gbps_per_gpu = (
                (self.total_read_bytes_per_gpu + self.total_write_bytes_per_gpu)
                / delta_time
                / 1e9
            )

        log_fn(
            "%sMFU: %.1f TF/s/GPU %.1f GB/s/GPU",
            log_prefix,
            avg_tflops_per_gpu,
            avg_gbps_per_gpu,
        )

        if self.debug_logging:
            self.debug_logging.log(log_fn, log_prefix, delta_time)

        self.reset()


#### Prometheus Integration ####
# Prometheus 指标集成


class PerfMetricsProm:
    """Record performance metrics in Prometheus.

    Average TFLOPS (tera floating-point operations per second) can be
    calculated using a PromQL query:

      rate(vllm:estimated_flops_per_gpu_total[1m]) / 1e12

    Average memory bandwidth in GB/s can be calculated using:

      (rate(vllm:estimated_read_bytes_per_gpu_total[1m]) +
       rate(vllm:estimated_write_bytes_per_gpu_total[1m])) / 1e9
    """

    """性能指标的 Prometheus 上报器。

    将 MFU 相关的性能指标作为 Prometheus Counter 上报。
    这些是累积计数器，可以在 Prometheus/Grafana 中使用 rate() 函数
    计算平均值。

    上报的指标：
    1. vllm:estimated_flops_per_gpu_total：每 GPU 的估算总 FLOPs
       用于计算平均 TFLOPS：
       rate(vllm:estimated_flops_per_gpu_total[1m]) / 1e12

    2. vllm:estimated_read_bytes_per_gpu_total：每 GPU 的估算内存读取总字节数
    3. vllm:estimated_write_bytes_per_gpu_total：每 GPU 的估算内存写入总字节数
       用于计算平均内存带宽：
       (rate(read_bytes[1m]) + rate(write_bytes[1m])) / 1e9

    Attributes:
        counter_flops: 每引擎的 FLOPs 计数器。
        counter_read_bytes: 每引擎的内存读取计数器。
        counter_write_bytes: 每引擎的内存写入计数器。
    """

    _counter_cls = prometheus_client.Counter

    def __init__(
        self,
        vllm_config: VllmConfig,
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ):
        counter_flops = self._counter_cls(
            name="vllm:estimated_flops_per_gpu_total",
            documentation=(
                "Estimated number of floating point operations per GPU "
                "(for Model Flops Utilization calculations)."
            ),
            labelnames=labelnames,
        )
        self.counter_flops = create_metric_per_engine(
            counter_flops, per_engine_labelvalues
        )

        counter_read_bytes = self._counter_cls(
            name="vllm:estimated_read_bytes_per_gpu_total",
            documentation=(
                "Estimated number of bytes read from memory per GPU "
                "(for Model Flops Utilization calculations)."
            ),
            labelnames=labelnames,
        )
        self.counter_read_bytes = create_metric_per_engine(
            counter_read_bytes, per_engine_labelvalues
        )

        counter_write_bytes = self._counter_cls(
            name="vllm:estimated_write_bytes_per_gpu_total",
            documentation=(
                "Estimated number of bytes written to memory per GPU "
                "(for Model Flops Utilization calculations)."
            ),
            labelnames=labelnames,
        )
        self.counter_write_bytes = create_metric_per_engine(
            counter_write_bytes, per_engine_labelvalues
        )

    def observe(self, perf_stats: PerfStats, engine_idx: int = 0):
        """上报一步的性能统计到 Prometheus。

        Args:
            perf_stats: 单步的性能统计数据。
            engine_idx: 引擎索引（用于多引擎场景）。
        """
        if not (
            perf_stats.num_flops_per_gpu
            or perf_stats.num_read_bytes_per_gpu
            or perf_stats.num_write_bytes_per_gpu
        ):
            return
        self.counter_flops[engine_idx].inc(perf_stats.num_flops_per_gpu)
        self.counter_read_bytes[engine_idx].inc(perf_stats.num_read_bytes_per_gpu)
        self.counter_write_bytes[engine_idx].inc(perf_stats.num_write_bytes_per_gpu)


## util functions
# 工具函数


def get_required(obj: object, attr: str):
    """从对象中获取指定属性，如果属性不存在则抛出 InvalidComponent 异常。

    Args:
        obj: 要查询的对象。
        attr: 属性名称。

    Returns:
        属性值。

    Raises:
        InvalidComponent: 如果对象没有指定属性。
    """
    if not hasattr(obj, attr):
        raise InvalidComponent(f"Missing required attr {attr} in config")
    return getattr(obj, attr)


def getattr_from_list(obj: object, attrs: list[str], default: object = None):
    """尝试从对象中获取第一个存在的属性。

    用于处理不同模型配置中属性名称不一致的情况。
    例如：num_experts_per_tok vs moe_topk

    Args:
        obj: 要查询的对象。
        attrs: 候选属性名列表（按优先级排列）。
        default: 如果所有属性都不存在时的默认值。

    Returns:
        第一个存在的属性值，或默认值。
    """
    for attr in attrs:
        if hasattr(obj, attr):
            return getattr(obj, attr)
    return default
