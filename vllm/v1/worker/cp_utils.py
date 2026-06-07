# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
cp_utils.py - 上下文并行 (Context Parallelism) 兼容性检查与工具函数模块。

【模块功能概述】
本模块在 vLLM 引擎启动阶段（Worker 初始化时）被调用，核心职责是：
  验证用户配置的上下文并行策略是否与当前使用的 attention 后端兼容。

如果验证失败，引擎会直接抛出 AssertionError，避免在运行时出现
难以排查的分布式通信或数值错误。

【什么是上下文并行】
上下文并行是一种将单个长序列的 KV cache 切分到多个 GPU 上的技术，
用于解决单卡显存无法容纳超长上下文（如 128K+ tokens）的问题。
与 TP（Tensor Parallelism）切分模型权重不同，CP 切分的是序列维度。

上下文并行包含两种可组合的策略：

1. PCP（Prefill Context Parallelism，预填充上下文并行）
   - 作用阶段：prefill（首次计算 prompt 的全部 token）
   - 原理：将超长 prompt 的 token 序列均分到 PCP 组内的多个 GPU，
     每个 GPU 只负责计算自己那部分 token 的 Q/K/V 和 attention，
     然后通过集合通信合并结果。
   - 典型场景：处理 128K+ 超长上下文输入。

2. DCP（Decode Context Parallelism，解码上下文并行）
   - 作用阶段：decode（逐 token 生成阶段）
   - 原理：不增加总 world_size，而是复用 TP 组内的 GPU，
     将 KV cache 按序列维度切分存储，decode 时通过 AllGather 等
     通信操作合并 KV 进行 attention 计算。
   - 典型场景：长序列 decode 阶段的 KV cache 显存优化。

【组合关系】
  总 CP world size = PCP world size × DCP world size
  例如 PCP=2, DCP=4 → 总 CP = 8，即 KV cache 按 8 路切分。

【KV cache 交错存储】
cp_kv_cache_interleave_size 控制 token 在 CP rank 间的分布方式：
  - interleave_size=1：token 级交错，token i 存储在 rank i % total_cp_world_size
  - interleave_size=block_size：block 级交错，token 优先填充到前面的 rank

【与 v1 引擎链路的关系】
本模块的函数在 GPU Worker 初始化阶段被调用（见 gpu_worker.py 的
init_device 流程），检查通过后，后续 Scheduler 和 Model Runner
才能正常使用 CP 策略进行调度和计算。
"""

from typing import TYPE_CHECKING, Any, cast

from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.distributed import get_dcp_group, get_pcp_group

if TYPE_CHECKING:
    from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
else:
    AttentionLayerBase = object


def check_attention_cp_compatibility(vllm_config: VllmConfig) -> None:
    """检查注意力层（Attention Layer）与上下文并行（CP）配置的兼容性。

    【调用时机】
    在 GPU Worker 初始化设备阶段调用（早于模型加载和第一次 forward）。
    这是一个"fail-fast"机制——如果 attention 后端不支持所配置的 CP 策略，
    引擎在启动时就会报错，而不是在运行时出现静默的数值错误或通信死锁。

    【为什么需要这个检查】
    CP 策略要求 attention 后端具备特殊能力：
    - DCP 需要 attention 在 decode 阶段返回 LSE（Log-Sum-Exp），
      因为 DCP 组内每个 GPU 只计算部分 KV 的 attention 输出，
      合并时需要 LSE 来正确加权融合多个部分结果。
    - PCP 需要 attention 后端原生支持 PCP 的分块计算和通信模式。
    - MTP（投机解码的多 token 预测）与非平凡交错大小组合时，
      需要 attention 后端能正确处理复杂的 token 分布模式。

    【检查流程】
    1. 从配置中读取 PCP size、DCP size、KV cache 交错大小
    2. 仅当总 CP > 1（即启用了上下文并行）时才执行检查
    3. 通过 get_layers_from_vllm_config 获取模型中所有 Attention 层
    4. 对每个有 impl（attention 实现）的层，按以下优先级检查：
       (a) MTP + 非平凡交错：需要 supports_mtp_with_cp_non_trivial_interleave_size
       (b) DCP > 1：需要 need_to_return_lse_for_decode
       (c) PCP > 1：需要 supports_pcp

    【attention 后端能力标记】
    上述三个能力标记（supports_pcp 等）定义在 attention 后端类中
    （见 vllm/v1/attention/backend.py），不同的 attention 后端
    （FlashAttention、FlashInfer、Triton 等）需要显式声明自己支持哪些 CP 特性。

    Args:
        vllm_config: vLLM 全局配置对象，包含 parallel_config 中的 CP 相关参数，
                     以及所有已注册的 attention 层信息。

    Raises:
        AssertionError: 当任何 attention 层不满足所配置的 CP 兼容性要求时抛出，
                        错误消息会指明哪个层的哪个后端不支持哪种 CP 特性，
                        并提示用户更换 attention 后端或禁用 CP。
    """
    # 从 parallel_config 中读取三种 CP 相关配置参数
    # pcp_size: 预填充上下文并行度，>1 表示启用 PCP
    pcp_size = vllm_config.parallel_config.prefill_context_parallel_size
    # dcp_size: 解码上下文并行度，>1 表示启用 DCP
    dcp_size = vllm_config.parallel_config.decode_context_parallel_size
    # interleave_size: KV cache 在 CP rank 间的交错粒度，
    #   1 = token 级交错，block_size = block 级交错
    interleave_size = vllm_config.parallel_config.cp_kv_cache_interleave_size

    # 仅当总 CP world size > 1 时（即确实启用了某种 CP 策略）才需要检查
    if pcp_size * dcp_size > 1:
        # 从 vllm_config 中获取模型中所有 AttentionLayerBase 类型的层
        # 这些层在模型构建时被注册到 vllm_config 的层注册表中
        layer_type = cast(type[Any], AttentionLayerBase)
        layers = get_layers_from_vllm_config(vllm_config, layer_type)

        # 逐层检查 attention 实现是否满足 CP 要求
        for layer in layers.values():
            # layer.impl 是该层实际使用的 attention 后端实现对象
            # （如 FlashAttentionImpl、FlashInferImpl 等）
            layer_impl = getattr(layer, "impl", None)
            if layer_impl is None:
                # 跳过没有绑定 attention 实现的层（如某些非 attention 层）
                continue

            # --- 检查 1：MTP（多 token 预测）+ 非平凡交错大小 ---
            # 当同时启用投机解码（speculative_config != None）和
            # cp_kv_cache_interleave_size > 1 时，token 在 CP rank 间的
            # 分布模式变得复杂，需要 attention 后端显式支持。
            # 不支持的后端可能导致 MTP 的 draft token 无法正确对齐 KV cache。
            if vllm_config.speculative_config is not None and interleave_size > 1:
                assert layer_impl.supports_mtp_with_cp_non_trivial_interleave_size, (
                    "MTP with cp_kv_cache_interleave_size > 1 is not "
                    f"supported in {layer_impl.__class__.__name__}."
                )

            # --- 检查 2：DCP 要求 attention 后端在 decode 阶段返回 LSE ---
            # LSE（Log-Sum-Exp）是 softmax 的归一化因子。
            # DCP 将 KV cache 切分到多个 GPU，每个 GPU 计算部分 KV 的 attention，
            # 最终需要通过 LSE 来正确合并多个部分结果：
            #   output = Σ(LSE_weighted_partial_output) / Σ(LSE_weights)
            # 如果后端不返回 LSE，DCP 组内无法正确合并结果。
            if dcp_size > 1:
                assert layer_impl.need_to_return_lse_for_decode, (
                    "Decode Context Parallelism (DCP) requires attention "
                    "implementations to return the softmax LSE during decode, "
                    f"but {layer_impl.__class__.__name__} does not. "
                    "Try a different backend by setting "
                    "--attention-backend or disable DCP."
                )

            # --- 检查 3：PCP 要求 attention 后端原生支持 PCP ---
            # PCP 在 prefill 阶段将长序列切分到多个 GPU 并行计算，
            # 需要 attention 后端支持分块 Q/K/V 计算、集合通信合并等
            # 特殊逻辑，不是所有后端都实现了这些功能。
            if pcp_size > 1:
                assert layer_impl.supports_pcp, (
                    "PCP requires attention impls' support, "
                    f"but the impl {layer_impl.__class__.__name__} "
                    "does not support PCP."
                )


def get_total_cp_world_size():
    """获取总上下文并行（CP）的 world size。

    【用途】
    该函数在需要知道"当前系统中 CP 总共涉及多少个 GPU"时被调用。
    例如，KV cache manager 需要知道 CP world size 来决定每个请求的
    KV cache 应该被切分成多少份，以及每份的大小。

    【计算公式】
    总 CP world size = PCP world size × DCP world size

    PCP 和 DCP 是两种独立的上下文并行策略，可以同时启用：
    - PCP 负责 prefill 阶段的序列切分
    - DCP 负责 decode 阶段的 KV cache 切分
    - 两者组合时，总切分数 = pcp_size × dcp_size

    【容错设计】
    使用 try-except 捕获 AssertionError，因为 PCP/DCP 的进程组
    （process group）可能尚未初始化，典型场景包括：
    - 单元测试中未启动分布式环境
    - 某些只使用部分并行策略的配置
    在这些情况下，未初始化的组默认 world_size 为 1（即不切分）。

    【与分布式进程组的关系】
    get_pcp_group() / get_dcp_group() 返回的是 GroupCoordinator 对象，
    它们在 parallel_state.py 中通过 init_model_parallel_group() 初始化，
    底层基于 NCCL/Gloo 等通信后端。world_size 即该通信组中的 GPU 数量。

    Returns:
        int: 总上下文并行 world size。
             返回 1 表示未启用任何 CP 策略。

    示例：
        - PCP=2, DCP=4 → 返回 8（KV cache 按 8 路切分）
        - PCP=1, DCP=1 → 返回 1（未启用 CP，单 GPU 存储完整 KV cache）
    """
    try:
        # 获取 PCP（预填充上下文并行）通信组的 world size
        pcp_world_size = get_pcp_group().world_size
    except AssertionError:
        # PCP 可能在测试环境中未初始化，安全回退为 1
        # PCP might not be initialized in testing
        pcp_world_size = 1
    try:
        # 获取 DCP（解码上下文并行）通信组的 world size
        dcp_world_size = get_dcp_group().world_size
    except AssertionError:
        # DCP 可能在测试环境中未初始化，安全回退为 1
        # DCP might not be initialized in testing
        dcp_world_size = 1
    return dcp_world_size * pcp_world_size
