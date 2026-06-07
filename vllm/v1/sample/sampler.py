# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""从模型输出中采样下一个token的采样层。

本模块实现了一个完整的采样流水线，包括:
1. logprobs计算
2. logits处理（坏词排除、惩罚、温度缩放等）
3. 贪心/随机采样
4. Top-K/Top-P截断采样

该采样器是vLLM v1引擎的核心组件之一，负责将模型输出的logits转换为具体的token ID。
"""

import torch
import torch.nn as nn

from vllm.config.model import LogprobsMode
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.v1.outputs import LogprobsTensors, SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.ops.bad_words import apply_bad_words
from vllm.v1.sample.ops.logprobs import batched_count_greater_than
from vllm.v1.sample.ops.penalties import apply_all_penalties
from vllm.v1.sample.ops.topk_topp_sampler import TopKTopPSampler

# 采样epsilon阈值：用于区分贪心采样和随机采样
# 当temperature < _SAMPLING_EPS时，视为贪心采样（temperature=0）
_SAMPLING_EPS = 1e-5


class Sampler(nn.Module):
    """主采样器：从模型输出的logits中采样下一个token。

    该模块实现了一个完整的采样流水线，按照以下顺序执行：

    1. 如果请求了logprobs：
        a) 如果 `logprobs_mode` 是 `raw_logprobs`，将logits转换为logprobs作为最终返回值。
        b) 如果 `logprobs_mode` 是 `raw_logits`，克隆logits作为最终返回值。
    2. 将logits转换为float32精度。
    3. 应用允许的token ID白名单（禁止不在白名单中的token）。
    4. 应用坏词排除。
    5. 应用非argmax不变的logits处理器（会影响贪心采样结果的处理器）：
        a) 最少token数处理器
        b) Logit偏置处理器
    6. 应用惩罚：
        a) 重复惩罚
        b) 频率惩罚
        c) 存在惩罚
    7. 采样下一个token。`sample` 方法执行以下步骤：
        a) 如果不是全随机模式，执行贪心采样。如果是全贪心模式，返回贪心采样的token和logprobs。
        b) 应用温度缩放。
        c) 应用argmax不变的logits处理器（默认是min_p处理器）。
        d) 应用top_k和/或top_p截断。
        e) 从概率分布中采样下一个token。
        f) 如果是全随机模式或temperature >= epsilon(1e-5)，返回随机采样的token和logprobs。
           否则，返回贪心采样的token和logprobs。
    8. 收集top `max_num_logprobs`个logprobs和采样token的logprob（如果请求了）。
       注意：如果采样token在top `max_num_logprobs`中，最终输出可能包含
       `max_num_logprobs + 1`或`max_num_logprobs`个logprobs。
    9. 返回最终的 `SamplerOutput`。

    工作流程总结:
        logits -> logprobs计算(可选) -> float32转换 -> 白名单过滤 -> 坏词排除
        -> 非argmax不变处理器 -> 惩罚应用 -> 温度缩放 -> argmax不变处理器
        -> Top-K/Top-P -> 采样 -> logprobs收集 -> 输出

    属性:
        topk_topp_sampler: Top-K/Top-P采样器
        pin_memory: 是否使用pin memory加速CPU-GPU数据传输
        logprobs_mode: logprobs计算模式
    """

    def __init__(self, logprobs_mode: LogprobsMode = "raw_logprobs"):
        """
        初始化采样器。

        参数:
            logprobs_mode: logprobs的计算模式，可选值包括:
                - "raw_logprobs": 使用原始logits计算的logprobs
                - "raw_logits": 直接使用原始logits作为logprobs
                - "processed_logprobs": 使用处理后的logits计算的logprobs
                - "processed_logits": 直接使用处理后的logits作为logprobs
        """
        super().__init__()
        # 初始化Top-K/Top-P采样器
        self.topk_topp_sampler = TopKTopPSampler(logprobs_mode)
        # 检测是否可以使用pin memory
        self.pin_memory = is_pin_memory_available()
        self.logprobs_mode = logprobs_mode

    def forward(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        predict_bonus_token: bool = False,
        logprobs_mode_override: LogprobsMode | None = None,
    ) -> SamplerOutput:
        """执行采样前向传播。

        从模型输出的logits中采样下一个token，并可选地计算logprobs。

        参数:
            logits: 模型输出的logits张量 [batch_size, vocab_size]
            sampling_metadata: 采样元数据，包含温度、top_k、top_p等参数
            predict_bonus_token: 是否预测bonus token（用于投机解码）
            logprobs_mode_override: 覆盖默认的logprobs计算模式

        返回:
            SamplerOutput: 包含采样的token IDs和logprobs的输出
        """
        # 确定logprobs计算模式（优先使用override）
        logprobs_mode = logprobs_mode_override or self.logprobs_mode

        # NOTE(woosuk): 使用原始logits（在任何惩罚或温度缩放之前）来计算top-k logprobs。
        # 这与V0采样器不同，V0使用的是用于采样的logits（经过惩罚和温度缩放后）。
        num_logprobs = sampling_metadata.max_num_logprobs
        raw_logprobs: torch.Tensor | None = None

        # 步骤1: 计算logprobs（如果请求了）
        if num_logprobs is not None or sampling_metadata.logprob_token_ids:
            if logprobs_mode == "raw_logprobs":
                # 使用log_softmax计算原始logprobs
                raw_logprobs = self.compute_logprobs(logits)
            elif logprobs_mode == "raw_logits":
                # 直接使用logits作为logprobs
                if logits.dtype == torch.float32:
                    raw_logprobs = logits.clone()
                else:
                    raw_logprobs = logits.to(torch.float32)

        # 步骤2: 使用float32精度处理logits
        logits = logits.to(torch.float32)

        # 步骤3-6: 应用所有logits处理器（白名单、坏词、惩罚等）
        logits = self.apply_logits_processors(
            logits, sampling_metadata, predict_bonus_token
        )

        # 步骤7: 采样下一个token
        sampled, processed_logprobs = self.sample(logits, sampling_metadata)
        if processed_logprobs is not None:
            raw_logprobs = processed_logprobs

        # 将采样的token ID转换为int64类型，确保与后续操作的兼容性。
        # 这个转换是必要的，因为FlashInfer采样操作返回int32
        # （而PyTorch的argmax和topk返回int64）。
        sampled = sampled.long()

        # 处理特定token ID的logprobs请求（比全词表更高效）
        # 这用于generative_scoring API，获取特定token的logprobs
        logprob_token_ids_tensors = None
        if sampling_metadata.logprob_token_ids:
            assert raw_logprobs is not None
            logprob_token_ids_tensors = self.gather_specific_token_logprobs(
                raw_logprobs, sampling_metadata.logprob_token_ids, sampled
            )

        # 步骤8: 收集logprobs
        if num_logprobs is None:
            # 没有请求logprobs，使用特定token的logprobs（如果有）
            logprobs_tensors = logprob_token_ids_tensors
        elif num_logprobs == -1:
            # 返回完整的、未排序、未排名的logprobs
            logprobs_tensors = LogprobsTensors(
                torch.empty(0), raw_logprobs, torch.empty(0)
            )
        else:
            # 收集top-k logprobs和采样token的logprob
            logprobs_tensors = self.gather_logprobs(
                raw_logprobs, num_logprobs, token_ids=sampled
            )

        # 如果同时有num_logprobs和logprob_token_ids，优先使用logprob_token_ids（更具体）
        if logprob_token_ids_tensors is not None and num_logprobs is not None:
            logprobs_tensors = logprob_token_ids_tensors

        # 使用int32减少张量大小
        sampled = sampled.to(torch.int32)

        # 步骤9: 构建并返回最终输出
        # 这些是GPU张量
        sampler_output = SamplerOutput(
            # 采样的token被扩展为2D张量，形状为 [num_requests, 1]
            # 每行表示每个请求生成的一个token
            sampled_token_ids=sampled.unsqueeze(-1),
            logprobs_tensors=logprobs_tensors,
        )
        return sampler_output

    def gather_specific_token_logprobs(
        self,
        logprobs: torch.Tensor,
        logprob_token_ids: dict[int, list[int]],
        sampled: torch.Tensor,
    ) -> LogprobsTensors | None:
        """收集特定token ID的logprobs。

        用于generative_scoring API，返回指定token ID集合的logprobs，而不是top-k。
        处理不同请求之间异构的token ID列表，通过填充较短的列表到最大长度。

        参数:
            logprobs: [batch_size, vocab_size] 张量，包含（原始）logprobs
            logprob_token_ids: 请求索引到token ID列表的映射
            sampled: [batch_size] 张量，包含采样的token ID

        返回:
            LogprobsTensors: 包含指定token的logprobs，如果没有请求有logprob_token_ids则返回None
        """
        if not logprob_token_ids:
            return None

        batch_size = logprobs.shape[0]
        device = logprobs.device

        # 找出所有请求中token ID的最大数量
        max_num_tokens = max(len(tids) for tids in logprob_token_ids.values())
        pin = self.pin_memory

        # 在pinned CPU上构建填充后的token_ids和valid_mask矩阵，然后非阻塞上传到GPU
        token_ids_cpu = torch.zeros(
            batch_size, max_num_tokens + 1, dtype=torch.int64, pin_memory=pin
        )
        # 创建有效位置掩码（True = 有效，False = 填充）
        valid_mask_cpu = torch.zeros(
            batch_size, max_num_tokens + 1, dtype=torch.bool, pin_memory=pin
        )
        valid_mask_cpu[:, 0] = True  # 采样token始终有效
        for req_idx, token_ids in logprob_token_ids.items():
            num_tokens = len(token_ids)
            token_ids_cpu[req_idx, 1 : num_tokens + 1] = torch.as_tensor(
                token_ids, dtype=torch.int64
            )
            valid_mask_cpu[req_idx, 1 : num_tokens + 1] = True

        # 非阻塞传输到GPU
        token_ids_tensor = token_ids_cpu.to(device, non_blocking=True)
        valid_mask = valid_mask_cpu.to(device, non_blocking=True)
        # 采样token在第0列 - 在GPU上从采样的GPU张量填充，避免D2H传输后重新上传
        token_ids_tensor[:, 0] = sampled

        # 在请求的token ID处收集logprobs
        gathered_logprobs = logprobs.gather(-1, token_ids_tensor)

        # 用-inf掩码无效（填充）位置
        gathered_logprobs = gathered_logprobs.masked_fill(~valid_mask, float("-inf"))

        # 计算采样token的排名。log_softmax相对于原始logits是单调的，因此从logprobs计算的排名是等价的。
        sampled_logprobs = logprobs.gather(-1, sampled.unsqueeze(-1))
        # 避免在batch维度上进行0/1特化重编译。
        # 参见gather_logprobs了解上下文。
        torch._dynamo.decorators.mark_unbacked(logprobs, 0)
        torch._dynamo.decorators.mark_unbacked(sampled_logprobs, 0)
        token_ranks = batched_count_greater_than(logprobs, sampled_logprobs)

        return LogprobsTensors(
            logprob_token_ids=token_ids_tensor.to(torch.int32),
            logprobs=gathered_logprobs,
            selected_token_ranks=token_ranks,
        )

    @staticmethod
    def apply_temperature(
        logits: torch.Tensor,
        temp: torch.Tensor,
        all_random: bool,
    ) -> torch.Tensor:
        """应用温度缩放到logits。

        温度缩放公式: scaled_logits = logits / temperature
        - temperature < 1: 使分布更尖锐（更确定性）
        - temperature > 1: 使分布更平坦（更随机）
        - temperature = 0: 等价于贪心采样

        参数:
            logits: 输入logits张量 [batch_size, vocab_size]
            temp: 温度张量 [batch_size]
            all_random: 是否所有请求都是随机采样

        返回:
            温度缩放后的logits张量（原地修改）
        """
        # 使用原地除法避免创建新张量。
        # 如果有贪心请求（temp=0），将temp替换为1.0避免除零错误。
        if not all_random:
            temp = torch.where(temp < _SAMPLING_EPS, 1.0, temp)
        return logits.div_(temp.unsqueeze(dim=1))

    @staticmethod
    def greedy_sample(logits: torch.Tensor) -> torch.Tensor:
        """执行贪心采样：选择logits最大的token。

        参数:
            logits: 输入logits张量 [batch_size, vocab_size]

        返回:
            采样的token IDs [batch_size]
        """
        return logits.argmax(dim=-1).view(-1)

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        logprobs_mode_override: LogprobsMode | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """基于采样元数据从logits中采样token。

        该方法中调用的各种logits处理函数可能会原地更新logits张量。

        采样流程:
        1. 如果不是全随机模式，先执行贪心采样
        2. 如果是全贪心模式，直接返回贪心采样结果
        3. 应用温度缩放
        4. 应用argmax不变的logits处理器（如min_p）
        5. 应用Top-K/Top-P截断
        6. 执行随机采样
        7. 根据temperature选择贪心或随机采样结果

        参数:
            logits: 输入logits张量 [batch_size, vocab_size]
            sampling_metadata: 采样元数据
            logprobs_mode_override: 覆盖默认的logprobs计算模式

        返回:
            (采样的token IDs, 处理后的logprobs或None)
        """
        logprobs_mode = logprobs_mode_override or self.logprobs_mode

        # 验证: all_greedy和all_random不能同时为True
        assert not (sampling_metadata.all_greedy and sampling_metadata.all_random)

        # 步骤1: 如果不是全随机模式，先执行贪心采样
        if sampling_metadata.all_random:
            greedy_sampled = None
        else:
            greedy_sampled = self.greedy_sample(logits)

            # 步骤2: 如果是全贪心模式，直接返回
            if sampling_metadata.all_greedy:
                processed_logprobs = None
                if (
                    sampling_metadata.max_num_logprobs is not None
                    or sampling_metadata.logprob_token_ids
                ):
                    if logprobs_mode == "processed_logits":
                        processed_logprobs = logits
                    elif logprobs_mode == "processed_logprobs":
                        processed_logprobs = self.compute_logprobs(logits)
                return greedy_sampled, processed_logprobs

        # 步骤3: 应用温度缩放
        assert sampling_metadata.temperature is not None
        logits = self.apply_temperature(
            logits, sampling_metadata.temperature, sampling_metadata.all_random
        )

        # 步骤4: 应用argmax不变的logits处理器（仅影响随机采样）
        for processor in sampling_metadata.logitsprocs.argmax_invariant:
            logits = processor.apply(logits)

        # 步骤5-6: 应用Top-K和/或Top-P，然后执行随机采样
        random_sampled, processed_logprobs = self.topk_topp_sampler(
            logits,
            sampling_metadata.generators,
            sampling_metadata.top_k,
            sampling_metadata.top_p,
        )

        # 步骤7: 根据temperature选择采样结果
        if greedy_sampled is None:
            return random_sampled, processed_logprobs

        # 对于temperature < epsilon的请求使用贪心采样，否则使用随机采样
        sampled = torch.where(
            sampling_metadata.temperature < _SAMPLING_EPS,
            greedy_sampled,
            random_sampled,
            out=greedy_sampled,  # 复用张量以节省内存
        )
        return sampled, processed_logprobs

    @staticmethod
    def compute_logprobs(logits: torch.Tensor) -> torch.Tensor:
        """将logits转换为log概率。

        使用log_softmax函数: logprobs = log(softmax(logits))

        参数:
            logits: 输入logits张量 [..., vocab_size]

        返回:
            log概率张量 [..., vocab_size]
        """
        return logits.log_softmax(dim=-1, dtype=torch.float32)

    @staticmethod
    def gather_logprobs(
        logprobs: torch.Tensor,
        num_logprobs: int,
        token_ids: torch.Tensor,
    ) -> LogprobsTensors:
        """收集top-k logprobs和采样/提示token的logprob。

        参数:
            logprobs: (num tokens) x (vocab) 张量，包含log概率
            num_logprobs: 每个token保留的最大logprobs数量
            token_ids: 提示token（如果是提示logprobs）或采样token（如果是采样logprobs）；
                       1D token ID张量，包含 (num tokens) 个元素，必须是int64类型

        返回:
            LogprobsTensors: 包含以下内容:
                - Top-k int索引张量, (num tokens) x (num_logprobs + 1)
                - Top-k float logprobs张量, (num tokens) x (num_logprobs + 1)
                - 采样token排名张量, (num tokens)
        """
        assert token_ids.dtype == torch.int64

        # 找出Top-K值
        topk_logprobs, topk_indices = torch.topk(logprobs, num_logprobs, dim=-1)

        # 获取提示token或采样token的logprob
        token_ids = token_ids.unsqueeze(-1)
        token_logprobs = logprobs.gather(-1, token_ids)

        # 计算实际token的排名。
        # 避免在batch维度上进行0/1特化重编译。
        # mark_unbacked使size完全符号化，这样dynamo不会在batch_size从1变为>=2时进行特化。
        torch._dynamo.decorators.mark_unbacked(logprobs, 0)
        torch._dynamo.decorators.mark_unbacked(token_logprobs, 0)
        token_ranks = batched_count_greater_than(logprobs, token_logprobs)

        # 将采样token的logprob与top-k连接在一起
        indices = torch.cat((token_ids, topk_indices), dim=1)
        logprobs = torch.cat((token_logprobs, topk_logprobs), dim=1)

        # 使用int32减少张量大小
        indices = indices.to(torch.int32)

        return LogprobsTensors(indices, logprobs, token_ranks)

    @staticmethod
    def _combine_outputs_with_spec_tokens(
        output_token_ids: list[list[int]],
        spec_token_ids: list[list[int]] | None = None,
    ) -> list[list[int]]:
        """将基础输出token与投机解码token组合。

        在投机解码启用时，将已生成的输出token与投机token序列合并，
        以便正确计算惩罚和坏词排除。

        参数:
            output_token_ids: 每个请求的基础输出token ID列表
            spec_token_ids: 每个请求的投机token ID列表

        返回:
            组合后的输出token ID列表
        """
        if spec_token_ids is None:
            return output_token_ids

        return [
            [*out, *spec] if spec else out
            for out, spec in zip(output_token_ids, spec_token_ids)
        ]

    def apply_logits_processors(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        predict_bonus_token: bool,
    ) -> torch.Tensor:
        """应用所有logits处理器。

        按照以下顺序应用处理器:
        1. 允许的token ID白名单（禁止不在白名单中的token）
        2. 坏词排除
        3. 非argmax不变的logits处理器
        4. 惩罚（频率惩罚、存在惩罚、重复惩罚）
        5. 思考预算状态处理（如果启用）

        参数:
            logits: 输入logits张量 [batch_size, vocab_size]
            sampling_metadata: 采样元数据
            predict_bonus_token: 是否预测bonus token

        返回:
            处理后的logits张量（可能原地修改）
        """
        # 检查是否有坏词或惩罚需要处理
        bad_words_token_ids = sampling_metadata.bad_words_token_ids
        any_penalties_or_bad_words = (
            bool(bad_words_token_ids) or not sampling_metadata.no_penalties
        )

        # 检查是否需要思考预算处理
        holder = sampling_metadata.thinking_budget_state_holder
        needs_thinking_combine = holder is not None and holder.has_tracked_requests()

        output_token_ids = sampling_metadata.output_token_ids
        if predict_bonus_token and (
            any_penalties_or_bad_words or needs_thinking_combine
        ):
            # 当投机解码启用时，将基础输出与投机token组合
            output_token_ids = self._combine_outputs_with_spec_tokens(
                output_token_ids,
                sampling_metadata.spec_token_ids,
            )

        # 应用允许的token ID白名单
        if sampling_metadata.allowed_token_ids_mask is not None:
            logits.masked_fill_(sampling_metadata.allowed_token_ids_mask, float("-inf"))

        # 应用坏词排除
        if bad_words_token_ids:
            apply_bad_words(logits, bad_words_token_ids, output_token_ids)

        # 应用非argmax不变的logits处理器（会影响贪心采样的处理器）
        for processor in sampling_metadata.logitsprocs.non_argmax_invariant:
            logits = processor.apply(logits)

        # 应用惩罚（如频率惩罚等）
        logits = self.apply_penalties(logits, sampling_metadata, output_token_ids)

        # 应用思考预算状态处理
        if holder is not None and holder.has_tracked_requests():
            holder.update_state(
                output_token_ids,
                sampling_metadata.spec_token_ids,
                repeat_indices=None,
            )
            logits = holder.apply_to_logits(
                logits,
                predict_bonus_token,
                sampling_metadata.spec_token_ids,
            )
        return logits

    @staticmethod
    def apply_penalties(
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        output_token_ids: list[list[int]],
    ) -> torch.Tensor:
        """应用采样惩罚。

        包括重复惩罚(repetition penalty)、频率惩罚(frequency penalty)和存在惩罚(presence penalty)。

        参数:
            logits: 输入logits张量 [batch_size, vocab_size]
            sampling_metadata: 采样元数据
            output_token_ids: 每个请求已生成的输出token ID列表

        返回:
            应用惩罚后的logits张量
        """
        if sampling_metadata.no_penalties:
            return logits

        assert sampling_metadata.prompt_token_ids is not None
        return apply_all_penalties(
            logits,
            sampling_metadata.prompt_token_ids,
            sampling_metadata.presence_penalties,
            sampling_metadata.frequency_penalties,
            sampling_metadata.repetition_penalties,
            output_token_ids,
        )
