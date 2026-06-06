# vLLM 并行策略深度解析：TP、PP、DP、EP、CP

> 本文档从面试角度深入剖析 vLLM 中所有并行策略的实现细节，包括张量并行 (TP)、流水线并行 (PP)、数据并行 (DP)、专家并行 (EP)、上下文并行 (CP)，以及它们在代码中的具体体现。

---

## 目录

- [1. 并行策略总览](#1-并行策略总览)
- [2. 张量并行 (Tensor Parallelism)](#2-张量并行-tensor-parallelism)
- [3. 流水线并行 (Pipeline Parallelism)](#3-流水线并行-pipeline-parallelism)
- [4. 数据并行 (Data Parallelism)](#4-数据并行-data-parallelism)
- [5. 专家并行 (Expert Parallelism)](#5-专家并行-expert-parallelism)
- [6. 上下文并行 (Context Parallelism)](#6-上下文并行-context-parallelism)
- [7. 并行策略的组合与交互](#7-并行策略的组合与交互)
- [8. 面试深度问答](#8-面试深度问答)

---

## 1. 并行策略总览

### 1.1 并行维度关系图

```
┌─────────────────────────────────────────────────────────────────┐
│                    vLLM 并行维度                                 │
│                                                                  │
│  总 GPU 数 = TP × PP × PCP × DP                                 │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  模型并行 (单个模型拆分到多卡)                             │    │
│  │  ├── TP: 张量并行 (层内拆分)                             │    │
│  │  ├── PP: 流水线并行 (层间拆分)                           │    │
│  │  └── EP: 专家并行 (MoE 专家拆分)                         │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  数据并行 (模型复制，处理不同数据)                         │    │
│  │  └── DP: 数据并行 (多个模型副本)                         │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  序列并行 (长序列拆分)                                    │    │
│  │  ├── PCP: 预填充上下文并行 (prefill 阶段)                │    │
│  │  └── DCP: 解码上下文并行 (decode 阶段)                   │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  Rank 布局顺序: ExternalDP × DP × PP × PCP × TP                │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 各并行策略对比

| 策略 | 拆分维度 | 通信模式 | 适用场景 | 代码位置 |
|------|----------|----------|----------|----------|
| **TP** | 层内权重 | AllReduce/AllGather | 单层太大放不下单卡 | `model_executor/layers/linear.py` |
| **PP** | 层间 | Send/Recv | 模型太深放不下单卡 | `v1/worker/gpu_worker.py` |
| **DP** | 数据 | AllReduce (状态同步) | 提高吞吐量 | `v1/engine/core.py` |
| **EP** | MoE 专家 | All-to-All | MoE 模型 | `distributed/device_communicators/all2all.py` |
| **DCP** | KV Cache (decode) | AllGather + LSE Reduce | 长上下文 decode | `model_executor/layers/attention/mla_attention.py` |
| **PCP** | KV Cache (prefill) | AllGather/ReduceScatter | 长上下文 prefill | `model_executor/layers/fused_moe/runner/` |

### 1.3 并行组创建

**文件：** `vllm/distributed/parallel_state.py`, line 1506

```python
def initialize_model_parallel(
    tensor_model_parallel_size: int,
    pipeline_model_parallel_size: int,
    data_parallel_size: int,
    prefill_context_model_parallel_size: int,
    decode_context_model_parallel_size: int,
    ...
):
    # 将全局 rank 空间重塑为 5D 张量
    all_ranks = torch.arange(world_size).reshape(
        -1,                              # ExternalDP
        data_parallel_size,              # DP
        pipeline_model_parallel_size,    # PP
        prefill_context_model_parallel_size,  # PCP
        tensor_model_parallel_size,      # TP
    )

    # 通过转置+重塑+unbind 提取各维度的组
    # TP 组: 相邻 rank
    # PP 组: 同一 TP/PCP 位置的不同 PP 阶段
    # DP 组: 同一 TP/PP/PCP 位置的不同 DP 副本
    # EP 组: 跨 DP × PCP × TP
```

### 1.4 GroupCoordinator —— 通信核心

```python
# vllm/distributed/parallel_state.py, line 290
class GroupCoordinator:
    """每个并行维度的通信协调器"""

    ranks: list[int]           # 组内的全局 rank 列表
    world_size: int            # 组大小
    rank_in_group: int         # 组内本地 rank

    cpu_group: ProcessGroup    # Gloo 后端 (CPU 通信)
    device_group: ProcessGroup # NCCL 后端 (GPU 通信)
    device_communicator: DeviceCommunicatorBase  # 优化通信路径

    # 通信操作
    def all_reduce(self, input_): ...
    def all_gather(self, input_, dim): ...
    def reduce_scatter(self, input_, dim): ...
    def send(self, tensor, dst): ...
    def recv(self, tensor, src): ...
    def send_tensor_dict(self, tensor_dict, dst): ...
    def recv_tensor_dict(self, src): ...
    def dispatch(self, hidden_states, router_logits): ...  # EP dispatch
    def combine(self, hidden_states): ...                  # EP combine
```

---

## 2. 张量并行 (Tensor Parallelism)

### 2.1 核心思想

将每一层的权重矩阵拆分到多个 GPU 上，每个 GPU 计算部分结果，然后通过通信合并。

### 2.2 两种并行模式

```
Column Parallel (列并行):
  权重 A 按列拆分: A = [A_1 | A_2 | ... | A_p]
  每个 GPU 计算: Y_i = X × A_i
  最后: AllGather(Y_1, Y_2, ..., Y_p) → Y

Row Parallel (行并行):
  权重 A 按行拆分: A = [A_1; A_2; ...; A_p]
  输入 X 也拆分: X = [X_1; X_2; ...; X_p]
  每个 GPU 计算: Y_i = X_i × A_i
  最后: AllReduce(Y_1, Y_2, ..., Y_p) → Y = Σ Y_i
```

### 2.3 ColumnParallelLinear 实现

**文件：** `vllm/model_executor/layers/linear.py`, line 407

```python
class ColumnParallelLinear(LinearBase):
    """列并行线性层：权重按输出维度拆分"""

    def __init__(self, input_size, output_size, ...):
        # 输出维度拆分到 TP 个 GPU
        self.output_size_per_partition = output_size // tp_size
        # 每个 GPU 只持有 A_i (输出维度的 1/tp_size)
        self.weight = Parameter(
            torch.empty(self.output_size_per_partition, input_size, ...)
        )

    def forward(self, input_):
        # 本地矩阵乘法: Y_i = X × A_i
        output_parallel = F.linear(input_, self.weight, self.bias)

        if self.gather_output:
            # AllGather: 拼接所有 GPU 的结果
            output = tensor_model_parallel_all_gather(output_parallel)
            return output
        else:
            # 返回本地分片 (后续层会处理)
            return output_parallel
```

**权重加载：**
```python
def weight_loader(self, param, loaded_weight, ...):
    # 只加载当前 TP rank 对应的分片
    start_idx = self.tp_rank * shard_size
    loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)
    param.data.copy_(loaded_weight)
```

### 2.4 RowParallelLinear 实现

**文件：** `vllm/model_executor/layers/linear.py`, line 1389

```python
class RowParallelLinear(LinearBase):
    """行并行线性层：权重按输入维度拆分"""

    def __init__(self, input_size, output_size, ...):
        # 输入维度拆分到 TP 个 GPU
        self.input_size_per_partition = input_size // tp_size
        self.weight = Parameter(
            torch.empty(output_size, self.input_size_per_partition, ...)
        )

    def forward(self, input_):
        # 输入已经拆分好了 (来自上一层的 ColumnParallel)
        # 本地矩阵乘法: Y_i = X_i × A_i
        output_parallel = F.linear(input_, self.weight)

        if self.reduce_results and tp_size > 1:
            # AllReduce: 求和所有 GPU 的结果
            output = tensor_model_parallel_all_reduce(output_parallel)
            return output
        else:
            return output_parallel
```

### 2.5 TP 在 Transformer 层中的应用

```
Transformer Layer 中的 TP 拆分:

Attention:
  Q, K, V 投影 (ColumnParallel):
    W_q: [d_model, d_model] → 每 GPU: [d_model, d_model/tp]
    W_k: [d_model, d_model] → 每 GPU: [d_model, d_model/tp]
    W_v: [d_model, d_model] → 每 GPU: [d_model, d_model/tp]

  注意力计算: 每个 GPU 独立计算自己的 head

  输出投影 (RowParallel):
    W_o: [d_model, d_model] → 每 GPU: [d_model/tp, d_model]
    AllReduce 合并

FFN:
  W1 (ColumnParallel): [d_model, 4*d_model] → [d_model, 4*d_model/tp]
  W2 (RowParallel): [4*d_model, d_model] → [4*d_model/tp, d_model]
  AllReduce 合并
```

### 2.6 AllReduce 优化

```python
# vllm/distributed/device_communicators/cuda_communicator.py
def all_reduce(self, input_):
    # 选择最快的实现:
    # 1. NCCL Symmetric Memory (最快，条件满足时)
    # 2. QuickReduce (AMD MI300)
    # 3. FlashInfer All-Reduce
    # 4. CustomAllreduce (IPC, NVLink 全连接)
    # 5. PyNCCL (兜底)
```

**CustomAllreduce (IPC 优化)：**
```python
# vllm/distributed/device_communicators/custom_all_reduce.py
# 使用 cudaIpcGetMemHandle / cudaIpcOpenMemHandle 实现零拷贝
# 限制: 同节点、NVLink 连接、最多 8 GPU
```

### 2.7 面试考点

**Q: TP 的通信量是多少？**

A: 每层需要 2 次通信：
- ColumnParallel 输出: AllGather，数据量 = `batch_size × seq_len × hidden_size`
- RowParallel 输出: AllReduce，数据量 = `batch_size × seq_len × hidden_size`

**Q: TP 什么时候效率最高？**

A: 当单层太大放不下单卡时。TP 的通信是同步的，每层都需要等待，所以层数越多、每层通信量越大，效率越低。

---

## 3. 流水线并行 (Pipeline Parallelism)

### 3.1 核心思想

将模型的不同层分配到不同 GPU，形成流水线。数据从第一个 PP 阶段流向最后一个。

### 3.2 通信机制

**文件：** `vllm/v1/worker/gpu_worker.py`, line 784

```python
# Worker.execute_model()
def execute_model(self, scheduler_output):
    # 非第一个 PP 阶段: 接收中间张量
    if not is_first_pp_rank:
        tensor_dict, comm_handles = get_pp_group().irecv_tensor_dict(
            all_gather_group=get_tp_group(),
            all_gather_tensors=all_gather_tensors,
        )
        intermediate_tensors = AsyncIntermediateTensors(tensor_dict, comm_handles)

    # 运行模型
    output = self.model_runner.execute_model(scheduler_output, intermediate_tensors)

    # 非最后一个 PP 阶段: 发送中间张量
    if not is_last_pp_rank:
        self._pp_send_work = get_pp_group().isend_tensor_dict(
            output.tensors,
            all_gather_group=get_tp_group(),
            all_gather_tensors=all_gather_tensors,
        )
        return None  # 不是最终输出

    return output
```

### 3.3 send_tensor_dict / recv_tensor_dict

```python
# vllm/distributed/parallel_state.py, line 833
def send_tensor_dict(self, tensor_dict, dst):
    # 1. 通过 CPU 组 (Gloo) 发送元数据 (key, shape, dtype)
    cpu_group.send_object(metadata)

    # 2. 通过设备组 (NCCL) 发送实际张量
    for key, tensor in tensor_dict.items():
        device_group.isend(tensor, dst)
```

**优化：TP 内 AllGather**
```python
# 每个 TP rank 只发送自己的分片
# 接收方通过 TP 内 AllGather 重建完整张量
# 通信量减少 TP 倍
```

### 3.4 AsyncIntermediateTensors

```python
# vllm/v1/worker/gpu_worker.py, line 80
class AsyncIntermediateTensors:
    """惰性同步包装器"""
    def __init__(self, tensor_dict, comm_handles):
        self.comm_handles = comm_handles
        self._tensors = None

    @property
    def tensors(self):
        if self._tensors is None:
            self.wait_for_comm()  # 首次访问时同步
            self._tensors = ...
        return self._tensors
```

### 3.5 Pipeline 调度

```python
# vllm/v1/engine/core.py, line 538
def step_with_batch_queue(self):
    # 提交 batch 到队列 (非阻塞)
    future = self.model_executor.execute_model(scheduler_output, non_block=True)
    batch_queue.appendleft((future, scheduler_output))

    # 如果队列未满，立即返回 (不等待结果)
    if len(batch_queue) < batch_queue_size:
        return None, True  # "先调度，后等待"

    # 队列满了，等待最旧的 batch 完成
    oldest_future = batch_queue.pop()
    return oldest_future.result()
```

### 3.6 面试考点

**Q: PP 的通信量是多少？**

A: 每层边界传输一次中间张量，数据量 = `batch_size × seq_len × hidden_size`。但通过 TP 内 AllGather 优化，实际通信量减少 TP 倍。

**Q: PP 的 bubble 如何减少？**

A: vLLM 使用 batch queue 实现微批处理流水线：
```
Batch 1: [PP0] [PP1] [PP2] [PP3]
Batch 2:      [PP0] [PP1] [PP2] [PP3]
Batch 3:           [PP0] [PP1] [PP2] [PP3]
```
多个 batch 同时在 pipeline 中流动，减少 bubble。

---

## 4. 数据并行 (Data Parallelism)

### 4.1 核心思想

每个 DP rank 持有完整的模型副本，处理不同的请求。DP 之间通过 AllReduce 同步梯度（训练）或状态（推理）。

### 4.2 DP 引擎架构

**文件：** `vllm/v1/engine/core.py`, line 1760

```python
class DPEngineCoreProc(EngineCoreProc):
    """MoE 模型的数据并行引擎核心"""

    def run_busy_loop(self):
        while self._handle_shutdown():
            self._process_input_queue()
            self._process_engine_step()

            # 每 32 步检查全局状态
            if step_count % 32 == 0:
                has_unfinished = self._has_global_unfinished_reqs()
                if not has_unfinished:
                    self._pause()  # 所有 rank 都空闲，暂停

    def _has_global_unfinished_reqs(self):
        """AllReduce 检查所有 DP rank 是否有未完成请求"""
        local_has = self.scheduler.has_requests()
        # 使用 Gloo AllReduce (CPU 通信)
        global_has = all_reduce(local_has, self.dp_group)
        return global_has
```

### 4.3 DP 状态同步

```python
# vllm/config/parallel.py, line 681
def sync_dp_state(self):
    """同步 DP 状态: has_unfinished + pending_pause"""
    # 2 元素张量的 AllReduce
    # 元素 0: has_unfinished (OR 语义)
    # 元素 1: pending_pause (共识检查)
    tensor = torch.tensor([has_unfinished, pending_pause])
    all_reduce(tensor, dp_group, op=SUM)
```

### 4.4 Wave 机制

```
DP Wave 协调:

1. 所有 DP rank 空闲 → PAUSED 状态
2. 新请求到达 → 前端通知 DPCoordinator
3. DPCoordinator 广播 START_DP_WAVE → 所有 DP rank 唤醒
4. 所有 DP rank 处理请求
5. 所有 DP rank 空闲 → 再次 PAUSED

DPCoordinator 进程:
  XPUB (stats) → API Server (负载统计)
  PULL (stats) ← DP engines (负载统计)
  XPUB (wave) → DP engines (wave 命令)
```

### 4.5 DP 的三种负载均衡模式

| 模式 | 说明 | 适用场景 |
|------|------|----------|
| **Internal LB** | 单 API 端点，内部负载均衡 | 默认模式 |
| **Hybrid LB** | 每节点独立 API，上游 LB 分发 | 多节点部署 |
| **External LB** | 每 DP rank 独立端点，外部路由 | 大规模部署 |

### 4.6 面试考点

**Q: DP 和 EP 的关系？**

A: 对于 MoE 模型，`EP_SIZE = DP_SIZE × PCP_SIZE × TP_SIZE`。DP 不仅用于数据并行，还参与 EP 的 All-to-All 通信。

**Q: DP rank 之间如何同步？**

A: 使用 Gloo AllReduce（CPU 通信），每 32 步检查一次全局状态。不需要 NCCL，因为同步的是布尔值，不是大张量。

---

## 5. 专家并行 (Expert Parallelism)

### 5.1 核心思想

将 MoE 的专家分布到多个 GPU 上，每个 GPU 只持有部分专家。Token 通过 All-to-All 通信路由到对应专家。

### 5.2 EP 的并行维度

```python
# EP 跨越 DP × PCP × TP 维度
ep_size = dp_size * pcp_size * tp_size

# 当 EP 启用时，MoE 层的 TP 变为 1
# 每个 GPU 持有 num_experts / ep_size 个本地专家
```

### 5.3 All-to-All 通信

**文件：** `vllm/distributed/device_communicators/all2all.py`

```
MoE 前向传播:
  1. Gate 计算: gate_logits = gate_model(hidden_states)
  2. Top-K 选择: topk_indices = topk(gate_logits, k=8)
  3. Dispatch: token → 专家 GPU (All-to-All)
  4. 专家计算: expert_output = expert(token)
  5. Combine: 专家 GPU → 原始 GPU (All-to-All)
  6. 加权求和: output = sum(weight × expert_output)
```

### 5.4 All-to-All 后端

| 后端 | 适用场景 | 特点 |
|------|----------|------|
| **AgRS All2All** | 通用 | AllGather + ReduceScatter 模拟 |
| **DeepEP HT** | Prefill | 高吞吐，支持 DBO 微批处理 |
| **DeepEP LL** | Decode | 低延迟，支持 FP8 dispatch |
| **FlashInfer NVLink** | NVLink 系统 | 单边/双边通信 |
| **NIXL EP** | RDMA | 弹性扩缩容支持 |
| **MoRI** | 通用 | MoRI EP 内核 |

### 5.5 FusedMoEParallelConfig

```python
# vllm/model_executor/layers/fused_moe/config.py, line 1080
class FusedMoEParallelConfig:
    tp_size: int       # MoE 层的 TP 大小 (EP 启用时为 1)
    tp_rank: int       # MoE 层的 TP rank
    ep_size: int       # EP 大小
    ep_rank: int       # EP rank
    dp_size: int       # DP 大小
    dp_rank: int       # DP rank

    @staticmethod
    def make(vllm_config, ...):
        if enable_expert_parallel:
            # EP 启用: MoE 层的 TP=1，EP = DP × PCP × TP
            return FusedMoEParallelConfig(
                tp_size=1, tp_rank=0,
                ep_size=dp_size * pcp_size * tp_size,
                ep_rank=...,
            )
        else:
            # EP 未启用: MoE 层使用标准 TP
            return FusedMoEParallelConfig(
                tp_size=tp_size, tp_rank=tp_rank,
                ep_size=1, ep_rank=0,
            )
```

### 5.6 Expert Map Manager

```python
# vllm/model_executor/layers/fused_moe/expert_map_manager.py
class ExpertMapManager:
    """管理全局专家 ID 到本地专家 ID 的映射"""

    def determine_expert_map(self):
        # 1. 均匀分布专家
        base_experts = global_num_experts // ep_size

        # 2. 创建映射: global_id → local_id (-1 表示非本地)
        expert_map = torch.full((global_num_experts,), -1)
        for i, expert_id in enumerate(local_experts):
            expert_map[expert_id] = i

        return expert_map
```

### 5.7 面试考点

**Q: EP 和 TP 在 MoE 层的区别？**

A:
- **TP (EP 未启用)**: 每个专家的权重拆分到多个 GPU (ColumnParallel + RowParallel)
- **EP (EP 启用)**: 每个专家完整放在一个 GPU 上，不同专家在不同 GPU

**Q: EP 的通信模式？**

A: All-to-All，每个 token 需要发送到 6 个不同 GPU (Top-6 routing)。通信量 = `num_tokens × hidden_size × 6 / ep_size`。

**Q: EPLB 如何工作？**

A: 收集负载统计 → 计算新映射 → 复制热门专家到多个 GPU → 重新分配权重。冗余专家减少跨 GPU 路由。

---

## 6. 上下文并行 (Context Parallelism)

### 6.1 核心思想

将长序列的 KV Cache 拆分到多个 GPU 上，每个 GPU 只持有部分 KV。注意力计算时，每个 GPU 计算局部注意力，然后合并结果。

### 6.2 DCP (Decode Context Parallelism)

**配置：** `decode_context_parallel_size`

**组创建：** DCP 复用 TP GPU。每个 TP 组拆分为 `tp_size/dcp_size` 个 DCP 子组。

**文件：** `vllm/model_executor/layers/attention/mla_attention.py`, line 764

```python
# MLA 注意力中的 DCP 实现
def forward_mqa(self, ...):
    # 1. Query AllGather: 每个 DCP rank 获取完整查询
    mqa_q = get_dcp_group().all_gather(mqa_q, dim=1)

    # 2. 局部注意力: 每个 DCP rank 只 attend 到自己的 KV 分片
    attn_output = flash_mla_with_kvcache(mqa_q, local_kv_cache, ...)

    # 3. 合并结果: 使用 LSE (Log-Sum-Exp) 合并
    if dcp_comm_backend == "a2a":
        # All-to-All 方式: 交换部分输出 + LSE
        output = dcp_a2a_lse_reduce(attn_output, lse)
    else:
        # AllGather + ReduceScatter 方式
        output = cp_lse_ag_out_rs(attn_output, lse)
```

**LSE 合并公式：**
```
对于两个 DCP rank 的输出 (out1, lse1) 和 (out2, lse2):
  max_lse = max(lse1, lse2)
  scale1 = exp(lse1 - max_lse)
  scale2 = exp(lse2 - max_lse)
  out = (out1 * scale1 + out2 * scale2) / (scale1 + scale2)
```

### 6.3 PCP (Prefill Context Parallelism)

**配置：** `prefill_context_parallel_size`

**组创建：** PCP rank 与 TP rank 交错排列。

**通信：** MoE 层使用 AllGather/ReduceScatter。

```python
# vllm/model_executor/layers/fused_moe/runner/moe_runner.py
# MoE 前向传播中的 PCP 处理

# Dispatch: 从所有 PCP rank 收集 token
if pcp_size > 1:
    hidden_states = get_pcp_group().all_gather(hidden_states, dim=0)
    router_logits = get_pcp_group().all_gather(router_logits, dim=0)

# Combine: 分散结果回各 PCP rank
if pcp_size > 1:
    hidden_states = get_pcp_group().reduce_scatter(hidden_states, dim=0)
```

### 6.4 面试考点

**Q: DCP 和 PCP 的区别？**

A:
- **DCP**: 用于 decode 阶段，拆分 KV Cache，在注意力层内通信
- **PCP**: 用于 prefill 阶段，拆分序列，在 MoE 层间通信

**Q: DCP 为什么需要 LSE 合并？**

A: 因为 softmax 不是线性操作，不能直接对部分结果求和。LSE (Log-Sum-Exp) 提供了数值稳定的合并方式。

**Q: DCP 和 TP 的关系？**

A: DCP 复用 TP GPU。例如 TP=8, DCP=4: 每个 TP 组 (8 GPU) 拆分为 2 个 DCP 子组 (各 4 GPU)。

---

## 7. 并行策略的组合与交互

### 7.1 组合关系

```
总 GPU 数 = TP × PP × PCP × DP

DeepSeek V4 示例 (2 节点, 16 GPU):
  TP = 1 (MLA 不需要 TP)
  PP = 1
  PCP = 1
  DP = 16
  EP = DP × PCP × TP = 16

  每个 GPU: 完整模型 + 416/16 = 26 个物理专家
```

### 7.2 通信模式总结

| 并行 | 通信方向 | 通信频率 | 数据量 |
|------|----------|----------|--------|
| TP | 层内 | 每层 2 次 | batch × seq × hidden |
| PP | 层间 | 每层边界 | batch × seq × hidden / TP |
| DP | 引擎级 | 每 32 步 | 2 元素 (布尔值) |
| EP | MoE 层 | 每 MoE 层 2 次 | tokens × hidden × topk / EP |
| DCP | 注意力层 | 每注意力层 | batch × seq × hidden / DCP |
| PCP | MoE 层 | 每 MoE 层 2 次 | tokens × hidden / PCP |

### 7.3 Rank 布局示例

```
16 GPU, TP=2, PP=2, DP=2:

Rank 布局: DP × PP × TP
  DP=0, PP=0, TP=0: GPU 0
  DP=0, PP=0, TP=1: GPU 1
  DP=0, PP=1, TP=0: GPU 2
  DP=0, PP=1, TP=1: GPU 3
  DP=1, PP=0, TP=0: GPU 4
  DP=1, PP=0, TP=1: GPU 5
  DP=1, PP=1, TP=0: GPU 6
  DP=1, PP=1, TP=1: GPU 7

TP 组: [0,1], [2,3], [4,5], [6,7]
PP 组: [0,2], [1,3], [4,6], [5,7]
DP 组: [0,4], [1,5], [2,6], [3,7]
```

---

## 8. 面试深度问答

### Q1: 为什么 DeepSeek V4 用 DP=16 而不是 TP=16？

**A:** 因为 MLA 的 KV Cache 是压缩的 (576 维)，单层计算量不大，TP 的通信开销反而成为瓶颈。DP 让每个 GPU 持有完整模型，避免了 TP 的每层通信。EP 让专家分布在不同 GPU 上，通过 All-to-All 通信。

### Q2: TP 和 EP 在 MoE 层的区别？

**A:**
- **TP (EP 未启用)**: 每个专家的权重拆分到多个 GPU，每个 GPU 计算部分结果
- **EP (EP 启用)**: 每个专家完整放在一个 GPU 上，token 通过 All-to-All 发送到专家

EP 的优势是减少了专家内的通信，但增加了 All-to-All 通信。

### Q3: DCP 如何与 MLA 配合？

**A:** DCP 将 KV Cache 分片到多个 GPU。在 decode 阶段：
1. Query AllGather：每个 DCP rank 获取完整查询
2. 局部注意力：每个 DCP rank 只 attend 到自己的 KV 分片
3. LSE 合并：使用 Log-Sum-Exp 合并部分结果

对于 MLA，DCP 的 a2a 后端只需 2 次 NCCL 调用（vs ag_rs 的 3 次）。

### Q4: vLLM 的 TP 通信优化有哪些？

**A:**
1. **CustomAllreduce**：IPC 零拷贝，NVLink 全连接
2. **NCCL Symmetric Memory**：最快的 AllReduce
3. **TP 内 AllGather**：PP 通信时，每个 TP rank 只发送分片，接收方 AllGather 重建
4. **FlashInfer AllReduce**：FlashInfer 优化的 AllReduce

### Q5: EP 的 All-to-All 有哪些优化？

**A:**
1. **DeepEP HT**：高吞吐内核，支持 DBO 微批处理
2. **DeepEP LL**：低延迟内核，FP8 dispatch 减少 2x 通信量
3. **FlashInfer NVLink**：NVLink 单边/双边通信
4. **NIXL EP**：RDMA，支持弹性扩缩容
