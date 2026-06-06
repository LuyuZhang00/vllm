# vLLM 分布式推理通信、P/D 分离与 DeepSeek V4 多级多卡部署详解

> 本文档深入剖析 vLLM 的分布式通信机制、Prefill/Decode 分离架构、以及通过 lws 部署 DeepSeek V4 模型的多级多卡推理方案。

---

## 目录

- [1. 分布式推理中不同 Pod 之间的通信机制](#1-分布式推理中不同-pod-之间的通信机制)
- [2. Prefill 与 Decode 节点之间的通信](#2-prefill-与-decode-节点之间的通信)
- [3. 通过 lws 部署 DeepSeek V4 的多级多卡推理](#3-通过-lws-部署-deepseek-v4-的多级多卡推理)

---

## 1. 分布式推理中不同 Pod 之间的通信机制

### 1.1 通信架构总览

vLLM 的分布式通信建立在 `torch.distributed` 之上，采用分层架构：

```
┌─────────────────────────────────────────────────────────────────┐
│                    应用层通信                                      │
│  TP: AllReduce/AllGather    PP: Send/Recv    EP: All-to-All     │
│  DP: Wave Sync (AllReduce)  KV Transfer: RDMA/NCCL              │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                  GroupCoordinator (通信协调器)                     │
│  每个并行维度 (TP/PP/DP/EP/DCP/PCP) 一个实例                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │device_group  │  │cpu_group     │  │device_comm.  │           │
│  │(NCCL)        │  │(Gloo)        │  │(优化路径)     │           │
│  └──────────────┘  └──────────────┘  └──────────────┘           │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                  底层通信后端                                      │
│  NCCL (GPU-GPU)  │  Gloo (CPU-CPU)  │  RDMA (节点间)             │
│  共享内存 (进程间) │  ZMQ (控制平面)   │  TCP (元数据)              │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 并行维度与 Rank 布局

vLLM 支持多种并行维度，rank 布局顺序为：

```
ExternalDP × DP × PP × PCP × TP
```

| 并行维度 | 说明 | 通信模式 |
|----------|------|----------|
| **TP (Tensor Parallelism)** | 张量并行，拆分每层的权重 | AllReduce, AllGather, ReduceScatter |
| **PP (Pipeline Parallelism)** | 流水线并行，拆分不同层 | Send/Recv (P2P) |
| **DP (Data Parallelism)** | 数据并行，复制模型 | AllReduce (同步), ZMQ (统计) |
| **EP (Expert Parallelism)** | 专家并行，拆分 MoE 专家 | All-to-All |
| **DCP (Decode Context Parallel)** | 解码上下文并行 | AllReduce |
| **PCP (Prefill Context Parallel)** | 预填充上下文并行 | AllReduce |

**关系：** `EP_SIZE = TP_SIZE × DP_SIZE`

### 1.3 GroupCoordinator —— 通信核心抽象

每个并行维度都有一个 `GroupCoordinator` 实例：

```python
# vllm/distributed/parallel_state.py, line 290
class GroupCoordinator:
    device_group: ProcessGroup       # NCCL 后端，GPU-GPU 通信
    cpu_group: ProcessGroup          # Gloo 后端，CPU-CPU 协调
    device_communicator: DeviceCommunicatorBase  # 优化通信路径
    mq_broadcaster: MessageQueue     # 共享内存广播 (TP 组)
```

**提供的操作：**
- `all_reduce(input_)` — 全归约
- `all_gather(input_, dim)` — 全收集
- `reduce_scatter(input_, dim)` — 归约分散
- `send(tensor, dst)` / `recv(tensor, src)` — 点对点
- `send_tensor_dict(dict)` / `recv_tensor_dict()` — 张量字典传输
- `dispatch(tokens)` / `combine(tokens)` — MoE 专家路由

### 1.4 Tensor Parallelism (TP) 通信

TP 将模型的每一层拆分到多个 GPU 上，需要频繁的 AllReduce 操作。

**通信路径选择链：**

```python
# CudaCommunicator.all_reduce() 的选择逻辑:
def all_reduce(self, input_):
    # 1. NCCL Symmetric Memory (最快，条件满足时)
    if symmetric_memory_enabled and world_size_ok:
        return nccl_symm_mem_all_reduce(input_)

    # 2. QuickReduce (ROCm/AMD MI300 系列)
    if is_rocm_mi300:
        return quickreduce_all_reduce(input_)

    # 3. FlashInfer All-Reduce
    if flashinfer_available:
        return flashinfer_all_reduce(input_)

    # 4. CustomAllreduce (NVLink 全连接，最多 8 GPU)
    if is_fully_connected_nvlink and world_size in [2,4,6,8]:
        return custom_allreduce(input_)  # 使用 IPC 共享内存

    # 5. SymmMem All-Reduce
    if symmetric_memory_available:
        return symm_mem_all_reduce(input_)

    # 6. PyNCCL (兜底方案)
    return pynccl_all_reduce(input_)
```

**CustomAllreduce 的 IPC 机制：**

```
GPU 0: cudaIpcGetMemHandle() → handle_0
GPU 1: cudaIpcGetMemHandle() → handle_1
  ↓ 通过 CPU 交换 handle
GPU 0: cudaIpcOpenMemHandle(handle_1) → 访问 GPU 1 的内存
GPU 1: cudaIpcOpenMemHandle(handle_0) → 访问 GPU 0 的内存
  ↓ 零拷贝 AllReduce (无需通过 CPU 中转)
```

**TP 在模型中的应用：**

```python
# ColumnParallelLinear (列并行线性层)
class ColumnParallelLinear:
    def forward(self, x):
        # 每个 GPU 计算部分结果
        partial_output = F.linear(x, self.weight)  # weight 已拆分
        # AllReduce 同步
        output = all_reduce(partial_output)
        return output

# RowParallelLinear (行并行线性层)
class RowParallelLinear:
    def forward(self, x):
        # 每个 GPU 计算部分结果
        partial_output = F.linear(x, self.weight)  # weight 已拆分
        # AllReduce 同步
        output = all_reduce(partial_output)
        return output
```

### 1.5 Pipeline Parallelism (PP) 通信

PP 将不同层分配到不同 GPU，相邻 PP 阶段之间传输中间张量。

```python
# Worker.execute_model() (gpu_worker.py, line 829)

# 非第一个 PP 阶段: 接收中间张量
if not is_first_pp_rank:
    intermediate_tensors = get_pp_group().irecv_tensor_dict()  # 非阻塞

# 运行模型
output = self.model_runner.execute_model(scheduler_output, intermediate_tensors)

# 非最后一个 PP 阶段: 发送中间张量
if not is_last_pp_rank:
    get_pp_group().isend_tensor_dict(output)  # 非阻塞
    return None  # 信号: 不是最终输出

# 最后一个 PP 阶段: 返回最终输出
return output
```

**传输机制：**
```python
# send_tensor_dict / recv_tensor_dict
def send_tensor_dict(self, tensor_dict, dst):
    # 1. 通过 CPU 组 (Gloo) 发送元数据 (key, shape, dtype)
    cpu_group.send_object(metadata)

    # 2. 通过设备组 (NCCL) 发送实际张量
    for key, tensor in tensor_dict.items():
        device_group.isend(tensor, dst)
```

**异步中间张量：**
```python
class AsyncIntermediateTensors:
    """惰性同步包装器，首次访问时等待通信完成"""
    def __init__(self, comm_handles):
        self.comm_handles = comm_handles
        self._tensors = None

    @property
    def tensors(self):
        if self._tensors is None:
            self.wait_for_comm()  # 首次访问时同步
            self._tensors = ...
        return self._tensors
```

### 1.6 Data Parallelism (DP) 通信

DP 复制模型到多组 GPU 上，每组处理不同的请求。

**DP 组创建：**
```python
# parallel_state.py, line 1665
# rank 布局: ExternalDP × DP × PP × PCP × TP
# 通过转置 DP 维度到最后一维，然后 unbind 提取 DP 组
dp_ranks = global_rank_array.transpose(dp_dim, -1).unbind(-1)
```

**DP 的三种负载均衡模式：**

| 模式 | 说明 | 适用场景 |
|------|------|----------|
| **Internal LB** | 单 API 端点，内部负载均衡 | 默认模式 |
| **Hybrid LB** | 每节点独立 API，上游 LB 分发 | 多节点部署 |
| **External LB** | 每 DP rank 独立端点，外部路由 | 大规模部署 |

**DPCoordinator 协调器：**

```python
# vllm/v1/engine/coordinator.py
class DPCoordinator:
    # 三个 ZMQ 套接字:
    publish_front: XPUB   # 向前端发布统计和 wave 状态
    output_back: PULL     # 从 DP 引擎接收统计和 wave 通知
    publish_back: XPUB    # 向 DP 引擎发布 wave 命令

    # 职责:
    # 1. 收集每个 DP 引擎的负载统计 (等待/运行队列长度)
    # 2. 管理 "请求 wave" 同步
    # 3. 广播 START_DP_WAVE 唤醒暂停的引擎
```

**Wave 机制：**
```
状态: RUNNING ↔ PAUSED

1. 引擎启动 → RUNNING 状态
2. 所有请求处理完毕 → 每 32 步 all-reduce 检查
3. all-reduce 结果: 所有 rank 都空闲 → PAUSED 状态
4. 新请求到达 → 前端通知 DPCoordinator
5. DPCoordinator 广播 START_DP_WAVE → 所有引擎唤醒 → RUNNING
```

### 1.7 Expert Parallelism (EP) 通信

EP 将 MoE 模型的专家分布到多个 GPU 上，使用 All-to-All 通信。

**All-to-All 后端：**

| 后端 | 说明 | 适用场景 |
|------|------|----------|
| `allgather_reducescatter` | AllGather + ReduceScatter 模拟 | 通用 |
| `deepep_high_throughput` | DeepEP 高吞吐内核 | Prefill 阶段 |
| `deepep_low_latency` | DeepEP 低延迟内核 | Decode 阶段 |
| `flashinfer_nvlink_one_sided` | FlashInfer NVLink 单边 | MNNVL 系统 |
| `flashinfer_nvlink_two_sided` | FlashInfer NVLink 双边 | MNNVL 系统 |
| `mori_high_throughput` | Mori 高吞吐 | 通用 |
| `nixl_ep` | NIXL-based EP | RDMA 集群 |

**EP 的工作流程：**
```
Token 路由:
1. Gate 计算: gate_logits = gate_model(token)  → 每个专家的分数
2. Top-K 选择: selected_experts = topk(gate_logits, k=2)  → 选 2 个专家
3. All-to-All Dispatch: 将 token 发送到对应专家所在的 GPU
4. 专家计算: expert_output = expert(token)  → 在专家 GPU 上计算
5. All-to-All Combine: 将专家输出发回原始 GPU
6. 加权求和: output = sum(weight_i * expert_output_i)
```

### 1.8 节点间发现与连接建立

**单机多进程 (MultiprocExecutor)：**
```
父进程启动 N 个子进程
  ↓ 使用 loopback IP (127.0.0.1) + TCPStore
子进程 0..N-1 通过 TCPStore 交换 NCCL UniqueID
  ↓ ncclCommInitRank 建立 NCCL 通信器
  ↓ 共享内存 MessageQueue 用于 SchedulerOutput 广播
```

**多机部署 (Ray)：**
```
Node 0 (Driver):
  ray start --head --port=6379
  RayDistributedExecutor 创建 Ray Actor
  ↓ Actor 排序: Driver 节点优先，然后按 IP 排序
  ↓ 计算 distributed_init_method = driver_ip:open_port

Node 1..N:
  ray start --address=node0:6379
  Ray Actor 自动调度到对应 GPU
  ↓ torch.distributed.init_process_group()
  ↓ NCCL UniqueID 通过 ProcessGroup broadcast 交换
```

**KV Transfer 发现 (NIXL)：**
```
Decode 节点:
  ZMQ REQ → Prefill 节点的 side_channel (host:port)
  ↓ 发送 GET_META_MSG 请求

Prefill 节点:
  ZMQ REP → 返回 NixlHandshakePayload
  ↓ 包含: agent_metadata, KV cache 地址, 兼容性哈希

Decode 节点:
  验证兼容性哈希
  ↓ nixl_wrapper.add_remote_agent() 注册远程代理
  ↓ 创建目标描述符列表
```

### 1.9 通信后端总结

| 通信模式 | 后端 | 用途 |
|----------|------|------|
| **TP AllReduce** | NCCL, CustomAllreduce, QuickReduce, FlashInfer | 层内权重拆分 |
| **PP Send/Recv** | NCCL (PyNCCL) | 层间中间张量 |
| **DP 同步** | Gloo (元数据), NCCL (可选) | Wave 同步 |
| **EP All-to-All** | DeepEP, FlashInfer, Mori, NIXL | 专家路由 |
| **KV Transfer** | NIXL (RDMA), NCCL (P2P), Mooncake | P/D 分离 |
| **Worker RPC** | 共享内存 MessageQueue + ZMQ | 调度器输出广播 |
| **元数据交换** | ZMQ, TCPStore, Gloo | 控制平面 |

---

## 2. Prefill 与 Decode 节点之间的通信

### 2.1 P/D 分离架构

```
┌─────────────────────────────────────────────────────────────────┐
│                          Proxy (代理)                            │
│         路由请求到 P 节点，转发 KV 参数到 D 节点                   │
└────────────┬──────────────────────────────────┬─────────────────┘
             │                                  │
             ▼                                  ▼
┌────────────────────────────┐  ┌─────────────────────────────────┐
│     Prefill (P) 节点        │  │      Decode (D) 节点             │
│  ┌───────────────────────┐ │  │  ┌────────────────────────────┐ │
│  │ KV Producer           │ │  │  │ KV Consumer                │ │
│  │ 计算 prompt KV Cache   │ │  │  │ 接收 KV Cache              │ │
│  │ 通过 KVConnector 传输  │ │  │  │ 执行自回归解码              │ │
│  └───────────────────────┘ │  │  └────────────────────────────┘ │
└────────────────────────────┘  └─────────────────────────────────┘
             │                                  │
             └──────── KV 传输 (RDMA/NCCL) ─────┘
```

### 2.2 KVConnectorBase_V1 接口

所有 KV 传输连接器的统一抽象：

```python
# vllm/distributed/kv_transfer/kv_connector/v1/base.py
class KVConnectorBase_V1:
    # === 调度器侧 (决策) ===

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        """返回可以从远程加载的 token 数"""
        return (num_tokens, is_async)

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        """块分配后记录传输信息"""

    def build_connector_meta(self, scheduler_output):
        """构建传递给 Worker 的传输元数据"""

    def request_finished(self, request, block_ids):
        """请求完成时，返回 kv_transfer_params 给代理"""

    # === Worker 侧 (数据传输) ===

    def register_kv_caches(self, kv_caches):
        """注册 GPU KV Cache 内存区域 (RDMA 需要)"""

    def start_load_kv(self, forward_context):
        """开始异步 KV 加载"""

    def wait_for_layer_load(self, layer_name):
        """等待特定层加载完成"""

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata):
        """开始异步 KV 保存"""

    def wait_for_save(self):
        """等待所有保存完成"""
```

### 2.3 已注册的 KV 连接器

| 连接器 | 传输方式 | 特点 |
|--------|----------|------|
| **NixlConnector** | RDMA (UCX/RoCE/IB) | 生产推荐，零拷贝，异步 |
| **P2pNcclConnector** | NCCL P2P | 简单，GPU-GPU 直连 |
| **MooncakeConnector** | RDMA (Mooncake) | Mooncake 传输引擎 |
| **LMCacheConnectorV1** | LMCache 引擎 | 分布式 KV 缓存 |
| **OffloadingConnector** | CPU/磁盘 | KV 卸载 |
| **MultiConnector** | 组合多个 | 同时使用多个连接器 |
| **FlexKVConnectorV1** | FlexKV | 灵活 KV 存储 |
| **HF3FSKVConnector** | HF3FS | HuggingFace 3FS |

### 2.4 NIXL 连接器 (生产推荐)

NIXL 是 vLLM 推荐的生产级 KV 传输方案，基于 NVIDIA 的 NIXL 库实现 RDMA 传输。

#### 2.4.1 内存注册

```python
# NixlConnectorWorker.register_kv_caches()
def register_kv_caches(self, kv_caches):
    for layer_name, kv_cache in kv_caches.items():
        # 将每层的 K 和 V 张量注册为 NIXL 区域
        # 支持零拷贝 RDMA 读取
        self.nixl_wrapper.register_memory(kv_cache)
```

#### 2.4.2 握手协议

```
Decode 节点                          Prefill 节点
    │                                    │
    │──── ZMQ REQ: GET_META_MSG ────────→│
    │                                    │
    │←── ZMQ REP: NixlHandshakePayload ──│
    │     (agent_metadata, KV 地址,       │
    │      兼容性哈希)                     │
    │                                    │
    │  验证兼容性哈希                      │
    │  nixl_wrapper.add_remote_agent()    │
    │  创建目标描述符列表                   │
    │                                    │
    │──── ZMQ REQ: GET_META_MSG ────────→│ (后续 TP rank)
    │     ...                             │
```

**兼容性哈希包含：** 模型名称、dtype、注意力后端、块大小、KV Cache 布局等。

#### 2.4.3 RDMA 传输流程

```
Prefill 节点                          Decode 节点
    │                                    │
    │  计算 KV Cache                      │
    │  分配 KV 块                         │
    │                                    │
    │←────── NIXL RDMA READ ─────────────│
    │     (零拷贝，直接读取 P 的 GPU 内存)  │
    │                                    │
    │  [传输进行中...]                     │
    │                                    │
    │←────── NIXL 通知 (完成) ────────────│
    │                                    │
    │  释放 KV 块                          │
    │                                    │
    │                               解码生成 token
```

**核心代码：**
```python
# NixlConnectorWorker._read_blocks()
def _read_blocks(self, req_id, local_blocks, remote_blocks):
    # 计算本地和远程描述符 ID
    local_desc_ids = self._get_desc_ids(local_blocks)
    remote_desc_ids = self._get_remote_desc_ids(remote_blocks)

    # 准备 RDMA 传输
    handle = self.nixl_wrapper.make_prepped_xfer(
        "READ",           # 读取模式
        local_desc_ids,   # 本地目标
        remote_desc_ids   # 远程源
    )

    # 启动异步 RDMA 传输
    self.nixl_wrapper.transfer(handle)

    # 存储 handle 用于后续检查完成状态
    self._recving_transfers[req_id].append(handle)
```

#### 2.4.4 KV 块租约机制

NIXL 实现了基于租约的块生命周期管理：

```
时间线:
t=0    P 完成 prefill，KV 块分配
t=0    P 设置租约: kv_lease_duration = 30s
t=5    D 发送心跳，续租到 t=35
t=10   D 发送心跳，续租到 t=40
t=15   D 完成 RDMA 读取
t=15   D 发送 NIXL 通知给 P
t=15   P 释放 KV 块

如果 D 超时未读取:
t=30   租约过期，P 释放 KV 块 (打印警告)
```

**心跳机制：**
```python
# NixlConnectorWorker._send_heartbeats()
def _send_heartbeats(self):
    # 每 kv_lease_duration / 6 秒发送一次 (默认 5 秒)
    for engine_id in self._remote_agents:
        self.nixl_wrapper.send_notif(engine_id, heartbeat_msg)
```

#### 2.4.5 异构 TP 支持

NIXL 支持 P 和 D 使用不同的 Tensor Parallelism 大小：

```
场景 1: D_TP > P_TP (D 的 TP 大于 P)
  P: TP=2, 每个 rank 有 1/2 的 KV heads
  D: TP=4, 每个 rank 需要 1/4 的 KV heads

  D rank 0: 从 P rank 0 读取 KV heads [0:32]
  D rank 1: 从 P rank 0 读取 KV heads [32:64]
  D rank 2: 从 P rank 1 读取 KV heads [0:32]
  D rank 3: 从 P rank 1 读取 KV heads [32:64]

场景 2: P_TP > D_TP (P 的 TP 大于 D)
  P: TP=4, 每个 rank 有 1/4 的 KV heads
  D: TP=2, 每个 rank 需要 1/2 的 KV heads

  D rank 0: 从 P rank 0 和 P rank 1 分别读取，合并
  D rank 1: 从 P rank 2 和 P rank 3 分别读取，合并

MLA 模型 (DeepSeek):
  KV Cache 是复制的 (不是拆分的)
  无论 TP 比例如何，只需读取一次
```

### 2.5 P2pNccl 连接器 (简单方案)

使用 NCCL 点对点通信，适合原型验证：

```
连接建立:
1. P 创建 ZMQ DEALER → D 的地址
2. P 生成 NCCL UniqueID，通过 ZMQ 发送给 D
3. 双方调用 ncclCommInitRank (P=rank0, D=rank1)

传输模式:
- PUT: P 同步发送 → D 接收
- PUT_ASYNC: P 异步发送 (后台线程) → D 接收
- GET: D 请求 → P 发送

每层每请求独立传输 (不像 NIXL 批量传输)
```

### 2.6 Mooncake 连接器

使用 Mooncake Transfer Engine 实现 RDMA 传输：

```
P-push 模型:
1. D 发送 MooncakeXferMetadata 给 P (包含 D 的 KV 地址和块 ID)
2. P 等待自己的块就绪
3. P 调用 engine.batch_transfer_sync_write() 主动写入 D 的内存
4. P 发送 MooncakeXferResponse 给 D
```

### 2.7 请求路由流程

```
1. 客户端 → Proxy (代理)
2. Proxy → P 节点 (kv_role="kv_producer", max_tokens=1)
3. P 节点:
   - 计算 prompt KV Cache
   - request_finished() 返回 kv_transfer_params:
     {
       "do_remote_decode": true,
       "remote_engine_id": "...",
       "remote_request_id": "...",
       "remote_block_ids": [0, 3, 7, ...],
       "remote_host": "p-node-ip",
       "remote_port": 5555,
       "tp_size": 8
     }
4. Proxy 提取 kv_transfer_params
5. Proxy → D 节点 (kv_role="kv_consumer", 附带 kv_transfer_params)
6. D 节点:
   - get_num_new_matched_tokens() 检测 do_remote_prefill=true
   - 报告所有 prompt tokens 可从远程加载
   - allocate_slots() 分配 KV 块
   - Worker 执行 RDMA/NCCL 读取
   - 执行自回归解码
   - 流式返回结果
```

### 2.8 连接器对比

| 特性 | NIXL | P2pNccl | Mooncake |
|------|------|---------|----------|
| 传输方式 | RDMA (UCX/RoCE/IB) | NCCL P2P | RDMA (Mooncake) |
| 内存注册 | ✅ 零拷贝 | ❌ | ✅ |
| 异构 TP | ✅ 完整支持 | ❌ | ✅ |
| 批量传输 | ✅ 块级 | ❌ 逐层逐请求 | ✅ 块级 |
| 异步传输 | ✅ 完全异步 | 部分 (PUT_ASYNC) | ✅ 完全异步 |
| 块大小不匹配 | ✅ | ❌ | ✅ |
| KV 租约/心跳 | ✅ | ❌ | 超时机制 |
| 跨层块 | ✅ | ❌ | ❌ |
| 混合 SSM 模型 | ✅ (Mamba) | ❌ | ❌ |
| 复杂度 | 高 | 低 | 中 |
| 适用场景 | 生产、大规模 | 原型、同 GPU | Mooncake 集群 |

---

## 3. 通过 lws 部署 DeepSeek V4 的多级多卡推理

### 3.1 DeepSeek V4 模型特点

DeepSeek V4 是一个 Mixture-of-Experts (MoE) 模型，具有以下特点：

| 特性 | 值 |
|------|-----|
| 模型类型 | MoE + MLA (Multi-head Latent Attention) |
| 逻辑专家数 | 384 |
| 冗余专家数 | 32 (用于负载均衡) |
| 物理专家数 | 384 + 32 = 416 |
| 每 token 激活专家 | 6 (Top-6 路由) |
| KV Cache 格式 | MLA 压缩表示 (576 维/token) |
| 注意力类型 | MLA + 滑动窗口 (混合) |

### 3.2 LeaderWorkerSet (LWS) 部署架构

LWS 是 Kubernetes 的 API，专为多节点分布式推理设计。

```
┌─────────────────────────────────────────────────────────────────┐
│                    Kubernetes Cluster                            │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              LeaderWorkerSet (LWS)                       │    │
│  │                                                          │    │
│  │  ┌──────────────────────┐  ┌──────────────────────┐     │    │
│  │  │    Leader Pod         │  │    Worker Pod         │     │    │
│  │  │  ┌────────────────┐  │  │  ┌────────────────┐  │     │    │
│  │  │  │ Ray Head Node  │  │  │  │ Ray Worker     │  │     │    │
│  │  │  │ + vllm serve   │  │  │  │ + join Ray     │  │     │    │
│  │  │  │ GPU 0-7        │  │  │  │ GPU 8-15       │  │     │    │
│  │  │  └────────────────┘  │  │  └────────────────┘  │     │    │
│  │  └──────────────────────┘  └──────────────────────┘     │    │
│  │                                                          │    │
│  │  ┌──────────────────────┐                                │    │
│  │  │    Service (ClusterIP)│  ← 只选择 role=leader 的 Pod  │    │
│  │  │    API 端点            │                                │    │
│  │  └──────────────────────┘                                │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

**LWS 资源配置：**
```yaml
apiVersion: leaderworkerset.x-k8s.io/v1
kind: LeaderWorkerSet
metadata:
  name: deepseek-v4
spec:
  replicas: 1
  leaderWorkerTemplate:
    size: 2  # 1 leader + 1 worker
    leaderTemplate:
      containers:
      - name: vllm-leader
        image: vllm/vllm-openai
        command: ["multi-node-serving.sh", "leader"]
        resources:
          limits:
            nvidia.com/gpu: 8
            memory: 1124Gi
        env:
        - name: VLLM_HOST_IP
          valueFrom:
            fieldRef:
              fieldPath: status.podIP
    workerTemplate:
      containers:
      - name: vllm-worker
        image: vllm/vllm-openai
        command: ["multi-node-serving.sh", "worker"]
        resources:
          limits:
            nvidia.com/gpu: 8
            memory: 1124Gi
```

### 3.3 Ray 集群形成

```bash
# multi-node-serving.sh

# Leader Pod:
ray start --head --port=6379
# 等待所有 worker 就绪
while [ $(ray status | grep "Active:" | wc -l) -lt $LWS_GROUP_SIZE ]; do
    sleep 1
done
# 启动 vLLM 服务
vllm serve deepseek-ai/DeepSeek-V4 \
    --tensor-parallel-size 8 \
    --pipeline-parallel-size 2 \
    --distributed-executor-backend ray

# Worker Pod:
# LWS_LEADER_ADDRESS 由 LWS 控制器自动注入
ray start --address=$LWS_LEADER_ADDRESS:6379 --block
```

### 3.4 多级并行策略

DeepSeek V4 的多级多卡推理采用多种并行策略的组合：

```
┌─────────────────────────────────────────────────────────────────┐
│                    DeepSeek V4 多级并行                          │
│                                                                  │
│  Level 1: Data Parallelism (DP=16)                              │
│  ┌────────┐ ┌────────┐ ┌────────┐      ┌────────┐              │
│  │ DP=0   │ │ DP=1   │ │ DP=2   │ ...  │ DP=15  │              │
│  │GPU 0   │ │GPU 1   │ │GPU 2   │      │GPU 15  │              │
│  └────┬───┘ └────┬───┘ └────┬───┘      └────┬───┘              │
│       │          │          │                │                   │
│  Level 2: Expert Parallelism (EP=16 = TP×DP)                    │
│  每个 GPU 持有 416/16 = 26 个物理专家                             │
│  All-to-All 通信: DeepEP low_latency                            │
│                                                                  │
│  Level 3: Tensor Parallelism (TP=1, 每 DP rank 独立)             │
│  注意力层在每个 DP rank 上复制                                     │
│  MLA KV Cache 是压缩表示，无需 TP 拆分                            │
│                                                                  │
│  Level 4: Pipeline Parallelism (PP=1, 不使用)                    │
│  DeepSeek V4 通常不需要 PP                                       │
└─────────────────────────────────────────────────────────────────┘
```

**部署配置示例 (2 节点，16 GPU)：**

```bash
# Node 1 (Primary, 处理 API 请求):
vllm serve deepseek-ai/DeepSeek-V4 \
    --all2all-backend deepep_low_latency \
    --tensor-parallel-size 1 \
    --enable-expert-parallel \
    --data-parallel-size 16 \
    --data-parallel-size-local 8 \
    --data-parallel-address 192.168.1.100 \
    --data-parallel-rpc-port 13345 \
    --api-server-count=8

# Node 2 (Headless, 仅 Worker):
vllm serve deepseek-ai/DeepSeek-V4 \
    --all2all-backend deepep_low_latency \
    --tensor-parallel-size 1 \
    --enable-expert-parallel \
    --data-parallel-size 16 \
    --data-parallel-size-local 8 \
    --data-parallel-start-rank 8 \
    --data-parallel-address 192.168.1.100 \
    --data-parallel-rpc-port 13345 \
    --headless
```

### 3.5 DPEngineCoreProc —— DP 引擎核心

每个 DP rank 运行一个 `DPEngineCoreProc`：

```python
# vllm/v1/engine/core.py, line 1673
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
        """All-reduce 检查所有 DP rank 是否有未完成请求"""
        local_has = self.scheduler.has_requests()
        global_has = all_reduce(local_has, self.dp_group)
        return global_has

    def _process_engine_step(self):
        """处理引擎步骤"""
        if self.scheduler.has_requests():
            self.step_fn()  # 正常执行
        else:
            # 本 rank 无请求，但其他 rank 有
            # 执行 dummy batch (MoE 需要 all-to-all 同步)
            self._execute_dummy_batch()
```

### 3.6 Expert Parallel Load Balancer (EPLB)

MoE 模型的 token 路由通常不均匀，EPLB 动态重分配专家：

```
问题:
  专家 0: 接收 30% 的 token (热点)
  专家 1: 接收 2% 的 token (冷门)
  ...

EPLB 解决方案:
  复制热点专家: 专家 0 → 专家 0 (GPU 0) + 专家 0' (GPU 1)
  每个 GPU 持有 416/32 = 13 个物理专家 (含冗余)

配置:
  --eplb-config '{
    "window_size": 1000,      # 跟踪步数
    "step_interval": 100,     # 重平衡频率
    "num_redundant_experts": 32,
    "use_async": true,
    "policy": "balanced"
  }'

每个冗余专家的内存开销: ~2.4 GB (DeepSeek V3)
```

### 3.7 ElasticEP —— 弹性专家并行

ElasticEP 允许在运行时动态调整 DP 大小，无需重启服务：

```
┌─────────────────────────────────────────────────────────────────┐
│                    ElasticEP 扩展流程                            │
│                                                                  │
│  初始状态: DP=8 (8 个 GPU)                                       │
│                                                                  │
│  收到 POST /scale_elastic_ep {"new_data_parallel_size": 16}     │
│                                                                  │
│  1. ScalingMiddleware 返回 503 (拒绝新请求)                       │
│  2. 向所有现有引擎发送 ReconfigureDistributedRequest              │
│  3. 创建 8 个新 Ray Actor                                        │
│                                                                  │
│  现有引擎状态机:                                                  │
│  WAIT_NEW_CORE_ENGINES_INIT                                      │
│    → CREATE_STANDBY_GROUPS                                       │
│    → TRANSFER_EXPERT_MAPPING                                     │
│    → WAIT_NEW_CORE_ENGINES_WEIGHTS_INIT                          │
│    → TRANSFER_WEIGHTS (P2P 传输非专家权重)                        │
│    → SYNC_KV_CACHE_MEMORY_SIZE                                   │
│    → SWITCH_AND_PREPARE                                          │
│    → EPLB_RESHUFFLE (重分配专家)                                  │
│    → COMPLETE                                                    │
│                                                                  │
│  新引擎状态机:                                                    │
│  PRE_KV_INIT → PREPARE → EPLB_RESHUFFLE → COMPLETE              │
│                                                                  │
│  最终状态: DP=16 (16 个 GPU)                                     │
│  ScalingMiddleware 恢复正常请求处理                                │
└─────────────────────────────────────────────────────────────────┘
```

**API 端点：**
```bash
# 扩容
curl -X POST http://localhost:8000/scale_elastic_ep \
    -H "Content-Type: application/json" \
    -d '{"new_data_parallel_size": 16}'

# 缩容
curl -X POST http://localhost:8000/scale_elastic_ep \
    -H "Content-Type: application/json" \
    -d '{"new_data_parallel_size": 8}'
```

### 3.8 完整部署栈

```
┌─────────────────────────────────────────────────────────────────┐
│  Layer 1: Kubernetes + LWS                                      │
│  编排 Pod (leader + workers)，管理网络，注入环境变量               │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│  Layer 2: Ray Cluster                                           │
│  multi-node-serving.sh 形成集群，提供分布式任务调度                │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│  Layer 3: Data Parallelism (DP=16)                              │
│  每个 DP rank 是独立的 EngineCoreProc                            │
│  DPCoordinator 同步 wave 状态和负载统计                           │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────┬──────────────────────────────────────┐
│  Layer 4a: Expert Parallelism (EP=16)                           │
│  MoE 专家分布在所有 GPU 上                                       │
│  All-to-All 通信: DeepEP low_latency                            │
├─────────────────────────────────────────────────────────────────┤
│  Layer 4b: Tensor Parallelism (TP=1)                            │
│  注意力层在每个 DP rank 上复制 (MLA 无需 TP 拆分)                 │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│  Layer 5: EPLB + ElasticEP                                      │
│  EPLB: 动态重分配热点专家                                        │
│  ElasticEP: 运行时增减 DP rank                                   │
└─────────────────────────────────────────────────────────────────┘
```

### 3.9 关键配置参数总结

| 参数 | 说明 | DeepSeek V4 推荐值 |
|------|------|-------------------|
| `--tensor-parallel-size` | 张量并行度 | 1 (MLA 无需 TP) |
| `--data-parallel-size` | 数据并行度 | 8-32 (根据 GPU 数) |
| `--data-parallel-size-local` | 本节点 DP 数 | 每节点 GPU 数 |
| `--enable-expert-parallel` | 启用专家并行 | true |
| `--all2all-backend` | EP 通信后端 | deepep_low_latency |
| `--data-parallel-address` | DP 协调器地址 | 主节点 IP |
| `--data-parallel-rpc-port` | DP RPC 端口 | 13345 |
| `--api-server-count` | API 服务器数 | = 本节点 DP 数 |
| `--headless` | 无头模式 (Worker 节点) | Node 2+ 设为 true |
| `--enable-eplb` | 启用 EPLB | true |
| `--eplb-config` | EPLB 配置 | 见上文 |

### 3.10 通信开销分析

| 通信类型 | 频率 | 数据量 | 后端 |
|----------|------|--------|------|
| EP All-to-All | 每层每步 | tokens × hidden_dim | DeepEP |
| DP Wave 同步 | 每 32 步 | 1 bit (all-reduce) | NCCL |
| KV Transfer (P→D) | 每请求一次 | KV Cache 大小 | RDMA |
| 权重传输 (ElasticEP) | 扩容时 | 模型大小 | NCCL P2P |
| 专家重分配 (EPLB) | 重平衡时 | 专家大小 | NCCL |

---

## 附录

### A. 关键文件索引

| 文件 | 内容 |
|------|------|
| `vllm/distributed/parallel_state.py` | GroupCoordinator, 并行状态管理 |
| `vllm/distributed/device_communicators/` | NCCL, CustomAllreduce, All2All 等 |
| `vllm/distributed/kv_transfer/` | KV 传输框架和连接器 |
| `vllm/v1/engine/core.py` | EngineCore, DPEngineCoreProc |
| `vllm/v1/engine/coordinator.py` | DPCoordinator |
| `vllm/v1/executor/ray_executor.py` | Ray 分布式执行器 |
| `vllm/v1/executor/multiproc_executor.py` | 多进程执行器 |
| `vllm/config/kv_transfer.py` | KVTransferConfig |
| `vllm/config/parallel.py` | ParallelConfig |
| `vllm/distributed/eplb/` | Expert Parallel Load Balancer |
| `vllm/distributed/elastic_ep/` | ElasticEP |
| `docs/deployment/frameworks/lws.md` | LWS 部署文档 |
| `docs/serving/expert_parallel_deployment.md` | EP 部署文档 |
| `docs/serving/data_parallel_deployment.md` | DP 部署文档 |
| `examples/ray_serving/` | Ray 部署示例 |
| `examples/disaggregated/` | P/D 分离示例 |

### B. 术语表

| 术语 | 说明 |
|------|------|
| **LWS** | LeaderWorkerSet, Kubernetes 多节点部署 API |
| **TP** | Tensor Parallelism, 张量并行 |
| **PP** | Pipeline Parallelism, 流水线并行 |
| **DP** | Data Parallelism, 数据并行 |
| **EP** | Expert Parallelism, 专家并行 |
| **EPLB** | Expert Parallel Load Balancer, 专家并行负载均衡 |
| **ElasticEP** | 弹性专家并行，运行时增减 DP rank |
| **P/D** | Prefill/Decode, 预填充/解码分离 |
| **NIXL** | NVIDIA Interconnect Library, RDMA 传输库 |
| **RDMA** | Remote Direct Memory Access, 远程直接内存访问 |
| **NCCL** | NVIDIA Collective Communications Library |
| **MLA** | Multi-head Latent Attention, DeepSeek 的注意力机制 |
| **MoE** | Mixture of Experts, 混合专家模型 |
| **DPCoordinator** | 数据并行协调器 |
| **Wave** | DP 引擎的运行/暂停同步机制 |
