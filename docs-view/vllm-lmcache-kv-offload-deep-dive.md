# vLLM + LMCache：KV Cache 卸载与 PD 分离深度剖析

## 概述

本文档详细剖析 vLLM 与 LMCache 集成时的 KV Cache 管理机制，重点解答三个核心问题：

1. **什么时候** vLLM 会把 HBM 上的 KV Cache offload 到 DRAM？
2. **什么场景下**、**什么形式**的 offload 会发生？
3. **什么时候** vLLM 会使用 LMCache 侧的 KV Cache？

---

## 一、整体架构

### 1.1 KV Connector 双进程架构

vLLM 的 KV Connector 采用 **Scheduler-Worker 双进程架构**：

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Scheduler Process（调度器进程）                                          │
│                                                                         │
│  每个 engine step:                                                       │
│  1. get_num_new_matched_tokens(req) ──→ 查询 LMCache 有多少 token 命中   │
│  2. alloc_blocks(num_hit_tokens)    ──→ 为命中的 token 分配 GPU Block     │
│  3. build_connector_meta()          ──→ 构建 load/save 元数据             │
│  4. update_connector_output()       ──→ 收集 Worker 的完成信号            │
│                                                                         │
│  元数据通过 scheduler_output.kv_connector_metadata 传递给 Worker          │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │ scheduler_output
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Worker Process（Worker 进程，每个 GPU 一个）                             │
│                                                                         │
│  每个 engine step:                                                       │
│  1. bind_connector_metadata()   ──→ 接收 Scheduler 的元数据              │
│  2. start_load_kv()             ──→ 【模型 forward 之前】启动异步加载      │
│  3. ──── 模型 forward 执行 ────                                         │
│  │   ├─ save_kv_layer(layer_0) ──→ 【每层 Attention】异步保存该层 KV      │
│  │   ├─ wait_for_layer_load(layer_0) ──→ 等待该层加载完成                 │
│  │   ├─ save_kv_layer(layer_1)                                        │
│  │   ├─ wait_for_layer_load(layer_1)                                  │
│  │   └─ ...                                                           │
│  4. wait_for_save()             ──→ 【模型 forward 之后】等待所有保存完成   │
│  5. get_finished()              ──→ 报告完成的异步传输                     │
│  6. clear_connector_metadata()  ──→ 清除元数据                           │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.2 关键代码入口

**`gpu_model_runner.py` 中的调用点**:

```python
# gpu_model_runner.py:4291
with set_forward_context(attn_metadata, self.vllm_config, ...):
    # maybe_get_kv_connector_output 是一个 context manager
    # 它在 forward 之前调用 start_load_kv()
    # 在 forward 之后调用 wait_for_save() 和 get_finished()
    with self.maybe_get_kv_connector_output(
        scheduler_output,
        defer_finalize=defer_kv_connector_finalize,
    ) as kv_connector_output:
        model_output = self._model_forward(...)
```

**`kv_connector.py` 中的 `ActiveKVConnector`**:

```python
# kv_connector.py:125
def pre_forward(self, scheduler_output):
    """模型 forward 之前调用"""
    self.kv_connector.handle_preemptions(kv_connector_metadata)
    self.kv_connector.bind_connector_metadata(kv_connector_metadata)
    self.kv_connector.start_load_kv(get_forward_context())  # 启动异步加载

# kv_connector.py:151
def post_forward(self, finished_req_ids, wait_for_save=True):
    """模型 forward 之后调用"""
    self.kv_connector.wait_for_save()          # 等待保存完成
    output.finished_sending, output.finished_recving = (
        self.kv_connector.get_finished(finished_req_ids)  # 收集完成信号
    )
```

---

## 二、LMCache Connector 类型与适用场景

### 2.1 三种 LMCache Connector

| Connector | 文件 | 传输方式 | 适用场景 |
|-----------|------|----------|----------|
| `LMCacheConnectorV1` | `lmcache_connector.py` | LMCache Engine（进程内） | 单机 PD 混部、PD 分离 |
| `LMCacheMPConnector` | `lmcache_mp_connector.py` | ZMQ + CUDA IPC（多进程） | 多进程 LMCache Server |
| `OffloadingConnector` | `offloading_connector.py` | 通用卸载框架 | CPU/磁盘卸载 |
| `SimpleCPUOffloadConnector` | `simple_cpu_offload_connector.py` | DMA 拷贝 | 简单 HBM→DRAM 卸载 |

### 2.2 LMCacheConnectorV1 的两种子模式

```python
# lmcache_connector.py:96
if use_native:
    # 使用 vLLM 内置的 LMCache 实现
    from vllm.distributed.kv_transfer.kv_connector.v1.lmcache_integration.vllm_v1_adapter
    self._lmcache_engine = LMCacheConnectorV1Impl(...)
else:
    # 使用 pip 安装的 lmcache 包的实现
    from lmcache.integration.vllm.vllm_v1_adapter
    self._lmcache_engine = LMCacheConnectorV1Impl(...)
```

---

## 三、HBM → DRAM Offload：什么时候、什么场景

### 3.1 场景一：PD 混部（同一实例既做 Prefill 又做 Decode）

**配置**:
```yaml
# lmcache-config.yaml
enable_nixl: False  # 不启用远程传输
local_cpu: True     # 启用本地 CPU 缓存
max_local_cpu_size: 10  # 10GB CPU 内存
```

**Offload 触发时机**:

#### (1) Prefill 阶段结束后 offload

当一个请求完成 prefill 后，其 KV Cache 会被 offload 到 DRAM：

```python
# vllm_v1_adapter.py:304-318 (ReqMeta.from_request_tracker)
# 判断是否需要保存
skip_leading_tokens = tracker.num_saved_tokens
chunk_boundary = (
    cdiv(tracker.num_saved_tokens + 1, lmcache_chunk_size) * lmcache_chunk_size
)

# 跳过保存的条件（满足任一则跳过）：
# 1. 已经保存过（num_saved_tokens > 0）且未达到 chunk 边界
# 2. 在 decode 阶段且 save_decode_cache=False
skip_save = (
    tracker.num_saved_tokens > 0
    and input_token_len < chunk_boundary
) or (tracker.is_decode_phase and not save_decode_cache)
```

**触发条件**:
- `is_last_prefill = True`（prefill 最后一个 chunk）
- token 数量达到 `lmcache_chunk_size` 的整数倍（默认 256）
- `skip_save = False`

#### (2) Decode 阶段的增量 offload

如果配置了 `save_decode_cache=True`，每个 decode step 都会 offload 新生成的 KV：

```python
# vllm_v1_adapter.py:316
# decode 阶段且 save_decode_cache=True 时，不跳过保存
(tracker.is_decode_phase and not save_decode_cache)
```

#### (3) Layerwise 保存流程

```python
# vllm_v1_adapter.py:931 (save_kv_layer)
def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
    if not self.use_layerwise:
        return  # 非 layerwise 模式，在 wait_for_save 中批量保存

    if self.kv_role == "kv_consumer":
        return  # PD 分离的 Decode 端不保存

    if self.current_layer == 0:
        # 第一层时创建 layerwise storers（生成器）
        self.layerwise_storers = []
        for request in connector_metadata.requests:
            if save_spec is None or not save_spec.can_save:
                continue
            # 创建逐层存储的生成器
            storer = self.lmcache_engine.store_layer(
                token_ids, mask=store_mask, kvcaches=kvcaches,
                slot_mapping=slot_mapping, offset=skip_leading_tokens,
            )
            next(storer)  # 启动生成器
            self.layerwise_storers.append(storer)
    else:
        # 后续层：推进生成器，触发当前层的实际数据传输
        for storer in self.layerwise_storers:
            next(storer)  # 这里会触发当前层的 GPU→CPU 拷贝

    self.current_layer += 1
```

**关键点**: Layerwise 模式下，每层 Attention 执行完后，立即将该层的 KV Cache 从 GPU HBM 拷贝到 CPU DRAM。这与模型计算是**流水线重叠**的。

#### (4) Bulk 保存流程

```python
# vllm_v1_adapter.py:1033 (wait_for_save)
def wait_for_save(self):
    if self.use_layerwise:
        return  # layerwise 模式已在 save_kv_layer 中完成

    # bulk 模式：在 forward 结束后一次性保存所有层
    for request in connector_metadata.requests:
        if save_spec is None or not save_spec.can_save:
            continue
        self.lmcache_engine.store(
            token_ids, mask=store_mask, kvcaches=kvcaches,
            slot_mapping=slot_mapping, offset=skip_leading_tokens,
        )
```

### 3.2 场景二：PD 分离（Prefill 和 Decode 在不同实例）

**配置**:
```yaml
# Prefill 端配置
enable_nixl: True
nixl_role: "sender"
nixl_peer_host: "decode_node_ip"
nixl_peer_port: 55555

# Decode 端配置
enable_nixl: True
nixl_role: "receiver"
nixl_peer_host: "prefill_node_ip"
nixl_peer_port: 55555
```

**Offload 触发时机**:

PD 分离时，Prefill 端的 "offload" 实际上是**跨节点传输**到 Decode 端的 GPU：

```python
# vllm_v1_adapter.py:89-97 (DisaggSpec)
@dataclass
class DisaggSpec:
    req_id: str
    receiver_id: str          # Decode 端的 ID
    receiver_host: str        # Decode 端的 IP
    receiver_init_port: int   # 初始化端口
    receiver_alloc_port: int  # 分配端口
    is_last_prefill: bool = False  # 是否是最后一个 prefill chunk
    num_transferred_tokens: int = 0  # 已传输的 token 数
```

**PD 分离的保存逻辑**:

```python
# vllm_v1_adapter.py:1115
self.lmcache_engine.store(
    token_ids,
    mask=store_mask,
    kvcaches=kvcaches,
    slot_mapping=slot_mapping,
    offset=skip_leading_tokens,
    transfer_spec=request.disagg_spec,  # ← 关键：指定远程接收方
    request_configs=request.request_configs,
)
```

**PD 分离 vs PD 混部的区别**:

| 特性 | PD 混部 | PD 分离 |
|------|---------|---------|
| offload 目标 | 本地 CPU DRAM | 远程 Decode 端 GPU |
| 触发时机 | prefill 结束 / decode 每步 | prefill 结束（is_last_prefill=True） |
| 数据流向 | GPU HBM → CPU DRAM | GPU HBM → 网络 → 远程 GPU HBM |
| GPU 临时缓冲区 | 需要（`need_gpu_interim_buffer=True`） | 不需要（`enable_pd=True`） |
| Decode 端保存 | 保存（本地缓存） | 不保存（`kv_role="kv_consumer"`） |

```python
# vllm_v1_adapter.py:401
def need_gpu_interim_buffer(lmcache_config):
    # PD 分离时不需要 GPU 临时缓冲区，因为数据直接传输到远程
    return not lmcache_config.enable_pd
```

### 3.3 场景三：SimpleCPUOffload（纯 HBM→DRAM 卸载）

**配置**:
```json
{
    "kv_connector": "SimpleCPUOffloadConnector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
        "cpu_bytes_to_use": "8589934592"
    }
}
```

**Offload 触发时机**:

SimpleCPUOffload 采用**完全延迟**设计——所有 HBM↔DRAM 传输都在 forward 之外执行：

```python
# simple_cpu_offload_connector.py:143-159
def start_load_kv(self, forward_context, **kwargs):
    pass  # 不在 forward 前加载，延迟到 get_finished()

def wait_for_layer_load(self, layer_name):
    pass  # 始终异步，不需要等待

def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
    pass  # 不在 forward 中保存，延迟到 get_finished()

def wait_for_save(self):
    pass  # 所有传输由 get_finished() 驱动

def get_finished(self, finished_req_ids):
    # 实际的 HBM↔DRAM 传输在这里发生！
    return self.worker_handler.get_finished(finished_req_ids)
```

**底层实现**:

```python
# simple_kv_offload/worker.py:33
class SimpleCPUOffloadWorker:
    def __init__(self, ...):
        # 分配 pinned CPU 内存（避免 PyTorch 的 2 次幂对齐浪费）
        # 使用 cudaHostRegister 直接注册
        self.cpu_kv_caches = ...  # 与 GPU KV Cache 形状相同的 CPU 张量

        # 两个独立的 CUDA Stream
        self.load_stream = torch.cuda.Stream()   # CPU → GPU
        self.store_stream = torch.cuda.Stream()  # GPU → CPU

        # DMA 拷贝后端（后台线程执行批量内存拷贝）
        self._backend = DmaCopyBackend()
```

**传输时序**:
```
Engine Step N:
  1. bind_metadata()        ← 接收 load/store 映射
  2. [模型 forward 执行]     ← 不做任何传输
  3. get_finished():
     ├─ 提交 store 作业: GPU HBM → CPU DRAM（在 store_stream 上异步执行）
     ├─ 提交 load 作业:  CPU DRAM → GPU HBM（在 load_stream 上异步执行）
     └─ 通过 CUDA Event 跟踪完成状态

Engine Step N+1:
  1. bind_metadata()        ← 接收新的 load/store 映射
  2. [模型 forward 执行]     ← Step N 的传输可能仍在进行
  3. get_finished():
     ├─ 检查 Step N 的传输是否完成
     └─ 提交新的传输
```

### 3.4 场景四：OffloadingConnector（通用卸载框架）

**Offload 触发时机**:

OffloadingConnector 的 store 操作被**延迟到下一个 engine step**：

```python
# offloading/worker.py:252
def prepare_store_kv(self, metadata):
    for job_id, entry in metadata.store_jobs.items():
        # 延迟到下一个 step 的 start_kv_transfers()
        # 目的：offloading 在 token 采样相关传输之后才开始，
        # 避免延迟 token 生成
        self._unsubmitted_store_jobs.append((job_id, entry.transfer_spec))
```

**时序**:
```
Step N:
  1. start_kv_transfers():
     ├─ 提交 Step N-1 延迟的 store 作业（GPU → CPU/磁盘）
     └─ 提提交 load 作业（CPU/磁盘 → GPU）
  2. [模型 forward]
  3. prepare_store_kv(): 将 Step N 的 store 延迟

Step N+1:
  1. start_kv_transfers():
     ├─ 提交 Step N 延迟的 store 作业
     └─ ...
```

---

## 四、LMCache KV Cache 的使用时机

### 4.1 Lookup 阶段（Scheduler 进程）

**什么时候查询 LMCache？**

在 Scheduler 处理每个新请求时：

```python
# vllm_v1_adapter.py:1141
def get_num_new_matched_tokens(self, request, num_computed_tokens):
    # 1. 提取 token IDs
    token_ids = request.prompt_token_ids

    # 2. 处理多模态输入的哈希
    mm_hashes, mm_positions = extract_mm_features(request)
    if mm_hashes and mm_positions:
        apply_mm_hashes_to_token_ids(token_ids, mm_hashes, mm_positions)

    # 3. 查询 LMCache
    num_external_hit_tokens = self.lookup_client.lookup(
        token_ids, request_id=request.request_id
    )

    # 4. 返回可从外部加载的 token 数量
    #    = 外部命中数 - 本地已计算数
    return num_external_hit_tokens - num_computed_tokens
```

**Lookup 的判断逻辑**:
```
请求 token: [A, B, C, D, E, F, G, H]  (8 个 token)
vLLM 本地 prefix cache 命中: [A, B, C]  (3 个 token)
LMCache 查询结果: [A, B, C, D, E, F]  (6 个 token)

可从 LMCache 加载的 token 数 = 6 - 3 = 3 个 (D, E, F)
```

### 4.2 Load 阶段（Worker 进程）

**什么时候从 LMCache 加载 KV Cache？**

#### (1) Layerwise 加载

```python
# vllm_v1_adapter.py:798
def start_load_kv(self, forward_context, **kwargs):
    """在模型 forward 之前调用"""
    self.layerwise_retrievers = []

    for request in metadata.requests:
        if request.load_spec is None:
            continue

        # 创建 token mask：排除 vLLM 本地已缓存的部分
        token_mask = torch.ones(len(tokens), dtype=torch.bool)
        masked_token_count = (
            request.load_spec.vllm_cached_tokens
            // self._lmcache_chunk_size
            * self._lmcache_chunk_size
        )
        token_mask[:masked_token_count] = False  # 本地已有的不从 LMCache 加载

        if self.use_layerwise:
            # 创建逐层检索的生成器
            retriever = self.lmcache_engine.retrieve_layer(
                tokens[:lmcache_cached_tokens],
                token_mask[:lmcache_cached_tokens],
                kvcaches=kvcaches,
                slot_mapping=slot_mapping[:lmcache_cached_tokens],
            )
            next(retriever)  # 启动生成器，开始加载第 0 层
            next(retriever)  # 预加载第 1 层
            self.layerwise_retrievers.append(retriever)
```

```python
# vllm_v1_adapter.py:908
def wait_for_layer_load(self, layer_name):
    """在每层 Attention 执行前调用"""
    for retriever in self.layerwise_retrievers:
        next(retriever)  # 推进生成器，等待当前层加载完成
    self.current_layer += 1
```

#### (2) Bulk 加载

```python
# vllm_v1_adapter.py:882
else:  # bulk 模式
    self.lmcache_engine.retrieve(
        tokens[:lmcache_cached_tokens],
        token_mask[:lmcache_cached_tokens],
        kvcaches=kvcaches,
        slot_mapping=slot_mapping[:lmcache_cached_tokens],
    )
```

### 4.3 PD 分离场景下的加载

PD 分离时，Decode 端的加载流程：

```
Decode 端 Engine Step:
  1. get_num_new_matched_tokens():
     ├─ 查询 LMCache: "这个请求的 KV Cache 在 LMCache 中吗？"
     ├─ LMCache 通过 NIXL 从 Prefill 端拉取（或等待推送）
     └─ 返回命中的 token 数

  2. alloc_blocks(): 为命中的 token 分配 GPU Block

  3. build_connector_meta(): 构建 load_spec

  4. start_load_kv():
     └─ 从 LMCache（实际来自 Prefill 端）加载 KV Cache 到 GPU

  5. 模型 forward:
     └─ 使用已加载的 KV Cache，只计算新 token
```

### 4.4 完整时序图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    PD 混部 + LMCache 完整时序                             │
│                                                                         │
│  请求到达: "Hello, my name is" (6 tokens)                               │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Scheduler Process                                                │   │
│  │                                                                  │   │
│  │  1. get_num_new_matched_tokens("Hello, my name is")              │   │
│  │     └─ LMCache.lookup() → 命中 4 tokens ("Hello, my name")      │   │
│  │     └─ 返回 (4, False)                                           │   │
│  │                                                                  │   │
│  │  2. alloc_blocks(4) → 分配 4 个 GPU Block                        │   │
│  │                                                                  │   │
│  │  3. build_connector_meta()                                       │   │
│  │     └─ load_spec: {lmcache_cached_tokens: 4, can_load: True}    │   │
│  │     └─ save_spec: {can_save: True}                               │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                 │                                       │
│                                 ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Worker Process                                                   │   │
│  │                                                                  │   │
│  │  4. bind_connector_metadata()                                    │   │
│  │                                                                  │   │
│  │  5. start_load_kv()                                              │   │
│  │     └─ LMCache.retrieve(tokens[:4], kvcaches, slot_mapping)      │   │
│  │     └─ 从 DRAM 加载 4 个 token 的 KV Cache 到 GPU HBM            │   │
│  │                                                                  │   │
│  │  6. 模型 forward (layerwise):                                     │   │
│  │     ├─ wait_for_layer_load(layer_0) → 等待第 0 层加载完成          │   │
│  │     ├─ Attention(layer_0) → 使用已加载的 KV + 计算新 token         │   │
│  │     ├─ save_kv_layer(layer_0) → 将第 0 层新 token 的 KV 保存到 DRAM│   │
│  │     ├─ wait_for_layer_load(layer_1)                              │   │
│  │     ├─ Attention(layer_1)                                        │   │
│  │     ├─ save_kv_layer(layer_1)                                    │   │
│  │     └─ ...                                                       │   │
│  │                                                                  │   │
│  │  7. wait_for_save() → 等待所有层保存完成                           │   │
│  │                                                                  │   │
│  │  8. get_finished() → 报告完成状态                                 │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  结果: 利用 LMCache 中的 4 个 token KV Cache，只计算了 2 个新 token      │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 五、Chunk 对齐机制

### 5.1 为什么需要 Chunk 对齐？

LMCache 以 **chunk** 为单位管理 KV Cache（默认 256 tokens）。这带来两个约束：

1. **保存时**: 只有达到 chunk 边界才保存（避免碎片化）
2. **加载时**: 只能加载整数个 chunk

```python
# vllm_v1_adapter.py:305-306
chunk_boundary = (
    cdiv(tracker.num_saved_tokens + 1, lmcache_chunk_size) * lmcache_chunk_size
)
```

### 5.2 Chunk 对齐示例

```
lmcache_chunk_size = 256

请求 token 数: 600

第 1 次 prefill (tokens 0-255):
  - 保存: tokens 0-255 (256 tokens, 恰好 1 个 chunk) ✓

第 2 次 prefill (tokens 256-511):
  - 保存: tokens 256-511 (256 tokens, 恰好 1 个 chunk) ✓

第 3 次 prefill (tokens 512-599):
  - 保存: tokens 512-599 (88 tokens, 不足 1 个 chunk)
  - 如果 is_last_prefill=True: 保存（最后一个 chunk 允许不满）
  - 如果 is_last_prefill=False: 不保存（等待凑满）
```

---

## 六、多进程模式（LMCacheMPConnector）

### 6.1 架构

```
┌──────────────┐     ZMQ      ┌──────────────────┐     CUDA IPC     ┌──────────┐
│  vLLM Worker │◄────────────►│  LMCache Server   │◄───────────────►│  GPU HBM │
│  Process     │              │  Process          │                  │          │
└──────────────┘              └──────────────────┘                  └──────────┘
```

### 6.2 KV Cache 注册

```python
# multi_process_adapter.py:413
def register_kv_caches(self, kv_caches):
    # 将 GPU KV Cache 的 CUDA IPC handle 发送给 LMCache Server
    for layer_name, kv_cache in kv_caches.items():
        ipc_handle = torch.multiprocessing.reductions.reduce_tensor(kv_cache)
        self.zmq_socket.send(ipc_handle)
```

### 6.3 Store/Retrieve 操作

```python
# multi_process_adapter.py:500
def batched_submit_store_requests(self, store_entries):
    # 1. 在当前 CUDA Stream 上记录事件
    event = torch.cuda.Event(interprocess=True)
    event.record()

    # 2. 通过 ZMQ 发送 store 请求（包含 CUDA IPC event handle）
    for entry in store_entries:
        msg = {
            "type": "STORE",
            "layer_name": entry.layer_name,
            "token_ids": entry.token_ids,
            "slot_mapping": entry.slot_mapping,
            "cuda_event": event.ipc_handle(),  # 跨进程同步
        }
        self.zmq_socket.send(msg)

# multi_process_adapter.py:559
def batched_submit_retrieve_requests(self, retrieve_entries):
    # 类似 store，但方向相反
    event = torch.cuda.Event(interprocess=True)
    event.record()
    for entry in retrieve_entries:
        msg = {"type": "RETRIEVE", ...}
        self.zmq_socket.send(msg)
```

---

## 七、总结：什么时候用 LMCache

### 7.1 时间线总结

```
请求生命周期:
  │
  ├─ 1. Scheduler: get_num_new_matched_tokens()
  │     └─ 【查询 LMCache】: 有多少 token 的 KV Cache 可用？
  │
  ├─ 2. Scheduler: alloc_blocks()
  │     └─ 为 LMCache 命中的 token 分配 GPU Block
  │
  ├─ 3. Worker: start_load_kv()
  │     └─ 【从 LMCache 加载】: 将 KV Cache 从 DRAM/远程 加载到 GPU HBM
  │
  ├─ 4. Worker: 模型 forward (逐层)
  │     ├─ wait_for_layer_load(layer_N)
  │     │     └─ 【等待加载】: 等待第 N 层从 LMCache 加载完成
  │     ├─ Attention(layer_N)
  │     │     └─ 使用已加载的 KV Cache + 计算新 token
  │     └─ save_kv_layer(layer_N)
  │           └─ 【保存到 LMCache】: 将第 N 层的新 KV Cache 保存
  │
  ├─ 5. Worker: wait_for_save()
  │     └─ 【等待保存完成】: 确保所有 KV Cache 已安全保存
  │
  └─ 6. Worker: get_finished()
        └─ 报告异步传输完成状态
```

### 7.2 各场景触发条件汇总

| 场景 | HBM→DRAM 触发条件 | DRAM→HBM 触发条件 |
|------|-------------------|-------------------|
| **PD 混部** | prefill 结束 + chunk 对齐；或 decode 每步（save_decode_cache=True） | 新请求到达 + LMCache 命中 |
| **PD 分离 (Prefill 端)** | is_last_prefill=True → 传输到远程 | 不加载（Prefill 端只生产） |
| **PD 分离 (Decode 端)** | 不保存（kv_consumer） | 新请求到达 + 远程 KV 到达 |
| **SimpleCPUOffload** | get_finished() 中异步 DMA | get_finished() 中异步 DMA |
| **OffloadingConnector** | 下一个 engine step 开始时 | start_kv_transfers() 中 |

### 7.3 关键配置项

```yaml
# LMCache 配置文件
local_cpu: True                    # 启用本地 CPU 缓存
max_local_cpu_size: 10             # CPU 缓存容量 (GB)
enable_nixl: True                  # 启用 NIXL 远程传输
nixl_role: "sender" / "receiver"   # NIXL 角色
lmcache_chunk_size: 256            # chunk 大小 (tokens)
save_decode_cache: False           # 是否在 decode 阶段保存
use_layerwise: True                # 是否使用逐层流水线
```

```json
// vLLM 启动参数
{
    "kv_connector": "LMCacheConnectorV1",
    "kv_role": "kv_both",          // PD 混部
    // "kv_role": "kv_producer",   // PD 分离 Prefill 端
    // "kv_role": "kv_consumer",   // PD 分离 Decode 端
    "kv_connector_extra_config": {
        "use_native": true,
        "use_layerwise": true
    }
}
```
