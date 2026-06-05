# vLLM V1 调度器与 Engine Core 深度解析

> 本文档详细剖析 vLLM V1 的调度器 (`Scheduler`) 和引擎核心 (`EngineCore`) 的完整逻辑，包含详细的架构图、数据流图和代码级分析，适合面试讲解。

---

## 目录

- [1. 整体架构图](#1-整体架构图)
- [2. EngineCore 初始化流程](#2-enginecore-初始化流程)
- [3. EngineCore.step() 核心循环](#3-enginecorestep-核心循环)
- [4. Scheduler.schedule() 调度详解](#4-schedulerschedule-调度详解)
- [5. KV Cache 管理与调度的协作](#5-kv-cache-管理与调度的协作)
- [6. KVConnector 与 P/D 分离](#6-kvconnector-与-pd-分离)
- [7. SchedulerOutput 到 Model Runner 的链路](#7-scheduleroutput-到-model-runner-的链路)
- [8. update_from_output 输出处理](#8-update_from_output-输出处理)
- [9. 关键数据结构](#9-关键数据结构)
- [10. 代码索引](#10-代码索引)

---

## 1. 整体架构图

### 1.1 Engine Core 三层架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Layer 1: 前端层 (Frontend)                       │
│                                                                         │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐  │
│  │   AsyncLLM       │    │   LLMEngine      │    │   API Server     │  │
│  │  (异步在线服务)    │    │  (同步离线推理)    │    │  (OpenAI 兼容)   │  │
│  └────────┬─────────┘    └────────┬─────────┘    └────────┬─────────┘  │
│           │                       │                       │             │
│           └───────────────────────┼───────────────────────┘             │
│                                   │                                     │
│                    EngineCoreClient (ZMQ IPC)                           │
│                    ├─ InprocClient (进程内)                              │
│                    ├─ SyncMPClient (同步多进程)                          │
│                    └─ AsyncMPClient (异步多进程)                         │
└───────────────────────────────────┬─────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼─────────────────────────────────────┐
│                        Layer 2: 引擎核心层 (EngineCore)                  │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                     EngineCore.__init__()                        │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │   │
│  │  │   Executor   │  │  Scheduler   │  │  KVCacheManager      │  │   │
│  │  │  (模型执行器) │  │  (调度器)     │  │  (KV 缓存管理)       │  │   │
│  │  └──────┬───────┘  └──────┬───────┘  └──────────┬───────────┘  │   │
│  │         │                 │                      │              │   │
│  │         │    ┌────────────┴──────────────────────┘              │   │
│  │         │    │                                                   │   │
│  │         │    ▼                                                   │   │
│  │         │  ┌──────────────────────────────────────────────────┐ │   │
│  │         │  │            KVCacheCoordinator                    │ │   │
│  │         │  │  ┌─────────────┐  ┌─────────────┐  ┌─────────┐ │ │   │
│  │         │  │  │FullAttention│  │ SlidingWindow│  │ Mamba   │ │ │   │
│  │         │  │  │  Manager    │  │   Manager    │  │ Manager │ │ │   │
│  │         │  │  └──────┬──────┘  └──────┬──────┘  └────┬────┘ │ │   │
│  │         │  │         └────────────────┼──────────────┘       │ │   │
│  │         │  │                          ▼                      │ │   │
│  │         │  │                    BlockPool                    │ │   │
│  │         │  │  ┌────────────────┐  ┌────────────────────┐    │ │   │
│  │         │  │  │FreeBlockQueue  │  │BlockHashToBlockMap │    │ │   │
│  │         │  │  │(空闲块双向链表) │  │(前缀缓存哈希表)    │    │ │   │
│  │         │  │  └────────────────┘  └────────────────────┘    │ │   │
│  │         │  └──────────────────────────────────────────────────┘ │   │
│  │         │                                                       │   │
│  │  ┌──────┴───────────────────────────────────────────────────┐  │   │
│  │  │                    KVConnector                            │  │   │
│  │  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────┐ │  │   │
│  │  │  │  NIXL    │  │ LMCache  │  │ P2pNccl  │  │ Offload │ │  │   │
│  │  │  │(RDMA)    │  │          │  │          │  │         │ │  │   │
│  │  │  └──────────┘  └──────────┘  └──────────┘  └─────────┘ │  │   │
│  │  └──────────────────────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└───────────────────────────────────┬─────────────────────────────────────┘
                                    │ collective_rpc / 共享内存 MQ
┌───────────────────────────────────▼─────────────────────────────────────┐
│                        Layer 3: 工作层 (Worker)                          │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    GPUModelRunner (~7400 行)                     │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │   │
│  │  │ _update_     │  │ _prepare_    │  │   _model_forward()   │  │   │
│  │  │ states()     │  │ inputs()     │  │   ┌──────────────┐   │  │   │
│  │  │              │  │              │  │   │ Transformer  │   │  │   │
│  │  │ 更新持久化    │  │ 构建 GPU     │  │   │ Layers       │   │  │   │
│  │  │ 批处理状态    │  │ 输入张量     │  │   │ ┌──────────┐ │   │  │   │
│  │  └──────────────┘  └──────────────┘  │   │ │Attention │ │   │  │   │
│  │                                      │   │ │ + KV写入 │ │   │  │   │
│  │  ┌──────────────┐  ┌──────────────┐  │   │ └──────────┘ │   │  │   │
│  │  │ sample_      │  │  InputBatch  │  │   │ ┌──────────┐ │   │  │   │
│  │  │ tokens()     │  │ (持久化状态)  │  │   │ │   FFN    │ │   │  │   │
│  │  │              │  │              │  │   │ └──────────┘ │   │  │   │
│  │  │ 采样 + 后处理 │  │ block_table  │  │   └──────────────┘   │  │   │
│  │  └──────────────┘  │ slot_mapping │  └──────────────────────┘  │   │
│  │                    └──────────────┘                             │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.2 单步执行数据流图

```
时间 →
═══════════════════════════════════════════════════════════════════════════

EngineCore.step()
    │
    ├─── ① Scheduler.schedule() ──────────────────────────────────────────
    │         │
    │         ├── Phase 1: 调度 RUNNING 请求
    │         │   for request in self.running:
    │         │     num_new_tokens = num_tokens_with_spec
    │         │                     + num_output_placeholders
    │         │                     - num_computed_tokens
    │         │     blocks = kv_cache_manager.allocate_slots()
    │         │     if blocks is None → preempt lowest priority
    │         │
    │         ├── Phase 2: 调度 WAITING 请求
    │         │   for request in self.waiting:
    │         │     computed_blocks = kv_cache_manager.get_computed_blocks()
    │         │     remote_tokens = connector.get_num_new_matched_tokens()
    │         │     num_new_tokens = num_tokens - num_computed_tokens
    │         │     blocks = kv_cache_manager.allocate_slots()
    │         │
    │         └── 构建 SchedulerOutput
    │               ├── scheduled_new_reqs (新请求完整数据)
    │               ├── scheduled_cached_reqs (已知请求差量数据)
    │               ├── num_scheduled_tokens (每请求 token 计数)
    │               └── kv_connector_metadata (KV 传输元数据)
    │
    ├─── ② Executor.execute_model(scheduler_output, non_block=True) ──────
    │         │
    │         └── Worker.execute_model()
    │               │
    │               ├── GPUModelRunner._update_states()
    │               │   ├── 移除完成的请求
    │               │   ├── 清零新分配的块
    │               │   ├── 添加新请求到 InputBatch
    │               │   └── 更新已有请求的块 IDs
    │               │
    │               ├── GPUModelRunner._prepare_inputs()
    │               │   ├── 构建 input_ids, positions
    │               │   ├── 计算 slot_mapping
    │               │   ├── 构建 attention metadata
    │               │   └── 构建 logits_indices
    │               │
    │               ├── GPUModelRunner._model_forward()
    │               │   └── 每层 Transformer:
    │               │       ├── 算 Q, K, V
    │               │       ├── KV 写入: key_cache[slot_mapping] = K
    │               │       ├── Attention: Q × K^T → softmax → × V
    │               │       └── FFN
    │               │
    │               └── 返回 hidden_states (非最后 PP 阶段)
    │                   或 存储 execute_model_state (最后 PP 阶段)
    │
    ├─── ③ Scheduler.get_grammar_bitmask() ── CPU 与 GPU 并行 ───────────
    │         └── 计算结构化输出的语法掩码 (JSON schema, regex 等)
    │
    ├─── ④ future.result() ── 等待 GPU 完成 ──────────────────────────────
    │
    ├─── ⑤ Executor.sample_tokens(grammar_output) ───────────────────────
    │         │
    │         └── GPUModelRunner.sample_tokens()
    │               ├── 应用 grammar 掩码到 logits
    │               ├── 运行 Sampler (temperature, top_p, top_k)
    │               ├── 推测解码: 提议草稿 token
    │               ├── 后处理: logprobs, D2H 拷贝
    │               └── 返回 ModelRunnerOutput
    │
    └─── ⑥ Scheduler.update_from_output(scheduler_output, model_output) ─
              │
              ├── 处理采样 token: 追加到请求, 检查停止条件
              ├── 处理推测解码拒绝: 回退 num_computed_tokens
              ├── 释放完成的请求: free KV blocks
              └── 返回 EngineCoreOutputs → 发送给前端
```

### 1.3 请求生命周期状态图

```
                            ┌──────────────────────────────────────────┐
                            │            请求生命周期                    │
                            └──────────────────────────────────────────┘

    add_request()
         │
         ▼
    ┌─────────┐   schedule() Phase 2    ┌─────────┐   update_from_output()
    │ WAITING │ ──────────────────────→ │ RUNNING │ ──────────────────────→ FINISHED_*
    └─────────┘   allocate_slots 成功    └─────────┘   检测到停止条件
         │              │                    │
         │              │                    │ schedule() Phase 1
         │              │                    │ allocate_slots 失败
         │              │                    ▼
         │              │              ┌──────────┐
         │              │              │PREEMPTED │
         │              │              └──────────┘
         │              │                    │
         │              │                    │ 释放 KV blocks
         │              │                    │ num_computed_tokens = 0
         │              │                    │ 放回 waiting 队列头部
         │              │                    ▼
         │              └────────────────────┘ (重新调度)
         │
         │  额外的等待状态:
         ├──→ WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR (等待语法编译)
         ├──→ WAITING_FOR_REMOTE_KVS (等待远程 KV 传输)
         └──→ WAITING_FOR_STREAMING_REQ (等待流式输入)

    结束状态:
    ├── FINISHED_STOPPED      (遇到 stop token)
    ├── FINISHED_LENGTH_CAPPED (达到 max_tokens)
    ├── FINISHED_ABORTED       (客户端取消)
    ├── FINISHED_IGNORED       (prompt 超长)
    ├── FINISHED_ERROR         (执行出错)
    └── FINISHED_REPETITION    (重复惩罚)
```

### 1.4 KV Cache 分配与释放流程图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    KV Cache 块生命周期                                   │
└─────────────────────────────────────────────────────────────────────────┘

    ┌──────────────────┐
    │   空闲块池        │
    │  (FreeBlockQueue) │
    │  LRU 顺序排列     │
    └────────┬─────────┘
             │
             │ get_new_blocks()
             │ 从头部弹出 LRU 块
             │ ref_cnt = 0 → 1
             │ 如果有缓存哈希 → 驱逐
             ▼
    ┌──────────────────┐
    │   已分配块        │
    │  ref_cnt = 1     │
    │  存储 KV 数据     │
    └────────┬─────────┘
             │
             │ cache_full_blocks()
             │ 块满后计算链式哈希
             │ 存入前缀缓存哈希表
             ▼
    ┌──────────────────┐
    │   已缓存块        │
    │  ref_cnt >= 1    │
    │  可被前缀缓存命中 │
    └────────┬─────────┘
             │
             ├──→ touch() (前缀缓存命中)
             │    ref_cnt++ , 从空闲队列移除
             │    块被保护，不会被驱逐
             │
             ├──→ free_blocks() (请求完成)
             │    ref_cnt--
             │    如果 ref_cnt == 0 → 放回空闲队列尾部 (MRU 端)
             │
             └──→ _preempt_request() (预抢占)
                  释放所有块
                  ref_cnt-- → 0 → 放回空闲队列

    ┌──────────────────┐
    │   null_block     │
    │  (占位符块)       │
    │  block_id = 0    │
    │  用于 SWA 填充    │
    │  ref_cnt 不维护   │
    └──────────────────┘
```

### 1.5 P/D 分离数据流图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    P/D 分离架构                                          │
└─────────────────────────────────────────────────────────────────────────┘

    客户端请求
         │
         ▼
    ┌─────────┐
    │  Proxy  │ (路由层)
    └────┬────┘
         │
    ┌────┴────────────────────────────────────────┐
    │                                              │
    ▼                                              ▼
┌──────────────────┐                    ┌──────────────────┐
│  Prefill 节点     │                    │  Decode 节点      │
│                  │                    │                  │
│  Scheduler:      │                    │  Scheduler:      │
│  kv_role=        │                    │  kv_role=        │
│  "kv_producer"   │                    │  "kv_consumer"   │
│                  │                    │                  │
│  1. 处理 prompt   │                    │  1. 收到请求      │
│  2. 计算 KV Cache │                    │  2. connector.   │
│  3. request_     │   ┌─────────────┐  │     get_num_     │
│     finished()   │──→│  KV 传输    │──→│     new_matched  │
│  4. 返回 kv_     │   │  (RDMA/     │  │     _tokens()    │
│     transfer_    │   │   NCCL)     │  │  3. 分配 KV 块   │
│     params       │   └─────────────┘  │  4. start_load_  │
│                  │                    │     kv() 拉取    │
│                  │                    │  5. 逐 token 解码 │
└──────────────────┘                    └──────────────────┘

    KVConnector 交互时序:
    ┌──────────┐          ┌──────────┐          ┌──────────┐
    │ Scheduler│          │ KVConn.  │          │  Worker  │
    └────┬─────┘          └────┬─────┘          └────┬─────┘
         │                     │                     │
         │ get_num_new_        │                     │
         │ matched_tokens()    │                     │
         │────────────────────→│                     │
         │ (num_tokens,        │                     │
         │  is_async)          │                     │
         │←────────────────────│                     │
         │                     │                     │
         │ allocate_slots()    │                     │
         │────────────────────→│                     │
         │                     │                     │
         │ build_connector_    │                     │
         │ meta()              │                     │
         │────────────────────→│                     │
         │                     │                     │
         │                     │  start_load_kv()   │
         │                     │────────────────────→│
         │                     │                     │
         │                     │  wait_for_layer_   │
         │                     │  load()            │
         │                     │←────────────────────│
         │                     │                     │
         │                     │  save_kv_layer()   │
         │                     │←────────────────────│
         │                     │                     │
```

---

## 2. EngineCore 初始化流程

### 2.1 初始化序列图

```
EngineCore.__init__(vllm_config, executor_class)
    │
    ├── ① load_general_plugins()
    │      加载插件系统
    │
    ├── ② executor = executor_class(vllm_config)
    │      创建模型执行器 (UniProc/Multiproc/Ray)
    │
    ├── ③ _initialize_kv_caches()
    │      │
    │      ├── 3a. executor.get_kv_cache_specs()
    │      │      每个 Worker 收集注意力层的 KVCacheSpec
    │      │      返回 dict[layer_name, KVCacheSpec]
    │      │
    │      ├── 3b. executor.determine_available_memory()
    │      │      运行 dummy forward pass，测量可用 GPU 内存
    │      │      available = total * util - weights - peak - cudagraph
    │      │
    │      ├── 3c. get_kv_cache_configs()
    │      │      合并规格 → 分组 → 计算块数 → 跨 Worker 归一化
    │      │      返回 KVCacheConfig(num_blocks, kv_cache_tensors, groups)
    │      │
    │      └── 3d. executor.initialize_from_config(kv_cache_configs)
    │             每个 Worker:
    │             ├── 分配 KV Cache GPU 张量
    │             ├── 绑定到注意力层
    │             └── 编译/预热模型 (CUDA Graph 等)
    │
    ├── ④ scheduler = Scheduler(vllm_config, kv_cache_config, ...)
    │      │
    │      ├── 创建 KVCacheManager (含 KVCacheCoordinator)
    │      ├── 创建 KVConnector (如果配置了 P/D 分离)
    │      ├── 创建 ECConnector (编码器缓存传输)
    │      ├── 初始化请求队列: waiting, running, skipped_waiting
    │      └── 设置调度参数: max_num_running_reqs, max_num_scheduled_tokens
    │
    ├── ⑤ 设置 batch_queue (流水线并行)
    │      if max_concurrent_batches > 1:
    │          batch_queue = deque(maxlen=max_concurrent_batches)
    │
    └── ⑥ 选择 step 函数
           if batch_queue_size > 1:
               step_fn = step_with_batch_queue  # 流水线并行模式
           else:
               step_fn = step                   # 普通模式
```

### 2.2 KV Cache 初始化详解

```
_initialize_kv_caches()
    │
    ├── get_kv_cache_specs()
    │      │
    │      │  每个 Worker 调用 GPUModelRunner.get_kv_cache_spec():
    │      │
    │      │  for layer_name, attn_module in attention_layers:
    │      │      if kv_sharing_target_layer_name:
    │      │          shared_kv_cache_layers[layer_name] = target
    │      │          continue  # 跳过共享层
    │      │      spec = attn_module.get_kv_cache_spec()
    │      │      kv_cache_spec[layer_name] = spec
    │      │
    │      │  KVCacheSpec 类型:
    │      │  ├── FullAttentionSpec    (标准全注意力)
    │      │  ├── MLAAttentionSpec     (DeepSeek MLA)
    │      │  ├── SlidingWindowSpec    (滑动窗口)
    │      │  ├── MambaSpec            (SSM)
    │      │  └── CrossAttentionSpec   (交叉注意力)
    │      │
    │      ▼
    │
    ├── determine_available_memory()
    │      │
    │      │  available = total_memory * gpu_memory_utilization
    │      │            - weights_memory
    │      │            - peak_activation_memory
    │      │            - cudagraph_memory_estimate
    │      │
    │      ▼
    │
    ├── get_kv_cache_configs()
    │      │
    │      │  1. 合并所有 Worker 的 specs
    │      │  2. 分组: get_kv_cache_groups()
    │      │     ├── 单组 (所有层相同)
    │      │     ├── UniformTypeKVCacheSpecs (MLA 等)
    │      │     └── 混合组 (全注意力 + SWA)
    │      │  3. 计算块数: num_blocks = available / page_size / num_layers
    │      │  4. 跨 Worker 归一化: 取最小值
    │      │
    │      ▼
    │
    └── initialize_from_config()
           │
           │  每个 Worker:
           │  1. 分配原始 int8 缓冲区
           │  2. 重塑为后端期望的形状
           │  3. 处理块拆分 (调度器块大小 vs 内核块大小)
           │  4. 绑定到注意力层
           │  5. 编译/预热模型
           │
           ▼
```

---

## 3. EngineCore.step() 核心循环

### 3.1 普通模式 (step)

```python
# vllm/v1/engine/core.py, line 483
def step(self):
    # ① 调度: 决定哪些请求执行，分配 KV 块
    scheduler_output = self.scheduler.schedule()

    # ② 异步执行模型前向传播 (GPU 开始工作)
    future = self.model_executor.execute_model(
        scheduler_output, non_block=True)

    # ③ CPU 计算语法掩码 (与 GPU 并行!)
    grammar_output = self.scheduler.get_grammar_bitmask(
        scheduler_output)

    # ④ 等待 GPU 完成
    model_output = future.result()

    # ⑤ 采样 token (使用语法掩码)
    if model_output is None:
        model_output = self.model_executor.sample_tokens(
            grammar_output)

    # ⑥ 更新调度器状态
    engine_core_outputs = self.scheduler.update_from_output(
        scheduler_output, model_output)

    return engine_core_outputs
```

### 3.2 流水线并行模式 (step_with_batch_queue)

```python
# vllm/v1/engine/core.py, line 538
def step_with_batch_queue(self):
    # ① 调度
    scheduler_output = self.scheduler.schedule()

    # ② 异步执行 (不等待结果)
    future = self.model_executor.execute_model(
        scheduler_output, non_block=True)

    # ③ 将 batch 加入队列左侧 (最新)
    batch_queue.appendleft((future, scheduler_output, exec_future))

    # ④ 如果队列未满，立即返回 (不等待结果)
    if len(batch_queue) < batch_queue_size:
        return None, True  # "先调度，后等待"

    # ⑤ 队列满了，等待最旧的 batch 完成
    oldest_future, oldest_output, _ = batch_queue.pop()
    model_output = oldest_future.result()

    # ⑥ 更新调度器状态
    engine_core_outputs = self.scheduler.update_from_output(
        oldest_output, model_output)

    return engine_core_outputs
```

**流水线并行时间线：**

```
时间 →
═══════════════════════════════════════════════════════════════════

batch_queue_size = 3

Batch 1: [Schedule] → [Forward PP Stage 0] → [Forward PP Stage 1] → [Done]
Batch 2:              [Schedule] → [Forward PP Stage 0] → [Forward PP Stage 1] → [Done]
Batch 3:                           [Schedule] → [Forward PP Stage 0] → [Forward PP Stage 1]
                                    ↑ 多个 batch 同时在 pipeline 中

CPU:     [S1] [S2] [S3] [Wait B1] [S4] [Wait B2] [S5] [Wait B3]
GPU:          [F1]      [F2]      [F3]      [F4]      [F5]
```

---

## 4. Scheduler.schedule() 调度详解

### 4.1 调度算法核心思想

```python
# vllm/v1/core/sched/scheduler.py, line 438
# NOTE: There's no "decoding phase" nor "prefill phase" in the scheduler.
# Each request just has num_computed_tokens and num_tokens_with_spec.
# At each step, the scheduler tries to assign tokens to the requests
# so that each request's num_computed_tokens can catch up its
# num_tokens_with_spec.
```

**核心思想：** 不区分 prefill 和 decode，只看差距。

```
请求 A: num_tokens=100, num_computed_tokens=0   → 差距=100 (新请求)
请求 B: num_tokens=100, num_computed_tokens=99   → 差距=1 (decode)
请求 C: num_tokens=100, num_computed_tokens=50   → 差距=50 (分块 prefill 中)

调度器不关心这些请求是 "prefill" 还是 "decode"，
只关心每个请求还需要算多少 token。
```

### 4.2 Phase 1: 调度 RUNNING 请求

```
┌─────────────────────────────────────────────────────────────────┐
│  Phase 1: 调度 RUNNING 请求 (已在运行的请求)                      │
│                                                                   │
│  输入: self.running 队列                                          │
│  输出: scheduled_running_reqs, num_scheduled_tokens               │
│                                                                   │
│  流程:                                                            │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ for request in self.running:                                │ │
│  │                                                             │ │
│  │   ① 计算 num_new_tokens                                     │ │
│  │      = num_tokens_with_spec + num_output_placeholders       │ │
│  │      - num_computed_tokens                                  │ │
│  │                                                             │ │
│  │   ② 截断到 token_budget                                     │ │
│  │      num_new_tokens = min(num_new_tokens, token_budget)     │ │
│  │                                                             │ │
│  │   ③ 截断到 max_model_len                                    │ │
│  │      num_new_tokens = min(num_new_tokens,                   │ │
│  │                           max_model_len - 1 - computed)     │ │
│  │                                                             │ │
│  │   ④ 分配 KV 块                                              │ │
│  │      blocks = kv_cache_manager.allocate_slots(              │ │
│  │          request, num_new_tokens)                           │ │
│  │                                                             │ │
│  │   ⑤ 如果分配失败 → 预抢占                                    │ │
│  │      while blocks is None:                                  │ │
│  │        preempted = max(running, key=priority)               │ │
│  │        _preempt_request(preempted)                          │ │
│  │        # 释放 KV, 状态→PREEMPTED, 放回 waiting 头部          │ │
│  │        blocks = allocate_slots(request, num_new_tokens)     │ │
│  │                                                             │ │
│  │   ⑥ 记录调度结果                                             │ │
│  │      scheduled_running_reqs.append(request)                 │ │
│  │      num_scheduled_tokens[req_id] = num_new_tokens          │ │
│  │      token_budget -= num_new_tokens                         │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### 4.3 Phase 2: 调度 WAITING 请求

```
┌─────────────────────────────────────────────────────────────────┐
│  Phase 2: 调度 WAITING 请求 (等待中的新请求)                      │
│                                                                   │
│  前提条件: Phase 1 没有发生预抢占 (避免 thrashing)                 │
│                                                                   │
│  输入: self.waiting 队列                                          │
│  输出: scheduled_new_reqs, scheduled_resumed_reqs                 │
│                                                                   │
│  流程:                                                            │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ while waiting and token_budget > 0:                         │ │
│  │                                                             │ │
│  │   ① 检查最大运行请求数                                       │ │
│  │      if len(running) >= max_num_running_reqs: break         │ │
│  │                                                             │ │
│  │   ② 前缀缓存查找                                             │ │
│  │      computed_blocks, num_computed =                        │ │
│  │          kv_cache_manager.get_computed_blocks(request)       │ │
│  │      # 返回缓存的块和已计算的 token 数                        │ │
│  │                                                             │ │
│  │   ③ 外部 KV 缓存查找 (P/D 分离)                              │ │
│  │      if connector:                                          │ │
│  │        remote_tokens, is_async =                            │ │
│  │            connector.get_num_new_matched_tokens(request)     │ │
│  │                                                             │ │
│  │   ④ 计算 num_new_tokens                                     │ │
│  │      num_new_tokens = num_tokens - num_computed              │ │
│  │      # num_computed = 本地缓存 + 远程缓存                     │ │
│  │                                                             │ │
│  │   ⑤ 分块预填充截断                                           │ │
│  │      if enable_chunked_prefill:                             │ │
│  │        num_new_tokens = min(num_new_tokens, token_budget)   │ │
│  │      elif num_new_tokens > token_budget:                    │ │
│  │        break  # 不分块，停止调度                              │ │
│  │                                                             │ │
│  │   ⑥ 分配 KV 块                                              │ │
│  │      blocks = kv_cache_manager.allocate_slots(              │ │
│  │          request, num_new_tokens,                           │ │
│  │          num_new_computed_tokens=num_computed,               │ │
│  │          new_computed_blocks=computed_blocks,                │ │
│  │          num_external_computed_tokens=remote_tokens)         │ │
│  │                                                             │ │
│  │   ⑦ 如果分配失败 → 停止 (不预抢占等待请求)                    │ │
│  │      if blocks is None: break                               │ │
│  │                                                             │ │
│  │   ⑧ 移入 running 队列                                       │ │
│  │      request.status = RUNNING                               │ │
│  │      running.append(request)                                │ │
│  │      token_budget -= num_new_tokens                         │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### 4.4 token_budget 机制

```
┌─────────────────────────────────────────────────────────────────┐
│  token_budget = max_num_scheduled_tokens (例如 8192)             │
│                                                                   │
│  Phase 1: RUNNING 请求消耗 budget                                 │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ Request A (decode):  num_new_tokens = 1   → budget -= 1    │ │
│  │ Request B (decode):  num_new_tokens = 1   → budget -= 1    │ │
│  │ Request C (prefill): num_new_tokens = 500 → budget -= 500  │ │
│  │ 剩余 budget = 8192 - 1 - 1 - 500 = 7690                   │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                   │
│  Phase 2: WAITING 请求消耗剩余 budget                             │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ Request D (新 prefill): num_new_tokens = 2000               │ │
│  │   → budget -= 2000, 剩余 5690                               │ │
│  │ Request E (新 prefill): num_new_tokens = 3000               │ │
│  │   → budget -= 3000, 剩余 2690                               │ │
│  │ Request F (新 prefill): num_new_tokens = 5000               │ │
│  │   → 截断到 2690 (enable_chunked_prefill)                    │ │
│  │   → budget -= 2690, 剩余 0                                  │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                   │
│  最终 batch: [A:1, B:1, C:500, D:2000, E:3000, F:2690]          │
│  总计: 8192 tokens                                                │
└─────────────────────────────────────────────────────────────────┘
```

### 4.5 SchedulerOutput 构建

```python
# vllm/v1/core/sched/scheduler.py, line 1053
scheduler_output = SchedulerOutput(
    # 新请求: 携带完整数据 (prompt tokens, 采样参数, 块 IDs 等)
    scheduled_new_reqs=new_reqs_data,

    # 已知请求: 只携带差量数据 (新块 IDs, 新 token IDs)
    # 减少 scheduler → worker 的数据传输量
    scheduled_cached_reqs=cached_reqs_data,

    # 每请求 token 计数
    num_scheduled_tokens=num_scheduled_tokens,
    total_num_scheduled_tokens=total_num_scheduled_tokens,

    # 推测解码 token
    scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,

    # 编码器输入
    scheduled_encoder_inputs=scheduled_encoder_inputs,

    # 级联注意力的公共前缀块数
    num_common_prefix_blocks=num_common_prefix_blocks,

    # 状态变更
    preempted_req_ids={req.request_id for req in preempted_reqs},
    finished_req_ids=self.finished_req_ids,

    # KV 传输元数据
    kv_connector_metadata=kv_connector_metadata,
)
```

---

## 5. KV Cache 管理与调度的协作

### 5.1 allocate_slots 详解

```
┌─────────────────────────────────────────────────────────────────┐
│  KVCacheManager.allocate_slots()                                │
│                                                                   │
│  块布局:                                                          │
│  | < comp > | < new_comp > | < ext_comp > | < new > | < look > | │
│     已计算      前缀缓存命中    外部缓存       新计算      推测     │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ Stage 1: 释放滑动窗口外的块 + 容量检查                        │ │
│  │   coordinator.remove_skipped_blocks()                       │ │
│  │   num_blocks = coordinator.get_num_blocks_to_allocate()     │ │
│  │   if num_blocks > free_blocks: return None                  │ │
│  ├─────────────────────────────────────────────────────────────┤ │
│  │ Stage 2: 附加前缀缓存命中的块                                 │ │
│  │   coordinator.allocate_new_computed_blocks()                 │ │
│  │   # touch 缓存块, 增加 ref_cnt, 从空闲队列移除                │ │
│  ├─────────────────────────────────────────────────────────────┤ │
│  │ Stage 3: 分配新块                                            │ │
│  │   coordinator.allocate_new_blocks()                         │ │
│  │   # 从空闲池弹出 LRU 块, ref_cnt = 1                         │ │
│  ├─────────────────────────────────────────────────────────────┤ │
│  │ Stage 4: 缓存新填满的块                                      │ │
│  │   coordinator.cache_blocks()                                │ │
│  │   # 计算链式哈希, 存入前缀缓存哈希表                          │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                   │
│  返回: KVCacheBlocks 或 None (内存不足)                           │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 get_computed_blocks 前缀缓存查找

```
┌─────────────────────────────────────────────────────────────────┐
│  KVCacheManager.get_computed_blocks(request)                     │
│                                                                   │
│  ① 如果禁用缓存 → 返回 (empty, 0)                                │
│                                                                   │
│  ② max_cache_hit_length = request.num_tokens - 1                 │
│     # 最后一个 token 必须重算 (为了拿到 logits)                    │
│                                                                   │
│  ③ coordinator.find_longest_cache_hit(block_hashes, max_length)  │
│     │                                                             │
│     │  UnitaryCoordinator (单组):                                 │
│     │  for i in range(max_num_blocks):                           │
│     │    cached = block_pool.get_cached_block(hash[i], groups)   │
│     │    if cached: computed_blocks.append(cached)               │
│     │    else: break  # 中断即停                                  │
│     │                                                             │
│     │  HybridCoordinator (混合组):                                │
│     │  # 固定点迭代: 每种注意力类型接受或缩短候选长度               │
│     │  while True:                                                │
│     │    for manager in managers:                                 │
│     │      hits = manager.find_longest_cache_hit(hash, length)   │
│     │      length = min(length, hits_length)                     │
│     │    if converged: break                                      │
│     │                                                             │
│     ▼                                                             │
│  ④ 返回 (KVCacheBlocks, num_computed_tokens)                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. KVConnector 与 P/D 分离

### 6.1 KVConnector 接口

```
┌─────────────────────────────────────────────────────────────────┐
│  KVConnectorBase_V1 接口                                         │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │  Scheduler 侧 (在调度器进程中运行)                            │ │
│  │                                                             │ │
│  │  get_num_new_matched_tokens(request, num_computed)           │ │
│  │    → 返回远程缓存命中的 token 数                              │ │
│  │                                                             │ │
│  │  update_state_after_alloc(request, blocks, num_external)    │ │
│  │    → 块分配后更新连接器状态                                   │ │
│  │                                                             │ │
│  │  build_connector_meta(scheduler_output)                     │ │
│  │    → 构建传递给 Worker 的元数据                               │ │
│  │                                                             │ │
│  │  request_finished(request, block_ids)                       │ │
│  │    → 请求完成时，返回 kv_transfer_params                     │ │
│  ├─────────────────────────────────────────────────────────────┤ │
│  │  Worker 侧 (在 Worker 进程中运行)                            │ │
│  │                                                             │ │
│  │  start_load_kv(forward_context)                             │ │
│  │    → 开始异步加载 KV                                         │ │
│  │                                                             │ │
│  │  wait_for_layer_load(layer_name)                            │ │
│  │    → 等待特定层的 KV 加载完成                                 │ │
│  │                                                             │ │
│  │  save_kv_layer(layer_name, kv_layer, attn_metadata)         │ │
│  │    → 开始异步保存 KV                                         │ │
│  │                                                             │ │
│  │  wait_for_save()                                            │ │
│  │    → 等待所有保存完成                                        │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 已注册的 KV 连接器

| 连接器 | 传输方式 | 适用场景 |
|--------|----------|----------|
| `NixlConnector` | RDMA (UCX/RoCE/IB) | 生产推荐，零拷贝，异步 |
| `LMCacheConnectorV1` | LMCache 引擎 | 分布式 KV 缓存 |
| `P2pNcclConnector` | NCCL P2P | 简单，GPU-GPU 直连 |
| `OffloadingConnector` | CPU/磁盘 | KV 卸载 |
| `MultiConnector` | 组合多个 | 同时使用多个连接器 |
| `FlexKVConnectorV1` | FlexKV | 灵活 KV 存储 |

---

## 7. SchedulerOutput 到 Model Runner 的链路

### 7.1 数据转换流程

```
┌─────────────────────────────────────────────────────────────────┐
│  SchedulerOutput → GPUModelRunner 转换链路                       │
└─────────────────────────────────────────────────────────────────┘

    SchedulerOutput
         │
         ▼
    ┌─────────────────────────────────────────────────────────────┐
    │  _update_states(scheduler_output)  [line 1120]              │
    │                                                             │
    │  ① 移除完成的请求                                            │
    │     for req_id in finished_req_ids:                         │
    │       del self.requests[req_id]                             │
    │       input_batch.remove_request(req_id)                    │
    │                                                             │
    │  ② 清零新分配的块                                            │
    │     for block_id in new_block_ids_to_zero:                  │
    │       kv_cache[block_id].zero_()                            │
    │                                                             │
    │  ③ 添加新请求                                                │
    │     for new_req in scheduled_new_reqs:                      │
    │       req_state = CachedRequestState(...)                   │
    │       self.requests[req_id] = req_state                     │
    │                                                             │
    │  ④ 更新已有请求                                              │
    │     for cached_req in scheduled_cached_reqs:                │
    │       req_state.num_computed_tokens = num_computed_tokens   │
    │       req_state.block_ids.append(new_block_ids)             │
    │                                                             │
    │  ⑤ 提交到 InputBatch                                        │
    │     input_batch.add_request(req_state)                      │
    └─────────────────────────────────────────────────────────────┘
         │
         ▼
    ┌─────────────────────────────────────────────────────────────┐
    │  _prepare_inputs(scheduler_output)  [line 1867]             │
    │                                                             │
    │  ① 提交 block_table 到 GPU                                  │
    │     input_batch.block_table.commit_block_table(num_reqs)    │
    │                                                             │
    │  ② 构建 req_indices                                          │
    │     # [0,0,0, 1,1, 2,2,2,2,2] (每个请求的 token 数)         │
    │     req_indices = repeat(arange(num_reqs), num_tokens)      │
    │                                                             │
    │  ③ 构建 positions                                            │
    │     # 每个 token 在序列中的位置                               │
    │     positions = num_computed_tokens + query_pos              │
    │                                                             │
    │  ④ 构建 input_ids                                            │
    │     # 从持久化存储中 gather                                   │
    │     input_ids = gather(token_ids_storage, token_indices)    │
    │                                                             │
    │  ⑤ 计算 slot_mapping                                         │
    │     # token 位置 → KV Cache 物理 slot                        │
    │     slot_id = block_table[req, pos//block_size]             │
    │              * block_size + pos % block_size                │
    │                                                             │
    │  ⑥ 构建 attention metadata                                   │
    │     query_start_loc, seq_lens, block_table, slot_mapping    │
    │                                                             │
    │  ⑦ 构建 logits_indices                                       │
    │     # 哪些位置需要计算 logits (通常是每个请求的最后一个 token)  │
    └─────────────────────────────────────────────────────────────┘
         │
         ▼
    ┌─────────────────────────────────────────────────────────────┐
    │  _model_forward()  [line 3676]                              │
    │                                                             │
    │  model(input_ids, positions, inputs_embeds, ...)            │
    │                                                             │
    │  每层 Transformer:                                           │
    │  ┌───────────────────────────────────────────────────────┐  │
    │  │ Attention.forward(query, key, value):                 │  │
    │  │   # ① 算 Q, K, V                                      │  │
    │  │   Q = W_q @ hidden_states                             │  │
    │  │   K = W_k @ hidden_states                             │  │
    │  │   V = W_v @ hidden_states                             │  │
    │  │                                                       │  │
    │  │   # ② 写入 KV Cache (通过 slot_mapping)               │  │
    │  │   key_cache[slot_mapping] = K                         │  │
    │  │   value_cache[slot_mapping] = V                       │  │
    │  │                                                       │  │
    │  │   # ③ 注意力计算 (通过 block_table 间接寻址)           │  │
    │  │   attn_out = flash_attention(Q, key_cache,            │  │
    │  │                              value_cache,             │  │
    │  │                              block_table)             │  │
    │  └───────────────────────────────────────────────────────┘  │
    │                                                             │
    │  hidden_states → compute_logits → logits                    │
    └─────────────────────────────────────────────────────────────┘
```

---

## 8. update_from_output 输出处理

### 8.1 输出处理流程

```
┌─────────────────────────────────────────────────────────────────┐
│  Scheduler.update_from_output(scheduler_output, model_output)    │
│                                                                   │
│  输入: scheduler_output (调度信息) + model_output (采样结果)       │
│  输出: dict[int, EngineCoreOutputs] (按 client_index 分组)        │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ for each scheduled request:                                 │ │
│  │                                                             │ │
│  │   ① 获取采样 token                                          │ │
│  │      token_ids = sampled_token_ids[req_index]               │ │
│  │                                                             │ │
│  │   ② 处理推测解码拒绝                                         │ │
│  │      if num_rejected > 0:                                   │ │
│  │        request.num_computed_tokens -= num_rejected          │ │
│  │        request.spec_token_ids = []                          │ │
│  │                                                             │ │
│  │   ③ 追加输出 token                                          │ │
│  │      request.append_output_token_ids(token_ids)             │ │
│  │                                                             │ │
│  │   ④ 检查停止条件                                             │ │
│  │      stopped, stop_reason = check_stop(request)             │ │
│  │      # 检查: EOS, stop_tokens, max_tokens, repetition       │ │
│  │                                                             │ │
│  │   ⑤ 如果停止                                                │ │
│  │      request.status = FINISHED_STOPPED / FINISHED_*         │ │
│  │      kv_cache_manager.free(request)                         │ │
│  │      finished_req_ids.add(request.request_id)               │ │
│  │                                                             │ │
│  │   ⑥ 构建 EngineCoreOutput                                   │ │
│  │      EngineCoreOutput(                                      │ │
│  │        new_token_ids=token_ids,                             │ │
│  │        new_logprobs=logprobs,                               │ │
│  │        finish_reason=finish_reason,                         │ │
│  │        ...)                                                 │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ 后处理:                                                      │ │
│  │   - 移除完成的请求从 running 队列                              │ │
│  │   - 处理 KV connector 输出 (完成的传输)                       │ │
│  │   - 发布 KV cache 事件                                       │ │
│  │   - 返回 EngineCoreOutputs 按 client_index 分组              │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## 9. 关键数据结构

### 9.1 SchedulerOutput

```python
# vllm/v1/core/sched/output.py
class SchedulerOutput:
    # 新请求的完整数据
    scheduled_new_reqs: list[NewRequestData]
    # 已知请求的差量数据 (减少通信量)
    scheduled_cached_reqs: CachedRequestData
    # 每请求 token 计数
    num_scheduled_tokens: dict[str, int]
    # 总 token 数
    total_num_scheduled_tokens: int
    # 推测解码 token
    scheduled_spec_decode_tokens: dict[str, list[int]]
    # 编码器输入
    scheduled_encoder_inputs: dict[str, list[int]]
    # 级联注意力的公共前缀块数
    num_common_prefix_blocks: list[int]
    # 状态变更
    preempted_req_ids: set[str]
    finished_req_ids: set[str]
    # KV 传输元数据
    kv_connector_metadata: KVConnectorMetadata | None
    # 需要清零的新块 ID
    new_block_ids_to_zero: list[int] | None
```

### 9.2 EngineCoreOutput

```python
# vllm/v1/engine/__init__.py
class EngineCoreOutput:
    request_id: str
    new_token_ids: list[int]           # 新生成的 token IDs
    new_logprobs: list[LogprobsList]   # logprobs
    new_prompt_logprobs: PromptLogprobsList | None
    finish_reason: FinishReason | None
    stop_reason: int | str | None
    events: list[EngineCoreEvent] | None
    kv_transfer_params: dict[str, Any] | None
```

### 9.3 KVCacheBlock

```python
# vllm/v1/core/kv_cache_utils.py
@dataclass(slots=True)
class KVCacheBlock:
    block_id: int                    # 物理块 ID
    ref_cnt: int = 0                 # 引用计数
    _block_hash: BlockHashWithGroupId | None = None  # 内容哈希
    prev_free_block: KVCacheBlock | None = None       # 空闲链表前驱
    next_free_block: KVCacheBlock | None = None       # 空闲链表后继
    is_null: bool = False            # 是否为空块 (SWA 填充)
```

---

## 10. 代码索引

### 10.1 调度器相关

| 文件 | 关键方法 | 行号 | 用途 |
|------|----------|------|------|
| `vllm/v1/core/sched/scheduler.py` | `__init__()` | 87 | 初始化调度器 |
| | `schedule()` | 428 | 核心调度方法 |
| | `update_from_output()` | 1588 | 处理模型输出 |
| | `_preempt_request()` | 1151 | 预抢占请求 |
| | `add_request()` | 2112 | 添加新请求 |
| | `finish_requests()` | 2149 | 完成请求 |
| `vllm/v1/core/sched/output.py` | `SchedulerOutput` | 181 | 调度输出数据结构 |
| `vllm/v1/core/sched/interface.py` | `SchedulerInterface` | 36 | 调度器接口 |
| | `PauseState` | 22 | 暂停状态枚举 |

### 10.2 KV Cache 相关

| 文件 | 关键方法 | 行号 | 用途 |
|------|----------|------|------|
| `vllm/v1/core/kv_cache_manager.py` | `allocate_slots()` | 253 | 分配 KV 块 |
| | `get_computed_blocks()` | 202 | 前缀缓存查找 |
| | `free()` | 498 | 释放 KV 块 |
| `vllm/v1/core/block_pool.py` | `get_new_blocks()` | 371 | 从空闲池分配 |
| | `free_blocks()` | 466 | 释放块到空闲池 |
| | `touch()` | 444 | 前缀缓存命中 |
| | `cache_full_blocks()` | 240 | 缓存完整块 |
| `vllm/v1/core/single_type_kv_cache_manager.py` | `find_longest_cache_hit()` | 各子类 | 缓存命中查找 |
| | `allocate_new_computed_blocks()` | 170 | 附加缓存块 |
| `vllm/v1/core/kv_cache_coordinator.py` | `find_longest_cache_hit()` | 263 | 协调器缓存查找 |
| `vllm/v1/core/kv_cache_utils.py` | `KVCacheBlock` | 116 | 块数据结构 |
| | `hash_block_tokens()` | 633 | 链式块哈希 |

### 10.3 Engine Core 相关

| 文件 | 关键方法 | 行号 | 用途 |
|------|----------|------|------|
| `vllm/v1/engine/core.py` | `EngineCore.__init__()` | 97 | 引擎初始化 |
| | `step()` | 483 | 核心调度循环 |
| | `step_with_batch_queue()` | 538 | 流水线并行循环 |
| | `add_request()` | 354 | 添加请求 |
| | `_initialize_kv_caches()` | 256 | KV 缓存初始化 |
| `vllm/v1/engine/core_client.py` | `EngineCoreClient` | 70 | 客户端基类 |
| | `MPClient` | 461 | 多进程客户端 |
| | `AsyncMPClient` | 919 | 异步多进程客户端 |

### 10.4 Model Runner 相关

| 文件 | 关键方法 | 行号 | 用途 |
|------|----------|------|------|
| `vllm/v1/worker/gpu_model_runner.py` | `execute_model()` | 3963 | 模型前向传播 |
| | `sample_tokens()` | 4427 | Token 采样 |
| | `_update_states()` | 1120 | 更新批处理状态 |
| | `_prepare_inputs()` | 1867 | 准备 GPU 输入 |
| | `_model_forward()` | 3676 | 模型 forward |

### 10.5 KV Transfer 相关

| 文件 | 关键方法 | 行号 | 用途 |
|------|----------|------|------|
| `vllm/distributed/kv_transfer/kv_connector/v1/base.py` | `KVConnectorBase_V1` | 171 | 连接器基类 |
| | `get_num_new_matched_tokens()` | 454 | 查询远程缓存 |
| | `start_load_kv()` | 293 | 开始加载 KV |
| | `save_kv_layer()` | 325 | 保存层 KV |
| `vllm/distributed/kv_transfer/kv_connector/factory.py` | `KVConnectorFactory` | - | 连接器工厂 |

---

## 11. 核心变量详解

> 本章详细列出调度器及其周边文件中的所有核心变量，包括类型、定义位置、含义和相互关系。

### 11.1 Scheduler 实例变量

#### 配置类变量

| 变量 | 类型 | 行号 | 含义 |
|------|------|------|------|
| `vllm_config` | `VllmConfig` | 99 | 顶层 vLLM 配置对象 |
| `scheduler_config` | `SchedulerConfig` | 101 | 调度器专用配置（最大序列数、分块预填充策略等） |
| `cache_config` | `CacheConfig` | 103 | 缓存配置（前缀缓存开关、块大小、GPU 块数） |
| `lora_config` | `LoRAConfig \| None` | 105 | LoRA 适配器配置，未启用时为 None |
| `kv_cache_config` | `KVCacheConfig` | 107 | KV 缓存拓扑（组、规格、张量布局） |
| `parallel_config` | `ParallelConfig` | 111 | 并行配置（PP、TP、DCP、PCP） |
| `log_stats` | `bool` | 113 | 是否收集和记录调度器统计信息 |

#### 调度约束变量

| 变量 | 类型 | 行号 | 含义 |
|------|------|------|------|
| `max_num_running_reqs` | `int` | 143 | 同时处于 RUNNING 状态的最大请求数，来自 `scheduler_config.max_num_seqs` |
| `max_num_scheduled_tokens` | `int` | 145-149 | 单步可调度的最大 token 总数，即 **token_budget 的上限**。未设置时回退到 `max_num_batched_tokens` |
| `max_model_len` | `int` | 151 | 模型支持的最大序列长度，用于截断 `num_new_tokens` |
| `block_size` | `int` | 197 | 每个 KV 缓存块的 token 数 |
| `dcp_world_size` | `int` | 199 | 解码上下文并行的世界大小 |
| `pcp_world_size` | `int` | 201 | 预填充上下文并行的世界大小 |

#### 请求追踪变量

| 变量 | 类型 | 行号 | 含义 |
|------|------|------|------|
| `requests` | `dict[str, Request]` | 206 | 全局注册表：request_id → Request。所有活跃（未完成）请求都在此 |
| `waiting` | `RequestQueue` | 219 | 等待调度的请求队列（WAITING 或 PREEMPTED 状态） |
| `skipped_waiting` | `RequestQueue` | 223 | 被跳过的等待请求队列（因阻塞状态：等待语法编译、远程 KV、流式输入） |
| `running` | `list[Request]` | 225 | 当前处于 RUNNING 状态的请求列表 |
| `finished_req_ids` | `set[str]` | 234 | 上一步和当前步之间完成的请求 ID 集合，每步刷新通知 Worker 释放缓存 |
| `prev_step_scheduled_req_ids` | `set[str]` | 139 | 上一步调度的请求 ID，用于 `_make_cached_request_data` 决定是否发送完整 `all_token_ids` |

#### KV 连接器变量

| 变量 | 类型 | 行号 | 含义 |
|------|------|------|------|
| `connector` | `KVConnectorBase_V1 \| None` | 164 | KV 连接器，用于 P/D 分离和卸载 |
| `connector_prefix_cache_stats` | `PrefixCacheStats \| None` | 165 | 连接器前缀缓存命中/未命中统计 |
| `recompute_kv_load_failures` | `bool` | 166 | 外部 KV 加载失败时是否重算（vs 中止请求） |
| `finished_recving_kv_req_ids` | `set[str]` | 241 | 完成异步 KV 接收的请求 |
| `failed_recving_kv_req_ids` | `set[str]` | 242 | 异步 KV 接收失败的请求 |
| `ec_connector` | `ECConnector \| None` | 187 | 编码器缓存连接器，用于分布式编码器缓存 |

#### 推测解码变量

| 变量 | 类型 | 行号 | 含义 |
|------|------|------|------|
| `use_eagle` | `bool` | 280 | 是否使用 EAGLE 推测解码 |
| `num_spec_tokens` | `int` | 282 | 每请求每步的推测 token 数 |
| `num_lookahead_tokens` | `int` | 282 | 需要分配 KV slot 的前瞻 token 数。EAGLE/draft-model 等于 `num_spec_tokens`；DFlash 等于 `num_spec_tokens + 1` |

#### 其他重要变量

| 变量 | 类型 | 行号 | 含义 |
|------|------|------|------|
| `kv_cache_manager` | `KVCacheManager` | 305-317 | 核心 KV 缓存管理器，处理块分配、前缀缓存、驱逐 |
| `use_pp` | `bool` | 326 | 是否启用流水线并行（PP > 1） |
| `scheduler_reserve_full_isl` | `bool` | 330-332 | 调度器是否为完整输入序列长度预留块（准入控制） |
| `has_mamba_layers` | `bool` | 336 | 模型是否有 Mamba (SSM) 层 |
| `needs_kv_cache_zeroing` | `bool` | 338 | 新分配的 KV 缓存块是否需要清零 |
| `_pause_state` | `PauseState` | 375 | 调度暂停控制：UNPAUSED（正常）、PAUSED_NEW（不调度新请求）、PAUSED_ALL（完全暂停）。用于 DP 协调 |

### 11.2 schedule() 局部变量

#### 请求分类列表

| 变量 | 类型 | 行号 | 含义 |
|------|------|------|------|
| `scheduled_new_reqs` | `list[Request]` | 449 | 首次进入 RUNNING 的请求（从 WAITING 新来） |
| `scheduled_resumed_reqs` | `list[Request]` | 450 | 预抢占后恢复的请求（PREEMPTED → RUNNING） |
| `scheduled_running_reqs` | `list[Request]` | 451 | 继续处于 RUNNING 的请求（典型的 decode 步骤） |
| `preempted_reqs` | `list[Request]` | 452 | 本步因 KV 缓存压力被抢占的请求 |

#### 预算和记账变量

| 变量 | 类型 | 行号 | 含义 |
|------|------|------|------|
| `token_budget` | `int` | 456 | 本步剩余的 token 预算。初始化为 `max_num_scheduled_tokens`，每个请求调度后减去 `num_new_tokens`。PAUSED_ALL 时设为 0 |
| `total_num_scheduled_tokens` | `int` | 1029 | 所有 `num_scheduled_tokens.values()` 的总和，验证不超过 `max_num_scheduled_tokens` |
| `num_scheduled_tokens` | `dict[str, int]` | 455 | request_id → 本步调度的 token 数。调度的核心输出 |
| `req_to_new_blocks` | `dict[str, KVCacheBlocks]` | 454 | request_id → 新分配的 KV 缓存块 |

#### 编码器变量

| 变量 | 类型 | 行号 | 含义 |
|------|------|------|------|
| `scheduled_encoder_inputs` | `dict[str, list[int]]` | 463 | request_id → 本步需要处理的编码器输入索引列表 |
| `encoder_compute_budget` | `int` | 464 | 剩余编码器 token 预算，初始化为 `max_num_encoder_input_tokens` |
| `encoder_inputs_to_schedule` | `list[int] \| None` | 526 | 当前请求的编码器输入索引 |

#### Phase 1 每请求变量

| 变量 | 类型 | 行号 | 含义 |
|------|------|------|------|
| `num_new_tokens` | `int` | 503-507 | **核心变量**。本步需要计算的新 token 数。公式：`num_tokens_with_spec + num_output_placeholders - num_computed_tokens`。被 `token_budget`、`max_model_len`、`long_prefill_token_threshold` 截断 |
| `new_blocks` | `KVCacheBlocks \| None` | 580 | `allocate_slots()` 的返回值。None 表示分配失败（触发预抢占） |
| `num_scheduled_spec_tokens` | `int` | 650-655 | 实际调度的推测 token 数：`num_new_tokens + num_computed_tokens - num_tokens - num_output_placeholders` |
| `scheduled_loras` | `set[int]` | 686 | 当前已调度的 LoRA 适配器 ID 集合，检查不超过 `max_loras` |

#### Phase 2 每请求变量

| 变量 | 类型 | 行号 | 含义 |
|------|------|------|------|
| `num_external_computed_tokens` | `int` | 756 | 外部缓存的 token 数（由 KV 连接器/远程节点缓存） |
| `load_kv_async` | `bool` | 757 | 是否从远程异步加载 KV 数据 |
| `new_computed_blocks` | `KVCacheBlocks` | 766 | 本地前缀缓存命中的块 |
| `num_new_local_computed_tokens` | `int` | 766 | 本地前缀缓存命中的 token 数 |
| `num_computed_tokens` | `int` | 799-800 | 总已计算 token 数：`num_new_local_computed_tokens + num_external_computed_tokens` |
| `effective_lookahead_tokens` | `int` | 890-892 | 实际前瞻 token 数。异步 KV 加载 + EAGLE 时为 0（避免块不匹配） |

### 11.3 Request 类核心字段

#### 标识和配置字段

| 字段 | 类型 | 行号 | 含义 |
|------|------|------|------|
| `request_id` | `str` | 81 | 请求唯一标识 |
| `client_index` | `int` | 82 | 提交此请求的客户端索引（多引擎场景） |
| `priority` | `int` | 83 | 调度优先级（值越小优先级越高）。优先级调度策略中使用 |
| `sampling_params` | `SamplingParams \| None` | 84 | 采样参数（temperature、top_k、top_p、max_tokens、stop tokens 等） |
| `arrival_time` | `float` | 95 | 请求到达时间，用于 FCFS 策略的平局打破 |
| `max_tokens` | `int` | 106-110 | 最大输出 token 数。池化模型为 1，生成模型为 `sampling_params.max_tokens` |

#### Token/序列字段

| 字段 | 类型 | 行号 | 含义 |
|------|------|------|------|
| `prompt_token_ids` | `list[int] \| None` | 121 | Prompt 的 token IDs |
| `num_prompt_tokens` | `int` | 130-132 | Prompt 的 token 数 |
| `_output_token_ids` | `list[int]` | 133 | 内部可变列表：已生成的输出 token IDs |
| `_all_token_ids` | `list[int]` | 134-138 | 所有 token IDs（prompt + output） |
| `num_tokens` | `int` (property) | 270-271 | 总 token 数：`len(all_token_ids)` = prompt + output |
| `num_tokens_with_spec` | `int` (property) | 274-275 | 包含推测 token 的总数：`len(all_token_ids) + len(spec_token_ids)` |

#### 核心调度字段

| 字段 | 类型 | 行号 | 含义 |
|------|------|------|------|
| `num_computed_tokens` | `int` | 159 | **核心变量**。已被模型前向计算过的 token 数。调度器用 `num_tokens_with_spec - num_computed_tokens` 决定本轮调度多少 token。预抢占时重置为 0。在 `_update_after_schedule` 中推进 |
| `num_output_placeholders` | `int` | 146 | 异步调度预留的输出 token 占位符数。调度器在模型实际产出前就预留了这些 token 的空间。实际输出到达后递减 |
| `async_tokens_to_discard` | `int` | 147 | 异步调度占位符与实际输出不匹配时需要丢弃的 token 数 |
| `spec_token_ids` | `list[int]` | 153 | 推测解码的草稿 token IDs。调度器将这些 token 送入主模型验证。调度后清空；由 `update_draft_token_ids` 重新填充 |
| `is_prefill_chunk` | `bool` | 179 | 标记当前是否为分块 prefill 的中间块（非最后一块）。为 True 时输出不应发送给客户端。在 `_update_after_schedule` 中根据 `num_computed_tokens < num_tokens + num_output_placeholders` 设置 |
| `status` | `RequestStatus` | 97 | 当前状态：WAITING → RUNNING → FINISHED/PREEMPTED |
| `num_preemptions` | `int` | 192 | 被抢占次数。用于指标和前缀缓存统计 |

#### 前缀缓存字段

| 字段 | 类型 | 行号 | 含义 |
|------|------|------|------|
| `block_hashes` | `list[BlockHash]` | 202 | 链式块哈希列表。每个块的哈希包含前驱块的哈希，形成 Merkle 链。KV 缓存管理器通过比较这些哈希找到最长缓存命中 |
| `skip_reading_prefix_cache` | `bool` | 209 | 是否跳过前缀缓存查找（需要 prompt logprobs 时设置） |

#### KV 传输字段

| 字段 | 类型 | 行号 | 含义 |
|------|------|------|------|
| `kv_transfer_params` | `dict[str, Any] \| None` | 102 | P/D 分离的 KV 传输参数 |

### 11.4 num_new_tokens 计算详解

`num_new_tokens` 是调度器中最核心的变量，决定每个请求本轮计算多少 token。

```
┌─────────────────────────────────────────────────────────────────┐
│  num_new_tokens 的计算过程                                       │
└─────────────────────────────────────────────────────────────────┘

Phase 1 (RUNNING 请求):
┌─────────────────────────────────────────────────────────────────┐
│ ① 原始值:                                                       │
│    num_new_tokens = num_tokens_with_spec                        │
│                    + num_output_placeholders                    │
│                    - num_computed_tokens                         │
│                                                                 │
│    其中:                                                         │
│    num_tokens_with_spec = len(all_token_ids) + len(spec_tokens) │
│    num_output_placeholders = 异步调度预留的占位符                  │
│    num_computed_tokens = 已经被模型计算过的 token 数               │
│                                                                 │
│ ② 截断到 long_prefill_token_threshold:                          │
│    if 0 < threshold < num_new_tokens:                           │
│        num_new_tokens = threshold                               │
│                                                                 │
│ ③ 截断到 token_budget:                                          │
│    num_new_tokens = min(num_new_tokens, token_budget)           │
│                                                                 │
│ ④ 截断到 max_model_len:                                         │
│    num_new_tokens = min(num_new_tokens,                         │
│                         max_model_len - 1 - num_computed_tokens)│
└─────────────────────────────────────────────────────────────────┘

Phase 2 (WAITING 请求):
┌─────────────────────────────────────────────────────────────────┐
│ ① 原始值:                                                       │
│    num_new_tokens = request.num_tokens - num_computed_tokens    │
│    # num_computed_tokens = 本地缓存 + 远程缓存                    │
│                                                                 │
│ ② 分块预填充截断:                                                │
│    if enable_chunked_prefill:                                    │
│        num_new_tokens = min(num_new_tokens, token_budget)       │
│    elif num_new_tokens > token_budget:                          │
│        break  # 不分块，停止调度                                  │
└─────────────────────────────────────────────────────────────────┘

示例:
  请求 A (decode):  num_tokens=100, num_computed=99  → num_new=1
  请求 B (prefill): num_tokens=1000, num_computed=0  → num_new=1000
  请求 C (chunk):   num_tokens=1000, num_computed=500 → num_new=500
  token_budget = 2048
  → A: 1, B: 1000, C: 500, 剩余 budget = 547
```

### 11.5 token_budget 机制详解

```
┌─────────────────────────────────────────────────────────────────┐
│  token_budget 的生命周期                                         │
└─────────────────────────────────────────────────────────────────┘

初始化:
  token_budget = max_num_scheduled_tokens  (例如 8192)

特殊状态:
  if _pause_state == PauseState.PAUSED_ALL:
      token_budget = 0  # 完全暂停，不调度任何请求

Phase 1 消耗:
  for request in running:
      num_new_tokens = ... (计算如上)
      # 调度成功后
      token_budget -= num_new_tokens

Phase 2 消耗:
  for request in waiting:
      num_new_tokens = ... (计算如上)
      # 调度成功后
      token_budget -= num_new_tokens

约束:
  token_budget >= 0  (调度结束时验证)
  total_num_scheduled_tokens <= max_num_scheduled_tokens  (验证)

示例:
  token_budget = 8192
  Phase 1: A(1) + B(1) + C(500) = 502 → budget = 7690
  Phase 2: D(2000) + E(3000) + F(2690) = 7690 → budget = 0
  总计: 8192 tokens
```

### 11.6 num_scheduled_tokens 详解

```
┌─────────────────────────────────────────────────────────────────┐
│  num_scheduled_tokens: dict[str, int]                           │
│  调度的核心输出，记录每个请求本轮调度了多少 token                   │
└─────────────────────────────────────────────────────────────────┘

结构:
  {
      "request-001": 1,      # decode: 1 个 token
      "request-002": 1,      # decode: 1 个 token
      "request-003": 500,    # prefill chunk: 500 个 token
      "request-004": 2000,   # prefill: 2000 个 token
  }

用途:
  ① 传给 ModelRunner: 告诉 GPU 每个请求算多少 token
  ② 构建 SchedulerOutput: 填充 num_scheduled_tokens 字段
  ③ 更新 num_computed_tokens: _update_after_schedule() 中推进
  ④ 统计: total_num_scheduled_tokens = sum(values())

在 _update_after_schedule() 中:
  for req_id, num_tokens in num_scheduled_tokens.items():
      request.num_computed_tokens += num_tokens
```

### 11.7 KVCacheManager.allocate_slots() 参数详解

```
┌─────────────────────────────────────────────────────────────────┐
│  allocate_slots() 的参数和内部变量                                │
└─────────────────────────────────────────────────────────────────┘

参数:
┌──────────────────────────┬──────────────────────────────────────┐
│ request                  │ 要分配块的请求                        │
│ num_new_tokens           │ 本步要计算的新 token 数                │
│ num_new_computed_tokens  │ 本地前缀缓存命中的 token 数            │
│ new_computed_blocks      │ 前缀缓存命中的块                      │
│ num_lookahead_tokens     │ 推测解码的前瞻 token 数                │
│ num_external_computed_   │ 外部缓存的 token 数 (P/D 分离)        │
│   tokens                 │                                      │
│ delay_cache_blocks       │ 是否延迟缓存 (异步 KV 传输中)          │
│ num_encoder_tokens       │ 交叉注意力的编码器 token 数            │
│ full_sequence_must_fit   │ 准入门控: 只有完整序列能放入才分配      │
└──────────────────────────┴──────────────────────────────────────┘

内部变量:
┌──────────────────────────┬──────────────────────────────────────┐
│ num_local_computed_      │ request.num_computed_tokens          │
│   tokens                 │   + num_new_computed_tokens          │
├──────────────────────────┼──────────────────────────────────────┤
│ total_computed_tokens    │ min(num_local + num_external,        │
│                          │     max_model_len)                   │
├──────────────────────────┼──────────────────────────────────────┤
│ num_tokens_main_model    │ total_computed + num_new_tokens      │
├──────────────────────────┼──────────────────────────────────────┤
│ num_tokens_need_slot     │ min(num_tokens_main_model            │
│                          │     + num_lookahead, max_model_len)  │
├──────────────────────────┼──────────────────────────────────────┤
│ num_blocks_to_allocate   │ coordinator.get_num_blocks_to_       │
│                          │   allocate() 的返回值                 │
├──────────────────────────┼──────────────────────────────────────┤
│ num_tokens_to_cache      │ min(total_computed + num_new,        │
│                          │     request.num_tokens)              │
│                          │ # 排除未验证的草稿 token               │
└──────────────────────────┴──────────────────────────────────────┘

块布局:
  | < comp > | < new_comp > | < ext_comp > | < new > | < lookahead > |
     已计算      前缀缓存命中    外部缓存       新计算      推测 token
```

### 11.8 KVCacheBlock 字段详解

| 字段 | 类型 | 含义 |
|------|------|------|
| `block_id` | `int` | 物理块 ID (0 到 num_gpu_blocks-1)，GPU 显存中的索引 |
| `ref_cnt` | `int` | 引用计数。> 0 表示被一个或多个请求使用；= 0 表示空闲可驱逐 |
| `_block_hash` | `BlockHashWithGroupId \| None` | 内容哈希（块哈希 + 组 ID）。块满且缓存后设置，只能设置一次 |
| `prev_free_block` | `KVCacheBlock \| None` | 空闲双向链表前驱指针 |
| `next_free_block` | `KVCacheBlock \| None` | 空闲双向链表后继指针 |
| `is_null` | `bool` | 是否为空块占位符（SWA 填充用），ref_cnt 不维护 |

**引用计数变化：**
```
分配: ref_cnt 0 → 1 (从空闲队列弹出)
Touch: ref_cnt++ (前缀缓存命中，从空闲队列移除)
释放: ref_cnt-- (归零后放回空闲队列尾部)
```

### 11.9 SchedulerOutput 字段详解

| 字段 | 类型 | 含义 |
|------|------|------|
| `scheduled_new_reqs` | `list[NewRequestData]` | 首次调度的请求，携带完整数据（prompt tokens、采样参数、块 IDs） |
| `scheduled_cached_reqs` | `CachedRequestData` | 已知请求的差量数据（新块 IDs、新 token IDs），减少通信量 |
| `num_scheduled_tokens` | `dict[str, int]` | request_id → 本步调度的 token 数 |
| `total_num_scheduled_tokens` | `int` | 所有请求的 token 总数 |
| `scheduled_spec_decode_tokens` | `dict[str, list[int]]` | request_id → 推测草稿 token IDs |
| `scheduled_encoder_inputs` | `dict[str, list[int]]` | request_id → 编码器输入索引 |
| `num_common_prefix_blocks` | `list[int]` | 每 KV 缓存组的公共前缀块数（级联注意力用） |
| `preempted_req_ids` | `set[str]` | 本步被抢占的请求 ID |
| `finished_req_ids` | `set[str]` | 上一步和当前步之间完成的请求 ID |
| `new_block_ids_to_zero` | `list[int] \| None` | 需要清零的新块 ID（防止旧数据污染） |
| `kv_connector_metadata` | `KVConnectorMetadata \| None` | KV 连接器元数据（P/D 分离） |

### 11.10 update_from_output() 变量详解

| 变量 | 类型 | 含义 |
|------|------|------|
| `sampled_token_ids` | `list[list[int]]` | 每请求的生成 token IDs，外层按 req_index 索引 |
| `generated_token_ids` | `list[int]` | 当前请求的生成 token IDs |
| `num_draft_tokens` | `int` | 提交验证的草稿 token 数 |
| `num_accepted` | `int` | 被接受的草稿 token 数：`len(generated_token_ids) - 1` |
| `num_rejected` | `int` | 被拒绝的草稿 token 数：`num_draft_tokens - num_accepted` |
| `stopped` | `bool` | 请求是否停止（stop token、max length 等） |
| `finish_reason` | `FinishReason \| None` | 停止原因（STOP、LENGTH、ABORT、ERROR、REPETITION） |
| `stopped_running_reqs` | `set[Request]` | 在 RUNNING 状态停止的请求 |
| `stopped_preempted_reqs` | `set[Request]` | 在 PREEMPTED 状态停止的请求 |
| `outputs` | `dict[int, list[EngineCoreOutput]]` | 按 client_index 分组的累积输出 |

### 11.11 RequestStatus 枚举详解

```
┌─────────────────────────────────────────────────────────────────┐
│  RequestStatus 生命周期                                          │
└─────────────────────────────────────────────────────────────────┘

正常路径:
  WAITING  ──→  RUNNING  ──→  FINISHED_*
  新请求入队     被调度执行      正常/异常结束

预抢占路径:
  RUNNING  ──→  PREEMPTED  ──→  WAITING  ──→  RUNNING ...
  正在执行       被抢占释放KV     重新排队       恢复执行

额外等待状态:
  WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR  (等待语法编译)
  WAITING_FOR_REMOTE_KVS                 (等待远程 KV 传输)
  WAITING_FOR_STREAMING_REQ              (等待流式输入)

结束状态 (值 > PREEMPTED 的都是结束):
  FINISHED_STOPPED       遇到 stop token
  FINISHED_LENGTH_CAPPED 达到 max_tokens
  FINISHED_ABORTED       客户端取消
  FINISHED_IGNORED       prompt 超长
  FINISHED_ERROR         执行出错
  FINISHED_REPETITION    重复惩罚
```

### 11.12 变量交互关系图

```
┌─────────────────────────────────────────────────────────────────┐
│  核心变量交互关系                                                │
└─────────────────────────────────────────────────────────────────┘

Request 到达
    │
    ▼
add_request() → requests[req_id] = request
              → waiting.append(request)
              → request.status = WAITING

schedule() Phase 1 (RUNNING):
    │
    ├── num_new_tokens = num_tokens_with_spec + num_output_placeholders
    │                    - num_computed_tokens
    │
    ├── token_budget -= num_new_tokens
    │
    ├── blocks = allocate_slots(request, num_new_tokens)
    │     │
    │     ├── num_blocks_to_allocate = coordinator.get_num_blocks_to_allocate()
    │     ├── coordinator.allocate_new_computed_blocks()  # 前缀缓存
    │     ├── coordinator.allocate_new_blocks()           # 新块
    │     └── coordinator.cache_blocks()                  # 缓存
    │
    ├── if blocks is None → _preempt_request()
    │     ├── kv_cache_manager.free(request)
    │     ├── request.num_computed_tokens = 0
    │     ├── request.status = PREEMPTED
    │     └── waiting.prepend_request(request)
    │
    └── num_scheduled_tokens[req_id] = num_new_tokens

schedule() Phase 2 (WAITING):
    │
    ├── computed_blocks, num_computed = get_computed_blocks(request)
    │     └── coordinator.find_longest_cache_hit(block_hashes)
    │
    ├── remote_tokens = connector.get_num_new_matched_tokens()
    │
    ├── num_new_tokens = num_tokens - num_computed - remote_tokens
    │
    ├── token_budget -= num_new_tokens
    │
    ├── blocks = allocate_slots(request, num_new_tokens, ...)
    │
    ├── request.status = RUNNING
    │
    └── running.append(request)

_update_after_schedule():
    │
    └── for req_id, num_tokens in num_scheduled_tokens.items():
            request.num_computed_tokens += num_tokens
            request.is_prefill_chunk = (num_computed < num_tokens + placeholders)

execute_model() → GPU 前向传播

update_from_output():
    │
    ├── generated_token_ids = sampled_token_ids[req_index]
    │
    ├── num_rejected = num_draft_tokens - num_accepted
    │   request.num_computed_tokens -= num_rejected  # 回退
    │
    ├── request.append_output_token_ids(token_ids)
    │
    ├── stopped, reason = check_stop(request)
    │
    └── if stopped:
            kv_cache_manager.free(request)
            del requests[req_id]
            finished_req_ids.add(req_id)
```
