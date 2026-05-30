# vLLM 架构深度解析

> 本文档基于 vLLM v1 架构，全面剖析 vLLM 的系统设计、核心组件、初始化流程、调度机制、执行引擎以及关键优化技术。

---

## 目录

- [第一章 系统架构概述](#第一章-系统架构概述)
- [第二章 项目目录结构](#第二章-项目目录结构)
- [第三章 核心组件关系](#第三章-核心组件关系)
- [第四章 vLLM v1 初始化过程](#第四章-vllm-v1-初始化过程)
- [第五章 调度器详解](#第五章-调度器详解)
- [第六章 Worker 与 Executor 详解](#第六章-worker-与-executor-详解)
- [第七章 KV Cache 与内存管理](#第七章-kv-cache-与内存管理)
- [第八章 整体调用流程](#第八章-整体调用流程)
- [第九章 API 服务器与入口点](#第九章-api-服务器与入口点)
- [第十章 模型加载与配置系统](#第十章-模型加载与配置系统)
- [第十一章 关键优化技术总结](#第十一章-关键优化技术总结)

---

## 第一章 系统架构概述

### 1.1 vLLM 是什么

vLLM 是一个高吞吐量、低延迟的 LLM 推理和服务引擎。其核心创新是 **PagedAttention** 技术，借鉴操作系统虚拟内存的分页机制，将 KV Cache 分成固定大小的块进行管理，避免了传统连续内存分配带来的内存碎片和浪费问题。

### 1.2 三层架构

vLLM v1 采用清晰的三层架构设计：

```
┌─────────────────────────────────────────────────────────┐
│                    Layer 1: 前端层                        │
│         LLMEngine (同步) / AsyncLLM (异步)                │
│         对外提供用户友好的 API 接口                         │
└──────────────────────┬──────────────────────────────────┘
                       │ EngineCoreClient (IPC 传输层)
                       │ ZMQ / 进程内直接调用
┌──────────────────────▼──────────────────────────────────┐
│                    Layer 2: 核心引擎层                     │
│                    EngineCore                             │
│         负责调度 (Scheduler) 和执行 (Executor)              │
└──────────────────────┬──────────────────────────────────┘
                       │ collective_rpc / 共享内存 MQ
                       │ NCCL (分布式通信)
┌──────────────────────▼──────────────────────────────────┐
│                    Layer 3: 工作层                         │
│         Worker → GPUModelRunner → Model                  │
│         负责模型加载、KV Cache 管理、前向推理                 │
└─────────────────────────────────────────────────────────┘
```

### 1.3 关键设计原则

1. **分层解耦**：前端 API、调度逻辑、模型执行完全分离，各层可独立演进
2. **进程隔离**：EngineCore 可运行在独立进程中，通过 ZMQ 通信，避免 GIL 限制
3. **Token 级调度**：不区分 prefill/decode 阶段，统一以 token 为单位调度
4. **分页内存管理**：KV Cache 使用分页机制，支持前缀缓存、滑动窗口回收
5. **可插拔后端**：注意力后端、量化方法、模型加载器均可通过插件扩展

---

## 第二章 项目目录结构

### 2.1 顶层目录

| 目录 | 用途 |
|------|------|
| `benchmarks/` | 性能基准测试脚本（延迟、吞吐量、服务、注意力、块池等） |
| `cmake/` | CMake 构建配置，用于原生 C++/CUDA 扩展 |
| `csrc/` | 原生 C++/CUDA 源码：注意力 kernel、MoE、量化、自定义 all-reduce、内存分配器 |
| `docker/` | 多平台 Dockerfile：CUDA、ROCm、CPU、TPU、XPU 等 |
| `docs/` | 文档源码（Markdown） |
| `examples/` | 示例脚本：基础用法、部署、特性演示等 |
| `requirements/` | 依赖文件，按平台/用途拆分 |
| `rust/` | Rust 前端组件：聊天渲染器、工具解析器、推理解析器 |
| `tests/` | 综合测试套件 |
| `tools/` | 构建/安装辅助脚本 |

### 2.2 `vllm/` 源码目录

这是 vLLM 的核心 Python 源码目录，包含以下关键子目录：

| 目录 | 用途 |
|------|------|
| `vllm/v1/` | **v1 引擎架构**（当前主力实现） |
| `vllm/engine/` | v0 引擎的薄壳/别名层，重定向到 v1 |
| `vllm/config/` | 所有配置数据类：`VllmConfig`、`ModelConfig`、`CacheConfig`、`ParallelConfig` 等 |
| `vllm/model_executor/` | 模型执行框架：模型加载器、~291 个模型定义、注意力层、线性层、MoE 层等 |
| `vllm/entrypoints/` | 所有用户入口：`LLM` 类、API 服务器、OpenAI 兼容服务器、CLI 等 |
| `vllm/attention/` | 注意力机制抽象和后端实现 |
| `vllm/distributed/` | 分布式执行：并行状态管理、通信操作、KV 传输 |
| `vllm/lora/` | LoRA 适配器支持 |
| `vllm/spec_decode/` | 推测解码实现 |
| `vllm/quantization/` | 量化后端和方法（29 种） |
| `vllm/inputs/` | 输入处理：`PromptType`、`TextPrompt`、`TokensPrompt` |
| `vllm/multimodal/` | 多模态输入处理：音频、图像、视频 |
| `vllm/platforms/` | 平台抽象层：CUDA、ROCm、CPU、TPU、XPU |
| `vllm/kernels/` | Python 内核包装器：Triton 内核、Helion 内核 |
| `vllm/tokenizers/` | 分词器抽象层 |
| `vllm/transformers_utils/` | HuggingFace Transformers 工具 |
| `vllm/utils/` | 通用工具 |

### 2.3 `vllm/v1/` 子目录结构

这是 vLLM v1 引擎的核心实现：

| 目录 | 用途 |
|------|------|
| `v1/engine/` | v1 引擎实现：`LLMEngine`、`AsyncLLM`、`EngineCore`、`EngineCoreClient` |
| `v1/core/` | 核心调度和 KV Cache 管理：`Scheduler`、`KVCacheManager`、`BlockPool` |
| `v1/core/sched/` | 调度器接口、异步调度器、请求队列、输出数据结构 |
| `v1/executor/` | 执行器实现：单进程、多进程、Ray 执行器 |
| `v1/worker/` | Worker 实现：GPU Worker、GPUModelRunner、CPU Worker |
| `v1/attention/` | v1 注意力抽象和后端 |
| `v1/sample/` | Token 采样：采样器、拒绝采样、logits 处理器 |
| `v1/spec_decode/` | 推测解码：EAGLE、Medusa、N-gram 等 |
| `v1/kv_offload/` | KV Cache 卸载到 CPU/主机内存 |
| `v1/structured_output/` | 结构化/语法约束输出 |
| `v1/metrics/` | 指标收集和报告 |
| `v1/pool/` | 池化模型支持 |

### 2.4 `vllm/engine/` 目录（v0 壳层）

这是一个薄壳层，将调用重定向到 v1 实现：

| 文件 | 用途 |
|------|------|
| `llm_engine.py` | 别名：`LLMEngine = V1LLMEngine` |
| `async_llm_engine.py` | 别名：`AsyncLLMEngine = AsyncLLM` |
| `arg_utils.py` | `EngineArgs` 和 `AsyncEngineArgs` 数据类，定义所有 CLI 参数 |
| `protocol.py` | `EngineClient` 抽象基类（协议），定义所有引擎客户端必须实现的接口 |

### 2.5 `vllm/v1/engine/` 目录（v1 实现）

| 文件 | 用途 |
|------|------|
| `__init__.py` | 定义核心数据结构：`EngineCoreRequest`、`EngineCoreOutput`、`EngineCoreOutputs` |
| `llm_engine.py` | `LLMEngine` —— 同步引擎包装器 |
| `async_llm.py` | `AsyncLLM` —— 异步引擎客户端（API 服务器使用） |
| `core.py` | `EngineCore` 和 `EngineCoreProc` —— 核心调度/执行循环 |
| `core_client.py` | `EngineCoreClient` 抽象基类及具体实现：`InprocClient`、`SyncMPClient`、`AsyncMPClient` |
| `input_processor.py` | `InputProcessor` —— 将用户输入转换为 `EngineCoreRequest` |
| `output_processor.py` | `OutputProcessor` —— 将引擎输出转换为 `RequestOutput` |
| `detokenizer.py` | 增量反分词器 |
| `coordinator.py` | `DPCoordinator` —— 数据并行协调器 |
| `tensor_ipc.py` | 零拷贝张量 IPC |
| `parallel_sampling.py` | 并行采样（n>1）管理 |

---

## 第三章 核心组件关系

### 3.1 组件依赖关系图

```
用户代码
  │
  ├─── LLM (离线推理)
  │      │
  │      └─── LLMEngine (同步前端)
  │             │
  │             ├── InputProcessor (输入处理)
  │             ├── OutputProcessor (输出处理)
  │             └── EngineCoreClient
  │                    │
  │                    ├── InprocClient (进程内) ──→ EngineCore
  │                    └── SyncMPClient (ZMQ)   ──→ EngineCoreProc
  │
  └─── API Server (在线服务)
         │
         └─── AsyncLLM (异步前端)
                │
                ├── InputProcessor
                ├── OutputProcessor
                └── AsyncMPClient (ZMQ) ──→ EngineCoreProc
                                              │
                                              ├── Scheduler
                                              │    ├── KVCacheManager
                                              │    │    ├── KVCacheCoordinator
                                              │    │    └── BlockPool
                                              │    └── RequestQueue (FCFS/Priority)
                                              │
                                              └── Executor
                                                   │
                                                   ├── UniProcExecutor
                                                   │    └── Worker → GPUModelRunner → Model
                                                   │
                                                   ├── MultiprocExecutor
                                                   │    └── [WorkerProc × N] → GPUModelRunner → Model
                                                   │
                                                   └── RayDistributedExecutor
                                                        └── [RayWorkerWrapper × N] → GPUModelRunner → Model
```

### 3.2 数据流

```
用户请求 (HTTP/Python API)
    │
    ▼
InputProcessor.process_inputs()
    │  tokenization, 验证, 多模态处理
    ▼
EngineCoreRequest (msgspec struct)
    │
    ▼  (ZMQ / 进程内)
EngineCore.add_request()
    │
    ▼
Scheduler.add_request() → waiting 队列
    │
    ▼
Scheduler.schedule() → SchedulerOutput
    │  分配 KV 块, 决定 token 预算
    ▼
Executor.execute_model(scheduler_output)
    │
    ▼
Worker.execute_model()
    │  GPUModelRunner: 准备输入, 构建注意力元数据, 前向传播
    ▼
Model.forward()
    │
    ▼
Worker.sample_tokens()
    │  采样, 反分词
    ▼
ModelRunnerOutput
    │
    ▼
Scheduler.update_from_output()
    │  更新请求状态, 检查停止条件
    ▼
EngineCoreOutputs
    │
    ▼  (ZMQ / 进程内)
OutputProcessor.process_outputs()
    │  反分词, 构建 RequestOutput
    ▼
RequestOutput → 用户 (流式/非流式)
```

### 3.3 关键数据结构

| 数据结构 | 定义位置 | 用途 |
|----------|----------|------|
| `EngineCoreRequest` | `v1/engine/__init__.py` | 引擎核心请求，包含 token IDs、采样参数、LoRA 请求等 |
| `EngineCoreOutput` | `v1/engine/__init__.py` | 单个请求的输出，包含新 token IDs、logprobs、完成原因 |
| `EngineCoreOutputs` | `v1/engine/__init__.py` | 所有请求的输出集合 |
| `SchedulerOutput` | `v1/core/sched/output.py` | 调度输出，包含新请求数据、缓存请求数据、token 计数等 |
| `NewRequestData` | `v1/core/sched/output.py` | 新请求的完整数据（prompt tokens、采样参数、块 IDs 等） |
| `CachedRequestData` | `v1/core/sched/output.py` | 已知请求的增量数据（新块 IDs、新 token IDs） |
| `ModelRunnerOutput` | `v1/worker/` | 模型运行器输出，包含采样 token IDs、logprobs 等 |
| `KVCacheBlock` | `v1/core/kv_cache_utils.py` | KV Cache 块，包含 block_id、ref_cnt、block_hash |
| `KVCacheBlocks` | `v1/core/kv_cache_manager.py` | KV Cache 块集合，返回给 Scheduler |
| `RequestOutput` | `outputs.py` | 用户面对的请求输出 |

---

## 第四章 vLLM v1 初始化过程

### 4.1 离线模式初始化 (LLM 类)

```python
# 用户代码
from vllm import LLM
llm = LLM(model="meta-llama/Llama-3-8B")
```

**初始化链：**

```
LLM.__init__()
  │
  ├── 1. 构造 EngineArgs (从参数)
  │
  ├── 2. LLMEngine.from_engine_args(engine_args)
  │      │
  │      ├── 2.1 创建 VllmConfig
  │      │      ├── ModelConfig (模型配置)
  │      │      ├── CacheConfig (缓存配置)
  │      │      ├── ParallelConfig (并行配置)
  │      │      ├── SchedulerConfig (调度器配置)
  │      │      ├── DeviceConfig (设备配置)
  │      │      ├── LoadConfig (加载配置)
  │      │      ├── LoRAConfig (LoRA 配置, 可选)
  │      │      ├── SpeculativeConfig (推测解码配置, 可选)
  │      │      ├── CompilationConfig (编译配置)
  │      │      └── ... (~20 个子配置)
  │      │
  │      ├── 2.2 创建 LLMEngine
  │      │      ├── InputProcessor (输入处理器)
  │      │      ├── OutputProcessor (输出处理器)
  │      │      ├── Renderer (渲染器, 处理聊天模板)
  │      │      └── EngineCoreClient.make_client()
  │      │           │
  │      │           └── InprocClient (进程内模式)
  │      │                │
  │      │                └── 创建 EngineCore
  │      │
  │      └── 2.3 EngineCore 初始化 (见 4.3)
  │
  └── 3. 完成
```

### 4.2 在线模式初始化 (API Server)

```bash
vllm serve meta-llama/Llama-3-8B
```

**初始化链：**

```
CLI 入口 (vllm.entrypoints.cli.main)
  │
  ├── 1. 解析命令行参数 → AsyncEngineArgs
  │
  ├── 2. AsyncLLM.from_vllm_config(vllm_config)
  │      │
  │      ├── 2.1 InputProcessor, OutputProcessor, Renderer
  │      │
  │      └── 2.2 EngineCoreClient.make_async_mp_client()
  │           │
  │           └── AsyncMPClient
  │                │
  │                └── 启动 EngineCoreProc (独立进程)
  │                     │
  │                     └── 通过 ZMQ 套接字通信
  │
  ├── 3. 创建 FastAPI 应用
  │      ├── 注册路由 (/v1/chat/completions, /v1/completions 等)
  │      └── 创建 OpenAIServingChat, OpenAIServingCompletion 等
  │
  └── 4. 启动 Uvicorn HTTP 服务器
```

### 4.3 EngineCore 初始化详解

`EngineCore.__init__()` 是整个引擎的核心初始化流程：

```python
# vllm/v1/engine/core.py, EngineCore.__init__()

def __init__(self, vllm_config, executor_class, ...):
    # 1. 创建 Executor (模型执行器)
    self.model_executor = executor_class(vllm_config)

    # 2. 初始化 KV Cache
    #    2.1 获取 KV Cache 规格 (每层的形状、dtype)
    #    2.2 内存分析: 运行 dummy forward pass 确定可用内存
    #    2.3 计算 KV Cache 配置: 块数、张量布局、缓存组
    #    2.4 在 Worker 上分配 KV Cache 张量
    self._initialize_kv_caches()

    # 3. 创建 Scheduler
    #    根据 SchedulerConfig 选择调度器类
    self.scheduler = scheduler_config.get_scheduler_cls()(
        scheduler_config, cache_config, ...
    )

    # 4. 设置步进函数
    #    如果启用异步调度, 使用 step_with_batch_queue
    #    否则使用 step
    self.step_fn = self.step  # 或 self.step_with_batch_queue
```

### 4.4 Worker 初始化详解

Worker 的初始化分为多个阶段：

```python
# 阶段 1: init_device()
Worker.init_device()
  ├── 设置 CUDA 设备
  ├── 初始化分布式环境 (NCCL)
  ├── 设置随机种子
  ├── 内存快照
  ├── 初始化 WorkspaceManager
  └── 创建 GPUModelRunner

# 阶段 2: load_model()
Worker.load_model()
  ├── CuMemAllocator 上下文 (可选)
  └── GPUModelRunner.load_model()
       ├── 模型加载器加载权重
       ├── LoRA 层包装 (可选)
       ├── 推测解码草稿模型加载 (可选)
       └── EPLB 设置 (MoE 模型)

# 阶段 3: determine_available_memory()
Worker.determine_available_memory()
  ├── 运行 profile_run() (dummy forward pass)
  ├── 测量峰值内存
  └── 计算可用 KV Cache 内存

# 阶段 4: initialize_from_config()
Worker.initialize_from_config(kv_cache_config)
  ├── 更新 CacheConfig
  ├── 初始化 KV 传输连接器
  └── GPUModelRunner.initialize_kv_cache()
       ├── 分配 KV Cache 张量
       ├── 重塑张量形状
       └── 绑定到注意力层

# 阶段 5: compile_or_warm_up_model()
Worker.compile_or_warm_up_model()
  ├── 编译模型 (torch.compile)
  ├── 内核预热
  ├── CUDA Graph 捕获
  ├── 采样器预热
  └── 预分配 logits 缓冲区
```

### 4.5 EngineCoreProc 初始化 (多进程模式)

当使用多进程模式时，`EngineCoreProc` 在独立进程中运行：

```python
# EngineCoreProc.worker_main() - 独立进程入口

def worker_main(vllm_config, ...):
    # 1. 设置信号处理器 (SIGTERM, SIGINT)
    # 2. 创建 EngineCoreProc 实例 (继承自 EngineCore)
    #    ├── 创建 Executor → 创建 Workers
    #    ├── 初始化 KV Cache
    #    ├── 创建 Scheduler
    #    └── 设置 ZMQ 通信
    # 3. 发送 READY 信号给父进程
    # 4. 进入 busy_loop

def run_busy_loop(self):
    while self._handle_shutdown():
        self._process_input_queue()    # 处理输入队列 (add_requests, aborts)
        self._process_engine_step()    # 执行 step_fn(), 将输出放入输出队列
```

---

## 第五章 调度器详解

### 5.1 调度器架构

```
SchedulerInterface (抽象基类)
    │
    ├── Scheduler (同步调度器)
    │
    └── AsyncScheduler (异步调度器, 支持流水线调度)

请求队列:
    ├── FCFSRequestQueue (先进先出, 基于 deque)
    └── PriorityRequestQueue (优先级, 基于堆)

KV Cache 管理:
    ├── KVCacheManager
    │    └── KVCacheCoordinator
    │         ├── KVCacheCoordinatorNoPrefixCache
    │         ├── UnitaryKVCacheCoordinator (单缓存组)
    │         └── HybridKVCacheCoordinator (混合缓存组)
    └── BlockPool
         ├── FreeKVCacheBlockQueue (空闲块队列)
         └── BlockHashToBlockMap (前缀缓存哈希表)
```

### 5.2 调度算法

vLLM v1 的调度器采用 **Token 级调度**，不区分 prefill/decode 阶段。每个请求维护 `num_computed_tokens`（已计算的 token 数）和 `num_tokens_with_spec`（总 token 数 + 推测 token 数）。

**调度流程 (`schedule()` 方法)：**

```
schedule() 每步执行:

Phase 1: 调度 RUNNING 请求 (已在运行的请求)
─────────────────────────────────────────
  for each request in self.running:
      num_new_tokens = num_tokens_with_spec - num_computed_tokens
      clamp(num_new_tokens, token_budget, max_model_len, long_prefill_token_threshold)
      blocks = kv_cache_manager.allocate_slots(request, num_new_tokens)
      if blocks is None:
          # 内存不足, 预抢占最低优先级的运行请求
          preempt(lowest_priority_running_request)
          retry allocation
      schedule request with num_new_tokens

Phase 2: 调度 WAITING 请求 (等待中的请求)
─────────────────────────────────────────
  # 仅当 Phase 1 没有发生预抢占时执行
  for each request in self.waiting:
      check if request is blocked (grammar, remote KV, streaming)
      lookup prefix cache hits
      num_new_tokens = request.num_tokens - num_computed_tokens

      if enable_chunked_prefill and num_new_tokens > token_budget:
          num_new_tokens = min(num_new_tokens, token_budget)  # 分块
      elif num_new_tokens > token_budget:
          break  # 不分块, 停止调度

      blocks = kv_cache_manager.allocate_slots(request, num_new_tokens)
      if blocks is None:
          break  # 内存不足, 停止调度 (不预抢占等待请求)
      move request from waiting to running
```

### 5.3 调度策略

通过 `SchedulerConfig.policy` 配置：

- **FCFS (默认)**：先进先出，使用 `deque` 实现
- **Priority**：优先级调度，使用最小堆 `(priority, arrival_time, request_id)` 排序

在 Priority 模式下，预抢占也遵循优先级：优先抢占 `(priority, arrival_time)` 值最大的请求。

### 5.4 分块预填充 (Chunked Prefill)

分块预fill 允许长 prompt 跨多个调度步骤完成，与其它请求的 decode token 交错执行。

**启用时** (`enable_chunked_prefill=True`, 默认)：
- 如果等待请求的 `num_new_tokens` 超过剩余 `token_budget`，调度器将其截断到 budget
- 长 prompt 可以在多个步骤中逐步完成

**禁用时**：
- 如果请求的 token 数超过 budget，调度器直接停止调度
- 长 prefill 会阻塞所有其他请求

**相关配置：**
- `long_prefill_token_threshold`：超过此阈值的请求每步被截断
- `max_num_partial_prefills`：最大并发部分 prefill 数
- `scheduler_reserve_full_isl`：是否在准入时检查完整输入序列长度

### 5.5 预抢占策略

vLLM v1 使用 **重计算 (recompute)** 预抢占（不支持 swap 到 CPU）：

```python
def _preempt_request(self, request, timestamp):
    self.kv_cache_manager.free(request)      # 释放所有 KV 块
    request.num_computed_tokens = 0           # 重置已计算 token 数
    request.spec_token_ids = []               # 清除推测 token
    request.num_preemptions += 1
    self.waiting.prepend_request(request)     # 放回等待队列头部
```

**关键特性：**
- 所有 KV Cache 块被释放，已计算状态完全丢失
- 被抢占的请求放回等待队列头部，优先重新调度
- 仅 RUNNING 请求可能被抢占，WAITING 请求不会被抢占
- 如果 WAITING 请求分配 KV 块失败，调度直接停止

### 5.6 请求状态机

```
WAITING  ──→  RUNNING  ──→  FINISHED_STOPPED / FINISHED_LENGTH_CAPPED / ...
   ^            |
   |            v
   +── PREEMPTED

额外的等待状态:
  - WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR (等待语法编译)
  - WAITING_FOR_REMOTE_KVS (等待远程 KV 传输)
  - WAITING_FOR_STREAMING_REQ (等待下一个流式输入块)
```

### 5.7 异步调度 (AsyncScheduler)

`AsyncScheduler` 扩展 `Scheduler`，支持流水线调度——在当前步骤的 forward pass 仍在运行时，计算下一步的调度。

**关键差异：**
1. 调度后，为 decode 请求添加 `num_output_placeholders`（1 个采样 token + 推测 token）
2. 下一步调度时可以看到这些占位符 token，提前分配 KV 块
3. 当实际输出到达时，递减 `num_output_placeholders`，提交最终化的块

### 5.8 Scheduler 与 KV Cache Manager 的交互

每个调度步骤中的交互：

```
1. kv_cache_manager.new_step_starts()
   │  通知新步骤开始
   │
2. kv_cache_manager.get_computed_blocks(request)
   │  查找前缀缓存命中
   │  返回 (KVCacheBlocks, num_computed_tokens)
   │
3. kv_cache_manager.allocate_slots(request, num_new_tokens)
   │  释放滑动窗口外的块
   │  计算需要的新块数
   │  从空闲池分配新块
   │  返回新分配的 KVCacheBlocks
   │
4. kv_cache_manager.free(request)
   │  释放请求的所有块 (预抢占或请求完成时)
   │
5. kv_cache_manager.get_blocks(request_id)
   │  获取请求当前所有块 (用于构建 SchedulerOutput)
```

### 5.9 SchedulerOutput 数据结构

```python
class SchedulerOutput:
    # 新请求的完整数据
    scheduled_new_reqs: list[NewRequestData]
    # 已知请求的增量数据 (差量通信)
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
    # 已完成的请求 ID
    finished_req_ids: set[str]
    # 被抢占的请求 ID
    preempted_req_ids: set[str]
    # 需要清零的新块 ID
    new_block_ids_to_zero: list[int] | None
```

**通信优化：**
- 新请求 (`NewRequestData`) 携带完整数据
- 已知请求 (`CachedRequestData`) 只携带差量数据（新块 IDs、新 token IDs），使用并行列表而非每请求对象

---

## 第六章 Worker 与 Executor 详解

### 6.1 Executor 架构

```
Executor (抽象基类)
    │
    ├── UniProcExecutor (单进程)
    │    └── WorkerWrapperBase → Worker → GPUModelRunner → Model
    │
    ├── MultiprocExecutor (多进程, 主要生产环境)
    │    └── [WorkerProc × N] → Worker → GPUModelRunner → Model
    │
    └── RayDistributedExecutor (Ray 分布式)
         └── [RayWorkerWrapper × N] → Worker → GPUModelRunner → Model
```

**选择逻辑 (`Executor.get_class()`)：**
- `"uni"` → `UniProcExecutor`
- `"mp"` → `MultiprocExecutor`
- `"ray"` → `RayDistributedExecutor`

### 6.2 UniProcExecutor (单进程执行器)

最简单的执行器，所有操作在同一进程内完成：

```python
class UniProcExecutor:
    def __init__(self, vllm_config):
        self.driver_worker = WorkerWrapperBase(rpc_rank=0)
        self.driver_worker.init_worker()
        self.driver_worker.init_device()
        self.driver_worker.load_model()

    def collective_rpc(self, method, args, kwargs):
        # 直接函数调用，无 IPC 开销
        result = getattr(self.driver_worker, method)(*args, **kwargs)
        return [result]
```

### 6.3 MultiprocExecutor (多进程执行器)

生产环境的主要执行器，使用共享内存消息队列：

```python
class MultiprocExecutor:
    def __init__(self, vllm_config):
        # 1. 创建共享内存广播队列
        self.rpc_broadcast_mq = MessageQueue()

        # 2. 为每个 local_rank 创建 WorkerProc
        for local_rank in range(local_world_size):
            WorkerProc.make_worker_process()  # 启动子进程

        # 3. 等待所有 Worker 就绪
        WorkerProc.wait_for_ready()

        # 4. 启动监控线程 (检测 Worker 意外死亡)
```

**WorkerProc 生命周期：**

```
WorkerProc.worker_main()  [子进程入口]
    │
    ├── 设置信号处理器
    ├── 创建 WorkerWrapperBase
    │    ├── 解析 worker 类
    │    ├── 注入扩展类 (可选)
    │    └── 实例化 Worker
    ├── init_worker(), init_device(), load_model()
    ├── 初始化消息队列
    ├── 发送 READY 信号
    └── 进入 worker_busy_loop()

worker_busy_loop():
    while True:
        (method, args, kwargs, output_rank) = rpc_broadcast_mq.dequeue()
        result = getattr(self.worker, method)(*args, **kwargs)
        if output_rank is None or self.rank == output_rank:
            worker_response_mq.enqueue(result)
```

**输出排名优化：**
只有最后一个 PP 阶段的第一个 TP rank 返回 `ModelRunnerOutput`，其它 Worker 返回 `None`，减少 IPC 开销。

### 6.4 collective_rpc 机制

所有 Executor 到 Worker 的通信都通过 `collective_rpc`：

```python
def collective_rpc(self, method, args, kwargs, non_block=False):
    # 1. 广播方法调用到所有 Worker
    rpc_broadcast_mq.enqueue((method, args, kwargs, output_rank))

    # 2. 创建 FutureWrapper
    future = FutureWrapper()
    future.result = lambda: dequeue from response_mqs

    # 3. 如果 non_block=True, 立即返回 future
    if non_block:
        return future
    return future.result()
```

### 6.5 Worker 类层次

```
WorkerBase (抽象基类)
    │  定义接口: init_device, load_model, execute_model, ...
    │
    └── Worker (GPU Worker)
         │  具体实现: CUDA 设备初始化, NCCL 分布式, 内存分析
         │
         └── GPUModelRunner (~7400 行)
              │  模型推理的核心: 输入准备, 注意力元数据, 前向传播, 采样
              │
              ├── LoRAModelRunnerMixin (LoRA 支持)
              ├── KVConnectorModelRunnerMixin (KV 传输)
              └── ECConnectorModelRunnerMixin (编码器连接)
```

### 6.6 GPUModelRunner 的核心方法

#### 6.6.1 execute_model() —— 前向传播

```python
def execute_model(self, scheduler_output, intermediate_tensors=None):
    # 1. 预处理: _update_states()
    #    - 从 InputBatch 移除完成的请求
    #    - 清零新分配的缓存块
    #    - 添加新请求到 InputBatch
    #    - 更新运行/恢复请求的块 IDs、token 计数

    # 2. 准备输入: _prepare_inputs()
    #    - 计算每请求 token 计数、位置、slot mappings
    #    - 构建 logits_indices
    #    - 处理推测解码元数据

    # 3. 确定批处理执行模式: _determine_batch_execution_and_padding()
    #    - 检查是否统一 decode
    #    - 分发 CUDA Graph 模式 (NONE/PIECEWISE/FULL)

    # 4. 构建注意力元数据: _build_attention_metadata()
    #    - 使用注册的元数据构建器
    #    - 处理级联注意力、微批处理

    # 5. 预处理输入: _preprocess()
    #    - 收集 input_ids, positions, inputs_embeds
    #    - 处理多模态输入

    # 6. 模型前向传播: _model_forward()
    #    - 调用 self.model(input_ids, positions, ...)
    #    - 可能被 CUDAGraphWrapper 或 UBatchWrapper 包装

    # 7. 后处理
    #    - 非最后 PP 阶段: 返回 IntermediateTensors
    #    - 最后 PP 阶段: 提取 hidden states, 计算 logits
    #    - 存储状态用于延迟采样

    # 8. 返回 None (信号: 需要调用 sample_tokens)
```

#### 6.6.2 sample_tokens() —— 令牌采样

```python
def sample_tokens(self, grammar_output):
    # 1. 解包 execute_model_state
    # 2. 应用语法掩码 (结构化输出)
    # 3. 运行 Sampler
    # 4. 更新 InputBatch 状态
    # 5. 处理推测解码: 提议草稿 token
    # 6. 提取 logprobs
    # 7. 复制 token IDs 到 CPU
    # 8. 返回 ModelRunnerOutput
```

**执行/采样分离的设计目的：**
允许在 forward pass 和采样之间计算语法位图（结构化输出），实现异步调度。

### 6.7 InputBatch —— 持久化批处理状态

`InputBatch` 维护 GPU 上的持久化状态，跨迭代保持：

- `input_ids`: 输入 token IDs
- `positions`: 位置编码
- `block_table`: 块表 (请求 → 块 ID 映射)
- `slot_mapping`: slot 映射 (token 位置 → KV Cache slot)
- `sampling_metadata`: 采样元数据

`_update_states()` 方法增量更新 InputBatch，而非每步重建。

### 6.8 CUDA Graph 支持

`CudagraphDispatcher` 根据批处理属性选择 CUDA Graph 模式：

- **NONE**: 不使用 CUDA Graph（小批量或异构批处理）
- **PIECEWISE**: 分段 CUDA Graph（支持条件分支）
- **FULL**: 完整 CUDA Graph（统一 decode 批处理）

CUDA Graph 在 `capture_model()` 阶段捕获，运行时通过 `CUDAGraphWrapper` 回放。

### 6.9 Pipeline Parallelism

Pipeline Parallelism 通过 NCCL 张量字典通信实现：

```python
# Worker.execute_model()

# 非第一个 PP 阶段: 接收中间张量
if not is_first_pp_rank:
    intermediate_tensors = get_pp_group().irecv_tensor_dict()  # 非阻塞

# 运行模型
output = self.model_runner.execute_model(scheduler_output, intermediate_tensors)

# 非最后一个 PP 阶段: 发送中间张量
if not is_last_pp_rank:
    get_pp_group().isend_tensor_dict(output)  # 非阻塞
    return None

# 最后一个 PP 阶段: 返回最终输出
return output
```

`AsyncIntermediateTensors` 包装接收到的张量，实现惰性同步（首次访问时等待）。

---

## 第七章 KV Cache 与内存管理

### 7.1 PagedAttention 核心思想

传统 LLM 推理需要为每个请求预分配连续的 KV Cache 内存，导致：
- 内存碎片化
- 内存浪费（预分配但未使用）
- 无法动态调整

vLLM 的 PagedAttention 借鉴操作系统虚拟内存：
- 将 KV Cache 分成固定大小的 **块 (Block)**
- 使用 **块表 (Block Table)** 将逻辑位置映射到物理块
- 请求可以使用非连续的物理块
- 块可以按需分配和释放

### 7.2 核心数据结构

#### 7.2.1 KVCacheBlock

```python
@dataclass(slots=True)
class KVCacheBlock:
    block_id: int          # 物理块 ID (0 到 num_gpu_blocks-1)
    ref_cnt: int           # 引用计数 (>0 表示在使用中)
    _block_hash: BlockHashWithGroupId | None  # 内容哈希 (前缀缓存)
    prev_free_block: KVCacheBlock | None      # 空闲链表前驱
    next_free_block: KVCacheBlock | None      # 空闲链表后继
    is_null: bool          # 是否为空块 (滑动窗口填充)
```

#### 7.2.2 FreeKVCacheBlockQueue

空闲块的双向链表，直接操作 `KVCacheBlock` 的链表指针，无需额外 Python 对象分配：

- **O(1)** 的 `popleft`、`append`、`remove` 操作
- **LRU 顺序**：最近释放的块在尾部，最久未使用的块在头部
- 使用哨兵节点简化边界逻辑

#### 7.2.3 BlockPool

中央块管理器，拥有：
- `blocks: list[KVCacheBlock]` —— 所有 GPU 块
- `free_block_queue: FreeKVCacheBlockQueue` —— 空闲块队列
- `cached_block_hash_to_block: BlockHashToBlockMap` —— 前缀缓存哈希表
- `null_block` —— 空块占位符

#### 7.2.4 KVCacheSpec 层次

```python
KVCacheSpec (基类)
    ├── AttentionSpec
    │    ├── FullAttentionSpec (完整注意力)
    │    │    ├── MLAAttentionSpec (Multi-head Latent Attention, DeepSeek)
    │    │    └── TQFullAttentionSpec (TurboQuant 感知)
    │    ├── SlidingWindowSpec (滑动窗口注意力)
    │    │    └── SlidingWindowMLASpec
    │    ├── ChunkedLocalAttentionSpec (分块局部注意力, Gemma)
    │    ├── EncoderOnlyAttentionSpec (零内存, 无需 KV Cache)
    │    └── CrossAttentionSpec (编码器-解码器交叉注意力)
    ├── MambaSpec (SSM 层)
    └── UniformTypeKVCacheSpecs (聚合多个同类型层)
```

### 7.3 块分配流程

```
KVCacheManager.allocate_slots(request, num_new_tokens)

块布局:
| < comp > | < new_comp > | < ext_comp > | < new > | < lookahead > |
   已计算      前缀缓存命中    外部缓存       新计算      推测 token

步骤:
1. remove_skipped_blocks()
   │  释放滑动窗口外的块
   │
2. get_num_blocks_to_allocate()
   │  计算需要的新块数
   │  考虑前缀缓存命中、滑动窗口上限
   │
3. allocate_new_computed_blocks()
   │  处理前缀缓存命中的块 (touch 增加引用计数)
   │
4. allocate_new_blocks()
   │  从空闲池弹出新块
   │  如果块有缓存哈希, 从前缀缓存中移除
   │  增加 ref_cnt
   │
5. 返回 KVCacheBlocks 或 None (内存不足)
```

### 7.4 块释放流程

```python
def free(self, request):
    # 按逆序释放, 使尾部块先被驱逐 (保留前缀)
    for block in reversed(request.blocks):
        block.ref_cnt -= 1
        if block.ref_cnt == 0 and not block.is_null:
            free_block_queue.append(block)  # 放回空闲队列尾部
```

### 7.5 前缀缓存 (Prefix Caching)

#### 7.5.1 哈希机制

块哈希是 **内容可寻址且链式依赖** 的：

```python
block_hash = hash_function((parent_block_hash, tuple(token_ids), extra_keys))
```

- 每个块的哈希依赖于其 token IDs 和父块的哈希
- 相同 token IDs 但在序列不同位置的块有不同的哈希
- extra_keys 可以包含多模态特征哈希、LoRA 名称、缓存盐值

#### 7.5.2 缓存查找

当请求到达时：
1. 计算 prompt 所有完整块的哈希
2. 沿块哈希列表遍历，检查 `BlockPool.get_cached_block()`
3. 返回最长连续前缀缓存命中

最大缓存命中长度 = `request.num_tokens - 1`（必须至少重新计算最后一个 token）

#### 7.5.3 缓存插入

当块计算完成后：
1. 计算块哈希
2. 存储到 `cached_block_hash_to_block` 映射
3. 设置块的 `_block_hash` 字段
4. 只有完整块被缓存，部分块不缓存

#### 7.5.4 缓存驱逐

驱逐通过空闲队列的 LRU 顺序隐式实现：
- `get_new_blocks()` 从空闲队列头部弹出块
- 调用 `_maybe_evict_cached_block()` 移除块的前缀缓存条目
- `ref_cnt > 0` 的块永远不会被驱逐

### 7.6 滑动窗口注意力的块回收

对于滑动窗口注意力 (SWA)：
- 窗口外的块被释放并替换为空块
- `get_num_skipped_tokens()` 计算需要跳过的 token 数
- 缓存命中查找从右到左扫描（只有序列尾部对 SWA 重要）
- `max_admission_blocks_per_request()` 计算回收感知的准入上限

### 7.7 KV Cache 内存布局

不同注意力后端的 KV Cache 张量布局：

**FlashAttention:**
```
形状: (num_blocks, 2, block_size, num_kv_heads, head_size)
       维度 1 是 K/V
支持 NHD 和 HND 两种内存布局
```

**PagedAttention:**
```
key_cache:   (num_blocks, num_kv_heads, head_size // x, block_size, x)
value_cache: (num_blocks, num_kv_heads, head_size, block_size)
其中 x = 16 // element_size
```

**MLA (DeepSeek):**
```
存储 kv_c_normed (潜在表示) 和 k_pe (位置编码) 的拼接
DeepSeek V4 fp8: 每 token 584 字节 (448B NoPE + 128B RoPE + 8B fp8 scale)
```

### 7.8 Block Table 与 Slot Mapping

```python
class BlockTable:
    # 2D 张量 [max_num_reqs, max_num_blocks_per_req]
    # 映射请求索引到块 ID
    block_table: torch.Tensor

    # 1D 张量 [max_num_batched_tokens]
    # 映射每个 token 位置到 KV Cache 中的 flat slot 索引
    # slot = block_number * block_size + block_offset
    slot_mapping: torch.Tensor
```

`MultiGroupBlockTable` 包装每个 KV Cache 组的 `BlockTable`，支持混合模型（全注意力 + SWA）。

### 7.9 KV Cache 量化

支持多种 KV Cache 量化模式 (`KVQuantMode`)：

| 模式 | 说明 |
|------|------|
| `NONE` | 无量化 |
| `FP8_PER_TENSOR` | FP8 每张量量化 |
| `INT8_PER_TOKEN_HEAD` | INT8 每 token 每 head 量化 |
| `FP8_PER_TOKEN_HEAD` | FP8 每 token 每 head 量化 |
| `NVFP4` | NVFP4 量化 |

每 token-head 模式在量化数据旁存储 float32 缩放因子。

### 7.10 内存管理策略总结

| 策略 | 说明 |
|------|------|
| **分页内存** | GPU 内存分成固定大小块，通过块表间接映射 |
| **引用计数** | 跟踪每个块被多少请求引用，`ref_cnt > 0` 不可驱逐 |
| **LRU 驱逐** | 空闲队列维护 LRU 顺序，最近最少使用的块优先被回收 |
| **前缀缓存** | 内容可寻址、链式依赖的块哈希，跨请求复用公共前缀 |
| **滑动窗口回收** | SWA 和分块局部注意力的窗口外块被释放 |
| **统一页面大小** | 混合模型统一页面大小，小页面填充到最大页面 |
| **每 Worker 一致性** | Pipeline Parallelism 所有 Worker 使用相同块数（最小值） |

---

## 第八章 整体调用流程

### 8.1 离线推理流程

```python
from vllm import LLM, SamplingParams

# 1. 初始化
llm = LLM(model="meta-llama/Llama-3-8B")

# 2. 推理
outputs = llm.generate(["Hello, world!"], SamplingParams(temperature=0.7))

# 3. 获取结果
for output in outputs:
    print(output.outputs[0].text)
```

**内部流程：**

```
LLM.generate(prompts, sampling_params)
    │
    ├── _preprocess_cmpl_one(prompt)
    │    │  应用聊天模板, 分词
    │    ▼
    │    EngineInput
    │
    ├── LLMEngine.add_request(request_id, engine_input, sampling_params)
    │    │
    │    ├── InputProcessor.process_inputs()
    │    │    │  验证参数, 处理多模态输入
    │    │    ▼
    │    │    EngineCoreRequest
    │    │
    │    ├── OutputProcessor.add_request()
    │    │    │  创建 RequestState
    │    │    ▼
    │    │
    │    └── EngineCoreClient.add_request(engine_core_request)
    │         │  发送到 EngineCore
    │         ▼
    │
    └── _run_engine()
         │  循环调用 step() 直到所有请求完成
         │
         └── LLMEngine.step() [循环]
              │
              ├── EngineCoreClient.get_output()
              │    │  从 EngineCore 获取输出
              │    ▼
              │    EngineCoreOutputs
              │
              ├── OutputProcessor.process_outputs()
              │    │  反分词, 构建 RequestOutput
              │    ▼
              │    [RequestOutput]
              │
              └── 返回 [RequestOutput]
```

### 8.2 在线服务流程 (Chat Completion)

```
POST /v1/chat/completions
    │
    ├── FastAPI 路由 → OpenAIServingChat.create_chat_completion()
    │
    ├── render_chat_request()
    │    │  验证模型, 应用聊天模板, 分词
    │    ▼
    │    [EngineInput]
    │
    ├── request.to_sampling_params()
    │    │  转换 OpenAI 参数为 SamplingParams
    │    ▼
    │
    └── AsyncLLM.generate(engine_input, sampling_params)
         │
         ├── add_request()
         │    ├── InputProcessor.process_inputs() → EngineCoreRequest
         │    ├── 创建 RequestOutputCollector (asyncio 队列)
         │    └── EngineCoreClient.add_request_async() → ZMQ 发送
         │
         └── yield RequestOutput [流式]
              │
              │  ← output_handler 后台任务:
              │     EngineCoreClient.get_output_async()
              │     OutputProcessor.process_outputs()
              │     RequestOutputCollector.put()
              │
              ▼
         chat_completion_stream_generator()
              │  构建 ChatCompletionStreamResponse
              │  SSE 格式: data: {json}\n\n
              ▼
         StreamingResponse → HTTP 客户端
```

### 8.3 EngineCore.step() 详解

这是核心引擎的每一步执行：

```python
def step(self):
    # 1. 调度
    scheduler_output = self.scheduler.schedule()
    if not scheduler_output:
        return {}, False

    # 2. 执行模型 (异步)
    future = self.model_executor.execute_model(scheduler_output, non_block=True)

    # 3. 获取语法掩码 (结构化输出)
    grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)

    # 4. 等待模型执行结果
    model_output = future.result()

    # 5. 如果需要, 执行采样
    if model_output is None:
        model_output = self.model_executor.sample_tokens(grammar_output)

    # 6. 处理中止请求
    self._process_aborts_queue()

    # 7. 更新调度器状态
    engine_core_outputs = self.scheduler.update_from_output(
        scheduler_output, model_output
    )

    return engine_core_outputs, True
```

### 8.4 完整请求生命周期

```
1. 用户请求 → API Server / LLM 类
2. InputProcessor.process_inputs() → 分词、验证 → EngineCoreRequest
3. EngineCoreClient → ZMQ/进程内 → EngineCore
4. Scheduler.add_request() → waiting 队列
5. Scheduler.schedule() → 分配 KV 块 → SchedulerOutput
6. Executor.execute_model() → Worker.execute_model()
7. GPUModelRunner:
   a. _update_states() → 更新 InputBatch
   b. _prepare_inputs() → 构建输入
   c. _build_attention_metadata() → 注意力元数据
   d. _model_forward() → 模型前向传播
   e. sample_tokens() → 采样
8. ModelRunnerOutput → EngineCore
9. Scheduler.update_from_output() → 更新请求状态
10. EngineCoreOutputs → ZMQ/进程内 → OutputProcessor
11. OutputProcessor.process_outputs() → 反分词 → RequestOutput
12. 请求完成? → 是 → FINISHED
    请求未完成? → 回到步骤 5
```

### 8.5 流式响应机制

流式响应通过生产者-消费者模式实现：

```
生产者: EngineCore (独立进程)
    │  每步产生 EngineCoreOutputs
    │  包含增量 token IDs、logprobs、完成原因
    ▼
传输: AsyncMPClient (ZMQ IPC)
    │  从后台进程传输到异步事件循环
    ▼
输出处理器: AsyncLLM.output_handler (asyncio 任务)
    │  拉取输出, 通过 OutputProcessor 处理
    │  推送到每请求的 RequestOutputCollector 队列
    ▼
消费者: AsyncLLM.generate() 异步生成器
    │  从 RequestOutputCollector 拉取 RequestOutput
    │  yield 给调用者
    ▼
API 层: chat_completion_stream_generator()
    │  转换为 SSE 格式的 ChatCompletionStreamResponse
    ▼
FastAPI StreamingResponse → HTTP 客户端
```

---

## 第九章 API 服务器与入口点

### 9.1 两种使用模式

| 模式 | 类 | 引擎 | 用途 |
|------|-----|------|------|
| 离线推理 | `LLM` | `LLMEngine` (同步) | 批处理/脚本 |
| 在线服务 | API Server | `AsyncLLM` (异步) | 生产部署 |

### 9.2 入口点一览

| 入口 | 文件 | 用途 |
|------|------|------|
| `LLM` | `vllm/entrypoints/llm.py` | 离线推理 Python API |
| OpenAI API Server | `vllm/entrypoints/openai/api_server.py` | OpenAI 兼容 HTTP 服务 |
| Anthropic API | `vllm/entrypoints/anthropic/` | Anthropic 兼容 API |
| gRPC Server | `vllm/entrypoints/grpc/` | gRPC 服务 |
| CLI | `vllm/entrypoints/cli/` | 命令行工具 |
| MCP Server | `vllm/entrypoints/mcp/` | MCP 服务 |
| Sagemaker | `vllm/entrypoints/sagemaker/` | AWS Sagemaker 部署 |

### 9.3 OpenAI 兼容 API

**支持的端点：**
- `POST /v1/chat/completions` —— 聊天补全
- `POST /v1/completions` —— 文本补全
- `POST /v1/embeddings` —— 文本嵌入
- `POST /v1/responses` —— 响应 API

**关键组件：**
- `OpenAIServingChat` —— 聊天补全处理器
- `OpenAIServingCompletion` —— 文本补全处理器
- `OpenAIServing` —— 基类，提供通用功能

### 9.4 采样参数处理

`SamplingParams` 定义在 `vllm/sampling_params.py`：

```python
class SamplingParams(msgspec.Struct):
    n: int = 1                    # 每 prompt 的输出数
    temperature: float = 1.0      # 温度
    top_p: float = 1.0            # Top-p 采样
    top_k: int = -1               # Top-k 采样
    min_p: float = 0.0            # Min-p 采样
    presence_penalty: float = 0.0 # 存在惩罚
    frequency_penalty: float = 0.0 # 频率惩罚
    max_tokens: int | None = None # 最大 token 数
    stop: list[str] | None = None # 停止条件
    logprobs: int | None = None   # logprobs 数量
    structured_outputs: dict | None = None  # 结构化输出
    output_kind: RequestOutputKind = "CUMULATIVE"  # 输出模式
```

`ChatCompletionRequest.to_sampling_params()` 将 OpenAI 格式转换为 `SamplingParams`。

---

## 第十章 模型加载与配置系统

### 10.1 配置系统

配置系统已从单体 `config.py` 重构为模块化目录：

```
vllm/config/
    ├── __init__.py          # 重新导出所有配置类
    ├── vllm.py              # VllmConfig (主配置容器, ~2250 行)
    ├── model.py             # ModelConfig (~2200 行)
    ├── cache.py             # CacheConfig
    ├── parallel.py          # ParallelConfig
    ├── scheduler.py         # SchedulerConfig
    ├── device.py            # DeviceConfig
    ├── load.py              # LoadConfig
    ├── lora.py              # LoRAConfig
    ├── speculative.py       # SpeculativeConfig
    ├── quantization.py      # QuantizationConfigArgs
    ├── compilation.py       # CompilationConfig
    ├── kernel.py            # KernelConfig
    └── ... (~25 个配置文件)
```

#### VllmConfig

主配置容器，聚合所有子配置：

```python
class VllmConfig:
    model_config: ModelConfig
    cache_config: CacheConfig
    parallel_config: ParallelConfig
    scheduler_config: SchedulerConfig
    device_config: DeviceConfig
    load_config: LoadConfig
    lora_config: LoRAConfig | None
    speculative_config: SpeculativeConfig | None
    quant_config: QuantizationConfig | None
    compilation_config: CompilationConfig
    kernel_config: KernelConfig
    # ... ~20 个子配置
```

`__post_init__` 执行大量跨配置验证和自动配置（~2000 行）。

#### ModelConfig

模型相关设置的 Pydantic 数据类：

```python
class ModelConfig:
    model: str                    # 模型路径
    tokenizer: str                # 分词器路径
    dtype: str                    # 数据类型
    quantization: str | None      # 量化方法
    max_model_len: int            # 最大模型长度
    model_impl: str               # 模型实现 (auto/transformers/vllm)
    trust_remote_code: bool       # 信任远程代码
    runner: str                   # 运行器类型 (generate/pooling/draft)
```

### 10.2 模型注册表

模型注册表是将 HuggingFace 架构名映射到 vLLM 模型实现的核心机制：

```python
# 注册表字典
_TEXT_GENERATION_MODELS = {      # ~150+ 文本生成架构
    "LlamaForCausalLM": ("llama", "LlamaForCausalLM"),
    "Qwen2ForCausalLM": ("qwen2", "Qwen2ForCausalLM"),
    ...
}
_MULTIMODAL_MODELS = {           # ~100+ 多模态架构
    "LlavaForConditionalGeneration": ("llava", "LlavaForConditionalGeneration"),
    ...
}
_EMBEDDING_MODELS = { ... }      # 嵌入模型
_SPECULATIVE_DECODING_MODELS = { ... }  # 推测解码模型
```

**惰性加载机制：**
- 模型类在主进程中不导入
- `load_model_cls()` 使用 `importlib.import_module()` 按需加载
- `inspect_model_cls()` 在 **子进程** 中运行，避免 CUDA 初始化
- 结果缓存到磁盘 JSON 文件

**模型解析流程：**
1. 如果 `model_impl == "transformers"`，尝试 Transformers 后端
2. 尝试每个架构名（带后缀规范化）
3. 回退到 Transformers（如果 `model_impl == "auto"`）

### 10.3 模型加载器

```python
_LOAD_FORMAT_TO_MODEL_LOADER = {
    "auto": DefaultModelLoader,
    "hf": DefaultModelLoader,
    "bitsandbytes": BitsAndBytesModelLoader,
    "dummy": DummyModelLoader,
    "gguf": GGUFModelLoader,
    "runai_streamer": RunaiModelStreamerLoader,
    "sharded_state": ShardedStateLoader,
    "tensorizer": TensorizerLoader,
    ...
}
```

**DefaultModelLoader 加载流程：**
1. `initialize_model()` —— 解析模型类，实例化模型
2. `load_weights()` —— 加载权重（支持 safetensors、pytorch bin、GGUF 等格式）
3. `process_weights_after_loading()` —— 后处理（量化、注意力权重初始化）

### 10.4 量化系统

vLLM 支持 29 种量化方法：

```
awq, fp8, fbgemm_fp8, fp_quant, modelopt, modelopt_fp4, modelopt_mxfp8,
modelopt_mixed, gguf, auto_gptq, gptq, gptq_marlin, awq_marlin, humming,
compressed-tensors, bitsandbytes, experts_int8, quark, moe_wna16, torchao,
inc, mxfp4, gpt_oss_mxfp4, deepseek_v4_fp8, online,
fp8_per_tensor, fp8_per_block, int8_per_channel_weight_only, mxfp8
```

**架构：**
- `QuantizationConfig` —— 全局配置（ABC）
- `QuantizeMethodBase` —— 每层量化接口（ABC）
  - `create_weights()` —— 创建量化权重
  - `apply()` —— 应用量化计算
  - `process_weights_after_loading()` —— 加载后处理

### 10.5 LoRA 支持

LoRA (Low-Rank Adaptation) 通过以下组件支持：

```
LoRAConfig
    ├── max_lora_rank: 最大 LoRA 秩 (1/8/16/32/64/128/256/320/512)
    ├── max_loras: 批处理中最大并发 LoRA 数
    ├── max_cpu_loras: CPU 缓存的最大 LoRA 数
    └── target_modules: 限制哪些模块使用 LoRA

LoRAModelManager
    ├── 模块包装: 将兼容层替换为 LoRA 包装版本
    ├── 适配器生命周期: add → activate → deactivate
    ├── 打包模块合并: 合并单独的 LoRA 权重
    └── MoE 支持: 2D/3D MoE LoRA 格式

LoRALayerWeights
    ├── lora_a: 低秩矩阵 A
    ├── lora_b: 低秩矩阵 B
    ├── rank: 秩
    └── scaling: 缩放因子 (lora_alpha / rank)
```

### 10.6 推测解码

支持多种推测解码方法：

| 方法 | 说明 |
|------|------|
| `ngram` / `ngram_gpu` | N-gram 提示查找 |
| `medusa` | Medusa 多头预测 |
| `mlp_speculator` | MLP 推测器 |
| `draft_model` | 独立草稿模型 |
| `eagle` / `eagle3` | EAGLE 系列 |
| `mtp` | 多 Token 预测 (DeepSeek, MiMo, GLM4 等) |
| `dflash` | DFlash 方法 |
| `suffix` | 后缀解码 |
| `custom_class` | 用户自定义推测器 |

**MTP (Multi-Token Prediction)** 是统一方法，覆盖多种模型：
- `hf_config_override()` 自动检测基础模型类型并重写配置
- 支持 DeepSeek、MiMo、GLM、NemotronH、Qwen、Exaone 等

---

## 第十一章 关键优化技术总结

### 11.1 PagedAttention

**问题：** 传统连续内存分配导致内存碎片和浪费。

**解决方案：** 借鉴操作系统虚拟内存，将 KV Cache 分成固定大小的块，通过块表间接映射。

**收益：**
- 消除内存碎片
- 按需分配，减少浪费
- 支持非连续内存分配
- 启用前缀缓存

### 11.2 前缀缓存 (Prefix Caching)

**问题：** 多个请求共享相同前缀时（如系统 prompt），重复计算和存储。

**解决方案：** 内容可寻址、链式依赖的块哈希，自动识别和复用公共前缀。

**收益：**
- 减少重复计算
- 减少内存使用
- 自动透明，无需用户干预

### 11.3 分块预填充 (Chunked Prefill)

**问题：** 长 prompt 的 prefill 阶段会阻塞所有其他请求的 decode。

**解决方案：** 将长 prompt 分成多个块，与其它请求的 decode token 交错执行。

**收益：**
- 降低 decode 请求的延迟
- 提高 GPU 利用率
- 平滑批处理负载

### 11.4 CUDA Graph

**问题：** 小批量推理时，CPU 开销（内核启动、内存分配）成为瓶颈。

**解决方案：** 预捕获 CUDA Graph，运行时回放，消除 CPU 开销。

**实现：**
- `CudagraphDispatcher` 根据批处理属性选择模式
- NONE / PIECEWISE / FULL 三种模式
- 在 `capture_model()` 阶段捕获

### 11.5 连续批处理 (Continuous Batching)

**问题：** 传统静态批处理中，短请求必须等待最长请求完成。

**解决方案：** 每个调度步骤动态决定批处理内容，请求完成立即释放资源。

**实现：**
- Scheduler 每步运行 `schedule()`
- 完成的请求立即从批处理中移除
- 新请求可以立即加入

### 11.6 Token 级调度

**问题：** 传统的 prefill/decode 阶段分离限制了调度灵活性。

**解决方案：** 不区分阶段，统一以 token 为单位调度。

**收益：**
- 自然支持分块预填充
- 统一处理前缀缓存、推测解码
- 更灵活的资源分配

### 11.7 异步调度

**问题：** 调度和执行串行进行，浪费 GPU 空闲时间。

**解决方案：** 在当前步骤的 forward pass 运行时，预计算下一步的调度。

**实现：**
- `AsyncScheduler` 使用输出占位符
- 下一步可以提前分配 KV 块
- 调度和执行流水线化

### 11.8 持久化批处理状态

**问题：** 每步重建批处理状态（input_ids, positions, block_table）开销大。

**解决方案：** `InputBatch` 维护 GPU 上的持久化状态，每步增量更新。

**收益：**
- 减少 CPU-GPU 数据传输
- 减少内存分配
- 提高批处理准备效率

### 11.9 执行/采样分离

**问题：** 结构化输出的语法位图计算需要在 forward pass 和采样之间进行。

**解决方案：** 将 `execute_model` 和 `sample_tokens` 分离为两个独立调用。

**收益：**
- 允许异步计算语法位图
- 支持异步调度
- 更灵活的执行流程

### 11.10 输出排名优化

**问题：** 多 Worker 通信开销大。

**解决方案：** 只有最后一个 PP 阶段的第一个 TP rank 返回 `ModelRunnerOutput`。

**收益：**
- 减少 IPC 数据量
- 降低通信开销

### 11.11 零拷贝张量 IPC

**问题：** 多模态嵌入等大张量在进程间传输开销大。

**解决方案：** 使用 `torch.multiprocessing.Queue` 和共享内存实现零拷贝传输。

**实现：**
- `TensorIpcSender` / `TensorIpcReceiver`
- 通过共享内存避免数据复制

### 11.12 滑动窗口块回收

**问题：** 滑动窗口注意力只需要最近的 token，但传统方式保留所有历史。

**解决方案：** 自动释放滑动窗口外的块，替换为空块。

**收益：**
- 显著减少内存使用
- 自动透明处理

### 11.13 增量反分词

**问题：** 每步对整个序列反分词开销大。

**解决方案：** `IncrementalDetokenizer` 只处理新增 token，使用 HuggingFace 的 `DecodeStream` 实现快速反分词。

### 11.14 混合精度 KV Cache

**问题：** FP16 KV Cache 内存占用大。

**解决方案：** 支持 FP8、INT8、NVFP4 等量化模式。

**收益：**
- 减少 KV Cache 内存占用 50%-75%
- 精度损失可控

### 11.15 级联注意力 (Cascade Attention)

**问题：** 多个请求共享长公共前缀时，重复计算前缀注意力。

**解决方案：** 检测公共前缀，计算一次前缀注意力并共享结果。

**实现：**
- `get_num_common_prefix_blocks()` 检测公共前缀
- FlashAttention 后端的 `cascade_attention()` 函数

### 11.16 Workspace 管理

**问题：** GPU 内存碎片化导致大块分配失败。

**解决方案：** `WorkspaceManager` 提供单一连续 GPU 缓冲区，按需增长，预热后锁定。

**收益：**
- 减少内存碎片
- 可预测的内存使用
- 防止运行时分配失败

---

## 附录

### A. 关键文件速查表

| 文件 | 用途 |
|------|------|
| `vllm/v1/engine/llm_engine.py` | 同步 LLMEngine (前端) |
| `vllm/v1/engine/async_llm.py` | 异步 AsyncLLM (前端) |
| `vllm/v1/engine/core.py` | EngineCore (核心循环) |
| `vllm/v1/engine/core_client.py` | EngineCoreClient (IPC 传输) |
| `vllm/v1/engine/input_processor.py` | 输入处理器 |
| `vllm/v1/engine/output_processor.py` | 输出处理器 |
| `vllm/v1/core/sched/scheduler.py` | 调度器 |
| `vllm/v1/core/sched/output.py` | 调度输出数据结构 |
| `vllm/v1/core/kv_cache_manager.py` | KV Cache 管理器 |
| `vllm/v1/core/block_pool.py` | 块池 |
| `vllm/v1/core/kv_cache_utils.py` | KV Cache 工具和数据结构 |
| `vllm/v1/executor/abstract.py` | Executor 抽象基类 |
| `vllm/v1/executor/multiproc_executor.py` | 多进程执行器 |
| `vllm/v1/worker/worker_base.py` | Worker 基类 |
| `vllm/v1/worker/gpu_worker.py` | GPU Worker |
| `vllm/v1/worker/gpu_model_runner.py` | GPU Model Runner (~7400 行) |
| `vllm/config/vllm.py` | VllmConfig (主配置) |
| `vllm/config/model.py` | ModelConfig |
| `vllm/model_executor/models/registry.py` | 模型注册表 |
| `vllm/entrypoints/llm.py` | LLM 离线推理类 |
| `vllm/entrypoints/openai/api_server.py` | OpenAI API 服务器 |

### B. 术语表

| 术语 | 说明 |
|------|------|
| **PagedAttention** | 分页注意力，vLLM 的核心创新 |
| **KV Cache** | Key-Value 缓存，存储注意力计算的中间结果 |
| **Block** | KV Cache 的固定大小分配单元 |
| **Block Table** | 逻辑位置到物理块的映射表 |
| **Slot Mapping** | Token 位置到 KV Cache slot 的映射 |
| **Prefix Caching** | 前缀缓存，复用公共前缀的 KV Cache |
| **Chunked Prefill** | 分块预填充，长 prompt 分块处理 |
| **Continuous Batching** | 连续批处理，动态调整批处理内容 |
| **Speculative Decoding** | 推测解码，使用草稿模型加速 |
| **LoRA** | Low-Rank Adaptation，低秩适配 |
| **Tensor Parallelism (TP)** | 张量并行 |
| **Pipeline Parallelism (PP)** | 流水线并行 |
| **Data Parallelism (DP)** | 数据并行 |
| **CUDA Graph** | CUDA 图，预捕获的 GPU 操作序列 |
| **EngineCore** | 核心引擎，负责调度和执行 |
| **Scheduler** | 调度器，决定每步执行哪些请求 |
| **Worker** | 工作者，负责模型加载和推理执行 |
| **Executor** | 执行器，管理 Worker 生命周期 |
| **GPUModelRunner** | GPU 模型运行器，负责实际推理 |
| **InputBatch** | 持久化批处理状态 |
| **SchedulerOutput** | 调度器输出，传递给 Worker |

### C. 参考资料

- [vLLM 官方文档](https://docs.vllm.ai/)
- [vLLM GitHub 仓库](https://github.com/vllm-project/vllm)
- [PagedAttention 论文](https://arxiv.org/abs/2309.06180)
- [vLLM 论文](https://arxiv.org/abs/2309.06180)
