# vLLM V1 进程与线程模型深度解析

> 本文档详细剖析 vLLM V1 的进程和线程架构，包括启动流程、请求处理、单机多卡和多机多卡通信机制。

---

## 目录

- [1. 整体进程架构图](#1-整体进程架构图)
- [2. 启动流程详解](#2-启动流程详解)
- [3. 请求处理时的进程交互](#3-请求处理时的进程交互)
- [4. 单机多卡通信机制](#4-单机多卡通信机制)
- [5. 多机多卡通信机制](#5-多机多卡通信机制)
- [6. 各进程/线程详解](#6-各进程线程详解)
- [7. ZMQ 通信详解](#7-zmq-通信详解)
- [8. 共享内存通信详解](#8-共享内存通信详解)
- [9. 关键代码索引](#9-关键代码索引)

---

## 1. 整体进程架构图

### 1.1 单机单卡 (最简模式)

```
┌─────────────────────────────────────────────────────────────────┐
│                        API Server 进程                          │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  AsyncLLM (asyncio event loop)                          │   │
│  │  ┌─────────────────┐  ┌─────────────────────────────┐  │   │
│  │  │ output_handler  │  │ EngineCoreOutputQueueTask   │  │   │
│  │  │ (asyncio.Task)  │  │ (asyncio.Task)              │  │   │
│  │  │                 │  │ 从 output_socket 读取输出    │  │   │
│  │  │ 处理输出，推送到 │  │ 放入 asyncio.Queue          │  │   │
│  │  │ 每请求队列      │  │                             │  │   │
│  │  └─────────────────┘  └─────────────────────────────┘  │   │
│  │                                                         │   │
│  │  AsyncMPClient                                          │   │
│  │  ┌─────────────────┐  ┌─────────────────────────────┐  │   │
│  │  │ input_socket    │  │ output_socket               │  │   │
│  │  │ (ZMQ ROUTER)    │  │ (ZMQ PULL)                  │  │   │
│  │  │ bind            │  │ bind                        │  │   │
│  │  └────────┬────────┘  └──────────────┬──────────────┘  │   │
│  └───────────┼──────────────────────────┼─────────────────┘   │
└──────────────┼──────────────────────────┼─────────────────────┘
               │ ZMQ DEALER/PUSH          │ ZMQ PUSH/PULL
               │                          │
┌──────────────▼──────────────────────────▼─────────────────────┐
│                     EngineCore 进程                            │
│                     (EngineCoreProc)                           │
│                                                                │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  主线程: run_busy_loop                                   │  │
│  │  ┌──────────────────────────────────────────────────┐   │  │
│  │  │ while True:                                      │   │  │
│  │  │   _process_input_queue()  # 从 input_queue 读取  │   │  │
│  │  │   _process_engine_step()  # 调度+执行+输出       │   │  │
│  │  └──────────────────────────────────────────────────┘   │  │
│  │                                                         │  │
│  │  Input 守护线程 (process_input_sockets)                  │  │
│  │  ┌──────────────────────────────────────────────────┐   │  │
│  │  │ ZMQ DEALER socket → 反序列化 → input_queue       │   │  │
│  │  └──────────────────────────────────────────────────┘   │  │
│  │                                                         │  │
│  │  Output 守护线程 (process_output_sockets)                │  │
│  │  ┌──────────────────────────────────────────────────┐   │  │
│  │  │ output_queue → 序列化 → ZMQ PUSH socket          │   │  │
│  │  └──────────────────────────────────────────────────┘   │  │
│  │                                                         │  │
│  │  input_queue ──→ 主线程 ──→ output_queue                │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                                │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  Worker 进程 (WorkerProc)                                │  │
│  │  ┌──────────────────────────────────────────────────┐   │  │
│  │  │ 主线程: worker_busy_loop                          │   │  │
│  │  │ rpc_broadcast_mq.dequeue() → 执行方法 → 结果入队  │   │  │
│  │  └──────────────────────────────────────────────────┘   │  │
│  │                                                         │  │
│  │  GPUModelRunner → Model → KV Cache                      │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                                │
│  rpc_broadcast_mq (共享内存) ──→ 所有 Worker                  │
│  worker_response_mq (共享内存) ──→ EngineCore                  │
└────────────────────────────────────────────────────────────────┘
```

### 1.2 单机多卡 (TP=2, PP=1)

```
┌─────────────────────────────────────────────────────────────────┐
│                        API Server 进程                          │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  AsyncLLM + AsyncMPClient                               │   │
│  │  input_socket (ROUTER) ←─┐                              │   │
│  │  output_socket (PULL) ←──┤                              │   │
│  └──────────────────────────┼──────────────────────────────┘   │
└─────────────────────────────┼──────────────────────────────────┘
                              │ ZMQ
┌─────────────────────────────▼──────────────────────────────────┐
│                     EngineCore 进程                            │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  主线程: run_busy_loop                                   │  │
│  │  Input 线程 ←─ ZMQ DEALER                               │  │
│  │  Output 线程 → ZMQ PUSH                                 │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                                │
│  ┌──────────────────────┐  ┌──────────────────────┐          │
│  │  Worker 0 (GPU 0)    │  │  Worker 1 (GPU 1)    │          │
│  │  TP rank 0           │  │  TP rank 1           │          │
│  │                      │  │                      │          │
│  │  worker_busy_loop    │  │  worker_busy_loop    │          │
│  │  ↓                   │  │  ↓                   │          │
│  │  GPUModelRunner      │  │  GPUModelRunner      │          │
│  │  Model (half)        │  │  Model (half)        │          │
│  │  KV Cache            │  │  KV Cache            │          │
│  └──────────┬───────────┘  └──────────┬───────────┘          │
│             │                         │                       │
│             └──────── NCCL ───────────┘                       │
│                  AllReduce / AllGather                        │
│                                                                │
│  rpc_broadcast_mq (共享内存) ──→ Worker 0, Worker 1           │
│  worker_response_mq[0] (共享内存) ←── Worker 0                │
│  worker_response_mq[1] (共享内存) ←── Worker 1                │
└────────────────────────────────────────────────────────────────┘
```

### 1.3 单机多卡 (TP=2, PP=2, 4 GPU)

```
┌─────────────────────────────────────────────────────────────────┐
│                        API Server 进程                          │
│  AsyncLLM + AsyncMPClient                                       │
└─────────────────────────────┬──────────────────────────────────┘
                              │ ZMQ
┌─────────────────────────────▼──────────────────────────────────┐
│                     EngineCore 进程                            │
│  主线程 + Input 线程 + Output 线程                               │
│                                                                │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  PP Stage 0                                              │  │
│  │  ┌──────────────┐  ┌──────────────┐                    │  │
│  │  │ Worker 0     │  │ Worker 1     │                    │  │
│  │  │ GPU 0        │  │ GPU 1        │                    │  │
│  │  │ TP rank 0    │  │ TP rank 1    │                    │  │
│  │  │ Layer 0-15   │  │ Layer 0-15   │                    │  │
│  │  └──────┬───────┘  └──────┬───────┘                    │  │
│  │         └──── NCCL ───────┘                            │  │
│  │              AllReduce (TP 通信)                        │  │
│  └─────────────────────┬───────────────────────────────────┘  │
│                        │ NCCL Send/Recv (PP 通信)              │
│  ┌─────────────────────▼───────────────────────────────────┐  │
│  │  PP Stage 1                                              │  │
│  │  ┌──────────────┐  ┌──────────────┐                    │  │
│  │  │ Worker 2     │  │ Worker 3     │                    │  │
│  │  │ GPU 2        │  │ GPU 3        │                    │  │
│  │  │ TP rank 0    │  │ TP rank 1    │                    │  │
│  │  │ Layer 16-31  │  │ Layer 16-31  │                    │  │
│  │  └──────┬───────┘  └──────┬───────┘                    │  │
│  │         └──── NCCL ───────┘                            │  │
│  │              AllReduce (TP 通信)                        │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                                │
│  rpc_broadcast_mq (共享内存) ──→ Worker 0,1,2,3               │
│  worker_response_mq[0..3] (共享内存) ←── Worker 0,1,2,3       │
└────────────────────────────────────────────────────────────────┘
```

### 1.4 多机多卡 (DP=2, TP=4, 2 节点)

```
┌─────────────────────────────────────────────────────────────────┐
│                         Node 0                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  API Server 进程                                         │   │
│  │  AsyncLLM + DPLBAsyncMPClient                            │   │
│  │  ┌─────────────────────────────────────────────────┐    │   │
│  │  │ 负载均衡: 选择 DP rank 0 或 DP rank 1           │    │   │
│  │  └─────────────────────────────────────────────────┘    │   │
│  └──────────────────────┬──────────────────────────────────┘   │
│                         │ ZMQ                                    │
│  ┌──────────────────────▼──────────────────────────────────┐   │
│  │  EngineCore 进程 (DP rank 0)                             │   │
│  │  主线程 + Input 线程 + Output 线程                         │   │
│  │                                                          │   │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐    │   │
│  │  │ Worker 0     │ │ Worker 1     │ │ Worker 2     │    │   │
│  │  │ GPU 0        │ │ GPU 1        │ │ GPU 2        │    │   │
│  │  │ TP rank 0    │ │ TP rank 1    │ │ TP rank 2    │    │   │
│  │  └──────────────┘ └──────────────┘ └──────────────┘    │   │
│  │         │               │               │               │   │
│  │         └───────────── NCCL ────────────┘               │   │
│  │              AllReduce (TP 通信, 节点内 NVLink)          │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  DPCoordinator 进程                                       │   │
│  │  XPUB (stats) → API Server                               │   │
│  │  PULL (stats) ← EngineCore DP0, DP1                      │   │
│  │  XPUB (wave) → EngineCore DP0, DP1                       │   │
│  └──────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
         │
         │ NCCL AllReduce (跨节点 TP 通信, InfiniBand/RoCE)
         │
┌──────────────────────────────────────────────────────────────────┐
│                         Node 1                                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  EngineCore 进程 (DP rank 1)                              │   │
│  │  主线程 + Input 线程 + Output 线程                         │   │
│  │                                                          │   │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐    │   │
│  │  │ Worker 3     │ │ Worker 4     │ │ Worker 5     │    │   │
│  │  │ GPU 0        │ │ GPU 1        │ │ GPU 2        │    │   │
│  │  │ TP rank 3    │ │ TP rank 4    │ │ TP rank 5    │    │   │
│  │  └──────────────┘ └──────────────┘ └──────────────┘    │   │
│  │         │               │               │               │   │
│  │         └───────────── NCCL ────────────┘               │   │
│  │              AllReduce (TP 通信, 节点内 NVLink)          │   │
│  └──────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
```

### 1.5 P/D 分离部署

```
┌─────────────────────────────────────────────────────────────────┐
│                      Prefill 节点                               │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  API Server + Proxy                                      │   │
│  └──────────────────────┬──────────────────────────────────┘   │
│                         │                                        │
│  ┌──────────────────────▼──────────────────────────────────┐   │
│  │  EngineCore (kv_role="kv_producer")                      │   │
│  │  ┌─────────────────────────────────────────────────┐    │   │
│  │  │ Scheduler: 只调度 prefill 请求                    │    │   │
│  │  │ KVConnector: 保存 KV 到远程存储                   │    │   │
│  │  └─────────────────────────────────────────────────┘    │   │
│  │  ┌──────────────┐                                       │   │
│  │  │ Worker (GPU) │                                       │   │
│  │  │ 计算 KV      │                                       │   │
│  │  │ save_kv_layer│                                       │   │
│  │  └──────────────┘                                       │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              │ KV 传输 (RDMA/NCCL)
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│                      Decode 节点                                │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  EngineCore (kv_role="kv_consumer")                      │   │
│  │  ┌─────────────────────────────────────────────────┐    │   │
│  │  │ Scheduler: 查询远程 KV, 分配块, 调度 decode      │    │   │
│  │  │ KVConnector: 从远程加载 KV                        │    │   │
│  │  └─────────────────────────────────────────────────┘    │   │
│  │  ┌──────────────┐                                       │   │
│  │  │ Worker (GPU) │                                       │   │
│  │  │ start_load_kv│                                       │   │
│  │  │ 逐 token 生成│                                       │   │
│  │  └──────────────┘                                       │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. 启动流程详解

### 2.1 启动序列图

```
用户执行: vllm serve model_name --tensor-parallel-size 2
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  1. CLI 入口 (vllm.entrypoints.cli.main)                        │
│     解析命令行参数 → 创建 VllmConfig                              │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  2. 创建 AsyncLLM (vllm.v1.engine.async_llm)                    │
│     │                                                           │
│     ├── 2.1 创建 InputProcessor, OutputProcessor, Renderer      │
│     │                                                           │
│     ├── 2.2 EngineCoreClient.make_async_mp_client()             │
│     │    │                                                      │
│     │    ├── 2.2.1 分配 ZMQ 地址                                │
│     │    │    input_socket:  ROUTER (bind)                      │
│     │    │    output_socket: PULL (bind)                        │
│     │    │                                                      │
│     │    └── 2.2.2 launch_core_engines()                        │
│     │         │                                                 │
│     │         ├── 创建 CoreEngineProcManager                    │
│     │         │    │                                            │
│     │         │    └── 为每个 DP rank 启动进程:                   │
│     │         │         Process(target=EngineCoreProc.run_engine_core)
    │         │         │
    │         │         ▼
    │         │    ┌─────────────────────────────────────────┐
    │         │    │  EngineCore 进程启动                      │
    │         │    │  1. 创建 Executor                        │
    │         │    │  2. 初始化 KV Cache                      │
    │         │    │  3. 创建 Scheduler                       │
    │         │    │  4. ZMQ 握手 (HELLO → READY)             │
    │         │    │  5. 启动 Input/Output 守护线程            │
    │         │    │  6. 进入 run_busy_loop                   │
    │         │    └─────────────────────────────────────────┘
    │         │
    │         └── 等待所有 EngineCore 发送 READY
    │
    └── 2.3 启动 output_handler 后台任务
         │
         ▼
    AsyncLLM 就绪，可以接受请求
```

### 2.2 EngineCore 进程内部启动

```
EngineCoreProc.run_engine_core()
    │
    ├── 1. 设置进程标题: "EngineCore" 或 "EngineCore_DP{rank}"
    │
    ├── 2. 设置信号处理器: SIGTERM, SIGINT
    │
    ├── 3. 创建 EngineCoreProc 实例
    │    │
    │    ├── 3.1 创建 Executor
    │    │    │
    │    │    │  MultiprocExecutor:
    │    │    ├── 创建 MessageQueue (rpc_broadcast_mq)
    │    │    ├── 为每个 GPU 启动 Worker 进程
    │    │    │    │
    │    │    │    └── WorkerProc.worker_main()
    │    │    │         ├── 创建 WorkerWrapperBase
    │    │    │         ├── init_worker() → 加载模型
    │    │    │         ├── init_device() → 初始化 GPU
    │    │    │         ├── load_model() → 加载权重
    │    │    │         ├── 发送 READY
    │    │    │         └── 进入 worker_busy_loop
    │    │    │
    │    │    └── 等待所有 Worker 就绪
    │    │
    │    ├── 3.2 初始化 KV Cache
    │    │    ├── get_kv_cache_specs()
    │    │    ├── determine_available_memory()
    │    │    ├── get_kv_cache_configs()
    │    │    └── initialize_from_config()
    │    │
    │    ├── 3.3 创建 Scheduler
    │    │
    │    └── 3.4 设置 step_fn
    │
    ├── 4. ZMQ 握手
    │    │
    │    │  EngineCore                    Frontend (AsyncMPClient)
    │    │     │                              │
    │    │     │──── HELLO ──────────────────→│
    │    │     │                              │
    │    │     │←─── EngineHandshakeMetadata ─│
    │    │     │     (addresses, config)      │
    │    │     │                              │
    │    │     │──── READY ──────────────────→│
    │    │     │                              │
    │
    ├── 5. 启动守护线程
    │    │
    │    ├── Input 线程 (process_input_sockets)
    │    │    ├── 创建 ZMQ DEALER socket → 连接 Frontend ROUTER
    │    │    ├── 创建 XSUB socket → 连接 DPCoordinator (可选)
    │    │    └── 循环: recv → 反序列化 → input_queue.put()
    │    │
    │    └── Output 线程 (process_output_sockets)
    │         ├── 创建 ZMQ PUSH socket → 连接 Frontend PULL
    │         ├── 创建 PUSH socket → 连接 DPCoordinator (可选)
    │         └── 循环: output_queue.get() → 序列化 → send()
    │
    └── 6. 进入 run_busy_loop
         while True:
             _process_input_queue()   # 处理输入
             _process_engine_step()   # 执行一步
```

### 2.3 Worker 进程启动

```
WorkerProc.worker_main()
    │
    ├── 1. 设置信号处理器
    │
    ├── 2. 创建 WorkerProc 实例
    │    │
    │    ├── 2.1 创建 WorkerWrapperBase
    │    │    │
    │    │    ├── 解析 worker 类 (如 GPUWorker)
    │    │    ├── 注入扩展类 (可选)
    │    │    └── 实例化 Worker
    │    │
    │    ├── 2.2 init_worker()
    │    │    └── 初始化分布式环境
    │    │
    │    ├── 2.3 init_device()
    │    │    ├── 设置 CUDA 设备
    │    │    ├── 初始化 NCCL
    │    │    ├── 创建 GPUModelRunner
    │    │    └── 初始化 WorkspaceManager
    │    │
    │    ├── 2.4 load_model()
    │    │    └── 加载模型权重到 GPU
    │    │
    │    └── 2.5 初始化消息队列
    │         ├── rpc_broadcast_mq (共享内存, 接收命令)
    │         └── worker_response_mq (共享内存, 发送结果)
    │
    ├── 3. 启动 DeathPipeMonitor 线程
    │    └── 监控父进程是否退出
    │
    ├── 4. 通过 ready_pipe 发送 READY
    │
    └── 5. 进入 worker_busy_loop
         while True:
             (method, args, kwargs, output_rank) = rpc_broadcast_mq.dequeue()
             result = getattr(worker, method)(*args, **kwargs)
             if output_rank is None or rank == output_rank:
                 worker_response_mq.enqueue(result)
```

---

## 3. 请求处理时的进程交互

### 3.1 完整请求流程图

```
┌─────────────────────────────────────────────────────────────────┐
│  Step 1: 用户发送请求                                            │
│  POST /v1/chat/completions                                      │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 2: API Server 进程                                        │
│  OpenAIServingChat.create_chat_completion()                      │
│     │                                                           │
│     ├── 渲染聊天模板, 分词                                        │
│     ├── 创建 SamplingParams                                      │
│     └── AsyncLLM.generate()                                     │
│          │                                                      │
│          ├── InputProcessor.process_inputs()                     │
│          │   → EngineCoreRequest                                 │
│          │                                                      │
│          ├── 创建 RequestOutputCollector (asyncio.Queue)         │
│          │                                                      │
│          └── engine_core.add_request_async(request)              │
│               │                                                  │
│               └── AsyncMPClient._send_input()                   │
│                    └── ZMQ ROUTER socket.send()                  │
└─────────────────────────┬───────────────────────────────────────┘
                          │ ZMQ DEALER/ROUTER
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 3: EngineCore 进程 - Input 线程                           │
│  process_input_sockets()                                        │
│     │                                                           │
│     ├── ZMQ DEALER socket.recv()                               │
│     ├── MsgpackDecoder.decode() → 反序列化                      │
│     ├── preprocess_add_request() → 创建 Request 对象            │
│     └── input_queue.put((ADD, request))                         │
└─────────────────────────┬───────────────────────────────────────┘
                          │ Python queue.Queue
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 4: EngineCore 进程 - 主线程                               │
│  _process_input_queue()                                         │
│     │                                                           │
│     ├── input_queue.get()                                       │
│     └── scheduler.add_request(request)                          │
│          └── request → waiting 队列                             │
│                                                                  │
│  _process_engine_step()                                         │
│     │                                                           │
│     ├── scheduler.schedule()                                    │
│     │    ├── Phase 1: 调度 RUNNING 请求                         │
│     │    ├── Phase 2: 调度 WAITING 请求                         │
│     │    └── 构建 SchedulerOutput                               │
│     │                                                           │
│     └── executor.execute_model(scheduler_output)                │
│          │                                                      │
│          └── collective_rpc("execute_model", ...)               │
│               │                                                  │
│               └── rpc_broadcast_mq.enqueue()  ← 共享内存广播     │
└─────────────────────────┬───────────────────────────────────────┘
                          │ 共享内存 (MessageQueue)
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 5: Worker 进程                                            │
│  worker_busy_loop()                                             │
│     │                                                           │
│     ├── rpc_broadcast_mq.dequeue()                              │
│     │   → (execute_model, scheduler_output, ...)                │
│     │                                                           │
│     ├── GPUModelRunner._update_states()                         │
│     │   → 更新 InputBatch                                      │
│     │                                                           │
│     ├── GPUModelRunner._prepare_inputs()                        │
│     │   → 构建 input_ids, positions, slot_mapping               │
│     │                                                           │
│     ├── GPUModelRunner._model_forward()                         │
│     │   │                                                       │
│     │   └── Transformer Layers:                                 │
│     │       ├── 算 Q, K, V                                      │
│     │       ├── KV 写入: key_cache[slot_mapping] = K            │
│     │       ├── Attention: flash_attn(Q, K, V, block_table)     │
│     │       └── FFN                                             │
│     │                                                           │
│     ├── GPUModelRunner.sample_tokens()                          │
│     │   → 采样 + 后处理                                         │
│     │                                                           │
│     └── worker_response_mq.enqueue(ModelRunnerOutput)           │
└─────────────────────────┬───────────────────────────────────────┘
                          │ 共享内存 (MessageQueue)
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 6: EngineCore 进程 - 主线程                               │
│  _process_engine_step() (续)                                    │
│     │                                                           │
│     ├── executor.get_output()                                   │
│     │   └── worker_response_mq.dequeue()                        │
│     │       → ModelRunnerOutput                                 │
│     │                                                           │
│     ├── scheduler.update_from_output()                          │
│     │   → 处理采样 token, 检查停止条件                           │
│     │   → EngineCoreOutputs                                     │
│     │                                                           │
│     └── output_queue.put((client_index, engine_core_outputs))   │
└─────────────────────────┬───────────────────────────────────────┘
                          │ Python queue.Queue
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 7: EngineCore 进程 - Output 线程                          │
│  process_output_sockets()                                       │
│     │                                                           │
│     ├── output_queue.get()                                      │
│     ├── MsgpackEncoder.encode() → 序列化                        │
│     └── ZMQ PUSH socket.send()                                 │
└─────────────────────────┬───────────────────────────────────────┘
                          │ ZMQ PUSH/PULL
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 8: API Server 进程 - AsyncMPClient                       │
│  EngineCoreOutputQueueTask                                      │
│     │                                                           │
│     ├── output_socket.recv_multipart()                          │
│     ├── MsgpackDecoder.decode() → EngineCoreOutputs             │
│     └── asyncio.Queue.put(engine_core_outputs)                  │
│                                                                  │
│  output_handler 任务                                             │
│     │                                                           │
│     ├── await engine_core.get_output_async()                    │
│     │   → EngineCoreOutputs                                     │
│     │                                                           │
│     └── output_processor.process_outputs()                      │
│          │                                                      │
│          ├── 反分词 (detokenize)                                │
│          ├── 构建 RequestOutput                                 │
│          └── request_collector.put(request_output)              │
└─────────────────────────┬───────────────────────────────────────┘
                          │ asyncio.Queue
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 9: API Server 进程 - generate() 协程                      │
│  AsyncLLM.generate()                                            │
│     │                                                           │
│     ├── await request_collector.get()                           │
│     │   → RequestOutput                                        │
│     │                                                           │
│     └── yield RequestOutput                                     │
│          → StreamingResponse → HTTP 客户端                      │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 请求处理时序图

```
API Server    Input Thread    Main Thread    Output Thread    Worker
    │              │              │              │              │
    │──send_req──→│              │              │              │
    │              │──input_q──→│              │              │
    │              │              │              │              │
    │              │              │──add_req──→│              │
    │              │              │  (scheduler)│              │
    │              │              │              │              │
    │              │              │──schedule──→│              │
    │              │              │              │              │
    │              │              │──broadcast_mq──────────────→│
    │              │              │              │              │
    │              │              │              │    ┌─────────┤
    │              │              │              │    │ forward │
    │              │              │              │    │ sample  │
    │              │              │              │    └─────────┤
    │              │              │              │              │
    │              │              │←──response_mq───────────────│
    │              │              │              │              │
    │              │              │──update──→│              │
    │              │              │              │              │
    │              │              │──output_q──→│              │
    │              │              │              │──send_out──→│
    │              │              │              │              │
    │←──ZMQ─────────────────────────────────────│              │
    │              │              │              │              │
    │──yield──→│              │              │              │
    │  (HTTP)   │              │              │              │
```

---

## 4. 单机多卡通信机制

### 4.1 Tensor Parallelism (TP) 通信

```
┌─────────────────────────────────────────────────────────────────┐
│  TP=2, 单机 2 GPU                                               │
│                                                                  │
│  GPU 0 (TP rank 0)                GPU 1 (TP rank 1)            │
│  ┌─────────────────────┐         ┌─────────────────────┐       │
│  │ Model (half)        │         │ Model (half)        │       │
│  │ W_q[:, :d/2]        │         │ W_q[:, d/2:]        │       │
│  │ W_k[:, :d/2]        │         │ W_k[:, d/2:]        │       │
│  │ W_v[:, :d/2]        │         │ W_v[:, d/2:]        │       │
│  └──────────┬──────────┘         └──────────┬──────────┘       │
│             │                                │                   │
│             └────────── NCCL AllReduce ──────┘                   │
│                        (NVLink 互联)                             │
│                                                                  │
│  通信操作:                                                        │
│  - Column Parallel: 各算各的, 最后 AllReduce                      │
│  - Row Parallel: 先 AllReduce, 再各算各的                        │
└─────────────────────────────────────────────────────────────────┘
```

**NCCL 通信路径选择：**
```python
# vllm/distributed/device_communicators/cuda_communicator.py
def all_reduce(self, input_):
    # 1. NCCL Symmetric Memory (最快)
    # 2. QuickReduce (AMD)
    # 3. FlashInfer
    # 4. CustomAllreduce (IPC, NVLink)
    # 5. PyNCCL (兜底)
```

### 4.2 Pipeline Parallelism (PP) 通信

```
┌─────────────────────────────────────────────────────────────────┐
│  PP=2, TP=1, 单机 2 GPU                                         │
│                                                                  │
│  GPU 0 (PP Stage 0)                 GPU 1 (PP Stage 1)         │
│  ┌─────────────────────┐          ┌─────────────────────┐      │
│  │ Layer 0-15          │          │ Layer 16-31         │      │
│  │                     │          │                     │      │
│  │ forward(input)      │          │ forward(intermediate)│     │
│  │   ↓                │          │   ↓                 │      │
│  │ intermediate ──────NCCL──────→│ intermediate        │      │
│  │                     │          │   ↓                 │      │
│  │                     │          │ output              │      │
│  └─────────────────────┘          └─────────────────────┘      │
│                                                                  │
│  通信方式: NCCL Send/Recv (非阻塞)                               │
│  isend_tensor_dict() / irecv_tensor_dict()                       │
└─────────────────────────────────────────────────────────────────┘
```

### 4.3 Worker 与 EngineCore 通信

```
┌─────────────────────────────────────────────────────────────────┐
│  共享内存通信 (MessageQueue)                                      │
│                                                                  │
│  EngineCore 进程                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  rpc_broadcast_mq.enqueue(method, args, kwargs)         │   │
│  │  │                                                      │   │
│  │  │  共享内存环形缓冲区                                    │   │
│  │  │  ┌─────┬─────┬─────┬─────┐                          │   │
│  │  │  │ msg │ msg │ ... │     │                          │   │
│  │  │  └─────┴─────┴─────┴─────┘                          │   │
│  │  │                                                      │   │
│  │  └───→ Worker 0: rpc_broadcast_mq.dequeue()            │   │
│  │  └───→ Worker 1: rpc_broadcast_mq.dequeue()            │   │
│  │  └───→ Worker 2: rpc_broadcast_mq.dequeue()            │   │
│  │                                                          │   │
│  │  Worker 0: worker_response_mq.enqueue(result)           │   │
│  │  Worker 1: worker_response_mq.enqueue(result)           │   │
│  │  Worker 2: worker_response_mq.enqueue(result)           │   │
│  │       │                                                  │   │
│  │       └──→ EngineCore: response_mqs[i].dequeue()        │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│  特点:                                                           │
│  - 单写多读广播 (rpc_broadcast_mq)                               │
│  - 单写单读 (worker_response_mq, 每 Worker 一个)                 │
│  - 零拷贝: 大张量通过共享内存直接传递                              │
│  - 低延迟: 无序列化开销                                          │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. 多机多卡通信机制

### 5.1 跨节点 TP 通信

```
┌─────────────────────────────────────────────────────────────────┐
│  Node 0                                    Node 1               │
│  ┌──────────────────────┐                 ┌──────────────────┐ │
│  │ GPU 0 (TP rank 0)    │                 │ GPU 2 (TP rank 2)│ │
│  │ GPU 1 (TP rank 1)    │                 │ GPU 3 (TP rank 3)│ │
│  └──────────┬───────────┘                 └──────────┬───────┘ │
│             │                                        │         │
│             └──────────── NCCL AllReduce ─────────────┘         │
│                          (InfiniBand / RoCE)                    │
│                                                                  │
│  NCCL 自动选择通信路径:                                           │
│  - 同节点: NVLink (600 GB/s)                                    │
│  - 跨节点: InfiniBand (400 Gb/s) 或 RoCE                       │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 跨节点 DP 通信

```
┌─────────────────────────────────────────────────────────────────┐
│  Node 0 (DP rank 0)                       Node 1 (DP rank 1)   │
│  ┌──────────────────────┐                 ┌──────────────────┐ │
│  │ EngineCore DP0       │                 │ EngineCore DP1   │ │
│  │                      │                 │                  │ │
│  │ Input 线程 ←─ ZMQ   │                 │ Input 线程 ←─ ZMQ│ │
│  │ Output 线程 → ZMQ   │                 │ Output 线程 → ZMQ│ │
│  └──────────┬───────────┘                 └──────────┬───────┘ │
│             │                                        │         │
│             └──────── ZMQ (TCP) ─────────────────────┘         │
│                          │                                      │
│             ┌────────────▼────────────┐                        │
│             │   DPCoordinator 进程     │                        │
│             │   (Node 0 上)           │                        │
│             │                         │                        │
│             │   XPUB → stats → API    │                        │
│             │   PULL ← stats ← engines│                        │
│             │   XPUB → wave → engines │                        │
│             └─────────────────────────┘                        │
│                                                                  │
│  DP 同步: 每 32 步 AllReduce 检查全局状态                        │
│  通信: NCCL AllReduce (跨节点)                                   │
└─────────────────────────────────────────────────────────────────┘
```

### 5.3 P/D 分离通信

```
┌─────────────────────────────────────────────────────────────────┐
│  Prefill 节点                              Decode 节点          │
│  ┌──────────────────────┐                 ┌──────────────────┐ │
│  │ EngineCore           │                 │ EngineCore       │ │
│  │ kv_role=producer     │                 │ kv_role=consumer │ │
│  │                      │                 │                  │ │
│  │ KVConnector          │                 │ KVConnector      │ │
│  │ ┌──────────────────┐ │    RDMA/NCCL   │ ┌──────────────┐ │ │
│  │ │ NIXL Agent       │ │───────────────→│ │ NIXL Agent   │ │ │
│  │ │ save_kv_layer()  │ │                │ │ start_load() │ │ │
│  │ └──────────────────┘ │                │ └──────────────┘ │ │
│  └──────────────────────┘                └──────────────────┘ │
│                                                                  │
│  ZMQ 握手:                                                       │
│  D → P: GET_META_MSG                                             │
│  P → D: NixlHandshakePayload (agent metadata, KV addresses)     │
│  D: add_remote_agent()                                           │
│                                                                  │
│  数据传输: RDMA READ (零拷贝)                                    │
│  租约机制: P 设置 30s 租约, D 定期发心跳续租                      │
└─────────────────────────────────────────────────────────────────┘
```

### 5.4 DeepSeek V4 四机 32 卡部署详解

#### 5.4.1 部署架构总览

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    DeepSeek V4 四机 32 卡部署架构                          │
│                                                                          │
│  配置: DP=32, TP=1, EP=32 (EP = DP × PCP × TP)                         │
│  每节点: 8 GPU (H100/A100)                                              │
│  模型: DeepSeek V4 (MoE + MLA, 128K 上下文)                             │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                        API Server 集群                           │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │    │
│  │  │ API Srv 0│  │ API Srv 1│  │ API Srv 2│  │ API Srv 3│       │    │
│  │  │ (Node 0) │  │ (Node 1) │  │ (Node 2) │  │ (Node 3) │       │    │
│  │  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘       │    │
│  └───────┼──────────────┼──────────────┼──────────────┼─────────────┘    │
│          │              │              │              │                   │
│          ▼              ▼              ▼              ▼                   │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    DPCoordinator 进程 (Node 0)                   │    │
│  │  XPUB → stats → API Servers                                      │    │
│  │  PULL ← stats ← 32 个 DP Engine                                  │    │
│  │  XPUB → wave → 32 个 DP Engine                                    │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    EngineCore 进程 × 32                          │    │
│  │  (每 GPU 一个 EngineCore，每个是独立的 DP rank)                    │    │
│  │                                                                  │    │
│  │  Node 0:                    Node 1:                              │    │
│  │  ┌───────────────────┐     ┌───────────────────┐                │    │
│  │  │ EngineCore DP 0   │     │ EngineCore DP 8   │                │    │
│  │  │ EngineCore DP 1   │     │ EngineCore DP 9   │                │    │
│  │  │ ...               │     │ ...               │                │    │
│  │  │ EngineCore DP 7   │     │ EngineCore DP 15  │                │    │
│  │  └───────────────────┘     └───────────────────┘                │    │
│  │                                                                  │    │
│  │  Node 2:                    Node 3:                              │    │
│  │  ┌───────────────────┐     ┌───────────────────┐                │    │
│  │  │ EngineCore DP 16  │     │ EngineCore DP 24  │                │    │
│  │  │ ...               │     │ ...               │                │    │
│  │  │ EngineCore DP 23  │     │ EngineCore DP 31  │                │    │
│  │  └───────────────────┘     └───────────────────┘                │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    Worker 进程 × 32                              │    │
│  │  每个 Worker:                                                    │    │
│  │  ┌─────────────────────────────────────────────────────────┐    │    │
│  │  │  GPUModelRunner                                         │    │    │
│  │  │  ├── 完整 DeepSeek V4 模型 (权重)                        │    │    │
│  │  │  ├── 416 / 32 = 13 个本地物理专家                       │    │    │
│  │  │  ├── MLA 注意力层                                       │    │    │
│  │  │  ├── KV Cache (MLA 压缩格式, 576 维/token)              │    │    │
│  │  │  └── Indexer (稀疏注意力, V4)                           │    │    │
│  │  └─────────────────────────────────────────────────────────┘    │    │
│  └─────────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────────┘
```

#### 5.4.2 启动流程

```
四机 32 卡 DeepSeek V4 启动流程:

Node 0 (主节点):
┌─────────────────────────────────────────────────────────────────┐
│  vllm serve deepseek-ai/DeepSeek-V4 \                          │
│    --tensor-parallel-size 1 \                                   │
│    --enable-expert-parallel \                                   │
│    --data-parallel-size 32 \                                    │
│    --data-parallel-size-local 8 \                               │
│    --data-parallel-address 192.168.1.100 \                      │
│    --data-parallel-rpc-port 13345 \                             │
│    --api-server-count=8                                         │
│                                                                  │
│  启动: 8 EngineCore 进程 + 8 Worker 进程 + DPCoordinator       │
│        + 8 API Server 实例                                      │
└─────────────────────────────────────────────────────────────────┘

Node 1-3 (工作节点):
┌─────────────────────────────────────────────────────────────────┐
│  vllm serve deepseek-ai/DeepSeek-V4 \                          │
│    --tensor-parallel-size 1 \                                   │
│    --enable-expert-parallel \                                   │
│    --data-parallel-size 32 \                                    │
│    --data-parallel-size-local 8 \                               │
│    --data-parallel-start-rank 8/16/24 \                         │
│    --data-parallel-address 192.168.1.100 \                      │
│    --data-parallel-rpc-port 13345 \                             │
│    --headless                                                   │
│                                                                  │
│  启动: 8 EngineCore 进程 + 8 Worker 进程                        │
└─────────────────────────────────────────────────────────────────┘
```

#### 5.4.3 请求处理全流程

```
Step 1: 请求到达 API Server
  POST /v1/chat/completions → Node 0 的 API Server

Step 2: 负载均衡选择 DP rank
  DPLBAsyncMPClient 选择负载最低的 DP rank
  评分 = waiting × 4 + running
  例如: 选择 DP rank 12 (Node 2, GPU 4)

Step 3: EngineCore DP 12 接收请求
  Input 线程 → input_queue → 主线程
  scheduler.add_request(request) → waiting 队列

Step 4: 调度器调度
  scheduler.schedule()
  ├── 查前缀缓存: get_computed_blocks()
  ├── 分配 MLA KV 块: allocate_slots()
  └── 构建 SchedulerOutput

Step 5: Worker 执行前向传播
  对每层 DeepSeekV4DecoderLayer:
  ├── MLA Attention:
  │   ├── Q/KV 压缩投影 → 576 维
  │   ├── KV 写入: key_cache[slot_mapping] = K
  │   ├── FlashMLA 注意力计算
  │   └── DCP: AllGather + LSE 合并 (如果启用)
  │
  └── MoE:
      ├── Gate 计算 → Top-K 路由
      ├── EP Dispatch: All-to-All (DeepEP LL)
      ├── 专家计算 (本地 9 个专家)
      ├── EP Combine: All-to-All (DeepEP LL)
      └── 共享专家

Step 6: 输出 → API Server → HTTP 客户端
```

#### 5.4.4 EP All-to-All 通信

```
EP=32, 每 GPU 9 个物理专家

Dispatch:
  GPU 0: [token A → 专家 5 (GPU 5)]
         [token B → 专家 12 (GPU 12)]
         [token C → 专家 28 (GPU 28)]

Combine:
  GPU 0: [← 专家 0 的结果 (来自 GPU 0)]
         [← 专家 15 的结果 (来自 GPU 15)]

通信后端:
  DeepEP LL: 低延迟，decode 优化
  DeepEP HT: 高吞吐，prefill 优化
  FP8 Dispatch: 减少 2x 通信量
```

#### 5.4.5 性能瓶颈分析

| 瓶颈 | 原因 | 影响指标 | 优化方案 |
|------|------|----------|----------|
| **EP All-to-All** | 每步 decode 需要跨节点通信 | TPOT | DeepEP LL + FP8 + EPLB |
| **KV Cache 读取** | 每步读取 144 MB (128K 上下文) | TPOT | FP8 KV + DCP + Sparse Attn |
| **Prefill 计算** | 128K prefill ≈ 83T FLOPs | TTFT | Chunked Prefill + PCP |
| **KV 传输 (PD)** | 144 MB/请求 RDMA 传输 | TTFT | NIXL RDMA + 流式传输 |
| **负载不均衡** | 长短请求混合 | 尾延迟 | DPLB + EPLB + Preemption |

**通信量估算 (每步 decode):**
```
EP 通信: ~228 KB (Dispatch + Combine)
KV Cache 读取: 144 MB
比例: KV Cache 读取是 EP 通信的 631 倍

结论: KV Cache 读取是主要瓶颈，EP 通信是次要瓶颈
```

---

## 6. 各进程/线程详解

### 6.1 进程列表

| 进程 | 启动方式 | 职责 | 数量 |
|------|----------|------|------|
| **API Server** | 用户启动 | HTTP 服务、请求路由 | 1+ |
| **EngineCore** | multiprocessing.Process | 调度、执行、状态管理 | 每 DP rank 1 个 |
| **Worker** | multiprocessing.Process | GPU 推理、模型前向 | 每 GPU 1 个 |
| **DPCoordinator** | multiprocessing.Process | DP 协调、负载统计 | DP>1 时 1 个 |

### 6.2 线程列表

| 进程 | 线程 | 类型 | 职责 |
|------|------|------|------|
| **API Server** | asyncio event loop | 主线程 | 处理 HTTP 请求 |
| | output_handler | asyncio.Task | 处理引擎输出 |
| | EngineCoreOutputQueueTask | asyncio.Task | 从 ZMQ 读取输出 |
| | stats_update_task | asyncio.Task | 更新 DP 统计 (DP 模式) |
| **EngineCore** | run_busy_loop | 主线程 | 调度+执行循环 |
| | process_input_sockets | daemon | ZMQ 输入处理 |
| | process_output_sockets | daemon | ZMQ 输出处理 |
| **Worker** | worker_busy_loop | 主线程 | 执行 RPC 命令 |
| | DeathPipeMonitor | daemon | 监控父进程存活 |
| | AsyncOutputCopyThread | daemon | 异步输出拷贝 (可选) |

### 6.3 各进程的生命周期

```
API Server 进程:
  启动 → 创建 AsyncLLM → 启动 EngineCore 进程 → 等待 READY
  → 启动 output_handler → 接受请求 → 处理请求 → 关闭

EngineCore 进程:
  启动 → 创建 Executor → 启动 Worker 进程 → 初始化 KV Cache
  → 创建 Scheduler → ZMQ 握手 → 启动 IO 线程 → run_busy_loop
  → SIGTERM → 排空请求 → 关闭

Worker 进程:
  启动 → init_worker → init_device → load_model → 发送 READY
  → worker_busy_loop → 收到关闭信号 → 清理 → 退出

DPCoordinator 进程:
  启动 → 创建 ZMQ 套接字 → 处理统计和 wave → 关闭
```

---

## 7. ZMQ 通信详解

### 7.1 ZMQ 套接字类型

| 套接字 | 方向 | 用途 |
|--------|------|------|
| **ROUTER** (Frontend) | 接收请求 | API Server 接收来自 EngineCore 的身份帧和请求 |
| **DEALER** (EngineCore) | 发送请求 | EngineCore 发送请求到 Frontend |
| **PUSH/PULL** | 单向输出 | EngineCore 发送输出到 Frontend |
| **XPUB/XSUB** | 发布/订阅 | DPCoordinator 发布统计和 wave 状态 |

### 7.2 ZMQ 消息格式

```
请求消息:
  [identity, request_type, serialized_data]
  │           │              │
  │           │              └── MsgpackEncoder 编码的 EngineCoreRequest
  │           └── ADD / ABORT / UTILITY 等
  └── 2 字节 DP rank 标识

输出消息:
  [client_index, serialized_data]
  │              │
  │              └── MsgpackEncoder 编码的 EngineCoreOutputs
  └── 客户端索引
```

### 7.3 握手协议

```
EngineCore                         Frontend (AsyncMPClient)
    │                                   │
    │──── HELLO (identity) ────────────→│
    │                                   │
    │←─── EngineHandshakeMetadata ──────│
    │     (input_addresses,             │
    │      output_addresses,            │
    │      parallel_config)             │
    │                                   │
    │──── READY (identity) ────────────→│
    │     (max_model_len,               │
    │      num_gpu_blocks)              │
    │                                   │
    │     连接建立完成                    │
```

---

## 8. 共享内存通信详解

### 8.1 MessageQueue 实现

```python
# vllm/distributed/device_communicators/shm_broadcast.py
class MessageQueue:
    # 共享内存环形缓冲区
    # 单写多读 (rpc_broadcast_mq)
    # 或单写单读 (worker_response_mq)

    def enqueue(self, data):
        # 1. 序列化 data 到共享内存
        # 2. 更新写指针
        # 3. 通过 ZMQ PUB 通知读者

    def dequeue(self):
        # 1. 等待数据可用 (spin + ZMQ SUB 通知)
        # 2. 从共享内存读取
        # 3. 反序列化
        # 4. 更新读指针
```

### 8.2 零拷贝传递

```
大张量传递:
  Worker 0: tensor 在 GPU 0 上
      │
      ├── 序列化: 只传元数据 (形状、dtype、指针)
      │
      └── 共享内存: 其他进程通过 IPC handle 直接访问

  Worker 1: 通过 cudaIpcOpenMemHandle 直接访问 GPU 0 的内存
  → 零拷贝！
```

---

## 9. 关键代码索引

### 9.1 进程管理

| 文件 | 关键类/方法 | 用途 |
|------|-------------|------|
| `vllm/v1/engine/core.py` | `EngineCoreProc` | 引擎核心进程 |
| | `run_busy_loop()` | 主循环 |
| | `process_input_sockets()` | 输入线程 |
| | `process_output_sockets()` | 输出线程 |
| `vllm/v1/engine/utils.py` | `CoreEngineProcManager` | 进程管理器 |
| | `CoreEngineActorManager` | Ray Actor 管理器 |
| | `launch_core_engines()` | 启动引擎进程 |
| `vllm/v1/engine/coordinator.py` | `DPCoordinator` | DP 协调器 |

### 9.2 Worker 管理

| 文件 | 关键类/方法 | 用途 |
|------|-------------|------|
| `vllm/v1/executor/multiproc_executor.py` | `MultiprocExecutor` | 多进程执行器 |
| | `WorkerProc` | Worker 进程 |
| | `worker_busy_loop()` | Worker 主循环 |
| `vllm/v1/executor/ray_executor.py` | `RayDistributedExecutor` | Ray 执行器 |
| `vllm/v1/worker/worker_base.py` | `WorkerWrapperBase` | Worker 包装器 |

### 9.3 通信

| 文件 | 关键类/方法 | 用途 |
|------|-------------|------|
| `vllm/v1/engine/core_client.py` | `AsyncMPClient` | 异步 ZMQ 客户端 |
| | `SyncMPClient` | 同步 ZMQ 客户端 |
| `vllm/distributed/device_communicators/shm_broadcast.py` | `MessageQueue` | 共享内存消息队列 |
| `vllm/distributed/device_communicators/cuda_communicator.py` | `CudaCommunicator` | NCCL 通信 |
| `vllm/distributed/device_communicators/custom_all_reduce.py` | `CustomAllreduce` | IPC AllReduce |
