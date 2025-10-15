# vLLM KV Cache 管理机制详解

## 目录
- [1. 概述](#1-概述)
- [2. 入口示例](#2-入口示例)
- [3. KV Cache 初始化流程](#3-kv-cache-初始化流程)
- [4. KV Cache 架构设计](#4-kv-cache-架构设计)
- [5. KV Cache 管理和调度](#5-kv-cache-管理和调度)
- [6. KV Cache 复用机制(Prefix Caching)](#6-kv-cache-复用机制prefix-caching)
- [7. LMCache 对接](#7-lmcache-对接)
- [8. 关键类和方法](#8-关键类和方法)
- [9. 总结](#9-总结)

---

## 1. 概述

vLLM 是一个高性能的大语言模型推理框架，其核心优化之一是高效的 KV Cache 管理机制。KV Cache (Key-Value Cache) 用于存储 Transformer 模型中注意力机制的键值对，避免重复计算，显著提升推理性能。

本文档基于 vLLM V1 架构，详细介绍 KV Cache 的初始化、管理、调度、复用以及与外部缓存系统（如 LMCache）的对接机制。

**核心特性：**
- **分块管理（Block Management）**：KV Cache 以固定大小的块（block）为单位进行分配和管理
- **前缀缓存（Prefix Caching）**：通过哈希机制复用相同前缀的计算结果
- **动态调度**：根据请求优先级和资源可用性动态分配 KV Cache
- **外部缓存对接**：支持与 LMCache 等外部缓存系统集成，实现跨实例的 KV Cache 共享

---

## 2. 入口示例

让我们从一个简单的示例开始，了解 vLLM 的基本使用方式：

```python
from vllm import LLM, SamplingParams

if __name__ == "__main__":
    # 准备输入的 prompts
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    # 创建采样参数
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

    # 创建 LLM 实例，在这个过程中也创建了 llm_engine
    # gpu_memory_utilization 控制 KV Cache 的内存占用比例
    llm = LLM(
        model="facebook/opt-125m",
        gpu_memory_utilization=0.9,  # 90% GPU 内存用于 KV Cache
    )

    # 执行 offline batching 推理，得到这批 prompts 的输出
    outputs = llm.generate(prompts, sampling_params)

    # 打印输出
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

**关键参数说明：**
- `gpu_memory_utilization`：控制 GPU 内存中分配给 KV Cache 的比例（默认 0.9）
- `kv_cache_memory_bytes`：直接指定 KV Cache 的内存大小（字节）
- `block_size`：KV Cache 块的大小（token 数量，默认 16）
- `enable_prefix_caching`：是否启用前缀缓存（默认 False）

---

## 3. KV Cache 初始化流程

### 3.1 LLM 类初始化

KV Cache 的初始化从 `LLM` 类的构造函数开始：

**文件位置：** `vllm/entrypoints/llm.py:188-328`

```python
class LLM:
    def __init__(self, model: str, ...):
        # 1. 创建 EngineArgs，包含所有配置参数
        engine_args = EngineArgs(
            model=model,
            gpu_memory_utilization=gpu_memory_utilization,
            kv_cache_memory_bytes=kv_cache_memory_bytes,
            ...
        )

        # 2. 从 EngineArgs 创建 LLMEngine（自动选择 V0 或 V1）
        self.llm_engine = LLMEngine.from_engine_args(
            engine_args=engine_args,
            usage_context=UsageContext.LLM_CLASS
        )
```

### 3.2 LLMEngine 初始化

**文件位置：** `vllm/v1/engine/llm_engine.py:46-150`

```python
class LLMEngine:
    def __init__(self, vllm_config: VllmConfig, ...):
        self.vllm_config = vllm_config
        self.cache_config = vllm_config.cache_config

        # 创建 Processor（处理输入）
        self.processor = Processor(self.vllm_config, tokenizer)

        # 创建 EngineCore（核心执行引擎）
        self.engine_core = EngineCoreClient.make_client(
            vllm_config=vllm_config,
            executor_class=executor_class,
            ...
        )
```

### 3.3 KV Cache 配置生成

KV Cache 的配置在 Worker 初始化时生成：

**文件位置：** `vllm/v1/core/kv_cache_utils.py:1211-1310`

```python
def get_kv_cache_configs(
    vllm_config: VllmConfig,
    kv_cache_specs: list[dict[str, KVCacheSpec]],
    available_memory: list[int],
) -> list[KVCacheConfig]:
    """
    生成 KV Cache 配置的关键步骤：
    1. 检查每个 worker 的可用内存是否足够
    2. 合并所有 worker 的 KV Cache 规格
    3. 根据层的比例生成 KV Cache 组
    4. 为每个 worker 生成 KV Cache 配置
    5. 统一所有 worker 的块数量为最小值
    """

    # 1. 检查内存是否足够
    for kv_cache_spec_one_worker, available_memory_one_worker in zip(
        kv_cache_specs, available_memory
    ):
        check_enough_kv_cache_memory(
            vllm_config, kv_cache_spec_one_worker, available_memory_one_worker
        )

    # 2. 合并所有 worker 的 KV Cache 规格
    merged_kv_cache_specs: dict[str, KVCacheSpec] = {}
    for kv_cache_spec_one_worker in kv_cache_specs:
        for layer_name, layer_spec in kv_cache_spec_one_worker.items():
            if layer_name not in merged_kv_cache_specs:
                merged_kv_cache_specs[layer_name] = layer_spec

    # 3. 生成 KV Cache 组
    global_kv_cache_groups = get_kv_cache_groups(vllm_config, merged_kv_cache_specs)

    # 4. 为每个 worker 生成配置
    kv_cache_configs: list[KVCacheConfig] = []
    for kv_cache_spec_one_worker, available_memory_one_worker in zip(
        kv_cache_specs, available_memory
    ):
        kv_cache_config = get_kv_cache_config_from_groups(
            vllm_config,
            kv_cache_groups_one_worker,
            kv_cache_spec_one_worker,
            available_memory_one_worker,
        )
        kv_cache_configs.append(kv_cache_config)

    # 5. 统一块数量
    min_num_blocks = min(cfg.num_blocks for cfg in kv_cache_configs)
    for kv_cache_config in kv_cache_configs:
        kv_cache_config.num_blocks = min_num_blocks

    return kv_cache_configs
```

**关键概念：**

1. **KVCacheSpec**: 定义单个层的 KV Cache 规格（块大小、头数、头维度等）
2. **KVCacheGroup**: 将具有相同规格的层分组，共享块表
3. **KVCacheConfig**: 完整的 KV Cache 配置，包含块数量、张量大小等

### 3.4 Scheduler 初始化 KV Cache Manager

**文件位置：** `vllm/v1/core/sched/scheduler.py:40-169`

```python
class Scheduler:
    def __init__(self, vllm_config: VllmConfig, kv_cache_config: KVCacheConfig, ...):
        # 创建 KV Cache Manager
        self.kv_cache_manager = KVCacheManager(
            kv_cache_config=kv_cache_config,
            max_model_len=self.max_model_len,
            enable_caching=self.cache_config.enable_prefix_caching,
            use_eagle=self.use_eagle,  # EAGLE 推测解码
            log_stats=self.log_stats,
            enable_kv_cache_events=self.enable_kv_cache_events,
            dcp_world_size=self.dcp_world_size,
        )
```

### 3.5 KVCacheManager 初始化

**文件位置：** `vllm/v1/core/kv_cache_manager.py:81-133`

```python
class KVCacheManager:
    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        enable_caching: bool = True,
        use_eagle: bool = False,
        log_stats: bool = False,
        enable_kv_cache_events: bool = False,
        dcp_world_size: int = 1,
    ) -> None:
        self.max_model_len = max_model_len
        self.enable_caching = enable_caching

        # 获取块大小
        if self.enable_caching:
            self.block_size = kv_cache_config.kv_cache_groups[0].kv_cache_spec.block_size
            if dcp_world_size > 1:
                self.block_size *= dcp_world_size

        # 创建 KV Cache Coordinator（协调器）
        self.coordinator = get_kv_cache_coordinator(
            kv_cache_config=kv_cache_config,
            max_model_len=self.max_model_len,
            use_eagle=self.use_eagle,
            enable_caching=self.enable_caching,
            enable_kv_cache_events=enable_kv_cache_events,
            dcp_world_size=dcp_world_size,
        )

        # BlockPool：管理所有 KV Cache 块
        self.block_pool = self.coordinator.block_pool
```

### 3.6 BlockPool 初始化

**文件位置：** `vllm/v1/core/block_pool.py:125-168`

```python
class BlockPool:
    def __init__(
        self,
        num_gpu_blocks: int,
        enable_caching: bool,
        enable_kv_cache_events: bool = False,
    ):
        self.num_gpu_blocks = num_gpu_blocks
        self.enable_caching = enable_caching

        # 创建所有 KV Cache 块
        self.blocks: list[KVCacheBlock] = [
            KVCacheBlock(idx) for idx in range(num_gpu_blocks)
        ]

        # 空闲块队列（双向链表，按 LRU 顺序）
        self.free_block_queue = FreeKVCacheBlockQueue(self.blocks)

        # 缓存的块映射（block_hash -> KVCacheBlock）
        self.cached_block_hash_to_block: BlockHashToBlockMap = BlockHashToBlockMap()

        # 创建 null_block（占位符，block_id=0）
        self.null_block = self.free_block_queue.popleft()
        self.null_block.is_null = True
```

**初始化流程总结：**

```
LLM.__init__()
  └─> LLMEngine.from_engine_args()
       └─> LLMEngine.__init__()
            ├─> Processor.__init__()
            └─> EngineCoreClient.make_client()
                 └─> EngineCore.__init__()
                      └─> Scheduler.__init__()
                           └─> KVCacheManager.__init__()
                                ├─> get_kv_cache_coordinator()
                                │    └─> UnitaryKVCacheCoordinator (单一类型)
                                │    └─> HybridKVCacheCoordinator (混合类型)
                                │    └─> KVCacheCoordinatorNoPrefixCache (无缓存)
                                └─> BlockPool.__init__()
                                     ├─> 创建所有 KVCacheBlock
                                     ├─> FreeKVCacheBlockQueue (空闲块队列)
                                     └─> BlockHashToBlockMap (哈希映射)
```

---

## 4. KV Cache 架构设计

### 4.1 核心数据结构

#### 4.1.1 KVCacheBlock

**文件位置：** `vllm/v1/core/kv_cache_utils.py:103-150`

```python
@dataclass
class KVCacheBlock:
    """KV Cache 块元数据"""

    # 块 ID (0 到 num_gpu_blocks - 1)
    block_id: int

    # 引用计数（支持多个请求共享）
    ref_cnt: int = 0

    # 块哈希（用于前缀缓存，只有满块才有哈希）
    _block_hash: BlockHashWithGroupId | None = None

    # 双向链表指针（用于空闲块队列）
    prev_free_block: "KVCacheBlock | None" = None
    next_free_block: "KVCacheBlock | None" = None

    # 是否为 null 块（占位符，永不缓存）
    is_null: bool = False
```

**关键属性说明：**
- `block_id`: 物理块 ID，对应 GPU 内存中的实际位置
- `ref_cnt`: 引用计数，支持多个请求共享同一个块（前缀缓存）
- `block_hash`: 块的哈希值（block_hash + group_id），用于查找缓存
- `prev_free_block/next_free_block`: 用于构建 LRU 空闲块队列

#### 4.1.2 BlockPool

**文件位置：** `vllm/v1/core/block_pool.py:125-426`

```python
class BlockPool:
    """
    BlockPool 管理所有 KV Cache 块

    核心功能：
    1. 分配新块（get_new_blocks）
    2. 释放块（free_blocks）
    3. 缓存满块（cache_full_blocks）
    4. 查找缓存块（get_cached_block）
    5. LRU 驱逐（_maybe_evict_cached_block）
    """

    def __init__(self, num_gpu_blocks: int, enable_caching: bool, ...):
        # 所有块
        self.blocks: list[KVCacheBlock] = [...]

        # 空闲块队列（LRU 顺序）
        self.free_block_queue = FreeKVCacheBlockQueue(self.blocks)

        # 哈希到块的映射
        self.cached_block_hash_to_block: BlockHashToBlockMap = BlockHashToBlockMap()

    def get_cached_block(
        self, block_hash: BlockHash, kv_cache_group_ids: list[int]
    ) -> list[KVCacheBlock] | None:
        """根据哈希查找缓存块"""
        cached_blocks = []
        for group_id in kv_cache_group_ids:
            block_hash_with_group_id = make_block_hash_with_group_id(
                block_hash, group_id
            )
            block = self.cached_block_hash_to_block.get_one_block(
                block_hash_with_group_id
            )
            if not block:
                return None
            cached_blocks.append(block)
        return cached_blocks
```

#### 4.1.3 KVCacheManager

**文件位置：** `vllm/v1/core/kv_cache_manager.py:81-405`

```python
class KVCacheManager:
    """
    KV Cache 管理器，Scheduler 和底层 BlockPool 之间的接口

    核心方法：
    - get_computed_blocks(): 获取已计算的块（前缀缓存命中）
    - allocate_slots(): 为请求分配新的 KV Cache 槽位
    - free(): 释放请求的所有块
    - cache_blocks(): 缓存请求的满块
    """

    def __init__(self, kv_cache_config: KVCacheConfig, ...):
        self.block_size = kv_cache_config.kv_cache_groups[0].kv_cache_spec.block_size

        # Coordinator：协调多个 KV Cache 组
        self.coordinator = get_kv_cache_coordinator(...)

        # BlockPool：实际的块管理
        self.block_pool = self.coordinator.block_pool
```

#### 4.1.4 KVCacheCoordinator

**文件位置：** `vllm/v1/core/kv_cache_coordinator.py:16-479`

```python
class KVCacheCoordinator(ABC):
    """
    协调不同 KV Cache 组的抽象基类

    子类：
    - UnitaryKVCacheCoordinator: 单一类型（所有层相同）
    - HybridKVCacheCoordinator: 混合类型（如 full attention + sliding window）
    - KVCacheCoordinatorNoPrefixCache: 禁用前缀缓存
    """

    def __init__(self, kv_cache_config: KVCacheConfig, ...):
        # BlockPool（所有组共享）
        self.block_pool = BlockPool(...)

        # 每个组的管理器
        self.single_type_managers = tuple(
            get_manager_for_kv_cache_spec(
                kv_cache_spec=kv_cache_group.kv_cache_spec,
                block_pool=self.block_pool,
                kv_cache_group_id=i,
                ...
            )
            for i, kv_cache_group in enumerate(kv_cache_config.kv_cache_groups)
        )

    @abstractmethod
    def find_longest_cache_hit(
        self, block_hashes: list[BlockHash], max_cache_hit_length: int
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        """查找最长的缓存命中"""
        pass
```

### 4.2 架构层次

```
┌─────────────────────────────────────────────────────────────┐
│                         Scheduler                            │
│  (调度请求，分配 token budget，管理运行/等待队列)              │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    KVCacheManager                            │
│  - get_computed_blocks() (前缀缓存查找)                      │
│  - allocate_slots() (分配 KV Cache 槽位)                     │
│  - free() (释放块)                                           │
│  - cache_blocks() (缓存满块)                                 │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                 KVCacheCoordinator                           │
│  协调多个 KV Cache 组（如 full attention + sliding window） │
│  - UnitaryKVCacheCoordinator (单一类型)                      │
│  - HybridKVCacheCoordinator (混合类型)                       │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                      BlockPool                               │
│  - blocks: list[KVCacheBlock] (所有块)                       │
│  - free_block_queue (LRU 空闲队列)                           │
│  - cached_block_hash_to_block (哈希映射)                     │
│                                                               │
│  核心操作：                                                   │
│  - get_new_blocks() (分配新块)                               │
│  - free_blocks() (释放块)                                    │
│  - cache_full_blocks() (缓存满块)                            │
│  - get_cached_block() (查找缓存)                             │
│  - touch() (增加引用计数)                                    │
└─────────────────────────────────────────────────────────────┘
```

### 4.3 KV Cache 组（KVCacheGroup）

vLLM 支持混合注意力类型的模型（如 Gemma3，部分层使用 full attention，部分使用 sliding window）。为了高效管理，vLLM 将具有相同 KV Cache 规格的层分组：

```python
# 示例：Gemma3 模型的 KV Cache 组
# - 5 层 sliding window attention
# - 1 层 full attention
# - 重复这个模式

kv_cache_groups = [
    KVCacheGroupSpec(
        layer_names=['layer.0', 'layer.6', ...],  # 每 6 层中的第 0 层
        kv_cache_spec=FullAttentionSpec(block_size=16, ...)
    ),
    KVCacheGroupSpec(
        layer_names=['layer.1', 'layer.7', ...],  # 每 6 层中的第 1 层
        kv_cache_spec=SlidingWindowSpec(block_size=16, window_size=4096, ...)
    ),
    ...
]
```

**分组策略：**
1. 相同类型的层分为同一组
2. 组内层数量相同（不足则补 padding）
3. 每个组共享一个块表（block table）

---

## 5. KV Cache 管理和调度

### 5.1 请求调度流程

**文件位置：** `vllm/v1/core/sched/scheduler.py:172-663`

```python
class Scheduler:
    def schedule(self) -> SchedulerOutput:
        """
        调度算法核心步骤：
        1. 调度 RUNNING 请求
        2. 调度 WAITING 请求
        3. 处理抢占（preemption）
        4. 收集 KV Cache 事件
        """

        scheduled_new_reqs: list[Request] = []
        scheduled_running_reqs: list[Request] = []
        preempted_reqs: list[Request] = []

        req_to_new_blocks: dict[str, KVCacheBlocks] = {}
        num_scheduled_tokens: dict[str, int] = {}
        token_budget = self.max_num_scheduled_tokens

        # 步骤 1: 调度 RUNNING 请求
        req_index = 0
        while req_index < len(self.running) and token_budget > 0:
            request = self.running[req_index]

            # 计算需要调度的 token 数量
            num_new_tokens = min(
                request.num_tokens_with_spec - request.num_computed_tokens,
                token_budget
            )

            # 为请求分配 KV Cache 槽位
            while True:
                new_blocks = self.kv_cache_manager.allocate_slots(
                    request,
                    num_new_tokens,
                    num_lookahead_tokens=self.num_lookahead_tokens,
                )

                if new_blocks is not None:
                    break  # 分配成功

                # 分配失败，抢占低优先级请求
                preempted_req = self.running.pop()  # FCFS: 抢占最后一个
                self.kv_cache_manager.free(preempted_req)
                preempted_req.num_computed_tokens = 0
                self.waiting.prepend_request(preempted_req)
                preempted_reqs.append(preempted_req)

            # 成功调度
            scheduled_running_reqs.append(request)
            req_to_new_blocks[request.request_id] = new_blocks
            num_scheduled_tokens[request.request_id] = num_new_tokens
            token_budget -= num_new_tokens
            req_index += 1

        # 步骤 2: 调度 WAITING 请求
        while self.waiting and token_budget > 0:
            request = self.waiting.peek_request()

            # 获取前缀缓存命中
            if request.num_computed_tokens == 0:
                new_computed_blocks, num_computed_tokens = (
                    self.kv_cache_manager.get_computed_blocks(request)
                )
            else:
                new_computed_blocks = self.kv_cache_manager.create_empty_block_list()
                num_computed_tokens = request.num_computed_tokens

            # 计算需要调度的 token 数量
            num_new_tokens = min(
                request.num_tokens - num_computed_tokens,
                token_budget
            )

            # 分配 KV Cache 槽位
            new_blocks = self.kv_cache_manager.allocate_slots(
                request,
                num_new_tokens,
                num_computed_tokens,
                new_computed_blocks,
                ...
            )

            if new_blocks is None:
                break  # 无法分配，停止调度

            # 成功调度
            request = self.waiting.pop_request()
            self.running.append(request)
            scheduled_new_reqs.append(request)
            request.num_computed_tokens = num_computed_tokens
            token_budget -= num_new_tokens

        return SchedulerOutput(...)
```

### 5.2 KV Cache 分配

**文件位置：** `vllm/v1/core/kv_cache_manager.py:203-319`

```python
class KVCacheManager:
    def allocate_slots(
        self,
        request: Request,
        num_new_tokens: int,
        num_new_computed_tokens: int = 0,
        new_computed_blocks: KVCacheBlocks | None = None,
        num_lookahead_tokens: int = 0,
        delay_cache_blocks: bool = False,
        num_encoder_tokens: int = 0,
    ) -> KVCacheBlocks | None:
        """
        为请求分配 KV Cache 槽位

        块布局：
        -----------------------------------------------------------------------
        | < computed > | < new computed > |    < new >    | < pre-allocated > |
        -----------------------------------------------------------------------
        |                  < required >                   |
        --------------------------------------------------

        Args:
            request: 请求对象
            num_new_tokens: 需要分配的新 token 数量（包括外部 token）
            num_new_computed_tokens: 前缀缓存命中的 token 数量
            new_computed_blocks: 缓存命中的块
            num_lookahead_tokens: EAGLE 推测解码的 lookahead token 数量
            delay_cache_blocks: 是否延迟缓存（用于 P/D）
            num_encoder_tokens: encoder-decoder 模型的 encoder token 数量

        Returns:
            新分配的块，如果无法分配则返回 None
        """

        # 1. 释放滑动窗口外的块
        self.coordinator.remove_skipped_blocks(
            request.request_id, request.num_computed_tokens
        )

        # 2. 计算需要分配的块数量
        num_computed_tokens = request.num_computed_tokens + num_new_computed_tokens
        num_tokens_need_slot = min(
            num_computed_tokens + num_new_tokens + num_lookahead_tokens,
            self.max_model_len,
        )

        num_blocks_to_allocate = self.coordinator.get_num_blocks_to_allocate(
            request_id=request.request_id,
            num_tokens=num_tokens_need_slot,
            new_computed_blocks=new_computed_block_list,
            num_encoder_tokens=num_encoder_tokens,
        )

        # 3. 检查是否有足够的空闲块
        if num_blocks_to_allocate > self.block_pool.get_num_free_blocks():
            return None  # 无法分配

        # 4. Touch 缓存命中的块（增加引用计数，避免驱逐）
        if self.enable_caching:
            self.block_pool.touch(new_computed_block_list)

        # 5. 保存新的计算块
        self.coordinator.save_new_computed_blocks(
            request.request_id, new_computed_block_list
        )

        # 6. 分配新块
        new_blocks = self.coordinator.allocate_new_blocks(
            request.request_id, num_tokens_need_slot, num_encoder_tokens
        )

        # 7. 缓存满块（如果启用）
        if not delay_cache_blocks and self.enable_caching:
            num_tokens_to_cache = min(
                num_computed_tokens + num_new_tokens, request.num_tokens
            )
            self.coordinator.cache_blocks(request, num_tokens_to_cache)

        return KVCacheBlocks(new_blocks)
```

### 5.3 块分配详细流程

```python
# SingleTypeKVCacheManager (在 coordinator 内部)
def allocate_new_blocks(self, request_id: str, num_tokens: int) -> list[KVCacheBlock]:
    """为请求分配新块"""

    # 1. 获取已有的块
    blocks = self.req_to_blocks.get(request_id, [])
    num_required_blocks = cdiv(num_tokens, self.block_size)
    num_current_blocks = len(blocks)
    num_new_blocks = num_required_blocks - num_current_blocks

    # 2. 从 BlockPool 获取新块
    new_blocks = self.block_pool.get_new_blocks(num_new_blocks)

    # 3. 更新请求的块列表
    blocks.extend(new_blocks)
    self.req_to_blocks[request_id] = blocks

    return new_blocks
```

---

## 6. KV Cache 复用机制(Prefix Caching)

### 6.1 前缀缓存概述

前缀缓存是 vLLM 的重要优化，通过哈希机制识别和复用相同的 prompt 前缀，避免重复计算。

**核心原理：**
1. 将 prompt 的 token_ids 按块大小（如 16）分块
2. 为每个满块计算哈希值（基于 parent_hash 和当前块的 token_ids）
3. 在 BlockPool 中查找相同哈希的块
4. 如果命中，复用已计算的 KV Cache

**哈希计算：**

**文件位置：** `vllm/v1/core/kv_cache_utils.py:494-521`

```python
def hash_block_tokens(
    hash_function: Callable[[Any], bytes],
    parent_block_hash: BlockHash | None,
    curr_block_token_ids: Sequence[int],
    extra_keys: tuple[Any, ...] | None = None,
) -> BlockHash:
    """
    计算块哈希

    Args:
        hash_function: 哈希函数（如 sha256_cbor）
        parent_block_hash: 父块的哈希（链式哈希，确保顺序）
        curr_block_token_ids: 当前块的 token IDs
        extra_keys: 额外的键（如 LoRA ID、多模态特征等）

    Returns:
        块哈希
    """
    if not parent_block_hash:
        parent_block_hash = NONE_HASH  # 第一个块的父哈希

    curr_block_token_ids_tuple = tuple(curr_block_token_ids)
    return BlockHash(
        hash_function((parent_block_hash, curr_block_token_ids_tuple, extra_keys))
    )
```

**额外键（Extra Keys）：**

为了确保缓存正确性，某些请求需要额外的哈希键：

```python
def generate_block_hash_extra_keys(
    request: Request, start_token_idx: int, end_token_idx: int, start_mm_idx: int
) -> tuple[tuple[Any, ...] | None, int]:
    """
    生成额外的哈希键

    额外键包括：
    1. LoRA ID：不同 LoRA 的 KV Cache 不能共享
    2. 多模态特征哈希：包含图像/视频的请求
    3. Cache Salt：用户指定的缓存盐值
    """
    lora_extra_keys = _gen_lora_extra_hash_keys(request)
    mm_extra_keys, new_start_mm_idx = _gen_mm_extra_hash_keys(
        request, start_token_idx, end_token_idx, start_mm_idx
    )
    cache_salt_keys = (
        [request.cache_salt] if (start_token_idx == 0 and request.cache_salt) else []
    )

    extra_keys = lora_extra_keys + mm_extra_keys + cache_salt_keys

    if not extra_keys:
        return None, new_start_mm_idx

    return tuple(extra_keys), new_start_mm_idx
```

### 6.2 前缀缓存查找

**文件位置：** `vllm/v1/core/kv_cache_manager.py:155-201`

```python
class KVCacheManager:
    def get_computed_blocks(self, request: Request) -> tuple[KVCacheBlocks, int]:
        """
        获取请求的缓存块

        Returns:
            (缓存的块, 缓存命中的 token 数量)
        """
        # 1. 检查是否启用缓存
        if not self.enable_caching:
            return self.create_empty_block_list(), 0

        # 2. 检查是否需要 prompt logprobs（禁用缓存）
        if request.sampling_params and request.sampling_params.prompt_logprobs:
            return self.create_empty_block_list(), 0

        # 3. 最多缓存到倒数第二个 token（最后一个 token 需要生成 logits）
        max_cache_hit_length = request.num_tokens - 1

        # 4. 查找最长缓存命中
        computed_blocks, num_new_computed_tokens = (
            self.coordinator.find_longest_cache_hit(
                request.block_hashes, max_cache_hit_length
            )
        )

        # 5. 记录统计
        if self.log_stats:
            if request.num_preemptions > 0:
                self.prefix_cache_stats.preempted_requests += 1
                self.prefix_cache_stats.preempted_hits += num_new_computed_tokens
            else:
                self.prefix_cache_stats.requests += 1
                self.prefix_cache_stats.hits += num_new_computed_tokens

        return KVCacheBlocks(computed_blocks), num_new_computed_tokens
```

### 6.3 最长缓存命中查找

**文件位置：** `vllm/v1/core/kv_cache_coordinator.py:270-284`

```python
class UnitaryKVCacheCoordinator(KVCacheCoordinator):
    def find_longest_cache_hit(
        self,
        block_hashes: list[BlockHash],
        max_cache_hit_length: int,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        """
        查找最长的缓存命中（单一类型）

        Args:
            block_hashes: 请求的块哈希列表
            max_cache_hit_length: 最大缓存命中长度（token 数量）

        Returns:
            (命中的块, 命中的 token 数量)
        """
        # 调用 SingleTypeKVCacheManager 的方法
        hit_blocks = self.single_type_managers[0].find_longest_cache_hit(
            block_hashes=block_hashes,
            max_length=max_cache_hit_length,
            kv_cache_group_ids=[0],
            block_pool=self.block_pool,
            kv_cache_spec=self.kv_cache_spec,
            use_eagle=self.use_eagle,
            dcp_world_size=self.dcp_world_size,
        )
        return hit_blocks, len(hit_blocks[0]) * self.block_size
```

```python
# SingleTypeKVCacheManager.find_longest_cache_hit() 的实现
@staticmethod
def find_longest_cache_hit(
    block_hashes: list[BlockHash],
    max_length: int,
    kv_cache_group_ids: list[int],
    block_pool: BlockPool,
    kv_cache_spec: KVCacheSpec,
    use_eagle: bool,
    dcp_world_size: int = 1,
) -> tuple[list[KVCacheBlock], ...]:
    """查找最长的缓存命中"""

    block_size = kv_cache_spec.block_size * dcp_world_size
    max_num_blocks = cdiv(max_length, block_size)

    # 逐块查找缓存
    hit_blocks: tuple[list[KVCacheBlock], ...] = tuple(
        [] for _ in range(len(kv_cache_group_ids))
    )
    for block_idx in range(min(max_num_blocks, len(block_hashes))):
        block_hash = block_hashes[block_idx]

        # 从 BlockPool 查找缓存块
        cached_blocks = block_pool.get_cached_block(
            block_hash, kv_cache_group_ids
        )

        if cached_blocks is None:
            # 缓存未命中，停止查找
            break

        # 添加到命中列表
        for i, cached_block in enumerate(cached_blocks):
            hit_blocks[i].append(cached_block)

    return hit_blocks
```

### 6.4 缓存块

**文件位置：** `vllm/v1/core/block_pool.py:196-265`

```python
class BlockPool:
    def cache_full_blocks(
        self,
        request: Request,
        blocks: list[KVCacheBlock],
        num_cached_blocks: int,
        num_full_blocks: int,
        block_size: int,
        kv_cache_group_id: int,
    ) -> None:
        """
        缓存满块

        Args:
            request: 请求对象
            blocks: 请求的所有块
            num_cached_blocks: 已缓存的块数量
            num_full_blocks: 满块的数量
            block_size: 块大小
            kv_cache_group_id: KV Cache 组 ID
        """
        if num_cached_blocks >= num_full_blocks:
            return  # 已全部缓存

        # 获取新的满块
        new_full_blocks = blocks[num_cached_blocks:num_full_blocks]
        new_block_hashes = request.block_hashes[num_cached_blocks:]

        # 为每个块设置哈希并加入缓存
        for i, blk in enumerate(new_full_blocks):
            block_hash = new_block_hashes[i]

            # 设置块哈希
            block_hash_with_group_id = make_block_hash_with_group_id(
                block_hash, kv_cache_group_id
            )
            blk.block_hash = block_hash_with_group_id

            # 加入缓存映射
            self.cached_block_hash_to_block.insert(block_hash_with_group_id, blk)
```

### 6.5 前缀缓存示例

```python
# 示例：两个请求具有相同的前缀

# Request 1: "Hello, my name is Alice. I am a software engineer."
# Tokens: [1234, 5678, 9012, 3456, 7890, ...]
# Block 0 (tokens 0-15): hash_0 = hash(NONE_HASH, [1234, ..., 9012])
# Block 1 (tokens 16-31): hash_1 = hash(hash_0, [3456, ..., 7890])

# Request 2: "Hello, my name is Bob. I am a data scientist."
# Tokens: [1234, 5678, 9012, 1111, 2222, ...]
# Block 0 (tokens 0-15): hash_0 = hash(NONE_HASH, [1234, ..., 9012])  # 相同!
#   -> 缓存命中，复用 Request 1 的 Block 0
# Block 1 (tokens 16-31): hash_1' = hash(hash_0, [1111, ..., 2222])  # 不同
#   -> 缓存未命中，需要计算

# 结果：Request 2 节省了 Block 0 的计算（16 tokens）
```

---

## 7. LMCache 对接

### 7.1 LMCache 概述

LMCache 是一个外部 KV Cache 存储和共享系统，允许多个 vLLM 实例之间共享 KV Cache。

**核心功能：**
- **远程存储**：KV Cache 存储在独立的服务器上
- **跨实例共享**：多个 vLLM 实例可以读取和写入同一个缓存
- **序列化/反序列化**：支持多种序列化方式（naive、safetensors 等）
- **分块传输**：按 chunk（如 256 tokens）进行传输

### 7.2 配置 LMCache

**文件位置：** `examples/others/lmcache/kv_cache_sharing_lmcache_v1.py`

```python
import os
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig

# LMCache 环境变量配置
os.environ["LMCACHE_USE_EXPERIMENTAL"] = "True"
os.environ["LMCACHE_CHUNK_SIZE"] = "256"  # 每个 chunk 的 token 数量
os.environ["LMCACHE_LOCAL_CPU"] = "False"  # 禁用本地 CPU 缓存
os.environ["LMCACHE_REMOTE_URL"] = "lm://localhost:8100"  # LMCache 服务器地址
os.environ["LMCACHE_REMOTE_SERDE"] = "naive"  # 序列化方式

# 创建 LLM 实例（发送端）
ktc_store = KVTransferConfig(
    kv_connector="LMCacheConnectorV1",
    kv_role="kv_both"  # 既发送又接收
)
llm_store = LLM(
    model="mistralai/Mistral-7B-Instruct-v0.2",
    kv_transfer_config=ktc_store,
    enforce_eager=True,
)

# 生成并存储 KV Cache
outputs_store = llm_store.generate(prompts, sampling_params)

# 创建 LLM 实例（接收端）
ktc_retrieve = KVTransferConfig(
    kv_connector="LMCacheConnectorV1",
    kv_role="kv_both"
)
llm_retrieve = LLM(
    model="mistralai/Mistral-7B-Instruct-v0.2",
    kv_transfer_config=ktc_retrieve,
    enforce_eager=True,
)

# 复用 KV Cache
outputs_retrieve = llm_retrieve.generate(prompts, sampling_params)
```

### 7.3 KV Connector 架构

**文件位置：** `vllm/v1/core/sched/scheduler.py:82-93`

```python
class Scheduler:
    def __init__(self, ...):
        # 创建 KV Connector
        self.connector = None
        if self.vllm_config.kv_transfer_config is not None:
            self.connector = KVConnectorFactory.create_connector(
                config=self.vllm_config,
                role=KVConnectorRole.SCHEDULER
            )
```

**KV Connector 核心方法：**

```python
class KVConnectorBase_V1(ABC):
    @abstractmethod
    def get_num_new_matched_tokens(
        self, request: Request, num_local_computed_tokens: int
    ) -> tuple[int | None, bool]:
        """
        查询远程缓存的匹配 token 数量

        Returns:
            (匹配的 token 数量, 是否异步加载)
        """
        pass

    @abstractmethod
    def update_state_after_alloc(
        self, request: Request, blocks: KVCacheBlocks, num_external_computed_tokens: int
    ):
        """分配块后更新状态"""
        pass

    @abstractmethod
    def build_connector_meta(self, scheduler_output: SchedulerOutput):
        """构建 KV 传输元数据"""
        pass

    @abstractmethod
    def request_finished(
        self, request: Request, block_ids: list[int]
    ) -> tuple[bool, dict[str, Any] | None]:
        """请求完成，准备发送 KV Cache"""
        pass
```

### 7.4 KV Transfer 流程

#### 7.4.1 KV Cache 发送（Store）

```python
# 步骤 1: Scheduler 调度请求
scheduler_output = scheduler.schedule()

# 步骤 2: Worker 执行模型推理，生成 KV Cache
model_runner_output = worker.execute_model(scheduler_output)

# 步骤 3: 请求完成，Scheduler 调用 connector.request_finished()
delay_free_blocks, kv_transfer_params = self.connector.request_finished(
    request, block_ids
)

# 步骤 4: Worker 将 KV Cache 发送到 LMCache 服务器
# (在后台异步进行)

# 步骤 5: 发送完成后，释放块
if kv_connector_output.finished_sending:
    for req_id in kv_connector_output.finished_sending:
        self._free_blocks(self.requests[req_id])
```

#### 7.4.2 KV Cache 接收（Retrieve）

```python
# 步骤 1: Scheduler 查询远程缓存
num_external_computed_tokens, load_kv_async = (
    self.connector.get_num_new_matched_tokens(
        request, num_new_local_computed_tokens
    )
)

# 步骤 2: 分配块（用于接收 KV Cache）
new_blocks = self.kv_cache_manager.allocate_slots(
    request,
    num_new_tokens + num_external_computed_tokens,
    num_new_local_computed_tokens,
    new_computed_blocks,
    delay_cache_blocks=load_kv_async,  # 延迟缓存
)

# 步骤 3: 更新状态
self.connector.update_state_after_alloc(
    request,
    new_computed_blocks + new_blocks,
    num_external_computed_tokens,
)

# 步骤 4: 如果异步加载，将请求设为 WAITING_FOR_REMOTE_KVS
if load_kv_async:
    request.status = RequestStatus.WAITING_FOR_REMOTE_KVS

# 步骤 5: Worker 从 LMCache 服务器接收 KV Cache
# (在后台异步进行)

# 步骤 6: 接收完成后，更新状态
if kv_connector_output.finished_recving:
    for req_id in kv_connector_output.finished_recving:
        self.finished_recving_kv_req_ids.add(req_id)

# 步骤 7: 下一次调度时，缓存接收到的块
if request.request_id in self.finished_recving_kv_req_ids:
    self.kv_cache_manager.cache_blocks(request, num_computed_tokens)
    request.num_computed_tokens = num_computed_tokens
    request.status = RequestStatus.WAITING
```

### 7.5 LMCache 工作流程

```
┌─────────────────────────────────────────────────────────────┐
│                      vLLM Instance 1                         │
│  (Store Mode: 发送 KV Cache)                                │
│                                                               │
│  1. Generate with prompt                                     │
│  2. Compute KV Cache                                         │
│  3. Send KV Cache to LMCache Server                          │
│     (按 chunk 发送，每个 chunk 256 tokens)                   │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    LMCache Server                            │
│  (Remote KV Cache Storage)                                   │
│                                                               │
│  - 存储 KV Cache (key: block_hash, value: KV tensors)       │
│  - 支持多种后端 (Redis, S3, 本地文件等)                     │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                      vLLM Instance 2                         │
│  (Retrieve Mode: 接收 KV Cache)                             │
│                                                               │
│  1. Generate with same prompt                                │
│  2. Query LMCache Server for matched tokens                  │
│  3. Allocate blocks for receiving                            │
│  4. Receive KV Cache from LMCache Server                     │
│  5. Cache received blocks                                    │
│  6. Continue generation (reusing KV Cache)                   │
└─────────────────────────────────────────────────────────────┘
```

### 7.6 LMCache 优势

1. **跨实例共享**：多个 vLLM 实例可以共享相同的 KV Cache
2. **降低延迟**：新实例可以快速复用已计算的 KV Cache
3. **节省计算**：避免重复计算相同的 prompt
4. **灵活扩展**：支持多种存储后端（Redis、S3 等）

---

## 8. 关键类和方法

### 8.1 LLM 类

**文件位置：** `vllm/entrypoints/llm.py:91-1721`

| 方法 | 说明 | 文件行号 |
|------|------|----------|
| `__init__` | 初始化 LLM，创建 LLMEngine | 188-328 |
| `generate` | 生成文本 | 366-432 |
| `reset_prefix_cache` | 重置前缀缓存 | 1467 |

### 8.2 LLMEngine 类

**文件位置：** `vllm/v1/engine/llm_engine.py:46-410`

| 方法 | 说明 | 文件行号 |
|------|------|----------|
| `__init__` | 初始化 Engine | 49-150 |
| `from_engine_args` | 从 EngineArgs 创建 | 169-195 |
| `step` | 执行一步推理 | 288-319 |
| `add_request` | 添加请求 | 227-286 |

### 8.3 Scheduler 类

**文件位置：** `vllm/v1/core/sched/scheduler.py:40-1511`

| 方法 | 说明 | 文件行号 |
|------|------|----------|
| `__init__` | 初始化 Scheduler 和 KVCacheManager | 41-169 |
| `schedule` | 调度请求，分配 KV Cache | 172-663 |
| `update_from_output` | 处理模型输出，更新请求状态 | 909-1089 |
| `add_request` | 添加新请求到等待队列 | 1164-1168 |
| `finish_requests` | 完成/中止请求 | 1170-1212 |

### 8.4 KVCacheManager 类

**文件位置：** `vllm/v1/core/kv_cache_manager.py:81-405`

| 方法 | 说明 | 文件行号 |
|------|------|----------|
| `__init__` | 初始化 Manager 和 Coordinator | 82-133 |
| `get_computed_blocks` | 获取前缀缓存命中 | 155-201 |
| `allocate_slots` | 分配 KV Cache 槽位 | 203-319 |
| `free` | 释放请求的所有块 | 321-329 |
| `cache_blocks` | 缓存满块 | 397-400 |
| `reset_prefix_cache` | 重置前缀缓存 | 331-345 |

### 8.5 BlockPool 类

**文件位置：** `vllm/v1/core/block_pool.py:125-426`

| 方法 | 说明 | 文件行号 |
|------|------|----------|
| `__init__` | 初始化 BlockPool | 139-168 |
| `get_cached_block` | 查找缓存块 | 169-194 |
| `cache_full_blocks` | 缓存满块 | 196-265 |
| `get_new_blocks` | 分配新块 | 267-293 |
| `touch` | 增加块的引用计数 | 331-345 |
| `free_blocks` | 释放块 | 347-361 |
| `reset_prefix_cache` | 重置前缀缓存 | 363-393 |

### 8.6 KVCacheCoordinator 类

**文件位置：** `vllm/v1/core/kv_cache_coordinator.py:16-479`

| 方法 | 说明 | 文件行号 |
|------|------|----------|
| `__init__` | 初始化 Coordinator 和 BlockPool | 21-48 |
| `get_num_blocks_to_allocate` | 计算需要分配的块数量 | 50-84 |
| `allocate_new_blocks` | 分配新块 | 100-125 |
| `find_longest_cache_hit` | 查找最长缓存命中（抽象方法） | 188-194 |
| `cache_blocks` | 缓存块 | 127-138 |
| `free` | 释放块 | 140-148 |

---

## 9. 总结

### 9.1 核心特性

1. **高效的块管理**
   - 以固定大小的块（如 16 tokens）为单位管理 KV Cache
   - 使用 LRU 策略进行驱逐
   - 支持块的引用计数，实现前缀缓存

2. **智能的前缀缓存**
   - 通过链式哈希识别相同前缀
   - 自动复用已计算的 KV Cache
   - 支持多模态、LoRA 等复杂场景

3. **灵活的调度策略**
   - 支持 FCFS 和优先级调度
   - 动态抢占和恢复
   - 考虑 token budget 和 GPU 内存限制

4. **外部缓存对接**
   - 与 LMCache 无缝集成
   - 支持跨实例 KV Cache 共享
   - 异步传输，最小化性能影响

### 9.2 关键流程总结

#### 初始化流程
```
LLM.__init__()
  -> LLMEngine.from_engine_args()
    -> EngineCore.__init__()
      -> Scheduler.__init__()
        -> KVCacheManager.__init__()
          -> KVCacheCoordinator
            -> BlockPool.__init__()
```

#### 推理流程
```
LLM.generate()
  -> LLMEngine.step() [循环]
    -> Scheduler.schedule()
      -> KVCacheManager.get_computed_blocks() [前缀缓存]
      -> KVCacheManager.allocate_slots() [分配槽位]
    -> Worker.execute_model()
    -> Scheduler.update_from_output()
      -> KVCacheManager.cache_blocks() [缓存满块]
      -> KVCacheManager.free() [释放完成的请求]
```

#### 前缀缓存流程
```
新请求到达
  -> Scheduler.schedule()
    -> KVCacheManager.get_computed_blocks()
      -> Coordinator.find_longest_cache_hit()
        -> BlockPool.get_cached_block() [查找缓存]
          -> 返回命中的块
    -> 复用命中的块，只计算新 token
```

### 9.3 性能优化建议

1. **启用前缀缓存**
   ```python
   llm = LLM(
       model="...",
       enable_prefix_caching=True,  # 启用前缀缓存
   )
   ```

2. **合理设置 GPU 内存占用**
   ```python
   llm = LLM(
       model="...",
       gpu_memory_utilization=0.9,  # 根据实际情况调整
   )
   ```

3. **使用 LMCache 共享 KV Cache**
   ```python
   llm = LLM(
       model="...",
       kv_transfer_config=KVTransferConfig(
           kv_connector="LMCacheConnectorV1",
           kv_role="kv_both",
       ),
   )
   ```

4. **调整块大小**
   ```python
   llm = LLM(
       model="...",
       block_size=16,  # 较小的块大小提高缓存命中率，但增加管理开销
   )
   ```

### 9.4 参考文献

- [vLLM 官方文档](https://docs.vllm.ai/)
- [vLLM GitHub 仓库](https://github.com/vllm-project/vllm)
- [LMCache GitHub 仓库](https://github.com/LMCache/LMCache)
- [PagedAttention 论文](https://arxiv.org/abs/2309.06180)

---

**文档版本：** 1.0
**创建日期：** 2025-01-XX
**作者：** Claude Code
**适用 vLLM 版本：** v1 (最新版本)
