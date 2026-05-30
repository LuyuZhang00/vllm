# vLLM KV Cache 管理机制详解

> 本文档深入剖析 vLLM 的 KV Cache 管理机制，涵盖初始化流程、架构设计、调度管理、前缀缓存、LMCache 集成等核心内容。

---

## 目录

- [1. 概述](#1-概述)
- [2. 入口示例](#2-入口示例)
- [3. KV Cache 初始化流程](#3-kv-cache-初始化流程)
- [4. KV Cache 架构设计](#4-kv-cache-架构设计)
- [5. KV Cache 管理和调度](#5-kv-cache-管理和调度)
- [6. KV Cache 复用机制 (Prefix Caching)](#6-kv-cache-复用机制-prefix-caching)
- [7. LMCache 对接](#7-lmcache-对接)
- [8. 关键类和方法](#8-关键类和方法)
- [9. 总结](#9-总结)

---

## 1. 概述

### 1.1 什么是 KV Cache

在 Transformer 模型的自回归推理过程中，每生成一个新 token，都需要对之前所有 token 进行注意力计算。KV Cache 缓存了每一层注意力的 Key 和 Value 张量，避免重复计算，是 LLM 推理性能的关键。

### 1.2 vLLM 的核心创新：PagedAttention

传统 LLM 推理框架为每个请求预分配连续的 KV Cache 内存，导致：

- **内存碎片化**：不同请求长度不同，分配和释放产生碎片
- **内存浪费**：必须按最大长度预分配，短请求浪费大量空间
- **无法动态调整**：请求长度变化时无法灵活扩展

vLLM 借鉴操作系统虚拟内存的 **分页机制**，将 KV Cache 分成固定大小的 **块 (Block)**：

- 每个块存储固定数量 token 的 KV 数据
- 通过 **块表 (Block Table)** 将逻辑位置映射到物理块
- 请求可以使用 **非连续** 的物理块
- 块可以 **按需分配和释放**，支持跨请求共享

### 1.3 KV Cache 管理的整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                      Scheduler (调度器)                       │
│  决定每步执行哪些请求，分配/释放 KV 块                         │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                   KVCacheManager (KV 缓存管理器)              │
│  顶层入口，协调调度器与底层块管理                              │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              KVCacheCoordinator (KV 缓存协调器)               │
│  管理多个 KV 缓存组 (全注意力、滑动窗口、Mamba 等)              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │ UnitaryCoord.   │  │ HybridCoord.    │  │ NoPrefixCoord│ │
│  │ (单缓存组)       │  │ (混合缓存组)     │  │ (无前缀缓存)  │ │
│  └────────┬────────┘  └────────┬────────┘  └──────────────┘ │
└───────────┼────────────────────┼────────────────────────────┘
            │                    │
            ▼                    ▼
┌─────────────────────────────────────────────────────────────┐
│          SingleTypeKVCacheManager (单类型缓存管理器)           │
│  每种注意力类型一个实例，处理特定的缓存逻辑                     │
│  ┌──────────────┐ ┌──────────────┐ ┌───────────────────────┐│
│  │FullAttention │ │SlidingWindow │ │ChunkedLocalAttention  ││
│  │Manager       │ │Manager       │ │Manager                ││
│  └──────────────┘ └──────────────┘ └───────────────────────┘│
│  ┌──────────────┐ ┌──────────────┐                          │
│  │MambaManager  │ │CrossAttention│                          │
│  │              │ │Manager       │                          │
│  └──────────────┘ └──────────────┘                          │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    BlockPool (块池)                           │
│  管理所有 GPU 块的分配、释放、前缀缓存                         │
│  ┌───────────────────┐  ┌──────────────────────────────┐    │
│  │FreeKVCacheBlock   │  │BlockHashToBlockMap           │    │
│  │Queue (空闲队列)    │  │(前缀缓存哈希表)              │    │
│  └───────────────────┘  └──────────────────────────────┘    │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│               KVCacheBlock (KV 缓存块)                       │
│  最小分配单元：block_id, ref_cnt, block_hash                 │
└─────────────────────────────────────────────────────────────┘
```

### 1.4 关键设计原则

| 原则 | 说明 |
|------|------|
| **分页管理** | KV Cache 分成固定大小块，通过块表间接映射 |
| **引用计数** | 跟踪每个块被多少请求引用，支持安全共享 |
| **LRU 驱逐** | 空闲队列维护 LRU 顺序，最近最少使用的块优先回收 |
| **前缀缓存** | 内容可寻址、链式依赖的块哈希，跨请求复用公共前缀 |
| **分层卸载** | GPU → CPU → 磁盘的多级缓存层次 |
| **统一接口** | KVConnector 抽象统一所有 KV 传输/卸载机制 |

---

## 2. 入口示例

### 2.1 离线推理中的 KV Cache

```python
from vllm import LLM, SamplingParams

# 初始化时自动配置 KV Cache
llm = LLM(
    model="meta-llama/Llama-3-8B",
    gpu_memory_utilization=0.9,      # GPU 内存使用率
    max_model_len=8192,               # 最大模型长度
    kv_cache_dtype="auto",            # KV Cache 数据类型
    # enable_prefix_caching=True,     # 启用前缀缓存 (默认 True)
)

# 推理时 KV Cache 自动管理
outputs = llm.generate(
    ["Hello, world!", "How are you?"],
    SamplingParams(temperature=0.7, max_tokens=100)
)
# - 每个请求的 prompt tokens 自动分配 KV 块
# - 生成过程中按需分配新块
# - 请求完成后自动释放块
# - 相同前缀的请求自动共享缓存块
```

### 2.2 API 服务中的 KV Cache

```bash
# 启动服务时配置 KV Cache
vllm serve meta-llama/Llama-3-8B \
    --gpu-memory-utilization 0.9 \
    --max-model-len 8192 \
    --kv-cache-dtype auto \
    --enable-prefix-caching \
    --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
```

### 2.3 KV Cache 相关配置项

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `gpu_memory_utilization` | float | 0.9 | GPU 内存使用率上限 |
| `max_model_len` | int | 自动 | 最大序列长度 |
| `kv_cache_dtype` | str | "auto" | KV Cache 数据类型 (auto/fp8/e5m2/e4m3) |
| `enable_prefix_caching` | bool | True | 启用前缀缓存 |
| `enable_chunked_prefill` | bool | True | 启用分块预填充 |
| `block_size` | int | 自动 | 块大小 (token 数) |
| `num_gpu_blocks_override` | int | None | 覆盖自动计算的 GPU 块数 |
| `kv_transfer_config` | dict | None | KV 传输配置 (P/D 分离、LMCache) |
| `kv_connector_extra_config` | dict | {} | KV 连接器额外配置 |

### 2.4 KV Cache 状态查看

```python
# 查看 KV Cache 使用情况
llm = LLM(model="meta-llama/Llama-3-8B")
engine = llm.llm_engine

# 通过 EngineCore 访问调度器
scheduler = engine.engine_core.scheduler
kv_cache_manager = scheduler.kv_cache_manager

# 查看使用率
print(f"KV Cache 使用率: {kv_cache_manager.usage:.2%}")

# 查看块数
print(f"总块数: {kv_cache_manager.block_pool.num_gpu_blocks}")
print(f"空闲块数: {kv_cache_manager.block_pool.get_num_free_blocks()}")
```

---

## 3. KV Cache 初始化流程

### 3.1 初始化总览

KV Cache 的初始化在 `EngineCore.__init__()` 中完成，位于 Executor 创建之后、Scheduler 创建之前：

```
EngineCore.__init__()
    │
    ├── 1. 创建 Executor (模型执行器)
    │
    ├── 2. _initialize_kv_caches()  ← KV Cache 初始化入口
    │      │
    │      ├── 2.1 获取 KV Cache 规格 (每层的形状、dtype)
    │      ├── 2.2 内存分析 (确定可用 GPU 内存)
    │      ├── 2.3 计算 KV Cache 配置 (块数、张量布局、缓存组)
    │      ├── 2.4 生成调度器 KV Cache 配置
    │      └── 2.5 在 Worker 上分配 KV Cache 张量
    │
    └── 3. 创建 Scheduler
```

### 3.2 步骤一：获取 KV Cache 规格

```python
# EngineCore._initialize_kv_caches() (core.py, line 239)
kv_cache_specs = self.model_executor.get_kv_cache_specs()
```

每个 Worker 调用 `GPUModelRunner.get_kv_cache_spec()`，遍历所有注意力层：

```python
# GPUModelRunner.get_kv_cache_spec() (gpu_model_runner.py, line 7315)
def get_kv_cache_spec(self):
    kv_cache_spec = {}
    attn_layers = get_layers_from_vllm_config(self.vllm_config, layer_type)

    for layer_name, attn_module in attn_layers.items():
        # 跳过共享 KV Cache 的层
        if kv_tgt_layer := attn_module.kv_sharing_target_layer_name:
            self.shared_kv_cache_layers[layer_name] = kv_tgt_layer
            continue

        # 获取每层的 KV Cache 规格
        if spec := attn_module.get_kv_cache_spec(self.vllm_config):
            kv_cache_spec[layer_name] = spec

    return kv_cache_spec
```

每层返回一个 `KVCacheSpec` 子类，描述该层的缓存格式：

| 规格类型 | 说明 | 页面大小计算 |
|----------|------|-------------|
| `FullAttentionSpec` | 标准全注意力 | `2 * block_size * num_kv_heads * head_size * dtype_size` |
| `MLAAttentionSpec` | Multi-head Latent Attention (DeepSeek) | 压缩表示: `block_size * (kv_lora_rank + rope_dim) * dtype_size` |
| `SlidingWindowSpec` | 滑动窗口注意力 | 同 FullAttention，但有窗口限制 |
| `ChunkedLocalAttentionSpec` | 分块局部注意力 (Gemma) | 同 FullAttention，但有块大小限制 |
| `MambaSpec` | SSM (Mamba) 层 | 多个状态张量的总和 |
| `CrossAttentionSpec` | 编码器-解码器交叉注意力 | 基于 `max_encoder_len` |
| `EncoderOnlyAttentionSpec` | 仅编码器 | 0 (无需 KV Cache) |

### 3.3 步骤二：内存分析

```python
# EngineCore._initialize_kv_caches() (core.py, line 253)
available_gpu_memory = self.model_executor.determine_available_memory()
```

```python
# GPUWorker.determine_available_memory() (gpu_worker.py, line 361)
def determine_available_memory(self):
    # 1. 如果配置了固定 KV Cache 内存，直接返回
    if self.cache_config.kv_cache_memory_bytes:
        return self.cache_config.kv_cache_memory_bytes

    # 2. 运行内存分析
    with memory_profiling(self.init_snapshot, weights_memory=...) as profile_result:
        self.model_runner.profile_run()  # 运行 dummy forward pass

    # 3. 估算 CUDA Graph 额外内存
    cudagraph_memory = estimate_cudagraph_memory() if cudagraph_enabled else 0

    # 4. 计算可用 KV Cache 内存
    non_kv_cache_memory = (
        profile_result.non_torch_increase      # 非 PyTorch 分配
        + profile_result.torch_peak_increase    # PyTorch 峰值
        + profile_result.weights_memory         # 模型权重
    )
    available_memory = (
        requested_memory * gpu_memory_utilization
        - non_kv_cache_memory
        - cudagraph_memory
    )
    return available_memory
```

**内存分析的关键步骤：**

1. **`profile_run()`**：运行一次 dummy forward pass，触发所有延迟初始化
2. **测量峰值内存**：记录 PyTorch 峰值内存使用和非 PyTorch 分配
3. **减去模型权重**：模型权重已经占用的内存不能用于 KV Cache
4. **减去 CUDA Graph 预估**：CUDA Graph 捕获需要额外内存
5. **跨 Worker 取最小值**：确保所有 Worker 使用相同块数

### 3.4 步骤三：计算 KV Cache 配置

```python
# EngineCore._initialize_kv_caches() (core.py, line 264)
kv_cache_configs = get_kv_cache_configs(
    vllm_config, kv_cache_specs, available_gpu_memory
)
```

`get_kv_cache_configs()` 是核心配置算法：

#### 3.4.1 合并所有 Worker 的规格

```python
# kv_cache_utils.py, line 1983
merged_kv_cache_specs = {}
for kv_cache_spec_one_worker in kv_cache_specs:
    for layer_name, layer_spec in kv_cache_spec_one_worker.items():
        merged_kv_cache_specs[layer_name] = layer_spec
```

#### 3.4.2 分组缓存层

`get_kv_cache_groups()` 将兼容的层分组：

| 情况 | 分组策略 |
|------|----------|
| 所有层规格相同 | 一个组，包含所有层 |
| 所有层类型相同但大小不同 (如 MLA) | 一个 `UniformTypeKVCacheSpecs` 组 |
| 混合注意力类型 (如全注意力 + 滑动窗口) | 多个组，每种类型一个组 |
| DeepSeek V4 (MLA + SWA) | 多个 `UniformTypeKVCacheSpecs` 组 |

#### 3.4.3 计算块数和张量布局

```python
# kv_cache_utils.py, line 1236
def get_kv_cache_config_from_groups(vllm_config, kv_cache_groups, available_memory):
    # 单组情况
    if len(kv_cache_groups) == 1:
        num_blocks = available_memory // page_size_bytes
        # 每层一个 KVCacheTensor
        kv_cache_tensors = [
            KVCacheTensor(size=per_layer_page_size * num_blocks, shared_by=[layer_name])
            for layer_name in group.layer_names
        ]

    # 多组情况
    else:
        # 统一页面大小 (取最大值)
        page_size = get_uniform_page_size([g.kv_cache_spec for g in groups])
        # 计算块数
        num_blocks = available_memory // page_size // num_layers
        # 共享内存池
        kv_cache_tensors = [KVCacheTensor(size=..., shared_by=[...]) for ...]

    return KVCacheConfig(num_blocks, kv_cache_tensors, kv_cache_groups)
```

#### 3.4.4 跨 Worker 归一化

```python
# kv_cache_utils.py, line 2061
min_num_blocks = min(config.num_blocks for config in kv_cache_configs)
for config in kv_cache_configs:
    config.num_blocks = min_num_blocks
    for tensor in config.kv_cache_tensors:
        tensor.size = tensor.size // old_blocks * min_num_blocks
```

所有 Worker 使用 **最小块数**，张量大小按比例缩小，避免内存浪费。

### 3.5 步骤四：分配 KV Cache 张量

```python
# EngineCore._initialize_kv_caches() (core.py, line 286)
self.model_executor.initialize_from_config(kv_cache_configs)
```

每个 Worker 调用 `GPUModelRunner.initialize_kv_cache()`：

```python
# GPUModelRunner.initialize_kv_cache() (gpu_model_runner.py, line 7159)
def initialize_kv_cache(self, kv_cache_config):
    # 1. 深拷贝配置 (后续会修改)
    kv_cache_config = deepcopy(kv_cache_config)

    # 2. 添加编码器专用层
    self.may_add_encoder_only_layers_to_kv_cache_config()

    # 3. 添加 KV 共享层到组
    self.maybe_add_kv_sharing_layers_to_kv_cache_groups(kv_cache_config)

    # 4. 初始化注意力后端
    self.initialize_attn_backend(kv_cache_config)

    # 5. 计算内核块大小 (处理块拆分)
    kernel_block_sizes = prepare_kernel_block_sizes(kv_cache_config, self.attn_groups)

    # 6. 初始化元数据构建器
    self.initialize_metadata_builders(kv_cache_config, kernel_block_sizes)

    # 7. 分配 KV Cache 张量
    kv_caches = self.initialize_kv_cache_tensors(kv_cache_config, kernel_block_sizes)
```

#### 3.5.1 两种分配路径

**统一路径 (Uniform)**：当 KV 连接器需要跨层块时

```python
# 分配单个连续缓冲区，包含所有层
total_size = tensor_size * num_layers
cross_layers_kv_cache = torch.zeros(total_size, dtype=torch.int8, device=device)
    .view(kv_cache_dtype)
    .view(kv_cache_shape_with_layers)

# 每层获得一个视图
for i, tensor in enumerate(kv_cache_tensors):
    kv_caches[layer_name] = permuted_kv_cache[i]
```

**通用路径 (General)**：大多数情况

```python
# 为每个 KVCacheTensor 分配原始 int8 缓冲区
for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
    tensor = torch.zeros(kv_cache_tensor.size, dtype=torch.int8, device=device)
    for layer_name in kv_cache_tensor.shared_by:
        kv_cache_raw_tensors[layer_name] = tensor

# 重塑为后端期望的形状
for layer_name, raw_tensor in kv_cache_raw_tensors.items():
    if isinstance(spec, AttentionSpec):
        # 计算块数
        num_blocks = raw_tensor.numel() // page_size_bytes
        kernel_num_blocks = num_blocks * (block_size // kernel_block_size)

        # 获取后端形状: (num_blocks, 2, block_size, num_kv_heads, head_size)
        kv_cache_shape = backend.get_kv_cache_shape(kernel_num_blocks, ...)

        # 应用步幅顺序排列 (NHD 或 HND)
        stride_order = backend.get_kv_cache_stride_order()
        kv_cache = raw_tensor.view(dtype).view(kv_cache_shape).permute(*stride_order)

    elif isinstance(spec, MambaSpec):
        # Mamba 层使用多个状态张量，通过 strided views 切分
        for state_info in spec.state_info:
            tensor = torch.as_strided(raw_tensor.view(dtype), size=shape, stride=stride)
```

#### 3.5.2 块大小拆分

当调度器使用较大块大小 (如 256) 但注意力内核只支持较小块大小 (如 64) 时：

```python
# utils.py, line 329
def prepare_kernel_block_sizes(kv_cache_config, attn_groups):
    for group in kv_cache_config.kv_cache_groups:
        if isinstance(spec, AttentionSpec):
            # 256 / 64 = 4，每个调度块拆分为 4 个内核块
            kernel_block_size = select_common_block_size(block_size, backends)
        elif isinstance(spec, MambaSpec):
            kernel_block_size = spec.block_size  # 不拆分
```

#### 3.5.3 绑定到注意力层

```python
# utils.py, line 460
def bind_kv_cache(kv_caches, forward_context, runner_kv_caches, num_attn_module):
    # 1. 按层索引排序，填充 runner_kv_caches 列表
    for layer_index in sorted(index2name.keys()):
        for layer_name in index2name[layer_index]:
            runner_kv_caches.append(kv_caches[layer_name])

    # 2. 绑定到注意力模块
    for layer_name, kv_cache in kv_caches.items():
        forward_context[layer_name].kv_cache = kv_cache
```

#### 3.5.4 跨层 KV 共享

```python
# 共享层不分配独立内存，直接引用目标层的张量
for layer_name, target_layer_name in self.shared_kv_cache_layers.items():
    kv_caches[layer_name] = kv_caches[target_layer_name]
```

### 3.6 初始化流程图

```
EngineCore.__init__()
    │
    ▼
Executor.get_kv_cache_specs()
    │  每个 Worker 调用 GPUModelRunner.get_kv_cache_spec()
    │  遍历注意力层，收集 KVCacheSpec
    ▼
[dict[layer_name, KVCacheSpec]] × num_workers
    │
    ▼
Executor.determine_available_memory()
    │  每个 Worker 运行 profile_run()
    │  测量峰值内存，计算可用 KV Cache 内存
    ▼
available_memory × num_workers
    │
    ▼
get_kv_cache_configs()
    │  合并规格 → 分组 → 计算块数 → 归一化
    ▼
KVCacheConfig(num_blocks, kv_cache_tensors, kv_cache_groups)
    │
    ▼
Executor.initialize_from_config()
    │  每个 Worker:
    │  1. 分配原始 int8 缓冲区
    │  2. 重塑为注意力后端形状
    │  3. 处理块拆分和跨层共享
    │  4. 绑定到注意力模块
    ▼
KV Cache 初始化完成，创建 Scheduler
```

---

## 4. KV Cache 架构设计

### 4.1 核心数据结构

#### 4.1.1 KVCacheBlock —— 最小分配单元

```python
# vllm/v1/core/kv_cache_utils.py, line 116
@dataclass(slots=True)
class KVCacheBlock:
    block_id: int                          # 物理块 ID (0 到 num_gpu_blocks-1)
    ref_cnt: int = 0                       # 引用计数
    _block_hash: BlockHashWithGroupId | None = None  # 内容哈希
    prev_free_block: KVCacheBlock | None = None       # 空闲链表前驱
    next_free_block: KVCacheBlock | None = None       # 空闲链表后继
    is_null: bool = False                  # 是否为空块 (滑动窗口填充)
```

**引用计数语义：**
- `ref_cnt > 0`：块正在被一个或多个请求使用
- `ref_cnt == 0`：块是空闲的，可以被驱逐或重新分配
- `is_null = True`：空块占位符，`ref_cnt` 不维护

#### 4.1.2 FreeKVCacheBlockQueue —— 空闲块队列

```python
# vllm/v1/core/kv_cache_utils.py, line 164
class FreeKVCacheBlockQueue:
    """双向链表，直接操作 KVCacheBlock 的链表指针"""

    # O(1) 操作: popleft, append, remove
    # LRU 顺序: 头部 = 最久未使用 (优先驱逐)
    #            尾部 = 最近释放 (最后驱逐)
```

**为什么不使用 Python `deque`？**
`deque` 无法 O(1) 移除中间元素。`FreeKVCacheBlockQueue` 直接操作 `KVCacheBlock` 的 `prev_free_block`/`next_free_block` 指针，实现 O(1) 的任意位置移除（`touch()` 操作需要）。

**哨兵节点：**
- `fake_free_list_head`：头部哨兵，简化边界逻辑
- `fake_free_list_tail`：尾部哨兵，简化边界逻辑
- 每个真实块始终有有效的 `prev`/`next` 邻居

#### 4.1.3 BlockHashToBlockMap —— 前缀缓存哈希表

```python
# vllm/v1/core/block_pool.py, line 34
class BlockHashToBlockMap:
    """映射 BlockHashWithGroupId → KVCacheBlock(s)"""

    # 使用联合类型减少 GC 开销:
    # - 单块情况 (常见): _cache[key] = KVCacheBlock
    # - 多块情况 (哈希碰撞): _cache[key] = dict[int, KVCacheBlock]
```

#### 4.1.4 BlockPool —— 块池

```python
# vllm/v1/core/block_pool.py, line 130
class BlockPool:
    blocks: list[KVCacheBlock]                    # 所有 GPU 块
    free_block_queue: FreeKVCacheBlockQueue       # 空闲块队列
    cached_block_hash_to_block: BlockHashToBlockMap  # 前缀缓存哈希表
    null_block: KVCacheBlock                      # 空块占位符
```

#### 4.1.5 KVCacheConfig —— KV Cache 配置

```python
# vllm/v1/kv_cache_interface.py, line 838
@dataclass
class KVCacheConfig:
    num_blocks: int                              # 总块数
    kv_cache_tensors: list[KVCacheTensor]        # 张量描述
    kv_cache_groups: list[KVCacheGroupSpec]      # 缓存组

@dataclass
class KVCacheTensor:
    size: int                                    # 张量大小 (字节)
    shared_by: list[str]                         # 共享此张量的层名

@dataclass
class KVCacheGroupSpec:
    layer_names: list[str]                       # 组内的层名
    kv_cache_spec: KVCacheSpec                   # 缓存规格
    is_eagle_group: bool = False                 # 是否为 EAGLE 组
```

### 4.2 KV Cache 张量布局

#### 4.2.1 标准注意力 (FlashAttention / FlashInfer)

```
逻辑形状: [num_blocks, 2, block_size, num_kv_heads, head_size]
                │   │
                │   └── 维度 1: K/V 分割
                └── 维度 0: 块索引

NHD 布局 (默认):
  物理形状 = 逻辑形状
  内存顺序: block → K/V → token → head → dim

HND 布局 (Blackwell SM100 必需):
  物理形状: [num_blocks, 2, num_kv_heads, block_size, head_size]
  步幅顺序: (0, 1, 3, 2, 4)
  内存顺序: block → K/V → head → token → dim
```

#### 4.2.2 MLA 注意力 (DeepSeek)

```
逻辑形状: [num_blocks, block_size, head_size]
                              │
                              └── head_size = kv_lora_rank + qk_rope_head_dim
                                   例如 DeepSeek V3: 512 + 64 = 576

每 token 存储: [kv_c_normed; k_pe]
  - kv_c_normed: 压缩 KV 潜在表示 (512 维)
  - k_pe: 解耦的 RoPE 位置编码 (64 维)

DeepSeek V4 fp8: 每 token 584 字节
  = 448B NoPE + 128B RoPE + 8B fp8 scale
```

#### 4.2.3 Mamba (SSM)

```
每层存储多个状态张量，形状和 dtype 各不相同
通过 strided views 从原始缓冲区切分:

tensor = torch.as_strided(
    raw_tensor.view(dtype),
    size=target_shape,
    stride=target_stride,
    storage_offset=offset_bytes // dtype_size,
)
```

### 4.3 Block Table 与 Slot Mapping

#### 4.3.1 Block Table —— 逻辑到物理的映射

```python
# vllm/v1/worker/block_table.py
class BlockTable:
    # 2D 张量: [max_num_reqs, max_num_blocks_per_req]
    # block_table[req_idx, block_idx] = 物理块 ID
    block_table: torch.Tensor
```

#### 4.3.2 Slot Mapping —— Token 到 Cache Slot 的映射

```python
# slot_mapping[token_position] = flat_slot_index
# slot_index = block_number * block_size + block_offset
```

Slot Mapping 由 Triton 内核计算 (`_compute_slot_mapping_kernel`)：

```python
# 每个 token:
# 1. 获取 position
# 2. 计算 block_indices = position // virtual_block_size
# 3. 查找块号: block_numbers = block_table[req_idx, block_indices]
# 4. 计算 slot: slot_id = block_numbers * block_size + local_offset
```

#### 4.3.3 块大小拆分

当内核块大小 < 调度器块大小时：

```
调度器块大小: 256 tokens
内核块大小: 64 tokens
拆分因子: 256 / 64 = 4

调度器块 ID 3 → 内核块 ID [6, 7, 8, 9]
  kernel_block_ids = kv_manager_block_ids * blocks_per_kv_block + arange(blocks_per_kv_block)
```

### 4.4 注意力后端与 KV Cache 的交互

#### 4.4.1 KV Cache 写入 (do_kv_cache_update)

在 forward pass 之前，新的 K/V 值被写入缓存：

```python
# FlashAttention (flash_attn.py, line 850)
def do_kv_cache_update(self, key, value, kv_cache, slot_mapping):
    key_cache, value_cache = kv_cache.unbind(1)  # 拆分 K/V
    reshape_and_cache_flash(key, value, key_cache, value_cache, slot_mapping)

# MLA (backend.py, line 917)
def do_kv_cache_update(self, kv_c_normed, k_pe, kv_cache, slot_mapping):
    ops.concat_and_cache_mla(kv_c_normed, k_pe, kv_cache, slot_mapping)
```

`reshape_and_cache_flash` Triton 内核：

```python
# 每个 program 处理一个 token:
# 1. 从 slot_mapping 加载 slot_idx
# 2. 计算 block_idx = slot_idx // block_size
# 3. 计算 block_offset = slot_idx % block_size
# 4. 将 K/V 值写入 cache[block_idx, k/v, block_offset, head, dim]
# 支持 FP8 量化写入 (除以 scale 后存储)
```

#### 4.4.2 KV Cache 读取 (Attention Forward)

```python
# FlashAttention (flash_attn.py, line 796)
flash_attn_varlen_func(
    q=query,
    k=key_cache,          # 完整 KV Cache 张量
    v=value_cache,
    cu_seqlens_q=cu_seqlens_q,
    seqused_k=seqused_k,  # 每请求的 KV token 数
    block_table=block_table,  # 块表，用于间接寻址
    ...
)
# FlashAttention 内核通过 block_table 间接寻址:
# 对于请求 r 的 KV 位置 p:
#   读取 cache[block_table[r, p // block_size], p % block_size, ...]
```

#### 4.4.3 MLA 的两种读取路径

**MHA 路径 (prefill)**：计算友好，重建完整 K/V

```
kv_nope = kv_b_proj(kv_c_normed)    # 投影到完整表示
k_nope, v = split(kv_nope)
k = concat(k_nope, k_pe)            # 拼接 RoPE
attn_out = flash_attn(q, k, v)
```

**MQA 路径 (decode)**：数据移动友好，直接使用压缩缓存

```
ql_nope = einsum("snh,lnh->snl", q_nope, W_UK)  # 查询投影到潜在空间
q = concat(ql_nope, q_pe)
attn_out = flash_attn(q, kv_cache)                # 直接使用压缩缓存
o = einsum("snl,lnv->snv", attn_out, W_UV)       # 投影回完整空间
```

### 4.5 级联注意力 (Cascade Attention)

当多个请求共享长公共前缀时，级联注意力优化：

```python
# flash_attn.py, line 1132
def cascade_attention(query, key_cache, value_cache, block_table, ...):
    # 1. 前缀注意力: 所有查询 attend 到共享前缀
    prefix_out = flash_attn(
        q=query, k=key_cache, v=value_cache,
        block_table=block_table[:1],  # 使用第一个请求的块表
        causal=False
    )

    # 2. 后缀注意力: 每个查询 attend 到自己的后缀
    suffix_out = flash_attn(
        q=query, k=key_cache, v=value_cache,
        block_table=block_table[:, num_common_blocks:],
        causal=True
    )

    # 3. 合并结果 (基于 LSE 的重缩放)
    output = merge_attn_states(prefix_out, suffix_out)
```

**启用条件：**
- `common_prefix_len >= 256`
- 批处理中至少 8 个请求
- 不使用 ALiBi、滑动窗口或局部注意力

---

## 5. KV Cache 管理和调度

### 5.1 调度器与 KV Cache Manager 的交互

每个调度步骤中的交互序列：

```
Scheduler.schedule()
    │
    ├── 1. kv_cache_manager.new_step_starts()
    │      通知新步骤开始
    │
    ├── 2. 对每个等待请求:
    │      kv_cache_manager.get_computed_blocks(request)
    │      │  查找前缀缓存命中
    │      │  返回 (KVCacheBlocks, num_computed_tokens)
    │      ▼
    │
    ├── 3. 对每个请求:
    │      kv_cache_manager.allocate_slots(request, num_new_tokens)
    │      │  释放滑动窗口外的块
    │      │  计算需要的新块数
    │      │  分配新块
    │      │  返回 KVCacheBlocks 或 None
    │      ▼
    │
    └── 4. 对完成/抢占的请求:
           kv_cache_manager.free(request)
           释放所有块
```

### 5.2 allocate_slots 详解

这是 KV Cache 管理的核心方法：

```python
# vllm/v1/core/kv_cache_manager.py, line 236
def allocate_slots(
    self, request, num_new_tokens,
    num_new_computed_tokens=0,
    new_computed_blocks=None,
    num_lookahead_tokens=0,
    num_external_computed_tokens=0,
    delay_cache_blocks=False,
    num_encoder_tokens=0,
    full_sequence_must_fit=False,
) -> KVCacheBlocks | None:
```

**块布局：**

```
| < comp > | < new_comp > | < ext_comp > | < new > | < lookahead > |
   已计算      前缀缓存命中    外部缓存       新计算      推测 token
```

**步骤详解：**

```python
# Step 1: 计算总已计算 token 数
num_local_computed_tokens = request.num_computed_tokens + num_new_computed_tokens
total_computed_tokens = min(
    num_local_computed_tokens + num_external_computed_tokens,
    max_model_len
)

# Step 2: 准入检查 (full_sequence_must_fit=True 时)
if full_sequence_must_fit:
    num_blocks = get_num_blocks_to_allocate(apply_admission_cap=True)
    if num_blocks > free_blocks:
        return None  # 内存不足，拒绝准入

# Step 3: 释放滑动窗口外的块
coordinator.remove_skipped_blocks(request_id, total_computed_tokens)

# Step 4: 计算需要分配的块数
num_blocks_to_allocate = coordinator.get_num_blocks_to_allocate(...)

# Step 5: 容量检查
if num_blocks_to_allocate > block_pool.get_num_free_blocks():
    return None  # 内存不足

# Step 6: 附加前缀缓存命中的块
if new_computed_blocks:
    coordinator.allocate_new_computed_blocks(...)

# Step 7: 分配新块
new_blocks = coordinator.allocate_new_blocks(...)

# Step 8: 缓存块 (使它们可以被未来的前缀缓存命中)
if enable_caching and not delay_cache_blocks:
    coordinator.cache_blocks(request, num_tokens_to_cache)

return new_blocks
```

### 5.3 块分配流程

```python
# BlockPool.get_new_blocks() (block_pool.py, line 333)
def get_new_blocks(self, num_blocks):
    # 1. 从空闲队列头部弹出 LRU 块
    blocks = free_block_queue.popleft_n(num_blocks)

    for block in blocks:
        # 2. 如果块有缓存哈希，从前缀缓存中移除 (驱逐)
        _maybe_evict_cached_block(block)

        # 3. 增加引用计数
        block.ref_cnt = 1

    return blocks
```

### 5.4 块释放流程

```python
# SingleTypeKVCacheManager.free() (single_type_kv_cache_manager.py, line 346)
def free(self, request_id):
    req_blocks = req_to_blocks.pop(request_id, [])

    # 按逆序释放，使尾部块先被驱逐 (保留前缀)
    for block in reversed(req_blocks):
        block.ref_cnt -= 1
        if block.ref_cnt == 0 and not block.is_null:
            # 放回空闲队列尾部 (MRU 端)
            block_pool.free_block_queue.append(block)
```

### 5.5 滑动窗口块回收

对于滑动窗口注意力 (SWA)，窗口外的块被自动回收：

```python
# SlidingWindowManager.get_num_skipped_tokens()
def get_num_skipped_tokens(self, num_computed_tokens):
    return max(0, num_computed_tokens - self.sliding_window + 1)

# 示例: sliding_window=4, num_computed_tokens=7
# 返回 4 (token 0-3 在窗口外)
```

```python
# SingleTypeKVCacheManager.remove_skipped_blocks()
def remove_skipped_blocks(self, request_id, total_computed_tokens):
    num_skipped_tokens = self.get_num_skipped_tokens(total_computed_tokens)
    num_skipped_blocks = num_skipped_tokens // block_size

    # 从后往前遍历，释放窗口外的块
    for i in range(num_skipped_blocks - 1, -1, -1):
        if blocks[i].is_null:
            break  # 已经被释放过
        removed_blocks.append(blocks[i])
        blocks[i] = null_block  # 替换为空块

    block_pool.free_blocks(removed_blocks)
```

### 5.6 准入上限 (Admission Cap)

防止 SWA/分块局部注意力过度预留内存：

```python
# SlidingWindowSpec.max_admission_blocks_per_request()
max_blocks = cdiv(min(sliding_window - 1 + max_num_batched_tokens, max_model_len), block_size) + 1

# ChunkedLocalAttentionSpec.max_admission_blocks_per_request()
max_blocks = cdiv(min(attention_chunk_size + max_num_batched_tokens, max_model_len), block_size)
```

在 `get_num_blocks_to_allocate()` 中应用：

```python
if apply_admission_cap and _max_admission_blocks_per_request:
    num_required_blocks = min(num_required_blocks, _max_admission_blocks_per_request)
```

### 5.7 预抢占策略

vLLM v1 使用 **重计算 (recompute)** 预抢占：

```python
# Scheduler._preempt_request()
def _preempt_request(self, request):
    kv_cache_manager.free(request)      # 释放所有 KV 块
    request.num_computed_tokens = 0     # 重置已计算 token 数
    request.spec_token_ids = []         # 清除推测 token
    request.num_preemptions += 1
    waiting.prepend_request(request)    # 放回等待队列头部
```

**预抢占触发条件：**
- Phase 1 (调度 RUNNING 请求) 中，如果 `allocate_slots` 返回 `None`
- 选择最低优先级的 RUNNING 请求进行预抢占
- 被抢占的请求放回等待队列头部，优先重新调度

### 5.8 块的完整生命周期

```
1. 初始化: 所有块在空闲队列中。Block 0 成为 null_block。

2. 分配 (get_new_blocks):
   从空闲队列头部弹出 LRU 块
   如果有缓存哈希，从前缀缓存中移除 (驱逐)
   增加 ref_cnt 到 1

3. 缓存 (cache_full_blocks):
   块满后计算链式哈希
   存储到 cached_block_hash_to_block
   设置块的 _block_hash 字段

4. 前缀缓存查找 (get_cached_block):
   对每个 KV 缓存组，查找哈希
   返回所有组的块或 None

5. Touch (touch):
   缓存命中时，从空闲队列移除块 (不再是驱逐候选)
   增加 ref_cnt

6. 释放 (free_blocks):
   递减 ref_cnt
   当 ref_cnt == 0 时，放回空闲队列尾部 (MRU 端)

7. 驱逐 (_maybe_evict_cached_block):
   LRU 最旧的空闲块被重新分配时
   移除其前缀缓存条目
   重置 _block_hash
```

---

## 6. KV Cache 复用机制 (Prefix Caching)

### 6.1 概述

前缀缓存允许跨请求共享相同前缀的 KV Cache。当多个请求有相同的系统 prompt 或共同前缀时，只需计算一次，后续请求直接复用。

### 6.2 块哈希算法

块哈希是 **内容可寻址且链式依赖** 的：

```python
# vllm/v1/core/kv_cache_utils.py, line 541
def hash_block_tokens(hash_function, parent_block_hash, curr_block_token_ids, extra_keys):
    if parent_block_hash is None:
        parent_block_hash = NONE_HASH  # 首块的种子哈希

    return hash_function((parent_block_hash, tuple(curr_block_token_ids), extra_keys))
```

**关键特性：**
- **内容可寻址**：相同 token IDs 产生相同哈希（给定相同父哈希）
- **链式依赖**：块 N 的哈希依赖于块 N-1 的哈希，形成 Merkle 链
- **位置感知**：相同 token IDs 但在序列不同位置的块有不同哈希

```
Block 0: hash = H(NONE_HASH, tokens[0:32])
Block 1: hash = H(hash_0, tokens[32:64])
Block 2: hash = H(hash_1, tokens[64:96])

两个序列在 block 3 分叉:
  Seq A block 3: H(hash_2, tokens_A[96:128])
  Seq B block 3: H(hash_2, tokens_B[96:128])
  即使 block 4+ 有相同 token IDs，哈希也不同
```

### 6.3 额外哈希键

```python
# vllm/v1/core/kv_cache_utils.py, line 503
def generate_block_hash_extra_keys(request, start_token_idx, end_token_idx, start_mm_idx):
    extra_keys = []

    # 1. LoRA 键
    if request.lora_request:
        extra_keys.append(request.lora_request.lora_name)

    # 2. 多模态键
    for mm_feature in request.mm_features[start_mm_idx:]:
        if mm_feature overlaps block range:
            extra_keys.append((mm_feature.identifier, offset))

    # 3. 缓存盐值 (仅第一个块)
    if start_token_idx == 0 and request.cache_salt:
        extra_keys.append(request.cache_salt)

    # 4. Prompt 嵌入哈希
    if request.prompt_embeds:
        extra_keys.append(hash(block's prompt_embeds slice))

    return tuple(extra_keys) if extra_keys else None
```

### 6.4 前缀缓存查找

```python
# KVCacheManager.get_computed_blocks() (kv_cache_manager.py, line 194)
def get_computed_blocks(self, request):
    if not enable_caching or request.skip_reading_prefix_cache:
        return (empty_blocks, 0)

    # 最大缓存命中长度 = 总 token 数 - 1 (最后一个 token 必须重新计算)
    max_cache_hit_length = request.num_tokens - 1

    # 委托给协调器
    hit_blocks = coordinator.find_longest_cache_hit(
        request.block_hashes, max_cache_hit_length
    )

    return (hit_blocks, len(hit_blocks[0]) * block_size)
```

#### 6.4.1 全注意力的查找 (从左到右)

```python
# FullAttentionManager.find_longest_cache_hit()
def find_longest_cache_hit(self, block_hashes, max_length):
    computed_blocks = [[], ...]  # 每组一个列表

    for i in range(max_num_blocks):
        # 查找第 i 个块的哈希
        cached = block_pool.get_cached_block(block_hashes[i], group_ids)
        if cached:
            for group_id, block in enumerate(cached):
                computed_blocks[group_id].append(block)
        else:
            break  # 前缀缓存是连续的，中断即停

    return computed_blocks
```

#### 6.4.2 滑动窗口的查找 (从右到左)

```python
# SlidingWindowManager.find_longest_cache_hit()
def find_longest_cache_hit(self, block_hashes, max_length):
    # 需要连续的 sliding_window_contiguous_blocks 个块
    needed = cdiv(sliding_window - 1, block_size)

    # 从右到左搜索 (只有序列尾部对 SWA 重要)
    for i in range(max_num_blocks - 1, -1, -1):
        cached = block_pool.get_cached_block(block_hashes[i], group_ids)
        if cached:
            computed_blocks[i] = cached
            num_contiguous += 1
            if num_contiguous >= needed:
                break
        else:
            num_contiguous = 0

    return computed_blocks
```

#### 6.4.3 混合模型的查找 (固定点迭代)

```python
# HybridKVCacheCoordinator.find_longest_cache_hit()
def find_longest_cache_hit(self, block_hashes, max_cache_hit_length):
    hit_length = max_cache_hit_length

    while True:
        prev_hit_length = hit_length

        # 全注意力先处理 (下闭属性: 只会缩短)
        full_attn_hits = full_attn_manager.find_longest_cache_hit(...)
        hit_length = min(hit_length, full_attn_hits_length)

        # 然后处理其他注意力类型
        for manager in other_managers:
            hits = manager.find_longest_cache_hit(block_hashes, hit_length, ...)
            hit_length = min(hit_length, hits_length)

        # 收敛检查
        if hit_length >= prev_hit_length:
            break  # 不再缩短，收敛

    return (hit_blocks, hit_length)
```

### 6.5 前缀缓存插入

```python
# BlockPool.cache_full_blocks() (block_pool.py, line 211)
def cache_full_blocks(self, request, blocks, num_cached_blocks, num_full_blocks, ...):
    for i in range(num_cached_blocks, num_full_blocks):
        block = blocks[i]
        if block.is_null:
            continue

        # 计算块哈希 (带组 ID)
        block_hash = request.block_hashes[i]
        block_hash_with_group_id = make_block_hash_with_group_id(block_hash, group_id)

        # 设置块哈希 (只能设置一次)
        block.block_hash = block_hash_with_group_id

        # 插入前缀缓存
        cached_block_hash_to_block.insert(block_hash_with_group_id, block)
```

### 6.6 前缀缓存驱逐

驱逐通过空闲队列的 LRU 顺序隐式实现：

```python
# BlockPool._maybe_evict_cached_block() (block_pool.py, line 365)
def _maybe_evict_cached_block(self, block):
    if block.block_hash is None:
        return False  # 没有缓存

    # 从前缀缓存中移除
    cached_block_hash_to_block.pop(block.block_hash, block.block_id)

    # 重置块哈希
    block.reset_hash()

    return True
```

**驱逐时机：**
- 当 LRU 最旧的空闲块被 `get_new_blocks()` 重新分配时
- 块在空闲队列中时仍保留缓存条目（惰性驱逐）
- 只有被重新分配时才真正从缓存中移除

### 6.7 Touch 机制

当一个缓存命中的块被新请求使用时：

```python
# BlockPool.touch() (block_pool.py, line 402)
def touch(self, blocks):
    for block in blocks:
        if block.ref_cnt == 0 and not block.is_null:
            # 块在空闲队列中 (是驱逐候选)
            # 从空闲队列移除 (O(1) 操作)
            free_block_queue.remove(block)

        # 增加引用计数
        block.ref_cnt += 1
```

**Touch 的效果：**
- 块离开空闲队列（不再是驱逐候选）
- 引用计数增加
- 当新请求完成后，块会以 ref_cnt 递减回到空闲队列尾部（MRU 端）

### 6.8 Mamba 的前缀缓存

Mamba 层的前缀缓存有特殊限制：

```python
# MambaManager.cache_blocks()
def cache_blocks(self, request, num_tokens):
    super().cache_blocks(request, num_tokens)
    # 记录本步骤缓存的块，防止同一步骤内其他请求复用
    self.cached_blocks_this_step.add(block_hash)

# MambaManager.get_num_blocks_to_allocate()
def get_num_blocks_to_allocate(self, ...):
    # 如果最后的新计算块哈希在本步骤缓存中，强制分配失败
    # Mamba 不能复用同一步骤内其他请求缓存的块
    if last_block_hash in self.cached_blocks_this_step:
        return num_gpu_blocks + 1  # 触发 "内存不足"
```

---

## 7. LMCache 对接

### 7.1 KVConnector 抽象

`KVConnectorBase_V1` 是所有 KV 传输/卸载机制的统一接口：

```python
# vllm/distributed/kv_transfer/kv_connector/v1/base.py
class KVConnectorBase_V1:
    # === 调度器侧方法 (运行在调度器进程) ===

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        """返回可以从外部存储加载的 token 数"""
        return (num_tokens, is_async)

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        """块分配后更新状态"""

    def build_connector_meta(self, scheduler_output):
        """构建传递给 Worker 的元数据"""

    def update_connector_output(self, connector_output):
        """接收 Worker 的完成通知"""

    def request_finished(self, request, block_ids):
        """请求完成时调用，可延迟块释放"""

    # === Worker 侧方法 (运行在每个 Worker 进程) ===

    def register_kv_caches(self, kv_caches):
        """注册 GPU KV Cache 张量"""

    def start_load_kv(self, forward_context):
        """开始异步 KV 加载"""

    def wait_for_layer_load(self, layer_name):
        """等待特定层的 KV 加载完成"""

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata):
        """开始异步 KV 保存"""

    def wait_for_save(self):
        """等待所有保存完成"""
```

### 7.2 已注册的 KV 连接器

| 连接器 | 用途 |
|--------|------|
| `LMCacheConnectorV1` | LMCache 集成 (主要) |
| `LMCacheMPConnector` | LMCache 多进程模式 |
| `NixlConnector` | NIXL RDMA 传输 |
| `P2pNcclConnector` | 点对点 NCCL 传输 |
| `MooncakeConnector` | Mooncake 分布式 KV 存储 |
| `OffloadingConnector` | 原生 CPU/分层卸载 |
| `SimpleCPUOffloadConnector` | 简化 CPU 卸载 |
| `MultiConnector` | 组合多个连接器 |
| `FlexKVConnectorV1` | FlexKV 传输 |

### 7.3 LMCache 集成

LMCache 是一个外部 KV Cache 管理系统，提供分布式存储、检索和 P/D 分离能力。

#### 7.3.1 两种 LMCache 连接器

**LMCacheConnectorV1** (主要连接器)：

```python
# vllm/distributed/kv_transfer/kv_connector/v1/lmcache_connector.py
class LMCacheConnectorV1(KVConnectorBase_V1):
    def __init__(self, vllm_config):
        # 调度器侧: LookupClient 用于缓存命中检查
        self.lookup_client = create_lookup_client(...)

        # Worker 侧: LMCacheEngine 管理数据移动
        self.lmcache_engine = LMCacheEngine(...)
```

**LMCacheMPConnector** (多进程模式)：

```python
# LMCache 作为独立服务器进程运行
# 通过 ZMQ 消息队列通信
# Worker 使用 CUDA IPC 事件进行跨进程同步
```

#### 7.3.2 LMCache 数据流

```
Lookup (查找):
  Scheduler → get_num_new_matched_tokens()
    → LookupClient 查询 LMCache
    → 返回匹配的 token 数

Load (加载):
  Worker → start_load_kv()
    → LMCacheEngine 从远程存储检索 KV
    → 使用 slot_mapping 写入 vLLM 的分页 KV 缓冲区

Save (保存):
  Worker → wait_for_save()
    → LMCacheEngine 存储新计算的 KV
    → 支持分块存储 (对齐到 lmcache_chunk_size)
```

#### 7.3.3 LMCache 配置

```python
# 通过环境变量
export LMCACHE_CONFIG_FILE=/path/to/config.yaml

# 或通过 kv_connector_extra_config
kv_transfer_config = {
    "kv_connector": "LMCacheConnectorV1",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
        "lmcache.use_native": True,
        "lmcache.use_layerwise": True,
        "lmcache.chunk_size": 256,
        "lmcache.enable_blending": False,
    }
}
```

### 7.4 原生 KV 卸载系统

vLLM 内置的 KV Cache 卸载机制，不依赖外部系统。

#### 7.4.1 架构

```
OffloadingSpec (工厂)
    │
    ├── CPUOffloadingSpec → CPUOffloadingManager (调度器侧)
    │                        CpuGpuOffloadingHandlers (Worker 侧)
    │
    └── TieringOffloadingSpec → TieringOffloadingManager (调度器侧)
                                 支持多级存储
```

#### 7.4.2 CPU 卸载

```python
# vllm/v1/kv_offload/cpu/manager.py
class CPUOffloadingManager:
    """调度器侧管理器，跟踪卸载的块"""

    def lookup(self, block_hashes):
        """查找已卸载的块"""

    def prepare_store(self, blocks):
        """准备存储操作"""

    def prepare_load(self, block_hashes):
        """准备加载操作"""
```

**Worker 侧传输：**

```python
# 使用专用 CUDA 流进行 GPU ↔ CPU DMA 拷贝
# 支持批量拷贝: ops.swap_blocks_batch
# 使用共享内存 (/dev/shm/vllm_offload_{id}.mmap) 跨 Worker 协调
```

#### 7.4.3 分层存储

```
GPU (主存) → CPU (一级) → 磁盘/网络 (二级)

Store: GPU → CPU → 二级 (级联)
Load:  二级 → CPU → GPU (提升)
```

**已注册的二级存储：**
- `fs`：文件系统存储，使用 `os.write`/`os.readv` 和双队列线程池

### 7.5 P/D 分离 (Prefill/Decode Disaggregation)

P/D 分离将预填充和解码阶段部署在不同的 GPU 上：

```
Prefill Node (预填充节点):
  - 处理长 prompt 的 prefill
  - 计算 KV Cache
  - 通过 KVConnector 传输 KV Cache

Decode Node (解码节点):
  - 接收 KV Cache
  - 执行自回归解码
  - 生成输出 token
```

配置示例：

```bash
# Prefill 节点
vllm serve model --kv-transfer-config \
    '{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_rank":0}'

# Decode 节点
vllm serve model --kv-transfer-config \
    '{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_rank":1}'
```

---

## 8. 关键类和方法

### 8.1 类层次总览

```
KVCacheManager
    │
    └── KVCacheCoordinator (抽象基类)
         ├── KVCacheCoordinatorNoPrefixCache
         ├── UnitaryKVCacheCoordinator (单缓存组)
         └── HybridKVCacheCoordinator (混合缓存组)

SingleTypeKVCacheManager (抽象基类)
    ├── FullAttentionManager (全注意力、MLA、TQ)
    ├── SlidingWindowManager (滑动窗口)
    ├── ChunkedLocalAttentionManager (分块局部注意力)
    ├── MambaManager (SSM/Mamba)
    ├── CrossAttentionManager (交叉注意力)
    └── SinkFullAttentionManager (注意力汇聚)

BlockPool
    ├── FreeKVCacheBlockQueue (空闲块双向链表)
    └── BlockHashToBlockMap (前缀缓存哈希表)

KVCacheBlock (数据类)

KVConnectorBase_V1 (抽象基类)
    ├── LMCacheConnectorV1
    ├── NixlConnector
    ├── OffloadingConnector
    └── ...
```

### 8.2 关键方法速查表

| 方法 | 所属类 | 用途 |
|------|--------|------|
| `allocate_slots()` | KVCacheManager | 分配 KV 块 (核心方法) |
| `free()` | KVCacheManager | 释放请求的所有块 |
| `get_computed_blocks()` | KVCacheManager | 查找前缀缓存命中 |
| `new_step_starts()` | KVCacheManager | 通知新调度步骤开始 |
| `find_longest_cache_hit()` | SingleTypeKVCacheManager | 查找最长缓存命中 |
| `allocate_new_computed_blocks()` | SingleTypeKVCacheManager | 附加前缀缓存命中的块 |
| `allocate_new_blocks()` | SingleTypeKVCacheManager | 从空闲池分配新块 |
| `cache_blocks()` | SingleTypeKVCacheManager | 使块可被前缀缓存命中 |
| `remove_skipped_blocks()` | SingleTypeKVCacheManager | 释放窗口外的块 |
| `get_num_skipped_tokens()` | SingleTypeKVCacheManager | 计算需要跳过的 token 数 |
| `get_num_blocks_to_allocate()` | SingleTypeKVCacheManager | 计算需要分配的块数 |
| `get_new_blocks()` | BlockPool | 从空闲池弹出 LRU 块 |
| `free_blocks()` | BlockPool | 释放块到空闲池 |
| `cache_full_blocks()` | BlockPool | 缓存完整的块 |
| `get_cached_block()` | BlockPool | 查找缓存的块 |
| `touch()` | BlockPool | 处理缓存命中 (防止驱逐) |
| `_maybe_evict_cached_block()` | BlockPool | 驱逐块的缓存条目 |
| `hash_block_tokens()` | 工具函数 | 计算链式块哈希 |
| `bind_kv_cache()` | 工具函数 | 绑定张量到注意力层 |

### 8.3 核心文件索引

| 文件 | 关键内容 |
|------|----------|
| `vllm/v1/core/kv_cache_manager.py` | KVCacheManager、KVCacheBlocks |
| `vllm/v1/core/single_type_kv_cache_manager.py` | 各类注意力的缓存管理器 |
| `vllm/v1/core/kv_cache_coordinator.py` | 缓存协调器 |
| `vllm/v1/core/block_pool.py` | BlockPool、BlockHashToBlockMap |
| `vllm/v1/core/kv_cache_utils.py` | KVCacheBlock、哈希函数、FreeKVCacheBlockQueue |
| `vllm/v1/kv_cache_interface.py` | KVCacheSpec、KVCacheConfig、KVCacheTensor |
| `vllm/v1/worker/block_table.py` | BlockTable、Slot Mapping |
| `vllm/v1/worker/gpu_model_runner.py` | initialize_kv_cache、get_kv_cache_spec |
| `vllm/v1/worker/gpu_worker.py` | determine_available_memory |
| `vllm/v1/engine/core.py` | _initialize_kv_caches |
| `vllm/v1/attention/backends/flash_attn.py` | FlashAttention 后端 |
| `vllm/v1/attention/backends/flashinfer.py` | FlashInfer 后端 |
| `vllm/v1/attention/backend.py` | 注意力后端接口 |
| `vllm/v1/kv_offload/` | KV 卸载系统 |
| `vllm/distributed/kv_transfer/` | KV 传输和连接器 |

---

## 9. 总结

### 9.1 vLLM KV Cache 管理的核心优势

| 优势 | 说明 |
|------|------|
| **PagedAttention** | 分页管理消除内存碎片，按需分配减少浪费 |
| **前缀缓存** | 内容可寻址、链式依赖的块哈希，自动复用公共前缀 |
| **滑动窗口回收** | 窗口外块自动释放，显著减少内存使用 |
| **分层卸载** | GPU → CPU → 磁盘的多级缓存，扩展可用容量 |
| **统一接口** | KVConnector 抽象统一所有传输/卸载机制 |
| **混合模型支持** | 全注意力、滑动窗口、Mamba 等异构缓存类型统一管理 |
| **跨层共享** | 支持层间 KV Cache 共享，减少内存使用 |
| **量化支持** | FP8、INT8、NVFP4 等 KV Cache 量化模式 |

### 9.2 性能优化技术

| 技术 | 收益 |
|------|------|
| **链式块哈希** | O(1) 前缀缓存查找，自动位置感知 |
| **LRU 空闲队列** | O(1) 分配/释放/驱逐，无额外 Python 对象分配 |
| **Touch 机制** | O(1) 缓存命中处理，防止活跃块被驱逐 |
| **惰性驱逐** | 块在空闲队列中保留缓存条目，只在重新分配时移除 |
| **块大小拆分** | 调度器和内核使用不同的块大小，优化各自性能 |
| **统一页面大小** | 混合模型统一页面大小，简化内存管理 |
| **准入上限** | 防止 SWA/分块局部注意力过度预留内存 |

### 9.3 典型使用场景

| 场景 | KV Cache 策略 |
|------|---------------|
| **单模型服务** | 默认 PagedAttention + 前缀缓存 |
| **长上下文** | 分块预填充 + 滑动窗口回收 |
| **多轮对话** | 前缀缓存自动复用历史对话 |
| **P/D 分离** | KVConnector + NIXL/LMCache 传输 |
| **大规模部署** | CPU 卸载 + 分层存储扩展容量 |
| **MoE 模型** | 跨层 KV 共享 + 数据并行 |
| **推测解码** | EAGLE/Medusa 使用独立的 KV Cache 组 |

### 9.4 配置建议

| 场景 | 推荐配置 |
|------|----------|
| **通用服务** | `enable_prefix_caching=True`, `gpu_memory_utilization=0.9` |
| **长序列** | `enable_chunked_prefill=True`, 适当增大 `max_model_len` |
| **多轮对话** | `enable_prefix_caching=True`, 使用 `cache_salt` 区分会话 |
| **内存受限** | `kv_cache_dtype="fp8"`, 降低 `gpu_memory_utilization` |
| **P/D 分离** | 配置 `kv_transfer_config` 和 KVConnector |
| **高吞吐** | `enable_prefix_caching=True`, `enable_chunked_prefill=True` |

---

## 附录

### A. 术语表

| 术语 | 说明 |
|------|------|
| **Block** | KV Cache 的固定大小分配单元 |
| **Block Table** | 逻辑位置到物理块的映射表 |
| **Slot Mapping** | Token 位置到 KV Cache slot 的映射 |
| **Prefix Caching** | 前缀缓存，复用公共前缀的 KV Cache |
| **PagedAttention** | 分页注意力，vLLM 的核心创新 |
| **KVCacheSpec** | 每层 KV Cache 的规格描述 |
| **KVCacheConfig** | KV Cache 的最终配置 |
| **KVCacheGroup** | 共享相同块表的层组 |
| **Admission Cap** | 准入上限，防止过度预留 |
| **Touch** | 缓存命中时防止块被驱逐的操作 |
| **LRU** | Least Recently Used，最近最少使用 |
| **MRU** | Most Recently Used，最近最常使用 |
| **Null Block** | 空块占位符，用于滑动窗口填充 |
| **KVConnector** | KV 传输/卸载的统一接口 |
| **LMCache** | 外部 KV Cache 管理系统 |
| **P/D Disaggregation** | 预填充/解码分离部署 |

### B. 参考资料

- [vLLM 官方文档](https://docs.vllm.ai/)
- [PagedAttention 论文](https://arxiv.org/abs/2309.06180)
- [LMCache 项目](https://github.com/LMCache/LMCache)
- [vLLM GitHub 仓库](https://github.com/vllm-project/vllm)
