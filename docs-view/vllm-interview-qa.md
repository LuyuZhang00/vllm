# vLLM v1 面试题完整版

> 基于 vLLM v1 架构，每个问题采用 **总分结构**：先是面试文字回答版（可直接背诵），再是结合具体代码的分析版。

---

## 目录

- [1. 请求到达前端后的完整处理流程](#1-请求到达前端后的完整处理流程)
- [2. KV Cache 是如何计算的](#2-kv-cache-是如何计算的)
- [3. Prefix Cache 是如何计算的](#3-prefix-cache-是如何计算的)
- [4. Prefill 和 Decode 能否放在同一个 Batch 中](#4-prefill-和-decode-能否放在同一个-batch-中)
- [5. Decode 会用到 Prefill 做的 KV Cache 吗](#5-decode-会用到-prefill-做的-kv-cache-吗)
- [6. vLLM 的 Overlap 机制](#6-vllm-的-overlap-机制)
- [7. 什么时候区分 P 和 D](#7-什么时候区分-p-和-d)
- [8. D 节点 Scheduler 的调度逻辑与 Overlap](#8-d-节点-scheduler-的调度逻辑与-overlap)
- [9. GPU 前向传播做了什么](#9-gpu-前向传播做了什么)

---

## 1. 请求到达前端后的完整处理流程

### 面试回答版

Request 到达 vLLM 前端后，先解析请求、套 chat template、tokenize，得到 prompt token ids。然后进入 Engine Scheduler，调度器结合 token budget、KV Cache 空间和 prefix cache 命中情况，决定本轮执行哪些请求的多少个 token。送进 GPU 做前向传播时，所有请求的 token 混在同一个 batch 里，每层 attention 统一执行：算 Q/K/V → 写 KV Cache → 注意力计算 → FFN。最后采样出 token 返回给用户。

### 代码分析版

#### Step 1：前端接收请求

```python
# vllm/v1/engine/async_llm.py, line 524
async def generate(self, prompt, sampling_params, request_id, ...):
    # 1. 提交请求
    await self.add_request(prompt, sampling_params, request_id, ...)

    # 2. 从队列中循环获取输出
    q = self.output_queues[request_id]
    while True:
        if (output := q.get_nowait()) is not None:
            yield output
        else:
            output = await q.get()
            yield output
```

#### Step 2：输入处理（分词、验证）

```python
# vllm/v1/engine/input_processor.py, line 242
def process_inputs(self, request_id, prompt, params, ...) -> EngineCoreRequest:
    self._validate_params(params)                    # 验证参数
    processed_inputs = self.input_preprocessor.preprocess(prompt, ...)  # 分词
    mm_features = self._process_mm_features(...)     # 处理多模态
    return EngineCoreRequest(
        request_id=request_id,
        prompt_token_ids=processed_inputs.prompt_token_ids,
        sampling_params=params,
    )
```

#### Step 3：EngineCore 接收请求

```python
# vllm/v1/engine/core.py, line 354
def add_request(self, request: Request, request_wave: int = 0):
    self._validate_request(request)
    self.scheduler.add_request(request)  # 放入 waiting 队列
```

#### Step 4：调度器调度

```python
# vllm/v1/core/sched/scheduler.py, line 428
def schedule(self) -> SchedulerOutput:
    token_budget = self.max_num_scheduled_tokens

    # Phase 1: RUNNING 请求 (已经在跑的)
    for request in self.running:
        num_new_tokens = (
            request.num_tokens_with_spec
            + request.num_output_placeholders
            - request.num_computed_tokens
        )
        blocks = self.kv_cache_manager.allocate_slots(request, num_new_tokens)
        if blocks is None:
            self._preempt_request(lowest_priority_request)
            continue
        token_budget -= num_new_tokens

    # Phase 2: WAITING 请求 (新来的)
    if not any_preempted:
        for request in self.waiting:
            # ★ 查找前缀缓存
            computed_blocks, num_computed = \
                self.kv_cache_manager.get_computed_blocks(request)

            num_new_tokens = request.num_tokens - num_computed

            # 分块预填充：截断到 token_budget
            if self.enable_chunked_prefill:
                num_new_tokens = min(num_new_tokens, token_budget)

            # 分配 KV 块 (包含前缀缓存命中的块)
            blocks = self.kv_cache_manager.allocate_slots(
                request, num_new_tokens,
                num_new_computed_tokens=num_computed,
                new_computed_blocks=computed_blocks,
            )
            if blocks is None:
                break  # 内存不足，停止调度

            token_budget -= num_new_tokens
            request.status = RequestStatus.RUNNING
            self.running.append(request)
```

#### Step 5：模型前向传播

```python
# vllm/v1/worker/gpu_model_runner.py, line 3963
def execute_model(self, scheduler_output, intermediate_tensors=None):
    # 1. 更新持久化批处理状态
    self._update_states(scheduler_output)

    # 2. 准备输入张量
    self._prepare_inputs(scheduler_output, num_scheduled_tokens)

    # 3. 构建注意力元数据 (block_table, slot_mapping 等)
    self._build_attention_metadata(scheduler_output)

    # 4. 模型前向传播
    hidden_states = self.model(input_ids, positions, ...)

    # 5. 计算 logits
    logits = self.model.compute_logits(hidden_states)

    # 6. 存储状态，返回 None (延迟采样)
    self.execute_model_state = (logits, ...)
    return None
```

#### Step 6：采样和输出

```python
# vllm/v1/worker/gpu_model_runner.py, line 4427
def sample_tokens(self, grammar_output):
    logits, ... = self.execute_model_state

    # 1. 应用语法掩码 (结构化输出)
    if grammar_output:
        logits = apply_grammar_mask(logits, grammar_output)

    # 2. 采样
    sampled_token_ids = self.sampler(logits, ...)

    # 3. 更新状态
    self._update_states_after_model_execute(sampled_token_ids)

    # 4. 构建输出
    return ModelRunnerOutput(sampled_token_ids=sampled_token_ids, ...)
```

#### 完整流程图

```
请求 "你好世界"
    │
    ▼
InputProcessor: 分词 + 验证 → [token_0, token_1, ..., token_99]
    │
    ▼
EngineCore.add_request() → Scheduler.waiting 队列
    │
    ▼
Scheduler.schedule():
    查 prefix cache → 分配 KV 块 → 决定跑多少 token
    │
    ▼
GPUModelRunner.execute_model():
    更新状态 → 准备输入 → 前向传播 → 算 logits
    │
    ▼
sample_tokens(): 采样 → 返回 token
    │
    ▼
OutputProcessor: 反分词 → 返回给用户
    │
    ▼
循环 schedule → execute → sample 直到结束
```

---

## 2. KV Cache 是如何计算的

### 面试回答版

KV Cache 在每层 attention 的 forward 中计算和写入。每层先算出当前 token 的 K 和 V，然后通过 slot_mapping 散射写入 paged KV cache 的指定位置。之后做注意力计算时，Q attend 到整个 KV cache（包含所有历史 K/V）。写入路径对所有 token 完全一致，不区分它是 prompt 的第一个 token 还是生成的第一百个 token。

### 代码分析版

#### KV Cache 写入发生在注意力层

```python
# vllm/model_executor/layers/attention/attention.py, line 480
class Attention(nn.Module):
    def forward(self, query, key, value, ...):
        # 1. 计算新的 K, V
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        # 2. 写入 KV Cache (在注意力计算之前)
        if not self.attn_backend.forward_includes_kv_cache_update:
            unified_kv_cache_update(key, value, self.layer_name)
            # ↑ 这个函数将 K, V 写入 paged KV cache 的指定位置

        # 3. 执行注意力计算 (读取 KV Cache)
        output = unified_attention_with_output(query, key, value, ...)
        return output
```

#### 具体的 KV Cache 写入操作

```python
# vllm/v1/attention/backends/flash_attn.py, line 850
def do_kv_cache_update(self, layer, key, value, kv_cache, slot_mapping):
    # 1. 拆分 K 和 V 缓存
    key_cache, value_cache = kv_cache.unbind(1)

    # 2. 散射写入 (scatter write)
    #    slot_mapping 告诉每个 token 应该写入缓存的哪个位置
    reshape_and_cache_flash(
        key,           # 新的 K 张量
        value,         # 新的 V 张量
        key_cache,     # K 缓存 (整个 paged cache)
        value_cache,   # V 缓存 (整个 paged cache)
        slot_mapping,  # 每个 token 对应的物理 slot 位置
        ...
    )
```

#### slot_mapping 的计算

```python
# vllm/v1/worker/block_table.py
# 对于请求 r 的第 pos 个 token:
block_number = block_table[r, pos // block_size]
block_offset = pos % block_size
slot_id = block_number * block_size + block_offset

# 示例: 请求 0, 位置 45, block_size=16, block_table[0, 2]=7
# slot_id = 7 * 16 + (45 % 16) = 7 * 16 + 13 = 125
# K_cache[125] = new_key, V_cache[125] = new_value
```

#### Paged KV Cache 的物理布局

```
物理块 (block_size=4):
Block 0: [K_A, K_B, K_C, K_D]     ← 4 个 token 的 K
Block 1: [K_E, K_F, -, -]          ← 2 个 token 的 K + 2 个空位

块表 (Block Table):
Request 0 → [Block 0, Block 1]

slot_mapping:
token A → slot 0, token B → slot 1, token C → slot 2, token D → slot 3
token E → slot 4, token F → slot 5
```

#### Prefill vs Decode 的区别

```
Prefill: num_new_tokens = prompt_length (可能是几百到几千)
  → slot_mapping 包含 prompt 每个 token 的位置
  → reshape_and_cache_flash 一次性写入所有 token 的 K/V

Decode: num_new_tokens = 1 (或 1 + num_spec_tokens)
  → slot_mapping 只包含 1 个位置
  → reshape_and_cache_flash 写入 1 个 token 的 K/V

代码路径完全相同！只是 slot_mapping 的长度不同。
```

---

## 3. Prefix Cache 是如何计算的

### 面试回答版

Prefix cache 在调度阶段按 block 计算 hash，如果命中已有 KV block，就直接复用，跳过共享 prefix 的计算。块哈希是链式依赖的——每个块的哈希依赖于父块的哈希，形成 Merkle 链，确保位置感知。查找从左到右连续进行，中断即停。

### 代码分析版

#### 查找过程

```python
# vllm/v1/core/kv_cache_manager.py, line 194
def get_computed_blocks(self, request):
    # 最后一个 token 必须重算（为了拿到 logits）
    max_cache_hit_length = request.num_tokens - 1

    # 委托给协调器查找
    hit_blocks = self.coordinator.find_longest_cache_hit(
        request.block_hashes, max_cache_hit_length
    )
    return (hit_blocks, len(hit_blocks[0]) * block_size)
```

#### 协调器查找（从左到右，连续命中）

```python
# vllm/v1/core/kv_cache_coordinator.py, line 373 (UnitaryKVCacheCoordinator)
def find_longest_cache_hit(self, block_hashes, max_cache_hit_length):
    computed_blocks = []

    # 从左到右连续查找
    for i in range(max_num_blocks):
        block_hash = block_hashes[i]

        # 在 BlockPool 中查找
        cached = self.block_pool.get_cached_block(block_hash, [0])

        if cached is not None:
            computed_blocks.append(cached[0])
        else:
            break  # 前缀缓存是连续的，中断即停

    return computed_blocks
```

#### BlockPool 查找缓存的块

```python
# vllm/v1/core/block_pool.py, line 184
def get_cached_block(self, block_hash, kv_cache_group_ids):
    results = []
    for group_id in kv_cache_group_ids:
        # 构建带组 ID 的哈希
        key = make_block_hash_with_group_id(block_hash, group_id)

        # 在哈希表中查找
        cached = self.cached_block_hash_to_block.get_one_block(key)

        if cached is None:
            return None  # 任何一个组未命中，返回 None

        results.append(cached)

    return results  # 所有组都命中
```

#### 链式块哈希

```python
# vllm/v1/core/kv_cache_utils.py, line 633
def hash_block_tokens(hash_function, parent_block_hash, token_ids, extra_keys):
    # 首块使用 NONE_HASH 作为种子
    if parent_block_hash is None:
        parent_block_hash = NONE_HASH

    # 哈希 = f(父哈希, 当前块 token IDs, 额外键)
    return hash_function((parent_block_hash, tuple(token_ids), extra_keys))
```

**示例：**

```
Block 0: hash = H(NONE_HASH, tokens[0:16])
Block 1: hash = H(hash_0, tokens[16:32])
Block 2: hash = H(hash_1, tokens[32:48])

两个请求有相同前缀:
  Request A: [Hello, world, how, are, you, ...] → Block 0, 1, 2 哈希相同
  Request B: [Hello, world, how, are, you, ...] → Block 0, 1, 2 哈希相同

  Request B 查找时: block 0 命中, block 1 命中, block 2 命中
  → 直接复用 Request A 的 KV Cache，无需重新计算！
```

#### 块缓存的插入

```python
# vllm/v1/core/block_pool.py, line 211
def cache_full_blocks(self, request, blocks, num_cached_blocks, num_full_blocks, ...):
    # 遍历新填满的块
    for i in range(num_cached_blocks, num_full_blocks):
        block = blocks[i]
        if block.is_null:
            continue

        # 获取块哈希
        block_hash = request.block_hashes[i]
        block_hash_with_group_id = make_block_hash_with_group_id(block_hash, group_id)

        # 设置块哈希 (只能设置一次)
        block.block_hash = block_hash_with_group_id

        # 插入前缀缓存哈希表
        self.cached_block_hash_to_block.insert(block_hash_with_group_id, block)
```

#### 前缀缓存的效果

```
请求到达，prompt 1000 token
  ↓ prefix cache 查找
  ↓ 命中 800 个 token 的 KV Cache
  ↓ 只需要算 200 个 token
  ↓ 节省 80% 计算！
```

---

## 4. Prefill 和 Decode 能否放在同一个 Batch 中

### 面试回答版

能，而且这是 vLLM v1 的核心设计。调度器不区分 prefill 和 decode，一个 batch 里既有 prefill token 又有 decode token，混在一起算。调度器只关心每个请求的 `num_computed_tokens` 和 `num_tokens` 的差距，用统一的 token budget 来限制总量。

### 代码分析版

#### 调度器的原话

```python
# vllm/v1/core/sched/scheduler.py, line 438
# NOTE(woosuk) on the scheduling algorithm:
# There's no "decoding phase" nor "prefill phase" in the scheduler.
# Each request just has the num_computed_tokens and
# num_tokens_with_spec. At each step, the scheduler tries to
# assign tokens to the requests so that each request's
# num_computed_tokens can catch up its num_tokens_with_spec.
```

#### 混合 batch 的攒法

```
当前状态:
  Running (decode): Request A (已算 100/101), Request B (已算 50/51)
  Waiting (prefill): Request C (prompt 1000 token, 已算 0/1000)

token_budget = 2048

调度过程:
  Phase 1 - RUNNING:
    Request A: num_new_tokens = 101 - 100 = 1 (decode)
    Request B: num_new_tokens = 51 - 50 = 1 (decode)
    token_budget = 2048 - 1 - 1 = 2046

  Phase 2 - WAITING:
    Request C: num_new_tokens = 1000 - 0 = 1000 (prefill)
    token_budget = 2046 - 1000 = 1046

最终 batch:
  Request A: 1 token (decode)
  Request B: 1 token (decode)
  Request C: 1000 tokens (prefill)
  总计: 1002 tokens，混在一起
```

#### GPU 怎么处理混合 batch

```python
# vllm/v1/worker/gpu_model_runner.py, line 1867
def _prepare_inputs(self, scheduler_output, num_scheduled_tokens_np):
    # num_scheduled_tokens_np 示例: [1, 1, 1000]

    # 1. 扩展请求索引
    req_indices = np.repeat(self.arange_np[:num_reqs], num_scheduled_tokens_np)
    # 结果: [0, 1, 2, 2, 2, ..., 2]  (Request C 占 1000 个位置)

    # 2. 计算位置编码
    positions = []
    for req_idx, num_tokens in enumerate(num_scheduled_tokens_np):
        start_pos = requests[req_idx].num_computed_tokens
        positions.extend(range(start_pos, start_pos + num_tokens))
    # Request A: [100], Request B: [50], Request C: [0, 1, 2, ..., 999]

    # 3. 所有 token 打包成连续张量
    input_ids = gather(input_ids_storage, token_indices)
    positions = torch.tensor(positions)
    slot_mapping = compute_slot_mapping(positions, block_table)
```

#### 注意力计算

```python
# vllm/v1/attention/backends/flash_attn.py, line 796
# 使用 variable-length FlashAttention
flash_attn_varlen_func(
    q=query,                    # [total_tokens, num_heads, head_size]
    k=key_cache,                # [num_blocks, 2, block_size, num_kv_heads, head_size]
    v=value_cache,
    cu_seqlens_q=cu_seqlens_q,  # 每个请求的查询起始位置
    seqused_k=seqused_k,        # 每个请求的 KV 长度
    block_table=block_table,    # 块表
    ...
)

# cu_seqlens_q 示例: [0, 1, 2, 1002]
# Request A: query[0:1], Request B: query[1:2], Request C: query[2:1002]

# seqused_k 示例: [101, 51, 1000]
# Request A: KV 长度 101 (100 历史 + 1 新)
# Request B: KV 长度 51 (50 历史 + 1 新)
# Request C: KV 长度 1000 (全部是新)
```

#### KV Cache 写入

```python
# vllm/v1/attention/backends/flash_attn.py, line 850
def do_kv_cache_update(self, layer, key, value, kv_cache, slot_mapping):
    key_cache, value_cache = kv_cache.unbind(1)

    # 同一个 kernel 处理 prefill 和 decode token
    # slot_mapping 自动处理不同长度
    reshape_and_cache_flash(key, value, key_cache, value_cache, slot_mapping)

# slot_mapping 示例: [slot_A_100, slot_B_50, slot_C_0, slot_C_1, ..., slot_C_999]
# Request A: 写入 1 个位置 (decode)
# Request B: 写入 1 个位置 (decode)
# Request C: 写入 1000 个位置 (prefill)
```

#### 分块预填充与 decode 交错

```
Request C prompt 8192 token, token_budget = 2048

Step 1: [A:decode(1), B:decode(1), C:prefill(2048)]
Step 2: [A:decode(1), B:decode(1), C:prefill(2048)]
Step 3: [A:decode(1), B:decode(1), C:prefill(2048)]
Step 4: [A:decode(1), B:decode(1), C:prefill(2048)]
Step 5: [A:decode(1), B:decode(1), C:decode(1)]

A 和 B 的 decode 从未被阻塞！它们与 C 的 prefill 在同一个 batch 中交错执行。
```

---

## 5. Decode 会用到 Prefill 做的 KV Cache 吗

### 面试回答版

会，每一步都读整个 KV Cache。不管当前步需要算几个 token，Q 都 attend 到整个 KV cache，包含之前所有步存的 K/V。这就是 KV Cache 存在的意义——算过的 K/V 不用重算，直接读。

### 代码分析版

#### 用例子说明

```
Request: prompt [A, B, C, D, E]，生成 [F, G, H]

Step 1: 算 A~E 的 K/V → 写入 KV Cache
  KV Cache = [K_A, K_B, K_C, K_D, K_E]
  采样 → F

Step 2: 算 F 的 K/V → 写入 KV Cache
  KV Cache = [K_A, K_B, K_C, K_D, K_E, K_F]
  Q_F attend 到 [K_A, K_B, K_C, K_D, K_E, K_F]
  ↑ F 的注意力包含了前面所有 token

Step 3: 算 G 的 K/V → 写入 KV Cache
  KV Cache = [K_A, K_B, K_C, K_D, K_E, K_F, K_G]
  Q_G attend 到 [K_A, K_B, K_C, K_D, K_E, K_F, K_G]
  ↑ G 的注意力也包含了前面所有 token
```

#### 代码路径

```python
# 注意力层的 forward
# vllm/model_executor/layers/attention/attention.py, line 480

# 1. 计算新 token 的 K, V
key = self.k_proj(hidden_states)      # [1, hidden_size] → [1, num_kv_heads, head_size]
value = self.v_proj(hidden_states)

# 2. 写入 KV Cache (新 token 的位置)
unified_kv_cache_update(key, value, self.layer_name)
# → reshape_and_cache_flash(key, value, key_cache, value_cache, slot_mapping)
# → key_cache[slot_mapping] = key, value_cache[slot_mapping] = value

# 3. 注意力计算 (读取整个 KV Cache)
output = flash_attn_varlen_func(
    q=query,           # 新 token 的 Q
    k=key_cache,       # 整个 KV Cache (包含 prefill 的 K)
    v=value_cache,     # 整个 KV Cache (包含 prefill 的 V)
    block_table=block_table,  # 用于间接寻址
    ...
)
# FlashAttention 内核通过 block_table 间接寻址:
# 对于请求 r 的 KV 位置 p:
#   k = key_cache[block_table[r, p // block_size], 0, p % block_size, :, :]
#   v = value_cache[block_table[r, p // block_size], 1, p % block_size, :, :]
```

#### 为什么能读到之前算的

KV Cache 是持久化的，存在 GPU 显存的 paged blocks 里：

```
Step 1 算完后:
  Block 0: [K_A, K_B, K_C, K_D]  ← 存在 GPU 显存

Step 2 算 F 时:
  K_F 写入 Block 1 → Block 1: [K_E, K_F, -, -]
  Q_F attend 到 Block 0 + Block 1 的所有 K
  Block 0 的 K_A~K_D 是 Step 1 存的，还在，直接读
```

---

## 6. vLLM 的 Overlap 机制

### 面试回答版

vLLM v1 在多个层面实现了 GPU 计算与 CPU 处理的重叠。核心思想是：GPU 在干活的时候，CPU 也别闲着。具体包括 7 种 overlap：

1. **GPU Forward vs CPU 调度**：GPU 在跑当前 batch 时，CPU 已经在调度下一个 batch。
2. **GPU Forward vs CPU 语法计算**：GPU 跑 forward 时，CPU 在算结构化输出的语法掩码，两者并行。
3. **D2H 拷贝 vs 下一次 Forward**：采样结果从 GPU 拷贝到 CPU 用独立的 CUDA stream，和下一次 forward 重叠。
4. **GPU 端 token 缓存**：异步模式下，采样后的 token 留在 GPU 上直接当下一步输入，省掉 GPU→CPU→GPU 的来回拷贝。
5. **Execute/Sample 分离**：forward 返回 None 延迟采样，中间插入 CPU 的 grammar 计算。
6. **Pipeline 多 batch 流水**：PP 模式下，batch 1 在 stage 1 跑时，batch 2 已经进入 stage 0。
7. **ZMQ IO 独立线程**：网络收发在独立的后台线程，和 GPU 计算完全解耦。

一句话总结：**能并行的都并行了——GPU 跑时 CPU 调度，拷贝时 GPU 计算，网络 IO 独立线程。**

### 代码分析版

#### ① GPU Forward vs CPU 调度

```python
# vllm/v1/engine/core.py, line 467
def step(self):
    # CPU: 调度 (决定哪些请求执行)
    scheduler_output = self.scheduler.schedule()

    # GPU: 启动前向传播 (非阻塞)
    future = self.model_executor.execute_model(scheduler_output, non_block=True)
    # ↑ 立即返回 Future，GPU 开始执行

    # CPU: 在 GPU 执行时，计算语法掩码 (结构化输出)
    grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)
    # ↑ 这个 CPU 操作与 GPU 前向传播重叠！

    # GPU: 等待结果
    model_output = future.result()

    # GPU: 采样
    if model_output is None:
        model_output = self.model_executor.sample_tokens(grammar_output)
```

```
时间线:
CPU:  [Schedule] ─────────────────→ [Grammar] ─────────→ [Wait] → [Update]
GPU:               [Forward Pass] ─────────────────────→ [Done]
                    ↑ GPU 在执行时，CPU 在做其他工作
```

#### ② 异步调度 (AsyncScheduler)

```python
# vllm/v1/core/sched/async_scheduler.py, line 18
class AsyncScheduler(Scheduler):
    def _update_after_schedule(self, request, num_scheduled_token, ...):
        # ★ 乐观地推进 num_computed_tokens
        request.num_computed_tokens += num_scheduled_token

        # ★ 添加输出占位符
        request.num_output_placeholders += 1 + cur_num_spec_tokens
```

```
效果:
Step N:
  GPU: [Forward Batch N]
  CPU: [Schedule Batch N+1]  ← 使用乐观推进的 num_computed_tokens

Step N+1:
  GPU: [Forward Batch N+1]
  CPU: [Schedule Batch N+2]
  CPU: [Update Batch N 结果]  ← 修正乐观推进的误差
```

#### ③ D2H 拷贝与下一次迭代重叠

```python
# vllm/v1/worker/gpu_model_runner.py, line 684
# 创建独立的 CUDA 流用于 D2H 拷贝
if self.use_async_scheduling:
    self.async_output_copy_stream = torch.cuda.Stream()

# vllm/v1/worker/gpu_model_runner.py, line 260 (AsyncGPUModelRunnerOutput)
class AsyncGPUModelRunnerOutput:
    def __init__(self, sampled_token_ids, ...):
        # 切换到异步拷贝流
        with torch.cuda.stream(async_output_copy_stream):
            # 等待默认流完成
            async_output_copy_stream.wait_stream(default_stream)

            # 非阻塞 D2H 拷贝
            self.sampled_token_ids_cpu = sampled_token_ids.to("cpu", non_blocking=True)

            # 记录事件
            self.async_copy_ready_event.record()
```

```
时间线:
Default Stream:  [Forward] → [Sample] → [Next Forward] → ...
Copy Stream:                    [D2H Copy] ────────→ [Done]
                                                ↑ D2H 与 Next Forward 重叠
```

#### ④ GPU 端 token 缓存避免 CPU 同步

```python
# vllm/v1/worker/gpu_model_runner.py, line 3599
# 异步路径：采样后的 token 留在 GPU 上
if self.use_async_scheduling:
    # 不拷贝到 CPU，直接缓存在 GPU
    if self.input_batch.prev_sampled_token_ids is None:
        self.input_batch.prev_sampled_token_ids = sampled_token_ids

# vllm/v1/worker/gpu_model_runner.py, line 1927 (下一步的 _prepare_inputs)
# 直接使用 GPU 上的 token 作为下一步的输入
if self.input_batch.prev_sampled_token_ids is not None:
    input_ids = self.input_batch.prev_sampled_token_ids
    # ↑ 无需 CPU-GPU 拷贝！
```

#### ⑤ Execute/Sample 分离

```python
# vllm/v1/engine/core.py, line 467
def step(self):
    # 1. 启动前向传播 (非阻塞)
    future = self.model_executor.execute_model(scheduler_output, non_block=True)

    # 2. GPU 执行中... CPU 计算语法掩码
    grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)

    # 3. 等待 GPU 完成
    model_output = future.result()  # execute_model 返回 None

    # 4. 采样 (使用语法掩码)
    if model_output is None:
        model_output = self.model_executor.sample_tokens(grammar_output)
        # ↑ 语法掩码计算与 GPU forward 重叠！
```

#### ⑥ Pipeline Parallelism 的 Batch Queue

```python
# vllm/v1/engine/core.py, line 497
def step_with_batch_queue(self):
    # 提交 batch 到队列 (非阻塞)
    future = self.model_executor.execute_model(scheduler_output, non_block=True)
    batch_queue.appendleft((future, scheduler_output, exec_future))

    # 如果队列未满，不等待结果
    if len(batch_queue) < self.batch_queue_size and not batch_queue[-1][0].done():
        return None, True  # 立即返回，调度下一个 batch

    # 队列满了，等待最旧的 batch 完成
    oldest_future, oldest_output, _ = batch_queue.pop()
    model_output = oldest_future.result()
    return model_output
```

```
时间线:
Batch 1: [Schedule] → [Forward PP Stage 0] → [Forward PP Stage 1] → [Done]
Batch 2:              [Schedule] → [Forward PP Stage 0] → [Forward PP Stage 1] → [Done]
Batch 3:                           [Schedule] → [Forward PP Stage 0] → ...
                       ↑ 多个 batch 同时在 pipeline 中
```

#### ⑦ 网络 IO 与 GPU 执行

```python
# vllm/v1/engine/core.py, line 1438
# ZMQ 输入线程 (后台)
def process_input_sockets(self):
    while True:
        request = zmq_socket.recv()  # 阻塞等待
        self.input_queue.put(request)

# ZMQ 输出线程 (后台)
# vllm/v1/engine/core.py, line 1534
def process_output_sockets(self):
    while True:
        output = self.output_queue.get()  # 阻塞等待
        zmq_socket.send(msgpack.pack(output))  # 序列化并发送

# GPU 执行与网络 IO 完全独立，由不同线程处理
```

#### GPU-CPU Overlap 的完整时间线

```
时间 →
─────────────────────────────────────────────────────────────────

CPU:  [Schedule N] ───→ [Grammar N] ───→ [Schedule N+1] ───→ [Grammar N+1]
       │                 │                │                   │
GPU:  ─┤ [Forward N] ────┤────────────────┤ [Forward N+1] ───┤
       │                 │                │                   │
Copy: ─┤─────────────────┤ [D2H N] ───────┤──────────────────┤ [D2H N+1]
       │                 │                │                   │
ZMQ:  ─┤ [IO Thread] ───┤ [IO Thread] ───┤ [IO Thread] ─────┤ [IO Thread]

关键重叠:
1. GPU Forward N 与 CPU Grammar N 重叠
2. GPU Forward N 与 CPU Schedule N+1 重叠 (异步调度)
3. D2H N 与 GPU Forward N+1 重叠
4. ZMQ IO 与 GPU 计算完全独立
```

#### 总结表

| Overlap 机制 | CPU 操作 | GPU 操作 |
|--------------|----------|----------|
| Forward + 调度 | schedule(N+1) | forward(N) |
| Forward + 语法 | grammar(N) | forward(N) |
| D2H + Forward | 下一步输入准备 | sampled_token_ids D2H |
| Token 缓存 | 无 CPU 参与 | GPU 直接复用 |
| Execute/Sample 分离 | grammar 计算 | forward pass |
| Pipeline | schedule | 多 batch 流水 |
| ZMQ IO | 序列化/反序列化 | forward(N) |

---

## 7. 什么时候区分 P 和 D

### 面试回答版

vLLM v1 的调度器和执行层面不区分 P 和 D。P/D 的区分只在两个地方：一是 P/D 分离部署时，P 节点和 D 节点是独立的引擎，各有各的调度器；二是 `is_prefill_chunk` 标记用于观测和丢弃中间 token，但不影响调度逻辑。

### 代码分析版

#### 场景 1：PD 分离部署（两个独立节点）

```
Prefill 节点                     Decode 节点
  Scheduler 只管 prefill           Scheduler 只管 decode
  (处理完 prompt 就结束)            (逐 token 生成)
  kv_role = "kv_producer"         kv_role = "kv_consumer"
        │                               │
        └──── KV 传输 (RDMA/NCCL) ──────┘
```

两个 Scheduler 各自独立运行，互不干扰。

#### 场景 2：PD 混部（同一个节点）

```
单个 Scheduler:
  waiting 队列: [新请求 A, 新请求 B]
  running 队列: [正在生成的 C, 正在生成的 D]

  schedule() 时:
    Phase 1: C, D 各跑 1 个 token (decode)
    Phase 2: A, B 各跑 N 个 token (prefill)
    → 混在一个 batch 里
```

#### 场景 3：`is_prefill_chunk` 标记

```python
# scheduler.py, line 1206
request.is_prefill_chunk = request.num_computed_tokens < (
    request.num_tokens + request.num_output_placeholders
)
```

这个标记不影响调度，只用于丢弃中间 token：

```python
if request.is_prefill_chunk:
    # prompt 还没算完，这个 token 是中间结果，不返回给用户
    pass
else:
    # 这是真正的输出 token
    output_tokens.append(sampled_token)
```

#### 总结表

| 场景 | 是否区分 P/D | 在哪里区分 |
|------|-------------|-----------|
| 调度器内部 | ❌ 不区分 | — |
| GPU 前向传播 | ❌ 不区分 | — |
| PD 分离部署 | ✅ 区分 | 架构层面，不同节点 |
| is_prefill_chunk | ⚠️ 标记但不影响调度 | 用于丢弃中间 token |
| 监控指标 | ⚠️ 标记但不影响执行 | 用于统计延迟 |

---

## 8. D 节点 Scheduler 的调度逻辑与 Overlap

### 面试回答版

D 节点 Scheduler 的特殊之处在于：请求到达时 prompt 的 KV Cache 还没到，需要从 P 节点异步拉取。调度器多了一步 "问连接器远程 KV 有多少"，其余调度逻辑和混部一样。Overlap 主要是：KV 异步拉取与 GPU 计算重叠，调度与执行重叠。

### 代码分析版

#### D 节点的请求处理流程

```
请求到达 D 节点
    │
    ▼
Scheduler.get_num_new_matched_tokens()
    "KV 要从 P 节点拉，报告 1000 个 token 可从远程加载"
    │
    ▼
allocate_slots() → 分配 KV 块，标记 "等待远程 KV"
    │
    ▼
build_connector_meta() → 告诉 Worker "去 P 节点拉 KV"
    │
    ▼
Worker.start_load_kv() → RDMA 异步拉取
    │
    ▼
后续 step: KV 到了 → 正常 decode
```

#### 和混部 Scheduler 的唯一区别

```python
# 混部 Scheduler
for request in waiting:
    num_new_tokens = request.num_tokens - num_computed_tokens
    blocks = allocate_slots(request, num_new_tokens)

# D 节点 Scheduler
for request in waiting:
    # 先问连接器：KV 能从远程拉到吗？
    remote_tokens = connector.get_num_new_matched_tokens(request)
    # remote_tokens = 1000

    num_new_tokens = request.num_tokens - num_computed_tokens - remote_tokens
    blocks = allocate_slots(request, num_new_tokens, num_external=remote_tokens)
```

#### D 节点的 Overlap

**① KV 拉取 vs GPU 计算**
```
GPU:  [其他请求 decode] ──────→ [本请求 decode]
RDMA: [拉 KV ──────────] ──→ 完成
      ↑ 拉取和 GPU 计算重叠
```

**② 调度 vs 执行（和混部一样）**
```
GPU: [Forward N ──────────]
CPU:     [Schedule N+1]     ← GPU 在跑时，CPU 在调度下一批
```

**③ 跨步 Overlap（AsyncScheduler）**
```
当前 batch 在 GPU 上跑时，Scheduler 已经根据乐观推进的
num_computed_tokens 调度下一批。
```

---

## 9. GPU 前向传播做了什么

### 面试回答版

GPU 前向传播就是把 token 变成 logits（每个词的概率分布）。每个 token 经过 N 层 Transformer，每层做 "算 Q/K/V → 写 KV Cache → 注意力计算 → FFN"，最后输出每个词的概率。Prefill 和 Decode 的代码路径完全一样，区别只是 token 数量不同。

### 代码分析版

#### 整体流程

```
输入: token_ids = [15496, 995]  ("Hello world")
        │
        ▼
    ① Embedding: token_id → 向量
       [15496] → [0.1, -0.3, 0.5, ...]
       [995]   → [0.2,  0.1, -0.4, ...]
        │
        ▼
    ② N 层 Transformer 重复执行:
       ┌─────────────────────────────────┐
       │ a. 算 Q, K, V (线性层)           │
       │ b. K, V 写入 KV Cache            │
       │ c. Attention: Q × K^T → score    │
       │ d. softmax(score) × V → output   │
       │ e. 残差连接 + LayerNorm           │
       │ f. FFN: W2 × GELU(W1 × hidden)  │
       │ g. 残差连接 + LayerNorm           │
       └─────────────────────────────────┘
       × 32 层 (或更多)
        │
        ▼
    ③ logits = [0.1, 0.5, -0.2, ...] (50000 维)
       每个维度对应词表中一个词的概率分数
```

#### 每一层的代码

```python
class TransformerLayer:
    def forward(self, hidden_states, kv_cache, slot_mapping, block_table):
        # ① 算 Q, K, V
        Q = self.W_q(hidden_states)
        K = self.W_k(hidden_states)
        V = self.W_v(hidden_states)

        # ② 写 KV Cache
        key_cache[slot_mapping] = K
        value_cache[slot_mapping] = V

        # ③ 注意力 (读整个 KV Cache)
        attn_out = flash_attention(Q, key_cache, value_cache, block_table)

        # ④ 残差 + Norm
        hidden = layer_norm(hidden_states + attn_out)

        # ⑤ FFN
        ffn_out = self.W2(gelu(self.W1(hidden)))

        # ⑥ 残差 + Norm
        hidden = layer_norm(hidden + ffn_out)

        return hidden
```

#### Prefill vs Decode 的区别

```
Prefill: num_tokens = prompt_length (几百到几千)
  → 并行度高，计算密集

Decode: num_tokens = 1 (或 1 + spec_tokens)
  → 并行度低，访存密集

但代码路径完全一样！区别只是 num_tokens 不同。
```

---

## 10. PagedAttention 中多序列间的内存共享（Beam Search 场景）

### 面试回答版

vLLM 通过 **引用计数 (ref_cnt)** 实现多序列间的 KV Cache 内存共享。每个 KV Cache 块有一个 `ref_cnt` 字段，当多个请求共享相同前缀时，它们指向同一个物理块，`ref_cnt` 递增。只有当 `ref_cnt` 降为 0 时，块才被释放回空闲池。

在 Beam Search 场景中，vLLM v1 在应用层实现 beam search，每个 beam 候选作为独立请求提交。所有 beam 候选共享相同的 prompt 前缀，通过前缀缓存机制自然地共享物理块。当 beam 在位置 t 分叉时，`[0, t)` 的块被所有 beam 共享（`ref_cnt = beam_width`），分叉后的部分各自分配新块。

### 代码分析版

#### 引用计数机制

```python
# vllm/v1/core/kv_cache_utils.py, line 116
@dataclass(slots=True)
class KVCacheBlock:
    block_id: int          # 物理块 ID
    ref_cnt: int = 0       # 引用计数：多少个请求在使用这个块
    _block_hash: ...       # 内容哈希 (用于前缀缓存)
```

#### touch()：共享时增加引用

```python
# vllm/v1/core/block_pool.py, line 402
def touch(self, blocks: Sequence[KVCacheBlock]) -> None:
    for block in blocks:
        if block.ref_cnt == 0 and not block.is_null:
            # 块在空闲队列中，移除它（防止被驱逐）
            self.free_block_queue.remove(block)
        # 引用计数 +1
        block.ref_cnt += 1
```

#### free_blocks()：释放时减少引用

```python
# vllm/v1/core/block_pool.py, line 419
def free_blocks(self, ordered_blocks: Iterable[KVCacheBlock]) -> None:
    for block in ordered_blocks:
        block.ref_cnt -= 1
    # 只有 ref_cnt == 0 的块才回到空闲队列
    self.free_block_queue.append_n(
        [block for block in blocks if block.ref_cnt == 0 and not block.is_null]
    )
```

#### Beam Search 的内存共享流程

```
Beam Search 场景: beam_width=3, prompt 100 token

Step 1: 所有 beam 共享 prompt 前缀
  Beam 0: [Block 0, Block 1, Block 2, Block 3]  ← ref_cnt=3
  Beam 1: [Block 0, Block 1, Block 2, Block 3]  ← 同样的物理块
  Beam 2: [Block 0, Block 1, Block 2, Block 3]  ← 同样的物理块
  Block 0~3 的 ref_cnt = 3 (3 个 beam 共享)

Step 2: beam 在 token 100 处分叉
  Beam 0: [Block 0~3, Block 4]  ← Block 4 是新的
  Beam 1: [Block 0~3, Block 5]  ← Block 5 是新的
  Beam 2: [Block 0~3, Block 6]  ← Block 6 是新的
  Block 0~3 的 ref_cnt = 3 (仍然共享)
  Block 4~6 的 ref_cnt = 1 (各自独立)

Step 3: Beam 2 被剪枝
  Beam 2 释放 Block 6 → ref_cnt=0 → 回到空闲池
  Block 0~3 的 ref_cnt = 2 (Beam 0 和 Beam 1 仍然共享)

Step 4: Beam 0 和 Beam 1 完成
  Beam 0 释放 Block 4 → ref_cnt=0 → 回到空闲池
  Beam 1 释放 Block 5 → ref_cnt=0 → 回到空闲池
  Block 0~3 的 ref_cnt = 0 → 回到空闲池
```

#### vLLM v1 的 Beam Search 实现

```python
# vllm/v1/engine/llm_engine.py, line 270
# 当 SamplingParams.n > 1 时，创建 n 个子请求
parent_req = ParentRequest(request)
for idx in range(n):
    request_id, child_params = parent_req.get_child_info(idx)
    child_request = request if idx == n - 1 else copy(request)
    child_request.request_id = request_id
    child_request.sampling_params = child_params
    # 每个子请求有相同的 prompt token IDs 和 block hashes
    # 前缀缓存机制自然地让它们共享物理块
```

**关键点：** vLLM v1 没有显式的 "fork" 或 "clone" 机制。内存共享完全通过前缀缓存隐式实现：多个请求有相同 token 前缀时，`get_computed_blocks()` 会找到相同的缓存块，`touch()` 增加引用计数。

---

## 11. Scheduler 调度器的核心职责和工作流程

### 面试回答版

Scheduler 是 vLLM v1 的核心组件，负责协调 Continuous Batching 和 PagedAttention。其核心职责有三个：**请求管理**（维护 waiting 和 running 队列）、**资源分配**（为请求分配 KV Cache 块）、**批处理构建**（决定每步执行哪些请求的多少个 token）。

工作流程是每步调用 `schedule()` 方法：Phase 1 调度 RUNNING 请求（通常是 decode），Phase 2 调度 WAITING 请求（通常是 prefill），两个 Phase 共享同一个 token budget。调度器不区分 prefill 和 decode，只看每个请求的 `num_computed_tokens` 和 `num_tokens` 的差距。通过统一的 token 级调度，自然地实现了 Continuous Batching（请求完成立即移除，新请求立即加入）和 PagedAttention（通过 KVCacheManager 分配和释放块）。

### 代码分析版

#### 核心职责

```python
# vllm/v1/core/sched/scheduler.py
class Scheduler:
    # 1. 请求管理
    waiting: RequestQueue          # 等待队列 (新请求)
    running: list[Request]         # 运行队列 (正在执行的请求)
    skipped_waiting: RequestQueue  # 被跳过的等待请求
    requests: dict[str, Request]   # 所有活跃请求的注册表

    # 2. 资源分配
    kv_cache_manager: KVCacheManager   # KV Cache 块管理
    encoder_cache_manager: ...         # 编码器缓存管理

    # 3. 配置
    max_num_scheduled_tokens: int      # 每步最大 token 数
    max_num_running_reqs: int          # 最大并发请求数
    enable_chunked_prefill: bool       # 是否启用分块预填充
```

#### 工作流程

```python
def schedule(self) -> SchedulerOutput:
    token_budget = self.max_num_scheduled_tokens

    # Phase 1: 调度 RUNNING 请求
    for request in self.running:
        num_new_tokens = num_tokens_with_spec - num_computed_tokens
        blocks = kv_cache_manager.allocate_slots(request, num_new_tokens)
        if blocks is None:
            preempt(lowest_priority_request)  # 内存不足，预抢占
        token_budget -= num_new_tokens

    # Phase 2: 调度 WAITING 请求 (前提: 没有预抢占)
    if not preempted_reqs:
        for request in self.waiting:
            computed_blocks = kv_cache_manager.get_computed_blocks(request)
            num_new_tokens = request.num_tokens - num_computed
            if enable_chunked_prefill:
                num_new_tokens = min(num_new_tokens, token_budget)
            blocks = kv_cache_manager.allocate_slots(request, num_new_tokens)
            if blocks is None:
                break  # 内存不足，停止
            request.status = RUNNING
            self.running.append(request)
            token_budget -= num_new_tokens

    return SchedulerOutput(...)
```

#### 如何协调 Continuous Batching

```
Continuous Batching 的实现:

1. 请求完成 → update_from_output() 检测停止条件
   → 从 running 队列移除
   → kv_cache_manager.free() 释放块

2. 新请求到达 → add_request() 放入 waiting 队列

3. 每步 schedule() 重新构建 batch:
   - Phase 1: 处理 running 中的请求
   - Phase 2: 从 waiting 中取出新请求加入 running
   - 两个 Phase 共享 token_budget

结果: batch 内容每步都变，请求完成立即释放，新请求立即加入
```

#### 如何协调 PagedAttention

```
PagedAttention 的协调:

1. 调度时: kv_cache_manager.allocate_slots() 分配 KV 块
   - 查前缀缓存 → 复用已有块
   - 分配新块 → 从空闲池弹出

2. 执行时: Worker 通过 slot_mapping 写入 KV Cache
   - slot_mapping 由 block_table 计算

3. 完成时: kv_cache_manager.free() 释放块
   - ref_cnt 递减，归零后回到空闲池

4. 预抢占时: kv_cache_manager.free() 释放所有块
   - 请求放回 waiting 队列头部
```

---

## 12. Block Manager 的作用和工作原理

### 面试回答版

在 vLLM v1 中，Block Manager 的功能由 **KVCacheManager** + **BlockPool** 实现。KVCacheManager 是调度器和块管理之间的接口，负责分配、释放、缓存 KV 块。BlockPool 是底层的块池，管理所有物理块的分配和驱逐。

内存分配流程：当请求需要新块时，`allocate_slots()` 先释放滑动窗口外的块，然后计算需要的新块数，从 BlockPool 的空闲队列中弹出 LRU 块（`get_new_blocks()`），如果块有缓存哈希则先驱逐。分配成功后，新块的 `ref_cnt` 设为 1。当请求完成或被预抢占时，`free()` 递减 `ref_cnt`，归零后块回到空闲队列尾部（MRU 端）。

### 代码分析版

#### 架构

```
Scheduler
    │
    ▼
KVCacheManager (调度器接口)
    │  - allocate_slots(): 分配 KV 块
    │  - free(): 释放块
    │  - get_computed_blocks(): 查前缀缓存
    │
    ▼
KVCacheCoordinator (多缓存组协调)
    │  - UnitaryKVCacheCoordinator: 单缓存组
    │  - HybridKVCacheCoordinator: 混合缓存组
    │
    ▼
SingleTypeKVCacheManager (每种注意力类型一个)
    │  - FullAttentionManager
    │  - SlidingWindowManager
    │  - MambaManager
    │
    ▼
BlockPool (底层块池)
    │  - blocks: 所有物理块
    │  - free_block_queue: 空闲块双向链表
    │  - cached_block_hash_to_block: 前缀缓存哈希表
```

#### 分配流程

```python
# KVCacheManager.allocate_slots()
def allocate_slots(request, num_new_tokens, ...):
    # 1. 释放滑动窗口外的块
    coordinator.remove_skipped_blocks(request_id, total_computed_tokens)

    # 2. 计算需要的新块数
    num_blocks = coordinator.get_num_blocks_to_allocate(...)

    # 3. 容量检查
    if num_blocks > block_pool.get_num_free_blocks():
        return None  # 内存不足

    # 4. 附加前缀缓存命中的块 (touch 增加 ref_cnt)
    coordinator.allocate_new_computed_blocks(...)

    # 5. 从空闲池分配新块
    new_blocks = coordinator.allocate_new_blocks(...)

    # 6. 缓存新填满的块 (供未来前缀缓存命中)
    coordinator.cache_blocks(request, num_tokens_to_cache)

    return new_blocks
```

#### BlockPool 的块分配

```python
# BlockPool.get_new_blocks()
def get_new_blocks(self, num_blocks):
    # 从空闲队列头部弹出 LRU 块
    blocks = free_block_queue.popleft_n(num_blocks)

    for block in blocks:
        # 如果块有缓存哈希，驱逐它
        _maybe_evict_cached_block(block)
        # 设置 ref_cnt = 1
        block.ref_cnt = 1

    return blocks
```

#### 释放流程

```python
# SingleTypeKVCacheManager.free()
def free(self, request_id):
    req_blocks = req_to_blocks.pop(request_id, [])
    # 按逆序释放 (尾部块先释放，保留前缀)
    for block in reversed(req_blocks):
        block.ref_cnt -= 1
        if block.ref_cnt == 0:
            # 放回空闲队列尾部 (MRU 端)
            block_pool.free_block_queue.append(block)
```

---

## 13. 请求抢占机制

### 面试回答版

vLLM v1 的请求抢占发生在 **Phase 1 调度 RUNNING 请求时**，当 `allocate_slots()` 返回 `None`（KV Cache 内存不足）时触发。抢占策略：FCFS 模式下抢占最后加入 running 的请求，Priority 模式下抢占优先级最低的请求。

被抢占的请求：释放所有 KV 块、`num_computed_tokens` 重置为 0、状态设为 PREEMPTED、放回 waiting 队列头部。恢复时通过前缀缓存可能部分恢复已计算的 token。**Phase 2 不会触发抢占**——如果 waiting 请求分配失败，调度直接停止。

### 代码分析版

#### 触发条件

```python
# scheduler.py, Phase 1 (调度 RUNNING 请求)
while True:
    new_blocks = self.kv_cache_manager.allocate_slots(request, num_new_tokens)
    if new_blocks is not None:
        break  # 分配成功

    # 分配失败 → 触发抢占
    if self.policy == SchedulingPolicy.PRIORITY:
        # Priority 模式: 抢占优先级最低且到达最晚的请求
        preempted_req = max(self.running,
            key=lambda r: (r.priority, r.arrival_time))
    else:
        # FCFS 模式: 抢占最后加入的请求
        preempted_req = self.running.pop()

    self._preempt_request(preempted_req)
```

#### 抢占执行

```python
# scheduler.py, line 1151
def _preempt_request(self, request, timestamp):
    # 1. 释放所有 KV 块
    self.kv_cache_manager.free(request)
    self.encoder_cache_manager.free(request)

    # 2. 状态设为 PREEMPTED
    request.status = RequestStatus.PREEMPTED

    # 3. 重置已计算 token 数
    request.num_computed_tokens = 0

    # 4. 清除推测 token
    request.spec_token_ids = []

    # 5. 记录抢占次数
    request.num_preemptions += 1

    # 6. 放回 waiting 队列头部 (优先重新调度)
    self.waiting.prepend_request(request)
```

#### 为什么 Phase 2 不抢占

```
Phase 1 (RUNNING): 分配失败 → 抢占 → 重试
Phase 2 (WAITING): 分配失败 → break (停止调度)

原因: 如果 Phase 2 也抢占，可能导致:
  1. 刚调度的 WAITING 请求抢占 RUNNING 请求
  2. RUNNING 请求放回 WAITING 队列头部
  3. 下一步又抢占回来 → 死循环 (thrashing)

所以 Phase 2 只是停止调度，不抢占。
```

---

## 14. Continuous Batching 中请求的动态加入与退出

### 面试回答版

vLLM v1 通过每步重新调用 `schedule()` 实现请求的动态加入与退出。**加入**：新请求通过 `add_request()` 放入 waiting 队列，Phase 2 调度时分配 KV 块后移入 running。**退出**：`update_from_output()` 检测停止条件（EOS、max_tokens、stop strings），将完成的请求从 running 移除并释放 KV 块。

核心机制：token budget 共享（Phase 1 和 Phase 2 共享同一个 budget）、请求状态机（WAITING → RUNNING → FINISHED/PREEMPTED）、每步重新调度（batch 内容每步都变）。

### 代码分析版

#### 动态加入

```python
# 1. 新请求到达
def add_request(self, request):
    self._enqueue_waiting_request(request)  # 放入 waiting 队列
    self.requests[request.request_id] = request

# 2. 调度时加入 running
def schedule(self):
    # Phase 2: 从 waiting 取出请求
    for request in self.waiting:
        blocks = kv_cache_manager.allocate_slots(request, num_new_tokens)
        if blocks is not None:
            request = request_queue.pop_request()  # 从 waiting 移除
            self.running.append(request)            # 加入 running
            request.status = RequestStatus.RUNNING
```

#### 动态退出

```python
# 1. 检测停止条件
def update_from_output(self, scheduler_output, model_output):
    for request in scheduled_requests:
        # 追加采样 token
        request.append_output_token(sampled_token)

        # 检查停止条件
        stopped, stop_reason = check_stop(request, self.max_model_len)
        if stopped:
            request.status = FINISHED_STOPPED / FINISHED_LENGTH_CAPPED / ...
            stopped_running_reqs.append(request)

# 2. 从 running 移除
self.running = remove_all(self.running, stopped_running_reqs)

# 3. 释放资源
for request in stopped_requests:
    kv_cache_manager.free(request)
    del self.requests[request.request_id]
```

#### 请求状态机

```
WAITING ──→ RUNNING ──→ FINISHED_STOPPED / FINISHED_LENGTH_CAPPED / ...
   ↑            |
   |            v
   +── PREEMPTED

状态转换:
  WAITING → RUNNING:    schedule() Phase 2 分配成功
  RUNNING → FINISHED:   update_from_output() 检测到停止条件
  RUNNING → PREEMPTED:  schedule() Phase 1 分配失败
  PREEMPTED → RUNNING:  schedule() Phase 2 恢复成功
```

---

## 15. FlashAttention 的在线 Softmax 数值稳定性

### 面试回答版

FlashAttention 通过 **在线 Softmax (Online Softmax)** 算法解决 Tiling 过程中的数值稳定性问题。核心思想：维护一个 **运行最大值 M** 和 **运行和 L**，每处理一个 KV 块（tile），先更新最大值，然后用 `exp(old_max - new_max)` 对之前的累积结果重缩放，防止指数溢出。

更新公式：
```
m_j = max(M, max(S_j))           # 新的最大值
α = exp(M - m_j)                  # 重缩放因子
L = L * α + sum(exp(S_j - m_j))   # 更新运行和
acc = acc * α + exp(S_j - m_j) @ V_j  # 更新累积输出
M = m_j                           # 更新最大值
最终输出: acc / L
```

这种设计使得每个 tile 独立计算，通过 M 和 L 合并结果，无需物化完整的 `[seq_len, seq_len]` 注意力矩阵。

### 代码分析版

#### Triton 内核中的在线 Softmax

```python
# vllm/v1/attention/ops/chunked_prefill_paged_decode.py, line 126

# 初始化运行最大值和运行和
M = tl.full([num_queries], float("-inf"), dtype=tl.float32)  # 运行最大值
L = tl.zeros([num_queries], dtype=tl.float32)                # 运行和
acc = tl.zeros([num_queries, HEAD_SIZE], dtype=tl.float32)   # 累积输出

# 遍历 KV 块 (tiles)
for j in range(0, num_blocks):
    # 加载 K, V 块，计算 QK 点积
    S = tl.where(head_mask & seq_mask, qk, float("-inf"))

    # 1. 更新最大值
    m_j = tl.maximum(M, tl.max(S, axis=1))

    # 2. 计算 softmax (减去新最大值，数值稳定)
    p = tl.exp(S - m_j[:, None])

    # 3. 计算当前块的 softmax 和
    l_j = tl.sum(p, axis=1)

    # 4. 重缩放因子: 旧最大值和新最大值的差异
    alpha = tl.exp(M - m_j)
    alpha = tl.where(float("-inf") == M, 0.0, alpha)

    # 5. 重缩放累积结果
    acc = acc * alpha[:, None]    # 累积输出重缩放
    L = L * alpha + l_j           # 运行和重缩放 + 新贡献

    # 6. 更新最大值
    M = m_j

    # 7. 累加加权 V
    acc += tl.dot(p.to(V.dtype), V)

# 最终归一化
acc = acc / (L[:, None] + 1e-10)
```

#### 为什么需要在线 Softmax

```
问题: softmax(x) = exp(x_i) / sum(exp(x_j))

如果 x 的值很大 (如 100)，exp(100) 会溢出 (inf)
如果 x 的值很小 (如 -100)，exp(-100) 会下溢 (0)

传统做法: 先减去最大值 max(x)
  softmax(x) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
  这保证 exp 的参数 <= 0，不会溢出

FlashAttention 的问题: 数据分成了多个 tile，
  每个 tile 只看到部分数据，不知道全局最大值

在线 Softmax 的解决: 维护运行最大值 M
  每处理一个 tile:
    1. 计算新最大值 m_j = max(M, max(S_j))
    2. 用 α = exp(M - m_j) 重缩放之前的累积结果
    3. 更新 M = m_j
  这样每个 tile 都能正确归一化，无需知道后续 tile 的数据
```

#### 跨内核调用的合并 (Cascade Attention)

```python
# vllm/v1/attention/ops/triton_merge_attn_states.py, line 118
# 合并两个部分注意力结果 (前缀 + 后缀)

# 加载两个部分的 log-sum-exp (LSE)
p_lse = tl.load(prefix_lse)   # 前缀的 LSE
s_lse = tl.load(suffix_lse)   # 后缀的 LSE

# 找到全局最大值
max_lse = tl.maximum(p_lse, s_lse)

# 重缩放
p_scale = tl.exp(p_lse - max_lse)
s_scale = tl.exp(s_lse - max_lse)

# 加权合并
out = (p_out * p_scale + s_out * s_scale) / (p_scale + s_scale)
```

---

## 16. 线上推理延迟升高的排查和决策流程

### 面试回答版

当线上推理延迟突然升高，我会按以下流程排查：

1. **监控指标检查**：查看 GPU 利用率、KV Cache 使用率、请求队列长度、批处理大小
2. **瓶颈定位**：
   - GPU 利用率低 + 队列长 → CPU 瓶颈（调度开销、D2H 拷贝）
   - GPU 利用率高 + 队列长 → GPU 算力不足（batch 太大或模型太大）
   - KV Cache 使用率高 → 内存瓶颈（可能触发预抢占）
3. **是否需要调整 batch 大小**：
   - 如果 KV Cache 使用率 > 90% 且频繁预抢占 → 减小 `max_num_batched_tokens`
   - 如果 GPU 利用率低且队列短 → 增大 batch 可能无帮助
   - 如果 GPU 利用率高且延迟高 → 考虑增大 batch（如果内存允许）

### 代码分析版

#### 关键监控指标

```python
# KV Cache 使用率
usage = kv_cache_manager.block_pool.get_usage()
# = 1.0 - (free_blocks / (total_blocks - 1))

# 请求队列长度
waiting_count = len(scheduler.waiting)
running_count = len(scheduler.running)

# 每步 token 数
total_tokens = scheduler_output.total_num_scheduled_tokens

# 预抢占次数
preempted_count = len(scheduler_output.preempted_req_ids)
```

#### 排查决策树

```
延迟升高
    │
    ├── KV Cache 使用率 > 90%?
    │   ├── 是 → 内存瓶颈
    │   │   ├── 频繁预抢占? → 减小 max_num_batched_tokens
    │   │   └── 无预抢占? → 增大 gpu_memory_utilization 或使用量化
    │   │
    │   └── 否 → 继续排查
    │
    ├── GPU 利用率?
    │   ├── 低 (< 50%) → CPU 瓶颈
    │   │   ├── 调度开销大? → 检查 scheduler 复杂度
    │   │   ├── D2H 拷贝慢? → 检查异步调度是否启用
    │   │   └── batch 太小? → 增大 max_num_batched_tokens
    │   │
    │   ├── 高 (> 90%) → GPU 算力瓶颈
    │   │   ├── batch 太大? → 减小 max_num_batched_tokens
    │   │   ├── 模型太大? → 考虑量化或 TP
    │   │   └── 长 prefill 阻塞? → 启用 chunked prefill
    │   │
    │   └── 中等 → 检查是否有异常请求
    │
    └── 是否有异常请求?
        ├── 超长请求占用大量 KV 块? → 设置 max_model_len
        └── 请求积压? → 增大 max_num_running_reqs
```

---

## 17. Decode 阶段的计算特点及与 Prefill 的本质区别

### 面试回答版

**Decode 阶段的计算特点：**
- 每步只处理 1 个 token（或 1 + spec_tokens）
- Q 只有 1 行，K/V 有 S 行（S 是序列长度）
- 计算量小（O(S×d)），但需要读取整个 KV Cache
- **访存密集 (memory-bound)**：瓶颈是从 GPU 显存读取 KV Cache

**Prefill 阶段的计算特点：**
- 处理整个 prompt（可能几百到几千 token）
- Q、K、V 都是 L 行（L 是 prompt 长度）
- 计算量大（O(L²×d)），矩阵乘法维度大
- **计算密集 (compute-bound)**：瓶颈是 GPU 算力

**本质区别：**

| 维度 | Prefill | Decode |
|------|---------|--------|
| query 长度 | L (几百~几千) | 1 |
| 计算量 | O(L² × d) | O(S × d) |
| 瓶颈 | GPU 算力 | 显存带宽 |
| 矩阵乘法 | 大矩阵 × 大矩阵 | 矩阵 × 向量 |
| CUDA Graph | 难以使用 (长度不一) | 最佳 (统一 decode) |
| 批处理效率 | 高 (大矩阵) | 低 (小向量) |

### 代码分析版

#### vLLM v1 不区分 Prefill 和 Decode

```python
# scheduler.py, line 438
# NOTE: There's no "decoding phase" nor "prefill phase" in the scheduler.
# Each request just has num_computed_tokens and num_tokens_with_spec.

# 调度器只看差距:
num_new_tokens = num_tokens_with_spec - num_computed_tokens
# Prefill: num_new_tokens = prompt_length (几百~几千)
# Decode:  num_new_tokens = 1 (或 1 + spec_tokens)
```

#### 注意力计算的维度差异

```python
# FlashAttention 调用
flash_attn_varlen_func(
    q=query,           # Prefill: [L, h, d]  Decode: [1, h, d]
    k=key_cache,       # [S, h, d] (整个 KV Cache)
    v=value_cache,     # [S, h, d]
    cu_seqlens_q=...,  # Prefill: [0, L]  Decode: [0, 1]
    ...
)

# Prefill: Q×K^T = [L, d] × [d, S] → [L, S]  (大矩阵乘法)
# Decode:  Q×K^T = [1, d] × [d, S] → [1, S]  (矩阵×向量)
```

#### 为什么 Decode 是访存密集

```
Prefill (L=1024, d=4096, S=1024):
  Q×K^T: [1024, 4096] × [4096, 1024] → [1024, 1024]
  FLOPs: 1024 × 4096 × 1024 × 2 ≈ 8.6G
  数据读取: 1024 × 4096 × 2 (Q+K) ≈ 32MB
  算术强度: 8.6G / 32MB ≈ 269 FLOPs/byte  ← 计算密集

Decode (L=1, d=4096, S=1024):
  Q×K^T: [1, 4096] × [4096, 1024] → [1, 1024]
  FLOPs: 1 × 4096 × 1024 × 2 ≈ 8.4M
  数据读取: 1024 × 4096 × 2 (Q+KV Cache) ≈ 32MB
  算术强度: 8.4M / 32MB ≈ 0.26 FLOPs/byte  ← 访存密集

Decode 的算术强度比 Prefill 低 1000 倍！
GPU 算力没用满，瓶颈在读取 KV Cache 的带宽。
```

#### vLLM 对 Decode 的优化

```python
# 1. 统一 Decode 的 CUDA Graph 优化
# gpu_model_runner.py, line 3709
def _is_uniform_decode(...):
    # 所有请求都有相同的 query 长度 (1 或 1+spec)
    return (max_num_scheduled_tokens == uniform_decode_query_len
            and num_tokens == max_num_scheduled_tokens * num_reqs)

# 统一 decode batch 可以使用 FULL CUDA Graph
# 消除 CPU 开销，直接回放 GPU 计算图

# 2. PagedAttention 减少内存浪费
# 按需分配块，不需要预分配连续内存

# 3. 前缀缓存减少重复计算
# 共享前缀的请求复用 KV Cache

# 4. 推测解码增加每步 token 数
# 每步生成 >1 个 token，提高 GPU 利用率
```
