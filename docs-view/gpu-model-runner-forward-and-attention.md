# GPU Model Runner 前向传播与 Attention 后端调用链路详解

## 概述

本文档详细描述 `gpu_model_runner.py` 中 `_model_forward()` 之后的完整调用逻辑，以及模型如何通过全局 `ForwardContext` 调用 Attention 后端。

---

## 完整调用链路总览

```
execute_model()
  └─ _model_forward()
       └─ self.model(input_ids, positions, ...)  # 例如 LlamaModel
            └─ for layer in self.layers:          # 遍历每个 DecoderLayer
                 └─ layer(positions, hidden_states, residual)
                      └─ self.self_attn(positions, hidden_states)  # LlamaAttention
                           └─ self.attn(q, k, v)                  # Attention.forward()
                                └─ unified_attention_with_output()
                                     └─ get_attention_context()    # 从 ForwardContext 获取 attn_metadata
                                     └─ self.impl.forward(...)     # 调用具体后端 (FlashInfer/FlashAttention 等)
```

---

## 1. `_model_forward()` 做了什么

`gpu_model_runner.py:3700` 非常简单，就是直接调用 `self.model`：

```python
def _model_forward(self, input_ids, positions, intermediate_tensors, inputs_embeds, **model_kwargs):
    return self.model(
        input_ids=input_ids,
        positions=positions,
        intermediate_tensors=intermediate_tensors,
        inputs_embeds=inputs_embeds,
        **model_kwargs,
    )
```

`self.model` 是具体模型的实例（如 `LlamaModel`）。调用后进入模型的 `forward()` 方法。

---

## 2. 模型内部的调用流程（以 Llama 为例）

### 2.1 `LlamaModel.forward()` (`llama.py:395`)

```python
def forward(self, input_ids, positions, intermediate_tensors, inputs_embeds=None, **extra_layer_kwargs):
    # 1. 嵌入层：将 input_ids 转换为 hidden_states
    if get_pp_group().is_first_rank:
        hidden_states = self.embed_input_ids(input_ids)  # 或 inputs_embeds
    else:
        hidden_states = intermediate_tensors["hidden_states"]  # PP 中间张量

    # 2. 逐层遍历所有 DecoderLayer
    for idx, layer in enumerate(islice(self.layers, self.start_layer, self.end_layer)):
        hidden_states, residual = layer(positions, hidden_states, residual, **extra_layer_kwargs)

    # 3. 最终 LayerNorm
    hidden_states, _ = self.norm(hidden_states, residual)
    return hidden_states
```

### 2.2 `LlamaDecoderLayer.forward()` (`llama.py:316`)

```python
def forward(self, positions, hidden_states, residual):
    # 1. Self Attention
    hidden_states, residual = self.input_layernorm(hidden_states, residual)
    hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)

    # 2. Feed-Forward Network
    hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
    hidden_states = self.mlp(hidden_states)
    return hidden_states, residual
```

### 2.3 `LlamaAttention.forward()` (`llama.py:223`)

```python
def forward(self, positions, hidden_states):
    # 1. QKV 投影：将 hidden_states 投影到 Q, K, V
    qkv, _ = self.qkv_proj(hidden_states)
    q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

    # 2. 旋转位置编码 (RoPE)
    q, k = self.rotary_emb(positions, q, k)

    # 3. 调用 Attention 层（核心！）
    attn_output = self.attn(q, k, v)

    # 4. 输出投影
    output, _ = self.o_proj(attn_output)
    return output
```

---

## 3. Attention 层如何调用后端（核心机制）

### 3.1 `Attention.forward()` (`attention.py:437`)

这是最关键的一步。Attention 层的 `forward()` 方法并不直接实现注意力计算，而是通过 **全局 ForwardContext** 获取注意力元数据，然后委托给具体的后端实现。

```python
def forward(self, query, key, value, output_shape=None):
    # 1. 重塑张量形状
    query = query.view(-1, self.num_heads, self.head_size)
    key = key.view(-1, self.num_kv_heads, self.head_size)
    value = value.view(-1, self.num_kv_heads, self.head_size_v)
    output = output.view(-1, self.num_heads, self.head_size_v)

    # 2. KV Cache 更新（如果后端不自动包含）
    if not self.attn_backend.forward_includes_kv_cache_update and self.kv_sharing_target_layer_name is None:
        kv_cache_dummy_dep = unified_kv_cache_update(key, value, self.layer_name)

    # 3. 调用统一的 attention 算子（注册为 torch custom op）
    unified_attention_with_output(query, key, value, output, self.layer_name, ...)

    return output.view(-1, hidden_size)
```

### 3.2 `unified_attention_with_output()` (`attention.py:734`)

这是一个注册为 `torch.ops.vllm.unified_attention_with_output` 的自定义算子，它从全局上下文中获取注意力元数据，然后调用后端的 `forward()`：

```python
def unified_attention_with_output(query, key, value, output, layer_name, ...):
    # 1. 从全局 ForwardContext 获取注意力元数据
    attn_metadata, self, kv_cache, _ = get_attention_context(layer_name)

    # 2. 调用具体后端的 forward 实现
    self.impl.forward(
        self, query, key, value, kv_cache, attn_metadata, output=output, ...
    )
```

### 3.3 `get_attention_context()` (`attention.py:660`)

这个函数从全局的 `ForwardContext` 中提取当前层需要的注意力元数据：

```python
def get_attention_context(layer_name):
    # 1. 获取全局 ForwardContext（由 set_forward_context 设置）
    forward_context = get_forward_context()

    # 2. 从 attn_metadata 中按层名查找对应的元数据
    attn_metadata_raw = forward_context.attn_metadata
    if isinstance(attn_metadata_raw, dict):
        attn_metadata = attn_metadata_raw[layer_name]        # 普通模式：dict[layer_name, metadata]
    elif isinstance(attn_metadata_raw, list):
        attn_metadata = attn_metadata_raw[0][layer_name]     # 推测解码模式：list[dict]

    # 3. 获取 Attention 层实例和 KV Cache
    attn_layer = forward_context.no_compile_layers[layer_name]
    kv_cache = attn_layer.kv_cache
    slot_mapping = forward_context.slot_mapping[layer_name]

    return attn_metadata, attn_layer, kv_cache, slot_mapping
```

---

## 4. `set_forward_context` 如何建立这个桥梁

回到 `gpu_model_runner.py:4291`，`set_forward_context` 是一个 context manager，它在模型 forward 之前设置好全局上下文：

```python
# gpu_model_runner.py:4291
with set_forward_context(attn_metadata, self.vllm_config, ...):
    model_output = self._model_forward(input_ids=input_ids, positions=positions, ...)
```

`set_forward_context` (`forward_context.py:250`) 做的事情：

1. **创建 `ForwardContext` 对象**，包含：
   - `attn_metadata`：由 `_build_attention_metadata()` 构建的注意力元数据字典
   - `slot_mapping`：slot 映射
   - `no_compile_layers`：所有 Attention 层的引用
   - `cudagraph_runtime_mode`：CUDA Graph 模式
   - `batch_descriptor`：批描述符

2. **通过 `override_forward_context()` 设置到全局线程局部变量**，这样模型内部的任何层都可以通过 `get_forward_context()` 访问。

3. **在 `with` 块退出时清理上下文**。

---

## 5. 注意力元数据的构建 (`_build_attention_metadata`)

`gpu_model_runner.py:2187` 中的 `_build_attention_metadata()` 构建了后端所需的所有元数据：

```python
def _build_attention_metadata(self, num_tokens, num_reqs, max_query_len, ...):
    # 1. 构建 CommonAttentionMetadata（公共元数据）
    cm_base = CommonAttentionMetadata(
        query_start_loc=...,      # 每个请求的查询起始位置
        seq_lens=...,             # 每个请求的 KV 序列长度
        block_table_tensor=...,   # 逻辑块 → 物理块映射表
        slot_mapping=...,         # 每个 token 的 KV Cache 写入位置
        max_seq_len=...,          # 最大序列长度
        positions=...,            # 位置编码
        is_prefilling=...,        # 是否在 prefill 阶段
    )

    # 2. 对每个 KV Cache 组，调用对应的 MetadataBuilder 构建后端特定的元数据
    for kv_cache_gid, attn_groups in enumerate(self.attn_groups):
        for attn_gid, attn_group in enumerate(attn_groups):
            builder = attn_group.get_metadata_builder()
            attn_metadata_i = builder.build(common_prefix_len=..., common_attn_metadata=cm_base)

            # 3. 将元数据按层名存入字典
            for layer_name in attn_group.layer_names:
                attn_metadata[layer_name] = attn_metadata_i

    return attn_metadata, spec_decode_common_attn_metadata
```

关键点：**同一个 KV Cache 组内的所有层共享同一个注意力元数据对象**，因为它们的注意力模式完全相同。

---

## 6. 后端 `forward()` 的具体实现（以 FlashInfer 为例）

`flashinfer.py:1696` 中 FlashInfer 后端的 `forward()`：

```python
def forward(self, layer, query, key, value, kv_cache, attn_metadata, output, ...):
    if attn_metadata is None:
        return output.fill_(0)  # Profiling 阶段

    # 1. 计算缩放因子（考虑量化）
    bmm1_scale = self.scale  # 1/sqrt(head_size)
    if is_quantized_kv_cache(self.kv_cache_dtype):
        bmm1_scale *= layer._q_scale_float * layer._k_scale_float

    # 2. 分别处理 prefill 和 decode
    #    prefill: 处理新输入的 token（长序列）
    #    decode:  处理自回归生成的 token（通常只有 1 个）
    if attn_metadata.num_prefills > 0:
        # 调用 FlashInfer 的 prefill kernel
        ...
    if attn_metadata.num_decodes > 0:
        # 调用 FlashInfer 的 decode kernel
        ...
```

---

## 7. `_model_forward()` 之后的后处理逻辑

`_model_forward()` 返回后，`execute_model()` 继续执行后处理（`gpu_model_runner.py:4322`）：

### 7.1 解包模型输出

```python
# EAGLE 3: 模型返回 (hidden_states, aux_hidden_states)
if self.use_aux_hidden_state_outputs:
    hidden_states, aux_hidden_states = model_output
else:
    hidden_states = model_output
    aux_hidden_states = None
```

### 7.2 Pipeline Parallel 处理

```python
if not get_pp_group().is_last_rank:
    # 非最后 PP 阶段: 返回中间张量给下一个 PP 阶段
    assert isinstance(hidden_states, IntermediateTensors)
    return hidden_states

if self.is_pooling_model:
    # 池化模型: 直接返回池化输出
    return self._pool(hidden_states, ...)
```

### 7.3 计算 Logits

```python
# 只对需要采样的位置计算 logits (logits_indices)
# 例如: decode 请求只需要最后一个 token 的 logits
sample_hidden_states = hidden_states[logits_indices]
logits = self.model.compute_logits(sample_hidden_states)
```

### 7.4 存储状态并返回 None

```python
# 将 forward pass 的结果存储在 execute_model_state 中
# 等待后续的 sample_tokens() 来消费
self.execute_model_state = ExecuteModelState(
    scheduler_output, logits, spec_decode_metadata, hidden_states, ...
)

# 返回 None，信号 EngineCore 需要调用 sample_tokens()
return None
```

返回 `None` 的设计目的：
1. 允许 EngineCore 在 forward 和采样之间计算 grammar 位图
2. 支持异步调度：GPU 可以立即开始下一批的 forward
3. 支持 Execute/Sample 分离的 overlap 模式

---

## 8. 总结：完整的数据流

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  gpu_model_runner.execute_model()                                           │
│                                                                             │
│  1. _prepare_inputs()          → 准备 input_ids, positions, slot_mapping    │
│  2. _build_attention_metadata() → 构建 attn_metadata (按层名字典)            │
│  3. set_forward_context(attn_metadata) → 设置全局 ForwardContext             │
│  4. _model_forward()            → 调用模型 forward                          │
│                                                                             │
│     ┌─────────────────────────────────────────────────────────────────────┐ │
│     │  LlamaModel.forward()                                               │ │
│     │    └─ LlamaDecoderLayer.forward()                                   │ │
│     │         └─ LlamaAttention.forward()                                 │ │
│     │              └─ self.attn(q, k, v)  # Attention.forward()           │ │
│     │                   └─ unified_attention_with_output()                │ │
│     │                        ├─ get_attention_context()                   │ │
│     │                        │    └─ get_forward_context()                │ │
│     │                        │    └─ attn_metadata[layer_name]            │ │
│     │                        └─ self.impl.forward()  # 后端实现            │ │
│     │                             └─ FlashInfer/FlashAttention kernel     │ │
│     └─────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  5. compute_logits()            → 计算输出 logits                           │
│  6. sample_tokens()             → 采样生成 token                            │
└─────────────────────────────────────────────────────────────────────────────┘
```

**核心设计思想**：Attention 层本身不持有注意力元数据，而是通过全局的 `ForwardContext` 在运行时动态获取。这种设计使得：

- 同一个模型中的所有 Attention 层共享同一个 `ForwardContext`
- 不同的 KV Cache 组可以使用不同的注意力后端
- 推测解码等复杂场景可以通过修改 `attn_metadata` 的结构来支持
