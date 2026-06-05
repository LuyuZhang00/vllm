# vLLM 核心算子与 CUDA Kernel 深度解析

> 本文档覆盖 LLM 推理中必须掌握的核心算子，包括注意力机制、矩阵乘法、归一化、位置编码、采样、MoE、量化等，结合 vLLM 代码和面试考点进行详细说明。

---

## 目录

- [1. 注意力算子](#1-注意力算子)
- [2. 矩阵乘法算子](#2-矩阵乘法算子)
- [3. 归一化算子](#3-归一化算子)
- [4. 位置编码算子](#4-位置编码算子)
- [5. 采样算子](#5-采样算子)
- [6. MoE 算子](#6-moe-算子)
- [7. 量化算子](#7-量化算子)
- [8. 通信算子](#8-通信算子)
- [9. 面试高频考点](#9-面试高频考点)

---

## 1. 注意力算子

### 1.1 标准注意力 (Scaled Dot-Product Attention)

**数学公式：**
```
Attention(Q, K, V) = softmax(Q × K^T / √d_k) × V
```

**面试考点：**
- 为什么除以 `√d_k`？→ 防止点积值过大导致 softmax 梯度消失
- 时间复杂度：O(n² × d)，空间复杂度：O(n²)
- 自回归生成中，KV Cache 避免重复计算

**vLLM 实现：**
```python
# vllm/model_executor/layers/attention/attention.py
class Attention(nn.Module):
    def forward(self, query, key, value, ...):
        # 1. 算 Q, K, V
        # 2. 写入 KV Cache
        # 3. 执行注意力计算
        output = self.impl.forward(query, key, value, ...)
```

### 1.2 Multi-Head Attention (MHA)

**数学公式：**
```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) × W_O
where head_i = Attention(Q × W_Q_i, K × W_K_i, V × W_V_i)
```

**面试考点：**
- h 个独立的注意力头，每个头有独立的 QKV 投影
- 总参数量：4 × d_model × d_model (Q, K, V, O 投影)
- 多头的意义：不同头关注不同的语义关系

### 1.3 Multi-Query Attention (MQA)

**与 MHA 的区别：**
```
MHA:  每个头有独立的 K, V  → num_heads 个 KV head
MQA:  所有头共享同一组 K, V → 1 个 KV head
GQA:  分组共享 K, V        → num_kv_heads 个 KV head (1 < num_kv_heads < num_heads)
```

**面试考点：**
- MQA 大幅减少 KV Cache 大小：从 `2 × num_heads × d_head` 降到 `2 × 1 × d_head`
- GQA 是 MQA 和 MHA 的折中：Llama 2 70B 使用 GQA (num_heads=64, num_kv_heads=8)
- 推理时 KV Cache 内存减少比例：`num_kv_heads / num_heads`

**vLLM 实现：**
```python
# 注意力计算时，K/V 需要广播到所有 Q head
# FlashAttention 内部处理 GQA 的 head 扩展
flash_attn_varlen_func(
    q=query,        # [num_tokens, num_heads, head_size]
    k=key_cache,    # [num_blocks, 2, block_size, num_kv_heads, head_size]
    v=value_cache,
    ...
)
# FlashAttention 内部: 每个 Q head attend 到对应的 KV head 组
```

### 1.4 Multi-head Latent Attention (MLA)

**DeepSeek V2/V3/V4 的创新：**

```
传统 MHA: 每 token 存储 num_heads × head_size 维的 K, V
MLA:      每 token 只存储 kv_lora_rank 维的压缩潜在表示

存储: [kv_c_normed; k_pe]
  kv_c_normed: 压缩 KV 潜在表示 (512 维)
  k_pe: 解耦的 RoPE 位置编码 (64 维)
  总计: 576 维 (vs MHA 的 4096+ 维)
```

**面试考点：**
- MLA 通过低秩压缩减少 KV Cache 大小
- Decode 阶段使用 MQA 路径：查询投影到潜在空间，直接与压缩 KV 计算
- Prefill 阶段使用 MHA 路径：先重建完整 K/V，再计算注意力

**vLLM 实现：**
```python
# vllm/v1/attention/backends/flashmla.py
# MLA 的 KV Cache 形状: [num_blocks, block_size, head_size]
# head_size = kv_lora_rank + qk_rope_head_dim (如 512 + 64 = 576)

# 写入: concat_and_cache_mla(kv_c_normed, k_pe, kv_cache, slot_mapping)
# 读取: flash_mla_with_kvcache(q, k_cache, block_table, ...)
```

### 1.5 FlashAttention

**核心思想：** 分块计算 + 在线 Softmax，避免物化完整的 n×n 注意力矩阵

**面试考点：**
- 传统注意力需要 O(n²) 显存存储注意力矩阵
- FlashAttention 通过 Tiling 将 K, V 分成块，逐块计算
- 使用在线 Softmax 维护运行最大值 M 和运行和 L，逐块合并结果
- 显存复杂度从 O(n²) 降到 O(n)
- 计算量不变，但减少了 HBM 访问，实际速度更快

**在线 Softmax 公式：**
```
对于每个 KV 块 j:
  m_j = max(M, max(S_j))           # 更新最大值
  α = exp(M - m_j)                  # 重缩放因子
  L = L × α + sum(exp(S_j - m_j))  # 更新运行和
  acc = acc × α + exp(S_j - m_j) × V_j  # 更新累积输出
  M = m_j                           # 更新最大值

最终输出: acc / L
```

**vLLM Triton 实现：**
```python
# vllm/v1/attention/ops/chunked_prefill_paged_decode.py
# 在线 Softmax 的 Triton 内核 (line 126-246)

M = tl.full([num_queries], float("-inf"), dtype=tl.float32)  # 运行最大值
L = tl.zeros([num_queries], dtype=tl.float32)                # 运行和
acc = tl.zeros([num_queries, HEAD_SIZE], dtype=tl.float32)   # 累积输出

for j in range(0, num_blocks):
    S = tl.where(mask, qk, float("-inf"))
    m_j = tl.maximum(M, tl.max(S, axis=1))     # 新最大值
    p = tl.exp(S - m_j[:, None])                 # softmax 分子
    l_j = tl.sum(p, axis=1)                      # 当前块和
    alpha = tl.exp(M - m_j)                      # 重缩放因子
    acc = acc * alpha[:, None]                    # 重缩放累积
    L = L * alpha + l_j                           # 更新运行和
    M = m_j                                       # 更新最大值
    acc += tl.dot(p.to(V.dtype), V)              # 累加加权 V

acc = acc / (L[:, None] + 1e-10)                 # 最终归一化
```

### 1.6 PagedAttention

**核心思想：** 将 KV Cache 分成固定大小的块，通过块表间接寻址

**面试考点：**
- 传统方式需要连续内存，导致碎片和浪费
- PagedAttention 使用块表 (Block Table) 映射逻辑位置到物理块
- 每个 token 的 KV 写入位置由 slot_mapping 决定
- 支持非连续内存分配，消除碎片

**vLLM 实现：**
```python
# vllm/v1/attention/backends/flash_attn.py, line 850
def do_kv_cache_update(self, layer, key, value, kv_cache, slot_mapping):
    key_cache, value_cache = kv_cache.unbind(1)
    # scatter write: 将 K, V 写入 slot_mapping 指定的位置
    reshape_and_cache_flash(key, value, key_cache, value_cache, slot_mapping)

# 注意力计算时，通过 block_table 间接寻址
flash_attn_varlen_func(
    q=query, k=key_cache, v=value_cache,
    block_table=block_table,  # [batch_size, max_num_blocks_per_seq]
    ...
)
```

### 1.7 Cascade Attention

**核心思想：** 多个请求共享长公共前缀时，前缀注意力只计算一次

**面试考点：**
- 当所有请求有相同系统 prompt 时，前缀的 KV 可以共享
- 分为前缀注意力（共享）和后缀注意力（独立）
- 使用 LSE (Log-Sum-Exp) 合并两部分结果

**vLLM 实现：**
```python
# vllm/v1/attention/backends/flash_attn.py, line 1132
def cascade_attention(query, key_cache, value_cache, block_table, ...):
    # 1. 前缀注意力: 所有 Q attend 到共享前缀
    prefix_out = flash_attn(q=query, k=key_cache, v=value_cache,
                            block_table=block_table[:1], causal=False)

    # 2. 后缀注意力: 每个 Q attend 到自己的后缀
    suffix_out = flash_attn(q=query, k=key_cache, v=value_cache,
                            block_table=block_table[:, num_common_blocks:],
                            causal=True)

    # 3. LSE 合并
    output = merge_attn_states(prefix_out, suffix_out)
```

---

## 2. 矩阵乘法算子

### 2.1 GEMM (General Matrix Multiplication)

**数学公式：**
```
C = A × B + bias
其中 A: [M, K], B: [K, N], C: [M, N]
```

**面试考点：**
- GEMM 是 LLM 推理中计算量最大的操作
- Prefill 阶段：大矩阵 × 大矩阵 → 计算密集
- Decode 阶段：向量 × 大矩阵 → 访存密集
- GPU 利用率取决于矩阵大小和硬件特性

**vLLM 中的 GEMM：**
```python
# 线性层
class Linear(nn.Module):
    def forward(self, x):
        return F.linear(x, self.weight, self.bias)
        # 等价于 x @ weight.T + bias

# QKV 投影
Q = W_q @ hidden_states  # [M, d_model] @ [d_model, d_model] → [M, d_model]
K = W_k @ hidden_states
V = W_v @ hidden_states

# FFN
hidden = W1 @ x          # [M, d_model] @ [d_model, 4*d_model] → [M, 4*d_model]
hidden = GELU(hidden)
output = W2 @ hidden     # [M, 4*d_model] @ [4*d_model, d_model] → [M, d_model]
```

### 2.2 Grouped GEMM (MoE 场景)

**面试考点：**
- MoE 模型有多个专家，每个专家是一个独立的 FFN
- Grouped GEMM 将多个小 GEMM 合并为一个大 GEMM
- 减少 kernel 启动开销，提高 GPU 利用率

**vLLM 实现：**
```python
# vllm/model_executor/layers/fused_moe/
# 使用 DeepGEMM 或 Triton 实现 fused MoE GEMM
# 将多个专家的 GEMM 合并为一次 kernel 调用
```

### 2.3 CUTLASS / cuBLAS

**面试考点：**
- cuBLAS：NVIDIA 官方 BLAS 库，通用但不一定最优
- CUTLASS：NVIDIA 的模板库，可自定义 GEMM kernel
- vLLM 根据矩阵大小和硬件选择最优的 GEMM 实现

---

## 3. 归一化算子

### 3.1 Layer Normalization

**数学公式：**
```
LayerNorm(x) = γ × (x - μ) / √(σ² + ε) + β
其中 μ = mean(x), σ² = var(x)
```

**面试考点：**
- 对每个样本的每个特征维度独立归一化
- 需要两次遍历：一次算均值方差，一次归一化
- 参数量：2 × d_model (γ 和 β)

### 3.2 RMSNorm (Root Mean Square Normalization)

**数学公式：**
```
RMSNorm(x) = x / √(mean(x²) + ε) × γ
```

**面试考点：**
- 比 LayerNorm 更简单：没有均值减法，没有偏置项
- Llama、Qwen 等现代模型普遍使用 RMSNorm
- 计算量比 LayerNorm 少约 10-15%
- 只需一次遍历计算 RMS

**vLLM 实现：**
```python
# vllm/model_executor/layers/layernorm.py
class RMSNorm(CustomOp):
    def forward_native(self, x, residual=None):
        # 原生 PyTorch 实现
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        x = x.to(orig_dtype) * self.weight
        return x

    def forward_cuda(self, x, residual=None):
        # CUDA 自定义 kernel 实现
        ...
```

### 3.3 Fused Add + RMSNorm

**面试考点：**
- 将残差加法和 RMSNorm 融合为一个 kernel
- 减少一次 HBM 读写（不需要先写残差结果再读回来）
- vLLM 的 CustomOp 支持 CUDA 和 Triton 两种实现

---

## 4. 位置编码算子

### 4.1 RoPE (Rotary Position Embedding)

**数学公式：**
```
RoPE(x, pos) = x × R(pos)
其中 R(pos) 是旋转矩阵:
  R(pos)[2i, 2i] = cos(pos × θ_i)
  R(pos)[2i, 2i+1] = -sin(pos × θ_i)
  R(pos)[2i+1, 2i] = sin(pos × θ_i)
  R(pos)[2i+1, 2i+1] = cos(pos × θ_i)
θ_i = 1 / (10000^(2i/d))
```

**面试考点：**
- RoPE 将位置信息编码到 Q, K 中，而不是加到输入 embedding
- 相对位置编码：两个 token 的注意力分数只取决于它们的相对距离
- 旋转操作等价于复数乘法：`(x + iy) × (cos θ + i sin θ)`
- 长度外推：NTK-Aware Scaling、YaRN 等方法扩展上下文长度

**vLLM 实现：**
```python
# vllm/model_executor/layers/rotary_embedding.py
class RotaryEmbedding(CustomOp):
    def forward_native(self, positions, query, key, offsets=None):
        # 计算 cos 和 sin
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        # 应用旋转
        query = query * cos + rotate_half(query) * sin
        key = key * cos + rotate_half(key) * sin
        return query, key

    def forward_cuda(self, positions, query, key, offsets=None):
        # CUDA 自定义 kernel
        ...
```

### 4.2 RoPE 变体

| 变体 | 特点 | 代表模型 |
|------|------|----------|
| 标准 RoPE | 基础旋转位置编码 | Llama, Qwen |
| NTK-Aware | 动态调整频率基数 | CodeLlama |
| YaRN | 注意力缩放 + NTK | Yi |
| ALiBi | 线性偏置替代旋转 | BLOOM |
| Dynamic NTK | 推理时动态调整 | ChatGLM |

---

## 5. 采样算子

### 5.1 Top-K 采样

**算法：**
```
1. 计算 logits → probabilities (softmax)
2. 保留概率最高的 K 个 token
3. 重新归一化
4. 从 K 个 token 中采样
```

**面试考点：**
- K 太小：生成重复、无创意
- K 太大：可能选到低质量 token
- 通常 K=50 是一个不错的默认值

### 5.2 Top-P (Nucleus) 采样

**算法：**
```
1. 计算 logits → probabilities (softmax)
2. 按概率降序排序
3. 累加概率直到超过 p
4. 只保留累加范围内的 token
5. 重新归一化并采样
```

**面试考点：**
- 动态截断：token 数量不固定，取决于概率分布
- 比 Top-K 更灵活：高置信度时只保留少数 token，低置信度时保留更多
- 通常 p=0.9 是一个不错的默认值

### 5.3 Temperature 采样

**数学公式：**
```
p_i = exp(logit_i / T) / Σ exp(logit_j / T)
```

**面试考点：**
- T < 1：分布更尖锐，生成更确定性
- T > 1：分布更平坦，生成更多样性
- T → 0：等价于 greedy (argmax)
- T → ∞：等价于均匀分布

### 5.4 Min-P 采样

**算法：**
```
1. 找到最大概率 p_max
2. 计算阈值 p_threshold = p_max × min_p
3. 保留概率 >= p_threshold 的 token
4. 重新归一化并采样
```

**面试考点：**
- 比 Top-P 更直观的截断方式
- 自适应于概率分布的形状

### 5.5 采样 kernel 实现

**vLLM 实现：**
```python
# vllm/v1/sample/ops/
# 使用 Triton kernel 实现高效的批量采样

# 1. Top-K: 使用 partial sort 或 heap
# 2. Top-P: 使用 prefix sum + binary search
# 3. Temperature: 逐元素除法
# 4. 最终采样: multinomial 或 Gumbel-max
```

---

## 6. MoE 算子

### 6.1 Mixture of Experts 基本原理

**面试考点：**
- MoE 用多个专家替代单一 FFN，每个 token 只激活部分专家
- Gate 网络决定每个 token 路由到哪些专家
- 总参数量大但每次前向只使用一小部分

**数学公式：**
```
MoE(x) = Σ_i g_i(x) × Expert_i(x)
其中 g_i(x) = softmax(TopK(Gate(x)))_i
```

### 6.2 Top-K Gate

**算法：**
```
1. gate_logits = Gate(x)  # [num_tokens, num_experts]
2. topk_values, topk_indices = TopK(gate_logits, k=2)
3. gate_weights = softmax(topk_values)
4. 路由 token 到对应的专家
```

**面试考点：**
- 通常 K=2（每个 token 激活 2 个专家）
- 负载均衡损失防止所有 token 路由到少数专家
- Token dropping：超出容量的 token 被丢弃

### 6.3 Expert Parallelism All-to-All

**面试考点：**
- EP 将专家分布到多个 GPU
- All-to-All 通信：每个 GPU 将 token 发送到对应专家所在的 GPU
- 计算完成后，再 All-to-All 将结果发回

**vLLM 实现：**
```python
# vllm/distributed/device_communicators/all2all.py
# 多种 All-to-All 后端:
# - allgather_reducescatter: 通用
# - deepep_high_throughput: DeepEP 高吞吐
# - deepep_low_latency: DeepEP 低延迟
# - flashinfer_nvlink: FlashInfer NVLink
```

### 6.4 Fused MoE Kernel

**面试考点：**
- 将多个专家的 GEMM 融合为一个 kernel
- 减少 kernel 启动开销
- 使用 grouped GEMM 或 Triton 实现

---

## 7. 量化算子

### 7.1 FP8 量化

**数学公式：**
```
q = round(x / scale)
scale = max(|x|) / 448.0  # FP8 E4M3 最大值

反量化: x ≈ q × scale
```

**面试考点：**
- FP8 E4M3：4 位指数 + 3 位尾数，范围 [-448, 448]
- FP8 E5M2：5 位指数 + 2 位尾数，范围更大但精度更低
- 权重量化：静态，训练后量化
- 激活量化：动态，推理时量化

**vLLM 实现：**
```python
# 权重量化
weight_fp8 = (weight / scale).to(torch.float8_e4m3fn)

# 计算
output = F.linear(x, weight_fp8.to(x.dtype) * scale)

# KV Cache FP8
key_cache_fp8 = (key / key_scale).to(torch.float8_e4m3fn)
```

### 7.2 INT8 量化 (SmoothQuant)

**面试考点：**
- 权重 INT8 + 激活 INT8
- SmoothQuant：将激活的量化难度转移到权重
- `Y = (X × diag(s)) × (diag(1/s) × W)`
- s 是平滑因子，使权重和激活的分布更均匀

### 7.3 GPTQ / AWQ

**面试考点：**
- GPTQ：基于 Hessian 的逐层量化，每列独立量化
- AWQ：保护显著权重（activation-aware），对重要通道不量化
- 两者都是 4-bit 权重量化方法
- 量化后需要特殊的 GEMM kernel (Marlin, ExLlamaV2)

### 7.4 GGUF

**面试考点：**
- GGUF 是 llama.cpp 的量化格式
- 支持多种量化级别：Q4_0, Q4_K_M, Q5_K_M, Q8_0 等
- 混合精度：不同层使用不同的量化精度
- 适合 CPU 推理

---

## 8. 通信算子

### 8.1 AllReduce

**数学公式：**
```
AllReduce(x_0, x_1, ..., x_{n-1}) = Σ x_i
每个 GPU 拥有相同的求和结果
```

**面试考点：**
- Ring AllReduce：分两步（reduce-scatter + all-gather），通信量与 GPU 数无关
- Tree AllReduce：树形结构，延迟更低但带宽利用率低
- vLLM 使用 NCCL 实现，支持多种算法

**vLLM 实现：**
```python
# vllm/distributed/device_communicators/cuda_communicator.py
def all_reduce(self, input_):
    # 选择最快的实现:
    # 1. NCCL Symmetric Memory
    # 2. QuickReduce (AMD)
    # 3. FlashInfer
    # 4. CustomAllreduce (IPC)
    # 5. PyNCCL (兜底)
```

### 8.2 AllGather

**数学公式：**
```
AllGather(x_0, x_1, ..., x_{n-1}) = Concat(x_0, x_1, ..., x_{n-1})
每个 GPU 拥有所有 GPU 的数据拼接结果
```

**面试考点：**
- 用于张量并行的列并行层：每个 GPU 计算部分结果，然后 AllGather

### 8.3 ReduceScatter

**数学公式：**
```
ReduceScatter(x_0, x_1, ..., x_{n-1})[i] = Σ x_j[i]
每个 GPU 拥有求和结果的一个分片
```

**面试考点：**
- 用于张量并行的行并行层：先 ReduceScatter，再本地计算

### 8.4 All-to-All

**面试考点：**
- 用于 MoE 的专家并行
- 每个 GPU 将 token 发送到对应专家所在的 GPU
- 通信模式完全不同于 AllReduce

---

## 9. 面试高频考点

### 9.1 FlashAttention 详细原理

**Q: FlashAttention 如何实现 O(n) 显存？**

A: 核心是分块计算 + 在线 Softmax：
1. 将 K, V 分成大小为 B 的块
2. 逐块计算 Q×K^T，得到局部 softmax 分子
3. 维护运行最大值 M 和运行和 L
4. 每处理一个块，更新 M 和 L，重缩放之前的累积结果
5. 最终 acc / L 得到输出

**Q: FlashAttention 的在线 Softmax 公式？**

A: 见 1.5 节，核心是 `alpha = exp(M_old - M_new)` 重缩放因子。

**Q: FlashAttention 与标准注意力的计算量对比？**

A: 计算量相同 O(n²d)，但 HBM 访问从 O(n²) 降到 O(n²d/M)（M 是 SRAM 大小）。

### 9.2 PagedAttention 详细原理

**Q: PagedAttention 如何工作？**

A: 
1. KV Cache 分成固定大小的块（如 16 tokens）
2. 每个请求有一个块表，映射逻辑块到物理块
3. slot_mapping 计算每个 token 的物理写入位置
4. 注意力计算时通过块表间接寻址读取 KV

**Q: 为什么 PagedAttention 能减少内存浪费？**

A: 传统方式按最大长度预分配连续内存，短请求浪费严重。PagedAttention 按需分配块，无碎片。

### 9.3 GQA vs MQA vs MHA

**Q: 三种注意力的区别？**

A:
- MHA：每个头独立的 K, V → 最大 KV Cache
- MQA：所有头共享 K, V → 最小 KV Cache
- GQA：分组共享 K, V → 折中方案

**Q: GQA 的 KV Cache 大小？**

A: `2 × num_kv_heads × head_size × seq_len × dtype_size`

### 9.4 RoPE 详细原理

**Q: RoPE 如何实现相对位置编码？**

A: 通过旋转矩阵，`<RoPE(q, m), RoPE(k, n)> = <q, k>` 仅依赖于 `m-n`（相对距离）。

**Q: RoPE 的长度外推方法？**

A:
- NTK-Aware：调整频率基数 `base' = base × α^(d/(d-2))`
- YaRN：NTK + 注意力缩放
- Dynamic NTK：推理时根据序列长度动态调整

### 9.5 量化方法对比

**Q: FP8 vs INT8 vs GPTQ vs AWQ 的区别？**

A:
- FP8：硬件原生支持（H100+），推理速度快
- INT8：通用，需要校准
- GPTQ：4-bit 权重，基于 Hessian 的逐层量化
- AWQ：4-bit 权重，保护显著权重

**Q: 量化的精度损失如何控制？**

A: 
- 校准数据集：用代表性数据量化
- 混合精度：关键层保持高精度
- 平滑量化：将激活难度转移到权重

### 9.6 MoE 负载均衡

**Q: MoE 的负载不均衡问题如何解决？**

A:
- 辅助损失：鼓励均匀路由
- 容量因子：限制每个专家的 token 数
- 冗余专家：复制热门专家
- EPLB：运行时动态重分配专家

### 9.7 CUDA 优化技巧

**Q: LLM 推理中有哪些 CUDA 优化技巧？**

A:
1. **Kernel 融合**：将多个小 kernel 合并为一个大 kernel
2. **CUDA Graph**：预捕获 kernel 序列，消除 CPU 开销
3. **异步执行**：D2H 拷贝与计算重叠
4. **Shared Memory**：减少 HBM 访问
5. **Warp 级原语**：使用 warp shuffle 进行归约
6. **向量化加载**：使用 vectorized load 提高带宽利用率

---

## 附录：vLLM 算子文件索引

| 算子类别 | 文件位置 | 关键实现 |
|----------|----------|----------|
| FlashAttention | `vllm/v1/attention/backends/flash_attn.py` | flash_attn_varlen_func |
| FlashMLA | `vllm/v1/attention/backends/flashmla.py` | flash_mla_with_kvcache |
| PagedAttention | `vllm/v1/attention/ops/chunked_prefill_paged_decode.py` | Triton kernel |
| 在线 Softmax | `vllm/v1/attention/ops/chunked_prefill_paged_decode.py` | Triton kernel |
| Merge Attn States | `vllm/v1/attention/ops/triton_merge_attn_states.py` | LSE 合并 |
| RMSNorm | `vllm/model_executor/layers/layernorm.py` | CustomOp (CUDA + Triton) |
| RoPE | `vllm/model_executor/layers/rotary_embedding.py` | CustomOp |
| Sampler | `vllm/v1/sample/` | TopK, TopP, Temperature |
| Fused MoE | `vllm/model_executor/layers/fused_moe/` | DeepGEMM, Triton |
| All2All | `vllm/distributed/device_communicators/all2all.py` | 多种 EP 后端 |
| AllReduce | `vllm/distributed/device_communicators/cuda_communicator.py` | NCCL, CustomAllreduce |
| 量化 | `vllm/model_executor/layers/quantization/` | 29 种量化方法 |
| Reshape & Cache | `vllm/v1/attention/ops/` | Triton scatter write |

---

## 10. vLLM 特色算子详解

### 10.1 Reshape And Cache (KV Cache 写入算子)

**功能：** 将新计算的 K, V 张量散射写入 Paged KV Cache 的指定位置。

**数学操作：**
```
对于每个 token i:
  slot_id = slot_mapping[i]
  block_id = slot_id // block_size
  block_offset = slot_id % block_size
  key_cache[block_id, block_offset, :, :] = key[i]
  value_cache[block_id, block_offset, :, :] = value[i]
```

**面试考点：**
- 这是一个 **scatter write** 操作：每个 token 写入不同的物理位置
- slot_mapping 由 block_table 和 position 计算得出
- 支持 FP8 量化写入：写入时除以 scale

**vLLM Triton 实现：**
```python
# vllm/v1/attention/ops/triton_reshape_and_cache_flash.py
@triton.jit
def reshape_and_cache_kernel_flash(
    key_ptr, value_ptr, key_cache_ptr, value_cache_ptr,
    slot_mapping_ptr, ...
):
    # 每个 program 处理一个 token
    token_idx = tl.program_id(0)

    # 1. 加载 slot_mapping
    slot_idx = tl.load(slot_mapping_ptr + token_idx)

    # 2. 计算块号和块内偏移
    block_idx = slot_idx // block_size
    block_offset = slot_idx % block_size

    # 3. 计算目标地址
    # NHD 布局: cache[block_idx, k/v, block_offset, head, dim]
    target_idx = block_idx * block_stride + block_offset * page_stride

    # 4. 散射写入
    for head in range(num_heads):
        for dim in range(head_size):
            key_val = tl.load(key_ptr + ...)
            tl.store(key_cache_ptr + target_idx + head * head_stride + dim, key_val)
```

### 10.2 Slot Mapping 计算算子

**功能：** 将 token 的逻辑位置映射到 KV Cache 的物理 slot。

**数学公式：**
```
slot_id = block_table[req_idx, position // block_size] * block_size + position % block_size
```

**面试考点：**
- slot_mapping 是 PagedAttention 的核心：决定了每个 token 的 KV 写到哪里
- 计算简单但必须高效：每个 token 一次查表 + 一次乘法 + 一次取模
- vLLM 使用 Triton kernel 批量计算

**vLLM 实现：**
```python
# vllm/v1/worker/block_table.py
@triton.jit
def _compute_slot_mapping_kernel(
    positions_ptr, block_table_ptr, slot_mapping_ptr,
    block_size, ...
):
    # 每个 program 处理一个 token
    token_idx = tl.program_id(0)

    # 1. 加载 position
    pos = tl.load(positions_ptr + token_idx)

    # 2. 计算块索引
    block_idx = pos // block_size

    # 3. 查找物理块号
    block_number = tl.load(block_table_ptr + req_idx * max_blocks + block_idx)

    # 4. 计算 slot
    block_offset = pos % block_size
    slot_id = block_number * block_size + block_offset

    # 5. 存储
    tl.store(slot_mapping_ptr + token_idx, slot_id)
```

### 10.3 Top-K 采样算子

**功能：** 从 logits 中选择概率最高的 K 个 token。

**数学操作：**
```
1. probs = softmax(logits)
2. topk_values, topk_indices = topk(probs, k)
3. 从 topk_indices 中采样
```

**面试考点：**
- Top-K 的实现方式：
  - **排序法**：完整排序后取前 K → O(n log n)
  - **堆法**：维护大小为 K 的堆 → O(n log K)
  - **快速选择**：QuickSelect 算法 → O(n) 平均
- GPU 上通常使用 **radix sort** 或 **bitonic sort**
- vLLM 使用 Triton kernel 实现高效的批量 Top-K

**vLLM 实现：**
```python
# vllm/v1/sample/ops/topk_topp_ops.py
# 使用 Triton kernel 实现批量 Top-K
@triton.jit
def topk_kernel(
    logits_ptr, topk_values_ptr, topk_indices_ptr,
    num_tokens, vocab_size, k, ...
):
    # 每个 program 处理一个 token
    token_idx = tl.program_id(0)

    # 1. 加载 logits
    logits = tl.load(logits_ptr + token_idx * vocab_size + arange)

    # 2. 计算 softmax
    probs = tl.softmax(logits)

    # 3. Top-K 选择 (使用 partial sort)
    # ... Triton 实现
```

### 10.4 Top-P (Nucleus) 采样算子

**功能：** 从 logits 中选择累积概率超过 p 的最小 token 集合。

**数学操作：**
```
1. probs = softmax(logits)
2. sorted_probs, sorted_indices = sort(probs, descending=True)
3. cum_probs = cumsum(sorted_probs)
4. mask = cum_probs - sorted_probs < p  # 保留累积到 p 之前的
5. 从 mask 为 True 的 token 中采样
```

**面试考点：**
- Top-P 是动态截断：token 数量不固定
- 实现难点：需要排序 + 前缀和 + 掩码
- GPU 上可以融合为一个 kernel

**vLLM 实现：**
```python
# vllm/v1/sample/ops/topk_topp_ops.py
@triton.jit
def topp_kernel(
    logits_ptr, topp_values_ptr, topp_indices_ptr,
    num_tokens, vocab_size, p, ...
):
    # 1. 加载 logits 并计算 softmax
    # 2. 排序 (Triton 内置 sort)
    # 3. 计算前缀和
    # 4. 生成掩码
    # 5. 采样
```

### 10.5 Gumbel-Max 采样算子

**功能：** 从分类分布中采样，避免显式计算 softmax。

**数学公式：**
```
sampled_token = argmax(logit_i + Gumbel_i)
其中 Gumbel_i = -log(-log(uniform_i))
```

**面试考点：**
- Gumbel-Max 不需要计算 softmax，直接在 logit 空间操作
- 数学上等价于从 softmax 分布中采样
- GPU 上更高效：避免了 softmax 的指数和求和

### 10.6 Fused Softmax 算子

**功能：** 将 softmax 的多个步骤融合为一个 kernel。

**传统 softmax 的三步：**
```
1. max_val = max(logits)
2. sum_exp = sum(exp(logits - max_val))
3. result = exp(logits - max_val) / sum_exp
```

**融合 kernel：**
```
单个 kernel 内完成:
1. 使用 shared memory 存储中间结果
2. 第一次遍历: 计算 max_val
3. 第二次遍历: 计算 sum_exp 并直接输出结果
```

**面试考点：**
- 为什么要融合？减少 HBM 读写次数
- 需要两次遍历：第一次找最大值，第二次计算结果
- 使用 shared memory 存储中间结果

### 10.7 Rotary Embedding 算子

**功能：** 将旋转位置编码应用到 Q, K 张量。

**数学操作：**
```
对于 Q 或 K 的每对相邻维度 (x, y):
  x' = x * cos(pos * θ) - y * sin(pos * θ)
  y' = x * sin(pos * θ) + y * cos(pos * θ)
```

**面试考点：**
- 等价于复数乘法：`(x + iy) × (cos θ + i sin θ)`
- 可以用矩阵乘法实现：`[x', y'] = [[cos, -sin], [sin, cos]] × [x, y]`
- GPU 实现通常使用向量化操作

**vLLM 实现：**
```python
# vllm/model_executor/layers/rotary_embedding.py
@triton.jit
def rotary_embedding_kernel(
    positions_ptr, query_ptr, key_ptr,
    cos_ptr, sin_ptr, ...
):
    # 1. 计算 cos 和 sin
    freq = positions * theta
    cos_val = tl.cos(freq)
    sin_val = tl.sin(freq)

    # 2. 应用旋转
    q_rotated = q * cos_val + rotate_half(q) * sin_val
    k_rotated = k * cos_val + rotate_half(k) * sin_val
```

### 10.8 Custom AllReduce 算子

**功能：** 使用 IPC 共享内存实现高效的 AllReduce。

**实现原理：**
```
1. 每个 GPU 分配共享内存缓冲区
2. 使用 cudaIpcGetMemHandle 获取 IPC handle
3. 通过 CPU 交换 handle
4. 使用 cudaIpcOpenMemHandle 打开远程内存
5. 直接读写远程 GPU 的内存 (零拷贝)
```

**面试考点：**
- 比 NCCL 更快的原因：避免了 NCCL 的额外开销
- 限制条件：必须在同一节点、NVLink 连接
- 适用于 TP 场景：每层都需要 AllReduce

**vLLM 实现：**
```python
# vllm/distributed/device_communicators/custom_all_reduce.py
class CustomAllreduce:
    def __init__(self, ...):
        # 1. 分配共享内存
        self.meta = custom_ar.init_custom_ar(handles, rank, opts)

    def all_reduce(self, input_, output=None):
        # 2. 直接调用 CUDA kernel
        custom_ar.all_reduce(self.meta, input_, output, ...)
```

### 10.9 Reshape and Cache MLA 算子

**功能：** 将 MLA 的压缩 KV 写入 Paged Cache。

**数学操作：**
```
对于每个 token i:
  slot_id = slot_mapping[i]
  cache[slot_id, :] = concat(kv_c_normed[i], k_pe[i])
```

**面试考点：**
- MLA 存储的是压缩表示，不是完整的 K, V
- 每个 token 存储 kv_lora_rank + qk_rope_head_dim 维
- DeepSeek V3: 512 + 64 = 576 维

**vLLM 实现：**
```python
# 调用 C++ kernel
ops.concat_and_cache_mla(
    kv_c_normed,     # [num_tokens, kv_lora_rank]
    k_pe,            # [num_tokens, qk_rope_head_dim]
    kv_cache,        # [num_blocks, block_size, head_size]
    slot_mapping,
)
```

### 10.10 Merge Attention States 算子

**功能：** 合并多个部分注意力结果（用于 Cascade Attention 和 DCP）。

**数学公式：**
```
对于两个部分结果 (out1, lse1) 和 (out2, lse2):
  max_lse = max(lse1, lse2)
  scale1 = exp(lse1 - max_lse)
  scale2 = exp(lse2 - max_lse)
  out = (out1 * scale1 + out2 * scale2) / (scale1 + scale2)
  lse = log(scale1 + scale2) + max_lse
```

**面试考点：**
- 使用 LSE (Log-Sum-Exp) 而不是直接合并概率
- 数值稳定性：通过 max_lse 防止指数溢出
- 用于 Cascade Attention 的前缀+后缀合并

**vLLM 实现：**
```python
# vllm/v1/attention/ops/triton_merge_attn_states.py
@triton.jit
def merge_attn_states_kernel(
    prefix_out_ptr, prefix_lse_ptr,
    suffix_out_ptr, suffix_lse_ptr,
    output_ptr, output_lse_ptr, ...
):
    # 1. 加载两个部分的 LSE
    p_lse = tl.load(prefix_lse_ptr + ...)
    s_lse = tl.load(suffix_lse_ptr + ...)

    # 2. 计算最大值
    max_lse = tl.maximum(p_lse, s_lse)

    # 3. 计算缩放因子
    p_scale = tl.exp(p_lse - max_lse)
    s_scale = tl.exp(s_lse - max_lse)

    # 4. 加权合并
    out = (p_out * p_scale + s_out * s_scale) / (p_scale + s_scale)
    lse = tl.log(p_scale + s_scale) + max_lse
```

---

## 11. CUDA 编程面试必知

### 11.1 CUDA 基础概念

**Grid, Block, Thread 层次：**
```
Grid (网格)
  └── Block (线程块)
       └── Thread (线程)

Grid: 由多个 Block 组成，对应一个 kernel 调用
Block: 由多个 Thread 组成，共享 Shared Memory
Thread: 最小执行单位，有唯一的 (blockIdx, threadIdx)
```

**面试考点：**
- Block 内的 Thread 可以通过 Shared Memory 通信
- Block 间不能直接通信，需要通过 Global Memory 或多次 kernel 调用
- Warp = 32 个 Thread，是 GPU 调度的最小单位

### 11.2 内存层次

```
寄存器 (Register)     ← 最快，每线程私有
  ↓
共享内存 (Shared Memory) ← Block 内共享，低延迟
  ↓
L1 Cache              ← 硬件管理
  ↓
L2 Cache              ← 硬件管理
  ↓
全局内存 (Global Memory) ← 最慢，所有线程可见
```

**面试考点：**
- Shared Memory 大小有限（通常 48KB-163KB per SM）
- Global Memory 访问有高延迟（~400 cycles）
- 合并访问 (Coalesced Access) 可以提高带宽利用率

### 11.3 Triton 编程模型

**Triton vs CUDA：**
```
CUDA: 显式管理线程、共享内存、同步
Triton: 编译器自动管理，只需关注计算逻辑

CUDA: __global__ void kernel(...)
Triton: @triton.jit def kernel(...)
```

**面试考点：**
- Triton 适合矩阵和向量操作
- Triton 自动处理内存合并、共享内存分配
- vLLM 大量使用 Triton 实现自定义 kernel

### 11.4 CUDA Graph

**原理：**
```
普通执行:
  CPU: [启动kernel1] [启动kernel2] [启动kernel3] ...
  GPU: [执行kernel1] [执行kernel2] [执行kernel3] ...
  问题: CPU 启动开销大

CUDA Graph:
  捕获: CPU 启动所有 kernel，GPU 记录执行图
  回放: GPU 直接回放整个图，无需 CPU 参与
  优势: 消除 CPU 启动开销
```

**面试考点：**
- 适用于固定形状的计算（如统一 decode batch）
- vLLM 在 `_is_uniform_decode()` 为 True 时使用 FULL CUDA Graph
- 混合 prefill+decode 时使用 PIECEWISE CUDA Graph

### 11.5 Warp 级原语

**Warp Shuffle：**
```cuda
// 同一个 Warp 内的线程交换数据
__shfl_xor_sync(mask, val, delta)  // XOR 排列
__shfl_down_sync(mask, val, delta)  // 向下移位
__shfl_up_sync(mask, val, delta)    // 向上移位
```

**面试考点：**
- Warp Shuffle 比 Shared Memory 更快（无需显式加载/存储）
- 用于 Warp 级归约（如求最大值、求和）
- FlashAttention 中用 Warp Shuffle 进行 softmax 归约

### 11.6 Bank Conflict

**问题：**
```
Shared Memory 分成 32 个 Bank
如果同一个 Warp 的多个线程访问同一个 Bank 的不同地址 → 冲突
冲突导致串行访问，性能下降
```

**解决：**
```
方法1: Padding - 在数组末尾加一个元素
方法2: 调整访问模式 - 避免同一 Bank 冲突
```

**面试考点：**
- Bank Conflict 是 Shared Memory 性能的主要杀手
- 矩阵转置时容易出现 Bank Conflict
- Triton 编译器自动处理 Bank Conflict

---

## 12. 算子性能分析

### 12.1 Roofline 模型

**概念：**
```
性能 = min(计算能力, 带宽 × 算术强度)

算术强度 = FLOPs / Bytes (计算量 / 数据量)

如果 算术强度 < 瓦片比 (ridge point):
  → 访存密集 (Memory-bound)
  → 瓶颈在显存带宽
  → 优化方向: 减少数据搬运

如果 算术强度 > 瓦片比:
  → 计算密集 (Compute-bound)
  → 瓶颈在计算能力
  → 优化方向: 减少计算量
```

**面试考点：**
- Prefill 阶段：大矩阵乘法 → 计算密集
- Decode 阶段：向量×矩阵 → 访存密集
- KV Cache 读取：每步读取整个缓存 → 访存密集

### 12.2 Kernel 融合

**原理：**
```
未融合:
  kernel1: A → B (写回 Global Memory)
  kernel2: B → C (从 Global Memory 读取)
  总开销: kernel1 + kernel2 + 2次 Global Memory 访问

融合后:
  fused_kernel: A → C (中间结果在 Register/Shared Memory)
  总开销: fused_kernel + 0次额外 Global Memory 访问
```

**vLLM 中的融合示例：**
- Fused Add + RMSNorm
- Fused QKV Projection
- Fused Softmax + Sampling
- Reshape And Cache (融合 reshape 和写入)

### 12.3 内存合并访问

**好的访问模式（合并）：**
```
Thread 0: addr[0]
Thread 1: addr[1]
Thread 2: addr[2]
...
Thread 31: addr[31]
→ 一次内存事务
```

**坏的访问模式（分散）：**
```
Thread 0: addr[0]
Thread 1: addr[1000]
Thread 2: addr[2000]
...
→ 32次内存事务
```

**面试考点：**
- 合并访问是 GPU 性能的关键
- 矩阵按行存储时，按列访问会导致分散
- Triton 编译器自动优化访问模式

---

## 13. 面试真题解析

### 13.1 实现一个 Triton Softmax Kernel

```python
@triton.jit
def softmax_kernel(
    input_ptr, output_ptr,
    input_row_stride, output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # 每个 program 处理一行
    row_idx = tl.program_id(0)

    # 1. 计算该行的起始地址
    row_start_ptr = input_ptr + row_idx * input_row_stride

    # 2. 加载一行数据
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    mask = col_offsets < n_cols
    row = tl.load(input_ptrs, mask=mask, other=-float('inf'))

    # 3. 计算 softmax (在线算法)
    row_max = tl.max(row, axis=0)
    numerator = tl.exp(row - row_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator

    # 4. 存储结果
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=mask)
```

### 13.2 实现一个简单的 Attention Kernel

```python
@triton.jit
def attention_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    seq_len, head_dim,
    BLOCK_SIZE: tl.constexpr,
):
    # 简化版本：单头、单序列
    # 1. 加载 Q
    q = tl.load(Q_ptr + tl.arange(0, head_dim))

    # 2. 遍历 K, V 块
    acc = tl.zeros([head_dim], dtype=tl.float32)
    max_score = tl.full([1], -float('inf'), dtype=tl.float32)
    sum_exp = tl.zeros([1], dtype=tl.float32)

    for start in range(0, seq_len, BLOCK_SIZE):
        # 加载 K, V 块
        k = tl.load(K_ptr + start + tl.arange(0, BLOCK_SIZE))
        v = tl.load(V_ptr + start + tl.arange(0, BLOCK_SIZE))

        # 计算 Q×K^T
        score = tl.dot(q, k.T) / tl.sqrt(head_dim)

        # 在线 softmax
        new_max = tl.maximum(max_score, tl.max(score))
        exp_score = tl.exp(score - new_max)
        sum_exp = sum_exp * tl.exp(max_score - new_max) + tl.sum(exp_score)
        acc = acc * tl.exp(max_score - new_max) + tl.dot(exp_score, v)
        max_score = new_max

    # 3. 归一化
    output = acc / sum_exp
    tl.store(O_ptr, output)
```

### 13.3 解释 FlashAttention 的在线 Softmax

**面试回答：**

FlashAttention 使用在线 Softmax 算法，核心思想是维护运行最大值 M 和运行和 L：

```
初始化: M = -∞, L = 0, acc = 0

对于每个 KV 块 j:
  1. 计算局部注意力分数 S_j = Q × K_j^T
  2. 更新最大值: m_j = max(M, max(S_j))
  3. 重缩放因子: α = exp(M - m_j)
  4. 更新累积: acc = acc × α + exp(S_j - m_j) × V_j
  5. 更新和: L = L × α + sum(exp(S_j - m_j))
  6. 更新最大值: M = m_j

最终输出: acc / L
```

关键点：
- 不需要物化完整的 n×n 注意力矩阵
- 每个块独立计算，通过 M 和 L 合并
- 数值稳定：通过 max 防止指数溢出

### 13.4 解释 PagedAttention 的工作原理

**面试回答：**

PagedAttention 将 KV Cache 分成固定大小的块，通过块表间接寻址：

```
1. KV Cache 分成 16-token 的块
2. 每个请求有一个块表，映射逻辑块到物理块
3. slot_mapping 计算每个 token 的物理写入位置

例如: 请求有 50 个 token，block_size=16
  逻辑块: [0, 1, 2]
  物理块: [7, 3, 15]  (通过块表映射)
  slot_mapping[0] = 7*16 + 0 = 112
  slot_mapping[1] = 7*16 + 1 = 113
  ...
  slot_mapping[16] = 3*16 + 0 = 48
  ...
```

优势：
- 消除内存碎片：按需分配块
- 支持共享：多个请求可以共享相同前缀的块
- 减少浪费：短请求不需要预分配最大长度

### 13.5 解释 RoPE 的数学原理

**面试回答：**

RoPE 通过旋转矩阵将位置信息编码到 Q, K 中：

```
对于 Q 或 K 的每对相邻维度 (x, y):
  x' = x * cos(mθ) - y * sin(mθ)
  y' = x * sin(mθ) + y * cos(mθ)

其中 m 是位置，θ = 10000^(-2i/d)
```

等价于复数乘法：
```
(x + iy) × (cos(mθ) + i sin(mθ))
= (x cos(mθ) - y sin(mθ)) + i(x sin(mθ) + y cos(mθ))
```

关键性质：
- `<RoPE(q, m), RoPE(k, n)>` 只依赖于 `m-n`（相对位置）
- 不需要额外的位置编码参数
- 可以通过调整 θ 实现长度外推
