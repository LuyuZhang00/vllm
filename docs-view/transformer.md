这张图是经典 **Transformer Encoder-Decoder 架构**，来自 “Attention Is All You Need” 这类原始 Transformer 结构。它不是现在 GPT/LLaMA/Qwen 这类 **Decoder-only 大语言模型**的完整结构，但它是理解所有 Transformer 的基础。

整体可以先这样概括：

> Transformer 由 **Encoder 编码器** 和 **Decoder 解码器** 组成。Encoder 负责理解输入序列，把输入变成上下文表示；Decoder 负责根据已经生成的输出和 Encoder 的上下文表示，逐步生成目标序列。核心组件包括 Embedding、位置编码、Multi-Head Attention、Masked Multi-Head Attention、Feed Forward、残差连接、LayerNorm、Linear 和 Softmax。

---

# 1. 整体架构先看懂

图左边是 **Encoder**：

```text
Inputs
  ↓
Input Embedding
  ↓
Positional Encoding
  ↓
N 层 Encoder Block
  ↓
Encoder 输出 memory
```

图右边是 **Decoder**：

```text
Outputs shifted right
  ↓
Output Embedding
  ↓
Positional Encoding
  ↓
N 层 Decoder Block
  ↓
Linear
  ↓
Softmax
  ↓
Output Probabilities
```

Encoder 和 Decoder 之间有一条连接线，表示：

```text
Decoder 的 Cross-Attention 会使用 Encoder 的输出
```

也就是说：

```text
Encoder 负责理解输入
Decoder 负责基于输入生成输出
```

例如机器翻译：

```text
输入：I love you
输出：我 爱 你
```

Encoder 看英文句子，Decoder 逐步生成中文句子。

---

# 2. Inputs：输入序列

图中最下面左边的：

```text
Inputs
```

指原始输入文本经过 tokenizer 后得到的 token id 序列。

比如：

```text
"I love you"
```

经过 tokenizer 后变成：

```text
[101, 2345, 6789, 102]
```

模型不能直接处理字符串，只能处理数字 id。

面试官可能问：

**为什么要 tokenize？**

可以答：

> 因为神经网络只能处理数值张量，不能直接处理字符串。Tokenizer 把文本切成 token，并映射成 token id，后续再通过 embedding 变成连续向量。

---

# 3. Input Embedding：输入嵌入层

图中左下角：

```text
Input Embedding
```

作用是：

> 把 token id 映射成 dense vector，也就是连续向量。

如果输入是：

```text
[101, 2345, 6789]
```

Embedding 后变成：

```text
[3, d_model]
```

如果有 batch：

```text
[batch_size, seq_len, d_model]
```

例如：

```text
batch_size = 2
seq_len = 128
d_model = 4096
```

Embedding 输出就是：

```text
[2, 128, 4096]
```

Embedding 本质上是一个查表矩阵：

```text
Embedding Table: [vocab_size, d_model]
```

每个 token id 对应其中一行。

面试官可能问：

**Embedding 是训练出来的吗？**

答：

> 是的，Embedding 是模型参数的一部分，训练过程中通过反向传播学习出来。它把离散 token 映射到连续语义空间。

---

# 4. Positional Encoding：位置编码

图中 Input Embedding 后面有一个加号和波浪符号：

```text
Input Embedding + Positional Encoding
```

Transformer 的 Attention 本身不感知顺序。如果没有位置编码：

```text
我 爱 你
你 爱 我
```

模型可能很难区分，因为 token 集合类似，只是顺序不同。

位置编码的作用是：

> 给 token 注入位置信息，让模型知道每个 token 在序列中的位置。

原始 Transformer 使用的是 sinusoidal positional encoding，也就是正弦余弦位置编码：

```text
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

然后和 embedding 相加：

```text
x = token_embedding + positional_encoding
```

现代大模型更多使用 RoPE：

```text
对 Q/K 做旋转位置编码
```

面试官可能问：

**为什么位置编码不是 concat，而是 add？**

答：

> 因为 embedding 和 position encoding 维度相同，直接相加可以保持 hidden size 不变，后续网络结构不需要改变。模型可以学习同时利用 token 语义和位置信息。

---

# 5. Encoder Block：编码器层

图左边大框表示：

```text
N × Encoder Layer
```

一个 Encoder Layer 包含：

```text
Multi-Head Self-Attention
Add & Norm
Feed Forward
Add & Norm
```

完整顺序是：

```text
x
 ↓
Multi-Head Self-Attention
 ↓
Residual Add + LayerNorm
 ↓
Feed Forward
 ↓
Residual Add + LayerNorm
```

图中写的是原始 Transformer 的 **Post-LN** 结构：

```text
Sublayer → Add & Norm
```

现代大模型更多使用 **Pre-LN**：

```text
Norm → Sublayer → Add
```

面试时可以主动补一句：

> 原始 Transformer 图里是 Post-LN，现在很多 LLM 使用 Pre-LN 或 RMSNorm，训练更稳定。

---

# 6. Multi-Head Attention：多头自注意力

Encoder 里的：

```text
Multi-Head Attention
```

是 **Self-Attention**，因为 Q、K、V 都来自同一个输入序列。

## 6.1 Attention 的核心公式

Attention 公式是：

```text
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
```

含义是：

```text
Q：Query，我要查什么
K：Key，被查对象的索引
V：Value，被查对象的内容
```

对于每个 token，它会拿自己的 Q 去和所有 token 的 K 做相似度，得到 attention score，然后根据 score 加权求和 V。

通俗理解：

> 每个 token 会动态决定自己应该关注序列中的哪些 token。

例如句子：

```text
The animal didn't cross the street because it was tired.
```

模型要理解 `it` 指代 `animal`，Attention 可以让 `it` 关注到 `animal`。

---

## 6.2 Q、K、V 怎么来的？

输入 hidden states 是：

```text
X: [batch, seq_len, d_model]
```

通过三个线性层得到：

```text
Q = X Wq
K = X Wk
V = X Wv
```

其中：

```text
Wq: [d_model, d_model]
Wk: [d_model, d_model]
Wv: [d_model, d_model]
```

如果是多头，会再 reshape：

```text
Q/K/V: [batch, num_heads, seq_len, head_dim]
```

其中：

```text
head_dim = d_model / num_heads
```

---

## 6.3 为什么要除以 sqrt(d_k)？

Attention score 是：

```text
QK^T
```

如果 `d_k` 很大，点积结果方差会变大，softmax 容易进入饱和区，梯度变小。

所以除以：

```text
sqrt(d_k)
```

让数值更稳定。

面试官问：

**为什么不是除以 d_k？**

答：

> 因为点积的方差和维度 d_k 成正比，标准差和 sqrt(d_k) 成正比，所以除以 sqrt(d_k) 可以把方差稳定到合理范围。

---

## 6.4 为什么要 Multi-Head？

单头 Attention 只能在一个表示子空间里关注关系。

Multi-Head 是把 hidden 维度拆成多个 head：

```text
head1 关注语法关系
head2 关注指代关系
head3 关注局部关系
head4 关注长距离依赖
```

数学上：

```text
head_i = Attention(Q_i, K_i, V_i)
MultiHead = Concat(head_1, ..., head_h) Wo
```

优势是：

> 让模型从多个子空间并行学习不同类型的 token 关系。

---

## 6.5 Encoder Self-Attention 有没有 Mask？

原始 Encoder Self-Attention 通常没有 causal mask。

因为 Encoder 是双向理解输入：

```text
每个 token 可以看到输入序列中的所有 token
```

例如翻译任务中，英文输入已经完整给定，所以每个词可以看前后文。

---

# 7. Add & Norm：残差连接 + 归一化

图里每个子层后面都有：

```text
Add & Norm
```

它包含两件事：

```text
Residual Connection
LayerNorm
```

原始 Transformer 中是：

```text
LayerNorm(x + Sublayer(x))
```

## 7.1 Add 是什么？

Add 是残差连接：

```text
x + Sublayer(x)
```

作用是：

1. 缓解深层网络梯度消失；
2. 保留原始输入信息；
3. 让模型更容易训练；
4. 如果子层暂时学不好，至少可以走 identity path。

面试官可能问：

**残差连接为什么有用？**

答：

> 残差连接给梯度提供了一条直接路径，缓解深层网络训练困难，同时保留原始 token 表示，使网络学习的是增量变化而不是完全重构表示。

---

## 7.2 Norm 是什么？

这里原始图是 LayerNorm。

LayerNorm 对每个 token 的 hidden dimension 做归一化：

```text
LayerNorm(x) = (x - mean) / sqrt(var + eps) * gamma + beta
```

作用是：

1. 稳定激活分布；
2. 加速训练收敛；
3. 防止层数加深后数值爆炸或消失。

现代 LLM 常用 RMSNorm：

```text
RMSNorm(x) = x / sqrt(mean(x^2) + eps) * gamma
```

RMSNorm 去掉了减均值，计算更简单。

---

## 7.3 Pre-LN 和 Post-LN 区别

图中是 Post-LN：

```text
x → Sublayer → Add → Norm
```

现代 LLM 常用 Pre-LN：

```text
x → Norm → Sublayer → Add
```

Post-LN 原始 Transformer 使用，但深层模型训练可能不稳定。

Pre-LN 的优点是：

> 梯度路径更稳定，适合更深的大模型。

面试官如果问：

**现在大模型是这个结构吗？**

答：

> 图是原始 Transformer 的 Encoder-Decoder 结构，且是 Post-LN。现在 GPT/LLaMA/Qwen 等 decoder-only LLM 通常使用 Pre-LN + RMSNorm + RoPE + SwiGLU/SiLU MLP。

---

# 8. Feed Forward：前馈网络 FFN

Encoder 和 Decoder 里都有：

```text
Feed Forward
```

也叫：

```text
FFN / MLP
```

它对每个 token 独立做非线性变换。

原始 Transformer FFN 是：

```text
FFN(x) = max(0, xW1 + b1) W2 + b2
```

即：

```text
d_model → d_ff → d_model
```

例如：

```text
4096 → 11008 → 4096
```

它的作用是：

> Attention 负责 token 之间的信息交互，FFN 负责对每个 token 的表示做非线性特征变换。

面试官可能问：

**Attention 和 FFN 分别负责什么？**

答：

> Attention 负责跨 token 信息混合，FFN 负责 token 内部特征变换和非线性表达能力。

---

## 8.1 为什么 FFN 是逐 token 的？

FFN 不混合不同 token。

输入 shape：

```text
[batch, seq_len, d_model]
```

FFN 对每个 token 的 `d_model` 维向量独立做 MLP：

```text
[batch, seq_len, d_model] → [batch, seq_len, d_model]
```

token 之间的信息已经通过 Attention 混合了。

---

## 8.2 现代 LLM 的 FFN 有什么变化？

原始 Transformer 使用 ReLU FFN。

现代 LLM 常用：

```text
SwiGLU / GeGLU / SiLU
```

例如 LLaMA 的 MLP：

```text
down_proj( SiLU(gate_proj(x)) * up_proj(x) )
```

MoE 模型则把 FFN 替换成多个 expert：

```text
每个 token 只经过 top-k experts
```

---

# 9. Decoder：解码器层

图右边是 Decoder。

Decoder 每层包含三个主要子层：

```text
Masked Multi-Head Self-Attention
Add & Norm
Cross-Attention
Add & Norm
Feed Forward
Add & Norm
```

比 Encoder 多了一个：

```text
Encoder-Decoder Attention / Cross-Attention
```

---

# 10. Outputs shifted right：右移输出

图右下角：

```text
Outputs (shifted right)
```

这是训练阶段 Decoder 的输入。

假设目标句子是：

```text
我 爱 你 <eos>
```

训练时 Decoder 输入是右移后的：

```text
<bos> 我 爱 你
```

目标输出是：

```text
我 爱 你 <eos>
```

也就是：

```text
输入第 t 个位置时，只能预测第 t+1 个 token
```

这种方式叫 teacher forcing。

面试官可能问：

**为什么要 shifted right？**

答：

> 因为 Decoder 是自回归生成，训练时第 t 个位置只能基于之前 token 预测下一个 token。右移目标序列可以让模型学习 next-token prediction，同时避免看到当前要预测的 token。

---

# 11. Output Embedding：输出嵌入

Decoder 的输入也要经过 embedding：

```text
Output Embedding
```

它把目标语言 token id 转成向量。

在机器翻译里：

```text
Encoder 输入是英文 token
Decoder 输入是中文 token
```

所以会有：

```text
Input Embedding
Output Embedding
```

两者可以共享，也可以不共享。

在 decoder-only LLM 中，只有一个 token embedding。

---

# 12. Decoder Positional Encoding

Decoder 的输出 token 也需要位置信息。

所以：

```text
Output Embedding + Positional Encoding
```

这让 Decoder 知道当前生成到第几个位置。

现代 LLM 里通常还是使用 RoPE，而不是图里的 sinusoidal PE。

---

# 13. Masked Multi-Head Attention：带 Mask 的自注意力

Decoder 第一层是：

```text
Masked Multi-Head Attention
```

它和 Encoder Self-Attention 的区别是：

> Decoder 自注意力不能看到未来 token。

比如生成到：

```text
我 爱
```

模型不能提前看到：

```text
你
```

否则训练时会作弊。

所以要加 causal mask：

```text
位置 i 只能关注位置 <= i 的 token
```

attention mask 大概是下三角矩阵：

```text
1 0 0 0
1 1 0 0
1 1 1 0
1 1 1 1
```

面试官可能问：

**为什么 Decoder 要 Mask，Encoder 不需要？**

答：

> Encoder 是双向理解完整输入，可以看全句；Decoder 是自回归生成，预测当前位置时不能看未来 token，所以必须使用 causal mask。

---

# 14. Decoder 的 Cross-Attention

图右侧中间的：

```text
Multi-Head Attention
```

它不是普通 self-attention，而是 **Encoder-Decoder Attention / Cross-Attention**。

这里：

```text
Q 来自 Decoder 当前 hidden states
K/V 来自 Encoder 输出
```

也就是：

```text
Q = decoder_hidden Wq
K = encoder_output Wk
V = encoder_output Wv
```

作用是：

> Decoder 在生成每个 token 时，去关注 Encoder 对输入序列的表示。

比如翻译时生成中文“爱”，Decoder 可以关注英文输入里的 “love”。

面试官可能问：

**Self-Attention 和 Cross-Attention 区别？**

答：

> Self-Attention 的 Q/K/V 来自同一个序列；Cross-Attention 的 Q 来自 Decoder，K/V 来自 Encoder 输出，用来让 Decoder 对输入序列进行条件生成。

---

# 15. Decoder Feed Forward

Decoder 中的 FFN 和 Encoder 中一样：

```text
对每个 token 独立做 MLP 非线性变换
```

不过输入已经融合了：

```text
已生成 token 上下文 + encoder 输入信息
```

---

# 16. N×：堆叠 N 层

图中 Encoder 和 Decoder 两侧都有：

```text
N×
```

表示同样的 block 堆叠 N 次。

原始 Transformer Base：

```text
N = 6
d_model = 512
num_heads = 8
d_ff = 2048
```

现代 LLM：

```text
N = 32 / 80 / 120 ...
d_model = 4096 / 8192 ...
num_heads = 32 / 64 / 128 ...
```

为什么要堆叠多层？

> 低层学习局部和词法特征，中高层学习语法、语义、推理和任务相关表示。

---

# 17. Linear：线性输出层

Decoder 最上方：

```text
Linear
```

它把最后一层 hidden states 映射到词表大小。

如果 hidden states 是：

```text
[batch, seq_len, d_model]
```

Linear 后是：

```text
[batch, seq_len, vocab_size]
```

例如：

```text
d_model = 4096
vocab_size = 100000
```

Linear 权重：

```text
[d_model, vocab_size]
```

输出叫：

```text
logits
```

每个位置都有对整个词表的打分。

面试官可能问：

**Linear 和 Embedding 有什么关系？**

答：

> Embedding 是 token id 到 hidden vector，Linear/lm_head 是 hidden vector 到 vocab logits。很多模型会使用 weight tying，让输入 embedding 和输出 lm_head 共享权重。

---

# 18. Softmax：概率归一化

Linear 输出 logits 后，经过：

```text
Softmax
```

把 logits 转成概率分布：

```text
p_i = exp(logit_i) / sum_j exp(logit_j)
```

输出：

```text
Output Probabilities
```

也就是每个 token 成为下一个 token 的概率。

例如：

```text
"我爱" 后面：
你: 0.45
她: 0.12
这个: 0.08
...
```

推理时还会接采样策略：

```text
Greedy
Top-k
Top-p
Temperature
Beam Search
```

训练时则用 softmax 后的 cross entropy loss。

---

# 19. Output Probabilities：输出概率

图最上面：

```text
Output Probabilities
```

表示词表上每个 token 的概率。

训练时：

```text
用真实下一个 token 计算 cross entropy loss
```

推理时：

```text
根据概率选择下一个 token
```

生成过程是自回归的：

```text
生成 token1
把 token1 拼回输入
生成 token2
把 token2 拼回输入
...
直到 eos 或 max_tokens
```

---

# 20. Encoder-Decoder 训练流程

以翻译为例：

```text
输入：I love you
目标：我 爱 你
```

训练时：

Encoder 输入：

```text
I love you
```

Decoder 输入：

```text
<bos> 我 爱
```

Decoder 目标：

```text
我 爱 你
```

每个位置都预测下一个 token。

Loss 是：

```text
CrossEntropy(logits, target_token)
```

---

# 21. Encoder-Decoder 推理流程

推理时没有目标输出，所以 Decoder 要一步步生成。

1. Encoder 先处理完整输入：

```text
encoder_memory = Encoder(input_tokens)
```

2. Decoder 从 `<bos>` 开始：

```text
decoder_input = <bos>
```

3. 生成第一个 token：

```text
我
```

4. 拼回去：

```text
<bos> 我
```

5. 再生成：

```text
爱
```

6. 直到生成 `<eos>`。

---

# 22. 这个结构和 GPT/LLaMA/Qwen 有什么区别？

这张图是 **Encoder-Decoder Transformer**。

GPT/LLaMA/Qwen 是 **Decoder-only Transformer**。

Decoder-only 没有 Encoder，也没有 Cross-Attention。

结构类似：

```text
Token Embedding
RoPE
N × Decoder Block:
  Masked Self-Attention
  MLP
Final Norm
LM Head
Softmax
```

也就是说现代 LLM 通常是：

```text
只有右边 Decoder 的一部分
```

但也有区别：

| 模型类型            | 结构          | 典型模型                       |
| --------------- | ----------- | -------------------------- |
| Encoder-only    | 只编码理解       | BERT                       |
| Encoder-Decoder | 输入理解 + 输出生成 | T5, BART, 原始 Transformer   |
| Decoder-only    | 自回归生成       | GPT, LLaMA, Qwen, DeepSeek |

面试官可能问：

**为什么现在 LLM 多用 Decoder-only？**

答：

> Decoder-only 结构简单，天然适合 next-token prediction，可以统一预训练和生成任务，扩展到大规模数据和参数时效果很好。Encoder-Decoder 更适合条件生成、翻译、摘要等 seq2seq 任务。

---

# 23. 面试高频细节问题

## 问题 1：Q、K、V 分别是什么？

答：

> Q 是 query，表示当前位置想找什么信息；K 是 key，表示每个位置能被匹配的索引；V 是 value，表示被聚合的内容。Attention 用 Q 和 K 算权重，再用权重加权 V。

---

## 问题 2：为什么 Attention 复杂度是 O(n²)？

答：

> 因为每个 token 都要和所有 token 计算 QK 相似度。如果序列长度是 n，attention score 矩阵大小是 n×n，所以时间和显存复杂度是 O(n²)。

---

## 问题 3：为什么 Decode 阶段可以用 KVCache？

答：

> 自回归生成时，历史 token 的 K/V 不会变化。每生成一个新 token，只需要计算新 token 的 Q/K/V，然后把新的 K/V 追加到 cache 中。历史 K/V 可以复用，不需要每步重新计算。

---

## 问题 4：KVCache 缓存的是什么？

答：

> 缓存每一层 Attention 中历史 token 的 Key 和 Value。Decode 时新 token 的 Query 会和历史 Key 做 attention，再聚合历史 Value。

---

## 问题 5：为什么不缓存 Q？

答：

> Q 只用于当前位置作为查询，历史 token 的 Q 后续不会再被用来和新 token 计算 attention。新 token 只需要自己的 Q 和历史 K/V，所以缓存 K/V 即可。

---

## 问题 6：Multi-Head Attention 拼接后为什么还要一个输出线性层 Wo？

答：

> 多个 head 的输出只是拼接在一起，还没有充分融合。Wo 用来把不同 head 的信息重新线性组合，映射回 d_model 维度。

---

## 问题 7：Attention 和 MLP 哪个参数多？

答：

> 通常 MLP 参数更多。Attention 主要是 Q/K/V/O 四个 d_model×d_model 矩阵，大约 4d²。MLP 如果是 d_model → 4d_model → d_model，大约 8d²。现代 SwiGLU MLP 也通常占大头。MoE 模型中 expert MLP 参数更多。

---

## 问题 8：LayerNorm 放在 attention 前还是后？

答：

> 原始 Transformer 是 Post-LN，即 sublayer 后 Add & Norm。现代大模型多用 Pre-LN，即 Norm → Attention/MLP → Residual Add。Pre-LN 深层训练更稳定。

---

## 问题 9：为什么 Decoder 的第一层 attention 叫 Masked？

答：

> 因为自回归生成不能看未来 token。训练时虽然完整目标序列都在，但要用 causal mask 防止当前位置看到后面的真实 token。

---

## 问题 10：为什么 Cross-Attention 的 K/V 来自 Encoder？

答：

> 因为 Decoder 需要根据输入序列生成输出。Q 来自 Decoder 当前生成状态，K/V 来自 Encoder 对输入的表示，这样 Decoder 每一步都可以关注输入序列相关部分。

---

# 24. 用一句话总结每个组件

| 组件                          | 作用                          |
| --------------------------- | --------------------------- |
| Inputs                      | 原始输入 token id               |
| Input Embedding             | token id → 向量               |
| Positional Encoding         | 注入位置信息                      |
| Multi-Head Attention        | 让 token 之间相互关注              |
| Add                         | 残差连接，保留原信息                  |
| Norm                        | 稳定数值和训练                     |
| Feed Forward                | 对每个 token 做非线性变换            |
| Outputs shifted right       | Decoder 训练时右移目标序列           |
| Output Embedding            | 目标 token id → 向量            |
| Masked Multi-Head Attention | 自回归，只看历史不看未来                |
| Cross-Attention             | Decoder 关注 Encoder 输入表示     |
| Linear                      | hidden state → vocab logits |
| Softmax                     | logits → 概率                 |
| Output Probabilities        | 下一个 token 的概率分布             |
| N×                          | 多层堆叠提升表达能力                  |

---

# 25. 面试版完整回答

你可以这样说：

> 这张图是经典 Encoder-Decoder Transformer。左边是 Encoder，负责把输入序列编码成上下文表示；右边是 Decoder，负责基于已生成 token 和 Encoder 输出逐步生成目标序列。
>
> 输入文本首先经过 tokenizer 得到 token id，然后通过 Input Embedding 映射成 d_model 维向量。由于 Attention 本身不感知顺序，所以还要加 Positional Encoding。Encoder 每层由 Multi-Head Self-Attention、残差连接、LayerNorm 和 Feed Forward 组成。Self-Attention 让每个 token 关注输入序列中的其他 token，FFN 对每个 token 做非线性特征变换，Add & Norm 用来稳定训练和保留残差信息。
>
> Decoder 也有 embedding 和位置编码，但它的第一层 attention 是 Masked Self-Attention，因为自回归生成不能看到未来 token。Decoder 中间还有 Cross-Attention，Q 来自 Decoder，K/V 来自 Encoder 输出，用来在生成时关注输入序列。最后 Decoder 输出经过 Linear 映射到词表大小，再经过 Softmax 得到下一个 token 的概率。
>
> 原始图中是 Encoder-Decoder 结构，适合翻译、摘要等 seq2seq 任务。现代 GPT/LLaMA/Qwen 这类大语言模型通常是 Decoder-only，去掉 Encoder 和 Cross-Attention，只保留 masked self-attention、MLP、残差和归一化，并通常使用 Pre-LN、RMSNorm 和 RoPE。

---

# 26. 最后给你一个背诵框架

面试官让你讲这张图，你可以按这个顺序讲：

```text
1. 这是 Encoder-Decoder Transformer
2. 输入先 Tokenize，再 Embedding
3. 加 Positional Encoding 解决顺序问题
4. Encoder Block = Self-Attention + FFN + Add&Norm
5. Decoder Block = Masked Self-Attention + Cross-Attention + FFN
6. Linear + Softmax 输出下一个 token 概率
7. 训练时输出 shifted right，推理时自回归生成
8. 现代 LLM 多是 Decoder-only，使用 RoPE/RMSNorm/SwiGLU/KVCache
```

这一套讲下来，面试官继续往下问 QKV、mask、残差、LayerNorm、KVCache、RoPE、LLM 结构演变，你都能接住。
