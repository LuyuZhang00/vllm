# vLLM 模型适配指南：以 DeepSeek V4 为例

> 本文档详细说明如何在 vLLM 中添加新模型适配，以 DeepSeek V4 为例，覆盖模型注册、模型实现、注意力后端、KV Cache 规格、量化配置等所有需要修改的模块。

---

## 目录

- [1. 总览：需要修改的模块](#1-总览需要修改的模块)
- [2. 模型注册](#2-模型注册)
- [3. 模型实现](#3-模型实现)
- [4. 注意力后端适配](#4-注意力后端适配)
- [5. KV Cache 规格定义](#5-kv-cache-规格定义)
- [6. 量化配置](#6-量化配置)
- [7. 其他适配模块](#7-其他适配模块)
- [8. DeepSeek V4 完整文件清单](#8-deepseek-v4-完整文件清单)
- [9. 关键类和函数清单](#9-关键类和函数清单)

---

## 1. 总览：需要修改的模块

### 1.1 适配层次图

```
┌─────────────────────────────────────────────────────────────────┐
│                    模型适配层次                                    │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Layer 1: 模型注册 (registry.py)                         │    │
│  │  让 vLLM 知道 HF 架构名 → vLLM 模型类的映射               │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Layer 2: 模型实现 (model.py)                            │    │
│  │  ForCausalLM / Model / DecoderLayer / Attention / MoE    │    │
│  │  实现 forward(), compute_logits(), load_weights()        │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Layer 3: 注意力后端 (attention backends)                 │    │
│  │  自定义注意力 kernel (如 FlashMLA)                        │    │
│  │  实现 AttentionBackend + AttentionImpl                    │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Layer 4: KV Cache 规格 (kv_cache_interface.py)          │    │
│  │  定义 KV Cache 的形状、大小、量化模式                      │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Layer 5: 辅助模块                                       │    │
│  │  量化配置 / 分词器 / 渲染器 / 工具解析器 / 推理解析器      │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 必须修改 vs 可选修改

| 模块 | 必须？ | 说明 |
|------|--------|------|
| `registry.py` 注册 | ✅ 必须 | 让 vLLM 找到模型类 |
| 模型实现 | ✅ 必须 | `ForCausalLM` + `Model` + `DecoderLayer` |
| `load_weights()` | ✅ 必须 | 加载模型权重 |
| KV Cache 规格 | ✅ 必须 | 定义注意力层的缓存格式 |
| 注意力后端 | 可选 | 使用标准注意力则不需要 |
| 量化配置 | 可选 | 使用标准量化则不需要 |
| 分词器 | 可选 | 使用标准 HF 分词器则不需要 |
| 渲染器 | 可选 | 使用标准聊天模板则不需要 |

---

## 2. 模型注册

### 2.1 注册文件

**文件：** `vllm/model_executor/models/registry.py`

### 2.2 注册字典

```python
# 文本生成模型 (line 71)
_TEXT_GENERATION_MODELS = {
    "DeepseekV4ForCausalLM": ("vllm.models.deepseek_v4", "DeepseekV4ForCausalLM"),
    # ...
}

# 多模态模型 (line 343)
_MULTIMODAL_MODELS = { ... }

# 推测解码模型 (line 591)
_SPECULATIVE_DECODING_MODELS = {
    "DeepSeekV4MTPModel": ("vllm.models.deepseek_v4", "DeepSeekV4MTP"),
    # ...
}
```

### 2.3 注册格式

```python
"<HF_ArchitectureClassName>": ("<module_path>", "<vllm_class_name>"),
```

- `HF_ArchitectureClassName`：HuggingFace config 中的 `architectures` 字段值
- `module_path`：vLLM 中的 Python 模块路径
- `vllm_class_name`：vLLM 中的类名

### 2.4 新旧目录结构

```
旧结构 (标准模型):
  vllm/model_executor/models/llama.py
  注册: "LlamaForCausalLM": ("llama", "LlamaForCausalLM")
  解析: 自动加前缀 vllm.model_executor.models.

新结构 (硬件隔离模型，如 DeepSeek V4):
  vllm/models/deepseek_v4/nvidia/model.py
  注册: "DeepseekV4ForCausalLM": ("vllm.models.deepseek_v4", "DeepseekV4ForCausalLM")
  解析: 以 vllm. 开头，直接使用
```

### 2.5 惰性加载机制

```python
# registry.py, line 810
class _LazyRegisteredModel:
    def load_model_cls(self) -> type[nn.Module]:
        # 不在主进程导入，避免 CUDA 初始化
        mod = importlib.import_module(self.module_name)
        return getattr(mod, self.class_name)

    def inspect_model_cls(self) -> _ModelInfo:
        # 在子进程中检查模型能力
        # 结果缓存到磁盘 JSON
```

---

## 3. 模型实现

### 3.1 模型类层次

```
DeepseekV4ForCausalLM (顶层)
    ├── DeepseekV4Model (内部模型)
    │    ├── VocabParallelEmbedding
    │    ├── DeepseekV4DecoderLayer × N
    │    │    ├── DeepseekV4Attention (MLA)
    │    │    └── DeepseekV4MoE
    │    └── RMSNorm
    ├── ParallelLMHead
    └── LogitsProcessor
```

### 3.2 ForCausalLM 类（必须实现）

**文件：** `vllm/models/deepseek_v4/nvidia/model.py`

```python
class DeepseekV4ForCausalLM(nn.Module, SupportsPP):
    """DeepSeek V4 模型的顶层类"""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.model = DeepseekV4Model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )
        self.lm_head = ParallelLMHead(...)
        self.logits_processor = LogitsProcessor(...)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | IntermediateTensors:
        """模型前向传播"""
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """计算 logits"""
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        """加载模型权重"""
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
```

**关键点：**
- 构造函数必须接受 `vllm_config: VllmConfig` 和 `prefix: str`
- `forward()` 返回 `hidden_states` 或 `IntermediateTensors` (PP 中间传递)
- `compute_logits()` 单独实现，与 forward 分离（支持 execute/sample 分离）
- `load_weights()` 使用 `AutoWeightsLoader` 自动处理权重映射

### 3.3 Model 类（内部模型）

```python
class DeepseekV4Model(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str):
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(...)
        self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: DeepseekV4DecoderLayer(
                vllm_config=vllm_config, prefix=prefix
            ),
            prefix=maybe_prefix(prefix, "layers"),
        )
        self.norm = RMSNorm(...)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """嵌入 token IDs"""
        return self.embed_tokens(input_ids)

    def make_empty_intermediate_tensors(self, ...):
        """创建空中间张量 (用于 PP)"""
        ...

    def forward(self, input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs):
        """模型前向传播"""
        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)
        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = layer(hidden_states, ...)
        hidden_states = self.norm(hidden_states)
        return hidden_states
```

### 3.4 DecoderLayer 类

```python
class DeepseekV4DecoderLayer(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str):
        super().__init__()
        self.self_attn = DeepseekV4Attention(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "self_attn")
        )
        self.mlp = DeepseekV4MoE(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "mlp")
        )
        self.input_layernorm = RMSNorm(...)
        self.post_attention_layernorm = RMSNorm(...)

    def forward(self, hidden_states, ...):
        # 1. Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, ...)
        hidden_states = residual + hidden_states

        # 2. MoE
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, ...)
        hidden_states = residual + hidden_states

        return hidden_states
```

### 3.5 Attention 类（MLA 特殊实现）

```python
class DeepseekV4Attention(nn.Module):
    """DeepSeek V4 的 MLA 注意力层"""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str):
        super().__init__()
        config = vllm_config.model_config.hf_config

        # MLA 核心维度
        self.kv_lora_rank = config.kv_lora_rank        # 512
        self.qk_rope_head_dim = config.qk_rope_head_dim  # 64
        self.qk_nope_head_dim = config.qk_nope_head_dim  # 128
        self.v_head_dim = config.v_head_dim              # 128

        # KV 压缩投影
        self.fused_wqa_wkv = FusedQKVParallelLinear(...)  # 融合 Q/KV 投影
        self.kv_norm = RMSNorm(self.kv_lora_rank, ...)
        self.wq_b = ColumnParallelLinear(...)  # Q 解压

        # 注意力包装器
        self.mla_wrapper = DeepseekV4MultiHeadLatentAttentionWrapper(...)

    def get_kv_cache_spec(self, vllm_config) -> KVCacheSpec:
        """声明 KV Cache 规格"""
        return MLAAttentionSpec(
            block_size=self.cache_config.block_size,
            num_kv_heads=1,
            head_size=self.kv_lora_rank + self.qk_rope_head_dim,  # 576
            dtype=self.dtype,
        )

    def forward(self, hidden_states, ...):
        # 1. Q/KV 压缩投影
        q, kv_c, k_pe = self.fused_wqa_wkv(hidden_states)

        # 2. KV 归一化
        kv_c = self.kv_norm(kv_c)

        # 3. Q 解压
        q = self.wq_b(q)

        # 4. MLA 注意力计算
        output = self.mla_wrapper(q, kv_c, k_pe, ...)

        # 5. 输出投影
        output = self.wo(output)
        return output
```

### 3.6 MoE 类

```python
class DeepseekV4MoE(nn.Module):
    """DeepSeek V4 的 MoE 层"""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str):
        super().__init__()
        config = vllm_config.model_config.hf_config

        # Gate 网络
        self.gate = GateLinear(...)

        # 专家 (FusedMoE 或 MegaMoE)
        self.experts = FusedMoE(
            num_experts=config.n_routed_experts,  # 384 (DeepSeek V4 Pro)
            top_k=config.num_experts_per_tok,      # 8
            ...
        )

        # 共享专家
        self.shared_experts = DeepseekV4MLP(...)

    def forward(self, hidden_states, ...):
        # 1. Gate 路由
        router_logits = self.gate(hidden_states)
        topk_weights, topk_indices = grouped_topk(router_logits, ...)

        # 2. 专家计算
        expert_output = self.experts(hidden_states, topk_weights, topk_indices)

        # 3. 共享专家
        shared_output = self.shared_experts(hidden_states)

        # 4. 合并
        return expert_output + shared_output
```

### 3.7 load_weights() 实现

```python
class DeepseekV4ForCausalLM(nn.Module):
    # 权重名称映射
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_substr={
            "model.layers.{}.self_attn.wq_a.": "model.layers.{}.self_attn.fused_wqa_wkv.q_proj.",
            "model.layers.{}.self_attn.wkv_a.": "model.layers.{}.self_attn.fused_wqa_wkv.kv_proj.",
            # ...
        }
    )

    def load_weights(self, weights):
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
```

---

## 4. 注意力后端适配

### 4.1 何时需要自定义注意力后端

| 场景 | 需要自定义后端？ |
|------|----------------|
| 标准 MHA/MQA/GQA | ❌ 使用 FlashAttention/FlashInfer |
| MLA (Multi-head Latent Attention) | ✅ 需要 MLA 后端 |
| 稀疏注意力 | ✅ 需要 Sparse 后端 |
| 自定义 kernel | ✅ 需要自定义后端 |

### 4.2 MLA 注意力后端目录

```
vllm/v1/attention/backends/mla/
├── flashmla.py              # FlashMLA (Hopper/Blackwell)
├── flashmla_sparse.py       # FlashMLA Sparse (V3.2/V4)
├── flashattn_mla.py         # FlashAttention MLA
├── triton_mla.py            # Triton MLA (通用)
├── cutlass_mla.py           # CUTLASS MLA
├── flashinfer_mla.py        # FlashInfer MLA
├── rocm_aiter_mla.py        # ROCm AITER MLA
├── tokenspeed_mla.py        # TokenSpeed MLA
├── indexer.py               # 稀疏注意力 Indexer
├── sparse_swa.py            # 稀疏滑动窗口注意力
└── prefill/                 # Prefill 专用变体
    ├── base.py
    ├── flashmla.py
    ├── flashmla_sparse.py
    └── ...
```

### 4.3 注意力后端接口

```python
# vllm/v1/attention/backend.py
class AttentionBackend(ABC):
    """注意力后端基类"""

    def get_builder_cls(self) -> type[AttentionMetadataBuilder]:
        """返回元数据构建器类"""
        ...

    def get_impl_cls(self) -> type[AttentionImpl]:
        """返回注意力实现类"""
        ...

class AttentionImpl(ABC):
    """注意力实现基类"""

    def forward(self, layer, query, key, value, kv_cache, attn_metadata, ...):
        """执行注意力计算"""
        ...

    def do_kv_cache_update(self, layer, key, value, kv_cache, slot_mapping):
        """更新 KV Cache"""
        ...
```

### 4.4 注意力后端选择

```python
# vllm/v1/attention/selector.py
def get_attn_backend(
    head_size: int,
    dtype: torch.dtype,
    kv_cache_dtype: CacheDType | None,
    block_size: int | None,
    use_mla: bool = False,
    use_sparse: bool = False,
    ...
) -> type[AttentionBackend]:
    # 根据平台和配置选择最佳后端
    config = AttentionSelectorConfig(...)
    return current_platform.get_attn_backend_cls(config)
```

### 4.5 DeepSeek V4 的注意力后端

```python
# vllm/models/deepseek_v4/__init__.py
if current_platform.is_rocm():
    from .amd.flashmla import DeepseekV4ROCMAiterMLASparseImpl
    # 使用 ROCm AITER MLA
else:
    from .nvidia.flashmla import DeepseekV4FlashMLASparseImpl
    # 使用 FlashMLA Sparse
```

---

## 5. KV Cache 规格定义

### 5.1 KVCacheSpec 层次

```
KVCacheSpec (基类)
    ├── AttentionSpec
    │    ├── FullAttentionSpec (标准全注意力)
    │    │    ├── MLAAttentionSpec (MLA)
    │    │    └── TQFullAttentionSpec (TurboQuant)
    │    ├── SlidingWindowSpec (滑动窗口)
    │    │    └── SlidingWindowMLASpec
    │    ├── ChunkedLocalAttentionSpec (分块局部注意力)
    │    ├── EncoderOnlyAttentionSpec (零内存)
    │    └── CrossAttentionSpec (交叉注意力)
    ├── MambaSpec (SSM)
    └── UniformTypeKVCacheSpecs (聚合多个同类型层)
```

### 5.2 MLAAttentionSpec 定义

```python
# vllm/v1/kv_cache_interface.py, line 337
@dataclass(frozen=True)
class MLAAttentionSpec(FullAttentionSpec):
    """Multi-head Latent Attention 的 KV Cache 规格"""

    # 压缩相关
    cache_dtype_str: str | None = None    # 缓存数据类型 (如 "fp8_ds_mla")
    alignment: int | None = None          # 对齐要求
    compress_ratio: int = 1               # 压缩比 (1, 4, 128)
    model_version: str | None = None      # 模型版本 (如 "deepseek_v4")

    @property
    def real_page_size_bytes(self) -> int:
        """实际页面大小（考虑压缩）"""
        if self.cache_dtype_str == "fp8_ds_mla":
            if self.model_version == "deepseek_v4":
                # 448B NoPE + 128B RoPE + 8B fp8 scale = 584 bytes
                return self.storage_block_size * 584
            else:
                # 512B NoPE + 16B scales + 128B RoPE = 656 bytes
                return self.storage_block_size * 656
        # 标准 fp16/bf16
        return self.storage_block_size * 1152  # 576 * 2
```

### 5.3 模型如何声明 KV Cache 规格

```python
# 在 Attention 类中实现 get_kv_cache_spec()
class DeepseekV4Attention(nn.Module):
    def get_kv_cache_spec(self, vllm_config) -> KVCacheSpec:
        return MLAAttentionSpec(
            block_size=self.cache_config.block_size,
            num_kv_heads=1,              # MLA 只有 1 个 KV head
            head_size=self.kv_lora_rank + self.qk_rope_head_dim,  # 576
            dtype=self.dtype,
            cache_dtype_str=self.cache_dtype_str,
            compress_ratio=self.compress_ratio,
            model_version="deepseek_v4",
        )
```

---

## 6. 量化配置

### 6.1 自定义量化配置

**文件：** `vllm/models/deepseek_v4/quant_config.py`

```python
class DeepseekV4FP8Config(Fp8Config):
    """DeepSeek V4 专用 FP8 量化配置"""

    @property
    def expert_dtype(self) -> str:
        """专家数据类型 (fp4 或 fp8)"""
        ...

    @property
    def is_scale_e8m0(self) -> bool:
        """是否使用 E8M0 缩放格式"""
        ...
```

### 6.2 注册量化方法

```python
# vllm/model_executor/layers/quantization/__init__.py
@register_quantization_config("deepseek_v4_fp8")
class DeepseekV4FP8Config(Fp8Config):
    ...
```

---

## 7. 其他适配模块

### 7.1 HuggingFace 配置

**文件：** `vllm/transformers_utils/configs/deepseek_v4.py`

```python
class DeepseekV4Config(PretrainedConfig):
    model_type = "deepseek_v4"

    def __init__(self, max_position_embeddings, rope_scaling, ...):
        super().__init__(**kwargs)
        self.max_position_embeddings = max_position_embeddings
        self.rope_scaling = rope_scaling
        ...
```

### 7.2 分词器（可选）

```
vllm/tokenizers/
├── deepseek_v32.py    # DeepSeek V32 分词器
└── deepseek_v4.py     # DeepSeek V4 分词器 (如果需要)
```

### 7.3 渲染器（可选）

```
vllm/renderers/
├── deepseek_v32.py    # DeepSeek V32 聊天模板渲染器
└── deepseek_v4.py     # DeepSeek V4 渲染器 (如果需要)
```

### 7.4 推理解析器（可选）

```
vllm/reasoning/
├── deepseek_r1.py     # DeepSeek R1 推理解析器
└── deepseek_v4.py     # DeepSeek V4 推理解析器 (如果需要)
```

---

## 8. DeepSeek V4 完整文件清单

### 8.1 核心文件

| # | 文件 | 用途 |
|---|------|------|
| 1 | `vllm/model_executor/models/registry.py` | 注册 `DeepseekV4ForCausalLM` 和 `DeepSeekV4MTPModel` |
| 2 | `tests/models/registry.py` | 添加测试用的 `_HfExamplesInfo` |
| 3 | `vllm/models/deepseek_v4/__init__.py` | 平台分发 (NVIDIA/AMD) |
| 4 | `vllm/models/deepseek_v4/nvidia/model.py` | 模型主体：ForCausalLM, Model, DecoderLayer |
| 5 | `vllm/models/deepseek_v4/attention.py` | MLA 注意力层 + 包装器 |
| 6 | `vllm/models/deepseek_v4/compressor.py` | 压缩后端 + 元数据构建器 |
| 7 | `vllm/models/deepseek_v4/quant_config.py` | DeepSeek V4 FP8 量化配置 |

### 8.2 内核和算子

| # | 文件 | 用途 |
|---|------|------|
| 8 | `vllm/models/deepseek_v4/common/ops/` | 通用融合内核 (Triton, CuteDSL) |
| 9 | `vllm/models/deepseek_v4/nvidia/ops/` | NVIDIA 专用内核 |
| 10 | `vllm/models/deepseek_v4/nvidia/flashmla.py` | FlashMLA Sparse 实现 |
| 11 | `vllm/models/deepseek_v4/nvidia/mtp.py` | MTP 推测解码模型 |

### 8.3 注意力后端

| # | 文件 | 用途 |
|---|------|------|
| 12 | `vllm/v1/attention/backends/mla/flashmla.py` | FlashMLA 后端 |
| 13 | `vllm/v1/attention/backends/mla/flashmla_sparse.py` | FlashMLA Sparse 后端 |
| 14 | `vllm/v1/attention/backends/mla/indexer.py` | 稀疏注意力 Indexer |
| 15 | `vllm/v1/attention/backends/mla/sparse_swa.py` | 稀疏滑动窗口 |

### 8.4 KV Cache

| # | 文件 | 用途 |
|---|------|------|
| 16 | `vllm/v1/kv_cache_interface.py` | `MLAAttentionSpec` 定义 |

### 8.5 配置和工具

| # | 文件 | 用途 |
|---|------|------|
| 17 | `vllm/transformers_utils/configs/deepseek_v4.py` | HF 配置类 |
| 18 | `vllm/compilation/passes/fusion/mla_rope_kvcache_cat_fusion.py` | MLA 编译器融合 Pass |

---

## 9. 关键类和函数清单

### 9.1 必须实现的类/函数

| 类/函数 | 文件 | 用途 |
|---------|------|------|
| `ForCausalLM.__init__(vllm_config, prefix)` | model.py | 模型构造函数 |
| `ForCausalLM.forward(input_ids, positions, ...)` | model.py | 前向传播 |
| `ForCausalLM.compute_logits(hidden_states)` | model.py | 计算 logits |
| `ForCausalLM.load_weights(weights)` | model.py | 加载权重 |
| `Model.forward(input_ids, positions, ...)` | model.py | 内部模型前向 |
| `Model.embed_input_ids(input_ids)` | model.py | 嵌入 token IDs |
| `DecoderLayer.forward(hidden_states, ...)` | model.py | 解码层前向 |
| `Attention.get_kv_cache_spec(vllm_config)` | attention.py | 声明 KV Cache 规格 |
| `Attention.forward(hidden_states, ...)` | attention.py | 注意力前向 |

### 9.2 可选实现的类/函数

| 类/函数 | 文件 | 用途 |
|---------|------|------|
| `AttentionBackend.get_builder_cls()` | backend.py | 自定义注意力后端 |
| `AttentionImpl.forward()` | backend.py | 自定义注意力实现 |
| `AttentionImpl.do_kv_cache_update()` | backend.py | 自定义 KV Cache 写入 |
| `QuantizationConfig` | quant_config.py | 自定义量化配置 |
| `PretrainedConfig` | config.py | 自定义 HF 配置 |
| `TokenizerMode` | config.py | 自定义分词器模式 |

### 9.3 注册函数

| 函数 | 文件 | 用途 |
|------|------|------|
| `register_model()` | registry.py | 注册模型类 |
| `register_quantization_config()` | quantization/__init__.py | 注册量化方法 |
| `register_model_loader()` | model_loader/__init__.py | 注册模型加载器 |

---

## 附录：快速适配检查清单

```
□ 1. 在 registry.py 中添加模型架构映射
□ 2. 创建模型目录和 __init__.py
□ 3. 实现 ForCausalLM 类
□    □ __init__(vllm_config, prefix)
□    □ forward(input_ids, positions, ...)
□    □ compute_logits(hidden_states)
□    □ load_weights(weights)
□ 4. 实现 Model 类
□    □ embed_input_ids(input_ids)
□    □ forward(...)
□ 5. 实现 DecoderLayer 类
□    □ forward(hidden_states, ...)
□ 6. 实现 Attention 类
□    □ get_kv_cache_spec(vllm_config)
□    □ forward(hidden_states, ...)
□ 7. (可选) 实现自定义注意力后端
□ 8. (可选) 实现自定义 KVCacheSpec
□ 9. (可选) 实现自定义量化配置
□ 10. (可选) 实现自定义 HF 配置
□ 11. (可选) 实现自定义分词器/渲染器
□ 12. 添加测试
```
