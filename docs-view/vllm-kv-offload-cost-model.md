# vLLM KV Cache Offload 动态成本模型设计

## 一、背景与动机

### 1.1 当前问题

vLLM 的 KV cache offloading 在决定"从 offload 存储拉取 KV cache"还是"在 GPU 上重新计算"时，采用的是**纯二元决策**：offload 层有这个 block 就拉取，没有就重新计算。不存在任何基于成本的动态权衡。

这意味着在以下场景中会做出次优决策：

- **GPU 空闲时**：recompute 几乎免费（GPU 有大量空闲算力），但系统仍然会去拉取 KV cache，浪费 PCIe/网络带宽。
- **GPU 满载时**：recompute 需要排队等待 GPU 资源，延迟很高，但系统没有利用这个信号来优先选择 fetch。
- **小 token 数量**：传输的固定开销（CUDA event 创建、同步等）可能超过 recompute 的计算量。
- **网络/PCIe 带宽受限时**：远端存储（如 Mooncake 的分布式 KV store）的传输延迟可能远高于本地 recompute。

### 1.2 目标

设计一个统一的成本模型框架，让所有 KV connector（CPU offload、LMCache、Mooncake）都能基于实时信号做出更优的 fetch vs recompute 决策。

---

## 二、现有架构分析

### 2.1 两套并行的 KV offload 系统

vLLM 的 KV cache offloading 通过两套系统实现，它们在 Scheduler 的同一个决策点汇合：

**系统 A：`vllm/v1/kv_offload/` — OffloadingManager 框架**
- 用于 CPU offload 和多级存储
- 三层架构：OffloadingSpec（规范层）→ OffloadingManager（管理层）→ OffloadingHandler（执行层）

**系统 B：`vllm/distributed/kv_transfer/` — KVConnector 框架**
- 用于外部系统（LMCache、Mooncake）和 P/D 分离
- 通过 `KVConnectorBase_V1` 抽象接口统一

两套系统在 Scheduler 的调度循环中通过同一个接口 `get_num_new_matched_tokens()` 被调用。

### 2.2 当前决策流程

决策发生在 `vllm/v1/core/sched/scheduler.py` 的调度循环中（约第 811-843 行）：

```
请求到达
  │
  ├─ 第 1 步：查询 GPU 本地 prefix cache
  │   kv_cache_manager.get_computed_blocks(request)
  │   → 返回 num_new_local_computed_tokens
  │
  ├─ 第 2 步：查询外部/offload 缓存
  │   connector.get_num_new_matched_tokens(request, num_local_computed_tokens)
  │   → 返回 num_hit_tokens（有就返回数量，没有就返回 0）
  │
  └─ 总计算令牌 = 本地缓存 + 外部缓存
     需要计算的令牌 = 请求总令牌 - 总计算令牌
```

### 2.3 OffloadingConnectorScheduler 的 lookup 算法

在 `vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py` 中：

**Full-attention 模型**（`_maximal_prefix_lookup`，第 321-338 行）：
- 从第一个 block 开始顺序扫描
- 每个 block 调用 `manager.lookup(key)`
- 遇到第一个 miss 就停止，返回连续命中数
- 命中即拉取，miss 即 recompute，没有阈值

**Sliding-window 模型**（`_sliding_window_lookup`，第 340-364 行）：
- 从末尾向前扫描
- 寻找最后 `sliding_window_size` 个连续命中的 block
- 同样是纯 hit/miss 判断

**TieringOffloadingManager 的多级查询**（`vllm/v1/kv_offload/tiering/manager.py` 第 408-477 行）：
```
查询一级层 (CPU) → 命中? 返回 True (立即可用)
                 → 传输中? 返回 None (延迟重试)
                 → 未命中? 查询二级层...
                    → 二级层命中? 发起 promote，返回 None
                    → 全部未命中? 返回 False (recompute)
```

### 2.4 Mooncake 的决策

在 `vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/scheduler.py` 第 78-121 行：

```python
def get_num_new_matched_tokens(self, request, num_computed_tokens):
    token_len = request.num_tokens // self._block_size * self._block_size
    num_external_hit_tokens = self.client.lookup(token_len, request.block_hashes)
    # 纯 hit/miss，没有成本比较
    need_to_allocate = num_external_hit_tokens - num_computed_tokens
    return need_to_allocate, self.load_async
```

### 2.5 LMCache 的决策

在 `vllm/distributed/kv_transfer/kv_connector/v1/lmcache_connector.py` 中：
- 决策逻辑完全在 LMCache 自己的 adapter 中（`lmcache.integration.vllm.vllm_v1_adapter`）
- vLLM 侧只是调用 `get_num_new_matched_tokens()`，不参与决策

### 2.6 关键发现

**所有系统都缺少一个关键组件：基于当前 GPU 负载、传输带宽、token 数量的动态成本比较算法。**

---

## 三、现有可用信号分析

### 3.1 已有但未暴露的信号

| 信号 | 位置 | 说明 |
|------|------|------|
| `TransferResult.transfer_time` | `vllm/v1/kv_offload/worker/worker.py` 第 53-72 行 | 每次传输的 CUDA event 计时（秒） |
| `TransferResult.transfer_size` | 同上 | 每次传输的数据量（字节） |
| `TransferResult.transfer_type` | 同上 | 传输方向，如 `("GPU", "CPU")` |

这些数据目前只在 Worker 进程内部使用，**不暴露到 Scheduler 侧，也不出现在 Prometheus 指标中**。

### 3.2 Scheduler 侧已有信号

| 信号 | 来源 | 说明 |
|------|------|------|
| `len(self.running)` | `Scheduler` 的 running 队列 | 当前正在执行的请求数 |
| `self.max_num_running_reqs` | `SchedulerConfig.max_num_seqs` | 最大并发请求数 |
| `self.kv_cache_manager.usage` | `BlockPool.get_usage()` | KV cache 使用率 (0.0-1.0) |
| `PerfStats.num_flops_per_gpu` | `ModelMetrics.get_step_perf_stats_per_gpu()` | 预估 FLOPs（分析值，非实测） |
| `PerfStats.num_read_bytes_per_gpu` | 同上 | 预估内存读取字节数 |

### 3.3 不存在但需要新增的信号

| 信号 | 说明 |
|------|------|
| 每步迭代的 wall-clock 时间 | 需要在 `schedule()` 前后添加计时 |
| 传输带宽的滑动平均 | 需要从 TransferResult 中计算并传递到 Scheduler |
| GPU SM 利用率 | 当前代码中完全没有，可用队列深度作为代理 |

### 3.4 Prometheus 指标现状

已有的相关指标：
- `vllm:kv_cache_usage_perc` — KV cache 使用率
- `vllm:external_prefix_cache_queries` / `vllm:external_prefix_cache_hits` — 外部缓存查询/命中
- `vllm:prompt_tokens_by_source` — 按来源分类的 prompt token（含 `external_kv_transfer`）
- `vllm:time_to_first_token_seconds` — TTFT
- `vllm:inter_token_latency_seconds` — ITL

**不存在的指标**：
- KV offload 传输带宽
- KV offload 传输延迟
- fetch vs recompute 的决策统计

---

## 四、动态成本模型设计

### 4.1 核心公式

```
Cost_recompute = T_compute(N) × contention_factor
Cost_fetch     = N × block_bytes / bandwidth_ema + latency_overhead

决策：if Cost_fetch < Cost_recompute → fetch
      else → recompute（返回 0 命中 tokens）
```

其中：
- `N` = 需要处理的 token 数
- `T_compute(N)` = GPU 上 recompute N 个 token 的时间
- `contention_factor` = GPU 繁忙时 recompute 排队等待的放大系数
- `bandwidth_ema` = 实测传输带宽的指数移动平均
- `latency_overhead` = 传输的固定开销

### 4.2 信号采集

#### 4.2.1 传输带宽和延迟

从 `TransferResult` 中采集，通过 EMA（指数移动平均）平滑：

```python
bandwidth_ema.update(transfer_size / transfer_time)
latency_ema.update(transfer_time)
```

数据流：Worker 进程的 `TransferResult` → 通过回调传递到 Scheduler 侧的 `OffloadingManager` → 更新 `CostModel`。

#### 4.2.2 GPU 繁忙度

使用**组合信号**，不需要额外的 GPU 测量：

```python
# 1. 队列深度代理（已有）
queue_ratio = num_running_reqs / max_running_reqs

# 2. KV cache 使用率（已有）
kv_usage = kv_cache_manager.usage  # 0.0-1.0

# 3. 预估 FLOPs（已有）
est_flops = perf_stats.num_flops_per_gpu
```

#### 4.2.3 每步迭代时间

在 `Scheduler.schedule()` 的开头和结尾添加 `time.monotonic()` 计时：

```python
step_start = time.monotonic()
# ... 调度逻辑 ...
step_end = time.monotonic()
self._last_step_time = step_end - step_start
```

### 4.3 冷启动策略

**默认 recompute**（保守策略）：
- 在收集到至少 N=10 次传输样本之前，成本模型不启用
- 此阶段保持现有的纯 hit/miss 逻辑
- 收集到足够样本后，自动切换到成本模型决策

### 4.4 成本计算细节

#### Fetch 成本

```python
data_size = num_tokens * block_bytes_per_token  # 需要传输的数据量
bw = bandwidth_ema.value                        # 实测带宽 EMA
latency = latency_ema.value                     # 实测延迟 EMA
cost_fetch = data_size / bw + latency           # 传输时间
```

#### Recompute 成本

```python
# 用 step_time 作为 GPU 计算能力的代理
step_time = step_time_ema.value
# 估算每 token 的计算时间
tokens_per_step = est_flops // 1_000_000_000  # 粗略估算
cost_recompute_per_token = step_time / max(1, tokens_per_step)

# GPU 繁忙度放大系数
queue_ratio = num_running_reqs / max_running_reqs
contention = 1.0 + queue_ratio * 0.5  # 线性放大，最大 1.5x

cost_recompute = cost_recompute_per_token * num_tokens * contention
```

### 4.5 边界条件处理

| 条件 | 处理 |
|------|------|
| `bandwidth_ema.value <= 0` | 返回 False（recompute） |
| `step_time_ema.value <= 0` | 返回 False（recompute） |
| `num_tokens == 0` | 返回 False（recompute） |
| 冷启动（样本不足） | 返回 False（recompute） |
| `latency_ema.value` 异常大 | 设置上限裁剪 |

---

## 五、实现计划

### Step 1：新建 CostModel 核心组件

**新文件**：`vllm/v1/core/cost_model.py`

包含：
- `ExponentialMovingAverage`：指数移动平均工具类
- `GpuState`：GPU 状态快照数据类（从 Scheduler 注入）
- `CostModelConfig`：配置数据类
- `FetchVsRecomputeCostModel`：核心成本模型类

### Step 2：暴露 TransferResult 到 Scheduler 侧

**修改文件**：`vllm/v1/kv_offload/worker/worker.py`

在 `OffloadingWorker` 中新增回调机制，让 `TransferResult` 能传递到 Scheduler 侧的 `OffloadingManager`。

**修改文件**：`vllm/v1/kv_offload/cpu/manager.py`

在 `CPUOffloadingManager` 中：
1. 新增 `cost_model: FetchVsRecomputeCostModel` 成员
2. 在处理 `TransferResult` 时调用 `cost_model.update_transfer()`
3. 暴露 `cost_model` 供 connector 查询

### Step 3：修改 OffloadingConnectorScheduler 的决策逻辑

**修改文件**：`vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py`

在 `get_num_new_matched_tokens()` 中，`_lookup()` 返回 `num_hit_tokens` 后，新增成本比较：

```python
def get_num_new_matched_tokens(self, request, num_computed_tokens):
    # ... 现有 lookup 逻辑 ...
    num_hit_tokens = self._lookup(req_status)

    if num_hit_tokens and num_hit_tokens > 0:
        # 新增：成本模型决策
        cost_model = self.manager.get_cost_model()
        if cost_model is not None:
            gpu_state = self._get_gpu_state()  # 从外部注入
            block_bytes = self._get_block_bytes_per_token()
            if not cost_model.should_fetch(num_hit_tokens, block_bytes, gpu_state):
                logger.debug(
                    "Cost model: skip fetch %d tokens, recompute cheaper",
                    num_hit_tokens,
                )
                return 0, False

    req_status.update_num_hit_blocks(...)
    self._touch(req_status)
    return num_hit_tokens, bool(num_hit_tokens)
```

### Step 4：Scheduler 注入 GPU 状态到 Connector

**修改文件**：`vllm/v1/core/sched/scheduler.py`

新增 `GpuStateProvider` 类，让 connector 能获取当前 GPU 状态：

```python
class GpuStateProvider:
    def __init__(self, scheduler: 'Scheduler'):
        self._scheduler = scheduler

    def get_state(self) -> GpuState:
        return GpuState(
            num_running_reqs=len(self._scheduler.running),
            max_running_reqs=self._scheduler.max_num_running_reqs,
            kv_cache_usage=self._scheduler.kv_cache_manager.usage,
            estimated_flops_per_gpu=...,
        )
```

### Step 5：添加 Step 时间测量

**修改文件**：`vllm/v1/core/sched/scheduler.py`

在 `schedule()` 方法的开头和结尾添加计时，并更新成本模型。

### Step 6：统一 Connector 接口

**修改文件**：`vllm/distributed/kv_transfer/kv_connector/v1/base.py`

在 `KVConnectorBase_V1` 中新增可选方法：

```python
def set_gpu_state_provider(self, provider: 'GpuStateProvider') -> None:
    """可选：让 connector 获取 GPU 状态用于成本决策"""
    pass

def update_step_time(self, step_time_s: float, num_tokens: int) -> None:
    """可选：每步结束后更新 step 时间"""
    pass
```

### Step 7：添加 Prometheus 指标

**修改文件**：`vllm/v1/metrics/loggers.py`

新增指标：
- `vllm:kv_offload_bandwidth_bytes_per_sec` — Gauge，当前测量的传输带宽
- `vllm:kv_offload_fetch_skipped_total` — Counter，成本模型跳过 fetch 的次数
- `vllm:kv_offload_fetch_selected_total` — Counter，成本模型选择 fetch 的次数
- `vllm:kv_cost_model_enabled` — Gauge，成本模型是否已启用（0/1）

### Step 8：添加配置项

**新文件**：`vllm/config/cost_model.py`

```python
@config
class CostModelConfig:
    enabled: bool = True
    min_samples: int = 10
    ema_alpha: float = 0.3
    cold_start_policy: Literal["fetch", "recompute"] = "recompute"
```

---

## 六、涉及文件清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `vllm/v1/core/cost_model.py` | **新建** | 核心成本模型类 |
| `vllm/v1/kv_offload/worker/worker.py` | 修改 | 暴露 TransferResult 回调 |
| `vllm/v1/kv_offload/cpu/manager.py` | 修改 | 集成成本模型，接收 TransferResult |
| `vllm/v1/kv_offload/base.py` | 修改 | OffloadingManager 接口新增 get_cost_model() |
| `vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py` | 修改 | 在 get_num_new_matched_tokens 中加入成本比较 |
| `vllm/distributed/kv_transfer/kv_connector/v1/base.py` | 修改 | 统一接口新增可选方法 |
| `vllm/v1/core/sched/scheduler.py` | 修改 | 注入 GpuStateProvider，测量 step 时间 |
| `vllm/v1/metrics/loggers.py` | 修改 | 新增 Prometheus 指标 |
| `vllm/config/cost_model.py` | **新建** | 配置数据类 |
| `tests/v1/core/test_cost_model.py` | **新建** | 单元测试 |

---

## 七、验证方案

### 7.1 单元测试

`tests/v1/core/test_cost_model.py`：
- 测试 EMA 收敛
- 测试冷启动行为（默认 recompute）
- 测试边界条件（bw=0, latency=0, gpu_state 全忙）
- 测试 fetch vs recompute 决策正确性

### 7.2 集成测试

修改 `tests/v1/engine/` 中的现有测试，验证：
- 成本模型启用后不影响正确性
- Prometheus 指标正确输出

### 7.3 手动验证

- 启动 vLLM server，观察 Prometheus 指标
- 对比启用/禁用成本模型的 TTFT 和 ITL

---

## 八、附录：关键代码位置索引

| 组件 | 文件路径 | 关键行号 |
|------|----------|----------|
| Scheduler 调度循环 | `vllm/v1/core/sched/scheduler.py` | 811-843 |
| OffloadingConnectorScheduler.get_num_new_matched_tokens | `vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py` | 535-569 |
| _maximal_prefix_lookup | 同上 | 321-338 |
| _sliding_window_lookup | 同上 | 340-364 |
| _lookup | 同上 | 385-479 |
| TransferResult 定义 | `vllm/v1/kv_offload/worker/worker.py` | 53-72 |
| TransferResult 产生 | `vllm/v1/kv_offload/cpu/gpu_worker.py` | 476-491 |
| MooncakeStoreScheduler.get_num_new_matched_tokens | `vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/scheduler.py` | 78-121 |
| TieringOffloadingManager.lookup | `vllm/v1/kv_offload/tiering/manager.py` | 408-477 |
| SchedulerStats | `vllm/v1/metrics/stats.py` | 275+ |
| PerfStats | `vllm/v1/metrics/perf.py` | 149+ |
| BlockPool.get_usage | `vllm/v1/core/block_pool.py` | 581-592 |
| Prometheus 指标注册 | `vllm/v1/metrics/loggers.py` | 601+ |
| KVConnectorBase_V1 接口 | `vllm/distributed/kv_transfer/kv_connector/v1/base.py` | 全文 |
| OffloadingConnector | `vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py` | 全文 |
| LMCacheConnectorV1 | `vllm/distributed/kv_transfer/kv_connector/v1/lmcache_connector.py` | 全文 |
