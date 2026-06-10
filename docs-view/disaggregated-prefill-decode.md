# vLLM PD 分离（Disaggregated Prefill/Decode）详解

## 概述

PD 分离是 vLLM 的一种架构优化，将 LLM 推理的两个阶段——**Prefill（预填充）** 和 **Decode（解码）**——拆分到不同的 GPU 实例上运行。

- **Prefill 阶段**: 计算密集型，一次性处理所有输入 token，生成 KV Cache
- **Decode 阶段**: 访存密集型，逐 token 自回归生成，依赖 KV Cache

PD 分离的核心思想：**让计算密集的 Prefill 和访存密集的 Decode 运行在不同的 GPU 上，各自独立扩缩容，通过高速网络传输 KV Cache**。

---

## 一、整体架构框架

### 1.1 核心组件

```
┌─────────────────────────────────────────────────────────────────────┐
│                         PD 分离整体架构                              │
│                                                                     │
│  ┌──────────┐                                                       │
│  │  Client   │                                                       │
│  └────┬─────┘                                                       │
│       │                                                             │
│       ▼                                                             │
│  ┌──────────┐     ┌──────────────┐     ┌──────────────┐             │
│  │  Proxy    │────▶│  Prefill P/T │────▶│  Decode P/T  │             │
│  │  Server   │     │  Instance(s) │     │  Instance(s) │             │
│  └──────────┘     └──────┬───────┘     └──────┬───────┘             │
│                          │                     │                     │
│                          │   KV Transfer       │                     │
│                          │   (NCCL/NIXL/       │                     │
│                          │    Mooncake/...)    │                     │
│                          └─────────────────────┘                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 三大核心模块

| 模块 | 职责 | 关键文件 |
|------|------|----------|
| **Proxy Server** | 接收客户端请求，路由到 Prefill/Decode 实例，协调两阶段执行 | `disagg_proxy_demo.py`, `disagg_proxy_p2p_nccl_xpyd.py` |
| **Prefill Instance** | 执行 Prefill 阶段，生成 KV Cache，通过 KV Transfer 发送给 Decode | `vllm serve --kv-transfer-config '{"kv_role":"kv_producer"}'` |
| **Decode Instance** | 接收 KV Cache，执行 Decode 阶段，逐 token 生成输出 | `vllm serve --kv-transfer-config '{"kv_role":"kv_consumer"}'` |

### 1.3 请求流转过程

```
步骤 1: Client 发送请求到 Proxy Server
步骤 2: Proxy 将请求转发给 Prefill Instance（max_tokens=1，只做 prefill）
步骤 3: Prefill Instance 执行前向传播，生成 KV Cache
步骤 4: KV Cache 通过 KV Transfer 机制传输到 Decode Instance
步骤 5: Proxy 将原始请求转发给 Decode Instance
步骤 6: Decode Instance 利用已有的 KV Cache 继续 decode，返回结果给 Client
```

---

## 二、KV Transfer 连接器类型

vLLM 支持多种 KV Cache 传输机制，通过 `--kv-transfer-config` 中的 `kv_connector` 字段指定。

### 2.1 P2pNcclConnector（点对点 NCCL）

**原理**: 使用 NVIDIA NCCL 库进行 GPU 间的点对点通信，通过 NVLink 或 PCIe 直接传输 KV Cache。

**配置示例**:
```json
{
    "kv_connector": "P2pNcclConnector",
    "kv_role": "kv_producer",
    "kv_rank": 0,
    "kv_parallel_size": 2,
    "kv_buffer_size": "1e9",
    "kv_port": "14579",
    "kv_connector_extra_config": {
        "proxy_ip": "127.0.0.1",
        "proxy_port": "30001",
        "http_ip": "127.0.0.1",
        "http_port": "8100",
        "send_type": "PUT_ASYNC"
    }
}
```

**关键参数**:
- `kv_role`: `kv_producer`（Prefill 端）或 `kv_consumer`（Decode 端）
- `kv_rank`: 当前实例在 KV 传输组中的排名
- `kv_parallel_size`: 参与 KV 传输的实例总数
- `kv_buffer_size`: KV Cache 传输缓冲区大小
- `kv_port`: NCCL 通信端口
- `send_type`: 传输模式（`PUT_ASYNC` 异步推送）

**优点**:
1. 低延迟: GPU 直接通信，无需 CPU 中转
2. 高带宽: 利用 NVLink/PCIe 高速互联
3. 简单部署: 无需额外的存储或消息中间件

**缺点**:
1. 距离限制: 仅支持同一节点内或通过 InfiniBand 连接的节点
2. GPU 资源占用: NCCL 通信会占用部分 GPU 计算资源
3. 弹性差: 实例数量在启动时固定（`kv_parallel_size`）

**适用场景**: 同节点多 GPU 部署，或有 InfiniBand 的集群

---

### 2.2 NixlConnector（NVIDIA NIXL）

**原理**: 使用 NVIDIA NIXL（NVIDIA Interconnect Library）进行高性能 KV Cache 传输，支持 RDMA 和 TCP。

**配置示例**（通过 LMCache 配置文件）:
```yaml
# prefiller-config.yaml
enable_nixl: True
nixl_role: "sender"
nixl_peer_host: "localhost"
nixl_peer_port: 55555
nixl_buffer_size: 1073741824  # 1GB
nixl_buffer_device: "cuda"
nixl_enable_gc: True

# decoder-config.yaml
enable_nixl: True
nixl_role: "receiver"
nixl_peer_host: "localhost"
nixl_peer_port: 55555
nixl_buffer_size: 1073741824  # 1GB
nixl_buffer_device: "cuda"
nixl_enable_gc: True
```

**关键参数**:
- `nixl_role`: `sender`（Prefill 端）或 `receiver`（Decode 端）
- `nixl_peer_host/port`: 对端地址
- `nixl_buffer_size`: 传输缓冲区大小
- `nixl_buffer_device`: 缓冲区所在设备（`cuda` 或 `cpu`）
- `nixl_enable_gc`: 启用垃圾回收

**优点**:
1. 高性能: 支持 RDMA 零拷贝传输
2. 跨节点: 支持远程节点间的 KV Cache 传输
3. 灵活: 可配置缓冲区位置（GPU/CPU）

**缺点**:
1. 依赖 NVIDIA NIXL 库
2. 配置相对复杂

**适用场景**: 跨节点 PD 分离，有 RDMA 网络的环境

---

### 2.3 MooncakeConnector

**原理**: 使用 Mooncake 分布式 KV Store 进行 KV Cache 传输，支持 RDMA 和 TCP，具备分布式调度能力。

**配置示例**:
```json
{
    "kv_connector": "MooncakeConnector",
    "kv_role": "kv_producer"
}
```

**启动参数**:
```bash
# Prefill 端需要设置 bootstrap port
VLLM_MOONCAKE_BOOTSTRAP_PORT=8998 vllm serve MODEL --port 8010 \
    --kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_producer"}'

# Decode 端
vllm serve MODEL --port 8020 \
    --kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_consumer"}'
```

**优点**:
1. 分布式调度: 内置 bootstrap 服务发现机制
2. 高性能: 支持 RDMA 传输
3. 弹性: 支持动态添加/移除实例
4. 生态: 与 Mooncake 生态系统集成

**缺点**:
1. 额外依赖: 需要安装 `mooncake.engine`
2. 配置复杂: 需要 bootstrap port 等额外配置

**适用场景**: 大规模生产环境，需要弹性扩缩容

---

### 2.4 ExampleConnector（基于本地存储）

**原理**: 将 KV Cache 保存到本地磁盘，通过文件系统实现离线的 Prefill-Decode 分离。

**配置示例**:
```json
{
    "kv_connector": "ExampleConnector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
        "shared_storage_path": "local_storage"
    }
}
```

**工作流程**:
1. Prefill 阶段: 将 KV Cache 序列化到 `local_storage/` 目录
2. Decode 阶段: 从 `local_storage/` 目录加载 KV Cache

**优点**:
1. 简单: 无需网络通信，基于文件系统
2. 调试友好: 可以直接查看 KV Cache 文件
3. 无额外依赖: 不需要 NCCL、NIXL 等库

**缺点**:
1. 性能差: 磁盘 I/O 成为瓶颈
2. 不适合在线服务: 仅适用于离线批处理
3. 存储开销: 需要额外的磁盘空间

**适用场景**: 开发调试、离线批处理、教学演示

---

### 2.5 FlexKVConnector

**原理**: 使用 FlexKV 分布式 KV Store 进行多级缓存管理，支持 CPU/磁盘/GPU 多级存储。

**配置示例**:
```json
{
    "kv_connector": "FlexKVConnectorV1",
    "kv_role": "kv_both"
}
```

**FlexKV 配置**:
```json
{
    "server_recv_port": "ipc:///tmp/flexkv_test",
    "cache_config": {
        "enable_cpu": true,
        "num_cpu_blocks": 10240
    }
}
```

**优点**:
1. 多级缓存: 支持 GPU → CPU → 磁盘的层次化存储
2. 前缀缓存: 自动识别共享前缀，避免重复计算
3. 大容量: 可以利用 CPU 内存和磁盘扩展缓存容量

**缺点**:
1. 额外依赖: 需要安装 FlexKV
2. 延迟: CPU/磁盘访问比 GPU 显存慢

**适用场景**: 长上下文、多请求共享前缀的场景

---

### 2.6 LMCacheConnector（通过 NIXL）

**原理**: LMCache 是一个 KV Cache 管理框架，通过 NIXL 实现高性能的 KV Cache 传输。

**架构**:
```
┌─────────────┐    NIXL     ┌─────────────┐
│  Prefiller   │◄──────────►│   Decoder    │
│  (LMCache)   │            │  (LMCache)   │
└─────────────┘            └─────────────┘
```

**优点**:
1. 高级抽象: LMCache 提供统一的 KV Cache 管理接口
2. 灵活配置: 支持多种后端（NIXL、CPU、磁盘）
3. 社区活跃: 持续更新和优化

**缺点**:
1. 额外依赖: 需要安装 LMCache 和 NIXL
2. 实验性: 功能仍在快速迭代

**适用场景**: 需要灵活 KV Cache 管理的场景

---

### 2.7 连接器对比总结

| 连接器 | 传输方式 | 跨节点 | 延迟 | 复杂度 | 适用场景 |
|--------|----------|--------|------|--------|----------|
| P2pNcclConnector | NCCL P2P | 否* | 低 | 低 | 同节点 GPU 直连 |
| NixlConnector | RDMA/TCP | 是 | 低 | 中 | 跨节点高性能 |
| MooncakeConnector | RDMA/TCP | 是 | 低 | 中 | 大规模生产环境 |
| ExampleConnector | 本地文件 | N/A | 高 | 低 | 开发调试 |
| FlexKVConnector | 多级存储 | 是 | 中 | 中 | 多级缓存 |
| LMCacheConnector | NIXL | 是 | 低 | 中 | 灵活管理 |

*InfiniBand 环境下可跨节点

---

## 三、部署模式详解

### 3.1 1P1D（单 Prefill + 单 Decode）

**架构**:
```
Client → Proxy → [Prefill GPU0] → KV Transfer → [Decode GPU1] → Response
```

**启动脚本** (`disaggregated_prefill.sh`):
```bash
# Prefill 实例 (GPU 0, port 8100)
CUDA_VISIBLE_DEVICES=0 vllm serve MODEL \
    --port 8100 \
    --kv-transfer-config '{"kv_connector":"P2pNcclConnector","kv_role":"kv_producer",...}'

# Decode 实例 (GPU 1, port 8200)
CUDA_VISIBLE_DEVICES=1 vllm serve MODEL \
    --port 8200 \
    --kv-transfer-config '{"kv_connector":"P2pNcclConnector","kv_role":"kv_consumer",...}'

# Proxy Server (port 8000)
python3 disagg_prefill_proxy_server.py
```

**优点**:
1. 最简单的部署模式
2. 适合小规模测试和验证

**缺点**:
1. 无法水平扩展
2. 单点故障

---

### 3.2 XpYd（多 Prefill + 多 Decode）

**架构**:
```
Client → Proxy ─┬─ Prefill 0 ─┬─ KV Transfer ─┬─ Decode 0
                ├─ Prefill 1 ─┤               ├─ Decode 1
                └─ ...        └               └─ ...
```

**启动脚本** (`disagg_example_p2p_nccl_xpyd.sh`):
```bash
# 配置: 1P3D (1 Prefill + 3 Decode)
PREFILL_GPUS=0
DECODE_GPUS=1,2,3
PREFILL_PORTS=20003
DECODE_PORTS=20005,20007,20009

# Proxy Server
python3 disagg_proxy_p2p_nccl_xpyd.py

# Prefill Server (GPU 0)
CUDA_VISIBLE_DEVICES=0 vllm serve MODEL --port 20003 \
    --kv-transfer-config '{"kv_connector":"P2pNcclConnector","kv_role":"kv_producer",...}'

# Decode Servers (GPU 1,2,3)
for i in 1 2 3; do
    CUDA_VISIBLE_DEVICES=$i vllm serve MODEL --port $((20005+i*2)) \
        --kv-transfer-config '{"kv_connector":"P2pNcclConnector","kv_role":"kv_consumer",...}'
done
```

**Proxy 调度策略**:
- **Round Robin**: 轮询分配请求到各实例
- **Least Load**: 分配到负载最低的实例
- **Random**: 随机选择实例

**优点**:
1. 独立扩展: Prefill 和 Decode 可以独立扩缩容
2. 负载均衡: 多实例分担负载
3. 高可用: 单实例故障不影响整体服务

**缺点**:
1. 配置复杂: 需要管理多个实例的 GPU/端口分配
2. 资源碎片化: 可能存在实例间负载不均

**适用场景**: 生产环境，需要高吞吐和弹性

---

### 3.3 EPD（Encoder + Prefill + Decode，多模态）

**架构**:
```
Client → Proxy ─┬─ Encoder ──┬─ KV Transfer ─┬─ Prefill ──┬─ KV Transfer ─┬─ Decode
                │  (图像编码) │               │  (KV生成)  │               │  (文本生成)
                └────────────┘               └───────────┘               └────────
```

**部署模式**:

#### 模式 A: 1E + 1PD（Encoder + Prefill/Decode 合一）
```bash
# Encoder 实例: 只做图像编码
CUDA_VISIBLE_DEVICES=0 vllm serve MODEL --port 8001 \
    --mm-encoder-only \
    --ec-transfer-config '{"ec_connector":"ECExampleConnector","ec_role":"ec_producer",...}'

# Prefill+Decode 实例: 接收编码结果，做 prefill 和 decode
CUDA_VISIBLE_DEVICES=1 vllm serve MODEL --port 8002 \
    --ec-transfer-config '{"ec_connector":"ECExampleConnector","ec_role":"ec_consumer",...}'
```

#### 模式 B: 1E + 1P + 1D（三层分离）
```bash
# Encoder 实例
CUDA_VISIBLE_DEVICES=0 vllm serve MODEL --port 8001 \
    --mm-encoder-only \
    --ec-transfer-config '{"ec_connector":"ECExampleConnector","ec_role":"ec_producer",...}'

# Prefill 实例
CUDA_VISIBLE_DEVICES=1 vllm serve MODEL --port 8002 \
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer"}' \
    --ec-transfer-config '{"ec_connector":"ECExampleConnector","ec_role":"ec_consumer",...}'

# Decode 实例
CUDA_VISIBLE_DEVICES=2 vllm serve MODEL --port 8003 \
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer"}'

# Proxy
python3 disagg_epd_proxy.py \
    --encode-servers-urls "http://localhost:8001" \
    --prefill-servers-urls "http://localhost:8002" \
    --decode-servers-urls "http://localhost:8003"
```

**Proxy 参数说明**:
| 参数 | 说明 |
|------|------|
| `--encode-servers-urls` | Encoder 实例地址列表，轮询分配多模态输入 |
| `--prefill-servers-urls` | Prefill 实例地址列表。设为 `disable` 跳过独立 Prefill |
| `--decode-servers-urls` | Decode 实例地址列表 |

**优点**:
1. 适合多模态: 图像编码和文本推理分离
2. 灵活: 可选择 2 层（E+PD）或 3 层（E+P+D）架构
3. 资源优化: Encoder 可以使用较小的 GPU

**缺点**:
1. 复杂度高: 需要管理 Encoder Cache (EC) 和 KV Cache 两种传输
2. 延迟增加: 多一跳传输

**适用场景**: 多模态模型（如 Qwen2-VL、Gemma3）

---

## 四、Proxy Server 详解

### 4.1 Proxy 的核心职责

1. **请求路由**: 将客户端请求路由到 Prefill/Decode 实例
2. **两阶段协调**: 先发送到 Prefill（max_tokens=1），再发送到 Decode
3. **负载均衡**: 在多个实例间分配请求
4. **故障处理**: 检测并移除故障实例

### 4.2 Proxy 工作流程

```python
async def create_completion(self, raw_request):
    # 1. 复制请求，设置 max_tokens=1（只做 prefill）
    kv_prepare_request = request.copy()
    kv_prepare_request["max_tokens"] = 1

    # 2. 选择 Prefill 实例并发送请求
    prefill_instance = self.schedule(self.prefill_cycler)
    await self.forward_request(
        f"http://{prefill_instance}/v1/completions",
        kv_prepare_request
    )

    # 3. 选择 Decode 实例，发送完整请求
    decode_instance = self.schedule(self.decode_cycler)
    return StreamingResponse(
        self.forward_request(
            f"http://{decode_instance}/v1/completions",
            request
        )
    )
```

### 4.3 服务发现机制（P2pNccl XpYd）

P2pNccl XpYd 模式使用 ZMQ 进行服务发现：

```python
# Prefill/Decode 实例启动后，向 Proxy 注册
# 消息格式: {"type": "P"/"D", "http_address": "ip:port", "zmq_address": "ip:port"}

# Proxy 维护实例列表，超时未心跳的实例自动移除
def _remove_oldest_instances(instances):
    for key, value in instances.items():
        if value[1] > time.time():  # 未过期
            break
        instances.pop(key)
```

### 4.4 Mooncake Proxy 的特殊处理

Mooncake Proxy 需要额外的 bootstrap 信息：

```python
# 从 Prefill 实例获取 bootstrap 信息
response = await prefill_client["client"].get(
    bootstrap_addr + "/query"
)
data = response.json()
# data 包含 dp_engine_id 等信息，用于 KV 传输路由
```

---

## 五、KV Cache 事件系统

### 5.1 概述

KV Cache 事件系统允许外部订阅 KV Cache 的变化事件，用于缓存管理、监控等。

**配置**:
```json
{
    "enable_kv_cache_events": true,
    "publisher": "zmq",
    "topic": "kv-events"
}
```

### 5.2 事件类型

- **KV Cache 创建**: 新的 KV Cache 块被分配
- **KV Cache 淘汰**: KV Cache 块被回收
- **KV Cache 共享**: 多个请求共享同一 KV Cache 块

### 5.3 使用场景

1. **分布式缓存管理**: 外部缓存系统监听事件，决定何时缓存/淘汰
2. **监控**: 跟踪 KV Cache 使用率
3. **调试**: 分析 KV Cache 行为

---

## 六、KV Cache 加载失败恢复

### 6.1 问题背景

在 PD 分离场景中，KV Cache 传输可能失败（网络中断、节点故障等）。vLLM 支持自动恢复：

### 6.2 恢复机制

1. **检测**: Decode 端在加载 KV Cache 时检测到无效块
2. **重调度**: 将受影响的请求重新调度到 Prefill 端
3. **重试**: 重新执行 Prefill 并传输 KV Cache

### 6.3 自定义 Connector 示例

```python
class LoadRecoveryExampleConnector(ExampleConnector):
    """模拟 KV 加载失败的 Connector"""
    def load_kv(self, ...):
        # 第一次请求模拟失败
        if self.is_first_request:
            self.is_first_request = False
            raise KVLoadFailure("Simulated failure")
        return super().load_kv(...)
```

---

## 七、部署最佳实践

### 7.1 GPU 分配建议

| 场景 | Prefill GPU | Decode GPU | 说明 |
|------|-------------|------------|------|
| 1P1D | GPU 0 | GPU 1 | 同节点两卡 |
| 1P3D | GPU 0 | GPU 1,2,3 | Prefill 计算密集，Decode 访存密集 |
| 3P1D | GPU 0,1,2 | GPU 3 | 长输入场景，多 Prefill 并行 |
| EPD | GPU 0 (E) | GPU 1 (PD) | 多模态模型 |

### 7.2 内存配置建议

```bash
# Prefill 实例: 较高的 KV Cache 缓冲区
--gpu-memory-utilization 0.9
--kv-buffer-size 1e9

# Decode 实例: 较大的 KV Cache 缓冲区（需要缓存更多请求）
--gpu-memory-utilization 0.7
--kv-buffer-size 8e9
```

### 7.3 网络配置建议

```bash
# 同节点: 使用 NCCL
--kv-connector P2pNcclConnector

# 跨节点 (InfiniBand): 使用 NIXL
--kv-connector NixlConnector

# 跨节点 (TCP): 使用 Mooncake
--kv-connector MooncakeConnector
```

### 7.4 监控指标

关注以下指标来评估 PD 分离效果：

1. **TTFT (Time To First Token)**: Prefill 延迟
2. **TPS (Tokens Per Second)**: Decode 吞吐
3. **KV Cache 命中率**: 共享前缀的复用率
4. **KV 传输延迟**: KV Cache 传输耗时
5. **GPU 利用率**: Prefill/Decode GPU 的计算利用率

---

## 八、总结

### 8.1 选择指南

| 需求 | 推荐方案 |
|------|----------|
| 开发调试 | ExampleConnector（本地文件） |
| 同节点高性能 | P2pNcclConnector |
| 跨节点部署 | NixlConnector 或 MooncakeConnector |
| 多模态模型 | EPD 架构（E+P+D） |
| 多级缓存 | FlexKVConnector |
| 灵活管理 | LMCacheConnector |

### 8.2 注意事项

1. **实验性**: PD 分离功能仍在快速迭代，API 可能变化
2. **兼容性**: 不同 Connector 的配置参数不兼容，切换需要修改配置
3. **调试**: 建议先用 ExampleConnector 验证流程，再切换到高性能 Connector
4. **监控**: 生产环境务必监控 KV 传输延迟和成功率

---

## 九、深度剖析：ExampleConnector 完整实现分析

本章以 `ExampleConnector` 为案例，深入剖析 KV Connector 的内部实现机制。
ExampleConnector 是最简单的 Connector 实现，基于本地文件系统传输 KV Cache，
非常适合理解 PD 分离的核心原理。

### 9.1 文件结构

```
examples/disaggregated/example_connector/
├── run.sh                 # 启动脚本（顺序执行 prefill → decode）
├── prefill_example.py     # Prefill 端：生成 KV Cache 并保存到磁盘
├── decode_example.py      # Decode 端：从磁盘加载 KV Cache 并继续生成
└── README.md

vllm/distributed/kv_transfer/kv_connector/v1/
├── base.py                # KVConnectorBase_V1 抽象基类（定义 Connector 接口）
└── example_connector.py   # ExampleConnector 具体实现
```

### 9.2 KVConnectorBase_V1 接口详解

所有 KV Connector 都必须继承 `KVConnectorBase_V1` 并实现其抽象方法。
该接口分为 **Scheduler 侧** 和 **Worker 侧** 两部分：

#### 9.2.1 Scheduler 侧方法（调度器进程中运行）

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Scheduler 侧 Connector 生命周期                    │
│                                                                     │
│  1. get_num_new_matched_tokens(request)                             │
│     └─ 查询外部 KV Cache 中有多少 token 可以复用                      │
│                                                                     │
│  2. update_state_after_alloc(request, blocks, num_external_tokens)  │
│     └─ Block 分配后更新状态，标记需要加载的请求                         │
│                                                                     │
│  3. build_connector_meta(scheduler_output)                          │
│     └─ 构建本轮 step 的元数据，传递给 Worker 侧                       │
│                                                                     │
│  4. request_finished(request, block_ids)                            │
│     └─ 请求完成时调用，决定是否异步释放 Block                           │
│                                                                     │
│  5. update_connector_output(connector_output)                       │
│     └─ 从 Worker 侧接收输出，更新 Scheduler 侧状态                    │
└─────────────────────────────────────────────────────────────────────┘
```

#### 9.2.2 Worker 侧方法（Worker 进程中运行，在模型 forward 期间调用）

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Worker 侧 Connector 生命周期                       │
│                                                                     │
│  模型 forward 开始前:                                                 │
│  1. bind_connector_metadata(meta)                                   │
│     └─ 绑定 Scheduler 传递的元数据                                    │
│                                                                     │
│  set_forward_context 进入时:                                          │
│  2. start_load_kv(forward_context)                                  │
│     └─ 开始从外部加载 KV Cache 到 vLLM 的 paged buffer                │
│                                                                     │
│  每个 Attention 层执行时:                                              │
│  3. save_kv_layer(layer_name, kv_layer, attn_metadata)              │
│     └─ 将当前层的 KV Cache 从 paged buffer 保存到外部                  │
│                                                                     │
│  4. wait_for_layer_load(layer_name)                                 │
│     └─ 等待当前层的 KV 加载完成（用于逐层流水线）                       │
│                                                                     │
│  set_forward_context 退出时:                                          │
│  5. wait_for_save()                                                 │
│     └─ 等待所有保存操作完成                                           │
│                                                                     │
│  模型 forward 结束后:                                                  │
│  6. clear_connector_metadata()                                      │
│     └─ 清除元数据                                                   │
└─────────────────────────────────────────────────────────────────────┘
```

### 9.3 ExampleConnector 核心数据结构

#### 9.3.1 ReqMeta（请求元数据）

```python
@dataclass
class ReqMeta:
    token_ids: torch.Tensor    # 请求的 token IDs
    slot_mapping: torch.Tensor # 每个 token 在 KV Cache 中的物理地址
    is_store: bool             # True=保存 KV，False=加载 KV
    mm_hashes: list[str]       # 多模态输入的哈希（用于缓存键）
```

**slot_mapping 计算公式**:
```python
# block_ids = [3, 7, 12]  (逻辑块 → 物理块映射)
# block_size = 16
# slot_mapping = [3*16+0, 3*16+1, ..., 3*16+15, 7*16+0, ..., 12*16+15]
#              = [48, 49, ..., 63, 112, ..., 207]
slot_mapping = block_offsets + block_ids * block_size
```

#### 9.3.2 ExampleConnectorMetadata（连接器元数据）

```python
@dataclass
class ExampleConnectorMetadata(KVConnectorMetadata):
    requests: list[ReqMeta]  # 本轮需要处理的请求列表
```

### 9.4 ExampleConnector 完整工作流程

#### 9.4.1 Prefill 端工作流程（kv_role="kv_both", is_store=True）

```
┌─────────────────────────────────────────────────────────────────────┐
│  Prefill 端: 保存 KV Cache 到磁盘                                    │
│                                                                     │
│  1. build_connector_meta()                                          │
│     ├─ 遍历 scheduler_output.scheduled_new_reqs                     │
│     ├─ 对每个新请求: 检查本地是否已有缓存 → 没有 → 标记 is_store=True    │
│     └─ 生成 ExampleConnectorMetadata                                │
│                                                                     │
│  2. 模型 forward 执行（正常计算）                                      │
│     ├─ 每个 Attention 层执行后:                                       │
│     │   └─ save_kv_layer(layer_name, kv_layer, attn_metadata)       │
│     │       ├─ 遍历 metadata.requests 中 is_store=True 的请求        │
│     │       ├─ 从 paged KV buffer 中提取该请求的 KV 数据              │
│     │       │   └─ kv_cache = layer[block_idxs, :, offsets]         │
│     │       └─ 保存到文件: local_storage/{hash}/{layer_name}.safetensors │
│     └─ wait_for_save() → 无操作（同步保存已完成）                      │
│                                                                     │
│  3. 输出: 生成的文本 + 磁盘上的 KV Cache 文件                          │
└─────────────────────────────────────────────────────────────────────┘
```

#### 9.4.2 Decode 端工作流程（kv_role="kv_both", is_store=False）

```
┌─────────────────────────────────────────────────────────────────────┐
│  Decode 端: 从磁盘加载 KV Cache                                      │
│                                                                     │
│  1. get_num_new_matched_tokens(request)                             │
│     ├─ 计算缓存键: hash(prompt_token_ids[:-1])                       │
│     ├─ 检查 local_storage/{hash}/ 目录是否存在                        │
│     ├─ 存在 → "External Cache Hit!" → 返回可加载的 token 数           │
│     └─ 不存在 → 返回 0（无外部缓存）                                   │
│                                                                     │
│  2. update_state_after_alloc(request, blocks, num_external_tokens)  │
│     └─ num_external_tokens > 0 → 将请求加入 _requests_need_load     │
│                                                                     │
│  3. build_connector_meta()                                          │
│     ├─ 遍历 _requests_need_load 中的请求                              │
│     ├─ 生成 is_store=False 的 ReqMeta                                │
│     └─ 清空 _requests_need_load                                     │
│                                                                     │
│  4. 模型 forward 执行前:                                               │
│     └─ start_load_kv(forward_context)                               │
│         ├─ 遍历 metadata.requests 中 is_store=False 的请求           │
│         ├─ 对每个请求的每个 Attention 层:                              │
│         │   ├─ 从文件加载: safetensors.torch.load_file(filename)     │
│         │   └─ 注入到 paged KV buffer:                               │
│         │       layer[block_idxs, :, offsets] = kv_cache             │
│         └─ 日志: "Inject KV cache of N tokens to the paged memory"  │
│                                                                     │
│  5. 模型 forward 执行:                                                │
│     └─ Attention 层使用已注入的 KV Cache，只需计算最后一个 token        │
│                                                                     │
│  6. 输出: 利用已有 KV Cache 生成的文本                                  │
└─────────────────────────────────────────────────────────────────────┘
```

### 9.5 缓存键生成机制

ExampleConnector 使用 **token IDs 的哈希** 作为缓存键：

```python
def _generate_foldername_debug(self, token_ids, mm_hashes, create_folder=False):
    # 1. 将 token IDs 转换为字节
    token_bytes = token_ids.numpy().tobytes()

    # 2. 如果有多模态输入，将 mm_hashes 也加入哈希
    if mm_hashes:
        mm_str = "-".join(mm_hashes)
        token_bytes += mm_str.encode("utf-8")

    # 3. 计算 SHA-256 哈希
    input_ids_hash = safe_hash(token_bytes, usedforsecurity=False).hexdigest()

    # 4. 生成文件夹路径
    foldername = os.path.join(self._storage_path, input_ids_hash)
    return foldername
```

**缓存键的组成**:
```
缓存键 = SHA256(token_ids_bytes + mm_hashes_bytes)

示例:
  token_ids = [1, 2, 3, 4, 5]
  mm_hashes = ["img_abc123"]
  缓存键 = SHA256(b'\x01\x00\x00\x00\x02\x00\x00\x00...' + b'img_abc123')
  文件夹 = local_storage/a1b2c3d4e5f6.../
```

**缓存命中判断**:
```python
def _found_match_for_prompt(self, prompt_token_ids, mm_hashes):
    # 对齐到 block_size 的整数倍（去掉最后一个 token，因为那是新生成的）
    num_tokens_to_check = align_to_block_size(len(prompt_token_ids) - 1, block_size)

    # 检查对应的文件夹是否存在
    foldername = self._generate_foldername_debug(
        torch.tensor(prompt_token_ids)[:num_tokens_to_check], mm_hashes
    )
    return os.path.exists(foldername)
```

### 9.6 KV Cache 文件格式

每个 Attention 层的 KV Cache 保存为一个独立的 safetensors 文件：

```
local_storage/
└── a1b2c3d4e5f6.../                    # 哈希文件夹
    ├── model.layers.0.self_attn.safetensors   # 第 0 层 KV Cache
    ├── model.layers.1.self_attn.safetensors   # 第 1 层 KV Cache
    ├── ...
    └── model.layers.31.self_attn.safetensors  # 第 31 层 KV Cache
```

**文件内容**:
```python
# 保存时
tensors = {"kv_cache": kv_cache.detach().cpu()}
safetensors.torch.save_file(tensors, filename)

# 加载时
kv_cache = safetensors.torch.load_file(filename, device=str(kv_cache_layer.device))["kv_cache"]
```

**KV Cache 形状**:
- 标准 Attention: `[num_pages, 2, page_size, num_kv_heads, head_dim]`
  - `2` 表示 K 和 V 两个张量
- MLA Attention: `[num_pages, page_size, hidden_dim]`
  - MLA 将 K、V 压缩为一个张量

### 9.7 调用时序图

```
┌──────────┐     ┌──────────┐     ┌──────────────────┐     ┌──────────────────┐
│  Client   │     │  Proxy   │     │ Prefill Instance  │     │ Decode Instance   │
└────┬─────┘     └────┬─────┘     └────────┬─────────┘     └────────┬─────────┘
     │                │                     │                        │
     │  1. POST /v1/completions             │                        │
     │───────────────▶│                     │                        │
     │                │                     │                        │
     │                │  2. POST (max_tokens=1)                      │
     │                │────────────────────▶│                        │
     │                │                     │                        │
     │                │                     │  3. build_connector_meta()
     │                │                     │     → is_store=True     │
     │                │                     │                        │
     │                │                     │  4. 模型 forward        │
     │                │                     │     ├─ Attention layer 0│
     │                │                     │     │  └─ save_kv_layer │
     │                │                     │     │     → 写入磁盘     │
     │                │                     │     ├─ Attention layer 1│
     │                │                     │     │  └─ save_kv_layer │
     │                │                     │     │     → 写入磁盘     │
     │                │                     │     └─ ...              │
     │                │                     │                        │
     │                │  5. Response (1 token)                       │
     │                │◀────────────────────│                        │
     │                │                     │                        │
     │                │  6. POST (full request)                      │
     │                │─────────────────────────────────────────────▶│
     │                │                     │                        │
     │                │                     │     7. get_num_new_matched_tokens()
     │                │                     │        → 检查磁盘缓存    │
     │                │                     │        → "Cache Hit!"   │
     │                │                     │                        │
     │                │                     │     8. build_connector_meta()
     │                │                     │        → is_store=False  │
     │                │                     │                        │
     │                │                     │     9. start_load_kv()  │
     │                │                     │        → 从磁盘读取      │
     │                │                     │        → 注入 paged buffer│
     │                │                     │                        │
     │                │                     │     10. 模型 forward     │
     │                │                     │        → 使用已有 KV     │
     │                │                     │        → 逐 token 生成   │
     │                │                     │                        │
     │  11. Response (streaming)            │                        │
     │◀──────────────────────────────────────────────────────────────│
     │                │                     │                        │
```

### 9.8 Prefill 端代码逐行分析

```python
# prefill_example.py

# 1. 准备带长前缀的 prompt（模拟共享前缀场景）
def read_prompts():
    context = "Hi " * 1000        # 1000 个 "Hi " 作为共享前缀
    context2 = "Hey " * 500       # 500 个 "Hey " 作为另一个共享前缀
    return [
        context + "Hello, my name is",    # prompt 1: 长前缀 + 短后缀
        context + "The capital of France is",  # prompt 2: 同一前缀 + 不同后缀
        context2 + "Your name is",        # prompt 3: 不同前缀
        context2 + "The capital of China is",  # prompt 4: 同一前缀 + 不同后缀
    ]

# 2. 配置 KV Connector
llm = LLM(
    model="meta-llama/Llama-3.2-1B-Instruct",
    enforce_eager=True,              # 禁用 CUDA Graph（简化调试）
    gpu_memory_utilization=0.8,
    kv_transfer_config=KVTransferConfig(
        kv_connector="ExampleConnector",
        kv_role="kv_both",           # 同时支持 produce 和 consume
        kv_connector_extra_config={
            "shared_storage_path": "local_storage"  # KV Cache 存储路径
        },
    ),
)

# 3. 执行 prefill（max_tokens=1，只生成 1 个 token）
outputs = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=1))

# 4. 保存 prompt + 生成的 token 到文件（供 decode 端使用）
for output in outputs:
    new_prompts.append(output.prompt + output.outputs[0].text)
with open("output.txt", "w") as f:
    for prompt in new_prompts:
        f.write(prompt + "\n")
```

### 9.9 Decode 端代码逐行分析

```python
# decode_example.py

# 1. 从文件加载 prompt（包含 prefill 阶段生成的 token）
def read_prompts():
    prompts = []
    with open("output.txt") as f:
        for line in f:
            prompts.append(line.strip())
    return prompts

# 2. 配置 KV Connector（与 prefill 端相同）
llm = LLM(
    model="meta-llama/Llama-3.2-1B-Instruct",
    enforce_eager=True,
    gpu_memory_utilization=0.8,
    max_num_batched_tokens=64,       # 限制 batch 大小
    max_num_seqs=16,                 # 限制并发请求数
    kv_transfer_config=KVTransferConfig(
        kv_connector="ExampleConnector",
        kv_role="kv_both",
        kv_connector_extra_config={
            "shared_storage_path": "local_storage"
        },
    ),
)

# 3. 执行 decode（max_tokens=10，生成 10 个 token）
# 内部流程:
#   a. get_num_new_matched_tokens() → 检查磁盘缓存 → 命中！
#   b. start_load_kv() → 从磁盘加载 KV Cache 到 GPU
#   c. 模型 forward → 使用已有 KV Cache，只需计算新 token
outputs = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=10))
```

### 9.10 关键设计点分析

#### 9.10.1 为什么 max_tokens=1？

Prefill 端设置 `max_tokens=1` 的原因：
1. Prefill 的目的是计算 KV Cache，只需要执行一次完整的前向传播
2. 生成 1 个 token 是因为 vLLM 的 generate() 需要至少生成 1 个 token 才会触发 forward
3. 这 1 个 token 的 KV Cache 也会被保存，供 Decode 端使用

#### 9.10.2 为什么 align_to_block_size(len - 1)？

```python
num_tokens_to_check = align_to_block_size(len(prompt_token_ids) - 1, self._block_size)
```

去掉最后一个 token 的原因：
1. Decode 端的 prompt = 原始 prompt + prefill 生成的 1 个 token
2. 缓存键应该基于原始 prompt（不含新生成的 token）
3. 所以用 `prompt_token_ids[:-1]` 来计算缓存键

#### 9.10.3 为什么 kv_role="kv_both"？

ExampleConnector 使用 `kv_both` 而非分离的 `kv_producer`/`kv_consumer`，因为：
1. 同一个 LLM 实例既做 prefill 又做 decode（离线模式）
2. Prefill 阶段：`is_store=True` → 保存 KV 到磁盘
3. Decode 阶段：`is_store=False` → 从磁盘加载 KV
4. 两个阶段在同一个进程中顺序执行

#### 9.10.4 Slot Mapping 的作用

Slot mapping 告诉 KV Cache 每个 token 存储在哪个物理位置：

```
token_idx:    0    1    2    3    4    5   ...   15   16   17  ...
              │    │    │    │    │    │         │    │    │
              ▼    ▼    ▼    ▼    ▼    ▼         ▼    ▼    ▼
slot_mapping: 48   49   50   51   52   53  ...   63   112  113 ...

含义:
  token 0 → 物理 block 3 的 offset 0 → 物理地址 3*16+0 = 48
  token 1 → 物理 block 3 的 offset 1 → 物理地址 3*16+1 = 49
  ...
  token 15 → 物理 block 3 的 offset 15 → 物理地址 3*16+15 = 63
  token 16 → 物理 block 7 的 offset 0 → 物理地址 7*16+0 = 112
```

保存 KV 时：`kv_cache = layer[block_idxs, :, offsets]`
加载 KV 时：`layer[block_idxs, :, offsets] = kv_cache`

### 9.11 扩展 ExampleConnector 实现自定义 Connector

基于 ExampleConnector，可以轻松扩展为自定义 Connector：

```python
class MyConnector(KVConnectorBase_V1):
    """自定义 Connector：使用 Redis 传输 KV Cache"""

    def __init__(self, vllm_config, role, kv_cache_config):
        super().__init__(vllm_config, role, kv_cache_config)
        self.redis = redis.Redis(host='localhost', port=6379)

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        # 检查 Redis 中是否有缓存
        key = self._make_key(request)
        if self.redis.exists(key):
            return len(request.prompt_token_ids) - 1 - num_computed_tokens, False
        return 0, False

    def start_load_kv(self, forward_context, **kwargs):
        # 从 Redis 加载 KV Cache
        for request in self._get_connector_metadata().requests:
            if not request.is_store:
                for layer_name in forward_context.no_compile_layers:
                    key = f"{request.key}:{layer_name}"
                    kv_data = self.redis.get(key)
                    kv_cache = torch.frombuffer(kv_data, dtype=torch.float16)
                    # 注入到 paged buffer...

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
        # 保存 KV Cache 到 Redis
        for request in self._get_connector_metadata().requests:
            if request.is_store:
                kv_cache = extract_kv(kv_layer, request.slot_mapping)
                key = f"{request.key}:{layer_name}"
                self.redis.set(key, kv_cache.numpy().tobytes())

    # ... 实现其他必需的方法
```

### 9.12 调试技巧

#### 9.12.1 查看 KV Cache 文件

```bash
# 查看缓存目录结构
ls -la local_storage/

# 查看某个缓存的文件
ls -la local_storage/a1b2c3d4.../

# 使用 Python 查看 safetensors 内容
python3 -c "
import safetensors.torch
data = safetensors.torch.load_file('local_storage/a1b2c3d4.../model.layers.0.self_attn.safetensors')
print(data['kv_cache'].shape)  # 例如: torch.Size([64, 2, 16, 8, 128])
print(data['kv_cache'].dtype)  # 例如: torch.float16
"
```

#### 9.12.2 启用详细日志

```bash
VLLM_LOGGING_LEVEL=DEBUG python3 prefill_example.py
```

#### 9.12.3 验证缓存命中

在 Decode 端的日志中查找：
- `"External Cache Hit!"` → 缓存命中
- `"Inject KV cache of N tokens to the paged memory"` → KV 加载成功

#### 9.12.4 手动测试缓存

```python
# 预先创建缓存目录，模拟缓存命中
import os
os.makedirs("local_storage/test_hash", exist_ok=True)
# 然后运行 decode_example.py，观察是否命中缓存
```
