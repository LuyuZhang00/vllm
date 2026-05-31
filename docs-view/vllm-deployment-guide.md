# vLLM 部署文档完整指南

> 本文档详细介绍 `/home/luyuzhang/code/vllm/docs/deployment` 目录下的所有内容，涵盖容器部署、Kubernetes 部署、负载均衡、12 个 Kubernetes 集成平台、8 个云服务平台、以及 10 个应用框架的集成方式。

---

## 目录

- [1. 概述](#1-概述)
- [2. 容器部署 (Docker)](#2-容器部署-docker)
- [3. Kubernetes 原生部署](#3-kubernetes-原生部署)
- [4. Nginx 负载均衡](#4-nginx-负载均衡)
- [5. Kubernetes 部署框架](#5-kubernetes-部署框架)
- [6. Kubernetes 集成平台](#6-kubernetes-集成平台)
- [7. 云服务平台](#7-云服务平台)
- [8. 应用框架集成](#8-应用框架集成)
- [9. 部署选型指南](#9-部署选型指南)

---

## 1. 概述

### 1.1 文档目录结构

```
docs/deployment/
├── docker.md                    # Docker 容器部署
├── k8s.md                       # Kubernetes 原生部署
├── nginx.md                     # Nginx 负载均衡
├── frameworks/                  # 部署框架和应用集成
│   ├── lws.md                   # LeaderWorkerSet (多节点)
│   ├── helm.md                  # Helm Chart
│   ├── triton.md                # NVIDIA Triton
│   ├── anyscale.md              # Anyscale (托管 Ray)
│   ├── skypilot.md              # SkyPilot (多云)
│   ├── modal.md                 # Modal (Serverless)
│   ├── runpod.md                # RunPod (GPU 云)
│   ├── cerebrium.md             # Cerebrium (Serverless)
│   ├── dstack.md                # dstack (多云)
│   ├── bentoml.md               # BentoML (模型服务)
│   ├── hf_inference_endpoints.md # HF Inference Endpoints
│   ├── litellm.md               # LiteLLM (统一代理)
│   ├── open-webui.md            # Open WebUI (聊天界面)
│   ├── dify.md                  # Dify (LLM 应用平台)
│   ├── lobe-chat.md             # Lobe Chat (聊天界面)
│   ├── streamlit.md             # Streamlit (Web 应用)
│   ├── haystack.md              # Haystack (RAG 框架)
│   ├── autogen.md               # AutoGen (多代理)
│   ├── chatbox.md               # Chatbox (桌面客户端)
│   ├── anything-llm.md          # AnythingLLM (RAG 应用)
│   └── retrieval_augmented_generation.md  # RAG 示例
└── integrations/                # Kubernetes 集成平台
    ├── aibrix.md                # AIBrix (vLLM 控制面)
    ├── dynamo.md                # NVIDIA Dynamo
    ├── kaito.md                 # KAITO (K8s AI 操作符)
    ├── kserve.md                # KServe (模型服务)
    ├── kthena.md                # Kthena (Volcano 调度)
    ├── kubeai.md                # KubeAI (AI 操作符)
    ├── kuberay.md               # KubeRay (Ray on K8s)
    ├── llamastack.md            # Llama Stack (Meta)
    ├── llm-d.md                 # llm-d (分布式推理)
    ├── llmaz.md                 # llmaz (推理平台)
    └── production-stack.md      # vLLM Production Stack
```

### 1.2 部署方式分类

| 类别 | 方式 | 适用场景 |
|------|------|----------|
| **容器部署** | Docker/Podman | 单机开发、测试 |
| **K8s 原生** | Deployment + Service | 简单 K8s 部署 |
| **K8s 框架** | Helm, LWS, KubeRay | 生产级 K8s 部署 |
| **K8s 集成** | KServe, AIBrix, Kthena 等 | 企业级 K8s 平台 |
| **云服务** | SkyPilot, Modal, RunPod 等 | 快速云端部署 |
| **应用集成** | Open WebUI, Dify, LiteLLM 等 | 应用层集成 |

---

## 2. 容器部署 (Docker)

### 2.1 概述

Docker 是最简单的 vLLM 部署方式，适合单机开发和测试。

### 2.2 使用预构建镜像

```bash
# 基本用法 (root 用户)
docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HF_TOKEN=$HF_TOKEN" \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model Qwen/Qwen3-0.6B

# 非 root 用户 (更安全)
docker run --rm --gpus all \
    --user 2000:0 \
    -v ~/.cache/huggingface:/home/vllm/.cache/huggingface \
    -p 8000:8000 \
    vllm/vllm-openai:latest \
    meta-llama/Llama-3.1-8B-Instruct
```

### 2.3 关键配置参数

| 参数 | 说明 |
|------|------|
| `--runtime nvidia` | 使用 NVIDIA 运行时 |
| `--gpus all` | 暴露所有 GPU |
| `--ipc=host` | 共享主机 IPC (PyTorch 需要) |
| `--user 2000:0` | 非 root 用户运行 |
| `-p 8000:8000` | 端口映射 |
| `-v ~/.cache/huggingface:...` | 挂载 HF 缓存 |
| `HF_TOKEN` | HuggingFace 访问令牌 |

### 2.4 从源码构建

```bash
# 构建非 root 镜像
docker build --target vllm-openai-nonroot \
    -t vllm-openai-nonroot:local \
    -f docker/Dockerfile .

# 构建 root 镜像
DOCKER_BUILDKIT=1 docker build . \
    --target vllm-openai \
    --tag vllm/vllm-openai \
    --file docker/Dockerfile
```

### 2.5 注意事项

| 事项 | 说明 |
|------|------|
| **共享内存** | 必须使用 `--ipc=host` 或 `--shm-size` |
| **非 root 卷** | 挂载到 `/home/vllm` 而非 `/root` |
| **OpenShift** | 支持任意 UID，设置 `runAsGroup: 0` |
| **ARM64** | 使用 `--platform "linux/arm64"` |
| **CUDA 兼容性** | 设置 `VLLM_ENABLE_CUDA_COMPATIBILITY=1` |
| **预编译轮子** | 使用 `VLLM_USE_PRECOMPILED=1` 加速构建 |

---

## 3. Kubernetes 原生部署

### 3.1 概述

使用标准 Kubernetes Deployment 和 Service 部署 vLLM，适合简单的 K8s 环境。

### 3.2 部署架构

```
┌─────────────────────────────────────────────────────────┐
│                    Kubernetes Cluster                    │
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   PVC        │  │   Secret     │  │   Service    │  │
│  │  (模型缓存)   │  │  (HF Token)  │  │  (ClusterIP) │  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  │
│         │                 │                 │           │
│         └────────┬────────┘                 │           │
│                  ▼                          │           │
│  ┌───────────────────────────────┐         │           │
│  │         Deployment            │         │           │
│  │  ┌─────────────────────────┐  │         │           │
│  │  │    vLLM Container       │  │◀────────┘           │
│  │  │  ┌───────────────────┐  │  │                     │
│  │  │  │ GPU (nvidia.com)  │  │  │                     │
│  │  │  │ /dev/shm (Memory) │  │  │                     │
│  │  │  │ Liveness Probe    │  │  │                     │
│  │  │  │ Readiness Probe   │  │  │                     │
│  │  │  └───────────────────┘  │  │                     │
│  │  └─────────────────────────┘  │                     │
│  └───────────────────────────────┘                     │
└─────────────────────────────────────────────────────────┘
```

### 3.3 关键配置

```yaml
# PVC - 模型缓存
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: mistral-7b-pvc
spec:
  accessModes: [ReadWriteOnce]
  resources:
    requests:
      storage: 50Gi

# Secret - HF Token
apiVersion: v1
kind: Secret
metadata:
  name: hf-token-secret
type: Opaque
data:
  HF_TOKEN: <base64-encoded-token>

# Deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mistral-7b
spec:
  replicas: 1
  selector:
    matchLabels:
      app: mistral-7b
  template:
    metadata:
      labels:
        app: mistral-7b
    spec:
      containers:
      - name: mistral-7b
        image: vllm/vllm-openai:latest
        command: ["vllm", "serve", "mistralai/Mistral-7B-Instruct-v0.1"]
        resources:
          limits:
            nvidia.com/gpu: 1
        ports:
        - containerPort: 8000
          protocol: TCP
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 5
        volumeMounts:
        - name: cache-volume
          mountPath: /root/.cache
        - name: shm
          mountPath: /dev/shm
        env:
        - name: HF_TOKEN
          valueFrom:
            secretKeyRef:
              name: hf-token-secret
              key: HF_TOKEN
      volumes:
      - name: cache-volume
        persistentVolumeClaim:
          claimName: mistral-7b-pvc
      - name: shm
        emptyDir:
          medium: Memory
          sizeLimit: 2Gi

# Service
apiVersion: v1
kind: Service
metadata:
  name: mistral-7b
spec:
  type: ClusterIP
  selector:
    app: mistral-7b
  ports:
  - port: 80
    targetPort: 8000
```

### 3.4 注意事项

| 事项 | 说明 |
|------|------|
| **AMD GPU** | 需要 `hostNetwork: true`, `hostIPC: true`, `SYS_PTRACE` |
| **gRPC** | 使用 `--grpc` 启用 gRPC 服务 (端口 50051) |
| **探针失败** | 增加 `failureThreshold`，避免启动超时被杀 |
| **CPU 部署** | 仅用于演示，性能不如 GPU |

---

## 4. Nginx 负载均衡

### 4.1 概述

使用 Nginx 作为反向代理，在多个 vLLM 实例之间负载均衡。

### 4.2 架构

```
Client → Nginx (port 8000) → upstream backend
                              ├── vllm0:8000 (GPU 0)
                              └── vllm1:8000 (GPU 1)
```

### 4.3 Nginx 配置

```nginx
upstream backend {
    least_conn;  # 最少连接负载均衡
    server vllm0:8000 max_fails=3 fail_timeout=10000s;
    server vllm1:8000 max_fails=3 fail_timeout=10000s;
}

server {
    listen 80;
    location / {
        proxy_pass http://backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### 4.4 部署步骤

```bash
# 1. 创建 Docker 网络
docker network create vllm_nginx

# 2. 启动 vLLM 容器
docker run -d --name vllm0 --gpus device=0 \
    --network vllm_nginx --ipc=host --shm-size=10.24gb \
    -p 8081:8000 -v $hf_cache:/root/.cache/huggingface \
    vllm --model Qwen/Qwen1.5-0.5B-Chat

docker run -d --name vllm1 --gpus device=1 \
    --network vllm_nginx --ipc=host --shm-size=10.24gb \
    -p 8082:8000 -v $hf_cache:/root/.cache/huggingface \
    vllm --model Qwen/Qwen1.5-0.5B-Chat

# 3. 启动 Nginx
docker run -d --name nginx-lb \
    --network vllm_nginx -p 8000:80 \
    -v $(pwd)/nginx_conf:/etc/nginx/nginx.conf:ro \
    nginx-lb
```

---

## 5. Kubernetes 部署框架

### 5.1 LeaderWorkerSet (LWS)

**用途：** 多节点分布式推理（如 405B 模型）

**架构：**
```
Kubernetes Cluster
├── LeaderWorkerSet
│   ├── Leader Pod (Ray Head + vLLM Server, GPU 0-7)
│   └── Worker Pod (Ray Worker, GPU 8-15)
└── Service (ClusterIP, 仅选择 Leader)
```

**关键配置：**
```yaml
apiVersion: leaderworkerset.x-k8s.io/v1
kind: LeaderWorkerSet
metadata:
  name: vllm
spec:
  leaderWorkerTemplate:
    size: 2  # 1 leader + 1 worker
    restartPolicy: RecreateGroupOnPodRestart
    leaderTemplate:
      containers:
      - name: vllm-leader
        command: ["multi-node-serving.sh", "leader"]
        resources:
          limits:
            nvidia.com/gpu: 8
            memory: 1124Gi
    workerTemplate:
      containers:
      - name: vllm-worker
        command: ["multi-node-serving.sh", "worker"]
```

**适用场景：** 超大模型（405B+）、多节点推理

### 5.2 Helm Chart

**用途：** 标准化 K8s 部署，支持自动模型下载

**关键配置：**
```yaml
# values.yaml
image:
  repository: vllm/vllm-openai
  tag: latest
  command: "vllm serve /data/model --served-model-name opt-125m"

replicaCount: 1

resources:
  limits:
    nvidia.com/gpu: 1
    cpu: 4
    memory: 16Gi

autoscaling:
  enabled: false
  minReplicas: 1
  maxReplicas: 100
  targetCPUUtilizationPercentage: 80

extraInit:
  modelDownload:
    enabled: true
    s3modelpath: "s3://bucket/model"
    pvcStorage: 1Gi
```

**部署命令：**
```bash
helm upgrade --install --create-namespace \
    --namespace=ns-vllm test-vllm . \
    -f values.yaml
```

### 5.3 NVIDIA Triton

**用途：** 在 Triton 推理服务器中使用 vLLM 后端

**说明：** 文档指向外部 Triton 教程，使用 `facebook/opt-125m` 模型进行快速测试。

---

## 6. Kubernetes 集成平台

### 6.1 平台对比

| 平台 | 特点 | 复杂度 | 适用场景 |
|------|------|--------|----------|
| **AIBrix** | vLLM 官方控制面 | 中 | vLLM 专属 K8s 管理 |
| **Dynamo** | NVIDIA 分布式推理 | 高 | 大规模分布式 |
| **KAITO** | K8s AI 操作符 | 低 | 快速部署 |
| **KServe** | 模型服务平台 | 中 | 企业级模型服务 |
| **Kthena** | Volcano 调度 | 中 | 多节点推理 |
| **KubeAI** | AI 操作符 | 低 | 简单 K8s 部署 |
| **KubeRay** | Ray on K8s | 中 | Ray 生态 |
| **Llama Stack** | Meta 统一 API | 低 | Llama 模型 |
| **llm-d** | 分布式推理栈 | 高 | 生产级分布式 |
| **llmaz** | 推理平台 | 低 | 快速部署 |
| **Production Stack** | 官方生产方案 | 中 | 生产部署 |

### 6.2 AIBrix

**简介：** vLLM 项目官方维护的云原生控制面

**功能：**
- vLLM 部署和管理
- 自动扩缩容
- 请求路由
- LoRA 适配器管理

**文档：** [aibrix.readthedocs.io](https://aibrix.readthedocs.io)

### 6.3 Kthena

**简介：** 基于 Volcano 调度器的 K8s LLM 推理平台

**特点：**
- 使用 `ModelServing` CRD 声明式部署
- Volcano gang 调度确保资源就绪
- 支持多节点 Ray 集群

**部署步骤：**
```bash
# 1. 安装 Volcano
helm repo add volcano https://volcano-charts.storage.googleapis.com
helm install volcano volcano/volcano

# 2. 安装 Kthena
helm install kthena oci://ghcr.io/volcano-sh/charts/kthena --version v0.1.0

# 3. 创建 HF Token Secret
kubectl create secret generic hf-token --from-literal=HF_TOKEN=$HF_TOKEN

# 4. 部署模型
kubectl apply -f modelserving.yaml

# 5. 验证
kubectl get modelserving
kubectl get pod
```

**ModelServing YAML 示例：**
```yaml
apiVersion: workload.serving.volcano.sh/v1alpha1
kind: ModelServing
metadata:
  name: llama-405b
spec:
  schedulerName: volcano
  gangPolicy:
    minRoleReplicas:
      - name: entry
        minReplicas: 1
      - name: worker
        minReplicas: 1
  template:
    roles:
    - name: entry
      replicas: 1
      entryTemplate:
        spec:
          containers:
          - name: vllm-leader
            image: vllm/vllm-openai
            command: ["multi-node-serving.sh", "leader"]
            args: ["--tensor-parallel-size", "8", "--pipeline-parallel-size", "2"]
            resources:
              limits:
                nvidia.com/gpu: 8
    - name: worker
      replicas: 1
      workerTemplate:
        spec:
          containers:
          - name: vllm-worker
            image: vllm/vllm-openai
            command: ["multi-node-serving.sh", "worker"]
```

### 6.4 KubeRay

**简介：** Ray 集群的 K8s 操作符

**特点：**
- 声明式 `RayCluster` 和 `RayService` CRD
- 自动扩缩容
- 蓝绿部署

**适用场景：** 已有 Ray 生态的团队

### 6.5 vLLM Production Stack

**简介：** vLLM 官方发布的生产部署方案

**特点：**
- Helm 一键部署
- Grafana 可观测性
- 多模型支持
- 模型感知和前缀感知路由
- LMCache KV 缓存卸载

**部署步骤：**
```bash
# 1. 添加 Helm 仓库
helm repo add vllm https://vllm-project.github.io/production-stack

# 2. 创建 values.yaml
cat <<EOF > values.yaml
servingEngineSpec:
  modelSpec:
  - name: "qwen"
    repository: "vllm/vllm-openai"
    tag: "latest"
    modelURL: "Qwen/Qwen3-0.6B"
    replicaCount: 1
    requestCPU: 4
    requestMemory: "16Gi"
    requestGPU: 1
    pvcStorage: "10Gi"
EOF

# 3. 部署
helm install vllm vllm/vllm-stack -f values.yaml

# 4. 验证
kubectl get pods

# 5. 端口转发
kubectl port-forward svc/vllm-router-service 30080:80

# 6. 测试
curl http://localhost:30080/v1/completions
```

### 6.6 其他平台

**KServe：**
- K8s 原生模型服务平台
- 支持 HuggingFace serving runtime 和 `LLMInferenceService`

**KAITO：**
- K8s AI Toolchain Operator
- 自动 GPU 节点配置
- 预置模型配置

**KubeAI：**
- Substratus AI 的 K8s AI 操作符
- 从零扩展、基于负载的自动扩缩容
- 零外部依赖

**llm-d：**
- K8s 原生分布式推理服务栈
- 通过 KServe 的 `LLMInferenceService` 使用

**llmaz：**
- 简单易用的 K8s 推理平台
- vLLM 是默认后端

**Llama Stack：**
- Meta 的统一 LLM API 框架
- 支持远程和嵌入式 vLLM 模式

---

## 7. 云服务平台

### 7.1 平台对比

| 平台 | 类型 | 特点 | 价格模型 |
|------|------|------|----------|
| **SkyPilot** | 开源框架 | 多云、自动扩缩容 | 按云计费 |
| **Modal** | Serverless | 快速自动扩缩容 | 按使用付费 |
| **RunPod** | GPU 云 | 简单易用 | 按时计费 |
| **Cerebrium** | Serverless | 代码优先 | 按使用付费 |
| **dstack** | 开源框架 | 多云、成本透明 | 按云计费 |
| **BentoML** | 开源框架 | 模型服务框架 | 免费 (自托管) |
| **HF Inference Endpoints** | 托管服务 | HF Hub 集成 | 按时计费 |
| **Anyscale** | 托管 Ray | Ray 生态 | 按云计费 |

### 7.2 SkyPilot

**简介：** 开源多云 LLM 部署框架

**特点：**
- YAML 驱动部署
- 支持 Spot 实例
- 内置自动扩缩容和负载均衡
- 支持多种 GPU 类型

**部署示例：**
```yaml
# serving.yaml
resources:
  accelerators: {L4, A10g, A10, L40, A40, A100, A100-80GB}
  use_spot: True
  disk_size: 512
  ports: 8081

env:
  MODEL_NAME: mistralai/Mistral-7B-Instruct-v0.1
  HF_TOKEN: null

setup: |
  conda create -n vllm python=3.10 -y
  conda activate vllm
  pip install vllm flash-attn

run: |
  conda activate vllm
  vllm serve $MODEL_NAME --tensor-parallel-size $SKYPILOT_NUM_GPUS_PER_NODE
```

**部署命令：**
```bash
# 单实例
sky launch serving.yaml --env HF_TOKEN

# 多副本服务
sky serve up -n vllm serving.yaml --env HF_TOKEN
```

### 7.3 RunPod

**简介：** GPU 云平台

**部署步骤：**
1. 创建 RunPod 账户
2. 启动 GPU Pod (CUDA 模板)
3. SSH 进入 Pod
4. 运行 `vllm serve <model> --host 0.0.0.0 --port 8000`
5. 在仪表板暴露端口 8000
6. 访问 `https://<pod-id>-8000.proxy.runpod.net`

**注意事项：**
- 必须使用 `--host 0.0.0.0` (不能用 127.0.0.1)
- 端口必须在仪表板中配置

### 7.4 HF Inference Endpoints

**简介：** HuggingFace 托管推理服务

**三种部署方式：**

1. **从目录部署：** 浏览 Endpoints Catalog，筛选 `vLLM`
2. **引导部署：** 从模型卡片点击 "Deploy" 按钮
3. **手动部署：** 自定义容器 URI 和参数

**特点：**
- 与 HF Hub 深度集成
- Day 0 模型支持
- 完全托管

### 7.5 其他平台

**Modal：**
- Serverless GPU 计算
- 按使用付费
- 快速自动扩缩容

**Cerebrium：**
- 代码优先的 Serverless AI 平台
- 自动 HTTP 端点和扩缩容
- 按使用付费

**dstack：**
- 开源多云框架
- 成本透明比较
- 支持 Spot 实例

**BentoML：**
- 开源模型服务框架
- OpenAI 兼容端点
- 支持容器化和 K8s

**Anyscale：**
- 托管 Ray 平台
- 自动集群管理
- 支持 AWS/GCP/Azure

---

## 8. 应用框架集成

### 8.1 集成模式

所有应用框架都通过 vLLM 的 **OpenAI 兼容 API** 进行集成：

```
应用框架 → OpenAI SDK → http://<vllm-host>:8000/v1/chat/completions
```

### 8.2 聊天界面

| 工具 | 特点 | 配置方式 |
|------|------|----------|
| **Open WebUI** | 功能丰富、支持 RAG | `OPENAI_API_BASE_URL` 环境变量 |
| **Lobe Chat** | 现代设计、插件系统 | 文档链接 |
| **Chatbox** | 桌面客户端 | UI 配置 API Host |
| **Streamlit** | Python Web 应用 | `VLLM_API_BASE` 环境变量 |

**Open WebUI 部署：**
```bash
# 启动 vLLM
vllm serve Qwen/Qwen3-0.6B-Chat --host 0.0.0.0 --port 8000

# 启动 Open WebUI
docker run -d -p 3000:8080 \
    -e OPENAI_API_BASE_URL=http://host.docker.internal:8000/v1 \
    -v open-webui:/app/backend/data \
    ghcr.io/open-webui/open-webui:main
```

### 8.3 LLM 应用平台

| 工具 | 特点 | 配置方式 |
|------|------|----------|
| **Dify** | LLM 应用开发平台 | UI 配置 Model Provider |
| **AnythingLLM** | RAG 应用 | UI 配置 Generic OpenAI |
| **Haystack** | RAG 框架 | Python SDK |

**Dify 部署：**
```bash
# 启动 vLLM
vllm serve Qwen/Qwen1.5-7B-Chat

# 启动 Dify
git clone https://github.com/langgenius/dify.git
cd dify/docker
docker compose up -d

# 配置: Settings → Model Provider → vLLM
# API Endpoint URL: http://<vllm-host>:8000/v1
```

### 8.4 代理和多代理

| 工具 | 特点 | 配置方式 |
|------|------|----------|
| **LiteLLM** | 统一 LLM 代理 | `hosted_vllm/` 前缀 |
| **AutoGen** | 多代理框架 | Python SDK |

**LiteLLM 集成：**
```python
import litellm

# 聊天补全
response = litellm.completion(
    model="hosted_vllm/Qwen/Qwen1.5-0.5B-Chat",
    api_base="http://localhost:8000/v1",
    messages=[{"role": "user", "content": "Hello!"}]
)

# 嵌入
import os
os.environ["HOSTED_VLLM_API_BASE"] = "http://localhost:8000/v1"
response = litellm.embedding(
    model="hosted_vllm/<embedding-model>",
    input=["Hello world"]
)
```

### 8.5 RAG 应用

**RAG 架构：**
```
文档 → 嵌入模型 (vLLM) → 向量数据库 (Milvus) → 检索
                                                        ↓
用户查询 → 聊天模型 (vLLM) ← 检索结果 ←──────────────┘
```

**部署步骤：**
```bash
# 1. 启动嵌入模型服务器
vllm serve ssmits/Qwen2-7B-Instruct-embed-base --port 8000

# 2. 启动聊天模型服务器
vllm serve qwen/Qwen1.5-0.5B-Chat --port 8001

# 3. 运行 RAG 示例
python retrieval_augmented_generation_with_langchain.py
```

---

## 9. 部署选型指南

### 9.1 按场景选择

| 场景 | 推荐方案 | 原因 |
|------|----------|------|
| **本地开发/测试** | Docker | 简单快速 |
| **单机生产** | Docker + Nginx | 负载均衡 |
| **K8s 简单部署** | Helm Chart | 标准化 |
| **K8s 多节点** | LWS 或 Kthena | 分布式推理 |
| **K8s 生产级** | Production Stack | 官方方案 |
| **企业级 K8s** | KServe 或 AIBrix | 功能全面 |
| **快速云端部署** | SkyPilot 或 RunPod | 简单易用 |
| **Serverless** | Modal 或 Cerebrium | 按需付费 |
| **HF 生态** | HF Inference Endpoints | 深度集成 |
| **聊天界面** | Open WebUI 或 Dify | 用户友好 |
| **RAG 应用** | Haystack + Milvus | 功能完善 |
| **多代理** | AutoGen | 多代理框架 |

### 9.2 按规模选择

| 规模 | 推荐方案 |
|------|----------|
| **1 GPU** | Docker |
| **2-8 GPU (单机)** | Docker + Nginx 或 Helm |
| **16+ GPU (多机)** | LWS, Kthena, KubeRay |
| **大规模集群** | Production Stack, KServe, AIBrix |

### 9.3 按团队选择

| 团队类型 | 推荐方案 |
|----------|----------|
| **个人开发者** | Docker, RunPod, Modal |
| **小团队** | Helm, SkyPilot |
| **中型团队** | KubeRay, Kthena |
| **大团队/企业** | KServe, AIBrix, Production Stack |

### 9.4 关键决策因素

| 因素 | 考虑点 |
|------|--------|
| **复杂度** | Docker 最简单，KServe 最复杂 |
| **成本** | Spot 实例 (SkyPilot) 可节省 60-70% |
| **扩展性** | K8s 方案支持水平扩展 |
| **运维** | 托管服务 (HF, RunPod) 无需运维 |
| **生态** | KubeRay 适合 Ray 生态 |
| **功能** | Production Stack 提供最全面的功能 |

---

## 附录

### A. 关键链接

| 资源 | 链接 |
|------|------|
| vLLM 官方文档 | https://docs.vllm.ai |
| Docker Hub | https://hub.docker.com/r/vllm/vllm-openai |
| Production Stack | https://github.com/vllm-project/production-stack |
| AIBrix | https://aibrix.readthedocs.io |
| SkyPilot | https://skypilot.readthedocs.io |
| RunPod | https://www.runpod.io |
| HF Inference Endpoints | https://huggingface.co/inference-endpoints |

### B. 通用配置模板

```yaml
# 通用 vLLM 服务配置
model: <model-name>
host: 0.0.0.0
port: 8000
tensor-parallel-size: <tp-size>
pipeline-parallel-size: <pp-size>
gpu-memory-utilization: 0.9
max-model-len: 8192
enable-prefix-caching: true
enable-chunked-prefill: true
kv-cache-dtype: auto
```

### C. 故障排查

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| OOM | GPU 内存不足 | 降低 `gpu-memory-utilization` 或使用量化 |
| 502 Bad Gateway | 模型还在加载 | 等待加载完成 |
| 连接拒绝 | 端口未暴露 | 检查端口映射和防火栏 |
| 探针失败 | 启动超时 | 增加 `failureThreshold` |
| 共享内存错误 | IPC 未配置 | 添加 `--ipc=host` 或 `--shm-size` |
