
# 一、开场与个人定位类问题

## 1. 你说自己专注于大模型推理系统调度编排、推理框架开发和推理 Infra 建设，这三个方向边界是什么？

可以这样答：

> 我理解这三个方向是从上到下的一条完整链路。
> **调度编排**主要解决模型服务如何在 Kubernetes 集群里部署、扩缩容、成组调度、拓扑亲和和故障恢复，比如 KubeInfer、LWS、Volcano、RankTable。
> **推理框架开发**主要解决请求进入推理引擎之后如何调度、batch、管理 KVCache、执行 prefill/decode，比如 vLLM、PagedAttention、PD 分离、LMCache 接入。
> **推理 Infra 基础设施**则更偏系统底座，包括通信、缓存、存储、算子、硬件适配和性能观测，比如 Mooncake、KVCache 多级缓存、Ascend C 算子优化。
> 我自己的优势是能把这三层串起来：上层能做 K8s Operator 和分布式调度，中层理解 vLLM/LMCache 的 KVCache 链路，底层也做过 Ascend C 算子和 NPU 性能优化。

---

## 2. 你最强的是哪一块？

可以这样答：

> 我最强的是**大模型推理 Infra 的系统集成和性能优化**。我不是只做单点功能，而是能从 workload 编排、KVCache 链路到算子性能一起看瓶颈。比如 KubeInfer 项目里我关注多机推理如何稳定部署；KVCache 项目里我关注长上下文 TTFT 和吞吐；Ascend C 项目里我关注具体算子如何减少 GM/UB 搬运。我的优势是能在 Kubernetes、推理框架和异构硬件之间做端到端优化。

---

## 3. 请用 3 分钟讲一个在线推理请求从进入 vLLM 到输出 token 的完整链路。

可以这样答：

> 在线请求一般先进入 API Server，然后转成 vLLM 内部的 request，经过 tokenizer、sampling 参数解析后提交给 engine。engine 侧由 Scheduler 维护 waiting/running 队列，根据 max_num_batched_tokens、max_num_seqs、KVCache 可用 block 等约束做 continuous batching。
> 请求先进入 prefill 阶段，模型一次性处理 prompt tokens，并把每层 attention 的 K/V 写入 PagedAttention 管理的 KVCache block 中。prefill 完成后请求进入 decode 阶段，每一步生成一个 token，decode attention 会读取历史 KVCache，再经过 MLP/MoE、logits、sampling 得到下一个 token。
> vLLM 的核心优化点是 PagedAttention 和 continuous batching：PagedAttention 让 KVCache 以 block 方式管理，减少显存碎片；continuous batching 让新请求和未完成请求在 iteration 级别动态合批，提高 GPU/NPU 利用率。最终 token 会通过 output processor 按 request_id 返回给上层，在线服务场景下可以 streaming 返回。

---

## 4. 你如何证明自己不是每块都碰过一点？

可以这样答：

> 我会从三个层面证明。
> 第一，我在 KubeInfer 项目里做的是 Controller、CRD、Webhook、RankTable 生成和 LWS 工作负载抽象，不只是部署模型。
> 第二，我在 KVCache 项目里不仅接入 LMCache/Mooncake，还处理了 vLLM 分页 KVCache 与连续缓存之间的转换、NPU 上的 KV 搬运和跨实例复用。
> 第三，我做过 Ascend C 算子融合，理解 GM、UB、Tiling、CopyIn/Compute/CopyOut 这些底层性能点。
> 所以我的经验不是单点使用框架，而是围绕推理性能瓶颈做全链路优化。

---

## 5. 如果负责 70B/671B MoE 模型线上推理服务，你怎么判断瓶颈？

可以这样答：

> 我会先把推理拆成 prefill、decode、多机通信和 serving 调度四个维度。
> prefill 阶段主要看 compute-bound，关注 attention、MLP/MoE GEMM、prefill tokens/s。
> decode 阶段通常 memory-bound，关注 ITL、KVCache 读取带宽、HBM bandwidth、小 batch GEMM 效率。
> 多机 MoE 还要看 EP all-to-all、expert dispatch/combine、expert load balance。
> serving 层看 queue time、TTFT P90/P99、request throughput、KVCache block 使用率、preemption 次数。
> 工具上会先做 benchmark 分析 TTFT/ITL，再用 profiler 看 kernel、HCCL/NCCL、HBM 带宽和 Host gap。

---

# 二、项目 1：KubeInfer/LWS 调度编排

你的 PPT 里写到 KubeInfer 原来是 Deployment 模式，你升级为 Deployment/LWS 双模式，并围绕多机推理、PD 分离、Pod 成组启动、网络拓扑感知、RankTable、Webhook 准入校验做了优化。

---

## 1. 为什么传统 Deployment 不适合大模型多机推理？

可以这样答：

> Deployment 不是不能管理多个 Pod，它的问题是**只能做扁平副本管理**，缺乏分布式推理组语义。
> 大模型多机推理不是简单起多个 Pod，而是要求这些 Pod 之间有 Role、Group、Worker、Device、Rank 的拓扑关系。比如 TP/PP/EP/PD 分离场景下，Pod 必须成组启动、Rank 必须稳定、设备和通信拓扑必须一致。
> Deployment 只关心副本数量和滚动更新，不天然表达“这一组 Pod 是一个分布式推理实例”。所以在 RankTable 生成、Pod 成组调度、拓扑亲和、故障恢复上都不够。

---

## 2. LWS 相比 Deployment 多了什么语义？

可以这样答：

> LWS 的核心是 Leader-Worker 和 Group 语义。它不是把 Pod 当成完全独立副本，而是把一组 Pod 作为一个 replica group 进行管理。
> 对大模型推理来说，这很适合表达一个分布式推理实例：Leader 可以承担协调角色，Worker 承担计算角色，Group 可以对应一个推理副本或一个并行组。
> 这样 Controller 就能基于 LWS 的层级结构生成 RankTable，并且在生命周期管理上按组创建、更新和恢复，而不是把所有 Pod 当成扁平副本。

---

## 3. 单机多卡还需要 LWS 吗？

可以这样答：

> 单机单卡基本不需要 LWS。单机多卡如果只是一个 Pod 内申请 8 张卡，也不一定需要 LWS。
> LWS 的价值主要出现在**多 Pod 协同**场景，比如多机多卡、PD 分离、Prefill/Decode 多组件协同、或者一个推理实例需要多个 Pod 共同完成。
> 所以不是所有单机推理都需要 LWS，LWS 解决的是跨 Pod、跨节点、跨角色的组级编排问题。

---

## 4. PD 分离为什么会加剧编排复杂度？

可以这样答：

> PD 分离把 Prefill 和 Decode 拆成不同实例。Prefill 负责长 prompt 计算并生成 KVCache，Decode 负责自回归生成。
> 这会引入几个复杂点：第一，Prefill 和 Decode 的资源配比不同；第二，两者之间需要 KVCache 传输；第三，请求生命周期跨多个组件；第四，多机部署时 Prefill/Decode 之间还要考虑网络亲和和缓存 locality。
> 所以 PD 分离不是简单多起几个 Pod，而是要保证组件之间的拓扑、通信、缓存和生命周期协同。

---

## 5. 什么是网络拓扑感知调度？

可以这样答：

> 网络拓扑感知调度是指调度器不仅看 CPU、内存、NPU/GPU 资源是否足够，还要看 Pod 之间的通信距离。
> 对多机推理来说，TP/EP/PD 分离都可能有大量跨卡或跨节点通信。如果把同一个通信组调到不同交换机、不同机架甚至网络质量差的节点上，HCCL/RDMA 延迟会明显增加。
> 拓扑感知调度就是尽量把强通信相关的 Pod 放到同一个高性能通信域，比如同节点、同机架、同超节点或同 RoCE 网络域，减少跨节点通信抖动。

---

## 6. Reconcile 流程怎么设计？

可以这样答：

> 我的 Reconcile 会按声明式资源管理方式设计：
> 第一，读取 Instance CR，解析 workload kind，是 Deployment 还是 LWS。
> 第二，根据 spec 构建目标子资源，比如 LWS/Deployment、Service、ConfigMap、RankTable。
> 第三，检查实际状态和期望状态是否一致，不一致就 create/update。
> 第四，等待 Pod 调度完成、Pod IP 和 device 信息齐全后生成或刷新 RankTable。
> 第五，更新 Instance Status，比如 WorkloadReady、RankTableReady、ServiceReady。
> 整个流程必须幂等，因为 Kubernetes 事件可能重复触发，任何一步失败后下一轮 Reconcile 都要能继续推进。

---

## 7. 为什么 Reconcile 必须幂等？

可以这样答：

> Controller 不是一次性脚本，而是持续把实际状态拉回期望状态。Pod Ready、ConfigMap 更新、Spec 修改、子资源变化都会触发 Reconcile。
> 如果 Reconcile 不幂等，重复执行可能创建重复资源、覆盖错误状态或者造成 ConfigMap 脏写。
> 所以每一步都要先 get，再判断是否需要 create/update；RankTable 生成也要基于稳定排序和当前拓扑信息，保证重复执行结果一致。

---

## 8. 如何区分 Deployment 和 LWS 两种 WorkloadKind？

可以这样答：

> 我会在 CRD spec 里设计 workload kind 或 template 分支，比如 template 对应 Deployment，lwsTemplate 对应 LWS，并通过 Webhook 保证两者互斥。
> Controller 内部不建议到处写 if/else，而是抽象 WorkloadBuilder 或策略接口，不同 workload 实现自己的 build、status、rank discovery 逻辑。这样后续如果支持 StatefulSet 或其他工作负载，也可以扩展。

---

## 9. 为什么要设置 OwnerReference？

可以这样答：

> OwnerReference 用来建立 CR 和子资源的归属关系。
> 第一，它让 Kubernetes garbage collector 能在删除 Instance 时自动清理 LWS、Deployment、Service、ConfigMap 等子资源。
> 第二，Controller 可以通过 owner 关系过滤自己管理的资源。
> 第三，子资源事件也可以反向触发父资源 Reconcile。
> 如果不设置，删除 CR 后容易残留资源，也不利于状态追踪。

---

## 10. Pod IP 变化后 RankTable 怎么处理？

可以这样答：

> RankTable 依赖 Pod IP、device、rank 等信息。如果 Pod 重启或重调度导致 IP 变化，Controller 需要监听 Pod 状态变化，重新计算 RankTable。
> 关键是 Rank ID 分配规则要稳定，不能简单按当前列表随机顺序。一般要按 LWS group index、worker index、device id 这种确定性字段排序。
> 更新 RankTable 后，如果推理进程不能热加载，需要通过重启策略或生命周期控制让进程重新读取 RankTable，避免 HCCL 使用旧拓扑。

---

## 11. Rank ID 为什么必须稳定？

可以这样答：

> HCCL/NCCL 通信初始化要求每个 rank 对应的设备、IP 和进程是确定的。
> 如果 Pod 重启后 Rank ID 漂移，可能出现 rank0 认为自己要连接 rank1，但对端实际已经变成另一个设备的情况，导致通信建链失败或数据错乱。
> 所以 RankTable 自动生成不能只追求生成成功，还要保证排序稳定、拓扑一致和更新时机正确。

---

## 12. 单机多卡和多机多卡 RankTable 有什么区别？

可以这样答：

> 单机多卡主要关注 local rank、device id、进程到设备的映射，通信通常在节点内完成。
> 多机多卡还要额外描述 server id、Pod IP、节点 IP、global rank、node rank 等信息，因为跨节点通信要知道不同机器上的 rank 如何互联。
> 简单说，单机多卡是本地设备拓扑，多机多卡是全局通信拓扑。

---

## 13. RankTable 放 ConfigMap 还是 EmptyDir？

可以这样答：

> 全局 RankTable 适合放 ConfigMap，因为它由 Controller 生成，需要被多个 Pod 挂载读取，也便于统一更新和审计。
> 本地临时 rank 文件或运行时生成文件可以放 EmptyDir，因为它只属于当前 Pod 生命周期。
> 但 ConfigMap volume 更新有延迟，而且应用不一定自动重读，所以如果 RankTable 更新影响通信初始化，通常需要配合重启或显式 reload 机制。

---

## 14. Webhook 为什么重要？

可以这样答：

> Webhook 的价值是把错误配置前置拦截，而不是等到 Pod 启动或 HCCL 初始化时才失败。
> 比如 Template 和 LwsTemplate 同时配置、workload kind 非法、smooth 参数缺失、RankTable 依赖字段为空，这些都可以在 admission 阶段拒绝。
> 这样能减少运行时故障，也能提升用户体验和系统鲁棒性。

---

## 15. Mutating Webhook 和 Validating Webhook 区别？

可以这样答：

> Mutating Webhook 用来默认值填充或自动注入字段，比如补 label、annotation、默认策略。
> Validating Webhook 用来校验用户配置是否合法，比如 Template/LwsTemplate 互斥、字段不可变、资源组合是否合法。
> 这个项目里更核心的是 Validating Webhook，因为要提前拦截非法 workload 配置。

---

## 16. Webhook failurePolicy 选 Ignore 还是 Fail？

可以这样答：

> 如果是安全性、资源一致性强依赖的准入校验，我倾向于 Fail。因为如果 Webhook 挂了还允许非法 CR 进入，后面可能导致 RankTable 错配、HCCL 初始化失败。
> 但如果 Webhook 只做非关键默认值注入，可以考虑 Ignore 提升可用性。
> 具体取舍要看业务对可用性和正确性的要求。分布式推理里配置正确性很关键，所以核心校验应当 Fail。

---

## 17. Gang 调度是什么？为什么分布式推理需要？

可以这样答：

> Gang 调度要求一组 Pod 要么同时获得资源并启动，要么都不启动。
> 分布式推理里多个 Pod 往往共同组成一个模型实例。如果只有部分 Pod 启动，其他 Pod Pending，已启动的进程可能一直等待通信对端，最终 HCCL 超时或资源空占。
> Gang 调度可以避免部分启动造成的死等和资源浪费，尤其适合 TP/PP/PD 分离这种强协同任务。

---

## 18. Volcano PodGroup 怎么用？

可以这样答：

> Volcano 通过 PodGroup 表达一组 Pod 的最小可运行集合。minAvailable 表示至少多少 Pod 同时调度成功，这组任务才能启动。
> 对分布式推理来说，minAvailable 通常等于该推理组所需的全部 Pod 数，或者至少等于通信初始化需要的 Pod 数。
> KubeInfer 可以为 LWS 生成对应 PodGroup，让调度器知道这些 Pod 是一个整体，而不是独立副本。

---

## 19. LWS 和 Volcano 的关系是什么？

可以这样答：

> LWS 是 workload API，负责表达 Leader-Worker、Group、Replica 这类工作负载层级语义；Volcano 是调度器，负责 Gang、队列、优先级、资源协同和拓扑调度。
> LWS 解决“这组 Pod 是什么关系”，Volcano 解决“这组 Pod 怎么一起被调度”。
> 两者结合，才能既有分布式推理组语义，又有成组调度能力。

---

## 20. 部署耗时降低 40% 怎么定义？

可以这样答：

> 我会把部署耗时定义为从提交 Instance CR 到服务可用的时间，包括子资源创建、Pod Ready、RankTable 生成、配置挂载和推理服务 ready。
> 优化前多机推理需要手工配置 Deployment、RankTable 和组件协同；优化后通过 LWS template 和 Controller 自动生成，减少人工步骤和等待时间。
> 所以 40% 不是模型计算加速，而是部署链路端到端耗时降低。面试时我会明确 baseline、测试规模和统计方式。

---

## 21. Deployment、StatefulSet、DaemonSet、Job、LWS 区别？

可以这样答：

> Deployment 管理无状态副本，适合普通服务。
> StatefulSet 管理有序、有稳定身份的 Pod，适合有状态服务。
> DaemonSet 保证每个节点运行一个 Pod，适合日志、监控、device plugin。
> Job/CronJob 面向一次性或周期性任务。
> LWS 面向 Leader-Worker 结构的多 Pod 协同任务，适合大模型多机推理这种成组生命周期场景。

---

## 22. Requests/Limits 怎么设置？GPU/NPU 能不能 overcommit？

可以这样答：

> CPU/Memory 可以设置 requests 和 limits，K8s 基于 requests 调度，limits 做运行时限制。
> GPU/NPU 这类扩展资源通常按整数独占分配，不能像 CPU 那样 overcommit，除非底层 device plugin 支持切分或虚拟化。
> 大模型推理通常建议核心资源使用 Guaranteed QoS，避免被驱逐和干扰。

---

## 23. ConfigMap 挂载为环境变量和 volume 有什么区别？

可以这样答：

> 环境变量只在容器启动时注入，ConfigMap 后续更新不会反映到进程环境变量里。
> volume 挂载的 ConfigMap 理论上可以被 kubelet 周期性更新，但有延迟，而且应用程序必须重新读取文件。
> RankTable 这种启动时强依赖配置，如果更新后推理进程不 reload，可能需要重启进程保证一致性。

---

## 24. CRD status subresource 有什么作用？

可以这样答：

> spec 表示用户期望状态，status 表示系统实际状态。
> status subresource 可以让 Controller 单独更新状态而不影响 spec，也方便 RBAC 权限隔离。
> 在 KubeInfer 里，status 可以记录 WorkloadReady、RankTableReady、AvailableReplicas、Conditions 等信息，帮助用户判断实例当前阶段。

---

# 三、项目 2：KVCache 三级缓存优化

你的 PPT 里写到 KVCache 显存占用是长上下文推理瓶颈，GPU 生态的 LMCache/Mooncake 在昇腾 NPU 上存在通信库、内存管理、算子链路适配问题，并设计 HBM+DRAM+SSD 三级缓存，打通 Prefill 端缓存生成和 Decode 端复用。

---

## 1. 为什么长上下文推理里 KVCache 是核心瓶颈？

可以这样答：

> KVCache 大小随 batch、序列长度、层数线性增长。公式是：
> `KVCache ≈ batch × seq_len × layers × 2 × kv_heads × head_dim × dtype_size`。
> 其中 2 表示 K 和 V。
> 长上下文比如 64K/128K 时，decode 每生成一个 token 都要读取大量历史 KV，所以不仅显存占用高，而且 HBM 带宽压力也很大。
> 因此长上下文推理的瓶颈往往不是参数权重，而是 KVCache 容量和访问带宽。

---

## 2. Prefill 和 Decode 对 KVCache 的访问模式有什么区别？

可以这样答：

> Prefill 是一次性处理整个 prompt，计算密集，主要是把输入 tokens 的 K/V 写入 KVCache。
> Decode 是逐 token 生成，每一步只处理新 token，但要读取所有历史 token 的 KVCache。
> 所以 prefill 更偏 compute-bound，decode 更偏 memory-bound。长上下文下 decode 的 KV 读取会成为主要瓶颈。

---

## 3. TTFT 为什么会高？

可以这样答：

> TTFT 包括排队时间、prefill 计算时间、KVCache 分配/加载时间、调度等待和首 token decode 时间。
> 对长 prompt 来说，prefill 需要处理大量 token，所以 TTFT 很高。
> 如果能通过 prefix cache 或 LMCache 命中复用已有 KV，就可以把重复 prefill 计算变成 KV 加载，从而显著降低 TTFT。

---

## 4. TTFT P90 降低 70% 是什么口径？

可以这样答：

> 这个指标应当明确是 64K 长上下文缓存命中场景下的 P90 TTFT，而不是所有请求混合场景。
> baseline 是不使用 KVCache 复用或重新 prefill 的情况；优化后是 LMCache/Mooncake 命中并复用 KV 的情况。
> 面试里我会说明模型、硬件、并发、输入/输出长度、cache hit ratio，以及优化前后具体 P90 数值。这样 70% 才可信。

---

## 5. 吞吐提升 3.1 倍是什么吞吐？

可以这样答：

> 这里需要区分 request throughput、tokens/s 和 decode tokens/s。
> 我会把它定义为系统整体吞吐，比如单位时间完成的请求数或输出 tokens/s，并说明 benchmark workload。
> 如果是在缓存命中场景下，吞吐提升主要来自减少重复 prefill，把算力释放给更多请求或 decode。
> 如果 cache miss 很多，吞吐提升会下降，所以必须和命中率一起看。

---

## 6. 如果缓存命中率低，LMCache/Mooncake 会不会变慢？

可以这样答：

> 会有可能。缓存系统本身有 lookup、metadata、传输和调度开销。
> 如果命中率低，或者远端 KV 拉取比重新 prefill 更慢，缓存收益可能抵消甚至变成负收益。
> 所以需要 cost model：比较 `remote_load_time + restore_time` 和 `recompute_prefill_time`，只有前者更低时才复用远端 KV。

---

## 7. RAG、Agent、多轮对话、重复 Prefix 场景收益有什么区别？

可以这样答：

> 重复 Prefix 场景收益最大，因为相同前缀可以直接复用 KV。
> RAG 场景如果多个问题共享同一长文档上下文，也有明显收益。
> Agent 多轮对话中，如果历史上下文不断增长，部分 prefix 可以复用，收益取决于请求路由是否能命中缓存所在实例。
> 普通随机请求如果上下文不重复，缓存收益就有限。

---

## 8. KVCache 复用是否影响模型输出一致性？

可以这样答：

> 理论上在相同模型权重、相同输入、相同位置编码、相同 attention mask 和相同数值精度下，复用 KV 与重新 prefill 应该等价。
> 但工程上要注意 dtype、layout 转换、位置编码、block table、slotMapping、精度误差和采样随机性。
> 如果使用随机采样，还要固定 seed 或比较 logits/首 token，而不能只比较最终文本。

---

## 9. 什么是 PagedAttention？

可以这样答：

> PagedAttention 是 vLLM 的 KVCache 管理方式，它把连续 KVCache 切成固定大小的 block，用 block table 建立逻辑 token 到物理 block 的映射。
> 这样可以避免每个请求申请一整块连续显存，降低显存碎片，并支持动态调度、prefix cache 和 preemption。
> 它类似操作系统里的分页思想，把逻辑序列映射到非连续物理 KV blocks。

---

## 10. block、slot、block table、slotMapping 是什么？

可以这样答：

> block 是 KVCache 的固定大小物理页。
> slot 是 block 内的 token 位置。
> block table 是每个请求的逻辑 block 到物理 block 的映射表。
> slotMapping 是更细粒度的逻辑 token 到物理 slot 的映射。
> 在 KV 搬运时，我们需要根据 slotMapping 把 vLLM 分页布局转换成 LMCache 连续 buffer，或者反向恢复。

---

## 11. 为什么分页 KVCache 和连续 buffer 之间需要转换？

可以这样答：

> vLLM 为了高效管理显存，KVCache 是分页的，物理上可能不连续。
> LMCache/Mooncake 做缓存存储和传输时，更适合连续 buffer，因为连续内存便于批量拷贝、传输和序列化。
> 所以需要一个转换层，把逻辑 token 对应的分页 KV gather 到连续 buffer，restore 时再 scatter 回 vLLM 的物理 slot。

---

## 12. Prefix caching 和 LMCache 的关系是什么？

可以这样答：

> Prefix caching 通常是推理引擎内部对相同前缀的复用，主要在本实例或本进程内生效。
> LMCache 更像一个外部 KVCache 管理层，可以支持多级缓存、跨实例共享和远端存储。
> 二者目标一致，都是减少重复 prefill，但作用范围不同：prefix cache 更本地，LMCache 更系统化、跨实例。

---

## 13. vLLM 和 SGLang 在缓存管理上有什么区别？

可以这样答：

> vLLM 代表性机制是 PagedAttention 和 block-based KVCache 管理，强调高吞吐 continuous batching。
> SGLang 更强调 structured generation 和 RadixAttention，通过 radix tree 管理共享 prefix，对复杂 Agent、多轮和分支式请求比较友好。
> 如果请求有大量共享前缀和程序化执行，SGLang 的 radix cache 有优势；如果是通用 OpenAI serving，vLLM 生态成熟度和吞吐优化更强。

---

## 14. LMCache/Mooncake 原生面向 GPU，昇腾适配难点是什么？

可以这样答：

> 主要有三类：
> 第一，通信库不同。GPU 生态通常围绕 CUDA/NCCL/NVLink/RDMA，昇腾是 ACL/HCCL/HCCS/RoCE 等机制。
> 第二，内存和设备抽象不同。GPU 的 device pointer、stream、event、cudaMemcpy 不能直接复用到 NPU。
> 第三，KVCache layout 和搬运算子缺失。vLLM Ascend 上的分页 KVCache 要和 LMCache 连续 buffer 打通，需要 NPU 侧 gather/scatter 和同步机制。
> 所以不是简单改编译，而是要重做 connector、内存管理和 KV transfer。

---

## 15. monkey-patching 的优点和风险是什么？

可以这样答：

> 优点是低侵入，不需要大规模 fork 上游 LMCache，可以动态替换 GPUConnector、缓存管理和设备抽象，后续跟上游版本更容易。
> 风险是对上游内部接口依赖较强，如果上游类名、方法签名或初始化流程变化，patch 可能失效。
> 所以需要封装适配层、做版本检查、CI 测试和最小侵入，避免把 patch 逻辑散落在业务代码里。

---

## 16. LMCache 的 store、lookup、retrieve 时机？

可以这样答：

> store 通常在 prefill 完成后，把生成的 KVCache 写入缓存。
> lookup 在新请求进入时，根据 prefix hash 或 token 序列查找是否已有可复用 KV。
> retrieve 在命中后把 KV 从缓存后端加载回来，并恢复到推理引擎需要的 KV layout。
> 对 PD 分离来说，Prefill 端负责 store，Decode 端负责 retrieve/restore。

---

## 17. Mooncake Store 和 Transfer Engine 职责？

可以这样答：

> LMCache 负责缓存管理语义，比如 store、lookup、retrieve、缓存层级和元数据。
> Mooncake 更偏远端分布式 KV 存储和传输能力。Mooncake Store 提供分布式 KV 存储，Transfer Engine 负责 RDMA/TCP 等传输通道。
> 简单说，LMCache 决定“存什么、去哪找、什么时候取”，Mooncake 解决“远端怎么存、怎么高效传”。

---

## 18. HBM、DRAM、SSD 三级缓存怎么划分？

可以这样答：

> HBM 是 NPU/GPU 本地显存，带宽最高、延迟最低，但容量最小，适合当前活跃请求的 KVCache。
> DRAM 是 Host 内存，容量更大、延迟更高，适合 warm cache 或即将复用的 KV。
> SSD 容量最大但延迟最高，适合冷数据或大规模长上下文缓存，不适合每 token 关键路径。
> 三级缓存的核心是用容量换延迟，用策略决定 KV 放在哪一层。

---

## 19. 远端缓存一定比重新 prefill 快吗？

可以这样答：

> 不一定。
> 如果上下文非常长，prefill 计算成本高，远端 KV 传输可能更快。
> 但如果网络慢、KV 很大、命中数据分散，远端拉取可能比重新计算更慢。
> 所以要做 cost model，比较远端传输时间和 prefill 计算时间，同时考虑带宽、延迟、KV 大小和当前负载。

---

## 20. multi_layer_kv_transfer 的输入输出是什么？

可以这样答：

> 输入包括 vLLM 的分页 KVCache、slotMapping、layer 信息、token 范围、目标连续 buffer 信息。
> 输出是 LMCache 可管理的连续 KV buffer，或者反向将连续 KV restore 回 vLLM 的分页 KVCache。
> 它本质上做双向 gather/scatter：store 时从 paged KV gather 到 continuous buffer；retrieve 时从 continuous buffer scatter 回 paged KV slots。

---

## 21. 为什么连续区间合并能提升性能？

可以这样答：

> 如果按 token 粒度搬运，每个 token 都可能触发一次小 copy 和 Host 调度，调用次数非常多，带宽利用率低。
> 通过 slotMapping 识别连续 physical slot，可以把多个 token 合并成一个连续内存块异步拷贝，减少 copy 次数、Host 调度和同步开销。
> 本质是把小粒度随机搬运变成大块连续搬运，提高有效带宽。

---

## 22. Host 调度次数怎么量化？

可以这样答：

> 朴素逐 token copy 次数大约是 `layers × 2 × tokens`，2 表示 K/V。
> 合并后是 `layers × 2 × segments`，segments 是连续区间数量。
> 如果 64K token 合并成几百个连续段，copy 调用次数会从百万级下降到万级甚至更少。
> 这个指标比单纯说“效率提升”更能解释优化来源。

---

## 23. KV 搬运效率提升 2.8 倍怎么测？

可以这样答：

> 我会用 microbenchmark 测 KV 数据量和搬运耗时。
> 有效带宽 = KV 数据量 / 搬运耗时。
> baseline 是逐 token 或未合并搬运，优化后是连续区间合并 + 异步拷贝。
> 2.8 倍指的是 KV 搬运链路的有效带宽或耗时加速，不直接等同于端到端吞吐提升。端到端还要看 cache hit ratio、prefill 计算和 decode 瓶颈。

---

## 24. Ascend 上 GM、HBM、UB、Local Memory 怎么理解？

可以这样答：

> 在 Ascend C 语境下，GM 通常指全局内存，物理上对应设备 HBM。UB 是 AI Core 上的片上 buffer，用于算子内部临时数据和向量计算。
> 数据通常从 GM 搬到 UB，在 UB 中计算，再写回 GM。
> 算子优化的关键是减少 GM↔UB 的搬运次数，提高 UB 复用和流水效率。

---

## 25. cache miss、local hit、remote hit 三种场景怎么比较？

可以这样答：

> cache miss 要重新 prefill，TTFT 最高。
> local hit 从本地 HBM/DRAM 恢复 KV，通常最快。
> remote hit 需要通过 Mooncake/RDMA/TCP 拉取远端 KV，性能取决于网络和 KV 大小。
> 因此 benchmark 应该分开统计三种场景，否则平均值会掩盖真实瓶颈。

---

## 26. 如果并发升高，缓存系统会不会成为瓶颈？

可以这样答：

> 会。高并发下 lookup、metadata 锁、远端传输带宽、Host 内存带宽、SSD IO 都可能成为瓶颈。
> 解决方式包括缓存分片、异步 prefetch、请求路由到 cache locality 更高的实例、限制远端拉取并发、分层淘汰策略和传输带宽隔离。
> 所以缓存系统要和调度器协同，而不是单独优化。

---

## 27. 长上下文从 64K 到 128K，收益会线性增长吗？

可以这样答：

> 不一定。理论上重复 prefill 计算越长，缓存复用收益越大。
> 但 KV 数据量也线性增长，远端传输和本地 restore 的成本也会上升。
> 如果传输带宽或 DRAM/HBM 带宽成为瓶颈，收益可能低于线性增长。
> 所以要看 prefill compute cost 和 KV restore cost 的相对关系。

---

# 四、项目 3：Ascend C 算子融合优化

你上传的 AddRmsNormDynamicQuantV2 README 明确说明：该算子将 RmsNorm 前的 Add 和 RmsNorm 输出后的 1 个或 2 个 DynamicQuant 融合，以减少搬入搬出操作；计算包含 `x=x1+x2`、RMSNorm、可选 smooth scale 和动态量化。 代码入口根据 tiling key 选择 Normal、SingleRow、SliceD 三种 kernel 路径。

---

## 1. AddRmsNormDynamicQuantV2 融合了哪些算子？

可以这样答：

> 它融合了 Add、RmsNorm 和 1 到 2 路 DynamicQuant。
> 先计算 `x = x1 + x2`，再做 RMSNorm 得到 `y = x / rms(x) * gamma`，然后根据 smooth_scale1、smooth_scale2 是否存在，对 y 做一路或两路动态量化，输出 y1/y2 和 scale1/scale2。
> 这个融合减少了 Add 输出和 RmsNorm 输出反复写回 GM 再读回的开销。

---

## 2. RMSNorm 和 LayerNorm 区别？

可以这样答：

> LayerNorm 会对输入做均值和方差归一化，形式是 `(x - mean) / sqrt(var + eps)`。
> RMSNorm 去掉了减均值，只用 root mean square，即 `x / sqrt(mean(x^2)+eps)`，再乘 gamma。
> RMSNorm 计算更简单、访存和 reduce 更少，所以很多大模型采用 RMSNorm 来提升训练和推理效率。

---

## 3. DynamicQuant 是 per-token 还是 per-channel？

可以这样答：

> 这里更接近按 row 的动态量化。README 里 scale 是 `row_max(abs(input))/127`，row_max 表示每行求最大值。
> 如果输入是 `[N, D]`，那么每一行会得到一个 scale，所以 scale shape 通常是 reduce 后的 `[N]` 或前面维度。
> 这样能根据每个 token/row 的动态范围自适应量化到 INT8。

---

## 4. 为什么除以 127？

可以这样答：

> INT8 对称量化通常使用 [-127, 127] 的有效范围，避免 -128 的非对称问题。
> scale = max_abs / 127，表示最大绝对值映射到 127。
> 量化时 `q = round(input / scale)`，这样 q 落在 INT8 可表示范围内。
> 反量化时可用 `q * scale` 近似原值。

---

## 5. y1、y2、y3、y4、x、scale1、scale2 分别是什么？

可以这样答：

> x 是 `x1 + x2` 的结果。
> y 是 RMSNorm 之后的结果。
> y1 是第一路动态量化后的 INT8 输出，scale1 是第一路量化 scale。
> y2 是第二路动态量化后的 INT8 输出，scale2 是第二路量化 scale。
> y3 是 y cast 到 FP32 的输出。
> y4 是 y 保持 FP16/BF16 的输出。
> 其中 smooth_scale2 不存在时，y2 和 scale2 没有实际意义。README 里也说明了不同 smooth 输入组合下输出有效性的差异。

---

## 6. smooth_scale1 和 smooth_scale2 什么作用？

可以这样答：

> smooth_scale 用于在量化前对 RMSNorm 输出 y 做缩放，改善量化分布。
> 如果 smooth_scale1 存在，则第一路输入是 `y * smooth_scale1`；否则第一路直接使用 y。
> 如果 smooth_scale2 存在，则第二路输入是 `y * smooth_scale2`。
> smooth_scale2 不能单独存在，因为语义上第二路依赖第一路量化配置，代码和 shape 校验里也会拒绝只有 smooth2 的情况。

---

## 7. epsilon 作用是什么？

可以这样答：

> epsilon 是为了避免 RMS 分母为 0，同时提升数值稳定性。
> 如果 epsilon 太小，极小输入下可能有数值不稳定；如果太大，会影响归一化结果，使输出幅值偏小。
> 一般大模型 RMSNorm 里 epsilon 是固定超参，比如 1e-6。

---

## 8. 输入 FP16/BF16，为什么内部可能转 FP32？

可以这样答：

> FP16/BF16 存储带宽友好，但 RMSNorm 涉及平方、求和、rsqrt 和归一化，如果完全用低精度可能误差较大。
> 所以常见做法是输入/输出保持 FP16/BF16，内部统计和部分计算用 FP32，提高数值稳定性。
> 代码里 Normal kernel 也会把 x1/x2 cast 到 FP32 再做 Add 和后续计算。

---

## 9. 图模式融合是在图层做 pattern fusion 还是新增融合 op？

可以这样答：

> 这里可以分两层理解。
> 从图语义上看，是把 Add、RmsNorm、DynamicQuant 这个 pattern 融合为一个 AddRmsNormDynamicQuantV2 op。
> 从实现上看，是新增一个融合算子，并提供 op proto、infer shape、tiling 和 Ascend C kernel，让图构建或图优化阶段直接调用这个融合 op。
> 所以我会说这是图模式融合 + 自定义融合算子实现。

---

## 10. 融合前哪些中间结果会写 GM？

可以这样答：

> x1 和 x2 是输入，不应该标成中间结果。
> 融合前的中间结果主要是 Add 的输出 x，以及 RmsNorm 的输出 y。如果 Add、RmsNorm、DynamicQuant 分别是独立算子，x 会写回 GM 再被 RmsNorm 读入，y 也会写回 GM 再被 DynamicQuant 读入。
> 如果有两路 DynamicQuant，y 还可能被重复读取。融合后可以在一个 kernel 内复用中间结果，减少这些 GM 往返。

---

## 11. 融合后仍输出 x/y3/y4，为什么还能快？

可以这样答：

> 融合后并不是完全不写 GM，而是减少了**作为下游输入的中间张量反复落地和重新搬入**。
> x、y3、y4 如果是图的真实输出，仍然要写出；但在融合前，x 和 y 既要写出又要被下游算子重新读入。
> 融合后，内部计算可以在 UB/LocalTensor 中复用 x/y，最后只把必要输出写回 GM，所以仍然减少了读写次数和 kernel 调度开销。

---

## 12. 融合是否改变精度？

可以这样答：

> 理论上不应该改变语义，但可能因为计算顺序、内部精度、round 策略和 reduce 实现不同产生微小误差。
> 验证时要分别对 y1/y2、scale1/scale2、y3/y4、x 做误差对比。
> 对 FP 输出看 atol/rtol，对 INT8 输出看量化结果一致率或允许 1 个量化单位误差。
> 还要覆盖 smooth 不存在、只有 smooth1、smooth1+smooth2 三种路径。

---

## 13. Ascend C 编程模型是什么？

可以这样答：

> Ascend C 中，数据通常从 GM 搬到 UB，在 UB 里用 Vector/Scalar 计算，再写回 GM。
> TPipe 用来管理流水，TQue/LocalTensor 用来管理输入输出队列和片上 buffer。
> 一般 kernel 会拆成 CopyIn、Compute、CopyOut 三个阶段，通过 buffer 和 pipeline 尽量重叠搬运与计算。
> 优化重点是减少 GM 访问、提升 UB 复用、提高向量计算效率和减少 pipeline stall。

---

## 14. MTE2、MTE3、Vector、Scalar 分别做什么？

可以这样答：

> MTE2 通常负责 GM 到 UB 的搬入，MTE3 负责 UB 到 GM 的搬出。
> Vector pipeline 负责向量计算，比如 Add、Mul、Sqrt、Reduce、Cast、Round。
> Scalar pipeline 负责标量控制和一些标量计算。
> 高性能 Ascend C kernel 要尽量让搬运和计算流水化，避免 Vector 等 MTE 或 MTE 等 Vector。

---

## 15. Normal、SingleRow、SliceD 三类 kernel 怎么理解？

可以这样答：

> Normal 适合 UB 能容纳多行数据的情况，可以一次处理多个 row，吞吐更好。
> SingleRow 适合 UB 只能较稳妥地处理单行的情况。
> SliceD 适合 D 维很大，整行放不进 UB，需要沿 hidden dimension 分片处理。
> kernel 入口根据 tiling key 选择不同实现路径，代码里也能看到 TILING_KEY 1/2/3 分别调不同 kernel。

---

## 16. UB 不够时为什么 SliceD？

可以这样答：

> RMSNorm 的 reduce 维通常是 hidden dimension D。如果 D 很大，一整行的 x、y、gamma、smooth、临时 FP32 buffer 都放进 UB 会超出容量。
> SliceD 就是把 D 维切成多个片段，分片搬入和计算。
> 代价是 reduce 需要跨 slice 累积，可能要 workspace 保存中间统计，流程更复杂，但能支持更大 hidden size。

---

## 17. Tiling 的作用是什么？

可以这样答：

> Tiling 是 host 侧根据 shape、dtype、UB 大小、core 数等信息生成 kernel 执行策略。
> 它决定用多少 core、每个 core 处理多少行、每轮处理多少行、D 维是否切片、workspace 多大，以及 kernel 选择哪个 tiling key。
> 没有合理 tiling，kernel 要么 UB 放不下，要么 core 利用率低，要么搬运和计算不平衡。
> 你 PPT 里可以不突出 Tiling，但面试里一定要能解释。

---

## 18. TilingData 中字段怎么解释？

可以这样答：

> useCore 是使用的 AI Core 数。
> numFirstDim 是除 gamma 维外展平后的行数 N。
> numLastDim 是最后归一化维度 D。
> numLastDimAligned 是 D 对齐后的长度，方便 vector/block 操作。
> firstDimPerCore 是每个 core 处理的行数。
> firstDimPerLoop 是每轮在 UB 中处理的行数。
> lastDimSliceLen、lastDimLoopNum、lastDimSliceLenTail 用于 SliceD 模式下切分 D 维。
> smoothNum 表示 smooth 输入数量，epsilon 和 avgFactor 用于 RMS 计算。TilingData 的结构在 tiling 头文件里有定义。

---

## 19. InferShape 里 y1/y3/y4/x shape 为什么等于 x1？

可以这样答：

> y1 是对每个元素量化后的 INT8 输出，所以 shape 和 x1 一致。
> y3/y4 都是 RMSNorm 输出 y 的不同 dtype 表示，shape 也和 x1 一致。
> x 是 x1+x2 的结果，shape 和 x1/x2 一致。
> scale1/scale2 是按 row 量化得到的 scale，所以 shape 是 reduce shape，也就是去掉 gamma 对应的最后维度之后的前面维度。infer shape 代码里就是根据 xShape 和 gammaShape 推出 reduce shape。

---

## 20. 如果 smooth2 不存在，y2/scale2 怎么处理？

可以这样答：

> 从语义上说，smooth2 不存在时第二路量化无效，y2 和 scale2 没有实际意义。
> 工程上为了保持固定输出数量，可能给 y2/scale2 设置占位 shape，比如 `{1}`。
> README 里也明确说明 smoothScale2 不存在时，y2Out 和 scale2Out 输出无实际意义。

---

## 21. 性能提升 2.5 倍相对什么 baseline？

可以这样答：

> 这个要明确是相对融合前 Add + RmsNorm + DynamicQuant 分散执行的 baseline，而不是端到端模型吞吐。
> baseline 下多个算子各自读写 GM，中间结果落地，kernel 调度也更多。
> 融合后在一个 Ascend C kernel 内完成 Add、RMS 统计、RmsNorm、scale 计算和量化，减少 GM/UB 往返和中间结果写回，所以单算子链路性能提升最高 2.5 倍。

---

## 22. 这个算子是 memory-bound 还是 compute-bound？

可以这样答：

> 这个算子更偏 memory-bound。
> 它的计算主要是逐元素 Add/Mul/Cast/Round，加上 row 级 reduce 和 max，FLOPs 不算特别高，但需要读 x1/x2/gamma/smooth，写 y1/y2/y3/y4/x/scale，访存压力明显。
> 融合优化能带来 2.5 倍提升，说明瓶颈很大部分来自 GM 读写和中间结果落地，而不是纯计算能力。

---

## 23. 如何按张量大小估算节省带宽？

可以这样答：

> 假设输入 shape 是 `[N, D]`，dtype 是 FP16/BF16。
> 融合前 Add 输出 x 要写一次 GM，再被 RmsNorm 读一次；RmsNorm 输出 y 要写一次 GM，再被 DynamicQuant 读一次或两次。
> 融合后 x/y 可以在 UB 内复用，只在最终需要作为输出时写回。
> 所以节省的主要是中间 x/y 作为下游输入的重复写读，规模大约是若干个 `[N,D]` tensor 的读写量。D 越大、分支越多，收益越明显。

---

## 24. 怎么测算子耗时？

可以这样答：

> 用 msprof 或 Ascend Profiler 测 kernel task time。
> 要注意 warm-up、重复运行取平均或 P50/P90，确保 stream 同步，避免异步执行导致计时偏小。
> 还要固定 shape、dtype、smooth 输入组合，对比融合前多算子链路和融合后单算子链路。
> 如果看微观指标，还要看 Vector 利用率、MTE 搬运时间、UB 使用、pipeline stall 和 GM 带宽。

---

## 25. 如何公平比较 GPU 和 Ascend？

可以这样答：

> 不能只比绝对耗时，因为硬件峰值算力、带宽、内存层次不同。
> 我会比较两个层面：第一是用户感知的绝对 latency；第二是归一化利用率，比如有效带宽/峰值带宽、实际 FLOPS/峰值 FLOPS。
> 对这个算子来说更关注有效访存带宽和中间读写减少，而不是单纯 TFLOPS。
> 如果 GPU 用 Triton/CUDA fused kernel，比较时也要保证融合范围、输出数量、dtype、shape 和数据路径一致。

---

# 五、C++/系统基础类

## 1. C++ 模板在 Ascend C kernel 中有什么作用？

可以这样答：

> 模板主要用于 dtype、tiling key、buffer num 等编译期参数特化。
> 这样编译器可以在编译期展开分支、内联函数、优化寄存器和 buffer 使用，减少运行时判断。
> Ascend C kernel 对性能敏感，所以模板和 constexpr 很常见。

---

## 2. inline、constexpr 对性能有什么影响？

可以这样答：

> inline 可以减少函数调用开销，并帮助编译器跨函数优化。
> constexpr 把常量计算放到编译期，减少运行时开销，也能用于模板参数、数组大小和分支优化。
> 但 inline 不是强制内联，最终取决于编译器；过度内联也可能增加代码体积。

---

## 3. shared_ptr 什么时候不适合？

可以这样答：

> shared_ptr 有引用计数，涉及原子操作，多线程下会有额外开销。
> 高性能热路径里，如果所有权明确，更倾向 unique_ptr、裸指针引用或对象池。
> shared_ptr 适合所有权共享但生命周期复杂的场景，不适合极致性能内层循环。

---

## 4. vector 和 deque 内存布局区别？

可以这样答：

> vector 是连续内存，随机访问快，cache locality 好，但扩容可能整体搬迁。
> deque 是分段连续结构，头尾插入更稳定，但整体不完全连续，cache locality 不如 vector。
> 高性能计算和批量数据搬运通常优先 vector，因为连续内存更适合 SIMD、DMA 和 cache。

---

## 5. unordered_map 底层和 rehash 影响？

可以这样答：

> unordered_map 底层通常是哈希表，元素按 bucket 分布。
> 查找平均 O(1)，但哈希冲突严重会退化。
> rehash 会重新分配 bucket 并移动元素，可能导致性能抖动，迭代器也可能失效。
> 性能敏感场景要提前 reserve，减少 rehash。

---

## 6. 内存对齐为什么重要？

可以这样答：

> 对齐能让硬件按 block/vector 更高效访问内存，减少跨 cache line、跨 block 或非对齐搬运。
> 算子里 numLastDimAligned、BLOCK_SIZE 等就是为了让向量计算和 DataCopy 更高效。
> 非对齐访问可能导致额外 padding、额外搬运或性能下降。

---

## 7. memory order 怎么回答？

可以这样答：

> relaxed 只保证原子性，不保证顺序。
> acquire 用在读侧，保证后续读写不会重排到 acquire 前。
> release 用在写侧，保证之前读写不会重排到 release 后。
> acquire-release 常用于生产者消费者同步。
> seq_cst 是最强顺序，最容易理解但开销可能更高。
> 在高性能系统里要根据同步语义选择，不要无脑 seq_cst。

---

# 六、综合系统设计题

## 1. 设计一个支持 DeepSeek-V3/R1 的企业级推理平台

可以这样答：

> 我会分五层设计。
> 第一是接入层，提供 OpenAI API、鉴权、限流和请求路由。
> 第二是调度层，根据模型、上下文长度、缓存 locality、负载和 SLA 路由请求。
> 第三是推理引擎层，使用 vLLM/SGLang，支持 continuous batching、PagedAttention、prefix cache、PD 分离。
> 第四是缓存与通信层，使用 LMCache/Mooncake 做 KVCache 多级缓存和跨实例复用。
> 第五是集群编排层，用 KubeInfer/LWS/Volcano 做多机部署、Gang 调度、RankTable 和拓扑亲和。
> 监控上看 TTFT、ITL、QPS、tokens/s、cache hit、HCCL/NCCL、HBM、队列时间和错误率。

---

## 2. 80% 短请求 + 20% 64K 长请求，如何避免长请求拖垮短请求？

可以这样答：

> 首先做请求分类，按 prompt length 分队列。
> 短请求走低延迟队列，长请求走 chunked prefill 或专门长上下文实例。
> Scheduler 上限制单轮 prefill token budget，避免长 prefill 长时间占用计算资源。
> 缓存上对长请求启用 prefix cache/LMCache，减少重复 prefill。
> 如果资源允许，可以做长短请求隔离部署，避免 P99 互相影响。

---

## 3. 如何做 cache locality-aware routing？

可以这样答：

> 请求进入时先计算 prefix hash 或文档 hash，查询哪些实例有对应 KVCache。
> 路由时综合 cache hit、实例负载、队列长度和网络距离。
> 如果本地命中且负载可接受，优先路由到该实例；如果远端命中但传输成本高，要比较 remote retrieve 和 recompute 的成本。
> 这样能提高命中率，同时避免把所有请求打到同一个热点实例。

---

## 4. PD 分离下 Prefill:Decode 资源比例怎么调？

可以这样答：

> 取决于 workload 的输入输出比例。
> 长输入短输出，Prefill 压力大，需要更多 Prefill 资源。
> 短输入长输出，Decode 压力大，需要更多 Decode 资源。
> 可以监控 Prefill queue time、Decode ITL、KV transfer time 和各自利用率，动态调整 P/D 副本数。
> 目标是让 Prefill 不堆积，Decode ITL 稳定，KV 传输不成为瓶颈。

---

## 5. HCCL all-reduce 成为瓶颈怎么办？

可以这样答：

> 先确认是 TP 通信、EP 通信还是 PP stage 传输。
> 如果 TP all-reduce 瓶颈严重，可以降低 TP、增加 PP/DP，或者保证 TP 组在同一高性能通信域。
> 如果是 EP all-to-all，考虑优化 expert placement、DeepEP 类通信、负载均衡。
> 调度上通过拓扑感知把强通信组放近。
> 同时用 profiler 看通信是否能和计算 overlap。

---

## 6. RankTable 正确但 HCCL 初始化失败怎么排查？

可以这样答：

> 我会分层排查。
> 第一，检查 Pod 内看到的 RankTable 是否是最新版本，挂载路径是否正确。
> 第二，检查每个 rank 的环境变量、device id、local rank/global rank 是否一致。
> 第三，检查 Pod IP、端口、防火墙、RoCE 网络是否可达。
> 第四，检查 HCCL 版本、驱动、固件和 NPU 状态。
> 第五，查看 HCCL 日志定位是连接失败、超时、rank mismatch 还是设备错误。

---

# 七、数据真实性与 Ownership 追问

## 1. 这些性能数字怎么证明可信？

可以这样答：

> 我会对每个数字给出四要素：baseline、测试环境、workload、统计口径。
> 比如部署耗时降低 40%，说明从 apply Instance 到服务 ready 的时间。
> TTFT P90 降低 70%，说明模型、硬件、64K 输入、cache hit 场景和 P90 定义。
> 吞吐提升 3.1 倍，说明是 req/s 还是 tokens/s。
> 算子提升 2.5 倍，说明是单算子链路耗时对比，而不是端到端模型吞吐。
> 这样面试官追问时可以闭环。

---

## 2. 如果 reviewer 质疑 3.1 倍来自构造 workload，怎么回应？

可以这样答：

> 这个质疑是合理的。KVCache 优化的收益确实依赖 cache hit workload。
> 我会明确说 3.1 倍是针对长上下文重复前缀或 cache hit 场景，不代表所有线上流量。
> 同时我会补充不同命中率下的收益曲线，比如 miss/local hit/remote hit，以及在真实 RAG/Agent workload 下的平均收益。
> 这样不是夸大，而是明确适用边界。

---

## 3. 如果算子提升 2.5 倍但端到端提升不明显，怎么解释？

可以这样答：

> 单算子提升不一定线性转化为端到端提升。
> 如果该算子在模型总耗时中占比很低，根据 Amdahl 定律，端到端收益有限。
> 但它仍然有价值，因为它减少了关键路径中某个高频算子的访存开销。
> 端到端是否明显，还取决于 attention、MoE、通信、KVCache、scheduler 是否成为新的瓶颈。

---

## 4. 你最能体现技术深度的项目是哪个？

可以这样答：

> 我会选 KVCache 三级缓存项目，因为它同时涉及推理框架、缓存系统、NPU 适配和性能数据。
> 它不是简单调用 LMCache/Mooncake，而是要打通 vLLM PagedAttention KV layout、LMCache 连续 buffer、Mooncake 远端共享和昇腾 NPU 内存/通信机制。
> 同时它有明确性能收益：64K 场景 TTFT P90 降低、吞吐提升，并且能支撑 Agent/RAG 长上下文场景。
> 这个项目最能体现我对推理系统端到端瓶颈的理解。

---

# 八、压力面试连续追问：标准应对话术

## 1. “Deployment 也能管理多个 Pod，为什么非要 LWS？”

可以这样答：

> 是的，Deployment 能管理多个 Pod，但它管理的是扁平副本，不表达这些 Pod 之间的分布式推理关系。
> 多机推理需要组级生命周期、rank 拓扑、leader-worker 关系、成组调度和故障恢复。
> LWS 的价值不是“能起更多 Pod”，而是“能把多个 Pod 表达为一个分布式推理实例”。

---

## 2. “x1/x2 是输入，你为什么说中间结果写 GM？”

可以这样答：

> 这个说法需要修正。x1/x2 是输入，不是中间结果。
> 正确的访存减少点是 Add 的输出 x 和 RmsNorm 的输出 y。
> 融合前 x/y 会作为中间结果写回 GM，再被下游算子重新读入；融合后可以在一个 kernel 内部复用，减少 GM↔UB 往返。

---

## 3. “smooth2 能不能单独存在？”

可以这样答：

> 不能。语义上 smooth2 是第二路量化的可选输入，只有 smooth1 存在时 smooth2 才有意义。
> 如果只有 smooth2，没有 smooth1，输入组合不合法。
> 代码的 shape/base info 校验里也会拒绝 smooth2 单独存在的情况，README 也说明了不同 smooth 输入组合下 y2/scale2 的有效性。

---

## 4. “2.8 倍、2.5 倍、3.1 倍分别是什么？”

可以这样答：

> 这三个数字口径不同。
> 2.8 倍是 KVCache 搬运链路的 microbenchmark 提升，来自连续区间合并和异步拷贝。
> 2.5 倍是 AddRmsNormDynamicQuantV2 融合算子的单算子链路性能提升。
> 3.1 倍是 64K 长上下文 cache hit 场景下系统整体吞吐提升。
> 我会明确区分 microbenchmark、单算子 benchmark 和端到端 serving benchmark，避免混淆。

---

# 九、最后给你的答题策略

面试时你最好始终用这个结构回答：

> **背景 → 瓶颈 → 方案 → 难点 → 数据 → 边界**

比如讲 KubeInfer：

> 背景是多机推理和 PD 分离部署复杂；瓶颈是 Deployment 缺乏分布式推理组语义；方案是引入 LWS、RankTable 自动生成、Webhook 校验和 Gang/拓扑调度；难点是 Kubernetes 异步状态和 Rank 稳定性；数据是部署耗时降低 40%；边界是单机单卡场景不一定需要 LWS。

讲 KVCache：

> 背景是长上下文 KVCache 显存和 TTFT 瓶颈；瓶颈是重复 prefill 和 KV 搬运；方案是 LMCache/Mooncake + HBM/DRAM/SSD + multi_layer_kv_transfer；难点是分页 KV 和连续 buffer 转换；数据是 TTFT P90 降低 70%、吞吐 3.1 倍；边界是收益依赖 cache hit ratio。

讲 Ascend C：

> 背景是 DeepSeek V3 中 Add/RMSNorm/DynamicQuant 高频出现；瓶颈是中间结果 GM 落地和重复搬入搬出；方案是图模式融合为 AddRmsNormDynamicQuantV2；难点是数值一致性、可选 smooth 分支、shape/tiling 和 UB 管理；数据是关键算子性能提升 2.5 倍；边界是端到端收益取决于该算子占比。

这套逻辑比单纯背答案更重要。面试官真正想判断的是：你能不能把**项目、原理、指标和边界条件**都讲清楚。
