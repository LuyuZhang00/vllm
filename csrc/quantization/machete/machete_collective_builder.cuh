#pragma once

// =============================================================================
// 中文注释: Machete CUTLASS Collective Builder
// =============================================================================
// 本文件将 Machete 的自定义 MMA 操作注册到 CUTLASS 的 Collective Builder 框架中。
//
// CUTLASS Collective Builder 是 CUTLASS 3.x 的核心抽象，它将 GEMM 分解为:
//   - Collective Mainloop: 负责矩阵乘法的主循环 (加载、计算、存储)
//   - Collective Epilogue: 负责后处理 (缩放、激活函数等)
//
// 本文件的作用:
//   1. 定义 MacheteKernelTag 作为 Machete kernel 的标识
//   2. 通过模板特化，将 MacheteCollectiveMma 注册为 VLLMCollectiveBuilder
//      在 SM90 + TensorOp 条件下的实现
//   3. 支持多种 KernelSchedule:
//      - KernelTmaWarpSpecialized: 基础的 TMA warp 专用化调度
//      - KernelTmaWarpSpecializedPingpong: Pingpong 调度 (双缓冲优化)
//      - KernelTmaWarpSpecializedCooperative: 协作调度 (多 warp 协作)
//
// TMA (Tensor Memory Accelerator) 是 Hopper (SM90) 引入的硬件单元，
// 可以异步地将数据从全局内存传输到共享内存，不占用计算单元。
// =============================================================================

#include "cutlass_extensions/vllm_collective_builder.cuh"
#include "machete_mainloop.cuh"

namespace cutlass::gemm::collective {
using namespace cute;

// 中文注释: Machete kernel 标识类型，用于模板分发
struct MacheteKernelTag {};

// 中文注释: 将 MacheteCollectiveMma 注册到 CUTLASS 的 Collective Builder
// 当 KernelScheduleType 是 TMA warp 专用化变体时，使用 Machete 的自定义实现
template <class ElementPairA_, class GmemLayoutA_, int AlignmentA,
          class ElementPairB_, class GmemLayoutB_, int AlignmentB,
          class ElementAccumulator, class TileShape_MNK, class ClusterShape_MNK,
          class StageCountType, class KernelScheduleType>
struct VLLMCollectiveBuilder<
    MacheteKernelTag, arch::Sm90, arch::OpClassTensorOp, ElementPairA_,
    GmemLayoutA_, AlignmentA, ElementPairB_, GmemLayoutB_, AlignmentB,
    ElementAccumulator, TileShape_MNK, ClusterShape_MNK, StageCountType,
    KernelScheduleType,
    cute::enable_if_t<(
        cute::is_same_v<KernelScheduleType, KernelTmaWarpSpecialized> ||
        cute::is_same_v<KernelScheduleType, KernelTmaWarpSpecializedPingpong> ||
        cute::is_same_v<KernelScheduleType,
                        KernelTmaWarpSpecializedCooperative>)>> {
  using CollectiveOp = machete::MacheteCollectiveMma<
      ElementPairA_, GmemLayoutA_, AlignmentA, ElementPairB_, GmemLayoutB_,
      AlignmentB, ElementAccumulator, TileShape_MNK, ClusterShape_MNK,
      StageCountType, KernelScheduleType>;
};

};  // namespace cutlass::gemm::collective
