#pragma once

// 中文注释：本文件定义了 vLLM 自定义的 GEMM Collective Builder（集合构建器）。
// CUTLASS 的 CollectiveBuilder 用于构建 GEMM 操作的核心计算集合（collective），
// 包括数据加载、计算和存储的完整流水线。
//
// VLLMCollectiveBuilder 是对标准 CollectiveBuilder 的封装，引入了 KernelTag 模板参数，
// 允许 vLLM 注入自定义的 kernel 标签来构建自定义的 collective 操作。
// 这样可以在不修改 CUTLASS 库头文件的情况下扩展 GEMM 行为。
//
// 使用方式：
//   - 传入 CutlassKernelTag：使用标准 CUTLASS collective（回退到默认行为）
//   - 传入自定义 KernelTag：构建自定义 collective（需要在其他地方特化）

#include "cutlass/gemm/collective/collective_builder.hpp"

namespace cutlass::gemm::collective {
using namespace cute;

//
// VLLMCollectiveBuilder is a wrapper around CollectiveBuilder that allows for
// for custom kernel tags, allowing you to build custom collectives. Without
// touching the cutlass library headers, using `CutlassKernelTag` will mean it
// will resort to using the standard cutlass collective builder.
//

// Use the default Cutlass collective builder, i.e. use an unmodified cutless
// collective
// 中文注释：默认的 CUTLASS kernel 标签，使用此标签时将回退到标准 CUTLASS collective builder。
struct CutlassKernelTag {};

// 中文注释：VLLMCollectiveBuilder 的主模板声明。
// 当 KernelTag 没有匹配到任何特化时，static_assert 会失败，提示无法为给定参数构建 collective。
template <class KernelTag, class ArchTag, class OpClass, class ElementA,
          class GmemLayoutA, int AlignmentA, class ElementB, class GmemLayoutB,
          int AlignmentB, class ElementAccumulator, class TileShape_MNK,
          class ClusterShape_MNK, class StageCountType,
          class KernelScheduleType, class Enable = void>
struct VLLMCollectiveBuilder {
  static_assert(sizeof(ElementA) == 0,
                "Could not build a collective for given parameters.");
};

// 中文注释：当 KernelTag 为 CutlassKernelTag 时的特化。
// 直接委托给标准 CUTLASS 的 CollectiveBuilder，使用其默认的 CollectiveOp。
// 这是"回退到标准行为"的路径。
template <class ArchTag, class OpClass, class ElementA, class GmemLayoutA,
          int AlignmentA, class ElementB, class GmemLayoutB, int AlignmentB,
          class ElementAccumulator, class TileShape_MNK, class ClusterShape_MNK,
          class StageCountType, class KernelScheduleType>
struct VLLMCollectiveBuilder<
    CutlassKernelTag, ArchTag, OpClass, ElementA, GmemLayoutA, AlignmentA,
    ElementB, GmemLayoutB, AlignmentB, ElementAccumulator, TileShape_MNK,
    ClusterShape_MNK, StageCountType, KernelScheduleType> {
  using CollectiveOp = typename CollectiveBuilder<
      ArchTag, OpClass, ElementA, GmemLayoutA, AlignmentA, ElementB,
      GmemLayoutB, AlignmentB, ElementAccumulator, TileShape_MNK,
      ClusterShape_MNK, StageCountType, KernelScheduleType>::CollectiveOp;
};

};  // namespace cutlass::gemm::collective