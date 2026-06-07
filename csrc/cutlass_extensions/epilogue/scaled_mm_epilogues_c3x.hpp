#pragma once

#include "cutlass_extensions/epilogue/broadcast_load_epilogue_c3x.hpp"
#include "cutlass_extensions/epilogue/broadcast_load_epilogue_array_c3x.hpp"

// This header is shared by both _C (unstable ABI) and _C_stable_libtorch
// (stable ABI) targets. When compiled under the stable ABI target,
// TORCH_TARGET_VERSION is defined and Tensor is unavailable, so we
// use torch::stable::Tensor instead.
#ifdef TORCH_TARGET_VERSION
  #include <torch/csrc/stable/tensor.h>
#endif

/*
   This file defines custom epilogues for fusing channel scales, token scales,
   bias, and activation zero-points onto a GEMM operation using the
   CUTLASS 3.x API, for NVIDIA GPUs with sm90a (Hopper) or later.

   Epilogues must contain a public type named EVTCompute of type Sm90EVT,
   as well as a static prepare_args function that constructs an
   EVTCompute::Arguments struct.
*/
/*
   中文注释：本文件定义了用于量化矩阵乘法（Scaled Matrix Multiplication）的
   SM90 epilogue 函数集合，类似于 PyTorch 的 torch.scaled_mm。

   核心概念 - Epilogue Visitor Tree (EVT)：
     CUTLASS 3.x 使用树形结构的 epilogue visitor 来组织后处理操作。
     每个节点是一个计算操作（如乘法、加法），叶子节点是数据加载操作。
     例如 ScaledEpilogue 的计算树为：
       D = ScaleA * (ScaleB * Accumulator)
           Compute1
           ├── ScaleA (列广播加载)
           └── Compute0
               ├── ScaleB (行广播加载)
               └── Accumulator (从寄存器获取)

   支持的 epilogue 类型（按复杂度递增）：
     1. TrivialEpilogue：无后处理，直接输出
     2. ScaledEpilogue：D = a_scale * (b_scale * Accum)
     3. ScaledEpilogueBias：D = a_scale * (b_scale * Accum) + bias
     4. ScaledEpilogueColumnBias：同上，但 bias 是列向量（用于稀疏 GEMM）
     5. ScaledEpilogueBiasAzp：带 per-tensor 非对称零点修正
     6. ScaledEpilogueBiasAzpToken：带 per-token 非对称零点修正
     7. ScaledEpilogueArray：group GEMM 使用的数组版缩放

   支持的量化模式：
     - 对称量化（zero_point = 0）：只需 scale
     - 非对称量化（zero_point != 0）：需要 scale + azp（激活零点）
     - per-tensor / per-token / per-channel 的任意组合
*/

namespace vllm::c3x {

#ifdef TORCH_TARGET_VERSION
using TensorType = torch::stable::Tensor;
#else
using TensorType = torch::Tensor;
#endif

using namespace cute;

// 中文注释：恒等函数，不做任何变换。
template <typename T>
struct identity {
  CUTLASS_HOST_DEVICE
  T operator()(T lhs) const { return lhs; }
};

// 中文注释：平凡（trivial）epilogue，不做任何后处理。
// 仅将累加器（accumulator）的类型截断/转换为目标类型 ElementD 后输出。
// 用于不需要缩放、偏置等操作的场景。
template <typename ElementAcc, typename ElementD, typename TileShape>
struct TrivialEpilogue {
 private:
  using Accum = cutlass::epilogue::fusion::Sm90AccFetch;
  using Compute = cutlass::epilogue::fusion::Sm90Compute<
      cutlass::epilogue::thread::Identity, ElementD, ElementAcc,
      cutlass::FloatRoundStyle::round_to_nearest>;

 public:
  using EVTCompute = cutlass::epilogue::fusion::Sm90EVT<Compute, Accum>;
  using ArgumentType = typename EVTCompute::Arguments;

  template <typename... Args>
  static ArgumentType prepare_args(Args... args) {
    return {};
  }
};

/*
 * This class provides the common load descriptors for the
 * ScaledEpilogue[...] classes
 */
// 中文注释：ScaledEpilogue 系列的基类，提供通用的加载描述符（load descriptor）。
// 加载描述符定义了如何从全局内存加载量化参数（scale、bias、azp 等）。
// 每种描述符对应一种广播模式：
//   - ColOrScalarLoad<T>：列方向广播或标量广播（用于 scale_A 或 per-token 参数）
//   - RowOrScalarLoad<T>：行方向广播或标量广播（用于 scale_B 或 per-channel 参数）
//   - ColLoad<T>：纯列向量加载（用于 per-token 参数，不支持标量）
//   - RowLoad<T>：纯行向量加载（用于 bias 等）
//   - ColOrScalarLoadArray<T> / RowOrScalarLoadArray<T>：数组版本，用于 group GEMM
template <typename ElementAcc, typename ElementD, typename TileShape>
struct ScaledEpilogueBase {
 protected:
  using Accum = cutlass::epilogue::fusion::Sm90AccFetch;

  template <typename T>
  using ColOrScalarLoad = cutlass::epilogue::fusion::Sm90ColOrScalarBroadcast<
      0 /*Stages*/, TileShape, T, Stride<Int<1>, Int<0>, Int<0>>>;

  template <typename T>
  using RowOrScalarLoad = cutlass::epilogue::fusion::Sm90RowOrScalarBroadcast<
      0 /*Stages*/, TileShape, T, Stride<Int<0>, Int<1>, Int<0>>>;

  // Don't want to support nullptr by default
  template <typename T, bool EnableNullPtr = false>
  using ColLoad = cutlass::epilogue::fusion::Sm90ColBroadcast<
      0 /*Stages*/, TileShape, T, T, Stride<Int<1>, Int<0>, Int<0>>,
      128 / sizeof_bits_v<T>, EnableNullPtr>;

  // Don't want to support nullptr by default
  template <typename T, bool EnableNullPtr = false>
  using RowLoad = cutlass::epilogue::fusion::Sm90RowBroadcast<
      0 /*Stages*/, TileShape, T, T, Stride<Int<0>, Int<1>, Int<0>>,
      128 / sizeof_bits_v<T>, EnableNullPtr>;

  template <typename T>
  using ColOrScalarLoadArray =
      cutlass::epilogue::fusion::Sm90ColOrScalarBroadcastArray<
          0 /*Stages*/, TileShape, T, Stride<Int<1>, Int<0>, Int<0>>>;

  template <typename T>
  using RowOrScalarLoadArray =
      cutlass::epilogue::fusion::Sm90RowOrScalarBroadcastArray<
          0 /*Stages*/, TileShape, T, Stride<Int<0>, Int<1>, Int<0>>>;

  // This utility function constructs the arguments for the load descriptors
  // from a tensor. It can handle both row and column, as well as row/column or
  // scalar cases.
  // 中文注释：从 PyTorch Tensor 构造加载描述符的参数。
  // 对于 ColOrScalarLoad/RowOrScalarLoad，会检查 tensor.numel() 是否为 1：
  //   - numel == 1：标量广播模式（row_broadcast/col_broadcast = false）
  //   - numel > 1：向量广播模式（row_broadcast/col_broadcast = true）
  // 这样一个函数就能处理 per-tensor（标量）和 per-channel/per-token（向量）两种情况。
  template <typename Descriptor, typename T>
  static auto args_from_tensor(TensorType const& tensor) {
    using Arguments = typename Descriptor::Arguments;
    auto* data_ptr = static_cast<T*>(tensor.data_ptr());
    if constexpr (std::is_same_v<Descriptor, ColOrScalarLoad<T>> ||
                  std::is_same_v<Descriptor, RowOrScalarLoad<T>>) {
      return Arguments{data_ptr, tensor.numel() != 1};
    } else {
      static_assert(!std::is_same_v<Descriptor, ColLoad<T, true>> &&
                    !std::is_same_v<Descriptor, RowLoad<T, true>>);
      return Arguments{data_ptr};
    }
  }

  // This overload handles the case where there might not be a tensor, in which
  // case a nullptr is passed and a constant (0) is used.
  template <typename Descriptor, typename T>
  static auto args_from_tensor(std::optional<TensorType> const& tensor) {
    using Arguments = typename Descriptor::Arguments;
    auto* data_ptr = tensor ? static_cast<T*>(tensor->data_ptr()) : nullptr;
    static_assert(std::is_same_v<Descriptor, ColLoad<T, true>> ||
                  std::is_same_v<Descriptor, RowLoad<T, true>>);
    return Arguments{data_ptr};
  }

  template <typename Descriptor, typename T>
  static auto args_from_tensor(const T* const* data_ptr, bool do_broadcast) {
    using Arguments = typename Descriptor::Arguments;
    static_assert(std::is_same_v<Descriptor, ColOrScalarLoadArray<T>> ||
                  std::is_same_v<Descriptor, RowOrScalarLoadArray<T>>);
    return Arguments{data_ptr, do_broadcast};
  }
};

/*
   This epilogue function defines a quantized GEMM operation similar to
   torch.scaled_mm_.

   A and B may be both either int8 or fp8_e4m3. A can be
   quantized per-tensor or per-row. B can be quantized per-tensor or per-column.
   Any combination of per-tensor and per-row or column is supported.
   A and B must have symmetric quantization (zero point == 0).

   So the GEMM operation is D = (a_scales * A) (b_scales * B), where the
   scales are applied elementwise with numpy-style broadcasting.

   ScaleA and ScaleB define the epilogue functions that apply the scales for
   the A and B operands respectively. These scales may be either per-tensor or
   per row or column.
*/
// 中文注释：对称量化的缩放 epilogue，实现 D = a_scale * (b_scale * Accumulator)。
// 这是最基本的量化矩阵乘法 epilogue，等价于 torch.scaled_mm。
//
// EVT 计算树结构：
//   Compute1 (a_scale * temp -> D)
//   ├── ScaleA (ColOrScalarLoad: 加载 a_scale，per-tensor 或 per-token)
//   └── EVTCompute0
//       ├── ScaleB (RowOrScalarLoad: 加载 b_scale，per-tensor 或 per-channel)
//       └── Accumulator (从寄存器获取累加结果)
//
// 所有计算在 float 精度下进行，最终截断为 ElementD 输出。
template <typename ElementAcc, typename ElementD, typename TileShape>
struct ScaledEpilogue
    : private ScaledEpilogueBase<ElementAcc, ElementD, TileShape> {
 private:
  using SUPER = ScaledEpilogueBase<ElementAcc, ElementD, TileShape>;
  using Accum = typename SUPER::Accum;
  using ScaleA = typename SUPER::template ColOrScalarLoad<float>;
  using ScaleB = typename SUPER::template RowOrScalarLoad<float>;

  using Compute0 = cutlass::epilogue::fusion::Sm90Compute<
      cutlass::multiplies, float, float,
      cutlass::FloatRoundStyle::round_to_nearest>;

  using EVTCompute0 =
      cutlass::epilogue::fusion::Sm90EVT<Compute0, ScaleB, Accum>;

  using Compute1 = cutlass::epilogue::fusion::Sm90Compute<
      cutlass::multiplies, ElementD, float,
      cutlass::FloatRoundStyle::round_to_nearest>;

 public:
  using EVTCompute =
      cutlass::epilogue::fusion::Sm90EVT<Compute1, ScaleA, EVTCompute0>;
  using ArgumentType = typename EVTCompute::Arguments;

  static ArgumentType prepare_args(TensorType const& a_scales,
                                   TensorType const& b_scales) {
    auto a_args = SUPER::template args_from_tensor<ScaleA, float>(a_scales);
    auto b_args = SUPER::template args_from_tensor<ScaleB, float>(b_scales);

    typename EVTCompute0::Arguments evt0_args{b_args, {}, {}};
    return ArgumentType{a_args, evt0_args, {}};
  }
};

/*
 * This epilogue performs the same operation as ScaledEpilogue, but adds a bias.
 * This bias can also be used in the per-tensor azp case, where the activation
 * zero point (azp) is used to compute an azp correction term,
 * which is folded into the bias.
 *
 * The bias tensor must be per-output channel.
 * ScaleA and ScaleB can be per-tensor or per-token/per-channel.
 */
// 中文注释：带偏置的缩放 epilogue，实现 D = a_scale * (b_scale * Accumulator) + bias。
// bias 必须是 per-output channel 的行向量。
//
// 与 ScaledEpilogue 的区别：
//   Compute1 使用 homogeneous_multiply_add 而非 multiplies，
//   即执行 a * b + c 的融合乘加操作，将 bias 融合到计算中。
//
// 对于非对称量化场景，azp 修正项可以预计算后折叠到 bias 中。
template <typename ElementAcc, typename ElementD, typename TileShape>
struct ScaledEpilogueBias
    : private ScaledEpilogueBase<ElementAcc, ElementD, TileShape> {
 private:
  using SUPER = ScaledEpilogueBase<ElementAcc, ElementD, TileShape>;
  using Accum = typename SUPER::Accum;
  using ScaleA = typename SUPER::template ColOrScalarLoad<float>;
  using ScaleB = typename SUPER::template RowOrScalarLoad<float>;
  using Bias = typename SUPER::template RowLoad<ElementD>;

  using Compute0 = cutlass::epilogue::fusion::Sm90Compute<
      cutlass::multiplies, float, float,
      cutlass::FloatRoundStyle::round_to_nearest>;

  using EVTCompute0 =
      cutlass::epilogue::fusion::Sm90EVT<Compute0, ScaleB, Accum>;

  using Compute1 = cutlass::epilogue::fusion::Sm90Compute<
      cutlass::homogeneous_multiply_add, ElementD, float,
      cutlass::FloatRoundStyle::round_to_nearest>;

 public:
  using EVTCompute =
      cutlass::epilogue::fusion::Sm90EVT<Compute1, ScaleA, EVTCompute0, Bias>;

  using ArgumentType = typename EVTCompute::Arguments;
  static ArgumentType prepare_args(TensorType const& a_scales,
                                   TensorType const& b_scales,
                                   TensorType const& bias) {
    auto a_args = SUPER::template args_from_tensor<ScaleA, float>(a_scales);
    auto b_args = SUPER::template args_from_tensor<ScaleB, float>(b_scales);
    auto bias_args = SUPER::template args_from_tensor<Bias, ElementD>(bias);

    typename EVTCompute0::Arguments evt0_args{b_args, {}, {}};
    return ArgumentType{a_args, evt0_args, bias_args, {}};
  }
};

/*
 * This epilogue performs the same operation as ScaledEpilogueBias, but the
 * bias is a column vector instead of a row vector. Useful e.g. if we are
 * computing a GEMM via C^T += B^T A^T. This happens in the 2:4 sparse kernels.
 */
// 中文注释：带列向量偏置的缩放 epilogue。
// 与 ScaledEpilogueBias 相同的计算，但 bias 是列向量（per-row）而非行向量。
// 用于稀疏 GEMM（2:4 structured sparsity）中，此时 GEMM 的转置关系导致
// 原本的行偏置变成了列偏置。
template <typename ElementAcc, typename ElementD, typename TileShape>
struct ScaledEpilogueColumnBias
    : private ScaledEpilogueBase<ElementAcc, ElementD, TileShape> {
 private:
  using SUPER = ScaledEpilogueBase<ElementAcc, ElementD, TileShape>;
  using Accum = typename SUPER::Accum;
  using ScaleA = typename SUPER::template ColOrScalarLoad<float>;
  using ScaleB = typename SUPER::template RowOrScalarLoad<float>;
  using Bias = typename SUPER::template ColLoad<ElementD>;

  using Compute0 = cutlass::epilogue::fusion::Sm90Compute<
      cutlass::multiplies, float, float,
      cutlass::FloatRoundStyle::round_to_nearest>;

  using EVTCompute0 =
      cutlass::epilogue::fusion::Sm90EVT<Compute0, ScaleB, Accum>;

  using Compute1 = cutlass::epilogue::fusion::Sm90Compute<
      cutlass::homogeneous_multiply_add, ElementD, float,
      cutlass::FloatRoundStyle::round_to_nearest>;

 public:
  using EVTCompute =
      cutlass::epilogue::fusion::Sm90EVT<Compute1, ScaleA, EVTCompute0, Bias>;

  using ArgumentType = typename EVTCompute::Arguments;
  static ArgumentType prepare_args(TensorType const& a_scales,
                                   TensorType const& b_scales,
                                   TensorType const& bias) {
    auto a_args = SUPER::template args_from_tensor<ScaleA, float>(a_scales);
    auto b_args = SUPER::template args_from_tensor<ScaleB, float>(b_scales);
    auto bias_args = SUPER::template args_from_tensor<Bias, ElementD>(bias);

    typename EVTCompute0::Arguments evt0_args{b_args, {}, {}};
    return ArgumentType{a_args, evt0_args, bias_args, {}};
  }
};

/*
 * This epilogue directly supports per-tensor azp in int32 form.
 * As opposed to the per-token epilogue below, this epilogue only has an azp_adj
 * term, which should already be multiplied with the scalar azp.
 * The azp_adj term is a 1D tensor of shape (1,n), computed as azp * J @ B.
 *
 * This epilogue also supports bias, which remains per-channel.
 */
// 中文注释：带 per-tensor 非对称零点（azp）修正的缩放 epilogue。
//
// 非对称量化的数学模型：
//   量化：q = round(x / scale) + azp
//   反量化：x = (q - azp) * scale
//
// 当 A 使用非对称量化时，GEMM 需要修正 azp 的影响：
//   D = a_scale * b_scale * (QuantA @ QuantB)
//     - a_scale * b_scale * azp * J @ B
//   其中 J 是全 1 矩阵。
//
// 本 epilogue 中 azp_adj = azp * J @ B 已预计算好（形状 (1,n)），
// 直接在 epilogue 中从累加器中减去。
//
// EVT 计算树：
//   ComputeScaleBiasA (a_scale * temp + bias -> D)
//   ├── ScaleA
//   └── EVTComputeScaleB (b_scale * temp -> temp)
//       ├── ScaleB
//       └── EVTComputeAzp (Accumulator - azp_adj -> temp)
//           ├── Accumulator
//           └── AzpWithAdj (加载 azp_adj)
template <typename ElementAcc, typename ElementD, typename TileShape>
struct ScaledEpilogueBiasAzp
    : private ScaledEpilogueBase<ElementAcc, ElementD, TileShape> {
 private:
  using SUPER = ScaledEpilogueBase<ElementAcc, ElementD, TileShape>;
  using Accum = typename SUPER::Accum;
  using ScaleA = typename SUPER::template ColOrScalarLoad<float>;
  using ScaleB = typename SUPER::template RowOrScalarLoad<float>;
  using Bias = typename SUPER::template RowLoad<ElementD, true>;

  // This is the full AZP term, azp * J @ B, shape (1,n)
  using AzpWithAdj = typename SUPER::template RowLoad<int32_t>;

  // Compute float(accum - azp_adj), both operands are int32_t
  using ComputeAzp = cutlass::epilogue::fusion::Sm90Compute<
      cutlass::minus, float, int32_t,
      cutlass::FloatRoundStyle::round_to_nearest>;

  using EVTComputeAzp =
      cutlass::epilogue::fusion::Sm90EVT<ComputeAzp, Accum, AzpWithAdj>;

  using ComputeScaleB = cutlass::epilogue::fusion::Sm90Compute<
      cutlass::multiplies, float, float,
      cutlass::FloatRoundStyle::round_to_nearest>;

  using EVTComputeScaleB =
      cutlass::epilogue::fusion::Sm90EVT<ComputeScaleB, ScaleB, EVTComputeAzp>;

  using ComputeScaleBiasA = cutlass::epilogue::fusion::Sm90Compute<
      cutlass::homogeneous_multiply_add, ElementD, float,
      cutlass::FloatRoundStyle::round_to_nearest>;

 public:
  using EVTCompute =
      cutlass::epilogue::fusion::Sm90EVT<ComputeScaleBiasA, ScaleA,
                                         EVTComputeScaleB, Bias>;
  using ArgumentType = typename EVTCompute::Arguments;

  static ArgumentType prepare_args(TensorType const& a_scales,
                                   TensorType const& b_scales,
                                   TensorType const& azp_adj,
                                   std::optional<TensorType> const& bias) {
    auto a_args = SUPER::template args_from_tensor<ScaleA, float>(a_scales);
    auto b_args = SUPER::template args_from_tensor<ScaleB, float>(b_scales);
    auto bias_args = SUPER::template args_from_tensor<Bias, ElementD>(bias);
    auto azp_adj_args =
        SUPER::template args_from_tensor<AzpWithAdj, int32_t>(azp_adj);

    typename EVTComputeAzp::Arguments evt_azp_args{{}, azp_adj_args, {}};
    typename EVTComputeScaleB::Arguments evt_scale_b_args{
        b_args, evt_azp_args, {}};
    return ArgumentType{a_args, evt_scale_b_args, bias_args, {}};
  }
};

/*
 * This epilogue supports per-token azp by computing and applying
 * the correction term using a rank-1 update. If the term were materialized,
 * it would require O(m*n) space, and this way it only requires O(m+n) space.
 * The azp term is a 1D tensor of shape (m,1), and represents the unscaled zero
 * point for each row of A.
 * The azp_adj term is a 1D tensor of shape (1,n), computed as J @ B.
 *
 * This epilogue also supports bias, which remains per-channel.
 */
// 中文注释：带 per-token 非对称零点修正的缩放 epilogue。
//
// 与 ScaledEpilogueBiasAzp 的区别：
//   - per-tensor azp：azp 是标量，azp_adj = azp * (J @ B) 已是完整修正项
//   - per-token azp：azp 是 (m,1) 向量（每行不同），azp_adj = J @ B 是 (1,n) 向量
//     需要在 epilogue 中实时计算 azp * azp_adj 的秩 1 修正
//
// 内存效率：如果显式计算 azp * azp_adj 需要 O(m*n) 空间，
// 而本实现通过秩 1 外积在 epilogue 中隐式计算，只需 O(m+n) 空间。
//
// 计算流程（EVT 树）：
//   D = a_scale * (b_scale * (Accumulator - azp * azp_adj)) + bias
template <typename ElementAcc, typename ElementD, typename TileShape>
struct ScaledEpilogueBiasAzpToken
    : private ScaledEpilogueBase<ElementAcc, ElementD, TileShape> {
 private:
  using SUPER = ScaledEpilogueBase<ElementAcc, ElementD, TileShape>;
  using Accum = typename SUPER::Accum;
  using ScaleA = typename SUPER::template ColOrScalarLoad<float>;
  using ScaleB = typename SUPER::template RowOrScalarLoad<float>;
  using Bias = typename SUPER::template RowLoad<ElementD, true>;

  // Per-token azp term, shape (m,1)
  using Azp = typename SUPER::template ColLoad<int32_t>;

  // This is the AZP adjustment term, J @ B, shape (1,n)
  using AzpAdj = typename SUPER::template RowLoad<int32_t>;

  // Compute azp * azp_adj
  using ComputeAzp = cutlass::epilogue::fusion::Sm90Compute<
      cutlass::multiplies, int32_t, int32_t,
      cutlass::FloatRoundStyle::round_to_nearest>;

  using EVTComputeAzp =
      cutlass::epilogue::fusion::Sm90EVT<ComputeAzp, Azp, AzpAdj>;

  // Compute float(accum - azp*azp_adj), all operands are int32_t
  using ComputeAcc = cutlass::epilogue::fusion::Sm90Compute<
      cutlass::minus, float, int32_t,
      cutlass::FloatRoundStyle::round_to_nearest>;

  using EVTComputeAcc =
      cutlass::epilogue::fusion::Sm90EVT<ComputeAcc, Accum, EVTComputeAzp>;

  using ComputeScaleB = cutlass::epilogue::fusion::Sm90Compute<
      cutlass::multiplies, float, float,
      cutlass::FloatRoundStyle::round_to_nearest>;

  using EVTComputeScaleB =
      cutlass::epilogue::fusion::Sm90EVT<ComputeScaleB, ScaleB, EVTComputeAcc>;

  using ComputeScaleBiasA = cutlass::epilogue::fusion::Sm90Compute<
      cutlass::homogeneous_multiply_add, ElementD, float,
      cutlass::FloatRoundStyle::round_to_nearest>;

 public:
  using EVTCompute =
      cutlass::epilogue::fusion::Sm90EVT<ComputeScaleBiasA, ScaleA,
                                         EVTComputeScaleB, Bias>;
  using ArgumentType = typename EVTCompute::Arguments;

  static ArgumentType prepare_args(TensorType const& a_scales,
                                   TensorType const& b_scales,
                                   TensorType const& azp_adj,
                                   TensorType const& azp,
                                   std::optional<TensorType> const& bias) {
    auto a_args = SUPER::template args_from_tensor<ScaleA, float>(a_scales);
    auto b_args = SUPER::template args_from_tensor<ScaleB, float>(b_scales);
    auto bias_args = SUPER::template args_from_tensor<Bias, ElementD>(bias);
    auto azp_args = SUPER::template args_from_tensor<Azp, int32_t>(azp);
    auto azp_adj_args =
        SUPER::template args_from_tensor<AzpAdj, int32_t>(azp_adj);

    typename EVTComputeAzp::Arguments evt_azp_args{azp_args, azp_adj_args, {}};
    typename EVTComputeAcc::Arguments evt_acc_args{{}, evt_azp_args, {}};
    typename EVTComputeScaleB::Arguments evt_scale_b_args{
        b_args, evt_acc_args, {}};
    return ArgumentType{a_args, evt_scale_b_args, bias_args, {}};
  }
};

/*
    This epilogue works like ScaledEpilogue, but ScaleA and ScaleB are pointers
    to arrays containing different scales used in group gemm. The number of
   pointers in ScaleA and the number of pointers in ScaleB are equal to the
   group size.
*/
// 中文注释：Group GEMM 使用的数组版缩放 epilogue。
//
// Group GEMM 是将多个小 GEMM 合并为一个大 GEMM 的技术。
// 每个子 GEMM 可能有不同的 scale_A 和 scale_B。
// ScaleA 和 ScaleB 是指针数组，每个元素指向对应子 GEMM 的缩放因子。
// 内部通过 batch 维度 L 索引到正确的 scale。
//
// 这对于 MoE（Mixture of Experts）模型中的批量矩阵乘法非常有用，
// 因为每个 expert 可能有不同的量化参数。
template <typename ElementAcc, typename ElementD, typename EpilogueDescriptor>
struct ScaledEpilogueArray
    : private ScaledEpilogueBase<ElementAcc, ElementD, EpilogueDescriptor> {
 private:
  using SUPER = ScaledEpilogueBase<ElementAcc, ElementD, EpilogueDescriptor>;
  using Accum = typename SUPER::Accum;
  using ScaleA = typename SUPER::template ColOrScalarLoadArray<float>;
  using ScaleB = typename SUPER::template RowOrScalarLoadArray<float>;

  using Compute0 = cutlass::epilogue::fusion::Sm90Compute<
      cutlass::multiplies, float, float,
      cutlass::FloatRoundStyle::round_to_nearest>;

  using EVTCompute0 =
      cutlass::epilogue::fusion::Sm90EVT<Compute0, ScaleB, Accum>;

  using Compute1 = cutlass::epilogue::fusion::Sm90Compute<
      cutlass::multiplies, ElementD, float,
      cutlass::FloatRoundStyle::round_to_nearest>;

 public:
  using EVTCompute =
      cutlass::epilogue::fusion::Sm90EVT<Compute1, ScaleA, EVTCompute0>;
  using ArgumentType = typename EVTCompute::Arguments;

  using ScaleAArray = typename SUPER::template ColOrScalarLoadArray<float>;
  using ScaleBArray = typename SUPER::template RowOrScalarLoadArray<float>;

  static ArgumentType prepare_args(float const* const* a_scales_ptr,
                                   float const* const* b_scales_ptr,
                                   bool a_col_broadcast, bool b_row_broadcast) {
    auto a_args = SUPER::template args_from_tensor<ScaleAArray, float>(
        a_scales_ptr, a_col_broadcast);
    auto b_args = SUPER::template args_from_tensor<ScaleBArray, float>(
        b_scales_ptr, b_row_broadcast);

    typename EVTCompute0::Arguments evt0_args{b_args, {}, {}};
    return ArgumentType{a_args, evt0_args, {}};
  }
};

};  // namespace vllm::c3x
