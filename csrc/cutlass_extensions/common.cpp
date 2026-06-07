#include "cutlass_extensions/common.hpp"

// 中文注释：获取当前 GPU 的 SM（Streaming Multiprocessor）计算能力版本号。
// 返回格式为 major * 10 + minor，例如：
//   - SM 8.0 (A100) 返回 80
//   - SM 9.0 (H100) 返回 90
// 该函数用于在运行时判断 GPU 架构，以选择合适的 kernel 或代码路径。
int32_t get_sm_version_num() {
  int32_t major_capability, minor_capability;
  cudaDeviceGetAttribute(&major_capability, cudaDevAttrComputeCapabilityMajor,
                         0);
  cudaDeviceGetAttribute(&minor_capability, cudaDevAttrComputeCapabilityMinor,
                         0);
  int32_t version_num = major_capability * 10 + minor_capability;
  return version_num;
}