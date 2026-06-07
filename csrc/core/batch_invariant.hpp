#pragma once
#include <cstdlib>
#include <string>

namespace vllm {

// 中文注释：模块功能概述
// =========================================================================
// batch_invariant.hpp 提供批次不变性（Batch Invariance）检查功能
//
// 批次不变性是指：无论 batch size 如何变化，同一个请求的计算结果应该完全一致
// 这对于调试和结果复现非常重要
//
// 使用场景：
// - 调试：开启批次不变性模式，确保单请求和多请求 batch 结果一致
// - 测试：验证算子实现的正确性
// - 性能分析：在受控环境下比较不同 batch size 的性能
//
// 使用方法：
// - 设置环境变量 VLLM_BATCH_INVARIANT=1 开启
// - 不设置或设为0则关闭（默认）
// =========================================================================

// 中文注释：检查是否开启批次不变性模式
// 返回 true 表示开启，此时 CUDA kernel 可能会使用更保守的实现
// 以确保不同 batch size 下结果一致
//
// 实现特点：
// - 使用 static 局部变量缓存结果，避免重复读取环境变量
// - lambda 表达式在第一次调用时执行，后续调用直接返回缓存值
// - 这是线程安全的（C++11 保证 static 局部变量初始化是线程安全的）
inline bool vllm_is_batch_invariant() {
  static bool cached = []() {
    std::string env_key = "VLLM_BATCH_INVARIANT";
    const char* val = std::getenv(env_key.c_str());
    return (val && std::atoi(val) != 0) ? 1 : 0;
  }();
  return cached;
}

}  // namespace vllm
