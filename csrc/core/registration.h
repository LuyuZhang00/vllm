#pragma once

#include <Python.h>

// 中文注释：模块功能概述
// =========================================================================
// registration.h 提供 C++ 扩展库的注册宏
//
// 这些宏用于：
// 1. PyTorch 自定义算子注册：将 C++ 函数注册为 PyTorch 算子
// 2. Python 模块初始化：创建 Python 可导入的共享库
// 3. 宏展开辅助：处理宏嵌套和字符串化
//
// 使用场景：
// - 注册自定义 CUDA kernel 为 PyTorch 算子
// - 创建可被 Python import 的 C++ 扩展模块
// - vLLM 的 C++ 核心功能（如 PagedAttention、量化算子等）
// =========================================================================

// 中文注释：宏展开辅助宏
// _CONCAT/CONCAT：连接两个 token（处理宏嵌套）
// _STRINGIFY/STRINGIFY：将 token 转换为字符串（处理宏嵌套）
//
// 为什么需要两层宏？
// 因为 C 预处理器在展开宏时，不会对宏参数进行完全展开
// 使用两层宏可以确保参数先被完全展开，再进行连接或字符串化
#define _CONCAT(A, B) A##B
#define CONCAT(A, B) _CONCAT(A, B)

#define _STRINGIFY(A) #A
#define STRINGIFY(A) _STRINGIFY(A)

// 中文注释：PyTorch 库注册宏的扩展版本
// 这些宏允许 NAME 参数本身是一个宏（而不是字面量 token）
//
// 为什么需要这个？
// 因为标准的 TORCH_LIBRARY 宏不支持 NAME 是宏的情况
// 通过添加一层展开，可以解决这个问题
//
// 使用示例：
//   #define MY_OP_NAME my_namespace_my_op
//   TORCH_LIBRARY_EXPAND(my_namespace, m) {
//     m.def("my_op", &my_op_impl);
//   }
#define TORCH_LIBRARY_EXPAND(NAME, MODULE) TORCH_LIBRARY(NAME, MODULE)

// A version of the TORCH_LIBRARY_IMPL macro that expands the NAME, i.e. so NAME
// could be a macro instead of a literal token.
#define TORCH_LIBRARY_IMPL_EXPAND(NAME, DEVICE, MODULE) \
  TORCH_LIBRARY_IMPL(NAME, DEVICE, MODULE)

// 中文注释：Python 模块注册宏
// 创建一个可被 Python import 的 C++ 扩展模块
//
// 工作原理：
// 1. 定义 PyInit_<NAME> 函数（Python 解释器调用的入口点）
// 2. 创建 PyModuleDef 结构体（模块元数据）
// 3. 调用 PyModule_Create 创建模块对象
//
// 使用示例：
//   // 在 my_extension.cpp 中
//   REGISTER_EXTENSION(_C)
//
//   // Python 端
//   import my_extension._C  # 触发 PyInit__C
//
// 注意：这个宏通常与 TORCH_LIBRARY 一起使用
// TORCH_LIBRARY 注册算子，REGISTER_EXTENSION 创建 Python 模块入口
#define REGISTER_EXTENSION(NAME)                                               \
  PyMODINIT_FUNC CONCAT(PyInit_, NAME)() {                                     \
    static struct PyModuleDef module = {PyModuleDef_HEAD_INIT,                 \
                                        STRINGIFY(NAME), nullptr, 0, nullptr}; \
    return PyModule_Create(&module);                                           \
  }
