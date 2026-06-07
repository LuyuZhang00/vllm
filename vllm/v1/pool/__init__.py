# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
池化 (Pooling) 模块 (vllm/v1/pool/__init__.py)

本模块实现了 vLLM v1 引擎中的池化（pooling）功能，主要用于嵌入模型和重排序模型。

池化与生成 (generation) 的区别：
- 生成模型：自回归地逐个生成 token，输出是 token 序列
- 池化模型：将输入序列的所有 token 表示聚合为固定大小的向量，输出是嵌入向量

池化的典型应用场景：
1. 文本嵌入 (Text Embedding): 将文本转换为向量表示，用于语义搜索、相似度计算
2. 重排序 (Reranking): 计算查询和文档的相关性分数
3. 晚期交互 (Late Interaction): 如 ColBERT，计算 token 级别的相似度

模块组成：
- metadata.py: PoolingMetadata 和 PoolingCursor，存储池化相关的元数据
- late_interaction.py: 晚期交互模式的支持函数

池化流程：
1. 模型运行器执行前向传播，获取所有 token 的隐藏状态
2. 根据 PoolingMetadata 中的信息，提取每个序列的首/尾 token 隐藏状态
3. 对隐藏状态执行池化操作（如取平均、取首 token、取尾 token 等）
4. 返回池化后的向量作为结果
"""
