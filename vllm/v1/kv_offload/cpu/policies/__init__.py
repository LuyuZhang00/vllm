# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
缓存淘汰策略模块 (vllm/v1/kv_offload/cpu/policies/__init__.py)

本模块提供 CPU KV 缓存的淘汰策略实现。

淘汰策略决定了当 CPU 缓存空间不足时，哪些缓存块应该被淘汰以腾出空间。

支持的策略：
1. LRU (Least Recently Used): 最近最少使用策略
   - 淘汰最长时间未被访问的缓存块
   - 实现简单，适用于大多数场景
   - 实现在 lru.py 中

2. ARC (Adaptive Replacement Cache): 自适应替换缓存策略
   - 自动在"最近使用"和"最频繁使用"之间平衡
   - 对访问模式变化有更好的适应性
   - 实现在 arc.py 中

策略接口（CachePolicy 基类定义在 base.py 中）：
- get(key): 查找缓存块
- insert(key, block): 插入新缓存块
- remove(key): 移除缓存块
- touch(keys): 更新访问记录
- evict(count, protected): 淘汰指定数量的块
- clear(): 清除所有块
"""
