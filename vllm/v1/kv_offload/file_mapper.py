# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# =============================================================================
# 文件概述：file_mapper.py —— KV block 哈希到文件路径的映射器
# =============================================================================
#
# 本模块的核心职责是：将每个 KV cache block 的唯一标识（OffloadKey，由
# block hash + group_idx 构成）映射到磁盘上的具体文件路径。
#
# 在 vLLM 的 KV cache offloading 体系中，当 GPU 显存不足时，会将不活跃的
# KV block 持久化到磁盘。FileMapper 负责确定"某个 block 应该存到哪个文件"。
#
# 目录结构设计（三级分层 + rank 隔离）：
#   <root_dir>/<model_sha256>/              ← 基础路径（由配置哈希决定）
#     └── _r<rank>/                         ← 每个 worker rank 独立目录
#         └── <hash前3位>/                  ← 第一级子目录（16^3 = 4096 个桶）
#             └── <hash第4-5位>_g<group>/   ← 第二级子目录 + group 索引
#                 └── <完整hash>.bin        ← 实际的 KV block 文件
#
# 这种分层设计有三个目的：
#   1. 避免单目录文件数过多导致文件系统性能下降（分散到 ~1M 个桶中）
#   2. rank 隔离确保不同并行 worker 的缓存互不干扰
#   3. group_idx 在目录名中体现，支持多 KV cache group 的混合模型
#
# config.json 位于 base_path 下（不含 rank 后缀），所有 rank 共享同一份配置，
# 因为 rank 信息在 _compute_base_path 的哈希计算中被排除。
# =============================================================================

import hashlib
import json

from vllm.v1.kv_offload.base import (
    OffloadingSpec,
    OffloadKey,
    get_offload_block_hash,
    get_offload_group_idx,
)

# 中文注释：base_path 哈希前缀的长度（十六进制字符数）。
# 使用 12 个十六进制字符 = 48 bit，碰撞概率极低（2^48 种可能），
# 同时足够短，不会使路径过长。
_BASE_PATH_HASH_LEN = 12

# 中文注释：共享配置文件的文件名。
# config.json 位于 base_path 根目录下，所有 rank 共享，
# 记录了模型名、block 大小、并行度等运行配置。
_CONFIG_FILENAME = "config.json"


class FileMapper:
    """
    FileMapper maps KV blocks (given by their hash) to file names.
    """

    # 中文注释：FileMapper 的核心作用是将 OffloadKey（block 的唯一标识）
    # 映射到磁盘上的文件路径。
    #
    # 映射规则：
    #   输入：OffloadKey（= block_hash bytes + group_idx bytes）
    #   输出：多级目录 + 文件名，如：
    #         /data/llama-7b_a1b2c3d4e5f6_r0/abc/de_g0/abcdef123456.bin
    #
    # 设计考量：
    #   - 每个 Worker 进程构造自己的 FileMapper 实例
    #   - config.json 通过 _compute_base_path 排除 rank 来实现跨 rank 共享
    #   - parallel_agnostic 模式下强制 rank=0、并行度=1，
    #     使得不同并行布局（如 TP=2 vs TP=4）可以共享同一缓存目录

    def __init__(
        self,
        root_dir: str,
        model_name: str,
        hash_block_size: int,
        gpu_blocks_per_file: int,
        tp_size: int,
        pp_size: int,
        pcp_size: int,
        dcp_size: int,
        rank: int,
        dtype: str,
        kv_cache_groups: list[dict] | None = None,
        inference_engine: str = "vllm",
        parallel_agnostic: bool = False,
    ):
        """
        Initialize the file mapper. Each worker constructs its own, but
        `config.json` is shared across workers since rank lives outside the hash.
        When `parallel_agnostic=True`, tp/pp/pcp/dcp are forced to 1 and rank
        to 0 so multiple parallelism layouts collapse into the same folder.
        """
        # 中文注释：parallel_agnostic 模式的核心逻辑——
        # 强制将所有并行维度设为 1、rank 设为 0，这样不同并行配置
        # （如 TP=2/PP=1 和 TP=4/PP=1）会生成相同的 base_path，
        # 从而共享磁盘上的 KV cache 缓存。
        if parallel_agnostic:
            tp_size = pp_size = pcp_size = dcp_size = 1
            rank = 0
        self.rank: int = rank

        # 中文注释：fields 字典包含所有影响 base_path 哈希计算的配置字段。
        # 这些字段会被序列化为 JSON 并计算 SHA-256 哈希，
        # 作为 base_path 的一部分，确保不同配置的缓存不会混淆。
        # 注意：rank 不包含在 fields 中，因此 config.json 对所有 rank 共享。
        self.fields: dict = {
            "model_name": model_name,
            "hash_block_size": hash_block_size,
            "gpu_blocks_per_file": gpu_blocks_per_file,
            "tp_size": tp_size,
            "pp_size": pp_size,
            "pcp_size": pcp_size,
            "dcp_size": dcp_size,
            "dtype": str(dtype),
            "kv_cache_groups": kv_cache_groups or [],
            "inference_engine": inference_engine,
        }
        self.base_path: str = self._compute_base_path(root_dir, self.fields)

    @classmethod
    def from_offloading_spec(
        cls,
        root_dir: str,
        offloading_spec: OffloadingSpec,
        gpu_blocks_per_file: int = 1,
        parallel_agnostic: bool = False,
    ) -> "FileMapper":
        """Build a FileMapper from an OffloadingSpec."""
        # 中文注释：工厂方法，从 OffloadingSpec 中提取所有需要的配置参数
        # 来构造 FileMapper。OffloadingSpec 是 vLLM offloading 的配置规范，
        # 包含了模型配置、并行配置、KV cache 配置等所有信息。
        vllm_config = offloading_spec.vllm_config
        kv_cache_config = offloading_spec.kv_cache_config

        # 中文注释：从并行配置中提取 tensor parallel、pipeline parallel 等参数。
        parallel_config = vllm_config.parallel_config
        # 中文注释：获取 KV cache 的数据类型（如 float16、bfloat16），
        # 并去掉 "torch." 前缀以便序列化到 JSON。
        dtype = str(vllm_config.cache_config.cache_dtype).replace("torch.", "")

        # 中文注释：构建 kv_cache_groups 的元数据列表。
        # 每个 KV cache group 可能有不同的 block_size 和 layer_names，
        # 例如在混合模型（如 Mamba+Attention）中不同层使用不同的缓存规格。
        kv_cache_groups = [
            {
                "block_size": group.kv_cache_spec.block_size,
                "layer_names": list(group.layer_names),
            }
            for group in kv_cache_config.kv_cache_groups
        ]
        return cls(
            root_dir=root_dir,
            model_name=vllm_config.model_config.model,
            hash_block_size=vllm_config.cache_config.block_size,
            gpu_blocks_per_file=gpu_blocks_per_file,
            tp_size=parallel_config.tensor_parallel_size,
            pp_size=parallel_config.pipeline_parallel_size,
            pcp_size=parallel_config.prefill_context_parallel_size,
            dcp_size=parallel_config.decode_context_parallel_size,
            rank=parallel_config.rank,
            dtype=dtype,
            kv_cache_groups=kv_cache_groups,
            parallel_agnostic=parallel_agnostic,
        )

    def get_file_name(self, key: OffloadKey) -> str:
        """Map an OffloadKey to <base>_r<rank>/<hhh>/<hh>_g<group_idx>/<hash>.bin."""
        # 中文注释：核心映射方法——将 OffloadKey 转换为磁盘文件路径。
        #
        # 映射步骤：
        #   1. 从 OffloadKey 中提取 block hash（字节）并转为十六进制字符串
        #   2. 从 OffloadKey 中提取 group_idx（KV cache 组索引）
        #   3. 将 hash 的前 3 位作为第一级子目录（约 4096 个桶）
        #   4. 将 hash 的第 4-5 位作为第二级子目录（约 256 个桶），附加 group 索引
        #   5. 完整 hash 作为文件名
        #
        # 最终路径示例：
        #   /data/llama-7b_a1b2c3d4e5f6_r0/abc/de_g0/abcdef1234567890abcdef.bin
        #   └──────────── base_path ─────────┘     └──── hash_hex ────┘
        hash_hex = get_offload_block_hash(key).hex()
        group_idx = get_offload_group_idx(key)
        subfolder1, subfolder2 = hash_hex[:3], hash_hex[3:5]
        return (
            f"{self.base_path}_r{self.rank}"
            f"/{subfolder1}/{subfolder2}_g{group_idx}/{hash_hex}.bin"
        )

    def get_run_config(self) -> dict:
        return dict(self.fields)

    def get_config_file_path(self) -> str:
        # 中文注释：返回共享配置文件路径。
        # config.json 位于 base_path（不含 _r<rank> 后缀）下，
        # 因此所有 rank 的 worker 共享同一份配置文件。
        return f"{self.base_path}/{_CONFIG_FILENAME}"

    @staticmethod
    def _compute_base_path(root_dir: str, fields: dict) -> str:
        """
        Layout: <root_dir>/<safe_model_name>_<sha256-prefix>/.
        safe_model_name replaces '/' with '_' so HuggingFace IDs don't nest.
        """
        # 中文注释：计算基础路径的核心算法。
        #
        # 步骤：
        #   1. 将 fields 字典序列化为紧凑的 JSON 字符串（key 排序保证确定性）
        #   2. 对 JSON 字符串计算 SHA-256 哈希
        #   3. 取哈希值的前 12 个十六进制字符（48 bit）作为唯一标识
        #   4. 将模型名中的 '/' 替换为 '_'（适配 HuggingFace 的 org/model 命名格式）
        #   5. 拼接：<root_dir>/<safe_model_name>_<hash_prefix>
        #
        # 这样设计保证：
        #   - 相同配置生成相同路径（缓存可复用）
        #   - 不同配置生成不同路径（缓存不混淆）
        #   - rank 不参与哈希计算，因此 config.json 对所有 rank 一致
        canonical = json.dumps(fields, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[
            :_BASE_PATH_HASH_LEN
        ]
        safe_model_name = fields["model_name"].replace("/", "_")
        return f"{root_dir}/{safe_model_name}_{digest}"
