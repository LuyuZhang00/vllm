# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# =============================================================================
# vLLM 网络设备管理模块 -- GPU 到 RDMA 网卡的 PCIe 映射
# =============================================================================
# 本模块实现 GPU 到网卡（NIC）的 PCIe 地址映射，用于 RDMA 远程直接内存访问
# 传输（如 UCX、NVSHMEM 等高性能通信协议）。
#
# 核心功能：
#   1. 解析和规范化 PCI BDF（Bus:Device.Function）地址
#   2. 维护 GPU PCIe 地址到 NIC PCIe 地址的映射关系
#   3. 通过 Linux sysfs 文件系统查找 RDMA 设备名称
#   4. 为每个 worker 进程设置对应的网络设备环境变量
#
# 使用场景：
#   - 单进程执行器（UniProcExecutor，TP=1）：直接调用 set_worker_net_device
#   - 多进程执行器（MultiprocExecutor，TP>1）：每个 worker 子进程调用
#     set_worker_net_device 设置自己的网卡映射
#
# 所需的环境变量（必须同时设置）：
#   - VLLM_GPU_NIC_PCIE_MAPPING：
#     逗号分隔的 GPU_BDF=NIC_BDF 对，定义 GPU 到网卡的 PCIe 地址映射
#     示例：0000:01:00.0=0000:02:00.0,0000:03:00.0=0000:04:00.0
#
#   - VLLM_NIC_SELECTION_VARS：
#     逗号分隔的环境变量列表，表示需要设置的网卡选择变量
#     每个条目可以附加后缀（用冒号分隔），后缀会追加到 RDMA 设备名后
#     示例：UCX_NET_DEVICES:1,NCCL_IB_HCA:1
#     这将设置 UCX_NET_DEVICES=mlx5_0:1 和 NCCL_IB_HCA=mlx5_0:1
#
# 工作流程：
#   1. 解析环境变量获取 GPU-NIC 映射关系
#   2. 根据 worker 的 local_rank 找到对应的 GPU PCIe 地址
#   3. 通过映射表找到对应的 NIC PCIe 地址
#   4. 通过 sysfs 查找 NIC 对应的 RDMA 设备名称（如 mlx5_0）
#   5. 设置相应的环境变量，使通信库能找到正确的网卡
# =============================================================================

import os
from pathlib import Path

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)


def normalize_pci(addr: str) -> tuple[int, int, int, int]:
    """Parse PCI BDF/domain-bus-device-function into comparable ints (all hex).

    Supported shapes:
    - ``domain:bus:dev.fn`` -- domain width varies (e.g. ``00000001:00:00.0``,
      ``0001:00:00.0``, ``0000:3f:00.0``).
    - ``bus:dev.fn`` -- domain **0** (e.g. ``01:00.0``, ``40:00.0``).

    Function suffix is hex (typically ``0``--``7``). Raises ``ValueError`` if malformed.
    """
    # ------------------------------------------------------------------
    # 第一步：预处理输入字符串
    # ------------------------------------------------------------------
    # 去除首尾空白，转为小写，去除空格
    # 确保输入格式统一，方便后续解析
    s = addr.strip().lower().replace(" ", "")

    # 去除可能存在的 "0x" 前缀（某些系统输出带此前缀）
    if s.startswith("0x"):
        s = s[2:]

    # ------------------------------------------------------------------
    # 第二步：解析 function（功能号）部分
    # ------------------------------------------------------------------
    # PCI BDF 格式中，function 以 "." 分隔，位于最后
    # 例如 "0001:00:00.0" 中的 "0" 就是 function 号
    if "." not in s:
        raise ValueError(f"invalid PCI BDF (missing function suffix): {addr!r}")
    body, fn_s = s.rsplit(".", 1)
    if not fn_s or any(c not in "0123456789abcdef" for c in fn_s):
        raise ValueError(f"invalid PCI function in BDF: {addr!r}")
    fn = int(fn_s, 16)
    if fn > 0xFF:
        raise ValueError(f"PCI function out of range: {addr!r}")

    # ------------------------------------------------------------------
    # 第三步：解析 domain（域）、bus（总线）、device（设备）部分
    # ------------------------------------------------------------------
    # 支持两种格式：
    #   - 三段式：domain:bus:device（如 0001:00:00.0）
    #   - 两段式：bus:device（省略 domain，默认为 0，如 01:00.0）
    parts = body.split(":")
    if len(parts) == 2:
        domain = 0
        bus = int(parts[0], 16)
        device = int(parts[1], 16)
    elif len(parts) == 3:
        domain = int(parts[0], 16)
        bus = int(parts[1], 16)
        device = int(parts[2], 16)
    else:
        raise ValueError(
            f"invalid PCI BDF (want domain:bus:dev.fn or bus:dev.fn): {addr!r}"
        )

    # ------------------------------------------------------------------
    # 第四步：校验数值范围
    # ------------------------------------------------------------------
    # PCI 规范中 bus 最大 255（0xFF），device 最大 31（0x1F）
    if bus > 0xFF or device > 0x1F:
        raise ValueError(f"PCI bus or device out of range: {addr!r}")

    # 返回四元组 (domain, bus, device, function)，均为整数
    return (domain, bus, device, fn)


def parse_gpu_nic_mapping(
    raw: str,
) -> dict[tuple[int, int, int, int], tuple[int, int, int, int]]:
    # ------------------------------------------------------------------
    # 解析 VLLM_GPU_NIC_PCIE_MAPPING 环境变量
    # ------------------------------------------------------------------
    # 输入格式：逗号分隔的 "GPU_BDF=NIC_BDF" 对
    # 示例："0000:01:00.0=0000:02:00.0,0000:03:00.0=0000:04:00.0"
    #
    # 返回值：字典，键为 GPU 的 PCI 地址四元组，值为对应 NIC 的 PCI 地址四元组
    out: dict[tuple[int, int, int, int], tuple[int, int, int, int]] = {}
    for segment in raw.split(","):
        segment = segment.strip()
        if not segment:
            continue
        if "=" not in segment:
            raise ValueError(
                "VLLM_GPU_NIC_PCIE_MAPPING: expected comma-separated"
                f" gpu_bdf=nic_bdf pairs; ambiguous segment: {segment!r}"
            )
        # 以 "=" 分隔得到 GPU 和 NIC 的 PCIe 地址字符串
        gpu_s, nic_s = segment.split("=", 1)
        # 将地址字符串规范化为 (domain, bus, device, fn) 四元组
        gpu_key = normalize_pci(gpu_s.strip())
        nic_val = normalize_pci(nic_s.strip())
        out[gpu_key] = nic_val
    return out


def rdma_name_for_nic_pci(nic_pci: tuple[int, int, int, int]) -> str:
    """Map NIC PCI BDF to sysfs RDMA name (mlx5_*, ibp*, ...).

    Under ``/sys/class/infiniband/<name>/``, ``device`` is a **symlink** to the PCI
    device directory (e.g. ``.../0101:00:00.0``). We take ``Path(...).resolve().name``
    as the BDF string.

    ``VLLM_GPU_NIC_PCIE_MAPPING`` NIC keys must **normalize** (via ``normalize_pci``)
    to the same tuple as this basename.
    """
    # ------------------------------------------------------------------
    # 通过 Linux sysfs 文件系统查找 RDMA 设备名称
    # ------------------------------------------------------------------
    # Linux InfiniBand/RDMA 子系统在 /sys/class/infiniband/ 下注册设备
    # 每个设备目录（如 /sys/class/infiniband/mlx5_0/）包含一个 "device" 符号链接
    # 该符号链接指向 PCI 设备目录（如 /sys/bus/pci/devices/0000:01:00.0/）
    #
    # 查找策略：
    #   1. 遍历 /sys/class/infiniband/ 下的所有设备目录
    #   2. 解析每个设备的 "device" 符号链接获取 PCI 地址
    #   3. 将解析到的 PCI 地址与目标 NIC 的 PCI 地址匹配
    #   4. 返回匹配的 RDMA 设备名称（如 mlx5_0、ibp1s0 等）
    ib = Path("/sys/class/infiniband")
    if not ib.is_dir():
        raise RuntimeError("/sys/class/infiniband not found or not a directory")
    names = sorted(p.name for p in ib.iterdir() if p.is_dir())
    for name in names:
        dev_link = ib / name / "device"
        if not dev_link.exists():
            continue
        try:
            # 解析符号链接，获取实际的 PCI 设备目录路径
            resolved = dev_link.resolve()
        except OSError:
            continue
        # 提取路径最后一段作为 PCI BDF 字符串
        # 例如路径 /sys/bus/pci/devices/0000:01:00.0 -> 取 "0000:01:00.0"
        pci_name = resolved.name
        try:
            if normalize_pci(pci_name) == nic_pci:
                return name
        except ValueError:
            continue
    raise RuntimeError(
        f"No /sys/class/infiniband device for NIC PCI {nic_pci}; have entries: {names}"
    )


def parse_nic_selection_vars(raw: str) -> list[tuple[str, str]]:
    """Parse ``VLLM_NIC_SELECTION_VARS`` into ``(env_var_name, suffix)`` pairs.

    Each entry is ``VAR_NAME`` or ``VAR_NAME:<suffix>``.  The colon and
    everything after it is appended verbatim to the RDMA device name.
    """
    # ------------------------------------------------------------------
    # 解析 VLLM_NIC_SELECTION_VARS 环境变量
    # ------------------------------------------------------------------
    # 输入格式：逗号分隔的变量名列表，每个变量名可选带冒号后缀
    # 示例："UCX_NET_DEVICES:1,NCCL_IB_HCA:1"
    #
    # 解析结果：
    #   [("UCX_NET_DEVICES", ":1"), ("NCCL_IB_HCA", ":1")]
    #
    # 后缀的含义：
    #   某些 RDMA 库（如 UCX）支持在设备名后附加参数
    #   例如 "mlx5_0:1" 表示使用 mlx5_0 设备的第 1 个端口
    result: list[tuple[str, str]] = []
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        if ":" in entry:
            var_name, suffix = entry.split(":", 1)
            result.append((var_name, ":" + suffix))
        else:
            result.append((entry, ""))
    return result


def set_worker_gpu_nic_mapping(local_rank: int) -> None:
    """Set NIC selection env vars from VLLM_GPU_NIC_PCIE_MAPPING for a worker.

    Which env vars are set is controlled by ``VLLM_NIC_SELECTION_VARS``.
    """
    # ------------------------------------------------------------------
    # 主逻辑：为指定 worker 设置网卡选择环境变量
    # ------------------------------------------------------------------
    # 整体流程：
    #   1. 读取并解析 GPU-NIC PCIe 映射表
    #   2. 根据 worker 的 local_rank 找到对应的 GPU PCIe 地址
    #   3. 在映射表中查找该 GPU 对应的 NIC PCIe 地址
    #   4. 通过 sysfs 将 NIC PCIe 地址转换为 RDMA 设备名称
    #   5. 为每个需要设置的环境变量赋值

    raw = envs.VLLM_GPU_NIC_PCIE_MAPPING.strip()
    if not raw:
        return
    selection_raw = envs.VLLM_NIC_SELECTION_VARS.strip()
    selection_vars = parse_nic_selection_vars(selection_raw)
    mapping = parse_gpu_nic_mapping(raw)

    # 获取所有 GPU 的 PCI 总线地址，按物理设备索引排列
    pci_by_index = current_platform.get_all_gpu_pci_bus_ids()

    # ------------------------------------------------------------------
    # 将 CUDA 相对 local_rank 转换为物理设备索引
    # ------------------------------------------------------------------
    # 当使用 CUDA_VISIBLE_DEVICES 限制可见 GPU 时（如 DP 分片场景），
    # CUDA 的 local_rank 是相对于可见设备的编号，而我们需要的是
    # 在所有 GPU 中的物理设备索引，才能正确查找 PCIe 地址
    physical_id = current_platform.device_id_to_physical_device_id(local_rank)
    if physical_id not in pci_by_index:
        raise RuntimeError(
            f"No GPU PCI for physical device index {physical_id} "
            f"(local_rank={local_rank}) in map "
            f"(have indices {sorted(pci_by_index.keys())})"
        )

    # 获取当前 worker 对应 GPU 的 BDF（Bus:Device.Function）地址
    gpu_bdf = pci_by_index[physical_id]
    gpu_key = normalize_pci(gpu_bdf)

    # 在映射表中查找该 GPU 对应的 NIC
    if gpu_key not in mapping:
        keys_fmt = ", ".join(
            f"{d:04x}:{b:02x}:{dev:02x}.{fn}"
            for d, b, dev, fn in sorted(mapping.keys())
        )
        raise RuntimeError(
            f"No VLLM_GPU_NIC_PCIE_MAPPING entry for GPU PCI {gpu_bdf} "
            f"(worker local_rank={local_rank}); mapped GPUs: {keys_fmt}"
        )

    # 获取 NIC 的 PCI 地址，并转换为 RDMA 设备名称（如 mlx5_0）
    nic_pci = mapping[gpu_key]
    rdma_dev = rdma_name_for_nic_pci(nic_pci)

    # ------------------------------------------------------------------
    # 设置环境变量
    # ------------------------------------------------------------------
    # 遍历所有需要设置的环境变量（如 UCX_NET_DEVICES、NCCL_IB_HCA 等）
    # 将 RDMA 设备名称（可能带后缀）设置为对应环境变量的值
    # 如果该环境变量已存在，将新值追加到已有值前面（用逗号分隔）
    set_vars: list[str] = []
    for var_name, suffix in selection_vars:
        value = f"{rdma_dev}{suffix}"
        existing = os.environ.get(var_name, "").strip()
        if existing:
            value = f"{value},{existing}"
        os.environ[var_name] = value
        set_vars.append(f"{var_name}={value}")

    # ------------------------------------------------------------------
    # 记录日志
    # ------------------------------------------------------------------
    # 输出映射结果，方便调试和运维排查
    # 格式：GPU rank X (PCIe addr) -> NIC RDMA名 (PCIe addr)，设置的环境变量
    nic_fmt = f"{nic_pci[0]:04x}:{nic_pci[1]:02x}:{nic_pci[2]:02x}.{nic_pci[3]}"
    logger.info(
        "GPU rank %s (PCIe addr %s) mapped to NIC %s (PCIe addr %s) via env vars: %s",
        local_rank,
        gpu_bdf,
        rdma_dev,
        nic_fmt,
        ", ".join(set_vars),
    )


def _dp_adjusted_local_rank(tp_local_rank: int, vllm_config: VllmConfig) -> int:
    """Compute the node-wide GPU index accounting for data parallelism.

    On CUDA-alike platforms without env-var device isolation (the common
    MP-backend path), the worker sees *all* GPUs on the node and selects
    its device via ``torch.accelerator.set_device_index()`` using::

        dp_local_rank * tp_pp_world_size + tp_local_rank

    This mirrors the adjustment in ``Worker.init_device()`` so we resolve
    the correct GPU PCI address *before* the CUDA device is initialised.
    """
    # ------------------------------------------------------------------
    # 计算考虑数据并行（DP）后的节点级 GPU 索引
    # ------------------------------------------------------------------
    # 背景说明：
    #   在 vLLM 中，当同时使用数据并行（DP）和张量并行（TP）时，
    #   每个节点上的 GPU 按如下方式编号：
    #     DP_rank_0: GPU_0, GPU_1, ..., GPU_{TP-1}
    #     DP_rank_1: GPU_{TP}, GPU_{TP+1}, ..., GPU_{2*TP-1}
    #     ...
    #
    #   因此，对于 DP 内的 TP local_rank，其节点级 GPU 索引为：
    #     dp_local_rank * (tp_size * pp_size) + tp_local_rank
    #
    #   例如，TP=2, PP=1, DP=2 的 4 GPU 场景：
    #     DP0-TP0 -> GPU_0
    #     DP0-TP1 -> GPU_1
    #     DP1-TP0 -> GPU_2
    #     DP1-TP1 -> GPU_3
    #
    # 条件说明：
    #   此调整仅在以下情况下需要：
    #   - 不使用 Ray 或外部启动器作为分布式后端（即使用本地多进程）
    #   - 数据并行后端不是 Ray
    #   - 数据并行仅限单节点（nnodes_within_dp == 1）
    #   在这些条件下，worker 可以看到节点上的所有 GPU，
    #   需要通过计算得到正确的物理 GPU 索引
    pc = vllm_config.parallel_config
    if (
        pc.distributed_executor_backend not in ("ray", "external_launcher")
        and pc.data_parallel_backend != "ray"
        and pc.nnodes_within_dp == 1
    ):
        dp_local_rank = pc.data_parallel_rank_local
        if dp_local_rank is None:
            dp_local_rank = pc.data_parallel_index
        tp_pp_world_size = pc.pipeline_parallel_size * pc.tensor_parallel_size
        return dp_local_rank * tp_pp_world_size + tp_local_rank
    # 在 Ray 后端或其他场景下，worker 只能看到分配给自己的 GPU，
    # local_rank 直接对应物理设备，无需额外调整
    return tp_local_rank


def set_worker_net_device(local_rank: int, vllm_config: VllmConfig) -> None:
    """Top-level entry point for both UniProcExecutor and MultiprocExecutor.

    Sets NIC selection env vars from ``VLLM_GPU_NIC_PCIE_MAPPING`` and
    ``VLLM_NIC_SELECTION_VARS`` if present; no-op otherwise.
    """
    # ------------------------------------------------------------------
    # 顶层入口函数：设置 worker 的网络设备环境变量
    # ------------------------------------------------------------------
    # 本函数是 UniProcExecutor 和 MultiprocExecutor 的统一入口
    # 调用者只需传入 worker 的 local_rank 和 vllm_config 即可
    #
    # 处理逻辑：
    #   1. 检查 VLLM_GPU_NIC_PCIE_MAPPING 和 VLLM_NIC_SELECTION_VARS
    #      是否都已设置（两者必须同时存在或同时不存在）
    #   2. 如果两个环境变量都未设置，直接返回（无操作）
    #   3. 计算考虑数据并行后的实际 GPU 索引
    #   4. 调用 set_worker_gpu_nic_mapping 设置网卡映射

    has_pcie_mapping = bool(envs.VLLM_GPU_NIC_PCIE_MAPPING.strip())
    has_selection_vars = bool(envs.VLLM_NIC_SELECTION_VARS.strip())

    # ------------------------------------------------------------------
    # 校验：两个环境变量必须同时设置或同时不设置
    # ------------------------------------------------------------------
    # 如果只设置了一个而未设置另一个，说明配置不完整，应报错提示用户
    if has_pcie_mapping and not has_selection_vars:
        raise RuntimeError(
            "VLLM_GPU_NIC_PCIE_MAPPING is set but VLLM_NIC_SELECTION_VARS "
            "is not; both must be set together."
        )
    if has_selection_vars and not has_pcie_mapping:
        raise RuntimeError(
            "VLLM_NIC_SELECTION_VARS is set but VLLM_GPU_NIC_PCIE_MAPPING "
            "is not; both must be set together."
        )

    # 如果两个环境变量都未设置，则无需进行网卡映射，直接返回
    if not has_pcie_mapping and not has_selection_vars:
        return

    # ------------------------------------------------------------------
    # 计算调整后的 local_rank 并设置网卡映射
    # ------------------------------------------------------------------
    # 在多 DP 场景下，需要将 TP local_rank 转换为节点级 GPU 索引
    # 然后调用 set_worker_gpu_nic_mapping 完成实际的环境变量设置
    adjusted_rank = _dp_adjusted_local_rank(local_rank, vllm_config)
    set_worker_gpu_nic_mapping(adjusted_rank)
