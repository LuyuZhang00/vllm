# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import copy
import multiprocessing
import multiprocessing.connection
import time
import weakref

import msgspec.msgpack
import zmq

from vllm.config import ParallelConfig
from vllm.logger import init_logger
from vllm.utils.network_utils import make_zmq_socket
from vllm.utils.system_utils import get_mp_context, set_process_title
from vllm.v1.engine import EngineCoreOutputs, EngineCoreRequestType
from vllm.v1.serial_utils import MsgpackDecoder
from vllm.v1.utils import get_engine_client_zmq_addr, shutdown

logger = init_logger(__name__)


class DPCoordinator:
    """
    数据并行协调器 —— 用于 DP>1 的部署场景。

    架构:
    ┌─────────────────────────────────────────────────────────────────┐
    │                    DPCoordinator 进程                           │
    │                                                                 │
    │  ┌───────────────┐    ┌───────────────┐    ┌───────────────┐  │
    │  │  publish_front │    │  output_back  │    │  publish_back │  │
    │  │  (XPUB)        │    │  (PULL)       │    │  (XPUB)       │  │
    │  │  → API Server  │    │  ← EngineCore │    │  → EngineCore │  │
    │  └───────────────┘    └───────────────┘    └───────────────┘  │
    └─────────────────────────────────────────────────────────────────┘

    职责:
    ① 收集每个 DP 引擎的负载统计 (waiting/running 队列长度)
       发布给所有前端 API Server，用于负载均衡决策

    ② 追踪 DP "请求 wave" 编号和引擎运行状态
       引擎在全局 running/paused 状态之间交替
       wave 编号 = 引擎从 running → paused 的次数

    ③ 广播 START_DP_WAVE 消息，唤醒暂停的引擎
       触发条件:
       - 前端发送新请求时引擎处于暂停状态
       - 引擎收到过期 wave 的请求

    Wave 机制:
    ┌─────────────────────────────────────────────────────────────┐
    │  1. 所有引擎空闲 → PAUSED 状态                              │
    │  2. 新请求到达 → 前端通知协调器                               │
    │  3. 协调器广播 START_DP_WAVE → 所有引擎唤醒                  │
    │  4. 引擎处理请求                                             │
    │  5. 所有引擎空闲 → 再次 PAUSED (wave++)                     │
    └─────────────────────────────────────────────────────────────┘
    """

    def _wait_for_zmq_addrs(self, zmq_addr_pipe) -> tuple[str, str, str]:
        try:
            timeout = 120
            ready = multiprocessing.connection.wait(
                [zmq_addr_pipe, self.proc.sentinel], timeout=timeout
            )
            if not ready:
                raise RuntimeError(
                    "DP Coordinator process failed to report ZMQ addresses "
                    f"within timeout={timeout} seconds during startup."
                )
            try:
                return zmq_addr_pipe.recv()
            except EOFError:
                raise RuntimeError(
                    "DP Coordinator process failed during startup."
                ) from None
        finally:
            zmq_addr_pipe.close()

    def __init__(
        self, parallel_config: ParallelConfig, enable_wave_coordination: bool = True
    ):
        """
        初始化 DPCoordinator。

        流程:
        ① 验证 DP 配置 (dp_size > 1)
        ② 分配 ZMQ 地址 (前端/后端)
        ③ 启动协调器子进程
        ④ 等待子进程报告 ZMQ 地址
        """
        dp_size = parallel_config.data_parallel_size
        assert dp_size > 1, "Coordinator only used for data parallel"

        host = parallel_config.data_parallel_master_ip

        # 假设协调器与前端进程同节点 (除非是外部或混合 LB 模式)
        local_only = not parallel_config.local_engines_only
        local_only_eng = dp_size == parallel_config.data_parallel_size_local
        # 处理从节点内扩展到节点间的场景
        if parallel_config.enable_elastic_ep:
            local_only_eng = False

        # 分配 ZMQ 地址
        # front_publish_address: 协调器 → API Server (统计和 wave 状态)
        # back_publish_address: 协调器 → EngineCore (wave 命令)
        # back_output_address: EngineCore → 协调器 (统计和 wave 通知)
        front_publish_address = get_engine_client_zmq_addr(local_only, host=host)
        back_publish_address = get_engine_client_zmq_addr(local_only_eng, host=host)
        back_output_address = get_engine_client_zmq_addr(local_only_eng, host=host)

        context = get_mp_context()
        parent_zmq_addr_pipe, child_zmq_addr_pipe = context.Pipe(duplex=False)
        self.proc: multiprocessing.Process = context.Process(
            target=DPCoordinatorProc.run_coordinator,
            name="VLLM_DP_Coordinator",
            kwargs={
                "engine_count": parallel_config.data_parallel_size,
                "front_publish_address": front_publish_address,
                "back_output_address": back_output_address,
                "back_publish_address": back_publish_address,
                "zmq_addr_pipe": child_zmq_addr_pipe,
                "enable_wave_coordination": enable_wave_coordination,
            },
            daemon=True,
        )
        self.proc.start()
        child_zmq_addr_pipe.close()
        (
            front_publish_address,
            back_output_address,
            back_publish_address,
        ) = self._wait_for_zmq_addrs(parent_zmq_addr_pipe)

        self.stats_publish_address = front_publish_address
        self.coord_in_address = back_publish_address
        self.coord_out_address = back_output_address
        self._finalizer = weakref.finalize(self, shutdown, [self.proc])

    def get_stats_publish_address(self) -> str:
        return self.stats_publish_address

    def get_engine_socket_addresses(self) -> tuple[str, str]:
        """Returns tuple of ZMQ input address, output address."""
        return self.coord_in_address, self.coord_out_address

    def shutdown(self, timeout: float | None = None) -> None:
        """Shutdown coordinator process with configurable timeout."""
        if self._finalizer.detach() is not None:
            shutdown([self.proc], timeout=timeout)


class EngineState:
    def __init__(self):
        self.request_counts = [0, 0]  # [waiting, running]


class DPCoordinatorProc:
    def __init__(
        self,
        engine_count: int,
        min_stats_update_interval_ms: int = 100,
        enable_wave_coordination: bool = True,
    ):
        set_process_title("DPCoordinator")
        self.ctx = zmq.Context()

        self.engines = [EngineState() for _ in range(engine_count)]

        self.stats_update_interval_ms = min_stats_update_interval_ms
        self.enable_wave_coordination = enable_wave_coordination

    @staticmethod
    def run_coordinator(
        engine_count: int,
        front_publish_address: str,
        back_output_address: str,
        back_publish_address: str,
        zmq_addr_pipe=None,
        min_stats_update_interval_ms: int = 100,
        enable_wave_coordination: bool = True,
    ):
        coordinator = DPCoordinatorProc(
            engine_count=engine_count,
            min_stats_update_interval_ms=min_stats_update_interval_ms,
            enable_wave_coordination=enable_wave_coordination,
        )
        try:
            coordinator.process_input_socket(
                front_publish_address,
                back_output_address,
                back_publish_address,
                zmq_addr_pipe,
            )
        except KeyboardInterrupt:
            logger.info("DP Coordinator process exiting")
        finally:
            if zmq_addr_pipe is not None:
                zmq_addr_pipe.close()

    def process_input_socket(
        self,
        front_publish_address: str,
        back_output_address: str,
        back_publish_address: str,
        zmq_addr_pipe=None,
    ):
        """
        协调器主循环 —— 处理所有 ZMQ 消息。

        三个 ZMQ 套接字:
        ┌─────────────────────────────────────────────────────────────┐
        │  publish_front (XPUB):                                      │
        │    → 发送给 API Server                                      │
        │    内容: (engine_req_counts, current_wave, engines_running)  │
        │                                                             │
        │  output_back (PULL):                                        │
        │    ← 接收来自 EngineCore 的消息                              │
        │    内容: EngineCoreOutputs (统计 + wave 通知)                │
        │                                                             │
        │  publish_back (XPUB):                                       │
        │    → 发送给 EngineCore                                      │
        │    内容: START_DP_WAVE (wave, exclude_engine_index)          │
        └─────────────────────────────────────────────────────────────┘

        主循环逻辑:
        ① 轮询三个套接字 (zmq.Poller)
        ② 超时时发布统计给前端
        ③ 处理引擎订阅消息
        ④ 处理前端的新请求通知 (唤醒引擎)
        ⑤ 处理引擎的统计更新和 wave 通知
        """
        decoder = MsgpackDecoder(EngineCoreOutputs)

        # Wave 追踪状态
        current_wave = 0          # 当前 wave 编号
        engines_running = False   # 引擎是否处于运行状态

        # 统计追踪状态
        stats_changed = False
        last_stats_step = -1
        last_stats_wave = -1
        last_step_counts: list[list[int]] | None = None

        with (
            make_zmq_socket(
                path=front_publish_address,  # IPC
                ctx=self.ctx,
                socket_type=zmq.XPUB,
                bind=True,
            ) as publish_front,
            make_zmq_socket(
                path=back_output_address,  # IPC 或 TCP
                ctx=self.ctx,
                socket_type=zmq.PULL,
                bind=True,
            ) as output_back,
            make_zmq_socket(
                path=back_publish_address,  # IPC 或 TCP
                ctx=self.ctx,
                socket_type=zmq.XPUB,
                bind=True,
            ) as publish_back,
        ):
            if zmq_addr_pipe is not None:
                try:
                    zmq_addr_pipe.send(
                        (
                            publish_front.getsockopt(zmq.LAST_ENDPOINT).decode(),
                            output_back.getsockopt(zmq.LAST_ENDPOINT).decode(),
                            publish_back.getsockopt(zmq.LAST_ENDPOINT).decode(),
                        )
                    )
                finally:
                    zmq_addr_pipe.close()
            # ① 等待所有引擎订阅
            # 每个引擎启动后会订阅 publish_back 套接字
            for _ in self.engines:
                if publish_back.recv() != b"\x01":
                    logger.error(
                        "DP Coordinator received unexpected message while "
                        "waiting for engines to subscribe"
                    )
                    return
            # ② 发送 READY 消息给所有引擎
            publish_back.send(b"READY")

            logger.info("All engine subscriptions received by DP coordinator")

            # ③ 创建 Poller 监听三个套接字
            poller = zmq.Poller()
            poller.register(publish_front, zmq.POLLIN)  # 前端消息
            poller.register(publish_back, zmq.POLLIN)    # 引擎订阅消息
            poller.register(output_back, zmq.POLLIN)     # 引擎输出消息
            last_publish_time = 0

            # ④ 主事件循环
            while True:
                elapsed = int(time.time() * 1000) - last_publish_time
                # 统计变化时每 100ms 发布一次，否则每 5 秒发布一次
                wait_for = self.stats_update_interval_ms if stats_changed else 5000

                # 至少等待 50ms 确保收到当前步骤的所有统计
                min_timeout = 50 if last_step_counts is None else 0

                # 轮询套接字事件
                events = poller.poll(timeout=max(min_timeout, wait_for - elapsed))
                if not events:
                    # 超时: 发布当前统计给前端
                    if last_step_counts is not None:
                        engine_req_counts_list = last_step_counts
                        last_step_counts = None
                    else:
                        engine_req_counts_list = self._get_engine_counts()
                        stats_changed = False

                    # 发布: (每引擎的请求计数, wave 编号, 引擎是否运行)
                    to_publish = (engine_req_counts_list, current_wave, engines_running)
                    publish_front.send(msgspec.msgpack.encode(to_publish))
                    last_publish_time = int(time.time() * 1000)
                    continue

                events = dict(events)
                wave_state_changed = False

                # 处理引擎订阅消息 (publish_back 套接字)
                if publish_back in events:
                    buffer = publish_back.recv()
                    if buffer == b"\x01":
                        # 新启动的引擎订阅
                        # 发送 READY 消息 (SCALE_ELASTIC_EP 在引擎初始化完成后才发送)
                        publish_back.send(b"READY")
                    elif buffer != b"\x00":
                        logger.error(
                            "DP Coordinator received unexpected message from engines"
                        )

                # 处理前端消息 (publish_front 套接字)
                if publish_front in events:
                    buffer = publish_front.recv()
                    if buffer in (b"\x01", b"\x00"):
                        # 忽略订阅消息
                        continue

                    decoded = msgspec.msgpack.decode(buffer)

                    # 处理弹性 EP 扩缩容通知
                    if (
                        isinstance(decoded, (list, tuple))
                        and len(decoded) == 2
                        and decoded[0] == "SCALE_ELASTIC_EP"
                    ):
                        new_engine_count = decoded[1]
                        current_count = len(self.engines)
                        if new_engine_count > current_count:
                            # 扩容: 添加新引擎状态
                            for _ in range(new_engine_count - current_count):
                                self.engines.append(EngineState())
                            logger.info(
                                "DPCoordinator scaled up from %s to %s engines",
                                current_count,
                                new_engine_count,
                            )
                        else:
                            # 缩容: 移除多余引擎状态
                            self.engines = self.engines[:new_engine_count]
                            logger.info(
                                "DPCoordinator scaled down from %s to %s engines",
                                current_count,
                                new_engine_count,
                            )
                        continue

                    # Wave 协调: 处理前端的新请求通知
                    if self.enable_wave_coordination:
                        # 前端发送新请求时引擎处于暂停状态，需要唤醒其他引擎
                        engine_to_exclude, wave = decoded
                        if not engines_running:
                            if wave < current_wave:
                                # wave 编号过期，确保所有引擎都处理
                                engine_to_exclude = None

                            engines_running = True
                            wave_state_changed = True
                            self._send_start_wave(
                                publish_back, current_wave, engine_to_exclude
                            )

                # 处理引擎输出消息 (output_back 套接字)
                if output_back in events:
                    # 收到引擎的消息
                    buffer = output_back.recv()
                    outputs: EngineCoreOutputs = decoder.decode(buffer)

                    assert not outputs.outputs
                    assert outputs.utility_output is None

                    eng_index = outputs.engine_index
                    scheduler_stats = outputs.scheduler_stats

                    # ① 更新负载统计
                    if scheduler_stats:
                        stats = self.engines[eng_index].request_counts
                        stats_step = scheduler_stats.step_counter
                        stats_wave = scheduler_stats.current_wave
                        # 检查统计顺序 (防止乱序)
                        if (
                            stats_wave > last_stats_wave
                            or stats_wave == last_stats_wave
                            and stats_step > last_stats_step
                        ):
                            if stats_changed:
                                last_step_counts = self._get_engine_counts(do_copy=True)
                            last_stats_step = stats_step
                            last_stats_wave = stats_wave
                        elif stats_wave != last_stats_wave or (
                            stats_step != last_stats_step
                        ):
                            logger.warning(
                                "Received stats for out-of-order "
                                "step (%d, %d) from engine %d (expected "
                                "> (%d, %d))",
                                stats_wave,
                                stats_step,
                                eng_index,
                                last_stats_wave,
                                last_stats_step,
                            )
                        stats[0] = scheduler_stats.num_waiting_reqs
                        stats[1] = scheduler_stats.num_running_reqs
                        stats_changed = True

                    # ② Wave 协调: 处理 wave 完成和开始通知
                    if self.enable_wave_coordination:
                        if (wave := outputs.wave_complete) is not None:
                            # 引擎报告 wave 完成 (所有引擎都空闲)
                            if current_wave <= wave:
                                new_wave = wave + 1
                                logger.debug(
                                    "Moving DP wave from %d to %d.",
                                    current_wave,
                                    new_wave,
                                )
                                current_wave = new_wave
                                engines_running = False  # 进入暂停状态
                                wave_state_changed = True
                        elif (wave := outputs.start_wave) is not None and (
                            wave > current_wave
                            or (wave == current_wave and not engines_running)
                        ):
                            # 引擎收到过期 wave 的请求，需要唤醒其他引擎
                            logger.debug(
                                "Starting wave %d after notification of "
                                "stale wave request from engine.",
                                wave,
                            )
                            current_wave = wave
                            engines_running = True
                            wave_state_changed = True
                            self._send_start_wave(publish_back, wave, eng_index)

                # ③ 发布 wave 状态变化给前端
                if wave_state_changed:
                    message = (None, current_wave, engines_running)
                    publish_front.send(msgspec.msgpack.encode(message))

    @staticmethod
    def _send_start_wave(
        socket: zmq.Socket, wave: int, exclude_engine_index: int | None
    ):
        """
        广播 START_DP_WAVE 消息给所有引擎。

        参数:
          wave: 当前 wave 编号
          exclude_engine_index: 已经收到此 wave 请求的引擎索引 (不需要再次通知)

        消息格式: (wave, exclude_engine_index) 通过 msgpack 编码
        """
        wave_encoded = msgspec.msgpack.encode((wave, exclude_engine_index))
        socket.send_multipart((EngineCoreRequestType.START_DP_WAVE.value, wave_encoded))

    def _get_engine_counts(self, do_copy=False) -> list[list[int]]:
        """返回每个引擎的 [waiting, running] 请求计数列表。"""
        if do_copy:
            return [copy.copy(e.request_counts) for e in self.engines]
        return [e.request_counts for e in self.engines]
