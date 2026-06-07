# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
序列化工具模块 (Serialization Utilities)
==========================================

本模块为 vLLM v1 引擎提供高性能的序列化/反序列化基础设施。

核心设计目标：
    1. 高效传输 torch.Tensor 和 numpy.ndarray：避免不必要的数据拷贝，
       通过零拷贝 (zero-copy) 或非阻塞传输提高性能。
    2. 支持大张量的带外 (Out-of-Band, OOB) 传输：超过阈值的张量
       不内联到主消息中，而是通过专用通道传输。
    3. 多模态输入支持：专门处理 MultiModalKwargs 的序列化。
    4. 安全性控制：通过环境变量控制是否允许不安全的 pickle 序列化。

技术架构：
    - 基于 msgspec.msgpack 实现高效的 MessagePack 编解码。
    - 通过自定义的 enc_hook/dec_hook 扩展支持 PyTorch 和 NumPy 类型。
    - 使用 aux_buffers 列表实现大张量的零拷贝传输。
    - 集成 Pydantic 支持，使 msgspec.Struct 可用于 API 模型。

主要类：
    1. MsgpackEncoder: 编码器，将 Python 对象序列化为 MessagePack 格式。
    2. MsgpackDecoder: 解码器，将 MessagePack 数据反序列化为 Python 对象。
    3. OOBTensorConsumer: 带外张量消费者的抽象接口。
    4. PydanticMsgspecMixin: 使 msgspec.Struct 兼容 Pydantic 的混入类。
    5. UtilityResult: 特殊序列化处理的包装器。
"""

import dataclasses
import importlib
import pickle
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from functools import partial
from inspect import isclass
from types import FunctionType
from typing import Any, ClassVar, TypeAlias, cast, get_type_hints

import cloudpickle
import msgspec
import numpy as np
import torch
import zmq
from msgspec import msgpack
from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema

from vllm import envs
from vllm.logger import init_logger
from vllm.multimodal.inputs import (
    BaseMultiModalField,
    MultiModalBatchedField,
    MultiModalFieldConfig,
    MultiModalFieldElem,
    MultiModalFlatField,
    MultiModalKwargsItem,
    MultiModalKwargsItems,
    MultiModalSharedField,
    NestedTensors,
)
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.v1.utils import tensor_data

logger = init_logger(__name__)

# 自定义 MessagePack 扩展类型编码：
#   - CUSTOM_TYPE_PICKLE (1): 使用标准 pickle 序列化
#   - CUSTOM_TYPE_CLOUDPICKLE (2): 使用 cloudpickle 序列化 (支持更多类型)
#   - CUSTOM_TYPE_RAW_VIEW (3): 原始内存视图，用于零拷贝传输
CUSTOM_TYPE_PICKLE = 1
CUSTOM_TYPE_CLOUDPICKLE = 2
CUSTOM_TYPE_RAW_VIEW = 3

# MultiModalField class serialization type map.
# These need to list all possible field types and match them
# to factory methods in `MultiModalFieldConfig`.
# 多模态字段类到工厂方法名称的映射。
# 序列化时通过类名找到对应的工厂方法名，反序列化时通过工厂方法名重建字段。
MMF_CLASS_TO_FACTORY: dict[type[BaseMultiModalField], str] = {
    MultiModalFlatField: "flat",
    MultiModalSharedField: "shared",
    MultiModalBatchedField: "batched",
}

# 字节类型别名，涵盖所有可能的字节缓冲区类型。
# ZMQ Frame 也包含在内，因为 ZMQ 消息可以直接作为字节流使用。
bytestr: TypeAlias = bytes | bytearray | memoryview | zmq.Frame


class OOBTensorConsumer(ABC):
    """
    带外 (Out-of-Band) 张量消费者的抽象接口。

    在分布式推理中，大张量 (如 KV 缓存、模型权重) 通过专用通道传输
    比内联到主消息中更高效。此接口定义了 OOB 传输的消费者端。

    工作流程：
        1. 编码器在处理大张量时调用消费者的 __call__ 方法。
        2. 消费者决定是否接管该张量的传输：
           - 返回 None：拒绝接管，张量走普通序列化路径。
           - 返回 dict：接管张量，返回的占位数据会被内联到主消息中。
        3. 接收端通过 OOBTensorProvider 从占位数据重建张量。
    """

    @abstractmethod
    def __call__(self, tensor: torch.Tensor) -> dict | None:
        """
        Called with tensors for the current message.
        Returns None to reject the tensor (falls back to regular serialization),
        otherwise a dict with arbitrary placeholder data to be included
        in the serialized message.
        """
        return None

    @abstractmethod
    def new_message(self) -> None:
        """Called at the start of each new encoded message."""
        pass


# dtype, shape, metadata -> tensor
# 带外张量提供者的类型别名。
# 接收端通过此回调从 dtype、shape 和元数据重建张量。
OOBTensorProvider = Callable[[str, tuple[int, ...], dict], torch.Tensor]


def _log_insecure_serialization_warning():
    """记录不安全序列化的警告信息。"""
    logger.warning_once(
        "Allowing insecure serialization using pickle due to "
        "VLLM_ALLOW_INSECURE_SERIALIZATION=1"
    )


def _typestr(val: Any) -> tuple[str, str] | None:
    """
    获取对象的类型字符串表示。

    返回 (模块名, 限定类名) 元组，用于序列化时记录类型信息，
    反序列化时可通过 importlib.import_module 重建类型。

    参数：
        val: 任意对象。

    返回：
        tuple[str, str] | None: (模块名, 限定类名) 或 None (输入为 None 时)。
    """
    if val is None:
        return None
    t = type(val)
    return t.__module__, t.__qualname__


def _encode_type_info_recursive(obj: Any) -> Any:
    """Recursively encode type information for nested structures of
    lists/dicts."""
    """
    递归编码嵌套结构中的类型信息。

    对于 UtilityResult 中的复杂嵌套结构 (list/dict 中包含自定义类型)，
    需要递归记录每个元素的类型信息，以便反序列化时正确重建。

    参数：
        obj: 可能包含嵌套 list/dict/自定义类型的对象。

    返回：
        与输入结构相同但叶子节点替换为类型字符串的结构。
    """
    if obj is None:
        return None
    if type(obj) is list:
        return [_encode_type_info_recursive(item) for item in obj]
    if type(obj) is dict:
        return {k: _encode_type_info_recursive(v) for k, v in obj.items()}
    return _typestr(obj)


def _decode_type_info_recursive(
    type_info: Any, data: Any, convert_fn: Callable[[Sequence[str], Any], Any]
) -> Any:
    """Recursively decode type information for nested structures of
    lists/dicts."""
    """
    递归解码嵌套结构中的类型信息。

    与 _encode_type_info_recursive 对应，根据编码时记录的类型信息
    将原始数据转换回正确的 Python 类型。

    参数：
        type_info: 编码的类型信息结构。
        data: 原始数据。
        convert_fn: 类型转换函数，接受 (类型字符串, 数据) 并返回转换结果。

    返回：
        转换后的数据结构。
    """
    if type_info is None:
        return data
    if isinstance(type_info, dict):
        assert isinstance(data, dict)
        return {
            k: _decode_type_info_recursive(type_info[k], data[k], convert_fn)
            for k in type_info
        }
    if isinstance(type_info, list) and (
        # Exclude serialized tensors/numpy arrays.
        # 排除已序列化的张量/NumPy 数组 (长度为 2 且第一个元素是字符串的元组)
        len(type_info) != 2 or not isinstance(type_info[0], str)
    ):
        assert isinstance(data, list)
        return [
            _decode_type_info_recursive(ti, d, convert_fn)
            for ti, d in zip(type_info, data)
        ]
    return convert_fn(type_info, data)


class UtilityResult:
    """Wrapper for special handling when serializing/deserializing."""
    """
    特殊序列化处理的包装器。

    当需要序列化 msgspec 原生不支持的自定义类型时，将其包装在
    UtilityResult 中。编码器会为其添加额外的类型信息以支持正确反序列化。

    使用场景：
        - 允许不安全序列化时 (VLLM_ALLOW_INSECURE_SERIALIZATION=1)：
          记录完整的类型信息，支持自定义类型的反序列化。
        - 不允许时：类型信息设为 None，数据保持原样。

    属性：
        result: 被包装的实际数据。
    """

    def __init__(self, r: Any = None):
        self.result = r


class MsgpackEncoder:
    """Encoder with custom torch tensor and numpy array serialization.

    Note that unlike vanilla `msgspec` Encoders, this interface is generally
    not thread-safe when encoding tensors / numpy arrays.

    By default, arrays below 256B are serialized inline Larger will get sent
    via dedicated messages. Note that this is a per-tensor limit.

    When a ``oob_tensor_consumer`` is provided, tensors (CUDA and CPU) will be
    offered to it for out-of-band handling.
    """
    """
    自定义 MessagePack 编码器，支持 torch.Tensor 和 numpy.ndarray。

    设计要点：
        1. 零拷贝优化：大张量不内联到主消息中，而是通过 aux_buffers
           列表保存对底层内存的引用，直接返回给调用者。
        2. 大小阈值：小于阈值 (默认 256B) 的张量内联编码；
           大于阈值的张量通过索引引用或 OOB 通道传输。
        3. OOB 支持：当提供 oob_tensor_consumer 时，张量会被
           交给消费者处理，返回占位数据。
        4. 线程安全注意：编码 torch.Tensor 和 numpy.ndarray 时
           不是线程安全的 (因为共享 aux_buffers 状态)。

    编码流程：
        1. 调用 encode() 或 encode_into() 开始编码。
        2. msgspec 在遇到不支持的类型时调用 enc_hook()。
        3. enc_hook() 根据类型分发到对应的编码方法。
        4. 张量/数组数据被添加到 aux_buffers，主消息中只存索引。
        5. 返回 [主消息, 辅助缓冲区1, 辅助缓冲区2, ...]。
    """

    def __init__(
        self,
        size_threshold: int | None = None,
        oob_tensor_consumer: OOBTensorConsumer | None = None,
    ):
        """
        初始化编码器。

        参数：
            size_threshold: 张量内联编码的大小阈值 (字节)。
                小于此阈值的张量会内联到主消息中。
                默认使用环境变量 VLLM_MSGPACK_ZERO_COPY_THRESHOLD。
            oob_tensor_consumer: 带外张量消费者。提供时，所有张量
                (包括 CUDA 和 CPU) 会先交给消费者处理。
        """
        if size_threshold is None:
            size_threshold = envs.VLLM_MSGPACK_ZERO_COPY_THRESHOLD
        self.encoder = msgpack.Encoder(enc_hook=self.enc_hook)
        # This is used as a local stash of buffers that we can then access from
        # our custom `msgspec` hook, `enc_hook`. We don't have a way to
        # pass custom data to the hook otherwise.
        # 辅助缓冲区列表：用于存储大张量/数组的底层内存引用。
        # enc_hook 无法接收自定义参数，因此通过实例变量传递。
        self.aux_buffers: list[bytestr] | None = None
        self.size_threshold = size_threshold
        self.oob_tensor_consumer = oob_tensor_consumer
        if envs.VLLM_ALLOW_INSECURE_SERIALIZATION:
            _log_insecure_serialization_warning()

    def encode(self, obj: Any) -> Sequence[bytestr]:
        """
        将对象编码为 MessagePack 字节序列。

        返回的序列第一个元素是主消息，后续元素是大张量/数组的
        底层内存缓冲区。调用者可以将这些缓冲区一起发送以实现零拷贝。

        参数：
            obj: 要编码的对象。

        返回：
            Sequence[bytestr]: 编码后的字节序列。
        """
        try:
            if self.oob_tensor_consumer is not None:
                self.oob_tensor_consumer.new_message()
            self.aux_buffers = bufs = [b""]
            bufs[0] = self.encoder.encode(obj)
            # This `bufs` list allows us to collect direct pointers to backing
            # buffers of tensors and np arrays, and return them along with the
            # top-level encoded buffer instead of copying their data into the
            # new buffer.
            return bufs
        finally:
            self.aux_buffers = None

    def encode_into(self, obj: Any, buf: bytearray) -> Sequence[bytestr]:
        """
        将对象编码到预分配的字节缓冲区中。

        与 encode() 类似，但复用调用者提供的缓冲区，避免额外分配。

        参数：
            obj: 要编码的对象。
            buf: 预分配的字节缓冲区。

        返回：
            Sequence[bytestr]: 编码后的字节序列 (buf 作为第一个元素)。
        """
        try:
            if self.oob_tensor_consumer is not None:
                self.oob_tensor_consumer.new_message()
            self.aux_buffers = [buf]
            bufs = self.aux_buffers
            self.encoder.encode_into(obj, buf)
            return bufs
        finally:
            self.aux_buffers = None

    def enc_hook(self, obj: Any) -> Any:
        """
        msgspec 的自定义编码钩子，处理原生不支持的类型。

        按类型优先级分发：
            1. torch.Tensor -> _encode_tensor()
            2. numpy.ndarray (非 object/void) -> _encode_ndarray()
            3. slice -> 转换为 (start, stop, step) 元组
            4. MultiModalKwargsItem -> _encode_mm_item()
            5. MultiModalKwargsItems -> _encode_mm_items()
            6. UtilityResult -> 带类型信息的特殊编码
            7. 其他类型 -> pickle/cloudpickle (需允许不安全序列化)

        参数：
            obj: 需要编码的对象。

        返回：
            编码后的原生类型数据。
        """
        if isinstance(obj, torch.Tensor):
            return self._encode_tensor(obj)

        # Fall back to pickle for object or void kind ndarrays.
        # 对于 object 或 void 类型的 ndarray，无法直接编码，
        # 需要回退到 pickle。
        if isinstance(obj, np.ndarray) and obj.dtype.kind not in ("O", "V"):
            return self._encode_ndarray(obj)

        if isinstance(obj, slice):
            # We are assuming only int-based values will be used here.
            return tuple(
                int(v) if v is not None else None
                for v in (obj.start, obj.stop, obj.step)
            )

        if isinstance(obj, MultiModalKwargsItem):
            return self._encode_mm_item(obj)

        if isinstance(obj, MultiModalKwargsItems):
            return self._encode_mm_items(obj)

        if isinstance(obj, UtilityResult):
            result = obj.result
            if not envs.VLLM_ALLOW_INSECURE_SERIALIZATION:
                return None, result
            # Since utility results are not strongly typed, we recursively
            # encode type information for nested structures of lists/dicts
            # to help with correct msgspec deserialization.
            return _encode_type_info_recursive(result), result

        if not envs.VLLM_ALLOW_INSECURE_SERIALIZATION:
            raise TypeError(
                f"Object of type {type(obj)} is not serializable"
                "Set VLLM_ALLOW_INSECURE_SERIALIZATION=1 to allow "
                "fallback to pickle-based serialization."
            )

        if isinstance(obj, FunctionType):
            # `pickle` is generally faster than cloudpickle, but can have
            # problems serializing methods.
            # 函数类型使用 cloudpickle，因为 pickle 无法正确序列化方法。
            return msgpack.Ext(CUSTOM_TYPE_CLOUDPICKLE, cloudpickle.dumps(obj))

        # 其他类型使用标准 pickle 序列化
        return msgpack.Ext(
            CUSTOM_TYPE_PICKLE, pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        )

    def _encode_ndarray(
        self, obj: np.ndarray
    ) -> tuple[str, tuple[int, ...], int | memoryview]:
        """
        编码 NumPy 数组。

        策略：
            1. 小数组或标量：使用 CUSTOM_TYPE_RAW_VIEW 内联编码，
               反序列化时可直接零拷贝访问。
            2. 大数组：将底层内存添加到 aux_buffers，主消息中只存索引，
               实现零拷贝传输。

        参数：
            obj: 要编码的 NumPy 数组。

        返回：
            (dtype字符串, shape, 数据或索引) 三元组。
        """
        assert self.aux_buffers is not None
        # If the array is non-contiguous, we need to copy it first
        # 非连续数组需要先拷贝为连续内存
        arr_data = obj.data if obj.flags.c_contiguous else obj.tobytes()
        if not obj.shape or obj.nbytes < self.size_threshold:
            # Encode small arrays and scalars inline. Using this extension type
            # ensures we can avoid copying when decoding.
            data = msgpack.Ext(CUSTOM_TYPE_RAW_VIEW, arr_data)
        else:
            # Otherwise encode index of backing buffer to avoid copy.
            # 大数组：存索引到 aux_buffers，避免数据拷贝
            data = len(self.aux_buffers)
            self.aux_buffers.append(arr_data)

        # We serialize the ndarray as a tuple of native types.
        # The data is either inlined if small, or an index into a list of
        # backing buffers that we've stashed in `aux_buffers`.
        return obj.dtype.str, obj.shape, data

    def _encode_tensor(
        self, obj: torch.Tensor
    ) -> tuple[str, tuple[int, ...], int | dict | memoryview]:
        """
        编码 PyTorch 张量。

        策略 (按优先级)：
            1. 小 CPU 张量 (< size_threshold)：使用 CUSTOM_TYPE_RAW_VIEW
               内联编码。
            2. OOB 消费者接受：交给消费者处理，返回占位字典。
            3. 其他：将底层内存添加到 aux_buffers，存索引实现零拷贝。

        参数：
            obj: 要编码的 PyTorch 张量。

        返回：
            (dtype字符串, shape, 数据/索引/占位字典) 三元组。
        """
        oob_consumer = self.oob_tensor_consumer
        # view the tensor as a contiguous 1D array of bytes
        if obj.nbytes < self.size_threshold and obj.is_cpu:
            # Smaller tensors are encoded inline, just like ndarrays.
            data = msgpack.Ext(CUSTOM_TYPE_RAW_VIEW, tensor_data(obj))
        elif oob_consumer is not None and (data := oob_consumer(obj)) is not None:
            assert isinstance(data, dict)
        else:
            # Otherwise encode index of backing buffer to avoid copy.
            assert self.aux_buffers is not None
            data = len(self.aux_buffers)
            self.aux_buffers.append(tensor_data(obj))
        dtype = str(obj.dtype).removeprefix("torch.")
        return dtype, obj.shape, data

    def _encode_mm_items(self, items: MultiModalKwargsItems) -> dict[str, Any]:
        """
        编码 MultiModalKwargsItems (多模态输入的批量容器)。

        按模态 (modality) 分组编码每个 MultiModalKwargsItem。

        返回：
            dict: {模态名: [编码后的 item, ...]} 字典。
        """
        return {
            modality: [self._encode_mm_item(item) for item in itemlist]
            for modality, itemlist in items.items()
        }

    def _encode_mm_item(self, item: MultiModalKwargsItem) -> dict[str, Any]:
        """
        编码单个 MultiModalKwargsItem。

        将每个字段元素 (MultiModalFieldElem) 编码为字典。

        返回：
            dict: {字段名: 编码后的元素} 字典。
        """
        return {key: self._encode_mm_field_elem(elem) for key, elem in item.items()}

    def _encode_mm_field_elem(self, elem: MultiModalFieldElem) -> dict[str, Any]:
        """
        编码单个多模态字段元素。

        包含两部分：
            1. data: 字段数据 (嵌套张量)，可能为 None。
            2. field: 字段处理器的序列化表示 (工厂方法名 + 参数)。

        返回：
            dict: {"data": ..., "field": ...} 字典。
        """
        return {
            "data": (
                None if elem.data is None else self._encode_nested_tensors(elem.data)
            ),
            "field": self._encode_mm_field(elem.field),
        }

    def _encode_nested_tensors(self, nt: NestedTensors) -> Any:
        """
        递归编码嵌套张量结构。

        NestedTensors 可以是：
            - 单个 torch.Tensor
            - 数值标量 (int/float，虽然违反类型定义但实际存在)
            - 嵌套的列表

        返回：
            编码后的数据结构。
        """
        if isinstance(nt, torch.Tensor):
            return self._encode_tensor(nt)
        if isinstance(nt, (int, float)):
            # Although it violates NestedTensors type, MultiModalKwargs
            # values are sometimes floats.
            return nt
        return [self._encode_nested_tensors(x) for x in nt]

    def _encode_mm_field(self, field: BaseMultiModalField):
        """
        编码多模态字段处理器。

        序列化字段的所有属性，以便反序列化时通过工厂方法重建。

        返回：
            (工厂方法名, 参数字典) 元组。
            工厂方法名对应 MultiModalFieldConfig 的方法 (如 "flat", "shared", "batched")。
        """
        # Figure out the factory name for the field type.
        name = MMF_CLASS_TO_FACTORY.get(field.__class__)
        if not name:
            raise TypeError(f"Unsupported field type: {field.__class__}")

        # We just need to copy all of the field values in order
        # which will be then used to reconstruct the field.
        factory_kw = {f.name: getattr(field, f.name) for f in dataclasses.fields(field)}
        return name, factory_kw


class MsgpackDecoder:
    """Decoder with custom torch tensor and numpy array serialization.

    Note that unlike vanilla `msgspec` Decoders, this interface is generally
    not thread-safe when encoding tensors / numpy arrays.

    ``oob_tensor_provider`` must be used when an OOBTensorConsumer is used on the
    encoder side.
    """
    """
    自定义 MessagePack 解码器，支持 torch.Tensor 和 numpy.ndarray。

    与 MsgpackEncoder 对应，负责将编码后的字节序列还原为 Python 对象。

    设计要点：
        1. 零拷贝解码：从 aux_buffers 中直接引用底层内存，避免数据拷贝。
        2. 内存共享控制：share_mem=True 时，解码后的数组共享原始缓冲区；
           share_mem=False 时，会拷贝数据以确保独立性。
        3. Pinned Memory：自动将张量放入 pinned memory，加速后续
           CPU 到 GPU 的传输。
        4. OOB 支持：当编码端使用 OOBTensorConsumer 时，解码端必须
           提供对应的 OOBTensorProvider 来重建张量。

    解码流程：
        1. 调用 decode() 开始解码。
        2. msgspec 在遇到自定义类型时调用 dec_hook() 或 ext_hook()。
        3. dec_hook() 处理结构化类型 (Tensor, ndarray, slice 等)。
        4. ext_hook() 处理扩展类型 (RAW_VIEW, pickle 等)。
        5. 张量从 aux_buffers 中零拷贝重建。
    """

    def __init__(
        self,
        t: Any | None = None,
        share_mem: bool = True,
        oob_tensor_provider: OOBTensorProvider | None = None,
    ):
        """
        初始化解码器。

        参数：
            t: 目标类型，用于 msgspec 的类型化解码。为 None 时
               不做类型检查。
            share_mem: 是否共享内存。True 时解码后的数组直接引用
               原始缓冲区 (零拷贝)；False 时会拷贝数据。
            oob_tensor_provider: 带外张量提供者。当编码端使用
                OOBTensorConsumer 时必须提供。
        """
        self.share_mem = share_mem
        # 检查是否可用 pinned memory (需要 CUDA 支持)
        self.pin_tensors = is_pin_memory_available()
        args = () if t is None else (t,)
        self.decoder = msgpack.Decoder(
            *args, ext_hook=self.ext_hook, dec_hook=self.dec_hook
        )
        # 辅助缓冲区序列，解码时用于零拷贝访问大张量的底层内存
        self.aux_buffers: Sequence[bytestr] = ()
        self.oob_tensor_provider = oob_tensor_provider
        if envs.VLLM_ALLOW_INSECURE_SERIALIZATION:
            _log_insecure_serialization_warning()

    def decode(self, bufs: bytestr | Sequence[bytestr]) -> Any:
        """
        从字节序列解码对象。

        参数：
            bufs: 单个字节缓冲区或缓冲区序列。
                - 单个缓冲区：直接解码，无辅助数据。
                - 缓冲区序列：第一个元素是主消息，后续是辅助缓冲区。

        返回：
            解码后的 Python 对象。
        """
        if isinstance(bufs, bytestr):  # type: ignore
            return self.decoder.decode(bufs)

        self.aux_buffers = bufs
        try:
            return self.decoder.decode(bufs[0])
        finally:
            self.aux_buffers = ()

    def dec_hook(self, t: type, obj: Any) -> Any:
        # Given native types in `obj`, convert to type `t`.
        """
        msgspec 的自定义解码钩子，将原生类型转换为目标类型。

        根据目标类型 t 分发到对应的解码方法。

        参数：
            t: 目标类型。
            obj: 编码后的原生类型数据。

        返回：
            转换后的目标类型实例。
        """
        if isclass(t):
            if issubclass(t, np.ndarray):
                return self._decode_ndarray(obj)
            if issubclass(t, torch.Tensor):
                return self._decode_tensor(obj)
            if t is slice:
                return slice(*obj)
            if issubclass(t, MultiModalKwargsItem):
                return self._decode_mm_item(obj)
            if issubclass(t, MultiModalKwargsItems):
                return self._decode_mm_items(obj)
            if t is UtilityResult:
                return self._decode_utility_result(obj)
        return obj

    def _decode_utility_result(self, obj: Any) -> UtilityResult:
        """
        解码 UtilityResult 对象。

        根据编码时记录的类型信息，递归解码嵌套结构中的自定义类型。

        参数：
            obj: (类型信息, 数据) 元组。

        返回：
            UtilityResult: 解码后的包装器。
        """
        result_type, result = obj
        if result_type is not None:
            if not envs.VLLM_ALLOW_INSECURE_SERIALIZATION:
                raise TypeError(
                    "VLLM_ALLOW_INSECURE_SERIALIZATION must "
                    "be set to use custom utility result types"
                )
            # Use recursive decoding to handle nested structures
            result = _decode_type_info_recursive(
                result_type, result, self._convert_result
            )
        return UtilityResult(result)

    def _convert_result(self, result_type: Sequence[str], result: Any) -> Any:
        """
        将数据转换为指定类型。

        通过模块名和类名动态导入类型，然后使用 msgspec.convert 进行转换。

        参数：
            result_type: (模块名, 类名) 序列。
            result: 要转换的数据。

        返回：
            转换后的类型实例。
        """
        if result_type is None:
            return result
        mod_name, name = result_type
        mod = importlib.import_module(mod_name)
        result_type = getattr(mod, name)
        return msgspec.convert(result, result_type, dec_hook=self.dec_hook)

    def _decode_ndarray(self, arr: Any) -> np.ndarray:
        """
        解码 NumPy 数组。

        实现零拷贝解码：直接从 aux_buffers 中引用原始内存。
        前提是解码后的数组不会被长期持有，否则会锁定整个消息缓冲区。

        参数：
            arr: (dtype字符串, shape, 数据或索引) 三元组。

        返回：
            np.ndarray: 解码后的 NumPy 数组。
        """
        dtype, shape, data = arr
        # zero-copy decode. We assume the ndarray will not be kept around,
        # as it now locks the whole received message buffer in memory.
        buffer = self.aux_buffers[data] if isinstance(data, int) else data
        arr = np.frombuffer(buffer, dtype=dtype)
        if not self.share_mem:
            arr = arr.copy()
        return arr.reshape(shape)

    def _decode_tensor(self, arr: Any) -> torch.Tensor:
        """
        解码 PyTorch 张量。

        解码流程：
            1. 如果数据是 dict，说明是 OOB 张量，通过 provider 重建。
            2. 从 aux_buffers 或内联数据获取原始字节缓冲区。
            3. 通过 torch.frombuffer 创建 uint8 张量。
            4. 根据情况决定是否 clone 或 pin_memory：
               - 非辅助缓冲区：clone 以确保内存归 PyTorch 管理。
               - 辅助缓冲区但不共享内存：pin_memory (如果可用) 或 clone。
            5. 转换为正确的 dtype 和 shape。

        参数：
            arr: (dtype字符串, shape, 数据/索引/占位字典) 三元组。

        返回：
            torch.Tensor: 解码后的 PyTorch 张量。
        """
        dtype, shape, data = arr
        if isinstance(data, dict):
            assert self.oob_tensor_provider, (
                "Received OOB tensor but tensor provider is not set"
            )
            return self.oob_tensor_provider(dtype, shape, data)

        is_aux = isinstance(data, int)
        buffer = self.aux_buffers[data] if is_aux else data
        buffer = buffer if isinstance(buffer, memoryview) else memoryview(buffer)
        torch_dtype = getattr(torch, dtype)
        assert isinstance(torch_dtype, torch.dtype)
        if not buffer.nbytes:  # torch.frombuffer doesn't like empty buffers
            assert 0 in shape
            return torch.empty(shape, dtype=torch_dtype)
        # Create uint8 array
        arr = torch.frombuffer(buffer, dtype=torch.uint8)
        # Clone ensures tensor is backed by pytorch-owned memory for safe
        # future async CPU->GPU transfer.
        # Pin larger tensors for more efficient CPU->GPU transfer.
        if not is_aux:
            # 内联数据：clone 确保张量内存归 PyTorch 管理，
            # 支持安全的异步 CPU->GPU 传输
            arr = arr.clone()
        elif not self.share_mem:
            # 辅助缓冲区且不共享内存：尝试 pin_memory 以加速传输
            arr = arr.pin_memory() if self.pin_tensors else arr.clone()
        # Convert back to proper shape & type
        return arr.view(torch_dtype).view(shape)

    def _decode_mm_items(self, obj: dict[str, Any]) -> MultiModalKwargsItems:
        """
        解码 MultiModalKwargsItems (多模态输入的批量容器)。

        按模态分组解码每个 MultiModalKwargsItem。

        返回：
            MultiModalKwargsItems: 解码后的多模态输入容器。
        """
        return MultiModalKwargsItems(
            {
                modality: [self._decode_mm_item(item) for item in itemlist]
                for modality, itemlist in obj.items()
            }
        )

    def _decode_mm_item(self, obj: dict[str, Any]) -> MultiModalKwargsItem:
        """
        解码单个 MultiModalKwargsItem。

        返回：
            MultiModalKwargsItem: 解码后的多模态输入项。
        """
        return MultiModalKwargsItem(
            {key: self._decode_mm_field_elem(elem) for key, elem in obj.items()}
        )

    def _decode_mm_field_elem(self, obj: dict[str, Any]) -> MultiModalFieldElem:
        """
        解码单个多模态字段元素。

        步骤：
            1. 解码 data (嵌套张量数据)，可能为 None。
            2. 从 field 中提取工厂方法名和参数。
            3. 通过 MultiModalFieldConfig 的工厂方法重建字段处理器。
            4. 特殊处理 MultiModalFlatField 的 slices 字段
               (union 类型需要额外解码)。

        返回：
            MultiModalFieldElem: 解码后的字段元素。
        """
        if obj["data"] is not None:
            obj["data"] = self._decode_nested_tensors(obj["data"])

        # Reconstruct the field processor using MultiModalFieldConfig
        factory_meth_name, factory_kw = obj["field"]
        factory_meth = getattr(MultiModalFieldConfig, factory_meth_name)

        # Special case: decode the union "slices" field of
        # MultiModalFlatField
        if factory_meth_name == "flat":
            factory_kw["slices"] = self._decode_nested_slices(factory_kw["slices"])

        obj["field"] = factory_meth("", **factory_kw).field
        return MultiModalFieldElem(**obj)

    def _decode_nested_tensors(self, obj: Any) -> NestedTensors:
        """
        递归解码嵌套张量结构。

        与 _encode_nested_tensors 对应，根据编码后的格式还原。

        参数：
            obj: 编码后的嵌套结构。

        返回：
            NestedTensors: 解码后的嵌套张量。
        """
        if isinstance(obj, (int, float)):
            # Although it violates NestedTensors type, MultiModalKwargs
            # values are sometimes floats.
            return obj
        if not isinstance(obj, list):
            raise TypeError(f"Unexpected NestedTensors contents: {type(obj)}")
        if obj and isinstance(obj[0], str):
            return self._decode_tensor(obj)
        return [self._decode_nested_tensors(x) for x in obj]

    def _decode_nested_slices(self, obj: Any) -> Any:
        """
        递归解码嵌套的 slice 结构。

        MultiModalFlatField 中的 slices 字段可能是嵌套的 slice 结构，
        需要递归还原。

        参数：
            obj: 编码后的 slice 数据 (元组或嵌套列表)。

        返回：
            还原后的 slice 或嵌套 slice 结构。
        """
        assert isinstance(obj, (list, tuple))
        if obj and not isinstance(obj[0], (list, tuple)):
            return slice(*obj)
        return [self._decode_nested_slices(x) for x in obj]

    def ext_hook(self, code: int, data: memoryview) -> Any:
        """
        msgspec 的扩展类型解码钩子。

        处理编码时通过 msgpack.Ext 注册的自定义扩展类型。

        支持的扩展类型：
            - CUSTOM_TYPE_RAW_VIEW (3): 直接返回 memoryview (零拷贝)。
            - CUSTOM_TYPE_PICKLE (1): 使用 pickle 反序列化 (需允许不安全序列化)。
            - CUSTOM_TYPE_CLOUDPICKLE (2): 使用 cloudpickle 反序列化。

        参数：
            code: 扩展类型编码。
            data: 扩展类型数据。

        返回：
            解码后的对象。
        """
        if code == CUSTOM_TYPE_RAW_VIEW:
            return data

        if envs.VLLM_ALLOW_INSECURE_SERIALIZATION:
            if code == CUSTOM_TYPE_PICKLE:
                return pickle.loads(data)
            if code == CUSTOM_TYPE_CLOUDPICKLE:
                return cloudpickle.loads(data)

        raise NotImplementedError(f"Extension type code {code} is not supported")


def run_method(
    obj: Any,
    method: str | bytes | Callable,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    """
    Run a method of an object with the given arguments and keyword arguments.
    If the method is string, it will be converted to a method using getattr.
    If the method is serialized bytes and will be deserialized using
    cloudpickle.
    If the method is a callable, it will be called directly.
    """
    """
    通用方法调用工具，支持多种方法表示形式。

    在分布式 RPC 场景中，方法可能以不同形式传输：
        1. 字符串：方法名，通过 getattr 查找。
        2. 字节序列：cloudpickle 序列化的方法，需要反序列化。
        3. 可调用对象：直接调用。

    此函数统一处理这三种情况，简化 RPC 方法调用逻辑。

    参数：
        obj: 方法所属的对象实例。
        method: 方法表示，可以是字符串、序列化字节或可调用对象。
        args: 位置参数元组。
        kwargs: 关键字参数字典。

    返回：
        方法调用的结果。
    """
    if isinstance(method, bytes):
        # 方法以 cloudpickle 字节形式传输，需要反序列化
        func = partial(cloudpickle.loads(method), obj)
    elif isinstance(method, str):
        # 方法以字符串形式传输，通过 getattr 查找
        try:
            func = getattr(obj, method)
        except AttributeError:
            raise NotImplementedError(
                f"Method {method!r} is not implemented."
            ) from None
    else:
        # 方法是可调用对象，直接绑定到 obj
        func = partial(method, obj)  # type: ignore
    return func(*args, **kwargs)


class PydanticMsgspecMixin:
    """Make a ``msgspec.Struct`` compatible with Pydantic for both
    **validation** (JSON/dict -> Struct) and **serialization**
    (Struct -> JSON-safe dict).

    Subclasses may set ``__pydantic_msgspec_exclude__`` (a ``set[str]``)
    to list non-underscore field names that should also be stripped from
    serialized output.  Fields whose names start with ``_`` are always
    excluded automatically.
    """
    """
    使 msgspec.Struct 兼容 Pydantic 的混入类。

    背景：
        vLLM 使用 msgspec.Struct 作为高性能的数据结构 (比 dataclass 更快)，
        但 API 层使用 Pydantic 进行输入验证和文档生成。
        此混入类桥接两者，使 msgspec.Struct 可以：
        1. 作为 API 输入接受 JSON/dict 数据并自动验证。
        2. 生成正确的 JSON/OpenAPI Schema 文档。
        3. 序列化时自动排除私有字段。

    使用方法：
        class MyConfig(msgspec.Struct, PydanticMsgspecMixin):
            name: str
            _internal: int = 0  # 自动排除
            __pydantic_msgspec_exclude__ = {"debug_flag"}  # 手动排除

    字段排除规则：
        1. 以 "_" 开头的字段自动排除。
        2. __pydantic_msgspec_exclude__ 中列出的字段排除。
    """

    # Subclasses can override to exclude additional public-but-internal keys.
    # 子类可以覆盖此集合以排除额外的公共但内部的字段
    __pydantic_msgspec_exclude__: ClassVar[set[str]] = set()

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        """
        Make msgspec.Struct compatible with Pydantic, respecting defaults.
        Handle JSON=>msgspec.Struct. Used when exposing msgspec.Struct to the
        API as input or in `/docs`. Note this is cached by Pydantic and not
        called on every validation.
        """
        """
        生成 Pydantic 核心 Schema，使 msgspec.Struct 兼容 Pydantic。

        此方法被 Pydantic 缓存，不会在每次验证时调用。

        处理逻辑：
            1. 获取 msgspec.Struct 的所有字段定义。
            2. 为每个字段构建 Pydantic 的 typed_dict_field：
               - 跳过 ClassVar 等非结构体注解。
               - 跳过以 "_" 开头的私有字段。
               - 有默认值的字段标记为 required=False。
            3. 构建验证器：接受已构造的 Struct 实例或 JSON/dict 输入。
            4. 构建序列化器：自动排除私有和被排除的字段。

        返回：
            core_schema.CoreSchema: Pydantic 核心 Schema。
        """
        msgspec_fields = {f.name: f for f in msgspec.structs.fields(source_type)}
        type_hints = get_type_hints(source_type)

        # Build the Pydantic typed_dict_field for each msgspec field
        fields = {}
        for name, hint in type_hints.items():
            if name not in msgspec_fields:
                # Skip ClassVar and other non-struct annotations.
                continue
            # Skip private fields — they are excluded from serialization
            # and should not appear in the generated JSON/OpenAPI schema.
            if name.startswith("_"):
                continue
            msgspec_field = msgspec_fields[name]

            # typed_dict_field using the handler to get the schema
            field_schema = handler(hint)

            # Add default value to the schema.
            # Mark fields with defaults as not required so the generated
            # JSON Schema stays consistent with ``omit_defaults=True``
            # serialization (fields at their default value may be absent).
            if msgspec_field.default_factory is not msgspec.NODEFAULT:
                # 有 default_factory 的字段
                wrapped_schema = core_schema.with_default_schema(
                    schema=field_schema,
                    default_factory=msgspec_field.default_factory,
                )
                fields[name] = core_schema.typed_dict_field(
                    wrapped_schema, required=False
                )
            elif msgspec_field.default is not msgspec.NODEFAULT:
                # 有默认值的字段
                wrapped_schema = core_schema.with_default_schema(
                    schema=field_schema,
                    default=msgspec_field.default,
                )
                fields[name] = core_schema.typed_dict_field(
                    wrapped_schema, required=False
                )
            else:
                # No default, so Pydantic will treat it as required
                # 无默认值的字段，Pydantic 将视为必填
                fields[name] = core_schema.typed_dict_field(field_schema)
        typed_dict_then_convert = core_schema.no_info_after_validator_function(
            cls._validate_msgspec,
            core_schema.typed_dict_schema(fields),
        )

        # Build a serializer that strips private / excluded fields.
        # 构建序列化器，自动剥离私有和被排除的字段
        serializer = core_schema.plain_serializer_function_ser_schema(
            cls._serialize_msgspec,
            info_arg=False,
        )

        # Accept either an already-constructed msgspec.Struct instance or a
        # JSON/dict-like payload.
        # 接受已构造的 Struct 实例或 JSON/dict 载荷
        return core_schema.union_schema(
            [
                core_schema.is_instance_schema(source_type),
                typed_dict_then_convert,
            ],
            serialization=serializer,
        )

    @classmethod
    def _validate_msgspec(cls, value: Any) -> Any:
        """Validate and convert input to msgspec.Struct instance."""
        """
        验证并转换输入为 msgspec.Struct 实例。

        支持三种输入形式：
            1. 已是 Struct 实例：直接返回。
            2. dict：使用关键字参数构造 Struct。
            3. 其他：使用 msgspec.convert 转换。

        参数：
            value: 输入数据。

        返回：
            msgspec.Struct 实例。
        """
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            return cls(**value)
        return msgspec.convert(value, type=cls)

    @staticmethod
    def _serialize_msgspec(value: Any) -> Any:
        """Serialize a msgspec.Struct to a JSON-compatible dict, stripping
        private (``_``-prefixed) and explicitly excluded fields.

        Uses ``msgspec.to_builtins`` which respects ``omit_defaults=True``,
        so only fields that differ from their declared defaults are included.
        """
        """
        将 msgspec.Struct 序列化为 JSON 兼容的字典。

        自动剥离：
            1. 以 "_" 开头的私有字段。
            2. __pydantic_msgspec_exclude__ 中列出的字段。

        使用 msgspec.to_builtins 并配合 omit_defaults=True，
        只包含与默认值不同的字段。

        参数：
            value: msgspec.Struct 实例。

        返回：
            dict: JSON 兼容的字典。
        """
        raw = msgspec.to_builtins(value)
        if not isinstance(raw, dict):
            return raw

        exclude: set[str] = cast(
            set[str],
            getattr(type(value), "__pydantic_msgspec_exclude__", set()),
        )
        for key in list(raw):
            if key.startswith("_") or key in exclude:
                del raw[key]

        return raw
