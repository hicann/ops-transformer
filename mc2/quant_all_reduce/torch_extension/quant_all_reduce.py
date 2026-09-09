# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from typing import Optional
import torch
import torch_npu
from torch.library import impl
from torch_npu.utils._error_code import ErrCode, ops_error
from cann_ops_transformer.op_builder import OpBuilder, get_as_library
from ..common import CommContextManager
import atexit


TORCH_DTYPE_ENUM_VALUE_TO_SCALAR_TYPE_MAP = {
    0: torch.uint8,
    1: torch.int8,
    2: torch.int16,
    3: torch.int32,
    4: torch.int64,
    5: torch.float16,
    6: torch.float32,
    7: torch.float64,
    8: torch.complex32,
    9: torch.complex64,
    10: torch.complex128,
    11: torch.bool,
    12: torch.qint8,
    13: torch.quint8,
    14: torch.qint32,
    15: torch.bfloat16,
    16: torch.quint4x2,
    21: torch.bits8,
    23: torch.float8_e5m2,
    24: torch.float8_e4m3fn,
    285: torch.uint8,  # torch_npu.int4 use torch.uint8
    290: torch.uint8,  # torch_npu.hifloat8 use torch.uint8
    291: torch.float8_e5m2,
    292: torch.float8_e4m3fn,
    296: torch.uint8,  # torch_npu.float4_e2m1fn_x2 use torch.uint8
    297: torch.uint8,  # torch_npu.float4_e1m2fn_x2 use torch.uint8
}


class QuantAllReduceOpBuilder(OpBuilder):
    def __init__(self):
        super(QuantAllReduceOpBuilder, self).__init__(
            "npu_quant_all_reduce", category="mc2"
        )

    def sources(self):
        """Path to C++ source code."""
        return ["csrc/mc2/quant_all_reduce.cpp"]

    def schema(self) -> str:
        """PyTorch operator signature."""
        return [
            "npu_quant_all_reduce(Tensor context, Tensor x, Tensor scales, int hccl_buffer_size, "
            "int world_size, *, str? reduce_op='sum', int? output_dtype=None, int? x_dtype=None, "
            "int? scales_dtype=None) -> Tensor"
        ]

    def register_meta(self):
        """
        Registers the Meta implementation (Shape/Dtype inference).
        Essential for Autograd and FakeTensor support.
        """

        @impl(get_as_library(), self.name, "Meta")
        def npu_quant_all_reduce_meta(
            context,
            x,
            scales,
            hccl_buffer_size,
            world_size,
            reduce_op="sum",
            output_dtype=None,
            x_dtype=None,
            scales_dtype=None,
        ):
            torch._check(
                x is not None,
                lambda: "x cannot be None, please input some value"
                + ops_error(ErrCode.TYPE),
            )
            torch._check(
                scales is not None,
                lambda: "scales cannot be None, please input some value"
                + ops_error(ErrCode.TYPE),
            )

            world_size_list = [2, 4, 8]
            torch._check(
                world_size in world_size_list,
                lambda: "world_size must be in "
                + str(world_size_list)
                + ", but actual value is: "
                + str(world_size)
                + ops_error(ErrCode.VALUE),
            )

            size = x.size()
            dtype = x.dtype
            if output_dtype is not None:
                dtype = TORCH_DTYPE_ENUM_VALUE_TO_SCALAR_TYPE_MAP[output_dtype]
            else:
                dtype = torch.bfloat16

            return torch.empty(size, dtype=dtype, device="meta")


# Instantiate the builder
quant_all_reduce_op_builder = QuantAllReduceOpBuilder()
quant_all_reduce_op_builder._ensure_initialized()


@impl(get_as_library(), quant_all_reduce_op_builder.name, "PrivateUse1")
def npu_quant_all_reduce(
    context,
    x,
    scales,
    hccl_buffer_size,
    world_size,
    reduce_op="sum",
    output_dtype=None,
    x_dtype=None,
    scales_dtype=None,
):
    """
    Dispatcher implementation for NPU.
    'PrivateUse1' is the dispatch key for custom NPU backends.
    """
    # Compiles/loads the .so file
    op_module = quant_all_reduce_op_builder.load()
    return op_module.npu_quant_all_reduce(
        context,
        x,
        scales,
        hccl_buffer_size,
        world_size,
        reduce_op,
        output_dtype,
        x_dtype,
        scales_dtype,
    )


class QuantAllReduceBuffer:
    def __init__(
        self,
        group,
        world_size: int,
        x: torch.Tensor,
        scales: torch.Tensor,
        op_name: str = "quant_all_reduce",
    ):
        self.group = group
        self.world_size = world_size
        self.rank_id = torch.distributed.get_rank(group)
        self.group_name = group._get_backend(torch.device("npu")).get_hccl_comm_name(
            self.rank_id, init_comm=False
        )
        required_ccl_buffer_size = _get_quant_all_reduce_ccl_buffer_size(
            x, scales, world_size
        )

        self._ctx_manager = CommContextManager(
            self.group_name,
            self.world_size,
            backend={
                "Ascend950": "channel",
            },
            opName=op_name,
            customCclBufferSize=required_ccl_buffer_size,
        )
        self.context = self._ctx_manager.create_context()
        self.hccl_buffer_size = self._ctx_manager.ccl_buffer_size

    def destroy(self):
        self._ctx_manager.destroy()


def get_quant_all_reduce_buffer(
    group, world_size, x, scales, op_name: str = "quant_all_reduce"
) -> QuantAllReduceBuffer:
    return QuantAllReduceBuffer(group, world_size, x, scales, op_name=op_name)


def _get_quant_all_reduce_ccl_buffer_size(
    x: torch.Tensor, scales: torch.Tensor, world_size: int
) -> int:
    def inline_align(value, base):
        return (value + base - 1) // base * base

    WIN_ADDR_ALIGN = 32
    MB_SIZE = 1024 * 1024
    HCCL_BUFFSIZE_FACTOR = 2

    x_data_size = inline_align(x.numel() * x.element_size(), WIN_ADDR_ALIGN)
    scales_data_size = inline_align(
        scales.numel() * scales.element_size(), WIN_ADDR_ALIGN
    )
    actual_win_size = (x_data_size + scales_data_size) * world_size + MB_SIZE
    return HCCL_BUFFSIZE_FACTOR * actual_win_size


_buffer_cache: dict[str, QuantAllReduceBuffer] = {}
_context_data_cache: dict[tuple, list[int]] = {}
_buffer_seq = 0


def _dynamo_disable(fn):
    # no-op兜底：dynamo不可用时保持原函数
    try:
        return torch._dynamo.disable(fn)
    except Exception:
        return fn


@_dynamo_disable
def _create_and_cache_buffer(
    hcom: str,
    group,
    world_size: int,
    x: torch.Tensor,
    scales: torch.Tensor,
    op_name: str = "quant_all_reduce",
) -> QuantAllReduceBuffer:
    """
    缓存未命中时创建buffer路径。必须对dynamo透明(graph break)，
    context.cpu().tolist()会把2048个int32展开成数千个符号量，
    若被trace，npu backend将在aten._local_scalar_dense上编译失败，
    并把整段字节码反汇编打进warning，浪费一次编译+重trace
    """
    cached = _buffer_cache.get(hcom)
    if cached is not None:
        # 旧buffer尺寸不够，同步后销毁重建
        torch.npu.synchronize()
        cached.destroy()
        _buffer_cache.pop(hcom, None)
    buffer = get_quant_all_reduce_buffer(group, world_size, x, scales, op_name=op_name)
    _buffer_cache[hcom] = buffer
    cache_key = (world_size, buffer.hccl_buffer_size)
    # 始终用最新buffer的context数据覆盖，避免命中已销毁buffer的过期context
    _context_data_cache[cache_key] = buffer.context.cpu().tolist()
    return buffer


def _get_or_create_buffer(
    hcom: str,
    group,
    world_size: int,
    x: torch.Tensor,
    scales: torch.Tensor,
    op_name: str = "quant_all_reduce",
) -> QuantAllReduceBuffer:
    """
    按hcom字符串查找缓存：
    1.命中且尺寸足够时直接复用(可被完整trace，走GE入图)
    2.未命中或现有buffer尺寸不足时新建(对dynamo透明，graph break后急执行)
    注意：destroy不会反注册HCCL channel/mem资源，频繁销毁重建会在HCCL内部残留失效注册，
    可能引发后续用例kernel软同步死锁，因此尺寸足够时必须复用而非重建
    """
    required_size = _get_quant_all_reduce_ccl_buffer_size(x, scales, world_size)
    cached = _buffer_cache.get(hcom)
    if (
        cached is not None
        and cached.world_size == world_size
        and cached.hccl_buffer_size >= required_size
    ):
        return cached
    return _create_and_cache_buffer(hcom, group, world_size, x, scales, op_name=op_name)


def _cleanup_all_buffers():
    for buf in _buffer_cache.values():
        try:
            buf.destroy()
        except Exception:
            pass
    _buffer_cache.clear()
    _context_data_cache.clear()


atexit.register(_cleanup_all_buffers)


def get_cached_context_data(world_size: int, hccl_buffer_size: int) -> list[int]:
    return _context_data_cache.get((world_size, hccl_buffer_size))


def quant_all_reduce(
    x: torch.Tensor,
    scales: torch.Tensor,
    hcom: str,
    world_size: int,
    *,
    reduce_op: Optional[str] = "sum",
    output_dtype: Optional[int] = None,
    x_dtype: Optional[int] = None,
    scales_dtype: Optional[int] = None,
):
    global _buffer_seq
    # 不再每次销毁重建buffer：优先复用缓存（_get_or_create_buffer），仅尺寸不足时才重建
    _buffer_seq += 1
    op_name = f"quant_all_reduce_{_buffer_seq}"
    group = torch.distributed.distributed_c10d._get_default_group()
    group_name = (
        hcom
        if isinstance(hcom, str)
        else group._get_backend(torch.device("npu")).get_hccl_comm_name(
            torch.distributed.get_rank(group), init_comm=False
        )
    )
    buffer = _get_or_create_buffer(
        group_name, group, world_size, x, scales, op_name=op_name
    )
    return torch.ops.cann_ops_transformer.npu_quant_all_reduce(
        buffer.context,
        x,
        scales,
        buffer.hccl_buffer_size,
        buffer.world_size,
        reduce_op=reduce_op,
        output_dtype=output_dtype,
        x_dtype=x_dtype,
        scales_dtype=scales_dtype,
    )
