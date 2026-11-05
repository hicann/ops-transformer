# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from typing import Sequence

import torch
from torch.library import impl

from cann_ops_transformer.op_builder import OpBuilder, get_as_library


def _check_list_lengths(src, dst):
    if not src or len(src) != len(dst):
        raise ValueError("group_memcpy: src/dst must have the same nonzero list length")
    if len(src) > 2**31 - 1:
        raise RuntimeError("group_memcpy: too many pairs")


def _group_memcpy_meta(src, dst):
    _check_list_lengths(src, dst)
    dtype = src[0].dtype
    supported = (
        torch.int8,
        torch.uint8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.bool,
    )
    if dtype not in supported:
        raise RuntimeError("group_memcpy: unsupported dtype")
    elements = src[0].numel()
    for source, destination in zip(src, dst):
        for tensor in (source, destination):
            if tensor.device.type != "meta":
                raise RuntimeError("group_memcpy: expected Meta tensors")
            if tensor.layout != torch.strided or not tensor.is_contiguous():
                raise RuntimeError("group_memcpy: contiguous strided tensors required")
            if tensor.requires_grad:
                raise RuntimeError("group_memcpy: autograd is unsupported")
            if tensor.dtype != dtype:
                raise RuntimeError("group_memcpy: all dtypes must match")
            if tensor.numel() != elements:
                raise RuntimeError("group_memcpy: all copy byte counts must match")
        if source.shape != destination.shape:
            raise RuntimeError("group_memcpy: paired shapes must match")
    if elements > (2**63 - 1 - 64) // src[0].element_size():
        raise RuntimeError("group_memcpy: copy byte count overflows")
    for tensor in dst:
        if not tensor.is_inference():
            torch.autograd.graph.increment_version(tensor)


class GroupMemcpyOpBuilder(OpBuilder):
    def __init__(self):
        super().__init__("group_memcpy", category="attention")

    def sources(self):
        return ["csrc/attention/group_memcpy.cpp"]

    def schema(self) -> str:
        return "group_memcpy(Tensor[] src, Tensor(a!)[] dst) -> ()"

    def register_meta(self):
        impl(get_as_library(), self.name, "Meta")(_group_memcpy_meta)


_group_memcpy_op_builder = GroupMemcpyOpBuilder()
_group_memcpy_op_builder._ensure_initialized()


@impl(get_as_library(), _group_memcpy_op_builder.name, "PrivateUse1")
@impl(get_as_library(), _group_memcpy_op_builder.name, "AutogradPrivateUse1")
def _group_memcpy(src, dst):
    _check_list_lengths(src, dst)
    op_module = _group_memcpy_op_builder.load()
    return op_module.group_memcpy(src, dst)


def group_memcpy(src: Sequence[torch.Tensor], dst: Sequence[torch.Tensor]) -> None:
    """Copy equal-sized contiguous NPU tensor pairs into preallocated destinations.

    Args:
        src: Nonempty source sequence of contiguous ND tensors on one NPU.
            Supported dtypes are int8, uint8, int16, int32, int64, float16,
            bfloat16, float32 and bool. All tensors have the same element count.
        dst: Equally long destination sequence. Each shape matches its source;
            dtype, element count and device match all sources. Destinations
            must not overlap any source or another destination.

    Returns:
        None. Updates dst in place on the current NPU stream, preserving raw
        bytes. Nonempty zero-element lists are a no-op. Autograd is unsupported.
        Meta inputs are validated without compiling or executing the NPU code.
    """
    _check_list_lengths(src, dst)
    return torch.ops.cann_ops_transformer.group_memcpy.default(list(src), list(dst))
