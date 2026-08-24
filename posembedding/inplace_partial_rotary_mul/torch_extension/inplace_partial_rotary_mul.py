# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from typing import List, Optional
import torch
from torch.library import impl
from cann_ops_transformer.op_builder import OpBuilder, get_as_library


class InplacePartialRotaryMulFn(torch.autograd.Function):
    """Autograd wrapper: forward -> inplace_partial_rotary_mul,
    backward -> inplace_partial_rotary_mul_backward.

    Backward does NOT compute gradients for r1 (cos) and r2 (sin).
    These tensors are treated as constants in RoPE usage.
    """

    @staticmethod
    def forward(ctx, x, r1, r2, rotary_mode, partial_slice):
        ctx.save_for_backward(r1, r2)
        ctx.rotary_mode = rotary_mode
        ctx.partial_slice = partial_slice
        ctx.mark_dirty(x)
        torch.ops.cann_ops_transformer.inplace_partial_rotary_mul(
            x, r1, r2, rotary_mode=rotary_mode, partial_slice=partial_slice
        )
        return x

    @staticmethod
    def backward(ctx, grad_output):
        r1, r2 = ctx.saved_tensors
        if not grad_output.is_contiguous():
            grad_input = grad_output.contiguous()
        else:
            grad_input = grad_output
        torch.ops.cann_ops_transformer.inplace_partial_rotary_mul_backward(
            grad_input,
            r1,
            r2,
            rotary_mode=ctx.rotary_mode,
            partial_slice=ctx.partial_slice,
        )
        return grad_input, None, None, None, None


class InplacePartialRotaryMulOpBuilder(OpBuilder):
    def __init__(self):
        super(InplacePartialRotaryMulOpBuilder, self).__init__(
            "inplace_partial_rotary_mul", category="posembedding"
        )

    def sources(self):
        """Path to C++ source code."""
        return ["csrc/posembedding/inplace_partial_rotary_mul.cpp"]

    def schema(self) -> str:
        """PyTorch operator signature."""
        return (
            "inplace_partial_rotary_mul(Tensor(a!) x, Tensor r1, Tensor r2, *, "
            'str rotary_mode="interleave", int[2] partial_slice=[0, 0]) -> ()'
        )

    def register_meta(self):
        """
        Registers the Meta implementation.
        """

        @impl(get_as_library(), self.name, "Meta")
        def inplace_partial_rotary_mul_meta(
            x: torch.Tensor,
            r1: torch.Tensor,
            r2: torch.Tensor,
            *,
            rotary_mode: str = "interleave",
            partial_slice: Optional[List[int]] = None,
        ) -> None:
            partial_slice = [0, 0] if partial_slice is None else partial_slice
            if x.dim() != 4:
                raise ValueError(
                    f"Input tensor x's dim num should be 4, actual {x.dim()}."
                )
            if rotary_mode != "interleave":
                raise ValueError("rotary_mode only supports 'interleave'.")
            if len(partial_slice) != 2:
                raise ValueError("partial_slice must contain exactly two integers.")
            return


# Instantiate the builder
inplace_partial_rotary_mul_op_builder = InplacePartialRotaryMulOpBuilder()
inplace_partial_rotary_mul_op_builder._ensure_initialized()
# Pre-load the op module so that torch.compile / dynamo tracing does not
# encounter torch.utils.cpp_extension.load() at trace time (it is marked
# as "skipped" by the dynamo trace rules).
inplace_partial_rotary_mul_op_builder.load()


@impl(get_as_library(), inplace_partial_rotary_mul_op_builder.name, "PrivateUse1")
def _inplace_partial_rotary_mul(
    x: torch.Tensor,
    r1: torch.Tensor,
    r2: torch.Tensor,
    *,
    rotary_mode: str = "interleave",
    partial_slice: Optional[List[int]] = None,
) -> None:
    """
    dispatcher implementation for NPU.
    'PrivateUse1' is the combine key for custom NPU backends.
    """
    partial_slice = [0, 0] if partial_slice is None else partial_slice
    op_module = inplace_partial_rotary_mul_op_builder.load()
    op_module.inplace_partial_rotary_mul(x, r1, r2, rotary_mode, partial_slice)


def inplace_partial_rotary_mul(
    x: torch.Tensor,
    r1: torch.Tensor,
    r2: torch.Tensor,
    *,
    rotary_mode: str = "interleave",
    partial_slice: Optional[List[int]] = None,
) -> None:
    """
    Applies partial rotary position embedding to ``x`` in-place on NPU.

    Supports automatic differentiation when ``x.requires_grad`` is True:
    the backward pass will automatically call
    :func:`inplace_partial_rotary_mul_backward` to compute the gradient of
    ``x``.  Gradients for ``r1`` and ``r2`` are NOT computed (treated as
    constants, which is the standard RoPE usage).

    Args:
        x (Tensor): Input tensor to be modified in-place.
        r1 (Tensor): Cosine component tensor.
        r2 (Tensor): Sine component tensor.
        rotary_mode (str): Only ``"interleave"`` is currently supported.
            Defaults to ``"interleave"``.
        partial_slice (List[int]): Two-element slice range along the head
            dimension. Defaults to ``[0, 0]``.
    """
    partial_slice = [0, 0] if partial_slice is None else partial_slice

    # r1/r2 在 RoPE 场景中通常不需要梯度，此处仅按 x.requires_grad 判断
    if x.requires_grad:
        InplacePartialRotaryMulFn.apply(x, r1, r2, rotary_mode, partial_slice)
        return

    # 非 autograd 路径：直接走 backend
    torch.ops.cann_ops_transformer.inplace_partial_rotary_mul(
        x, r1, r2, rotary_mode=rotary_mode, partial_slice=partial_slice
    )
