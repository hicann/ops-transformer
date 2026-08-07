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
from cann_ops_transformer.op_builder.builder import OpBuilder
from cann_ops_transformer.op_builder.builder import AS_LIBRARY


class InplacePartialRotaryMulBackwardOpBuilder(OpBuilder):
    def __init__(self):
        super(InplacePartialRotaryMulBackwardOpBuilder, self).__init__(
            "inplace_partial_rotary_mul_backward"
        )

    def sources(self):
        """Path to C++ source code."""
        return ["ops/csrc/inplace_partial_rotary_mul_backward.cpp"]

    def schema(self) -> str:
        """PyTorch operator signature."""
        return (
            "inplace_partial_rotary_mul_backward(Tensor(a!) grad_output, Tensor r1, Tensor r2, *, "
            'str rotary_mode="interleave", int[2] partial_slice=[0, 0]) -> ()'
        )

    def register_meta(self):
        """
        Registers the Meta implementation.
        """

        @impl(AS_LIBRARY, self.name, "Meta")
        def inplace_partial_rotary_mul_backward_meta(
            grad_output: torch.Tensor,
            r1: torch.Tensor,
            r2: torch.Tensor,
            *,
            rotary_mode: str = "interleave",
            partial_slice: Optional[List[int]] = None,
        ) -> None:
            partial_slice = [0, 0] if partial_slice is None else partial_slice
            if grad_output.dim() != 4:
                raise ValueError(
                    f"Input tensor grad_output's dim num should be 4, actual {grad_output.dim()}."
                )
            if rotary_mode != "interleave":
                raise ValueError("rotary_mode only supports 'interleave'.")
            if len(partial_slice) != 2:
                raise ValueError("partial_slice must contain exactly two integers.")
            start, end = partial_slice
            if start < 0 or end < 0:
                raise ValueError(
                    f"partial_slice start and end must be >= 0, got start={start}, end={end}."
                )
            if start > end:
                raise ValueError(
                    f"partial_slice start must be <= end, got start={start}, end={end}."
                )
            if end > grad_output.shape[-1]:
                raise ValueError(
                    f"partial_slice end must be <= D ({grad_output.shape[-1]}), got end={end}."
                )
            if not grad_output.is_contiguous():
                raise ValueError("grad_output must be a contiguous tensor.")
            if not r1.is_contiguous():
                raise ValueError("r1 must be a contiguous tensor.")
            if not r2.is_contiguous():
                raise ValueError("r2 must be a contiguous tensor.")
            return


# Instantiate the builder
inplace_partial_rotary_mul_backward_op_builder = (
    InplacePartialRotaryMulBackwardOpBuilder()
)


@impl(AS_LIBRARY, inplace_partial_rotary_mul_backward_op_builder.name, "PrivateUse1")
def _inplace_partial_rotary_mul_backward(
    grad_output: torch.Tensor,
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
    op_module = inplace_partial_rotary_mul_backward_op_builder.load()
    op_module.inplace_partial_rotary_mul_backward(
        grad_output, r1, r2, rotary_mode, partial_slice
    )


def inplace_partial_rotary_mul_backward(
    grad_output: torch.Tensor,
    r1: torch.Tensor,
    r2: torch.Tensor,
    *,
    rotary_mode: str = "interleave",
    partial_slice: Optional[List[int]] = None,
) -> None:
    """
    Applies gradient of partial rotary position embedding to ``grad_output``
    in-place on NPU.

    This computes the gradient of :func:`inplace_partial_rotary_mul` with
    respect to the input ``x``.  The result is written back into
    ``grad_output`` in-place — elements within ``partial_slice`` are
    replaced by the RoPE gradient, while elements outside the slice are
    left unchanged.

    Args:
        grad_output (Tensor): Gradient of the loss w.r.t. the output of
            the forward pass (i.e. the modified ``x``).
            **Modified in-place** to become the gradient w.r.t. the
            original input ``x``.
        r1 (Tensor): Cosine component tensor (same as forward).
        r2 (Tensor): Sine component tensor (same as forward).
        rotary_mode (str): Only ``"interleave"`` is currently supported.
            Defaults to ``"interleave"``.
        partial_slice (List[int]): Two-element slice range along the head
            dimension. Defaults to ``[0, 0]``.
    """
    partial_slice = [0, 0] if partial_slice is None else partial_slice
    return torch.ops.cann_ops_transformer.inplace_partial_rotary_mul_backward(
        grad_output,
        r1,
        r2,
        rotary_mode=rotary_mode,
        partial_slice=partial_slice,
    )
