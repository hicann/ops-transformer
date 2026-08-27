# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import os
import warnings
from typing import List, Optional

import torch
from cann_ops_transformer.op_builder import OpBuilder, get_as_library
from torch.library import impl


class _MoeFinalizeRoutingOpBuilder(OpBuilder):
    def __init__(self):
        super(_MoeFinalizeRoutingOpBuilder, self).__init__(
            "moe_finalize_routing", category="moe"
        )

    def sources(self):
        return ["csrc/moe/moe_finalize_routing.cpp"]

    def include_paths(self):
        paths = super().include_paths()
        paths.append(
            os.path.abspath(
                os.path.join(
                    self._package_path,
                    "..",
                    "..",
                    "moe",
                    "moe_finalize_routing_v2",
                    "op_host",
                    "op_api",
                )
            )
        )
        return paths

    def schema(self) -> str:
        return (
            "moe_finalize_routing(Tensor expanded_x, "
            "Tensor expanded_row_idx, "
            "Tensor? x1, Tensor? x2, Tensor? bias, Tensor? scales, "
            "Tensor? expert_idx, Tensor? x, Tensor? alpha1, Tensor? alpha2, Tensor? v, "
            "int? drop_pad_mode=0, "
            "int[]? zero_expert_range=None, int[]? copy_expert_range=None, "
            "int[]? constant_expert_range=None, int? k=1) -> Tensor"
        )

    def register_meta(self):
        @impl(get_as_library(), self.name, "Meta")
        def moe_finalize_routing_meta(
            expanded_x,
            expanded_row_idx,
            x1=None,
            x2=None,
            bias=None,
            scales=None,
            expert_idx=None,
            x=None,
            alpha1=None,
            alpha2=None,
            v=None,
            drop_pad_mode=0,
            zero_expert_range=None,
            copy_expert_range=None,
            constant_expert_range=None,
            k=1,
        ):
            k_val = k if k is not None else 1
            mode = drop_pad_mode if drop_pad_mode is not None else 0
            dimm = expanded_row_idx.size(0)
            if scales is not None:
                dimm = scales.size(0)
            elif k_val > 0:
                dimm = dimm // k_val
            if mode == 1 or mode == 3:
                dimn = expanded_x.size(2)
            else:
                dimn = expanded_x.size(1)
            return expanded_x.new_empty((dimm, dimn))


_moe_finalize_routing_builder = _MoeFinalizeRoutingOpBuilder()
_moe_finalize_routing_builder._ensure_initialized()


@impl(get_as_library(), _moe_finalize_routing_builder.name, "PrivateUse1")
def _moe_finalize_routing(
    expanded_x,
    expanded_row_idx,
    x1=None,
    x2=None,
    bias=None,
    scales=None,
    expert_idx=None,
    x=None,
    alpha1=None,
    alpha2=None,
    v=None,
    drop_pad_mode=0,
    zero_expert_range=None,
    copy_expert_range=None,
    constant_expert_range=None,
    k=1,
):
    _op_module = _moe_finalize_routing_builder.load()
    return _op_module.moe_finalize_routing(
        expanded_x,
        expanded_row_idx,
        x1,
        x2,
        bias,
        scales,
        expert_idx,
        x,
        alpha1,
        alpha2,
        v,
        drop_pad_mode,
        zero_expert_range,
        copy_expert_range,
        constant_expert_range,
        k,
    )


def _has_value(t):
    if t is None:
        return False
    if isinstance(t, torch.Tensor):
        return t.numel() > 0
    return True


def _is_valid_expert_range(r):
    """A range like [start, end) is valid (non-empty) only when start >= 0 and end > start.
    The sentinel [-1, -1] means 'no such experts' and is treated as invalid."""
    if r is None or len(r) < 2:
        return False
    return r[0] >= 0 and r[1] > r[0]


_NUM_FORWARD_INPUTS = 16


def _check_backward_supported(ctx):
    """Raise NotImplementedError if the V4-specific or unsupported drop_pad_mode path is taken."""
    if ctx.has_v4_extras:
        raise NotImplementedError(
            "moe_finalize_routing autograd is not supported when V4-specific "
            "inputs (x, alpha1, alpha2, v) or zero_expert_range are provided. "
            "The aclnnMoeFinalizeRoutingV2Grad op only handles the regular-expert case."
        )
    if ctx.drop_pad_mode not in (0, 1):
        raise NotImplementedError(
            "moe_finalize_routing autograd is only supported for drop_pad_mode 0 or 1 "
            "(column-major expanded_row_idx layout). The backward op "
            "(aclnnMoeFinalizeRoutingV2Grad) does not support drop_pad_mode "
            f"{ctx.drop_pad_mode}."
        )


def _transpose_row_idx(expanded_row_idx, top_k, row_num):
    """Convert expanded_row_idx from forward (K, R) layout to grad op (R, K) layout."""
    if top_k > 1:
        return expanded_row_idx.reshape(top_k, row_num).t().contiguous().reshape(-1)
    return expanded_row_idx


def _synthesize_scales(scales, top_k, row_num, grad_out):
    """Return effective scales, synthesizing a ones tensor when scales is None and K > 1."""
    if scales is None and top_k > 1:
        return torch.ones(row_num, top_k, dtype=grad_out.dtype, device=grad_out.device)
    return scales


def _infer_grad_shape(expanded_x, drop_pad_mode):
    """Infer (active_num, expert_num, expert_capacity) for the grad op from expanded_x shape."""
    if drop_pad_mode == 1:
        return 0, expanded_x.shape[0], expanded_x.shape[1]
    return expanded_x.shape[0], 0, 0


def _build_grad_return_tuple(grad_expanded_x, grad_scales):
    """Build the 16-element backward return tuple (only positions 0 and 5 are non-None)."""
    grads = [None] * _NUM_FORWARD_INPUTS
    grads[0] = grad_expanded_x
    grads[5] = grad_scales
    return tuple(grads)


class MoeFinalizeRoutingFn(torch.autograd.Function):
    """Autograd binding: forward -> aclnnMoeFinalizeRoutingV4,
    backward -> aclnnMoeFinalizeRoutingV2Grad.

    The V2Grad op only handles the regular-expert case.  Autograd is therefore
    only supported when V4 reduces to V2 — i.e. none of the V4-specific inputs
    (``x``, ``alpha1``, ``alpha2``, ``v``) are provided and ``zero_expert_range``
    is empty.  When V4-specific features are used, backward raises
    ``NotImplementedError``.

    Additionally, the grad op only supports ``drop_pad_mode`` 0 or 1
    (column-major ``expanded_row_idx`` layout); ``drop_pad_mode`` 2 or 3
    (row-major layout) is not supported.

    The V2Grad op only outputs ``grad_expanded_x`` and ``grad_scales``.
    Gradients for ``x1``, ``x2``, and ``bias`` are not computed (returned as
    ``None``).

    Note: the forward op indexes ``expanded_row_idx`` with ``(K, R)`` layout
    (``idx[k*R + row]``) when ``drop_pad_mode`` is 0 or 1, while the grad op
    expects ``(R, K)`` layout (``idx[row*K + k]``).  Backward transposes the
    index layout accordingly.
    """

    @staticmethod
    def forward(
        ctx,
        expanded_x,
        expanded_row_idx,
        x1,
        x2,
        bias,
        scales,
        expert_idx,
        x,
        alpha1,
        alpha2,
        v,
        drop_pad_mode,
        zero_expert_range,
        copy_expert_range,
        constant_expert_range,
        k,
    ):
        ctx.drop_pad_mode = drop_pad_mode if drop_pad_mode is not None else 0
        ctx.k = k if k is not None else 1
        ctx.has_scales = _has_value(scales)
        ctx.has_bias = _has_value(bias)
        ctx.has_expert_idx = _has_value(expert_idx)
        ctx.has_v4_extras = (
            _has_value(x)
            or _has_value(alpha1)
            or _has_value(alpha2)
            or _has_value(v)
            or _is_valid_expert_range(zero_expert_range)
            or _is_valid_expert_range(copy_expert_range)
            or _is_valid_expert_range(constant_expert_range)
        )
        ctx.save_for_backward(
            expanded_x,
            expanded_row_idx,
            scales if ctx.has_scales else expanded_x,
            bias if ctx.has_bias else expanded_x,
            expert_idx if ctx.has_expert_idx else expanded_row_idx,
        )
        return torch.ops.cann_ops_transformer.moe_finalize_routing(
            expanded_x,
            expanded_row_idx,
            x1,
            x2,
            bias,
            scales,
            expert_idx,
            x,
            alpha1,
            alpha2,
            v,
            drop_pad_mode,
            zero_expert_range,
            copy_expert_range,
            constant_expert_range,
            k,
        )

    @staticmethod
    def backward(ctx, grad_out):
        _check_backward_supported(ctx)
        needs = ctx.needs_input_grad
        if needs[2] or needs[3] or needs[4]:
            warnings.warn(
                "Gradients for x1/x2/bias are not computed by aclnnMoeFinalizeRoutingV2Grad "
                "and will be None. If x1/x2 require gradients, use external residual "
                "addition instead of passing them as forward inputs.",
                stacklevel=1,
            )
        if not needs[0] and not (ctx.has_scales and needs[5]):
            return _build_grad_return_tuple(None, None)

        expanded_x, expanded_row_idx, scales, bias, expert_idx = ctx.saved_tensors
        scales = scales if ctx.has_scales else None
        bias = bias if ctx.has_bias else None
        expert_idx = expert_idx if ctx.has_expert_idx else None
        drop_pad_mode = ctx.drop_pad_mode

        grad_out = grad_out.contiguous()
        row_num = grad_out.shape[0]
        top_k = scales.shape[1] if ctx.has_scales else ctx.k

        grad_row_idx = _transpose_row_idx(expanded_row_idx, top_k, row_num)
        eff_scales = _synthesize_scales(scales, top_k, row_num, grad_out)
        grad_active_num, grad_expert_num, grad_expert_capacity = _infer_grad_shape(
            expanded_x, drop_pad_mode
        )

        grad_expanded_x, grad_scales = (
            torch.ops.cann_ops_transformer.moe_finalize_routing_grad(
                grad_out,
                grad_row_idx,
                expanded_x,
                eff_scales,
                expert_idx,
                bias,
                drop_pad_mode,
                grad_active_num,
                grad_expert_num,
                grad_expert_capacity,
            )
        )

        grad_expanded_x = grad_expanded_x if needs[0] else None
        grad_scales = grad_scales if ctx.has_scales and needs[5] else None
        return _build_grad_return_tuple(grad_expanded_x, grad_scales)


def moe_finalize_routing(
    expanded_x: torch.Tensor,
    expanded_row_idx: torch.Tensor,
    x1: Optional[torch.Tensor] = None,
    x2: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    scales: Optional[torch.Tensor] = None,
    expert_idx: Optional[torch.Tensor] = None,
    x: Optional[torch.Tensor] = None,
    alpha1: Optional[torch.Tensor] = None,
    alpha2: Optional[torch.Tensor] = None,
    v: Optional[torch.Tensor] = None,
    drop_pad_mode: Optional[int] = 0,
    zero_expert_range: Optional[List[int]] = None,
    copy_expert_range: Optional[List[int]] = None,
    constant_expert_range: Optional[List[int]] = None,
    k: Optional[int] = 1,
) -> torch.Tensor:
    needs_grad = (
        expanded_x.requires_grad
        or (x1 is not None and x1.requires_grad)
        or (x2 is not None and x2.requires_grad)
        or (bias is not None and bias.requires_grad)
        or (scales is not None and scales.requires_grad)
        or (x is not None and x.requires_grad)
        or (alpha1 is not None and alpha1.requires_grad)
        or (alpha2 is not None and alpha2.requires_grad)
        or (v is not None and v.requires_grad)
    )
    if needs_grad:
        return MoeFinalizeRoutingFn.apply(
            expanded_x,
            expanded_row_idx,
            x1,
            x2,
            bias,
            scales,
            expert_idx,
            x,
            alpha1,
            alpha2,
            v,
            drop_pad_mode,
            zero_expert_range,
            copy_expert_range,
            constant_expert_range,
            k,
        )
    return torch.ops.cann_ops_transformer.moe_finalize_routing(
        expanded_x,
        expanded_row_idx,
        x1,
        x2,
        bias,
        scales,
        expert_idx,
        x,
        alpha1,
        alpha2,
        v,
        drop_pad_mode,
        zero_expert_range,
        copy_expert_range,
        constant_expert_range,
        k,
    )
