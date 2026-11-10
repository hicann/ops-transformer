# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from typing import Optional, Tuple, Union

import torch
from torch.autograd.function import once_differentiable
from torch.library import impl
from cann_ops_transformer.op_builder import OpBuilder, get_as_library


class TextGatedResidualOpBuilder(OpBuilder):
    def __init__(self):
        super(TextGatedResidualOpBuilder, self).__init__(
            "text_gated_residual", category="mhc"
        )

    def sources(self):
        return ["csrc/mhc/text_gated_residual.cpp"]

    def schema(self) -> str:
        return (
            "text_gated_residual(Tensor x, Tensor w_down, Tensor w_up, *, "
            "Tensor? w_gate=None, int channels=4, Tensor? gamma=None, "
            "float eps=1e-6, bool need_backward=False) -> "
            "(Tensor, Tensor, Tensor, Tensor, Tensor)"
        )

    def register_meta(self):
        @impl(get_as_library(), self.name, "Meta")
        def text_gated_residual_meta(
            x,
            w_down,
            w_up,
            w_gate=None,
            channels=4,
            gamma=None,
            eps=1e-6,
            need_backward=False,
        ):
            torch._check(x.dim() == 3)
            torch._check(channels > 0)
            bsz, seq, dim = x.shape
            torch._check(dim % channels == 0)
            torch._check(x.dtype == torch.float16 or x.dtype == torch.bfloat16)
            if gamma is not None:
                torch._check(gamma.dim() == 1)
                torch._check(gamma.numel() == dim)
                torch._check(gamma.dtype == torch.float32)
            hidden = dim // channels
            mixed = x.new_empty((bsz, seq, hidden))
            inject = (
                x.new_empty((bsz, seq, channels))
                if w_gate is not None
                else x.new_empty((0,))
            )
            if need_backward:
                inv_rms = x.new_empty((bsz, seq, channels), dtype=torch.float32)
                down_pre_act = x.new_empty((bsz, seq, w_down.shape[0]))
                up_pre_act = x.new_empty((bsz, seq, dim))
            else:
                inv_rms = x.new_empty((0,), dtype=torch.float32)
                down_pre_act = x.new_empty((0,))
                up_pre_act = x.new_empty((0,))
            return mixed, inject, inv_rms, down_pre_act, up_pre_act


text_gated_residual_op_builder = TextGatedResidualOpBuilder()
text_gated_residual_op_builder._ensure_initialized()


@impl(get_as_library(), text_gated_residual_op_builder.name, "PrivateUse1")
def _text_gated_residual(
    x,
    w_down,
    w_up,
    w_gate=None,
    channels=4,
    gamma=None,
    eps=1e-6,
    need_backward=False,
):
    op_module = text_gated_residual_op_builder.load()
    return op_module.text_gated_residual(
        x, w_down, w_up, w_gate, channels, gamma, eps, need_backward
    )


def text_gated_residual(
    x: torch.Tensor,
    w_down: torch.Tensor,
    w_up: torch.Tensor,
    w_gate: Optional[torch.Tensor] = None,
    channels: int = 4,
    gamma: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
    """Fused forward; automatically save private caches and bind first-order backward."""
    needs_backward = torch.is_grad_enabled() and any(
        tensor is not None and tensor.requires_grad
        for tensor in (x, w_down, w_up, w_gate, gamma)
    )
    if needs_backward:
        mixed, inject = _TextGatedResidualFunction.apply(
            x, w_down, w_up, w_gate, gamma, channels, eps
        )
    else:
        mixed, inject, _, _, _ = _run_text_gated_residual(
            x, w_down, w_up, w_gate, gamma, channels, eps, False
        )
    return mixed if w_gate is None else (mixed, inject)


def _run_text_gated_residual(
    x, w_down, w_up, w_gate, gamma, channels, eps, need_backward
):
    return torch.ops.cann_ops_transformer.text_gated_residual(
        x,
        w_down,
        w_up,
        w_gate=w_gate,
        channels=channels,
        gamma=gamma,
        eps=eps,
        need_backward=need_backward,
    )


class _TextGatedResidualFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w_down, w_up, w_gate, gamma, channels, eps):
        use_combine = w_gate is not None
        mixed, inject, inv_rms, down_pre_act, up_pre_act = _run_text_gated_residual(
            x, w_down, w_up, w_gate, gamma, channels, eps, True
        )
        if not use_combine:
            inject = None
        ctx.channels, ctx.eps, ctx.use_combine = channels, eps, use_combine
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(
            x, w_down, w_up, w_gate, gamma, inv_rms, down_pre_act, inject, up_pre_act
        )
        return mixed, inject

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_mixed, grad_inject):
        x, wd, wu, wg, gamma, inv, down, inject, up = ctx.saved_tensors
        if grad_mixed is None:
            grad_mixed = x.new_zeros((*x.shape[:-1], x.shape[-1] // ctx.channels))
        if wg is not None and grad_inject is None:
            grad_inject = torch.zeros_like(inject)
        grads = _text_gated_residual_backward(
            grad_mixed,
            grad_inject,
            x,
            wd,
            wu,
            wg,
            gamma,
            inv,
            inject,
            down,
            up,
            ctx.channels,
        )
        inputs = (x, wd, wu, wg, gamma)
        grads = tuple(
            g if g is not None and t is not None and needed else None
            for g, t, needed in zip(grads, inputs, ctx.needs_input_grad[:5])
        )
        return (*grads, None, None)


@torch.library.custom_op(
    "cann_ops_transformer::_text_gated_residual_backward",
    mutates_args=(),
    device_types="npu",
    schema="(Tensor grad_mixed, Tensor? grad_inject, Tensor x, Tensor w_down, Tensor w_up, "
    "Tensor? w_gate, Tensor? gamma, Tensor inv_rms, Tensor? inject, Tensor down_pre_act, "
    "Tensor up, int channels) -> (Tensor, Tensor, Tensor, Tensor?, Tensor?)",
)
def _text_gated_residual_backward(
    grad_mixed: torch.Tensor,
    grad_inject: Optional[torch.Tensor],
    x: torch.Tensor,
    w_down: torch.Tensor,
    w_up: torch.Tensor,
    w_gate: Optional[torch.Tensor],
    gamma: Optional[torch.Tensor],
    inv_rms: torch.Tensor,
    inject: Optional[torch.Tensor],
    down_pre_act: torch.Tensor,
    up: torch.Tensor,
    channels: int,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    return text_gated_residual_op_builder.load()._text_gated_residual_backward(
        grad_mixed,
        grad_inject,
        x,
        w_down,
        w_up,
        w_gate,
        gamma,
        inv_rms,
        inject,
        down_pre_act,
        up,
        channels,
    )


@torch.library.register_fake("cann_ops_transformer::_text_gated_residual_backward")
def _text_gated_residual_backward_fake(
    grad_mixed,
    grad_inject,
    x,
    w_down,
    w_up,
    w_gate,
    gamma,
    inv_rms,
    inject,
    down_pre_act,
    up,
    channels,
):
    torch._check(x.dim() == 3)
    torch._check(channels > 0)
    torch._check(x.shape[-1] % channels == 0)
    torch._check(x.dtype in (torch.float16, torch.bfloat16))
    torch._check((grad_inject is None) == (w_gate is None))
    torch._check((inject is None) == (w_gate is None))
    return (
        torch.empty_like(x),
        torch.empty_like(w_down),
        torch.empty_like(w_up),
        torch.empty_like(w_gate) if w_gate is not None else None,
        torch.empty_like(gamma) if gamma is not None else None,
    )
