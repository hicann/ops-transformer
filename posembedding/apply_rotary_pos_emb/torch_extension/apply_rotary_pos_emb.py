# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from typing import Tuple
import torch
from torch.library import impl
from cann_ops_transformer.op_builder import OpBuilder, get_as_library


OP_NAME = "apply_rotary_pos_emb"
ROTARY_MODE_HALF_STR = "half"
ROTARY_MODE_QUARTER_STR = "quarter"
ROTARY_MODE_INTERLEAVE_STR = "interleave"
LAYOUT_BSND_STR = "BSND"
LAYOUT_BSH_STR = "BSH"
LAYOUT_SBND_STR = "SBND"
LAYOUT_BNSD_STR = "BNSD"
LAYOUT_TND_STR = "TND"
GRAD_LAYOUT_BSND = 1
GRAD_LAYOUT_SBND = 2
GRAD_LAYOUT_BNSD = 3
GRAD_LAYOUT_TND = 4
SUPPORTED_LAYOUTS = (
    LAYOUT_BSND_STR,
    LAYOUT_BSH_STR,
    LAYOUT_SBND_STR,
    LAYOUT_BNSD_STR,
    LAYOUT_TND_STR,
)
SUPPORTED_ROTARY_MODES = (
    ROTARY_MODE_HALF_STR,
    ROTARY_MODE_QUARTER_STR,
    ROTARY_MODE_INTERLEAVE_STR,
)
GRAD_LAYOUT_BY_LAYOUT = {
    LAYOUT_BSND_STR: GRAD_LAYOUT_BSND,
    LAYOUT_BSH_STR: GRAD_LAYOUT_BSND,
    LAYOUT_SBND_STR: GRAD_LAYOUT_SBND,
    LAYOUT_BNSD_STR: GRAD_LAYOUT_BNSD,
    LAYOUT_TND_STR: GRAD_LAYOUT_TND,
}


def _check_layout(layout: str):
    if layout not in SUPPORTED_LAYOUTS:
        raise ValueError(
            "apply_rotary_pos_emb: layout must be one of BSND/BSH/SBND/BNSD/TND, "
            f"got {layout}."
        )


def _check_rotary_mode(rotary_mode: str):
    if rotary_mode not in SUPPORTED_ROTARY_MODES:
        raise ValueError(
            "apply_rotary_pos_emb: rotary_mode should be half/quarter/interleave, "
            f"but got {rotary_mode}."
        )


def _check_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    layout: str,
    rotary_mode: str,
):
    _check_layout(layout)
    _check_rotary_mode(rotary_mode)
    for name, tensor in (
        ("query", query),
        ("key", key),
        ("cos", cos),
        ("sin", sin),
    ):
        if tensor.numel() == 0:
            raise ValueError(f"apply_rotary_pos_emb: {name} must not be empty.")


class ApplyRotaryPosEmbFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, cos, sin, layout, rotary_mode):
        _check_inputs(query, key, cos, sin, layout, rotary_mode)
        op_module = apply_rotary_pos_emb_op_builder.load()
        query_emb, key_emb = op_module.apply_rotary_pos_emb(
            query, key, cos, sin, layout, rotary_mode
        )
        ctx.save_for_backward(query, key, cos, sin)
        ctx.layout = layout
        ctx.needs_query_grad = ctx.needs_input_grad[0]
        ctx.needs_key_grad = ctx.needs_input_grad[1]
        ctx.needs_cos_grad = ctx.needs_input_grad[2]
        ctx.needs_sin_grad = ctx.needs_input_grad[3]
        return query_emb, key_emb

    @staticmethod
    def backward(ctx, grad_query_embed, grad_key_embed):
        query, key, cos, sin = ctx.saved_tensors
        if grad_query_embed is None:
            grad_query_embed = torch.zeros_like(query)
        if grad_key_embed is None:
            grad_key_embed = torch.zeros_like(key)

        need_cos_sin_grad = ctx.needs_cos_grad or ctx.needs_sin_grad
        grad_query, grad_key, grad_cos, grad_sin = (
            torch.ops.cann_ops_transformer.apply_rotary_pos_emb_grad(
                grad_query_embed,
                grad_key_embed,
                cos,
                sin,
                query=query if need_cos_sin_grad else None,
                key=key if need_cos_sin_grad else None,
                rotary_mode=ROTARY_MODE_HALF_STR,
                layout=GRAD_LAYOUT_BY_LAYOUT[ctx.layout],
            )
        )

        return (
            grad_query if ctx.needs_query_grad else None,
            grad_key if ctx.needs_key_grad else None,
            grad_cos if ctx.needs_cos_grad else None,
            grad_sin if ctx.needs_sin_grad else None,
            None,
            None,
        )


class ApplyRotaryPosEmbOpBuilder(OpBuilder):
    def __init__(self):
        super(ApplyRotaryPosEmbOpBuilder, self).__init__(
            OP_NAME, category="posembedding"
        )

    def sources(self):
        return ["csrc/posembedding/apply_rotary_pos_emb.cpp"]

    def schema(self) -> str:
        return (
            "apply_rotary_pos_emb(Tensor query, Tensor key, Tensor cos, Tensor sin, "
            "str layout, str rotary_mode) -> (Tensor, Tensor)"
        )

    def register_meta(self):
        @impl(get_as_library(), self.name, "Meta")
        def apply_rotary_pos_emb_meta(query, key, cos, sin, layout, rotary_mode):
            _check_inputs(query, key, cos, sin, layout, rotary_mode)
            return torch.empty_like(query), torch.empty_like(key)


apply_rotary_pos_emb_op_builder = ApplyRotaryPosEmbOpBuilder()
apply_rotary_pos_emb_op_builder.load()


@impl(get_as_library(), apply_rotary_pos_emb_op_builder.name, "PrivateUse1")
def _apply_rotary_pos_emb(query, key, cos, sin, layout, rotary_mode):
    _check_inputs(query, key, cos, sin, layout, rotary_mode)
    op_module = apply_rotary_pos_emb_op_builder.load()
    return op_module.apply_rotary_pos_emb(query, key, cos, sin, layout, rotary_mode)


def apply_rotary_pos_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    layout: str = LAYOUT_BSND_STR,
    rotary_mode: str = ROTARY_MODE_HALF_STR,
) -> Tuple[torch.Tensor, torch.Tensor]:
    _check_inputs(query, key, cos, sin, layout, rotary_mode)
    needs_grad = torch.is_grad_enabled() and (
        query.requires_grad
        or key.requires_grad
        or cos.requires_grad
        or sin.requires_grad
    )
    if needs_grad:
        if rotary_mode != ROTARY_MODE_HALF_STR:
            raise ValueError(
                "apply_rotary_pos_emb: autograd only supports rotary_mode='half'."
            )
        return ApplyRotaryPosEmbFunction.apply(
            query, key, cos, sin, layout, rotary_mode
        )
    return torch.ops.cann_ops_transformer.apply_rotary_pos_emb(
        query, key, cos, sin, layout, rotary_mode
    )
