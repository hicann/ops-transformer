# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from typing import Optional, Tuple
import torch
from torch.library import impl
from cann_ops_transformer.op_builder import OpBuilder, get_as_library


OP_NAME = "apply_rotary_pos_emb_grad"
ROTARY_MODE_HALF = "half"
LAYOUT_BSND = 1
LAYOUT_SBND = 2
LAYOUT_BNSD = 3
LAYOUT_TND = 4
SUPPORTED_LAYOUTS = (LAYOUT_BSND, LAYOUT_SBND, LAYOUT_BNSD, LAYOUT_TND)
SUPPORTED_DTYPES = (torch.float16, torch.float32, torch.bfloat16)


class ApplyRotaryPosEmbGradOpBuilder(OpBuilder):
    def __init__(self):
        super(ApplyRotaryPosEmbGradOpBuilder, self).__init__(
            OP_NAME, category="posembedding"
        )

    def sources(self):
        """Path to C++ source code."""
        return ["csrc/posembedding/apply_rotary_pos_emb_grad.cpp"]

    def schema(self) -> str:
        """PyTorch operator signature."""
        return (
            "apply_rotary_pos_emb_grad(Tensor grad_query_embed, Tensor grad_key_embed, Tensor cos, Tensor sin, "
            '*, Tensor? query=None, Tensor? key=None, str? rotary_mode="half", int? layout=1) -> '
            "(Tensor, Tensor, Tensor?, Tensor?)"
        )

    def register_meta(self):
        """
        Registers the Meta implementation (Shape/Dtype inference).
        Essential for Autograd and FakeTensor support.
        """

        @impl(get_as_library(), self.name, "Meta")
        def apply_rotary_pos_emb_grad_meta(
            grad_query_embed: torch.Tensor,
            grad_key_embed: torch.Tensor,
            cos: torch.Tensor,
            sin: torch.Tensor,
            *,
            query: Optional[torch.Tensor] = None,
            key: Optional[torch.Tensor] = None,
            rotary_mode: Optional[str] = ROTARY_MODE_HALF,
            layout: Optional[int] = LAYOUT_BSND,
        ):
            _check_inputs(
                grad_query_embed,
                grad_key_embed,
                cos,
                sin,
                query,
                key,
                rotary_mode,
                layout,
            )
            grad_query = torch.empty(
                grad_query_embed.shape, dtype=grad_query_embed.dtype, device="meta"
            )
            grad_key = torch.empty(
                grad_key_embed.shape, dtype=grad_key_embed.dtype, device="meta"
            )
            grad_cos = (
                torch.empty(cos.shape, dtype=cos.dtype, device="meta")
                if query is not None
                else None
            )
            grad_sin = (
                torch.empty(sin.shape, dtype=sin.dtype, device="meta")
                if query is not None
                else None
            )
            return grad_query, grad_key, grad_cos, grad_sin


def _check_dim_range(tensor: torch.Tensor, name: str):
    if tensor.dim() < 3 or tensor.dim() > 4:
        raise ValueError(
            f"apply_rotary_pos_emb_grad: {name} dim must be 3 or 4, but got {tensor.dim()}."
        )


def _check_same_dtype(tensor: torch.Tensor, reference: torch.Tensor, name: str):
    if tensor.dtype != reference.dtype:
        raise ValueError(
            f"apply_rotary_pos_emb_grad: {name} dtype must be same as "
            f"grad_query_embed, but got {tensor.dtype} vs {reference.dtype}."
        )


def _check_inputs(
    grad_query_embed: torch.Tensor,
    grad_key_embed: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    query: Optional[torch.Tensor],
    key: Optional[torch.Tensor],
    rotary_mode: Optional[str],
    layout: Optional[int],
):
    for name, tensor in (
        ("grad_query_embed", grad_query_embed),
        ("grad_key_embed", grad_key_embed),
        ("cos", cos),
        ("sin", sin),
    ):
        if tensor.numel() == 0:
            raise ValueError(f"apply_rotary_pos_emb_grad: {name} must not be empty.")
        _check_dim_range(tensor, name)

    if grad_query_embed.dtype not in SUPPORTED_DTYPES:
        raise ValueError(
            "apply_rotary_pos_emb_grad: dtype only supports float16, float32, and bfloat16."
        )
    _check_same_dtype(grad_key_embed, grad_query_embed, "grad_key_embed")
    _check_same_dtype(cos, grad_query_embed, "cos")
    _check_same_dtype(sin, grad_query_embed, "sin")

    mode = ROTARY_MODE_HALF if rotary_mode is None else rotary_mode
    if mode != ROTARY_MODE_HALF:
        raise ValueError(
            f"apply_rotary_pos_emb_grad: rotary_mode only supports 'half', got '{mode}'."
        )
    layout_value = LAYOUT_BSND if layout is None else layout
    if layout_value not in SUPPORTED_LAYOUTS:
        raise ValueError(
            "apply_rotary_pos_emb_grad: layout must be one of 1(BSND), "
            f"2(SBND), 3(BNSD), 4(TND), got {layout_value}."
        )
    if layout_value == LAYOUT_TND:
        if grad_query_embed.dim() != 3:
            raise ValueError(
                "apply_rotary_pos_emb_grad: TND(4) layout requires 3D inputs, "
                f"but got {grad_query_embed.dim()}D."
            )
    elif grad_query_embed.dim() != 4:
        raise ValueError(
            "apply_rotary_pos_emb_grad: BSND(1)/SBND(2) layout requires 4D "
            f"inputs, but got {grad_query_embed.dim()}D."
        )

    if (query is None) != (key is None):
        raise ValueError(
            "apply_rotary_pos_emb_grad: query and key must both be provided or both be None."
        )
    if query is not None:
        _check_dim_range(query, "query")
        _check_same_dtype(query, grad_query_embed, "query")
        if query.shape != grad_query_embed.shape:
            raise ValueError(
                "apply_rotary_pos_emb_grad: query shape must equal grad_query_embed shape."
            )
    if key is not None:
        _check_dim_range(key, "key")
        _check_same_dtype(key, grad_query_embed, "key")
        if key.shape != grad_key_embed.shape:
            raise ValueError(
                "apply_rotary_pos_emb_grad: key shape must equal grad_key_embed shape."
            )


apply_rotary_pos_emb_grad_op_builder = ApplyRotaryPosEmbGradOpBuilder()
apply_rotary_pos_emb_grad_op_builder.load()


@impl(get_as_library(), apply_rotary_pos_emb_grad_op_builder.name, "PrivateUse1")
def _apply_rotary_pos_emb_grad(
    grad_query_embed: torch.Tensor,
    grad_key_embed: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    query: Optional[torch.Tensor] = None,
    key: Optional[torch.Tensor] = None,
    rotary_mode: Optional[str] = ROTARY_MODE_HALF,
    layout: Optional[int] = LAYOUT_BSND,
):
    """
    Dispatcher implementation for NPU.
    'PrivateUse1' is the combine key for custom NPU backends.
    """
    _check_inputs(
        grad_query_embed,
        grad_key_embed,
        cos,
        sin,
        query,
        key,
        rotary_mode,
        layout,
    )
    op_module = apply_rotary_pos_emb_grad_op_builder.load()
    return op_module.apply_rotary_pos_emb_grad(
        grad_query_embed,
        grad_key_embed,
        cos,
        sin,
        query,
        key,
        rotary_mode,
        layout,
    )


def apply_rotary_pos_emb_grad(
    grad_query_embed: torch.Tensor,
    grad_key_embed: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    query: Optional[torch.Tensor] = None,
    key: Optional[torch.Tensor] = None,
    rotary_mode: Optional[str] = ROTARY_MODE_HALF,
    layout: Optional[int] = LAYOUT_BSND,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    """
    Computes gradients for apply_rotary_pos_emb on NPU.

    When both ``query`` and ``key`` are provided, the operator also computes
    ``grad_cos`` and ``grad_sin``.  When both are ``None``, the optional
    cosine and sine gradients are skipped and returned as ``None``.
    """
    return torch.ops.cann_ops_transformer.apply_rotary_pos_emb_grad(
        grad_query_embed,
        grad_key_embed,
        cos,
        sin,
        query=query,
        key=key,
        rotary_mode=rotary_mode,
        layout=layout,
    )
