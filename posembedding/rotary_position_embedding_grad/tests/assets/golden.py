#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""Golden reference for rotary_position_embedding_grad (torch).

Implements the formulas in docs/aclnnRotaryPositionEmbeddingGrad.md for all
four modes (0=half, 1=interleave, 2=quarter, 3=interleave-half):

    dx   = rotary_grad(dy, cos, sin)          (broadcast cos/sin to dy)
    dcos = sum(dy * x, dims)                  (dims = broadcast axes)
    dsin = sum(dy * rotary_half(x), dims)

Golden computes in the input dtype (fp32 aligned with the kernel); for
fp16/bf16 cases run ttk with `--golden-mode Promote` to let the framework
promote inputs before golden.
"""

__spec__ = {
    "rotary_position_embedding_grad": "RotaryPositionEmbeddingGradTestSpec",
    "aclnnRotaryPositionEmbeddingGrad": "AclnnRotaryPositionEmbeddingGradTestSpec",
}

import numpy as np
import torch


def _broadcast_dims(dy, cos):
    """Axes of dy's shape over which cos/sin are broadcast (cos dim == 1).

    Note: a zero-sized dy axis still counts (sum over an empty axis yields zeros,
    matching dcos/dsin semantics for empty inputs)."""
    ndim = dy.dim()
    cos_shape = [1] * (ndim - cos.dim()) + list(cos.shape)
    return [i for i in range(ndim) if cos_shape[i] == 1 and dy.shape[i] != 1]


def _chunk2(t):
    return t.chunk(2, dim=-1)


def _chunk4(t):
    return t.chunk(4, dim=-1)


def _rotary_grad_dx(dy, cos, sin, mode):
    if mode == 0:  # half
        dy1, dy2 = _chunk2(dy)
        cos1, cos2 = _chunk2(cos)
        sin1, sin2 = _chunk2(sin)
        return torch.cat((cos1 * dy1 + sin2 * dy2, cos2 * dy2 - sin1 * dy1), dim=-1)
    if mode == 1:  # interleave
        dy1, dy2 = dy[..., ::2], dy[..., 1::2]
        cos1, cos2 = cos[..., ::2], cos[..., 1::2]
        sin1, sin2 = sin[..., ::2], sin[..., 1::2]
        return torch.stack(
            (cos1 * dy1 + sin2 * dy2, cos2 * dy2 - sin1 * dy1), dim=-1
        ).reshape(dy.shape)
    if mode == 2:  # quarter
        dy1, dy2, dy3, dy4 = _chunk4(dy)
        cos1, cos2, cos3, cos4 = _chunk4(cos)
        sin1, sin2, sin3, sin4 = _chunk4(sin)
        return torch.cat(
            (
                cos1 * dy1 + sin2 * dy2,
                cos2 * dy2 - sin1 * dy1,
                cos3 * dy3 + sin4 * dy4,
                cos4 * dy4 - sin3 * dy3,
            ),
            dim=-1,
        )
    if mode == 3:  # interleave-half
        dy1, dy2 = _chunk2(dy)
        cos1, cos2 = _chunk2(cos)
        sin1, sin2 = _chunk2(sin)
        return torch.stack(
            (cos1 * dy1 + sin2 * dy2, cos2 * dy2 - sin1 * dy1), dim=-1
        ).reshape(dy.shape)
    raise ValueError(f"unsupported mode: {mode}")


def _rotated_x(x, mode):
    """The x-derived factor inside dsin (mode 3 uses it for both dcos/dsin)."""
    if mode in (0, 3):  # half / interleave-half
        x1, x2 = _chunk2(x) if mode == 0 else (x[..., ::2], x[..., 1::2])
        if mode == 3:
            return torch.cat((x1, x2), dim=-1), torch.cat((-x2, x1), dim=-1)
        return None, torch.cat((-x2, x1), dim=-1)
    if mode == 1:  # interleave
        x1, x2 = x[..., ::2], x[..., 1::2]
        return None, torch.stack((-x2, x1), dim=-1).reshape(x.shape)
    if mode == 2:  # quarter
        x1, x2, x3, x4 = _chunk4(x)
        return None, torch.cat((-x2, x1, -x4, x3), dim=-1)
    raise ValueError(f"unsupported mode: {mode}")


def _golden_impl(dy, cos, sin, x, mode):
    """Shared torch implementation; returns [dx] or [dx, dcos, dsin]."""
    mode = int(mode)
    dx = _rotary_grad_dx(dy, cos, sin, mode)
    if x is None:
        return [dx]

    dims = _broadcast_dims(dy, cos)
    dcos_full = dy * x
    dcos_factor, dsin_factor = _rotated_x(x, mode)
    if mode == 3:  # interleave-half: dcos sums dy * cat(x1, x2)
        dcos_full = dy * dcos_factor
    if dims:
        dcos = dcos_full.sum(dim=dims, keepdim=True)
        dsin = (dy * dsin_factor).sum(dim=dims, keepdim=True)
    else:  # cos/sin shape == dy shape: no reduction (torch sum(dim=[]) sums ALL dims!)
        dcos = dcos_full
        dsin = dy * dsin_factor
    return [dx, dcos, dsin]


class RotaryPositionEmbeddingGradTestSpec:
    """RotaryPositionEmbeddingGrad 测试规范（kernel 流程，numpy 入参）

    Parameters follow rotary_position_embedding_grad_def.cpp: dy, cos, sin, x + attr mode.
    """

    def golden(dy, cos, sin, x=None, mode=0, **kwargs):
        tensors = [torch.from_numpy(np.ascontiguousarray(t)) for t in (dy, cos, sin)]
        x_t = torch.from_numpy(np.ascontiguousarray(x)) if x is not None else None
        results = _golden_impl(*tensors, x_t, mode)
        return [r.numpy() for r in results]

    tolerance = {"float32": {"standard": "stat_rel_err"}}


class AclnnRotaryPositionEmbeddingGradTestSpec:
    """RotaryPositionEmbeddingGrad 测试规范（aclnn 流程，torch 入参）

    Parameters follow aclnnRotaryPositionEmbeddingGradGetWorkspaceSize
    (without workspaceSize & executor); all are passed positionally.
    """

    def golden(
        dy,
        cos,
        sin,
        xOptional=None,
        mode=0,
        dxOut=None,
        dcosOut=None,
        dsinOut=None,
        **kwargs,
    ):
        return _golden_impl(dy, cos, sin, xOptional, mode)

    tolerance = {"float32": {"standard": "stat_rel_err"}}
