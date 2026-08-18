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
"""Golden reference for rotary_position_embedding (torch).

Implements the formulas in docs/aclnnRotaryPositionEmbedding.md for all four
modes (0=half, 1=interleave, 2=quarter, 3=interleave-half):

    x_rotate = rotary_rearrange(x, mode)
    y        = x * cos + x_rotate * sin      (mode 3 uses the re-arranged x)
    For V2 with a rotate matrix: y = x * cos + (x @ rotate) * sin

The rotate-matrix branch is SoC dependent: only the DAV_2201 arch (Atlas
A2/A3) consumes it. On Ascend 950 the V2 op_api rejects a non-null rotate
(ACLNN_ERR_PARAM_INVALID) and the kernel never reads it, so the golden falls
back to the mode-based rotary there. ops-test-kit passes `short_soc_version`
in the golden kwargs for exactly this.

Golden computes in the input dtype (fp32 aligned with the kernel); for
fp16/bf16 cases run ttk with `--golden-mode Promote` to let the framework
promote inputs before golden. Run with `--compare close` too: stat_rel_err is
the only 2.1 spec-declarable float standard, but its |a-g|/(|g|+1e-7) blows
up on y = x*cos + x_rot*sin ~ 0 for arbitrary (-1,1) inputs, while isclose
(rtol/atol from the CSV precision_tolerances) handles near-zero via atol.
"""

__spec__ = {
    "rotary_position_embedding": "RotaryPositionEmbeddingTestSpec",
    "aclnnRotaryPositionEmbedding": "AclnnRotaryPositionEmbeddingTestSpec",
    "aclnnRotaryPositionEmbeddingV2": "AclnnRotaryPositionEmbeddingV2TestSpec",
}

import numpy as np
import torch

# torch.from_numpy cannot consume ml_dtypes extension dtypes (e.g. bfloat16)
_TORCH_NATIVE_NP_DTYPES = (
    "float16",
    "float32",
    "float64",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "bool",
    "complex64",
    "complex128",
)


def _as_tensor(arr):
    """numpy (incl. ml_dtypes bf16) -> torch.Tensor; None passthrough.

    bf16 -> fp32 is an exact conversion, done only when the framework did not
    already Promote the input (--golden-mode Promote)."""
    if arr is None:
        return None
    if str(arr.dtype) not in _TORCH_NATIVE_NP_DTYPES:
        arr = arr.astype(np.float32)
    return torch.from_numpy(np.ascontiguousarray(arr))


def _chunk2(t):
    return t.chunk(2, dim=-1)


def _chunk4(t):
    return t.chunk(4, dim=-1)


def _rotary_rearrange(x, mode):
    """The mode-dependent rearrangement of x (x_rotate / x_part1 & x_part2)."""
    if mode == 0:  # half
        x1, x2 = _chunk2(x)
        return torch.cat((-x2, x1), dim=-1)
    if mode == 1:  # interleave
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.stack((-x2, x1), dim=-1).reshape(x.shape)
    if mode == 2:  # quarter
        x1, x2, x3, x4 = _chunk4(x)
        return torch.cat((-x2, x1, -x4, x3), dim=-1)
    raise ValueError(f"unsupported mode: {mode}")


# The V2 rotate matrix is consumed only on the DAV_2201 arch (Atlas A2/A3:
# short_soc_version "Ascend910B" / "Ascend910_93"). On Ascend 950 (DAV_3510)
# the V2 op_api rejects a non-null rotate (ACLNN_ERR_PARAM_INVALID) and the
# kernel never reads it, so the golden must compute the mode-based rotary
# there — never x @ rotate.
_ROTATE_MATRIX_SHORT_SOC = ("Ascend910B", "Ascend910_93")


def _rotate_matrix_used(short_soc_version):
    if not short_soc_version:
        return True  # no SoC info (e.g. CPU golden): assume documented V2 semantics
    return short_soc_version in _ROTATE_MATRIX_SHORT_SOC


def _rotary_forward(x, cos, sin, mode, rotate=None, short_soc_version=None):
    """Shared torch implementation; returns [y]."""
    mode = int(mode)
    if rotate is not None and _rotate_matrix_used(short_soc_version):
        return [x * cos + torch.matmul(x, rotate) * sin]
    if mode == 3:  # interleave-half: interleave odd/even halves first
        x1, x2 = x[..., ::2], x[..., 1::2]
        x_part1 = torch.cat((x1, x2), dim=-1)
        x_part2 = torch.cat((-x2, x1), dim=-1)
        return [x_part1 * cos + x_part2 * sin]
    return [x * cos + _rotary_rearrange(x, mode) * sin]


class RotaryPositionEmbeddingTestSpec:
    """RotaryPositionEmbedding 测试规范（kernel/geir 流程，numpy 入参）

    Parameters follow rotary_position_embedding_def.cpp: x, cos, sin,
    rotate(optional) + attr mode.
    """

    def golden(x, cos, sin, rotate=None, mode=0, **kwargs):
        tensors = [_as_tensor(t) for t in (x, cos, sin)]
        rotate_t = _as_tensor(rotate)
        soc = kwargs.get("short_soc_version")
        return [r.numpy() for r in _rotary_forward(*tensors, mode, rotate_t, soc)]

    tolerance = {"float32": {"standard": "stat_rel_err"}}


class AclnnRotaryPositionEmbeddingTestSpec:
    """RotaryPositionEmbedding 测试规范（aclnn 流程，torch 入参）

    Parameters follow aclnnRotaryPositionEmbeddingGetWorkspaceSize
    (without workspaceSize & executor); all are passed positionally.
    """

    def golden(x, cos, sin, mode=0, out=None, **kwargs):
        return _rotary_forward(x, cos, sin, mode, None, kwargs.get("short_soc_version"))

    tolerance = {"float32": {"standard": "stat_rel_err"}}


class AclnnRotaryPositionEmbeddingV2TestSpec:
    """RotaryPositionEmbeddingV2 测试规范（aclnn 流程，torch 入参）

    Parameters follow aclnnRotaryPositionEmbeddingV2GetWorkspaceSize
    (without workspaceSize & executor); rotate is optional. Ascend 950 does
    not support the rotate matrix, so rotate stays None there and the V2
    interface computes the same mode-based rotary as V1.
    """

    def golden(x, cos, sin, mode=0, rotate=None, out=None, **kwargs):
        return _rotary_forward(
            x, cos, sin, mode, rotate, kwargs.get("short_soc_version")
        )

    tolerance = {"float32": {"standard": "stat_rel_err"}}
