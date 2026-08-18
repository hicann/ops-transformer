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
"""Golden reference for InplacePartialRotaryMulGrad (interleave mode, partial slice).

Matches the kernel formula in op_kernel/arch35/inplace_partial_rotary_mul_grad_common.h
(InterleaveModeGradVF), rotary_mode = 1 (interleave):

    dx[2k]   = cos[2k]*dy[2k]     + sin[2k+1]*dy[2k+1]
    dx[2k+1] = cos[2k+1]*dy[2k+1] - sin[2k]*dy[2k]

Only the slice [start, end) on the last dim is rotated; outside the slice dx == dy.
cos/sin have length == sliceLength (end - start) on the last dim and broadcast over
the leading (B, S, N) dims of dy.
"""

__golden__ = {
    "kernel": {"inplace_partial_rotary_mul_grad": "InplacePartialRotaryMulGrad"},
    "aclnn": {"aclnnInplacePartialRotaryMulGrad": "AclnnInplacePartialRotaryMulGrad"},
}

import numpy as np


def InplacePartialRotaryMulGrad(
    dy, cos, sin, *, rotary_mode=0, partial_slice=(0, 0), **extra
):
    out_dtype = dy.dtype
    dx = np.asarray(dy, dtype=np.float32).copy()

    start, end = int(partial_slice[0]), int(partial_slice[1])
    # Empty slice -> no rope, dx == dy
    if start == end:
        return dx.astype(out_dtype)

    if int(rotary_mode) != 1:
        raise NotImplementedError(
            f"golden only supports interleave mode (rotary_mode=1), got {rotary_mode}"
        )

    slice_len = end - start
    # Empty dy / cos / sin -> device treats as no-op (TILING_KEY_EMPTY, dx == dy).
    if dy.size == 0 or cos.size == 0 or sin.size == 0:
        return dx.astype(out_dtype)
    d = dx[..., start:end]  # (..., L)
    c = np.asarray(cos, dtype=np.float32)[..., :slice_len]  # (..., L)
    s = np.asarray(sin, dtype=np.float32)[..., :slice_len]
    c = np.broadcast_to(c, d.shape)
    s = np.broadcast_to(s, d.shape)

    even = np.arange(0, slice_len, 2)  # 2k
    odd = np.arange(1, slice_len, 2)  # 2k+1

    out = np.empty_like(d)
    out[..., even] = c[..., even] * d[..., even] + s[..., odd] * d[..., odd]
    out[..., odd] = c[..., odd] * d[..., odd] - s[..., even] * d[..., even]

    dx[..., start:end] = out
    return dx.astype(out_dtype)


def AclnnInplacePartialRotaryMulGrad(
    dyRef, cos, sin, rotaryMode=0, partialSlice=None, *args, **kwargs
):
    """aclnn 流程 golden（torch 入参，dyRef 原地输入/输出）。

    Parameters follow aclnnInplacePartialRotaryMulGradGetWorkspaceSize
    (without workspaceSize & executor): dyRef, cos, sin, rotaryMode, partialSlice.
    """
    import torch

    def _np(t):
        if t is None:
            return None
        if t.dtype == torch.bfloat16:
            t = t.float()
        return t.detach().cpu().numpy()

    sl = (
        (0, 0) if partialSlice is None else (int(partialSlice[0]), int(partialSlice[1]))
    )
    return [
        InplacePartialRotaryMulGrad(
            _np(dyRef),
            _np(cos),
            _np(sin),
            rotary_mode=int(rotaryMode),
            partial_slice=sl,
        )
    ]
