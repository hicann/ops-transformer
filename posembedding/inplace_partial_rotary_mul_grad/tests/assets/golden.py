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

Three test paths share this reference:
  * kernel (numpy inputs)  — "InplacePartialRotaryMulGrad"
  * aclnn  (torch inputs)  — "AclnnInplacePartialRotaryMulGrad"
  * e2e torch path via cann_ops_transformer.ops.inplace_partial_rotary_mul_backward
    (torch inputs, str rotary_mode) — "CannOpsInplacePartialRotaryMulBackward"
"""

__golden__ = {
    "kernel": {"inplace_partial_rotary_mul_grad": "InplacePartialRotaryMulGrad"},
    "aclnn": {"aclnnInplacePartialRotaryMulGrad": "AclnnInplacePartialRotaryMulGrad"},
    "e2e": {
        "cann_ops_transformer.ops.inplace_partial_rotary_mul_backward": "CannOpsInplacePartialRotaryMulBackward",
    },
}

import numpy as np

# torch 前端的字符串 rotary_mode -> 算子整数 mode（见 torch_extension/
# inplace_partial_rotary_mul_backward.cpp 的 mode_map；当前仅实现 interleave）
_TORCH_MODE_INT = {"half": 0, "interleave": 1, "quarter": 2, "interleave-half": 3}

_BF16_NP = None


def _bf16_np():
    """ml_dtypes.bfloat16 惰性解析（torch bf16 张量无法直接 .numpy()）。"""
    global _BF16_NP
    if _BF16_NP is None:
        import ml_dtypes

        _BF16_NP = ml_dtypes.bfloat16
    return _BF16_NP


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


def CannOpsInplacePartialRotaryMulBackward(
    grad_output, r1, r2, *, rotary_mode="interleave", partial_slice=None, **kwargs
):
    """e2e 通路 cann_ops_transformer.ops.inplace_partial_rotary_mul_backward golden
    （torch 入参，grad_output 原地输入/输出）。

    torch 前端语义（torch_extension/inplace_partial_rotary_mul_backward.py）：
    rotary_mode 为字符串（当前仅 'interleave'），partial_slice 为二元组，结果原地
    写回 grad_output —— 输入位即输出位，golden 返回写回后的期望值。

    与 aclnn 分支不同，本分支把 fp32 参考值回 cast 到设备输出 dtype（bf16 经
    ml_dtypes.bfloat16），使半精度舍入被正确建模。

    返回值首位补 None 对齐框架的输出结构：该 torch API 无返回值（schema -> ()），
    ttk e2e 侧 result 为 [None, 原地回读的 grad_output]，zip 比对要求 golden 同构；
    None 位在比对器里按 golden-suppressed 处理（SUPPRESSED/pass），真正的精度比对
    发生在第二位。
    """
    import torch

    def _np(t):
        if t is None:
            return None
        if t.dtype == torch.bfloat16:
            t = t.float()
        return t.detach().cpu().numpy()

    if isinstance(rotary_mode, str):
        mode = _TORCH_MODE_INT.get(rotary_mode.lower())
        if mode is None:
            raise NotImplementedError(
                f"golden only supports rotary_mode in {sorted(_TORCH_MODE_INT)}, got {rotary_mode!r}"
            )
    else:
        mode = int(rotary_mode)

    sl = (
        (0, 0)
        if partial_slice is None
        else (int(partial_slice[0]), int(partial_slice[1]))
    )
    out = InplacePartialRotaryMulGrad(
        _np(grad_output), _np(r1), _np(r2), rotary_mode=mode, partial_slice=sl
    )

    out_dtype = grad_output.dtype
    if out_dtype == torch.bfloat16:
        return [None, out.astype(_bf16_np())]
    if out_dtype == torch.float16:
        return [None, out.astype(np.float16)]
    return [None, out]
