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

实现为纯 torch 切片，兼容 Kernel/GEIR（numpy 入参）与 ACLNN/E2E（任意设备上的
torch tensor 入参，含 NPU/CUDA 三方执行）。
"""

__spec__ = {
    "inplace_partial_rotary_mul_grad": "InplacePartialRotaryMulGradTestSpec",
    "aclnnInplacePartialRotaryMulGrad": "AclnnInplacePartialRotaryMulGradTestSpec",
    "cann_ops_transformer.ops.inplace_partial_rotary_mul_backward": "CannOpsInplacePartialRotaryMulBackwardTestSpec",
}

import sys

import numpy as np
import torch

_TORCH_MODE_INT = {"half": 0, "interleave": 1, "quarter": 2, "interleave-half": 3}
_BF16_NP = None


def _bf16_np():
    """ml_dtypes.bfloat16 惰性解析（torch bf16 张量无法直接 .numpy()）。"""
    global _BF16_NP
    if _BF16_NP is None:
        import ml_dtypes

        _BF16_NP = ml_dtypes.bfloat16
    return _BF16_NP


def _rotary_grad_compute(dy_f32, cos_f32, sin_f32, rotary_mode, start, end):
    """共享 torch 计算（所有入参为 float32 torch tensor），返回 float32 torch tensor。

    dy_f32/out 的 shape 为 (..., D)，cos_f32/sin_f32 的 shape 为 (..., L)，
    其中 L = end - start，广播到 dy 的 shape。
    """
    dx = dy_f32.clone()

    if (
        start == end
        or dy_f32.numel() == 0
        or cos_f32.numel() == 0
        or sin_f32.numel() == 0
    ):
        return dx

    if rotary_mode != 1:
        raise NotImplementedError(
            f"golden only supports interleave mode (rotary_mode=1), got {rotary_mode}"
        )

    slice_len = end - start
    # aten.slice 纯算子切片（不经过 numpy，兼容 CPU/NPU/CUDA 任意设备上的 tensor）
    d = torch.ops.aten.slice(dy_f32, -1, start, end)
    c = torch.ops.aten.slice(cos_f32, -1, 0, slice_len).broadcast_to(d.shape)
    s = torch.ops.aten.slice(sin_f32, -1, 0, slice_len).broadcast_to(d.shape)

    even = torch.arange(0, slice_len, 2, device=d.device)
    odd = torch.arange(1, slice_len, 2, device=d.device)

    out_even = c[..., even] * d[..., even] + s[..., odd] * d[..., odd]
    out_odd = c[..., odd] * d[..., odd] - s[..., even] * d[..., even]
    out = torch.cat([out_even.unsqueeze(-1), out_odd.unsqueeze(-1)], dim=-1).flatten(-2)
    dx = torch.cat(
        [
            torch.ops.aten.slice(dx, -1, 0, start),
            out,
            torch.ops.aten.slice(dx, -1, end, sys.maxsize),
        ],
        dim=-1,
    )
    return dx


class InplacePartialRotaryMulGradTestSpec:
    """InplacePartialRotaryMulGrad 测试规范（kernel/geir 流程，numpy 入参）

    Parameters follow inplace_partial_rotary_mul_grad_def.cpp:
    dy, cos, sin, rotary_mode, partial_slice.
    """

    def golden(dy, cos, sin, rotary_mode=0, partial_slice=(0, 0), **kwargs):
        def _from_numpy(t):
            if t.dtype == _bf16_np():
                return torch.from_numpy(t.view(np.float32))
            return torch.from_numpy(t).float()

        result_f32 = _rotary_grad_compute(
            _from_numpy(dy),
            _from_numpy(cos),
            _from_numpy(sin),
            int(rotary_mode),
            int(partial_slice[0]),
            int(partial_slice[1]),
        )
        result_np = result_f32.numpy()
        if dy.dtype == np.float16:
            result_np = result_np.astype(np.float16)
        elif dy.dtype == _bf16_np():
            result_np = result_np.astype(_bf16_np())
        return [result_np]

    class ThirdPartyImpl:
        def __init__(self, dy, **kwargs):
            self.out_dtype = dy.dtype

        def __call__(self, dy, cos, sin, rotary_mode=0, partial_slice=(0, 0), **kwargs):
            result_f32 = _rotary_grad_compute(
                dy.float(),
                cos.float(),
                sin.float(),
                int(rotary_mode),
                int(partial_slice[0]),
                int(partial_slice[1]),
            )
            if self.out_dtype == torch.bfloat16:
                result_f32 = result_f32.to(torch.bfloat16)
            elif self.out_dtype == torch.float16:
                result_f32 = result_f32.half()
            return [result_f32]

    third_party = {"torch": ThirdPartyImpl}
    tolerance = {
        "float16": {"standard": "cross_check", "level": "L1"},
        "bfloat16": {"standard": "cross_check", "level": "L1"},
    }


class AclnnInplacePartialRotaryMulGradTestSpec:
    """InplacePartialRotaryMulGrad 测试规范（aclnn 流程，torch 入参，已在设备上）

    Parameters follow aclnnInplacePartialRotaryMulGradGetWorkspaceSize
    (without workspaceSize & executor): dyRef, cos, sin, rotaryMode,
    partialSliceOptional.
    """

    def golden(
        dyRef,
        cos,
        sin,
        rotaryMode=0,
        partialSlice=None,
        partialSliceOptional=None,
        **kwargs,
    ):
        ps = partialSliceOptional if partialSliceOptional is not None else partialSlice
        start, end = (0, 0) if ps is None else (int(ps[0]), int(ps[1]))

        result_f32 = _rotary_grad_compute(
            dyRef.float(),
            cos.float(),
            sin.float(),
            int(rotaryMode),
            start,
            end,
        )
        if dyRef.dtype == torch.bfloat16:
            result_f32 = result_f32.to(torch.bfloat16)
        elif dyRef.dtype == torch.float16:
            result_f32 = result_f32.half()
        return [result_f32]

    class ThirdPartyImpl:
        def __init__(self, dyRef, **kwargs):
            self.out_dtype = dyRef.dtype

        def __call__(
            self,
            dyRef,
            cos,
            sin,
            rotaryMode=0,
            partialSlice=None,
            partialSliceOptional=None,
            **kwargs,
        ):
            ps = (
                partialSliceOptional
                if partialSliceOptional is not None
                else partialSlice
            )
            start, end = (0, 0) if ps is None else (int(ps[0]), int(ps[1]))
            result_f32 = _rotary_grad_compute(
                dyRef.float(),
                cos.float(),
                sin.float(),
                int(rotaryMode),
                start,
                end,
            )
            if self.out_dtype == torch.bfloat16:
                result_f32 = result_f32.to(torch.bfloat16)
            elif self.out_dtype == torch.float16:
                result_f32 = result_f32.half()
            return [result_f32]

    third_party = {"torch": ThirdPartyImpl}
    tolerance = {
        "float16": {"standard": "cross_check", "level": "L1"},
        "bfloat16": {"standard": "cross_check", "level": "L1"},
    }


class CannOpsInplacePartialRotaryMulBackwardTestSpec:
    """InplacePartialRotaryMulBackward 测试规范（e2e 流程，torch 入参，已在设备上）

    Parameters follow the torch dispatcher schema
    inplace_partial_rotary_mul_backward(Tensor grad_output, Tensor r1,
    Tensor r2, *, str rotary_mode, (int, int) partial_slice) -> ().

    torch 前端语义：grad_output 原地修改，API 无返回值，TTK 回读 grad_output 与 golden 比对。
    返回值首位补 None 对齐 TTK e2e 侧的 [None, grad_output] 输出结构。
    """

    def golden(
        grad_output, r1, r2, *, rotary_mode="interleave", partial_slice=None, **kwargs
    ):
        mode = (
            _TORCH_MODE_INT.get(rotary_mode.lower(), 1)
            if isinstance(rotary_mode, str)
            else int(rotary_mode)
        )
        start, end = (
            (0, 0)
            if partial_slice is None
            else (int(partial_slice[0]), int(partial_slice[1]))
        )

        result_f32 = _rotary_grad_compute(
            grad_output.float(),
            r1.float(),
            r2.float(),
            mode,
            start,
            end,
        )
        if grad_output.dtype == torch.bfloat16:
            result_f32 = result_f32.to(torch.bfloat16)
        elif grad_output.dtype == torch.float16:
            result_f32 = result_f32.half()
        return [None, result_f32]

    class ThirdPartyImpl:
        def __init__(self, grad_output, **kwargs):
            self.out_dtype = grad_output.dtype

        def __call__(
            self,
            grad_output,
            r1,
            r2,
            *,
            rotary_mode="interleave",
            partial_slice=None,
            **kwargs,
        ):
            mode = (
                _TORCH_MODE_INT.get(rotary_mode.lower(), 1)
                if isinstance(rotary_mode, str)
                else int(rotary_mode)
            )
            start, end = (
                (0, 0)
                if partial_slice is None
                else (int(partial_slice[0]), int(partial_slice[1]))
            )
            result_f32 = _rotary_grad_compute(
                grad_output.float(),
                r1.float(),
                r2.float(),
                mode,
                start,
                end,
            )
            if self.out_dtype == torch.bfloat16:
                result_f32 = result_f32.to(torch.bfloat16)
            elif self.out_dtype == torch.float16:
                result_f32 = result_f32.half()
            return [None, result_f32]

    third_party = {"torch": ThirdPartyImpl}
    tolerance = {
        "float16": {"standard": "cross_check", "level": "L1"},
        "bfloat16": {"standard": "cross_check", "level": "L1"},
    }
