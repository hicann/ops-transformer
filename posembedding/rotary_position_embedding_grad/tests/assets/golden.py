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

The golden models the NPU kernel's arithmetic: it computes in float32
(`compute_dtype`) even for fp16/bf16 I/O — the kernel does fp32 multiply-add
and accumulates the dcos/dsin reductions internally — then casts each output
(dx/dcos/dsin) back to the output dtype (`out_dtype`) so a real overflow still
saturates to ``±inf``/``nan`` exactly as the kernel's output-format
conversion does.

Do NOT run this golden with ``--golden-mode Promote``: Promote computes in the
promoted dtype but never returns to the output dtype range, so cases where the
device saturates fail spuriously. Run with the default ``--golden-mode Enable``
plus ``--compare close``. bf16 results are returned as ``ml_dtypes.bfloat16``
ndarrays because a torch bf16 tensor can neither pass through ``.numpy()`` nor
be consumed by ``np.isclose``.
"""

__spec__ = {
    "rotary_position_embedding_grad": "RotaryPositionEmbeddingGradTestSpec",
    "aclnnRotaryPositionEmbeddingGrad": "AclnnRotaryPositionEmbeddingGradTestSpec",
    "torch_npu.npu_rotary_mul_backward": "NpuRotaryMulBackwardTestSpec",
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

_TORCH_DTYPE_BY_NAME = {
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "bfloat16": torch.bfloat16,
}

_BF16_NP = None


def _bf16_np():
    """Lazily resolved ml_dtypes.bfloat16 (kept behind a helper so the import
    only happens for bf16 goldens and is easy to stub in tests)."""
    global _BF16_NP
    if _BF16_NP is None:
        import ml_dtypes

        _BF16_NP = ml_dtypes.bfloat16
    return _BF16_NP


def _as_tensor(arr):
    """numpy (incl. ml_dtypes bf16) -> torch.Tensor; None passthrough.

    bf16 -> fp32 is an exact conversion, done only when the framework did not
    already Promote the input (--golden-mode Promote)."""
    if arr is None:
        return None
    if str(arr.dtype) not in _TORCH_NATIVE_NP_DTYPES:
        arr = arr.astype(np.float32)
    return torch.from_numpy(np.ascontiguousarray(arr))


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


def _golden_impl(dy, cos, sin, x, mode, compute_dtype=torch.float32, out_dtype=None):
    """Shared torch implementation; returns [dx] or [dx, dcos, dsin].

    compute_dtype (default fp32) mirrors the NPU kernel's internal fp32
    arithmetic for fp16/bf16 I/O; out_dtype casts every output back to the
    output dtype so a real overflow still saturates to ``±inf``/``nan``
    (IEEE) exactly as the kernel's output-format conversion does.
    """
    mode = int(mode)
    if compute_dtype is not None:
        dy, cos, sin = (
            t.to(compute_dtype) if t is not None else None for t in (dy, cos, sin)
        )
        if x is not None:
            x = x.to(compute_dtype)
    dx = _rotary_grad_dx(dy, cos, sin, mode)
    results = [dx]
    if x is not None:
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
        results = [dx, dcos, dsin]
    if out_dtype is not None:
        results = [r.to(out_dtype) for r in results]
    return results


def _golden_result(y, out_dtype, to_numpy=False):
    """Convert a torch golden tensor to the form ttk's comparison consumes.

    bf16 must become an ml_dtypes.bfloat16 ndarray: a torch bf16 tensor can
    neither pass through ``.numpy()`` nor be consumed by ``np.isclose``.
    fp16/fp32 stay a torch tensor by default (aclnn specs) or become a numpy
    ndarray (kernel spec, ``to_numpy=True``).
    """
    if isinstance(out_dtype, str):
        out_dtype = _TORCH_DTYPE_BY_NAME.get(out_dtype, y.dtype)
    if out_dtype == torch.bfloat16:
        # y is fp32/bf16; .float() is exact for bf16 and a no-op otherwise.
        return y.float().detach().cpu().numpy().astype(_bf16_np())
    y = y.detach().cpu()
    return y.numpy() if to_numpy else y


class RotaryPositionEmbeddingGradTestSpec:
    """RotaryPositionEmbeddingGrad 测试规范（kernel 流程，numpy 入参）

    Parameters follow rotary_position_embedding_grad_def.cpp: dy, cos, sin, x + attr mode.
    """

    def golden(dy, cos, sin, x=None, mode=0, **kwargs):
        tensors = [_as_tensor(t) for t in (dy, cos, sin)]
        x_t = _as_tensor(x)
        # Output dtype == input dtype for RPE grad; the framework passes it through.
        # Under --golden-mode Enable it is the declared dtype ('float16'/...);
        # fall back to the raw input dtype, then to the (possibly bf16→fp32
        # lifted) tensor dtype.
        out_dtype = kwargs.get("output_dtypes")
        if isinstance(out_dtype, (list, tuple)) and out_dtype:
            out_dtype = out_dtype[0]
        out_torch = _TORCH_DTYPE_BY_NAME.get(
            str(out_dtype)
        ) or _TORCH_DTYPE_BY_NAME.get(str(getattr(dy, "dtype", "")), tensors[0].dtype)
        results = _golden_impl(
            *tensors, x_t, mode, compute_dtype=torch.float32, out_dtype=out_torch
        )
        return [_golden_result(r, out_torch, to_numpy=True) for r in results]

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
        results = _golden_impl(
            dy,
            cos,
            sin,
            xOptional,
            mode,
            compute_dtype=torch.float32,
            out_dtype=dy.dtype,
        )
        return [_golden_result(r, dy.dtype) for r in results]

    tolerance = {"float32": {"standard": "stat_rel_err"}}


class NpuRotaryMulBackwardTestSpec:
    """E2E 通路 torch_npu.npu_rotary_mul_backward 测试规范（torch.Tensor 入参）

    npu_rotary_mul 的原生反向接口，返回 (dx, dr1, dr2) = grad wrt (input, r1, r2)。
    rotary_mode 'half' → RPE mode 0、'interleave' → mode 1。设备对广播 r1/r2 会把
    dr1/dr2 在广播轴上归约回 r1/r2 形状 —— 与 _golden_impl 的 dcos/dsin 广播归约
    （keepdim）行为一致；全宽 r1/r2 则无归约。golden 已用设备 autograd
    （NpuRotaryMulBackward0）与 CPU torch.autograd 双重验证，三梯度一致。

    op-plugin（RotaryMulBackwardKernelNpuOpApi.cpp）按 r1/r2.requires_grad() 决定
    是否把 x 传给 aclnnRotaryPositionEmbeddingGrad —— requires_grad=False 时 x 空、
    只算 dx、dcos/dsin 全 0。故 customize_inputs 必须给入参置 requires_grad_(True)；
    ttk 需在 input 插件跑完后复用插件 tensor（保留该标记），否则 e2e 比对 dcos/dsin
    恒 FAIL（见 ops-test-kit input_generation.generate_inputs）。
    """

    def customize_inputs(*args, **kwargs):
        for t in args:
            if isinstance(t, torch.Tensor):
                t.requires_grad_(True)

    def golden(grad, input, r1, r2, rotary_mode="half", **kwargs):
        mode = 0 if str(rotary_mode).lower() == "half" else 1
        results = _golden_impl(
            grad,
            r1,
            r2,
            input,
            mode,
            compute_dtype=torch.float32,
            out_dtype=grad.dtype,
        )
        return [_golden_result(r, grad.dtype) for r in results]

    tolerance = {"float32": {"standard": "stat_rel_err"}}
