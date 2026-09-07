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

The golden models the NPU kernel's arithmetic: it computes in float32
(`compute_dtype`) even for fp16/bf16 I/O — the kernel does fp32 multiply-add
internally regardless of the declared input/output dtype — then casts the
result back to the output dtype (`out_dtype`) so a real overflow still
saturates to ``±inf``/``nan`` exactly as the kernel's output-format
conversion does. This is what makes the golden agree with the device on
extreme/large-magnitude cases:

* a finite product that fits the output dtype stays finite in both;
* a product that exceeds the output dtype range becomes ``±inf`` in both.

Do NOT run this golden with ``--golden-mode Promote``: Promote computes in
the promoted dtype but never returns to the output dtype range, so cases
where the device saturates (``isclose(finite, inf)``) fail spuriously. Run
with the default ``--golden-mode Enable`` plus ``--compare close``. bf16
results are returned as ``ml_dtypes.bfloat16`` ndarrays because a torch bf16
tensor can neither pass through ``.numpy()`` nor be consumed by
``np.isclose``.
"""

__spec__ = {
    "rotary_position_embedding": "RotaryPositionEmbeddingTestSpec",
    "aclnnRotaryPositionEmbedding": "AclnnRotaryPositionEmbeddingTestSpec",
    "aclnnRotaryPositionEmbeddingV2": "AclnnRotaryPositionEmbeddingV2TestSpec",
    "torch_npu.npu_rotary_mul": "NpuRotaryMulTestSpec",
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


def _rotary_forward(
    x,
    cos,
    sin,
    mode,
    rotate=None,
    short_soc_version=None,
    compute_dtype=torch.float32,
    out_dtype=None,
):
    """Shared torch implementation; returns [y].

    compute_dtype (default fp32) mirrors the NPU kernel's internal fp32
    arithmetic for fp16/bf16 I/O; out_dtype casts the result back to the
    output dtype so a real overflow still saturates to ``±inf``/``nan``
    (IEEE) exactly as the kernel's output-format conversion does.
    """
    mode = int(mode)
    if compute_dtype is not None:
        x, cos, sin = (
            t.to(compute_dtype) if t is not None else None for t in (x, cos, sin)
        )
        if rotate is not None:
            rotate = rotate.to(compute_dtype)
    if rotate is not None and _rotate_matrix_used(short_soc_version):
        y = x * cos + torch.matmul(x, rotate) * sin
    elif mode == 3:  # interleave-half: interleave odd/even halves first
        x1, x2 = x[..., ::2], x[..., 1::2]
        x_part1 = torch.cat((x1, x2), dim=-1)
        x_part2 = torch.cat((-x2, x1), dim=-1)
        y = x_part1 * cos + x_part2 * sin
    else:
        y = x * cos + _rotary_rearrange(x, mode) * sin
    if out_dtype is not None:
        y = y.to(out_dtype)
    return [y]


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


class RotaryPositionEmbeddingTestSpec:
    """RotaryPositionEmbedding 测试规范（kernel/geir 流程，numpy 入参）

    Parameters follow rotary_position_embedding_def.cpp: x, cos, sin,
    rotate(optional) + attr mode.
    """

    def golden(x, cos, sin, rotate=None, mode=0, **kwargs):
        tensors = [_as_tensor(t) for t in (x, cos, sin)]
        rotate_t = _as_tensor(rotate)
        soc = kwargs.get("short_soc_version")
        # Output dtype == input dtype for RPE; the framework passes it through.
        # Under --golden-mode Enable it is the declared dtype ('float16'/...);
        # fall back to the raw input dtype, then to the (possibly bf16→fp32
        # lifted) tensor dtype.
        out_dtype = kwargs.get("output_dtypes")
        if isinstance(out_dtype, (list, tuple)) and out_dtype:
            out_dtype = out_dtype[0]
        out_torch = _TORCH_DTYPE_BY_NAME.get(
            str(out_dtype)
        ) or _TORCH_DTYPE_BY_NAME.get(str(getattr(x, "dtype", "")), tensors[0].dtype)
        y = _rotary_forward(
            *tensors,
            mode,
            rotate_t,
            soc,
            compute_dtype=torch.float32,
            out_dtype=out_torch,
        )[0]
        return [_golden_result(y, out_torch, to_numpy=True)]

    tolerance = {"float32": {"standard": "stat_rel_err"}}


class AclnnRotaryPositionEmbeddingTestSpec:
    """RotaryPositionEmbedding 测试规范（aclnn 流程，torch 入参）

    Parameters follow aclnnRotaryPositionEmbeddingGetWorkspaceSize
    (without workspaceSize & executor); all are passed positionally.
    """

    def golden(x, cos, sin, mode=0, out=None, **kwargs):
        y = _rotary_forward(
            x,
            cos,
            sin,
            mode,
            None,
            kwargs.get("short_soc_version"),
            compute_dtype=torch.float32,
            out_dtype=x.dtype,
        )[0]
        return [_golden_result(y, x.dtype)]

    tolerance = {"float32": {"standard": "stat_rel_err"}}


class AclnnRotaryPositionEmbeddingV2TestSpec:
    """RotaryPositionEmbeddingV2 测试规范（aclnn 流程，torch 入参）

    Parameters follow aclnnRotaryPositionEmbeddingV2GetWorkspaceSize
    (without workspaceSize & executor); rotate is optional. Ascend 950 does
    not support the rotate matrix, so rotate stays None there and the V2
    interface computes the same mode-based rotary as V1.
    """

    def golden(x, cos, sin, mode=0, rotate=None, out=None, **kwargs):
        y = _rotary_forward(
            x,
            cos,
            sin,
            mode,
            rotate,
            kwargs.get("short_soc_version"),
            compute_dtype=torch.float32,
            out_dtype=x.dtype,
        )[0]
        return [_golden_result(y, x.dtype)]

    tolerance = {"float32": {"standard": "stat_rel_err"}}


class NpuRotaryMulTestSpec:
    """E2E 通路 torch_npu.npu_rotary_mul 测试规范（torch.Tensor 入参）

    torch_npu 的 RotaryEmbedding 旋转位置编码接口，仅两种 rotary_mode，对应
    RPE 的前两种 mode：'half' → mode 0（chunk 两半旋转）、'interleave' →
    mode 1（奇偶交错旋转）。rotate 矩阵仅 DAV_2201（910B/A3）生效，Ascend 950
    上 op 拒绝非空 rotate（device error），故按 short_soc_version 分支走 mode
    公式。同 RPE golden：fp32 计算对齐设备内部算术 + 回 cast 输出 dtype。
    """

    def golden(input, r1, r2, rotary_mode="half", rotate=None, **kwargs):
        mode = 0 if str(rotary_mode).lower() == "half" else 1
        y = _rotary_forward(
            input,
            r1,
            r2,
            mode,
            rotate,
            kwargs.get("short_soc_version"),
            compute_dtype=torch.float32,
            out_dtype=input.dtype,
        )[0]
        return [_golden_result(y, input.dtype)]

    tolerance = {"float32": {"standard": "stat_rel_err"}}
