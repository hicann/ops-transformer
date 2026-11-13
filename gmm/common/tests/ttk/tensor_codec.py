# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""CPU conversions for TTK tensor formats and packed dtypes."""

import numpy as np
import torch


def _nd(value, fmt, ori_shape):
    if value is None:
        return None
    if fmt != "FRACTAL_NZ":
        return value
    shape = tuple(ori_shape)
    # [..., N1, M1, M0, N0] -> [..., M1, M0, N1, N0].
    d = value.ndim
    order = list(range(d - 4)) + [d - 3, d - 2, d - 4, d - 1]
    dense = value.transpose(order).reshape(
        value.shape[:-4]
        + (value.shape[-3] * value.shape[-2], value.shape[-4] * value.shape[-1])
    )
    return dense[tuple(slice(0, n) for n in shape)]


def _cast(value, dtype):
    if dtype == "bfloat16":
        return value.astype(_resolve_bf16_dtype())
    if dtype in ("int8", "uint8"):
        info = np.iinfo(dtype)
        return np.clip(np.rint(value), info.min, info.max).astype(dtype)
    return value.astype(dtype)


def _resolve_bf16_dtype():
    import ml_dtypes

    return ml_dtypes.bfloat16


def _numpy(tensor):
    if tensor is None:
        return None
    if tensor.dtype in (torch.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2):
        return tensor.detach().float().cpu().numpy()
    return tensor.detach().cpu().numpy()


def _decode(tensor, semantic):
    a = _numpy(tensor)
    if a is None:
        return None
    if semantic in ("hifloat8", "float8_e8m0"):
        import en_dtypes

        return (
            np.ascontiguousarray(a)
            .view(getattr(en_dtypes, semantic))
            .astype(np.float32)
        )
    if semantic == "int4" or semantic.startswith("float4"):
        trans = tensor.ndim >= 2 and tensor.stride(-2) == 1 and tensor.stride(-1) > 1
        if trans:
            a = a.swapaxes(-1, -2)
        bits = a.dtype.itemsize * 8
        raw = np.ascontiguousarray(a).view(np.dtype(f"uint{bits}"))
        codes = (
            raw[..., None].astype(np.uint64)
            >> (np.arange(bits // 4, dtype=np.uint64) * 4)
        ) & 15
        codes = codes.astype(np.uint8).reshape(
            a.shape[:-1] + (a.shape[-1] * (bits // 4),)
        )
        if semantic == "int4":
            values = np.where(
                codes >= 8, codes.astype(np.int32) - 16, codes.astype(np.int32)
            )
        else:
            import en_dtypes

            values = codes.view(getattr(en_dtypes, semantic)).astype(np.float32)
        return values.swapaxes(-1, -2) if trans else values
    return a


def _np(t):
    if t is None:
        return None
    if isinstance(t, torch.Tensor):
        if "float8" in str(t.dtype) or t.dtype == torch.bfloat16:
            return t.detach().float().cpu().numpy()
        return t.detach().cpu().numpy()
    return np.asarray(t)


def _whole(t):
    if isinstance(t, torch.Tensor):
        size = t.untyped_storage().nbytes() // t.element_size()
        return _np(t.as_strided((size,), (1,), 0))
    a = t
    storage = t
    while getattr(a, "base", None) is not None:
        a = a.base
        if isinstance(a, np.ndarray):
            storage = a
    return np.asarray(storage).reshape(-1)


def _numeric(t, semantic, carrier=""):
    a = _np(t)
    if carrier:
        bits = np.ascontiguousarray(a).view(np.uint32)
        codes = (
            (bits[..., None].astype(np.uint64) >> (np.arange(8, dtype=np.uint64) * 4))
            & 15
        ).astype(np.uint8)
        codes = codes.reshape(a.shape[:-1] + (a.shape[-1] * 8,))
        if carrier == "int4":
            return np.where(codes >= 8, codes.astype(np.int32) - 16, codes)
        import en_dtypes

        return codes.view(en_dtypes.float4_e2m1).astype(np.float32)
    return (
        a.astype(np.float32)
        if semantic not in ("int64", "uint64", "int32", "int8")
        else a
    )


def _weights(weight, semantic, trans, c0, carrier, k, n):
    values = []
    for t in weight:
        if c0:
            # Each TensorList element may have its own K/N and no expert axis.
            tk, tn = t.shape[-2:]
            if carrier:
                tk, tn = (tk * 8, tn) if trans else (tk, tn * 8)
            sr, sc = (tn, tk) if trans else (tk, tn)
            batch = tuple(t.shape[:-2])
            last = c0 // 8 if carrier else c0
            raw = _whole(t).reshape(
                batch + ((sc + c0 - 1) // c0, (sr + 15) // 16, 16, last)
            )
            decoded = _numeric(raw, semantic, carrier)
            dense = _nd(decoded, "FRACTAL_NZ", batch + (sr, sc))
            values.append(dense.swapaxes(-1, -2) if trans else dense)
        else:
            values.append(_numeric(t, semantic, carrier))
    return values
