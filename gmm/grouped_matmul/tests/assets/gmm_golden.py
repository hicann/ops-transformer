# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""TTK Kernel, ACLNN and Torch specs and CPU reference for GroupedMatmul."""

import math
from ast import literal_eval
from math import prod
import sys
from pathlib import Path
from runpy import run_path

import numpy as np
import torch


_SHARED = Path(__file__).resolve().parents[3] / "common/tests/ttk"
_CODEC = run_path(str(_SHARED / "tensor_codec.py"))
_nd, _cast, _numpy, _decode = (
    _CODEC[name] for name in ("_nd", "_cast", "_numpy", "_decode")
)
_np, _whole, _numeric, _weights = (
    _CODEC[name] for name in ("_np", "_whole", "_numeric", "_weights")
)


def u64_to_deq_scale(scale):
    return (
        (np.asarray(scale, dtype=np.uint64) & np.uint64(0xFFFFFFFF))
        .astype(np.uint32)
        .view(np.float32)
    )


def scale_generate(value):
    value = np.ascontiguousarray(value, dtype=np.float32)
    return (value.view(np.uint32) & np.uint32(0xFFFFE000)).view(np.float32)


def transform_tensor(value):
    return value.transpose(0, 2, 1).reshape(-1, value.shape[1])


def _leaf(value):
    while isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    return value


def _slots(value):
    if value is None:
        return []
    return list(value) if isinstance(value, (tuple, list)) else [value]


def _present(value):
    return value is not None and np.asarray(value).size > 0


def _kernel_dtype(meta, index, default="float32"):
    value = meta[index] if index < len(meta) else default
    while isinstance(value, (list, tuple)):
        value = value[0] if value else default
    return {"float": "float32", "fp8_e4m3fn": "float8_e4m3fn"}.get(value, value)


def _expand_scale(value, k, transposed=False):
    value = np.asarray(value, dtype=np.float32)
    if value.ndim == 3:
        # K major: [ceil(K/64),N,2]; N major: [N,ceil(K/64),2].
        value = (
            value.reshape(value.shape[0], -1)
            if transposed
            else value.transpose(0, 2, 1).reshape(-1, value.shape[1])
        )
    if transposed:
        value = value.T
    return np.repeat(value, 32, axis=0)[:k]


def _mx_scale_nd(value, ori_shape):
    """MX scale NZ: [..., N1, K64, 16, 2] -> [..., K64, N, 2]."""
    if value is None:
        return None
    d = value.ndim
    order = list(range(d - 4)) + [d - 3, d - 4, d - 2, d - 1]
    dense = value.transpose(order).reshape(
        value.shape[:-4]
        + (value.shape[-3], value.shape[-4] * value.shape[-2], value.shape[-1])
    )
    return dense[tuple(slice(0, n) for n in ori_shape)]


def s8s4_correction(weight, scale):
    """Required offline bias: reduce_sum(8 * weight * scale, K)."""
    weight = np.asarray(weight, dtype=np.float32)
    scales = u64_to_deq_scale(scale)
    if scales.ndim == 2:
        scales = scales[:, None, :]
    expanded = np.repeat(scales, weight.shape[-2] // scales.shape[-2], axis=-2)
    return np.sum(weight * expanded * np.float32(8), axis=-2, dtype=np.float32)


def integer_pergroup(args, output_name, group_list_type):
    a = args[0][0].astype(np.int32)
    weights = args[1][0].astype(np.int32)
    scales = u64_to_deq_scale(args[3][0])
    if scales.ndim == 2:
        scales = scales[:, None, :]
    counts = np.diff(np.r_[0, args[7]]) if group_list_type == 0 else args[7]
    group_size = a.shape[1] // scales.shape[1]
    result = []
    start = 0
    for e, count in enumerate(counts):
        end = start + int(count)
        c = np.zeros((int(count), weights.shape[-1]), np.float16)
        for j in range(scales.shape[1]):
            sl = slice(j * group_size, (j + 1) * group_size)
            part = (a[start:end, sl] @ weights[e, sl]).astype(np.float32) * scales[e, j]
            c = (
                c.astype(np.float32) + part.astype(np.float16).astype(np.float32)
            ).astype(np.float16)
        out = c.astype(np.float32)
        if args[4] is not None:
            off = args[4][0][e].reshape(1, -1)
            out += a[start:end].sum(axis=1, keepdims=True).astype(np.float32) * off
        if args[2] is not None:
            correction = (
                s8s4_correction(weights, args[3][0])[e]
                if args[0][0].dtype == np.int8
                else 0
            )
            out += args[2][0][e] - correction
        if args[8] is not None:
            out *= args[8][start:end].reshape(-1, 1)
        result.append(_cast(out, output_name))
        start = end
    return [np.concatenate(result)]


def grouped_matmul_golden(
    x,
    weight,
    bias,
    scale,
    offset,
    antiquant_scale,
    antiquant_offset,
    group_list,
    per_token_scale,
    *,
    transpose_x=False,
    transpose_weight=False,
    group_type=-1,
    group_list_type=0,
    split_item=0,
    **kwargs,
):
    """Independent CPU reference. Metadata selects semantics, never tensor .any()."""
    types = kwargs.get("input_dtypes", ())
    formats = kwargs.get("input_formats", ())
    shapes = kwargs.get("input_ori_shapes", ())
    output_types = kwargs.get("output_dtypes", ("float32",))
    args = [
        x,
        weight,
        bias,
        scale,
        offset,
        antiquant_scale,
        antiquant_offset,
        group_list,
        per_token_scale,
    ]
    logical = []
    for index, arg in enumerate(args):
        slots = _slots(arg)
        fmts = _slots(formats[index]) if index < len(formats) else ["ND"]
        original = shapes[index] if index < len(shapes) else None
        original_slots = (
            _slots(original)
            if original and isinstance(original[0], (tuple, list))
            else [original] * len(slots)
        )
        values = []
        for j, v in enumerate(slots):
            fmt = _leaf(fmts[min(j, len(fmts) - 1)])
            # Scale NZ affinity tiles N by 16 and retains the two E8M0
            # lanes, unlike the matrix NZ layout used for x/weight.
            if (
                index == 3
                and fmt == "FRACTAL_NZ"
                and _kernel_dtype(types, 3) == "float8_e8m0"
            ):
                values.append(_mx_scale_nd(v, original_slots[j]))
            else:
                values.append(_nd(v, fmt, original_slots[j]))
        logical.append(values)
    xs, ws, bs, ss, os, aq, ao, gl, ps = logical
    groups = np.asarray(gl[0], dtype=np.int64) if gl and _present(gl[0]) else None
    xa, wb = _kernel_dtype(types, 0), _kernel_dtype(types, 1)
    sd, ad = _kernel_dtype(types, 3), _kernel_dtype(types, 5)
    weight_only = (
        bool(aq and _present(aq[0]))
        and (
            xa in ("float16", "bfloat16")
            and xa != wb
            or xa == "int8"
            and wb == "int4"
            or "float4" in wb
            and ad == "float8_e8m0"
        )
        and sd != "float8_e8m0"
    )
    full_quant = not weight_only and xa in (
        "int4",
        "int8",
        "float8_e4m3fn",
        "float8_e5m2",
        "hifloat8",
        "float4_e2m1",
        "float4_e1m2",
    )
    mx = full_quant and sd == "float8_e8m0"
    if (
        xa == "int4"
        and wb == "int4"
        and ss
        and ss[0].ndim == 3
        or xa == "int8"
        and wb == "int4"
        and sd in ("uint64", "int64")
    ):
        token = (
            None
            if not ps or not _present(ps[0])
            else ps[0]
            if len(ps) == 1
            else np.concatenate(ps)
        )
        bias_values = bs if bs and _present(bs[0]) else None
        offset_values = os if os and _present(os[0]) else None
        # NZ decoding restores the physical source matrix [N,K]. The
        # arithmetic helper consumes logical [K,N], including its early path.
        weights = [v.swapaxes(-1, -2) for v in ws] if transpose_weight else ws
        return integer_pergroup(
            [xs, weights, bias_values, ss, offset_values, aq, ao, groups, token],
            _kernel_dtype(output_types, 0),
            group_list_type,
        )

    def at(values, expert):
        if not values or not _present(values[0]):
            return None
        if len(values) > 1:
            return values[expert]
        v = values[0]
        if group_type == -1:
            return v
        # Single batched weight/scale/bias; scalar single-expert vector is not batched.
        return v[expert] if v.ndim > 1 else v

    base_x = xs[0].astype(np.float32)
    if transpose_x:
        base_x = base_x.T
    if groups is None:
        count = max(len(xs), len(ws))
        groups = (
            np.array([v.shape[-1 if transpose_weight else -2] for v in ws])
            if group_type == 2 and len(ws) > 1
            else np.array(
                [xs[i].shape[-1 if transpose_x else -2] for i in range(len(xs))]
            )
        )
        if len(groups) == 1:
            groups = np.repeat(groups, count)
        group_list_type = 1
    if group_list_type == 2:
        ids = groups[:, 0]
        counts = groups[:, 1]
    else:
        ids = np.arange(len(groups))
        counts = np.diff(np.r_[0, groups]) if group_list_type == 0 else groups
    if np.any(counts < 0):
        raise ValueError("group_list contains negative group sizes")
    ends = np.cumsum(counts)
    starts = np.r_[0, ends[:-1]]
    outs = []
    scale_k_offset = 0
    for index, (start_value, end_value, expert_id) in enumerate(zip(starts, ends, ids)):
        start, end, expert = int(start_value), int(end_value), int(expert_id)
        if group_type == 2:
            a = base_x[:, start:end]
            b = (ws[index] if len(ws) > 1 else ws[0]).astype(np.float32)
            if transpose_weight:
                b = b.T
            if len(ws) == 1:
                b = b[start:end]
        else:
            a = xs[index].astype(np.float32) if len(xs) > 1 else base_x[start:end]
            if len(xs) > 1 and transpose_x:
                a = a.T
            raw = (
                ws[expert]
                if len(ws) > 1
                else ws[0][expert]
                if ws[0].ndim == 3
                else ws[0]
            )
            b = raw.astype(np.float32)
            if transpose_weight:
                b = b.T
        m, k = a.shape
        n = b.shape[-1]
        qbias = at(bs, expert)
        qscale = at(ss, expert)
        p = ps[0] if ps and _present(ps[0]) else None
        if weight_only:
            ant = at(aq, expert)
            off = at(ao, expert)
            if ad == "float8_e8m0":
                expanded = _expand_scale(ant, k, transpose_weight)
                b = b * expanded
            else:
                ant = np.asarray(ant, dtype=np.float32)
                if ant.ndim == 1:
                    expanded = ant[None, :]
                else:
                    # Per-group weight-only scales are [G,N] regardless of the
                    # weight's transpose flag (unlike MX paired-scale storage).
                    expanded = np.repeat(ant, math.ceil(k / ant.shape[0]), axis=0)[:k]
                if off is not None:
                    b = _cast(
                        _cast(b, xa).astype(np.float32)
                        + np.asarray(off, dtype=np.float32).reshape(1, -1),
                        xa,
                    ).astype(np.float32)
                b = b * expanded
            b = (
                np.rint(b.astype(np.float16)).astype(np.int32)
                if xa == "int8"
                else _cast(b, xa).astype(np.float32)
            )
            if xa == "float8_e4m3fn" and p is not None:
                pflat = p.reshape(p.shape[0], -1).astype(np.float32)
                a = a * np.repeat(pflat, 32, axis=-1)[start:end, :k]
        if mx:
            # Split-K scales include per-expert alignment windows in their K-major storage.
            if group_type == 2:
                stop = (end // 64 + index + 1) * 2
                ast = transform_tensor(p)[scale_k_offset:stop].T
                bst = transform_tensor(ss[0])[scale_k_offset:stop]
                scale_k_offset = stop
                a = a * np.repeat(ast.astype(np.float32), 32, axis=-1)[:, :k]
                b = b * np.repeat(bst.astype(np.float32), 32, axis=0)[:k]
            else:
                # MX TensorLists store local [K64,N,2] (or [N,K64,2])
                # scales. A single batched scale has an extra expert axis;
                # unlike ordinary channel scales, rank three is not batched.
                qscale = (
                    ss[expert]
                    if len(ss) > 1
                    else ss[0][expert]
                    if ss[0].ndim == 4
                    else ss[0]
                )
                # Adapters may keep token scales local to each expert or join
                # them along M. Select before flattening the K64/2 axes so
                # nonuniform expert partitions never reuse the first scale.
                local_tokens = len(ps) > 1
                p = ps[index] if local_tokens else ps[0]
                token_rows = (
                    a.shape[0] if local_tokens or len(xs) > 1 else base_x.shape[0]
                )
                ast = (
                    p.reshape(token_rows, math.ceil(k / 64) * 2)
                    if not transpose_x
                    else p.transpose(1, 0, 2).reshape(token_rows, math.ceil(k / 64) * 2)
                )
                if not local_tokens and len(xs) == 1:
                    ast = ast[start:end]
                a = a * np.repeat(ast.astype(np.float32), 32, axis=-1)[:, :k]
                b = b * _expand_scale(qscale, k, transpose_weight)
        # Integer accumulation uses exact int32 arithmetic; floating inputs use FP32.
        out = (
            a.astype(np.int32) @ b.astype(np.int32) if xa in ("int8", "int4") else a @ b
        )
        if qbias is not None and _kernel_dtype(types, 2) == "int32":
            out = out + qbias
        if full_quant and not mx and _kernel_dtype(output_types, 0) != "int32":
            scales = ss[0]
            if sd in ("uint64", "int64"):
                scales = u64_to_deq_scale(scales)
            scales = np.asarray(scales, dtype=np.float32)
            # Per-group/per-block (GB) uses K128/N128 scales.
            is_gb = p is not None and (
                (group_type != 2 and scales.ndim == 3)
                or (
                    group_type == 2
                    and p.ndim == 2
                    and scales.ndim == 2
                    and p.shape[0] == base_x.shape[1] // 128 + len(groups)
                    and p.shape[1] == base_x.shape[0]
                    and scales.shape[1] == math.ceil(n / 128)
                )
            )
            if is_gb:
                if group_type == 2:
                    off = start // 128 + index
                    ac = np.asarray(p, dtype=np.float32).T[
                        :, off : off + math.ceil(k / 128)
                    ]
                    bc = scales[off : off + math.ceil(k / 128)]
                else:
                    ac = np.asarray(p[start:end], dtype=np.float32)
                    bc = scales[expert]
                    if transpose_weight:
                        bc = bc.T
                out = np.zeros((m, n), np.float32)
                for j in range(math.ceil(k / 128)):
                    sl = slice(j * 128, min((j + 1) * 128, k))
                    out += (
                        (a[:, sl] @ b[sl])
                        * ac[:, j, None]
                        * np.repeat(bc[j], 128)[:n][None, :]
                    )
            else:
                sc = (
                    scales[expert]
                    if scales.ndim > 1 or scales.size == len(groups)
                    else scales
                )
                floating_bias = qbias is not None and _kernel_dtype(types, 2) != "int32"
                if (
                    p is None
                    and not floating_bias
                    and sd == "float32"
                    and np.asarray(sc).size == 1
                ):
                    # Static T-T uses the same 10-bit-mantissa dequant scalar as
                    # the fused dynamic T-T path, not an unrestricted FP32 scale.
                    sc = scale_generate(sc)
                out = out.astype(np.float32) * np.asarray(sc).reshape(1, -1)
                if xa == "int4" and wb == "int4":
                    # S4S4 Fixpipe writes half workspace before the vector per-token multiply.
                    out = out.astype(np.float16).astype(np.float32)
                if p is not None:
                    pv = (
                        p[index]
                        if group_type == 2
                        else p[expert]
                        if p.ndim == 1
                        and p.size == len(groups)
                        or p.shape == (len(groups), 1)
                        else p[start:end]
                    )
                    if (
                        xa != "int4"
                        and (
                            (p.ndim == 1 and p.size == len(groups))
                            or p.shape == (len(groups), 1)
                        )
                        and scales.shape
                        in (
                            (len(groups),),
                            (len(groups), 1),
                        )
                    ):
                        fused = scale_generate(
                            np.asarray(sc, dtype=np.float32)
                            * np.asarray(pv, dtype=np.float32)
                        )
                        out = (a @ b) * fused.reshape(1, -1)
                    else:
                        out = out * np.asarray(pv, dtype=np.float32).reshape(-1, 1)
                # GMM offset is reserved; it is not an output zero-point.
        if weight_only and xa == "int8":
            out = (
                out.astype(np.float32)
                * np.asarray(qscale, dtype=np.float32).reshape(1, -1)
                * np.asarray(p[start:end], dtype=np.float32).reshape(-1, 1)
            )
        if qbias is not None and _kernel_dtype(types, 2) != "int32":
            out = out + qbias
        outs.append(
            _cast(out, _kernel_dtype(output_types, min(index, len(output_types) - 1)))
        )
    if group_type == 2:
        return outs if split_item in (0, 1) else [np.stack(outs)]
    if split_item in (0, 1):
        return outs
    requested = _leaf(kwargs.get("output_ori_shapes", ()))
    requested_m = (
        requested[0]
        if isinstance(requested, (list, tuple))
        and len(requested) == 2
        and isinstance(requested[0], int)
        else base_x.shape[0]
    )
    if len(xs) == 1 and (requested_m > base_x.shape[0] or int(ends[-1]) > requested_m):
        raise ValueError("Declared output rows do not contain the expert partition")
    if len(xs) == 1 and int(ends[-1]) < requested_m:
        # TTK a996a41 kernel/golden.py::_alloc_output_buffers initializes output
        # to ones. Rows outside the expert partition retain that fixture state;
        # this is not a promise that GMM mathematically generates ones.
        outs.append(
            np.ones(
                (requested_m - int(ends[-1]), outs[0].shape[1]), dtype=outs[0].dtype
            )
        )
    elif len(xs) == 1 and int(ends[-1]) > base_x.shape[0]:
        raise ValueError("group_list exceeds M")
    return [np.concatenate(outs, axis=0)]


_HINTS = {
    285: "int4",
    290: "hifloat8",
    291: "float8_e5m2",
    292: "float8_e4m3fn",
    293: "float8_e8m0",
    296: "float4_e2m1",
    297: "float4_e1m2",
}
_OUTPUTS = {
    1: torch.int8,
    3: torch.int32,
    5: torch.float16,
    6: torch.float32,
    15: torch.bfloat16,
}


def _first(values):
    if isinstance(values, (list, tuple)):
        return values[0] if values else None
    return values


def _dtype(tensor, hint=None):
    if hint is not None:
        return _HINTS.get(hint, str(hint).removeprefix("torch."))
    if tensor is None:
        return "float32"
    return (
        str(tensor.dtype)
        .removeprefix("torch.")
        .replace("float8_e8m0fnu", "float8_e8m0")
    )


def _convert(values, semantic):
    if values is None:
        return None
    return (
        [_decode(v, semantic) for v in values]
        if isinstance(values, (list, tuple))
        else _decode(values, semantic)
    )


def _reference_shapes(args):
    """Describe decoded logical tensors without collapsing TensorList boundaries."""
    return [
        [tuple(v.shape) for v in value]
        if isinstance(value, (list, tuple))
        else tuple(value.shape)
        if value is not None
        else None
        for value in args
    ]


def _legacy_helper_args(args):
    # The dedicated integer/MX A8W4 paths retain their existing single-x contract.
    values = list(args)
    values[8] = None if args[8] is None else np.concatenate(args[8], axis=0)
    return values


def _integer_pergroup(args, output_name, group_list_type):
    return integer_pergroup(args, output_name, group_list_type)


def _mx_a8w4(args, output_name, group_list_type):
    """Apply MX scales to decoded operands, without re-quantizing the product."""
    x, weight, bias, _, _, anti, _, groups, tokens = args
    if len(x) != 1 or len(weight) != 1 or weight[0].ndim != 3:
        raise ValueError(
            "MX A8W4 reference currently requires one batched weight and one x"
        )
    counts = np.diff(np.r_[0, groups]) if group_list_type == 0 else groups
    result = []
    start = 0
    k = x[0].shape[-1]
    for e, count in enumerate(counts):
        end = start + int(count)
        a_scale = np.repeat(tokens[start:end].reshape(int(count), -1), 32, axis=1)[
            :, :k
        ]
        b_scale = np.repeat(
            anti[0][e].transpose(0, 2, 1).reshape(-1, weight[0].shape[-1]), 32, axis=0
        )[:k]
        out = (x[0][start:end].astype(np.float32) * a_scale) @ (weight[0][e] * b_scale)
        if bias is not None:
            out += bias[0][e]
        result.append(_cast(out, output_name))
        start = end
    return [np.concatenate(result)]


_ACLNN = (
    "x",
    "weight",
    "biasOptional",
    "scaleOptional",
    "offsetOptional",
    "antiquantScaleOptional",
    "antiquantOffsetOptional",
    "perTokenScaleOptional",
    "groupListOptional",
    "activationInputOptional",
    "activationQuantScaleOptional",
    "activationQuantOffsetOptional",
    "out",
    "activationFeatureOutOptional",
    "dynQuantScaleOutOptional",
)
_E2E = (
    "x",
    "weight",
    "bias",
    "scale",
    "offset",
    "antiquant_scale",
    "antiquant_offset",
    "per_token_scale",
    "group_list",
    "activation_input",
    "activation_quant_scale",
    "activation_quant_offset",
)


def _swap(shape, axis):
    axes = list(range(len(shape)))
    axes[axis], axes[axis + 1] = axes[axis + 1], axes[axis]
    return tuple(shape[i] for i in axes)


def _strides(shape, axis):
    dense = _swap(shape, axis)
    return _swap(tuple(prod(dense[i + 1 :]) for i in range(len(dense))), axis)


def _nested(shape):
    return bool(shape and isinstance(shape[0], (list, tuple)))


def _map_shape(shape, func):
    return [func(s) for s in shape] if _nested(shape) else func(shape)


def _descriptor_strides(shape, axis):
    values = _map_shape(shape, lambda s: _strides(s, axis))
    if isinstance(values, list) and all(v == values[0] for v in values):
        return values[0]
    return values


def _packed_type(cell):
    semantic = cell.get("logical_dtype", cell["dtype"])
    return (
        semantic
        if cell["dtype"] in ("float32", "int32") and semantic in ("float4_e2m1", "int4")
        else ""
    )


def _logical_weight(shape, trans, packed):
    k, n = shape[-2:]
    if packed:
        k, n = (k * 8, n) if trans else (k, n * 8)
    return k, n


def _nz_weight(shape, c0, trans, packed):
    k, n = _logical_weight(shape, trans, packed)
    rows, cols = (n, k) if trans else (k, n)
    return (
        *shape[:-2],
        (cols + c0 - 1) // c0,
        (rows + 15) // 16,
        16,
        c0 // 8 if packed else c0,
    )


def _nz_scale(shape):
    *prefix, k_blocks, n, pair = shape
    return (*prefix, (n + 15) // 16, k_blocks, 16, pair)


def _describe(params, names, *, aclnn):
    cells = [params[name] for name in names]
    shapes = tuple(cell["shape"] for cell in cells)
    dtypes = tuple(cell["dtype"] for cell in cells)
    formats, storage, strides = (
        ["ND"] * len(names),
        [None] * len(names),
        [None] * len(names),
    )
    for i, (cell, shape) in enumerate(zip(cells, shapes)):
        if shape is None:
            continue
        layout = cell.get("layout", "")
        trans = (i == 0 and params["transpose_x"]) or (
            i == 1 and params["transpose_weight"]
        )
        if trans or layout.startswith("transpose"):
            axis = -3 if layout.startswith("transpose_blocks") else -2
            strides[i] = _descriptor_strides(shape, axis)
            if layout.endswith("storage"):
                storage[i] = _map_shape(shape, lambda s: _swap(s, axis))
        if aclnn and i == 1 and params["weight_format"] == "nz":
            formats[i] = "FRACTAL_NZ"
            storage[i] = _map_shape(
                shape,
                lambda s: _nz_weight(
                    s, cell["nz_tile_cols"], trans, _packed_type(cell)
                ),
            )
        if aclnn and i == 3 and cell.get("format") == "nz":
            formats[i] = "FRACTAL_NZ"
            storage[i] = _map_shape(shape, _nz_scale)
    props = {
        "tensor_formats": tuple(formats),
        "tensor_storage_shapes": tuple(storage),
        "tensor_view_strides": tuple(strides),
        "input_data_ranges": tuple(cell.get("range") for cell in cells),
    }
    group_name = "groupListOptional" if aclnn else "group_list"
    if "values" in params[group_name]:
        props["attribute_defaults"] = {group_name: params[group_name]["values"]}
    if aclnn:
        props["output_tensor_indexes"] = tuple(
            i for i in range(names.index("out"), len(names)) if shapes[i] is not None
        )
    if not aclnn:
        props.pop("tensor_storage_shapes", None)
    return {"shapes": shapes, "dtypes": dtypes, "tensor_properties": props}


def describe_aclnn(params):
    return _describe(params, _ACLNN, aclnn=True)


def describe_e2e(params):
    return _describe(params, _E2E, aclnn=False)


def aclnn_reference_metadata(fields):
    cells = [literal_eval(fields[name]) for name in _ACLNN]
    trans = literal_eval(fields["transpose_weight"])
    weight = cells[1]
    packed = _packed_type(weight)
    shape = weight["shape"]
    if _nested(shape):
        shape = shape[0]
    k, n = _logical_weight(shape, trans, packed)
    return dict(
        gmm_semantic=[cell.get("logical_dtype", cell["dtype"]) for cell in cells[:12]],
        gmm_transpose_weight=trans,
        gmm_nz_c0=weight.get("nz_tile_cols", 0),
        gmm_weight_carrier=packed,
        gmm_k=k,
        gmm_n=n,
        gmm_scale_nz=cells[3].get("format") == "nz",
    )


class GroupedMatmulTestSpec:
    @staticmethod
    def golden(*args, **kwargs):
        outputs = grouped_matmul_golden(*args, **kwargs)

        def shapes(v):
            if v is None:
                return []
            if isinstance(v, (list, tuple)) and all(isinstance(i, int) for i in v):
                return [tuple(v)]
            return [shape for item in v for shape in shapes(item)]

        declared = shapes(kwargs.get("output_ori_shapes"))
        if declared and [a.shape for a in outputs] != declared:
            raise ValueError(
                f"Golden outputs {[a.shape for a in outputs]} disagree with declared {declared}"
            )
        return outputs

    @staticmethod
    def customize_inputs(
        x,
        weight,
        bias,
        scale,
        offset,
        antiquant_scale,
        antiquant_offset,
        group_list,
        per_token_scale,
        *,
        csv_fields=None,
        ttk_group_list=None,
        **kwargs,
    ):
        raw_group_list = (csv_fields or {}).get("group_list_values")
        expected = (
            ttk_group_list
            if ttk_group_list is not None
            else (literal_eval(raw_group_list) if raw_group_list else None)
        )
        if expected is not None:
            group_list = np.asarray(expected, dtype=np.int64)
        elif group_list is not None and np.asarray(group_list).size:
            # Random integers are not a valid expert partition. Build a balanced partition.
            a = _leaf(x)
            gt = kwargs.get("group_type", 0)
            tx = kwargs.get("transpose_x", False)
            total = a.shape[0 if tx else 1] if gt == 2 else a.shape[1 if tx else 0]
            count = len(group_list)
            group_list = np.full(count, total // count, dtype=np.int64)
            group_list[: total % count] += 1
            if kwargs.get("group_list_type", 0) == 0:
                group_list = np.cumsum(group_list)
        types = kwargs.get("input_dtypes", ())
        if _kernel_dtype(types, 3) in ("uint64", "int64"):

            def packed(v):
                if v is None:
                    return None
                if isinstance(v, (list, tuple)):
                    return type(v)(packed(t) for t in v)
                values = (np.asarray(v, dtype=np.uint64) % 1000 + 1).astype(
                    np.float32
                ) * np.float32(0.001)
                bits = (values.view(np.uint32) & np.uint32(0xFFFFE000)).astype(
                    np.uint64
                )
                if _kernel_dtype(kwargs.get("output_dtypes", ()), 0) == "int8":
                    bits |= np.uint64(1) << np.uint64(46)
                return bits.astype(v.dtype)

            scale = packed(scale)

        if (
            _kernel_dtype(types, 0) == "int8"
            and _kernel_dtype(types, 1) == "int4"
            and _kernel_dtype(types, 3) in ("uint64", "int64")
        ):
            metadata = kwargs.get("input_ori_shapes", ())
            formats = kwargs.get("input_formats", ())
            original = metadata[1][0] if metadata else None
            fmt = _leaf(formats[1]) if len(formats) > 1 else "ND"
            dense = _nd(_slots(weight)[0], fmt, original)
            _slots(bias)[0][...] = s8s4_correction(dense, _slots(scale)[0])

        def finite_e8(value):
            if isinstance(value, (list, tuple)):
                return type(value)(finite_e8(t) for t in value)
            if value is not None and "e8m0" in str(value.dtype):
                value = value.copy()
                bad = ~np.isfinite(value.astype(np.float32))
                value[bad] = 1
            return value

        scale = finite_e8(scale)
        antiquant_scale = finite_e8(antiquant_scale)
        per_token_scale = finite_e8(per_token_scale)
        return (
            x,
            weight,
            bias,
            scale,
            offset,
            antiquant_scale,
            antiquant_offset,
            group_list,
            per_token_scale,
        )


def _scales(scale, weights, meta, group_type):
    """Decode MX scale NZ's N16/pair layout, independently of weight NZ."""
    if scale is None:
        return None
    formats = meta.get("tensor_formats", ())
    scale_formats = (
        "FRACTAL_NZ"
        if meta.get("gmm_scale_nz", False)
        else formats[3]
        if len(formats) > 3
        else formats[0]
        if len(formats) == 1
        else "ND"
    )
    scale_formats = _slots(scale_formats)
    values = []
    d = meta["gmm_semantic"]
    for i, t in enumerate(scale):
        fmt = _leaf(scale_formats[min(i, len(scale_formats) - 1)])
        if fmt != "FRACTAL_NZ":
            values.append(_numeric(t, d[3]))
            continue
        if (
            d[0] != "float8_e4m3fn"
            or d[1] != "float8_e4m3fn"
            or d[3] != "float8_e8m0"
            or group_type != 0
            or meta["gmm_transpose_weight"]
        ):
            raise ValueError(
                "MX scale FRACTAL_NZ requires FP8 E4M3FN, group0 and non-transposed weight"
            )
        w = weights[i] if len(weights) > 1 else weights[0]
        k, n = w.shape[-2:]
        batch = w.shape[:-2]
        blocks = (k + 63) // 64
        raw = _numeric(_whole(t), d[3]).reshape(batch + ((n + 15) // 16, blocks, 16, 2))
        # [..., N1, K64, N0, pair] -> [..., K64, N1, N0, pair].
        rank = raw.ndim
        order = list(range(rank - 4)) + [rank - 3, rank - 4, rank - 2, rank - 1]
        dense = raw.transpose(order).reshape(batch + (blocks, ((n + 15) // 16) * 16, 2))
        values.append(dense[..., :n, :])
    return values


def _run(
    x,
    weight,
    bias,
    scale,
    offset,
    anti,
    anti_offset,
    tokens,
    groups,
    split,
    group_type,
    group_list_type,
    out,
    meta,
):
    d = meta["gmm_semantic"]
    trans = meta["gmm_transpose_weight"]
    nz = meta["gmm_nz_c0"]
    carrier = meta["gmm_weight_carrier"]

    def cv(v, dt):
        return None if v is None else [_numeric(t, dt) for t in v]

    aa = cv(x, d[0])
    bb = _weights(weight, d[1], trans, nz, carrier, meta["gmm_k"], meta["gmm_n"])
    args = [
        aa,
        bb,
        cv(bias, d[2]),
        _scales(scale, bb, meta, group_type),
        cv(offset, d[4]),
        cv(anti, d[5]),
        cv(anti_offset, d[6]),
        _np(groups),
        cv(tokens, d[7]),
    ]
    if (
        group_type == 2
        and d[3] == "float8_e8m0"
        and tokens is not None
        and tokens[0].ndim == 3
    ):
        token_view = tokens[0]
        stride0 = (
            token_view.stride(0)
            if isinstance(token_view, torch.Tensor)
            else token_view.strides[0] // token_view.itemsize
        )
        if stride0 == 2:
            args[8] = [args[8][0].swapaxes(0, 1)]
    # groupType=2 x is already its logical [M,K] view; K-C is [E,M],
    # G-B is [M,K128+E] with the reference's scale-window convention.
    if (
        group_type == 2
        and args[8] is not None
        and len(args[8]) == 1
        and args[8][0].ndim == 2
        and args[3] is not None
        and args[3][0].ndim == 2
        and args[8][0].shape[1] == aa[0].shape[-1] // 128 + len(groups)
        and args[3][0].shape[-1] == (meta["gmm_n"] + 127) // 128
    ):
        args[8] = [args[8][0].T]
    od = (
        str(out[0].dtype).removeprefix("torch.")
        if isinstance(out[0], torch.Tensor)
        else out[0].dtype.name
    )
    if (
        d[0].startswith("float8")
        and d[1].startswith("float4")
        and d[5] == "float8_e8m0"
    ):
        outputs = _mx_a8w4(_legacy_helper_args(args), od, group_list_type)
    elif (
        carrier == "int4"
        and d[3] in ("int64", "uint64")
        or d[0] == "int4"
        and args[3][0].ndim == 3
    ):
        outputs = _integer_pergroup(_legacy_helper_args(args), od, group_list_type)
    else:
        if d[3] != "float8_e8m0" and args[8] is not None:
            # Ordinary paths historically pass one token-scale array to the reference.
            args[8] = np.concatenate(args[8], axis=0)
        outputs = grouped_matmul_golden(
            *args,
            split_item=split,
            group_type=group_type,
            group_list_type=group_list_type,
            input_dtypes=(d[0], d[1], d[2], d[3], d[4], d[5], d[6], "int64", d[7]),
            input_ori_shapes=_reference_shapes(args),
            output_dtypes=(od,),
        )
    if [tuple(v.shape) for v in outputs] != [tuple(t.shape) for t in out]:
        raise ValueError(
            "CPU reference output shapes disagree with ACLNN output descriptors"
        )
    return outputs


class GroupedMatmulV5TestSpec:
    describe = staticmethod(describe_aclnn)

    @staticmethod
    def customize_inputs(
        x,
        weight,
        bias,
        scale,
        offset,
        anti,
        anti_offset,
        tokens,
        groups,
        activation_input,
        activation_scale,
        activation_offset,
        split,
        group_type,
        group_list_type,
        act_type,
        tuning,
        out,
        feature,
        dynamic,
        **kwargs,
    ):
        reference = kwargs.get("gmm_reference")
        if reference is None:
            reference = aclnn_reference_metadata(kwargs["testcase"].original_dict)
        group_values = kwargs.get("ttk_group_list", reference.get("group_list"))
        if groups is not None and group_values is not None:
            group_values = np.asarray(group_values, dtype=np.int64)
            if isinstance(groups, torch.Tensor):
                groups.copy_(torch.from_numpy(group_values))
            else:
                groups[...] = group_values
        kwargs.update(reference)
        d = kwargs["gmm_semantic"]
        for seq, idx in [(x, 0), (weight, 1), (scale, 3), (anti, 5), (tokens, 7)]:
            if seq is None:
                continue
            for t in seq:
                a = _np(t)
                if idx == 3 and d[idx] in ("int64", "uint64"):
                    vals = (a.astype(np.uint64) % 1000 + 1).astype(
                        np.float32
                    ) * np.float32(0.001)
                    value = (vals.view(np.uint32) & np.uint32(0xFFFFE000)).astype(
                        np.uint64
                    )
                    od = str(out[0].dtype)
                    if od in ("torch.int8", "int8"):
                        value |= np.uint64(1) << np.uint64(46)
                    value = value.astype(a.dtype)
                elif idx == 1 and kwargs["gmm_weight_carrier"]:
                    codes = (
                        (a.astype(np.int64)[..., None] + np.arange(8) * 3) & 15
                    ).astype(np.uint64)
                    value = (
                        np.sum(codes << (np.arange(8, dtype=np.uint64) * 4), axis=-1)
                        .astype(np.uint32)
                        .view(a.dtype)
                    )
                else:
                    continue
                if isinstance(t, torch.Tensor):
                    t.copy_(torch.from_numpy(value))
                else:
                    t[...] = value

        if (
            d[0] == "int8"
            and d[1] == "int4"
            and d[3] in ("int64", "uint64")
            and bias is not None
        ):
            dense = _weights(
                weight,
                d[1],
                kwargs["gmm_transpose_weight"],
                kwargs["gmm_nz_c0"],
                kwargs["gmm_weight_carrier"],
                kwargs["gmm_k"],
                kwargs["gmm_n"],
            )[0]
            correction = s8s4_correction(dense, _np(scale[0]))
            if isinstance(bias[0], torch.Tensor):
                bias[0].copy_(torch.from_numpy(correction))
            else:
                bias[0][...] = correction

    @staticmethod
    def golden(
        x,
        weight,
        bias,
        scale,
        offset,
        anti,
        anti_offset,
        tokens,
        groups,
        activation_input,
        activation_scale,
        activation_offset,
        split,
        group_type,
        group_list_type,
        act_type,
        tuning,
        out,
        feature,
        dynamic,
        *,
        csv_fields=None,
        gmm_reference=None,
        **kwargs,
    ):
        if act_type != 0:
            raise ValueError("Activation fusion is outside the GMM reference")
        kwargs.update(
            gmm_reference
            if gmm_reference is not None
            else aclnn_reference_metadata(csv_fields)
        )
        return _run(
            x,
            weight,
            bias,
            scale,
            offset,
            anti,
            anti_offset,
            tokens,
            groups,
            split,
            group_type,
            group_list_type,
            out,
            kwargs,
        )


class GroupedMatmulWeightNzTestSpec:
    describe = staticmethod(describe_aclnn)

    @staticmethod
    def customize_inputs(
        x,
        weight,
        bias,
        scale,
        offset,
        anti,
        anti_offset,
        tokens,
        groups,
        activation_input,
        activation_scale,
        activation_offset,
        split,
        group_type,
        group_list_type,
        act_type,
        tuning,
        quant_group_size,
        out,
        feature,
        dynamic,
        **kwargs,
    ):
        return GroupedMatmulV5TestSpec.customize_inputs(
            x,
            weight,
            bias,
            scale,
            offset,
            anti,
            anti_offset,
            tokens,
            groups,
            activation_input,
            activation_scale,
            activation_offset,
            split,
            group_type,
            group_list_type,
            act_type,
            tuning,
            out,
            feature,
            dynamic,
            **kwargs,
        )

    @staticmethod
    def golden(
        x,
        weight,
        bias,
        scale,
        offset,
        anti,
        anti_offset,
        tokens,
        groups,
        activation_input,
        activation_scale,
        activation_offset,
        split,
        group_type,
        group_list_type,
        act_type,
        tuning,
        quant_group_size,
        out,
        feature,
        dynamic,
        *,
        csv_fields=None,
        **kwargs,
    ):
        return GroupedMatmulV5TestSpec.golden(
            x,
            weight,
            bias,
            scale,
            offset,
            anti,
            anti_offset,
            tokens,
            groups,
            activation_input,
            activation_scale,
            activation_offset,
            split,
            group_type,
            group_list_type,
            act_type,
            tuning,
            out,
            feature,
            dynamic,
            csv_fields=csv_fields,
            **kwargs,
        )


def _restore_graph_views(reference, **tensors):
    if reference is None:
        return
    for name, strides in reference["strides"].items():
        values = tensors[name]
        for i, value in enumerate(values):
            target_stride = strides[i] if isinstance(strides[0], tuple) else strides
            if value.stride() != target_stride:
                values[i] = value.contiguous().as_strided(value.shape, target_stride)


def _optional_torch_tensors(reference, **tensors):
    if reference is not None:
        for name in reference.get("empty_tensorlists", ()):
            if name in tensors:
                tensors[name] = None
    return tuple(tensors.values())


sys.path.insert(0, str(_SHARED))
try:
    from gmm_ttk_graph_models import GroupedMatmulGraph
finally:
    sys.path.pop(0)


class TorchNpuGroupedMatmulTestSpec:
    describe = staticmethod(describe_e2e)
    torch_graph = GroupedMatmulGraph

    @staticmethod
    def customize_inputs(
        x,
        weight,
        *,
        bias=None,
        scale=None,
        offset=None,
        antiquant_scale=None,
        antiquant_offset=None,
        per_token_scale=None,
        group_list=None,
        activation_input=None,
        activation_quant_scale=None,
        activation_quant_offset=None,
        x_dtype=None,
        weight_dtype=None,
        scale_dtype=None,
        per_token_scale_dtype=None,
        ttk_group_list=None,
        **kwargs,
    ):
        """Only mutate CPU tensors, preserving the framework's backing storage."""
        if group_list is not None and ttk_group_list is not None:
            group_list.copy_(group_list.new_tensor(ttk_group_list))
        reference = kwargs.get("gmm_graph_reference")
        bias, scale, antiquant_scale, per_token_scale = _optional_torch_tensors(
            reference,
            bias=bias,
            scale=scale,
            antiquant_scale=antiquant_scale,
            per_token_scale=per_token_scale,
        )
        _restore_graph_views(
            reference,
            x=x,
            weight=weight,
            scale=scale,
            antiquant_scale=antiquant_scale,
            per_token_scale=per_token_scale,
        )
        for values, hint in [
            (x, x_dtype),
            (weight, weight_dtype),
            (scale, scale_dtype),
            (per_token_scale, per_token_scale_dtype),
        ]:
            if values is None:
                continue
            for t in values:
                if (
                    t.dtype == torch.int32
                    and values is x
                    or t.dtype == torch.int32
                    and values is weight
                ):
                    raw = _numpy(t).astype(np.int64)
                    codes = (raw[..., None] + np.arange(8) * 3) & 15
                    packed = np.sum(
                        codes.astype(np.uint64) << (np.arange(8, dtype=np.uint64) * 4),
                        axis=-1,
                    )
                    t.copy_(torch.from_numpy(packed.astype(np.uint32).view(np.int32)))
                elif (
                    t.dtype == torch.float32
                    and values is weight
                    and antiquant_scale is not None
                    and _first(antiquant_scale).dtype == torch.uint8
                ):
                    raw = (_numpy(t) * 100).astype(np.int64)
                    codes = (raw[..., None] + np.arange(8) * 3) & 15
                    packed = np.sum(
                        codes.astype(np.uint64) << (np.arange(8, dtype=np.uint64) * 4),
                        axis=-1,
                    )
                    t.copy_(torch.from_numpy(packed.astype(np.uint32).view(np.float32)))
                elif _HINTS.get(hint) == "float8_e8m0":
                    t.copy_((t.to(torch.int16) % 3 + 126).to(t.dtype))
                elif _HINTS.get(hint) == "hifloat8":
                    import en_dtypes

                    numeric = (_numpy(t).astype(np.int32) % 17 - 8).astype(
                        np.float32
                    ) * 0.25
                    codes = numeric.astype(en_dtypes.hifloat8).view(np.uint8)
                    t.copy_(torch.from_numpy(codes).to(t.dtype))
        if antiquant_scale is not None and _first(antiquant_scale).dtype == torch.uint8:
            for t in antiquant_scale:
                t.copy_((t.to(torch.int16) % 3 + 126).to(t.dtype))
        if scale is not None:
            for t in scale:
                if t.dtype in (torch.int64, torch.uint64):
                    values = (_numpy(t).astype(np.uint64) % 1000 + 1).astype(
                        np.float32
                    ) * np.float32(0.001)
                    bits = (values.view(np.uint32) & np.uint32(0xFFFFE000)).astype(
                        np.uint64
                    )
                    if kwargs.get("output_dtype") == 1:
                        bits |= np.uint64(1) << np.uint64(46)
                    t.copy_(torch.from_numpy(bits.view(np.int64)).to(t.dtype))

        if (
            x[0].dtype == torch.int8
            and weight[0].dtype == torch.int32
            and bias is not None
            and scale is not None
            and scale[0].dtype in (torch.int64, torch.uint64)
        ):
            correction = s8s4_correction(_decode(weight[0], "int4"), _numpy(scale[0]))
            bias[0].copy_(torch.from_numpy(correction))

    @staticmethod
    def npu_preprocess(
        x, weight, *, csv_fields=None, gmm_graph_reference=None, **kwargs
    ):
        if gmm_graph_reference is None:
            weight_format = (csv_fields or {}).get("weight_format")
            scale_nz = literal_eval(csv_fields["scale"]).get("format") == "nz"
        else:
            weight_format = gmm_graph_reference["weight_format"]
            scale_nz = gmm_graph_reference["scale_nz"]
            for name in gmm_graph_reference.get("empty_tensorlists", ()):
                kwargs[name].clear()
            _restore_graph_views(gmm_graph_reference, x=x, weight=weight, **kwargs)
        if weight_format not in ("nd", "nz"):
            raise ValueError(
                f"weight_format column must be 'nd' or 'nz', got {weight_format!r}"
            )
        if weight_format == "nd" and not scale_nz:
            return
        import torch_npu

        if scale_nz:
            for i, value in enumerate(kwargs["scale"]):
                kwargs["scale"][i] = torch_npu.npu_format_cast(
                    value,
                    29,
                    customize_dtype=torch_npu.float8_e8m0fnu,
                    input_dtype=torch_npu.float8_e8m0fnu,
                )
        if weight_format == "nd":
            return
        for i, value in enumerate(weight):
            trans = value.ndim >= 2 and value.stride(-2) == 1 and value.stride(-1) > 1
            source = value.transpose(-1, -2).contiguous() if trans else value
            if value.dtype in (torch.int32, torch.float32):
                # Packed words must be converted using the logical element layout.
                semantic = "int4" if value.dtype == torch.int32 else "float4_e2m1"
                logical = torch.from_numpy(
                    np.ascontiguousarray(_decode(value.cpu(), semantic))
                )
                logical = logical.to(device=value.device, dtype=value.dtype)
                source = logical.transpose(-1, -2).contiguous() if trans else logical
                layout_dtype = (
                    torch.int8 if value.dtype == torch.int32 else torch.float16
                )
                if value.dtype == torch.float32 and x[0].dtype == torch.float8_e4m3fn:
                    # MX A8W4 uses the FP8 K32 tile, unlike the A16W4 K16 word layout.
                    layout_dtype = torch.float8_e4m3fn
                nz = torch_npu.npu_format_cast(source, 29, customize_dtype=layout_dtype)
                converted = torch_npu.npu_convert_weight_to_int4pack(nz)
            elif (
                value.dtype == torch.uint8
                and x[0].dtype == torch.float8_e4m3fn
                and kwargs.get("weight_dtype") in (296, 297)
            ):
                converted = torch_npu.npu_format_cast(
                    source,
                    29,
                    customize_dtype=torch.float8_e4m3fn,
                    input_dtype=kwargs["weight_dtype"],
                )
            else:
                converted = torch_npu.npu_format_cast(source, 29)
            weight[i] = converted.transpose(-1, -2) if trans else converted
        return

    @staticmethod
    def golden(
        x,
        weight,
        *,
        bias=None,
        scale=None,
        offset=None,
        antiquant_scale=None,
        antiquant_offset=None,
        per_token_scale=None,
        group_list=None,
        activation_input=None,
        activation_quant_scale=None,
        activation_quant_offset=None,
        split_item=0,
        group_type=None,
        group_list_type=0,
        act_type=0,
        output_dtype=None,
        x_dtype=None,
        weight_dtype=None,
        scale_dtype=None,
        per_token_scale_dtype=None,
        **kwargs,
    ):
        if act_type not in (None, 0):
            raise ValueError(
                "Activation fusion requires its own reference; not silently ignored"
            )
        bias, scale, offset, antiquant_scale, antiquant_offset, per_token_scale = (
            _optional_torch_tensors(
                kwargs.get("gmm_graph_reference"),
                bias=bias,
                scale=scale,
                offset=offset,
                antiquant_scale=antiquant_scale,
                antiquant_offset=antiquant_offset,
                per_token_scale=per_token_scale,
            )
        )
        _restore_graph_views(
            kwargs.get("gmm_graph_reference"),
            x=x,
            weight=weight,
            scale=scale,
            antiquant_scale=antiquant_scale,
            per_token_scale=per_token_scale,
        )
        xd = _dtype(x[0], x_dtype)
        wd = _dtype(weight[0], weight_dtype)
        sd = _dtype(_first(scale), scale_dtype)
        pd = _dtype(_first(per_token_scale), per_token_scale_dtype)
        if weight[0].dtype == torch.int32:
            wd = "int4"
        if x[0].dtype == torch.int32 and wd == "int4":
            xd = "int4"
        ad = _dtype(_first(antiquant_scale))
        if antiquant_scale is not None and _first(antiquant_scale).dtype == torch.uint8:
            ad = "float8_e8m0"
            if weight[0].dtype == torch.float32:
                wd = "float4_e2m1"
        bd = _dtype(_first(bias))
        od = (
            _OUTPUTS.get(output_dtype, x[0].dtype)
            if not isinstance(output_dtype, torch.dtype)
            else output_dtype
        )
        output_name = str(od).removeprefix("torch.")
        args = [
            _convert(x, xd),
            _convert(weight, wd),
            _convert(bias, bd),
            _convert(scale, sd),
            _convert(offset, _dtype(_first(offset))),
            _convert(antiquant_scale, ad),
            _convert(antiquant_offset, _dtype(_first(antiquant_offset))),
            _numpy(group_list),
            _convert(per_token_scale, pd),
        ]
        if (
            group_type == 2
            and sd == "float8_e8m0"
            and per_token_scale is not None
            and len(per_token_scale) == 1
            and per_token_scale[0].ndim == 3
            and per_token_scale[0].stride(-3) == 2
        ):
            # API exposes the transposed logical view; Split-K math consumes K64-major storage.
            args[8] = [args[8][0].transpose(1, 0, 2)]
        if (
            group_type == 2
            and args[8] is not None
            and len(args[8]) == 1
            and args[8][0].ndim == 2
            and args[3] is not None
            and args[3][0].ndim == 2
            and args[3][0].shape[-1] == (args[1][0].shape[-1] + 127) // 128
        ):
            args[8] = [args[8][0].T]
        if xd.startswith("float8") and wd.startswith("float4") and ad == "float8_e8m0":
            if group_type != 0 or split_item not in (2, 3):
                raise ValueError(
                    "MX A8W4 reference currently supports M grouping and a single output"
                )
            outputs = _mx_a8w4(_legacy_helper_args(args), output_name, group_list_type)
        elif (
            xd == "int4"
            and wd == "int4"
            and args[3][0].ndim == 3
            or xd == "int8"
            and wd == "int4"
            and sd in ("int64", "uint64")
        ):
            outputs = _integer_pergroup(
                _legacy_helper_args(args), output_name, group_list_type
            )
        else:
            if sd != "float8_e8m0":
                # Preserve the ordinary quantization adapter's legacy layout.
                if args[8] is not None:
                    args[8] = np.concatenate(args[8], axis=0)
                if args[3] is not None and len(args[3]) > 1:
                    args[3] = [np.stack(args[3])]
            outputs = grouped_matmul_golden(
                *args,
                split_item=split_item,
                group_type=group_type,
                group_list_type=group_list_type,
                input_dtypes=(xd, wd, bd, sd, "float32", ad, "float32", "int64", pd),
                input_ori_shapes=_reference_shapes(args),
                output_dtypes=(output_name,),
            )
        return [
            torch.from_numpy(np.asarray(v, dtype=np.float32)).to(od)
            if output_name == "bfloat16"
            else torch.from_numpy(np.asarray(v))
            for v in outputs
        ]


__spec__ = {
    "grouped_matmul": "GroupedMatmulTestSpec",
    "aclnnGroupedMatmulV5": "GroupedMatmulV5TestSpec",
    "aclnnGroupedMatmulWeightNz": "GroupedMatmulWeightNzTestSpec",
    "torch_npu.npu_grouped_matmul": "TorchNpuGroupedMatmulTestSpec",
}
