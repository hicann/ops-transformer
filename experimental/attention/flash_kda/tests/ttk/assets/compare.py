# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import numpy as np
import torch

from kda_ttk_ops import load_asset

SLOTS = range(3)
FORWARD_SLOTS = range(16)


def numerical_slots(c, *, forward=False):
    """Inference compares only public results, regardless of the entry point."""
    return FORWARD_SLOTS if forward and not c["inference"] else SLOTS


def check_arity(outputs, *, forward=False):
    slots = FORWARD_SLOTS if forward else SLOTS
    if not isinstance(outputs, (tuple, list)) or len(outputs) != len(slots):
        entry = "flash_kda_fwd" if forward else "flash_kda"
        raise AssertionError(
            f"{entry}: expected fixed {len(slots)} return slots, got {type(outputs)}"
        )


def empty():
    return torch.empty(0, dtype=torch.float32)


def effective(c):
    return load_asset("inputs").effective(c)


def sequences(*args, **kwargs):
    return load_asset("inputs").sequences(*args, **kwargs)


def compare_contracts(actual, expected, reference, *, slots=SLOTS):
    """Finite values and design-specific tolerances, independent of L2."""
    if not len(actual) == len(expected) == len(slots):
        raise AssertionError("output/reference arity mismatch")
    reports = []
    for slot, a, e in zip(slots, actual, expected):
        a, e = (x.detach().cpu().double().numpy() for x in (a, e))
        tol = 1e-3 if slot in (11, 13) else 1e-6 if slot in (12, 14, 15) else None
        finite = np.isfinite(a).all() and np.isfinite(e).all()
        passed = (
            a.shape == e.shape
            and finite
            and (tol is None or np.allclose(a, e, atol=tol, rtol=tol, equal_nan=False))
        )
        error = float(np.max(np.abs(a - e))) if a.shape == e.shape and a.size else 0.0
        reports.append(
            {
                "slot": slot,
                "reference": reference,
                "pass": bool(passed),
                "precision": (
                    "NOT_APPLICABLE"
                    if not a.size and not e.size
                    else "PASS"
                    if passed
                    else "FAIL"
                ),
                "error_info": f"{reference} slot={slot}, shape={a.shape}/{e.shape}, atol=rtol={tol}, max_abs={error}, finite={finite}",
                "metrics": {"max_abs": error, "atol": tol, "rtol": tol},
            }
        )
    return reports


def chunk_base(start, b):
    return start // 64 + b


def check(outputs, inputs, c, *, meta=False, forward=False):
    check_arity(outputs, forward=forward)
    f = effective(c)
    train = not c["inference"]
    b, nq, nv, s = c["batch_size"], c["nheads_qk"], c["nheads_v"], c["seqlens"]
    packed = c["layout"] == "TND"
    j = (s + 63) // 64 + (b if packed else 0)
    chunk = (j, nv) if packed else (b, j, nv)
    qshape = (s, nq, 128) if packed else (b, nq, s, 128)
    bshape = (s, nv) if packed else (b, s, nv)
    token_prefix = (
        (s, nv) if packed else ((b, nv, s) if c["layout"] == "BNSD" else (b, s, nv))
    )
    key_cache = token_prefix + (inputs[0].shape[-1],)
    value_cache = token_prefix + (inputs[2].shape[-1],)
    shape = [
        tuple(inputs[2].shape),
        (b, nv, 128, 128),
        chunk + (128, 128),
        chunk + (64, 64),
        chunk + (64, 64),
        key_cache,
        value_cache,
        key_cache,
        key_cache,
        value_cache,
        chunk + (64, 128),
        qshape,
        qshape[:-1],
        qshape,
        qshape[:-1],
        bshape,
    ]
    enabled = (
        [
            True,
            f["output_final_state"],
            not train and f["return_intermediate_states"],
            train,
            train,
        ]
        + [train and f["disable_recompute"]] * 6
        + [train and f["use_qk_l2norm_in_kernel"]] * 4
        + [train and f["use_beta_sigmoid_in_kernel"]]
    )
    fp32 = {1, 2, 10, 12, 14, 15}
    for i, (value, on) in enumerate(zip(outputs, enabled)):
        if not on:
            if value is not None:
                raise AssertionError(f"{c['id']} slot {i}: expected None")
        else:
            dtype = torch.float32 if i in fp32 else torch.bfloat16
            if (
                not isinstance(value, torch.Tensor)
                or tuple(value.shape) != shape[i]
                or value.dtype != dtype
            ):
                raise AssertionError(
                    f"{c['id']} slot {i}: expected Tensor {shape[i]} {dtype}, got {getattr(value, 'shape', None)} {getattr(value, 'dtype', None)}"
                )
    if meta or not forward:
        return
    if train:
        start = 0
        for bi, n in enumerate(c["valid_lengths"]):
            base = chunk_base(start, bi) if packed else 0
            for ci in range((n + 63) // 64):
                valid = min(64, n - ci * 64)
                aq = outputs[3][base + ci] if packed else outputs[3][bi, ci]
                ak = outputs[4][base + ci] if packed else outputs[4][bi, ci]
                aq = aq[:, :valid, :valid]
                ak = ak[:, :valid, :valid]
                if not torch.equal(aq.triu(1), torch.zeros_like(aq)):
                    raise AssertionError("Aqk valid upper triangle is not zero")
                if not torch.equal(
                    ak.diagonal(dim1=-2, dim2=-1),
                    torch.ones_like(ak.diagonal(dim1=-2, dim2=-1)),
                ):
                    raise AssertionError("Akk valid diagonal is not one")
            start += n


def pack(outputs, c, *, forward=False):
    check_arity(outputs, forward=forward)
    result = []
    for slot in numerical_slots(c, forward=forward):
        value = outputs[slot]
        if value is None:
            result.append(empty().to(outputs[0].device))
            continue
        if slot == 1:
            parts = [value.flatten()]
        elif slot in (2, 3, 4, 10):
            parts = []
            start = 0
            for b, n in enumerate(c["valid_lengths"]):
                count = (n + 63) // 64
                v = (
                    value[chunk_base(start, b) : chunk_base(start, b) + count]
                    if c["layout"] == "TND"
                    else value[b, :count]
                )
                if slot in (3, 4, 10):
                    for ci, block in enumerate(v):
                        valid = min(64, n - ci * 64)
                        active = (
                            block[:, :valid, :valid]
                            if slot in (3, 4)
                            else block[:, :valid]
                        )
                        parts.append(active.flatten())
                else:
                    parts.append(v.flatten())
                start += n
        elif slot == 15:
            if c["layout"] == "TND":
                parts = [value.flatten()]
            else:
                parts = [
                    value[b, :n].flatten() for b, n in enumerate(c["valid_lengths"])
                ]
        else:
            parts = [
                x.flatten() for x in sequences(value, c, norm=slot in (11, 12, 13, 14))
            ]
        result.append(torch.cat(parts))
    return result
