# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import numpy as np
import torch

from arch92.kda_ttk_ops import load_asset

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
    qshape = tuple(inputs[0].shape)
    rshape = (nq, s) if packed else (b, nq, s)
    bshape = (nv, s) if packed else (b, nv, s)
    token_prefix = (nv, s) if packed else (b, nv, s)
    key_cache = token_prefix + (inputs[0].shape[-1],)
    value_cache = token_prefix + (inputs[2].shape[-1],)
    out_shape = (s, nv) if packed else (b, s, nv)
    out_shape += (inputs[2].shape[-1],)
    shape = [
        out_shape,
        (b, nv, 128, 128),
        chunk + (128, 128),
        token_prefix + (64,),
        token_prefix + (64,),
        key_cache,
        value_cache,
        key_cache,
        key_cache,
        value_cache,
        key_cache,
        qshape,
        rshape,
        qshape,
        rshape,
        bshape,
    ]
    enabled = (
        [
            True,
            f["output_final_state"],
            (not train and f["return_intermediate_states"])
            or (train and f["disable_recompute"]),
            train,
            train,
        ]
        + [train and f["disable_recompute"]] * 6
        + [train and f["use_qk_l2norm_in_kernel"]] * 4
        + [train and f["use_beta_sigmoid_in_kernel"]]
    )
    fp32 = {1, 10, 12, 14, 15}
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
    if meta:
        return
    if outputs[2] is not None:
        # Numeric packing below selects valid chunks, but unused public capacity
        # must also be deterministic across allocations and graph executions.
        states = outputs[2] if packed else outputs[2].flatten(0, 1)
        valid_slots = set()
        start = 0
        for bi, n in enumerate(c["valid_lengths"]):
            base = chunk_base(start, bi) if packed else bi * j
            valid_slots.update(range(base, base + (n + 63) // 64))
            start += n
        for slot in set(range(states.shape[0])) - valid_slots:
            if not torch.equal(states[slot], torch.zeros_like(states[slot])):
                raise AssertionError(f"chunk_states unused slot {slot} is not zero")
    if not forward:
        return
    if train:
        start = 0
        for bi, n in enumerate(c["valid_lengths"]):
            aq_sequence = (
                outputs[3][:, start : start + n] if packed else outputs[3][bi, :, :n]
            )
            ak_sequence = (
                outputs[4][:, start : start + n] if packed else outputs[4][bi, :, :n]
            )
            for ci in range((n + 63) // 64):
                valid = min(64, n - ci * 64)
                rows = slice(ci * 64, ci * 64 + valid)
                aq = aq_sequence[:, rows, :valid]
                ak = ak_sequence[:, rows, :valid]
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
        elif slot == 2:
            parts = []
            start = 0
            for b, n in enumerate(c["valid_lengths"]):
                count = (n + 63) // 64
                states = (
                    value[chunk_base(start, b) : chunk_base(start, b) + count]
                    if c["layout"] == "TND"
                    else value[b, :count]
                )
                parts.append(states.flatten())
                start += n
        elif slot in (3, 4):
            parts = []
            start = 0
            for b, n in enumerate(c["valid_lengths"]):
                sequence = (
                    value[:, start : start + n]
                    if c["layout"] == "TND"
                    else value[b, :, :n]
                )
                for row_start in range(0, n, 64):
                    valid = min(64, n - row_start)
                    parts.append(
                        sequence[:, row_start : row_start + valid, :valid].flatten()
                    )
                start += n
        else:
            # 将头维度在前的标量和缓存转换为切片辅助函数要求的 token 维度在前的布局。
            if slot in (5, 6, 7, 8, 9, 10, 12, 14, 15):
                if c["layout"] == "TND":
                    value = value.transpose(0, 1)
                elif c["layout"] == "BSND":
                    value = value.transpose(1, 2)
            # 定长 out 固定为 BSND，切片辅助函数按输入布局解释各轴。
            if slot == 0 and c["layout"] == "BNSD":
                value = value.transpose(1, 2)
            parts = [x.flatten() for x in sequences(value, c)]
        result.append(torch.cat(parts))
    return result
