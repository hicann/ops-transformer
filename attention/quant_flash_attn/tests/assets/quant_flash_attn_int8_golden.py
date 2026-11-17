# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Independent no-mask/causal FP64 and FP16-numeric references with static INT8 P."""

import torch


def assert_atk_accuracy(golden, actual, reference):
    """Apply the existing full-INT8 ATK double-benchmark acceptance limits."""
    from atk.tasks.post_process.accuracy_compare_v2 import DoubleBenchmarkComparator

    result = DoubleBenchmarkComparator(
        {"dtype": "bfloat16", "max_re_cv": 10.0, "avg_re_cv": 2.0, "rmse_cv": 2.0},
        golden,
        actual,
        reference,
    ).run()
    assert torch.isfinite(actual).all() and result.result == "pass", str(result)


def attention_reference(
    batch,
    vs,
    causal=False,
    rows=None,
    high_precision=False,
    p_scale=120.0,
    win_left=-1,
    win_right=-1,
    softmax_scale=1.0,
):
    q, k, v, qs, ks = batch
    sq, sk, heads, hk = q.size(2), k.size(2), q.size(1), k.size(1)
    rows = list(range(sq)) if rows is None else rows
    q, qs = q[:, :, rows], qs[:, :, rows]
    repeat = heads // hk
    k, v = k.repeat_interleave(repeat, 1), v.repeat_interleave(repeat, 1)
    ks = ks.repeat_interleave(repeat, 1).unsqueeze(-2)
    vs = vs.repeat_interleave(repeat).reshape(1, heads, 1, 1)
    integer_scores = q.int() @ k.int().transpose(-1, -2)
    dtype = torch.float64 if high_precision else torch.float32
    if high_precision:
        score = (
            integer_scores.double()
            * ks.double()
            * softmax_scale
            * qs.double().unsqueeze(-1)
        )
    else:
        # VDEQF16 consumes scale bits [31:13]. Its FP16 numeric benchmark
        # includes that documented Fixpipe conversion before multiplying QK;
        # the FP64 golden above retains the original FP32 scale values.
        kscale = (ks * softmax_scale).contiguous()
        rounded = ((kscale.view(torch.int32) + 0x1000) & -8192).view(torch.float32)
        score = (
            (integer_scores.float() * rounded).half() * qs.half().unsqueeze(-1)
        ).half()
    centers = torch.tensor(rows).reshape(1, 1, -1, 1) + sk - sq
    columns = torch.arange(sk).reshape(1, 1, 1, -1)
    valid = torch.ones_like(columns + centers, dtype=torch.bool)
    if causal:
        valid &= columns <= centers
    if win_left >= 0:
        valid &= columns >= centers - win_left
    if win_right >= 0:
        valid &= columns <= centers + win_right
    maximum = torch.full((*score.shape[:-1], 1), -torch.inf, dtype=dtype)
    denominator = torch.zeros_like(maximum)
    acc = torch.zeros((*score.shape[:-1], v.size(-1)), dtype=dtype)
    # Keep absolute N512 boundaries for online static-P quantization.
    for start in range(0, sk, 512):
        tile = score[..., start : start + 512].to(dtype)
        mask = valid[..., start : start + 512]
        tile_max = tile.masked_fill(~mask, -torch.inf).amax(-1, keepdim=True)
        updated = torch.maximum(maximum, tile_max)
        weight = torch.where(torch.isfinite(maximum), torch.exp(maximum - updated), 0.0)
        if high_precision:
            exponents = torch.exp(tile - updated).masked_fill(~mask, 0.0)
            pq = (exponents * p_scale).round().clamp(0, 127).int()
        else:
            exponents = (
                torch.exp((tile.half() - updated.half()).half())
                .half()
                .masked_fill(~mask, 0.0)
            )
            pq = (exponents * p_scale).half().round().clamp(0, 127).int()
        pv = pq @ v[:, :, start : start + 512].int()
        pv = pv.to(dtype) * vs.to(dtype) / p_scale
        if not high_precision:
            pv = pv.half().float()
        acc = acc * weight + pv
        denominator = denominator * weight + exponents.to(dtype).sum(-1, keepdim=True)
        maximum = updated
    return torch.where(denominator > 0, acc / denominator.clamp_min(1.0e-30), 0.0).to(
        torch.float64 if high_precision else torch.float16
    )


def _inputs(
    lengths_q, lengths_k, heads=4, kv_heads=2, dim=72, seed=None, value_range=(-8, 9)
):
    g = torch.Generator().manual_seed(218 + dim if seed is None else seed)
    batches = []
    for sq, sk in zip(lengths_q, lengths_k):
        q = torch.randint(
            *value_range, (1, heads, sq, dim), generator=g, dtype=torch.int8
        )
        k = torch.randint(
            *value_range, (1, kv_heads, sk, dim), generator=g, dtype=torch.int8
        )
        v = torch.randint(*value_range, k.shape, generator=g, dtype=torch.int8)
        qs = torch.rand(q.shape[:-1], generator=g) * 1.1 + 0.2
        ks = torch.rand(k.shape[:-1], generator=g) * 0.004 + 0.003
        batches.append((q, k, v, qs, ks))
    vs = torch.arange(kv_heads).float() * 0.02 + 0.08
    return batches, vs


def _pack(batches, layout, item):
    tensors = [b[item] for b in batches]
    if layout == "BNSD":
        return torch.cat(tensors, dim=0).contiguous()
    if layout == "BSND":
        return torch.cat(tensors, dim=0).transpose(1, 2).contiguous()
    return torch.cat(
        [t.squeeze(0).transpose(0, 1) for t in tensors], dim=0
    ).contiguous()


def unpack(tensor, layout, lengths):
    """Return per-sequence BNSD tensors, including empty TND members."""
    if layout == "BNSD":
        return list(tensor.split(1))
    if layout == "BSND":
        return list(tensor.transpose(1, 2).split(1))
    return [part.transpose(0, 1).unsqueeze(0) for part in tensor.split(tuple(lengths))]


def sample_rows(length, requested=None):
    if requested is None:
        return list(range(length))
    return (
        sorted(
            {
                min(row, length - 1) if row >= 0 else max(0, length + row)
                for row in requested
            }
        )
        if length
        else []
    )


def lse_reference(
    batch, *, rows=None, causal=False, win_left=-1, win_right=-1, softmax_scale=1.0
):
    """Independent FP64 logsumexp; LSE does not depend on P or V scales."""
    q, k, _, qs, ks = batch
    rows = list(range(q.size(2))) if rows is None else rows
    repeat = q.size(1) // k.size(1)
    scores = q[:, :, rows].double() @ k.double().repeat_interleave(repeat, 1).transpose(
        -1, -2
    )
    scores *= softmax_scale * qs[:, :, rows].double().unsqueeze(-1)
    scores *= ks.double().repeat_interleave(repeat, 1).unsqueeze(-2)
    centers = torch.tensor(rows).reshape(1, 1, -1, 1) + k.size(2) - q.size(2)
    columns = torch.arange(k.size(2)).reshape(1, 1, 1, -1)
    valid = torch.ones_like(centers + columns, dtype=torch.bool)
    if causal:
        valid &= columns <= centers
    if win_left >= 0:
        valid &= columns >= centers - win_left
    if win_right >= 0:
        valid &= columns <= centers + win_right
    return scores.masked_fill(~valid, -torch.inf).logsumexp(-1)
