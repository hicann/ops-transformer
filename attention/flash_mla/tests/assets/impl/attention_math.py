# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Shared mixed-precision reference primitives (not bitwise kernel emulation).

Cube operands retain input precision; products accumulate into FP32. PyTorch's
ordinary half matmul returns half, so use FP32 matmul on *tile-local* widened
operands to emulate the accumulator without rounding QK or PV to half.
Softmax max/sum and output accumulation remain FP32. The unnormalised exp
weights round to the input dtype BEFORE PV, as in the Ascend vector-to-Cube
handoff. Tile shape/reduction order may differ from the selected kernel.
"""

import torch


def attention_single(
    q,
    k,
    v,
    scale,
    tile_size,
    mask_fn,
    sink=None,
    kv_bounds=None,
    rounding_block_size=None,
):
    sq, skv, dv = q.shape[0], k.shape[0], v.shape[-1]
    out = torch.zeros((sq, dv), dtype=torch.float32, device=q.device)
    lse = torch.full((sq,), float("inf"), dtype=torch.float32, device=q.device)
    for qs in range(0, sq, tile_size):
        qe = min(qs + tile_size, sq)
        qi = q[qs:qe].float()
        acc = torch.zeros((qe - qs, dv), dtype=torch.float32, device=q.device)
        total = torch.full(
            (qe - qs, 1),
            0.0 if sink is None else 1.0,
            dtype=torch.float32,
            device=q.device,
        )
        maximum = torch.full_like(total, float("-inf") if sink is None else sink)
        first, stop = (0, skv) if kv_bounds is None else kv_bounds(qs, qe)
        first = max(0, first // tile_size * tile_size)
        stop = min(skv, max(0, stop))
        for ks in range(first, stop, tile_size):
            ke = min(ks + tile_size, skv)
            scores = (qi @ k[ks:ke].float().T).mul_(scale)
            mask = mask_fn(qs, qe, ks, ke)
            if mask is not None:
                scores.masked_fill_(mask, float("-inf"))
            block = min(rounding_block_size or (ke - ks), ke - ks)
            if (
                rounding_block_size is not None
                and q.dtype in (torch.float16, torch.bfloat16)
                and block < ke - ks
            ):
                # Several kernel KV blocks share one large GEMM. Reproduce
                # their running maxima/rounding with vector operations, then
                # express all PV terms relative to the final maximum.
                width = ke - ks
                padded = (width + block - 1) // block * block
                if padded != width:
                    scores = torch.nn.functional.pad(
                        scores, (0, padded - width), value=float("-inf")
                    )
                blocks = scores.reshape(qe - qs, -1, block)
                running = torch.maximum(
                    blocks.amax(-1, keepdim=True).cummax(1).values, maximum[:, None, :]
                )
                new_max = running[:, -1, :]
                safe = torch.where(torch.isneginf(running), 0.0, running)
                safe_max = torch.where(torch.isneginf(new_max), 0.0, new_max)
                weights = blocks.sub_(safe).exp_()
                rescale = (running - safe_max[:, None, :]).exp_()
                correction = (maximum - safe_max).exp_()
                total.mul_(correction).add_(
                    (weights.sum(-1, keepdim=True) * rescale).sum(1)
                )
                rounded = (
                    weights.to(q.dtype)
                    .float()
                    .mul_(rescale)
                    .reshape(qe - qs, padded)[:, :width]
                )
                del blocks, running, safe, rescale
            else:
                new_max = torch.maximum(maximum, scores.amax(-1, keepdim=True))
                # Entirely masked tiles must not produce exp(-inf - -inf).
                safe_max = torch.where(torch.isneginf(new_max), 0.0, new_max)
                weights = scores.sub_(safe_max).exp_()
                correction = (maximum - safe_max).exp_()
                total.mul_(correction).add_(weights.sum(-1, keepdim=True))
                # Sum uses unrounded weights, PV uses low-precision weights.
                rounded = weights.to(q.dtype).float()
            del scores, weights
            acc.mul_(correction).add_(rounded @ v[ks:ke].float())
            del rounded
            maximum = new_max
        denom = torch.where(total > 0, total, 1.0)
        out[qs:qe] = acc / denom
        rows_lse = (total.log() + maximum).squeeze(-1)
        lse[qs:qe] = torch.where(total.squeeze(-1) == 0, float("inf"), rows_lse)
    return out, lse


def gather_pages(cache, block_table, batch, length, layout):
    """Restore one batch in native dtype, gathering bounded groups of pages.

    Validate only referenced entries: unused -1 padding is legal. Moving the
    row once to CPU avoids one NPU scalar synchronisation per page.
    """
    bs = (
        cache.shape[3]
        if layout == "PA_NZ"
        else cache.shape[1 if layout == "PA_BBND" else 2]
    )
    heads = cache.shape[2] if layout == "PA_BBND" else cache.shape[1]
    width = cache.shape[2] * cache.shape[4] if layout == "PA_NZ" else cache.shape[-1]
    if length < 0:
        raise ValueError("KV length must be nonnegative")
    count = (length + bs - 1) // bs
    if (
        block_table.ndim != 2
        or batch >= block_table.shape[0]
        or count > block_table.shape[1]
    ):
        raise ValueError("block_table does not cover the requested KV length")
    ids = block_table[batch, :count].to(device="cpu", dtype=torch.int64)
    if ((ids < 0) | (ids >= cache.shape[0])).any():
        raise ValueError("referenced block_table index is outside the cache")
    result = torch.empty(
        (1, heads, length, width), dtype=cache.dtype, device=cache.device
    )
    ids = ids.to(cache.device)
    for start in range(0, count, 128):
        pages = cache.index_select(0, ids[start : start + 128])
        if layout == "PA_NZ":
            pages = pages.permute(1, 0, 3, 2, 4).reshape(heads, -1, width)
        elif layout == "PA_BBND":
            pages = pages.permute(2, 0, 1, 3).reshape(heads, -1, width)
        else:
            pages = pages.permute(1, 0, 2, 3).reshape(heads, -1, width)
        offset = start * bs
        valid = min(pages.shape[1], length - offset)
        result[0, :, offset : offset + valid] = pages[:, :valid]
    return result
