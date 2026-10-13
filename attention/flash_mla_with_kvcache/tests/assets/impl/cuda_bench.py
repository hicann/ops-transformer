# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""CPU numerical model of DeepSeek FlashMLA's dense CUDA decode path.

Source: deepseek-ai/FlashMLA, c2067be (2025-04-22),
csrc/kernels/{config.h,splitkv_mla.cu}; see ../README_cross_check.md.
Models 64-token online exp2 softmax, input-dtype P, FP32 QK/PV and output
normalisation. Uses one KV split; does not emulate WGMMA/FMA instruction
rounding, warp reductions, hardware scheduling or CUDA non-finite quirks.
"""

import math

import torch

from .flash_mla_with_kvcache_golden import FlashMlaWithKvcacheGolden


def cuda_attention_single(
    q, k, v, scale, tile_size, mask_fn, sink=None, kv_bounds=None
):
    """Bounded-memory, single-split dense CUDA numerical path on CPU."""
    del tile_size
    if sink is not None:
        raise ValueError("Dense FlashMLA CUDA bench does not support attention sinks")
    if q.device.type != "cpu" or k.device.type != "cpu" or v.device.type != "cpu":
        raise ValueError("CUDA bench requires CPU tensors")
    if q.dtype not in (torch.float16, torch.bfloat16) or k.dtype != q.dtype:
        raise ValueError("CUDA bench requires matching FP16/BF16 Q and KV inputs")
    rows, length = q.shape[0], k.shape[0]
    out = torch.zeros((rows, v.shape[-1]), dtype=torch.float32)
    lse = torch.full((rows,), float("inf"), dtype=torch.float32)
    scale_log2 = torch.tensor(scale * math.log2(math.e), dtype=torch.float32)
    for qs in range(0, rows, 64):
        qe = min(qs + 64, rows)
        query = q[qs:qe].float()
        maximum = torch.full((qe - qs, 1), float("-inf"))
        total = torch.zeros_like(maximum)
        acc = torch.zeros((qe - qs, v.shape[-1]), dtype=torch.float32)
        first, stop = (0, length) if kv_bounds is None else kv_bounds(qs, qe)
        first = max(0, first // 64 * 64)
        stop = min(length, max(0, stop))
        for ks in range(first, stop, 64):
            ke = min(ks + 64, length)
            scores = (query @ k[ks:ke].float().T) * scale_log2
            mask = mask_fn(qs, qe, ks, ke)
            if mask is not None:
                scores.masked_fill_(mask, float("-inf"))
            new_max = torch.maximum(maximum, scores.amax(dim=-1, keepdim=True))
            safe_max = torch.where(torch.isneginf(new_max), 0.0, new_max)
            correction = torch.exp2(maximum - safe_max)
            weights = torch.exp2(scores - safe_max)
            total = total * correction + weights.sum(dim=-1, keepdim=True)
            # CUDA accumulates the unrounded sum, but converts P before PV.
            acc = acc * correction + weights.to(q.dtype).float() @ v[ks:ke].float()
            maximum = new_max
        empty = total == 0
        out[qs:qe] = acc / torch.where(empty, 1.0, total)
        row_lse = total.log() + maximum / math.log2(math.e)
        lse[qs:qe] = torch.where(empty, float("inf"), row_lse).squeeze(-1)
    return out, lse


class CudaCpuBench(FlashMlaWithKvcacheGolden):
    """Reuse only layout, page gathering and length/mask adapters."""

    TILE_SIZE = 64
    TILED_CROSSOVER = 0
    _attention_single = staticmethod(cuda_attention_single)

    def forward_golden(self, *args, attn_mask=None, mask_mode=0, **kwargs):
        # The dense CUDA API has only causal=True/False, no arbitrary mask.
        if mask_mode == 0 and attn_mask is not None and bool(attn_mask.any()):
            raise ValueError("CUDA bench cannot represent an arbitrary attention mask")
        return super().forward_golden(
            *args, attn_mask=attn_mask, mask_mode=mask_mode, **kwargs
        )
