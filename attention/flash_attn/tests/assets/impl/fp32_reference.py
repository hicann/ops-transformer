#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Independent FP32 arithmetic for FlashAttn three-way comparison."""

import torch
from .golden import FlashAttnGolden


class FP32Reference(FlashAttnGolden):
    def forward_golden(self, q, k, v, **kwargs):
        return super().forward_golden(q.float(), k.float(), v.float(), **kwargs)

    def _attention(self, q, k, v, scale, tile_size, mask_fn, sink, dropout=None):
        rows, length = q.shape[0], k.shape[0]
        output = torch.zeros((rows, v.shape[-1]), dtype=torch.float32, device=q.device)
        lse = torch.full((rows,), float("inf"), dtype=torch.float32, device=q.device)
        for start in range(0, rows, tile_size):
            stop = min(start + tile_size, rows)
            maximum = torch.full((stop - start, 1), float("-inf"), device=q.device)
            denominator = torch.zeros_like(maximum)
            numerator = torch.zeros((stop - start, v.shape[-1]), device=q.device)
            for begin in range(0, length, tile_size):
                end = min(begin + tile_size, length)
                scores = torch.mm(q[start:stop], k[begin:end].T) * scale
                mask = mask_fn(start, stop, begin, end)
                if mask is not None:
                    scores = scores.masked_fill(mask, float("-inf"))
                next_max = torch.maximum(maximum, scores.amax(dim=1, keepdim=True))
                safe_max = torch.where(torch.isneginf(next_max), 0.0, next_max)
                alpha = torch.exp(maximum - safe_max)
                exponentials = torch.exp(scores - safe_max)
                weights = (
                    exponentials
                    if dropout is None
                    else dropout(exponentials, start, stop, begin, end)
                )
                numerator = alpha * numerator + torch.mm(weights, v[begin:end])
                denominator = alpha * denominator + exponentials.sum(
                    dim=1, keepdim=True
                )
                maximum = next_max
            empty = denominator.squeeze(-1) == 0
            normalizer = denominator.log().squeeze(-1) + maximum.squeeze(-1)
            values = numerator / torch.where(denominator == 0, 1.0, denominator)
            if sink is not None:
                combined = torch.logaddexp(normalizer, sink)
                values *= torch.exp(normalizer - combined)[:, None]
                normalizer = combined
            output[start:stop] = values.masked_fill(empty[:, None], 0.0)
            lse[start:stop] = (
                normalizer
                if sink is not None
                else normalizer.masked_fill(empty, float("inf"))
            )
        return output, lse
