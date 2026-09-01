#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import numpy as np
import torch


__golden__ = {
    "kernel": {"block_attn_res_prepare": "block_attn_res_prepare_golden"},
}

DEFAULT_EPS = 1.0e-6
T_DIM_INDEX = 0
S_DIM_INDEX = 0
N_DIM_INDEX = 1
D_DIM_INDEX = 2
LAST_DIM_INDEX = -1
FLATTENED_SIZE = -1


def block_attn_res_prepare_golden(
    block_res, valid_blocks, pseudo_query, eps=DEFAULT_EPS, **kwargs
):
    """FP32 Phase 1 reference for BlockAttnResPrepare."""
    inputs_are_numpy = not hasattr(block_res, "detach")
    residual = torch.as_tensor(block_res, dtype=torch.float32)
    valid = int(
        torch.as_tensor(valid_blocks, dtype=torch.uint64)
        .reshape(FLATTENED_SIZE)[0]
        .item()
    )
    query = torch.as_tensor(pseudo_query, dtype=torch.float32)
    valid = min(valid, residual.shape[N_DIM_INDEX])
    if valid == 0:
        numerator = torch.zeros(
            (
                query.shape[S_DIM_INDEX],
                residual.shape[T_DIM_INDEX],
                residual.shape[D_DIM_INDEX],
            ),
            dtype=torch.float32,
        )
        logit_max = torch.full(
            (query.shape[S_DIM_INDEX], residual.shape[T_DIM_INDEX]),
            torch.finfo(torch.float32).min,
            dtype=torch.float32,
        )
        exp_sum = torch.zeros_like(logit_max)
        outputs = (numerator, logit_max, exp_sum)
        if inputs_are_numpy:
            return tuple(output.cpu().numpy().astype(np.float32) for output in outputs)
        return outputs
    history = residual[:, :valid, :]
    inv_rms = torch.rsqrt(
        torch.mean(history * history, dim=LAST_DIM_INDEX) + float(eps)
    )
    logits = torch.einsum("sd,tnd->stn", query, history) * inv_rms.unsqueeze(0)
    logit_max = torch.max(logits, dim=LAST_DIM_INDEX).values
    weights = torch.exp(logits - logit_max.unsqueeze(LAST_DIM_INDEX))
    exp_sum = torch.sum(weights, dim=LAST_DIM_INDEX)
    numerator = torch.einsum("stn,tnd->std", weights, history)
    outputs = (
        numerator.to(torch.float32),
        logit_max.to(torch.float32),
        exp_sum.to(torch.float32),
    )
    if inputs_are_numpy:
        return tuple(output.cpu().numpy().astype(np.float32) for output in outputs)
    return outputs
