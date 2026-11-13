#!/usr/bin/python3
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

"""Only materialize explicit *_values for standalone metadata tests.

TTK already injects attributes named after tensor parameters. With no sidecar
value, preserve those tensors (and manual-data inputs) unchanged. This module
performs no dtype/rank/value validation on behalf of the operator.
"""

import numpy as np
import torch


def expect_error(kwargs):
    value = kwargs.get("expect_error", False)
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def generate_quant_flash_mla_metadata_inputs(
    cache_seqlens,
    num_heads_q,
    num_heads_kv,
    quant_mode,
    *,
    cu_seqlens_q=None,
    seqused_q=None,
    **kwargs,
):
    if expect_error(kwargs):
        return None

    del num_heads_q, num_heads_kv, quant_mode
    # Core-slot passthrough never permits dropping an explicit vector request.
    policy = kwargs.get("input_mismatch_policy", "error")
    if policy not in ("passthrough", "error"):
        raise ValueError("input_mismatch_policy must be 'passthrough' or 'error'")
    for name, target in (
        ("cache_seqlens", cache_seqlens),
        ("cu_seqlens_q", cu_seqlens_q),
        ("seqused_q", seqused_q),
    ):
        values = kwargs.get(f"{name}_values")
        if target is None or values is None:
            continue
        if isinstance(target, torch.Tensor):
            source = torch.as_tensor(values, dtype=target.dtype, device=target.device)
            if tuple(source.shape) != tuple(target.shape):
                raise ValueError(
                    f"[INPUTS] metadata {name} slot shape {tuple(target.shape)} != explicit values {tuple(source.shape)}"
                )
            if not torch.equal(target, source):
                target.copy_(source)
        else:
            source = np.asarray(values, dtype=target.dtype)
            if source.shape != target.shape:
                raise ValueError(
                    f"[INPUTS] metadata {name} slot shape {target.shape} != explicit values {source.shape}"
                )
            if not np.array_equal(target, source):
                np.copyto(target, source)
