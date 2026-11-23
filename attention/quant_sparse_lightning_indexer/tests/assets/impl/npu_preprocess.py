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

"""QSLI metadata arguments inferred from the actual test inputs."""

import numpy as np
import torch

QUANT_MODE_MXFP4 = 5


def get_attribute(kwargs, name, default=None, aliases=()):
    for key in (name, f"pytest_{name}", *aliases):
        value = kwargs.get(key)
        if value is not None:
            return value
    return default


def expect_error(kwargs):
    value = kwargs.get("expect_error", False)
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def get_values(kwargs, name, tensor):
    value = None if expect_error(kwargs) else kwargs.get(f"{name}_values")
    if value is None:
        value = tensor
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.detach().cpu().reshape(-1).tolist()
    return [int(item) for item in value]


def max_sequence(prefix, used, fallback):
    if used and any(used):
        return max(int(value) for value in used)
    if prefix and len(prefix) > 1:
        return max(
            int(prefix[index + 1]) - int(prefix[index])
            for index in range(len(prefix) - 1)
        )
    return int(fallback)


def build_metadata_arguments(
    query, key, topk, quant_mode, layout_q, layout_k, mask_mode, cmp_ratio, kwargs
):
    q_shape = tuple(int(value) for value in query.shape)
    k_shape = tuple(int(value) for value in key.shape)
    num_heads_q = q_shape[2] if layout_q == "BSND" else q_shape[1]
    num_heads_k = k_shape[1] if layout_k == "TND" else k_shape[2]
    head_dim = q_shape[-1]
    is_aclnn_float4 = isinstance(query, np.ndarray) and "float4" in str(query.dtype)
    if int(quant_mode) == QUANT_MODE_MXFP4 and not is_aclnn_float4:
        head_dim *= 2
    cu_q = kwargs.get("cu_seqlens_q")
    cu_k = kwargs.get("cu_seqlens_k")
    seq_q = kwargs.get("seqused_q")
    seq_k = kwargs.get("seqused_k")
    cu_q_values = get_values(kwargs, "cu_seqlens_q", cu_q)
    cu_k_values = get_values(kwargs, "cu_seqlens_k", cu_k)
    seq_q_values = get_values(kwargs, "seqused_q", seq_q)
    seq_k_values = get_values(kwargs, "seqused_k", seq_k)

    batch_size = get_attribute(kwargs, "batch_size")
    if batch_size is None:
        if seq_q_values is not None:
            batch_size = len(seq_q_values)
        elif cu_q_values is not None:
            batch_size = len(cu_q_values) - 1
        elif layout_q == "BSND":
            batch_size = q_shape[0]
        else:
            batch_size = 0

    q_fallback = q_shape[1] if layout_q == "BSND" else q_shape[0]
    if layout_k == "BSND":
        k_fallback = k_shape[1]
    elif layout_k == "TND":
        k_fallback = k_shape[0]
    else:
        k_fallback = int(get_attribute(kwargs, "max_seqlen_k", k_shape[1]))

    return {
        "num_heads_q": int(
            get_attribute(kwargs, "num_heads_q", num_heads_q, ("pytest_q_head_num",))
        ),
        "num_heads_k": int(
            get_attribute(kwargs, "num_heads_k", num_heads_k, ("pytest_k_head_num",))
        ),
        "head_dim": int(get_attribute(kwargs, "head_dim", head_dim)),
        "topk": int(topk),
        "quant_mode": int(quant_mode),
        "cu_seqlens_q": cu_q,
        "cu_seqlens_k": cu_k,
        "seqused_q": seq_q,
        "seqused_k": seq_k,
        "cmp_residual_k": kwargs.get("cmp_residual_k"),
        "batch_size": int(batch_size),
        "max_seqlen_q": int(
            get_attribute(
                kwargs,
                "metadata_max_seqlen_q",
                max_sequence(cu_q_values, seq_q_values, q_fallback),
            )
        ),
        "max_seqlen_k": int(
            get_attribute(
                kwargs,
                "metadata_max_seqlen_k",
                max_sequence(cu_k_values, seq_k_values, k_fallback),
            )
        ),
        "layout_q": str(layout_q),
        "layout_k": str(layout_k),
        "mask_mode": int(mask_mode),
        "cmp_ratio": int(cmp_ratio),
    }
