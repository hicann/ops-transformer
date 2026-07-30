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

import math

import torch


SUPPORTED_SPARSE_PATTERNS = {"sequential", "dense", "reverse", "tail", "random", "empty", "empty_tail", "causal"}
SUPPORTED_BLOCK_TABLE_PATTERNS = {"sequential", "reverse", "random"}


def _require_positive(params, key):
    value = params.get(key)
    if isinstance(value, float) and value.is_integer():
        params[key] = int(value)
        value = params[key]
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{key} should be a positive int, got {value}")


def check_valid_param(params):
    if not isinstance(params, dict):
        raise ValueError("quant_block_sparse_attn params should be a dict aligned with op def inputs and attrs")

    params = dict(params)
    params.setdefault("sparse_q_block_size", 128)
    params.setdefault("sparse_kv_block_size", 128)
    params.setdefault("layout_kv", "PA_BNSD")
    params.setdefault("layout_sparse_indices", "B_N_Qb_Kb")
    params.setdefault("layout_out", "TND")
    params.setdefault("quant_mode", 1)
    params.setdefault("mask_mode", 3)
    params.setdefault("output_dtype", torch.bfloat16)
    params.setdefault("sparse_pattern", "sequential")
    params.setdefault("block_table_pattern", "sequential")
    params.setdefault("pa_block_padding_bytes", 0)
    params.setdefault("p_scale_value", 1.0)
    params["sparse_count"] = params.get("sparse_count") or math.ceil(params["S2"] / params["sparse_kv_block_size"])
    params["softmax_scale"] = params.get("softmax_scale") or (1.0 / math.sqrt(params["D"]))
    for key in ("quant_mode", "mask_mode"):
        if isinstance(params.get(key), float) and params[key].is_integer():
            params[key] = int(params[key])

    for key in ("B", "S1", "S2", "N1", "N2", "D", "sparse_q_block_size", "sparse_kv_block_size", "sparse_count"):
        _require_positive(params, key)

    pa_block_padding_bytes = params["pa_block_padding_bytes"]
    if not isinstance(pa_block_padding_bytes, int) or pa_block_padding_bytes != 0:
        raise ValueError(f"pa_block_padding_bytes should be 0 for segmented KV cache, got {pa_block_padding_bytes}")

    if params["layout_q"] not in ("TND", "NTD"):
        raise ValueError(f"unsupported layout_q: {params['layout_q']}")

    if params.get("layout_kv") != "PA_BNSD":
        raise ValueError(f"unsupported layout_kv: {params.get('layout_kv')}")

    if params.get("layout_sparse_indices") != "B_N_Qb_Kb":
        raise ValueError(f"unsupported layout_sparse_indices: {params.get('layout_sparse_indices')}")

    if params["N1"] % params["N2"] != 0:
        raise ValueError(f"N1 should be divisible by N2, got N1={params['N1']}, N2={params['N2']}")

    if params["sparse_q_block_size"] != 128 or params["sparse_kv_block_size"] != 128:
        raise ValueError("sparse_q_block_size and sparse_kv_block_size should both be 128 for the current kernel")

    if params.get("quant_mode", 1) != 1:
        raise ValueError(f"unsupported quant_mode: {params.get('quant_mode')}")

    if params.get("mask_mode", 3) not in (0, 3):
        raise ValueError(f"unsupported mask_mode: {params.get('mask_mode')}")

    if params.get("output_dtype", torch.bfloat16) not in (
        torch.bfloat16, "torch.bfloat16", "bfloat16", "bf16"
    ):
        raise ValueError(f"unsupported output_dtype: {params.get('output_dtype')}")

    if params["sparse_pattern"] not in SUPPORTED_SPARSE_PATTERNS:
        raise ValueError(f"unsupported sparse_pattern: {params['sparse_pattern']}")

    if params["block_table_pattern"] not in SUPPORTED_BLOCK_TABLE_PATTERNS:
        raise ValueError(f"unsupported block_table_pattern: {params['block_table_pattern']}")

    if not isinstance(params["p_scale_value"], (int, float)) or params["p_scale_value"] <= 0:
        raise ValueError(f"p_scale_value should be a positive number, got {params['p_scale_value']}")

    if params.get("softmax_scale") is None:
        raise ValueError("softmax_scale should not be None")
