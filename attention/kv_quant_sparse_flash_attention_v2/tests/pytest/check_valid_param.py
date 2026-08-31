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

import torch


def check_valid_param(params):
    (
        _,
        layout_query,
        layout_kv,
        q_type,
        kv_dtype,
        B,
        S1,
        S2,
        N1,
        N2,
        D,
        K,
        scale_value,
        key_quant_mode,
        value_quant_mode,
        sparse_block_size,
        tile_size,
        rope_head_dim,
        sparse_mode,
        attention_mode,
        quant_scale_repo_mode,
        block_size,
        block_num,
        actual_seq_q,
        actual_seq_kv,
    ) = params

    if q_type not in (torch.float16, torch.bfloat16):
        raise ValueError(
            f"q_type should be torch.float16 or torch.bfloat16, but got {q_type}"
        )

    if layout_query not in ("BSND", "TND"):
        raise ValueError(f"layout_query only supports BSND/TND, but got {layout_query}")

    if layout_kv not in ("BSND", "TND", "PA_BSND"):
        raise ValueError(
            f"layout_kv only supports BSND/TND/PA_BSND, but got {layout_kv}"
        )

    if layout_kv != "PA_BSND" and layout_query != layout_kv:
        raise ValueError("layout_query and layout_kv must match in non-PA mode")

    if N2 != 1:
        raise ValueError(f"N2 only supports 1, but got {N2}")

    if N1 not in (1, 2, 4, 8, 16, 32, 48, 64):
        raise ValueError(f"N1 only supports 1/2/4/8/16/32/48/64, but got {N1}")

    if sparse_block_size != 1:
        raise ValueError(
            f"sparse_block_size only supports 1, but got {sparse_block_size}"
        )

    if sparse_mode not in (0, 3):
        raise ValueError(f"sparse_mode only supports 0/3, but got {sparse_mode}")

    if key_quant_mode != 2 or value_quant_mode != 2:
        raise ValueError("key_quant_mode and value_quant_mode must both be 2")

    if tile_size != 128:
        raise ValueError(f"tile_size only supports 128, but got {tile_size}")

    if quant_scale_repo_mode != 1:
        raise ValueError(
            f"quant_scale_repo_mode only supports 1, but got {quant_scale_repo_mode}"
        )

    if attention_mode not in (0, 2):
        raise ValueError(f"attention_mode only supports 0/2, but got {attention_mode}")

    if attention_mode == 2 and rope_head_dim != 64:
        raise ValueError(
            f"rope_head_dim must be 64 when attention_mode=2, but got {rope_head_dim}"
        )

    for name, value in (("B", B), ("S1", S1), ("S2", S2), ("D", D), ("K", K)):
        if value <= 0:
            raise ValueError(f"{name} should be greater than 0, but got {value}")

    if actual_seq_q is not None and len(actual_seq_q) != B:
        raise ValueError(
            f"actual_seq_q length should equal B={B}, but got {len(actual_seq_q)}"
        )
    if actual_seq_kv is not None and len(actual_seq_kv) != B:
        raise ValueError(
            f"actual_seq_kv length should equal B={B}, but got {len(actual_seq_kv)}"
        )

    if layout_kv == "PA_BSND":
        if block_size is None or block_size <= 0:
            raise ValueError(
                f"block_size should be greater than 0 in PA mode, but got {block_size}"
            )
        if block_size % 16 != 0:
            raise ValueError(
                f"block_size should be a multiple of 16 in PA mode, but got {block_size}"
            )
        if block_num is not None and block_num <= 0:
            raise ValueError(
                f"block_num should be greater than 0 in PA mode, but got {block_num}"
            )
