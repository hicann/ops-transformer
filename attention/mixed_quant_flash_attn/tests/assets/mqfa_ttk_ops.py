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

"""TTK metadata-first adapter for the installed SparseFlashMla API."""

from typing import Optional

import torch

from core.npu import run_npu


def call_npu(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_descale: torch.Tensor,
    v_descale: torch.Tensor,
    block_table: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_kv: Optional[torch.Tensor] = None,
    sinks: Optional[torch.Tensor] = None,
    attn_mask: Optional[torch.Tensor] = None,
    metadata: Optional[torch.Tensor] = None,
    quant_compute_mode: int = 1,
    softmax_scale: float = 1.0,
    mask_mode: int = 0,
    win_left: int = -1,
    win_right: int = -1,
    max_seqlen_q: int = -1,
    max_seqlen_kv: int = -1,
    layout_q: str = "BSND",
    layout_kv: str = "PA_BBND",
    layout_attn_out: str = "BSND",
    return_softmax_lse: bool = False,
    batch_size: int = 1,
    num_heads_q: int = 1,
    num_heads_kv: int = 1,
    head_dim: int = 128,
):
    inputs = {
        "q": q,
        "k": k,
        "v": v,
        "k_descale": k_descale,
        "v_descale": v_descale,
        "block_table": block_table,
        "cu_seqlens_q": cu_seqlens_q,
        "seqused_q": seqused_q,
        "seqused_kv": seqused_kv,
        "sinks": sinks,
        "attn_mask": attn_mask,
        "metadata": metadata,
        "quant_compute_mode": quant_compute_mode,
        "softmax_scale": softmax_scale,
        "mask_mode": mask_mode,
        "win_left": win_left,
        "win_right": win_right,
        "max_seqlen_q": max_seqlen_q,
        "max_seqlen_kv": max_seqlen_kv,
        "layout_q": layout_q,
        "layout_kv": layout_kv,
        "layout_attn_out": layout_attn_out,
        "return_softmax_lse": return_softmax_lse,
        "batch_size": batch_size,
        "num_heads_q": num_heads_q,
        "num_heads_kv": num_heads_kv,
        "head_dim": head_dim,
    }
    result = run_npu(inputs)
    return result.get("attn_out"), result.get("softmax_lse")
