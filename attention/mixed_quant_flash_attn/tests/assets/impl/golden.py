#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

"""TTK e2e golden for mixed_quant_flash_attn.

直接调用 pytest 的 core.cpu.tforward 作为 CPU golden。
TTK 传入的 tensor 可能是 numpy array，需转为 torch tensor 后传给 tforward。
"""

__golden__ = {
    "e2e": {
        "torch.ops.cann_ops_transformer.mixed_quant_flash_attn": "cpu_mixed_quant_flash_attn"
    }
}

import numpy as np
import torch

from core.cpu import run_cpu_golden


def _to_torch(val, dtype=None):
    """把 numpy/torch/scalar 转为 torch tensor。
    处理 numpy float8_e8m0 和 bfloat16 等非原生 dtype（通过 uint16/uint8 view）。
    """
    if val is None:
        return None
    if torch.is_tensor(val):
        return val.detach().cpu()
    if isinstance(val, np.ndarray):
        dt = str(val.dtype)
        if dt.startswith("float8"):
            return torch.from_numpy(val.view(np.uint8).copy()).view(
                torch.float8_e8m0fnu
            )
        if dt == "bfloat16":
            return torch.from_numpy(val.view(np.uint16).copy()).view(torch.bfloat16)
        return torch.from_numpy(val.copy())
    if isinstance(val, (list, tuple)):
        return torch.tensor(val, dtype=dtype or torch.int32)
    return torch.tensor(val)


def cpu_mixed_quant_flash_attn(*args, **kwargs):
    return run_mixed_quant_flash_attn(*args, **kwargs)


def run_mixed_quant_flash_attn(
    q,
    k,
    v,
    k_descale,
    v_descale,
    block_table=None,
    cu_seqlens_q=None,
    seqused_q=None,
    seqused_kv=None,
    sinks=None,
    attn_mask=None,
    metadata=None,
    quant_mode=1,
    softmax_scale=1.0,
    mask_mode=0,
    win_left=-1,
    win_right=-1,
    max_seqlen_q=-1,
    max_seqlen_kv=-1,
    layout_q="BSND",
    layout_kv="PA_BBND",
    layout_attn_out="BSND",
    return_softmax_lse=False,
    batch_size=1,
    num_heads_q=1,
    num_heads_kv=1,
    head_dim=128,
    **kwargs,
):
    """TTK golden adapter: 调用 pytest tforward 计算 CPU golden。"""
    mode = kwargs.get("mode")
    if mode is None:
        mode = "cpu_high" if kwargs.get("golden_mode") == "Promote" else "cpu"
    # numpy → torch 转换
    input_kwargs = {
        "q": _to_torch(q, torch.bfloat16),
        "k": _to_torch(k, torch.uint8),
        "v": _to_torch(v, torch.uint8),
        "k_descale": _to_torch(k_descale),
        "v_descale": _to_torch(v_descale),
        "block_table": _to_torch(block_table, torch.int32),
        "cu_seqlens_q": _to_torch(cu_seqlens_q, torch.int32),
        "seqused_q": _to_torch(seqused_q, torch.int32),
        "seqused_kv": _to_torch(seqused_kv, torch.int32),
        "sinks": _to_torch(sinks, torch.float32),
        "attn_mask": _to_torch(attn_mask, torch.int8),
        "quant_mode": int(quant_mode),
        "softmax_scale": float(softmax_scale),
        "mask_mode": int(mask_mode),
        "win_left": int(win_left),
        "win_right": int(win_right),
        "max_seqlen_q": int(max_seqlen_q),
        "max_seqlen_kv": int(max_seqlen_kv),
        "layout_q": layout_q,
        "layout_kv": layout_kv,
        "layout_attn_out": layout_attn_out,
        "return_softmax_lse": bool(return_softmax_lse),
        "sinner": int(kwargs.get("sinner", 512)),
        "souter": int(kwargs.get("souter", 32)),
    }

    result = run_cpu_golden(input_kwargs, mode=mode)
    return result["attn_out"], result["softmax_lse"]
