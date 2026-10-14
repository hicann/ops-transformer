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

"""All-FP32 bench for the FlashAttn TTK cross-check.

三方 bench：复用现有 golden 的布局/长度/页表/mask 适配层（子类继承），
数值路径为独立实现的全 FP32 计算：任何阶段都不做中间 cast、不做 P 舍入、
不引入低精度中间量。
"""

import importlib.util
import sys
from pathlib import Path

import torch


def load_golden_module():
    """Lazily load the sibling golden module by a stable module name."""
    name = "fa_ttk_flash_attn_golden"
    if name in sys.modules:
        return sys.modules[name]
    path = Path(__file__).with_name("golden.py")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot create import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


_flash_attn_golden = load_golden_module()
FlashAttnGolden = _flash_attn_golden.FlashAttnGolden


def fp32_attention_single(q, k, v, mask, scale, sink):
    """One-tile all-FP32 reference: no intermediate casts at any stage."""
    scores = torch.matmul(q, k.t()) * scale
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))

    row_max = scores.amax(dim=-1, keepdim=True)
    row_max = torch.where(torch.isfinite(row_max), row_max, torch.zeros_like(row_max))
    if sink is not None:
        row_max = torch.maximum(row_max, torch.full_like(row_max, sink))

    exp_scores = torch.exp(scores - row_max)
    row_sum = exp_scores.sum(dim=-1, keepdim=True)
    if sink is not None:
        row_sum = row_sum + torch.exp(sink - row_max)

    # P 不 cast、不量化，全程保持 FP32 参与 PV。
    probs = exp_scores / (row_sum + FlashAttnGolden.EPSILON)
    out = torch.matmul(probs, v)

    lse = torch.log(row_sum + FlashAttnGolden.EPSILON) + row_max
    lse = lse.squeeze(-1)
    lse = torch.where(row_sum.squeeze(-1) <= 0, torch.full_like(lse, float("inf")), lse)
    return out, lse


class FlashAttnFp32Bench(FlashAttnGolden):
    """Three-way bench: the golden adapters with an all-FP32 numerical path."""

    def _compute_full_single(self, q, k, v, mask, scale, sink):
        return fp32_attention_single(q, k, v, mask, scale, sink)
