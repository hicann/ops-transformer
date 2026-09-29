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

import logging
import os
import sys

import torch

_ASSETS_DIR = os.path.dirname(os.path.abspath(__file__))
_ASSETS_ROOT = os.path.dirname(_ASSETS_DIR)
if _ASSETS_ROOT not in sys.path:
    sys.path.insert(0, _ASSETS_ROOT)
import quant_flash_attn_mxfp4_golden as golden_mod

logger = logging.getLogger(__name__)


def _apply_golden_globals(attrs):
    golden_mod._apply_golden_globals(attrs)
    golden_mod._apply_kwargs_globals(attrs)


def cpu_qfa_mxfp4(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_descale: torch.Tensor,
    k_descale: torch.Tensor,
    v_descale: torch.Tensor,
    quant_mode: int,
    block_table: torch.Tensor = None,
    p_scale: torch.Tensor = None,
    cu_seqlens_q: torch.Tensor = None,
    cu_seqlens_kv: torch.Tensor = None,
    seqused_q: torch.Tensor = None,
    seqused_kv: torch.Tensor = None,
    sinks: torch.Tensor = None,
    attn_mask: torch.Tensor = None,
    metadata: torch.Tensor = None,
    softmax_scale: float = 1.0,
    mask_mode: int = 0,
    win_left: int = -1,
    win_right: int = -1,
    max_seqlen_q: int = -1,
    max_seqlen_kv: int = -1,
    layout_q: str = "BSND",
    layout_q_descale: str = "BSND",
    layout_kv: str = "BSND",
    layout_out: str = "BSND",
    return_softmax_lse: bool = False,
    **kwargs,
):
    attrs = dict(kwargs)
    attrs.setdefault("quant_mode", quant_mode)
    attrs.setdefault("softmax_scale", softmax_scale)
    attrs.setdefault("mask_mode", mask_mode)
    attrs.setdefault("win_left", win_left)
    attrs.setdefault("win_right", win_right)
    attrs.setdefault("max_seqlen_q", max_seqlen_q)
    attrs.setdefault("max_seqlen_kv", max_seqlen_kv)
    attrs.setdefault("layout_q", layout_q)
    attrs.setdefault("layout_q_descale", layout_q_descale)
    attrs.setdefault("layout_kv", layout_kv)
    attrs.setdefault("layout_out", layout_out)
    attrs.setdefault("return_softmax_lse", return_softmax_lse)
    _apply_golden_globals(attrs)

    golden_mod._inject_physical_s_override(q, v, layout_q, layout_kv)
    try:
        data_dict = golden_mod.generate_data()
    finally:
        golden_mod._clear_physical_s_override()

    cpu_out, cpu_lse = golden_mod.cpu_mxfp4_golden(data_dict)
    if return_softmax_lse and cpu_lse is not None:
        return [cpu_out, cpu_lse]
    return [cpu_out, torch.empty(0, dtype=torch.float32)]
