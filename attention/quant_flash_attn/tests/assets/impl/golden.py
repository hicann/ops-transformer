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
from typing import List

import torch

_ASSETS_DIR = os.path.dirname(os.path.abspath(__file__))
_TESTS_DIR = os.path.join(_ASSETS_DIR, "..", "pytest", "fia_fullquant_mxfp8_test")
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)
from common import fia_fullquant_mxfp8_golden as golden_mod

logger = logging.getLogger(__name__)

__golden__ = {
    "e2e": {
        "qfa_mxfp8_wrapper.npu_qfa_mxfp8": "cpu_qfa_mxfp8"
    }
}


def _apply_golden_globals(attrs):
    """把 case 属性注入 golden 模块全局变量。"""
    mapping = {
        "B": "B", "N_q": "N_q", "N_kv": "N_kv", "D": "D",
        "enable_pa": "ENABLE_PA", "kv_cache_layout": "KV_CACHE_LAYOUT",
        "block_size": "BLOCK_SIZE", "mask_mode": "SPARSE_MODE",
        "q_scale_layout": "Q_SCALE_LAYOUT",
        "enable_lse": "ENABLE_LSE", "input_layout": "INPUT_LAYOUT",
        "is_contiguous": "IS_CONTIGUOUS", "device_id": "DEVICE_ID",
        "graph_path": "GRAPH_PATH",
        "cu_seqlens_q": "CU_SEQLENS_Q", "cu_seqlens_kv": "CU_SEQLENS_KV",
        "seqused_q": "SEQUSED_Q", "seqused_kv": "SEQUSED_KV",
        "max_seqlen_q": "MAX_SEQLEN_Q", "max_seqlen_kv": "MAX_SEQLEN_KV",
    }
    for attr_key, golden_key in mapping.items():
        if attr_key in attrs:
            setattr(golden_mod, golden_key, attrs[attr_key])
    setattr(golden_mod, "FP8_DTYPE", torch.float8_e4m3fn)
    setattr(golden_mod, "QUANT_GROUP_SIZE", 32)
    p_scale = attrs.get("p_scale_value", 1.0)
    if isinstance(p_scale, torch.Tensor):
        p_scale = float(p_scale.item())
    setattr(golden_mod, "P_SCALE", float(p_scale))


def _cu_seqlens_to_actual(cu_seqlens):
    """cu_seqlens → actual_seq (差分还原)。

    cu_seqlens = [0, 3, 7, 10] → actual_seq = [3, 4, 3]
    """
    if cu_seqlens is None or len(cu_seqlens) < 2:
        return []
    return [cu_seqlens[i + 1] - cu_seqlens[i] for i in range(len(cu_seqlens) - 1)]


def cpu_qfa_mxfp8(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                   dequant_scale_q: torch.Tensor, dequant_scale_k: torch.Tensor, v_descale: torch.Tensor,
                   p_scale: torch.Tensor, block_table: torch.Tensor,
                   *,
                   B: int, N_q: int, N_kv: int, D: int,
                   cu_seqlens_q: List[int], cu_seqlens_kv: List[int],
                   seqused_q: List[int], seqused_kv: List[int],
                   max_seqlen_q: int, max_seqlen_kv: int,
                   enable_pa: bool, kv_cache_layout: str, block_size: int,
                   mask_mode: int, q_scale_layout: str,
                   quant_mode: int = 1,
                   enable_lse: bool = False, graph_path: int = 0,
                   input_layout: str = "TND",
                   is_contiguous: bool = True, device_id: int = 0,
                   softmax_scale: float = None,
                   data_range_q: float = 1.0, data_range_k: float = 1.0, data_range_v: float = 1.0,
                   **kwargs):
    """CPU golden:从 golden_mod 缓存取真实 FP8 数据,调 cpu_mxfp8_golden 算参考输出。"""
    # 从 golden_mod 取 customize_inputs 缓存的真实 FP8 数据
    cached = getattr(golden_mod, "_cached_mxfp8_inputs", None)
    if cached is not None:
        q, k, v, dequant_scale_q, dequant_scale_k, v_descale, p_scale, block_table = cached[:8]

    # cu_seqlens → actual_seq (差分还原)
    actual_seq_q = _cu_seqlens_to_actual(cu_seqlens_q)
    # PA 模式下 cu_seqlens_kv 可能为空，用 seqused_kv 推导
    if cu_seqlens_kv is not None and len(cu_seqlens_kv) > 1:
        actual_seq_kv = _cu_seqlens_to_actual(cu_seqlens_kv)
    elif seqused_kv is not None and len(seqused_kv) > 0:
        actual_seq_kv = list(seqused_kv)
    else:
        actual_seq_kv = []

    # cu_seqlens_q/kv 需要转为 list(传入时可能是 tuple 或其他类型)
    cu_seqlens_q_list = list(cu_seqlens_q) if cu_seqlens_q is not None else [0]
    cu_seqlens_kv_list = list(cu_seqlens_kv) if cu_seqlens_kv is not None else [0]

    _apply_golden_globals({
        "B": B, "N_q": N_q, "N_kv": N_kv, "D": D,
        "cu_seqlens_q": cu_seqlens_q_list, "cu_seqlens_kv": cu_seqlens_kv_list,
        "seqused_q": list(seqused_q) if seqused_q is not None else None,
        "seqused_kv": list(seqused_kv) if seqused_kv is not None else None,
        "max_seqlen_q": max_seqlen_q, "max_seqlen_kv": max_seqlen_kv,
        "enable_pa": enable_pa, "kv_cache_layout": kv_cache_layout,
        "block_size": block_size, "mask_mode": mask_mode,
        "q_scale_layout": q_scale_layout,
        "enable_lse": enable_lse, "input_layout": input_layout,
        "is_contiguous": is_contiguous, "device_id": device_id,
        "graph_path": graph_path,
        "p_scale_value": p_scale,
    })

    cpu_out, cpu_lse = golden_mod.cpu_mxfp8_golden(
        q, k, v,
        dequant_scale_q, dequant_scale_k, v_descale, p_scale,
        actual_seq_q, actual_seq_kv,
        softmax_scale=softmax_scale,
    )

    compare_layout = "TND" if enable_pa else input_layout
    cpu_out_aligned = golden_mod.convert_q_bnsd_to_layout(cpu_out, actual_seq_q, compare_layout,
                                                          cu_seqlens=cu_seqlens_q if enable_pa else None)

    if enable_lse:
        cpu_lse_aligned = golden_mod.convert_q_bnsd_to_layout(cpu_lse, actual_seq_q, compare_layout,
                                                              cu_seqlens=cu_seqlens_q if enable_pa else None)
        return [cpu_out_aligned, cpu_lse_aligned]
    return [cpu_out_aligned]
