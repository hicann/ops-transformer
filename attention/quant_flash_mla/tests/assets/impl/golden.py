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


"""TTK golden plugin for quant_flash_mla.

Regenerates BNSD data via generate_data() (same seeds → same data as inputs),
calls the CPU golden reference, converts output to target layout, and returns
(attention_out, lse_out) for comparison.
"""

import logging
import math
import os
import sys

import numpy as np
import torch

_ASSETS_DIR = os.path.dirname(os.path.abspath(__file__))
_GOLDEN_DIR = os.path.join(_ASSETS_DIR, "..")  # assets/
if _GOLDEN_DIR not in sys.path:
    sys.path.insert(0, _GOLDEN_DIR)
# golden 模块依赖 pytest/common/ 下的 result_compare_method
_COMMON_DIR = os.path.join(_ASSETS_DIR, "..", "..", "pytest", "common")
if _COMMON_DIR not in sys.path:
    sys.path.insert(0, _COMMON_DIR)

import qfmla_golden as golden_mod

logger = logging.getLogger(__name__)

_KV_LAYOUT_REVERSE = {
    "PA_BNBD": "BnNBsD",
    "PA_BBND": "BnBsND",
    "PA_NZ": "NZ",
}


def _set_golden_globals(
    quant_mode,
    mask_mode,
    layout_q,
    layout_kv,
    layout_out,
    max_seqlen_q,
    max_seqlen_kv,
    head_dim_v,
    return_softmax_lse,
    softmax_scale,
    **kwargs,
):
    """Set golden module globals from op params and sidecar kwargs."""
    B = kwargs.get("B", 1)
    N_q = kwargs.get("N_q", 6)
    N_kv = kwargs.get("N_kv", 1)
    D = kwargs.get("D", 512)
    D_V = kwargs.get("D_V", 512)
    D_rope = kwargs.get("D_rope", 64)
    block_size = kwargs.get("block_size", 128)
    seqused_q_vals = kwargs.get("seqused_q_values", [1])
    cache_seqlens_vals = kwargs.get("cache_seqlens_values", [1024])
    kv_cache_layout = kwargs.get("kv_cache_layout") or _KV_LAYOUT_REVERSE.get(
        layout_kv, "BnNBsD"
    )
    input_layout = kwargs.get("input_layout", layout_q)
    output_layout = kwargs.get("output_layout", layout_out)

    golden_mod.B = B
    golden_mod.N_q = N_q
    golden_mod.N_kv = N_kv
    golden_mod.D = D
    golden_mod.D_V = D_V
    golden_mod.D_rope = D_rope
    golden_mod.QFMLA_QK_HEAD_DIM = D + D_rope
    golden_mod.QFMLA_V_HEAD_DIM = D_V
    golden_mod.QFMLA_QUANT_MODE = quant_mode
    golden_mod.SEQUSED_Q = list(seqused_q_vals)
    golden_mod.CACHE_SEQLENS = list(cache_seqlens_vals)
    golden_mod.ENABLE_PA = True
    golden_mod.ENABLE_LSE = return_softmax_lse
    golden_mod.GOLDEN_MODE = True
    golden_mod.BLOCK_SIZE = block_size
    golden_mod.MASK_MODE = mask_mode
    golden_mod.INPUT_LAYOUT = input_layout
    golden_mod.OUTPUT_LAYOUT = output_layout
    golden_mod.KV_CACHE_LAYOUT = kv_cache_layout
    golden_mod.P_SCALE = float(kwargs.get("p_scale", 1.0))
    golden_mod.GRAPH_PATH = kwargs.get("graph_path", 0)
    golden_mod.NUM_BLOCKS = kwargs.get("num_blocks", 0)
    golden_mod.SCALE_VALUE = softmax_scale
    golden_mod.MAX_SEQLEN_Q = max_seqlen_q
    golden_mod.MAX_SEQLEN_KV = max_seqlen_kv

    if kwargs.get("q_data_range"):
        golden_mod.Q_DATA_RANGE = tuple(kwargs["q_data_range"])
    if kwargs.get("k_data_range"):
        golden_mod.K_DATA_RANGE = tuple(kwargs["k_data_range"])
    if kwargs.get("v_data_range"):
        golden_mod.V_DATA_RANGE = tuple(kwargs["v_data_range"])

    for s in ("SEED_Q", "SEED_K", "SEED_V", "SEED_QR", "SEED_KR", "SEED_BLOCK_TABLE"):
        key = s.lower()
        if key in kwargs:
            setattr(golden_mod, s, kwargs[key])

    return {
        "B": B,
        "N_q": N_q,
        "N_kv": N_kv,
        "D": D,
        "D_V": D_V,
        "D_rope": D_rope,
        "block_size": block_size,
        "seqused_q_vals": seqused_q_vals,
        "cache_seqlens_vals": cache_seqlens_vals,
        "kv_cache_layout": kv_cache_layout,
        "input_layout": input_layout,
        "output_layout": output_layout,
    }


def cpu_qfmla_mxfp8(
    q,
    k_cache,
    q_descale,
    k_descale,
    block_table,
    cache_seqlens,
    quant_mode,
    *,
    cu_seqlens_q=None,
    seqused_q=None,
    attn_mask=None,
    metadata=None,
    softmax_scale=1.0,
    mask_mode=0,
    max_seqlen_q=-1,
    max_seqlen_kv=-1,
    head_dim_v=512,
    layout_q="BSND",
    layout_kv="PA_BNBD",
    layout_out="BSND",
    return_softmax_lse=False,
    **kwargs,
):
    """CPU golden reference for quant_flash_mla.

    Regenerates BNSD data with same seeds as inputs, computes CPU golden,
    converts output to target layout, returns (attention_out, lse_out).
    """
    testcase_name = kwargs.get("testcase_name", "unknown")
    logger.info(
        "[%s] golden: quant_mode=%s layout_q=%s layout_out=%s",
        testcase_name,
        quant_mode,
        layout_q,
        layout_out,
    )

    # ---- 1. Set golden globals ----
    cfg = _set_golden_globals(
        quant_mode=quant_mode,
        mask_mode=mask_mode,
        layout_q=layout_q,
        layout_kv=layout_kv,
        layout_out=layout_out,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_kv=max_seqlen_kv,
        head_dim_v=head_dim_v,
        return_softmax_lse=return_softmax_lse,
        softmax_scale=softmax_scale,
        **kwargs,
    )

    # ---- 2. Regenerate BNSD data (same seeds → same data) ----
    (q_fp8, k_fp8, v_fp8, deq_q, deq_k, deq_v, p_scale_t, qr_bf16, kr_bf16) = (
        golden_mod.generate_data(device="cpu")
    )

    logger.info(
        "[%s] golden: regenerated q=%s k=%s deq_q=%s",
        testcase_name,
        tuple(q_fp8.shape),
        tuple(k_fp8.shape),
        tuple(deq_q.shape),
    )

    # ---- 3. Call CPU golden ----
    result_bnsd, lse_bnsd = golden_mod.cpu_fp8_fullquant_mla_golden(
        q_fp8,
        k_fp8,
        v_fp8,
        deq_q,
        deq_k,
        deq_v,
        p_scale_t,
        list(cfg["seqused_q_vals"]),
        list(cfg["cache_seqlens_vals"]),
        qr_bf16=qr_bf16,
        kr_bf16=kr_bf16,
    )

    logger.info(
        "[%s] golden: result=%s lse=%s",
        testcase_name,
        tuple(result_bnsd.shape) if result_bnsd is not None else None,
        tuple(lse_bnsd.shape) if lse_bnsd is not None else None,
    )

    # ---- 4. Convert output to target layout ----
    output_layout = cfg["output_layout"]
    result_final = golden_mod.convert_q_bnsd_to_layout(
        result_bnsd, cfg["seqused_q_vals"], output_layout
    )

    # ---- 5. Convert lse if needed ----
    lse_final = None
    if return_softmax_lse and lse_bnsd is not None:
        # lse_bnsd is (B, N, S, 1); convert to target layout
        lse_layout = "BNSD" if output_layout in ("BSND", "BNSD") else output_layout
        lse_final = golden_mod.convert_q_bnsd_to_layout(
            lse_bnsd, cfg["seqused_q_vals"], lse_layout
        )
        if output_layout == "TND":
            # (T, N, 1) → (N, T)
            lse_final = lse_final.permute(1, 0, 2).squeeze(-1).contiguous()
        else:
            # squeeze last dim: (B, N, S, 1) → (B, N, S)
            lse_final = lse_final.squeeze(-1).contiguous()

    # Convert result to bfloat16 to match op output dtype
    result_final = result_final.to(torch.bfloat16)

    logger.info(
        "[%s] golden: final result=%s lse=%s",
        testcase_name,
        tuple(result_final.shape),
        tuple(lse_final.shape) if lse_final is not None else None,
    )

    return result_final, lse_final
