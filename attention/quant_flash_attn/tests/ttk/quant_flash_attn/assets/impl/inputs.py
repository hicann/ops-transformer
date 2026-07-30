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
import math
import os
import sys
from typing import List

import numpy
import torch

_ASSETS_DIR = os.path.dirname(os.path.abspath(__file__))
_TESTS_DIR = os.path.join(_ASSETS_DIR, "..", "..")
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

import quant_flash_attn_golden as golden_mod

logger = logging.getLogger(__name__)

__input__ = {"e2e": {"qfa_mxfp8_wrapper.npu_qfa_mxfp8": "generate_qfa_mxfp8_inputs"}}

_SEED_MAP = {"q": 54, "k": 3, "v": 4}


def generate_qfa_mxfp8_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dequant_scale_q: torch.Tensor,
    dequant_scale_k: torch.Tensor,
    dequant_scale_v: torch.Tensor,
    p_scale: torch.Tensor,
    block_table: torch.Tensor,
    *,
    B: int,
    N_q: int,
    N_kv: int,
    D: int,
    cu_seqlens_q: List[int],
    cu_seqlens_kv: List[int],
    seqused_q: List[int],
    seqused_kv: List[int],
    max_seqlen_q: int,
    max_seqlen_kv: int,
    enable_pa: bool,
    kv_cache_layout: str,
    block_size: int,
    mask_mode: int,
    q_scale_layout: str,
    quant_mode: int = 1,
    enable_lse: bool = False,
    graph_path: int = 0,
    input_layout: str = "TND",
    is_contiguous: bool = True,
    device_id: int = 0,
    softmax_scale: float = None,
    data_range_q: float = 1.0,
    data_range_k: float = 1.0,
    data_range_v: float = 1.0,
    **kwargs,
):
    # cu_seqlens -> actual_seq (差分还原)
    cu_seqlens_q = list(cu_seqlens_q) if cu_seqlens_q is not None else [0]
    cu_seqlens_kv = list(cu_seqlens_kv) if cu_seqlens_kv is not None else [0]
    actual_seq_q = (
        [cu_seqlens_q[i + 1] - cu_seqlens_q[i] for i in range(len(cu_seqlens_q) - 1)]
        if len(cu_seqlens_q) > 1
        else [0]
    )
    if len(cu_seqlens_kv) > 1:
        actual_seq_kv = [
            cu_seqlens_kv[i + 1] - cu_seqlens_kv[i]
            for i in range(len(cu_seqlens_kv) - 1)
        ]
    elif seqused_kv is not None and len(seqused_kv) > 0:
        actual_seq_kv = list(seqused_kv)
    else:
        actual_seq_kv = [0]

    max_sq = max(actual_seq_q) if actual_seq_q else D
    max_skv = max(actual_seq_kv) if actual_seq_kv else D

    for gkey, gval in [
        ("B", B),
        ("N_q", N_q),
        ("N_kv", N_kv),
        ("D", D),
        ("FP8_DTYPE", torch.float8_e4m3fn),
        ("QUANT_GROUP_SIZE", 32),
        ("CU_SEQLENS_Q", cu_seqlens_q),
        ("CU_SEQLENS_KV", cu_seqlens_kv if cu_seqlens_kv else None),
        ("SEQUSED_Q", list(seqused_q) if seqused_q is not None else None),
        ("SEQUSED_KV", list(seqused_kv) if seqused_kv is not None else None),
        ("MAX_SEQLEN_Q", max_seqlen_q),
        ("MAX_SEQLEN_KV", max_seqlen_kv),
        ("ENABLE_PA", enable_pa),
        ("KV_CACHE_LAYOUT", kv_cache_layout),
        ("BLOCK_SIZE", block_size),
        ("Q_SCALE_LAYOUT", q_scale_layout),
        ("INPUT_LAYOUT", input_layout),
    ]:
        setattr(golden_mod, gkey, gval)

    # csv input_data_ranges 列由 ttk 作为 input_ranges kwarg 传入
    input_ranges = kwargs.get("input_ranges")

    def _range_of(idx, default=(-1.0, 1.0)):
        if input_ranges is None or idx >= len(input_ranges):
            return default
        r = input_ranges[idx]
        if r is None:
            return default
        return (float(r[0]), float(r[1]))

    rq_min, rq_max = _range_of(0)
    rk_min, rk_max = _range_of(1)
    rv_min, rv_max = _range_of(2)

    # ----- Step 1: 生成 bf16 BNSD random  -----
    torch.manual_seed(_SEED_MAP["q"])
    q_bf16 = (
        torch.rand(B, N_q, max_sq, D, dtype=torch.bfloat16) * (rq_max - rq_min) + rq_min
    )
    torch.manual_seed(_SEED_MAP["k"])
    k_bf16 = (
        torch.rand(B, N_kv, max_skv, D, dtype=torch.bfloat16) * (rk_max - rk_min)
        + rk_min
    )
    torch.manual_seed(_SEED_MAP["v"])
    v_bf16 = (
        torch.rand(B, N_kv, max_skv, D, dtype=torch.bfloat16) * (rv_max - rv_min)
        + rv_min
    )

    # ----- Step 2: 量化 bf16 -> fp8 BNSD + fp32 scale BNSD  -----
    fp8_dtype = torch.float8_e4m3fn
    group_size = 32
    fp8_max = 448.0

    quant_scale_q_bnsd = golden_mod.get_mxfp8_per_token_group_quant_scale(
        q_bf16, fp8_dtype, group_size
    )
    quant_scale_k_bnsd = golden_mod.get_mxfp8_per_token_group_quant_scale(
        k_bf16, fp8_dtype, group_size
    )
    quant_scale_v_bnsd = golden_mod.get_mxfp8_per_channel_group_quant_scale(
        v_bf16, fp8_dtype, group_size
    )

    q_fp8_bnsd_f32 = (
        golden_mod.mxfp8_per_token_group_quant(q_bf16, quant_scale_q_bnsd, group_size)
        .clamp(-fp8_max, fp8_max)
        .to(fp8_dtype)
    )
    k_fp8_bnsd_f32 = (
        golden_mod.mxfp8_per_token_group_quant(k_bf16, quant_scale_k_bnsd, group_size)
        .clamp(-fp8_max, fp8_max)
        .to(fp8_dtype)
    )
    v_fp8_bnsd_f32 = (
        golden_mod.mxfp8_per_channel_group_quant(v_bf16, quant_scale_v_bnsd, group_size)
        .clamp(-fp8_max, fp8_max)
        .to(fp8_dtype)
    )

    # ----- Step 3: layout 转换 (BNSD -> final layout) -----
    q_fp8_final = golden_mod.convert_q_bnsd_to_layout(
        q_fp8_bnsd_f32, actual_seq_q, "TND", cu_seqlens=cu_seqlens_q
    )
    q_scale_final_layout = golden_mod.convert_q_scale_bnsd_to_layout(
        quant_scale_q_bnsd, actual_seq_q, q_scale_layout, cu_seqlens=cu_seqlens_q
    )

    if enable_pa:
        total_blocks_k = int(dequant_scale_k.shape[0])
        total_blocks_v = int(dequant_scale_v.shape[0])
        total_blocks = min(total_blocks_k, total_blocks_v)

        if block_table.size == 0:
            raise ValueError(
                "[INPUTS] enable_pa=True but block_table CSV shape is (0,) — "
                "Excel AI column empty for PA row"
            )
        max_blocks = block_table.shape[1] if block_table.ndim >= 2 else 0
        if max_blocks <= 0:
            raise ValueError(
                f"[INPUTS] PA mode block_table max_blocks={max_blocks} invalid"
            )
        torch.manual_seed(42)  # 确定性 block_table
        blockid_pool = torch.randperm(total_blocks, dtype=torch.int32)
        bt_real = torch.full((B, max_blocks), -1, dtype=torch.int32)
        idx = 0
        for b in range(B):
            n_blocks = math.ceil(actual_seq_kv[b] / block_size)
            for j in range(n_blocks):
                bt_real[b, j] = blockid_pool[idx % total_blocks]
                idx += 1
        k_fp8_final = golden_mod.mxfp8_pa_preprocessing(
            k_fp8_bnsd_f32,
            actual_seq_kv,
            block_size,
            bt_real,
            is_scale=False,
            kv_layout=kv_cache_layout,
            total_blocks=total_blocks_k,
        )
        v_fp8_final = golden_mod.mxfp8_pa_preprocessing(
            v_fp8_bnsd_f32,
            actual_seq_kv,
            block_size,
            bt_real,
            is_scale=False,
            kv_layout=kv_cache_layout,
            total_blocks=total_blocks_v,
        )

        k_scale_pa = golden_mod.mxfp8_pa_preprocessing(
            quant_scale_k_bnsd,
            actual_seq_kv,
            block_size,
            bt_real,
            is_scale=True,
            is_vscale=False,
            kv_layout=kv_cache_layout,
            total_blocks=total_blocks_k,
        )
        v_scale_pa = golden_mod.mxfp8_pa_preprocessing(
            quant_scale_v_bnsd,
            actual_seq_kv,
            block_size,
            bt_real,
            is_scale=True,
            is_vscale=True,
            kv_layout=kv_cache_layout,
            total_blocks=total_blocks_v,
        )
        k_scale_final_layout = k_scale_pa
        v_scale_final_layout = v_scale_pa
    else:
        # 非 PA 模式: k/v -> TND
        k_fp8_final = golden_mod.convert_kv_bnsd_to_layout(
            k_fp8_bnsd_f32,
            actual_seq_kv,
            "TND",
            cu_seqlens=cu_seqlens_kv if cu_seqlens_kv else None,
        )
        v_fp8_final = golden_mod.convert_kv_bnsd_to_layout(
            v_fp8_bnsd_f32,
            actual_seq_kv,
            "TND",
            cu_seqlens=cu_seqlens_kv if cu_seqlens_kv else None,
        )
        # k/v scale: convert 函数内部做 pack, 输入 UNPACKED 4D scale (B, N, S, D)
        k_scale_final_layout = golden_mod.convert_k_scale_bnsd_to_layout(
            quant_scale_k_bnsd,
            actual_seq_kv,
            "TND",
            cu_seqlens=cu_seqlens_kv if cu_seqlens_kv else None,
        )
        v_scale_final_layout = golden_mod.convert_v_scale_bnsd_to_layout(
            quant_scale_v_bnsd, actual_seq_kv, "TND"
        )

    # ----- Step 4: fp32 scale -> e8m0 -----

    q_scale_e8m0 = golden_mod.fp32_to_e8m0fnu_safe(q_scale_final_layout, "Q scale")
    k_scale_e8m0 = golden_mod.fp32_to_e8m0fnu_safe(k_scale_final_layout, "K scale")
    v_scale_e8m0 = golden_mod.fp32_to_e8m0fnu_safe(v_scale_final_layout, "V scale")

    def _inplace_write(dst_np, src_torch, slot_name):
        if tuple(dst_np.shape) != tuple(src_torch.shape):
            raise ValueError(
                f"[INPUTS] {slot_name} shape mismatch: CSV storage {tuple(dst_np.shape)} "
                f"!= computed {tuple(src_torch.shape)}. "
                f"Check Excel shape vs layout conversion logic."
            )
        dst_dtype_np = dst_np.dtype
        ts = str(src_torch.dtype)
        if "float8" in ts:
            src_np = src_torch.view(torch.uint8).numpy().view(dst_dtype_np)
        else:
            src_np = src_torch.numpy()
        if src_np.dtype != dst_dtype_np:
            raise ValueError(
                f"[INPUTS] {slot_name} dtype mismatch: CSV storage {dst_dtype_np} "
                f"!= computed {src_np.dtype} (torch {src_torch.dtype}). "
                f"Check CSV dtype vs inputs.py quant/layout output dtype."
            )
        # numpy in-place 拷贝 (同 dtype 同 shape, bit-exact 内存拷贝)
        dst_np[...] = src_np

    _inplace_write(q, q_fp8_final, "q (slot 0)")
    _inplace_write(k, k_fp8_final, "k (slot 1)")
    _inplace_write(v, v_fp8_final, "v (slot 2)")
    _inplace_write(dequant_scale_q, q_scale_e8m0, "descale_q (slot 3)")
    _inplace_write(dequant_scale_k, k_scale_e8m0, "descale_k (slot 4)")
    _inplace_write(dequant_scale_v, v_scale_e8m0, "descale_v (slot 5)")

    # ----- p_scale: in-place 写 fp32 -----
    p_scale_value = kwargs.get("p_scale_value", 1.0)
    if isinstance(p_scale_value, torch.Tensor):
        p_scale_value = float(p_scale_value.item())
    p_scale[...] = numpy.array([float(p_scale_value)], dtype=numpy.float32)

    if enable_pa:
        block_table[...] = bt_real.numpy()
    else:
        block_table[...] = 0

    logger.info(
        "[INPUTS] in-place wrote fp8 q/k/v (q=%s), e8m0 descale (dq=%s, dk=%s, dv=%s), "
        "fp32 p_scale, int32 block_table (pa=%s)",
        tuple(q.shape),
        tuple(dequant_scale_q.shape),
        tuple(dequant_scale_k.shape),
        tuple(dequant_scale_v.shape),
        enable_pa,
    )
