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
    # cu_seqlens → actual_seq (差分还原)
    cu_seqlens_q = list(cu_seqlens_q) if cu_seqlens_q is not None else [0]
    cu_seqlens_kv = list(cu_seqlens_kv) if cu_seqlens_kv is not None else [0]
    actual_seq_q = (
        [cu_seqlens_q[i + 1] - cu_seqlens_q[i] for i in range(len(cu_seqlens_q) - 1)]
        if len(cu_seqlens_q) > 1
        else [0]
    )
    # PA 模式下 cu_seqlens_kv 可能为空, 用 seqused_kv 推导
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
    ]:
        setattr(golden_mod, gkey, gval)

    # csv input_data_ranges 列由 ttk 作为 input_ranges kwarg 传入 (list of (min,max) per tensor)
    # q/k/v 是前 3 个; csv 省略或 None -> 用默认 (-1, 1)
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

    # ----- in-place 写 bf16 q/k/v (dtype 一致, 合法; 共享内存直接落 np_storages) -----
    # q/k/v 的 shape 必须和 CSV tensor_view_shapes 一致 (excel_to_csv.py 已对齐 max_sq 推导)
    torch.manual_seed(_SEED_MAP["q"])
    q_real = (
        torch.rand(B, N_q, max_sq, D, dtype=torch.bfloat16) * (rq_max - rq_min) + rq_min
    )
    q[:] = q_real

    torch.manual_seed(_SEED_MAP["k"])
    k_real = (
        torch.rand(B, N_kv, max_skv, D, dtype=torch.bfloat16) * (rk_max - rk_min)
        + rk_min
    )
    k[:] = k_real

    torch.manual_seed(_SEED_MAP["v"])
    v_real = (
        torch.rand(B, N_kv, max_skv, D, dtype=torch.bfloat16) * (rv_max - rv_min)
        + rv_min
    )
    v[:] = v_real

    # ----- p_scale: in-place 写 fp32 (dtype 一致) -----
    p_scale_value = kwargs.get("p_scale_value", 1.0)
    if isinstance(p_scale_value, torch.Tensor):
        p_scale_value = float(p_scale_value.item())
    p_scale[:] = torch.tensor([float(p_scale_value)], dtype=torch.float32)

    # ----- block_table: in-place 写 int32 -----
    # PA 模式生成真实 block_table; 非 PA 模式写零占位 (wrapper 内 enable_pa=False 时忽略)
    if enable_pa:
        block_num = sum(math.ceil(s / block_size) for s in actual_seq_kv)
        max_blocks = (
            max(math.ceil(s / block_size) for s in actual_seq_kv)
            if actual_seq_kv
            else 0
        )
        block_idx_list = torch.randperm(block_num, dtype=torch.int32)
        bt_real = torch.full((B, max_blocks), -1, dtype=torch.int32)
        idx = 0
        for b in range(B):
            n_blocks = math.ceil(actual_seq_kv[b] / block_size)
            for j in range(n_blocks):
                bt_real[b, j] = block_idx_list[idx]
                idx += 1
        block_table[:] = bt_real
    else:
        block_table[:] = 0

    # descale_q/k/v 不动 (占位 fp8 tensor, wrapper/golden 内现算真实 e8m0 覆盖)

    logger.info(
        "[INPUTS] in-place wrote bf16 q/k/v (shape q=%s), fp32 p_scale, int32 block_table; "
        "descale kept as placeholder",
        tuple(q.shape),
    )
