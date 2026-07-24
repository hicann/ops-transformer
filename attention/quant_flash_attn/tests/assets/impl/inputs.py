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
_TESTS_DIR = os.path.join(_ASSETS_DIR, "..", "pytest", "fia_fullquant_mxfp8_test")
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)
from common import fia_fullquant_mxfp8_golden as golden_mod

logger = logging.getLogger(__name__)

__input__ = {
    "e2e": {
        "qfa_mxfp8_wrapper.npu_qfa_mxfp8": "generate_qfa_mxfp8_inputs"
    }
}

_SEED_MAP = {"q": 54, "k": 3, "v": 4}


def get_cached_inputs():
    """wrapper 调用前从这里取 customize_inputs 生成的真实数据。
    存在 golden_mod 上,避免模块重复加载导致缓存丢失。
    """
    return getattr(golden_mod, "_cached_mxfp8_inputs", None)


def generate_qfa_mxfp8_inputs(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
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
    """生成 BNSD FP8 Q/K/V + fp32 scale + block_table,缓存到 golden_mod 上。"""
    # cu_seqlens → actual_seq (差分还原)
    cu_seqlens_q = list(cu_seqlens_q) if cu_seqlens_q is not None else [0]
    cu_seqlens_kv = list(cu_seqlens_kv) if cu_seqlens_kv is not None else [0]
    actual_seq_q = [cu_seqlens_q[i + 1] - cu_seqlens_q[i] for i in range(len(cu_seqlens_q) - 1)] if len(cu_seqlens_q) > 1 else [0]
    # PA 模式下 cu_seqlens_kv 可能为空，用 seqused_kv 推导
    if len(cu_seqlens_kv) > 1:
        actual_seq_kv = [cu_seqlens_kv[i + 1] - cu_seqlens_kv[i] for i in range(len(cu_seqlens_kv) - 1)]
    elif seqused_kv is not None and len(seqused_kv) > 0:
        actual_seq_kv = list(seqused_kv)
    else:
        actual_seq_kv = [0]

    max_sq = max(actual_seq_q) if actual_seq_q else D
    max_skv = max(actual_seq_kv) if actual_seq_kv else D

    fp8_dtype = torch.float8_e4m3fn
    group_size = 32

    for gkey, gval in [("B", B), ("N_q", N_q), ("N_kv", N_kv), ("D", D),
                       ("FP8_DTYPE", fp8_dtype), ("QUANT_GROUP_SIZE", group_size)]:
        setattr(golden_mod, gkey, gval)

    torch.manual_seed(_SEED_MAP["q"])
    q_fp16 = (torch.rand(B, N_q, max_sq, D, dtype=torch.float16) * 2 - 1)
    torch.manual_seed(_SEED_MAP["k"])
    k_fp16 = (torch.rand(B, N_kv, max_skv, D, dtype=torch.float16) * 2 - 1)
    torch.manual_seed(_SEED_MAP["v"])
    v_fp16 = (torch.rand(B, N_kv, max_skv, D, dtype=torch.float16) * 2 - 1)

    quant_scale_q = golden_mod.get_mxfp8_per_token_group_quant_scale(q_fp16, fp8_dtype, group_size)
    quant_scale_k = golden_mod.get_mxfp8_per_token_group_quant_scale(k_fp16, fp8_dtype, group_size)
    quant_scale_v = golden_mod.get_mxfp8_per_channel_group_quant_scale(v_fp16, fp8_dtype, group_size)

    dequant_scale_q = quant_scale_q
    dequant_scale_k = quant_scale_k
    v_descale = quant_scale_v

    fp8_max = 448.0
    q_fp8 = golden_mod.mxfp8_per_token_group_quant(q_fp16, quant_scale_q, group_size).clamp(-fp8_max, fp8_max).to(fp8_dtype)
    k_fp8 = golden_mod.mxfp8_per_token_group_quant(k_fp16, quant_scale_k, group_size).clamp(-fp8_max, fp8_max).to(fp8_dtype)
    v_fp8 = golden_mod.mxfp8_per_channel_group_quant(v_fp16, quant_scale_v, group_size).clamp(-fp8_max, fp8_max).to(fp8_dtype)

    p_scale_t = torch.tensor([1.0], dtype=torch.float32)

    block_table_t = None
    if enable_pa:
        block_num = sum(math.ceil(s / block_size) for s in actual_seq_kv)
        max_blocks = max(math.ceil(s / block_size) for s in actual_seq_kv) if actual_seq_kv else 0
        block_idx_list = torch.randperm(block_num, dtype=torch.int32)
        block_table_t = torch.full((B, max_blocks), -1, dtype=torch.int32)
        idx = 0
        for b in range(B):
            n_blocks = math.ceil(actual_seq_kv[b] / block_size)
            for j in range(n_blocks):
                block_table_t[b, j] = block_idx_list[idx]
                idx += 1

    # 缓存到 golden_mod 上(不是本模块,避免 spec.py 重复加载导致缓存丢失)
    golden_mod._cached_mxfp8_inputs = [q_fp8, k_fp8, v_fp8,
                                       dequant_scale_q, dequant_scale_k, v_descale,
                                       p_scale_t, block_table_t]
    # 无返回值(符合 TTK 约定)
