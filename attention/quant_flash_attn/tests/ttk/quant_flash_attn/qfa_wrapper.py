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
import torch_npu

_TESTS_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "..",
        "pytest",
        "quant_flash_attn",
    )
)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)
import quant_flash_attn_golden as mxfp8_golden_mod
import quant_flash_attn_fp8_golden as fp8_golden_mod


logger = logging.getLogger(__name__)


def _apply_golden_globals(params, quant_mode=1):
    """把 case 参数注入 golden 模块全局变量 (按 quant_mode 选择目标模块).

    quant_mode=6 → fp8_golden_mod (GQA FP8 全量化路径)
    其他 → mxfp8_golden_mod (MXFP8 路径)
    """
    target = fp8_golden_mod if quant_mode == 6 else mxfp8_golden_mod
    for k, v in params.items():
        setattr(target, k, v)


def npu_qfa(
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
    torch_npu.npu.set_device(int(device_id))
    env_graph_path = os.environ.get("GRAPH_PATH")
    if env_graph_path is not None:
        graph_path = int(env_graph_path)
        logger.info(
            "[WRAPPER] 使用环境变量 GRAPH_PATH=%d (覆盖 CSV graph_path)", graph_path
        )

    cu_seqlens_q_list = list(cu_seqlens_q) if cu_seqlens_q is not None else None
    cu_seqlens_kv_list = list(cu_seqlens_kv) if cu_seqlens_kv is not None else None

    _apply_golden_globals(
        {
            "B": B,
            "N_q": N_q,
            "N_kv": N_kv,
            "D": D,
            "CU_SEQLENS_Q": cu_seqlens_q_list,
            "CU_SEQLENS_KV": cu_seqlens_kv_list,
            "SEQUSED_Q": list(seqused_q) if seqused_q is not None else None,
            "SEQUSED_KV": list(seqused_kv) if seqused_kv is not None else None,
            "MAX_SEQLEN_Q": max_seqlen_q,
            "MAX_SEQLEN_KV": max_seqlen_kv,
            "ENABLE_PA": enable_pa,
            "KV_CACHE_LAYOUT": kv_cache_layout,
            "BLOCK_SIZE": block_size,
            "SPARSE_MODE": mask_mode,
            "Q_SCALE_LAYOUT": q_scale_layout,
            "P_SCALE": float(p_scale.item())
            if isinstance(p_scale, torch.Tensor)
            else float(p_scale),
            "ENABLE_LSE": enable_lse,
            "FP8_DTYPE": torch.float8_e4m3fn,
            "QUANT_MODE": quant_mode,
            "QUANT_GROUP_SIZE": 32,
            "INPUT_LAYOUT": input_layout,
            "IS_CONTIGUOUS": is_contiguous,
            "DEVICE_ID": device_id,
            "GRAPH_PATH": graph_path,
            "SOFTMAX_SCALE": softmax_scale,
        },
        quant_mode=quant_mode,
    )

    logger.info(
        "[WRAPPER] graph_path=%d, quant_mode=%d, 透传 fp8 (q=%s, k=%s, dq=%s, dk=%s)",
        graph_path,
        quant_mode,
        tuple(q.shape),
        tuple(k.shape),
        tuple(dequant_scale_q.shape),
        tuple(dequant_scale_k.shape),
    )
    try:
        if quant_mode == 6:
            atten_out, lse_out = fp8_golden_mod.npu_gqa_fp8_fa(
                q,
                k,
                v,
                dequant_scale_q,
                dequant_scale_k,
                dequant_scale_v,
                p_scale,
                cu_seqlens_q_list,
                cu_seqlens_kv_list,
                list(seqused_q) if seqused_q is not None else None,
                list(seqused_kv) if seqused_kv is not None else None,
                max_seqlen_q,
                max_seqlen_kv,
                block_table
                if isinstance(block_table, torch.Tensor) and enable_pa
                else None,
            )
        else:
            atten_out, lse_out = mxfp8_golden_mod.npu_mxfp8_fa(
                q,
                k,
                v,
                dequant_scale_q,
                dequant_scale_k,
                dequant_scale_v,
                p_scale,
                cu_seqlens_q_list,
                cu_seqlens_kv_list,
                list(seqused_q) if seqused_q is not None else None,
                list(seqused_kv) if seqused_kv is not None else None,
                max_seqlen_q,
                max_seqlen_kv,
                block_table
                if isinstance(block_table, torch.Tensor) and enable_pa
                else None,
            )
    except Exception as e:
        logger.error("[WRAPPER] NPU 调用失败: %s", str(e))
        raise

    if not enable_lse:
        lse_out = None
    elif isinstance(lse_out, torch.Tensor) and lse_out.ndim == 2:
        lse_out = lse_out.reshape(lse_out.shape[1], lse_out.shape[0]).contiguous()
        if lse_out.shape[0] > atten_out.shape[0]:
            lse_out = lse_out[:atten_out.shape[0], :].contiguous()
    return atten_out, lse_out
