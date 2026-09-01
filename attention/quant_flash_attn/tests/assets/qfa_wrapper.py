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
from typing import Optional

import torch
import torch_npu

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)
import quant_flash_attn_golden as mxfp8_golden_mod
import quant_flash_attn_fp8_golden as fp8_golden_mod
import quant_flash_attn_hif8_golden as hif8_golden_mod


logger = logging.getLogger(__name__)


def _apply_golden_globals(params, quant_mode=1):
    """把 case 参数注入 golden 模块全局变量 (按 quant_mode 选择目标模块).

    quant_mode=6 → fp8_golden_mod (GQA FP8 全量化路径)
    quant_mode=0 → hif8_golden_mod (HIF8 per-tensor 量化路径)
    其他 → mxfp8_golden_mod (MXFP8 路径)
    """
    if quant_mode == 6:
        target = fp8_golden_mod
    elif quant_mode == 0:
        target = hif8_golden_mod
    else:
        target = mxfp8_golden_mod
    for k, v in params.items():
        setattr(target, k, v)


def npu_qfa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dequant_scale_q: torch.Tensor,
    dequant_scale_k: torch.Tensor,
    dequant_scale_v: torch.Tensor,
    p_scale: torch.Tensor = None,
    block_table: torch.Tensor = None,
    cu_seqlens_q_t: torch.Tensor = None,
    cu_seqlens_kv_t: torch.Tensor = None,
    seqused_q_t: torch.Tensor = None,
    seqused_kv_t: torch.Tensor = None,
    sinks_t: torch.Tensor = None,
    attn_mask_t: torch.Tensor = None,
    metadata_t: torch.Tensor = None,
    *,
    batch_size: int,
    N_q: int,
    N_kv: int,
    D: int,
    head_dim_v: Optional[int] = None,
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
    layout_q: str = "TND",
    layout_q_descale: str = "TND",
    layout_kv: str = "TND",
    layout_out: str = "TND",
    attn_mask_shape: tuple = None,
    **kwargs,
):
    """qfa_wrapper.npu_qfa — 15 个位置 tensor 与 CSV tensor_view_shapes 顺序一致:

      0 q, 1 k, 2 v, 3 q_descale, 4 k_descale, 5 v_descale, 6 p_scale, 7 block_table,
      8 cu_seqlens_q, 9 cu_seqlens_kv, 10 seqused_q, 11 seqused_kv,
      12 sinks, 13 attn_mask, 14 metadata.

    slot 8-14 的 tensor 值由 inputs.py (customize_inputs) 用 attributes 真实值覆盖；
    本函数从 cu_seqlens_q_t/seqused_q_t 等 tensor（.cpu().tolist()）读回 list 计算。
    """
    torch_npu.npu.set_device(int(device_id))
    env_graph_path = os.environ.get("GRAPH_PATH")
    if env_graph_path is not None:
        graph_path = int(env_graph_path)
        logger.info(
            "[WRAPPER] 使用环境变量 GRAPH_PATH=%d (覆盖 CSV graph_path)", graph_path
        )

    # cu_seqlens/seqused 的真实值已由 inputs.py in-place 写入 tensor slot (_t)，
    # 这里从 tensor 读回 list（wrapper 端 _t 在 NPU，需 .cpu().tolist()）。
    def _tolist_t(t):
        if t is None:
            return None
        return t.detach().cpu().tolist() if t.numel() > 0 else None

    cu_seqlens_q_list = _tolist_t(cu_seqlens_q_t)
    cu_seqlens_kv_list = _tolist_t(cu_seqlens_kv_t)
    seqused_q_list = _tolist_t(seqused_q_t)
    seqused_kv_list = _tolist_t(seqused_kv_t)

    # batch_size: CSV 原始值 (可为 -1), 透传给 metadata 的 batch_size;
    # B: 从 cu_seqlens_q 推导的正整数, 供 inputs/golden 生成 BNSD 张量。
    B = (
        batch_size
        if batch_size is not None and batch_size > 0
        else (
            max(1, len(cu_seqlens_q_list) - 1)
            if cu_seqlens_q_list and len(cu_seqlens_q_list) >= 2
            else len(list(seqused_q_list))
            if seqused_q_list is not None and len(list(seqused_q_list)) > 0
            else 1
        )
    )

    attn_mask_shape = tuple(attn_mask_t.shape) if attn_mask_t is not None else None

    _apply_golden_globals(
        {
            "B": B,
            "BATCH_SIZE": batch_size,
            "N_q": N_q,
            "N_kv": N_kv,
            "D": D,
            "HEAD_DIM_V": head_dim_v,
            "CU_SEQLENS_Q": cu_seqlens_q_list,
            "CU_SEQLENS_KV": cu_seqlens_kv_list,
            "SEQUSED_Q": seqused_q_list,
            "SEQUSED_KV": seqused_kv_list,
            "MAX_SEQLEN_Q": max_seqlen_q,
            "MAX_SEQLEN_KV": max_seqlen_kv,
            "ENABLE_PA": enable_pa,
            "KV_CACHE_LAYOUT": kv_cache_layout,
            "BLOCK_SIZE": block_size,
            "SPARSE_MODE": mask_mode,
            "ATTN_MASK_SHAPE": attn_mask_shape or (2048, 2048),
            "Q_SCALE_LAYOUT": q_scale_layout,
            "P_SCALE": (float(p_scale.item()) if p_scale.numel() > 0 else None)
            if isinstance(p_scale, torch.Tensor)
            else float(p_scale),
            "ENABLE_LSE": enable_lse,
            "FP8_DTYPE": torch.float8_e4m3fn,
            "QUANT_MODE": quant_mode,
            "QUANT_GROUP_SIZE": 32,
            "INPUT_LAYOUT": layout_q,
            "IS_CONTIGUOUS": is_contiguous,
            "DEVICE_ID": device_id,
            "GRAPH_PATH": graph_path,
            "SOFTMAX_SCALE": softmax_scale,
            "LAYOUT_Q": layout_q,
            "LAYOUT_Q_DESCALE": layout_q_descale,
            "LAYOUT_KV": layout_kv,
            "LAYOUT_OUT": layout_out,
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
                seqused_q_list,
                seqused_kv_list,
                max_seqlen_q,
                max_seqlen_kv,
                block_table
                if isinstance(block_table, torch.Tensor) and enable_pa
                else None,
                cu_seqlens_q_t=cu_seqlens_q_t,
                cu_seqlens_kv_t=cu_seqlens_kv_t,
                seqused_q_t=seqused_q_t,
                seqused_kv_t=seqused_kv_t,
            )
        elif quant_mode == 0:
            atten_out, lse_out = hif8_golden_mod.npu_hif8_fa(
                q,
                k,
                v,
                dequant_scale_q,
                dequant_scale_k,
                dequant_scale_v,
                p_scale,
                cu_seqlens_q_list,
                cu_seqlens_kv_list,
                seqused_q_list,
                seqused_kv_list,
                max_seqlen_q,
                max_seqlen_kv,
                block_table
                if isinstance(block_table, torch.Tensor) and enable_pa
                else None,
                cu_seqlens_q_t=cu_seqlens_q_t,
                cu_seqlens_kv_t=cu_seqlens_kv_t,
                seqused_q_t=seqused_q_t,
                seqused_kv_t=seqused_kv_t,
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
                seqused_q_list,
                seqused_kv_list,
                max_seqlen_q,
                max_seqlen_kv,
                block_table
                if isinstance(block_table, torch.Tensor) and enable_pa
                else None,
                cu_seqlens_q_t=cu_seqlens_q_t,
                cu_seqlens_kv_t=cu_seqlens_kv_t,
                seqused_q_t=seqused_q_t,
                seqused_kv_t=seqused_kv_t,
            )
    except Exception as e:
        logger.error("[WRAPPER] NPU 调用失败: %s", str(e))
        raise

    if not enable_lse:
        lse_out = None
    else:
        layout_out_up = str(layout_out).upper() if layout_out else str(layout_q).upper()
        if layout_out_up == "TND":
            lse_out = lse_out[:, : atten_out.shape[0]].contiguous()
        elif layout_out_up == "BNSD":
            lse_out = lse_out[:, :, : atten_out.shape[2]].contiguous()
        else:
            lse_out = lse_out[:, :, : atten_out.shape[1]].contiguous()
    return atten_out, lse_out
