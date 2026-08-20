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
from typing import List, Optional

import numpy as np
import torch

_ASSETS_DIR = os.path.dirname(os.path.abspath(__file__))
_TESTS_DIR = os.path.join(_ASSETS_DIR, "..", "ttk", "qfa_mxfp4_test")
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)
from common import qfa_mxfp4_golden as golden_mod

logger = logging.getLogger(__name__)

__input__ = {"e2e": {"qfa_mxfp4_wrapper.npu_qfa_mxfp4": "generate_qfa_mxfp4_inputs"}}


# 环境变量 QFA_PASS_THROUGH=1 时跳过 golden_mod.generate_data() (含 _resolve_s /
# transpose_qscale / rearrange_by_layout / v_descale.view 等所有会报错的步骤),
# CSV 框架分配的 tensor 槽位保持原样透传给 wrapper, 由 wrapper 的 _PASS_THROUGH 分支
# 或 op_host checker 处理. 用于异常用例测试 (s=0 / kv shape 不一致 / D 不支持 /
# descale 维度异常 / seq 缺失 等), 在 CSV 配异常 shape + 设本开关即可.
_PASS_THROUGH = os.environ.get("QFA_PASS_THROUGH", "").lower() in ("1", "true", "yes")


def _apply_golden_globals(attrs):
    for k, v in attrs.items():
        setattr(golden_mod, k, v)


def _inplace_write(dst, src_torch, slot_name):
    if dst is None:
        raise ValueError(f"[INPUTS] {slot_name}: dst is None, CSV 未分配该张量槽位")
    if hasattr(dst, "numpy"):  # torch tensor
        dst_t = dst
        if tuple(dst_t.shape) != tuple(src_torch.shape):
            logger.warning(
                "[INPUTS] %s shape mismatch: CSV %s != computed %s, 按 CSV shape 适配",
                slot_name,
                tuple(dst_t.shape),
                tuple(src_torch.shape),
            )
            adapted = golden_mod._adapt_tensor_to_shape(src_torch, dst_t.shape)
            if dst_t.dtype == adapted.dtype:
                dst_t[...] = adapted
            else:
                dst_t[...] = adapted.to(dst_t.dtype)
            return
        if dst_t.dtype == src_torch.dtype:
            dst_t[...] = src_torch
        elif dst_t.dtype == torch.uint8 and src_torch.dtype == torch.uint8:
            dst_t[...] = src_torch
        else:
            dst_t[...] = src_torch.to(dst_t.dtype)
        return
    # numpy array
    dst_np = np.asarray(dst)
    if tuple(dst_np.shape) != tuple(src_torch.shape):
        logger.warning(
            "[INPUTS] %s shape mismatch: CSV %s != computed %s, 按 CSV shape 适配",
            slot_name,
            tuple(dst_np.shape),
            tuple(src_torch.shape),
        )
        adapted = golden_mod._adapt_tensor_to_shape(src_torch, dst_np.shape)
        src_np = (
            adapted.numpy() if adapted.device.type == "cpu" else adapted.cpu().numpy()
        )
        if src_np.dtype != dst_np.dtype:
            src_np = (
                src_np.view(dst_np.dtype)
                if src_np.dtype.itemsize == dst_np.dtype.itemsize
                else src_np.astype(dst_np.dtype)
            )
        dst_np[...] = src_np
        return
    src_np = (
        src_torch.numpy() if src_torch.device.type == "cpu" else src_torch.cpu().numpy()
    )
    if src_np.dtype != dst_np.dtype:
        src_np = (
            src_np.view(dst_np.dtype)
            if src_np.dtype.itemsize == dst_np.dtype.itemsize
            else src_np.astype(dst_np.dtype)
        )
    dst_np[...] = src_np


def generate_qfa_mxfp4_inputs(
    # ========== Tensor inputs (CSV 框架分配, 原位写入) ==========
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_descale: torch.Tensor,
    k_descale: torch.Tensor,
    v_descale: torch.Tensor,
    block_table: Optional[torch.Tensor],
    # ========== Scalar / list attrs ==========
    B: int,
    N_q: int,
    N_kv: int,
    G: int,
    D: int,
    V_D: int,
    Rope_D: int,
    act_seq_lens_q: List[int],
    act_seq_lens_kv: List[int],
    input_layout: str,
    layout_q_descale: str,
    layout_kv: str,
    layout_out: str,
    kv_storage_mode: str,
    block_size: int,
    q_dtype: str,
    kv_dtype: str,
    out_dtype: str,
    q_quant_mode: int,
    mask_mode: int,
    pre_tokens: int,
    next_tokens: int,
    enable_mask: bool,
    enable_lse: bool,
    inner_precise: int,
    device_id: int,
    graph_path: int,
    softmax_scale: Optional[float] = None,
    data_range_q: str = "1.0",
    data_range_k: str = "1.0",
    data_range_v: str = "1.0",
    max_seqlen_q: int = -1,
    max_seqlen_kv: int = -1,
    cu_seqlens_q: List[int] = None,
    cu_seqlens_kv: List[int] = None,
    # 可选 tensor 入参 (shape 为空 -> golden 传 None)
    block_table_shape: List[int] = None,
    block_table_dtype: str = None,
    p_scale_value: Optional[float] = None,
    p_scale_shape: List[int] = None,
    p_scale_dtype: str = None,
    p_scale_datarange: str = None,
    sinks_shape: List[int] = None,
    sinks_dtype: str = None,
    sinks_datarange: str = None,
    attn_mask_shape: List[int] = None,
    attn_mask_dtype: str = None,
    attn_mask_datarange: str = None,
    # dtype 透传 (表格不传 -> None, golden 侧用默认值)
    q_descale_dtype: str = None,
    k_descale_dtype: str = None,
    v_descale_dtype: str = None,
    seqused_q_dtype: str = None,
    seqused_kv_dtype: str = None,
    cu_seqlens_q_dtype: str = None,
    cu_seqlens_kv_dtype: str = None,
    softmax_lse_dtype: str = None,
    # metadata 伪 tensor 入参 (对齐 run_main 签名, PASS_THROUGH 时按表格构造)
    metadata_shape: List[int] = None,
    metadata_dtype: str = None,
    **kwargs,
):
    # 注入 golden 全局变量 (generate_data 读这些配置)
    _apply_golden_globals(
        {
            "B": B,
            "N_q": N_q,
            "N_kv": N_kv,
            "G": G,
            "D": D,
            "V_D": V_D if V_D is not None else D,
            "Rope_D": Rope_D,
            "INPUT_LAYOUT": input_layout,
            "LAYOUT_Q_DESCALE": layout_q_descale,
            "LAYOUT_KV": layout_kv,
            "LAYOUT_OUT": layout_out,
            "KV_STORAGE_MODE": kv_storage_mode,
            "BLOCK_SIZE": block_size,
            "Q_DTYPE": q_dtype,
            "KV_DTYPE": kv_dtype,
            "OUT_DTYPE": out_dtype,
            "Q_QUANT_MODE": q_quant_mode,
            "SPARSE_MODE": mask_mode,
            "PRE_TOKENS": pre_tokens,
            "NEXT_TOKENS": next_tokens,
            "ENABLE_MASK": enable_mask,
            "ENABLE_LSE": enable_lse,
            "INNER_PRECISE": inner_precise,
            "DEVICE_ID": device_id,
            "GRAPH_PATH": graph_path,
            "SOFTMAX_SCALE": softmax_scale,
            "DATA_RANGE_Q": data_range_q,
            "DATA_RANGE_K": data_range_k,
            "DATA_RANGE_V": data_range_v,
            "ACT_SEQ_LENS_Q": list(act_seq_lens_q) if act_seq_lens_q else [],
            "ACT_SEQ_LENS_KV": list(act_seq_lens_kv) if act_seq_lens_kv else [],
            "MAX_SEQLEN_Q": max_seqlen_q,
            "MAX_SEQLEN_KV": max_seqlen_kv,
            "CU_SEQLENS_Q": list(cu_seqlens_q) if cu_seqlens_q else [],
            "CU_SEQLENS_KV": list(cu_seqlens_kv) if cu_seqlens_kv else [],
            "BLOCK_TABLE_SHAPE": list(block_table_shape) if block_table_shape else [],
            "BLOCK_TABLE_DTYPE": block_table_dtype,
            "P_SCALE_VALUE": p_scale_value,
            "P_SCALE_SHAPE": list(p_scale_shape) if p_scale_shape else [],
            "P_SCALE_DTYPE": p_scale_dtype,
            "P_SCALE_DATARANGE": p_scale_datarange,
            "SINKS_SHAPE": list(sinks_shape) if sinks_shape else [],
            "SINKS_DTYPE": sinks_dtype,
            "SINKS_DATARANGE": sinks_datarange,
            "ATTN_MASK_SHAPE": list(attn_mask_shape) if attn_mask_shape else [],
            "ATTN_MASK_DTYPE": attn_mask_dtype,
            "ATTN_MASK_DATARANGE": attn_mask_datarange,
            "Q_DESCALE_DTYPE": q_descale_dtype,
            "K_DESCALE_DTYPE": k_descale_dtype,
            "V_DESCALE_DTYPE": v_descale_dtype,
            "SEQUSED_Q_DTYPE": seqused_q_dtype,
            "SEQUSED_KV_DTYPE": seqused_kv_dtype,
            "CU_SEQLENS_Q_DTYPE": cu_seqlens_q_dtype,
            "CU_SEQLENS_KV_DTYPE": cu_seqlens_kv_dtype,
            "SOFTMAX_LSE_DTYPE": softmax_lse_dtype,
            "METADATA_SHAPE": list(metadata_shape) if metadata_shape else [],
            "METADATA_DTYPE": metadata_dtype,
        }
    )

    # 数据准备分两种模式:
    #   1) QFA_PASS_THROUGH=1: 跳过 generate_data, CSV 框架分配的 tensor 槽位保持原样
    #      (随机/零初始化) 透传给 wrapper, 由 wrapper 的 _PASS_THROUGH 分支拼 fallback
    #      data_dict 喂 NPU 算子, op_host 拦截非法输入. 用于异常用例.
    #   2) 都不设: 走 generate_data 完整流程; 任意异常也跳过原位写 (CSV tensor 保持原样),
    #      由 wrapper 的 try/except 兜底回退到 fallback data_dict.
    if _PASS_THROUGH:
        logger.info(
            "[INPUTS] QFA_PASS_THROUGH=1, 跳过 generate_data, CSV tensor 保持原样透传给 wrapper"
        )
        return

    golden_mod._inject_physical_s_override(q, v, input_layout, layout_kv)
    try:
        data_dict = golden_mod.generate_data()
    except Exception as e:
        logger.warning(
            "[INPUTS] generate_data 失败: %s, CSV tensor 保持原样透传给 wrapper",
            str(e),
        )
        return
    finally:
        golden_mod._clear_physical_s_override()

    # 原位写 CSV 框架分配的张量 (q/k/v/q_descale/k_descale/v_descale)
    _inplace_write(q, data_dict["q"], "q (slot 0)")
    _inplace_write(k, data_dict["k"], "k (slot 1)")
    _inplace_write(v, data_dict["v"], "v (slot 2)")
    _inplace_write(q_descale, data_dict["q_descale"], "q_descale (slot 3)")
    _inplace_write(k_descale, data_dict["k_descale"], "k_descale (slot 4)")
    _inplace_write(v_descale, data_dict["v_descale"], "v_descale (slot 5)")

    # block_table: CSV 分配了 int32 张量槽位; generate_data 里 continue KV 模式下为 None
    bt_src = data_dict.get("block_table")
    if block_table is not None:
        if bt_src is not None:
            _inplace_write(block_table, bt_src.to(torch.int32), "block_table (slot 6)")
        else:
            # continue KV: block_table 槽位填 0 (算子内部不使用)
            block_table[...] = 0

    logger.info(
        "[INPUTS] in-place wrote MXFP4 q/k/v (q=%s), e8m0 descale (dq=%s, dk=%s, dv=%s), "
        "block_table (pa=%s)",
        tuple(q.shape),
        tuple(q_descale.shape),
        tuple(k_descale.shape),
        tuple(v_descale.shape),
        bt_src is not None,
    )
