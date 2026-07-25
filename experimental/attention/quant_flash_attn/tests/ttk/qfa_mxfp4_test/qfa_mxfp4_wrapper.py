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

import torch
import torch_npu

# 复用 common/ 里的 golden 模块 (generate_data/cpu_mxfp4_golden/npu_mxfp4_fa + 配置区)
_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)
from common import qfa_mxfp4_golden as golden_mod

logger = logging.getLogger(__name__)


def _apply_golden_globals(attrs):
    """把 case attributes 注入 golden 模块全局变量 (B, N_q, ...)."""
    for k, v in attrs.items():
        setattr(golden_mod, k, v)


def npu_qfa_mxfp4(
    # ========== Tensor inputs (TTK 占位, 真实数据从 golden_mod._cached_data 取) ==========
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_descale: torch.Tensor,
    k_descale: torch.Tensor,
    v_descale: torch.Tensor,
    block_table: Optional[torch.Tensor],
    # ========== Scalar / list attrs (从 CSV attributes 解析) ==========
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
    **kwargs,
):
    """MXFP4 quant_flash_attn TTK wrapper.

    graph_path:
      - 0: eager 模式, 直接调用 quant_flash_attn_metadata + quant_flash_attn
      - 7: aclgraph 模式, 用 npugraph_ex backend 编译 Network
      - 优先级: 环境变量 GRAPH_PATH > CSV graph_path 属性 > 默认值 0
    """
    torch_npu.npu.set_device(int(device_id))

    # 优先级: 环境变量 GRAPH_PATH > CSV graph_path 属性 > 默认值 0
    env_graph_path = os.environ.get("GRAPH_PATH")
    if env_graph_path is not None and env_graph_path.strip() != "":
        graph_path = int(env_graph_path)
        logger.info(
            "[WRAPPER] 使用环境变量 GRAPH_PATH=%d (覆盖 CSV graph_path)", graph_path
        )
    else:
        logger.info(
            "[WRAPPER] 环境变量 GRAPH_PATH 未设置, 使用 CSV graph_path=%d", graph_path
        )

    # 注入 golden 全局变量 (generate_data / cpu_mxfp4_golden / npu_mxfp4_fa 都读这些)
    _apply_golden_globals(
        {
            "B": B,
            "N_q": N_q,
            "N_kv": N_kv,
            "G": G,
            "D": D,
            "V_D": V_D,
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
            "ACT_SEQ_LENS_Q": list(act_seq_lens_q),
            "ACT_SEQ_LENS_KV": list(act_seq_lens_kv),
            "MAX_SEQLEN_Q": max_seqlen_q,
            "MAX_SEQLEN_KV": max_seqlen_kv,
            "CU_SEQLENS_Q": list(cu_seqlens_q) if cu_seqlens_q else [],
            "CU_SEQLENS_KV": list(cu_seqlens_kv) if cu_seqlens_kv else [],
            # 可选 tensor 入参 (shape 为空 -> golden 传 None)
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
            # dtype 透传 (表格不传 -> None, golden 侧用默认值)
            "Q_DESCALE_DTYPE": q_descale_dtype,
            "K_DESCALE_DTYPE": k_descale_dtype,
            "V_DESCALE_DTYPE": v_descale_dtype,
            "SEQUSED_Q_DTYPE": seqused_q_dtype,
            "SEQUSED_KV_DTYPE": seqused_kv_dtype,
            "CU_SEQLENS_Q_DTYPE": cu_seqlens_q_dtype,
            "CU_SEQLENS_KV_DTYPE": cu_seqlens_kv_dtype,
            "SOFTMAX_LSE_DTYPE": softmax_lse_dtype,
        }
    )

    # 从 golden_mod 取 customize_inputs 缓存的真实 MXFP4 数据
    cached = getattr(golden_mod, "_cached_data", None)
    if cached is None:
        logger.warning("[WRAPPER] customize_inputs 缓存为空, 重新生成数据")
        cached = golden_mod.generate_data()
        golden_mod._cached_data = cached
    else:
        logger.info("[WRAPPER] 使用 customize_inputs 缓存的真实 MXFP4 数据")

    # 调用 golden 的 npu_mxfp4_fa (内部根据 GRAPH_PATH 选 eager / npugraph_ex)
    logger.info(
        "[WRAPPER] graph_path=%d, layout=%s, B=%d, N_q=%d, N_kv=%d, D=%d",
        graph_path,
        input_layout,
        B,
        N_q,
        N_kv,
        D,
    )
    try:
        atten_out, lse_out = golden_mod.npu_mxfp4_fa(cached)
    except Exception as e:
        logger.error("[WRAPPER] NPU 调用失败: %s", str(e))
        raise

    return atten_out, lse_out
