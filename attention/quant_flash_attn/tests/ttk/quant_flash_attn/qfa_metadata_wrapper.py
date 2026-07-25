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

"""Metadata-only wrapper for quant_flash_attn mxfp8 split testing (T4).

只调 quant_flash_attn_metadata (metadata-only op), 返回 metadata 对象。
配合 qfa_mxfp8_excel_metadata.csv 使用, 验证 metadata 算子独立正确性。
"""

import logging
import os
import sys
from typing import List

import torch
import torch_npu

try:
    from cann_ops_transformer.ops import quant_flash_attn_metadata

    _HAS_NPU = True
except ImportError as e:
    _HAS_NPU = False

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
import quant_flash_attn_golden as golden_mod

logger = logging.getLogger(__name__)


def _apply_golden_globals(params):
    """把 case 参数注入 golden 模块全局变量。"""
    for k, v in params.items():
        setattr(golden_mod, k, v)


def run_metadata(
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
    """只调 quant_flash_attn_metadata, 返回 [metadata]。

    分离测试入口: 验证 metadata 算子独立正确性, 不调主算子。
    """
    torch_npu.npu.set_device(int(device_id))

    env_graph_path = os.environ.get("GRAPH_PATH")
    if env_graph_path is not None:
        graph_path = int(env_graph_path)
        logger.info("[METADATA_WRAPPER] 使用环境变量 GRAPH_PATH=%d", graph_path)

    cached = getattr(golden_mod, "_cached_mxfp8_inputs", None)
    if cached is not None:
        (
            q,
            k,
            v,
            dequant_scale_q,
            dequant_scale_k,
            dequant_scale_v,
            p_scale,
            block_table,
        ) = cached[:8]
        logger.info("[METADATA_WRAPPER] 使用 customize_inputs 缓存的真实 FP8 数据")
    else:
        logger.warning(
            "[METADATA_WRAPPER] customize_inputs 缓存为空,使用 TTK 传入的占位 tensor(可能出错)"
        )

    p_scale_val = (
        float(p_scale.item()) if isinstance(p_scale, torch.Tensor) else float(p_scale)
    )

    cu_seqlens_q_list = list(cu_seqlens_q) if cu_seqlens_q is not None else [0]
    cu_seqlens_kv_list = list(cu_seqlens_kv) if cu_seqlens_kv is not None else [0]

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
            "P_SCALE": p_scale_val,
            "ENABLE_LSE": enable_lse,
            "FP8_DTYPE": torch.float8_e4m3fn,
            "QUANT_GROUP_SIZE": 32,
            "INPUT_LAYOUT": input_layout,
            "IS_CONTIGUOUS": is_contiguous,
            "DEVICE_ID": device_id,
            "GRAPH_PATH": graph_path,
            "SOFTMAX_SCALE": softmax_scale,
            "SEED_Q": kwargs.get("seed_q"),
            "SEED_K": kwargs.get("seed_k"),
            "SEED_V": kwargs.get("seed_v"),
            "DATA_RANGE_Q": kwargs.get("data_range_q"),
            "DATA_RANGE_K": kwargs.get("data_range_k"),
            "DATA_RANGE_V": kwargs.get("data_range_v"),
        }
    )

    inputs = golden_mod.prepare_npu_inputs(
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
        block_table if isinstance(block_table, torch.Tensor) else None,
    )

    cu_seqlens_q_t = (
        torch.tensor(inputs["cu_seqlens_q"], dtype=torch.int32).npu()
        if inputs["cu_seqlens_q"] is not None
        else None
    )
    cu_seqlens_kv_t = (
        torch.tensor(inputs["cu_seqlens_kv"], dtype=torch.int32).npu()
        if inputs["cu_seqlens_kv"] is not None
        else None
    )
    seqused_q_t = (
        torch.tensor(inputs["seqused_q"], dtype=torch.int32).npu()
        if inputs["seqused_q"] is not None
        else None
    )
    seqused_kv_t = (
        torch.tensor(inputs["seqused_kv"], dtype=torch.int32).npu()
        if inputs["seqused_kv"] is not None
        else None
    )

    layout_q = inputs["layout_q"]
    layout_kv = inputs["layout_kv"]
    is_tnd_q = layout_q == "TND"
    is_tnd_kv = layout_kv == "TND"

    torch.npu.synchronize()

    logger.info("[METADATA_WRAPPER] 调用 quant_flash_attn_metadata")
    try:
        metadata = quant_flash_attn_metadata(
            num_heads_q=inputs["q_n"],
            num_heads_kv=inputs["kv_n"],
            head_dim=inputs["q"].shape[-1],
            quant_mode=1,
            cu_seqlens_q=cu_seqlens_q_t if is_tnd_q else None,
            cu_seqlens_kv=cu_seqlens_kv_t if is_tnd_kv else None,
            seqused_q=seqused_q_t,
            seqused_kv=seqused_kv_t,
            dequant_scale_v=inputs["dequant_scale_v"],
            mask_mode=inputs["sparse_mode"],
            layout_q=layout_q,
            layout_q_descale=inputs["layout_q_descale"],
            layout_kv=layout_kv,
            layout_out=inputs["layout_out"],
            max_seqlen_q=inputs["max_seqlen_q"],
            max_seqlen_kv=inputs["max_seqlen_kv"],
        )
    except Exception as e:
        logger.error(
            "[METADATA_WRAPPER] quant_flash_attn_metadata 调用失败: %s", str(e)
        )
        raise

    return [metadata]
