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
import torch_npu

# 复用 common/ 里的 npu_mxfp8_fa + layout 转换 / e8m0 打包函数
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


def npu_qfa_mxfp8(
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
    """
    graph_path 支持:
      - 0: eager 模式,直接调用 API
      - 7: aclgraph 模式,用 npugraph_ex 编译(在 golden 代码中处理)
      - 优先级: 环境变量 GRAPH_PATH > CSV graph_path 属性 > 默认值 0
    """
    torch_npu.npu.set_device(int(device_id))
    # 优先级: 环境变量 GRAPH_PATH > CSV graph_path 属性 > 默认值 0
    env_graph_path = os.environ.get("GRAPH_PATH")
    if env_graph_path is not None:
        graph_path = int(env_graph_path)
        logger.info(
            "[WRAPPER] 使用环境变量 GRAPH_PATH=%d (覆盖 CSV graph_path)", graph_path
        )

    p_scale_val = (
        float(p_scale.item()) if isinstance(p_scale, torch.Tensor) else float(p_scale)
    )

    # cu_seqlens → actual_seq (差分还原,用于 golden 全局变量)
    cu_seqlens_q_list = list(cu_seqlens_q) if cu_seqlens_q is not None else None
    cu_seqlens_kv_list = list(cu_seqlens_kv) if cu_seqlens_kv is not None else None

    # 注入 golden 全局变量,npu_mxfp8_fa 读这些
    # golden 模块使用 CU_SEQLENS_Q/KV + SEQUSED_Q/KV + MAX_SEQLEN_Q/KV
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

    fp8_dtype = torch.float8_e4m3fn
    group_size = 32
    fp8_max = 448.0

    def _to_cpu(t):
        return (
            t.detach().cpu()
            if isinstance(t, torch.Tensor) and t.device.type == "npu"
            else t
        )

    q_cpu = _to_cpu(q)
    k_cpu = _to_cpu(k)
    v_cpu = _to_cpu(v)
    p_scale_cpu = _to_cpu(p_scale)
    block_table_cpu = _to_cpu(block_table)

    quant_scale_q = golden_mod.get_mxfp8_per_token_group_quant_scale(
        q_cpu, fp8_dtype, group_size
    )
    quant_scale_k = golden_mod.get_mxfp8_per_token_group_quant_scale(
        k_cpu, fp8_dtype, group_size
    )
    quant_scale_v = golden_mod.get_mxfp8_per_channel_group_quant_scale(
        v_cpu, fp8_dtype, group_size
    )

    q_fp8 = (
        golden_mod.mxfp8_per_token_group_quant(q_cpu, quant_scale_q, group_size)
        .clamp(-fp8_max, fp8_max)
        .to(fp8_dtype)
    )
    k_fp8 = (
        golden_mod.mxfp8_per_token_group_quant(k_cpu, quant_scale_k, group_size)
        .clamp(-fp8_max, fp8_max)
        .to(fp8_dtype)
    )
    v_fp8 = (
        golden_mod.mxfp8_per_channel_group_quant(v_cpu, quant_scale_v, group_size)
        .clamp(-fp8_max, fp8_max)
        .to(fp8_dtype)
    )

    # 调用 golden 的 npu_mxfp8_fa
    # - graph_path=0: eager 模式,直接调用 API
    # - graph_path=7: aclgraph 模式,用 npugraph_ex 编译
    logger.info("[WRAPPER] graph_path=%d, 调用 npu_mxfp8_fa", graph_path)
    try:
        atten_out, lse_out = golden_mod.npu_mxfp8_fa(
            q_fp8,
            k_fp8,
            v_fp8,
            quant_scale_q,
            quant_scale_k,
            quant_scale_v,
            p_scale_cpu,
            cu_seqlens_q_list,
            cu_seqlens_kv_list,
            list(seqused_q) if seqused_q is not None else None,
            list(seqused_kv) if seqused_kv is not None else None,
            max_seqlen_q,
            max_seqlen_kv,
            block_table_cpu
            if isinstance(block_table_cpu, torch.Tensor) and enable_pa
            else None,
        )
    except Exception as e:
        logger.error("[WRAPPER] NPU 调用失败: %s", str(e))
        raise

    if not enable_lse:
        lse_out = None
    elif isinstance(lse_out, torch.Tensor) and lse_out.ndim == 2:
        lse_out = lse_out.reshape(lse_out.shape[1], lse_out.shape[0]).contiguous()
    return atten_out, lse_out
