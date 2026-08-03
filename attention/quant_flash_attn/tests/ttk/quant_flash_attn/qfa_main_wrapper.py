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

"""Main-op wrapper for quant_flash_attn mxfp8 split testing (T4).

内部重建 metadata (调 quant_flash_attn_metadata), 再调 quant_flash_attn 主算子。
配合 qfa_mxfp8_excel_main.csv 使用, 验证主算子 + metadata 完整链路。

TTK test case 是隔离进程, 不能跨 case 内存传 metadata; run_main 接收与 run_metadata
相同的输入参数, 内部重新调 metadata 算子重建 metadata 对象, 再喂给主算子。
"""

import logging
import os
import sys
from typing import List

import torch
import torch_npu

try:
    from cann_ops_transformer.ops import quant_flash_attn_metadata, quant_flash_attn

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


def run_main(
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
    """重建 metadata 后调 quant_flash_attn 主算子, 返回 [atten_out, lse_out]。

    分离测试入口: 验证主算子 + metadata 完整链路。
    metadata 不跨 testcase 传递, 本函数内部重新调 metadata 算子重建。
    """
    torch_npu.npu.set_device(int(device_id))

    env_graph_path = os.environ.get("GRAPH_PATH")
    if env_graph_path is not None:
        graph_path = int(env_graph_path)
        logger.info("[MAIN_WRAPPER] 使用环境变量 GRAPH_PATH=%d", graph_path)
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

    inputs = golden_mod.prepare_npu_inputs(
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

    logger.info("[MAIN_WRAPPER] 重建 metadata (调 quant_flash_attn_metadata)")
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
        logger.error("[MAIN_WRAPPER] quant_flash_attn_metadata 重建失败: %s", str(e))
        raise

    main_kwargs = dict(
        q=inputs["q"],
        k=inputs["k"],
        v=inputs["v"],
        q_descale=inputs["dequant_scale_q"],
        k_descale=inputs["dequant_scale_k"],
        v_descale=inputs["dequant_scale_v"],
        quant_mode=1,
        block_table=inputs["block_table"],
        p_scale=inputs["p_scale"],
        cu_seqlens_q=cu_seqlens_q_t if is_tnd_q else None,
        cu_seqlens_kv=cu_seqlens_kv_t if is_tnd_kv else None,
        seqused_q=seqused_q_t,
        seqused_kv=seqused_kv_t,
        attn_mask=inputs["mask"],
        metadata=metadata,
        softmax_scale=inputs["softmax_scale"],
        mask_mode=inputs["sparse_mode"],
        layout_q=layout_q,
        layout_q_descale=inputs["layout_q_descale"],
        layout_kv=layout_kv,
        layout_out=inputs["layout_out"],
        max_seqlen_q=inputs["max_seqlen_q"],
        max_seqlen_kv=inputs["max_seqlen_kv"],
        return_softmax_lse=enable_lse,
    )

    logger.info("[MAIN_WRAPPER] 调用 quant_flash_attn 主算子")
    try:
        atten_out, lse_out = quant_flash_attn(**main_kwargs)
    except Exception as e:
        logger.error("[MAIN_WRAPPER] quant_flash_attn 调用失败: %s", str(e))
        raise

    torch.npu.synchronize()

    act_seqused_q = golden_mod._actual_seq_q()
    T_actual = (
        cu_seqlens_q_list[-1]
        if cu_seqlens_q_list is not None and len(cu_seqlens_q_list) > 1
        else sum(act_seqused_q)
    )
    if atten_out.shape[0] > T_actual:
        atten_out = atten_out[:T_actual]
    logger.info("[MAIN_WRAPPER] output=%s", atten_out.shape)

    #   enable_lse=False → lse_out=None (与 golden slot 1 None 双向匹配, replay load_goldens 通过)
    #   enable_lse=True  → NPU lse_out (N_q, T) T-major 列优先 → transpose 成 (T, N_q) T-major
    #                      contiguous, 与 golden (T, N_q) shape 和内存都对齐。
    if not enable_lse:
        lse_out = None
    elif isinstance(lse_out, torch.Tensor) and lse_out.ndim == 2:
        lse_out = lse_out.transpose(0, 1).contiguous()
    return atten_out, lse_out
