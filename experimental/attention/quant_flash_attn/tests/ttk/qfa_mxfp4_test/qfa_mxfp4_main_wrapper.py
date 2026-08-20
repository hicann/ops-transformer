#!/usr/bin/python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms and conditions of
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

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)
from common import qfa_mxfp4_golden as golden_mod

logger = logging.getLogger(__name__)


# 环境变量 QFA_PASS_THROUGH=1 时完全跳过 golden_mod.generate_data() (含 _resolve_s /
# transpose_qscale / rearrange_by_layout / v_descale.view 等所有会报错的步骤),
# 直接用 CSV 定义的 tensor shape 透传给 NPU 主算子, 由 op_host checker 拦截非法输入。
# 用途: 异常用例测试 (s=0 / k v shape 不一致 / D 不支持 / descale 维度异常 / seq 缺失 等),
#       在 CSV 里配异常 shape + 设 QFA_PASS_THROUGH=1 即可, 框架按 CSV shape 随机生成 tensor
#       原样透传, 不进 golden。
_PASS_THROUGH = os.environ.get("QFA_PASS_THROUGH", "").lower() in ("1", "true", "yes")


def _apply_golden_globals(attrs):
    """把 case attributes 注入 golden 模块全局变量."""
    for k, v in attrs.items():
        setattr(golden_mod, k, v)


def _build_fallback_data_dict(
    q,
    k,
    v,
    q_descale,
    k_descale,
    v_descale,
    block_table,
    B,
    N_q,
    N_kv,
    D,
    V_D,
    act_seq_lens_q,
    act_seq_lens_kv,
    input_layout,
    layout_kv,
    layout_out,
    softmax_scale,
    cu_seqlens_q,
    cu_seqlens_kv,
    max_seqlen_q=-1,
    max_seqlen_kv=-1,
):
    """异常用例回退: generate_data 失败或检测到异常 shape/seq 时用 CSV tensor 直接拼
    最小 data_dict, 跳过 golden 让 NPU 先跑, 交给 op_host 拦截非法输入 (如 s2=0).

    block_table/p_scale/sinks/attn_mask: 透传或按 CSV attributes 生成, 保证异常 shape
    能传到算子让 op_host checker 拦截.

    seq_lens 处理:
      - CSV 传了 act_seq_lens -> 用之
      - CSV 没传 act 但 max>=0 -> 用 max * B
      - CSV act=[] 且 max<0 -> 保持空 list, 让算子拦截 "seq 完全未指定"
    """
    import math

    qk_d = D
    s1_phys = q.shape[2] if q.ndim >= 3 and q.shape[0] == B else None
    s2_phys = k.shape[2] if k.ndim >= 3 and k.shape[0] == B else None

    # Q seq: act 优先, 否则用 max (>=0), 都没给则留空让算子拦截
    if act_seq_lens_q:
        act_q_eff = list(act_seq_lens_q)
    elif max_seqlen_q is not None and max_seqlen_q >= 0:
        act_q_eff = [int(max_seqlen_q)] * B
    else:
        act_q_eff = []
    # KV seq: 同上
    if act_seq_lens_kv:
        act_kv_eff = list(act_seq_lens_kv)
    elif max_seqlen_kv is not None and max_seqlen_kv >= 0:
        act_kv_eff = [int(max_seqlen_kv)] * B
    else:
        act_kv_eff = []

    # 可选 tensor: 透传 block_table; 按 golden 全局变量生成 p_scale/sinks/attn_mask
    # 这样 CSV 配的异常 shape (如 p_scale_shape=[2,3]) 也能传到算子被 checker 拦截
    block_table_t = (
        block_table
        if block_table is not None
        else golden_mod._gen_opt_tensor(
            golden_mod.BLOCK_TABLE_SHAPE,
            golden_mod.BLOCK_TABLE_DTYPE or "int32",
            golden_mod.BLOCK_TABLE_DATARANGE,
            seed=100,
        )
    )
    if golden_mod.P_SCALE_VALUE is not None:
        p_scale_dt = golden_mod.get_dtype(golden_mod.P_SCALE_DTYPE) or torch.float32
        p_scale_t = torch.tensor(
            [float(golden_mod.P_SCALE_VALUE)], dtype=p_scale_dt
        ).reshape([1] * len(golden_mod.P_SCALE_SHAPE or [1]))
        if golden_mod.P_SCALE_SHAPE:
            p_scale_t = p_scale_t.reshape(golden_mod.P_SCALE_SHAPE)
    else:
        p_scale_t = golden_mod._gen_opt_tensor(
            golden_mod.P_SCALE_SHAPE,
            golden_mod.P_SCALE_DTYPE or "float32",
            golden_mod.P_SCALE_DATARANGE,
            seed=101,
        )
    sinks_t = golden_mod._gen_opt_tensor(
        golden_mod.SINKS_SHAPE,
        golden_mod.SINKS_DTYPE or "float32",
        golden_mod.SINKS_DATARANGE,
        seed=102,
    )
    attn_mask_t = golden_mod._gen_opt_tensor(
        golden_mod.ATTN_MASK_SHAPE,
        golden_mod.ATTN_MASK_DTYPE or "int8",
        golden_mod.ATTN_MASK_DATARANGE,
        seed=103,
    )

    return dict(
        q=q.contiguous(),
        k=k.contiguous(),
        v=v.contiguous(),
        q_descale=q_descale.contiguous(),
        k_descale=k_descale.contiguous(),
        v_descale=v_descale.contiguous(),
        block_table=block_table_t,
        p_scale=p_scale_t,
        sinks=sinks_t,
        attn_mask=attn_mask_t,
        cu_seqlens_q=list(cu_seqlens_q) if cu_seqlens_q else None,
        cu_seqlens_kv=list(cu_seqlens_kv) if cu_seqlens_kv else None,
        act_seq_lens_q=act_q_eff,
        act_seq_lens_kv=act_kv_eff,
        s1_physical=s1_phys,
        s2_physical=s2_phys,
        s1_effective=max(act_q_eff) if act_q_eff else (s1_phys or 0),
        s2_effective=max(act_kv_eff) if act_kv_eff else (s2_phys or 0),
        act_seq_q_eff=act_q_eff,
        act_seq_kv_eff=act_kv_eff,
        query_layout=input_layout,
        kv_layout=layout_kv,
        attn_out_layout=layout_out,
        num_heads=N_q,
        num_key_value_heads=N_kv,
        softmax_scale=softmax_scale
        if softmax_scale is not None
        else 1.0 / math.sqrt(qk_d),
        fp32_bnsd=None,
    )


def run_main(
    # ========== Tensor inputs (TTK 占位, 真实数据由 golden_mod.generate_data 重新生成) ==========
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
    # metadata 伪 tensor 入参 (表格 metadata_shape/metadata_dtype, 不传则 golden 用默认)
    metadata_shape: List[int] = None,
    metadata_dtype: str = None,
    **kwargs,
):
    """构造 metadata 后调 quant_flash_attn 主算子, 返回 [atten_out, lse_out]。

    分离测试入口: 验证主算子 + metadata 完整链路。
    metadata 不跨 testcase 传递, 本函数内部按表格 metadata_shape/metadata_dtype
    构造伪 tensor 传入主算子。
    """
    torch_npu.npu.set_device(int(device_id))

    env_graph_path = os.environ.get("GRAPH_PATH")
    if env_graph_path is not None and env_graph_path.strip() != "":
        graph_path = int(env_graph_path)
        logger.info("[MAIN_WRAPPER] 使用环境变量 GRAPH_PATH=%d", graph_path)

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

    # 表格全量透传列 (CSV attrs 里的 extra keys 走 kwargs) 注入 golden,
    # QFA_PASS_THROUGH=1 时按表格值构造算子入参
    golden_mod._apply_kwargs_globals(kwargs)

    # 数据准备分两种模式:
    #   1) QFA_PASS_THROUGH=1: 完全跳过 generate_data, CSV tensor 原样透传给 NPU 主算子,
    #      由 op_host 拦截非法输入. 用于异常用例 (s=0 / kv shape 不一致 / D 不支持 /
    #      descale 维度异常 / seq 缺失 等), 在 CSV 配异常 shape + 设本开关即可.
    #   2) 都不设: 走 generate_data 完整流程; 任意异常也回退到 CSV tensor 透传 (try/except 兜底).
    if _PASS_THROUGH:
        logger.info(
            "[MAIN_WRAPPER] QFA_PASS_THROUGH=1, 跳过 generate_data, CSV tensor 直接透传给 NPU 主算子"
        )
        data_dict = _build_fallback_data_dict(
            q,
            k,
            v,
            q_descale,
            k_descale,
            v_descale,
            block_table,
            B,
            N_q,
            N_kv,
            D,
            V_D,
            act_seq_lens_q,
            act_seq_lens_kv,
            input_layout,
            layout_kv,
            layout_out,
            softmax_scale,
            cu_seqlens_q,
            cu_seqlens_kv,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv,
        )
    else:
        golden_mod._inject_physical_s_override(q, v, input_layout, layout_kv)
        try:
            data_dict = golden_mod.generate_data()
            golden_mod._apply_csv_shape_override(
                data_dict,
                {
                    "q": q,
                    "k": k,
                    "v": v,
                    "q_descale": q_descale,
                    "k_descale": k_descale,
                    "v_descale": v_descale,
                },
            )
        except Exception as e:
            logger.warning(
                "[MAIN_WRAPPER] generate_data 失败: %s, 回退到 CSV tensor 透传给 NPU 让 op_host 拦截",
                str(e),
            )
            data_dict = _build_fallback_data_dict(
                q,
                k,
                v,
                q_descale,
                k_descale,
                v_descale,
                block_table,
                B,
                N_q,
                N_kv,
                D,
                V_D,
                act_seq_lens_q,
                act_seq_lens_kv,
                input_layout,
                layout_kv,
                layout_out,
                softmax_scale,
                cu_seqlens_q,
                cu_seqlens_kv,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_kv=max_seqlen_kv,
            )
        finally:
            golden_mod._clear_physical_s_override()
    logger.info(
        "[MAIN_WRAPPER] graph_path=%d, layout=%s, B=%d, N_q=%d, N_kv=%d, D=%d, pass_through=%s",
        graph_path,
        input_layout,
        B,
        N_q,
        N_kv,
        D,
        _PASS_THROUGH,
    )
    try:
        atten_out, lse_out = golden_mod.call_npu_main(data_dict)
    except Exception as e:
        logger.error("[MAIN_WRAPPER] quant_flash_attn 主算子调用失败: %s", str(e))
        raise

    # lse 处理 (与 e2e wrapper 一致):
    #   enable_lse=False → lse_out=None (与 golden slot 1 None 双向匹配)
    #   enable_lse=True  → NPU lse_out (N_q, T) T-major 列优先 → transpose 成 (T, N_q) T-major
    if not enable_lse:
        lse_out = None
    elif isinstance(lse_out, torch.Tensor) and lse_out.ndim == 2:
        lse_out = lse_out.transpose(0, 1).contiguous()

    return atten_out, lse_out
