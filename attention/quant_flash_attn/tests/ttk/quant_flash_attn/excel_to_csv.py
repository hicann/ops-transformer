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
"""
把 redline.xlsx 的 mxfp8 sheet 转成符合 ttk e2e 标准的 csv。

环境：无 openpyxl/pandas，xlsx 解析走 _xlsx_minireader（zipfile + minidom）。

输入：redline.xlsx（同目录），xl/worksheets/sheet1.xml 是 mxfp8 sheet
输出：3 个 csv 到同目录
  - qfa_mxfp8_excel.csv            api_name=qfa_wrapper.npu_qfa
  - qfa_mxfp8_excel_metadata.csv   api_name=qfa_metadata_wrapper.run_metadata
  - qfa_mxfp8_excel_main.csv       api_name=qfa_main_wrapper.run_main

空值契约：excel 空单元格 → csv attributes 省略该 key（不写 key:None）。
"""

import csv
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from _xlsx_minireader import col_to_idx, load_sheet, load_sheet1, list_sheet_names, header_to_name_idx

_XLSX = os.path.join(_HERE, "redline.xlsx")

# 11 列 header
HEADER = [
    "testcase_name",
    "api_name",
    "tensor_view_shapes",
    "tensor_dtypes",
    "tensor_formats",
    "attributes",
    "output_tensor_indexes",
    "golden_api",
    "input_data_ranges",
    "precision_tolerances",
    "absolute_precision",
]

# 3 个 csv 的 api_name 和 testcase_name 后缀 (MXFP8, quant_mode=1)
CSV_PROFILES_MXFP8 = [
    ("qfa_mxfp8_excel.csv", "qfa_wrapper.npu_qfa", ""),
    ("qfa_mxfp8_excel_metadata.csv", "qfa_metadata_wrapper.run_metadata", "_metadata"),
    ("qfa_mxfp8_excel_main.csv", "qfa_main_wrapper.run_main", "_main"),
]

# 3 个 csv 的 api_name 和 testcase_name 后缀 (GQA FP8 全量化, quant_mode=6)
# api_name 与 MXFP8 相同 (wrapper 复用, 内部按 quant_mode 分支),
# 仅 CSV 文件名和 testcase 后缀区分, 避免与 mxfp8 testcase 重名。
CSV_PROFILES_GQA_FP8 = [
    ("qfa_gqa_fp8_excel.csv", "qfa_wrapper.npu_qfa", "_gqa_fp8"),
    ("qfa_gqa_fp8_excel_metadata.csv", "qfa_metadata_wrapper.run_metadata", "_gqa_fp8_metadata"),
    ("qfa_gqa_fp8_excel_main.csv", "qfa_main_wrapper.run_main", "_gqa_fp8_main"),
]

# 向后兼容别名 (现有脚本若 import CSV_PROFILES)
CSV_PROFILES = CSV_PROFILES_MXFP8

# excel dtype → csv dtype
DTYPE_MAP = {
    "FLOAT8_E4M3FN": "float8_e4m3fn",
    "BF16": "bfloat16",
    "FLOAT8_E8M0": "float8_e8m0",
    "INT32": "int32",
    "INT8": "int8",
    "FP32": "float32",
    "FLOAT32": "float32",
}

# --- Excel 列索引（mxfp8 sheet 列 A=0 ... CD=81） ---
COL = {
    name: col_to_idx(letter)
    for name, letter in [
        ("A_testcase_name", "A"),
        ("K_attn_out_shape", "K"),
        ("L_attn_out_dtype", "L"),
        ("M_attn_out_datarange", "M"),
        ("N_lse_shape", "N"),
        ("O_lse_dtype", "O"),
        ("P_lse_datarange", "P"),
        ("Q_q_shape", "Q"),
        ("R_q_dtype", "R"),
        ("S_q_datarange", "S"),
        ("T_k_shape", "T"),
        ("U_k_dtype", "U"),
        ("V_k_datarange", "V"),
        ("W_v_shape", "W"),
        ("X_v_dtype", "X"),
        ("Y_v_datarange", "Y"),
        ("Z_qdescale_shape", "Z"),
        ("AA_qdescale_dtype", "AA"),
        ("AB_qdescale_datarange", "AB"),
        ("AC_kdescale_shape", "AC"),
        ("AD_kdescale_dtype", "AD"),
        ("AE_kdescale_datarange", "AE"),
        ("AF_vdescale_shape", "AF"),
        ("AG_vdescale_dtype", "AG"),
        ("AH_vdescale_datarange", "AH"),
        ("AI_block_table_shape", "AI"),
        ("AJ_block_table_dtype", "AJ"),
        ("AK_block_table_datarange", "AK"),
        ("AL_p_scale_value", "AL"),
        ("AM_p_scale_shape", "AM"),
        ("AN_p_scale_dtype", "AN"),
        ("AO_p_scale_datarange", "AO"),
        ("AP_cu_seqlens_q_value", "AP"),
        ("AQ_cu_seqlens_q_shape", "AQ"),
        ("AR_cu_seqlens_q_dtype", "AR"),
        ("AS_cu_seqlens_q_datarange", "AS"),
        ("AT_cu_seqlens_kv_value", "AT"),
        ("AU_cu_seqlens_kv_shape", "AU"),
        ("AV_cu_seqlens_kv_dtype", "AV"),
        ("AW_cu_seqlens_kv_datarange", "AW"),
        ("AX_seqused_q_value", "AX"),
        ("AY_seqused_q_shape", "AY"),
        ("AZ_seqused_q_dtype", "AZ"),
        ("BA_seqused_q_datarange", "BA"),
        ("BB_seqused_kv_value", "BB"),
        ("BC_seqused_kv_shape", "BC"),
        ("BD_seqused_kv_dtype", "BD"),
        ("BE_seqused_kv_datarange", "BE"),
        ("BI_attn_mask_shape", "BI"),
        ("BJ_attn_mask_dtype", "BJ"),
        ("BK_attn_mask_datarange", "BK"),
        ("BO_quant_mode", "BO"),
        ("BP_softmax_scale", "BP"),
        ("BQ_mask_mode", "BQ"),
        ("BR_win_left", "BR"),
        ("BS_win_right", "BS"),
        ("BT_max_seqlen_q", "BT"),
        ("BU_max_seqlen_kv", "BU"),
        ("BV_layout_q", "BV"),
        ("BW_layout_q_descale", "BW"),
        ("BX_layout_kv", "BX"),
        ("BY_layout_out", "BY"),
        ("BZ_return_softmax_lse", "BZ"),
        ("CA_batch_size", "CA"),
        ("CB_num_heads_q", "CB"),
        ("CC_num_heads_kv", "CC"),
        ("CD_head_dim", "CD"),
    ]
}

# COL 逻辑 key → xlsx 表头实际列名 (用于按列名动态定位)
# 不同 sheet 列顺序可能偏移 (如 fp8 sheet 多 blocknum理论值/uncontiguous_dim 两列),
# 按列名查找可避免错位。COL 固定字母表保留作 mxfp8 fallback。
COL_NAME = {
    "A_testcase_name": "testcase_name",
    "K_attn_out_shape": "attn_out_shape",
    "L_attn_out_dtype": "attn_out_dtype",
    "M_attn_out_datarange": "attn_out_datarange",
    "N_lse_shape": "softmax_lse_shape",
    "O_lse_dtype": "softmax_lse_dtype",
    "P_lse_datarange": "softmax_lse_datarange",
    "Q_q_shape": "q_shape",
    "R_q_dtype": "q_dtype",
    "S_q_datarange": "q_datarange",
    "T_k_shape": "k_shape",
    "U_k_dtype": "k_dtype",
    "V_k_datarange": "k_datarange",
    "W_v_shape": "v_shape",
    "X_v_dtype": "v_dtype",
    "Y_v_datarange": "v_datarange",
    "Z_qdescale_shape": "q_descale_shape",
    "AA_qdescale_dtype": "q_descale_dtype",
    "AB_qdescale_datarange": "q_descale_datarange",
    "AC_kdescale_shape": "k_descale_shape",
    "AD_kdescale_dtype": "k_descale_dtype",
    "AE_kdescale_datarange": "k_descale_datarange",
    "AF_vdescale_shape": "v_descale_shape",
    "AG_vdescale_dtype": "v_descale_dtype",
    "AH_vdescale_datarange": "v_descale_datarange",
    "AI_block_table_shape": "block_table_shape",
    "AJ_block_table_dtype": "block_table_dtype",
    "AK_block_table_datarange": "block_table_datarange",
    "AL_p_scale_value": "p_scale_value",
    "AM_p_scale_shape": "p_scale_shape",
    "AN_p_scale_dtype": "p_scale_dtype",
    "AO_p_scale_datarange": "p_scale_datarange",
    "AP_cu_seqlens_q_value": "cu_seqlens_q_value",
    "AQ_cu_seqlens_q_shape": "cu_seqlens_q_shape",
    "AR_cu_seqlens_q_dtype": "cu_seqlens_q_dtype",
    "AS_cu_seqlens_q_datarange": "cu_seqlens_q_datarange",
    "AT_cu_seqlens_kv_value": "cu_seqlens_kv_value",
    "AU_cu_seqlens_kv_shape": "cu_seqlens_kv_shape",
    "AV_cu_seqlens_kv_dtype": "cu_seqlens_kv_dtype",
    "AW_cu_seqlens_kv_datarange": "cu_seqlens_kv_datarange",
    "AX_seqused_q_value": "seqused_q_value",
    "AY_seqused_q_shape": "seqused_q_shape",
    "AZ_seqused_q_dtype": "seqused_q_dtype",
    "BA_seqused_q_datarange": "seqused_q_datarange",
    "BB_seqused_kv_value": "seqused_kv_value",
    "BC_seqused_kv_shape": "seqused_kv_shape",
    "BD_seqused_kv_dtype": "seqused_kv_dtype",
    "BE_seqused_kv_datarange": "seqused_kv_datarange",
    "BI_attn_mask_shape": "attn_mask_shape",
    "BJ_attn_mask_dtype": "attn_mask_dtype",
    "BK_attn_mask_datarange": "attn_mask_datarange",
    "BO_quant_mode": "quant_mode",
    "BP_softmax_scale": "softmax_scale",
    "BQ_mask_mode": "mask_mode",
    "BR_win_left": "win_left",
    "BS_win_right": "win_right",
    "BT_max_seqlen_q": "max_seqlen_q",
    "BU_max_seqlen_kv": "max_seqlen_kv",
    "BV_layout_q": "layout_q",
    "BW_layout_q_descale": "layout_q_descale",
    "BX_layout_kv": "layout_kv",
    "BY_layout_out": "layout_out",
    "BZ_return_softmax_lse": "return_softmax_lse",
    "CA_batch_size": "batch_size",
    "CB_num_heads_q": "num_heads_q",
    "CC_num_heads_kv": "num_heads_kv",
    "CD_head_dim": "head_dim",
}


def _build_col_by_name(header_row: dict) -> dict:
    """从 sheet 表头行构建 dict[COL逻辑key→col_idx], 按列名动态定位。

    优先用表头列名查找 (COL_NAME 映射); 若表头缺某列, fallback 到 COL 固定字母
    (mxfp8 sheet 向后兼容)。
    """
    name_idx = header_to_name_idx(header_row)
    col_by_name = {}
    for logical_key, col_name in COL_NAME.items():
        if col_name in name_idx:
            col_by_name[logical_key] = name_idx[col_name]
        elif logical_key in COL:
            col_by_name[logical_key] = COL[logical_key]
    return col_by_name

ABSOLUTE_PRECISION_DEFAULT = 1e-8


# --- 值转换器：excel 字符串 → python 值 ---
def _str_to_shape(s):
    """'4096,20,64' → (4096, 20, 64)。空 → None。"""
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    return tuple(int(p) for p in parts)


def _str_to_datarange(s):
    """'-1,1' → (-1, 1)。空 → None。支持 '[-128, 127]' 这种带括号的写法。"""
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    s = s.strip("[]()")
    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    if len(parts) != 2:
        raise ValueError(f"unexpected datarange: {s!r}")
    return (float(parts[0]), float(parts[1]))


def _str_to_int(s):
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    return int(float(s))


def _str_to_float(s):
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    return float(s)


def _str_to_int_list(s):
    """'0,4096' → [0, 4096]。空 → None。"""
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    return [int(float(p)) for p in parts]


def _map_dtype(s):
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    if s not in DTYPE_MAP:
        raise ValueError(f"unknown excel dtype: {s!r}")
    return DTYPE_MAP[s]


def _bool_to_int(s):
    """Excel 'TRUE'/'FALSE'/'1'/'0' → 1/0。空 → None。"""
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    if s in ("True", "TRUE", "1", "true"):
        return 1
    if s in ("False", "FALSE", "0", "false"):
        return 0
    raise ValueError(f"unexpected bool: {s!r}")


def _strip_or_none(s):
    if s is None:
        return None
    s = s.strip()
    return s or None


# --- 每行映射：excel row → csv record ---
# Tensor 顺序（索引），与 wrapper 签名一致：
#   npu_qfa(q, k, v, dequant_scale_q, dequant_scale_k, dequant_scale_v,
#                 p_scale, block_table, *, B, N_q, ...)
#   0 q              (Q-S)
#   1 k              (T-V)
#   2 v              (W-Y)
#   3 q_descale      (Z-AB)    datarange 不映射
#   4 k_descale      (AC-AE)   datarange 不映射
#   5 v_descale      (AF-AH)   datarange 不映射
#   6 p_scale        (AM-AO)   datarange 不映射（p_scale 是标量）
#   7 block_table    (AI-AK)   空则 None 占位
# attn_mask (BI-BK) 不映射：wrapper 无此参数


def _build_tensor_lists(row, col_by_name):
    shapes = []
    dtypes = []
    data_ranges = []

    # descale dtype 默认值按 quant_mode 分支:
    #   quant_mode=1 (MXFP8) → float8_e8m0
    #   quant_mode=6 (GQA FP8) → float32 (descale 不是 e8m0)
    quant_mode = _str_to_int(row.get(col_by_name["BO_quant_mode"])) or 1
    descale_default_dtype = "float32" if quant_mode == 6 else "float8_e8m0"

    def _push(shape, dtype, drange):
        shapes.append(shape)
        dtypes.append(dtype)
        data_ranges.append(drange)

    # data_range 透传 Excel
    _push(
        _str_to_shape(row.get(col_by_name["Q_q_shape"])),
        _map_dtype(row.get(col_by_name["R_q_dtype"])) or "float8_e4m3fn",
        _str_to_datarange(row.get(col_by_name["S_q_datarange"])),
    )
    _push(
        _str_to_shape(row.get(col_by_name["T_k_shape"])),
        _map_dtype(row.get(col_by_name["U_k_dtype"])) or "float8_e4m3fn",
        _str_to_datarange(row.get(col_by_name["V_k_datarange"])),
    )
    _push(
        _str_to_shape(row.get(col_by_name["W_v_shape"])),
        _map_dtype(row.get(col_by_name["X_v_dtype"])) or "float8_e4m3fn",
        _str_to_datarange(row.get(col_by_name["Y_v_datarange"])),
    )

    _DRANGE_PLACEHOLDER = (0, 1)
    _push(
        _str_to_shape(row.get(col_by_name["Z_qdescale_shape"])),
        _map_dtype(row.get(col_by_name["AA_qdescale_dtype"])) or descale_default_dtype,
        _DRANGE_PLACEHOLDER,
    )
    _push(
        _str_to_shape(row.get(col_by_name["AC_kdescale_shape"])),
        _map_dtype(row.get(col_by_name["AD_kdescale_dtype"])) or descale_default_dtype,
        _DRANGE_PLACEHOLDER,
    )
    _push(
        _str_to_shape(row.get(col_by_name["AF_vdescale_shape"])),
        _map_dtype(row.get(col_by_name["AG_vdescale_dtype"])) or descale_default_dtype,
        _DRANGE_PLACEHOLDER,
    )
    p_scale_shape = _str_to_shape(row.get(col_by_name["AM_p_scale_shape"]))
    p_scale_dtype = _map_dtype(row.get(col_by_name["AN_p_scale_dtype"]))
    if p_scale_shape is None and quant_mode == 6:
        p_scale_shape = (1,)
        p_scale_dtype = "float32"
    _push(p_scale_shape, p_scale_dtype, _DRANGE_PLACEHOLDER)

    bt_dtype = _map_dtype(row.get(col_by_name["AJ_block_table_dtype"])) or "int32"
    bt_shape_excel = _str_to_shape(row.get(col_by_name["AI_block_table_shape"]))
    bt_shape = bt_shape_excel if bt_shape_excel is not None else (0,)
    _push(bt_shape, bt_dtype, _DRANGE_PLACEHOLDER)

    return shapes, dtypes, data_ranges


def _build_attributes(row, col_by_name):
    """空单元格 → 省略 key（不写 key:None）。

    precision_tolerances / absolute_precision 同时写入 attributes dict,
    使其经 testcase.attributes → extra_attrs → golden **kwargs 链路传递到
    compare 插件（ttk 框架不把 testcase 对象直接传给 custom compare）。

    wrapper 必选 keyword-only 参数（无默认值）必须全部有 key，
    否则 ttk build_args 不放入 kwargs → TypeError。
    以下参数是 wrapper 接口适配参数（不是 op 参数推导）：
      - enable_pa: 从 layout_kv 前缀推导（PA_ → True, 否则 False）
      - kv_cache_layout: Excel layout_kv → wrapper 参数名 kv_cache_layout
      - q_scale_layout: Excel layout_q_descale → wrapper 参数名 q_scale_layout
      - block_size: Excel 无此列，PA 模式从 KV shape 第三维提取（算子要求 512 或 1024）
    """
    attrs = {}

    def _set(key, val):
        if val is not None:
            attrs[key] = val

    def _set_force(key, val):
        """必选参数：即使 None 也写入（wrapper 无默认值，需要 key 存在）。"""
        attrs[key] = val

    _set("precision_tolerances", _precision_tolerances(row, col_by_name))
    _set("absolute_precision", ABSOLUTE_PRECISION_DEFAULT)
    _set("p_scale_value", _str_to_float(row.get(col_by_name["AL_p_scale_value"])))
    # cu_seqlens value → list，shape/dtype 不传（由 NPU 推）
    # cu_seqlens_q/kv 是 wrapper 必选参数（无默认值），空则写 None（PA 模式 kv 可能为空）
    _set_force("cu_seqlens_q", _str_to_int_list(row.get(col_by_name["AP_cu_seqlens_q_value"])))
    _set_force(
        "cu_seqlens_kv", _str_to_int_list(row.get(col_by_name["AT_cu_seqlens_kv_value"]))
    )
    # seqused_q/kv：空则写 None（wrapper 无默认值，需要 key 存在，golden 透传 None 给 NPU）
    _set_force("seqused_q", _str_to_int_list(row.get(col_by_name["AX_seqused_q_value"])))
    _set_force("seqused_kv", _str_to_int_list(row.get(col_by_name["BB_seqused_kv_value"])))
    _set("quant_mode", _str_to_int(row.get(col_by_name["BO_quant_mode"])))
    _set("softmax_scale", _str_to_float(row.get(col_by_name["BP_softmax_scale"])))
    _set("mask_mode", _str_to_int(row.get(col_by_name["BQ_mask_mode"])))
    _set("win_left", _str_to_int(row.get(col_by_name["BR_win_left"])))
    _set("win_right", _str_to_int(row.get(col_by_name["BS_win_right"])))
    # max_seqlen_q/kv：空则写 None。 -1 是显式值，保留。
    _set_force("max_seqlen_q", _str_to_int(row.get(col_by_name["BT_max_seqlen_q"])))
    _set_force("max_seqlen_kv", _str_to_int(row.get(col_by_name["BU_max_seqlen_kv"])))
    _set("layout_q", _strip_or_none(row.get(col_by_name["BV_layout_q"])))
    _set("layout_q_descale", _strip_or_none(row.get(col_by_name["BW_layout_q_descale"])))
    _set("layout_kv", _strip_or_none(row.get(col_by_name["BX_layout_kv"])))
    _set("layout_out", _strip_or_none(row.get(col_by_name["BY_layout_out"])))
    _set("enable_lse", _bool_to_int(row.get(col_by_name["BZ_return_softmax_lse"])))
    _set("N_q", _str_to_int(row.get(col_by_name["CB_num_heads_q"])))
    _set("N_kv", _str_to_int(row.get(col_by_name["CC_num_heads_kv"])))
    _set("D", _str_to_int(row.get(col_by_name["CD_head_dim"])))

    # --- wrapper 接口适配参数（不是 op 参数推导，是 wrapper 签名必选参数） ---
    layout_kv = _strip_or_none(row.get(col_by_name["BX_layout_kv"]))
    layout_q_descale = _strip_or_none(row.get(col_by_name["BW_layout_q_descale"]))
    # B: Excel CA 列，空则从 cu_seqlens_q 推（len-1，与 wrapper 约定一致）
    B_val = _str_to_int(row.get(col_by_name["CA_batch_size"]))
    if B_val is None:
        cu_q = _str_to_int_list(row.get(col_by_name["AP_cu_seqlens_q_value"]))
        B_val = max(1, len(cu_q) - 1) if cu_q and len(cu_q) >= 2 else 1
    _set_force("B", B_val)
    # enable_pa: 从 layout_kv 前缀推导（接口适配，不是 op 参数推导）
    _set_force("enable_pa", isinstance(layout_kv, str) and layout_kv.startswith("PA_"))
    # kv_cache_layout: Excel layout_kv → wrapper 参数名
    _set_force("kv_cache_layout", layout_kv)
    # q_scale_layout: Excel layout_q_descale → wrapper 参数名
    _set_force("q_scale_layout", layout_q_descale)
    # block_size: Excel 无此列，PA 模式从 KV shape 第三维提取（算子要求 512 或 1024）
    #   PA_BNBD: [Bn, N, Bs, D]        → shape[2]
    #   PA_NZ:   [Bn, N, D//32, Bs, 32] → shape[3]
    # 非 PA 模式 block_size=0
    # GQA FP8 (quant_mode=6): K cache 第三维含 K_SCALE_ROWS=4 行 FP32 scale,
    #   真实 block_size = K shape[2] - 4
    # MXFP8 (quant_mode=1): K shape 第三维即真实 block_size
    K_SCALE_ROWS_GQA = 4
    if isinstance(layout_kv, str) and layout_kv.startswith("PA_"):
        k_shape = _str_to_shape(row.get(col_by_name["T_k_shape"]))
        if k_shape is not None:
            if layout_kv == "PA_NZ":
                bs_idx = 3
            else:
                bs_idx = 2
            raw_bs = k_shape[bs_idx] if len(k_shape) > bs_idx else 0
            quant_mode_for_bs = _str_to_int(row.get(col_by_name["BO_quant_mode"])) or 1
            if quant_mode_for_bs == 6:
                block_size = max(0, raw_bs - K_SCALE_ROWS_GQA)
            else:
                block_size = raw_bs
        else:
            block_size = 0
    else:
        block_size = 0
    _set_force("block_size", block_size)

    return attrs


def _precision_tolerances(row, col_by_name):
    """默认 ((0.005, 0.000025, 0.005, 0.005, 10),);
    若 attn_out_dtype (L) 含 BF16 → ((0.0078125, 0.0001, 0.005, 0.005, 10),)

    5-tuple = (rtol, atol, diff_thd, pct_thd, max_diff_hd)。后三位为 check_result的三个阈值, 默认 0.005/0.005/10
    """
    out_dtype = _strip_or_none(row.get(col_by_name["L_attn_out_dtype"]))
    if out_dtype and "BF16" in out_dtype.upper():
        return ((0.0078125, 0.0001, 0.005, 0.005, 10),)
    return ((0.005, 0.000025, 0.005, 0.005, 10),)


def _row_to_csv(excel_row, api_name, testcase_suffix, col_by_name):
    """把一行 excel 数据转成一条 csv 行（list of 11 fields）。"""
    raw_name = _strip_or_none(excel_row.get(col_by_name["A_testcase_name"]))
    if not raw_name:
        raise ValueError("testcase_name (A) is empty")
    testcase_name = raw_name + testcase_suffix

    shapes, dtypes, data_ranges = _build_tensor_lists(excel_row, col_by_name)
    attrs = _build_attributes(excel_row, col_by_name)

    # 嵌套字段：双引号包裹的 Python tuple 字面量，csv.writer 自动加引号。
    # attributes 用 repr() 生成单引号 Python dict 字面量（HANDOFF.md 坑2：json.dumps 双引号会失败）。
    return [
        testcase_name,
        api_name,
        repr(tuple(shapes)),
        repr(tuple(dtypes)),
        "",  # tensor_formats 留空，框架默认 ('ND',)
        repr(attrs),
        "",  # output_tensor_indexes 留空（wrapper 输出是返回值，HANDOFF 坑8）
        "",  # golden_api 留空
        repr(tuple(data_ranges)),
        repr(_precision_tolerances(excel_row, col_by_name)),
        repr(ABSOLUTE_PRECISION_DEFAULT),
    ]


def _parse_sheet_arg(s):
    """--sheet 参数解析: 纯数字 → int (1-based 序号), 否则按 sheet 名"""
    if s is None:
        return 1
    s = s.strip()
    if s.isdigit():
        return int(s)
    return s


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="redline.xlsx → 3 个 ttk 标准 CSV (mxfp8 或 gqa_fp8)"
    )
    parser.add_argument(
        "--sheet",
        default=None,
        help="xlsx sheet 序号 (1-based int, 如 1/2) 或 sheet 名 (str, 如 mxfp8/gqa_fp8)。"
        "默认 1 (sheet1)。可用 --list-sheets 查看所有 sheet 名。",
    )
    parser.add_argument(
        "--list-sheets",
        action="store_true",
        help="列出 redline.xlsx 中所有 sheet 名后退出",
    )
    parser.add_argument(
        "--xlsx",
        default=_XLSX,
        help=f"输入 xlsx 路径 (默认 {_XLSX})",
    )
    args = parser.parse_args()

    if args.list_sheets:
        names = list_sheet_names(args.xlsx)
        print(f"sheets in {args.xlsx}:")
        for i, nm in enumerate(names, 1):
            print(f"  [{i}] {nm}")
        return

    sheet = _parse_sheet_arg(args.sheet)
    xlsx_path = args.xlsx

    print(f"[excel_to_csv] reading sheet={sheet!r} from {xlsx_path}")
    rows = load_sheet(xlsx_path, sheet)
    if not rows:
        raise RuntimeError(f"xlsx sheet {sheet!r} has no rows")
    header_row = rows[0]
    data_rows = rows[1:]
    if len(data_rows) < 1:
        raise RuntimeError(f"expected >=1 data rows in sheet, got {len(data_rows)}")

    # 按表头列名动态构建列索引表 (避免不同 sheet 列顺序偏移导致错位)
    col_by_name = _build_col_by_name(header_row)

    a_header = _strip_or_none(header_row.get(col_by_name["A_testcase_name"]))
    if a_header != "testcase_name":
        raise RuntimeError(f"unexpected header A: {a_header!r}")

    # 检测整 sheet 的 quant_mode (redline.xlsx 单 mode 约定: 全 mxfp8 或全 gqa_fp8)
    # 取首数据行 BO_quant_mode 列判断; 省略默认 1 (MXFP8)
    first_qm = _str_to_int(data_rows[0].get(col_by_name["BO_quant_mode"])) or 1
    if first_qm == 6:
        profiles = CSV_PROFILES_GQA_FP8
        mode_label = "GQA FP8 (quant_mode=6)"
    else:
        profiles = CSV_PROFILES_MXFP8
        mode_label = "MXFP8 (quant_mode=1)"

    print(f"[excel_to_csv] detected mode: {mode_label}, {len(data_rows)} data rows")

    for fname, api_name, suffix in profiles:
        out_path = os.path.join(_HERE, fname)
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(HEADER)
            for row in data_rows:
                writer.writerow(_row_to_csv(row, api_name, suffix, col_by_name))
        print(f"wrote {out_path}")

    print(f"\n3 csv files written ({mode_label}), each with {len(data_rows)} case rows.")


if __name__ == "__main__":
    main()
