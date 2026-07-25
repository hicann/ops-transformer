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
  - qfa_mxfp8_excel.csv            api_name=qfa_mxfp8_wrapper.npu_qfa_mxfp8
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

from _xlsx_minireader import col_to_idx, load_sheet1

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

# 3 个 csv 的 api_name 和 testcase_name 后缀
CSV_PROFILES = [
    ("qfa_mxfp8_excel.csv", "qfa_mxfp8_wrapper.npu_qfa_mxfp8", ""),
    ("qfa_mxfp8_excel_metadata.csv", "qfa_metadata_wrapper.run_metadata", "_metadata"),
    ("qfa_mxfp8_excel_main.csv", "qfa_main_wrapper.run_main", "_main"),
]

# excel dtype → csv dtype
DTYPE_MAP = {
    "FLOAT8_E4M3FN": "float8_e4m3fn",
    "BF16": "bfloat16",
    "FLOAT8_E8M0": "float8_e8m0fnu",
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
    return int(float(s))  # Excel 可能给 "8.8387999999999994E-2" 这种浮点字符串


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
#   npu_qfa_mxfp8(q, k, v, dequant_scale_q, dequant_scale_k, dequant_scale_v,
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
def _build_tensor_lists(row):
    shapes = []
    dtypes = []
    data_ranges = []

    def _push(shape, dtype, drange):
        shapes.append(shape)
        dtypes.append(dtype)
        data_ranges.append(drange)

    # q / k / v
    _push(
        _str_to_shape(row.get(COL["Q_q_shape"])),
        _map_dtype(row.get(COL["R_q_dtype"])),
        _str_to_datarange(row.get(COL["S_q_datarange"])),
    )
    _push(
        _str_to_shape(row.get(COL["T_k_shape"])),
        _map_dtype(row.get(COL["U_k_dtype"])),
        _str_to_datarange(row.get(COL["V_k_datarange"])),
    )
    _push(
        _str_to_shape(row.get(COL["W_v_shape"])),
        _map_dtype(row.get(COL["X_v_dtype"])),
        _str_to_datarange(row.get(COL["Y_v_datarange"])),
    )
    # q/k/v descale：真实值由 customize_inputs 生成（amax 计算产物），datarange 不从 Excel 映射。
    # 但 ttk 默认 input 生成路径对所有非 None shape 的 tensor 都要 RandomData(dtype, shape, data_range)，
    # data_range=None 会让 get(None, 0) 崩溃，所以用 (0, 1) 占位（customize_inputs 会覆盖真实值）。
    # dtype: Excel 里是 FLOAT8_E8M0，但 ttk numpy_to_torch_tensor 不支持 float8_e8m0fnu 转换
    # （is_torch_native_dtype 用 hasattr 判断，NPU torch 有此属性但 numpy_to_torch_tensor 白名单没加）。
    # descale 真实 dtype 由 customize_inputs 生成 e8m0 tensor 覆盖，这里用 float8_e4m3fn 占位让 ttk 默认路径不崩。
    _DRANGE_PLACEHOLDER = (0, 1)
    _DESCALE_PLACEHOLDER_DTYPE = "float8_e4m3fn"
    _push(
        _str_to_shape(row.get(COL["Z_qdescale_shape"])),
        _DESCALE_PLACEHOLDER_DTYPE,
        _DRANGE_PLACEHOLDER,
    )
    _push(
        _str_to_shape(row.get(COL["AC_kdescale_shape"])),
        _DESCALE_PLACEHOLDER_DTYPE,
        _DRANGE_PLACEHOLDER,
    )
    _push(
        _str_to_shape(row.get(COL["AF_vdescale_shape"])),
        _DESCALE_PLACEHOLDER_DTYPE,
        _DRANGE_PLACEHOLDER,
    )
    # p_scale：标量，真实值由 customize_inputs 生成，datarange 用占位
    _push(
        _str_to_shape(row.get(COL["AM_p_scale_shape"])),
        _map_dtype(row.get(COL["AN_p_scale_dtype"])),
        _DRANGE_PLACEHOLDER,
    )
    # block_table：wrapper 签名中是必选位置参数（无默认值），
    # 即使非 PA 模式 Excel 为空，也必须给占位 shape 让 ttk 计入 input_count，
    # 否则 match_overload 会因 input_count < required 而返回 PARAM_PLAN_FAILURE。
    # wrapper 内部从 golden_mod._cached_mxfp8_inputs 取真实 block_table（非 PA 时为 None）覆盖占位。
    bt_shape = _str_to_shape(row.get(COL["AI_block_table_shape"]))
    if bt_shape is not None:
        _push(
            bt_shape,
            _map_dtype(row.get(COL["AJ_block_table_dtype"])),
            _str_to_datarange(row.get(COL["AK_block_table_datarange"]))
            or _DRANGE_PLACEHOLDER,
        )
    else:
        _push((0,), "int32", _DRANGE_PLACEHOLDER)

    return shapes, dtypes, data_ranges


def _build_attributes(row):
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

    _set("precision_tolerances", _precision_tolerances(row))
    _set("absolute_precision", ABSOLUTE_PRECISION_DEFAULT)
    _set("p_scale_value", _str_to_float(row.get(COL["AL_p_scale_value"])))
    # cu_seqlens value → list，shape/dtype 不传（由 NPU 推）
    # cu_seqlens_q/kv 是 wrapper 必选参数（无默认值），空则写 None（PA 模式 kv 可能为空）
    _set_force("cu_seqlens_q", _str_to_int_list(row.get(COL["AP_cu_seqlens_q_value"])))
    _set_force(
        "cu_seqlens_kv", _str_to_int_list(row.get(COL["AT_cu_seqlens_kv_value"]))
    )
    # seqused_q/kv：空则写 None（wrapper 无默认值，需要 key 存在，golden 透传 None 给 NPU）
    _set_force("seqused_q", _str_to_int_list(row.get(COL["AX_seqused_q_value"])))
    _set_force("seqused_kv", _str_to_int_list(row.get(COL["BB_seqused_kv_value"])))
    _set("quant_mode", _str_to_int(row.get(COL["BO_quant_mode"])))
    _set("softmax_scale", _str_to_float(row.get(COL["BP_softmax_scale"])))
    _set("mask_mode", _str_to_int(row.get(COL["BQ_mask_mode"])))
    _set("win_left", _str_to_int(row.get(COL["BR_win_left"])))
    _set("win_right", _str_to_int(row.get(COL["BS_win_right"])))
    # max_seqlen_q/kv：空则写 None。 -1 是显式值，保留。
    _set_force("max_seqlen_q", _str_to_int(row.get(COL["BT_max_seqlen_q"])))
    _set_force("max_seqlen_kv", _str_to_int(row.get(COL["BU_max_seqlen_kv"])))
    _set("layout_q", _strip_or_none(row.get(COL["BV_layout_q"])))
    _set("layout_q_descale", _strip_or_none(row.get(COL["BW_layout_q_descale"])))
    _set("layout_kv", _strip_or_none(row.get(COL["BX_layout_kv"])))
    _set("layout_out", _strip_or_none(row.get(COL["BY_layout_out"])))
    _set("enable_lse", _bool_to_int(row.get(COL["BZ_return_softmax_lse"])))
    _set("N_q", _str_to_int(row.get(COL["CB_num_heads_q"])))
    _set("N_kv", _str_to_int(row.get(COL["CC_num_heads_kv"])))
    _set("D", _str_to_int(row.get(COL["CD_head_dim"])))

    # --- wrapper 接口适配参数（不是 op 参数推导，是 wrapper 签名必选参数） ---
    layout_kv = _strip_or_none(row.get(COL["BX_layout_kv"]))
    layout_q_descale = _strip_or_none(row.get(COL["BW_layout_q_descale"]))
    # B: Excel CA 列，空则从 cu_seqlens_q 推（len-1，与 wrapper 约定一致）
    B_val = _str_to_int(row.get(COL["CA_batch_size"]))
    if B_val is None:
        cu_q = _str_to_int_list(row.get(COL["AP_cu_seqlens_q_value"]))
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
    if isinstance(layout_kv, str) and layout_kv.startswith("PA_"):
        k_shape = _str_to_shape(row.get(COL["T_k_shape"]))
        if k_shape is not None:
            if layout_kv == "PA_NZ":
                bs_idx = 3
            else:
                bs_idx = 2
            block_size = k_shape[bs_idx] if len(k_shape) > bs_idx else 0
        else:
            block_size = 0
    else:
        block_size = 0
    _set_force("block_size", block_size)

    return attrs


def _precision_tolerances(row):
    """默认 ((0.005, 0.000025),)；若 attn_out_dtype (L) 含 BF16 → ((0.0078125, 0.0001),)"""
    out_dtype = _strip_or_none(row.get(COL["L_attn_out_dtype"]))
    if out_dtype and "BF16" in out_dtype.upper():
        return ((0.0078125, 0.0001),)
    return ((0.005, 0.000025),)


def _row_to_csv(excel_row, api_name, testcase_suffix):
    """把一行 excel 数据转成一条 csv 行（list of 11 fields）。"""
    raw_name = _strip_or_none(excel_row.get(COL["A_testcase_name"]))
    if not raw_name:
        raise ValueError("testcase_name (A) is empty")
    testcase_name = raw_name + testcase_suffix

    shapes, dtypes, data_ranges = _build_tensor_lists(excel_row)
    attrs = _build_attributes(excel_row)

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
        repr(_precision_tolerances(excel_row)),
        repr(ABSOLUTE_PRECISION_DEFAULT),
    ]


def main():
    rows = load_sheet1(_XLSX)
    if not rows:
        raise RuntimeError("redline.xlsx sheet1 has no rows")
    header_row = rows[0]
    data_rows = rows[1:]
    if len(data_rows) != 6:
        raise RuntimeError(f"expected 6 data rows in mxfp8 sheet, got {len(data_rows)}")

    a_header = _strip_or_none(header_row.get(COL["A_testcase_name"]))
    if a_header != "testcase_name":
        raise RuntimeError(f"unexpected header A: {a_header!r}")

    for fname, api_name, suffix in CSV_PROFILES:
        out_path = os.path.join(_HERE, fname)
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(HEADER)
            for row in data_rows:
                writer.writerow(_row_to_csv(row, api_name, suffix))
        print(f"wrote {out_path}")

    print(f"\n3 csv files written, each with {len(data_rows)} case rows.")


if __name__ == "__main__":
    main()
