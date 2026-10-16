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
"""把 red.xlsx → ttk e2e 标准 csv（MLA FP8 quant_flash_mla）。

参照 quant_flash_attn/tests/assets/excel_to_csv.py 的架构，适配 MLA 的 tensor 顺序和
算子参数差异。

环境：无 openpyxl/pandas 依赖，xlsx 解析走 _xlsx_minireader（zipfile + minidom）。

输入：red.xlsx（--xlsx 指定，默认同目录 red.xlsx）
输出：1 个 csv 到同目录（metadata 在主算子调用前由 spec 的 npu_preprocess 生成并
  回填 metadata slot，主算子直接由 ttk 调用）
  - qfmla_excel.csv     api_name=torch.ops.cann_ops_transformer.quant_flash_mla

空值契约：excel 空单元格 → csv attributes 省略该 key（不写 key:None）。

tensor 顺序（共 10 个，对齐算子 schema 位置参数顺序）：
  0  q               q_shape / q_dtype / q_datarange
  1  k_cache          k_cache_shape / k_cache_dtype / k_cache_datarange
  2  q_descale        q_descale_shape / q_descale_dtype / q_descale_datarange
  3  k_descale        k_descale_shape / k_descale_dtype / k_descale_datarange
  4  block_table      block_table_shape / block_table_dtype / block_table_datarange
  5  cache_seqlens    cache_seqlens_shape / cache_seqlens_dtype / cache_seqlens_datarange
  6  cu_seqlens_q     cu_seqlens_q_shape / cu_seqlens_q_dtype / cu_seqlens_q_datarange
  7  seqused_q        seqused_q_shape / seqused_q_dtype / seqused_q_datarange
  8  attn_mask        attn_mask_shape / attn_mask_dtype / attn_mask_datarange
  9  metadata         metadata_shape / metadata_dtype / metadata_datarange

  上述 shape 从 attributes 抽出成为 tensor 入参；无 shape（Excel 空）→ None（可选 tensor 省略）。
  cu_seqlens_q、seqused_q 的真实 value 以 `*_values` 属性保留在 attributes
  （不与算子 schema 的同名 Tensor 参数撞名，避免 ttk match_overload 的
  scalar_cover 重复计数；npu_preprocess/inputs 用 `*_values` 覆盖随机 tensor），
  因此 value 属性不能删，只删 shape。
  datarange：q/k_cache 取 Excel 真实值；其余用 (0,1) 占位（不参与真实数据生成）。

TND 约定（与原始 red.xlsx 一致）：
  - T (padded) = B * Q_S  （tensor dim 0）
  - cu_seqlens_q[i] = sum(seqused_q[:i]) for i < B  （实际 token 边界）
  - cu_seqlens_q[B] = T  （padded tensor 大小）
  - seqused_q[i] <= Q_S  （每 batch 有效 token 数）
  对于非 TND 布局转 TND 的行（原 BSND/BNSD），自动生成：
    seqused_q_value = [Q_S] * B（均匀分布）
    cu_seqlens_q_value = [0, Q_S, 2*Q_S, ..., B*Q_S]

metadata shape 透传（与 mxfp8 对齐）：
  - Excel metadata_shape 列填 -1,-1 → CSV 写 (-1,-1)，ttk 识别为动态槽位，
    给主算子传 metadata=None，由 npu_preprocess 调
    quant_flash_mla_metadata 生成真实 tensor 并回填。
  - Excel 填具体 shape（如 2,4096）→ 按静态槽位处理（npu_preprocess resize/copy 回填）。
  - Excel 空 → 自动推导：MLA metadata op 输出 (2, align4096(...))，与 mxfp8 同公式。
    MLA metadata op 参数: cache_seqlens, num_heads_q, num_heads_kv, quant_mode,
      cu_seqlens_q, seqused_q, max_seqlen_q, max_seqlen_kv, head_dim_qk, head_dim_v,
      mask_mode, layout_q
    输出 shape = (2, align4096(16 + (aic+aiv)*16*B*N_kv))

空行处理：跳过 testcase_name 为空的整行。
"""

import argparse
import csv
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from _xlsx_minireader import load_sheet, list_sheet_names

_XLSX = os.path.join(_HERE, "red.xlsx")

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

API_NAME = "torch.ops.cann_ops_transformer.quant_flash_mla"

CSV_PROFILES = [
    ("qfmla_excel.csv", API_NAME, ""),
]

# excel dtype → csv dtype
DTYPE_MAP = {
    "FLOAT8_E4M3FN": "float8_e4m3fn",
    "FP8_E4M3": "float8_e4m3fn",
    "FP8_E4M3FN": "float8_e4m3fn",
    "HIFP8": "float8_e4m3fn",  # HIFP8 → FP8_E4M3FN (转换)
    "HIFLOAT8": "float8_e4m3fn",
    "HFLOAT8": "float8_e4m3fn",
    "FLOAT8_E5M2": "float8_e5m2",
    "FP8_E5M2": "float8_e5m2",
    "BF16": "bfloat16",
    "FP16": "float16",
    "FP32": "float32",
    "FLOAT32": "float32",
    "INT32": "int32",
    "INT8": "int8",
    "BOOL": "bool",
}

ABSOLUTE_PRECISION_DEFAULT = 1e-8

# tensor 顺序（10 个），对齐 MLA 算子 schema 位置参数顺序。
# 每项：(shape 列名, dtype 列名, datarange 列名, dtype 缺省值, 是否读 Excel datarange)。
_TENSOR_SPECS = [
    ("q_shape", "q_dtype", "q_datarange", "float8_e4m3fn", True),
    ("k_cache_shape", "k_cache_dtype", "k_cache_datarange", "float8_e4m3fn", True),
    ("q_descale_shape", "q_descale_dtype", "q_descale_datarange", "float32", False),
    ("k_descale_shape", "k_descale_dtype", "k_descale_datarange", "float32", False),
    ("block_table_shape", "block_table_dtype", "block_table_datarange", "int32", False),
    (
        "cache_seqlens_shape",
        "cache_seqlens_dtype",
        "cache_seqlens_datarange",
        "int32",
        False,
    ),
    (
        "cu_seqlens_q_shape",
        "cu_seqlens_q_dtype",
        "cu_seqlens_q_datarange",
        "int32",
        False,
    ),
    ("seqused_q_shape", "seqused_q_dtype", "seqused_q_datarange", "int32", False),
    ("attn_mask_shape", "attn_mask_dtype", "attn_mask_datarange", "int8", False),
    ("metadata_shape", "metadata_dtype", "metadata_datarange", "int32", False),
]

_DRANGE_PLACEHOLDER = (0, 1)

# MLA MLA: V = K 的 nope 前 512 维, K 有 576 维 (nope 512 + rope 64)
# MLA 不使用 V descale (V 与 K 共享 per-tensor scale), MLA 不使用 p_scale descale (固定 1.0)

METADATA_STRIDE = 16

# PA layout_kv → golden kv_cache_layout 名
_KV_LAYOUT_MAP = {
    "PA_BNBD": "BnNBsD",
    "PA_BBND": "BnBsND",
    "PA_NZ": "NZ",
}


# -----------------------------------------------------------------------------------------------------------
# metadata slot shape 推导（镜像 quant_flash_mla_metadata 的输出推导）
# -----------------------------------------------------------------------------------------------------------
_CORE_NUMS = None


def _get_core_nums():
    """有 NPU 时查询真实核数，无 NPU 默认 (36, 72)。"""
    global _CORE_NUMS
    if _CORE_NUMS is None:
        try:
            import os

            os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
            import torch
            import torch_npu  # noqa: F401

            if torch.npu.is_available():
                props = torch.npu.get_device_properties()
                _CORE_NUMS = (props.cube_core_num, props.vector_core_num)
            else:
                _CORE_NUMS = (36, 72)
        except Exception:
            _CORE_NUMS = (36, 72)
    return _CORE_NUMS


def _tensor_numel(shape):
    if not shape:
        return 0
    numel = 1
    for dim in shape:
        numel *= dim
    return numel


def _metadata_batch_size(row, cols):
    """推导 metadata 用的 batch_size，优先级：
    seqused_q(value) → cu_seqlens_q(value)-1 → batch_size 列 → q_shape[0](BSND) → 0
    """
    seq_values = _str_to_int_list(row.get(cols.get("seqused_q_value")))
    if seq_values:
        return len(seq_values)
    cu_values = _str_to_int_list(row.get(cols.get("cu_seqlens_q_value")))
    if cu_values:
        return max(len(cu_values) - 1, 0)
    explicit = _str_to_int(row.get(cols.get("B")))
    if explicit is not None and explicit > 0:
        return explicit
    return 0


def _metadata_num_heads_kv(row, cols):
    """num_heads_kv 列优先，空 → k_cache_shape[1]（PA_BNBD/PA_NZ）或 [2]（PA_BBND）。"""
    n_kv = _str_to_int(row.get(cols.get("KV_N")))
    if n_kv is not None:
        return n_kv
    layout_kv = _strip_or_none(row.get(cols.get("layout_kv")))
    k_shape = _str_to_shape(row.get(cols.get("k_cache_shape")))
    if k_shape is None:
        return None
    if layout_kv == "PA_BBND":
        return k_shape[2] if len(k_shape) > 2 else None
    return k_shape[1] if len(k_shape) > 1 else None


def _calculate_max_schedule_size(batch_size, num_heads_kv, aic_num, aiv_num):
    align_size = 4096
    head_size = METADATA_STRIDE
    batch_size = batch_size if batch_size and batch_size > 0 else 1
    fa_size = aic_num * METADATA_STRIDE * batch_size * num_heads_kv
    fd_size = aiv_num * METADATA_STRIDE * batch_size * num_heads_kv
    schedule_size = head_size + fa_size + fd_size
    return ((schedule_size + align_size - 1) // align_size) * align_size


def _metadata_slot_shape(row, cols):
    num_heads_kv = _metadata_num_heads_kv(row, cols)
    if num_heads_kv is None:
        name = _strip_or_none(row.get(cols.get("testcase_name"))) or "<unnamed>"
        raise ValueError(f"[{name}] cannot derive num_heads_kv for metadata shape")
    batch_size = _metadata_batch_size(row, cols)
    aic_num, aiv_num = _get_core_nums()
    return (
        2,
        _calculate_max_schedule_size(batch_size, num_heads_kv, aic_num, aiv_num),
    )


# -----------------------------------------------------------------------------------------------------------
# 按表头名查列索引
# -----------------------------------------------------------------------------------------------------------
def _norm_name(name: str) -> str:
    return name.strip().lower().replace("_", "")


_HEADER_ALIASES = {
    # Excel 有 .1 后缀的列存实际值，无后缀列为空
    "cache_seqlens_value": "cache_seqlens_value.1",
    "cu_seqlens_q_value": "cu_seqlens_q_value.1",
}
_HEADER_ALIASES_NORM = {
    _norm_name(k): _norm_name(v) for k, v in _HEADER_ALIASES.items()
}


class _Columns:
    def __init__(self, header_row: dict):
        self._by_name = {}
        for col_idx, text in header_row.items():
            if not text:
                continue
            key = _norm_name(text)
            if key not in self._by_name:
                self._by_name[key] = col_idx

    def get(self, name):
        if name is None:
            return None
        key = _norm_name(name)
        # Try alias first, then fall back to original name
        alias_key = _HEADER_ALIASES_NORM.get(key)
        if alias_key is not None and alias_key in self._by_name:
            return self._by_name[alias_key]
        return self._by_name.get(key)


# -----------------------------------------------------------------------------------------------------------
# 值转换器
# -----------------------------------------------------------------------------------------------------------
def _strip_or_none(s):
    if s is None:
        return None
    s = s.strip()
    return s or None


def _is_int_literal(s: str) -> bool:
    try:
        int(s)
        return True
    except ValueError:
        try:
            int(float(s))
            return True
        except ValueError:
            return False


def _str_to_shape(s):
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    s = s.strip("[]()")
    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    if not parts:
        return None
    dims = []
    for p in parts:
        if not _is_int_literal(p):
            return None
        dims.append(int(float(p)))
    return tuple(dims)


def _str_to_datarange(s):
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


def _map_dtype_or(s, default):
    v = _map_dtype(s)
    return v if v is not None else default


# -----------------------------------------------------------------------------------------------------------
# TND 转换：非 TND 行自动生成 seqused_q/cu_seqlens_q
# -----------------------------------------------------------------------------------------------------------
def _ensure_tnd_fields(row, cols):
    """确保 TND 布局所需的 seqused_q_value 和 cu_seqlens_q_value 存在。

    对于原 BSND/BNSD 行（seqused_q 为空），自动生成：
      seqused_q = [Q_S] * B（均匀分布）
      cu_seqlens_q = [0, Q_S, 2*Q_S, ..., B*Q_S]
    对于原 TND 行（已有 seqused_q），保持不变。

    同时确保 layout_q = layout_out = TND。
    """
    layout_q = _strip_or_none(row.get(cols.get("layout_q")))
    layout_out = _strip_or_none(row.get(cols.get("layout_out")))

    B = _str_to_int(row.get(cols.get("B"))) or 1
    Q_S = _str_to_int(row.get(cols.get("Q_S"))) or 1

    # Generate seqused_q if missing
    sq_values = _str_to_int_list(row.get(cols.get("seqused_q_value")))
    if not sq_values:
        sq_values = [Q_S] * B

    # Generate cu_seqlens_q if missing or last element != B*Q_S (padded convention)
    cu_values = _str_to_int_list(row.get(cols.get("cu_seqlens_q_value")))
    if not cu_values or (cu_values and cu_values[-1] != B * Q_S):
        cu_values = [0]
        for s in sq_values:
            cu_values.append(cu_values[-1] + s)
        # Fix last element to T_padded = B * Q_S (padded convention)
        cu_values[-1] = B * Q_S

    return sq_values, cu_values, layout_q, layout_out


def _calc_tnd_shapes(B, Q_S, Q_N, Q_D, V_D, has_lse, sq_values):
    """计算 TND 布局下的 tensor shapes (使用 compact T = sum(seqused_q))。"""
    T = sum(sq_values)
    q_shape = (T, Q_N, Q_D)
    attn_out_shape = (T, Q_N, V_D)
    q_descale_shape = (T, Q_N)
    lse_shape = (Q_N, T) if has_lse else None
    return q_shape, attn_out_shape, q_descale_shape, lse_shape


def _calc_k_cache_shape(cache_seqlens, block_size, n_kv, d_dim, layout_kv):
    """计算 PA K cache shape。"""
    import math

    blocks_per_batch = [math.ceil(s / block_size) for s in cache_seqlens]
    total_blocks = sum(blocks_per_batch)
    kv_layout = _KV_LAYOUT_MAP.get(layout_kv, layout_kv)
    if kv_layout == "BnNBsD":
        return (total_blocks, n_kv, block_size, d_dim)
    elif kv_layout == "BnBsND":
        return (total_blocks, block_size, n_kv, d_dim)
    elif kv_layout == "NZ":
        d0 = 32  # fp8 element_size = 1, so 32//1 = 32
        d1 = d_dim // d0
        return (total_blocks, n_kv, d1, block_size, d0)
    else:
        raise ValueError(f"Unknown kv_layout: {kv_layout}")


# -----------------------------------------------------------------------------------------------------------
# 每行映射：excel row → csv record
# -----------------------------------------------------------------------------------------------------------
def _build_tensor_lists(row, cols, sq_values, cu_values):
    """返回 (shapes, dtypes, data_ranges)，10 个 tensor slot。"""
    import math

    shapes = []
    dtypes = []
    data_ranges = []

    B = _str_to_int(row.get(cols.get("B"))) or 1
    Q_S = _str_to_int(row.get(cols.get("Q_S"))) or 1
    Q_N = _str_to_int(row.get(cols.get("Q_N"))) or 1
    KV_N = _str_to_int(row.get(cols.get("KV_N"))) or 1
    Q_D = _str_to_int(row.get(cols.get("Q_D"))) or 576
    V_D = _str_to_int(row.get(cols.get("V_D"))) or 512
    block_size = _str_to_int(row.get(cols.get("blocksize"))) or 128
    has_lse = _bool_to_int(row.get(cols.get("return_softmax_lse"))) == 1
    layout_kv = _strip_or_none(row.get(cols.get("layout_kv"))) or "PA_BNBD"

    # Compute TND shapes (compact T = sum(seqused_q))
    T = sum(sq_values)
    q_shape = (T, Q_N, Q_D)
    attn_out_shape = (T, Q_N, V_D)
    q_descale_shape = (T, Q_N)
    lse_shape = (Q_N, T) if has_lse else None

    # K cache shape
    cache_seqlens = _str_to_int_list(row.get(cols.get("cache_seqlens_value"))) or [1024]
    k_cache_shape = _calc_k_cache_shape(cache_seqlens, block_size, KV_N, Q_D, layout_kv)

    # block_table shape
    blocks_per_batch = [math.ceil(s / block_size) for s in cache_seqlens]
    max_blocks = max(blocks_per_batch)
    block_table_shape = (B, max_blocks)

    # attn_mask
    mask_mode = _str_to_int(row.get(cols.get("mask_mode"))) or 0
    has_mask = mask_mode == 3
    attn_mask_shape = (2048, 2048) if has_mask else None

    # Shape overrides for TND layout
    shape_overrides = {
        "q_shape": q_shape,
        "k_cache_shape": k_cache_shape,
        "q_descale_shape": q_descale_shape,
        "block_table_shape": block_table_shape,
        "cache_seqlens_shape": (B,),
        "cu_seqlens_q_shape": (B + 1,),
        "seqused_q_shape": (B,),
        "attn_mask_shape": attn_mask_shape,
    }

    for (
        shape_col,
        dtype_col,
        drange_col,
        default_dtype,
        use_real_drange,
    ) in _TENSOR_SPECS:
        # Use computed shape if available, otherwise read from Excel
        if shape_col in shape_overrides:
            shape = shape_overrides[shape_col]
        else:
            shape = _str_to_shape(row.get(cols.get(shape_col)))

        dtype = _map_dtype_or(row.get(cols.get(dtype_col)), default_dtype)
        if use_real_drange:
            drange = _str_to_datarange(row.get(cols.get(drange_col)))
        else:
            drange = None

        # Optional tensors with None shape → None (省略)
        if shape is None:
            shape = (
                None
                if shape_col
                in (
                    "cu_seqlens_q_shape",
                    "seqused_q_shape",
                    "attn_mask_shape",
                )
                else (0,)
            )

        # metadata slot: 优先取 Excel metadata_shape 列原值（透传）。
        # Excel 填 -1,-1 → CSV 写 (-1,-1)，ttk 识别为动态槽位后给主算子传
        # metadata=None，由 npu_preprocess 调 quant_flash_mla_metadata
        # 生成真实 tensor 并回填主算子。
        # Excel 填具体 shape（如 2,4096）→ 按静态槽位处理（npu_preprocess resize/copy 回填）。
        # Excel 空 / 0,0 → 默认 (2, 4096)（当前 NPU metadata op 实际输出）。
        if shape_col == "metadata_shape":
            excel_meta_shape = _str_to_shape(row.get(cols.get("metadata_shape")))
            if excel_meta_shape is not None and excel_meta_shape != (0, 0):
                shape = excel_meta_shape
            else:
                shape = (2, 4096)
            dtype = "int32"

        # k_descale is per-tensor scalar (1,)
        if shape_col == "k_descale_shape" and (shape is None or shape == (0,)):
            shape = (1,)

        if drange is None:
            drange = _DRANGE_PLACEHOLDER

        shapes.append(shape)
        dtypes.append(dtype)
        data_ranges.append(drange)

    return shapes, dtypes, data_ranges


def _build_attributes(row, cols, sq_values, cu_values):
    """构建 attributes dict。空单元格 → 省略 key。"""
    attrs = {}

    def _set(key, val):
        if val is not None:
            attrs[key] = val

    def _set_force(key, val):
        attrs[key] = val

    B = _str_to_int(row.get(cols.get("B"))) or 1
    Q_N = _str_to_int(row.get(cols.get("Q_N"))) or 1
    KV_N = _str_to_int(row.get(cols.get("KV_N"))) or 1
    Q_D = _str_to_int(row.get(cols.get("Q_D"))) or 576
    V_D = _str_to_int(row.get(cols.get("V_D"))) or 512
    D_rope = 64  # MLA 固定 D_rope=64
    block_size = _str_to_int(row.get(cols.get("blocksize"))) or 128
    quant_mode = _str_to_int(row.get(cols.get("quant_mode"))) or 1
    softmax_scale = (
        _str_to_float(row.get(cols.get("softmax_scale"))) or 0.041666666666666664
    )
    mask_mode = _str_to_int(row.get(cols.get("mask_mode"))) or 0
    max_seqlen_q = _str_to_int(row.get(cols.get("max_seqlen_q")))
    max_seqlen_kv = _str_to_int(row.get(cols.get("max_seqlen_kv")))
    head_dim_v = _str_to_int(row.get(cols.get("head_dim_v"))) or V_D
    layout_q = _strip_or_none(row.get(cols.get("layout_q"))) or "TND"
    layout_kv = _strip_or_none(row.get(cols.get("layout_kv"))) or "PA_BNBD"
    layout_out = _strip_or_none(row.get(cols.get("layout_out"))) or "TND"
    return_lse = _bool_to_int(row.get(cols.get("return_softmax_lse"))) or 0
    graph_path = _str_to_int(row.get(cols.get("graph_path"))) or 0

    # Force TND layout
    layout_q = "TND"
    layout_out = "TND"

    kv_cache_layout = _KV_LAYOUT_MAP.get(layout_kv, layout_kv)

    cache_seqlens = _str_to_int_list(row.get(cols.get("cache_seqlens_value"))) or [1024]

    # max_seqlen auto-derive
    if max_seqlen_q is None or max_seqlen_q <= 0:
        max_seqlen_q = max(sq_values) if sq_values else Q_S_value(B, Q_S)
    if max_seqlen_kv is None or max_seqlen_kv <= 0:
        max_seqlen_kv = max(cache_seqlens) if cache_seqlens else 1024

    # Data ranges
    q_range = _str_to_datarange(row.get(cols.get("q_datarange"))) or (-1.0, 1.0)
    k_range = _str_to_datarange(row.get(cols.get("k_cache_datarange"))) or (-5.0, 5.0)
    v_range = (-5.0, 5.0)

    _set("precision_tolerances", _precision_tolerances(row, cols))
    _set("absolute_precision", ABSOLUTE_PRECISION_DEFAULT)
    _set("B", B)
    _set("N_q", Q_N)
    _set("N_kv", KV_N)
    _set("D", 512)
    _set("D_V", V_D)
    _set("D_rope", D_rope)
    _set("quant_mode", quant_mode)
    _set("softmax_scale", softmax_scale)
    _set_force("seqused_q_values", sq_values)
    _set_force("cache_seqlens_values", cache_seqlens)
    _set("kv_cache_layout", kv_cache_layout)
    _set("block_size", block_size)
    _set("mask_mode", mask_mode)
    _set("enable_lse", return_lse)
    _set("layout_q", layout_q)
    _set("layout_kv", layout_kv)
    _set("layout_out", layout_out)
    _set("return_softmax_lse", return_lse)
    _set("graph_path", graph_path)
    _set("q_data_range", list(q_range))
    _set("k_data_range", list(k_range))
    _set("v_data_range", list(v_range))
    _set("p_scale", 1.0)
    _set("input_layout", layout_q)
    _set("output_layout", layout_out)
    _set_force("max_seqlen_q", max_seqlen_q)
    _set_force("max_seqlen_kv", max_seqlen_kv)
    _set("head_dim_v", head_dim_v)
    _set_force("cu_seqlens_q_values", cu_values)

    return attrs


def Q_S_value(B, Q_S):
    return Q_S


def _precision_tolerances(row, cols):
    """默认 ((0.005, 0.000025, 0.005, 0.005, 10),);
    若 attn_out_dtype 含 BF16 → ((0.0078125, 0.0001, 0.005, 0.005, 10),)
    """
    out_dtype = _strip_or_none(row.get(cols.get("attn_out_dtype")))
    if out_dtype and "BF16" in out_dtype.upper():
        return ((0.0078125, 0.0001, 0.005, 0.005, 10),)
    return ((0.005, 0.000025, 0.005, 0.005, 10),)


def _row_to_csv(excel_row, api_name, testcase_suffix, cols):
    """把一行 excel 数据转成一条 csv 行。"""
    raw_name = _strip_or_none(excel_row.get(cols.get("testcase_name")))
    if not raw_name:
        return None
    testcase_name = raw_name.lower() + testcase_suffix

    # Ensure TND fields
    sq_values, cu_values, _, _ = _ensure_tnd_fields(excel_row, cols)

    shapes, dtypes, data_ranges = _build_tensor_lists(
        excel_row, cols, sq_values, cu_values
    )
    attrs = _build_attributes(excel_row, cols, sq_values, cu_values)

    return [
        testcase_name,
        api_name,
        repr(tuple(shapes)),
        repr(tuple(dtypes)),
        "",  # tensor_formats
        repr(attrs),
        "",  # output_tensor_indexes
        "",  # golden_api
        repr(tuple(data_ranges)),
        repr(_precision_tolerances(excel_row, cols)),
        repr(ABSOLUTE_PRECISION_DEFAULT),
    ]


def _parse_sheet_arg(s):
    if s is None:
        return 1
    s = s.strip()
    if s.isdigit():
        return int(s)
    return s


def main():
    parser = argparse.ArgumentParser(
        description="red.xlsx → ttk 标准 CSV (MLA quant_flash_mla, TND layout)"
    )
    parser.add_argument(
        "--sheet",
        default=None,
        help="xlsx sheet 序号 (1-based int) 或 sheet 名 (str)。默认 1。",
    )
    parser.add_argument(
        "--list-sheets",
        action="store_true",
        help="列出 xlsx 中所有 sheet 名后退出",
    )
    parser.add_argument(
        "--xlsx",
        default=_XLSX,
        help=f"输入 xlsx 路径 (默认 {_XLSX})",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="输出 CSV 路径 (默认同目录 qfmla_excel.csv)",
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

    cols = _Columns(header_row)

    a_header = _strip_or_none(header_row.get(cols.get("testcase_name")))
    if a_header != "testcase_name":
        raise RuntimeError(f"unexpected header A: {a_header!r}")

    aic_num, aiv_num = _get_core_nums()
    print(
        f"[excel_to_csv] core nums: aic={aic_num}, aiv={aiv_num} "
        "(metadata dim1 = align4096(16 + (aic+aiv)*16*B*N_kv))"
    )
    print(f"[excel_to_csv] {len(data_rows)} data rows, all forced to TND layout")

    for fname, api_name, suffix in CSV_PROFILES:
        out_path = args.out or os.path.join(_HERE, fname)
        written = 0
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(HEADER)
            for row in data_rows:
                csv_row = _row_to_csv(row, api_name, suffix, cols)
                if csv_row is None:
                    continue
                writer.writerow(csv_row)
                written += 1
        print(f"wrote {out_path} ({written} case rows)")

    print(f"\n1 csv file written with {written} case rows.")


if __name__ == "__main__":
    main()
