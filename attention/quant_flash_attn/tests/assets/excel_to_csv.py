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
输出：每个 mode 1 个 csv 到同目录（metadata 在主算子调用前由 spec 的
  npu_preprocess 生成并回填 metadata slot，主算子直接由 ttk 调用）
  - qfa_mxfp8_excel.csv            api_name=torch.ops.cann_ops_transformer.quant_flash_attn

空值契约：excel 空单元格 → csv attributes 省略该 key（不写 key:None）。

列定位（自 excel_to_csv_exception.py 学习）：
  按表头语义名查列索引（_Columns + _norm_name + _HEADER_ALIASES），不再用写死的
  COL 列字母表，对列顺序变化稳健。

tensor 顺序（共 15 个，对齐算子 schema 位置参数顺序）：
  0  q               q_shape / q_dtype / q_datarange
  1  k               k_shape / k_dtype / k_datarange
  2  v               v_shape / v_dtype / v_datarange
  3  q_descale       q_descale_shape / q_descale_dtype / q_descale_datarange
  4  k_descale       k_descale_shape / k_descale_dtype / k_descale_datarange
  5  v_descale       v_descale_shape / v_descale_dtype / v_descale_datarange
  6  block_table     block_table_shape / block_table_dtype / block_table_datarange
  7  p_scale         p_scale_shape / p_scale_dtype / p_scale_datarange（标量）
  8  cu_seqlens_q    cu_seqlens_q_shape / cu_seqlens_q_dtype / cu_seqlens_q_datarange
  9  cu_seqlens_kv   cu_seqlens_kv_shape / cu_seqlens_kv_dtype / cu_seqlens_kv_datarange
  10 seqused_q       seqused_q_shape / seqused_q_dtype / seqused_q_datarange
  11 seqused_kv      seqused_kv_shape / seqused_kv_dtype / seqused_kv_datarange
  12 sinks           learnable_sink_shape / learnable_sink_dtype / learnable_sink_datarange
  13 attn_mask       attn_mask_shape / attn_mask_dtype / attn_mask_datarange
  14 metadata        metadata_shape / metadata_dtype / metadata_datarange

  上述 shape 从 attributes 抽出成为 tensor 入参；无 shape（Excel 空）→ (0,)。
  cu_seqlens_q/kv、seqused_q/kv 的真实 value 以 `*_values` 属性保留在 attributes
  （不与算子 schema 的同名 Tensor 参数撞名，避免 ttk match_overload 的
  scalar_cover 重复计数；npu_preprocess/inputs 用 `*_values` 覆盖随机 tensor），
  因此 value 属性不能删，只删 shape。
  datarange：q/k/v 取 Excel 真实值；descale/p_scale/block_table/cu_seqlens/
  seqused/sinks/attn_mask/metadata 用 (0,1) 占位（这些 tensor 不参与真实数据生成）。

metadata shape 推导（slot 14，不读 Excel metadata_shape 列）：
  镜像 torch_extension/quant_flash_attn.py 的 quant_flash_attn_metadata 输出推导：
    (2, align4096(METADATA_STRIDE + (aic_num+aiv_num)*METADATA_STRIDE*B*N_kv))
  B 的优先级与 npu_preprocess._derive_batch_size 一致（seqused_q → cu_seqlens_q-1
  → batch_size 列 → BSND 的 q_shape[0]；TND 无后两项兜底）；
  N_kv 取 num_heads_kv 列（空则按 layout_kv 从 k_shape 推导）；
  核数镜像 _get_core_nums（NPU 真实核数，无 NPU 默认 36/72）。
  推导结果与 npu_preprocess 生成的 shape 不一致会报 metadata shape mismatch。

空行处理：跳过 testcase_name 为空的整行（redline 末尾有空拖行）。
"""

import argparse
import csv
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from _xlsx_minireader import load_sheet, list_sheet_names

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

API_NAME = "torch.ops.cann_ops_transformer.quant_flash_attn"

# 每个 mode 1 个 csv 的 api_name 和 testcase_name 后缀 (MXFP8, quant_mode=1)
CSV_PROFILES_MXFP8 = [
    ("qfa_mxfp8_excel.csv", API_NAME, ""),
]

# 每个 mode 1 个 csv 的 api_name 和 testcase_name 后缀 (GQA FP8 全量化, quant_mode=6)
# api_name 与 MXFP8 相同 (内部按 quant_mode 分支),
# 仅 CSV 文件名和 testcase 后缀区分, 避免与 mxfp8 testcase 重名。
CSV_PROFILES_GQA_FP8 = [
    ("qfa_gqa_fp8_excel.csv", API_NAME, "_gqa_fp8"),
]

# 每个 mode 1 个 csv 的 api_name 和 testcase_name 后缀 (HIF8 per-tensor, quant_mode=0)
CSV_PROFILES_HIF8 = [
    ("qfa_hif8_excel.csv", API_NAME, "_hif8"),
]

# 向后兼容别名 (现有脚本若 import CSV_PROFILES)
CSV_PROFILES = CSV_PROFILES_MXFP8

# excel dtype → csv dtype
DTYPE_MAP = {
    "FLOAT8_E4M3FN": "float8_e4m3fn",
    "FP8_E4M3": "float8_e4m3fn",
    "FLOAT8_E5M2": "float8_e5m2",
    "FP8_E5M2": "float8_e5m2",
    "FLOAT8_E8M0": "float8_e8m0",
    "FP8_E8M0": "float8_e8m0",
    "FLOAT4_E2M1": "float4_e2m1",
    "HIFLOAT8": "uint8",
    "HFLOAT8": "uint8",
    "BF16": "bfloat16",
    "FP16": "float16",
    "FP32": "float32",
    "FLOAT32": "float32",
    "INT32": "int32",
    "INT8": "int8",
    "BOOL": "bool",
}

ABSOLUTE_PRECISION_DEFAULT = 1e-8

# tensor 顺序（15 个），对齐算子 schema 位置参数顺序 / tensor_view_shapes。
# 每项：(shape 列名, dtype 列名, datarange 列名, dtype 缺省值, 是否读 Excel datarange)。
#   use_real_drange=True  → 取 Excel datarange（q/k/v 参与真实数据生成）
#   use_real_drange=False → 用 (0,1) 占位（descale/p_scale/block_table/cu_seqlens/
#                           seqused/sinks/attn_mask/metadata 不参与真实数据生成）
_TENSOR_SPECS = [
    ("q_shape", "q_dtype", "q_datarange", "float8_e4m3fn", True),
    ("k_shape", "k_dtype", "k_datarange", "float8_e4m3fn", True),
    ("v_shape", "v_dtype", "v_datarange", "float8_e4m3fn", True),
    ("q_descale_shape", "q_descale_dtype", "q_descale_datarange", "float8_e8m0", False),
    ("k_descale_shape", "k_descale_dtype", "k_descale_datarange", "float8_e8m0", False),
    ("v_descale_shape", "v_descale_dtype", "v_descale_datarange", "float8_e8m0", False),
    ("block_table_shape", "block_table_dtype", "block_table_datarange", "int32", False),
    ("p_scale_shape", "p_scale_dtype", "p_scale_datarange", "float32", False),
    (
        "cu_seqlens_q_shape",
        "cu_seqlens_q_dtype",
        "cu_seqlens_q_datarange",
        "int32",
        False,
    ),
    (
        "cu_seqlens_kv_shape",
        "cu_seqlens_kv_dtype",
        "cu_seqlens_kv_datarange",
        "int32",
        False,
    ),
    ("seqused_q_shape", "seqused_q_dtype", "seqused_q_datarange", "int32", False),
    ("seqused_kv_shape", "seqused_kv_dtype", "seqused_kv_datarange", "int32", False),
    (
        "learnable_sink_shape",
        "learnable_sink_dtype",
        "learnable_sink_datarange",
        "float32",
        False,
    ),
    ("attn_mask_shape", "attn_mask_dtype", "attn_mask_datarange", "int8", False),
    ("metadata_shape", "metadata_dtype", "metadata_datarange", "int32", False),
]

_DRANGE_PLACEHOLDER = (0, 1)

# 每核 metadata 字段数, 与 torch_extension/quant_flash_attn.py 的
# METADATA_STRIDE 同名同值
METADATA_STRIDE = 16


# -----------------------------------------------------------------------------------------------------------
# metadata slot shape 推导（镜像 torch_extension/quant_flash_attn.py 的
# quant_flash_attn_metadata 输出推导, 两侧不一致会触发 npu_preprocess 的
# metadata shape mismatch）
# -----------------------------------------------------------------------------------------------------------
_CORE_NUMS = None


def _get_core_nums():
    """镜像 torch_extension/quant_flash_attn.py 的 _get_core_nums：
    有 NPU 时查询真实核数，无 NPU（torch/torch_npu 缺失或无设备）默认 (36, 72)；
    真机上查询失败向上抛，避免静默用默认核数推出偏小的 metadata shape。
    """
    global _CORE_NUMS
    if _CORE_NUMS is None:
        try:
            import torch
        except ImportError:
            torch = None
        npu = getattr(torch, "npu", None) if torch is not None else None
        if npu is None or not npu.is_available():
            _CORE_NUMS = (36, 72)
        else:
            props = npu.get_device_properties()
            _CORE_NUMS = (props.cube_core_num, props.vector_core_num)
    return _CORE_NUMS


def _tensor_numel(shape):
    """shape tuple → 元素数；None/空/含 0 维 → 0（空 tensor 运行时归一成 None）。"""
    if not shape:
        return 0
    numel = 1
    for dim in shape:
        numel *= dim
    return numel


def _metadata_batch_size(row, cols):
    """推导 metadata 用的 batch_size，优先级镜像 npu_preprocess._derive_batch_size
    （最终与 torch_extension._calculate_batch_size 等价），以 Excel 列表达：
      seqused_q(shape/value) → cu_seqlens_q(shape/value)-1 → batch_size 列 →
      BSND 的 q_shape[0] → 0。
    TND 时 npu_preprocess 给 metadata op 传 batch_size=None，无后两项兜底。
    """
    layout_q = _strip_or_none(row.get(cols.get("layout_q")))
    seq_shape = _str_to_shape(row.get(cols.get("seqused_q_shape")))
    seq_values = _str_to_int_list(row.get(cols.get("seqused_q_value")))
    cu_shape = _str_to_shape(row.get(cols.get("cu_seqlens_q_shape")))
    cu_values = _str_to_int_list(row.get(cols.get("cu_seqlens_q_value")))

    seq_numel = _tensor_numel(seq_shape)
    if seq_numel > 0:
        return seq_numel
    if seq_values:
        return len(seq_values)
    if layout_q == "TND":
        cu_numel = _tensor_numel(cu_shape)
        if cu_numel > 0:
            return max(cu_numel - 1, 0)
        if cu_values:
            return max(len(cu_values) - 1, 0)
        return 0
    if cu_values:
        return max(len(cu_values) - 1, 0)
    explicit = _str_to_int(row.get(cols.get("batch_size")))
    if explicit is not None:
        return explicit
    if layout_q in (None, "BSND"):
        q_shape = _str_to_shape(row.get(cols.get("q_shape")))
        if _tensor_numel(q_shape) > 0:
            return q_shape[0]
    return 0


def _metadata_num_heads_kv(row, cols):
    """镜像 npu_preprocess 的 num_heads_kv 派生：num_heads_kv 列优先，
    空 → 按 layout_kv 从 k_shape 推导（PA_BBND: [Bn, Bs, N, D] → k_shape[2]，
    其余 layout → k_shape[1]）。推导不出 → None（调用方报错）。
    """
    n_kv = _str_to_int(row.get(cols.get("num_heads_kv")))
    if n_kv is not None:
        return n_kv
    layout_kv = _strip_or_none(row.get(cols.get("layout_kv")))
    k_shape = _str_to_shape(row.get(cols.get("k_shape")))
    if k_shape is None:
        return None
    if layout_kv == "PA_BBND":
        return k_shape[2] if len(k_shape) > 2 else None
    return k_shape[1] if len(k_shape) > 1 else None


def _calculate_max_schedule_size(batch_size, num_heads_kv, aic_num, aiv_num):
    """镜像 torch_extension/quant_flash_attn.py 的同名函数：
    dim0 按 sectionNum 最坏值 (batch*num_heads_kv) 动态计算并按 4096 对齐；
    batch_size 为 -1/None/0（未知）时按 1 兜底。
    """
    align_size = 4096
    head_size = METADATA_STRIDE
    batch_size = batch_size if batch_size and batch_size > 0 else 1
    fa_size = aic_num * METADATA_STRIDE * batch_size * num_heads_kv
    fd_size = aiv_num * METADATA_STRIDE * batch_size * num_heads_kv

    schedule_size = head_size + fa_size + fd_size
    return ((schedule_size + align_size - 1) // align_size) * align_size


def _metadata_slot_shape(row, cols):
    """metadata 槽位 shape = quant_flash_attn_metadata 的输出
    (2, max_schedule_size)。npu_preprocess copy_ 回填前会校验两侧 shape
    一致（不一致报 metadata shape mismatch），因此必须与算子侧同公式推导。
    """
    num_heads_kv = _metadata_num_heads_kv(row, cols)
    if num_heads_kv is None:
        name = _strip_or_none(row.get(cols.get("testcase_name"))) or "<unnamed>"
        raise ValueError(
            f"[{name}] cannot derive num_heads_kv for metadata shape: "
            "num_heads_kv column empty and k_shape unusable"
        )
    batch_size = _metadata_batch_size(row, cols)
    aic_num, aiv_num = _get_core_nums()
    return (
        2,
        _calculate_max_schedule_size(batch_size, num_heads_kv, aic_num, aiv_num),
    )


# -----------------------------------------------------------------------------------------------------------
# 按表头名查列索引（学自 excel_to_csv_exception.py，替代写死的 COL 列字母表）
# -----------------------------------------------------------------------------------------------------------
def _norm_name(name: str) -> str:
    """列名归一化：转小写并去掉下划线，用于容忍表头大小写/下划线风格差异。"""
    return name.strip().lower().replace("_", "")


# 内部查询名 → xlsx 实际表头名的别名映射（归一化后仍不同的别名）。
# redline 表头本身就是语义名（q_shape / cu_seqlens_q_value ...），
# 以下同义词映射用于容忍像 'Q_N' vs 'num_heads_q' 这类不同命名的列。
_HEADER_ALIASES = {
    "Q_N": "num_heads_q",
    "KV_N": "num_heads_kv",
    "D": "head_dim",
    "B": "batch_size",
    "softmax_lse_shape": "z_softmax_lse_shape",
}
_HEADER_ALIASES_NORM = {
    _norm_name(k): _norm_name(v) for k, v in _HEADER_ALIASES.items()
}


class _Columns:
    """按表头名查列索引。表头是第 1 行的字符串值。

    查找时:
      1) 先按 _HEADER_ALIASES 把内部查询名映射到 xlsx 真实表头名;
      2) 再用 _norm_name 做大小写/下划线归一化匹配。
    对列顺序变化稳健（异常脚本同款做法）。
    """

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
        if key in _HEADER_ALIASES_NORM:
            key = _HEADER_ALIASES_NORM[key]
        return self._by_name.get(key)


# -----------------------------------------------------------------------------------------------------------
# 值转换器：excel 字符串 → python 值
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
    """'4096,20,64' → (4096, 20, 64)；'2'（cu_seqlens 长度）→ (2,)；空 → None。"""
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


def _map_dtype_or(s, default):
    """dtype 缺省回退：读 *_dtype 列，空/未知 → 回退默认。"""
    v = _map_dtype(s)
    return v if v is not None else default


# -----------------------------------------------------------------------------------------------------------
# 每行映射：excel row → csv record
# -----------------------------------------------------------------------------------------------------------
def _build_tensor_lists(row, cols, quant_mode):
    """返回 (shapes, dtypes, data_ranges)，15 个 tensor slot，顺序与 _TENSOR_SPECS 一致。

    无 shape（Excel 空）→ (0,) 占位（可选 tensor → None 省略）。
    datarange：q/k/v 读 Excel；其余用 (0,1) 占位。
    GQA FP8 (quant_mode=6)：descale dtype 回退 float32（非 e8m0），p_scale 空 shape → (1,)。
    metadata slot：shape 按 quant_flash_attn_metadata 输出公式动态推导（见
    _metadata_slot_shape），dtype 固定 int32。
    """
    shapes = []
    dtypes = []
    data_ranges = []

    descale_default = "float32" if quant_mode in (6, 0) else "float8_e8m0"

    for (
        shape_col,
        dtype_col,
        drange_col,
        default_dtype,
        use_real_drange,
    ) in _TENSOR_SPECS:
        # descale dtype 缺省按 quant_mode 分支
        if shape_col in ("q_descale_shape", "k_descale_shape", "v_descale_shape"):
            default_dtype = descale_default

        shape = _str_to_shape(row.get(cols.get(shape_col)))
        dtype = _map_dtype_or(row.get(cols.get(dtype_col)), default_dtype)
        if use_real_drange:
            drange = _str_to_datarange(row.get(cols.get(drange_col)))
        else:
            drange = None

        if shape is None:
            # 可选 tensor 空 → None (省略): 主算子收到 None 而非空 (0,) 张量。
            # 空张量会被 checker 拦截 (如 seqused_q 的 dim0 必须等于 B)。
            shape = (
                None
                if shape_col
                in (
                    "cu_seqlens_q_shape",
                    "cu_seqlens_kv_shape",
                    "seqused_q_shape",
                    "seqused_kv_shape",
                    "learnable_sink_shape",
                    "attn_mask_shape",
                    "block_table_shape",
                )
                else (0,)
            )

        # HIF8 (quant_mode=0): descale 是 per-tensor 标量 (1,), Excel 可能不填 shape
        if (
            shape_col in ("q_descale_shape", "k_descale_shape", "v_descale_shape")
            and shape == (0,)
            and quant_mode == 0
        ):
            shape = (1,)

        # GQA FP8 / HIF8: p_scale 空 shape → 标量 (1,) float32（与原脚本分支一致）
        if shape_col == "p_scale_shape" and shape == (0,) and quant_mode in (6, 0):
            shape = (1,)
            dtype = "float32"

        # metadata slot: shape 按 quant_flash_attn_metadata 的输出推导
        # (2, max_schedule_size)，不读 Excel metadata_shape；npu_preprocess
        # copy_ 回填前校验两侧 shape 一致，推导必须与算子侧同公式
        # （固定 (2,4096) 只在 B*N_kv <= 2 时碰巧成立）。
        if shape_col == "metadata_shape":
            shape = _metadata_slot_shape(row, cols)
            dtype = "int32"

        if drange is None:
            drange = _DRANGE_PLACEHOLDER

        shapes.append(shape)
        dtypes.append(dtype)
        data_ranges.append(drange)

    return shapes, dtypes, data_ranges


def _build_attributes(row, cols):
    """空单元格 → 省略 key（不写 key:None）。

    cu_seqlens_q/kv、seqused_q/kv 的 value 属性保留：ttk 按 tensor_view_shapes
    生成随机 tensor，wrapper 需要 attributes 里的真实 value 覆盖生成结果。
    shape 列（cu_seqlens_q_shape 等）已抽出为 tensor_view_shapes，不再写入 attributes。
    例外：attn_mask_shape / attn_mask_dtype 仍写入 attributes（供 golden
    _build_causal_mask() 按 shape 重建 mask），同时 attn_mask 也保留为 tensor slot 13。

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

    _set("precision_tolerances", _precision_tolerances(row, cols))
    _set("absolute_precision", ABSOLUTE_PRECISION_DEFAULT)
    _set("p_scale_value", _str_to_float(row.get(cols.get("p_scale_value"))))
    # cu_seqlens value → list，shape/dtype 不传（由 tensor_view_shapes 表达）
    # shape 已抽出为 tensor；value 以 `*_values` 保留（不与算子 schema 同名 Tensor
    # 参数撞名），供 inputs.py/npu_preprocess 覆盖随机 tensor。
    _set_force(
        "cu_seqlens_q_values",
        _str_to_int_list(row.get(cols.get("cu_seqlens_q_value"))),
    )
    _set_force(
        "cu_seqlens_kv_values",
        _str_to_int_list(row.get(cols.get("cu_seqlens_kv_value"))),
    )
    # seqused_q/kv：空则写 None（需 key 存在，golden 透传 None 给 NPU）
    _set_force(
        "seqused_q_values", _str_to_int_list(row.get(cols.get("seqused_q_value")))
    )
    _set_force(
        "seqused_kv_values", _str_to_int_list(row.get(cols.get("seqused_kv_value")))
    )
    # attn_mask 的 shape/dtype 也写入 attributes：mask 由 golden _build_causal_mask()
    # 按此 shape 重建（如 (1,2048,2048)）；同时 attn_mask 仍是 tensor slot 13（shape 信息双份）。
    # attn_mask_dtype 缺省 int8（golden causal mask 用 int8）。
    _set("attn_mask_shape", _str_to_shape(row.get(cols.get("attn_mask_shape"))))
    _set("attn_mask_dtype", _map_dtype_or(row.get(cols.get("attn_mask_dtype")), "int8"))
    _set("quant_mode", _str_to_int(row.get(cols.get("quant_mode"))))
    _set("softmax_scale", _str_to_float(row.get(cols.get("softmax_scale"))))
    _set("mask_mode", _str_to_int(row.get(cols.get("mask_mode"))))
    _set("win_left", _str_to_int(row.get(cols.get("win_left"))))
    _set("win_right", _str_to_int(row.get(cols.get("win_right"))))
    # max_seqlen_q/kv：空则写 None。 -1 是显式值，保留。
    _set_force("max_seqlen_q", _str_to_int(row.get(cols.get("max_seqlen_q"))))
    _set_force("max_seqlen_kv", _str_to_int(row.get(cols.get("max_seqlen_kv"))))
    _set("layout_q", _strip_or_none(row.get(cols.get("layout_q"))))
    _set("layout_q_descale", _strip_or_none(row.get(cols.get("layout_q_descale"))))
    _set("layout_kv", _strip_or_none(row.get(cols.get("layout_kv"))))
    _set("layout_out", _strip_or_none(row.get(cols.get("layout_out"))))
    _set("return_softmax_lse", _bool_to_int(row.get(cols.get("return_softmax_lse"))))
    _set("N_q", _str_to_int(row.get(cols.get("num_heads_q"))))
    _set("N_kv", _str_to_int(row.get(cols.get("num_heads_kv"))))
    _set("D", _str_to_int(row.get(cols.get("head_dim"))))
    # head_dim_v: csv 留空/省略 → None, 透传给算子; 算子内部 None 时回退 head_dim
    _set_force(
        "head_dim_v",
        _str_to_int(row.get(cols.get("head_dim_v")))
        if row.get(cols.get("head_dim_v")) is not None
        else None,
    )

    # --- wrapper 接口适配参数（不是 op 参数推导，是 wrapper 签名必选参数） ---
    layout_kv = _strip_or_none(row.get(cols.get("layout_kv")))
    layout_q_descale = _strip_or_none(row.get(cols.get("layout_q_descale")))
    if layout_q_descale is None:
        _qm_raw = _str_to_int(row.get(cols.get("quant_mode")))
        quant_mode = _qm_raw if _qm_raw is not None else 1
        layout_q_descale = "BSND" if quant_mode == 0 else "TND"
    # batch_size: Excel pure passthrough (可为 -1/正整数/None)，透传给 metadata 的 batch_size。
    # wrapper 内部从 cu_seqlens_q 推导正整数 B 供 inputs/golden 生成 BNSD 张量。
    _set_force(
        "batch_size",
        _str_to_int(row.get(cols.get("batch_size")))
        if row.get(cols.get("batch_size")) is not None
        else None,
    )
    # enable_pa: 从 layout_kv 前缀推导（接口适配，不是 op 参数推导）
    _set_force("enable_pa", isinstance(layout_kv, str) and layout_kv.startswith("PA_"))
    # kv_cache_layout: Excel layout_kv → wrapper 参数名
    _set_force("kv_cache_layout", layout_kv)
    # q_scale_layout: Excel layout_q_descale → wrapper 参数名
    _set_force("q_scale_layout", layout_q_descale)
    # block_size: Excel 无此列，PA 模式从 KV shape 第三维提取（算子要求 512 或 1024）
    #   PA_BNBD: [Bn, N, Bs, D]        → shape[2]
    #   PA_BBND: [Bn, Bs, N, D]        → shape[2]
    #   PA_NZ:   [Bn, N, D//32, Bs, 32] → shape[3]
    # 非 PA 模式 block_size=0
    # GQA FP8 (quant_mode=6): K cache 第三维含 K_SCALE_ROWS=4 行 FP32 scale,
    #   真实 block_size = K shape[2] - 4
    # MXFP8 (quant_mode=1): K shape 第三维即真实 block_size
    K_SCALE_ROWS_GQA = 4
    if isinstance(layout_kv, str) and layout_kv.startswith("PA_"):
        k_shape = _str_to_shape(row.get(cols.get("k_shape")))
        if k_shape is not None:
            if layout_kv == "PA_NZ":
                bs_idx = 3
            elif layout_kv == "PA_BBND":
                bs_idx = 1
            else:
                bs_idx = 2
            raw_bs = k_shape[bs_idx] if len(k_shape) > bs_idx else 0
            _qm_bs_raw = _str_to_int(row.get(cols.get("quant_mode")))
            quant_mode_for_bs = _qm_bs_raw if _qm_bs_raw is not None else 1
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


def _precision_tolerances(row, cols):
    """默认 ((0.005, 0.000025, 0.005, 0.005, 10),);
    若 attn_out_dtype 含 BF16 → ((0.0078125, 0.0001, 0.005, 0.005, 10),)

    5-tuple = (rtol, atol, diff_thd, pct_thd, max_diff_hd)。后三位为 check_result的三个阈值, 默认 0.005/0.005/10
    """
    out_dtype = _strip_or_none(row.get(cols.get("attn_out_dtype")))
    if out_dtype and "BF16" in out_dtype.upper():
        return ((0.0078125, 0.0001, 0.005, 0.005, 10),)
    return ((0.005, 0.000025, 0.005, 0.005, 10),)


def _row_to_csv(excel_row, api_name, testcase_suffix, cols, quant_mode):
    """把一行 excel 数据转成一条 csv 行（list of 11 fields）。"""
    raw_name = _strip_or_none(excel_row.get(cols.get("testcase_name")))
    if not raw_name:
        return None
    testcase_name = raw_name + testcase_suffix

    shapes, dtypes, data_ranges = _build_tensor_lists(excel_row, cols, quant_mode)
    attrs = _build_attributes(excel_row, cols)

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
        repr(_precision_tolerances(excel_row, cols)),
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
    parser = argparse.ArgumentParser(
        description="redline.xlsx → 每个 mode 1 个 ttk 标准 CSV (mxfp8 / gqa_fp8 / hif8)"
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

    # 按表头语义名动态构建列索引表（异常脚本同款，对列顺序稳健）
    cols = _Columns(header_row)

    a_header = _strip_or_none(header_row.get(cols.get("testcase_name")))
    if a_header != "testcase_name":
        raise RuntimeError(f"unexpected header A: {a_header!r}")

    # 检测整 sheet 的 quant_mode (redline.xlsx 单 mode 约定: 全 mxfp8 或全 gqa_fp8 或全 hif8)
    # 取首数据行 quant_mode 列判断; 省略默认 1 (MXFP8)
    _first_qm_raw = _str_to_int(data_rows[0].get(cols.get("quant_mode")))
    first_qm = _first_qm_raw if _first_qm_raw is not None else 1
    if first_qm == 6:
        profiles = CSV_PROFILES_GQA_FP8
        mode_label = "GQA FP8 (quant_mode=6)"
    elif first_qm == 0:
        profiles = CSV_PROFILES_HIF8
        mode_label = "HIF8 (quant_mode=0)"
    else:
        profiles = CSV_PROFILES_MXFP8
        mode_label = "MXFP8 (quant_mode=1)"

    print(f"[excel_to_csv] detected mode: {mode_label}, {len(data_rows)} data rows")
    aic_num, aiv_num = _get_core_nums()
    print(
        f"[excel_to_csv] core nums: aic={aic_num}, aiv={aiv_num} "
        "(metadata dim1 = align4096(16 + (aic+aiv)*16*B*N_kv))"
    )

    for fname, api_name, suffix in profiles:
        out_path = os.path.join(_HERE, fname)
        written = 0
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(HEADER)
            for row in data_rows:
                csv_row = _row_to_csv(row, api_name, suffix, cols, first_qm)
                if csv_row is None:
                    continue  # 跳过 testcase_name 为空的整行（redline 末尾空拖行）
                writer.writerow(csv_row)
                written += 1
        print(f"wrote {out_path} ({written} case rows)")

    print(f"\n1 csv file written ({mode_label}) with {written} case rows.")


if __name__ == "__main__":
    main()
