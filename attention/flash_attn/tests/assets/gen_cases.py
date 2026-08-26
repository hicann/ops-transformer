#!/usr/bin/python
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

"""Convert pytests TestCases dict → TTK E2E CSV.

Usage:
    # 默认: 读取 functional_stc 用例, 输出到 testcase/flash_attn_stc.csv
    python3 gen_cases.py

    # 指定输出路径 (相对 assets/ 目录解析, 也支持绝对路径)
    python3 gen_cases.py -o /tmp/result.csv
    python3 gen_cases.py -o ../output/flash_attn_stc.csv

    # 指定用例模块 (对应 pytests/test_cases/<module>.py, 默认 functional_stc)
    python3 gen_cases.py --case-file functional_stc -o /tmp/result.csv

    # 生成的 CSV 用 ttk e2e 跑:
    python3 -m ttk e2e -i testcase/flash_attn_stc.csv --plugin .

参数说明:
    --output, -o         输出 CSV 路径, 相对 assets/ 目录或绝对路径
                         (默认: testcase/flash_attn_stc.csv)
    --case-file          用例模块名, 对应 pytests/test_cases/<name>.py
                         (默认: functional_stc)

注意:
    - -o 路径是相对 assets/ 目录拼接的, 不要再加 assets/ 前缀
    - 父目录不存在会自动创建
    - 非法用例 (nc_kv_dims 非法组合) 会直接报错, 中断生成

Reads functional_stc.py TestCases, normalizes params (same logic as
case_loader.normalize_params), computes tensor shapes, and writes a
TTK E2E CSV file compatible with:
    python3 -m ttk e2e -i flash_attn_stc.csv --plugin .

Tensor order in CSV (matches flash_attn_ttk signature keyword-only order):
    [0] q              [3] block_table       [6] seqused_q
    [1] k              [4] cu_seqlens_q      [7] seqused_kv
    [2] v              [5] cu_seqlens_kv      [8] sinks
                                                   [9] attn_mask
                                                  [10] metadata

Unused positions use None. Small integer tensors (cu_seqlens, seqused)
get their values from attributes dict → override_tensors_from_attributes.
block_table is filled by customize_inputs.
"""

import argparse
import csv
import itertools
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PYTESTS = os.path.join(_HERE, "..", "pytests")
if _PYTESTS not in sys.path:
    sys.path.insert(0, _PYTESTS)

API_NAME = "flash_attn_ttk_ops.flash_attn_ttk"

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

# kv 0 轴非连续声明列(仅套件含 nc_kv_dims 用例时输出)
NC_HEADER = ["tensor_storage_shapes", "tensor_view_offsets", "tensor_view_strides"]

DTYPE_MAP = {"fp16": "float16", "bf16": "bfloat16"}
INT_DTYPE = "int32"
ABSOLUTE_PRECISION_DEFAULT = 1e-8


# ── param normalization (mirrors case_loader.normalize_params) ──


def normalize_params(raw):
    c = dict(raw)
    layout_q = c.get("layout_q", "BNSD")
    layout_kv = c.get("layout_kv", layout_q)
    nc_kv_dims = c.get("nc_kv_dims")
    # 非法用例校验: 非 PA 必须连续; PA_BBND 仅 dim0 合法。
    if nc_kv_dims is not None:
        if layout_kv not in ("PA_BBND", "PA_BNBD", "PA_NZ"):
            raise ValueError(
                f"nc_kv_dims={nc_kv_dims} 仅支持 PA 布局, 当前 layout_kv={layout_kv}"
            )
        if layout_kv == "PA_BBND" and nc_kv_dims == (0, 1):
            raise ValueError(
                f"PA_BBND 布局 nc_kv_dims 仅支持 (0,), 当前 nc_kv_dims={nc_kv_dims}"
            )
    c.setdefault("layout_kv", layout_q)
    c.setdefault("layout_out", layout_q)
    c.setdefault("N2", c["N1"])
    c.setdefault("S2", c.get("S1"))
    c.setdefault("DV", c.get("D"))
    c.setdefault("mask_mode", 0)
    c.setdefault("win_left", -1)
    c.setdefault("win_right", -1)
    # mask_mode=0(无掩码)/3(因果掩码)时算子不接受 win_left/win_right 有值:
    # mask_checker 要求其必须为 -1 (op_host/checkers/mask_checker.cpp CheckParaExistence),
    # 历史用例透传的 65536("无限窗口"哨兵值)语义与 -1 等价, 此处统一归 -1
    if int(c.get("mask_mode", 0)) in (0, 3):
        c["win_left"] = -1
        c["win_right"] = -1
    c.setdefault("return_softmax_lse", False)

    for key in ("cu_seqlens_q", "cu_seqlens_kv", "seqused_q", "seqused_kv"):
        if c.get(key) == [None] or c.get(key) is None:
            c.pop(key, None)

    # cu_seqlens 首值必须为 0; 若非 0 则整体平移(段长不变)
    for key in ("cu_seqlens_q", "cu_seqlens_kv"):
        v = c.get(key)
        if isinstance(v, (list, tuple)) and len(v) > 0:
            first = v[0]
            if isinstance(first, (int, float)) and int(first) != 0:
                c[key] = [int(x) - int(first) for x in v]

    if layout_q == "TND":
        cu_q = c.get("cu_seqlens_q")
        if cu_q:
            c.setdefault(
                "seqused_q", [cu_q[i + 1] - cu_q[i] for i in range(len(cu_q) - 1)]
            )
            layout_kv_val = c.get("layout_kv", layout_q)
            if layout_kv_val not in ("PA_BBND", "PA_BNBD", "PA_NZ"):
                c.setdefault("cu_seqlens_kv", list(cu_q))
                cu_kv = c.get("cu_seqlens_kv")
                c.setdefault(
                    "seqused_kv",
                    [cu_kv[i + 1] - cu_kv[i] for i in range(len(cu_kv) - 1)],
                )
        c["B"] = 1

    layout_kv_val = c.get("layout_kv", layout_q)
    if layout_kv_val in ("PA_BBND", "PA_BNBD", "PA_NZ"):
        c.setdefault("block_size", 128)
        if "seqused_kv" not in c:
            c["seqused_kv"] = [c.get("S2", c.get("S1"))] * c.get("B", 1)

        if layout_kv_val == "PA_NZ":
            dtype_str = c.get("Dtype", "fp16")
            c["nz_blk_elem"] = 16 if dtype_str in ("fp16", "bf16") else 32
            c["D_nz_sub"] = c["D"] // c["nz_blk_elem"]
            c["DV_nz_sub"] = c.get("DV", c["D"]) // c["nz_blk_elem"]

        if c.get("block_table") is not None and isinstance(c.get("block_table"), list):
            bt_raw = c["block_table"]
            c["num_blocks"] = int(max(max(row) for row in bt_raw)) + 1
            c["_bt_b"] = len(bt_raw)
            c["_bt_max_blk"] = max(len(row) for row in bt_raw)
            return c

        seqused_kv = c["seqused_kv"]
        block_size = c["block_size"]
        bt_shape = c.get("block_table_shape", [])
        if bt_shape:
            b_val = bt_shape[0]
            max_blk = bt_shape[1]
        else:
            b_val = c.get("B", 1)
            if layout_q == "TND":
                cu_q = c.get("cu_seqlens_q")
                b_val = len(cu_q) - 1 if cu_q else 1
            max_blk = (
                (max(seqused_kv) + block_size - 1) // block_size if seqused_kv else 1
            )
        # num_blocks 取按 seqused_kv 顺序填充实际占用的 page 数(与 inputs.py 填充一致),
        # 避免按 max_blk*B 整体分配导致 K/V 缓冲过度分配(如 (30,4096) 多出约 300 倍)
        used = 0
        for i in range(b_val):
            s = seqused_kv[i] if i < len(seqused_kv) else seqused_kv[-1]
            used += min((int(s) + block_size - 1) // block_size, max_blk)
        c["num_blocks"] = max(used, 1)
        c["_bt_b"] = b_val
        c["_bt_max_blk"] = max_blk
    return c


# ── shape computation ──


def _qkv_shapes(p):
    layout_q = p.get("layout_q", "BNSD")
    layout_kv = p.get("layout_kv", layout_q)
    B = p.get("B", 1)
    N1 = p["N1"]
    N2 = p.get("N2", N1)
    S1 = p.get("S1", 1)
    S2 = p.get("S2", S1)
    D = p["D"]
    DV = p.get("DV", D)

    if layout_q == "BNSD":
        q_shape = (B, N1, S1, D)
    elif layout_q == "BSND":
        q_shape = (B, S1, N1, D)
    elif layout_q == "TND":
        cu_q = p.get("cu_seqlens_q")
        total_s1 = cu_q[-1] if cu_q else S1
        q_shape = (total_s1, N1, D)
    else:
        q_shape = (B, N1, S1, D)

    if layout_kv == "BNSD":
        k_shape = (B, N2, S2, D)
        v_shape = (B, N2, S2, DV)
    elif layout_kv == "BSND":
        k_shape = (B, S2, N2, D)
        v_shape = (B, S2, N2, DV)
    elif layout_kv == "TND":
        cu_kv = p.get("cu_seqlens_kv")
        total_s2 = cu_kv[-1] if cu_kv else S2
        k_shape = (total_s2, N2, D)
        v_shape = (total_s2, N2, DV)
    elif layout_kv == "PA_BBND":
        num_blocks = p.get("num_blocks", 1)
        bs = p.get("block_size", 128)
        k_shape = (num_blocks, bs, N2, D)
        v_shape = (num_blocks, bs, N2, DV)
    elif layout_kv == "PA_BNBD":
        num_blocks = p.get("num_blocks", 1)
        bs = p.get("block_size", 128)
        k_shape = (num_blocks, N2, bs, D)
        v_shape = (num_blocks, N2, bs, DV)
    elif layout_kv == "PA_NZ":
        num_blocks = p.get("num_blocks", 1)
        bs = p.get("block_size", 128)
        nz_sub = p.get("D_nz_sub", D // 16)
        dv_nz_sub = p.get("DV_nz_sub", DV // 16)
        nz_blk = p.get("nz_blk_elem", 16)
        k_shape = (num_blocks, N2, nz_sub, bs, nz_blk)
        v_shape = (num_blocks, N2, dv_nz_sub, bs, nz_blk)
    else:
        k_shape = (B, N2, S2, D)
        v_shape = (B, N2, S2, DV)

    return q_shape, k_shape, v_shape


def _aux_shapes(p):
    """Compute shapes for block_table, cu_seqlens_q/kv, seqused_q/kv."""
    layout_q = p.get("layout_q", "BNSD")
    layout_kv = p.get("layout_kv", layout_q)
    B = p.get("B", 1)

    bt_shape = None
    if layout_kv in ("PA_BBND", "PA_BNBD", "PA_NZ"):
        bt_b = p.get("_bt_b", B)
        bt_max_blk = p.get("_bt_max_blk", 1)
        bt_shape = (bt_b, bt_max_blk)

    cu_q = p.get("cu_seqlens_q")
    cu_kv = p.get("cu_seqlens_kv")
    sq = p.get("seqused_q")
    skv = p.get("seqused_kv")

    cu_q_shape = (len(cu_q),) if cu_q else None
    cu_kv_shape = (len(cu_kv),) if cu_kv else None
    sq_shape = (len(sq),) if sq else None
    skv_shape = (len(skv),) if skv else None

    return bt_shape, cu_q_shape, cu_kv_shape, sq_shape, skv_shape


# ── attributes builder ──


def _build_attrs(p):
    attrs = {}

    def _set(key, val):
        if val is not None:
            attrs[key] = val

    dtype_str = p.get("Dtype", "fp16")
    D = p["D"]
    scale = p.get("scale")
    if scale is None:
        scale = 1.0 / (D**0.5)
    _set("softmax_scale", float(scale))

    _set("mask_mode", int(p.get("mask_mode", 0)))

    wl = p.get("win_left", -1)
    wr = p.get("win_right", -1)
    wl = int(float(wl)) if wl is not None else -1
    wr = int(float(wr)) if wr is not None else -1
    _set("win_left", wl)
    _set("win_right", wr)

    _set("layout_q", p.get("layout_q", "BNSD"))
    _set("layout_kv", p.get("layout_kv", p.get("layout_q", "BNSD")))
    _set("layout_out", p.get("layout_out", p.get("layout_q", "BNSD")))

    rsl = p.get("return_softmax_lse", False)
    _set(
        "return_softmax_lse",
        bool(int(rsl)) if isinstance(rsl, (int, str)) else bool(rsl),
    )

    layout_q = p.get("layout_q", "BNSD")
    if layout_q == "TND":
        cu_q = p.get("cu_seqlens_q")
        if cu_q:
            _set("batch_size", len(cu_q) - 1)
    else:
        _set("batch_size", int(p.get("B", 1)))

    cu_q = p.get("cu_seqlens_q")
    cu_kv = p.get("cu_seqlens_kv")
    sq = p.get("seqused_q")
    skv = p.get("seqused_kv")
    # 小整数张量的值不直接放同名 attr: TTK match_overload 会把
    # "张量参数名出现在 attrs" 重复计入输入数, 超出 flash_attn_ttk 的
    # 11 参数上限导致 PARAM_PLAN_FAILURE。改用 *_values 键,
    # 由 impl/inputs.py customize_inputs 填入对应张量。
    _set("cu_seqlens_q_values", list(cu_q) if cu_q else None)
    _set("cu_seqlens_kv_values", list(cu_kv) if cu_kv else None)
    _set("seqused_q_values", list(sq) if sq else None)
    _set("seqused_kv_values", list(skv) if skv else None)

    _set("max_seqlen_q", _resolve_max_seqlen(p, "q", cu_q, sq))
    _set("max_seqlen_kv", _resolve_max_seqlen(p, "kv", cu_kv, skv))

    layout_kv = p.get("layout_kv", layout_q)
    if layout_kv in ("PA_BBND", "PA_BNBD", "PA_NZ"):
        _set("block_size", int(p.get("block_size", 128)))

    return attrs


def _resolve_max_seqlen(p, suffix, cu_seqlens, seqused):
    """Same logic as pytests data.py _resolve_max_seqlen."""
    explicit = p.get(f"max_seqlen_{suffix}")
    if explicit is not None:
        return int(explicit)
    if seqused:
        return max(int(x) for x in seqused)
    if cu_seqlens and len(cu_seqlens) > 1:
        return max(
            cu_seqlens[i + 1] - cu_seqlens[i] for i in range(len(cu_seqlens) - 1)
        )
    fallback = p.get("S2", p.get("S1", 1)) if suffix == "kv" else p.get("S1", 1)
    return int(fallback)


# ── precision ──


def _precision(p):
    dtype_str = p.get("Dtype", "fp16")
    if dtype_str == "bf16":
        return ((0.0078125, 0.0001),)
    return ((0.005, 0.000025),)


# ── data ranges ──


def _data_ranges(p, num_tensors):
    q_range = p.get("q_range", (-5.0, 5.0))
    k_range = p.get("k_range", (-5.0, 5.0))
    v_range = p.get("v_range", (-5.0, 5.0))
    main = [tuple(q_range), tuple(k_range), tuple(v_range)]
    aux = [(None, None)] * (num_tensors - 3)
    return tuple(main + aux)


# ── CSV row builder ──


def _nc_fields(shape, nc_kv_dims):
    """构造 kv 非连续视图的 storage/stride/offset 三元组。

    与 pytests utils/data.py make_noncontiguous 逐位一致:
      nc_kv_dims == 0      -> pad dim1 (+1 元素), 0 轴 stride 膨胀
      nc_kv_dims == (0, 1) -> pad dim2 (+1 元素), 0/1 轴 stride 联动膨胀
    返回 (storage_shape, view_stride, view_offset); 无 NC 时返回 (None, None, None)。
    """
    if nc_kv_dims == 0:
        pad_dim = 1
    elif nc_kv_dims == (0, 1):
        pad_dim = 2
    else:
        return None, None, None
    storage = list(shape)
    storage[pad_dim] += 1
    strides = [1] * len(shape)
    for d in range(len(shape) - 2, -1, -1):
        strides[d] = strides[d + 1] * storage[d + 1]
    return tuple(storage), tuple(strides), 0


def _contiguous_stride(shape):
    """计算 shape 的连续 strides (numpy/torch 行主序约定)。"""
    if not shape:
        return ()
    strides = [1] * len(shape)
    for d in range(len(shape) - 2, -1, -1):
        strides[d] = strides[d + 1] * shape[d + 1]
    return tuple(strides)


def _fa_metadata_size(batch, kv_heads):
    """metadata 槽位的 int32 占位元素数, 逐字镜像 op 公式。

    Mirrors torch_extension/flash_attn.py _calculate_metadata_size:
        metadata_size = ((36 + 72) * batch * kv_heads + 1) * 16
        再向上 4096 对齐(元素个数, 非字节数)。
    """
    metadata_size = ((36 + 72) * int(batch) * int(kv_heads) + 1) * 16
    return ((metadata_size + 4095) // 4096) * 4096


def _build_row(case_name, p):
    q_shape, k_shape, v_shape = _qkv_shapes(p)
    bt_shape, cu_q_shape, cu_kv_shape, sq_shape, skv_shape = _aux_shapes(p)
    attrs = _build_attrs(p)

    mask_mode = int(p.get("mask_mode", 0))
    attn_mask_shape = (2048, 2048) if mask_mode in (3, 4) else None

    # metadata 槽位(索引10) int32 占位 shape, 大小镜像 op 公式。batch 复用
    # _build_attrs 已写入的 batch_size attr 语义(非 TND=B, TND=len(cu_q)-1),
    # 严禁直接取 p["B"] —— TND 下 normalize_params 会把裸 B 强制为 1。
    # kv_heads 用 N2(非 N1)。
    batch = int(attrs["batch_size"])
    kv_heads = int(p.get("N2", p["N1"]))
    metadata_shape = (_fa_metadata_size(batch, kv_heads),)

    shapes = [
        q_shape,
        k_shape,
        v_shape,
        bt_shape,
        cu_q_shape,
        cu_kv_shape,
        sq_shape,
        skv_shape,
        None,
        attn_mask_shape,
        metadata_shape,
    ]
    dtype_str = DTYPE_MAP.get(p.get("Dtype", "fp16"), "float16")

    dtypes = [dtype_str, dtype_str, dtype_str]
    for i in range(3, len(shapes)):
        if shapes[i] is None:
            dtypes.append(None)
        elif i == 9:
            dtypes.append("int8")
        else:
            dtypes.append(INT_DTYPE)

    data_ranges = list(_data_ranges(p, len(shapes)))
    data_ranges[10] = (0, 0)  # metadata 占位: TTK 按 range 填充 0 即可
    data_ranges = tuple(data_ranges)

    prec = _precision(p)

    # kv 0 轴非连续声明列(仅套件含 nc_kv_dims 用例时输出)。
    # 注意: TTK 对这三列没有"留空=None=默认连续"的回退 —— flat_storage_shape/
    # flat_view_offset 会把显式 None 原样返回 (RandomData 与 torch.as_strided 均会
    # 抛错导致 INPUT_GEN_FAILURE), 因此所有有张量的槽位都必须填显式连续值。
    nc_kv_dims = p.get("nc_kv_dims")
    storage_shapes = [None] * len(shapes)
    view_strides = [None] * len(shapes)
    view_offsets = [None] * len(shapes)
    if nc_kv_dims is not None:
        for i, shape in enumerate(shapes):
            if shape is None:
                continue  # 槽位无张量, TTK 跳过
            if i in (1, 2):
                storage_shapes[i], view_strides[i], view_offsets[i] = _nc_fields(
                    shape, nc_kv_dims
                )
            else:
                storage_shapes[i] = shape
                view_strides[i] = _contiguous_stride(shape)
                view_offsets[i] = 0

    return [
        case_name,
        API_NAME,
        repr(tuple(shapes)),
        repr(tuple(dtypes)),
        "",
        repr(attrs),
        "",
        "",
        repr(data_ranges),
        repr(prec),
        repr(ABSOLUTE_PRECISION_DEFAULT),
        repr(tuple(storage_shapes)),
        repr(tuple(view_offsets)),
        repr(tuple(view_strides)),
    ]


# ── main ──


def main():
    parser = argparse.ArgumentParser(
        description="Convert pytests TestCases → TTK E2E CSV"
    )
    parser.add_argument(
        "--output",
        "-o",
        default="testcase/flash_attn_stc.csv",
        help="Output CSV path relative to assets (default: testcase/flash_attn_stc.csv)",
    )
    parser.add_argument(
        "--case-file",
        default="functional_stc",
        help="Test case module (default: functional_stc)",
    )
    args = parser.parse_args()

    from core.case_loader import load_case_modules

    cases = load_case_modules([f"test_cases.{args.case_file}"])

    has_nc = any("nc_kv_dims" in raw for raw in cases.values())
    header = HEADER + NC_HEADER if has_nc else HEADER

    out_path = os.path.join(_HERE, args.output)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    written = 0
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for case_id, raw_params in sorted(cases.items()):
            short_name = case_id.rsplit("/", 1)[-1]
            p = normalize_params(raw_params)
            row = _build_row(short_name, p)
            writer.writerow(row if has_nc else row[: len(HEADER)])
            written += 1

    print(f"wrote {out_path} ({written} cases)")


if __name__ == "__main__":
    main()
