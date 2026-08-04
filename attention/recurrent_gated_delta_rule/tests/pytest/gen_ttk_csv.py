#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS PROGRAM IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""将 test_recurrent_gated_delta_rule_paramset_rdv.py 的全部用例转换为 TTK CSV。

生成 E2E + ACLNN 两份 CSV，覆盖 RDV 全部用例（含 fp32 state、非连续 state）。

用法:
    cd attention/recurrent_gated_delta_rule/tests/pytest
    python3 gen_ttk_csv.py
"""

import csv
import itertools
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from test_recurrent_gated_delta_rule_paramset_rdv import ENABLED_PARAMS_RDV  # noqa: E402

DTYPE_MAP = {
    torch.bfloat16: "bfloat16",
    torch.float32: "float32",
    torch.float16: "float16",
}

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "assets" / "rgdr_assets"


def dtype_str(dt):
    return DTYPE_MAP.get(dt, str(dt))


def adjust_range(datarange):
    left, right = datarange
    if right < 0:
        return [left, right]
    if left > 0:
        return [-right, -left]
    return [left, 0]


def expand(param_dict):
    keys = list(param_dict.keys())
    return [
        dict(zip(keys, combo))
        for combo in itertools.product(*[param_dict[k] for k in keys])
    ]


def fmt(val):
    """格式化 Python 值为 CSV 单元格字符串（仅去空格，保留字符串引号）。"""
    return str(val).replace(" ", "")


def compute_derived(case):
    b = case["batch_size"]
    mtp = case["mtp"]
    nk = case["nk"]
    nv = case["nv"]
    dk = case["dk"]
    dv = case["dv"]
    block_num = b * mtp
    scale = case["scale_value"] if case["scale_value"] is not None else dk**-0.5
    t = b * mtp
    return b, mtp, nk, nv, dk, dv, block_num, scale, t


def case_name(prefix, idx, b, mtp, nk, nv, dk, dv, sdt_str, non_contig):
    name = f"{prefix}_{idx}_b{b}_m{mtp}_nk{nk}_nv{nv}_dk{dk}_dv{dv}"
    if sdt_str != "bfloat16":
        name += f"_s{sdt_str}"
    if non_contig:
        name += "_nc"
    return name


def contiguous_stride(shape):
    """计算连续 stride（row-major）。"""
    strides = []
    s = 1
    for dim in reversed(shape):
        strides.append(s)
        s *= dim
    return tuple(reversed(strides))


def non_contig_fields(tensor_count, state_idx, state_shape, block_num, nv, dv, dk):
    """构造非连续 state 的 storage/stride/offset 字段。

    非 state 张量填连续值（storage=view shape, stride=contiguous, offset=0），
    state 张量填 padded storage + 跨步 stride（模拟 [::2, ::2, :, :] 切片）。
    """
    p0 = block_num * 2
    p1 = nv * 2
    nc_storage = (p0, p1, dv, dk)
    nc_stride = (p1 * dv * dk * 2, dv * dk * 2, dk, 1)

    storage_shapes = []
    view_strides = []
    view_offsets = []
    for i in range(tensor_count):
        if i == state_idx:
            storage_shapes.append(nc_storage)
            view_strides.append(nc_stride)
            view_offsets.append(0)
        else:
            shape = state_shape[i] if i < len(state_shape) else ()
            storage_shapes.append(shape)
            view_strides.append(contiguous_stride(shape))
            view_offsets.append(0)
    return tuple(storage_shapes), tuple(view_strides), tuple(view_offsets)


def build_e2e_rows():
    rows = []
    idx = 0
    for param_dict in ENABLED_PARAMS_RDV:
        for case in expand(param_dict):
            b, mtp, nk, nv, dk, dv, block_num, scale, t = compute_derived(case)
            has_g = case["has_gamma"] in (True, "True")
            has_gk = case["has_gamma_k"] in (True, "True")
            has_nat = case["has_num_accepted_tokens"] in (True, "True")
            nc = case["state_non_contiguous"]
            dt = dtype_str(case["data_type"])
            sdt = dtype_str(case["state_data_type"])

            shapes = [
                (t, nk, dk),
                (t, nk, dk),
                (t, nv, dv),
                (block_num, nv, dv, dk),
                (t, nv),
                (b,),
                (t,),
            ]
            dtypes = [dt, dt, dt, sdt, dt, "int32", "int32"]
            ranges = [
                tuple(case["query_datarange"]),
                tuple(case["key_datarange"]),
                tuple(case["value_datarange"]),
                tuple(case["state_datarange"]),
                tuple(case["beta_datarange"]),
                (1, mtp),
                (0, t - 1),
            ]
            if has_nat:
                shapes.append((b,))
                dtypes.append("int32")
                ranges.append((1, mtp))
            if has_g:
                shapes.append((t, nv))
                dtypes.append("float32")
                ranges.append(tuple(adjust_range(case["gamma_datarange"])))
            if has_gk:
                shapes.append((t, nv, dk))
                dtypes.append("float32")
                ranges.append(tuple(adjust_range(case["gamma_k_datarange"])))

            row = {
                "testcase_name": case_name("rdv", idx, b, mtp, nk, nv, dk, dv, sdt, nc),
                "api_name": "torch_npu.npu_recurrent_gated_delta_rule",
                "tensor_view_shapes": fmt(tuple(shapes)),
                "tensor_dtypes": fmt(tuple(dtypes)),
                "attributes": "{'scale':" + str(scale) + "}",
                "inplace_input_indexes": "(3,)",
                "input_data_ranges": fmt(tuple(ranges)),
                "is_enabled": True,
            }
            if nc:
                ss, vs, vo = non_contig_fields(
                    len(shapes), 3, shapes, block_num, nv, dv, dk
                )
                row["tensor_storage_shapes"] = fmt(ss)
                row["tensor_view_strides"] = fmt(vs)
                row["tensor_view_offsets"] = fmt(vo)

            rows.append(row)
            idx += 1
    return rows


def build_aclnn_rows():
    rows = []
    idx = 0
    for param_dict in ENABLED_PARAMS_RDV:
        for case in expand(param_dict):
            b, mtp, nk, nv, dk, dv, block_num, scale, t = compute_derived(case)
            has_g = case["has_gamma"] in (True, "True")
            has_gk = case["has_gamma_k"] in (True, "True")
            has_nat = case["has_num_accepted_tokens"] in (True, "True")
            nc = case["state_non_contiguous"]
            dt = dtype_str(case["data_type"])
            sdt = dtype_str(case["state_data_type"])

            # aclnn 顺序: query, key, value, beta, stateRef, actualSeqLengths,
            #             ssmStateIndices, g, gk, numAcceptedTokens, out
            shapes = [
                (t, nk, dk),
                (t, nk, dk),
                (t, nv, dv),
                (t, nv),
                (block_num, nv, dv, dk),
                (b,),
                (t,),
            ]
            dtypes = [dt, dt, dt, dt, sdt, "int32", "int32"]
            ranges = [
                tuple(case["query_datarange"]),
                tuple(case["key_datarange"]),
                tuple(case["value_datarange"]),
                tuple(case["beta_datarange"]),
                tuple(case["state_datarange"]),
                (1, mtp),
                (0, t - 1),
            ]

            if has_g:
                shapes.append((t, nv))
                dtypes.append("float32")
                ranges.append(tuple(adjust_range(case["gamma_datarange"])))
            else:
                shapes.append(None)
                dtypes.append(None)
                ranges.append(None)

            if has_gk:
                shapes.append((t, nv, dk))
                dtypes.append("float32")
                ranges.append(tuple(adjust_range(case["gamma_k_datarange"])))
            else:
                shapes.append(None)
                dtypes.append(None)
                ranges.append(None)

            if has_nat:
                shapes.append((b,))
                dtypes.append("int32")
                ranges.append((1, mtp))
            else:
                shapes.append(None)
                dtypes.append(None)
                ranges.append(None)

            shapes.append((t, nv, dv))
            dtypes.append(dt)
            ranges.append(None)

            row = {
                "testcase_name": case_name(
                    "aclnn_rdv", idx, b, mtp, nk, nv, dk, dv, sdt, nc
                ),
                "api_name": "aclnnRecurrentGatedDeltaRule",
                "tensor_view_shapes": fmt(tuple(shapes)),
                "tensor_dtypes": fmt(tuple(dtypes)),
                "attributes": "{'scaleValue':" + str(scale) + "}",
                "output_tensor_indexes": "(10,4)",
                "output_inplace_indexes": "(4,)",
                "input_data_ranges": fmt(tuple(ranges)),
                "is_enabled": True,
            }
            if nc:
                ss, vs, vo = non_contig_fields(
                    len(shapes), 4, shapes, block_num, nv, dv, dk
                )
                row["tensor_storage_shapes"] = fmt(ss)
                row["tensor_view_strides"] = fmt(vs)
                row["tensor_view_offsets"] = fmt(vo)

            rows.append(row)
            idx += 1
    return rows


def write_csv(path, rows, is_aclnn):
    has_nc = any("tensor_storage_shapes" in r for r in rows)
    if is_aclnn:
        cols = [
            "testcase_name",
            "api_name",
            "tensor_view_shapes",
            "tensor_dtypes",
            "attributes",
            "output_tensor_indexes",
            "output_inplace_indexes",
            "input_data_ranges",
        ]
    else:
        cols = [
            "testcase_name",
            "api_name",
            "tensor_view_shapes",
            "tensor_dtypes",
            "attributes",
            "inplace_input_indexes",
            "input_data_ranges",
        ]
    if has_nc:
        cols += ["tensor_storage_shapes", "tensor_view_offsets", "tensor_view_strides"]
    cols.append("is_enabled")

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in cols})


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    e2e_rows = build_e2e_rows()
    e2e_path = OUTPUT_DIR / "recurrent_gated_delta_rule_rdv.csv"
    write_csv(e2e_path, e2e_rows, is_aclnn=False)

    aclnn_rows = build_aclnn_rows()
    aclnn_path = OUTPUT_DIR / "aclnn_recurrent_gated_delta_rule_rdv.csv"
    write_csv(aclnn_path, aclnn_rows, is_aclnn=True)

    total = len(e2e_rows)
    nc_count = sum(1 for r in e2e_rows if "tensor_storage_shapes" in r)
    fp32_count = sum(1 for r in e2e_rows if "_sfloat32" in r["testcase_name"])
    print(f"E2E  CSV: {e2e_path} ({total} cases)")
    print(f"ACLNN CSV: {aclnn_path} ({total} cases)")
    print(f"  非连续 state: {nc_count} cases")
    print(f"  fp32 state:   {fp32_count} cases")


if __name__ == "__main__":
    main()
