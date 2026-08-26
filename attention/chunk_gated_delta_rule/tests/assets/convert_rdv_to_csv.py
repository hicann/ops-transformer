#!/usr/bin/env python3
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

"""Convert test_chunk_gated_delta_rule_paramset_rdv.py cases to ttk e2e + aclnn CSV files.

Usage:
    python3 convert_rdv_to_csv.py [--output-dir DIR]

Reads ENABLED_PARAMS_RDV from test_chunk_gated_delta_rule_paramset_rdv.py and
generates:
  - chunk_gated_delta_rule_rdv.csv      (e2e mode)
  - aclnn_chunk_gated_delta_rule_rdv.csv (aclnn mode)
"""

import logging

logger = logging.getLogger(__name__)
import argparse
import importlib.util
import sys
from pathlib import Path


def _load_rdv_module(rdv_path):
    spec = importlib.util.spec_from_file_location("rdv_cases", rdv_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["rdv_cases"] = module
    spec.loader.exec_module(module)
    return module


_TORCH_DTYPE_TO_STR = {
    "torch.bfloat16": "bfloat16",
    "torch.float16": "float16",
    "torch.float32": "float32",
    "torch.int32": "int32",
    "torch.int64": "int64",
}


def _dtype_str(dt):
    s = str(dt)
    return _TORCH_DTYPE_TO_STR.get(s, s.split(".")[-1])


_DTYPE_ORDER = [
    "bfloat16",
    "float16",
    "float32",
    "int32",
    "int64",
]


def _dtype_rank(name):
    try:
        return _DTYPE_ORDER.index(name)
    except ValueError:
        return len(_DTYPE_ORDER)


def _tuple_str(items):
    items = list(items)
    parts = []
    for i in items:
        if i is None:
            parts.append("None")
        else:
            parts.append(str(i))
    if len(parts) == 1:
        return f"({parts[0]},)"
    return "(" + ",".join(parts) + ")"


def _shape_str(s):
    if s is None:
        return "None"
    s = list(s)
    if len(s) == 1:
        return f"({s[0]},)"
    return "(" + ",".join(str(d) for d in s) + ")"


def _shape_tuple_str(shapes):
    return _tuple_str([_shape_str(s) for s in shapes])


def _ranges_tuple_str(ranges):
    parts = []
    for r in ranges:
        if r is None:
            parts.append("None")
        else:
            lo, hi = r
            parts.append(f"({lo},{hi})")
    return _tuple_str(parts)


def _contig_stride(shape):
    """Row-major contiguous strides for a shape (same convention as numpy/torch)."""
    if shape is None:
        return None
    dims = list(shape)
    if not dims:
        return ()
    strides = [1] * len(dims)
    for i in range(len(dims) - 2, -1, -1):
        strides[i] = strides[i + 1] * dims[i + 1]
    return tuple(strides)


def _build_e2e_row(name, B, seqlen, Nk, Nv, Dk, Dv, has_g, scale, dt, sdt, is_contig):
    T = B * seqlen if isinstance(seqlen, int) else sum(seqlen)

    qkv_dt = _dtype_str(dt)
    state_dt = _dtype_str(sdt)

    shapes = [
        (T, Nk, Dk),
        (T, Nk, Dk),
        (T, Nv, Dv),
        (T, Nv),
        (B, Nv, Dv, Dk),
        (B,),
    ]
    dtypes = [qkv_dt, qkv_dt, qkv_dt, qkv_dt, state_dt, "int32"]

    if has_g:
        shapes.append((T, Nv))
        dtypes.append("float32")

    tensor_view_shapes = _shape_tuple_str(shapes)
    tensor_dtypes = "('" + "','".join(dtypes) + "')"

    ranges = [(-1, 1), (-1, 1), (-1, 1), (0, 1), (-1, 1), (1, T)]
    if has_g:
        ranges.append((-1, 0))
    input_data_ranges = _ranges_tuple_str(ranges)

    attrs = f"{{'scale':{scale}}}"
    if is_contig:
        line = (
            f'{name},torch_npu.npu_chunk_gated_delta_rule,"{tensor_view_shapes}",'
            f'"{tensor_dtypes}",{attrs},"{input_data_ranges}",,,,True'
        )
    else:
        state_storage = (B, Nv, Dv + 1, Dk)
        state_stride = (Nv * (Dv + 1) * Dk, (Dv + 1) * Dk, Dk, 1)
        storage_shapes = [s for s in shapes]
        storage_shapes[4] = state_storage
        strides = [_contig_stride(s) for s in shapes]
        strides[4] = state_stride
        offsets = ["0"] * len(shapes)
        ts = _tuple_str(storage_shapes)
        tvs = _tuple_str(strides)
        tvo = _tuple_str(offsets)
        line = (
            f'{name},torch_npu.npu_chunk_gated_delta_rule,"{tensor_view_shapes}",'
            f'"{tensor_dtypes}",{attrs},"{input_data_ranges}",'
            f'"{ts}","{tvs}","{tvo}",True'
        )

    return line


def _build_aclnn_row(name, B, seqlen, Nk, Nv, Dk, Dv, has_g, scale, dt, sdt, is_contig):
    T = B * seqlen if isinstance(seqlen, int) else sum(seqlen)

    qkv_dt = _dtype_str(dt)
    state_dt = _dtype_str(sdt)
    out_dt = qkv_dt

    shapes = [
        (T, Nk, Dk),
        (T, Nk, Dk),
        (T, Nv, Dv),
        (T, Nv),
        (B, Nv, Dv, Dk),
        (B,),
    ]
    dtypes = [qkv_dt, qkv_dt, qkv_dt, qkv_dt, state_dt, "int32"]

    if has_g:
        shapes.append((T, Nv))
        dtypes.append("float32")
    else:
        shapes.append(None)
        dtypes.append(None)

    shapes.append((T, Nv, Dv))
    dtypes.append(out_dt)
    shapes.append((B, Nv, Dv, Dk))
    dtypes.append(state_dt)

    tensor_view_shapes = _shape_tuple_str(shapes)
    dtype_parts = [f"'{d}'" if d is not None else "None" for d in dtypes]
    tensor_dtypes = "(" + ",".join(dtype_parts) + ")"

    out_idx = len(shapes) - 2
    output_tensor_indexes = f"({out_idx},{out_idx + 1})"

    ranges = [(-1, 1), (-1, 1), (-1, 1), (0, 1), (-1, 1), (1, T)]
    if has_g:
        ranges.append((-1, 0))
    else:
        ranges.append(None)
    ranges.append(None)
    ranges.append(None)
    input_data_ranges = _ranges_tuple_str(ranges)

    if is_contig:
        line = (
            f'{name},aclnnChunkGatedDeltaRule,"{tensor_view_shapes}","{tensor_dtypes}",'
            f'{{\'scaleValue\':{scale}}},"{output_tensor_indexes}","{input_data_ranges}",,,,True'
        )
    else:
        state_storage = (B, Nv, Dv + 1, Dk)
        state_stride = (Nv * (Dv + 1) * Dk, (Dv + 1) * Dk, Dk, 1)
        storage_shapes = [s for s in shapes]
        storage_shapes[4] = state_storage
        strides = [_contig_stride(s) for s in shapes]
        strides[4] = state_stride
        offsets = ["0"] * len(shapes)
        ts = _tuple_str(storage_shapes)
        tvs = _tuple_str(strides)
        tvo = _tuple_str(offsets)
        line = (
            f'{name},aclnnChunkGatedDeltaRule,"{tensor_view_shapes}","{tensor_dtypes}",'
            f'{{\'scaleValue\':{scale}}},"{output_tensor_indexes}","{input_data_ranges}",'
            f'"{ts}","{tvs}","{tvo}",True'
        )

    return line


def main():
    parser = argparse.ArgumentParser(
        description="Convert rdv paramset cases to ttk CSV"
    )
    parser.add_argument(
        "--rdv-file",
        default=str(
            Path(__file__).resolve().parents[1]
            / "pytest"
            / "test_chunk_gated_delta_rule_paramset_rdv.py"
        ),
        help="Path to test_chunk_gated_delta_rule_paramset_rdv.py",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent),
        help="Output directory for CSV files",
    )
    args = parser.parse_args()

    rdv_path = Path(args.rdv_file).resolve()
    if not rdv_path.exists():
        logger.error(f"ERROR: rdv file not found: {rdv_path}")
        sys.exit(1)

    mod = _load_rdv_module(rdv_path)
    cases = mod.ENABLED_PARAMS_RDV

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    e2e_header = "testcase_name,api_name,tensor_view_shapes,tensor_dtypes,attributes,input_data_ranges,tensor_storage_shapes,tensor_view_strides,tensor_view_offsets,is_enabled"
    aclnn_header = "testcase_name,api_name,tensor_view_shapes,tensor_dtypes,attributes,output_tensor_indexes,input_data_ranges,tensor_storage_shapes,tensor_view_strides,tensor_view_offsets,is_enabled"

    e2e_lines = [e2e_header]
    aclnn_lines = [aclnn_header]

    for case in cases:
        name = case["_name"][0]
        B = case["B"][0]
        seqlen = case["seqlen"][0]
        Nk = case["nk"][0]
        Nv = case["nv"][0]
        Dk = case["dk"][0]
        Dv = case["dv"][0]
        has_g = case["has_g"][0]
        dt = case["data_type"][0]
        sdt = case["state_data_type"][0]
        is_contig = case["is_contiguous"][0]

        scale_val = 1.0 / (Dk**0.5)

        e2e_lines.append(
            _build_e2e_row(
                name, B, seqlen, Nk, Nv, Dk, Dv, has_g, scale_val, dt, sdt, is_contig
            )
        )
        aclnn_lines.append(
            _build_aclnn_row(
                name, B, seqlen, Nk, Nv, Dk, Dv, has_g, scale_val, dt, sdt, is_contig
            )
        )

    e2e_csv = output_dir / "chunk_gated_delta_rule_rdv.csv"
    aclnn_csv = output_dir / "aclnn_chunk_gated_delta_rule_rdv.csv"

    e2e_csv.write_text("\n".join(e2e_lines) + "\n", encoding="utf-8")
    aclnn_csv.write_text("\n".join(aclnn_lines) + "\n", encoding="utf-8")

    logger.info(f"Generated {len(cases)} cases:")
    logger.info(f"  e2e:   {e2e_csv}")
    logger.info(f"  aclnn: {aclnn_csv}")


if __name__ == "__main__":
    main()
