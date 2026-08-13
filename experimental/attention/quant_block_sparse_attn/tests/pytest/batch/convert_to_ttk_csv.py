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

"""Convert quant_block_sparse_attn pytest parameter CSVs to TTK E2E CSVs."""

import argparse
import ast
import csv
import math
import os
import tempfile
from pathlib import Path
from typing import Dict, List, NoReturn, Optional, Sequence, Tuple


TTK_HEADER = (
    "testcase_name",
    "api_name",
    "tensor_view_shapes",
    "tensor_dtypes",
    "tensor_formats",
    "attributes",
    "input_data_ranges",
    "precision_tolerances",
    "absolute_precision",
)

SOURCE_OUTPUT_PAIRS = (
    ("cases.csv", "ttk_cases.csv"),
    ("cases_generalized.csv", "ttk_cases_generalized.csv"),
    ("cases_stc.csv", "ttk_cases_stc.csv"),
)

TENSOR_DTYPES = (
    "float8_e4m3fn",
    "float8_e4m3fn",
    "float8_e4m3fn",
    "float32",
    "float32",
    "float32",
    "int32",
    "int32",
    "float32",
    "int32",
    "int32",
    "int32",
    "int32",
    "int32",
    "uint8",
    "int32",
)
TENSOR_FORMATS = ("ND",) * 16
PRECISION_TOLERANCES = (
    (0.0078125, 0.0001, 0.005, 0.005, 10),
    (0.005, 2.5e-05, 0.005, 0.005, 10),
)
ABSOLUTE_PRECISION = (0.0001, 0.0001)
API_NAME = "bsa_ttk_ops.quant_block_sparse_attn"

POSITIVE_INT_FIELDS = (
    "B",
    "N1",
    "N2",
    "S1",
    "S2",
    "D",
    "sparse_q_block_size",
    "sparse_kv_block_size",
    "block_num",
    "max_block_per_batch",
)
STRING_FIELDS = (
    "case_name",
    "sparse_mode",
    "sparse_pattern",
    "block_table_pattern",
    "layout_q",
    "layout_out",
    "layout_kv",
    "layout_sparse_indices",
    "output_dtype",
)
INT_LIST_FIELDS = (
    "cu_seqlens_q_value",
    "cu_seqlens_kv_value",
    "seqused_q_value",
    "seqused_kv_value",
)
RANGE_FIELDS = ("q_datarange", "k_datarange", "v_datarange")
REQUIRED_SOURCE_FIELDS = (
    STRING_FIELDS
    + POSITIVE_INT_FIELDS
    + INT_LIST_FIELDS
    + RANGE_FIELDS
    + (
        "enable",
        "quant_mode",
        "mask_mode",
        "seed",
        "p_scale_value",
        "softmax_scale",
        "return_softmax_lse",
    )
)


class ConversionError(ValueError):
    """Raised when a source CSV row cannot be converted safely."""


def _fail(source_name: str, line_number: int, field: str, detail: str) -> NoReturn:
    raise ConversionError(f"{source_name}:{line_number}: invalid {field}: {detail}")


def _required(
    record: Dict[str, str], field: str, source_name: str, line_number: int
) -> str:
    if field not in record:
        _fail(source_name, line_number, field, "missing column")
    value = record[field]
    if value is None or not str(value).strip():
        _fail(source_name, line_number, field, "value is empty")
    return str(value).strip()


def _parse_int(
    record: Dict[str, str],
    field: str,
    source_name: str,
    line_number: int,
    *,
    positive: bool = False,
) -> int:
    raw = _required(record, field, source_name, line_number)
    try:
        value = int(raw)
    except ValueError:
        _fail(source_name, line_number, field, f"expected integer, got {raw!r}")
    if positive and value <= 0:
        _fail(
            source_name, line_number, field, f"expected positive integer, got {value}"
        )
    return value


def _parse_float(
    record: Dict[str, str], field: str, source_name: str, line_number: int
) -> float:
    raw = _required(record, field, source_name, line_number)
    try:
        return float(raw)
    except ValueError:
        _fail(source_name, line_number, field, f"expected float, got {raw!r}")


def _parse_optional_float(
    record: Dict[str, str], field: str, source_name: str, line_number: int
) -> Optional[float]:
    if field not in record:
        _fail(source_name, line_number, field, "missing column")
    raw = record[field]
    if raw is None or not str(raw).strip() or str(raw).strip().upper() == "NONE":
        return None
    try:
        return float(str(raw).strip())
    except ValueError:
        _fail(source_name, line_number, field, f"expected float or NONE, got {raw!r}")


def _parse_bool(
    record: Dict[str, str], field: str, source_name: str, line_number: int
) -> bool:
    raw = _required(record, field, source_name, line_number).upper()
    if raw == "TRUE":
        return True
    if raw == "FALSE":
        return False
    _fail(source_name, line_number, field, f"expected TRUE or FALSE, got {raw!r}")


def _parse_int_list(
    record: Dict[str, str], field: str, source_name: str, line_number: int
) -> List[int]:
    if field not in record:
        _fail(source_name, line_number, field, "missing column")
    raw = record[field]
    if raw is None or not str(raw).strip():
        return []
    try:
        value = ast.literal_eval(str(raw).strip())
    except (SyntaxError, ValueError) as error:
        _fail(source_name, line_number, field, f"cannot parse integer list: {error}")
    if not isinstance(value, list):
        _fail(
            source_name,
            line_number,
            field,
            f"expected list, got {type(value).__name__}",
        )
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        _fail(source_name, line_number, field, "list must contain integers only")
    return list(value)


def _parse_range(
    record: Dict[str, str], field: str, source_name: str, line_number: int
) -> Tuple[object, object]:
    raw = _required(record, field, source_name, line_number)
    try:
        value = ast.literal_eval(raw)
    except (SyntaxError, ValueError) as error:
        _fail(source_name, line_number, field, f"cannot parse range: {error}")
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        _fail(source_name, line_number, field, f"expected [min, max], got {value!r}")
    if any(
        isinstance(item, bool) or not isinstance(item, (int, float)) for item in value
    ):
        _fail(source_name, line_number, field, "range bounds must be numeric")
    if value[0] > value[1]:
        _fail(source_name, line_number, field, f"minimum exceeds maximum: {value!r}")
    return value[0], value[1]


def _parse_source_record(
    record: Dict[str, str], source_name: str, line_number: int
) -> Dict[str, object]:
    params: Dict[str, object] = {}
    for field in STRING_FIELDS:
        params[field] = _required(record, field, source_name, line_number)
    for field in POSITIVE_INT_FIELDS:
        params[field] = _parse_int(
            record, field, source_name, line_number, positive=True
        )
    for field in INT_LIST_FIELDS:
        params[field] = _parse_int_list(record, field, source_name, line_number)
    for field in RANGE_FIELDS:
        params[field] = _parse_range(record, field, source_name, line_number)

    params["enable"] = _parse_bool(record, "enable", source_name, line_number)
    params["quant_mode"] = _parse_int(record, "quant_mode", source_name, line_number)
    params["mask_mode"] = _parse_int(record, "mask_mode", source_name, line_number)
    params["seed"] = _parse_int(record, "seed", source_name, line_number)
    params["p_scale_value"] = _parse_optional_float(
        record, "p_scale_value", source_name, line_number
    )
    params["softmax_scale"] = _parse_float(
        record, "softmax_scale", source_name, line_number
    )
    params["return_softmax_lse"] = _parse_bool(
        record, "return_softmax_lse", source_name, line_number
    )

    if params["N1"] % params["N2"] != 0:
        _fail(source_name, line_number, "N1/N2", "N1 must be divisible by N2")
    if params["layout_q"] not in {"TND", "NTD"}:
        _fail(
            source_name,
            line_number,
            "layout_q",
            f"expected TND or NTD, got {params['layout_q']!r}",
        )
    if params["quant_mode"] != 1:
        _fail(
            source_name,
            line_number,
            "quant_mode",
            f"FP8 TTK conversion requires 1, got {params['quant_mode']}",
        )
    if params["sparse_mode"] not in {"random", "dense"}:
        _fail(
            source_name,
            line_number,
            "sparse_mode",
            f"expected random or dense, got {params['sparse_mode']!r}",
        )
    if params["output_dtype"] not in {"bfloat16", "bf16"}:
        _fail(
            source_name,
            line_number,
            "output_dtype",
            f"expected bfloat16, got {params['output_dtype']!r}",
        )

    cu_q = params["cu_seqlens_q_value"]
    seq_kv = params["seqused_kv_value"]
    if len(cu_q) != params["B"] + 1:
        _fail(
            source_name,
            line_number,
            "cu_seqlens_q_value",
            f"expected B+1={params['B'] + 1} values, got {len(cu_q)}",
        )
    if not cu_q or cu_q[0] != 0:
        _fail(source_name, line_number, "cu_seqlens_q_value", "must start with 0")
    if any(right <= left for left, right in zip(cu_q, cu_q[1:])):
        _fail(
            source_name,
            line_number,
            "cu_seqlens_q_value",
            "values must be strictly increasing",
        )
    if len(seq_kv) != params["B"]:
        _fail(
            source_name,
            line_number,
            "seqused_kv_value",
            f"expected B={params['B']} values, got {len(seq_kv)}",
        )
    if any(value <= 0 for value in seq_kv):
        _fail(
            source_name,
            line_number,
            "seqused_kv_value",
            "values must be positive",
        )

    q_lengths = [right - left for left, right in zip(cu_q, cu_q[1:])]
    max_q_length = max(q_lengths)
    if max_q_length > params["S1"]:
        _fail(
            source_name,
            line_number,
            "S1",
            f"maximum actual Q length {max_q_length} exceeds S1={params['S1']}",
        )
    max_kv_length = max(seq_kv)
    if max_kv_length > params["S2"]:
        _fail(
            source_name,
            line_number,
            "S2",
            f"maximum actual KV length {max_kv_length} exceeds S2={params['S2']}",
        )

    required_table_width = math.ceil(max_kv_length / params["sparse_kv_block_size"])
    if params["max_block_per_batch"] < required_table_width:
        _fail(
            source_name,
            line_number,
            "max_block_per_batch",
            f"requires at least {required_table_width}, got {params['max_block_per_batch']}",
        )
    if params["block_num"] < params["B"]:
        _fail(
            source_name,
            line_number,
            "block_num",
            f"requires at least B={params['B']}, got {params['block_num']}",
        )
    if required_table_width > 1 and params["block_num"] < params["B"] + 1:
        _fail(
            source_name,
            line_number,
            "block_num",
            f"multi-block KV sequences require at least B+1={params['B'] + 1}, "
            f"got {params['block_num']}",
        )
    if params["cu_seqlens_kv_value"]:
        _fail(
            source_name,
            line_number,
            "cu_seqlens_kv_value",
            "FP8 TTK requires an empty list",
        )
    if params["seqused_q_value"]:
        _fail(
            source_name,
            line_number,
            "seqused_q_value",
            "FP8 TTK requires an empty list",
        )
    return params


def _tensor_shapes(params: Dict[str, object]) -> Tuple[Tuple[int, ...], ...]:
    total_q = params["cu_seqlens_q_value"][-1]
    q_blocks = math.ceil(params["S1"] / params["sparse_q_block_size"])
    kv_blocks = math.ceil(params["S2"] / params["sparse_kv_block_size"])
    max_kb = min(kv_blocks, params["max_block_per_batch"])
    query_shape = (
        (total_q, params["N1"], params["D"])
        if params["layout_q"] == "TND"
        else (params["N1"], total_q, params["D"])
    )
    q_descale_shape = query_shape[:-1]
    return (
        query_shape,
        (
            params["block_num"],
            params["N2"],
            params["sparse_kv_block_size"],
            params["D"],
        ),
        (
            params["block_num"],
            params["N2"],
            params["sparse_kv_block_size"],
            params["D"],
        ),
        q_descale_shape,
        (
            params["block_num"],
            params["N2"],
            params["sparse_kv_block_size"],
            1,
        ),
        (params["N2"],),
        (params["B"], params["N1"], q_blocks, max_kb),
        (params["B"], params["N1"], q_blocks),
        (0,) if params["p_scale_value"] is None else (1,),
        (len(params["cu_seqlens_q_value"]),),
        (len(params["cu_seqlens_kv_value"]),),
        (len(params["seqused_q_value"]),),
        (len(params["seqused_kv_value"]),),
        (params["B"], params["max_block_per_batch"]),
        (2048, 2048) if params["mask_mode"] == 3 else (0,),
        (0,),
    )


def _attributes(params: Dict[str, object]) -> Dict[str, object]:
    return {
        "p_scale_value": params["p_scale_value"],
        "cu_seqlens_q_value": params["cu_seqlens_q_value"],
        "cu_seqlens_kv_value": params["cu_seqlens_kv_value"],
        "seqused_q_value": params["seqused_q_value"],
        "seqused_kv_value": params["seqused_kv_value"],
        "quant_mode": params["quant_mode"],
        "softmax_scale": params["softmax_scale"],
        "mask_mode": params["mask_mode"],
        "blocksize": params["sparse_kv_block_size"],
        "sparse_block_size_q": params["sparse_q_block_size"],
        "sparse_block_size_kv": params["sparse_kv_block_size"],
        "layout_q": params["layout_q"],
        "layout_kv": params["layout_kv"],
        "layout_out": params["layout_out"],
        "layout_sparse_indices": params["layout_sparse_indices"],
        "return_softmax_lse": params["return_softmax_lse"],
        "batch_size": params["B"],
        "num_heads_q": params["N1"],
        "num_heads_kv": params["N2"],
        "head_dim": params["D"],
        "sparse_mode": params["sparse_mode"],
        "sparse_pattern": params["sparse_pattern"],
        "block_table_pattern": params["block_table_pattern"],
        "seed": params["seed"],
    }


def _input_ranges(params: Dict[str, object]) -> Tuple[object, ...]:
    kv_blocks = math.ceil(params["S2"] / params["sparse_kv_block_size"])
    p_scale_range = (
        ()
        if params["p_scale_value"] is None
        else (params["p_scale_value"], params["p_scale_value"])
    )
    return (
        params["q_datarange"],
        params["k_datarange"],
        params["v_datarange"],
        (-1, 1),
        (-1, 1),
        (-1, 1),
        (0, kv_blocks - 1),
        (0, kv_blocks),
        p_scale_range,
        (0, params["cu_seqlens_q_value"][-1]),
        (),
        (),
        (0, max(params["seqused_kv_value"])),
        (0, params["block_num"] - 1),
        (0, 1) if params["mask_mode"] == 3 else (),
        (0, 1),
    )


def _validate_ttk_row(row: Dict[str, str], source_name: str, line_number: int) -> None:
    parsed = {}
    for field in (
        "tensor_view_shapes",
        "tensor_dtypes",
        "tensor_formats",
        "attributes",
        "input_data_ranges",
    ):
        try:
            parsed[field] = ast.literal_eval(row[field])
        except (SyntaxError, ValueError) as error:
            _fail(source_name, line_number, field, f"invalid output literal: {error}")
    lengths = tuple(
        len(parsed[field])
        for field in (
            "tensor_view_shapes",
            "tensor_dtypes",
            "tensor_formats",
            "input_data_ranges",
        )
    )
    if lengths != (16, 16, 16, 16):
        _fail(
            source_name,
            line_number,
            "TTK tensor contract",
            f"expected four 16-entry fields, got lengths {lengths}",
        )
    sparse_indices_shape = parsed["tensor_view_shapes"][6]
    block_table_shape = parsed["tensor_view_shapes"][13]
    if sparse_indices_shape[-1] > block_table_shape[1]:
        _fail(
            source_name,
            line_number,
            "tensor_view_shapes",
            f"sparseIndices max_Kb {sparse_indices_shape[-1]} exceeds "
            f"blockTable max_block_per_batch {block_table_shape[1]}",
        )
    if not isinstance(parsed["attributes"], dict):
        _fail(source_name, line_number, "attributes", "expected dictionary")


def convert_record(
    record: Dict[str, str], source_name: str, line_number: int
) -> Dict[str, str]:
    params = _parse_source_record(record, source_name, line_number)
    row = {
        "testcase_name": str(params["case_name"]),
        "api_name": API_NAME,
        "tensor_view_shapes": repr(_tensor_shapes(params)),
        "tensor_dtypes": repr(TENSOR_DTYPES),
        "tensor_formats": repr(TENSOR_FORMATS),
        "attributes": repr(_attributes(params)),
        "input_data_ranges": repr(_input_ranges(params)),
        "precision_tolerances": repr(PRECISION_TOLERANCES),
        "absolute_precision": repr(ABSOLUTE_PRECISION),
    }
    _validate_ttk_row(row, source_name, line_number)
    return row


def convert_source(source: Path) -> List[Dict[str, str]]:
    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ConversionError(f"{source.name}: missing CSV header")
        missing_fields = [
            field for field in REQUIRED_SOURCE_FIELDS if field not in reader.fieldnames
        ]
        if missing_fields:
            raise ConversionError(
                f"{source.name}: missing required columns: {missing_fields}"
            )
        rows = []
        first_line_by_name = {}
        for line_number, record in enumerate(reader, start=2):
            row = convert_record(record, source.name, line_number)
            testcase_name = row["testcase_name"]
            if testcase_name in first_line_by_name:
                _fail(
                    source.name,
                    line_number,
                    "case_name",
                    f"duplicate of line {first_line_by_name[testcase_name]}: {testcase_name!r}",
                )
            first_line_by_name[testcase_name] = line_number
            rows.append(row)
        if not rows:
            raise ConversionError(f"{source.name}: contains no testcase rows")
    return rows


def write_ttk_csv(destination: Path, rows: List[Dict[str, str]]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="",
            delete=False,
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
        ) as stream:
            temporary = Path(stream.name)
            writer = csv.DictWriter(stream, fieldnames=TTK_HEADER, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, destination)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="directory for generated TTK CSVs (default: this script's directory)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    source_dir = Path(__file__).resolve().parent
    converted = []
    for source_name, output_name in SOURCE_OUTPUT_PAIRS:
        source = source_dir / source_name
        destination = args.output_dir.resolve() / output_name
        rows = convert_source(source)
        converted.append((source_name, output_name, destination, rows))

    for source_name, output_name, destination, rows in converted:
        write_ttk_csv(destination, rows)
        print(f"{source_name} -> {output_name}: {len(rows)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
