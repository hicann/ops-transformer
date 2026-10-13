#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Calculate FlashMlaWithKvcache MFU/MBU directly from a TTK result CSV.

Example (peaks are decimal TFLOPS and TB/s, duration defaults to microseconds):
  python calculate_mfu_mbu.py --result result.csv \
      --peak-tflops 378 --peak-bandwidth-tb-s 1.6 --output metrics.csv

By default, use api_name, tensor_view_shapes, tensor_dtypes and attributes in
each result row. For reduced results, supply --cases testcase/cases.csv to join
external case CSV(s) by testcase_name instead.
Input encoding defaults to UTF-8 (with optional BOM), falling back to GB18030
(compatible with GBK). Use --encoding to override; output is always UTF-8.

For graph results, select --duration-column graph_aclgraph_device_perf_us.
Other duration columns can be selected explicitly, with --duration-unit us/ms/s.
No torch, NPU, TTK installation or external repository is required.

MFU = 2*Nq*(576+512)*visible_pairs / seconds / peak_FLOPS.
MBU = (visible Q/shared-KV reads + complete Out/LSE writes) / seconds / peak_BW.
MLA V aliases K's first 512 lanes: shared KV is counted once, not per Q head.
FP16/BF16 use 2 bytes/element; LSE uses 4. Mask modes 0/3 are supported.
Logical bytes exclude metadata, masks, page tables, workspace, block padding,
repeated loads and cross-batch page sharing. This is not measured HBM traffic.
Use main operator kernel duration excluding metadata for kernel utilization;
event durations also include dispatch gaps. Ratios are not capped at 1.
Output mfu/mbu are ratios with four decimal places (0.5000 means 50%);
values are not capped at 1.
Invalid/missing/ambiguous cases or durations get blank metrics and metrics_error;
the script writes all rows and exits 1 if any row could not be calculated.
"""

import argparse
import ast
import csv
import io
import math
from pathlib import Path
import sys


API_NAMES = {
    "flash_mla_with_kvcache_ttk_ops.flash_mla_with_kvcache_ttk",
    "torch.ops.cann_ops_transformer.flash_mla",
}
METRIC_FIELDS = (
    "metrics_duration_us",
    "theoretical_flops",
    "theoretical_bytes",
    "achieved_tflops",
    "achieved_bandwidth_tb_s",
    "peak_tflops",
    "peak_bandwidth_tb_s",
    "mfu",
    "mbu",
    "metrics_error",
)


def literal(row, field, default=None):
    value = row.get(field, "")
    if not value.strip():
        return default
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"invalid {field}: {value!r}") from exc


def lengths(value, field):
    if not isinstance(value, (tuple, list)) or not value:
        raise ValueError(f"{field}: expected nonempty integer sequence")
    if any(type(x) is not int or x < 0 for x in value):
        raise ValueError(f"{field}: lengths must be nonnegative integers")
    return list(value)


def capacity(shape, layout, side, attrs):
    if layout == "TND":
        field = f"cu_seqlens_{side}_values"
        cu = lengths(attrs.get(field), field)
        if (
            len(cu) < 2
            or cu[0] != 0
            or cu[-1] != shape[0]
            or any(a > b for a, b in zip(cu, cu[1:]))
        ):
            raise ValueError(f"{field}: expected monotone offsets from 0 to {shape[0]}")
        return [b - a for a, b in zip(cu, cu[1:])]
    return [shape[2] if layout == "BNSD" else shape[1]] * shape[0]


def used_lengths(attrs, side, capacities):
    field = f"seqused_{side}_values"
    if attrs.get(field) is None:
        return capacities
    used = lengths(attrs[field], field)
    if len(used) != len(capacities) or any(u > c for u, c in zip(used, capacities)):
        raise ValueError(
            f"{field}: batch count mismatch or used length exceeds capacity"
        )
    return used


def theoretical_cost(row):
    """Decode MLA shapes/sequence values; no torch or operator import required."""
    if row.get("api_name") not in API_NAMES:
        raise ValueError(f"unsupported api_name: {row.get('api_name')!r}")
    shapes = literal(row, "tensor_view_shapes")
    dtypes = literal(row, "tensor_dtypes")
    attrs = literal(row, "attributes", {})
    if (
        not isinstance(attrs, dict)
        or not isinstance(shapes, (list, tuple))
        or len(shapes) < 5
    ):
        raise ValueError("expected attributes dict and Q/KV/sequence tensor shapes")
    if not isinstance(dtypes, (tuple, list)) or len(dtypes) < 2:
        raise ValueError("missing Q/KV tensor_dtypes")
    if any(x not in ("float16", "bfloat16", "fp16", "bf16") for x in dtypes[:2]):
        raise ValueError("only FP16/BF16 Q/KV are supported")
    q, kv = shapes[:2]
    lq, lk = attrs.get("layout_q", "BSND"), attrs.get("layout_kv", "PA_BBND")
    if lq not in ("TND", "BNSD", "BSND") or lk not in ("PA_BBND", "PA_NZ"):
        raise ValueError(f"unsupported layouts: {lq}/{lk}")
    for name, shape, rank in (
        ("Q", q, 3 if lq == "TND" else 4),
        ("KV", kv, 5 if lk == "PA_NZ" else 4),
    ):
        if not isinstance(shape, (list, tuple)) or len(shape) != rank:
            raise ValueError(f"{name}: invalid shape rank")
        if any(type(x) is not int or x < 0 for x in shape):
            raise ValueError(f"{name}: expected concrete nonnegative shape")
    heads = q[1 if lq in ("TND", "BNSD") else 2]
    kv_heads = kv[1 if lk == "PA_NZ" else 2]
    kd = kv[2] * kv[4] if lk == "PA_NZ" else kv[-1]
    if (
        heads <= 0
        or kv_heads != 1
        or q[-1] != 576
        or kd != 576
        or attrs.get("head_dim_v", 512) != 512
    ):
        raise ValueError("MLA requires positive Q heads, one KV head, QK=576 and V=512")
    capacities = capacity(q, lq, "q", attrs)
    q_used = used_lengths(attrs, "q", capacities)
    kv_used = lengths(attrs.get("cache_seqlens_values"), "cache_seqlens_values")
    if len(q_used) != len(kv_used):
        raise ValueError("Q/KV batch counts differ")
    for index, field in ((4, "cu_seqlens_q"), (5, "seqused_q")):
        if (
            len(shapes) > index
            and shapes[index] is not None
            and attrs.get(field + "_values") is None
        ):
            raise ValueError(f"missing {field}_values for active sequence tensor")
    page_size = kv[3 if lk == "PA_NZ" else 1]
    if any(k > kv[0] * page_size for k in kv_used):
        raise ValueError("cache length exceeds the KV page pool")
    mode = attrs.get("mask_mode", 0)
    if mode not in (0, 3):
        raise ValueError("only mask_mode 0/3 are supported")
    if mode == 0 and len(shapes) > 6 and shapes[6] is not None:
        raise ValueError("explicit mode-0 mask needs values to count visible pairs")
    pairs = rows = columns = 0
    for sq, sk in zip(q_used, kv_used):
        if not sq or not sk:
            continue
        visible_q = min(sq, sk) if mode == 3 else sq
        pairs += visible_q * sk - (visible_q * (visible_q - 1) // 2 if mode == 3 else 0)
        rows += visible_q
        columns += sk
    outputs = sum(capacities)
    flops = 2 * heads * (576 + 512) * pairs
    byte_count = 2 * (heads * 576 * rows + 576 * columns + heads * 512 * outputs)
    if attrs.get("return_softmax_lse", False):
        byte_count += 4 * heads * outputs
    return flops, byte_count


def read_csv(path, encoding="auto"):
    data = path.read_bytes()
    candidates = ("utf-8-sig", "gb18030") if encoding == "auto" else (encoding,)
    for candidate in candidates:
        try:
            contents = data.decode(candidate)
            break
        except LookupError as exc:
            raise ValueError(f"unknown encoding: {candidate}") from exc
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(
            f"{path}: cannot decode using {', '.join(candidates)}; "
            "specify the CSV encoding with --encoding"
        )
    if encoding == "auto" and candidate != "utf-8-sig":
        print(f"{path}: using {candidate} input encoding", file=sys.stderr)
    with io.StringIO(contents, newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or "testcase_name" not in reader.fieldnames:
            raise ValueError(f"{path}: missing testcase_name column")
        return reader.fieldnames, list(reader)


def load_cases(paths, encoding="auto"):
    cases = {}
    for path in paths:
        files = sorted(path.glob("*.csv")) if path.is_dir() else [path]
        if not files:
            raise ValueError(f"no case CSVs found in {path}")
        for file in files:
            _, rows = read_csv(file, encoding)
            for row in rows:
                cases.setdefault(row["testcase_name"], []).append((file, row))
    return cases


def positive(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("expected a finite positive number")
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--result", "-r", required=True, type=Path)
    parser.add_argument(
        "--encoding",
        default="auto",
        help="input CSV encoding for result/cases: auto (UTF-8 then GB18030), gbk, etc.",
    )
    parser.add_argument(
        "--cases",
        "-c",
        nargs="+",
        type=Path,
        help="optional external case CSV(s) or directories; default: use result rows",
    )
    parser.add_argument(
        "--output", "-o", type=Path, help="default: <result_stem>_metrics.csv"
    )
    parser.add_argument(
        "--peak-tflops",
        required=True,
        type=positive,
        help="peak compute in decimal TFLOPS",
    )
    parser.add_argument(
        "--peak-bandwidth-tb-s",
        required=True,
        type=positive,
        help="peak bandwidth in decimal TB/s",
    )
    parser.add_argument(
        "--duration-column",
        help="default: eager_device_perf_us, duration_us or duration",
    )
    parser.add_argument("--duration-unit", choices=("us", "ms", "s"), default="us")
    args = parser.parse_args(argv)
    try:
        fields, rows = read_csv(args.result, args.encoding)
        cases = load_cases(args.cases, args.encoding) if args.cases else {}
        column = args.duration_column or next(
            (
                x
                for x in ("eager_device_perf_us", "duration_us", "duration")
                if x in fields
            ),
            None,
        )
        if column is None or column not in fields:
            raise ValueError(
                "duration column not found; select --duration-column explicitly"
            )
        output = args.output or args.result.with_name(args.result.stem + "_metrics.csv")
        case_files = {
            file.resolve() for matches in cases.values() for file, _ in matches
        }
        if output.resolve() == args.result.resolve() or output.resolve() in case_files:
            raise ValueError("output must not overwrite a result or case input")
        errors = 0
        for row in rows:
            row.update(dict.fromkeys(METRIC_FIELDS, ""))
            row.update(
                peak_tflops=args.peak_tflops,
                peak_bandwidth_tb_s=args.peak_bandwidth_tb_s,
            )
            try:
                if row.get("precision_status", "").strip().upper() in (
                    "FAIL",
                    "FAILED",
                ):
                    raise ValueError("result reports a failed case")
                duration = (
                    float(row[column])
                    * {"us": 1, "ms": 1000, "s": 1e6}[args.duration_unit]
                )
                if not math.isfinite(duration) or duration <= 0:
                    raise ValueError("duration must be finite and positive")
                if args.cases:
                    matches = cases.get(row["testcase_name"], [])
                    if not matches:
                        raise ValueError("case not found")
                    if row.get("api_name") and any(
                        row["api_name"] != case.get("api_name") for _, case in matches
                    ):
                        raise ValueError(
                            "result and case api_name differ; narrow --cases"
                        )
                    costs = {theoretical_cost(case) for _, case in matches}
                    if len(costs) != 1:
                        raise ValueError(
                            "ambiguous case name with different costs; narrow --cases"
                        )
                    flops, byte_count = costs.pop()
                else:
                    required = (
                        "api_name",
                        "tensor_view_shapes",
                        "tensor_dtypes",
                        "attributes",
                    )
                    missing = [
                        key for key in required if not (row.get(key) or "").strip()
                    ]
                    if missing:
                        raise ValueError(
                            f"result lacks {', '.join(missing)}; provide --cases"
                        )
                    flops, byte_count = theoretical_cost(row)
                tflops, bandwidth = flops / duration / 1e6, byte_count / duration / 1e6
                row.update(
                    metrics_duration_us=duration,
                    theoretical_flops=flops,
                    theoretical_bytes=byte_count,
                    achieved_tflops=tflops,
                    achieved_bandwidth_tb_s=bandwidth,
                    mfu=f"{tflops / args.peak_tflops:.4f}",
                    mbu=f"{bandwidth / args.peak_bandwidth_tb_s:.4f}",
                )
            except (ValueError, TypeError, KeyError, IndexError) as exc:
                errors += 1
                row["metrics_error"] = str(exc)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=fields + [x for x in METRIC_FIELDS if x not in fields],
            )
            writer.writeheader()
            writer.writerows(rows)
        print(
            f"{output}: {len(rows) - errors}/{len(rows)} calculated, {errors} errors (see metrics_error)"
        )
        return 1 if errors else 0
    except (OSError, ValueError, csv.Error) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    sys.exit(main())
