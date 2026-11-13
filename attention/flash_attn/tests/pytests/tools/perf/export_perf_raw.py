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
"""把一次性能测试的原始 op_summary 行导出成逐条宽表。

每个 case 的每次运行占一行，保留 op_summary 原始列，不做 pct 换算。
"""

import argparse
import csv
import sys
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TESTS_DIR))

from core.case_loader import load_case_modules, normalize_params  # noqa: E402


def _read_rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def _find_task_time(perf_dir, summary):
    candidates = list(Path(summary).parent.glob("task_time_*.csv"))
    candidates += list(
        (Path(perf_dir) / "msprof_output").glob(
            "PROF_*/mindstudio_profiler_output/task_time_*.csv"
        )
    )
    files = sorted(set(candidates), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def _find_summary(perf_dir):
    top = Path(perf_dir) / "op_summary.csv"
    if top.exists() and top.stat().st_size > 0:
        return top
    files = sorted(
        (Path(perf_dir) / "msprof_output").glob(
            "PROF_*/mindstudio_profiler_output/op_summary_*.csv"
        ),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not files:
        raise FileNotFoundError(f"no op_summary.csv under {perf_dir}")
    return files[0]


def _max_seqlen(params, name, fallback):
    explicit = params.get(f"max_seqlen_{name}")
    if explicit is not None:
        return int(explicit)
    seqused = params.get(f"seqused_{name}")
    if seqused:
        return int(max(seqused))
    cu = params.get(f"cu_seqlens_{name}")
    if cu:
        return int(max(cu[i + 1] - cu[i] for i in range(len(cu) - 1)))
    return int(fallback)


def _seq_value(value):
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return ";".join(str(x) for x in value)
    return str(value)


def _case_meta(case_id, params):
    n1 = params["N1"]
    n2 = params.get("N2", n1)
    s1 = params["S1"]
    s2 = params.get("S2", s1)
    layout_q = params.get("layout_q", params.get("input_layout", "BNSD"))
    layout_kv = params.get("layout_kv", layout_q)
    layout_out = params.get("layout_out", layout_q)
    name = case_id.split("/")[-1]
    if "_prefill_" in name:
        phase = "prefill"
    elif "_decode_" in name:
        phase = "decode"
    else:
        phase = "prefill" if s1 > 1 else "decode"
    return {
        "B": params.get("B", 1),
        "N2": n2,
        "G": n1 // n2,
        "D": params["D"],
        "DV": params.get("DV", params["D"]),
        "S1": s1,
        "S2": s2,
        "mask": params.get("mask_mode", 0),
        "dtype": params.get("Dtype", "fp16"),
        "layout_q": layout_q,
        "layout_kv": layout_kv,
        "layout_out": layout_out,
        "max_seqlen_q": _max_seqlen(params, "q", s1),
        "max_seqlen_kv": _max_seqlen(params, "kv", s2),
        "cu_seqlens_q": _seq_value(params.get("cu_seqlens_q")),
        "cu_seqlens_kv": _seq_value(params.get("cu_seqlens_kv")),
        "seqused_q": _seq_value(params.get("seqused_q")),
        "seqused_kv": _seq_value(params.get("seqused_kv")),
        "case": name,
        "case_id": case_id,
        "phase": phase,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--perf-dir", required=True)
    parser.add_argument(
        "--case-files",
        required=True,
        help="同 test_flash_attn.py，如 perf.base_generated",
    )
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--cold-thr", type=int, default=16)
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--case-names",
        default=None,
        help="只导出这些 case（逗号分隔的完整 case_id），用于分段跑时按段导出",
    )
    args = parser.parse_args()

    modules = [
        x.strip() if x.strip().startswith("test_cases.") else "test_cases." + x.strip()
        for x in args.case_files.split(",")
    ]
    all_cases = load_case_modules(modules)
    names = sorted(all_cases)
    hot = [
        n for n in names if normalize_params(all_cases[n]).get("S1", 1) > args.cold_thr
    ]
    cold = [
        n for n in names if normalize_params(all_cases[n]).get("S1", 1) <= args.cold_thr
    ]
    if args.case_names:
        want = {x.strip() for x in args.case_names.split(",") if x.strip()}
        names = [n for n in names if n in want]
        hot = [n for n in hot if n in want]
        cold = [n for n in cold if n in want]
    if args.case_names:
        want = {x.strip() for x in args.case_names.split(",") if x.strip()}
        names = [n for n in names if n in want]
        hot = [n for n in hot if n in want]
        cold = [n for n in cold if n in want]

    summary = _find_summary(args.perf_dir)
    task_time = _find_task_time(args.perf_dir, summary)
    kernel_names = {}
    if task_time:
        for r in _read_rows(task_time):
            kernel_names[(r.get("Device_id"), r.get("stream_id"), r.get("task_id"))] = (
                r.get("kernel_name", "")
            )
    raw_rows = [r for r in _read_rows(summary) if r.get("OP Type") == "FlashAttn"]
    runs = args.runs
    expected = runs * len(names)
    if len(raw_rows) != expected:
        raise RuntimeError(
            f"{summary}: expected {expected} FlashAttn rows ({len(names)} cases x {runs}), "
            f"got {len(raw_rows)}; use batch --perf_mode (not --one_by_one)"
        )

    out_rows = []
    pos = 0
    for case_id in hot + cold:
        meta = _case_meta(case_id, normalize_params(all_cases[case_id]))
        for run in range(1, runs + 1):
            row = {
                "case": meta["case"],
                "case_id": meta["case_id"],
                "phase": meta["phase"],
                "mode": "hot" if case_id in hot else "cold",
                "run": run,
            }
            row.update(
                {k: v for k, v in meta.items() if k not in ("case", "case_id", "phase")}
            )
            row.update(raw_rows[pos])
            row["kernel_name"] = kernel_names.get(
                (row.get("Device_id"), row.get("Stream ID"), row.get("Task ID")), ""
            )
            pos += 1
            out_rows.append(row)

    out = Path(args.out) if args.out else Path(args.perf_dir) / "profiling_raw.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)
    print(f"exported {len(out_rows)} rows -> {out}")


if __name__ == "__main__":
    main()
