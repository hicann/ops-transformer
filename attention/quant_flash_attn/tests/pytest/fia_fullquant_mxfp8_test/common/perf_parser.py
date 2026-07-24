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

import csv
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def parse_op_summary(csv_path: str,
                     operator_name: str = "FusedInferAttentionScore") -> List[Dict]:
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    entries = []
    for r in rows:
        if r["OP Type"] != operator_name:
            continue
        entries.append({
            "task_id": int(r["Task ID"]),
            "start_time_us": float(r["Task Start Time(us)"]),
            "duration_us": float(r["Task Duration(us)"]),
            "input_shapes": r.get("Input Shapes", ""),
            "output_shapes": r.get("Output Shapes", ""),
            "aicore_time_us": float(r.get("aicore_time(us)", 0)),
            "aic_total_cycles": int(r.get("aic_total_cycles", 0)),
            "cube_utilization": float(r.get("cube_utilization(%)", 0)),
        })
    entries.sort(key=lambda x: x["start_time_us"])
    return entries


def _parse_shapes_str(shapes_str: str) -> List[List[int]]:
    s = shapes_str.strip().strip('"')
    if not s:
        return []
    parts = [p.strip() for p in s.split(";") if p.strip()]
    result = []
    for p in parts:
        dims = [int(x) for x in p.split(",") if x.strip()]
        result.append(dims)
    return result


def extract_shapes_from_entry(entry: Dict) -> str:
    input_shapes = _parse_shapes_str(entry.get("input_shapes", ""))
    labels = ["Q", "K", "V"]
    parts = []
    for i, label in enumerate(labels):
        if i < len(input_shapes):
            parts.append(f"{label}={input_shapes[i]}")
    return "  ".join(parts)


def find_op_summary_csv(prof_dir: str) -> Optional[Path]:
    prof_path = Path(prof_dir)
    csv_files = sorted(prof_path.glob("mindstudio_profiler_output/op_summary_*.csv"))
    if csv_files:
        return csv_files[-1]
    return None


def parse_prof_directory(prof_dir: str,
                         operator_name: str = "FusedInferAttentionScore") -> Tuple[List[Dict], Optional[Path]]:
    csv_path = find_op_summary_csv(prof_dir)
    if csv_path is None:
        print(f"[perf] No op_summary CSV found in {prof_dir}")
        return [], None

    entries = parse_op_summary(str(csv_path), operator_name)
    print(f"[perf] Found {len(entries)} {operator_name} entries in {csv_path.name}")

    results = [
        {
            "task_id": e["task_id"],
            "duration_us": e["duration_us"],
            "aicore_time_us": e["aicore_time_us"],
            "cube_utilization": e["cube_utilization"],
            "shapes": extract_shapes_from_entry(e),
        }
        for e in entries
    ]
    return results, csv_path


def print_report(results: List[Dict]):
    print()
    print("=" * 120)
    print(f"{'#':>3}  {'Duration':>10} {'AI Core':>10} {'Cube%':>7}  {'Input Shapes'}")
    print("-" * 120)
    for i, r in enumerate(results):
        print(f"{i+1:>3}  {r['duration_us']:>8.1f}us {r['aicore_time_us']:>8.1f}us {r['cube_utilization']:>6.1f}%  {r['shapes']}")
    print("=" * 120)
    print(f"\nTotal: {len(results)} entries")


def save_report(results: List[Dict], csv_path: Optional[Path], output_dir: str = ".") -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if csv_path:
        timestamp = csv_path.stem.replace("op_summary_", "")
    else:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    log_file = output_dir / f"perf_report_{timestamp}.log"

    lines = []
    lines.append("=" * 120)
    lines.append(f"{'#':>3}  {'Duration':>10} {'AI Core':>10} {'Cube%':>7}  {'Input Shapes'}")
    lines.append("-" * 120)
    for i, r in enumerate(results):
        lines.append(f"{i+1:>3}  {r['duration_us']:>8.1f}us {r['aicore_time_us']:>8.1f}us {r['cube_utilization']:>6.1f}%  {r['shapes']}")
    lines.append("=" * 120)
    lines.append(f"\nTotal: {len(results)} entries")

    with open(log_file, "w") as f:
        f.write("\n".join(lines))

    if csv_path and csv_path.exists():
        shutil.copy2(csv_path, output_dir / csv_path.name)

    return log_file


def parse_baseline_log(log_path: str) -> List[Dict]:
    path = Path(log_path)
    if not path.exists():
        return []

    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("=") or line.startswith("-") or line.startswith("#") or line.startswith("Total"):
                continue
            m = re.match(r"\s*(\d+)\s+([\d.]+)us\s+([\d.]+)us\s+([\d.]+)%\s+(.*)", line)
            if m:
                entries.append({
                    "duration_us": float(m.group(2)),
                    "aicore_time_us": float(m.group(3)),
                    "cube_utilization": float(m.group(4)),
                    "shapes": m.group(5).strip(),
                })
    return entries


def compare_with_baseline(current: List[Dict], baseline: List[Dict], threshold: float = 8.0) -> List[Dict]:
    baseline_by_shapes = {e["shapes"]: e for e in baseline}
    results = []
    for cur in current:
        shapes = cur["shapes"]
        base = baseline_by_shapes.get(shapes)
        if base is None:
            results.append({
                **cur,
                "baseline_us": None,
                "diff_pct": None,
                "status": "NEW",
            })
            continue

        base_dur = base["duration_us"]
        cur_dur = cur["duration_us"]
        diff_pct = ((cur_dur - base_dur) / base_dur) * 100 if base_dur > 0 else 0

        if diff_pct > threshold:
            status = "FAILED"
        elif diff_pct < -threshold:
            status = "IMPROVED"
        else:
            status = "PASS"

        results.append({
            **cur,
            "baseline_us": base_dur,
            "diff_pct": diff_pct,
            "status": status,
        })
    return results


def print_comparison_report(results: List[Dict], threshold: float = 8.0):
    print()
    print("=" * 140)
    print(f"{'#':>3}  {'Current':>10} {'Baseline':>10} {'Diff':>8} {'Status':>8}  {'Input Shapes'}")
    print("-" * 140)
    for i, r in enumerate(results):
        cur_str = f"{r['duration_us']:>8.1f}us"
        base_str = f"{r['baseline_us']:>8.1f}us" if r["baseline_us"] is not None else "       N/A"
        diff_str = f"{r['diff_pct']:>+7.1f}%" if r["diff_pct"] is not None else "      N/A"
        status = r["status"]
        print(f"{i+1:>3}  {cur_str} {base_str} {diff_str} {status:>8}  {r['shapes']}")
    print("=" * 140)

    failed = sum(1 for r in results if r["status"] == "FAILED")
    passed = sum(1 for r in results if r["status"] == "PASS")
    improved = sum(1 for r in results if r["status"] == "IMPROVED")
    new_cases = sum(1 for r in results if r["status"] == "NEW")

    print(f"\nTotal: {len(results)} entries | PASS: {passed} | FAILED: {failed} | IMPROVED: {improved} | NEW: {new_cases}")
    print(f"Threshold: {threshold}%")

    if failed > 0:
        print(f"\n*** PERFORMANCE REGRESSION DETECTED: {failed} case(s) exceeded {threshold}% threshold ***")
