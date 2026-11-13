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
"""把 profiling_raw.csv 汇总成每个 case 一行的最终性能表。

除耗时外，按模型 FLOPs 追加两列：
  theory_us  理论耗时(us) = 2*B*N1*可见对*(D+DV) / 峰值算力
  mfu_pct    实测 MFU(%)  = theory_us / avg_us * 100
可见对: mask=0 取 S1*S2；mask=3 取右对齐 causal(第 i 行可见 min(S2, max(0, i+S2-S1+1)))；
mask=4 需要窗口参数，留空。峰值算力由 --peak-tflops 指定（<=0 时不计算这两列）。
"""

import argparse
import csv
from pathlib import Path

CASE_FIELDS = [
    "B",
    "N2",
    "G",
    "D",
    "DV",
    "S1",
    "S2",
    "mask",
    "dtype",
    "layout_q",
    "layout_kv",
    "layout_out",
    "max_seqlen_q",
    "max_seqlen_kv",
    "cu_seqlens_q",
    "cu_seqlens_kv",
    "seqused_q",
    "seqused_kv",
]


def _visible_pairs(mask, s1, s2):
    """模型视角的可见 QK 对数；mask=4 需要窗口，返回 None。"""
    if mask == 0:
        return s1 * s2
    if mask == 3:  # rightDownCausal: 右下对齐
        return sum(min(s2, max(0, i + s2 - s1 + 1)) for i in range(s1))
    return None


def _mfu_fields(row, avg_us, peak_tflops):
    try:
        b = int(float(row["B"]))
        n2 = int(float(row["N2"]))
        g = int(float(row["G"]))
        d = int(float(row["D"]))
        dv = int(float(row["DV"]))
        s1 = int(float(row["S1"]))
        s2 = int(float(row["S2"]))
        mask = int(float(row["mask"]))
    except (KeyError, TypeError, ValueError):
        return "", ""
    if avg_us <= 0 or peak_tflops <= 0:
        return "", ""
    pairs = _visible_pairs(mask, s1, s2)
    if not pairs:
        return "", ""
    flops = 2.0 * b * n2 * g * pairs * (d + dv)
    theory_us = flops / (peak_tflops * 1e12) * 1e6
    return f"{theory_us:.3f}", f"{theory_us / avg_us * 100:.2f}"


def _trimmed(durations, mode):
    if not durations:
        return []
    if mode == "hot":
        return durations[1:]
    vals = sorted(durations)
    return vals[1:-1] if len(vals) >= 3 else vals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--peak-tflops",
        type=float,
        default=0.0,
        help="峰值算力 TFLOPS(fp16/bf16 dense)，用于理论耗时/MFU；默认 0=不计算",
    )
    args = parser.parse_args()

    raw_path = Path(args.raw)
    grouped = {}
    with open(raw_path) as f:
        for row in csv.DictReader(f):
            grouped.setdefault(row["case_id"], []).append(row)

    out_rows = []
    for case_id, rows in grouped.items():
        rows.sort(key=lambda r: int(r["run"]))
        first = rows[0]
        durations = [
            float(r["Task Duration(us)"]) for r in rows if r.get("Task Duration(us)")
        ]
        kept = _trimmed(durations, first["mode"])
        kernel_name = next((r["kernel_name"] for r in rows if r.get("kernel_name")), "")
        out = {
            "case": first["case"],
            "case_id": case_id,
            "phase": first["phase"],
            "mode": first["mode"],
        }
        for key in CASE_FIELDS:
            out[key] = first[key]
        avg_us = sum(kept) / len(kept) if kept else 0.0
        out["avg_us"] = f"{avg_us:.3f}" if kept else ""
        out["theory_us"], out["mfu_pct"] = _mfu_fields(first, avg_us, args.peak_tflops)
        out["kernel_name"] = kernel_name
        out_rows.append(out)

    out_path = (
        Path(args.out) if args.out else raw_path.with_name("profiling_summary.csv")
    )
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)
    print(f"summarized {len(out_rows)} cases -> {out_path}")


if __name__ == "__main__":
    main()
