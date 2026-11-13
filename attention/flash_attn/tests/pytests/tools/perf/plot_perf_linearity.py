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
"""绘制 B/S1/S2 性能线性度图（横轴参数，纵轴耗时 us）。"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

for _font in (
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/opentype/unifont/unifont_jp.otf",
):
    if Path(_font).exists():
        font_manager.fontManager.addfont(_font)
        plt.rcParams["font.family"] = font_manager.FontProperties(
            fname=_font
        ).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

AXES = ["B", "S1", "S2"]
PHASES = ["prefill", "decode"]


def _read_durations(case_dir):
    path = case_dir / "op_summary.csv"
    if not path.exists():
        return []
    with open(path) as f:
        rows = list(csv.DictReader(f))
    rows = [r for r in rows if r.get("OP Type") == "FlashAttn"]
    rows.sort(key=lambda r: float(r["Task Start Time(us)"]))
    return [float(r["Task Duration(us)"]) for r in rows]


def _trimmed(durations, cold):
    if not durations:
        return []
    if cold:
        vals = sorted(durations)
        return vals[1:-1] if len(vals) >= 3 else vals
    return durations[1:]


def load(perf_dir, manifest_path):
    manifest = json.loads(Path(manifest_path).read_text())["cases"]
    result = []
    with open(Path(perf_dir) / "perf.csv") as f:
        for row in csv.DictReader(f):
            name = row["case"].split("/")[-1]
            meta = manifest.get(name)
            if meta is None:
                continue
            case_dir = Path(perf_dir) / row["case"].replace("/", "_")
            cold = row.get("mode", "") == "cold"
            durations = _trimmed(_read_durations(case_dir), cold)
            avg = float(row["avg_us"]) if row.get("avg_us") else float("nan")
            if durations:
                lo, hi = min(durations), max(durations)
                std = float(np.std(durations))
            else:
                lo = hi = avg
                std = 0.0
            result.append(
                {
                    "case": name,
                    "axis": meta["axis"],
                    "phase": meta["phase"],
                    "B": meta["B"],
                    "S1": meta["S1"],
                    "S2": meta["S2"],
                    "D": meta["D"],
                    "DV": meta["DV"],
                    "M": meta["mask_mode"],
                    "G": meta["G"],
                    "N2": meta["N2"],
                    "layout": meta["layout"],
                    "avg_us": avg,
                    "min_us": lo,
                    "max_us": hi,
                    "std_us": std,
                    "mode": row.get("mode", ""),
                }
            )
    return result


def _fixed_text(pts, axis):
    parts = []
    for key in ("B", "S1", "S2"):
        if key == axis:
            continue
        vals = sorted({r[key] for r in pts})
        parts.append(f"{key}={vals[0]}" if len(vals) == 1 else f"{key}=var")
    return ", ".join(parts)


def _common_text(rows):
    r = rows[0]
    return (
        f"固定参数: D={r['D']}, DV={r['DV']}, mask={r['M']}, "
        f"layout={r['layout']}, G={r['G']}, N2={r['N2']}, fp16"
    )


def load_raw(raw_path, manifest_path):
    manifest = json.loads(Path(manifest_path).read_text())["cases"]
    grouped = {}
    with open(raw_path) as f:
        for row in csv.DictReader(f):
            grouped.setdefault(row["case"], []).append(row)

    result = []
    for name, rows in grouped.items():
        meta = manifest.get(name)
        if meta is None:
            continue
        rows.sort(key=lambda r: int(r["run"]))
        cold = rows[0].get("mode", "") == "cold"
        durations = _trimmed(
            [float(r["Task Duration(us)"]) for r in rows if r.get("Task Duration(us)")],
            cold,
        )
        avg = float(np.mean(durations)) if durations else float("nan")
        result.append(
            {
                "case": name,
                "axis": meta["axis"],
                "phase": meta["phase"],
                "B": meta["B"],
                "S1": meta["S1"],
                "S2": meta["S2"],
                "D": meta["D"],
                "DV": meta["DV"],
                "M": meta["mask_mode"],
                "G": meta["G"],
                "N2": meta["N2"],
                "layout": meta["layout"],
                "avg_us": avg,
                "min_us": min(durations) if durations else avg,
                "max_us": max(durations) if durations else avg,
                "std_us": float(np.std(durations)) if durations else 0.0,
                "mode": rows[0].get("mode", ""),
            }
        )
    return result


def _plot_axis(rows, axis, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), squeeze=False)
    for col, phase in enumerate(PHASES):
        ax = axes[0][col]
        pts = [r for r in rows if r["axis"] == axis and r["phase"] == phase]
        pts.sort(key=lambda r: r[axis])
        if not pts:
            ax.set_visible(False)
            continue
        x = np.array([r[axis] for r in pts], dtype=float)
        y = np.array([r["avg_us"] for r in pts], dtype=float)
        lo = np.array([r["min_us"] for r in pts], dtype=float)
        hi = np.array([r["max_us"] for r in pts], dtype=float)
        lo = np.minimum(lo, y)
        hi = np.maximum(hi, y)
        ax.errorbar(
            x,
            y,
            yerr=[y - lo, hi - y],
            fmt="o-",
            capsize=3,
            color="#1f77b4",
            ecolor="#9ecae1",
            linewidth=1.5,
            markersize=4,
        )
        if len(x) >= 2:
            k, b = np.polyfit(x, y, 1)
            y_fit = k * x + b
            ss_res = float(np.sum((y - y_fit) ** 2))
            ss_tot = float(np.sum((y - np.mean(y)) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
            ax.plot(x, y_fit, "--", color="gray", linewidth=1, alpha=0.8)
            ax.text(
                0.03,
                0.95,
                f"slope={k:.4g} us/{axis},  R2={r2:.4f}",
                transform=ax.transAxes,
                va="top",
                fontsize=8,
            )
        for r in pts:
            if r["mode"] == "CRASH" or not np.isfinite(r["avg_us"]):
                ax.annotate(
                    "CRASH",
                    (r[axis], 0),
                    textcoords="offset points",
                    xytext=(0, 6),
                    ha="center",
                    fontsize=7,
                    color="red",
                )
        ax.set_title(f"{axis} 线性度 - {phase}  ({_fixed_text(pts, axis)})")
        ax.set_xlabel(axis)
        ax.set_ylabel("耗时 (us)")
        ax.grid(alpha=0.3)
    fig.suptitle(_common_text(rows), fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--perf-dir", default=None)
    parser.add_argument("--raw", default=None)
    parser.add_argument(
        "--manifest",
        default=str(
            Path(__file__).resolve().parents[2]
            / "test_cases/perf/linearity_manifest.json"
        ),
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.raw:
        rows = load_raw(args.raw, args.manifest)
    elif args.perf_dir:
        rows = load(args.perf_dir, args.manifest)
    else:
        parser.error("need --raw or --perf-dir")

    with open(out / "linearity_summary.csv", "w", newline="") as f:
        fields = [
            "case",
            "phase",
            "axis",
            "B",
            "S1",
            "S2",
            "D",
            "DV",
            "M",
            "G",
            "N2",
            "layout",
            "avg_us",
            "min_us",
            "max_us",
            "std_us",
            "mode",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    for axis in AXES:
        _plot_axis(rows, axis, out / f"linearity_{axis}.png")

    fig, axes = plt.subplots(3, 2, figsize=(11, 12), squeeze=False)
    for row_i, axis in enumerate(AXES):
        for col, phase in enumerate(PHASES):
            ax = axes[row_i][col]
            pts = [r for r in rows if r["axis"] == axis and r["phase"] == phase]
            pts.sort(key=lambda r: r[axis])
            if not pts:
                ax.set_visible(False)
                continue
            x = [r[axis] for r in pts]
            y = [r["avg_us"] for r in pts]
            ax.plot(x, y, "o-", color="#1f77b4", linewidth=1.5, markersize=3)
            ax.set_title(f"{axis} - {phase}\n{_fixed_text(pts, axis)}", fontsize=9)
            ax.set_xlabel(axis)
            ax.set_ylabel("us")
            ax.grid(alpha=0.3)
    fig.suptitle(_common_text(rows), fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out / "linearity_all.png", dpi=160)
    plt.close(fig)
    print(f"cases={len(rows)} -> {out}")


if __name__ == "__main__":
    main()
