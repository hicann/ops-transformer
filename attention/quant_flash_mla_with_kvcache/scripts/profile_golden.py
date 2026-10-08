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
"""Collect ordinary operator performance with testkit-style L2 flushing.

Run through build_and_run.sh profile after a successful all run. Each capture
generates inputs directly with fixed seeds; --mode=npu excludes the reference
and comparison kernels from the measurements. No tensor cache is used.
"""

import csv
import datetime
import json
from pathlib import Path
import statistics
import subprocess
import sys


def main():
    op_dir = Path(__file__).resolve().parents[1]
    repo = op_dir.parents[1]
    output = (
        repo
        / "build_out"
        / "qmla_profile"
        / datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    output.mkdir(parents=True)
    script = op_dir / "tests/pytest/common/qmla_with_kvcache_golden.py"
    application = [sys.executable, str(script), "--mode=npu", "--flush-l2"]
    operator = "QuantFlashMlaWithKvcache"

    def run(label, command):
        print(f"[{label}] {command}", flush=True)
        record = {"argv": command, "cwd": str(repo)}
        try:
            with (output / f"{label}.log").open("w") as log:
                result = subprocess.run(
                    command, cwd=repo, stdout=log, stderr=subprocess.STDOUT
                )
            record["returncode"] = result.returncode
            if result.returncode:
                raise RuntimeError(f"{label} failed; see {output / (label + '.log')}")
        finally:
            (output / f"{label}.command.json").write_text(json.dumps(record, indent=2))

    run(
        "performance",
        [
            "msprof",
            f"--output={output / 'performance'}",
            "--aic-mode=task-based",
            *application,
            "--warmup=5",
            "--runs=20",
        ],
    )
    summaries = list(
        (output / "performance").glob(
            "PROF_*/mindstudio_profiler_output/op_summary_*.csv"
        )
    )
    rows = []
    for path in summaries:
        with path.open() as stream:
            rows.extend(
                row for row in csv.DictReader(stream) if row["OP Type"] == operator
            )
    rows.sort(key=lambda row: float(row["Task Start Time(us)"]))
    if len(rows) != 25:
        raise RuntimeError(
            f"Expected 25 {operator} tasks, found {len(rows)}; refusing ambiguous timing"
        )
    durations = [float(row["Task Duration(us)"]) for row in rows[5:]]
    summary = {
        "operator": operator,
        "warmup": 5,
        "samples": len(durations),
        "mean_us": statistics.mean(durations),
        "median_us": statistics.median(durations),
        "min_us": min(durations),
        "max_us": max(durations),
        "durations_us": durations,
    }
    # Same cold aggregation as testkit: discard one min and one max sample.
    cold_us = statistics.mean(sorted(durations)[1:-1])
    flops = 2 * 96 * 8 * 102400 * (576 + 512)
    peak = 756.94e12
    summary.update(
        {
            "flush_l2": "256 MiB read per iteration",
            "cold_trimmed_mean_us": cold_us,
            "effective_flops": flops,
            "fp8_peak_flops_per_s": peak,
            "achieved_tflops": flops / (cold_us * 1e-6) / 1e12,
            "mfu_percent": 100 * flops / (cold_us * 1e-6) / peak,
        }
    )
    (output / "performance_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Profile artifacts: {output}", flush=True)


if __name__ == "__main__":
    main()
