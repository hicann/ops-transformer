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
import argparse
import csv
import math
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("profile_dir", type=Path)
    parser.add_argument("--kernel-name", default="ChunkKdaFwd")
    parser.add_argument(
        "--component",
        action="append",
        default=[],
        nargs=2,
        metavar=("KERNEL_NAME", "PROFILE_DIR"),
        help="add a separately profiled kernel to the end-to-end device-time gate",
    )
    parser.add_argument("--max-ms", type=float, default=11.0)
    return parser.parse_args()


def collect_samples(profile_dir, kernel_name):
    samples = []
    for path in sorted(profile_dir.rglob("OpBasicInfo*.csv")):
        with path.open(newline="", encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                if kernel_name not in row.get("Op Name", ""):
                    continue
                try:
                    duration_ms = float(row["Task Duration(us)"]) / 1000.0
                except (KeyError, TypeError, ValueError):
                    continue
                if math.isfinite(duration_ms) and duration_ms > 0:
                    samples.append(duration_ms)
    if not samples:
        raise RuntimeError(f"no {kernel_name} samples found in {profile_dir}")
    return samples


def main():
    args = parse_args()
    components = [(args.kernel_name, args.profile_dir)]
    components.extend((name, Path(path)) for name, path in args.component)
    maximum_total = 0.0

    for kernel_name, profile_dir in components:
        samples = collect_samples(profile_dir, kernel_name)
        minimum = min(samples)
        mean = sum(samples) / len(samples)
        maximum = max(samples)
        maximum_total += maximum
        print(
            f"kernel={kernel_name} samples={len(samples)} "
            f"min/mean/max={minimum:.3f}/{mean:.3f}/{maximum:.3f} ms"
        )
    print(
        f"end_to_end_device_max={maximum_total:.3f} ms hard_limit={args.max_ms:.3f} ms"
    )
    if maximum_total > args.max_ms:
        raise SystemExit(
            f"performance gate failed: device max {maximum_total:.3f} ms "
            f"> {args.max_ms:.3f} ms"
        )


if __name__ == "__main__":
    main()
