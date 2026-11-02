#!/usr/bin/python3
# coding=utf-8

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Parse msprof op_summary CSVs for the CCU AllToAllQuantMatmul ST.

  --case N   : analyze case N only (1-indexed, sorted by time)
  --latest   : analyze the most recent case (default)
  --all      : analyze all cases

读数口径：每个 case 取后 5 次 AIC kernel 迭代的平均（op_summary 无 CCU kernel 条目，
CCU 通信窗口与 AIC kernel overlap，AIC kernel duration 即融合整体耗时）。
"""

import argparse
import csv
import glob
import os
import re
from datetime import datetime

_DEFAULT_PROF = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "prof"
)
KERNEL_SUBSTR = "AllToAllQuantMatmulKernelCcu"
LAST_N = 5  # 后 5 次取平均

_TS_RE = re.compile(r"PROF_\d+_(\d{17})_")


def parse_ts_from_dir(dirname):
    m = _TS_RE.search(dirname)
    if not m:
        return None
    return datetime.strptime(m.group(1), "%Y%m%d%H%M%S%f")


def find_csvs(prof_base, prof_dir_name):
    return glob.glob(
        os.path.join(
            prof_base, prof_dir_name, "mindstudio_profiler_output", "op_summary_*.csv"
        )
    )


def parse_csv(csv_path):
    durs = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            name = row.get("Name") or row.get("Kernel Name") or ""
            if KERNEL_SUBSTR not in name:
                continue
            # op_summary 的 duration 单位 us
            try:
                durs.append(float(row["Duration(us)"]))
            except (KeyError, ValueError, TypeError):
                continue
    durs.sort(key=lambda _: 0)  # 保持 csv 顺序（起跑时间序）
    return durs


def main():
    parser = argparse.ArgumentParser(
        description="Parse msprof op_summary for CCU a2a ST."
    )
    parser.add_argument("--prof", default=_DEFAULT_PROF, help="prof output base dir")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--case", type=int, help="analyze case N (1-indexed)")
    group.add_argument(
        "--latest", action="store_true", help="most recent case (default)"
    )
    group.add_argument("--all", action="store_true", help="all cases")
    args = parser.parse_args()

    if not os.path.isdir(args.prof):
        print("prof dir not found: %s" % args.prof)
        return 1

    cases = []
    for d in sorted(os.listdir(args.prof)):
        ts = parse_ts_from_dir(d)
        if ts is None:
            continue
        cases.append((ts, d))
    cases.sort()
    if not cases:
        print("no PROF_* dirs under %s" % args.prof)
        return 1

    if args.all:
        selected = cases
    elif args.case:
        if not (1 <= args.case <= len(cases)):
            print("case %d out of range (total %d)" % (args.case, len(cases)))
            return 1
        selected = [cases[args.case - 1]]
    else:
        selected = [cases[-1]]

    for ts, d in selected:
        durs = []
        for csv_path in find_csvs(args.prof, d):
            durs.extend(parse_csv(csv_path))
        if not durs:
            print("[%s] no '%s' records" % (d, KERNEL_SUBSTR))
            continue
        last = durs[-LAST_N:]
        avg = sum(last) / len(last)
        print("[%s] kernel=%s iters=%d" % (d, KERNEL_SUBSTR, len(durs)))
        print(
            "  all: min=%.1fus max=%.1fus | last-%d avg=%.1fus"
            % (min(durs), max(durs), len(last), avg)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
