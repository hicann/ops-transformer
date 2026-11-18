# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Compare benchmark.py exports; reject different workloads or measurement counts."""

import argparse
import json
import math
from pathlib import Path
import statistics


def read(folder):
    environment = json.loads((folder / "environment.json").read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in (folder / "samples.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    grouped = {}
    for row in rows:
        key = row["case_id"], row["requested_template"], row["round"]
        if key in grouped:
            raise ValueError(f"duplicate measurement {key}")
        for field in ("kernel_us", "e2e_us"):
            if len(row[field]) != environment["samples"] or not all(
                math.isfinite(v) and v > 0 for v in row[field]
            ):
                raise ValueError(f"incomplete/invalid {field}: {key}")
        grouped[key] = row
    cases = {(key[0], key[1]) for key in grouped}
    if not cases or set(grouped) != {
        (case, template, repeat)
        for case, template in cases
        for repeat in range(environment["rounds"])
    }:
        raise ValueError("missing measurement rounds")
    return environment, grouped


def compare(a5, a6):
    left_env, left = read(Path(a5))
    right_env, right = read(Path(a6))
    if (left_env["arch"], right_env["arch"]) != ("arch35", "arch92"):
        raise ValueError("compare requires A5 then A6")
    contract = left_env.get("interface_contract")
    if not contract or contract != right_env.get("interface_contract"):
        raise ValueError(
            "missing or different interface contract; rerun matching interfaces"
        )
    for field in ("warmup", "samples", "rounds", "timing"):
        if left_env[field] != right_env[field]:
            raise ValueError(f"different measurement protocol: {field}")
    if left.keys() != right.keys():
        raise ValueError("different measured case/template/round sets")
    grouped = {}
    for key, baseline in left.items():
        candidate = right[key]
        for field in ("input_sha256", "layout", "inference", "kwargs"):
            if baseline[field] != candidate[field]:
                raise ValueError(f"different workload {field}: {key}")
        group = grouped.setdefault(
            key[:2], {f: ([], []) for f in ("kernel_us", "e2e_us")}
        )
        for field, (old, new) in group.items():
            old.append(statistics.mean(baseline[field]))
            new.append(statistics.mean(candidate[field]))
    result = []
    for (case, template), values in sorted(grouped.items()):
        row = dict(case_id=case, requested_template=template)
        for field, (old, new) in values.items():
            a, b = statistics.median(old), statistics.median(new)
            row[field] = dict(a5=a, a6=b, a6_over_a5=b / a)
        row["pass"] = all(row[f]["a6_over_a5"] <= 1 for f in values)
        result.append(row)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("a5", type=Path)
    parser.add_argument("a6", type=Path)
    args = parser.parse_args()
    result = compare(args.a5, args.a6)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if all(row["pass"] for row in result) else 1)


if __name__ == "__main__":
    main()
