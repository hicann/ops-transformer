# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""The single case manifest and CLI selection (no NPU imports)."""

import argparse
import ast
import csv
import json
import os
from pathlib import Path


CSV = Path(__file__).with_name("cases") / "ttk_cases.csv"
ARCHITECTURES = ("arch35", "arch92")


def architecture():
    arch = os.environ.get("FLASH_KDA_ARCH", "arch35")
    if arch not in ARCHITECTURES:
        raise ValueError("FLASH_KDA_ARCH must be arch35 or arch92")
    return arch


def attributes(row):
    attrs = ast.literal_eval(row["attributes"])
    attrs["testcase_name"] = row["testcase_name"]
    return attrs


def recipe(row):
    return json.loads(attributes(row).get("test_config", "{}"))


def read_cases(suite="public", names=None, *, dut=None, arch=None):
    dut = dut or os.environ.get("KDA_DUT", "installed")
    arch = arch or architecture()
    with CSV.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len({r["testcase_name"] for r in rows}) != len(rows):
        raise ValueError("duplicate testcase_name in shared CSV")
    selected = []
    for row in rows:
        if row["is_enabled"].lower() != "true":
            continue
        test = recipe(row)
        if names and row["testcase_name"] not in names:
            continue
        if not names:
            if suite == "public" and (
                test.get("requires_source") or test["kind"] == "performance"
            ):
                continue
            if suite == "whitebox" and not test.get("requires_source"):
                continue
            if suite not in ("all", "public", "whitebox") and test["category"] != suite:
                continue
        if arch not in test.get("architectures", ARCHITECTURES):
            raise ValueError(
                f"{row['testcase_name']} is not applicable to {arch}: {test.get('reason')}"
            )
        if test.get("requires_source") and dut != "source":
            raise ValueError(
                f"{row['testcase_name']} requires KDA_DUT=source (whitebox)"
            )
        selected.append(row)
    if names:
        missing = set(names) - {r["testcase_name"] for r in selected}
        if missing:
            raise ValueError(f"unknown/disabled cases: {sorted(missing)}")
    if not selected:
        raise ValueError(f"no cases selected: {suite}")
    return selected


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        default="public",
        choices=(
            "public",
            "stc",
            "invalid",
            "generalized",
            "redline",
            "profiling",
            "whitebox",
            "all",
        ),
    )
    parser.add_argument("--cases", help="comma-separated case IDs")
    parser.add_argument(
        "--ids", action="store_true", help="print one comma-separated list for TTK -t"
    )
    args = parser.parse_args()
    rows = read_cases(args.suite, args.cases.split(",") if args.cases else None)
    print(("," if args.ids else "\n").join(row["testcase_name"] for row in rows))
