#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""逐个单跑 func_rdv 用例，规避 pytest 批跑可能的 507015 级联问题"""

import subprocess
import sys
import os

os.environ.setdefault("MALLOC_ARENA_MAX", "10")

# 从 paramset 提取所有 case 名称
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qmla_paramset_func_rdv as paramset

CASES = paramset.CASES
SKIP_CASES = getattr(paramset, "SKIP_CASES", set())

passed = 0
failed = 0
skipped = 0
failed_cases = []

for i, case in enumerate(CASES):
    name = case["name"]
    if name in SKIP_CASES:
        print(f"[{i + 1}/{len(CASES)}] SKIP: {name}")
        skipped += 1
        continue

    print(f"\n{'=' * 80}")
    print(f"[{i + 1}/{len(CASES)}] Running: {name}")
    print(f"{'=' * 80}", flush=True)

    log_file = f"single_run_logs/{name}.log"
    os.makedirs("single_run_logs", exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-m",
        "func_rdv",
        "-v",
        "-s",
        "-k",
        name,
        "--tb=short",
    ]

    with open(log_file, "w") as f:
        result = subprocess.run(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )

    if result.returncode == 0:
        print("  -> PASSED", flush=True)
        passed += 1
    else:
        print(f"  -> FAILED (returncode={result.returncode})", flush=True)
        failed += 1
        failed_cases.append(name)

        # 检查是否有 507015
        with open(log_file, "r") as f:
            content = f.read()
            if "507015" in content:
                print("  -> 507015 detected!", flush=True)

print(f"\n{'=' * 80}")
print(
    f"Summary: {passed} passed, {failed} failed, {skipped} skipped, total {len(CASES)}"
)
if failed_cases:
    print("Failed cases:")
    for c in failed_cases:
        print(f"  - {c}")
print(f"{'=' * 80}")
