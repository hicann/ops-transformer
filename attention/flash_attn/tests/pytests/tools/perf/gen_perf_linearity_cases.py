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
"""生成 B/S1/S2 线性度扫描用例（单一代表配置）。

配置固定为 D=128、mask=3、BSND、G=8、fp16，只改变一个轴。
输出 linearity_generated.py 和 linearity_manifest.json。
"""

import argparse
import json
import math
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = TESTS_DIR / "test_cases" / "perf"

D = 128
DV = 128
N2 = 4
G = 8
N1 = G * N2
MASK = 3
LAYOUT = "BSND"
DTYPE = "fp16"
RUNS = 5

B_PREFILL = [1, 2, 4, 8, 16, 32]
B_DECODE = [1, 2, 4, 8, 16, 32, 64]
S1_PREFILL = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
S1_DECODE = [1, 2, 3, 4, 8, 16]
S2_PREFILL = [2048, 4096, 8192, 16384, 32768]
S2_DECODE = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]


LICENSE_HEADER = """# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""


def _name(axis, phase, b, s1, s2):
    return (
        f"Perf_Lin_{axis}_{phase}_{LAYOUT}_D{D}_M{MASK}_QN{N1}_KVN{N2}"
        f"_B{b}_QS{s1}_KVS{s2}_{DTYPE}"
    )


def _case(b, s1, s2):
    return {
        "B": [b],
        "N1": [N1],
        "N2": [N2],
        "S1": [s1],
        "S2": [s2],
        "D": [D],
        "layout_q": [LAYOUT],
        "layout_kv": [LAYOUT],
        "layout_out": [LAYOUT],
        "Dtype": [DTYPE],
        "q_range": [(-1, 1)],
        "k_range": [(-1, 1)],
        "v_range": [(-1, 1)],
        "scale": [1.0 / math.sqrt(D)],
        "mask_mode": [MASK],
    }


def _add(cases, meta, axis, phase, b, s1, s2):
    name = _name(axis, phase, b, s1, s2)
    if name in cases:
        return
    cases[name] = _case(b, s1, s2)
    meta[name] = {
        "layer": "Perf_Lin",
        "group": f"{axis}_{phase}",
        "phase": phase,
        "axis": axis,
        "label": f"线性度 {axis} {phase}",
        "runs": RUNS,
        "B": b,
        "S1": s1,
        "S2": s2,
        "D": D,
        "DV": DV,
        "mask_mode": MASK,
        "G": G,
        "layout": LAYOUT,
        "N2": N2,
    }


def gen():
    cases, meta = {}, {}
    for b in B_PREFILL:
        _add(cases, meta, "B", "prefill", b, 2048, 2048)
    for b in B_DECODE:
        _add(cases, meta, "B", "decode", b, 1, 8192)
    for s1 in S1_PREFILL:
        _add(cases, meta, "S1", "prefill", 8, s1, 8192)
    for s1 in S1_DECODE:
        _add(cases, meta, "S1", "decode", 32, s1, 8192)
    for s2 in S2_PREFILL:
        _add(cases, meta, "S2", "prefill", 8, 2048, s2)
    for s2 in S2_DECODE:
        _add(cases, meta, "S2", "decode", 32, 1, s2)
    return cases, meta


def _render(cases):
    lines = [
        *LICENSE_HEADER.splitlines(),
        '"""自动生成，请勿手改；重新生成: python3 tools/perf/gen_perf_linearity_cases.py"""',
        "",
        "TestCases = {",
    ]
    for name, case in cases.items():
        lines.append(f"    {name!r}: {{")
        for key, value in case.items():
            lines.append(f"        {key!r}: {value!r},")
        lines.append("    },")
    lines.append("}")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cases, meta = gen()
    if args.dry_run:
        for name in cases:
            print(name)
        print(f"total {len(cases)}")
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "linearity_generated.py").write_text(_render(cases))
    manifest = {
        "layer": "Perf_Lin",
        "dtype": DTYPE,
        "count": len(cases),
        "cases": meta,
    }
    (args.out_dir / "linearity_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    print(f"generated {len(cases)} linearity cases -> {args.out_dir}")


if __name__ == "__main__":
    main()
