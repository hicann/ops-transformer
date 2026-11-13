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

"""基础性能用例生成器。

按 CASE_GROUPS 生成 base_generated.py 和 manifest.json。
win_left_right 只对 mask_mode=4 生效；只修正了原配置中 (256,258) 的笔误。
"""

import argparse
import importlib.util
import json
import math
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = TESTS_DIR / "test_cases" / "perf"
DTYPE = "fp16"
RUNS = 5

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

S2_ALL = [128, 256, 512, 2048, 8192, 32768, 131072]
S2_FROM256 = [256, 512, 2048, 8192, 32768, 131072]
S2_NOT_LAST = [128, 256, 512, 2048, 8192, 32768]
S2_FROM256_NOT_LAST = [256, 512, 2048, 8192, 32768]
S2_LONG = [8192, 32768, 131072]
N2_N1_BOTH = [(16, 16), (4, 64)]
LAYOUTS = ["BSND", "BNSD"]
D_64_72_128 = [(64, 64), (72, 72), (128, 128)]
D_64_128 = [(64, 64), (128, 128)]
D_64_128_256 = [(64, 64), (128, 128), (256, 256)]
D_256 = [(256, 256)]
D_192 = [(192, 128)]

CASE_GROUPS = [
    {
        "D": D_64_72_128,
        "mask_mode": [0],
        "N2_N1": N2_N1_BOTH,
        "layout_qkvo": LAYOUTS,
        "S1": [128],
        "S2": S2_ALL,
        "B": [32],
    },
    {
        "D": D_256,
        "mask_mode": [0],
        "N2_N1": [(4, 64)],
        "layout_qkvo": LAYOUTS,
        "S1": [128],
        "S2": S2_ALL,
        "B": [32],
    },
    {
        "D": D_256,
        "mask_mode": [0],
        "N2_N1": [(16, 16)],
        "layout_qkvo": ["BNSD"],
        "S1": [128],
        "S2": S2_ALL,
        "B": [32],
    },
    {
        "D": D_256,
        "mask_mode": [0],
        "N2_N1": [(16, 16)],
        "layout_qkvo": ["BSND"],
        "S1": [128],
        "S2": S2_NOT_LAST,
        "B": [32],
    },
    {
        "D": D_64_72_128,
        "mask_mode": [0],
        "N2_N1": N2_N1_BOTH,
        "layout_qkvo": LAYOUTS,
        "S1": [256],
        "S2": S2_FROM256,
        "B": [32],
    },
    {
        "D": D_256,
        "mask_mode": [0],
        "N2_N1": [(4, 64)],
        "layout_qkvo": LAYOUTS,
        "S1": [256],
        "S2": S2_FROM256,
        "B": [32],
    },
    {
        "D": D_256,
        "mask_mode": [0],
        "N2_N1": [(16, 16)],
        "layout_qkvo": ["BNSD"],
        "S1": [256],
        "S2": S2_FROM256_NOT_LAST,
        "B": [32],
    },
    {
        "D": D_256,
        "mask_mode": [0],
        "N2_N1": [(16, 16)],
        "layout_qkvo": ["BSND"],
        "S1": [256],
        "S2": S2_FROM256_NOT_LAST,
        "B": [32],
    },
    {
        "D": D_64_72_128,
        "mask_mode": [0],
        "N2_N1": N2_N1_BOTH,
        "layout_qkvo": LAYOUTS,
        "S1": [8192],
        "S2": S2_LONG,
        "B": [32],
    },
    {
        "D": D_256,
        "mask_mode": [0],
        "N2_N1": [(4, 64)],
        "layout_qkvo": ["BNSD"],
        "S1": [8192],
        "S2": [8192, 131072],
        "B": [32],
    },
    {
        "D": D_256,
        "mask_mode": [0],
        "N2_N1": [(4, 64)],
        "layout_qkvo": ["BSND"],
        "S1": [8192],
        "S2": [8192],
        "B": [32],
    },
    {
        "D": D_256,
        "mask_mode": [0],
        "N2_N1": [(16, 16)],
        "layout_qkvo": ["BNSD"],
        "S1": [8192],
        "S2": [8192, 131072],
        "B": [32],
    },
    {
        "D": D_256,
        "mask_mode": [0],
        "N2_N1": [(16, 16)],
        "layout_qkvo": ["BSND"],
        "S1": [8192],
        "S2": [8192],
        "B": [32],
    },
    {
        "D": D_64_128,
        "mask_mode": [3],
        "N2_N1": N2_N1_BOTH,
        "layout_qkvo": LAYOUTS,
        "S1": [128],
        "S2": S2_ALL,
        "B": [32],
    },
    {
        "D": D_256,
        "mask_mode": [3],
        "N2_N1": [(4, 64)],
        "layout_qkvo": LAYOUTS,
        "S1": [128],
        "S2": S2_ALL,
        "B": [32],
    },
    {
        "D": D_256,
        "mask_mode": [3],
        "N2_N1": [(16, 16)],
        "layout_qkvo": ["BNSD"],
        "S1": [128],
        "S2": S2_ALL,
        "B": [32],
    },
    {
        "D": D_256,
        "mask_mode": [3],
        "N2_N1": [(16, 16)],
        "layout_qkvo": ["BSND"],
        "S1": [128],
        "S2": S2_NOT_LAST,
        "B": [32],
    },
    {
        "D": D_64_128,
        "mask_mode": [3],
        "N2_N1": N2_N1_BOTH,
        "layout_qkvo": LAYOUTS,
        "S1": [256],
        "S2": S2_FROM256,
        "B": [32],
    },
    {
        "D": D_256,
        "mask_mode": [3],
        "N2_N1": [(4, 64)],
        "layout_qkvo": LAYOUTS,
        "S1": [256],
        "S2": S2_FROM256,
        "B": [32],
    },
    {
        "D": D_256,
        "mask_mode": [3],
        "N2_N1": [(16, 16)],
        "layout_qkvo": ["BNSD"],
        "S1": [256],
        "S2": S2_FROM256_NOT_LAST,
        "B": [32],
    },
    {
        "D": D_256,
        "mask_mode": [3],
        "N2_N1": [(16, 16)],
        "layout_qkvo": ["BSND"],
        "S1": [256],
        "S2": S2_FROM256_NOT_LAST,
        "B": [32],
    },
    {
        "D": D_64_128,
        "mask_mode": [3],
        "N2_N1": N2_N1_BOTH,
        "layout_qkvo": LAYOUTS,
        "S1": [8192],
        "S2": S2_LONG,
        "B": [32],
    },
    {
        "D": D_256,
        "mask_mode": [3],
        "N2_N1": [(4, 64)],
        "layout_qkvo": ["BNSD"],
        "S1": [8192],
        "S2": [8192, 131072],
        "B": [32],
    },
    {
        "D": D_256,
        "mask_mode": [3],
        "N2_N1": [(4, 64)],
        "layout_qkvo": ["BSND"],
        "S1": [8192],
        "S2": [8192],
        "B": [32],
    },
    {
        "D": D_256,
        "mask_mode": [3],
        "N2_N1": [(16, 16)],
        "layout_qkvo": ["BNSD"],
        "S1": [8192],
        "S2": [8192, 131072],
        "B": [32],
    },
    {
        "D": D_256,
        "mask_mode": [3],
        "N2_N1": [(16, 16)],
        "layout_qkvo": ["BSND"],
        "S1": [8192],
        "S2": [8192],
        "B": [32],
    },
    {
        "D": D_192,
        "mask_mode": [3],
        "N2_N1": [(16, 16)],
        "layout_qkvo": LAYOUTS,
        "S1": [128],
        "S2": S2_ALL,
        "B": [32],
    },
    {
        "D": D_192,
        "mask_mode": [3],
        "N2_N1": [(16, 16)],
        "layout_qkvo": LAYOUTS,
        "S1": [256],
        "S2": S2_FROM256,
        "B": [32],
    },
    {
        "D": D_192,
        "mask_mode": [3],
        "N2_N1": [(16, 16)],
        "layout_qkvo": LAYOUTS,
        "S1": [8192],
        "S2": S2_LONG,
        "B": [32],
    },
    {
        "D": [(128, 128)],
        "mask_mode": [3],
        "N2_N1": N2_N1_BOTH,
        "layout_qkvo": LAYOUTS,
        "S1": [1, 3, 8],
        "S2": S2_ALL,
        "B": [32],
    },
    {
        "D": D_256,
        "mask_mode": [3],
        "N2_N1": [(4, 64)],
        "layout_qkvo": LAYOUTS,
        "S1": [1, 3, 8],
        "S2": S2_ALL,
        "B": [32],
    },
    {
        "D": D_256,
        "mask_mode": [3],
        "N2_N1": [(16, 16)],
        "layout_qkvo": LAYOUTS,
        "S1": [1, 3, 8],
        "S2": S2_NOT_LAST,
        "B": [32],
    },
    {
        "D": D_64_128_256,
        "mask_mode": [4],
        "N2_N1": N2_N1_BOTH,
        "layout_qkvo": LAYOUTS,
        "S1": [128, 256, 8192],
        "S2": [8192],
        "B": [32],
        "win_left_right": [(127, 0), (511, 0)],
    },
    # ---- 全块皆尾（S2 = actS2 < tileCols）：每个 s2 循环只有 1 个块且它就是尾块 ----
    # B=512 / N1=16 / N2=4（G=4）/ S1=512 → 32768 个尾块，kernel 耗时 >100us（避免固定开销主导），
    # 用来隔离 softmax VF 的"按 actS2 裁剪列数"收益；每组各带一条满块(S2==tileCols)对照。
    {
        "D": [(128, 128)],  # tileCols=256 → S2<256 全尾块
        "mask_mode": [3],
        "N2_N1": [(4, 16)],
        "layout_qkvo": ["BSND"],
        "S1": [512],
        "S2": [32, 64, 128, 224, 256],
        "B": [512],
    },
    {
        "D": [(128, 128)],
        "mask_mode": [0],
        "N2_N1": [(4, 16)],
        "layout_qkvo": ["BSND"],
        "S1": [512],
        "S2": [224],
        "B": [512],
    },
    {
        "D": [(64, 64)],  # tileCols=128（maxSeqKv<1024）→ S2<128 全尾块
        "mask_mode": [3],
        "N2_N1": [(4, 16)],
        "layout_qkvo": ["BSND"],
        "S1": [512],
        "S2": [32, 64, 96, 127, 128],
        "B": [512],
    },
]


def _name(d, dv, mask, n1, n2, layout, b, s1, s2, win):
    suffix = f"_W{win[0]}_{win[1]}" if win else ""
    return (
        f"Perf_Base_D{d}_DV{dv}_M{mask}_QN{n1}_KVN{n2}_{layout}"
        f"_B{b}_QS{s1}_KVS{s2}{suffix}_{DTYPE}"
    )


def _case(d, dv, mask, n1, n2, layout, b, s1, s2, win):
    case = {
        "B": [b],
        "N1": [n1],
        "N2": [n2],
        "S1": [s1],
        "S2": [s2],
        "D": [d],
        "layout_q": [layout],
        "layout_kv": [layout],
        "layout_out": [layout],
        "Dtype": [DTYPE],
        "q_range": [(-1, 1)],
        "k_range": [(-1, 1)],
        "v_range": [(-1, 1)],
        "scale": [1.0 / math.sqrt(d)],
        "mask_mode": [mask],
    }
    if dv != d:
        case["DV"] = [dv]
    if win:
        case["win_left"] = [win[0]]
        case["win_right"] = [win[1]]
    return case


def gen_base_cases():
    cases = {}
    meta = {}

    for group in CASE_GROUPS:
        for d, dv in group["D"]:
            for mask in group["mask_mode"]:
                for n2, n1 in group["N2_N1"]:
                    for layout in group["layout_qkvo"]:
                        for b in group["B"]:
                            for s1 in group["S1"]:
                                for s2 in group["S2"]:
                                    if mask == 4:
                                        windows = group.get("win_left_right")
                                        if not windows:
                                            raise ValueError(
                                                f"mask4 requires win_left_right: D={d}"
                                            )
                                    else:
                                        windows = [None]
                                    for win in windows:
                                        name = _name(
                                            d,
                                            dv,
                                            mask,
                                            n1,
                                            n2,
                                            layout,
                                            b,
                                            s1,
                                            s2,
                                            win,
                                        )
                                        if name in cases:
                                            raise ValueError(f"duplicate case: {name}")
                                        cases[name] = _case(
                                            d,
                                            dv,
                                            mask,
                                            n1,
                                            n2,
                                            layout,
                                            b,
                                            s1,
                                            s2,
                                            win,
                                        )
                                        meta[name] = {
                                            "layer": "Perf_Base",
                                            "group": f"D{d}_DV{dv}",
                                            "phase": "decode"
                                            if s1 <= 16
                                            else "prefill",
                                            "label": f"V2 D={d}/{dv} M={mask} N1={n1} N2={n2}",
                                            "runs": RUNS,
                                            "golden_cost": b
                                            * n1
                                            * s1
                                            * s2
                                            * dv
                                            // 1000,
                                            "D": d,
                                            "DV": dv,
                                            "mask_mode": mask,
                                            "G": n1 // n2,
                                            "N1": n1,
                                            "N2": n2,
                                            "B": b,
                                            "S1": s1,
                                            "S2": s2,
                                            "layout": layout,
                                        }
                                        if win:
                                            meta[name]["win_left"] = win[0]
                                            meta[name]["win_right"] = win[1]
    return cases, meta


def _render(cases):
    lines = [
        *LICENSE_HEADER.splitlines(),
        '"""自动生成，请勿手改；重新生成: python3 tools/perf/gen_perf_base_cases.py"""',
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


def _check(out_dir):
    expected, expected_meta = gen_base_cases()
    path = out_dir / "base_generated.py"
    spec = importlib.util.spec_from_file_location("base_generated", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cases = mod.TestCases
    manifest = json.loads((out_dir / "manifest.json").read_text())

    # 以生成器为准做一致性校验（不写死条数）
    assert set(cases) == set(expected), "base_generated.py 与生成器不一致（用例集合）"
    for name, case in expected.items():
        assert cases[name] == case, f"base_generated.py 与生成器不一致: {name}"
    assert set(manifest["cases"]) == set(expected), (
        "manifest.json 与生成器不一致（用例集合）"
    )
    assert manifest["count"] == len(expected), "manifest.json 的 count 与生成器不一致"
    for name, meta in expected_meta.items():
        assert manifest["cases"][name] == meta, f"manifest.json 与生成器不一致: {name}"

    # 生成规则自身的语义约束
    for name, case in cases.items():
        mask = case["mask_mode"][0]
        if mask == 4:
            assert "win_left" in case and "win_right" in case, name
            assert "win_left" in manifest["cases"][name], name
        else:
            assert "win_left" not in case and "win_right" not in case, name
            assert "win_left" not in manifest["cases"][name], name
        d = case["D"][0]
        if "DV" in case:
            assert case["DV"][0] != d
        if d == 256:
            assert case["D"][0] == 256 and case.get("DV", [256])[0] == 256
    print(f"OK: {len(cases)} base cases")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    if args.check:
        _check(args.out_dir)
        return

    cases, meta = gen_base_cases()
    if args.dry_run:
        for name in cases:
            print(name)
        print(f"total {len(cases)}")
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "base_generated.py").write_text(_render(cases))
    manifest = {
        "layer": "Perf_Base",
        "dtype": DTYPE,
        "count": len(cases),
        "cases": meta,
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    print(f"generated {len(cases)} base cases -> {args.out_dir}")


if __name__ == "__main__":
    main()
