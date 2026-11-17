# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Parameter expansion shared by the functional and performance entrances."""

from itertools import product

TEST_PARAMS_DEFAULTS = {
    "N_q": [4],
    "N_kv": [2],
    "D": [128],
    "layout": ["BNSD"],
    "actual_seq_q": [(17, 17)],
    "actual_seq_kv": [(33, 33)],
    "mask_mode": [0],
    "enable_lse": [False],
    "win_left": [-1],
    "win_right": [-1],
    "p_scale": [120.0],
    "seed": [218],
    "sample_rows": [None],
    "warmup": [0],
    "repeat": [1],
}


def _length_id(lengths):
    return (
        f"{len(lengths)}x{lengths[0]}"
        if len(lengths) > 4 and len(set(lengths)) == 1
        else "x".join(map(str, lengths))
    )


def expand_paramset_to_cases(test_params):
    cases = []
    for group, parameters in test_params.items():
        parameters = {**TEST_PARAMS_DEFAULTS, **parameters}
        for values in product(*parameters.values()):
            case = dict(zip(parameters, values))
            qlens, klens = case["actual_seq_q"], case["actual_seq_kv"]
            assert len(qlens) == len(klens) and qlens
            assert case["N_q"] % case["N_kv"] == 0
            case.setdefault("softmax_scale", case["D"] ** -0.5)
            case.setdefault("layout_kv", case["layout"])
            case.setdefault("layout_out", case["layout"])
            case["name"] = (
                f"{group}_{case['layout']}_B{len(qlens)}_G{case['N_q'] // case['N_kv']}"
                f"_Nq{case['N_q']}_Nkv{case['N_kv']}_D{case['D']}"
                f"_SM{case['mask_mode']}_LSE{int(case['enable_lse'])}"
                f"_Q{_length_id(qlens)}_KV{_length_id(klens)}"
                f"_WL{case['win_left']}_WR{case['win_right']}"
            )
            if case["p_scale"] != 120.0:
                case["name"] += f"_P{case['p_scale']:g}"
            cases.append(case)
    assert len({case["name"] for case in cases}) == len(cases), "duplicate case IDs"
    return cases
