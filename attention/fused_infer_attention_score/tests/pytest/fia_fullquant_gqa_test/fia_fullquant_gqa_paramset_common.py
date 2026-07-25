#!/usr/bin/python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import itertools

PARAM_NAMES = [
    "B",
    "N_q",
    "N_kv",
    "D",
    "actual_seq_q",
    "actual_seq_kv",
    "enable_pa",
    "enable_lse",
    "golden_mode",
    "block_size",
    "sparse_mode",
    "input_layout",
    "output_layout",
    "q_scale_layout",
    "kv_cache_layout",
    "p_scale",
    "scale_value",
    "is_contiguous",
    "num_blocks",
    "graph_path",
    "device_id",
    "q_data_range",
    "k_data_range",
    "v_data_range",
    "seed_q",
    "seed_k",
    "seed_v",
    "seed_block_table",
]

TEST_PARAMS_DEFAULTS = {
    "D": [128],
    "enable_pa": [True],
    "enable_lse": [True],
    "golden_mode": [True],
    "block_size": [128],
    "sparse_mode": [3],
    "input_layout": ["NTD_TND"],
    "output_layout": ["TND"],
    "q_scale_layout": ["NT"],
    "kv_cache_layout": ["BnNBsD"],
    "p_scale": [1.0],
    "scale_value": [None],
    "is_contiguous": [True],
    "num_blocks": [0],
    "graph_path": [0],
    "device_id": [0],
    "q_data_range": [(-1.0, 1.0)],
    "k_data_range": [(-1.0, 1.0)],
    "v_data_range": [(-1.0, 1.0)],
    "seed_q": [54],
    "seed_k": [3],
    "seed_v": [20],
    "seed_block_table": [1234],
}


def expand_paramset_to_cases(test_params):
    cases = []
    for name, params in test_params.items():
        expanded = dict(params)
        for key, default_vals in TEST_PARAMS_DEFAULTS.items():
            if key not in expanded:
                expanded[key] = default_vals
        param_values = [expanded[n] for n in PARAM_NAMES]
        for combo in itertools.product(*param_values):
            case = {"name": name}
            for key, val in zip(PARAM_NAMES, combo):
                case[key] = val
            cases.append(case)
    return cases
