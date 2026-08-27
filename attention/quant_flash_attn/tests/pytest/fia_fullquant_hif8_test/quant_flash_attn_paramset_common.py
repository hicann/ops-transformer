# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import itertools

PARAM_NAMES = [
    "B",
    "N_q",
    "N_kv",
    "D",
    "cu_seqlens_q",
    "cu_seqlens_kv",
    "seqused_q",
    "seqused_kv",
    "max_seqlen_q",
    "max_seqlen_kv",
    "enable_pa",
    "kv_cache_layout",
    "block_size",
    "mask_mode",
    "q_scale_layout",
    "p_scale",
    "softmax_scale",
    "data_range_q",
    "data_range_k",
    "data_range_v",
    "enable_lse",
    "quant_mode",
    "device_id",
    "is_contiguous",
]

TEST_PARAMS_DEFAULTS = {
    "softmax_scale": [None],
    "max_seqlen_q": [-1],
    "max_seqlen_kv": [-1],
    "data_range_q": [1.0],
    "data_range_k": [1.0],
    "data_range_v": [1.0],
    "quant_mode": [0],
    "device_id": [0],
    "is_contiguous": [True],
    "enable_pa": [False],
    "kv_cache_layout": ["TND"],
    "block_size": [0],
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
