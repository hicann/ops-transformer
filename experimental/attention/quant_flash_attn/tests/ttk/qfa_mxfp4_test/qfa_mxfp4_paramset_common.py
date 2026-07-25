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

# 参数顺序与 wrapper 函数签名保持一致 (除 tensor 占位外, 全部 scalar/list 都列出)
PARAM_NAMES = [
    "B",
    "N_q",
    "N_kv",
    "G",
    "D",
    "V_D",
    "Rope_D",
    "act_seq_lens_q",
    "act_seq_lens_kv",
    "input_layout",
    "layout_q_descale",
    "kv_storage_mode",
    "block_size",
    "q_dtype",
    "kv_dtype",
    "out_dtype",
    "q_quant_mode",
    "mask_mode",
    "pre_tokens",
    "next_tokens",
    "enable_mask",
    "enable_lse",
    "inner_precise",
    "device_id",
    "graph_path",
    "softmax_scale",
    "data_range_q",
    "data_range_k",
    "data_range_v",
]

TEST_PARAMS_DEFAULTS = {
    "V_D": [None],  # 默认等于 D, gen_csv 时 fill
    "Rope_D": [0],  # mxfp4 不支持 rope, 固定 0
    "layout_q_descale": ["BSND"],
    "kv_storage_mode": ["continue"],
    "block_size": [0],
    "q_dtype": ["fp4_e2m1"],
    "kv_dtype": ["fp4_e2m1"],
    "out_dtype": ["bfloat16"],
    "q_quant_mode": [3],  # MXFP4 固定 3
    "mask_mode": [0],
    "pre_tokens": [2147483647],
    "next_tokens": [2147483647],
    "enable_mask": [False],
    "enable_lse": [False],
    "inner_precise": [0],
    "device_id": [0],
    "graph_path": [0],
    "softmax_scale": [None],
    "data_range_q": [1.0],
    "data_range_k": [1.0],
    "data_range_v": [1.0],
}


def expand_paramset_to_cases(test_params):
    """把 TEST_PARAMS 展开为 case 列表. 默认值缺失时用 TEST_PARAMS_DEFAULTS 填充."""
    cases = []
    for name, params in test_params.items():
        expanded = dict(params)
        for key, default_vals in TEST_PARAMS_DEFAULTS.items():
            if key not in expanded:
                expanded[key] = default_vals
        # V_D 默认等于 D
        if expanded.get("V_D") == [None]:
            expanded["V_D"] = [expanded["D"][0]]
        param_values = [expanded[n] for n in PARAM_NAMES]
        for combo in itertools.product(*param_values):
            case = {"name": name}
            for key, val in zip(PARAM_NAMES, combo):
                case[key] = val
            cases.append(case)
    return cases
