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


"""MLA pytest paramset 公共展开逻辑（对齐 mxfp8 的 quant_flash_attn_paramset_common.py）。

PARAM_NAMES 定义 MLA 全部合法参数键；
TEST_PARAMS_DEFAULTS 提供缺失字段的默认值；
expand_paramset_to_cases 用 itertools.product 把 TEST_PARAMS 展开成扁平 case list。
"""

import itertools

PARAM_NAMES = [
    "B",
    "N_q",
    "N_kv",
    "D",
    "D_V",
    "D_rope",
    "cu_seqlens_q",
    "cache_seqlens",
    "seqused_q",
    "max_seqlen_q",
    "max_seqlen_kv",
    "enable_pa",
    "kv_cache_layout",
    "block_size",
    "mask_mode",
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
    "D_V": [512],
    "D_rope": [64],
    "cache_seqlens": [None],
    "softmax_scale": [None],
    "max_seqlen_q": [-1],
    "max_seqlen_kv": [-1],
    "data_range_q": [1.0],
    "data_range_k": [1.0],
    "data_range_v": [1.0],
    "enable_lse": [False],
    "p_scale": [1.0],
    "quant_mode": [1],
    "device_id": [0],
    "is_contiguous": [True],
}


def expand_paramset_to_cases(test_params):
    cases = []
    for name, params in test_params.items():
        expanded = dict(params)
        for key, default_vals in TEST_PARAMS_DEFAULTS.items():
            if key not in expanded:
                expanded[key] = default_vals
        param_values = [expanded.get(n, [None]) for n in PARAM_NAMES]
        for combo in itertools.product(*param_values):
            case = {"name": name}
            for key, val in zip(PARAM_NAMES, combo):
                case[key] = val
            cases.append(case)
    return cases
