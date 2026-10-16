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
    "D_v",
    "D_rope",
    "seqused_q",
    "cache_seqlens",
    "max_seqlen_q",
    "max_seqlen_kv",
    "enable_pa",
    "enable_lse",
    "golden_mode",
    "block_size",
    "mask_mode",
    "input_layout",
    "output_layout",
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
    "seed_qr",
    "seed_kr",
    "seed_block_table",
]

TEST_PARAMS_DEFAULTS = {
    "D": [512],
    "D_v": [512],
    "D_rope": [64],
    "seqused_q": [[1]],
    "cache_seqlens": [[1024]],
    "max_seqlen_q": [-1],
    "max_seqlen_kv": [-1],
    "enable_pa": [True],
    "enable_lse": [True],
    "golden_mode": [True],
    "block_size": [128],
    "mask_mode": [0],
    "input_layout": ["TND"],
    "output_layout": ["TND"],
    "kv_cache_layout": ["BnNBsD"],
    "p_scale": [1.0],
    "scale_value": [None],
    "is_contiguous": [True],
    "num_blocks": [0],
    "graph_path": [0],
    "device_id": [0],
    "q_data_range": [(-1.0, 1.0)],
    "k_data_range": [(-5.0, 5.0)],
    "v_data_range": [(-5.0, 5.0)],
    "seed_q": [54],
    "seed_k": [3],
    "seed_v": [20],
    "seed_qr": [8],
    "seed_kr": [9],
    "seed_block_table": [1234],
}


def _split_layout_spec(val):
    """layout 参数格式: "输入layout_输出layout"（如 TND_BSND/TND_NTD）；
    无下划线（如 BSND/BNSD）时输入输出 layout 相同"""
    if "_" in val:
        in_layout, out_layout = val.split("_", 1)
        return in_layout, out_layout
    return val, val


def expand_paramset_to_cases(test_params):
    cases = []
    for name, params in test_params.items():
        expanded = dict(params)
        # layout 参数: 解析为 input_layout/output_layout，优先级高于二者显式配置。
        # 展开后 input_layout/output_layout 会进入笛卡尔积，需按原始配对过滤非法组合
        layout_pair_to_spec = None
        if "layout" in expanded:
            layout_vals = expanded.pop("layout")
            in_vals, out_vals = [], []
            for layout_val in layout_vals:
                in_l, out_l = _split_layout_spec(layout_val)
                in_vals.append(in_l)
                out_vals.append(out_l)
            expanded["input_layout"] = in_vals
            expanded["output_layout"] = out_vals
            layout_pair_to_spec = {
                (in_l, out_l): spec
                for spec, in_l, out_l in zip(layout_vals, in_vals, out_vals)
            }
        for key, default_vals in TEST_PARAMS_DEFAULTS.items():
            if key not in expanded:
                expanded[key] = default_vals
        # Auto-derive max_seqlen_q / max_seqlen_kv if still default (-1)
        if expanded.get("max_seqlen_q", [-1])[0] == -1:
            expanded["max_seqlen_q"] = [max(expanded["seqused_q"][0])]
        if expanded.get("max_seqlen_kv", [-1])[0] == -1:
            expanded["max_seqlen_kv"] = [max(expanded["cache_seqlens"][0])]
        param_values = [expanded[n] for n in PARAM_NAMES]
        for combo in itertools.product(*param_values):
            case = {"name": name}
            for key, val in zip(PARAM_NAMES, combo):
                case[key] = val
            # 过滤 layout 笛卡尔积产生的非法配对，保留原始 layout 配置串用于 case id
            if layout_pair_to_spec is not None:
                pair = (case["input_layout"], case["output_layout"])
                if pair not in layout_pair_to_spec:
                    continue
                case["layout"] = layout_pair_to_spec[pair]
            cases.append(case)
    return cases
