# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Public accuracy cases, including representative former regression boundaries."""

from .quant_flash_attn_int8_paramset_common import expand_paramset_to_cases

TEST_PARAMS = {
    "Model": {
        "D": [64, 72, 128, 256],
        "layout": ["BNSD", "BSND", "TND"],
        "mask_mode": [0, 3],
        "metadata": [False, True],
        "enable_lse": [False, True],
    },
    "DimensionBoundary": {
        "D": [
            1,
            7,
            16,
            17,
            31,
            32,
            33,
            63,
            65,
            71,
            73,
            95,
            96,
            97,
            127,
            129,
            191,
            192,
            193,
            255,
        ],
        "layout": ["TND"],
        "metadata": [True],
        "enable_lse": [True],
    },
    "DenseDimensionBoundary": {
        "D": [1, 17, 33, 65, 129, 255],
        "layout": ["BNSD", "BSND"],
        "mask_mode": [3],
        "metadata": [True],
        "enable_lse": [True],
    },
    "SlidingWindow": {
        "layout": ["BNSD", "BSND", "TND"],
        "mask_mode": [0, 3],
        "win_left": [0, 37],
        "win_right": [0, 17],
        "actual_seq_q": [(65, 65)],
        "actual_seq_kv": [(33, 33)],
        "metadata": [True],
        "enable_lse": [True],
    },
    "PackedMixedEmpty": {
        "layout": ["TND"],
        "actual_seq_q": [(0, 65, 17)],
        "actual_seq_kv": [(0, 33, 129)],
        "N_q": [6],
        "N_kv": [1],
        "mask_mode": [0, 3],
        "metadata": [False, True],
        "enable_lse": [True],
    },
    "OneSidedWindow": {
        "layout": ["BNSD", "BSND", "TND"],
        "mask_mode": [0, 3],
        "actual_seq_q": [(65, 65)],
        "actual_seq_kv": [(33, 33)],
        "win_left": [-1],
        "win_right": [31],
        "metadata": [True],
        "enable_lse": [True],
    },
    "UnboundedRightWindow": {
        "layout": ["BNSD", "BSND", "TND"],
        "mask_mode": [0, 3],
        "actual_seq_q": [(65, 65)],
        "actual_seq_kv": [(33, 33)],
        "win_left": [37],
        "win_right": [-1],
        "metadata": [True],
        "enable_lse": [True],
    },
    "MhaMaskedRows": {
        "layout": ["BNSD", "TND"],
        "N_q": [2],
        "N_kv": [2],
        "D": [64, 128, 256],
        "actual_seq_q": [(129, 129)],
        "actual_seq_kv": [(65, 65)],
        "mask_mode": [3],
        "win_left": [-1],
        "win_right": [31],
        "metadata": [True],
        "enable_lse": [True],
    },
    "TokenFragments": {
        "layout": ["BSND", "TND"],
        "N_q": [3, 6, 7],
        "N_kv": [1],
        "D": [64, 256],
        "actual_seq_q": [(129, 129)],
        "actual_seq_kv": [(193, 193)],
        "mask_mode": [3],
        "metadata": [True],
        "enable_lse": [True],
    },
    "ScaleRange": {
        "layout": ["BNSD", "BSND", "TND"],
        "input_range": [(-128, 128)],
        "q_scale_factor": [0.01],
        "p_scale": [1.0, 127.0],
        "enable_lse": [True],
    },
    "ZeroQScale": {"zero_q_scale": [True], "mask_mode": [3], "enable_lse": [True]},
}

# Each long case represents a different data dependency or output boundary;
# model head/D/layout throughput coverage remains in the performance paramset.
for index, (layout, qlens, klens, heads, kv_heads, dim, mask) in enumerate(
    [
        ("TND", (0, 17, 0, 2049), (7, 16385, 0, 40961), 6, 1, 256, 3),
        ("TND", (20480, 33, 127, 1), (20480, 129, 257, 513), 12, 2, 128, 3),
        ("TND", (33, 10240, 127), (129, 10240, 257), 16, 2, 64, 0),
        ("BNSD", (6145,) * 2, (6145,) * 2, 16, 16, 72, 3),
        ("BSND", (6145,) * 8, (6145,) * 8, 4, 4, 72, 0),
        ("BSND", (513,) * 3, (8193,) * 3, 6, 1, 128, 3),
        ("TND", (0, 513, 129), (0, 257, 529), 8, 8, 72, 3),
        ("BNSD", (17,) * 32, (33,) * 32, 4, 2, 248, 0),
    ]
):
    TEST_PARAMS[f"PrefillBoundary{index}"] = {
        "layout": [layout],
        "actual_seq_q": [qlens],
        "actual_seq_kv": [klens],
        "N_q": [heads],
        "N_kv": [kv_heads],
        "D": [dim],
        "mask_mode": [mask],
        "metadata": [False, True],
        "enable_lse": [True],
        "sample_rows": [(0, 127, 511, -1)],
        "repeat": [3],
    }
for index, (layout, kv_layout, output_layout) in enumerate(
    [
        ("BNSD", "BSND", "BSND"),
        ("BSND", "BNSD", "BNSD"),
    ]
):
    TEST_PARAMS[f"MixedLayout{index}"] = {
        "layout": [layout],
        "layout_kv": [kv_layout],
        "layout_out": [output_layout],
        "actual_seq_q": [(513, 513)],
        "actual_seq_kv": [(257, 257)],
        "mask_mode": [0, 3],
        "metadata": [True],
        "enable_lse": [True],
        "sample_rows": [(0, 127, 511, -1)],
    }

CASES = expand_paramset_to_cases(TEST_PARAMS)
SKIP_CASES = set()
