#!/usr/bin/python
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

import os
from copy import deepcopy

import torch

# 当前由于torchair拦截的原因，kv_type为float8_e4m3fn时只能传None
TEST_PARAMS = {
    "TND_PA": {
        "Testcase_Prefix": ["runpy_style"],
        "layout_query": ["TND"],
        "layout_kv": ["PA_BSND"],
        "q_type": [torch.bfloat16],
        "kv_type": ["hifloat8"],
        "B": [22],
        "T1": [22],
        "S1": [1],
        "S2": [128],
        "N1": [48],
        "N2": [1],
        "D": [512],
        "K": [2048],
        "scale_value": [0.041666666666666664],
        "key_quant_mode": [2],
        "value_quant_mode": [2],
        "sparse_block_size": [1],
        "tile_size": [128],
        "rope_head_dim": [64],
        "sparse_mode": [3],
        "attention_mode": [2],
        "quant_scale_repo_mode": [1],
        "block_size": [128],
        "return_softmax_lse": [True],
        "actual_seq_q": [
            [
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
                22,
            ]
        ],
        "actual_seq_kv": [
            [
                12,
                12,
                12,
                12,
                12,
                12,
                12,
                12,
                12,
                12,
                12,
                12,
                12,
                12,
                12,
                12,
                12,
                12,
                12,
                12,
                12,
                128,
            ]
        ],
    },
    "BSND_BSND": {
        "Testcase_Prefix": ["runpy_style"],
        "layout_query": ["BSND"],
        "layout_kv": ["BSND"],
        "q_type": [torch.bfloat16],
        "kv_type": [None],
        "B": [1],
        "S1": [1],
        "S2": [4096],
        "N1": [64],
        "N2": [1],
        "D": [512],
        "K": [2048],
        "scale_value": [0.041666666666666664],
        "key_quant_mode": [2],
        "value_quant_mode": [2],
        "sparse_block_size": [1],
        "tile_size": [128],
        "rope_head_dim": [64],
        "sparse_mode": [3],
        "attention_mode": [2],
        "quant_scale_repo_mode": [1],
        "block_size": [256],
        "return_softmax_lse": [True],
        "actual_seq_q": [[1]],
        "actual_seq_kv": [[4096]],
    },
    "BSND_PA": {
        "Testcase_Prefix": ["runpy_style"],
        "layout_query": ["BSND"],
        "layout_kv": ["PA_BSND"],
        "q_type": [torch.bfloat16],
        "kv_type": [None],
        "B": [18],
        "T1": [54],
        "S1": [3],
        "S2": [3072],
        "N1": [64],
        "N2": [1],
        "D": [512],
        "K": [2048],
        "scale_value": [0.041666666666666664],
        "key_quant_mode": [2],
        "value_quant_mode": [2],
        "sparse_block_size": [1],
        "tile_size": [128],
        "rope_head_dim": [64],
        "sparse_mode": [3],
        "attention_mode": [2],
        "quant_scale_repo_mode": [1],
        "block_size": [16],
        "return_softmax_lse": [True],
        "actual_seq_q": [[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]],
        "actual_seq_kv": [
            [
                2208,
                2112,
                800,
                2112,
                2720,
                2080,
                1248,
                2816,
                2272,
                2976,
                3008,
                2784,
                2272,
                1696,
                2496,
                2144,
                2432,
                3072,
            ]
        ],
        "enable_sinks": [False],
    },
    "TND_TND": {
        "Testcase_Prefix": ["runpy_style"],
        "layout_query": ["TND"],
        "layout_kv": ["TND"],
        "q_type": [torch.bfloat16],
        "kv_type": [None],
        "B": [22],
        "T1": [22],
        "T2": [128],
        "S1": [1],
        "S2": [128],
        "N1": [48],
        "N2": [1],
        "D": [512],
        "K": [2048],
        "scale_value": [0.041666666666666664],
        "key_quant_mode": [2],
        "value_quant_mode": [2],
        "sparse_block_size": [1],
        "tile_size": [128],
        "rope_head_dim": [64],
        "sparse_mode": [3],
        "attention_mode": [2],
        "quant_scale_repo_mode": [1],
        "block_size": [128],
        "return_softmax_lse": [True],
        "actual_seq_q": [
            [
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
                22,
            ]
        ],
        "actual_seq_kv": [
            [
                12,
                12,
                12,
                12,
                12,
                12,
                12,
                12,
                12,
                12,
                12,
                12,
                12,
                12,
                12,
                12,
                12,
                12,
                12,
                12,
                12,
                128,
            ]
        ],
        "enable_sinks": [True],
    },
}

ENABLED_PARAMS = [TEST_PARAMS[key] for key in TEST_PARAMS.keys()]


def _make_910b_case(base, name, heads, **overrides):
    case = deepcopy(TEST_PARAMS[base])
    case.update(
        {
            "Testcase_Prefix": [name],
            "kv_type": ["int8"],
            "S2": [128],
            "K": [128],
            "N1": [heads],
            "actual_seq_kv": [[128]],
            "return_softmax_lse": [True],
        }
    )
    case.update({key: [value] for key, value in overrides.items()})
    return case


PARAMS_910B = [
    _make_910b_case("BSND_BSND", "qsfa_v2_910b_bsnd_8h", 8),
    _make_910b_case("BSND_BSND", "qsfa_v2_910b_bsnd_96h", 96),
    _make_910b_case(
        "TND_PA",
        "qsfa_v2_910b_tnd_pa_96h",
        96,
        B=1,
        T1=2,
        S1=2,
        block_size=16,
        actual_seq_q=[2],
    ),
    _make_910b_case(
        "BSND_PA",
        "qsfa_v2_910b_bsnd_pa_64h",
        64,
        B=2,
        T1=4,
        S1=2,
        block_size=16,
        actual_seq_q=[2, 2],
        actual_seq_kv=[128, 96],
    ),
    _make_910b_case(
        "TND_TND",
        "qsfa_v2_910b_tnd_64h_sinks",
        64,
        B=2,
        T1=2,
        T2=128,
        S1=1,
        S2=64,
        actual_seq_q=[1, 2],
        actual_seq_kv=[64, 128],
    ),
    _make_910b_case(
        "BSND_BSND",
        "qsfa_v2_910b_bsnd_96h_no_lse",
        96,
        return_softmax_lse=False,
    ),
]


def get_model_params(arch=None):
    """Return the unchanged A2/A5 model profile without changing default cases."""
    arch = (arch or os.environ.get("QSFA_MODEL_ARCH", "A2")).upper()
    if arch not in ("A2", "A5"):
        raise ValueError(f"QSFA_MODEL_ARCH must be A2 or A5, got {arch}")
    model_case = deepcopy(TEST_PARAMS["TND_PA"])
    model_case.update(
        {
            "Testcase_Prefix": [f"qsfa_v2_{arch.lower()}_model_8x128"],
            "kv_type": ["int8" if arch == "A2" else None],
            "B": [8],
            "T1": [8],
            "S1": [1],
            "S2": [32768],
            "N1": [128],
            "K": [2048],
            "block_size": [128],
            "block_num": [7644],
            "actual_seq_q": [[1, 2, 3, 4, 5, 6, 7, 8]],
            "actual_seq_kv": [[32768] * 8],
            "return_softmax_lse": [True],
        }
    )
    return [model_case]


def get_enabled_params(profile="default", arch=None):
    """Select default, 910b or model cases from this parameter module."""
    if profile == "default":
        return deepcopy(ENABLED_PARAMS)
    if profile == "910b":
        return deepcopy(PARAMS_910B)
    if profile == "model":
        return get_model_params(arch)
    raise ValueError(f"Unknown QSFA V2 parameter profile: {profile}")
