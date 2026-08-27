# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import torch
import itertools

BASE_CONFIG = {
    "nk": [128],
    "nv": [128],
    "actual_seq_lengths": [None],
    "ssm_state_indices": [None],
    "has_gamma": ["True"],
    "has_gamma_k": ["True"],
    "has_num_accepted_tokens": ["True"],
    "scale_value": [None],
    "num_accepted_tokens": [None],
    "block_num": [None],
    "data_type": [torch.bfloat16],
    "state_data_type": [torch.bfloat16],
    "query_datarange": [[-1, 1]],
    "key_datarange": [[-1, 1]],
    "value_datarange": [[-10, 10]],
    "gamma_datarange": [[-1, 0]],
    "gamma_k_datarange": [[-1, 0]],
    "beta_datarange": [[0, 1]],
    "state_datarange": [[-10, 10]],
    "state_non_contiguous": [False],
}


def generate_test_cases(batch_sizes, mtps, dk, dv, **kwargs):
    config = BASE_CONFIG.copy()
    config.update(kwargs)
    return [
        {
            "batch_size": [bs],
            "mtp": [mtp],
            "nk": config["nk"],
            "nv": config["nv"],
            "dk": [dk],
            "dv": [dv],
            "actual_seq_lengths": config["actual_seq_lengths"],
            "ssm_state_indices": config["ssm_state_indices"],
            "has_gamma": config["has_gamma"],
            "has_gamma_k": config["has_gamma_k"],
            "has_num_accepted_tokens": config["has_num_accepted_tokens"],
            "scale_value": config["scale_value"],
            "num_accepted_tokens": config["num_accepted_tokens"],
            "block_num": config["block_num"],
            "data_type": config["data_type"],
            "state_data_type": config["state_data_type"],
            "query_datarange": config["query_datarange"],
            "key_datarange": config["key_datarange"],
            "value_datarange": config["value_datarange"],
            "gamma_datarange": config["gamma_datarange"],
            "gamma_k_datarange": config["gamma_k_datarange"],
            "beta_datarange": config["beta_datarange"],
            "state_datarange": config["state_datarange"],
            "state_non_contiguous": config["state_non_contiguous"],
        }
        for bs, mtp in itertools.product(batch_sizes, mtps)
    ]


SPECIAL_CASES = [
    {
        "batch_size": [64],
        "mtp": [2],
        "nk": [32],
        "nv": [32],
        "dk": [128],
        "dv": [128],
        "actual_seq_lengths": [None],
        "ssm_state_indices": [None],
        "has_gamma": ["True"],
        "has_gamma_k": ["True"],
        "has_num_accepted_tokens": ["True"],
        "scale_value": [0.08838834],
        "num_accepted_tokens": [None],
        "block_num": [None],
        "data_type": [torch.bfloat16],
        "state_data_type": [torch.bfloat16],
        "query_datarange": [[-1, 1]],
        "key_datarange": [[-1, 1]],
        "value_datarange": [[-10, 10]],
        "gamma_datarange": [[-1, 0]],
        "gamma_k_datarange": [[-1, 0]],
        "beta_datarange": [[0, 1]],
        "state_datarange": [[-10, 10]],
        "state_non_contiguous": [False],
    },
    {
        "batch_size": [1],
        "mtp": [1],
        "nk": [65],
        "nv": [65],
        "dk": [1],
        "dv": [1],
        "actual_seq_lengths": [None],
        "ssm_state_indices": [None],
        "has_gamma": ["True"],
        "has_gamma_k": ["True"],
        "has_num_accepted_tokens": ["True"],
        "scale_value": [None],
        "num_accepted_tokens": [None],
        "block_num": [None],
        "data_type": [torch.bfloat16],
        "state_data_type": [torch.bfloat16],
        "query_datarange": [[-1, 1]],
        "key_datarange": [[-1, 1]],
        "value_datarange": [[-10, 10]],
        "gamma_datarange": [[-1, 0]],
        "gamma_k_datarange": [[-1, 0]],
        "beta_datarange": [[0, 1]],
        "state_datarange": [[-10, 10]],
        "state_non_contiguous": [False],
    },
]

GROUP_1 = SPECIAL_CASES

GROUP_2 = generate_test_cases(
    batch_sizes=[1, 2, 4, 8, 16, 32, 64], mtps=[1, 2], dk=4, dv=8
)

GROUP_3 = generate_test_cases(
    batch_sizes=[1, 2, 4, 8, 16, 32, 64], mtps=[1, 2], dk=16, dv=32
)

GROUP_4 = generate_test_cases(batch_sizes=[2, 4, 8, 16, 32], mtps=[1], dk=8, dv=16)

GROUP_5 = [
    {
        "batch_size": [64],
        "mtp": [8],
        "nk": [128],
        "nv": [128],
        "dk": [4],
        "dv": [8],
        "actual_seq_lengths": [None],
        "ssm_state_indices": [None],
        "has_gamma": ["True"],
        "has_gamma_k": ["True"],
        "has_num_accepted_tokens": ["True"],
        "scale_value": [None],
        "num_accepted_tokens": [None],
        "block_num": [None],
        "data_type": [torch.bfloat16],
        "state_data_type": [torch.bfloat16],
        "query_datarange": [[-1, 1]],
        "key_datarange": [[-1, 1]],
        "value_datarange": [[-10, 10]],
        "gamma_datarange": [[-1, 0]],
        "gamma_k_datarange": [[-1, 0]],
        "beta_datarange": [[0, 1]],
        "state_datarange": [[-10, 10]],
        "state_non_contiguous": [False],
    }
]

GROUP_6 = [
    {
        "batch_size": [64],
        "mtp": [2],
        "nk": [128],
        "nv": [128],
        "dk": [256],
        "dv": [256],
        "actual_seq_lengths": [None],
        "ssm_state_indices": [None],
        "has_gamma": ["True"],
        "has_gamma_k": ["True"],
        "has_num_accepted_tokens": ["True"],
        "scale_value": [None],
        "num_accepted_tokens": [None],
        "block_num": [None],
        "data_type": [torch.bfloat16],
        "state_data_type": [torch.bfloat16],
        "query_datarange": [[-1, 1]],
        "key_datarange": [[-1, 1]],
        "value_datarange": [[-10, 10]],
        "gamma_datarange": [[-1, 0]],
        "gamma_k_datarange": [[-1, 0]],
        "beta_datarange": [[0, 1]],
        "state_datarange": [[-10, 10]],
        "state_non_contiguous": [False],
    }
]


L0_L1_GAMMA_KWARGS = {
    "gamma_datarange": [[-10, 0]],
    "gamma_k_datarange": [[-10, 0]],
}

L0_L1_GROUP_1 = generate_test_cases(
    batch_sizes=[1, 4, 8, 32, 128, 256],
    mtps=[2],
    dk=128,
    dv=128,
    nk=[4],
    nv=[8],
    **L0_L1_GAMMA_KWARGS,
)

L0_L1_GROUP_2 = generate_test_cases(
    batch_sizes=[64],
    mtps=[1, 3, 4, 8],
    dk=128,
    dv=128,
    nk=[4],
    nv=[8],
    **L0_L1_GAMMA_KWARGS,
)

L0_L1_GROUP_3 = [
    c
    for nk, nv in [
        (1, 1),
        (1, 2),
        (1, 4),
        (2, 2),
        (2, 4),
        (2, 8),
        (4, 4),
        (4, 8),
        (4, 16),
        (8, 8),
        (8, 16),
        (8, 32),
        (16, 16),
        (16, 32),
        (16, 64),
        (32, 32),
        (32, 64),
        (32, 256),
    ]
    for c in generate_test_cases(
        batch_sizes=[64],
        mtps=[2],
        dk=128,
        dv=128,
        nk=[nk],
        nv=[nv],
        **L0_L1_GAMMA_KWARGS,
    )
]

L0_L1_GROUP_4 = [
    c
    for dk, dv in [(32, 32), (64, 64), (256, 256), (512, 512)]
    for c in generate_test_cases(
        batch_sizes=[64], mtps=[2], dk=dk, dv=dv, nk=[4], nv=[8], **L0_L1_GAMMA_KWARGS
    )
]

L0_L1_CASES = L0_L1_GROUP_1 + L0_L1_GROUP_2 + L0_L1_GROUP_3 + L0_L1_GROUP_4


def _gen_fp32_state_cases(cases):
    result = []
    for c in cases:
        c2 = dict(c)
        c2["state_data_type"] = [torch.float32]
        result.append(c2)
    return result


def _gen_non_contiguous_state_cases(cases):
    result = []
    for c in cases:
        c2 = dict(c)
        c2["state_non_contiguous"] = [True]
        result.append(c2)
    return result


_ALL_BF16_CASES = (
    GROUP_1 + GROUP_2 + GROUP_3 + GROUP_4 + GROUP_5 + GROUP_6 + L0_L1_CASES
)

ENABLED_PARAMS_RDV = (
    _ALL_BF16_CASES
    + _gen_fp32_state_cases(_ALL_BF16_CASES)
    + _gen_non_contiguous_state_cases(_ALL_BF16_CASES)
    + _gen_non_contiguous_state_cases(_gen_fp32_state_cases(_ALL_BF16_CASES))
)
