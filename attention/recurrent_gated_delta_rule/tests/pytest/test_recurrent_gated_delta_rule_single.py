# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import itertools
import torch_npu
import os
import logging
import pytest

from test_recurrent_gated_delta_rule_paramset import ENABLED_PARAMS
from test_recurrent_gated_delta_rule_paramset_rdv import ENABLED_PARAMS_RDV
import recurrent_gated_delta_rule_operator_single

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TEST_MODE = os.environ.get("TEST_MODE", "single")

if TEST_MODE not in ["single", "rdv"]:
    raise ValueError(f"Invalid TEST_MODE: {TEST_MODE}, must be 'single' or 'rdv'")

if TEST_MODE == "rdv":
    PARAM_SET = ENABLED_PARAMS_RDV
else:
    PARAM_SET = ENABLED_PARAMS

logger.info(f"TEST_MODE: {TEST_MODE}")

param_names = [
    "batch_size",
    "mtp",
    "nk",
    "nv",
    "dk",
    "dv",
    "actual_seq_lengths",
    "ssm_state_indices",
    "has_gamma",
    "has_gamma_k",
    "has_num_accepted_tokens",
    "scale_value",
    "num_accepted_tokens",
    "block_num",
    "data_type",
    "state_data_type",
    "query_datarange",
    "key_datarange",
    "value_datarange",
    "gamma_datarange",
    "gamma_k_datarange",
    "beta_datarange",
    "state_datarange",
]

param_combinations = []

for _, params in enumerate(PARAM_SET):
    param_values = [
        params["batch_size"],
        params["mtp"],
        params["nk"],
        params["nv"],
        params["dk"],
        params["dv"],
        params["actual_seq_lengths"],
        params["ssm_state_indices"],
        params["has_gamma"],
        params["has_gamma_k"],
        params["has_num_accepted_tokens"],
        params["scale_value"],
        params["num_accepted_tokens"],
        params["block_num"],
        params["data_type"],
        params["state_data_type"],
        params["query_datarange"],
        params["key_datarange"],
        params["value_datarange"],
        params["gamma_datarange"],
        params["gamma_k_datarange"],
        params["beta_datarange"],
        params["state_datarange"],
    ]

    for combo in itertools.product(*param_values):
        param_dict = dict(zip(param_names, combo))
        param_combinations.append(param_dict)

logger.info(f"Total test cases: {len(param_combinations)}")


@pytest.mark.ci
@pytest.mark.parametrize("param_combinations", param_combinations)
def test_recurrent_gated_delta_rule(param_combinations):
    batch_size = param_combinations["batch_size"]
    mtp = param_combinations["mtp"]
    nk = param_combinations["nk"]
    nv = param_combinations["nv"]
    dk = param_combinations["dk"]
    dv = param_combinations["dv"]
    actual_seq_lengths = param_combinations["actual_seq_lengths"]
    ssm_state_indices = param_combinations["ssm_state_indices"]
    has_gamma = param_combinations["has_gamma"]
    has_gamma_k = param_combinations["has_gamma_k"]
    has_num_accepted_tokens = param_combinations["has_num_accepted_tokens"]
    scale_value = param_combinations["scale_value"]
    num_accepted_tokens = param_combinations["num_accepted_tokens"]
    block_num = param_combinations["block_num"]
    data_type = param_combinations["data_type"]
    state_data_type = param_combinations["state_data_type"]
    query_datarange = param_combinations["query_datarange"]
    key_datarange = param_combinations["key_datarange"]
    value_datarange = param_combinations["value_datarange"]
    gamma_datarange = param_combinations["gamma_datarange"]
    gamma_k_datarange = param_combinations["gamma_k_datarange"]
    beta_datarange = param_combinations["beta_datarange"]
    state_datarange = param_combinations["state_datarange"]

    test_data = (
        batch_size,
        mtp,
        nk,
        nv,
        dk,
        dv,
        actual_seq_lengths,
        ssm_state_indices,
        has_gamma,
        has_gamma_k,
        has_num_accepted_tokens,
        scale_value,
        num_accepted_tokens,
        block_num,
        data_type,
        state_data_type,
        query_datarange,
        key_datarange,
        value_datarange,
        gamma_datarange,
        gamma_k_datarange,
        beta_datarange,
        state_datarange,
    )

    torch_npu.npu.set_device(0)
    recurrent_gated_delta_rule_operator_single.output_operator(test_data)
