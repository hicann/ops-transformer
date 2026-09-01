# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import itertools
import random
import torch
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

if TEST_MODE not in ["single", "rdv", "random"]:
    raise ValueError(
        f"Invalid TEST_MODE: {TEST_MODE}, must be 'single', 'rdv' or 'random'"
    )

if TEST_MODE == "rdv":
    PARAM_SET = ENABLED_PARAMS_RDV
elif TEST_MODE == "single":
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
    "state_non_contiguous",
]

# 随机模式：在算子约束内从0生成用例（不依赖 rdv 参数池）
_RANDOM_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128]
_RANDOM_MTPS = [1, 2, 3, 4, 8]
_RANDOM_GAMMA_RANGES = [[-1, 0], [-10, 0], [-5, 0], [-0.5, 0]]
_RANDOM_VALUE_RANGES = [[-10, 10], [-1, 1]]
# host 内存上限驱动的 state 元素数上限。
# 实测单 case host 峰值 ≈ baseline(1.6GB) + 10 * state_elems 字节
#   (bf16 state 最坏: 原2E + .to(fp32)4E + .clone()4E = 10E; fp32 state 仅8E)
# 取 2.0B -> host 峰值 ≈ 1.6 + 20 = 21.6GB < 24GB host 上限。
# NPU HBM(123GB) 远不构成瓶颈，host 才是绑定约束。
_STATE_ELEM_CAP = 2_000_000_000


def _generate_random_param_dict(rng):
    """在算子约束内从0随机生成一条用例参数。

    约束：0<Nk<=256、0<Nv<=256 且 Nv>=Nk 且 Nv%Nk==0、0<Dk<=512、0<Dv<=512、
    mtp<=8、BlockNum>=T(默认 B*mtp)；query/key∈[-1,1]、g/gk<0、0<beta<1。
    Dk*Dv 按 state 元素上限分配，规避大 shape 单进程 OOM。
    """
    batch_size = rng.choice(_RANDOM_BATCH_SIZES)
    mtp = rng.choice(_RANDOM_MTPS)
    nk = rng.randint(1, 256)
    nv = nk * rng.randint(1, 256 // nk)
    block_num = batch_size * mtp
    dk_dv_budget = max(1, _STATE_ELEM_CAP // (block_num * nv))
    dk = rng.randint(1, min(512, dk_dv_budget))
    dv = rng.randint(1, min(512, max(1, dk_dv_budget // dk)))
    return {
        "batch_size": batch_size,
        "mtp": mtp,
        "nk": nk,
        "nv": nv,
        "dk": dk,
        "dv": dv,
        "actual_seq_lengths": None,
        "ssm_state_indices": None,
        "has_gamma": rng.choice(["True", "False"]),
        "has_gamma_k": rng.choice(["True", "False"]),
        "has_num_accepted_tokens": rng.choice(["True", "False"]),
        "scale_value": None,
        "num_accepted_tokens": None,
        "block_num": None,
        "data_type": torch.bfloat16,
        "state_data_type": rng.choice([torch.bfloat16, torch.float32]),
        "query_datarange": [-1, 1],
        "key_datarange": [-1, 1],
        "value_datarange": rng.choice(_RANDOM_VALUE_RANGES),
        "gamma_datarange": rng.choice(_RANDOM_GAMMA_RANGES),
        "gamma_k_datarange": rng.choice(_RANDOM_GAMMA_RANGES),
        "beta_datarange": [0, 1],
        "state_datarange": [-10, 10],
        "state_non_contiguous": rng.choice([False, True]),
    }


param_combinations = []

if TEST_MODE == "random":
    seed_env = os.environ.get("RANDOM_SEED")
    random_seed = int(seed_env) if seed_env else random.randrange(2**31)
    # 回写实际使用的种子，供 conftest.py 落 CSV（复现时 RANDOM_SEED=<seed> 即可）
    os.environ["RANDOM_SEED"] = str(random_seed)
    rng = random.Random(random_seed)
    random_count = int(os.environ.get("RANDOM_CASE_COUNT", "100"))
    logger.info(
        f"Random seed: {random_seed} (set RANDOM_SEED to reproduce), count: {random_count}"
    )
    param_combinations = [_generate_random_param_dict(rng) for _ in range(random_count)]
else:
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
            params["state_non_contiguous"],
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
    state_non_contiguous = param_combinations["state_non_contiguous"]

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
        state_non_contiguous,
    )

    torch_npu.npu.set_device(0)
    recurrent_gated_delta_rule_operator_single.output_operator(test_data)
