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
import os
import json
import logging
import random

import torch
import torch_npu

# ******入参调用
from test_chunk_gated_delta_rule_paramset import ENABLED_PARAMS
from test_chunk_gated_delta_rule_paramset_rdv import ENABLED_PARAMS_RDV

# ******CPU侧算子逻辑实现获取golden与npu算子直调结果
import chunk_gated_delta_rule_operator_single
import pytest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}

CUSTOM_CASE = os.environ.get("CUSTOM_CASE", "")

_RANDOM_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128]
_RANDOM_SEQLENS = [
    1,
    3,
    7,
    32,
    64,
    100,
    128,
    200,
    256,
    300,
    512,
    1000,
    1024,
    2048,
    4096,
    5000,
    8192,
    10000,
    16384,
    32768,
    65535,
]
_RANDOM_VALUE_RANGES = [[-10, 10], [-1, 1]]
_RANDOM_GAMMA_RANGES = [[-1, 0], [-0.5, 0], [-0.1, 0], [-1, -0.5]]
_STATE_ELEM_CAP = 2_000_000_000
_QKV_ELEM_CAP = 1_000_000_000


def _generate_random_param_dict(rng):
    """在算子约束内从0随机生成一条用例参数。

    约束：B>0、seqlen>0、0<Nk<=64、0<Nv<=64 且 Nv>=Nk 且 Nv%Nk==0、0<Dk<=128、0<Dv<=128。
    chunk_size 固定 64（算子 tiling 硬编码）。
    Dk*Dv 按 state 元素上限分配，T*max(Nk*Dk,Nv*Dv) 按 QKV 张量上限分配，防 OOM。
    30% 概率生成变长序列（B>1 时每个 batch 的 seqlen 独立随机），覆盖 actual_seq_lengths 路径。
    datarange 随机选择，覆盖不同数据范围对算子精度的影响。
    """
    batch_size = rng.choice(_RANDOM_BATCH_SIZES)
    nk = rng.randint(1, 64)
    nv = nk * rng.randint(1, 64 // nk)
    dk_dv_budget = max(1, _STATE_ELEM_CAP // (batch_size * nv))
    dk = rng.randint(1, min(128, dk_dv_budget))
    dv = rng.randint(1, min(128, max(1, dk_dv_budget // dk)))
    qkv_per_token = max(nk * dk, nv * dv)
    max_t = max(1, _QKV_ELEM_CAP // qkv_per_token)
    max_seqlen = max(1, max_t // batch_size)
    valid_seqlens = [s for s in _RANDOM_SEQLENS if s <= max_seqlen]
    if not valid_seqlens:
        valid_seqlens = [1]
    if batch_size > 1 and rng.random() < 0.3:
        seqlen = [rng.choice(valid_seqlens) for _ in range(batch_size)]
    else:
        seqlen = rng.choice(valid_seqlens)
    return {
        "_name": "random",
        "B": batch_size,
        "seqlen": seqlen,
        "nk": nk,
        "nv": nv,
        "dk": dk,
        "dv": dv,
        "chunk_size": 64,
        "data_type": torch.bfloat16,
        "state_data_type": rng.choice([torch.bfloat16, torch.float32]),
        "has_g": rng.choice([True, False]),
        "is_contiguous": rng.choice([True, False]),
        "pt_path": "",
        "query_datarange": [-1, 1],
        "key_datarange": [-1, 1],
        "value_datarange": rng.choice(_RANDOM_VALUE_RANGES),
        "gamma_datarange": rng.choice(_RANDOM_GAMMA_RANGES),
        "beta_datarange": [0, 1],
        "state_datarange": [-10, 10],
    }


if CUSTOM_CASE:
    _case = json.loads(CUSTOM_CASE)
    _case["data_type"] = _DTYPE_MAP.get(
        _case.get("data_type", "bfloat16"), torch.bfloat16
    )
    _case["state_data_type"] = _DTYPE_MAP.get(
        _case.get("state_data_type", "bfloat16"), torch.bfloat16
    )
    _case.setdefault("_name", "custom")
    _case.setdefault("chunk_size", 64)
    _case.setdefault("has_g", True)
    _case.setdefault("is_contiguous", True)
    _case.setdefault("pt_path", "")
    param_combinations = [_case]
    logger.info(f"CUSTOM_CASE mode: {_case}")
else:
    TEST_MODE = os.environ.get("TEST_MODE", "single")

    if TEST_MODE not in ["single", "rdv", "random"]:
        raise ValueError(
            f"Invalid TEST_MODE: {TEST_MODE}, must be 'single', 'rdv' or 'random'"
        )

    logger.info(f"TEST_MODE: {TEST_MODE}")

    if TEST_MODE == "random":
        seed_env = os.environ.get("RANDOM_SEED")
        random_seed = int(seed_env) if seed_env else random.randrange(2**31)
        os.environ["RANDOM_SEED"] = str(random_seed)
        rng = random.Random(random_seed)
        random_count = int(os.environ.get("RANDOM_CASE_COUNT", "100"))
        logger.info(
            f"Random seed: {random_seed} (set RANDOM_SEED to reproduce), "
            f"count: {random_count}"
        )
        param_combinations = [
            _generate_random_param_dict(rng) for _ in range(random_count)
        ]
    else:
        if TEST_MODE == "rdv":
            PARAM_SET = ENABLED_PARAMS_RDV
        else:
            PARAM_SET = ENABLED_PARAMS

        param_combinations = []

        for _, params in enumerate(PARAM_SET):
            param_names = [
                "_name",
                "B",
                "seqlen",
                "nk",
                "nv",
                "dk",
                "dv",
                "chunk_size",
                "data_type",
                "state_data_type",
                "has_g",
                "is_contiguous",
                "pt_path",
                "query_datarange",
                "key_datarange",
                "value_datarange",
                "gamma_datarange",
                "beta_datarange",
                "state_datarange",
            ]

            param_values = [
                params["_name"] if "_name" in params else [""],
                params["B"],
                params["seqlen"],
                params["nk"],
                params["nv"],
                params["dk"],
                params["dv"],
                params["chunk_size"],
                params["data_type"],
                params["state_data_type"],
                params["has_g"],
                params.get("is_contiguous", [True]),
                params.get("pt_path", [""]),
                params.get("query_datarange", [[0, 1]]),
                params.get("key_datarange", [[0, 1]]),
                params.get("value_datarange", [[0, 1]]),
                params.get("gamma_datarange", [[-1, 0]]),
                params.get("beta_datarange", [[0, 1]]),
                params.get("state_datarange", [[0, 1]]),
            ]

            for combo in itertools.product(*param_values):
                param_dict = dict(zip(param_names, combo))
                param_combinations.append(param_dict)

logger.info(f"Total test cases: {len(param_combinations)}")


@pytest.mark.ci
@pytest.mark.parametrize("param_combinations", param_combinations)
def test_chunk_gated_delta_rule(param_combinations):
    # 初始化参数和tensor
    B = param_combinations["B"]
    seqlen = param_combinations["seqlen"]
    nk = param_combinations["nk"]
    nv = param_combinations["nv"]
    dk = param_combinations["dk"]
    dv = param_combinations["dv"]
    chunk_size = param_combinations["chunk_size"]
    data_type = param_combinations["data_type"]
    state_data_type = param_combinations["state_data_type"]
    has_g = param_combinations["has_g"]
    is_contiguous = param_combinations["is_contiguous"]
    pt_path = param_combinations.get("pt_path", "")

    datarange_kwargs = {}
    for dr_key in (
        "query_datarange",
        "key_datarange",
        "value_datarange",
        "gamma_datarange",
        "beta_datarange",
        "state_datarange",
    ):
        if dr_key in param_combinations:
            datarange_kwargs[dr_key] = param_combinations[dr_key]

    test_data = (
        B,
        seqlen,
        nk,
        nv,
        dk,
        dv,
        chunk_size,
        data_type,
        state_data_type,
        has_g,
        is_contiguous,
    )

    torch_npu.npu.set_device(0)

    # 获取cpu结果(真值)和算子结果（测试值)
    chunk_gated_delta_rule_operator_single.run_precision_test(
        test_data, pt_path=pt_path, **datarange_kwargs
    )
