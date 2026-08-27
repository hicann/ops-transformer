#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import itertools
import torch
import torch_npu
from testcases import ENABLED_PARAMS
import check_valid_param
import fa_no_quant_bnsd_bsnd
import pytest
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
logger = logging.getLogger(__name__)

DEBUG_ON = 0

all_test_cases = []
param_names = [
    "batch_size",
    "q_head_num",
    "kv_head_num",
    "q_seq",
    "kv_seq",
    "head_dim",
    "dtype",
    "in_layout",
    "rope",
]

for case_idx, params in enumerate(ENABLED_PARAMS):
    # 提取当前所有参数列表
    param_values = [params[name] for name in param_names]
    # 生成当前case的所有参数组合
    for combo in itertools.product(*param_values):
        param_dict = dict(zip(param_names, combo))
        param_dict["case_idx"] = case_idx
        param_dict["total_cases"] = len(ENABLED_PARAMS)
        all_test_cases.append(param_dict)


@pytest.mark.ci
@pytest.mark.parametrize("test_case", all_test_cases)
def test_flash_attention(test_case):  # 初始化参数和tensor
    # 提取参数
    case_idx = test_case.pop("case_idx")
    total_cases = test_case.pop("total_cases")
    batch_size = test_case["batch_size"]
    q_head_num = test_case["q_head_num"]
    kv_head_num = test_case["kv_head_num"]
    q_seq = test_case["q_seq"]
    kv_seq = test_case["kv_seq"]
    head_dim = test_case["head_dim"]
    dtype = test_case["dtype"]
    in_layout = test_case["in_layout"]
    rope = test_case["rope"]

    act_seq_len = [q_seq] * batch_size
    act_seq_len_kv = torch.randint(
        low=1, high=kv_seq + 1, size=(batch_size,), dtype=torch.int32
    )

    torch_npu.npu.set_device(0)

    scaled_value = 1 / (head_dim**0.5)

    try:
        test_data = (
            batch_size,
            q_head_num,
            kv_head_num,
            q_seq,
            kv_seq,
            head_dim,
            dtype,
            in_layout,
            rope,
        )
        check_valid_param.validate_config(test_data)
    except ValueError as e:
        pytest.skip(f"参数校验失败: {e}")
    if in_layout == "BNSD" or in_layout == "BSND":
        test_data = (
            batch_size,
            q_head_num,
            kv_head_num,
            q_seq,
            kv_seq,
            head_dim,
            dtype,
            in_layout,
            act_seq_len,
            act_seq_len_kv,
            scaled_value,
            rope,
        )
        expect, result = fa_no_quant_bnsd_bsnd.test_fa_no_quant(test_data)
    check_valid_param.check_result(expect, result)
    logger.info(f"current case: {case_idx}, {case_idx + 1}/{total_cases} passed!")
