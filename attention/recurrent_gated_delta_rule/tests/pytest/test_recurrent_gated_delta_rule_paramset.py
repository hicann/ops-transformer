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

# 定义测试参数组合
TEST_PARAMS = {
    "Testcase0": {
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
        "gamma_datarange": [[0, 1]],
        "gamma_k_datarange": [[0, 1]],
        "beta_datarange": [[0, 1]],
        "state_datarange": [[-10, 10]],
        "state_non_contiguous": [False],
    },
    "Testcase1": {
        "batch_size": [2],
        "mtp": [2],
        "nk": [8],
        "nv": [32],
        "dk": [7],
        "dv": [18],
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
        "gamma_datarange": [[0, 1]],
        "gamma_k_datarange": [[0, 1]],
        "beta_datarange": [[0, 1]],
        "state_datarange": [[-10, 10]],
        "state_non_contiguous": [False],
    },
}  # 注意单个用例组内的用例尽量不要超过32
FIRST_CASE = [TEST_PARAMS["Testcase0"], TEST_PARAMS["Testcase1"]]
# 按需选择要启用的测试参数（例如默认启用所有）
ENABLED_PARAMS = FIRST_CASE  # 按需增加需要的case即可
