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

from fia_fullquant_gqa_paramset_common import expand_paramset_to_cases

TEST_PARAMS = {
    "B2_G1_Nq1_Nkv1_D128_SM3_LSE1_Q128_KV256": {
        "B": [2],
        "N_q": [1],
        "N_kv": [1],
        "actual_seq_q": [[128, 128]],
        "actual_seq_kv": [[256, 0]],
        "enable_lse": [True],
        "sparse_mode": [3],
        "p_scale": [1.0],
    },
    "B2_G8_Nq8_Nkv1_D128_SM3_LSE1_Q128_KV256": {
        "B": [2],
        "N_q": [8],
        "N_kv": [1],
        "actual_seq_q": [[128, 128]],
        "actual_seq_kv": [[256, 1]],
        "enable_lse": [True],
        "sparse_mode": [3],
        "p_scale": [1.0],
    },
}

CASES = expand_paramset_to_cases(TEST_PARAMS)
