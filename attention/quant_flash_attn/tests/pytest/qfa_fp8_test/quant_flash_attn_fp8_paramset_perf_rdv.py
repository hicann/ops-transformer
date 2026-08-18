#!/usr/bin/python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from quant_flash_attn_fp8_paramset_common import expand_paramset_to_cases

TEST_PARAMS = {
    "B1_G8_Nq16_Nkv2_D128_SM3_LSE0_Q16384_KVS16384_P256_Perf": {
        "B": [1],
        "N_q": [16],
        "N_kv": [2],
        "actual_seq_q": [[16384]],
        "actual_seq_kv": [[16384]],
        "enable_lse": [False],
        "mask_mode": [3],
        "p_scale": [256.0],
        "graph_path": [0],
    },
    "B1_G8_Nq16_Nkv2_D128_SM3_LSE0_Q12544_KVS28928_P256_Perf": {
        "B": [1],
        "N_q": [16],
        "N_kv": [2],
        "actual_seq_q": [[12544]],
        "actual_seq_kv": [[28928]],
        "enable_lse": [False],
        "mask_mode": [3],
        "p_scale": [256.0],
        "graph_path": [0],
    },
    "B1_G8_Nq16_Nkv2_D128_SM3_LSE0_Q10496_KVS39424_P256_Perf": {
        "B": [1],
        "N_q": [16],
        "N_kv": [2],
        "actual_seq_q": [[10496]],
        "actual_seq_kv": [[39424]],
        "enable_lse": [False],
        "mask_mode": [3],
        "p_scale": [256.0],
        "graph_path": [0],
    },
    "B1_G8_Nq16_Nkv2_D128_SM3_LSE0_Q9216_KVS48640_P256_Perf": {
        "B": [1],
        "N_q": [16],
        "N_kv": [2],
        "actual_seq_q": [[9216]],
        "actual_seq_kv": [[48640]],
        "enable_lse": [False],
        "mask_mode": [3],
        "p_scale": [256.0],
        "graph_path": [0],
    },
    "B1_G8_Nq16_Nkv2_D128_SM3_LSE0_Q8320_KVS56960_P256_Perf": {
        "B": [1],
        "N_q": [16],
        "N_kv": [2],
        "actual_seq_q": [[8320]],
        "actual_seq_kv": [[56960]],
        "enable_lse": [False],
        "mask_mode": [3],
        "p_scale": [256.0],
        "graph_path": [0],
    },
    "B1_G8_Nq16_Nkv2_D128_SM3_LSE0_Q4040_KVS61000_P256_Perf": {
        "B": [1],
        "N_q": [16],
        "N_kv": [2],
        "actual_seq_q": [[4040]],
        "actual_seq_kv": [[61000]],
        "enable_lse": [False],
        "mask_mode": [3],
        "p_scale": [256.0],
        "graph_path": [0],
    },
}

CASES = expand_paramset_to_cases(TEST_PARAMS)
SKIP_CASES = set()
