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

from fia_fullquant_mla_paramset_common import expand_paramset_to_cases

TEST_PARAMS = {
    "noPA_B1_QS5_KVS81_Nq6_Nkv1_D512_SP3_LSE0": {
        "B": [1],
        "N_q": [6],
        "N_kv": [1],
        "actual_seq_q": [[5]],
        "actual_seq_kv": [[81]],
        "enable_pa": [False],
        "kv_cache_layout": ["TND"],
        "sparse_mode": [3],
        "enable_lse": [False],
    },
    "PA_BnNBsD_B1_QS9_KVS542_Nq6_Nkv1_D512_SP3_LSE1": {
        "B": [1],
        "N_q": [6],
        "N_kv": [1],
        "actual_seq_q": [[9]],
        "actual_seq_kv": [[542]],
        "enable_pa": [True],
        "kv_cache_layout": ["BnNBsD"],
        "sparse_mode": [3],
        "enable_lse": [True],
    },
    "PA_BnBsH_B1_QS11_KVS4915_Nq96_Nkv1_D512_SP3_LSE1": {
        "B": [1],
        "N_q": [96],
        "N_kv": [1],
        "actual_seq_q": [[11]],
        "actual_seq_kv": [[4915]],
        "enable_pa": [True],
        "kv_cache_layout": ["BnBsH"],
        "sparse_mode": [3],
        "enable_lse": [True],
    },
    "PA_NZ_B1_QS14_KVS7039_Nq24_Nkv1_D512_SP3_LSE0": {
        "B": [1],
        "N_q": [24],
        "N_kv": [1],
        "actual_seq_q": [[14]],
        "actual_seq_kv": [[7039]],
        "enable_pa": [True],
        "kv_cache_layout": ["NZ"],
        "sparse_mode": [3],
        "enable_lse": [False],
    },
}

CASES = expand_paramset_to_cases(TEST_PARAMS)
