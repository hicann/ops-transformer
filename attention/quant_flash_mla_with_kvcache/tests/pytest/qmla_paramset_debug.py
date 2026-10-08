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

from qmla_paramset_common import expand_paramset_to_cases

TEST_PARAMS = {
    "PA_BnNBsD_B1_QS8_KVS131072_Nq96_Nkv1_D512_SP0_LSE0": {
        "B": [1],
        "N_q": [96],
        "N_kv": [1],
        "seqused_q": [[8]],
        "cache_seqlens": [[131072]],
        "enable_pa": [True],
        "kv_cache_layout": ["BnNBsD"],
        "mask_mode": [0],
        "enable_lse": [False],
    },
    "PA_NZ_B1_QS8_KVS131072_Nq96_Nkv1_D512_SP0_LSE0": {
        "B": [1],
        "N_q": [96],
        "N_kv": [1],
        "seqused_q": [[8]],
        "cache_seqlens": [[131072]],
        "enable_pa": [True],
        "kv_cache_layout": ["NZ"],
        "mask_mode": [0],
        "enable_lse": [False],
    },
    "PA_BnNBsD_B1_QS8_KVS542_Nq6_Nkv1_D512_SP3_LSE1": {
        "B": [1],
        "N_q": [6],
        "N_kv": [1],
        "seqused_q": [[8]],
        "cache_seqlens": [[542]],
        "enable_pa": [True],
        "kv_cache_layout": ["BnNBsD"],
        "mask_mode": [3],
        "enable_lse": [True],
    },
}

CASES = expand_paramset_to_cases(TEST_PARAMS)

FAIL_CASES = set()
